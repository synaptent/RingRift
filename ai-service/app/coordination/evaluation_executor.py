"""Evaluation execution helpers for :mod:`evaluation_daemon`.

April 2026: Extracted from ``evaluation_daemon.py`` so the daemon shell keeps
subscription, scheduling, and status responsibilities while this mixin owns the
heavy execution path: queue processing, gauntlet dispatch/fallback, result
emission, retry/backpressure handling, and head-to-head comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from app.config.coordination_defaults import DistributionDefaults
from app.coordination.event_router import DataEventType, safe_emit_event
from app.coordination.event_utils import make_config_key
from app.coordination.unified_distribution_daemon import (
    verify_model_distribution,
    wait_for_model_availability,
)
from app.models import BoardType
from app.training.architecture_tracker import extract_architecture_from_model_path
from app.training.game_gauntlet import (
    BaselineOpponent,
    _create_gauntlet_recording_config,
    run_baseline_gauntlet,
)
from app.training.tournament import Tournament
from app.utils.game_discovery import get_game_counts_summary

try:
    from app.coordination.hashgraph import (
        EvaluationConsensusConfig,
        get_evaluation_consensus_manager,
    )
    HAS_HASHGRAPH_CONSENSUS = True
except ImportError:
    HAS_HASHGRAPH_CONSENSUS = False
    get_evaluation_consensus_manager = None
    EvaluationConsensusConfig = None

logger = logging.getLogger(__name__)


class EvaluationExecutorMixin:
    """Execution-heavy helpers for ``EvaluationDaemon``."""

    async def _evaluation_worker(self) -> None:
        """Worker that processes evaluation requests from the queue."""
        logger.info("[EvaluationDaemon] Evaluation worker started")
        while self._running:
            try:
                # December 29, 2025: Process retry queue first
                await self._process_retry_queue()

                # January 7, 2026: Try to get from in-memory queue first
                request = None
                try:
                    request = await asyncio.wait_for(
                        self._evaluation_queue.get(),
                        timeout=1.0,  # Short timeout to check persistent queue
                    )
                except asyncio.TimeoutError:
                    pass  # No in-memory request, check persistent queue

                # January 7, 2026 (Session 17.50): Check persistent queue if no in-memory request
                # This fixes the bug where startup_scan items were never processed
                if request is None and self._persistent_queue:
                    logger.debug("[EvaluationDaemon] Checking persistent queue...")
                    persistent_request = self._persistent_queue.claim_next()
                    if persistent_request:
                        # Feb 2026: Batch-claim all other harness entries for the
                        # same model to prevent 4x redundant multi-harness evaluations.
                        # The multi-harness gauntlet runs ALL harnesses in one go,
                        # so we need to claim all queue entries for this model.
                        sibling_ids = self._persistent_queue.claim_siblings(
                            persistent_request.model_path,
                            persistent_request.request_id,
                        )

                        # Convert to in-memory request format
                        request = {
                            "model_path": persistent_request.model_path,
                            "board_type": persistent_request.board_type,
                            "num_players": persistent_request.num_players,
                            "config_key": persistent_request.config_key,
                            "timestamp": persistent_request.started_at,
                            "source": persistent_request.source,
                            "priority": persistent_request.priority,
                            "_persistent_request_id": persistent_request.request_id,
                            "_sibling_request_ids": sibling_ids,
                        }
                        logger.info(
                            f"[EvaluationDaemon] Claimed from persistent queue: "
                            f"{persistent_request.model_path} ({persistent_request.config_key})"
                            f"{f' + {len(sibling_ids)} siblings' if sibling_ids else ''}"
                        )

                if request is None:
                    # No request from either queue, wait a bit
                    await asyncio.sleep(2.0)
                    continue

                # Skip if already evaluating this model
                model_path = request["model_path"]
                if model_path in self._active_evaluations:
                    logger.debug(f"[EvaluationDaemon] Skipping duplicate: {model_path}")
                    continue

                # Check concurrency limit
                if len(self._active_evaluations) >= self.config.max_concurrent_evaluations:
                    # Re-queue and wait
                    await self._evaluation_queue.put(request)
                    await asyncio.sleep(1.0)
                    continue

                # Mar 6, 2026: System load guard — skip evaluation if system is
                # already overloaded. Prevents cascading process explosion that
                # caused kernel watchdog panics on mac-studio.
                try:
                    load_1m = os.getloadavg()[0]
                    cpu_count = os.cpu_count() or 1
                    if load_1m > cpu_count * 2:
                        logger.warning(
                            f"[EvaluationDaemon] System overloaded (load={load_1m:.0f}, "
                            f"cpus={cpu_count}), deferring evaluation"
                        )
                        await self._evaluation_queue.put(request)
                        await asyncio.sleep(30.0)
                        continue
                except OSError:
                    pass

                # Sprint 15 (Jan 3, 2026): Download OWC models before evaluation
                source = request.get("source", "")
                local_path = model_path

                if source == "backlog_owc" and not Path(model_path).exists():
                    # Model is on OWC, need to download it first
                    download_path = await self._download_owc_model(model_path)
                    if download_path:
                        local_path = str(download_path)
                        request["local_path"] = local_path
                        logger.info(f"[EvaluationDaemon] Downloaded OWC model to: {local_path}")
                    else:
                        logger.error(f"[EvaluationDaemon] Failed to download OWC model: {model_path}")
                        safe_emit_event(
                            DataEventType.OWC_MODEL_EVALUATION_FAILED,
                            {
                                "model_path": model_path,
                                "reason": "download_failed",
                                "source": source,
                            },
                        )
                        self._eval_stats.evaluations_failed += 1
                        continue

                # January 3, 2026: Check model exists before evaluation
                # Feb 23, 2026: For candidate models being fetched from training
                # nodes, wait up to 90s for the rsync to complete
                if not Path(local_path).exists():
                    is_candidate = "candidate_" in local_path
                    if is_candidate:
                        logger.info(
                            f"[EvaluationDaemon] Candidate model not yet local: {local_path}, "
                            f"waiting for transfer..."
                        )
                        for _wait in range(18):  # 18 × 5s = 90s
                            await asyncio.sleep(5)
                            if Path(local_path).exists():
                                break
                    if not Path(local_path).exists():
                        logger.warning(
                            f"[EvaluationDaemon] Model not found: {local_path}"
                        )
                        safe_emit_event(
                            DataEventType.EVALUATION_FAILED,
                            {
                                "model_path": model_path,
                                "reason": "model_not_found",
                                "config_key": request.get("config_key", "unknown"),
                            },
                        )
                        self._eval_stats.evaluations_failed += 1
                        continue

                # Run evaluation
                self._active_evaluations.add(model_path)
                try:
                    await self._run_evaluation(request)
                finally:
                    self._active_evaluations.discard(model_path)
                    # December 29, 2025 (Phase 4): Check for backpressure release
                    # Session 17.24: Require stable time below threshold before release
                    queue_depth = self._evaluation_queue.qsize()
                    if self._backpressure_active and queue_depth <= self.config.backpressure_release_threshold:
                        if self._should_release_backpressure():
                            self._emit_backpressure(queue_depth, activate=False)
                    elif queue_depth > self.config.backpressure_release_threshold:
                        # Queue is above release threshold - reset stability tracking
                        self._below_threshold_since = 0.0

            except asyncio.TimeoutError:
                continue  # Normal - check running status
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"[EvaluationDaemon] Worker error: {e}")
                await asyncio.sleep(1.0)
    async def _check_model_availability(
        self,
        model_path: str,
    ) -> tuple[bool, int]:
        """Check if model is available on sufficient nodes for fair evaluation.

        December 2025 - Phase 3B: Pre-evaluation distribution check.
        Ensures models are properly distributed before evaluation to prevent
        unfair Elo ratings from models only available on 1-2 nodes.

        Args:
            model_path: Path to the model file

        Returns:
            Tuple of (available, node_count)
        """
        try:
            min_nodes = DistributionDefaults.MIN_NODES_FOR_EVALUATION
            timeout = 120.0  # 2 minutes for pre-eval check

            # February 2026: Count local node if model exists locally
            import os
            local_count = 1 if os.path.exists(model_path) else 0
            if local_count >= min_nodes:
                logger.debug(
                    f"[EvaluationDaemon] Model {model_path} available locally "
                    f"(min_nodes={min_nodes}), skipping distribution check"
                )
                return (True, local_count)

            # First quick check
            success, count = await verify_model_distribution(model_path, min_nodes)
            count = max(count, local_count)  # Include local node
            if success or count >= min_nodes:
                logger.debug(
                    f"[EvaluationDaemon] Model {model_path} available on {count} nodes"
                )
                return (True, count)

            # If not enough, trigger priority distribution and wait
            logger.info(
                f"[EvaluationDaemon] Model {model_path} only on {count}/{min_nodes} nodes, "
                f"waiting for distribution (timeout: {timeout}s)"
            )

            success, count = await wait_for_model_availability(
                model_path, min_nodes=min_nodes, timeout=timeout
            )

            if not success:
                logger.warning(
                    f"[EvaluationDaemon] Model {model_path} distribution incomplete: "
                    f"{count}/{min_nodes} nodes"
                )
                # Emit MODEL_EVALUATION_BLOCKED event
                safe_emit_event(
                    event_type="MODEL_EVALUATION_BLOCKED",
                    payload={
                        "model_path": model_path,
                        "required_nodes": min_nodes,
                        "actual_nodes": count,
                        "reason": "insufficient_distribution",
                    },
                    source="evaluation_daemon",
                )

            return (success, count)

        except ImportError as e:
            logger.debug(f"[EvaluationDaemon] Distribution check unavailable: {e}")
            return (True, 0)  # Allow evaluation if check unavailable
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"[EvaluationDaemon] Distribution check error: {e}")
            return (True, 0)  # Allow evaluation on error
    async def _ensure_model_local(
        self, model_path: str, board_type: str, num_players: int
    ) -> str | None:
        """Ensure model is available locally, syncing from remote if needed.

        January 9, 2026 (Sprint 17.9): Support for remote model sync.
        When ComprehensiveModelScanDaemon discovers models on cluster nodes,
        we need to sync them to local before evaluation.

        Args:
            model_path: Model path, may be local or remote (cluster:node_id prefix)
            board_type: Board type for model lookup
            num_players: Number of players for model lookup

        Returns:
            Local path to model, or None if sync failed
        """
        # Check if this is a remote model reference (source: cluster:node_id)
        # Remote paths are stored with the full remote path in the queue
        if not model_path.startswith("/") and ":" not in model_path:
            # Relative local path - verify it actually exists
            if Path(model_path).exists():
                return model_path
            # Feb 23, 2026: Candidate models are trained on GPU nodes and synced
            # back via rsync (triggered by handle_work_complete). Wait for the
            # transfer to complete before giving up.
            logger.info(
                f"[EvaluationDaemon] Model not found locally: {model_path}, "
                f"waiting for transfer from training node..."
            )
            for wait_round in range(12):  # Wait up to 60s (12 × 5s)
                await asyncio.sleep(5)
                if Path(model_path).exists():
                    logger.info(
                        f"[EvaluationDaemon] Model arrived after {(wait_round + 1) * 5}s: "
                        f"{model_path}"
                    )
                    return model_path
            logger.warning(
                f"[EvaluationDaemon] Model not found after 60s wait: {model_path}, "
                f"trying remote sync..."
            )
            # Fall through to remote sync code below
        if Path(model_path).exists():
            # Already local (absolute path)
            return model_path

        # Try to sync from cluster
        try:
            from app.models.cluster_discovery import (
                get_cluster_model_discovery,
            )

            discovery = get_cluster_model_discovery()

            # Find the model on the cluster
            remote_models = await asyncio.to_thread(
                discovery.discover_cluster_models,
                board_type=board_type,
                num_players=num_players,
                include_local=False,
                include_remote=True,
                max_remote_nodes=10,
                timeout=60.0,
            )

            # Find matching model by path suffix
            model_name = Path(model_path).name
            for rm in remote_models:
                if Path(rm.remote_path).name == model_name:
                    logger.info(
                        f"[EvaluationDaemon] Syncing remote model from {rm.node_id}: {model_name}"
                    )
                    local_path = await asyncio.to_thread(
                        discovery.sync_model_to_local,
                        remote_model=rm,
                        local_dir=Path("models/synced"),
                        timeout=180.0,
                    )
                    if local_path and local_path.exists():
                        logger.info(f"[EvaluationDaemon] Model synced to: {local_path}")
                        return str(local_path)

            logger.warning(
                f"[EvaluationDaemon] Could not find remote model {model_name} on cluster"
            )
            return None

        except ImportError:
            logger.debug("[EvaluationDaemon] ClusterModelDiscovery not available")
            return None
        except (OSError, RuntimeError, TimeoutError) as e:
            logger.warning(f"[EvaluationDaemon] Remote model sync failed: {e}")
            return None
    async def _run_evaluation(self, request: dict) -> None:
        """Run gauntlet evaluation for a model."""
        model_path = request["model_path"]
        board_type = request["board_type"]
        num_players = request["num_players"]

        # January 9, 2026 (Sprint 17.9): Ensure model is available locally
        # This handles remote models discovered by ComprehensiveModelScanDaemon
        local_model_path = await self._ensure_model_local(model_path, board_type, num_players)
        if local_model_path is None:
            logger.warning(
                f"[EvaluationDaemon] Model not available locally and sync failed: {model_path}"
            )
            await self._emit_evaluation_failed(
                model_path, board_type, num_players,
                "model_sync_failed"
            )
            self._eval_stats.evaluations_failed += 1
            return
        model_path = local_model_path

        # Mar 18, 2026: ALWAYS try cluster dispatch first (GPU nodes with MCTS
        # search and strong baselines up to ~1900 Elo), regardless of
        # gauntlet_enabled. Previously, gauntlet_enabled=true on coordinator
        # SKIPPED cluster dispatch and ran locally with weak baselines (max
        # ~1400 Elo), directly causing the 1982 Elo ceiling for all configs.
        # Now: cluster first → local fallback only if cluster unavailable.
        from app.config.env import env
        dispatch_ok = await self._dispatch_gauntlet_to_cluster_with_fallback(
            model_path, board_type, num_players, request
        )
        if dispatch_ok:
            logger.info(
                f"[EvaluationDaemon] Dispatched gauntlet to cluster (MCTS): {model_path}"
            )
            return
        # Cluster dispatch failed.
        # Mar 4, 2026: For large boards (hexagonal, square19), don't fall
        # back to local — they're too slow on CPU. Re-queue for later
        # cluster dispatch when the gauntlet queue has capacity.
        if board_type in ("hexagonal", "square19"):
            logger.info(
                f"[EvaluationDaemon] Cluster dispatch full, deferring large board: {model_path}"
            )
            return  # Will be re-queued on next startup scan

        # Small boards: fall back to local if gauntlet_enabled
        import os
        gauntlet_override = os.environ.get("RINGRIFT_GAUNTLET_ENABLED", "").lower()
        if gauntlet_override in ("1", "true", "yes") or env.gauntlet_enabled:
            logger.info(
                f"[EvaluationDaemon] Cluster dispatch failed, running local gauntlet: {model_path}"
            )
        else:
            # No local fallback available
            logger.info(
                f"[EvaluationDaemon] Cluster dispatch failed, running lightweight local gauntlet (policy-only): {model_path}"
            )
            await self._run_lightweight_local_gauntlet(
                model_path, board_type, num_players, request
            )
            return

        # December 2025 - Phase 3B: Pre-evaluation distribution check
        available, node_count = await self._check_model_availability(model_path)
        if not available:
            logger.warning(
                f"[EvaluationDaemon] Skipping evaluation: {model_path} "
                f"not available on sufficient nodes ({node_count} nodes)"
            )
            # December 29, 2025: Queue for retry - distribution may complete later
            retry_attempt = request.get("_retry_attempt", 0)
            if self._queue_for_retry(
                model_path, board_type, num_players,
                f"distribution_incomplete:{node_count}", retry_attempt
            ):
                return  # Will retry after distribution completes
            self._eval_stats.evaluations_failed += 1
            await self._emit_evaluation_failed(
                model_path, board_type, num_players,
                f"Distribution incomplete: only {node_count} nodes"
            )
            return

        start_time = time.time()
        config_key = make_config_key(board_type, num_players)
        run_id = str(uuid.uuid4())

        # Mar 6, 2026: Cross-process resource governor.
        # Prevents evaluation + export from running simultaneously and
        # exhausting RAM (kernel panic root cause on mac-studio).
        _governor_slot = None
        try:
            from app.utils.coordinator_governor import get_governor, OperationType
            _governor_slot = get_governor().try_acquire(
                OperationType.EVALUATION,
                description=f"gauntlet:{config_key}:{Path(model_path).name}",
            )
            if _governor_slot is None:
                logger.info(
                    f"[EvaluationDaemon] Governor denied evaluation for {model_path}: "
                    "system at capacity, re-queuing"
                )
                await self._evaluation_queue.put(request)
                return
        except Exception as _gov_err:
            logger.debug(f"[EvaluationDaemon] Governor unavailable: {_gov_err}")

        logger.info(f"[EvaluationDaemon] Starting evaluation: {model_path}")

        # December 30, 2025: Record gauntlet run start for observability
        self._record_gauntlet_start(run_id, config_key)

        # December 30, 2025: Emit EVALUATION_STARTED (Gap #3 integration fix)
        await self._emit_evaluation_started(model_path, board_type, num_players)

        try:
            # Run the gauntlet
            result = await self._run_gauntlet(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )

            elapsed = time.time() - start_time
            self._eval_stats.evaluations_completed += 1
            self._eval_stats.last_evaluation_time = elapsed
            self._update_average_time(elapsed)

            # December 29, 2025: Track successful retries
            retry_attempt = request.get("_retry_attempt", 0)
            if retry_attempt > 0:
                self._retry_stats["retries_succeeded"] += 1
                logger.info(
                    f"[EvaluationDaemon] Retry #{retry_attempt} succeeded for {model_path}"
                )

            # January 2026: Track OOM recovery success and gradually restore parallel_games
            config_key = make_config_key(board_type, num_players)
            if config_key in self._oom_parallel_games:
                current_parallel = self._oom_parallel_games[config_key]
                self._oom_recovery_stats["oom_recoveries"] += 1
                logger.info(
                    f"[EvaluationDaemon] OOM recovery succeeded for {config_key} "
                    f"with parallel_games={current_parallel}"
                )
                # Gradually restore parallel_games: if at 8 -> try 12, at 4 -> try 6, etc.
                # This allows the system to recover to full speed over multiple evaluations
                if current_parallel < 16:
                    restored = min(16, current_parallel + current_parallel // 2)
                    self._oom_parallel_games[config_key] = restored
                    logger.debug(
                        f"[EvaluationDaemon] Restoring parallel_games to {restored} "
                        f"for {config_key}"
                    )
                else:
                    # Back at default, remove from tracking
                    del self._oom_parallel_games[config_key]

            # Count games played
            total_games = sum(
                opp.get("games_played", 0)
                for opp in result.get("opponent_results", {}).values()
            )
            self._eval_stats.games_played += total_games

            # Feb 23, 2026: Compute Elo from gauntlet results via EloService.
            # The gauntlet produces per-opponent win/loss records but doesn't
            # record them as Elo matches. Wire results into the Elo system so
            # models get proper ratings that feed into promotion decisions.
            estimated_elo = await self._compute_elo_from_gauntlet(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result,
            )
            if estimated_elo is not None:
                result["estimated_elo"] = estimated_elo
                result["best_elo"] = estimated_elo

            # Emit evaluation completed event
            await self._emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result,
            )

            # January 6, 2026: Head-to-head evaluation against previous model
            # Run asynchronously to not block the main evaluation loop
            self._safe_create_task(
                self._evaluate_vs_previous(model_path, board_type, num_players),
                context="head_to_head_evaluation",
            )

            # December 2025: Mark as recently evaluated for deduplication
            self._recently_evaluated[model_path] = time.time()

            # December 30, 2025: Record gauntlet completion for observability
            self._record_gauntlet_complete(run_id, 1, total_games, "completed")

            logger.info(
                f"[EvaluationDaemon] Evaluation completed: {model_path} "
                f"(win_rate={result.get('overall_win_rate', 0):.1%}, "
                f"{total_games} games, {elapsed:.1f}s)"
            )

            # January 7, 2026 (Session 17.50): Update persistent queue if this came from it
            # Feb 2026: Also complete sibling harness entries to prevent 4x redundancy
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                estimated_elo = result.get("estimated_elo", result.get("best_elo", 0.0))
                self._persistent_queue.complete(persistent_request_id, elo=estimated_elo)
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    self._persistent_queue.complete_batch(sibling_ids, elo=estimated_elo)
                logger.debug(
                    f"[EvaluationDaemon] Marked persistent request complete: {persistent_request_id}"
                    f"{f' + {len(sibling_ids)} siblings' if sibling_ids else ''}"
                )

        except asyncio.TimeoutError:
            self._eval_stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation timed out: {model_path}")
            # December 29, 2025: Queue for retry on timeout (transient failure)
            retry_attempt = request.get("_retry_attempt", 0)
            if self._queue_for_retry(
                model_path, board_type, num_players, "timeout", retry_attempt
            ):
                self._record_gauntlet_complete(run_id, 0, 0, "retry_queued")
                return  # Will retry, don't emit permanent failure
            # Emit EVALUATION_FAILED event (Dec 2025 - critical gap fix)
            self._record_gauntlet_complete(run_id, 0, 0, "failed:timeout")
            await self._emit_evaluation_failed(model_path, board_type, num_players, "timeout")
            # January 7, 2026: Mark persistent queue item as failed
            # Feb 2026: Also fail sibling harness entries
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, "timeout")
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    self._persistent_queue.fail_batch(sibling_ids, "timeout")
        except (MemoryError, RuntimeError) as e:
            # December 29, 2025: GPU OOM and RuntimeError (CUDA) are retryable
            # January 2026: With adaptive batch size reduction to prevent infinite loops
            self._eval_stats.evaluations_failed += 1
            error_str = str(e).lower()
            is_gpu_error = "cuda" in error_str or "out of memory" in error_str
            logger.error(f"[EvaluationDaemon] Evaluation failed ({type(e).__name__}): {model_path}: {e}")
            if is_gpu_error:
                # January 2026: Reduce parallel_games on OOM to prevent infinite retry loop
                config_key = make_config_key(board_type, num_players)
                current_parallel = self._oom_parallel_games.get(config_key, 16)
                if current_parallel > 1:
                    # Reduce by half: 16 -> 8 -> 4 -> 2 -> 1
                    reduced_parallel = max(1, current_parallel // 2)
                    self._oom_parallel_games[config_key] = reduced_parallel
                    self._oom_recovery_stats["oom_reductions"] += 1
                    logger.warning(
                        f"[EvaluationDaemon] OOM recovery: reducing parallel_games "
                        f"from {current_parallel} to {reduced_parallel} for {config_key}"
                    )

                    retry_attempt = request.get("_retry_attempt", 0)
                    if self._queue_for_retry(
                        model_path, board_type, num_players,
                        f"GPU OOM: reduced parallel_games to {reduced_parallel}",
                        retry_attempt
                    ):
                        self._record_gauntlet_complete(run_id, 0, 0, "retry_queued_oom")
                        return  # Will retry with reduced batch (uses _oom_parallel_games lookup)
                else:
                    # Already at minimum batch size, cannot reduce further
                    self._oom_recovery_stats["oom_exhausted"] += 1
                    logger.error(
                        f"[EvaluationDaemon] OOM with parallel_games=1, cannot reduce further: {model_path}"
                    )
            # Emit permanent failure
            self._record_gauntlet_complete(run_id, 0, 0, f"failed:{type(e).__name__}")
            await self._emit_evaluation_failed(model_path, board_type, num_players, str(e))
            # January 7, 2026: Mark persistent queue item as failed
            # Feb 2026: Also fail sibling harness entries
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, str(e))
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    self._persistent_queue.fail_batch(sibling_ids, str(e))
        except Exception as e:  # noqa: BLE001
            self._eval_stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation failed: {model_path}: {e}")
            # Emit EVALUATION_FAILED event (Dec 2025 - critical gap fix)
            self._record_gauntlet_complete(run_id, 0, 0, f"failed:{type(e).__name__}")
            await self._emit_evaluation_failed(model_path, board_type, num_players, str(e))
            # January 7, 2026: Mark persistent queue item as failed
            # Feb 2026: Also fail sibling harness entries
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, str(e))
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    self._persistent_queue.fail_batch(sibling_ids, str(e))
        finally:
            # Mar 6, 2026: Release governor slot
            if _governor_slot is not None:
                try:
                    from app.utils.coordinator_governor import get_governor
                    get_governor().release(_governor_slot)
                except Exception:
                    pass  # Best-effort cleanup: governor has TTL-based auto-expiry
    async def _run_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run baseline gauntlet with optional early stopping.

        December 30, 2025: Added multi-harness evaluation support.
        When config.enable_multi_harness is True, uses MultiHarnessGauntlet
        to evaluate under multiple algorithms (GUMBEL_MCTS, MINIMAX, etc.)
        and produces composite participant IDs for per-(model, harness) Elo tracking.
        """
        # December 30, 2025: Use multi-harness evaluation if enabled
        if self.config.enable_multi_harness:
            return await self._run_multi_harness_gauntlet(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )

        # Fallback to baseline-only gauntlet
        return await self._run_baseline_only_gauntlet(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
        )
    async def _run_baseline_only_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run baseline-only gauntlet (original behavior)."""
        # Map baseline names to enum values
        # Jan 13, 2026: Complete mapping for all baselines including NNUE harness diversity
        baseline_map = {
            "random": BaselineOpponent.RANDOM,
            "heuristic": BaselineOpponent.HEURISTIC,
            "heuristic_strong": BaselineOpponent.HEURISTIC_STRONG,
            "weak_heuristic": BaselineOpponent.WEAK_HEURISTIC,
            "mcts_light": BaselineOpponent.MCTS_LIGHT,
            "mcts_medium": BaselineOpponent.MCTS_MEDIUM,
            "mcts_strong": BaselineOpponent.MCTS_STRONG,
            "mcts_master": BaselineOpponent.MCTS_MASTER,
            "mcts_grandmaster": BaselineOpponent.MCTS_GRANDMASTER,
            "gumbel_b64": BaselineOpponent.GUMBEL_B64,
            "gumbel_b200": BaselineOpponent.GUMBEL_B200,
            "gumbel_nnue": BaselineOpponent.GUMBEL_NNUE,
            "policy_only_nn": BaselineOpponent.POLICY_ONLY_NN,
            "policy_only_nnue": BaselineOpponent.POLICY_ONLY_NNUE,
            "descent_nn": BaselineOpponent.DESCENT_NN,
            "descent_nnue": BaselineOpponent.DESCENT_NNUE,
            # NNUE baselines for harness diversity (Jan 13, 2026)
            "nnue_minimax_d4": BaselineOpponent.NNUE_MINIMAX_D4,
            "nnue_maxn_d3": BaselineOpponent.NNUE_MAXN_D3,
            "nnue_brs_d3": BaselineOpponent.NNUE_BRS_D3,
        }
        opponents = [
            baseline_map[b]
            for b in self.config.baselines
            if b in baseline_map
        ]

        # Dec 30, 2025: Get game count for graduated thresholds
        config_key = make_config_key(board_type, num_players)
        try:
            game_counts = get_game_counts_summary()
            game_count = game_counts.get(config_key, 0)
        except (OSError, RuntimeError) as e:
            logger.debug(f"[EvaluationDaemon] Failed to get game counts: {e}")
            game_count = None  # Will use fallback thresholds

        # Jan 10, 2026: Get model's current Elo for bootstrap fast evaluation
        # Weak models (< 1300 Elo) use fewer games per baseline for faster iteration
        model_elo = None
        try:
            from app.coordination.elo_service import get_elo_service
            elo_service = get_elo_service()
            model_elo = elo_service.get_config_elo(config_key)
            if model_elo:
                logger.debug(f"[EvaluationDaemon] Model Elo for {config_key}: {model_elo}")
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug(f"[EvaluationDaemon] Failed to get model Elo: {e}")

        # Use bootstrap games for weak models
        games_per_baseline = self.config.get_games_per_baseline(model_elo)
        if model_elo and model_elo < self.config.bootstrap_elo_threshold:
            logger.info(
                f"[EvaluationDaemon] Using bootstrap fast eval ({games_per_baseline} games) "
                f"for {config_key} (Elo: {model_elo:.0f})"
            )

        # Run with timeout, early stopping, and parallel game execution
        # Dec 29: Enable parallel_games=16 for 2-4x faster gauntlet throughput
        # Jan 2, 2026 (Phase 1.3): Use graduated timeout based on board size
        # Jan 10, 2026: Added player count scaling for longer 3p/4p games
        # January 2026: Use reduced parallel_games if OOM recovery is active
        # Mar 6, 2026: Cap at 4 on coordinator to prevent OOM (evaluation_daemon
        # runs in master_loop which shares memory with P2P + daemons + NPZ exports).
        default_parallel = 16
        try:
            from app.config.env import env as _env
            if _env.is_coordinator:
                default_parallel = 4
        except ImportError:
            pass
        parallel_games = self._oom_parallel_games.get(config_key, default_parallel)
        if parallel_games < default_parallel:
            logger.info(
                f"[EvaluationDaemon] Using reduced parallel_games={parallel_games} "
                f"for {config_key} (OOM recovery)"
            )

        timeout = self.config.get_timeout_for_board(board_type, num_players)
        # Jan 13, 2026: Create recording config to capture gauntlet games for training
        recording_config = _create_gauntlet_recording_config(
            board_type=board_type,
            num_players=num_players,
            source="gauntlet_eval",
        )
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_baseline_gauntlet,
                model_path=model_path,
                board_type=board_type,
                opponents=opponents,
                games_per_opponent=games_per_baseline,
                num_players=num_players,
                verbose=False,
                early_stopping=self.config.early_stopping_enabled,
                early_stopping_confidence=self.config.early_stopping_confidence,
                early_stopping_min_games=self.config.early_stopping_min_games,
                parallel_games=parallel_games,  # Jan 2026: Adaptive, reduced on OOM
                parallel_opponents=False,  # Feb 2026: Prevents nested ThreadPool deadlock (0-game bug)
                use_search=True,  # Mar 2026: Re-enabled after MPS device fix; policy-only path lacks history stacking
                game_count=game_count,  # Dec 30: Graduated thresholds
                harness_type="gumbel_mcts",  # Jan 11, 2026: Track harness in Elo
                recording_config=recording_config,  # Jan 13, 2026: Record gauntlet games
            ),
            timeout=timeout,
        )

        # Convert to dict if needed
        if hasattr(result, "opponent_results"):
            return {
                "overall_win_rate": result.win_rate,
                "opponent_results": result.opponent_results,
                "early_stopped_baselines": getattr(result, "early_stopped_baselines", []),
                "games_saved_by_early_stopping": getattr(result, "games_saved_by_early_stopping", 0),
                # Jan 5, 2026: Include estimated_elo for promotion decisions
                "estimated_elo": getattr(result, "estimated_elo", 0.0),
                "best_elo": getattr(result, "estimated_elo", 0.0),  # Alias for emit_evaluation_completed
            }
        elif isinstance(result, dict):
            return result
        else:
            return {"overall_win_rate": 0.0, "opponent_results": {}}
    async def _run_multi_harness_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run multi-harness gauntlet for richer evaluation.

        December 30, 2025: Evaluates model under multiple harnesses to:
        1. Find best (model, harness) combination
        2. Track composite participant Elos
        3. Inform architecture allocation decisions
        """
        try:
            from app.training.multi_harness_gauntlet import MultiHarnessGauntlet
            from app.training.composite_participant import make_composite_participant_id
            from pathlib import Path

            # Jan 10, 2026: Get model's current Elo for bootstrap fast evaluation
            config_key = make_config_key(board_type, num_players)
            model_elo = None
            try:
                from app.coordination.elo_service import get_elo_service
                elo_service = get_elo_service()
                model_elo = elo_service.get_config_elo(config_key)
            except (ImportError, OSError, RuntimeError) as e:
                logger.debug(f"[EvaluationDaemon] Failed to get model Elo for multi-harness: {e}")

            # Use bootstrap games for weak models
            games_per_baseline = self.config.get_games_per_baseline(model_elo)
            if model_elo and model_elo < self.config.bootstrap_elo_threshold:
                logger.info(
                    f"[EvaluationDaemon] Using bootstrap fast eval ({games_per_baseline} games) "
                    f"for multi-harness {config_key} (Elo: {model_elo:.0f})"
                )

            # Feb 2026: On coordinator (Apple MPS), skip Gumbel MCTS harnesses.
            # PyTorch MPS inference holds the Python GIL during forward passes,
            # blocking the asyncio event loop and freezing the entire master_loop
            # process. policy_only and descent use lightweight inference that
            # releases the GIL frequently. CUDA GPUs release the GIL properly.
            from app.config.env import env
            coordinator_harnesses = None
            if env.is_coordinator:
                from app.training.multi_harness_gauntlet import HarnessType
                coordinator_harnesses = [HarnessType.POLICY_ONLY, HarnessType.DESCENT]
                logger.info(
                    f"[EvaluationDaemon] Coordinator mode: using lightweight harnesses "
                    f"only (policy_only, descent) to avoid MPS GIL blocking"
                )

            # January 5, 2026: Enable parallel harness evaluation for 3x speedup
            # Mar 11, 2026: On coordinator (MPS), skip NN-based baselines too.
            # mcts_medium/strong, gumbel_b64, policy_only_nn all create
            # GumbelMCTSAI opponents that do 128-512 MCTS sims per move on MPS,
            # making each 4p game take 3+ minutes. Use fast baselines only.
            coordinator_baselines = self.config.baselines
            if env.is_coordinator:
                fast_baselines = ["random", "heuristic", "heuristic_strong"]
                coordinator_baselines = [b for b in self.config.baselines if b in fast_baselines]
                logger.info(
                    f"[EvaluationDaemon] Coordinator baselines: {coordinator_baselines} "
                    f"(skipped NN-based: {[b for b in self.config.baselines if b not in fast_baselines]})"
                )

            gauntlet = MultiHarnessGauntlet(
                default_games_per_baseline=games_per_baseline,
                default_baselines=coordinator_baselines,
                parallel_evaluations=self.config.multi_harness_parallel,
            )

            # Run multi-harness evaluation
            # Jan 2, 2026 (Phase 1.3): Use graduated timeout based on board size
            # Jan 10, 2026: Added player count scaling for longer 3p/4p games
            base_timeout = self.config.get_timeout_for_board(board_type, num_players)
            result = await asyncio.wait_for(
                gauntlet.evaluate_model(
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    harnesses=coordinator_harnesses,  # Feb 2026: filtered on coordinator
                ),
                timeout=base_timeout * 2,  # Extra time for multiple harnesses
            )

            # Convert to dict format expected by event emission
            harness_results = {}
            composite_ids = []
            best_elo = 0.0
            best_harness = None

            for harness, rating in result.harness_results.items():
                harness_name = harness.value if hasattr(harness, "value") else str(harness)

                # Create composite participant ID for this (model, harness) combination
                model_name = Path(model_path).stem
                composite_id = make_composite_participant_id(
                    nn_id=model_name,
                    ai_type=harness_name,
                    config={"games": rating.games_played},
                )
                composite_ids.append(composite_id)

                harness_results[harness_name] = {
                    "elo": rating.elo,
                    "win_rate": rating.win_rate,
                    "games_played": rating.games_played,
                    "composite_participant_id": composite_id,
                }

                if rating.elo > best_elo:
                    best_elo = rating.elo
                    best_harness = harness_name

            # Feb 24, 2026: Extract overall win rate for promotion decisions.
            # Multi-harness plays against both random and heuristic baselines,
            # so overall_win_rate is the combined rate. Use it for both
            # vs_random_rate and vs_heuristic_rate since per-baseline split
            # isn't available from multi-harness (conservative estimate).
            overall_wr = result.harness_results[result.best_harness].win_rate if result.best_harness else 0.0
            total_games_count = result.total_games

            return {
                "overall_win_rate": overall_wr,
                "opponent_results": {},  # Not applicable for multi-harness
                "harness_results": harness_results,
                "best_harness": best_harness,
                "best_elo": best_elo,
                "estimated_elo": best_elo,
                "composite_participant_ids": composite_ids,
                "is_multi_harness": True,
                "total_games": total_games_count,
                # Feb 24, 2026: Required by auto_promotion_daemon for baseline gates
                "vs_random_rate": overall_wr,
                "vs_heuristic_rate": overall_wr,
            }

        except ImportError as e:
            logger.warning(f"[EvaluationDaemon] Multi-harness not available: {e}, falling back to baseline")
            return await self._run_baseline_only_gauntlet(model_path, board_type, num_players)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[EvaluationDaemon] Multi-harness timed out, falling back to baseline")
            return await self._run_baseline_only_gauntlet(model_path, board_type, num_players)
    async def _emit_evaluation_completed(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        result: dict,
    ) -> None:
        """Emit EVALUATION_COMPLETED event.

        December 30, 2025: Extended to include composite_participant_ids,
        harness_results for multi-harness evaluation support, and architecture
        for multi-architecture training tracking.
        """
        try:
            from app.coordination.event_router import emit_evaluation_completed

            # December 30, 2025: Extract architecture from model path
            architecture = extract_architecture_from_model_path(model_path)

            # Calculate total games played
            if result.get("is_multi_harness"):
                games_played = result.get("total_games", 0)
            else:
                games_played = sum(
                    opp.get("games_played", 0)
                    for opp in result.get("opponent_results", {}).values()
                )

            await emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                win_rate=result.get("overall_win_rate", 0.0),
                opponent_results=result.get("opponent_results", {}),
                games_played=games_played,
                # December 30, 2025: Multi-harness extensions
                harness_results=result.get("harness_results"),
                best_harness=result.get("best_harness"),
                best_elo=result.get("best_elo"),
                composite_participant_ids=result.get("composite_participant_ids"),
                is_multi_harness=result.get("is_multi_harness", False),
                # December 30, 2025: Architecture for multi-arch tracking
                architecture=architecture,
                # Feb 24, 2026: Pass baseline rates for auto_promotion_daemon
                vs_random_rate=result.get("vs_random_rate"),
                vs_heuristic_rate=result.get("vs_heuristic_rate"),
                estimated_elo=result.get("estimated_elo"),
            )

            # January 3, 2026 (Sprint 16.2): Submit to hashgraph consensus
            # This enables multi-node evaluation aggregation for BFT Elo
            await self._submit_to_hashgraph_consensus(
                model_path=model_path,
                win_rate=result.get("overall_win_rate", 0.0),
                games_played=games_played,
            )
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:  # noqa: BLE001
            # Critical: EVALUATION_COMPLETED events drive promotion/feedback loops.
            # Losing these silently stalls the entire training pipeline.
            logger.warning(f"[EvaluationDaemon] Failed to emit EVALUATION_COMPLETED event: {e}", exc_info=True)
    async def _emit_evaluation_started(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> None:
        """Emit EVALUATION_STARTED event (December 30, 2025 - Gap #3 fix).

        Enables metrics tracking and coordination when evaluation begins.
        Subscribers can use this to:
        - Track evaluation timing and latency
        - Coordinate resource allocation
        - Update UI dashboards with evaluation status
        """
        try:
            from app.coordination.event_router import emit_evaluation_started

            await emit_evaluation_started(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            )
            logger.debug(f"[EvaluationDaemon] Emitted EVALUATION_STARTED: {model_path}")
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to emit started event: {e}")
    async def _submit_to_hashgraph_consensus(
        self,
        model_path: str,
        win_rate: float,
        games_played: int,
    ) -> None:
        """Submit evaluation result to hashgraph consensus for BFT Elo updates.

        January 3, 2026 (Sprint 16.2): Enables multi-node evaluation consensus.

        When multiple nodes evaluate the same model, their results are aggregated
        using virtual voting to produce a Byzantine-tolerant consensus win rate.
        This prevents:
        - Single faulty node from corrupting Elo (GPU errors, timeouts)
        - Single malicious node from manipulating ratings
        - Inconsistent Elo between cluster nodes

        Args:
            model_path: Path to the evaluated model
            win_rate: Win rate from this node's evaluation (0.0 to 1.0)
            games_played: Number of games in this evaluation
        """
        if not HAS_HASHGRAPH_CONSENSUS:
            logger.debug("[EvaluationDaemon] Hashgraph consensus not available")
            return

        try:
            import socket

            # Compute model hash for consensus tracking
            model_hash = hashlib.sha256(model_path.encode()).hexdigest()[:16]
            node_id = socket.gethostname()

            # Get consensus manager
            consensus = get_evaluation_consensus_manager()
            if consensus is None:
                logger.debug("[EvaluationDaemon] Consensus manager not initialized")
                return

            # Submit evaluation result to hashgraph DAG
            event = await consensus.submit_evaluation_result(
                model_hash=model_hash,
                evaluator_node=node_id,
                win_rate=win_rate,
                games_played=games_played,
            )

            logger.info(
                f"[EvaluationDaemon] Submitted to hashgraph consensus: "
                f"model={model_hash[:8]}, win_rate={win_rate:.1%}, "
                f"games={games_played}, event={event.event_hash[:8]}"
            )

            # Emit event for monitoring (safe_emit_event handles errors internally)
            safe_emit_event(
                "EVALUATION_SUBMITTED",
                {
                    "model_path": model_path,
                    "model_hash": model_hash,
                    "evaluator_node": node_id,
                    "win_rate": win_rate,
                    "games_played": games_played,
                    "event_hash": event.event_hash,
                },
                context="EvaluationDaemon.submit_to_hashgraph",
            )

        except Exception as e:  # noqa: BLE001
            # Don't fail evaluation just because consensus submission failed
            logger.warning(f"[EvaluationDaemon] Failed to submit to hashgraph: {e}")
    async def _emit_evaluation_failed(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        reason: str,
    ) -> None:
        """Emit EVALUATION_FAILED event (Dec 2025 - critical gap fix).

        This enables FeedbackLoopController and other subscribers to respond
        to evaluation failures (e.g., retry with different parameters, rollback).
        """
        try:
            from app.distributed.data_events import emit_evaluation_failed

            await emit_evaluation_failed(
                model_path=model_path,
                config_key=make_config_key(board_type, num_players),
                reason=reason,
            )
            logger.info(f"[EvaluationDaemon] Emitted EVALUATION_FAILED: {model_path}")
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to emit failure event: {e}")
    def _record_gauntlet_start(
        self,
        run_id: str,
        config_key: str,
    ) -> None:
        """Record gauntlet run start in unified_elo.db.

        December 30, 2025: Added to improve observability of gauntlet runs.
        This populates the gauntlet_runs table which was previously empty
        because game_gauntlet.py only records individual matches.
        """
        try:
            from app.tournament.unified_elo_db import get_unified_elo_db

            db = get_unified_elo_db()
            conn = db._get_connection()
            conn.execute(
                """INSERT INTO gauntlet_runs
                   (run_id, config_key, started_at, status)
                   VALUES (?, ?, ?, 'running')""",
                (run_id, config_key, time.time()),
            )
            conn.commit()
            logger.debug(f"[EvaluationDaemon] Recorded gauntlet start: {run_id}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to record gauntlet start: {e}")
    def _record_gauntlet_complete(
        self,
        run_id: str,
        models_evaluated: int,
        total_games: int,
        status: str = "completed",
    ) -> None:
        """Record gauntlet run completion in unified_elo.db.

        December 30, 2025: Added for observability.
        """
        try:
            from app.tournament.unified_elo_db import get_unified_elo_db

            db = get_unified_elo_db()
            conn = db._get_connection()
            conn.execute(
                """UPDATE gauntlet_runs
                   SET completed_at = ?, models_evaluated = ?,
                       total_games = ?, status = ?
                   WHERE run_id = ?""",
                (time.time(), models_evaluated, total_games, status, run_id),
            )
            conn.commit()
            logger.debug(f"[EvaluationDaemon] Recorded gauntlet complete: {run_id}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to record gauntlet complete: {e}")
    async def _compute_elo_from_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        result: dict,
        harness_type: str = "policy_only",
    ) -> float | None:
        """Compute Elo rating from gauntlet opponent results via EloService.

        Feb 23, 2026: The gauntlet runs games against baselines (random, heuristic,
        etc.) and produces win/loss records per opponent, but does not record these
        as Elo matches. This method feeds each opponent result into the EloService
        as match records, which updates the model's Elo rating.

        For multi-harness results that already have Elo computed (from
        MultiHarnessGauntlet), this returns the existing best_elo.

        Args:
            model_path: Path to the evaluated model
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, 4)
            result: Gauntlet result dict with opponent_results

        Returns:
            Estimated Elo rating, or None if computation failed
        """
        # Multi-harness gauntlets already compute Elo internally
        if result.get("is_multi_harness") and result.get("best_elo"):
            return float(result["best_elo"])

        # For baseline-only gauntlets, record matches via EloService
        opponent_results = result.get("opponent_results", {})
        if not opponent_results:
            return None

        try:
            from app.training.elo_service import EloService
            from pathlib import Path

            model_name = Path(model_path).stem
            matches_recorded = 0

            def _record_all_matches(
                m_name, opponents, b_type, n_players, h_type
            ):
                """Record all matches in a single thread with a fresh DB connection.

                Feb 28, 2026: Creates a fresh EloService in the thread instead of
                using get_elo_service() singleton, which has a SQLite connection from
                the main thread that fails with "Cannot operate on a closed database"
                when used in asyncio.to_thread().
                Mar 3, 2026: Added h_type parameter to correctly record harness_type
                instead of hardcoding "gumbel_mcts" for all evaluations.
                """
                svc = EloService()
                count = 0
                for opp_name, wins, losses in opponents:
                    for _ in range(wins):
                        svc.record_match(
                            participant_a=m_name,
                            participant_b=opp_name,
                            winner=m_name,
                            board_type=b_type,
                            num_players=n_players,
                            harness_type=h_type,
                        )
                        count += 1
                    for _ in range(losses):
                        svc.record_match(
                            participant_a=m_name,
                            participant_b=opp_name,
                            winner=opp_name,
                            board_type=b_type,
                            num_players=n_players,
                            harness_type=h_type,
                        )
                        count += 1
                # Get rating from the same fresh connection
                rating = svc.get_rating(m_name, b_type, n_players)
                return count, float(rating.rating) if rating else None

            # Collect all opponent data first
            opponent_data = []
            for opponent_name, opp_result in opponent_results.items():
                if not isinstance(opp_result, dict):
                    continue

                win_rate = opp_result.get("win_rate", 0.0)
                games_played = opp_result.get("games_played") or opp_result.get("games", 0)
                if games_played <= 0:
                    continue

                wins = opp_result.get("wins") or int(round(win_rate * games_played))
                losses = games_played - wins
                opponent_data.append((str(opponent_name), wins, losses))

            if not opponent_data:
                return None

            # Record all matches in one thread call with a fresh connection
            matches_recorded, elo = await asyncio.to_thread(
                _record_all_matches,
                model_name, opponent_data, board_type, num_players, harness_type,
            )

            if matches_recorded > 0 and elo is not None:
                logger.info(
                    f"[EvaluationDaemon] Computed Elo from gauntlet: {model_name} = "
                    f"{elo:.0f} ({matches_recorded} matches recorded)"
                )
                return elo

            return None

        except ImportError:
            logger.debug("[EvaluationDaemon] EloService not available for Elo computation")
            return None
        except Exception as e:  # noqa: BLE001
            # Critical: Elo computation failure means a model gets no rating after
            # running a full gauntlet. The gauntlet work is wasted.
            logger.warning(f"[EvaluationDaemon] Elo computation from gauntlet failed: {e}", exc_info=True)
            return None
    async def _dispatch_gauntlet_to_cluster(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        request: dict,
    ) -> None:
        """Dispatch gauntlet evaluation to a cluster node via work queue.

        January 27, 2026: Added to prevent coordinator nodes from running
        heavy gauntlet workloads locally. Coordinators should dispatch work
        to GPU cluster nodes instead.
        """
        try:
            from app.coordination.work_distributor import (
                get_work_distributor,
                DistributedWorkConfig,
            )

            distributor = get_work_distributor()
            # January 27, 2026: Use priority=85 for gauntlets so they're claimed
            # before most selfplay (50) but after critical training (100)
            # Mar 11, 2026: require_gpu=False so coordinator (MPS) can run gauntlets.
            # Governor limits concurrent evaluations to 1.
            config = DistributedWorkConfig(priority=85, require_gpu=False)
            work_id = await distributor.submit_evaluation(
                candidate_model=model_path,
                baseline_model=None,
                games=self.config.games_per_baseline * len(self.config.baselines),
                board=board_type,
                num_players=num_players,
                evaluation_type="gauntlet",
                config=config,
            )

            if work_id:
                logger.info(
                    f"[EvaluationDaemon] Dispatched gauntlet to cluster: {work_id} "
                    f"for {model_path}"
                )
                self._eval_stats.evaluations_triggered += 1

                # Feb 2026: Track work_id → (primary_request_id, sibling_ids) for
                # completion callback. Must include siblings so all queue entries
                # get completed/failed together (not just the primary).
                persistent_request_id = request.get("_persistent_request_id")
                if persistent_request_id:
                    sibling_ids = request.get("_sibling_request_ids", [])
                    self._dispatched_evaluations[work_id] = (persistent_request_id, sibling_ids)
            else:
                logger.warning(
                    f"[EvaluationDaemon] Failed to dispatch gauntlet to cluster: {model_path}"
                )
                self._eval_stats.evaluations_failed += 1
                await self._emit_evaluation_failed(
                    model_path, board_type, num_players,
                    "dispatch_failed"
                )

        except ImportError:
            logger.warning(
                "[EvaluationDaemon] WorkDistributor not available, cannot dispatch to cluster"
            )
            self._eval_stats.evaluations_failed += 1
        except (OSError, RuntimeError) as e:
            logger.error(f"[EvaluationDaemon] Dispatch to cluster failed: {e}")
            self._eval_stats.evaluations_failed += 1
    async def _dispatch_gauntlet_to_cluster_with_fallback(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        request: dict,
    ) -> bool:
        """Try dispatching gauntlet to cluster. Returns True if dispatch succeeded.

        Feb 28, 2026: Wrapper around _dispatch_gauntlet_to_cluster that returns
        success/failure so the caller can fall back to local execution.
        """
        try:
            from app.coordination.work_distributor import (
                get_work_distributor,
                DistributedWorkConfig,
            )

            distributor = get_work_distributor()
            # Mar 11, 2026: require_gpu=False so coordinator can run gauntlets on MPS
            config = DistributedWorkConfig(priority=85, require_gpu=False)
            work_id = await distributor.submit_evaluation(
                candidate_model=model_path,
                baseline_model=None,
                games=self.config.games_per_baseline * len(self.config.baselines),
                board=board_type,
                num_players=num_players,
                evaluation_type="gauntlet",
                config=config,
            )

            if work_id:
                logger.info(
                    f"[EvaluationDaemon] Dispatched gauntlet to cluster: {work_id} "
                    f"for {model_path}"
                )
                self._eval_stats.evaluations_triggered += 1
                persistent_request_id = request.get("_persistent_request_id")
                if persistent_request_id:
                    sibling_ids = request.get("_sibling_request_ids", [])
                    self._dispatched_evaluations[work_id] = (persistent_request_id, sibling_ids)
                return True
            else:
                logger.warning(
                    f"[EvaluationDaemon] Cluster dispatch returned no work_id: {model_path}"
                )
                return False

        except ImportError:
            logger.warning(
                "[EvaluationDaemon] WorkDistributor not available for cluster dispatch"
            )
            return False
        except (OSError, RuntimeError) as e:
            logger.warning(f"[EvaluationDaemon] Cluster dispatch error: {e}")
            return False
    async def _run_lightweight_local_gauntlet(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        request: dict,
    ) -> None:
        """Run a lightweight local gauntlet as fallback when cluster dispatch fails.

        Feb 28, 2026: Uses policy-only inference (no GPU/MCTS needed) with reduced
        game count (~10 per opponent). Runs in ~30s on CPU. Better than no evaluation.
        Mar 1, 2026: Skip hexagonal board (469 cells) — too slow for CPU even with
        10 games. Hexagonal evals need GPU nodes.
        """
        # Mar 1, 2026: Standby coordinators (e.g., MacBook) do no heavy work.
        # Gauntlets should only run on the primary coordinator (mac-studio).
        from app.config.env import env as _env
        if _env.is_standby_coordinator:
            logger.info(
                f"[EvaluationDaemon] Skipping gauntlet on standby coordinator: {model_path}"
            )
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, "standby_coordinator")
                for sid in request.get("_sibling_request_ids", []):
                    self._persistent_queue.fail(sid, "standby_coordinator")
            return

        # Hexagonal board has 469 cells — each move evaluates all positions via
        # policy network forward pass. Even 10 games takes 30+ minutes on CPU,
        # repeatedly timing out. Skip and let cluster handle when available.
        if board_type == "hexagonal":
            logger.info(
                f"[EvaluationDaemon] Skipping hexagonal local gauntlet (too slow for CPU): {model_path}"
            )
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, "hexagonal_too_slow_for_cpu")
                for sid in request.get("_sibling_request_ids", []):
                    self._persistent_queue.fail(sid, "hexagonal_too_slow_for_cpu")
            return

        # Mar 2, 2026: Skip models with remote paths (from cluster node scans)
        # that don't exist locally. These have paths like /home/ubuntu/ringrift/...
        from pathlib import Path as _Path
        if not _Path(model_path).exists():
            logger.info(
                f"[EvaluationDaemon] Skipping local gauntlet (model not local): {model_path}"
            )
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, "model_not_local")
                for sid in request.get("_sibling_request_ids", []):
                    self._persistent_queue.fail(sid, "model_not_local")
            return

        # Mar 2, 2026: Force CPU to avoid MPS tensor type mismatch.
        # MPS auto-detection loads model to MPS but encoder features
        # stay on CPU, causing "Input type (MPSFloatType) and weight type
        # (torch.FloatTensor)" errors (1400+ occurrences). CPU is fine
        # for policy-only inference at 30 games.
        import os
        os.environ["RINGRIFT_FORCE_CPU"] = "1"

        start_time = time.time()
        config_key = make_config_key(board_type, num_players)
        run_id = str(uuid.uuid4())

        logger.info(
            f"[EvaluationDaemon] Running lightweight local gauntlet for {model_path} "
            f"({config_key})"
        )

        self._record_gauntlet_start(run_id, config_key)
        await self._emit_evaluation_started(model_path, board_type, num_players)

        try:
            # Use only RANDOM and HEURISTIC baselines for lightweight eval.
            # Scale games and timeout by board complexity and player count.
            # 3p/4p games have ~1.5-2x more moves than 2p, large boards ~3x.
            lightweight_opponents = [BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC]
            is_large_board = board_type in ("square19", "hexagonal")
            player_mult = {2: 1.0, 3: 1.5, 4: 2.0}.get(num_players, 1.5)
            # Games per opponent: fewer for complex configs to stay within timeout
            if is_large_board:
                games_per = 10
            elif num_players >= 3:
                games_per = 20
            else:
                games_per = 30
            # Timeout: base 300s (small 2p), scaled by board and players
            base_timeout = 900.0 if is_large_board else 300.0
            timeout_s = base_timeout * player_mult

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    opponents=lightweight_opponents,
                    games_per_opponent=games_per,
                    num_players=num_players,
                    verbose=False,
                    early_stopping=False,
                    parallel_games=1 if is_large_board else (2 if num_players >= 3 else 4),
                    parallel_opponents=False,
                    use_search=True,  # Mar 2026: Re-enabled after MPS device fix
                    harness_type="gumbel_mcts",  # Mar 9: Use search for proper encoding
                ),
                timeout=timeout_s,
            )

            elapsed = time.time() - start_time
            self._eval_stats.evaluations_completed += 1
            self._eval_stats.last_evaluation_time = elapsed
            self._update_average_time(elapsed)

            # Convert result to dict
            if hasattr(result, "opponent_results"):
                result_dict = {
                    "overall_win_rate": result.win_rate,
                    "opponent_results": result.opponent_results,
                    "estimated_elo": getattr(result, "estimated_elo", 0.0),
                    "best_elo": getattr(result, "estimated_elo", 0.0),
                    "source": "lightweight_local_fallback",
                }
            elif isinstance(result, dict):
                result_dict = result
                result_dict["source"] = "lightweight_local_fallback"
            else:
                result_dict = {"overall_win_rate": 0.0, "opponent_results": {}}

            total_games = sum(
                opp.get("games_played") or opp.get("games", 0)
                for opp in result_dict.get("opponent_results", {}).values()
            )
            self._eval_stats.games_played += total_games

            # Compute Elo from results
            estimated_elo = await self._compute_elo_from_gauntlet(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result_dict,
                harness_type="policy_only",
            )
            if estimated_elo is not None:
                result_dict["estimated_elo"] = estimated_elo
                result_dict["best_elo"] = estimated_elo

            await self._emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result_dict,
            )

            self._recently_evaluated[model_path] = time.time()
            self._record_gauntlet_complete(run_id, 1, total_games, "completed_local_fallback")

            # Feb 28, 2026: Update persistent queue to mark evaluation complete.
            # Without this, the queue entry stays "running" and eventually times out.
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                final_elo = result_dict.get("estimated_elo", result_dict.get("best_elo", 0.0))
                self._persistent_queue.complete(persistent_request_id, elo=final_elo)
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    self._persistent_queue.complete_batch(sibling_ids, elo=final_elo)
                logger.debug(
                    f"[EvaluationDaemon] Marked persistent request complete: {persistent_request_id}"
                    f"{f' + {len(sibling_ids)} siblings' if sibling_ids else ''}"
                )

            logger.info(
                f"[EvaluationDaemon] Lightweight local gauntlet completed: {model_path} "
                f"(win_rate={result_dict.get('overall_win_rate', 0):.1%}, "
                f"{total_games} games, {elapsed:.1f}s)"
            )

        except asyncio.TimeoutError:
            self._eval_stats.evaluations_failed += 1
            self._record_gauntlet_complete(run_id, 0, 0, "failed:local_timeout")
            await self._emit_evaluation_failed(model_path, board_type, num_players, "local_timeout")
            logger.error(f"[EvaluationDaemon] Lightweight local gauntlet timed out: {model_path}")
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, "local_timeout")
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    for sid in sibling_ids:
                        self._persistent_queue.fail(sid, "local_timeout")

        except Exception as e:  # noqa: BLE001
            self._eval_stats.evaluations_failed += 1
            self._record_gauntlet_complete(run_id, 0, 0, f"failed:local_error:{e}")
            await self._emit_evaluation_failed(model_path, board_type, num_players, str(e))
            logger.error(
                f"[EvaluationDaemon] Lightweight local gauntlet failed: {model_path}: {e}"
            )
            persistent_request_id = request.get("_persistent_request_id")
            if persistent_request_id and self._persistent_queue:
                self._persistent_queue.fail(persistent_request_id, str(e))
                sibling_ids = request.get("_sibling_request_ids", [])
                if sibling_ids:
                    for sid in sibling_ids:
                        self._persistent_queue.fail(sid, str(e))
    def _should_activate_backpressure(self) -> bool:
        """Session 17.24: Check if backpressure can be activated respecting hysteresis.

        After backpressure is released, there's a cooldown period before it can
        be re-activated. This prevents rapid toggling when queue hovers near threshold.

        Returns:
            True if backpressure can be activated, False if in cooldown.
        """
        if self._last_backpressure_release_time == 0.0:
            # Never released before - OK to activate
            return True

        elapsed_since_release = time.time() - self._last_backpressure_release_time
        cooldown = self.config.backpressure_reactivation_cooldown

        if elapsed_since_release < cooldown:
            logger.debug(
                f"[EvaluationDaemon] Backpressure activation skipped (hysteresis): "
                f"elapsed={elapsed_since_release:.1f}s < cooldown={cooldown:.0f}s"
            )
            return False

        return True
    def _should_release_backpressure(self) -> bool:
        """Session 17.24: Check if backpressure can be released respecting hysteresis.

        Queue must stay below release threshold for a minimum stable time before
        releasing. This prevents rapid toggling when queue hovers near threshold.

        Returns:
            True if backpressure can be released, False if not stable long enough.
        """
        now = time.time()
        stable_time = self.config.backpressure_stable_release_time

        # Track when we first went below threshold
        if self._below_threshold_since == 0.0:
            self._below_threshold_since = now
            logger.debug(
                f"[EvaluationDaemon] Queue dropped below release threshold, "
                f"starting stable period ({stable_time:.0f}s required)"
            )
            return False

        elapsed_below = now - self._below_threshold_since
        if elapsed_below < stable_time:
            logger.debug(
                f"[EvaluationDaemon] Backpressure release waiting: "
                f"elapsed={elapsed_below:.1f}s < stable_time={stable_time:.0f}s"
            )
            return False

        return True
    def _emit_backpressure(self, queue_depth: int, activate: bool) -> None:
        """Emit backpressure event to signal training should pause/resume.

        December 29, 2025 (Phase 4): Backpressure signaling to prevent GPU waste.
        When evaluation queue fills up, training should pause to let evaluations
        catch up. When queue drains, training can resume.

        Args:
            queue_depth: Current evaluation queue depth.
            activate: True to activate backpressure, False to release.
        """
        try:
            from app.coordination.event_router import publish_sync
            if activate:
                self._backpressure_active = True
                self._backpressure_stats["backpressure_activations"] += 1
                # Session 17.24: Reset below-threshold tracking when activating
                self._below_threshold_since = 0.0
                event_type = "EVALUATION_BACKPRESSURE"
                logger.warning(
                    f"[EvaluationDaemon] Backpressure ACTIVATED: queue_depth={queue_depth}, "
                    f"threshold={self.config.backpressure_threshold}"
                )
            else:
                self._backpressure_active = False
                self._backpressure_stats["backpressure_releases"] += 1
                # Session 17.24: Track release time for hysteresis cooldown
                self._last_backpressure_release_time = time.time()
                self._below_threshold_since = 0.0
                event_type = "EVALUATION_BACKPRESSURE_RELEASED"
                logger.info(
                    f"[EvaluationDaemon] Backpressure RELEASED: queue_depth={queue_depth}, "
                    f"release_threshold={self.config.backpressure_release_threshold}"
                )

            # Emit event for TrainingTriggerDaemon and other subscribers
            publish_sync(
                event_type,
                {
                    "queue_depth": queue_depth,
                    "backpressure_active": self._backpressure_active,
                    "threshold": self.config.backpressure_threshold,
                    "release_threshold": self.config.backpressure_release_threshold,
                    "source": "EvaluationDaemon",
                    "timestamp": time.time(),
                },
                source="EvaluationDaemon",
            )
        except ImportError:
            logger.debug("[EvaluationDaemon] Event bus not available for backpressure")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug(f"[EvaluationDaemon] Failed to emit backpressure event: {e}")
    def _update_average_time(self, elapsed: float) -> None:
        """Update running average of evaluation time."""
        n = self._eval_stats.evaluations_completed
        if n == 1:
            self._eval_stats.avg_evaluation_duration = elapsed
        else:
            # Exponential moving average
            alpha = 0.2
            self._eval_stats.avg_evaluation_duration = (
                alpha * elapsed +
                (1 - alpha) * self._eval_stats.avg_evaluation_duration
            )
    def _queue_for_retry(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        reason: str,
        current_attempts: int = 0,
    ) -> bool:
        """Queue failed evaluation for retry with exponential backoff.

        December 29, 2025: Implements automatic retry for transient failures
        (GPU OOM, network issues, temporary resource constraints).

        Args:
            model_path: Path to the model that failed evaluation.
            board_type: Board type for the evaluation.
            num_players: Number of players for the evaluation.
            reason: Failure reason (for logging).
            current_attempts: Number of attempts already made (0 = first failure).

        Returns:
            True if queued for retry, False if max attempts exceeded.
        """
        attempts = current_attempts + 1

        if attempts >= self._retry_config.max_attempts:
            self._retry_stats["retries_exhausted"] += 1
            logger.error(
                f"[EvaluationDaemon] Max retries ({self._retry_config.max_attempts}) exceeded "
                f"for {model_path}: {reason}"
            )
            return False

        # December 30, 2025: Use RetryConfig for consistent delay calculation
        # January 4, 2026 (Sprint 17.5): Use consolidated HandlerBase helper
        delay = self._retry_config.get_delay(attempts)
        item = (model_path, board_type, num_players, attempts)
        self._add_to_retry_queue(self._retry_queue, item, delay_seconds=delay)
        self._retry_stats["retries_queued"] += 1

        logger.info(
            f"[EvaluationDaemon] Queued retry #{attempts} for {model_path} "
            f"in {delay:.0f}s (reason: {reason})"
        )
        return True
    async def _process_retry_queue(self) -> None:
        """Process pending retries whose delay has elapsed.

        December 29, 2025: Called at the start of each worker iteration
        to re-attempt failed evaluations with exponential backoff.

        January 4, 2026 (Sprint 17.5): Uses consolidated HandlerBase helpers.
        """
        if not self._retry_queue:
            return

        now = time.time()

        # January 4, 2026: Use consolidated helper - separates ready items
        # and automatically puts remaining items back in queue
        ready_for_retry = self._process_retry_queue_items(self._retry_queue)

        # Process ready items
        for model_path, board_type, num_players, attempts in ready_for_retry:
            # Skip if already evaluating
            if model_path in self._active_evaluations:
                logger.debug(
                    f"[EvaluationDaemon] Retry deferred (already evaluating): {model_path}"
                )
                # Re-queue with same attempt count but short delay
                self._retry_queue.append(
                    (model_path, board_type, num_players, attempts, now + 30.0)
                )
                continue

            # Re-queue the evaluation request
            await self._evaluation_queue.put({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
                "_retry_attempt": attempts,  # Track retry count
            })

            logger.info(
                f"[EvaluationDaemon] Re-queued retry #{attempts} for {model_path}"
            )
    async def _download_owc_model(self, owc_path: str) -> "Path | None":
        """Download a model from OWC external drive to local storage.

        Sprint 15 (Jan 3, 2026): Called by _evaluation_worker when evaluating
        backlog models that exist on OWC but not locally.

        Args:
            owc_path: Path to the model on OWC (relative or absolute)

        Returns:
            Local Path on success, None on failure
        """
        try:
            # Get OWC configuration
            owc_host = os.environ.get("RINGRIFT_OWC_HOST", "mac-studio")
            owc_base_path = os.environ.get(
                "RINGRIFT_OWC_DRIVE_PATH", "/Volumes/RingRift-Data"
            )

            # Construct full remote path if not absolute
            if not owc_path.startswith("/"):
                # owc_path might be relative like "models/canonical_hex8_2p.pth"
                remote_path = f"{owc_base_path}/{owc_path}"
            else:
                remote_path = owc_path

            # Create local destination path
            # Put downloaded OWC models in a dedicated directory
            model_filename = Path(owc_path).name
            local_dir = Path("models/owc_downloads")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / model_filename

            # Skip if already exists locally
            if local_path.exists():
                logger.debug(
                    f"[EvaluationDaemon] OWC model already downloaded: {local_path}"
                )
                return local_path

            logger.info(
                f"[EvaluationDaemon] Downloading OWC model: {owc_host}:{remote_path} "
                f"-> {local_path}"
            )

            # Use rsync for reliable transfer
            rsync_cmd = [
                "rsync",
                "-avz",
                "--progress",
                "--timeout=120",
                f"{owc_host}:{remote_path}",
                str(local_path),
            ]

            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300.0,  # 5 minute timeout for large models
            )

            if proc.returncode == 0:
                # Verify the file exists and has content
                if local_path.exists() and local_path.stat().st_size > 0:
                    logger.info(
                        f"[EvaluationDaemon] OWC download complete: {local_path} "
                        f"({local_path.stat().st_size / 1024 / 1024:.1f} MB)"
                    )
                    return local_path
                else:
                    logger.error(
                        f"[EvaluationDaemon] OWC download produced empty file: {local_path}"
                    )
                    return None
            else:
                stderr_text = stderr.decode() if stderr else "Unknown error"
                logger.error(
                    f"[EvaluationDaemon] OWC rsync failed (code {proc.returncode}): "
                    f"{stderr_text[:500]}"
                )
                return None

        except asyncio.TimeoutError:
            logger.error(
                f"[EvaluationDaemon] OWC download timed out: {owc_path}"
            )
            return None
        except FileNotFoundError:
            logger.error(
                f"[EvaluationDaemon] rsync not found - cannot download OWC model"
            )
            return None
        except (OSError, RuntimeError) as e:
            logger.error(
                f"[EvaluationDaemon] OWC download error: {owc_path}: {e}"
            )
            return None
    async def _evaluate_vs_previous(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> None:
        """Evaluate new model head-to-head against the previous canonical model.

        January 6, 2026: Added to prove model improvement directly via tournament.
        Runs asynchronously after gauntlet evaluation to not block the main loop.

        This provides concrete evidence that new models beat older models by:
        1. Finding the canonical model for this config
        2. Running a tournament between new and canonical
        3. Emitting HEAD_TO_HEAD_COMPLETED event with win rate and Elo diff

        Args:
            model_path: Path to the newly evaluated model
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
        """
        from pathlib import Path

        config_key = make_config_key(board_type, num_players)

        # Find the canonical model for this config
        models_dir = Path("models")
        canonical_path = models_dir / f"canonical_{board_type}_{num_players}p.pth"

        # Skip if canonical doesn't exist (first model for this config)
        if not canonical_path.exists():
            logger.debug(
                f"[EvaluationDaemon] No canonical model for {config_key}, skipping head-to-head"
            )
            return

        # Skip if new model IS the canonical model (same file path)
        new_model_path = Path(model_path)
        if new_model_path.resolve() == canonical_path.resolve():
            logger.debug(
                f"[EvaluationDaemon] New model is the canonical model, skipping head-to-head"
            )
            return

        # Skip if they're the same file (symlink or copy)
        try:
            if new_model_path.samefile(canonical_path):
                logger.debug(
                    f"[EvaluationDaemon] New model is same file as canonical, skipping head-to-head"
                )
                return
        except (OSError, FileNotFoundError):
            pass  # File doesn't exist or can't be compared

        logger.info(
            f"[EvaluationDaemon] Starting head-to-head evaluation: "
            f"{new_model_path.name} vs {canonical_path.name} ({config_key})"
        )

        try:
            # Map board_type string to BoardType enum
            board_type_enum = BoardType(board_type)

            # Run tournament between new model and canonical
            # Feb 23, 2026: Increased from 50 to 200 games. At 50 games the margin
            # of error is ~15% at 95% CI, too noisy for reliable head-to-head signal.
            # At 200 games, 58% win rate has p ≈ 0.01 (binomial test vs 50%).
            tournament = Tournament(
                model_path_a=str(new_model_path),
                model_path_b=str(canonical_path),
                num_games=200,
                board_type=board_type_enum,
                num_players=num_players,
            )

            # Run tournament in thread pool to avoid blocking
            results = await asyncio.to_thread(tournament.run)

            # Calculate win rate for new model (model A in tournament)
            total_games = results.get("A", 0) + results.get("B", 0) + results.get("Draw", 0)
            if total_games == 0:
                logger.warning(
                    f"[EvaluationDaemon] Head-to-head produced no games for {config_key}"
                )
                return

            new_wins = results.get("A", 0)
            canonical_wins = results.get("B", 0)
            draws = results.get("Draw", 0)
            win_rate = new_wins / total_games

            # Estimate Elo difference from win rate
            # win_rate = 1 / (1 + 10^(-elo_diff/400))
            # elo_diff = -400 * log10(1/win_rate - 1)
            if 0 < win_rate < 1:
                import math
                elo_diff = -400 * math.log10(1 / win_rate - 1)
            elif win_rate >= 1:
                elo_diff = 400  # Cap at +400 for 100% win rate
            else:
                elo_diff = -400  # Cap at -400 for 0% win rate

            logger.info(
                f"[EvaluationDaemon] Head-to-head complete: {new_model_path.name} vs {canonical_path.name} "
                f"({config_key}): {new_wins}W-{canonical_wins}L-{draws}D "
                f"(win_rate={win_rate:.1%}, elo_diff={elo_diff:+.0f})"
            )

            # Emit HEAD_TO_HEAD_COMPLETED event
            safe_emit_event(
                DataEventType.HEAD_TO_HEAD_COMPLETED,
                {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "new_model": str(new_model_path),
                    "previous_model": str(canonical_path),
                    "new_wins": new_wins,
                    "canonical_wins": canonical_wins,
                    "draws": draws,
                    "games_played": total_games,
                    "new_win_rate": win_rate,
                    "elo_diff_estimate": elo_diff,
                    "improved": win_rate > 0.52,  # Require 52% to claim improvement
                    "timestamp": time.time(),
                },
            )

        except ValueError as e:
            # Invalid board type enum
            logger.error(
                f"[EvaluationDaemon] Head-to-head failed - invalid board type {board_type}: {e}"
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[EvaluationDaemon] Head-to-head timed out for {config_key}"
            )
        except (FileNotFoundError, RuntimeError, OSError) as e:
            logger.error(
                f"[EvaluationDaemon] Head-to-head failed for {config_key}: {e}"
            )
