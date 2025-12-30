"""EvaluationDaemon - Auto-evaluate models after training completes.

December 2025: Part of Phase 11 (Auto-Evaluation Pipeline).
December 27, 2025: Migrated to BaseEventHandler (Wave 4 Phase 1).

This daemon subscribes to TRAINING_COMPLETE events and automatically triggers
gauntlet evaluation for newly trained models. This closes the training loop
by ensuring every trained model gets evaluated without manual intervention.

Key Features:
- Subscribes to TRAINING_COMPLETE events
- Runs baseline gauntlet evaluation against RANDOM and HEURISTIC
- Emits EVALUATION_COMPLETED events for promotion consideration
- Supports early stopping based on statistical confidence

Usage:
    from app.coordination.evaluation_daemon import (
        EvaluationDaemon,
        get_evaluation_daemon,
    )

    # Start the daemon
    daemon = get_evaluation_daemon()
    await daemon.start()

    # Or use via DaemonManager
    from app.coordination.daemon_manager import get_daemon_manager, DaemonType
    manager = get_daemon_manager()
    await manager.start(DaemonType.EVALUATION)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import EvaluationDaemonStats

# December 2025: Event types and contracts
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_router import DataEventType, safe_emit_event

# December 2025: Distribution defaults and utilities
from app.config.coordination_defaults import DistributionDefaults
# December 30, 2025: Use centralized retry utilities
from app.utils.retry import RetryConfig
from app.coordination.unified_distribution_daemon import (
    verify_model_distribution,
    wait_for_model_availability,
)

# December 27, 2025: Use HandlerBase for subscription lifecycle (canonical location)
from app.coordination.handler_base import BaseEventHandler, EventHandlerConfig

# December 2025: Gauntlet evaluation
from app.training.game_gauntlet import BaselineOpponent, run_baseline_gauntlet

__all__ = [
    "EvaluationConfig",
    "EvaluationDaemon",
    "EvaluationStats",
    "get_evaluation_daemon",
    "start_evaluation_daemon",
]

# Singleton instance
_daemon: EvaluationDaemon | None = None


@dataclass
class EvaluationStats(EvaluationDaemonStats):
    """Statistics for the evaluation daemon.

    December 2025: Now extends EvaluationDaemonStats for consistent tracking.
    Inherits: evaluations_triggered, evaluations_completed, evaluations_failed,
              games_played, models_evaluated, promotions_triggered,
              last_evaluation_time, avg_evaluation_duration, is_healthy(), etc.
    """

    # Note: All fields now inherited from base class.
    # Backward compatibility aliases below.

    @property
    def total_games_played(self) -> int:
        """Alias for games_played (backward compatibility)."""
        return self.games_played

    @property
    def average_evaluation_time(self) -> float:
        """Alias for avg_evaluation_duration (backward compatibility)."""
        return self.avg_evaluation_duration


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation daemon."""

    # Games per baseline opponent
    # Dec 29: Increased from 20 to 50 for more statistically significant eval
    # (±5% confidence interval instead of ±10%)
    games_per_baseline: int = 50

    # Baselines to evaluate against
    baselines: list[str] = field(default_factory=lambda: ["random", "heuristic"])

    # Early stopping configuration
    early_stopping_enabled: bool = True
    early_stopping_confidence: float = 0.95
    early_stopping_min_games: int = 10

    # Concurrency
    # Dec 29: Increased from 8 to 24 for faster eval throughput
    # With 50 games per baseline, each eval still completes in ~5 min
    max_concurrent_evaluations: int = 24

    # Timeouts
    # Dec 29: Reduced from 600s to 300s for faster iteration (5 min per eval)
    # Full cycle time: 2h → 1h (12 configs × 5 min = 1h total evaluation time)
    evaluation_timeout_seconds: float = 300.0  # 5 minutes

    # Deduplication settings (December 2025)
    dedup_cooldown_seconds: float = 300.0  # 5 minute cooldown per model
    dedup_max_tracked_models: int = 1000  # Max models to track for dedup

    # December 29, 2025 (Phase 4): Backpressure settings
    # When evaluation queue depth exceeds backpressure_threshold, emit EVALUATION_BACKPRESSURE
    # to signal training should pause. Resume when queue drains below backpressure_release.
    # Dec 29: Increased thresholds for higher training throughput
    max_queue_depth: int = 100  # Maximum pending evaluations (increased from 60)
    backpressure_threshold: int = 70  # Emit backpressure at this depth (increased from 40)
    backpressure_release_threshold: int = 35  # Release backpressure at this depth (increased from 20)

    # December 30, 2025: Multi-harness evaluation
    # When enabled, models are evaluated under all compatible harnesses (GUMBEL_MCTS, MINIMAX, etc.)
    # This produces composite Elo ratings per (model, harness) combination
    enable_multi_harness: bool = True  # Use MultiHarnessGauntlet for richer evaluation
    multi_harness_max_harnesses: int = 3  # Max harnesses to evaluate (limit for speed)


class EvaluationDaemon(BaseEventHandler):
    """Daemon that auto-evaluates models after training completes.

    December 27, 2025: Migrated to BaseEventHandler - inherits:
    - Automatic event subscription/unsubscription lifecycle
    - Standard health_check() implementation
    - Error counting and last_error tracking
    - get_metrics() and get_status() for DaemonManager
    """

    def __init__(self, config: EvaluationConfig | None = None):
        # Initialize BaseEventHandler with custom config
        handler_config = EventHandlerConfig()
        handler_config.register_with_registry = False  # Singleton pattern managed externally
        super().__init__("EvaluationDaemon", handler_config)

        self.config = config or EvaluationConfig()
        self._eval_stats = EvaluationStats()  # Use _eval_stats to avoid conflict with parent stats property
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()
        self._active_evaluations: set[str] = set()

        # Deduplication tracking (December 2025)
        # Track recently evaluated models: model_path -> last_evaluation_timestamp
        self._recently_evaluated: dict[str, float] = {}
        # Track seen event content hashes to prevent duplicate triggers
        self._seen_event_hashes: set[str] = set()
        # Stats for deduplication
        self._dedup_stats = {
            "cooldown_skips": 0,
            "content_hash_skips": 0,
            "concurrent_skips": 0,
        }
        # Task reference for proper cleanup (December 2025)
        self._worker_task: asyncio.Task | None = None
        # December 29, 2025 (Phase 4): Backpressure tracking
        self._backpressure_active = False
        self._backpressure_stats = {
            "backpressure_activations": 0,
            "backpressure_releases": 0,
            "queue_full_rejections": 0,
        }
        # December 29, 2025: Retry queue for failed evaluations
        # Tuple: (model_path, board_type, num_players, attempts, next_retry_time)
        self._retry_queue: deque[tuple[str, str, int, int, float]] = deque()
        # December 30, 2025: Use centralized RetryConfig for consistent retry behavior
        self._retry_config = RetryConfig(
            max_attempts=3,
            base_delay=60.0,
            max_delay=240.0,
            jitter=0.1,  # Add slight jitter to avoid thundering herd
        )
        self._retry_stats = {
            "retries_queued": 0,
            "retries_succeeded": 0,
            "retries_exhausted": 0,
        }

    def _get_subscriptions(self) -> Dict[Any, Callable]:
        """Return event subscriptions for BaseEventHandler.

        Returns:
            Dict mapping event types to handler methods.
        """
        return {
            DataEventType.TRAINING_COMPLETED: self._on_training_complete,
        }

    async def start(self) -> bool:
        """Start the evaluation daemon.

        Returns:
            True if successfully started.
        """
        # Call parent start for event subscription
        success = await super().start()
        if not success:
            return False

        # Start the evaluation worker and store task for proper cleanup
        self._worker_task = asyncio.create_task(self._evaluation_worker())

        logger.info(
            f"[EvaluationDaemon] Started. "
            f"Games per baseline: {self.config.games_per_baseline}, "
            f"Early stopping: {self.config.early_stopping_enabled}"
        )
        return True

    async def stop(self) -> None:
        """Stop the evaluation daemon."""
        # Cancel worker task if running (December 2025)
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation

        # Call parent stop for event cleanup
        await super().stop()

        logger.info(
            f"[EvaluationDaemon] Stopped. "
            f"Evaluations: {self._eval_stats.evaluations_completed}/{self._eval_stats.evaluations_triggered}"
        )

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    async def _run_cycle(self) -> None:
        """No-op: EvaluationDaemon is purely event-driven via queue worker.

        December 29, 2025: Added to satisfy BaseEventHandler abstract requirement.
        The actual work is done by _evaluation_worker() processing the queue.
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for DaemonManager health monitoring.

        December 2025: Added to fix missing status method (P0 gap).
        December 27, 2025: Enhanced with BaseEventHandler metrics.

        Returns:
            Status dict with running state, stats, and dedup metrics.
        """
        # Get base status from parent
        base_status = super().get_status()

        # Add evaluation-specific fields
        base_status.update({
            "queue_size": self._evaluation_queue.qsize(),
            "active_evaluations": list(self._active_evaluations),
            "stats": {
                "evaluations_triggered": self._eval_stats.evaluations_triggered,
                "evaluations_completed": self._eval_stats.evaluations_completed,
                "evaluations_failed": self._eval_stats.evaluations_failed,
                "games_played": self._eval_stats.games_played,
                "models_evaluated": self._eval_stats.models_evaluated,
                "promotions_triggered": self._eval_stats.promotions_triggered,
                "last_evaluation_time": self._eval_stats.last_evaluation_time,
            },
            "dedup_stats": dict(self._dedup_stats),
            "config": {
                "games_per_baseline": self.config.games_per_baseline,
                "baselines": self.config.baselines,
                "early_stopping_enabled": self.config.early_stopping_enabled,
                "dedup_cooldown_seconds": self.config.dedup_cooldown_seconds,
            },
        })
        return base_status

    def _compute_event_hash(self, model_path: str, board_type: str, num_players: int) -> str:
        """Compute a content hash for deduplication.

        December 2025: Prevents duplicate evaluations from multiple event sources.
        """
        content = f"{model_path}:{board_type}:{num_players}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_duplicate_event(self, event_hash: str) -> bool:
        """Check if this event has been seen recently.

        December 2025: Content-based deduplication.
        """
        if event_hash in self._seen_event_hashes:
            return True

        # Add to seen set with LRU eviction
        self._seen_event_hashes.add(event_hash)
        if len(self._seen_event_hashes) > self.config.dedup_max_tracked_models:
            # Remove oldest (arbitrary in set, but prevents unbounded growth)
            self._seen_event_hashes.pop()

        return False

    def _is_in_cooldown(self, model_path: str) -> bool:
        """Check if model was recently evaluated (within cooldown period).

        December 2025: Time-based deduplication.
        """
        now = time.time()

        # Clean up old entries
        expired = [
            path for path, ts in self._recently_evaluated.items()
            if now - ts > self.config.dedup_cooldown_seconds
        ]
        for path in expired:
            del self._recently_evaluated[path]

        # Check if model is in cooldown
        last_eval = self._recently_evaluated.get(model_path)
        return last_eval is not None and now - last_eval < self.config.dedup_cooldown_seconds

    async def _on_training_complete(self, event: Any) -> None:
        """Handle TRAINING_COMPLETE event."""
        try:
            # December 30, 2025: Use consolidated extraction helpers from HandlerBase
            metadata = self._get_payload(event)
            model_path = self._extract_model_path(metadata)
            board_type, num_players = self._extract_board_config(metadata)

            if not model_path:
                logger.warning("[EvaluationDaemon] No checkpoint_path/model_path in TRAINING_COMPLETE event")
                return

            # December 2025: Deduplication checks
            # Check 1: Content hash deduplication (same event from multiple sources)
            event_hash = self._compute_event_hash(model_path, board_type, num_players)
            if self._is_duplicate_event(event_hash):
                self._dedup_stats["content_hash_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping duplicate event (content hash): {model_path}"
                )
                return

            # Check 2: Cooldown period (recently evaluated model)
            if self._is_in_cooldown(model_path):
                self._dedup_stats["cooldown_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping model in cooldown: {model_path}"
                )
                return

            # Check 3: Already being evaluated
            if model_path in self._active_evaluations:
                self._dedup_stats["concurrent_skips"] += 1
                logger.debug(
                    f"[EvaluationDaemon] Skipping already-evaluating model: {model_path}"
                )
                return

            # December 29, 2025 (Phase 4): Backpressure check
            queue_depth = self._evaluation_queue.qsize()
            if queue_depth >= self.config.max_queue_depth:
                self._backpressure_stats["queue_full_rejections"] += 1
                logger.warning(
                    f"[EvaluationDaemon] Queue full ({queue_depth}), rejecting: {model_path}"
                )
                return

            # Check and emit backpressure if needed
            if queue_depth >= self.config.backpressure_threshold and not self._backpressure_active:
                self._emit_backpressure(queue_depth, activate=True)

            # Queue the evaluation
            await self._evaluation_queue.put({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
            })

            self._eval_stats.evaluations_triggered += 1
            logger.info(
                f"[EvaluationDaemon] Queued evaluation for {model_path} "
                f"({board_type}_{num_players}p), queue_depth={queue_depth + 1}"
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[EvaluationDaemon] Invalid event data: {e}")
        except OSError as e:
            logger.error(f"[EvaluationDaemon] I/O error handling training complete: {e}")

    async def _evaluation_worker(self) -> None:
        """Worker that processes evaluation requests from the queue."""
        while self._running:
            try:
                # December 29, 2025: Process retry queue first
                await self._process_retry_queue()

                # Wait for an evaluation request
                request = await asyncio.wait_for(
                    self._evaluation_queue.get(),
                    timeout=5.0,  # Check running status periodically
                )

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

                # Run evaluation
                self._active_evaluations.add(model_path)
                try:
                    await self._run_evaluation(request)
                finally:
                    self._active_evaluations.discard(model_path)
                    # December 29, 2025 (Phase 4): Check for backpressure release
                    queue_depth = self._evaluation_queue.qsize()
                    if self._backpressure_active and queue_depth <= self.config.backpressure_release_threshold:
                        self._emit_backpressure(queue_depth, activate=False)

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

            # First quick check
            success, count = await verify_model_distribution(model_path, min_nodes)
            if success:
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

    async def _run_evaluation(self, request: dict) -> None:
        """Run gauntlet evaluation for a model."""
        model_path = request["model_path"]
        board_type = request["board_type"]
        num_players = request["num_players"]

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
        logger.info(f"[EvaluationDaemon] Starting evaluation: {model_path}")

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

            # Count games played
            total_games = sum(
                opp.get("games_played", 0)
                for opp in result.get("opponent_results", {}).values()
            )
            self._eval_stats.total_games_played += total_games

            # Emit evaluation completed event
            await self._emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result,
            )

            # December 2025: Mark as recently evaluated for deduplication
            self._recently_evaluated[model_path] = time.time()

            logger.info(
                f"[EvaluationDaemon] Evaluation completed: {model_path} "
                f"(win_rate={result.get('overall_win_rate', 0):.1%}, "
                f"{total_games} games, {elapsed:.1f}s)"
            )

        except asyncio.TimeoutError:
            self._eval_stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation timed out: {model_path}")
            # December 29, 2025: Queue for retry on timeout (transient failure)
            retry_attempt = request.get("_retry_attempt", 0)
            if self._queue_for_retry(
                model_path, board_type, num_players, "timeout", retry_attempt
            ):
                return  # Will retry, don't emit permanent failure
            # Emit EVALUATION_FAILED event (Dec 2025 - critical gap fix)
            await self._emit_evaluation_failed(model_path, board_type, num_players, "timeout")
        except (MemoryError, RuntimeError) as e:
            # December 29, 2025: GPU OOM and RuntimeError (CUDA) are retryable
            self._eval_stats.evaluations_failed += 1
            error_str = str(e).lower()
            is_gpu_error = "cuda" in error_str or "out of memory" in error_str
            logger.error(f"[EvaluationDaemon] Evaluation failed ({type(e).__name__}): {model_path}: {e}")
            if is_gpu_error:
                retry_attempt = request.get("_retry_attempt", 0)
                if self._queue_for_retry(
                    model_path, board_type, num_players, f"GPU error: {e}", retry_attempt
                ):
                    return  # Will retry
            # Emit permanent failure
            await self._emit_evaluation_failed(model_path, board_type, num_players, str(e))
        except Exception as e:  # noqa: BLE001
            self._eval_stats.evaluations_failed += 1
            logger.error(f"[EvaluationDaemon] Evaluation failed: {model_path}: {e}")
            # Emit EVALUATION_FAILED event (Dec 2025 - critical gap fix)
            await self._emit_evaluation_failed(model_path, board_type, num_players, str(e))

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
        baseline_map = {
            "random": BaselineOpponent.RANDOM,
            "heuristic": BaselineOpponent.HEURISTIC,
        }
        opponents = [
            baseline_map[b]
            for b in self.config.baselines
            if b in baseline_map
        ]

        # Run with timeout, early stopping, and parallel game execution
        # Dec 29: Enable parallel_games=16 for 2-4x faster gauntlet throughput
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_baseline_gauntlet,
                model_path=model_path,
                board_type=board_type,
                opponents=opponents,
                games_per_opponent=self.config.games_per_baseline,
                num_players=num_players,
                verbose=False,
                early_stopping=self.config.early_stopping_enabled,
                early_stopping_confidence=self.config.early_stopping_confidence,
                early_stopping_min_games=self.config.early_stopping_min_games,
                parallel_games=16,  # Dec 29: Increased for faster evaluation
            ),
            timeout=self.config.evaluation_timeout_seconds,
        )

        # Convert to dict if needed
        if hasattr(result, "opponent_results"):
            return {
                "overall_win_rate": result.win_rate,
                "opponent_results": result.opponent_results,
                "early_stopped_baselines": getattr(result, "early_stopped_baselines", []),
                "games_saved_by_early_stopping": getattr(result, "games_saved_by_early_stopping", 0),
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

            gauntlet = MultiHarnessGauntlet(
                default_games_per_baseline=self.config.games_per_baseline,
                default_baselines=self.config.baselines,
            )

            # Run multi-harness evaluation
            result = await asyncio.wait_for(
                gauntlet.evaluate_model(
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                ),
                timeout=self.config.evaluation_timeout_seconds * 2,  # Extra time for multiple harnesses
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

            return {
                "overall_win_rate": result.harness_results[result.best_harness].win_rate if result.best_harness else 0.0,
                "opponent_results": {},  # Not applicable for multi-harness
                "harness_results": harness_results,
                "best_harness": best_harness,
                "best_elo": best_elo,
                "composite_participant_ids": composite_ids,
                "is_multi_harness": True,
                "total_games": result.total_games,
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

        December 30, 2025: Extended to include composite_participant_ids and
        harness_results for multi-harness evaluation support.
        """
        try:
            from app.coordination.event_router import emit_evaluation_completed

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
            )
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to emit event: {e}")

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
                config_key=f"{board_type}_{num_players}p",
                reason=reason,
            )
            logger.info(f"[EvaluationDaemon] Emitted EVALUATION_FAILED: {model_path}")
        except ImportError:
            logger.debug("[EvaluationDaemon] Event emitters not available")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Failed to emit failure event: {e}")

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
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if activate:
                self._backpressure_active = True
                self._backpressure_stats["backpressure_activations"] += 1
                event_type = "EVALUATION_BACKPRESSURE"
                logger.warning(
                    f"[EvaluationDaemon] Backpressure ACTIVATED: queue_depth={queue_depth}, "
                    f"threshold={self.config.backpressure_threshold}"
                )
            else:
                self._backpressure_active = False
                self._backpressure_stats["backpressure_releases"] += 1
                event_type = "EVALUATION_BACKPRESSURE_RELEASED"
                logger.info(
                    f"[EvaluationDaemon] Backpressure RELEASED: queue_depth={queue_depth}, "
                    f"release_threshold={self.config.backpressure_release_threshold}"
                )

            # Emit event for TrainingTriggerDaemon and other subscribers
            bus.publish_sync(
                event_type,
                {
                    "queue_depth": queue_depth,
                    "backpressure_active": self._backpressure_active,
                    "threshold": self.config.backpressure_threshold,
                    "release_threshold": self.config.backpressure_release_threshold,
                    "source": "EvaluationDaemon",
                    "timestamp": time.time(),
                },
            )
        except ImportError:
            logger.debug("[EvaluationDaemon] Event bus not available for backpressure")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug(f"[EvaluationDaemon] Failed to emit backpressure event: {e}")

    def _update_average_time(self, elapsed: float) -> None:
        """Update running average of evaluation time."""
        n = self._eval_stats.evaluations_completed
        if n == 1:
            self._eval_stats.average_evaluation_time = elapsed
        else:
            # Exponential moving average
            alpha = 0.2
            self._eval_stats.average_evaluation_time = (
                alpha * elapsed +
                (1 - alpha) * self._eval_stats.average_evaluation_time
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
        delay = self._retry_config.get_delay(attempts)
        next_retry = time.time() + delay

        self._retry_queue.append((model_path, board_type, num_players, attempts, next_retry))
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
        """
        if not self._retry_queue:
            return

        now = time.time()
        ready_for_retry: list[tuple[str, str, int, int]] = []

        # Collect items ready for retry (next_retry_time has passed)
        # Use a temporary list to avoid modifying deque during iteration
        remaining: list[tuple[str, str, int, int, float]] = []

        while self._retry_queue:
            item = self._retry_queue.popleft()
            model_path, board_type, num_players, attempts, next_retry_time = item

            if next_retry_time <= now:
                ready_for_retry.append((model_path, board_type, num_players, attempts))
            else:
                remaining.append(item)

        # Put back items not yet ready
        for item in remaining:
            self._retry_queue.append(item)

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

    def get_stats(self) -> dict:
        """Get daemon statistics."""
        return {
            "running": self._running,
            "evaluations_triggered": self._eval_stats.evaluations_triggered,
            "evaluations_completed": self._eval_stats.evaluations_completed,
            "evaluations_failed": self._eval_stats.evaluations_failed,
            "evaluations_pending": self._evaluation_queue.qsize(),
            "active_evaluations": len(self._active_evaluations),
            "total_games_played": self._eval_stats.total_games_played,
            "average_evaluation_time": round(self._eval_stats.average_evaluation_time, 1),
            # December 2025: Deduplication stats
            "dedup_cooldown_skips": self._dedup_stats["cooldown_skips"],
            "dedup_content_hash_skips": self._dedup_stats["content_hash_skips"],
            "dedup_concurrent_skips": self._dedup_stats["concurrent_skips"],
            "tracked_recently_evaluated": len(self._recently_evaluated),
            # December 29, 2025: Retry stats
            "retry_queue_size": len(self._retry_queue),
            "retries_queued": self._retry_stats["retries_queued"],
            "retries_succeeded": self._retry_stats["retries_succeeded"],
            "retries_exhausted": self._retry_stats["retries_exhausted"],
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        December 27, 2025: Extends BaseEventHandler health_check with
        evaluation-specific failure rate detection.

        Returns:
            HealthCheckResult with status and details
        """
        # Get base health check first
        base_result = super().health_check()

        # If base check failed, return it
        if not base_result.healthy:
            return base_result

        # Additional evaluation-specific checks
        total = self._eval_stats.evaluations_triggered
        failed = self._eval_stats.evaluations_failed
        if total > 5 and failed / total > 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"High evaluation failure rate: {failed}/{total}",
                details=self.get_stats(),
            )

        # Return enhanced result with evaluation stats
        return HealthCheckResult(
            healthy=True,
            status=base_result.status,
            message=f"Evaluation daemon running ({self._eval_stats.evaluations_completed} completed)",
            details=self.get_stats(),
        )


def get_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Get or create the singleton evaluation daemon.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        EvaluationDaemon: The singleton daemon instance.
    """
    global _daemon
    if _daemon is None:
        _daemon = EvaluationDaemon(config)
    return _daemon


async def start_evaluation_daemon(config: EvaluationConfig | None = None) -> EvaluationDaemon:
    """Start the evaluation daemon (convenience function).

    Combines get_evaluation_daemon() and start() in one call.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        EvaluationDaemon: The started daemon instance.
    """
    daemon = get_evaluation_daemon(config)
    await daemon.start()
    return daemon
