"""TrainingCoordinator: Training job dispatch and completion workflows.

Extracted from p2p_orchestrator.py for better modularity.
Handles training readiness checks, job dispatch, gauntlet evaluation,
and model promotion workflows.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import ClientTimeout, web
    from ..models import TrainingJob, TrainingThresholds, NodeInfo

from scripts.p2p.p2p_mixin_base import EventSubscriptionMixin

logger = logging.getLogger(__name__)

# Import constants from canonical source to avoid duplication
try:
    from scripts.p2p.constants import (
        LEADERLESS_TRAINING_TIMEOUT,
        MIN_MEMORY_GB_FOR_TASKS,
    )
except ImportError:
    # Fallback for testing/standalone use
    MIN_MEMORY_GB_FOR_TASKS = 64  # Dec 2025: GH200 nodes
    LEADERLESS_TRAINING_TIMEOUT = 30  # Match constants.py

# Dec 2025: Import event emission helpers for training lifecycle events
try:
    from app.coordination.event_emitters import (
        emit_training_started_sync,
        emit_training_completed_sync,
    )
    _HAS_EVENT_EMITTERS = True
except ImportError:
    _HAS_EVENT_EMITTERS = False


class TrainingCoordinator(EventSubscriptionMixin):
    """Coordinates training job dispatch and completion workflows.

    Responsibilities:
    - Check training readiness based on data thresholds
    - Manage training job dispatch and coordination
    - Handle training completion workflows (gauntlet, promotion)
    - Prevent duplicate training triggers via hash-based deduplication

    Inherits from EventSubscriptionMixin for standardized event handling (Dec 2025).

    Usage (dependency injection pattern):
        coordinator = TrainingCoordinator(
            ringrift_path=Path("/path/to/ringrift"),
            get_cluster_data_manifest=lambda: orchestrator.cluster_data_manifest,
            get_training_jobs=lambda: orchestrator.training_jobs,
            get_training_lock=lambda: orchestrator.training_lock,
            get_peers=lambda: orchestrator.peers,
            get_peers_lock=lambda: orchestrator.peers_lock,
            get_self_info=lambda: orchestrator.self_info,
            training_thresholds=orchestrator.training_thresholds,
        )

        # Check what training jobs should start
        jobs = coordinator.check_training_readiness()

        # Dispatch a job
        job = await coordinator.dispatch_training_job(job_config)

        # Handle completion
        await coordinator.handle_training_job_completion(job)
    """

    MIXIN_TYPE = "training_coordinator"

    def __init__(
        self,
        ringrift_path: Path,
        get_cluster_data_manifest: callable,
        get_training_jobs: callable,
        get_training_lock: callable,
        get_peers: callable,
        get_peers_lock: callable,
        get_self_info: callable,
        training_thresholds: Any,  # TrainingThresholds
        games_at_last_nnue_train: dict[str, int] | None = None,
        games_at_last_cmaes_train: dict[str, int] | None = None,
        improvement_cycle_manager: Any = None,
        auth_headers: callable | None = None,
        urls_for_peer: callable | None = None,
        save_state_callback: callable | None = None,
    ):
        """Initialize the TrainingCoordinator.

        Args:
            ringrift_path: Path to RingRift repository root
            get_cluster_data_manifest: Callable that returns cluster data manifest
            get_training_jobs: Callable that returns training jobs dict
            get_training_lock: Callable that returns training lock
            get_peers: Callable that returns peers dict
            get_peers_lock: Callable that returns peers lock
            get_self_info: Callable that returns self NodeInfo
            training_thresholds: TrainingThresholds instance
            games_at_last_nnue_train: Dict tracking game counts at last NNUE training
            games_at_last_cmaes_train: Dict tracking game counts at last CMA-ES training
            improvement_cycle_manager: Optional improvement cycle manager
            auth_headers: Callable that returns auth headers dict
            urls_for_peer: Callable that returns list of URLs for a peer
            save_state_callback: Callable to save orchestrator state
        """
        self.ringrift_path = ringrift_path
        self.get_cluster_data_manifest = get_cluster_data_manifest
        self.get_training_jobs = get_training_jobs
        self.get_training_lock = get_training_lock
        self.get_peers = get_peers
        self.get_peers_lock = get_peers_lock
        self.get_self_info = get_self_info
        self.training_thresholds = training_thresholds
        self.games_at_last_nnue_train = games_at_last_nnue_train or {}
        self.games_at_last_cmaes_train = games_at_last_cmaes_train or {}
        self.improvement_cycle_manager = improvement_cycle_manager
        self.auth_headers = auth_headers or (lambda: {})
        self.urls_for_peer = urls_for_peer or (lambda peer, endpoint: [])
        self.save_state_callback = save_state_callback or (lambda: None)

        # Training trigger deduplication cache
        self._training_trigger_cache: dict[str, float] = {}

        # Event subscription state (December 2025)
        # Dec 28, 2025: Added _subscription_lock to prevent race conditions during subscribe_to_events
        self._subscribed = False
        self._subscription_lock = threading.Lock()
        self._get_training_jobs = get_training_jobs  # Store for health_check

    # =========================================================================
    # Event Subscriptions (December 2025 - uses EventSubscriptionMixin)
    # =========================================================================

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions for EventSubscriptionMixin.

        Dec 28, 2025: Migrated to use EventSubscriptionMixin pattern.

        Returns:
            Dict mapping event names to handler methods
        """
        return {
            "SELFPLAY_COMPLETE": self._on_selfplay_complete,
            "DATA_SYNC_COMPLETED": self._on_data_sync_completed,
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "REGRESSION_DETECTED": self._on_regression_detected,
            "TASK_ABANDONED": self._on_task_abandoned,
            "P2P_NODE_DEAD": self._on_node_dead,
        }

    async def _on_selfplay_complete(self, event) -> None:
        """Handle SELFPLAY_COMPLETE events - check training readiness."""
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        game_count = payload.get("game_count", 0)

        logger.debug(
            f"[TrainingCoordinator] SELFPLAY_COMPLETE: {config_key} with {game_count} games"
        )

        # Check training readiness - this will emit TRAINING_THRESHOLD_REACHED if ready
        jobs = self.check_training_readiness()
        if jobs:
            self._log_info(f"Training ready for {len(jobs)} configs after selfplay")

    async def _on_data_sync_completed(self, event) -> None:
        """Handle DATA_SYNC_COMPLETED events - check training readiness."""
        payload = self._extract_event_payload(event)
        sync_type = payload.get("sync_type", "")

        logger.debug(f"[TrainingCoordinator] DATA_SYNC_COMPLETED: {sync_type}")

        # Check training readiness after data becomes available
        jobs = self.check_training_readiness()
        if jobs:
            self._log_info(f"Training ready for {len(jobs)} configs after sync")

    async def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED events - potential model promotion."""
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        model_path = payload.get("model_path", "")
        win_rate = payload.get("win_rate", 0.0)
        passed = payload.get("passed", False)

        self._log_info(
            f"EVALUATION_COMPLETED: {config_key} win_rate={win_rate:.1%}, passed={passed}"
        )

        # Model promotion is handled by auto_promotion_daemon, but we log here
        if passed:
            self._log_info(
                f"Model {model_path} passed evaluation, "
                f"promotion will be handled by AutoPromotionDaemon"
            )

    async def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED events - may pause training."""
        payload = self._extract_event_payload(event)
        config_key = payload.get("config_key", "")
        severity = payload.get("severity", "unknown")
        elo_drop = payload.get("elo_drop", 0)

        logger.warning(
            f"[TrainingCoordinator] REGRESSION_DETECTED: {config_key} "
            f"severity={severity}, elo_drop={elo_drop}"
        )

        # For severe regressions, we could pause training
        # The RollbackManager handles actual model rollback
        if severity in ("severe", "critical"):
            logger.warning(
                f"[TrainingCoordinator] Severe regression on {config_key} - "
                f"training may need to be paused"
            )

    async def _on_task_abandoned(self, event) -> None:
        """Handle TASK_ABANDONED events - cleanup training state for abandoned jobs.

        TASK_ABANDONED is emitted when a training job is intentionally cancelled
        (e.g., user requested stop, resource constraints, or cluster rebalancing).
        This is different from TASK_FAILED which indicates an error.

        Actions taken:
        1. Clear the training trigger cache for this config (allow re-trigger)
        2. Log the abandonment for debugging
        3. Update metrics if available
        """
        payload = self._extract_event_payload(event)
        task_id = payload.get("task_id", "")
        task_type = payload.get("task_type", "")
        config_key = payload.get("config_key", "")
        reason = payload.get("reason", "unknown")

        # Only handle training-related task abandonments
        if task_type not in ("training", "nnue_training", "cmaes_training"):
            return

        self._log_info(
            f"TASK_ABANDONED: {task_type} task {task_id} for {config_key}, reason={reason}"
        )

        # Clear training trigger cache to allow re-triggering
        # Use config_key as cache key pattern
        if config_key:
            cache_keys_to_clear = [
                k for k in self._training_trigger_cache
                if config_key in k
            ]
            for key in cache_keys_to_clear:
                self._training_trigger_cache.pop(key, None)
                logger.debug(f"[TrainingCoordinator] Cleared trigger cache: {key}")

        # Reset game tracking if this was an NNUE training job
        if task_type == "nnue_training" and config_key:
            # Don't reset - abandoned job means we still have the games
            # Just allow re-triggering on next threshold check
            pass

    async def _on_node_dead(self, event) -> None:
        """Handle P2P_NODE_DEAD events - mark training jobs on dead node as failed.

        When a node dies with an active training job:
        1. Mark the training job as failed (not abandoned - this was involuntary)
        2. Clear trigger cache to allow re-triggering on a healthy node
        3. Log for debugging and alerting

        The training job reassignment is handled by the main training dispatch loop.
        """
        payload = self._extract_event_payload(event)
        node_id = payload.get("node_id", "")
        reason = payload.get("reason", "unknown")

        if not node_id:
            return

        logger.warning(f"[TrainingCoordinator] P2P_NODE_DEAD: {node_id}, reason={reason}")

        # Check if any training jobs were running on this node
        training_jobs = self.get_training_jobs()
        training_lock = self.get_training_lock()
        jobs_on_dead_node = []

        with training_lock:
            for job_id, job in training_jobs.items():
                job_node = getattr(job, "node_id", None) or getattr(job, "assigned_node", None)
                if job_node == node_id and getattr(job, "status", "") == "running":
                    jobs_on_dead_node.append((job_id, job))

        if jobs_on_dead_node:
            logger.warning(
                f"[TrainingCoordinator] Found {len(jobs_on_dead_node)} training jobs "
                f"on dead node {node_id}"
            )

            for job_id, job in jobs_on_dead_node:
                config_key = f"{getattr(job, 'board_type', 'unknown')}_{getattr(job, 'num_players', 0)}p"
                logger.warning(
                    f"[TrainingCoordinator] Training job {job_id} for {config_key} "
                    f"was running on dead node {node_id} - will be reassigned"
                )

                # Clear trigger cache to allow re-triggering
                if config_key:
                    cache_keys_to_clear = [
                        k for k in self._training_trigger_cache
                        if config_key in k
                    ]
                    for key in cache_keys_to_clear:
                        self._training_trigger_cache.pop(key, None)

    # =========================================================================
    # Training Readiness Checking
    # =========================================================================

    def check_training_readiness(self) -> list[dict[str, Any]]:
        """Check cluster data manifest for training readiness.

        Returns list of training jobs that should be triggered based on
        accumulated selfplay data.

        Called periodically by leader to check if automatic training should start.
        """
        jobs_to_start = []

        cluster_data_manifest = self.get_cluster_data_manifest()
        if not cluster_data_manifest:
            return jobs_to_start

        current_time = time.time()
        thresholds = self.training_thresholds

        # Update adaptive thresholds based on current cluster state
        peers = self.get_peers()
        self_info = self.get_self_info()
        gpu_node_count = len([p for p in peers.values()
                              if getattr(p, 'has_gpu', False) and getattr(p, 'gpu_name', '')]
                             ) + (1 if getattr(self_info, 'has_gpu', False) else 0)
        thresholds.update_from_cluster_state(gpu_node_count)

        def _cooldown_ok(job_type: str, config_key: str) -> bool:
            cooldown = thresholds.get_effective_cooldown()
            if cooldown <= 0:
                return True
            last_seen = 0.0
            training_lock = self.get_training_lock()
            training_jobs = self.get_training_jobs()
            with training_lock:
                for job in training_jobs.values():
                    if str(getattr(job, "job_type", "")) != job_type:
                        continue
                    job_key = f"{job.board_type}_{job.num_players}p"
                    if job_key != config_key:
                        continue
                    last_seen = max(
                        last_seen,
                        float(getattr(job, "completed_at", 0.0) or 0.0),
                        float(getattr(job, "started_at", 0.0) or 0.0),
                        float(getattr(job, "created_at", 0.0) or 0.0),
                    )
            if last_seen <= 0:
                return True
            return (current_time - last_seen) >= cooldown

        # Check each board type / player count combination
        for config_key, config_data in cluster_data_manifest.by_board_type.items():
            parts = config_key.split("_")
            if len(parts) < 2:
                continue
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            total_games = config_data.get("total_games", 0)

            # Check NNUE training threshold (using adaptive thresholds)
            if thresholds.auto_nnue_enabled:
                last_nnue_games = self.games_at_last_nnue_train.get(config_key, 0)
                min_games = thresholds.get_effective_min_games("nnue")
                incremental = thresholds.get_effective_incremental("nnue")
                if total_games >= min_games:
                    new_games = total_games - last_nnue_games
                    if new_games >= incremental or last_nnue_games == 0:
                        # Check cooldown
                        if not _cooldown_ok("nnue", config_key):
                            continue
                        existing_job = self.find_running_training_job("nnue", config_key)
                        if not existing_job:
                            jobs_to_start.append({
                                "job_type": "nnue",
                                "board_type": board_type,
                                "num_players": num_players,
                                "config_key": config_key,
                                "total_games": total_games,
                            })

            # Check CMA-ES optimization threshold (using adaptive thresholds)
            if thresholds.auto_cmaes_enabled:
                last_cmaes_games = self.games_at_last_cmaes_train.get(config_key, 0)
                min_games = thresholds.get_effective_min_games("cmaes")
                incremental = thresholds.get_effective_incremental("cmaes")
                if total_games >= min_games:
                    new_games = total_games - last_cmaes_games
                    if new_games >= incremental or last_cmaes_games == 0:
                        if not _cooldown_ok("cmaes", config_key):
                            continue
                        existing_job = self.find_running_training_job("cmaes", config_key)
                        if not existing_job:
                            jobs_to_start.append({
                                "job_type": "cmaes",
                                "board_type": board_type,
                                "num_players": num_players,
                                "config_key": config_key,
                                "total_games": total_games,
                            })

        return jobs_to_start

    def find_running_training_job(self, job_type: str, config_key: str) -> Any | None:
        """Find a running training job of the given type for the config."""
        training_lock = self.get_training_lock()
        training_jobs = self.get_training_jobs()
        with training_lock:
            for job in training_jobs.values():
                if (job.job_type == job_type and
                    f"{job.board_type}_{job.num_players}p" == config_key and
                    job.status in ("pending", "queued", "running")):
                    return job
        return None

    def find_resumable_training_job(self, job_type: str, config_key: str) -> Any | None:
        """Find a failed/interrupted training job with a valid checkpoint.

        TRAINING CHECKPOINTING: When a training job fails or is interrupted,
        this function finds it if it has a valid checkpoint that can be resumed.

        Returns:
            TrainingJob with valid checkpoint, or None
        """
        training_lock = self.get_training_lock()
        training_jobs = self.get_training_jobs()
        with training_lock:
            for job in training_jobs.values():
                if (job.job_type == job_type and
                    f"{job.board_type}_{job.num_players}p" == config_key and
                    job.status == "failed" and
                    job.checkpoint_path and
                    job.checkpoint_epoch > 0):
                    # Found a failed job with checkpoint
                    return job
        return None

    # =========================================================================
    # Training Job Dispatch
    # =========================================================================

    async def dispatch_training_job(self, job_config: dict[str, Any]) -> Any | None:
        """Dispatch a training job to an appropriate worker.

        Finds a GPU node for NNUE training, or any available node for CMA-ES.
        Creates a TrainingJob and sends it to the worker.

        TRAINING CHECKPOINTING: If a failed job with checkpoint exists for this
        config, includes resume info in the dispatch.
        """
        # Import TrainingJob here to avoid circular imports
        from ..models import TrainingJob

        job_type = job_config["job_type"]
        board_type = job_config["board_type"]
        num_players = job_config["num_players"]
        config_key = job_config["config_key"]

        # TRAINING CHECKPOINTING: Check for resumable failed job
        resumable = self.find_resumable_training_job(job_type, config_key)
        if resumable and not job_config.get("resume_checkpoint_path"):
            # Found a failed job with checkpoint - add resume info
            job_config["resume_checkpoint_path"] = resumable.checkpoint_path
            job_config["resume_epoch"] = resumable.checkpoint_epoch
            logger.info(f"Found resumable job {resumable.job_id} with checkpoint at epoch {resumable.checkpoint_epoch}")

        # Generate job ID
        job_id = f"{job_type}_{config_key}_{int(time.time())}"

        # Create TrainingJob
        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            status="pending",
            data_games_count=job_config.get("total_games", 0),
        )

        # Find suitable worker (CPU/GPU-aware + load-balanced)
        peers = self.get_peers()
        peers_lock = self.get_peers_lock()
        self_info = self.get_self_info()

        with peers_lock:
            all_nodes = list(peers.values())
        all_nodes.append(self_info)

        # Filter for healthy nodes with sufficient memory
        healthy_nodes = [
            n for n in all_nodes
            if n.is_healthy() and int(getattr(n, "memory_gb", 0) or 0) >= MIN_MEMORY_GB_FOR_TASKS
        ]

        # Policy-based filtering: check if work type is allowed on each node
        policy_manager = None
        try:
            from app.coordination.node_policies import get_policy_manager
            policy_manager = get_policy_manager()
        except ImportError:
            pass

        # Determine work type for policy check
        policy_work_type = "training" if job_type == "nnue" else "cpu_cmaes"

        if policy_manager:
            # Filter nodes that allow this work type
            healthy_nodes = [
                n for n in healthy_nodes
                if policy_manager.is_work_allowed(n.node_id, policy_work_type)
            ]

        # Get set of nodes already running training jobs (for parallel training across configs)
        training_lock = self.get_training_lock()
        training_jobs = self.get_training_jobs()
        with training_lock:
            nodes_with_training = {
                job.worker_node for job in training_jobs.values()
                if job.status in ("pending", "queued", "running") and job.worker_node
            }

        worker_node: Any = None
        if job_type == "nnue":
            # NNUE training prefers accelerator nodes (CUDA/MPS).
            # Exclude nodes already running training to enable parallel training across configs
            gpu_nodes = [n for n in healthy_nodes if n.has_gpu and n.node_id not in nodes_with_training]
            if not gpu_nodes:
                # Fall back to allowing nodes with training if no free GPU nodes
                gpu_nodes = [n for n in healthy_nodes if n.has_gpu]
            gpu_nodes.sort(key=lambda n: (-n.gpu_power_score(), n.get_load_score()))
            worker_node = gpu_nodes[0] if gpu_nodes else None
        else:
            # CMA-ES is CPU-heavy. Prefer high-CPU nodes (vast nodes have 256-512 CPUs).
            # Use cpu_power_score() to prioritize vast nodes over lambda nodes.
            cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node() and n.node_id not in nodes_with_training]
            if not cpu_nodes:
                cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node()]
            candidates = cpu_nodes if cpu_nodes else healthy_nodes
            # Sort by CPU power (descending) then load score (ascending)
            candidates.sort(key=lambda n: (-n.cpu_power_score(), n.get_load_score()))
            worker_node = candidates[0] if candidates else None

        if not worker_node:
            logger.info(f"No suitable worker for {job_type} training job")
            return None

        job.worker_node = worker_node.node_id
        job.status = "queued"

        # Store job
        with training_lock:
            training_jobs[job_id] = job

        # Update games count at training start
        if job_type == "nnue":
            self.games_at_last_nnue_train[config_key] = job_config.get("total_games", 0)
        else:
            self.games_at_last_cmaes_train[config_key] = job_config.get("total_games", 0)

        # TRAINING CHECKPOINTING: Check for resumable job with checkpoint
        resume_checkpoint = job_config.get("resume_checkpoint_path", "")
        resume_epoch = job_config.get("resume_epoch", 0)
        if resume_checkpoint:
            job.checkpoint_path = resume_checkpoint
            job.checkpoint_epoch = resume_epoch
            job.resume_from_checkpoint = True
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint} (epoch {resume_epoch})")

        # Send to worker
        try:
            from aiohttp import ClientTimeout

            # Use shared network utility to avoid circular import with p2p_orchestrator
            from scripts.p2p.network import get_client_session

            endpoint = f"/training/{job_type}/start"
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                payload = {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "learning_rate": job.learning_rate,
                    # TRAINING CHECKPOINTING: Include resume info
                    "resume_checkpoint": resume_checkpoint,
                    "resume_epoch": resume_epoch,
                }
                last_err: str | None = None
                for url in self.urls_for_peer(worker_node, endpoint):
                    try:
                        async with session.post(url, json=payload, headers=self.auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            logger.info(f"Started {job_type} training job {job_id} on {worker_node.node_id}")
                            self.save_state_callback()

                            # Dec 2025: Emit TRAINING_STARTED event for pipeline coordination
                            if _HAS_EVENT_EMITTERS:
                                try:
                                    emit_training_started_sync(
                                        config_key=config_key,
                                        node_name=worker_node.node_id,
                                        job_id=job_id,
                                        job_type=job_type,
                                    )
                                    logger.debug(f"Emitted TRAINING_STARTED for {config_key}")
                                except Exception as e:
                                    logger.warning(f"Failed to emit TRAINING_STARTED: {e}")

                            return job
                        job.status = "failed"
                        job.error_message = str(result.get("error") or "Unknown error")
                        return job
                    except Exception as e:
                        last_err = str(e)
                        continue
                job.status = "failed"
                job.error_message = last_err or "dispatch_failed"
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            logger.error(f"Failed to dispatch {job_type} training to {worker_node.node_id}: {e}")

        return job

    # =========================================================================
    # Training Job Completion
    # =========================================================================

    async def handle_training_job_completion(self, job: Any) -> None:
        """Handle training job completion - run gauntlet, notify cycle manager, trigger evaluation.

        This method bridges the training completion with the improvement cycle:
        1. Runs immediate gauntlet evaluation against median model
        2. Archives model if gauntlet fails (< 50% win rate vs median)
        3. Notifies improvement_cycle_manager of training completion
        4. Schedules a model comparison tournament
        """
        if not self.improvement_cycle_manager:
            return

        try:
            logger.info(f"Training job {job.job_id} completed, triggering evaluation")

            # Dec 2025: Emit TRAINING_COMPLETED event for pipeline coordination
            config_key = f"{job.board_type}_{job.num_players}p"
            if _HAS_EVENT_EMITTERS:
                try:
                    emit_training_completed_sync(
                        config_key=config_key,
                        model_id=Path(job.output_model_path).stem if job.output_model_path else "unknown",
                        job_id=job.job_id,
                        val_loss=getattr(job, "val_loss", 0.0),
                    )
                    logger.debug(f"Emitted TRAINING_COMPLETED for {config_key}")
                except Exception as e:
                    logger.warning(f"Failed to emit TRAINING_COMPLETED: {e}")

            # Run immediate gauntlet evaluation
            passed = await self._run_post_training_gauntlet(job)

            if not passed:
                # Archive model that failed gauntlet
                await self._archive_failed_model(
                    job.output_model_path,
                    job.board_type,
                    job.num_players,
                    reason="failed_post_training_gauntlet"
                )
                logger.info("Model archived: failed post-training gauntlet (< 50% vs median)")
                return  # Don't proceed with tournament scheduling

            # Notify improvement cycle manager
            self.improvement_cycle_manager.handle_training_complete(
                job.board_type,
                job.num_players,
                job.output_model_path,
                job.data_games_count or 0
            )

            # Schedule model comparison tournament
            await self._schedule_model_comparison_tournament(job)

        except Exception as e:
            logger.error(f"handling training completion for {job.job_id}: {e}")

    async def _schedule_model_comparison_tournament(self, job: Any) -> None:
        """Schedule a tournament to compare the new model against baseline."""
        if not job.output_model_path:
            return

        try:
            # Get tournament matchups from cycle manager
            matchups = self.improvement_cycle_manager.get_tournament_matchups(
                job.board_type,
                job.num_players,
                new_model_path=job.output_model_path
            )

            if not matchups:
                logger.info(f"No tournament matchups for {job.board_type}_{job.num_players}p")
                return

            logger.info(f"Scheduling {len(matchups)} tournament matchups for new model")

            # Run evaluation games (simplified - in production would dispatch to workers)
            total_games = 0

            for matchup in matchups:
                if matchup.get("purpose") == "primary_evaluation":
                    # Primary evaluation against best model
                    games = matchup.get("games", 20)
                    total_games += games
                    # Placeholder: actual tournament execution would go here
                    # For now, mark as needing external evaluation
                    logger.info(f"Tournament: {matchup['agent_a']} vs {matchup['agent_b']} ({games} games)")

            # Update cycle state - evaluation is now pending
            cycle_key = f"{job.board_type}_{job.num_players}p"
            if cycle_key in self.improvement_cycle_manager.state.cycles:
                self.improvement_cycle_manager.state.cycles[cycle_key].pending_evaluation = True
                self.improvement_cycle_manager._save_state()

        except Exception as e:
            logger.error(f"scheduling tournament: {e}")

    # =========================================================================
    # Post-Training Gauntlet
    # =========================================================================

    def _get_median_model(self, config_key: str) -> str | None:
        """Get the median-rated model for a config from ELO database.

        Returns the model_id at the 50th percentile by rating, or None if
        no models exist for this config.
        """
        elo_db_path = self.ringrift_path / "ai-service" / "data" / "unified_elo.db"
        if not elo_db_path.exists():
            return None

        # Parse config_key like "square8_2p"
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            return None
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p"))

        try:
            conn = sqlite3.connect(str(elo_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT participant_id FROM elo_ratings
                WHERE board_type = ? AND num_players = ? AND archived_at IS NULL
                ORDER BY rating
            """, (board_type, num_players))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            # Return median model (middle of sorted list)
            median_idx = len(rows) // 2
            return rows[median_idx][0]
        except Exception as e:
            logger.error(f"getting median model: {e}")
            return None

    def _get_model_path_from_participant(self, participant_id: str, config_key: str) -> str | None:
        """Get the model path for a participant from the ELO database.

        Args:
            participant_id: The participant ID (e.g., "policy_only:canonical_hex8_2p")
            config_key: The config key (e.g., "hex8_2p")

        Returns:
            The model path if found, None otherwise.
        """
        elo_db_path = self.ringrift_path / "ai-service" / "data" / "unified_elo.db"
        if not elo_db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(elo_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_path FROM participants
                WHERE participant_id = ?
            """, (participant_id,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                model_path = row[0]
                # Resolve relative paths
                if not os.path.isabs(model_path):
                    model_path = str(self.ringrift_path / "ai-service" / model_path)
                return model_path

            # Try to infer model path from participant_id
            # Format is usually "algorithm:model_id" e.g., "policy_only:canonical_hex8_2p"
            parts = participant_id.split(":", 1)
            model_id = parts[-1] if len(parts) > 1 else participant_id

            # Check common model locations
            for prefix in ["models", "models/checkpoints"]:
                candidate = self.ringrift_path / "ai-service" / prefix / f"{model_id}.pth"
                if candidate.exists():
                    return str(candidate)

            return None
        except Exception as e:
            logger.error(f"getting model path for {participant_id}: {e}")
            return None

    async def _run_post_training_gauntlet(self, job: Any) -> bool:
        """Run quick gauntlet evaluation for newly trained model.

        Model must beat the median-rated model with 50%+ win rate to pass.
        Runs 8 games total (4 as player 1, 4 as player 2) for fairness.

        Returns True if model passes, False if it should be archived.
        """
        # Check for skip flag
        if os.environ.get("RINGRIFT_SKIP_POST_TRAINING_GAUNTLET", "0") == "1":
            logger.info("Post-training gauntlet skipped (RINGRIFT_SKIP_POST_TRAINING_GAUNTLET=1)")
            return True

        config_key = f"{job.board_type}_{job.num_players}p"
        model_path = job.output_model_path

        if not model_path or not os.path.exists(model_path):
            logger.info(f"Model path not found: {model_path}, skipping gauntlet")
            return True

        model_id = os.path.splitext(os.path.basename(model_path))[0]

        # Get median model from ELO database
        median_model_id = self._get_median_model(config_key)
        if not median_model_id:
            logger.info(f"No median model for {config_key}, skipping gauntlet")
            return True  # Pass if no baseline to compare against

        # Get model path for median model
        median_model_path = self._get_model_path_from_participant(median_model_id, config_key)
        if not median_model_path or not os.path.exists(median_model_path):
            logger.info(f"Median model path not found for {median_model_id}, skipping gauntlet")
            return True

        logger.info(f"Running post-training gauntlet: {model_id} vs {median_model_id} (median)")

        try:
            # Import game playing infrastructure
            import functools
            from app.training.game_gauntlet import play_single_game, GameResult
            from app.ai.universal_ai import UniversalAI
            from app.models import BoardType
            from app.training.elo_service import get_elo_service

            # Get EloService for recording matches
            elo_service = get_elo_service()

            # Parse board type
            board_type_map = {
                "square8": BoardType.SQUARE8,
                "square19": BoardType.SQUARE19,
                "hex8": BoardType.HEX8,
                "hexagonal": BoardType.HEXAGONAL,
            }
            board_type = board_type_map.get(job.board_type)
            if not board_type:
                logger.warning(f"Unknown board type {job.board_type}, skipping gauntlet")
                return True

            num_players = job.num_players
            games_per_side = 4  # 4 games as player 1, 4 as player 2

            # Run games in executor to not block event loop
            loop = asyncio.get_event_loop()
            wins = 0
            total_games = 0

            for candidate_player in [1, 2]:
                for game_num in range(games_per_side):
                    try:
                        # Load AIs fresh for each game to avoid state issues
                        candidate_ai = UniversalAI.from_checkpoint(
                            model_path,
                            player_number=candidate_player,
                            board_type=board_type,
                            num_players=num_players,
                        )

                        opponent_player = 2 if candidate_player == 1 else 1
                        opponent_ai = UniversalAI.from_checkpoint(
                            median_model_path,
                            player_number=opponent_player,
                            board_type=board_type,
                            num_players=num_players,
                        )

                        # Play game in executor - use functools.partial to avoid closure issues
                        game_func = functools.partial(
                            play_single_game,
                            candidate_ai=candidate_ai,
                            opponent_ai=opponent_ai,
                            board_type=board_type,
                            num_players=num_players,
                            candidate_player=candidate_player,
                            seed=game_num + (1000 if candidate_player == 1 else 2000),
                        )
                        result = await loop.run_in_executor(None, game_func)

                        total_games += 1
                        if result.winner == candidate_player:
                            wins += 1
                        elif result.winner == 0:
                            wins += 0.5  # Draw counts as half win

                        # Record match to EloService for unified Elo tracking
                        try:
                            # Determine winner model ID
                            if result.winner == candidate_player:
                                winner_id = model_id
                            elif result.winner == 0:
                                winner_id = None  # Draw
                            else:
                                winner_id = median_model_id

                            elo_service.record_match(
                                participant_a=model_id,
                                participant_b=median_model_id,
                                winner=winner_id,
                                board_type=job.board_type,
                                num_players=num_players,
                                tournament_id=f"post_training_gauntlet_{config_key}",
                            )
                        except Exception as elo_err:
                            logger.debug(f"Failed to record match to Elo: {elo_err}")

                    except Exception as e:
                        logger.warning(f"Game {game_num} failed: {e}")
                        continue

            if total_games == 0:
                logger.warning("No games completed in gauntlet, passing by default")
                return True

            win_rate = wins / total_games
            passed = win_rate >= 0.5

            logger.info(
                f"Post-training gauntlet result: {wins}/{total_games} ({win_rate:.1%}) - "
                f"{'PASSED' if passed else 'FAILED'}"
            )

            return passed

        except ImportError as e:
            logger.warning(f"Could not import game gauntlet modules: {e}, skipping gauntlet")
            return True
        except Exception as e:
            logger.error(f"Gauntlet evaluation failed: {e}", exc_info=True)
            return True  # Pass on error to avoid blocking training pipeline

    async def _archive_failed_model(self, model_path: str, board_type: str,
                                     num_players: int, reason: str) -> None:
        """Archive a model that failed gauntlet evaluation.

        Moves the model file to models/archived/{config_key}/ and updates
        the ELO database to mark it as archived.
        """
        if not model_path or not os.path.exists(model_path):
            return

        config_key = f"{board_type}_{num_players}p"
        archive_dir = self.ringrift_path / "ai-service" / "models" / "archived" / config_key
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move model to archive
        model_name = os.path.basename(model_path)
        archive_path = archive_dir / model_name

        try:
            shutil.move(model_path, str(archive_path))
            logger.info(f"Archived {model_name} to {archive_dir} ({reason})")
        except Exception as e:
            logger.error(f"moving model to archive: {e}")
            return

        # Update ELO database to mark as archived
        model_id = os.path.splitext(model_name)[0]
        elo_db_path = self.ringrift_path / "ai-service" / "data" / "unified_elo.db"

        if elo_db_path.exists():
            try:
                conn = sqlite3.connect(str(elo_db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE elo_ratings
                    SET archived_at = ?, archive_reason = ?
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (time.time(), reason, model_id, board_type, num_players))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"updating ELO database for archived model: {e}")

    # =========================================================================
    # Model Promotion
    # =========================================================================

    async def promote_to_baseline(self, model_path: str, board_type: str, num_players: int, model_type: str):
        """Promote a model to the best baseline for its board type."""
        try:
            baseline_dir = self.ringrift_path / "ai-service" / "models" / model_type
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_path = baseline_dir / f"{board_type}_{num_players}p_best.pt"
            if baseline_path.exists():
                backup_path = baseline_dir / f"{board_type}_{num_players}p_prev_{int(time.time())}.pt"
                shutil.copy2(baseline_path, backup_path)
                logger.info(f"Backed up previous baseline to {backup_path}")

            shutil.copy2(model_path, baseline_path)
            logger.info(f"Promoted {model_path} to baseline at {baseline_path}")

        except Exception as e:
            logger.info(f"Baseline promotion error: {e}")

    # =========================================================================
    # Training Trigger Idempotency
    # =========================================================================

    def compute_training_trigger_hash(self, config_key: str, game_count: int) -> str:
        """Compute a hash for training trigger deduplication.

        IDEMPOTENCY: Hash is based on:
        - config_key (board_type + num_players)
        - game_count bucket (rounded to 1000 to allow minor variations)
        - time bucket (15-minute windows)

        This allows the same trigger to be rejected if attempted multiple times
        within a 15-minute window for the same approximate data state.
        """
        # Round game count to nearest 1000 to tolerate minor variations
        game_bucket = (game_count // 1000) * 1000

        # Use 15-minute time buckets
        time_bucket = int(time.time() // 900) * 900

        hash_input = f"{config_key}:{game_bucket}:{time_bucket}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def is_training_trigger_duplicate(self, trigger_hash: str) -> bool:
        """Check if a training trigger is a duplicate.

        IDEMPOTENCY: Returns True if this trigger hash was seen recently.
        """
        now = time.time()
        ttl = 900  # 15-minute TTL for trigger cache

        # Cleanup old entries
        expired = [h for h, ts in self._training_trigger_cache.items() if now - ts > ttl]
        for h in expired:
            del self._training_trigger_cache[h]

        # Check if duplicate
        if trigger_hash in self._training_trigger_cache:
            return True

        return False

    def record_training_trigger(self, trigger_hash: str) -> None:
        """Record a training trigger for deduplication."""
        self._training_trigger_cache[trigger_hash] = time.time()

    def check_training_idempotency(self, config_key: str, game_count: int) -> tuple[bool, str]:
        """Check if training can proceed (idempotency check).

        Returns:
            (can_proceed, trigger_hash) - can_proceed is False if duplicate
        """
        trigger_hash = self.compute_training_trigger_hash(config_key, game_count)

        if self.is_training_trigger_duplicate(trigger_hash):
            logger.info(f"IDEMPOTENT: Training trigger {trigger_hash[:8]} for {config_key} is duplicate, skipping")
            return False, trigger_hash

        return True, trigger_hash

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self):
        """Check health status of TrainingCoordinator.

        Returns:
            HealthCheckResult with status, training metrics, and error info
        """
        # Import HealthCheckResult with fallback
        try:
            from app.coordination.protocols import HealthCheckResult, CoordinatorStatus
        except ImportError:
            from dataclasses import dataclass as _dc, field as _field

            @_dc
            class HealthCheckResult:
                healthy: bool
                status: str = "running"
                message: str = ""
                timestamp: float = _field(default_factory=time.time)
                details: dict = _field(default_factory=dict)

            class CoordinatorStatus:
                RUNNING = "running"
                DEGRADED = "degraded"
                ERROR = "error"

        status = CoordinatorStatus.RUNNING
        is_healthy = True
        errors_count = 0
        last_error: str | None = None

        # Check training jobs
        running_jobs = 0
        failed_jobs = 0
        total_jobs = 0

        if self._get_training_jobs:
            training_jobs = self._get_training_jobs()
            for job in training_jobs.values():
                total_jobs += 1
                job_status = getattr(job, "status", "unknown")
                if job_status in ("running", "queued"):
                    running_jobs += 1
                elif job_status == "failed":
                    failed_jobs += 1
                    errors_count += 1

        # Check failure rate
        if total_jobs > 0:
            failure_rate = failed_jobs / total_jobs
            if failure_rate > 0.5:
                status = CoordinatorStatus.ERROR
                is_healthy = False
                last_error = f"High training job failure rate: {failure_rate:.0%}"
            elif failure_rate > 0.2:
                status = CoordinatorStatus.DEGRADED
                last_error = f"Elevated training job failure rate: {failure_rate:.0%}"

        # Check trigger cache size (if too large, may indicate stuck triggers)
        cache_size = len(self._training_trigger_cache)
        if cache_size > 100:
            if is_healthy:
                status = CoordinatorStatus.DEGRADED
                last_error = f"Large trigger cache ({cache_size} entries)"

        # Check if subscribed to events
        if not self._subscribed:
            if is_healthy:
                status = CoordinatorStatus.DEGRADED
                last_error = "Not subscribed to events"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status if isinstance(status, str) else status,
            message=last_error or "TrainingCoordinator healthy",
            details={
                "operations_count": total_jobs,
                "errors_count": errors_count,
                "running_jobs": running_jobs,
                "failed_jobs": failed_jobs,
                "total_jobs": total_jobs,
                "trigger_cache_size": cache_size,
                "subscribed": self._subscribed,
            },
        )
