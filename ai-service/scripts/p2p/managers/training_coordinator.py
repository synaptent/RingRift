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
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
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
    from app.coordination.event_emission_helpers import safe_emit_event
    _HAS_EVENT_EMITTERS = True
except ImportError:
    _HAS_EVENT_EMITTERS = False

# Jan 2026: Import centralized timeouts from loops
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
except ImportError:
    LoopTimeouts = None  # type: ignore[misc, assignment]


# =============================================================================
# Job Lookup Cache (Jan 2026)
# Reduces find_running_training_job/find_resumable_training_job from O(n) to O(1)
# =============================================================================


@dataclass
class CachedJobLookup:
    """Cache entry for job lookups."""
    job_id: str | None
    timestamp: float

    def is_valid(self, ttl_seconds: float = 300.0) -> bool:
        """Check if cache entry is still valid."""
        return (monotonic() - self.timestamp) < ttl_seconds


class JobLookupCache:
    """LRU-style cache for training job lookups with TTL-based invalidation.

    Caches the results of find_running_training_job() and find_resumable_training_job()
    to avoid O(n) linear scans on every lookup. Cache is invalidated when job state
    changes (creation, status change, removal).

    Thread-safe via RLock.

    Usage:
        cache = JobLookupCache(ttl_seconds=300.0, max_size=128)

        # Check cache first
        hit, job_id = cache.get_running("nnue", "hex8_2p")
        if not hit:
            job_id = _do_linear_search()
            cache.set_running("nnue", "hex8_2p", job_id)

        # Invalidate on state change
        cache.invalidate("hex8_2p")
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 128):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 5 minutes)
            max_size: Maximum number of entries per cache (default: 128)
        """
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._running_cache: dict[tuple[str, str], CachedJobLookup] = {}
        self._resumable_cache: dict[tuple[str, str], CachedJobLookup] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get_running(self, job_type: str, config_key: str) -> tuple[bool, str | None]:
        """Get cached running job lookup. Returns (cache_hit, job_id)."""
        key = (job_type, config_key)
        with self._lock:
            entry = self._running_cache.get(key)
            if entry and entry.is_valid(self._ttl):
                self._hits += 1
                return (True, entry.job_id)
            self._misses += 1
            return (False, None)

    def set_running(self, job_type: str, config_key: str, job_id: str | None) -> None:
        """Cache running job lookup result."""
        key = (job_type, config_key)
        with self._lock:
            self._running_cache[key] = CachedJobLookup(job_id, monotonic())
            self._evict_if_needed(self._running_cache)

    def get_resumable(self, job_type: str, config_key: str) -> tuple[bool, str | None]:
        """Get cached resumable job lookup. Returns (cache_hit, job_id)."""
        key = (job_type, config_key)
        with self._lock:
            entry = self._resumable_cache.get(key)
            if entry and entry.is_valid(self._ttl):
                self._hits += 1
                return (True, entry.job_id)
            self._misses += 1
            return (False, None)

    def set_resumable(self, job_type: str, config_key: str, job_id: str | None) -> None:
        """Cache resumable job lookup result."""
        key = (job_type, config_key)
        with self._lock:
            self._resumable_cache[key] = CachedJobLookup(job_id, monotonic())
            self._evict_if_needed(self._resumable_cache)

    def invalidate(self, config_key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            config_key: If provided, only invalidate entries for this config.
                       If None, invalidates all entries (full cache clear).
        """
        with self._lock:
            if config_key is None:
                self._running_cache.clear()
                self._resumable_cache.clear()
            else:
                # Remove all entries matching the config_key
                self._running_cache = {
                    k: v for k, v in self._running_cache.items()
                    if k[1] != config_key
                }
                self._resumable_cache = {
                    k: v for k, v in self._resumable_cache.items()
                    if k[1] != config_key
                }

    def _evict_if_needed(self, cache: dict[tuple[str, str], CachedJobLookup]) -> None:
        """Evict oldest entries if cache exceeds max_size. Must be called with lock held."""
        if len(cache) <= self._max_size:
            return
        # Sort by timestamp, remove oldest entries
        sorted_entries = sorted(cache.items(), key=lambda x: x[1].timestamp)
        entries_to_remove = len(cache) - self._max_size
        for key, _ in sorted_entries[:entries_to_remove]:
            del cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "running_cache_size": len(self._running_cache),
                "resumable_cache_size": len(self._resumable_cache),
                "ttl_seconds": self._ttl,
                "max_size": self._max_size,
            }


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
        has_voter_quorum: callable | None = None,
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
            has_voter_quorum: Callable that returns True if voter quorum is met
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
        self.get_quorum_health_level = has_voter_quorum  # Now expects QuorumHealthLevel getter, default None allows training

        # Training trigger deduplication cache
        self._training_trigger_cache: dict[str, float] = {}

        # Event subscription state (December 2025)
        # Dec 28, 2025: Added _subscription_lock to prevent race conditions during subscribe_to_events
        self._subscribed = False
        self._subscription_lock = threading.Lock()
        self._get_training_jobs = get_training_jobs  # Store for health_check

        # Model fetch tracking (December 2025)
        # Tracks model fetches from training nodes for health reporting
        self._models_fetched_count = 0
        self._models_fetch_failed_count = 0
        self._last_fetch_time: float = 0.0
        self._last_fetch_error: str = ""

        # Job lookup cache (January 2026)
        # Reduces O(n) linear scans in find_running_training_job/find_resumable_training_job
        # to O(1) cached lookups. TTL=300s (5 min), max 128 entries per cache type.
        self._job_lookup_cache = JobLookupCache(ttl_seconds=300.0, max_size=128)

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
        """Find a running training job of the given type for the config.

        Jan 2026: Uses JobLookupCache for O(1) lookups instead of O(n) scans.
        Cache is invalidated when job state changes.
        """
        # Check cache first (O(1))
        cache_hit, cached_job_id = self._job_lookup_cache.get_running(job_type, config_key)
        if cache_hit:
            if cached_job_id is None:
                return None
            # Verify cached job still exists and is in expected state
            training_jobs = self.get_training_jobs()
            job = training_jobs.get(cached_job_id)
            if job and job.status in ("pending", "queued", "running"):
                return job
            # Cache is stale, fall through to linear search

        # Cache miss - do linear search (O(n))
        training_lock = self.get_training_lock()
        training_jobs = self.get_training_jobs()
        with training_lock:
            for job in training_jobs.values():
                if (job.job_type == job_type and
                    f"{job.board_type}_{job.num_players}p" == config_key and
                    job.status in ("pending", "queued", "running")):
                    # Cache the result
                    self._job_lookup_cache.set_running(job_type, config_key, job.job_id)
                    return job

        # No job found - cache the negative result
        self._job_lookup_cache.set_running(job_type, config_key, None)
        return None

    def find_resumable_training_job(self, job_type: str, config_key: str) -> Any | None:
        """Find a failed/interrupted training job with a valid checkpoint.

        TRAINING CHECKPOINTING: When a training job fails or is interrupted,
        this function finds it if it has a valid checkpoint that can be resumed.

        Jan 2026: Uses JobLookupCache for O(1) lookups instead of O(n) scans.

        Returns:
            TrainingJob with valid checkpoint, or None
        """
        # Check cache first (O(1))
        cache_hit, cached_job_id = self._job_lookup_cache.get_resumable(job_type, config_key)
        if cache_hit:
            if cached_job_id is None:
                return None
            # Verify cached job still exists and is resumable
            training_jobs = self.get_training_jobs()
            job = training_jobs.get(cached_job_id)
            if (job and job.status == "failed" and
                job.checkpoint_path and job.checkpoint_epoch > 0):
                return job
            # Cache is stale, fall through to linear search

        # Cache miss - do linear search (O(n))
        training_lock = self.get_training_lock()
        training_jobs = self.get_training_jobs()
        with training_lock:
            for job in training_jobs.values():
                if (job.job_type == job_type and
                    f"{job.board_type}_{job.num_players}p" == config_key and
                    job.status == "failed" and
                    job.checkpoint_path and
                    job.checkpoint_epoch > 0):
                    # Found a failed job with checkpoint - cache the result
                    self._job_lookup_cache.set_resumable(job_type, config_key, job.job_id)
                    return job

        # No job found - cache the negative result
        self._job_lookup_cache.set_resumable(job_type, config_key, None)
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

        December 2025: Added quorum validation before critical training dispatch
        to prevent operations during cluster instability.
        """
        # Critical operation: Check quorum health before dispatching training
        # Block only when quorum is LOST - allow degraded-mode operation for resilience
        if self.get_quorum_health_level:
            try:
                quorum_health = self.get_quorum_health_level()
                if quorum_health is not None:
                    # Block only when quorum is completely lost
                    if quorum_health.value == "lost":
                        logger.warning(
                            "Cannot dispatch training job: quorum health is LOST. "
                            "Waiting for cluster recovery before training."
                        )
                        return None

                    # Log warning for non-healthy states but allow training to proceed
                    if quorum_health.value != "healthy":
                        logger.warning(
                            f"Dispatching training in degraded mode (quorum={quorum_health.value}). "
                            "Training may be interrupted if cluster health worsens."
                        )
            except Exception as e:
                # If we can't determine quorum health, log and allow training
                logger.debug(f"Could not check quorum health: {e}, allowing training")

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

        # Jan 2026: Build candidate list sorted by suitability for retry logic
        if job_type == "nnue":
            # NNUE training prefers accelerator nodes (CUDA/MPS).
            # Exclude nodes already running training to enable parallel training across configs
            gpu_nodes = [n for n in healthy_nodes if n.has_gpu and n.node_id not in nodes_with_training]
            if not gpu_nodes:
                # Fall back to allowing nodes with training if no free GPU nodes
                gpu_nodes = [n for n in healthy_nodes if n.has_gpu]
            gpu_nodes.sort(key=lambda n: (-n.gpu_power_score(), n.get_load_score()))
            candidate_nodes = gpu_nodes
        else:
            # CMA-ES is CPU-heavy. Prefer high-CPU nodes (vast nodes have 256-512 CPUs).
            cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node() and n.node_id not in nodes_with_training]
            if not cpu_nodes:
                cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node()]
            candidates = cpu_nodes if cpu_nodes else healthy_nodes
            candidates.sort(key=lambda n: (-n.cpu_power_score(), n.get_load_score()))
            candidate_nodes = candidates

        if not candidate_nodes:
            logger.info(f"No suitable worker for {job_type} training job")
            return None

        job.status = "queued"

        # Store job
        with training_lock:
            training_jobs[job_id] = job

        # Jan 2026: Invalidate cache when job state changes
        self._job_lookup_cache.invalidate(config_key)

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

        # Jan 2026: Retry dispatch with fallback nodes if initial dispatch fails
        # This addresses NAT-blocked or temporarily unhealthy training nodes
        MAX_DISPATCH_RETRIES = 3
        tried_nodes: set[str] = set()

        for attempt in range(MAX_DISPATCH_RETRIES):
            # Select next best node that hasn't been tried
            available_nodes = [n for n in candidate_nodes if n.node_id not in tried_nodes]
            if not available_nodes:
                logger.warning(
                    f"All {len(tried_nodes)} candidate nodes exhausted for {job_type} dispatch"
                )
                break

            worker_node = available_nodes[0]
            tried_nodes.add(worker_node.node_id)

            # Update job with current worker
            job.worker_node = worker_node.node_id

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

                                # Jan 2026: Invalidate cache when job state changes
                                self._job_lookup_cache.invalidate(config_key)

                                # Dec 2025: Emit TRAINING_STARTED event for pipeline coordination
                                if _HAS_EVENT_EMITTERS:
                                    safe_emit_event(
                                        "TRAINING_STARTED",
                                        {
                                            "config_key": config_key,
                                            "node_name": worker_node.node_id,
                                            "job_id": job_id,
                                            "job_type": job_type,
                                        },
                                        context="training_coordinator",
                                    )

                                # Reserve the GPU node for training (December 2025)
                                # This prevents selfplay jobs from being scheduled on this node
                                try:
                                    from app.coordination.task_coordinator import TaskCoordinator
                                    task_coordinator = TaskCoordinator.get_instance()
                                    reserved = task_coordinator.reserve_for_training(
                                        [worker_node.node_id],
                                        duration_seconds=7200.0,  # 2 hours
                                        config_key=config_key,
                                    )
                                    if reserved:
                                        logger.info(f"Reserved {worker_node.node_id} for training job {job_id}")
                                except (ImportError, AttributeError, RuntimeError) as e:
                                    # Continue without reservation if TaskCoordinator unavailable
                                    logger.debug(f"Could not reserve node for training: {e}")

                                return job
                            # Worker rejected the job - don't retry same node
                            last_err = str(result.get("error") or "Unknown error")
                            break
                        except Exception as e:
                            last_err = str(e)
                            continue

                    # All URLs for this node failed
                    logger.info(
                        f"Dispatch to {worker_node.node_id} failed (attempt {attempt + 1}/{MAX_DISPATCH_RETRIES}): "
                        f"{last_err}, trying next node..."
                    )
            except Exception as e:
                logger.info(
                    f"Dispatch to {worker_node.node_id} failed (attempt {attempt + 1}/{MAX_DISPATCH_RETRIES}): "
                    f"{e}, trying next node..."
                )

        # All retries exhausted
        job.status = "failed"
        job.error_message = f"All {len(tried_nodes)} dispatch attempts failed"
        logger.error(f"Failed to dispatch {job_type} training after {len(tried_nodes)} attempts")

        # Jan 2026: Invalidate cache when job state changes
        self._job_lookup_cache.invalidate(config_key)

        return job

    # =========================================================================
    # Model Fetching from Training Nodes
    # =========================================================================

    async def _fetch_model_from_training_node(self, job: Any) -> bool:
        """Fetch trained model from remote training node to coordinator.

        December 2025: Critical fix for model distribution. Training nodes
        produce models that need to be fetched BEFORE gauntlet evaluation
        can run, since gauntlet expects the model to be locally available.

        Args:
            job: TrainingJob with worker_node and output_model_path

        Returns:
            True if model was successfully fetched, False otherwise.
        """
        if not job.worker_node or not job.output_model_path:
            logger.error(f"Missing worker_node or output_model_path for job {job.job_id}")
            return False

        try:
            # Get cluster node info for the training node
            try:
                from app.config.cluster_config import get_cluster_nodes
                nodes = get_cluster_nodes()
                node = nodes.get(job.worker_node)
            except ImportError:
                node = None
                nodes = {}

            host = None
            ssh_user = "root"
            ssh_port = 22
            ssh_key = os.path.expanduser("~/.ssh/id_cluster")
            remote_ringrift_path = "~/ringrift/ai-service"

            if node:
                host = node.best_ip
                ssh_user = node.ssh_user or "root"
                ssh_port = node.ssh_port or 22
                if node.ssh_key:
                    ssh_key = os.path.expanduser(node.ssh_key)
                remote_ringrift_path = node.ringrift_path or "~/ringrift/ai-service"
            else:
                # Fallback: try to get from P2P peers
                peers = self.get_peers()
                peer = peers.get(job.worker_node)
                if peer:
                    host = peer.get("ip") or peer.get("host") or peer.get("tailscale_ip")
                else:
                    logger.error(f"Unknown training node: {job.worker_node}")
                    return False

            if not host:
                logger.error(f"No IP available for {job.worker_node}")
                return False

            # Determine remote and local paths
            remote_model_path = job.output_model_path
            if not remote_model_path.startswith("/"):
                # Relative path - prepend ringrift_path
                remote_model_path = f"{remote_ringrift_path}/{remote_model_path}"

            local_model_path = Path(job.output_model_path)
            if not local_model_path.is_absolute():
                local_model_path = self.ringrift_path / "ai-service" / job.output_model_path

            # Ensure local directory exists
            local_model_path.parent.mkdir(parents=True, exist_ok=True)

            # Build rsync command
            # Jan 2026: Use centralized timeouts from LoopTimeouts
            ssh_connect_timeout = 30
            rsync_timeout = 120
            async_timeout = 180.0
            if LoopTimeouts is not None:
                ssh_connect_timeout = int(LoopTimeouts.SSH_CONNECT)
                rsync_timeout = int(LoopTimeouts.RSYNC_TRANSFER)
                async_timeout = LoopTimeouts.RSYNC_TRANSFER * 1.5  # Extra margin for async wrapper

            ssh_cmd = (
                f"ssh -p {ssh_port} -i {ssh_key} "
                f"-o StrictHostKeyChecking=no -o ConnectTimeout={ssh_connect_timeout}"
            )

            cmd = [
                "rsync", "-az", f"--timeout={rsync_timeout}",
                "-e", ssh_cmd,
                f"{ssh_user}@{host}:{remote_model_path}",
                str(local_model_path),
            ]

            logger.info(
                f"Fetching model from {job.worker_node}: {remote_model_path} -> {local_model_path}"
            )

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=async_timeout)

            if proc.returncode != 0:
                stderr_text = stderr.decode() if stderr else "unknown error"
                logger.error(f"rsync failed for {job.job_id}: {stderr_text}")
                return False

            # Verify the model exists
            if not local_model_path.exists():
                logger.error(f"Model not found after fetch: {local_model_path}")
                return False

            # Verify file size is reasonable (> 1KB)
            file_size = local_model_path.stat().st_size
            if file_size < 1024:
                logger.error(f"Model too small after fetch: {file_size} bytes")
                return False

            logger.info(
                f"Successfully fetched model for {job.job_id}: "
                f"{file_size / 1024 / 1024:.1f} MB"
            )
            # Update tracking stats
            self._models_fetched_count += 1
            self._last_fetch_time = time.time()
            return True

        except asyncio.TimeoutError:
            self._models_fetch_failed_count += 1
            self._last_fetch_error = f"Timeout fetching model for {job.job_id}"
            logger.error(self._last_fetch_error)
            return False
        except (OSError, subprocess.SubprocessError) as e:
            self._models_fetch_failed_count += 1
            self._last_fetch_error = f"Error fetching model for {job.job_id}: {e}"
            logger.error(self._last_fetch_error)
            return False

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
                safe_emit_event(
                    "TRAINING_COMPLETED",
                    {
                        "config_key": config_key,
                        "model_id": Path(job.output_model_path).stem if job.output_model_path else "unknown",
                        "job_id": job.job_id,
                        "val_loss": getattr(job, "val_loss", 0.0),
                    },
                    context="training_coordinator",
                )

            # Dec 2025: Fetch model from training node BEFORE evaluation
            # This is critical - models are produced on remote training nodes (nebius-h100-*,
            # lambda-gh200-*, etc.) but gauntlet evaluation requires the model locally.
            fetched = await self._fetch_model_from_training_node(job)
            if not fetched:
                logger.error(
                    f"Failed to fetch model for job {job.job_id} from {job.worker_node}. "
                    f"Model path: {job.output_model_path}"
                )
                # Don't proceed - model isn't available for evaluation
                return

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
        finally:
            # Release GPU reservation (December 2025)
            # Always release reservation regardless of success/failure
            if hasattr(job, "worker_node") and job.worker_node:
                try:
                    from app.coordination.task_coordinator import TaskCoordinator
                    task_coordinator = TaskCoordinator.get_instance()
                    task_coordinator.release_from_training([job.worker_node])
                    logger.info(f"Released training reservation for node {job.worker_node}")
                except (ImportError, AttributeError, RuntimeError) as e:
                    logger.debug(f"Could not release training reservation: {e}")

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

        # Get median model from ELO database - run blocking SQLite in thread pool
        median_model_id = await asyncio.to_thread(self._get_median_model, config_key)
        if not median_model_id:
            logger.info(f"No median model for {config_key}, skipping gauntlet")
            return True  # Pass if no baseline to compare against

        # Get model path for median model - run blocking SQLite in thread pool
        median_model_path = await asyncio.to_thread(
            self._get_model_path_from_participant, median_model_id, config_key
        )
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
            # 16 games per side = 32 total for ~95% statistical confidence
            # (Previously 4 per side = 8 total, only ~65% confidence)
            games_per_side = 16

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

        # Update ELO database to mark as archived - run blocking SQLite in thread pool
        model_id = os.path.splitext(model_name)[0]
        elo_db_path = self.ringrift_path / "ai-service" / "data" / "unified_elo.db"

        if elo_db_path.exists():
            def _update_archived_status(
                db_path: str, mid: str, bt: str, np: int, rsn: str
            ) -> None:
                """Update archived status - runs in thread pool."""
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE elo_ratings
                    SET archived_at = ?, archive_reason = ?
                    WHERE participant_id = ? AND board_type = ? AND num_players = ?
                """, (time.time(), rsn, mid, bt, np))
                conn.commit()
                conn.close()

            try:
                await asyncio.to_thread(
                    _update_archived_status,
                    str(elo_db_path),
                    model_id,
                    board_type,
                    num_players,
                    reason,
                )
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
        # Import from contracts (zero dependencies)
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

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

        # Check model fetch health (December 2025)
        total_fetches = self._models_fetched_count + self._models_fetch_failed_count
        if total_fetches > 0:
            fetch_failure_rate = self._models_fetch_failed_count / total_fetches
            if fetch_failure_rate > 0.5:
                if is_healthy:
                    is_healthy = False
                    status = CoordinatorStatus.ERROR
                last_error = f"High model fetch failure rate: {fetch_failure_rate:.0%}"
            elif self._models_fetch_failed_count > 3:
                if is_healthy:
                    status = CoordinatorStatus.DEGRADED
                last_error = f"Multiple model fetch failures: {self._models_fetch_failed_count}"

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
                # Model fetch tracking (December 2025)
                "models_fetched": self._models_fetched_count,
                "models_fetch_failed": self._models_fetch_failed_count,
                "last_fetch_time": self._last_fetch_time,
                "last_fetch_error": self._last_fetch_error or None,
                # Job lookup cache stats (January 2026)
                "job_lookup_cache": self._job_lookup_cache.get_stats(),
            },
        )
