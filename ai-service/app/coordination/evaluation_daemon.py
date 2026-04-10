"""EvaluationDaemon - Auto-evaluate models after training completes.

December 2025: Part of Phase 11 (Auto-Evaluation Pipeline).
December 27, 2025: Migrated to HandlerBase (Wave 4 Phase 1).

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
import os
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import EvaluationDaemonStats

# December 2025: Event types and contracts
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.event_router import DataEventType, safe_emit_event

# December 30, 2025: Use centralized retry utilities
from app.utils.retry import RetryConfig

from app.coordination.handler_base import HandlerBase
from app.coordination.evaluation_executor import EvaluationExecutorMixin

# January 3, 2026 (Sprint 13 Session 4): Persistent evaluation queue
from app.coordination.evaluation_queue import (
    PersistentEvaluationQueue,
    get_evaluation_queue,
    RequestStatus,
)

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
    # Dec 31: Reduced to 30 with 6 baselines for faster iteration (30×6=180 games)
    # Feb 2026: Reduced from 30 to 10. 30 games × 5 baselines × 4 harnesses =
    # 600 games per model, which takes hours on Apple MPS for large boards.
    # 10 games is enough for directional Elo estimation and promotion decisions.
    # Feb 23, 2026: Raised from 10 to 30. At 10 games the 95% CI is ±31%,
    # making win rates statistically meaningless. 30 games gives ±18% CI,
    # which is the minimum for reliable promotion decisions.
    games_per_baseline: int = 30

    # Jan 10, 2026: Bootstrap fast evaluation for weak models
    # Models below bootstrap_elo_threshold use fewer games per baseline for faster iteration
    # This helps break the promotion logjam during early training
    # Feb 23, 2026: Raised from 5 to 15 to give bootstrap models a meaningful
    # evaluation signal (95% CI ±25% instead of ±44%).
    bootstrap_games_per_baseline: int = 15
    bootstrap_elo_threshold: float = 1300.0

    # Baselines to evaluate against
    # Dec 31, 2025: Expanded from 2 to 6 baselines for better Elo resolution
    # Previous: ["random", "heuristic"] capped Elo measurement at ~1200
    # Now covers ~400-1600 Elo range for meaningful model ranking
    # Jan 13, 2026: Added NNUE/MINIMAX/MAXN/BRS baselines for harness diversity
    # Feb 2026: Reduced from 9 to 5 baselines. NNUE baselines (minimax_d4, maxn_d3,
    # brs_d3) removed - they require loading a separate NNUE model and are very slow
    # on Apple MPS. The remaining 5 baselines cover the full Elo range (400-1600)
    # and are sufficient for promotion decisions.
    baselines: list[str] = field(default_factory=lambda: [
        "random",           # ~400 Elo (sanity check - model should crush this)
        "heuristic",        # ~1200 Elo (basic baseline)
        "heuristic_strong", # ~1400 Elo (tuned heuristic weights)
        "policy_only_nn",   # ~1350 Elo (NN without search, tests policy head)
        "gumbel_b64",       # ~1400 Elo (search baseline with budget=64)
        "mcts_medium",      # ~1700 Elo (MCTS 128 sims, breaks 1982 ceiling)
        "mcts_strong",      # ~1900 Elo (MCTS 512 sims, enables 2000+ ratings)
    ])

    # Early stopping configuration
    early_stopping_enabled: bool = True
    early_stopping_confidence: float = 0.95
    # Feb 23, 2026: Raised from 10 to 20 so early stopping doesn't kick in
    # before enough games have been played for a meaningful signal.
    early_stopping_min_games: int = 20

    # Concurrency
    # Mar 6, 2026: Default reduced from 3 to 1.
    # Each evaluation spawns 7 baselines × 30 games × parallel_games threads.
    # On coordinator (mac-studio), 2 concurrent evals created 478 multiprocessing
    # workers → load 137 → kernel watchdog panic. 1 concurrent is safe.
    # GPU nodes can override via RINGRIFT_MAX_CONCURRENT_EVALUATIONS env var.
    max_concurrent_evaluations: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_MAX_CONCURRENT_EVALUATIONS", "1"))
    )

    # Timeouts
    # Dec 29: Reduced from 600s to 300s for faster iteration (5 min per eval)
    # Full cycle time: 2h → 1h (12 configs × 5 min = 1h total evaluation time)
    # Jan 2, 2026: Now used as fallback; board-specific timeouts in get_timeout_for_board()
    evaluation_timeout_seconds: float = 300.0  # 5 minutes (default/fallback)

    # January 2, 2026 (Phase 1.3): Graduated timeouts by board size
    # Larger boards need more time per game, so evaluation timeouts must scale.
    # hex8/square8: 64/61 cells → fast games → 1 hour
    # square19: 361 cells → 4-5x longer games → 3 hours
    # hexagonal: 469 cells → longest games → 4 hours
    # Feb 2026: With 30 games/baseline × 5 baselines = 150 games per harness,
    # timeouts need to account for the full evaluation. On Apple MPS, each game
    # takes ~2-10s depending on board size and simulation budget.
    board_timeout_seconds: dict = field(default_factory=lambda: {
        "hex8": 1800,       # 30 min - small board, fast games
        "square8": 2400,    # 40 min - small board, medium complexity
        "square19": 5400,   # 90 min - large board (Go-sized)
        "hexagonal": 7200,  # 120 min - largest board
    })

    def get_timeout_for_board(self, board_type: str, num_players: int = 2) -> float:
        """Get evaluation timeout based on board type and player count.

        January 2, 2026 (Phase 1.3): Large boards (square19, hexagonal) were
        timing out prematurely with the fixed 5-minute timeout. This method
        returns graduated timeouts based on board complexity.

        January 10, 2026: Added player count scaling. 4-player games take
        significantly longer due to more complex game trees and longer games.

        Args:
            board_type: Board type (hex8, square8, square19, hexagonal)
            num_players: Number of players (2, 3, or 4)

        Returns:
            Timeout in seconds for this board/player combination.
        """
        base_timeout = self.board_timeout_seconds.get(board_type, self.evaluation_timeout_seconds)

        # Scale timeout by player count: 4-player games need 2x time
        # 3-player games need 1.5x time
        player_multiplier = {2: 1.0, 3: 1.5, 4: 2.0}.get(num_players, 1.0)

        return base_timeout * player_multiplier

    def get_games_per_baseline(self, model_elo: float | None = None) -> int:
        """Get games per baseline based on model Elo.

        January 10, 2026: Bootstrap fast evaluation for weak models.
        Models below bootstrap_elo_threshold use fewer games for faster iteration.
        This helps break the promotion logjam during early training phases.

        Args:
            model_elo: Current model Elo rating, or None for full evaluation

        Returns:
            Number of games per baseline opponent
        """
        if model_elo is not None and model_elo < self.bootstrap_elo_threshold:
            return self.bootstrap_games_per_baseline
        return self.games_per_baseline

    # Deduplication settings (December 2025)
    # January 4, 2026: Reduced from 300s to 30s to allow rapid re-evaluations
    # after training. Previous 5-minute window was skipping valid evaluations.
    dedup_cooldown_seconds: float = 30.0  # 30 second cooldown per model
    dedup_max_tracked_models: int = 1000  # Max models to track for dedup

    # December 29, 2025 (Phase 4): Backpressure settings
    # When evaluation queue depth exceeds backpressure_threshold, emit EVALUATION_BACKPRESSURE
    # to signal training should pause. Resume when queue drains below backpressure_release.
    # Dec 29: Increased thresholds for higher training throughput
    # Jan 5, 2026: Further increased from 70/35 to 100/50 to reduce training pauses.
    # 70 was too aggressive - training blocked 5-15 min per cycle during eval queue spikes.
    max_queue_depth: int = 200  # Maximum pending evaluations (increased from 100)
    backpressure_threshold: int = 150  # Emit backpressure at this depth (increased from 70)
    backpressure_release_threshold: int = 75  # Release backpressure at this depth (increased from 35)

    # Session 17.24 (Jan 2026): Backpressure hysteresis to prevent rapid toggling
    # When queue hovers near threshold, it can toggle frequently. Hysteresis adds:
    # 1. Cooldown after release - don't re-activate for N seconds
    # 2. Minimum stable time before release - must be below threshold for N seconds
    # Session 17.31 (Jan 5, 2026): Reduced from 60s to 30s for faster backpressure cycles
    backpressure_reactivation_cooldown: float = 30.0  # Seconds before re-activation allowed
    backpressure_stable_release_time: float = 15.0  # Seconds below threshold before release

    # December 30, 2025: Multi-harness evaluation
    # When enabled, models are evaluated under all compatible harnesses (GUMBEL_MCTS, MINIMAX, etc.)
    # This produces composite Elo ratings per (model, harness) combination
    enable_multi_harness: bool = True  # Use MultiHarnessGauntlet for richer evaluation
    multi_harness_max_harnesses: int = 3  # Max harnesses to evaluate (limit for speed)

    # January 5, 2026: Parallel multi-harness evaluation
    # Feb 2026: Changed from 2 to 1 (sequential). With thread pool capped at 4 workers
    # (d983420d8), running 2 concurrent harnesses consumed half the pool, starving
    # other daemons. Sequential harness evaluation is fine with the reduced game counts.
    multi_harness_parallel: int = 1  # Number of concurrent harness evaluations

    # January 3, 2026 (Sprint 13 Session 4): Stuck evaluation recovery
    stuck_check_interval_seconds: float = 1800.0  # 30 minutes
    startup_scan_enabled: bool = True  # Scan for unevaluated models on startup
    startup_scan_canonical_priority: int = 75  # Priority for canonical models


class EvaluationDaemon(HandlerBase, EvaluationExecutorMixin):
    """Daemon that auto-evaluates models after training completes.

    December 27, 2025: Migrated to HandlerBase - inherits:
    - Automatic event subscription/unsubscription lifecycle
    - Standard health_check() implementation
    - Error counting and last_error tracking
    - get_metrics() and get_status() for DaemonManager
    """

    def __init__(self, config: EvaluationConfig | None = None):
        resolved_config = config or EvaluationConfig()
        super().__init__(name="EvaluationDaemon", config=resolved_config)
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
            "hysteresis_skips": 0,  # Session 17.24: Skips due to hysteresis
        }
        # Session 17.24 (Jan 2026): Hysteresis state tracking
        self._last_backpressure_release_time: float = 0.0  # Time of last release
        self._below_threshold_since: float = 0.0  # When queue dropped below release threshold
        # December 29, 2025: Retry queue for failed evaluations
        # Tuple: (model_path, board_type, num_players, attempts, next_retry_time)
        # March 2026: Added maxlen=200 to prevent unbounded growth during 7-day autonomous operation.
        # At max_attempts=3, this supports ~66 unique failed evaluations before oldest are evicted.
        self._retry_queue: deque[tuple[str, str, int, int, float]] = deque(maxlen=200)
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

        # January 3, 2026 (Sprint 13 Session 4): Persistent evaluation queue
        # Provides SQLite-backed persistence, stuck detection, and startup scan
        self._persistent_queue: PersistentEvaluationQueue | None = None
        self._stuck_check_task: asyncio.Task | None = None

        # January 7, 2026: Periodic unevaluated model scan (48h autonomous operation)
        # Runs every 5 minutes to catch models trained on cluster that weren't
        # triggered via TRAINING_COMPLETED event (event may not reach coordinator)
        # Reduced from 30 minutes to fix stale Elo ratings
        self._last_model_scan: float | None = None
        self._model_scan_interval_seconds: float = 300.0  # 5 minutes

        # January 2026: OOM recovery with adaptive batch size reduction
        # Maps config_key -> reduced parallel_games count (default 16 -> 8 -> 4 -> 2 -> 1)
        # This prevents infinite OOM retry loops by progressively reducing memory usage
        self._oom_parallel_games: dict[str, int] = {}
        self._oom_recovery_stats = {
            "oom_reductions": 0,
            "oom_recoveries": 0,
            "oom_exhausted": 0,
        }

        # Feb 2026: Track dispatched cluster evaluations for completion callback
        # Maps work_id -> (primary_request_id, sibling_request_ids) so WORK_COMPLETED
        # events can update ALL persistent queue entries (primary + siblings).
        # Previously only stored primary, leaving siblings stuck in "running" forever.
        self._dispatched_evaluations: dict[str, tuple[str, list[str]]] = {}

        # Feb 2026: Track work IDs already processed via polling to avoid duplicates
        # This bridges the cross-process gap between P2P orchestrator and master_loop
        self._polled_work_ids: set[str] = set()
        self._poll_completions_stats = {
            "polls": 0,
            "completions_found": 0,
            "errors": 0,
        }

    def _get_subscriptions(self) -> dict[Any, Callable]:
        """Return event subscriptions for HandlerBase.

        Returns:
            Dict mapping event types to handler methods.
        """
        subs: dict[Any, Callable] = {
            DataEventType.TRAINING_COMPLETED: self._on_training_complete,
            DataEventType.EVALUATION_REQUESTED: self._on_evaluation_requested,
        }
        # Feb 2026: Subscribe to WORK_COMPLETED/WORK_FAILED to track cluster evaluations
        if hasattr(DataEventType, "WORK_COMPLETED"):
            subs[DataEventType.WORK_COMPLETED] = self._on_work_completed
        if hasattr(DataEventType, "WORK_FAILED"):
            subs[DataEventType.WORK_FAILED] = self._on_work_failed
        if hasattr(DataEventType, "WORK_TIMEOUT"):
            subs[DataEventType.WORK_TIMEOUT] = self._on_work_timeout
        return subs

    async def start(self) -> bool:
        """Start the evaluation daemon.

        Returns:
            True if successfully started.
        """
        # Call parent start for event subscription
        # Note: parent start() returns None, not bool
        await super().start()
        if not self._running:
            return False

        # January 3, 2026 (Sprint 13 Session 4): Initialize persistent queue
        self._persistent_queue = get_evaluation_queue()

        # Start the evaluation worker and store task for proper cleanup
        self._worker_task = asyncio.create_task(self._evaluation_worker())

        # January 3, 2026: Start stuck evaluation check task
        self._stuck_check_task = asyncio.create_task(self._stuck_evaluation_check_loop())

        # January 3, 2026: Run startup scan for unevaluated models
        # Sprint 17.4: Use safe task creation for error handling
        if self.config.startup_scan_enabled:
            self._safe_create_task(
                self._startup_scan_for_unevaluated_models(),
                context="startup_scan_unevaluated",
            )

        logger.info(
            f"[EvaluationDaemon] Started. "
            f"Games per baseline: {self.config.games_per_baseline}, "
            f"Early stopping: {self.config.early_stopping_enabled}, "
            f"Startup scan: {self.config.startup_scan_enabled}"
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

        # January 3, 2026: Cancel stuck check task
        if self._stuck_check_task and not self._stuck_check_task.done():
            self._stuck_check_task.cancel()
            try:
                await self._stuck_check_task
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
        """Run periodic unevaluated model scan for 48h autonomous operation.

        January 7, 2026: Scans for unevaluated models every 30 minutes.
        This catches models trained on cluster nodes where the TRAINING_COMPLETED
        event didn't reach the coordinator (network issues, event drops, etc.).

        February 2026: Added cross-process completion polling. The P2P orchestrator
        runs in a separate process from master_loop, so WORK_COMPLETED events emitted
        on the P2P event bus never reach this daemon's event bus. This polling bridges
        the gap by querying the P2P work queue via HTTP for completed evaluations.

        December 29, 2025: Added to satisfy HandlerBase abstract requirement.
        The actual work is done by _evaluation_worker() processing the queue.

        March 2026: Added worker task death monitoring. If _worker_task has died
        (exception, cancellation), restart it so the daemon doesn't appear healthy
        while silently processing nothing.
        """
        import time

        current_time = time.time()

        # March 2026: Monitor worker task health — restart if it died silently
        if hasattr(self, '_worker_task') and self._worker_task is not None and self._worker_task.done():
            exc = self._worker_task.exception() if not self._worker_task.cancelled() else None
            if exc:
                logger.error(f"[EvaluationDaemon] Worker task died with: {exc}")
            else:
                logger.warning("[EvaluationDaemon] Worker task ended unexpectedly, restarting")
            self._worker_task = asyncio.create_task(self._evaluation_worker())

        # Feb 2026: Poll P2P work queue for completed evaluations (cross-process bridge)
        # This fixes the 97% evaluation failure rate caused by process isolation between
        # P2P orchestrator and master_loop - events can't cross process boundaries
        await self._poll_cluster_completions()

        # Check if it's time for a periodic model scan
        if self._last_model_scan is None or \
           (current_time - self._last_model_scan) > self._model_scan_interval_seconds:
            logger.info(
                "[EvaluationDaemon] Running periodic unevaluated model scan "
                f"(interval={self._model_scan_interval_seconds/60:.0f}min)"
            )
            await self._startup_scan_for_unevaluated_models()
            self._last_model_scan = current_time

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for DaemonManager health monitoring.

        December 2025: Added to fix missing status method (P0 gap).
        December 27, 2025: Enhanced with HandlerBase metrics.

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
            "poll_completions_stats": dict(self._poll_completions_stats),
            "dispatched_evaluations_pending": len(self._dispatched_evaluations),
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
        """Handle TRAINING_COMPLETE event.

        Sprint 15 (Jan 3, 2026): Added support for backlog evaluation sources.
        Events with source="backlog_*" are queued with lower priority.
        """
        try:
            # December 30, 2025: Use consolidated extraction helpers from HandlerBase
            metadata = self._get_payload(event)
            model_path = self._extract_model_path(metadata)
            board_type, num_players = self._extract_board_config(metadata)

            if not model_path:
                logger.warning("[EvaluationDaemon] No checkpoint_path/model_path in TRAINING_COMPLETE event")
                return

            # Sprint 15: Detect backlog evaluation source
            source = metadata.get("source", "training")
            is_backlog = source.startswith("backlog_")

            # Set priority: 0-50 for fresh training, 100-200 for backlog
            if is_backlog:
                priority = 150  # Lower priority for backlog models
            else:
                priority = 25  # Higher priority for fresh training

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
            # Session 17.24: Respect hysteresis cooldown before re-activation
            if queue_depth >= self.config.backpressure_threshold and not self._backpressure_active:
                if self._should_activate_backpressure():
                    self._emit_backpressure(queue_depth, activate=True)
                else:
                    self._backpressure_stats["hysteresis_skips"] += 1

            # Queue the evaluation with source and priority
            # Feb 23, 2026: Include trained_by so _ensure_model_local knows
            # which node to sync the candidate model from
            trained_by = metadata.get("trained_by", "")
            await self._evaluation_queue.put({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
                "source": source,
                "priority": priority,
                "trained_by": trained_by,
            })

            self._eval_stats.evaluations_triggered += 1
            source_info = f" (source={source}, priority={priority})" if is_backlog else ""
            logger.info(
                f"[EvaluationDaemon] Queued evaluation for {model_path} "
                f"({board_type}_{num_players}p), queue_depth={queue_depth + 1}{source_info}"
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[EvaluationDaemon] Invalid event data: {e}")
        except OSError as e:
            logger.error(f"[EvaluationDaemon] I/O error handling training complete: {e}")

    async def _on_evaluation_requested(self, event: Any) -> None:
        """Handle EVALUATION_REQUESTED event from model discovery daemons.

        January 3, 2026: Added to enable automated evaluation of discovered models.
        Sources include ModelDiscoveryDaemon, OWCModelEvaluationDaemon, StaleEvaluationDaemon.

        Expected payload:
            model_path: str - Path to the model file
            board_type: str - Board type (hex8, square8, etc.)
            num_players: int - Number of players (2, 3, 4)
            source: str - Source daemon (discovery, owc, stale)
            priority: int - Priority level (0=high, 1=normal, 2=low)
        """
        try:
            metadata = self._get_payload(event)
            model_path = metadata.get("model_path")
            board_type = metadata.get("board_type")
            num_players = metadata.get("num_players")
            source = metadata.get("source", "unknown")
            priority = metadata.get("priority", 1)

            if not model_path:
                logger.warning("[EvaluationDaemon] No model_path in EVALUATION_REQUESTED event")
                return

            if not board_type or not num_players:
                # Try to extract from model path filename
                from pathlib import Path
                model_name = Path(model_path).stem
                # Pattern: canonical_hex8_2p or similar
                parts = model_name.split("_")
                if len(parts) >= 2:
                    for part in parts:
                        if part.endswith("p") and part[:-1].isdigit():
                            num_players = int(part[:-1])
                        elif part in ("hex8", "square8", "square19", "hexagonal"):
                            board_type = part

            if not board_type or not num_players:
                logger.warning(
                    f"[EvaluationDaemon] Cannot determine config from EVALUATION_REQUESTED: {model_path}"
                )
                return

            # Deduplication checks (same as _on_training_complete)
            event_hash = self._compute_event_hash(model_path, board_type, num_players)
            if self._is_duplicate_event(event_hash):
                self._dedup_stats["content_hash_skips"] += 1
                logger.debug(f"[EvaluationDaemon] Skipping duplicate (content hash): {model_path}")
                return

            if self._is_in_cooldown(model_path):
                self._dedup_stats["cooldown_skips"] += 1
                logger.debug(f"[EvaluationDaemon] Skipping model in cooldown: {model_path}")
                return

            if model_path in self._active_evaluations:
                self._dedup_stats["concurrent_skips"] += 1
                logger.debug(f"[EvaluationDaemon] Skipping already-evaluating model: {model_path}")
                return

            # Backpressure check
            queue_depth = self._evaluation_queue.qsize()
            if queue_depth >= self.config.max_queue_depth:
                self._backpressure_stats["queue_full_rejections"] += 1
                logger.warning(
                    f"[EvaluationDaemon] Queue full ({queue_depth}), rejecting: {model_path}"
                )
                return

            # Session 17.24: Respect hysteresis cooldown before re-activation
            if queue_depth >= self.config.backpressure_threshold and not self._backpressure_active:
                if self._should_activate_backpressure():
                    self._emit_backpressure(queue_depth, activate=True)
                else:
                    self._backpressure_stats["hysteresis_skips"] += 1

            # Queue the evaluation
            await self._evaluation_queue.put({
                "model_path": model_path,
                "board_type": board_type,
                "num_players": num_players,
                "timestamp": time.time(),
                "source": source,
                "priority": priority,
            })

            self._eval_stats.evaluations_triggered += 1
            logger.info(
                f"[EvaluationDaemon] Queued evaluation (source={source}) for {model_path} "
                f"({board_type}_{num_players}p), queue_depth={queue_depth + 1}"
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"[EvaluationDaemon] Invalid EVALUATION_REQUESTED data: {e}")
        except OSError as e:
            logger.error(f"[EvaluationDaemon] I/O error handling evaluation requested: {e}")

    async def _on_work_completed(self, event: Any) -> None:
        """Handle WORK_COMPLETED event from cluster work queue.

        Feb 2026: Closes the loop for evaluations dispatched to cluster nodes.
        Without this, dispatched evaluations stay in RUNNING state until they
        time out as STUCK (was causing 88% evaluation failure rate).
        """
        from app.coordination.event_router import get_event_payload

        payload = get_event_payload(event)
        work_id = payload.get("work_id", "")
        work_type = payload.get("work_type", "")

        # Only handle evaluation work items we dispatched
        if work_type != "evaluation" or work_id not in self._dispatched_evaluations:
            return

        persistent_request_id, sibling_ids = self._dispatched_evaluations.pop(work_id)
        result = payload.get("result", {})
        estimated_elo = result.get("estimated_elo", result.get("best_elo", 0.0))

        logger.info(
            f"[EvaluationDaemon] Cluster evaluation completed: work_id={work_id}, "
            f"elo={estimated_elo:.0f}"
        )

        # Update persistent queue - complete primary AND all siblings
        if self._persistent_queue and persistent_request_id:
            self._persistent_queue.complete(persistent_request_id, elo=estimated_elo)
            for sid in sibling_ids:
                self._persistent_queue.complete(sid, elo=estimated_elo)

        self._eval_stats.evaluations_completed += 1

        # Emit EVALUATION_COMPLETED so the promotion pipeline can proceed
        board_type = payload.get("board_type", "")
        num_players = payload.get("num_players", 2)
        model_path = result.get("model_path", payload.get("config", {}).get("candidate_model", ""))
        if board_type and num_players and model_path:
            await self._emit_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                result=result,
            )

    async def _on_work_failed(self, event: Any) -> None:
        """Handle WORK_FAILED event from cluster work queue.

        Feb 2026: Marks dispatched evaluations as failed in persistent queue.
        """
        from app.coordination.event_router import get_event_payload

        payload = get_event_payload(event)
        work_id = payload.get("work_id", "")
        work_type = payload.get("work_type", "")

        if work_type != "evaluation" or work_id not in self._dispatched_evaluations:
            return

        persistent_request_id, sibling_ids = self._dispatched_evaluations.pop(work_id)
        error = payload.get("error", "cluster_work_failed")

        logger.warning(
            f"[EvaluationDaemon] Cluster evaluation failed: work_id={work_id}, "
            f"error={error}"
        )

        # Fail primary AND all siblings
        if self._persistent_queue and persistent_request_id:
            self._persistent_queue.fail(persistent_request_id, error)
            for sid in sibling_ids:
                self._persistent_queue.fail(sid, error)

        self._eval_stats.evaluations_failed += 1

    async def _on_work_timeout(self, event: Any) -> None:
        """Handle WORK_TIMEOUT event from cluster work queue.

        Timed-out dispatched evaluations must be removed from in-memory tracking
        or they accumulate until process restart.
        """
        from app.coordination.event_router import get_event_payload

        payload = get_event_payload(event)
        work_id = payload.get("work_id", "")
        work_type = payload.get("work_type", "")

        if work_type != "evaluation" or work_id not in self._dispatched_evaluations:
            return

        persistent_request_id, sibling_ids = self._dispatched_evaluations.pop(work_id)
        error = payload.get("error", "cluster_work_timeout")

        logger.warning(
            f"[EvaluationDaemon] Cluster evaluation timed out: work_id={work_id}, "
            f"error={error}"
        )

        if self._persistent_queue and persistent_request_id:
            self._persistent_queue.fail(persistent_request_id, error)
            for sid in sibling_ids:
                self._persistent_queue.fail(sid, error)

        self._eval_stats.evaluations_failed += 1

    async def _poll_cluster_completions(self) -> None:
        """Poll P2P work queue for completed evaluation work items.

        February 2026: Bridges the cross-process gap between P2P orchestrator
        and master_loop. The P2P orchestrator runs in a separate Python process,
        so events emitted on its event bus never reach this daemon's event bus.
        This caused 97% of dispatched evaluations to time out as STUCK.

        Approach:
        1. Query P2P leader via HTTP for recently completed evaluation items
        2. Match against dispatched evaluations (in-memory) or RUNNING persistent
           queue entries (survives restarts)
        3. Process completions the same way as _on_work_completed
        """
        import aiohttp

        self._poll_completions_stats["polls"] += 1

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                url = "http://localhost:8770/work/history?status=completed&limit=50"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()

            history = data.get("history", [])
            if not history:
                return

            completions_processed = 0
            for item in history:
                work_id = item.get("work_id", "")
                work_type = item.get("work_type", "")

                # Only care about evaluation/gauntlet work.
                # Feb 28, 2026: Removed "tournament" — tournaments are NOT evaluations.
                # Matching tournament completions against evaluation requests caused
                # phantom 0-game results (91.5% of evaluations had elo=0.0).
                if work_type not in ("evaluation", "gauntlet"):
                    continue

                # Skip already-processed items
                if work_id in self._polled_work_ids:
                    continue

                config = item.get("config", {})
                result = item.get("result", {})
                board_type = config.get("board_type", "")
                num_players = config.get("num_players", 0)
                model_path = config.get("candidate_model", "")
                estimated_elo = result.get("estimated_elo", result.get("best_elo", 0.0))

                # Feb 28, 2026: Compute proper Elo from opponent_results if the
                # default 1500.0/0.0 was returned. The gauntlet executor sends
                # opponent win/loss data but doesn't compute Elo itself.
                if (not estimated_elo or estimated_elo == 1500.0) and board_type and num_players and model_path:
                    computed_elo = await self._compute_elo_from_gauntlet(
                        model_path=model_path,
                        board_type=board_type,
                        num_players=int(num_players),
                        result=result,
                    )
                    if computed_elo is not None:
                        estimated_elo = computed_elo
                        result["estimated_elo"] = estimated_elo
                        result["best_elo"] = estimated_elo

                # Mark as seen regardless of whether we can match it
                self._polled_work_ids.add(work_id)

                # Strategy 1: Match against in-memory dispatched evaluations
                if work_id in self._dispatched_evaluations:
                    persistent_request_id, sibling_ids = self._dispatched_evaluations.pop(work_id)
                    logger.info(
                        f"[EvaluationDaemon] Poll: matched dispatched evaluation "
                        f"work_id={work_id}, elo={estimated_elo:.0f}"
                        f"{f' (+{len(sibling_ids)} siblings)' if sibling_ids else ''}"
                    )
                    if self._persistent_queue and persistent_request_id:
                        self._persistent_queue.complete(persistent_request_id, elo=estimated_elo)
                        for sid in sibling_ids:
                            self._persistent_queue.complete(sid, elo=estimated_elo)
                    self._eval_stats.evaluations_completed += 1
                    completions_processed += 1

                    if board_type and num_players and model_path:
                        await self._emit_evaluation_completed(
                            model_path=model_path,
                            board_type=board_type,
                            num_players=num_players,
                            result=result,
                        )
                    continue

                # Strategy 2: Match against RUNNING persistent queue entries
                # This handles the case where master_loop restarted and lost
                # the in-memory _dispatched_evaluations mapping
                if self._persistent_queue and board_type and num_players:
                    config_key = f"{board_type}_{num_players}p"
                    matched = await self._match_running_evaluation(
                        config_key, model_path, estimated_elo, result
                    )
                    if matched:
                        completions_processed += 1

            if completions_processed > 0:
                self._poll_completions_stats["completions_found"] += completions_processed
                logger.info(
                    f"[EvaluationDaemon] Poll: processed {completions_processed} "
                    f"completed evaluations from P2P work queue"
                )

            # Prevent unbounded growth of polled_work_ids
            if len(self._polled_work_ids) > 500:
                # Keep only most recent 200
                excess = len(self._polled_work_ids) - 200
                for _ in range(excess):
                    self._polled_work_ids.pop()

        except (OSError, aiohttp.ClientError, asyncio.TimeoutError):
            # P2P not reachable - normal during startup or network issues
            self._poll_completions_stats["errors"] += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EvaluationDaemon] Poll error: {e}")
            self._poll_completions_stats["errors"] += 1

    async def _match_running_evaluation(
        self,
        config_key: str,
        model_path: str,
        estimated_elo: float,
        result: dict,
    ) -> bool:
        """Match a completed P2P work item against RUNNING persistent queue entries.

        February 2026: Handles the case where master_loop restarted and the
        in-memory _dispatched_evaluations mapping was lost. Matches by config_key
        and model_path against RUNNING entries in the persistent queue.

        Returns:
            True if a match was found and processed.
        """
        if not self._persistent_queue:
            return False

        try:

            # Query RUNNING evaluations for this config from persistent queue
            rows = await asyncio.to_thread(
                self._query_running_evaluations, config_key
            )

            for row in rows:
                request_id = row["request_id"]
                row_model = row["model_path"]

                # Feb 28, 2026: Require model_path match to prevent phantom
                # completions. Previously, empty model_path would match ANY
                # running evaluation for the same config_key.
                if not model_path or not row_model:
                    continue  # Skip if either side has no model info
                if model_path != row_model:
                    continue

                logger.info(
                    f"[EvaluationDaemon] Poll: matched RUNNING persistent request "
                    f"{request_id} ({config_key}), elo={estimated_elo:.0f}"
                )
                self._persistent_queue.complete(request_id, elo=estimated_elo)
                self._eval_stats.evaluations_completed += 1

                board_type = row["board_type"]
                num_players = row["num_players"]
                effective_model = model_path or row_model
                if board_type and num_players and effective_model:
                    await self._emit_evaluation_completed(
                        model_path=effective_model,
                        board_type=board_type,
                        num_players=num_players,
                        result=result,
                    )
                return True

        except Exception as e:  # noqa: BLE001
            # Critical path: failure here means completed P2P evaluations are not
            # matched to persistent queue entries, causing lost evaluation results.
            logger.warning(f"[EvaluationDaemon] Match running evaluation error: {e}", exc_info=True)

        return False

    def _query_running_evaluations(self, config_key: str) -> list:
        """Query RUNNING evaluations from persistent queue (blocking, run in thread)."""

        with self._persistent_queue._lock:
            with self._persistent_queue._get_connection() as conn:
                conn.row_factory = __import__("sqlite3").Row
                return conn.execute(
                    """
                    SELECT * FROM evaluation_requests
                    WHERE status = ? AND config_key = ?
                    ORDER BY started_at DESC
                    LIMIT 5
                    """,
                    (RequestStatus.RUNNING, config_key),
                ).fetchall()
























    def get_stats(self) -> dict:
        """Get daemon statistics."""
        stats = {
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
            # January 2026: OOM recovery stats
            "oom_reductions": self._oom_recovery_stats["oom_reductions"],
            "oom_recoveries": self._oom_recovery_stats["oom_recoveries"],
            "oom_exhausted": self._oom_recovery_stats["oom_exhausted"],
            "oom_active_configs": len(self._oom_parallel_games),
        }

        # January 3, 2026: Add persistent queue stats
        if self._persistent_queue:
            queue_status = self._persistent_queue.get_queue_status()
            stats["persistent_queue"] = queue_status

        return stats

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health (December 2025: CoordinatorProtocol compliance).

        December 27, 2025: Extends HandlerBase health_check with
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

    # =========================================================================
    # January 3, 2026 (Sprint 13 Session 4): Persistent Queue Integration
    # =========================================================================

    async def _startup_scan_for_unevaluated_models(self) -> None:
        """Scan for models without Elo ratings on startup.

        January 3, 2026: Ensures all canonical models get evaluated.
        February 2026: Also scans candidate_*.pth models from training pipeline.
        This catches models that were trained but never evaluated due to
        daemon restarts, event drops, or system failures.
        """

        logger.info("[EvaluationDaemon] Starting scan for unevaluated models...")
        scanned = 0
        queued = 0

        try:
            models_dir = Path("models")
            if not models_dir.exists():
                logger.warning("[EvaluationDaemon] Models directory not found")
                return

            # Jan 2026: Track seen actual paths to avoid duplicate evaluation
            # (ringrift_best_* are symlinks to canonical_* models)
            seen_actual_paths: set[str] = set()

            # Mar 2026: Only scan candidate models on startup. Canonical/ringrift_best
            # models already have Elo ratings from their promotion gauntlet. Scanning
            # all 116+ models × 4 harness types = 415+ queue items that block candidate
            # evaluations for days on MPS hardware.
            for pattern in ["candidate_*.pth"]:
                for model_path in models_dir.glob(pattern):
                    # Resolve symlinks to get actual model path
                    actual_path = model_path.resolve() if model_path.is_symlink() else model_path

                    # Skip duplicates (ringrift_best_* symlinks to canonical_*)
                    if str(actual_path) in seen_actual_paths:
                        logger.debug(
                            f"[EvaluationDaemon] Skipping duplicate (via symlink): {model_path.name}"
                        )
                        continue
                    seen_actual_paths.add(str(actual_path))

                    scanned += 1

                    # Extract board_type and num_players from filename
                    # Format: {prefix}_{board_type}_{n}p.pth (canonical, candidate, or ringrift_best)
                    stem = model_path.stem  # e.g., "candidate_hex8_2p" or "canonical_hex8_2p"
                    parts = stem.split("_")

                    # Determine the prefix length: "canonical" = 1 part, "ringrift_best" = 2 parts
                    if stem.startswith("ringrift_best_"):
                        prefix_parts = 2  # "ringrift_best" is 2 parts
                    else:
                        prefix_parts = 1  # "canonical" is 1 part

                    if len(parts) < prefix_parts + 2:
                        logger.debug(f"[EvaluationDaemon] Skipping unrecognized filename: {model_path}")
                        continue

                    board_type = parts[prefix_parts]
                    players_part = parts[prefix_parts + 1]

                    # Handle architectures like canonical_hex8_2p_v5heavy.pth
                    if not players_part.endswith("p"):
                        continue

                    try:
                        num_players = int(players_part[:-1])
                    except ValueError:
                        continue

                    # January 2026: Iterate over compatible harnesses instead of just
                    # checking if model has ANY Elo rating. This enables per-harness
                    # Elo tracking for multi-harness evaluation.
                    try:
                        from app.ai.harness.harness_registry import get_harnesses_for_model_and_players
                        from app.ai.harness.base_harness import ModelType

                        # Assume NN model type for canonical/ringrift_best models
                        compatible_harnesses = get_harnesses_for_model_and_players(
                            model_type=ModelType.NEURAL_NET,
                            num_players=num_players,
                        )
                    except ImportError:
                        # Fallback to simple check if harness registry not available
                        if self._has_elo_rating(str(model_path)):
                            logger.debug(
                                f"[EvaluationDaemon] Already has Elo: {model_path.name}"
                            )
                            continue
                        compatible_harnesses = []  # Will skip harness loop below

                        # Queue without harness_type (legacy behavior)
                        if self._persistent_queue:
                            request_id = self._persistent_queue.add_request(
                                model_path=str(model_path),
                                board_type=board_type,
                                num_players=num_players,
                                priority=self.config.startup_scan_canonical_priority,
                                source="startup_scan",
                            )
                            if request_id:
                                queued += 1
                                logger.info(
                                    f"[EvaluationDaemon] Queued unevaluated model: {model_path.name} "
                                    f"({board_type}_{num_players}p)"
                                )
                        continue  # Skip harness loop

                    # Queue evaluation for each harness that needs it
                    for harness_type in compatible_harnesses:
                        harness_name = harness_type.value

                        # Check if this specific (model, harness) combo needs evaluation
                        if not self._needs_harness_evaluation(str(model_path), harness_name):
                            logger.debug(
                                f"[EvaluationDaemon] Already evaluated: {model_path.name} "
                                f"under {harness_name}"
                            )
                            continue

                        # Add to persistent queue with harness_type
                        if self._persistent_queue:
                            request_id = self._persistent_queue.add_request(
                                model_path=str(model_path),
                                board_type=board_type,
                                num_players=num_players,
                                priority=self.config.startup_scan_canonical_priority,
                                source="startup_scan",
                                harness_type=harness_name,
                            )

                            if request_id:
                                queued += 1
                                logger.info(
                                    f"[EvaluationDaemon] Queued for {harness_name}: "
                                    f"{model_path.name} ({board_type}_{num_players}p)"
                                )

            logger.info(
                f"[EvaluationDaemon] Startup scan complete: "
                f"{scanned} models scanned, {queued} queued for evaluation"
            )

        except Exception as e:
            # Critical: startup scan failure means no canonical models get queued
            # for evaluation, silently stalling the entire eval pipeline on restart.
            logger.error(f"[EvaluationDaemon] Startup scan failed: {e}", exc_info=True)

    def _has_elo_rating(self, model_path: str) -> bool:
        """Check if a model has an Elo rating in EloService.

        January 3, 2026: Used by startup scan to skip already-rated models.

        Args:
            model_path: Path to the model file

        Returns:
            True if model has an Elo rating, False otherwise
        """
        try:
            from app.training.elo_service import get_elo_service
            from pathlib import Path

            elo_service = get_elo_service()
            model_name = Path(model_path).stem

            # Try to get rating — need board_type/num_players for the new API.
            # These checks scan models without config context, so try all configs.
            _CONFIGS = [
                ("hex8", 2), ("hex8", 3), ("hex8", 4),
                ("square8", 2), ("square8", 3), ("square8", 4),
                ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
                ("square19", 2), ("square19", 3), ("square19", 4),
            ]
            for bt, np_int in _CONFIGS:
                try:
                    rating = elo_service.get_rating(model_name, bt, np_int)
                    if rating is not None:
                        return True
                except Exception as exc:
                    logger.debug(
                        "[EvaluationDaemon] Skipping Elo lookup for model=%s config=%s_%sp: %s",
                        model_name,
                        bt,
                        np_int,
                        exc,
                    )
            return False

        except ImportError:
            logger.debug("[EvaluationDaemon] EloService not available for Elo check")
            return False
        except Exception as e:
            logger.debug(f"[EvaluationDaemon] Elo check failed: {e}")
            return False

    def _needs_harness_evaluation(self, model_path: str, harness_type: str) -> bool:
        """Check if model needs evaluation under a specific harness.

        January 2026: Enables per-harness Elo tracking. A model may have been
        evaluated under gumbel_mcts but not under minimax, so we need to check
        each harness separately.

        Args:
            model_path: Path to the model file
            harness_type: The harness/AI type (e.g., "gumbel_mcts", "minimax")

        Returns:
            True if model needs evaluation under this harness, False otherwise
        """
        try:
            from app.training.composite_participant import make_composite_participant_id
            from app.training.elo_service import get_elo_service
            from pathlib import Path

            model_name = Path(model_path).stem
            composite_id = make_composite_participant_id(
                nn_id=model_name,
                ai_type=harness_type,
                config=None,  # Use default config for harness
            )

            elo_service = get_elo_service()
            _CONFIGS = [
                ("hex8", 2), ("hex8", 3), ("hex8", 4),
                ("square8", 2), ("square8", 3), ("square8", 4),
                ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
                ("square19", 2), ("square19", 3), ("square19", 4),
            ]
            for bt, np_int in _CONFIGS:
                try:
                    rating = elo_service.get_rating(composite_id, bt, np_int)
                    if rating is not None:
                        return False  # Has rating = doesn't need eval
                except Exception as exc:
                    logger.debug(
                        "[EvaluationDaemon] Skipping harness Elo lookup for model=%s harness=%s config=%s_%sp: %s",
                        model_name,
                        harness_type,
                        bt,
                        np_int,
                        exc,
                    )
            return True  # No rating found = needs eval

        except ImportError:
            logger.error(
                f"[EvaluationDaemon] Dependencies not available for harness check: {harness_type}"
            )
            return False  # Fail closed: don't assume needs-eval when check failed
        except Exception as e:
            logger.error(f"[EvaluationDaemon] Harness Elo check failed: {e}")
            return False  # Fail closed: check failure must not default to allow/needs-eval

    async def _stuck_evaluation_check_loop(self) -> None:
        """Periodically check for and recover stuck evaluations.

        January 3, 2026: Runs every stuck_check_interval_seconds to detect
        RUNNING evaluations that have exceeded their timeout.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.stuck_check_interval_seconds)

                if not self._persistent_queue:
                    continue

                # Get stuck evaluations
                stuck = self._persistent_queue.get_stuck_evaluations()

                if not stuck:
                    continue

                logger.warning(
                    f"[EvaluationDaemon] Found {len(stuck)} stuck evaluations"
                )

                # Process each stuck evaluation
                for request in stuck:
                    if request.attempts < request.max_attempts:
                        # Reset to pending for retry
                        self._persistent_queue.reset_stuck(request.request_id)

                        # Emit recovery event
                        safe_emit_event(
                            DataEventType.EVALUATION_RECOVERED,
                            {
                                "request_id": request.request_id,
                                "model_path": request.model_path,
                                "config_key": request.config_key,
                                "attempts": request.attempts,
                                "stuck_duration_seconds": time.time() - request.started_at,
                            },
                        )

                        logger.info(
                            f"[EvaluationDaemon] Recovered stuck evaluation: {request.model_path} "
                            f"(attempt {request.attempts}/{request.max_attempts})"
                        )
                    else:
                        # Max retries exceeded, mark as failed
                        self._persistent_queue.fail(
                            request.request_id,
                            f"Stuck timeout exceeded after {request.max_attempts} attempts",
                        )

                        # Emit stuck event (not recovered)
                        safe_emit_event(
                            DataEventType.EVALUATION_STUCK,
                            {
                                "request_id": request.request_id,
                                "model_path": request.model_path,
                                "config_key": request.config_key,
                                "attempts": request.attempts,
                                "stuck_duration_seconds": time.time() - request.started_at,
                            },
                        )

                        logger.error(
                            f"[EvaluationDaemon] Evaluation permanently stuck: {request.model_path}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EvaluationDaemon] Stuck check error: {e}")
                await asyncio.sleep(60)  # Back off on error

    def _track_in_persistent_queue(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        priority: int = 50,
        source: str = "training",
    ) -> str | None:
        """Track an evaluation request in the persistent queue.

        January 3, 2026: Called when adding to the in-memory queue.
        This provides persistence and deduplication via SQLite.

        Args:
            model_path: Path to the model file
            board_type: Board type
            num_players: Number of players
            priority: Priority (higher = sooner)
            source: Source of the request

        Returns:
            Request ID if added, None if duplicate
        """
        if not self._persistent_queue:
            return None

        return self._persistent_queue.add_request(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
            priority=priority,
            source=source,
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
