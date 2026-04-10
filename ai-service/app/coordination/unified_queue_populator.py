"""Unified Queue Populator - Automatic work queue population (December 2025).

This module consolidates queue_populator.py and queue_populator_daemon.py into
a single unified implementation that provides:

1. Elo-based target tracking with velocity calculations
2. Async daemon lifecycle for background operation
3. P2P cluster health integration
4. Curriculum-weighted prioritization
5. Backpressure-aware population
6. SelfplayScheduler integration

Work distribution (default):
- 60% selfplay (data generation)
- 30% training (model improvement)
- 10% tournament (Elo measurement)

Usage:
    # As daemon (recommended for production)
    daemon = UnifiedQueuePopulatorDaemon()
    await daemon.start()

    # Synchronous usage
    populator = get_queue_populator()
    added = populator.populate()

December 2025: Created as consolidation of queue_populator.py and
queue_populator_daemon.py. Saves ~500 LOC through deduplication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

# Canonical types (December 2025 consolidation)
from app.coordination.types import BackpressureLevel, BoardType
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.event_handler_utils import (
    extract_config_key,
)

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult
    from app.coordination.selfplay_scheduler import SelfplayScheduler
    from app.coordination.work_queue import WorkQueue

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

from app.coordination.queue_strategies.common import (
    BOARD_CONFIGS,
    DEFAULT_CURRICULUM_WEIGHTS,
    EXPLORATION_CONFIGS_PER_CYCLE,
    EXPLORATION_STALE_THRESHOLD_HOURS,
    LARGE_BOARDS,
    MINIMUM_EXPLORATION_GAMES,
)
from app.coordination.queue_strategies.population_health import QueuePopulationHealthMixin
from app.coordination.queue_strategies.population_state import QueuePopulationStateMixin
from app.coordination.queue_strategies.population_work import QueuePopulationWorkMixin

# Note: BoardType is now imported from app.coordination.types (December 2025)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QueuePopulatorConfig:
    """Unified configuration for queue populator.

    Combines settings from both queue_populator.py (Elo-based) and
    queue_populator_daemon.py (daemon-based).
    """

    # === General Settings ===
    enabled: bool = True

    # Minimum queue depth to maintain
    # December 29, 2025: Increased from 50 to 200 for steadier queue state
    # January 30, 2026: Reduced from 200 to 100 to reduce backlog (2,686 pending)
    min_queue_depth: int = 100

    # Maximum pending items before stopping generation (unused legacy setting)
    max_pending_items: int = 50

    # === Per-Type Pending Limits (January 30, 2026) ===
    # Prevent backlog accumulation by limiting pending items per work type
    # These limits are checked before adding new items to prevent runaway queues
    # Feb 27, 2026: Reduced from 500/300/200 — only 12 nodes alive, queue was 3,000+ deep
    max_pending_selfplay: int = 100
    max_pending_training: int = 50
    max_pending_tournament: int = 50
    # Mar 5, 2026: Set to 0 — no handler exists in p2p_orchestrator for this work
    # type, so items are permanently stuck. Stop creating new ones.
    max_pending_hyperparam_sweep: int = 0

    # Target queue depth to aim for (queue will fill to this level)
    # December 29, 2025: Added to reduce queue variance from 2,170% to <50%
    # January 30, 2026: Reduced from 300 to 150 to reduce backlog (2,686 pending)
    target_queue_depth: int = 150

    # Maximum items to add per populate cycle (prevents burst releases)
    # December 29, 2025: Added to prevent queue variance spikes
    max_batch_per_cycle: int = 100

    # Check/scan interval (reduced from 60s for faster job allocation)
    # Dec 30, 2025: Reduced from 10s to 5s for faster queue recovery after P2P restarts
    check_interval_seconds: int = 5

    # === Elo Targets ===
    target_elo: float = 2000.0

    # === Work Distribution (must sum to 1.0) ===
    selfplay_ratio: float = 0.60
    training_ratio: float = 0.30
    tournament_ratio: float = 0.10

    # === Board/Player Configuration ===
    board_types: list[str] = field(default_factory=lambda: [
        "square8", "square19", "hex8", "hexagonal"
    ])
    player_counts: list[int] = field(default_factory=lambda: [2, 3, 4])

    # === Selfplay Settings ===
    selfplay_games_per_item: int = 50
    # Feb 2026: Raised from 50 to 75. At priority 50, selfplay was starved by
    # training(100), gauntlet(85), and tournament(80) — no games produced for days.
    selfplay_priority: int = 75
    selfplay_timeout_seconds: float = 3600.0

    # === Training Settings ===
    training_priority: int = 100
    min_games_for_training: int = 100  # Dec 27, 2025: Lowered from 300 to accelerate training

    # === Tournament Settings ===
    tournament_games: int = 50
    tournament_priority: int = 80

    # === Export Settings ===
    export_priority: int = 70
    export_timeout_seconds: float = 3600.0

    # === Validation Settings ===
    validation_priority: int = 60
    validation_timeout_seconds: float = 1800.0

    # === Cluster-Aware Settings ===
    min_idle_nodes_to_populate: int = 1
    target_games_per_config: int = 10000
    data_gap_priority_boost: int = 20

    # === Trickle Mode Settings (Phase 15.1.2 - Dec 2025) ===
    # Trickle mode ensures work queue never completely starves under backpressure.
    # Even at CRITICAL/STOP backpressure, we add trickle_min_items to prevent
    # the pipeline from halting entirely.
    # Dec 31, 2025: Increased from 2 to 10 to better utilize cluster capacity
    # during backpressure events. With 40+ nodes, 10 items/cycle keeps pipeline moving.
    # January 2026 - Phase 3 Task 5: Dynamic trickle mode scales with cluster size.
    # Session 17.31 (Jan 5, 2026): Increased min from 10 to 50 to prevent GPU starvation
    trickle_mode_enabled: bool = True
    trickle_min_items: int = 50  # Minimum items to add even under max backpressure
    trickle_dynamic_scaling: bool = True  # Scale trickle count based on active nodes
    trickle_max_items: int = 100  # Maximum items even for very large clusters

    # === Backoff Settings (January 14, 2026) ===
    # Exponential backoff when queue is at hard limit prevents tight loops
    backoff_enabled: bool = True
    backoff_initial_seconds: float = 1.0  # Initial backoff duration
    backoff_max_seconds: float = 60.0  # Maximum backoff duration
    backoff_multiplier: float = 2.0  # Exponential growth factor
    backoff_jitter: float = 0.1  # Random jitter factor (0.1 = ±10%)

    # === Health Logging Settings (January 14, 2026) ===
    # Periodic health logging during backpressure to maintain visibility
    health_log_interval_seconds: float = 30.0  # Log health every N seconds during backpressure
    health_log_always: bool = False  # Log health even during normal operation

    # === Partition Detection Settings (January 14, 2026) ===
    # Detect cluster partition when queue drain rate drops to zero
    partition_detection_enabled: bool = True
    partition_drain_window_seconds: float = 120.0  # Time window to measure drain rate
    partition_min_completions: int = 1  # Minimum completions in window to be "healthy"
    partition_alert_threshold: int = 3  # Consecutive windows with no drain = partition alert

    # === Circuit Breaker Settings (January 14, 2026) ===
    # Circuit breaker to prevent overwhelming a failing queue
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_reset_timeout_seconds: float = 30.0  # Time before half-open state
    circuit_breaker_half_open_successes: int = 2  # Successes to close circuit


# =============================================================================
# State Tracking
# =============================================================================

@dataclass
class ConfigTarget:
    """Unified target state for a board/player configuration.

    Combines Elo tracking (from queue_populator.py) with data state
    tracking (from queue_populator_daemon.py).
    """

    board_type: str
    num_players: int

    # === Elo Tracking ===
    target_elo: float = 2000.0
    current_best_elo: float = 1500.0
    best_model_id: str | None = None

    # === Game/Training Counts ===
    games_played: int = 0
    games_since_last_export: int = 0
    training_runs: int = 0
    total_samples: int = 0

    # === Timestamps ===
    last_updated: float = field(default_factory=time.time)
    last_game_time: float = 0.0
    last_export_time: float = 0.0

    # === Pending Work ===
    pending_selfplay_count: int = 0
    pending_export: bool = False

    # === Prioritization ===
    curriculum_weight: float = 1.0

    # === Elo History for Velocity Tracking ===
    elo_history: list[tuple[float, float]] = field(default_factory=list)
    _previous_velocity: float = 0.0

    @property
    def target_met(self) -> bool:
        return self.current_best_elo >= self.target_elo

    @property
    def elo_gap(self) -> float:
        return max(0, self.target_elo - self.current_best_elo)

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @property
    def elo_velocity(self) -> float:
        """Calculate Elo velocity in points per day using linear regression."""
        if len(self.elo_history) < 2:
            return 0.0

        # Filter to last 7 days
        now = time.time()
        week_ago = now - (7 * 24 * 3600)
        recent = [(t, e) for t, e in self.elo_history if t >= week_ago]

        if len(recent) < 2:
            return 0.0

        # Simple linear regression for velocity
        times = [t for t, _ in recent]
        elos = [e for _, e in recent]

        n = len(times)
        sum_t = sum(times)
        sum_e = sum(elos)
        sum_te = sum(t * e for t, e in recent)
        sum_t2 = sum(t * t for t in times)

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0

        # Slope is Elo per second, convert to per day
        slope = (n * sum_te - sum_t * sum_e) / denom
        return slope * 86400

    @property
    def days_to_target(self) -> float | None:
        """Estimate days to reach target at current velocity."""
        if self.target_met:
            return 0.0

        velocity = self.elo_velocity
        if velocity <= 0:
            return None

        return self.elo_gap / velocity

    def record_elo(self, elo: float, timestamp: float | None = None) -> None:
        """Record an Elo measurement for velocity tracking.

        Emits ELO_VELOCITY_CHANGED event if velocity changes significantly.
        """
        ts = timestamp or time.time()
        self.elo_history.append((ts, elo))

        # Keep only last 30 days of history
        cutoff = ts - (30 * 24 * 3600)
        self.elo_history = [(t, e) for t, e in self.elo_history if t >= cutoff]

        # Check for significant velocity change (>10 Elo/day)
        new_velocity = self.elo_velocity
        velocity_change = abs(new_velocity - self._previous_velocity)

        if velocity_change > 10.0:
            if new_velocity > self._previous_velocity + 5:
                trend = "accelerating"
            elif new_velocity < self._previous_velocity - 5:
                trend = "decelerating"
            else:
                trend = "stable"

            # Emit event asynchronously
            try:
                from app.coordination.event_router import emit_elo_velocity_changed

                async def _emit():
                    await emit_elo_velocity_changed(
                        config_key=self.config_key,
                        velocity=new_velocity,
                        previous_velocity=self._previous_velocity,
                        trend=trend,
                    )

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_emit())
                except RuntimeError:
                    pass  # No running loop, skip emission

                logger.debug(
                    f"[ConfigTarget] Velocity changed for {self.config_key}: "
                    f"{self._previous_velocity:.1f} → {new_velocity:.1f} Elo/day ({trend})"
                )
            except ImportError:
                pass

            self._previous_velocity = new_velocity


# =============================================================================
# Unified Queue Populator
# =============================================================================

class UnifiedQueuePopulator(
    QueuePopulationStateMixin,
    QueuePopulationHealthMixin,
    QueuePopulationWorkMixin,
):
    """Unified work queue populator with Elo tracking and daemon support.

    This class consolidates the core business logic from queue_populator.py
    with the daemon lifecycle from queue_populator_daemon.py.
    """

    def __init__(
        self,
        config: QueuePopulatorConfig | None = None,
        work_queue: Optional["WorkQueue"] = None,
        elo_db_path: str | None = None,
        selfplay_scheduler: Optional["SelfplayScheduler"] = None,
    ):
        self.config = config or QueuePopulatorConfig()
        self._work_queue = work_queue
        self._elo_db_path = elo_db_path
        self._selfplay_scheduler = selfplay_scheduler

        # Configuration targets
        self._targets: dict[str, ConfigTarget] = {}

        # Scale queue depth to cluster size
        self._scale_queue_depth_to_cluster()

        # Initialize targets and load existing Elo
        self._init_targets()
        self._load_existing_elo()
        self._load_curriculum_weights()
        # Mar 2026: Deferred from __init__ to avoid blocking event loop for 6+ min.
        # _load_game_counts() scans 50-100+ databases via GameDiscovery. Now called
        # lazily in ensure_game_counts_loaded() before first populate().
        self._game_counts_loaded = False

        # Work tracking
        self._queued_work_ids: set[str] = set()
        self._queued_work_tracked_at: dict[str, float] = {}
        self._last_populate_time: float = 0

        # P2P health tracking (December 2025)
        self._dead_nodes: set[str] = set()
        self._cluster_health_factor: float = 1.0

        # Event subscriptions
        self._event_subscriptions: list[Callable] = []

        # Backpressure event tracking (January 2026)
        # Tracks last backpressure level for hysteresis-based event emission
        self._last_backpressure_level: BackpressureLevel = BackpressureLevel.NONE

        # Queue exhaustion event tracking (January 2026 - Phase 3 Task 4)
        # Tracks if queue was previously exhausted for hysteresis-based event emission
        self._was_queue_exhausted: bool = False

        # === Backoff State (January 14, 2026) ===
        # Exponential backoff to prevent tight loops when queue is at hard limit
        self._backoff_current_seconds: float = 0.0
        self._backoff_until: float = 0.0  # Unix timestamp when backoff expires
        self._consecutive_hard_limit_hits: int = 0

        # === Health Logging State (January 14, 2026) ===
        self._last_health_log_time: float = 0.0
        self._backpressure_start_time: float = 0.0
        self._total_backpressure_duration: float = 0.0

        # === Partition Detection State (January 14, 2026) ===
        # Track queue drain rate to detect cluster partitions
        self._completion_timestamps: list[float] = []
        self._consecutive_zero_drain_windows: int = 0
        self._partition_detected: bool = False
        self._partition_detected_at: float = 0.0

        # === Circuit Breaker State (January 14, 2026) ===
        from enum import Enum as PyEnum
        class _CircuitState(PyEnum):
            CLOSED = "closed"  # Normal operation
            OPEN = "open"  # Failing, reject operations
            HALF_OPEN = "half_open"  # Testing if recovered
        self._CircuitState = _CircuitState
        self._circuit_state: _CircuitState = _CircuitState.CLOSED
        self._circuit_failure_count: int = 0
        self._circuit_opened_at: float = 0.0
        self._circuit_half_open_successes: int = 0

    def _track_queued_work_id(self, work_id: str) -> None:
        """Track queued work IDs with timestamps for stale-entry cleanup."""
        now = time.time()
        self._queued_work_ids.add(work_id)
        self._queued_work_tracked_at[work_id] = now

    def _discard_queued_work_id(self, work_id: str) -> bool:
        """Discard a tracked work ID, returning whether it was present."""
        was_tracked = work_id in self._queued_work_ids
        self._queued_work_ids.discard(work_id)
        self._queued_work_tracked_at.pop(work_id, None)
        return was_tracked

    def _prune_stale_queued_work_ids(self, now: float | None = None) -> int:
        """Remove stale queued work IDs when terminal events were missed."""
        now = now if now is not None else time.time()
        stale_after_seconds = max(
            self.config.selfplay_timeout_seconds,
            self.config.export_timeout_seconds,
            self.config.validation_timeout_seconds,
            3600.0,
        ) * 2
        stale_ids = [
            work_id
            for work_id, tracked_at in self._queued_work_tracked_at.items()
            if now - tracked_at > stale_after_seconds
        ]
        for work_id in stale_ids:
            self._discard_queued_work_id(work_id)
        if stale_ids:
            logger.warning(
                f"[QueuePopulator] Pruned {len(stale_ids)} stale tracked work IDs "
                f"(remaining={len(self._queued_work_ids)})"
            )
        return len(stale_ids)








    def set_work_queue(self, work_queue: "WorkQueue") -> None:
        """Set the work queue reference."""
        self._work_queue = work_queue

    def set_selfplay_scheduler(self, scheduler: "SelfplayScheduler") -> None:
        """Set the selfplay scheduler reference."""
        self._selfplay_scheduler = scheduler
        logger.info("[QueuePopulator] SelfplayScheduler integration enabled")

    # =========================================================================
    # Elo and State Updates
    # =========================================================================

    def update_target_elo(
        self,
        board_type: str,
        num_players: int,
        elo: float,
        model_id: str | None = None,
    ) -> None:
        """Update the current best Elo for a configuration."""
        key = make_config_key(board_type, num_players)
        if key in self._targets:
            target = self._targets[key]
            if elo > target.current_best_elo:
                target.current_best_elo = elo
                target.best_model_id = model_id
                target.last_updated = time.time()
                target.record_elo(elo)

                velocity = target.elo_velocity
                eta = target.days_to_target
                eta_str = f"{eta:.1f} days" if eta else "N/A"
                logger.info(
                    f"Updated {key} best Elo: {elo:.1f} "
                    f"(gap: {target.elo_gap:.1f}, velocity: {velocity:+.1f}/day, "
                    f"ETA: {eta_str}, model: {model_id})"
                )

    def increment_games(
        self, board_type: str, num_players: int, count: int = 1
    ) -> None:
        """Increment games played for a configuration."""
        key = make_config_key(board_type, num_players)
        if key in self._targets:
            target = self._targets[key]
            target.games_played += count
            target.games_since_last_export += count
            target.last_game_time = time.time()

    def increment_training(self, board_type: str, num_players: int) -> None:
        """Increment training runs for a configuration."""
        key = make_config_key(board_type, num_players)
        if key in self._targets:
            self._targets[key].training_runs += 1

    def mark_export_complete(self, board_type: str, num_players: int, samples: int = 0) -> None:
        """Mark export as complete for a configuration."""
        key = make_config_key(board_type, num_players)
        if key in self._targets:
            target = self._targets[key]
            target.games_since_last_export = 0
            target.last_export_time = time.time()
            target.total_samples = samples
            target.pending_export = False

    def all_targets_met(self) -> bool:
        """Check if all configurations have reached target Elo."""
        return all(t.target_met for t in self._targets.values())

    def get_unmet_targets(self) -> list[ConfigTarget]:
        """Get configurations that haven't reached target Elo."""
        return [t for t in self._targets.values() if not t.target_met]

    def get_priority_target(self) -> ConfigTarget | None:
        """Get the configuration that needs the most attention."""
        unmet = self.get_unmet_targets()
        if not unmet:
            return None

        # Sort by Elo gap (smallest first) - focus on configs closest to target
        unmet.sort(key=lambda t: (t.elo_gap, -t.games_played))
        return unmet[0]

    def get_least_recent_configs(self, count: int = EXPLORATION_CONFIGS_PER_CYCLE) -> list[ConfigTarget]:
        """Get configs that haven't had recent selfplay activity.

        Phase 1.2 (Jan 2026): Ensures exploration work for stale configs.
        Returns configs sorted by staleness (oldest first).

        Args:
            count: Maximum number of configs to return

        Returns:
            List of ConfigTarget objects, sorted by last_game_time (oldest first)
        """
        now = time.time()
        stale_threshold = now - (EXPLORATION_STALE_THRESHOLD_HOURS * 3600)

        # Find stale configs (no games in threshold period)
        stale_configs = [
            t for t in self._targets.values()
            if t.last_game_time < stale_threshold
        ]

        # Sort by staleness (oldest first)
        stale_configs.sort(key=lambda t: t.last_game_time)

        return stale_configs[:count]

    def get_pending_selfplay_games(self, config_key: str) -> int:
        """Get number of pending selfplay games for a config.

        Phase 1.2 (Jan 2026): Used to check if exploration work is needed.

        Args:
            config_key: Config identifier (e.g., 'hex8_2p')

        Returns:
            Number of pending selfplay games (pending_selfplay_count * games_per_item)
        """
        target = self._targets.get(config_key)
        if not target:
            return 0

        return target.pending_selfplay_count * self.config.selfplay_games_per_item

    # =========================================================================
    # Queue Status
    # =========================================================================






    # =========================================================================
    # Backpressure and Priority
    # =========================================================================




    # =========================================================================
    # Backoff, Health, Partition Detection, and Circuit Breaker (Jan 14, 2026)
    # =========================================================================















    # =========================================================================
    # Main Population Logic
    # =========================================================================










    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get populator status for monitoring."""
        unmet = self.get_unmet_targets()
        met = [t for t in self._targets.values() if t.target_met]

        velocities = [t.elo_velocity for t in unmet if t.elo_velocity > 0]
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0

        return {
            "enabled": self.config.enabled,
            "min_queue_depth": self.config.min_queue_depth,
            "target_queue_depth": self.config.target_queue_depth,
            "max_batch_per_cycle": self.config.max_batch_per_cycle,
            "current_queue_depth": self.get_current_queue_depth(),
            "target_elo": self.config.target_elo,
            "total_configs": len(self._targets),
            "configs_met": len(met),
            "configs_unmet": len(unmet),
            "all_targets_met": self.all_targets_met(),
            "avg_velocity": avg_velocity,
            "cluster_health_factor": self._cluster_health_factor,
            "dead_nodes": len(self._dead_nodes),
            "unmet_configs": [
                {
                    "config": t.config_key,
                    "current_elo": t.current_best_elo,
                    "gap": t.elo_gap,
                    "velocity": t.elo_velocity,
                    "days_to_target": t.days_to_target,
                    "games": t.games_played,
                    "training_runs": t.training_runs,
                    "pending_selfplay": t.pending_selfplay_count,
                    "curriculum_weight": t.curriculum_weight,
                }
                for t in unmet
            ],
            "last_populate_time": self._last_populate_time,
            "total_queued": len(self._queued_work_ids),
        }


# =============================================================================
# Daemon Wrapper
# =============================================================================

class UnifiedQueuePopulatorDaemon:
    """Async daemon wrapper for UnifiedQueuePopulator.

    Provides:
    - Background monitoring loop
    - Event subscriptions for automatic updates
    - P2P health integration
    - Graceful start/stop lifecycle
    """

    def __init__(
        self,
        config: QueuePopulatorConfig | None = None,
        work_queue: Optional["WorkQueue"] = None,
        elo_db_path: str | None = None,
        selfplay_scheduler: Optional["SelfplayScheduler"] = None,
    ):
        # Jan 4, 2026: Auto-get work queue singleton if not passed
        # Fixes "No work queue set, cannot populate" issue
        if work_queue is None:
            try:
                from app.coordination.work_queue import get_work_queue
                work_queue = get_work_queue()
                logger.debug("[QueuePopulatorDaemon] Using singleton work queue")
            except Exception as e:
                logger.warning(f"[QueuePopulatorDaemon] Failed to get work queue: {e}")

        self._populator = UnifiedQueuePopulator(
            config=config,
            work_queue=work_queue,
            elo_db_path=elo_db_path,
            selfplay_scheduler=selfplay_scheduler,
        )
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def populator(self) -> UnifiedQueuePopulator:
        """Get the underlying populator instance."""
        return self._populator

    async def start(self) -> None:
        """Start the daemon."""
        if self._running:
            logger.warning("[QueuePopulatorDaemon] Already running")
            return

        self._running = True
        logger.info("[QueuePopulatorDaemon] Starting")

        # Subscribe to events
        await self._subscribe_to_events()

        # Start background loop
        self._task = asyncio.create_task(self._monitor_loop())
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        for unsub in self._populator._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except (TypeError, RuntimeError) as e:
                logger.debug(f"[QueuePopulatorDaemon] Unsubscribe failed: {e}")

        logger.info("[QueuePopulatorDaemon] Stopped")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Handle task completion or failure."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[QueuePopulatorDaemon] Task failed: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

    async def _monitor_loop(self) -> None:
        """Background loop to periodically populate queue.

        January 5, 2026: Added leader awareness and follower takeover mechanism.
        - Leaders populate the queue normally
        - Followers monitor queue depth and take over if empty for 30+ seconds
        - This reduces queue empty gap from 60-75s to <30s during leader failover
        """
        # Follower takeover configuration (Jan 5, 2026 - Session 17.34)
        FOLLOWER_TAKEOVER_THRESHOLD = 30.0  # seconds queue empty before follower takes over
        FOLLOWER_TAKEOVER_BATCH = 200  # items to add during takeover
        FOLLOWER_CHECK_INTERVAL = 5.0  # shorter interval for followers to detect empty queue

        queue_empty_since: float | None = None
        not_leader_skips = 0

        while self._running:
            try:
                # Check if this node is the P2P leader (Jan 5, 2026)
                is_leader = True
                leader_id = None
                try:
                    from app.core.node import check_p2p_leader_status
                    is_leader, leader_id = await check_p2p_leader_status(timeout=5.0)
                except Exception as e:
                    # If we can't determine leadership, default to populating (fail-open)
                    logger.debug(f"[QueuePopulatorDaemon] Leader check failed, assuming leader: {e}")
                    is_leader = True

                if is_leader:
                    # We are the leader - populate normally
                    if not_leader_skips > 0:
                        logger.info(
                            f"[QueuePopulatorDaemon] Resuming as leader "
                            f"(was follower for {not_leader_skips} cycles)"
                        )
                        not_leader_skips = 0
                    queue_empty_since = None  # Reset empty tracker when leader
                    self._populator.populate()
                    await asyncio.sleep(self._populator.config.check_interval_seconds)
                else:
                    # We are a follower - monitor queue and potentially take over
                    not_leader_skips += 1

                    # Check queue depth for follower takeover
                    queue_depth = 0
                    try:
                        if self._populator._work_queue:
                            queue_depth = self._populator._work_queue.size()
                    except Exception:
                        pass

                    if queue_depth == 0:
                        # Queue is empty - track duration
                        if queue_empty_since is None:
                            queue_empty_since = time.time()
                            logger.debug("[QueuePopulatorDaemon] Queue empty detected (follower)")

                        empty_duration = time.time() - queue_empty_since
                        if empty_duration >= FOLLOWER_TAKEOVER_THRESHOLD:
                            # Follower takeover: queue has been empty too long
                            logger.warning(
                                f"[QueuePopulatorDaemon] Follower takeover: queue empty for "
                                f"{empty_duration:.1f}s (threshold: {FOLLOWER_TAKEOVER_THRESHOLD}s), "
                                f"leader: {leader_id}"
                            )
                            # Populate with limited batch to avoid overwhelming
                            original_batch = self._populator.config.max_batch_per_cycle
                            self._populator.config.max_batch_per_cycle = min(
                                original_batch, FOLLOWER_TAKEOVER_BATCH
                            )
                            try:
                                added = self._populator.populate()
                                if added > 0:
                                    logger.info(
                                        f"[QueuePopulatorDaemon] Follower takeover added {added} items"
                                    )
                                    queue_empty_since = None  # Reset after successful takeover
                            finally:
                                self._populator.config.max_batch_per_cycle = original_batch
                    else:
                        # Queue has items - reset empty tracker
                        if queue_empty_since is not None:
                            logger.debug(
                                f"[QueuePopulatorDaemon] Queue recovered with {queue_depth} items"
                            )
                        queue_empty_since = None

                    # Log occasionally to avoid spam
                    if not_leader_skips % 20 == 1:
                        logger.debug(
                            f"[QueuePopulatorDaemon] Not leader (leader: {leader_id}, "
                            f"queue: {queue_depth}, skips: {not_leader_skips})"
                        )

                    # Shorter interval for followers to detect empty queue quickly
                    await asyncio.sleep(FOLLOWER_CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[QueuePopulatorDaemon] Monitor loop error: {e}")
                await asyncio.sleep(30)

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        await self._subscribe_to_data_events()
        await self._subscribe_to_p2p_health_events()

    async def _subscribe_to_data_events(self) -> None:
        """Subscribe to data/training events."""
        global _events_wired

        # Skip if already wired (by wire_queue_populator_events() or another daemon)
        if _events_wired:
            logger.debug("[QueuePopulatorDaemon] Events already wired, skipping")
            return

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            def _extract_payload(event: Any) -> dict[str, Any]:
                if isinstance(event, dict):
                    return event
                return getattr(event, "payload", {})

            def _on_elo_updated(event: Any) -> None:
                payload = _extract_payload(event)
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                elo = payload.get("elo") or payload.get("rating")
                model_id = payload.get("model_id") or payload.get("participant_id")
                if board_type and num_players and elo:
                    self._populator.update_target_elo(board_type, num_players, elo, model_id)

            def _on_training_completed(event: Any) -> None:
                payload = _extract_payload(event)
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                if board_type and num_players:
                    self._populator.increment_training(board_type, num_players)
                    # CRITICAL: Replenish queue after training completes
                    self._populator.populate()

            def _on_new_games(event: Any) -> None:
                payload = _extract_payload(event)
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                count = payload.get("count", 1)
                if board_type and num_players:
                    self._populator.increment_games(board_type, num_players, count)
                    # December 29, 2025: Trigger population when new games available
                    # This ensures the queue stays filled as data becomes available
                    self._populator.populate()

            def _on_selfplay_complete(event: Any) -> None:
                payload = _extract_payload(event)
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                games = payload.get("games_generated", 0)
                config_key = make_config_key(board_type, num_players) if board_type and num_players else ""
                if config_key in self._populator._targets:
                    target = self._populator._targets[config_key]
                    if target.pending_selfplay_count > 0:
                        target.pending_selfplay_count -= 1
                    if games:
                        self._populator.increment_games(board_type, num_players, games)
                # Replenish queue when selfplay slot becomes available
                self._populator.populate()

            def _on_training_blocked(event: Any) -> None:
                """Handle TRAINING_BLOCKED_BY_QUALITY - queue extra selfplay."""
                payload = _extract_payload(event)
                config_key = extract_config_key(payload)
                if not config_key:
                    return

                # Parse config_key using canonical utility
                parsed = parse_config_key(config_key)
                if not parsed:
                    return
                board_type = parsed.board_type
                num_players = parsed.num_players

                if self._populator._work_queue is None:
                    return

                # Add 3 priority selfplay items
                added = 0
                for _ in range(3):
                    try:
                        item = self._populator._create_selfplay_item(board_type, num_players)
                        item.priority = self._populator.config.selfplay_priority + 30
                        # January 2026: Force-add for starved configs to bypass backpressure
                        force_add = self._populator._should_force_queue_add(config_key)
                        self._populator._work_queue.add_work(item, force=force_add)
                        self._populator._track_queued_work_id(item.work_id)
                        added += 1
                    except (ValueError, KeyError, AttributeError) as e:
                        logger.debug(f"[QueuePopulator] Failed to create work item: {e}")

                if added > 0:
                    logger.info(
                        f"[QueuePopulator] Queued {added} priority selfplay for {config_key}"
                    )

            def _discard_tracked_work(payload: dict[str, Any], event_name: str) -> None:
                work_id = payload.get("work_id") or payload.get("task_id")
                if not work_id:
                    return
                if self._populator._discard_queued_work_id(work_id):
                    logger.debug(
                        f"[QueuePopulator] {event_name}: removed tracked work {work_id}, "
                        f"remaining tracked: {len(self._populator._queued_work_ids)}"
                    )

            def _on_work_failed(event: Any) -> None:
                """Handle WORK_FAILED - decrement pending count for failed work."""
                payload = _extract_payload(event)
                _discard_tracked_work(payload, "WORK_FAILED")
                work_type = payload.get("work_type")
                if work_type != "selfplay":
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                reason = payload.get("reason", "unknown")
                config_key = make_config_key(board_type, num_players) if board_type and num_players else ""

                if config_key and config_key in self._populator._targets:
                    target = self._populator._targets[config_key]
                    if target.pending_selfplay_count > 0:
                        target.pending_selfplay_count -= 1
                        logger.info(
                            f"[QueuePopulator] Work failed for {config_key} ({reason}), "
                            f"pending: {target.pending_selfplay_count}"
                        )
                # Replace failed work immediately
                self._populator.populate()

            def _on_work_timeout(event: Any) -> None:
                """Handle WORK_TIMEOUT - decrement pending count for timed out work."""
                payload = _extract_payload(event)
                _discard_tracked_work(payload, "WORK_TIMEOUT")
                work_type = payload.get("work_type")
                if work_type != "selfplay":
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                node_id = payload.get("node_id", "unknown")
                config_key = make_config_key(board_type, num_players) if board_type and num_players else ""

                if config_key and config_key in self._populator._targets:
                    target = self._populator._targets[config_key]
                    if target.pending_selfplay_count > 0:
                        target.pending_selfplay_count -= 1
                        logger.warning(
                            f"[QueuePopulator] Work timed out for {config_key} on {node_id}, "
                            f"pending: {target.pending_selfplay_count}"
                        )
                # Replace timed out work immediately
                self._populator.populate()

            def _on_work_completed(event: Any) -> None:
                """Handle WORK_COMPLETED - remove from tracked work IDs set.

                This is critical for accurate tracking: without this handler, the
                _queued_work_ids set grows indefinitely, causing the populator to
                incorrectly believe work is still pending when it has completed.
                """
                payload = _extract_payload(event)
                _discard_tracked_work(payload, "WORK_COMPLETED")
                work_type = payload.get("work_type")

                # For selfplay work, also update pending counts
                if work_type == "selfplay":
                    board_type = payload.get("board_type")
                    num_players = payload.get("num_players")
                    config_key = make_config_key(board_type, num_players) if board_type and num_players else ""

                    if config_key and config_key in self._populator._targets:
                        target = self._populator._targets[config_key]
                        if target.pending_selfplay_count > 0:
                            target.pending_selfplay_count -= 1

            def _on_backpressure_released(event: Any) -> None:
                """Handle BACKPRESSURE_RELEASED - resume queue population."""
                payload = _extract_payload(event)
                source = payload.get("source", "unknown")
                logger.info(f"[QueuePopulator] Backpressure released from {source}, repopulating queue")
                # Resume population immediately when backpressure lifted
                self._populator.populate()

            def _on_task_abandoned(event: Any) -> None:
                """Handle TASK_ABANDONED - decrement pending count for abandoned tasks.

                TASK_ABANDONED is emitted when a task is intentionally cancelled (e.g.,
                due to backpressure, resource constraints, or pipeline requirements).
                Unlike WORK_FAILED (unexpected errors) or WORK_TIMEOUT (deadline exceeded),
                abandonment is a controlled termination.
                """
                payload = _extract_payload(event)
                _discard_tracked_work(payload, "TASK_ABANDONED")
                task_type = payload.get("task_type", "")
                if "selfplay" not in task_type.lower():
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                reason = payload.get("reason", "unknown")
                config_key = make_config_key(board_type, num_players) if board_type and num_players else ""

                if config_key and config_key in self._populator._targets:
                    target = self._populator._targets[config_key]
                    if target.pending_selfplay_count > 0:
                        target.pending_selfplay_count -= 1
                        logger.info(
                            f"[QueuePopulator] Task abandoned for {config_key} ({reason}), "
                            f"pending: {target.pending_selfplay_count}"
                        )

            def _on_selfplay_target_updated(event: Any) -> None:
                """Handle SELFPLAY_TARGET_UPDATED - repopulate queue when targets change."""
                payload = _extract_payload(event)
                config_key = extract_config_key(payload)
                new_target = payload.get("target_games") or payload.get("games_target")
                logger.info(
                    f"[QueuePopulator] Selfplay target updated for {config_key}: {new_target}"
                )
                # December 29, 2025: Repopulate queue when targets change
                self._populator.populate()

            router.subscribe(DataEventType.ELO_UPDATED.value, _on_elo_updated)
            router.subscribe(DataEventType.TRAINING_COMPLETED.value, _on_training_completed)
            router.subscribe(DataEventType.NEW_GAMES_AVAILABLE.value, _on_new_games)
            router.subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY.value, _on_training_blocked)

            if hasattr(DataEventType, 'SELFPLAY_COMPLETE'):
                router.subscribe(DataEventType.SELFPLAY_COMPLETE.value, _on_selfplay_complete)

            # December 29, 2025: Wire SELFPLAY_TARGET_UPDATED to adjust queue when targets change
            if hasattr(DataEventType, 'SELFPLAY_TARGET_UPDATED'):
                router.subscribe(DataEventType.SELFPLAY_TARGET_UPDATED.value, _on_selfplay_target_updated)

            # Wire WORK_FAILED, WORK_TIMEOUT, WORK_COMPLETED, TASK_ABANDONED for accurate pending count tracking
            if hasattr(DataEventType, 'WORK_FAILED'):
                router.subscribe(DataEventType.WORK_FAILED.value, _on_work_failed)
            if hasattr(DataEventType, 'WORK_TIMEOUT'):
                router.subscribe(DataEventType.WORK_TIMEOUT.value, _on_work_timeout)
            if hasattr(DataEventType, 'WORK_COMPLETED'):
                router.subscribe(DataEventType.WORK_COMPLETED.value, _on_work_completed)
            if hasattr(DataEventType, 'TASK_ABANDONED'):
                router.subscribe(DataEventType.TASK_ABANDONED.value, _on_task_abandoned)

            # Wire BACKPRESSURE_RELEASED to resume population when cluster pressure drops
            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                router.subscribe(DataEventType.BACKPRESSURE_RELEASED.value, _on_backpressure_released)

            _events_wired = True
            logger.info("[QueuePopulatorDaemon] Subscribed to data events (incl. WORK_FAILED/TIMEOUT/COMPLETED/TASK_ABANDONED/BACKPRESSURE_RELEASED)")

        except ImportError:
            logger.debug("[QueuePopulatorDaemon] Event router not available")

    async def _subscribe_to_p2p_health_events(self) -> None:
        """Subscribe to P2P cluster health events."""
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            def _on_node_dead(event: Any) -> None:
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "")
                if node_id:
                    self._populator._dead_nodes.add(node_id)
                    logger.warning(
                        f"[QueuePopulator] Node {node_id} marked dead. "
                        f"Dead nodes: {len(self._populator._dead_nodes)}"
                    )

            def _on_node_recovered(event: Any) -> None:
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                node_id = payload.get("node_id", "")
                if node_id:
                    self._populator._dead_nodes.discard(node_id)
                    logger.info(
                        f"[QueuePopulator] Node {node_id} recovered. "
                        f"Dead nodes: {len(self._populator._dead_nodes)}"
                    )

            def _on_cluster_unhealthy(event: Any) -> None:
                payload = event if isinstance(event, dict) else getattr(event, "payload", {})
                healthy = payload.get("healthy_nodes", 0)
                total = payload.get("total_nodes", 0)
                logger.warning(
                    f"[QueuePopulator] Cluster unhealthy: {healthy}/{total}"
                )
                if total > 0:
                    self._populator._cluster_health_factor = max(0.2, healthy / total)
                else:
                    self._populator._cluster_health_factor = 0.5

            def _on_cluster_healthy(event: Any) -> None:
                logger.info("[QueuePopulator] Cluster healthy")
                self._populator._cluster_health_factor = 1.0
                self._populator._dead_nodes.clear()

            events_subscribed = []

            for event_name, handler in [
                ('P2P_NODE_DEAD', _on_node_dead),
                ('NODE_UNHEALTHY', _on_node_dead),
                ('NODE_RECOVERED', _on_node_recovered),
                ('P2P_CLUSTER_UNHEALTHY', _on_cluster_unhealthy),
                ('P2P_CLUSTER_HEALTHY', _on_cluster_healthy),
            ]:
                if hasattr(DataEventType, event_name):
                    router.subscribe(getattr(DataEventType, event_name).value, handler)
                    events_subscribed.append(event_name)

            if events_subscribed:
                logger.info(
                    f"[QueuePopulatorDaemon] Subscribed to P2P health: {', '.join(events_subscribed)}"
                )

        except ImportError:
            logger.debug("[QueuePopulatorDaemon] Event router not available for P2P health")

    def get_status(self) -> dict[str, Any]:
        """Get daemon status."""
        status = self._populator.get_status()
        status["daemon_running"] = self._running
        return status

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status.

        December 2025: Added to satisfy CoordinatorProtocol for unified health monitoring.
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Queue populator daemon not running",
            )

        # Check queue depth
        current_depth = self._populator.get_current_queue_depth()
        min_depth = self._populator.config.min_queue_depth

        if current_depth == 0:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="Queue is empty - no work items available",
                details=self.get_status(),
            )

        # Check if all targets met
        if self._populator.all_targets_met():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="All Elo targets met - queue populator idle",
                details=self.get_status(),
            )

        # Check cluster health factor
        if self._populator._cluster_health_factor < 0.5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Cluster health degraded ({self._populator._cluster_health_factor:.1%})",
                details=self.get_status(),
            )

        unmet = len(self._populator.get_unmet_targets())
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Queue populator running (depth: {current_depth}, unmet: {unmet})",
            details=self.get_status(),
        )


# =============================================================================
# Singleton Pattern
# =============================================================================

_populator: UnifiedQueuePopulator | None = None
_daemon: UnifiedQueuePopulatorDaemon | None = None
_events_wired: bool = False  # Track if events already subscribed to prevent duplicates


def get_queue_populator(
    config: QueuePopulatorConfig | None = None,
    work_queue: Optional["WorkQueue"] = None,
) -> UnifiedQueuePopulator:
    """Get or create the singleton QueuePopulator instance."""
    global _populator
    if _populator is None:
        _populator = UnifiedQueuePopulator(config=config, work_queue=work_queue)
    elif work_queue is not None and _populator._work_queue is None:
        _populator.set_work_queue(work_queue)
    return _populator


def get_queue_populator_daemon(
    config: QueuePopulatorConfig | None = None,
) -> UnifiedQueuePopulatorDaemon:
    """Get or create the singleton daemon instance."""
    global _daemon
    if _daemon is None:
        _daemon = UnifiedQueuePopulatorDaemon(config=config)
    return _daemon


def reset_queue_populator() -> None:
    """Reset singletons for testing."""
    global _populator, _daemon, _events_wired
    _populator = None
    _daemon = None
    _events_wired = False


async def start_queue_populator_daemon(
    config: QueuePopulatorConfig | None = None,
) -> UnifiedQueuePopulatorDaemon:
    """Start the queue populator daemon."""
    daemon = get_queue_populator_daemon(config)
    await daemon.start()
    return daemon


def wire_queue_populator_events() -> UnifiedQueuePopulator:
    """Wire queue populator to the event bus for automatic updates.

    This is a synchronous convenience function for non-daemon usage.
    For full async support, use UnifiedQueuePopulatorDaemon instead.

    Note: If UnifiedQueuePopulatorDaemon is already running, this is a no-op
    to prevent duplicate event handlers.
    """
    global _events_wired
    populator = get_queue_populator()

    # Skip if already wired (by daemon or previous call)
    if _events_wired:
        logger.debug("[QueuePopulator] Events already wired, skipping")
        return populator

    try:
        from app.coordination.event_router import DataEventType, get_router

        router = get_router()

        def _extract_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            return getattr(event, "payload", {})

        def _on_elo_updated(event: Any) -> None:
            payload = _extract_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            elo = payload.get("elo") or payload.get("rating")
            model_id = payload.get("model_id") or payload.get("participant_id")
            if board_type and num_players and elo:
                populator.update_target_elo(board_type, num_players, elo, model_id)

        def _on_training_completed(event: Any) -> None:
            payload = _extract_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            if board_type and num_players:
                populator.increment_training(board_type, num_players)

        def _on_new_games(event: Any) -> None:
            payload = _extract_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            count = payload.get("count", 1)
            if board_type and num_players:
                populator.increment_games(board_type, num_players, count)

        router.subscribe(DataEventType.ELO_UPDATED.value, _on_elo_updated)
        router.subscribe(DataEventType.TRAINING_COMPLETED.value, _on_training_completed)
        router.subscribe(DataEventType.NEW_GAMES_AVAILABLE.value, _on_new_games)

        _events_wired = True
        logger.info("[QueuePopulator] Wired to event bus")

    except ImportError:
        logger.warning("[QueuePopulator] Event router not available")

    return populator


def load_populator_config_from_yaml(yaml_config: dict[str, Any]) -> QueuePopulatorConfig:
    """Load QueuePopulatorConfig from YAML configuration dict."""
    populator = yaml_config.get("queue_populator", {})

    return QueuePopulatorConfig(
        enabled=populator.get("enabled", True),
        min_queue_depth=populator.get("min_queue_depth", 200),
        max_pending_items=populator.get("max_pending_items", 50),
        target_queue_depth=populator.get("target_queue_depth", 300),
        max_batch_per_cycle=populator.get("max_batch_per_cycle", 100),
        check_interval_seconds=populator.get("check_interval_seconds", 10),
        target_elo=populator.get("target_elo", 2000.0),
        selfplay_ratio=populator.get("selfplay_ratio", 0.60),
        training_ratio=populator.get("training_ratio", 0.30),
        tournament_ratio=populator.get("tournament_ratio", 0.10),
        board_types=populator.get("board_types", ["square8", "square19", "hex8", "hexagonal"]),
        player_counts=populator.get("player_counts", [2, 3, 4]),
        selfplay_games_per_item=populator.get("selfplay_games_per_item", 50),
        selfplay_priority=populator.get("selfplay_priority", 75),
        training_priority=populator.get("training_priority", 100),
        min_games_for_training=populator.get("min_games_for_training", 300),
        tournament_games=populator.get("tournament_games", 50),
        tournament_priority=populator.get("tournament_priority", 80),
    )


# =============================================================================
# Module Exports
# =============================================================================

# =============================================================================
# Backward-Compatible Aliases (Dec 2025)
# =============================================================================
# These aliases maintain compatibility with code that imports from the
# deprecated app.coordination.queue_populator module.

PopulatorConfig = QueuePopulatorConfig  # Alias for backward compatibility
QueuePopulator = UnifiedQueuePopulator  # Alias for backward compatibility

__all__ = [
    # Constants
    "BOARD_CONFIGS",
    "LARGE_BOARDS",
    "DEFAULT_CURRICULUM_WEIGHTS",
    "MINIMUM_EXPLORATION_GAMES",
    "EXPLORATION_CONFIGS_PER_CYCLE",
    "EXPLORATION_STALE_THRESHOLD_HOURS",
    # Enums
    "BoardType",
    # Data classes
    "QueuePopulatorConfig",
    "ConfigTarget",
    # Main classes
    "UnifiedQueuePopulator",
    "UnifiedQueuePopulatorDaemon",
    # Singleton functions
    "get_queue_populator",
    "get_queue_populator_daemon",
    "reset_queue_populator",
    "start_queue_populator_daemon",
    # Utilities
    "wire_queue_populator_events",
    "load_populator_config_from_yaml",
    # Backward-compatible aliases
    "PopulatorConfig",  # Alias for QueuePopulatorConfig
    "QueuePopulator",   # Alias for UnifiedQueuePopulator
]
