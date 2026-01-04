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
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

# Canonical types (December 2025 consolidation)
from app.coordination.types import BackpressureLevel, BoardType
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.event_handler_utils import (
    extract_config_key,
    extract_board_type,
    extract_num_players,
)

if TYPE_CHECKING:
    from app.coordination.selfplay_scheduler import SelfplayScheduler
    from app.coordination.work_queue import WorkItem, WorkQueue

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# All supported board configurations
BOARD_CONFIGS: list[tuple[str, int]] = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Large boards require specialized engine selection (Gumbel MCTS)
# These have many more cells (361-469) and benefit from quality over quantity
LARGE_BOARDS = frozenset({"square19", "hexagonal", "fullhex", "full_hex"})

# Minimum exploration constants (Phase 1.2 - Jan 2026)
# Ensures cluster never idles completely, even when Elo targets are met
MINIMUM_EXPLORATION_GAMES = 100  # Minimum pending games per config
EXPLORATION_CONFIGS_PER_CYCLE = 3  # Number of configs to explore each cycle
EXPLORATION_STALE_THRESHOLD_HOURS = 4.0  # Config is "stale" if no games in this many hours

# Default curriculum weights (priority multipliers)
# Dec 27, 2025: REBALANCED to prioritize 3p/4p configs which are severely starved
# Previous weights heavily favored 2p (0.7-1.0) over 3p/4p (0.4-0.6)
# New weights: 2p reduced, 3p/4p significantly increased to catch up
DEFAULT_CURRICULUM_WEIGHTS: dict[str, float] = {
    # Square8: Good 2p data, need more 3p/4p
    "square8_2p": 0.6, "square8_3p": 1.4, "square8_4p": 1.5,
    # Square19: All configs need more data
    "square19_2p": 0.8, "square19_3p": 1.3, "square19_4p": 1.4,
    # Hex8: 2p has some data, 3p/4p critical (especially 4p at 30% win rate)
    "hex8_2p": 0.5, "hex8_3p": 1.5, "hex8_4p": 1.6,
    # Hexagonal: All configs starved
    "hexagonal_2p": 1.0, "hexagonal_3p": 1.4, "hexagonal_4p": 1.3,
}


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
    min_queue_depth: int = 200

    # Maximum pending items before stopping generation
    max_pending_items: int = 50

    # Target queue depth to aim for (queue will fill to this level)
    # December 29, 2025: Added to reduce queue variance from 2,170% to <50%
    target_queue_depth: int = 300

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
    selfplay_priority: int = 50
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
    trickle_mode_enabled: bool = True
    trickle_min_items: int = 10  # Minimum items to add even under max backpressure


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

class UnifiedQueuePopulator:
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

        # Work tracking
        self._queued_work_ids: set[str] = set()
        self._last_populate_time: float = 0

        # P2P health tracking (December 2025)
        self._dead_nodes: set[str] = set()
        self._cluster_health_factor: float = 1.0

        # Event subscriptions
        self._event_subscriptions: list[Callable] = []

    def _scale_queue_depth_to_cluster(self) -> None:
        """Scale min_queue_depth based on cluster size."""
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=False,
                include_disk_usage=False,
            )
            active_nodes = status.active_nodes

            if active_nodes > 0:
                scaled_depth = max(50, active_nodes * 2)
                old_depth = self.config.min_queue_depth
                self.config.min_queue_depth = scaled_depth
                logger.info(
                    f"[QueuePopulator] Scaled queue depth: {old_depth} -> {scaled_depth} "
                    f"(for {active_nodes} active nodes)"
                )

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[QueuePopulator] Failed to scale queue depth: {e}")

    def _init_targets(self) -> None:
        """Initialize targets for all board/player configurations."""
        for board_type in self.config.board_types:
            for num_players in self.config.player_counts:
                target = ConfigTarget(
                    board_type=board_type,
                    num_players=num_players,
                    target_elo=self.config.target_elo,
                )
                self._targets[target.config_key] = target

    def _load_existing_elo(self) -> None:
        """Load existing Elo ratings from the database."""
        if self._elo_db_path:
            db_path = Path(self._elo_db_path)
        else:
            candidates = [
                Path(__file__).parent.parent.parent / "data" / "unified_elo.db",
                Path("/lambda/nfs/RingRift/elo/unified_elo.db"),
                Path.home() / "ringrift" / "ai-service" / "data" / "unified_elo.db",
            ]
            db_path = None
            for candidate in candidates:
                if candidate.exists():
                    db_path = candidate
                    break

        if not db_path or not db_path.exists():
            logger.info("No Elo database found, starting with default 1500 Elo")
            return

        try:
            # December 27, 2025: Use context manager to prevent connection leaks
            with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT e.board_type, e.num_players, e.rating as best_elo,
                           e.participant_id, e.games_played
                    FROM elo_ratings e
                    INNER JOIN (
                        SELECT board_type, num_players, MAX(rating) as max_rating
                        FROM elo_ratings
                        WHERE archived_at IS NULL
                        GROUP BY board_type, num_players
                    ) m ON e.board_type = m.board_type
                       AND e.num_players = m.num_players
                       AND e.rating = m.max_rating
                    WHERE e.archived_at IS NULL
                """)

                rows = cursor.fetchall()

            for row in rows:
                board_type, num_players, best_elo, model_id, games = row
                key = f"{board_type}_{num_players}p"
                if key in self._targets:
                    target = self._targets[key]
                    target.current_best_elo = best_elo
                    target.best_model_id = model_id
                    target.games_played = games or 0
                    target.record_elo(best_elo)
                    logger.info(
                        f"Loaded existing Elo for {key}: {best_elo:.1f} "
                        f"(model: {model_id}, games: {games})"
                    )

            met = sum(1 for t in self._targets.values() if t.target_met)
            logger.info(
                f"Loaded Elo data: {met}/{len(self._targets)} configs at target "
                f"({self.config.target_elo}+ Elo)"
            )

        except Exception as e:
            logger.warning(f"Failed to load existing Elo data: {e}")

    def _load_curriculum_weights(self) -> None:
        """Load curriculum weights for prioritization."""
        try:
            from app.coordination.curriculum_weights import load_curriculum_weights

            weights = load_curriculum_weights()
            for config_key, weight in weights.items():
                if config_key in self._targets:
                    self._targets[config_key].curriculum_weight = weight
            logger.info("[QueuePopulator] Loaded curriculum weights")
        except ImportError:
            for config_key, weight in DEFAULT_CURRICULUM_WEIGHTS.items():
                if config_key in self._targets:
                    self._targets[config_key].curriculum_weight = weight
            logger.debug("[QueuePopulator] Using default curriculum weights")

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
        key = f"{board_type}_{num_players}p"
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
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            target = self._targets[key]
            target.games_played += count
            target.games_since_last_export += count
            target.last_game_time = time.time()

    def increment_training(self, board_type: str, num_players: int) -> None:
        """Increment training runs for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            self._targets[key].training_runs += 1

    def mark_export_complete(self, board_type: str, num_players: int, samples: int = 0) -> None:
        """Mark export as complete for a configuration."""
        key = f"{board_type}_{num_players}p"
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

    def get_current_queue_depth(self) -> int:
        """Get current work queue depth."""
        if self._work_queue is None:
            return 0
        status = self._work_queue.get_queue_status()
        pending = len(status.get("pending", []))
        running = len(status.get("running", []))
        return pending + running

    def calculate_items_needed(self) -> int:
        """Calculate how many items to add to reach target depth.

        December 29, 2025: Now targets target_queue_depth instead of min_queue_depth,
        and caps at max_batch_per_cycle to prevent burst releases that cause
        queue variance spikes (was 2,170% variance, target <50%).
        """
        current = self.get_current_queue_depth()
        # Use target_queue_depth for filling, but only add if below min_queue_depth
        if current >= self.config.min_queue_depth:
            # Already above minimum, gradually fill to target
            needed = max(0, self.config.target_queue_depth - current)
        else:
            # Below minimum, fill more aggressively to min_queue_depth
            needed = max(0, self.config.target_queue_depth - current)

        # Cap at max_batch_per_cycle to prevent burst releases
        return min(needed, self.config.max_batch_per_cycle)

    # =========================================================================
    # Backpressure and Priority
    # =========================================================================

    def _check_backpressure(self) -> tuple[BackpressureLevel, float]:
        """Check current backpressure level."""
        try:
            from app.coordination.queue_monitor import get_queue_monitor

            monitor = get_queue_monitor()
            if monitor:
                status = monitor.get_overall_status()
                bp_level = status.get("backpressure_level", "none")
                if isinstance(bp_level, str):
                    bp_level = BackpressureLevel(bp_level)
                elif hasattr(bp_level, "value"):
                    bp_level = BackpressureLevel(bp_level.value)

                if bp_level.should_stop():
                    return bp_level, 0.0

                return bp_level, bp_level.reduction_factor()

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[QueuePopulator] Backpressure check failed: {e}")

        return BackpressureLevel.NONE, 1.0

    def _get_scheduler_priorities(self) -> dict[str, float]:
        """Get priority scores from the SelfplayScheduler."""
        if self._selfplay_scheduler is None:
            return {}

        try:
            # Check if we're in a running event loop (Dec 2025: use get_running_loop)
            try:
                asyncio.get_running_loop()
                # We're in a running loop - can't use run_until_complete
                # Fall back to cached priorities to avoid deadlock
                priorities_list = getattr(self._selfplay_scheduler, "_cached_priorities", [])
                if priorities_list:
                    return dict(priorities_list)
                return {}
            except RuntimeError:
                # No running loop - safe to create one and run sync
                priorities_list = asyncio.run(
                    self._selfplay_scheduler.get_priority_configs(top_n=12)
                )
                return dict(priorities_list)
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[QueuePopulator] Could not get scheduler priorities: {e}")
            return {}

    def _compute_work_priority(
        self,
        base_priority: int,
        config_key: str,
        scheduler_priorities: dict[str, float],
    ) -> int:
        """Compute adjusted work priority based on scheduler priorities."""
        if not scheduler_priorities:
            return base_priority

        scheduler_score = scheduler_priorities.get(config_key, 0.0)
        if scheduler_score <= 0:
            return base_priority

        max_score = max(scheduler_priorities.values()) if scheduler_priorities else 1.0
        normalized = scheduler_score / max(max_score, 0.01)
        priority_boost = int(normalized * 50)

        return base_priority + priority_boost

    # =========================================================================
    # Work Item Creation
    # =========================================================================

    # Jan 1, 2026: Engine modes for diversity in selfplay
    # Cycle through different harness types to ensure diverse training data
    # Format: (engine_mode, requires_model, player_restriction)
    # player_restriction: None = all, 2 = 2p only, (3,4) = 3-4p only
    _ENGINE_MODES_2P = [
        ("gumbel", True, None),       # High-quality MCTS with NN
        ("heuristic-only", False, None),  # Fast bootstrap
        ("minimax", True, 2),         # Alpha-beta (2p only)
        ("policy-only", True, None),  # Policy network only
        ("descent", True, None),      # Gradient descent search
    ]
    _ENGINE_MODES_MP = [
        ("gumbel", True, None),       # High-quality MCTS with NN
        ("heuristic-only", False, None),  # Fast bootstrap
        ("maxn", True, (3, 4)),       # Max-N for multiplayer
        ("brs", True, (3, 4)),        # Best-Reply Search
        ("policy-only", True, None),  # Policy network only
    ]
    _engine_mode_counter = 0

    def _create_selfplay_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a selfplay work item with diverse engine types.

        Jan 1, 2026: Added engine mode rotation for training data diversity.
        Cycles through available harness types (gumbel, heuristic, minimax, etc.)
        to ensure training data comes from varied play styles.
        """
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"selfplay_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        key = f"{board_type}_{num_players}p"
        target = self._targets.get(key)
        best_model = target.best_model_id if target else None
        model_elo = target.current_best_elo if target else 1500.0

        # Select engine mode with diversity rotation
        # Jan 1, 2026: Rotate through multiple engine modes instead of always using one
        if board_type in LARGE_BOARDS:
            # Large boards: heuristic-only (gumbel too slow without GPU tree)
            engine_mode = "heuristic-only"
        else:
            # Select engine mode based on player count and rotation counter
            modes = self._ENGINE_MODES_2P if num_players == 2 else self._ENGINE_MODES_MP

            # Find valid modes for this configuration
            valid_modes = []
            for mode, requires_model, player_restrict in modes:
                # Check player restriction
                if player_restrict is not None:
                    if isinstance(player_restrict, int) and num_players != player_restrict:
                        continue
                    if isinstance(player_restrict, tuple) and num_players not in player_restrict:
                        continue
                # Check model requirement
                if requires_model and not best_model:
                    continue
                valid_modes.append(mode)

            if valid_modes:
                # Rotate through valid modes
                UnifiedQueuePopulator._engine_mode_counter += 1
                engine_mode = valid_modes[UnifiedQueuePopulator._engine_mode_counter % len(valid_modes)]
            else:
                # Fallback to heuristic if no valid modes
                engine_mode = "heuristic-only"

        config = {
            "board_type": board_type,
            "num_players": num_players,
            "games": self.config.selfplay_games_per_item,
            "source": "queue_populator",
            "engine_mode": engine_mode,
            # Dec 30, 2025: Prevent coordinator/CPU nodes from claiming selfplay
            "requires_gpu": True,
        }

        if best_model and model_elo >= 1600:
            config["model_id"] = best_model
            config["model_elo"] = model_elo

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.SELFPLAY,
            priority=self.config.selfplay_priority,
            config=config,
        )

    def _is_training_ready(
        self, board_type: str, num_players: int, min_samples: int = 5000
    ) -> tuple[bool, int]:
        """Check if training data is available for a config.

        Dec 31, 2025: Added to prevent adding TRAINING work items when no
        training data exists. This was causing training jobs to complete
        instantly with loss=0.0000 because nodes had nothing to train on.

        Args:
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)
            min_samples: Minimum samples required for training (default 5000)

        Returns:
            Tuple of (is_ready, sample_count). is_ready is True if sufficient
            training data exists.
        """
        config_key = f"{board_type}_{num_players}p"

        try:
            from app.distributed.data_catalog import DataCatalog

            catalog = DataCatalog()
            npz_sources = catalog.discover_npz_files(
                board_type=board_type,
                num_players=num_players,
                min_samples=min_samples,
            )

            if npz_sources:
                total_samples = sum(s.sample_count for s in npz_sources)
                if total_samples >= min_samples:
                    return True, total_samples

            # Also check TrainingTriggerDaemon state if available
            try:
                from app.coordination.training_trigger_daemon import TrainingTriggerDaemon

                daemon = TrainingTriggerDaemon.get_instance_if_exists()
                if daemon:
                    state = daemon._training_states.get(config_key)
                    if state and state.npz_sample_count >= min_samples:
                        return True, state.npz_sample_count
            except (ImportError, AttributeError):
                pass

            return False, 0
        except (ImportError, OSError, AttributeError) as e:
            logger.debug(f"Training readiness check failed for {config_key}: {e}")
            return False, 0

    def _create_training_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a training work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"training_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        is_hex = board_type.startswith("hex")

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TRAINING,
            priority=self.config.training_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "source": "queue_populator",
                "enable_augmentation": True,
                "use_integrated_enhancements": True,
                "augment_hex_symmetry": is_hex,
            },
        )

    def _create_tournament_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a tournament work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"tournament_{board_type}_{num_players}p_{int(time.time() * 1000)}"
        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TOURNAMENT,
            priority=self.config.tournament_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "games": self.config.tournament_games,
                "source": "queue_populator",
            },
        )

    def _create_sweep_item(
        self,
        board_type: str,
        num_players: int,
        base_model_id: str,
        base_elo: float,
    ) -> "WorkItem":
        """Create a hyperparameter sweep work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"sweep_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        if base_elo >= 1900:
            strategy = "bayesian"
            trials = 20
        else:
            strategy = "random"
            trials = 30

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.HYPERPARAM_SWEEP,
            priority=60,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "base_model_id": base_model_id,
                "base_elo": base_elo,
                "strategy": strategy,
                "trials": trials,
                "source": "queue_populator",
                "search_params": ["learning_rate", "batch_size", "weight_decay"],
            },
        )

    # =========================================================================
    # Main Population Logic
    # =========================================================================

    def populate(self) -> int:
        """Populate the work queue to maintain minimum depth.

        Returns:
            Number of items added
        """
        if not self.config.enabled:
            return 0

        if self._work_queue is None:
            logger.warning("No work queue set, cannot populate")
            return 0

        if self.all_targets_met():
            # Phase 1.2: Even when all targets met, add exploration work for stale configs
            # This prevents cluster idling and maintains training data diversity
            exploration_added = self._populate_exploration_work()
            if exploration_added > 0:
                logger.info(f"All Elo targets met - added {exploration_added} exploration items")
                return exploration_added
            logger.info("All Elo targets met, no population needed")
            return 0

        # Check backpressure
        bp_level, reduction_factor = self._check_backpressure()
        if bp_level.should_stop():
            # Phase 15.1.2: Trickle mode - never completely stop population
            # This prevents the pipeline from starving when backpressure is high
            if self.config.trickle_mode_enabled:
                logger.warning(
                    f"[QueuePopulator] Backpressure {bp_level.value} - TRICKLE MODE: "
                    f"adding {self.config.trickle_min_items} items to prevent starvation"
                )
                return self._populate_trickle_items()
            else:
                logger.info(
                    f"[QueuePopulator] Backpressure {bp_level.value} - skipping population"
                )
                return 0

        items_needed = self.calculate_items_needed()
        if items_needed <= 0:
            return 0

        # Apply backpressure reduction
        if reduction_factor < 1.0:
            original_needed = items_needed
            items_needed = max(1, int(items_needed * reduction_factor))
            logger.info(
                f"[QueuePopulator] Backpressure {bp_level.value}: {original_needed} -> {items_needed}"
            )

        # Apply cluster health factor
        if self._cluster_health_factor < 1.0:
            original = items_needed
            items_needed = max(1, int(items_needed * self._cluster_health_factor))
            logger.debug(
                f"[QueuePopulator] Cluster health {self._cluster_health_factor:.2f}: "
                f"{original} -> {items_needed}"
            )

        # Get scheduler priorities
        scheduler_priorities = self._get_scheduler_priorities()

        # Calculate distribution
        selfplay_count = int(items_needed * self.config.selfplay_ratio)
        training_count = int(items_needed * self.config.training_ratio)
        tournament_count = items_needed - selfplay_count - training_count

        # Get unmet targets sorted by priority
        unmet = self.get_unmet_targets()
        if not unmet:
            return 0

        if scheduler_priorities:
            unmet.sort(
                key=lambda t: scheduler_priorities.get(t.config_key, 0.0),
                reverse=True,
            )

        added = 0

        # Add selfplay items
        for i in range(selfplay_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                target.pending_selfplay_count += 1
                added += 1
            except Exception as e:
                logger.error(f"Failed to add selfplay item: {e}")

        # Add training items (only if training data exists)
        # Dec 31, 2025: Check training readiness before adding TRAINING work.
        # Previously, training items were added blindly at a 30% ratio, causing
        # jobs to complete instantly with loss=0.0000 when no data existed.
        training_added = 0
        training_skipped = 0
        for i in range(training_count):
            target = unmet[i % len(unmet)]
            try:
                # Check if training data exists before creating work item
                is_ready, sample_count = self._is_training_ready(
                    target.board_type, target.num_players
                )
                if not is_ready:
                    # No training data - skip this config and add selfplay instead
                    training_skipped += 1
                    logger.debug(
                        f"[QueuePopulator] Skipping training for {target.config_key}: "
                        f"insufficient data ({sample_count} samples)"
                    )
                    # Add selfplay item instead to generate more training data
                    try:
                        selfplay_item = self._create_selfplay_item(
                            target.board_type, target.num_players
                        )
                        if scheduler_priorities:
                            selfplay_item.priority = self._compute_work_priority(
                                selfplay_item.priority, target.config_key, scheduler_priorities
                            )
                        self._work_queue.add_work(selfplay_item)
                        self._queued_work_ids.add(selfplay_item.work_id)
                        target.pending_selfplay_count += 1
                        added += 1
                    except Exception as sp_err:
                        logger.error(f"Failed to add replacement selfplay item: {sp_err}")
                    continue

                item = self._create_training_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
                training_added += 1
            except Exception as e:
                logger.error(f"Failed to add training item: {e}")

        if training_skipped > 0:
            logger.info(
                f"[QueuePopulator] Training skipped for {training_skipped} configs "
                f"(no data), added {training_added} training + {training_skipped} extra selfplay"
            )

        # Add tournament items
        for i in range(tournament_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_tournament_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add tournament item: {e}")

        # Add hyperparameter sweeps opportunistically
        sweep_added = 0
        for target in unmet:
            if target.current_best_elo >= 1600 and target.best_model_id:
                sweep_key = f"sweep_{target.board_type}_{target.num_players}p_"
                if not any(wid.startswith(sweep_key) for wid in self._queued_work_ids):
                    try:
                        item = self._create_sweep_item(
                            target.board_type,
                            target.num_players,
                            target.best_model_id,
                            target.current_best_elo,
                        )
                        self._work_queue.add_work(item)
                        self._queued_work_ids.add(item.work_id)
                        added += 1
                        sweep_added += 1
                        break
                    except Exception as e:
                        logger.error(f"Failed to add sweep item: {e}")

        # Phase 1.2: Also add exploration work for stale configs
        # This ensures diversity even when focusing on unmet targets
        exploration_added = self._populate_exploration_work()
        added += exploration_added

        self._last_populate_time = time.time()
        # Dec 31, 2025: Show actual training items added vs planned
        # Training items may be skipped if no training data exists
        logger.info(
            f"Populated queue with {added} items "
            f"(selfplay={selfplay_count + training_skipped}, training={training_added}, "
            f"tournament={tournament_count}, sweeps={sweep_added}, exploration={exploration_added})"
        )

        return added

    def populate_queue(self) -> int:
        """Backward-compatible alias for populate()."""
        return self.populate()

    def _populate_exploration_work(self) -> int:
        """Add exploration work for stale configs (Phase 1.2 - Jan 2026).

        This ensures the cluster never completely idles, even when all Elo
        targets are met. It maintains training data diversity by exploring
        configs that haven't had recent activity.

        Returns:
            Number of exploration items added
        """
        if self._work_queue is None:
            return 0

        # Get stale configs that need exploration
        stale_configs = self.get_least_recent_configs(EXPLORATION_CONFIGS_PER_CYCLE)
        if not stale_configs:
            return 0

        added = 0
        for target in stale_configs:
            pending_games = self.get_pending_selfplay_games(target.config_key)

            if pending_games >= MINIMUM_EXPLORATION_GAMES:
                # Already have enough pending work
                continue

            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Slightly boost priority for exploration items
                item.priority = self.config.selfplay_priority + 10
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                target.pending_selfplay_count += 1
                added += 1
            except Exception as e:
                logger.error(f"[Exploration] Failed to add item for {target.config_key}: {e}")

        if added > 0:
            logger.info(
                f"[Exploration] Added {added} exploration items for stale configs: "
                f"{', '.join(t.config_key for t in stale_configs[:added])}"
            )

        return added

    def _populate_trickle_items(self) -> int:
        """Add minimal items under extreme backpressure (Phase 15.1.2).

        This prevents complete pipeline starvation when backpressure is at
        CRITICAL or STOP levels. We add a small number of selfplay items
        focusing on the highest priority config.

        Returns:
            Number of items added (always <= trickle_min_items)
        """
        if self._work_queue is None:
            return 0

        unmet = self.get_unmet_targets()
        if not unmet:
            return 0

        # Sort by curriculum weight (highest priority first)
        scheduler_priorities = self._get_scheduler_priorities()
        if scheduler_priorities:
            unmet.sort(
                key=lambda t: scheduler_priorities.get(t.config_key, 0.0),
                reverse=True,
            )
        else:
            unmet.sort(key=lambda t: t.curriculum_weight, reverse=True)

        added = 0
        items_to_add = min(self.config.trickle_min_items, len(unmet))

        # Add selfplay items for highest priority configs only
        for i in range(items_to_add):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Boost priority for trickle items to ensure they get processed
                item.priority = self.config.selfplay_priority + 50
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"[TrickleMode] Failed to add item: {e}")

        if added > 0:
            logger.info(f"[TrickleMode] Added {added} emergency items to prevent starvation")

        return added

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
        """Background loop to periodically populate queue."""
        while self._running:
            try:
                self._populator.populate()
                await asyncio.sleep(self._populator.config.check_interval_seconds)

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
                config_key = f"{board_type}_{num_players}p"
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
                        self._populator._work_queue.add_work(item)
                        self._populator._queued_work_ids.add(item.work_id)
                        added += 1
                    except (ValueError, KeyError, AttributeError) as e:
                        logger.debug(f"[QueuePopulator] Failed to create work item: {e}")

                if added > 0:
                    logger.info(
                        f"[QueuePopulator] Queued {added} priority selfplay for {config_key}"
                    )

            def _on_work_failed(event: Any) -> None:
                """Handle WORK_FAILED - decrement pending count for failed work."""
                payload = _extract_payload(event)
                work_type = payload.get("work_type")
                if work_type != "selfplay":
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                reason = payload.get("reason", "unknown")
                config_key = f"{board_type}_{num_players}p" if board_type and num_players else ""

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
                work_type = payload.get("work_type")
                if work_type != "selfplay":
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                node_id = payload.get("node_id", "unknown")
                config_key = f"{board_type}_{num_players}p" if board_type and num_players else ""

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
                work_id = payload.get("work_id")
                work_type = payload.get("work_type")

                # Remove from tracking set regardless of work type
                if work_id and work_id in self._populator._queued_work_ids:
                    self._populator._queued_work_ids.discard(work_id)
                    logger.debug(
                        f"[QueuePopulator] Work completed: {work_id}, "
                        f"remaining tracked: {len(self._populator._queued_work_ids)}"
                    )

                # For selfplay work, also update pending counts
                if work_type == "selfplay":
                    board_type = payload.get("board_type")
                    num_players = payload.get("num_players")
                    config_key = f"{board_type}_{num_players}p" if board_type and num_players else ""

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
                task_type = payload.get("task_type", "")
                if "selfplay" not in task_type.lower():
                    return

                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                reason = payload.get("reason", "unknown")
                config_key = f"{board_type}_{num_players}p" if board_type and num_players else ""

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
        selfplay_priority=populator.get("selfplay_priority", 50),
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
