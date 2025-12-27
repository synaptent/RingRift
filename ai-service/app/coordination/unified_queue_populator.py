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

# Default curriculum weights (priority multipliers)
DEFAULT_CURRICULUM_WEIGHTS: dict[str, float] = {
    "square8_2p": 1.0, "square8_3p": 0.7, "square8_4p": 0.5,
    "square19_2p": 0.8, "square19_3p": 0.5, "square19_4p": 0.4,
    "hex8_2p": 0.9, "hex8_3p": 0.6, "hex8_4p": 0.5,
    "hexagonal_2p": 0.7, "hexagonal_3p": 0.5, "hexagonal_4p": 0.4,
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
    min_queue_depth: int = 50

    # Maximum pending items before stopping generation
    max_pending_items: int = 50

    # Check/scan interval (reduced from 60s for faster job allocation)
    check_interval_seconds: int = 15

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
    min_games_for_training: int = 300

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
            conn = sqlite3.connect(str(db_path))
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
            conn.close()

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
        """Calculate how many items to add to reach minimum depth."""
        current = self.get_current_queue_depth()
        needed = max(0, self.config.min_queue_depth - current)
        return needed

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

    def _create_selfplay_item(
        self, board_type: str, num_players: int
    ) -> "WorkItem":
        """Create a selfplay work item."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"selfplay_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        key = f"{board_type}_{num_players}p"
        target = self._targets.get(key)
        best_model = target.best_model_id if target else None
        model_elo = target.current_best_elo if target else 1500.0

        # Engine selection based on board size and model quality
        if board_type in LARGE_BOARDS:
            engine_mode = "gumbel"
        elif model_elo >= 1600 and best_model:
            engine_mode = "nnue-guided"
        else:
            engine_mode = "gpu_heuristic"

        config = {
            "board_type": board_type,
            "num_players": num_players,
            "games": self.config.selfplay_games_per_item,
            "source": "queue_populator",
            "engine_mode": engine_mode,
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
            logger.info("All Elo targets met, no population needed")
            return 0

        # Check backpressure
        bp_level, reduction_factor = self._check_backpressure()
        if bp_level.should_stop():
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

        # Add training items
        for i in range(training_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_training_item(target.board_type, target.num_players)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add training item: {e}")

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

        self._last_populate_time = time.time()
        logger.info(
            f"Populated queue with {added} items "
            f"(selfplay={selfplay_count}, training={training_count}, "
            f"tournament={tournament_count}, sweeps={sweep_added})"
        )

        return added

    def populate_queue(self) -> int:
        """Backward-compatible alias for populate()."""
        return self.populate()

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

            def _on_new_games(event: Any) -> None:
                payload = _extract_payload(event)
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")
                count = payload.get("count", 1)
                if board_type and num_players:
                    self._populator.increment_games(board_type, num_players, count)

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

            def _on_training_blocked(event: Any) -> None:
                """Handle TRAINING_BLOCKED_BY_QUALITY - queue extra selfplay."""
                payload = _extract_payload(event)
                config_key = payload.get("config_key", "") or payload.get("config", "")
                if not config_key:
                    return

                parts = config_key.rsplit("_", 1)
                if len(parts) != 2 or not parts[1].endswith("p"):
                    return

                board_type = parts[0]
                try:
                    num_players = int(parts[1][:-1])
                except ValueError:
                    return

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

            router.subscribe(DataEventType.ELO_UPDATED.value, _on_elo_updated)
            router.subscribe(DataEventType.TRAINING_COMPLETED.value, _on_training_completed)
            router.subscribe(DataEventType.NEW_GAMES_AVAILABLE.value, _on_new_games)
            router.subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY.value, _on_training_blocked)

            if hasattr(DataEventType, 'SELFPLAY_COMPLETE'):
                router.subscribe(DataEventType.SELFPLAY_COMPLETE.value, _on_selfplay_complete)

            # Wire WORK_FAILED, WORK_TIMEOUT, TASK_ABANDONED for accurate pending count tracking
            if hasattr(DataEventType, 'WORK_FAILED'):
                router.subscribe(DataEventType.WORK_FAILED.value, _on_work_failed)
            if hasattr(DataEventType, 'WORK_TIMEOUT'):
                router.subscribe(DataEventType.WORK_TIMEOUT.value, _on_work_timeout)
            if hasattr(DataEventType, 'TASK_ABANDONED'):
                router.subscribe(DataEventType.TASK_ABANDONED.value, _on_task_abandoned)

            _events_wired = True
            logger.info("[QueuePopulatorDaemon] Subscribed to data events (incl. WORK_FAILED/TIMEOUT/TASK_ABANDONED)")

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
        min_queue_depth=populator.get("min_queue_depth", 50),
        max_pending_items=populator.get("max_pending_items", 50),
        check_interval_seconds=populator.get("check_interval_seconds", 15),
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
