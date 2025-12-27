"""
Work queue populator to maintain minimum queue depth until Elo targets are met.

DEPRECATED (December 2025): Use unified_queue_populator.py instead.

This module is deprecated in favor of UnifiedQueuePopulator which consolidates
this file with queue_populator_daemon.py for a ~500 LOC reduction.

Migration:
    # Old
    from app.coordination.queue_populator import QueuePopulator, get_queue_populator

    # New
    from app.coordination.unified_queue_populator import (
        UnifiedQueuePopulator,
        get_queue_populator,
    )

The unified version provides:
- Same Elo-based target tracking and velocity calculations
- Async daemon lifecycle support
- P2P cluster health integration
- Curriculum-weighted prioritization
- All existing functionality

This module remains for backward compatibility and will be removed in Q2 2026.

Work distribution:
- 60% selfplay (data generation)
- 30% training (model improvement)
- 10% tournament (Elo measurement)
"""

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

# Emit deprecation warning on import
warnings.warn(
    "queue_populator.py is deprecated. Use unified_queue_populator.py instead. "
    "See module docstring for migration guide. Will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Canonical types (December 2025 consolidation)
from app.coordination.types import BackpressureLevel

if TYPE_CHECKING:
    from app.coordination.selfplay_scheduler import SelfplayScheduler
    from app.coordination.work_queue import WorkItem, WorkQueue

logger = logging.getLogger(__name__)


class BoardType(str, Enum):
    """Board types to train."""
    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEX8 = "hex8"
    HEXAGONAL = "hexagonal"


# Large boards require specialized engine selection (Gumbel MCTS)
# These have many more cells (361-469) and benefit from quality over quantity
LARGE_BOARDS = frozenset({"square19", "hexagonal", "fullhex", "full_hex"})


@dataclass
class ConfigTarget:
    """Target for a specific board/player configuration."""
    board_type: str
    num_players: int
    target_elo: float = 2000.0
    current_best_elo: float = 1500.0
    best_model_id: str | None = None
    games_played: int = 0
    training_runs: int = 0
    last_updated: float = field(default_factory=time.time)
    # Elo history for velocity tracking: list of (timestamp, elo) tuples
    elo_history: list[tuple] = field(default_factory=list)
    # P10-LOOP-3: Track previous velocity for change detection
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
        """Calculate Elo velocity in points per day.

        Uses linear regression on recent history (last 7 days).
        Returns 0 if insufficient data.
        """
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
        return slope * 86400  # points per day

    @property
    def days_to_target(self) -> float | None:
        """Estimate days to reach target at current velocity.

        Returns None if target already met or velocity is zero/negative.
        """
        if self.target_met:
            return 0.0

        velocity = self.elo_velocity
        if velocity <= 0:
            return None  # Not progressing

        return self.elo_gap / velocity

    def record_elo(self, elo: float, timestamp: float | None = None) -> None:
        """Record an Elo measurement for velocity tracking.

        P10-LOOP-3 (Dec 2025): Also emits ELO_VELOCITY_CHANGED event if
        velocity changes significantly (>10 Elo/day difference).
        """
        ts = timestamp or time.time()
        self.elo_history.append((ts, elo))

        # Keep only last 30 days of history to prevent unbounded growth
        cutoff = ts - (30 * 24 * 3600)
        self.elo_history = [(t, e) for t, e in self.elo_history if t >= cutoff]

        # P10-LOOP-3: Check for significant velocity change
        new_velocity = self.elo_velocity
        velocity_change = abs(new_velocity - self._previous_velocity)

        # Only emit if velocity change is significant (>10 Elo/day)
        if velocity_change > 10.0:
            # Determine trend
            if new_velocity > self._previous_velocity + 5:
                trend = "accelerating"
            elif new_velocity < self._previous_velocity - 5:
                trend = "decelerating"
            else:
                trend = "stable"

            # Emit event asynchronously
            try:
                import asyncio
                from app.coordination.event_router import emit_elo_velocity_changed

                async def _emit():
                    await emit_elo_velocity_changed(
                        config_key=self.config_key,
                        velocity=new_velocity,
                        previous_velocity=self._previous_velocity,
                        trend=trend,
                    )

                try:
                    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                    loop = asyncio.get_running_loop()
                    loop.create_task(_emit())
                except RuntimeError:
                    # No running loop - create one
                    try:
                        asyncio.run(_emit())
                    except Exception as e:
                        logger.debug(f"Failed to emit velocity change event: {e}")

                logger.debug(
                    f"[ConfigTarget] Velocity changed for {self.config_key}: "
                    f"{self._previous_velocity:.1f} → {new_velocity:.1f} Elo/day ({trend})"
                )
            except ImportError:
                pass  # data_events not available

            self._previous_velocity = new_velocity


@dataclass
class PopulatorConfig:
    """Configuration for the queue populator."""
    # Minimum queue depth to maintain
    min_queue_depth: int = 50

    # Target Elo for all configurations
    target_elo: float = 2000.0

    # Work distribution (must sum to 1.0)
    selfplay_ratio: float = 0.60
    training_ratio: float = 0.30
    tournament_ratio: float = 0.10

    # Board types to train
    board_types: list[str] = field(default_factory=lambda: [
        "square8", "square19", "hex8", "hexagonal"
    ])

    # Player counts to train
    player_counts: list[int] = field(default_factory=lambda: [2, 3, 4])

    # Selfplay settings
    selfplay_games_per_item: int = 50
    selfplay_priority: int = 50

    # Training settings
    training_priority: int = 100
    min_games_for_training: int = 300

    # Tournament settings
    tournament_games: int = 50
    tournament_priority: int = 80

    # Enabled flag
    enabled: bool = True

    # Check interval (Dec 2025: Reduced from 60s for faster job allocation)
    check_interval_seconds: int = 15


class QueuePopulator:
    """
    Maintains minimum work queue depth until Elo targets are met.

    Automatically populates the work queue with selfplay, training, and
    tournament work items to ensure continuous progress toward the
    target Elo rating for each board/player configuration.
    """

    def __init__(
        self,
        config: PopulatorConfig | None = None,
        work_queue: Optional["WorkQueue"] = None,
        elo_db_path: str | None = None,
        selfplay_scheduler: Optional["SelfplayScheduler"] = None,
    ):
        self.config = config or PopulatorConfig()
        self._work_queue = work_queue
        self._elo_db_path = elo_db_path
        self._selfplay_scheduler = selfplay_scheduler

        # Scale queue depth to cluster size (Phase 2B.1 - December 2025)
        self._scale_queue_depth_to_cluster()

        # Track configuration targets
        self._targets: dict[str, ConfigTarget] = {}
        self._init_targets()

        # Load existing Elo ratings from database
        self._load_existing_elo()

        # Track what we've queued
        self._queued_work_ids: set[str] = set()
        self._last_populate_time: float = 0

    def _scale_queue_depth_to_cluster(self) -> None:
        """Scale min_queue_depth based on cluster size.

        December 2025 - Phase 2B.1: Dynamic queue scaling to prevent
        queue starvation with large clusters (43+ nodes).

        Target: 2 items per active node, minimum 50.
        """
        try:
            from app.distributed.cluster_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            # Quick check without expensive operations
            status = monitor.get_cluster_status(
                include_game_counts=False,
                include_training_status=False,
                include_disk_usage=False,
            )
            active_nodes = status.active_nodes

            if active_nodes > 0:
                # 2 items per node minimum, floor of 50
                scaled_depth = max(50, active_nodes * 2)
                old_depth = self.config.min_queue_depth
                self.config.min_queue_depth = scaled_depth
                logger.info(
                    f"[QueuePopulator] Scaled queue depth: {old_depth} -> {scaled_depth} "
                    f"(for {active_nodes} active nodes)"
                )
            else:
                logger.debug("[QueuePopulator] No active nodes detected, using default queue depth")

        except ImportError:
            logger.debug("[QueuePopulator] ClusterMonitor not available, using default queue depth")
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
        """Load existing Elo ratings from the database.

        Queries the unified_elo.db to get the best current Elo for each
        board/player configuration, so the populator knows the actual
        starting point rather than assuming 1500 for everything.
        """
        import sqlite3
        from pathlib import Path

        # Find the Elo database
        if self._elo_db_path:
            db_path = Path(self._elo_db_path)
        else:
            # Try common locations
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

            # Get best Elo per board_type/num_players combination
            # Use subquery to get the correct model_id for the max rating
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
                    # Seed velocity tracking with initial measurement
                    target.record_elo(best_elo)
                    logger.info(
                        f"Loaded existing Elo for {key}: {best_elo:.1f} "
                        f"(model: {model_id}, games: {games})"
                    )

            # Log summary
            met = sum(1 for t in self._targets.values() if t.target_met)
            logger.info(
                f"Loaded Elo data: {met}/{len(self._targets)} configs already at target "
                f"({self.config.target_elo}+ Elo)"
            )

        except Exception as e:
            logger.warning(f"Failed to load existing Elo data: {e}")

    def set_work_queue(self, work_queue: "WorkQueue") -> None:
        """Set the work queue reference."""
        self._work_queue = work_queue

    def set_selfplay_scheduler(self, scheduler: "SelfplayScheduler") -> None:
        """Set the selfplay scheduler reference for priority-based allocation.

        When a scheduler is set, the queue populator will use scheduler
        priorities to determine work item priorities and config ordering.
        """
        self._selfplay_scheduler = scheduler
        logger.info("[QueuePopulator] SelfplayScheduler integration enabled")

    def _get_scheduler_priorities(self) -> dict[str, float]:
        """Get priority scores from the SelfplayScheduler.

        Returns:
            Dict mapping config_key -> priority_score (higher = more priority).
            Empty dict if scheduler not available or async call fails.
        """
        if self._selfplay_scheduler is None:
            return {}

        try:
            # Check if we're in a running event loop (Dec 2025: use get_running_loop)
            try:
                loop = asyncio.get_running_loop()
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
        except Exception as e:
            logger.debug(f"[QueuePopulator] Failed to get scheduler priorities: {e}")
            return {}

    def _compute_work_priority(
        self,
        base_priority: int,
        config_key: str,
        scheduler_priorities: dict[str, float],
    ) -> int:
        """Compute adjusted work priority based on scheduler priorities.

        Args:
            base_priority: Base priority from config
            config_key: Config key like "hex8_2p"
            scheduler_priorities: Scheduler priority scores

        Returns:
            Adjusted priority (higher = more urgent)
        """
        if not scheduler_priorities:
            return base_priority

        scheduler_score = scheduler_priorities.get(config_key, 0.0)
        if scheduler_score <= 0:
            return base_priority

        # Scale scheduler score (typically 0-10) to priority boost (0-50)
        # Top priority configs get +50 boost
        max_score = max(scheduler_priorities.values()) if scheduler_priorities else 1.0
        normalized = scheduler_score / max(max_score, 0.01)
        priority_boost = int(normalized * 50)

        return base_priority + priority_boost

    def _check_backpressure(self) -> tuple[BackpressureLevel, float]:
        """Check current backpressure level from queue and resource monitors.

        Returns:
            Tuple of (backpressure_level, reduction_factor)
            where reduction_factor is 0.0-1.0 (1.0 = no reduction)
        """
        try:
            # Check queue monitor for queue depth backpressure
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

        # Also check resource coordinator if available
        try:
            from app.coordination.resource_monitoring_coordinator import (
                get_resource_coordinator,
            )

            coordinator = get_resource_coordinator()
            if coordinator and coordinator.is_backpressure_active():
                status = coordinator.get_status()
                bp_level_str = status.get("backpressure_level", "none")
                bp_level = BackpressureLevel(bp_level_str)
                return bp_level, bp_level.reduction_factor()

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[QueuePopulator] Resource backpressure check failed: {e}")

        return BackpressureLevel.NONE, 1.0

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

                # Record for velocity tracking
                target.record_elo(elo)

                # Log with velocity info
                velocity = target.elo_velocity
                eta = target.days_to_target
                eta_str = f"{eta:.1f} days" if eta else "N/A"
                logger.info(
                    f"Updated {key} best Elo: {elo:.1f} "
                    f"(gap: {target.elo_gap:.1f}, velocity: {velocity:+.1f}/day, "
                    f"ETA: {eta_str}, model: {model_id})"
                )

    def increment_games(self, board_type: str, num_players: int, count: int = 1) -> None:
        """Increment games played for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            self._targets[key].games_played += count

    def increment_training(self, board_type: str, num_players: int) -> None:
        """Increment training runs for a configuration."""
        key = f"{board_type}_{num_players}p"
        if key in self._targets:
            self._targets[key].training_runs += 1

    def all_targets_met(self) -> bool:
        """Check if all configurations have reached target Elo."""
        return all(t.target_met for t in self._targets.values())

    def get_unmet_targets(self) -> list[ConfigTarget]:
        """Get configurations that haven't reached target Elo."""
        return [t for t in self._targets.values() if not t.target_met]

    def get_priority_target(self) -> ConfigTarget | None:
        """Get the configuration that needs the most attention.

        Prioritizes by:
        1. Largest Elo gap (furthest from target)
        2. Fewest games played (needs more data)
        """
        unmet = self.get_unmet_targets()
        if not unmet:
            return None

        # Sort by Elo gap (smallest first) to prioritize configs closest to target
        # This gets us to 2000 Elo faster by focusing resources
        unmet.sort(key=lambda t: (t.elo_gap, -t.games_played))
        return unmet[0]

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

    def _create_selfplay_item(
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Create a selfplay work item.

        Uses the best model for this config if available (1700+ Elo preferred).
        This ensures selfplay generates high-quality training data.
        """
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"selfplay_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        # Get best model for this config
        key = f"{board_type}_{num_players}p"
        target = self._targets.get(key)
        best_model = target.best_model_id if target else None
        model_elo = target.current_best_elo if target else 1500.0

        # Engine selection:
        # - Large boards (sq19, hexagonal): ALWAYS use gumbel_mcts for quality
        # - Small boards with good model (1600+ Elo): use nnue-guided
        # - Small boards without good model: use gpu_heuristic
        if board_type in LARGE_BOARDS:
            # Large boards need quality selfplay - Gumbel MCTS is 100% parity verified
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

        # Include model info for NN-guided selfplay
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
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Create a training work item with symmetry augmentation enabled."""
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"training_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        # Determine symmetry type based on board
        # D4 for square boards (8 transforms), D6 for hex (12 transforms)
        is_hex = board_type.startswith("hex")

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.TRAINING,
            priority=self.config.training_priority,
            config={
                "board_type": board_type,
                "num_players": num_players,
                "source": "queue_populator",
                # Enable symmetry augmentation for 8-12x data multiplier
                "enable_augmentation": True,
                "use_integrated_enhancements": True,
                "augment_hex_symmetry": is_hex,
            },
        )

    def _create_tournament_item(
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
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
        """Create a hyperparameter sweep work item for fine-tuning a high-Elo model.

        Triggers automated hyperparameter search to push models past plateaus.
        Only used for models at 1800+ Elo where fine-tuning can yield gains.
        """
        from app.coordination.work_queue import WorkItem, WorkType

        work_id = f"sweep_{board_type}_{num_players}p_{int(time.time() * 1000)}"

        # Determine sweep strategy based on current Elo
        # Higher Elo = more focused search (smaller ranges)
        if base_elo >= 1900:
            strategy = "bayesian"  # Efficient for fine-tuning
            trials = 20
        else:
            strategy = "random"  # Broader exploration
            trials = 30

        return WorkItem(
            work_id=work_id,
            work_type=WorkType.HYPERPARAM_SWEEP,
            priority=60,  # Lower than training, runs opportunistically
            config={
                "board_type": board_type,
                "num_players": num_players,
                "base_model_id": base_model_id,
                "base_elo": base_elo,
                "strategy": strategy,
                "trials": trials,
                "source": "queue_populator",
                # Focus on learning rate and batch size for fine-tuning
                "search_params": ["learning_rate", "batch_size", "weight_decay"],
            },
        )

    def populate(self) -> int:
        """Populate the work queue to maintain minimum depth.

        Uses SelfplayScheduler priorities (if available) to:
        1. Order configs by priority (staleness, velocity, training needs)
        2. Boost work item priorities for high-priority configs

        Respects backpressure signals from queue and resource monitors:
        - STOP/CRITICAL: Add no items
        - HARD/HIGH: Add only 10% of normal items
        - SOFT/MEDIUM: Add 25-50% of normal items
        - LOW: Add 75% of normal items
        - NONE: Add full amount

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

        # Check backpressure before populating (December 2025)
        bp_level, reduction_factor = self._check_backpressure()
        if bp_level.should_stop():
            logger.info(
                f"[QueuePopulator] Backpressure {bp_level.value} active - skipping population"
            )
            return 0

        items_needed = self.calculate_items_needed()
        if items_needed <= 0:
            return 0

        # Apply backpressure reduction (December 2025)
        if reduction_factor < 1.0:
            original_needed = items_needed
            items_needed = max(1, int(items_needed * reduction_factor))
            logger.info(
                f"[QueuePopulator] Backpressure {bp_level.value}: reduced items "
                f"{original_needed} -> {items_needed} (factor={reduction_factor:.2f})"
            )

        # Get scheduler priorities for intelligent ordering (December 2025)
        scheduler_priorities = self._get_scheduler_priorities()
        if scheduler_priorities:
            logger.debug(
                f"[QueuePopulator] Using scheduler priorities for {len(scheduler_priorities)} configs"
            )

        # Calculate distribution
        selfplay_count = int(items_needed * self.config.selfplay_ratio)
        training_count = int(items_needed * self.config.training_ratio)
        tournament_count = items_needed - selfplay_count - training_count

        # Get unmet targets sorted by priority
        unmet = self.get_unmet_targets()
        if not unmet:
            return 0

        # Sort unmet configs by scheduler priority (December 2025 integration)
        if scheduler_priorities:
            unmet.sort(
                key=lambda t: scheduler_priorities.get(t.config_key, 0.0),
                reverse=True,  # Higher priority first
            )

        added = 0

        # Add selfplay items with priority boosting
        for i in range(selfplay_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_selfplay_item(target.board_type, target.num_players)
                # Boost priority based on scheduler (December 2025)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add selfplay item: {e}")

        # Add training items with priority boosting
        for i in range(training_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_training_item(target.board_type, target.num_players)
                # Boost priority based on scheduler (December 2025)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add training item: {e}")

        # Add tournament items with priority boosting
        for i in range(tournament_count):
            target = unmet[i % len(unmet)]
            try:
                item = self._create_tournament_item(target.board_type, target.num_players)
                # Boost priority based on scheduler (December 2025)
                if scheduler_priorities:
                    item.priority = self._compute_work_priority(
                        item.priority, target.config_key, scheduler_priorities
                    )
                self._work_queue.add_work(item)
                self._queued_work_ids.add(item.work_id)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add tournament item: {e}")

        # Opportunistically queue hyperparameter sweeps for high-Elo models
        # Only 1 sweep per populate cycle to avoid hogging GPU resources
        sweep_added = 0
        sweep_threshold = 1600  # Lower threshold to help break plateaus
        for target in unmet:
            if target.current_best_elo >= sweep_threshold and target.best_model_id:
                # Check if we already have a pending sweep for this config
                sweep_key = f"sweep_{target.board_type}_{target.num_players}p_"
                has_pending_sweep = any(
                    wid.startswith(sweep_key) for wid in self._queued_work_ids
                )
                if not has_pending_sweep:
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
                        logger.info(
                            f"Queued hyperparameter sweep for {target.board_type}_{target.num_players}p "
                            f"(model: {target.best_model_id}, Elo: {target.current_best_elo:.0f})"
                        )
                        break  # Only one sweep per cycle
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

    def get_status(self) -> dict[str, Any]:
        """Get populator status for monitoring."""
        unmet = self.get_unmet_targets()
        met = [t for t in self._targets.values() if t.target_met]

        # Calculate aggregate velocity metrics
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
            "unmet_configs": [
                {
                    "config": t.config_key,
                    "current_elo": t.current_best_elo,
                    "gap": t.elo_gap,
                    "velocity": t.elo_velocity,
                    "days_to_target": t.days_to_target,
                    "games": t.games_played,
                    "training_runs": t.training_runs,
                }
                for t in unmet
            ],
            "last_populate_time": self._last_populate_time,
            "total_queued": len(self._queued_work_ids),
        }


def load_populator_config_from_yaml(yaml_config: dict[str, Any]) -> PopulatorConfig:
    """Load PopulatorConfig from YAML configuration dict."""
    populator = yaml_config.get("queue_populator", {})

    return PopulatorConfig(
        min_queue_depth=populator.get("min_queue_depth", 50),
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
        enabled=populator.get("enabled", True),
        check_interval_seconds=populator.get("check_interval_seconds", 60),
    )


# =============================================================================
# Singleton pattern
# =============================================================================

_populator: QueuePopulator | None = None


def get_queue_populator(
    config: PopulatorConfig | None = None,
    work_queue: Optional["WorkQueue"] = None,
) -> QueuePopulator:
    """Get or create the singleton QueuePopulator instance."""
    global _populator
    if _populator is None:
        _populator = QueuePopulator(config=config, work_queue=work_queue)
    elif work_queue is not None and _populator._work_queue is None:
        _populator.set_work_queue(work_queue)
    return _populator


def reset_queue_populator() -> None:
    """Reset the singleton for testing."""
    global _populator
    _populator = None


def wire_queue_populator_events() -> QueuePopulator:
    """Wire queue populator to the event bus for automatic updates.

    Subscribes to:
    - ELO_UPDATED: Update target Elo for configurations
    - TRAINING_COMPLETED: Increment training count
    - NEW_GAMES_AVAILABLE: Increment games count
    - TRAINING_BLOCKED_BY_QUALITY: Trigger extra selfplay to regenerate data

    Returns:
        The configured QueuePopulator instance
    """
    populator = get_queue_populator()

    try:
        from app.coordination.event_router import get_router
        from app.coordination.event_router import DataEventType

        router = get_router()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_elo_updated(event: Any) -> None:
            """Handle Elo update - update target tracking."""
            payload = _event_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            elo = payload.get("elo") or payload.get("rating")
            model_id = payload.get("model_id") or payload.get("participant_id")
            if board_type and num_players and elo:
                populator.update_target_elo(board_type, num_players, elo, model_id)

        def _on_training_completed(event: Any) -> None:
            """Handle training completion - increment count."""
            payload = _event_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            if board_type and num_players:
                populator.increment_training(board_type, num_players)

        def _on_new_games(event: Any) -> None:
            """Handle new games - increment count."""
            payload = _event_payload(event)
            board_type = payload.get("board_type")
            num_players = payload.get("num_players")
            count = payload.get("count", 1)
            if board_type and num_players:
                populator.increment_games(board_type, num_players, count)

        def _on_training_blocked_by_quality(event: Any) -> None:
            """Handle training blocked by quality - add extra selfplay work items.

            Phase 7 Fix (Dec 2025): Closes the quality gate deadlock by
            immediately queueing extra selfplay work for the blocked config.

            This completes the feedback loop:
            TRAINING_BLOCKED_BY_QUALITY → Extra selfplay items → Fresh data → Training resumes
            """
            payload = _event_payload(event)
            config_key = payload.get("config_key", "") or payload.get("config", "")
            reason = payload.get("reason", "unknown")
            quality_score = payload.get("quality_score", 0.0)

            if not config_key:
                logger.debug("[QueuePopulator] TRAINING_BLOCKED_BY_QUALITY without config_key")
                return

            # Parse config_key (e.g., "hex8_2p" -> board_type="hex8", num_players=2)
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].endswith("p"):
                logger.warning(f"[QueuePopulator] Invalid config_key format: {config_key}")
                return

            board_type = parts[0]
            try:
                num_players = int(parts[1][:-1])
            except ValueError:
                logger.warning(f"[QueuePopulator] Cannot parse num_players from: {config_key}")
                return

            # Check if work queue is available
            if populator._work_queue is None:
                logger.warning("[QueuePopulator] No work queue, cannot queue extra selfplay")
                return

            # Add 3 extra selfplay items for the blocked config (150 games = 3 * 50)
            # This should generate enough fresh data to pass quality gate
            added = 0
            for _ in range(3):
                try:
                    item = populator._create_selfplay_item(board_type, num_players)
                    # Boost priority to ensure these run soon
                    item.priority = populator.config.selfplay_priority + 30
                    populator._work_queue.add_work(item)
                    populator._queued_work_ids.add(item.work_id)
                    added += 1
                except Exception as e:
                    logger.error(f"[QueuePopulator] Failed to add priority selfplay item: {e}")

            if added > 0:
                logger.info(
                    f"[QueuePopulator] Queued {added} priority selfplay items for {config_key} "
                    f"(reason: {reason}, quality: {quality_score:.2f})"
                )

        router.subscribe(DataEventType.ELO_UPDATED.value, _on_elo_updated)
        router.subscribe(DataEventType.TRAINING_COMPLETED.value, _on_training_completed)
        router.subscribe(DataEventType.NEW_GAMES_AVAILABLE.value, _on_new_games)
        router.subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY.value, _on_training_blocked_by_quality)

        logger.info(
            "[QueuePopulator] Wired to event bus "
            "(ELO_UPDATED, TRAINING_COMPLETED, NEW_GAMES_AVAILABLE, TRAINING_BLOCKED_BY_QUALITY)"
        )

    except ImportError:
        logger.warning("[QueuePopulator] data_events not available, running without event bus")

    return populator


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "BoardType",
    # Data classes
    "ConfigTarget",
    "PopulatorConfig",
    # Main class
    "QueuePopulator",
    "get_queue_populator",
    # Functions
    "load_populator_config_from_yaml",
    "reset_queue_populator",
    "wire_queue_populator_events",
]
