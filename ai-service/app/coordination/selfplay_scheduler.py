"""SelfplayScheduler - Priority-based selfplay allocation across cluster.

This module provides intelligent scheduling of selfplay jobs across the cluster,
prioritizing configurations based on:
- Data staleness (fresher data gets lower priority)
- ELO improvement velocity (fast-improving configs get more resources)
- Training pipeline needs (configs waiting for training get boosted)
- Node capabilities (allocate based on GPU power)

Architecture:
    SelfplayScheduler
    ├── TrainingFreshness: Track data age per config
    ├── ELO Velocity: Track improvement rate
    ├── Node Allocator: Distribute based on capabilities
    └── Event Integration: React to pipeline events

Usage:
    from app.coordination.selfplay_scheduler import (
        SelfplayScheduler,
        get_selfplay_scheduler,
    )

    scheduler = get_selfplay_scheduler()

    # Get priority-ordered configs for selfplay
    priorities = await scheduler.get_priority_configs()

    # Allocate games across nodes
    allocation = await scheduler.allocate_selfplay_batch(games_per_config=500)

    # Start jobs based on allocation
    for config, nodes in allocation.items():
        for node_id, num_games in nodes.items():
            await start_selfplay_job(node_id, config, num_games)

December 2025: Created as part of strategic integration plan.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "ConfigPriority",
    "NodeCapability",
    "SelfplayScheduler",
    # Constants
    "ALL_CONFIGS",
    "CURRICULUM_WEIGHT",
    "DATA_DEFICIT_WEIGHT",
    "DEFAULT_GAMES_PER_CONFIG",
    "ELO_VELOCITY_WEIGHT",
    "EXPLORATION_BOOST_WEIGHT",
    "FRESH_DATA_THRESHOLD",
    "IMPROVEMENT_BOOST_WEIGHT",
    "LARGE_BOARD_TARGET_MULTIPLIER",
    "MAX_STALENESS_HOURS",
    "MIN_GAMES_PER_ALLOCATION",
    "PRIORITY_OVERRIDE_MULTIPLIERS",
    "STALE_DATA_THRESHOLD",
    "STALENESS_WEIGHT",
    "TRAINING_NEED_WEIGHT",
    # Functions
    "get_selfplay_scheduler",
    "reset_selfplay_scheduler",
    # New Dec 2025
    "get_priority_configs_sync",
]

import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

from app.config.thresholds import (
    SELFPLAY_GAMES_PER_NODE,
    is_ephemeral_node,
    get_gpu_weight,
)
from app.coordination.protocols import HealthCheckResult

# Import interfaces for type hints (no circular dependency)
from app.coordination.interfaces import IBackpressureMonitor

# Note: backpressure concrete import moved to lazy loading in allocate_selfplay_batch()
# to break circular dependency cycle (Dec 2025). The IBackpressureMonitor protocol
# from interfaces allows type hints without importing the concrete class.

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# All supported configurations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Priority calculation weights
STALENESS_WEIGHT = 0.30  # Data freshness importance
ELO_VELOCITY_WEIGHT = 0.20  # ELO improvement velocity
TRAINING_NEED_WEIGHT = 0.10  # Waiting for training
EXPLORATION_BOOST_WEIGHT = 0.10  # Feedback loop exploration signal
CURRICULUM_WEIGHT = 0.10  # Curriculum-based priority (Phase 2C.3)
IMPROVEMENT_BOOST_WEIGHT = 0.15  # Phase 5: ImprovementOptimizer boost
DATA_DEFICIT_WEIGHT = 0.25  # Dec 2025: Boost configs with low game counts

# Target games per config for data deficit calculation
TARGET_GAMES_FOR_2000_ELO = 100000  # Need 100K games for strong AI
LARGE_BOARD_TARGET_MULTIPLIER = 1.5  # Large boards need more data

# Priority override multipliers (Dec 2025)
# Maps priority level (0-3) to score multiplier
# 0 = CRITICAL, 1 = HIGH, 2 = MEDIUM, 3 = LOW
PRIORITY_OVERRIDE_MULTIPLIERS = {
    0: 3.0,  # CRITICAL: 3x boost for critically data-starved configs
    1: 2.0,  # HIGH: 2x boost
    2: 1.25,  # MEDIUM: 25% boost
    3: 1.0,  # LOW: no boost (normal priority)
}

# Staleness thresholds (hours)
FRESH_DATA_THRESHOLD = 1.0  # Data < 1hr old is fresh
STALE_DATA_THRESHOLD = 4.0  # Data > 4hr old is stale
MAX_STALENESS_HOURS = 24.0  # Cap staleness factor

# Default allocation (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import SelfplayAllocationDefaults

DEFAULT_GAMES_PER_CONFIG = SelfplayAllocationDefaults.GAMES_PER_CONFIG
MIN_GAMES_PER_ALLOCATION = SelfplayAllocationDefaults.MIN_GAMES_PER_ALLOCATION

# Resource management thresholds (for get_target_jobs_for_node)
MIN_MEMORY_GB_FOR_TASKS = SelfplayAllocationDefaults.MIN_MEMORY_GB
DISK_WARNING_THRESHOLD = SelfplayAllocationDefaults.DISK_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = SelfplayAllocationDefaults.MEMORY_WARNING_THRESHOLD


@dataclass
class ConfigPriority:
    """Priority information for a configuration."""
    config_key: str

    # Priority factors
    staleness_hours: float = 0.0
    elo_velocity: float = 0.0  # ELO points per day
    training_pending: bool = False
    exploration_boost: float = 1.0
    exploration_boost_expires_at: float = 0.0  # Phase 12: When boost decays back to 1.0
    curriculum_weight: float = 1.0  # Phase 2C.3: Curriculum-based weight
    improvement_boost: float = 0.0  # Phase 5: From ImprovementOptimizer (-0.10 to +0.15)
    quality_penalty: float = 0.0  # Phase 5: Quality degradation penalty (0.0 to -0.20)
    momentum_multiplier: float = 1.0  # Phase 19: From FeedbackAccelerator (0.5 to 1.5)
    game_count: int = 0  # Dec 2025: Current game count for this config
    is_large_board: bool = False  # Dec 2025: True for square19, hexagonal
    priority_override: int = 3  # Dec 2025: From config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
    search_budget: int = 400  # Dec 28 2025: Gumbel MCTS budget from velocity feedback

    # Computed priority
    priority_score: float = 0.0

    # Allocation
    games_allocated: int = 0
    nodes_allocated: list[str] = field(default_factory=list)

    @property
    def staleness_factor(self) -> float:
        """Compute staleness factor (0-1, higher = more stale)."""
        if self.staleness_hours < FRESH_DATA_THRESHOLD:
            return 0.0
        if self.staleness_hours > MAX_STALENESS_HOURS:
            return 1.0
        # Linear interpolation
        return min(1.0, (self.staleness_hours - FRESH_DATA_THRESHOLD) /
                   (STALE_DATA_THRESHOLD - FRESH_DATA_THRESHOLD))

    @property
    def velocity_factor(self) -> float:
        """Compute velocity factor (0-1, higher = faster improvement)."""
        # Positive velocity means improvement, negative means regression
        if self.elo_velocity <= 0:
            return 0.0
        # Cap at 100 ELO/day for factor calculation
        return min(1.0, self.elo_velocity / 100.0)

    @property
    def data_deficit_factor(self) -> float:
        """Compute data deficit factor (0-1, higher = more data needed).

        December 2025: Prioritizes configs with low game counts, especially
        large boards (square19, hexagonal) which need more training data.
        """
        # Target games adjusted for board size
        target = TARGET_GAMES_FOR_2000_ELO
        if self.is_large_board:
            target = int(target * LARGE_BOARD_TARGET_MULTIPLIER)

        # Deficit is how far below target we are
        if self.game_count >= target:
            return 0.0  # No deficit

        # Higher factor for configs further from target
        deficit_ratio = 1.0 - (self.game_count / target)
        return min(1.0, deficit_ratio)


@dataclass
class NodeCapability:
    """Capability information for a cluster node."""
    node_id: str
    gpu_type: str = "unknown"
    gpu_memory_gb: float = 0.0
    is_ephemeral: bool = False
    current_load: float = 0.0  # 0-1, current utilization
    data_lag_seconds: float = 0.0  # Sync lag from coordinator

    @property
    def capacity_weight(self) -> float:
        """Get capacity weight based on GPU type."""
        return get_gpu_weight(self.gpu_type)

    @property
    def available_capacity(self) -> float:
        """Get available capacity (0-1)."""
        return max(0.0, 1.0 - self.current_load) * self.capacity_weight


class SelfplayScheduler:
    """Priority-based selfplay scheduler across cluster nodes.

    Responsibilities:
    - Track data freshness per configuration
    - Calculate priority scores for each config
    - Allocate selfplay games based on node capabilities
    - Integrate with feedback loop signals
    - Calculate target selfplay jobs per node (Dec 2025)
    """

    def __init__(
        self,
        # Optional callbacks for external integrations (Dec 2025)
        # These enable full delegation from P2P orchestrator
        get_cluster_elo_fn: Callable[[], dict[str, Any]] | None = None,
        load_curriculum_weights_fn: Callable[[], dict[str, float]] | None = None,
        get_board_priority_overrides_fn: Callable[[], dict[str, int]] | None = None,
        # Backpressure callbacks
        should_stop_production_fn: Callable[..., bool] | None = None,
        should_throttle_production_fn: Callable[..., bool] | None = None,
        get_throttle_factor_fn: Callable[..., float] | None = None,
        # Resource targeting callbacks
        record_utilization_fn: Callable[..., None] | None = None,
        get_host_targets_fn: Callable[[str], Any] | None = None,
        get_target_job_count_fn: Callable[..., int] | None = None,
        should_scale_up_fn: Callable[..., tuple[bool, str]] | None = None,
        should_scale_down_fn: Callable[..., tuple[bool, int, str]] | None = None,
        # Hardware-aware limits
        get_max_selfplay_for_node_fn: Callable[..., int] | None = None,
        get_hybrid_selfplay_limits_fn: Callable[..., dict[str, int]] | None = None,
        # Safeguard callback
        is_emergency_active_fn: Callable[[], bool] | None = None,
        # Dependency injection for breaking circular dependencies (Dec 2025)
        backpressure_monitor: Optional[IBackpressureMonitor] = None,
        # Verbosity
        verbose: bool = False,
    ):
        # Store callbacks (Dec 2025)
        self._get_cluster_elo_fn = get_cluster_elo_fn
        self._load_curriculum_weights_fn = load_curriculum_weights_fn
        self._get_board_priority_overrides_fn = get_board_priority_overrides_fn
        self._should_stop_production_fn = should_stop_production_fn
        self._should_throttle_production_fn = should_throttle_production_fn
        self._get_throttle_factor_fn = get_throttle_factor_fn
        self._record_utilization_fn = record_utilization_fn
        self._get_host_targets_fn = get_host_targets_fn
        self._get_target_job_count_fn = get_target_job_count_fn
        self._should_scale_up_fn = should_scale_up_fn
        self._should_scale_down_fn = should_scale_down_fn
        self._get_max_selfplay_for_node_fn = get_max_selfplay_for_node_fn
        self._get_hybrid_selfplay_limits_fn = get_hybrid_selfplay_limits_fn
        self._is_emergency_active_fn = is_emergency_active_fn
        self._verbose = verbose

        # Priority tracking
        self._config_priorities: dict[str, ConfigPriority] = {
            cfg: ConfigPriority(config_key=cfg) for cfg in ALL_CONFIGS
        }

        # Node tracking
        self._node_capabilities: dict[str, NodeCapability] = {}

        # Timing
        self._last_priority_update = 0.0
        self._priority_update_interval = 15.0  # Dec 2025: Update every 15s (was 60s)

        # Node capability refresh timing.
        # NOTE: ClusterMonitor probes can be expensive (SSH/subprocess). We rate-limit
        # them and treat externally pre-seeded capabilities as already up-to-date.
        self._last_node_capability_update = 0.0
        self._node_capability_update_interval = 60.0

        # Event subscription
        self._subscribed = False

        # Lazy dependencies
        self._training_freshness = None
        self._cluster_manifest = None
        # Injected backpressure monitor (breaks circular dep with backpressure.py)
        self._backpressure_monitor: Optional[IBackpressureMonitor] = backpressure_monitor

        # Load priority overrides from config (Dec 2025)
        self._load_priority_overrides()

        # Allocation metrics (rolling 1h window)
        self._allocation_window_seconds = 3600
        self._allocation_history: deque[tuple[float, int]] = deque()
        self._games_allocated_total = 0

    def _load_priority_overrides(self) -> None:
        """Load board_priority_overrides from unified_loop.yaml.

        Maps priority levels (0-3) to multipliers for the priority score.
        0 = CRITICAL (3x), 1 = HIGH (2x), 2 = MEDIUM (1.25x), 3 = LOW (1x)
        """
        config_paths = [
            Path(__file__).parent.parent.parent / "config" / "unified_loop.yaml",
            Path("config/unified_loop.yaml"),
            Path("/etc/ringrift/unified_loop.yaml"),
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    overrides = config.get("selfplay", {}).get("board_priority_overrides", {})
                    if overrides:
                        for config_key, priority_level in overrides.items():
                            if config_key in self._config_priorities:
                                self._config_priorities[config_key].priority_override = priority_level
                                logger.debug(
                                    f"[SelfplayScheduler] Priority override: {config_key} = {priority_level}"
                                )
                        logger.info(
                            f"[SelfplayScheduler] Loaded {len(overrides)} priority overrides from config"
                        )
                    return
                except Exception as e:
                    logger.warning(f"[SelfplayScheduler] Failed to load config {config_path}: {e}")

    # =========================================================================
    # Priority Calculation
    # =========================================================================

    async def get_priority_configs(self, top_n: int = 12) -> list[tuple[str, float]]:
        """Get configs ranked by priority for selfplay allocation.

        Args:
            top_n: Number of top priority configs to return (default: 12 for all configs)

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority

        December 28, 2025: Changed default from 6 to 12 to include all board/player configs.
        """
        await self._update_priorities()

        priorities = [
            (cfg, p.priority_score)
            for cfg, p in self._config_priorities.items()
        ]

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])

        return priorities[:top_n]

    def get_priority_configs_sync(
        self, top_n: int | None = None, filter_configs: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Get configs ranked by priority (sync version using cached data).

        This method returns cached priority data without triggering an async update.
        Use this from synchronous contexts where you need priority-ordered configs.

        Args:
            top_n: Optional limit on number of configs to return (default: all)
            filter_configs: Optional list of config keys to filter by

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority descending

        December 2025: Added for IdleResourceDaemon and other sync callers.
        """
        priorities = [
            (cfg, p.priority_score)
            for cfg, p in self._config_priorities.items()
            if filter_configs is None or cfg in filter_configs
        ]

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])

        if top_n is not None:
            return priorities[:top_n]
        return priorities

    async def _update_priorities(self) -> None:
        """Update priority scores for all configurations."""
        now = time.time()
        if now - self._last_priority_update < self._priority_update_interval:
            return

        self._last_priority_update = now

        # Get data freshness
        freshness_data = await self._get_data_freshness()

        # Get ELO velocities
        elo_data = await self._get_elo_velocities()

        # Get feedback loop signals
        feedback_data = await self._get_feedback_signals()

        # Get curriculum weights (Phase 2C.3)
        curriculum_data = await self._get_curriculum_weights()

        # Get improvement boosts (Phase 5)
        improvement_data = self._get_improvement_boosts()

        # Get momentum multipliers (Phase 19)
        momentum_data = self._get_momentum_multipliers()

        # Get game counts (Dec 2025)
        game_count_data = await self._get_game_counts()

        # Update each config
        for config_key, priority in self._config_priorities.items():
            # Update staleness
            if config_key in freshness_data:
                priority.staleness_hours = freshness_data[config_key]

            # Update ELO velocity
            if config_key in elo_data:
                priority.elo_velocity = elo_data[config_key]

            # Update feedback signals
            if config_key in feedback_data:
                priority.exploration_boost = feedback_data[config_key].get("exploration_boost", 1.0)
                priority.training_pending = feedback_data[config_key].get("training_pending", False)

            # Update curriculum weight (Phase 2C.3)
            if config_key in curriculum_data:
                priority.curriculum_weight = curriculum_data[config_key]

            # Update improvement boost (Phase 5)
            if config_key in improvement_data:
                priority.improvement_boost = improvement_data[config_key]

            # Update momentum multiplier (Phase 19)
            if config_key in momentum_data:
                priority.momentum_multiplier = momentum_data[config_key]

            # Update game count and large board flag (Dec 2025)
            if config_key in game_count_data:
                priority.game_count = game_count_data[config_key]
            # Mark large boards for higher data deficit weight
            priority.is_large_board = config_key.startswith("square19") or config_key.startswith("hexagonal")

            # Compute priority score
            priority.priority_score = self._compute_priority_score(priority)

        # Phase 12: Check for expired exploration boosts and decay them
        decayed_count = self._decay_expired_boosts(now)
        if decayed_count > 0:
            logger.info(f"[SelfplayScheduler] Decayed {decayed_count} expired exploration boosts")

        logger.debug(f"[SelfplayScheduler] Updated priorities for {len(self._config_priorities)} configs")

        # Phase 6: Record cluster utilization in ImprovementOptimizer
        # This enables the training loop to adapt based on resource usage
        try:
            from app.training.improvement_optimizer import get_improvement_optimizer

            # Get approximate utilization from node count and active configs
            active_configs = len([p for p in self._config_priorities.values() if p.training_pending])
            total_configs = len(self._config_priorities)

            # Estimate GPU utilization: assume 50% baseline + (active configs / total) * 50%
            gpu_util = min(100.0, 50.0 + (active_configs / max(1, total_configs)) * 50.0)
            cpu_util = gpu_util * 0.6  # CPU typically lower than GPU for selfplay

            optimizer = get_improvement_optimizer()
            rec = optimizer.record_cluster_utilization(
                cpu_utilization=cpu_util,
                gpu_utilization=gpu_util,
            )
            logger.debug(
                f"[SelfplayScheduler] Recorded cluster utilization: CPU={cpu_util:.0f}%, "
                f"GPU={gpu_util:.0f}% (signal: {rec.signal.name})"
            )
        except ImportError:
            pass  # Improvement optimizer not available
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to record utilization: {e}")

    def _compute_priority_score(self, priority: ConfigPriority) -> float:
        """Compute overall priority score for a configuration.

        Higher score = higher priority for selfplay allocation.

        December 2025 - Phase 2C.3: Now includes curriculum weight factor.
        December 2025 - Phase 5: Now includes improvement boost and quality penalty.
        December 2025 - Phase 19: Now includes momentum multiplier from FeedbackAccelerator.
        December 2025 - Data deficit factor for large boards with low game counts.
        """
        # Base factors
        staleness = priority.staleness_factor * STALENESS_WEIGHT
        velocity = priority.velocity_factor * ELO_VELOCITY_WEIGHT
        training = (1.0 if priority.training_pending else 0.0) * TRAINING_NEED_WEIGHT
        exploration = (priority.exploration_boost - 1.0) * EXPLORATION_BOOST_WEIGHT

        # Phase 2C.3: Curriculum weight factor (normalized around 1.0)
        # Higher curriculum weight = more data needed for this config
        curriculum = (priority.curriculum_weight - 1.0) * CURRICULUM_WEIGHT

        # Phase 5: Improvement optimizer boost (-0.10 to +0.15)
        # Positive when config is on a promotion streak
        improvement = priority.improvement_boost * IMPROVEMENT_BOOST_WEIGHT

        # Phase 5: Quality penalty (0.0 to -0.20)
        # Applied when quality degrades below threshold
        quality = priority.quality_penalty

        # Dec 2025: Data deficit factor - boost configs with low game counts
        # Large boards (square19, hexagonal) especially need more data
        data_deficit = priority.data_deficit_factor * DATA_DEFICIT_WEIGHT

        # Combine factors
        score = staleness + velocity + training + exploration + curriculum + improvement + quality + data_deficit

        # Apply exploration boost as multiplier
        score *= priority.exploration_boost

        # Phase 19: Apply momentum multiplier from FeedbackAccelerator
        # This provides Elo momentum → Selfplay rate coupling:
        # - ACCELERATING: 1.5x (capitalize on positive momentum)
        # - IMPROVING: 1.25x (boost for continued improvement)
        # - STABLE: 1.0x (normal rate)
        # - PLATEAU: 1.1x (slight boost to try to break plateau)
        # - REGRESSING: 0.75x (reduce noise, focus on quality)
        score_before_momentum = score
        score *= priority.momentum_multiplier

        # Log when momentum multiplier significantly affects priority (>10% change)
        if abs(priority.momentum_multiplier - 1.0) > 0.1:
            logger.info(
                f"[SelfplayScheduler] Momentum multiplier applied to {priority.config_key}: "
                f"{priority.momentum_multiplier:.2f}x (score: {score_before_momentum:.3f} → {score:.3f})"
            )

        # Dec 2025: Apply priority override from config
        # Boosts critically data-starved configs (hexagonal_*, square19_3p/4p)
        # 0=CRITICAL (3x), 1=HIGH (2x), 2=MEDIUM (1.25x), 3=LOW (1x)
        override_multiplier = PRIORITY_OVERRIDE_MULTIPLIERS.get(priority.priority_override, 1.0)
        score *= override_multiplier

        return score

    async def _get_data_freshness(self) -> dict[str, float]:
        """Get data freshness (hours since last export) per config.

        Returns:
            Dict mapping config_key to hours since last export
        """
        result = {}

        try:
            # Try using TrainingFreshness if available
            if self._training_freshness is None:
                try:
                    from app.coordination.training_freshness import get_training_freshness
                    self._training_freshness = get_training_freshness()
                except ImportError:
                    pass

            if self._training_freshness:
                for config_key in ALL_CONFIGS:
                    staleness = await self._training_freshness.get_staleness(config_key)
                    result[config_key] = staleness
            else:
                # Fallback: check NPZ file modification times
                from pathlib import Path
                now = time.time()

                for config_key in ALL_CONFIGS:
                    parts = config_key.rsplit("_", 1)
                    if len(parts) != 2:
                        continue

                    board_type = parts[0]
                    npz_path = Path(f"data/training/{board_type}_{parts[1]}.npz")

                    if npz_path.exists():
                        mtime = npz_path.stat().st_mtime
                        result[config_key] = (now - mtime) / 3600.0
                    else:
                        result[config_key] = MAX_STALENESS_HOURS

        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Error getting freshness (using defaults): {e}")

        return result

    async def _get_elo_velocities(self) -> dict[str, float]:
        """Get ELO improvement velocities per config.

        Returns:
            Dict mapping config_key to ELO points per day
        """
        result = {}

        try:
            # Try using QueuePopulator's ConfigTarget if available
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    result[config_key] = target.elo_velocity
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Error getting ELO velocities (using defaults): {e}")

        return result

    async def _get_feedback_signals(self) -> dict[str, dict[str, Any]]:
        """Get feedback loop signals per config.

        Returns:
            Dict mapping config_key to feedback data
        """
        result = {}

        try:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller

            controller = get_feedback_loop_controller()
            if controller:
                for config_key in ALL_CONFIGS:
                    state = controller._get_or_create_state(config_key)
                    result[config_key] = {
                        "exploration_boost": state.current_exploration_boost,
                        "training_pending": state.current_training_intensity == "accelerated",
                    }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Error getting feedback signals (using defaults): {e}")

        return result

    async def _get_curriculum_weights(self) -> dict[str, float]:
        """Get curriculum weights per config.

        December 2025 - Phase 2C.3: Wire curriculum weights into priority calculation.
        Higher curriculum weight = config needs more training data.

        Returns:
            Dict mapping config_key to curriculum weight (default 1.0)
        """
        result = {}

        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            if feedback:
                # Use get_curriculum_weights() which computes weights from metrics
                # (win rates, Elo trends, weak opponent data) instead of accessing
                # _current_weights directly which may be empty initially
                computed_weights = feedback.get_curriculum_weights()
                if computed_weights:
                    result = computed_weights
                else:
                    # Fall back to manually tracked weights if metrics unavailable
                    result = dict(feedback._current_weights)

        except ImportError:
            logger.debug("[SelfplayScheduler] curriculum_feedback not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting curriculum weights: {e}")

        # Ensure all configs have a default weight
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1.0

        # December 2025: Normalize weights so average is 1.0
        # This ensures curriculum weights represent relative priority,
        # not absolute scaling that could distort priority scores
        if result:
            total = sum(result.values())
            target_sum = len(result)  # Average of 1.0
            if total > 0 and abs(total - target_sum) > 0.01:
                scale = target_sum / total
                result = {k: v * scale for k, v in result.items()}
                logger.debug(
                    f"[SelfplayScheduler] Normalized curriculum weights: "
                    f"scale={scale:.3f}, sum={sum(result.values()):.2f}"
                )

        return result

    def _get_improvement_boosts(self) -> dict[str, float]:
        """Get improvement boosts from ImprovementOptimizer per config.

        Phase 5 (Dec 2025): Connects selfplay scheduling to training success signals.
        When a config is on a promotion streak, boost its selfplay priority.

        Returns:
            Dict mapping config_key to boost value (-0.10 to +0.15)
        """
        result: dict[str, float] = {}

        try:
            from app.training.improvement_optimizer import get_selfplay_priority_boost

            for config_key in ALL_CONFIGS:
                boost = get_selfplay_priority_boost(config_key)
                if boost != 0.0:
                    result[config_key] = boost
                    logger.debug(
                        f"[SelfplayScheduler] Improvement boost for {config_key}: {boost:+.2f}"
                    )

        except ImportError:
            logger.debug("[SelfplayScheduler] improvement_optimizer not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting improvement boosts: {e}")

        return result

    def _get_momentum_multipliers(self) -> dict[str, float]:
        """Get momentum multipliers from FeedbackAccelerator per config.

        Phase 19 (Dec 2025): Connects selfplay scheduling to Elo momentum.
        This provides Elo momentum → Selfplay rate coupling:
        - ACCELERATING: 1.5x (capitalize on positive momentum)
        - IMPROVING: 1.25x (boost for continued improvement)
        - STABLE: 1.0x (normal rate)
        - PLATEAU: 1.1x (slight boost to try to break plateau)
        - REGRESSING: 0.75x (reduce noise, focus on quality)

        Returns:
            Dict mapping config_key to multiplier value (0.5 to 1.5)
        """
        result: dict[str, float] = {}

        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            for config_key in ALL_CONFIGS:
                multiplier = accelerator.get_selfplay_multiplier(config_key)
                if multiplier != 1.0:  # Only log non-default values
                    result[config_key] = multiplier
                    logger.debug(
                        f"[SelfplayScheduler] Momentum multiplier for {config_key}: {multiplier:.2f}x"
                    )
                else:
                    result[config_key] = multiplier

        except ImportError:
            logger.debug("[SelfplayScheduler] feedback_accelerator not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting momentum multipliers: {e}")

        return result

    async def _get_game_counts(self) -> dict[str, int]:
        """Get game counts per config from local databases.

        December 2025: Used for data deficit prioritization.
        Configs with fewer games get higher priority to reach 100K target.

        Returns:
            Dict mapping config_key to game count
        """
        result: dict[str, int] = {}

        try:
            from app.utils.game_discovery import get_game_counts_summary

            # get_game_counts_summary returns {config_key: count}
            result = get_game_counts_summary()
            logger.debug(
                f"[SelfplayScheduler] Game counts: "
                f"{sum(result.values()):,} total across {len(result)} configs"
            )

        except ImportError:
            logger.debug("[SelfplayScheduler] game_discovery not available")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Error getting game counts (using defaults): {e}")

        # Ensure all configs have a count (0 if not found)
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 0

        return result

    # =========================================================================
    # Node Allocation
    # =========================================================================

    async def allocate_selfplay_batch(
        self,
        games_per_config: int = DEFAULT_GAMES_PER_CONFIG,
        max_configs: int = 12,
    ) -> dict[str, dict[str, int]]:
        """Allocate selfplay games across cluster nodes.

        Args:
            games_per_config: Target games per config
            max_configs: Maximum configs to allocate (default: 12 for all configs)

        Returns:
            Dict mapping config_key to {node_id: num_games}

        December 28, 2025: Changed default from 6 to 12 to include all board/player configs.
        """
        # Check backpressure before allocating (Dec 2025)
        try:
            # Use injected monitor if available, otherwise lazy import
            # (Dec 2025: IBackpressureMonitor protocol enables dependency injection)
            if self._backpressure_monitor is None:
                from app.coordination.backpressure import get_backpressure_monitor
                self._backpressure_monitor = get_backpressure_monitor()
            bp_signal = await self._backpressure_monitor.get_signal()

            if bp_signal.should_pause:
                logger.warning(
                    f"[SelfplayScheduler] Backpressure pause: pressure={bp_signal.overall_pressure:.2f}"
                )
                return {}

            # Scale games by spawn rate multiplier
            spawn_multiplier = bp_signal.spawn_rate_multiplier
            if spawn_multiplier < 1.0:
                scaled_games = int(games_per_config * spawn_multiplier)
                scaled_games = max(MIN_GAMES_PER_ALLOCATION, scaled_games)
                logger.info(
                    f"[SelfplayScheduler] Backpressure scaling: {games_per_config} -> {scaled_games} "
                    f"(multiplier={spawn_multiplier:.2f}, pressure={bp_signal.overall_pressure:.2f})"
                )
                games_per_config = scaled_games
        except Exception as e:
            # Don't let backpressure failures block allocation
            logger.warning(f"[SelfplayScheduler] Backpressure check failed (continuing): {e}")

        # Get priority configs
        priorities = await self.get_priority_configs(top_n=max_configs)

        # Get available nodes
        await self._update_node_capabilities()

        allocation: dict[str, dict[str, int]] = {}

        for config_key, priority_score in priorities:
            if priority_score <= 0:
                continue

            # Allocate based on priority weight
            config_games = int(games_per_config * (0.5 + priority_score))
            config_games = max(MIN_GAMES_PER_ALLOCATION, config_games)

            # Distribute across nodes
            node_allocation = self._allocate_to_nodes(config_key, config_games)
            if node_allocation:
                allocation[config_key] = node_allocation

                # Update priority tracking
                priority = self._config_priorities[config_key]
                priority.games_allocated = sum(node_allocation.values())
                priority.nodes_allocated = list(node_allocation.keys())

        total_allocated = sum(
            sum(node_games.values()) for node_games in allocation.values()
        )
        self._record_allocation(total_allocated)

        logger.info(
            f"[SelfplayScheduler] Allocated {len(allocation)} configs: "
            f"{', '.join(f'{k}={sum(v.values())}' for k, v in allocation.items())}"
        )

        # Dec 2025: Emit SELFPLAY_ALLOCATION_UPDATED for downstream consumers
        # (IdleResourceDaemon, feedback loops, etc.)
        if total_allocated > 0:
            self._emit_allocation_updated(allocation, total_allocated, trigger="allocate_batch")

        return allocation

    def _allocate_to_nodes(
        self,
        config_key: str,
        total_games: int,
    ) -> dict[str, int]:
        """Allocate games for a config across available nodes.

        December 2025 - Phase 2B.4: Ephemeral node short-job prioritization.
        Short jobs (square8, hex8) are boosted for ephemeral nodes.
        Long jobs (square19, hexagonal) are reduced for ephemeral nodes.

        December 2025 - P2P Integration: Excludes unhealthy nodes and applies
        cluster health factor to allocation.

        Args:
            config_key: Configuration key
            total_games: Total games to allocate

        Returns:
            Dict mapping node_id to num_games
        """
        # Get unhealthy nodes from P2P health tracking
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())

        # Apply cluster health factor to total games
        cluster_health = getattr(self, "_cluster_health_factor", 1.0)
        if cluster_health < 1.0:
            adjusted_games = max(MIN_GAMES_PER_ALLOCATION, int(total_games * cluster_health))
            logger.debug(
                f"[SelfplayScheduler] Cluster health {cluster_health:.2f} reducing allocation: "
                f"{total_games} → {adjusted_games} games"
            )
            total_games = adjusted_games

        # Get available nodes sorted by capacity, excluding unhealthy nodes
        available_nodes = sorted(
            [
                n for n in self._node_capabilities.values()
                if n.available_capacity > 0.1
                and n.node_id not in unhealthy_nodes  # Exclude unhealthy nodes
            ],
            key=lambda n: (-n.available_capacity, n.data_lag_seconds),
        )

        if not available_nodes:
            return {}

        allocation: dict[str, int] = {}
        remaining = total_games

        # Phase 2B.4: Determine if this is a short job (quick selfplay)
        # Small boards complete in <10 minutes, good for ephemeral nodes
        is_short_job = config_key.startswith(("square8", "hex8"))

        # Calculate total capacity
        total_capacity = sum(n.available_capacity for n in available_nodes)

        for node in available_nodes:
            if remaining <= 0:
                break

            # Proportional allocation based on capacity
            proportion = node.available_capacity / total_capacity
            node_games = max(
                MIN_GAMES_PER_ALLOCATION,
                int(total_games * proportion),
            )

            # Adjust based on GPU type
            gpu_games = SELFPLAY_GAMES_PER_NODE.get(node.gpu_type, 500)
            node_games = min(node_games, gpu_games)

            # Phase 2B.4: Ephemeral node job-duration matching
            if node.is_ephemeral:
                if is_short_job:
                    # Boost allocation to ephemeral for short jobs
                    node_games = int(node_games * 1.5)
                    logger.debug(
                        f"[SelfplayScheduler] Boosted {node.node_id} (ephemeral) "
                        f"for short job {config_key}"
                    )
                else:
                    # Reduce for long jobs (risk of termination)
                    node_games = int(node_games * 0.5)
                    logger.debug(
                        f"[SelfplayScheduler] Reduced {node.node_id} (ephemeral) "
                        f"for long job {config_key}"
                    )

            # Cap at remaining games
            node_games = min(node_games, remaining)

            if node_games >= MIN_GAMES_PER_ALLOCATION:
                allocation[node.node_id] = node_games
                remaining -= node_games

        return allocation

    async def _update_node_capabilities(self) -> None:
        """Update node capability information from cluster.

        This is intentionally rate-limited because ClusterMonitor may probe remote
        hosts via subprocess/SSH. In unit tests and some callers, node capabilities
        may be pre-seeded directly on the instance; in that case we treat them as
        fresh and skip expensive probing.
        """
        now = time.time()

        # If capabilities were pre-seeded externally (common in tests) and we have
        # never performed a refresh, treat the injected snapshot as up-to-date.
        if self._node_capabilities and self._last_node_capability_update == 0.0:
            self._last_node_capability_update = now
            return

        if self._node_capabilities and (now - self._last_node_capability_update) < self._node_capability_update_interval:
            return

        # Mark refresh time up-front to avoid repeated expensive probes if the
        # monitor fails repeatedly.
        self._last_node_capability_update = now

        try:
            # Try getting from cluster monitor
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status()

            for node_id, node_status in status.nodes.items():
                if node_id not in self._node_capabilities:
                    self._node_capabilities[node_id] = NodeCapability(node_id=node_id)

                cap = self._node_capabilities[node_id]
                cap.gpu_type = node_status.gpu or "unknown"
                cap.gpu_memory_gb = node_status.gpu_memory_total_gb
                cap.is_ephemeral = is_ephemeral_node(node_id)
                cap.current_load = node_status.gpu_utilization_percent / 100.0
                cap.data_lag_seconds = node_status.sync_lag_seconds

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error updating node capabilities: {e}")

    # =========================================================================
    # External Boost Interface (December 2025)
    # =========================================================================

    def boost_config_allocation(self, config_key: str, multiplier: float = 1.5) -> bool:
        """Boost selfplay allocation for a specific configuration.

        December 2025: Called by TrainingTriggerDaemon when gauntlet evaluation
        shows poor performance, triggering additional selfplay to generate
        more training data for struggling configurations.

        Args:
            config_key: Configuration to boost (e.g., "hex8_2p")
            multiplier: Boost multiplier (default 1.5x, capped at 2.0)

        Returns:
            True if boost was applied, False if config not found
        """
        if config_key not in self._config_priorities:
            logger.warning(
                f"[SelfplayScheduler] Cannot boost unknown config: {config_key}"
            )
            return False

        priority = self._config_priorities[config_key]

        # Apply multiplier to exploration boost (capped at 2.0)
        old_boost = priority.exploration_boost
        priority.exploration_boost = min(2.0, priority.exploration_boost * multiplier)

        # Also boost momentum multiplier temporarily
        old_momentum = priority.momentum_multiplier
        priority.momentum_multiplier = min(1.5, priority.momentum_multiplier * 1.2)

        logger.info(
            f"[SelfplayScheduler] Boosted {config_key}: "
            f"exploration {old_boost:.2f}x → {priority.exploration_boost:.2f}x, "
            f"momentum {old_momentum:.2f}x → {priority.momentum_multiplier:.2f}x"
        )

        # Emit SELFPLAY_RATE_CHANGED if momentum changed by >20%
        if abs(priority.momentum_multiplier - old_momentum) / max(old_momentum, 0.01) > 0.20:
            change_percent = ((priority.momentum_multiplier - old_momentum) / old_momentum) * 100.0
            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_RATE_CHANGED, {
                        "config_key": config_key,
                        "old_rate": old_momentum,
                        "new_rate": priority.momentum_multiplier,
                        "change_percent": change_percent,
                        "reason": "config_boost",
                    })
                    logger.debug(
                        f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED for {config_key}: "
                        f"{old_momentum:.2f} → {priority.momentum_multiplier:.2f} ({change_percent:+.1f}%)"
                    )
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit SELFPLAY_RATE_CHANGED: {emit_err}")

        # Force priority recalculation
        self._last_priority_update = 0.0

        return True

    def get_config_priority(self, config_key: str) -> ConfigPriority | None:
        """Get current priority state for a configuration.

        December 2025: Useful for monitoring and debugging priority decisions.

        Args:
            config_key: Configuration to query

        Returns:
            ConfigPriority object or None if not found
        """
        return self._config_priorities.get(config_key)

    # =========================================================================
    # Event Integration
    # =========================================================================

    def subscribe_to_events(self) -> None:
        """Subscribe to relevant pipeline events."""
        if self._subscribed:
            return

        try:
            # Dec 2025 fix: Use get_router() instead of get_event_bus() because:
            # 1. get_router() always returns a valid UnifiedEventRouter singleton
            # 2. get_event_bus() can return None if data_events module unavailable
            # 3. UnifiedEventRouter handles event type normalization automatically
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Dec 2025: Use per-subscription error handling to ensure one failure
            # doesn't prevent other subscriptions from being registered
            subscribed_count = 0
            failed_count = 0

            def _safe_subscribe(event_type, handler, name: str) -> bool:
                """Subscribe with individual error handling."""
                nonlocal subscribed_count, failed_count
                try:
                    # UnifiedEventRouter.subscribe() handles both enum and string types
                    router.subscribe(event_type, handler)
                    subscribed_count += 1
                    return True
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"[SelfplayScheduler] Failed to subscribe to {name}: {e}")
                    return False

            # Core subscriptions (always attempt)
            _safe_subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete, "SELFPLAY_COMPLETE")
            _safe_subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete, "TRAINING_COMPLETED")
            _safe_subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_complete, "MODEL_PROMOTED")
            # Phase 5: Subscribe to feedback events
            _safe_subscribe(DataEventType.SELFPLAY_TARGET_UPDATED, self._on_selfplay_target_updated, "SELFPLAY_TARGET_UPDATED")
            _safe_subscribe(DataEventType.QUALITY_DEGRADED, self._on_quality_degraded, "QUALITY_DEGRADED")
            # Phase 4A.1: Subscribe to curriculum rebalancing (December 2025)
            _safe_subscribe(DataEventType.CURRICULUM_REBALANCED, self._on_curriculum_rebalanced, "CURRICULUM_REBALANCED")
            # P0.1 (Dec 2025): Subscribe to SELFPLAY_RATE_CHANGED from FeedbackAccelerator
            _safe_subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed, "SELFPLAY_RATE_CHANGED")
            # P1.1 (Dec 2025): Subscribe to TRAINING_BLOCKED_BY_QUALITY for selfplay acceleration
            _safe_subscribe(DataEventType.TRAINING_BLOCKED_BY_QUALITY, self._on_training_blocked_by_quality, "TRAINING_BLOCKED_BY_QUALITY")

            # Optional subscriptions (graceful degradation if event type doesn't exist)
            if hasattr(DataEventType, 'OPPONENT_MASTERED'):
                _safe_subscribe(DataEventType.OPPONENT_MASTERED, self._on_opponent_mastered, "OPPONENT_MASTERED")
            if hasattr(DataEventType, 'TRAINING_EARLY_STOPPED'):
                _safe_subscribe(DataEventType.TRAINING_EARLY_STOPPED, self._on_training_early_stopped, "TRAINING_EARLY_STOPPED")
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                _safe_subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed, "ELO_VELOCITY_CHANGED")
            if hasattr(DataEventType, 'EXPLORATION_BOOST'):
                _safe_subscribe(DataEventType.EXPLORATION_BOOST, self._on_exploration_boost, "EXPLORATION_BOOST")
            # Dec 2025: Subscribe to CURRICULUM_ADVANCED for curriculum progression
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                _safe_subscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced, "CURRICULUM_ADVANCED")
            # Dec 2025: Subscribe to ADAPTIVE_PARAMS_CHANGED for parameter adjustments
            if hasattr(DataEventType, 'ADAPTIVE_PARAMS_CHANGED'):
                _safe_subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, self._on_adaptive_params_changed, "ADAPTIVE_PARAMS_CHANGED")
            if hasattr(DataEventType, 'LOW_QUALITY_DATA_WARNING'):
                _safe_subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, self._on_low_quality_warning, "LOW_QUALITY_DATA_WARNING")

            # P2P cluster health events (December 2025)
            if hasattr(DataEventType, 'NODE_UNHEALTHY'):
                _safe_subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy, "NODE_UNHEALTHY")
            if hasattr(DataEventType, 'NODE_RECOVERED'):
                _safe_subscribe(DataEventType.NODE_RECOVERED, self._on_node_recovered, "NODE_RECOVERED")
            if hasattr(DataEventType, 'P2P_NODE_DEAD'):
                _safe_subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_unhealthy, "P2P_NODE_DEAD")
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy, "P2P_CLUSTER_UNHEALTHY")
            if hasattr(DataEventType, 'P2P_CLUSTER_HEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy, "P2P_CLUSTER_HEALTHY")
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                _safe_subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline, "HOST_OFFLINE")

            # Dec 2025: Subscribe to regression events for curriculum rebalancing
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                _safe_subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected, "REGRESSION_DETECTED")

            self._subscribed = subscribed_count > 0
            if self._subscribed:
                logger.info(
                    f"[SelfplayScheduler] Subscribed to {subscribed_count} pipeline events "
                    f"(failed: {failed_count}, includes P2P health events)"
                )
            else:
                logger.warning("[SelfplayScheduler] No subscriptions succeeded, reactive scheduling disabled")

        except ImportError as e:
            logger.warning(f"[SelfplayScheduler] Event router unavailable: {e}")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to subscribe to events: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            if config_key in self._config_priorities:
                # Reset staleness for this config
                self._config_priorities[config_key].staleness_hours = 0.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            if config_key in self._config_priorities:
                # Clear training pending flag
                self._config_priorities[config_key].training_pending = False
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            success = event.payload.get("success", False)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                if success:
                    # Boost for continued improvement
                    priority.exploration_boost = 1.0
                else:
                    # Increase exploration on failure
                    priority.exploration_boost = min(2.0, priority.exploration_boost * 1.3)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling promotion complete: {e}")

    def _on_selfplay_target_updated(self, event: Any) -> None:
        """Handle selfplay target update request.

        Phase 5 (Dec 2025): Responds to requests for more/fewer selfplay games.
        Typically emitted when training needs more data urgently.

        Dec 28 2025: Extended to handle search_budget and velocity feedback
        from FeedbackLoopController for reaching 2000+ Elo.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            target_games = payload.get("target_games", 0)
            priority_val = payload.get("priority", "normal")
            search_budget = payload.get("search_budget", 0)
            exploration_boost = payload.get("exploration_boost", 1.0)
            velocity = payload.get("velocity", 0.0)
            reason = payload.get("reason", "")

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Dec 28 2025: Apply search budget from velocity feedback
                if search_budget > 0 and reason == "velocity_feedback":
                    old_budget = getattr(priority, "search_budget", 400)
                    priority.search_budget = search_budget
                    logger.info(
                        f"[SelfplayScheduler] Updating {config_key} search budget: "
                        f"{old_budget}→{search_budget} (velocity={velocity:.1f} Elo/hr)"
                    )

                # Apply exploration boost
                if exploration_boost != 1.0:
                    priority.exploration_boost = exploration_boost

                # Boost priority based on urgency
                if priority_val.upper() == "HIGH":
                    priority.training_pending = True
                    priority.exploration_boost = max(1.2, priority.exploration_boost)
                    logger.info(
                        f"[SelfplayScheduler] Boosting {config_key} priority "
                        f"(target: {target_games} games, priority: {priority_val})"
                    )
                elif priority_val.lower() == "low":
                    # Reduce priority for this config
                    priority.exploration_boost = min(0.8, priority.exploration_boost)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay target: {e}")

    def _on_quality_degraded(self, event: Any) -> None:
        """Handle quality degradation event.

        Phase 5 (Dec 2025): When game quality drops below threshold,
        apply a penalty to reduce selfplay allocation for this config.
        """
        try:
            config_key = event.payload.get("config_key", "")
            quality_score = event.payload.get("quality_score", 0.0)
            threshold = event.payload.get("threshold", 0.6)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Apply quality penalty proportional to how far below threshold
                # quality_score=0.4, threshold=0.6 → penalty = -0.20 * (0.6-0.4)/0.6 = -0.067
                if quality_score < threshold:
                    degradation = (threshold - quality_score) / threshold
                    priority.quality_penalty = -0.20 * degradation
                    logger.warning(
                        f"[SelfplayScheduler] Quality degradation for {config_key}: "
                        f"score={quality_score:.2f} < {threshold:.2f}, "
                        f"penalty={priority.quality_penalty:.3f}"
                    )
                else:
                    # Quality recovered, clear penalty
                    if priority.quality_penalty < 0:
                        logger.info(
                            f"[SelfplayScheduler] Quality recovered for {config_key}: "
                            f"score={quality_score:.2f}, clearing penalty"
                        )
                    priority.quality_penalty = 0.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling quality degraded: {e}")

    def _on_regression_detected(self, event: Any) -> None:
        """Handle model regression detection.

        Dec 2025: When evaluation detects model regression (win rate dropping),
        trigger curriculum rebalancing to adjust selfplay priorities.
        This helps recover from training instability by increasing exploration.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            severity = payload.get("severity", "moderate")  # mild, moderate, severe
            win_rate_drop = payload.get("win_rate_drop", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Apply exploration boost based on regression severity
                boost_factor = {"mild": 0.1, "moderate": 0.2, "severe": 0.3}.get(severity, 0.15)
                priority.exploration_boost = boost_factor

                logger.warning(
                    f"[SelfplayScheduler] Regression detected for {config_key}: "
                    f"severity={severity}, win_rate_drop={win_rate_drop:.2%}, "
                    f"exploration_boost={boost_factor}"
                )

                # Emit curriculum rebalance event to trigger downstream updates
                try:
                    from app.coordination.event_router import publish_sync

                    publish_sync("CURRICULUM_REBALANCED", {
                        "config_key": config_key,
                        "reason": "regression_detected",
                        "severity": severity,
                        "new_exploration_boost": boost_factor,
                        "source": "selfplay_scheduler",  # Loop guard identifier
                    })
                except ImportError:
                    pass  # Event system not available

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling regression: {e}")

    def _on_curriculum_rebalanced(self, event: Any) -> None:
        """Handle curriculum rebalancing event.

        Phase 4A.1 (December 2025): Updates priority weights when curriculum
        feedback adjusts config priorities based on training progress.

        December 2025 Guard: Skip events originated by SelfplayScheduler itself
        to prevent echo loops in the event system.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            # Loop guard: Skip events we emitted (prevents echo loops)
            source = payload.get("source", "")
            if source == "selfplay_scheduler":
                logger.debug("[SelfplayScheduler] Skipping self-originated CURRICULUM_REBALANCED")
                return

            config_key = payload.get("config_key", "")
            new_weight = payload.get("weight", 1.0)
            reason = payload.get("reason", "")

            if config_key in self._config_priorities:
                old_weight = self._config_priorities[config_key].curriculum_weight
                self._config_priorities[config_key].curriculum_weight = new_weight

                # Only log significant changes
                if abs(new_weight - old_weight) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Curriculum rebalanced: {config_key} "
                        f"weight {old_weight:.2f} → {new_weight:.2f}"
                        + (f" ({reason})" if reason else "")
                    )

            # Also handle batch updates (multiple configs at once)
            weights = payload.get("weights", {})
            if weights:
                for cfg, weight in weights.items():
                    if cfg in self._config_priorities:
                        old_w = self._config_priorities[cfg].curriculum_weight
                        self._config_priorities[cfg].curriculum_weight = weight
                        if abs(weight - old_w) > 0.1:
                            logger.info(
                                f"[SelfplayScheduler] Curriculum batch update: "
                                f"{cfg} weight {old_w:.2f} → {weight:.2f}"
                            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling curriculum rebalanced: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change event from FeedbackAccelerator.

        P0.1 (Dec 2025): Closes the Elo momentum → Selfplay rate feedback loop.
        FeedbackAccelerator emits SELFPLAY_RATE_CHANGED when it detects:
        - ACCELERATING: 1.5x multiplier (capitalize on positive momentum)
        - IMPROVING: 1.25x multiplier (boost for continued improvement)
        - STABLE: 1.0x multiplier (normal rate)
        - PLATEAU: 1.1x multiplier (slight boost to break plateau)
        - REGRESSING: 0.75x multiplier (reduce noise, focus on quality)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "") or payload.get("config", "")
            new_rate = payload.get("new_rate", 1.0) or payload.get("rate_multiplier", 1.0)
            momentum_state = payload.get("momentum_state", "unknown")

            if config_key in self._config_priorities:
                old_rate = self._config_priorities[config_key].momentum_multiplier
                self._config_priorities[config_key].momentum_multiplier = float(new_rate)

                # Log significant changes
                if abs(new_rate - old_rate) > 0.1:
                    logger.info(
                        f"[SelfplayScheduler] Selfplay rate changed: {config_key} "
                        f"multiplier {old_rate:.2f} → {new_rate:.2f} "
                        f"(momentum: {momentum_state})"
                    )
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received rate change for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay rate changed: {e}")

    def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle training blocked by quality event.

        P1.1 (Dec 2025): Strategic Improvement Plan Gap 1.1 fix.
        When training is blocked due to stale/low-quality data, boost selfplay
        allocation by 1.5x to accelerate data generation for that config.

        This closes the critical feedback loop:
        TRAINING_BLOCKED_BY_QUALITY → Selfplay acceleration → Fresh data → Training resumes
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "") or payload.get("config", "")
            reason = payload.get("reason", "unknown")
            data_age_hours = payload.get("data_age_hours", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Boost selfplay allocation by 1.5x for this config
                old_boost = priority.exploration_boost
                priority.exploration_boost = max(1.5, old_boost * 1.5)
                priority.training_pending = True

                # Log the quality block
                logger.warning(
                    f"[SelfplayScheduler] Training blocked for {config_key} "
                    f"(reason: {reason}, age: {data_age_hours:.1f}h). "
                    f"Boosted selfplay: exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

                # Emit a target update event to propagate the boost
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                            "config_key": config_key,
                            "priority": "high",
                            "reason": f"training_blocked:{reason}",
                            "exploration_boost": priority.exploration_boost,
                        })
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received training blocked for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training blocked by quality: {e}")

    def _on_opponent_mastered(self, event: Any) -> None:
        """Handle opponent mastered event.

        P1.4 (Dec 2025): When a model has mastered its current opponent level,
        advance the curriculum and slightly reduce selfplay allocation for this
        config (it's now "easier" and should allocate resources elsewhere).

        This closes the feedback loop:
        OPPONENT_MASTERED → Curriculum advancement → Harder opponents → Better training
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "") or payload.get("config", "")
            opponent_level = payload.get("opponent_level", "unknown")
            win_rate = payload.get("win_rate", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Reduce curriculum weight slightly since opponent was mastered
                old_weight = priority.curriculum_weight
                priority.curriculum_weight = max(0.5, old_weight * 0.9)

                # Also slightly reduce exploration (we don't need as much variance)
                old_boost = priority.exploration_boost
                priority.exploration_boost = max(0.8, old_boost * 0.95)

                logger.info(
                    f"[SelfplayScheduler] Opponent mastered for {config_key} "
                    f"(level={opponent_level}, win_rate={win_rate:.1%}). "
                    f"curriculum_weight {old_weight:.2f} → {priority.curriculum_weight:.2f}, "
                    f"exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

                # Emit curriculum rebalanced event via centralized emitter (Dec 2025)
                try:
                    from app.coordination.event_emitters import emit_curriculum_updated
                    import asyncio

                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(emit_curriculum_updated(
                            config_key=config_key,
                            new_weight=priority.curriculum_weight,
                            trigger=f"opponent_mastered:{opponent_level}",
                        ))
                    else:
                        asyncio.run(emit_curriculum_updated(
                            config_key=config_key,
                            new_weight=priority.curriculum_weight,
                            trigger=f"opponent_mastered:{opponent_level}",
                        ))
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit curriculum rebalanced: {emit_err}")
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received opponent mastered for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling opponent mastered: {e}")

    def _on_training_early_stopped(self, event: Any) -> None:
        """Handle training early stopped event.

        P10-LOOP-1 (Dec 2025): When training early stops due to stagnation or regression,
        aggressively boost selfplay to generate fresh, diverse training data.

        This closes the feedback loop:
        TRAINING_EARLY_STOPPED → Selfplay boost → More diverse data → Better next training run

        Early stopping typically indicates:
        - Loss plateau (need more diverse positions)
        - Overfitting (need fresher data)
        - Regression (need different exploration)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "") or payload.get("config", "")
            reason = payload.get("reason", "unknown")
            final_loss = payload.get("final_loss", 0.0)
            epochs_completed = payload.get("epochs_completed", 0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Boost exploration significantly - we need diverse data
                old_boost = priority.exploration_boost
                if "regression" in reason.lower():
                    # Regression needs more aggressive exploration
                    priority.exploration_boost = min(2.0, old_boost * 1.5)
                elif "plateau" in reason.lower() or "stagnation" in reason.lower():
                    # Plateau needs moderate exploration boost
                    priority.exploration_boost = min(1.8, old_boost * 1.3)
                else:
                    # General early stop - moderate boost
                    priority.exploration_boost = min(1.5, old_boost * 1.2)

                # Also boost curriculum weight to prioritize this config
                old_weight = priority.curriculum_weight
                priority.curriculum_weight = min(2.0, old_weight * 1.3)

                # Mark as needing urgent data (increases staleness factor effect)
                priority.staleness_hours = max(priority.staleness_hours, STALE_DATA_THRESHOLD)

                logger.info(
                    f"[SelfplayScheduler] Training early stopped for {config_key} "
                    f"(reason={reason}, epochs={epochs_completed}, loss={final_loss:.4f}). "
                    f"Boosted exploration {old_boost:.2f} → {priority.exploration_boost:.2f}, "
                    f"curriculum_weight {old_weight:.2f} → {priority.curriculum_weight:.2f}"
                )

                # Emit SELFPLAY_TARGET_UPDATED to trigger immediate selfplay allocation
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                            "config_key": config_key,
                            "priority": "urgent",
                            "reason": f"training_early_stopped:{reason}",
                            "exploration_boost": priority.exploration_boost,
                            "curriculum_weight": priority.curriculum_weight,
                        })
                        logger.debug(
                            f"[SelfplayScheduler] Emitted urgent SELFPLAY_TARGET_UPDATED for {config_key}"
                        )
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received training early stopped for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training early stopped: {e}")

    def _on_elo_velocity_changed(self, event: Any) -> None:
        """Handle Elo velocity change event.

        P10-LOOP-3 (Dec 2025): Adjusts selfplay rate based on Elo velocity trends.

        This closes the feedback loop:
        ELO_VELOCITY_CHANGED → Selfplay rate adjustment → Optimal resource allocation

        Actions based on trend:
        - accelerating: Increase selfplay to capitalize on momentum
        - decelerating: Reduce selfplay, shift focus to training quality
        - stable: Maintain current allocation
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            velocity = payload.get("velocity", 0.0)
            previous_velocity = payload.get("previous_velocity", 0.0)
            trend = payload.get("trend", "stable")

            if config_key not in self._config_priorities:
                logger.debug(
                    f"[SelfplayScheduler] Received velocity change for unknown config: {config_key}"
                )
                return

            priority = self._config_priorities[config_key]

            # Update elo_velocity tracking
            priority.elo_velocity = velocity

            # Adjust momentum multiplier based on trend
            old_momentum = priority.momentum_multiplier

            if trend == "accelerating":
                # Capitalize on positive momentum - increase selfplay rate
                priority.momentum_multiplier = min(1.5, old_momentum * 1.2)
                logger.info(
                    f"[SelfplayScheduler] Accelerating velocity for {config_key}: "
                    f"{velocity:.1f} Elo/day. Boosted momentum {old_momentum:.2f} → {priority.momentum_multiplier:.2f}"
                )
            elif trend == "decelerating":
                # Slow down and focus on quality
                priority.momentum_multiplier = max(0.6, old_momentum * 0.85)
                logger.info(
                    f"[SelfplayScheduler] Decelerating velocity for {config_key}: "
                    f"{velocity:.1f} Elo/day. Reduced momentum {old_momentum:.2f} → {priority.momentum_multiplier:.2f}"
                )
            else:  # stable
                # Slight adjustment toward 1.0
                if old_momentum > 1.0:
                    priority.momentum_multiplier = max(1.0, old_momentum * 0.95)
                elif old_momentum < 1.0:
                    priority.momentum_multiplier = min(1.0, old_momentum * 1.05)

            # If velocity is negative, also boost exploration
            if velocity < 0:
                old_boost = priority.exploration_boost
                priority.exploration_boost = min(1.8, old_boost * 1.15)
                logger.info(
                    f"[SelfplayScheduler] Negative velocity for {config_key}: "
                    f"Boosted exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

            # Emit SELFPLAY_RATE_CHANGED if momentum changed by >20%
            if abs(priority.momentum_multiplier - old_momentum) / max(old_momentum, 0.01) > 0.20:
                change_percent = ((priority.momentum_multiplier - old_momentum) / old_momentum) * 100.0
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_RATE_CHANGED, {
                            "config_key": config_key,
                            "old_rate": old_momentum,
                            "new_rate": priority.momentum_multiplier,
                            "change_percent": change_percent,
                            "reason": f"elo_momentum:{trend}",
                        })
                        logger.debug(
                            f"[SelfplayScheduler] Emitted SELFPLAY_RATE_CHANGED for {config_key}: "
                            f"{old_momentum:.2f} → {priority.momentum_multiplier:.2f} ({change_percent:+.1f}%)"
                        )
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit SELFPLAY_RATE_CHANGED: {emit_err}")

            # Emit SELFPLAY_TARGET_UPDATED for downstream consumers
            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "normal",
                        "reason": f"velocity_changed:{trend}",
                        "momentum_multiplier": priority.momentum_multiplier,
                        "exploration_boost": priority.exploration_boost,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling elo velocity changed: {e}")

    def _on_exploration_boost(self, event: Any) -> None:
        """Handle exploration boost event from training feedback.

        P11-CRITICAL-1 (Dec 2025): React to EXPLORATION_BOOST events emitted
        by FeedbackLoopController when training anomalies (loss spikes, stalls)
        are detected.

        This closes the feedback loop:
        Training Anomaly → EXPLORATION_BOOST → Increased selfplay diversity

        Actions:
        - Update exploration_boost for the config
        - Increase temperature/noise in selfplay to generate diverse games
        - Emit SELFPLAY_TARGET_UPDATED for downstream consumers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            boost_factor = payload.get("boost_factor", 1.0)
            reason = payload.get("reason", "unknown")
            anomaly_count = payload.get("anomaly_count", 0)

            if config_key not in self._config_priorities:
                logger.debug(
                    f"[SelfplayScheduler] Received exploration boost for unknown config: {config_key}"
                )
                return

            priority = self._config_priorities[config_key]
            old_boost = priority.exploration_boost

            # Apply the boost factor directly from FeedbackLoopController
            priority.exploration_boost = max(priority.exploration_boost, boost_factor)

            # Phase 12: Set boost expiry (15 minutes from now by default)
            # This ensures temporary anomalies don't cause permanent boosts
            import time
            boost_duration = float(os.environ.get("RINGRIFT_EXPLORATION_BOOST_DURATION", "900"))  # 15 min default
            priority.exploration_boost_expires_at = time.time() + boost_duration

            logger.info(
                f"[SelfplayScheduler] Exploration boost for {config_key}: "
                f"{old_boost:.2f} → {priority.exploration_boost:.2f} "
                f"(reason={reason}, anomaly_count={anomaly_count}, expires_in={boost_duration}s)"
            )

            # Emit SELFPLAY_TARGET_UPDATED for downstream consumers
            try:
                from app.coordination.event_router import DataEventType, get_event_bus

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "high" if boost_factor > 1.3 else "normal",
                        "reason": f"exploration_boost:{reason}",
                        "exploration_boost": priority.exploration_boost,
                        "anomaly_count": anomaly_count,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")

            # Dec 2025: Also emit SELFPLAY_ALLOCATION_UPDATED for feedback loop tracking
            self._emit_allocation_updated(
                allocation=None,
                total_games=0,
                trigger=f"exploration_boost:{reason}",
                config_key=config_key,
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling exploration boost: {e}")

    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED event - curriculum stage progressed.

        Dec 2025: When a config achieves consecutive successful promotions,
        the curriculum advances. This signals we should shift focus to the
        next curriculum stage (harder opponents, more complex positions).

        Actions:
        - Update priority weights for the advanced config
        - Potentially reduce focus on "graduated" configs
        - Log curriculum progression for tracking
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            new_stage = payload.get("stage", 1)
            consecutive_promotions = payload.get("consecutive_promotions", 0)

            if not config_key:
                return

            logger.info(
                f"[SelfplayScheduler] CURRICULUM_ADVANCED: {config_key} "
                f"stage={new_stage}, consecutive_promotions={consecutive_promotions}"
            )

            # Update curriculum stage if tracked
            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Slightly reduce selfplay priority for graduated configs
                # to focus resources on configs that still need improvement
                if consecutive_promotions >= 3:
                    priority.curriculum_weight = max(0.5, priority.curriculum_weight * 0.9)
                    logger.debug(
                        f"[SelfplayScheduler] Reduced curriculum weight for {config_key}: "
                        f"{priority.curriculum_weight:.2f} (graduated)"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling curriculum advanced: {e}")

    def _on_adaptive_params_changed(self, event) -> None:
        """Handle ADAPTIVE_PARAMS_CHANGED event - training parameters adjusted.

        Dec 2025: When gauntlet feedback controller adjusts training parameters
        (learning rate, batch size, etc.), this event is emitted. We respond by
        adjusting selfplay parameters accordingly.

        Actions:
        - Update exploration parameters if temperature changed
        - Adjust selfplay rate if training intensity changed
        - Log parameter changes for tracking
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            param_type = payload.get("param_type", "")
            old_value = payload.get("old_value")
            new_value = payload.get("new_value")
            reason = payload.get("reason", "adaptive_adjustment")

            if not config_key:
                return

            logger.info(
                f"[SelfplayScheduler] ADAPTIVE_PARAMS_CHANGED: {config_key} "
                f"{param_type}: {old_value} → {new_value} (reason={reason})"
            )

            # Respond to temperature changes
            if param_type == "temperature" and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Higher temperature = more exploration, adjust boost accordingly
                if new_value and old_value and new_value > old_value:
                    # Temperature increased, boost exploration
                    priority.exploration_boost = max(priority.exploration_boost, 1.2)
                    logger.debug(f"[SelfplayScheduler] Boosted exploration for {config_key} due to temperature increase")

            # Respond to learning rate changes (training intensity)
            elif param_type == "learning_rate" and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Lower LR = more careful training, may need more selfplay data
                if new_value and old_value and new_value < old_value:
                    priority.target_games_multiplier = min(2.0, getattr(priority, 'target_games_multiplier', 1.0) * 1.1)
                    logger.debug(f"[SelfplayScheduler] Increased target games for {config_key} due to LR decrease")

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling adaptive params changed: {e}")

    def _decay_expired_boosts(self, now: float) -> int:
        """Decay exploration boosts that have expired.

        Phase 12 (Dec 2025): Prevents temporary training anomalies from causing
        permanent exploration boosts. After the boost expires, gradually decay
        back to 1.0 (normal exploration level).

        Args:
            now: Current timestamp

        Returns:
            Number of boosts that were decayed
        """
        decayed_count = 0

        for config_key, priority in self._config_priorities.items():
            # Skip if no boost active
            if priority.exploration_boost <= 1.0:
                continue

            # Skip if boost hasn't expired yet
            if priority.exploration_boost_expires_at > now:
                continue

            # Decay the boost
            old_boost = priority.exploration_boost
            priority.exploration_boost = 1.0
            priority.exploration_boost_expires_at = 0.0
            decayed_count += 1

            logger.info(
                f"[SelfplayScheduler] Exploration boost expired for {config_key}: "
                f"{old_boost:.2f} → 1.0"
            )

            # Emit event for downstream consumers
            try:
                from app.coordination.event_router import get_event_bus, DataEventType

                bus = get_event_bus()
                if bus:
                    bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                        "config_key": config_key,
                        "priority": "normal",
                        "reason": "exploration_boost_expired",
                        "exploration_boost": 1.0,
                    })
            except Exception as emit_err:
                logger.debug(f"[SelfplayScheduler] Failed to emit boost decay: {emit_err}")

        return decayed_count

    def _on_low_quality_warning(self, event: Any) -> None:
        """Handle LOW_QUALITY_DATA_WARNING to throttle selfplay allocation.

        Wire orphaned event (Dec 2025): When QualityMonitorDaemon detects quality
        below warning threshold, reduce selfplay allocation to avoid generating
        more low-quality data.

        This closes the feedback loop:
        Low quality detected → Throttle selfplay → Focus on training existing data

        Actions:
        - Reduce exploration_boost by 0.7x (throttle by 30%)
        - Apply temporary quality penalty
        - Log throttling action
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            quality_score = payload.get("quality_score", 0.0)
            old_state = payload.get("old_state", "unknown")
            new_state = payload.get("new_state", "unknown")

            logger.warning(
                f"[SelfplayScheduler] Low quality warning: "
                f"score={quality_score:.2f}, state={old_state} → {new_state}"
            )

            # Throttle all configs proportionally based on quality
            # Worse quality = more aggressive throttling
            if quality_score < 0.4:
                throttle_factor = 0.5  # 50% reduction for very poor quality
            elif quality_score < 0.5:
                throttle_factor = 0.6  # 40% reduction for poor quality
            else:
                throttle_factor = 0.7  # 30% reduction for marginal quality

            throttled_count = 0
            for config_key, priority in self._config_priorities.items():
                old_boost = priority.exploration_boost

                # Apply throttling to exploration boost
                priority.exploration_boost = max(0.5, priority.exploration_boost * throttle_factor)

                # Apply quality penalty
                old_penalty = priority.quality_penalty
                priority.quality_penalty = -0.15 * (1.0 - quality_score)  # -0.15 at quality=0, 0 at quality=1

                if abs(priority.exploration_boost - old_boost) > 0.01:
                    throttled_count += 1
                    logger.debug(
                        f"[SelfplayScheduler] Throttled {config_key}: "
                        f"exploration {old_boost:.2f} → {priority.exploration_boost:.2f}, "
                        f"quality_penalty {old_penalty:.3f} → {priority.quality_penalty:.3f}"
                    )

            logger.info(
                f"[SelfplayScheduler] Throttled selfplay for {throttled_count} configs "
                f"due to low quality (score={quality_score:.2f}, throttle={throttle_factor:.2f}x)"
            )

            # Emit QUALITY_PENALTY_APPLIED event for curriculum feedback (Dec 2025)
            if throttled_count > 0:
                self._emit_quality_penalty_applied(
                    quality_score=quality_score,
                    throttle_factor=throttle_factor,
                    throttled_count=throttled_count,
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling low quality warning: {e}")

    def _emit_quality_penalty_applied(
        self,
        quality_score: float,
        throttle_factor: float,
        throttled_count: int,
    ) -> None:
        """Emit QUALITY_PENALTY_APPLIED event for each throttled config.

        Closes the feedback loop: LOW_QUALITY_DATA_WARNING → penalty applied → curriculum adjustment.
        """
        try:
            from app.core.async_context import fire_and_forget
            from app.coordination.event_emitters import emit_quality_penalty_applied

            for config_key, priority in self._config_priorities.items():
                if priority.quality_penalty < 0:  # Only emit for penalized configs
                    async def emit(key=config_key, penalty=-priority.quality_penalty):
                        await emit_quality_penalty_applied(
                            config_key=key,
                            penalty=penalty,
                            reason="low_quality_selfplay_data",
                            current_weight=priority.exploration_boost,
                            source="selfplay_scheduler",
                            quality_score=quality_score,
                            throttle_factor=throttle_factor,
                        )

                    fire_and_forget(emit())

            logger.debug(
                f"[SelfplayScheduler] Emitted QUALITY_PENALTY_APPLIED for {throttled_count} configs"
            )

        except ImportError:
            logger.debug("[SelfplayScheduler] Event emitters not available for penalty emission")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit quality penalty: {e}")

    # =========================================================================
    # P2P Cluster Health Event Handlers (December 2025 - Critical Gap Fix)
    # =========================================================================

    def _on_node_unhealthy(self, event: Any) -> None:
        """Handle NODE_UNHEALTHY or P2P_NODE_DEAD - mark node as unavailable.

        December 2025: Prevents allocating selfplay to failing/unhealthy nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "")
            reason = payload.get("reason", "unknown")

            if node_id:
                if not hasattr(self, "_unhealthy_nodes"):
                    self._unhealthy_nodes: set[str] = set()

                self._unhealthy_nodes.add(node_id)

                # Also mark node as unavailable in capabilities if tracked
                if node_id in self._node_capabilities:
                    self._node_capabilities[node_id].current_load = 1.0  # Mark as fully loaded

                logger.warning(
                    f"[SelfplayScheduler] Node {node_id} marked unhealthy: {reason}. "
                    f"Will not allocate selfplay to this node."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node unhealthy: {e}")

    def _on_node_recovered(self, event: Any) -> None:
        """Handle NODE_RECOVERED - re-enable node for allocation.

        December 2025: Restores node availability after recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            node_id = payload.get("node_id", "")

            if node_id:
                unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())
                if node_id in unhealthy_nodes:
                    self._unhealthy_nodes.discard(node_id)

                    logger.info(
                        f"[SelfplayScheduler] Node {node_id} recovered. "
                        f"Re-enabled for selfplay allocation."
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node recovered: {e}")

    def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE - mark P2P peer as unavailable after retirement.

        December 2025: P2P orchestrator emits this when a peer is retired after
        ~300s of being offline. This is a stronger signal than NODE_UNHEALTHY
        as the node has been definitively removed from the cluster.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "") or payload.get("node_id", "")
            reason = payload.get("reason", "retired")

            if host:
                if not hasattr(self, "_unhealthy_nodes"):
                    self._unhealthy_nodes: set[str] = set()

                self._unhealthy_nodes.add(host)

                # Mark node as fully loaded to prevent allocation
                if host in self._node_capabilities:
                    self._node_capabilities[host].current_load = 1.0

                logger.warning(
                    f"[SelfplayScheduler] Host {host} offline (reason: {reason}). "
                    f"Removed from selfplay allocation pool."
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling host offline: {e}")

    def _on_cluster_unhealthy(self, event: Any) -> None:
        """Handle P2P_CLUSTER_UNHEALTHY - reduce allocation rate.

        December 2025: When cluster health degrades, reduce overall selfplay
        allocation to avoid overwhelming healthy nodes.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            healthy_nodes = payload.get("healthy_nodes", 0)
            total_nodes = payload.get("total_nodes", 0)

            if not hasattr(self, "_cluster_health_factor"):
                self._cluster_health_factor = 1.0

            if total_nodes > 0:
                self._cluster_health_factor = max(0.3, healthy_nodes / total_nodes)
            else:
                self._cluster_health_factor = 0.5

            logger.warning(
                f"[SelfplayScheduler] Cluster unhealthy: {healthy_nodes}/{total_nodes}. "
                f"Reducing allocation to {self._cluster_health_factor:.0%}."
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling cluster unhealthy: {e}")

    def _on_cluster_healthy(self, event: Any) -> None:
        """Handle P2P_CLUSTER_HEALTHY - restore normal allocation.

        December 2025: Restores full allocation when cluster recovers.
        """
        try:
            logger.info("[SelfplayScheduler] Cluster healthy. Restoring normal allocation.")

            if hasattr(self, "_cluster_health_factor"):
                self._cluster_health_factor = 1.0

            # Clear unhealthy node tracking
            if hasattr(self, "_unhealthy_nodes"):
                self._unhealthy_nodes.clear()

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling cluster healthy: {e}")

    # =========================================================================
    # Status & Metrics
    # =========================================================================

    def _record_allocation(self, games_allocated: int) -> None:
        """Record allocation metrics for rolling throughput tracking."""
        if games_allocated <= 0:
            return
        now = time.time()
        self._games_allocated_total += games_allocated
        self._allocation_history.append((now, games_allocated))

        cutoff = now - self._allocation_window_seconds
        while self._allocation_history and self._allocation_history[0][0] < cutoff:
            self._allocation_history.popleft()

    def _emit_allocation_updated(
        self,
        allocation: dict[str, dict[str, int]] | None,
        total_games: int,
        trigger: str,
        config_key: str | None = None,
    ) -> None:
        """Emit SELFPLAY_ALLOCATION_UPDATED event.

        December 2025: Notifies downstream consumers (IdleResourceDaemon, feedback
        loops) when selfplay allocation has changed. This enables:
        - IdleResourceDaemon to know which configs are prioritized
        - Feedback loops to track allocation changes from their signals
        - Monitoring to track allocation patterns over time

        Args:
            allocation: Dict of config_key -> {node_id: games} for batch allocations
            total_games: Total games in this allocation
            trigger: What caused this allocation (e.g., "allocate_batch", "exploration_boost")
            config_key: Specific config that changed (for single-config updates)
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            # Build allocation summary
            if allocation:
                configs_allocated = list(allocation.keys())
                nodes_involved = set()
                for node_games in allocation.values():
                    nodes_involved.update(node_games.keys())
            else:
                configs_allocated = [config_key] if config_key else []
                nodes_involved = set()

            payload = {
                "trigger": trigger,
                "total_games": total_games,
                "configs": configs_allocated,
                "node_count": len(nodes_involved),
                "timestamp": time.time(),
            }

            # Include exploration boosts for tracking feedback loop efficacy
            if config_key and config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                payload["exploration_boost"] = priority.exploration_boost
                payload["curriculum_weight"] = priority.curriculum_weight

            bus.emit(DataEventType.SELFPLAY_ALLOCATION_UPDATED, payload)
            logger.debug(
                f"[SelfplayScheduler] Emitted SELFPLAY_ALLOCATION_UPDATED: "
                f"trigger={trigger}, games={total_games}, configs={len(configs_allocated)}"
            )

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to emit allocation update: {e}")

    # =========================================================================
    # Per-Node Job Targeting (Dec 2025)
    # =========================================================================

    def get_target_jobs_for_node(self, node: Any) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Args:
            node: NodeInfo-like object with attributes:
                - node_id: str
                - memory_gb: int
                - has_gpu: bool
                - cpu_count: int
                - cpu_percent: float
                - memory_percent: float
                - disk_percent: float
                - gpu_percent: float
                - gpu_memory_percent: float
                - selfplay_jobs: int
                - gpu_name: str (optional)
                - gpu_count: int (optional)

        Returns:
            Target number of selfplay jobs for this node (always >= 1 unless blocked)
        """
        # Check safeguards first
        if self._is_emergency_active_fn:
            try:
                if self._is_emergency_active_fn():
                    return 0
            except (TypeError, AttributeError, RuntimeError) as e:
                # Dec 2025: Narrow exception - callback may be misconfigured
                logger.debug(f"[SelfplayScheduler] Emergency check callback error: {e}")

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if self._should_stop_production_fn:
            try:
                # Import QueueType lazily to avoid circular imports
                from app.coordination.backpressure import QueueType
                if self._should_stop_production_fn(QueueType.TRAINING_DATA):
                    if self._verbose:
                        logger.info(f"Backpressure STOP: training queue full, halting selfplay on {getattr(node, 'node_id', 'unknown')}")
                    return 0
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure stop check error: {e}")

        if self._should_throttle_production_fn and self._get_throttle_factor_fn:
            try:
                from app.coordination.backpressure import QueueType
                if self._should_throttle_production_fn(QueueType.TRAINING_DATA):
                    backpressure_factor = self._get_throttle_factor_fn(QueueType.TRAINING_DATA)
                    if self._verbose:
                        logger.info(f"Backpressure throttle: factor={backpressure_factor:.2f}")
            except Exception as e:
                if self._verbose:
                    logger.debug(f"Backpressure throttle check error: {e}")

        # Extract node metrics
        node_id = getattr(node, "node_id", "unknown")
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)
        gpu_name = getattr(node, "gpu_name", "") or ""
        gpu_count = int(getattr(node, "gpu_count", 1) or 1) if has_gpu else 0

        # Minimum memory requirement
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Record utilization for adaptive feedback
        if self._record_utilization_fn:
            with contextlib.suppress(Exception):
                self._record_utilization_fn(node_id, cpu_percent, gpu_percent, mem_percent, current_jobs)

        # Try unified resource targets if available
        if self._get_host_targets_fn and self._get_target_job_count_fn:
            try:
                host_targets = self._get_host_targets_fn(node_id)
                target_selfplay = self._get_target_job_count_fn(
                    node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Scale up if underutilized
                if self._should_scale_up_fn:
                    scale_up, reason = self._should_scale_up_fn(
                        node_id, cpu_percent, gpu_percent, current_jobs
                    )
                    if scale_up and current_jobs < target_selfplay:
                        scale_up_increment = min(4, target_selfplay - current_jobs)
                        target_selfplay = current_jobs + scale_up_increment
                        if self._verbose:
                            logger.info(f"Scale-up on {node_id}: {reason}, target={target_selfplay}")

                # Scale down if overloaded
                if self._should_scale_down_fn:
                    scale_down, reduction, reason = self._should_scale_down_fn(
                        node_id, cpu_percent, gpu_percent, mem_percent
                    )
                    if scale_down:
                        target_selfplay = max(1, current_jobs - reduction)
                        logger.info(f"Scale-down on {node_id}: {reason}, target={target_selfplay}")

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                max_selfplay = getattr(host_targets, "max_selfplay", target_selfplay)
                target_selfplay = min(target_selfplay, max_selfplay)

                return int(max(1, target_selfplay))

            except Exception as e:
                if self._verbose:
                    logger.info(f"Resource targets error, falling back to hardware-aware: {e}")

        # FALLBACK: Use hardware-aware limits
        if self._get_max_selfplay_for_node_fn:
            max_selfplay = self._get_max_selfplay_for_node_fn(
                node_id=node_id,
                gpu_count=gpu_count,
                gpu_name=gpu_name,
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                has_gpu=has_gpu,
            )
        else:
            # Minimal fallback when callback unavailable
            max_selfplay = self._compute_hardware_limit(
                has_gpu, gpu_name, gpu_count, cpu_count, memory_gb
            )

        target_selfplay = max_selfplay

        # Utilization-aware adjustments
        gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
        cpu_overloaded = cpu_percent > 80
        gpu_has_headroom = gpu_percent < 60 and gpu_mem_percent < 75
        cpu_has_headroom = cpu_percent < 60

        if gpu_overloaded:
            target_selfplay = max(2, target_selfplay - 2)
        if cpu_overloaded:
            target_selfplay = max(2, target_selfplay - 1)

        if ((has_gpu and gpu_has_headroom and cpu_has_headroom) or
            (not has_gpu and cpu_has_headroom)) and current_jobs < target_selfplay:
            target_selfplay = min(target_selfplay, current_jobs + 2)

        # Resource pressure warnings
        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        return int(max(1, target_selfplay))

    def _compute_hardware_limit(
        self,
        has_gpu: bool,
        gpu_name: str,
        gpu_count: int,
        cpu_count: int,
        memory_gb: int,
    ) -> int:
        """Compute hardware-based max selfplay limit.

        This is a fallback when resource_optimizer callbacks are unavailable.
        """
        if has_gpu:
            gpu_upper = gpu_name.upper()
            if any(g in gpu_upper for g in ["GH200"]):
                return int(cpu_count * 0.8) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["H100", "H200"]):
                return min(int(cpu_count * 0.5), 48) if cpu_count > 0 else 32
            elif any(g in gpu_upper for g in ["A100", "L40"]):
                return min(int(cpu_count * 0.4), 32) if cpu_count > 0 else 24
            elif any(g in gpu_upper for g in ["5090"]):
                return min(int(cpu_count * 0.3), gpu_count * 12, 64) if cpu_count > 0 else 48
            elif any(g in gpu_upper for g in ["A10", "4090", "3090"]):
                return min(int(cpu_count * 0.3), 24) if cpu_count > 0 else 16
            elif any(g in gpu_upper for g in ["4080", "4070", "3080", "4060"]):
                return min(int(cpu_count * 0.25), 12) if cpu_count > 0 else 8
            elif any(g in gpu_upper for g in ["3070", "3060", "2060", "2070", "2080"]):
                return min(int(cpu_count * 0.2), 10) if cpu_count > 0 else 6
            else:
                return min(int(cpu_count * 0.2), 8) if cpu_count > 0 else 6
        else:
            return min(int(cpu_count * 0.3), 32) if cpu_count > 0 else 8

    def get_metrics(self) -> dict[str, Any]:
        """Get throughput metrics for monitoring."""
        recent_games = sum(games for _, games in self._allocation_history)
        window_hours = self._allocation_window_seconds / 3600.0
        games_per_hour = recent_games / window_hours if window_hours > 0 else 0.0

        return {
            "games_allocated_total": self._games_allocated_total,
            "games_allocated_last_hour": recent_games,
            "games_per_hour": games_per_hour,
            "allocation_window_seconds": self._allocation_window_seconds,
        }

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "subscribed": self._subscribed,
            "last_priority_update": self._last_priority_update,
            "node_count": len(self._node_capabilities),
            "config_priorities": {
                cfg: {
                    "priority_score": p.priority_score,
                    "staleness_hours": p.staleness_hours,
                    "elo_velocity": p.elo_velocity,
                    "exploration_boost": p.exploration_boost,
                    "curriculum_weight": p.curriculum_weight,
                    "games_allocated": p.games_allocated,
                }
                for cfg, p in self._config_priorities.items()
            },
        }

    def get_top_priorities(self, n: int = 5) -> list[dict[str, Any]]:
        """Get top N priority configurations with details."""
        sorted_configs = sorted(
            self._config_priorities.values(),
            key=lambda p: -p.priority_score,
        )

        return [
            {
                "config": p.config_key,
                "priority": p.priority_score,
                "staleness_hours": p.staleness_hours,
                "elo_velocity": p.elo_velocity,
                "exploration_boost": p.exploration_boost,
                "curriculum_weight": p.curriculum_weight,
            }
            for p in sorted_configs[:n]
        ]

    def health_check(self) -> HealthCheckResult:
        """Check scheduler health.

        Returns:
            Health check result with scheduler status and metrics.
        """
        # Calculate games in allocation window
        current_time = time.time()
        games_in_window = sum(
            count for ts, count in self._allocation_history
            if current_time - ts < self._allocation_window_seconds
        )

        # Determine health status
        stale_priority = current_time - self._last_priority_update > 300  # 5 min
        healthy = self._subscribed and not stale_priority

        message = "Running" if healthy else (
            "Not subscribed to events" if not self._subscribed else
            "Priority data stale (>5 min)"
        )

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details={
                "subscribed": self._subscribed,
                "configs_tracked": len(self._config_priorities),
                "nodes_tracked": len(self._node_capabilities),
                "last_priority_update": self._last_priority_update,
                "priority_age_seconds": current_time - self._last_priority_update,
                "games_allocated_total": self._games_allocated_total,
                "games_in_last_hour": games_in_window,
            },
        )


# =============================================================================
# Singleton
# =============================================================================

_scheduler_instance: SelfplayScheduler | None = None


def get_selfplay_scheduler() -> SelfplayScheduler:
    """Get the singleton SelfplayScheduler instance."""
    global _scheduler_instance

    if _scheduler_instance is None:
        _scheduler_instance = SelfplayScheduler()
        _scheduler_instance.subscribe_to_events()

    return _scheduler_instance


def reset_selfplay_scheduler() -> None:
    """Reset the scheduler singleton (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None
