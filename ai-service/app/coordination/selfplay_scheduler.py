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
    "DynamicWeights",  # December 29, 2025: Now exported
    "NodeCapability",
    "SelfplayScheduler",
    # Constants (also available via SelfplayPriorityWeightDefaults in coordination_defaults)
    "ALL_CONFIGS",
    "CURRICULUM_WEIGHT",
    "DATA_DEFICIT_WEIGHT",
    "DATA_STARVATION_CRITICAL_MULTIPLIER",
    "DATA_STARVATION_CRITICAL_THRESHOLD",
    "DATA_STARVATION_EMERGENCY_MULTIPLIER",
    "DATA_STARVATION_EMERGENCY_THRESHOLD",
    "DATA_STARVATION_ULTRA_MULTIPLIER",
    "DATA_STARVATION_ULTRA_THRESHOLD",
    "DEFAULT_GAMES_PER_CONFIG",
    "DEFAULT_TRAINING_SAMPLES_TARGET",
    "DYNAMIC_WEIGHT_BOUNDS",  # December 29, 2025: Now exported
    "ELO_VELOCITY_WEIGHT",
    "EXPLORATION_BOOST_WEIGHT",
    "FRESH_DATA_THRESHOLD",
    "IMPROVEMENT_BOOST_WEIGHT",
    "LARGE_BOARD_TARGET_MULTIPLIER",
    "MAX_STALENESS_HOURS",
    "MIN_GAMES_PER_ALLOCATION",
    "PRIORITY_OVERRIDE_MULTIPLIERS",
    "QUALITY_WEIGHT",  # December 29, 2025: Now exported
    "SAMPLES_PER_GAME_BY_BOARD",
    "STALE_DATA_THRESHOLD",
    "STALENESS_WEIGHT",
    "TRAINING_NEED_WEIGHT",
    "VOI_WEIGHT",
    "VOI_SAMPLE_COST_BY_BOARD",
    # Functions
    "get_selfplay_scheduler",
    "reset_selfplay_scheduler",
    # New Dec 2025
    "get_priority_configs_sync",
]

import contextlib
import logging
import math
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

# December 2025: Constants consolidated in priority_calculator.py
# December 29, 2025: Now also imports PriorityCalculator for delegation
from app.coordination.priority_calculator import (
    ALL_CONFIGS,
    ClusterState,
    PLAYER_COUNT_ALLOCATION_MULTIPLIER,
    PriorityCalculator,
    PriorityInputs,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    SAMPLES_PER_GAME_BY_BOARD,
    VOI_SAMPLE_COST_BY_BOARD,
    compute_dynamic_weights,
)

# December 2025: NodeCapability extracted to node_allocator.py
from app.coordination.node_allocator import NodeCapability

# December 2025: Budget calculation extracted to budget_calculator.py
from app.coordination.budget_calculator import (
    get_adaptive_budget_for_elo as _get_budget_for_elo,
    get_adaptive_budget_for_games as _get_budget_for_games,
    compute_target_games as _compute_target,
    parse_config_key,
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
# Note: ALL_CONFIGS, PRIORITY_OVERRIDE_MULTIPLIERS, PLAYER_COUNT_ALLOCATION_MULTIPLIER,
# SAMPLES_PER_GAME_BY_BOARD, and VOI_SAMPLE_COST_BY_BOARD are imported from
# app.coordination.priority_calculator (December 2025 consolidation)

# Priority calculation weights (BASE values - adjusted dynamically)
# Dec 29, 2025: These are now baseline weights that get adjusted based on cluster state.
# See _compute_dynamic_weights() for adaptive reweighting logic.
#
# IMPORTANT: For runtime tuning via environment variables, use SelfplayPriorityWeightDefaults
# from app.config.coordination_defaults. Example:
#   export RINGRIFT_STALENESS_WEIGHT=0.40  # Boost data freshness priority
#   export RINGRIFT_ELO_VELOCITY_WEIGHT=0.15  # Reduce velocity priority
#
# The module-level constants below are for backward compatibility and internal use.
# New code should import from coordination_defaults for env var support.
from app.config.coordination_defaults import SelfplayPriorityWeightDefaults

# Create singleton instance for env var resolution
_priority_weight_defaults = SelfplayPriorityWeightDefaults()

STALENESS_WEIGHT = _priority_weight_defaults.STALENESS_WEIGHT
ELO_VELOCITY_WEIGHT = _priority_weight_defaults.ELO_VELOCITY_WEIGHT
TRAINING_NEED_WEIGHT = _priority_weight_defaults.TRAINING_NEED_WEIGHT
EXPLORATION_BOOST_WEIGHT = _priority_weight_defaults.EXPLORATION_BOOST_WEIGHT
CURRICULUM_WEIGHT = _priority_weight_defaults.CURRICULUM_WEIGHT
IMPROVEMENT_BOOST_WEIGHT = _priority_weight_defaults.IMPROVEMENT_BOOST_WEIGHT
DATA_DEFICIT_WEIGHT = _priority_weight_defaults.DATA_DEFICIT_WEIGHT
QUALITY_WEIGHT = _priority_weight_defaults.QUALITY_WEIGHT
VOI_WEIGHT = _priority_weight_defaults.VOI_WEIGHT

# =============================================================================
# Dynamic Weight Bounds (Dec 29, 2025)
# =============================================================================
# Min/max bounds for dynamic weight adjustment based on cluster state
# These prevent any single factor from dominating allocation decisions
# Now sourced from centralized config with env var support
DYNAMIC_WEIGHT_BOUNDS = _priority_weight_defaults.get_weight_bounds()

# VOI target Elo (from coordination_defaults)
VOI_ELO_TARGET = _priority_weight_defaults.VOI_ELO_TARGET

# Thresholds for dynamic weight adjustment triggers (now env-configurable)
IDLE_GPU_HIGH_THRESHOLD = _priority_weight_defaults.IDLE_GPU_HIGH_THRESHOLD
IDLE_GPU_LOW_THRESHOLD = _priority_weight_defaults.IDLE_GPU_LOW_THRESHOLD
TRAINING_QUEUE_HIGH_THRESHOLD = _priority_weight_defaults.TRAINING_QUEUE_HIGH_THRESHOLD
CONFIGS_AT_TARGET_THRESHOLD = _priority_weight_defaults.CONFIGS_AT_TARGET_THRESHOLD
ELO_HIGH_THRESHOLD = _priority_weight_defaults.ELO_HIGH_THRESHOLD
ELO_MEDIUM_THRESHOLD = _priority_weight_defaults.ELO_MEDIUM_THRESHOLD

# Target games per config for data deficit calculation
TARGET_GAMES_FOR_2000_ELO = _priority_weight_defaults.TARGET_GAMES_FOR_2000_ELO
LARGE_BOARD_TARGET_MULTIPLIER = _priority_weight_defaults.LARGE_BOARD_TARGET_MULTIPLIER

# Dec 29, 2025: Data starvation thresholds (now env-configurable)
# Configs with fewer games than these thresholds get priority boosts
# Especially critical for 4-player configs which have near-zero games
# ULTRA tier added Dec 29, 2025 for critically starved configs (< 20 games)
DATA_STARVATION_ULTRA_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_ULTRA_THRESHOLD
DATA_STARVATION_EMERGENCY_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_THRESHOLD
DATA_STARVATION_CRITICAL_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_CRITICAL_THRESHOLD
DATA_STARVATION_ULTRA_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_ULTRA_MULTIPLIER
DATA_STARVATION_EMERGENCY_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_MULTIPLIER
DATA_STARVATION_CRITICAL_MULTIPLIER = _priority_weight_defaults.DATA_STARVATION_CRITICAL_MULTIPLIER

# Default training sample target per config
DEFAULT_TRAINING_SAMPLES_TARGET = 50000

# Staleness thresholds (hours) - now env-configurable
FRESH_DATA_THRESHOLD = _priority_weight_defaults.FRESH_DATA_THRESHOLD
STALE_DATA_THRESHOLD = _priority_weight_defaults.STALE_DATA_THRESHOLD
MAX_STALENESS_HOURS = _priority_weight_defaults.MAX_STALENESS_HOURS

# Default allocation (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import SelfplayAllocationDefaults

DEFAULT_GAMES_PER_CONFIG = SelfplayAllocationDefaults.GAMES_PER_CONFIG
MIN_GAMES_PER_ALLOCATION = SelfplayAllocationDefaults.MIN_GAMES_PER_ALLOCATION

# Resource management thresholds (for get_target_jobs_for_node)
MIN_MEMORY_GB_FOR_TASKS = SelfplayAllocationDefaults.MIN_MEMORY_GB
DISK_WARNING_THRESHOLD = SelfplayAllocationDefaults.DISK_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = SelfplayAllocationDefaults.MEMORY_WARNING_THRESHOLD


@dataclass
class DynamicWeights:
    """Dynamically computed priority weights based on cluster state.

    Dec 29, 2025: Implements adaptive reweighting to optimize resource allocation.
    Weights are adjusted based on:
    - Idle GPU fraction: More idle GPUs → boost staleness weight (generate more data)
    - Training backlog: Large queue → reduce staleness weight (don't flood queue)
    - Configs at Elo target: Many at target → reduce velocity weight (focus elsewhere)
    - Average model Elo: Higher Elo → boost curriculum weight (harder positions needed)

    All weights are bounded by DYNAMIC_WEIGHT_BOUNDS to prevent any single factor
    from dominating allocation decisions.
    """
    staleness: float = STALENESS_WEIGHT
    velocity: float = ELO_VELOCITY_WEIGHT
    training: float = TRAINING_NEED_WEIGHT
    exploration: float = EXPLORATION_BOOST_WEIGHT
    curriculum: float = CURRICULUM_WEIGHT
    improvement: float = IMPROVEMENT_BOOST_WEIGHT
    data_deficit: float = DATA_DEFICIT_WEIGHT
    quality: float = QUALITY_WEIGHT  # Dec 29, 2025: Data quality weight
    voi: float = VOI_WEIGHT  # Dec 29, 2025: Value of Information weight

    # Cluster state that drove these weights (for debugging/logging)
    idle_gpu_fraction: float = 0.0
    training_queue_depth: int = 0
    configs_at_target_fraction: float = 0.0
    average_elo: float = 1500.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for logging."""
        return {
            "staleness": self.staleness,
            "velocity": self.velocity,
            "training": self.training,
            "exploration": self.exploration,
            "curriculum": self.curriculum,
            "improvement": self.improvement,
            "data_deficit": self.data_deficit,
            "quality": self.quality,
            "voi": self.voi,
            "idle_gpu_fraction": self.idle_gpu_fraction,
            "training_queue_depth": self.training_queue_depth,
            "configs_at_target_fraction": self.configs_at_target_fraction,
            "average_elo": self.average_elo,
        }


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
    architecture_boost: float = 0.0  # Phase 5B: From ArchitectureTracker (0.0 to +0.30)
    momentum_multiplier: float = 1.0  # Phase 19: From FeedbackAccelerator (0.5 to 1.5)
    game_count: int = 0  # Dec 2025: Current game count for this config
    is_large_board: bool = False  # Dec 2025: True for square19, hexagonal
    priority_override: int = 3  # Dec 2025: From config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
    search_budget: int = 400  # Dec 28 2025: Gumbel MCTS budget from velocity feedback
    current_elo: float = 1500.0  # Dec 29 2025: Current Elo rating for dynamic weight calculation

    # Dec 29, 2025: VOI (Value of Information) based prioritization
    # Uncertainty in Elo estimate (confidence interval width in Elo points)
    elo_uncertainty: float = 200.0  # Default high uncertainty for new configs
    # Target Elo for this config
    target_elo: float = 2000.0

    # Dec 29, 2025: Game count normalization fields
    target_training_samples: int = DEFAULT_TRAINING_SAMPLES_TARGET  # Target samples for training
    samples_per_game_estimate: float = 50.0  # Historical average samples per game

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
    @property
    def player_count(self) -> int:
        """Extract player count from config_key (e.g., 'hex8_2p' -> 2).

        Dec 29, 2025: Added for player-count based allocation multiplier.
        """
        try:
            suffix = self.config_key.split("_")[-1]  # "2p", "3p", "4p"
            return int(suffix.rstrip("p"))
        except (ValueError, IndexError):
            return 2  # Default to 2-player

    @property
    def games_needed(self) -> int:
        """Calculate games needed to reach training sample target.

        Dec 29, 2025: Part of game count normalization feature.
        Uses current game count and samples-per-game estimate to determine
        how many more games are needed to meet the training target.

        Returns:
            Number of additional games needed (0 if target already met)
        """
        if self.samples_per_game_estimate <= 0:
            return 0
        current_samples = int(self.game_count * self.samples_per_game_estimate)
        remaining_samples = max(0, self.target_training_samples - current_samples)
        return int(remaining_samples / self.samples_per_game_estimate)

    @property
    def elo_gap(self) -> float:
        """Gap between current Elo and target Elo.

        Dec 29, 2025: Added for VOI-based prioritization.
        """
        return max(0.0, self.target_elo - self.current_elo)

    @property
    def info_gain_per_game(self) -> float:
        """Estimated information gain (uncertainty reduction) per new game.

        Dec 29, 2025: Added for VOI-based prioritization.
        Uses 1/sqrt(n) rule from statistical sampling theory.
        Each new game reduces Elo CI width by approximately this amount.
        """
        import math
        if self.game_count <= 0:
            return self.elo_uncertainty  # First game has max info gain
        # Standard error reduces with sqrt(n)
        return self.elo_uncertainty / math.sqrt(self.game_count)

    @property
    def voi_score(self) -> float:
        """Value of Information score for this config.

        Dec 29, 2025: Added for VOI-based prioritization.
        Combines:
        - Uncertainty (high uncertainty = high value of new data)
        - Elo gap (far from target = higher value)
        - Info gain efficiency (how much each game reduces uncertainty)

        Higher score = higher priority for resource allocation.
        """
        # Normalize components to 0-1 range
        uncertainty_factor = min(1.0, self.elo_uncertainty / 300.0)  # 300 Elo = max uncertainty
        gap_factor = min(1.0, self.elo_gap / 500.0)  # 500 Elo gap = max

        # Info gain normalized by reference (10 Elo/game is high)
        info_factor = min(1.0, self.info_gain_per_game / 10.0)

        # Combined VOI: weighted sum of factors
        # High uncertainty + high gap + high info gain = high VOI
        return (
            uncertainty_factor * 0.4 +  # 40% weight on uncertainty
            gap_factor * 0.3 +          # 30% weight on improvement potential
            info_factor * 0.3           # 30% weight on learning efficiency
        )


# Note: NodeCapability is now imported from app.coordination.node_allocator
# (December 2025 consolidation - removed ~20 lines of duplicate code)


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

        # Dec 29, 2025: Dynamic priority weights (adjusted based on cluster state)
        self._dynamic_weights = DynamicWeights()
        self._last_dynamic_weights_update = 0.0
        self._dynamic_weights_update_interval = 60.0  # Update every 60 seconds

        # Dec 29, 2025 - Phase 2: Elo velocity tracking
        # Track Elo history per config: list of (timestamp, elo) tuples
        # Used to compute Elo/hour velocity for priority adjustment
        self._elo_history: dict[str, list[tuple[float, float]]] = {}
        self._elo_velocity: dict[str, float] = {}  # Computed Elo change per hour

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

        # December 29, 2025: PriorityCalculator for delegated score computation
        # Callbacks are bound methods so PriorityCalculator can access scheduler state
        self._priority_calculator = PriorityCalculator(
            dynamic_weights=self._dynamic_weights,
            get_quality_score_fn=self._get_config_data_quality,
            get_elo_velocity_fn=self.get_elo_velocity,
            get_cascade_priority_fn=self._get_cascade_priority,
            data_starvation_emergency_threshold=DATA_STARVATION_EMERGENCY_THRESHOLD,
            data_starvation_critical_threshold=DATA_STARVATION_CRITICAL_THRESHOLD,
            data_starvation_emergency_multiplier=DATA_STARVATION_EMERGENCY_MULTIPLIER,
            data_starvation_critical_multiplier=DATA_STARVATION_CRITICAL_MULTIPLIER,
        )

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
                except (yaml.YAMLError, OSError, KeyError, AttributeError) as e:
                    logger.warning(f"[SelfplayScheduler] Failed to load config {config_path}: {e}")

    # =========================================================================
    # Game Count Normalization (Dec 29, 2025)
    # =========================================================================

    def _get_samples_per_game_estimate(self, config_key: str) -> float:
        """Get estimated samples per game for a config.

        Uses historical averages from SAMPLES_PER_GAME_BY_BOARD, falling back
        to a conservative default of 50 samples/game if config not found.

        Args:
            config_key: Config key like "hex8_2p", "square19_4p"

        Returns:
            Estimated samples per game
        """
        try:
            parts = config_key.split("_")
            if len(parts) >= 2:
                board_type = parts[0]  # "hex8", "square8", etc.
                player_key = parts[1]  # "2p", "3p", "4p"
                if board_type in SAMPLES_PER_GAME_BY_BOARD:
                    return float(SAMPLES_PER_GAME_BY_BOARD[board_type].get(player_key, 50))
        except (ValueError, IndexError):
            pass
        return 50.0  # Conservative default

    def set_target_training_samples(self, config_key: str, target_samples: int) -> None:
        """Set training sample target for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        DataPipelineOrchestrator calls this before requesting export to
        communicate how many samples are needed for training.

        This updates the ConfigPriority's target_training_samples and
        samples_per_game_estimate, which are then used by games_needed
        property to determine how many more games should be generated.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")
            target_samples: Number of training samples needed

        Example:
            scheduler.set_target_training_samples("hex8_2p", 100000)
            # Now scheduler.get_games_needed("hex8_2p") returns games needed
        """
        if config_key not in self._config_priorities:
            logger.warning(
                f"[SelfplayScheduler] Unknown config {config_key}, creating priority entry"
            )
            self._config_priorities[config_key] = ConfigPriority(config_key=config_key)

        priority = self._config_priorities[config_key]
        # Validate target: 0 or negative keeps existing value or uses default
        if target_samples <= 0:
            if priority.target_training_samples <= 0:
                # Use default if no existing value
                priority.target_training_samples = 100000
            # Otherwise keep existing value
            logger.debug(
                f"[SelfplayScheduler] Target {target_samples} <= 0 for {config_key}, "
                f"using existing/default: {priority.target_training_samples}"
            )
        else:
            priority.target_training_samples = target_samples
        priority.samples_per_game_estimate = self._get_samples_per_game_estimate(config_key)

        games_needed = priority.games_needed
        logger.info(
            f"[SelfplayScheduler] Set training target for {config_key}: "
            f"{target_samples} samples, {priority.samples_per_game_estimate:.0f} samples/game estimate, "
            f"{games_needed} games needed"
        )

    def get_games_needed(self, config_key: str) -> int:
        """Get number of additional games needed for a configuration.

        Dec 29, 2025: Part of game count normalization feature.
        Returns how many more games are needed to meet the training sample target.

        Args:
            config_key: Config key (e.g., "hex8_2p", "square19_4p")

        Returns:
            Number of games needed (0 if target already met or config unknown)
        """
        if config_key not in self._config_priorities:
            return 0
        return self._config_priorities[config_key].games_needed

    def get_all_games_needed(self) -> dict[str, int]:
        """Get games needed for all configurations.

        Dec 29, 2025: Part of game count normalization feature.
        Returns a dict mapping config_key to games_needed for all configs.

        Returns:
            Dict[config_key, games_needed] for all tracked configurations
        """
        return {
            cfg: priority.games_needed
            for cfg, priority in self._config_priorities.items()
        }

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

        # Get architecture boosts (Phase 5B)
        architecture_data = self._get_architecture_boosts()

        # Get game counts (Dec 2025)
        game_count_data = await self._get_game_counts()

        # Get current Elos for adaptive budget (Dec 29, 2025)
        elo_current_data = await self._get_current_elos()

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

            # Update architecture boost (Phase 5B)
            if config_key in architecture_data:
                priority.architecture_boost = architecture_data[config_key]

            # Update game count and large board flag (Dec 2025)
            if config_key in game_count_data:
                priority.game_count = game_count_data[config_key]
            # Mark large boards for higher data deficit weight
            priority.is_large_board = config_key.startswith("square19") or config_key.startswith("hexagonal")

            # Update current Elo and search budget (Dec 29, 2025)
            # Budget now considers BOTH game count AND Elo:
            # - Low game count (<1000): Use bootstrap budgets for faster data generation
            # - High game count (>=1000): Use Elo-based budgets for quality
            if config_key in elo_current_data:
                current_elo = elo_current_data[config_key]
                priority.current_elo = current_elo  # Store for dynamic weight calculation
                game_count = priority.game_count
                new_budget = self._get_adaptive_budget_for_games(game_count, current_elo)
                old_budget = priority.search_budget
                if new_budget != old_budget:
                    priority.search_budget = new_budget
                    logger.info(
                        f"[SelfplayScheduler] Adaptive budget for {config_key}: "
                        f"{old_budget}→{new_budget} (games={game_count}, Elo={current_elo:.0f})"
                    )

            # Dec 29, 2025: Update Elo uncertainty for VOI calculation
            # Uncertainty decreases with more games (statistical sampling theory)
            # Base uncertainty of 300 Elo, reduces with sqrt(game_count)
            BASE_UNCERTAINTY = 300.0
            MIN_UNCERTAINTY = 30.0  # Floor to prevent near-zero uncertainty
            if priority.game_count > 0:
                priority.elo_uncertainty = max(
                    MIN_UNCERTAINTY,
                    BASE_UNCERTAINTY / math.sqrt(priority.game_count)
                )
            else:
                priority.elo_uncertainty = BASE_UNCERTAINTY

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

    def _compute_dynamic_weights(self) -> DynamicWeights:
        """Compute dynamic priority weights based on current cluster state.

        Dec 29, 2025: Implements adaptive reweighting to optimize resource allocation.
        Now delegates to priority_calculator.compute_dynamic_weights() for the actual
        weight computation logic.

        Weight adjustment logic:
        - High idle GPU fraction → Boost staleness weight (generate more data)
        - Large training queue → Reduce staleness weight (don't flood queue)
        - Many configs at Elo target → Reduce velocity weight (focus on struggling configs)
        - High average Elo → Boost curriculum weight (need harder positions)

        Returns:
            DynamicWeights with adjusted values based on cluster state
        """
        now = time.time()

        # Rate-limit weight updates (expensive to compute cluster state)
        if now - self._last_dynamic_weights_update < self._dynamic_weights_update_interval:
            return self._dynamic_weights

        self._last_dynamic_weights_update = now

        # --- Gather cluster state ---
        cluster_state = self._gather_cluster_state()

        # --- Delegate weight computation to priority_calculator ---
        weights = compute_dynamic_weights(cluster_state)

        # Log significant weight changes
        old_weights = self._dynamic_weights
        if (abs(weights.staleness - old_weights.staleness) > 0.05 or
            abs(weights.curriculum - old_weights.curriculum) > 0.03):
            logger.info(
                f"[SelfplayScheduler] Dynamic weights updated: "
                f"staleness={weights.staleness:.2f} (was {old_weights.staleness:.2f}), "
                f"curriculum={weights.curriculum:.2f} (was {old_weights.curriculum:.2f}), "
                f"idle_gpus={cluster_state.idle_gpu_fraction:.1%}, "
                f"queue={cluster_state.training_queue_depth}, "
                f"at_target={cluster_state.configs_at_target_fraction:.1%}, "
                f"avg_elo={cluster_state.average_elo:.0f}"
            )

        self._dynamic_weights = weights

        # Update PriorityCalculator with new weights
        self._priority_calculator.update_weights(weights)

        return weights

    def _gather_cluster_state(self) -> ClusterState:
        """Gather current cluster state for dynamic weight computation.

        December 29, 2025: Extracted from _compute_dynamic_weights() for clarity.

        Returns:
            ClusterState with current metrics
        """
        # 1. Idle GPU fraction (from node capabilities)
        idle_gpu_fraction = 0.0
        if self._node_capabilities:
            total_nodes = len(self._node_capabilities)
            idle_nodes = sum(
                1 for cap in self._node_capabilities.values()
                if cap.current_jobs == 0 and cap.gpu_memory_gb > 0
            )
            idle_gpu_fraction = idle_nodes / max(1, total_nodes)

        # 2. Training queue depth (check backpressure monitor)
        training_queue_depth = 0
        if self._backpressure_monitor:
            try:
                # Synchronous check for queue depth - use hasattr to avoid MagicMock issues
                if hasattr(self._backpressure_monitor, '_last_queue_depth'):
                    cached_depth = self._backpressure_monitor._last_queue_depth
                    # Ensure numeric value (handles MagicMock in tests)
                    if isinstance(cached_depth, (int, float)):
                        training_queue_depth = int(cached_depth)
            except (AttributeError, TypeError, ValueError):
                pass  # Handle missing attributes or type conversion issues

        # 3. Configs at Elo target fraction
        elo_target = 2000.0  # From thresholds
        configs_at_target = 0
        total_configs = len(self._config_priorities)
        for cfg, priority in self._config_priorities.items():
            # Check if config has reached target Elo
            if hasattr(priority, 'current_elo') and priority.current_elo >= elo_target:
                configs_at_target += 1
        configs_at_target_fraction = configs_at_target / max(1, total_configs)

        # 4. Average model Elo
        elo_sum = 0.0
        elo_count = 0
        for priority in self._config_priorities.values():
            if hasattr(priority, 'current_elo') and priority.current_elo > 0:
                elo_sum += priority.current_elo
                elo_count += 1
        average_elo = elo_sum / max(1, elo_count) if elo_count > 0 else 1500.0

        return ClusterState(
            idle_gpu_fraction=idle_gpu_fraction,
            training_queue_depth=training_queue_depth,
            configs_at_target_fraction=configs_at_target_fraction,
            average_elo=average_elo,
        )

    def _config_priority_to_inputs(self, priority: ConfigPriority) -> PriorityInputs:
        """Convert ConfigPriority to PriorityInputs for PriorityCalculator.

        December 29, 2025: Helper for delegating priority calculation.

        Args:
            priority: ConfigPriority from scheduler state

        Returns:
            PriorityInputs for use with PriorityCalculator
        """
        return PriorityInputs(
            config_key=priority.config_key,
            staleness_hours=priority.staleness_hours,
            elo_velocity=priority.elo_velocity,
            training_pending=priority.training_pending,
            exploration_boost=priority.exploration_boost,
            curriculum_weight=priority.curriculum_weight,
            improvement_boost=priority.improvement_boost,
            quality_penalty=priority.quality_penalty,
            architecture_boost=priority.architecture_boost,
            momentum_multiplier=priority.momentum_multiplier,
            game_count=priority.game_count,
            is_large_board=priority.is_large_board,
            priority_override=priority.priority_override,
            current_elo=priority.current_elo,
            elo_uncertainty=priority.elo_uncertainty,
            target_elo=priority.target_elo,
        )

    def _compute_priority_score(self, priority: ConfigPriority) -> float:
        """Compute overall priority score for a configuration.

        Higher score = higher priority for selfplay allocation.

        December 2025: Refactored to delegate core computation to PriorityCalculator.
        Keeps scheduler-specific handling for:
        - Dynamic weight updates (triggers cluster state refresh)
        - ULTRA starvation tier (more severe than PriorityCalculator's emergency)
        - Detailed logging for starvation warnings
        - Momentum multiplier change logging
        """
        # Ensure dynamic weights are fresh and update calculator
        self._compute_dynamic_weights()

        # Convert ConfigPriority to PriorityInputs for calculator
        inputs = self._config_priority_to_inputs(priority)

        # Delegate core computation to PriorityCalculator
        # This handles: staleness, velocity, curriculum, quality, VOI, data deficit,
        # exploration boost, momentum, priority override, player count, cascade, starvation
        score = self._priority_calculator.compute_priority_score(inputs)

        # ULTRA starvation tier override (more severe than PriorityCalculator's emergency)
        # PriorityCalculator handles emergency (< 50) and critical (< 200) tiers
        # ULTRA (< 20) is even more urgent and needs special logging
        game_count = priority.game_count
        if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
            # ULTRA tier: divide out the emergency multiplier already applied by calculator,
            # then apply ULTRA multiplier instead
            score = score / DATA_STARVATION_EMERGENCY_MULTIPLIER * DATA_STARVATION_ULTRA_MULTIPLIER
            logger.warning(
                f"[SelfplayScheduler] ULTRA STARVATION: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_ULTRA_THRESHOLD}). "
                f"Applying {DATA_STARVATION_ULTRA_MULTIPLIER}x priority boost. URGENT DATA NEEDED!"
            )
        elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
            # Log for visibility (calculator already applied the multiplier)
            logger.warning(
                f"[SelfplayScheduler] EMERGENCY: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_EMERGENCY_THRESHOLD}). "
                f"Applying {DATA_STARVATION_EMERGENCY_MULTIPLIER}x priority boost."
            )
        elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
            # Log for visibility (calculator already applied the multiplier)
            logger.info(
                f"[SelfplayScheduler] CRITICAL: {priority.config_key} has only "
                f"{game_count} games (<{DATA_STARVATION_CRITICAL_THRESHOLD}). "
                f"Applying {DATA_STARVATION_CRITICAL_MULTIPLIER}x priority boost."
            )

        # Log momentum multiplier changes (>10% change from baseline)
        if abs(priority.momentum_multiplier - 1.0) > 0.1:
            logger.info(
                f"[SelfplayScheduler] Momentum multiplier applied to {priority.config_key}: "
                f"{priority.momentum_multiplier:.2f}x"
            )

        # Debug logging for priority component breakdown
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[SelfplayScheduler] Priority for {priority.config_key}: "
                f"score={score:.4f}, games={game_count}, "
                f"momentum={priority.momentum_multiplier:.2f}x"
            )

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

        except (OSError, ValueError, IndexError) as e:
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
        except (AttributeError, KeyError) as e:
            logger.warning(f"[SelfplayScheduler] Error getting ELO velocities (using defaults): {e}")

        return result

    async def _get_current_elos(self) -> dict[str, float]:
        """Get current Elo ratings per config.

        Dec 29, 2025: Added for adaptive Gumbel budget scaling.
        Uses 3-layer fallback: QueuePopulator -> EloService -> default 1500.

        Returns:
            Dict mapping config_key to current Elo rating
        """
        result = {}

        # Layer 1: Try QueuePopulator's ConfigTarget (fastest, in-memory)
        try:
            from app.coordination.unified_queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._targets.items():
                    if target.current_best_elo > 0:
                        result[config_key] = target.current_best_elo
        except ImportError:
            pass
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] QueuePopulator unavailable: {e}")

        # Layer 2: Fallback to EloService database for missing configs
        missing_configs = [c for c in ALL_CONFIGS if c not in result]
        if missing_configs:
            try:
                from app.training.elo_service import get_elo_service

                elo_service = get_elo_service()
                for config_key in missing_configs:
                    # Parse config_key: "hex8_2p" -> board_type="hex8", num_players=2
                    parts = config_key.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = int(parts[1].replace("p", ""))
                        # Get leaderboard and take top model's Elo
                        leaderboard = elo_service.get_leaderboard(
                            board_type=board_type,
                            num_players=num_players,
                            limit=1,
                            min_games=5,  # Require some confidence
                        )
                        if leaderboard:
                            result[config_key] = leaderboard[0].rating
                            logger.debug(
                                f"[SelfplayScheduler] Got Elo from EloService for {config_key}: "
                                f"{leaderboard[0].rating:.0f}"
                            )
            except ImportError:
                logger.debug("[SelfplayScheduler] EloService not available")
            except Exception as e:
                logger.debug(f"[SelfplayScheduler] EloService query failed: {e}")

        # Layer 3: Default 1500 for any still-missing configs
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1500.0  # Default starting Elo

        return result

    def _get_adaptive_budget_for_elo(self, elo: float) -> int:
        """Get Gumbel MCTS budget based on current Elo tier.

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.get_adaptive_budget_for_elo() for full docs.
        """
        return _get_budget_for_elo(elo)

    def _get_adaptive_budget_for_games(self, game_count: int, elo: float) -> int:
        """Get Gumbel MCTS budget based on game count (prioritizes bootstrapping).

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.get_adaptive_budget_for_games() for full docs.
        """
        return _get_budget_for_games(game_count, elo)

    def _compute_target_games(self, config: str, current_elo: float) -> int:
        """Compute dynamic target games needed based on Elo gap and board difficulty.

        December 2025: Delegated to budget_calculator module.
        See budget_calculator.compute_target_games() for full docs.
        """
        return _compute_target(config, current_elo)

    def get_target_games_for_config(self, config: str) -> int:
        """Get dynamic target games for a config (public accessor).

        December 29, 2025: Phase 8 - Replaces static TARGET_GAMES_FOR_2000_ELO.
        """
        # Get current Elo from cached data
        current_elo = 1500.0  # Default if not available
        for cfg_key, priority in self._config_priorities.items():
            if cfg_key == config:
                current_elo = getattr(priority, 'current_elo', 1500.0)
                break

        return self._compute_target_games(config, current_elo)

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
        except (AttributeError, KeyError) as e:
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

    def _get_config_data_quality(self, config: str) -> float:
        """Get data quality score for a config from QualityMonitorDaemon.

        Dec 29, 2025 - Phase 1: Quality-weighted selfplay allocation.
        Higher quality score = better training data (Gumbel MCTS, passed parity).
        Lower quality = heuristic-only games, parity failures.

        Args:
            config: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0 (default 0.7 if unavailable)
        """
        try:
            from app.coordination.quality_monitor_daemon import get_quality_daemon

            daemon = get_quality_daemon()
            if daemon:
                quality = daemon.get_config_quality(config)
                if quality is not None:
                    return quality
        except ImportError:
            logger.debug("[SelfplayScheduler] quality_monitor_daemon not available")
        except (AttributeError, KeyError) as e:
            logger.debug(f"[SelfplayScheduler] Error getting quality for {config}: {e}")

        return 0.7  # Default medium quality

    async def _get_all_config_qualities(self) -> dict[str, float]:
        """Get data quality scores for all configs.

        Dec 29, 2025 - Phase 1: Batch quality lookup for priority calculation.

        Returns:
            Dict mapping config_key to quality score (0.0-1.0)
        """
        result = {}
        for config in ALL_CONFIGS:
            result[config] = self._get_config_data_quality(config)
        return result

    def _get_cascade_priority(self, config_key: str) -> float:
        """Get cascade training priority boost for a config.

        Dec 29, 2025: Cascade training for multiplayer bootstrapping.
        Configs that are blocking cascade advancement (2p → 3p → 4p)
        get boosted priority to accelerate training.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Priority multiplier (1.0 = normal, >1.0 = boosted)
        """
        try:
            from app.coordination.cascade_training import get_cascade_orchestrator

            orchestrator = get_cascade_orchestrator()
            if orchestrator:
                return orchestrator.get_bootstrap_priority(config_key)
        except ImportError:
            logger.debug("[SelfplayScheduler] cascade_training not available")
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"[SelfplayScheduler] Error getting cascade priority for {config_key}: {e}")

        return 1.0  # No boost by default

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

    def _get_architecture_boosts(self) -> dict[str, float]:
        """Get architecture-based boosts per config.

        Phase 5B (Dec 2025): Connects selfplay scheduling to architecture performance.
        Configs where the best architecture is performing well get boosted priority.
        This creates a feedback loop where successful architectures get more training data.

        Returns:
            Dict mapping config_key to boost value (0.0 to +0.30)
        """
        result: dict[str, float] = {}

        try:
            from app.training.architecture_tracker import get_allocation_weights

            for config_key in ALL_CONFIGS:
                # Parse config_key to get board_type and num_players
                parts = config_key.rsplit("_", 1)
                if len(parts) < 2:
                    continue

                board_type = parts[0]
                num_players_str = parts[1]
                if not num_players_str.endswith("p"):
                    continue

                try:
                    num_players = int(num_players_str.rstrip("p"))
                except ValueError:
                    continue

                # Get allocation weights for this config
                weights = get_allocation_weights(board_type, num_players)

                if weights:
                    # Boost = max weight * 0.3 (so a dominant architecture with weight 1.0 gives +0.30)
                    max_weight = max(weights.values())
                    boost = max_weight * 0.30

                    if boost > 0.01:  # Only log significant boosts
                        result[config_key] = boost
                        logger.debug(
                            f"[SelfplayScheduler] Architecture boost for {config_key}: +{boost:.2f} "
                            f"(best arch weight: {max_weight:.2f})"
                        )

        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture boosts: {e}")

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

        # Dec 29, 2025: Track configs skipped due to zero/negative priority for logging
        skipped_configs = []

        for config_key, priority_score in priorities:
            if priority_score <= 0:
                skipped_configs.append((config_key, priority_score))
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

        # Dec 29, 2025 - Phase 5: Enforce starvation floor allocations
        # Configs with <100 games MUST get allocation even if priority was low
        allocation = self._enforce_starvation_floor(allocation, games_per_config)

        # Dec 29, 2025 - Phase 4: Enforce 4p allocation minimums
        # If 4p configs are under-allocated, steal from 2p configs
        allocation = self._enforce_4p_allocation_minimums(allocation, games_per_config)

        total_allocated = sum(
            sum(node_games.values()) for node_games in allocation.values()
        )
        self._record_allocation(total_allocated)

        logger.info(
            f"[SelfplayScheduler] Allocated {len(allocation)} configs: "
            f"{', '.join(f'{k}={sum(v.values())}' for k, v in allocation.items())}"
        )

        # Dec 29, 2025: Log skipped configs for transparency
        if skipped_configs:
            logger.warning(
                f"[SelfplayScheduler] {len(skipped_configs)} configs skipped (priority <= 0): "
                f"{', '.join(f'{k}({s:.3f})' for k, s in skipped_configs)}"
            )

        # Dec 2025: Emit SELFPLAY_ALLOCATION_UPDATED for downstream consumers
        # (IdleResourceDaemon, feedback loops, etc.)
        if total_allocated > 0:
            self._emit_allocation_updated(allocation, total_allocated, trigger="allocate_batch")

        return allocation

    def _enforce_starvation_floor(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocation for data-starved configs.

        Dec 29, 2025 - Phase 5: Data starvation floor enforcement.
        Any config with <100 games MUST receive allocation, even if it wasn't
        in the initial priority list. This addresses critical gaps like:
        - square19_4p: 1 game
        - hexagonal_4p: 400 games

        For emergency configs (<100 games): allocate 2x base games
        For critical configs (<1000 games): allocate 1.5x base games

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with starvation floors enforced
        """
        added_configs = []

        for config_key in ALL_CONFIGS:
            priority = self._config_priorities.get(config_key)
            if not priority:
                continue

            game_count = priority.game_count

            # Skip if already adequately allocated
            current_allocation = sum(
                allocation.get(config_key, {}).values()
            )

            # Determine minimum floor based on starvation level
            # Dec 29, 2025: Added ULTRA tier for critically starved configs
            if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
                # ULTRA: <20 games - must get 3x allocation (highest priority)
                min_floor = int(games_per_config * 3.0)
                level = "ULTRA"
            elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
                # EMERGENCY: <100 games - must get 2x allocation
                min_floor = int(games_per_config * 2.0)
                level = "EMERGENCY"
            elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
                # CRITICAL: <1000 games - must get 1.5x allocation
                min_floor = int(games_per_config * 1.5)
                level = "CRITICAL"
            else:
                # Not starved, no floor needed
                continue

            if current_allocation >= min_floor:
                continue  # Already meets floor

            # Need to allocate more for this starved config
            shortfall = min_floor - current_allocation

            # Allocate to available nodes
            additional_allocation = self._allocate_to_nodes(config_key, shortfall)

            if additional_allocation:
                if config_key not in allocation:
                    allocation[config_key] = {}

                # Merge with existing allocation
                for node_id, games in additional_allocation.items():
                    allocation[config_key][node_id] = (
                        allocation[config_key].get(node_id, 0) + games
                    )

                added_configs.append(
                    f"{config_key}({level}:{game_count}g→+{shortfall})"
                )
                logger.warning(
                    f"[SelfplayScheduler] Starvation floor: {config_key} "
                    f"({level}, {game_count} games) allocated +{shortfall} games"
                )

        if added_configs:
            logger.info(
                f"[SelfplayScheduler] Starvation floor enforcement: {', '.join(added_configs)}"
            )

        return allocation

    def _enforce_4p_allocation_minimums(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocations for 4-player configs.

        Dec 29, 2025 - Phase 4: 4-player allocation enforcement.
        The 4p multiplier (4x) can't be satisfied if cluster has limited GPUs.
        This method redistributes from 2p configs to ensure 4p gets minimum.

        Target: 4p configs get at least 1.5x their proportional share.
        If short, steal from 2p configs (which are easiest to generate).

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with enforced 4p minimums
        """
        if not allocation:
            return allocation

        # Calculate per-config totals
        totals: dict[str, int] = {}
        for config_key, node_alloc in allocation.items():
            totals[config_key] = sum(node_alloc.values())

        # Identify 4p and 2p configs
        four_p_configs = [c for c in totals if "_4p" in c]
        two_p_configs = [c for c in totals if "_2p" in c]

        if not four_p_configs or not two_p_configs:
            return allocation

        # Target: 4p should get at least 1.5x base allocation
        min_4p_games = int(games_per_config * 1.5)
        redistributed = 0

        for config in four_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_4p_games - current)

            if shortfall > 0:
                # Try to steal from 2p configs
                for donor in two_p_configs:
                    donor_current = totals.get(donor, 0)
                    donor_min = int(games_per_config * 0.5)  # 2p keeps at least 50%

                    available = max(0, donor_current - donor_min)
                    steal = min(shortfall, available)

                    if steal > 0 and donor in allocation:
                        # Remove from donor (proportionally from nodes)
                        for node_id in allocation[donor]:
                            node_games = allocation[donor][node_id]
                            node_steal = int(steal * node_games / donor_current)
                            allocation[donor][node_id] = max(0, node_games - node_steal)

                        # Add to 4p config (to first available node)
                        if config in allocation and allocation[config]:
                            first_node = next(iter(allocation[config]))
                            allocation[config][first_node] += steal
                        elif config not in allocation:
                            # Find a node that had this config's board type
                            allocation[config] = {list(allocation[donor].keys())[0]: steal}

                        totals[donor] -= steal
                        totals[config] = totals.get(config, 0) + steal
                        shortfall -= steal
                        redistributed += steal

                        logger.debug(
                            f"[SelfplayScheduler] 4p enforcement: {donor} -> {config}: {steal} games"
                        )

                    if shortfall <= 0:
                        break

        if redistributed > 0:
            logger.info(
                f"[SelfplayScheduler] 4p allocation enforcement: redistributed {redistributed} games"
            )

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

        # Dec 29, 2025: Log node allocation breakdown for this config
        if allocation and logger.isEnabledFor(logging.DEBUG):
            total_assigned = sum(allocation.values())
            logger.debug(
                f"[SelfplayScheduler] Node allocation for {config_key}: "
                f"{total_assigned}/{total_games} games across {len(allocation)} nodes "
                f"({', '.join(f'{n}={g}' for n, g in allocation.items())})"
            )
        elif not allocation:
            logger.debug(
                f"[SelfplayScheduler] No allocation for {config_key}: "
                f"no nodes available (unhealthy={len(unhealthy_nodes)}, "
                f"total_requested={total_games})"
            )

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
                # Dec 29 2025: Track job count for dynamic weight calculation
                cap.current_jobs = getattr(node_status, 'selfplay_jobs', 0) or 0

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
            # Dec 28, 2025: NODE_ACTIVATED from cluster activator/watchdog also means node is available
            if hasattr(DataEventType, 'NODE_ACTIVATED'):
                _safe_subscribe(DataEventType.NODE_ACTIVATED, self._on_node_recovered, "NODE_ACTIVATED")
            if hasattr(DataEventType, 'P2P_NODE_DEAD'):
                _safe_subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_unhealthy, "P2P_NODE_DEAD")
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy, "P2P_CLUSTER_UNHEALTHY")
            if hasattr(DataEventType, 'P2P_CLUSTER_HEALTHY'):
                _safe_subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy, "P2P_CLUSTER_HEALTHY")
            if hasattr(DataEventType, 'HOST_OFFLINE'):
                _safe_subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline, "HOST_OFFLINE")
            # Dec 2025: NODE_TERMINATED from idle shutdown - reuse host_offline handler
            if hasattr(DataEventType, 'NODE_TERMINATED'):
                _safe_subscribe(DataEventType.NODE_TERMINATED, self._on_host_offline, "NODE_TERMINATED")

            # Dec 2025: Subscribe to regression events for curriculum rebalancing
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                _safe_subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected, "REGRESSION_DETECTED")

            # Dec 29, 2025: Subscribe to backpressure events for reactive scheduling
            if hasattr(DataEventType, 'BACKPRESSURE_ACTIVATED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_ACTIVATED, self._on_backpressure_activated, "BACKPRESSURE_ACTIVATED")
            if hasattr(DataEventType, 'BACKPRESSURE_RELEASED'):
                _safe_subscribe(DataEventType.BACKPRESSURE_RELEASED, self._on_backpressure_released, "BACKPRESSURE_RELEASED")
            # Dec 29, 2025: Subscribe to NODE_OVERLOADED for per-node backpressure
            if hasattr(DataEventType, 'NODE_OVERLOADED'):
                _safe_subscribe(DataEventType.NODE_OVERLOADED, self._on_node_overloaded, "NODE_OVERLOADED")

            # Dec 29, 2025 - Phase 2: Subscribe to ELO_UPDATED for velocity tracking
            if hasattr(DataEventType, 'ELO_UPDATED'):
                _safe_subscribe(DataEventType.ELO_UPDATED, self._on_elo_updated, "ELO_UPDATED")

            # Dec 29, 2025: Subscribe to progress stall events for 48h autonomous operation
            if hasattr(DataEventType, 'PROGRESS_STALL_DETECTED'):
                _safe_subscribe(DataEventType.PROGRESS_STALL_DETECTED, self._on_progress_stall, "PROGRESS_STALL_DETECTED")
            if hasattr(DataEventType, 'PROGRESS_RECOVERED'):
                _safe_subscribe(DataEventType.PROGRESS_RECOVERED, self._on_progress_recovered, "PROGRESS_RECOVERED")

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

            # Collect penalized configs first to avoid creating unawaited coroutines on error
            penalized_configs: list[tuple[str, float, float]] = []
            for config_key, priority in self._config_priorities.items():
                if priority.quality_penalty < 0:  # Only emit for penalized configs
                    penalized_configs.append(
                        (config_key, -priority.quality_penalty, priority.exploration_boost)
                    )

            # Emit all penalties in a single batched coroutine
            async def emit_all_penalties():
                for key, penalty, weight in penalized_configs:
                    try:
                        await emit_quality_penalty_applied(
                            config_key=key,
                            penalty=penalty,
                            reason="low_quality_selfplay_data",
                            current_weight=weight,
                            source="selfplay_scheduler",
                            quality_score=quality_score,
                            throttle_factor=throttle_factor,
                        )
                    except Exception as e:
                        logger.debug(f"[SelfplayScheduler] Failed to emit penalty for {key}: {e}")

            if penalized_configs:
                coro = emit_all_penalties()
                try:
                    fire_and_forget(coro)
                except Exception as e:
                    # Close the coroutine to avoid "never awaited" warning
                    coro.close()
                    logger.debug(f"[SelfplayScheduler] Failed to schedule penalty emission: {e}")
                    return

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
    # Backpressure Event Handlers (Dec 29, 2025)
    # =========================================================================

    def _on_backpressure_activated(self, event: Any) -> None:
        """Handle BACKPRESSURE_ACTIVATED - reduce selfplay rate.

        Dec 29, 2025: When work queue hits backpressure limits, reduce selfplay
        rate to prevent generating games that can't be processed.

        Args:
            event: Event with payload containing pending_count, trigger, limits
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            pending_count = payload.get("pending_count", 0)
            trigger = payload.get("trigger", "unknown")
            soft_limit = payload.get("soft_limit", 500)

            # Track backpressure state
            if not hasattr(self, "_backpressure_active"):
                self._backpressure_active = False
            self._backpressure_active = True

            # Calculate reduction factor based on how far over soft limit we are
            overage = pending_count - soft_limit
            reduction_factor = max(0.5, 1.0 - (overage / soft_limit) * 0.5)

            # Apply reduction to exploration boost (reduces all allocations)
            if hasattr(self, "_exploration_boost"):
                old_boost = self._exploration_boost
                self._exploration_boost = max(0.5, self._exploration_boost * reduction_factor)
                logger.warning(
                    f"[SelfplayScheduler] Backpressure activated ({trigger}): "
                    f"queue={pending_count}, exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                )
            else:
                logger.warning(
                    f"[SelfplayScheduler] Backpressure activated ({trigger}): queue={pending_count}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling backpressure activated: {e}")

    def _on_backpressure_released(self, event: Any) -> None:
        """Handle BACKPRESSURE_RELEASED - restore normal selfplay rate.

        Dec 29, 2025: When work queue drains below recovery threshold,
        restore normal selfplay allocation rates.

        Args:
            event: Event with payload containing pending_count, recovery_threshold
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            pending_count = payload.get("pending_count", 0)

            # Track backpressure state
            if not hasattr(self, "_backpressure_active"):
                self._backpressure_active = False

            if self._backpressure_active:
                self._backpressure_active = False

                # Restore exploration boost gradually
                if hasattr(self, "_exploration_boost") and self._exploration_boost < 1.0:
                    old_boost = self._exploration_boost
                    self._exploration_boost = min(1.0, self._exploration_boost * 1.5)
                    logger.info(
                        f"[SelfplayScheduler] Backpressure released: "
                        f"queue={pending_count}, exploration_boost {old_boost:.2f} -> {self._exploration_boost:.2f}"
                    )
                else:
                    logger.info(
                        f"[SelfplayScheduler] Backpressure released: queue={pending_count}"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling backpressure released: {e}")

    def _on_node_overloaded(self, event: Any) -> None:
        """Handle NODE_OVERLOADED - add backoff period for overloaded node.

        Dec 29, 2025: When a node reports high CPU/GPU/memory utilization,
        temporarily reduce job dispatch to that node.

        Args:
            event: Event with payload containing host, cpu_percent, gpu_percent,
                   memory_percent, resource_type
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            host = payload.get("host", "")
            cpu_pct = payload.get("cpu_percent", 0)
            gpu_pct = payload.get("gpu_percent", 0)
            memory_pct = payload.get("memory_percent", 0)
            resource_type = payload.get("resource_type", "unknown")

            if not host:
                return

            # Initialize overloaded nodes tracking if needed
            if not hasattr(self, "_overloaded_nodes"):
                self._overloaded_nodes: dict[str, float] = {}

            # Add node to overloaded set with backoff timestamp (60 seconds default)
            backoff_duration = 60.0
            if resource_type == "consecutive_failures":
                backoff_duration = 120.0  # Longer backoff for failures
            elif resource_type == "memory":
                backoff_duration = 90.0  # Memory issues take longer to resolve

            import time
            self._overloaded_nodes[host] = time.time() + backoff_duration

            logger.warning(
                f"[SelfplayScheduler] Node overloaded ({resource_type}): {host} - "
                f"CPU={cpu_pct:.0f}%, GPU={gpu_pct:.0f}%, MEM={memory_pct:.0f}%, "
                f"backoff={backoff_duration:.0f}s"
            )

            # Clean up expired backoffs
            current_time = time.time()
            expired = [n for n, t in self._overloaded_nodes.items() if t < current_time]
            for n in expired:
                del self._overloaded_nodes[n]

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling node overloaded: {e}")

    def _on_elo_updated(self, event: Any) -> None:
        """Handle ELO_UPDATED - track Elo history and compute velocity.

        Dec 29, 2025 - Phase 2: Elo velocity integration.
        Tracks Elo changes over time to compute velocity (Elo/hour).
        Stagnant configs (velocity < 0.5) get reduced allocation.
        Fast-improving configs (velocity > 5.0) get boosted allocation.

        Args:
            event: Event with payload containing config_key, new_elo, old_elo
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            new_elo = payload.get("new_elo", 0.0)

            if not config_key or new_elo <= 0:
                return

            now = time.time()

            # Initialize history if needed
            if config_key not in self._elo_history:
                self._elo_history[config_key] = []

            # Add new data point
            self._elo_history[config_key].append((now, new_elo))

            # Keep only last 24 hours of history
            cutoff = now - 86400  # 24 hours
            self._elo_history[config_key] = [
                (t, e) for t, e in self._elo_history[config_key] if t >= cutoff
            ]

            # Compute velocity: Elo change per hour over last 24 hours
            recent = self._elo_history[config_key]
            if len(recent) >= 2:
                hours = (recent[-1][0] - recent[0][0]) / 3600
                if hours > 0.5:  # Need at least 30 min of data
                    velocity = (recent[-1][1] - recent[0][1]) / hours
                    old_velocity = self._elo_velocity.get(config_key, 0.0)
                    self._elo_velocity[config_key] = velocity

                    # Log significant velocity changes
                    if abs(velocity - old_velocity) > 1.0:
                        logger.info(
                            f"[SelfplayScheduler] Elo velocity for {config_key}: "
                            f"{velocity:.2f} Elo/hour (was {old_velocity:.2f})"
                        )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling Elo update: {e}")

    def _on_progress_stall(self, event: Any) -> None:
        """Handle PROGRESS_STALL_DETECTED - boost selfplay for stalled config.

        Dec 29, 2025: Subscribe to progress stall events for 48h autonomous operation.
        When a config's Elo progress stalls, boost selfplay priority to generate
        more training data and help break through the plateau.

        Args:
            event: Event with payload containing config_key, boost_multiplier, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            boost_multiplier = payload.get("boost_multiplier", 2.0)
            stall_duration = payload.get("stall_duration_hours", 0.0)

            if not config_key or config_key not in self._config_priorities:
                return

            priority = self._config_priorities[config_key]

            # Apply exploration boost to help break through plateau
            priority.exploration_boost = max(priority.exploration_boost, boost_multiplier)

            # Increase staleness to prioritize this config
            priority.staleness_hours = max(priority.staleness_hours, stall_duration * 2.0)

            logger.warning(
                f"[SelfplayScheduler] Progress stall detected for {config_key} "
                f"(stalled {stall_duration:.1f}h). Boosting exploration by {boost_multiplier}x"
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling progress stall: {e}")

    def _on_progress_recovered(self, event: Any) -> None:
        """Handle PROGRESS_RECOVERED - reset boost for recovered config.

        Dec 29, 2025: When a config recovers from a stall, reset the exploration
        boost to normal levels to allow other configs to get resources.

        Args:
            event: Event with payload containing config_key, velocity, etc.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", "")
            new_velocity = payload.get("current_velocity", 0.0)

            if not config_key or config_key not in self._config_priorities:
                return

            priority = self._config_priorities[config_key]

            # Reset exploration boost to normal
            priority.exploration_boost = 1.0

            logger.info(
                f"[SelfplayScheduler] Progress recovered for {config_key} "
                f"(velocity: {new_velocity:.2f} Elo/hour). Reset exploration boost."
            )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling progress recovery: {e}")

    def get_elo_velocity(self, config_key: str) -> float:
        """Get computed Elo velocity for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Elo change per hour (can be negative for regression)
        """
        return self._elo_velocity.get(config_key, 0.0)

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

    def is_node_under_backoff(self, node_id: str) -> bool:
        """Check if a node is under backoff due to overload.

        Dec 29, 2025: Used by job dispatch to avoid overloaded nodes.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node is in backoff period, False otherwise
        """
        if not hasattr(self, "_overloaded_nodes"):
            return False

        import time
        current_time = time.time()
        backoff_until = self._overloaded_nodes.get(node_id, 0)
        return backoff_until > current_time

    def get_overloaded_nodes(self) -> list[str]:
        """Get list of nodes currently under backoff.

        Dec 29, 2025: Returns nodes that should be avoided for job dispatch.

        Returns:
            List of node IDs currently in backoff period
        """
        if not hasattr(self, "_overloaded_nodes"):
            return []

        import time
        current_time = time.time()

        # Clean up expired backoffs and return active ones
        active = []
        expired = []
        for node_id, backoff_until in self._overloaded_nodes.items():
            if backoff_until > current_time:
                active.append(node_id)
            else:
                expired.append(node_id)

        for node_id in expired:
            del self._overloaded_nodes[node_id]

        return active

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "subscribed": self._subscribed,
            "last_priority_update": self._last_priority_update,
            "node_count": len(self._node_capabilities),
            "overloaded_nodes": self.get_overloaded_nodes(),
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

    # =========================================================================
    # Architecture Performance Tracking (December 29, 2025)
    # =========================================================================

    def get_architecture_weights(
        self,
        board_type: str,
        num_players: int,
        temperature: float = 0.5,
    ) -> dict[str, float]:
        """Get allocation weights for architectures based on Elo performance.

        Higher-performing architectures get more weight for selfplay/training allocation.
        Uses softmax with temperature to control concentration on best architecture.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            temperature: Softmax temperature (lower = more concentrated on best)

        Returns:
            Dictionary mapping architecture name to allocation weight (sums to 1.0)

        Example:
            >>> weights = scheduler.get_architecture_weights("hex8", 2)
            >>> # {"v5": 0.4, "v4": 0.3, "v3": 0.2, "v2": 0.1}
        """
        try:
            from app.training.architecture_tracker import get_allocation_weights

            return get_allocation_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=temperature,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return {}
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture weights: {e}")
            return {}

    def record_architecture_evaluation(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        elo: float,
        training_hours: float = 0.0,
        games_evaluated: int = 0,
    ) -> None:
        """Record an evaluation result for an architecture.

        Called after gauntlet evaluation to track architecture performance.
        The architecture tracker uses this data to compute allocation weights.

        Args:
            architecture: Architecture version (e.g., "v4", "v5_heavy")
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            elo: Elo rating from evaluation
            training_hours: Additional training time for this evaluation
            games_evaluated: Games used in evaluation
        """
        try:
            from app.training.architecture_tracker import record_evaluation

            stats = record_evaluation(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                elo=elo,
                training_hours=training_hours,
                games_evaluated=games_evaluated,
            )
            logger.info(
                f"[SelfplayScheduler] Architecture evaluation recorded: "
                f"{architecture} on {board_type}_{num_players}p -> Elo {elo:.0f} "
                f"(avg: {stats.avg_elo:.0f}, best: {stats.best_elo:.0f})"
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to record architecture evaluation: {e}")

    def get_architecture_boost(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        threshold_elo_diff: float = 50.0,
    ) -> float:
        """Get boost factor for an architecture based on relative performance.

        Returns a factor > 1.0 if this architecture is better than average,
        < 1.0 if worse, exactly 1.0 if at average or no data available.

        Args:
            architecture: Architecture to check (e.g., "v4", "v5")
            board_type: Board type
            num_players: Player count
            threshold_elo_diff: Minimum Elo difference for boost

        Returns:
            Boost factor (1.0 = no boost)
        """
        try:
            from app.training.architecture_tracker import get_architecture_tracker

            tracker = get_architecture_tracker()
            return tracker.get_architecture_boost(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                threshold_elo_diff=threshold_elo_diff,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return 1.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture boost: {e}")
            return 1.0

    def get_best_architecture(
        self,
        board_type: str,
        num_players: int,
        metric: str = "avg_elo",
    ) -> str | None:
        """Get the best-performing architecture for a configuration.

        Args:
            board_type: Board type
            num_players: Player count
            metric: Metric to rank by ("avg_elo", "best_elo", "efficiency_score")

        Returns:
            Architecture name (e.g., "v5") or None if no data available
        """
        try:
            from app.training.architecture_tracker import get_best_architecture

            stats = get_best_architecture(
                board_type=board_type,
                num_players=num_players,
                metric=metric,
            )
            if stats:
                return stats.architecture
            return None
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return None
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting best architecture: {e}")
            return None


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
