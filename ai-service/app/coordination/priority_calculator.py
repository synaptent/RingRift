"""Priority calculation engine for selfplay scheduling.

December 2025: Extracted from selfplay_scheduler.py to improve modularity.

This module provides pure functions for computing selfplay priority scores,
isolating the complex priority math from the scheduling orchestration logic.

Usage:
    from app.coordination.priority_calculator import (
        PriorityCalculator,
        compute_staleness_factor,
        compute_velocity_factor,
        compute_data_deficit_factor,
        compute_voi_score,
    )

    calculator = PriorityCalculator(
        config_priorities=config_priorities,
        dynamic_weights=dynamic_weights,
    )
    score = calculator.compute_priority_score(priority)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from app.config.coordination_defaults import SelfplayPriorityWeightDefaults
from app.config.thresholds import get_elo_gap, is_target_met
from app.coordination.event_utils import parse_config_key

logger = logging.getLogger(__name__)

# Get default weights from centralized config
_priority_weight_defaults = SelfplayPriorityWeightDefaults()

# Weight bounds for dynamic adjustment
DYNAMIC_WEIGHT_BOUNDS = _priority_weight_defaults.get_weight_bounds()

# Threshold constants (sourced from coordination_defaults)
FRESH_DATA_THRESHOLD = _priority_weight_defaults.FRESH_DATA_THRESHOLD
STALE_DATA_THRESHOLD = _priority_weight_defaults.STALE_DATA_THRESHOLD
MAX_STALENESS_HOURS = _priority_weight_defaults.MAX_STALENESS_HOURS
TARGET_GAMES_FOR_2000_ELO = _priority_weight_defaults.TARGET_GAMES_FOR_2000_ELO
LARGE_BOARD_TARGET_MULTIPLIER = _priority_weight_defaults.LARGE_BOARD_TARGET_MULTIPLIER

# =============================================================================
# Canonical Configuration Constants (December 2025)
# =============================================================================
# All supported board/player combinations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Priority override multipliers
PRIORITY_OVERRIDE_MULTIPLIERS = {
    -1: 4.0,  # EMERGENCY: 4x boost (bootstrap crisis)
    0: 3.0,   # CRITICAL: 3x boost
    1: 2.0,   # HIGH: 2x boost
    2: 1.25,  # MEDIUM: 25% boost
    3: 1.0,   # LOW: no boost (normal)
}

# Static fallback priorities (used when live metrics unavailable)
# Dynamic computation in compute_config_priority_override() takes precedence
# Feb 2026: Updated to match current Elo standings:
#   CRITICAL(0): hexagonal_2p (1483), square19_2p (1519)
#   HIGH(1): square8_2p (1685), hex8_2p (1703)
#   MEDIUM(2): hex8_3p (1808), square8_3p (1879), hexagonal_3p (1825), square19_3p (1825)
#   LOW(3): square19_4p (1982), hexagonal_4p (1982), hex8_4p (2016), square8_4p (2054)
CONFIG_PRIORITY_FALLBACK: dict[str, int] = {
    "hexagonal_2p": 0,   # 1483 Elo - CRITICAL, furthest from target
    "square19_2p": 0,    # 1519 Elo - CRITICAL
    "square8_2p": 1,     # 1685 Elo - HIGH
    "hex8_2p": 1,        # 1703 Elo - HIGH
    "hex8_3p": 2,        # 1808 Elo - MEDIUM
    "square8_3p": 2,     # 1879 Elo - MEDIUM
    "hexagonal_3p": 2,   # 1825 Elo - MEDIUM
    "square19_3p": 2,    # 1825 Elo - MEDIUM
    "square19_4p": 3,    # 1982 Elo - LOW (near target)
    "hexagonal_4p": 3,   # 1982 Elo - LOW (near target)
    "hex8_4p": 3,        # 2016 Elo - LOW (target met)
    "square8_4p": 3,     # 2054 Elo - LOW (target met)
}


def compute_config_priority_override(config_key: str, game_count: int | None, elo: float | None) -> int:
    """Compute priority tier from actual metrics instead of hardcoded table.

    Returns priority override level:
        -1: EMERGENCY (bootstrap crisis, <100 games)
         0: CRITICAL (< 500 games)
         1: HIGH (< 2000 games, Elo < 1800)
         2: MEDIUM (default)
         3: LOW (target met, Elo >= 2000)

    Falls back to CONFIG_PRIORITY_FALLBACK when game_count is None (live data unavailable).
    """
    if game_count is None:
        return CONFIG_PRIORITY_FALLBACK.get(config_key, 2)
    if elo is not None and elo >= 2000:
        return 3  # LOW - target met, free resources for others
    if game_count < 100:
        return -1  # EMERGENCY - bootstrap crisis
    if game_count < 500:
        return 0  # CRITICAL
    # Feb 26, 2026: Elo-based priority independent of game count.
    # Previously hexagonal_2p (1483 Elo, 3181 games) got MEDIUM because
    # game_count > 2000. A config with very low Elo needs more selfplay
    # with the current model, regardless of how many old games exist.
    if elo is not None and elo < 1600:
        return 0  # CRITICAL - severely underperforming
    if elo is not None and elo < 1800:
        return 1  # HIGH - needs improvement
    return 2  # MEDIUM

# Player count allocation multipliers
# Feb 24, 2026: Rebalanced — 4p configs at/above 2000 Elo target, 2p weakest.
# Previous 8x/4x ratios were starving 2p configs (hexagonal_2p: 1483, square19_2p: 1519).
# 2p needs 27-33% MORE games than 4p to reach same sample target (fewer samples/game)
# but was getting 1/8th the allocation. Narrowing the gap.
PLAYER_COUNT_ALLOCATION_MULTIPLIER = {
    2: 1.0,  # Baseline
    3: 1.5,  # Reduced from 4.0 — 3p no longer severely starved
    4: 2.0,  # Reduced from 8.0 — 4p at/near 2000 Elo, doesn't need aggressive boost
}

# Jan 14, 2026: CRITICAL priority should bypass player multiplier
# to ensure truly critical configs get max allocation regardless of player count.
# Without this, CRITICAL 2p configs (like square8_2p) get 3x fewer jobs than
# CRITICAL 3p configs due to 1.0 vs 3.0 player multiplier.
CRITICAL_BYPASSES_PLAYER_MULTIPLIER = True

# =============================================================================
# Game Estimation Constants (December 2025)
# =============================================================================
# Samples-per-game estimates by board type and player count
# Used for game count normalization - ensures selfplay generates enough games
# to meet training sample targets. Based on historical data from export scripts.
SAMPLES_PER_GAME_BY_BOARD = {
    "hex8": {"2p": 35, "3p": 40, "4p": 45},
    "square8": {"2p": 40, "3p": 50, "4p": 55},
    "square19": {"2p": 150, "3p": 180, "4p": 200},
    "hexagonal": {"2p": 250, "3p": 300, "4p": 350},
}

# VOI (Value of Information) sample cost by board type
# Relative cost per game (compute time * complexity)
# Small boards are baseline (1.0), large boards cost more
VOI_SAMPLE_COST_BY_BOARD = {
    "hex8": {"2p": 1.0, "3p": 1.3, "4p": 1.5},
    "square8": {"2p": 1.0, "3p": 1.3, "4p": 1.5},
    "square19": {"2p": 5.0, "3p": 7.0, "4p": 9.0},
    "hexagonal": {"2p": 8.0, "3p": 11.0, "4p": 14.0},
}


# =============================================================================
# Pure Calculation Functions
# =============================================================================


def compute_staleness_factor(
    staleness_hours: float,
    fresh_threshold: float = FRESH_DATA_THRESHOLD,
    stale_threshold: float = STALE_DATA_THRESHOLD,
    max_staleness: float = MAX_STALENESS_HOURS,
) -> float:
    """Compute staleness factor (0-1, higher = more stale).

    Args:
        staleness_hours: Hours since last data update
        fresh_threshold: Hours below which data is considered fresh
        stale_threshold: Hours above which data is considered stale
        max_staleness: Maximum staleness hours (caps at 1.0)

    Returns:
        Staleness factor between 0.0 and 1.0
    """
    if staleness_hours < fresh_threshold:
        return 0.0
    if staleness_hours > max_staleness:
        return 1.0
    # Linear interpolation between fresh and stale thresholds
    return min(1.0, (staleness_hours - fresh_threshold) /
               (stale_threshold - fresh_threshold))


def compute_velocity_factor(elo_velocity: float, max_velocity: float = 100.0) -> float:
    """Compute velocity factor (0-1, higher = faster improvement).

    Sprint 17.4: Changed from linear to exponential scaling.
    Exponential scaling gives more weight to fast improvers while still
    providing meaningful differentiation at lower velocities.

    Formula: 1 - exp(-velocity / scale)
    - At velocity=0: factor=0
    - At velocity=scale: factor=0.632 (1 - 1/e)
    - At velocity=2*scale: factor=0.865
    - Asymptotes to 1.0 for very high velocities

    Expected Elo improvement: +8-12 from better resource allocation to
    fast-improving configs.

    Args:
        elo_velocity: ELO points per day (positive = improvement)
        max_velocity: Scale parameter for exponential decay (default 100.0)

    Returns:
        Velocity factor between 0.0 and 1.0
    """
    if elo_velocity <= 0:
        return 0.0
    # Exponential saturation curve: faster configs get more weight
    # but with diminishing returns for extremely fast improvement
    scale = max_velocity / 3.0  # Scale so velocity=33 gives ~0.63 factor
    return 1.0 - math.exp(-elo_velocity / scale)


def compute_data_deficit_factor(
    game_count: int,
    is_large_board: bool,
    target_games: int = TARGET_GAMES_FOR_2000_ELO,
    large_board_multiplier: float = LARGE_BOARD_TARGET_MULTIPLIER,
) -> float:
    """Compute data deficit factor (0-1, higher = more data needed).

    Args:
        game_count: Current number of games for this config
        is_large_board: True for square19, hexagonal boards
        target_games: Base target game count
        large_board_multiplier: Multiplier for large boards

    Returns:
        Data deficit factor between 0.0 and 1.0
    """
    target = target_games
    if is_large_board:
        target = int(target * large_board_multiplier)

    if game_count >= target:
        return 0.0  # No deficit

    deficit_ratio = 1.0 - (game_count / target)
    return min(1.0, deficit_ratio)


def compute_cluster_deficit_boost(
    local_game_count: int,
    cluster_game_count: int,
    low_cluster_threshold: int = 1000,
    medium_cluster_threshold: int = 5000,
    high_cluster_threshold: int = 10000,
) -> float:
    """Compute priority boost based on cluster-wide game availability.

    Jan 2026: Part of Phase 2 of Cluster Manifest Training Integration.

    Configs with low cluster-wide game counts get boosted priority to fill
    the cluster data gap. This prevents duplicate selfplay generation when
    sufficient data already exists across the cluster.

    Args:
        local_game_count: Games available locally on this node
        cluster_game_count: Total games available across the entire cluster
        low_cluster_threshold: Below this, apply 2x boost (default: 1000)
        medium_cluster_threshold: Below this, apply 1.5x boost (default: 5000)
        high_cluster_threshold: At or above this, no boost (default: 10000)

    Returns:
        Boost multiplier (1.0 = no boost, 2.0 = high priority for under-represented configs)
    """
    # Use cluster count for the determination, not local
    total_games = cluster_game_count

    if total_games < low_cluster_threshold:
        return 2.0  # High priority for severely under-represented configs
    elif total_games < medium_cluster_threshold:
        return 1.5  # Medium priority for under-represented configs
    elif total_games < high_cluster_threshold:
        return 1.2  # Slight boost for moderately represented configs

    return 1.0  # No boost for well-represented configs


def compute_voi_score(
    elo_uncertainty: float,
    elo_gap: float,
    info_gain_per_game: float,
    max_uncertainty: float = 300.0,
    max_gap: float = 500.0,
    max_info_gain: float = 10.0,
) -> float:
    """Compute Value of Information score for a config.

    VOI prioritizes configs where more games yield the highest expected Elo gain.
    Combines uncertainty, Elo gap, and information gain efficiency.

    Args:
        elo_uncertainty: Confidence interval width in Elo points
        elo_gap: Distance from target Elo
        info_gain_per_game: Expected uncertainty reduction per game
        max_uncertainty: Max uncertainty for normalization
        max_gap: Max Elo gap for normalization
        max_info_gain: Max info gain for normalization

    Returns:
        VOI score between 0.0 and 1.0
    """
    # Normalize components to 0-1 range
    uncertainty_factor = min(1.0, elo_uncertainty / max_uncertainty)
    gap_factor = min(1.0, elo_gap / max_gap)
    info_factor = min(1.0, info_gain_per_game / max_info_gain)

    # Combined VOI: weighted sum
    return (
        uncertainty_factor * 0.4 +  # 40% weight on uncertainty
        gap_factor * 0.3 +          # 30% weight on improvement potential
        info_factor * 0.3           # 30% weight on learning efficiency
    )


def compute_info_gain_per_game(
    elo_uncertainty: float,
    game_count: int,
) -> float:
    """Compute expected information gain per new game.

    Uses 1/sqrt(n) rule from statistical sampling theory.
    Each new game reduces Elo CI width by approximately this amount.

    Args:
        elo_uncertainty: Current Elo uncertainty (CI width)
        game_count: Current number of games

    Returns:
        Expected Elo uncertainty reduction per game
    """
    if game_count <= 0:
        return elo_uncertainty  # First game has max info gain
    return elo_uncertainty / math.sqrt(game_count)


def clamp_weight(name: str, value: float, bounds: dict[str, tuple[float, float]] | None = None) -> float:
    """Clamp a weight value within its defined bounds.

    Args:
        name: Weight name for bounds lookup
        value: Weight value to clamp
        bounds: Optional custom bounds dict (defaults to DYNAMIC_WEIGHT_BOUNDS)

    Returns:
        Clamped weight value
    """
    if bounds is None:
        bounds = DYNAMIC_WEIGHT_BOUNDS
    min_w, max_w = bounds.get(name, (0.05, 0.50))
    return max(min_w, min(max_w, value))


def extract_player_count(config_key: str) -> int:
    """Extract player count from config_key (e.g., 'hex8_2p' -> 2).

    Args:
        config_key: Configuration key like 'hex8_2p'

    Returns:
        Player count (2, 3, or 4). Defaults to 2 on parse error.
    """
    parsed = parse_config_key(config_key)
    return parsed.num_players if parsed else 2


def compute_games_needed(
    game_count: int,
    samples_per_game: float,
    target_samples: int,
) -> int:
    """Calculate games needed to reach training sample target.

    Args:
        game_count: Current number of games
        samples_per_game: Historical average samples per game
        target_samples: Target training sample count

    Returns:
        Number of additional games needed (0 if target met)
    """
    if samples_per_game <= 0:
        return 0
    current_samples = int(game_count * samples_per_game)
    remaining_samples = max(0, target_samples - current_samples)
    return int(remaining_samples / samples_per_game)


# =============================================================================
# Dynamic Weight Computation
# =============================================================================


@dataclass
class ClusterState:
    """Current cluster state for dynamic weight calculation."""
    idle_gpu_fraction: float = 0.0
    training_queue_depth: int = 0
    configs_at_target_fraction: float = 0.0
    average_elo: float = 1500.0


@dataclass
class DynamicWeights:
    """Dynamically computed priority weights based on cluster state.

    Weights are adjusted based on cluster conditions to optimize resource allocation.
    All weights are bounded to prevent any single factor from dominating.

    January 2026 Sprint 10: Added diversity weight for opponent variety maximization.
    January 2026 Phase 2: Added cluster weight for cluster-wide data optimization.
    """
    staleness: float = 0.30
    velocity: float = 0.10
    training: float = 0.15
    exploration: float = 0.10
    curriculum: float = 0.10
    improvement: float = 0.08
    data_deficit: float = 0.10
    quality: float = 0.05
    voi: float = 0.02
    # January 2026 Sprint 10: Diversity weight for opponent variety
    diversity: float = 0.10
    # January 2026 Phase 2: Cluster weight for cluster-wide data optimization
    cluster: float = 0.15  # Prioritizes configs with low cluster-wide game counts

    # Cluster state that drove these weights (for debugging)
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
            "diversity": self.diversity,
            "cluster": self.cluster,  # January 2026 Phase 2
            "idle_gpu_fraction": self.idle_gpu_fraction,
            "training_queue_depth": self.training_queue_depth,
            "configs_at_target_fraction": self.configs_at_target_fraction,
            "average_elo": self.average_elo,
        }


def compute_dynamic_weights(
    cluster_state: ClusterState,
    base_weights: DynamicWeights | None = None,
    idle_gpu_high_threshold: float = 0.5,
    idle_gpu_low_threshold: float = 0.1,
    training_queue_high_threshold: int = 10,
    configs_at_target_threshold: float = 0.7,
    elo_high_threshold: float = 1800.0,
    elo_medium_threshold: float = 1500.0,
) -> DynamicWeights:
    """Compute dynamic priority weights based on current cluster state.

    Weight adjustment logic:
    - High idle GPU fraction → Boost staleness weight (generate more data)
    - Large training queue → Reduce staleness weight (don't flood queue)
    - Many configs at Elo target → Reduce velocity weight (focus on struggling configs)
    - High average Elo → Boost curriculum weight (need harder positions)

    Args:
        cluster_state: Current cluster state
        base_weights: Starting weights (defaults to DynamicWeights defaults)
        idle_gpu_high_threshold: Fraction above which GPUs are considered very idle
        idle_gpu_low_threshold: Fraction below which GPUs are considered busy
        training_queue_high_threshold: Queue depth above which to reduce generation
        configs_at_target_threshold: Fraction above which to reduce velocity focus
        elo_high_threshold: Elo above which to boost curriculum
        elo_medium_threshold: Elo above which to moderately boost curriculum

    Returns:
        DynamicWeights with adjusted values
    """
    if base_weights is None:
        base_weights = DynamicWeights()

    weights = DynamicWeights()

    # Copy cluster state for tracking
    weights.idle_gpu_fraction = cluster_state.idle_gpu_fraction
    weights.training_queue_depth = cluster_state.training_queue_depth
    weights.configs_at_target_fraction = cluster_state.configs_at_target_fraction
    weights.average_elo = cluster_state.average_elo

    # 1. Staleness weight adjustment
    staleness_adj = base_weights.staleness
    if cluster_state.idle_gpu_fraction > idle_gpu_high_threshold:
        staleness_adj = base_weights.staleness * 1.5
    elif cluster_state.idle_gpu_fraction < idle_gpu_low_threshold:
        staleness_adj = base_weights.staleness * 0.7
    if cluster_state.training_queue_depth > training_queue_high_threshold:
        staleness_adj *= 0.6
    weights.staleness = clamp_weight("staleness", staleness_adj)

    # 2. Velocity weight adjustment
    velocity_adj = base_weights.velocity
    if cluster_state.configs_at_target_fraction > configs_at_target_threshold:
        velocity_adj = base_weights.velocity * 0.6
    weights.velocity = clamp_weight("velocity", velocity_adj)

    # 3. Curriculum weight adjustment
    curriculum_adj = base_weights.curriculum
    if cluster_state.average_elo > elo_high_threshold:
        curriculum_adj = base_weights.curriculum * 1.8
    elif cluster_state.average_elo > elo_medium_threshold:
        curriculum_adj = base_weights.curriculum * 1.3
    weights.curriculum = clamp_weight("curriculum", curriculum_adj)

    # 4. Data deficit weight adjustment
    data_deficit_adj = base_weights.data_deficit
    if cluster_state.idle_gpu_fraction > idle_gpu_high_threshold:
        data_deficit_adj = base_weights.data_deficit * 1.4
    weights.data_deficit = clamp_weight("data_deficit", data_deficit_adj)

    # Keep other weights clamped at base values
    weights.exploration = clamp_weight("exploration", base_weights.exploration)
    weights.training = clamp_weight("training", base_weights.training)
    weights.improvement = clamp_weight("improvement", base_weights.improvement)
    weights.quality = clamp_weight("quality", base_weights.quality)
    weights.voi = clamp_weight("voi", base_weights.voi)

    # January 2026 Sprint 10: Diversity weight
    # Boost diversity weight when idle GPUs are available (more capacity for diverse games)
    diversity_adj = base_weights.diversity
    if cluster_state.idle_gpu_fraction > idle_gpu_high_threshold:
        diversity_adj = base_weights.diversity * 1.3  # More capacity → boost diversity
    weights.diversity = clamp_weight("diversity", diversity_adj)

    return weights


# =============================================================================
# Priority Calculator Class
# =============================================================================


@dataclass
class PriorityInputs:
    """Input data for priority score calculation.

    Simplified representation of ConfigPriority for pure calculation.
    """
    config_key: str
    staleness_hours: float = 0.0
    elo_velocity: float = 0.0
    training_pending: bool = False
    exploration_boost: float = 1.0
    curriculum_weight: float = 1.0
    improvement_boost: float = 0.0
    quality_penalty: float = 0.0
    architecture_boost: float = 0.0  # Phase 5B: From ArchitectureTracker (0.0 to +0.30)
    momentum_multiplier: float = 1.0
    game_count: int = 0
    is_large_board: bool = False
    priority_override: int = 3
    current_elo: float = 1500.0
    elo_uncertainty: float = 200.0
    target_elo: float = 2000.0
    # January 2026 Sprint 10: Diversity score for opponent variety (0.0=low, 1.0=high)
    diversity_score: float = 1.0
    # January 2026 Phase 2: Cluster-wide game count for deficit calculation
    cluster_game_count: int = 0


class PriorityCalculator:
    """Calculator for selfplay priority scores.

    Encapsulates the priority calculation logic, separating it from
    the scheduling orchestration in SelfplayScheduler.
    """

    def __init__(
        self,
        dynamic_weights: DynamicWeights | None = None,
        get_quality_score_fn: Callable[[str], float] | None = None,
        get_elo_velocity_fn: Callable[[str], float] | None = None,
        get_cascade_priority_fn: Callable[[str], float] | None = None,
        data_starvation_ultra_threshold: int = 500,
        data_starvation_emergency_threshold: int = 1500,
        data_starvation_critical_threshold: int = 3000,
        data_starvation_warning_threshold: int = 5000,
        data_starvation_ultra_multiplier: float = 500.0,
        data_starvation_emergency_multiplier: float = 100.0,
        data_starvation_critical_multiplier: float = 30.0,
        data_starvation_warning_multiplier: float = 3.0,
    ):
        """Initialize priority calculator.

        Args:
            dynamic_weights: Pre-computed dynamic weights (or defaults)
            get_quality_score_fn: Callback to get quality score for config
            get_elo_velocity_fn: Callback to get Elo velocity for config
            get_cascade_priority_fn: Callback to get cascade priority boost
            data_starvation_ultra_threshold: Games below which is ultra-critical
            data_starvation_emergency_threshold: Games below which is emergency
            data_starvation_critical_threshold: Games below which is critical
            data_starvation_warning_threshold: Games below which gets warning boost
            data_starvation_ultra_multiplier: Multiplier for ultra-critical
            data_starvation_emergency_multiplier: Multiplier for emergency
            data_starvation_critical_multiplier: Multiplier for critical
            data_starvation_warning_multiplier: Multiplier for warning
        """
        self._weights = dynamic_weights or DynamicWeights()
        self._get_quality_score_fn = get_quality_score_fn
        self._get_elo_velocity_fn = get_elo_velocity_fn
        self._get_cascade_priority_fn = get_cascade_priority_fn
        self._starvation_ultra = data_starvation_ultra_threshold
        self._starvation_emergency = data_starvation_emergency_threshold
        self._starvation_critical = data_starvation_critical_threshold
        self._starvation_warning = data_starvation_warning_threshold
        self._starvation_ultra_mult = data_starvation_ultra_multiplier
        self._starvation_emergency_mult = data_starvation_emergency_multiplier
        self._starvation_critical_mult = data_starvation_critical_multiplier
        self._starvation_warning_mult = data_starvation_warning_multiplier

    def compute_priority_score(self, inputs: PriorityInputs) -> float:
        """Compute overall priority score for a configuration.

        Higher score = higher priority for selfplay allocation.

        Args:
            inputs: Priority input data

        Returns:
            Priority score (higher = higher priority)
        """
        w = self._weights

        # Compute base factors
        staleness = compute_staleness_factor(inputs.staleness_hours) * w.staleness
        velocity = compute_velocity_factor(inputs.elo_velocity) * w.velocity
        training = (1.0 if inputs.training_pending else 0.0) * w.training
        exploration = (inputs.exploration_boost - 1.0) * w.exploration

        # Curriculum factor (normalized around 1.0)
        curriculum = (inputs.curriculum_weight - 1.0) * w.curriculum

        # Improvement boost
        improvement = inputs.improvement_boost * w.improvement

        # Architecture boost (Phase 5B: from ArchitectureTracker)
        # Added directly since architecture_boost is already scaled (0.0 to 0.30)
        architecture = inputs.architecture_boost

        # Quality factor
        quality_score = 0.7  # Default
        if self._get_quality_score_fn:
            try:
                quality_score = self._get_quality_score_fn(inputs.config_key)
            except (KeyError, ValueError, TypeError, AttributeError):
                # Config not found, invalid value, or callback misconfigured - use default
                pass
        quality = (quality_score - 0.7) * w.quality + inputs.quality_penalty

        # Data deficit factor
        data_deficit = compute_data_deficit_factor(
            inputs.game_count, inputs.is_large_board
        ) * w.data_deficit

        # VOI factor
        info_gain = compute_info_gain_per_game(inputs.elo_uncertainty, inputs.game_count)
        elo_gap = max(0.0, inputs.target_elo - inputs.current_elo)
        voi = compute_voi_score(inputs.elo_uncertainty, elo_gap, info_gain) * w.voi

        # January 2026 Sprint 10: Diversity factor
        # Low diversity_score (few opponent types) → high diversity_factor → boost priority
        diversity_factor = max(0.0, 1.0 - inputs.diversity_score)
        diversity = diversity_factor * w.diversity

        # January 2026 Phase 2: Cluster deficit factor
        # Prioritizes configs with low cluster-wide game counts to prevent duplicate generation
        cluster_boost = compute_cluster_deficit_boost(
            inputs.game_count, inputs.cluster_game_count
        )
        cluster = (cluster_boost - 1.0) * w.cluster  # Normalize around 1.0

        # Combine factors
        score = staleness + velocity + training + exploration + curriculum + improvement + architecture + quality + data_deficit + voi + diversity + cluster

        # Apply exploration boost as multiplier
        score *= inputs.exploration_boost

        # Apply momentum multiplier
        score *= inputs.momentum_multiplier

        # Apply Elo velocity multiplier - Sprint 17.4: Exponential scaling
        # Changed from step function to continuous exponential curve
        # for smoother resource allocation transitions
        elo_velocity = inputs.elo_velocity
        if self._get_elo_velocity_fn:
            try:
                elo_velocity = self._get_elo_velocity_fn(inputs.config_key)
            except (KeyError, ValueError, TypeError, AttributeError):
                # Config not found, invalid value, or callback misconfigured - use input default
                pass

        # Exponential velocity multiplier:
        # Feb 2026: Inverted penalty for regressing configs. Previously, stalled
        # and regressing configs both got 0.6x, creating a death spiral where
        # the configs that most needed resources got the least. Now:
        # - velocity < 0 (regressing): multiplier = 1.3 (BOOST - needs better data urgently)
        # - velocity = 0 (stalled): multiplier = 0.85 (mild reduction, not punitive)
        # - velocity = 0.5: multiplier = 0.76 → 0.92
        # - velocity = 2.0: multiplier = 1.0 (neutral)
        # - velocity = 5.0: multiplier = 1.22
        # - velocity = 10.0: multiplier = 1.35
        if elo_velocity < 0:
            velocity_multiplier = 1.3  # Regressing configs get priority boost
        elif elo_velocity == 0:
            velocity_multiplier = 0.85  # Stalled configs get mild reduction
        else:
            # Exponential saturation from 0.6 toward 1.0 as velocity increases
            base_recovery = 0.4 * (1.0 - math.exp(-elo_velocity / 2.0))
            # Linear boost component for very fast improvers
            linear_boost = 0.2 * min(elo_velocity / 10.0, 1.0)
            velocity_multiplier = 0.6 + base_recovery + linear_boost
        score *= velocity_multiplier

        # Feb 2026: Elo gap factor - configs further from target get more selfplay.
        # Replaces the old near_target_boost (1900+ only = 5x) which was backwards:
        # it boosted configs already close to target while starving configs that
        # needed the most help (e.g., hexagonal_2p at 1483 got nothing).
        #
        # Scale: gap_factor = 1.0 + (elo_gap / 500)
        #   - 0 gap (at target):   maintenance_mode = 0.3x
        #   - 100 gap (1900 Elo):  1.2x
        #   - 300 gap (1700 Elo):  1.6x
        #   - 500 gap (1500 Elo):  2.0x
        #   - 517 gap (1483 Elo):  2.03x
        # Capped at 3.0x to prevent runaway priority for extremely weak configs.
        elo_gap = get_elo_gap(inputs.config_key, inputs.current_elo)
        if is_target_met(inputs.config_key, inputs.current_elo):
            # Maintenance mode: config already at or above target.
            # Still want occasional selfplay to prevent staleness, but deprioritize.
            elo_gap_factor = 0.3
        else:
            elo_gap_factor = min(3.0, 1.0 + (elo_gap / 500.0))
        score *= elo_gap_factor

        # Apply priority override (Feb 2026: Dynamic overrides take precedence)
        # CONFIG_PRIORITY_FALLBACK used only when dynamic computation unavailable
        effective_override = CONFIG_PRIORITY_FALLBACK.get(
            inputs.config_key, inputs.priority_override
        )
        override_multiplier = PRIORITY_OVERRIDE_MULTIPLIERS.get(effective_override, 1.0)
        score *= override_multiplier

        # Apply player count multiplier (unless CRITICAL and bypass enabled)
        player_count = extract_player_count(inputs.config_key)
        is_critical = effective_override == 0  # CRITICAL priority level

        if CRITICAL_BYPASSES_PLAYER_MULTIPLIER and is_critical:
            # CRITICAL configs get max multiplier (4.0) regardless of player count
            # This ensures 2p CRITICAL configs get equal allocation as 3p/4p CRITICAL
            player_multiplier = 4.0
        else:
            player_multiplier = PLAYER_COUNT_ALLOCATION_MULTIPLIER.get(player_count, 1.0)
        score *= player_multiplier

        # Apply cascade priority boost
        cascade_boost = 1.0
        if self._get_cascade_priority_fn:
            try:
                cascade_boost = self._get_cascade_priority_fn(inputs.config_key)
            except (KeyError, ValueError, TypeError, AttributeError):
                # Config not found, invalid value, or callback misconfigured - use default
                pass
        score *= cascade_boost

        # Apply data starvation multiplier (most severe tier wins)
        if inputs.game_count < self._starvation_ultra:
            score *= self._starvation_ultra_mult
        elif inputs.game_count < self._starvation_emergency:
            score *= self._starvation_emergency_mult
        elif inputs.game_count < self._starvation_critical:
            score *= self._starvation_critical_mult
        elif inputs.game_count < self._starvation_warning:
            score *= self._starvation_warning_mult

        return score

    def get_starvation_tier(self, game_count: int) -> str:
        """Return the starvation tier name for a given game count.

        Returns:
            Tier name ("ULTRA", "EMERGENCY", "CRITICAL", "WARNING") or ""
        """
        if game_count < self._starvation_ultra:
            return "ULTRA"
        if game_count < self._starvation_emergency:
            return "EMERGENCY"
        if game_count < self._starvation_critical:
            return "CRITICAL"
        if game_count < self._starvation_warning:
            return "WARNING"
        return ""

    def update_weights(self, weights: DynamicWeights) -> None:
        """Update the dynamic weights used for calculation."""
        self._weights = weights
