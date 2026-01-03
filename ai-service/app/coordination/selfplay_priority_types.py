"""Selfplay scheduler priority types.

Extracted from selfplay_scheduler.py (January 2, 2026) to reduce module size
and improve testability.

This module contains the core data structures used for priority calculation:
- DynamicWeights: Cluster-state-adjusted priority weights
- ConfigPriority: Priority information for a configuration

These are pure data classes with no I/O or external dependencies beyond
the threshold constants.
"""

from __future__ import annotations

__all__ = [
    "ConfigPriority",
    "DynamicWeights",
]

import math
from dataclasses import dataclass, field

from app.config.coordination_defaults import SelfplayPriorityWeightDefaults
from app.coordination.budget_calculator import parse_config_key

# Create singleton instance for env var resolution
_priority_weight_defaults = SelfplayPriorityWeightDefaults()

# Priority calculation weights (BASE values - adjusted dynamically)
STALENESS_WEIGHT = _priority_weight_defaults.STALENESS_WEIGHT
ELO_VELOCITY_WEIGHT = _priority_weight_defaults.ELO_VELOCITY_WEIGHT
TRAINING_NEED_WEIGHT = _priority_weight_defaults.TRAINING_NEED_WEIGHT
EXPLORATION_BOOST_WEIGHT = _priority_weight_defaults.EXPLORATION_BOOST_WEIGHT
CURRICULUM_WEIGHT = _priority_weight_defaults.CURRICULUM_WEIGHT
IMPROVEMENT_BOOST_WEIGHT = _priority_weight_defaults.IMPROVEMENT_BOOST_WEIGHT
DATA_DEFICIT_WEIGHT = _priority_weight_defaults.DATA_DEFICIT_WEIGHT
QUALITY_WEIGHT = _priority_weight_defaults.QUALITY_WEIGHT
VOI_WEIGHT = _priority_weight_defaults.VOI_WEIGHT
# January 2026 Sprint 10: Diversity weight for opponent variety maximization
DIVERSITY_WEIGHT = _priority_weight_defaults.DIVERSITY_WEIGHT

# Staleness thresholds (hours)
FRESH_DATA_THRESHOLD = _priority_weight_defaults.FRESH_DATA_THRESHOLD
STALE_DATA_THRESHOLD = _priority_weight_defaults.STALE_DATA_THRESHOLD
MAX_STALENESS_HOURS = _priority_weight_defaults.MAX_STALENESS_HOURS

# Target games for data deficit calculation
TARGET_GAMES_FOR_2000_ELO = _priority_weight_defaults.TARGET_GAMES_FOR_2000_ELO
LARGE_BOARD_TARGET_MULTIPLIER = _priority_weight_defaults.LARGE_BOARD_TARGET_MULTIPLIER

# Default training sample target per config
DEFAULT_TRAINING_SAMPLES_TARGET = 50000


@dataclass
class DynamicWeights:
    """Dynamically computed priority weights based on cluster state.

    Dec 29, 2025: Implements adaptive reweighting to optimize resource allocation.
    Weights are adjusted based on:
    - Idle GPU fraction: More idle GPUs → boost staleness weight (generate more data)
    - Training backlog: Large queue → reduce staleness weight (don't flood queue)
    - Configs at Elo target: Many at target → reduce velocity weight (focus elsewhere)
    - Average model Elo: Higher Elo → boost curriculum weight (harder positions needed)

    January 2026 Sprint 10: Added diversity weight for opponent variety maximization.
    When diversity is low (same opponents repeatedly), boost diversity weight.

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
    quality: float = QUALITY_WEIGHT
    voi: float = VOI_WEIGHT
    # January 2026 Sprint 10: Diversity weight for opponent variety
    diversity: float = DIVERSITY_WEIGHT

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
            "diversity": self.diversity,
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
    exploration_boost_expires_at: float = 0.0  # When boost decays back to 1.0
    curriculum_weight: float = 1.0  # Curriculum-based weight
    improvement_boost: float = 0.0  # From ImprovementOptimizer (-0.10 to +0.15)
    quality_penalty: float = 0.0  # Quality degradation penalty (0.0 to -0.20)
    architecture_boost: float = 0.0  # From ArchitectureTracker (0.0 to +0.30)
    momentum_multiplier: float = 1.0  # From FeedbackAccelerator (0.5 to 1.5)
    game_count: int = 0  # Current game count for this config
    is_large_board: bool = False  # True for square19, hexagonal
    priority_override: int = 3  # From config (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
    search_budget: int = 400  # Gumbel MCTS budget from velocity feedback
    current_elo: float = 1500.0  # Current Elo rating for dynamic weight calculation

    # VOI (Value of Information) based prioritization
    # Uncertainty in Elo estimate (confidence interval width in Elo points)
    elo_uncertainty: float = 200.0  # Default high uncertainty for new configs
    # Target Elo for this config
    target_elo: float = 2000.0

    # Game count normalization fields
    target_training_samples: int = DEFAULT_TRAINING_SAMPLES_TARGET
    samples_per_game_estimate: float = 50.0  # Historical average samples per game

    # January 2026 Sprint 10: Diversity metrics for opponent variety
    # Lower diversity_score means less variety in opponents (needs more diverse games)
    diversity_score: float = 1.0  # 0.0 = low diversity, 1.0 = high diversity
    opponent_types_seen: int = 0  # Number of distinct opponent types played against

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

        Prioritizes configs with low game counts, especially
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
        """Extract player count from config_key (e.g., 'hex8_2p' -> 2)."""
        _, num_players = parse_config_key(self.config_key)
        return num_players

    @property
    def games_needed(self) -> int:
        """Calculate games needed to reach training sample target.

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
        """Gap between current Elo and target Elo."""
        return max(0.0, self.target_elo - self.current_elo)

    @property
    def info_gain_per_game(self) -> float:
        """Estimated information gain (uncertainty reduction) per new game.

        Uses 1/sqrt(n) rule from statistical sampling theory.
        Each new game reduces Elo CI width by approximately this amount.
        """
        if self.game_count <= 0:
            return self.elo_uncertainty  # First game has max info gain
        # Standard error reduces with sqrt(n)
        return self.elo_uncertainty / math.sqrt(self.game_count)

    @property
    def voi_score(self) -> float:
        """Value of Information score for this config.

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

    @property
    def diversity_factor(self) -> float:
        """Compute diversity factor (0-1, higher = needs more diversity).

        January 2026 Sprint 10: Prioritizes configs with low opponent variety.
        Configs that have played against few distinct opponent types get
        boosted priority to maximize training robustness.

        A low diversity_score (few opponent types) returns a high factor,
        encouraging allocation of games with diverse opponents.
        """
        # Invert diversity_score: low diversity → high factor
        # diversity_score of 0.0 → factor of 1.0 (max boost)
        # diversity_score of 1.0 → factor of 0.0 (no boost needed)
        return max(0.0, 1.0 - self.diversity_score)
