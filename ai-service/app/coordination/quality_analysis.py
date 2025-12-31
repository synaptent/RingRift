"""Quality analysis functions for training data quality assessment.

.. deprecated:: December 2025
    This module is being consolidated into app.quality. Use the unified
    quality framework instead:

    For game quality scoring:
        from app.quality import GameQualityScorer, get_game_quality_scorer
        scorer = get_game_quality_scorer()
        result = scorer.score(game_data)

    For database validation:
        from app.quality.validators import DatabaseValidator
        validator = DatabaseValidator()
        result = validator.validate("path/to/database.db")

    For quality thresholds:
        from app.quality import get_quality_thresholds
        thresholds = get_quality_thresholds()

    This module will continue to work until Q2 2026 for backward compatibility.

Extracted from feedback_loop_controller.py (December 2025).
Provides pure functions for quality scoring, intensity mapping, and curriculum
weight adjustment based on selfplay data quality.

Usage:
    from app.coordination.quality_analysis import (
        assess_selfplay_quality,
        compute_intensity_from_quality,
        compute_training_urgency,
        compute_curriculum_weight_adjustment,
        get_quality_threshold,
        QualityResult,
        IntensityLevel,
        UrgencyLevel,
    )

    # Assess quality of selfplay data
    result = assess_selfplay_quality("data/games/canonical_hex8_2p.db", games_count=500)

    # Map quality to training intensity
    intensity = compute_intensity_from_quality(result.quality_score)

    # Get training urgency from intensity
    urgency = compute_training_urgency(result.quality_score, intensity)
"""
import warnings

warnings.warn(
    "app.coordination.quality_analysis is deprecated and will be consolidated into "
    "app.quality by Q2 2026. Use app.quality.GameQualityScorer for game scoring, "
    "app.quality.validators.DatabaseValidator for validation, and "
    "app.quality.get_quality_thresholds() for thresholds.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from app.coordination.event_utils import parse_config_key

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Constants (imported from centralized thresholds)
# ============================================================================

# Import quality thresholds from canonical source
# Note: quality_analysis uses stricter thresholds than general training:
# - HIGH (0.90) = excellent quality, hot_path training
# - MEDIUM (0.70) = good quality, accelerated training
# - LOW (0.50) = acceptable quality, normal training
# - MINIMUM (0.30) = minimum for training, reduced intensity
from app.config.thresholds import (
    QUALITY_EXCELLENT_THRESHOLD as HIGH_QUALITY_THRESHOLD,  # 0.90
    HIGH_QUALITY_THRESHOLD as MEDIUM_QUALITY_THRESHOLD,  # 0.70
    MEDIUM_QUALITY_THRESHOLD as LOW_QUALITY_THRESHOLD,  # 0.50
    LOW_QUALITY_THRESHOLD as MINIMUM_QUALITY_THRESHOLD,  # 0.30
)

# Quality assessment parameters
DEFAULT_SAMPLE_LIMIT = 50  # Games to sample for quality assessment
FULL_QUALITY_GAME_COUNT = 500  # Games needed for full quality score


class IntensityLevel(str, Enum):
    """Training intensity levels based on data quality.

    Maps continuous quality scores to discrete intensity levels for
    training rate control.
    """

    PAUSED = "paused"  # Quality < 0.50 - pause training
    REDUCED = "reduced"  # Quality 0.50-0.65 - slow training
    NORMAL = "normal"  # Quality 0.65-0.80 - standard training
    ACCELERATED = "accelerated"  # Quality 0.80-0.90 - fast training
    HOT_PATH = "hot_path"  # Quality >= 0.90 - maximum training speed


class UrgencyLevel(str, Enum):
    """Training urgency levels for accelerator signaling.

    Maps intensity levels to urgency signals for the training accelerator.
    """

    NONE = "none"  # Paused - no training
    LOW = "low"  # Reduced - low priority
    NORMAL = "normal"  # Normal - standard priority
    HIGH = "high"  # Accelerated - high priority
    CRITICAL = "critical"  # Hot path - maximum priority


# Intensity to urgency mapping
INTENSITY_TO_URGENCY: dict[IntensityLevel, UrgencyLevel] = {
    IntensityLevel.HOT_PATH: UrgencyLevel.CRITICAL,
    IntensityLevel.ACCELERATED: UrgencyLevel.HIGH,
    IntensityLevel.NORMAL: UrgencyLevel.NORMAL,
    IntensityLevel.REDUCED: UrgencyLevel.LOW,
    IntensityLevel.PAUSED: UrgencyLevel.NONE,
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class QualityResult:
    """Result of quality assessment.

    Attributes:
        quality_score: Overall quality score (0.0-1.0)
        games_assessed: Number of games used in assessment
        avg_game_quality: Average quality per game (0.0-1.0)
        count_factor: Scaling factor based on total game count
        method: Method used for assessment ("unified", "count_heuristic")
    """

    quality_score: float
    games_assessed: int = 0
    avg_game_quality: float = 0.0
    count_factor: float = 1.0
    method: str = "unified"

    def __post_init__(self) -> None:
        """Validate score bounds."""
        if not 0.0 <= self.quality_score <= 1.0:
            object.__setattr__(
                self, "quality_score", max(0.0, min(1.0, self.quality_score))
            )


@dataclass
class CurriculumWeightChange:
    """Result of curriculum weight adjustment calculation.

    Attributes:
        old_weight: Previous curriculum weight
        new_weight: Adjusted curriculum weight
        changed: Whether the weight was modified
        reason: Reason for adjustment (or "no_change")
    """

    old_weight: float
    new_weight: float
    changed: bool = field(init=False)
    reason: str = "no_change"

    def __post_init__(self) -> None:
        """Calculate whether weight changed."""
        self.changed = abs(self.new_weight - self.old_weight) > 1e-6


@dataclass
class QualityThresholds:
    """Quality thresholds for a specific configuration.

    Provides board/player-specific quality thresholds that can be
    tuned per configuration.

    Attributes:
        min_quality: Minimum quality to proceed with training
        target_quality: Target quality for full training speed
        high_quality: High quality threshold for acceleration
        config_key: Configuration key these thresholds apply to
    """

    min_quality: float = LOW_QUALITY_THRESHOLD
    target_quality: float = MEDIUM_QUALITY_THRESHOLD
    high_quality: float = HIGH_QUALITY_THRESHOLD
    config_key: str = ""


# ============================================================================
# Pure Quality Assessment Functions
# ============================================================================


def assess_selfplay_quality(
    db_path: str | Path,
    games_count: int,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    quality_scorer_fn: Callable[[dict], float] | None = None,
) -> QualityResult:
    """Assess quality of selfplay data using game content analysis.

    Uses the UnifiedQualityScorer to evaluate actual game content quality,
    not just game count. Falls back to count-based heuristics if the scorer
    is unavailable.

    Args:
        db_path: Path to the game database
        games_count: Total number of games in the database
        sample_limit: Maximum games to sample for quality assessment
        quality_scorer_fn: Optional custom quality scorer function

    Returns:
        QualityResult with quality score and assessment metadata
    """
    db = Path(db_path) if isinstance(db_path, str) else db_path

    # If database doesn't exist, return low quality
    if not db.exists():
        logger.debug(f"Database not found: {db_path}")
        return QualityResult(
            quality_score=MINIMUM_QUALITY_THRESHOLD,
            games_assessed=0,
            avg_game_quality=0.0,
            count_factor=0.0,
            method="no_database",
        )

    try:
        # Try to use unified quality scorer
        from app.quality.unified_quality import compute_game_quality_from_params

        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT game_id, game_status, winner, termination_reason, total_moves
                FROM games
                WHERE game_status IN ('complete', 'finished', 'COMPLETE', 'FINISHED')
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (sample_limit,),
            )
            games = cursor.fetchall()

        if not games:
            logger.debug(f"No completed games in {db_path}")
            return QualityResult(
                quality_score=MINIMUM_QUALITY_THRESHOLD,
                games_assessed=0,
                avg_game_quality=0.0,
                count_factor=0.0,
                method="no_games",
            )

        # Compute quality scores for sampled games
        quality_scores: list[float] = []

        for game in games:
            try:
                if quality_scorer_fn is not None:
                    # Use custom scorer
                    quality = quality_scorer_fn(dict(game))
                else:
                    # Use unified scorer
                    result = compute_game_quality_from_params(
                        game_id=game["game_id"],
                        game_status=game["game_status"],
                        winner=game["winner"],
                        termination_reason=game["termination_reason"],
                        total_moves=game["total_moves"] or 0,
                    )
                    quality = result.quality_score
                quality_scores.append(quality)
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                logger.debug(
                    f"Failed to compute quality for game {game.get('game_id', 'unknown')}: {e}"
                )
                continue

        if not quality_scores:
            return QualityResult(
                quality_score=MINIMUM_QUALITY_THRESHOLD,
                games_assessed=len(games),
                avg_game_quality=0.0,
                count_factor=0.0,
                method="scoring_failed",
            )

        # Compute average quality and scale by game count
        avg_quality = sum(quality_scores) / len(quality_scores)
        count_factor = min(1.0, games_count / FULL_QUALITY_GAME_COUNT)

        # Final quality is scaled by count factor
        # Base of 0.3 ensures even small datasets have some quality signal
        final_quality = MINIMUM_QUALITY_THRESHOLD + (
            avg_quality - MINIMUM_QUALITY_THRESHOLD
        ) * count_factor

        logger.debug(
            f"Quality: avg={avg_quality:.3f}, count_factor={count_factor:.2f}, "
            f"final={final_quality:.3f}"
        )

        return QualityResult(
            quality_score=final_quality,
            games_assessed=len(quality_scores),
            avg_game_quality=avg_quality,
            count_factor=count_factor,
            method="unified",
        )

    except ImportError:
        logger.debug("UnifiedQualityScorer not available, using count heuristic")
    except (sqlite3.Error, AttributeError, TypeError, ValueError, RuntimeError) as e:
        logger.debug(f"Quality assessment error: {e}")

    # Fallback to count-based heuristic
    return _count_based_quality(games_count)


def _count_based_quality(games_count: int) -> QualityResult:
    """Fallback quality estimation based on game count only.

    Simple heuristic when content-based scoring is unavailable.

    Args:
        games_count: Number of games in database

    Returns:
        QualityResult with estimated quality score
    """
    if games_count < 100:
        quality = 0.30
    elif games_count < 500:
        quality = 0.60
    elif games_count < 1000:
        quality = 0.80
    else:
        quality = 0.95

    return QualityResult(
        quality_score=quality,
        games_assessed=0,
        avg_game_quality=0.0,
        count_factor=min(1.0, games_count / FULL_QUALITY_GAME_COUNT),
        method="count_heuristic",
    )


def compute_intensity_from_quality(quality_score: float) -> IntensityLevel:
    """Map continuous quality score to discrete intensity level.

    Uses a 5-tier gradient to map quality scores to training intensity:
    - hot_path (>= 0.90): Maximum training speed
    - accelerated (0.80-0.90): Fast training
    - normal (0.65-0.80): Standard training rate
    - reduced (0.50-0.65): Slow training
    - paused (< 0.50): Training paused

    Args:
        quality_score: Quality score (0.0-1.0)

    Returns:
        IntensityLevel enum value
    """
    if quality_score >= HIGH_QUALITY_THRESHOLD:
        return IntensityLevel.HOT_PATH
    elif quality_score >= 0.80:
        return IntensityLevel.ACCELERATED
    elif quality_score >= 0.65:
        return IntensityLevel.NORMAL
    elif quality_score >= LOW_QUALITY_THRESHOLD:
        return IntensityLevel.REDUCED
    else:
        return IntensityLevel.PAUSED


def compute_training_urgency(
    quality_score: float, intensity: IntensityLevel | None = None
) -> UrgencyLevel:
    """Map quality/intensity to training urgency for accelerator signaling.

    Args:
        quality_score: Quality score (0.0-1.0)
        intensity: Optional pre-computed intensity level

    Returns:
        UrgencyLevel for training accelerator
    """
    if intensity is None:
        intensity = compute_intensity_from_quality(quality_score)

    return INTENSITY_TO_URGENCY.get(intensity, UrgencyLevel.NORMAL)


def compute_curriculum_weight_adjustment(
    quality_score: float,
    current_weight: float,
    min_weight: float = 0.5,
    max_weight: float = 2.0,
) -> CurriculumWeightChange:
    """Compute curriculum weight adjustment based on quality.

    Logic:
    - Low quality (< 0.5): Increase weight by 15% (needs attention)
    - Medium quality (0.5-0.7): No change
    - High quality (>= 0.7): Decrease weight by 5% (stable, less urgent)

    This creates a feedback loop where struggling configs get more training
    attention, while stable configs can have slightly reduced priority.

    Args:
        quality_score: Quality score (0.0-1.0)
        current_weight: Current curriculum weight
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight

    Returns:
        CurriculumWeightChange with old and new weights
    """
    new_weight = current_weight

    if quality_score < LOW_QUALITY_THRESHOLD:
        # Low quality - needs more training focus
        new_weight = min(max_weight, current_weight * 1.15)
        reason = "low_quality_increase"
    elif quality_score >= MEDIUM_QUALITY_THRESHOLD:
        # High quality - can reduce priority slightly
        new_weight = max(min_weight, current_weight * 0.95)
        reason = "high_quality_decrease"
    else:
        reason = "no_change"

    return CurriculumWeightChange(
        old_weight=current_weight,
        new_weight=new_weight,
        reason=reason,
    )


def get_quality_threshold(
    config_key: str,
    base_threshold: float = MEDIUM_QUALITY_THRESHOLD,
) -> QualityThresholds:
    """Get quality thresholds for a specific configuration.

    Can be extended to provide per-config thresholds based on
    board type, player count, or other factors.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        base_threshold: Default threshold to use

    Returns:
        QualityThresholds for the configuration
    """
    # Parse config to potentially adjust thresholds
    parsed = parse_config_key(config_key)

    # 4-player games may need different thresholds due to game complexity
    if parsed and parsed.num_players >= 3:
        # Slightly lower thresholds for multiplayer (harder to achieve quality)
        return QualityThresholds(
            min_quality=LOW_QUALITY_THRESHOLD - 0.05,
            target_quality=base_threshold - 0.05,
            high_quality=HIGH_QUALITY_THRESHOLD - 0.05,
            config_key=config_key,
        )

    return QualityThresholds(
        min_quality=LOW_QUALITY_THRESHOLD,
        target_quality=base_threshold,
        high_quality=HIGH_QUALITY_THRESHOLD,
        config_key=config_key,
    )


def is_quality_acceptable(
    quality_score: float,
    threshold: float | None = None,
    thresholds: QualityThresholds | None = None,
) -> bool:
    """Check if quality score meets minimum threshold for training.

    Args:
        quality_score: Quality score (0.0-1.0)
        threshold: Single threshold to check against
        thresholds: QualityThresholds object for multi-tier checking

    Returns:
        True if quality is acceptable for training
    """
    if threshold is not None:
        return quality_score >= threshold

    if thresholds is not None:
        return quality_score >= thresholds.min_quality

    return quality_score >= MEDIUM_QUALITY_THRESHOLD


def should_accelerate_training(quality_score: float) -> bool:
    """Check if quality is high enough to accelerate training.

    Args:
        quality_score: Quality score (0.0-1.0)

    Returns:
        True if training should be accelerated
    """
    return quality_score >= 0.80


def should_pause_training(quality_score: float) -> bool:
    """Check if quality is too low to continue training.

    Args:
        quality_score: Quality score (0.0-1.0)

    Returns:
        True if training should be paused
    """
    return quality_score < LOW_QUALITY_THRESHOLD


# ============================================================================
# Quality Trend Analysis
# ============================================================================


@dataclass
class QualityTrend:
    """Analysis of quality trend over time.

    Attributes:
        trend: Direction of change ("improving", "stable", "declining")
        current_score: Most recent quality score
        previous_score: Previous quality score
        change: Absolute change in quality
        change_pct: Percentage change in quality
    """

    trend: str
    current_score: float
    previous_score: float
    change: float = field(init=False)
    change_pct: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate change metrics."""
        self.change = self.current_score - self.previous_score
        if self.previous_score > 0:
            self.change_pct = (self.change / self.previous_score) * 100
        else:
            self.change_pct = 0.0


def analyze_quality_trend(
    current_score: float,
    previous_score: float,
    significant_change_threshold: float = 0.05,
) -> QualityTrend:
    """Analyze quality trend between two scores.

    Args:
        current_score: Current quality score
        previous_score: Previous quality score
        significant_change_threshold: Minimum change to be considered significant

    Returns:
        QualityTrend with direction and metrics
    """
    change = current_score - previous_score

    if change > significant_change_threshold:
        trend = "improving"
    elif change < -significant_change_threshold:
        trend = "declining"
    else:
        trend = "stable"

    return QualityTrend(
        trend=trend,
        current_score=current_score,
        previous_score=previous_score,
    )


def compute_exploration_adjustment(
    quality_score: float,
    trend: str,
) -> tuple[float, float]:
    """Compute exploration temperature adjustments based on quality and trend.

    Returns exploration_temp_boost and noise_boost values for adjusting
    selfplay exploration based on current quality state.

    Logic:
    - Declining quality with low score: Boost exploration significantly
    - Low quality: Modest exploration boost
    - High quality with stable/improving: Allow slight exploitation bias
    - Default: No adjustment

    Args:
        quality_score: Current quality score (0.0-1.0)
        trend: Quality trend ("improving", "stable", "declining")

    Returns:
        Tuple of (exploration_temp_boost, noise_boost)
    """
    if trend == "declining" and quality_score < 0.6:
        # Declining quality - boost exploration to find better data
        return (1.5, 0.10)
    elif quality_score < LOW_QUALITY_THRESHOLD:
        # Low quality - modest exploration boost
        return (1.2, 0.05)
    elif quality_score < MEDIUM_QUALITY_THRESHOLD:
        # Medium quality - no change
        return (1.0, 0.0)
    elif quality_score > HIGH_QUALITY_THRESHOLD and trend in ("improving", "stable"):
        # High quality with positive trend - can exploit slightly
        return (0.9, -0.02)
    else:
        # Default - no adjustment
        return (1.0, 0.0)


# ============================================================================
# Backward Compatibility
# ============================================================================

# Aliases for easier migration
assess_quality = assess_selfplay_quality
get_intensity = compute_intensity_from_quality
get_urgency = compute_training_urgency
