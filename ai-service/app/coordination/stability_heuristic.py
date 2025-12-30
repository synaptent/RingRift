"""Stability heuristic for detecting volatile models.

December 2025: Part of targeted Elo improvements (adaptive K-factor,
confidence-weighted evaluation, stability heuristic).

This module provides tools to detect unstable models via rating variance,
flag models with high volatility for investigation, and integrate stability
assessment into promotion decisions.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.training.elo_service import EloService

logger = logging.getLogger(__name__)


class StabilityLevel(Enum):
    """Model stability classification."""
    STABLE = "stable"          # Low variance, consistent performance
    DEVELOPING = "developing"  # Still converging, moderate variance
    VOLATILE = "volatile"      # High variance, needs investigation
    DECLINING = "declining"    # Consistent downward trend
    UNKNOWN = "unknown"        # Insufficient data


@dataclass
class StabilityAssessment:
    """Result of stability assessment for a model."""

    model_id: str
    board_type: str
    num_players: int

    # Overall assessment
    level: StabilityLevel = StabilityLevel.UNKNOWN
    volatility_score: float = 0.0  # 0.0 = perfectly stable, 1.0+ = highly volatile

    # Detailed metrics
    rating_variance: float = 0.0
    rating_std_dev: float = 0.0
    max_swing: float = 0.0  # Largest single-period Elo change
    oscillation_count: int = 0  # Number of direction changes

    # Trend info (from get_elo_trend)
    is_plateau: bool = False
    is_declining: bool = False
    slope: float = 0.0
    trend_confidence: float = 0.0

    # Data availability
    sample_count: int = 0
    duration_hours: float = 0.0

    # Recommendations
    promotion_safe: bool = True
    investigation_needed: bool = False
    recommended_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "level": self.level.value,
            "volatility_score": round(self.volatility_score, 3),
            "rating_variance": round(self.rating_variance, 2),
            "rating_std_dev": round(self.rating_std_dev, 2),
            "max_swing": round(self.max_swing, 1),
            "oscillation_count": self.oscillation_count,
            "is_plateau": self.is_plateau,
            "is_declining": self.is_declining,
            "slope": round(self.slope, 3),
            "trend_confidence": round(self.trend_confidence, 3),
            "sample_count": self.sample_count,
            "duration_hours": round(self.duration_hours, 2),
            "promotion_safe": self.promotion_safe,
            "investigation_needed": self.investigation_needed,
            "recommended_actions": self.recommended_actions,
        }


def assess_model_stability(
    model_id: str,
    board_type: str,
    num_players: int,
    hours: int = 48,
    min_samples: int = 5,
    elo_service: EloService | None = None,
) -> StabilityAssessment:
    """Assess stability of a model based on rating history.

    Args:
        model_id: Model identifier (e.g., "canonical")
        board_type: Board type (e.g., "square8", "hex8")
        num_players: Number of players (2, 3, or 4)
        hours: Time window in hours for analysis (default 48h)
        min_samples: Minimum data points required (default 5)
        elo_service: Optional EloService instance (uses singleton if None)

    Returns:
        StabilityAssessment with detailed stability metrics and recommendations
    """
    assessment = StabilityAssessment(
        model_id=model_id,
        board_type=board_type,
        num_players=num_players,
    )

    # Get EloService
    if elo_service is None:
        try:
            from app.training.elo_service import get_elo_service
            elo_service = get_elo_service()
        except ImportError:
            logger.warning("Could not import elo_service")
            return assessment

    # Get trend data from existing infrastructure
    try:
        from app.training.elo_service import get_elo_trend, get_rating_history

        trend = get_elo_trend(
            elo_service, model_id, board_type, num_players, hours, min_samples
        )

        assessment.is_plateau = trend.get("is_plateau", False)
        assessment.is_declining = trend.get("is_declining", False)
        assessment.slope = trend.get("slope", 0.0)
        assessment.trend_confidence = trend.get("confidence", 0.0)
        assessment.sample_count = trend.get("sample_count", 0)
        assessment.duration_hours = trend.get("duration_hours", 0.0)

        # Get full rating history for variance analysis
        history = get_rating_history(
            elo_service, model_id, board_type, num_players, limit=100
        )

        if len(history) < min_samples:
            assessment.level = StabilityLevel.UNKNOWN
            assessment.recommended_actions.append(
                f"Need {min_samples - len(history)} more games for stability assessment"
            )
            return assessment

        # Extract ratings (oldest to newest for analysis)
        ratings = [h["rating"] for h in reversed(history)]

        # Calculate variance metrics
        if len(ratings) >= 2:
            assessment.rating_variance = statistics.variance(ratings)
            assessment.rating_std_dev = math.sqrt(assessment.rating_variance)

            # Calculate max swing (largest single-period change)
            deltas = [abs(ratings[i] - ratings[i - 1]) for i in range(1, len(ratings))]
            assessment.max_swing = max(deltas) if deltas else 0.0

            # Count oscillations (direction changes)
            if len(ratings) >= 3:
                directions = [ratings[i] - ratings[i - 1] for i in range(1, len(ratings))]
                oscillations = sum(
                    1 for i in range(1, len(directions))
                    if (directions[i] > 0) != (directions[i - 1] > 0) and directions[i - 1] != 0
                )
                assessment.oscillation_count = oscillations

        # Calculate volatility score
        assessment.volatility_score = _calculate_volatility_score(assessment)

        # Determine stability level
        assessment.level = _classify_stability(assessment)

        # Determine promotion safety and recommendations
        _add_recommendations(assessment)

    except Exception as e:
        logger.error(f"Error assessing stability for {model_id}: {e}")
        assessment.level = StabilityLevel.UNKNOWN
        assessment.recommended_actions.append(f"Assessment failed: {e}")

    return assessment


def _calculate_volatility_score(assessment: StabilityAssessment) -> float:
    """Calculate a normalized volatility score (0.0 = stable, 1.0+ = volatile).

    Components:
    - Rating standard deviation (normalized to expected range)
    - Max swing relative to std dev (catches outlier moves)
    - Oscillation frequency (direction changes per sample)
    - Declining trend penalty
    """
    score = 0.0

    # Base volatility from standard deviation
    # Expected std dev for stable model: ~20-40 Elo
    # Volatile model: 80+ Elo
    std_dev_component = assessment.rating_std_dev / 50.0  # Normalize: 50 Elo = 1.0
    score += std_dev_component * 0.4  # 40% weight

    # Max swing component (catches sudden jumps)
    # Normal swing: ~30 Elo, concerning: 100+ Elo
    if assessment.rating_std_dev > 0:
        swing_ratio = assessment.max_swing / max(assessment.rating_std_dev, 20.0)
        swing_component = min(swing_ratio / 3.0, 1.0)  # Cap at 1.0 when swing is 3x std dev
        score += swing_component * 0.3  # 30% weight

    # Oscillation frequency (rapid direction changes = unstable)
    if assessment.sample_count > 2:
        max_possible_oscillations = assessment.sample_count - 2
        oscillation_ratio = assessment.oscillation_count / max(max_possible_oscillations, 1)
        score += oscillation_ratio * 0.2  # 20% weight

    # Declining trend penalty
    if assessment.is_declining:
        # Declining faster = more volatile
        decline_penalty = min(abs(assessment.slope) / 5.0, 0.5)  # Max 0.5 for 5+ Elo/hour decline
        score += decline_penalty * 0.1  # 10% weight

    return score


def _classify_stability(assessment: StabilityAssessment) -> StabilityLevel:
    """Classify stability level based on metrics."""

    # Declining is a special case
    if assessment.is_declining and assessment.slope < -2.0:
        return StabilityLevel.DECLINING

    # Use volatility score as primary classifier
    if assessment.volatility_score < 0.3:
        return StabilityLevel.STABLE
    elif assessment.volatility_score < 0.6:
        return StabilityLevel.DEVELOPING
    else:
        return StabilityLevel.VOLATILE


def _add_recommendations(assessment: StabilityAssessment) -> None:
    """Add recommendations based on stability assessment."""

    if assessment.level == StabilityLevel.STABLE:
        assessment.promotion_safe = True
        assessment.investigation_needed = False
        # No specific recommendations for stable models

    elif assessment.level == StabilityLevel.DEVELOPING:
        assessment.promotion_safe = True  # Can promote, but monitor
        assessment.investigation_needed = False
        if assessment.sample_count < 20:
            assessment.recommended_actions.append(
                "Developing: Continue evaluation games to confirm convergence"
            )

    elif assessment.level == StabilityLevel.VOLATILE:
        assessment.promotion_safe = False
        assessment.investigation_needed = True
        assessment.recommended_actions.append(
            f"High volatility (score={assessment.volatility_score:.2f}): "
            f"Investigate training stability before promotion"
        )
        if assessment.max_swing > 80:
            assessment.recommended_actions.append(
                f"Large rating swings detected (max={assessment.max_swing:.0f} Elo): "
                f"Check for inconsistent opponent strength"
            )
        if assessment.oscillation_count > assessment.sample_count * 0.5:
            assessment.recommended_actions.append(
                "Frequent direction changes: Model may be oscillating around true strength"
            )

    elif assessment.level == StabilityLevel.DECLINING:
        assessment.promotion_safe = False
        assessment.investigation_needed = True
        assessment.recommended_actions.append(
            f"Declining trend (slope={assessment.slope:.2f} Elo/hour): "
            f"Model may be regressing. Investigate training data quality"
        )
        if assessment.is_plateau and assessment.slope < -1.0:
            assessment.recommended_actions.append(
                "Consider reverting to previous checkpoint or retraining"
            )


def is_promotion_safe(
    model_id: str,
    board_type: str,
    num_players: int,
    elo_service: EloService | None = None,
) -> tuple[bool, StabilityAssessment]:
    """Check if model is stable enough for promotion.

    Convenience function for promotion gates.

    Args:
        model_id: Model identifier
        board_type: Board type
        num_players: Player count
        elo_service: Optional EloService instance

    Returns:
        Tuple of (is_safe, assessment)
    """
    assessment = assess_model_stability(
        model_id, board_type, num_players, elo_service=elo_service
    )
    return assessment.promotion_safe, assessment


def get_stability_summary(
    config_key: str,
    participant_id: str = "canonical",
) -> dict[str, Any]:
    """Get stability summary for a config key.

    Convenience function matching get_elo_trend_for_config() signature.

    Args:
        config_key: Config identifier like "square8_2p"
        participant_id: Model ID (default "canonical")

    Returns:
        Dict with stability metrics
    """
    # Parse config key
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        return {"error": f"Invalid config_key format: {config_key}"}

    board_type = parts[0]
    try:
        num_players = int(parts[1][:-1])
    except ValueError:
        return {"error": f"Invalid player count in config_key: {config_key}"}

    assessment = assess_model_stability(participant_id, board_type, num_players)
    return assessment.to_dict()


# Export main functions
__all__ = [
    "StabilityLevel",
    "StabilityAssessment",
    "assess_model_stability",
    "is_promotion_safe",
    "get_stability_summary",
]
