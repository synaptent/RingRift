"""Game Quality Scoring for Training Data Prioritization.

.. deprecated::
    This module is deprecated. Use app.quality.unified_quality instead:

        from app.quality.unified_quality import (
            UnifiedQualityScorer,
            compute_game_quality_from_params,
            get_quality_category,
            QualityCategory,
            GameQuality,
        )

    This module now re-exports from unified_quality for backwards compatibility.
    New code should import directly from app.quality.unified_quality.
"""

from __future__ import annotations

import warnings
from typing import Optional

# Re-export from unified_quality for backwards compatibility
from app.quality.unified_quality import (
    QualityCategory,
    GameQuality as _UnifiedGameQuality,
    UnifiedQualityScorer,
    compute_game_quality_from_params,
    get_quality_category,
    get_quality_scorer as _get_unified_scorer,
)

# Emit deprecation warning on import
warnings.warn(
    "app.training.game_quality_scorer is deprecated. "
    "Use app.quality.unified_quality instead.",
    DeprecationWarning,
    stacklevel=2,
)


class GameQuality:
    """Backwards-compatible GameQuality wrapper.

    Wraps the new unified_quality.GameQuality to provide the old API
    with .score and .category attributes.
    """

    def __init__(
        self,
        game_id: str,
        score: float,
        category: QualityCategory,
        outcome_score: float = 0.0,
        length_score: float = 0.0,
        phase_balance_score: float = 0.0,
        diversity_score: float = 0.0,
        source_score: float = 1.0,
        total_moves: int = 0,
        phase_distribution: dict = None,
        reason: str = "",
    ):
        self.game_id = game_id
        self.score = score  # Old API uses .score
        self.category = category
        self.outcome_score = outcome_score
        self.length_score = length_score
        self.phase_balance_score = phase_balance_score
        self.diversity_score = diversity_score
        self.source_score = source_score
        self.total_moves = total_moves
        self.phase_distribution = phase_distribution or {}
        self.reason = reason

    @classmethod
    def from_unified(cls, unified: _UnifiedGameQuality) -> "GameQuality":
        """Create from unified GameQuality."""
        return cls(
            game_id=unified.game_id,
            score=unified.quality_score,  # Map quality_score -> score
            category=get_quality_category(unified.quality_score),
            outcome_score=unified.outcome_score,
            length_score=unified.length_score,
            phase_balance_score=unified.phase_balance_score,
            diversity_score=unified.diversity_score,
            source_score=unified.source_reputation_score,
            total_moves=unified.game_length,
        )

    def to_dict(self):
        """Convert to dictionary for storage."""
        return {
            "game_id": self.game_id,
            "score": self.score,
            "category": self.category.value,
            "outcome_score": self.outcome_score,
            "length_score": self.length_score,
            "phase_balance_score": self.phase_balance_score,
            "diversity_score": self.diversity_score,
            "source_score": self.source_score,
            "total_moves": self.total_moves,
            "phase_distribution": self.phase_distribution,
            "reason": self.reason,
        }


class GameQualityScorer:
    """Backwards-compatible wrapper around UnifiedQualityScorer."""

    def __init__(self, weights=None, source_reputations=None):
        warnings.warn(
            "GameQualityScorer is deprecated. Use UnifiedQualityScorer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._unified = _get_unified_scorer()

    def score_game(
        self,
        game_id: str,
        game_status: str,
        winner: Optional[int],
        termination_reason: Optional[str],
        total_moves: int,
        board_type: str = "square8",
        source: Optional[str] = None,
        move_entropies=None,
    ) -> GameQuality:
        """Compute quality score for a game."""
        unified_quality = compute_game_quality_from_params(
            game_id=game_id,
            game_status=game_status,
            winner=winner,
            termination_reason=termination_reason,
            total_moves=total_moves,
            board_type=board_type,
            source=source,
        )
        return GameQuality.from_unified(unified_quality)


# Singleton instance
_scorer_instance: Optional[GameQualityScorer] = None


def get_quality_scorer() -> GameQualityScorer:
    """Get or create the global quality scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = GameQualityScorer()
    return _scorer_instance


def compute_game_quality(
    game_id: str,
    game_status: str,
    winner: Optional[int],
    termination_reason: Optional[str],
    total_moves: int,
    board_type: str = "square8",
    source: Optional[str] = None,
) -> GameQuality:
    """Convenience function to compute game quality.

    .. deprecated:: Use compute_game_quality_from_params from unified_quality.
    """
    unified_quality = compute_game_quality_from_params(
        game_id=game_id,
        game_status=game_status,
        winner=winner,
        termination_reason=termination_reason,
        total_moves=total_moves,
        board_type=board_type,
        source=source,
    )
    return GameQuality.from_unified(unified_quality)


__all__ = [
    "GameQuality",
    "GameQualityScorer",
    "QualityCategory",
    "compute_game_quality",
    "get_quality_scorer",
]
