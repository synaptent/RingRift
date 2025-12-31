"""Game Quality Scorer.

Scores individual games based on outcome, length, phase balance, diversity,
and source reputation. Extends BaseQualityScorer for standardized scoring.

December 30, 2025: Created as part of Priority 3.3 migration.
Migrates scoring logic from unified_quality.py to the new scorer framework.

Usage:
    from app.quality.scorers.game_scorer import GameQualityScorer

    scorer = GameQualityScorer()
    result = scorer.score({
        "game_id": "abc123",
        "total_moves": 45,
        "winner": 1,
        "board_type": "hex8",
        "num_players": 2,
    })
    print(f"Score: {result.score}, Level: {result.level}")
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.quality.scorers.base import BaseQualityScorer, ScorerConfig
from app.quality.types import QualityResult, ValidationResult

__all__ = [
    "GameQualityScorer",
    "GameScorerConfig",
    "GameScorerWeights",
]

logger = logging.getLogger(__name__)


@dataclass
class GameScorerWeights:
    """Weights for game quality scoring.

    These weights determine how each component contributes to the final score.
    """

    # Component weights (should sum to 1.0 for normalized scoring)
    outcome_weight: float = 0.25
    length_weight: float = 0.25
    phase_balance_weight: float = 0.20
    diversity_weight: float = 0.15
    source_reputation_weight: float = 0.15

    # Elo normalization parameters
    min_elo: float = 1200.0
    max_elo: float = 2400.0
    default_elo: float = 1500.0

    # Game length parameters
    min_game_length: int = 10
    max_game_length: int = 200
    optimal_game_length: int = 80

    # Outcome scoring
    decisive_bonus: float = 1.0
    draw_credit: float = 0.3

    def to_dict(self) -> dict[str, float]:
        """Convert to component weights dict for BaseQualityScorer."""
        return {
            "outcome": self.outcome_weight,
            "length": self.length_weight,
            "phase_balance": self.phase_balance_weight,
            "diversity": self.diversity_weight,
            "source_reputation": self.source_reputation_weight,
        }


@dataclass
class GameScorerConfig(ScorerConfig):
    """Configuration for GameQualityScorer.

    Extends ScorerConfig with game-specific weights.
    """

    weights: GameScorerWeights = field(default_factory=GameScorerWeights)

    def __post_init__(self):
        """Set default_weights from GameScorerWeights."""
        self.default_weights = self.weights.to_dict()


class GameQualityScorer(BaseQualityScorer):
    """Quality scorer for individual games.

    Scores games based on:
    - Outcome: Whether the game had a decisive result
    - Length: How close to optimal length the game was
    - Phase balance: Distribution of game phases
    - Diversity: Move/position diversity
    - Source reputation: Reliability of the data source

    Can optionally use Elo ratings for quality adjustments.
    """

    SCORER_NAME = "game_quality"
    SCORER_VERSION = "1.1.0"

    def __init__(
        self,
        config: GameScorerConfig | None = None,
        elo_lookup: Callable[[str], float] | None = None,
    ):
        """Initialize the game scorer.

        Args:
            config: Scorer configuration with game-specific weights
            elo_lookup: Optional function to look up Elo by model ID
        """
        self._game_config = config or GameScorerConfig()
        super().__init__(
            config=self._game_config,
            weights=self._game_config.weights.to_dict(),
        )
        self.elo_lookup = elo_lookup
        self._game_weights = self._game_config.weights

    def set_elo_lookup(self, lookup: Callable[[str], float]) -> None:
        """Set the Elo lookup function."""
        self.elo_lookup = lookup

    def _validate_input(self, data: dict[str, Any]) -> ValidationResult:
        """Validate game data.

        Requires at minimum a game_id or some game-identifying field.
        """
        if not isinstance(data, dict):
            return ValidationResult.invalid("Input must be a dictionary")

        # At least one of these should be present
        has_identity = any(
            data.get(field)
            for field in ["game_id", "id", "total_moves", "move_count"]
        )
        if not has_identity:
            return ValidationResult.invalid(
                "Game data must have game_id, id, or move information"
            )

        return ValidationResult.valid()

    def _compute_score(self, data: dict[str, Any]) -> float:
        """Compute weighted game quality score."""
        components = self._compute_components(data)
        return self.weighted_average(components)

    def _compute_components(self, data: dict[str, Any]) -> dict[str, float]:
        """Compute individual component scores for a game."""
        w = self._game_weights

        # Extract game data
        game_length = (
            data.get("total_moves", 0)
            or data.get("move_count", 0)
            or 0
        )
        winner = data.get("winner")
        source = data.get("source", "")

        # Outcome score
        is_decisive = winner is not None and winner != -1
        outcome_score = w.decisive_bonus if is_decisive else w.draw_credit

        # Length score (bell curve around optimal)
        length_score = self._compute_length_score(game_length)

        # Phase balance score (from data or default)
        phase_balance_score = data.get("phase_balance_score", 0.7)

        # Diversity score (from data or default)
        diversity_score = data.get("diversity_score", 0.7)

        # Source reputation score
        source_reputation_score = 0.8 if source else 0.5

        return {
            "outcome": min(1.0, outcome_score),
            "length": length_score,
            "phase_balance": phase_balance_score,
            "diversity": diversity_score,
            "source_reputation": source_reputation_score,
        }

    def _compute_length_score(self, game_length: int) -> float:
        """Compute score based on game length.

        Uses a bell curve centered on optimal_game_length.
        """
        w = self._game_weights

        if game_length < w.min_game_length:
            return 0.0
        elif game_length > w.max_game_length:
            # Slight penalty for very long games
            return 0.8

        # Bell curve peaking at optimal length
        distance = abs(game_length - w.optimal_game_length)
        max_distance = max(
            w.optimal_game_length - w.min_game_length,
            w.max_game_length - w.optimal_game_length,
        )
        return 1.0 - (distance / max_distance) * 0.5

    def _compute_elo_score(self, data: dict[str, Any]) -> float:
        """Compute normalized Elo score if available."""
        w = self._game_weights
        model_version = data.get("model_version", "")

        if not self.elo_lookup or not model_version:
            return 0.5  # Neutral score

        try:
            elo = self.elo_lookup(model_version)
            if elo > 0:
                return max(0.0, min(1.0, (elo - w.min_elo) / (w.max_elo - w.min_elo)))
        except Exception as e:
            logger.debug(f"Elo lookup failed for {model_version}: {e}")

        return 0.5

    def score_with_metadata(self, data: dict[str, Any]) -> tuple[QualityResult, dict[str, Any]]:
        """Score game and return extended metadata.

        Returns both the QualityResult and additional game-specific metadata
        that may be useful for training or sync decisions.

        Args:
            data: Game data to score

        Returns:
            Tuple of (QualityResult, extended_metadata)
        """
        result = self.score(data)

        # Extract additional game metadata
        w = self._game_weights
        game_length = data.get("total_moves", 0) or data.get("move_count", 0) or 0
        winner = data.get("winner")
        is_decisive = winner is not None and winner != -1

        extended_metadata = {
            "game_id": data.get("game_id", ""),
            "game_length": game_length,
            "is_decisive": is_decisive,
            "elo_score": self._compute_elo_score(data),
            "model_version": data.get("model_version", ""),
            "board_type": data.get("board_type", ""),
            "num_players": data.get("num_players", 0),
            "created_at": data.get("created_at", 0.0),
        }

        return result, extended_metadata

    def compute_training_weight(
        self,
        result: QualityResult,
        recency_hours: float = 0.0,
        base_priority: float = 1.0,
        quality_weight: float = 0.4,
        recency_weight: float = 0.3,
        priority_weight: float = 0.3,
        recency_half_life_hours: float = 24.0,
    ) -> float:
        """Compute sample weight for training.

        Combines quality score, recency, and priority into a single weight.

        Args:
            result: QualityResult from scoring
            recency_hours: Hours since game was created
            base_priority: Base priority multiplier
            quality_weight: Weight for quality score component
            recency_weight: Weight for recency component
            priority_weight: Weight for priority component
            recency_half_life_hours: Half-life for recency decay

        Returns:
            Sample weight for training (>= 0)
        """
        # Recency factor with exponential decay
        recency_factor = math.exp(
            -recency_hours * math.log(2) / recency_half_life_hours
        )

        # Combine factors
        weight = (
            quality_weight * result.score
            + recency_weight * recency_factor
            + priority_weight * base_priority
        )

        return max(0.0, weight)

    def compute_sync_priority(
        self,
        result: QualityResult,
        elo_score: float = 0.5,
        is_decisive: bool = False,
        game_length: int = 0,
        sync_elo_weight: float = 0.4,
        sync_length_weight: float = 0.3,
        sync_decisive_weight: float = 0.3,
    ) -> float:
        """Compute priority score for sync ordering.

        Higher scores should be synced first.

        Args:
            result: QualityResult from scoring
            elo_score: Normalized Elo score (0-1)
            is_decisive: Whether game had decisive outcome
            game_length: Number of moves in game
            sync_elo_weight: Weight for Elo component
            sync_length_weight: Weight for length component
            sync_decisive_weight: Weight for decisive component

        Returns:
            Sync priority score (higher = more urgent)
        """
        w = self._game_weights

        # Length factor (normalized)
        if game_length < w.min_game_length:
            length_factor = 0.0
        elif game_length > w.max_game_length:
            length_factor = 0.8
        else:
            length_factor = min(
                1.0,
                (game_length - w.min_game_length)
                / (w.optimal_game_length - w.min_game_length),
            )

        # Decisive factor
        decisive_factor = 1.0 if is_decisive else 0.5

        # Combine factors
        priority = (
            sync_elo_weight * elo_score
            + sync_length_weight * length_factor
            + sync_decisive_weight * decisive_factor
        )

        return max(0.0, priority)


# Singleton instance
_game_scorer_instance: GameQualityScorer | None = None


def get_game_quality_scorer(
    config: GameScorerConfig | None = None,
    elo_lookup: Callable[[str], float] | None = None,
) -> GameQualityScorer:
    """Get singleton GameQualityScorer instance.

    Args:
        config: Optional configuration (only used on first call)
        elo_lookup: Optional Elo lookup function

    Returns:
        Singleton GameQualityScorer instance
    """
    global _game_scorer_instance
    if _game_scorer_instance is None:
        _game_scorer_instance = GameQualityScorer(config=config, elo_lookup=elo_lookup)
    elif elo_lookup is not None:
        _game_scorer_instance.set_elo_lookup(elo_lookup)
    return _game_scorer_instance


def reset_game_quality_scorer() -> None:
    """Reset singleton (for testing)."""
    global _game_scorer_instance
    _game_scorer_instance = None
