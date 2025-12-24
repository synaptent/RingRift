"""Unified Quality Scoring System.

This module provides THE SINGLE SOURCE OF TRUTH for all quality scoring
operations across the RingRift AI system. All quality-related computations
should use this module instead of scattered implementations.

Consolidates:
- GameQualityScorer (game_quality_scorer.py) - game finalization quality
- QualityExtractor (quality_extractor.py) - sync-time quality extraction
- StreamingPipeline weighting (streaming_pipeline.py) - sample weighting
- EloWeighting (elo_weighting.py) - Elo-based sample weighting
- GameQualityMetadata.compute_quality_score (unified_manifest.py) - manifest quality

The unified scorer provides:
1. compute_game_quality() - Full quality scoring at game finalization
2. compute_sample_weight() - Sample weighting for training
3. compute_sync_priority() - Priority scoring for data sync
4. Elo-based weighting with configurable parameters
5. Recency decay for freshness-weighted sampling
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "GameQuality",
    # Core classes
    "QualityCategory",
    "QualityWeights",
    "UnifiedQualityScorer",
    "compute_elo_weights_batch",
    # Convenience functions
    "compute_game_quality",
    "compute_game_quality_from_params",
    "compute_sample_weight",
    "compute_sync_priority",
    "get_quality_category",
    # Singleton access
    "get_quality_scorer",
]

logger = logging.getLogger(__name__)

# Event emission for quality updates (optional integration)
try:
    import asyncio

    from app.coordination.event_router import emit_quality_score_updated
    HAS_QUALITY_EVENTS = True
except ImportError:
    HAS_QUALITY_EVENTS = False
    emit_quality_score_updated = None


class QualityCategory(str, Enum):
    """Quality category for games.

    Thresholds match the deprecated game_quality_scorer for backwards compatibility.
    """
    EXCELLENT = "excellent"  # 0.85+
    GOOD = "good"            # 0.70-0.85
    ADEQUATE = "adequate"    # 0.50-0.70
    POOR = "poor"            # 0.30-0.50
    UNUSABLE = "unusable"    # <0.30

    @classmethod
    def from_score(cls, score: float) -> QualityCategory:
        """Get category from numeric score."""
        if score >= 0.85:
            return cls.EXCELLENT
        elif score >= 0.70:
            return cls.GOOD
        elif score >= 0.50:
            return cls.ADEQUATE
        elif score >= 0.30:
            return cls.POOR
        else:
            return cls.UNUSABLE


# Try to import config - fall back to defaults if unavailable
try:
    from app.config.unified_config import QualityConfig, get_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    QualityConfig = None


@dataclass
class GameQuality:
    """Complete quality assessment for a game.

    This consolidates all quality metrics into a single object that can
    be used for training selection, sync prioritization, and analytics.
    """
    game_id: str

    # Component scores (0-1 each)
    outcome_score: float = 0.0
    length_score: float = 0.0
    phase_balance_score: float = 0.0
    diversity_score: float = 0.0
    source_reputation_score: float = 0.0

    # Elo-based metrics
    avg_player_elo: float = 1500.0
    min_player_elo: float = 1500.0
    max_player_elo: float = 1500.0
    elo_difference: float = 0.0
    elo_score: float = 0.5  # Normalized Elo contribution

    # Game metadata
    game_length: int = 0
    is_decisive: bool = False
    termination_reason: str = ""
    model_version: str = ""
    created_at: float = 0.0

    # Computed scores
    quality_score: float = 0.0  # Primary quality score (0-1)
    training_weight: float = 1.0  # Weight for training sampling
    sync_priority: float = 0.0  # Priority for sync ordering

    @property
    def category(self) -> QualityCategory:
        """Get quality category based on quality score."""
        return QualityCategory.from_score(self.quality_score)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "quality_score": self.quality_score,
            "training_weight": self.training_weight,
            "sync_priority": self.sync_priority,
            "outcome_score": self.outcome_score,
            "length_score": self.length_score,
            "phase_balance_score": self.phase_balance_score,
            "diversity_score": self.diversity_score,
            "avg_player_elo": self.avg_player_elo,
            "elo_score": self.elo_score,
            "game_length": self.game_length,
            "is_decisive": self.is_decisive,
            "model_version": self.model_version,
        }


@dataclass
class QualityWeights:
    """Weights for quality score computation.

    Loaded from QualityConfig or uses sensible defaults.
    """
    # Game quality component weights (should sum to 1.0)
    outcome_weight: float = 0.25
    length_weight: float = 0.25
    phase_balance_weight: float = 0.20
    diversity_weight: float = 0.15
    source_reputation_weight: float = 0.15

    # Sync priority weights (should sum to 1.0)
    sync_elo_weight: float = 0.4
    sync_length_weight: float = 0.3
    sync_decisive_weight: float = 0.3

    # Elo normalization
    min_elo: float = 1200.0
    max_elo: float = 2400.0
    default_elo: float = 1500.0

    # Game length normalization
    min_game_length: int = 10
    max_game_length: int = 200
    optimal_game_length: int = 80

    # Sampling weights
    quality_weight: float = 0.4
    recency_weight: float = 0.3
    priority_weight: float = 0.3

    # Recency decay
    recency_half_life_hours: float = 24.0

    # Thresholds
    min_quality_for_training: float = 0.3
    min_quality_for_priority_sync: float = 0.5
    high_quality_threshold: float = 0.7

    # Decisive outcome
    decisive_bonus: float = 1.0
    draw_credit: float = 0.3

    @classmethod
    def from_config(cls, config: Any | None = None) -> QualityWeights:
        """Create weights from QualityConfig."""
        if config is None and HAS_CONFIG:
            try:
                unified_config = get_config()
                config = unified_config.quality
            except Exception:
                pass

        if config is None:
            return cls()

        return cls(
            outcome_weight=getattr(config, "outcome_weight", 0.25),
            length_weight=getattr(config, "length_weight", 0.25),
            phase_balance_weight=getattr(config, "phase_balance_weight", 0.20),
            diversity_weight=getattr(config, "diversity_weight", 0.15),
            source_reputation_weight=getattr(config, "source_reputation_weight", 0.15),
            sync_elo_weight=getattr(config, "sync_elo_weight", 0.4),
            sync_length_weight=getattr(config, "sync_length_weight", 0.3),
            sync_decisive_weight=getattr(config, "sync_decisive_weight", 0.3),
            min_elo=getattr(config, "min_elo", 1200.0),
            max_elo=getattr(config, "max_elo", 2400.0),
            default_elo=getattr(config, "default_elo", 1500.0),
            min_game_length=getattr(config, "min_game_length", 10),
            max_game_length=getattr(config, "max_game_length", 200),
            optimal_game_length=getattr(config, "optimal_game_length", 80),
            quality_weight=getattr(config, "quality_weight", 0.4),
            recency_weight=getattr(config, "recency_weight", 0.3),
            priority_weight=getattr(config, "priority_weight", 0.3),
            recency_half_life_hours=getattr(config, "recency_half_life_hours", 24.0),
            min_quality_for_training=getattr(config, "min_quality_for_training", 0.3),
            min_quality_for_priority_sync=getattr(config, "min_quality_for_priority_sync", 0.5),
            high_quality_threshold=getattr(config, "high_quality_threshold", 0.7),
            decisive_bonus=getattr(config, "decisive_bonus", 1.0),
            draw_credit=getattr(config, "draw_credit", 0.3),
        )


class UnifiedQualityScorer:
    """Unified quality scorer for all quality computations.

    This class provides a single entry point for all quality scoring
    operations across the system. It uses the QualityConfig for
    parameters and provides consistent scoring across all use cases.
    """

    _instance: UnifiedQualityScorer | None = None

    def __init__(
        self,
        weights: QualityWeights | None = None,
        elo_lookup: Callable[[str], float] | None = None,
    ):
        """Initialize the scorer.

        Args:
            weights: Quality weights (loaded from config if not provided)
            elo_lookup: Function to look up Elo by model ID
        """
        self.weights = weights or QualityWeights.from_config()
        self.elo_lookup = elo_lookup

    @classmethod
    def get_instance(
        cls,
        weights: QualityWeights | None = None,
    ) -> UnifiedQualityScorer:
        """Get singleton instance of the scorer."""
        if cls._instance is None:
            cls._instance = cls(weights=weights)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (mainly for testing)."""
        cls._instance = None

    def set_elo_lookup(self, lookup: Callable[[str], float]) -> None:
        """Set the Elo lookup function."""
        self.elo_lookup = lookup

    def compute_game_quality(
        self,
        game_data: dict[str, Any],
        elo_lookup: Callable[[str], float] | None = None,
    ) -> GameQuality:
        """Compute full quality assessment for a game.

        This is the primary quality computation used at game finalization.
        It considers all aspects: outcome, length, phase balance, diversity,
        and source reputation.

        Args:
            game_data: Dictionary with game metadata
            elo_lookup: Optional override for Elo lookup function

        Returns:
            GameQuality with all scores computed
        """
        lookup = elo_lookup or self.elo_lookup
        w = self.weights

        game_id = game_data.get("game_id", "")
        game_length = game_data.get("total_moves", 0) or game_data.get("move_count", 0) or 0
        winner = game_data.get("winner")
        termination_reason = game_data.get("termination_reason", "")
        source = game_data.get("source", "")
        created_at = game_data.get("created_at", 0.0)
        model_version = game_data.get("model_version", "")

        # Determine if decisive
        is_decisive = winner is not None and winner != -1

        # Compute outcome score
        outcome_score = w.decisive_bonus if is_decisive else w.draw_credit

        # Compute length score (bell curve around optimal)
        if game_length < w.min_game_length:
            length_score = 0.0
        elif game_length > w.max_game_length:
            # Slight penalty for very long games
            length_score = 0.8
        else:
            # Bell curve peaking at optimal length
            distance = abs(game_length - w.optimal_game_length)
            max_distance = max(
                w.optimal_game_length - w.min_game_length,
                w.max_game_length - w.optimal_game_length
            )
            length_score = 1.0 - (distance / max_distance) * 0.5

        # Compute phase balance score (from game data if available)
        phase_balance_score = game_data.get("phase_balance_score", 0.7)

        # Compute diversity score (from game data if available)
        diversity_score = game_data.get("diversity_score", 0.7)

        # Compute source reputation score (simplified)
        source_reputation_score = 0.8 if source else 0.5

        # Get Elo ratings
        avg_elo = w.default_elo
        min_elo = w.default_elo
        max_elo = w.default_elo
        elo_difference = 0.0

        if lookup and model_version:
            try:
                player_elo = lookup(model_version)
                if player_elo > 0:
                    avg_elo = player_elo
                    min_elo = player_elo
                    max_elo = player_elo
            except Exception as e:
                logger.debug(f"Elo lookup failed for {model_version}: {e}")

        # Compute Elo score (normalized)
        elo_score = max(0.0, min(1.0, (avg_elo - w.min_elo) / (w.max_elo - w.min_elo)))

        # Compute weighted quality score
        quality_score = (
            w.outcome_weight * outcome_score +
            w.length_weight * length_score +
            w.phase_balance_weight * phase_balance_score +
            w.diversity_weight * diversity_score +
            w.source_reputation_weight * source_reputation_score
        )
        quality_score = max(0.0, min(1.0, quality_score))

        # Create GameQuality object
        quality = GameQuality(
            game_id=game_id,
            outcome_score=outcome_score,
            length_score=length_score,
            phase_balance_score=phase_balance_score,
            diversity_score=diversity_score,
            source_reputation_score=source_reputation_score,
            avg_player_elo=avg_elo,
            min_player_elo=min_elo,
            max_player_elo=max_elo,
            elo_difference=elo_difference,
            elo_score=elo_score,
            game_length=game_length,
            is_decisive=is_decisive,
            termination_reason=termination_reason,
            model_version=model_version,
            created_at=created_at if isinstance(created_at, float) else 0.0,
            quality_score=quality_score,
        )

        # Compute training weight and sync priority
        quality.training_weight = self.compute_sample_weight(quality)
        quality.sync_priority = self.compute_sync_priority(quality)

        # Emit quality event for coordination (async, non-blocking)
        if HAS_QUALITY_EVENTS and emit_quality_score_updated is not None and game_id:
            try:
                asyncio.get_running_loop()
                asyncio.ensure_future(emit_quality_score_updated(
                    game_id=game_id,
                    quality_score=quality.quality_score,
                    quality_category=quality.category.value,
                    training_weight=quality.training_weight,
                    game_length=game_length,
                    is_decisive=is_decisive,
                    source="unified_quality",
                ))
            except RuntimeError:
                pass  # No running loop - skip event emission

        return quality

    def compute_sample_weight(
        self,
        quality: GameQuality,
        recency_hours: float | None = None,
        base_priority: float = 1.0,
    ) -> float:
        """Compute sample weight for training.

        Combines quality score, recency, and priority into a single weight.

        Args:
            quality: GameQuality object or quality score
            recency_hours: Hours since game was created (computed if not provided)
            base_priority: Base priority multiplier

        Returns:
            Sample weight for training (>= 0)
        """
        w = self.weights

        # Get quality score
        if isinstance(quality, GameQuality):
            quality_score = quality.quality_score
            created_at = quality.created_at
        else:
            quality_score = float(quality)
            created_at = 0.0

        # Compute recency if not provided
        if recency_hours is None and created_at > 0:
            recency_hours = (time.time() - created_at) / 3600.0
        elif recency_hours is None:
            recency_hours = 0.0

        # Recency factor with exponential decay
        half_life = w.recency_half_life_hours
        recency_factor = math.exp(-recency_hours * math.log(2) / half_life)

        # Combine factors
        weight = (
            w.quality_weight * quality_score +
            w.recency_weight * recency_factor +
            w.priority_weight * base_priority
        )

        return max(0.0, weight)

    def compute_sync_priority(
        self,
        quality: GameQuality,
        urgency_hours: float = 0.0,
    ) -> float:
        """Compute priority score for sync ordering.

        Higher scores should be synced first.

        Args:
            quality: GameQuality object
            urgency_hours: Hours the game has been waiting

        Returns:
            Priority score (0-1)
        """
        w = self.weights

        # Normalize Elo
        elo_normalized = max(0.0, min(1.0,
            (quality.avg_player_elo - w.min_elo) / (w.max_elo - w.min_elo)
        ))

        # Normalize length
        if quality.game_length < w.min_game_length:
            length_normalized = 0.0
        elif quality.game_length > w.max_game_length:
            length_normalized = 1.0
        else:
            length_normalized = (
                (quality.game_length - w.min_game_length) /
                (w.max_game_length - w.min_game_length)
            )

        # Decisive factor
        decisive_normalized = w.decisive_bonus if quality.is_decisive else w.draw_credit

        # Weighted combination
        priority = (
            w.sync_elo_weight * elo_normalized +
            w.sync_length_weight * length_normalized +
            w.sync_decisive_weight * decisive_normalized
        )

        # Add urgency bonus (older unsynced games get priority)
        urgency_bonus = min(0.2, urgency_hours / 24.0 * 0.2)
        priority += urgency_bonus

        return max(0.0, min(1.0, priority))

    def compute_elo_weight(
        self,
        player_elo: float,
        opponent_elo: float,
    ) -> float:
        """Compute Elo-based sample weight.

        Uses sigmoid transformation based on Elo difference.
        Games against stronger opponents get higher weight.

        Args:
            player_elo: Player's Elo rating
            opponent_elo: Opponent's Elo rating

        Returns:
            Weight multiplier (typically 0.5 - 2.0)
        """

        # Elo difference (positive = opponent is stronger)
        elo_diff = opponent_elo - player_elo

        # Sigmoid transformation
        scale = 400.0  # Standard Elo scale
        sigmoid = 1.0 / (1.0 + math.exp(-elo_diff / scale))

        # Map to weight range (0.5 - 2.0 by default)
        min_weight = 0.5
        max_weight = 2.0
        weight = min_weight + sigmoid * (max_weight - min_weight)

        return weight

    def compute_elo_weights_batch(
        self,
        opponent_elos: list[float],
        model_elo: float,
        elo_scale: float = 400.0,
        min_weight: float = 0.2,
        max_weight: float = 3.0,
        normalize: bool = True,
    ) -> list[float]:
        """Compute Elo-based sample weights for a batch of samples.

        This is the canonical implementation used by EloWeightedSampler.

        Args:
            opponent_elos: List of opponent Elo ratings
            model_elo: Current model's Elo rating
            elo_scale: Scaling factor for Elo difference (default: 400)
            min_weight: Minimum sample weight (default: 0.2)
            max_weight: Maximum sample weight (default: 3.0)
            normalize: Normalize weights to mean=1 (default: True)

        Returns:
            List of sample weights
        """
        if not opponent_elos:
            return []

        weights = []
        for opp_elo in opponent_elos:
            elo_diff = opp_elo - model_elo
            sigmoid = 1.0 / (1.0 + math.exp(-elo_diff / elo_scale))
            weight = min_weight + sigmoid * (max_weight - min_weight)
            weights.append(weight)

        if normalize and weights:
            mean_weight = sum(weights) / len(weights)
            if mean_weight > 0:
                weights = [w / mean_weight for w in weights]

        return weights

    def is_high_quality(self, quality: GameQuality) -> bool:
        """Check if a game is high quality."""
        return quality.quality_score >= self.weights.high_quality_threshold

    def is_training_worthy(self, quality: GameQuality) -> bool:
        """Check if a game meets minimum quality for training."""
        return quality.quality_score >= self.weights.min_quality_for_training

    def is_priority_sync_worthy(self, quality: GameQuality) -> bool:
        """Check if a game should be priority synced."""
        return quality.quality_score >= self.weights.min_quality_for_priority_sync

    def compute_freshness_score(
        self,
        game_timestamp: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """Compute freshness score using exponential decay.

        This method provides backwards compatibility with DataQualityScorer
        from training_enhancements.py. Recent games get higher scores.

        Args:
            game_timestamp: Unix timestamp when game was played
            current_time: Current time (default: time.time())

        Returns:
            Freshness score (0-1, where 1 = newest)
        """
        if game_timestamp is None:
            return 0.5  # Neutral if unknown

        if current_time is None:
            current_time = time.time()

        age_hours = (current_time - game_timestamp) / 3600
        if age_hours < 0:
            return 1.0  # Future timestamp = max freshness

        # Exponential decay using recency half-life from weights
        half_life = self.weights.recency_half_life_hours
        freshness = math.exp(-age_hours * math.log(2) / half_life)
        return max(0.0, min(1.0, freshness))


# Module-level convenience functions

def get_quality_scorer() -> UnifiedQualityScorer:
    """Get the singleton quality scorer instance."""
    return UnifiedQualityScorer.get_instance()


def compute_game_quality(
    game_data: dict[str, Any],
    elo_lookup: Callable[[str], float] | None = None,
) -> GameQuality:
    """Compute quality for a game (convenience function)."""
    scorer = get_quality_scorer()
    return scorer.compute_game_quality(game_data, elo_lookup)


def compute_sample_weight(
    quality: GameQuality,
    recency_hours: float | None = None,
    base_priority: float = 1.0,
) -> float:
    """Compute sample weight (convenience function)."""
    scorer = get_quality_scorer()
    return scorer.compute_sample_weight(quality, recency_hours, base_priority)


def compute_sync_priority(
    quality: GameQuality,
    urgency_hours: float = 0.0,
) -> float:
    """Compute sync priority (convenience function)."""
    scorer = get_quality_scorer()
    return scorer.compute_sync_priority(quality, urgency_hours)


def get_quality_category(score: float) -> QualityCategory:
    """Get quality category from numeric score (convenience function)."""
    return QualityCategory.from_score(score)


def compute_game_quality_from_params(
    game_id: str,
    game_status: str,
    winner: int | None,
    termination_reason: str | None,
    total_moves: int,
    board_type: str = "square8",
    source: str | None = None,
) -> GameQuality:
    """Compute game quality from individual parameters.

    This is a backwards-compatible wrapper that matches the old
    game_quality_scorer.compute_game_quality() signature.

    Args:
        game_id: Unique game identifier
        game_status: Game status string (e.g., "completed", "in_progress")
        winner: Winner player index, or None for draw/incomplete
        termination_reason: How the game ended
        total_moves: Number of moves in the game
        board_type: Board type (e.g., "square8", "hex8")
        source: Data source identifier

    Returns:
        GameQuality with all scores computed
    """
    game_data = {
        "game_id": game_id,
        "game_status": game_status,
        "winner": winner,
        "termination_reason": termination_reason,
        "total_moves": total_moves,
        "board_type": board_type,
        "source": source,
    }
    return compute_game_quality(game_data)


def compute_elo_weights_batch(
    opponent_elos: list[float],
    model_elo: float,
    elo_scale: float = 400.0,
    min_weight: float = 0.2,
    max_weight: float = 3.0,
    normalize: bool = True,
) -> list[float]:
    """Compute Elo-based sample weights for a batch (convenience function).

    This is the canonical Elo weight computation used across the codebase.

    Args:
        opponent_elos: List of opponent Elo ratings
        model_elo: Current model's Elo rating
        elo_scale: Scaling factor for Elo difference (default: 400)
        min_weight: Minimum sample weight (default: 0.2)
        max_weight: Maximum sample weight (default: 3.0)
        normalize: Normalize weights to mean=1 (default: True)

    Returns:
        List of sample weights
    """
    scorer = get_quality_scorer()
    return scorer.compute_elo_weights_batch(
        opponent_elos, model_elo, elo_scale, min_weight, max_weight, normalize
    )
