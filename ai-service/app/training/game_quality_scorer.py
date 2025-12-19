"""Game Quality Scoring for Training Data Prioritization.

This module computes quality scores for completed games to enable:
- Quality-aware data synchronization (sync best data first)
- Training sample weighting (prefer high-quality games)
- Data source reputation tracking

Quality Score Formula (0.0 - 1.0):
- Outcome validity (25%): Complete game with clear winner
- Game length score (25%): Sweet spot of 40-120 moves
- Phase balance (20%): Mix of early/mid/late game positions
- Move diversity (15%): Policy entropy (varied move selection)
- Source reputation (15%): Historical quality from this source

Usage:
    from app.training.game_quality_scorer import (
        GameQualityScorer,
        compute_game_quality,
        QualityCategory,
    )

    scorer = GameQualityScorer()
    quality = scorer.score_game(game_data)
    print(f"Quality: {quality.score:.2f} ({quality.category})")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityCategory(str, Enum):
    """Quality category for games."""
    EXCELLENT = "excellent"  # 0.85+
    GOOD = "good"            # 0.70-0.85
    ADEQUATE = "adequate"    # 0.50-0.70
    POOR = "poor"            # 0.30-0.50
    UNUSABLE = "unusable"    # <0.30

    @classmethod
    def from_score(cls, score: float) -> "QualityCategory":
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


@dataclass
class GameQuality:
    """Quality assessment for a single game."""
    game_id: str
    score: float  # 0.0 - 1.0
    category: QualityCategory

    # Component scores (0.0 - 1.0 each)
    outcome_score: float = 0.0
    length_score: float = 0.0
    phase_balance_score: float = 0.0
    diversity_score: float = 0.0
    source_score: float = 1.0  # Default to neutral

    # Metadata
    total_moves: int = 0
    phase_distribution: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class SourceReputation:
    """Track quality reputation of a data source."""
    source_id: str
    total_games: int = 0
    avg_quality: float = 1.0
    recent_quality: float = 1.0  # Last 100 games
    quality_trend: float = 0.0   # Positive = improving


class GameQualityScorer:
    """Compute quality scores for games.

    Scoring weights (configurable):
    - outcome_weight: 0.25 (game completed with valid outcome)
    - length_weight: 0.25 (optimal game length)
    - phase_weight: 0.20 (balanced early/mid/late positions)
    - diversity_weight: 0.15 (varied move selection)
    - source_weight: 0.15 (source reputation)
    """

    # Configurable weights
    DEFAULT_WEIGHTS = {
        "outcome": 0.25,
        "length": 0.25,
        "phase_balance": 0.20,
        "diversity": 0.15,
        "source": 0.15,
    }

    # Game length sweet spots (by board type)
    OPTIMAL_LENGTH_RANGE = {
        "square8": (40, 120),
        "hex8": (50, 150),
        "default": (40, 120),
    }

    # Phase boundaries (move number)
    PHASE_BOUNDARIES = {
        "early": (0, 39),
        "mid": (40, 79),
        "late": (80, float("inf")),
    }

    # Ideal phase distribution (for balanced training data)
    IDEAL_PHASE_DISTRIBUTION = {
        "early": 0.25,
        "mid": 0.40,
        "late": 0.35,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        source_reputations: Optional[Dict[str, SourceReputation]] = None,
    ):
        """Initialize scorer with optional custom weights."""
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.source_reputations = source_reputations or {}

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Quality weights sum to {total}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= total

    def score_game(
        self,
        game_id: str,
        game_status: str,
        winner: Optional[int],
        termination_reason: Optional[str],
        total_moves: int,
        board_type: str = "square8",
        source: Optional[str] = None,
        move_entropies: Optional[List[float]] = None,
    ) -> GameQuality:
        """Compute quality score for a game.

        Args:
            game_id: Unique game identifier
            game_status: 'completed', 'abandoned', etc.
            winner: Player number who won, or None
            termination_reason: How game ended
            total_moves: Total move count
            board_type: Board type for length calibration
            source: Data source identifier for reputation
            move_entropies: Optional policy entropies per move

        Returns:
            GameQuality with score and breakdown
        """
        # Compute individual component scores
        outcome = self._score_outcome(game_status, winner, termination_reason)
        length = self._score_length(total_moves, board_type)
        phase = self._score_phase_balance(total_moves)
        diversity = self._score_diversity(move_entropies)
        source_score = self._get_source_score(source)

        # Weighted combination
        score = (
            self.weights["outcome"] * outcome +
            self.weights["length"] * length +
            self.weights["phase_balance"] * phase +
            self.weights["diversity"] * diversity +
            self.weights["source"] * source_score
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Build reason string
        reasons = []
        if outcome < 0.5:
            reasons.append("incomplete/invalid outcome")
        if length < 0.5:
            reasons.append(f"suboptimal length ({total_moves} moves)")
        if phase < 0.5:
            reasons.append("unbalanced phases")
        if diversity < 0.5:
            reasons.append("low move diversity")

        reason = "; ".join(reasons) if reasons else "good quality"

        return GameQuality(
            game_id=game_id,
            score=score,
            category=QualityCategory.from_score(score),
            outcome_score=outcome,
            length_score=length,
            phase_balance_score=phase,
            diversity_score=diversity,
            source_score=source_score,
            total_moves=total_moves,
            phase_distribution=self._get_phase_distribution(total_moves),
            reason=reason,
        )

    def _score_outcome(
        self,
        game_status: str,
        winner: Optional[int],
        termination_reason: Optional[str],
    ) -> float:
        """Score based on game outcome validity."""
        # Incomplete games are low quality
        if game_status != "completed":
            return 0.2

        # Games with clear winner are best
        if winner is not None and winner > 0:
            # Check termination reason
            if termination_reason in ("elimination", "territory", "last_player_standing"):
                return 1.0
            elif termination_reason == "timeout":
                return 0.7  # Timeout wins are less informative
            elif termination_reason == "move_limit":
                return 0.6  # Hit move limit
            else:
                return 0.8  # Unknown but valid

        # Draws are valid but less informative
        if winner == 0 or termination_reason == "draw":
            return 0.6

        # No winner, completed - might be abandoned
        return 0.4

    def _score_length(self, total_moves: int, board_type: str) -> float:
        """Score based on game length (sweet spot preferred)."""
        # Get optimal range for board type
        min_opt, max_opt = self.OPTIMAL_LENGTH_RANGE.get(
            board_type, self.OPTIMAL_LENGTH_RANGE["default"]
        )

        # Too short games are low quality (not enough training data)
        if total_moves < 10:
            return 0.1
        if total_moves < 20:
            return 0.3
        if total_moves < min_opt:
            # Linear ramp up to optimal
            return 0.5 + 0.5 * (total_moves - 20) / (min_opt - 20)

        # In sweet spot
        if total_moves <= max_opt:
            return 1.0

        # Too long - gradual decay
        excess = total_moves - max_opt
        decay = math.exp(-excess / 50)  # Soft decay
        return max(0.5, decay)

    def _score_phase_balance(self, total_moves: int) -> float:
        """Score based on game phase distribution."""
        dist = self._get_phase_distribution(total_moves)

        # Compare to ideal distribution
        total_diff = 0.0
        for phase, ideal in self.IDEAL_PHASE_DISTRIBUTION.items():
            actual = dist.get(phase, 0.0)
            total_diff += abs(actual - ideal)

        # Convert difference to score (lower diff = higher score)
        # Max possible diff is 2.0 (completely wrong distribution)
        score = 1.0 - (total_diff / 2.0)
        return max(0.0, score)

    def _get_phase_distribution(self, total_moves: int) -> Dict[str, float]:
        """Compute phase distribution for a game."""
        if total_moves == 0:
            return {"early": 1.0, "mid": 0.0, "late": 0.0}

        dist = {}
        for phase, (start, end) in self.PHASE_BOUNDARIES.items():
            # Count moves in this phase
            phase_start = start
            phase_end = min(end, total_moves)
            if phase_start >= total_moves:
                dist[phase] = 0.0
            else:
                moves_in_phase = max(0, phase_end - phase_start)
                dist[phase] = moves_in_phase / total_moves

        return dist

    def _score_diversity(self, move_entropies: Optional[List[float]]) -> float:
        """Score based on move diversity (policy entropy)."""
        if not move_entropies:
            # No entropy data available - assume average
            return 0.7

        if len(move_entropies) < 5:
            # Too few moves to assess
            return 0.5

        # Average entropy (higher = more diverse)
        avg_entropy = sum(move_entropies) / len(move_entropies)

        # Normalize entropy to 0-1 (assume max entropy ~3.0 for typical games)
        normalized = min(1.0, avg_entropy / 3.0)

        # Some randomness is good, but too much might be noise
        # Sweet spot: 0.3-0.7 normalized entropy
        if 0.3 <= normalized <= 0.7:
            return 1.0
        elif normalized < 0.3:
            # Too deterministic
            return 0.5 + normalized
        else:
            # Too random
            return 1.0 - (normalized - 0.7) * 0.5

    def _get_source_score(self, source: Optional[str]) -> float:
        """Get reputation score for a data source."""
        if not source:
            return 1.0  # Unknown source, neutral

        rep = self.source_reputations.get(source)
        if not rep:
            return 1.0  # New source, neutral

        # Use recent quality with small boost for improving sources
        score = rep.recent_quality
        if rep.quality_trend > 0:
            score = min(1.0, score + 0.05)
        elif rep.quality_trend < 0:
            score = max(0.0, score - 0.05)

        return score

    def update_source_reputation(
        self,
        source: str,
        game_quality: float,
        window_size: int = 100,
    ) -> SourceReputation:
        """Update source reputation with new game quality.

        Args:
            source: Source identifier
            game_quality: Quality score of new game
            window_size: Window for recent quality calculation

        Returns:
            Updated SourceReputation
        """
        if source not in self.source_reputations:
            self.source_reputations[source] = SourceReputation(source_id=source)

        rep = self.source_reputations[source]
        old_recent = rep.recent_quality

        # Update counts
        rep.total_games += 1

        # Update averages (exponential moving average for efficiency)
        alpha = 1.0 / min(rep.total_games, window_size)
        rep.avg_quality = (1 - alpha) * rep.avg_quality + alpha * game_quality
        rep.recent_quality = (1 - alpha) * rep.recent_quality + alpha * game_quality

        # Update trend
        rep.quality_trend = rep.recent_quality - old_recent

        return rep


# Singleton instance for global access
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

    Uses the global scorer instance.
    """
    scorer = get_quality_scorer()
    return scorer.score_game(
        game_id=game_id,
        game_status=game_status,
        winner=winner,
        termination_reason=termination_reason,
        total_moves=total_moves,
        board_type=board_type,
        source=source,
    )
