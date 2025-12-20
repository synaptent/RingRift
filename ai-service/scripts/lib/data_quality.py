"""
Data Quality Assessment Library

Provides utilities for evaluating and filtering training data quality:
- Game quality scoring
- Victory type analysis
- Data deduplication
- Quality statistics

Usage:
    from scripts.lib.data_quality import GameQualityScorer, QualityFilter

    scorer = GameQualityScorer(config="square8_2p")
    quality = scorer.score(game_data)

    filter = QualityFilter(min_quality=0.7)
    high_quality_games = filter.filter(games)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class VictoryType(Enum):
    """Types of game victory conditions."""
    RING = "ring"
    RING_FORMATION = "ring_formation"
    ELIMINATION = "elimination"
    RING_ELIMINATION = "ring_elimination"
    TERRITORIAL = "territorial"
    TERRITORY = "territory"
    RESIGNATION = "resignation"
    TIMEOUT = "timeout"
    STALEMATE = "stalemate"
    DRAW = "draw"
    LPS = "lps"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "VictoryType":
        """Parse victory type from string."""
        value = value.lower().strip()
        for vt in cls:
            if vt.value == value:
                return vt
        return cls.UNKNOWN


# Training value of different victory types (0-1 scale)
VICTORY_TYPE_VALUE: dict[VictoryType, float] = {
    VictoryType.RING: 1.0,
    VictoryType.RING_FORMATION: 1.0,
    VictoryType.ELIMINATION: 0.9,
    VictoryType.RING_ELIMINATION: 0.9,
    VictoryType.TERRITORIAL: 0.8,
    VictoryType.TERRITORY: 0.8,
    VictoryType.RESIGNATION: 0.7,
    VictoryType.LPS: 0.6,
    VictoryType.UNKNOWN: 0.5,
    VictoryType.STALEMATE: 0.3,
    VictoryType.DRAW: 0.3,
    VictoryType.TIMEOUT: 0.2,
}


@dataclass
class GameLengthConfig:
    """Configuration for optimal game length by config."""
    min_moves: int
    max_moves: int
    ideal_min: int
    ideal_max: int

    @classmethod
    def for_config(cls, config_key: str) -> "GameLengthConfig":
        """Get optimal length config for a board/player configuration."""
        configs = {
            "square8_2p": cls(min_moves=15, max_moves=200, ideal_min=25, ideal_max=120),
            "square8_3p": cls(min_moves=20, max_moves=250, ideal_min=30, ideal_max=150),
            "square8_4p": cls(min_moves=25, max_moves=300, ideal_min=40, ideal_max=180),
            "hex8_2p": cls(min_moves=10, max_moves=150, ideal_min=20, ideal_max=100),
            "hex8_3p": cls(min_moves=15, max_moves=200, ideal_min=25, ideal_max=130),
            "hex8_4p": cls(min_moves=20, max_moves=250, ideal_min=35, ideal_max=160),
            "square19_2p": cls(min_moves=30, max_moves=500, ideal_min=50, ideal_max=300),
            "square19_3p": cls(min_moves=40, max_moves=600, ideal_min=60, ideal_max=400),
            "square19_4p": cls(min_moves=50, max_moves=700, ideal_min=80, ideal_max=500),
        }
        return configs.get(config_key, cls(min_moves=20, max_moves=200, ideal_min=25, ideal_max=150))


@dataclass
class QualityScores:
    """Breakdown of quality scores by criterion."""
    decisive_win: float = 0.0
    game_length: float = 0.0
    victory_type: float = 0.0
    move_diversity: float = 0.0
    recency: float = 0.0
    source_elo: float = 0.0
    tactical_content: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "decisive_win": self.decisive_win,
            "game_length": self.game_length,
            "victory_type": self.victory_type,
            "move_diversity": self.move_diversity,
            "recency": self.recency,
            "source_elo": self.source_elo,
            "tactical_content": self.tactical_content,
        }


@dataclass
class GameQuality:
    """Quality assessment result for a game."""
    game_id: str
    config_key: str
    total_score: float
    scores: QualityScores
    victory_type: VictoryType
    move_count: int
    source_file: str | None = None
    source_node: str | None = None

    @property
    def quality_bucket(self) -> str:
        """Get quality bucket (e.g., '0.8' for scores 0.80-0.89)."""
        return f"{int(self.total_score * 10) / 10:.1f}"


@dataclass
class QualityWeights:
    """Weights for quality scoring criteria."""
    decisive_win: float = 0.25
    game_length: float = 0.20
    victory_type: float = 0.15
    move_diversity: float = 0.15
    recency: float = 0.10
    source_elo: float = 0.10
    tactical_content: float = 0.05

    def normalize(self) -> "QualityWeights":
        """Normalize weights to sum to 1.0."""
        total = (
            self.decisive_win + self.game_length + self.victory_type +
            self.move_diversity + self.recency + self.source_elo + self.tactical_content
        )
        if total == 0:
            return self
        return QualityWeights(
            decisive_win=self.decisive_win / total,
            game_length=self.game_length / total,
            victory_type=self.victory_type / total,
            move_diversity=self.move_diversity / total,
            recency=self.recency / total,
            source_elo=self.source_elo / total,
            tactical_content=self.tactical_content / total,
        )


class GameQualityScorer:
    """Scores game quality for training data selection."""

    def __init__(
        self,
        config_key: str = "square8_2p",
        weights: QualityWeights | None = None,
        reference_time: datetime | None = None,
    ):
        """Initialize the scorer.

        Args:
            config_key: Board/player config (e.g., "square8_2p")
            weights: Custom quality weights (uses defaults if not specified)
            reference_time: Reference time for recency scoring (default: now)
        """
        self.config_key = config_key
        self.weights = (weights or QualityWeights()).normalize()
        self.reference_time = reference_time or datetime.now()
        self.length_config = GameLengthConfig.for_config(config_key)

    def score(
        self,
        game: dict[str, Any],
        source_file: str | None = None,
        source_node: str | None = None,
    ) -> GameQuality:
        """Score a game's quality for training.

        Args:
            game: Game data dictionary
            source_file: Optional source file path
            source_node: Optional source node name

        Returns:
            GameQuality with detailed scoring
        """
        scores = QualityScores()

        # Extract game info
        game_id = self._extract_game_id(game)
        victory_type = self._extract_victory_type(game)
        move_count = self._extract_move_count(game)
        moves = game.get("moves", [])

        # 1. Decisive win score
        scores.decisive_win = self._score_decisive_win(game, victory_type)

        # 2. Game length score
        scores.game_length = self._score_game_length(move_count)

        # 3. Victory type score
        scores.victory_type = VICTORY_TYPE_VALUE.get(victory_type, 0.5)

        # 4. Move diversity score
        scores.move_diversity = self._score_move_diversity(moves)

        # 5. Recency score
        scores.recency = self._score_recency(game)

        # 6. Source Elo score
        scores.source_elo = self._score_source_elo(game)

        # 7. Tactical content score
        scores.tactical_content = self._score_tactical_content(game, moves)

        # Calculate weighted total
        total = (
            scores.decisive_win * self.weights.decisive_win +
            scores.game_length * self.weights.game_length +
            scores.victory_type * self.weights.victory_type +
            scores.move_diversity * self.weights.move_diversity +
            scores.recency * self.weights.recency +
            scores.source_elo * self.weights.source_elo +
            scores.tactical_content * self.weights.tactical_content
        )

        return GameQuality(
            game_id=game_id,
            config_key=self.config_key,
            total_score=total,
            scores=scores,
            victory_type=victory_type,
            move_count=move_count,
            source_file=source_file,
            source_node=source_node,
        )

    def _extract_game_id(self, game: dict[str, Any]) -> str:
        """Extract or generate a game ID."""
        if "game_id" in game:
            return game["game_id"]
        # Generate from content hash
        content = json.dumps(game, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_victory_type(self, game: dict[str, Any]) -> VictoryType:
        """Extract victory type from game data."""
        vt_str = game.get("victory_type", "")
        if not vt_str:
            # Try termination reason
            term = game.get("termination_reason", "")
            if "timeout" in term.lower():
                return VictoryType.TIMEOUT
            if "stalemate" in term.lower():
                return VictoryType.STALEMATE
        return VictoryType.from_string(vt_str)

    def _extract_move_count(self, game: dict[str, Any]) -> int:
        """Extract move count from game data."""
        if "move_count" in game:
            return game["move_count"]
        if "moves" in game:
            return len(game["moves"])
        return 0

    def _score_decisive_win(self, game: dict[str, Any], victory_type: VictoryType) -> float:
        """Score based on whether the game had a decisive outcome."""
        winner = game.get("winner")

        if winner is None or winner <= 0:
            return 0.1  # No clear winner

        # Penalize timeout/stalemate wins
        if victory_type == VictoryType.TIMEOUT:
            return 0.2
        if victory_type in (VictoryType.STALEMATE, VictoryType.DRAW):
            return 0.4

        return 1.0

    def _score_game_length(self, move_count: int) -> float:
        """Score based on game length."""
        cfg = self.length_config

        if move_count < cfg.min_moves:
            return 0.2  # Too short

        if move_count > cfg.max_moves:
            return 0.2  # Too long (likely timeout)

        if cfg.ideal_min <= move_count <= cfg.ideal_max:
            return 1.0  # Optimal range

        # Linear interpolation for suboptimal ranges
        if move_count < cfg.ideal_min:
            return 0.5 + 0.5 * (move_count - cfg.min_moves) / (cfg.ideal_min - cfg.min_moves)
        else:
            return 0.5 + 0.5 * (cfg.max_moves - move_count) / (cfg.max_moves - cfg.ideal_max)

    def _score_move_diversity(self, moves: list[Any]) -> float:
        """Score based on move diversity (penalize repetitive play)."""
        if not moves:
            return 0.5

        # Convert moves to strings for comparison
        move_strs = [str(m) for m in moves]
        unique_moves = len(set(move_strs))
        diversity_ratio = unique_moves / len(move_strs)

        # Scale: 50% unique = 0.75, 100% unique = 1.0
        return min(1.0, 0.5 + diversity_ratio)

    def _score_recency(self, game: dict[str, Any]) -> float:
        """Score based on how recent the game is."""
        timestamp = game.get("timestamp") or game.get("created_at")

        if not timestamp:
            return 0.5  # Unknown recency

        try:
            if isinstance(timestamp, str):
                game_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, (int, float)):
                game_time = datetime.fromtimestamp(timestamp)
            else:
                return 0.5

            # Make timezone-aware comparison
            if game_time.tzinfo is None:
                game_time = game_time.replace(tzinfo=self.reference_time.tzinfo)

            age_hours = (self.reference_time - game_time).total_seconds() / 3600

            # Full score for <24h, decays to 0.3 over 7 days
            if age_hours <= 24:
                return 1.0
            elif age_hours >= 168:  # 7 days
                return 0.3
            else:
                return 1.0 - 0.7 * (age_hours - 24) / (168 - 24)

        except Exception as e:
            logger.debug(f"Failed to parse timestamp: {e}")
            return 0.5

    def _score_source_elo(self, game: dict[str, Any]) -> float:
        """Score based on the Elo of the model that generated the game."""
        elo = game.get("model_elo") or game.get("source_elo") or 1500

        # Normalize: 1200-2000 Elo range maps to 0.0-1.0
        return max(0.0, min(1.0, (elo - 1200) / 800))

    def _score_tactical_content(self, game: dict[str, Any], moves: list[Any]) -> float:
        """Score based on tactical complexity of the game."""
        score = 0.3  # Base score

        # Check for captures in move data
        moves_str = json.dumps(moves)
        if "capture" in moves_str.lower():
            score += 0.3

        # Check for diverse endgame (last 10 moves)
        if len(moves) > 10:
            last_moves = [str(m) for m in moves[-10:]]
            if len(set(last_moves)) > 7:
                score += 0.2

        # Check for comeback indicators
        if game.get("had_lead_change") or game.get("comeback"):
            score += 0.2

        return min(1.0, score)


class QualityFilter:
    """Filters games based on quality criteria."""

    def __init__(
        self,
        min_quality: float = 0.6,
        max_games: int | None = None,
        exclude_victory_types: set[VictoryType] | None = None,
        min_moves: int = 10,
        max_moves: int = 500,
        deduplicate: bool = True,
    ):
        """Initialize the filter.

        Args:
            min_quality: Minimum quality score to include (0-1)
            max_games: Maximum games to return
            exclude_victory_types: Victory types to exclude
            min_moves: Minimum game length
            max_moves: Maximum game length
            deduplicate: Remove duplicate games
        """
        self.min_quality = min_quality
        self.max_games = max_games
        self.exclude_victory_types = exclude_victory_types or {VictoryType.TIMEOUT}
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.deduplicate = deduplicate

    def filter(
        self,
        games: Iterator[tuple[GameQuality, dict[str, Any]]],
    ) -> list[tuple[GameQuality, dict[str, Any]]]:
        """Filter games based on quality criteria.

        Args:
            games: Iterator of (GameQuality, game_data) tuples

        Returns:
            Filtered and sorted list of (GameQuality, game_data) tuples
        """
        seen_ids: set[str] = set()
        filtered: list[tuple[GameQuality, dict[str, Any]]] = []

        for quality, game_data in games:
            # Check quality threshold
            if quality.total_score < self.min_quality:
                continue

            # Check victory type
            if quality.victory_type in self.exclude_victory_types:
                continue

            # Check move count
            if not (self.min_moves <= quality.move_count <= self.max_moves):
                continue

            # Deduplicate
            if self.deduplicate:
                if quality.game_id in seen_ids:
                    continue
                seen_ids.add(quality.game_id)

            filtered.append((quality, game_data))

            # Check max games
            if self.max_games and len(filtered) >= self.max_games * 2:
                # Keep 2x max_games for sorting, then trim
                break

        # Sort by quality (descending)
        filtered.sort(key=lambda x: x[0].total_score, reverse=True)

        # Trim to max_games
        if self.max_games:
            filtered = filtered[:self.max_games]

        return filtered


@dataclass
class QualityStats:
    """Statistics about data quality."""
    total_games: int = 0
    filtered_games: int = 0
    avg_quality: float = 0.0
    quality_distribution: dict[str, int] = field(default_factory=dict)
    victory_type_distribution: dict[str, int] = field(default_factory=dict)
    avg_move_count: float = 0.0
    source_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_games": self.total_games,
            "filtered_games": self.filtered_games,
            "avg_quality": self.avg_quality,
            "quality_distribution": self.quality_distribution,
            "victory_type_distribution": self.victory_type_distribution,
            "avg_move_count": self.avg_move_count,
            "source_distribution": self.source_distribution,
        }


def compute_quality_stats(qualities: list[GameQuality]) -> QualityStats:
    """Compute statistics from a list of quality assessments."""
    if not qualities:
        return QualityStats()

    stats = QualityStats(
        total_games=len(qualities),
        filtered_games=len(qualities),
    )

    # Quality distribution
    quality_scores = []
    move_counts = []

    for q in qualities:
        quality_scores.append(q.total_score)
        move_counts.append(q.move_count)

        # Quality bucket
        bucket = q.quality_bucket
        stats.quality_distribution[bucket] = stats.quality_distribution.get(bucket, 0) + 1

        # Victory type
        vt = q.victory_type.value
        stats.victory_type_distribution[vt] = stats.victory_type_distribution.get(vt, 0) + 1

        # Source
        if q.source_node:
            stats.source_distribution[q.source_node] = stats.source_distribution.get(q.source_node, 0) + 1

    stats.avg_quality = sum(quality_scores) / len(quality_scores)
    stats.avg_move_count = sum(move_counts) / len(move_counts)

    return stats
