"""Tests for scripts/lib/data_quality.py module.

Tests cover:
- VictoryType enum and parsing
- GameLengthConfig for different board types
- GameQualityScorer scoring functions
- QualityFilter filtering logic
- QualityStats computation
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import pytest

from scripts.lib.data_quality import (
    VictoryType,
    VICTORY_TYPE_VALUE,
    GameLengthConfig,
    QualityScores,
    GameQuality,
    QualityWeights,
    GameQualityScorer,
    QualityFilter,
    QualityStats,
    compute_quality_stats,
)


class TestVictoryType:
    """Tests for VictoryType enum."""

    def test_from_string_ring(self):
        """Test parsing 'ring' victory type."""
        assert VictoryType.from_string("ring") == VictoryType.RING

    def test_from_string_elimination(self):
        """Test parsing 'elimination' victory type."""
        assert VictoryType.from_string("elimination") == VictoryType.ELIMINATION

    def test_from_string_timeout(self):
        """Test parsing 'timeout' victory type."""
        assert VictoryType.from_string("timeout") == VictoryType.TIMEOUT

    def test_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert VictoryType.from_string("RING") == VictoryType.RING
        assert VictoryType.from_string("Ring") == VictoryType.RING

    def test_from_string_with_whitespace(self):
        """Test parsing with whitespace."""
        assert VictoryType.from_string("  ring  ") == VictoryType.RING

    def test_from_string_unknown(self):
        """Test unknown victory type."""
        assert VictoryType.from_string("foobar") == VictoryType.UNKNOWN

    def test_victory_type_values(self):
        """Test that all victory types have training values."""
        for vt in VictoryType:
            assert vt in VICTORY_TYPE_VALUE


class TestGameLengthConfig:
    """Tests for GameLengthConfig."""

    def test_square8_2p_config(self):
        """Test config for square8_2p."""
        config = GameLengthConfig.for_config("square8_2p")
        assert config.min_moves == 15
        assert config.max_moves == 200
        assert config.ideal_min <= config.ideal_max

    def test_hex8_2p_config(self):
        """Test config for hex8_2p."""
        config = GameLengthConfig.for_config("hex8_2p")
        assert config.min_moves == 10
        assert config.max_moves == 150

    def test_square19_2p_config(self):
        """Test config for larger board."""
        config = GameLengthConfig.for_config("square19_2p")
        assert config.max_moves > 200  # Larger board, longer games

    def test_unknown_config(self):
        """Test fallback for unknown config."""
        config = GameLengthConfig.for_config("unknown_board")
        assert config.min_moves > 0
        assert config.max_moves > config.min_moves


class TestQualityScores:
    """Tests for QualityScores dataclass."""

    def test_default_values(self):
        """Test default values are all zero."""
        scores = QualityScores()
        assert scores.decisive_win == 0.0
        assert scores.game_length == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        scores = QualityScores(decisive_win=0.8, game_length=0.9)
        d = scores.to_dict()
        assert d["decisive_win"] == 0.8
        assert d["game_length"] == 0.9


class TestQualityWeights:
    """Tests for QualityWeights."""

    def test_default_weights(self):
        """Test default weights are reasonable."""
        weights = QualityWeights()
        total = (
            weights.decisive_win + weights.game_length + weights.victory_type +
            weights.move_diversity + weights.recency + weights.source_elo +
            weights.tactical_content
        )
        assert abs(total - 1.0) < 0.01  # Should sum to ~1.0

    def test_normalize(self):
        """Test weight normalization."""
        weights = QualityWeights(
            decisive_win=2.0,
            game_length=2.0,
            victory_type=1.0,
            move_diversity=1.0,
            recency=1.0,
            source_elo=1.0,
            tactical_content=2.0,
        )
        normalized = weights.normalize()
        total = (
            normalized.decisive_win + normalized.game_length + normalized.victory_type +
            normalized.move_diversity + normalized.recency + normalized.source_elo +
            normalized.tactical_content
        )
        assert abs(total - 1.0) < 0.001


class TestGameQualityScorer:
    """Tests for GameQualityScorer."""

    def create_game(self, **kwargs) -> dict[str, Any]:
        """Helper to create a game dictionary."""
        base = {
            "game_id": "test-game-001",
            "board_type": "square8",
            "num_players": 2,
            "winner": 1,
            "victory_type": "ring",
            "moves": ["a1", "b2", "c3", "d4", "e5"] * 10,  # 50 moves
        }
        base.update(kwargs)
        return base

    def test_score_basic_game(self):
        """Test scoring a basic game."""
        scorer = GameQualityScorer(config_key="square8_2p")
        game = self.create_game()

        quality = scorer.score(game)

        assert quality.game_id == "test-game-001"
        assert quality.config_key == "square8_2p"
        assert 0.0 <= quality.total_score <= 1.0
        assert quality.move_count == 50

    def test_score_ring_victory(self):
        """Test that ring victories score highly."""
        scorer = GameQualityScorer(config_key="square8_2p")

        ring_game = self.create_game(victory_type="ring")
        timeout_game = self.create_game(victory_type="timeout")

        ring_quality = scorer.score(ring_game)
        timeout_quality = scorer.score(timeout_game)

        assert ring_quality.total_score > timeout_quality.total_score

    def test_score_game_length(self):
        """Test game length scoring."""
        scorer = GameQualityScorer(config_key="square8_2p")

        short_game = self.create_game(moves=["a1"] * 5)  # Too short
        ideal_game = self.create_game(moves=["a1"] * 50)  # Ideal
        long_game = self.create_game(moves=["a1"] * 250)  # Too long

        short_quality = scorer.score(short_game)
        ideal_quality = scorer.score(ideal_game)
        long_quality = scorer.score(long_game)

        assert ideal_quality.scores.game_length > short_quality.scores.game_length
        assert ideal_quality.scores.game_length > long_quality.scores.game_length

    def test_score_decisive_win(self):
        """Test decisive win scoring."""
        scorer = GameQualityScorer(config_key="square8_2p")

        decisive = self.create_game(winner=1, victory_type="ring")
        no_winner = self.create_game(winner=None, victory_type="draw")

        decisive_quality = scorer.score(decisive)
        no_winner_quality = scorer.score(no_winner)

        assert decisive_quality.scores.decisive_win > no_winner_quality.scores.decisive_win

    def test_score_move_diversity(self):
        """Test move diversity scoring."""
        scorer = GameQualityScorer(config_key="square8_2p")

        diverse = self.create_game(
            moves=["a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8"] * 5
        )
        repetitive = self.create_game(moves=["a1", "a2"] * 20)

        diverse_quality = scorer.score(diverse)
        repetitive_quality = scorer.score(repetitive)

        assert diverse_quality.scores.move_diversity >= repetitive_quality.scores.move_diversity

    def test_score_recency(self):
        """Test recency scoring."""
        now = datetime.now()
        scorer = GameQualityScorer(config_key="square8_2p", reference_time=now)

        recent = self.create_game(timestamp=now.isoformat())
        old = self.create_game(
            timestamp=(now - timedelta(days=10)).isoformat()
        )

        recent_quality = scorer.score(recent)
        old_quality = scorer.score(old)

        assert recent_quality.scores.recency > old_quality.scores.recency

    def test_score_source_elo(self):
        """Test source Elo scoring."""
        scorer = GameQualityScorer(config_key="square8_2p")

        high_elo = self.create_game(model_elo=1900)
        low_elo = self.create_game(model_elo=1100)

        high_quality = scorer.score(high_elo)
        low_quality = scorer.score(low_elo)

        assert high_quality.scores.source_elo > low_quality.scores.source_elo

    def test_extract_game_id_generates_hash(self):
        """Test game ID generation when not provided."""
        scorer = GameQualityScorer(config_key="square8_2p")
        game = self.create_game()
        del game["game_id"]

        quality = scorer.score(game)

        assert len(quality.game_id) == 12  # MD5 hash prefix

    def test_source_file_and_node(self):
        """Test source file and node are preserved."""
        scorer = GameQualityScorer(config_key="square8_2p")
        game = self.create_game()

        quality = scorer.score(
            game,
            source_file="/path/to/games.jsonl",
            source_node="node-01",
        )

        assert quality.source_file == "/path/to/games.jsonl"
        assert quality.source_node == "node-01"


class TestGameQuality:
    """Tests for GameQuality dataclass."""

    def test_quality_bucket(self):
        """Test quality bucket calculation."""
        quality = GameQuality(
            game_id="test",
            config_key="square8_2p",
            total_score=0.85,
            scores=QualityScores(),
            victory_type=VictoryType.RING,
            move_count=50,
        )
        assert quality.quality_bucket == "0.8"

    def test_quality_bucket_edge_cases(self):
        """Test quality bucket edge cases."""
        quality_low = GameQuality(
            game_id="test",
            config_key="square8_2p",
            total_score=0.1,
            scores=QualityScores(),
            victory_type=VictoryType.RING,
            move_count=50,
        )
        quality_high = GameQuality(
            game_id="test",
            config_key="square8_2p",
            total_score=0.99,
            scores=QualityScores(),
            victory_type=VictoryType.RING,
            move_count=50,
        )
        assert quality_low.quality_bucket == "0.1"
        assert quality_high.quality_bucket == "0.9"


class TestQualityFilter:
    """Tests for QualityFilter."""

    def create_quality(
        self,
        game_id: str,
        score: float,
        victory_type: VictoryType = VictoryType.RING,
        move_count: int = 50,
    ) -> GameQuality:
        """Helper to create GameQuality."""
        return GameQuality(
            game_id=game_id,
            config_key="square8_2p",
            total_score=score,
            scores=QualityScores(),
            victory_type=victory_type,
            move_count=move_count,
        )

    def test_filter_by_min_quality(self):
        """Test filtering by minimum quality."""
        filter = QualityFilter(min_quality=0.7)

        games = [
            (self.create_quality("g1", 0.9), {}),
            (self.create_quality("g2", 0.5), {}),
            (self.create_quality("g3", 0.8), {}),
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 2
        assert all(q.total_score >= 0.7 for q, _ in filtered)

    def test_filter_by_victory_type(self):
        """Test excluding certain victory types."""
        filter = QualityFilter(
            min_quality=0.0,
            exclude_victory_types={VictoryType.TIMEOUT},
        )

        games = [
            (self.create_quality("g1", 0.9, VictoryType.RING), {}),
            (self.create_quality("g2", 0.9, VictoryType.TIMEOUT), {}),
            (self.create_quality("g3", 0.9, VictoryType.ELIMINATION), {}),
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 2
        assert all(q.victory_type != VictoryType.TIMEOUT for q, _ in filtered)

    def test_filter_by_move_count(self):
        """Test filtering by move count."""
        filter = QualityFilter(
            min_quality=0.0,
            min_moves=20,
            max_moves=100,
        )

        games = [
            (self.create_quality("g1", 0.9, move_count=50), {}),
            (self.create_quality("g2", 0.9, move_count=10), {}),  # Too short
            (self.create_quality("g3", 0.9, move_count=150), {}),  # Too long
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 1
        assert filtered[0][0].game_id == "g1"

    def test_filter_deduplicates(self):
        """Test deduplication."""
        filter = QualityFilter(min_quality=0.0, deduplicate=True)

        games = [
            (self.create_quality("g1", 0.9), {}),
            (self.create_quality("g1", 0.8), {}),  # Duplicate ID
            (self.create_quality("g2", 0.7), {}),
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 2

    def test_filter_no_deduplication(self):
        """Test without deduplication."""
        filter = QualityFilter(min_quality=0.0, deduplicate=False)

        games = [
            (self.create_quality("g1", 0.9), {}),
            (self.create_quality("g1", 0.8), {}),
            (self.create_quality("g2", 0.7), {}),
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 3

    def test_filter_max_games(self):
        """Test max games limit."""
        filter = QualityFilter(min_quality=0.0, max_games=2)

        games = [
            (self.create_quality("g1", 0.9), {}),
            (self.create_quality("g2", 0.8), {}),
            (self.create_quality("g3", 0.7), {}),
            (self.create_quality("g4", 0.6), {}),
        ]

        filtered = filter.filter(iter(games))

        assert len(filtered) == 2

    def test_filter_sorts_by_quality(self):
        """Test that results are sorted by quality."""
        filter = QualityFilter(min_quality=0.0)

        games = [
            (self.create_quality("g1", 0.5), {}),
            (self.create_quality("g2", 0.9), {}),
            (self.create_quality("g3", 0.7), {}),
        ]

        filtered = filter.filter(iter(games))

        assert filtered[0][0].game_id == "g2"
        assert filtered[1][0].game_id == "g3"
        assert filtered[2][0].game_id == "g1"


class TestQualityStats:
    """Tests for QualityStats and compute_quality_stats."""

    def create_quality(
        self,
        score: float,
        victory_type: VictoryType = VictoryType.RING,
        move_count: int = 50,
        source_node: str = None,
    ) -> GameQuality:
        """Helper to create GameQuality."""
        return GameQuality(
            game_id=f"game-{score}",
            config_key="square8_2p",
            total_score=score,
            scores=QualityScores(),
            victory_type=victory_type,
            move_count=move_count,
            source_node=source_node,
        )

    def test_compute_stats_empty(self):
        """Test stats with no games."""
        stats = compute_quality_stats([])

        assert stats.total_games == 0
        assert stats.avg_quality == 0.0

    def test_compute_stats_basic(self):
        """Test basic stats computation."""
        qualities = [
            self.create_quality(0.8, VictoryType.RING, 50),
            self.create_quality(0.6, VictoryType.ELIMINATION, 60),
        ]

        stats = compute_quality_stats(qualities)

        assert stats.total_games == 2
        assert stats.avg_quality == 0.7
        assert stats.avg_move_count == 55

    def test_compute_stats_quality_distribution(self):
        """Test quality distribution."""
        qualities = [
            self.create_quality(0.81),
            self.create_quality(0.85),
            self.create_quality(0.92),
        ]

        stats = compute_quality_stats(qualities)

        assert stats.quality_distribution.get("0.8", 0) == 2
        assert stats.quality_distribution.get("0.9", 0) == 1

    def test_compute_stats_victory_distribution(self):
        """Test victory type distribution."""
        qualities = [
            self.create_quality(0.8, VictoryType.RING),
            self.create_quality(0.8, VictoryType.RING),
            self.create_quality(0.8, VictoryType.ELIMINATION),
        ]

        stats = compute_quality_stats(qualities)

        assert stats.victory_type_distribution.get("ring", 0) == 2
        assert stats.victory_type_distribution.get("elimination", 0) == 1

    def test_compute_stats_source_distribution(self):
        """Test source node distribution."""
        qualities = [
            self.create_quality(0.8, source_node="node-01"),
            self.create_quality(0.8, source_node="node-01"),
            self.create_quality(0.8, source_node="node-02"),
        ]

        stats = compute_quality_stats(qualities)

        assert stats.source_distribution.get("node-01", 0) == 2
        assert stats.source_distribution.get("node-02", 0) == 1

    def test_stats_to_dict(self):
        """Test stats conversion to dict."""
        stats = QualityStats(
            total_games=10,
            filtered_games=8,
            avg_quality=0.75,
        )

        d = stats.to_dict()

        assert d["total_games"] == 10
        assert d["filtered_games"] == 8
        assert d["avg_quality"] == 0.75
