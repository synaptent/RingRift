"""Tests for app.quality.scorers.game_scorer module.

Tests the GameQualityScorer and related configuration classes.
"""

import pytest

from app.quality.scorers.game_scorer import (
    GameQualityScorer,
    GameScorerConfig,
    GameScorerWeights,
    get_game_quality_scorer,
    reset_game_quality_scorer,
)
from app.quality.types import QualityLevel


class TestGameScorerWeights:
    """Tests for GameScorerWeights dataclass."""

    def test_default_values(self):
        """Test default weight values."""
        weights = GameScorerWeights()
        assert weights.outcome_weight == 0.25
        assert weights.length_weight == 0.25
        assert weights.phase_balance_weight == 0.20
        assert weights.diversity_weight == 0.15
        assert weights.source_reputation_weight == 0.15

    def test_default_sum_to_one(self):
        """Test default weights sum to 1.0."""
        weights = GameScorerWeights()
        total = (
            weights.outcome_weight
            + weights.length_weight
            + weights.phase_balance_weight
            + weights.diversity_weight
            + weights.source_reputation_weight
        )
        assert total == pytest.approx(1.0)

    def test_game_length_defaults(self):
        """Test game length parameter defaults."""
        weights = GameScorerWeights()
        assert weights.min_game_length == 10
        assert weights.max_game_length == 200
        assert weights.optimal_game_length == 80

    def test_to_dict(self):
        """Test conversion to component weights dict."""
        weights = GameScorerWeights()
        d = weights.to_dict()
        assert d["outcome"] == 0.25
        assert d["length"] == 0.25
        assert d["phase_balance"] == 0.20
        assert d["diversity"] == 0.15
        assert d["source_reputation"] == 0.15


class TestGameScorerConfig:
    """Tests for GameScorerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GameScorerConfig()
        assert isinstance(config.weights, GameScorerWeights)
        assert config.cache_size == 100

    def test_custom_weights(self):
        """Test custom weights configuration."""
        custom_weights = GameScorerWeights(
            outcome_weight=0.4,
            length_weight=0.3,
            phase_balance_weight=0.1,
            diversity_weight=0.1,
            source_reputation_weight=0.1,
        )
        config = GameScorerConfig(weights=custom_weights)
        assert config.weights.outcome_weight == 0.4


class TestGameQualityScorerBasic:
    """Basic tests for GameQualityScorer."""

    def test_scorer_creation(self):
        """Test scorer can be created."""
        scorer = GameQualityScorer()
        assert scorer.SCORER_NAME == "game_quality"
        assert scorer.SCORER_VERSION == "1.1.0"

    def test_score_minimal_game(self):
        """Test scoring a minimal game."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "test-123",
            "total_moves": 50,
        })
        assert 0.0 <= result.score <= 1.0
        assert result.is_valid is True

    def test_score_full_game(self):
        """Test scoring a game with all fields."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "test-full",
            "total_moves": 80,  # Optimal length
            "winner": 1,
            "source": "selfplay",
            "phase_balance_score": 0.8,
            "diversity_score": 0.9,
        })
        assert result.score > 0.7  # Should be high quality
        assert result.level == QualityLevel.HIGH

    def test_validation_rejects_empty(self):
        """Test validation rejects empty data."""
        scorer = GameQualityScorer()
        result = scorer.score({})
        assert result.is_valid is False
        assert result.level == QualityLevel.BLOCKED


class TestGameQualityScorerOutcome:
    """Tests for outcome scoring."""

    def test_decisive_game_high_score(self):
        """Test decisive games get high outcome score."""
        scorer = GameQualityScorer()

        # Decisive game (winner=1)
        result_decisive = scorer.score({
            "game_id": "decisive",
            "total_moves": 80,
            "winner": 1,
        })

        # Draw game (winner=-1)
        result_draw = scorer.score({
            "game_id": "draw",
            "total_moves": 80,
            "winner": -1,
        })

        assert result_decisive.score > result_draw.score

    def test_winner_none_is_draw(self):
        """Test winner=None is treated as draw."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "none",
            "total_moves": 80,
            "winner": None,
        })
        # Should have lower outcome score than decisive
        components = result.components
        assert components["outcome"] < 1.0


class TestGameQualityScorerLength:
    """Tests for length scoring."""

    def test_optimal_length_max_score(self):
        """Test optimal length gets maximum score."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "optimal",
            "total_moves": 80,  # Default optimal
        })
        assert result.components["length"] == 1.0

    def test_too_short_zero_score(self):
        """Test games below min_length get zero."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "short",
            "total_moves": 5,  # Below min (10)
        })
        assert result.components["length"] == 0.0

    def test_too_long_reduced_score(self):
        """Test very long games get reduced score."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "long",
            "total_moves": 250,  # Above max (200)
        })
        assert result.components["length"] == 0.8

    def test_medium_length_partial_score(self):
        """Test medium length games get partial score."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "medium",
            "total_moves": 40,  # Between min and optimal
        })
        assert 0.5 < result.components["length"] < 1.0


class TestGameQualityScorerSource:
    """Tests for source reputation scoring."""

    def test_with_source_higher_score(self):
        """Test games with source get higher reputation score."""
        scorer = GameQualityScorer()

        with_source = scorer.score({
            "game_id": "sourced",
            "total_moves": 50,
            "source": "selfplay",
        })

        without_source = scorer.score({
            "game_id": "nosource",
            "total_moves": 50,
        })

        assert with_source.components["source_reputation"] > without_source.components["source_reputation"]


class TestGameQualityScorerMetadata:
    """Tests for score_with_metadata method."""

    def test_returns_extended_metadata(self):
        """Test score_with_metadata returns game metadata."""
        scorer = GameQualityScorer()
        result, metadata = scorer.score_with_metadata({
            "game_id": "meta-test",
            "total_moves": 60,
            "winner": 2,
            "board_type": "hex8",
            "num_players": 4,
        })

        assert result.score > 0
        assert metadata["game_id"] == "meta-test"
        assert metadata["game_length"] == 60
        assert metadata["is_decisive"] is True
        assert metadata["board_type"] == "hex8"
        assert metadata["num_players"] == 4


class TestGameQualityScorerTrainingWeight:
    """Tests for training weight computation."""

    def test_training_weight_basic(self):
        """Test basic training weight computation."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "train",
            "total_moves": 80,
            "winner": 1,
        })

        weight = scorer.compute_training_weight(result)
        assert weight > 0.0

    def test_recency_decay(self):
        """Test recency affects training weight."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "decay",
            "total_moves": 50,
        })

        fresh = scorer.compute_training_weight(result, recency_hours=0.0)
        old = scorer.compute_training_weight(result, recency_hours=48.0)

        assert fresh > old

    def test_priority_boost(self):
        """Test priority affects training weight."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "priority",
            "total_moves": 50,
        })

        low_priority = scorer.compute_training_weight(result, base_priority=0.5)
        high_priority = scorer.compute_training_weight(result, base_priority=2.0)

        assert high_priority > low_priority


class TestGameQualityScorerSyncPriority:
    """Tests for sync priority computation."""

    def test_sync_priority_basic(self):
        """Test basic sync priority computation."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "sync",
            "total_moves": 80,
        })

        priority = scorer.compute_sync_priority(result)
        assert priority >= 0.0

    def test_decisive_higher_priority(self):
        """Test decisive games have higher sync priority."""
        scorer = GameQualityScorer()
        result = scorer.score({
            "game_id": "sync-decisive",
            "total_moves": 80,
        })

        decisive_priority = scorer.compute_sync_priority(
            result, is_decisive=True
        )
        draw_priority = scorer.compute_sync_priority(
            result, is_decisive=False
        )

        assert decisive_priority > draw_priority


class TestGameQualityScorerElo:
    """Tests for Elo-based scoring."""

    def test_elo_lookup_integration(self):
        """Test Elo lookup is used when provided."""
        def mock_lookup(model_id: str) -> float:
            return 2100.0  # High Elo (above midpoint of 1800)

        scorer = GameQualityScorer(elo_lookup=mock_lookup)
        _, metadata = scorer.score_with_metadata({
            "game_id": "elo-test",
            "total_moves": 50,
            "model_version": "v1",
        })

        # Elo score should be above neutral (0.5)
        # With min=1200, max=2400: (2100-1200)/(2400-1200) = 0.75
        assert metadata["elo_score"] == pytest.approx(0.75)

    def test_elo_lookup_failure_handled(self):
        """Test Elo lookup failures are handled gracefully."""
        def failing_lookup(model_id: str) -> float:
            raise ValueError("Lookup failed")

        scorer = GameQualityScorer(elo_lookup=failing_lookup)
        _, metadata = scorer.score_with_metadata({
            "game_id": "fail",
            "total_moves": 50,
            "model_version": "v1",
        })

        # Should fall back to neutral score
        assert metadata["elo_score"] == 0.5

    def test_set_elo_lookup(self):
        """Test setting Elo lookup after creation."""
        scorer = GameQualityScorer()
        assert scorer.elo_lookup is None

        def mock_lookup(model_id: str) -> float:
            return 1600.0

        scorer.set_elo_lookup(mock_lookup)
        assert scorer.elo_lookup is not None


class TestGameQualityScorerSingleton:
    """Tests for singleton access."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_game_quality_scorer()

    def test_get_singleton(self):
        """Test singleton accessor."""
        scorer1 = get_game_quality_scorer()
        scorer2 = get_game_quality_scorer()
        assert scorer1 is scorer2

    def test_singleton_accepts_config_first_call(self):
        """Test config is used on first call."""
        custom_weights = GameScorerWeights(optimal_game_length=100)
        config = GameScorerConfig(weights=custom_weights)

        scorer = get_game_quality_scorer(config=config)
        assert scorer._game_weights.optimal_game_length == 100

    def test_reset_singleton(self):
        """Test singleton can be reset."""
        scorer1 = get_game_quality_scorer()
        reset_game_quality_scorer()
        scorer2 = get_game_quality_scorer()

        assert scorer1 is not scorer2


class TestGameQualityScorerCustomWeights:
    """Tests for custom weight configurations."""

    def test_custom_outcome_weight(self):
        """Test custom outcome weight affects scoring."""
        # High outcome weight
        high_outcome = GameScorerWeights(
            outcome_weight=0.8,
            length_weight=0.05,
            phase_balance_weight=0.05,
            diversity_weight=0.05,
            source_reputation_weight=0.05,
        )
        scorer_high = GameQualityScorer(config=GameScorerConfig(weights=high_outcome))

        # Low outcome weight
        low_outcome = GameScorerWeights(
            outcome_weight=0.05,
            length_weight=0.8,
            phase_balance_weight=0.05,
            diversity_weight=0.05,
            source_reputation_weight=0.05,
        )
        scorer_low = GameQualityScorer(config=GameScorerConfig(weights=low_outcome))

        # Decisive game should benefit from high outcome weight
        game_data = {
            "game_id": "custom",
            "total_moves": 50,  # Not optimal length
            "winner": 1,  # Decisive
        }

        result_high = scorer_high.score(game_data)
        result_low = scorer_low.score(game_data)

        # With high outcome weight, decisive game should score higher
        assert result_high.score > result_low.score
