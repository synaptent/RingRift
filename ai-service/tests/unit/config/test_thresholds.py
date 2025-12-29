"""Tests for app.config.thresholds module.

Tests the unified threshold constants used across training, evaluation,
promotion, and rollback systems.
"""

import pytest

from app.config.thresholds import (
    # Training thresholds
    TRAINING_TRIGGER_GAMES,
    TRAINING_MIN_INTERVAL_SECONDS,
    TRAINING_STALENESS_HOURS,
    TRAINING_BOOTSTRAP_GAMES,
    TRAINING_MAX_CONCURRENT,
    # Regression thresholds
    ELO_DROP_ROLLBACK,
    WIN_RATE_DROP_ROLLBACK,
    ERROR_RATE_ROLLBACK,
    MIN_GAMES_REGRESSION,
    CONSECUTIVE_REGRESSIONS_FORCE,
    # Promotion thresholds
    ELO_IMPROVEMENT_PROMOTE,
    MIN_GAMES_PROMOTE,
    MIN_WIN_RATE_PROMOTE,
    WIN_RATE_BEAT_BEST,
    PROMOTION_COOLDOWN_SECONDS,
    # Production thresholds
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC,
    PRODUCTION_MIN_WIN_RATE_VS_RANDOM,
    # ELO tiers
    ELO_TIER_NOVICE,
    ELO_TIER_INTERMEDIATE,
    ELO_TIER_ADVANCED,
)


class TestTrainingThresholds:
    """Tests for training-related thresholds."""

    def test_trigger_games_positive(self):
        """Test training trigger games is positive."""
        assert TRAINING_TRIGGER_GAMES > 0
        # Value can vary (50 for fast iteration, 500 for stable training)
        assert TRAINING_TRIGGER_GAMES >= 50

    def test_min_interval_reasonable(self):
        """Test minimum interval is reasonable (5 mins to 2 hours)."""
        assert TRAINING_MIN_INTERVAL_SECONDS >= 300  # At least 5 mins (fast iteration mode)
        assert TRAINING_MIN_INTERVAL_SECONDS <= 7200  # At most 2 hours

    def test_staleness_hours_positive(self):
        """Test staleness hours is positive and reasonable."""
        assert TRAINING_STALENESS_HOURS > 0
        assert TRAINING_STALENESS_HOURS <= 24  # At most 1 day

    def test_bootstrap_less_than_or_equal_trigger(self):
        """Test bootstrap threshold is less than or equal to regular trigger."""
        # Bootstrap can equal trigger in fast iteration mode
        assert TRAINING_BOOTSTRAP_GAMES <= TRAINING_TRIGGER_GAMES

    def test_max_concurrent_positive(self):
        """Test max concurrent is positive and reasonable."""
        assert TRAINING_MAX_CONCURRENT >= 1
        assert TRAINING_MAX_CONCURRENT <= 10


class TestRegressionThresholds:
    """Tests for regression and rollback thresholds."""

    def test_elo_drop_positive(self):
        """Test Elo drop threshold is positive."""
        assert ELO_DROP_ROLLBACK > 0
        assert ELO_DROP_ROLLBACK == 50

    def test_win_rate_drop_valid_percentage(self):
        """Test win rate drop is valid percentage."""
        assert 0 < WIN_RATE_DROP_ROLLBACK <= 1.0
        assert WIN_RATE_DROP_ROLLBACK == 0.10

    def test_error_rate_valid_percentage(self):
        """Test error rate is valid percentage."""
        assert 0 < ERROR_RATE_ROLLBACK <= 1.0
        assert ERROR_RATE_ROLLBACK == 0.05

    def test_min_games_regression_positive(self):
        """Test minimum games for regression is positive."""
        assert MIN_GAMES_REGRESSION > 0

    def test_consecutive_regressions_positive(self):
        """Test consecutive regressions threshold is positive."""
        assert CONSECUTIVE_REGRESSIONS_FORCE >= 2


class TestPromotionThresholds:
    """Tests for model promotion thresholds."""

    def test_elo_improvement_positive(self):
        """Test Elo improvement threshold is positive."""
        assert ELO_IMPROVEMENT_PROMOTE > 0
        assert ELO_IMPROVEMENT_PROMOTE == 20

    def test_min_games_promote_positive(self):
        """Test minimum games for promotion is positive."""
        assert MIN_GAMES_PROMOTE > 0
        assert MIN_GAMES_PROMOTE >= MIN_GAMES_REGRESSION

    def test_min_win_rate_valid(self):
        """Test minimum win rate is valid."""
        assert 0 < MIN_WIN_RATE_PROMOTE < 1.0
        assert MIN_WIN_RATE_PROMOTE == 0.45

    def test_win_rate_beat_best_above_half(self):
        """Test win rate to beat best is above 50%."""
        assert WIN_RATE_BEAT_BEST > 0.5
        assert WIN_RATE_BEAT_BEST == 0.55

    def test_cooldown_positive(self):
        """Test promotion cooldown is positive."""
        assert PROMOTION_COOLDOWN_SECONDS > 0


class TestProductionThresholds:
    """Tests for production promotion thresholds."""

    def test_production_elo_high(self):
        """Test production Elo threshold is high."""
        assert PRODUCTION_ELO_THRESHOLD > ELO_TIER_INTERMEDIATE
        assert PRODUCTION_ELO_THRESHOLD == 1650

    def test_production_min_games_sufficient(self):
        """Test production requires sufficient games."""
        assert PRODUCTION_MIN_GAMES >= MIN_GAMES_PROMOTE
        assert PRODUCTION_MIN_GAMES == 100

    def test_production_win_rates_valid(self):
        """Test production win rates are valid."""
        assert 0.5 < PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC < 1.0
        assert 0.5 < PRODUCTION_MIN_WIN_RATE_VS_RANDOM <= 1.0

    def test_random_win_rate_higher_than_heuristic(self):
        """Test win rate vs random is higher than vs heuristic."""
        # Should beat random more easily than heuristic
        assert PRODUCTION_MIN_WIN_RATE_VS_RANDOM > PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC


class TestEloTiers:
    """Tests for Elo tier definitions."""

    def test_tiers_ordered(self):
        """Test Elo tiers are in ascending order."""
        assert ELO_TIER_NOVICE < ELO_TIER_INTERMEDIATE
        assert ELO_TIER_INTERMEDIATE < ELO_TIER_ADVANCED

    def test_novice_below_heuristic(self):
        """Test novice tier is below heuristic level."""
        # Heuristic is around 1200
        assert ELO_TIER_NOVICE < 1200

    def test_intermediate_at_heuristic(self):
        """Test intermediate tier is at heuristic level."""
        assert ELO_TIER_INTERMEDIATE == 1200

    def test_advanced_above_heuristic(self):
        """Test advanced tier is above heuristic level."""
        assert ELO_TIER_ADVANCED > ELO_TIER_INTERMEDIATE


class TestThresholdRelationships:
    """Tests for relationships between thresholds."""

    def test_bootstrap_enables_quick_start(self):
        """Test bootstrap threshold enables quick start."""
        # Bootstrap should be much smaller than regular trigger
        assert TRAINING_BOOTSTRAP_GAMES <= TRAINING_TRIGGER_GAMES / 5

    def test_regression_before_promotion(self):
        """Test regression detection is possible before promotion."""
        # Should be able to detect regression before a model can be promoted
        assert MIN_GAMES_REGRESSION <= MIN_GAMES_PROMOTE

    def test_elo_thresholds_sensible(self):
        """Test Elo thresholds have sensible relationships."""
        # Improvement needed for promotion < drop that triggers rollback
        # (It should be easier to trigger a rollback than earn a promotion)
        assert ELO_IMPROVEMENT_PROMOTE <= ELO_DROP_ROLLBACK

    def test_production_gates_are_strict(self):
        """Test production gates are stricter than regular promotion."""
        assert PRODUCTION_MIN_GAMES >= MIN_GAMES_PROMOTE
        # Production Elo threshold should be above advanced tier
        assert PRODUCTION_ELO_THRESHOLD >= ELO_TIER_ADVANCED
