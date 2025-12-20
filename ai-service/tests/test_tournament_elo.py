"""Tests for the tournament Elo rating module.

Comprehensive tests for:
- EloRating dataclass and properties
- EloCalculator operations
- Expected score calculations
- Rating updates after games
"""

import math
import pytest


class TestEloRating:
    """Test EloRating dataclass."""

    def test_default_values(self):
        """Test EloRating default values."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="test_agent")

        assert rating.agent_id == "test_agent"
        assert rating.rating == 1500.0
        assert rating.games_played == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0

    def test_custom_values(self):
        """Test EloRating with custom values."""
        from app.tournament.elo import EloRating

        rating = EloRating(
            agent_id="custom_agent",
            rating=1650.0,
            games_played=50,
            wins=30,
            losses=15,
            draws=5,
        )

        assert rating.agent_id == "custom_agent"
        assert rating.rating == 1650.0
        assert rating.games_played == 50
        assert rating.wins == 30

    def test_win_rate_zero_games(self):
        """Test win rate with no games played."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="new_agent")

        assert rating.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        from app.tournament.elo import EloRating

        rating = EloRating(
            agent_id="agent",
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
        )

        assert rating.win_rate == 0.6

    def test_expected_score_no_games(self):
        """Test expected score with no games."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="new_agent")

        assert rating.expected_score == 0.5

    def test_expected_score_calculation(self):
        """Test expected score calculation with draws."""
        from app.tournament.elo import EloRating

        rating = EloRating(
            agent_id="agent",
            games_played=100,
            wins=50,
            losses=40,
            draws=10,
        )

        # Expected score = (wins + 0.5 * draws) / games
        expected = (50 + 0.5 * 10) / 100
        assert rating.expected_score == expected

    def test_rating_deviation_new_player(self):
        """Test rating deviation for new player."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="new_agent")

        assert rating.rating_deviation == rating.INITIAL_RD

    def test_rating_deviation_decreases(self):
        """Test rating deviation decreases with games played."""
        from app.tournament.elo import EloRating

        new_rating = EloRating(agent_id="new", games_played=0)
        mid_rating = EloRating(agent_id="mid", games_played=50)
        exp_rating = EloRating(agent_id="exp", games_played=200)

        assert new_rating.rating_deviation > mid_rating.rating_deviation
        assert mid_rating.rating_deviation > exp_rating.rating_deviation

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="agent", rating=1600, games_played=50)
        lower, upper = rating.confidence_interval(0.95)

        assert lower < 1600
        assert upper > 1600
        assert upper - lower > 0

    def test_ci_95_property(self):
        """Test ci_95 convenience property."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="agent", rating=1600, games_played=50)
        ci = rating.ci_95

        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_uncertainty_str(self):
        """Test uncertainty string format."""
        from app.tournament.elo import EloRating

        rating = EloRating(agent_id="agent", rating=1600, games_played=50)
        unc_str = rating.uncertainty_str

        assert "1600" in unc_str
        assert "±" in unc_str

    def test_to_dict(self):
        """Test EloRating serialization."""
        from app.tournament.elo import EloRating

        rating = EloRating(
            agent_id="agent",
            rating=1650.0,
            games_played=100,
            wins=60,
        )
        d = rating.to_dict()

        assert d["agent_id"] == "agent"
        assert d["rating"] == 1650.0
        assert d["games_played"] == 100
        assert "win_rate" in d
        assert "rating_deviation" in d
        assert "ci_95_lower" in d
        assert "ci_95_upper" in d


class TestEloCalculator:
    """Test EloCalculator class."""

    def test_calculator_creation(self):
        """Test EloCalculator instantiation."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        assert calc is not None
        assert calc.initial_rating == 1500.0
        assert calc.k_factor == 32.0

    def test_custom_parameters(self):
        """Test EloCalculator with custom parameters."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator(
            initial_rating=1200.0,
            k_factor=24.0,
            k_factor_provisional=48.0,
        )

        assert calc.initial_rating == 1200.0
        assert calc.k_factor == 24.0
        assert calc.k_factor_provisional == 48.0

    def test_get_rating_new_agent(self):
        """Test getting rating for new agent."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()
        rating = calc.get_rating("new_agent")

        assert rating.agent_id == "new_agent"
        assert rating.rating == 1500.0
        assert rating.games_played == 0

    def test_get_rating_caches(self):
        """Test that get_rating returns same object."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()
        rating1 = calc.get_rating("agent")
        rating2 = calc.get_rating("agent")

        assert rating1 is rating2

    def test_k_factor_provisional(self):
        """Test K-factor for provisional player."""
        from app.tournament.elo import EloCalculator, EloRating

        calc = EloCalculator(provisional_games=30)
        rating = EloRating(agent_id="new", rating=1500, games_played=10)

        k = calc.get_k_factor(rating)

        assert k == calc.k_factor_provisional

    def test_k_factor_established(self):
        """Test K-factor for established player."""
        from app.tournament.elo import EloCalculator, EloRating

        calc = EloCalculator(provisional_games=30)
        rating = EloRating(agent_id="exp", rating=1800, games_played=50)

        k = calc.get_k_factor(rating)

        assert k == calc.k_factor

    def test_k_factor_high_rated(self):
        """Test K-factor for high-rated player."""
        from app.tournament.elo import EloCalculator, EloRating

        calc = EloCalculator(
            high_rated_threshold=2400.0,
            k_factor_high_rated=16.0,
        )
        rating = EloRating(agent_id="master", rating=2500, games_played=100)

        k = calc.get_k_factor(rating)

        assert k == calc.k_factor_high_rated


class TestExpectedScore:
    """Test expected score calculations."""

    def test_expected_score_equal_ratings(self):
        """Test expected score for equal ratings."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        # Equal ratings should give 0.5 expected score
        expected = calc.expected_score(1500.0, 1500.0)

        assert expected == pytest.approx(0.5, abs=0.001)

    def test_expected_score_higher_rated(self):
        """Test expected score when player A is higher rated."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        # Higher rated player should have >0.5 expected score
        expected = calc.expected_score(1700.0, 1500.0)

        assert expected > 0.5

    def test_expected_score_lower_rated(self):
        """Test expected score when player A is lower rated."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        # Lower rated player should have <0.5 expected score
        expected = calc.expected_score(1300.0, 1500.0)

        assert expected < 0.5

    def test_expected_scores_sum_to_one(self):
        """Test that expected scores for both players sum to 1."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        e_a = calc.expected_score(1600.0, 1450.0)
        e_b = calc.expected_score(1450.0, 1600.0)

        assert e_a + e_b == pytest.approx(1.0, abs=0.001)


class TestLeaderboard:
    """Test leaderboard functionality."""

    def test_get_leaderboard_empty(self):
        """Test leaderboard with no ratings."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()
        leaderboard = calc.get_leaderboard()

        assert isinstance(leaderboard, list)
        assert len(leaderboard) == 0

    def test_get_leaderboard_sorted(self):
        """Test leaderboard is sorted by rating."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()

        # Add ratings with different values
        calc.get_rating("low").rating = 1300
        calc.get_rating("mid").rating = 1500
        calc.get_rating("high").rating = 1700

        leaderboard = calc.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0].agent_id == "high"
        assert leaderboard[1].agent_id == "mid"
        assert leaderboard[2].agent_id == "low"

    def test_get_all_ratings(self):
        """Test getting all ratings as dict."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()
        calc.get_rating("a")
        calc.get_rating("b")

        all_ratings = calc.get_all_ratings()

        assert "a" in all_ratings
        assert "b" in all_ratings

    def test_reset(self):
        """Test resetting calculator."""
        from app.tournament.elo import EloCalculator

        calc = EloCalculator()
        calc.get_rating("a")
        calc.get_rating("b")

        calc.reset()

        assert len(calc.get_all_ratings()) == 0


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_main_exports(self):
        """Test importing main exports from elo module."""
        from app.tournament.elo import (
            EloRating,
            EloCalculator,
        )

        assert EloRating is not None
        assert EloCalculator is not None
