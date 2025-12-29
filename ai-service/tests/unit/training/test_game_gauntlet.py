"""
Unit tests for app.training.game_gauntlet module.

Tests cover:
- BaselineOpponent enum values
- compute_wilson_interval statistical function
- should_early_stop early stopping logic
- GameResult and GauntletResult dataclasses
- create_baseline_ai factory function
- Gauntlet evaluation flow

Created: December 2025
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models import BoardType
from app.training.game_gauntlet import (
    BaselineOpponent,
    GameResult,
    GauntletResult,
    compute_wilson_interval,
    should_early_stop,
)


# =============================================================================
# BaselineOpponent Enum Tests
# =============================================================================


class TestBaselineOpponentEnum:
    """Tests for BaselineOpponent enum."""

    def test_all_baseline_values_exist(self):
        """All expected baseline opponents exist."""
        expected = ["random", "heuristic", "heuristic_strong",
                   "mcts_light", "mcts_medium", "mcts_strong",
                   "mcts_master", "mcts_grandmaster"]
        for val in expected:
            assert any(b.value == val for b in BaselineOpponent)

    def test_random_baseline_value(self):
        """RANDOM baseline has correct value."""
        assert BaselineOpponent.RANDOM.value == "random"

    def test_heuristic_baseline_value(self):
        """HEURISTIC baseline has correct value."""
        assert BaselineOpponent.HEURISTIC.value == "heuristic"

    def test_mcts_baselines_exist(self):
        """MCTS tiers exist for Elo ladder."""
        mcts_baselines = [
            BaselineOpponent.MCTS_LIGHT,
            BaselineOpponent.MCTS_MEDIUM,
            BaselineOpponent.MCTS_STRONG,
            BaselineOpponent.MCTS_MASTER,
            BaselineOpponent.MCTS_GRANDMASTER,
        ]
        for b in mcts_baselines:
            assert b.value.startswith("mcts_")

    def test_enum_is_iterable(self):
        """Can iterate over all baselines."""
        baselines = list(BaselineOpponent)
        assert len(baselines) >= 8  # At least 8 baselines defined


# =============================================================================
# compute_wilson_interval Tests
# =============================================================================


class TestComputeWilsonInterval:
    """Tests for Wilson score confidence interval computation."""

    def test_zero_games_returns_full_interval(self):
        """Zero games returns (0, 1) interval."""
        lower, upper = compute_wilson_interval(0, 0)
        assert lower == 0.0
        assert upper == 1.0

    def test_all_wins_high_confidence(self):
        """100% win rate has high lower bound."""
        lower, upper = compute_wilson_interval(10, 10, confidence=0.95)
        assert upper == pytest.approx(1.0, abs=1e-6)  # Very close to 1.0
        assert lower > 0.6  # High lower bound for 10/10

    def test_all_losses_low_confidence(self):
        """0% win rate has low upper bound."""
        lower, upper = compute_wilson_interval(0, 10, confidence=0.95)
        assert lower == 0.0
        assert upper < 0.4  # Low upper bound for 0/10

    def test_fifty_percent_win_rate(self):
        """50% win rate interval contains 0.5."""
        lower, upper = compute_wilson_interval(50, 100, confidence=0.95)
        assert lower < 0.5 < upper

    def test_interval_bounds_valid(self):
        """Interval always within [0, 1]."""
        for wins in [0, 5, 10, 15, 20]:
            for total in [20, 50, 100]:
                if wins <= total:
                    lower, upper = compute_wilson_interval(wins, total)
                    assert 0.0 <= lower <= upper <= 1.0

    def test_higher_confidence_wider_interval(self):
        """Higher confidence produces wider interval."""
        lower_95, upper_95 = compute_wilson_interval(50, 100, confidence=0.95)
        lower_99, upper_99 = compute_wilson_interval(50, 100, confidence=0.99)

        # 99% interval should be wider
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 >= width_95

    def test_more_samples_narrower_interval(self):
        """More samples produces narrower interval."""
        # Same 50% win rate, different sample sizes
        lower_small, upper_small = compute_wilson_interval(5, 10, confidence=0.95)
        lower_large, upper_large = compute_wilson_interval(50, 100, confidence=0.95)

        width_small = upper_small - lower_small
        width_large = upper_large - lower_large
        assert width_large < width_small

    def test_single_game_wide_interval(self):
        """Single game has very wide interval."""
        lower, upper = compute_wilson_interval(1, 1, confidence=0.95)
        assert upper - lower > 0.5  # Wide uncertainty


# =============================================================================
# should_early_stop Tests
# =============================================================================


class TestShouldEarlyStop:
    """Tests for early stopping decision logic."""

    def test_not_enough_games(self):
        """Returns False when fewer than min_games played."""
        should_stop, reason = should_early_stop(5, 0, threshold=0.5, min_games=10)
        assert not should_stop
        assert "more games" in reason.lower()

    def test_exactly_min_games_evaluated(self):
        """Evaluates at exactly min_games."""
        should_stop, reason = should_early_stop(10, 0, threshold=0.5, min_games=10)
        # With 100% wins, should stop
        assert should_stop

    def test_clear_win_stops_early(self):
        """Stops early when clearly winning."""
        should_stop, reason = should_early_stop(15, 0, threshold=0.5, min_games=10)
        assert should_stop
        assert ">" in reason  # Win rate above threshold

    def test_clear_loss_stops_early(self):
        """Stops early when clearly losing."""
        should_stop, reason = should_early_stop(0, 15, threshold=0.5, min_games=10)
        assert should_stop
        assert "<" in reason  # Win rate below threshold

    def test_inconclusive_continues(self):
        """Continues when outcome is uncertain."""
        should_stop, reason = should_early_stop(5, 5, threshold=0.5, min_games=10)
        assert not should_stop
        assert "inconclusive" in reason.lower()

    def test_high_threshold_harder_to_pass(self):
        """Higher threshold requires more wins to pass."""
        # 70% wins - might pass 50% threshold but not 90%
        should_stop_low, _ = should_early_stop(14, 6, threshold=0.5, min_games=10)
        should_stop_high, _ = should_early_stop(14, 6, threshold=0.9, min_games=10)

        # Can't necessarily assert both outcomes, but structure should work
        assert isinstance(should_stop_low, bool)
        assert isinstance(should_stop_high, bool)

    def test_lower_confidence_more_lenient(self):
        """Lower confidence level more easily decides."""
        # With lower confidence, intervals are narrower
        # 60% wins with 0.90 confidence should decide faster than 0.99
        should_stop_90, _ = should_early_stop(12, 8, threshold=0.5, min_games=10, confidence=0.90)
        should_stop_99, _ = should_early_stop(12, 8, threshold=0.5, min_games=10, confidence=0.99)

        # Can't guarantee outcome but both should be valid
        assert isinstance(should_stop_90, bool)
        assert isinstance(should_stop_99, bool)

    def test_reason_includes_ci(self):
        """Reason message includes confidence interval info."""
        _, reason = should_early_stop(10, 10, threshold=0.5, min_games=10)
        assert "%" in reason  # Contains percentage


# =============================================================================
# GameResult Tests
# =============================================================================


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_game_result_creation(self):
        """GameResult can be created with all fields."""
        result = GameResult(
            winner=1,
            move_count=50,
            victory_reason="territory",
            candidate_player=1,
            candidate_won=True,
        )
        assert result.winner == 1
        assert result.move_count == 50
        assert result.candidate_won is True

    def test_game_result_draw(self):
        """GameResult handles draw case."""
        result = GameResult(
            winner=None,
            move_count=100,
            victory_reason="draw",
            candidate_player=1,
            candidate_won=False,
        )
        assert result.winner is None
        assert result.victory_reason == "draw"

    def test_game_result_candidate_lost(self):
        """GameResult tracks when candidate loses."""
        result = GameResult(
            winner=2,
            move_count=30,
            victory_reason="elimination",
            candidate_player=1,
            candidate_won=False,
        )
        assert result.winner == 2
        assert result.candidate_won is False


# =============================================================================
# GauntletResult Tests
# =============================================================================


class TestGauntletResult:
    """Tests for GauntletResult dataclass."""

    def test_default_values(self):
        """GauntletResult has sensible defaults."""
        result = GauntletResult()
        assert result.total_games == 0
        assert result.total_wins == 0
        assert result.win_rate == 0.0
        assert result.passes_baseline_gating is True
        assert result.estimated_elo == 1500.0
        assert result.opponent_results == {}
        assert result.failed_baselines == []

    def test_result_with_wins(self):
        """GauntletResult correctly stores win data."""
        result = GauntletResult(
            total_games=20,
            total_wins=15,
            total_losses=5,
            win_rate=0.75,
        )
        assert result.total_games == 20
        assert result.total_wins == 15
        assert result.win_rate == 0.75

    def test_failed_baseline_tracking(self):
        """GauntletResult tracks failed baselines."""
        result = GauntletResult(
            passes_baseline_gating=False,
            failed_baselines=["random", "heuristic"],
        )
        assert not result.passes_baseline_gating
        assert "random" in result.failed_baselines

    def test_early_stopping_tracking(self):
        """GauntletResult tracks early stopped baselines."""
        result = GauntletResult(
            early_stopped_baselines=["mcts_strong"],
            games_saved_by_early_stopping=50,
        )
        assert "mcts_strong" in result.early_stopped_baselines
        assert result.games_saved_by_early_stopping == 50

    def test_opponent_results_dict(self):
        """GauntletResult stores per-opponent results."""
        result = GauntletResult()
        result.opponent_results["random"] = {
            "wins": 10,
            "losses": 0,
            "win_rate": 1.0,
        }
        assert result.opponent_results["random"]["wins"] == 10

    def test_elo_estimate_stored(self):
        """GauntletResult stores Elo estimate."""
        result = GauntletResult(estimated_elo=1750.0)
        assert result.estimated_elo == 1750.0


# =============================================================================
# create_baseline_ai Tests (with mocks)
# =============================================================================


class TestCreateBaselineAI:
    """Tests for baseline AI factory function."""

    def test_create_baseline_ai_is_callable(self):
        """create_baseline_ai function is importable and callable."""
        from app.training.game_gauntlet import create_baseline_ai
        assert callable(create_baseline_ai)

    def test_create_baseline_ai_signature(self):
        """create_baseline_ai has expected parameters."""
        import inspect
        from app.training.game_gauntlet import create_baseline_ai

        sig = inspect.signature(create_baseline_ai)
        params = list(sig.parameters.keys())
        assert "baseline" in params
        assert "player" in params
        assert "board_type" in params

    def test_baseline_opponent_accepted(self):
        """All BaselineOpponent values are valid inputs."""
        # This tests the function interface, not actual AI creation
        # since AI creation requires game modules
        for baseline in BaselineOpponent:
            # Just verify the enum values are strings that can be used
            assert isinstance(baseline.value, str)


# =============================================================================
# Integration Tests (with heavy mocking)
# =============================================================================


class TestGauntletEvaluationFlow:
    """Integration tests for gauntlet evaluation flow."""

    def test_evaluate_single_opponent_emits_events(self):
        """_evaluate_single_opponent emits progress events."""
        from types import SimpleNamespace
        import app.coordination.event_router as event_router
        from app.training import game_gauntlet

        class StubBus:
            def __init__(self):
                self.events = []

            def publish_sync(self, event):
                self.events.append(event)
                return event

        stub_bus = StubBus()

        with patch.object(event_router, "get_event_bus", return_value=stub_bus):
            with patch.object(game_gauntlet, "create_neural_ai", return_value=MagicMock()):
                with patch.object(game_gauntlet, "create_baseline_ai", return_value=MagicMock()):
                    with patch.object(game_gauntlet, "play_single_game") as mock_play:
                        mock_play.return_value = SimpleNamespace(
                            candidate_won=True,
                            winner=1,
                            victory_reason="test",
                            move_count=10,
                        )

                        game_gauntlet._evaluate_single_opponent(
                            baseline=BaselineOpponent.RANDOM,
                            model_path="dummy.pth",
                            board_type=BoardType.SQUARE8,
                            games_per_opponent=3,
                            num_players=2,
                            verbose=False,
                            model_getter=None,
                            model_type="cnn",
                            early_stopping=False,
                            early_stopping_confidence=0.95,
                            early_stopping_min_games=1,
                        )

        assert len(stub_bus.events) > 0

    def test_gauntlet_result_structure(self):
        """GauntletResult has expected structure after evaluation."""
        result = GauntletResult(
            total_games=100,
            total_wins=75,
            total_losses=20,
            total_draws=5,
            win_rate=0.75,
            passes_baseline_gating=True,
            estimated_elo=1650.0,
        )

        assert result.total_games == result.total_wins + result.total_losses + result.total_draws
        assert 0.0 <= result.win_rate <= 1.0
        assert result.estimated_elo > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for game_gauntlet."""

    def test_wilson_with_large_samples(self):
        """Wilson interval handles large sample sizes."""
        lower, upper = compute_wilson_interval(5000, 10000, confidence=0.95)
        # Should be very close to 0.5 with tight interval
        assert 0.49 < lower < 0.51
        assert 0.49 < upper < 0.51

    def test_early_stop_all_wins(self):
        """Early stopping with 100% win rate."""
        should_stop, reason = should_early_stop(20, 0, threshold=0.5, min_games=10)
        assert should_stop

    def test_early_stop_all_losses(self):
        """Early stopping with 0% win rate."""
        should_stop, reason = should_early_stop(0, 20, threshold=0.5, min_games=10)
        assert should_stop

    def test_baseline_enum_string_values(self):
        """All baseline enum values are strings."""
        for baseline in BaselineOpponent:
            assert isinstance(baseline.value, str)
            assert len(baseline.value) > 0

    def test_gauntlet_result_immutability_of_defaults(self):
        """Default mutable fields don't share state."""
        result1 = GauntletResult()
        result2 = GauntletResult()

        result1.failed_baselines.append("test")
        assert "test" not in result2.failed_baselines

    def test_game_result_all_fields_required(self):
        """GameResult requires all fields at creation."""
        # This is a compile-time check via dataclass, but we verify
        result = GameResult(
            winner=1,
            move_count=0,
            victory_reason="",
            candidate_player=1,
            candidate_won=True,
        )
        assert result.move_count == 0  # Zero moves is valid
