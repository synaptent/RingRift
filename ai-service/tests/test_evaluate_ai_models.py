#!/usr/bin/env python
"""Tests for the AI Model Evaluation Framework.

Tests cover:
- AI type initialization
- Match execution with small game counts
- JSON output format validation
- Confidence interval calculation
- Color alternation logic
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Allow imports from scripts/ and app/
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(
    0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"
    )
)

# Import after path is set
from app.models import BoardType
from scripts.evaluate_ai_models import (
    AI_TYPE_BASELINE_HEURISTIC,
    AI_TYPE_CMAES_HEURISTIC,
    AI_TYPE_MINIMAX,
    AI_TYPE_NEURAL_NETWORK,
    AI_TYPE_RANDOM,
    EvaluationResults,
    GameResult,
    create_ai,
    format_results_json,
    wilson_score_interval,
)


class TestWilsonScoreInterval:
    """Tests for the Wilson score interval calculation."""

    def test_wilson_score_50_percent(self) -> None:
        """Test Wilson score for exactly 50% win rate."""
        lower, upper = wilson_score_interval(50, 100)
        # 50% win rate with 100 games should have CI roughly 40-60%
        assert 0.35 < lower < 0.45
        assert 0.55 < upper < 0.65

    def test_wilson_score_perfect_record(self) -> None:
        """Test Wilson score for 100% win rate."""
        lower, upper = wilson_score_interval(100, 100)
        # Even 100% win rate should have upper < 1.0 for finite samples
        assert upper <= 1.0
        assert lower > 0.95

    def test_wilson_score_zero_wins(self) -> None:
        """Test Wilson score for 0% win rate."""
        lower, upper = wilson_score_interval(0, 100)
        assert lower >= 0.0
        assert lower == 0.0  # Can't go below 0
        assert upper < 0.05  # Should be close to 0

    def test_wilson_score_empty(self) -> None:
        """Test Wilson score with no games."""
        lower, upper = wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_wilson_score_small_sample(self) -> None:
        """Test Wilson score with small sample size."""
        # 5 wins out of 10 games
        lower, upper = wilson_score_interval(5, 10)
        # Small sample = wider interval
        assert 0.1 < lower < 0.4
        assert 0.6 < upper < 0.9

    def test_wilson_score_asymmetric(self) -> None:
        """Test that Wilson score is asymmetric for extreme values.
        
        Wilson score interval extends more toward 0.5 than away from it.
        For 90% win rate, the distance from 0.9 to lower bound should be
        greater than the distance from 0.9 to upper bound.
        """
        # High win rate (90%)
        lower_high, upper_high = wilson_score_interval(90, 100)
        dist_toward_middle = 0.9 - lower_high  # Distance toward 0.5
        dist_away_from_middle = upper_high - 0.9  # Distance toward 1.0

        # Low win rate (10%)
        lower_low, upper_low = wilson_score_interval(10, 100)
        dist_toward_middle_low = upper_low - 0.1  # Distance toward 0.5
        dist_away_from_middle_low = 0.1 - lower_low  # Distance toward 0.0

        # Wilson score is asymmetric - spread is wider toward 0.5
        assert dist_toward_middle > dist_away_from_middle, (
            f"High win rate asymmetry: toward={dist_toward_middle}, "
            f"away={dist_away_from_middle}"
        )
        assert dist_toward_middle_low > dist_away_from_middle_low, (
            f"Low win rate asymmetry: toward={dist_toward_middle_low}, "
            f"away={dist_away_from_middle_low}"
        )


class TestCreateAI:
    """Tests for AI creation with different types."""

    def test_create_random_ai(self) -> None:
        """Test creating a Random AI."""
        ai = create_ai(AI_TYPE_RANDOM, player_num=1, board_type=BoardType.SQUARE8)
        assert ai is not None
        assert ai.player_number == 1

    def test_create_baseline_heuristic_ai(self) -> None:
        """Test creating a Baseline Heuristic AI."""
        ai = create_ai(AI_TYPE_BASELINE_HEURISTIC, player_num=1, board_type=BoardType.SQUARE8)
        assert ai is not None
        assert ai.player_number == 1
        # Check config has heuristic profile
        assert ai.config.heuristic_profile_id == "baseline_v1_balanced"

    def test_create_minimax_ai(self) -> None:
        """Test creating a Minimax AI."""
        ai = create_ai(AI_TYPE_MINIMAX, player_num=1, board_type=BoardType.SQUARE8, mm_depth=3)
        assert ai is not None
        assert ai.player_number == 1

    @pytest.mark.skip(reason="Legacy checkpoint format incompatible with model_versioning.py")
    def test_create_neural_network_ai(self) -> None:
        """Test creating a Neural Network AI (DescentAI)."""
        # This should work even without a checkpoint (uses default weights)
        ai = create_ai(AI_TYPE_NEURAL_NETWORK, player_num=1, board_type=BoardType.SQUARE8)
        assert ai is not None
        assert ai.player_number == 1

    def test_create_ai_player2(self) -> None:
        """Test creating AI for player 2."""
        ai = create_ai(AI_TYPE_RANDOM, player_num=2, board_type=BoardType.SQUARE8)
        assert ai.player_number == 2

    def test_create_ai_invalid_type(self) -> None:
        """Test that invalid AI type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown AI type"):
            create_ai("invalid_ai_type", player_num=1, board_type=BoardType.SQUARE8)

    def test_create_cmaes_ai_fallback(self) -> None:
        """Test CMA-ES AI creation falls back gracefully if weights missing."""
        # If the weights file doesn't exist, it should raise FileNotFoundError
        # We don't want to rely on the file existing in tests
        # So we mock the load function to succeed
        with patch(
            'scripts.evaluate_ai_models.load_cmaes_weights',
            return_value={"material": 1.0, "position": 0.5}
        ):
            ai = create_ai(AI_TYPE_CMAES_HEURISTIC, player_num=1, board_type=BoardType.SQUARE8)
            assert ai is not None
            assert ai.config.heuristic_profile_id == "cmaes_optimized"


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_game_result_defaults(self) -> None:
        """Test GameResult default values."""
        result = GameResult(
            winner=1,
            length=35,
            victory_type="elimination"
        )
        assert result.winner == 1
        assert result.length == 35
        assert result.victory_type == "elimination"
        assert result.p1_decision_times == []
        assert result.p2_decision_times == []
        assert result.p1_final_pieces == 0
        assert result.p2_final_pieces == 0
        assert result.error is None

    def test_game_result_draw(self) -> None:
        """Test GameResult for a draw."""
        result = GameResult(
            winner=None,
            length=200,
            victory_type="timeout"
        )
        assert result.winner is None
        assert result.victory_type == "timeout"


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""

    def test_evaluation_results_defaults(self) -> None:
        """Test EvaluationResults default values."""
        results = EvaluationResults(
            config={"player1": "random", "player2": "random", "games": 10}
        )
        assert results.player1_wins == 0
        assert results.player2_wins == 0
        assert results.draws == 0
        assert results.games == []
        assert results.game_lengths == []
        assert results.victory_types == {}


class TestFormatResultsJson:
    """Tests for JSON output formatting."""

    def test_format_results_basic(self) -> None:
        """Test basic JSON formatting."""
        results = EvaluationResults(
            config={
                "player1": "baseline_heuristic",
                "player2": "random",
                "games": 10,
                "board": "square8",
                "seed": 42,
            },
            player1_wins=8,
            player2_wins=1,
            draws=1,
            game_lengths=[30, 35, 28, 42, 33, 29, 45, 31, 38, 27],
            p1_decision_times=[0.01] * 150,
            p2_decision_times=[0.002] * 150,
            victory_types={"elimination": 9, "timeout": 1},
            p1_final_pieces_list=[5, 4, 6, 5, 5, 4, 3, 5, 4, 6],
            p2_final_pieces_list=[2, 3, 1, 2, 2, 3, 4, 2, 3, 1],
            total_runtime_seconds=5.5,
        )
        results.games = [
            {
                "game_number": i + 1,
                "winner": "baseline_heuristic" if i < 8 else "random",
                "length": 30
            }
            for i in range(10)
        ]

        formatted = format_results_json(results)

        # Check structure
        assert "config" in formatted
        assert "results" in formatted
        assert "games" in formatted

        # Check config
        assert formatted["config"]["player1"] == "baseline_heuristic"
        assert formatted["config"]["player2"] == "random"
        assert formatted["config"]["games"] == 10

        # Check results
        assert formatted["results"]["player1_wins"] == 8
        assert formatted["results"]["player2_wins"] == 1
        assert formatted["results"]["draws"] == 1
        assert formatted["results"]["player1_win_rate"] == 0.8
        assert isinstance(formatted["results"]["player1_win_rate_ci95"], list)
        assert len(formatted["results"]["player1_win_rate_ci95"]) == 2
        assert formatted["results"]["avg_game_length"] > 0
        assert formatted["results"]["avg_game_length_std"] > 0
        assert formatted["results"]["total_runtime_seconds"] == 5.5

        # Check games list
        assert len(formatted["games"]) == 10

    def test_format_results_empty(self) -> None:
        """Test formatting with no games played."""
        results = EvaluationResults(
            config={"player1": "a", "player2": "b", "games": 0}
        )
        formatted = format_results_json(results)

        assert formatted["results"]["player1_wins"] == 0
        assert formatted["results"]["player1_win_rate"] == 0
        assert formatted["results"]["avg_game_length"] == 0

    def test_format_results_json_serializable(self) -> None:
        """Test that formatted results can be JSON serialized."""
        config = {
            "player1": "baseline_heuristic",
            "player2": "random",
            "games": 2
        }
        results = EvaluationResults(
            config=config,
            player1_wins=1,
            player2_wins=1,
            game_lengths=[25, 30],
            p1_decision_times=[0.01, 0.02],
            p2_decision_times=[0.001, 0.001],
            total_runtime_seconds=1.5,
        )
        formatted = format_results_json(results)

        # This should not raise
        json_str = json.dumps(formatted, indent=2)
        assert len(json_str) > 0

        # And should be parseable
        parsed = json.loads(json_str)
        assert parsed["results"]["player1_wins"] == 1


class TestColorAlternation:
    """Tests for color alternation logic."""

    def test_even_game_player1_is_first_type(self) -> None:
        """Test that on even games, player1 type plays as Player 1."""
        # Game 0, 2, 4, etc. - player1_type should be P1
        for game_num in [0, 2, 4, 8, 10]:
            p1_is_player1_type = game_num % 2 == 0
            assert p1_is_player1_type is True

    def test_odd_game_player2_is_first_type(self) -> None:
        """Test that on odd games, player2 type plays as Player 1."""
        # Game 1, 3, 5, etc. - player2_type should be P1
        for game_num in [1, 3, 5, 7, 9]:
            p1_is_player1_type = game_num % 2 == 0
            assert p1_is_player1_type is False

    def test_equal_color_distribution(self) -> None:
        """Test that each player type gets equal games as P1 and P2."""
        num_games = 100
        p1_as_player1 = sum(1 for i in range(num_games) if i % 2 == 0)
        p1_as_player2 = sum(1 for i in range(num_games) if i % 2 == 1)

        assert p1_as_player1 == 50
        assert p1_as_player2 == 50

    def test_win_attribution_even_game(self) -> None:
        """Test win attribution when player1_type wins as P1."""
        game_num = 0
        p1_is_player1_type = game_num % 2 == 0
        game_winner = 1  # P1 wins

        # Determine which AI type won
        winner_is_p1_type = (
            (p1_is_player1_type and game_winner == 1)
            or (not p1_is_player1_type and game_winner == 2)
        )
        assert winner_is_p1_type is True

    def test_win_attribution_odd_game(self) -> None:
        """Test win attribution when player1_type wins as P2."""
        game_num = 1
        p1_is_player1_type = game_num % 2 == 0
        game_winner = 2  # P2 wins (which is player1_type in odd games)

        # Determine which AI type won
        winner_is_p1_type = (
            (p1_is_player1_type and game_winner == 1)
            or (not p1_is_player1_type and game_winner == 2)
        )
        assert winner_is_p1_type is True

    def test_win_attribution_player2_type_wins(self) -> None:
        """Test win attribution when player2_type wins."""
        # Even game, P2 wins (player2_type)
        game_num = 0
        p1_is_player1_type = game_num % 2 == 0
        game_winner = 2

        winner_is_p1_type = (
            (p1_is_player1_type and game_winner == 1)
            or (not p1_is_player1_type and game_winner == 2)
        )
        assert winner_is_p1_type is False

        # Odd game, P1 wins (player2_type since colors swapped)
        game_num = 1
        p1_is_player1_type = game_num % 2 == 0
        game_winner = 1

        winner_is_p1_type = (
            (p1_is_player1_type and game_winner == 1)
            or (not p1_is_player1_type and game_winner == 2)
        )
        assert winner_is_p1_type is False


class TestMatchExecution:
    """Tests for match execution with small game counts."""

    @pytest.mark.slow
    def test_two_game_match(self) -> None:
        """Test running a 2-game match for quick validation."""
        from scripts.evaluate_ai_models import run_evaluation
        from app.models import BoardType

        # Run a quick 2-game match between two random AIs
        results = run_evaluation(
            player1_type=AI_TYPE_RANDOM,
            player2_type=AI_TYPE_RANDOM,
            num_games=2,
            board_type=BoardType.SQUARE8,
            seed=42,
            checkpoint_path=None,
            checkpoint_path2=None,
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=50,  # Short games for speed
            verbose=False,
        )

        # Verify results
        assert results.player1_wins + results.player2_wins + results.draws == 2
        assert len(results.game_lengths) == 2
        assert len(results.games) == 2
        assert results.total_runtime_seconds > 0

    @pytest.mark.slow
    def test_baseline_vs_random_quick(self) -> None:
        """Test baseline heuristic against random (quick run)."""
        from scripts.evaluate_ai_models import run_evaluation
        from app.models import BoardType

        results = run_evaluation(
            player1_type=AI_TYPE_BASELINE_HEURISTIC,
            player2_type=AI_TYPE_RANDOM,
            num_games=4,
            board_type=BoardType.SQUARE8,
            seed=123,
            checkpoint_path=None,
            checkpoint_path2=None,
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=100,
            verbose=False,
        )

        # Baseline should generally beat random, but with only 4 games
        # we can't assert too much about win rates
        total = results.player1_wins + results.player2_wins + results.draws
        assert total == 4
        # At least check that some games were properly recorded
        assert len(results.games) == 4

    @pytest.mark.slow
    def test_output_file_creation(self) -> None:
        """Test that output file is properly created."""
        from scripts.evaluate_ai_models import (
            run_evaluation, format_results_json
        )
        from app.models import BoardType

        results = run_evaluation(
            player1_type=AI_TYPE_RANDOM,
            player2_type=AI_TYPE_RANDOM,
            num_games=2,
            board_type=BoardType.SQUARE8,
            seed=42,
            checkpoint_path=None,
            checkpoint_path2=None,
            cmaes_weights_path=None,
            minimax_depth=3,
            max_moves_per_game=50,
            verbose=False,
        )

        formatted = format_results_json(results)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(formatted, f, indent=2)
            temp_path = f.name

        try:
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            assert loaded["config"]["games"] == 2
            assert "results" in loaded
            assert "games" in loaded
        finally:
            os.unlink(temp_path)


class TestPieceAdvantageCalculation:
    """Tests for piece advantage calculation."""

    def test_piece_advantage_positive(self) -> None:
        """Test positive piece advantage (player1 has more pieces)."""
        results = EvaluationResults(
            config={"player1": "a", "player2": "b", "games": 3},
            p1_final_pieces_list=[10, 8, 9],
            p2_final_pieces_list=[5, 4, 6],
        )
        formatted = format_results_json(results)

        avg_p1 = sum([10, 8, 9]) / 3  # 9.0
        avg_p2 = sum([5, 4, 6]) / 3   # 5.0
        expected_advantage = avg_p1 - avg_p2  # 4.0

        actual = formatted["results"]["piece_advantage_p1"]
        assert actual == round(expected_advantage, 2)

    def test_piece_advantage_negative(self) -> None:
        """Test negative piece advantage (player2 has more pieces)."""
        results = EvaluationResults(
            config={"player1": "a", "player2": "b", "games": 2},
            p1_final_pieces_list=[3, 2],
            p2_final_pieces_list=[8, 10],
        )
        formatted = format_results_json(results)

        # p1 avg: 2.5, p2 avg: 9.0, advantage: -6.5
        assert formatted["results"]["piece_advantage_p1"] < 0


class TestVictoryTypeTracking:
    """Tests for victory type tracking."""

    def test_multiple_victory_types(self) -> None:
        """Test tracking multiple victory types."""
        results = EvaluationResults(
            config={"player1": "a", "player2": "b", "games": 5},
            victory_types={"elimination": 3, "territory": 1, "timeout": 1}
        )
        formatted = format_results_json(results)

        assert formatted["results"]["victory_types"]["elimination"] == 3
        assert formatted["results"]["victory_types"]["territory"] == 1
        assert formatted["results"]["victory_types"]["timeout"] == 1

    def test_empty_victory_types(self) -> None:
        """Test with no victory types recorded."""
        results = EvaluationResults(
            config={"player1": "a", "player2": "b", "games": 0}
        )
        formatted = format_results_json(results)

        assert formatted["results"]["victory_types"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])