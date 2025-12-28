"""Unit tests for app/training/generate_data.py.

Tests the core data generation utilities without requiring neural networks or GPUs.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.training.generate_data import (
    DataQualityTracker,
    MIN_UNIQUE_POSITIONS_PER_GAME,
    calculate_outcome,
    calculate_multi_player_outcome,
    _calculate_player_score,
    extract_mcts_visit_distribution,
)


# =============================================================================
# DataQualityTracker Tests
# =============================================================================


class TestDataQualityTracker:
    """Tests for the DataQualityTracker class."""

    def test_init_empty(self):
        """Test tracker initializes with empty state."""
        tracker = DataQualityTracker()
        summary = tracker.get_summary()

        assert summary["total_games"] == 0
        assert summary["avg_unique_positions"] == 0.0
        assert summary["low_uniqueness_games"] == 0

    def test_start_game_resets_state(self):
        """Test start_game resets per-game tracking."""
        tracker = DataQualityTracker()

        # Record some positions
        tracker.start_game()
        tracker.record_position(12345)
        tracker.record_position(67890)

        # Start a new game - should reset
        tracker.start_game()
        result = tracker.finish_game()

        assert result["unique_positions"] == 0
        assert result["total_moves"] == 0

    def test_record_position_tracks_unique(self):
        """Test that only unique positions are counted."""
        tracker = DataQualityTracker()
        tracker.start_game()

        # Record same position multiple times
        tracker.record_position(12345)
        tracker.record_position(12345)
        tracker.record_position(12345)

        result = tracker.finish_game()
        assert result["unique_positions"] == 1
        assert result["total_moves"] == 3

    def test_record_position_ignores_zero_hash(self):
        """Test that zero zobrist hashes are ignored."""
        tracker = DataQualityTracker()
        tracker.start_game()

        tracker.record_position(0)  # Should be ignored
        tracker.record_position(12345)
        tracker.record_position(0)  # Should be ignored

        result = tracker.finish_game()
        assert result["unique_positions"] == 1
        assert result["total_moves"] == 3

    def test_finish_game_calculates_metrics(self):
        """Test finish_game returns correct metrics."""
        tracker = DataQualityTracker()
        tracker.start_game()

        # Start from 1 since 0 is ignored
        for i in range(1, 101):
            tracker.record_position(i)

        result = tracker.finish_game()

        assert result["unique_positions"] == 100
        assert result["total_moves"] == 100
        assert result["uniqueness_ratio"] == 1.0
        assert result["below_threshold"] is False  # 100 >= 50

    def test_finish_game_detects_low_uniqueness(self):
        """Test finish_game detects games below threshold."""
        tracker = DataQualityTracker()
        tracker.start_game()

        # Only 30 unique positions (below MIN_UNIQUE_POSITIONS_PER_GAME=50)
        # Start from 1 since 0 is ignored
        for i in range(1, 31):
            tracker.record_position(i)

        result = tracker.finish_game()

        assert result["unique_positions"] == 30
        assert result["below_threshold"] is True

    def test_finish_game_handles_division_by_zero(self):
        """Test finish_game handles zero moves gracefully."""
        tracker = DataQualityTracker()
        tracker.start_game()
        # No moves recorded

        result = tracker.finish_game()

        assert result["unique_positions"] == 0
        assert result["total_moves"] == 0
        assert result["uniqueness_ratio"] == 0.0

    def test_get_summary_aggregates_multiple_games(self):
        """Test get_summary aggregates data across multiple games."""
        tracker = DataQualityTracker()

        # Game 1: 100 unique positions (start from 1 since 0 is ignored)
        tracker.start_game()
        for i in range(1, 101):
            tracker.record_position(i)
        tracker.finish_game()

        # Game 2: 60 unique positions
        tracker.start_game()
        for i in range(1001, 1061):
            tracker.record_position(i)
        tracker.finish_game()

        # Game 3: 30 unique positions (below threshold)
        tracker.start_game()
        for i in range(2001, 2031):
            tracker.record_position(i)
        tracker.finish_game()

        summary = tracker.get_summary()

        assert summary["total_games"] == 3
        assert summary["avg_unique_positions"] == pytest.approx(63.33, rel=0.01)
        assert summary["min_unique_positions"] == 30
        assert summary["max_unique_positions"] == 100
        assert summary["low_uniqueness_games"] == 1
        assert summary["low_uniqueness_pct"] == pytest.approx(33.33, rel=0.01)

    def test_log_summary_no_games(self, caplog):
        """Test log_summary handles zero games gracefully."""
        tracker = DataQualityTracker()
        tracker.log_summary()  # Should not raise

    def test_log_summary_warns_high_low_uniqueness(self, caplog):
        """Test log_summary warns when >10% games have low uniqueness."""
        import logging

        tracker = DataQualityTracker()

        # Create 3 games with low uniqueness (>10%)
        for _ in range(3):
            tracker.start_game()
            for i in range(20):  # Below threshold
                tracker.record_position(i)
            tracker.finish_game()

        with caplog.at_level(logging.WARNING):
            tracker.log_summary()

        assert "DATA_QUALITY_WARNING" in caplog.text


# =============================================================================
# calculate_outcome Tests
# =============================================================================


class TestCalculateOutcome:
    """Tests for the calculate_outcome function."""

    def test_winner_gets_positive_value(self):
        """Test that the winning player gets a positive outcome."""
        state = MagicMock()
        state.winner = 1
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result = calculate_outcome(state, player_number=1, depth=0)

        assert result > 0.0
        assert result <= 1.0

    def test_loser_gets_negative_value(self):
        """Test that a losing player gets a negative outcome."""
        state = MagicMock()
        state.winner = 2
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result = calculate_outcome(state, player_number=1, depth=0)

        assert result < 0.0
        assert result >= -1.0

    def test_no_winner_returns_zero(self):
        """Test that no winner returns zero outcome."""
        state = MagicMock()
        state.winner = None

        result = calculate_outcome(state, player_number=1, depth=0)

        assert result == 0.0

    def test_depth_discounting(self):
        """Test that deeper positions have discounted outcomes."""
        state = MagicMock()
        state.winner = 1
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        depth_0 = calculate_outcome(state, player_number=1, depth=0)
        depth_10 = calculate_outcome(state, player_number=1, depth=10)

        assert depth_10 < depth_0
        # gamma=0.99, so depth_10 should be approx 0.99^10 * depth_0
        expected_ratio = 0.99 ** 10
        assert depth_10 / depth_0 == pytest.approx(expected_ratio, rel=0.01)

    def test_territory_bonus(self):
        """Test that territory count adds bonus to outcome."""
        state_no_territory = MagicMock()
        state_no_territory.winner = 1
        state_no_territory.board.collapsed_spaces = {}
        state_no_territory.board.eliminated_rings = {}
        state_no_territory.board.markers = {}

        state_with_territory = MagicMock()
        state_with_territory.winner = 1
        # Large territory for visible bonus effect
        state_with_territory.board.collapsed_spaces = {f"pos{i}": 1 for i in range(100)}
        state_with_territory.board.eliminated_rings = {}
        state_with_territory.board.markers = {}

        # Use depth > 0 so the discount gives room for bonus to show
        # At depth=0, base value is 1.0 and clamped to max 1.0
        result_no = calculate_outcome(state_no_territory, player_number=1, depth=10)
        result_with = calculate_outcome(state_with_territory, player_number=1, depth=10)

        assert result_with > result_no

    def test_eliminated_rings_bonus(self):
        """Test that eliminated rings add bonus to outcome."""
        state_no_eliminated = MagicMock()
        state_no_eliminated.winner = 1
        state_no_eliminated.board.collapsed_spaces = {}
        state_no_eliminated.board.eliminated_rings = {}
        state_no_eliminated.board.markers = {}

        state_with_eliminated = MagicMock()
        state_with_eliminated.winner = 1
        state_with_eliminated.board.collapsed_spaces = {}
        # Large number for visible bonus effect (bonus = count * 0.001)
        state_with_eliminated.board.eliminated_rings = {"1": 500}
        state_with_eliminated.board.markers = {}

        # Use depth > 0 so the discount gives room for bonus to show
        result_no = calculate_outcome(state_no_eliminated, player_number=1, depth=10)
        result_with = calculate_outcome(state_with_eliminated, player_number=1, depth=10)

        assert result_with > result_no

    def test_markers_bonus(self):
        """Test that markers add small bonus to outcome."""
        state_no_markers = MagicMock()
        state_no_markers.winner = 1
        state_no_markers.board.collapsed_spaces = {}
        state_no_markers.board.eliminated_rings = {}
        state_no_markers.board.markers = {}

        marker = MagicMock()
        marker.player = 1
        state_with_markers = MagicMock()
        state_with_markers.winner = 1
        state_with_markers.board.collapsed_spaces = {}
        state_with_markers.board.eliminated_rings = {}
        # Large number of markers for visible bonus effect (bonus = count * 0.0001)
        state_with_markers.board.markers = {f"pos{i}": marker for i in range(1000)}

        # Use depth > 0 so the discount gives room for bonus to show
        result_no = calculate_outcome(state_no_markers, player_number=1, depth=10)
        result_with = calculate_outcome(state_with_markers, player_number=1, depth=10)

        assert result_with > result_no


# =============================================================================
# calculate_multi_player_outcome Tests
# =============================================================================


class TestCalculateMultiPlayerOutcome:
    """Tests for the calculate_multi_player_outcome function."""

    def test_two_player_winner_gets_positive(self):
        """Test 2-player game winner gets positive value."""
        state = MagicMock()
        state.winner = 1
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result = calculate_multi_player_outcome(state, num_players=2, depth=0)

        assert result.shape == (4,)  # max_players=4
        assert result[0] > 0.0  # Player 1 (index 0) wins
        assert result[1] < 0.0  # Player 2 (index 1) loses
        assert result[2] == 0.0  # Inactive
        assert result[3] == 0.0  # Inactive

    def test_four_player_winner_distribution(self):
        """Test 4-player game has correct outcome distribution."""
        state = MagicMock()
        state.winner = 3
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result = calculate_multi_player_outcome(state, num_players=4, depth=0)

        assert result.shape == (4,)
        assert result[2] > 0.0  # Player 3 (index 2) wins
        # Other players should have negative values
        assert result[0] < 0.0
        assert result[1] < 0.0
        assert result[3] < 0.0

    def test_no_winner_all_zeros(self):
        """Test that no winner results in all zeros."""
        state = MagicMock()
        state.winner = None

        result = calculate_multi_player_outcome(state, num_players=4, depth=0)

        assert np.all(result == 0.0)

    def test_graded_outcomes_second_place(self):
        """Test graded outcomes give intermediate value to second place."""
        state = MagicMock()
        state.winner = 1
        state.finish_order = [1, 3, 2, 4]  # Finish order
        state.board.collapsed_spaces = {"p1": 1, "p2": 1, "p3": 3, "p4": 2}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result = calculate_multi_player_outcome(
            state, num_players=4, depth=0, graded=True
        )

        assert result[0] > 0.0  # Winner (player 1)
        # With graded=True, non-winners should have graduated values
        # based on finish position

    def test_depth_discounting_multiplayer(self):
        """Test depth discounting applies to all players."""
        state = MagicMock()
        state.winner = 1
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        state.board.markers = {}

        result_depth_0 = calculate_multi_player_outcome(state, num_players=2, depth=0)
        result_depth_10 = calculate_multi_player_outcome(state, num_players=2, depth=10)

        # Winner value should be discounted
        assert abs(result_depth_10[0]) < abs(result_depth_0[0])


# =============================================================================
# _calculate_player_score Tests
# =============================================================================


class TestCalculatePlayerScore:
    """Tests for the _calculate_player_score function."""

    def test_winner_gets_highest_score(self):
        """Test that the winner gets the highest score."""
        state = MagicMock()
        state.winner = 1
        state.board.collapsed_spaces = {"a": 1, "b": 1, "c": 2}
        state.board.eliminated_rings = {"1": 5, "2": 3}
        marker1 = MagicMock()
        marker1.player = 1
        marker2 = MagicMock()
        marker2.player = 2
        state.board.markers = {"p1": marker1, "p2": marker2}

        score_p1 = _calculate_player_score(state, player_number=1)
        score_p2 = _calculate_player_score(state, player_number=2)

        assert score_p1 > score_p2

    def test_territory_contributes_to_score(self):
        """Test that territory count affects score."""
        state = MagicMock()
        state.winner = None
        state.board.collapsed_spaces = {"a": 1, "b": 1, "c": 1}  # 3 for player 1
        state.board.eliminated_rings = {}
        state.board.markers = {}

        score = _calculate_player_score(state, player_number=1)

        assert score > 0.0

    def test_eliminated_rings_contribute_to_score(self):
        """Test that eliminated rings affect score as a penalty.

        In _calculate_player_score, eliminated_rings is a PENALTY for the player
        who lost those rings (-2 per ring), not a bonus.
        """
        state = MagicMock()
        state.winner = None
        state.board.collapsed_spaces = {}
        # Player 1 has 10 rings eliminated - this is a PENALTY (-20 total)
        state.board.eliminated_rings = {"1": 10}
        state.board.markers = {}

        score = _calculate_player_score(state, player_number=1)

        # Eliminated rings are a penalty: -2 * 10 = -20
        assert score < 0.0
        assert score == -20.0  # Exactly -2 per eliminated ring

    def test_markers_contribute_to_score(self):
        """Test that markers affect score."""
        state = MagicMock()
        state.winner = None
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        marker = MagicMock()
        marker.player = 1
        state.board.markers = {"p1": marker, "p2": marker, "p3": marker}

        score = _calculate_player_score(state, player_number=1)

        assert score > 0.0

    def test_other_player_markers_dont_count(self):
        """Test that other player's markers don't contribute."""
        state = MagicMock()
        state.winner = None
        state.board.collapsed_spaces = {}
        state.board.eliminated_rings = {}
        marker = MagicMock()
        marker.player = 2  # Different player
        state.board.markers = {"p1": marker}

        score = _calculate_player_score(state, player_number=1)

        # Markers belong to player 2, so player 1 gets no marker bonus
        assert score == 0.0


# =============================================================================
# extract_mcts_visit_distribution Tests
# =============================================================================


class TestExtractMCTSVisitDistribution:
    """Tests for the extract_mcts_visit_distribution function."""

    def test_returns_empty_for_no_visits(self):
        """Test returns empty arrays when no visit distribution."""
        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([], [])
        ai.neural_net = None

        state = MagicMock()

        p_indices, p_values = extract_mcts_visit_distribution(ai, state)

        assert len(p_indices) == 0
        assert len(p_values) == 0

    def test_returns_empty_when_no_encoder(self):
        """Test returns empty when encoder is None and no neural_net."""
        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([MagicMock()], [0.5])
        ai.neural_net = None

        state = MagicMock()

        p_indices, p_values = extract_mcts_visit_distribution(
            ai, state, encoder=None, use_board_aware_encoding=False
        )

        assert len(p_indices) == 0
        assert len(p_values) == 0

    def test_encodes_moves_correctly(self):
        """Test that moves are encoded to policy indices."""
        move1 = MagicMock()
        move2 = MagicMock()

        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([move1, move2], [0.6, 0.4])

        encoder = MagicMock()
        encoder.encode_move.side_effect = [100, 200]

        state = MagicMock()

        p_indices, p_values = extract_mcts_visit_distribution(
            ai, state, encoder=encoder, use_board_aware_encoding=False
        )

        assert list(p_indices) == [100, 200]
        assert len(p_values) == 2

    def test_filters_invalid_moves(self):
        """Test that invalid move indices are filtered."""
        from app.ai.neural_net import INVALID_MOVE_INDEX

        move1 = MagicMock()
        move2 = MagicMock()

        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([move1, move2], [0.6, 0.4])

        encoder = MagicMock()
        encoder.encode_move.side_effect = [100, INVALID_MOVE_INDEX]

        state = MagicMock()

        p_indices, p_values = extract_mcts_visit_distribution(
            ai, state, encoder=encoder, use_board_aware_encoding=False
        )

        assert list(p_indices) == [100]
        assert len(p_values) == 1

    def test_renormalizes_probabilities(self):
        """Test that probabilities are renormalized after filtering."""
        from app.ai.neural_net import INVALID_MOVE_INDEX

        move1 = MagicMock()
        move2 = MagicMock()

        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([move1, move2], [0.6, 0.4])

        encoder = MagicMock()
        encoder.encode_move.side_effect = [100, INVALID_MOVE_INDEX]

        state = MagicMock()

        p_indices, p_values = extract_mcts_visit_distribution(
            ai, state, encoder=encoder, use_board_aware_encoding=False
        )

        # Only move1 remains, so its prob should be 1.0
        assert p_values[0] == pytest.approx(1.0)

    def test_board_aware_encoding(self):
        """Test board-aware encoding uses encode_move_for_board."""
        move = MagicMock()

        ai = MagicMock()
        ai.get_visit_distribution.return_value = ([move], [1.0])

        state = MagicMock()
        state.board = MagicMock()

        with patch(
            "app.training.generate_data.encode_move_for_board"
        ) as mock_encode:
            mock_encode.return_value = 42

            p_indices, p_values = extract_mcts_visit_distribution(
                ai, state, use_board_aware_encoding=True
            )

            mock_encode.assert_called_once_with(move, state.board)
            assert list(p_indices) == [42]


# =============================================================================
# MIN_UNIQUE_POSITIONS_PER_GAME constant test
# =============================================================================


def test_min_unique_positions_constant():
    """Test the threshold constant has expected value."""
    assert MIN_UNIQUE_POSITIONS_PER_GAME == 50
