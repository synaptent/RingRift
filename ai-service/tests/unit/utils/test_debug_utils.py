"""Tests for debug utilities.

Tests the state diffing and game replay debugging utilities.
"""

import pytest

from app.utils.debug_utils import (
    GPU_BOOKKEEPING_MOVES,
    StateDiff,
    StateDiffer,
    summarize_game_state,
    is_bookkeeping_move,
    compare_states,
)


class TestGpuBookkeepingMoves:
    """Tests for GPU_BOOKKEEPING_MOVES constant."""

    def test_is_frozen_set(self):
        assert isinstance(GPU_BOOKKEEPING_MOVES, frozenset)

    def test_contains_expected_moves(self):
        expected = {
            "skip_capture",
            "skip_recovery",
            "no_placement_action",
            "no_movement_action",
            "no_line_action",
            "no_territory_action",
            "process_line",
        }
        assert expected == GPU_BOOKKEEPING_MOVES


class TestIsBookkeepingMove:
    """Tests for is_bookkeeping_move function."""

    def test_bookkeeping_moves_return_true(self):
        assert is_bookkeeping_move("skip_capture") is True
        assert is_bookkeeping_move("no_line_action") is True
        assert is_bookkeeping_move("process_line") is True

    def test_regular_moves_return_false(self):
        assert is_bookkeeping_move("place_ring") is False
        assert is_bookkeeping_move("move_stack") is False
        assert is_bookkeeping_move("capture") is False


class TestStateDiff:
    """Tests for StateDiff dataclass."""

    def test_default_is_equal(self):
        diff = StateDiff()
        assert diff.are_equal is True

    def test_phase_mismatch_makes_unequal(self):
        diff = StateDiff(phase_match=False)
        assert diff.are_equal is False

    def test_player_mismatch_makes_unequal(self):
        diff = StateDiff(player_match=False)
        assert diff.are_equal is False

    def test_summary_equal(self):
        diff = StateDiff()
        assert "identical" in diff.summary().lower()

    def test_summary_unequal(self):
        diff = StateDiff(phase_match=False)
        summary = diff.summary()
        assert "differences" in summary.lower()
        assert "phase" in summary.lower()


class TestStateDiffer:
    """Tests for StateDiffer class."""

    def test_summarize_players_ts_empty(self):
        ts_state = {"players": []}
        result = StateDiffer.summarize_players_ts(ts_state)
        assert result == {}

    def test_summarize_players_ts_with_data(self):
        ts_state = {
            "players": [
                {
                    "playerNumber": 1,
                    "eliminatedRings": 2,
                    "territorySpaces": 5,
                    "ringsInHand": 3,
                },
                {
                    "playerNumber": 2,
                    "eliminatedRings": 1,
                    "territorySpaces": 4,
                    "ringsInHand": 4,
                },
            ]
        }
        result = StateDiffer.summarize_players_ts(ts_state)
        assert result[1] == (2, 5, 3)
        assert result[2] == (1, 4, 4)

    def test_summarize_stacks_ts_empty(self):
        ts_state = {"board": {"stacks": {}}}
        result = StateDiffer.summarize_stacks_ts(ts_state)
        assert result == {}

    def test_summarize_stacks_ts_with_data(self):
        ts_state = {
            "board": {
                "stacks": {
                    "0,0": {"stackHeight": 3, "controllingPlayer": 1},
                    "1,1": {"stackHeight": 2, "controllingPlayer": 2},
                }
            }
        }
        result = StateDiffer.summarize_stacks_ts(ts_state)
        assert result["0,0"] == (3, 1)
        assert result["1,1"] == (2, 2)

    def test_summarize_collapsed_ts_empty(self):
        ts_state = {"board": {"collapsedSpaces": {}}}
        result = StateDiffer.summarize_collapsed_ts(ts_state)
        assert result == {}

    def test_summarize_collapsed_ts_with_data(self):
        ts_state = {
            "board": {
                "collapsedSpaces": {
                    "2,2": 1,
                    "3,3": 2,
                }
            }
        }
        result = StateDiffer.summarize_collapsed_ts(ts_state)
        assert result["2,2"] == 1
        assert result["3,3"] == 2
