"""Tests for export_replay_dataset value target functions."""

import os

# Import the functions to test
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.export_replay_dataset import (
    value_from_final_ranking,
    value_from_final_winner,
)


class MockPlayer:
    """Mock Player for testing."""
    def __init__(self, number: int, eliminated_rings: int, territory_spaces: int = 0):
        self.number = number
        self.player_number = number  # Canonical attribute used by value_from_final_ranking
        self.eliminated_rings = eliminated_rings
        self.territory_spaces = territory_spaces


class MockGameState:
    """Mock GameState for testing."""
    def __init__(self, winner: int | None = None, players: list | None = None):
        self.winner = winner
        self.players = players or []


class TestValueFromFinalWinner:
    """Tests for the legacy binary value function."""

    def test_winner_gets_positive_one(self):
        """Winner should get +1."""
        state = MockGameState(winner=1)
        assert value_from_final_winner(state, perspective=1) == 1.0

    def test_loser_gets_negative_one(self):
        """Loser should get -1."""
        state = MockGameState(winner=1)
        assert value_from_final_winner(state, perspective=2) == -1.0

    def test_no_winner_gets_zero(self):
        """No winner (draw/incomplete) should get 0."""
        state = MockGameState(winner=None)
        assert value_from_final_winner(state, perspective=1) == 0.0


class TestValueFromFinalRanking:
    """Tests for the rank-aware value function."""

    def test_2player_winner_gets_positive_one(self):
        """2-player winner should get +1."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
        ]
        state = MockGameState(winner=1, players=players)
        assert value_from_final_ranking(state, perspective=1, num_players=2) == 1.0

    def test_2player_loser_gets_negative_one(self):
        """2-player loser should get -1."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
        ]
        state = MockGameState(winner=1, players=players)
        assert value_from_final_ranking(state, perspective=2, num_players=2) == -1.0

    def test_3player_first_gets_positive_one(self):
        """3-player 1st place should get +1."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
            MockPlayer(3, eliminated_rings=1),
        ]
        state = MockGameState(winner=1, players=players)
        # Player 1 has most eliminated rings = 1st place
        assert value_from_final_ranking(state, perspective=1, num_players=3) == 1.0

    def test_3player_second_gets_zero(self):
        """3-player 2nd place should get 0."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
            MockPlayer(3, eliminated_rings=1),
        ]
        state = MockGameState(winner=1, players=players)
        # Player 2 is 2nd place
        assert value_from_final_ranking(state, perspective=2, num_players=3) == 0.0

    def test_3player_third_gets_negative_one(self):
        """3-player 3rd place should get -1."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
            MockPlayer(3, eliminated_rings=1),
        ]
        state = MockGameState(winner=1, players=players)
        # Player 3 is 3rd place
        assert value_from_final_ranking(state, perspective=3, num_players=3) == -1.0

    def test_4player_ranking_values(self):
        """4-player should use linear interpolation: +1, +0.333, -0.333, -1."""
        players = [
            MockPlayer(1, eliminated_rings=10),  # 1st
            MockPlayer(2, eliminated_rings=7),   # 2nd
            MockPlayer(3, eliminated_rings=4),   # 3rd
            MockPlayer(4, eliminated_rings=1),   # 4th
        ]
        state = MockGameState(winner=1, players=players)

        # 1st place: +1
        assert value_from_final_ranking(state, perspective=1, num_players=4) == pytest.approx(1.0)

        # 2nd place: 1 - 2*(2-1)/3 = 1 - 2/3 = 0.333...
        assert value_from_final_ranking(state, perspective=2, num_players=4) == pytest.approx(1.0/3.0)

        # 3rd place: 1 - 2*(3-1)/3 = 1 - 4/3 = -0.333...
        assert value_from_final_ranking(state, perspective=3, num_players=4) == pytest.approx(-1.0/3.0)

        # 4th place: -1
        assert value_from_final_ranking(state, perspective=4, num_players=4) == pytest.approx(-1.0)

    def test_territory_tiebreaker(self):
        """When eliminated_rings are tied, territory_spaces should break ties."""
        players = [
            MockPlayer(1, eliminated_rings=5, territory_spaces=10),  # 1st (same rings, more territory)
            MockPlayer(2, eliminated_rings=5, territory_spaces=5),   # 2nd
            MockPlayer(3, eliminated_rings=3, territory_spaces=0),   # 3rd
        ]
        state = MockGameState(winner=1, players=players)

        # Player 1 should be 1st (more territory on tie)
        assert value_from_final_ranking(state, perspective=1, num_players=3) == 1.0
        # Player 2 should be 2nd
        assert value_from_final_ranking(state, perspective=2, num_players=3) == 0.0
        # Player 3 should be 3rd
        assert value_from_final_ranking(state, perspective=3, num_players=3) == -1.0

    def test_no_winner_returns_zero(self):
        """When there's no winner (incomplete game), should return 0."""
        players = [
            MockPlayer(1, eliminated_rings=5),
            MockPlayer(2, eliminated_rings=3),
        ]
        state = MockGameState(winner=None, players=players)
        assert value_from_final_ranking(state, perspective=1, num_players=2) == 0.0

    def test_empty_players_returns_zero(self):
        """When players list is empty, should return 0."""
        state = MockGameState(winner=1, players=[])
        assert value_from_final_ranking(state, perspective=1, num_players=2) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
