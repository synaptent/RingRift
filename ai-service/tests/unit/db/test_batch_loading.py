"""Tests for batch loading methods in game_replay.py.

Tests cover:
- get_initial_states_batch
- get_moves_batch
- iterate_games with batch loading
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional

from app.models import GameState, Move, BoardType


class TestGetInitialStatesBatch:
    """Tests for get_initial_states_batch method."""

    def test_empty_game_ids(self):
        """Test with empty list returns empty dict."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            result = db.get_initial_states_batch([])

            assert result == {}

    def test_returns_dict_structure(self):
        """Test that batch method returns correct dict structure."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            # Mock the connection
            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute = MagicMock()

            # Mock table check returns False (no game_initial_state table)
            mock_conn.execute.return_value.fetchone.return_value = None
            # Mock game metadata query
            mock_conn.execute.return_value.fetchall.return_value = []

            with patch.object(db, '_get_conn', return_value=mock_conn):
                result = db.get_initial_states_batch(["game1", "game2"])

            # Should return dict with None for missing games
            assert "game1" in result
            assert "game2" in result


class TestGetMovesBatch:
    """Tests for get_moves_batch method."""

    def test_empty_game_ids(self):
        """Test with empty list returns empty dict."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            result = db.get_moves_batch([])

            assert result == {}

    def test_returns_empty_lists_for_games(self):
        """Test that batch method returns empty lists for games without moves."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            mock_conn = MagicMock()
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.execute.return_value.fetchall.return_value = []

            with patch.object(db, '_get_conn', return_value=mock_conn):
                result = db.get_moves_batch(["game1", "game2"])

            assert result == {"game1": [], "game2": []}


class TestIterateGames:
    """Tests for iterate_games with batch loading."""

    def test_batch_size_parameter(self):
        """Test that batch_size parameter is accepted."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            # Mock query_games to return empty
            with patch.object(db, 'query_games', return_value=[]):
                # Should accept batch_size without error
                result = list(db.iterate_games(batch_size=50))

            assert result == []

    def test_uses_batch_methods(self):
        """Test that iterate_games uses batch loading methods."""
        from app.db.game_replay import GameReplayDB

        with patch.object(GameReplayDB, '__init__', lambda x, y: None):
            db = GameReplayDB.__new__(GameReplayDB)
            db._db_path = ":memory:"

            # Create mock game metadata
            games = [
                {"game_id": "game1", "board_type": "square8", "num_players": 2},
                {"game_id": "game2", "board_type": "square8", "num_players": 2},
            ]

            mock_state = MagicMock(spec=GameState)

            with patch.object(db, 'query_games', return_value=games):
                with patch.object(db, 'get_initial_states_batch') as mock_states:
                    with patch.object(db, 'get_moves_batch') as mock_moves:
                        mock_states.return_value = {
                            "game1": mock_state,
                            "game2": mock_state,
                        }
                        mock_moves.return_value = {
                            "game1": [],
                            "game2": [],
                        }

                        result = list(db.iterate_games(batch_size=100))

            # Should have called batch methods
            mock_states.assert_called_once()
            mock_moves.assert_called_once()

            # Should yield 2 games
            assert len(result) == 2


class TestNNUEDatasetBatchLoading:
    """Tests for NNUE dataset batch loading optimization."""

    def test_import_succeeds(self):
        """Test that nnue_dataset module imports successfully."""
        from app.training.nnue_dataset import NNUEDatasetConfig

        config = NNUEDatasetConfig(
            board_type=BoardType.SQUARE8,
            num_players=2,
        )
        assert config.board_type == BoardType.SQUARE8
