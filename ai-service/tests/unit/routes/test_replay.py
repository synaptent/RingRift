"""Tests for app.routes.replay - Game Replay API Routes.

This module tests the FastAPI replay endpoints including:
- GET /api/replay/games - List games
- GET /api/replay/games/{game_id} - Get game details
- GET /api/replay/games/{game_id}/moves - Get moves
- GET /api/replay/stats - Get database stats
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.routes.replay import (
    GameListResponse,
    GameMetadata,
    MovesResponse,
    PlayerMetadata,
    StatsResponse,
    get_replay_db,
    reset_replay_db,
    router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db():
    """Create a mock GameReplayDB."""
    db = MagicMock()
    return db


@pytest.fixture
def test_client(mock_db):
    """Create a test client with mocked database."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Reset and patch the DB
    reset_replay_db()
    with patch("app.routes.replay.get_replay_db", return_value=mock_db):
        yield TestClient(app)

    reset_replay_db()


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestPlayerMetadata:
    """Tests for PlayerMetadata model."""

    def test_required_fields(self):
        """Should require playerNumber and playerType."""
        player = PlayerMetadata(playerNumber=1, playerType="human")
        assert player.playerNumber == 1
        assert player.playerType == "human"

    def test_optional_fields(self):
        """Should allow optional AI fields."""
        player = PlayerMetadata(
            playerNumber=1,
            playerType="ai",
            aiType="gumbel-mcts",
            aiDifficulty=5,
        )
        assert player.aiType == "gumbel-mcts"
        assert player.aiDifficulty == 5


class TestGameMetadata:
    """Tests for GameMetadata model."""

    def test_required_fields(self):
        """Should accept required fields."""
        game = GameMetadata(
            gameId="test-game-id",
            boardType="square8",
            numPlayers=2,
            totalMoves=50,
            totalTurns=25,
            createdAt="2025-12-24T12:00:00Z",
        )
        assert game.gameId == "test-game-id"
        assert game.boardType == "square8"
        assert game.numPlayers == 2

    def test_optional_winner(self):
        """Should allow optional winner."""
        game = GameMetadata(
            gameId="test",
            boardType="hex8",
            numPlayers=2,
            totalMoves=50,
            totalTurns=25,
            createdAt="2025-12-24T12:00:00Z",
            winner=1,
        )
        assert game.winner == 1

    def test_optional_termination_reason(self):
        """Should allow optional termination reason."""
        game = GameMetadata(
            gameId="test",
            boardType="hex8",
            numPlayers=2,
            totalMoves=50,
            totalTurns=25,
            createdAt="2025-12-24T12:00:00Z",
            terminationReason="ring_elimination",
        )
        assert game.terminationReason == "ring_elimination"


class TestGameListResponse:
    """Tests for GameListResponse model."""

    def test_structure(self):
        """Should have correct structure."""
        response = GameListResponse(
            games=[],
            total=0,
            hasMore=False,
        )
        assert response.games == []
        assert response.total == 0
        assert response.hasMore is False


class TestStatsResponse:
    """Tests for StatsResponse model."""

    def test_structure(self):
        """Should have correct structure."""
        response = StatsResponse(
            totalGames=100,
            gamesByBoardType={"square8": 50, "hex8": 50},
            gamesByStatus={"completed": 100},
            gamesByTermination={"ring_elimination": 60, "territory": 40},
            totalMoves=5000,
            schemaVersion=5,
        )
        assert response.totalGames == 100
        assert response.gamesByBoardType["square8"] == 50


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestListGamesEndpoint:
    """Tests for GET /api/replay/games endpoint."""

    def test_list_games_empty(self, test_client, mock_db):
        """Should return empty list when no games."""
        mock_db.query_games.return_value = []
        mock_db.get_game_count.return_value = 0

        response = test_client.get("/api/replay/games")
        assert response.status_code == 200
        data = response.json()
        assert data["games"] == []
        assert data["total"] == 0
        assert data["hasMore"] is False

    def test_list_games_with_results(self, test_client, mock_db):
        """Should return games list."""
        mock_db.query_games.return_value = [
            {
                "game_id": "game-1",
                "board_type": "square8",
                "num_players": 2,
                "winner": 1,
                "termination_reason": "ring_elimination",
                "total_moves": 50,
                "total_turns": 25,
                "created_at": "2025-12-24T12:00:00Z",
                "completed_at": "2025-12-24T12:05:00Z",
            }
        ]
        mock_db.get_game_count.return_value = 1

        response = test_client.get("/api/replay/games")
        assert response.status_code == 200
        data = response.json()
        assert len(data["games"]) == 1
        assert data["games"][0]["gameId"] == "game-1"

    def test_list_games_with_filter(self, test_client, mock_db):
        """Should pass filters to query."""
        mock_db.query_games.return_value = []
        mock_db.get_game_count.return_value = 0

        response = test_client.get("/api/replay/games?board_type=hex8&num_players=2")
        assert response.status_code == 200
        mock_db.query_games.assert_called_once()
        call_kwargs = mock_db.query_games.call_args.kwargs
        assert call_kwargs.get("board_type") == "hex8"
        assert call_kwargs.get("num_players") == 2

    def test_list_games_pagination(self, test_client, mock_db):
        """Should handle pagination parameters."""
        mock_db.query_games.return_value = []
        mock_db.get_game_count.return_value = 0

        response = test_client.get("/api/replay/games?limit=10&offset=20")
        assert response.status_code == 200
        call_kwargs = mock_db.query_games.call_args.kwargs
        assert call_kwargs.get("limit") == 11  # +1 to check hasMore
        assert call_kwargs.get("offset") == 20

    def test_list_games_has_more(self, test_client, mock_db):
        """Should set hasMore when more results available."""
        # Return limit+1 items to trigger hasMore
        mock_db.query_games.return_value = [
            {
                "game_id": f"game-{i}",
                "board_type": "square8",
                "num_players": 2,
                "total_moves": 50,
                "total_turns": 25,
                "created_at": "2025-12-24T12:00:00Z",
            }
            for i in range(21)  # Default limit is 20
        ]
        mock_db.get_game_count.return_value = 100

        response = test_client.get("/api/replay/games")
        assert response.status_code == 200
        data = response.json()
        assert len(data["games"]) == 20  # Truncated to limit
        assert data["hasMore"] is True


class TestGetGameEndpoint:
    """Tests for GET /api/replay/games/{game_id} endpoint."""

    def test_get_game_found(self, test_client, mock_db):
        """Should return game details when found."""
        # The route uses get_game_with_players which includes players
        game_data = {
            "game_id": "test-game",
            "board_type": "hex8",
            "num_players": 2,
            "winner": 2,
            "termination_reason": "territory",
            "total_moves": 100,
            "total_turns": 50,
            "created_at": "2025-12-24T12:00:00Z",
            "completed_at": None,
            "duration_ms": None,
            "source": None,
            "time_control_type": None,
            "initial_time_ms": None,
            "time_increment_ms": None,
            "metadata_json": None,
            "players": [
                {"playerNumber": 1, "playerType": "ai"},
                {"playerNumber": 2, "playerType": "ai"},
            ],
        }
        mock_db.get_game_with_players.return_value = game_data

        response = test_client.get("/api/replay/games/test-game")
        assert response.status_code == 200
        data = response.json()
        assert data["gameId"] == "test-game"
        assert data["winner"] == 2

    def test_get_game_not_found(self, test_client, mock_db):
        """Should return 404 when game not found."""
        mock_db.get_game_with_players.return_value = None

        response = test_client.get("/api/replay/games/nonexistent")
        assert response.status_code == 404


class TestGetMovesEndpoint:
    """Tests for GET /api/replay/games/{game_id}/moves endpoint."""

    def test_get_moves(self, test_client, mock_db):
        """Should return moves for game."""
        # The route uses get_game_metadata, not get_game
        mock_db.get_game_metadata.return_value = {
            "game_id": "test-game",
            "total_moves": 10,
        }
        mock_db.get_move_records.return_value = [
            {
                "moveNumber": 1,
                "turnNumber": 1,
                "player": 1,
                "phase": "action",
                "moveType": "place",
                "move": {"from": [0, 0], "to": [1, 1]},
                "timestamp": None,
                "thinkTimeMs": None,
                "timeRemainingMs": None,
                "engineEval": None,
                "engineEvalType": None,
                "engineDepth": None,
                "engineNodes": None,
                "enginePV": None,
                "engineTimeMs": None,
            }
        ]

        response = test_client.get("/api/replay/games/test-game/moves")
        assert response.status_code == 200
        data = response.json()
        assert len(data["moves"]) == 1
        assert data["moves"][0]["moveNumber"] == 1

    def test_get_moves_game_not_found(self, test_client, mock_db):
        """Should return 404 when game not found."""
        mock_db.get_game_metadata.return_value = None

        response = test_client.get("/api/replay/games/nonexistent/moves")
        assert response.status_code == 404


class TestGetStatsEndpoint:
    """Tests for GET /api/replay/stats endpoint."""

    def test_get_stats(self, test_client, mock_db):
        """Should return database statistics."""
        mock_db.get_stats.return_value = {
            "total_games": 1000,
            "games_by_board_type": {"square8": 500, "hex8": 500},
            "games_by_status": {"completed": 1000},
            "games_by_termination": {"ring_elimination": 600, "territory": 400},
            "total_moves": 50000,
            "schema_version": 5,
        }

        response = test_client.get("/api/replay/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["totalGames"] == 1000
        assert data["gamesByBoardType"]["square8"] == 500


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetReplayDb:
    """Tests for get_replay_db function."""

    def test_creates_instance(self):
        """Should create DB instance on first call."""
        reset_replay_db()
        with patch("app.routes.replay.GameReplayDB") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            db = get_replay_db()
            assert db == mock_instance
            mock_class.assert_called_once()

        reset_replay_db()

    def test_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        reset_replay_db()
        with patch("app.routes.replay.GameReplayDB") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            db1 = get_replay_db()
            db2 = get_replay_db()
            assert db1 is db2
            mock_class.assert_called_once()  # Only created once

        reset_replay_db()


class TestResetReplayDb:
    """Tests for reset_replay_db function."""

    def test_resets_instance(self):
        """Should reset the DB instance."""
        reset_replay_db()
        with patch("app.routes.replay.GameReplayDB") as mock_class:
            mock_class.return_value = MagicMock()

            db1 = get_replay_db()
            reset_replay_db()
            db2 = get_replay_db()

            # Should have created two separate instances
            assert mock_class.call_count == 2

        reset_replay_db()
