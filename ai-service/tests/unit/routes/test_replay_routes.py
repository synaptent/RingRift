"""Comprehensive tests for replay API routes.

Tests all endpoints in app/routes/replay.py including:
- GET /api/replay/games (list with filters)
- GET /api/replay/games/{game_id} (single game with players)
- GET /api/replay/games/{game_id}/moves (move records)
- GET /api/replay/games/{game_id}/state (state reconstruction)
- GET /api/replay/games/{game_id}/choices (player choices)
- GET /api/replay/stats (database statistics)
- POST /api/replay/games (store game from sandbox)

Uses FastAPI TestClient and pytest fixtures for comprehensive coverage.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine
from app.models import (
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Position,
)
from app.routes.replay import reset_replay_db, router
from app.training.initial_state import create_initial_state


def _build_legal_move_sequence(
    initial_state: GameState,
    move_count: int,
) -> tuple[list[Move], GameState]:
    """Generate a canonical move sequence including required bookkeeping moves."""
    state = initial_state.model_copy(deep=True)
    moves: list[Move] = []

    for index in range(move_count):
        legal_moves = GameEngine.get_valid_moves(state, state.current_player)
        if legal_moves:
            base_move = legal_moves[0]
        else:
            requirement = GameEngine.get_phase_requirement(state, state.current_player)
            assert requirement is not None, f"No legal or bookkeeping move at step {index + 1}"
            synthesized = GameEngine.synthesize_bookkeeping_move(requirement, state)
            assert synthesized is not None, f"Failed to synthesize bookkeeping move at step {index + 1}"
            base_move = synthesized

        move = base_move.model_copy(
            update={
                "id": f"m{index + 1}",
                "move_number": index + 1,
                "timestamp": datetime.now(),
            }
        )
        moves.append(move)
        state = GameEngine.apply_move(state, move, trace_mode=True)

    return moves, state


def _moves_payload(moves: list[Move]) -> list[dict[str, Any]]:
    """Serialize moves for replay route POST payloads."""
    return [move.model_dump(mode="json", by_alias=True) for move in moves]


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Also clean up WAL and SHM files if they exist
    for ext in ["-wal", "-shm"]:
        wal_path = db_path + ext
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def db(temp_db_path):
    """Create a GameReplayDB instance for testing."""
    return GameReplayDB(temp_db_path, enforce_canonical_history=False)


@pytest.fixture
def test_app(temp_db_path):
    """Create a test FastAPI app with the replay router."""
    app = FastAPI()
    app.include_router(router)

    # Set the database path for the router
    os.environ["GAME_REPLAY_DB_PATH"] = temp_db_path

    # Reset the global DB instance to pick up the new path
    reset_replay_db()

    yield app

    # Cleanup
    reset_replay_db()
    if "GAME_REPLAY_DB_PATH" in os.environ:
        del os.environ["GAME_REPLAY_DB_PATH"]


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_game_state() -> GameState:
    """Create a minimal valid game state for testing."""
    return create_initial_state(board_type=BoardType.SQUARE8, num_players=2)


@pytest.fixture
def sample_completed_game_state(sample_game_state) -> GameState:
    """Create a completed game state."""
    state = sample_game_state.model_copy(deep=True)
    state.game_status = GameStatus.COMPLETED
    state.winner = 1
    state.players[0].eliminated_rings = 5
    state.players[1].eliminated_rings = 3
    return state


@pytest.fixture
def populated_db(db):
    """Create a database with sample games for testing."""
    # Game 1: Completed hex8 2-player game
    initial_state_1 = create_initial_state(board_type=BoardType.HEX8, num_players=2)
    moves_1, replayed_state_1 = _build_legal_move_sequence(initial_state_1, move_count=5)
    final_state_1 = replayed_state_1.model_copy(deep=True)
    final_state_1.game_status = GameStatus.COMPLETED
    final_state_1.winner = 1

    db.store_game(
        game_id="game-hex8-2p-1",
        initial_state=initial_state_1,
        final_state=final_state_1,
        moves=moves_1,
        metadata={"source": "test", "termination_reason": "ring_elimination"},
    )

    # Game 2: Completed square8 2-player game with more moves
    initial_state_2 = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    moves_2, replayed_state_2 = _build_legal_move_sequence(initial_state_2, move_count=8)
    final_state_2 = replayed_state_2.model_copy(deep=True)
    final_state_2.game_status = GameStatus.COMPLETED
    final_state_2.winner = 2

    db.store_game(
        game_id="game-square8-2p-1",
        initial_state=initial_state_2,
        final_state=final_state_2,
        moves=moves_2,
        metadata={"source": "selfplay", "termination_reason": "ring_elimination"},
    )

    # Game 3: In-progress square8 3-player game
    initial_state_3 = create_initial_state(board_type=BoardType.SQUARE8, num_players=3)
    moves_3, replayed_state_3 = _build_legal_move_sequence(initial_state_3, move_count=5)

    db.store_game(
        game_id="game-square8-3p-1",
        initial_state=initial_state_3,
        final_state=replayed_state_3,
        moves=moves_3,
        metadata={"source": "test"},
    )

    return db


class TestListGames:
    """Tests for GET /api/replay/games endpoint."""

    def test_list_all_games(self, client, populated_db):
        """Test listing all games without filters."""
        response = client.get("/api/replay/games")
        assert response.status_code == 200

        data = response.json()
        assert "games" in data
        assert "total" in data
        assert "hasMore" in data

        assert data["total"] == 3
        assert len(data["games"]) == 3
        assert data["hasMore"] is False

    def test_list_games_with_board_type_filter(self, client, populated_db):
        """Test filtering games by board type."""
        response = client.get("/api/replay/games?board_type=hex8")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert len(data["games"]) == 1
        assert data["games"][0]["boardType"] == "hex8"

    def test_list_games_with_num_players_filter(self, client, populated_db):
        """Test filtering games by number of players."""
        response = client.get("/api/replay/games?num_players=2")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert all(g["numPlayers"] == 2 for g in data["games"])

    def test_list_games_with_winner_filter(self, client, populated_db):
        """Test filtering games by winner."""
        response = client.get("/api/replay/games?winner=1")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["games"][0]["winner"] == 1

    def test_list_games_with_termination_reason_filter(self, client, populated_db):
        """Test filtering games by termination reason."""
        response = client.get("/api/replay/games?termination_reason=ring_elimination")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert all(g["terminationReason"] == "ring_elimination" for g in data["games"])

    def test_list_games_with_source_filter(self, client, populated_db):
        """Test filtering games by source."""
        response = client.get("/api/replay/games?source=selfplay")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["games"][0]["source"] == "selfplay"

    def test_list_games_with_min_moves_filter(self, client, populated_db):
        """Test filtering games by minimum move count."""
        response = client.get("/api/replay/games?min_moves=6")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["games"][0]["totalMoves"] >= 6

    def test_list_games_with_max_moves_filter(self, client, populated_db):
        """Test filtering games by maximum move count."""
        response = client.get("/api/replay/games?max_moves=5")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert all(g["totalMoves"] <= 5 for g in data["games"])

    def test_list_games_with_multiple_filters(self, client, populated_db):
        """Test combining multiple filters."""
        response = client.get(
            "/api/replay/games?board_type=square8&num_players=2&max_moves=10"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["games"][0]["boardType"] == "square8"
        assert data["games"][0]["numPlayers"] == 2

    def test_list_games_pagination_limit(self, client, populated_db):
        """Test pagination with limit."""
        response = client.get("/api/replay/games?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data["games"]) == 2
        assert data["total"] == 3
        assert data["hasMore"] is True

    def test_list_games_pagination_offset(self, client, populated_db):
        """Test pagination with offset."""
        response = client.get("/api/replay/games?limit=2&offset=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data["games"]) == 1
        assert data["total"] == 3
        assert data["hasMore"] is False

    def test_list_games_invalid_num_players(self, client, populated_db):
        """Test invalid num_players parameter."""
        response = client.get("/api/replay/games?num_players=1")
        assert response.status_code == 422  # Validation error

    def test_list_games_invalid_limit(self, client, populated_db):
        """Test invalid limit parameter."""
        response = client.get("/api/replay/games?limit=200")
        assert response.status_code == 422  # Validation error

    def test_list_games_empty_result(self, client, populated_db):
        """Test query that returns no results."""
        response = client.get("/api/replay/games?board_type=hexagonal")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0
        assert len(data["games"]) == 0
        assert data["hasMore"] is False

    def test_list_games_metadata_json_field(self, client, populated_db):
        """Test that metadata JSON is properly decoded."""
        response = client.get("/api/replay/games")
        assert response.status_code == 200

        data = response.json()
        # At least one game should have metadata
        games_with_metadata = [g for g in data["games"] if g.get("metadata")]
        assert len(games_with_metadata) > 0

        # Check metadata structure
        for game in games_with_metadata:
            assert isinstance(game["metadata"], dict)
            assert "source" in game["metadata"]


class TestGetGame:
    """Tests for GET /api/replay/games/{game_id} endpoint."""

    def test_get_existing_game(self, client, populated_db):
        """Test retrieving an existing game."""
        response = client.get("/api/replay/games/game-hex8-2p-1")
        assert response.status_code == 200

        data = response.json()
        assert data["gameId"] == "game-hex8-2p-1"
        assert data["boardType"] == "hex8"
        assert data["numPlayers"] == 2
        assert "players" in data
        assert len(data["players"]) == 2

    def test_get_game_player_details(self, client, populated_db):
        """Test that player details are included."""
        response = client.get("/api/replay/games/game-hex8-2p-1")
        assert response.status_code == 200

        data = response.json()
        players = data["players"]
        assert len(players) == 2

        # Check player structure
        for player in players:
            assert "playerNumber" in player
            assert "playerType" in player
            assert "finalEliminatedRings" in player

    def test_get_nonexistent_game(self, client, populated_db):
        """Test retrieving a non-existent game returns 404."""
        response = client.get("/api/replay/games/nonexistent-game-id")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_game_with_metadata(self, client, populated_db):
        """Test that game metadata is properly decoded."""
        response = client.get("/api/replay/games/game-hex8-2p-1")
        assert response.status_code == 200

        data = response.json()
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["source"] == "test"


class TestGetMoves:
    """Tests for GET /api/replay/games/{game_id}/moves endpoint."""

    def test_get_all_moves(self, client, populated_db):
        """Test retrieving all moves for a game."""
        response = client.get("/api/replay/games/game-hex8-2p-1/moves")
        assert response.status_code == 200

        data = response.json()
        assert "moves" in data
        assert "hasMore" in data
        assert len(data["moves"]) == 5
        assert data["hasMore"] is False

    def test_get_moves_with_range(self, client, populated_db):
        """Test retrieving moves in a specific range."""
        response = client.get("/api/replay/games/game-square8-2p-1/moves?start=0&end=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["moves"]) == 3

    def test_get_moves_with_limit(self, client, populated_db):
        """Test limiting the number of moves returned."""
        response = client.get("/api/replay/games/game-square8-2p-1/moves?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data["moves"]) == 2
        assert data["hasMore"] is True

    def test_get_moves_structure(self, client, populated_db):
        """Test move record structure."""
        response = client.get("/api/replay/games/game-hex8-2p-1/moves")
        assert response.status_code == 200

        data = response.json()
        move = data["moves"][0]

        # Required fields
        assert "moveNumber" in move
        assert "turnNumber" in move
        assert "player" in move
        assert "phase" in move
        assert "moveType" in move
        assert "move" in move

        # Optional v2 fields
        assert "engineEval" in move or move.get("engineEval") is None

    def test_get_moves_nonexistent_game(self, client, populated_db):
        """Test getting moves for non-existent game returns 404."""
        response = client.get("/api/replay/games/nonexistent-game/moves")
        assert response.status_code == 404

    def test_get_moves_invalid_limit(self, client, populated_db):
        """Test invalid limit parameter."""
        response = client.get("/api/replay/games/game-hex8-2p-1/moves?limit=2000")
        assert response.status_code == 422


class TestGetState:
    """Tests for GET /api/replay/games/{game_id}/state endpoint."""

    def test_get_initial_state(self, client, populated_db):
        """Test getting the initial state (move_number=0)."""
        response = client.get("/api/replay/games/game-hex8-2p-1/state?move_number=0")
        assert response.status_code == 200

        data = response.json()
        assert "gameState" in data
        assert "moveNumber" in data
        assert "totalMoves" in data
        assert data["moveNumber"] == 0
        assert data["totalMoves"] == 5

    def test_get_state_at_move(self, client, populated_db):
        """Test getting state at a specific move."""
        response = client.get("/api/replay/games/game-hex8-2p-1/state?move_number=1")
        assert response.status_code == 200

        data = response.json()
        assert data["moveNumber"] == 1
        assert isinstance(data["gameState"], dict)

    def test_get_state_game_state_structure(self, client, populated_db):
        """Test that game state has proper structure."""
        response = client.get("/api/replay/games/game-hex8-2p-1/state?move_number=0")
        assert response.status_code == 200

        data = response.json()
        game_state = data["gameState"]

        # Check required GameState fields (using camelCase for JSON)
        assert "boardType" in game_state
        assert "currentPlayer" in game_state
        assert "currentPhase" in game_state
        assert "gameStatus" in game_state
        assert "players" in game_state

    def test_get_state_nonexistent_game(self, client, populated_db):
        """Test getting state for non-existent game returns 404."""
        response = client.get("/api/replay/games/nonexistent-game/state?move_number=0")
        assert response.status_code == 404

    def test_get_state_move_exceeds_total(self, client, populated_db):
        """Test that requesting move beyond total returns 400."""
        response = client.get("/api/replay/games/game-hex8-2p-1/state?move_number=100")
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "exceeds" in data["detail"].lower()

    def test_get_state_with_legacy_flag(self, client, populated_db):
        """Test using legacy replay flag."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            response = client.get(
                "/api/replay/games/game-hex8-2p-1/state?move_number=1&legacy=true"
            )
        # Should work (legacy mode is backward compatible)
        assert response.status_code == 200

    def test_get_state_engine_eval_fields(self, client, populated_db):
        """Test that engine evaluation fields are present."""
        response = client.get("/api/replay/games/game-hex8-2p-1/state?move_number=1")
        assert response.status_code == 200

        data = response.json()
        # These fields should be present (may be None)
        assert "engineEval" in data
        assert "enginePV" in data


class TestGetChoices:
    """Tests for GET /api/replay/games/{game_id}/choices endpoint."""

    def test_get_choices_no_choices(self, client, populated_db):
        """Test getting choices when none exist."""
        response = client.get(
            "/api/replay/games/game-hex8-2p-1/choices?move_number=0"
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 0

    def test_get_choices_nonexistent_game(self, client, populated_db):
        """Test getting choices for non-existent game returns 404."""
        response = client.get(
            "/api/replay/games/nonexistent-game/choices?move_number=0"
        )
        assert response.status_code == 404

    def test_get_choices_missing_move_number(self, client, populated_db):
        """Test that move_number is required."""
        response = client.get("/api/replay/games/game-hex8-2p-1/choices")
        assert response.status_code == 422  # Validation error


class TestGetStats:
    """Tests for GET /api/replay/stats endpoint."""

    def test_get_stats(self, client, populated_db):
        """Test retrieving database statistics."""
        response = client.get("/api/replay/stats")
        assert response.status_code == 200

        data = response.json()
        assert "totalGames" in data
        assert "gamesByBoardType" in data
        assert "gamesByStatus" in data
        assert "gamesByTermination" in data
        assert "totalMoves" in data
        assert "schemaVersion" in data

        assert data["totalGames"] == 3
        assert data["totalMoves"] > 0

    def test_get_stats_board_type_breakdown(self, client, populated_db):
        """Test that board type breakdown is correct."""
        response = client.get("/api/replay/stats")
        assert response.status_code == 200

        data = response.json()
        board_types = data["gamesByBoardType"]

        assert "hex8" in board_types
        assert "square8" in board_types
        assert board_types["hex8"] == 1
        assert board_types["square8"] == 2

    def test_get_stats_empty_database(self, client):
        """Test stats on empty database."""
        response = client.get("/api/replay/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["totalGames"] == 0
        assert data["totalMoves"] == 0


class TestStoreGame:
    """Tests for POST /api/replay/games endpoint."""

    def test_store_game_success(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Test successfully storing a game."""
        moves, _ = _build_legal_move_sequence(sample_game_state, move_count=5)
        payload = {
            "gameId": "test-store-game-1",
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
            "metadata": {
                "source": "sandbox",
                "custom_field": "test_value",
                "termination_reason": "ring_elimination",
            },
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["gameId"] == "test-store-game-1"
        assert data["totalMoves"] == 5

    def test_store_game_auto_generate_id(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Test storing a game without providing game ID."""
        moves, _ = _build_legal_move_sequence(sample_game_state, move_count=5)
        payload = {
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "gameId" in data
        assert len(data["gameId"]) > 0  # Should be a UUID

    def test_store_game_with_choices(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Test storing a game with player choices."""
        moves, _ = _build_legal_move_sequence(sample_game_state, move_count=5)
        payload = {
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
            "choices": [
                {
                    "choiceType": "line_order",
                    "player": 1,
                    "options": [{"lineId": 1}, {"lineId": 2}],
                    "selected": {"lineId": 1},
                }
            ],
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_store_game_invalid_state(self, client):
        """Test storing a game with invalid state."""
        payload = {
            "initialState": {"invalid": "state"},
            "finalState": {"invalid": "state"},
            "moves": [],
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 422 or response.status_code == 500

    def test_store_game_missing_required_fields(self, client, sample_game_state):
        """Test storing a game without required fields."""
        payload = {
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            # Missing finalState and moves
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 422

    def test_store_game_too_many_moves(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Test that moves limit is enforced."""
        # Create a payload with too many moves (max is 10000)
        moves = [
            {
                "id": f"m{i}",
                "type": "place_ring",
                "player": 1,
                "to": {"x": 0, "y": 0},
            }
            for i in range(10001)
        ]

        payload = {
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": moves,
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 422  # Validation error

    def test_store_game_sets_default_source(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Test that source defaults to 'sandbox' when not provided."""
        moves, _ = _build_legal_move_sequence(sample_game_state, move_count=5)
        payload = {
            "gameId": "test-default-source",
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
        }

        response = client.post("/api/replay/games", json=payload)
        assert response.status_code == 200

        # Verify the game was stored with source='sandbox'
        get_response = client.get("/api/replay/games/test-default-source")
        assert get_response.status_code == 200

        game_data = get_response.json()
        assert game_data["metadata"]["source"] == "sandbox"

    def test_store_game_is_idempotent_by_game_id(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Duplicate game IDs are accepted as idempotent submissions."""
        moves, _ = _build_legal_move_sequence(sample_game_state, move_count=5)
        payload = {
            "gameId": "test-idempotent-store",
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
            "metadata": {"source": "sandbox", "termination_reason": "ring_elimination"},
        }

        first = client.post("/api/replay/games", json=payload)
        second = client.post("/api/replay/games", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        second_data = second.json()
        assert second_data["success"] is True
        assert second_data["deduplicated"] is True
        assert second_data["gameId"] == "test-idempotent-store"

    def test_non_replayable_human_sandbox_game_is_quarantined(
        self, client, sample_game_state, sample_completed_game_state
    ):
        """Human/sandbox games are recorded but excluded if Python replay rejects them."""
        legal_first_move = GameEngine.get_valid_moves(
            sample_game_state,
            sample_game_state.current_player,
        )[0].model_copy(
            update={
                "id": "m1",
                "move_number": 1,
                "timestamp": datetime.now(),
            }
        )
        invalid_skip = Move(
            id="bad-skip",
            type=MoveType.SKIP_PLACEMENT,
            player=sample_game_state.current_player,
            to=Position(x=0, y=0),
            timestamp=datetime.now(),
        )
        moves = [
            legal_first_move,
            *[
                invalid_skip.model_copy(update={"id": f"bad-skip-{idx}", "move_number": idx})
                for idx in range(2, 6)
            ],
        ]
        payload = {
            "gameId": "test-quarantined-human-sandbox",
            "initialState": sample_game_state.model_dump(mode='json', by_alias=True),
            "finalState": sample_completed_game_state.model_dump(mode='json', by_alias=True),
            "moves": _moves_payload(moves),
            "metadata": {
                "source": "sandbox",
                "playerTypes": ["human", "ai"],
                "termination_reason": "ring_elimination",
            },
        }

        response = client.post("/api/replay/games", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["acceptedForTraining"] is False
        assert data["parityStatus"] == "non_canonical_history"

        get_response = client.get("/api/replay/games/test-quarantined-human-sandbox")
        assert get_response.status_code == 200
        game_data = get_response.json()
        assert game_data["source"] == "human_vs_ai_quarantine"
        assert game_data["metadata"]["excluded_from_training"] is True
        assert "replay_validation_error" in game_data["metadata"]


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_missing_database(self, client):
        """Test behavior when database doesn't exist (should auto-create)."""
        # The database is created automatically, so this should work
        response = client.get("/api/replay/stats")
        assert response.status_code == 200

    def test_invalid_game_id_format(self, client, populated_db):
        """Test various invalid game ID formats."""
        # Empty string
        response = client.get("/api/replay/games/", follow_redirects=False)
        assert response.status_code in [307, 404, 405]

        # Special characters should work (URLEncoded)
        response = client.get("/api/replay/games/test%20game%20id")
        assert response.status_code == 404  # Game doesn't exist

    def test_concurrent_requests(self, client, populated_db):
        """Test that concurrent requests don't cause issues."""
        import concurrent.futures

        def make_request():
            return client.get("/api/replay/games")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == 200 for r in results)

    def test_malformed_json_in_post(self, client):
        """Test POST with malformed JSON."""
        response = client.post(
            "/api/replay/games",
            content="invalid json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestPaginationEdgeCases:
    """Tests for pagination edge cases."""

    def test_offset_beyond_total(self, client, populated_db):
        """Test offset beyond total games."""
        response = client.get("/api/replay/games?offset=1000")
        assert response.status_code == 200

        data = response.json()
        assert len(data["games"]) == 0
        assert data["total"] == 3
        assert data["hasMore"] is False

    def test_limit_zero(self, client, populated_db):
        """Test limit of zero (should fail validation)."""
        response = client.get("/api/replay/games?limit=0")
        assert response.status_code == 422

    def test_exact_page_boundary(self, client, populated_db):
        """Test when limit exactly matches remaining items."""
        response = client.get("/api/replay/games?limit=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["games"]) == 3
        assert data["hasMore"] is False


class TestResponseSchema:
    """Tests to verify response schema compliance."""

    def test_game_metadata_schema(self, client, populated_db):
        """Test GameMetadata response schema."""
        response = client.get("/api/replay/games/game-hex8-2p-1")
        assert response.status_code == 200

        data = response.json()

        # Required fields
        required_fields = [
            "gameId", "boardType", "numPlayers", "totalMoves", "totalTurns", "createdAt"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Optional fields should be present (may be None)
        optional_fields = [
            "winner", "terminationReason", "completedAt", "durationMs", "source",
            "timeControlType", "initialTimeMs", "timeIncrementMs", "metadata", "players"
        ]
        for field in optional_fields:
            assert field in data, f"Missing optional field: {field}"

    def test_move_record_schema(self, client, populated_db):
        """Test MoveRecord response schema."""
        response = client.get("/api/replay/games/game-hex8-2p-1/moves")
        assert response.status_code == 200

        data = response.json()
        move = data["moves"][0]

        # Required fields
        required_fields = [
            "moveNumber", "turnNumber", "player", "phase", "moveType", "move"
        ]
        for field in required_fields:
            assert field in move, f"Missing required field: {field}"

    def test_stats_response_schema(self, client, populated_db):
        """Test StatsResponse schema."""
        response = client.get("/api/replay/stats")
        assert response.status_code == 200

        data = response.json()

        required_fields = [
            "totalGames", "gamesByBoardType", "gamesByStatus",
            "gamesByTermination", "totalMoves", "schemaVersion"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check types
        assert isinstance(data["totalGames"], int)
        assert isinstance(data["gamesByBoardType"], dict)
        assert isinstance(data["schemaVersion"], int)
