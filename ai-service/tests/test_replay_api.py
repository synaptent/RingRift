"""Integration tests for the game replay REST API endpoints.

Tests the HTTP-level API defined in app/routes/replay.py, covering:
- GET /api/replay/games - list games with filters
- GET /api/replay/games/{game_id} - get game details with player info
- GET /api/replay/games/{game_id}/state - get reconstructed state at move
- GET /api/replay/games/{game_id}/moves - get move records
- GET /api/replay/games/{game_id}/choices - get player choices
- GET /api/replay/stats - get database statistics
- POST /api/replay/games - store game from sandbox

Uses an in-memory test database for isolation.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime

from fastapi.testclient import TestClient
from fastapi.encoders import jsonable_encoder

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    GameState,
    BoardState,
    BoardType,
    GamePhase,
    GameStatus,
    TimeControl,
    Player,
    Move,
    MoveType,
    Position,
)
from app.routes import replay  # noqa: E402
from app.db.game_replay import GameReplayDB  # noqa: E402


def make_test_game_state(
    game_id: str = "test-game-1",
    board_type: BoardType = BoardType.SQUARE8,
    current_player: int = 1,
    move_number: int = 0,
) -> GameState:
    """Create a minimal GameState for testing."""
    return GameState(
        id=game_id,
        boardType=board_type,
        board=BoardState(type=board_type, size=8),
        players=[
            Player(
                id="p1",
                username="Player1",
                type="ai",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=19,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=5,
            ),
            Player(
                id="p2",
                username="Player2",
                type="ai",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=19,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=5,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=current_player,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=38,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
    )


_move_counter = 0


def make_test_move(
    player: int = 1,
    position: dict = None,
    move_type: MoveType = MoveType.PLACE_RING,
) -> Move:
    """Create a minimal Move for testing."""
    global _move_counter
    _move_counter += 1
    pos = position or {"x": 0, "y": 0}
    return Move(
        id=f"test-move-{_move_counter}",
        type=move_type,
        player=player,
        to=Position(x=pos["x"], y=pos["y"]),
        timestamp=datetime.now(),
        thinkTime=100,
        moveNumber=_move_counter,
    )


class TestReplayAPIEndpoints(unittest.TestCase):
    """Integration tests for /api/replay/* endpoints."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary database for all tests in this class."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_replay.db")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up fresh database and client for each test."""
        global _move_counter
        _move_counter = 0

        # Reset the DB singleton and set environment variable
        replay.reset_replay_db()
        os.environ["GAME_REPLAY_DB_PATH"] = self.test_db_path

        # Remove existing test database to start fresh
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.client = TestClient(app)

        # Pre-populate with a test game
        self._seed_test_game()

    def tearDown(self):
        """Clean up after each test."""
        replay.reset_replay_db()

    def _seed_test_game(self):
        """Seed the database with a test game."""
        db = replay.get_replay_db()

        # Create initial state
        initial_state = make_test_game_state("seed-game-1")

        # Create final state (simulating game completion)
        final_state = make_test_game_state("seed-game-1")
        final_state.game_status = GameStatus.COMPLETED
        final_state.players[0].eliminated_rings = 0
        final_state.players[1].eliminated_rings = 19

        # Create moves
        moves = [
            make_test_move(1, {"x": 0, "y": 0}),
            make_test_move(2, {"x": 1, "y": 0}),
            make_test_move(1, {"x": 0, "y": 1}),
            make_test_move(2, {"x": 1, "y": 1}),
            make_test_move(1, {"x": 0, "y": 2}),
        ]

        # Store the game
        db.store_game(
            game_id="seed-game-1",
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            choices=None,
            metadata={"source": "test", "winner": 1, "termination_reason": "elimination"},
        )

    # =========================================================================
    # GET /api/replay/games - List Games
    # =========================================================================

    def test_list_games_returns_seeded_game(self):
        """GET /api/replay/games should return the seeded test game."""
        response = self.client.get("/api/replay/games")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("games", body)
        self.assertIn("total", body)
        self.assertIn("hasMore", body)

        self.assertGreaterEqual(body["total"], 1)
        self.assertGreaterEqual(len(body["games"]), 1)

        # Find our seeded game
        game_ids = [g["gameId"] for g in body["games"]]
        self.assertIn("seed-game-1", game_ids)

    def test_list_games_filter_by_board_type(self):
        """Filter by board_type should work correctly."""
        response = self.client.get("/api/replay/games", params={"board_type": "square8"})

        self.assertEqual(response.status_code, 200)
        body = response.json()

        for game in body["games"]:
            self.assertEqual(game["boardType"], "square8")

    def test_list_games_filter_by_nonexistent_board_type(self):
        """Filter by non-existent board type returns empty list."""
        response = self.client.get("/api/replay/games", params={"board_type": "hexagonal"})

        self.assertEqual(response.status_code, 200)
        body = response.json()

        # Our seeded game is square8, so hexagonal filter should return nothing
        self.assertEqual(body["total"], 0)
        self.assertEqual(len(body["games"]), 0)

    def test_list_games_pagination(self):
        """Pagination with limit and offset should work."""
        response = self.client.get("/api/replay/games", params={"limit": 1, "offset": 0})

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertLessEqual(len(body["games"]), 1)

    def test_list_games_filter_by_winner(self):
        """Filter by winner should work."""
        # Our seeded game has winner=1
        response = self.client.get("/api/replay/games", params={"winner": 1})

        self.assertEqual(response.status_code, 200)
        body = response.json()

        for game in body["games"]:
            self.assertEqual(game["winner"], 1)

    # =========================================================================
    # GET /api/replay/games/{game_id} - Get Single Game
    # =========================================================================

    def test_get_game_returns_details(self):
        """GET /api/replay/games/{game_id} should return game details with players."""
        response = self.client.get("/api/replay/games/seed-game-1")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["gameId"], "seed-game-1")
        self.assertEqual(body["boardType"], "square8")
        self.assertEqual(body["numPlayers"], 2)
        self.assertIn("players", body)
        self.assertEqual(len(body["players"]), 2)

        # Check player details
        p1 = body["players"][0]
        self.assertEqual(p1["playerNumber"], 1)
        self.assertEqual(p1["playerType"], "ai")

    def test_get_game_not_found(self):
        """GET for non-existent game should return 404."""
        response = self.client.get("/api/replay/games/nonexistent-game")

        self.assertEqual(response.status_code, 404)

    # =========================================================================
    # GET /api/replay/games/{game_id}/state - Get State at Move
    # =========================================================================

    def test_get_state_at_move_zero(self):
        """GET /api/replay/games/{game_id}/state?move_number=0 should return initial state."""
        response = self.client.get(
            "/api/replay/games/seed-game-1/state",
            params={"move_number": 0}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("gameState", body)
        self.assertEqual(body["moveNumber"], 0)
        self.assertIn("totalMoves", body)

        # Verify the game state structure
        state = body["gameState"]
        self.assertIn("board", state)
        self.assertIn("players", state)

    def test_get_state_at_move_invalid_number(self):
        """GET with move_number exceeding total moves should return 400."""
        response = self.client.get(
            "/api/replay/games/seed-game-1/state",
            params={"move_number": 9999}
        )

        self.assertEqual(response.status_code, 400)

    def test_get_state_at_move_game_not_found(self):
        """GET state for non-existent game should return 404."""
        response = self.client.get(
            "/api/replay/games/nonexistent-game/state",
            params={"move_number": 0}
        )

        self.assertEqual(response.status_code, 404)

    # =========================================================================
    # GET /api/replay/games/{game_id}/moves - Get Move Records
    # =========================================================================

    def test_get_moves_returns_list(self):
        """GET /api/replay/games/{game_id}/moves should return move records."""
        response = self.client.get("/api/replay/games/seed-game-1/moves")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("moves", body)
        self.assertIn("hasMore", body)

        # We seeded 5 moves
        self.assertEqual(len(body["moves"]), 5)

        # Check move structure
        move = body["moves"][0]
        self.assertIn("moveNumber", move)
        self.assertIn("player", move)
        self.assertIn("moveType", move)
        self.assertIn("move", move)

    def test_get_moves_with_range(self):
        """GET moves with start/end range should work."""
        response = self.client.get(
            "/api/replay/games/seed-game-1/moves",
            params={"start": 1, "end": 3}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        # Should return moves 1 and 2 (start inclusive, end exclusive)
        self.assertLessEqual(len(body["moves"]), 2)

    def test_get_moves_game_not_found(self):
        """GET moves for non-existent game should return 404."""
        response = self.client.get("/api/replay/games/nonexistent-game/moves")

        self.assertEqual(response.status_code, 404)

    # =========================================================================
    # GET /api/replay/games/{game_id}/choices - Get Choices at Move
    # =========================================================================

    def test_get_choices_returns_empty_for_no_choices(self):
        """GET choices when none exist should return empty list."""
        response = self.client.get(
            "/api/replay/games/seed-game-1/choices",
            params={"move_number": 0}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("choices", body)
        self.assertEqual(body["choices"], [])

    def test_get_choices_game_not_found(self):
        """GET choices for non-existent game should return 404."""
        response = self.client.get(
            "/api/replay/games/nonexistent-game/choices",
            params={"move_number": 0}
        )

        self.assertEqual(response.status_code, 404)

    # =========================================================================
    # GET /api/replay/stats - Database Statistics
    # =========================================================================

    def test_get_stats_returns_statistics(self):
        """GET /api/replay/stats should return database statistics."""
        response = self.client.get("/api/replay/stats")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("totalGames", body)
        self.assertIn("gamesByBoardType", body)
        self.assertIn("gamesByStatus", body)
        self.assertIn("totalMoves", body)
        self.assertIn("schemaVersion", body)

        # We have at least one seeded game
        self.assertGreaterEqual(body["totalGames"], 1)
        self.assertGreaterEqual(body["totalMoves"], 5)

    # =========================================================================
    # POST /api/replay/games - Store Game from Sandbox
    # =========================================================================

    def test_store_game_creates_new_game(self):
        """POST /api/replay/games should store a new game from sandbox."""
        initial_state = make_test_game_state("post-test-game")
        final_state = make_test_game_state("post-test-game")
        final_state.game_status = GameStatus.COMPLETED

        moves = [
            make_test_move(1, {"x": 2, "y": 0}),
            make_test_move(2, {"x": 3, "y": 0}),
        ]

        request_body = {
            "gameId": "post-test-game",
            "initialState": jsonable_encoder(initial_state, by_alias=True),
            "finalState": jsonable_encoder(final_state, by_alias=True),
            "moves": [jsonable_encoder(m, by_alias=True) for m in moves],
            "metadata": {"source": "sandbox-test"},
        }

        response = self.client.post("/api/replay/games", json=request_body)

        self.assertEqual(
            response.status_code,
            200,
            msg=f"status={response.status_code}, body={response.text}",
        )
        body = response.json()

        self.assertEqual(body["gameId"], "post-test-game")
        self.assertEqual(body["totalMoves"], 2)
        self.assertTrue(body["success"])

        # Verify game was actually stored
        get_response = self.client.get("/api/replay/games/post-test-game")
        self.assertEqual(get_response.status_code, 200)

    def test_store_game_generates_id_if_not_provided(self):
        """POST without gameId should generate a UUID."""
        initial_state = make_test_game_state("temp-id")
        final_state = make_test_game_state("temp-id")

        request_body = {
            "initialState": jsonable_encoder(initial_state, by_alias=True),
            "finalState": jsonable_encoder(final_state, by_alias=True),
            "moves": [],
        }

        response = self.client.post("/api/replay/games", json=request_body)

        self.assertEqual(response.status_code, 200)
        body = response.json()

        # Should have a generated UUID
        self.assertIsNotNone(body["gameId"])
        self.assertNotEqual(body["gameId"], "")
        # UUID format check (basic)
        self.assertGreater(len(body["gameId"]), 20)

    def test_store_game_with_choices(self):
        """POST with choices should store them correctly."""
        initial_state = make_test_game_state("choice-test-game")
        final_state = make_test_game_state("choice-test-game")

        choices = [
            {
                "choice_type": "line_reward",
                "player": 1,
                "move_number": 0,
                "options": [{"action": "eliminate"}, {"action": "skip"}],
                "selected": {"action": "eliminate"},
            }
        ]

        request_body = {
            "gameId": "choice-test-game",
            "initialState": jsonable_encoder(initial_state, by_alias=True),
            "finalState": jsonable_encoder(final_state, by_alias=True),
            "moves": [],
            "choices": choices,
        }

        response = self.client.post("/api/replay/games", json=request_body)

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["gameId"], "choice-test-game")
        self.assertTrue(body["success"])

    def test_store_game_invalid_state(self):
        """POST with invalid state should return 500."""
        request_body = {
            "gameId": "invalid-test",
            "initialState": {"invalid": "state"},
            "finalState": {"invalid": "state"},
            "moves": [],
        }

        response = self.client.post("/api/replay/games", json=request_body)

        # Should fail validation
        self.assertEqual(response.status_code, 500)


class TestReplayAPIEdgeCases(unittest.TestCase):
    """Edge case and error handling tests."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.temp_dir, "test_replay_edge.db")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        replay.reset_replay_db()
        os.environ["GAME_REPLAY_DB_PATH"] = self.test_db_path

        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        self.client = TestClient(app)

    def tearDown(self):
        replay.reset_replay_db()

    def test_empty_database_list_games(self):
        """List games on empty database should return empty list."""
        response = self.client.get("/api/replay/games")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["total"], 0)
        self.assertEqual(body["games"], [])
        self.assertFalse(body["hasMore"])

    def test_empty_database_stats(self):
        """Stats on empty database should return zeros."""
        response = self.client.get("/api/replay/stats")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["totalGames"], 0)
        self.assertEqual(body["totalMoves"], 0)

    def test_list_games_invalid_limit(self):
        """Invalid limit parameter should be handled."""
        # FastAPI should validate and return 422 for out-of-range limit
        response = self.client.get("/api/replay/games", params={"limit": 1000})

        self.assertEqual(response.status_code, 422)

    def test_list_games_invalid_offset(self):
        """Negative offset should return 422."""
        response = self.client.get("/api/replay/games", params={"offset": -1})

        self.assertEqual(response.status_code, 422)


class TestGameReplayDBMigration(unittest.TestCase):
    """Tests for the GameReplayDB migration system."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_v1_to_v2_migration_creates_new_columns(self):
        """Opening a v1-style DB should auto-migrate to v2 schema."""
        import sqlite3

        # Create a v1-style database (no schema_metadata table)
        v1_db_path = os.path.join(self.temp_dir, "test_v1_migration.db")

        # Manually create a minimal v1 schema (without the v2 columns)
        conn = sqlite3.connect(v1_db_path)
        conn.row_factory = sqlite3.Row

        # Create v1 tables (subset of fields, no schema_metadata)
        conn.executescript("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                rng_seed INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                game_status TEXT NOT NULL,
                winner INTEGER,
                termination_reason TEXT,
                total_moves INTEGER NOT NULL,
                total_turns INTEGER NOT NULL,
                duration_ms INTEGER,
                source TEXT,
                schema_version INTEGER NOT NULL
            );

            CREATE TABLE game_players (
                game_id TEXT NOT NULL,
                player_number INTEGER NOT NULL,
                player_type TEXT NOT NULL,
                ai_type TEXT,
                ai_difficulty INTEGER,
                ai_profile_id TEXT,
                final_eliminated_rings INTEGER,
                final_territory_spaces INTEGER,
                final_rings_in_hand INTEGER,
                PRIMARY KEY (game_id, player_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            );

            CREATE TABLE game_initial_state (
                game_id TEXT PRIMARY KEY,
                initial_state_json TEXT NOT NULL,
                compressed INTEGER DEFAULT 0,
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            );

            CREATE TABLE game_moves (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                phase TEXT NOT NULL,
                move_type TEXT NOT NULL,
                move_json TEXT NOT NULL,
                timestamp TEXT,
                think_time_ms INTEGER,
                PRIMARY KEY (game_id, move_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            );

            CREATE TABLE game_state_snapshots (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                compressed INTEGER DEFAULT 0,
                PRIMARY KEY (game_id, move_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            );

            CREATE TABLE game_choices (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                choice_type TEXT NOT NULL,
                player INTEGER NOT NULL,
                options_json TEXT NOT NULL,
                selected_option_json TEXT NOT NULL,
                ai_reasoning TEXT,
                PRIMARY KEY (game_id, move_number, choice_type),
                FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
            );
        """)

        # Insert a test game (v1 style)
        conn.execute("""
            INSERT INTO games (game_id, board_type, num_players, created_at, game_status,
                               total_moves, total_turns, schema_version)
            VALUES ('v1-test-game', 'square8', 2, '2024-01-01T00:00:00', 'completed', 10, 5, 1)
        """)
        conn.commit()
        conn.close()

        # Now open with GameReplayDB - should auto-migrate
        db = GameReplayDB(v1_db_path)

        # Verify migration happened by checking schema_metadata table exists
        with db._get_conn() as check_conn:
            result = check_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
            ).fetchone()
            self.assertIsNotNone(result, "schema_metadata table should exist after migration")

            # Check schema version
            version_row = check_conn.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            self.assertEqual(version_row["value"], "4", "Schema version should be 4 after migration")

            # Check that v2 columns were added to games table
            columns = check_conn.execute("PRAGMA table_info(games)").fetchall()
            column_names = [col["name"] for col in columns]
            self.assertIn("time_control_type", column_names)
            self.assertIn("initial_time_ms", column_names)
            self.assertIn("time_increment_ms", column_names)

            # Check that v2 columns were added to game_moves table
            columns = check_conn.execute("PRAGMA table_info(game_moves)").fetchall()
            column_names = [col["name"] for col in columns]
            self.assertIn("time_remaining_ms", column_names)
            self.assertIn("engine_eval", column_names)
            self.assertIn("engine_depth", column_names)

        # Verify the existing game is still accessible
        stats = db.get_stats()
        self.assertEqual(stats["total_games"], 1)

    def test_fresh_database_gets_v2_schema(self):
        """A fresh database should get v2 schema directly."""
        fresh_db_path = os.path.join(self.temp_dir, "test_fresh_v2.db")

        # Create fresh database
        db = GameReplayDB(fresh_db_path)

        # Verify it has v2 schema
        with db._get_conn() as conn:
            version_row = conn.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            self.assertEqual(version_row["value"], "4", "Fresh DB should have schema version 4")

    def test_already_v2_database_no_migration(self):
        """Opening an already v2 database should not run migration again."""
        v2_db_path = os.path.join(self.temp_dir, "test_already_v2.db")

        # Create a fresh v2 database
        db1 = GameReplayDB(v2_db_path)

        # Store a test game
        initial_state = make_test_game_state("migration-test")
        final_state = make_test_game_state("migration-test")
        db1.store_game(
            game_id="migration-test",
            initial_state=initial_state,
            final_state=final_state,
            moves=[],
            choices=None,
            metadata={"source": "test"},
        )

        # Re-open the database
        db2 = GameReplayDB(v2_db_path)

        # Verify data is still there
        stats = db2.get_stats()
        self.assertEqual(stats["total_games"], 1)


if __name__ == "__main__":
    unittest.main()
