"""Tests for GameReplayDB schema migration.

Tests the migration from v1 to v2 schema, including:
- Fresh database creation with v2 schema
- Migration of existing v1 database
- Backwards compatibility with v1 data
- New v2 field storage and retrieval
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.db.game_replay import (
    DEFAULT_SNAPSHOT_INTERVAL,
    SCHEMA_VERSION,
    GameReplayDB,
)
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    TimeControl,
)


def create_test_game_state(num_players: int = 2) -> GameState:
    """Create a minimal valid GameState for testing."""
    players = [
        Player(
            id=f"player-{i + 1}",
            username=f"TestPlayer{i + 1}",
            player_number=i + 1,
            type="ai",
            is_ready=True,
            time_remaining=60000,
            rings_in_hand=5,
            eliminated_rings=0,
            territory_spaces=0,
        )
        for i in range(num_players)
    ]

    now = datetime.now(timezone.utc)
    return GameState(
        id="test-game",
        board_type=BoardType.SQUARE8,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
        ),
        players=players,
        current_player=1,
        current_phase=GamePhase.RING_PLACEMENT,
        game_status=GameStatus.ACTIVE,
        created_at=now,
        last_move_at=now,
        rng_seed=12345,
        victory_threshold=5,
        territory_victory_threshold=20,
        time_control=TimeControl(initial_time=60000, increment=0, type="none"),
        is_rated=False,
        max_players=num_players,
        total_rings_in_play=num_players * 5,
        total_rings_eliminated=0,
    )


def create_test_move(player: int = 1, move_number: int = 0) -> Move:
    """Create a minimal valid Move for testing."""
    return Move(
        id="test-move",
        player=player,
        type=MoveType.PLACE_RING,
        from_pos=None,
        to=Position(x=3, y=3),
        timestamp=datetime.now(timezone.utc),
        think_time=100,
        move_number=move_number,
    )


class TestFreshDatabaseCreation:
    """Tests for creating a fresh v2 database."""

    def test_creates_v2_schema_on_fresh_db(self, tmp_path: Path):
        """Fresh database should have v2 schema."""
        db_path = tmp_path / "fresh.db"
        db = GameReplayDB(str(db_path))

        # Verify schema version
        stats = db.get_stats()
        assert stats["schema_version"] == 4  # Updated from 2 for v4 schema

    def test_creates_schema_metadata_table(self, tmp_path: Path):
        """Fresh database should have schema_metadata table."""
        db_path = tmp_path / "fresh.db"
        GameReplayDB(str(db_path))

        # Check raw SQL
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()

        assert "schema_metadata" in table_names

    def test_v2_columns_exist_in_games_table(self, tmp_path: Path):
        """Games table should have v2 columns."""
        db_path = tmp_path / "fresh.db"
        GameReplayDB(str(db_path))

        conn = sqlite3.connect(str(db_path))
        columns = conn.execute("PRAGMA table_info(games)").fetchall()
        column_names = [c[1] for c in columns]
        conn.close()

        assert "time_control_type" in column_names
        assert "initial_time_ms" in column_names
        assert "time_increment_ms" in column_names

    def test_v2_columns_exist_in_game_moves_table(self, tmp_path: Path):
        """game_moves table should have v2 columns."""
        db_path = tmp_path / "fresh.db"
        GameReplayDB(str(db_path))

        conn = sqlite3.connect(str(db_path))
        columns = conn.execute("PRAGMA table_info(game_moves)").fetchall()
        column_names = [c[1] for c in columns]
        conn.close()

        v2_columns = [
            "time_remaining_ms",
            "engine_eval",
            "engine_eval_type",
            "engine_depth",
            "engine_nodes",
            "engine_pv",
            "engine_time_ms",
        ]
        for col in v2_columns:
            assert col in column_names, f"Missing column: {col}"


class TestV1ToV2Migration:
    """Tests for migrating v1 schema to v2."""

    def create_v1_database(self, db_path: str) -> None:
        """Create a v1 schema database with some test data."""
        conn = sqlite3.connect(db_path)

        # Create v1 schema (without schema_metadata table)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
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

            CREATE TABLE IF NOT EXISTS game_players (
                game_id TEXT NOT NULL,
                player_number INTEGER NOT NULL,
                player_type TEXT NOT NULL,
                ai_type TEXT,
                ai_difficulty INTEGER,
                ai_profile_id TEXT,
                final_eliminated_rings INTEGER,
                final_territory_spaces INTEGER,
                final_rings_in_hand INTEGER,
                PRIMARY KEY (game_id, player_number)
            );

            CREATE TABLE IF NOT EXISTS game_initial_state (
                game_id TEXT PRIMARY KEY,
                initial_state_json TEXT NOT NULL,
                compressed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS game_moves (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                phase TEXT NOT NULL,
                move_type TEXT NOT NULL,
                move_json TEXT NOT NULL,
                timestamp TEXT,
                think_time_ms INTEGER,
                PRIMARY KEY (game_id, move_number)
            );

            CREATE TABLE IF NOT EXISTS game_state_snapshots (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                compressed INTEGER DEFAULT 0,
                PRIMARY KEY (game_id, move_number)
            );

            CREATE TABLE IF NOT EXISTS game_choices (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                choice_type TEXT NOT NULL,
                player INTEGER NOT NULL,
                options_json TEXT NOT NULL,
                selected_option_json TEXT NOT NULL,
                ai_reasoning TEXT,
                PRIMARY KEY (game_id, move_number, choice_type)
            );
            """
        )

        # Insert some test data
        conn.execute(
            """
            INSERT INTO games
            (game_id, board_type, num_players, created_at, game_status,
             total_moves, total_turns, source, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-v1-game",
                "square8",
                2,
                "2025-01-01T00:00:00Z",
                "completed",
                10,
                5,
                "test",
                1,
            ),
        )

        conn.execute(
            """
            INSERT INTO game_moves
            (game_id, move_number, turn_number, player, phase, move_type, move_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-v1-game",
                0,
                0,
                1,
                "placement",
                "placement",
                '{"type": "placement", "player": 1}',
            ),
        )

        conn.commit()
        conn.close()

    def test_migrates_v1_to_v2_on_open(self, tmp_path: Path):
        """Opening v1 database should trigger migration to v2."""
        db_path = tmp_path / "v1.db"
        self.create_v1_database(str(db_path))

        # Open with GameReplayDB (should trigger migration)
        db = GameReplayDB(str(db_path))

        # Verify schema version is now at latest (v4)
        stats = db.get_stats()
        assert stats["schema_version"] == 4  # Migrates through v2, v3 to v4

    def test_preserves_existing_data_after_migration(self, tmp_path: Path):
        """Migration should preserve existing game data."""
        db_path = tmp_path / "v1.db"
        self.create_v1_database(str(db_path))

        db = GameReplayDB(str(db_path))

        # Check original game still exists
        games = db.query_games()
        assert len(games) == 1
        assert games[0]["game_id"] == "test-v1-game"
        assert games[0]["total_moves"] == 10

    def test_adds_v2_columns_to_existing_tables(self, tmp_path: Path):
        """Migration should add v2 columns to existing tables."""
        db_path = tmp_path / "v1.db"
        self.create_v1_database(str(db_path))

        GameReplayDB(str(db_path))

        conn = sqlite3.connect(str(db_path))

        # Check games table
        game_columns = conn.execute("PRAGMA table_info(games)").fetchall()
        game_column_names = [c[1] for c in game_columns]
        assert "time_control_type" in game_column_names

        # Check game_moves table
        move_columns = conn.execute("PRAGMA table_info(game_moves)").fetchall()
        move_column_names = [c[1] for c in move_columns]
        assert "engine_eval" in move_column_names

        conn.close()

    def test_existing_moves_have_null_v2_fields(self, tmp_path: Path):
        """Existing moves should have NULL for new v2 fields."""
        db_path = tmp_path / "v1.db"
        self.create_v1_database(str(db_path))

        db = GameReplayDB(str(db_path))

        # Get move records (includes v2 fields)
        moves = db.get_move_records("test-v1-game")

        assert len(moves) == 1
        assert moves[0]["engineEval"] is None
        assert moves[0]["enginePV"] is None
        assert moves[0]["timeRemainingMs"] is None


class TestV2FieldStorage:
    """Tests for storing and retrieving v2 fields."""

    def test_stores_engine_eval_with_move(self, tmp_path: Path):
        """Should store and retrieve engine evaluation with moves."""
        db_path = tmp_path / "v2.db"
        db = GameReplayDB(str(db_path))

        initial_state = create_test_game_state()
        final_state = create_test_game_state()
        final_state.game_status = GameStatus.COMPLETED
        move = create_test_move()

        # Store game
        db.store_game(
            game_id="eval-test",
            initial_state=initial_state,
            final_state=final_state,
            moves=[move],
            metadata={"source": "test"},
        )

        # Now manually update the move to add engine eval
        # (since store_game doesn't yet pass through v2 fields)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            UPDATE game_moves
            SET engine_eval = ?, engine_eval_type = ?, engine_pv = ?
            WHERE game_id = ? AND move_number = ?
            """,
            (0.75, "heuristic", '["e2e4", "e7e5"]', "eval-test", 0),
        )
        conn.commit()
        conn.close()

        # Retrieve and verify
        moves = db.get_move_records("eval-test")
        assert len(moves) == 1
        assert moves[0]["engineEval"] == pytest.approx(0.75)
        assert moves[0]["engineEvalType"] == "heuristic"
        assert moves[0]["enginePV"] == ["e2e4", "e7e5"]

    def test_get_game_count_with_filters(self, tmp_path: Path):
        """Should count games with various filters."""
        db_path = tmp_path / "count.db"
        db = GameReplayDB(str(db_path))

        # Create multiple games
        for i in range(5):
            state = create_test_game_state()
            state.id = f"game-{i}"
            final = create_test_game_state()
            final.game_status = GameStatus.COMPLETED
            final.winner = 1 if i < 3 else 2

            db.store_game(
                game_id=f"game-{i}",
                initial_state=state,
                final_state=final,
                moves=[create_test_move()],
                metadata={
                    "source": "test",
                    "termination_reason": "ring_elimination" if i < 3 else "territory",
                },
            )

        # Test counts
        assert db.get_game_count() == 5
        assert db.get_game_count(winner=1) == 3
        assert db.get_game_count(winner=2) == 2
        assert db.get_game_count(source="test") == 5

    def test_get_game_with_players(self, tmp_path: Path):
        """Should retrieve game with player details."""
        db_path = tmp_path / "players.db"
        db = GameReplayDB(str(db_path))

        state = create_test_game_state(num_players=3)
        final = create_test_game_state(num_players=3)
        final.game_status = GameStatus.COMPLETED
        final.winner = 2

        db.store_game(
            game_id="player-test",
            initial_state=state,
            final_state=final,
            moves=[create_test_move()],
            metadata={"source": "test"},
        )

        game = db.get_game_with_players("player-test")

        assert game is not None
        assert len(game["players"]) == 3
        assert game["players"][0]["playerNumber"] == 1
        assert game["players"][1]["playerNumber"] == 2
        assert game["players"][2]["playerNumber"] == 3


class TestMigrationIdempotence:
    """Tests that migration is idempotent."""

    def test_reopening_db_does_not_remigrate(self, tmp_path: Path):
        """Opening an already-migrated DB should not fail."""
        db_path = tmp_path / "idempotent.db"

        # Create and migrate
        db1 = GameReplayDB(str(db_path))
        stats1 = db1.get_stats()

        # Open again
        db2 = GameReplayDB(str(db_path))
        stats2 = db2.get_stats()

        assert stats1["schema_version"] == stats2["schema_version"] == 4  # Updated from 2 for v4 schema

    def test_handles_partial_migration_gracefully(self, tmp_path: Path):
        """Should handle case where some columns already exist."""
        db_path = tmp_path / "partial.db"

        # Create v1 schema with some v2 columns already added
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
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
                schema_version INTEGER NOT NULL,
                time_control_type TEXT DEFAULT 'none'
            );

            CREATE TABLE IF NOT EXISTS game_moves (
                game_id TEXT NOT NULL,
                move_number INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                player INTEGER NOT NULL,
                phase TEXT NOT NULL,
                move_type TEXT NOT NULL,
                move_json TEXT NOT NULL,
                timestamp TEXT,
                think_time_ms INTEGER,
                PRIMARY KEY (game_id, move_number)
            );
            """
        )
        conn.commit()
        conn.close()

        # Should handle gracefully
        db = GameReplayDB(str(db_path))
        stats = db.get_stats()
        assert stats["schema_version"] == 4  # Migrates to latest version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
