"""Tests for game_replay.py - SQLite game storage and retrieval.

This module tests the GameReplayDB and GameWriter classes that handle
storing and retrieving complete RingRift games for training.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from app.db.game_replay import (
    DEFAULT_SNAPSHOT_INTERVAL,
    GameReplayDB,
    GameWriter,
    SCHEMA_VERSION,
)
from app.models import BoardType, GameState, Move, MoveType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_replay.db"


@pytest.fixture
def db(temp_db_path: Path) -> GameReplayDB:
    """Create a temporary GameReplayDB instance."""
    db = GameReplayDB(str(temp_db_path))
    yield db
    db.close()


@pytest.fixture
def sample_initial_state() -> GameState:
    """Create a sample initial game state for testing."""
    from app.training.initial_state import create_initial_state

    return create_initial_state(
        board_type=BoardType.HEX8,
        num_players=2,
    )


# =============================================================================
# GameReplayDB Initialization Tests
# =============================================================================


class TestGameReplayDBInit:
    """Tests for GameReplayDB initialization and lifecycle."""

    def test_init_creates_database(self, temp_db_path: Path) -> None:
        """Test that initialization creates the database file."""
        db = GameReplayDB(str(temp_db_path))
        try:
            assert temp_db_path.exists()
        finally:
            db.close()

    def test_init_creates_schema(self, temp_db_path: Path) -> None:
        """Test that initialization creates required tables."""
        db = GameReplayDB(str(temp_db_path))
        try:
            conn = sqlite3.connect(str(temp_db_path))
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            # Check for key tables
            assert "games" in tables
            assert "game_players" in tables
            assert "game_initial_state" in tables
            assert "game_moves" in tables
            assert "schema_metadata" in tables
        finally:
            db.close()

    def test_init_sets_schema_version(self, db: GameReplayDB) -> None:
        """Test that schema version is set correctly."""
        conn = sqlite3.connect(str(db._db_path))
        cursor = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert int(result[0]) == SCHEMA_VERSION

    def test_context_manager(self, temp_db_path: Path) -> None:
        """Test that GameReplayDB works as a context manager."""
        with GameReplayDB(str(temp_db_path)) as db:
            assert db is not None
            assert temp_db_path.exists()


# =============================================================================
# GameReplayDB Query Tests (Empty Database)
# =============================================================================


class TestGameReplayDBQueryEmpty:
    """Tests for querying empty databases."""

    def test_get_game_count_empty(self, db: GameReplayDB) -> None:
        """Test game count on empty database."""
        count = db.get_game_count()
        assert count == 0

    def test_query_games_empty(self, db: GameReplayDB) -> None:
        """Test query on empty database."""
        results = db.query_games()
        assert results == []

    def test_get_stats_empty(self, db: GameReplayDB) -> None:
        """Test stats on empty database."""
        stats = db.get_stats()
        assert stats["total_games"] == 0
        assert stats["total_moves"] == 0

    def test_get_nonexistent_game(self, db: GameReplayDB) -> None:
        """Test retrieving a game that doesn't exist."""
        metadata = db.get_game_metadata("nonexistent-game-id")
        assert metadata is None

        state = db.get_initial_state("nonexistent-game-id")
        assert state is None


# =============================================================================
# GameWriter Tests
# =============================================================================


class TestGameWriter:
    """Tests for GameWriter context manager."""

    def test_game_writer_creates_game_id(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test GameWriter generates a valid game_id."""
        with GameWriter(db, sample_initial_state, source="test") as writer:
            assert writer.game_id is not None
            # UUID format check
            try:
                uuid.UUID(writer.game_id)
            except ValueError:
                pytest.fail("game_id is not a valid UUID")

    def test_game_writer_abort(
        self, db: GameReplayDB, sample_initial_state: GameState
    ) -> None:
        """Test aborting a game via GameWriter removes it."""
        writer = GameWriter(db, sample_initial_state, source="test")
        writer.__enter__()
        game_id = writer.game_id
        writer.abort()

        # Game should be deleted
        metadata = db.get_game_metadata(game_id)
        assert metadata is None


# =============================================================================
# GameReplayDB Maintenance Tests
# =============================================================================


class TestGameReplayDBMaintenance:
    """Tests for database maintenance operations."""

    def test_vacuum(self, db: GameReplayDB) -> None:
        """Test VACUUM operation runs without error."""
        # VACUUM should run without error on empty DB
        db.vacuum()

    def test_checkpoint_wal(self, db: GameReplayDB) -> None:
        """Test WAL checkpoint operation."""
        # Checkpoint should run without error
        pages_checkpointed, pages_total = db.checkpoint_wal()
        assert pages_checkpointed >= 0
        assert pages_total >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_databases(self, tmp_path: Path) -> None:
        """Test working with multiple database instances."""
        db1_path = tmp_path / "db1.db"
        db2_path = tmp_path / "db2.db"

        with GameReplayDB(str(db1_path)) as db1:
            with GameReplayDB(str(db2_path)) as db2:
                # Both should work independently
                assert db1.get_game_count() == 0
                assert db2.get_game_count() == 0

    def test_close_is_idempotent(self, temp_db_path: Path) -> None:
        """Test that close() can be called multiple times."""
        db = GameReplayDB(str(temp_db_path))
        db.close()
        db.close()  # Should not raise

    def test_default_snapshot_interval(self) -> None:
        """Test default snapshot interval constant."""
        assert DEFAULT_SNAPSHOT_INTERVAL == 20


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemaMigrations:
    """Tests for schema migrations."""

    def test_fresh_database_has_latest_schema(self, temp_db_path: Path) -> None:
        """Test that fresh database has the latest schema version."""
        with GameReplayDB(str(temp_db_path)) as db:
            version = db._get_schema_version(db._get_conn())
            assert version == SCHEMA_VERSION

    def test_schema_includes_required_tables(self, temp_db_path: Path) -> None:
        """Test that schema includes all required tables."""
        with GameReplayDB(str(temp_db_path)) as db:
            conn = db._get_conn()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = {
                "games",
                "game_players",
                "game_initial_state",
                "game_moves",
                "game_state_snapshots",
                "game_choices",
                "game_history_entries",
                "game_nnue_features",
                "orphaned_games",
                "schema_metadata",
            }

            for table in required_tables:
                assert table in tables, f"Missing required table: {table}"

    def test_games_table_columns(self, temp_db_path: Path) -> None:
        """Test that games table has required columns."""
        with GameReplayDB(str(temp_db_path)) as db:
            conn = db._get_conn()
            cursor = conn.execute("PRAGMA table_info(games)")
            columns = {row[1] for row in cursor.fetchall()}

            required_columns = {
                "game_id",
                "board_type",
                "num_players",
                "created_at",
                "game_status",
                "total_moves",
                "total_turns",
                "source",
                "schema_version",
            }

            for col in required_columns:
                assert col in columns, f"Missing column in games: {col}"


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Tests for connection pooling and management."""

    def test_get_conn_returns_connection(self, db: GameReplayDB) -> None:
        """Test that _get_conn returns a valid connection."""
        conn = db._get_conn()
        assert conn is not None
        # Verify it's a valid SQLite connection
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone() == (1,)

    def test_connection_reuse(self, db: GameReplayDB) -> None:
        """Test that connections are reused within a session."""
        conn1 = db._get_conn()
        conn2 = db._get_conn()
        # Should be the same connection
        assert conn1 is conn2

    def test_wal_mode_enabled(self, temp_db_path: Path) -> None:
        """Test that WAL mode is enabled by default."""
        with GameReplayDB(str(temp_db_path)) as db:
            conn = db._get_conn()
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"
