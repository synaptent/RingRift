"""Tests for scripts/lib/database.py module.

Tests cover:
- safe_transaction context manager
- read_only_connection context manager
- Database path utilities
- Game counting and integrity checks
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from scripts.lib.database import (
    DATA_DIR,
    GAMES_DIR,
    UNIFIED_ELO_DB,
    check_integrity,
    count_games,
    get_db_size_mb,
    get_elo_db_path,
    get_game_db_path,
    read_only_connection,
    safe_transaction,
    table_exists,
    vacuum_database,
)


class TestSafeTransaction:
    """Tests for safe_transaction context manager."""

    def test_commit_on_success(self, tmp_path):
        """Test that changes are committed on success."""
        db_path = tmp_path / "test.db"

        # Create table and insert data
        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'hello')")

        # Verify data persisted
        with safe_transaction(db_path) as conn:
            result = conn.execute("SELECT value FROM test WHERE id=1").fetchone()
            assert result[0] == "hello"

    def test_rollback_on_error(self, tmp_path):
        """Test that changes are rolled back on error."""
        db_path = tmp_path / "test.db"

        # Create table
        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'original')")

        # Try to update and fail
        with pytest.raises(sqlite3.IntegrityError), safe_transaction(db_path) as conn:
            conn.execute("UPDATE test SET value='updated' WHERE id=1")
            # This will fail due to duplicate primary key
            conn.execute("INSERT INTO test VALUES (1, 'duplicate')")

        # Verify original value is preserved (rollback worked)
        with safe_transaction(db_path) as conn:
            result = conn.execute("SELECT value FROM test WHERE id=1").fetchone()
            assert result[0] == "original"

    def test_row_factory(self, tmp_path):
        """Test that row_factory enables dict-like access."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'test')")

        with safe_transaction(db_path, row_factory=True) as conn:
            row = conn.execute("SELECT * FROM test").fetchone()
            assert row["id"] == 1
            assert row["name"] == "test"


class TestReadOnlyConnection:
    """Tests for read_only_connection context manager."""

    def test_can_read_data(self, tmp_path):
        """Test reading from existing database."""
        db_path = tmp_path / "test.db"

        # Create data first
        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (value TEXT)")
            conn.execute("INSERT INTO test VALUES ('data')")

        # Read with read-only connection
        with read_only_connection(db_path) as conn:
            result = conn.execute("SELECT value FROM test").fetchone()
            assert result["value"] == "data"


class TestPathUtilities:
    """Tests for database path utilities."""

    def test_get_game_db_path(self):
        """Test game database path construction."""
        path = get_game_db_path("square8_2p")
        assert path.name == "selfplay_square8_2p.db"
        assert "games" in str(path)

    def test_get_elo_db_path(self):
        """Test ELO database path."""
        path = get_elo_db_path()
        assert path.name == "unified_elo.db"

    def test_data_dir_exists(self):
        """Test DATA_DIR points to valid location."""
        assert DATA_DIR.name == "data"

    def test_games_dir(self):
        """Test GAMES_DIR is under DATA_DIR."""
        assert GAMES_DIR.parent == DATA_DIR


class TestCountGames:
    """Tests for count_games function."""

    def test_count_empty_db(self, tmp_path):
        """Test counting games in empty database."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE games (id INTEGER, config_key TEXT)")

        count = count_games(db_path)
        assert count == 0

    def test_count_with_games(self, tmp_path):
        """Test counting games with data."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE games (id INTEGER, config_key TEXT)")
            conn.execute("INSERT INTO games VALUES (1, 'square8_2p')")
            conn.execute("INSERT INTO games VALUES (2, 'square8_2p')")
            conn.execute("INSERT INTO games VALUES (3, 'hex7_3p')")

        assert count_games(db_path) == 3
        assert count_games(db_path, "square8_2p") == 2
        assert count_games(db_path, "hex7_3p") == 1

    def test_count_nonexistent_db(self, tmp_path):
        """Test counting games in nonexistent database."""
        db_path = tmp_path / "nonexistent.db"
        count = count_games(db_path)
        assert count == 0


class TestTableExists:
    """Tests for table_exists function."""

    def test_table_exists(self, tmp_path):
        """Test detecting existing table."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE existing_table (id INTEGER)")

        assert table_exists(db_path, "existing_table") is True
        assert table_exists(db_path, "nonexistent_table") is False

    def test_nonexistent_db(self, tmp_path):
        """Test table_exists on nonexistent database."""
        db_path = tmp_path / "nonexistent.db"
        assert table_exists(db_path, "any_table") is False


class TestDbSizeMb:
    """Tests for get_db_size_mb function."""

    def test_existing_file(self, tmp_path):
        """Test getting size of existing database."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (data TEXT)")
            # Insert some data to make file have measurable size
            for _i in range(100):
                conn.execute("INSERT INTO test VALUES (?)", ("x" * 1000,))

        size = get_db_size_mb(db_path)
        assert size > 0

    def test_nonexistent_file(self, tmp_path):
        """Test getting size of nonexistent file."""
        db_path = tmp_path / "nonexistent.db"
        size = get_db_size_mb(db_path)
        assert size == 0.0


class TestVacuumDatabase:
    """Tests for vacuum_database function."""

    def test_vacuum_success(self, tmp_path):
        """Test successful vacuum."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")

        result = vacuum_database(db_path)
        assert result is True

    def test_vacuum_creates_db(self, tmp_path):
        """Test vacuum creates database if it doesn't exist."""
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        result = vacuum_database(db_path)
        # SQLite creates the database when connecting
        assert result is True
        assert db_path.exists()


class TestCheckIntegrity:
    """Tests for check_integrity function."""

    def test_healthy_db(self, tmp_path):
        """Test integrity check on healthy database."""
        db_path = tmp_path / "test.db"

        with safe_transaction(db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")

        is_healthy, message = check_integrity(db_path)
        assert is_healthy is True
        assert message == "OK"

    def test_nonexistent_db(self, tmp_path):
        """Test integrity check on nonexistent database."""
        db_path = tmp_path / "nonexistent.db"
        is_healthy, message = check_integrity(db_path)
        assert is_healthy is False
        assert "error" in message.lower()
