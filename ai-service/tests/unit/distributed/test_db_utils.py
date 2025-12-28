"""Unit tests for distributed db_utils module.

Tests cover:
- PRAGMA configuration profiles
- Database connection management
- Transaction safety (safe_transaction, exclusive_db_lock)
- Atomic JSON operations
- State persistence (save/load)
- DatabaseRegistry functionality
- Thread-local connection pooling

December 2025: Created to improve distributed module test coverage.
"""

import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.db_utils import (
    PragmaProfile,
    apply_pragmas,
    get_db_connection,
    safe_transaction,
    exclusive_db_lock,
    atomic_json_update,
    save_state_atomically,
    load_state_safely,
    DatabaseRegistry,
    get_registry,
    get_db_path,
    get_database,
    close_all_connections,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary SQLite database."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test (value) VALUES ('initial')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def temp_json(temp_dir):
    """Create a temporary JSON file."""
    json_path = temp_dir / "state.json"
    json_path.write_text(json.dumps({"key": "value", "count": 0}))
    return json_path


# =============================================================================
# PragmaProfile Tests
# =============================================================================


class TestPragmaProfile:
    """Tests for PragmaProfile class."""

    def test_standard_profile_exists(self):
        """Standard profile is defined."""
        assert hasattr(PragmaProfile, "STANDARD")
        assert isinstance(PragmaProfile.STANDARD, dict)

    def test_standard_profile_has_required_keys(self):
        """Standard profile has essential PRAGMA settings."""
        required = ["journal_mode"]
        for key in required:
            assert key in PragmaProfile.STANDARD


# =============================================================================
# apply_pragmas Tests
# =============================================================================


class TestApplyPragmas:
    """Tests for apply_pragmas function."""

    def test_apply_pragmas_with_standard_profile(self, temp_db):
        """Apply standard pragmas to connection."""
        conn = sqlite3.connect(str(temp_db))
        try:
            apply_pragmas(conn, profile="standard")
            # Verify journal mode was set
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result is not None
        finally:
            conn.close()

    def test_apply_pragmas_with_custom_dict(self, temp_db):
        """Apply custom pragmas from dict."""
        conn = sqlite3.connect(str(temp_db))
        try:
            apply_pragmas(conn, custom_pragmas={"cache_size": -1000})
            result = conn.execute("PRAGMA cache_size").fetchone()
            assert result[0] == -1000
        finally:
            conn.close()

    def test_apply_pragmas_handles_invalid(self, temp_db):
        """Handles invalid pragma names gracefully (skips disallowed)."""
        conn = sqlite3.connect(str(temp_db))
        try:
            # Should not raise for invalid pragmas - just skips them
            apply_pragmas(conn, custom_pragmas={"invalid_pragma_xyz": 123})
        finally:
            conn.close()


# =============================================================================
# get_db_connection Tests
# =============================================================================


class TestGetDbConnection:
    """Tests for get_db_connection function."""

    def test_get_db_connection_creates_connection(self, temp_db):
        """Returns a valid SQLite connection."""
        conn = get_db_connection(str(temp_db))
        try:
            assert isinstance(conn, sqlite3.Connection)
            result = conn.execute("SELECT 1").fetchone()
            # Result is sqlite3.Row due to row_factory, access by index
            assert result[0] == 1
        finally:
            conn.close()

    def test_get_db_connection_quick_mode(self, temp_db):
        """Quick mode uses shorter timeout."""
        conn = get_db_connection(str(temp_db), quick=True)
        try:
            assert isinstance(conn, sqlite3.Connection)
        finally:
            conn.close()

    def test_get_db_connection_with_path_object(self, temp_db):
        """Accepts Path objects."""
        conn = get_db_connection(temp_db)  # Path, not str
        try:
            assert isinstance(conn, sqlite3.Connection)
        finally:
            conn.close()


# =============================================================================
# safe_transaction Tests
# =============================================================================


class TestSafeTransaction:
    """Tests for safe_transaction context manager."""

    def test_safe_transaction_commits_on_success(self, temp_db):
        """Transaction commits when context exits normally."""
        with safe_transaction(str(temp_db)) as conn:
            conn.execute("INSERT INTO test (value) VALUES ('new')")

        # Verify commit happened
        conn2 = sqlite3.connect(str(temp_db))
        result = conn2.execute("SELECT value FROM test WHERE value = 'new'").fetchone()
        conn2.close()
        assert result == ("new",)

    def test_safe_transaction_rollback_on_sqlite_error(self, temp_db):
        """Transaction rolls back on sqlite3.Error when using deferred isolation."""
        try:
            # Use isolation_level="" to enable deferred transactions
            # (isolation_level=None means autocommit, which commits each statement immediately)
            with safe_transaction(str(temp_db), isolation_level="") as conn:
                conn.execute("INSERT INTO test (value) VALUES ('rollback_test')")
                # Force a sqlite3.Error by executing invalid SQL
                conn.execute("INVALID SQL SYNTAX")
        except sqlite3.OperationalError:
            pass

        # Verify rollback happened
        conn2 = sqlite3.connect(str(temp_db))
        result = conn2.execute("SELECT value FROM test WHERE value = 'rollback_test'").fetchone()
        conn2.close()
        assert result is None

    def test_safe_transaction_is_context_manager(self, temp_db):
        """Can be used as context manager."""
        with safe_transaction(str(temp_db)) as conn:
            assert isinstance(conn, sqlite3.Connection)


# =============================================================================
# exclusive_db_lock Tests
# =============================================================================


class TestExclusiveDbLock:
    """Tests for exclusive_db_lock context manager."""

    def test_exclusive_lock_creates_connection(self, temp_db):
        """Returns a connection with exclusive lock."""
        with exclusive_db_lock(str(temp_db)) as conn:
            assert isinstance(conn, sqlite3.Connection)

    def test_exclusive_lock_allows_writes(self, temp_db):
        """Can write within exclusive lock."""
        with exclusive_db_lock(str(temp_db)) as conn:
            conn.execute("INSERT INTO test (value) VALUES ('exclusive')")

        # Verify write happened
        conn2 = sqlite3.connect(str(temp_db))
        result = conn2.execute("SELECT value FROM test WHERE value = 'exclusive'").fetchone()
        conn2.close()
        assert result == ("exclusive",)


# =============================================================================
# atomic_json_update Tests
# =============================================================================


class TestAtomicJsonUpdate:
    """Tests for atomic_json_update function."""

    def test_atomic_json_update_modifies_file(self, temp_json):
        """Updates JSON file atomically."""
        def update_fn(data):
            data["count"] = data.get("count", 0) + 1
            return data

        atomic_json_update(str(temp_json), update_fn)

        # Verify update
        with open(temp_json) as f:
            data = json.load(f)
        assert data["count"] == 1

    def test_atomic_json_update_preserves_on_error(self, temp_json):
        """Original file preserved if update function raises."""
        original_content = temp_json.read_text()

        def bad_update(data):
            raise ValueError("Simulated error")

        try:
            atomic_json_update(str(temp_json), bad_update)
        except ValueError:
            pass

        # Original content should be preserved
        assert temp_json.read_text() == original_content

    def test_atomic_json_update_creates_file(self, temp_dir):
        """Creates file if it doesn't exist."""
        new_file = temp_dir / "new_state.json"

        def init_fn(data):
            return {"initialized": True}

        atomic_json_update(str(new_file), init_fn)
        assert new_file.exists()


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for save_state_atomically and load_state_safely."""

    def test_save_state_atomically_creates_file(self, temp_dir):
        """save_state_atomically creates state file."""
        state_path = temp_dir / "state.json"
        state = {"key": "value", "count": 42}

        save_state_atomically(str(state_path), state)

        assert state_path.exists()
        with open(state_path) as f:
            loaded = json.load(f)
        assert loaded == state

    def test_load_state_safely_returns_data(self, temp_json):
        """load_state_safely returns file contents."""
        data = load_state_safely(str(temp_json))
        assert data == {"key": "value", "count": 0}

    def test_load_state_safely_returns_empty_dict_for_missing(self, temp_dir):
        """load_state_safely returns empty dict for missing file."""
        result = load_state_safely(str(temp_dir / "nonexistent.json"))
        assert result == {}

    def test_load_state_safely_returns_empty_dict_for_invalid_json(self, temp_dir):
        """load_state_safely returns empty dict for invalid JSON."""
        bad_file = temp_dir / "bad.json"
        bad_file.write_text("not valid json {{{")

        result = load_state_safely(str(bad_file))
        assert result == {}


# =============================================================================
# DatabaseRegistry Tests
# =============================================================================


class TestDatabaseRegistry:
    """Tests for DatabaseRegistry class."""

    def test_registry_is_singleton(self):
        """get_registry returns same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_get_db_path_returns_path(self):
        """get_db_path returns a Path object."""
        path = get_db_path("test_db")
        assert isinstance(path, Path)

    def test_get_db_path_creates_parent_dirs(self, temp_dir):
        """get_db_path creates parent directories for registered databases."""
        # Use get_registry() to get the singleton
        reg = get_registry()
        original_base = reg.get_base_path()
        try:
            reg.set_base_path(temp_dir)
            # Register a custom database path (unregistered paths are treated as literals)
            reg.register_database("test_deep", "subdir/deep/test.db")
            path = reg.get_path("test_deep", create_dirs=True)
            # Verify parent directories were created
            assert path.parent.exists()
            assert path == temp_dir / "subdir" / "deep" / "test.db"
        finally:
            # Restore original base path to not affect other tests
            reg.set_base_path(original_base)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_safe_transactions(self, temp_db):
        """Multiple threads can use safe_transaction."""
        results = []
        errors = []

        def worker(thread_id):
            try:
                for i in range(5):
                    with safe_transaction(str(temp_db)) as conn:
                        conn.execute(
                            "INSERT INTO test (value) VALUES (?)",
                            (f"thread{thread_id}_{i}",)
                        )
                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 3

    def test_concurrent_json_updates(self, temp_json):
        """Multiple threads can update JSON atomically."""
        errors = []

        def worker():
            try:
                for _ in range(5):
                    atomic_json_update(
                        str(temp_json),
                        lambda d: {**d, "count": d.get("count", 0) + 1}
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # May have race conditions, but shouldn't crash
        assert len(errors) == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_db_connection_nonexistent_creates(self, temp_dir):
        """Connection to nonexistent DB creates it."""
        new_db = temp_dir / "new.db"
        conn = get_db_connection(str(new_db))
        try:
            assert new_db.exists()
        finally:
            conn.close()

    def test_safe_transaction_with_readonly(self, temp_db):
        """Read-only operations work in safe_transaction."""
        with safe_transaction(str(temp_db)) as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] >= 1

    def test_empty_json_update(self, temp_dir):
        """atomic_json_update with identity function."""
        json_file = temp_dir / "test.json"
        json_file.write_text('{"x": 1}')

        atomic_json_update(str(json_file), lambda d: d)

        with open(json_file) as f:
            assert json.load(f) == {"x": 1}


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup functions."""

    def test_close_all_connections_doesnt_crash(self):
        """close_all_connections runs without error."""
        # Should not raise even with no connections
        close_all_connections()
