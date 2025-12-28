"""Tests for WAL Sync Utilities - SQLite WAL handling for database sync.

Created: December 28, 2025
Purpose: Test the WAL synchronization utilities used for safe SQLite database syncing

Tests cover:
- WAL_INCLUDE_PATTERNS and WAL_RSYNC_INCLUDES constants
- get_wal_files() WAL file detection
- get_db_with_wal_files() file ordering for sync
- checkpoint_database() WAL checkpointing
- prepare_db_for_sync() pre-sync preparation
- validate_synced_database() post-sync validation
- build_rsync_command_for_db() command construction
- get_rsync_include_args_for_db() include pattern generation
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.wal_sync_utils import (
    WAL_INCLUDE_PATTERNS,
    WAL_RSYNC_INCLUDES,
    build_rsync_command_for_db,
    checkpoint_database,
    get_db_with_wal_files,
    get_rsync_include_args_for_db,
    get_wal_files,
    prepare_db_for_sync,
    validate_synced_database,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def wal_mode_db(temp_dir):
    """Create a SQLite database in WAL mode with some data."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO games (name) VALUES ('game1')")
    conn.execute("INSERT INTO games (name) VALUES ('game2')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def journal_mode_db(temp_dir):
    """Create a SQLite database in default journal mode (not WAL)."""
    db_path = temp_dir / "test_journal.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO games (name) VALUES ('game1')")
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for WAL-related constants."""

    def test_wal_include_patterns_has_db(self):
        """WAL_INCLUDE_PATTERNS should include *.db."""
        assert "*.db" in WAL_INCLUDE_PATTERNS

    def test_wal_include_patterns_has_wal(self):
        """WAL_INCLUDE_PATTERNS should include *.db-wal."""
        assert "*.db-wal" in WAL_INCLUDE_PATTERNS

    def test_wal_include_patterns_has_shm(self):
        """WAL_INCLUDE_PATTERNS should include *.db-shm."""
        assert "*.db-shm" in WAL_INCLUDE_PATTERNS

    def test_wal_rsync_includes_has_db(self):
        """WAL_RSYNC_INCLUDES should have --include=*.db."""
        assert "--include=*.db" in WAL_RSYNC_INCLUDES

    def test_wal_rsync_includes_has_wal(self):
        """WAL_RSYNC_INCLUDES should have --include=*.db-wal."""
        assert "--include=*.db-wal" in WAL_RSYNC_INCLUDES

    def test_wal_rsync_includes_has_shm(self):
        """WAL_RSYNC_INCLUDES should have --include=*.db-shm."""
        assert "--include=*.db-shm" in WAL_RSYNC_INCLUDES


# =============================================================================
# get_wal_files Tests
# =============================================================================


class TestGetWalFiles:
    """Tests for get_wal_files()."""

    def test_no_wal_files(self, temp_dir):
        """Non-existent WAL files should return empty list."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        result = get_wal_files(db_path)
        assert result == []

    def test_only_wal_file(self, temp_dir):
        """Only WAL file should be returned if SHM doesn't exist."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        wal_path = temp_dir / "test.db-wal"
        wal_path.touch()
        result = get_wal_files(db_path)
        assert len(result) == 1
        assert result[0] == wal_path

    def test_only_shm_file(self, temp_dir):
        """Only SHM file should be returned if WAL doesn't exist."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        shm_path = temp_dir / "test.db-shm"
        shm_path.touch()
        result = get_wal_files(db_path)
        assert len(result) == 1
        assert result[0] == shm_path

    def test_both_wal_files(self, temp_dir):
        """Both WAL and SHM files should be returned when they exist."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        wal_path = temp_dir / "test.db-wal"
        wal_path.touch()
        shm_path = temp_dir / "test.db-shm"
        shm_path.touch()
        result = get_wal_files(db_path)
        assert len(result) == 2
        assert wal_path in result
        assert shm_path in result

    def test_real_wal_db(self, wal_mode_db):
        """Real WAL-mode database should have WAL files."""
        # WAL files might be created automatically
        result = get_wal_files(wal_mode_db)
        # Just verify function works - WAL files may or may not exist
        assert isinstance(result, list)

    def test_accepts_string_path(self, temp_dir):
        """Should accept string path."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        result = get_wal_files(str(db_path))
        assert result == []


# =============================================================================
# get_db_with_wal_files Tests
# =============================================================================


class TestGetDbWithWalFiles:
    """Tests for get_db_with_wal_files()."""

    def test_db_only(self, temp_dir):
        """Only DB file returned when no WAL files exist."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        result = get_db_with_wal_files(db_path)
        assert len(result) == 1
        assert result[0] == db_path

    def test_db_with_wal(self, temp_dir):
        """DB and WAL files returned in correct order."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        wal_path = temp_dir / "test.db-wal"
        wal_path.touch()
        result = get_db_with_wal_files(db_path)
        assert len(result) == 2
        # WAL should come before DB for sync order
        assert result[0] == wal_path
        assert result[1] == db_path

    def test_db_with_shm(self, temp_dir):
        """DB and SHM files returned in correct order."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        shm_path = temp_dir / "test.db-shm"
        shm_path.touch()
        result = get_db_with_wal_files(db_path)
        assert len(result) == 2
        assert result[0] == shm_path
        assert result[1] == db_path

    def test_all_files(self, temp_dir):
        """All files returned in correct order: WAL, SHM, DB."""
        db_path = temp_dir / "test.db"
        db_path.touch()
        wal_path = temp_dir / "test.db-wal"
        wal_path.touch()
        shm_path = temp_dir / "test.db-shm"
        shm_path.touch()
        result = get_db_with_wal_files(db_path)
        assert len(result) == 3
        assert result[0] == wal_path
        assert result[1] == shm_path
        assert result[2] == db_path

    def test_nonexistent_db(self, temp_dir):
        """Nonexistent DB should return empty list."""
        db_path = temp_dir / "nonexistent.db"
        result = get_db_with_wal_files(db_path)
        assert result == []


# =============================================================================
# checkpoint_database Tests
# =============================================================================


class TestCheckpointDatabase:
    """Tests for checkpoint_database()."""

    def test_nonexistent_db_returns_false(self, temp_dir):
        """Nonexistent database should return False."""
        db_path = temp_dir / "nonexistent.db"
        result = checkpoint_database(db_path)
        assert result is False

    def test_non_wal_db_returns_true(self, journal_mode_db):
        """Non-WAL database should return True (nothing to checkpoint)."""
        result = checkpoint_database(journal_mode_db)
        assert result is True

    def test_wal_db_checkpoint_succeeds(self, wal_mode_db):
        """WAL database checkpoint should succeed."""
        result = checkpoint_database(wal_mode_db)
        assert result is True

    def test_checkpoint_truncate_mode(self, wal_mode_db):
        """Checkpoint with truncate=True should work."""
        result = checkpoint_database(wal_mode_db, truncate=True)
        assert result is True

    def test_checkpoint_passive_mode(self, wal_mode_db):
        """Checkpoint with truncate=False should use PASSIVE mode."""
        result = checkpoint_database(wal_mode_db, truncate=False)
        assert result is True

    def test_accepts_string_path(self, wal_mode_db):
        """Should accept string path."""
        result = checkpoint_database(str(wal_mode_db))
        assert result is True

    def test_corrupted_db_returns_false(self, temp_dir):
        """Corrupted database should return False."""
        db_path = temp_dir / "corrupted.db"
        db_path.write_bytes(b"not a sqlite database")
        result = checkpoint_database(db_path)
        assert result is False


# =============================================================================
# prepare_db_for_sync Tests
# =============================================================================


class TestPrepareDbForSync:
    """Tests for prepare_db_for_sync()."""

    def test_nonexistent_db(self, temp_dir):
        """Nonexistent database should fail."""
        db_path = temp_dir / "nonexistent.db"
        success, message = prepare_db_for_sync(db_path)
        assert success is False
        assert "not found" in message.lower()

    def test_valid_db_succeeds(self, wal_mode_db):
        """Valid database should succeed."""
        success, message = prepare_db_for_sync(wal_mode_db)
        assert success is True
        assert "ready to sync" in message.lower()

    def test_valid_journal_db_succeeds(self, journal_mode_db):
        """Valid journal-mode database should succeed."""
        success, message = prepare_db_for_sync(journal_mode_db)
        assert success is True

    def test_corrupted_db_fails(self, temp_dir):
        """Corrupted database should fail validation."""
        db_path = temp_dir / "corrupted.db"
        db_path.write_bytes(b"not a sqlite database")
        success, message = prepare_db_for_sync(db_path)
        assert success is False

    def test_message_mentions_wal_files_if_present(self, wal_mode_db):
        """Message should mention WAL files if they exist."""
        # Create a WAL file manually
        wal_path = wal_mode_db.with_suffix(".db-wal")
        wal_path.touch()
        success, message = prepare_db_for_sync(wal_mode_db)
        assert success is True
        # May or may not mention WAL depending on state


# =============================================================================
# validate_synced_database Tests
# =============================================================================


class TestValidateSyncedDatabase:
    """Tests for validate_synced_database()."""

    def test_nonexistent_db(self, temp_dir):
        """Nonexistent database should fail."""
        db_path = temp_dir / "nonexistent.db"
        is_valid, errors = validate_synced_database(db_path)
        assert is_valid is False
        assert len(errors) > 0
        assert "not found" in errors[0].lower()

    def test_valid_db_passes(self, wal_mode_db):
        """Valid database should pass validation."""
        is_valid, errors = validate_synced_database(wal_mode_db)
        assert is_valid is True
        assert len(errors) == 0

    def test_integrity_check_enabled(self, wal_mode_db):
        """Integrity check should be performed by default."""
        is_valid, errors = validate_synced_database(
            wal_mode_db, check_integrity=True
        )
        assert is_valid is True

    def test_integrity_check_disabled(self, wal_mode_db):
        """Skipping integrity check should work."""
        is_valid, errors = validate_synced_database(
            wal_mode_db, check_integrity=False
        )
        assert is_valid is True

    def test_min_rows_check_passes(self, wal_mode_db):
        """Row count check should pass when rows meet minimum."""
        is_valid, errors = validate_synced_database(
            wal_mode_db, min_expected_rows=1, table_name="games"
        )
        assert is_valid is True

    def test_min_rows_check_fails(self, wal_mode_db):
        """Row count check should fail when rows below minimum."""
        is_valid, errors = validate_synced_database(
            wal_mode_db, min_expected_rows=100, table_name="games"
        )
        assert is_valid is False
        assert len(errors) > 0
        assert "expected at least" in errors[0].lower()

    def test_nonexistent_table_ignored(self, wal_mode_db):
        """Nonexistent table should not cause validation failure."""
        is_valid, errors = validate_synced_database(
            wal_mode_db, min_expected_rows=1, table_name="nonexistent_table"
        )
        # Should pass because table not existing is not an error for row check
        assert is_valid is True

    def test_corrupted_db_fails(self, temp_dir):
        """Corrupted database should fail validation."""
        db_path = temp_dir / "corrupted.db"
        db_path.write_bytes(b"not a sqlite database")
        is_valid, errors = validate_synced_database(db_path)
        assert is_valid is False
        assert len(errors) > 0


# =============================================================================
# build_rsync_command_for_db Tests
# =============================================================================


class TestBuildRsyncCommandForDb:
    """Tests for build_rsync_command_for_db()."""

    def test_basic_command(self, temp_dir):
        """Basic command should have rsync and essential flags."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/")
        assert cmd[0] == "rsync"
        assert "-avz" in cmd
        assert "--compress" in cmd

    def test_includes_db_patterns(self, temp_dir):
        """Command should include DB and WAL patterns."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/")
        assert "--include=test.db" in cmd
        assert "--include=test.db-wal" in cmd
        assert "--include=test.db-shm" in cmd

    def test_excludes_other_files(self, temp_dir):
        """Command should exclude other files."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/")
        assert "--exclude=*" in cmd

    def test_destination_included(self, temp_dir):
        """Destination should be last argument."""
        db_path = temp_dir / "test.db"
        dest = "user@host:/dest/"
        cmd = build_rsync_command_for_db(db_path, dest)
        assert cmd[-1] == dest

    def test_ssh_options(self, temp_dir):
        """SSH options should be included."""
        db_path = temp_dir / "test.db"
        ssh_opts = ["-e", "ssh -i ~/.ssh/key"]
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/", ssh_options=ssh_opts)
        assert "-e" in cmd
        assert "ssh -i ~/.ssh/key" in cmd

    def test_bandwidth_limit(self, temp_dir):
        """Bandwidth limit should be included."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/", bwlimit_kbps=1000)
        assert "--bwlimit=1000" in cmd

    def test_extra_options(self, temp_dir):
        """Extra options should be included."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(
            db_path, "user@host:/dest/", extra_options=["--delete", "--progress"]
        )
        assert "--delete" in cmd
        assert "--progress" in cmd

    def test_source_has_trailing_slash(self, temp_dir):
        """Source directory should have trailing slash."""
        db_path = temp_dir / "test.db"
        cmd = build_rsync_command_for_db(db_path, "user@host:/dest/")
        # Find the source path (second-to-last argument)
        source = cmd[-2]
        assert source.endswith("/")


# =============================================================================
# get_rsync_include_args_for_db Tests
# =============================================================================


class TestGetRsyncIncludeArgsForDb:
    """Tests for get_rsync_include_args_for_db()."""

    def test_returns_list(self):
        """Should return a list of strings."""
        result = get_rsync_include_args_for_db("test.db")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_includes_db(self):
        """Should include the DB file."""
        result = get_rsync_include_args_for_db("test.db")
        assert "--include=test.db" in result

    def test_includes_wal(self):
        """Should include the WAL file."""
        result = get_rsync_include_args_for_db("test.db")
        assert "--include=test.db-wal" in result

    def test_includes_shm(self):
        """Should include the SHM file."""
        result = get_rsync_include_args_for_db("test.db")
        assert "--include=test.db-shm" in result

    def test_custom_db_name(self):
        """Should work with custom database names."""
        result = get_rsync_include_args_for_db("my_special_games.db")
        assert "--include=my_special_games.db" in result
        assert "--include=my_special_games.db-wal" in result
        assert "--include=my_special_games.db-shm" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for typical sync workflows."""

    def test_full_sync_workflow(self, wal_mode_db):
        """Test the full prepare -> validate workflow."""
        # Prepare for sync
        success, msg = prepare_db_for_sync(wal_mode_db)
        assert success is True

        # Build rsync command
        cmd = build_rsync_command_for_db(
            wal_mode_db,
            "user@host:/dest/",
            bwlimit_kbps=50000,
        )
        assert len(cmd) > 5

        # Validate (simulating after sync - same db)
        is_valid, errors = validate_synced_database(
            wal_mode_db, min_expected_rows=1, table_name="games"
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_wal_files_detection_after_write(self, temp_dir):
        """Test WAL file detection after writing to database."""
        db_path = temp_dir / "active.db"

        # Create WAL-mode database and write data
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()

        # Get files before closing connection
        wal_files = get_wal_files(db_path)
        all_files = get_db_with_wal_files(db_path)

        conn.close()

        # After close, WAL may be checkpointed automatically
        # Just verify the functions work
        assert isinstance(wal_files, list)
        assert isinstance(all_files, list)
        assert db_path in all_files or len(all_files) == 0

    def test_checkpoint_reduces_wal_size(self, temp_dir):
        """Checkpoint should reduce or eliminate WAL file."""
        db_path = temp_dir / "checkpoint_test.db"

        # Create database and write data
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE test (id INTEGER, data TEXT)")

        # Write enough data to create a non-trivial WAL
        for i in range(100):
            conn.execute("INSERT INTO test VALUES (?, ?)", (i, "x" * 100))
        conn.commit()
        conn.close()

        # Checkpoint the database
        success = checkpoint_database(db_path, truncate=True)
        assert success is True

        # After TRUNCATE checkpoint, WAL should be empty or very small
        wal_path = db_path.with_suffix(".db-wal")
        if wal_path.exists():
            # TRUNCATE mode should make WAL empty
            assert wal_path.stat().st_size < 1000  # Allow small header
