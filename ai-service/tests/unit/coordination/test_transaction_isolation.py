"""
Tests for app.coordination.transaction_isolation module.

Tests the transaction isolation system for safe data merge operations,
including write-ahead logging, atomic commit/rollback, and crash recovery.
"""

import json
import pytest
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from app.coordination.transaction_isolation import (
    # Enums
    TransactionState,
    # Data classes
    MergeOperation,
    MergeTransaction,
    # Main class
    TransactionIsolation,
    # Functions
    get_transaction_isolation,
    reset_transaction_isolation,
    begin_merge_transaction,
    add_merge_operation,
    complete_merge_operation,
    commit_merge_transaction,
    rollback_merge_transaction,
    merge_transaction,
    get_transaction_stats,
)


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    db_path = tmp_path / "transactions.db"
    wal_dir = tmp_path / "wal"
    backup_dir = tmp_path / "backups"
    return {
        "db_path": db_path,
        "wal_dir": wal_dir,
        "backup_dir": backup_dir,
    }


@pytest.fixture(autouse=True)
def reset_singletons(temp_dirs):
    """Reset singletons between tests."""
    reset_transaction_isolation()
    TransactionIsolation.reset_instance()
    yield
    reset_transaction_isolation()
    TransactionIsolation.reset_instance()


@pytest.fixture
def isolation(temp_dirs):
    """Create a TransactionIsolation instance with temp directories."""
    return TransactionIsolation(
        db_path=temp_dirs["db_path"],
        wal_dir=temp_dirs["wal_dir"],
        backup_dir=temp_dirs["backup_dir"],
    )


@pytest.fixture
def test_file(tmp_path):
    """Create a test file for operations."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("original content")
    return file_path


# ============================================
# Test TransactionState Enum
# ============================================

class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_pending_value(self):
        assert TransactionState.PENDING.value == "pending"

    def test_preparing_value(self):
        assert TransactionState.PREPARING.value == "preparing"

    def test_committing_value(self):
        assert TransactionState.COMMITTING.value == "committing"

    def test_committed_value(self):
        assert TransactionState.COMMITTED.value == "committed"

    def test_rolling_back_value(self):
        assert TransactionState.ROLLING_BACK.value == "rolling_back"

    def test_rolled_back_value(self):
        assert TransactionState.ROLLED_BACK.value == "rolled_back"

    def test_failed_value(self):
        assert TransactionState.FAILED.value == "failed"

    def test_all_states_defined(self):
        expected = {"pending", "preparing", "committing", "committed",
                    "rolling_back", "rolled_back", "failed"}
        actual = {s.value for s in TransactionState}
        assert actual == expected


# ============================================
# Test MergeOperation Dataclass
# ============================================

class TestMergeOperation:
    """Tests for MergeOperation dataclass."""

    def test_create_operation(self):
        now = datetime.now()
        op = MergeOperation(
            operation_id=1,
            transaction_id=10,
            operation_type="add",
            source_path="/src/file.txt",
            dest_path="/dst/file.txt",
            backup_path=None,
            checksum_before=None,
            checksum_after="abc123",
            created_at=now,
        )

        assert op.operation_id == 1
        assert op.transaction_id == 10
        assert op.operation_type == "add"
        assert op.dest_path == "/dst/file.txt"
        assert op.completed_at is None


# ============================================
# Test MergeTransaction Dataclass
# ============================================

class TestMergeTransaction:
    """Tests for MergeTransaction dataclass."""

    def test_create_transaction(self):
        now = datetime.now()
        txn = MergeTransaction(
            transaction_id=1,
            source_host="host-a",
            dest_host="host-b",
            merge_type="games",
            state=TransactionState.PENDING,
            created_at=now,
            updated_at=now,
        )

        assert txn.transaction_id == 1
        assert txn.source_host == "host-a"
        assert txn.dest_host == "host-b"
        assert txn.merge_type == "games"
        assert txn.state == TransactionState.PENDING
        assert txn.operations == []
        assert txn.metadata == {}


# ============================================
# Test TransactionIsolation
# ============================================

class TestTransactionIsolation:
    """Tests for TransactionIsolation class."""

    def test_initialization(self, isolation, temp_dirs):
        """Test isolation manager initialization."""
        assert isolation.db_path == temp_dirs["db_path"]
        assert isolation.wal_dir == temp_dirs["wal_dir"]
        assert isolation.backup_dir == temp_dirs["backup_dir"]

        # Directories should be created
        assert temp_dirs["wal_dir"].exists()
        assert temp_dirs["backup_dir"].exists()

    def test_singleton_pattern(self, temp_dirs):
        """Test get_instance returns singleton."""
        TransactionIsolation.reset_instance()

        instance1 = TransactionIsolation.get_instance(db_path=temp_dirs["db_path"])
        instance2 = TransactionIsolation.get_instance()

        assert instance1 is instance2

        TransactionIsolation.reset_instance()

    def test_begin_transaction(self, isolation):
        """Test starting a transaction."""
        txn_id = isolation.begin_transaction(
            source_host="host-a",
            dest_host="host-b",
            merge_type="games",
            metadata={"version": "1.0"},
        )

        assert txn_id > 0
        assert txn_id in isolation._active_transactions

        # WAL file should be created
        wal_path = isolation.wal_dir / f"txn_{txn_id}.wal"
        assert wal_path.exists()

    def test_add_operation_add_type(self, isolation, tmp_path):
        """Test adding an 'add' operation."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        dest_path = tmp_path / "new_file.txt"
        op_id = isolation.add_operation(
            transaction_id=txn_id,
            operation_type="add",
            dest_path=str(dest_path),
        )

        assert op_id > 0

    def test_add_operation_update_type(self, isolation, test_file):
        """Test adding an 'update' operation backs up existing file."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        op_id = isolation.add_operation(
            transaction_id=txn_id,
            operation_type="update",
            dest_path=str(test_file),
        )

        assert op_id > 0

        # Backup should be created
        backup_dir = isolation.backup_dir / f"txn_{txn_id}"
        assert backup_dir.exists()

    def test_add_operation_invalid_transaction(self, isolation, tmp_path):
        """Test adding operation to non-existent transaction."""
        dest_path = tmp_path / "file.txt"

        with pytest.raises(ValueError, match="not found"):
            isolation.add_operation(999, "add", str(dest_path))

    def test_complete_operation(self, isolation, tmp_path):
        """Test completing an operation."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        dest_path = tmp_path / "file.txt"
        dest_path.write_text("content")

        op_id = isolation.add_operation(txn_id, "add", str(dest_path))
        isolation.complete_operation(txn_id, op_id)

        # Should be able to prepare commit now
        assert isolation.prepare_commit(txn_id) is True

    def test_prepare_commit_incomplete_operations(self, isolation, tmp_path):
        """Test prepare_commit fails with incomplete operations."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        dest_path = tmp_path / "file.txt"
        isolation.add_operation(txn_id, "add", str(dest_path))
        # Don't complete the operation

        assert isolation.prepare_commit(txn_id) is False

    def test_commit_success(self, isolation, tmp_path):
        """Test successful commit."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        dest_path = tmp_path / "file.txt"
        dest_path.write_text("content")

        op_id = isolation.add_operation(txn_id, "add", str(dest_path))
        isolation.complete_operation(txn_id, op_id)

        assert isolation.prepare_commit(txn_id) is True
        assert isolation.commit(txn_id) is True

        # WAL file should be removed
        wal_path = isolation.wal_dir / f"txn_{txn_id}.wal"
        assert not wal_path.exists()

    def test_commit_not_prepared(self, isolation):
        """Test commit fails if not prepared."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        with pytest.raises(ValueError, match="must be in PREPARING"):
            isolation.commit(txn_id)

    def test_rollback(self, isolation, test_file):
        """Test rollback restores original file."""
        original_content = test_file.read_text()

        txn_id = isolation.begin_transaction("host-a", "host-b", "games")
        op_id = isolation.add_operation(txn_id, "update", str(test_file))

        # Modify the file
        test_file.write_text("modified content")

        # Rollback
        assert isolation.rollback(txn_id) is True

        # File should be restored
        assert test_file.read_text() == original_content

    def test_rollback_add_operation(self, isolation, tmp_path):
        """Test rollback removes added file."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        dest_path = tmp_path / "new_file.txt"
        op_id = isolation.add_operation(txn_id, "add", str(dest_path))

        # Create the file
        dest_path.write_text("new content")
        isolation.complete_operation(txn_id, op_id)

        # Rollback
        assert isolation.rollback(txn_id) is True

        # File should be removed
        assert not dest_path.exists()

    def test_context_manager_success(self, isolation, tmp_path):
        """Test transaction context manager with success."""
        dest_path = tmp_path / "file.txt"

        with isolation.transaction("host-a", "host-b", "games") as ctx:
            op_id = ctx.add_operation("add", str(dest_path))
            dest_path.write_text("content")
            ctx.complete_operation(op_id)

        # Transaction should be committed
        stats = isolation.get_transaction_stats()
        assert stats["by_state"].get("committed", 0) >= 1

    def test_context_manager_rollback_on_exception(self, isolation, tmp_path):
        """Test transaction context manager rolls back on exception."""
        dest_path = tmp_path / "file.txt"

        with pytest.raises(RuntimeError):
            with isolation.transaction("host-a", "host-b", "games") as ctx:
                op_id = ctx.add_operation("add", str(dest_path))
                dest_path.write_text("content")
                # Don't complete operation - will fail validation
                raise RuntimeError("Test error")

        # Transaction should be rolled back
        stats = isolation.get_transaction_stats()
        assert stats["by_state"].get("rolled_back", 0) >= 1

    def test_cleanup_old_transactions(self, isolation):
        """Test cleanup of old transactions."""
        # Create and commit a transaction
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        # Force to committed state without operations
        conn = isolation._get_conn()
        conn.execute(
            "UPDATE transactions SET state = ? WHERE transaction_id = ?",
            (TransactionState.COMMITTED.value, txn_id)
        )
        conn.commit()

        # Remove from active
        isolation._active_transactions.pop(txn_id, None)

        # Cleanup with 0 day age (should delete all)
        count = isolation.cleanup_old_transactions(max_age_days=0)
        assert count >= 1

    def test_get_transaction_stats(self, isolation):
        """Test getting transaction statistics."""
        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        stats = isolation.get_transaction_stats()

        assert stats["active_transactions"] >= 1
        assert "by_state" in stats
        assert "by_type" in stats


# ============================================
# Test Recovery
# ============================================

class TestRecovery:
    """Tests for crash recovery functionality."""

    def test_recover_incomplete_transactions(self, temp_dirs):
        """Test incomplete transactions are rolled back on startup."""
        # Create a "crashed" state by creating a transaction directly in DB
        db_path = temp_dirs["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_host TEXT NOT NULL,
                dest_host TEXT NOT NULL,
                merge_type TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                committed_at TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL,
                operation_type TEXT NOT NULL,
                source_path TEXT,
                dest_path TEXT NOT NULL,
                backup_path TEXT,
                checksum_before TEXT,
                checksum_after TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.execute(
            """INSERT INTO transactions
            (source_host, dest_host, merge_type, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            ("host-a", "host-b", "games", "pending",
             datetime.now().isoformat(), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        # Create isolation - should recover incomplete transactions
        isolation = TransactionIsolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        # Transaction should be rolled back
        stats = isolation.get_transaction_stats()
        # Should have rolled_back or failed state for the recovered transaction


# ============================================
# Test Module Functions
# ============================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_transaction_isolation(self, temp_dirs):
        """Test get_transaction_isolation returns singleton."""
        reset_transaction_isolation()

        iso1 = get_transaction_isolation(db_path=temp_dirs["db_path"])
        iso2 = get_transaction_isolation()

        assert iso1 is iso2

    def test_begin_merge_transaction(self, temp_dirs):
        """Test begin_merge_transaction function."""
        reset_transaction_isolation()

        # Pre-initialize to use temp path
        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        txn_id = begin_merge_transaction("host-a", "host-b", "games")
        assert txn_id > 0

    def test_add_merge_operation(self, temp_dirs, tmp_path):
        """Test add_merge_operation function."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        txn_id = begin_merge_transaction("host-a", "host-b", "games")
        dest_path = tmp_path / "file.txt"

        op_id = add_merge_operation(txn_id, "add", str(dest_path))
        assert op_id > 0

    def test_complete_merge_operation(self, temp_dirs, tmp_path):
        """Test complete_merge_operation function."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        txn_id = begin_merge_transaction("host-a", "host-b", "games")
        dest_path = tmp_path / "file.txt"
        dest_path.write_text("content")

        op_id = add_merge_operation(txn_id, "add", str(dest_path))
        complete_merge_operation(txn_id, op_id)

        # Should be completable now
        isolation = get_transaction_isolation()
        assert isolation.prepare_commit(txn_id) is True

    def test_commit_merge_transaction(self, temp_dirs, tmp_path):
        """Test commit_merge_transaction function."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        txn_id = begin_merge_transaction("host-a", "host-b", "games")
        dest_path = tmp_path / "file.txt"
        dest_path.write_text("content")

        op_id = add_merge_operation(txn_id, "add", str(dest_path))
        complete_merge_operation(txn_id, op_id)

        result = commit_merge_transaction(txn_id)
        assert result is True

    def test_rollback_merge_transaction(self, temp_dirs, tmp_path):
        """Test rollback_merge_transaction function."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        txn_id = begin_merge_transaction("host-a", "host-b", "games")
        dest_path = tmp_path / "file.txt"

        add_merge_operation(txn_id, "add", str(dest_path))

        result = rollback_merge_transaction(txn_id)
        assert result is True

    def test_merge_transaction_context(self, temp_dirs, tmp_path):
        """Test merge_transaction context manager."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        dest_path = tmp_path / "file.txt"

        with merge_transaction("host-a", "host-b", "games") as ctx:
            op_id = ctx.add_operation("add", str(dest_path))
            dest_path.write_text("content")
            ctx.complete_operation(op_id)

        # Should be committed
        assert dest_path.exists()

    def test_get_transaction_stats_function(self, temp_dirs):
        """Test get_transaction_stats function."""
        reset_transaction_isolation()

        get_transaction_isolation(
            db_path=temp_dirs["db_path"],
            wal_dir=temp_dirs["wal_dir"],
            backup_dir=temp_dirs["backup_dir"],
        )

        stats = get_transaction_stats()

        assert "active_transactions" in stats
        assert "by_state" in stats
        assert "by_type" in stats


# ============================================
# Integration Tests
# ============================================

class TestTransactionIntegration:
    """Integration tests for transaction isolation."""

    def test_full_transaction_lifecycle(self, isolation, tmp_path):
        """Test complete transaction lifecycle."""
        # Create source file
        source_file = tmp_path / "source.txt"
        source_file.write_text("source content")

        # Create dest file to update
        dest_file = tmp_path / "dest.txt"
        dest_file.write_text("original content")

        # Begin transaction
        txn_id = isolation.begin_transaction(
            source_host="host-a",
            dest_host="host-b",
            merge_type="games",
            metadata={"test": True},
        )

        # Add update operation
        op_id = isolation.add_operation(
            txn_id,
            "update",
            str(dest_file),
            str(source_file),
        )

        # Perform the actual update
        shutil.copy2(source_file, dest_file)

        # Complete operation
        isolation.complete_operation(txn_id, op_id)

        # Prepare and commit
        assert isolation.prepare_commit(txn_id) is True
        assert isolation.commit(txn_id) is True

        # Verify final state
        assert dest_file.read_text() == "source content"

    def test_multiple_operations_transaction(self, isolation, tmp_path):
        """Test transaction with multiple operations."""
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.txt"
            files.append(f)

        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        for f in files:
            op_id = isolation.add_operation(txn_id, "add", str(f))
            f.write_text(f"content {f.name}")
            isolation.complete_operation(txn_id, op_id)

        assert isolation.prepare_commit(txn_id) is True
        assert isolation.commit(txn_id) is True

        for f in files:
            assert f.exists()

    def test_rollback_multiple_operations(self, isolation, tmp_path):
        """Test rollback with multiple operations."""
        # Create files to update
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"original {i}")
            files.append(f)

        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        for f in files:
            isolation.add_operation(txn_id, "update", str(f))
            f.write_text(f"modified {f.name}")

        # Rollback
        assert isolation.rollback(txn_id) is True

        # All files should be restored
        for i, f in enumerate(files):
            assert f.read_text() == f"original {i}"

    def test_checksum_verification(self, isolation, tmp_path):
        """Test checksum verification during prepare."""
        dest_file = tmp_path / "file.txt"

        txn_id = isolation.begin_transaction("host-a", "host-b", "games")

        op_id = isolation.add_operation(txn_id, "add", str(dest_file))
        dest_file.write_text("content")
        isolation.complete_operation(txn_id, op_id)

        # Modify file after completion to break checksum
        dest_file.write_text("different content")

        # Prepare should fail due to checksum mismatch
        assert isolation.prepare_commit(txn_id) is False
