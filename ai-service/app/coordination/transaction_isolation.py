"""Transaction isolation for safe data merge operations.

This module provides transaction-like semantics for merge operations to prevent
partial or corrupted data from being committed during data synchronization.

Key features:
- Write-ahead logging for merge operations
- Atomic commit/rollback semantics
- Crash recovery with automatic rollback of incomplete merges
- Isolation between concurrent merge operations
"""

import json
import logging
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.coordination.database_sync_manager import atomic_copy
from app.coordination.singleton_mixin import SingletonMixin
from app.utils.checksum_utils import compute_file_checksum

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """State of a merge transaction."""
    PENDING = "pending"          # Transaction started, not yet committed
    PREPARING = "preparing"      # Validation in progress
    COMMITTING = "committing"    # Commit in progress
    COMMITTED = "committed"      # Successfully committed
    ROLLING_BACK = "rolling_back"  # Rollback in progress
    ROLLED_BACK = "rolled_back"  # Successfully rolled back
    FAILED = "failed"            # Transaction failed


@dataclass
class MergeOperation:
    """A single file operation within a merge transaction."""
    operation_id: int
    transaction_id: int
    operation_type: str  # 'add', 'update', 'delete', 'append'
    source_path: str | None
    dest_path: str
    backup_path: str | None  # Path to backup of original file
    checksum_before: str | None
    checksum_after: str | None
    created_at: datetime
    completed_at: datetime | None = None


@dataclass
class MergeTransaction:
    """A merge transaction containing multiple operations."""
    transaction_id: int
    source_host: str
    dest_host: str
    merge_type: str  # 'games', 'manifest', 'training_data'
    state: TransactionState
    created_at: datetime
    updated_at: datetime
    committed_at: datetime | None = None
    operations: list[MergeOperation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TransactionIsolation(SingletonMixin):
    """Manages transaction isolation for merge operations.

    Provides ACID-like guarantees for data merge operations:
    - Atomicity: All operations in a transaction succeed or all are rolled back
    - Consistency: Checksums verify data integrity before and after operations
    - Isolation: Concurrent transactions don't interfere with each other
    - Durability: Write-ahead log ensures recoverability after crashes

    December 27, 2025: Migrated to SingletonMixin (Wave 4 Phase 1).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        wal_dir: Path | None = None,
        backup_dir: Path | None = None,
        max_transaction_age_seconds: int = 3600,
    ):
        """Initialize transaction isolation manager.

        Args:
            db_path: Path to SQLite database for transaction log
            wal_dir: Directory for write-ahead log files
            backup_dir: Directory for file backups during transactions
            max_transaction_age_seconds: Maximum age before auto-rollback
        """
        self.db_path = db_path or Path("data/coordination/transactions.db")
        self.wal_dir = wal_dir or Path("data/coordination/wal")
        self.backup_dir = backup_dir or Path("data/coordination/backups")
        self.max_transaction_age = max_transaction_age_seconds

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Thread-local connections
        self._local = threading.local()

        # Lock for transaction operations
        self._transaction_lock = threading.RLock()

        # Active transactions by ID
        self._active_transactions: dict[int, MergeTransaction] = {}

        # Initialize database
        self._init_db()

        # Recover any incomplete transactions on startup
        self._recover_incomplete_transactions()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level="IMMEDIATE"
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
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
            );

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
                completed_at TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );

            CREATE INDEX IF NOT EXISTS idx_transactions_state
                ON transactions(state);
            CREATE INDEX IF NOT EXISTS idx_transactions_created
                ON transactions(created_at);
            CREATE INDEX IF NOT EXISTS idx_operations_transaction
                ON operations(transaction_id);
        """)
        conn.commit()

    def _recover_incomplete_transactions(self) -> None:
        """Recover and rollback any incomplete transactions from previous runs."""
        conn = self._get_conn()

        # Find transactions that were not committed or rolled back
        incomplete_states = (
            TransactionState.PENDING.value,
            TransactionState.PREPARING.value,
            TransactionState.COMMITTING.value,
            TransactionState.ROLLING_BACK.value,
        )

        cursor = conn.execute(
            """
            SELECT transaction_id, source_host, dest_host, merge_type, state
            FROM transactions
            WHERE state IN (?, ?, ?, ?)
            """,
            incomplete_states
        )

        for row in cursor.fetchall():
            transaction_id = row["transaction_id"]
            logger.warning(
                f"Found incomplete transaction {transaction_id} in state {row['state']}, "
                f"rolling back"
            )
            try:
                self._rollback_transaction(transaction_id)
            except Exception as e:
                logger.error(f"Failed to rollback transaction {transaction_id}: {e}")
                # Mark as failed
                conn.execute(
                    "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
                    (TransactionState.FAILED.value, datetime.now().isoformat(), transaction_id)
                )
                conn.commit()

    def begin_transaction(
        self,
        source_host: str,
        dest_host: str,
        merge_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Begin a new merge transaction.

        Args:
            source_host: Host providing the data
            dest_host: Host receiving the data
            merge_type: Type of merge ('games', 'manifest', 'training_data')
            metadata: Optional metadata to store with transaction

        Returns:
            Transaction ID
        """
        with self._transaction_lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            cursor = conn.execute(
                """
                INSERT INTO transactions (source_host, dest_host, merge_type, state,
                                         created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_host,
                    dest_host,
                    merge_type,
                    TransactionState.PENDING.value,
                    now,
                    now,
                    json.dumps(metadata or {}),
                )
            )
            conn.commit()

            transaction_id = cursor.lastrowid
            if transaction_id is None:
                raise RuntimeError("Database INSERT failed to return lastrowid")

            # Create WAL file for this transaction
            wal_path = self.wal_dir / f"txn_{transaction_id}.wal"
            with open(wal_path, "w") as f:
                json.dump({
                    "transaction_id": transaction_id,
                    "source_host": source_host,
                    "dest_host": dest_host,
                    "merge_type": merge_type,
                    "started_at": now,
                    "operations": [],
                }, f)

            # Create transaction object
            transaction = MergeTransaction(
                transaction_id=transaction_id,
                source_host=source_host,
                dest_host=dest_host,
                merge_type=merge_type,
                state=TransactionState.PENDING,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
                metadata=metadata or {},
            )
            self._active_transactions[transaction_id] = transaction

            logger.info(
                f"Started transaction {transaction_id}: {source_host} -> {dest_host} "
                f"({merge_type})"
            )

            return transaction_id

    def add_operation(
        self,
        transaction_id: int,
        operation_type: str,
        dest_path: str,
        source_path: str | None = None,
    ) -> int:
        """Add an operation to a transaction.

        Args:
            transaction_id: Transaction ID
            operation_type: Type of operation ('add', 'update', 'delete', 'append')
            dest_path: Destination file path
            source_path: Source file path (if applicable)

        Returns:
            Operation ID
        """
        with self._transaction_lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            # Verify transaction exists and is in valid state
            cursor = conn.execute(
                "SELECT state FROM transactions WHERE transaction_id = ?",
                (transaction_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Transaction {transaction_id} not found")
            if row["state"] not in (TransactionState.PENDING.value, TransactionState.PREPARING.value):
                raise ValueError(
                    f"Cannot add operation to transaction in state {row['state']}"
                )

            # Create backup of existing file if it exists
            backup_path = None
            checksum_before = None
            dest_path_obj = Path(dest_path)

            if dest_path_obj.exists() and operation_type in ("update", "delete", "append"):
                backup_path = str(
                    self.backup_dir / f"txn_{transaction_id}" / dest_path_obj.name
                )
                Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
                atomic_copy(Path(dest_path), Path(backup_path))
                checksum_before = self._compute_checksum(dest_path)

            cursor = conn.execute(
                """
                INSERT INTO operations (transaction_id, operation_type, source_path,
                                        dest_path, backup_path, checksum_before, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    operation_type,
                    source_path,
                    dest_path,
                    backup_path,
                    checksum_before,
                    now,
                )
            )
            conn.commit()

            operation_id = cursor.lastrowid
            if operation_id is None:
                raise RuntimeError("Database INSERT failed to return lastrowid")

            # Update WAL
            self._append_to_wal(transaction_id, {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "source_path": source_path,
                "dest_path": dest_path,
                "backup_path": backup_path,
                "checksum_before": checksum_before,
            })

            return operation_id

    def complete_operation(
        self,
        transaction_id: int,
        operation_id: int,
        checksum_after: str | None = None,
    ) -> None:
        """Mark an operation as complete.

        Args:
            transaction_id: Transaction ID
            operation_id: Operation ID
            checksum_after: Checksum of file after operation
        """
        with self._transaction_lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            # Get dest_path to compute checksum if not provided
            if checksum_after is None:
                cursor = conn.execute(
                    "SELECT dest_path FROM operations WHERE operation_id = ?",
                    (operation_id,)
                )
                row = cursor.fetchone()
                if row and Path(row["dest_path"]).exists():
                    checksum_after = self._compute_checksum(row["dest_path"])

            conn.execute(
                """
                UPDATE operations
                SET completed_at = ?, checksum_after = ?
                WHERE operation_id = ? AND transaction_id = ?
                """,
                (now, checksum_after, operation_id, transaction_id)
            )
            conn.commit()

    def prepare_commit(self, transaction_id: int) -> bool:
        """Prepare transaction for commit (validation phase).

        Args:
            transaction_id: Transaction ID

        Returns:
            True if validation passed, False otherwise
        """
        with self._transaction_lock:
            conn = self._get_conn()

            # Update state to preparing
            conn.execute(
                "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
                (TransactionState.PREPARING.value, datetime.now().isoformat(), transaction_id)
            )
            conn.commit()

            # Verify all operations completed
            cursor = conn.execute(
                """
                SELECT COUNT(*) as incomplete
                FROM operations
                WHERE transaction_id = ? AND completed_at IS NULL
                """,
                (transaction_id,)
            )
            if cursor.fetchone()["incomplete"] > 0:
                logger.error(f"Transaction {transaction_id} has incomplete operations")
                return False

            # Verify checksums where applicable
            cursor = conn.execute(
                """
                SELECT operation_id, dest_path, checksum_after
                FROM operations
                WHERE transaction_id = ? AND checksum_after IS NOT NULL
                """,
                (transaction_id,)
            )

            for row in cursor.fetchall():
                if Path(row["dest_path"]).exists():
                    current_checksum = self._compute_checksum(row["dest_path"])
                    if current_checksum != row["checksum_after"]:
                        logger.error(
                            f"Checksum mismatch for operation {row['operation_id']}: "
                            f"expected {row['checksum_after']}, got {current_checksum}"
                        )
                        return False

            logger.info(f"Transaction {transaction_id} passed validation")
            return True

    def commit(self, transaction_id: int) -> bool:
        """Commit a transaction.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if committed successfully
        """
        with self._transaction_lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            # Verify in preparing state
            cursor = conn.execute(
                "SELECT state FROM transactions WHERE transaction_id = ?",
                (transaction_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Transaction {transaction_id} not found")
            if row["state"] != TransactionState.PREPARING.value:
                raise ValueError(
                    f"Cannot commit transaction in state {row['state']}, "
                    f"must be in PREPARING state"
                )

            # Update state to committing
            conn.execute(
                "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
                (TransactionState.COMMITTING.value, now, transaction_id)
            )
            conn.commit()

            try:
                # Clean up backups since we're committing
                backup_dir = self.backup_dir / f"txn_{transaction_id}"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

                # Remove WAL file
                wal_path = self.wal_dir / f"txn_{transaction_id}.wal"
                if wal_path.exists():
                    wal_path.unlink()

                # Mark as committed
                conn.execute(
                    """
                    UPDATE transactions
                    SET state = ?, updated_at = ?, committed_at = ?
                    WHERE transaction_id = ?
                    """,
                    (TransactionState.COMMITTED.value, now, now, transaction_id)
                )
                conn.commit()

                # Remove from active transactions
                self._active_transactions.pop(transaction_id, None)

                logger.info(f"Transaction {transaction_id} committed successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to commit transaction {transaction_id}: {e}")
                # Attempt rollback
                self._rollback_transaction(transaction_id)
                return False

    def rollback(self, transaction_id: int) -> bool:
        """Rollback a transaction.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if rolled back successfully
        """
        with self._transaction_lock:
            return self._rollback_transaction(transaction_id)

    def _rollback_transaction(self, transaction_id: int) -> bool:
        """Internal rollback implementation."""
        conn = self._get_conn()
        now = datetime.now().isoformat()

        # Update state to rolling back
        conn.execute(
            "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
            (TransactionState.ROLLING_BACK.value, now, transaction_id)
        )
        conn.commit()

        try:
            # Restore backups in reverse order
            cursor = conn.execute(
                """
                SELECT operation_id, operation_type, dest_path, backup_path
                FROM operations
                WHERE transaction_id = ?
                ORDER BY operation_id DESC
                """,
                (transaction_id,)
            )

            for row in cursor.fetchall():
                try:
                    if row["backup_path"] and Path(row["backup_path"]).exists():
                        # Restore from backup
                        atomic_copy(Path(row["backup_path"]), Path(row["dest_path"]))
                        logger.debug(
                            f"Restored {row['dest_path']} from {row['backup_path']}"
                        )
                    elif row["operation_type"] == "add" and Path(row["dest_path"]).exists():
                        # Remove added file
                        Path(row["dest_path"]).unlink()
                        logger.debug(f"Removed {row['dest_path']}")
                except Exception as e:
                    logger.error(f"Failed to rollback operation {row['operation_id']}: {e}")

            # Clean up backup directory
            backup_dir = self.backup_dir / f"txn_{transaction_id}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            # Remove WAL file
            wal_path = self.wal_dir / f"txn_{transaction_id}.wal"
            if wal_path.exists():
                wal_path.unlink()

            # Mark as rolled back
            conn.execute(
                "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
                (TransactionState.ROLLED_BACK.value, now, transaction_id)
            )
            conn.commit()

            # Remove from active transactions
            self._active_transactions.pop(transaction_id, None)

            logger.info(f"Transaction {transaction_id} rolled back successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback transaction {transaction_id}: {e}")
            conn.execute(
                "UPDATE transactions SET state = ?, updated_at = ? WHERE transaction_id = ?",
                (TransactionState.FAILED.value, now, transaction_id)
            )
            conn.commit()
            return False

    @contextmanager
    def transaction(
        self,
        source_host: str,
        dest_host: str,
        merge_type: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Context manager for merge transactions.

        Usage:
            with transaction_isolation.transaction("host1", "host2", "games") as txn:
                op_id = txn.add_operation("add", "/path/to/dest", source="/path/to/src")
                # ... perform actual file operation ...
                txn.complete_operation(op_id)
            # Transaction auto-commits on success, auto-rollbacks on exception
        """
        transaction_id = self.begin_transaction(
            source_host, dest_host, merge_type, metadata
        )

        class TransactionContext:
            def __init__(ctx_self, isolation: TransactionIsolation, txn_id: int):
                ctx_self.isolation = isolation
                ctx_self.transaction_id = txn_id
                ctx_self.operations: list[int] = []

            def add_operation(
                ctx_self,
                operation_type: str,
                dest_path: str,
                source_path: str | None = None,
            ) -> int:
                op_id = ctx_self.isolation.add_operation(
                    ctx_self.transaction_id,
                    operation_type,
                    dest_path,
                    source_path,
                )
                ctx_self.operations.append(op_id)
                return op_id

            def complete_operation(
                ctx_self,
                operation_id: int,
                checksum_after: str | None = None,
            ) -> None:
                ctx_self.isolation.complete_operation(
                    ctx_self.transaction_id,
                    operation_id,
                    checksum_after,
                )

        ctx = TransactionContext(self, transaction_id)

        try:
            yield ctx

            # Validate and commit
            if self.prepare_commit(transaction_id):
                self.commit(transaction_id)
            else:
                raise RuntimeError(f"Transaction {transaction_id} validation failed")

        except Exception as e:
            logger.error(f"Transaction {transaction_id} failed: {e}")
            self.rollback(transaction_id)
            raise

    def _compute_checksum(self, path: str) -> str:
        """Compute SHA256 checksum of a file."""
        return compute_file_checksum(path, return_empty_for_missing=False)

    def _append_to_wal(self, transaction_id: int, operation: dict[str, Any]) -> None:
        """Append operation to write-ahead log."""
        wal_path = self.wal_dir / f"txn_{transaction_id}.wal"

        if wal_path.exists():
            with open(wal_path) as f:
                wal_data = json.load(f)
            wal_data["operations"].append(operation)
            with open(wal_path, "w") as f:
                json.dump(wal_data, f)

    def cleanup_old_transactions(self, max_age_days: int = 7) -> int:
        """Clean up old completed transactions.

        Args:
            max_age_days: Maximum age of transactions to keep

        Returns:
            Number of transactions cleaned up
        """
        conn = self._get_conn()
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

        # Delete old operations first
        conn.execute(
            """
            DELETE FROM operations
            WHERE transaction_id IN (
                SELECT transaction_id FROM transactions
                WHERE state IN (?, ?, ?) AND created_at < ?
            )
            """,
            (
                TransactionState.COMMITTED.value,
                TransactionState.ROLLED_BACK.value,
                TransactionState.FAILED.value,
                cutoff_iso,
            )
        )

        # Delete old transactions
        cursor = conn.execute(
            """
            DELETE FROM transactions
            WHERE state IN (?, ?, ?) AND created_at < ?
            """,
            (
                TransactionState.COMMITTED.value,
                TransactionState.ROLLED_BACK.value,
                TransactionState.FAILED.value,
                cutoff_iso,
            )
        )

        conn.commit()
        count = cursor.rowcount

        if count > 0:
            logger.info(f"Cleaned up {count} old transactions")

        return count

    def get_transaction_stats(self) -> dict[str, Any]:
        """Get transaction statistics."""
        conn = self._get_conn()

        stats = {
            "active_transactions": len(self._active_transactions),
            "by_state": {},
            "by_type": {},
            "recent_failures": 0,
        }

        # Count by state
        cursor = conn.execute(
            "SELECT state, COUNT(*) as count FROM transactions GROUP BY state"
        )
        for row in cursor.fetchall():
            stats["by_state"][row["state"]] = row["count"]

        # Count by merge type
        cursor = conn.execute(
            "SELECT merge_type, COUNT(*) as count FROM transactions GROUP BY merge_type"
        )
        for row in cursor.fetchall():
            stats["by_type"][row["merge_type"]] = row["count"]

        # Recent failures (last 24 hours)
        cutoff = datetime.now().timestamp() - 86400
        cursor = conn.execute(
            """
            SELECT COUNT(*) as count FROM transactions
            WHERE state IN (?, ?) AND created_at > ?
            """,
            (
                TransactionState.FAILED.value,
                TransactionState.ROLLED_BACK.value,
                datetime.fromtimestamp(cutoff).isoformat(),
            )
        )
        stats["recent_failures"] = cursor.fetchone()["count"]

        return stats

    def health_check(self) -> "HealthCheckResult":
        """Check transaction isolation health for daemon monitoring.

        December 2025 Phase 4: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with transaction status and failure metrics.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            stats = self.get_transaction_stats()
            active_count = stats["active_transactions"]
            recent_failures = stats["recent_failures"]

            # Check for stuck transactions (PENDING/PREPARING for too long)
            conn = self._get_conn()
            stuck_threshold = datetime.now().timestamp() - self.max_transaction_age
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM transactions
                WHERE state IN (?, ?) AND created_at < ?
                """,
                (
                    TransactionState.PENDING.value,
                    TransactionState.PREPARING.value,
                    datetime.fromtimestamp(stuck_threshold).isoformat(),
                )
            )
            stuck_count = cursor.fetchone()["count"]

            if stuck_count > 0:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"{stuck_count} stuck transactions older than {self.max_transaction_age}s",
                    details={
                        "stuck_transactions": stuck_count,
                        "active_transactions": active_count,
                        "recent_failures": recent_failures,
                    },
                )

            if recent_failures > 10:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High transaction failure rate: {recent_failures} failures in 24h",
                    details={
                        "active_transactions": active_count,
                        "recent_failures": recent_failures,
                    },
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Transaction isolation healthy: {active_count} active, {recent_failures} recent failures",
                details={
                    "active_transactions": active_count,
                    "recent_failures": recent_failures,
                    "by_state": stats["by_state"],
                },
            )

        except Exception as e:
            logger.error(f"Error checking TransactionIsolation health: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
            )


# Module-level singleton access
_instance: TransactionIsolation | None = None


def get_transaction_isolation(
    db_path: Path | None = None,
    **kwargs
) -> TransactionIsolation:
    """Get the singleton TransactionIsolation instance."""
    global _instance
    if _instance is None:
        _instance = TransactionIsolation.get_instance(db_path=db_path, **kwargs)
    return _instance


def reset_transaction_isolation() -> None:
    """Reset the singleton for testing."""
    global _instance
    _instance = None
    TransactionIsolation.reset_instance()


def begin_merge_transaction(
    source_host: str,
    dest_host: str,
    merge_type: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Begin a new merge transaction."""
    return get_transaction_isolation().begin_transaction(
        source_host, dest_host, merge_type, metadata
    )


def add_merge_operation(
    transaction_id: int,
    operation_type: str,
    dest_path: str,
    source_path: str | None = None,
) -> int:
    """Add an operation to a merge transaction."""
    return get_transaction_isolation().add_operation(
        transaction_id, operation_type, dest_path, source_path
    )


def complete_merge_operation(
    transaction_id: int,
    operation_id: int,
    checksum_after: str | None = None,
) -> None:
    """Mark a merge operation as complete."""
    get_transaction_isolation().complete_operation(
        transaction_id, operation_id, checksum_after
    )


def commit_merge_transaction(transaction_id: int) -> bool:
    """Commit a merge transaction."""
    isolation = get_transaction_isolation()
    if isolation.prepare_commit(transaction_id):
        return isolation.commit(transaction_id)
    return False


def rollback_merge_transaction(transaction_id: int) -> bool:
    """Rollback a merge transaction."""
    return get_transaction_isolation().rollback(transaction_id)


@contextmanager
def merge_transaction(
    source_host: str,
    dest_host: str,
    merge_type: str,
    metadata: dict[str, Any] | None = None,
):
    """Context manager for merge transactions."""
    with get_transaction_isolation().transaction(
        source_host, dest_host, merge_type, metadata
    ) as ctx:
        yield ctx


def get_transaction_stats() -> dict[str, Any]:
    """Get transaction statistics."""
    return get_transaction_isolation().get_transaction_stats()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "MergeOperation",
    "MergeTransaction",
    # Main class
    "TransactionIsolation",
    # Enums
    "TransactionState",
    "add_merge_operation",
    "begin_merge_transaction",
    "commit_merge_transaction",
    "complete_merge_operation",
    # Functions
    "get_transaction_isolation",
    "get_transaction_stats",
    "merge_transaction",
    "reset_transaction_isolation",
    "rollback_merge_transaction",
]
