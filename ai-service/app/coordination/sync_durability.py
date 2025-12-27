"""Unified Sync Durability Module - WAL and Dead Letter Queue for Sync Operations.

This module consolidates Write-Ahead Log (WAL) and Dead Letter Queue (DLQ) patterns
for robust, crash-safe data synchronization across the cluster.

Consolidates functionality from:
- app/distributed/unified_data_sync.py: Dead Letter Queue (lines 585-611, schema 457-468)
- app/distributed/unified_data_sync.py: WAL recovery (lines 1689-1713)
- app/coordination/ephemeral_sync.py: JSONL-based WAL (lines 160-257)

Key improvements over existing implementations:
1. SQLite-based persistence (more robust than JSONL)
2. Unified interface for WAL and DLQ
3. Connection pooling for performance
4. Comprehensive statistics and monitoring
5. Production-ready error handling and logging

Usage:
    from app.coordination.sync_durability import SyncWAL, DeadLetterQueue

    # Write-ahead log for crash recovery
    wal = SyncWAL(db_path=Path("data/sync_wal.db"))
    entry_id = wal.append(
        game_id="abc123",
        source="gh200_a",
        target="coordinator",
        data={"moves": [...]}
    )
    wal.mark_complete(entry_id)

    # Dead letter queue for failed syncs
    dlq = DeadLetterQueue(db_path=Path("data/sync_dlq.db"))
    dlq.add(game_id="xyz789", error="Connection timeout", retry_count=3)
    pending = dlq.get_pending(limit=10)
    dlq.resolve("xyz789")

See also:
- app/distributed/unified_wal.py - General-purpose WAL (ingestion, sync, etc.)
- This module specializes WAL/DLQ for sync coordination scenarios
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from app.utils.checksum_utils import compute_string_checksum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class SyncStatus(str, Enum):
    """Status of a sync operation."""
    PENDING = "pending"      # Awaiting sync
    IN_PROGRESS = "in_progress"  # Currently syncing
    COMPLETE = "complete"    # Successfully synced
    FAILED = "failed"        # Failed (moved to DLQ)


@dataclass
class SyncWALEntry:
    """A single entry in the sync write-ahead log."""
    entry_id: int
    game_id: str
    source_host: str
    target_host: str
    data_json: str
    data_hash: str
    status: SyncStatus
    created_at: float
    completed_at: float | None = None
    retry_count: int = 0
    error_message: str | None = None

    @property
    def data(self) -> dict[str, Any]:
        """Parse data_json to dictionary."""
        try:
            return json.loads(self.data_json)
        except json.JSONDecodeError:
            return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "game_id": self.game_id,
            "source_host": self.source_host,
            "target_host": self.target_host,
            "data_hash": self.data_hash,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
        }


@dataclass
class DeadLetterEntry:
    """A single entry in the dead letter queue."""
    entry_id: int
    game_id: str
    source_host: str
    target_host: str
    error_message: str
    error_type: str
    retry_count: int
    added_at: float
    last_retry_at: float | None = None
    resolved: bool = False
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "game_id": self.game_id,
            "source_host": self.source_host,
            "target_host": self.target_host,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "added_at": self.added_at,
            "last_retry_at": self.last_retry_at,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


@dataclass
class WALStats:
    """Statistics for the sync WAL."""
    total_entries: int = 0
    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    oldest_pending: float | None = None
    newest_entry: float | None = None


@dataclass
class DLQStats:
    """Statistics for the dead letter queue."""
    total_entries: int = 0
    unresolved: int = 0
    resolved: int = 0
    by_error_type: dict[str, int] = None  # type: ignore[assignment]
    oldest_unresolved: float | None = None
    avg_retry_count: float = 0.0

    def __post_init__(self):
        if self.by_error_type is None:
            self.by_error_type = {}


# =============================================================================
# Connection Pool (thread-local connections for performance)
# =============================================================================


class ConnectionPool:
    """Thread-local SQLite connection pool.

    Reuses connections per-thread to eliminate connection overhead.
    Critical for high-throughput sync operations.
    """

    def __init__(
        self,
        db_path: Path,
        timeout: float = 30.0,
    ):
        """Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.timeout = timeout

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.RLock()

        # Stats
        self._connections_created = 0
        self._connections_reused = 0

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")

        with self._lock:
            self._connections_created += 1

        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection from the pool (thread-local).

        Yields:
            sqlite3.Connection for this thread
        """
        # Check for existing thread-local connection
        conn = getattr(self._local, 'conn', None)

        if conn is None:
            # Create new connection for this thread
            conn = self._create_connection()
            self._local.conn = conn
        else:
            with self._lock:
                self._connections_reused += 1

        try:
            yield conn
        except sqlite3.Error:
            # On error, invalidate the connection
            self._local.conn = None
            try:
                conn.close()
            except (sqlite3.Error, OSError) as close_err:
                logger.debug(f"Error closing connection after failure: {close_err}")
            raise

    def close_all(self) -> None:
        """Close all connections in the pool."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            except (sqlite3.Error, OSError) as close_err:
                logger.debug(f"Error closing connection: {close_err}")
            self._local.conn = None

    def get_stats(self) -> dict[str, int]:
        """Get connection pool statistics."""
        with self._lock:
            total = self._connections_created + self._connections_reused
            return {
                "connections_created": self._connections_created,
                "connections_reused": self._connections_reused,
                "reuse_ratio": self._connections_reused / max(1, total),
            }


# =============================================================================
# Sync Write-Ahead Log (WAL)
# =============================================================================


class SyncWAL:
    """Write-Ahead Log for crash-safe sync operations.

    Records sync operations before they happen to enable recovery on crash.
    Uses SQLite for robust persistence (superior to JSONL approach).

    Features:
    - Crash recovery via recover() on startup
    - Idempotent operations (duplicate detection)
    - Connection pooling for performance
    - Automatic cleanup of old completed entries
    - Comprehensive statistics

    Thread-safe: All operations use internal locking.
    """

    def __init__(
        self,
        db_path: Path,
        max_pending: int = 10000,
        cleanup_interval: int = 1000,
        use_connection_pool: bool = True,
    ):
        """Initialize the sync WAL.

        Args:
            db_path: Path to SQLite database
            max_pending: Maximum pending entries before blocking
            cleanup_interval: Number of operations between automatic cleanup
            use_connection_pool: Use thread-local connection pooling
        """
        self.db_path = db_path
        self.max_pending = max_pending
        self.cleanup_interval = cleanup_interval
        self.use_connection_pool = use_connection_pool

        self._lock = threading.RLock()
        self._ops_since_cleanup = 0

        # Connection pool for performance
        self._conn_pool: ConnectionPool | None = None
        if use_connection_pool:
            self._conn_pool = ConnectionPool(db_path)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection (uses pool if available).

        Yields:
            sqlite3.Connection - pooled if use_connection_pool=True
        """
        if self._conn_pool is not None:
            with self._conn_pool.get_connection() as conn:
                yield conn
        else:
            # Fallback to per-operation connection
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        """Initialize WAL database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Dec 2025: Use context manager to prevent connection leaks
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS sync_wal (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    source_host TEXT NOT NULL,
                    target_host TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT
                );

                -- Indexes for efficient queries
                CREATE INDEX IF NOT EXISTS idx_sync_wal_status
                ON sync_wal(status, created_at);

                CREATE INDEX IF NOT EXISTS idx_sync_wal_game_id
                ON sync_wal(game_id);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_wal_dedup
                ON sync_wal(game_id, data_hash);

                -- Metadata table
                CREATE TABLE IF NOT EXISTS sync_wal_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                INSERT OR IGNORE INTO sync_wal_metadata (key, value, updated_at)
                VALUES ('schema_version', '1.0', strftime('%s', 'now'));
            """)
            conn.commit()
        logger.debug(f"Initialized SyncWAL at {self.db_path}")

    def _compute_hash(self, content: str) -> str:
        """Compute hash for deduplication."""
        return compute_string_checksum(content, truncate=32)

    def append(
        self,
        game_id: str,
        source: str,
        target: str,
        data: dict[str, Any],
    ) -> int:
        """Add entry to WAL before attempting sync.

        Args:
            game_id: Unique game identifier
            source: Source host name
            target: Target host name
            data: Game data to sync

        Returns:
            Entry ID for tracking

        Raises:
            RuntimeError: If WAL is full (max_pending exceeded)
        """
        with self._lock:
            # Check pending count
            pending = self._get_pending_count()
            if pending >= self.max_pending:
                raise RuntimeError(f"SyncWAL full: {pending} pending entries")

            # Compute hash for deduplication
            data_json = json.dumps(data, sort_keys=True)
            data_hash = self._compute_hash(f"{game_id}:{data_json}")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check for duplicate (idempotency)
                cursor.execute("""
                    SELECT entry_id FROM sync_wal
                    WHERE game_id = ? AND data_hash = ?
                """, (game_id, data_hash))

                existing = cursor.fetchone()
                if existing:
                    logger.debug(f"SyncWAL entry already exists for game {game_id}")
                    return existing[0]

                # Append entry
                now = time.time()
                cursor.execute("""
                    INSERT INTO sync_wal
                    (game_id, source_host, target_host, data_json, data_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (game_id, source, target, data_json, data_hash, now))

                entry_id = cursor.lastrowid
                conn.commit()

                if entry_id is None:
                    raise RuntimeError("Database INSERT failed to return lastrowid")

                self._ops_since_cleanup += 1
                self._maybe_cleanup()

                logger.debug(
                    f"SyncWAL: Added entry {entry_id} for game {game_id} "
                    f"({source} -> {target})"
                )
                return entry_id

    def mark_complete(self, entry_id: int) -> bool:
        """Mark entry as successfully synced.

        Args:
            entry_id: Entry ID to mark

        Returns:
            True if entry was found and marked, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.time()

            cursor.execute("""
                UPDATE sync_wal
                SET status = ?, completed_at = ?
                WHERE entry_id = ? AND status != ?
            """, (SyncStatus.COMPLETE.value, now, entry_id, SyncStatus.COMPLETE.value))

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.debug(f"SyncWAL: Marked entry {entry_id} as complete")

            self._ops_since_cleanup += 1
            self._maybe_cleanup()

            return updated

    def mark_failed(
        self,
        entry_id: int,
        error_message: str = "",
    ) -> bool:
        """Mark entry as failed.

        Args:
            entry_id: Entry ID to mark
            error_message: Error description

        Returns:
            True if entry was found and marked, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE sync_wal
                SET status = ?, error_message = ?, retry_count = retry_count + 1
                WHERE entry_id = ?
            """, (SyncStatus.FAILED.value, error_message, entry_id))

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.warning(
                    f"SyncWAL: Marked entry {entry_id} as failed: {error_message}"
                )

            return updated

    def recover(self) -> list[SyncWALEntry]:
        """Recover unsynced entries on startup.

        Finds all entries in PENDING or IN_PROGRESS state (indicates crash).
        Caller should attempt to sync these entries.

        Returns:
            List of entries awaiting sync
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_id, game_id, source_host, target_host,
                       data_json, data_hash, status, created_at, completed_at,
                       retry_count, error_message
                FROM sync_wal
                WHERE status IN (?, ?)
                ORDER BY created_at ASC
            """, (SyncStatus.PENDING.value, SyncStatus.IN_PROGRESS.value))

            entries = self._rows_to_entries(cursor.fetchall())

            if entries:
                logger.info(f"SyncWAL: Recovered {len(entries)} unsynced entries")

            return entries

    def get_pending(self, limit: int = 100) -> list[SyncWALEntry]:
        """Get pending entries for sync.

        Args:
            limit: Maximum entries to return

        Returns:
            List of pending entries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_id, game_id, source_host, target_host,
                       data_json, data_hash, status, created_at, completed_at,
                       retry_count, error_message
                FROM sync_wal
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (SyncStatus.PENDING.value, limit))

            entries = self._rows_to_entries(cursor.fetchall())
            return entries

    def clear_completed(self, older_than_hours: int = 24) -> int:
        """Remove old completed entries.

        Args:
            older_than_hours: Remove entries completed more than N hours ago

        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = time.time() - (older_than_hours * 3600)

            cursor.execute("""
                DELETE FROM sync_wal
                WHERE status = ? AND completed_at < ?
            """, (SyncStatus.COMPLETE.value, cutoff))

            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"SyncWAL: Cleaned up {removed} old completed entries")

            return removed

    def get_stats(self) -> WALStats:
        """Get WAL statistics.

        Returns:
            WALStats with current counts and timestamps
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    MIN(CASE WHEN status = 'pending' THEN created_at END) as oldest_pending,
                    MAX(created_at) as newest_entry
                FROM sync_wal
            """)

            row = cursor.fetchone()

            return WALStats(
                total_entries=row[0] or 0,
                pending=row[1] or 0,
                in_progress=row[2] or 0,
                completed=row[3] or 0,
                failed=row[4] or 0,
                oldest_pending=row[5],
                newest_entry=row[6],
            )

    def _get_pending_count(self) -> int:
        """Get count of pending entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM sync_wal WHERE status = ?
            """, (SyncStatus.PENDING.value,))
            count = cursor.fetchone()[0]
            return count

    def _rows_to_entries(self, rows: list[tuple]) -> list[SyncWALEntry]:
        """Convert database rows to SyncWALEntry objects."""
        return [
            SyncWALEntry(
                entry_id=row[0],
                game_id=row[1],
                source_host=row[2],
                target_host=row[3],
                data_json=row[4],
                data_hash=row[5],
                status=SyncStatus(row[6]),
                created_at=row[7],
                completed_at=row[8],
                retry_count=row[9],
                error_message=row[10],
            )
            for row in rows
        ]

    def _maybe_cleanup(self) -> None:
        """Trigger cleanup if interval exceeded."""
        if self._ops_since_cleanup >= self.cleanup_interval:
            self.clear_completed()
            self._ops_since_cleanup = 0


# =============================================================================
# Dead Letter Queue (DLQ)
# =============================================================================


class DeadLetterQueue:
    """Dead Letter Queue for permanently failed sync operations.

    Stores sync failures that exceeded retry limits for manual investigation.
    Separate from WAL to avoid polluting crash recovery logic.

    Features:
    - Tracks error types and retry counts
    - Manual resolution workflow
    - Statistics by error type
    - Connection pooling for performance

    Thread-safe: All operations use internal locking.
    """

    def __init__(
        self,
        db_path: Path,
        use_connection_pool: bool = True,
    ):
        """Initialize the dead letter queue.

        Args:
            db_path: Path to SQLite database
            use_connection_pool: Use thread-local connection pooling
        """
        self.db_path = db_path
        self.use_connection_pool = use_connection_pool

        self._lock = threading.RLock()

        # Connection pool for performance
        self._conn_pool: ConnectionPool | None = None
        if use_connection_pool:
            self._conn_pool = ConnectionPool(db_path)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection (uses pool if available).

        Yields:
            sqlite3.Connection - pooled if use_connection_pool=True
        """
        if self._conn_pool is not None:
            with self._conn_pool.get_connection() as conn:
                yield conn
        else:
            # Fallback to per-operation connection
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        """Initialize DLQ database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                source_host TEXT NOT NULL,
                target_host TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_type TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                added_at REAL NOT NULL,
                last_retry_at REAL,
                resolved INTEGER DEFAULT 0,
                resolved_at REAL
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_dlq_resolved
            ON dead_letter_queue(resolved, added_at);

            CREATE INDEX IF NOT EXISTS idx_dlq_game_id
            ON dead_letter_queue(game_id);

            CREATE INDEX IF NOT EXISTS idx_dlq_error_type
            ON dead_letter_queue(error_type);

            -- Metadata table
            CREATE TABLE IF NOT EXISTS dlq_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            INSERT OR IGNORE INTO dlq_metadata (key, value, updated_at)
            VALUES ('schema_version', '1.0', strftime('%s', 'now'));
        """)
        conn.commit()
        conn.close()
        logger.debug(f"Initialized DeadLetterQueue at {self.db_path}")

    def add(
        self,
        game_id: str,
        source: str,
        target: str,
        error: str,
        error_type: str = "unknown",
        retry_count: int = 0,
    ) -> int:
        """Add failed sync to dead letter queue.

        Args:
            game_id: Game identifier that failed to sync
            source: Source host name
            target: Target host name
            error: Error message
            error_type: Error category (e.g., "timeout", "validation", "network")
            retry_count: Number of retries attempted

        Returns:
            Entry ID for tracking
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()

                cursor.execute("""
                    INSERT INTO dead_letter_queue
                    (game_id, source_host, target_host, error_message, error_type,
                     retry_count, added_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (game_id, source, target, error, error_type, retry_count, now))

                entry_id = cursor.lastrowid
                conn.commit()

                if entry_id is None:
                    raise RuntimeError("Database INSERT failed to return lastrowid")

                logger.warning(
                    f"DLQ: Added game {game_id} ({source} -> {target}) "
                    f"after {retry_count} retries: {error}"
                )
                return entry_id

    def get_pending(self, limit: int = 100) -> list[DeadLetterEntry]:
        """Get unresolved DLQ entries for retry.

        Args:
            limit: Maximum entries to return

        Returns:
            List of unresolved entries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_id, game_id, source_host, target_host,
                       error_message, error_type, retry_count, added_at,
                       last_retry_at, resolved, resolved_at
                FROM dead_letter_queue
                WHERE resolved = 0
                ORDER BY added_at ASC
                LIMIT ?
            """, (limit,))

            entries = self._rows_to_entries(cursor.fetchall())
            return entries

    def resolve(self, game_id: str) -> int:
        """Mark DLQ entry as resolved.

        Args:
            game_id: Game identifier to resolve

        Returns:
            Number of entries resolved (should be 1)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.time()

            cursor.execute("""
                UPDATE dead_letter_queue
                SET resolved = 1, resolved_at = ?
                WHERE game_id = ? AND resolved = 0
            """, (now, game_id))

            updated = cursor.rowcount
            conn.commit()

            if updated > 0:
                logger.info(f"DLQ: Resolved game {game_id}")

            return updated

    def update_retry(
        self,
        game_id: str,
        error: str = "",
    ) -> bool:
        """Update retry timestamp and increment count.

        Args:
            game_id: Game identifier
            error: Optional updated error message

        Returns:
            True if entry was found and updated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = time.time()

            if error:
                cursor.execute("""
                    UPDATE dead_letter_queue
                    SET retry_count = retry_count + 1,
                        last_retry_at = ?,
                        error_message = ?
                    WHERE game_id = ? AND resolved = 0
                """, (now, error, game_id))
            else:
                cursor.execute("""
                    UPDATE dead_letter_queue
                    SET retry_count = retry_count + 1, last_retry_at = ?
                    WHERE game_id = ? AND resolved = 0
                """, (now, game_id))

            updated = cursor.rowcount > 0
            conn.commit()

            return updated

    def get_stats(self) -> DLQStats:
        """Get DLQ statistics.

        Returns:
            DLQStats with counts and breakdown by error type
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END) as unresolved,
                    SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved,
                    MIN(CASE WHEN resolved = 0 THEN added_at END) as oldest_unresolved,
                    AVG(retry_count) as avg_retry_count
                FROM dead_letter_queue
            """)

            row = cursor.fetchone()

            # By error type
            cursor.execute("""
                SELECT error_type, COUNT(*) as count
                FROM dead_letter_queue
                WHERE resolved = 0
                GROUP BY error_type
            """)

            by_error_type = {row[0]: row[1] for row in cursor.fetchall()}

            return DLQStats(
                total_entries=row[0] or 0,
                unresolved=row[1] or 0,
                resolved=row[2] or 0,
                oldest_unresolved=row[3],
                avg_retry_count=row[4] or 0.0,
                by_error_type=by_error_type,
            )

    def cleanup_resolved(self, older_than_days: int = 7) -> int:
        """Remove old resolved entries.

        Args:
            older_than_days: Remove entries resolved more than N days ago

        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = time.time() - (older_than_days * 86400)

            cursor.execute("""
                DELETE FROM dead_letter_queue
                WHERE resolved = 1 AND resolved_at < ?
            """, (cutoff,))

            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"DLQ: Cleaned up {removed} old resolved entries")

            return removed

    def _rows_to_entries(self, rows: list[tuple]) -> list[DeadLetterEntry]:
        """Convert database rows to DeadLetterEntry objects."""
        return [
            DeadLetterEntry(
                entry_id=row[0],
                game_id=row[1],
                source_host=row[2],
                target_host=row[3],
                error_message=row[4],
                error_type=row[5],
                retry_count=row[6],
                added_at=row[7],
                last_retry_at=row[8],
                resolved=bool(row[9]),
                resolved_at=row[10],
            )
            for row in rows
        ]


# =============================================================================
# Module-level utilities
# =============================================================================


_sync_wal_instance: SyncWAL | None = None
_dlq_instance: DeadLetterQueue | None = None
_module_lock = threading.RLock()


def get_sync_wal(db_path: Path | None = None) -> SyncWAL:
    """Get singleton SyncWAL instance.

    Args:
        db_path: Optional path to WAL database (uses default if None)

    Returns:
        SyncWAL instance
    """
    global _sync_wal_instance

    with _module_lock:
        if _sync_wal_instance is None:
            if db_path is None:
                # Default path
                db_path = Path(__file__).parents[2] / "data" / "sync_wal.db"
            _sync_wal_instance = SyncWAL(db_path)

        return _sync_wal_instance


def get_dlq(db_path: Path | None = None) -> DeadLetterQueue:
    """Get singleton DeadLetterQueue instance.

    Args:
        db_path: Optional path to DLQ database (uses default if None)

    Returns:
        DeadLetterQueue instance
    """
    global _dlq_instance

    with _module_lock:
        if _dlq_instance is None:
            if db_path is None:
                # Default path
                db_path = Path(__file__).parents[2] / "data" / "sync_dlq.db"
            _dlq_instance = DeadLetterQueue(db_path)

        return _dlq_instance


def reset_instances() -> None:
    """Reset singleton instances (for testing)."""
    global _sync_wal_instance, _dlq_instance
    with _module_lock:
        _sync_wal_instance = None
        _dlq_instance = None


__all__ = [
    "SyncWAL",
    "DeadLetterQueue",
    "SyncWALEntry",
    "DeadLetterEntry",
    "WALStats",
    "DLQStats",
    "SyncStatus",
    "get_sync_wal",
    "get_dlq",
]
