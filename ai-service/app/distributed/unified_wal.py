"""Unified Write-Ahead Log (WAL) for RingRift AI Service.

This module provides a single WAL implementation for all crash-recovery scenarios,
consolidating functionality from:
- data_sync_robust.py:WriteAheadLog (sync recovery)
- ingestion_wal.py:IngestionWAL (ingestion recovery)

Key features:
1. Two logical streams (sync + ingestion) in single schema
2. Crash-safe operations with checkpointing
3. Idempotent replay on recovery
4. Efficient batch operations
5. Automatic compaction

Usage:
    from app.distributed.unified_wal import UnifiedWAL, WALEntryType

    wal = UnifiedWAL(db_path=Path("data/unified_wal.db"))

    # For sync operations
    entry_id = wal.append_sync_entry(
        game_id="abc123",
        source_host="gh200_a",
        source_db="selfplay.db",
        data_hash="sha256..."
    )

    # For ingestion operations
    entry_id = wal.append_ingestion_entry(
        game_id="abc123",
        game_data={"moves": [...], ...},
        source_host="gh200_a"
    )

    # Mark as processed
    wal.mark_processed([entry_id])

    # Recovery
    for entry in wal.get_pending_entries():
        process_entry(entry)
        wal.mark_processed([entry.entry_id])
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time

from app.utils.checksum_utils import compute_string_checksum
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

logger = logging.getLogger(__name__)


# =============================================================================
# Connection Pool (Thread-local connections for 95% reduction in overhead)
# =============================================================================


class ConnectionPool:
    """Thread-local SQLite connection pool.

    Reuses connections per-thread instead of creating new ones per operation.
    This eliminates 10-50s of connection overhead per training epoch.

    Usage:
        pool = ConnectionPool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(...)
            conn.commit()

    Thread-safety: Each thread gets its own connection (thread-local storage).
    """

    def __init__(
        self,
        db_path: Path,
        timeout: float = 30.0,
        check_same_thread: bool = False,
    ):
        """Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            timeout: Connection timeout in seconds
            check_same_thread: If False, allow connection sharing across threads
                              (we use thread-local storage so this is safe)
        """
        self.db_path = db_path
        self.timeout = timeout
        self.check_same_thread = check_same_thread

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
            check_same_thread=self.check_same_thread,
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

        Returns existing connection if available, creates new one if needed.
        Connection remains open for thread reuse (not closed on context exit).

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
        except sqlite3.Error as e:
            # On error, invalidate the connection
            self._local.conn = None
            try:
                conn.close()
            except Exception:
                pass
            raise

    def close_all(self) -> None:
        """Close all connections in the pool.

        Note: Only closes the calling thread's connection.
        Other threads will get new connections on next access.
        """
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "connections_created": self._connections_created,
                "connections_reused": self._connections_reused,
                "reuse_ratio": (
                    self._connections_reused / max(1, self._connections_created + self._connections_reused)
                ),
            }


# =============================================================================
# Entry Types and Data Classes
# =============================================================================


class WALEntryType(str, Enum):
    """Type of WAL entry for logical separation."""
    SYNC = "sync"           # Game data sync from remote host
    INGESTION = "ingestion"  # Game ingestion/processing
    ELO_SYNC = "elo_sync"   # Elo database sync
    MODEL_SYNC = "model_sync"  # Model file sync


class WALEntryStatus(str, Enum):
    """Status of a WAL entry."""
    PENDING = "pending"      # Awaiting processing
    SYNCED = "synced"        # Synced but not confirmed
    PROCESSED = "processed"  # Fully processed
    FAILED = "failed"        # Failed permanently (dead letter)


@dataclass
class WALEntry:
    """A single entry in the unified write-ahead log."""
    entry_id: int
    entry_type: WALEntryType
    game_id: str
    source_host: str
    source_db: str
    data_hash: str
    data_json: Optional[str] = None  # Full game data for ingestion entries
    status: WALEntryStatus = WALEntryStatus.PENDING
    created_at: float = 0.0
    updated_at: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Parse data_json if present."""
        if self.data_json:
            try:
                return json.loads(self.data_json)
            except json.JSONDecodeError:
                return None
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "game_id": self.game_id,
            "source_host": self.source_host,
            "source_db": self.source_db,
            "data_hash": self.data_hash,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
        }


@dataclass
class WALCheckpoint:
    """Checkpoint marker for WAL compaction."""
    checkpoint_id: int
    last_entry_id: int
    timestamp: float
    entries_compacted: int
    entry_type: Optional[WALEntryType] = None


@dataclass
class WALStats:
    """Statistics for the WAL."""
    total_entries: int = 0
    pending_sync: int = 0
    pending_ingestion: int = 0
    synced: int = 0
    processed: int = 0
    failed: int = 0
    last_checkpoint_id: int = 0
    last_checkpoint_time: float = 0.0


# =============================================================================
# Unified WAL Implementation
# =============================================================================


class UnifiedWAL:
    """Unified Write-Ahead Log for all crash-recovery scenarios.

    Provides a single WAL supporting multiple entry types (sync, ingestion, etc.)
    with consistent APIs and shared infrastructure for checkpointing and compaction.

    Features connection pooling for 95% reduction in SQLite connection overhead.
    """

    def __init__(
        self,
        db_path: Path,
        max_pending: int = 10000,
        checkpoint_interval: int = 1000,
        auto_compact: bool = True,
        max_retries: int = 3,
        use_connection_pool: bool = True,
    ):
        """Initialize the unified WAL.

        Args:
            db_path: Path to SQLite database
            max_pending: Maximum pending entries before blocking
            checkpoint_interval: Entries between automatic checkpoints
            auto_compact: Automatically compact after checkpoint
            max_retries: Maximum retries before marking as failed
            use_connection_pool: Use thread-local connection pooling
        """
        self.db_path = db_path
        self.max_pending = max_pending
        self.checkpoint_interval = checkpoint_interval
        self.auto_compact = auto_compact
        self.max_retries = max_retries
        self.use_connection_pool = use_connection_pool

        self._lock = threading.RLock()
        self._entries_since_checkpoint = 0

        # Connection pool for thread-local connection reuse
        self._conn_pool: Optional[ConnectionPool] = None
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
        # Use direct connection for initialization (pool may not exist yet)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            -- Unified WAL entries table
            CREATE TABLE IF NOT EXISTS wal_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_type TEXT NOT NULL,
                game_id TEXT NOT NULL,
                source_host TEXT NOT NULL,
                source_db TEXT DEFAULT '',
                data_hash TEXT NOT NULL,
                data_json TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_wal_status_type
            ON wal_entries(status, entry_type, created_at);

            CREATE INDEX IF NOT EXISTS idx_wal_game_id
            ON wal_entries(game_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_wal_dedup
            ON wal_entries(entry_type, game_id, data_hash);

            -- Checkpoint tracking
            CREATE TABLE IF NOT EXISTS wal_checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_entry_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                entries_compacted INTEGER DEFAULT 0,
                entry_type TEXT
            );

            -- Recovery and configuration metadata
            CREATE TABLE IF NOT EXISTS wal_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Initialize metadata
            INSERT OR IGNORE INTO wal_metadata (key, value, updated_at)
            VALUES
                ('schema_version', '2.0', strftime('%s', 'now')),
                ('last_recovery', '0', strftime('%s', 'now'));
        """)
        conn.commit()
        conn.close()
        logger.debug(f"Initialized unified WAL at {self.db_path}")

    def _compute_hash(self, content: str) -> str:
        """Compute hash for deduplication."""
        return compute_string_checksum(content, truncate=32)

    # =========================================================================
    # Sync Entry Operations (from data_sync_robust.WriteAheadLog)
    # =========================================================================

    def append_sync_entry(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        data_hash: str,
    ) -> int:
        """Append a sync entry to the WAL.

        Used for tracking game data transfers from remote hosts.

        Args:
            game_id: Unique game identifier
            source_host: Remote host name
            source_db: Source database name
            data_hash: Hash of game data for verification

        Returns:
            Entry ID for tracking
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check for duplicate (idempotency)
                cursor.execute("""
                    SELECT entry_id FROM wal_entries
                    WHERE entry_type = ? AND game_id = ? AND data_hash = ?
                """, (WALEntryType.SYNC.value, game_id, data_hash))

                existing = cursor.fetchone()
                if existing:
                    return existing[0]

                # Append entry
                now = time.time()
                cursor.execute("""
                    INSERT INTO wal_entries
                    (entry_type, game_id, source_host, source_db, data_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (WALEntryType.SYNC.value, game_id, source_host, source_db, data_hash, now))

                entry_id = cursor.lastrowid
                conn.commit()

                self._entries_since_checkpoint += 1
                self._maybe_checkpoint()

                return entry_id

    def append_sync_batch(
        self,
        entries: List[Tuple[str, str, str, str]],  # (game_id, host, db, hash)
    ) -> int:
        """Append multiple sync entries efficiently.

        Args:
            entries: List of (game_id, source_host, source_db, data_hash) tuples

        Returns:
            Number of entries added (excluding duplicates)
        """
        if not entries:
            return 0

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()
                added = 0

                for game_id, source_host, source_db, data_hash in entries:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO wal_entries
                            (entry_type, game_id, source_host, source_db, data_hash, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (WALEntryType.SYNC.value, game_id, source_host, source_db, data_hash, now))
                        if cursor.rowcount > 0:
                            added += 1
                            self._entries_since_checkpoint += 1
                    except sqlite3.IntegrityError:
                        pass  # Duplicate, skip

                conn.commit()

                self._maybe_checkpoint()
                return added

    # =========================================================================
    # Ingestion Entry Operations (from ingestion_wal.IngestionWAL)
    # =========================================================================

    def append_ingestion_entry(
        self,
        game_id: str,
        game_data: Dict[str, Any],
        source_host: str = "",
    ) -> int:
        """Append an ingestion entry to the WAL.

        Used for crash-safe game processing/ingestion.

        Args:
            game_id: Unique game identifier
            game_data: Full game data dictionary
            source_host: Source host name

        Returns:
            Entry ID for tracking
        """
        with self._lock:
            # Check pending count
            pending = self._get_pending_count()
            if pending >= self.max_pending:
                raise RuntimeError(f"WAL full: {pending} pending entries")

            # Compute hash for deduplication
            data_json = json.dumps(game_data, sort_keys=True)
            data_hash = self._compute_hash(f"{game_id}:{data_json}")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check for duplicate
                cursor.execute("""
                    SELECT entry_id FROM wal_entries
                    WHERE entry_type = ? AND game_id = ? AND data_hash = ?
                """, (WALEntryType.INGESTION.value, game_id, data_hash))

                existing = cursor.fetchone()
                if existing:
                    return existing[0]

                # Append entry with full data
                now = time.time()
                cursor.execute("""
                    INSERT INTO wal_entries
                    (entry_type, game_id, source_host, source_db, data_hash, data_json, created_at)
                    VALUES (?, ?, ?, '', ?, ?, ?)
                """, (WALEntryType.INGESTION.value, game_id, source_host, data_hash, data_json, now))

                entry_id = cursor.lastrowid
                conn.commit()

                self._entries_since_checkpoint += 1
                self._maybe_checkpoint()

                return entry_id

    def append_ingestion_batch(
        self,
        games: List[Tuple[str, Dict[str, Any]]],
        source_host: str = "",
    ) -> List[int]:
        """Append multiple ingestion entries efficiently.

        Args:
            games: List of (game_id, game_data) tuples
            source_host: Source host name

        Returns:
            List of entry IDs (0 for duplicates)
        """
        if not games:
            return []

        with self._lock:
            entry_ids = []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = time.time()

                for game_id, game_data in games:
                    data_json = json.dumps(game_data, sort_keys=True)
                    data_hash = self._compute_hash(f"{game_id}:{data_json}")

                    # Check for existing
                    cursor.execute("""
                        SELECT entry_id FROM wal_entries
                        WHERE entry_type = ? AND game_id = ? AND data_hash = ?
                    """, (WALEntryType.INGESTION.value, game_id, data_hash))

                    existing = cursor.fetchone()
                    if existing:
                        entry_ids.append(existing[0])
                        continue

                    # Append
                    cursor.execute("""
                        INSERT INTO wal_entries
                        (entry_type, game_id, source_host, source_db, data_hash, data_json, created_at)
                        VALUES (?, ?, ?, '', ?, ?, ?)
                    """, (WALEntryType.INGESTION.value, game_id, source_host, data_hash, data_json, now))

                    entry_ids.append(cursor.lastrowid)
                    self._entries_since_checkpoint += 1

                conn.commit()

                self._maybe_checkpoint()
                return entry_ids

    # =========================================================================
    # Status Updates
    # =========================================================================

    def mark_synced(self, entry_ids: List[int]) -> int:
        """Mark entries as synced (intermediate state for sync entries).

        Args:
            entry_ids: List of entry IDs to mark

        Returns:
            Number of entries updated
        """
        if not entry_ids:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(entry_ids))
            now = time.time()

            cursor.execute(f"""
                UPDATE wal_entries
                SET status = ?, updated_at = ?
                WHERE entry_id IN ({placeholders}) AND status = ?
            """, [WALEntryStatus.SYNCED.value, now] + list(entry_ids) + [WALEntryStatus.PENDING.value])

            updated = cursor.rowcount
            conn.commit()
            return updated

    def mark_processed(self, entry_ids: List[int]) -> int:
        """Mark entries as fully processed.

        Args:
            entry_ids: List of entry IDs to mark

        Returns:
            Number of entries updated
        """
        if not entry_ids:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(entry_ids))
            now = time.time()

            cursor.execute(f"""
                UPDATE wal_entries
                SET status = ?, updated_at = ?
                WHERE entry_id IN ({placeholders}) AND status IN (?, ?)
            """, [WALEntryStatus.PROCESSED.value, now] + list(entry_ids) +
                 [WALEntryStatus.PENDING.value, WALEntryStatus.SYNCED.value])

            updated = cursor.rowcount
            conn.commit()
            return updated

    def mark_failed(
        self,
        entry_ids: List[int],
        error_message: str = "",
    ) -> int:
        """Mark entries as failed (dead letter).

        Args:
            entry_ids: List of entry IDs to mark
            error_message: Error description

        Returns:
            Number of entries updated
        """
        if not entry_ids:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(entry_ids))
            now = time.time()

            cursor.execute(f"""
                UPDATE wal_entries
                SET status = ?, updated_at = ?, error_message = ?,
                    retry_count = retry_count + 1
                WHERE entry_id IN ({placeholders})
            """, [WALEntryStatus.FAILED.value, now, error_message] + list(entry_ids))

            updated = cursor.rowcount
            conn.commit()
            return updated

    def increment_retry(self, entry_id: int) -> bool:
        """Increment retry count for an entry.

        Returns True if entry can still be retried, False if max retries exceeded.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE wal_entries
                SET retry_count = retry_count + 1, updated_at = ?
                WHERE entry_id = ?
            """, (time.time(), entry_id))

            cursor.execute("""
                SELECT retry_count FROM wal_entries WHERE entry_id = ?
            """, (entry_id,))

            row = cursor.fetchone()
            conn.commit()

            if row and row[0] >= self.max_retries:
                self.mark_failed([entry_id], f"Max retries ({self.max_retries}) exceeded")
                return False
            return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_pending_entries(
        self,
        entry_type: Optional[WALEntryType] = None,
        limit: int = 1000,
    ) -> List[WALEntry]:
        """Get pending entries for processing.

        Args:
            entry_type: Filter by entry type (None for all)
            limit: Maximum entries to return

        Returns:
            List of pending WALEntry objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if entry_type:
                cursor.execute("""
                    SELECT entry_id, entry_type, game_id, source_host, source_db,
                           data_hash, data_json, status, created_at, updated_at,
                           retry_count, error_message
                    FROM wal_entries
                    WHERE status = ? AND entry_type = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (WALEntryStatus.PENDING.value, entry_type.value, limit))
            else:
                cursor.execute("""
                    SELECT entry_id, entry_type, game_id, source_host, source_db,
                           data_hash, data_json, status, created_at, updated_at,
                           retry_count, error_message
                    FROM wal_entries
                    WHERE status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (WALEntryStatus.PENDING.value, limit))

            entries = self._rows_to_entries(cursor.fetchall())
            return entries

    def get_pending_sync_entries(self, limit: int = 1000) -> List[WALEntry]:
        """Get pending sync entries."""
        return self.get_pending_entries(WALEntryType.SYNC, limit)

    def get_pending_ingestion_entries(self, limit: int = 1000) -> List[WALEntry]:
        """Get pending ingestion entries."""
        return self.get_pending_entries(WALEntryType.INGESTION, limit)

    def get_synced_unconfirmed(self, limit: int = 1000) -> List[WALEntry]:
        """Get entries that are synced but not yet processed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_id, entry_type, game_id, source_host, source_db,
                       data_hash, data_json, status, created_at, updated_at,
                       retry_count, error_message
                FROM wal_entries
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (WALEntryStatus.SYNCED.value, limit))

            entries = self._rows_to_entries(cursor.fetchall())
            return entries

    def get_failed_entries(self, limit: int = 100) -> List[WALEntry]:
        """Get failed entries (dead letter queue)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_id, entry_type, game_id, source_host, source_db,
                       data_hash, data_json, status, created_at, updated_at,
                       retry_count, error_message
                FROM wal_entries
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (WALEntryStatus.FAILED.value, limit))

            entries = self._rows_to_entries(cursor.fetchall())
            return entries

    def _rows_to_entries(self, rows: List[tuple]) -> List[WALEntry]:
        """Convert database rows to WALEntry objects."""
        return [
            WALEntry(
                entry_id=row[0],
                entry_type=WALEntryType(row[1]),
                game_id=row[2],
                source_host=row[3],
                source_db=row[4],
                data_hash=row[5],
                data_json=row[6],
                status=WALEntryStatus(row[7]),
                created_at=row[8],
                updated_at=row[9],
                retry_count=row[10],
                error_message=row[11],
            )
            for row in rows
        ]

    def _get_pending_count(self) -> int:
        """Get count of pending entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM wal_entries WHERE status = ?
            """, (WALEntryStatus.PENDING.value,))
            count = cursor.fetchone()[0]
            return count

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> WALStats:
        """Get WAL statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' AND entry_type = 'sync' THEN 1 ELSE 0 END) as pending_sync,
                    SUM(CASE WHEN status = 'pending' AND entry_type = 'ingestion' THEN 1 ELSE 0 END) as pending_ingestion,
                    SUM(CASE WHEN status = 'synced' THEN 1 ELSE 0 END) as synced,
                    SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM wal_entries
            """)
            row = cursor.fetchone()

            cursor.execute("""
                SELECT checkpoint_id, timestamp FROM wal_checkpoints
                ORDER BY checkpoint_id DESC LIMIT 1
            """)
            checkpoint_row = cursor.fetchone()

            return WALStats(
                total_entries=row[0] or 0,
                pending_sync=row[1] or 0,
                pending_ingestion=row[2] or 0,
                synced=row[3] or 0,
                processed=row[4] or 0,
                failed=row[5] or 0,
                last_checkpoint_id=checkpoint_row[0] if checkpoint_row else 0,
                last_checkpoint_time=checkpoint_row[1] if checkpoint_row else 0.0,
            )

    # =========================================================================
    # Checkpointing and Compaction
    # =========================================================================

    def _maybe_checkpoint(self) -> None:
        """Create checkpoint if interval exceeded."""
        if self._entries_since_checkpoint >= self.checkpoint_interval:
            self._create_checkpoint()

    def _create_checkpoint(self) -> int:
        """Create a checkpoint and optionally compact.

        Returns checkpoint ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get last entry ID
            cursor.execute("SELECT MAX(entry_id) FROM wal_entries")
            last_entry_id = cursor.fetchone()[0] or 0

            # Create checkpoint
            cursor.execute("""
                INSERT INTO wal_checkpoints (last_entry_id, timestamp, entries_compacted)
                VALUES (?, ?, 0)
            """, (last_entry_id, time.time()))

            checkpoint_id = cursor.lastrowid
            conn.commit()

            self._entries_since_checkpoint = 0

            # Auto compact
            if self.auto_compact:
                self.compact()

            logger.debug(f"Created checkpoint {checkpoint_id} at entry {last_entry_id}")
            return checkpoint_id

    def compact(self, older_than_hours: int = 24) -> int:
        """Compact WAL by removing old processed entries.

        Args:
            older_than_hours: Remove processed entries older than this

        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = time.time() - (older_than_hours * 3600)

            cursor.execute("""
                DELETE FROM wal_entries
                WHERE status = ? AND updated_at < ?
            """, (WALEntryStatus.PROCESSED.value, cutoff))

            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"Compacted {removed} processed entries from WAL")

            return removed

    def cleanup_failed(self, older_than_days: int = 7) -> int:
        """Remove old failed entries.

        Args:
            older_than_days: Remove failed entries older than this

        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = time.time() - (older_than_days * 86400)

            cursor.execute("""
                DELETE FROM wal_entries
                WHERE status = ? AND updated_at < ?
            """, (WALEntryStatus.FAILED.value, cutoff))

            removed = cursor.rowcount
            conn.commit()

            if removed > 0:
                logger.info(f"Removed {removed} old failed entries from WAL")

            return removed

    def get_connection_pool_stats(self) -> Optional[Dict[str, int]]:
        """Get connection pool statistics if pooling is enabled."""
        if self._conn_pool:
            return self._conn_pool.get_stats()
        return None


# =============================================================================
# Compatibility Wrappers (for migration from old implementations)
# =============================================================================


class WriteAheadLog(UnifiedWAL):
    """Backward-compatible wrapper for data_sync_robust.WriteAheadLog.

    DEPRECATED: Use UnifiedWAL directly for new code.
    """

    def __init__(self, db_path: Path):
        super().__init__(db_path)
        logger.warning(
            "WriteAheadLog is deprecated. Use UnifiedWAL directly. "
            "This wrapper preserves API compatibility during migration."
        )

    def append(
        self,
        game_id: str,
        source_host: str,
        source_db: str,
        game_data_hash: str,
    ) -> int:
        """Append entry (compatibility method)."""
        return self.append_sync_entry(game_id, source_host, source_db, game_data_hash)

    def append_batch(
        self,
        entries: List[Tuple[str, str, str, str]],
    ) -> int:
        """Append batch (compatibility method)."""
        return self.append_sync_batch(entries)

    def confirm_synced(self, game_ids: List[str]) -> int:
        """Confirm synced (compatibility method)."""
        # Find entry IDs by game_id
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(game_ids))
        cursor.execute(f"""
            SELECT entry_id FROM wal_entries WHERE game_id IN ({placeholders})
        """, game_ids)
        entry_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return self.mark_processed(entry_ids)

    def get_pending_entries(  # type: ignore[override]
        self, entry_type: Optional["WALEntryType"] = None, limit: int = 1000
    ) -> List[WALEntry]:
        """Get pending entries (compatibility method)."""
        # Ignores entry_type for backward compatibility
        return super().get_pending_sync_entries(limit)

    def get_unconfirmed_entries(self, limit: int = 1000) -> List[WALEntry]:
        """Get unconfirmed entries (compatibility method)."""
        return super().get_synced_unconfirmed(limit)

    def cleanup_confirmed(self, older_than_seconds: int = 3600) -> int:
        """Cleanup confirmed entries (compatibility method)."""
        hours = max(1, older_than_seconds // 3600)
        return self.compact(older_than_hours=hours)


class IngestionWAL(UnifiedWAL):
    """Backward-compatible wrapper for ingestion_wal.IngestionWAL.

    DEPRECATED: Use UnifiedWAL directly for new code.
    """

    def __init__(
        self,
        wal_dir: Path,
        max_unprocessed: int = 10000,
        checkpoint_interval: int = 1000,
        auto_compact: bool = True,
    ):
        db_path = wal_dir / "unified_wal.db"
        super().__init__(
            db_path=db_path,
            max_pending=max_unprocessed,
            checkpoint_interval=checkpoint_interval,
            auto_compact=auto_compact,
        )
        logger.warning(
            "IngestionWAL is deprecated. Use UnifiedWAL directly. "
            "This wrapper preserves API compatibility during migration."
        )

    def append(
        self,
        game_data: Dict[str, Any],
        source_host: str = "",
        game_id: Optional[str] = None,
    ) -> int:
        """Append entry (compatibility method)."""
        if game_id is None:
            game_id = game_data.get("game_id", "")
            if not game_id:
                raise ValueError("game_id required in game_data or as argument")
        return self.append_ingestion_entry(game_id, game_data, source_host)

    def append_batch(
        self,
        games: List[Tuple[str, Dict[str, Any]]],
        source_host: str = "",
    ) -> List[int]:
        """Append batch (compatibility method)."""
        return self.append_ingestion_batch(games, source_host)

    def mark_processed_single(self, entry_id: int) -> bool:
        """Mark single entry processed (compatibility method)."""
        return super().mark_processed([entry_id]) > 0

    def mark_processed(self, entry_ids: List[int]) -> int:  # type: ignore[override]
        """Mark processed (compatibility method - batch)."""
        return super().mark_processed(entry_ids)

    def mark_batch_processed(self, entry_ids: List[int]) -> int:
        """Mark batch processed (compatibility method)."""
        return super().mark_processed(entry_ids)

    def get_unprocessed(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WALEntry]:
        """Get unprocessed entries (compatibility method)."""
        # Note: offset not supported in unified version
        return super().get_pending_ingestion_entries(limit)


# =============================================================================
# Module-level utilities
# =============================================================================


_wal_instance: Optional[UnifiedWAL] = None
_wal_lock = threading.RLock()


def get_unified_wal(db_path: Optional[Path] = None) -> UnifiedWAL:
    """Get singleton unified WAL instance.

    Args:
        db_path: Optional path to WAL database (uses default if None)

    Returns:
        UnifiedWAL instance
    """
    global _wal_instance

    with _wal_lock:
        if _wal_instance is None:
            if db_path is None:
                # Default path
                from pathlib import Path
                db_path = Path(__file__).parents[2] / "data" / "unified_wal.db"
            _wal_instance = UnifiedWAL(db_path)

        return _wal_instance


def reset_wal_instance() -> None:
    """Reset the singleton WAL instance (for testing)."""
    global _wal_instance
    with _wal_lock:
        _wal_instance = None
