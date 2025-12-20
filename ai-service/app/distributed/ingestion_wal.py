"""Distributed Write-Ahead Log (WAL) for Game Ingestion.

This module provides crash-safe game ingestion using a write-ahead log.
Games are written to the WAL before being processed, ensuring recovery
on crash without data loss or duplication.

IMPORTANT: This module has been consolidated into unified_wal.py.
For new code, use:
    from app.distributed.unified_wal import UnifiedWAL, WALEntry

This file is maintained for backward compatibility with existing code.
The IngestionWAL class is now a thin wrapper around UnifiedWAL.

Key features:
1. Crash-safe ingestion - games persisted before processing
2. Idempotent replay - safe to replay WAL on recovery
3. Distributed replication - WAL can be replicated to other hosts
4. Checkpoint and compaction - efficient cleanup of processed entries

Usage:
    wal = IngestionWAL(wal_dir=Path("data/ingestion_wal"))

    # Write game to WAL before processing
    entry_id = wal.append(game_data, source_host="gh200_a")

    # Process game (add to manifest, training data, etc.)
    process_game(game_data)

    # Mark as processed
    wal.mark_processed(entry_id)

    # On recovery, replay unprocessed entries
    for entry in wal.get_unprocessed():
        process_game(entry.data)
        wal.mark_processed(entry.entry_id)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import unified WAL implementation
try:
    from app.distributed.unified_wal import (
        IngestionWAL as UnifiedIngestionWAL,
        WALCheckpoint as UnifiedWALCheckpoint,
        WALEntry as UnifiedWALEntry,
    )
    HAS_UNIFIED_WAL = True
except ImportError:
    HAS_UNIFIED_WAL = False
    UnifiedIngestionWAL = None
    UnifiedWALEntry = None
    UnifiedWALCheckpoint = None

# For backward compatibility, also import legacy dependencies
import json
import sqlite3
import threading
import time
from dataclasses import dataclass

from app.utils.checksum_utils import compute_string_checksum

# =============================================================================
# Legacy Implementation (kept for fallback if unified_wal not available)
# =============================================================================


@dataclass
class _LegacyWALEntry:
    """A single entry in the write-ahead log (legacy format)."""
    entry_id: int
    game_id: str
    data: dict[str, Any]
    source_host: str
    checksum: str
    timestamp: float
    processed: bool = False
    processed_at: float | None = None


@dataclass
class _LegacyWALCheckpoint:
    """Checkpoint marker for WAL compaction (legacy)."""
    checkpoint_id: int
    last_entry_id: int
    timestamp: float
    entries_compacted: int


class _LegacyIngestionWAL:
    """Write-ahead log for crash-safe game ingestion (legacy implementation)."""

    def __init__(
        self,
        wal_dir: Path,
        max_unprocessed: int = 10000,
        checkpoint_interval: int = 1000,
        auto_compact: bool = True,
    ):
        """Initialize the ingestion WAL.

        Args:
            wal_dir: Directory for WAL storage
            max_unprocessed: Maximum unprocessed entries before blocking
            checkpoint_interval: Entries between automatic checkpoints
            auto_compact: Automatically compact after checkpoint
        """
        self.wal_dir = wal_dir
        self.max_unprocessed = max_unprocessed
        self.checkpoint_interval = checkpoint_interval
        self.auto_compact = auto_compact

        self._db_path = wal_dir / "ingestion_wal.db"
        self._lock = threading.RLock()
        self._entries_since_checkpoint = 0

        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize WAL storage."""
        self.wal_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.executescript("""
            -- Main WAL entries table
            CREATE TABLE IF NOT EXISTS wal_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                data_json TEXT NOT NULL,
                source_host TEXT,
                checksum TEXT NOT NULL,
                timestamp REAL NOT NULL,
                processed INTEGER DEFAULT 0,
                processed_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_wal_unprocessed
            ON wal_entries(processed, entry_id);

            CREATE INDEX IF NOT EXISTS idx_wal_game_id
            ON wal_entries(game_id);

            -- Checkpoint tracking
            CREATE TABLE IF NOT EXISTS wal_checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_entry_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                entries_compacted INTEGER DEFAULT 0
            );

            -- Recovery metadata
            CREATE TABLE IF NOT EXISTS wal_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            -- Initialize metadata
            INSERT OR IGNORE INTO wal_metadata (key, value)
            VALUES ('last_recovery', '0');
        """)
        conn.commit()
        conn.close()

    def _compute_checksum(self, game_id: str, data: dict[str, Any]) -> str:
        """Compute checksum for entry validation."""
        content = f"{game_id}:{json.dumps(data, sort_keys=True)}"
        return compute_string_checksum(content, truncate=32)

    def append(
        self,
        game_data: dict[str, Any],
        source_host: str = "",
        game_id: str | None = None,
    ) -> int:
        """Append a game to the WAL.

        Args:
            game_data: Game data dictionary
            source_host: Source host name
            game_id: Optional game ID (extracted from data if not provided)

        Returns:
            Entry ID for tracking

        Raises:
            RuntimeError: If WAL is full (max_unprocessed exceeded)
        """
        with self._lock:
            # Check if WAL is full
            unprocessed = self._get_unprocessed_count()
            if unprocessed >= self.max_unprocessed:
                raise RuntimeError(f"WAL full: {unprocessed} unprocessed entries")

            # Extract game_id if not provided
            if game_id is None:
                game_id = game_data.get("game_id", "")
                if not game_id:
                    raise ValueError("game_id required in game_data or as argument")

            # Compute checksum
            checksum = self._compute_checksum(game_id, game_data)

            # Check for duplicate (idempotency)
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT entry_id FROM wal_entries WHERE game_id = ? AND checksum = ?",
                (game_id, checksum)
            )
            existing = cursor.fetchone()
            if existing:
                conn.close()
                logger.debug(f"WAL entry already exists for game {game_id}")
                return existing[0]

            # Append entry
            cursor.execute("""
                INSERT INTO wal_entries (game_id, data_json, source_host, checksum, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                game_id,
                json.dumps(game_data),
                source_host,
                checksum,
                time.time(),
            ))
            entry_id = cursor.lastrowid
            conn.commit()
            conn.close()
            if entry_id is None:
                raise RuntimeError("Database INSERT failed to return lastrowid")

            self._entries_since_checkpoint += 1

            # Auto checkpoint
            if self._entries_since_checkpoint >= self.checkpoint_interval:
                self._create_checkpoint()

            return entry_id

    def append_batch(
        self,
        games: list[tuple[str, dict[str, Any]]],
        source_host: str = "",
    ) -> list[int]:
        """Append multiple games to the WAL efficiently.

        Args:
            games: List of (game_id, game_data) tuples
            source_host: Source host name

        Returns:
            List of entry IDs
        """
        with self._lock:
            entry_ids = []
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            for game_id, game_data in games:
                checksum = self._compute_checksum(game_id, game_data)

                # Check for existing
                cursor.execute(
                    "SELECT entry_id FROM wal_entries WHERE game_id = ? AND checksum = ?",
                    (game_id, checksum)
                )
                existing = cursor.fetchone()
                if existing:
                    entry_ids.append(existing[0])
                    continue

                # Append
                cursor.execute("""
                    INSERT INTO wal_entries (game_id, data_json, source_host, checksum, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    game_id,
                    json.dumps(game_data),
                    source_host,
                    checksum,
                    time.time(),
                ))
                entry_ids.append(cursor.lastrowid)
                self._entries_since_checkpoint += 1

            conn.commit()
            conn.close()

            # Auto checkpoint
            if self._entries_since_checkpoint >= self.checkpoint_interval:
                self._create_checkpoint()

            return entry_ids

    def mark_processed(self, entry_id: int) -> bool:
        """Mark an entry as processed.

        Returns True if entry was found and marked.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE wal_entries
            SET processed = 1, processed_at = ?
            WHERE entry_id = ? AND processed = 0
        """, (time.time(), entry_id))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def mark_batch_processed(self, entry_ids: list[int]) -> int:
        """Mark multiple entries as processed.

        Returns number of entries marked.
        """
        if not entry_ids:
            return 0

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(entry_ids))
        cursor.execute(f"""
            UPDATE wal_entries
            SET processed = 1, processed_at = ?
            WHERE entry_id IN ({placeholders}) AND processed = 0
        """, [time.time(), *entry_ids])
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return updated

    def get_unprocessed(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WALEntry]:
        """Get unprocessed entries for replay.

        Args:
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of unprocessed WALEntry objects
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_id, game_id, data_json, source_host, checksum, timestamp
            FROM wal_entries
            WHERE processed = 0
            ORDER BY entry_id ASC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        entries = []
        for row in cursor.fetchall():
            entries.append(_LegacyWALEntry(
                entry_id=row[0],
                game_id=row[1],
                data=json.loads(row[2]),
                source_host=row[3],
                checksum=row[4],
                timestamp=row[5],
                processed=False,
            ))

        conn.close()
        return entries

    def iter_unprocessed(self, batch_size: int = 100) -> Generator[_LegacyWALEntry, None, None]:
        """Iterate over all unprocessed entries.

        Yields entries in order, handling pagination automatically.
        """
        offset = 0
        while True:
            entries = self.get_unprocessed(limit=batch_size, offset=offset)
            if not entries:
                break
            yield from entries
            offset += batch_size

    def _get_unprocessed_count(self) -> int:
        """Get count of unprocessed entries."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM wal_entries WHERE processed = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _create_checkpoint(self) -> WALCheckpoint | None:
        """Create a checkpoint and optionally compact."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Get last processed entry
            cursor.execute("""
                SELECT MAX(entry_id) FROM wal_entries WHERE processed = 1
            """)
            last_processed = cursor.fetchone()[0]

            if last_processed is None:
                conn.close()
                return None

            # Create checkpoint
            cursor.execute("""
                INSERT INTO wal_checkpoints (last_entry_id, timestamp)
                VALUES (?, ?)
            """, (last_processed, time.time()))
            checkpoint_id = cursor.lastrowid
            conn.commit()

            self._entries_since_checkpoint = 0

            # Auto compact if enabled
            if self.auto_compact:
                cursor.execute("""
                    DELETE FROM wal_entries
                    WHERE processed = 1 AND entry_id <= ?
                """, (last_processed,))
                compacted = cursor.rowcount

                cursor.execute("""
                    UPDATE wal_checkpoints
                    SET entries_compacted = ?
                    WHERE checkpoint_id = ?
                """, (compacted, checkpoint_id))
                conn.commit()

                logger.info(f"WAL checkpoint {checkpoint_id}: compacted {compacted} entries")

            conn.close()

            return _LegacyWALCheckpoint(
                checkpoint_id=checkpoint_id,
                last_entry_id=last_processed,
                timestamp=time.time(),
                entries_compacted=compacted if self.auto_compact else 0,
            )

    def compact(self, keep_days: int = 1) -> int:
        """Compact the WAL by removing old processed entries.

        Args:
            keep_days: Keep processed entries from last N days

        Returns:
            Number of entries removed
        """
        cutoff = time.time() - (keep_days * 86400)

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM wal_entries
            WHERE processed = 1 AND processed_at < ?
        """, (cutoff,))
        removed = cursor.rowcount
        conn.commit()

        # Vacuum to reclaim space
        cursor.execute("VACUUM")
        conn.close()

        logger.info(f"WAL compacted: removed {removed} entries older than {keep_days} days")
        return removed

    def recover(
        self,
        processor: Callable[[WALEntry], bool],
        batch_size: int = 100,
    ) -> tuple[int, int]:
        """Recover by replaying unprocessed entries.

        Args:
            processor: Function to process each entry (returns True on success)
            batch_size: Process entries in batches of this size

        Returns:
            (processed_count, failed_count)
        """
        processed = 0
        failed = 0

        # Record recovery start
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE wal_metadata SET value = ? WHERE key = 'last_recovery'",
            (str(time.time()),)
        )
        conn.commit()
        conn.close()

        logger.info("Starting WAL recovery...")

        for entry in self.iter_unprocessed(batch_size=batch_size):
            try:
                # Validate checksum
                expected_checksum = self._compute_checksum(entry.game_id, entry.data)
                if expected_checksum != entry.checksum:
                    logger.warning(f"WAL entry {entry.entry_id} checksum mismatch, skipping")
                    failed += 1
                    continue

                # Process entry
                success = processor(entry)
                if success:
                    self.mark_processed(entry.entry_id)
                    processed += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to process WAL entry {entry.entry_id}")

            except Exception as e:
                failed += 1
                logger.error(f"Error processing WAL entry {entry.entry_id}: {e}")

        logger.info(f"WAL recovery complete: {processed} processed, {failed} failed")
        return processed, failed

    def get_statistics(self) -> dict[str, Any]:
        """Get WAL statistics."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM wal_entries")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM wal_entries WHERE processed = 0")
        unprocessed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM wal_entries WHERE processed = 1")
        processed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM wal_checkpoints")
        checkpoints = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wal_entries")
        row = cursor.fetchone()
        oldest = row[0]
        newest = row[1]

        cursor.execute("SELECT value FROM wal_metadata WHERE key = 'last_recovery'")
        last_recovery = float(cursor.fetchone()[0])

        conn.close()

        return {
            "total_entries": total,
            "unprocessed": unprocessed,
            "processed": processed,
            "checkpoints": checkpoints,
            "oldest_entry": oldest,
            "newest_entry": newest,
            "last_recovery": last_recovery,
            "entries_since_checkpoint": self._entries_since_checkpoint,
        }

    def clear(self) -> None:
        """Clear all WAL entries (use with caution!)."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM wal_entries")
        cursor.execute("DELETE FROM wal_checkpoints")
        conn.commit()
        cursor.execute("VACUUM")
        conn.close()
        self._entries_since_checkpoint = 0
        logger.warning("WAL cleared")


# =============================================================================
# Public API - Conditional Export
# =============================================================================
#
# If unified_wal is available, use the consolidated implementation.
# Otherwise, fall back to the legacy implementation.
# =============================================================================

if HAS_UNIFIED_WAL:
    # Use unified implementation from unified_wal.py
    IngestionWAL = UnifiedIngestionWAL
    # Re-export from unified_wal for convenience
    WALEntry = UnifiedWALEntry
    WALCheckpoint = UnifiedWALCheckpoint
    logger.debug("Using UnifiedIngestionWAL from unified_wal.py")
else:
    # Use legacy implementation
    IngestionWAL = _LegacyIngestionWAL
    # Re-export legacy types with standard names
    WALEntry = _LegacyWALEntry
    WALCheckpoint = _LegacyWALCheckpoint
    logger.warning("unified_wal not available, using legacy IngestionWAL")


def create_ingestion_wal(
    data_dir: Path,
    max_unprocessed: int = 10000,
) -> IngestionWAL:
    """Factory function to create an ingestion WAL.

    Args:
        data_dir: Base data directory
        max_unprocessed: Maximum unprocessed entries

    Returns:
        Configured IngestionWAL instance (unified or legacy)
    """
    wal_dir = data_dir / "ingestion_wal"
    return IngestionWAL(
        wal_dir=wal_dir,
        max_unprocessed=max_unprocessed,
    )


# Re-export for backward compatibility
__all__ = [
    "IngestionWAL",
    "WALCheckpoint",
    "WALEntry",
    "create_ingestion_wal",
]
