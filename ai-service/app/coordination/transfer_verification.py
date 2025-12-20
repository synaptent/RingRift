#!/usr/bin/env python3
"""Transfer Verification - Checksum verification for all data transfers.

This module provides checksum verification for data transfers to ensure
integrity and detect corruption:

1. Per-batch SHA256 checksums for streaming JSONL
2. Pre-transfer and post-transfer verification
3. Quarantine for failed/corrupted batches
4. Verification history tracking

Usage:
    from app.coordination.transfer_verification import (
        TransferVerifier,
        get_transfer_verifier,
        verify_transfer,
        compute_file_checksum,
        quarantine_file,
    )

    # Before transfer
    checksum = compute_file_checksum(source_path)

    # After transfer
    is_valid = verify_transfer(
        source_checksum=checksum,
        dest_path=dest_path,
    )

    if not is_valid:
        quarantine_file(dest_path, reason="checksum_mismatch")
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import (
    LARGE_CHUNK_SIZE,
    compute_bytes_checksum as _compute_bytes_checksum,
    compute_file_checksum as _compute_file_checksum,
)
from app.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_VERIFIER_DB = DATA_DIR / "coordination" / "transfer_verification.db"
QUARANTINE_DIR = DATA_DIR / "quarantine"

# Configuration (use centralized chunk size)
CHUNK_SIZE = LARGE_CHUNK_SIZE  # 64KB chunks for checksum computation
MAX_QUARANTINE_AGE_DAYS = 30  # Auto-cleanup quarantine after 30 days


@dataclass
class TransferRecord:
    """Record of a verified transfer."""
    transfer_id: int
    source_path: str
    dest_path: str
    source_checksum: str
    dest_checksum: str
    file_size: int
    transfer_time: float
    verified: bool
    verification_time: float
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "source_checksum": self.source_checksum[:16] + "..." if self.source_checksum else "",
            "dest_checksum": self.dest_checksum[:16] + "..." if self.dest_checksum else "",
            "file_size": self.file_size,
            "transfer_time": datetime.fromtimestamp(self.transfer_time).isoformat(),
            "verified": self.verified,
            "error_message": self.error_message,
        }


@dataclass
class QuarantineRecord:
    """Record of a quarantined file."""
    quarantine_id: int
    original_path: str
    quarantine_path: str
    reason: str
    quarantined_at: float
    file_size: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchChecksum:
    """Checksum for a batch of JSONL records."""
    batch_id: str
    checksum: str
    record_count: int
    byte_count: int
    first_record_id: str
    last_record_id: str
    created_at: float


class TransferVerifier:
    """Verifies integrity of data transfers with checksum validation."""

    _instance: TransferVerifier | None = None
    _lock = threading.RLock()

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_VERIFIER_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir = QUARANTINE_DIR
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Path | None = None) -> TransferVerifier:
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # Database Management
    # =========================================================================

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=10,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Transfer records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfer_records (
                transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                dest_path TEXT NOT NULL,
                source_checksum TEXT NOT NULL,
                dest_checksum TEXT DEFAULT '',
                file_size INTEGER DEFAULT 0,
                transfer_time REAL NOT NULL,
                verified INTEGER DEFAULT 0,
                verification_time REAL DEFAULT 0,
                error_message TEXT DEFAULT ''
            )
        """)

        # Quarantine records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quarantine_records (
                quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path TEXT NOT NULL,
                quarantine_path TEXT NOT NULL,
                reason TEXT NOT NULL,
                quarantined_at REAL NOT NULL,
                file_size INTEGER DEFAULT 0,
                checksum TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Batch checksums for streaming
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_checksums (
                batch_id TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                record_count INTEGER DEFAULT 0,
                byte_count INTEGER DEFAULT 0,
                first_record_id TEXT DEFAULT '',
                last_record_id TEXT DEFAULT '',
                created_at REAL NOT NULL
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transfers_time
            ON transfer_records(transfer_time DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quarantine_time
            ON quarantine_records(quarantined_at DESC)
        """)

        conn.commit()

    # =========================================================================
    # Checksum Computation
    # =========================================================================

    def compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        return _compute_file_checksum(path, chunk_size=CHUNK_SIZE)

    def compute_checksum_from_bytes(self, data: bytes) -> str:
        """Compute SHA256 checksum of bytes."""
        return _compute_bytes_checksum(data)

    def compute_batch_checksum(
        self,
        records: list[str],
        batch_id: str,
    ) -> BatchChecksum:
        """Compute checksum for a batch of JSONL records.

        Args:
            records: List of JSONL record strings
            batch_id: Unique identifier for this batch

        Returns:
            BatchChecksum with computed checksum
        """
        if not records:
            return BatchChecksum(
                batch_id=batch_id,
                checksum="",
                record_count=0,
                byte_count=0,
                first_record_id="",
                last_record_id="",
                created_at=time.time(),
            )

        # Concatenate all records for checksum
        combined = "\n".join(records)
        checksum = self.compute_checksum_from_bytes(combined.encode('utf-8'))

        # Extract game IDs from first and last records
        first_record_id = ""
        last_record_id = ""
        try:
            first_data = json.loads(records[0])
            first_record_id = first_data.get("game_id", "")
            last_data = json.loads(records[-1])
            last_record_id = last_data.get("game_id", "")
        except (json.JSONDecodeError, IndexError):
            pass

        batch = BatchChecksum(
            batch_id=batch_id,
            checksum=checksum,
            record_count=len(records),
            byte_count=len(combined.encode('utf-8')),
            first_record_id=first_record_id,
            last_record_id=last_record_id,
            created_at=time.time(),
        )

        # Store in database
        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO batch_checksums
            (batch_id, checksum, record_count, byte_count, first_record_id, last_record_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (batch.batch_id, batch.checksum, batch.record_count, batch.byte_count,
              batch.first_record_id, batch.last_record_id, batch.created_at))
        conn.commit()

        return batch

    def verify_batch(self, batch_id: str, records: list[str]) -> bool:
        """Verify a batch of records against stored checksum.

        Returns True if checksum matches.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT checksum, record_count FROM batch_checksums WHERE batch_id = ?",
            (batch_id,)
        )
        row = cursor.fetchone()

        if not row:
            logger.warning(f"[TransferVerifier] No stored checksum for batch {batch_id}")
            return False

        stored_checksum = row["checksum"]
        stored_count = row["record_count"]

        # Verify record count
        if len(records) != stored_count:
            logger.warning(f"[TransferVerifier] Batch {batch_id} record count mismatch: "
                          f"expected {stored_count}, got {len(records)}")
            return False

        # Verify checksum
        combined = "\n".join(records)
        computed_checksum = self.compute_checksum_from_bytes(combined.encode('utf-8'))

        if computed_checksum != stored_checksum:
            logger.warning(f"[TransferVerifier] Batch {batch_id} checksum mismatch")
            return False

        return True

    # =========================================================================
    # Transfer Verification
    # =========================================================================

    def record_transfer(
        self,
        source_path: str,
        dest_path: str,
        source_checksum: str,
    ) -> int:
        """Record start of a transfer."""
        conn = self._get_connection()
        cursor = conn.execute("""
            INSERT INTO transfer_records
            (source_path, dest_path, source_checksum, transfer_time)
            VALUES (?, ?, ?, ?)
        """, (source_path, dest_path, source_checksum, time.time()))
        conn.commit()
        return cursor.lastrowid

    def verify_transfer(
        self,
        transfer_id: int,
        dest_path: Path,
    ) -> bool:
        """Verify a completed transfer.

        Returns True if checksums match.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT source_checksum FROM transfer_records WHERE transfer_id = ?",
            (transfer_id,)
        )
        row = cursor.fetchone()

        if not row:
            return False

        source_checksum = row["source_checksum"]
        dest_checksum = self.compute_checksum(dest_path)

        verified = source_checksum == dest_checksum
        error_message = "" if verified else f"Checksum mismatch: source={source_checksum[:16]}... dest={dest_checksum[:16]}..."

        file_size = dest_path.stat().st_size if dest_path.exists() else 0

        conn.execute("""
            UPDATE transfer_records
            SET dest_checksum = ?, file_size = ?, verified = ?, verification_time = ?, error_message = ?
            WHERE transfer_id = ?
        """, (dest_checksum, file_size, 1 if verified else 0, time.time(), error_message, transfer_id))
        conn.commit()

        if not verified:
            logger.warning(f"[TransferVerifier] Transfer {transfer_id} failed verification: {error_message}")

        return verified

    def quick_verify(self, source_checksum: str, dest_path: Path) -> bool:
        """Quick verification without recording."""
        dest_checksum = self.compute_checksum(dest_path)
        return source_checksum == dest_checksum

    # =========================================================================
    # Quarantine Management
    # =========================================================================

    def quarantine(
        self,
        file_path: Path,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Move a file to quarantine.

        Returns the quarantine path.
        """
        if not file_path.exists():
            logger.warning(f"[TransferVerifier] Cannot quarantine non-existent file: {file_path}")
            return ""

        # Generate quarantine path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_name = f"{timestamp}_{file_path.name}"
        quarantine_path = self.quarantine_dir / quarantine_name

        # Compute checksum before moving
        checksum = self.compute_checksum(file_path)
        file_size = file_path.stat().st_size

        # Move to quarantine
        shutil.move(str(file_path), str(quarantine_path))

        # Record in database
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO quarantine_records
            (original_path, quarantine_path, reason, quarantined_at, file_size, checksum, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            str(quarantine_path),
            reason,
            time.time(),
            file_size,
            checksum,
            json.dumps(metadata or {}),
        ))
        conn.commit()

        logger.warning(f"[TransferVerifier] Quarantined {file_path} -> {quarantine_path}: {reason}")

        return str(quarantine_path)

    def get_quarantined_files(self, limit: int = 100) -> list[QuarantineRecord]:
        """Get list of quarantined files."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM quarantine_records
            ORDER BY quarantined_at DESC
            LIMIT ?
        """, (limit,))

        return [
            QuarantineRecord(
                quarantine_id=row["quarantine_id"],
                original_path=row["original_path"],
                quarantine_path=row["quarantine_path"],
                reason=row["reason"],
                quarantined_at=row["quarantined_at"],
                file_size=row["file_size"],
                checksum=row["checksum"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def cleanup_old_quarantine(self) -> int:
        """Remove quarantined files older than MAX_QUARANTINE_AGE_DAYS."""
        cutoff = time.time() - (MAX_QUARANTINE_AGE_DAYS * 86400)

        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT quarantine_path FROM quarantine_records WHERE quarantined_at < ?",
            (cutoff,)
        )

        deleted = 0
        for row in cursor.fetchall():
            path = Path(row["quarantine_path"])
            if path.exists():
                try:
                    path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"[TransferVerifier] Failed to delete {path}: {e}")

        # Remove from database
        conn.execute("DELETE FROM quarantine_records WHERE quarantined_at < ?", (cutoff,))
        conn.commit()

        if deleted > 0:
            logger.info(f"[TransferVerifier] Cleaned up {deleted} old quarantine files")

        return deleted

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get transfer verification statistics."""
        conn = self._get_connection()

        # Transfer stats
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_transfers,
                SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified_count,
                SUM(CASE WHEN verified = 0 AND verification_time > 0 THEN 1 ELSE 0 END) as failed_count,
                SUM(file_size) as total_bytes
            FROM transfer_records
            WHERE transfer_time > ?
        """, (time.time() - 86400,))  # Last 24 hours
        row = cursor.fetchone()

        # Quarantine stats
        cursor = conn.execute("SELECT COUNT(*) as count FROM quarantine_records")
        quarantine_count = cursor.fetchone()["count"]

        return {
            "transfers_24h": row["total_transfers"] or 0,
            "verified_24h": row["verified_count"] or 0,
            "failed_24h": row["failed_count"] or 0,
            "bytes_transferred_24h": row["total_bytes"] or 0,
            "quarantine_count": quarantine_count,
            "verification_rate": (
                (row["verified_count"] or 0) / (row["total_transfers"] or 1)
            ),
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_transfer_verifier() -> TransferVerifier:
    """Get the singleton transfer verifier."""
    return TransferVerifier.get_instance()


def compute_file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    return get_transfer_verifier().compute_checksum(path)


def verify_transfer(
    source_checksum: str,
    dest_path: Path,
) -> bool:
    """Verify a transfer matches expected checksum."""
    return get_transfer_verifier().quick_verify(source_checksum, dest_path)


def quarantine_file(
    file_path: Path,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Move a file to quarantine."""
    return get_transfer_verifier().quarantine(file_path, reason, metadata)


def verify_batch(batch_id: str, records: list[str]) -> bool:
    """Verify a batch of JSONL records."""
    return get_transfer_verifier().verify_batch(batch_id, records)


def compute_batch_checksum(records: list[str], batch_id: str) -> BatchChecksum:
    """Compute checksum for a batch of records."""
    return get_transfer_verifier().compute_batch_checksum(records, batch_id)


def reset_transfer_verifier() -> None:
    """Reset the singleton."""
    TransferVerifier.reset_instance()


def wire_transfer_verifier_events() -> TransferVerifier:
    """Wire transfer verifier to the event bus for automatic verification.

    Subscribes to:
    - DATA_SYNC_COMPLETED: Verify data integrity after sync

    Returns:
        The configured TransferVerifier instance
    """
    verifier = get_transfer_verifier()

    try:
        from app.distributed.data_events import DataEventType, get_event_bus

        bus = get_event_bus()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_sync_completed(event: Any) -> None:
            """Handle sync completion - verify transferred data."""
            payload = _event_payload(event)
            source_checksum = payload.get("source_checksum")
            dest_path = payload.get("dest_path")
            if source_checksum and dest_path:
                from pathlib import Path
                is_valid = verifier.quick_verify(source_checksum, Path(dest_path))
                if not is_valid:
                    logger.warning(
                        f"[TransferVerifier] Verification failed for {dest_path}"
                    )

        bus.subscribe(DataEventType.DATA_SYNC_COMPLETED, _on_sync_completed)

        logger.info("[TransferVerifier] Wired to event bus (DATA_SYNC_COMPLETED)")

    except ImportError:
        logger.warning("[TransferVerifier] data_events not available, running without event bus")

    return verifier


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Constants
    "CHUNK_SIZE",
    "MAX_QUARANTINE_AGE_DAYS",
    "BatchChecksum",
    "QuarantineRecord",
    # Data classes
    "TransferRecord",
    # Main class
    "TransferVerifier",
    "compute_batch_checksum",
    "compute_file_checksum",
    # Functions
    "get_transfer_verifier",
    "quarantine_file",
    "reset_transfer_verifier",
    "verify_batch",
    "verify_transfer",
    "wire_transfer_verifier_events",
]
