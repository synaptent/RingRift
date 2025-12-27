"""Persistent delivery ledger for tracking data distribution to cluster nodes.

Provides SQLite-backed persistence for delivery tracking, enabling:
- Visibility into data distribution status per node
- Automatic retry of failed deliveries
- Recovery after orchestrator restarts

December 2025: Created as part of Phase 3 infrastructure improvements.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DeliveryLedger",
    "DeliveryRecord",
    "DeliveryStatus",
    "get_delivery_ledger",
    "reset_delivery_ledger",
]

# Singleton instance
_ledger_instance: "DeliveryLedger | None" = None


class DeliveryStatus(Enum):
    """Status of a delivery operation."""
    PENDING = "pending"           # Queued for delivery
    TRANSFERRING = "transferring" # Transfer in progress
    TRANSFERRED = "transferred"   # Transfer complete, awaiting verification
    VERIFIED = "verified"         # Checksum verified on target
    FAILED = "failed"             # Delivery failed


@dataclass
class DeliveryRecord:
    """Record of a single delivery operation.

    Tracks the complete lifecycle of distributing data to a target node.
    """
    delivery_id: str
    data_type: str  # "model", "npz", "torrent", "games"
    data_path: str
    target_node: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    retry_count: int = 0
    max_retries: int = 4
    checksum: str = ""
    verified_checksum: str = ""
    file_size_bytes: int = 0
    transfer_method: str = ""  # http, rsync, bittorrent
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""

    @property
    def transfer_duration_seconds(self) -> float:
        """Calculate transfer duration."""
        if self.completed_at > 0 and self.started_at > 0:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def age_seconds(self) -> float:
        """Age of the delivery record in seconds."""
        return time.time() - self.created_at

    @property
    def is_terminal(self) -> bool:
        """Whether the delivery is in a terminal state."""
        return self.status in (DeliveryStatus.VERIFIED, DeliveryStatus.FAILED)

    @property
    def can_retry(self) -> bool:
        """Whether the delivery can be retried."""
        return self.status == DeliveryStatus.FAILED and self.retry_count < self.max_retries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "delivery_id": self.delivery_id,
            "data_type": self.data_type,
            "data_path": self.data_path,
            "target_node": self.target_node,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "checksum": self.checksum,
            "verified_checksum": self.verified_checksum,
            "file_size_bytes": self.file_size_bytes,
            "transfer_method": self.transfer_method,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "transfer_duration_seconds": self.transfer_duration_seconds,
        }


class DeliveryLedger:
    """SQLite-backed persistent ledger for delivery tracking.

    Provides durable tracking of data distribution to cluster nodes,
    with support for retry, verification, and status reporting.
    """

    DEFAULT_DB_PATH = Path("data/coordination/delivery_ledger.db")

    def __init__(self, db_path: Path | None = None):
        """Initialize the delivery ledger.

        Args:
            db_path: Path to SQLite database. Uses DEFAULT_DB_PATH if not specified.
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 4,
                    checksum TEXT DEFAULT '',
                    verified_checksum TEXT DEFAULT '',
                    file_size_bytes INTEGER DEFAULT 0,
                    transfer_method TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL DEFAULT 0,
                    completed_at REAL DEFAULT 0,
                    error_message TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliveries_status
                ON deliveries(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliveries_target_node
                ON deliveries(target_node)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deliveries_created_at
                ON deliveries(created_at)
            """)
            conn.commit()

    def _record_from_row(self, row: tuple) -> DeliveryRecord:
        """Convert database row to DeliveryRecord."""
        return DeliveryRecord(
            delivery_id=row[0],
            data_type=row[1],
            data_path=row[2],
            target_node=row[3],
            status=DeliveryStatus(row[4]),
            retry_count=row[5],
            max_retries=row[6],
            checksum=row[7],
            verified_checksum=row[8],
            file_size_bytes=row[9],
            transfer_method=row[10],
            created_at=row[11],
            updated_at=row[12],
            started_at=row[13],
            completed_at=row[14],
            error_message=row[15],
        )

    def record_delivery_started(
        self,
        data_type: str,
        data_path: str,
        target_node: str,
        checksum: str = "",
        file_size_bytes: int = 0,
        transfer_method: str = "http",
        max_retries: int = 4,
    ) -> DeliveryRecord:
        """Record a new delivery operation starting.

        Args:
            data_type: Type of data (model, npz, games, etc.)
            data_path: Path to the source data
            target_node: Target node identifier
            checksum: SHA256 checksum of the data
            file_size_bytes: Size of the data in bytes
            transfer_method: Method used for transfer
            max_retries: Maximum retry attempts

        Returns:
            The created DeliveryRecord
        """
        delivery_id = str(uuid.uuid4())
        now = time.time()

        record = DeliveryRecord(
            delivery_id=delivery_id,
            data_type=data_type,
            data_path=data_path,
            target_node=target_node,
            status=DeliveryStatus.TRANSFERRING,
            checksum=checksum,
            file_size_bytes=file_size_bytes,
            transfer_method=transfer_method,
            max_retries=max_retries,
            created_at=now,
            updated_at=now,
            started_at=now,
        )

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO deliveries (
                    delivery_id, data_type, data_path, target_node, status,
                    retry_count, max_retries, checksum, verified_checksum,
                    file_size_bytes, transfer_method, created_at, updated_at,
                    started_at, completed_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.delivery_id, record.data_type, record.data_path,
                record.target_node, record.status.value, record.retry_count,
                record.max_retries, record.checksum, record.verified_checksum,
                record.file_size_bytes, record.transfer_method, record.created_at,
                record.updated_at, record.started_at, record.completed_at,
                record.error_message,
            ))
            conn.commit()

        logger.debug(
            f"[DeliveryLedger] Started delivery {delivery_id[:8]} "
            f"of {data_type} to {target_node}"
        )
        return record

    def record_delivery_transferred(
        self,
        delivery_id: str,
        checksum: str = "",
    ) -> None:
        """Record that transfer completed, awaiting verification.

        Args:
            delivery_id: The delivery identifier
            checksum: Optional checksum if not provided at start
        """
        now = time.time()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                UPDATE deliveries
                SET status = ?, updated_at = ?, completed_at = ?, checksum = COALESCE(NULLIF(?, ''), checksum)
                WHERE delivery_id = ?
            """, (DeliveryStatus.TRANSFERRED.value, now, now, checksum, delivery_id))
            conn.commit()

        logger.debug(f"[DeliveryLedger] Transfer complete for {delivery_id[:8]}")

    def record_delivery_verified(
        self,
        delivery_id: str,
        verified_checksum: str = "",
    ) -> None:
        """Record that delivery was verified on target node.

        Args:
            delivery_id: The delivery identifier
            verified_checksum: Checksum computed on target node
        """
        now = time.time()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                UPDATE deliveries
                SET status = ?, updated_at = ?, verified_checksum = ?
                WHERE delivery_id = ?
            """, (DeliveryStatus.VERIFIED.value, now, verified_checksum, delivery_id))
            conn.commit()

        logger.debug(f"[DeliveryLedger] Verified delivery {delivery_id[:8]}")

    def record_delivery_failed(
        self,
        delivery_id: str,
        error: str,
        increment_retry: bool = True,
    ) -> None:
        """Record that a delivery failed.

        Args:
            delivery_id: The delivery identifier
            error: Error message describing the failure
            increment_retry: Whether to increment retry count
        """
        now = time.time()

        with sqlite3.connect(str(self.db_path)) as conn:
            if increment_retry:
                conn.execute("""
                    UPDATE deliveries
                    SET status = ?, updated_at = ?, error_message = ?,
                        retry_count = retry_count + 1
                    WHERE delivery_id = ?
                """, (DeliveryStatus.FAILED.value, now, error, delivery_id))
            else:
                conn.execute("""
                    UPDATE deliveries
                    SET status = ?, updated_at = ?, error_message = ?
                    WHERE delivery_id = ?
                """, (DeliveryStatus.FAILED.value, now, error, delivery_id))
            conn.commit()

        logger.debug(f"[DeliveryLedger] Failed delivery {delivery_id[:8]}: {error}")

    def record_retry_started(self, delivery_id: str) -> None:
        """Record that a retry attempt has started.

        Args:
            delivery_id: The delivery identifier
        """
        now = time.time()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                UPDATE deliveries
                SET status = ?, updated_at = ?, started_at = ?
                WHERE delivery_id = ?
            """, (DeliveryStatus.TRANSFERRING.value, now, now, delivery_id))
            conn.commit()

        logger.debug(f"[DeliveryLedger] Retry started for {delivery_id[:8]}")

    def get_delivery(self, delivery_id: str) -> DeliveryRecord | None:
        """Get a specific delivery record.

        Args:
            delivery_id: The delivery identifier

        Returns:
            DeliveryRecord if found, None otherwise
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT * FROM deliveries WHERE delivery_id = ?
            """, (delivery_id,))
            row = cursor.fetchone()
            if row:
                return self._record_from_row(row)
        return None

    def get_pending_deliveries(
        self,
        max_age_hours: float = 24.0,
    ) -> list[DeliveryRecord]:
        """Get deliveries that are still pending (not in terminal state).

        Args:
            max_age_hours: Maximum age of deliveries to include

        Returns:
            List of pending DeliveryRecords
        """
        cutoff = time.time() - (max_age_hours * 3600)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT * FROM deliveries
                WHERE status NOT IN (?, ?) AND created_at > ?
                ORDER BY created_at ASC
            """, (DeliveryStatus.VERIFIED.value, DeliveryStatus.FAILED.value, cutoff))
            return [self._record_from_row(row) for row in cursor.fetchall()]

    def get_retryable_deliveries(self, max_age_hours: float = 24.0) -> list[DeliveryRecord]:
        """Get failed deliveries that can be retried.

        Args:
            max_age_hours: Maximum age of deliveries to include

        Returns:
            List of retryable DeliveryRecords
        """
        cutoff = time.time() - (max_age_hours * 3600)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT * FROM deliveries
                WHERE status = ? AND retry_count < max_retries AND created_at > ?
                ORDER BY updated_at ASC
            """, (DeliveryStatus.FAILED.value, cutoff))
            return [self._record_from_row(row) for row in cursor.fetchall()]

    def get_deliveries_for_node(
        self,
        node_id: str,
        limit: int = 100,
    ) -> list[DeliveryRecord]:
        """Get recent deliveries for a specific node.

        Args:
            node_id: Target node identifier
            limit: Maximum number of records to return

        Returns:
            List of DeliveryRecords for the node
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT * FROM deliveries
                WHERE target_node = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (node_id, limit))
            return [self._record_from_row(row) for row in cursor.fetchall()]

    def get_node_delivery_status(self, node_id: str) -> dict[str, Any]:
        """Get delivery status summary for a node.

        Args:
            node_id: Target node identifier

        Returns:
            Dictionary with delivery statistics for the node
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # Count by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM deliveries
                WHERE target_node = ?
                GROUP BY status
            """, (node_id,))
            status_counts = dict(cursor.fetchall())

            # Get recent deliveries
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(completed_at - started_at) FROM deliveries
                WHERE target_node = ? AND status = ? AND completed_at > 0
            """, (node_id, DeliveryStatus.VERIFIED.value))
            row = cursor.fetchone()
            verified_count = row[0] or 0
            avg_transfer_time = row[1] or 0.0

            # Get failure rate
            cursor = conn.execute("""
                SELECT COUNT(*) FROM deliveries
                WHERE target_node = ? AND created_at > ?
            """, (node_id, time.time() - 86400))  # Last 24 hours
            total_24h = cursor.fetchone()[0] or 1

            cursor = conn.execute("""
                SELECT COUNT(*) FROM deliveries
                WHERE target_node = ? AND status = ? AND created_at > ?
            """, (node_id, DeliveryStatus.FAILED.value, time.time() - 86400))
            failed_24h = cursor.fetchone()[0] or 0

        return {
            "node_id": node_id,
            "status_counts": status_counts,
            "total_verified": verified_count,
            "avg_transfer_time_seconds": avg_transfer_time,
            "failure_rate_24h": failed_24h / total_24h if total_24h > 0 else 0.0,
            "failed_24h": failed_24h,
            "total_24h": total_24h,
        }

    def get_overall_status(self) -> dict[str, Any]:
        """Get overall delivery status summary.

        Returns:
            Dictionary with overall delivery statistics
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # Count by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM deliveries GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Get totals
            cursor = conn.execute("SELECT COUNT(*) FROM deliveries")
            total_deliveries = cursor.fetchone()[0]

            # Get success rate
            verified = status_counts.get(DeliveryStatus.VERIFIED.value, 0)
            failed = status_counts.get(DeliveryStatus.FAILED.value, 0)

            # Get unique nodes
            cursor = conn.execute("SELECT COUNT(DISTINCT target_node) FROM deliveries")
            unique_nodes = cursor.fetchone()[0]

        success_rate = verified / (verified + failed) if (verified + failed) > 0 else 0.0

        return {
            "total_deliveries": total_deliveries,
            "status_counts": status_counts,
            "verified": verified,
            "failed": failed,
            "success_rate": success_rate,
            "unique_nodes": unique_nodes,
        }

    def cleanup_old_records(self, max_age_days: int = 30) -> int:
        """Remove old delivery records.

        Args:
            max_age_days: Maximum age of records to keep

        Returns:
            Number of records deleted
        """
        cutoff = time.time() - (max_age_days * 86400)

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                DELETE FROM deliveries
                WHERE created_at < ? AND status IN (?, ?)
            """, (cutoff, DeliveryStatus.VERIFIED.value, DeliveryStatus.FAILED.value))
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"[DeliveryLedger] Cleaned up {deleted} old delivery records")

        return deleted


def get_delivery_ledger() -> DeliveryLedger:
    """Get the singleton DeliveryLedger instance.

    Returns:
        The global DeliveryLedger instance
    """
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = DeliveryLedger()
    return _ledger_instance


def reset_delivery_ledger() -> None:
    """Reset the singleton DeliveryLedger instance.

    Used primarily for testing.
    """
    global _ledger_instance
    _ledger_instance = None
