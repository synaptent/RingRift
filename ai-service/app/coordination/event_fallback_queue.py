"""SQLite-based fallback event queue for cluster nodes.

January 4, 2026: Created to prevent event loss when cluster nodes can't reach
the event router. Events are queued locally in SQLite and synced to the
coordinator on the next sync cycle.

Problem: Cluster nodes (Vast.ai, RunPod, etc.) emit events but may not have
a running event_router. Previously these events were lost, causing:
- TRAINING_COMPLETED events to be missed
- Elo updates to be delayed
- Pipeline coordination failures

Solution: This module provides:
1. SQLite-backed event queue for local persistence
2. Automatic fallback when get_router() returns None
3. Sync mechanism to transfer queued events to coordinator
4. QUEUED_EVENTS_SYNCED event for monitoring

Usage:
    from app.coordination.event_fallback_queue import (
        queue_event_fallback,
        sync_queued_events,
        get_fallback_queue,
    )

    # When emit_event() fails, fall back to queue
    try:
        emit_event(event_type, payload)
    except Exception:
        queue_event_fallback(event_type, payload)

    # Periodically sync queued events
    synced_count = await sync_queued_events()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_QUEUE_DB_PATH = Path("data/state/event_fallback_queue.db")
MAX_QUEUE_SIZE = 10000  # Maximum events to queue before dropping oldest
MAX_EVENT_AGE_HOURS = 24  # Drop events older than this
SYNC_BATCH_SIZE = 100  # Events per sync batch


@dataclass
class FallbackQueueConfig:
    """Configuration for the fallback event queue."""

    db_path: Path = field(default_factory=lambda: DEFAULT_QUEUE_DB_PATH)
    max_queue_size: int = MAX_QUEUE_SIZE
    max_event_age_hours: float = MAX_EVENT_AGE_HOURS
    sync_batch_size: int = SYNC_BATCH_SIZE


# =============================================================================
# Event Fallback Queue
# =============================================================================


class EventFallbackQueue:
    """SQLite-backed fallback queue for events when router is unavailable.

    Thread-safe queue that persists events locally until they can be synced
    to the coordinator's event router.
    """

    _instance: "EventFallbackQueue | None" = None
    _lock = threading.Lock()

    def __init__(self, config: FallbackQueueConfig | None = None):
        """Initialize the fallback queue.

        Args:
            config: Queue configuration (uses defaults if None)
        """
        self.config = config or FallbackQueueConfig()
        self._db_lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False
        self._queued_count = 0
        self._synced_count = 0

    @classmethod
    def get_instance(cls, config: FallbackQueueConfig | None = None) -> "EventFallbackQueue":
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None

    def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if self._initialized:
            return

        with self._db_lock:
            if self._initialized:
                return

            # Create parent directory
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Open connection and create schema
            self._conn = sqlite3.connect(
                str(self.config.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS queued_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    source TEXT,
                    queued_at REAL NOT NULL,
                    synced_at REAL,
                    sync_attempts INTEGER DEFAULT 0
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queued_events_synced
                ON queued_events (synced_at)
            """)
            self._conn.commit()

            self._initialized = True
            logger.info(f"EventFallbackQueue initialized: {self.config.db_path}")

    @contextmanager
    def _get_cursor(self):
        """Get a cursor with proper locking."""
        self._ensure_initialized()
        with self._db_lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cursor.close()

    def queue_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str | None = None,
    ) -> bool:
        """Queue an event for later sync.

        Args:
            event_type: Event type name (e.g., "TRAINING_COMPLETED")
            payload: Event payload dictionary
            source: Optional source identifier

        Returns:
            True if event was queued, False if queue is full
        """
        try:
            self._ensure_initialized()

            # Serialize payload
            payload_json = json.dumps(payload, default=str)
            queued_at = time.time()

            with self._get_cursor() as cursor:
                # Check queue size
                cursor.execute("SELECT COUNT(*) FROM queued_events WHERE synced_at IS NULL")
                current_size = cursor.fetchone()[0]

                if current_size >= self.config.max_queue_size:
                    # Drop oldest events
                    drop_count = current_size - self.config.max_queue_size + 100
                    cursor.execute("""
                        DELETE FROM queued_events
                        WHERE id IN (
                            SELECT id FROM queued_events
                            WHERE synced_at IS NULL
                            ORDER BY queued_at ASC
                            LIMIT ?
                        )
                    """, (drop_count,))
                    logger.warning(f"Fallback queue full, dropped {drop_count} oldest events")

                # Insert event
                cursor.execute("""
                    INSERT INTO queued_events (event_type, payload, source, queued_at)
                    VALUES (?, ?, ?, ?)
                """, (event_type, payload_json, source, queued_at))

            self._queued_count += 1
            logger.debug(f"Queued event: {event_type} (total queued: {self._queued_count})")
            return True

        except Exception as e:
            logger.error(f"Failed to queue event {event_type}: {e}")
            return False

    def get_pending_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get pending (unsynced) events.

        Args:
            limit: Maximum events to return (uses sync_batch_size if None)

        Returns:
            List of event dictionaries with id, event_type, payload, queued_at
        """
        limit = limit or self.config.sync_batch_size
        events = []

        try:
            self._ensure_initialized()

            # Calculate age cutoff
            cutoff = time.time() - (self.config.max_event_age_hours * 3600)

            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, event_type, payload, source, queued_at
                    FROM queued_events
                    WHERE synced_at IS NULL AND queued_at > ?
                    ORDER BY queued_at ASC
                    LIMIT ?
                """, (cutoff, limit))

                for row in cursor.fetchall():
                    events.append({
                        "id": row[0],
                        "event_type": row[1],
                        "payload": json.loads(row[2]),
                        "source": row[3],
                        "queued_at": row[4],
                    })

        except Exception as e:
            logger.error(f"Failed to get pending events: {e}")

        return events

    def mark_synced(self, event_ids: list[int]) -> int:
        """Mark events as synced.

        Args:
            event_ids: List of event IDs to mark as synced

        Returns:
            Number of events marked
        """
        if not event_ids:
            return 0

        try:
            synced_at = time.time()

            with self._get_cursor() as cursor:
                placeholders = ",".join("?" * len(event_ids))
                cursor.execute(f"""
                    UPDATE queued_events
                    SET synced_at = ?
                    WHERE id IN ({placeholders})
                """, [synced_at] + event_ids)

            self._synced_count += len(event_ids)
            return len(event_ids)

        except Exception as e:
            logger.error(f"Failed to mark events as synced: {e}")
            return 0

    def cleanup_old_events(self) -> int:
        """Remove old synced and expired events.

        Returns:
            Number of events removed
        """
        try:
            self._ensure_initialized()
            cutoff = time.time() - (self.config.max_event_age_hours * 3600)

            with self._get_cursor() as cursor:
                # Remove synced events older than 1 hour
                cursor.execute("""
                    DELETE FROM queued_events
                    WHERE synced_at IS NOT NULL AND synced_at < ?
                """, (time.time() - 3600,))
                synced_removed = cursor.rowcount

                # Remove expired unsynced events
                cursor.execute("""
                    DELETE FROM queued_events
                    WHERE synced_at IS NULL AND queued_at < ?
                """, (cutoff,))
                expired_removed = cursor.rowcount

            total = synced_removed + expired_removed
            if total > 0:
                logger.info(f"Cleaned up {total} old events ({synced_removed} synced, {expired_removed} expired)")
            return total

        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        try:
            self._ensure_initialized()

            with self._get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM queued_events WHERE synced_at IS NULL")
                pending = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM queued_events WHERE synced_at IS NOT NULL")
                synced = cursor.fetchone()[0]

                cursor.execute("SELECT MIN(queued_at) FROM queued_events WHERE synced_at IS NULL")
                oldest_row = cursor.fetchone()[0]
                oldest_age = time.time() - oldest_row if oldest_row else 0

            return {
                "pending_count": pending,
                "synced_count": synced,
                "total_queued": self._queued_count,
                "total_synced": self._synced_count,
                "oldest_pending_age_seconds": oldest_age,
                "db_path": str(self.config.db_path),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close database connection."""
        with self._db_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._initialized = False


# =============================================================================
# Sync Functions
# =============================================================================


async def sync_queued_events(
    coordinator_url: str | None = None,
    batch_size: int = SYNC_BATCH_SIZE,
) -> int:
    """Sync queued events to coordinator's event router.

    January 4, 2026: Called periodically by sync daemons to transfer
    locally-queued events to the coordinator for cluster-wide propagation.

    Args:
        coordinator_url: Coordinator HTTP URL (auto-detected if None)
        batch_size: Events per batch

    Returns:
        Number of events successfully synced
    """
    queue = EventFallbackQueue.get_instance()
    events = queue.get_pending_events(limit=batch_size)

    if not events:
        return 0

    # Try to emit events via router
    synced_ids = []
    try:
        from app.distributed.data_events import DataEventType, get_event_bus

        bus = get_event_bus()

        for event in events:
            try:
                # Get event type enum value
                event_type_str = event["event_type"]
                try:
                    event_type = DataEventType(event_type_str)
                except ValueError:
                    event_type = DataEventType[event_type_str] if hasattr(DataEventType, event_type_str) else None

                if event_type:
                    # Add fallback queue metadata
                    payload = event["payload"]
                    payload["_fallback_queue"] = {
                        "queued_at": event["queued_at"],
                        "synced_at": time.time(),
                        "original_source": event.get("source"),
                    }
                    bus.emit(event_type, payload)
                    synced_ids.append(event["id"])
                else:
                    logger.warning(f"Unknown event type: {event_type_str}")
                    synced_ids.append(event["id"])  # Mark as synced anyway to avoid infinite loop

            except Exception as e:
                logger.warning(f"Failed to sync event {event['id']}: {e}")
                # Don't mark as synced, will retry next time

        # Mark synced events
        if synced_ids:
            queue.mark_synced(synced_ids)

            # Emit sync completion event
            try:
                bus.emit(
                    DataEventType.QUEUED_EVENTS_SYNCED,
                    {
                        "count": len(synced_ids),
                        "batch_size": batch_size,
                        "remaining": len(events) - len(synced_ids),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                pass  # Non-critical

    except ImportError:
        logger.warning("Cannot sync events: event router not available")
        return 0

    logger.info(f"Synced {len(synced_ids)} queued events to coordinator")
    return len(synced_ids)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_fallback_queue(config: FallbackQueueConfig | None = None) -> EventFallbackQueue:
    """Get the singleton fallback queue instance."""
    return EventFallbackQueue.get_instance(config)


def queue_event_fallback(
    event_type: str | Any,
    payload: dict[str, Any],
    source: str | None = None,
) -> bool:
    """Queue an event as fallback when router is unavailable.

    Usage:
        try:
            emit_event(DataEventType.TRAINING_COMPLETED, payload)
        except Exception:
            queue_event_fallback("TRAINING_COMPLETED", payload)
    """
    # Handle DataEventType enum
    if hasattr(event_type, "value"):
        event_type = event_type.value
    elif hasattr(event_type, "name"):
        event_type = event_type.name

    return get_fallback_queue().queue_event(str(event_type), payload, source)


def get_queue_stats() -> dict[str, Any]:
    """Get fallback queue statistics."""
    return get_fallback_queue().get_stats()


__all__ = [
    # Config
    "FallbackQueueConfig",
    # Main class
    "EventFallbackQueue",
    # Functions
    "get_fallback_queue",
    "queue_event_fallback",
    "sync_queued_events",
    "get_queue_stats",
]
