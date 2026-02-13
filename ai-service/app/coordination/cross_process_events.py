#!/usr/bin/env python3
"""Cross-Process Event Queue using SQLite.

.. deprecated:: December 2025
    This module is being superseded by the unified event router.
    For new code, prefer using:

        from app.coordination.event_router import (
            get_router, publish, subscribe
        )

    The unified router automatically routes events to the cross-process queue
    when cross_process=True. This module remains functional for backwards
    compatibility and is used internally by the router.

This module provides a SQLite-backed event queue that allows different
processes (cluster_orchestrator, continuous_improvement_daemon, pipeline_orchestrator)
to communicate events across process boundaries.

Unlike the in-memory EventBus in data_events.py, this queue persists events
to disk and supports polling-based consumption from multiple processes.

Usage:
    from app.coordination.cross_process_events import (
        CrossProcessEventQueue,
        publish_event,
        poll_events,
        subscribe_process,
    )

    # Publish an event (from any process)
    publish_event(
        event_type="MODEL_PROMOTED",
        payload={"model_id": "abc123", "elo": 1850},
        source="improvement_daemon"
    )

    # Subscribe and poll for events (in a consumer process)
    subscriber_id = subscribe_process("pipeline_orchestrator")
    events = poll_events(subscriber_id, event_types=["MODEL_PROMOTED", "TRAINING_COMPLETED"])
    for event in events:
        handle_event(event)
        ack_event(subscriber_id, event["event_id"])
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_user_coordination_dir() -> Path:
    """Get user-specific coordination directory to avoid permission conflicts.

    This prevents conflicts when P2P orchestrator (root) and master_loop (ubuntu)
    both try to access the same coordination files.
    """
    # Allow override via environment
    custom_dir = os.environ.get("RINGRIFT_COORDINATION_DIR")
    if custom_dir:
        return Path(custom_dir)

    # Use XDG_RUNTIME_DIR if available (properly permissioned)
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / "ringrift_coordination"

    # Fall back to user-specific /tmp directory
    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0  # Windows

    if uid == 0:
        return Path("/tmp/ringrift_coordination")
    else:
        return Path(f"/tmp/ringrift_coordination_{uid}")


# Default database location - user-specific to avoid permission conflicts
DEFAULT_EVENT_DB = _get_user_coordination_dir() / "events.db"

# Import centralized timeout thresholds
try:
    from app.config.thresholds import SQLITE_BUSY_TIMEOUT_LONG_MS, SQLITE_TIMEOUT
except ImportError:
    SQLITE_BUSY_TIMEOUT_LONG_MS = 30000
    SQLITE_TIMEOUT = 30

# Event retention/timeout (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import CrossProcessDefaults
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.utils.retry import RetryConfig

DEFAULT_RETENTION_HOURS = CrossProcessDefaults.RETENTION_HOURS
SUBSCRIBER_TIMEOUT_SECONDS = CrossProcessDefaults.SUBSCRIBER_TIMEOUT

# Emit deprecation warning at import time (RR-CONSOLIDATION-2025-12)
# Only warn if imported directly, not when imported by event_router internally
import warnings as _warnings
import traceback as _traceback

def _is_internal_import() -> bool:
    """Check if this import is from event_router (internal use)."""
    stack = _traceback.extract_stack()
    for frame in stack:
        if "event_router.py" in frame.filename:
            return True
    return False

if not _is_internal_import():
    _warnings.warn(
        "cross_process_events is deprecated. Import from event_router instead:\n"
        "  from app.coordination.event_router import (\n"
        "      CrossProcessEvent, CrossProcessEventQueue, bridge_to_cross_process, ...\n"
        "  )\n"
        "This module will be removed in Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )


@dataclass
class CrossProcessEvent:
    """An event that can be shared across processes."""

    event_id: int
    event_type: str
    payload: dict[str, Any]
    source: str
    created_at: float
    hostname: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "created_at": self.created_at,
            "hostname": self.hostname,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CrossProcessEvent:
        """Create a CrossProcessEvent from a SQLite row.

        Args:
            row: SQLite Row object with event_id, event_type, payload, source,
                 created_at, and hostname columns.

        Returns:
            CrossProcessEvent instance reconstructed from the database row.
        """
        return cls(
            event_id=row["event_id"],
            event_type=row["event_type"],
            payload=json.loads(row["payload"]),
            source=row["source"],
            created_at=row["created_at"],
            hostname=row["hostname"],
        )


class CrossProcessEventQueue:
    """SQLite-backed event queue for cross-process communication.

    Features:
    - WAL mode for concurrent access
    - Subscriber tracking with heartbeat
    - Event acknowledgment for reliable delivery
    - Automatic cleanup of old events and dead subscribers
    - Thread-safe with connection pooling
    """

    def __init__(self, db_path: Path | str | None = None, retention_hours: int = DEFAULT_RETENTION_HOURS):
        # December 2025: Always convert to Path for consistency
        if db_path is None:
            self.db_path = DEFAULT_EVENT_DB
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path
        self.retention_hours = retention_hours
        self._local = threading.local()
        # December 2025: Lazy initialization for readonly filesystem support
        self._db_initialized = False
        self._readonly_mode = False
        # Defer db_path.parent.mkdir and _init_db to first use

    def _ensure_db(self) -> bool:
        """Lazily initialize database, returns True if writable.

        December 2025: This enables import on read-only filesystems.
        Database initialization is deferred until first actual use.

        Returns:
            True if database is writable, False if readonly or unavailable
        """
        if self._db_initialized:
            return not self._readonly_mode

        try:
            # Create parent directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._db_initialized = True
            return True
        except sqlite3.OperationalError as e:
            if "readonly" in str(e).lower() or "read-only" in str(e).lower():
                self._readonly_mode = True
                self._db_initialized = True
                logger.warning(f"[CrossProcessEventQueue] Readonly mode enabled: {e}")
                return False
            raise
        except PermissionError as e:
            self._readonly_mode = True
            self._db_initialized = True
            logger.warning(f"[CrossProcessEventQueue] Readonly mode (permission denied): {e}")
            return False
        except OSError as e:
            if "Read-only file system" in str(e):
                self._readonly_mode = True
                self._db_initialized = True
                logger.warning(f"[CrossProcessEventQueue] Readonly mode (filesystem): {e}")
                return False
            raise

    def _get_connection(self, _from_init: bool = False) -> sqlite3.Connection:
        """Get thread-local database connection with retry on BUSY errors.

        Args:
            _from_init: Internal flag to skip _ensure_db() when called from _init_db().
                        This prevents circular recursion: _ensure_db -> _init_db -> _get_connection -> _ensure_db.

        January 3, 2026: Migrated to RetryConfig for centralized retry behavior.
        """
        # Ensure database is initialized (lazy init)
        # Skip when called from _init_db to prevent recursion (December 2025 fix)
        if not _from_init:
            self._ensure_db()

        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Retry with jitter to prevent thundering herd
            retry_config = RetryConfig(
                max_attempts=3,
                base_delay=0.1,
                max_delay=0.8,
                jitter=0.5,  # 50% jitter for thundering herd prevention
            )

            for attempt in retry_config.attempts():
                try:
                    self._local.conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=float(SQLITE_TIMEOUT * 2),  # 60s for cross-process events
                        isolation_level=None,  # Autocommit for better concurrency
                    )
                    self._local.conn.row_factory = sqlite3.Row
                    # WAL mode for concurrent access
                    self._local.conn.execute('PRAGMA journal_mode=WAL')
                    self._local.conn.execute(f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_LONG_MS}')
                    self._local.conn.execute('PRAGMA synchronous=NORMAL')
                    # Feb 2026: Memory-limiting pragmas to prevent OOM on coordinator
                    self._local.conn.execute('PRAGMA cache_size=-65536')  # 64 MB max
                    self._local.conn.execute('PRAGMA mmap_size=67108864')  # 64 MB max
                    self._local.conn.execute('PRAGMA wal_autocheckpoint=4000')
                    # Increased checkpoint interval for better concurrency
                    self._local.conn.execute('PRAGMA wal_autocheckpoint=500')
                    self._local.conn.execute('PRAGMA cache_size=-4000')  # 4MB cache
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt.should_retry:
                        logger.warning(f"Database locked, retry {attempt.number} after {attempt.delay:.2f}s")
                        attempt.wait()
                    else:
                        raise
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        # Pass _from_init=True to break circular dependency (December 2025 fix)
        conn = self._get_connection(_from_init=True)
        conn.executescript('''
            -- Events table
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT '',
                hostname TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                INDEX_event_type TEXT GENERATED ALWAYS AS (event_type) STORED
            );

            -- Subscribers table
            CREATE TABLE IF NOT EXISTS subscribers (
                subscriber_id TEXT PRIMARY KEY,
                process_name TEXT NOT NULL,
                hostname TEXT NOT NULL,
                pid INTEGER NOT NULL,
                subscribed_types TEXT NOT NULL DEFAULT '[]',
                last_poll_at REAL NOT NULL,
                created_at REAL NOT NULL
            );

            -- Acknowledgments table (tracks which events each subscriber has processed)
            CREATE TABLE IF NOT EXISTS acks (
                subscriber_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                acked_at REAL NOT NULL,
                PRIMARY KEY (subscriber_id, event_id),
                FOREIGN KEY (subscriber_id) REFERENCES subscribers(subscriber_id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
            CREATE INDEX IF NOT EXISTS idx_acks_subscriber ON acks(subscriber_id);
            CREATE INDEX IF NOT EXISTS idx_acks_event ON acks(event_id);
        ''')
        conn.commit()

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        source: str = "",
        max_retries: int = 5,
    ) -> int:
        """Publish an event to the queue.

        Args:
            event_type: Type of event (e.g., "MODEL_PROMOTED")
            payload: Event data
            source: Component that generated the event
            max_retries: Maximum number of retry attempts on database lock

        Returns:
            The event_id of the published event, or -1 if database is readonly

        January 3, 2026: Migrated to RetryConfig for centralized retry behavior.
        """
        # Skip if readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            logger.debug(f"[CrossProcessEventQueue] Cannot publish {event_type} (readonly mode)")
            return -1

        retry_config = RetryConfig(
            max_attempts=max_retries,
            base_delay=0.1,
            max_delay=1.6,
        )
        last_error: Exception | None = None

        for attempt in retry_config.attempts():
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    '''INSERT INTO events (event_type, payload, source, hostname, created_at)
                       VALUES (?, ?, ?, ?, ?)''',
                    (
                        event_type,
                        json.dumps(payload or {}),
                        source,
                        socket.gethostname(),
                        time.time(),
                    )
                )
                event_id = cursor.lastrowid
                conn.commit()
                if event_id is None:
                    raise RuntimeError("Database INSERT failed to return lastrowid")
                return event_id
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" in str(e).lower() and attempt.should_retry:
                    attempt.wait()
                    continue
                raise
        if last_error:
            raise last_error
        return -1  # Should not reach here

    def subscribe(
        self,
        process_name: str,
        event_types: list[str] | None = None,
        stable: bool = False,
    ) -> str:
        """Register a subscriber for polling events.

        Args:
            process_name: Name of the subscribing process
            event_types: Optional list of event types to filter (None = all)
            stable: If True, use a stable subscriber ID without PID.
                    This allows a process to resume from where it left off
                    after restart. Use for singleton processes like event_router.

        Returns:
            Subscriber ID for use in poll() and ack()
        """
        if stable:
            # Stable ID: persists across restarts, inherits acks from previous runs
            subscriber_id = f"{socket.gethostname()}:stable:{process_name}"
        else:
            # Per-instance ID: each process instance gets its own ack tracking
            subscriber_id = f"{socket.gethostname()}:{os.getpid()}:{process_name}"

        # Skip DB write if readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            logger.debug(f"[CrossProcessEventQueue] Readonly mode, subscriber {subscriber_id} not persisted")
            return subscriber_id

        conn = self._get_connection()

        conn.execute(
            '''INSERT OR REPLACE INTO subscribers
               (subscriber_id, process_name, hostname, pid, subscribed_types, last_poll_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                subscriber_id,
                process_name,
                socket.gethostname(),
                os.getpid(),
                json.dumps(event_types or []),
                time.time(),
                time.time(),
            )
        )
        conn.commit()
        return subscriber_id

    def poll(
        self,
        subscriber_id: str,
        event_types: list[str] | None = None,
        limit: int = 100,
        since_event_id: int | None = None,
    ) -> list[CrossProcessEvent]:
        """Poll for new events.

        Returns events that:
        1. Match the specified event types (or all if None)
        2. Haven't been acknowledged by this subscriber
        3. Were created within the retention period

        Args:
            subscriber_id: Subscriber ID from subscribe()
            event_types: Filter by event types (None = all)
            limit: Maximum events to return
            since_event_id: Only return events after this ID

        Returns:
            List of unacknowledged events
        """
        conn = self._get_connection()

        # Update subscriber heartbeat
        conn.execute(
            'UPDATE subscribers SET last_poll_at = ? WHERE subscriber_id = ?',
            (time.time(), subscriber_id)
        )

        # Build query
        cutoff_time = time.time() - (self.retention_hours * 3600)

        query = '''
            SELECT e.event_id, e.event_type, e.payload, e.source, e.hostname, e.created_at
            FROM events e
            LEFT JOIN acks a ON e.event_id = a.event_id AND a.subscriber_id = ?
            WHERE a.event_id IS NULL
              AND e.created_at > ?
        '''
        params: list[Any] = [subscriber_id, cutoff_time]

        if since_event_id is not None:
            query += ' AND e.event_id > ?'
            params.append(since_event_id)

        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f' AND e.event_type IN ({placeholders})'
            params.extend(event_types)

        query += ' ORDER BY e.event_id ASC LIMIT ?'
        params.append(limit)

        cursor = conn.execute(query, params)
        events = [CrossProcessEvent.from_row(row) for row in cursor.fetchall()]
        conn.commit()

        return events

    def ack(self, subscriber_id: str, event_id: int) -> bool:
        """Acknowledge that an event has been processed.

        Args:
            subscriber_id: Subscriber ID
            event_id: Event ID to acknowledge

        Returns:
            True if ack was recorded, False if readonly or error
        """
        # Skip if readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return False
        conn = self._get_connection()
        try:
            conn.execute(
                'INSERT OR IGNORE INTO acks (subscriber_id, event_id, acked_at) VALUES (?, ?, ?)',
                (subscriber_id, event_id, time.time())
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error acking event {event_id}: {e}")
            return False

    def ack_batch(self, subscriber_id: str, event_ids: list[int]) -> int:
        """Acknowledge multiple events at once.

        Returns:
            Number of events acknowledged, 0 if readonly
        """
        if not event_ids:
            return 0
        # Skip if readonly mode (December 2025: Lazy init)
        if self._readonly_mode:
            return 0

        conn = self._get_connection()
        now = time.time()
        conn.executemany(
            'INSERT OR IGNORE INTO acks (subscriber_id, event_id, acked_at) VALUES (?, ?, ?)',
            [(subscriber_id, eid, now) for eid in event_ids]
        )
        conn.commit()
        return len(event_ids)

    def get_pending_count(self, subscriber_id: str, event_types: list[str] | None = None) -> int:
        """Get count of pending (unacknowledged) events for a subscriber."""
        conn = self._get_connection()
        cutoff_time = time.time() - (self.retention_hours * 3600)

        query = '''
            SELECT COUNT(*) FROM events e
            LEFT JOIN acks a ON e.event_id = a.event_id AND a.subscriber_id = ?
            WHERE a.event_id IS NULL AND e.created_at > ?
        '''
        params: list[Any] = [subscriber_id, cutoff_time]

        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f' AND e.event_type IN ({placeholders})'
            params.extend(event_types)

        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]

    def get_last_acked_event_id(self, subscriber_id: str) -> int:
        """Get the highest event_id that was acked by this subscriber.

        This allows resuming from where we left off after a restart.

        Returns:
            The last acked event_id, or 0 if no events have been acked.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            'SELECT MAX(event_id) FROM acks WHERE subscriber_id = ?',
            (subscriber_id,)
        )
        result = cursor.fetchone()[0]
        return result if result is not None else 0

    def cleanup(self) -> tuple[int, int, int]:
        """Clean up old events and dead subscribers.

        Returns:
            Tuple of (events_deleted, subscribers_deleted, acks_deleted)
        """
        conn = self._get_connection()

        # Delete old events
        cutoff_time = time.time() - (self.retention_hours * 3600)
        cursor = conn.execute('DELETE FROM events WHERE created_at < ?', (cutoff_time,))
        events_deleted = cursor.rowcount

        # Delete dead subscribers
        subscriber_cutoff = time.time() - SUBSCRIBER_TIMEOUT_SECONDS
        cursor = conn.execute('DELETE FROM subscribers WHERE last_poll_at < ?', (subscriber_cutoff,))
        subscribers_deleted = cursor.rowcount

        # Delete orphaned acks (events that no longer exist)
        cursor = conn.execute('''
            DELETE FROM acks WHERE event_id NOT IN (SELECT event_id FROM events)
        ''')
        acks_deleted = cursor.rowcount

        conn.commit()
        return events_deleted, subscribers_deleted, acks_deleted

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        conn = self._get_connection()

        # Total events
        cursor = conn.execute('SELECT COUNT(*) FROM events')
        total_events = cursor.fetchone()[0]

        # Events by type
        cursor = conn.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type')
        events_by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Active subscribers
        subscriber_cutoff = time.time() - SUBSCRIBER_TIMEOUT_SECONDS
        cursor = conn.execute(
            'SELECT subscriber_id, process_name, hostname, last_poll_at FROM subscribers WHERE last_poll_at > ?',
            (subscriber_cutoff,)
        )
        active_subscribers = [
            {
                "subscriber_id": row[0],
                "process_name": row[1],
                "hostname": row[2],
                "last_poll_ago": time.time() - row[3],
            }
            for row in cursor.fetchall()
        ]

        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "active_subscribers": active_subscribers,
            "retention_hours": self.retention_hours,
        }

    def health_check(self) -> "HealthCheckResult":
        """Check health of the cross-process event queue.

        December 2025: Added for DaemonManager integration.

        Returns:
            HealthCheckResult with queue health status
        """
        try:
            # Try to get connection and run a simple query
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            total_events = cursor.fetchone()[0]

            # Get active subscriber count
            subscriber_cutoff = time.time() - SUBSCRIBER_TIMEOUT_SECONDS
            cursor = conn.execute(
                "SELECT COUNT(*) FROM subscribers WHERE last_poll_at > ?",
                (subscriber_cutoff,)
            )
            active_subscribers = cursor.fetchone()[0]

            return HealthCheckResult(
                healthy=True,
                message="Cross-process event queue operational",
                details={
                    "total_events": total_events,
                    "active_subscribers": active_subscribers,
                    "db_path": str(self.db_path),
                    "retention_hours": self.retention_hours,
                },
            )
        except sqlite3.Error as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Database error: {e}",
                details={"db_path": str(self.db_path)},
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error_type": type(e).__name__},
            )

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global singleton instance
_event_queue: CrossProcessEventQueue | None = None
_queue_lock = threading.RLock()


def get_event_queue(db_path: Path | None = None) -> CrossProcessEventQueue:
    """Get the global cross-process event queue singleton."""
    global _event_queue
    with _queue_lock:
        if _event_queue is None:
            _event_queue = CrossProcessEventQueue(db_path)
        return _event_queue


def reset_event_queue() -> None:
    """Reset the global event queue (for testing)."""
    global _event_queue
    with _queue_lock:
        if _event_queue is not None:
            _event_queue.close()
        _event_queue = None


# Convenience functions for common operations


def publish_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "",
    db_path: Path | None = None,
) -> int:
    """Publish an event to the cross-process queue.

    This is the primary way for components to broadcast events
    that other processes should know about.
    """
    return get_event_queue(db_path).publish(event_type, payload, source)


def subscribe_process(
    process_name: str,
    event_types: list[str] | None = None,
    db_path: Path | None = None,
) -> str:
    """Register this process as a subscriber."""
    return get_event_queue(db_path).subscribe(process_name, event_types)


def poll_events(
    subscriber_id: str,
    event_types: list[str] | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[CrossProcessEvent]:
    """Poll for new events as a subscriber."""
    return get_event_queue(db_path).poll(subscriber_id, event_types, limit)


def ack_event(subscriber_id: str, event_id: int, db_path: Path | None = None) -> bool:
    """Acknowledge that an event has been processed."""
    return get_event_queue(db_path).ack(subscriber_id, event_id)


def ack_events(subscriber_id: str, event_ids: list[int], db_path: Path | None = None) -> int:
    """Acknowledge multiple events."""
    return get_event_queue(db_path).ack_batch(subscriber_id, event_ids)


# Bridge functions to integrate with existing DataEventType


def bridge_to_cross_process(event_type_value: str, payload: dict[str, Any], source: str = "") -> int:
    """Bridge an in-memory DataEvent to the cross-process queue.

    Call this after publishing to the in-memory EventBus to also
    propagate the event across processes.

    Example:
        # In your event emission code:
        await get_event_bus().publish(event)
        bridge_to_cross_process(event.event_type.value, event.payload, event.source)
    """
    return publish_event(event_type_value, payload, source)


class CrossProcessEventPoller:
    """Background poller for consuming cross-process events.

    Runs in a separate thread and calls registered handlers
    when events arrive.

    Usage:
        poller = CrossProcessEventPoller("my_daemon", ["MODEL_PROMOTED", "TRAINING_COMPLETED"])
        poller.register_handler("MODEL_PROMOTED", handle_model_promoted)
        poller.start()
        # ... later ...
        poller.stop()
    """

    def __init__(
        self,
        process_name: str,
        event_types: list[str] | None = None,
        poll_interval: float = 1.0,
        db_path: Path | None = None,
        stable: bool = False,
    ):
        """Initialize the poller.

        Args:
            process_name: Name of the subscribing process
            event_types: Event types to filter (None = all)
            poll_interval: Seconds between polls
            db_path: Optional custom database path
            stable: If True, use stable subscriber ID that persists across restarts.
                    This allows resuming from the last acked event after restart.
        """
        self.process_name = process_name
        self.event_types = event_types
        self.poll_interval = poll_interval
        self.db_path = db_path
        self.stable = stable
        self._handlers: dict[str, list[Callable[[CrossProcessEvent], None]]] = {}
        self._global_handlers: list[Callable[[CrossProcessEvent], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._subscriber_id: str | None = None

    def register_handler(
        self,
        event_type: str | None,
        handler: Callable[[CrossProcessEvent], None],
    ) -> None:
        """Register a handler for events.

        Args:
            event_type: Event type to handle, or None for all events
            handler: Function to call when event arrives
        """
        if event_type is None:
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def start(self) -> None:
        """Start the polling thread."""
        if self._running:
            return

        self._running = True
        queue = get_event_queue(self.db_path)
        self._subscriber_id = queue.subscribe(
            self.process_name, self.event_types, stable=self.stable
        )

        # Get the last acked event ID to resume from (for stable subscribers)
        self._last_acked_event_id = queue.get_last_acked_event_id(self._subscriber_id)
        if self._last_acked_event_id > 0:
            logger.info(
                f"[CrossProcessEventPoller] Resuming from event_id {self._last_acked_event_id}"
            )

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def health_check(self) -> "HealthCheckResult":
        """Check poller health for daemon monitoring.

        Returns:
            HealthCheckResult with polling status and handler metrics.
        """
        # If not running, report as stopped (healthy but inactive)
        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="CrossProcessEventPoller is stopped",
            )

        # Check if thread is alive
        if self._thread and not self._thread.is_alive():
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="Polling thread is not alive (crashed or stopped unexpectedly)",
            )

        # Check if subscriber is registered
        if not self._subscriber_id:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message="No subscriber ID assigned",
            )

        # All checks passed - running healthy
        handler_count = len(self._handlers)
        global_handler_count = len(self._global_handlers)

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Poller running, {handler_count} handler types, {global_handler_count} global handlers",
            details={
                "handler_types": handler_count,
                "global_handlers": global_handler_count,
                "subscriber_id": self._subscriber_id,
                "poll_interval": self.poll_interval,
            },
        )

    def _poll_loop(self) -> None:
        """Internal polling loop."""
        queue = get_event_queue(self.db_path)
        # Resume from last acked event (set in start() method)
        last_event_id = getattr(self, '_last_acked_event_id', 0)

        while self._running:
            try:
                events = queue.poll(
                    self._subscriber_id,
                    self.event_types,
                    limit=50,
                    since_event_id=last_event_id if last_event_id > 0 else None,
                )

                for event in events:
                    try:
                        # Call type-specific handlers
                        if event.event_type in self._handlers:
                            for handler in self._handlers[event.event_type]:
                                handler(event)

                        # Call global handlers
                        for handler in self._global_handlers:
                            handler(event)

                        # Ack the event
                        queue.ack(self._subscriber_id, event.event_id)
                        last_event_id = max(last_event_id, event.event_id)

                    except Exception as e:
                        # December 2025 hardening: Use proper logging instead of print
                        logger.error(
                            f"[CrossProcessPoller] Error handling event {event.event_id}: {e}",
                            exc_info=True,
                        )

            except Exception as e:
                # December 2025 hardening: Use proper logging instead of print
                logger.error(
                    f"[CrossProcessPoller] Poll error: {e}",
                    exc_info=True,
                )

            time.sleep(self.poll_interval)


# Command-line interface for debugging and management

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-process event queue management")
    parser.add_argument("--stats", action="store_true", help="Show queue statistics")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old events")
    parser.add_argument("--publish", type=str, help="Publish a test event (type)")
    parser.add_argument("--payload", type=str, default="{}", help="Event payload (JSON)")
    parser.add_argument("--subscribe", type=str, help="Subscribe as process and show events")
    parser.add_argument("--db", type=str, help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    queue = get_event_queue(db_path)

    if args.stats:
        stats = queue.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.cleanup:
        events, subs, acks = queue.cleanup()
        print(f"Cleaned up: {events} events, {subs} subscribers, {acks} acks")

    elif args.publish:
        payload = json.loads(args.payload)
        event_id = queue.publish(args.publish, payload, source="cli")
        print(f"Published event {event_id}: {args.publish}")

    elif args.subscribe:
        subscriber_id = queue.subscribe(args.subscribe)
        print(f"Subscribed as: {subscriber_id}")
        print("Polling for events (Ctrl+C to stop)...")
        try:
            while True:
                events = queue.poll(subscriber_id, limit=10)
                for event in events:
                    print(f"  [{event.event_id}] {event.event_type}: {event.payload}")
                    queue.ack(subscriber_id, event.event_id)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    # Classes
    "CrossProcessEventQueue",
    "ack_event",
    "ack_events",
    "bridge_to_cross_process",
    # Functions
    "get_event_queue",
    "poll_events",
    "publish_event",
    "reset_event_queue",
    "subscribe_process",
]
