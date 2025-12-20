#!/usr/bin/env python3
"""Cross-Process Event Queue using SQLite.

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
from pathlib import Path
from typing import Any

# Default database location
DEFAULT_EVENT_DB = Path("/tmp/ringrift_coordination/events.db")

# Event retention period (default: 24 hours)
DEFAULT_RETENTION_HOURS = 24

# Subscriber heartbeat timeout (subscribers not polling for this long are considered dead)
SUBSCRIBER_TIMEOUT_SECONDS = 300  # 5 minutes


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

    def __init__(self, db_path: Path | None = None, retention_hours: int = DEFAULT_RETENTION_HOURS):
        self.db_path = db_path or DEFAULT_EVENT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_hours = retention_hours
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=60.0,  # Increased from 30s
                isolation_level=None,  # Autocommit for better concurrency
            )
            self._local.conn.row_factory = sqlite3.Row
            # WAL mode for concurrent access
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA busy_timeout=30000')  # Increased from 10s
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn.execute('PRAGMA wal_autocheckpoint=100')  # Checkpoint every 100 pages
            self._local.conn.execute('PRAGMA cache_size=-2000')  # 2MB cache
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
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
            The event_id of the published event
        """
        last_error = None
        for attempt in range(max_retries):
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
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                raise
        raise last_error

    def subscribe(
        self,
        process_name: str,
        event_types: list[str] | None = None,
    ) -> str:
        """Register a subscriber for polling events.

        Args:
            process_name: Name of the subscribing process
            event_types: Optional list of event types to filter (None = all)

        Returns:
            Subscriber ID for use in poll() and ack()
        """
        subscriber_id = f"{socket.gethostname()}:{os.getpid()}:{process_name}"
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
            True if ack was recorded
        """
        conn = self._get_connection()
        try:
            conn.execute(
                'INSERT OR IGNORE INTO acks (subscriber_id, event_id, acked_at) VALUES (?, ?, ?)',
                (subscriber_id, event_id, time.time())
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[CrossProcessEvents] Error acking event {event_id}: {e}")
            return False

    def ack_batch(self, subscriber_id: str, event_ids: list[int]) -> int:
        """Acknowledge multiple events at once.

        Returns:
            Number of events acknowledged
        """
        if not event_ids:
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
    ):
        self.process_name = process_name
        self.event_types = event_types
        self.poll_interval = poll_interval
        self.db_path = db_path
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
        self._subscriber_id = queue.subscribe(self.process_name, self.event_types)

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _poll_loop(self) -> None:
        """Internal polling loop."""
        queue = get_event_queue(self.db_path)
        last_event_id = 0

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
                        print(f"[CrossProcessPoller] Error handling event {event.event_id}: {e}")

            except Exception as e:
                print(f"[CrossProcessPoller] Poll error: {e}")

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
