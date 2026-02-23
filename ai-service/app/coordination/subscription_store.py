"""Subscription Store - SQLite-backed persistence for event subscriptions.

P0 December 2025: Event subscriptions are lost on process restart, causing
events to be orphaned. This module provides persistent storage for subscription
state, enabling:

1. Subscription recovery on restart - handlers re-register from persistent state
2. DLQ replay - events in DLQ older than 5 minutes get replayed to restored handlers
3. Staleness alerts - DLQ events older than 24h without processing trigger alerts
4. Cross-process visibility - track which processes handle which events

Usage:
    from app.coordination.subscription_store import (
        SubscriptionStore,
        get_subscription_store,
        SubscriptionRecord,
    )

    # Register subscription (called when subscribe() is invoked)
    store = get_subscription_store()
    store.register_subscription(
        subscriber_name="DataPipelineOrchestrator",
        event_type="DATA_SYNC_COMPLETED",
        handler_path="app.coordination.data_pipeline_orchestrator:_on_sync_complete",
        pid=os.getpid(),
    )

    # On startup, restore subscriptions
    subscriptions = store.get_active_subscriptions()
    for sub in subscriptions:
        router.subscribe(sub.event_type, load_handler(sub.handler_path))

    # Replay DLQ events older than 5 minutes
    await store.replay_stale_dlq_events(min_age_seconds=300)

Created: December 2025
Purpose: Fix P0 critical blocker - subscription persistence (Phase 1.3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_SUBSCRIPTION_STORE_PATH = Path("data/coordination/subscription_store.db")

# Alert threshold for stale DLQ events (24 hours)
STALE_DLQ_ALERT_THRESHOLD_HOURS = 24

# Minimum age for DLQ replay (5 minutes)
DLQ_REPLAY_MIN_AGE_SECONDS = 300


@dataclass
class SubscriptionRecord:
    """A persistent subscription record.

    Attributes:
        subscriber_name: Name of the subscribing component (e.g., "DataPipelineOrchestrator")
        event_type: Normalized event type string (e.g., "DATA_SYNC_COMPLETED")
        handler_path: Full path to handler function (e.g., "app.coordination.foo:my_handler")
        pid: Process ID that registered this subscription
        created_at: ISO timestamp when subscription was created
        last_processed_at: ISO timestamp of last event processed by this subscriber
        is_active: Whether subscription is currently active
    """

    subscriber_name: str
    event_type: str
    handler_path: str
    pid: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_processed_at: str | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "subscriber_name": self.subscriber_name,
            "event_type": self.event_type,
            "handler_path": self.handler_path,
            "pid": self.pid,
            "created_at": self.created_at,
            "last_processed_at": self.last_processed_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SubscriptionRecord":
        """Create from SQLite row."""
        return cls(
            subscriber_name=row["subscriber_name"],
            event_type=row["event_type"],
            handler_path=row["handler_path"],
            pid=row["pid"],
            created_at=row["created_at"],
            last_processed_at=row["last_processed_at"],
            is_active=bool(row["is_active"]),
        )


class SubscriptionStore:
    """SQLite-backed persistence for event subscriptions.

    Provides:
    - Subscription registration and deregistration
    - Cross-process subscription tracking
    - DLQ integration for replay on restart
    - Staleness alerts for unprocessed events
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_SUBSCRIPTION_STORE_PATH,
        stale_dlq_alert_hours: float = STALE_DLQ_ALERT_THRESHOLD_HOURS,
        dlq_replay_min_age: float = DLQ_REPLAY_MIN_AGE_SECONDS,
    ):
        """Initialize subscription store.

        Args:
            db_path: Path to SQLite database for persistence
            stale_dlq_alert_hours: Hours before DLQ events trigger staleness alerts
            dlq_replay_min_age: Minimum age (seconds) for DLQ events to be replayed
        """
        self.db_path = Path(db_path)
        self.stale_dlq_alert_hours = stale_dlq_alert_hours
        self.dlq_replay_min_age = dlq_replay_min_age

        # Statistics
        self._subscriptions_registered = 0
        self._subscriptions_restored = 0
        self._dlq_events_replayed = 0
        self._stale_alerts_emitted = 0

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Subscriptions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subscriber_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    handler_path TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_processed_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    UNIQUE(subscriber_name, event_type)
                )
            """)

            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscriptions_event_type
                ON subscriptions(event_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscriptions_active
                ON subscriptions(is_active)
            """)

            # Last processed events per subscriber (for replay tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriber_state (
                    subscriber_name TEXT PRIMARY KEY,
                    last_processed_event_id TEXT,
                    last_processed_at TEXT,
                    events_processed INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    def register_subscription(
        self,
        subscriber_name: str,
        event_type: str,
        handler_path: str,
        pid: int | None = None,
    ) -> SubscriptionRecord:
        """Register a subscription for persistence.

        Args:
            subscriber_name: Name of subscribing component
            event_type: Event type to subscribe to
            handler_path: Module path to handler function
            pid: Process ID (defaults to current process)

        Returns:
            SubscriptionRecord that was created/updated
        """
        if pid is None:
            pid = os.getpid()

        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Upsert subscription
            conn.execute(
                """
                INSERT INTO subscriptions (
                    subscriber_name, event_type, handler_path, pid, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(subscriber_name, event_type) DO UPDATE SET
                    handler_path = excluded.handler_path,
                    pid = excluded.pid,
                    is_active = 1
                """,
                (subscriber_name, event_type, handler_path, pid, now),
            )
            conn.commit()

            # Fetch the record
            cursor = conn.execute(
                """
                SELECT * FROM subscriptions
                WHERE subscriber_name = ? AND event_type = ?
                """,
                (subscriber_name, event_type),
            )
            row = cursor.fetchone()

        self._subscriptions_registered += 1
        record = SubscriptionRecord.from_row(row)

        logger.debug(
            f"[SubscriptionStore] Registered: {subscriber_name} -> {event_type} "
            f"(handler={handler_path}, pid={pid})"
        )

        return record

    def deactivate_subscription(
        self,
        subscriber_name: str,
        event_type: str,
    ) -> bool:
        """Deactivate a subscription (soft delete).

        Args:
            subscriber_name: Name of subscribing component
            event_type: Event type to unsubscribe from

        Returns:
            True if subscription was found and deactivated
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE subscriptions
                SET is_active = 0
                WHERE subscriber_name = ? AND event_type = ?
                """,
                (subscriber_name, event_type),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(
                    f"[SubscriptionStore] Deactivated: {subscriber_name} -> {event_type}"
                )
                return True
            return False

    def deactivate_all_for_pid(self, pid: int) -> int:
        """Deactivate all subscriptions for a specific process.

        Called when a process exits cleanly to mark its subscriptions inactive.

        Args:
            pid: Process ID to deactivate subscriptions for

        Returns:
            Number of subscriptions deactivated
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE subscriptions
                SET is_active = 0
                WHERE pid = ? AND is_active = 1
                """,
                (pid,),
            )
            conn.commit()

            count = cursor.rowcount
            if count > 0:
                logger.info(
                    f"[SubscriptionStore] Deactivated {count} subscriptions for pid={pid}"
                )
            return count

    def get_active_subscriptions(
        self,
        event_type: str | None = None,
    ) -> list[SubscriptionRecord]:
        """Get all active subscriptions.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of active subscription records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if event_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM subscriptions
                    WHERE is_active = 1 AND event_type = ?
                    ORDER BY created_at
                    """,
                    (event_type,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM subscriptions
                    WHERE is_active = 1
                    ORDER BY event_type, created_at
                    """
                )

            return [SubscriptionRecord.from_row(row) for row in cursor.fetchall()]

    def get_subscriptions_for_subscriber(
        self,
        subscriber_name: str,
    ) -> list[SubscriptionRecord]:
        """Get all subscriptions for a specific subscriber.

        Args:
            subscriber_name: Name of subscribing component

        Returns:
            List of subscription records for this subscriber
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT * FROM subscriptions
                WHERE subscriber_name = ?
                ORDER BY event_type
                """,
                (subscriber_name,),
            )

            return [SubscriptionRecord.from_row(row) for row in cursor.fetchall()]

    def update_last_processed(
        self,
        subscriber_name: str,
        event_type: str,
        event_id: str,
    ) -> None:
        """Update last processed timestamp for a subscription.

        Called after a handler successfully processes an event.

        Args:
            subscriber_name: Name of subscribing component
            event_type: Event type that was processed
            event_id: ID of the event that was processed
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Update subscription record
            conn.execute(
                """
                UPDATE subscriptions
                SET last_processed_at = ?
                WHERE subscriber_name = ? AND event_type = ?
                """,
                (now, subscriber_name, event_type),
            )

            # Update subscriber state
            conn.execute(
                """
                INSERT INTO subscriber_state (
                    subscriber_name, last_processed_event_id, last_processed_at, events_processed
                ) VALUES (?, ?, ?, 1)
                ON CONFLICT(subscriber_name) DO UPDATE SET
                    last_processed_event_id = excluded.last_processed_event_id,
                    last_processed_at = excluded.last_processed_at,
                    events_processed = events_processed + 1
                """,
                (subscriber_name, event_id, now),
            )

            conn.commit()

    def get_subscriber_state(self, subscriber_name: str) -> dict[str, Any] | None:
        """Get the processing state for a subscriber.

        Args:
            subscriber_name: Name of subscribing component

        Returns:
            Dict with last_processed_event_id, last_processed_at, events_processed
            or None if subscriber not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT * FROM subscriber_state
                WHERE subscriber_name = ?
                """,
                (subscriber_name,),
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    async def replay_stale_dlq_events(
        self,
        min_age_seconds: float | None = None,
        max_events: int = 100,
    ) -> int:
        """Replay DLQ events that haven't been processed.

        Called on startup to ensure events aren't lost due to process restarts.

        Args:
            min_age_seconds: Minimum age for events to replay (default: dlq_replay_min_age)
            max_events: Maximum events to replay per call

        Returns:
            Number of events replayed

        Environment:
            RINGRIFT_DISABLE_DLQ_REPLAY: Set to "1" to disable DLQ replay (for tests)
        """
        # Check for test mode to avoid blocking on sync operations
        import os
        if os.environ.get("RINGRIFT_DISABLE_DLQ_REPLAY", "").lower() in ("1", "true", "yes"):
            return 0

        if min_age_seconds is None:
            min_age_seconds = self.dlq_replay_min_age

        try:
            from app.coordination.dead_letter_queue import get_dead_letter_queue
            from app.coordination.event_router import get_router
        except ImportError:
            logger.warning(
                "[SubscriptionStore] Cannot replay DLQ - missing dependencies"
            )
            return 0

        dlq = get_dead_letter_queue()
        router = get_router()

        # Get events older than min_age but not older than max_age
        cutoff = datetime.now() - timedelta(seconds=min_age_seconds)
        cutoff_str = cutoff.isoformat()
        # Feb 2026: Cap max age to 1 hour. Stale events from days/weeks ago
        # were being replayed indefinitely, emitting phantom TRAINING_COMPLETED
        # events with lost metrics (policy_accuracy=0.0).
        max_age_seconds = 3600
        max_age_cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        max_age_cutoff_str = max_age_cutoff.isoformat()

        # Wrap blocking SQLite call with asyncio.to_thread to avoid event loop blocking
        # December 30, 2025: Fixed async context blocking
        def _fetch_stale_events() -> list[sqlite3.Row]:
            with sqlite3.connect(dlq.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM dead_letter
                    WHERE created_at < ?
                    AND created_at > ?
                    AND retry_count < 5
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (cutoff_str, max_age_cutoff_str, max_events),
                )
                return cursor.fetchall()

        events = await asyncio.to_thread(_fetch_stale_events)

        replayed = 0
        for event_row in events:
            event_type = event_row["event_type"]
            payload = json.loads(event_row["payload"])
            event_id = event_row["event_id"]

            # Check if there are active subscribers
            if not router.has_subscribers(event_type):
                logger.debug(
                    f"[SubscriptionStore] Skipping DLQ replay for {event_type} - no subscribers"
                )
                continue

            try:
                # Republish to router
                await router.publish(
                    event_type=event_type,
                    payload=payload,
                    source=f"dlq_replay:{event_id}",
                )
                replayed += 1

                # Mark as replayed in DLQ (increment retry count)
                # Wrap blocking SQLite update with asyncio.to_thread
                def _mark_event_replayed(eid: str) -> None:
                    with sqlite3.connect(dlq.db_path) as conn:
                        conn.execute(
                            """
                            UPDATE dead_letter
                            SET retry_count = retry_count + 1,
                                last_retry_at = ?
                            WHERE event_id = ?
                            """,
                            (datetime.now().isoformat(), eid),
                        )
                        conn.commit()

                await asyncio.to_thread(_mark_event_replayed, event_id)

            except Exception as e:
                logger.warning(
                    f"[SubscriptionStore] Failed to replay DLQ event {event_id}: {e}"
                )

        if replayed > 0:
            logger.info(
                f"[SubscriptionStore] Replayed {replayed} stale DLQ events "
                f"(>={min_age_seconds}s old)"
            )

        self._dlq_events_replayed += replayed
        return replayed

    def check_stale_dlq_events(self) -> list[dict[str, Any]]:
        """Check for DLQ events that are older than alert threshold.

        Returns:
            List of stale events that should trigger alerts
        """
        try:
            from app.coordination.dead_letter_queue import get_dead_letter_queue
        except ImportError:
            return []

        dlq = get_dead_letter_queue()

        cutoff = datetime.now() - timedelta(hours=self.stale_dlq_alert_hours)
        cutoff_str = cutoff.isoformat()

        with sqlite3.connect(dlq.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT event_type, COUNT(*) as count, MIN(created_at) as oldest
                FROM dead_letter
                WHERE created_at < ?
                GROUP BY event_type
                """,
                (cutoff_str,),
            )

            stale_events = []
            for row in cursor.fetchall():
                stale_events.append({
                    "event_type": row["event_type"],
                    "count": row["count"],
                    "oldest": row["oldest"],
                    "threshold_hours": self.stale_dlq_alert_hours,
                })

        return stale_events

    async def emit_stale_dlq_alerts(self) -> int:
        """Emit alerts for stale DLQ events.

        Returns:
            Number of alerts emitted
        """
        # Wrap blocking sync method with asyncio.to_thread
        # December 30, 2025: Fixed async context blocking
        stale_events = await asyncio.to_thread(self.check_stale_dlq_events)

        if not stale_events:
            return 0

        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType
        except ImportError:
            logger.warning(
                "[SubscriptionStore] Cannot emit alerts - missing dependencies"
            )
            return 0

        router = get_router()
        alerts_emitted = 0

        for event_info in stale_events:
            logger.warning(
                f"[SubscriptionStore] STALE DLQ ALERT: {event_info['count']} "
                f"'{event_info['event_type']}' events older than "
                f"{event_info['threshold_hours']}h (oldest: {event_info['oldest']})"
            )

            # Emit DLQ alert event
            try:
                await router.publish(
                    event_type="DLQ_STALE_EVENTS",
                    payload={
                        "event_type": event_info["event_type"],
                        "count": event_info["count"],
                        "oldest": event_info["oldest"],
                        "threshold_hours": event_info["threshold_hours"],
                    },
                    source="subscription_store",
                )
                alerts_emitted += 1
            except Exception as e:
                logger.error(f"[SubscriptionStore] Failed to emit stale DLQ alert: {e}")

        self._stale_alerts_emitted += alerts_emitted
        return alerts_emitted

    def get_stats(self) -> dict[str, Any]:
        """Get subscription store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Count active subscriptions
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM subscriptions WHERE is_active = 1"
            )
            active_count = cursor.fetchone()["count"]

            # Count by event type
            cursor = conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM subscriptions
                WHERE is_active = 1
                GROUP BY event_type
                """
            )
            by_event_type = {row["event_type"]: row["count"] for row in cursor.fetchall()}

            # Count unique subscribers
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT subscriber_name) as count
                FROM subscriptions
                WHERE is_active = 1
                """
            )
            unique_subscribers = cursor.fetchone()["count"]

        return {
            "active_subscriptions": active_count,
            "by_event_type": by_event_type,
            "unique_subscribers": unique_subscribers,
            "subscriptions_registered": self._subscriptions_registered,
            "subscriptions_restored": self._subscriptions_restored,
            "dlq_events_replayed": self._dlq_events_replayed,
            "stale_alerts_emitted": self._stale_alerts_emitted,
        }

    def health_check(self) -> "HealthCheckResult":
        """Return health status for daemon integration."""
        try:
            from app.coordination.contracts import HealthCheckResult
        except ImportError:
            # Fallback for when contracts not available
            return {"healthy": True, "message": "OK", "details": self.get_stats()}

        try:
            stats = self.get_stats()
            stale_events = self.check_stale_dlq_events()

            # Unhealthy if many stale DLQ events
            total_stale = sum(e["count"] for e in stale_events)
            if total_stale > 1000:
                return HealthCheckResult(
                    healthy=False,
                    message=f"Critical: {total_stale} stale DLQ events",
                    details={**stats, "stale_dlq_events": stale_events},
                )

            if total_stale > 100:
                return HealthCheckResult(
                    healthy=True,
                    message=f"Warning: {total_stale} stale DLQ events",
                    details={**stats, "stale_dlq_events": stale_events},
                )

            return HealthCheckResult(
                healthy=True,
                message=f"OK - {stats['active_subscriptions']} active subscriptions",
                details=stats,
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )


# Singleton instance
_subscription_store: SubscriptionStore | None = None


def get_subscription_store(
    db_path: Path | str | None = None,
) -> SubscriptionStore:
    """Get or create the singleton subscription store.

    Args:
        db_path: Optional custom database path

    Returns:
        SubscriptionStore singleton instance
    """
    global _subscription_store

    if _subscription_store is None:
        if db_path is None:
            db_path = DEFAULT_SUBSCRIPTION_STORE_PATH
        _subscription_store = SubscriptionStore(db_path=db_path)
        logger.info(
            f"[SubscriptionStore] Initialized subscription store at {db_path}"
        )

    return _subscription_store


def reset_subscription_store() -> None:
    """Reset the singleton subscription store (for testing)."""
    global _subscription_store
    _subscription_store = None


# Module-level exports
__all__ = [
    "SubscriptionStore",
    "SubscriptionRecord",
    "get_subscription_store",
    "reset_subscription_store",
    "DEFAULT_SUBSCRIPTION_STORE_PATH",
    "STALE_DLQ_ALERT_THRESHOLD_HOURS",
    "DLQ_REPLAY_MIN_AGE_SECONDS",
]
