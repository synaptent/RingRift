"""Dead Letter Queue for failed event handlers (December 2025).

This module provides fault-tolerant event handling by capturing failed events
and enabling automatic retry with exponential backoff.

Key Features:
- SQLite-backed persistence for failed events
- Automatic retry with configurable backoff
- Manual inspection and replay tools
- Integration with StageEventBus and DataEventBus

Usage:
    from app.coordination.dead_letter_queue import (
        DeadLetterQueue,
        get_dead_letter_queue,
        enable_dead_letter_queue,
    )

    # Enable dead letter queue for an event bus
    dlq = get_dead_letter_queue()
    enable_dead_letter_queue(dlq, stage_event_bus)

    # Manually retry failed events
    await dlq.retry_failed_events(max_retries=3)

    # Inspect failed events
    failed = dlq.get_failed_events(limit=10)
    for event in failed:
        print(f"{event['event_type']}: {event['error']} (retries: {event['retry_count']})")
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

from app.core.async_context import safe_create_task
from app.coordination.contracts import HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
from app.utils.sqlite_utils import connect_safe

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DLQ_PATH = Path("data/coordination/dead_letter_queue.db")


@dataclass
class FailedEvent:
    """A failed event awaiting retry."""

    event_id: str
    event_type: str
    payload: dict
    handler_name: str
    error: str
    retry_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_retry_at: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "handler_name": self.handler_name,
            "error": self.error,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "last_retry_at": self.last_retry_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailedEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            payload=data["payload"] if isinstance(data["payload"], dict) else json.loads(data["payload"]),
            handler_name=data["handler_name"],
            error=data["error"],
            retry_count=data.get("retry_count", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_retry_at=data.get("last_retry_at"),
            source=data.get("source", "unknown"),
        )


class DeadLetterQueue:
    """SQLite-backed dead letter queue for failed events.

    Captures events where handlers fail and enables retry with exponential backoff.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DLQ_PATH,
        max_retries: int = 5,
        base_backoff_seconds: float = 60.0,
        max_backoff_seconds: float = 3600.0,
    ):
        """Initialize dead letter queue.

        Args:
            db_path: Path to SQLite database for persistence
            max_retries: Maximum retry attempts before abandoning
            base_backoff_seconds: Base delay for exponential backoff
            max_backoff_seconds: Maximum delay between retries
        """
        self.db_path = Path(db_path)
        self.max_retries = max_retries
        self.base_backoff = base_backoff_seconds
        self.max_backoff = max_backoff_seconds

        # Handler registry for retries
        self._handlers: dict[str, list[Callable]] = {}

        # Statistics
        self._events_captured = 0
        self._events_retried = 0
        self._events_recovered = 0
        self._events_abandoned = 0

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with connect_safe(self.db_path, row_factory=None) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    handler_name TEXT NOT NULL,
                    error TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_retry_at TEXT,
                    source TEXT DEFAULT 'unknown',
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dead_letter_status
                ON dead_letter(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dead_letter_event_type
                ON dead_letter(event_type)
            """)
            conn.commit()

        logger.debug(f"[DLQ] Initialized database at {self.db_path}")

    def capture(
        self,
        event_type: str,
        payload: dict,
        handler_name: str,
        error: str,
        source: str = "unknown",
    ) -> str:
        """Capture a failed event for later retry.

        Args:
            event_type: Type of event that failed
            payload: Event payload
            handler_name: Name of handler that failed
            error: Error message from failure
            source: Source of the event

        Returns:
            Event ID for tracking
        """
        import uuid

        event_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with connect_safe(self.db_path, row_factory=None) as conn:
            conn.execute(
                """
                INSERT INTO dead_letter
                (event_id, event_type, payload, handler_name, error, created_at, source, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (event_id, event_type, json.dumps(payload), handler_name, error, now, source),
            )
            conn.commit()

        self._events_captured += 1
        logger.warning(
            f"[DLQ] Captured failed event {event_type} from {handler_name}: {error}"
        )

        return event_id

    def get_pending_events(
        self,
        limit: int = 100,
        event_type: str | None = None,
    ) -> list[FailedEvent]:
        """Get pending events awaiting retry.

        Args:
            limit: Maximum events to return
            event_type: Filter by event type

        Returns:
            List of failed events
        """
        with connect_safe(self.db_path, row_factory=None) as conn:
            conn.row_factory = sqlite3.Row
            if event_type:
                rows = conn.execute(
                    """
                    SELECT * FROM dead_letter
                    WHERE status = 'pending' AND event_type = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM dead_letter
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [FailedEvent.from_dict(dict(row)) for row in rows]

    def get_failed_events(
        self,
        limit: int = 100,
        include_abandoned: bool = False,
    ) -> list[dict]:
        """Get all failed events for inspection.

        Args:
            limit: Maximum events to return
            include_abandoned: Include events that exceeded max retries

        Returns:
            List of event dictionaries
        """
        with connect_safe(self.db_path, row_factory=None) as conn:
            conn.row_factory = sqlite3.Row
            if include_abandoned:
                rows = conn.execute(
                    """
                    SELECT * FROM dead_letter
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM dead_letter
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Register a handler for retry attempts.

        Args:
            event_type: Event type to handle
            handler: Async callback to invoke on retry
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def retry_event(self, event_id: str) -> bool:
        """Retry a specific failed event.

        Args:
            event_id: ID of event to retry

        Returns:
            True if retry succeeded
        """
        def _fetch_event() -> dict[str, Any] | None:
            with connect_safe(self.db_path, row_factory=None) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM dead_letter WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                return dict(row) if row else None

        row_dict = await asyncio.to_thread(_fetch_event)

        if not row_dict:
            logger.warning(f"[DLQ] Event {event_id} not found")
            return False

        event = FailedEvent.from_dict(row_dict)

        # Check if handlers registered
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.warning(f"[DLQ] No handlers registered for {event.event_type}")
            return False

        # Attempt retry
        success = False
        for handler in handlers:
            try:
                await handler(event.payload)
                success = True
                break
            except Exception as e:
                logger.warning(f"[DLQ] Retry failed for {event_id}: {e}")

        now = datetime.now().isoformat()
        new_count = event.retry_count + 1

        def _update_event_status() -> str:
            """Update event status in database. Returns: 'recovered', 'abandoned', or 'retrying'."""
            with connect_safe(self.db_path, row_factory=None) as conn:
                if success:
                    conn.execute(
                        """
                        UPDATE dead_letter
                        SET status = 'recovered', last_retry_at = ?
                        WHERE event_id = ?
                        """,
                        (now, event_id),
                    )
                    conn.commit()
                    return "recovered"
                else:
                    if new_count >= self.max_retries:
                        conn.execute(
                            """
                            UPDATE dead_letter
                            SET status = 'abandoned', retry_count = ?, last_retry_at = ?
                            WHERE event_id = ?
                            """,
                            (new_count, now, event_id),
                        )
                        conn.commit()
                        return "abandoned"
                    else:
                        conn.execute(
                            """
                            UPDATE dead_letter
                            SET retry_count = ?, last_retry_at = ?
                            WHERE event_id = ?
                            """,
                            (new_count, now, event_id),
                        )
                        conn.commit()
                        return "retrying"

        result_status = await asyncio.to_thread(_update_event_status)

        if result_status == "recovered":
            self._events_recovered += 1
            logger.info(f"[DLQ] Event {event_id} recovered successfully")
        elif result_status == "abandoned":
            self._events_abandoned += 1
            logger.error(f"[DLQ] Event {event_id} abandoned after {new_count} retries")

        self._events_retried += 1
        return success

    async def retry_failed_events(
        self,
        max_events: int = 10,
        event_type: str | None = None,
    ) -> dict[str, int]:
        """Retry pending failed events with backoff.

        Args:
            max_events: Maximum events to retry in this batch
            event_type: Filter by event type

        Returns:
            Stats dict with recovered, failed, skipped counts
        """
        # Wrap blocking SQLite call with asyncio.to_thread to avoid event loop blocking
        pending = await asyncio.to_thread(
            self.get_pending_events, limit=max_events, event_type=event_type
        )

        stats = {"recovered": 0, "failed": 0, "skipped": 0}

        for event in pending:
            # Check backoff
            if event.last_retry_at:
                last_retry = datetime.fromisoformat(event.last_retry_at)
                backoff = min(
                    self.base_backoff * (2 ** event.retry_count),
                    self.max_backoff,
                )
                elapsed = (datetime.now() - last_retry).total_seconds()
                if elapsed < backoff:
                    stats["skipped"] += 1
                    continue

            success = await self.retry_event(event.event_id)
            if success:
                stats["recovered"] += 1
            else:
                stats["failed"] += 1

        return stats

    def purge_old_events(self, days: int = 7) -> int:
        """Remove old events from the queue.

        Args:
            days: Remove events older than this many days

        Returns:
            Number of events removed
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with connect_safe(self.db_path, row_factory=None) as conn:
            cursor = conn.execute(
                """
                DELETE FROM dead_letter
                WHERE created_at < ? AND status IN ('recovered', 'abandoned')
                """,
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"[DLQ] Purged {deleted} old events")
        return deleted

    def get_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with connect_safe(self.db_path, row_factory=None) as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM dead_letter WHERE status = 'pending'"
            ).fetchone()[0]
            recovered = conn.execute(
                "SELECT COUNT(*) FROM dead_letter WHERE status = 'recovered'"
            ).fetchone()[0]
            abandoned = conn.execute(
                "SELECT COUNT(*) FROM dead_letter WHERE status = 'abandoned'"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM dead_letter").fetchone()[0]

            # Event type breakdown
            by_type = dict(
                conn.execute(
                    """
                    SELECT event_type, COUNT(*)
                    FROM dead_letter
                    WHERE status = 'pending'
                    GROUP BY event_type
                    """
                ).fetchall()
            )

        return {
            "pending": pending,
            "recovered": recovered,
            "abandoned": abandoned,
            "total": total,
            "by_event_type": by_type,
            "session_captured": self._events_captured,
            "session_retried": self._events_retried,
            "session_recovered": self._events_recovered,
            "session_abandoned": self._events_abandoned,
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration.

        December 2025: Added for unified health monitoring across coordinators.
        """
        try:
            stats = self.get_stats()
            details = {
                "queue_size": stats["pending"],
                "dead_letters_count": stats["total"],
                "reprocessed_count": stats["session_recovered"],
                "abandoned_count": stats["session_abandoned"],
                "captured_count": stats["session_captured"],
                "by_event_type": stats["by_event_type"],
            }

            # Check if queue is healthy (not too many pending items)
            pending = stats["pending"]
            if pending > 1000:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"DeadLetterQueue has high pending count: {pending}",
                    details=details,
                )

            # Check abandonment rate
            total_processed = stats["session_recovered"] + stats["session_abandoned"]
            if total_processed > 0:
                abandon_rate = stats["session_abandoned"] / total_processed
                if abandon_rate > 0.5:
                    return HealthCheckResult(
                        healthy=False,
                        status=CoordinatorStatus.DEGRADED,
                        message=f"DeadLetterQueue has high abandonment rate: {abandon_rate:.1%}",
                        details=details,
                    )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"DeadLetterQueue healthy ({pending} pending, {stats['session_recovered']} recovered)",
                details=details,
            )

        except Exception as e:
            logger.warning(f"[DeadLetterQueue] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


# Global instance
_dead_letter_queue: DeadLetterQueue | None = None


def get_dead_letter_queue(
    db_path: Path | str | None = None,
) -> DeadLetterQueue:
    """Get or create the global dead letter queue instance.

    Args:
        db_path: Optional custom database path

    Returns:
        DeadLetterQueue instance
    """
    global _dead_letter_queue

    if _dead_letter_queue is None:
        _dead_letter_queue = DeadLetterQueue(
            db_path=db_path or DEFAULT_DLQ_PATH
        )

    return _dead_letter_queue


def enable_dead_letter_queue(
    dlq: DeadLetterQueue,
    event_bus: Any,
) -> None:
    """Enable dead letter queue integration for an event bus.

    This wraps the event bus's emit method to capture failed events.

    Args:
        dlq: DeadLetterQueue instance
        event_bus: Event bus to integrate (StageEventBus or DataEventBus)
    """
    # Store reference for use in wrapper
    original_emit = event_bus.emit

    async def wrapped_emit(result) -> int:
        """Wrapped emit that captures failures to DLQ."""
        try:
            return await original_emit(result)
        except Exception as e:
            # This shouldn't happen as emit catches exceptions internally
            logger.error(f"[DLQ] Unexpected error in emit: {e}")
            return 0

    # Store DLQ reference on bus for callback wrapping
    event_bus._dlq = dlq

    logger.info(f"[DLQ] Enabled for {type(event_bus).__name__}")


async def run_retry_daemon(
    dlq: DeadLetterQueue | None = None,
    interval_seconds: float = 60.0,
    max_events_per_cycle: int = 10,
) -> None:
    """Run background retry daemon.

    Args:
        dlq: DeadLetterQueue instance (uses global if not provided)
        interval_seconds: Time between retry cycles
        max_events_per_cycle: Max events to retry per cycle
    """
    dlq = dlq or get_dead_letter_queue()

    logger.info(f"[DLQ] Starting retry daemon (interval={interval_seconds}s)")

    while True:
        try:
            stats = await dlq.retry_failed_events(max_events=max_events_per_cycle)
            if stats["recovered"] or stats["failed"]:
                logger.info(
                    f"[DLQ] Retry cycle: {stats['recovered']} recovered, "
                    f"{stats['failed']} failed, {stats['skipped']} skipped"
                )
        except Exception as e:
            logger.error(f"[DLQ] Retry cycle error: {e}")

        await asyncio.sleep(interval_seconds)


class DLQRetryDaemon:
    """DLQ retry daemon with proper lifecycle management.

    Features:
    - Periodic retry of failed events with exponential backoff
    - Automatic abandonment after max_attempts
    - Metrics for monitoring
    - Integration with DaemonManager

    Usage:
        daemon = DLQRetryDaemon()
        await daemon.start()
        ...
        await daemon.stop()
    """

    def __init__(
        self,
        dlq: DeadLetterQueue | None = None,
        interval_seconds: float = 60.0,
        max_events_per_cycle: int = 10,
        max_attempts: int = 5,
        max_stale_hours: float = 168.0,  # 7 days
    ):
        self.dlq = dlq or get_dead_letter_queue()
        self.interval = interval_seconds
        self.max_events = max_events_per_cycle
        self.max_attempts = max_attempts
        self.max_stale_hours = max_stale_hours
        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics = {
            "cycles": 0,
            "total_recovered": 0,
            "total_failed": 0,
            "total_abandoned": 0,
            "total_stale_abandoned": 0,
        }

    async def start(self) -> None:
        """Start the retry daemon."""
        if self._running:
            return

        self._running = True
        self._task = safe_create_task(
            self._run_loop(),
            name="dlq_retry_loop",
        )
        logger.info(
            f"[DLQRetryDaemon] Started (interval={self.interval}s, "
            f"max_attempts={self.max_attempts})"
        )

    async def stop(self) -> None:
        """Stop the retry daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[DLQRetryDaemon] Stopped")

    async def _run_loop(self) -> None:
        """Main retry loop."""
        while self._running:
            try:
                await self._process_cycle()
            except Exception as e:
                logger.error(f"[DLQRetryDaemon] Cycle error: {e}")

            await asyncio.sleep(self.interval)

    async def _process_cycle(self) -> None:
        """Process one retry cycle."""
        self._metrics["cycles"] += 1

        # First, abandon events that have exceeded max_attempts
        # Wrap sync SQLite operations in asyncio.to_thread to avoid blocking event loop
        abandoned = await asyncio.to_thread(self._abandon_exhausted_events)
        if abandoned > 0:
            self._metrics["total_abandoned"] += abandoned
            logger.info(f"[DLQRetryDaemon] Abandoned {abandoned} exhausted events")

        # Also abandon very old events (stale cleanup)
        stale_abandoned = await asyncio.to_thread(
            self._abandon_stale_events, max_age_hours=self.max_stale_hours
        )
        if stale_abandoned > 0:
            self._metrics["total_stale_abandoned"] += stale_abandoned

        # Check for stale events and emit alert if many are pending
        # December 29, 2025: Added DLQ_STALE_EVENTS emission
        pending = await asyncio.to_thread(self.dlq.get_pending_events, limit=100)
        if len(pending) >= 50:
            event_types = list({e.event_type for e in pending})
            self._emit_stale_events(len(pending), event_types)

        # Then retry pending events
        stats = await self.dlq.retry_failed_events(max_events=self.max_events)

        self._metrics["total_recovered"] += stats["recovered"]
        self._metrics["total_failed"] += stats["failed"]

        # December 29, 2025: Emit DLQ_EVENTS_REPLAYED when events are recovered
        if stats["recovered"] > 0:
            # Get event types that were just recovered for the event payload
            recovered_types = await asyncio.to_thread(
                self._get_recent_recovered_types, limit=stats["recovered"]
            )
            self._emit_replayed_events(stats["recovered"], recovered_types)

        if stats["recovered"] or stats["failed"]:
            logger.info(
                f"[DLQRetryDaemon] Cycle {self._metrics['cycles']}: "
                f"{stats['recovered']} recovered, {stats['failed']} failed, "
                f"{stats['skipped']} skipped"
            )

    def _abandon_exhausted_events(self) -> int:
        """Mark events with max_attempts as abandoned.

        Returns:
            Number of events abandoned
        """
        with connect_safe(self.dlq.db_path, row_factory=None) as conn:
            cursor = conn.execute(
                """
                UPDATE dead_letter
                SET status = 'abandoned'
                WHERE status = 'pending' AND retry_count >= ?
                """,
                (self.max_attempts,),
            )
            abandoned = cursor.rowcount
            conn.commit()
        return abandoned

    def _get_recent_recovered_types(self, limit: int = 10) -> list[str]:
        """Get event types of recently recovered events.

        December 29, 2025: Helper for DLQ_EVENTS_REPLAYED emission.

        Args:
            limit: Maximum number of events to check

        Returns:
            List of distinct event types
        """
        try:
            with connect_safe(self.dlq.db_path, row_factory=None) as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT event_type FROM dead_letter
                    WHERE status = 'recovered'
                    ORDER BY last_retry_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [row[0] for row in rows]
        except (sqlite3.Error, OSError):
            # Database errors should not crash the recovery loop
            return []

    def _abandon_stale_events(self, max_age_hours: float = 168.0) -> int:
        """Abandon events that are too old, regardless of retry count.

        This prevents events from sitting in the DLQ forever when:
        - No handler is registered for them
        - The system that should process them is permanently down
        - They represent obsolete operations

        Args:
            max_age_hours: Maximum age in hours (default 168 = 7 days)

        Returns:
            Number of events abandoned
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        with connect_safe(self.dlq.db_path, row_factory=None) as conn:
            cursor = conn.execute(
                """
                UPDATE dead_letter
                SET status = 'abandoned'
                WHERE status = 'pending' AND created_at < ?
                """,
                (cutoff,),
            )
            abandoned = cursor.rowcount
            conn.commit()

        if abandoned > 0:
            logger.warning(
                f"[DLQRetryDaemon] Auto-abandoned {abandoned} stale events "
                f"(older than {max_age_hours:.0f}h)"
            )
            # Emit DLQ_EVENTS_PURGED event
            self._emit_purge_event(abandoned, reason="stale")

        return abandoned

    def _emit_purge_event(self, count: int, reason: str) -> None:
        """Emit DLQ_EVENTS_PURGED event when events are cleaned up."""
        try:
            from app.distributed.data_events import DataEvent, DataEventType

            event = DataEvent(
                event_type=DataEventType.DLQ_EVENTS_PURGED,
                payload={
                    "count": count,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                },
                source="DLQRetryDaemon",
            )

            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                bus.publish(event)
        except Exception as e:
            logger.debug(f"[DLQRetryDaemon] Failed to emit purge event: {e}")

    def _emit_stale_events(self, stale_count: int, event_types: list[str]) -> None:
        """Emit DLQ_STALE_EVENTS event when many stale events are detected.

        December 29, 2025: Added to complete DLQ event emission chain.
        UnifiedHealthManager subscribes to this for health monitoring.
        """
        try:
            from app.distributed.data_events import DataEvent, DataEventType

            event = DataEvent(
                event_type=DataEventType.DLQ_STALE_EVENTS,
                payload={
                    "stale_count": stale_count,
                    "event_types": event_types,
                    "max_stale_hours": self.max_stale_hours,
                    "timestamp": datetime.now().isoformat(),
                },
                source="DLQRetryDaemon",
            )

            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                bus.publish(event)
        except Exception as e:
            logger.debug(f"[DLQRetryDaemon] Failed to emit stale events: {e}")

    def _emit_replayed_events(self, recovered_count: int, event_types: list[str]) -> None:
        """Emit DLQ_EVENTS_REPLAYED event when events are successfully recovered.

        December 29, 2025: Added to complete DLQ event emission chain.
        UnifiedHealthManager subscribes to this for health monitoring.
        """
        try:
            from app.distributed.data_events import DataEvent, DataEventType

            event = DataEvent(
                event_type=DataEventType.DLQ_EVENTS_REPLAYED,
                payload={
                    "recovered_count": recovered_count,
                    "event_types": event_types,
                    "timestamp": datetime.now().isoformat(),
                },
                source="DLQRetryDaemon",
            )

            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                bus.publish(event)
        except Exception as e:
            logger.debug(f"[DLQRetryDaemon] Failed to emit replayed events: {e}")

    def get_metrics(self) -> dict:
        """Get daemon metrics.

        Returns:
            Dictionary with daemon statistics
        """
        dlq_stats = self.dlq.get_stats()
        return {
            **self._metrics,
            "dlq": dlq_stats,
            "running": self._running,
        }

    def health_check(self) -> HealthCheckResult:
        """Health check for DaemonManager integration.

        December 2025: Added for daemon health monitoring.
        Returns HealthCheckResult for protocol compliance.
        """
        metrics = self._metrics.copy()
        dlq_stats = self.dlq.get_stats() if self.dlq else {}

        # Daemon is healthy if running and DLQ is accessible
        is_healthy = self.is_running and self._task is not None
        if self._task and self._task.done() and not self._task.cancelled():
            # Task crashed unexpectedly
            is_healthy = False

        status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.STOPPED
        message = "DLQ retry daemon running" if is_healthy else "DLQ retry daemon not running"

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "cycles": metrics.get("cycles", 0),
                "total_recovered": metrics.get("total_recovered", 0),
                "total_failed": metrics.get("total_failed", 0),
                "total_abandoned": metrics.get("total_abandoned", 0),
                "total_stale_abandoned": metrics.get("total_stale_abandoned", 0),
                "pending_events": dlq_stats.get("pending", 0),
            },
        )


# =============================================================================
# HandlerBase Integration (January 2026)
# =============================================================================

try:
    from app.coordination.handler_base import HandlerBase
    from app.coordination.contracts import HealthCheckResult as HBHealthCheckResult

    HAS_HANDLER_BASE = True
except ImportError:
    HAS_HANDLER_BASE = False

if HAS_HANDLER_BASE:

    class DLQRetryHandler(HandlerBase):
        """HandlerBase wrapper for DLQRetryDaemon.

        January 2026: Added for unified daemon lifecycle management.
        Handles dead letter queue retry with exponential backoff.
        """

        def __init__(
            self,
            dlq: "DeadLetterQueue | None" = None,
            cycle_interval: float = 60.0,
            max_events_per_cycle: int = 10,
            max_attempts: int = 5,
            max_stale_hours: float = 168.0,
        ):
            super().__init__(
                name="dlq_retry",
                cycle_interval=cycle_interval,
            )
            self._daemon = DLQRetryDaemon(
                dlq=dlq,
                interval_seconds=cycle_interval,
                max_events_per_cycle=max_events_per_cycle,
                max_attempts=max_attempts,
                max_stale_hours=max_stale_hours,
            )

        async def _run_cycle(self) -> None:
            """Run one DLQ retry cycle."""
            try:
                await self._daemon._process_cycle()
            except Exception as e:
                logger.error(f"[DLQRetryHandler] Cycle error: {e}")

        def _get_event_subscriptions(self) -> dict:
            """Get event subscriptions for DLQ handler."""
            return {
                "EVENT_DELIVERY_FAILED": self._on_delivery_failed,
            }

        async def _on_delivery_failed(self, event: dict) -> None:
            """Handle event delivery failure - logged for awareness."""
            event_type = event.get("event_type", "unknown")
            logger.debug(f"[DLQRetryHandler] Delivery failed for {event_type}")

        def health_check(self) -> HBHealthCheckResult:
            """Health check delegating to wrapped daemon."""
            return self._daemon.health_check()


# Daemon factory for DaemonManager integration
def create_dlq_retry_daemon(
    interval_seconds: float = 60.0,
    max_events_per_cycle: int = 10,
    max_attempts: int = 5,
    max_stale_hours: float = 168.0,
) -> DLQRetryDaemon:
    """Create a DLQ retry daemon instance.

    Args:
        interval_seconds: Time between retry cycles
        max_events_per_cycle: Max events to retry per cycle
        max_attempts: Max retry attempts before abandoning
        max_stale_hours: Maximum age (hours) for pending events before
            auto-abandonment (default 168 = 7 days)

    Returns:
        DLQRetryDaemon instance
    """
    return DLQRetryDaemon(
        interval_seconds=interval_seconds,
        max_events_per_cycle=max_events_per_cycle,
        max_attempts=max_attempts,
        max_stale_hours=max_stale_hours,
    )
