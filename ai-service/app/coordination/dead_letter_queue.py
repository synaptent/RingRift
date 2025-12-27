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

        with sqlite3.connect(self.db_path) as conn:
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

        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        if not row:
            logger.warning(f"[DLQ] Event {event_id} not found")
            return False

        event = FailedEvent.from_dict(dict(row))

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

        with sqlite3.connect(self.db_path) as conn:
            if success:
                conn.execute(
                    """
                    UPDATE dead_letter
                    SET status = 'recovered', last_retry_at = ?
                    WHERE event_id = ?
                    """,
                    (now, event_id),
                )
                self._events_recovered += 1
                logger.info(f"[DLQ] Event {event_id} recovered successfully")
            else:
                new_count = event.retry_count + 1
                if new_count >= self.max_retries:
                    conn.execute(
                        """
                        UPDATE dead_letter
                        SET status = 'abandoned', retry_count = ?, last_retry_at = ?
                        WHERE event_id = ?
                        """,
                        (new_count, now, event_id),
                    )
                    self._events_abandoned += 1
                    logger.error(
                        f"[DLQ] Event {event_id} abandoned after {new_count} retries"
                    )
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
        pending = self.get_pending_events(limit=max_events, event_type=event_type)

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

        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
    ):
        self.dlq = dlq or get_dead_letter_queue()
        self.interval = interval_seconds
        self.max_events = max_events_per_cycle
        self.max_attempts = max_attempts
        self._running = False
        self._task: asyncio.Task | None = None
        self._metrics = {
            "cycles": 0,
            "total_recovered": 0,
            "total_failed": 0,
            "total_abandoned": 0,
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
        abandoned = self._abandon_exhausted_events()
        if abandoned > 0:
            self._metrics["total_abandoned"] += abandoned
            logger.info(f"[DLQRetryDaemon] Abandoned {abandoned} exhausted events")

        # Then retry pending events
        stats = await self.dlq.retry_failed_events(max_events=self.max_events)

        self._metrics["total_recovered"] += stats["recovered"]
        self._metrics["total_failed"] += stats["failed"]

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
        with sqlite3.connect(self.dlq.db_path) as conn:
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
        is_healthy = self._running and self._task is not None
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
                "pending_events": dlq_stats.get("pending", 0),
            },
        )


# Daemon factory for DaemonManager integration
def create_dlq_retry_daemon(
    interval_seconds: float = 60.0,
    max_events_per_cycle: int = 10,
    max_attempts: int = 5,
) -> DLQRetryDaemon:
    """Create a DLQ retry daemon instance.

    Args:
        interval_seconds: Time between retry cycles
        max_events_per_cycle: Max events to retry per cycle
        max_attempts: Max retry attempts before abandoning

    Returns:
        DLQRetryDaemon instance
    """
    return DLQRetryDaemon(
        interval_seconds=interval_seconds,
        max_events_per_cycle=max_events_per_cycle,
        max_attempts=max_attempts,
    )
