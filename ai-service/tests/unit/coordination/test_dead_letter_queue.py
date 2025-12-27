"""Unit tests for DeadLetterQueue (December 2025).

Tests the fault-tolerant event handling system that captures failed events
and enables automatic retry with exponential backoff.

Created: December 27, 2025
"""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.dead_letter_queue import (
    DEFAULT_DLQ_PATH,
    DeadLetterQueue,
    DLQRetryDaemon,
    FailedEvent,
    create_dlq_retry_daemon,
    enable_dead_letter_queue,
    get_dead_letter_queue,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def dlq(temp_db: Path) -> DeadLetterQueue:
    """Create a DeadLetterQueue with temporary database."""
    return DeadLetterQueue(
        db_path=temp_db,
        max_retries=3,
        base_backoff_seconds=1.0,
        max_backoff_seconds=10.0,
    )


@pytest.fixture
def failed_event() -> FailedEvent:
    """Create a sample failed event."""
    return FailedEvent(
        event_id="test-event-123",
        event_type="TRAINING_COMPLETED",
        payload={"model_path": "/path/to/model.pth", "config_key": "hex8_2p"},
        handler_name="test_handler",
        error="Connection timeout",
        retry_count=0,
        source="test",
    )


# ============================================================================
# FailedEvent Tests
# ============================================================================


class TestFailedEvent:
    """Tests for FailedEvent dataclass."""

    def test_to_dict(self, failed_event: FailedEvent) -> None:
        """Test conversion to dictionary."""
        d = failed_event.to_dict()
        assert d["event_id"] == "test-event-123"
        assert d["event_type"] == "TRAINING_COMPLETED"
        assert d["payload"]["config_key"] == "hex8_2p"
        assert d["handler_name"] == "test_handler"
        assert d["error"] == "Connection timeout"
        assert d["retry_count"] == 0
        assert d["source"] == "test"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "event_id": "event-456",
            "event_type": "MODEL_PROMOTED",
            "payload": {"model_id": "abc123"},
            "handler_name": "promotion_handler",
            "error": "Disk full",
            "retry_count": 2,
            "created_at": "2025-12-27T10:00:00",
            "last_retry_at": "2025-12-27T10:05:00",
            "source": "promotion",
        }
        event = FailedEvent.from_dict(data)
        assert event.event_id == "event-456"
        assert event.event_type == "MODEL_PROMOTED"
        assert event.payload == {"model_id": "abc123"}
        assert event.retry_count == 2
        assert event.last_retry_at == "2025-12-27T10:05:00"

    def test_from_dict_with_json_payload(self) -> None:
        """Test creation from dict with JSON-encoded payload."""
        import json

        data = {
            "event_id": "event-789",
            "event_type": "TEST_EVENT",
            "payload": json.dumps({"key": "value"}),
            "handler_name": "handler",
            "error": "Error",
        }
        event = FailedEvent.from_dict(data)
        assert event.payload == {"key": "value"}

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        event = FailedEvent(
            event_id="event-1",
            event_type="TYPE_A",
            payload={"data": "test"},
            handler_name="handler_a",
            error="Some error",
        )
        assert event.retry_count == 0
        assert event.last_retry_at is None
        assert event.source == "unknown"
        assert event.created_at is not None


# ============================================================================
# DeadLetterQueue Tests
# ============================================================================


class TestDeadLetterQueueInit:
    """Tests for DLQ initialization."""

    def test_init_creates_db(self, temp_db: Path) -> None:
        """Test that initialization creates database and tables."""
        dlq = DeadLetterQueue(db_path=temp_db)

        # Verify table exists
        with sqlite3.connect(temp_db) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "dead_letter" in table_names

    def test_init_creates_indexes(self, temp_db: Path) -> None:
        """Test that initialization creates indexes."""
        dlq = DeadLetterQueue(db_path=temp_db)

        with sqlite3.connect(temp_db) as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            index_names = [i[0] for i in indexes]
            assert "idx_dead_letter_status" in index_names
            assert "idx_dead_letter_event_type" in index_names

    def test_init_creates_parent_dirs(self) -> None:
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "dlq.db"
            dlq = DeadLetterQueue(db_path=db_path)
            assert db_path.parent.exists()
            db_path.unlink()

    def test_default_config(self, temp_db: Path) -> None:
        """Test default configuration values."""
        dlq = DeadLetterQueue(db_path=temp_db)
        assert dlq.max_retries == 5
        assert dlq.base_backoff == 60.0
        assert dlq.max_backoff == 3600.0


class TestDeadLetterQueueCapture:
    """Tests for event capture functionality."""

    def test_capture_event(self, dlq: DeadLetterQueue) -> None:
        """Test capturing a failed event."""
        event_id = dlq.capture(
            event_type="TRAINING_FAILED",
            payload={"error_code": 500},
            handler_name="training_handler",
            error="Out of memory",
            source="gpu_node",
        )

        assert event_id is not None
        assert dlq._events_captured == 1

    def test_capture_stores_in_db(self, dlq: DeadLetterQueue) -> None:
        """Test that captured events are stored in database."""
        event_id = dlq.capture(
            event_type="SYNC_FAILED",
            payload={"nodes": ["node1", "node2"]},
            handler_name="sync_handler",
            error="Network error",
        )

        with sqlite3.connect(dlq.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        assert row is not None
        assert row["event_type"] == "SYNC_FAILED"
        assert row["status"] == "pending"
        assert row["retry_count"] == 0

    def test_capture_increments_counter(self, dlq: DeadLetterQueue) -> None:
        """Test that capture increments the counter."""
        assert dlq._events_captured == 0

        dlq.capture(
            event_type="EVENT_1",
            payload={},
            handler_name="handler",
            error="Error 1",
        )
        assert dlq._events_captured == 1

        dlq.capture(
            event_type="EVENT_2",
            payload={},
            handler_name="handler",
            error="Error 2",
        )
        assert dlq._events_captured == 2


class TestDeadLetterQueueQuery:
    """Tests for querying events."""

    def test_get_pending_events(self, dlq: DeadLetterQueue) -> None:
        """Test getting pending events."""
        # Capture some events
        for i in range(5):
            dlq.capture(
                event_type=f"EVENT_{i}",
                payload={"index": i},
                handler_name="handler",
                error=f"Error {i}",
            )

        pending = dlq.get_pending_events(limit=3)
        assert len(pending) == 3
        assert all(isinstance(e, FailedEvent) for e in pending)

    def test_get_pending_events_by_type(self, dlq: DeadLetterQueue) -> None:
        """Test filtering pending events by type."""
        dlq.capture("TYPE_A", {}, "handler", "error")
        dlq.capture("TYPE_B", {}, "handler", "error")
        dlq.capture("TYPE_A", {}, "handler", "error")

        type_a = dlq.get_pending_events(event_type="TYPE_A")
        assert len(type_a) == 2

        type_b = dlq.get_pending_events(event_type="TYPE_B")
        assert len(type_b) == 1

    def test_get_failed_events(self, dlq: DeadLetterQueue) -> None:
        """Test getting all failed events."""
        dlq.capture("EVENT", {}, "handler", "error")

        events = dlq.get_failed_events(limit=10)
        assert len(events) == 1
        assert isinstance(events[0], dict)

    def test_get_failed_events_excludes_abandoned(self, dlq: DeadLetterQueue) -> None:
        """Test that abandoned events are excluded by default."""
        event_id = dlq.capture("EVENT", {}, "handler", "error")

        # Mark as abandoned
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET status = 'abandoned' WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()

        # Should not include abandoned
        events = dlq.get_failed_events(include_abandoned=False)
        assert len(events) == 0

        # Should include when requested
        events = dlq.get_failed_events(include_abandoned=True)
        assert len(events) == 1


class TestDeadLetterQueueRetry:
    """Tests for retry functionality."""

    @pytest.mark.asyncio
    async def test_retry_event_success(self, dlq: DeadLetterQueue) -> None:
        """Test successful event retry."""
        event_id = dlq.capture(
            event_type="TRAINING_COMPLETED",
            payload={"model": "test"},
            handler_name="handler",
            error="Temporary failure",
        )

        # Register a successful handler
        handler = AsyncMock()
        dlq.register_handler("TRAINING_COMPLETED", handler)

        success = await dlq.retry_event(event_id)
        assert success is True
        handler.assert_called_once_with({"model": "test"})

        # Check status updated
        with sqlite3.connect(dlq.db_path) as conn:
            status = conn.execute(
                "SELECT status FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()[0]
        assert status == "recovered"

    @pytest.mark.asyncio
    async def test_retry_event_failure(self, dlq: DeadLetterQueue) -> None:
        """Test failed event retry."""
        event_id = dlq.capture(
            event_type="SYNC_EVENT",
            payload={},
            handler_name="handler",
            error="Original error",
        )

        # Register a failing handler
        handler = AsyncMock(side_effect=RuntimeError("Retry also failed"))
        dlq.register_handler("SYNC_EVENT", handler)

        success = await dlq.retry_event(event_id)
        assert success is False

        # Check retry count incremented
        with sqlite3.connect(dlq.db_path) as conn:
            count = conn.execute(
                "SELECT retry_count FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_retry_event_abandons_after_max_retries(
        self, dlq: DeadLetterQueue
    ) -> None:
        """Test that events are abandoned after max retries."""
        event_id = dlq.capture(
            event_type="FAILING_EVENT",
            payload={},
            handler_name="handler",
            error="Persistent failure",
        )

        # Set retry count to max - 1
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET retry_count = ? WHERE event_id = ?",
                (dlq.max_retries - 1, event_id),
            )
            conn.commit()

        # Register a failing handler
        handler = AsyncMock(side_effect=RuntimeError("Still failing"))
        dlq.register_handler("FAILING_EVENT", handler)

        await dlq.retry_event(event_id)

        # Check status is abandoned
        with sqlite3.connect(dlq.db_path) as conn:
            status = conn.execute(
                "SELECT status FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()[0]
        assert status == "abandoned"

    @pytest.mark.asyncio
    async def test_retry_event_not_found(self, dlq: DeadLetterQueue) -> None:
        """Test retry of non-existent event."""
        success = await dlq.retry_event("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_retry_event_no_handler(self, dlq: DeadLetterQueue) -> None:
        """Test retry when no handler is registered."""
        event_id = dlq.capture(
            event_type="UNHANDLED_EVENT",
            payload={},
            handler_name="handler",
            error="Error",
        )

        # No handler registered
        success = await dlq.retry_event(event_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_retry_failed_events_with_backoff(
        self, dlq: DeadLetterQueue
    ) -> None:
        """Test retry respects exponential backoff."""
        event_id = dlq.capture(
            event_type="BACKOFF_EVENT",
            payload={},
            handler_name="handler",
            error="Error",
        )

        # Set last_retry_at to now
        now = datetime.now().isoformat()
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET last_retry_at = ?, retry_count = 1 WHERE event_id = ?",
                (now, event_id),
            )
            conn.commit()

        # Register handler
        handler = AsyncMock()
        dlq.register_handler("BACKOFF_EVENT", handler)

        # Should be skipped due to backoff
        stats = await dlq.retry_failed_events(max_events=10)
        assert stats["skipped"] == 1
        assert stats["recovered"] == 0
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_failed_events_batch(self, dlq: DeadLetterQueue) -> None:
        """Test batch retry of failed events."""
        # Capture multiple events
        for i in range(5):
            dlq.capture(
                event_type="BATCH_EVENT",
                payload={"index": i},
                handler_name="handler",
                error=f"Error {i}",
            )

        # Register successful handler
        handler = AsyncMock()
        dlq.register_handler("BATCH_EVENT", handler)

        stats = await dlq.retry_failed_events(max_events=3)
        assert stats["recovered"] == 3
        assert handler.call_count == 3


class TestDeadLetterQueuePurge:
    """Tests for purging old events."""

    def test_purge_old_events(self, dlq: DeadLetterQueue) -> None:
        """Test purging old events."""
        event_id = dlq.capture("OLD_EVENT", {}, "handler", "error")

        # Set old created_at and mark as recovered
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET created_at = ?, status = 'recovered' WHERE event_id = ?",
                (old_date, event_id),
            )
            conn.commit()

        # Purge events older than 7 days
        deleted = dlq.purge_old_events(days=7)
        assert deleted == 1

    def test_purge_keeps_pending_events(self, dlq: DeadLetterQueue) -> None:
        """Test that purge keeps pending events even if old."""
        event_id = dlq.capture("PENDING_EVENT", {}, "handler", "error")

        # Set old created_at but keep as pending
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET created_at = ? WHERE event_id = ?",
                (old_date, event_id),
            )
            conn.commit()

        # Purge should not delete pending events
        deleted = dlq.purge_old_events(days=7)
        assert deleted == 0


class TestDeadLetterQueueStats:
    """Tests for statistics."""

    def test_get_stats(self, dlq: DeadLetterQueue) -> None:
        """Test getting queue statistics."""
        # Capture events with different statuses
        dlq.capture("EVENT_A", {}, "handler", "error")
        dlq.capture("EVENT_B", {}, "handler", "error")
        event_id = dlq.capture("EVENT_C", {}, "handler", "error")

        # Mark one as recovered
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET status = 'recovered' WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()

        stats = dlq.get_stats()
        assert stats["pending"] == 2
        assert stats["recovered"] == 1
        assert stats["abandoned"] == 0
        assert stats["total"] == 3
        assert "by_event_type" in stats

    def test_get_stats_by_event_type(self, dlq: DeadLetterQueue) -> None:
        """Test stats breakdown by event type."""
        dlq.capture("TYPE_X", {}, "handler", "error")
        dlq.capture("TYPE_X", {}, "handler", "error")
        dlq.capture("TYPE_Y", {}, "handler", "error")

        stats = dlq.get_stats()
        assert stats["by_event_type"]["TYPE_X"] == 2
        assert stats["by_event_type"]["TYPE_Y"] == 1


class TestDeadLetterQueueHandlers:
    """Tests for handler registration."""

    def test_register_handler(self, dlq: DeadLetterQueue) -> None:
        """Test registering a handler."""
        handler = AsyncMock()
        dlq.register_handler("MY_EVENT", handler)
        assert "MY_EVENT" in dlq._handlers
        assert handler in dlq._handlers["MY_EVENT"]

    def test_register_multiple_handlers(self, dlq: DeadLetterQueue) -> None:
        """Test registering multiple handlers for same event type."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        dlq.register_handler("SHARED_EVENT", handler1)
        dlq.register_handler("SHARED_EVENT", handler2)

        assert len(dlq._handlers["SHARED_EVENT"]) == 2


# ============================================================================
# DLQRetryDaemon Tests
# ============================================================================


class TestDLQRetryDaemon:
    """Tests for DLQRetryDaemon class."""

    @pytest.mark.asyncio
    async def test_start_stop(self, dlq: DeadLetterQueue) -> None:
        """Test daemon start and stop."""
        daemon = DLQRetryDaemon(
            dlq=dlq,
            interval_seconds=0.1,
            max_events_per_cycle=5,
        )

        assert daemon._running is False
        assert daemon._task is None

        await daemon.start()
        assert daemon._running is True
        assert daemon._task is not None

        await asyncio.sleep(0.05)  # Let it run briefly

        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, dlq: DeadLetterQueue) -> None:
        """Test that starting twice is safe."""
        daemon = DLQRetryDaemon(dlq=dlq, interval_seconds=0.1)

        await daemon.start()
        task1 = daemon._task

        await daemon.start()  # Second start
        task2 = daemon._task

        assert task1 is task2  # Same task
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_process_cycle(self, dlq: DeadLetterQueue) -> None:
        """Test a single processing cycle."""
        daemon = DLQRetryDaemon(
            dlq=dlq,
            interval_seconds=60.0,  # Won't trigger during test
            max_events_per_cycle=10,
        )

        # Add an event
        dlq.capture("TEST_EVENT", {}, "handler", "error")

        # Register successful handler
        handler = AsyncMock()
        dlq.register_handler("TEST_EVENT", handler)

        await daemon._process_cycle()

        assert daemon._metrics["cycles"] == 1
        assert daemon._metrics["total_recovered"] == 1

    @pytest.mark.asyncio
    async def test_abandon_exhausted_events(self, dlq: DeadLetterQueue) -> None:
        """Test abandoning events that exceeded max attempts."""
        daemon = DLQRetryDaemon(
            dlq=dlq,
            max_attempts=3,
        )

        # Add event with retry_count >= max_attempts
        event_id = dlq.capture("EXHAUSTED_EVENT", {}, "handler", "error")
        with sqlite3.connect(dlq.db_path) as conn:
            conn.execute(
                "UPDATE dead_letter SET retry_count = 5 WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()

        abandoned = daemon._abandon_exhausted_events()
        assert abandoned == 1

        # Verify status
        with sqlite3.connect(dlq.db_path) as conn:
            status = conn.execute(
                "SELECT status FROM dead_letter WHERE event_id = ?",
                (event_id,),
            ).fetchone()[0]
        assert status == "abandoned"

    def test_get_metrics(self, dlq: DeadLetterQueue) -> None:
        """Test getting daemon metrics."""
        daemon = DLQRetryDaemon(dlq=dlq)
        metrics = daemon.get_metrics()

        assert "cycles" in metrics
        assert "total_recovered" in metrics
        assert "total_failed" in metrics
        assert "total_abandoned" in metrics
        assert "dlq" in metrics
        assert "running" in metrics

    def test_health_check_not_running(self, dlq: DeadLetterQueue) -> None:
        """Test health check when daemon is not running."""
        daemon = DLQRetryDaemon(dlq=dlq)
        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self, dlq: DeadLetterQueue) -> None:
        """Test health check when daemon is running."""
        daemon = DLQRetryDaemon(dlq=dlq, interval_seconds=0.1)
        await daemon.start()

        result = daemon.health_check()
        assert result.healthy is True
        assert "running" in result.message.lower()

        await daemon.stop()


class TestDLQRetryDaemonLoop:
    """Tests for daemon loop behavior."""

    @pytest.mark.asyncio
    async def test_loop_runs_multiple_cycles(self, dlq: DeadLetterQueue) -> None:
        """Test that loop runs multiple cycles."""
        daemon = DLQRetryDaemon(
            dlq=dlq,
            interval_seconds=0.05,
        )

        await daemon.start()
        await asyncio.sleep(0.15)  # Allow ~3 cycles
        await daemon.stop()

        assert daemon._metrics["cycles"] >= 2


# ============================================================================
# Module-level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_dead_letter_queue_singleton(self) -> None:
        """Test singleton behavior of get_dead_letter_queue."""
        # Reset global
        import app.coordination.dead_letter_queue as dlq_module

        dlq_module._dead_letter_queue = None

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            dlq1 = get_dead_letter_queue(db_path=db_path)
            dlq2 = get_dead_letter_queue()

            assert dlq1 is dlq2
        finally:
            dlq_module._dead_letter_queue = None
            db_path.unlink(missing_ok=True)

    def test_enable_dead_letter_queue(self, dlq: DeadLetterQueue) -> None:
        """Test enabling DLQ for an event bus."""
        mock_bus = MagicMock()
        mock_bus.emit = AsyncMock(return_value=1)

        enable_dead_letter_queue(dlq, mock_bus)

        assert hasattr(mock_bus, "_dlq")
        assert mock_bus._dlq is dlq

    def test_create_dlq_retry_daemon(self) -> None:
        """Test factory function for DLQRetryDaemon."""
        daemon = create_dlq_retry_daemon(
            interval_seconds=30.0,
            max_events_per_cycle=20,
            max_attempts=10,
        )

        assert isinstance(daemon, DLQRetryDaemon)
        assert daemon.interval == 30.0
        assert daemon.max_events == 20
        assert daemon.max_attempts == 10


class TestRunRetryDaemon:
    """Tests for run_retry_daemon function."""

    @pytest.mark.asyncio
    async def test_run_retry_daemon_loop(self, dlq: DeadLetterQueue) -> None:
        """Test run_retry_daemon function runs correctly."""
        from app.coordination.dead_letter_queue import run_retry_daemon

        # Create task but cancel it quickly
        task = asyncio.create_task(
            run_retry_daemon(
                dlq=dlq,
                interval_seconds=0.05,
                max_events_per_cycle=5,
            )
        )

        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
