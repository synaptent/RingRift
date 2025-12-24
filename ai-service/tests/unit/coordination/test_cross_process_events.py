"""Tests for cross_process_events.py - Cross-Process Event Queue using SQLite.

This module tests:
- CrossProcessEvent dataclass
- CrossProcessEventQueue class
- CrossProcessEventPoller for background event handling
- Global convenience functions
"""

import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Test fixtures
@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_events.db"


@pytest.fixture
def queue(temp_db_path):
    """Create a fresh CrossProcessEventQueue for each test."""
    # Import here to avoid deprecation warning at module level
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from app.coordination.cross_process_events import CrossProcessEventQueue

    q = CrossProcessEventQueue(db_path=temp_db_path)
    yield q
    q.close()


@pytest.fixture
def reset_global_queue():
    """Reset the global event queue singleton between tests."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from app.coordination.cross_process_events import reset_event_queue
    reset_event_queue()
    yield
    reset_event_queue()


class TestCrossProcessEvent:
    """Tests for the CrossProcessEvent dataclass."""

    def test_event_creation(self):
        """Test creating an event with all fields."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEvent

        event = CrossProcessEvent(
            event_id=1,
            event_type="TEST_EVENT",
            payload={"key": "value"},
            source="test_source",
            created_at=1234567890.0,
            hostname="testhost",
        )

        assert event.event_id == 1
        assert event.event_type == "TEST_EVENT"
        assert event.payload == {"key": "value"}
        assert event.source == "test_source"
        assert event.created_at == 1234567890.0
        assert event.hostname == "testhost"

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEvent

        event = CrossProcessEvent(
            event_id=42,
            event_type="MODEL_PROMOTED",
            payload={"model_id": "abc123"},
            source="promotion_daemon",
            created_at=1700000000.0,
            hostname="cluster-node-1",
        )

        d = event.to_dict()
        assert d["event_id"] == 42
        assert d["event_type"] == "MODEL_PROMOTED"
        assert d["payload"] == {"model_id": "abc123"}
        assert d["source"] == "promotion_daemon"
        assert d["created_at"] == 1700000000.0
        assert d["hostname"] == "cluster-node-1"

    def test_event_from_row(self):
        """Test creating event from SQLite row."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEvent

        # Create a mock row object that acts like sqlite3.Row
        class MockRow:
            def __getitem__(self, key):
                data = {
                    "event_id": 99,
                    "event_type": "TRAINING_COMPLETED",
                    "payload": '{"accuracy": 0.95}',
                    "source": "trainer",
                    "created_at": 1700000100.0,
                    "hostname": "gpu-node",
                }
                return data[key]

        event = CrossProcessEvent.from_row(MockRow())
        assert event.event_id == 99
        assert event.event_type == "TRAINING_COMPLETED"
        assert event.payload == {"accuracy": 0.95}
        assert event.source == "trainer"


class TestCrossProcessEventQueue:
    """Tests for the CrossProcessEventQueue class."""

    def test_queue_initialization(self, queue, temp_db_path):
        """Test queue creates database with proper schema."""
        assert temp_db_path.exists()

        # Verify tables exist
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "events" in tables
        assert "subscribers" in tables
        assert "acks" in tables

    def test_publish_event(self, queue):
        """Test publishing an event."""
        event_id = queue.publish(
            event_type="TEST_EVENT",
            payload={"data": 123},
            source="test",
        )

        assert event_id > 0
        assert isinstance(event_id, int)

    def test_publish_event_empty_payload(self, queue):
        """Test publishing event with empty payload."""
        event_id = queue.publish(event_type="PING", source="test")
        assert event_id > 0

    def test_publish_event_creates_record(self, queue, temp_db_path):
        """Test that publish actually creates database record."""
        event_id = queue.publish(
            event_type="CHECK_EVENT",
            payload={"foo": "bar"},
            source="checker",
        )

        # Verify in database
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT event_type, payload, source FROM events WHERE event_id = ?",
            (event_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "CHECK_EVENT"
        assert json.loads(row[1]) == {"foo": "bar"}
        assert row[2] == "checker"

    def test_subscribe_creates_subscriber(self, queue):
        """Test subscribing to events."""
        subscriber_id = queue.subscribe(
            process_name="test_process",
            event_types=["TYPE_A", "TYPE_B"],
        )

        assert subscriber_id is not None
        assert "test_process" in subscriber_id

    def test_subscribe_updates_heartbeat(self, queue):
        """Test that subscribe updates last_poll_at."""
        subscriber_id = queue.subscribe("my_process")

        # Get subscriber info from stats
        stats = queue.get_stats()

        # Subscriber should be active
        assert len(stats["active_subscribers"]) == 1
        assert stats["active_subscribers"][0]["process_name"] == "my_process"

    def test_poll_returns_unacked_events(self, queue):
        """Test polling returns only unacknowledged events."""
        # Publish some events
        event_id1 = queue.publish("TYPE_A", {"seq": 1}, "test")
        event_id2 = queue.publish("TYPE_A", {"seq": 2}, "test")

        # Subscribe and poll
        subscriber_id = queue.subscribe("poller")
        events = queue.poll(subscriber_id)

        assert len(events) == 2
        assert events[0].event_id == event_id1
        assert events[1].event_id == event_id2

    def test_poll_filters_by_event_type(self, queue):
        """Test polling with event type filter."""
        queue.publish("TYPE_A", {}, "test")
        queue.publish("TYPE_B", {}, "test")
        queue.publish("TYPE_A", {}, "test")

        subscriber_id = queue.subscribe("poller")
        events = queue.poll(subscriber_id, event_types=["TYPE_A"])

        assert len(events) == 2
        assert all(e.event_type == "TYPE_A" for e in events)

    def test_poll_since_event_id(self, queue):
        """Test polling with since_event_id filter."""
        event_id1 = queue.publish("TYPE_A", {"seq": 1}, "test")
        event_id2 = queue.publish("TYPE_A", {"seq": 2}, "test")
        event_id3 = queue.publish("TYPE_A", {"seq": 3}, "test")

        subscriber_id = queue.subscribe("poller")
        events = queue.poll(subscriber_id, since_event_id=event_id1)

        assert len(events) == 2
        assert events[0].event_id == event_id2
        assert events[1].event_id == event_id3

    def test_poll_respects_limit(self, queue):
        """Test poll limit parameter."""
        for i in range(10):
            queue.publish("TYPE_A", {"seq": i}, "test")

        subscriber_id = queue.subscribe("poller")
        events = queue.poll(subscriber_id, limit=3)

        assert len(events) == 3

    def test_poll_excludes_acked_events(self, queue):
        """Test that poll excludes acknowledged events."""
        event_id1 = queue.publish("TYPE_A", {"seq": 1}, "test")
        event_id2 = queue.publish("TYPE_A", {"seq": 2}, "test")

        subscriber_id = queue.subscribe("poller")

        # Ack first event
        queue.ack(subscriber_id, event_id1)

        # Poll should only return second event
        events = queue.poll(subscriber_id)
        assert len(events) == 1
        assert events[0].event_id == event_id2

    def test_ack_single_event(self, queue):
        """Test acknowledging a single event."""
        event_id = queue.publish("TYPE_A", {}, "test")
        subscriber_id = queue.subscribe("acker")

        result = queue.ack(subscriber_id, event_id)

        assert result is True

    def test_ack_batch(self, queue):
        """Test acknowledging multiple events at once."""
        event_ids = [
            queue.publish("TYPE_A", {"i": i}, "test")
            for i in range(5)
        ]

        subscriber_id = queue.subscribe("batch_acker")
        count = queue.ack_batch(subscriber_id, event_ids)

        assert count == 5

        # Verify events are acked by polling
        events = queue.poll(subscriber_id)
        assert len(events) == 0

    def test_ack_batch_empty(self, queue):
        """Test ack_batch with empty list."""
        subscriber_id = queue.subscribe("acker")
        count = queue.ack_batch(subscriber_id, [])
        assert count == 0

    def test_get_pending_count(self, queue):
        """Test getting pending event count."""
        subscriber_id = queue.subscribe("counter")

        # Initially empty
        assert queue.get_pending_count(subscriber_id) == 0

        # Publish events
        queue.publish("TYPE_A", {}, "test")
        queue.publish("TYPE_B", {}, "test")
        queue.publish("TYPE_A", {}, "test")

        assert queue.get_pending_count(subscriber_id) == 3
        assert queue.get_pending_count(subscriber_id, event_types=["TYPE_A"]) == 2

    def test_cleanup_old_events(self, queue, temp_db_path):
        """Test cleanup removes old events."""
        # Create queue with very short retention
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEventQueue

        short_retention_queue = CrossProcessEventQueue(
            db_path=temp_db_path,
            retention_hours=0,  # No retention
        )

        # Publish and immediately cleanup
        short_retention_queue.publish("OLD_EVENT", {}, "test")

        # Manually backdate the event
        conn = sqlite3.connect(str(temp_db_path))
        conn.execute("UPDATE events SET created_at = created_at - 3700")  # 1+ hour ago
        conn.commit()
        conn.close()

        events_deleted, _, _ = short_retention_queue.cleanup()
        assert events_deleted >= 1

        short_retention_queue.close()

    def test_cleanup_dead_subscribers(self, queue, temp_db_path):
        """Test cleanup removes dead subscribers."""
        # Subscribe
        subscriber_id = queue.subscribe("dead_process")

        # Backdate the subscriber's last poll
        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            "UPDATE subscribers SET last_poll_at = last_poll_at - 400 WHERE subscriber_id = ?",
            (subscriber_id,)
        )
        conn.commit()
        conn.close()

        _, subscribers_deleted, _ = queue.cleanup()
        assert subscribers_deleted >= 1

    def test_get_stats(self, queue):
        """Test getting queue statistics."""
        # Publish events
        queue.publish("TYPE_A", {}, "test")
        queue.publish("TYPE_B", {}, "test")
        queue.publish("TYPE_A", {}, "test")

        # Subscribe
        queue.subscribe("stats_test")

        stats = queue.get_stats()

        assert stats["total_events"] == 3
        assert stats["events_by_type"]["TYPE_A"] == 2
        assert stats["events_by_type"]["TYPE_B"] == 1
        assert len(stats["active_subscribers"]) == 1
        assert stats["retention_hours"] == 24  # Default

    def test_thread_safety(self, queue):
        """Test queue is thread-safe for concurrent operations."""
        results = {"published": 0, "errors": []}
        lock = threading.Lock()

        def publish_worker(count):
            for i in range(count):
                try:
                    queue.publish("CONCURRENT", {"worker": threading.current_thread().name}, "test")
                    with lock:
                        results["published"] += 1
                except Exception as e:
                    with lock:
                        results["errors"].append(str(e))

        threads = [
            threading.Thread(target=publish_worker, args=(10,))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["published"] == 50
        assert len(results["errors"]) == 0


class TestCrossProcessEventPoller:
    """Tests for the CrossProcessEventPoller background poller."""

    def test_poller_initialization(self, temp_db_path):
        """Test poller initializes correctly."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEventPoller

        poller = CrossProcessEventPoller(
            process_name="test_poller",
            event_types=["TYPE_A"],
            poll_interval=0.1,
            db_path=temp_db_path,
        )

        assert poller.process_name == "test_poller"
        assert poller.event_types == ["TYPE_A"]
        assert poller.poll_interval == 0.1

    def test_register_handler_specific_type(self, temp_db_path):
        """Test registering handler for specific event type."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEventPoller

        poller = CrossProcessEventPoller("test", db_path=temp_db_path)

        handler = MagicMock()
        poller.register_handler("TYPE_A", handler)

        assert "TYPE_A" in poller._handlers
        assert handler in poller._handlers["TYPE_A"]

    def test_register_global_handler(self, temp_db_path):
        """Test registering global handler (for all events)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEventPoller

        poller = CrossProcessEventPoller("test", db_path=temp_db_path)

        handler = MagicMock()
        poller.register_handler(None, handler)

        assert handler in poller._global_handlers

    def test_poller_start_stop(self, temp_db_path):
        """Test starting and stopping poller."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import CrossProcessEventPoller

        poller = CrossProcessEventPoller(
            "test",
            poll_interval=0.05,
            db_path=temp_db_path,
        )

        poller.start()
        assert poller._running is True
        assert poller._thread is not None

        poller.stop(timeout=1.0)
        assert poller._running is False

    def test_poller_calls_handlers(self, reset_global_queue):
        """Test poller calls registered handlers when events arrive."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                CrossProcessEventPoller,
                CrossProcessEventQueue,
            )

        # Use isolated temp directory to avoid SQLite locking issues
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "poller_test.db"
            queue = CrossProcessEventQueue(db_path=db_path)

            handler_called = threading.Event()
            received_events = []

            def handler(event):
                received_events.append(event)
                handler_called.set()

            poller = CrossProcessEventPoller(
                "handler_test",
                poll_interval=0.05,
                db_path=db_path,
            )
            poller.register_handler("TEST_EVENT", handler)
            poller.start()

            try:
                # Publish an event
                queue.publish("TEST_EVENT", {"test": "data"}, "test")

                # Wait for handler to be called
                handler_called.wait(timeout=2.0)

                assert len(received_events) == 1
                assert received_events[0].event_type == "TEST_EVENT"
                assert received_events[0].payload == {"test": "data"}
            finally:
                poller.stop()
                queue.close()

    def test_poller_calls_global_handlers(self, reset_global_queue):
        """Test poller calls global handlers for all events."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                CrossProcessEventPoller,
                CrossProcessEventQueue,
            )

        # Use isolated temp directory to avoid SQLite locking issues
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "global_handler_test.db"
            queue = CrossProcessEventQueue(db_path=db_path)

            received = []
            done = threading.Event()

            def global_handler(event):
                received.append(event.event_type)
                if len(received) >= 2:
                    done.set()

            poller = CrossProcessEventPoller(
                "global_test",
                poll_interval=0.05,
                db_path=db_path,
            )
            poller.register_handler(None, global_handler)
            poller.start()

            try:
                queue.publish("TYPE_A", {}, "test")
                queue.publish("TYPE_B", {}, "test")

                done.wait(timeout=2.0)

                assert "TYPE_A" in received
                assert "TYPE_B" in received
            finally:
                poller.stop()
                queue.close()


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def test_get_event_queue_singleton(self, temp_db_path, reset_global_queue):
        """Test get_event_queue returns singleton."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import get_event_queue

        q1 = get_event_queue(temp_db_path)
        q2 = get_event_queue()  # Should return same instance

        assert q1 is q2

    def test_reset_event_queue(self, temp_db_path, reset_global_queue):
        """Test reset_event_queue clears singleton."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                get_event_queue,
                reset_event_queue,
            )

        q1 = get_event_queue(temp_db_path)
        reset_event_queue()

        # Create new db path for fresh instance
        with tempfile.TemporaryDirectory() as tmpdir:
            q2 = get_event_queue(Path(tmpdir) / "new.db")
            assert q1 is not q2

    def test_publish_event_function(self, temp_db_path, reset_global_queue):
        """Test publish_event convenience function."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import publish_event

        event_id = publish_event(
            event_type="FUNC_TEST",
            payload={"data": 1},
            source="test_func",
            db_path=temp_db_path,
        )

        assert event_id > 0

    def test_subscribe_process_function(self, temp_db_path, reset_global_queue):
        """Test subscribe_process convenience function."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import subscribe_process

        subscriber_id = subscribe_process(
            process_name="func_subscriber",
            event_types=["TYPE_A"],
            db_path=temp_db_path,
        )

        assert "func_subscriber" in subscriber_id

    def test_poll_events_function(self, temp_db_path, reset_global_queue):
        """Test poll_events convenience function."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                poll_events,
                publish_event,
                subscribe_process,
            )

        # Publish event
        publish_event("POLL_TEST", {"x": 1}, "test", db_path=temp_db_path)

        # Subscribe and poll
        sub_id = subscribe_process("poll_tester", db_path=temp_db_path)
        events = poll_events(sub_id, db_path=temp_db_path)

        assert len(events) == 1
        assert events[0].event_type == "POLL_TEST"

    def test_ack_event_function(self, temp_db_path, reset_global_queue):
        """Test ack_event convenience function."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                ack_event,
                poll_events,
                publish_event,
                subscribe_process,
            )

        event_id = publish_event("ACK_TEST", {}, "test", db_path=temp_db_path)
        sub_id = subscribe_process("ack_tester", db_path=temp_db_path)

        result = ack_event(sub_id, event_id, db_path=temp_db_path)

        assert result is True

        # Verify acked
        events = poll_events(sub_id, db_path=temp_db_path)
        assert len(events) == 0

    def test_ack_events_function(self, temp_db_path, reset_global_queue):
        """Test ack_events convenience function (batch)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                ack_events,
                publish_event,
                subscribe_process,
            )

        event_ids = [
            publish_event("BATCH_ACK", {"i": i}, "test", db_path=temp_db_path)
            for i in range(3)
        ]

        sub_id = subscribe_process("batch_acker", db_path=temp_db_path)
        count = ack_events(sub_id, event_ids, db_path=temp_db_path)

        assert count == 3

    def test_bridge_to_cross_process(self, temp_db_path, reset_global_queue):
        """Test bridge_to_cross_process function."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.cross_process_events import (
                bridge_to_cross_process,
                get_event_queue,
                subscribe_process,
            )

        # Initialize queue first
        queue = get_event_queue(temp_db_path)

        # Bridge an event
        event_id = bridge_to_cross_process(
            "BRIDGED_EVENT",
            {"bridged": True},
            "bridge_test",
        )

        assert event_id > 0

        # Verify event exists
        sub_id = subscribe_process("bridge_checker", db_path=temp_db_path)
        events = queue.poll(sub_id)

        assert len(events) == 1
        assert events[0].event_type == "BRIDGED_EVENT"
        assert events[0].payload == {"bridged": True}
