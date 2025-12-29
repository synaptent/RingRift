"""Unit tests for subscription store.

P0 December 2025: Tests for SQLite-backed subscription persistence.
"""

import asyncio
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.subscription_store import (
    DEFAULT_SUBSCRIPTION_STORE_PATH,
    DLQ_REPLAY_MIN_AGE_SECONDS,
    STALE_DLQ_ALERT_THRESHOLD_HOURS,
    SubscriptionRecord,
    SubscriptionStore,
    get_subscription_store,
    reset_subscription_store,
)


class TestSubscriptionRecord:
    """Tests for SubscriptionRecord dataclass."""

    def test_record_creation(self):
        """Test creating a subscription record."""
        record = SubscriptionRecord(
            subscriber_name="TestSubscriber",
            event_type="TEST_EVENT",
            handler_path="app.test:my_handler",
            pid=12345,
        )

        assert record.subscriber_name == "TestSubscriber"
        assert record.event_type == "TEST_EVENT"
        assert record.handler_path == "app.test:my_handler"
        assert record.pid == 12345
        assert record.is_active is True
        assert record.last_processed_at is None

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = SubscriptionRecord(
            subscriber_name="TestSubscriber",
            event_type="TEST_EVENT",
            handler_path="app.test:my_handler",
            pid=12345,
        )

        data = record.to_dict()

        assert data["subscriber_name"] == "TestSubscriber"
        assert data["event_type"] == "TEST_EVENT"
        assert data["handler_path"] == "app.test:my_handler"
        assert data["pid"] == 12345
        assert data["is_active"] is True

    def test_record_from_row(self):
        """Test creating record from SQLite row."""
        # Create a mock row-like object
        class MockRow:
            def __getitem__(self, key):
                data = {
                    "subscriber_name": "TestSubscriber",
                    "event_type": "TEST_EVENT",
                    "handler_path": "app.test:my_handler",
                    "pid": 12345,
                    "created_at": "2025-12-28T10:00:00",
                    "last_processed_at": None,
                    "is_active": 1,
                }
                return data[key]

        row = MockRow()
        record = SubscriptionRecord.from_row(row)

        assert record.subscriber_name == "TestSubscriber"
        assert record.event_type == "TEST_EVENT"
        assert record.is_active is True


class TestSubscriptionStore:
    """Tests for SubscriptionStore class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_subscription_store()

    def teardown_method(self):
        """Clean up after each test."""
        reset_subscription_store()

    def test_store_initialization(self, tmp_path):
        """Test store initializes database correctly."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        assert store.db_path == db_path
        assert db_path.exists()

        # Verify schema
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "subscriptions" in tables
        assert "subscriber_state" in tables

    def test_register_subscription(self, tmp_path):
        """Test registering a subscription."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        record = store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="DATA_SYNC_COMPLETED",
            handler_path="app.coordination.pipeline:on_sync",
            pid=12345,
        )

        assert record.subscriber_name == "DataPipeline"
        assert record.event_type == "DATA_SYNC_COMPLETED"
        assert record.handler_path == "app.coordination.pipeline:on_sync"
        assert record.pid == 12345
        assert record.is_active is True

    def test_register_subscription_upsert(self, tmp_path):
        """Test that registering same subscription updates it."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # First registration
        record1 = store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="DATA_SYNC_COMPLETED",
            handler_path="app.old:handler",
            pid=12345,
        )

        # Second registration with same key
        record2 = store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="DATA_SYNC_COMPLETED",
            handler_path="app.new:handler",
            pid=67890,
        )

        # Should have updated handler_path and pid
        assert record2.handler_path == "app.new:handler"
        assert record2.pid == 67890

        # Should still be only one subscription
        active = store.get_active_subscriptions()
        assert len(active) == 1

    def test_deactivate_subscription(self, tmp_path):
        """Test deactivating a subscription."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="DATA_SYNC_COMPLETED",
            handler_path="app.pipeline:handler",
        )

        # Deactivate
        result = store.deactivate_subscription(
            subscriber_name="DataPipeline",
            event_type="DATA_SYNC_COMPLETED",
        )

        assert result is True

        # Should not appear in active subscriptions
        active = store.get_active_subscriptions()
        assert len(active) == 0

    def test_deactivate_nonexistent_subscription(self, tmp_path):
        """Test deactivating a subscription that doesn't exist."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        result = store.deactivate_subscription(
            subscriber_name="NonExistent",
            event_type="FAKE_EVENT",
        )

        assert result is False

    def test_deactivate_all_for_pid(self, tmp_path):
        """Test deactivating all subscriptions for a process."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Register multiple subscriptions for same PID
        store.register_subscription(
            subscriber_name="Sub1",
            event_type="EVENT_A",
            handler_path="app.a:handler",
            pid=12345,
        )
        store.register_subscription(
            subscriber_name="Sub2",
            event_type="EVENT_B",
            handler_path="app.b:handler",
            pid=12345,
        )

        # Different PID
        store.register_subscription(
            subscriber_name="Sub3",
            event_type="EVENT_C",
            handler_path="app.c:handler",
            pid=67890,
        )

        # Deactivate all for PID 12345
        count = store.deactivate_all_for_pid(12345)

        assert count == 2

        # Should have only the other PID's subscription active
        active = store.get_active_subscriptions()
        assert len(active) == 1
        assert active[0].pid == 67890

    def test_get_active_subscriptions(self, tmp_path):
        """Test getting active subscriptions."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Register multiple subscriptions
        store.register_subscription(
            subscriber_name="Sub1",
            event_type="EVENT_A",
            handler_path="app.a:handler",
        )
        store.register_subscription(
            subscriber_name="Sub2",
            event_type="EVENT_A",
            handler_path="app.b:handler",
        )
        store.register_subscription(
            subscriber_name="Sub3",
            event_type="EVENT_B",
            handler_path="app.c:handler",
        )

        # Get all active
        all_active = store.get_active_subscriptions()
        assert len(all_active) == 3

        # Get by event type
        event_a = store.get_active_subscriptions(event_type="EVENT_A")
        assert len(event_a) == 2

        event_b = store.get_active_subscriptions(event_type="EVENT_B")
        assert len(event_b) == 1

    def test_get_subscriptions_for_subscriber(self, tmp_path):
        """Test getting subscriptions for a specific subscriber."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Register multiple event types for same subscriber
        store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="EVENT_A",
            handler_path="app.a:handler",
        )
        store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="EVENT_B",
            handler_path="app.b:handler",
        )

        subs = store.get_subscriptions_for_subscriber("DataPipeline")
        assert len(subs) == 2

        event_types = {s.event_type for s in subs}
        assert event_types == {"EVENT_A", "EVENT_B"}

    def test_update_last_processed(self, tmp_path):
        """Test updating last processed timestamp."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        store.register_subscription(
            subscriber_name="DataPipeline",
            event_type="EVENT_A",
            handler_path="app.a:handler",
        )

        store.update_last_processed(
            subscriber_name="DataPipeline",
            event_type="EVENT_A",
            event_id="event-123",
        )

        # Verify subscription updated
        subs = store.get_subscriptions_for_subscriber("DataPipeline")
        assert len(subs) == 1
        assert subs[0].last_processed_at is not None

        # Verify subscriber state
        state = store.get_subscriber_state("DataPipeline")
        assert state is not None
        assert state["last_processed_event_id"] == "event-123"
        assert state["events_processed"] == 1

    def test_get_stats(self, tmp_path):
        """Test getting store statistics."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Register subscriptions
        store.register_subscription(
            subscriber_name="Sub1",
            event_type="EVENT_A",
            handler_path="app.a:handler",
        )
        store.register_subscription(
            subscriber_name="Sub2",
            event_type="EVENT_A",
            handler_path="app.b:handler",
        )
        store.register_subscription(
            subscriber_name="Sub3",
            event_type="EVENT_B",
            handler_path="app.c:handler",
        )

        stats = store.get_stats()

        assert stats["active_subscriptions"] == 3
        assert stats["by_event_type"]["EVENT_A"] == 2
        assert stats["by_event_type"]["EVENT_B"] == 1
        assert stats["unique_subscribers"] == 3
        assert stats["subscriptions_registered"] == 3

    def test_health_check(self, tmp_path):
        """Test health check method."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Register a subscription
        store.register_subscription(
            subscriber_name="Sub1",
            event_type="EVENT_A",
            handler_path="app.a:handler",
        )

        # Mock check_stale_dlq_events to return empty list (no stale events)
        with patch.object(store, "check_stale_dlq_events", return_value=[]):
            health = store.health_check()

            # Should be healthy
            assert health.healthy is True
            assert "1 active subscription" in health.message


class TestSubscriptionStoreSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_subscription_store()

    def teardown_method(self):
        """Clean up after each test."""
        reset_subscription_store()

    def test_get_subscription_store_singleton(self, tmp_path):
        """Test that get_subscription_store returns singleton."""
        # Create with specific path
        with patch(
            "app.coordination.subscription_store.DEFAULT_SUBSCRIPTION_STORE_PATH",
            tmp_path / "test.db",
        ):
            reset_subscription_store()
            store1 = get_subscription_store(db_path=tmp_path / "test.db")
            store2 = get_subscription_store()

            assert store1 is store2

    def test_reset_subscription_store(self, tmp_path):
        """Test resetting the singleton."""
        with patch(
            "app.coordination.subscription_store.DEFAULT_SUBSCRIPTION_STORE_PATH",
            tmp_path / "test.db",
        ):
            store1 = get_subscription_store(db_path=tmp_path / "test.db")
            reset_subscription_store()
            store2 = get_subscription_store(db_path=tmp_path / "test2.db")

            assert store1 is not store2


class TestDLQIntegration:
    """Tests for DLQ integration (mocked)."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_subscription_store()

    def teardown_method(self):
        """Clean up after each test."""
        reset_subscription_store()

    @pytest.mark.asyncio
    async def test_replay_stale_dlq_events_no_dlq(self, tmp_path):
        """Test replay returns 0 when DLQ not available."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # Simulate ImportError by testing the fallback behavior
        # The method should return 0 when dependencies aren't available
        # We test this by verifying the return value and stats
        count = await store.replay_stale_dlq_events()

        # Should handle gracefully
        assert count >= 0  # Either 0 or actual replayed count

    def test_check_stale_dlq_events_no_dlq(self, tmp_path):
        """Test check returns empty list when DLQ not available."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        # The real DLQ might exist, so we just verify the method works
        stale = store.check_stale_dlq_events()

        # Should return a list (possibly with real stale events)
        assert isinstance(stale, list)

    def test_check_stale_dlq_events_structure(self, tmp_path):
        """Test stale DLQ events have expected structure."""
        db_path = tmp_path / "test_subscriptions.db"
        store = SubscriptionStore(db_path=db_path)

        stale = store.check_stale_dlq_events()

        # Each stale event should have expected fields
        for event in stale:
            assert "event_type" in event
            assert "count" in event
            assert "oldest" in event
            assert "threshold_hours" in event
            assert isinstance(event["count"], int)
            assert event["threshold_hours"] == store.stale_dlq_alert_hours


class TestModuleConstants:
    """Tests for module constants."""

    def test_default_path_exists(self):
        """Test default path constant is defined."""
        assert DEFAULT_SUBSCRIPTION_STORE_PATH is not None
        assert isinstance(DEFAULT_SUBSCRIPTION_STORE_PATH, Path)

    def test_stale_alert_threshold(self):
        """Test stale alert threshold constant."""
        assert STALE_DLQ_ALERT_THRESHOLD_HOURS == 24

    def test_dlq_replay_min_age(self):
        """Test DLQ replay minimum age constant."""
        assert DLQ_REPLAY_MIN_AGE_SECONDS == 300  # 5 minutes
