"""Tests for UnifiedEventRouter module.

Tests the centralized event routing system that bridges all event buses
(data events, stage events, cross-process events) into a unified API.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.event_router import (
    CoordinatorStats,
    EventSource,
    RouterEvent,
    UnifiedEventRouter,
    get_coordinator_stats,
    get_event_coordinator,
    get_router,
    publish,
    publish_sync,
    reset_router,
    subscribe,
    unsubscribe,
)


# =============================================================================
# EventSource Enum Tests
# =============================================================================


class TestEventSource:
    """Tests for EventSource enum."""

    def test_all_sources_defined(self):
        """All expected event sources should exist."""
        assert EventSource.DATA_BUS.value == "data_bus"
        assert EventSource.STAGE_BUS.value == "stage_bus"
        assert EventSource.CROSS_PROCESS.value == "cross_process"
        assert EventSource.ROUTER.value == "router"

    def test_source_count(self):
        """Should have exactly 4 event sources."""
        assert len(EventSource) == 4


# =============================================================================
# RouterEvent Dataclass Tests
# =============================================================================


class TestRouterEvent:
    """Tests for RouterEvent dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        event = RouterEvent(event_type="TEST_EVENT")
        assert event.event_type == "TEST_EVENT"
        assert event.payload == {}
        assert event.timestamp > 0
        assert event.source == ""
        assert event.origin == EventSource.ROUTER
        assert event.data_event is None
        assert event.stage_result is None
        assert event.cross_process_event is None

    def test_with_payload(self):
        """Should store payload correctly."""
        payload = {"key": "value", "count": 42}
        event = RouterEvent(
            event_type="TEST_EVENT",
            payload=payload,
            source="test_source",
        )
        assert event.payload == payload
        assert event.source == "test_source"

    def test_with_origin(self):
        """Should set origin correctly."""
        event = RouterEvent(
            event_type="TEST_EVENT",
            origin=EventSource.DATA_BUS,
        )
        assert event.origin == EventSource.DATA_BUS


# =============================================================================
# UnifiedEventRouter Tests
# =============================================================================


class TestUnifiedEventRouter:
    """Tests for UnifiedEventRouter class."""

    @pytest.fixture
    def router(self):
        """Create a fresh router for each test."""
        reset_router()
        # Disable cross-process polling for tests
        router = UnifiedEventRouter(enable_cross_process_polling=False)
        yield router
        router.stop()
        reset_router()

    def test_initialization(self, router):
        """Should initialize correctly."""
        assert router._subscribers == {}
        assert router._global_subscribers == []
        assert router._event_history == []
        assert router._max_history == 1000

    def test_subscribe_specific_event(self, router):
        """Should subscribe to specific event type."""
        callback = MagicMock()
        router.subscribe("TEST_EVENT", callback)

        assert "TEST_EVENT" in router._subscribers
        assert callback in router._subscribers["TEST_EVENT"]

    def test_subscribe_global(self, router):
        """Should subscribe to all events with None."""
        callback = MagicMock()
        router.subscribe(None, callback)

        assert callback in router._global_subscribers

    def test_unsubscribe_specific(self, router):
        """Should unsubscribe from specific event."""
        callback = MagicMock()
        router.subscribe("TEST_EVENT", callback)

        result = router.unsubscribe("TEST_EVENT", callback)

        assert result is True
        assert callback not in router._subscribers.get("TEST_EVENT", [])

    def test_unsubscribe_global(self, router):
        """Should unsubscribe from global subscription."""
        callback = MagicMock()
        router.subscribe(None, callback)

        result = router.unsubscribe(None, callback)

        assert result is True
        assert callback not in router._global_subscribers

    def test_unsubscribe_not_found(self, router):
        """Should return False when callback not found."""
        callback = MagicMock()
        result = router.unsubscribe("NONEXISTENT", callback)
        assert result is False

    def test_register_handler_alias(self, router):
        """register_handler should be alias for subscribe."""
        callback = MagicMock()
        router.register_handler("TEST_EVENT", callback)

        assert "TEST_EVENT" in router._subscribers
        assert callback in router._subscribers["TEST_EVENT"]

    @pytest.mark.asyncio
    async def test_publish_creates_event(self, router):
        """publish should create RouterEvent with correct data."""
        event = await router.publish(
            event_type="TEST_EVENT",
            payload={"key": "value"},
            source="test_source",
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert event.event_type == "TEST_EVENT"
        assert event.payload == {"key": "value"}
        assert event.source == "test_source"
        assert event.origin == EventSource.ROUTER

    @pytest.mark.asyncio
    async def test_publish_calls_subscribers(self, router):
        """publish should invoke subscribed callbacks."""
        received_events = []

        def callback(event):
            received_events.append(event)

        router.subscribe("TEST_EVENT", callback)

        await router.publish(
            "TEST_EVENT",
            {"data": 42},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert len(received_events) == 1
        assert received_events[0].event_type == "TEST_EVENT"
        assert received_events[0].payload["data"] == 42

    @pytest.mark.asyncio
    async def test_publish_calls_global_subscribers(self, router):
        """publish should invoke global subscribers."""
        received_events = []

        def callback(event):
            received_events.append(event)

        router.subscribe(None, callback)  # Global subscription

        await router.publish(
            "ANY_EVENT",
            {},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_publish_async_callback(self, router):
        """publish should handle async callbacks."""
        received = []

        async def async_callback(event):
            await asyncio.sleep(0.01)
            received.append(event)

        router.subscribe("ASYNC_TEST", async_callback)

        await router.publish(
            "ASYNC_TEST",
            {},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_exception_handling(self, router):
        """publish should handle callback exceptions gracefully."""
        def bad_callback(event):
            raise ValueError("Test error")

        def good_callback(event):
            pass

        router.subscribe("ERROR_TEST", bad_callback)
        router.subscribe("ERROR_TEST", good_callback)

        # Should not raise
        await router.publish(
            "ERROR_TEST",
            {},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

    def test_publish_sync_no_loop(self, router):
        """publish_sync should work without event loop."""
        # Close any existing loop to test no-loop scenario
        received = []

        def callback(event):
            received.append(event)

        router.subscribe("SYNC_TEST", callback)

        result = router.publish_sync("SYNC_TEST", {})

        # Should return RouterEvent or Future
        assert result is not None

    @pytest.mark.asyncio
    async def test_dispatch_updates_history(self, router):
        """_dispatch should add events to history."""
        initial_history_len = len(router._event_history)

        await router.publish(
            "HISTORY_TEST",
            {},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert len(router._event_history) > initial_history_len

    @pytest.mark.asyncio
    async def test_dispatch_updates_metrics(self, router):
        """_dispatch should update routing metrics."""
        await router.publish(
            "METRICS_TEST",
            {},
            route_to_data_bus=False,
            route_to_stage_bus=False,
            route_to_cross_process=False,
        )

        assert router._events_routed.get("METRICS_TEST", 0) >= 1
        assert router._events_by_source.get("router", 0) >= 1

    @pytest.mark.asyncio
    async def test_history_limit(self, router):
        """History should be capped at max_history."""
        router._max_history = 10

        for i in range(20):
            await router.publish(
                f"EVENT_{i}",
                {},
                route_to_data_bus=False,
                route_to_stage_bus=False,
                route_to_cross_process=False,
            )

        assert len(router._event_history) <= 10

    def test_dispatch_sync(self, router):
        """_dispatch_sync should work without async."""
        event = RouterEvent(
            event_type="SYNC_DISPATCH",
            payload={},
        )

        received = []

        def callback(e):
            received.append(e)

        router.subscribe("SYNC_DISPATCH", callback)
        router._dispatch_sync(event)

        assert len(received) == 1

    def test_dispatch_sync_skips_async_callbacks(self, router):
        """_dispatch_sync should skip async callbacks with warning."""
        event = RouterEvent(
            event_type="ASYNC_SKIP",
            payload={},
        )

        async def async_callback(e):
            pass

        router.subscribe("ASYNC_SKIP", async_callback)

        # Should not raise, just log warning
        router._dispatch_sync(event)

    def test_get_history_unfiltered(self, router):
        """get_history should return all events."""
        # Add some events directly to history
        for i in range(5):
            router._event_history.append(RouterEvent(
                event_type=f"EVENT_{i}",
                payload={"index": i},
            ))

        history = router.get_history()
        assert len(history) == 5

    def test_get_history_by_type(self, router):
        """get_history should filter by event type."""
        router._event_history.append(RouterEvent(event_type="TYPE_A"))
        router._event_history.append(RouterEvent(event_type="TYPE_B"))
        router._event_history.append(RouterEvent(event_type="TYPE_A"))

        history = router.get_history(event_type="TYPE_A")
        assert len(history) == 2

    def test_get_history_by_origin(self, router):
        """get_history should filter by origin."""
        router._event_history.append(RouterEvent(
            event_type="E1", origin=EventSource.DATA_BUS
        ))
        router._event_history.append(RouterEvent(
            event_type="E2", origin=EventSource.STAGE_BUS
        ))
        router._event_history.append(RouterEvent(
            event_type="E3", origin=EventSource.DATA_BUS
        ))

        history = router.get_history(origin=EventSource.DATA_BUS)
        assert len(history) == 2

    def test_get_history_since(self, router):
        """get_history should filter by timestamp."""
        old_time = time.time() - 100
        new_time = time.time()

        router._event_history.append(RouterEvent(
            event_type="OLD", timestamp=old_time
        ))
        router._event_history.append(RouterEvent(
            event_type="NEW", timestamp=new_time
        ))

        history = router.get_history(since=new_time - 1)
        assert len(history) == 1
        assert history[0].event_type == "NEW"

    def test_get_history_limit(self, router):
        """get_history should respect limit."""
        for i in range(10):
            router._event_history.append(RouterEvent(event_type=f"E{i}"))

        history = router.get_history(limit=5)
        assert len(history) == 5

    def test_get_stats(self, router):
        """get_stats should return comprehensive statistics."""
        # Add some data
        router._events_routed["TEST"] = 5
        router._events_by_source["router"] = 5
        router._global_subscribers.append(lambda x: None)
        router._subscribers["TEST"] = [lambda x: None, lambda x: None]

        stats = router.get_stats()

        assert stats["total_events_routed"] == 5
        assert stats["global_subscriber_count"] == 1
        assert stats["subscriber_count"] == 2
        assert "has_data_events" in stats
        assert "has_stage_events" in stats
        assert "has_cross_process" in stats

    def test_stop(self, router):
        """stop should clean up poller."""
        mock_poller = MagicMock()
        router._cp_poller = mock_poller

        router.stop()

        mock_poller.stop.assert_called_once()
        assert router._cp_poller is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in event router."""

    @pytest.fixture
    def router(self):
        reset_router()
        router = UnifiedEventRouter(enable_cross_process_polling=False)
        yield router
        router.stop()
        reset_router()

    def test_handle_dispatch_task_error_logs_exception(self, router):
        """_handle_dispatch_task_error should log exceptions."""
        task = MagicMock()
        task.exception.return_value = ValueError("Test error")

        # Should not raise
        router._handle_dispatch_task_error(task)

        assert router._events_routed.get("__dispatch_failures__", 0) >= 1

    def test_handle_dispatch_task_error_cancelled(self, router):
        """_handle_dispatch_task_error should ignore cancelled tasks."""
        task = MagicMock()
        task.exception.side_effect = asyncio.CancelledError()

        # Should not raise or track failure
        router._handle_dispatch_task_error(task)

        assert router._events_routed.get("__dispatch_failures__", 0) == 0

    def test_handle_dispatch_task_error_no_exception(self, router):
        """_handle_dispatch_task_error should handle successful tasks."""
        task = MagicMock()
        task.exception.return_value = None

        # Should not raise or track failure
        router._handle_dispatch_task_error(task)

        assert router._events_routed.get("__dispatch_failures__", 0) == 0


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_between_tests(self):
        reset_router()
        yield
        reset_router()

    def test_get_router_singleton(self):
        """get_router should return singleton."""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_reset_router(self):
        """reset_router should create new instance."""
        router1 = get_router()
        reset_router()
        router2 = get_router()
        assert router1 is not router2

    @pytest.mark.asyncio
    async def test_publish_convenience(self):
        """publish() should use global router."""
        received = []

        def callback(event):
            received.append(event)

        subscribe("CONVENIENCE_TEST", callback)

        # Mock out external buses
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await publish("CONVENIENCE_TEST", {"test": True})

        assert len(received) == 1

    def test_subscribe_convenience(self):
        """subscribe() should use global router."""
        callback = MagicMock()
        subscribe("SUB_TEST", callback)

        router = get_router()
        assert callback in router._subscribers.get("SUB_TEST", [])

    def test_unsubscribe_convenience(self):
        """unsubscribe() should use global router."""
        callback = MagicMock()
        subscribe("UNSUB_TEST", callback)

        result = unsubscribe("UNSUB_TEST", callback)
        assert result is True


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with unified_event_coordinator."""

    @pytest.fixture(autouse=True)
    def reset_between_tests(self):
        reset_router()
        yield
        reset_router()

    def test_get_event_coordinator_alias(self):
        """get_event_coordinator should return router."""
        coordinator = get_event_coordinator()
        router = get_router()
        assert coordinator is router

    def test_coordinator_stats_dataclass(self):
        """CoordinatorStats should have expected fields."""
        stats = CoordinatorStats()
        assert stats.events_bridged_data_to_cross == 0
        assert stats.events_bridged_stage_to_cross == 0
        assert stats.events_bridged_cross_to_data == 0
        assert stats.events_dropped == 0
        assert stats.is_running is False

    def test_get_coordinator_stats(self):
        """get_coordinator_stats should return CoordinatorStats."""
        stats = get_coordinator_stats()
        assert isinstance(stats, CoordinatorStats)


# =============================================================================
# Event Emitter Tests
# =============================================================================


class TestEventEmitters:
    """Tests for convenience event emitter functions."""

    @pytest.fixture(autouse=True)
    def reset_between_tests(self):
        reset_router()
        yield
        reset_router()

    @pytest.mark.asyncio
    async def test_emit_training_started(self):
        """emit_training_started should publish correct event."""
        from app.coordination.event_router import emit_training_started

        received = []
        subscribe("TRAINING_STARTED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_training_started("hex8_2p", "node1")

        assert len(received) == 1
        assert received[0].payload["config_key"] == "hex8_2p"
        assert received[0].payload["node_name"] == "node1"

    @pytest.mark.asyncio
    async def test_emit_training_completed(self):
        """emit_training_completed should publish correct event."""
        from app.coordination.event_router import emit_training_completed

        received = []
        subscribe("TRAINING_COMPLETED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_training_completed("hex8_2p", "model_123", val_loss=0.5, epochs=10)

        assert len(received) == 1
        assert received[0].payload["config_key"] == "hex8_2p"
        assert received[0].payload["model_id"] == "model_123"
        assert received[0].payload["val_loss"] == 0.5
        assert received[0].payload["epochs"] == 10

    @pytest.mark.asyncio
    async def test_emit_training_failed(self):
        """emit_training_failed should publish correct event."""
        from app.coordination.event_router import emit_training_failed

        received = []
        subscribe("TRAINING_FAILED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_training_failed("hex8_2p", "OOM error")

        assert len(received) == 1
        assert received[0].payload["config_key"] == "hex8_2p"
        assert received[0].payload["error"] == "OOM error"

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed(self):
        """emit_evaluation_completed should publish correct event."""
        from app.coordination.event_router import emit_evaluation_completed

        received = []
        subscribe("EVALUATION_COMPLETED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_evaluation_completed("model_123", elo=1650, win_rate=0.75, games_played=100)

        assert len(received) == 1
        assert received[0].payload["model_id"] == "model_123"
        assert received[0].payload["elo"] == 1650
        assert received[0].payload["win_rate"] == 0.75

    @pytest.mark.asyncio
    async def test_emit_sync_completed(self):
        """emit_sync_completed should publish correct event."""
        from app.coordination.event_router import emit_sync_completed

        received = []
        subscribe("DATA_SYNC_COMPLETED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_sync_completed("rsync", files_synced=50, bytes_transferred=1024000)

        assert len(received) == 1
        assert received[0].payload["sync_type"] == "rsync"
        assert received[0].payload["files_synced"] == 50

    @pytest.mark.asyncio
    async def test_emit_model_promoted(self):
        """emit_model_promoted should publish correct event."""
        from app.coordination.event_router import emit_model_promoted

        received = []
        subscribe("MODEL_PROMOTED", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_model_promoted("model_123", tier="production", elo=1700)

        assert len(received) == 1
        assert received[0].payload["model_id"] == "model_123"
        assert received[0].payload["tier"] == "production"

    @pytest.mark.asyncio
    async def test_emit_selfplay_batch_completed(self):
        """emit_selfplay_batch_completed should publish correct event."""
        from app.coordination.event_router import emit_selfplay_batch_completed

        received = []
        subscribe("SELFPLAY_BATCH_COMPLETE", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await emit_selfplay_batch_completed("hex8_2p", games_generated=100, duration_seconds=60.5)

        assert len(received) == 1
        assert received[0].payload["config_key"] == "hex8_2p"
        assert received[0].payload["games_generated"] == 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for event router."""

    @pytest.fixture(autouse=True)
    def reset_between_tests(self):
        reset_router()
        yield
        reset_router()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self):
        """Multiple subscribers should all receive event."""
        received1 = []
        received2 = []
        received3 = []

        subscribe("MULTI_SUB", lambda e: received1.append(e))
        subscribe("MULTI_SUB", lambda e: received2.append(e))
        subscribe(None, lambda e: received3.append(e))  # Global

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await publish("MULTI_SUB", {"test": True})

        assert len(received1) == 1
        assert len(received2) == 1
        assert len(received3) == 1

    @pytest.mark.asyncio
    async def test_event_type_normalization(self):
        """Event types should be normalized to strings."""
        received = []

        # Subscribe with string
        subscribe("test_event", lambda e: received.append(e))

        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            # Publish with string
            await publish("test_event", {})

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_stats_accumulate(self):
        """Stats should accumulate across multiple events with different payloads."""
        # Note: Content-based deduplication means identical events are deduplicated.
        # Use different payloads to test accumulation of distinct events.
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            for i in range(5):
                await publish("STATS_TEST", {"index": i})  # Different payload each time

        router = get_router()
        assert router._events_routed.get("STATS_TEST", 0) == 5

    @pytest.mark.asyncio
    async def test_history_contains_all_events(self):
        """History should contain events from all sources."""
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            await publish("EVENT_A", {"a": 1})
            await publish("EVENT_B", {"b": 2})
            await publish("EVENT_C", {"c": 3})

        router = get_router()
        history = router.get_history()

        event_types = [e.event_type for e in history]
        assert "EVENT_A" in event_types
        assert "EVENT_B" in event_types
        assert "EVENT_C" in event_types


class TestDeduplication:
    """Tests for event deduplication (December 2025)."""

    @pytest.fixture(autouse=True)
    def reset_router_for_test(self):
        """Reset router before each test."""
        reset_router()
        yield
        reset_router()

    @pytest.mark.asyncio
    async def test_content_based_dedup_identical_events(self):
        """Identical events should be deduplicated based on content hash."""
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            # Publish same event 3 times
            for _ in range(3):
                await publish("DEDUP_TEST", {"key": "same_value"})

        router = get_router()
        # Only 1 should be processed (others deduplicated by content hash)
        assert router._events_routed.get("DEDUP_TEST", 0) == 1
        assert router._content_duplicates_prevented == 2

    @pytest.mark.asyncio
    async def test_content_based_dedup_different_payloads(self):
        """Events with different payloads should NOT be deduplicated."""
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            # Publish events with different payloads
            await publish("DEDUP_TEST_2", {"key": "value_1"})
            await publish("DEDUP_TEST_2", {"key": "value_2"})
            await publish("DEDUP_TEST_2", {"key": "value_3"})

        router = get_router()
        # All 3 should be processed (different content hashes)
        assert router._events_routed.get("DEDUP_TEST_2", 0) == 3
        assert router._content_duplicates_prevented == 0

    @pytest.mark.asyncio
    async def test_content_hash_ignores_timestamps(self):
        """Content hash should ignore timestamp fields for deduplication."""
        with patch("app.coordination.event_router.HAS_DATA_EVENTS", False), \
             patch("app.coordination.event_router.HAS_STAGE_EVENTS", False), \
             patch("app.coordination.event_router.HAS_CROSS_PROCESS", False):
            # Same logical event but different timestamps
            await publish("DEDUP_TS_TEST", {"key": "value", "timestamp": "2025-01-01"})
            await publish("DEDUP_TS_TEST", {"key": "value", "timestamp": "2025-01-02"})
            await publish("DEDUP_TS_TEST", {"key": "value", "created_at": 12345})

        router = get_router()
        # All should be deduplicated (timestamps ignored in content hash)
        assert router._events_routed.get("DEDUP_TS_TEST", 0) == 1
        assert router._content_duplicates_prevented == 2

    @pytest.mark.asyncio
    async def test_stats_include_dedup_metrics(self):
        """Stats should include deduplication metrics."""
        router = get_router()
        stats = router.get_stats()

        assert "duplicates_prevented" in stats
        assert "content_duplicates_prevented" in stats
        assert "total_duplicates_prevented" in stats
        assert "seen_events_count" in stats
        assert "seen_content_hashes_count" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
