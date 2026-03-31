"""Tests for the Unified Event Router.

Tests the event routing layer that consolidates:
- EventBus (data_events.py)
- StageEventBus (stage_events.py)
- CrossProcessEventQueue (cross_process_events.py)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module under test
from app.coordination.event_router import (
    EventSource,
    RouterEvent,
    UnifiedEventRouter,
    get_router,
    publish,
    publish_sync,
    reset_router,
    safe_emit_event,
    subscribe,
    unsubscribe,
)


@pytest.fixture
def router():
    """Create a fresh router for each test."""
    reset_router()
    return get_router()


@pytest.fixture
def received_events():
    """Track received events in tests."""
    return []


class TestUnifiedEventRouter:
    """Tests for UnifiedEventRouter class."""

    def test_singleton_pattern(self):
        """Router should be a singleton."""
        reset_router()
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        """Reset should create a new router instance."""
        r1 = get_router()
        reset_router()
        r2 = get_router()
        assert r1 is not r2

    def test_subscribe_callback(self, router, received_events):
        """Subscribe should register callbacks for event types."""
        def callback(event):
            received_events.append(event)

        router.subscribe("test_event", callback)
        assert "test_event" in router._subscribers
        assert callback in router._subscribers["test_event"]

    def test_subscribe_global_callback(self, router, received_events):
        """Subscribe with None should register global callback."""
        def callback(event):
            received_events.append(event)

        router.subscribe(None, callback)
        assert callback in router._global_subscribers

    def test_unsubscribe_removes_callback(self, router):
        """Unsubscribe should remove callbacks."""
        def callback(event):
            pass

        router.subscribe("test_event", callback)
        result = router.unsubscribe("test_event", callback)

        assert result is True
        assert callback not in router._subscribers.get("test_event", [])

    def test_unsubscribe_nonexistent_returns_false(self, router):
        """Unsubscribe for non-registered callback returns False."""
        def callback(event):
            pass

        result = router.unsubscribe("test_event", callback)
        assert result is False


class TestEventPublishing:
    """Tests for event publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_creates_router_event(self, router, received_events):
        """Publish should create and dispatch RouterEvent."""
        def callback(event):
            received_events.append(event)

        router.subscribe("test_event", callback)

        await router.publish(
            event_type="test_event",
            payload={"key": "value"},
            source="test_source",
        )

        assert len(received_events) == 1
        event = received_events[0]
        assert isinstance(event, RouterEvent)
        assert event.event_type == "test_event"
        assert event.payload == {"key": "value"}
        assert event.source == "test_source"

    @pytest.mark.asyncio
    async def test_publish_invokes_all_subscribers(self, router, received_events):
        """Publish should invoke all subscribers for event type."""
        def callback1(event):
            received_events.append(("cb1", event))

        def callback2(event):
            received_events.append(("cb2", event))

        router.subscribe("test_event", callback1)
        router.subscribe("test_event", callback2)

        await router.publish("test_event", {"data": 1}, "test")

        assert len(received_events) == 2
        assert received_events[0][0] == "cb1"
        assert received_events[1][0] == "cb2"

    @pytest.mark.asyncio
    async def test_publish_invokes_global_subscribers(self, router, received_events):
        """Publish should invoke global subscribers for any event."""
        def global_callback(event):
            received_events.append(event)

        router.subscribe(None, global_callback)

        await router.publish("any_event", {}, "test")
        await router.publish("other_event", {}, "test")

        # Should receive at least these 2 events (may receive more from cross-process)
        event_types = [e.event_type for e in received_events]
        assert "any_event" in event_types
        assert "other_event" in event_types

    @pytest.mark.asyncio
    async def test_publish_async_callback(self, router, received_events):
        """Publish should handle async callbacks."""
        async def async_callback(event):
            await asyncio.sleep(0.01)
            received_events.append(event)

        router.subscribe("test_event", async_callback)

        await router.publish("test_event", {}, "test")

        assert len(received_events) == 1

    def test_publish_sync_creates_event(self, router, received_events):
        """Synchronous publish should work."""
        def callback(event):
            received_events.append(event)

        router.subscribe("test_event", callback)
        router.publish_sync("test_event", {"sync": True}, "test")

        assert len(received_events) == 1
        assert received_events[0].payload == {"sync": True}

    @pytest.mark.asyncio
    async def test_safe_emit_event_from_worker_thread_uses_router_loop(self, router):
        """safe_emit_event should deliver from sync worker threads into the live router loop."""
        delivered = asyncio.Event()
        received_events = []

        async def callback(event):
            received_events.append(event)
            delivered.set()

        router.subscribe("threaded_event", callback)

        result = await asyncio.to_thread(
            safe_emit_event,
            "threaded_event",
            {"thread": True},
            "thread_test",
        )

        assert result is True
        await asyncio.wait_for(delivered.wait(), timeout=1.0)
        assert len(received_events) == 1
        assert received_events[0].payload == {"thread": True}
        assert received_events[0].source == "thread_test"


class TestEventHistory:
    """Tests for event history tracking."""

    @pytest.mark.asyncio
    async def test_events_tracked_in_history(self, router):
        """Published events should be tracked in history."""
        await router.publish("event1", {}, "test")
        await router.publish("event2", {}, "test")

        assert len(router._event_history) == 2

    @pytest.mark.asyncio
    async def test_history_respects_max_size(self, router):
        """History should not exceed max size."""
        router._max_history = 5

        for i in range(10):
            await router.publish(f"event_{i}", {}, "test")

        assert len(router._event_history) <= 5

    def test_get_event_history(self, router):
        """Should be able to retrieve event history via _event_history."""
        router.publish_sync("event1", {}, "test")
        router.publish_sync("event2", {}, "test")

        # Access history directly (no public method)
        history = router._event_history
        assert len(history) >= 2  # At least these 2 events


class TestEventMetrics:
    """Tests for event routing metrics."""

    @pytest.mark.asyncio
    async def test_events_routed_counter(self, router):
        """Should count events routed by type."""
        await router.publish("type_a", {}, "test")
        await router.publish("type_a", {}, "test")
        await router.publish("type_b", {}, "test")

        assert router._events_routed.get("type_a", 0) == 2
        assert router._events_routed.get("type_b", 0) == 1

    @pytest.mark.asyncio
    async def test_events_by_source_counter(self, router):
        """Should count events by source."""
        await router.publish("event", {}, "source1")
        await router.publish("event", {}, "source2")
        await router.publish("event", {}, "source1")

        # Source tracking uses EventSource enum values
        stats = router.get_stats()
        assert "events_by_source" in stats

    def test_get_stats(self, router):
        """Should return comprehensive stats."""
        router.publish_sync("event", {}, "test")

        stats = router.get_stats()
        assert "events_routed_by_type" in stats
        assert "events_by_source" in stats
        assert "subscriber_count" in stats
        assert "history_size" in stats


class TestEventSourceEnum:
    """Tests for EventSource enum."""

    def test_event_source_values(self):
        """EventSource enum should have expected values."""
        assert EventSource.DATA_BUS.value == "data_bus"
        assert EventSource.STAGE_BUS.value == "stage_bus"
        assert EventSource.CROSS_PROCESS.value == "cross_process"
        assert EventSource.ROUTER.value == "router"


class TestRouterEvent:
    """Tests for RouterEvent dataclass."""

    def test_router_event_creation(self):
        """RouterEvent should be created with required fields."""
        event = RouterEvent(
            event_type="test",
            payload={"key": "value"},
            source="test_source",
            origin=EventSource.ROUTER,
        )

        assert event.event_type == "test"
        assert event.payload == {"key": "value"}
        assert event.source == "test_source"
        assert event.origin == EventSource.ROUTER
        assert event.timestamp > 0

    def test_router_event_default_origin(self):
        """RouterEvent should default to ROUTER origin."""
        event = RouterEvent(
            event_type="test",
            payload={},
            source="test",
        )
        assert event.origin == EventSource.ROUTER


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_publish_function(self, received_events):
        """Module-level publish should work."""
        reset_router()

        def callback(event):
            received_events.append(event)

        subscribe("test", callback)
        await publish("test", {"module": "level"}, "test")

        assert len(received_events) == 1

    def test_publish_sync_function(self, received_events):
        """Module-level publish_sync should work."""
        reset_router()

        def callback(event):
            received_events.append(event)

        subscribe("test", callback)
        publish_sync("test", {"sync": True}, "test")

        assert len(received_events) == 1

    def test_subscribe_function(self):
        """Module-level subscribe should work."""
        reset_router()

        def callback(event):
            pass

        subscribe("test", callback)
        router = get_router()

        assert callback in router._subscribers.get("test", [])

    def test_unsubscribe_function(self):
        """Module-level unsubscribe should work."""
        reset_router()

        def callback(event):
            pass

        subscribe("test", callback)
        result = unsubscribe("test", callback)

        assert result is True


class TestCallbackErrorHandling:
    """Tests for error handling in callbacks."""

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_stop_others(self, router, received_events):
        """Error in one callback shouldn't stop other callbacks."""
        def bad_callback(event):
            raise ValueError("Intentional error")

        def good_callback(event):
            received_events.append(event)

        router.subscribe("test", bad_callback)
        router.subscribe("test", good_callback)

        # Should not raise, and good_callback should still be called
        await router.publish("test", {}, "test")

        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_async_callback_error_handling(self, router, received_events):
        """Async callback errors should be handled gracefully."""
        async def bad_async_callback(event):
            raise RuntimeError("Async error")

        async def good_async_callback(event):
            received_events.append(event)

        router.subscribe("test", bad_async_callback)
        router.subscribe("test", good_async_callback)

        await router.publish("test", {}, "test")

        assert len(received_events) == 1


class TestIntegrationWithDataEvents:
    """Tests for integration with data_events module."""

    @pytest.mark.asyncio
    async def test_data_event_type_subscription(self, router, received_events):
        """Should be able to subscribe to DataEventType values."""
        try:
            from app.distributed.data_events import DataEventType

            def callback(event):
                received_events.append(event)

            router.subscribe(DataEventType.TRAINING_COMPLETED, callback)

            await router.publish(
                DataEventType.TRAINING_COMPLETED,
                {"config": "test"},
                "test",
            )

            assert len(received_events) == 1
        except ImportError:
            pytest.skip("data_events not available")


class TestIntegrationWithStageEvents:
    """Tests for integration with stage_events module."""

    @pytest.mark.asyncio
    async def test_stage_event_subscription(self, router, received_events):
        """Should be able to subscribe to StageEvent values."""
        try:
            from app.coordination.stage_events import StageEvent

            def callback(event):
                received_events.append(event)

            router.subscribe(StageEvent.SELFPLAY_COMPLETE, callback)

            await router.publish(
                StageEvent.SELFPLAY_COMPLETE,
                {"games": 100},
                "test",
            )

            # May receive multiple events due to bidirectional routing
            # (router -> stage bus -> router). Check that we received at least one.
            assert len(received_events) >= 1
            # Router normalizes event types to canonical UPPERCASE forms.
            event_types = [e.event_type for e in received_events]
            assert "SELFPLAY_COMPLETE" in event_types
        except ImportError:
            pytest.skip("stage_events not available")
