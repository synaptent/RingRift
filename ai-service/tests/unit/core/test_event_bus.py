#!/usr/bin/env python3
"""Unit tests for app.core.event_bus module (December 2025).

Tests the unified event bus system:
- Event base class and common event types
- EventFilter for topic-based filtering
- Subscription management
- EventBus pub/sub functionality
- Event history and replay
- Module-level convenience functions

These tests cover both sync and async handlers.
"""

import asyncio
import pytest
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

from app.core.event_bus import (
    Event,
    SystemEvent,
    MetricEvent,
    LifecycleEvent,
    ErrorEvent,
    EventFilter,
    EventBus,
    Subscription,
    get_event_bus,
    reset_event_bus,
    subscribe,
    publish,
    publish_sync,
)


class TestEvent:
    """Tests for Event base class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(topic="test.topic")
        assert event.topic == "test.topic"
        assert event.timestamp > 0
        assert event.source == ""
        assert event.correlation_id == ""
        assert event.metadata == {}

    def test_event_with_metadata(self):
        """Test event with metadata."""
        event = Event(
            topic="data.received",
            source="worker_1",
            correlation_id="req-123",
            metadata={"count": 10, "type": "batch"},
        )
        assert event.source == "worker_1"
        assert event.correlation_id == "req-123"
        assert event.metadata["count"] == 10

    def test_event_to_dict(self):
        """Test event serialization to dict."""
        event = Event(
            topic="serialize.test",
            source="test",
        )
        d = event.to_dict()

        assert d["topic"] == "serialize.test"
        assert d["source"] == "test"
        assert "timestamp" in d
        assert d["type"] == "Event"

    def test_event_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        before = time.time()
        event = Event(topic="timing")
        after = time.time()

        assert before <= event.timestamp <= after


class TestCommonEvents:
    """Tests for common event types."""

    def test_system_event(self):
        """Test SystemEvent."""
        event = SystemEvent(
            topic="system.startup",
            level="info",
            message="System starting",
        )
        assert event.level == "info"
        assert event.message == "System starting"

    def test_metric_event(self):
        """Test MetricEvent."""
        event = MetricEvent(
            topic="metric.recorded",
            metric_name="cpu_usage",
            value=75.5,
            tags={"host": "server1"},
        )
        assert event.metric_name == "cpu_usage"
        assert event.value == 75.5
        assert event.tags["host"] == "server1"

    def test_lifecycle_event(self):
        """Test LifecycleEvent."""
        event = LifecycleEvent(
            topic="lifecycle.changed",
            component="worker",
            old_state="idle",
            new_state="running",
        )
        assert event.component == "worker"
        assert event.old_state == "idle"
        assert event.new_state == "running"

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(
            topic="error.occurred",
            error_type="ValueError",
            error_message="Invalid input",
            stack_trace="...",
        )
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"


class TestEventFilter:
    """Tests for EventFilter class."""

    def test_exact_topic_match(self):
        """Test filtering by exact topic."""
        filter = EventFilter(topic="user.created")

        assert filter.matches(Event(topic="user.created"))
        assert not filter.matches(Event(topic="user.deleted"))

    def test_topic_pattern_match(self):
        """Test filtering by topic pattern."""
        filter = EventFilter(topic_pattern="training\\..*")

        assert filter.matches(Event(topic="training.started"))
        assert filter.matches(Event(topic="training.completed"))
        assert not filter.matches(Event(topic="evaluation.started"))

    def test_event_type_match(self):
        """Test filtering by event type."""
        filter = EventFilter(event_type=MetricEvent)

        assert filter.matches(MetricEvent(topic="metric.test"))
        assert not filter.matches(Event(topic="generic"))

    def test_source_match(self):
        """Test filtering by source."""
        filter = EventFilter(source="worker_1")

        assert filter.matches(Event(topic="test", source="worker_1"))
        assert not filter.matches(Event(topic="test", source="worker_2"))

    def test_predicate_match(self):
        """Test filtering by custom predicate."""
        filter = EventFilter(
            predicate=lambda e: e.metadata.get("priority") == "high"
        )

        assert filter.matches(Event(topic="test", metadata={"priority": "high"}))
        assert not filter.matches(Event(topic="test", metadata={"priority": "low"}))

    def test_combined_filters(self):
        """Test combining multiple filter criteria."""
        filter = EventFilter(
            topic="data.processed",
            source="worker_1",
        )

        # Both criteria must match
        assert filter.matches(Event(topic="data.processed", source="worker_1"))
        assert not filter.matches(Event(topic="data.processed", source="worker_2"))
        assert not filter.matches(Event(topic="other.topic", source="worker_1"))

    def test_empty_filter_matches_all(self):
        """Test empty filter matches all events."""
        filter = EventFilter()

        assert filter.matches(Event(topic="anything"))
        assert filter.matches(MetricEvent(topic="metrics"))


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_subscription_creation(self):
        """Test subscription creation."""
        handler = lambda e: None
        filter = EventFilter(topic="test")

        sub = Subscription(handler=handler, filter=filter)

        assert sub.handler == handler
        assert sub.priority == 0
        assert sub.once is False
        assert sub.weak is False

    def test_subscription_with_priority(self):
        """Test subscription with priority."""
        sub = Subscription(
            handler=lambda e: None,
            filter=EventFilter(),
            priority=10,
        )

        assert sub.priority == 10

    def test_subscription_once(self):
        """Test one-time subscription."""
        sub = Subscription(
            handler=lambda e: None,
            filter=EventFilter(),
            once=True,
        )

        assert sub.once is True

    def test_get_handler_normal(self):
        """Test getting handler from normal subscription."""
        handler = lambda e: None
        sub = Subscription(handler=handler, filter=EventFilter())

        assert sub.get_handler() == handler


class TestEventBus:
    """Tests for EventBus class."""

    @pytest.fixture
    def bus(self):
        """Create a fresh event bus for each test."""
        return EventBus()

    def test_bus_creation(self, bus):
        """Test event bus creation."""
        assert bus._subscriptions == {}
        assert bus._pattern_subscriptions == []
        assert bus._all_subscriptions == []

    def test_subscribe_decorator(self, bus):
        """Test subscribe as decorator."""
        received = []

        @bus.subscribe("test.topic")
        def handler(event):
            received.append(event)

        assert "test.topic" in bus._subscriptions
        assert len(bus._subscriptions["test.topic"]) == 1

    def test_add_subscription(self, bus):
        """Test programmatic subscription."""
        handler = lambda e: None
        sub = bus.add_subscription(handler, "my.topic")

        assert sub is not None
        assert "my.topic" in bus._subscriptions

    def test_subscribe_with_filter(self, bus):
        """Test subscription with EventFilter."""
        handler = lambda e: None
        filter = EventFilter(topic_pattern="data\\..*")

        bus.add_subscription(handler, filter)

        assert len(bus._pattern_subscriptions) == 1

    def test_subscribe_all_events(self, bus):
        """Test subscription to all events."""
        handler = lambda e: None
        bus.add_subscription(handler, None)

        assert len(bus._all_subscriptions) == 1

    @pytest.mark.asyncio
    async def test_publish_event(self, bus):
        """Test publishing an event."""
        received = []

        @bus.subscribe("pub.test")
        def handler(event):
            received.append(event)

        await bus.publish(Event(topic="pub.test"))

        assert len(received) == 1
        assert received[0].topic == "pub.test"

    @pytest.mark.asyncio
    async def test_publish_to_multiple_handlers(self, bus):
        """Test publishing to multiple handlers."""
        count = {"value": 0}

        @bus.subscribe("multi.test")
        def handler1(event):
            count["value"] += 1

        @bus.subscribe("multi.test")
        def handler2(event):
            count["value"] += 1

        await bus.publish(Event(topic="multi.test"))

        assert count["value"] == 2

    @pytest.mark.asyncio
    async def test_publish_async_handler(self, bus):
        """Test publishing to async handler."""
        received = []

        @bus.subscribe("async.test")
        async def handler(event):
            await asyncio.sleep(0.01)
            received.append(event)

        await bus.publish(Event(topic="async.test"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_requires_topic(self, bus):
        """Test that publish requires a topic."""
        with pytest.raises(ValueError, match="must have a topic"):
            await bus.publish(Event())

    def test_publish_sync(self, bus):
        """Test synchronous publish."""
        received = []

        @bus.subscribe("sync.test")
        def handler(event):
            received.append(event)

        delivered = bus.publish_sync(Event(topic="sync.test"))

        assert delivered == 1
        assert len(received) == 1

    def test_publish_sync_skips_async_handlers(self, bus):
        """Test that publish_sync skips async handlers."""
        sync_received = []
        async_received = []

        @bus.subscribe("skip.async")
        def sync_handler(event):
            sync_received.append(event)

        @bus.subscribe("skip.async")
        async def async_handler(event):
            async_received.append(event)

        bus.publish_sync(Event(topic="skip.async"))

        assert len(sync_received) == 1
        assert len(async_received) == 0  # Async handler skipped

    @pytest.mark.asyncio
    async def test_unsubscribe_handler(self, bus):
        """Test unsubscribing a handler."""
        received = []

        def handler(event):
            received.append(event)

        bus.add_subscription(handler, "unsub.test")
        await bus.publish(Event(topic="unsub.test"))
        assert len(received) == 1

        removed = bus.unsubscribe(handler)
        await bus.publish(Event(topic="unsub.test"))

        assert removed == 1
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_unsubscribe_by_topic(self, bus):
        """Test unsubscribing all handlers for a topic."""
        @bus.subscribe("remove.all")
        def h1(e): pass

        @bus.subscribe("remove.all")
        def h2(e): pass

        removed = bus.unsubscribe(topic="remove.all")

        assert removed == 2
        assert "remove.all" not in bus._subscriptions

    @pytest.mark.asyncio
    async def test_once_subscription(self, bus):
        """Test one-time subscription."""
        received = []

        @bus.subscribe("once.test", once=True)
        def handler(event):
            received.append(event)

        await bus.publish(Event(topic="once.test"))
        await bus.publish(Event(topic="once.test"))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_priority_ordering(self, bus):
        """Test handlers are called in priority order."""
        order = []

        @bus.subscribe("priority.test", priority=1)
        def low_priority(event):
            order.append("low")

        @bus.subscribe("priority.test", priority=10)
        def high_priority(event):
            order.append("high")

        await bus.publish(Event(topic="priority.test"))

        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_handler_error_isolated(self, bus):
        """Test that handler errors don't affect other handlers."""
        received = []

        @bus.subscribe("error.test")
        def failing_handler(event):
            raise RuntimeError("Handler error")

        @bus.subscribe("error.test")
        def success_handler(event):
            received.append(event)

        await bus.publish(Event(topic="error.test"))

        assert len(received) == 1  # Second handler still called

    @pytest.mark.asyncio
    async def test_pattern_matching(self, bus):
        """Test pattern-based subscriptions."""
        received = []

        bus.add_subscription(
            lambda e: received.append(e),
            EventFilter(topic_pattern="data\\..*"),
        )

        await bus.publish(Event(topic="data.created"))
        await bus.publish(Event(topic="data.updated"))
        await bus.publish(Event(topic="other.topic"))

        assert len(received) == 2


class TestEventHistory:
    """Tests for event history functionality."""

    @pytest.fixture
    def bus(self):
        """Create event bus with history enabled."""
        return EventBus(enable_history=True, max_history=100)

    @pytest.mark.asyncio
    async def test_history_recorded(self, bus):
        """Test events are recorded in history."""
        await bus.publish(Event(topic="history.test"))

        history = bus.get_history()
        assert len(history) == 1
        assert history[0].topic == "history.test"

    @pytest.mark.asyncio
    async def test_history_by_topic(self, bus):
        """Test filtering history by topic."""
        await bus.publish(Event(topic="a"))
        await bus.publish(Event(topic="b"))
        await bus.publish(Event(topic="a"))

        history = bus.get_history(topic="a")
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self, bus):
        """Test history respects max_history limit."""
        bus._max_history = 5

        for i in range(10):
            await bus.publish(Event(topic=f"event.{i}"))

        history = bus.get_history()
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_history_most_recent_first(self, bus):
        """Test history returns most recent first."""
        await bus.publish(Event(topic="first"))
        await bus.publish(Event(topic="second"))
        await bus.publish(Event(topic="third"))

        history = bus.get_history()
        assert history[0].topic == "third"
        assert history[2].topic == "first"

    def test_clear_history(self, bus):
        """Test clearing history."""
        bus._history.append(Event(topic="test"))
        bus.clear_history()

        assert len(bus._history) == 0

    @pytest.mark.asyncio
    async def test_replay_events(self, bus):
        """Test replaying events to a handler."""
        await bus.publish(Event(topic="replay.1"))
        await bus.publish(Event(topic="replay.2"))

        replayed = []

        async def replay_handler(event):
            replayed.append(event)

        count = await bus.replay(replay_handler)

        assert count == 2
        assert len(replayed) == 2

    @pytest.mark.asyncio
    async def test_replay_with_filter(self, bus):
        """Test replay with topic filter."""
        await bus.publish(Event(topic="a"))
        await bus.publish(Event(topic="b"))

        replayed = []
        await bus.replay(lambda e: replayed.append(e), topic="a")

        assert len(replayed) == 1

    def test_history_disabled(self):
        """Test history can be disabled."""
        bus = EventBus(enable_history=False)

        # Manually check - history won't be recorded
        assert bus._enable_history is False


class TestEventBusStats:
    """Tests for event bus statistics."""

    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_stats_events_published(self, bus):
        """Test published events are counted."""
        await bus.publish(Event(topic="stats.1"))
        await bus.publish(Event(topic="stats.2"))

        stats = bus.get_stats()
        assert stats["events_published"] == 2

    @pytest.mark.asyncio
    async def test_stats_events_delivered(self, bus):
        """Test delivered events are counted."""
        @bus.subscribe("deliver.test")
        def handler(e): pass

        await bus.publish(Event(topic="deliver.test"))

        stats = bus.get_stats()
        assert stats["events_delivered"] == 1

    @pytest.mark.asyncio
    async def test_stats_delivery_errors(self, bus):
        """Test delivery errors are counted."""
        @bus.subscribe("error.stats")
        def handler(e):
            raise RuntimeError("Error")

        await bus.publish(Event(topic="error.stats"))

        stats = bus.get_stats()
        assert stats["delivery_errors"] == 1

    def test_stats_subscriptions(self, bus):
        """Test subscription counts in stats."""
        bus.add_subscription(lambda e: None, "topic1")
        bus.add_subscription(lambda e: None, EventFilter(topic_pattern=".*"))
        bus.add_subscription(lambda e: None, None)

        stats = bus.get_stats()
        assert stats["subscriptions"]["by_topic"] == 1
        assert stats["subscriptions"]["by_pattern"] == 1
        assert stats["subscriptions"]["all_events"] == 1


class TestModuleFunctions:
    """Tests for module-level functions."""

    def setup_method(self):
        """Reset global bus before each test."""
        reset_event_bus()

    def teardown_method(self):
        """Reset global bus after each test."""
        reset_event_bus()

    def test_get_event_bus_singleton(self):
        """Test get_event_bus returns singleton."""
        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            bus1 = get_event_bus()
        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus(self):
        """Test reset_event_bus clears singleton."""
        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            bus1 = get_event_bus()
        reset_event_bus()
        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            bus2 = get_event_bus()
        assert bus1 is not bus2

    def test_subscribe_function(self):
        """Test module-level subscribe function."""
        received = []

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            decorator = subscribe("module.test")

        @decorator
        def handler(event):
            received.append(event)

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            bus = get_event_bus()
        assert "module.test" in bus._subscriptions

    @pytest.mark.asyncio
    async def test_publish_function(self):
        """Test module-level publish function."""
        received = []

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            decorator = subscribe("pub.module")

        @decorator
        def handler(event):
            received.append(event)

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            await publish(Event(topic="pub.module"))
        assert len(received) == 1

    def test_publish_sync_function(self):
        """Test module-level publish_sync function."""
        received = []

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            decorator = subscribe("sync.module")

        @decorator
        def handler(event):
            received.append(event)

        with pytest.deprecated_call(match="get_event_bus\\(\\) from core\\.event_bus"):
            publish_sync(Event(topic="sync.module"))
        assert len(received) == 1


class TestCustomEvents:
    """Tests for custom event types."""

    def test_custom_event_subclass(self):
        """Test custom event subclass."""
        @dataclass
        class TrainingEvent(Event):
            config_key: str = ""
            epoch: int = 0

        event = TrainingEvent(
            topic="training.progress",
            config_key="hex8_2p",
            epoch=10,
        )

        assert event.topic == "training.progress"
        assert event.config_key == "hex8_2p"
        assert event.epoch == 10

    def test_filter_by_custom_type(self):
        """Test filtering by custom event type."""
        @dataclass
        class CustomEvent(Event):
            custom_field: str = ""

        filter = EventFilter(event_type=CustomEvent)

        assert filter.matches(CustomEvent(topic="custom"))
        assert not filter.matches(Event(topic="generic"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
