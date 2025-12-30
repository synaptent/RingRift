"""Unit tests for event_batch_buffer.py.

Tests the event batching and buffering infrastructure for high-throughput
event publishing. Covers:
- FlushStrategy enum behavior
- BatchConfig factory methods
- BufferedEvent ordering and deduplication
- EventBuffer lifecycle, add/flush, overflow handling
- EventCoalescer deduplication within time windows
- BatchPublisher batch processing
"""

from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.event_batch_buffer import (
    FlushStrategy,
    BatchConfig,
    BufferedEvent,
    EventBuffer,
    EventCoalescer,
    BatchPublisher,
    get_batch_publisher,
    initialize_batch_publisher,
    shutdown_batch_publisher,
)


# =============================================================================
# FlushStrategy Tests
# =============================================================================


class TestFlushStrategy:
    """Tests for FlushStrategy enum."""

    def test_size_strategy_value(self):
        """Test SIZE strategy value."""
        assert FlushStrategy.SIZE.value == "size"

    def test_time_strategy_value(self):
        """Test TIME strategy value."""
        assert FlushStrategy.TIME.value == "time"

    def test_hybrid_strategy_value(self):
        """Test HYBRID strategy value."""
        assert FlushStrategy.HYBRID.value == "hybrid"

    def test_immediate_strategy_value(self):
        """Test IMMEDIATE strategy value."""
        assert FlushStrategy.IMMEDIATE.value == "immediate"

    def test_all_strategies_enumerable(self):
        """Test all strategies can be enumerated."""
        strategies = list(FlushStrategy)
        assert len(strategies) == 4
        assert FlushStrategy.SIZE in strategies
        assert FlushStrategy.TIME in strategies
        assert FlushStrategy.HYBRID in strategies
        assert FlushStrategy.IMMEDIATE in strategies


# =============================================================================
# BatchConfig Tests
# =============================================================================


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.max_size == 100
        assert config.flush_interval == 0.5
        assert config.strategy == FlushStrategy.HYBRID
        assert config.coalesce_window == 0.0
        assert config.coalesce_event_types == set()
        assert config.max_batch_publish_time == 5.0
        assert config.drop_on_overflow is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BatchConfig(
            max_size=50,
            flush_interval=1.0,
            strategy=FlushStrategy.SIZE,
            coalesce_window=2.0,
            coalesce_event_types={"EVENT_A", "EVENT_B"},
            max_batch_publish_time=10.0,
            drop_on_overflow=True,
        )
        assert config.max_size == 50
        assert config.flush_interval == 1.0
        assert config.strategy == FlushStrategy.SIZE
        assert config.coalesce_window == 2.0
        assert "EVENT_A" in config.coalesce_event_types
        assert "EVENT_B" in config.coalesce_event_types
        assert config.max_batch_publish_time == 10.0
        assert config.drop_on_overflow is True

    def test_high_throughput_factory(self):
        """Test high_throughput factory method."""
        config = BatchConfig.high_throughput()
        assert config.max_size == 500
        assert config.flush_interval == 1.0
        assert config.strategy == FlushStrategy.SIZE
        assert config.coalesce_window == 0.5

    def test_low_latency_factory(self):
        """Test low_latency factory method."""
        config = BatchConfig.low_latency()
        assert config.max_size == 20
        assert config.flush_interval == 0.1
        assert config.strategy == FlushStrategy.HYBRID
        assert config.coalesce_window == 0.0

    def test_no_buffering_factory(self):
        """Test no_buffering factory method."""
        config = BatchConfig.no_buffering()
        assert config.strategy == FlushStrategy.IMMEDIATE


# =============================================================================
# BufferedEvent Tests
# =============================================================================


class TestBufferedEvent:
    """Tests for BufferedEvent dataclass."""

    def test_creation_with_defaults(self):
        """Test event creation with default values."""
        event = BufferedEvent(
            event_type="TEST_EVENT",
            payload={"key": "value"},
        )
        assert event.event_type == "TEST_EVENT"
        assert event.payload == {"key": "value"}
        assert event.priority == 0
        assert event.dedup_key is None
        assert event.timestamp > 0

    def test_creation_with_custom_values(self):
        """Test event creation with custom values."""
        event = BufferedEvent(
            event_type="TEST_EVENT",
            payload={"key": "value"},
            priority=10,
            dedup_key="custom-key",
        )
        assert event.priority == 10
        assert event.dedup_key == "custom-key"

    def test_comparison_by_priority(self):
        """Test events are compared by priority (higher first)."""
        low = BufferedEvent("E1", {}, priority=1)
        high = BufferedEvent("E2", {}, priority=10)

        # Higher priority should come first when sorted
        events = [low, high]
        events.sort()
        assert events[0].priority == 10
        assert events[1].priority == 1

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = BufferedEvent(
            event_type="TEST_EVENT",
            payload={"key": "value"},
            priority=5,
        )
        d = event.to_dict()
        assert d["event_type"] == "TEST_EVENT"
        assert d["payload"] == {"key": "value"}
        assert d["priority"] == 5
        assert "timestamp" in d


# =============================================================================
# EventBuffer Tests
# =============================================================================


class TestEventBuffer:
    """Tests for EventBuffer class."""

    @pytest.fixture
    def flush_handler(self):
        """Create a mock flush handler."""
        return AsyncMock()

    @pytest.fixture
    def buffer(self, flush_handler):
        """Create an event buffer with mock handler."""
        return EventBuffer(
            config=BatchConfig(max_size=5, flush_interval=0.1),
            on_flush=flush_handler,
        )

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, buffer):
        """Test buffer start and stop lifecycle."""
        assert not buffer._running

        await buffer.start()
        assert buffer._running

        await buffer.stop()
        assert not buffer._running

    @pytest.mark.asyncio
    async def test_start_idempotent(self, buffer):
        """Test starting twice is safe."""
        await buffer.start()
        await buffer.start()  # Should not raise
        assert buffer._running
        await buffer.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, buffer):
        """Test stopping twice is safe."""
        await buffer.start()
        await buffer.stop()
        await buffer.stop()  # Should not raise
        assert not buffer._running

    @pytest.mark.asyncio
    async def test_add_event(self, buffer, flush_handler):
        """Test adding events to buffer."""
        await buffer.start()

        result = await buffer.add("TEST_EVENT", {"key": "value"})
        assert result is True
        assert buffer.size == 1

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_add_with_priority(self, buffer, flush_handler):
        """Test adding events with priority."""
        await buffer.start()

        await buffer.add("LOW", {}, priority=1)
        await buffer.add("HIGH", {}, priority=10)

        assert buffer.size == 2
        await buffer.stop()

        # Verify high priority came first in flush
        flush_handler.assert_called()
        events = flush_handler.call_args[0][0]
        assert events[0].priority == 10
        assert events[1].priority == 1

    @pytest.mark.asyncio
    async def test_flush_on_max_size(self, buffer, flush_handler):
        """Test automatic flush when max_size is reached."""
        await buffer.start()

        # Add events up to max_size
        for i in range(5):
            await buffer.add(f"EVENT_{i}", {"index": i})

        # Buffer should have flushed automatically
        assert buffer.size == 0
        flush_handler.assert_called_once()

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_flush_explicit(self, buffer, flush_handler):
        """Test explicit flush."""
        await buffer.start()

        await buffer.add("EVENT", {})
        await buffer.add("EVENT", {})

        flushed = await buffer.flush()
        assert flushed == 2
        assert buffer.is_empty

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_drop_on_overflow(self, flush_handler):
        """Test drop_on_overflow behavior."""
        # Use TIME strategy - doesn't auto-flush on size, only on timer
        # And we don't start the buffer, so no timer runs
        config = BatchConfig(
            max_size=2,
            drop_on_overflow=True,
            strategy=FlushStrategy.TIME,  # TIME doesn't auto-flush on size
        )
        buffer = EventBuffer(config=config, on_flush=flush_handler)

        # Add events without starting buffer (no auto flush)
        await buffer.add("E1", {})
        await buffer.add("E2", {})

        # Buffer is now full
        assert buffer.size == 2

        # Third event should be dropped (buffer full, drop_on_overflow=True)
        result = await buffer.add("E3", {})
        assert result is False
        assert buffer._total_dropped == 1

    @pytest.mark.asyncio
    async def test_immediate_strategy(self, flush_handler):
        """Test IMMEDIATE strategy publishes without buffering."""
        config = BatchConfig(strategy=FlushStrategy.IMMEDIATE)
        buffer = EventBuffer(config=config, on_flush=flush_handler)

        await buffer.add("EVENT", {"key": "value"})

        # Should have been published immediately
        flush_handler.assert_called_once()
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_stats(self, buffer, flush_handler):
        """Test buffer statistics."""
        await buffer.start()

        await buffer.add("E1", {})
        await buffer.add("E2", {})
        await buffer.flush()

        stats = buffer.get_stats()
        assert stats["buffer_size"] == 0
        assert stats["max_size"] == 5
        assert stats["strategy"] == "hybrid"
        assert stats["running"] is True
        assert stats["total_added"] == 2
        assert stats["total_flushed"] == 2
        assert stats["total_dropped"] == 0
        assert stats["flush_count"] == 1

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_is_empty(self, buffer):
        """Test is_empty property."""
        await buffer.start()

        assert buffer.is_empty

        await buffer.add("EVENT", {})
        assert not buffer.is_empty

        await buffer.flush()
        assert buffer.is_empty

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_flush_handler_timeout(self):
        """Test flush handler timeout handling."""
        slow_handler = AsyncMock(side_effect=lambda _: asyncio.sleep(10))
        config = BatchConfig(max_batch_publish_time=0.1)
        buffer = EventBuffer(config=config, on_flush=slow_handler)

        await buffer.start()
        await buffer.add("EVENT", {})

        # Should handle timeout gracefully
        await buffer.flush()  # Should not hang

        await buffer.stop()

    @pytest.mark.asyncio
    async def test_flush_handler_exception(self):
        """Test flush handler exception handling."""
        error_handler = AsyncMock(side_effect=RuntimeError("Test error"))
        buffer = EventBuffer(on_flush=error_handler)

        await buffer.start()
        await buffer.add("EVENT", {})

        # Should handle exception gracefully
        await buffer.flush()  # Should not raise

        await buffer.stop()


# =============================================================================
# EventCoalescer Tests
# =============================================================================


class TestEventCoalescer:
    """Tests for EventCoalescer class."""

    @pytest.fixture
    def coalescer(self):
        """Create an event coalescer."""
        return EventCoalescer(window_seconds=0.1)  # Short window for tests

    @pytest.fixture
    def emit_handler(self):
        """Create a mock emit handler."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_add_single_event(self, coalescer, emit_handler):
        """Test adding a single event."""
        coalescer.set_emit_handler(emit_handler)

        await coalescer.add("EVENT", {"key": "value"})

        # Wait for window to expire
        await asyncio.sleep(0.15)

        emit_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_coalesce_same_event_type(self, coalescer, emit_handler):
        """Test coalescing multiple events of same type."""
        coalescer.set_emit_handler(emit_handler)

        # Rapid fire same event type
        await coalescer.add("PRIORITY", {"value": 1})
        await coalescer.add("PRIORITY", {"value": 2})
        await coalescer.add("PRIORITY", {"value": 3})

        # Wait for window
        await asyncio.sleep(0.15)

        # Only one event should be emitted (the last one)
        emit_handler.assert_called_once()
        event = emit_handler.call_args[0][0]
        assert event.payload["value"] == 3

    @pytest.mark.asyncio
    async def test_different_event_types_not_coalesced(self, coalescer, emit_handler):
        """Test different event types are not coalesced."""
        coalescer.set_emit_handler(emit_handler)

        await coalescer.add("EVENT_A", {"value": 1})
        await coalescer.add("EVENT_B", {"value": 2})

        # Wait for window
        await asyncio.sleep(0.15)

        # Both events should be emitted
        assert emit_handler.call_count == 2

    @pytest.mark.asyncio
    async def test_custom_key_extractor(self, emit_handler):
        """Test custom key extractor for coalescing."""

        def key_extractor(event_type: str, payload: dict) -> str:
            # Coalesce by event_type + config
            return f"{event_type}:{payload.get('config', '')}"

        coalescer = EventCoalescer(window_seconds=0.1, key_extractor=key_extractor)
        coalescer.set_emit_handler(emit_handler)

        # Same event type, different configs - should NOT coalesce
        await coalescer.add("UPDATE", {"config": "hex8", "value": 1})
        await coalescer.add("UPDATE", {"config": "sq8", "value": 2})

        await asyncio.sleep(0.15)

        # Both should be emitted (different keys)
        assert emit_handler.call_count == 2

    @pytest.mark.asyncio
    async def test_flush_immediate(self, coalescer, emit_handler):
        """Test flush() immediately emits pending events."""
        coalescer.set_emit_handler(emit_handler)

        await coalescer.add("EVENT", {"key": "value"})

        # Flush before window expires
        events = await coalescer.flush()

        # Event should be emitted
        assert len(events) == 1
        emit_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats(self, coalescer, emit_handler):
        """Test coalescer statistics."""
        coalescer.set_emit_handler(emit_handler)

        await coalescer.add("EVENT", {"value": 1})
        await coalescer.add("EVENT", {"value": 2})  # Coalesced
        await coalescer.add("EVENT", {"value": 3})  # Coalesced

        await asyncio.sleep(0.15)

        stats = coalescer.get_stats()
        assert stats["total_received"] == 3
        assert stats["total_coalesced"] == 2
        assert stats["total_emitted"] == 1


# =============================================================================
# BatchPublisher Tests
# =============================================================================


class TestBatchPublisher:
    """Tests for BatchPublisher class."""

    @pytest.fixture
    def publish_handler(self):
        """Create a mock publish handler."""
        return AsyncMock()

    @pytest.fixture
    def publisher(self, publish_handler):
        """Create a batch publisher."""
        return BatchPublisher(
            config=BatchConfig(max_size=5, flush_interval=0.1),
            publish_handler=publish_handler,
        )

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, publisher):
        """Test publisher start and stop lifecycle."""
        await publisher.start()
        assert publisher._running
        assert publisher._buffer._running

        await publisher.stop()
        assert not publisher._running
        assert not publisher._buffer._running

    @pytest.mark.asyncio
    async def test_publish_single_event(self, publisher, publish_handler):
        """Test publishing a single event."""
        await publisher.start()

        await publisher.publish("TEST_EVENT", {"key": "value"})
        await publisher._buffer.flush()

        publish_handler.assert_called()
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_batch(self, publisher, publish_handler):
        """Test batch publishing."""
        await publisher.start()

        # Add multiple events
        for i in range(3):
            await publisher.publish(f"EVENT_{i}", {"index": i})

        await publisher._buffer.flush()

        # Handler should be called for each event
        assert publish_handler.call_count == 3

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_auto_flush_on_size(self, publisher, publish_handler):
        """Test automatic flush when size threshold reached."""
        await publisher.start()

        # Add max_size events
        for i in range(5):
            await publisher.publish(f"EVENT_{i}", {})

        # Should have auto-flushed
        assert publish_handler.call_count == 5

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, publisher, publish_handler):
        """Test getting publisher stats."""
        await publisher.start()

        await publisher.publish("EVENT", {})
        await publisher._buffer.flush()

        stats = publisher.get_stats()
        assert "buffer" in stats
        assert stats["buffer"]["total_added"] == 1
        assert stats["buffer"]["total_flushed"] == 1

        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_not_running_returns_false(self, publisher):
        """Test publishing when not running returns False."""
        # Don't start
        result = await publisher.publish("EVENT", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_coalesce_types(self, publish_handler):
        """Test events in coalesce_types are coalesced."""
        publisher = BatchPublisher(
            config=BatchConfig(
                max_size=100,
                flush_interval=1.0,
                coalesce_window=0.1,  # Enable coalescing
            ),
            coalesce_types={"PRIORITY_UPDATED"},
            publish_handler=publish_handler,
        )

        await publisher.start()

        # Rapid fire same event type
        await publisher.publish("PRIORITY_UPDATED", {"value": 1})
        await publisher.publish("PRIORITY_UPDATED", {"value": 2})
        await publisher.publish("PRIORITY_UPDATED", {"value": 3})

        # Wait for coalesce window
        await asyncio.sleep(0.15)

        # Allow buffer to flush
        await publisher._buffer.flush()

        # Should only publish 1 event (coalesced)
        assert publish_handler.call_count == 1

        await publisher.stop()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestBatchPublisherSingleton:
    """Tests for batch publisher singleton functions."""

    @pytest.mark.asyncio
    async def test_get_before_init_returns_none(self):
        """Test get_batch_publisher returns None before initialization."""
        # Ensure clean state
        await shutdown_batch_publisher()

        publisher = get_batch_publisher()
        assert publisher is None

    @pytest.mark.asyncio
    async def test_initialize_and_get(self):
        """Test initializing and getting the singleton."""
        await shutdown_batch_publisher()  # Clean state

        publisher = await initialize_batch_publisher()

        assert publisher is not None
        assert get_batch_publisher() is publisher

        await shutdown_batch_publisher()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        """Test shutdown cleans up the singleton."""
        await shutdown_batch_publisher()  # Clean state

        await initialize_batch_publisher()
        assert get_batch_publisher() is not None

        await shutdown_batch_publisher()
        assert get_batch_publisher() is None

    @pytest.mark.asyncio
    async def test_double_init_returns_same(self):
        """Test double initialization returns same instance."""
        await shutdown_batch_publisher()  # Clean state

        publisher1 = await initialize_batch_publisher()
        publisher2 = await initialize_batch_publisher()

        assert publisher1 is publisher2

        await shutdown_batch_publisher()

    @pytest.mark.asyncio
    async def test_double_shutdown_safe(self):
        """Test double shutdown is safe."""
        await shutdown_batch_publisher()  # Clean state

        await initialize_batch_publisher()
        await shutdown_batch_publisher()
        await shutdown_batch_publisher()  # Should not raise

    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initializing with custom config."""
        await shutdown_batch_publisher()  # Clean state

        config = BatchConfig.high_throughput()
        publisher = await initialize_batch_publisher(config=config)

        assert publisher.config.max_size == 500
        assert publisher.config.strategy == FlushStrategy.SIZE

        await shutdown_batch_publisher()

    @pytest.mark.asyncio
    async def test_initialize_with_coalesce_types(self):
        """Test initializing with custom coalesce types."""
        await shutdown_batch_publisher()  # Clean state

        coalesce = {"CUSTOM_EVENT"}
        publisher = await initialize_batch_publisher(coalesce_types=coalesce)

        assert "CUSTOM_EVENT" in publisher._coalesce_types

        await shutdown_batch_publisher()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBufferCoalescerIntegration:
    """Integration tests for buffer with coalescer."""

    @pytest.mark.asyncio
    async def test_coalescing_with_buffer(self):
        """Test coalescing events before buffering."""
        emitted_events = []

        async def on_emit(event):
            emitted_events.append(event)

        # Create coalescer
        coalescer = EventCoalescer(window_seconds=0.1)
        coalescer.set_emit_handler(on_emit)

        # Rapid fire events
        for i in range(10):
            await coalescer.add("RAPID_EVENT", {"value": i})

        # Wait for coalesce window
        await asyncio.sleep(0.15)

        # Only the last value should be emitted
        assert len(emitted_events) == 1
        assert emitted_events[0].payload["value"] == 9

    @pytest.mark.asyncio
    async def test_batch_publisher_end_to_end(self):
        """Test BatchPublisher end-to-end flow."""
        published = []

        async def handler(event_type, payload):
            published.append((event_type, payload))

        publisher = BatchPublisher(
            config=BatchConfig(max_size=3, flush_interval=1.0),
            publish_handler=handler,
        )

        await publisher.start()

        # Publish events
        await publisher.publish("E1", {"n": 1})
        await publisher.publish("E2", {"n": 2})
        await publisher.publish("E3", {"n": 3})  # Should trigger flush

        # Verify events were published
        assert len(published) == 3

        await publisher.stop()
