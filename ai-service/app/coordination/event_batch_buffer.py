"""Event batching and buffering for high-throughput event publishing.

December 29, 2025: Created to optimize event router throughput during
high-activity phases (selfplay, training, curriculum updates).

Problem: The event router publishes serially, becoming a bottleneck when:
- SelfplayScheduler emits SELFPLAY_TARGET_UPDATED after priority recalculation
- Training coordinator emits TRAINING_COMPLETED, MODEL_PROMOTED in sequence
- Curriculum feedback loops emit CURRICULUM_UPDATED to 8+ subscribers

Solution: Buffer events and publish in batches to reduce per-event overhead:
- EventBuffer: Accumulates events until flush threshold or timeout
- BatchPublisher: Publishes batched events with configurable strategies
- EventCoalescer: Deduplicates rapid-fire events (same type within window)

Usage:
    from app.coordination.event_batch_buffer import (
        EventBuffer,
        BatchPublisher,
        EventCoalescer,
        get_batch_publisher,
    )

    # Simple batching
    buffer = EventBuffer(max_size=100, flush_interval=0.5)
    await buffer.add("SELFPLAY_COMPLETED", {"config": "hex8_2p"})
    # Auto-flushes when full or after 0.5s

    # With coalescing (dedup rapid events)
    coalescer = EventCoalescer(window_seconds=1.0)
    coalescer.add("PRIORITY_UPDATED", {"config": "hex8_2p", "priority": 0.8})
    coalescer.add("PRIORITY_UPDATED", {"config": "hex8_2p", "priority": 0.85})
    # Only the last event is published
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics (Sprint 10 - Jan 3, 2026)
# =============================================================================

try:
    from prometheus_client import Counter, Histogram, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    REGISTRY = None


def _get_or_create_counter(name: str, description: str, labels: list):
    """Get existing counter or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        return Counter(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_histogram(name: str, description: str, labels: list, buckets=None):
    """Get existing histogram or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        if buckets:
            return Histogram(name, description, labels, buckets=buckets)
        return Histogram(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


if HAS_PROMETHEUS:
    # Count of events that were coalesced (deduplicated)
    PROM_EVENTS_COALESCED = _get_or_create_counter(
        'ringrift_events_coalesced_total',
        'Events deduplicated by coalescer',
        ['event_type']
    )
    # Histogram of batch sizes when flushed
    PROM_BATCH_SIZE = _get_or_create_histogram(
        'ringrift_event_batch_size',
        'Number of events per batch when flushed',
        ['event_type'],
        buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
    )
    # Count of events published via batch buffer
    PROM_EVENTS_PUBLISHED = _get_or_create_counter(
        'ringrift_events_batched_total',
        'Total events published via batch buffer',
        ['event_type']
    )


# =============================================================================
# Configuration
# =============================================================================


class FlushStrategy(Enum):
    """Strategy for when to flush the buffer."""

    SIZE = "size"          # Flush when buffer reaches max_size
    TIME = "time"          # Flush after flush_interval seconds
    HYBRID = "hybrid"      # Flush on size OR time (whichever first)
    IMMEDIATE = "immediate"  # No buffering, publish immediately


@dataclass
class BatchConfig:
    """Configuration for event batching."""

    max_size: int = 100
    flush_interval: float = 0.5  # seconds
    strategy: FlushStrategy = FlushStrategy.HYBRID
    coalesce_window: float = 0.0  # 0 = no coalescing
    coalesce_event_types: set[str] = field(default_factory=set)
    max_batch_publish_time: float = 5.0  # Max time for batch publish
    drop_on_overflow: bool = False  # Drop events if buffer full

    @classmethod
    def high_throughput(cls) -> "BatchConfig":
        """Config optimized for high throughput (larger batches, less frequent)."""
        return cls(
            max_size=500,
            flush_interval=1.0,
            strategy=FlushStrategy.SIZE,
            coalesce_window=0.5,
        )

    @classmethod
    def low_latency(cls) -> "BatchConfig":
        """Config optimized for low latency (smaller batches, more frequent)."""
        return cls(
            max_size=20,
            flush_interval=0.1,
            strategy=FlushStrategy.HYBRID,
            coalesce_window=0.0,
        )

    @classmethod
    def no_buffering(cls) -> "BatchConfig":
        """Config for immediate publishing (no buffering)."""
        return cls(
            strategy=FlushStrategy.IMMEDIATE,
        )


# =============================================================================
# Buffered Event
# =============================================================================


@dataclass
class BufferedEvent:
    """Event waiting in buffer."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more important
    dedup_key: str | None = None

    def __lt__(self, other: "BufferedEvent") -> bool:
        """Compare by priority (for heapq)."""
        return self.priority > other.priority  # Higher priority first

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for publishing."""
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }


# =============================================================================
# Event Buffer
# =============================================================================


class EventBuffer:
    """Buffer for accumulating events before batch publishing.

    Events are accumulated until:
    - Buffer reaches max_size (SIZE or HYBRID strategy)
    - flush_interval elapses (TIME or HYBRID strategy)
    - flush() is called explicitly

    Example:
        buffer = EventBuffer(
            config=BatchConfig(max_size=100, flush_interval=0.5),
            on_flush=publish_batch,
        )
        await buffer.start()

        # Add events
        await buffer.add("EVENT_TYPE", {"key": "value"})

        # Events are auto-published in batches
        await buffer.stop()
    """

    def __init__(
        self,
        config: BatchConfig | None = None,
        on_flush: Callable[[list[BufferedEvent]], Any] | None = None,
    ) -> None:
        """Initialize event buffer.

        Args:
            config: Batch configuration
            on_flush: Callback to handle flushed events
        """
        self.config = config or BatchConfig()
        self._on_flush = on_flush

        self._buffer: list[BufferedEvent] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: asyncio.Task[None] | None = None
        self._last_flush_time = time.time()

        # Metrics
        self._total_added = 0
        self._total_flushed = 0
        self._total_dropped = 0
        self._flush_count = 0

    async def start(self) -> None:
        """Start the buffer (begins flush timer if needed)."""
        if self._running:
            return

        self._running = True
        self._last_flush_time = time.time()

        if self.config.strategy in (FlushStrategy.TIME, FlushStrategy.HYBRID):
            self._flush_task = asyncio.create_task(self._flush_loop())

        logger.debug(f"EventBuffer started (strategy={self.config.strategy.value})")

    async def stop(self) -> None:
        """Stop the buffer and flush remaining events."""
        if not self._running:
            return

        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()
        logger.debug(
            f"EventBuffer stopped. Stats: added={self._total_added}, "
            f"flushed={self._total_flushed}, dropped={self._total_dropped}"
        )

    async def add(
        self,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        dedup_key: str | None = None,
    ) -> bool:
        """Add an event to the buffer.

        Args:
            event_type: Type of event
            payload: Event payload
            priority: Event priority (higher = more important)
            dedup_key: Optional key for deduplication

        Returns:
            True if event was added, False if dropped
        """
        if self.config.strategy == FlushStrategy.IMMEDIATE:
            # No buffering - publish immediately
            event = BufferedEvent(
                event_type=event_type,
                payload=payload,
                priority=priority,
                dedup_key=dedup_key,
            )
            if self._on_flush:
                await self._call_flush_handler([event])
            self._total_added += 1
            self._total_flushed += 1
            return True

        async with self._lock:
            # Check buffer capacity
            if len(self._buffer) >= self.config.max_size:
                if self.config.drop_on_overflow:
                    self._total_dropped += 1
                    return False
                else:
                    # Force flush to make room
                    await self._flush_unlocked()

            event = BufferedEvent(
                event_type=event_type,
                payload=payload,
                priority=priority,
                dedup_key=dedup_key,
            )
            self._buffer.append(event)
            self._total_added += 1

            # Check if we should flush based on size
            if (
                self.config.strategy in (FlushStrategy.SIZE, FlushStrategy.HYBRID)
                and len(self._buffer) >= self.config.max_size
            ):
                await self._flush_unlocked()

        return True

    async def flush(self) -> int:
        """Flush all buffered events.

        Returns:
            Number of events flushed
        """
        async with self._lock:
            return await self._flush_unlocked()

    async def _flush_unlocked(self) -> int:
        """Flush without acquiring lock (caller must hold lock)."""
        if not self._buffer:
            return 0

        events = self._buffer.copy()
        self._buffer.clear()
        self._last_flush_time = time.time()
        self._flush_count += 1

        # Sort by priority
        events.sort()

        if self._on_flush:
            await self._call_flush_handler(events)

        self._total_flushed += len(events)
        return len(events)

    async def _call_flush_handler(self, events: list[BufferedEvent]) -> None:
        """Call flush handler with timeout."""
        try:
            result = self._on_flush(events)  # type: ignore
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=self.config.max_batch_publish_time)
        except asyncio.TimeoutError:
            logger.warning(f"EventBuffer flush handler timed out ({len(events)} events)")
        except Exception as e:
            logger.error(f"EventBuffer flush handler error: {e}")

    async def _flush_loop(self) -> None:
        """Background task for time-based flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval)

                async with self._lock:
                    if self._buffer:
                        await self._flush_unlocked()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBuffer flush loop error: {e}")

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Whether buffer is empty."""
        return len(self._buffer) == 0

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self._buffer),
            "max_size": self.config.max_size,
            "strategy": self.config.strategy.value,
            "running": self._running,
            "total_added": self._total_added,
            "total_flushed": self._total_flushed,
            "total_dropped": self._total_dropped,
            "flush_count": self._flush_count,
            "seconds_since_last_flush": time.time() - self._last_flush_time,
        }


# =============================================================================
# Event Coalescer
# =============================================================================


class EventCoalescer:
    """Coalesces rapid-fire events of the same type.

    When multiple events of the same type are received within a time window,
    only the last one is kept. Useful for rapidly-changing state updates.

    Example:
        coalescer = EventCoalescer(window_seconds=1.0)

        # Rapid priority updates
        await coalescer.add("PRIORITY_UPDATED", {"config": "hex8_2p", "priority": 0.8})
        await coalescer.add("PRIORITY_UPDATED", {"config": "hex8_2p", "priority": 0.85})
        await coalescer.add("PRIORITY_UPDATED", {"config": "hex8_2p", "priority": 0.9})

        # Only the last event (priority=0.9) is published after 1 second
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        key_extractor: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize coalescer.

        Args:
            window_seconds: Time window for coalescing
            key_extractor: Function to extract coalescing key from event.
                          Default: event_type + hash of payload keys
        """
        self.window_seconds = window_seconds
        self._key_extractor = key_extractor or self._default_key_extractor

        self._pending: dict[str, BufferedEvent] = {}
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._on_emit: Callable[[BufferedEvent], Any] | None = None
        self._lock = asyncio.Lock()

        # Metrics
        self._total_received = 0
        self._total_coalesced = 0
        self._total_emitted = 0

    def set_emit_handler(self, handler: Callable[[BufferedEvent], Any]) -> None:
        """Set handler for emitted events."""
        self._on_emit = handler

    async def add(
        self,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> None:
        """Add an event for coalescing.

        Args:
            event_type: Type of event
            payload: Event payload
            priority: Event priority (kept from latest event)
        """
        key = self._key_extractor(event_type, payload)

        async with self._lock:
            self._total_received += 1

            event = BufferedEvent(
                event_type=event_type,
                payload=payload,
                priority=priority,
                dedup_key=key,
            )

            if key in self._pending:
                # Replace existing pending event
                self._total_coalesced += 1
                self._pending[key] = event
                # Sprint 10: Record coalesced event metric
                if HAS_PROMETHEUS and PROM_EVENTS_COALESCED is not None:
                    PROM_EVENTS_COALESCED.labels(event_type=event_type).inc()
            else:
                # New event - start timer
                self._pending[key] = event
                timer = asyncio.create_task(self._emit_after_delay(key))
                self._timers[key] = timer

    async def _emit_after_delay(self, key: str) -> None:
        """Wait for window then emit event."""
        try:
            await asyncio.sleep(self.window_seconds)

            async with self._lock:
                event = self._pending.pop(key, None)
                self._timers.pop(key, None)

                if event and self._on_emit:
                    self._total_emitted += 1
                    # Sprint 10: Record emitted event metric
                    if HAS_PROMETHEUS and PROM_EVENTS_PUBLISHED is not None:
                        PROM_EVENTS_PUBLISHED.labels(event_type=event.event_type).inc()
                    result = self._on_emit(event)
                    if asyncio.iscoroutine(result):
                        await result

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"EventCoalescer emit error: {e}")

    async def flush(self) -> list[BufferedEvent]:
        """Immediately emit all pending events."""
        async with self._lock:
            events = list(self._pending.values())
            self._pending.clear()

            # Cancel timers
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

            # Sprint 10: Record batch size metric
            if HAS_PROMETHEUS and PROM_BATCH_SIZE is not None and events:
                PROM_BATCH_SIZE.labels(event_type="flush").observe(len(events))

            for event in events:
                self._total_emitted += 1
                # Sprint 10: Record emitted event metric
                if HAS_PROMETHEUS and PROM_EVENTS_PUBLISHED is not None:
                    PROM_EVENTS_PUBLISHED.labels(event_type=event.event_type).inc()
                if self._on_emit:
                    result = self._on_emit(event)
                    if asyncio.iscoroutine(result):
                        await result

            return events

    @staticmethod
    def _default_key_extractor(event_type: str, payload: dict[str, Any]) -> str:
        """Default key: event_type + sorted payload keys."""
        # Include event type and stable payload signature
        key_parts = [event_type]

        # Add config_key if present (common identifier)
        if "config_key" in payload:
            key_parts.append(str(payload["config_key"]))
        elif "config" in payload:
            key_parts.append(str(payload["config"]))

        return ":".join(key_parts)

    def get_stats(self) -> dict[str, Any]:
        """Get coalescer statistics."""
        return {
            "window_seconds": self.window_seconds,
            "pending_count": len(self._pending),
            "total_received": self._total_received,
            "total_coalesced": self._total_coalesced,
            "total_emitted": self._total_emitted,
            "coalesce_ratio": (
                self._total_coalesced / max(1, self._total_received)
            ),
        }


# =============================================================================
# Batch Publisher
# =============================================================================


class BatchPublisher:
    """High-level batch publishing with buffering and coalescing.

    Combines EventBuffer and EventCoalescer for optimal throughput.

    Example:
        publisher = BatchPublisher(
            config=BatchConfig.high_throughput(),
            coalesce_types={"PRIORITY_UPDATED", "QUALITY_SCORE_UPDATED"},
        )
        await publisher.start()

        # Publish events
        await publisher.publish("SELFPLAY_COMPLETED", {"config": "hex8_2p"})

        await publisher.stop()
    """

    def __init__(
        self,
        config: BatchConfig | None = None,
        coalesce_types: set[str] | None = None,
        publish_handler: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize batch publisher.

        Args:
            config: Batch configuration
            coalesce_types: Event types to coalesce (dedup rapid-fire)
            publish_handler: Function to publish individual events
        """
        self.config = config or BatchConfig()
        self._coalesce_types = coalesce_types or set()
        self._publish_handler = publish_handler

        # Initialize components
        self._buffer = EventBuffer(
            config=self.config,
            on_flush=self._handle_flush,
        )
        self._coalescer = EventCoalescer(
            window_seconds=self.config.coalesce_window,
        )
        self._coalescer.set_emit_handler(self._handle_coalesced_emit)

        self._running = False

    async def start(self) -> None:
        """Start the publisher."""
        if self._running:
            return

        self._running = True
        await self._buffer.start()
        logger.debug("BatchPublisher started")

    async def stop(self) -> None:
        """Stop the publisher and flush remaining events."""
        if not self._running:
            return

        self._running = False

        # Flush coalescer first
        await self._coalescer.flush()

        # Then buffer
        await self._buffer.stop()

        logger.debug("BatchPublisher stopped")

    async def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> bool:
        """Publish an event.

        Args:
            event_type: Type of event
            payload: Event payload
            priority: Event priority

        Returns:
            True if event was accepted
        """
        if not self._running:
            return False

        # Check if should coalesce
        if event_type in self._coalesce_types and self.config.coalesce_window > 0:
            await self._coalescer.add(event_type, payload, priority)
            return True

        # Otherwise buffer directly
        return await self._buffer.add(event_type, payload, priority)

    async def _handle_flush(self, events: list[BufferedEvent]) -> None:
        """Handle flushed events from buffer."""
        if not self._publish_handler:
            return

        for event in events:
            try:
                result = self._publish_handler(event.event_type, event.payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.debug(f"BatchPublisher publish error: {e}")

    async def _handle_coalesced_emit(self, event: BufferedEvent) -> None:
        """Handle emitted event from coalescer."""
        # Route through buffer for consistent handling
        await self._buffer.add(
            event.event_type,
            event.payload,
            event.priority,
            event.dedup_key,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get publisher statistics."""
        return {
            "running": self._running,
            "coalesce_types": list(self._coalesce_types),
            "buffer": self._buffer.get_stats(),
            "coalescer": self._coalescer.get_stats(),
        }


# =============================================================================
# Singleton Instance
# =============================================================================


_batch_publisher: BatchPublisher | None = None


def get_batch_publisher() -> BatchPublisher | None:
    """Get the singleton batch publisher instance."""
    return _batch_publisher


async def initialize_batch_publisher(
    config: BatchConfig | None = None,
    coalesce_types: set[str] | None = None,
) -> BatchPublisher:
    """Initialize and start the singleton batch publisher.

    Args:
        config: Batch configuration
        coalesce_types: Event types to coalesce

    Returns:
        The batch publisher instance
    """
    global _batch_publisher

    if _batch_publisher is not None:
        return _batch_publisher

    # Default coalesce types for high-frequency events
    default_coalesce = {
        "PRIORITY_UPDATED",
        "QUALITY_SCORE_UPDATED",
        "SELFPLAY_TARGET_UPDATED",
        "ELO_VELOCITY_UPDATED",
    }

    _batch_publisher = BatchPublisher(
        config=config or BatchConfig(),
        coalesce_types=coalesce_types or default_coalesce,
        publish_handler=_default_publish_handler,
    )
    await _batch_publisher.start()

    return _batch_publisher


async def shutdown_batch_publisher() -> None:
    """Shutdown the singleton batch publisher."""
    global _batch_publisher

    if _batch_publisher is not None:
        await _batch_publisher.stop()
        _batch_publisher = None


async def _default_publish_handler(event_type: str, payload: dict[str, Any]) -> None:
    """Default handler that routes to event router."""
    try:
        from app.coordination.event_router import get_event_bus

        bus = get_event_bus()
        if bus:
            bus.publish(event_type, payload)
    except (ImportError, AttributeError, TypeError, RuntimeError) as e:
        # Dec 29, 2025: Narrowed from bare Exception
        # ImportError: event_router not available
        # AttributeError: bus is None or missing publish method
        # TypeError: invalid event_type or payload
        # RuntimeError: event system shutdown
        logger.debug(f"Batch publish to event router failed: {e}")


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Configuration
    "FlushStrategy",
    "BatchConfig",
    # Data classes
    "BufferedEvent",
    # Core classes
    "EventBuffer",
    "EventCoalescer",
    "BatchPublisher",
    # Singleton access
    "get_batch_publisher",
    "initialize_batch_publisher",
    "shutdown_batch_publisher",
]
