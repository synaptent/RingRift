"""Unified Event Bus for RingRift AI Service.

Provides a centralized event system with:
- Topic-based pub/sub
- Async and sync handlers
- Event filtering and routing
- Event history and replay
- Type-safe events

Usage:
    from app.core.event_bus import EventBus, Event, get_event_bus

    # Define events
    @dataclass
    class TrainingCompleted(Event):
        config_key: str
        metrics: Dict[str, float]

    # Subscribe
    bus = get_event_bus()

    @bus.subscribe("training.completed")
    async def on_training_completed(event: TrainingCompleted):
        print(f"Training completed for {event.config_key}")

    # Publish
    await bus.publish(TrainingCompleted(
        topic="training.completed",
        config_key="square8_2p",
        metrics={"loss": 0.01},
    ))
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from re import Pattern
from typing import (
    Any,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "EventBus",
    "EventFilter",
    "EventHandler",
    "get_event_bus",
    "publish",
    "subscribe",
]

T = TypeVar("T", bound="Event")


# =============================================================================
# Event Base
# =============================================================================

@dataclass
class Event:
    """Base class for all events.

    Attributes:
        topic: Event topic/channel
        timestamp: When event was created
        source: Source of the event
        correlation_id: ID for correlating related events
        metadata: Additional event metadata
    """
    topic: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "topic": self.topic,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "type": self.__class__.__name__,
        }


# Common event types
@dataclass
class SystemEvent(Event):
    """System-level events."""
    level: str = "info"
    message: str = ""


@dataclass
class MetricEvent(Event):
    """Metric-related events."""
    metric_name: str = ""
    value: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LifecycleEvent(Event):
    """Component lifecycle events."""
    component: str = ""
    old_state: str = ""
    new_state: str = ""


@dataclass
class ErrorEvent(Event):
    """Error events."""
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""


# =============================================================================
# Event Filter
# =============================================================================

class EventFilter:
    """Filter for selecting events.

    Example:
        # Match exact topic
        filter = EventFilter(topic="training.completed")

        # Match topic pattern
        filter = EventFilter(topic_pattern="training.*")

        # Match with predicate
        filter = EventFilter(predicate=lambda e: e.source == "worker_1")
    """

    def __init__(
        self,
        topic: str | None = None,
        topic_pattern: str | None = None,
        event_type: type[Event] | None = None,
        source: str | None = None,
        predicate: Callable[[Event], bool] | None = None,
    ):
        """Initialize filter.

        Args:
            topic: Exact topic to match
            topic_pattern: Regex pattern for topic
            event_type: Event class to match
            source: Event source to match
            predicate: Custom filter function
        """
        self.topic = topic
        self.topic_pattern: Pattern | None = None
        if topic_pattern:
            self.topic_pattern = re.compile(topic_pattern)
        self.event_type = event_type
        self.source = source
        self.predicate = predicate

    def matches(self, event: Event) -> bool:
        """Check if event matches this filter."""
        if self.topic and event.topic != self.topic:
            return False

        if self.topic_pattern and not self.topic_pattern.match(event.topic):
            return False

        if self.event_type and not isinstance(event, self.event_type):
            return False

        if self.source and event.source != self.source:
            return False

        return not (self.predicate and not self.predicate(event))


# =============================================================================
# Event Handler
# =============================================================================

EventHandler = Union[
    Callable[[Event], None],
    Callable[[Event], Awaitable[None]],
]


@dataclass
class Subscription:
    """Represents an event subscription."""
    handler: EventHandler
    filter: EventFilter
    priority: int = 0
    once: bool = False
    weak: bool = False
    _handler_ref: weakref.ref | None = None

    def __post_init__(self):
        if self.weak:
            self._handler_ref = weakref.ref(self.handler)

    def get_handler(self) -> EventHandler | None:
        """Get the handler, resolving weak reference if needed."""
        if self.weak and self._handler_ref:
            return self._handler_ref()
        return self.handler


# =============================================================================
# Event Bus
# =============================================================================

class EventBus:
    """Central event bus for pub/sub communication.

    Thread-safe and supports both sync and async handlers.

    Example:
        bus = EventBus()

        # Subscribe to events
        @bus.subscribe("user.created")
        def on_user_created(event):
            print(f"User created: {event}")

        # Subscribe with filter
        @bus.subscribe(EventFilter(topic_pattern="training.*"))
        async def on_training_event(event):
            await process_training_event(event)

        # Publish events
        await bus.publish(Event(topic="user.created", metadata={"id": 123}))
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_history: bool = True,
    ):
        """Initialize event bus.

        Args:
            max_history: Maximum events to keep in history
            enable_history: Whether to record event history
        """
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pattern_subscriptions: list[Subscription] = []
        self._all_subscriptions: list[Subscription] = []
        self._lock = threading.RLock()
        self._history: list[Event] = []
        self._max_history = max_history
        self._enable_history = enable_history
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "delivery_errors": 0,
        }

    def subscribe(
        self,
        topic_or_filter: str | EventFilter | None = None,
        priority: int = 0,
        once: bool = False,
        weak: bool = False,
    ) -> Callable[[EventHandler], EventHandler]:
        """Subscribe to events.

        Can be used as a decorator or method.

        Args:
            topic_or_filter: Topic string, EventFilter, or None for all
            priority: Handler priority (higher = called first)
            once: Only receive one event
            weak: Use weak reference to handler

        Returns:
            Decorator function

        Example:
            @bus.subscribe("user.created")
            def handler(event):
                pass

            @bus.subscribe(EventFilter(topic_pattern="training.*"))
            async def async_handler(event):
                pass
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.add_subscription(
                handler,
                topic_or_filter,
                priority=priority,
                once=once,
                weak=weak,
            )
            return handler

        return decorator

    def add_subscription(
        self,
        handler: EventHandler,
        topic_or_filter: str | EventFilter | None = None,
        priority: int = 0,
        once: bool = False,
        weak: bool = False,
    ) -> Subscription:
        """Add a subscription programmatically.

        Args:
            handler: Event handler function
            topic_or_filter: Topic string, EventFilter, or None
            priority: Handler priority
            once: Only fire once
            weak: Use weak reference

        Returns:
            Subscription object
        """
        # Convert to filter
        if topic_or_filter is None:
            event_filter = EventFilter()
        elif isinstance(topic_or_filter, str):
            event_filter = EventFilter(topic=topic_or_filter)
        else:
            event_filter = topic_or_filter

        sub = Subscription(
            handler=handler,
            filter=event_filter,
            priority=priority,
            once=once,
            weak=weak,
        )

        with self._lock:
            if event_filter.topic:
                # Exact topic subscription
                self._subscriptions[event_filter.topic].append(sub)
                self._subscriptions[event_filter.topic].sort(
                    key=lambda s: -s.priority
                )
            elif event_filter.topic_pattern:
                # Pattern subscription
                self._pattern_subscriptions.append(sub)
                self._pattern_subscriptions.sort(key=lambda s: -s.priority)
            else:
                # All events subscription
                self._all_subscriptions.append(sub)
                self._all_subscriptions.sort(key=lambda s: -s.priority)

        logger.debug(f"Added subscription for {topic_or_filter}")
        return sub

    def unsubscribe(
        self,
        handler: EventHandler | None = None,
        topic: str | None = None,
    ) -> int:
        """Unsubscribe handler(s).

        Args:
            handler: Specific handler to remove
            topic: Remove all handlers for topic

        Returns:
            Number of subscriptions removed
        """
        removed = 0

        with self._lock:
            if topic:
                # Remove all for topic
                if topic in self._subscriptions:
                    if handler:
                        old_len = len(self._subscriptions[topic])
                        self._subscriptions[topic] = [
                            s for s in self._subscriptions[topic]
                            if s.handler != handler
                        ]
                        removed = old_len - len(self._subscriptions[topic])
                    else:
                        removed = len(self._subscriptions[topic])
                        del self._subscriptions[topic]
            elif handler:
                # Remove specific handler from all
                for topic_subs in self._subscriptions.values():
                    old_len = len(topic_subs)
                    topic_subs[:] = [s for s in topic_subs if s.handler != handler]
                    removed += old_len - len(topic_subs)

                old_len = len(self._pattern_subscriptions)
                self._pattern_subscriptions = [
                    s for s in self._pattern_subscriptions if s.handler != handler
                ]
                removed += old_len - len(self._pattern_subscriptions)

                old_len = len(self._all_subscriptions)
                self._all_subscriptions = [
                    s for s in self._all_subscriptions if s.handler != handler
                ]
                removed += old_len - len(self._all_subscriptions)

        return removed

    async def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: Event to publish

        Returns:
            Number of handlers that received the event
        """
        if not event.topic:
            raise ValueError("Event must have a topic")

        self._stats["events_published"] += 1

        # Record in history
        if self._enable_history:
            with self._lock:
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        # Get matching subscriptions
        subscriptions = self._get_matching_subscriptions(event)
        delivered = 0
        to_remove: list[Subscription] = []

        for sub in subscriptions:
            handler = sub.get_handler()
            if handler is None:
                to_remove.append(sub)
                continue

            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

                delivered += 1
                self._stats["events_delivered"] += 1

                if sub.once:
                    to_remove.append(sub)

            except Exception as e:
                self._stats["delivery_errors"] += 1
                logger.error(f"Error in event handler: {e}", exc_info=True)

        # Clean up one-time and dead subscriptions
        for sub in to_remove:
            self._remove_subscription(sub)

        return delivered

    def publish_sync(self, event: Event) -> int:
        """Publish an event synchronously.

        Only calls synchronous handlers.

        Args:
            event: Event to publish

        Returns:
            Number of handlers that received the event
        """
        if not event.topic:
            raise ValueError("Event must have a topic")

        self._stats["events_published"] += 1

        subscriptions = self._get_matching_subscriptions(event)
        delivered = 0

        for sub in subscriptions:
            handler = sub.get_handler()
            if handler is None or asyncio.iscoroutinefunction(handler):
                continue

            try:
                handler(event)
                delivered += 1
                self._stats["events_delivered"] += 1
            except Exception as e:
                self._stats["delivery_errors"] += 1
                logger.error(f"Error in sync event handler: {e}")

        return delivered

    def _get_matching_subscriptions(self, event: Event) -> list[Subscription]:
        """Get all subscriptions matching an event."""
        matching: list[Subscription] = []

        with self._lock:
            # Exact topic matches
            if event.topic in self._subscriptions:
                for sub in self._subscriptions[event.topic]:
                    if sub.filter.matches(event):
                        matching.append(sub)

            # Pattern matches
            for sub in self._pattern_subscriptions:
                if sub.filter.matches(event):
                    matching.append(sub)

            # All events
            for sub in self._all_subscriptions:
                if sub.filter.matches(event):
                    matching.append(sub)

        # Sort by priority
        matching.sort(key=lambda s: -s.priority)
        return matching

    def _remove_subscription(self, sub: Subscription) -> None:
        """Remove a subscription."""
        with self._lock:
            if sub.filter.topic:
                subs = self._subscriptions.get(sub.filter.topic, [])
                if sub in subs:
                    subs.remove(sub)
            elif sub.filter.topic_pattern:
                if sub in self._pattern_subscriptions:
                    self._pattern_subscriptions.remove(sub)
            else:
                if sub in self._all_subscriptions:
                    self._all_subscriptions.remove(sub)

    def get_history(
        self,
        topic: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get event history.

        Args:
            topic: Filter by topic
            limit: Maximum events to return

        Returns:
            List of events (most recent first)
        """
        with self._lock:
            events = list(self._history)

        if topic:
            events = [e for e in events if e.topic == topic]

        events.reverse()  # Most recent first
        return events[:limit]

    async def replay(
        self,
        handler: EventHandler,
        topic: str | None = None,
        since: float | None = None,
    ) -> int:
        """Replay historical events to a handler.

        Args:
            handler: Handler to receive events
            topic: Filter by topic
            since: Only events after this timestamp

        Returns:
            Number of events replayed
        """
        events = self.get_history(topic)
        if since:
            events = [e for e in events if e.timestamp >= since]

        events.reverse()  # Chronological order

        count = 0
        for event in events:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                count += 1
            except Exception as e:
                logger.error(f"Error replaying event: {e}")

        return count

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            return {
                **self._stats,
                "subscriptions": {
                    "by_topic": sum(len(s) for s in self._subscriptions.values()),
                    "by_pattern": len(self._pattern_subscriptions),
                    "all_events": len(self._all_subscriptions),
                },
                "history_size": len(self._history),
            }


# =============================================================================
# Global Instance
# =============================================================================

_event_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global event bus singleton.

    .. deprecated:: December 2025
        Use :func:`app.coordination.event_router.get_router` instead for unified
        event routing across all event systems.
    """
    import warnings
    warnings.warn(
        "get_event_bus() from core.event_bus is deprecated. "
        "Use get_router() from app.coordination.event_router instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _event_bus
    if _event_bus is None:
        with _bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    with _bus_lock:
        _event_bus = None


# Convenience functions using global bus
def subscribe(
    topic_or_filter: str | EventFilter | None = None,
    priority: int = 0,
    once: bool = False,
) -> Callable[[EventHandler], EventHandler]:
    """Subscribe to events on the global bus."""
    return get_event_bus().subscribe(topic_or_filter, priority, once)


async def publish(event: Event) -> int:
    """Publish an event to the global bus."""
    return await get_event_bus().publish(event)


def publish_sync(event: Event) -> int:
    """Publish an event synchronously to the global bus."""
    return get_event_bus().publish_sync(event)
