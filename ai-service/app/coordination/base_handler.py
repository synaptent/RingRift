"""Base classes for event handlers in the coordination system.

This module provides common functionality for all event handlers, reducing
code duplication across 60+ handler implementations.

December 2025: Created as part of code consolidation effort.
Saves ~75-110 LOC per handler through shared initialization, subscription,
statistics tracking, and singleton patterns.

Usage:
    from app.coordination.base_handler import BaseEventHandler, BaseSingletonHandler

    class MyHandler(BaseEventHandler):
        def __init__(self, custom_param: int = 10):
            super().__init__("MyHandler")
            self.custom_param = custom_param

        def _do_subscribe(self) -> bool:
            from app.coordination.event_router import get_event_bus
            bus = get_event_bus()
            bus.subscribe(EventType.MY_EVENT, self._handle_event)
            self._subscribed = True
            return True

        async def _handle_event(self, event: dict) -> None:
            # Business logic here
            self._record_success()

    # For singleton handlers:
    class MySingletonHandler(BaseSingletonHandler):
        _instance: ClassVar["MySingletonHandler | None"] = None

        @classmethod
        def get_instance(cls, **kwargs) -> "MySingletonHandler":
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

        @classmethod
        def reset_singleton(cls) -> None:
            if cls._instance is not None:
                cls._instance.unsubscribe()
            cls._instance = None
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class HandlerStats:
    """Statistics for an event handler.

    Provides consistent statistics tracking across all handlers.
    """

    subscribed: bool = False
    events_processed: int = 0
    success_count: int = 0
    error_count: int = 0
    last_event_time: float = 0.0
    last_error_time: float = 0.0
    last_error: str = ""
    custom_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.events_processed == 0:
            return 1.0
        return self.success_count / self.events_processed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "subscribed": self.subscribed,
            "events_processed": self.events_processed,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 3),
            "last_event_time": self.last_event_time,
            "last_error_time": self.last_error_time,
            "last_error": self.last_error,
        }
        result.update(self.custom_stats)
        return result


class BaseEventHandler(ABC):
    """Base class for all event handlers in the coordination system.

    Provides common initialization, subscription, statistics tracking,
    and cleanup patterns to reduce code duplication across 60+ handlers.

    Subclasses implement only:
    - __init__() to set custom params (call super().__init__(name))
    - _do_subscribe() for subscription logic
    - _handle_event() for business logic
    - _validate_event() for optional validation (default: True)
    - get_stats() can call super() to include base stats
    """

    def __init__(self, handler_name: str, emit_metrics: bool = True) -> None:
        """Initialize handler with common attributes.

        Args:
            handler_name: Name for logging (e.g., "SyncStallHandler")
            emit_metrics: Whether to emit event metrics (for future use)
        """
        self.handler_name = handler_name
        self.emit_metrics = emit_metrics

        # Subscription state
        self._subscribed = False

        # Statistics tracking
        self._stats = HandlerStats()

        # Timing
        self._started_at = time.time()

        logger.info(f"[{handler_name}] Initialized")

    @property
    def is_subscribed(self) -> bool:
        """Check if handler is actively subscribed."""
        return self._subscribed

    def subscribe(self) -> bool:
        """Subscribe to events.

        Returns:
            True if subscription successful, False otherwise
        """
        if self._subscribed:
            return True

        try:
            result = self._do_subscribe()
            if result:
                self._stats.subscribed = True
                logger.info(f"[{self.handler_name}] Subscribed successfully")
            return result
        except ImportError as e:
            logger.debug(f"[{self.handler_name}] Event system unavailable: {e}")
            return False
        except Exception as e:
            logger.error(f"[{self.handler_name}] Subscribe failed: {e}")
            return False

    @abstractmethod
    def _do_subscribe(self) -> bool:
        """Implement subscription logic (called by subscribe()).

        Must set self._subscribed = True and return True on success.

        Example:
            from app.coordination.event_router import get_event_bus
            bus = get_event_bus()
            bus.subscribe(DataEventType.SYNC_STALLED, self._handle_event)
            self._subscribed = True
            return True
        """

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if self._subscribed:
            try:
                self._do_unsubscribe()
                self._subscribed = False
                self._stats.subscribed = False
                logger.info(f"[{self.handler_name}] Unsubscribed")
            except Exception as e:
                logger.warning(f"[{self.handler_name}] Unsubscribe failed: {e}")

    def _do_unsubscribe(self) -> None:
        """Implement unsubscription logic. Override if needed.

        Example:
            bus = get_event_bus()
            bus.unsubscribe(DataEventType.SYNC_STALLED, self._handle_event)
        """

    @abstractmethod
    async def _handle_event(self, event: Any) -> None:
        """Handle an event. Subclasses implement business logic.

        Args:
            event: The event object (dict or typed Event)

        After processing, call:
        - self._record_success() on success
        - self._record_error(error) on failure
        """

    def _validate_event(self, event: Any) -> bool:
        """Validate event before processing. Override if needed.

        Args:
            event: The event to validate

        Returns:
            True if event is valid, False to skip processing
        """
        return True

    def _get_payload(self, event: Any) -> dict[str, Any]:
        """Extract payload from event (handles Event objects and dicts).

        Args:
            event: Event object or dict

        Returns:
            The event payload as a dictionary
        """
        if hasattr(event, "payload"):
            return event.payload if isinstance(event.payload, dict) else {}
        if isinstance(event, dict):
            return event
        return {}

    def _record_success(self, duration_ms: float = 0.0) -> None:
        """Record successful event handling."""
        self._stats.events_processed += 1
        self._stats.success_count += 1
        self._stats.last_event_time = time.time()

    def _record_error(self, error: str | Exception) -> None:
        """Record event handling error."""
        self._stats.events_processed += 1
        self._stats.error_count += 1
        self._stats.last_error_time = time.time()
        self._stats.last_error = str(error)
        logger.error(f"[{self.handler_name}] Error #{self._stats.error_count}: {error}")

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics.

        Subclasses can override to add custom stats:
            stats = super().get_stats()
            stats["my_custom_stat"] = self.my_value
            return stats

        Returns:
            Dict with subscription status, event counts, rates, last times
        """
        return self._stats.to_dict()

    def add_custom_stat(self, key: str, value: Any) -> None:
        """Add a custom statistic that will be included in get_stats().

        Args:
            key: Stat name
            value: Stat value
        """
        self._stats.custom_stats[key] = value

    def reset(self) -> None:
        """Reset handler state (for testing). Subclasses call super().reset()."""
        self._stats = HandlerStats(subscribed=self._subscribed)
        logger.debug(f"[{self.handler_name}] State reset")

    @property
    def uptime_seconds(self) -> float:
        """Get handler uptime in seconds."""
        return time.time() - self._started_at


class BaseSingletonHandler(BaseEventHandler):
    """Base class for singleton event handlers.

    Provides module-level singleton management pattern used by 10+ handlers.

    Subclasses must define:
        _instance: ClassVar[ClassName | None] = None

    And implement:
        @classmethod
        def get_instance(cls, **kwargs) -> ClassName: ...

        @classmethod
        def reset_singleton(cls) -> None: ...

    Example:
        class MyHandler(BaseSingletonHandler):
            _instance: ClassVar["MyHandler | None"] = None
            _lock: ClassVar[threading.Lock] = threading.Lock()

            @classmethod
            def get_instance(cls, **kwargs) -> "MyHandler":
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = cls(**kwargs)
                    return cls._instance

            @classmethod
            def reset_singleton(cls) -> None:
                with cls._lock:
                    if cls._instance is not None:
                        cls._instance.unsubscribe()
                    cls._instance = None
    """

    # Subclasses override these
    _instance: ClassVar[Any] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_instance(cls, **kwargs: Any) -> "BaseSingletonHandler":
        """Get or create singleton instance.

        Args:
            **kwargs: Constructor arguments

        Returns:
            Singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unsubscribe()
            cls._instance = None

    @classmethod
    def has_instance(cls) -> bool:
        """Check if singleton instance exists."""
        return cls._instance is not None


class MultiEventHandler(BaseEventHandler):
    """Base class for handlers that subscribe to multiple event types.

    Provides a mapping from event types to handler methods.

    Example:
        class PipelineHandler(MultiEventHandler):
            def __init__(self):
                super().__init__("PipelineHandler")
                self._event_handlers = {
                    DataEventType.SYNC_COMPLETED: self._on_sync_completed,
                    DataEventType.EXPORT_COMPLETED: self._on_export_completed,
                }

            def _do_subscribe(self) -> bool:
                from app.coordination.event_router import get_event_bus
                bus = get_event_bus()
                for event_type, handler in self._event_handlers.items():
                    bus.subscribe(event_type, handler)
                self._subscribed = True
                return True
    """

    def __init__(self, handler_name: str, emit_metrics: bool = True) -> None:
        """Initialize multi-event handler."""
        super().__init__(handler_name, emit_metrics)
        # Subclasses populate this mapping
        self._event_handlers: dict[Any, Any] = {}

    async def _handle_event(self, event: Any) -> None:
        """Route event to appropriate handler based on type.

        Override in subclasses for custom routing logic.
        """
        event_type = getattr(event, "type", None) or event.get("type")
        handler = self._event_handlers.get(event_type)
        if handler:
            await handler(event)
        else:
            logger.warning(f"[{self.handler_name}] No handler for event type: {event_type}")

    def _do_unsubscribe(self) -> None:
        """Unsubscribe from all registered event types."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            for event_type, handler in self._event_handlers.items():
                try:
                    bus.unsubscribe(event_type, handler)
                except Exception:
                    pass  # Continue unsubscribing others
        except ImportError:
            pass


# Convenience functions for handler patterns


def create_handler_stats(**custom: Any) -> HandlerStats:
    """Create a HandlerStats instance with optional custom stats.

    Args:
        **custom: Custom statistics to include

    Returns:
        HandlerStats instance
    """
    stats = HandlerStats()
    stats.custom_stats.update(custom)
    return stats


def safe_subscribe(
    handler: BaseEventHandler,
    fallback: bool = False,
) -> bool:
    """Safely subscribe a handler, catching all exceptions.

    Args:
        handler: The handler to subscribe
        fallback: Value to return on failure

    Returns:
        True on success, fallback value on failure
    """
    try:
        return handler.subscribe()
    except Exception as e:
        logger.error(f"Failed to subscribe {handler.handler_name}: {e}")
        return fallback


__all__ = [
    "BaseEventHandler",
    "BaseSingletonHandler",
    "HandlerStats",
    "MultiEventHandler",
    "create_handler_stats",
    "safe_subscribe",
]
