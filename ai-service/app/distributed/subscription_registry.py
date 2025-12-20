"""SubscriptionRegistry - Track and monitor event subscriptions (December 2025).

This module provides visibility into event bus subscriptions across the system.
It helps debug event flow issues by tracking:

1. Which components subscribe to which events
2. Subscription counts per event type
3. Handler metadata (function names, modules)
4. Subscription history

Usage:
    from app.distributed.subscription_registry import (
        SubscriptionRegistry,
        get_subscription_registry,
        track_subscription,
    )

    # Get singleton registry
    registry = get_subscription_registry()

    # Track a new subscription
    registry.track("MODEL_PROMOTED", "promotion_controller", handler_name="on_promoted")

    # Get subscription stats
    stats = registry.get_stats()
    print(f"Total subscriptions: {stats['total_subscriptions']}")

    # Get subscribers for an event type
    subscribers = registry.get_subscribers("MODEL_PROMOTED")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubscriptionInfo:
    """Information about a single subscription."""

    event_type: str
    subscriber_name: str
    handler_name: str = ""
    module_name: str = ""
    subscribed_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_called_at: float = 0.0
    errors: int = 0


@dataclass
class SubscriptionStats:
    """Aggregate subscription statistics."""

    total_subscriptions: int = 0
    event_types_subscribed: int = 0
    unique_subscribers: int = 0
    total_calls: int = 0
    total_errors: int = 0
    active_since: float = 0.0


class SubscriptionRegistry:
    """Registry tracking all event subscriptions.

    Provides visibility into which components are subscribed to which events,
    helping debug event flow issues and monitor system integration.
    """

    _instance: SubscriptionRegistry | None = None

    def __init__(self):
        """Initialize the subscription registry."""
        # Subscriptions indexed by event type
        self._by_event: dict[str, list[SubscriptionInfo]] = {}

        # Subscriptions indexed by subscriber name
        self._by_subscriber: dict[str, list[SubscriptionInfo]] = {}

        # All subscriptions for lookup
        self._subscriptions: list[SubscriptionInfo] = []

        # Set of unique subscribers
        self._unique_subscribers: set[str] = set()

        # Total call counts
        self._total_calls: int = 0
        self._total_errors: int = 0

        # Tracking start time
        self._started_at: float = time.time()

        # Hook into event bus if available
        self._hooked: bool = False

    @classmethod
    def get_instance(cls) -> SubscriptionRegistry:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def track(
        self,
        event_type: str,
        subscriber_name: str,
        handler: Callable | None = None,
        handler_name: str = "",
        module_name: str = "",
    ) -> SubscriptionInfo:
        """Track a new subscription.

        Args:
            event_type: Event type being subscribed to
            subscriber_name: Name of the subscribing component
            handler: Optional handler function (used to extract name/module)
            handler_name: Handler function name (if handler not provided)
            module_name: Module name (if handler not provided)

        Returns:
            SubscriptionInfo for the new subscription
        """
        # Extract handler info if provided
        if handler is not None:
            handler_name = handler_name or getattr(handler, '__name__', 'unknown')
            module_name = module_name or getattr(handler, '__module__', 'unknown')

        info = SubscriptionInfo(
            event_type=event_type,
            subscriber_name=subscriber_name,
            handler_name=handler_name,
            module_name=module_name,
        )

        # Index by event type
        if event_type not in self._by_event:
            self._by_event[event_type] = []
        self._by_event[event_type].append(info)

        # Index by subscriber
        if subscriber_name not in self._by_subscriber:
            self._by_subscriber[subscriber_name] = []
        self._by_subscriber[subscriber_name].append(info)

        # Add to full list
        self._subscriptions.append(info)
        self._unique_subscribers.add(subscriber_name)

        logger.debug(
            f"[SubscriptionRegistry] Tracked: {subscriber_name} -> {event_type} "
            f"(handler: {handler_name})"
        )

        return info

    def untrack(
        self,
        event_type: str,
        subscriber_name: str,
        handler_name: str | None = None,
    ) -> bool:
        """Remove a subscription from tracking.

        Args:
            event_type: Event type
            subscriber_name: Subscriber name
            handler_name: Optional handler name for exact match

        Returns:
            True if subscription was found and removed
        """
        # Find matching subscription
        found = None
        for sub in self._by_event.get(event_type, []):
            if sub.subscriber_name == subscriber_name:
                if handler_name is None or sub.handler_name == handler_name:
                    found = sub
                    break

        if found is None:
            return False

        # Remove from all indices
        self._by_event[event_type].remove(found)
        self._by_subscriber[subscriber_name].remove(found)
        self._subscriptions.remove(found)

        # Clean up empty entries
        if not self._by_event[event_type]:
            del self._by_event[event_type]
        if not self._by_subscriber[subscriber_name]:
            del self._by_subscriber[subscriber_name]
            self._unique_subscribers.discard(subscriber_name)

        logger.debug(
            f"[SubscriptionRegistry] Untracked: {subscriber_name} -> {event_type}"
        )

        return True

    def record_call(
        self,
        event_type: str,
        subscriber_name: str,
        success: bool = True,
    ) -> None:
        """Record that a subscription handler was called.

        Args:
            event_type: Event type that was delivered
            subscriber_name: Subscriber that received it
            success: Whether the handler succeeded
        """
        for sub in self._by_event.get(event_type, []):
            if sub.subscriber_name == subscriber_name:
                sub.call_count += 1
                sub.last_called_at = time.time()
                if not success:
                    sub.errors += 1
                    self._total_errors += 1
                self._total_calls += 1
                return

    def get_subscribers(self, event_type: str) -> list[SubscriptionInfo]:
        """Get all subscribers for an event type."""
        return list(self._by_event.get(event_type, []))

    def get_subscriptions_by_component(self, subscriber_name: str) -> list[SubscriptionInfo]:
        """Get all subscriptions for a component."""
        return list(self._by_subscriber.get(subscriber_name, []))

    def get_all_event_types(self) -> list[str]:
        """Get all event types that have subscribers."""
        return list(self._by_event.keys())

    def get_all_subscribers(self) -> list[str]:
        """Get all subscriber names."""
        return list(self._unique_subscribers)

    def get_stats(self) -> SubscriptionStats:
        """Get aggregate subscription statistics."""
        return SubscriptionStats(
            total_subscriptions=len(self._subscriptions),
            event_types_subscribed=len(self._by_event),
            unique_subscribers=len(self._unique_subscribers),
            total_calls=self._total_calls,
            total_errors=self._total_errors,
            active_since=self._started_at,
        )

    def get_status(self) -> dict[str, Any]:
        """Get registry status for monitoring."""
        stats = self.get_stats()

        # Get event type counts
        event_counts = {
            event_type: len(subs)
            for event_type, subs in self._by_event.items()
        }

        # Get subscriber counts
        subscriber_counts = {
            name: len(subs)
            for name, subs in self._by_subscriber.items()
        }

        return {
            "total_subscriptions": stats.total_subscriptions,
            "event_types_subscribed": stats.event_types_subscribed,
            "unique_subscribers": stats.unique_subscribers,
            "total_calls": stats.total_calls,
            "total_errors": stats.total_errors,
            "active_since": stats.active_since,
            "uptime_seconds": time.time() - stats.active_since,
            "event_type_counts": event_counts,
            "subscriber_counts": subscriber_counts,
        }

    def get_subscription_matrix(self) -> dict[str, dict[str, bool]]:
        """Get a matrix of which subscribers listen to which events.

        Returns:
            Dict mapping event_type -> {subscriber_name -> True}
        """
        matrix = {}
        for event_type, subs in self._by_event.items():
            matrix[event_type] = {
                sub.subscriber_name: True
                for sub in subs
            }
        return matrix

    def format_report(self) -> str:
        """Format a human-readable subscription report."""
        lines = ["=" * 60]
        lines.append("SUBSCRIPTION REGISTRY REPORT")
        lines.append("=" * 60)

        stats = self.get_stats()
        lines.append(f"Total subscriptions: {stats.total_subscriptions}")
        lines.append(f"Event types: {stats.event_types_subscribed}")
        lines.append(f"Unique subscribers: {stats.unique_subscribers}")
        lines.append(f"Total calls: {stats.total_calls}")
        lines.append(f"Total errors: {stats.total_errors}")
        lines.append("")

        lines.append("By Event Type:")
        lines.append("-" * 40)
        for event_type in sorted(self._by_event.keys()):
            subs = self._by_event[event_type]
            lines.append(f"  {event_type}: {len(subs)} subscriber(s)")
            for sub in subs:
                calls = f"calls={sub.call_count}" if sub.call_count > 0 else "no calls"
                lines.append(f"    - {sub.subscriber_name}.{sub.handler_name} ({calls})")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def hook_event_bus(self) -> bool:
        """Hook into the event bus to automatically track subscriptions.

        Returns:
            True if successfully hooked
        """
        if self._hooked:
            return True

        try:
            from app.distributed.data_events import get_event_bus

            bus = get_event_bus()

            # Store original subscribe method
            original_subscribe = bus.subscribe

            # Create wrapped subscribe that tracks
            def tracked_subscribe(event_type, handler, *args, **kwargs):
                # Get subscriber name from handler or caller
                subscriber_name = getattr(handler, '__qualname__', 'unknown')
                if '.' in subscriber_name:
                    subscriber_name = subscriber_name.rsplit('.', 1)[0]

                # Track the subscription
                self.track(
                    event_type=event_type.value if hasattr(event_type, 'value') else str(event_type),
                    subscriber_name=subscriber_name,
                    handler=handler,
                )

                # Call original
                return original_subscribe(event_type, handler, *args, **kwargs)

            # Replace subscribe method
            bus.subscribe = tracked_subscribe
            self._hooked = True

            logger.info("[SubscriptionRegistry] Hooked into event bus")
            return True

        except Exception as e:
            logger.warning(f"[SubscriptionRegistry] Failed to hook event bus: {e}")
            return False


# Singleton access
_registry: SubscriptionRegistry | None = None


def get_subscription_registry() -> SubscriptionRegistry:
    """Get the global subscription registry singleton."""
    global _registry
    if _registry is None:
        _registry = SubscriptionRegistry.get_instance()
    return _registry


def track_subscription(
    event_type: str,
    subscriber_name: str,
    handler: Callable | None = None,
    **kwargs,
) -> SubscriptionInfo:
    """Convenience function to track a subscription."""
    return get_subscription_registry().track(
        event_type=event_type,
        subscriber_name=subscriber_name,
        handler=handler,
        **kwargs,
    )


def get_subscription_stats() -> SubscriptionStats:
    """Convenience function to get subscription stats."""
    return get_subscription_registry().get_stats()


__all__ = [
    "SubscriptionInfo",
    "SubscriptionRegistry",
    "SubscriptionStats",
    "get_subscription_registry",
    "get_subscription_stats",
    "track_subscription",
]
