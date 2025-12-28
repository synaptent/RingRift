"""Event Subscription Mixin for Daemons (December 2025).

Provides standardized event subscription/unsubscription lifecycle management
for daemons that need to react to coordination events.

This mixin consolidates 12+ similar patterns found across the codebase:
- auto_export_daemon.py
- training_trigger_daemon.py
- orphan_detection_daemon.py
- curriculum_integration.py
- resource_monitoring_coordinator.py
- unified_health_manager.py
- task_lifecycle_coordinator.py
- model_lifecycle_coordinator.py
- sync_router.py
- daemon_manager.py
- optimization_coordinator.py
- model_performance_watchdog.py

Usage:
    class MyDaemon(BaseDaemon, EventSubscribingDaemonMixin):
        def __init__(self):
            super().__init__("my_daemon")
            self._init_event_subscriptions()

        def _get_event_subscriptions(self) -> dict[str, Callable]:
            '''Return event_type -> handler mapping.'''
            return {
                DataEventType.TRAINING_COMPLETED.value: self._on_training_completed,
                DataEventType.SYNC_COMPLETED.value: self._on_sync_completed,
            }

        async def start(self) -> None:
            await super().start()
            self._subscribe_all_events()

        async def stop(self) -> None:
            self._unsubscribe_all_events()
            await super().stop()

        async def _on_training_completed(self, event: dict) -> None:
            # Handle event
            pass
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.event_router import UnifiedEventRouter

logger = logging.getLogger(__name__)


class EventSubscribingDaemonMixin:
    """Mixin providing standardized event subscription management.

    Features:
    - Centralized subscription storage
    - Automatic cleanup in stop()
    - Graceful error handling with fallback
    - Support for both sync and async handlers
    - Import-safe (works even if event router unavailable)

    Subclasses must implement:
    - _get_event_subscriptions() -> dict[str, Callable]
    """

    # Instance variables (set by _init_event_subscriptions)
    _event_subscriptions: dict[str, Any]  # event_type -> unsub_callback
    _event_subscribed: bool
    _event_router: "UnifiedEventRouter | None"

    def _init_event_subscriptions(self) -> None:
        """Initialize subscription tracking. Call in __init__."""
        self._event_subscriptions = {}
        self._event_subscribed = False
        self._event_router = None

    def _get_event_subscriptions(self) -> dict[str, Callable[[dict], Any]]:
        """Return event_type -> handler mapping.

        Subclasses should override this to specify which events to subscribe to.

        Returns:
            Dict mapping event type strings to handler callables.

        Example:
            def _get_event_subscriptions(self) -> dict[str, Callable]:
                return {
                    DataEventType.TRAINING_COMPLETED.value: self._on_training_completed,
                    "custom_event": self._on_custom_event,
                }
        """
        return {}

    def _get_conditional_subscriptions(self) -> dict[str, tuple[str, Callable[[dict], Any]]]:
        """Return conditional subscriptions that check for event type existence.

        Returns:
            Dict mapping attribute name to (event_type, handler) tuple.
            The attribute name is checked on DataEventType before subscribing.

        Example:
            def _get_conditional_subscriptions(self) -> dict:
                return {
                    # Only subscribe if DataEventType.SELFPLAY_RATE_CHANGED exists
                    "SELFPLAY_RATE_CHANGED": (
                        "selfplay_rate_changed",  # event type value
                        self._on_selfplay_rate_changed,
                    ),
                }
        """
        return {}

    def _subscribe_all_events(self) -> bool:
        """Subscribe to all events defined in _get_event_subscriptions().

        Returns:
            True if subscription succeeded, False otherwise.
        """
        if self._event_subscribed:
            return True

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            self._event_router = router

            # Subscribe to required events
            subscriptions = self._get_event_subscriptions()
            for event_type, handler in subscriptions.items():
                try:
                    unsub = router.subscribe(event_type, handler)
                    self._event_subscriptions[event_type] = unsub
                    logger.debug(f"[{self._get_daemon_name()}] Subscribed to {event_type}")
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(
                        f"[{self._get_daemon_name()}] Failed to subscribe to {event_type}: {e}"
                    )

            # Subscribe to conditional events
            conditional = self._get_conditional_subscriptions()
            for attr_name, (event_type, handler) in conditional.items():
                if hasattr(DataEventType, attr_name):
                    try:
                        actual_event = getattr(DataEventType, attr_name).value
                        unsub = router.subscribe(actual_event, handler)
                        self._event_subscriptions[actual_event] = unsub
                        logger.debug(
                            f"[{self._get_daemon_name()}] Subscribed to conditional {actual_event}"
                        )
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug(
                            f"[{self._get_daemon_name()}] Skipping conditional {attr_name}: {e}"
                        )

            self._event_subscribed = bool(self._event_subscriptions)
            if self._event_subscribed:
                logger.info(
                    f"[{self._get_daemon_name()}] Event subscriptions active "
                    f"({len(self._event_subscriptions)} events)"
                )
            return self._event_subscribed

        except ImportError as e:
            logger.debug(f"[{self._get_daemon_name()}] Event router not available: {e}")
            return False
        except (RuntimeError, OSError) as e:
            logger.warning(f"[{self._get_daemon_name()}] Event subscription failed: {e}")
            return False

    def _subscribe_single_event(
        self,
        event_type: str,
        handler: Callable[[dict], Any],
    ) -> bool:
        """Subscribe to a single event.

        Args:
            event_type: Event type string
            handler: Event handler callable

        Returns:
            True if subscription succeeded
        """
        try:
            if self._event_router is None:
                from app.coordination.event_router import get_router

                self._event_router = get_router()

            unsub = self._event_router.subscribe(event_type, handler)
            self._event_subscriptions[event_type] = unsub
            logger.debug(f"[{self._get_daemon_name()}] Subscribed to {event_type}")
            return True

        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(
                f"[{self._get_daemon_name()}] Failed to subscribe to {event_type}: {e}"
            )
            return False

    def _unsubscribe_single_event(self, event_type: str) -> bool:
        """Unsubscribe from a single event.

        Args:
            event_type: Event type string

        Returns:
            True if unsubscription succeeded or wasn't needed
        """
        unsub = self._event_subscriptions.pop(event_type, None)
        if unsub is None:
            return True  # Not subscribed, nothing to do

        try:
            if callable(unsub):
                unsub()
            logger.debug(f"[{self._get_daemon_name()}] Unsubscribed from {event_type}")
            return True
        except (TypeError, RuntimeError, OSError) as e:
            logger.debug(
                f"[{self._get_daemon_name()}] Error unsubscribing from {event_type}: {e}"
            )
            return False

    def _unsubscribe_all_events(self) -> None:
        """Unsubscribe from all registered events. Call in stop()."""
        if not self._event_subscribed and not self._event_subscriptions:
            return

        errors = 0
        for event_type, unsub in list(self._event_subscriptions.items()):
            try:
                if callable(unsub):
                    unsub()
                logger.debug(f"[{self._get_daemon_name()}] Unsubscribed from {event_type}")
            except (TypeError, RuntimeError, OSError) as e:
                errors += 1
                logger.debug(
                    f"[{self._get_daemon_name()}] Error unsubscribing from {event_type}: {e}"
                )

        self._event_subscriptions.clear()
        self._event_subscribed = False
        self._event_router = None

        if errors > 0:
            logger.debug(
                f"[{self._get_daemon_name()}] Completed unsubscribe with {errors} errors"
            )
        else:
            logger.debug(f"[{self._get_daemon_name()}] All event subscriptions cleared")

    def _get_daemon_name(self) -> str:
        """Get the daemon name for logging.

        Override in subclass if needed.
        """
        # Try common patterns for daemon name
        if hasattr(self, "name"):
            return getattr(self, "name")
        if hasattr(self, "_name"):
            return getattr(self, "_name")
        if hasattr(self, "daemon_type"):
            return str(getattr(self, "daemon_type"))
        return self.__class__.__name__

    @property
    def is_event_subscribed(self) -> bool:
        """Check if event subscriptions are active."""
        return getattr(self, "_event_subscribed", False)

    @property
    def active_subscription_count(self) -> int:
        """Get count of active subscriptions."""
        subscriptions = getattr(self, "_event_subscriptions", {})
        return len(subscriptions)

    def get_subscription_status(self) -> dict[str, Any]:
        """Get detailed subscription status for health checks.

        Returns:
            Dict with subscription health info
        """
        return {
            "subscribed": self.is_event_subscribed,
            "subscription_count": self.active_subscription_count,
            "subscribed_events": list(
                getattr(self, "_event_subscriptions", {}).keys()
            ),
            "router_available": getattr(self, "_event_router", None) is not None,
        }


# Convenience function for migration
def create_event_subscribing_daemon(
    base_class: type,
    event_subscriptions: dict[str, Callable[[dict], Any]],
    name: str | None = None,
) -> type:
    """Factory to create a daemon class with event subscription support.

    This is a migration helper for existing daemons that want to adopt
    the mixin pattern without full refactoring.

    Args:
        base_class: The existing daemon base class
        event_subscriptions: Event type -> handler mapping
        name: Optional daemon name

    Returns:
        New class with EventSubscribingDaemonMixin

    Example:
        MyEnhancedDaemon = create_event_subscribing_daemon(
            MyDaemon,
            {
                "training_completed": my_handler,
            },
            name="my_enhanced_daemon",
        )
    """

    class EnhancedDaemon(base_class, EventSubscribingDaemonMixin):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._init_event_subscriptions()
            self._stored_subscriptions = event_subscriptions
            if name:
                self._name = name

        def _get_event_subscriptions(self) -> dict[str, Callable[[dict], Any]]:
            return self._stored_subscriptions

    return EnhancedDaemon


__all__ = [
    "EventSubscribingDaemonMixin",
    "create_event_subscribing_daemon",
]
