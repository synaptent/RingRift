"""Event Subscription Mixin - Re-export for backward compatibility.

DEPRECATED: This module is deprecated and will be archived in Q2 2026.
Use the event subscription patterns from handler_base.py or event_router.py directly.

Example migration:
    # Old pattern (deprecated):
    from app.coordination.event_subscription_mixin import EventSubscribingDaemonMixin

    class MyDaemon(BaseDaemon, EventSubscribingDaemonMixin):
        pass

    # New pattern (preferred):
    from app.coordination.handler_base import HandlerBase

    class MyDaemon(HandlerBase):
        def _get_event_subscriptions(self) -> dict[str, Callable]:
            return {"event_name": self._handler}

December 2025: Consolidated into handler_base.py.
"""

import warnings

warnings.warn(
    "app.coordination.event_subscription_mixin is deprecated. "
    "Use app.coordination.handler_base.HandlerBase instead, which includes "
    "built-in event subscription management via _get_event_subscriptions(). "
    "This module will be archived in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from archived location
from archive.deprecated_coordination._deprecated_event_subscription_mixin import (
    EventSubscribingDaemonMixin,
    create_event_subscribing_daemon,
)

__all__ = [
    "EventSubscribingDaemonMixin",
    "create_event_subscribing_daemon",
]
