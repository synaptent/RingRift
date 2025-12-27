"""DEPRECATED: Cross Process Events - Use event_router instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation
   (67 modules → 15). Cross-process events are now integrated into event_router.

Migration Guide:
   OLD:
      from app.coordination.cross_process_events import (
          publish_event,
          poll_events,
          subscribe_process,
      )

   NEW:
      from app.coordination.event_router import (
          publish,
          get_router,
      )

      # Publish - automatically routes to cross-process queue
      await publish(
          DataEventType.MODEL_PROMOTED,
          payload={"model": "hex8_2p"},
          source="promotion_daemon",
      )

      # Subscribe - receives from all event sources
      router = get_router()
      router.subscribe(DataEventType.MODEL_PROMOTED, my_handler)

The unified event router automatically handles cross-process routing,
deduplication, and integration with in-memory and stage events.

For direct cross-process queue access (advanced use only):
   from app.coordination.event_router import get_cross_process_queue

This wrapper will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "cross_process_events is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.event_router import publish, get_router' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export what we can from event_router
from app.coordination.event_router import (
    publish as publish_event,
    get_router,
    UnifiedEventRouter,
)

# For poll_events and subscribe_process, try to import from original
try:
    from app.coordination.cross_process_events import (
        poll_events,
        subscribe_process,
        ack_event,
        ack_events,
        get_event_queue,
    )
except ImportError:
    # If original module doesn't exist, provide stubs
    def poll_events(*args, **kwargs):
        warnings.warn(
            "poll_events() is deprecated. Use event_router.subscribe() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    def subscribe_process(*args, **kwargs):
        warnings.warn(
            "subscribe_process() is deprecated. Use event_router.subscribe() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    ack_event = None
    ack_events = None
    get_event_queue = None

__all__ = [
    "publish_event",
    "poll_events",
    "subscribe_process",
    "ack_event",
    "ack_events",
    "get_event_queue",
    "get_router",
    "UnifiedEventRouter",
]
