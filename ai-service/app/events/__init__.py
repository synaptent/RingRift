"""RingRift Events Package - Unified event types and utilities.

This package provides the single source of truth for all event types
used in the RingRift training infrastructure.

December 2025: Created for Phase 2 consolidation.

Usage:
    from app.events import RingRiftEventType, EventCategory

    # Use unified event types
    if event.event_type == RingRiftEventType.MODEL_PROMOTED:
        handle_promotion(event)

    # Check event category
    category = EventCategory.from_event(event.event_type)
    if category == EventCategory.TRAINING:
        log_training_event(event)
"""

from app.events.types import (
    CROSS_PROCESS_EVENT_TYPES,
    DataEventType,
    EventCategory,
    RingRiftEventType,
    StageEvent,
    get_events_by_category,
    is_cross_process_event,
)

__all__ = [
    # Main enum
    "RingRiftEventType",
    # Category support
    "EventCategory",
    "get_events_by_category",
    # Cross-process support
    "CROSS_PROCESS_EVENT_TYPES",
    "is_cross_process_event",
    # Backwards compatibility
    "DataEventType",
    "StageEvent",
]


def __dir__() -> list[str]:
    """Expose the intended events surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
