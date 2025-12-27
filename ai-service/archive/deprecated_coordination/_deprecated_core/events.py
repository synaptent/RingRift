"""Unified event system (December 2025).

DEPRECATED: Import directly from app.coordination.event_router instead.
This module will be removed in Q2 2026.

Consolidates event-related functionality from event_router.py and stage_events.py.

Usage (DEPRECATED):
    from app.coordination.core.events import (
        UnifiedEventRouter,
        get_router,
        publish,
        subscribe,
    )

Recommended:
    from app.coordination.event_router import (
        UnifiedEventRouter,
        get_router,
        publish,
        subscribe,
    )
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.core.events is deprecated. "
    "Import from app.coordination.event_router instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from event_router
# Re-export from event_router (canonical source for stage event types)
from app.coordination.event_router import (
    DataEvent,
    DataEventType,
    StageCompletionResult,
    StageEvent,
    UnifiedEventRouter,
    emit_model_promoted,
    emit_selfplay_batch_completed,
    emit_training_completed,
    emit_training_started,
    get_event_bus,
    get_router,
    get_stage_event_bus,
    publish,
    publish_sync,
    subscribe,
    unsubscribe,
)

# StageEventBus still needs direct import (implementation detail)
from app.coordination.stage_events import (
    StageEvent as StageEventModel,
    StageEventBus,
)

__all__ = [  # noqa: RUF022
    # From event_router
    "UnifiedEventRouter",
    "get_router",
    "publish",
    "publish_sync",
    "subscribe",
    "unsubscribe",
    "DataEventType",
    "DataEvent",
    "StageEvent",
    "get_event_bus",
    "emit_training_completed",
    "emit_training_started",
    "emit_selfplay_batch_completed",
    "emit_model_promoted",
    # From stage_events
    "StageEventModel",
    "StageEventBus",
    "StageCompletionResult",
    "get_stage_event_bus",
]
