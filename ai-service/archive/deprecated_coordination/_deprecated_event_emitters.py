"""DEPRECATED: Event Emitters - Use event_router instead.

.. deprecated:: December 2025
   This module is deprecated as part of the coordination module consolidation
   (67 modules → 15). Event emission is now handled by the unified event router.

Migration Guide:
   OLD:
      from app.coordination.event_emitters import (
          emit_training_completed,
          emit_model_promoted,
      )

   NEW:
      from app.coordination.event_router import (
          emit_training_completed,
          emit_model_promoted,
      )

   # Or use the unified publish API:
      from app.coordination.event_router import publish, DataEventType

      await publish(
          DataEventType.TRAINING_COMPLETED,
          payload={"config": "hex8_2p"},
          source="training_daemon",
      )

The unified event router provides better integration across in-memory,
stage-based, and cross-process events with automatic routing.

This wrapper will be removed in Q2 2026.
"""

import warnings

warnings.warn(
    "event_emitters is deprecated and will be removed in Q2 2026. "
    "Use 'from app.coordination.event_router import emit_*' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from event_router
from app.coordination.event_router import (
    emit_training_completed,
    emit_training_started,
    emit_selfplay_batch_completed,
    emit_model_promoted,
    publish,
    DataEventType,
)

__all__ = [
    "emit_training_completed",
    "emit_training_started",
    "emit_selfplay_batch_completed",
    "emit_model_promoted",
    "publish",
    "DataEventType",
]
