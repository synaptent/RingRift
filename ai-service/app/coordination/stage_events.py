"""Stage-based event system for pipeline orchestration.

.. deprecated:: December 2025
    This module is being superseded by the unified event router.
    For new code, prefer using:

        from app.coordination.event_router import (
            get_router, publish, subscribe
        )

    The unified router automatically routes to all event buses including
    stage events. This module remains functional for backwards compatibility.

This module provides an event bus for pipeline stage completion notifications,
enabling event-driven pipeline execution with immediate downstream triggering.

Consolidated from:
- scripts/archive/pipeline_orchestrator.py (StageEventBus, StageEvent, StageCompletionResult)

Usage:
    from app.coordination.stage_events import (
        StageEventBus,
        StageEvent,
        StageCompletionResult,
        get_event_bus,
    )

    # Subscribe to events
    bus = get_event_bus()

    async def on_selfplay_done(result):
        if result.success:
            await start_data_sync()

    bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_done)

    # Emit events when stages complete
    await bus.emit(StageCompletionResult(
        event=StageEvent.SELFPLAY_COMPLETE,
        success=True,
        iteration=1,
        timestamp=datetime.now().isoformat(),
        games_generated=500
    ))
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Jan 2026: Emit deprecation warning on import
warnings.warn(
    "app.coordination.stage_events is deprecated. "
    "Use app.coordination.event_router.get_router() instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


class StageEvent(Enum):
    """Events emitted when pipeline stages complete.

    These events allow downstream stages to be triggered immediately
    instead of relying on polling.
    """

    # Selfplay stages
    SELFPLAY_COMPLETE = "selfplay_complete"
    CANONICAL_SELFPLAY_COMPLETE = "canonical_selfplay_complete"
    GPU_SELFPLAY_COMPLETE = "gpu_selfplay_complete"

    # Data processing stages
    SYNC_COMPLETE = "sync_complete"
    PARITY_VALIDATION_COMPLETE = "parity_validation_complete"
    NPZ_EXPORT_STARTED = "npz_export_started"
    NPZ_EXPORT_COMPLETE = "npz_export_complete"

    # Training stages
    TRAINING_COMPLETE = "training_complete"
    TRAINING_STARTED = "training_started"
    TRAINING_FAILED = "training_failed"

    # Evaluation stages
    EVALUATION_COMPLETE = "evaluation_complete"
    SHADOW_TOURNAMENT_COMPLETE = "shadow_tournament_complete"
    ELO_CALIBRATION_COMPLETE = "elo_calibration_complete"

    # Optimization stages
    CMAES_COMPLETE = "cmaes_complete"
    PBT_COMPLETE = "pbt_complete"
    NAS_COMPLETE = "nas_complete"

    # Promotion stages
    PROMOTION_COMPLETE = "promotion_complete"
    TIER_GATING_COMPLETE = "tier_gating_complete"

    # Utility events
    ITERATION_COMPLETE = "iteration_complete"
    CLUSTER_SYNC_COMPLETE = "cluster_sync_complete"
    MODEL_SYNC_COMPLETE = "model_sync_complete"


@dataclass
class StageCompletionResult:
    """Data passed to completion callbacks when a stage finishes.

    Contains both success/failure status and relevant metrics from the stage.
    """

    event: StageEvent
    success: bool
    iteration: int
    timestamp: str
    board_type: str = "square8"
    num_players: int = 2

    # Selfplay metrics
    games_generated: int = 0

    # Training metrics
    model_path: str | None = None
    model_id: str | None = None
    train_loss: float | None = None
    val_loss: float | None = None

    # Evaluation metrics
    win_rate: float | None = None
    elo_delta: float | None = None

    # Promotion metrics
    promoted: bool = False
    promotion_reason: str | None = None

    # Error handling
    error: str | None = None
    error_details: str | None = None

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event": self.event.value,
            "success": self.success,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "games_generated": self.games_generated,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "win_rate": self.win_rate,
            "elo_delta": self.elo_delta,
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
            "error": self.error,
            "error_details": self.error_details,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageCompletionResult:
        """Create from dictionary."""
        event_value = data.get("event", "selfplay_complete")
        if isinstance(event_value, str):
            event = StageEvent(event_value)
        else:
            event = event_value

        return cls(
            event=event,
            success=data.get("success", False),
            iteration=data.get("iteration", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            board_type=data.get("board_type", "square8"),
            num_players=data.get("num_players", 2),
            games_generated=data.get("games_generated", 0),
            model_path=data.get("model_path"),
            model_id=data.get("model_id"),
            train_loss=data.get("train_loss"),
            val_loss=data.get("val_loss"),
            win_rate=data.get("win_rate"),
            elo_delta=data.get("elo_delta"),
            promoted=data.get("promoted", False),
            promotion_reason=data.get("promotion_reason"),
            error=data.get("error"),
            error_details=data.get("error_details"),
            metadata=data.get("metadata", {}),
        )


# Type alias for completion callbacks
StageCompletionCallback = Callable[[StageCompletionResult], Awaitable[None]]


class StageEventBus:
    """Event bus for pipeline stage completion notifications.

    Enables event-driven pipeline execution by allowing stages to register
    callbacks that fire immediately when upstream stages complete.

    Features:
    - Async callback support
    - Multiple subscribers per event
    - Event history tracking
    - Error isolation (one failing callback doesn't affect others)

    Example:
        bus = StageEventBus()

        async def on_selfplay_done(result):
            if result.success:
                await start_data_sync()

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_done)

        # Later, when selfplay completes:
        await bus.emit(StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp=datetime.now().isoformat(),
            games_generated=500
        ))
    """

    def __init__(self, max_history: int = 100):
        """Initialize the event bus.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: dict[StageEvent, list[StageCompletionCallback]] = {}
        self._log_callback: Callable[[str], None] | None = None
        self._history: list[StageCompletionResult] = []
        self._max_history = max_history
        self._callback_errors: list[dict[str, Any]] = []

    def set_logger(self, log_fn: Callable[[str], None]) -> None:
        """Set a logging function for event notifications.

        Args:
            log_fn: Function that accepts a log message string
        """
        self._log_callback = log_fn

    def subscribe(
        self,
        event: StageEvent,
        callback: StageCompletionCallback,
    ) -> None:
        """Register a callback for a stage completion event.

        Args:
            event: The event type to subscribe to
            callback: Async function to call when event is emitted
        """
        if event not in self._subscribers:
            self._subscribers[event] = []
        if callback not in self._subscribers[event]:
            self._subscribers[event].append(callback)
            logger.debug(f"Subscribed to {event.value}: {callback.__name__}")

    def unsubscribe(
        self,
        event: StageEvent,
        callback: StageCompletionCallback,
    ) -> bool:
        """Remove a callback from an event.

        Args:
            event: The event type
            callback: The callback to remove

        Returns:
            True if found and removed
        """
        if event in self._subscribers and callback in self._subscribers[event]:
            self._subscribers[event].remove(callback)
            logger.debug(f"Unsubscribed from {event.value}: {callback.__name__}")
            return True
        return False

    def clear_subscribers(self, event: StageEvent | None = None) -> int:
        """Clear all subscribers for an event, or all events if none specified.

        Args:
            event: Specific event to clear, or None for all events

        Returns:
            Number of subscribers removed
        """
        if event is not None:
            count = len(self._subscribers.get(event, []))
            self._subscribers[event] = []
            return count
        else:
            count = sum(len(callbacks) for callbacks in self._subscribers.values())
            self._subscribers.clear()
            return count

    async def emit(self, result: StageCompletionResult) -> int:
        """Emit a stage completion event to all subscribers.

        Args:
            result: The completion result to send to callbacks

        Returns:
            The number of callbacks invoked successfully
        """
        callbacks = self._subscribers.get(result.event, [])

        # Log the event
        status = "OK" if result.success else "FAILED"
        if self._log_callback:
            self._log_callback(
                f"[EVENT] {result.event.value} ({status}) - "
                f"invoking {len(callbacks)} callback(s)"
            )
        else:
            logger.info(
                f"Stage event {result.event.value} ({status}), "
                f"{len(callbacks)} subscribers"
            )

        # Track in history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Invoke callbacks
        invoked = 0
        for callback in callbacks:
            try:
                await callback(result)
                invoked += 1
            except Exception as e:
                error_info = {
                    "event": result.event.value,
                    "callback": callback.__name__,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                self._callback_errors.append(error_info)
                if len(self._callback_errors) > 100:
                    self._callback_errors.pop(0)

                if self._log_callback:
                    self._log_callback(
                        f"[EVENT] Callback error for {result.event.value}: {e}"
                    )
                else:
                    logger.error(
                        f"Callback {callback.__name__} failed for "
                        f"{result.event.value}: {e}"
                    )

                # Capture to dead letter queue if enabled (December 2025)
                if hasattr(self, "_dlq") and self._dlq is not None:
                    try:
                        payload = {
                            "event": result.event.value,
                            "success": result.success,
                            "iteration": result.iteration,
                            "timestamp": result.timestamp,
                            "metadata": result.metadata,
                        }
                        self._dlq.capture(
                            event_type=result.event.value,
                            payload=payload,
                            handler_name=callback.__name__,
                            error=str(e),
                            source="stage_events",
                        )
                    except Exception as dlq_error:
                        logger.warning(f"Failed to capture to DLQ: {dlq_error}")

        return invoked

    async def emit_and_wait(
        self,
        result: StageCompletionResult,
        timeout: float | None = None,
    ) -> list[Any]:
        """Emit event and wait for all callbacks to complete.

        Args:
            result: The completion result
            timeout: Optional timeout in seconds

        Returns:
            List of results from callbacks (None for failed callbacks)
        """
        callbacks = self._subscribers.get(result.event, [])
        if not callbacks:
            return []

        tasks = [callback(result) for callback in callbacks]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            return list(results)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {result.event.value} callbacks")
            return []

    def subscriber_count(self, event: StageEvent) -> int:
        """Get the number of subscribers for an event.

        Args:
            event: The event type

        Returns:
            Number of subscribed callbacks
        """
        return len(self._subscribers.get(event, []))

    def get_history(
        self,
        event: StageEvent | None = None,
        limit: int = 50,
    ) -> list[StageCompletionResult]:
        """Get event history, optionally filtered by event type.

        Args:
            event: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of recent events (newest first)
        """
        if event is not None:
            filtered = [r for r in self._history if r.event == event]
        else:
            filtered = list(self._history)

        return list(reversed(filtered[-limit:]))

    def get_callback_errors(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent callback errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error records
        """
        return list(reversed(self._callback_errors[-limit:]))

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics.

        Returns:
            Dict with subscriber counts, history size, error count
        """
        subscriber_counts = {
            event.value: len(callbacks)
            for event, callbacks in self._subscribers.items()
        }
        return {
            "total_subscribers": sum(subscriber_counts.values()),
            "subscribers_by_event": subscriber_counts,
            "history_size": len(self._history),
            "callback_errors": len(self._callback_errors),
            "supported_events": [e.value for e in StageEvent],
        }


# Global event bus singleton
_global_event_bus: StageEventBus | None = None


def get_event_bus() -> StageEventBus:
    """Get or create the global event bus.

    .. deprecated:: December 2025
        Use :func:`app.coordination.event_router.get_router` instead for unified
        event routing across all event systems.
    """
    import warnings
    warnings.warn(
        "get_event_bus() from stage_events is deprecated. "
        "Use get_router() from app.coordination.event_router instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = StageEventBus()
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_event_bus
    _global_event_bus = None


# ============================================================================
# Standard Pipeline Transition Callbacks
# ============================================================================


def create_pipeline_callbacks() -> dict[StageEvent, StageCompletionCallback]:
    """Create standard callbacks for pipeline stage transitions.

    These callbacks implement the typical pipeline flow:
    - SELFPLAY_COMPLETE -> trigger sync
    - SYNC_COMPLETE -> trigger parity validation
    - PARITY_VALIDATION_COMPLETE -> trigger NPZ export
    - NPZ_EXPORT_COMPLETE -> trigger training
    - TRAINING_COMPLETE -> trigger evaluation
    - EVALUATION_COMPLETE -> trigger promotion check

    Returns:
        Dict mapping events to their standard callbacks
    """
    callbacks: dict[StageEvent, StageCompletionCallback] = {}

    async def on_selfplay_complete(result: StageCompletionResult) -> None:
        """Standard handler for selfplay completion."""
        if result.success and result.games_generated > 0:
            logger.info(
                f"Selfplay complete: {result.games_generated} games, "
                f"ready for sync"
            )

    async def on_sync_complete(result: StageCompletionResult) -> None:
        """Standard handler for sync completion."""
        if result.success:
            logger.info("Sync complete, ready for parity validation")

    async def on_training_complete(result: StageCompletionResult) -> None:
        """Standard handler for training completion."""
        if result.success and result.model_path:
            logger.info(
                f"Training complete: {result.model_path}, ready for evaluation"
            )

    async def on_evaluation_complete(result: StageCompletionResult) -> None:
        """Standard handler for evaluation completion."""
        if result.success:
            logger.info(
                f"Evaluation complete: win_rate={result.win_rate}, "
                f"elo_delta={result.elo_delta}"
            )

    callbacks[StageEvent.SELFPLAY_COMPLETE] = on_selfplay_complete
    callbacks[StageEvent.SYNC_COMPLETE] = on_sync_complete
    callbacks[StageEvent.TRAINING_COMPLETE] = on_training_complete
    callbacks[StageEvent.EVALUATION_COMPLETE] = on_evaluation_complete

    return callbacks


def register_standard_callbacks(bus: StageEventBus | None = None) -> None:
    """Register standard pipeline callbacks on an event bus.

    Args:
        bus: Event bus to use (defaults to global bus)
    """
    bus = bus or get_event_bus()
    for event, callback in create_pipeline_callbacks().items():
        bus.subscribe(event, callback)


__all__ = [
    "StageCompletionCallback",
    "StageCompletionResult",
    "StageEvent",
    # Core classes
    "StageEventBus",
    # Pipeline helpers
    "create_pipeline_callbacks",
    # Global access
    "get_event_bus",
    "register_standard_callbacks",
    "reset_event_bus",
]
