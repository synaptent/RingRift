"""Data Pipeline Event System.

This module provides an event bus for coordinating components of the
AI self-improvement loop. Events allow loose coupling between:
- Data collection
- Training triggers
- Evaluation
- Model promotion
- Curriculum rebalancing

Usage:
    from app.distributed.data_events import DataEventType, DataEvent, get_event_bus

    # Subscribe to events
    bus = get_event_bus()
    bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handle_new_games)

    # Publish events
    await bus.publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={"host": "gh200-a", "new_games": 500}
    ))
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

# Global singleton instance
_event_bus: Optional["EventBus"] = None


class DataEventType(Enum):
    """Types of data pipeline events."""

    # Data collection events
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_STARTED = "sync_started"
    DATA_SYNC_COMPLETED = "sync_completed"
    DATA_SYNC_FAILED = "sync_failed"

    # Training events
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    ELO_UPDATED = "elo_updated"

    # Promotion events
    PROMOTION_CANDIDATE = "promotion_candidate"
    PROMOTION_STARTED = "promotion_started"
    MODEL_PROMOTED = "model_promoted"
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_REJECTED = "promotion_rejected"

    # Curriculum events
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    WEIGHT_UPDATED = "weight_updated"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"  # Triggers curriculum rebalance

    # Optimization events
    CMAES_TRIGGERED = "cmaes_triggered"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"

    # System events
    DAEMON_STARTED = "daemon_started"
    DAEMON_STOPPED = "daemon_stopped"
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    ERROR = "error"


@dataclass
class DataEvent:
    """A data pipeline event."""

    event_type: DataEventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataEvent":
        """Create from dictionary."""
        return cls(
            event_type=DataEventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", ""),
        )


EventCallback = Callable[[DataEvent], Union[None, asyncio.Future]]


class EventBus:
    """Async event bus for component coordination.

    Supports both sync and async callbacks. Events are delivered
    in order of subscription.
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[DataEventType, List[EventCallback]] = {}
        self._global_subscribers: List[EventCallback] = []
        self._event_history: List[DataEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        event_type: Optional[DataEventType],
        callback: EventCallback,
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: Specific event type, or None for all events
            callback: Function to call when event occurs. Can be sync or async.
        """
        if event_type is None:
            self._global_subscribers.append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: Optional[DataEventType],
        callback: EventCallback,
    ) -> bool:
        """Unsubscribe from events.

        Returns True if callback was found and removed.
        """
        if event_type is None:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
                return True
        else:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    return True
        return False

    async def publish(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event to all subscribers.

        Callbacks are invoked in order. Async callbacks are awaited.
        Errors in callbacks are logged but don't prevent delivery to
        other subscribers.

        Args:
            event: The event to publish
            bridge_cross_process: If True, also bridge to cross-process queue
        """
        async with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Bridge to cross-process queue for multi-daemon coordination
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        # Get all callbacks for this event
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Invoke each callback
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[EventBus] Error in subscriber for {event.event_type.value}: {e}")

    def publish_sync(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event synchronously (non-async context).

        Creates a new event loop if needed. Use publish() when possible.
        """
        # Bridge immediately in sync context (doesn't need async)
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        try:
            loop = asyncio.get_running_loop()
            # Don't bridge again in the async publish
            loop.create_task(self.publish(event, bridge_cross_process=False))
        except RuntimeError:
            # No running loop - run synchronously
            asyncio.run(self.publish(event, bridge_cross_process=False))

    def get_history(
        self,
        event_type: Optional[DataEventType] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[DataEvent]:
        """Get recent events from history.

        Args:
            event_type: Filter by event type (None for all)
            since: Only events after this timestamp
            limit: Maximum number of events to return
        """
        events = self._event_history

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if since is not None:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    _event_bus = None


# Cross-process bridging configuration
# Events that should be propagated to cross-process queue for multi-daemon coordination
CROSS_PROCESS_EVENT_TYPES = {
    DataEventType.MODEL_PROMOTED,
    DataEventType.TRAINING_STARTED,
    DataEventType.TRAINING_COMPLETED,
    DataEventType.TRAINING_FAILED,
    DataEventType.EVALUATION_COMPLETED,
    DataEventType.ELO_SIGNIFICANT_CHANGE,
    DataEventType.DAEMON_STARTED,
    DataEventType.DAEMON_STOPPED,
    DataEventType.HOST_ONLINE,
    DataEventType.HOST_OFFLINE,
}


def _bridge_to_cross_process(event: DataEvent) -> None:
    """Bridge event to cross-process queue if it's a cross-process event type.

    This allows events published to the in-memory EventBus to also be
    visible to other daemon processes via the SQLite-backed queue.
    """
    if event.event_type not in CROSS_PROCESS_EVENT_TYPES:
        return

    try:
        from app.coordination.cross_process_events import bridge_to_cross_process
        bridge_to_cross_process(event.event_type.value, event.payload, event.source)
    except Exception as e:
        # Don't fail the main event if cross-process bridging fails
        print(f"[EventBus] Cross-process bridge failed: {e}")


# Convenience functions for common events


async def emit_new_games(host: str, new_games: int, total_games: int, source: str = "") -> None:
    """Emit a NEW_GAMES_AVAILABLE event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={
            "host": host,
            "new_games": new_games,
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_training_threshold(config: str, games: int, source: str = "") -> None:
    """Emit a TRAINING_THRESHOLD_REACHED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
        payload={
            "config": config,
            "games": games,
        },
        source=source,
    ))


async def emit_training_completed(
    config: str,
    success: bool,
    duration: float,
    model_path: Optional[str] = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={
            "config": config,
            "success": success,
            "duration": duration,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_evaluation_completed(
    config: str,
    elo: float,
    games_played: int,
    win_rate: float,
    source: str = "",
) -> None:
    """Emit an EVALUATION_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_COMPLETED,
        payload={
            "config": config,
            "elo": elo,
            "games_played": games_played,
            "win_rate": win_rate,
        },
        source=source,
    ))


async def emit_model_promoted(
    model_id: str,
    config: str,
    elo: float,
    elo_gain: float,
    source: str = "",
) -> None:
    """Emit a MODEL_PROMOTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_PROMOTED,
        payload={
            "model_id": model_id,
            "config": config,
            "elo": elo,
            "elo_gain": elo_gain,
        },
        source=source,
    ))


async def emit_error(
    component: str,
    error: str,
    details: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> None:
    """Emit an ERROR event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ERROR,
        payload={
            "component": component,
            "error": error,
            "details": details or {},
        },
        source=source,
    ))


async def emit_elo_updated(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = "",
) -> None:
    """Emit an ELO_UPDATED event and check for significant changes.

    If the Elo change exceeds the threshold from unified config,
    also emits an ELO_SIGNIFICANT_CHANGE event to trigger curriculum rebalancing.
    """
    elo_change = new_elo - old_elo

    # Emit the standard Elo update event
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_UPDATED,
        payload={
            "config": config,
            "model_id": model_id,
            "new_elo": new_elo,
            "old_elo": old_elo,
            "elo_change": elo_change,
            "games_played": games_played,
        },
        source=source,
    ))

    # Check for significant change (threshold from unified config)
    try:
        from app.config.unified_config import get_config
        config_obj = get_config()
        threshold = config_obj.curriculum.elo_change_threshold
        should_rebalance = config_obj.curriculum.rebalance_on_elo_change
    except ImportError:
        threshold = 50  # Default if config not available
        should_rebalance = True

    if should_rebalance and abs(elo_change) >= threshold:
        await get_event_bus().publish(DataEvent(
            event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
            payload={
                "config": config,
                "model_id": model_id,
                "elo_change": elo_change,
                "new_elo": new_elo,
                "threshold": threshold,
            },
            source=source,
        ))


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    trigger: str = "scheduled",
    source: str = "",
) -> None:
    """Emit a CURRICULUM_REBALANCED event.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_weights: Previous curriculum weights
        new_weights: New curriculum weights
        trigger: What triggered the rebalance ("scheduled", "elo_change", "manual")
        source: Component that triggered the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_REBALANCED,
        payload={
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "trigger": trigger,
        },
        source=source,
    ))


async def emit_plateau_detected(
    config: str,
    current_elo: float,
    plateau_duration_games: int,
    plateau_duration_seconds: float,
    source: str = "",
) -> None:
    """Emit a PLATEAU_DETECTED event.

    Args:
        config: Board configuration
        current_elo: Current Elo rating
        plateau_duration_games: Number of games in plateau
        plateau_duration_seconds: Time duration of plateau
        source: Component that detected the plateau
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PLATEAU_DETECTED,
        payload={
            "config": config,
            "current_elo": current_elo,
            "plateau_duration_games": plateau_duration_games,
            "plateau_duration_seconds": plateau_duration_seconds,
        },
        source=source,
    ))


async def emit_cmaes_triggered(
    config: str,
    reason: str,
    current_params: Dict[str, Any],
    source: str = "",
) -> None:
    """Emit a CMAES_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why CMA-ES was triggered (e.g., "plateau_detected", "manual")
        current_params: Current hyperparameters before optimization
        source: Component that triggered CMA-ES
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CMAES_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "current_params": current_params,
        },
        source=source,
    ))


async def emit_nas_triggered(
    config: str,
    reason: str,
    search_space: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> None:
    """Emit a NAS_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why NAS was triggered
        search_space: Optional architecture search space configuration
        source: Component that triggered NAS
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NAS_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "search_space": search_space or {},
        },
        source=source,
    ))


async def emit_hyperparameter_updated(
    config: str,
    param_name: str,
    old_value: Any,
    new_value: Any,
    optimizer: str = "manual",
    source: str = "",
) -> None:
    """Emit a HYPERPARAMETER_UPDATED event.

    Args:
        config: Board configuration
        param_name: Name of the parameter that changed
        old_value: Previous value
        new_value: New value
        optimizer: What triggered the update ("cmaes", "nas", "manual")
        source: Component that updated the parameter
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HYPERPARAMETER_UPDATED,
        payload={
            "config": config,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "optimizer": optimizer,
        },
        source=source,
    ))
