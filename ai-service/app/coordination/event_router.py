"""Unified Event Router - Single entry point for all event systems.

This module consolidates the 3 separate event buses into a unified routing layer:
1. EventBus (data_events.py) - In-memory async event bus
2. StageEventBus (stage_events.py) - Pipeline stage completion events
3. CrossProcessEventQueue (cross_process_events.py) - SQLite-backed cross-process queue

The router provides:
- Single API for publishing events that routes to all appropriate buses
- Automatic event type mapping between systems
- Bidirectional cross-process integration
- Unified subscription management
- Event flow auditing

Usage:
    from app.coordination.event_router import (
        get_router,
        publish,
        subscribe,
        EventSource,
    )

    # Publish to all systems
    await publish(
        DataEventType.TRAINING_COMPLETED,
        payload={"config": "square8_2p", "success": True},
        source="training_daemon"
    )

    # Subscribe to events (receives from all systems)
    router = get_router()
    router.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)

Created: December 2025
Purpose: Consolidate event bus fragmentation (Phase 13 consolidation)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Union

logger = logging.getLogger(__name__)

# Import the 3 event systems
try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        EventBus,
        get_event_bus as get_data_event_bus,
    )
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEventType = None
    DataEvent = None
    EventBus = None

try:
    from app.coordination.stage_events import (
        StageCompletionResult,
        StageEvent,
        get_event_bus as get_stage_event_bus,
    )
    HAS_STAGE_EVENTS = True
except ImportError:
    HAS_STAGE_EVENTS = False
    StageEvent = None

try:
    from app.coordination.cross_process_events import (
        CrossProcessEvent,
        CrossProcessEventPoller,
        CrossProcessEventQueue,
        ack_event,
        ack_events,
        bridge_to_cross_process,
        get_event_queue as get_cross_process_queue,
        poll_events as cp_poll_events,
        publish_event as cp_publish,
        reset_event_queue as reset_cross_process_queue,
        subscribe_process,
    )
    HAS_CROSS_PROCESS = True
except ImportError:
    HAS_CROSS_PROCESS = False


class EventSource(str, Enum):
    """Source system for events."""
    DATA_BUS = "data_bus"           # From EventBus (data_events.py)
    STAGE_BUS = "stage_bus"         # From StageEventBus (stage_events.py)
    CROSS_PROCESS = "cross_process" # From CrossProcessEventQueue
    ROUTER = "router"               # Originated from this router


# Import centralized event mappings (DRY - consolidated in event_mappings.py)
from app.coordination.event_mappings import (
    DATA_TO_STAGE_EVENT_MAP,
    STAGE_TO_DATA_EVENT_MAP,
)


@dataclass
class RouterEvent:
    """Unified event representation across all bus types."""
    event_type: str  # String representation of the event type
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event
    origin: EventSource = EventSource.ROUTER  # Which bus it came from

    # Original event objects (for type-specific handling)
    data_event: Any | None = None
    stage_result: Any | None = None
    cross_process_event: Any | None = None


EventCallback = Callable[[RouterEvent], Union[None, Any]]


class UnifiedEventRouter:
    """Routes events between all event systems.

    Acts as a facade over the 3 event buses, providing:
    1. Unified publish API - events go to all appropriate buses
    2. Unified subscribe API - receive events from any bus
    3. Automatic type conversion between bus types
    4. Cross-process event polling and forwarding
    5. Event flow auditing and metrics
    """

    def __init__(
        self,
        enable_cross_process_polling: bool = True,
        poll_interval: float = 1.0,
    ):
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._global_subscribers: list[EventCallback] = []
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

        # Event history for auditing
        self._event_history: list[RouterEvent] = []
        self._max_history = 1000

        # Metrics
        self._events_routed: dict[str, int] = {}
        self._events_by_source: dict[str, int] = {}

        # Cross-process polling
        self._cp_poller: CrossProcessEventPoller | None = None
        self._enable_cp_polling = enable_cross_process_polling
        self._poll_interval = poll_interval

        # Connect to existing buses
        self._setup_bus_bridges()

    def _setup_bus_bridges(self) -> None:
        """Set up bidirectional bridges with existing event buses."""
        # Bridge: StageEventBus -> Router
        if HAS_STAGE_EVENTS:
            stage_bus = get_stage_event_bus()
            for event in StageEvent:
                stage_bus.subscribe(event, self._on_stage_event)

        # Bridge: CrossProcessEventQueue -> Router (via poller)
        if HAS_CROSS_PROCESS and self._enable_cp_polling:
            self._cp_poller = CrossProcessEventPoller(
                process_name="event_router",
                event_types=None,  # All events
                poll_interval=self._poll_interval,
            )
            self._cp_poller.register_handler(None, self._on_cross_process_event)
            self._cp_poller.start()

    async def _on_stage_event(self, result: StageCompletionResult) -> None:
        """Handle events from StageEventBus."""
        router_event = RouterEvent(
            event_type=result.event.value,
            payload=result.to_dict(),
            timestamp=time.time(),
            source=result.metadata.get("source", "stage_bus"),
            origin=EventSource.STAGE_BUS,
            stage_result=result,
        )
        await self._dispatch(router_event, exclude_origin=True)

    def _on_cross_process_event(self, event: CrossProcessEvent) -> None:
        """Handle events from CrossProcessEventQueue."""
        router_event = RouterEvent(
            event_type=event.event_type,
            payload=event.payload,
            timestamp=event.created_at,
            source=event.source,
            origin=EventSource.CROSS_PROCESS,
            cross_process_event=event,
        )
        # Run async dispatch in sync context
        try:
            asyncio.get_running_loop()
            task = asyncio.create_task(
                self._dispatch(router_event, exclude_origin=True)
            )
            # Add error callback to prevent silent failures (December 2025 hardening)
            task.add_done_callback(self._handle_dispatch_task_error)
        except RuntimeError:
            # No running loop - use sync dispatch
            self._dispatch_sync(router_event, exclude_origin=True)

    async def publish(
        self,
        event_type: str | DataEventType | StageEvent,
        payload: dict[str, Any] | None = None,
        source: str = "",
        route_to_data_bus: bool = True,
        route_to_stage_bus: bool = True,
        route_to_cross_process: bool = True,
    ) -> RouterEvent:
        """Publish an event to all appropriate buses.

        Args:
            event_type: Event type (string, DataEventType, or StageEvent)
            payload: Event data
            source: Component that generated the event
            route_to_data_bus: Whether to send to EventBus
            route_to_stage_bus: Whether to send to StageEventBus
            route_to_cross_process: Whether to send to CrossProcessEventQueue

        Returns:
            The RouterEvent that was published
        """
        # Normalize event type to string
        if hasattr(event_type, 'value'):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        payload = payload or {}

        router_event = RouterEvent(
            event_type=event_type_str,
            payload=payload,
            timestamp=time.time(),
            source=source,
            origin=EventSource.ROUTER,
        )

        # Route to EventBus (data_events.py)
        if route_to_data_bus and HAS_DATA_EVENTS:
            data_event_type = None
            try:
                data_event_type = DataEventType(event_type_str)
            except (ValueError, KeyError):
                stage_mapped = STAGE_TO_DATA_EVENT_MAP.get(event_type_str)
                if stage_mapped:
                    try:
                        data_event_type = DataEventType(stage_mapped)
                    except (ValueError, KeyError):
                        data_event_type = None

            if data_event_type is not None:
                data_event = DataEvent(
                    event_type=data_event_type,
                    payload=payload,
                    source=source,
                )
                router_event.data_event = data_event
                # Don't bridge to cross-process again (we'll do it separately)
                await get_data_event_bus().publish(data_event, bridge_cross_process=False)

        # Route to StageEventBus (stage_events.py)
        if route_to_stage_bus and HAS_STAGE_EVENTS:
            stage_event = None
            if isinstance(event_type, StageEvent):
                stage_event = event_type
            else:
                stage_event_name = DATA_TO_STAGE_EVENT_MAP.get(event_type_str)
                if stage_event_name:
                    try:
                        stage_event = StageEvent(stage_event_name)
                    except (ValueError, KeyError):
                        stage_event = None
                else:
                    try:
                        stage_event = StageEvent(event_type_str)
                    except (ValueError, KeyError):
                        stage_event = None

            if stage_event:
                try:
                    extra_fields = {
                        k: v
                        for k, v in payload.items()
                        if k in StageCompletionResult.__dataclass_fields__
                        and k not in {"event", "success", "iteration", "timestamp", "metadata"}
                    }
                    metadata = {}
                    payload_metadata = payload.get("metadata")
                    if isinstance(payload_metadata, dict):
                        metadata.update(payload_metadata)
                    if source:
                        metadata.setdefault("source", source)
                    metadata.setdefault("routed", True)

                    timestamp_value = payload.get("timestamp")
                    if isinstance(timestamp_value, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp_value).isoformat()
                    elif isinstance(timestamp_value, str) and timestamp_value:
                        timestamp = timestamp_value
                    else:
                        timestamp = datetime.now().isoformat()

                    stage_result = StageCompletionResult(
                        event=stage_event,
                        success=payload.get("success", True),
                        iteration=payload.get("iteration", 0),
                        timestamp=timestamp,
                        metadata=metadata,
                        **extra_fields,
                    )
                    router_event.stage_result = stage_result
                    await get_stage_event_bus().emit(stage_result)
                except (ValueError, KeyError):
                    pass

        # Route to CrossProcessEventQueue
        if route_to_cross_process and HAS_CROSS_PROCESS:
            cp_publish(event_type_str, payload, source)

        # Dispatch to router subscribers
        await self._dispatch(router_event, exclude_origin=False)

        return router_event

    def publish_sync(
        self,
        event_type: str | DataEventType | StageEvent,
        payload: dict[str, Any] | None = None,
        source: str = "",
    ) -> RouterEvent | asyncio.Future:
        """Synchronous version of publish for non-async contexts.

        Returns a RouterEvent when no loop is running, otherwise a scheduled Future.
        """
        try:
            asyncio.get_running_loop()
            future = asyncio.ensure_future(
                self.publish(event_type, payload, source)
            )
            return future
        except RuntimeError:
            # No running loop - run synchronously
            return asyncio.run(self.publish(event_type, payload, source))

    async def _dispatch(
        self,
        event: RouterEvent,
        exclude_origin: bool = False,
    ) -> None:
        """Dispatch event to router subscribers."""
        _ = exclude_origin
        async with self._lock:
            # Track in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Update metrics
            self._events_routed[event.event_type] = (
                self._events_routed.get(event.event_type, 0) + 1
            )
            self._events_by_source[event.origin.value] = (
                self._events_by_source.get(event.origin.value, 0) + 1
            )

        # Get callbacks
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Invoke callbacks
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")

    def _dispatch_sync(
        self,
        event: RouterEvent,
        exclude_origin: bool = False,
    ) -> None:
        """Synchronous dispatch for non-async contexts."""
        _ = exclude_origin
        with self._sync_lock:
            # Track in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Update metrics
            self._events_routed[event.event_type] = (
                self._events_routed.get(event.event_type, 0) + 1
            )
            self._events_by_source[event.origin.value] = (
                self._events_by_source.get(event.origin.value, 0) + 1
            )

        # Get callbacks
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Invoke callbacks (sync only)
        for callback in callbacks:
            try:
                result = callback(event)
                # Can't await in sync context
                if asyncio.iscoroutine(result):
                    logger.warning(
                        f"[EventRouter] Async callback {callback.__name__} "
                        f"called from sync context - skipping"
                    )
            except Exception as e:
                logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")

    def _handle_dispatch_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from async dispatch tasks (December 2025 hardening).

        This callback ensures that exceptions from fire-and-forget tasks are
        logged instead of being silently dropped.
        """
        try:
            exc = task.exception()
            if exc is not None:
                logger.error(
                    f"[EventRouter] Async dispatch task failed: {exc}",
                    exc_info=exc,
                )
                # Track failed dispatches
                self._events_routed["__dispatch_failures__"] = (
                    self._events_routed.get("__dispatch_failures__", 0) + 1
                )
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error
        except asyncio.InvalidStateError:
            pass  # Task hasn't completed yet (shouldn't happen in done callback)

    def subscribe(
        self,
        event_type: str | DataEventType | StageEvent | None,
        callback: EventCallback,
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: Event type to subscribe to (None for all events)
            callback: Function to call when event occurs
        """
        if event_type is None:
            self._global_subscribers.append(callback)
        else:
            # Normalize to string
            if hasattr(event_type, 'value'):
                event_type_str = event_type.value
            else:
                event_type_str = str(event_type)

            if event_type_str not in self._subscribers:
                self._subscribers[event_type_str] = []
            self._subscribers[event_type_str].append(callback)

    def unsubscribe(
        self,
        event_type: str | DataEventType | StageEvent | None,
        callback: EventCallback,
    ) -> bool:
        """Unsubscribe from events."""
        if event_type is None:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
                return True
        else:
            if hasattr(event_type, 'value'):
                event_type_str = event_type.value
            else:
                event_type_str = str(event_type)

            if event_type_str in self._subscribers and callback in self._subscribers[event_type_str]:
                self._subscribers[event_type_str].remove(callback)
                return True
        return False

    def register_handler(
        self,
        event_type: str,
        handler: EventCallback,
    ) -> None:
        """Register a handler for an event type (alias for subscribe).

        This method provides backwards compatibility with UnifiedEventCoordinator.

        Args:
            event_type: Event type string
            handler: Callback function to handle the event
        """
        self.subscribe(event_type, handler)

    def get_history(
        self,
        event_type: str | None = None,
        origin: EventSource | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[RouterEvent]:
        """Get event history with optional filters."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if origin:
            events = [e for e in events if e.origin == origin]
        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "events_routed_by_type": dict(self._events_routed),
            "events_by_source": dict(self._events_by_source),
            "total_events_routed": sum(self._events_routed.values()),
            "subscriber_count": sum(len(s) for s in self._subscribers.values()),
            "global_subscriber_count": len(self._global_subscribers),
            "history_size": len(self._event_history),
            "has_data_events": HAS_DATA_EVENTS,
            "has_stage_events": HAS_STAGE_EVENTS,
            "has_cross_process": HAS_CROSS_PROCESS,
            "cross_process_polling": self._cp_poller is not None,
        }

    def stop(self) -> None:
        """Stop the router (cleanup cross-process poller)."""
        if self._cp_poller:
            self._cp_poller.stop()
            self._cp_poller = None


# Global singleton
_router: UnifiedEventRouter | None = None
_router_lock = threading.Lock()


def get_router() -> UnifiedEventRouter:
    """Get the global event router singleton."""
    global _router
    with _router_lock:
        if _router is None:
            _router = UnifiedEventRouter()
        return _router


def reset_router() -> None:
    """Reset the global router (for testing)."""
    global _router
    with _router_lock:
        if _router is not None:
            _router.stop()
        _router = None


# Convenience functions


async def publish(
    event_type: str | DataEventType | StageEvent,
    payload: dict[str, Any] | None = None,
    source: str = "",
) -> RouterEvent:
    """Publish an event through the router."""
    return await get_router().publish(event_type, payload, source)


def publish_sync(
    event_type: str | DataEventType | StageEvent,
    payload: dict[str, Any] | None = None,
    source: str = "",
) -> RouterEvent | asyncio.Future:
    """Publish an event synchronously."""
    return get_router().publish_sync(event_type, payload, source)


def subscribe(
    event_type: str | DataEventType | StageEvent | None,
    callback: EventCallback,
) -> None:
    """Subscribe to events through the router."""
    get_router().subscribe(event_type, callback)


def unsubscribe(
    event_type: str | DataEventType | StageEvent | None,
    callback: EventCallback,
) -> bool:
    """Unsubscribe from events."""
    return get_router().unsubscribe(event_type, callback)


__all__ = [
    "DATA_TO_STAGE_EVENT_MAP",
    # Event type mappings
    "STAGE_TO_DATA_EVENT_MAP",
    "EventSource",
    "RouterEvent",
    # Core classes
    "UnifiedEventRouter",
    # Global access
    "get_router",
    # Convenience functions
    "publish",
    "publish_sync",
    "reset_router",
    "subscribe",
    "unsubscribe",
    # Re-exports from data_events for backward compatibility
    "DataEvent",
    "DataEventType",
    "EventBus",
    "get_event_bus",
    # Re-exports from stage_events for migration
    "StageEvent",
    "StageCompletionResult",
    "get_stage_event_bus",
    # Re-exports from cross_process_events for migration
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    "CrossProcessEventQueue",
    "ack_event",
    "ack_events",
    "bridge_to_cross_process",
    "get_cross_process_queue",
    "cp_poll_events",
    "cp_publish",
    "reset_cross_process_queue",
    "subscribe_process",
    # Backward compatibility aliases for unified_event_coordinator.py
    "CoordinatorStats",
    "UnifiedEventCoordinator",
    "emit_evaluation_completed",
    "emit_model_promoted",
    "emit_selfplay_batch_completed",
    "emit_sync_completed",
    "emit_training_completed",
    "emit_training_completed_sync",
    "emit_training_failed",
    "emit_training_started",
    "emit_training_started_sync",
    "get_coordinator_stats",
    "get_event_coordinator",
    "start_coordinator",
    "stop_coordinator",
]


# Re-export get_event_bus for backward compatibility
# Many files import: from app.coordination.event_router import get_event_bus
def get_event_bus():
    """Get the data event bus (re-exported for backward compatibility)."""
    if HAS_DATA_EVENTS:
        return get_data_event_bus()
    return None


# =============================================================================
# Backward Compatibility Layer for unified_event_coordinator.py
# These aliases allow code that imported from unified_event_coordinator to
# work with the consolidated event_router instead.
# =============================================================================

@dataclass
class CoordinatorStats:
    """Statistics for the event coordinator (backwards-compatible alias)."""
    events_bridged_data_to_cross: int = 0
    events_bridged_stage_to_cross: int = 0
    events_bridged_cross_to_data: int = 0
    events_dropped: int = 0
    last_bridge_time: str | None = None
    errors: list[str] = field(default_factory=list)
    start_time: str | None = None
    is_running: bool = False


# Alias for unified_event_coordinator.UnifiedEventCoordinator
UnifiedEventCoordinator = UnifiedEventRouter


def get_event_coordinator() -> UnifiedEventRouter:
    """Get the event coordinator (alias for get_router).

    This provides backwards compatibility for code using unified_event_coordinator.
    """
    return get_router()


async def start_coordinator() -> bool:
    """Start the event coordinator.

    Returns True (the router auto-starts on instantiation).
    """
    get_router()  # Ensure router is created
    return True


async def stop_coordinator() -> None:
    """Stop the event coordinator."""
    router = get_router()
    router.stop()


def get_coordinator_stats() -> CoordinatorStats:
    """Get coordinator statistics (mapped from router stats)."""
    router = get_router()
    stats = router.get_stats()

    # Map router stats to coordinator stats format
    return CoordinatorStats(
        events_bridged_data_to_cross=stats.get("events_by_source", {}).get("data_bus", 0),
        events_bridged_stage_to_cross=stats.get("events_by_source", {}).get("stage_bus", 0),
        events_bridged_cross_to_data=stats.get("events_by_source", {}).get("cross_process", 0),
        events_dropped=0,
        last_bridge_time=None,
        errors=[],
        start_time=None,
        is_running=stats.get("cross_process_polling", False),
    )


# =============================================================================
# Simple Event Emitters (migrated from unified_event_coordinator.py)
# Use these functions to emit events from anywhere in the codebase.
# =============================================================================

async def emit_training_started(
    config_key: str,
    node_name: str = "",
    **extra_payload
) -> None:
    """Emit TRAINING_STARTED event to all systems."""
    await publish(
        event_type="TRAINING_STARTED",
        payload={
            "config_key": config_key,
            "node_name": node_name,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_training_completed(
    config_key: str,
    model_id: str,
    val_loss: float = 0.0,
    epochs: int = 0,
    **extra_payload
) -> None:
    """Emit TRAINING_COMPLETED event to all systems."""
    await publish(
        event_type="TRAINING_COMPLETED",
        payload={
            "config_key": config_key,
            "model_id": model_id,
            "val_loss": val_loss,
            "epochs": epochs,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_training_failed(
    config_key: str,
    error: str,
    **extra_payload
) -> None:
    """Emit TRAINING_FAILED event to all systems."""
    await publish(
        event_type="TRAINING_FAILED",
        payload={
            "config_key": config_key,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="training",
    )


async def emit_evaluation_completed(
    model_id: str,
    elo: float,
    win_rate: float = 0.0,
    games_played: int = 0,
    **extra_payload
) -> None:
    """Emit EVALUATION_COMPLETED event to all systems."""
    await publish(
        event_type="EVALUATION_COMPLETED",
        payload={
            "model_id": model_id,
            "elo": elo,
            "win_rate": win_rate,
            "games_played": games_played,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="evaluation",
    )


async def emit_sync_completed(
    sync_type: str,
    files_synced: int = 0,
    bytes_transferred: int = 0,
    **extra_payload
) -> None:
    """Emit DATA_SYNC_COMPLETED event to all systems."""
    await publish(
        event_type="DATA_SYNC_COMPLETED",
        payload={
            "sync_type": sync_type,
            "files_synced": files_synced,
            "bytes_transferred": bytes_transferred,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="sync",
    )


async def emit_model_promoted(
    model_id: str,
    tier: str = "production",
    elo: float = 0.0,
    **extra_payload
) -> None:
    """Emit MODEL_PROMOTED event to all systems."""
    await publish(
        event_type="MODEL_PROMOTED",
        payload={
            "model_id": model_id,
            "tier": tier,
            "elo": elo,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="promotion",
    )


async def emit_selfplay_batch_completed(
    config_key: str,
    games_generated: int,
    duration_seconds: float = 0.0,
    **extra_payload
) -> None:
    """Emit SELFPLAY_BATCH_COMPLETE event to all systems."""
    await publish(
        event_type="SELFPLAY_BATCH_COMPLETE",
        payload={
            "config_key": config_key,
            "games_generated": games_generated,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
            **extra_payload,
        },
        source="selfplay",
    )


def emit_training_started_sync(
    config_key: str,
    node_name: str = "",
    **extra_payload
) -> None:
    """Sync version of emit_training_started for non-async contexts."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(emit_training_started(config_key, node_name, **extra_payload))
        else:
            loop.run_until_complete(emit_training_started(config_key, node_name, **extra_payload))
    except RuntimeError:
        asyncio.run(emit_training_started(config_key, node_name, **extra_payload))


def emit_training_completed_sync(
    config_key: str,
    model_id: str,
    val_loss: float = 0.0,
    epochs: int = 0,
    **extra_payload
) -> None:
    """Sync version of emit_training_completed."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload))
        else:
            loop.run_until_complete(emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload))
    except RuntimeError:
        asyncio.run(emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload))
