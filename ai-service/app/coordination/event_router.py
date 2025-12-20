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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import the 3 event systems
try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        get_event_bus as get_data_event_bus,
    )
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEventType = None
    DataEvent = None

try:
    from app.coordination.stage_events import (
        StageEvent,
        StageCompletionResult,
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
        publish_event as cp_publish,
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


# Mapping from StageEvent to DataEventType for automatic conversion
STAGE_TO_DATA_EVENT_MAP: Dict[str, str] = {
    "selfplay_complete": "new_games",
    "canonical_selfplay_complete": "new_games",
    "sync_complete": "sync_completed",
    "training_complete": "training_completed",
    "training_started": "training_started",
    "training_failed": "training_failed",
    "evaluation_complete": "evaluation_completed",
    "promotion_complete": "model_promoted",
    "cmaes_complete": "cmaes_completed",
    "pbt_complete": "pbt_generation_complete",
    "nas_complete": "nas_completed",
    "model_sync_complete": "p2p_model_synced",
}

# Reverse mapping
DATA_TO_STAGE_EVENT_MAP: Dict[str, str] = {v: k for k, v in STAGE_TO_DATA_EVENT_MAP.items()}


@dataclass
class RouterEvent:
    """Unified event representation across all bus types."""
    event_type: str  # String representation of the event type
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event
    origin: EventSource = EventSource.ROUTER  # Which bus it came from

    # Original event objects (for type-specific handling)
    data_event: Optional[Any] = None
    stage_result: Optional[Any] = None
    cross_process_event: Optional[Any] = None


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
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._global_subscribers: List[EventCallback] = []
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

        # Event history for auditing
        self._event_history: List[RouterEvent] = []
        self._max_history = 1000

        # Metrics
        self._events_routed: Dict[str, int] = {}
        self._events_by_source: Dict[str, int] = {}

        # Cross-process polling
        self._cp_poller: Optional[CrossProcessEventPoller] = None
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
            asyncio.create_task(self._dispatch(router_event, exclude_origin=True))
        except RuntimeError:
            # No running loop - use sync dispatch
            self._dispatch_sync(router_event, exclude_origin=True)

    async def publish(
        self,
        event_type: Union[str, DataEventType, StageEvent],
        payload: Dict[str, Any] = None,
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
        event_type: Union[str, DataEventType, StageEvent],
        payload: Dict[str, Any] = None,
        source: str = "",
    ) -> Union[RouterEvent, asyncio.Future]:
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

    def subscribe(
        self,
        event_type: Optional[Union[str, DataEventType, StageEvent]],
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
        event_type: Optional[Union[str, DataEventType, StageEvent]],
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

            if event_type_str in self._subscribers:
                if callback in self._subscribers[event_type_str]:
                    self._subscribers[event_type_str].remove(callback)
                    return True
        return False

    def get_history(
        self,
        event_type: Optional[str] = None,
        origin: Optional[EventSource] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[RouterEvent]:
        """Get event history with optional filters."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if origin:
            events = [e for e in events if e.origin == origin]
        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
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
_router: Optional[UnifiedEventRouter] = None
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
    event_type: Union[str, DataEventType, StageEvent],
    payload: Dict[str, Any] = None,
    source: str = "",
) -> RouterEvent:
    """Publish an event through the router."""
    return await get_router().publish(event_type, payload, source)


def publish_sync(
    event_type: Union[str, DataEventType, StageEvent],
    payload: Dict[str, Any] = None,
    source: str = "",
) -> Union[RouterEvent, asyncio.Future]:
    """Publish an event synchronously."""
    return get_router().publish_sync(event_type, payload, source)


def subscribe(
    event_type: Optional[Union[str, DataEventType, StageEvent]],
    callback: EventCallback,
) -> None:
    """Subscribe to events through the router."""
    get_router().subscribe(event_type, callback)


def unsubscribe(
    event_type: Optional[Union[str, DataEventType, StageEvent]],
    callback: EventCallback,
) -> bool:
    """Unsubscribe from events."""
    return get_router().unsubscribe(event_type, callback)


__all__ = [
    # Core classes
    "UnifiedEventRouter",
    "RouterEvent",
    "EventSource",
    # Global access
    "get_router",
    "reset_router",
    # Convenience functions
    "publish",
    "publish_sync",
    "subscribe",
    "unsubscribe",
    # Event type mappings
    "STAGE_TO_DATA_EVENT_MAP",
    "DATA_TO_STAGE_EVENT_MAP",
]
