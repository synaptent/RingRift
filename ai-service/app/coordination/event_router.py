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
import concurrent.futures
import hashlib
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from app.core.async_context import fire_and_forget, safe_create_task

logger = logging.getLogger(__name__)

# Import event normalization (December 2025)
from app.coordination.event_normalization import normalize_event_type

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult

# Import the 3 event systems
try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        EventBus,
        # Emit functions used by coordination modules (December 2025 consolidation)
        emit_cluster_capacity_changed,
        emit_curriculum_advanced,
        emit_daemon_status_changed,
        emit_data_event,
        emit_data_sync_failed,
        emit_elo_velocity_changed,
        emit_exploration_boost,
        emit_host_offline,
        emit_host_online,
        emit_idle_resource_detected,
        emit_leader_elected,
        emit_leader_lost,
        emit_node_overloaded,
        emit_promotion_candidate,
        emit_quality_check_requested,
        emit_quality_degraded,
        emit_quality_score_updated,
        emit_selfplay_target_updated,
        emit_task_abandoned,  # December 2025 Wave 2
        emit_training_early_stopped,
        emit_training_loss_anomaly,
        emit_training_loss_trend,
        get_event_bus as get_data_event_bus,
    )
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEventType = None
    DataEvent = None
    EventBus = None
    # Stubs for when data_events not available
    emit_cluster_capacity_changed = None
    emit_curriculum_advanced = None
    emit_daemon_status_changed = None
    emit_data_event = None
    emit_data_sync_failed = None
    emit_elo_velocity_changed = None
    emit_exploration_boost = None
    emit_host_offline = None
    emit_host_online = None
    emit_idle_resource_detected = None
    emit_leader_elected = None
    emit_leader_lost = None
    emit_node_overloaded = None
    emit_promotion_candidate = None
    emit_quality_check_requested = None
    emit_quality_degraded = None
    emit_quality_score_updated = None
    emit_selfplay_target_updated = None
    emit_task_abandoned = None  # December 2025 Wave 2
    emit_training_early_stopped = None
    emit_training_loss_anomaly = None
    emit_training_loss_trend = None

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

# Dead Letter Queue for capturing failed event handlers (December 2025)
try:
    from app.coordination.dead_letter_queue import get_dead_letter_queue
    HAS_DLQ = True
except ImportError:
    HAS_DLQ = False
    get_dead_letter_queue = None


# Handler timeout to prevent hung event handlers from blocking dispatch
# Environment variable override: RINGRIFT_EVENT_HANDLER_TIMEOUT
DEFAULT_HANDLER_TIMEOUT_SECONDS = float(
    os.environ.get("RINGRIFT_EVENT_HANDLER_TIMEOUT", "30.0")
)


def _validate_event_subsystems() -> None:
    """Validate event subsystem availability at startup.

    Logs warnings for missing subsystems and raises if all are unavailable.
    Called at module import time to catch configuration issues early.
    """
    available = []
    missing = []

    if HAS_DATA_EVENTS:
        available.append("data_events")
    else:
        missing.append("data_events (app.distributed.data_events)")

    if HAS_STAGE_EVENTS:
        available.append("stage_events")
    else:
        missing.append("stage_events (app.coordination.stage_events)")

    if HAS_CROSS_PROCESS:
        available.append("cross_process")
    else:
        missing.append("cross_process (app.coordination.cross_process_events)")

    # Log status
    if missing:
        if len(missing) >= 2:
            logger.error(
                f"[EventRouter] Multiple event subsystems unavailable: {', '.join(missing)}. "
                f"Events may be silently dropped. Available: {', '.join(available) or 'NONE'}"
            )
        else:
            logger.warning(
                f"[EventRouter] Event subsystem unavailable: {missing[0]}. "
                f"Some event routing features disabled."
            )

    if not available:
        raise ImportError(
            "EventRouter: All event subsystems failed to import. "
            "At least one of data_events, stage_events, or cross_process_events is required. "
            "Check PYTHONPATH and module dependencies."
        )

    logger.debug(f"[EventRouter] Available subsystems: {', '.join(available)}")


# Validate at module load time
_validate_event_subsystems()


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


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def _compute_content_hash(event_type: str, payload: dict[str, Any]) -> str:
    """Compute a hash of event content for deduplication.

    This allows detecting duplicate events even if they have different event_ids
    (which happens when the same event is forwarded through different buses).

    Only considers stable payload fields - excludes timestamps and node-specific data.
    """
    # Extract stable fields for hashing (exclude timestamps, node names, etc.)
    stable_payload = {
        k: v for k, v in payload.items()
        if k not in ("timestamp", "created_at", "node_name", "source", "routed")
    }
    # Create deterministic string representation
    content = f"{event_type}:{sorted(stable_payload.items())}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class RouterEvent:
    """Unified event representation across all bus types.

    Phase 12 (December 2025): Added trace_id, correlation_id, parent_event_id
    for distributed tracing and request correlation across the cluster.
    """
    event_type: str  # String representation of the event type
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event
    origin: EventSource = EventSource.ROUTER  # Which bus it came from
    event_id: str = field(default_factory=_generate_event_id)  # Unique ID for deduplication

    # Phase 12: Distributed tracing and correlation fields
    trace_id: str = ""  # Trace ID for distributed tracing (propagated across services)
    correlation_id: str = ""  # Correlation ID for request grouping (e.g., all events from one user action)
    parent_event_id: str = ""  # Parent event ID for event causality chains

    # Original event objects (for type-specific handling)
    data_event: Any | None = None
    stage_result: Any | None = None
    cross_process_event: Any | None = None

    def __post_init__(self):
        """Compute content hash after initialization and auto-populate trace context."""
        # Content hash for detecting duplicate events forwarded through different buses
        self._content_hash: str = _compute_content_hash(self.event_type, self.payload)

        # Phase 12: Auto-populate trace_id from current context if not set
        if not self.trace_id:
            try:
                from app.coordination.tracing import get_trace_id
                current_trace = get_trace_id()
                if current_trace:
                    object.__setattr__(self, 'trace_id', current_trace)
            except ImportError:
                pass  # Tracing module not available

    @property
    def content_hash(self) -> str:
        """Get content-based hash for deduplication across buses."""
        if not hasattr(self, '_content_hash') or not self._content_hash:
            self._content_hash = _compute_content_hash(self.event_type, self.payload)
        return self._content_hash


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
        max_seen_events: int = 10000,
    ):
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._global_subscribers: list[EventCallback] = []
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

        # Event history for auditing (bounded by self._max_history).
        # Note: tests and some legacy code mutate _max_history directly.
        # To respect that, we enforce the bound manually on append rather than
        # relying on deque(maxlen=...) which cannot be updated via attribute writes.
        self._max_history = 1000
        self._event_history: deque[RouterEvent] = deque()

        # Deduplication: track seen event IDs and content hashes to prevent loops
        self._seen_events: set[str] = set()  # event_id based
        self._seen_content_hashes: set[str] = set()  # content-based (Dec 2025)
        # Use deque for O(1) popleft instead of O(n) list.pop(0) - Dec 2025 perf fix
        self._seen_events_order: deque[str] = deque()  # For LRU eviction (event_id)
        self._seen_hashes_order: deque[str] = deque()  # For LRU eviction (content hash)
        self._max_seen_events = max_seen_events
        self._duplicates_prevented = 0
        self._content_duplicates_prevented = 0  # Content-hash based duplicates

        # Metrics
        self._events_routed: dict[str, int] = {}
        self._events_by_source: dict[str, int] = {}

        # Cross-process polling
        self._cp_poller: CrossProcessEventPoller | None = None
        self._enable_cp_polling = enable_cross_process_polling
        self._poll_interval = poll_interval

        # Thread pool for async callbacks in sync contexts (Dec 2025 - resource leak fix)
        # Initialize with proper size based on CPU count, max 4 workers
        cpu_count = os.cpu_count() or 4
        max_workers = min(cpu_count, 4)
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="event_router",
        )

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
        # Use stable=True to resume from last acked event on restart
        if HAS_CROSS_PROCESS and self._enable_cp_polling:
            self._cp_poller = CrossProcessEventPoller(
                process_name="event_router",
                event_types=None,  # All events
                poll_interval=self._poll_interval,
                stable=True,  # Persist acks across restarts to reduce duplicates
            )
            self._cp_poller.register_handler(None, self._on_cross_process_event)
            self._cp_poller.start()

    async def _on_stage_event(self, result: StageCompletionResult) -> None:
        """Handle events from StageEventBus."""
        router_event = RouterEvent(
            event_type=normalize_event_type(result.event.value),
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
            event_type=normalize_event_type(event.event_type),
            payload=event.payload,
            timestamp=event.created_at,
            source=event.source,
            origin=EventSource.CROSS_PROCESS,
            cross_process_event=event,
        )
        # Run async dispatch in sync context
        try:
            asyncio.get_running_loop()
            # Use safe_create_task for automatic error handling (December 2025)
            safe_create_task(
                self._dispatch(router_event, exclude_origin=True),
                name="event_router_dispatch",
            )
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

        December 2025: Now normalizes event types to canonical form.
        This ensures consistent event names across the system:
        - SYNC_COMPLETE → DATA_SYNC_COMPLETED
        - TRAINING_COMPLETE → TRAINING_COMPLETED
        - etc.

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
        # Normalize event type to string (extract from enum if needed)
        if hasattr(event_type, 'value'):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        # Normalize to canonical form (December 2025)
        # This standardizes SYNC_COMPLETE → DATA_SYNC_COMPLETED, etc.
        original_event_type = event_type_str
        event_type_str = normalize_event_type(event_type_str)

        # Log normalization for debugging (only if changed)
        if original_event_type != event_type_str:
            logger.debug(
                f"[EventRouter] Normalized '{original_event_type}' → '{event_type_str}'"
            )

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
                except (ValueError, KeyError) as e:
                    logger.debug(f"[EventRouter] Failed to create stage result: {e}")

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
        """Dispatch event to router subscribers with deduplication."""
        _ = exclude_origin  # Kept for API compatibility
        async with self._lock:
            # Deduplication: skip if we've already seen this event by ID
            if event.event_id in self._seen_events:
                self._duplicates_prevented += 1
                logger.debug(
                    f"[EventRouter] Skipping duplicate event {event.event_type} "
                    f"(id={event.event_id[:8]})"
                )
                return

            # Content-based deduplication: catch events forwarded through different buses
            # (Dec 2025 - prevents amplification loops)
            content_hash = event.content_hash
            content_seen = content_hash in self._seen_content_hashes

            # Router-originated events are allowed to repeat (same type/payload)
            # because publish() may legitimately emit identical events multiple
            # times. Content dedup is intended primarily to suppress forwarded
            # duplicates (stage/cross-process echo).
            if content_seen and event.origin != EventSource.ROUTER:
                self._content_duplicates_prevented += 1
                logger.debug(
                    f"[EventRouter] Skipping content-duplicate event {event.event_type} "
                    f"(hash={content_hash}, id={event.event_id[:8]})"
                )
                return

            # Mark as seen with LRU eviction (both event_id and content_hash)
            self._seen_events.add(event.event_id)
            self._seen_events_order.append(event.event_id)
            while len(self._seen_events) > self._max_seen_events:
                oldest = self._seen_events_order.popleft()  # O(1) with deque
                self._seen_events.discard(oldest)

            if not content_seen:
                self._seen_content_hashes.add(content_hash)
            self._seen_hashes_order.append(content_hash)

            # Bound the LRU deque even if we allow repeated router-origin events
            # with identical content hashes.
            while len(self._seen_hashes_order) > self._max_seen_events:
                oldest_hash = self._seen_hashes_order.popleft()  # O(1) with deque
                if oldest_hash not in self._seen_hashes_order:
                    self._seen_content_hashes.discard(oldest_hash)

            # Track in history (bounded by _max_history)
            self._event_history.append(event)
            while len(self._event_history) > self._max_history:
                self._event_history.popleft()

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

        # Invoke callbacks with timeout protection (Dec 2025)
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    try:
                        await asyncio.wait_for(
                            result,
                            timeout=DEFAULT_HANDLER_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        callback_name = getattr(callback, "__name__", str(callback))
                        logger.error(
                            f"[EventRouter] Handler timeout for {event.event_type}: "
                            f"{callback_name} exceeded {DEFAULT_HANDLER_TIMEOUT_SECONDS}s"
                        )
            except Exception as e:  # noqa: BLE001
                callback_name = getattr(callback, "__name__", str(callback))
                logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")
                # Capture to DLQ for retry/analysis (December 2025)
                if HAS_DLQ and get_dead_letter_queue:
                    try:
                        dlq = get_dead_letter_queue()
                        dlq.capture(
                            event_type=str(event.event_type),
                            payload=event.payload,
                            handler_name=callback_name,
                            error=str(e),
                            source="EventRouter.async",
                        )
                    except (RuntimeError, AttributeError, ImportError) as dlq_err:
                        # DLQ capture is best-effort - log at trace level
                        logger.debug(f"[EventRouter] DLQ capture failed: {dlq_err}")

    def _dispatch_sync(
        self,
        event: RouterEvent,
        exclude_origin: bool = False,
    ) -> None:
        """Synchronous dispatch for non-async contexts with deduplication."""
        _ = exclude_origin  # Kept for API compatibility
        with self._sync_lock:
            # Deduplication: skip if we've already seen this event by ID
            if event.event_id in self._seen_events:
                self._duplicates_prevented += 1
                logger.debug(
                    f"[EventRouter] Skipping duplicate event {event.event_type} "
                    f"(id={event.event_id[:8]})"
                )
                return

            # Content-based deduplication: catch events forwarded through different buses
            # (Dec 2025 - prevents amplification loops)
            content_hash = event.content_hash
            content_seen = content_hash in self._seen_content_hashes

            # Router-originated events are allowed to repeat (same type/payload).
            if content_seen and event.origin != EventSource.ROUTER:
                self._content_duplicates_prevented += 1
                logger.debug(
                    f"[EventRouter] Skipping content-duplicate event {event.event_type} "
                    f"(hash={content_hash}, id={event.event_id[:8]})"
                )
                return

            # Mark as seen with LRU eviction (both event_id and content_hash)
            self._seen_events.add(event.event_id)
            self._seen_events_order.append(event.event_id)
            while len(self._seen_events) > self._max_seen_events:
                oldest = self._seen_events_order.popleft()  # O(1) with deque
                self._seen_events.discard(oldest)

            if not content_seen:
                self._seen_content_hashes.add(content_hash)
            self._seen_hashes_order.append(content_hash)

            while len(self._seen_hashes_order) > self._max_seen_events:
                oldest_hash = self._seen_hashes_order.popleft()  # O(1) with deque
                if oldest_hash not in self._seen_hashes_order:
                    self._seen_content_hashes.discard(oldest_hash)

            # Track in history (bounded by _max_history)
            self._event_history.append(event)
            while len(self._event_history) > self._max_history:
                self._event_history.popleft()

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

        # Invoke callbacks (handle both sync and async)
        # Dec 2025 fix: Use thread pool to avoid asyncio.run() blocking the caller
        for callback in callbacks:
            try:
                result = callback(event)
                # Run async callbacks in thread pool (non-blocking)
                # Previously used asyncio.run() which blocked and failed on nested loops
                if asyncio.iscoroutine(result):
                    def run_async_in_thread(coro):
                        """Run async callback in a new thread with its own event loop."""
                        try:
                            asyncio.run(coro)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"[EventRouter] Async callback failed: {e}")

                    # Use thread pool (initialized in __init__)
                    self._thread_pool.submit(run_async_in_thread, result)
                    logger.debug(
                        f"[EventRouter] Dispatched async callback {callback.__name__} "
                        f"for {event.event_type} to thread pool"
                    )
            except Exception as e:  # noqa: BLE001
                callback_name = getattr(callback, "__name__", str(callback))
                logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")
                # Capture to DLQ for retry/analysis (December 2025)
                if HAS_DLQ and get_dead_letter_queue:
                    try:
                        dlq = get_dead_letter_queue()
                        dlq.capture(
                            event_type=str(event.event_type),
                            payload=event.payload,
                            handler_name=callback_name,
                            error=str(e),
                            source="EventRouter.sync",
                        )
                    except (RuntimeError, AttributeError, ImportError) as dlq_err:
                        # DLQ capture is best-effort - log at trace level
                        logger.debug(f"[EventRouter] DLQ capture failed: {dlq_err}")

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

            # Keep subscription keys consistent with publish() normalization.
            event_type_str = normalize_event_type(event_type_str)

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

            # Keep unsubscribe keys consistent with subscribe()/publish() normalization.
            event_type_str = normalize_event_type(event_type_str)

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

        # Ensure slicing works regardless of whether we filtered into a list
        events_list = list(events)
        return events_list[-limit:]

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
            # Deduplication metrics (Dec 2025 enhanced)
            "duplicates_prevented": self._duplicates_prevented,  # By event_id
            "content_duplicates_prevented": self._content_duplicates_prevented,  # By content hash
            "total_duplicates_prevented": self._duplicates_prevented + self._content_duplicates_prevented,
            "seen_events_count": len(self._seen_events),
            "seen_content_hashes_count": len(self._seen_content_hashes),
            "max_seen_events": self._max_seen_events,
        }

    def get_orphaned_events(self) -> dict[str, list[str]]:
        """Identify events that are emitted but have no subscribers (December 2025).

        This helps detect:
        - Events that should be wired but aren't
        - Dead code paths that emit events no one handles

        Returns:
            Dictionary with:
            - "emitted_no_subscribers": Events that have been routed but have no subscribers
            - "subscribed_event_types": All event types that have subscribers
            - "emitted_event_types": All event types that have been routed
        """
        # Get all subscribed event types (from self._subscribers)
        subscribed_types = set(self._subscribers.keys())

        # Get all emitted event types (from routing stats)
        emitted_types = set(self._events_routed.keys())

        # Find orphaned: emitted but no subscribers
        orphaned = emitted_types - subscribed_types

        return {
            "emitted_no_subscribers": sorted(orphaned),
            "subscribed_event_types": sorted(subscribed_types),
            "emitted_event_types": sorted(emitted_types),
        }

    def validate_event_flow(self) -> dict[str, Any]:
        """Validate event flow health and return diagnostics (December 2025).

        This method checks:
        1. Whether events are being routed through all buses
        2. If there are any stuck or missing events
        3. Recent event activity patterns
        4. Orphaned events (emitted but no subscribers)

        Returns:
            Dictionary with validation results and recommendations
        """
        stats = self.get_stats()
        issues: list[str] = []
        recommendations: list[str] = []

        # Check if events are being routed
        total_routed = stats["total_events_routed"]
        if total_routed == 0:
            issues.append("No events have been routed - pipeline may not be active")
            recommendations.append("Start the pipeline with: python scripts/run_training_loop.py")

        # Check bus availability
        if not stats["has_data_events"]:
            issues.append("DataEventBus not available")
        if not stats["has_stage_events"]:
            issues.append("StageEventBus not available")
        if not stats["has_cross_process"]:
            issues.append("CrossProcessEventQueue not available")

        # Check for high duplicate rate (potential event loop)
        if total_routed > 100:
            dup_rate = stats["total_duplicates_prevented"] / (total_routed + stats["total_duplicates_prevented"])
            if dup_rate > 0.3:
                issues.append(f"High duplicate rate: {dup_rate:.1%} - possible event loop")
                recommendations.append("Check for circular event subscriptions")

        # Check subscriber count
        if stats["subscriber_count"] == 0 and stats["global_subscriber_count"] == 0:
            issues.append("No event subscribers registered")
            recommendations.append("Ensure pipeline components are initialized with wire_pipeline_events()")

        # Get recent event types
        recent_events = [e.event_type for e in self._event_history[-20:]] if self._event_history else []
        recent_event_types = list(set(recent_events))

        # Check for orphaned events (December 2025 - semantic validation)
        orphaned = self.get_orphaned_events()
        orphaned_events = orphaned["emitted_no_subscribers"]
        if orphaned_events:
            # Only warn about events that have been emitted more than 5 times
            # to reduce noise from one-off events
            significant_orphans = [
                ev for ev in orphaned_events
                if self._events_routed.get(ev, 0) >= 5
            ]
            if significant_orphans:
                issues.append(f"Orphaned events (emitted but no subscribers): {significant_orphans[:5]}")
                recommendations.append("Wire subscribers for orphaned events or remove emission code")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "total_events_routed": total_routed,
            "subscriber_count": stats["subscriber_count"] + stats["global_subscriber_count"],
            "recent_event_types": recent_event_types,
            "buses_available": {
                "data_events": stats["has_data_events"],
                "stage_events": stats["has_stage_events"],
                "cross_process": stats["has_cross_process"],
            },
            "duplicates_prevented": stats["total_duplicates_prevented"],
            "orphaned_events": orphaned,  # December 2025: Include full orphan analysis
        }

    def health_check(self) -> HealthCheckResult:
        """Perform health check for CoordinatorProtocol compliance (Dec 2025).

        Returns:
            HealthCheckResult with healthy status and diagnostics
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        validation = self.validate_event_flow()
        stats = self.get_stats()

        if validation["healthy"]:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="Event router operational",
                details={
                    "total_events_routed": stats["total_events_routed"],
                    "subscriber_count": validation["subscriber_count"],
                    "recent_event_types": validation["recent_event_types"],
                },
            )
        else:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="; ".join(validation["issues"]),
                details={
                    "issues": validation["issues"],
                    "recommendations": validation["recommendations"],
                    "buses_available": validation["buses_available"],
                },
            )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive router status for monitoring dashboards (Dec 2025).

        Returns:
            Dictionary with router status, stats, and health information.
            Used by DaemonManager for monitoring and dashboards.
        """
        stats = self.get_stats()
        validation = self.validate_event_flow()
        orphan_analysis = self.get_orphaned_events()

        return {
            "name": "UnifiedEventRouter",
            "running": True,  # Router is always running once initialized
            "healthy": validation["healthy"],
            "stats": {
                "total_events_routed": stats["total_events_routed"],
                "events_by_type": stats["events_routed_by_type"],
                "subscriber_count": stats["subscriber_count"],
                "global_subscriber_count": stats["global_subscriber_count"],
                "duplicates_prevented": stats["total_duplicates_prevented"],
                "history_size": stats["history_size"],
            },
            "capabilities": {
                "data_events": stats["has_data_events"],
                "stage_events": stats["has_stage_events"],
                "cross_process": stats["has_cross_process"],
                "cross_process_polling": stats["cross_process_polling"],
            },
            "health": {
                "issues": validation.get("issues", []),
                "recommendations": validation.get("recommendations", []),
                "buses_available": validation.get("buses_available", {}),
            },
            "orphan_events": {
                "count": len(orphan_analysis.get("orphaned_events", [])),
                "types": orphan_analysis.get("orphaned_events", [])[:10],  # Limit to 10
            },
        }

    def stop(self) -> None:
        """Stop the router (cleanup cross-process poller and thread pool)."""
        if self._cp_poller:
            self._cp_poller.stop()
            self._cp_poller = None

        # Shutdown thread pool gracefully
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True, cancel_futures=False)
            self._thread_pool = None

    def close(self) -> None:
        """Close the router (alias for stop)."""
        self.stop()


# Global singleton
_router: UnifiedEventRouter | None = None
_router_lock = threading.Lock()


def get_router() -> UnifiedEventRouter:
    """Get the global event router singleton."""
    global _router
    with _router_lock:
        if _router is None:
            # In pytest runs, disable cross-process polling by default. The
            # cross-process queue is backed by a persistent SQLite DB under /tmp,
            # and polling can replay historical events into otherwise isolated
            # unit tests.
            enable_cp_polling = os.environ.get("PYTEST_CURRENT_TEST") is None
            _router = UnifiedEventRouter(enable_cross_process_polling=enable_cp_polling)
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


def validate_event_wiring(
    raise_on_error: bool = False,
    log_warnings: bool = True,
) -> dict[str, Any]:
    """Validate that critical pipeline events have subscribers (December 2025).

    This function should be called at startup to ensure all critical events
    are properly wired. It checks for subscribers to events that are essential
    for the training pipeline to function correctly.

    Critical events checked:
    - TRAINING_COMPLETED: Should trigger evaluation, feedback, model distribution
    - EVALUATION_COMPLETED: Should trigger curriculum updates, promotion decisions
    - MODEL_PROMOTED: Should trigger model distribution to cluster
    - DATA_SYNC_COMPLETED: Should trigger NPZ export pipeline
    - NEW_GAMES_AVAILABLE: Should trigger training data processing

    Args:
        raise_on_error: If True, raises RuntimeError on critical missing subscriptions
        log_warnings: If True, logs warnings for missing subscriptions

    Returns:
        Dictionary with:
        - "valid": True if all critical events have subscribers
        - "missing_critical": List of critical events without subscribers
        - "missing_optional": List of recommended events without subscribers
        - "all_subscribed": List of all event types that have subscribers

    Usage:
        from app.coordination.event_router import validate_event_wiring

        # At startup
        result = validate_event_wiring(raise_on_error=True)

        # Or just check without raising
        result = validate_event_wiring()
        if not result["valid"]:
            for event in result["missing_critical"]:
                print(f"CRITICAL: {event} has no subscribers!")
    """
    router = get_router()

    # Critical events that MUST have subscribers for pipeline to work
    critical_events = [
        "training_completed",  # TRAINING_COMPLETED.value
        "evaluation_completed",  # EVALUATION_COMPLETED.value
        "model_promoted",  # MODEL_PROMOTED.value
        "sync_completed",  # DATA_SYNC_COMPLETED.value
    ]

    # Recommended events (nice to have but not blocking)
    recommended_events = [
        "new_games",  # NEW_GAMES_AVAILABLE.value
        "curriculum_rebalanced",  # CURRICULUM_REBALANCED.value
        "elo_velocity_changed",  # ELO_VELOCITY_CHANGED.value
        "training_started",  # TRAINING_STARTED.value
        "selfplay_complete",  # SELFPLAY_COMPLETE.value
    ]

    # Get all subscribed event types from router
    subscribed_types = set(router._subscribers.keys())

    # Check critical events
    missing_critical = []
    for event in critical_events:
        normalized = normalize_event_type(event)
        if normalized not in subscribed_types:
            missing_critical.append(event)

    # Check recommended events
    missing_optional = []
    for event in recommended_events:
        normalized = normalize_event_type(event)
        if normalized not in subscribed_types:
            missing_optional.append(event)

    # Log warnings
    if log_warnings:
        for event in missing_critical:
            logger.error(
                f"[EventRouter] CRITICAL: Event '{event}' has no subscribers! "
                f"Pipeline may not function correctly."
            )
        for event in missing_optional:
            logger.warning(
                f"[EventRouter] Recommended event '{event}' has no subscribers."
            )

    # Raise if critical events are missing
    if raise_on_error and missing_critical:
        raise RuntimeError(
            f"Critical events without subscribers: {missing_critical}. "
            f"Ensure pipeline components are initialized with wire_pipeline_events() "
            f"or coordination_bootstrap.bootstrap_coordinators()."
        )

    is_valid = len(missing_critical) == 0

    return {
        "valid": is_valid,
        "missing_critical": missing_critical,
        "missing_optional": missing_optional,
        "all_subscribed": sorted(subscribed_types),
        "critical_events_checked": critical_events,
        "recommended_events_checked": recommended_events,
    }


def validate_event_flow() -> dict[str, Any]:
    """Validate event flow health (December 2025).

    Returns diagnostics about the event system including:
    - Whether events are being routed
    - Bus availability
    - Recent event types
    - Orphaned events (emitted but no subscribers)
    - Any issues detected

    Usage:
        from app.coordination.event_router import validate_event_flow
        result = validate_event_flow()
        if not result["healthy"]:
            print("Issues:", result["issues"])
        if result["orphaned_events"]["emitted_no_subscribers"]:
            print("Orphaned:", result["orphaned_events"]["emitted_no_subscribers"])
    """
    return get_router().validate_event_flow()


def get_orphaned_events() -> dict[str, list[str]]:
    """Identify events that are emitted but have no subscribers (December 2025).

    Returns:
        Dictionary with:
        - "emitted_no_subscribers": Events that have been routed but have no subscribers
        - "subscribed_event_types": All event types that have subscribers
        - "emitted_event_types": All event types that have been routed

    Usage:
        from app.coordination.event_router import get_orphaned_events
        orphaned = get_orphaned_events()
        for ev in orphaned["emitted_no_subscribers"]:
            print(f"Event {ev} has no subscribers")
    """
    return get_router().get_orphaned_events()


def get_event_stats() -> dict[str, Any]:
    """Get event router statistics."""
    return get_router().get_stats()


def get_event_payload(event: RouterEvent | dict[str, Any] | Any) -> dict[str, Any]:
    """Extract payload from an event, regardless of event format.

    December 2025: Standard helper for event handlers that need to work with
    both RouterEvent objects and raw dictionaries. Use this in callbacks
    instead of custom payload extraction logic.

    Args:
        event: Either a RouterEvent, a dict, or any object with a .payload attribute

    Returns:
        The event payload as a dictionary

    Usage:
        from app.coordination.event_router import get_event_payload

        def my_handler(event: RouterEvent) -> None:
            payload = get_event_payload(event)
            host = payload.get("host")
            # ... process event
    """
    if isinstance(event, dict):
        return event
    payload = getattr(event, "payload", None)
    return payload if isinstance(payload, dict) else {}


__all__ = [  # noqa: RUF022
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
    "validate_event_flow",
    "validate_event_wiring",  # Dec 2025: Startup validation for critical events
    "get_orphaned_events",
    "get_event_stats",
    "get_event_payload",  # Dec 2025: Standard helper for event handlers
    # Re-exports from data_events for backward compatibility
    "DataEvent",
    "DataEventType",
    "EventBus",
    "get_event_bus",
    # Emit functions re-exported from data_events (December 2025 consolidation)
    "emit_cluster_capacity_changed",
    "emit_curriculum_advanced",
    "emit_daemon_status_changed",
    "emit_data_event",
    "emit_data_sync_failed",
    "emit_elo_velocity_changed",
    "emit_exploration_boost",
    "emit_host_offline",
    "emit_host_online",
    "emit_idle_resource_detected",
    "emit_leader_elected",
    "emit_leader_lost",
    "emit_node_overloaded",
    "emit_promotion_candidate",
    "emit_quality_check_requested",
    "emit_quality_degraded",
    "emit_quality_score_updated",
    "emit_selfplay_target_updated",
    "emit_training_early_stopped",
    "emit_training_loss_anomaly",
    "emit_training_loss_trend",
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
        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
        asyncio.get_running_loop()
        fire_and_forget(
            emit_training_started(config_key, node_name, **extra_payload),
            name=f"emit_training_started_{config_key}",
        )
    except RuntimeError:
        # No running loop - create one with asyncio.run()
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
        # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
        asyncio.get_running_loop()
        fire_and_forget(
            emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload),
            name=f"emit_training_completed_{config_key}",
        )
    except RuntimeError:
        # No running loop - create one with asyncio.run()
        asyncio.run(emit_training_completed(config_key, model_id, val_loss, epochs, **extra_payload))
