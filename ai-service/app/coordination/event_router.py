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
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from app.core.async_context import fire_and_forget, safe_create_task

logger = logging.getLogger(__name__)

# Import event normalization (December 2025)
from app.coordination.event_normalization import (
    normalize_event_type,
    validate_event_type,
    UnknownEventTypeError,
)
from app.coordination.event_utils import make_config_key

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
        emit_node_suspect,
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
    emit_node_suspect = None
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


def _is_dlq_retry_event(event: "RouterEvent") -> bool:
    """Check if an event originated from DLQ retry/replay.

    Feb 2026: Events re-published from DLQ retry should NOT be re-captured
    to the DLQ on failure, as this would create infinite retry loops.
    """
    return event.source.startswith("dlq_retry:") or event.source.startswith("dlq_replay:")


# Handler timeout to prevent hung event handlers from blocking dispatch
# Environment variable override: RINGRIFT_EVENT_HANDLER_TIMEOUT
# December 29, 2025: Increased from 30s to 600s (10 min) for autonomous operation
# - Gauntlet evaluation: 500s+ (50 games)
# - Data sync: 600s+ (network transfers)
# - NPZ export: 300s+ (sample encoding)
DEFAULT_HANDLER_TIMEOUT_SECONDS = float(
    os.environ.get("RINGRIFT_EVENT_HANDLER_TIMEOUT", "600.0")
)


class EventHandlerTimeout(Exception):
    """Exception raised when an event handler exceeds its timeout.

    Phase 5.1 (Dec 29, 2025): Handler timeouts are now raised instead of
    silently swallowed. This allows proper error tracking and DLQ capture.
    """

    def __init__(self, handler_name: str, event_type: str, timeout: float) -> None:
        self.handler_name = handler_name
        self.event_type = event_type
        self.timeout = timeout
        super().__init__(
            f"Handler '{handler_name}' timed out after {timeout}s "
            f"while processing {event_type}"
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
    CROSS_PROCESS_TO_DATA_MAP,
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

    Phase 5.1 (December 29, 2025): Added success and iteration fields
    for handler result tracking and pipeline iteration context.
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

    # Phase 5.1: Handler result and iteration tracking (December 29, 2025)
    success: bool = True  # Whether the event was processed successfully
    iteration: int = 0  # Pipeline iteration number for context

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
        # Dec 30, 2025: Counter for O(1) hash occurrence tracking (replaces O(n) deque search)
        self._hash_counts: Counter[str] = Counter()
        self._max_seen_events = max_seen_events
        self._duplicates_prevented = 0
        self._content_duplicates_prevented = 0  # Content-hash based duplicates
        self._handler_timeouts = 0  # Phase 5.1 (Dec 29, 2025): Track handler timeouts

        # Metrics
        self._events_routed: dict[str, int] = {}
        self._events_by_source: dict[str, int] = {}
        # Dec 29, 2025: Track cross-process degradation (when events fail to route)
        self._cross_process_failures: int = 0
        self._cross_process_degraded: bool = False
        self._last_cp_failure_time: float = 0.0

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

        # Bridge: DataEventBus -> Router (December 27, 2025)
        # This ensures events published via emit_host_offline(), emit_leader_elected(), etc.
        # reach coordinators that subscribe via get_router().subscribe()
        if HAS_DATA_EVENTS:
            data_bus = get_data_event_bus()
            # Subscribe to all DataEventType values
            for event_type in DataEventType:
                data_bus.subscribe(event_type, self._on_data_bus_event)

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

    async def start(self) -> None:
        """Start the event router.

        Note: The router auto-initializes during __init__ via _setup_bus_bridges().
        This method exists for consistency with the daemon lifecycle pattern,
        allowing daemon_runners.py to call start() uniformly on all daemons.

        December 27, 2025: Added to fix 'UnifiedEventRouter has no attribute start'.
        """
        logger.info("[UnifiedEventRouter] Router active (initialized during __init__)")

    async def stop(self) -> None:
        """Stop the event router and cleanup resources.

        December 27, 2025: Added for daemon lifecycle consistency.
        """
        if self._cp_poller:
            try:
                self._cp_poller.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                # RuntimeError - event loop issues, AttributeError - poller not initialized
                # OSError - I/O errors during cleanup
                logger.warning(f"[UnifiedEventRouter] Error stopping poller: {e}")

        # Shutdown thread pool (December 2025 - prevent resource leak)
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)

        logger.info("[UnifiedEventRouter] Router stopped")

    @property
    def is_running(self) -> bool:
        """Check if router is running (for daemon lifecycle compatibility)."""
        return True  # Router is always running after init

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

    async def _on_data_bus_event(self, event: DataEvent) -> None:
        """Handle events from DataEventBus (December 27, 2025).

        This bridges events published via emit_host_offline(), emit_leader_elected(),
        emit_data_sync_completed(), etc. to the UnifiedEventRouter so coordinators
        subscribing via get_router().subscribe() receive them.
        """
        router_event = RouterEvent(
            event_type=normalize_event_type(event.event_type.value),
            payload=event.payload,
            timestamp=event.timestamp if hasattr(event, "timestamp") else time.time(),
            source=event.source,
            origin=EventSource.DATA_BUS,
            data_event=event,
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
        # Dec 29 2025: Handle case where DataEvent object is passed instead of event type
        # This happens when code calls router.publish(DataEvent(...)) instead of
        # router.publish(event_type, payload). Extract the event type and payload.
        if hasattr(event_type, 'event_type') and hasattr(event_type, 'payload'):
            # This is a DataEvent object - extract its components
            data_event = event_type
            event_type = data_event.event_type
            if payload is None:
                payload = getattr(data_event, 'payload', {})
            if not source:
                source = getattr(data_event, 'source', '')

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

        # P0 December 2025: Validate event type against canonical set
        # This catches typos and misconfigurations that would silently break pipelines
        try:
            is_valid, validation_msg = validate_event_type(event_type_str)
            if not is_valid:
                logger.warning(
                    f"[EventRouter] {validation_msg}. Event will still be published "
                    f"but may not be received by subscribers."
                )
        except UnknownEventTypeError as e:
            # Strict mode - re-raise to caller
            logger.error(f"[EventRouter] Event validation failed: {e}")
            raise

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
            # Dec 2025: Try multiple strategies to resolve DataEventType:
            # 1. Direct value lookup (for lowercase enum values like "sync_completed")
            # 2. CROSS_PROCESS_TO_DATA_MAP (for canonical uppercase like "DATA_SYNC_COMPLETED")
            # 3. STAGE_TO_DATA_EVENT_MAP (for stage event names like "sync_complete")
            # 4. Enum name lookup (for uppercase enum names)
            try:
                data_event_type = DataEventType(event_type_str)
            except (ValueError, KeyError):
                # Try mapping from canonical uppercase to lowercase enum value
                data_value = CROSS_PROCESS_TO_DATA_MAP.get(event_type_str)
                if data_value:
                    try:
                        data_event_type = DataEventType(data_value)
                    except (ValueError, KeyError):
                        data_event_type = None

                # Try stage event mapping
                if data_event_type is None:
                    stage_mapped = STAGE_TO_DATA_EVENT_MAP.get(event_type_str)
                    if stage_mapped:
                        try:
                            data_event_type = DataEventType(stage_mapped)
                        except (ValueError, KeyError):
                            data_event_type = None

                # Try enum name lookup (DataEventType['DATA_SYNC_COMPLETED'])
                if data_event_type is None:
                    try:
                        data_event_type = DataEventType[event_type_str]
                    except KeyError:
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
            try:
                event_id = cp_publish(event_type_str, payload, source)
                # Dec 29, 2025: Track degradation when cross-process queue returns -1 (readonly)
                if event_id == -1:
                    self._cross_process_failures += 1
                    self._cross_process_degraded = True
                    self._last_cp_failure_time = time.time()
                    payload_size = len(json.dumps(payload or {})) if payload else 0
                    logger.warning(
                        f"[EventRouter] Cross-process degraded: {event_type_str} "
                        f"(payload_size={payload_size}B, failures={self._cross_process_failures})"
                    )
                elif self._cross_process_degraded:
                    # Recovered from degraded state
                    self._cross_process_degraded = False
                    logger.info("[EventRouter] Cross-process recovered from degradation")
            except Exception as e:
                self._cross_process_failures += 1
                self._cross_process_degraded = True
                self._last_cp_failure_time = time.time()
                logger.warning(f"[EventRouter] Cross-process publish failed: {e}")

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

    # Alias for backward compatibility with code using emit()
    # December 28, 2025: Added to provide API consistency
    emit = publish
    emit_sync = publish_sync

    async def _dispatch(
        self,
        event: RouterEvent,
        exclude_origin: bool = False,
    ) -> None:
        """Dispatch event to router subscribers with deduplication.

        Core event routing algorithm (December 2025):

        1. **Event-ID Deduplication**: Each RouterEvent has a unique UUID. Events
           already in _seen_events set are skipped to prevent duplicate processing.

        2. **Content-Hash Deduplication**: Events forwarded through multiple buses
           (stage, cross-process) may arrive as different RouterEvent objects with
           identical payloads. The content_hash (SHA256 of type+payload) catches
           these. Exception: Router-originated events can repeat same content.

        3. **LRU Eviction**: Both seen event IDs and content hashes are bounded
           by _max_seen_events (default 10000) using deque-based FIFO eviction.

        4. **Subscriber Invocation**: Callbacks from _global_subscribers (all events)
           and _subscribers[event_type] (type-specific) are called in registration
           order with timeout protection (DEFAULT_HANDLER_TIMEOUT_SECONDS).

        5. **Error Handling**: Failed handlers log errors and push to dead-letter
           queue (DLQ) for later analysis. SystemExit/KeyboardInterrupt propagate.

        6. **Metrics**: Tracks events_routed by type and events_by_source for
           debugging and monitoring.

        Args:
            event: The RouterEvent to dispatch
            exclude_origin: Deprecated, kept for API compatibility

        Note:
            This is an internal method. Use publish() or publish_sync() for
            public event emission.
        """
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
            self._hash_counts[content_hash] += 1  # O(1) increment

            # Bound the LRU deque even if we allow repeated router-origin events
            # with identical content hashes.
            while len(self._seen_hashes_order) > self._max_seen_events:
                oldest_hash = self._seen_hashes_order.popleft()  # O(1) with deque
                self._hash_counts[oldest_hash] -= 1  # O(1) decrement
                if self._hash_counts[oldest_hash] <= 0:  # O(1) check
                    self._seen_content_hashes.discard(oldest_hash)
                    del self._hash_counts[oldest_hash]  # Clean up counter entry

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
                        # Phase 5.1 (Dec 29, 2025): Capture handler timeouts to DLQ
                        callback_name = getattr(callback, "__name__", str(callback))
                        error_msg = (
                            f"Handler timeout: {callback_name} exceeded "
                            f"{DEFAULT_HANDLER_TIMEOUT_SECONDS}s"
                        )
                        logger.error(f"[EventRouter] {error_msg}")

                        # Track timeout for metrics
                        self._handler_timeouts += 1

                        # Capture to DLQ for retry/analysis
                        # Feb 2026: Skip capture for DLQ retry events to prevent infinite loops
                        if HAS_DLQ and get_dead_letter_queue and not _is_dlq_retry_event(event):
                            try:
                                dlq = get_dead_letter_queue()
                                dlq.capture(
                                    event_type=str(event.event_type),
                                    payload=event.payload,
                                    handler_name=callback_name,
                                    error=error_msg,
                                    source="EventRouter.async.timeout",
                                )
                            except (RuntimeError, AttributeError, ImportError) as dlq_err:
                                # DLQ capture failed - log at warning level (not debug)
                                logger.warning(
                                    f"[EventRouter] DLQ capture failed for timeout: {dlq_err}"
                                )
            except (SystemExit, KeyboardInterrupt):
                # Signal exceptions must propagate for graceful shutdown (Dec 2025)
                raise
            except asyncio.CancelledError:
                # Task cancellation must propagate
                raise
            except RecursionError as e:
                # Recursion in event handlers indicates infinite loop
                callback_name = getattr(callback, "__name__", str(callback))
                logger.critical(f"[EventRouter] Recursion in handler {callback_name}: {e}")
                raise
            except (NameError, AttributeError, TypeError) as e:
                # Dec 29, 2025: Programming errors - log CRITICAL for visibility
                # Jan 14, 2026: Add helpful suggestion for common .get() AttributeError
                callback_name = getattr(callback, "__name__", str(callback))
                error_str = str(e)
                if isinstance(e, AttributeError) and "'get'" in error_str:
                    # Common mistake: calling event.get() instead of get_event_payload(event).get()
                    logger.critical(
                        f"[EventRouter] Handler bug in {callback_name} for "
                        f"{event.event_type}: {type(e).__name__}: {e}. "
                        f"HINT: Use get_event_payload(event) to extract payload from RouterEvent"
                    )
                else:
                    logger.critical(
                        f"[EventRouter] Handler bug in {callback_name} for "
                        f"{event.event_type}: {type(e).__name__}: {e}"
                    )
                # Still capture to DLQ for analysis with elevated severity
                # Feb 2026: Skip capture for DLQ retry events to prevent infinite loops
                if HAS_DLQ and get_dead_letter_queue and not _is_dlq_retry_event(event):
                    try:
                        dlq = get_dead_letter_queue()
                        dlq.capture(
                            event_type=str(event.event_type),
                            payload=event.payload,
                            handler_name=callback_name,
                            error=f"BUG: {type(e).__name__}: {e}",
                            source="EventRouter.programming_error",
                        )
                    except (RuntimeError, AttributeError, ImportError):
                        pass  # DLQ capture is best-effort
            except Exception as e:  # noqa: BLE001 - expected runtime/data errors
                callback_name = getattr(callback, "__name__", str(callback))
                logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")
                # Capture to DLQ for retry/analysis (December 2025)
                # Feb 2026: Skip capture for DLQ retry events to prevent infinite loops
                if HAS_DLQ and get_dead_letter_queue and not _is_dlq_retry_event(event):
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
        """Synchronous dispatch for non-async contexts with deduplication.

        Identical to _dispatch() but uses threading locks instead of asyncio locks.
        Used when no event loop is running (e.g., signal handlers, module init).

        See _dispatch() for full algorithm documentation.

        Args:
            event: The RouterEvent to dispatch
            exclude_origin: Deprecated, kept for API compatibility
        """
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
            self._hash_counts[content_hash] += 1  # O(1) increment

            while len(self._seen_hashes_order) > self._max_seen_events:
                oldest_hash = self._seen_hashes_order.popleft()  # O(1) with deque
                self._hash_counts[oldest_hash] -= 1  # O(1) decrement
                if self._hash_counts[oldest_hash] <= 0:  # O(1) check
                    self._seen_content_hashes.discard(oldest_hash)
                    del self._hash_counts[oldest_hash]  # Clean up counter entry

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
                        except (SystemExit, KeyboardInterrupt):
                            # Signal exceptions propagate even in thread pool (Dec 2025)
                            raise
                        except Exception as e:  # noqa: BLE001 - only runtime errors reach here
                            logger.error(f"[EventRouter] Async callback failed: {e}")

                    # Use thread pool (initialized in __init__)
                    self._thread_pool.submit(run_async_in_thread, result)
                    logger.debug(
                        f"[EventRouter] Dispatched async callback {callback.__name__} "
                        f"for {event.event_type} to thread pool"
                    )
            except (SystemExit, KeyboardInterrupt):
                # Signal exceptions must propagate for graceful shutdown (Dec 2025)
                raise
            except RecursionError as e:
                # Recursion in event handlers indicates infinite loop
                callback_name = getattr(callback, "__name__", str(callback))
                logger.critical(f"[EventRouter] Recursion in sync handler {callback_name}: {e}")
                raise
            except Exception as e:  # noqa: BLE001 - only runtime errors reach here
                callback_name = getattr(callback, "__name__", str(callback))
                error_str = str(e)
                # Jan 14, 2026: Add helpful suggestion for common .get() AttributeError
                if isinstance(e, AttributeError) and "'get'" in error_str:
                    logger.error(
                        f"[EventRouter] Callback error for {event.event_type}: {e}. "
                        f"HINT: Use get_event_payload(event) to extract payload from RouterEvent"
                    )
                else:
                    logger.error(f"[EventRouter] Callback error for {event.event_type}: {e}")
                # Capture to DLQ for retry/analysis (December 2025)
                # Feb 2026: Skip capture for DLQ retry events to prevent infinite loops
                if HAS_DLQ and get_dead_letter_queue and not _is_dlq_retry_event(event):
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
        subscriber_name: str | None = None,
        persist: bool = False,
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: Event type to subscribe to. Can be:
                - String: "MODEL_PROMOTED", "TRAINING_COMPLETED"
                - DataEventType enum: DataEventType.MODEL_PROMOTED
                - StageEvent enum: StageEvent.SELFPLAY_COMPLETE
                - None: Subscribe to ALL events (global subscriber)
            callback: Async or sync function to call when event occurs
            subscriber_name: Optional name for persistence tracking (P0 Dec 2025)
            persist: Whether to persist subscription to SQLite (P0 Dec 2025)

        Important - Subscription Timing:
            Subscribers MUST be registered BEFORE emitters start to avoid
            missing early events. The recommended startup order is:
            1. EVENT_ROUTER starts first
            2. FEEDBACK_LOOP and DATA_PIPELINE (subscribers)
            3. AUTO_SYNC and other daemons (emitters)

            Events emitted before subscribers are ready go to the dead-letter
            queue and are retried by DLQ_RETRY daemon for eventual consistency.

        Example:
            router = get_router()
            router.subscribe(DataEventType.MODEL_PROMOTED, on_model_promoted)
            router.subscribe("TRAINING_COMPLETED", on_training_completed)
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

            # P0 Dec 2025: Persist subscription for restart recovery
            if persist and subscriber_name:
                self._persist_subscription(subscriber_name, event_type_str, callback)

    def _persist_subscription(
        self,
        subscriber_name: str,
        event_type: str,
        callback: EventCallback,
    ) -> None:
        """Persist subscription to SQLite store for restart recovery.

        P0 December 2025: Subscriptions are lost on process restart, causing
        events to be orphaned. This method saves subscription metadata.

        Args:
            subscriber_name: Name of subscribing component
            event_type: Normalized event type string
            callback: Handler function (used to extract module path)
        """
        try:
            from app.coordination.subscription_store import get_subscription_store

            # Build handler path from callback
            handler_path = f"{callback.__module__}:{callback.__qualname__}"

            store = get_subscription_store()
            store.register_subscription(
                subscriber_name=subscriber_name,
                event_type=event_type,
                handler_path=handler_path,
            )
        except ImportError:
            logger.debug(
                "[EventRouter] Subscription persistence unavailable - "
                "subscription_store not importable"
            )
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            # Dec 29: Narrowed from Exception to specific store errors
            logger.warning(
                f"[EventRouter] Failed to persist subscription: {e}"
            )

    def unsubscribe(
        self,
        event_type: str | DataEventType | StageEvent | None,
        callback: EventCallback,
        subscriber_name: str | None = None,
    ) -> bool:
        """Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from (None for global)
            callback: Handler to remove
            subscriber_name: Optional name for deactivating persisted subscription
        """
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

                # P0 Dec 2025: Deactivate persisted subscription
                if subscriber_name:
                    self._deactivate_persisted_subscription(subscriber_name, event_type_str)

                return True
        return False

    def _deactivate_persisted_subscription(
        self,
        subscriber_name: str,
        event_type: str,
    ) -> None:
        """Deactivate a persisted subscription.

        P0 December 2025: Mark subscription as inactive in the store.
        """
        try:
            from app.coordination.subscription_store import get_subscription_store

            store = get_subscription_store()
            store.deactivate_subscription(subscriber_name, event_type)
        except ImportError:
            pass  # Subscription store not available
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            # Dec 29: Narrowed from Exception to specific store errors
            logger.warning(
                f"[EventRouter] Failed to deactivate persisted subscription: {e}"
            )

    async def restore_subscriptions(self) -> int:
        """Restore subscriptions from persistent store on startup.

        P0 December 2025: Called during coordination bootstrap to restore
        subscriptions that were persisted before a restart.

        Returns:
            Number of subscriptions restored
        """
        try:
            from app.coordination.subscription_store import get_subscription_store
        except ImportError:
            logger.debug(
                "[EventRouter] Subscription persistence unavailable - "
                "cannot restore subscriptions"
            )
            return 0

        store = get_subscription_store()
        subscriptions = store.get_active_subscriptions()

        restored = 0
        for sub in subscriptions:
            try:
                # Try to import the handler
                handler = self._load_handler_from_path(sub.handler_path)
                if handler:
                    # Re-subscribe (don't persist again - already in store)
                    self.subscribe(sub.event_type, handler)
                    restored += 1
                    logger.debug(
                        f"[EventRouter] Restored: {sub.subscriber_name} -> {sub.event_type}"
                    )
                else:
                    logger.warning(
                        f"[EventRouter] Could not load handler for {sub.subscriber_name}: "
                        f"{sub.handler_path}"
                    )
            except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError) as e:
                # Dec 29: Narrowed from Exception to specific handler loading errors
                logger.warning(
                    f"[EventRouter] Failed to restore subscription {sub.subscriber_name}: {e}"
                )

        if restored > 0:
            logger.info(
                f"[EventRouter] Restored {restored} subscriptions from persistent store"
            )

        return restored

    def _load_handler_from_path(self, handler_path: str) -> EventCallback | None:
        """Load a handler function from its module path.

        Args:
            handler_path: Path like "app.coordination.foo:my_handler"

        Returns:
            Handler callable or None if not loadable
        """
        try:
            if ":" not in handler_path:
                return None

            module_path, func_name = handler_path.rsplit(":", 1)

            import importlib
            module = importlib.import_module(module_path)

            # Handle nested qualnames like "ClassName.method_name"
            handler = module
            for attr in func_name.split("."):
                handler = getattr(handler, attr)

            if callable(handler):
                return handler
            return None

        except (ImportError, AttributeError) as e:
            logger.debug(f"[EventRouter] Could not load handler {handler_path}: {e}")
            return None

    async def replay_stale_dlq_events(self, min_age_seconds: float = 300) -> int:
        """Replay DLQ events that haven't been processed.

        P0 December 2025: Called during startup to ensure events aren't lost
        due to process restarts.

        Args:
            min_age_seconds: Minimum age for events to replay (default: 5 minutes)

        Returns:
            Number of events replayed

        Environment:
            RINGRIFT_DISABLE_DLQ_REPLAY: Set to "1" to disable DLQ replay (for tests)
        """
        # Check for test mode to avoid blocking on sync operations
        if os.environ.get("RINGRIFT_DISABLE_DLQ_REPLAY", "").lower() in ("1", "true", "yes"):
            logger.debug("[EventRouter] DLQ replay disabled via environment variable")
            return 0

        try:
            from app.coordination.subscription_store import get_subscription_store

            store = get_subscription_store()
            return await store.replay_stale_dlq_events(min_age_seconds=min_age_seconds)
        except ImportError:
            logger.debug(
                "[EventRouter] Subscription store unavailable - cannot replay DLQ"
            )
            return 0
        except (RuntimeError, OSError, ValueError, TypeError, asyncio.CancelledError) as e:
            # Dec 29: Narrowed from Exception to specific store/async errors
            logger.warning(f"[EventRouter] Failed to replay DLQ events: {e}")
            return 0

    async def _dlq_replay_loop(self) -> None:
        """Background loop that periodically replays DLQ events.

        Phase 6 - December 29, 2025: Auto-replay DLQ events with exponential backoff.

        Runs every _dlq_replay_interval seconds and replays events older than 5 minutes.
        Events with retry_count >= 5 are skipped (permanent failure).
        """
        logger.info("[EventRouter] DLQ auto-replay loop started")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                await asyncio.sleep(self._dlq_replay_interval)

                # Replay stale events (older than 5 minutes)
                replayed = await self.replay_stale_dlq_events(min_age_seconds=300)
                if replayed > 0:
                    self._dlq_events_replayed += replayed
                    logger.info(f"[EventRouter] DLQ auto-replay: {replayed} events replayed")

                # Reset error counter on success
                consecutive_errors = 0

            except asyncio.CancelledError:
                logger.info("[EventRouter] DLQ auto-replay loop cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    f"[EventRouter] DLQ replay loop error ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                # Exponential backoff on errors
                if consecutive_errors >= max_consecutive_errors:
                    backoff = min(300, self._dlq_replay_interval * (2 ** consecutive_errors))
                    logger.warning(f"[EventRouter] DLQ replay backing off for {backoff}s")
                    await asyncio.sleep(backoff)

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

    def has_subscribers(self, event_type: str | DataEventType | StageEvent) -> bool:
        """Check if an event type has any subscribers.

        December 2025: Added to support startup verification that critical
        event subscribers are wired before emitters start.

        Args:
            event_type: Event type to check (string or enum)

        Returns:
            True if at least one subscriber exists for this event type
        """
        if hasattr(event_type, 'value'):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        event_type_str = normalize_event_type(event_type_str)
        return bool(self._subscribers.get(event_type_str))

    def reset(self) -> None:
        """Clear all subscriptions and state (for testing).

        Clears:
        - All event type subscriptions
        - All global subscribers
        - Event history
        - Seen events set (deduplication)

        December 2025: Added for singleton registry test cleanup.
        """
        self._subscribers.clear()
        self._global_subscribers.clear()
        self._event_history.clear()
        self._seen_events.clear()
        self._seen_events_order.clear()
        self._seen_content_hashes.clear()
        self._seen_hashes_order.clear()
        self._hash_counts.clear()  # Dec 30, 2025: Clear O(1) counter
        # Also clear statistics
        self._events_routed.clear()
        self._events_by_source.clear()
        logger.debug("[EventRouter] Reset: cleared all subscriptions and state")

    def unsubscribe_all(self, event_type: str | DataEventType | StageEvent | None = None) -> int:
        """Remove all subscribers for an event type (or all if None).

        Args:
            event_type: Event type to clear subscribers for.
                       If None, clears ALL subscribers (global and type-specific).

        Returns:
            Number of subscribers removed.

        December 2025: Added for targeted test cleanup.
        """
        count = 0

        if event_type is None:
            # Clear all subscribers
            for subs_list in self._subscribers.values():
                count += len(subs_list)
            count += len(self._global_subscribers)
            self._subscribers.clear()
            self._global_subscribers.clear()
        else:
            # Clear subscribers for specific event type
            if hasattr(event_type, 'value'):
                event_type_str = event_type.value
            else:
                event_type_str = str(event_type)

            event_type_str = normalize_event_type(event_type_str)

            if event_type_str in self._subscribers:
                count = len(self._subscribers[event_type_str])
                del self._subscribers[event_type_str]

        logger.debug(f"[EventRouter] Unsubscribed {count} callbacks for {event_type}")
        return count

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
            # Handler timeout metrics (Phase 5.1 - Dec 29, 2025)
            "handler_timeouts": self._handler_timeouts,
            # Cross-process degradation metrics (Dec 29, 2025)
            "cross_process_failures": self._cross_process_failures,
            "cross_process_degraded": self._cross_process_degraded,
            "last_cp_failure_time": self._last_cp_failure_time,
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
            # Dec 29, 2025: Cross-process degradation monitoring
            "cross_process_status": {
                "degraded": stats.get("cross_process_degraded", False),
                "failures": stats.get("cross_process_failures", 0),
                "last_failure_time": stats.get("last_cp_failure_time", 0.0),
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


def has_subscribers(event_type: str) -> bool:
    """Check if an event type has any subscribers.

    December 2025: Module-level helper for startup verification.

    Args:
        event_type: Event type string to check

    Returns:
        True if at least one subscriber exists for this event type
    """
    return get_router().has_subscribers(event_type)


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
    # Exceptions (Phase 5.1 - Dec 29, 2025)
    "EventHandlerTimeout",
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
    "emit_node_suspect",
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
def get_event_bus() -> "EventBus | None":
    """Get the data event bus (re-exported for backward compatibility)."""
    if HAS_DATA_EVENTS:
        return get_data_event_bus()
    return None


# =============================================================================
# Module Validation
# =============================================================================


def _validate_router_class() -> None:
    """Verify UnifiedEventRouter has all required methods.

    December 27, 2025: Added to catch circular import issues that leave the
    class definition incomplete. This is called at module load time.

    If methods are missing (due to circular import during class definition),
    we emit a warning rather than raising an exception to allow the module
    to load partially and provide better error messages later.
    """
    required_methods = ["start", "stop", "publish", "subscribe", "is_running"]
    missing = [m for m in required_methods if not hasattr(UnifiedEventRouter, m)]
    if missing:
        import warnings

        warnings.warn(
            f"UnifiedEventRouter missing methods: {missing}. "
            "This may indicate a circular import issue.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            f"[event_router] UnifiedEventRouter missing required methods: {missing}"
        )


# Run validation at module load time
_validate_router_class()


# =============================================================================
# Backward Compatibility Layer for unified_event_coordinator.py
# These aliases allow code that imported from unified_event_coordinator to
# work with the consolidated event_router instead.
# =============================================================================

@dataclass
class EventRouterStats:
    """Statistics for the event router.

    December 28, 2025: Renamed from CoordinatorStats to avoid collision
    with the generic CoordinatorStats in coordinator_base.py.
    """
    events_bridged_data_to_cross: int = 0
    events_bridged_stage_to_cross: int = 0
    events_bridged_cross_to_data: int = 0
    events_dropped: int = 0
    last_bridge_time: str | None = None
    errors: list[str] = field(default_factory=list)
    start_time: str | None = None
    is_running: bool = False


# Aliases for backward compatibility
UnifiedEventCoordinator = UnifiedEventRouter
EventRouter = UnifiedEventRouter  # Common alias used in documentation
CoordinatorStats = EventRouterStats  # Backward-compat alias (deprecated Dec 28, 2025)


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


def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "unknown",
    log_on_failure: bool = True,
) -> bool:
    """Safely emit an event without raising exceptions.

    Wraps event emission in try-catch to prevent event failures from crashing
    the caller. Works in both sync and async contexts.

    This helper eliminates the repetitive try/except boilerplate:
        try:
            emit_something(...)
        except Exception as e:
            logger.debug(f"Event failed: {e}")

    Instead use:
        safe_emit_event("MY_EVENT", {"key": "value"}, source="my_component")

    Args:
        event_type: Event type string (e.g., "TRAINING_COMPLETED")
        payload: Event payload dict (default: empty dict)
        source: Source identifier for logging
        log_on_failure: Whether to log failures (default: True)

    Returns:
        True if event was emitted successfully, False otherwise

    December 2025: Added to reduce 960+ LOC of boilerplate across codebase.
    """
    if payload is None:
        payload = {}

    try:
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        async def _emit():
            await publish(event_type=event_type, payload=payload, source=source)

        if loop is not None:
            # Async context - fire and forget
            from app.core.async_context import fire_and_forget

            fire_and_forget(_emit(), name=f"safe_emit_{event_type}")
        else:
            # Sync context - create new loop
            asyncio.run(_emit())

        return True

    except (ImportError, RuntimeError, OSError, AttributeError) as e:
        if log_on_failure:
            logger.debug(f"[{source}] Event {event_type} emission failed: {e}")
        return False
    except Exception as e:
        # Catch-all for unexpected errors to ensure caller never crashes
        if log_on_failure:
            logger.debug(f"[{source}] Event {event_type} emission error: {e}")
        return False


def emit_event(
    event_type: str | Any,  # Can be string or DataEventType enum
    payload: dict[str, Any] | None = None,
    source: str = "unknown",
) -> bool:
    """Emit an event to the event system.

    Convenience wrapper around safe_emit_event that handles enum types.

    Args:
        event_type: Event type string or DataEventType enum
        payload: Event payload dict (default: empty dict)
        source: Source identifier for logging

    Returns:
        True if event was emitted successfully, False otherwise
    """
    # Convert enum to string if needed
    event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
    return safe_emit_event(event_type_str, payload, source)


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


async def emit_evaluation_started(
    model_path: str,
    board_type: str,
    num_players: int,
    config_key: str | None = None,
    **extra_payload
) -> None:
    """Emit EVALUATION_STARTED event to all systems.

    December 30, 2025: Added for Gap #3 integration fix.
    Enables metrics tracking and coordination when evaluation begins.
    """
    if config_key is None:
        config_key = make_config_key(board_type, num_players)

    payload = {
        "model_path": model_path,
        "board_type": board_type,
        "num_players": num_players,
        "config_key": config_key,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra_payload)

    await publish(
        event_type="EVALUATION_STARTED",
        payload=payload,
        source="evaluation",
    )


async def emit_evaluation_completed(
    model_id: str | None = None,
    elo: float | None = None,
    win_rate: float = 0.0,
    games_played: int = 0,
    # December 30, 2025: Added explicit parameters for evaluation_daemon compatibility
    model_path: str | None = None,
    board_type: str | None = None,
    num_players: int | None = None,
    opponent_results: dict | None = None,
    harness_results: dict | None = None,
    best_harness: str | None = None,
    best_elo: float | None = None,
    composite_participant_ids: list | None = None,
    is_multi_harness: bool = False,
    # December 30, 2025: Architecture for multi-architecture training support
    architecture: str | None = None,
    **extra_payload
) -> None:
    """Emit EVALUATION_COMPLETED event to all systems.

    December 30, 2025: Extended with multi-harness evaluation support and
    multi-architecture training tracking.
    - harness_results: Dict of harness_name -> {elo, win_rate, games_played, composite_participant_id}
    - composite_participant_ids: List of composite IDs for (model, harness) combinations
    - is_multi_harness: True if evaluated under multiple harnesses
    - architecture: Model architecture (v2, v3, v4, v5, v5_heavy, etc.)
    """
    # Use model_path as model_id if not provided
    effective_model_id = model_id or model_path or "unknown"
    # Use best_elo if available, otherwise passed elo
    effective_elo = best_elo if best_elo is not None else (elo or 0.0)

    payload = {
        "model_id": effective_model_id,
        "model_path": model_path,
        "elo": effective_elo,
        "win_rate": win_rate,
        "games_played": games_played,
        "timestamp": datetime.now().isoformat(),
    }

    # Add optional fields if provided
    if board_type is not None:
        payload["board_type"] = board_type
    if num_players is not None:
        payload["num_players"] = num_players
    # Feb 24, 2026: config_key is required by auto_promotion_daemon
    if board_type is not None and num_players is not None and "config_key" not in payload:
        payload["config_key"] = f"{board_type}_{num_players}p"
    if opponent_results is not None:
        payload["opponent_results"] = opponent_results
    if harness_results is not None:
        payload["harness_results"] = harness_results
    if best_harness is not None:
        payload["best_harness"] = best_harness
    if best_elo is not None:
        payload["best_elo"] = best_elo
    if composite_participant_ids is not None:
        payload["composite_participant_ids"] = composite_participant_ids
    if is_multi_harness:
        payload["is_multi_harness"] = is_multi_harness
    # December 30, 2025: Add architecture for multi-architecture support
    if architecture is not None:
        payload["architecture"] = architecture

    # Add any extra payload
    payload.update(extra_payload)

    await publish(
        event_type="EVALUATION_COMPLETED",
        payload=payload,
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
