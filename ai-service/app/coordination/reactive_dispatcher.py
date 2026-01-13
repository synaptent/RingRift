"""Reactive Selfplay Dispatcher (December 2025).

Dispatches selfplay immediately in response to cluster events, replacing
polling-based detection with event-driven dispatch.

Part of 48-hour autonomous operation optimization:
- Reduces event-to-dispatch latency from 60-120s to <5s
- Respects backpressure to prevent overload
- Uses asyncio.Queue for event serialization
- 10s deduplication window prevents duplicate dispatches

Key events handled:
- NODE_RECOVERED: Priority 70 - Node back online, needs work
- TRAINING_COMPLETED: Priority 80 - Training done, selfplay capacity freed
- IDLE_RESOURCE_DETECTED: Priority 50 - GPU idle, spawn selfplay
- BACKPRESSURE_RELEASED: Priority 65 - Cluster can handle more work
- HOST_ONLINE: Priority 60 - New host joined cluster

Usage:
    from app.coordination.reactive_dispatcher import ReactiveDispatcher

    dispatcher = ReactiveDispatcher.get_instance()
    await dispatcher.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.coordination.event_handler_utils import extract_config_key
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.p2p_integration import get_p2p_leader_url
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


@dataclass
class DispatchEvent:
    """Represents an event that may trigger selfplay dispatch."""

    event_type: str
    priority: int  # Higher = more important
    timestamp: float = field(default_factory=time.time)
    config_key: str | None = None
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: DispatchEvent) -> bool:
        """Priority queue ordering - higher priority first."""
        return self.priority > other.priority


@dataclass
class ReactiveDispatcherConfig:
    """Configuration for ReactiveDispatcher."""

    enabled: bool = True
    # Deduplication window (seconds) - skip if same event type within this window
    dedup_window_seconds: float = 10.0
    # Maximum events in queue before dropping low-priority events
    max_queue_size: int = 100
    # Backpressure threshold - pause dispatching if queue exceeds this
    backpressure_threshold: int = 50
    # Dispatch cooldown per config (seconds)
    per_config_cooldown_seconds: float = 5.0
    # P2P orchestrator endpoint for dispatch
    p2p_dispatch_endpoint: str = "/dispatch_selfplay"
    # P2P orchestrator port
    p2p_port: int = 8770
    # Timeout for dispatch requests (seconds)
    # Jan 2026: Increased from 30s to 120s - P2P startup takes >2 minutes
    dispatch_timeout_seconds: float = 120.0
    # Whether to respect cluster backpressure signals
    respect_backpressure: bool = True


# Event type to priority mapping
EVENT_PRIORITIES: dict[str, int] = {
    "training_completed": 80,  # High - training freed resources
    "node_recovered": 70,  # High - node needs work
    "backpressure_released": 65,  # Medium-high - can accept more
    "host_online": 60,  # Medium - new capacity
    "idle_resource_detected": 50,  # Medium - GPU idle
}


class ReactiveDispatcher(HandlerBase):
    """Dispatches selfplay immediately in response to cluster events.

    December 2025: Part of 48-hour autonomous operation optimization.
    Replaces polling-based dispatch with event-driven dispatch.

    Inherits from HandlerBase providing:
    - Automatic event subscription via _get_event_subscriptions()
    - Singleton pattern via get_instance()
    - Standardized health check format
    - Lifecycle management (start/stop)
    """

    def __init__(self, config: ReactiveDispatcherConfig | None = None):
        self._dispatcher_config = config or ReactiveDispatcherConfig()
        super().__init__(
            name="reactive_dispatcher",
            config=self._dispatcher_config,
            cycle_interval=60.0,  # Fallback check every minute
        )
        # Event queue for serialization
        self._event_queue: asyncio.PriorityQueue[DispatchEvent] = asyncio.PriorityQueue(
            maxsize=self._dispatcher_config.max_queue_size
        )
        # Deduplication tracking: event_type -> last_dispatch_time
        self._last_dispatch: dict[str, float] = {}
        # Per-config cooldown tracking: config_key -> last_dispatch_time
        self._config_cooldowns: dict[str, float] = {}
        # Backpressure state
        self._under_backpressure = False
        # Stats
        self._events_received = 0
        self._events_dispatched = 0
        self._events_deduplicated = 0
        self._events_dropped_backpressure = 0
        # Worker task
        self._worker_task: asyncio.Task | None = None

    @property
    def config(self) -> ReactiveDispatcherConfig:
        """Get the dispatcher configuration."""
        return self._dispatcher_config

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        Subscribes to cluster events that indicate selfplay dispatch opportunity.
        """
        return {
            "node_recovered": self._on_node_recovered,
            "training_completed": self._on_training_completed,
            "idle_resource_detected": self._on_idle_resource,
            "backpressure_released": self._on_backpressure_released,
            "backpressure_activated": self._on_backpressure_activated,
            "host_online": self._on_host_online,
        }

    async def _on_start(self) -> None:
        """Start the event processing worker."""
        if not self._dispatcher_config.enabled:
            logger.info("[ReactiveDispatcher] Disabled by configuration")
            return

        # Start worker task
        self._worker_task = safe_create_task(
            self._process_events(),
            name="reactive_dispatcher_worker",
        )
        logger.info("[ReactiveDispatcher] Started event processing worker")

    async def _on_stop(self) -> None:
        """Stop the event processing worker."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[ReactiveDispatcher] Stopped")

    # ========== Event Handlers ==========

    async def _on_node_recovered(self, event: Any) -> None:
        """Handle node recovery - dispatch selfplay to recovered node."""
        # Jan 2026: Use _get_payload() to handle both RouterEvent and dict types
        payload = self._get_payload(event)
        await self._enqueue_event(
            event_type="node_recovered",
            node_id=payload.get("node_id"),
            metadata={"event": str(event)[:200]},
        )

    async def _on_training_completed(self, event: Any) -> None:
        """Handle training completion - GPU resources freed."""
        payload = self._get_payload(event)
        config_key = extract_config_key(payload)

        await self._enqueue_event(
            event_type="training_completed",
            config_key=config_key,
            metadata={"event": str(event)[:200]},
        )

    async def _on_idle_resource(self, event: Any) -> None:
        """Handle idle resource detection - spawn selfplay on idle GPU."""
        node_id = None
        if hasattr(event, "node_id"):
            node_id = event.node_id
        elif hasattr(event, "payload"):
            node_id = event.payload.get("node_id")
        elif isinstance(event, dict):
            node_id = event.get("node_id")

        await self._enqueue_event(
            event_type="idle_resource_detected",
            node_id=node_id,
            metadata={"event": str(event)[:200]},
        )

    async def _on_backpressure_released(self, event: Any) -> None:
        """Handle backpressure release - cluster can accept more work."""
        self._under_backpressure = False
        await self._enqueue_event(
            event_type="backpressure_released",
            metadata={"event": str(event)[:200]},
        )

    async def _on_backpressure_activated(self, event: Any) -> None:
        """Handle backpressure activation - pause dispatching."""
        self._under_backpressure = True
        logger.info("[ReactiveDispatcher] Backpressure activated, pausing dispatch")

    async def _on_host_online(self, event: Any) -> None:
        """Handle new host coming online."""
        node_id = None
        if hasattr(event, "node_id"):
            node_id = event.node_id
        elif hasattr(event, "payload"):
            node_id = event.payload.get("node_id") or event.payload.get("host")
        elif isinstance(event, dict):
            node_id = event.get("node_id") or event.get("host")

        await self._enqueue_event(
            event_type="host_online",
            node_id=node_id,
            metadata={"event": str(event)[:200]},
        )

    # ========== Event Processing ==========

    async def _enqueue_event(
        self,
        event_type: str,
        config_key: str | None = None,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Enqueue an event for processing with deduplication."""
        self._events_received += 1

        # Check deduplication window
        now = time.time()
        last_time = self._last_dispatch.get(event_type, 0.0)
        if now - last_time < self._dispatcher_config.dedup_window_seconds:
            self._events_deduplicated += 1
            logger.debug(
                f"[ReactiveDispatcher] Deduplicated {event_type} "
                f"(within {self._dispatcher_config.dedup_window_seconds}s window)"
            )
            return

        # Check backpressure for low-priority events
        priority = EVENT_PRIORITIES.get(event_type, 40)
        if (
            self._dispatcher_config.respect_backpressure
            and self._under_backpressure
            and priority < 60
        ):
            self._events_dropped_backpressure += 1
            logger.debug(
                f"[ReactiveDispatcher] Dropped {event_type} due to backpressure "
                f"(priority {priority} < 60)"
            )
            return

        # Create and enqueue event
        event = DispatchEvent(
            event_type=event_type,
            priority=priority,
            config_key=config_key,
            node_id=node_id,
            metadata=metadata or {},
        )

        try:
            self._event_queue.put_nowait(event)
            self._last_dispatch[event_type] = now
            logger.debug(
                f"[ReactiveDispatcher] Enqueued {event_type} "
                f"(priority={priority}, queue_size={self._event_queue.qsize()})"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"[ReactiveDispatcher] Queue full, dropping {event_type}"
            )

    async def _process_events(self) -> None:
        """Worker loop that processes events from the queue."""
        logger.info("[ReactiveDispatcher] Event processing worker started")

        while self._running:
            try:
                # Wait for next event with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the event
                await self._dispatch_for_event(event)
                self._event_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"[ReactiveDispatcher] Error processing event: {e}")
                await asyncio.sleep(1.0)

    async def _dispatch_for_event(self, event: DispatchEvent) -> None:
        """Dispatch selfplay based on the event."""
        # Check per-config cooldown
        if event.config_key:
            last_config_dispatch = self._config_cooldowns.get(event.config_key, 0.0)
            if time.time() - last_config_dispatch < self._dispatcher_config.per_config_cooldown_seconds:
                logger.debug(
                    f"[ReactiveDispatcher] Config {event.config_key} in cooldown, skipping"
                )
                return

        logger.info(
            f"[ReactiveDispatcher] Dispatching selfplay for {event.event_type} "
            f"(config={event.config_key}, node={event.node_id})"
        )

        try:
            # Call P2P orchestrator dispatch endpoint
            success = await self._call_dispatch_endpoint(event)
            if success:
                self._events_dispatched += 1
                if event.config_key:
                    self._config_cooldowns[event.config_key] = time.time()
            else:
                logger.warning(
                    f"[ReactiveDispatcher] Dispatch failed for {event.event_type}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ReactiveDispatcher] Dispatch error: {e}")

    async def _call_dispatch_endpoint(self, event: DispatchEvent) -> bool:
        """Call the P2P orchestrator dispatch endpoint."""
        try:
            import aiohttp

            # Get the actual P2P leader URL (not hardcoded localhost)
            leader_url = await get_p2p_leader_url()
            if not leader_url:
                logger.warning("[ReactiveDispatcher] P2P leader unknown, cannot dispatch")
                return False

            url = f"{leader_url}{self._dispatcher_config.p2p_dispatch_endpoint}"

            payload = {
                "event_type": event.event_type,
                "config_key": event.config_key,
                "node_id": event.node_id,
                "priority": event.priority,
                "source": "reactive_dispatcher",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._dispatcher_config.dispatch_timeout_seconds),
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.warning(
                            f"[ReactiveDispatcher] Dispatch returned {response.status}"
                        )
                        return False

        except ImportError:
            logger.warning("[ReactiveDispatcher] aiohttp not available, skipping HTTP dispatch")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"[ReactiveDispatcher] HTTP dispatch error: {e}")
            return False

    async def _run_cycle(self) -> None:
        """Fallback cycle - check for stuck state."""
        if not self._dispatcher_config.enabled:
            return

        # Log stats periodically
        if self._events_received > 0:
            logger.info(
                f"[ReactiveDispatcher] Stats: received={self._events_received}, "
                f"dispatched={self._events_dispatched}, "
                f"deduplicated={self._events_deduplicated}, "
                f"dropped_backpressure={self._events_dropped_backpressure}, "
                f"queue_size={self._event_queue.qsize()}"
            )

    def health_check(self) -> HealthCheckResult:
        """Check dispatcher health.

        Returns:
            HealthCheckResult with status and details
        """
        from app.coordination.contracts import CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="ReactiveDispatcher not running",
            )

        if not self._dispatcher_config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="ReactiveDispatcher disabled by configuration",
                details={"enabled": False},
            )

        # Check if worker is alive
        worker_alive = self._worker_task is not None and not self._worker_task.done()
        if not worker_alive:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="ReactiveDispatcher worker not running",
                details=self.get_stats(),
            )

        # Check queue health
        queue_size = self._event_queue.qsize()
        if queue_size > self._dispatcher_config.backpressure_threshold:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"ReactiveDispatcher queue backlogged ({queue_size} events)",
                details=self.get_stats(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"ReactiveDispatcher running ({self._events_dispatched} dispatched)",
            details=self.get_stats(),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "running": self._running,
            "enabled": self._dispatcher_config.enabled,
            "events_received": self._events_received,
            "events_dispatched": self._events_dispatched,
            "events_deduplicated": self._events_deduplicated,
            "events_dropped_backpressure": self._events_dropped_backpressure,
            "queue_size": self._event_queue.qsize(),
            "under_backpressure": self._under_backpressure,
            "worker_alive": self._worker_task is not None and not self._worker_task.done(),
        }


# Singleton accessors
def get_reactive_dispatcher() -> ReactiveDispatcher:
    """Get or create the singleton ReactiveDispatcher instance."""
    return ReactiveDispatcher.get_instance()


def reset_reactive_dispatcher() -> None:
    """Reset the singleton instance (for testing)."""
    ReactiveDispatcher.reset_instance()


async def start_reactive_dispatcher() -> ReactiveDispatcher:
    """Start the reactive dispatcher (convenience function)."""
    dispatcher = get_reactive_dispatcher()
    await dispatcher.start()
    return dispatcher


__all__ = [
    "DispatchEvent",
    "ReactiveDispatcher",
    "ReactiveDispatcherConfig",
    "get_reactive_dispatcher",
    "reset_reactive_dispatcher",
    "start_reactive_dispatcher",
]
