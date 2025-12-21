"""Unified Event Coordinator - Bridge between event systems.

This module provides a coordinator that bridges events between:
1. EventBus (app.distributed.data_events) - In-memory async events
2. StageEventBus (app.coordination.stage_events) - Pipeline stage events
3. CrossProcessEventQueue (app.coordination.cross_process_events) - SQLite-backed IPC

The coordinator enables:
- Automatic forwarding of events between systems
- Event translation (mapping between different event type enums)
- Centralized event monitoring and logging
- Quality and training event propagation across process boundaries

Usage:
    from app.coordination.unified_event_coordinator import (
        UnifiedEventCoordinator,
        get_event_coordinator,
        start_coordinator,
    )

    # Get the singleton coordinator
    coordinator = get_event_coordinator()

    # Start background bridging
    await coordinator.start()

    # Events are automatically bridged between systems
    # For example, TRAINING_COMPLETED from data_events will be
    # forwarded to CrossProcessEventQueue for other processes

    # Stop when done
    await coordinator.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Event Type Mappings (centralized in event_mappings.py)
# =============================================================================

# Import centralized event mappings (DRY consolidation - December 2025)
from app.coordination.event_mappings import (
    CROSS_PROCESS_TO_DATA_MAP,
    DATA_TO_CROSS_PROCESS_MAP,
    STAGE_TO_CROSS_PROCESS_MAP,
)


# =============================================================================
# Import Event Systems (with graceful fallback)
# =============================================================================

# DataEventBus
try:
    from app.distributed.data_events import (
        DataEvent,
        DataEventType,
        EventBus as DataEventBus,
        get_event_bus as get_data_event_bus,
    )
    HAS_DATA_EVENTS = True
except ImportError:
    HAS_DATA_EVENTS = False
    DataEventType = None
    DataEvent = None
    DataEventBus = None

    def get_data_event_bus():
        return None

# StageEventBus
try:
    from app.coordination.stage_events import (
        StageCompletionResult,
        StageEvent,
        StageEventBus,
        get_event_bus as get_stage_event_bus,
    )
    HAS_STAGE_EVENTS = True
except ImportError:
    HAS_STAGE_EVENTS = False
    StageEvent = None
    StageCompletionResult = None
    StageEventBus = None

    def get_stage_event_bus():
        return None

# CrossProcessEventQueue
try:
    from app.coordination.cross_process_events import (
        CrossProcessEvent,
        CrossProcessEventQueue,
        get_event_queue as get_cross_process_queue,
    )
    HAS_CROSS_PROCESS = True
except ImportError:
    HAS_CROSS_PROCESS = False
    CrossProcessEvent = None
    CrossProcessEventQueue = None

    def get_cross_process_queue():
        return None


# =============================================================================
# Coordinator Stats
# =============================================================================

@dataclass
class CoordinatorStats:
    """Statistics for the event coordinator."""
    events_bridged_data_to_cross: int = 0
    events_bridged_stage_to_cross: int = 0
    events_bridged_cross_to_data: int = 0
    events_dropped: int = 0
    last_bridge_time: str | None = None
    errors: list[str] = field(default_factory=list)
    start_time: str | None = None
    is_running: bool = False


# =============================================================================
# Unified Event Coordinator
# =============================================================================

class UnifiedEventCoordinator:
    """Coordinates events between different event systems.

    This coordinator:
    1. Subscribes to DataEventBus and forwards relevant events to CrossProcess
    2. Subscribes to StageEventBus and forwards to CrossProcess
    3. Polls CrossProcess and forwards back to DataEventBus
    4. Provides centralized monitoring of all event traffic
    """

    _instance: UnifiedEventCoordinator | None = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize the coordinator."""
        self._data_bus: DataEventBus | None = None
        self._stage_bus: StageEventBus | None = None
        self._cross_queue: CrossProcessEventQueue | None = None

        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._subscriber_id: str | None = None

        self._stats = CoordinatorStats()
        self._event_handlers: dict[str, list[Callable]] = {}

        # Events we're bridging (avoid re-forwarding)
        self._recently_bridged: set[str] = set()
        self._bridge_dedup_window = 5.0  # seconds

    @classmethod
    def get_instance(cls) -> UnifiedEventCoordinator:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def start(self) -> bool:
        """Start the coordinator.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Coordinator already running")
            return True

        self._stats.start_time = datetime.now().isoformat()

        # Initialize event buses
        if HAS_DATA_EVENTS:
            self._data_bus = get_data_event_bus()
            if self._data_bus:
                self._subscribe_to_data_events()
                logger.info("Subscribed to DataEventBus")

        if HAS_STAGE_EVENTS:
            self._stage_bus = get_stage_event_bus()
            if self._stage_bus:
                self._subscribe_to_stage_events()
                logger.info("Subscribed to StageEventBus")

        if HAS_CROSS_PROCESS:
            self._cross_queue = get_cross_process_queue()
            if self._cross_queue:
                # Register as subscriber
                self._subscriber_id = self._cross_queue.subscribe(
                    "unified_coordinator",
                    list(CROSS_PROCESS_TO_DATA_MAP.keys()),
                )
                logger.info(f"Subscribed to CrossProcess queue: {self._subscriber_id}")

        self._running = True
        self._stats.is_running = True

        # Start polling task for cross-process events
        if self._cross_queue:
            self._poll_task = asyncio.create_task(self._poll_cross_process_loop())

        logger.info("UnifiedEventCoordinator started")
        return True

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        self._stats.is_running = False

        if self._poll_task:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        if self._cross_queue and self._subscriber_id:
            self._cross_queue.unsubscribe(self._subscriber_id)
            self._subscriber_id = None

        logger.info("UnifiedEventCoordinator stopped")

    def _subscribe_to_data_events(self) -> None:
        """Subscribe to relevant DataEventBus events."""
        if not self._data_bus or DataEventType is None:
            return

        # Subscribe to events we want to bridge to cross-process
        for data_event_value in DATA_TO_CROSS_PROCESS_MAP:
            try:
                event_type = DataEventType(data_event_value)
                self._data_bus.subscribe(event_type, self._handle_data_event)
            except ValueError:
                logger.debug(f"DataEventType not found: {data_event_value}")

    def _subscribe_to_stage_events(self) -> None:
        """Subscribe to relevant StageEventBus events."""
        if not self._stage_bus or StageEvent is None:
            return

        # Subscribe to events we want to bridge to cross-process
        for stage_event_value in STAGE_TO_CROSS_PROCESS_MAP:
            try:
                event_type = StageEvent(stage_event_value)
                self._stage_bus.subscribe(event_type, self._handle_stage_event)
            except ValueError:
                logger.debug(f"StageEvent not found: {stage_event_value}")

    async def _handle_data_event(self, event: DataEvent) -> None:
        """Handle a DataEventBus event and bridge to CrossProcess."""
        if not self._running or not self._cross_queue:
            return

        event_key = f"data:{event.event_type.value}:{event.timestamp}"
        if event_key in self._recently_bridged:
            return  # Dedup

        cross_event_type = DATA_TO_CROSS_PROCESS_MAP.get(event.event_type.value)
        if not cross_event_type:
            return

        try:
            self._cross_queue.publish(
                event_type=cross_event_type,
                payload=event.payload,
                source=f"data_events:{event.source}",
            )
            self._stats.events_bridged_data_to_cross += 1
            self._stats.last_bridge_time = datetime.now().isoformat()
            await self._dispatch_handlers(
                event_type=cross_event_type,
                payload=event.payload,
                source=f"data_events:{event.source}",
                origin="data_bus",
            )

            # Track to avoid re-forwarding
            self._recently_bridged.add(event_key)
            asyncio.get_event_loop().call_later(
                self._bridge_dedup_window,
                lambda k=event_key: self._recently_bridged.discard(k)
            )

            logger.debug(f"Bridged data event to cross-process: {event.event_type.value}")

        except Exception as e:
            self._stats.events_dropped += 1
            self._stats.errors.append(f"Data bridge error: {e}")
            logger.warning(f"Failed to bridge data event: {e}")

    async def _handle_stage_event(self, result: StageCompletionResult) -> None:
        """Handle a StageEventBus event and bridge to CrossProcess."""
        if not self._running or not self._cross_queue:
            return

        event_key = f"stage:{result.event.value}:{result.timestamp}"
        if event_key in self._recently_bridged:
            return  # Dedup

        cross_event_type = STAGE_TO_CROSS_PROCESS_MAP.get(result.event.value)
        if not cross_event_type:
            return

        try:
            self._cross_queue.publish(
                event_type=cross_event_type,
                payload=result.to_dict(),
                source=f"stage_events:{result.board_type}_{result.num_players}p",
            )
            self._stats.events_bridged_stage_to_cross += 1
            self._stats.last_bridge_time = datetime.now().isoformat()
            await self._dispatch_handlers(
                event_type=cross_event_type,
                payload=result.to_dict(),
                source=f"stage_events:{result.board_type}_{result.num_players}p",
                origin="stage_bus",
            )

            # Track to avoid re-forwarding
            self._recently_bridged.add(event_key)
            asyncio.get_event_loop().call_later(
                self._bridge_dedup_window,
                lambda k=event_key: self._recently_bridged.discard(k)
            )

            logger.debug(f"Bridged stage event to cross-process: {result.event.value}")

        except Exception as e:
            self._stats.events_dropped += 1
            self._stats.errors.append(f"Stage bridge error: {e}")
            logger.warning(f"Failed to bridge stage event: {e}")

    async def _poll_cross_process_loop(self) -> None:
        """Poll cross-process queue and forward to data events."""
        while self._running:
            try:
                await self._poll_and_bridge_cross_process()
                await asyncio.sleep(1.0)  # Poll every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cross-process poll error: {e}")
                await asyncio.sleep(5.0)

    async def _poll_and_bridge_cross_process(self) -> None:
        """Poll cross-process events and bridge to DataEventBus."""
        if not self._cross_queue or not self._subscriber_id:
            return

        events = self._cross_queue.poll(
            self._subscriber_id,
            limit=100,
        )

        for event in events:
            event_key = f"cross:{event.event_type}:{event.created_at}"
            if event_key in self._recently_bridged:
                self._cross_queue.ack(self._subscriber_id, event.event_id)
                continue  # Dedup

            # Bridge to DataEventBus
            data_event_value = CROSS_PROCESS_TO_DATA_MAP.get(event.event_type)
            if data_event_value and self._data_bus and DataEventType is not None and DataEvent is not None:
                try:
                    event_type = DataEventType(data_event_value)
                    data_event = DataEvent(
                        event_type=event_type,
                        payload=event.payload,
                        timestamp=event.created_at,
                        source=f"cross_process:{event.source}",
                    )
                    await self._data_bus.publish(data_event, bridge_cross_process=False)
                    self._stats.events_bridged_cross_to_data += 1
                    self._stats.last_bridge_time = datetime.now().isoformat()
                    await self._dispatch_handlers(
                        event_type=event.event_type,
                        payload=event.payload,
                        source=f"cross_process:{event.source}",
                        origin="cross_process",
                    )

                    logger.debug(f"Bridged cross-process to data: {event.event_type}")

                except Exception as e:
                    self._stats.events_dropped += 1
                    logger.debug(f"Failed to bridge cross-process event: {e}")

            # Track and ack
            self._recently_bridged.add(event_key)
            asyncio.get_event_loop().call_later(
                self._bridge_dedup_window,
                lambda k=event_key: self._recently_bridged.discard(k)
            )
            self._cross_queue.ack(self._subscriber_id, event.event_id)

    def get_stats(self) -> CoordinatorStats:
        """Get coordinator statistics."""
        return self._stats

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a custom handler for specific event types.

        Handlers are called for all bridged events of the specified type.

        Args:
            event_type: Event type string (e.g., "TRAINING_COMPLETED")
            handler: Callback function receiving event payload
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _dispatch_handlers(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str,
        origin: str,
    ) -> None:
        """Dispatch custom handlers for a bridged event."""
        handlers = self._event_handlers.get(event_type, [])
        if not handlers:
            return

        if isinstance(payload, dict):
            payload_with_context = dict(payload)
        else:
            payload_with_context = {"payload": payload}
        payload_with_context.setdefault("event_type", event_type)
        payload_with_context.setdefault("source", source)
        payload_with_context.setdefault("origin", origin)

        for handler in handlers:
            try:
                result = handler(payload_with_context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._stats.errors.append(f"Handler error for {event_type}: {e}")
                logger.debug(f"Handler error for {event_type}: {e}")

    async def emit_to_all(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "unified_coordinator",
    ) -> None:
        """Emit an event to all event systems.

        Args:
            event_type: Event type string
            payload: Event payload
            source: Event source identifier
        """
        # Emit to cross-process
        if self._cross_queue:
            self._cross_queue.publish(
                event_type=event_type,
                payload=payload,
                source=source,
            )

        # Emit to data events
        data_event_value = CROSS_PROCESS_TO_DATA_MAP.get(event_type)
        if data_event_value and self._data_bus and DataEventType is not None and DataEvent is not None:
            try:
                et = DataEventType(data_event_value)
                await self._data_bus.publish(DataEvent(
                    event_type=et,
                    payload=payload,
                    source=source,
                ), bridge_cross_process=False)
            except Exception:
                pass

        await self._dispatch_handlers(
            event_type=event_type,
            payload=payload,
            source=source,
            origin="emit",
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_event_coordinator() -> UnifiedEventCoordinator:
    """Get the singleton event coordinator."""
    return UnifiedEventCoordinator.get_instance()


async def start_coordinator() -> bool:
    """Start the event coordinator."""
    return await get_event_coordinator().start()


async def stop_coordinator() -> None:
    """Stop the event coordinator."""
    await get_event_coordinator().stop()


def get_coordinator_stats() -> CoordinatorStats:
    """Get coordinator statistics."""
    return get_event_coordinator().get_stats()


# =============================================================================
# Simple Event Emitters (December 2025)
# Use these functions to emit events from anywhere in the codebase.
# They automatically route to all event systems.
# =============================================================================

async def emit_training_started(
    config_key: str,
    node_name: str = "",
    **extra_payload
) -> None:
    """Emit TRAINING_STARTED event to all systems.

    Usage:
        await emit_training_started("square8_2p", node_name="lambda-1")
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Emit TRAINING_COMPLETED event to all systems.

    Usage:
        await emit_training_completed(
            "square8_2p", "model_v42", val_loss=0.123, epochs=50
        )
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Emit EVALUATION_COMPLETED event to all systems.

    Usage:
        await emit_evaluation_completed("model_v42", elo=1650, win_rate=0.58)
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Emit DATA_SYNC_COMPLETED event to all systems.

    Usage:
        await emit_sync_completed("games", files_synced=150)
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Emit MODEL_PROMOTED event to all systems.

    Usage:
        await emit_model_promoted("model_v42", tier="production", elo=1650)
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Emit SELFPLAY_BATCH_COMPLETE event to all systems.

    Usage:
        await emit_selfplay_batch_completed("square8_2p", games_generated=500)
    """
    coordinator = get_event_coordinator()
    await coordinator.emit_to_all(
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
    """Sync version of emit_training_started for non-async contexts.

    Creates an event loop if needed, or uses fire-and-forget.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(emit_training_started(config_key, node_name, **extra_payload))
        else:
            loop.run_until_complete(emit_training_started(config_key, node_name, **extra_payload))
    except RuntimeError:
        # No event loop, use new one
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


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Unified Event Coordinator")
    parser.add_argument("--start", action="store_true", help="Start coordinator")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--run-time", type=int, default=60, help="Run time in seconds")
    args = parser.parse_args()

    if args.stats:
        stats = get_coordinator_stats()
        print(json.dumps({
            "events_bridged_data_to_cross": stats.events_bridged_data_to_cross,
            "events_bridged_stage_to_cross": stats.events_bridged_stage_to_cross,
            "events_bridged_cross_to_data": stats.events_bridged_cross_to_data,
            "events_dropped": stats.events_dropped,
            "last_bridge_time": stats.last_bridge_time,
            "is_running": stats.is_running,
        }, indent=2))

    elif args.start:
        async def main():
            coordinator = get_event_coordinator()
            await coordinator.start()
            print(f"Coordinator running for {args.run_time}s...")
            await asyncio.sleep(args.run_time)
            await coordinator.stop()
            print("Coordinator stopped")

        asyncio.run(main())


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "CoordinatorStats",
    # Main class
    "UnifiedEventCoordinator",
    "emit_evaluation_completed",
    "emit_model_promoted",
    "emit_selfplay_batch_completed",
    "emit_sync_completed",
    "emit_training_completed",
    "emit_training_completed_sync",
    "emit_training_failed",
    # Async event emitters
    "emit_training_started",
    # Sync event emitters
    "emit_training_started_sync",
    "get_coordinator_stats",
    # Functions
    "get_event_coordinator",
    "start_coordinator",
    "stop_coordinator",
]
