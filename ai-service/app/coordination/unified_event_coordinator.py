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
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Event Type Mappings
# =============================================================================

# Map DataEventType values to CrossProcess event types
DATA_TO_CROSS_PROCESS_MAP = {
    # Training events
    "training_started": "TRAINING_STARTED",
    "training_completed": "TRAINING_COMPLETED",
    "training_failed": "TRAINING_FAILED",
    "training_threshold": "TRAINING_THRESHOLD_REACHED",
    # Evaluation events
    "evaluation_completed": "EVALUATION_COMPLETED",
    "elo_updated": "ELO_UPDATED",
    # Promotion events
    "model_promoted": "MODEL_PROMOTED",
    "promotion_failed": "PROMOTION_FAILED",
    # Data events
    "new_games": "NEW_GAMES_AVAILABLE",
    "sync_completed": "DATA_SYNC_COMPLETED",
    # Quality events
    "quality_score_updated": "QUALITY_SCORE_UPDATED",
    "quality_distribution_changed": "QUALITY_DISTRIBUTION_CHANGED",
    "high_quality_data_available": "HIGH_QUALITY_DATA_AVAILABLE",
    "low_quality_data_warning": "LOW_QUALITY_DATA_WARNING",
    # Regression events
    "regression_detected": "REGRESSION_DETECTED",
    "regression_critical": "REGRESSION_CRITICAL",
    "regression_cleared": "REGRESSION_CLEARED",
}

# Map StageEvent values to CrossProcess event types (December 2025 - complete mapping)
STAGE_TO_CROSS_PROCESS_MAP = {
    # Training events
    "training_complete": "TRAINING_COMPLETED",
    "training_started": "TRAINING_STARTED",
    "training_failed": "TRAINING_FAILED",
    # Evaluation events
    "evaluation_complete": "EVALUATION_COMPLETED",
    "shadow_tournament_complete": "SHADOW_TOURNAMENT_COMPLETE",
    "elo_calibration_complete": "ELO_CALIBRATION_COMPLETE",
    # Promotion events
    "promotion_complete": "MODEL_PROMOTED",
    "tier_gating_complete": "TIER_GATING_COMPLETE",
    # Selfplay events
    "selfplay_complete": "SELFPLAY_BATCH_COMPLETE",
    "canonical_selfplay_complete": "CANONICAL_SELFPLAY_COMPLETE",
    "gpu_selfplay_complete": "GPU_SELFPLAY_COMPLETE",
    # Data/sync events
    "sync_complete": "DATA_SYNC_COMPLETED",
    "parity_validation_complete": "PARITY_VALIDATION_COMPLETE",
    "npz_export_complete": "NPZ_EXPORT_COMPLETE",
    "cluster_sync_complete": "CLUSTER_SYNC_COMPLETE",
    "model_sync_complete": "MODEL_SYNC_COMPLETE",
    # Optimization events
    "cmaes_complete": "CMAES_COMPLETE",
    "pbt_complete": "PBT_COMPLETE",
    "nas_complete": "NAS_COMPLETE",
    # Utility events
    "iteration_complete": "ITERATION_COMPLETE",
}

# Events to bridge from CrossProcess back to DataEventBus (December 2025 - expanded)
CROSS_PROCESS_TO_DATA_MAP = {
    # Training events
    "TRAINING_STARTED": "training_started",
    "TRAINING_COMPLETED": "training_completed",
    "TRAINING_FAILED": "training_failed",
    "TRAINING_THRESHOLD_REACHED": "training_threshold",
    # Evaluation events
    "EVALUATION_COMPLETED": "evaluation_completed",
    "ELO_UPDATED": "elo_updated",
    # Promotion events
    "MODEL_PROMOTED": "model_promoted",
    "PROMOTION_FAILED": "promotion_failed",
    # Data events
    "NEW_GAMES_AVAILABLE": "new_games",
    "DATA_SYNC_COMPLETED": "sync_completed",
    # Quality events
    "QUALITY_SCORE_UPDATED": "quality_score_updated",
    "QUALITY_DISTRIBUTION_CHANGED": "quality_distribution_changed",
    "HIGH_QUALITY_DATA_AVAILABLE": "high_quality_data_available",
    # Regression events
    "REGRESSION_DETECTED": "regression_detected",
    "REGRESSION_CRITICAL": "regression_critical",
    "REGRESSION_CLEARED": "regression_cleared",
}


# =============================================================================
# Import Event Systems (with graceful fallback)
# =============================================================================

# DataEventBus
try:
    from app.distributed.data_events import (
        DataEventType,
        DataEvent,
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
        StageEvent,
        StageCompletionResult,
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
    last_bridge_time: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    start_time: Optional[str] = None
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

    _instance: Optional['UnifiedEventCoordinator'] = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize the coordinator."""
        self._data_bus: Optional[DataEventBus] = None
        self._stage_bus: Optional[StageEventBus] = None
        self._cross_queue: Optional[CrossProcessEventQueue] = None

        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._subscriber_id: Optional[str] = None

        self._stats = CoordinatorStats()
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Events we're bridging (avoid re-forwarding)
        self._recently_bridged: Set[str] = set()
        self._bridge_dedup_window = 5.0  # seconds

    @classmethod
    def get_instance(cls) -> 'UnifiedEventCoordinator':
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
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._cross_queue and self._subscriber_id:
            self._cross_queue.unsubscribe(self._subscriber_id)
            self._subscriber_id = None

        logger.info("UnifiedEventCoordinator stopped")

    def _subscribe_to_data_events(self) -> None:
        """Subscribe to relevant DataEventBus events."""
        if not self._data_bus or not DataEventType:
            return

        # Subscribe to events we want to bridge to cross-process
        for data_event_value in DATA_TO_CROSS_PROCESS_MAP.keys():
            try:
                event_type = DataEventType(data_event_value)
                self._data_bus.subscribe(event_type, self._handle_data_event)
            except ValueError:
                logger.debug(f"DataEventType not found: {data_event_value}")

    def _subscribe_to_stage_events(self) -> None:
        """Subscribe to relevant StageEventBus events."""
        if not self._stage_bus or not StageEvent:
            return

        # Subscribe to events we want to bridge to cross-process
        for stage_event_value in STAGE_TO_CROSS_PROCESS_MAP.keys():
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
            if data_event_value and self._data_bus and DataEventType and DataEvent:
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
        handler: Callable[[Dict[str, Any]], None]
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

    async def emit_to_all(
        self,
        event_type: str,
        payload: Dict[str, Any],
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
        if data_event_value and self._data_bus and DataEventType and DataEvent:
            try:
                et = DataEventType(data_event_value)
                await self._data_bus.publish(DataEvent(
                    event_type=et,
                    payload=payload,
                    source=source,
                ), bridge_cross_process=False)
            except Exception:
                pass


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
