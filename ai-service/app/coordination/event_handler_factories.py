"""Event Handler Factories - Reusable event handler generators.

December 29, 2025: Created to reduce code duplication across 40+ files
that use nearly identical lambda handlers for common patterns.

Common Patterns Consolidated:
1. Logging handlers - Log event receipt with extracted key
2. Forwarding handlers - Delegate to orchestrator methods
3. Remapping handlers - Transform and re-emit as different event type
4. Filtering handlers - Only process events matching criteria
5. Aggregating handlers - Collect events before processing

Usage:
    from app.coordination.event_handler_factories import (
        create_logging_handler,
        create_forwarding_handler,
        create_remapping_handler,
    )

    # Instead of:
    router.subscribe("MY_EVENT", lambda evt: logger.info(f"MY_EVENT: {evt.get('node_id')}"))

    # Use:
    router.subscribe("MY_EVENT", create_logging_handler("MY_EVENT", extract_key="node_id"))

    # Instead of:
    router.subscribe("TRAINING_COMPLETED", lambda evt: orchestrator.on_training_complete(evt))

    # Use:
    router.subscribe("TRAINING_COMPLETED", create_forwarding_handler(orchestrator.on_training_complete))
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TypeVar, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    # Handler factories
    "create_logging_handler",
    "create_forwarding_handler",
    "create_remapping_handler",
    "create_filtering_handler",
    "create_debouncing_handler",
    "create_aggregating_handler",
    # Configuration
    "HandlerConfig",
]


# Type aliases
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
SyncHandler = Callable[[dict[str, Any]], None]
FilterFunc = Callable[[dict[str, Any]], bool]


@dataclass
class HandlerConfig:
    """Configuration for handler factories.

    Attributes:
        log_level: Logging level for logging handlers (default: INFO)
        include_timestamp: Include timestamp in log messages
        extract_keys: Keys to extract from event for logging
        error_handler: Optional error handler callback
    """

    log_level: int = logging.INFO
    include_timestamp: bool = False
    extract_keys: list[str] = field(default_factory=lambda: ["node_id", "config_key"])
    error_handler: Optional[Callable[[Exception, dict], None]] = None


def create_logging_handler(
    event_name: str,
    extract_key: str = "node_id",
    log_level: int = logging.INFO,
    logger_name: Optional[str] = None,
) -> EventHandler:
    """Create a handler that logs event receipt.

    Args:
        event_name: Name to include in log message
        extract_key: Key to extract from event payload
        log_level: Logging level (default: INFO)
        logger_name: Optional logger name (default: module logger)

    Returns:
        Async event handler that logs and returns

    Example:
        handler = create_logging_handler("TRAINING_COMPLETED", "model_path")
        # Logs: "TRAINING_COMPLETED: models/canonical_hex8_2p.pth"
    """
    log = logging.getLogger(logger_name) if logger_name else logger

    async def handler(event: dict[str, Any]) -> None:
        value = event.get(extract_key, "unknown")
        log.log(log_level, f"{event_name}: {value}")

    return handler


def create_forwarding_handler(
    target_method: Callable[[dict[str, Any]], Any],
    error_handler: Optional[Callable[[Exception, dict], None]] = None,
) -> EventHandler:
    """Create a handler that forwards events to a method.

    Handles both sync and async target methods.

    Args:
        target_method: Method to call with event payload
        error_handler: Optional callback for errors

    Returns:
        Async event handler that forwards to target

    Example:
        handler = create_forwarding_handler(orchestrator.on_training_complete)
        # Calls orchestrator.on_training_complete(event)
    """
    async def handler(event: dict[str, Any]) -> None:
        try:
            result = target_method(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            if error_handler:
                error_handler(e, event)
            else:
                logger.error(f"Forwarding handler error: {e}", exc_info=True)

    return handler


def create_remapping_handler(
    target_event_type: str,
    transform: Optional[Callable[[dict], dict]] = None,
    source_key_mapping: Optional[dict[str, str]] = None,
) -> EventHandler:
    """Create a handler that re-emits event as a different type.

    Args:
        target_event_type: Event type to emit
        transform: Optional function to transform payload
        source_key_mapping: Optional key renaming map {old_key: new_key}

    Returns:
        Async event handler that re-emits transformed event

    Example:
        # Re-emit TRAINING_COMPLETED as EVALUATION_TRIGGERED
        handler = create_remapping_handler("EVALUATION_TRIGGERED")

        # With key renaming
        handler = create_remapping_handler(
            "MODEL_SYNC_REQUESTED",
            source_key_mapping={"model_path": "path", "node_id": "target_node"}
        )
    """
    async def handler(event: dict[str, Any]) -> None:
        try:
            from app.coordination.event_router import get_router

            # Transform payload
            if transform:
                payload = transform(event)
            elif source_key_mapping:
                payload = {
                    source_key_mapping.get(k, k): v
                    for k, v in event.items()
                }
            else:
                payload = event.copy()

            # Emit remapped event
            router = get_router()
            await router.publish_async(target_event_type, payload)

        except ImportError:
            logger.debug(f"Cannot remap to {target_event_type}: event_router not available")
        except Exception as e:
            logger.error(f"Remapping handler error: {e}", exc_info=True)

    return handler


def create_filtering_handler(
    inner_handler: EventHandler,
    filter_func: FilterFunc,
) -> EventHandler:
    """Create a handler that only processes events matching a filter.

    Args:
        inner_handler: Handler to call for matching events
        filter_func: Function returning True for events to process

    Returns:
        Async event handler that filters before calling inner handler

    Example:
        # Only process events for specific config
        handler = create_filtering_handler(
            my_handler,
            lambda evt: evt.get("config_key") == "hex8_2p"
        )
    """
    async def handler(event: dict[str, Any]) -> None:
        if filter_func(event):
            await inner_handler(event)

    return handler


def create_debouncing_handler(
    inner_handler: EventHandler,
    debounce_seconds: float = 1.0,
    key_func: Optional[Callable[[dict], str]] = None,
) -> EventHandler:
    """Create a handler that debounces rapid event bursts.

    Only the last event in a burst (within debounce window) triggers the handler.

    Args:
        inner_handler: Handler to call after debounce
        debounce_seconds: Time to wait for more events
        key_func: Optional function to extract debounce key (for per-key debouncing)

    Returns:
        Async event handler with debouncing

    Example:
        # Debounce quality updates to avoid excessive processing
        handler = create_debouncing_handler(
            process_quality_update,
            debounce_seconds=5.0,
            key_func=lambda evt: evt.get("config_key")
        )
    """
    pending: dict[str, tuple[float, dict]] = {}
    lock = asyncio.Lock()

    async def handler(event: dict[str, Any]) -> None:
        key = key_func(event) if key_func else "_default"
        now = time.time()

        async with lock:
            pending[key] = (now, event)

        # Wait for debounce period
        await asyncio.sleep(debounce_seconds)

        async with lock:
            # Check if this is still the latest event
            if key in pending:
                stored_time, stored_event = pending[key]
                if stored_time == now:
                    del pending[key]
                    await inner_handler(stored_event)

    return handler


@dataclass
class EventAggregator:
    """Aggregates events for batch processing.

    Collects events until either:
    - max_count events are collected
    - max_wait_seconds have elapsed since first event
    """

    max_count: int = 10
    max_wait_seconds: float = 5.0
    _events: list[dict] = field(default_factory=list)
    _first_event_time: Optional[float] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self) -> list[dict]:
        """Reset and return collected events."""
        events = self._events.copy()
        self._events = []
        self._first_event_time = None
        return events


def create_aggregating_handler(
    batch_handler: Callable[[list[dict]], Awaitable[None]],
    max_count: int = 10,
    max_wait_seconds: float = 5.0,
) -> EventHandler:
    """Create a handler that aggregates events for batch processing.

    Args:
        batch_handler: Handler to call with list of aggregated events
        max_count: Max events to aggregate before triggering
        max_wait_seconds: Max time to wait before triggering

    Returns:
        Async event handler that aggregates events

    Example:
        # Batch process game completions
        async def process_games(events: list[dict]) -> None:
            game_ids = [e.get("game_id") for e in events]
            await bulk_update_database(game_ids)

        handler = create_aggregating_handler(
            process_games,
            max_count=50,
            max_wait_seconds=10.0
        )
    """
    events: list[dict] = []
    first_event_time: Optional[float] = None
    lock = asyncio.Lock()
    pending_flush: Optional[asyncio.Task] = None

    async def _do_flush_unlocked() -> None:
        """Flush without lock - caller must hold lock."""
        nonlocal events, first_event_time, pending_flush
        if events:
            batch = events.copy()
            events = []
            first_event_time = None
            pending_flush = None
            try:
                await batch_handler(batch)
            except Exception as e:
                logger.error(f"Aggregating handler batch error: {e}", exc_info=True)

    async def delayed_flush() -> None:
        """Called from background task - needs to acquire lock."""
        await asyncio.sleep(max_wait_seconds)
        async with lock:
            await _do_flush_unlocked()

    async def handler(event: dict[str, Any]) -> None:
        nonlocal events, first_event_time, pending_flush

        async with lock:
            events.append(event)

            if first_event_time is None:
                first_event_time = time.time()
                # Schedule flush after max_wait
                pending_flush = asyncio.create_task(delayed_flush())

            # Check if we should flush now
            if len(events) >= max_count:
                if pending_flush:
                    pending_flush.cancel()
                # Already holding lock, call unlocked version
                await _do_flush_unlocked()

    return handler


# Convenience factory for common patterns
def create_config_specific_handler(
    config_key: str,
    inner_handler: EventHandler,
) -> EventHandler:
    """Create a handler that only processes events for a specific config.

    Args:
        config_key: Config key to filter for (e.g., "hex8_2p")
        inner_handler: Handler to call for matching events

    Returns:
        Filtered event handler
    """
    return create_filtering_handler(
        inner_handler,
        lambda evt: evt.get("config_key") == config_key
    )


def create_node_specific_handler(
    node_id: str,
    inner_handler: EventHandler,
) -> EventHandler:
    """Create a handler that only processes events for a specific node.

    Args:
        node_id: Node ID to filter for
        inner_handler: Handler to call for matching events

    Returns:
        Filtered event handler
    """
    return create_filtering_handler(
        inner_handler,
        lambda evt: evt.get("node_id") == node_id or evt.get("source_node") == node_id
    )
