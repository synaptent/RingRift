"""Unified safe event emission mixin.

Consolidates 6 duplicate `_safe_emit_event()` implementations across:
- availability/node_monitor.py
- availability/recovery_engine.py
- availability/capacity_planner.py
- availability/provisioner.py
- scripts/p2p/managers/state_manager.py
- scripts/p2p/p2p_mixin_base.py

Usage:
    from app.coordination.safe_event_emitter import SafeEventEmitterMixin

    class MyCoordinator(SafeEventEmitterMixin):
        _event_source = "MyCoordinator"

        def do_something(self):
            self._safe_emit_event("MY_EVENT", {"key": "value"})

    class MyAsyncDaemon(SafeEventEmitterMixin):
        _event_source = "MyAsyncDaemon"

        async def do_something_async(self):
            await self._safe_emit_event_async("MY_EVENT", {"key": "value"})
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class SafeEventEmitterMixin:
    """Mixin providing unified safe event emission for coordinators and daemons.

    Provides both sync and async emission methods that:
    - Never raise exceptions (log failures instead)
    - Return bool indicating success/failure
    - Use lazy imports to avoid circular dependencies
    - Track event source for debugging

    Example - Sync coordinator:
        >>> class MyCoordinator(SafeEventEmitterMixin):
        ...     _event_source = "MyCoordinator"
        ...
        ...     def process_data(self, config_key: str):
        ...         # Do processing...
        ...         success = self._safe_emit_event(
        ...             "DATA_PROCESSED",
        ...             {"config_key": config_key, "status": "complete"},
        ...         )
        ...         if not success:
        ...             logger.warning("Event emission failed, continuing anyway")

    Example - Async daemon:
        >>> class MyAsyncDaemon(SafeEventEmitterMixin):
        ...     _event_source = "MyAsyncDaemon"
        ...
        ...     async def run_cycle(self):
        ...         result = await self.do_work()
        ...         await self._safe_emit_event_async(
        ...             "CYCLE_COMPLETED",
        ...             {"result": result, "timestamp": time.time()},
        ...         )

    Example - Check emission success:
        >>> if self._safe_emit_event("CRITICAL_EVENT", payload):
        ...     logger.info("Event emitted successfully")
        ... else:
        ...     logger.warning("Event bus unavailable, using fallback")
        ...     self._queue_for_retry(payload)

    Attributes:
        _event_source: Class-level identifier for the event source.
                      Override in subclasses to set custom source name.
    """

    _event_source: ClassVar[str] = "unknown"

    def _safe_emit_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Safely emit an event via the event router.

        Wraps event emission in try-catch to prevent event failures
        from crashing the caller.

        Args:
            event_type: Event type string to emit (e.g., "TRAINING_COMPLETED")
            payload: Optional event payload dict

        Returns:
            True if event was scheduled successfully, False otherwise

        Example:
            self._safe_emit_event(
                "HOST_OFFLINE",
                {"node_id": peer_id, "reason": "timeout"},
            )
        """
        try:
            # Lazy imports to avoid circular dependencies
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEvent

            bus = get_event_bus()
            if bus is None:
                logger.debug(f"[{self._event_source}] Event bus unavailable")
                return False

            event = DataEvent(
                event_type=event_type,
                payload=payload or {},
                source=self._event_source,
            )

            # bus.publish() is async - schedule it properly
            # Use fire-and-forget if there's a running loop
            try:
                loop = asyncio.get_running_loop()
                # Schedule as fire-and-forget task
                task = loop.create_task(bus.publish(event))
                # Add error callback to log failures without crashing
                task.add_done_callback(
                    lambda t: (
                        logger.debug(
                            f"[{self._event_source}] Event publish failed: {t.exception()}"
                        )
                        if t.exception()
                        else None
                    )
                )
                return True
            except RuntimeError:
                # No running event loop - can't schedule async publish
                # This is acceptable in pure sync contexts
                logger.debug(
                    f"[{self._event_source}] No event loop, event not published"
                )
                return False

        except (AttributeError, ImportError, TypeError) as e:
            # AttributeError - event bus missing attribute
            # ImportError - module unavailable during shutdown
            # TypeError - wrong DataEvent signature
            logger.debug(f"[{self._event_source}] Event emission failed: {e}")
            return False

    async def _safe_emit_event_async(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Async version of safe event emission.

        For use in async contexts where blocking on the event bus
        could cause issues.

        Args:
            event_type: Event type string to emit
            payload: Optional event payload dict

        Returns:
            True if event was emitted successfully, False otherwise
        """
        try:
            return await asyncio.to_thread(
                self._safe_emit_event,
                event_type,
                payload,
            )
        except RuntimeError:
            # No event loop available - fall back to sync
            return self._safe_emit_event(event_type, payload)


# Module-level helper for non-class contexts
def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "module",
) -> bool:
    """Module-level safe event emission.

    For use in module-level functions or contexts without a class.

    Args:
        event_type: Event type string to emit
        payload: Optional event payload dict
        source: Source identifier for the event

    Returns:
        True if event was scheduled successfully, False otherwise

    Example:
        from app.coordination.safe_event_emitter import safe_emit_event

        def my_function():
            safe_emit_event("MY_EVENT", {"key": "value"}, source="my_module")
    """
    try:
        from app.coordination.event_router import get_event_bus
        from app.distributed.data_events import DataEvent

        bus = get_event_bus()
        if bus is None:
            logger.debug(f"[{source}] Event bus unavailable")
            return False

        event = DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        )

        # bus.publish() is async - schedule it properly
        try:
            loop = asyncio.get_running_loop()
            # Schedule as fire-and-forget task
            task = loop.create_task(bus.publish(event))
            # Add error callback to log failures without crashing
            task.add_done_callback(
                lambda t: (
                    logger.debug(f"[{source}] Event publish failed: {t.exception()}")
                    if t.exception()
                    else None
                )
            )
            return True
        except RuntimeError:
            # No running event loop - can't schedule async publish
            logger.debug(f"[{source}] No event loop, event not published")
            return False

    except (AttributeError, ImportError, TypeError) as e:
        logger.debug(f"[{source}] Event emission failed: {e}")
        return False


__all__ = [
    "SafeEventEmitterMixin",
    "safe_emit_event",
]
