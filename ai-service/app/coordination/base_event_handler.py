"""Base Event Handler - Unified base class for event-driven coordinators.

Provides standardized patterns for:
- Event subscription lifecycle management
- Error handling with logging
- Health check protocol compliance
- Graceful shutdown with event cleanup

This reduces ~800 LOC of duplicated boilerplate across 12+ handler files.

Usage:
    from app.coordination.base_event_handler import BaseEventHandler

    class MyHandler(BaseEventHandler):
        def __init__(self):
            super().__init__("MyHandler")
            # Handler-specific initialization

        def _get_subscriptions(self) -> dict[DataEventType, Callable]:
            return {
                DataEventType.TRAINING_COMPLETED: self._on_training_completed,
                DataEventType.EVALUATION_COMPLETED: self._on_evaluation_completed,
            }

        async def _on_training_completed(self, event: dict) -> None:
            # Handle event
            pass

Created: December 2025
Purpose: Wave 4 Phase 1 - Reduce event handler duplication
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from app.coordination.contracts import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)

__all__ = [
    "BaseEventHandler",
    "EventHandlerConfig",
]


class EventHandlerConfig:
    """Configuration for event handlers.

    Override in subclasses for custom behavior.
    """

    # Whether to register with coordinator registry
    register_with_registry: bool = True

    # Whether handler methods are async (True) or sync (False)
    async_handlers: bool = True

    # Whether to use fire_and_forget for sync event callbacks
    use_fire_and_forget: bool = True

    # Timeout for handler execution (0 = no timeout)
    handler_timeout_seconds: float = 0.0


class BaseEventHandler(ABC):
    """Base class for event-driven coordinators.

    Provides:
    - Automatic event subscription on start()
    - Automatic unsubscription on stop()
    - Error counting and last error tracking
    - Standard health_check() implementation
    - Coordinator registry integration

    Subclasses must implement:
    - _get_subscriptions(): Return dict of event_type -> handler method
    """

    def __init__(
        self,
        name: str,
        config: Optional[EventHandlerConfig] = None,
    ):
        """Initialize base event handler.

        Args:
            name: Handler name for logging and registration
            config: Optional configuration overrides
        """
        self._name = name
        self._config = config or EventHandlerConfig()
        self._logger = logging.getLogger(f"coordination.{name}")

        # State
        self._running = False
        self._subscribed = False
        self._status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0

        # Metrics
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        # Subscription tracking
        self._subscriptions: Dict[Any, Callable] = {}
        self._wrapped_handlers: Dict[Any, Callable] = {}

        # Concurrency
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Handler name for identification."""
        return self._name

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the handler."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the handler is actively processing events."""
        return self._running and self._subscribed

    @property
    def uptime_seconds(self) -> float:
        """Time since handler started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _get_subscriptions(self) -> Dict[Any, Callable]:
        """Return dict of event_type -> handler method.

        Returns:
            Dictionary mapping DataEventType to handler methods.
            Handler methods should be async if config.async_handlers is True.

        Example:
            def _get_subscriptions(self):
                return {
                    DataEventType.TRAINING_COMPLETED: self._on_training_completed,
                    DataEventType.EVALUATION_COMPLETED: self._on_evaluation_completed,
                }
        """
        pass

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> bool:
        """Start the handler and subscribe to events.

        Returns:
            True if successfully started
        """
        if self._running:
            return True

        self._logger.info(f"[{self._name}] Starting...")

        success = await self._subscribe_to_events()
        if success:
            self._running = True
            self._status = CoordinatorStatus.RUNNING
            self._start_time = time.time()

            if self._config.register_with_registry:
                register_coordinator(self)

            self._logger.info(f"[{self._name}] Started - subscribed to {len(self._subscriptions)} event(s)")
        else:
            self._status = CoordinatorStatus.ERROR
            self._last_error = "Failed to subscribe to events"
            self._logger.error(f"[{self._name}] Failed to start")

        return success

    async def stop(self) -> None:
        """Stop the handler and cleanup subscriptions."""
        if not self._running:
            return

        self._logger.info(f"[{self._name}] Stopping...")
        self._status = CoordinatorStatus.STOPPING

        await self._unsubscribe_from_events()

        if self._config.register_with_registry:
            unregister_coordinator(self._name)

        self._running = False
        self._status = CoordinatorStatus.STOPPED

        self._logger.info(f"[{self._name}] Stopped (processed {self._events_processed} events, {self._errors_count} errors)")

    # =========================================================================
    # Event Subscription
    # =========================================================================

    async def _subscribe_to_events(self) -> bool:
        """Subscribe to all events returned by _get_subscriptions().

        Returns:
            True if all subscriptions successful
        """
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus is None:
                self._logger.warning(f"[{self._name}] Event bus not available")
                return False

            self._subscriptions = self._get_subscriptions()
            if not self._subscriptions:
                self._logger.warning(f"[{self._name}] No subscriptions defined")
                return True  # Not an error, just no events to subscribe to

            for event_type, handler in self._subscriptions.items():
                wrapped = self._wrap_handler(handler, event_type)
                self._wrapped_handlers[event_type] = wrapped
                bus.subscribe(event_type, wrapped)
                self._logger.debug(f"[{self._name}] Subscribed to {event_type}")

            self._subscribed = True
            return True

        except ImportError as e:
            self._logger.warning(f"[{self._name}] Event router not available: {e}")
            return False
        except (RuntimeError, ValueError, AttributeError) as e:
            self._logger.warning(f"[{self._name}] Failed to subscribe: {e}")
            return False

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                for event_type, wrapped_handler in self._wrapped_handlers.items():
                    try:
                        bus.unsubscribe(event_type, wrapped_handler)
                        self._logger.debug(f"[{self._name}] Unsubscribed from {event_type}")
                    except (ValueError, KeyError, AttributeError):
                        pass  # Handler not found, ignore

            self._subscribed = False
            self._wrapped_handlers.clear()

        except (ImportError, RuntimeError, ValueError, AttributeError):
            pass
        finally:
            # Always mark as unsubscribed even if cleanup fails
            self._subscribed = False

    def _wrap_handler(self, handler: Callable, event_type: Any) -> Callable:
        """Wrap handler with error handling and metrics.

        Args:
            handler: The original handler method
            event_type: The event type (for logging)

        Returns:
            Wrapped handler with error handling
        """
        if self._config.async_handlers:
            # Handler is async, wrap for sync event bus callback
            def sync_wrapper(event: Any) -> None:
                try:
                    from app.core.async_context import fire_and_forget

                    fire_and_forget(self._handle_event_async(handler, event, event_type))
                except (ImportError, RuntimeError) as e:
                    self._logger.warning(f"[{self._name}] Failed to schedule handler: {e}")

            return sync_wrapper
        else:
            # Handler is sync, wrap directly
            def sync_handler(event: Any) -> None:
                self._handle_event_sync(handler, event, event_type)

            return sync_handler

    async def _handle_event_async(
        self,
        handler: Callable,
        event: Any,
        event_type: Any,
    ) -> None:
        """Handle event asynchronously with error tracking.

        Args:
            handler: The handler method to call
            event: The event payload
            event_type: The event type (for logging)
        """
        async with self._lock:
            try:
                self._events_processed += 1
                await handler(event)
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                self._logger.error(f"[{self._name}] Error handling {event_type}: {e}")

    def _handle_event_sync(
        self,
        handler: Callable,
        event: Any,
        event_type: Any,
    ) -> None:
        """Handle event synchronously with error tracking.

        Args:
            handler: The handler method to call
            event: The event payload
            event_type: The event type (for logging)
        """
        try:
            self._events_processed += 1
            handler(event)
        except Exception as e:
            self._errors_count += 1
            self._last_error = str(e)
            self._logger.error(f"[{self._name}] Error handling {event_type}: {e}")

    # =========================================================================
    # Health Check and Metrics
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Check handler health.

        Returns:
            HealthCheckResult (CoordinatorProtocol compliant)
        """
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Handler in error state: {self._last_error}",
                details={
                    "subscribed": self._subscribed,
                    "errors_count": self._errors_count,
                },
            )

        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Handler is stopped",
            )

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message="Handler running but not subscribed to events",
                details={
                    "events_processed": self._events_processed,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=self._status,
            message=f"{self._name}: {self._events_processed} events, {self._errors_count} errors",
            details={
                "subscribed": self._subscribed,
                "uptime_seconds": self.uptime_seconds,
                "events_processed": self._events_processed,
                "errors_count": self._errors_count,
                "subscription_count": len(self._subscriptions),
            },
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get handler metrics.

        Returns:
            Dictionary of metrics (CoordinatorProtocol compliant)
        """
        return {
            "name": self._name,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            "subscribed": self._subscribed,
            "subscription_count": len(self._subscriptions),
        }

    def get_status(self) -> dict[str, Any]:
        """Get handler status for DaemonManager.

        Returns:
            Status dict with daemon state
        """
        return {
            "daemon": self._name,
            "running": self._running,
            "subscribed": self._subscribed,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "events_processed": self._events_processed,
            "errors_count": self._errors_count,
        }
