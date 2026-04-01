"""Lifecycle management mixin for daemons and handlers.

December 29, 2025: Created to consolidate async lifecycle patterns across:
- 30+ daemons with start/stop/shutdown methods
- Handler base classes with lifecycle hooks
- Monitor classes with background loops

This mixin provides:
- Standard lifecycle states (CREATED, STARTING, RUNNING, STOPPING, STOPPED)
- Async context manager support
- Startup/shutdown hooks
- Graceful shutdown with timeout
- Health check integration

Usage:
    from app.coordination.mixins.lifecycle_mixin import LifecycleMixin

    class MyDaemon(LifecycleMixin):
        async def _on_start(self) -> None:
            # Setup logic
            pass

        async def _on_stop(self) -> None:
            # Cleanup logic
            pass

        async def _run_loop(self) -> None:
            while self.is_running:
                await self._do_work()
                await asyncio.sleep(self.cycle_interval)

    # Usage
    daemon = MyDaemon()
    async with daemon:
        # Daemon is running
        await asyncio.sleep(60)
    # Daemon is stopped
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Lifecycle States
# =============================================================================


class LifecycleState(Enum):
    """Standard lifecycle states for managed components."""

    CREATED = "created"       # Instance created but not started
    STARTING = "starting"     # Start in progress
    RUNNING = "running"       # Actively running
    PAUSED = "paused"         # Temporarily paused (optional)
    STOPPING = "stopping"     # Stop in progress
    STOPPED = "stopped"       # Cleanly stopped
    FAILED = "failed"         # Failed with error


# =============================================================================
# Lifecycle Mixin
# =============================================================================


class LifecycleMixin(ABC):
    """Mixin providing standard lifecycle management for async components.

    Subclasses should implement:
    - _on_start(): Called during startup
    - _on_stop(): Called during shutdown
    - _run_loop(): Optional main execution loop

    The mixin handles:
    - State transitions (CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED)
    - Graceful shutdown with timeout
    - Error handling during lifecycle transitions
    - Health check integration
    - Async context manager support
    """

    def __init__(
        self,
        name: str = "component",
        cycle_interval: float = 60.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        """Initialize lifecycle mixin.

        Args:
            name: Component name for logging
            cycle_interval: Interval between loop iterations
            shutdown_timeout: Maximum time to wait for graceful shutdown
        """
        self._lifecycle_name = name
        self._lifecycle_state = LifecycleState.CREATED
        self._cycle_interval = cycle_interval
        self._shutdown_timeout = shutdown_timeout

        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._last_error: Exception | None = None

        # Metrics
        self._cycle_count = 0
        self._error_count = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def lifecycle_state(self) -> LifecycleState:
        """Current lifecycle state."""
        return self._lifecycle_state

    @property
    def is_running(self) -> bool:
        """Whether component is in running state."""
        return self._lifecycle_state == LifecycleState.RUNNING

    @property
    def is_stopped(self) -> bool:
        """Whether component is stopped."""
        return self._lifecycle_state in (LifecycleState.STOPPED, LifecycleState.FAILED)

    @property
    def uptime(self) -> float | None:
        """Uptime in seconds (None if not started)."""
        if self._start_time is None:
            return None
        end_time = self._stop_time or time.time()
        return end_time - self._start_time

    @property
    def cycle_interval(self) -> float:
        """Interval between loop iterations."""
        return self._cycle_interval

    @cycle_interval.setter
    def cycle_interval(self, value: float) -> None:
        """Set interval between loop iterations."""
        self._cycle_interval = max(0.1, value)  # Minimum 100ms

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def start(self) -> bool:
        """Start the component.

        Returns:
            True if started successfully, False otherwise
        """
        if self._lifecycle_state == LifecycleState.RUNNING:
            logger.debug(f"[{self._lifecycle_name}] Already running")
            return True

        if self._lifecycle_state not in (LifecycleState.CREATED, LifecycleState.STOPPED):
            logger.warning(
                f"[{self._lifecycle_name}] Cannot start from state {self._lifecycle_state.value}"
            )
            return False

        self._lifecycle_state = LifecycleState.STARTING
        self._shutdown_event = asyncio.Event()
        self._start_time = time.time()
        self._stop_time = None
        self._last_error = None

        try:
            # Call startup hook
            await self._on_start()

            # Start main loop if implemented
            if hasattr(self, "_run_loop"):
                self._loop_task = asyncio.create_task(
                    self._lifecycle_loop_wrapper(),
                    name=f"{self._lifecycle_name}-loop",
                )

            self._lifecycle_state = LifecycleState.RUNNING
            logger.info(f"[{self._lifecycle_name}] Started")
            return True

        except Exception as e:
            self._lifecycle_state = LifecycleState.FAILED
            self._last_error = e
            self._error_count += 1
            logger.error(f"[{self._lifecycle_name}] Start failed: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the component gracefully.

        Returns:
            True if stopped successfully, False otherwise
        """
        if self._lifecycle_state in (LifecycleState.STOPPED, LifecycleState.CREATED):
            return True

        if self._lifecycle_state == LifecycleState.STOPPING:
            logger.debug(f"[{self._lifecycle_name}] Already stopping")
            return True

        self._lifecycle_state = LifecycleState.STOPPING

        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        try:
            # Wait for loop to finish
            if self._loop_task and not self._loop_task.done():
                try:
                    await asyncio.wait_for(
                        self._loop_task,
                        timeout=self._shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self._lifecycle_name}] Loop didn't stop within timeout, cancelling"
                    )
                    self._loop_task.cancel()
                    try:
                        await self._loop_task
                    except asyncio.CancelledError:
                        pass

            # Call shutdown hook
            await self._on_stop()

            self._lifecycle_state = LifecycleState.STOPPED
            self._stop_time = time.time()
            logger.info(f"[{self._lifecycle_name}] Stopped (uptime: {self.uptime:.1f}s)")
            return True

        except Exception as e:
            self._lifecycle_state = LifecycleState.FAILED
            self._last_error = e
            self._error_count += 1
            logger.error(f"[{self._lifecycle_name}] Stop failed: {e}")
            return False

    async def restart(self) -> bool:
        """Restart the component.

        Returns:
            True if restarted successfully
        """
        await self.stop()
        return await self.start()

    async def pause(self) -> bool:
        """Pause the component (if supported).

        Returns:
            True if paused successfully
        """
        if self._lifecycle_state != LifecycleState.RUNNING:
            return False

        self._lifecycle_state = LifecycleState.PAUSED
        logger.debug(f"[{self._lifecycle_name}] Paused")
        return True

    async def resume(self) -> bool:
        """Resume from paused state.

        Returns:
            True if resumed successfully
        """
        if self._lifecycle_state != LifecycleState.PAUSED:
            return False

        self._lifecycle_state = LifecycleState.RUNNING
        logger.debug(f"[{self._lifecycle_name}] Resumed")
        return True

    # -------------------------------------------------------------------------
    # Hooks (Override in Subclasses)
    # -------------------------------------------------------------------------

    async def _on_start(self) -> None:
        """Called during startup. Override in subclasses."""
        pass

    async def _on_stop(self) -> None:
        """Called during shutdown. Override in subclasses."""
        pass

    async def _on_cycle(self) -> None:
        """Called each loop iteration. Override if using default loop."""
        pass

    async def _on_error(self, error: Exception) -> None:
        """Called when an error occurs. Override for custom handling."""
        pass

    # -------------------------------------------------------------------------
    # Loop Management
    # -------------------------------------------------------------------------

    async def _lifecycle_loop_wrapper(self) -> None:
        """Wrapper around _run_loop with error handling."""
        try:
            if hasattr(self, "_run_loop"):
                await self._run_loop()
            else:
                # Default loop using _on_cycle
                await self._default_loop()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._last_error = e
            self._error_count += 1
            logger.error(f"[{self._lifecycle_name}] Loop error: {e}")
            await self._on_error(e)

    async def _default_loop(self) -> None:
        """Default execution loop using _on_cycle."""
        while not self._should_stop():
            try:
                await self._on_cycle()
                self._cycle_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"[{self._lifecycle_name}] Cycle error: {e}")
                await self._on_error(e)

            await self._wait_for_next_cycle()

    async def _wait_for_next_cycle(self) -> None:
        """Wait for next cycle or shutdown signal."""
        if self._shutdown_event:
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._cycle_interval,
                )
            except asyncio.TimeoutError:
                pass

    def _should_stop(self) -> bool:
        """Check if loop should stop."""
        return (
            self._lifecycle_state != LifecycleState.RUNNING
            or (self._shutdown_event and self._shutdown_event.is_set())
        )

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "LifecycleMixin":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        """Async context manager exit."""
        await self.stop()
        return False  # Don't suppress exceptions

    # -------------------------------------------------------------------------
    # Health Check Integration
    # -------------------------------------------------------------------------

    def get_lifecycle_health(self) -> dict[str, Any]:
        """Get health information for lifecycle.

        Returns:
            Health check result dict
        """
        return {
            "name": self._lifecycle_name,
            "state": self._lifecycle_state.value,
            "running": self.is_running,
            "uptime_seconds": self.uptime,
            "cycle_count": self._cycle_count,
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None,
        }


# =============================================================================
# Event Subscription Mixin
# =============================================================================


class EventSubscriptionMixin:
    """Mixin for components that subscribe to events.

    Provides:
    - Declarative event subscriptions
    - Automatic subscription on start
    - Automatic unsubscription on stop

    Usage:
        class MyHandler(LifecycleMixin, EventSubscriptionMixin):
            def _get_event_subscriptions(self) -> dict[str, Callable]:
                return {
                    "TRAINING_COMPLETED": self._on_training_completed,
                    "MODEL_PROMOTED": self._on_model_promoted,
                }

            async def _on_training_completed(self, event: Any) -> None:
                # Handle event
                pass
    """

    def __init__(self) -> None:
        """Initialize event subscription mixin."""
        self._subscription_ids: list[tuple[Any, Any]] = []

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions. Override in subclasses.

        Returns:
            Dict mapping event types to handler functions
        """
        return {}

    async def _subscribe_to_events(self) -> None:
        """Subscribe to all configured events."""
        try:
            from app.coordination.event_router import subscribe

            subscriptions = self._get_event_subscriptions()
            for event_type, handler in subscriptions.items():
                try:
                    subscribe(event_type, handler)
                    self._subscription_ids.append((event_type, handler))
                except Exception as e:
                    name = getattr(self, "_lifecycle_name", self.__class__.__name__)
                    logger.debug(f"[{name}] Failed to subscribe to {event_type}: {e}")

        except ImportError:
            pass

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscription_ids:
            return

        try:
            from app.coordination.event_router import unsubscribe

            for event_type, handler in self._subscription_ids:
                try:
                    unsubscribe(event_type, handler)
                except (KeyError, ValueError, AttributeError) as e:
                    # Dec 29, 2025: Narrowed from bare Exception
                    # KeyError: subscription not found (already unsubscribed)
                    # ValueError: invalid subscription format
                    # AttributeError: router is None
                    logger.debug(f"Unsubscribe warning for {event_type}: {e}")
        except ImportError:
            pass

        self._subscription_ids.clear()


# =============================================================================
# Combined Mixin
# =============================================================================


class ManagedComponent(LifecycleMixin, EventSubscriptionMixin):
    """Combined mixin for fully managed components.

    Provides:
    - Full lifecycle management (start/stop/restart)
    - Event subscription management
    - Health check integration

    Usage:
        class MyDaemon(ManagedComponent):
            def __init__(self):
                super().__init__(name="my_daemon", cycle_interval=30.0)

            async def _on_start(self) -> None:
                await self._subscribe_to_events()

            async def _on_stop(self) -> None:
                await self._unsubscribe_from_events()

            async def _on_cycle(self) -> None:
                # Main work
                pass

            def _get_event_subscriptions(self) -> dict:
                return {"SOME_EVENT": self._handle_event}
    """

    def __init__(
        self,
        name: str = "component",
        cycle_interval: float = 60.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        """Initialize managed component."""
        LifecycleMixin.__init__(
            self,
            name=name,
            cycle_interval=cycle_interval,
            shutdown_timeout=shutdown_timeout,
        )
        EventSubscriptionMixin.__init__(self)

    async def _on_start(self) -> None:
        """Start hook - subscribes to events."""
        await self._subscribe_to_events()

    async def _on_stop(self) -> None:
        """Stop hook - unsubscribes from events."""
        await self._unsubscribe_from_events()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "LifecycleState",
    "LifecycleMixin",
    "EventSubscriptionMixin",
    "ManagedComponent",
]
