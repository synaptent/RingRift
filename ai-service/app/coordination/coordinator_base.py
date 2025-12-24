"""Base classes and protocols for coordinators.

This module provides a unified base for all coordinator/manager classes in the
system, consolidating common patterns:
- SQLite persistence with thread-local connections
- State management (running, status)
- Dependency injection via setters
- Async locking for concurrency control
- Stats/metrics reporting
- Lifecycle management (initialize, start, stop, shutdown)

Usage:
    from app.coordination.coordinator_base import (
        CoordinatorBase,
        CoordinatorProtocol,
        SQLitePersistenceMixin,
    )

    class MyCoordinator(CoordinatorBase, SQLitePersistenceMixin):
        def __init__(self, db_path: Path):
            super().__init__()
            self.init_db(db_path)

        def _get_schema(self) -> str:
            return '''CREATE TABLE IF NOT EXISTS my_table (...)'''

        async def get_stats(self) -> Dict[str, Any]:
            return {"status": self.status, "count": self._count}

See: docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.distributed.db_utils import ThreadLocalConnectionPool

logger = logging.getLogger(__name__)


class CoordinatorStatus(str, Enum):
    """Common status values for coordinators."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CoordinatorStats:
    """Base statistics for coordinators.

    Subclasses can extend this with additional fields.
    """
    status: CoordinatorStatus = CoordinatorStatus.INITIALIZING
    uptime_seconds: float = 0.0
    operations_count: int = 0
    errors_count: int = 0
    last_operation_time: float = 0.0
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "operations_count": self.operations_count,
            "errors_count": self.errors_count,
            "last_operation_time": self.last_operation_time,
            "last_error": self.last_error,
            **self.extra,
        }


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Protocol for coordinator classes.

    This defines the common interface that all coordinators should implement.
    Using Protocol allows structural subtyping - classes that have these methods
    are considered coordinators even without explicit inheritance.
    """

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the coordinator is actively running."""
        ...

    async def initialize(self) -> None:
        """Initialize the coordinator (async setup)."""
        ...

    async def start(self) -> None:
        """Start the coordinator's main operation."""
        ...

    async def stop(self) -> None:
        """Stop the coordinator gracefully."""
        ...

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        ...


class CoordinatorBase(ABC):
    """Base class for coordinators with common functionality.

    Provides:
    - Status and running state management
    - Async lock for concurrency control
    - Lifecycle management (initialize, start, stop, shutdown)
    - Stats tracking
    - Dependency injection pattern via setters

    Subclasses should implement:
    - async def _do_start(self): Main startup logic
    - async def _do_stop(self): Main shutdown logic
    - async def get_stats(self) -> Dict[str, Any]: Custom stats
    """

    def __init__(self, name: str | None = None):
        """Initialize the coordinator base.

        Args:
            name: Optional name for logging. Defaults to class name.
        """
        self._name = name or self.__class__.__name__
        self._status = CoordinatorStatus.INITIALIZING
        self._running = False
        self._start_time: float = 0.0
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = CoordinatorStats()
        self._operations_count = 0
        self._errors_count = 0
        self._last_operation_time: float = 0.0
        self._last_error: str | None = None

        # Dependency injection slots
        self._dependencies: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Coordinator name for logging."""
        return self._name

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the coordinator is actively running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Seconds since start (0 if not running)."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def set_dependency(self, name: str, value: Any) -> None:
        """Set a named dependency.

        This is the preferred pattern for dependency injection,
        allowing dependencies to be set after construction.

        Args:
            name: Dependency name
            value: Dependency value
        """
        self._dependencies[name] = value
        logger.debug(f"[{self._name}] Set dependency: {name}")

    def get_dependency(self, name: str, default: Any = None) -> Any:
        """Get a named dependency.

        Args:
            name: Dependency name
            default: Default value if not set

        Returns:
            The dependency value or default
        """
        return self._dependencies.get(name, default)

    def has_dependency(self, name: str) -> bool:
        """Check if a dependency is set."""
        return name in self._dependencies

    async def initialize(self) -> None:
        """Initialize the coordinator.

        Override _do_initialize for custom initialization.
        """
        async with self._lock:
            if self._status != CoordinatorStatus.INITIALIZING:
                return

            try:
                await self._do_initialize()
                self._status = CoordinatorStatus.READY
                logger.info(f"[{self._name}] Initialized")
            except Exception as e:
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Initialization failed: {e}")
                raise

    async def start(self) -> None:
        """Start the coordinator.

        Override _do_start for custom start logic.
        """
        async with self._lock:
            if self._running:
                return

            if self._status == CoordinatorStatus.INITIALIZING:
                await self._do_initialize()

            try:
                self._running = True
                self._start_time = time.time()
                self._status = CoordinatorStatus.RUNNING
                await self._do_start()
                logger.info(f"[{self._name}] Started")
            except Exception as e:
                self._running = False
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Start failed: {e}")
                raise

    async def stop(self) -> None:
        """Stop the coordinator gracefully.

        Override _do_stop for custom stop logic.
        """
        async with self._lock:
            if not self._running:
                return

            try:
                self._status = CoordinatorStatus.DRAINING
                await self._do_stop()
                self._running = False
                self._status = CoordinatorStatus.STOPPED
                logger.info(f"[{self._name}] Stopped")
            except Exception as e:
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Stop failed: {e}")
                raise

    async def shutdown(self) -> None:
        """Shutdown and cleanup all resources.

        Override _do_shutdown for custom cleanup.
        """
        await self.stop()
        await self._do_shutdown()
        logger.info(f"[{self._name}] Shutdown complete")

    async def pause(self) -> None:
        """Pause the coordinator temporarily."""
        if self._running and self._status == CoordinatorStatus.RUNNING:
            self._status = CoordinatorStatus.PAUSED
            logger.info(f"[{self._name}] Paused")

    async def resume(self) -> None:
        """Resume from paused state."""
        if self._running and self._status == CoordinatorStatus.PAUSED:
            self._status = CoordinatorStatus.RUNNING
            logger.info(f"[{self._name}] Resumed")

    def record_operation(self) -> None:
        """Record a successful operation."""
        self._operations_count += 1
        self._last_operation_time = time.time()

    def record_error(self, error: Exception | None = None) -> None:
        """Record an error."""
        self._errors_count += 1
        if error:
            self._last_error = str(error)

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get current statistics.

        Subclasses should call super().get_stats() and extend.

        Returns:
            Dictionary of statistics
        """
        return {
            "name": self._name,
            "status": self._status.value,
            "is_running": self._running,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "operations_count": self._operations_count,
            "errors_count": self._errors_count,
            "last_operation_time": self._last_operation_time,
            "last_error": self._last_error,
        }

    # Hooks for subclasses - override these (optional, not abstract)
    async def _do_initialize(self) -> None:  # noqa: B027 - optional hook
        """Custom initialization logic. Override in subclass."""

    async def _do_start(self) -> None:  # noqa: B027 - optional hook
        """Custom start logic. Override in subclass."""

    async def _do_stop(self) -> None:  # noqa: B027 - optional hook
        """Custom stop logic. Override in subclass."""

    async def _do_shutdown(self) -> None:  # noqa: B027 - optional hook
        """Custom shutdown/cleanup logic. Override in subclass."""


class SQLitePersistenceMixin:
    """Mixin for SQLite-based persistence.

    Provides:
    - Thread-local database connections
    - WAL mode for concurrent access
    - Schema initialization

    Usage:
        class MyManager(CoordinatorBase, SQLitePersistenceMixin):
            def __init__(self, db_path: Path):
                super().__init__()
                self.init_db(db_path)

            def _get_schema(self) -> str:
                return '''CREATE TABLE IF NOT EXISTS my_table (...)'''
    """

    _db_path: Path | None = None
    _db_pool: ThreadLocalConnectionPool | None = None  # December 2025 - consolidated

    def init_db(
        self,
        db_path: Path,
        profile: str = "standard",
    ) -> None:
        """Initialize database with schema.

        Args:
            db_path: Path to SQLite database file
            profile: PRAGMA profile ("standard", "extended", "quick")
                     - standard: 10s busy_timeout, WAL, NORMAL sync
                     - extended: 30s busy_timeout, extra cache (for cross-process)
                     - quick: 5s busy_timeout (for registry lookups)

        December 2025 - Now uses ThreadLocalConnectionPool with centralized
        PRAGMA settings from app/config/thresholds.py
        """
        # Import here to avoid circular imports
        from app.distributed.db_utils import ThreadLocalConnectionPool

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use centralized connection pool (December 2025)
        self._db_pool = ThreadLocalConnectionPool(
            db_path=db_path,
            profile=profile,
        )

        # Initialize schema
        conn = self._get_connection()
        schema = self._get_schema()
        if schema:
            conn.executescript(schema)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Returns:
            SQLite connection configured with centralized PRAGMAs

        December 2025 - Delegates to ThreadLocalConnectionPool for consistent
        PRAGMA configuration across all coordinators.
        """
        if self._db_pool is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")

        return self._db_pool.get_connection()

    def _close_connection(self) -> None:
        """Close the thread-local database connection."""
        if self._db_pool is not None:
            self._db_pool.close_connection()

    def _get_schema(self) -> str:
        """Get database schema SQL.

        Override in subclass to define tables and indexes.

        Returns:
            SQL CREATE statements
        """
        return ""


class SingletonMixin:
    """Mixin for singleton pattern.

    Ensures only one instance per class exists.

    Usage:
        class MyManager(CoordinatorBase, SingletonMixin):
            @classmethod
            def get_instance(cls) -> 'MyManager':
                return cls._get_or_create_instance()
    """

    _instances: dict[type, Any] = {}
    _instance_lock = threading.Lock()

    @classmethod
    def _get_or_create_instance(cls, *args, **kwargs) -> Any:
        """Get or create the singleton instance.

        Args:
            *args, **kwargs: Constructor arguments (only used on first call)

        Returns:
            The singleton instance
        """
        if cls not in cls._instances:
            with cls._instance_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def _clear_instance(cls) -> None:
        """Clear the singleton instance (mainly for testing)."""
        with cls._instance_lock:
            cls._instances.pop(cls, None)


class CallbackMixin:
    """Mixin for callback-based extensibility.

    Provides a pattern for registering and invoking callbacks.

    Usage:
        class MyCoordinator(CoordinatorBase, CallbackMixin):
            async def on_event(self, event_type: str, data: dict):
                await self.invoke_callbacks(event_type, data)

        # Register callbacks
        coordinator.register_callback("job_complete", my_handler)
    """

    _callbacks: dict[str, list[Callable]]

    def __init_callbacks__(self) -> None:
        """Initialize callbacks dictionary. Call in __init__."""
        if not hasattr(self, "_callbacks"):
            self._callbacks = {}

    def register_callback(
        self,
        event_type: str,
        callback: Callable,
    ) -> None:
        """Register a callback for an event type.

        Args:
            event_type: Type of event to handle
            callback: Callback function (sync or async)
        """
        if not hasattr(self, "_callbacks"):
            self._callbacks = {}
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def unregister_callback(
        self,
        event_type: str,
        callback: Callable,
    ) -> None:
        """Unregister a callback.

        Args:
            event_type: Type of event
            callback: Callback to remove
        """
        if hasattr(self, "_callbacks") and event_type in self._callbacks:
            with contextlib.suppress(ValueError):
                self._callbacks[event_type].remove(callback)

    async def invoke_callbacks(
        self,
        event_type: str,
        *args,
        **kwargs,
    ) -> list[Any]:
        """Invoke all callbacks for an event type.

        Args:
            event_type: Type of event
            *args, **kwargs: Arguments to pass to callbacks

        Returns:
            List of callback return values
        """
        if not hasattr(self, "_callbacks"):
            return []

        results = []
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Callback error for {event_type}: {e}")
                results.append(None)
        return results


class EventDrivenMonitorMixin:
    """Mixin for converting polling-based monitors to event-driven.

    This mixin provides a pattern for monitors that should react to events
    instead of (or in addition to) polling.

    Migration Guide (December 2025):
        BEFORE (polling loop):
            class MyMonitor:
                async def run(self):
                    while self._running:
                        status = await self._check_status()
                        if status.changed:
                            await self._handle_change(status)
                        await asyncio.sleep(60)  # Poll every minute

        AFTER (event-driven):
            class MyMonitor(CoordinatorBase, EventDrivenMonitorMixin):
                async def _do_start(self):
                    # Subscribe to relevant events
                    await self.subscribe_to_events([
                        "TRAINING_COMPLETED",
                        "MODEL_PROMOTED",
                    ])

                async def _handle_event(self, event_type: str, payload: dict):
                    if event_type == "TRAINING_COMPLETED":
                        await self._on_training_complete(payload)

    Benefits:
        - Immediate reaction to events (no polling delay)
        - Reduced CPU usage (no busy-waiting)
        - Better coordination with other components
        - Automatic deduplication of events

    Usage:
        class MyMonitor(CoordinatorBase, EventDrivenMonitorMixin):
            async def _do_initialize(self):
                await self.init_event_subscriptions()
                self.register_event_handler("TRAINING_COMPLETED", self._on_training)

            async def _on_training(self, payload: dict):
                # Handle training completion
                pass
    """

    _event_handlers: dict[str, list[Callable]]
    _event_subscriptions: list[str]
    _event_bus_connected: bool

    def init_event_subscriptions(self) -> None:
        """Initialize event subscription tracking. Call in __init__ or _do_initialize."""
        self._event_handlers = {}
        self._event_subscriptions = []
        self._event_bus_connected = False

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Event type string (e.g., "TRAINING_COMPLETED")
            handler: Async or sync function taking payload dict
        """
        if not hasattr(self, "_event_handlers"):
            self._event_handlers = {}
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event: {event_type}")

    async def subscribe_to_events(self, event_types: list[str]) -> bool:
        """Subscribe to events from the unified event coordinator.

        Args:
            event_types: List of event type strings to subscribe to

        Returns:
            True if subscription succeeded
        """
        try:
            from app.coordination.event_router import get_event_coordinator

            coordinator = get_event_coordinator()
            for event_type in event_types:
                coordinator.register_handler(event_type, self._dispatch_event)
                if not hasattr(self, "_event_subscriptions"):
                    self._event_subscriptions = []
                self._event_subscriptions.append(event_type)

            self._event_bus_connected = True
            logger.info(f"Subscribed to events: {event_types}")
            return True

        except ImportError:
            logger.warning("Event coordinator not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")
            return False

    async def _dispatch_event(self, payload: dict[str, Any]) -> None:
        """Dispatch event to registered handlers.

        Args:
            payload: Event payload including 'event_type' key
        """
        event_type = payload.get("event_type", payload.get("type", ""))

        if not hasattr(self, "_event_handlers"):
            return

        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as e:
                logger.warning(f"Event handler error for {event_type}: {e}")

    def get_event_status(self) -> dict[str, Any]:
        """Get status of event subscriptions.

        Returns:
            Dict with subscription status info
        """
        return {
            "connected": getattr(self, "_event_bus_connected", False),
            "subscriptions": getattr(self, "_event_subscriptions", []),
            "handlers": list(getattr(self, "_event_handlers", {}).keys()),
        }


# Convenience function to check if something is a coordinator
def is_coordinator(obj: Any) -> bool:
    """Check if an object implements the CoordinatorProtocol.

    Args:
        obj: Object to check

    Returns:
        True if obj is a coordinator
    """
    return isinstance(obj, CoordinatorProtocol)


class CoordinatorRegistry:
    """Registry for managing multiple coordinators and orchestrating shutdown.

    This singleton registry tracks all active coordinators and provides:
    - Centralized coordinator registration
    - Graceful shutdown orchestration
    - Signal handler integration for SIGTERM/SIGINT

    Usage:
        from app.coordination.coordinator_base import CoordinatorRegistry

        # Get the singleton registry
        registry = CoordinatorRegistry.get_instance()

        # Register coordinators
        registry.register(my_coordinator)

        # Install signal handlers for graceful shutdown
        registry.install_signal_handlers()

        # Or manually trigger shutdown
        await registry.shutdown_all(timeout=30.0)
    """

    _instance: CoordinatorRegistry | None = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the registry. Use get_instance() instead."""
        self._coordinators: dict[str, CoordinatorBase] = {}
        self._priorities: dict[str, int] = {}
        self._shutdown_order: list[str] = []
        self._shutting_down = False
        self._shutdown_complete = asyncio.Event()

    @classmethod
    def get_instance(cls) -> CoordinatorRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def register(
        self,
        coordinator: CoordinatorBase,
        shutdown_priority: int = 0,
    ) -> None:
        """Register a coordinator with the registry.

        Args:
            coordinator: The coordinator to register
            shutdown_priority: Lower values shut down first (default 0)
        """
        name = coordinator.name
        if name in self._coordinators:
            logger.warning(f"Coordinator {name} already registered, replacing")

        self._coordinators[name] = coordinator
        self._priorities[name] = shutdown_priority
        self._update_shutdown_order()
        logger.debug(f"Registered coordinator: {name} (priority={shutdown_priority})")

    def unregister(self, name: str) -> CoordinatorBase | None:
        """Unregister a coordinator by name.

        Args:
            name: Name of the coordinator to remove

        Returns:
            The removed coordinator or None
        """
        coord = self._coordinators.pop(name, None)
        if coord:
            self._priorities.pop(name, None)
            self._update_shutdown_order()
            logger.debug(f"Unregistered coordinator: {name}")
        return coord

    def get(self, name: str) -> CoordinatorBase | None:
        """Get a coordinator by name."""
        return self._coordinators.get(name)

    def list_coordinators(self) -> list[str]:
        """List all registered coordinator names."""
        return list(self._coordinators.keys())

    def _update_shutdown_order(self) -> None:
        """Update the shutdown order based on priority.

        Lower priority values shut down first. Coordinators with the same
        priority are shut down in reverse registration order.
        """
        # Sort by priority (ascending), then by reverse insertion order for ties
        names = list(self._coordinators.keys())
        # Reverse so that for same priority, later registered shuts down first
        names.reverse()
        self._shutdown_order = sorted(names, key=lambda n: self._priorities.get(n, 0))

    async def shutdown_all(
        self,
        timeout: float = 30.0,
        force_after_timeout: bool = True,
    ) -> dict[str, bool]:
        """Gracefully shutdown all registered coordinators.

        Args:
            timeout: Maximum seconds to wait for each coordinator
            force_after_timeout: If True, continue even if a coordinator times out

        Returns:
            Dict mapping coordinator names to success status
        """
        if self._shutting_down:
            logger.warning("Shutdown already in progress")
            await self._shutdown_complete.wait()
            return {}

        self._shutting_down = True
        results: dict[str, bool] = {}

        logger.info(f"Starting graceful shutdown of {len(self._coordinators)} coordinators")

        for name in self._shutdown_order:
            coord = self._coordinators.get(name)
            if not coord:
                continue

            try:
                logger.info(f"Shutting down coordinator: {name}")
                await asyncio.wait_for(coord.shutdown(), timeout=timeout)
                results[name] = True
                logger.info(f"Coordinator {name} shutdown complete")
            except asyncio.TimeoutError:
                logger.error(f"Coordinator {name} shutdown timed out after {timeout}s")
                results[name] = False
                if not force_after_timeout:
                    break
            except Exception as e:
                logger.error(f"Coordinator {name} shutdown failed: {e}")
                results[name] = False

        self._shutdown_complete.set()
        logger.info(f"Graceful shutdown complete: {sum(results.values())}/{len(results)} succeeded")
        return results

    async def drain_all(self, timeout: float = 60.0) -> None:
        """Put all coordinators into draining mode.

        This allows coordinators to finish in-progress work before shutdown.

        Args:
            timeout: Maximum seconds to wait for draining
        """
        logger.info(f"Draining {len(self._coordinators)} coordinators")

        async def drain_one(coord: CoordinatorBase) -> None:
            try:
                await coord.stop()
            except Exception as e:
                logger.warning(f"Error draining {coord.name}: {e}")

        tasks = [drain_one(c) for c in self._coordinators.values() if c.is_running]
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install signal handlers for graceful shutdown.

        Handles SIGTERM and SIGINT to trigger graceful shutdown.

        Args:
            loop: Event loop to use (defaults to running loop)
        """
        import signal

        def shutdown_handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown")

            # Schedule the async shutdown
            try:
                event_loop = loop or asyncio.get_running_loop()
                event_loop.create_task(self.shutdown_all())
            except RuntimeError:
                # No running loop, try to run synchronously
                logger.warning("No event loop running, attempting sync shutdown")
                try:
                    asyncio.run(self.shutdown_all(timeout=10.0))
                except Exception as e:
                    logger.error(f"Sync shutdown failed: {e}")

        # Install handlers for common termination signals
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, shutdown_handler)
                logger.debug(f"Installed shutdown handler for {sig.name}")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not install handler for {sig.name}: {e}")

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary of all registered coordinators.

        Returns:
            Dict with overall health and per-coordinator status
        """
        statuses = {}
        all_healthy = True

        for name, coord in self._coordinators.items():
            status = coord.status.value
            is_healthy = coord.status in (CoordinatorStatus.READY, CoordinatorStatus.RUNNING)
            if not is_healthy and coord.is_running:
                all_healthy = False

            statuses[name] = {
                "status": status,
                "healthy": is_healthy,
                "uptime_seconds": coord.uptime_seconds,
            }

        return {
            "healthy": all_healthy,
            "coordinator_count": len(self._coordinators),
            "shutting_down": self._shutting_down,
            "coordinators": statuses,
        }


# Module-level convenience functions
def get_coordinator_registry() -> CoordinatorRegistry:
    """Get the global coordinator registry."""
    return CoordinatorRegistry.get_instance()


async def shutdown_all_coordinators(timeout: float = 30.0) -> dict[str, bool]:
    """Shutdown all registered coordinators.

    Convenience function for shutdown_all on the global registry.
    """
    return await get_coordinator_registry().shutdown_all(timeout=timeout)


def get_all_coordinators() -> dict[str, CoordinatorBase]:
    """Get all registered coordinators.

    Returns:
        Dict mapping coordinator names to their instances

    Usage:
        from app.coordination.coordinator_base import get_all_coordinators

        coordinators = get_all_coordinators()
        for name, coord in coordinators.items():
            print(f"{name}: {coord.status.value}")
    """
    return get_coordinator_registry()._coordinators.copy()


def get_coordinator_statuses() -> dict[str, dict[str, Any]]:
    """Get status of all registered coordinators.

    Returns:
        Dict mapping coordinator names to their status dicts

    Usage:
        from app.coordination.coordinator_base import get_coordinator_statuses

        statuses = get_coordinator_statuses()
        for name, status in statuses.items():
            print(f"{name}: {status['status']}, uptime: {status['uptime_seconds']}s")
    """
    result = {}
    for name, coord in get_coordinator_registry()._coordinators.items():
        # Try to get get_status if available, otherwise use basic info
        if hasattr(coord, 'get_status') and callable(coord.get_status):
            try:
                result[name] = coord.get_status()
            except Exception:
                result[name] = {
                    "status": coord.status.value,
                    "is_running": coord.is_running,
                    "uptime_seconds": coord.uptime_seconds,
                }
        else:
            result[name] = {
                "status": coord.status.value,
                "is_running": coord.is_running,
                "uptime_seconds": coord.uptime_seconds,
            }
    return result


__all__ = [
    "CallbackMixin",
    # Base classes
    "CoordinatorBase",
    # Protocols
    "CoordinatorProtocol",
    # Registry
    "CoordinatorRegistry",
    "CoordinatorStats",
    # Enums and data classes
    "CoordinatorStatus",
    "EventDrivenMonitorMixin",
    # Mixins
    "SQLitePersistenceMixin",
    "SingletonMixin",
    "get_all_coordinators",
    "get_coordinator_registry",
    "get_coordinator_statuses",
    # Functions
    "is_coordinator",
    "shutdown_all_coordinators",
]
