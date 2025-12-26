"""Centralized Async Bridge Manager for RingRift AI (December 2025).

Provides a shared executor pool and lifecycle management for all async bridges
that wrap synchronous components. This eliminates duplicate thread pools and
provides unified shutdown coordination.

Components managed:
- AsyncTrainingBridge: Wraps TrainingCoordinator
- SyncOrchestrator: Wraps data/model sync operations
- BackgroundEvaluator: Wraps evaluation operations

Benefits:
- Single shared ThreadPoolExecutor (reduced memory, better utilization)
- Coordinated lifecycle management
- Health monitoring across all bridges
- Graceful shutdown coordination

Usage:
    from app.coordination.async_bridge_manager import (
        AsyncBridgeManager,
        get_bridge_manager,
        get_shared_executor,
    )

    # Get shared executor for any bridge
    executor = get_shared_executor()
    result = await loop.run_in_executor(executor, sync_function)

    # Or use the manager directly
    manager = get_bridge_manager()
    await manager.initialize()

    # Graceful shutdown
    await manager.shutdown()
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar
from weakref import WeakSet

logger = logging.getLogger(__name__)

# Type variable for generic executor operations
T = TypeVar("T")


@dataclass
class BridgeConfig:
    """Configuration for AsyncBridgeManager."""

    # Thread pool settings
    max_workers: int = 8
    thread_name_prefix: str = "ringrift_bridge"

    # Lifecycle settings
    shutdown_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 60.0

    # Performance settings
    queue_size_warning_threshold: int = 50


@dataclass
class BridgeStats:
    """Statistics for bridge operations."""

    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    active_tasks: int = 0
    peak_active_tasks: int = 0
    avg_task_duration_ms: float = 0.0
    last_task_time: float = 0.0
    bridges_registered: int = 0


@dataclass
class RegisteredBridge:
    """Information about a registered bridge."""

    name: str
    bridge: Any
    registered_at: float
    shutdown_callback: Callable[[], None] | None = None


class AsyncBridgeManager:
    """Manages async bridges and provides shared executor pool.

    This singleton class provides:
    1. A shared ThreadPoolExecutor for all sync->async bridges
    2. Bridge registration and lifecycle management
    3. Coordinated shutdown across all bridges
    4. Health monitoring and statistics

    Example:
        manager = get_bridge_manager()
        await manager.initialize()

        # Register bridges
        manager.register_bridge("training", training_bridge, training_bridge.shutdown)
        manager.register_bridge("sync", sync_orchestrator, sync_orchestrator.shutdown)

        # Execute sync operations in shared pool
        result = await manager.run_sync(blocking_function, arg1, arg2)

        # Graceful shutdown
        await manager.shutdown()
    """

    def __init__(self, config: BridgeConfig | None = None):
        """Initialize AsyncBridgeManager.

        Args:
            config: Configuration (default: BridgeConfig())
        """
        self.config = config or BridgeConfig()
        self._executor: ThreadPoolExecutor | None = None
        self._bridges: dict[str, RegisteredBridge] = {}
        self._stats = BridgeStats()
        self._initialized = False
        self._shutting_down = False
        # Use asyncio.Lock for async methods
        self._async_lock = asyncio.Lock()
        # Keep threading.Lock for sync methods (initialize, register_bridge, etc.)
        self._sync_lock = threading.Lock()

        # Track active futures for graceful shutdown
        self._active_futures: WeakSet[Future] = WeakSet()

    def initialize(self) -> None:
        """Initialize the bridge manager and executor pool.

        Can be called synchronously from any context.
        """
        with self._sync_lock:
            if self._initialized:
                return

            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix=self.config.thread_name_prefix,
            )
            self._initialized = True

            # Register atexit handler for cleanup
            atexit.register(self._atexit_cleanup)

            logger.info(
                f"[AsyncBridgeManager] Initialized with {self.config.max_workers} workers"
            )

    async def initialize_async(self) -> None:
        """Initialize the bridge manager (async version)."""
        self.initialize()

    def _atexit_cleanup(self) -> None:
        """Cleanup on interpreter exit."""
        if self._executor and not self._shutting_down:
            with contextlib.suppress(Exception):
                self._executor.shutdown(wait=False)

    def get_executor(self) -> ThreadPoolExecutor:
        """Get the shared executor pool.

        Automatically initializes if not already done.

        Returns:
            ThreadPoolExecutor instance
        """
        if not self._initialized:
            self.initialize()
        return self._executor

    async def run_sync(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Run a synchronous function in the shared executor pool.

        Args:
            func: Synchronous function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Example:
            result = await manager.run_sync(blocking_io_call, "arg1", key="value")
        """
        if self._shutting_down:
            raise RuntimeError("AsyncBridgeManager is shutting down")

        if not self._initialized:
            self.initialize()

        start_time = time.time()
        # Dec 2025: Use get_running_loop() for proper async context detection
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Track statistics
        async with self._async_lock:
            self._stats.total_tasks_submitted += 1
            self._stats.active_tasks += 1
            self._stats.peak_active_tasks = max(
                self._stats.peak_active_tasks,
                self._stats.active_tasks
            )

        try:
            # Create wrapper for kwargs support
            def wrapper():
                return func(*args, **kwargs)

            future = loop.run_in_executor(self._executor, wrapper)
            self._active_futures.add(future)
            result = await future

            # Update success stats
            duration_ms = (time.time() - start_time) * 1000
            async with self._async_lock:
                self._stats.total_tasks_completed += 1
                self._stats.last_task_time = time.time()
                # Update rolling average
                total_completed = self._stats.total_tasks_completed
                self._stats.avg_task_duration_ms = (
                    (self._stats.avg_task_duration_ms * (total_completed - 1) + duration_ms)
                    / total_completed
                )

            return result

        except (RuntimeError, ValueError, OSError, TypeError, AttributeError):
            async with self._async_lock:
                self._stats.total_tasks_failed += 1
            raise

        finally:
            async with self._async_lock:
                self._stats.active_tasks -= 1

    def register_bridge(
        self,
        name: str,
        bridge: Any,
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        """Register a bridge for lifecycle management.

        Args:
            name: Unique bridge name
            bridge: Bridge instance
            shutdown_callback: Optional callback for shutdown coordination
        """
        with self._sync_lock:
            self._bridges[name] = RegisteredBridge(
                name=name,
                bridge=bridge,
                registered_at=time.time(),
                shutdown_callback=shutdown_callback,
            )
            self._stats.bridges_registered = len(self._bridges)

        logger.debug(f"[AsyncBridgeManager] Registered bridge: {name}")

    def unregister_bridge(self, name: str) -> None:
        """Unregister a bridge.

        Args:
            name: Bridge name to unregister
        """
        with self._sync_lock:
            if name in self._bridges:
                del self._bridges[name]
                self._stats.bridges_registered = len(self._bridges)

    def get_bridge(self, name: str) -> Any | None:
        """Get a registered bridge by name.

        Args:
            name: Bridge name

        Returns:
            Bridge instance or None
        """
        with self._sync_lock:
            reg = self._bridges.get(name)
            return reg.bridge if reg else None

    async def shutdown(self, wait: bool = True) -> None:
        """Gracefully shutdown all bridges and executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info("[AsyncBridgeManager] Starting graceful shutdown...")

        # Call shutdown callbacks for all bridges
        for name, reg in list(self._bridges.items()):
            if reg.shutdown_callback:
                try:
                    logger.debug(f"[AsyncBridgeManager] Shutting down bridge: {name}")
                    callback_result = reg.shutdown_callback()
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                except (RuntimeError, ValueError, OSError, AttributeError) as e:
                    logger.warning(f"[AsyncBridgeManager] Error shutting down {name}: {e}")

        # Wait for active tasks if requested
        if wait and self._active_futures:
            logger.debug(
                f"[AsyncBridgeManager] Waiting for {len(self._active_futures)} active tasks"
            )
            try:
                await asyncio.wait_for(
                    self._wait_for_active_tasks(),
                    timeout=self.config.shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[AsyncBridgeManager] Shutdown timeout, some tasks may be cancelled"
                )

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

        self._initialized = False
        logger.info("[AsyncBridgeManager] Shutdown complete")

    async def _wait_for_active_tasks(self) -> None:
        """Wait for all active tasks to complete."""
        while self._stats.active_tasks > 0:
            await asyncio.sleep(0.1)

    def get_stats(self) -> dict[str, Any]:
        """Get bridge manager statistics.

        Returns:
            Dict with statistics
        """
        with self._sync_lock:
            return {
                "initialized": self._initialized,
                "shutting_down": self._shutting_down,
                "max_workers": self.config.max_workers,
                "total_tasks_submitted": self._stats.total_tasks_submitted,
                "total_tasks_completed": self._stats.total_tasks_completed,
                "total_tasks_failed": self._stats.total_tasks_failed,
                "active_tasks": self._stats.active_tasks,
                "peak_active_tasks": self._stats.peak_active_tasks,
                "avg_task_duration_ms": self._stats.avg_task_duration_ms,
                "last_task_time": self._stats.last_task_time,
                "bridges_registered": self._stats.bridges_registered,
                "bridge_names": list(self._bridges.keys()),
            }

    def get_health(self) -> dict[str, Any]:
        """Get health status.

        Returns:
            Dict with health information
        """
        stats = self.get_stats()

        # Determine health status
        healthy = True
        warnings = []

        if not self._initialized:
            healthy = False
            warnings.append("Not initialized")

        if self._shutting_down:
            healthy = False
            warnings.append("Shutting down")

        if stats["active_tasks"] > self.config.queue_size_warning_threshold:
            warnings.append(f"High queue depth: {stats['active_tasks']}")

        failure_rate = (
            stats["total_tasks_failed"] / max(stats["total_tasks_submitted"], 1)
        )
        if failure_rate > 0.1:
            warnings.append(f"High failure rate: {failure_rate:.1%}")

        return {
            "healthy": healthy and len(warnings) == 0,
            "warnings": warnings,
            "stats": stats,
        }


# =============================================================================
# Singleton Management
# =============================================================================

_manager: AsyncBridgeManager | None = None
_manager_lock = threading.Lock()


def get_bridge_manager(config: BridgeConfig | None = None) -> AsyncBridgeManager:
    """Get the global AsyncBridgeManager singleton.

    Args:
        config: Configuration (only used on first call)

    Returns:
        AsyncBridgeManager instance
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = AsyncBridgeManager(config)
        return _manager


def get_shared_executor() -> ThreadPoolExecutor:
    """Get the shared executor pool.

    Convenience function for getting the executor without
    going through the manager instance.

    Returns:
        ThreadPoolExecutor instance
    """
    return get_bridge_manager().get_executor()


def reset_bridge_manager() -> None:
    """Reset the bridge manager singleton (for testing)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            try:
                # Synchronous cleanup - use sync_lock if available
                with _manager._sync_lock if hasattr(_manager, '_sync_lock') else contextlib.suppress():
                    if _manager._executor:
                        _manager._executor.shutdown(wait=False)
            except (RuntimeError, OSError):
                pass
        _manager = None


async def run_in_bridge_pool(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a synchronous function in the shared bridge pool.

    Convenience function that wraps manager.run_sync().

    Args:
        func: Synchronous function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result

    Example:
        result = await run_in_bridge_pool(blocking_function, arg1, key=value)
    """
    return await get_bridge_manager().run_sync(func, *args, **kwargs)


__all__ = [
    # Main class
    "AsyncBridgeManager",
    "BridgeConfig",
    "BridgeStats",
    "RegisteredBridge",
    # Singleton access
    "get_bridge_manager",
    "get_shared_executor",
    "reset_bridge_manager",
    # Convenience functions
    "run_in_bridge_pool",
]
