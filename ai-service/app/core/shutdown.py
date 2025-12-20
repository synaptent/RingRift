"""Unified Shutdown Manager for graceful process termination.

This module provides centralized shutdown coordination for all components:
- Signal handling (SIGTERM, SIGINT, SIGHUP)
- Ordered resource cleanup with priorities
- Timeout enforcement
- Both sync and async shutdown hooks
- Integration with DaemonManager

Usage:
    from app.core.shutdown import (
        ShutdownManager,
        get_shutdown_manager,
        on_shutdown,
        request_shutdown,
    )

    # Register cleanup handlers
    @on_shutdown(priority=10)  # Higher priority runs first
    def cleanup_connections():
        db.close()

    @on_shutdown(priority=5, async_handler=True)
    async def async_cleanup():
        await client.close()

    # Or programmatic registration
    manager = get_shutdown_manager()
    manager.register("database", db.close, priority=10)

    # Request graceful shutdown
    request_shutdown("User requested exit")

Integration with scripts.lib.process:
    The SignalHandler from scripts.lib.process is suitable for simple scripts.
    Use ShutdownManager for complex services with multiple components.

Priority Guidelines:
    - 100+: Critical infrastructure (event loops, thread pools)
    - 50-99: Core services (databases, message queues)
    - 10-49: Application services (caches, connections)
    - 1-9: Optional cleanup (temp files, logs)
    - 0: Best-effort (no guarantee of execution)
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
import threading
import time
from collections.abc import Callable, Coroutine
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

logger = logging.getLogger(__name__)

__all__ = [
    "ShutdownConfig",
    "ShutdownHook",
    "ShutdownManager",
    "ShutdownPhase",
    "get_shutdown_manager",
    "is_shutting_down",
    "on_shutdown",
    "request_shutdown",
]


class ShutdownPhase(Enum):
    """Phases of shutdown process."""
    RUNNING = "running"
    SIGNALED = "signaled"      # Signal received
    DRAINING = "draining"      # Stop accepting new work
    CLEANUP = "cleanup"        # Running cleanup hooks
    TERMINATING = "terminating"  # Final termination
    COMPLETE = "complete"


@dataclass
class ShutdownHook:
    """Registered shutdown hook."""
    name: str
    handler: Union[Callable[[], None], Callable[[], Coroutine[Any, Any, None]]]
    priority: int = 10
    timeout: float = 10.0
    is_async: bool = False
    critical: bool = False  # If True, shutdown waits for completion

    def __lt__(self, other: ShutdownHook) -> bool:
        # Higher priority runs first
        return self.priority > other.priority


@dataclass
class ShutdownConfig:
    """Configuration for ShutdownManager."""
    timeout: float = 30.0  # Total shutdown timeout
    hook_timeout: float = 10.0  # Per-hook timeout
    force_kill_delay: float = 5.0  # Delay before force kill after timeout
    handle_signals: bool = True  # Register signal handlers
    signals: tuple = (signal.SIGTERM, signal.SIGINT)
    drain_timeout: float = 5.0  # Time to wait for work draining


class ShutdownManager:
    """Unified shutdown manager for graceful process termination.

    Manages ordered shutdown of all registered components with proper
    cleanup sequencing and timeout enforcement.

    Features:
    - Priority-based hook execution (higher priority first)
    - Both sync and async hook support
    - Configurable timeouts per hook
    - Signal handling integration
    - Thread-safe operation
    """

    _instance: ShutdownManager | None = None
    _lock = threading.Lock()

    def __init__(self, config: ShutdownConfig | None = None):
        """Initialize shutdown manager.

        Args:
            config: Shutdown configuration
        """
        self.config = config or ShutdownConfig()
        self._hooks: list[ShutdownHook] = []
        self._phase = ShutdownPhase.RUNNING
        self._shutdown_event = threading.Event()
        self._async_shutdown_event: asyncio.Event | None = None
        self._shutdown_reason: str | None = None
        self._hooks_lock = threading.Lock()
        self._original_handlers: dict[int, Any] = {}
        self._drain_callbacks: list[Callable[[], None]] = []

        # Register signal handlers if configured
        if self.config.handle_signals:
            self._register_signal_handlers()

        # Register atexit handler
        atexit.register(self._atexit_handler)

    @classmethod
    def get_instance(cls) -> ShutdownManager:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._restore_signal_handlers()
            cls._instance = None

    @property
    def phase(self) -> ShutdownPhase:
        """Get current shutdown phase."""
        return self._phase

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._phase != ShutdownPhase.RUNNING

    @property
    def shutdown_reason(self) -> str | None:
        """Get the reason for shutdown."""
        return self._shutdown_reason

    def register(
        self,
        name: str,
        handler: Union[Callable[[], None], Callable[[], Coroutine[Any, Any, None]]],
        *,
        priority: int = 10,
        timeout: float = 10.0,
        is_async: bool = False,
        critical: bool = False,
    ) -> None:
        """Register a shutdown hook.

        Args:
            name: Unique name for this hook
            handler: Cleanup function (sync or async)
            priority: Execution priority (higher runs first)
            timeout: Max time for this hook
            is_async: Whether handler is async
            critical: If True, shutdown waits for completion
        """
        with self._hooks_lock:
            # Remove existing hook with same name
            self._hooks = [h for h in self._hooks if h.name != name]

            hook = ShutdownHook(
                name=name,
                handler=handler,
                priority=priority,
                timeout=timeout,
                is_async=is_async,
                critical=critical,
            )
            self._hooks.append(hook)
            self._hooks.sort()

        logger.debug(f"Registered shutdown hook: {name} (priority={priority})")

    def unregister(self, name: str) -> bool:
        """Unregister a shutdown hook.

        Args:
            name: Name of hook to unregister

        Returns:
            True if hook was found and removed
        """
        with self._hooks_lock:
            original_len = len(self._hooks)
            self._hooks = [h for h in self._hooks if h.name != name]
            removed = len(self._hooks) < original_len

        if removed:
            logger.debug(f"Unregistered shutdown hook: {name}")
        return removed

    def register_drain_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for the drain phase.

        Drain callbacks are called first to stop accepting new work.
        """
        self._drain_callbacks.append(callback)

    def request_shutdown(self, reason: str = "shutdown requested") -> None:
        """Request graceful shutdown.

        Args:
            reason: Reason for shutdown (logged)
        """
        if self._phase != ShutdownPhase.RUNNING:
            logger.debug(f"Shutdown already in progress, ignoring: {reason}")
            return

        logger.info(f"Shutdown requested: {reason}")
        self._shutdown_reason = reason
        self._phase = ShutdownPhase.SIGNALED
        self._shutdown_event.set()

        if self._async_shutdown_event:
            # Schedule setting async event in event loop
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self._async_shutdown_event.set)
            except RuntimeError:
                pass  # No running event loop

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal.

        Args:
            timeout: Max time to wait (None for indefinite)

        Returns:
            True if shutdown was signaled, False on timeout
        """
        return self._shutdown_event.wait(timeout)

    async def wait_for_shutdown_async(self, timeout: float | None = None) -> bool:
        """Async version of wait_for_shutdown.

        Args:
            timeout: Max time to wait (None for indefinite)

        Returns:
            True if shutdown was signaled, False on timeout
        """
        if self._async_shutdown_event is None:
            self._async_shutdown_event = asyncio.Event()

        if self._shutdown_event.is_set():
            self._async_shutdown_event.set()

        try:
            await asyncio.wait_for(
                self._async_shutdown_event.wait(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    def execute_shutdown(self) -> None:
        """Execute the shutdown sequence synchronously."""
        if self._phase == ShutdownPhase.COMPLETE:
            return

        start_time = time.time()
        logger.info("Starting shutdown sequence")

        try:
            # Phase 1: Drain
            self._phase = ShutdownPhase.DRAINING
            self._execute_drain()

            # Phase 2: Cleanup hooks
            self._phase = ShutdownPhase.CLEANUP
            self._execute_hooks_sync()

            # Phase 3: Terminating
            self._phase = ShutdownPhase.TERMINATING

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self._phase = ShutdownPhase.COMPLETE
            elapsed = time.time() - start_time
            logger.info(f"Shutdown complete in {elapsed:.2f}s")

    async def execute_shutdown_async(self) -> None:
        """Execute the shutdown sequence asynchronously."""
        if self._phase == ShutdownPhase.COMPLETE:
            return

        start_time = time.time()
        logger.info("Starting async shutdown sequence")

        try:
            # Phase 1: Drain
            self._phase = ShutdownPhase.DRAINING
            await self._execute_drain_async()

            # Phase 2: Cleanup hooks
            self._phase = ShutdownPhase.CLEANUP
            await self._execute_hooks_async()

            # Phase 3: Terminating
            self._phase = ShutdownPhase.TERMINATING

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self._phase = ShutdownPhase.COMPLETE
            elapsed = time.time() - start_time
            logger.info(f"Async shutdown complete in {elapsed:.2f}s")

    def _execute_drain(self) -> None:
        """Execute drain phase."""
        logger.debug("Executing drain phase")
        for callback in self._drain_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Drain callback error: {e}")
        time.sleep(min(1.0, self.config.drain_timeout))

    async def _execute_drain_async(self) -> None:
        """Execute async drain phase."""
        logger.debug("Executing async drain phase")
        for callback in self._drain_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Drain callback error: {e}")
        await asyncio.sleep(min(1.0, self.config.drain_timeout))

    def _execute_hooks_sync(self) -> None:
        """Execute sync shutdown hooks."""
        with self._hooks_lock:
            hooks = list(self._hooks)

        for hook in hooks:
            self._run_hook_sync(hook)

    def _run_hook_sync(self, hook: ShutdownHook) -> None:
        """Run a single hook synchronously."""
        logger.debug(f"Running shutdown hook: {hook.name}")
        start = time.time()

        try:
            if hook.is_async:
                # Run async hook in new event loop
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        asyncio.wait_for(hook.handler(), timeout=hook.timeout)
                    )
                finally:
                    loop.close()
            else:
                hook.handler()

            elapsed = time.time() - start
            logger.debug(f"Hook {hook.name} completed in {elapsed:.2f}s")

        except asyncio.TimeoutError:
            logger.warning(f"Hook {hook.name} timed out after {hook.timeout}s")
        except Exception as e:
            logger.error(f"Hook {hook.name} failed: {e}")

    async def _execute_hooks_async(self) -> None:
        """Execute shutdown hooks asynchronously."""
        with self._hooks_lock:
            hooks = list(self._hooks)

        # Group by priority for concurrent execution at same level
        priority_groups: dict[int, list[ShutdownHook]] = {}
        for hook in hooks:
            if hook.priority not in priority_groups:
                priority_groups[hook.priority] = []
            priority_groups[hook.priority].append(hook)

        # Execute priority groups in order (highest first)
        for priority in sorted(priority_groups.keys(), reverse=True):
            group = priority_groups[priority]
            await self._execute_hook_group_async(group)

    async def _execute_hook_group_async(self, hooks: list[ShutdownHook]) -> None:
        """Execute a group of hooks concurrently."""
        tasks = []
        for hook in hooks:
            tasks.append(self._run_hook_async(hook))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_hook_async(self, hook: ShutdownHook) -> None:
        """Run a single hook asynchronously."""
        logger.debug(f"Running async shutdown hook: {hook.name}")
        start = time.time()

        try:
            if hook.is_async:
                await asyncio.wait_for(hook.handler(), timeout=hook.timeout)
            else:
                # Run sync handler in executor
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, hook.handler),
                    timeout=hook.timeout,
                )

            elapsed = time.time() - start
            logger.debug(f"Hook {hook.name} completed in {elapsed:.2f}s")

        except asyncio.TimeoutError:
            logger.warning(f"Hook {hook.name} timed out after {hook.timeout}s")
        except Exception as e:
            logger.error(f"Hook {hook.name} failed: {e}")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers."""
        for sig in self.config.signals:
            try:
                self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
            except (ValueError, OSError) as e:
                # Can't set signal handler (e.g., not main thread)
                logger.debug(f"Could not register handler for signal {sig}: {e}")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            with suppress(ValueError, OSError):
                signal.signal(sig, handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name
        self.request_shutdown(f"Received {sig_name}")

        # Execute shutdown in main thread
        self.execute_shutdown()

    def _atexit_handler(self) -> None:
        """Handle process exit."""
        if self._phase == ShutdownPhase.RUNNING:
            self.request_shutdown("Process exiting")
        if self._phase != ShutdownPhase.COMPLETE:
            self.execute_shutdown()


# Module-level convenience functions
def get_shutdown_manager() -> ShutdownManager:
    """Get the global ShutdownManager instance."""
    return ShutdownManager.get_instance()


def request_shutdown(reason: str = "shutdown requested") -> None:
    """Request graceful shutdown."""
    get_shutdown_manager().request_shutdown(reason)


def is_shutting_down() -> bool:
    """Check if shutdown is in progress."""
    return get_shutdown_manager().is_shutting_down


def on_shutdown(
    priority: int = 10,
    timeout: float = 10.0,
    async_handler: bool = False,
    critical: bool = False,
    name: str | None = None,
) -> Callable:
    """Decorator to register a shutdown hook.

    Usage:
        @on_shutdown(priority=50)
        def cleanup_database():
            db.close()

        @on_shutdown(priority=30, async_handler=True)
        async def cleanup_connections():
            await pool.close()
    """
    def decorator(func: Callable) -> Callable:
        hook_name = name or func.__name__
        get_shutdown_manager().register(
            name=hook_name,
            handler=func,
            priority=priority,
            timeout=timeout,
            is_async=async_handler,
            critical=critical,
        )
        return func
    return decorator


@contextmanager
def shutdown_scope(
    name: str,
    cleanup: Callable[[], None],
    priority: int = 10,
):
    """Context manager that registers cleanup on enter and unregisters on exit.

    Usage:
        with shutdown_scope("temp_file", lambda: os.remove(temp_path)):
            process_file(temp_path)
    """
    manager = get_shutdown_manager()
    manager.register(name, cleanup, priority=priority)
    try:
        yield
    finally:
        manager.unregister(name)
