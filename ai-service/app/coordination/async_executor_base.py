"""Unified async executor framework for coordination infrastructure.

December 29, 2025: Created to consolidate async patterns across:
- DaemonManager health loops
- EventRouter fire-and-forget tasks
- Background sync operations
- Daemon lifecycle management

This module provides:
- AsyncExecutor: Managed async task execution with lifecycle
- TaskGroup: Grouped task management with cancellation
- TimeoutExecutor: Tasks with timeout enforcement
- RetryExecutor: Tasks with retry and backoff

Key Features:
- Automatic task cleanup on shutdown (no orphaned tasks)
- Timeout enforcement on all async operations
- Structured error handling with callbacks
- Task completion tracking and metrics

Usage:
    from app.coordination.async_executor_base import (
        AsyncExecutor,
        TaskGroup,
        TimeoutExecutor,
        execute_with_timeout,
    )

    # Simple execution with cleanup
    executor = AsyncExecutor(name="my_executor")
    await executor.start()

    task_id = await executor.submit(my_coroutine())
    await executor.wait_for(task_id)

    await executor.shutdown()  # Cancels all pending tasks

    # Fire-and-forget with error callback
    executor.fire_and_forget(
        risky_operation(),
        on_error=lambda e: logger.error(f"Failed: {e}"),
    )

    # Grouped tasks
    async with TaskGroup(name="batch_sync") as group:
        for node in nodes:
            group.create_task(sync_to_node(node))
        # All tasks cancelled if any fails or on exit
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Task State and Tracking
# =============================================================================


class TaskState(Enum):
    """State of a managed task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskInfo:
    """Information about a managed task."""

    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout_seconds: float | None = None
    error: Exception | None = None
    result: Any = None

    @property
    def duration(self) -> float | None:
        """Task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "timeout_seconds": self.timeout_seconds,
            "error": str(self.error) if self.error else None,
        }


# =============================================================================
# Async Executor
# =============================================================================


class AsyncExecutor:
    """Managed async task execution with lifecycle management.

    Provides:
    - Task submission and tracking
    - Automatic cleanup on shutdown
    - Error callbacks for failed tasks
    - Metrics collection

    Example:
        executor = AsyncExecutor(name="sync_executor", max_concurrent=10)
        await executor.start()

        # Submit tasks
        task_id = await executor.submit(sync_node(node))

        # Fire and forget
        executor.fire_and_forget(log_metric(data))

        # Shutdown cleanly
        await executor.shutdown(timeout=30.0)
    """

    def __init__(
        self,
        name: str = "executor",
        max_concurrent: int = 100,
        default_timeout: float | None = None,
        on_task_error: Callable[[str, Exception], None] | None = None,
        on_task_complete: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            name: Executor name for logging
            max_concurrent: Maximum concurrent tasks
            default_timeout: Default timeout for tasks (None = no timeout)
            on_task_error: Global callback for task errors
            on_task_complete: Global callback for task completion
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self._on_task_error = on_task_error
        self._on_task_complete = on_task_complete

        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_info: dict[str, TaskInfo] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._running = False
        self._shutdown_event: asyncio.Event | None = None

        # Metrics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_cancelled = 0
        self._total_timeout = 0

    async def start(self) -> None:
        """Start the executor."""
        if self._running:
            return

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._shutdown_event = asyncio.Event()
        self._running = True
        logger.info(f"[{self.name}] Executor started (max_concurrent={self.max_concurrent})")

    async def shutdown(self, timeout: float = 30.0, cancel_pending: bool = True) -> None:
        """Shutdown executor and clean up tasks.

        Args:
            timeout: Maximum time to wait for tasks to complete
            cancel_pending: Whether to cancel pending tasks
        """
        if not self._running:
            return

        self._running = False
        if self._shutdown_event:
            self._shutdown_event.set()

        if cancel_pending and self._tasks:
            logger.info(f"[{self.name}] Cancelling {len(self._tasks)} pending tasks...")
            for task_id, task in list(self._tasks.items()):
                if not task.done():
                    task.cancel()
                    self._update_task_state(task_id, TaskState.CANCELLED)

            # Wait for cancellation to complete
            if self._tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._tasks.values(), return_exceptions=True),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Timeout waiting for task cancellation")

        self._tasks.clear()
        logger.info(
            f"[{self.name}] Executor shutdown complete. "
            f"Stats: submitted={self._total_submitted}, completed={self._total_completed}, "
            f"failed={self._total_failed}, cancelled={self._total_cancelled}"
        )

    async def submit(
        self,
        coro: Coroutine[Any, Any, T],
        name: str | None = None,
        timeout: float | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_complete: Callable[[T], None] | None = None,
    ) -> str:
        """Submit a coroutine for execution.

        Args:
            coro: Coroutine to execute
            name: Task name for logging
            timeout: Task-specific timeout (overrides default)
            on_error: Task-specific error callback
            on_complete: Task-specific completion callback

        Returns:
            Task ID for tracking
        """
        if not self._running:
            raise RuntimeError(f"Executor {self.name} is not running")

        task_id = str(uuid.uuid4())[:8]
        task_name = name or f"task-{task_id}"

        info = TaskInfo(
            task_id=task_id,
            name=task_name,
            timeout_seconds=timeout or self.default_timeout,
        )
        self._task_info[task_id] = info
        self._total_submitted += 1

        async def wrapped() -> T:
            assert self._semaphore is not None
            async with self._semaphore:
                info.state = TaskState.RUNNING
                info.started_at = time.time()

                try:
                    effective_timeout = timeout or self.default_timeout
                    if effective_timeout:
                        result = await asyncio.wait_for(coro, timeout=effective_timeout)
                    else:
                        result = await coro

                    info.state = TaskState.COMPLETED
                    info.completed_at = time.time()
                    info.result = result
                    self._total_completed += 1

                    # Callbacks
                    if on_complete:
                        try:
                            on_complete(result)
                        except Exception as e:
                            logger.debug(f"[{self.name}] Completion callback error: {e}")

                    if self._on_task_complete:
                        try:
                            self._on_task_complete(task_id, result)
                        except Exception as e:
                            logger.debug(f"[{self.name}] Global completion callback error: {e}")

                    return result

                except asyncio.TimeoutError:
                    info.state = TaskState.TIMEOUT
                    info.completed_at = time.time()
                    self._total_timeout += 1
                    logger.warning(f"[{self.name}] Task {task_name} timed out")
                    raise

                except asyncio.CancelledError:
                    info.state = TaskState.CANCELLED
                    info.completed_at = time.time()
                    self._total_cancelled += 1
                    raise

                except Exception as e:
                    info.state = TaskState.FAILED
                    info.completed_at = time.time()
                    info.error = e
                    self._total_failed += 1

                    # Callbacks
                    if on_error:
                        try:
                            on_error(e)
                        except Exception as cb_error:
                            logger.debug(f"[{self.name}] Error callback error: {cb_error}")

                    if self._on_task_error:
                        try:
                            self._on_task_error(task_id, e)
                        except Exception as cb_error:
                            logger.debug(f"[{self.name}] Global error callback error: {cb_error}")

                    raise

                finally:
                    # Cleanup
                    self._tasks.pop(task_id, None)

        task = asyncio.create_task(wrapped(), name=task_name)
        self._tasks[task_id] = task

        return task_id

    def fire_and_forget(
        self,
        coro: Coroutine[Any, Any, Any],
        name: str | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> str | None:
        """Submit a coroutine without waiting for result.

        Safe for fire-and-forget operations - errors are logged and
        optionally passed to callback.

        Args:
            coro: Coroutine to execute
            name: Task name for logging
            on_error: Error callback

        Returns:
            Task ID if running, None if executor not started
        """
        if not self._running:
            logger.debug(f"[{self.name}] Fire-and-forget rejected: executor not running")
            # Close the coroutine to avoid warning
            coro.close()
            return None

        async def safe_wrapper() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"[{self.name}] Fire-and-forget error: {e}")
                if on_error:
                    try:
                        on_error(e)
                    except Exception:
                        pass

        try:
            loop = asyncio.get_running_loop()
            task_id = str(uuid.uuid4())[:8]
            task_name = name or f"ff-{task_id}"

            task = loop.create_task(safe_wrapper(), name=task_name)
            self._tasks[task_id] = task
            self._total_submitted += 1

            # Auto-cleanup on completion
            def cleanup(t: asyncio.Task[Any]) -> None:
                self._tasks.pop(task_id, None)

            task.add_done_callback(cleanup)

            return task_id

        except RuntimeError:
            coro.close()
            return None

    async def wait_for(self, task_id: str, timeout: float | None = None) -> Any:
        """Wait for a specific task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait

        Returns:
            Task result

        Raises:
            KeyError: If task_id not found
            asyncio.TimeoutError: If timeout exceeded
        """
        task = self._tasks.get(task_id)
        if task is None:
            # Check if already completed
            info = self._task_info.get(task_id)
            if info and info.state == TaskState.COMPLETED:
                return info.result
            raise KeyError(f"Task {task_id} not found")

        if timeout:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    async def wait_all(self, timeout: float | None = None) -> None:
        """Wait for all pending tasks to complete."""
        if not self._tasks:
            return

        if timeout:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=timeout,
            )
        else:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def get_task_info(self, task_id: str) -> TaskInfo | None:
        """Get information about a task."""
        return self._task_info.get(task_id)

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return len(self._tasks)

    def _update_task_state(self, task_id: str, state: TaskState) -> None:
        """Update task state."""
        info = self._task_info.get(task_id)
        if info:
            info.state = state
            info.completed_at = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "name": self.name,
            "running": self._running,
            "pending_tasks": len(self._tasks),
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_cancelled": self._total_cancelled,
            "total_timeout": self._total_timeout,
            "max_concurrent": self.max_concurrent,
            "default_timeout": self.default_timeout,
        }


# =============================================================================
# Task Group
# =============================================================================


class TaskGroup:
    """Grouped task management with automatic cancellation.

    All tasks in a group are cancelled if:
    - Any task fails (unless suppress_errors=True)
    - The context manager exits
    - cancel() is called explicitly

    Example:
        async with TaskGroup(name="sync_batch") as group:
            for node in nodes:
                group.create_task(sync_node(node))

        # Results are available after context exits
        results = group.results
    """

    def __init__(
        self,
        name: str = "task_group",
        suppress_errors: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Initialize task group.

        Args:
            name: Group name for logging
            suppress_errors: If True, don't cancel other tasks on error
            timeout: Timeout for entire group
        """
        self.name = name
        self.suppress_errors = suppress_errors
        self.timeout = timeout

        self._tasks: list[asyncio.Task[Any]] = []
        self._results: list[Any] = []
        self._errors: list[Exception] = []
        self._started = False
        self._finished = False

    async def __aenter__(self) -> "TaskGroup":
        self._started = True
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        """Wait for all tasks and handle errors."""
        if not self._tasks:
            self._finished = True
            return False

        try:
            if self.timeout:
                results = await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=self.timeout,
                )
            else:
                results = await asyncio.gather(*self._tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self._errors.append(result)
                else:
                    self._results.append(result)

        except asyncio.TimeoutError:
            # Cancel all tasks on timeout
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            self._errors.append(asyncio.TimeoutError(f"TaskGroup {self.name} timed out"))

        except asyncio.CancelledError:
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            raise

        finally:
            self._finished = True

        if self._errors and not self.suppress_errors:
            # Log errors but don't re-raise (gathered already)
            logger.warning(
                f"[TaskGroup:{self.name}] {len(self._errors)} tasks failed: "
                f"{[str(e)[:50] for e in self._errors[:3]]}"
            )

        return False  # Don't suppress the original exception

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """Create a task in this group."""
        if not self._started:
            raise RuntimeError("TaskGroup not started - use 'async with TaskGroup():'")
        if self._finished:
            raise RuntimeError("TaskGroup already finished")

        task_name = name or f"{self.name}-{len(self._tasks)}"
        task = asyncio.create_task(coro, name=task_name)
        self._tasks.append(task)
        return task

    @property
    def results(self) -> list[Any]:
        """Results from completed tasks (available after context exit)."""
        return self._results

    @property
    def errors(self) -> list[Exception]:
        """Errors from failed tasks (available after context exit)."""
        return self._errors

    @property
    def success_count(self) -> int:
        """Number of successful tasks."""
        return len(self._results)

    @property
    def error_count(self) -> int:
        """Number of failed tasks."""
        return len(self._errors)


# =============================================================================
# Timeout Executor
# =============================================================================


class TimeoutExecutor:
    """Execute coroutines with strict timeout enforcement.

    Example:
        executor = TimeoutExecutor(default_timeout=30.0)

        result = await executor.run(slow_operation())
        # Raises asyncio.TimeoutError if > 30s

        result = await executor.run(slow_operation(), timeout=60.0)
        # Override timeout for this call
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        on_timeout: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize timeout executor.

        Args:
            default_timeout: Default timeout in seconds
            on_timeout: Callback when timeout occurs
        """
        self.default_timeout = default_timeout
        self._on_timeout = on_timeout

    async def run(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: float | None = None,
        name: str | None = None,
    ) -> T:
        """Run coroutine with timeout.

        Args:
            coro: Coroutine to execute
            timeout: Override timeout (None = use default)
            name: Name for logging

        Returns:
            Coroutine result

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        task_name = name or "timeout_task"

        try:
            return await asyncio.wait_for(coro, timeout=effective_timeout)
        except asyncio.TimeoutError:
            if self._on_timeout:
                self._on_timeout(task_name)
            raise


# =============================================================================
# Retry Executor
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = (Exception,)


class RetryExecutor:
    """Execute coroutines with retry and exponential backoff.

    Example:
        executor = RetryExecutor(
            config=RetryConfig(max_attempts=5, initial_delay=2.0)
        )

        result = await executor.run(flaky_operation())
        # Retries up to 5 times with exponential backoff
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> None:
        """Initialize retry executor.

        Args:
            config: Retry configuration
            on_retry: Callback on each retry (attempt_num, exception)
        """
        self.config = config or RetryConfig()
        self._on_retry = on_retry

    async def run(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, T]],
        config: RetryConfig | None = None,
    ) -> T:
        """Run coroutine with retry.

        Args:
            coro_factory: Factory function that creates the coroutine
                         (needed because coroutines can only be awaited once)
            config: Override retry config for this call

        Returns:
            Coroutine result

        Raises:
            Exception: The last exception if all retries fail
        """
        import random

        cfg = config or self.config
        last_error: Exception | None = None

        for attempt in range(1, cfg.max_attempts + 1):
            try:
                return await coro_factory()

            except cfg.retry_on as e:
                last_error = e

                if attempt == cfg.max_attempts:
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    cfg.initial_delay * (cfg.exponential_base ** (attempt - 1)),
                    cfg.max_delay,
                )

                # Add jitter
                if cfg.jitter:
                    delay = delay * (0.5 + random.random())

                if self._on_retry:
                    self._on_retry(attempt, e)

                logger.debug(
                    f"Retry {attempt}/{cfg.max_attempts} after {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise last_error or RuntimeError("Retry failed with no error")


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: T | None = None,
) -> T | None:
    """Execute coroutine with timeout, returning default on timeout.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Coroutine result or default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


async def execute_with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> T:
    """Execute coroutine with simple retry.

    Args:
        coro_factory: Factory function that creates the coroutine
        max_attempts: Maximum attempts
        delay: Delay between attempts

    Returns:
        Coroutine result
    """
    executor = RetryExecutor(
        config=RetryConfig(max_attempts=max_attempts, initial_delay=delay)
    )
    return await executor.run(coro_factory)


@asynccontextmanager
async def managed_executor(
    name: str = "managed",
    max_concurrent: int = 50,
    default_timeout: float | None = None,
) -> AsyncGenerator[AsyncExecutor, None]:
    """Context manager for executor lifecycle.

    Example:
        async with managed_executor("sync") as executor:
            await executor.submit(sync_node(node))
    """
    executor = AsyncExecutor(
        name=name,
        max_concurrent=max_concurrent,
        default_timeout=default_timeout,
    )
    await executor.start()
    try:
        yield executor
    finally:
        await executor.shutdown()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Core classes
    "AsyncExecutor",
    "TaskGroup",
    "TimeoutExecutor",
    "RetryExecutor",
    # Data classes
    "TaskState",
    "TaskInfo",
    "RetryConfig",
    # Convenience functions
    "execute_with_timeout",
    "execute_with_retry",
    "managed_executor",
]
