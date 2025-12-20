"""Background Task Management for RingRift AI Service.

Provides simplified APIs for managing background tasks with proper lifecycle:
- @background_task decorator for async background work
- TaskManager for centralized task tracking
- Automatic cleanup on shutdown
- Integration with ShutdownManager

Usage:
    from app.core.tasks import (
        background_task,
        TaskManager,
        get_task_manager,
    )

    # Decorator for background tasks
    @background_task(name="sync_data")
    async def sync_data():
        while True:
            await do_sync()
            await asyncio.sleep(60)

    # Start the task
    task = await sync_data.start()

    # Or use TaskManager directly
    manager = get_task_manager()
    task_id = await manager.spawn("my_task", my_coroutine())

    # Graceful shutdown
    await manager.shutdown_all()

Integration with Shutdown (December 2025):
    TaskManager automatically registers with ShutdownManager to ensure
    all background tasks are cancelled during graceful shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    "TaskInfo",
    "TaskManager",
    "TaskState",
    "background_task",
    "get_task_manager",
]

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


class TaskState(Enum):
    """State of a managed task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a managed task."""
    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: Any = None

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.finished_at or time.time()
        return end_time - self.started_at

    @property
    def is_active(self) -> bool:
        """Check if task is still active."""
        return self.state in (TaskState.PENDING, TaskState.RUNNING)


class TaskManager:
    """Manages background tasks with lifecycle tracking.

    Features:
    - Spawn and track async tasks
    - Automatic cleanup on shutdown
    - Task cancellation with timeout
    - Statistics and monitoring
    """

    _instance: TaskManager | None = None

    def __init__(
        self,
        shutdown_timeout: float = 10.0,
        register_shutdown_hook: bool = True,
    ):
        """Initialize TaskManager.

        Args:
            shutdown_timeout: Timeout for task cancellation during shutdown
            register_shutdown_hook: Whether to register with ShutdownManager
        """
        self.shutdown_timeout = shutdown_timeout
        self._tasks: dict[str, asyncio.Task] = {}
        self._task_info: dict[str, TaskInfo] = {}
        self._running = True

        if register_shutdown_hook:
            self._register_shutdown_hook()

    @classmethod
    def get_instance(cls) -> TaskManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_shutdown_hook(self) -> None:
        """Register with ShutdownManager for graceful shutdown."""
        try:
            from app.core.shutdown import get_shutdown_manager

            manager = get_shutdown_manager()
            manager.register(
                name="task_manager",
                handler=self._async_shutdown,
                priority=80,  # Run early in shutdown
                is_async=True,
                timeout=self.shutdown_timeout + 5,
            )
        except ImportError:
            logger.debug("ShutdownManager not available, skipping hook registration")

    async def _async_shutdown(self) -> None:
        """Async shutdown handler."""
        await self.shutdown_all()

    async def spawn(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        *,
        task_id: str | None = None,
    ) -> str:
        """Spawn a new background task.

        Args:
            name: Human-readable task name
            coro: Coroutine to run
            task_id: Optional custom task ID

        Returns:
            Task ID
        """
        task_id = task_id or f"{name}_{uuid.uuid4().hex[:8]}"

        info = TaskInfo(task_id=task_id, name=name)
        self._task_info[task_id] = info

        async def wrapper():
            info.state = TaskState.RUNNING
            info.started_at = time.time()
            try:
                result = await coro
                info.state = TaskState.COMPLETED
                info.result = result
                return result
            except asyncio.CancelledError:
                info.state = TaskState.CANCELLED
                raise
            except Exception as e:
                info.state = TaskState.FAILED
                info.error = str(e)
                logger.error(f"Task {name} failed: {e}")
                raise
            finally:
                info.finished_at = time.time()
                # Clean up task reference
                self._tasks.pop(task_id, None)

        task = asyncio.create_task(wrapper())
        self._tasks[task_id] = task

        logger.debug(f"Spawned background task: {name} ({task_id})")
        return task_id

    async def cancel(self, task_id: str, timeout: float = 5.0) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel
            timeout: Time to wait for cancellation

        Returns:
            True if task was cancelled
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.done():
            return False

        task.cancel()

        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=timeout,
            )

        return True

    async def shutdown_all(self, timeout: float | None = None) -> int:
        """Cancel all running tasks and wait for completion.

        Args:
            timeout: Total timeout for shutdown (uses instance default if None)

        Returns:
            Number of tasks cancelled
        """
        timeout = timeout or self.shutdown_timeout
        self._running = False

        if not self._tasks:
            return 0

        logger.info(f"Shutting down {len(self._tasks)} background tasks")

        # Cancel all tasks
        for _task_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()

        # Wait for all to complete
        if self._tasks:
            tasks = list(self._tasks.values())
            _done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            if pending:
                logger.warning(f"{len(pending)} tasks did not complete within timeout")

        cancelled = sum(
            1 for info in self._task_info.values()
            if info.state == TaskState.CANCELLED
        )

        return cancelled

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information by ID."""
        return self._task_info.get(task_id)

    def get_active_tasks(self) -> list[TaskInfo]:
        """Get all active (pending/running) tasks."""
        return [
            info for info in self._task_info.values()
            if info.is_active
        ]

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tasks (including completed)."""
        return list(self._task_info.values())

    def clear_completed(self) -> int:
        """Clear completed task info from memory.

        Returns:
            Number of entries cleared
        """
        completed_ids = [
            task_id for task_id, info in self._task_info.items()
            if not info.is_active
        ]
        for task_id in completed_ids:
            del self._task_info[task_id]
        return len(completed_ids)

    @property
    def active_count(self) -> int:
        """Number of active tasks."""
        return len([t for t in self._tasks.values() if not t.done()])

    @property
    def is_running(self) -> bool:
        """Whether the manager is accepting new tasks."""
        return self._running


def get_task_manager() -> TaskManager:
    """Get the global TaskManager instance."""
    return TaskManager.get_instance()


class BackgroundTaskWrapper:
    """Wrapper for background task functions."""

    def __init__(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        name: str,
        restart_on_failure: bool,
        restart_delay: float,
    ):
        self.func = func
        self.name = name
        self.restart_on_failure = restart_on_failure
        self.restart_delay = restart_delay
        self._task_id: str | None = None
        functools.update_wrapper(self, func)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly."""
        return await self.func(*args, **kwargs)

    async def start(self, *args: Any, **kwargs: Any) -> str:
        """Start the task in the background.

        Returns:
            Task ID
        """
        manager = get_task_manager()

        if self.restart_on_failure:
            coro = self._run_with_restart(*args, **kwargs)
        else:
            coro = self.func(*args, **kwargs)

        self._task_id = await manager.spawn(self.name, coro)
        return self._task_id

    async def _run_with_restart(self, *args: Any, **kwargs: Any) -> None:
        """Run with automatic restart on failure."""
        while get_task_manager().is_running:
            try:
                await self.func(*args, **kwargs)
                break  # Normal completion
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Task {self.name} failed, restarting in {self.restart_delay}s: {e}"
                )
                await asyncio.sleep(self.restart_delay)

    async def stop(self, timeout: float = 5.0) -> bool:
        """Stop the running task.

        Returns:
            True if task was stopped
        """
        if self._task_id is None:
            return False

        manager = get_task_manager()
        return await manager.cancel(self._task_id, timeout)

    @property
    def task_id(self) -> str | None:
        """Get the current task ID."""
        return self._task_id

    @property
    def is_running(self) -> bool:
        """Check if the task is currently running."""
        if self._task_id is None:
            return False
        info = get_task_manager().get_task(self._task_id)
        return info is not None and info.is_active


def background_task(
    name: str | None = None,
    restart_on_failure: bool = False,
    restart_delay: float = 5.0,
) -> Callable[[F], BackgroundTaskWrapper]:
    """Decorator to mark a function as a background task.

    Args:
        name: Task name (defaults to function name)
        restart_on_failure: Automatically restart on exceptions
        restart_delay: Delay before restart in seconds

    Returns:
        Decorated function with start/stop methods

    Example:
        @background_task(name="data_sync", restart_on_failure=True)
        async def sync_data():
            while True:
                await perform_sync()
                await asyncio.sleep(60)

        # Start the background task
        task_id = await sync_data.start()

        # Check if running
        if sync_data.is_running:
            print("Sync is running")

        # Stop the task
        await sync_data.stop()
    """
    def decorator(func: F) -> BackgroundTaskWrapper:
        task_name = name or func.__name__
        return BackgroundTaskWrapper(
            func=func,
            name=task_name,
            restart_on_failure=restart_on_failure,
            restart_delay=restart_delay,
        )
    return decorator
