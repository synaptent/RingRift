"""Task Spawning Abstraction for RingRift AI Service.

Provides unified task spawning with:
- Supervised task execution
- Automatic restart policies
- Resource limiting (concurrency, rate)
- Task grouping and cancellation
- Progress tracking

Usage:
    from app.core.task_spawner import TaskSpawner, RestartPolicy

    spawner = TaskSpawner()

    # Spawn a supervised task
    task = await spawner.spawn(
        my_coroutine(),
        name="my_task",
        restart_policy=RestartPolicy.ON_FAILURE,
    )

    # Spawn with resource limits
    async with spawner.limited(max_concurrent=10):
        tasks = [spawner.spawn(work(i)) for i in range(100)]
        await asyncio.gather(*tasks)

    # Clean shutdown
    await spawner.shutdown()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TaskSpawner",
    "SpawnedTask",
    "TaskGroup",
    "RestartPolicy",
    "TaskState",
    "spawn",
    "get_spawner",
]

T = TypeVar("T")


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RestartPolicy(Enum):
    """Restart policy for failed tasks."""
    NEVER = "never"           # Never restart
    ON_FAILURE = "on_failure" # Restart only on exception
    ALWAYS = "always"         # Always restart (for daemons)


class TaskState(Enum):
    """State of a spawned task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESTARTING = "restarting"


@dataclass
class TaskResult:
    """Result of a completed task."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    restarts: int = 0


@dataclass
class SpawnedTask(Generic[T]):
    """Represents a spawned and managed task.

    Attributes:
        name: Task name
        state: Current state
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        restarts: Number of restarts
        last_error: Last error if any
    """
    name: str
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    restarts: int = 0
    last_error: Optional[Exception] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _result: Any = field(default=None, repr=False)

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_done(self) -> bool:
        """Check if task is done."""
        return self.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)

    @property
    def is_running(self) -> bool:
        """Check if task is running."""
        return self.state == TaskState.RUNNING

    async def wait(self) -> Any:
        """Wait for task to complete and get result."""
        if self._task:
            return await self._task
        return self._result

    def cancel(self) -> bool:
        """Cancel the task."""
        if self._task and not self._task.done():
            self._task.cancel()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "restarts": self.restarts,
            "error": str(self.last_error) if self.last_error else None,
        }


# =============================================================================
# Task Group
# =============================================================================

class TaskGroup:
    """Group of related tasks.

    Allows managing multiple tasks as a unit.

    Example:
        group = TaskGroup("workers")

        for i in range(10):
            await group.spawn(worker(i), name=f"worker_{i}")

        await group.wait_all()  # Wait for all
        await group.cancel_all()  # Or cancel all
    """

    def __init__(self, name: str, spawner: "TaskSpawner"):
        self.name = name
        self._spawner = spawner
        self._tasks: Dict[str, SpawnedTask] = {}
        self._lock = asyncio.Lock()

    async def spawn(
        self,
        coro: Coroutine[Any, Any, T],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> SpawnedTask[T]:
        """Spawn a task in this group."""
        task = await self._spawner.spawn(coro, name=name, group=self.name, **kwargs)
        async with self._lock:
            self._tasks[task.name] = task
        return task

    async def wait_all(self, timeout: Optional[float] = None) -> List[SpawnedTask]:
        """Wait for all tasks to complete."""
        async with self._lock:
            tasks = list(self._tasks.values())

        if not tasks:
            return []

        aws = [t.wait() for t in tasks if t._task]
        try:
            await asyncio.wait_for(
                asyncio.gather(*aws, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass

        return tasks

    async def cancel_all(self) -> int:
        """Cancel all tasks in group."""
        async with self._lock:
            tasks = list(self._tasks.values())

        cancelled = 0
        for task in tasks:
            if task.cancel():
                cancelled += 1

        return cancelled

    @property
    def tasks(self) -> List[SpawnedTask]:
        """Get all tasks in group."""
        return list(self._tasks.values())

    @property
    def running_count(self) -> int:
        """Number of running tasks."""
        return sum(1 for t in self._tasks.values() if t.is_running)

    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        return sum(1 for t in self._tasks.values() if t.is_done)


# =============================================================================
# Task Spawner
# =============================================================================

class TaskSpawner:
    """Unified task spawner with supervision.

    Manages async task lifecycle including:
    - Spawning with automatic tracking
    - Restart policies for failed tasks
    - Concurrency limiting
    - Task grouping
    - Graceful shutdown

    Example:
        spawner = TaskSpawner()

        # Simple spawn
        task = await spawner.spawn(my_coro(), name="task1")

        # With restart policy
        task = await spawner.spawn(
            daemon_coro(),
            name="daemon",
            restart_policy=RestartPolicy.ALWAYS,
            max_restarts=10,
        )

        # With concurrency limit
        async with spawner.limited(max_concurrent=5) as limited:
            await asyncio.gather(*[
                limited.spawn(work(i)) for i in range(100)
            ])

        # Shutdown
        await spawner.shutdown()
    """

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        default_restart_policy: RestartPolicy = RestartPolicy.NEVER,
    ):
        """Initialize spawner.

        Args:
            max_concurrent: Global concurrency limit
            default_restart_policy: Default restart policy
        """
        self._max_concurrent = max_concurrent
        self._default_restart_policy = default_restart_policy
        self._tasks: Dict[str, SpawnedTask] = {}
        self._groups: Dict[str, TaskGroup] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()
        self._counter = 0
        self._shutting_down = False

        if max_concurrent:
            self._semaphore = asyncio.Semaphore(max_concurrent)

        self._stats = {
            "tasks_spawned": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_restarts": 0,
        }

    async def spawn(
        self,
        coro: Coroutine[Any, Any, T],
        name: Optional[str] = None,
        restart_policy: Optional[RestartPolicy] = None,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        group: Optional[str] = None,
        on_complete: Optional[Callable[[SpawnedTask], None]] = None,
        on_error: Optional[Callable[[SpawnedTask, Exception], None]] = None,
    ) -> SpawnedTask[T]:
        """Spawn a supervised task.

        Args:
            coro: Coroutine to run
            name: Task name (auto-generated if None)
            restart_policy: Restart behavior on failure
            max_restarts: Maximum restart attempts
            restart_delay: Delay between restarts
            group: Task group name
            on_complete: Callback when task completes
            on_error: Callback on task error

        Returns:
            SpawnedTask handle
        """
        if self._shutting_down:
            raise RuntimeError("Spawner is shutting down")

        # Generate name if needed
        if name is None:
            async with self._lock:
                self._counter += 1
                name = f"task_{self._counter}"

        restart_policy = restart_policy or self._default_restart_policy

        # Create spawned task
        spawned = SpawnedTask(name=name)
        spawned.state = TaskState.PENDING

        # Create wrapper for supervision
        async def supervised_wrapper() -> T:
            spawned.started_at = time.time()
            spawned.state = TaskState.RUNNING

            restarts = 0
            last_error: Optional[Exception] = None

            while True:
                try:
                    # Apply semaphore if set
                    if self._semaphore:
                        async with self._semaphore:
                            result = await coro
                    else:
                        result = await coro

                    # Success
                    spawned.state = TaskState.COMPLETED
                    spawned.completed_at = time.time()
                    spawned._result = result
                    self._stats["tasks_completed"] += 1

                    if on_complete:
                        try:
                            on_complete(spawned)
                        except Exception as e:
                            logger.warning(f"on_complete callback error: {e}")

                    return result

                except asyncio.CancelledError:
                    spawned.state = TaskState.CANCELLED
                    spawned.completed_at = time.time()
                    self._stats["tasks_cancelled"] += 1
                    raise

                except Exception as e:
                    last_error = e
                    spawned.last_error = e
                    spawned.restarts = restarts
                    self._stats["tasks_failed"] += 1

                    logger.error(f"Task {name} failed: {e}")

                    if on_error:
                        try:
                            on_error(spawned, e)
                        except Exception as cb_err:
                            logger.warning(f"on_error callback error: {cb_err}")

                    # Check restart policy
                    should_restart = (
                        restart_policy in (RestartPolicy.ON_FAILURE, RestartPolicy.ALWAYS)
                        and restarts < max_restarts
                        and not self._shutting_down
                    )

                    if should_restart:
                        restarts += 1
                        spawned.restarts = restarts
                        spawned.state = TaskState.RESTARTING
                        self._stats["total_restarts"] += 1

                        logger.info(f"Restarting task {name} (attempt {restarts}/{max_restarts})")
                        await asyncio.sleep(restart_delay)
                        continue
                    else:
                        spawned.state = TaskState.FAILED
                        spawned.completed_at = time.time()
                        raise

        # Create and store task
        task = asyncio.create_task(supervised_wrapper())
        spawned._task = task
        self._stats["tasks_spawned"] += 1

        async with self._lock:
            self._tasks[name] = spawned

        logger.debug(f"Spawned task: {name}")
        return spawned

    def spawn_sync(
        self,
        coro: Coroutine[Any, Any, T],
        **kwargs: Any,
    ) -> SpawnedTask[T]:
        """Spawn a task from synchronous code.

        Creates a task that will run in the event loop.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.spawn(coro, **kwargs))

    def create_group(self, name: str) -> TaskGroup:
        """Create a task group.

        Args:
            name: Group name

        Returns:
            TaskGroup instance
        """
        group = TaskGroup(name, self)
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> Optional[TaskGroup]:
        """Get a task group by name."""
        return self._groups.get(name)

    @asynccontextmanager
    async def limited(
        self,
        max_concurrent: int,
    ) -> AsyncGenerator["LimitedSpawner", None]:
        """Context manager for concurrency-limited spawning.

        Args:
            max_concurrent: Maximum concurrent tasks

        Yields:
            LimitedSpawner with concurrency limit

        Example:
            async with spawner.limited(10) as limited:
                tasks = [limited.spawn(work(i)) for i in range(100)]
                await asyncio.gather(*[t.wait() for t in tasks])
        """
        limited = LimitedSpawner(self, max_concurrent)
        try:
            yield limited
        finally:
            await limited.wait_all()

    async def get_task(self, name: str) -> Optional[SpawnedTask]:
        """Get a task by name."""
        async with self._lock:
            return self._tasks.get(name)

    async def cancel_task(self, name: str) -> bool:
        """Cancel a task by name."""
        task = await self.get_task(name)
        if task:
            return task.cancel()
        return False

    async def cancel_all(self) -> int:
        """Cancel all running tasks."""
        async with self._lock:
            tasks = list(self._tasks.values())

        cancelled = 0
        for task in tasks:
            if task.cancel():
                cancelled += 1

        return cancelled

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown the spawner.

        Args:
            timeout: Maximum wait time for tasks to complete
        """
        self._shutting_down = True
        logger.info("Task spawner shutting down...")

        # Cancel all running tasks
        cancelled = await self.cancel_all()
        logger.info(f"Cancelled {cancelled} tasks")

        # Wait for tasks to complete
        async with self._lock:
            tasks = [t._task for t in self._tasks.values() if t._task]

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown timed out after {timeout}s")

        logger.info("Task spawner shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get spawner statistics."""
        running = sum(1 for t in self._tasks.values() if t.is_running)
        completed = sum(1 for t in self._tasks.values() if t.is_done)

        return {
            **self._stats,
            "tasks_running": running,
            "tasks_tracked": len(self._tasks),
            "groups": len(self._groups),
            "shutting_down": self._shutting_down,
        }

    def list_tasks(
        self,
        state: Optional[TaskState] = None,
        group: Optional[str] = None,
    ) -> List[SpawnedTask]:
        """List tasks matching criteria."""
        tasks = list(self._tasks.values())

        if state:
            tasks = [t for t in tasks if t.state == state]

        if group:
            task_group = self._groups.get(group)
            if task_group:
                group_names = {t.name for t in task_group.tasks}
                tasks = [t for t in tasks if t.name in group_names]

        return tasks


class LimitedSpawner:
    """Spawner with concurrency limit."""

    def __init__(self, parent: TaskSpawner, max_concurrent: int):
        self._parent = parent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: List[SpawnedTask] = []

    async def spawn(
        self,
        coro: Coroutine[Any, Any, T],
        **kwargs: Any,
    ) -> SpawnedTask[T]:
        """Spawn a task with concurrency limit."""
        async def limited_coro() -> T:
            async with self._semaphore:
                return await coro

        task = await self._parent.spawn(limited_coro(), **kwargs)
        self._tasks.append(task)
        return task

    async def wait_all(self) -> None:
        """Wait for all spawned tasks."""
        aws = [t.wait() for t in self._tasks if t._task]
        if aws:
            await asyncio.gather(*aws, return_exceptions=True)


# =============================================================================
# Global Instance
# =============================================================================

_spawner: Optional[TaskSpawner] = None
_spawner_lock = threading.Lock()


def get_spawner() -> TaskSpawner:
    """Get the global task spawner singleton."""
    global _spawner
    if _spawner is None:
        with _spawner_lock:
            if _spawner is None:
                _spawner = TaskSpawner()
    return _spawner


def reset_spawner() -> None:
    """Reset the global spawner (for testing)."""
    global _spawner
    with _spawner_lock:
        _spawner = None


# Convenience function
async def spawn(
    coro: Coroutine[Any, Any, T],
    **kwargs: Any,
) -> SpawnedTask[T]:
    """Spawn a task using the global spawner."""
    return await get_spawner().spawn(coro, **kwargs)
