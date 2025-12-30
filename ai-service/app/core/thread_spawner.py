"""Thread Spawning Abstraction for RingRift AI Service.

Provides unified thread spawning with:
- Supervised thread execution
- Automatic restart policies
- Thread grouping and cancellation
- Progress tracking and health checks

This mirrors TaskSpawner but for synchronous threaded code.

Usage:
    from app.core.thread_spawner import ThreadSpawner, RestartPolicy

    spawner = ThreadSpawner()

    # Spawn a supervised thread
    thread = spawner.spawn(
        target=my_function,
        name="my_thread",
        restart_policy=RestartPolicy.ON_FAILURE,
    )

    # Check status
    status = spawner.get_thread_status("my_thread")

    # Graceful shutdown
    spawner.shutdown()
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
)

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)

__all__ = [
    "RestartPolicy",
    "SpawnedThread",
    "ThreadGroup",
    "ThreadSpawner",
    "ThreadState",
    "get_thread_spawner",
    "spawn_thread",
]


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RestartPolicy(Enum):
    """Restart policy for failed threads."""
    NEVER = "never"           # Never restart
    ON_FAILURE = "on_failure" # Restart only on exception
    ALWAYS = "always"         # Always restart (for daemons)


class ThreadState(Enum):
    """State of a spawned thread."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    RESTARTING = "restarting"


@dataclass
class SpawnedThread:
    """Represents a spawned and managed thread.

    Attributes:
        name: Thread name
        state: Current state
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        restarts: Number of restarts
        last_error: Last error if any
    """
    name: str
    target: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    state: ThreadState = ThreadState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    restarts: int = 0
    last_error: Exception | None = None
    restart_policy: RestartPolicy = RestartPolicy.NEVER
    max_restarts: int = 3
    restart_delay: float = 1.0
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _result: Any = field(default=None, repr=False)

    @property
    def duration(self) -> float | None:
        """Get thread duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_done(self) -> bool:
        """Check if thread is done."""
        return self.state in (ThreadState.COMPLETED, ThreadState.FAILED, ThreadState.STOPPED)

    @property
    def is_running(self) -> bool:
        """Check if thread is running."""
        return self.state == ThreadState.RUNNING

    @property
    def is_alive(self) -> bool:
        """Check if underlying thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> bool:
        """Signal thread to stop."""
        self._stop_event.set()
        return True

    def should_stop(self) -> bool:
        """Check if stop was requested (use in target function)."""
        return self._stop_event.is_set()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for thread to complete."""
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "restarts": self.restarts,
            "is_alive": self.is_alive,
            "error": str(self.last_error) if self.last_error else None,
        }


# =============================================================================
# Thread Group
# =============================================================================

class ThreadGroup:
    """Group of related threads.

    Allows managing multiple threads as a unit.

    Example:
        group = ThreadGroup("workers", spawner)

        for i in range(10):
            group.spawn(target=worker, args=(i,), name=f"worker_{i}")

        group.join_all()  # Wait for all
        group.stop_all()  # Or stop all
    """

    def __init__(self, name: str, spawner: ThreadSpawner):
        self.name = name
        self._spawner = spawner
        self._threads: dict[str, SpawnedThread] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        target: Callable[..., Any],
        name: str | None = None,
        **kwargs: Any,
    ) -> SpawnedThread:
        """Spawn a thread in this group."""
        thread = self._spawner.spawn(target, name=name, group=self.name, **kwargs)
        with self._lock:
            self._threads[thread.name] = thread
        return thread

    def join_all(self, timeout: float | None = None) -> bool:
        """Wait for all threads to complete."""
        with self._lock:
            threads = list(self._threads.values())

        all_done = True
        for thread in threads:
            if not thread.join(timeout=timeout):
                all_done = False

        return all_done

    def stop_all(self) -> int:
        """Stop all threads in group."""
        with self._lock:
            threads = list(self._threads.values())

        stopped = 0
        for thread in threads:
            if thread.stop():
                stopped += 1

        return stopped

    @property
    def threads(self) -> list[SpawnedThread]:
        """Get all threads in group."""
        return list(self._threads.values())

    @property
    def running_count(self) -> int:
        """Number of running threads."""
        return sum(1 for t in self._threads.values() if t.is_running)

    @property
    def completed_count(self) -> int:
        """Number of completed threads."""
        return sum(1 for t in self._threads.values() if t.is_done)


# =============================================================================
# Thread Spawner
# =============================================================================

class ThreadSpawner:
    """Unified thread spawner with supervision.

    Manages thread lifecycle including:
    - Spawning with automatic tracking
    - Restart policies for failed threads
    - Thread grouping
    - Graceful shutdown

    Example:
        spawner = ThreadSpawner()

        # Simple spawn
        thread = spawner.spawn(target=my_func, name="thread1")

        # With restart policy
        thread = spawner.spawn(
            target=daemon_func,
            name="daemon",
            restart_policy=RestartPolicy.ALWAYS,
            max_restarts=10,
        )

        # Shutdown
        spawner.shutdown()
    """

    def __init__(
        self,
        default_restart_policy: RestartPolicy = RestartPolicy.NEVER,
    ):
        """Initialize spawner.

        Args:
            default_restart_policy: Default restart policy
        """
        self._default_restart_policy = default_restart_policy
        self._threads: dict[str, SpawnedThread] = {}
        self._groups: dict[str, ThreadGroup] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self._shutting_down = False

        self._stats = {
            "threads_spawned": 0,
            "threads_completed": 0,
            "threads_failed": 0,
            "threads_stopped": 0,
            "total_restarts": 0,
        }

    def spawn(
        self,
        target: Callable[..., Any],
        name: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        restart_policy: RestartPolicy | None = None,
        max_restarts: int = 3,
        restart_delay: float = 1.0,
        group: str | None = None,
        daemon: bool = True,
        on_complete: Callable[[SpawnedThread], None] | None = None,
        on_error: Callable[[SpawnedThread, Exception], None] | None = None,
    ) -> SpawnedThread:
        """Spawn a supervised thread.

        Args:
            target: Function to run in thread
            name: Thread name (auto-generated if None)
            args: Positional arguments for target
            kwargs: Keyword arguments for target
            restart_policy: Restart behavior on failure
            max_restarts: Maximum restart attempts
            restart_delay: Delay between restarts
            group: Thread group name
            daemon: Whether thread is daemon
            on_complete: Callback when thread completes
            on_error: Callback on thread error

        Returns:
            SpawnedThread handle
        """
        if self._shutting_down:
            raise RuntimeError("Spawner is shutting down")

        kwargs = kwargs or {}

        # Generate name if needed
        with self._lock:
            if name is None:
                self._counter += 1
                name = f"thread_{self._counter}"

        restart_policy = restart_policy or self._default_restart_policy

        # Create spawned thread
        spawned = SpawnedThread(
            name=name,
            target=target,
            args=args,
            kwargs=kwargs,
            restart_policy=restart_policy,
            max_restarts=max_restarts,
            restart_delay=restart_delay,
        )

        # Create wrapper for supervision
        def supervised_wrapper():
            spawned.started_at = time.time()
            spawned.state = ThreadState.RUNNING

            restarts = 0

            while True:
                try:
                    result = target(*args, **kwargs)

                    # Success
                    spawned.state = ThreadState.COMPLETED
                    spawned.completed_at = time.time()
                    spawned._result = result

                    with self._lock:
                        self._stats["threads_completed"] += 1

                    if on_complete:
                        try:
                            on_complete(spawned)
                        except Exception as e:
                            logger.warning(f"on_complete callback error: {e}")

                    return

                except Exception as e:
                    # Check if this was a stop request
                    if spawned.should_stop():
                        spawned.state = ThreadState.STOPPED
                        spawned.completed_at = time.time()
                        with self._lock:
                            self._stats["threads_stopped"] += 1
                        return

                    spawned.last_error = e
                    spawned.restarts = restarts

                    with self._lock:
                        self._stats["threads_failed"] += 1

                    logger.error(
                        f"Thread {name} failed: {e}\n"
                        f"{traceback.format_exc()}"
                    )

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
                        and not spawned.should_stop()
                    )

                    if should_restart:
                        restarts += 1
                        spawned.restarts = restarts
                        spawned.state = ThreadState.RESTARTING

                        with self._lock:
                            self._stats["total_restarts"] += 1

                        logger.info(f"Restarting thread {name} (attempt {restarts}/{max_restarts})")
                        time.sleep(restart_delay)
                        continue
                    else:
                        spawned.state = ThreadState.FAILED
                        spawned.completed_at = time.time()
                        return

        # Create and start thread
        thread = threading.Thread(target=supervised_wrapper, name=name, daemon=daemon)
        spawned._thread = thread
        thread.start()

        with self._lock:
            self._stats["threads_spawned"] += 1
            self._threads[name] = spawned

        logger.debug(f"Spawned thread: {name}")
        return spawned

    def create_group(self, name: str) -> ThreadGroup:
        """Create a thread group.

        Args:
            name: Group name

        Returns:
            ThreadGroup instance
        """
        group = ThreadGroup(name, self)
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> ThreadGroup | None:
        """Get a thread group by name."""
        return self._groups.get(name)

    def get_thread(self, name: str) -> SpawnedThread | None:
        """Get a thread by name."""
        with self._lock:
            return self._threads.get(name)

    def stop_thread(self, name: str, timeout: float = 5.0) -> bool:
        """Stop a thread by name."""
        thread = self.get_thread(name)
        if thread:
            thread.stop()
            return thread.join(timeout=timeout)
        return False

    def stop_all(self, timeout: float = 5.0) -> int:
        """Stop all running threads."""
        with self._lock:
            threads = list(self._threads.values())

        stopped = 0
        for thread in threads:
            if thread.is_running:
                thread.stop()
                stopped += 1

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=timeout)

        return stopped

    def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown the spawner.

        Args:
            timeout: Maximum wait time for threads to complete
        """
        self._shutting_down = True
        logger.info("Thread spawner shutting down...")

        # Stop all running threads
        stopped = self.stop_all(timeout=timeout / 2)
        logger.info(f"Stopped {stopped} threads")

        # Wait for threads to complete
        with self._lock:
            threads = list(self._threads.values())

        for thread in threads:
            remaining = max(0, timeout - (time.time() - thread.started_at if thread.started_at else 0))
            thread.join(timeout=remaining / len(threads) if threads else timeout)

        logger.info("Thread spawner shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get spawner statistics."""
        with self._lock:
            running = sum(1 for t in self._threads.values() if t.is_running)
            completed = sum(1 for t in self._threads.values() if t.is_done)

            return {
                **self._stats,
                "threads_running": running,
                "threads_completed": completed,
                "threads_tracked": len(self._threads),
                "groups": len(self._groups),
                "shutting_down": self._shutting_down,
            }

    def list_threads(
        self,
        state: ThreadState | None = None,
        group: str | None = None,
    ) -> list[SpawnedThread]:
        """List threads matching criteria."""
        with self._lock:
            threads = list(self._threads.values())

        if state:
            threads = [t for t in threads if t.state == state]

        if group:
            thread_group = self._groups.get(group)
            if thread_group:
                group_names = {t.name for t in thread_group.threads}
                threads = [t for t in threads if t.name in group_names]

        return threads

    def health_check(self) -> HealthCheckResult:
        """Get health status of all threads.

        Returns:
            HealthCheckResult with healthy status, message, and details.
        """
        with self._lock:
            threads = list(self._threads.values())

        healthy = True
        unhealthy_threads = []

        for thread in threads:
            # Check for threads that should be running but aren't
            if (thread.restart_policy == RestartPolicy.ALWAYS and not thread.is_alive
                    and thread.state not in (ThreadState.STOPPED, ThreadState.COMPLETED)):
                healthy = False
                unhealthy_threads.append(thread.name)

            # Check for threads that failed without restart
            if thread.state == ThreadState.FAILED:
                healthy = False
                unhealthy_threads.append(thread.name)

        running_count = sum(1 for t in threads if t.is_running)
        failed_count = sum(1 for t in threads if t.state == ThreadState.FAILED)
        completed_count = sum(1 for t in threads if t.state == ThreadState.COMPLETED)

        # Determine status and message
        if not healthy:
            status = CoordinatorStatus.DEGRADED
            message = f"{len(unhealthy_threads)} threads unhealthy: {', '.join(unhealthy_threads[:3])}"
            if len(unhealthy_threads) > 3:
                message += f" (+{len(unhealthy_threads) - 3} more)"
        elif running_count > 0:
            status = CoordinatorStatus.RUNNING
            message = f"{running_count} threads running, {completed_count} completed"
        else:
            status = CoordinatorStatus.READY
            message = f"No threads running, {completed_count} completed"

        return HealthCheckResult(
            healthy=healthy,
            status=status,
            message=message,
            details={
                "unhealthy_threads": unhealthy_threads,
                "running": running_count,
                "failed": failed_count,
                "completed": completed_count,
            },
        )


# =============================================================================
# Global Instance
# =============================================================================

_spawner: ThreadSpawner | None = None
_spawner_lock = threading.Lock()


def get_thread_spawner() -> ThreadSpawner:
    """Get the global thread spawner singleton."""
    global _spawner
    if _spawner is None:
        with _spawner_lock:
            if _spawner is None:
                _spawner = ThreadSpawner()
    return _spawner


def reset_thread_spawner() -> None:
    """Reset the global spawner (for testing)."""
    global _spawner
    with _spawner_lock:
        if _spawner:
            _spawner.shutdown(timeout=5.0)
        _spawner = None


# Convenience function
def spawn_thread(
    target: Callable[..., Any],
    **kwargs: Any,
) -> SpawnedThread:
    """Spawn a thread using the global spawner."""
    return get_thread_spawner().spawn(target, **kwargs)
