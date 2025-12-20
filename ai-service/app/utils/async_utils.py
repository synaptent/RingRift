"""Async utilities for safe task management.

This module provides utilities for managing asyncio tasks safely,
preventing issues like task garbage collection and unhandled exceptions.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Global set to hold references to background tasks
# This prevents garbage collection from cancelling them
_background_tasks: set[asyncio.Task[Any]] = set()

T = TypeVar("T")


def fire_and_forget(coro: Coroutine[Any, Any, T], name: str | None = None) -> asyncio.Task[T]:
    """Schedule a coroutine to run in the background without awaiting it.

    This is a safe alternative to `asyncio.create_task()` for fire-and-forget
    scenarios. The task is stored in a global set to prevent garbage collection,
    and exceptions are logged instead of being silently dropped.

    Args:
        coro: The coroutine to run in the background.
        name: Optional name for the task (for debugging).

    Returns:
        The created task (can be ignored for fire-and-forget usage).

    Example:
        # Instead of:
        asyncio.create_task(some_async_function())  # Risky: may be GC'd

        # Use:
        fire_and_forget(some_async_function())  # Safe: tracked and exceptions logged
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_task_done_callback)
    return task


def _task_done_callback(task: asyncio.Task[Any]) -> None:
    """Callback to handle task completion and cleanup."""
    _background_tasks.discard(task)
    try:
        # Check if task raised an exception
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Background task %s raised exception: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )
    except asyncio.CancelledError:
        # Task was cancelled - this is expected in shutdown scenarios
        logger.debug("Background task %s was cancelled", task.get_name())


def get_pending_tasks() -> int:
    """Get the number of pending background tasks."""
    return len(_background_tasks)


async def wait_all_background_tasks(timeout: float | None = None) -> None:
    """Wait for all background tasks to complete.

    Useful for graceful shutdown scenarios.

    Args:
        timeout: Maximum time to wait in seconds. None means wait indefinitely.
    """
    if not _background_tasks:
        return

    logger.info("Waiting for %d background tasks to complete...", len(_background_tasks))
    _done, pending = await asyncio.wait(
        _background_tasks,
        timeout=timeout,
        return_when=asyncio.ALL_COMPLETED,
    )

    if pending:
        logger.warning("Timed out waiting for %d tasks, cancelling...", len(pending))
        for task in pending:
            task.cancel()


def cancel_all_background_tasks() -> None:
    """Cancel all pending background tasks.

    Useful for immediate shutdown scenarios.
    """
    for task in _background_tasks:
        task.cancel()
    logger.info("Cancelled %d background tasks", len(_background_tasks))
