"""Async Context Patterns for RingRift AI Service.

Provides reusable async context managers and patterns:
- Timeout contexts with cleanup
- Retry contexts with backoff
- Resource acquisition with fallback
- Async semaphore pools
- Cancellation-safe contexts

Usage:
    from app.core.async_context import timeout_context, retry_context

    async with timeout_context(5.0) as ctx:
        result = await slow_operation()
        if ctx.timed_out:
            handle_timeout()

    async with retry_context(max_attempts=3) as ctx:
        result = await flaky_operation()
        print(f"Succeeded after {ctx.attempts} attempts")
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    TypeVar,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RateLimiter",
    "ResourcePool",
    "RetryContext",
    "TimeoutContext",
    "cancellation_scope",
    "fire_and_forget",
    "gather_with_limit",
    "rate_limiter",
    "retry_context",
    "safe_create_task",
    "timeout_context",
]

T = TypeVar("T")


# =============================================================================
# Timeout Context
# =============================================================================

@dataclass
class TimeoutContext:
    """Context for timeout operations with state tracking."""
    timeout: float
    timed_out: bool = False
    elapsed: float = 0.0
    cancelled: bool = False
    start_time: float = field(default_factory=time.time)

    @property
    def remaining(self) -> float:
        """Get remaining time before timeout."""
        return max(0, self.timeout - self.elapsed)


@asynccontextmanager
async def timeout_context(
    timeout: float,
    suppress_timeout: bool = False,
) -> AsyncGenerator[TimeoutContext]:
    """Async context manager with timeout tracking.

    Unlike asyncio.timeout, this provides context about whether
    a timeout occurred without raising an exception.

    Args:
        timeout: Timeout in seconds
        suppress_timeout: If True, don't raise TimeoutError

    Yields:
        TimeoutContext with state information

    Example:
        async with timeout_context(5.0) as ctx:
            try:
                await slow_operation()
            except asyncio.CancelledError:
                if ctx.timed_out:
                    logger.warning("Operation timed out")
                raise
    """
    ctx = TimeoutContext(timeout=timeout)

    try:
        async with asyncio.timeout(timeout):
            yield ctx
    except asyncio.TimeoutError:
        ctx.timed_out = True
        ctx.elapsed = timeout
        if not suppress_timeout:
            raise
    except asyncio.CancelledError:
        ctx.cancelled = True
        ctx.elapsed = time.time() - ctx.start_time
        raise
    finally:
        if not ctx.timed_out and not ctx.cancelled:
            ctx.elapsed = time.time() - ctx.start_time


# =============================================================================
# Retry Context
# =============================================================================

@dataclass
class RetryContext:
    """Context for retry operations with state tracking."""
    max_attempts: int
    attempts: int = 0
    last_error: Exception | None = None
    succeeded: bool = False
    total_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)


@asynccontextmanager
async def retry_context(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    jitter: bool = True,
    retry_on: tuple = (Exception,),
) -> AsyncGenerator[RetryContext]:
    """Async context manager with automatic retry on failure.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff
        jitter: Add random jitter to delays
        retry_on: Exception types to retry on

    Yields:
        RetryContext with state information

    Example:
        async with retry_context(max_attempts=3) as ctx:
            result = await flaky_api_call()
        print(f"Succeeded after {ctx.attempts} attempts")
    """
    ctx = RetryContext(max_attempts=max_attempts)

    for attempt in range(1, max_attempts + 1):
        ctx.attempts = attempt

        try:
            yield ctx
            ctx.succeeded = True
            return
        except retry_on as e:
            ctx.last_error = e
            ctx.errors.append(e)

            if attempt == max_attempts:
                raise

            # Calculate delay
            if exponential:
                delay = base_delay * (2 ** (attempt - 1))
            else:
                delay = base_delay

            delay = min(delay, max_delay)

            if jitter:
                delay *= (0.5 + random.random())

            ctx.total_delay += delay
            logger.debug(f"Retry {attempt}/{max_attempts}, waiting {delay:.2f}s")
            await asyncio.sleep(delay)


# =============================================================================
# Resource Pool
# =============================================================================

class ResourcePool(Generic[T]):
    """Async resource pool with automatic management.

    Manages a pool of resources (connections, workers, etc.)
    with automatic acquisition and release.

    Example:
        pool = ResourcePool(
            factory=create_connection,
            max_size=10,
            min_size=2,
        )
        await pool.initialize()

        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

        await pool.close()
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 300.0,
        destroy: Callable[[T], Awaitable[None]] | None = None,
        validate: Callable[[T], Awaitable[bool]] | None = None,
    ):
        """Initialize the pool.

        Args:
            factory: Async function to create new resources
            max_size: Maximum pool size
            min_size: Minimum pool size to maintain
            max_idle_time: Max seconds a resource can be idle
            destroy: Async function to destroy a resource
            validate: Async function to validate a resource
        """
        self._factory = factory
        self._destroy = destroy
        self._validate = validate
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time

        self._available: asyncio.Queue[tuple[T, float]] = asyncio.Queue()
        self._in_use: set[T] = set()
        self._size = 0
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Current pool size."""
        return self._size

    @property
    def available(self) -> int:
        """Number of available resources."""
        return self._available.qsize()

    @property
    def in_use(self) -> int:
        """Number of resources in use."""
        return len(self._in_use)

    async def initialize(self) -> None:
        """Initialize pool with minimum resources."""
        for _ in range(self._min_size):
            resource = await self._create()
            await self._available.put((resource, time.time()))

    async def _create(self) -> T:
        """Create a new resource."""
        resource = await self._factory()
        self._size += 1
        return resource

    async def _destroy_resource(self, resource: T) -> None:
        """Destroy a resource."""
        self._size -= 1
        if self._destroy:
            try:
                await self._destroy(resource)
            except Exception as e:
                logger.warning(f"Error destroying resource: {e}")

    @asynccontextmanager
    async def acquire(
        self,
        timeout: float | None = None,
    ) -> AsyncGenerator[T]:
        """Acquire a resource from the pool.

        Args:
            timeout: Timeout for acquiring resource

        Yields:
            Resource from pool

        Example:
            async with pool.acquire() as resource:
                await resource.do_something()
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        resource = await self._get_resource(timeout)
        self._in_use.add(resource)

        try:
            yield resource
        finally:
            self._in_use.discard(resource)
            if not self._closed:
                await self._available.put((resource, time.time()))

    async def _get_resource(self, timeout: float | None) -> T:
        """Get a resource, creating if needed."""
        while True:
            # Try to get from pool
            try:
                resource, _created_at = self._available.get_nowait()

                # Validate if we have a validator
                if self._validate:
                    try:
                        if not await self._validate(resource):
                            await self._destroy_resource(resource)
                            continue
                    except Exception:
                        await self._destroy_resource(resource)
                        continue

                return resource

            except asyncio.QueueEmpty:
                pass

            # Create new if under limit
            async with self._lock:
                if self._size < self._max_size:
                    return await self._create()

            # Wait for available resource
            try:
                resource, _ = await asyncio.wait_for(
                    self._available.get(),
                    timeout=timeout,
                )
                return resource
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Timeout acquiring resource from pool")

    async def close(self) -> None:
        """Close the pool and all resources."""
        self._closed = True

        # Destroy available resources
        while not self._available.empty():
            try:
                resource, _ = self._available.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break

        # Wait for in-use resources (with timeout)
        for _ in range(10):
            if not self._in_use:
                break
            await asyncio.sleep(0.1)

    def stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "size": self._size,
            "available": self.available,
            "in_use": self.in_use,
            "max_size": self._max_size,
            "min_size": self._min_size,
            "closed": self._closed,
        }


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Async rate limiter using token bucket algorithm.

    Example:
        limiter = RateLimiter(rate=10.0, burst=20)

        async with limiter:
            await api_call()  # Rate limited
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
    ):
        """Initialize rate limiter.

        Args:
            rate: Tokens per second to add
            burst: Maximum tokens (bucket size)
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            waited = 0.0

            while True:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self._rate

                await asyncio.sleep(wait_time)
                waited += wait_time

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        self._tokens = min(
            self._burst,
            self._tokens + elapsed * self._rate,
        )

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@asynccontextmanager
async def rate_limiter(
    rate: float,
    burst: int = 1,
) -> AsyncGenerator[RateLimiter]:
    """Context manager for rate limiting.

    Args:
        rate: Operations per second
        burst: Maximum burst size

    Example:
        async with rate_limiter(10.0, burst=5) as limiter:
            for item in items:
                async with limiter:
                    await process(item)
    """
    yield RateLimiter(rate=rate, burst=burst)


# =============================================================================
# Cancellation Scope
# =============================================================================

@dataclass
class CancellationScope:
    """Scope for managing cancellation."""
    cancelled: bool = False
    shield: bool = False
    _tasks: list[asyncio.Task] = field(default_factory=list)

    def cancel(self) -> None:
        """Cancel all tasks in scope."""
        self.cancelled = True
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def spawn(self, coro: Awaitable[T]) -> asyncio.Task[T]:
        """Spawn a task in this scope."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task


@asynccontextmanager
async def cancellation_scope(
    shield: bool = False,
) -> AsyncGenerator[CancellationScope]:
    """Context manager for cancellation-safe operations.

    Args:
        shield: Shield from external cancellation

    Yields:
        CancellationScope for managing tasks

    Example:
        async with cancellation_scope() as scope:
            task1 = scope.spawn(operation1())
            task2 = scope.spawn(operation2())
            # Both tasks cancelled on scope exit or error
    """
    scope = CancellationScope(shield=shield)

    try:
        if shield:
            # Shield from external cancellation
            yield scope
        else:
            yield scope
    finally:
        # Cancel all spawned tasks
        scope.cancel()

        # Wait for tasks to complete
        if scope._tasks:
            await asyncio.gather(*scope._tasks, return_exceptions=True)


# =============================================================================
# Gather with Concurrency Limit
# =============================================================================

async def gather_with_limit(
    *coros: Awaitable[T],
    limit: int = 10,
    return_exceptions: bool = False,
) -> list[T | Exception]:
    """Like asyncio.gather but with concurrency limit.

    Args:
        coros: Coroutines to run
        limit: Maximum concurrent tasks
        return_exceptions: Return exceptions instead of raising

    Returns:
        List of results in same order as input

    Example:
        results = await gather_with_limit(
            *[fetch(url) for url in urls],
            limit=5,
        )
    """
    semaphore = asyncio.Semaphore(limit)
    results: list[Any] = [None] * len(coros)

    async def run_with_semaphore(index: int, coro: Awaitable[T]) -> None:
        async with semaphore:
            try:
                results[index] = await coro
            except Exception as e:
                if return_exceptions:
                    results[index] = e
                else:
                    raise

    await asyncio.gather(
        *[run_with_semaphore(i, c) for i, c in enumerate(coros)],
        return_exceptions=return_exceptions,
    )

    return results


# =============================================================================
# Async iteration helpers
# =============================================================================

async def async_batch(
    items: AsyncIterator[T],
    batch_size: int,
) -> AsyncIterator[list[T]]:
    """Batch async iterator into chunks.

    Args:
        items: Async iterator to batch
        batch_size: Size of each batch

    Yields:
        Batches of items

    Example:
        async for batch in async_batch(stream, 100):
            await process_batch(batch)
    """
    batch: list[T] = []

    async for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


async def async_timeout_iter(
    items: AsyncIterator[T],
    timeout: float,
) -> AsyncIterator[T]:
    """Add timeout to async iterator.

    Args:
        items: Async iterator
        timeout: Timeout per item in seconds

    Yields:
        Items from iterator

    Raises:
        asyncio.TimeoutError: If next item takes too long
    """
    while True:
        try:
            item = await asyncio.wait_for(
                items.__anext__(),
                timeout=timeout,
            )
            yield item
        except StopAsyncIteration:
            break


# =============================================================================
# Safe Task Creation (December 2025 - async/sync bridge hardening)
# =============================================================================


def _default_task_error_handler(task: asyncio.Task) -> None:
    """Default error handler for fire-and-forget tasks.

    Logs exceptions from tasks that would otherwise be silently dropped.
    This prevents the "Task exception was never retrieved" warning.
    """
    try:
        exc = task.exception()
        if exc is not None:
            task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
            logger.error(
                f"[AsyncTask] Fire-and-forget task '{task_name}' failed: {exc}",
                exc_info=exc,
            )
    except asyncio.CancelledError:
        pass  # Task was cancelled, not an error
    except asyncio.InvalidStateError:
        pass  # Task not done yet (shouldn't happen in done callback)


def safe_create_task(
    coro: Awaitable[T],
    *,
    name: str | None = None,
    error_callback: Callable[[asyncio.Task], None] | None = None,
    suppress_errors: bool = False,
) -> asyncio.Task[T]:
    """Create an asyncio task with automatic error handling.

    This is a drop-in replacement for asyncio.create_task() that adds
    error callbacks to prevent silent failures in fire-and-forget patterns.

    Args:
        coro: The coroutine to run as a task
        name: Optional name for the task (for debugging)
        error_callback: Custom error handler (default: logs to error level)
        suppress_errors: If True, don't log errors (just consume them)

    Returns:
        The created task

    Example:
        # Instead of:
        asyncio.create_task(send_notification())  # Errors silently dropped!

        # Use:
        safe_create_task(send_notification())  # Errors logged

        # With custom handler:
        safe_create_task(
            send_notification(),
            error_callback=lambda t: alert_admin(t.exception())
        )
    """
    task = asyncio.create_task(coro, name=name)

    if not suppress_errors:
        callback = error_callback or _default_task_error_handler
        task.add_done_callback(callback)

    return task


def fire_and_forget(
    coro: Awaitable[Any],
    *,
    name: str | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> asyncio.Task:
    """Fire-and-forget a coroutine with explicit error handling.

    This makes the intent clear that we don't await the result,
    while ensuring errors are not silently dropped.

    Args:
        coro: The coroutine to run
        name: Optional name for debugging
        on_error: Optional callback when error occurs (receives the exception)

    Returns:
        The created task (usually ignored by caller)

    Example:
        # Fire and forget with logging
        fire_and_forget(emit_event(data))

        # With error callback
        fire_and_forget(
            send_notification(),
            on_error=lambda e: metrics.increment("notification_errors")
        )
    """
    def error_handler(task: asyncio.Task) -> None:
        try:
            exc = task.exception()
            if exc is not None:
                if on_error:
                    on_error(exc)
                else:
                    task_name = name or (task.get_name() if hasattr(task, 'get_name') else "unnamed")
                    logger.error(f"[FireAndForget] Task '{task_name}' failed: {exc}")
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass

    return safe_create_task(coro, name=name, error_callback=error_handler)
