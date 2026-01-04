"""Centralized retry utilities for resilient operations.

Provides decorators and utilities for automatic retry with exponential backoff.
This module consolidates retry patterns across the codebase.

Usage:
    from app.utils.retry import retry, retry_async, RetryConfig

    # Simple retry decorator
    @retry(max_attempts=3, delay=1.0)
    def flaky_operation():
        ...

    # Async retry
    @retry_async(max_attempts=3, delay=1.0)
    async def async_flaky_operation():
        ...

    # Retry with specific exceptions
    @retry_on_exception(ConnectionError, TimeoutError, max_attempts=5)
    def network_call():
        ...

    # Manual retry loop
    config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0)
    for attempt in config.attempts():
        try:
            result = do_something()
            break
        except Exception as e:
            if not attempt.should_retry:
                raise
            attempt.wait()

December 2025: Consolidated from scripts/lib/retry.py into app/utils/ for use
across all coordination, sync, and training modules.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field
from typing import Any, TypeVar

# ParamSpec requires Python 3.10+, use typing_extensions for 3.9 compatibility
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

logger = logging.getLogger(__name__)

__all__ = [
    "RetryConfig",
    "RetryAttempt",
    "retry",
    "retry_async",
    "retry_on_exception",
    "retry_on_exception_async",
    "with_timeout",
    # Common retry configs
    "RETRY_QUICK",
    "RETRY_STANDARD",
    "RETRY_PATIENT",
    "RETRY_SSH",
    "RETRY_HTTP",
]

T = TypeVar("T")
P = ParamSpec("P")
ExceptionTypes = type[Exception] | tuple[type[Exception], ...]


# Common retry configurations
RETRY_QUICK = lambda: RetryConfig(max_attempts=2, base_delay=0.5, max_delay=2.0)
RETRY_STANDARD = lambda: RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0)
RETRY_PATIENT = lambda: RetryConfig(max_attempts=5, base_delay=2.0, max_delay=60.0)
RETRY_SSH = lambda: RetryConfig(max_attempts=3, base_delay=5.0, max_delay=30.0, jitter=0.2)
RETRY_HTTP = lambda: RetryConfig(max_attempts=4, base_delay=1.0, max_delay=15.0)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Whether to use exponential backoff
        jitter: Add random jitter to delays (0-1 range, 0.1 = 10% jitter)
        retryable_exceptions: Exception types to catch and retry (default: Exception)
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential: bool = True
    jitter: float = 0.1
    retryable_exceptions: ExceptionTypes = field(default=Exception)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        if attempt == 0:
            return 0.0

        if self.exponential:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    def attempts(self) -> Generator[RetryAttempt, None, None]:
        """Generate retry attempts for use in a for loop.

        Example:
            config = RetryConfig(max_attempts=3)
            for attempt in config.attempts():
                try:
                    result = do_something()
                    break
                except Exception:
                    if not attempt.should_retry:
                        raise
                    attempt.wait()
        """
        for i in range(self.max_attempts):
            yield RetryAttempt(
                number=i + 1,
                max_attempts=self.max_attempts,
                delay=self.get_delay(i),
            )


@dataclass
class RetryAttempt:
    """Represents a single retry attempt."""

    number: int
    max_attempts: int
    delay: float

    @property
    def is_first(self) -> bool:
        """Check if this is the first attempt."""
        return self.number == 1

    @property
    def is_last(self) -> bool:
        """Check if this is the last attempt."""
        return self.number >= self.max_attempts

    @property
    def should_retry(self) -> bool:
        """Check if we should retry after this attempt fails."""
        return not self.is_last

    def wait(self) -> None:
        """Wait for the configured delay before next attempt (blocking)."""
        if self.delay > 0:
            time.sleep(self.delay)

    async def wait_async(self) -> None:
        """Wait for the configured delay before next attempt (async)."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: ExceptionTypes = Exception,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff
        exceptions: Exception types to catch and retry
        on_retry: Callback called on each retry (exception, attempt_number)

    Returns:
        Decorated function that retries on failure

    Example:
        @retry(max_attempts=3, delay=1.0)
        def fetch_data():
            return requests.get(url).json()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        max_delay=max_delay,
        exponential=exponential,
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in config.attempts():
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt.is_last:
                        logger.warning(
                            f"{func.__name__} failed after {attempt.number} attempts: {e}"
                        )
                        raise

                    logger.debug(
                        f"{func.__name__} attempt {attempt.number} failed: {e}, "
                        f"retrying in {attempt.delay:.1f}s"
                    )

                    if on_retry:
                        on_retry(e, attempt.number)

                    attempt.wait()

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result or exception")

        return wrapper

    return decorator


def retry_on_exception(
    *exceptions: type[Exception],
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retry on specific exception types.

    Args:
        *exceptions: Exception types to catch and retry
        max_attempts: Maximum number of attempts
        delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff

    Example:
        @retry_on_exception(ConnectionError, TimeoutError, max_attempts=5)
        def connect_to_server():
            ...
    """
    if not exceptions:
        exceptions = (Exception,)

    return retry(
        max_attempts=max_attempts,
        delay=delay,
        max_delay=max_delay,
        exponential=exponential,
        exceptions=exceptions,
    )


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: ExceptionTypes = Exception,
    on_retry: Callable[[Exception, int], Awaitable[None] | None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Async version of retry decorator.

    Args:
        max_attempts: Maximum number of attempts
        delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff
        exceptions: Exception types to catch and retry
        on_retry: Callback called on each retry (exception, attempt_number).
                  Can be sync or async.

    Example:
        @retry_async(max_attempts=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        max_delay=max_delay,
        exponential=exponential,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in config.attempts():
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt.is_last:
                        logger.warning(
                            f"{func.__name__} failed after {attempt.number} attempts: {e}"
                        )
                        raise

                    logger.debug(
                        f"{func.__name__} attempt {attempt.number} failed: {e}, "
                        f"retrying in {attempt.delay:.1f}s"
                    )

                    if on_retry:
                        result = on_retry(e, attempt.number)
                        if asyncio.iscoroutine(result):
                            await result

                    await attempt.wait_async()

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result or exception")

        return wrapper

    return decorator


def retry_on_exception_async(
    *exceptions: type[Exception],
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Async decorator for retry on specific exception types.

    Args:
        *exceptions: Exception types to catch and retry
        max_attempts: Maximum number of attempts
        delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff

    Example:
        @retry_on_exception_async(ConnectionError, TimeoutError, max_attempts=5)
        async def connect_to_server():
            ...
    """
    if not exceptions:
        exceptions = (Exception,)

    return retry_async(
        max_attempts=max_attempts,
        delay=delay,
        max_delay=max_delay,
        exponential=exponential,
        exceptions=exceptions,
    )


def with_timeout(
    timeout: float,
    default: T | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Decorator to add timeout to a function (using threading).

    Note: This uses threading and may not interrupt all operations.
    For subprocess calls, use the timeout parameter directly.

    Args:
        timeout: Maximum execution time in seconds
        default: Value to return on timeout (None if not specified)

    Example:
        @with_timeout(5.0)
        def slow_operation():
            ...
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeout:
                    logger.warning(f"{func.__name__} timed out after {timeout}s")
                    return default

        return wrapper

    return decorator
