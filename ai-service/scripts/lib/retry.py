"""Retry utilities for resilient script execution.

Provides decorators and utilities for automatic retry with exponential backoff.

Usage:
    from scripts.lib.retry import retry, retry_on_exception, RetryConfig

    # Simple retry decorator
    @retry(max_attempts=3, delay=1.0)
    def flaky_operation():
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
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExceptionTypes = Union[type[Exception], tuple[type[Exception], ...]]


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential: Whether to use exponential backoff
        jitter: Add random jitter to delays (0-1 range, 0.1 = 10% jitter)
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential: bool = True
    jitter: float = 0.1

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

        return max(0, delay)

    def attempts(self) -> Generator["RetryAttempt", None, None]:
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
        """Wait for the configured delay before next attempt."""
        if self.delay > 0:
            time.sleep(self.delay)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: ExceptionTypes = Exception,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
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

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
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
) -> Callable[[Callable[..., T]], Callable[..., T]]:
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


def with_timeout(
    timeout: float,
    default: T | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
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
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeout:
                    logger.warning(f"{func.__name__} timed out after {timeout}s")
                    return default

        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: ExceptionTypes = Exception,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Async version of retry decorator.

    Args:
        max_attempts: Maximum number of attempts
        delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff
        exceptions: Exception types to catch and retry

    Example:
        @retry_async(max_attempts=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                ...
    """
    import asyncio

    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        max_delay=max_delay,
        exponential=exponential,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in config.attempts():
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt.is_last:
                        raise

                    logger.debug(
                        f"{func.__name__} attempt {attempt.number} failed: {e}"
                    )

                    if attempt.delay > 0:
                        await asyncio.sleep(attempt.delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result or exception")

        return wrapper
    return decorator
