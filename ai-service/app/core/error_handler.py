"""Unified error handling and retry logic for the RingRift AI service.

This module provides:
- Custom exception hierarchy
- Retry decorators (sync and async)
- Emergency halt integration
- Error recovery patterns

Usage:
    from app.core.error_handler import retry, retry_async, with_emergency_halt_check

    @retry(max_attempts=3, delay=1.0)
    def flaky_operation():
        ...

    @retry_async(max_attempts=3, backoff=2.0)
    async def async_flaky_operation():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Type vars for decorators
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Any])

# Emergency halt file location (same as unified_ai_loop.py)
AI_SERVICE_ROOT = Path(__file__).parent.parent.parent
EMERGENCY_HALT_FILE = AI_SERVICE_ROOT / "data" / "coordination" / "EMERGENCY_HALT"


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class RingRiftError(Exception):
    """Base exception for all RingRift-specific errors."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}


class RetryableError(RingRiftError):
    """Error that can be retried (network issues, transient failures)."""
    pass


class FatalError(RingRiftError):
    """Error that should not be retried (invalid config, data corruption)."""
    pass


class EmergencyHaltError(RingRiftError):
    """Raised when emergency halt is detected."""
    pass


class SSHError(RetryableError):
    """SSH connection or command execution error."""
    pass


class DatabaseError(RingRiftError):
    """Database access error."""
    pass


class ConfigurationError(FatalError):
    """Configuration error that cannot be recovered."""
    pass


# ============================================================================
# Emergency Halt Functions
# ============================================================================

def check_emergency_halt() -> bool:
    """Check if emergency halt flag is set.

    Returns:
        True if emergency halt is active, False otherwise
    """
    return EMERGENCY_HALT_FILE.exists()


def set_emergency_halt(reason: str = "Manual halt") -> None:
    """Set emergency halt flag to stop all loops.

    Args:
        reason: Reason for the halt (stored in the file)
    """
    EMERGENCY_HALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMERGENCY_HALT_FILE.write_text(f"{reason}\nSet at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.warning(f"Emergency halt set: {reason}")


def clear_emergency_halt() -> bool:
    """Clear emergency halt flag to allow operations to resume.

    Returns:
        True if flag was cleared, False if it wasn't set
    """
    if EMERGENCY_HALT_FILE.exists():
        EMERGENCY_HALT_FILE.unlink()
        logger.info("Emergency halt cleared")
        return True
    return False


def with_emergency_halt_check(func: F) -> F:
    """Decorator that checks emergency halt before each call.

    Raises EmergencyHaltError if halt is active.

    Usage:
        @with_emergency_halt_check
        def long_running_operation():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if check_emergency_halt():
            raise EmergencyHaltError(
                f"Emergency halt active, refusing to run {func.__name__}"
            )
        return func(*args, **kwargs)
    return wrapper  # type: ignore


def with_emergency_halt_check_async(func: AF) -> AF:
    """Async version of with_emergency_halt_check."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if check_emergency_halt():
            raise EmergencyHaltError(
                f"Emergency halt active, refusing to run {func.__name__}"
            )
        return await func(*args, **kwargs)
    return wrapper  # type: ignore


# ============================================================================
# Retry Decorators
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
) -> Callable[[F], F]:
    """Decorator for retrying a function on failure.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry on
        on_retry: Optional callback called on each retry with (exception, attempt)
        reraise: If True, reraise the last exception. If False, return None.

    Returns:
        Decorated function

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        @retry(max_attempts=5, exceptions=(SSHError, TimeoutError))
        def run_remote_command(host, cmd):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry FatalError
                    if isinstance(e, FatalError):
                        raise

                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )

                        time.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            if reraise and last_exception:
                raise last_exception
            return None

        return wrapper  # type: ignore

    return decorator


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    reraise: bool = True,
) -> Callable[[AF], AF]:
    """Async version of retry decorator.

    Args:
        Same as retry()

    Returns:
        Decorated async function

    Example:
        @retry_async(max_attempts=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """
    def decorator(func: AF) -> AF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry FatalError
                    if isinstance(e, FatalError):
                        raise

                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )

                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            if reraise and last_exception:
                raise last_exception
            return None

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# Error Recovery Helpers
# ============================================================================

def safe_execute(
    func: Callable[..., Any],
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Execute a function safely, returning default on any error.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default value

    Example:
        result = safe_execute(parse_config, "config.yaml", default={})
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"safe_execute: {func.__name__} failed: {e}")
        return default


async def safe_execute_async(
    func: Callable[..., Any],
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Async version of safe_execute."""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"safe_execute_async: {func.__name__} failed: {e}")
        return default


class ErrorAggregator:
    """Collect multiple errors during batch operations.

    Usage:
        errors = ErrorAggregator("batch processing")
        for item in items:
            try:
                process(item)
            except Exception as e:
                errors.add(e, context={"item": item})

        if errors.has_errors:
            logger.warning(errors.summary())
        # Or raise if needed:
        errors.raise_if_any()
    """

    def __init__(self, operation: str):
        self.operation = operation
        self.errors: list[tuple[Exception, dict]] = []

    def add(self, error: Exception, context: Optional[dict] = None) -> None:
        """Add an error to the collection."""
        self.errors.append((error, context or {}))

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    @property
    def count(self) -> int:
        """Number of errors collected."""
        return len(self.errors)

    def summary(self) -> str:
        """Get a summary of all errors."""
        if not self.errors:
            return f"{self.operation}: No errors"

        lines = [f"{self.operation}: {len(self.errors)} error(s)"]
        for i, (err, ctx) in enumerate(self.errors[:10], 1):  # Limit to first 10
            ctx_str = f" [{ctx}]" if ctx else ""
            lines.append(f"  {i}. {type(err).__name__}: {err}{ctx_str}")

        if len(self.errors) > 10:
            lines.append(f"  ... and {len(self.errors) - 10} more")

        return "\n".join(lines)

    def raise_if_any(self, error_class: Type[Exception] = RingRiftError) -> None:
        """Raise an exception if any errors were collected."""
        if self.errors:
            raise error_class(self.summary())
