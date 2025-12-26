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

Migration Guide (December 2025):
    When updating existing code with bare exception handling, prefer these decorators:

    # BEFORE (bare exception handling):
    def fetch_data():
        for attempt in range(3):
            try:
                return do_fetch()
            except IOError as e:
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed after 3 attempts")

    # AFTER (using retry decorator):
    @retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(IOError,))
    def fetch_data():
        return do_fetch()

    Benefits:
    - Consistent retry logic across codebase
    - Circuit breaker integration via circuit_breaker_key parameter
    - Automatic logging and metrics
    - Jitter support to prevent thundering herd
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

__all__ = [
    # Constants
    "EMERGENCY_HALT_FILE",
    "EmergencyHaltError",
    "ErrorAggregator",
    "FatalError",
    "RetryPolicy",
    # Retry policy classes
    "RetryStrategy",
    "RetryableError",
    # Exception types (re-exported from app.errors)
    "RingRiftError",
    # Emergency halt functions
    "check_emergency_halt",
    "clear_emergency_halt",
    # Retry decorators
    "retry",
    "retry_async",
    # Safe execution
    "safe_execute",
    "safe_execute_async",
    "set_emergency_halt",
    "with_emergency_halt_check",
    "with_emergency_halt_check_async",
    "with_retry_policy",
    "with_retry_policy_async",
]

logger = logging.getLogger(__name__)

# Type vars for decorators
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Any])

# Use centralized path constants
from app.utils.paths import DATA_DIR, ensure_parent_dir

# Emergency halt file location (same as unified_ai_loop.py)
EMERGENCY_HALT_FILE = DATA_DIR / "coordination" / "EMERGENCY_HALT"


# ============================================================================
# Custom Exception Hierarchy - Imported from canonical source
# ============================================================================

# Import error classes from the unified errors module
from app.errors import (
    EmergencyHaltError,
    FatalError,  # Alias for NonRetryableError
    RetryableError,
    RingRiftError,
)

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
    ensure_parent_dir(EMERGENCY_HALT_FILE)
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
    exceptions: Sequence[type[Exception]] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
    reraise: bool = True,
    jitter: bool = False,
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
        jitter: If True, add random jitter to prevent thundering herd (default False)

    Returns:
        Decorated function

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

        @retry(max_attempts=5, exceptions=(SSHError, TimeoutError), jitter=True)
        def run_remote_command(host, cmd):
            ...
    """
    import random

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
                        # Apply jitter if enabled (±25% of delay)
                        actual_delay = current_delay
                        if jitter:
                            actual_delay = current_delay * (0.75 + random.random() * 0.5)

                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                                f"Retrying in {actual_delay:.1f}s..."
                            )

                        time.sleep(actual_delay)
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
    exceptions: Sequence[type[Exception]] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
    reraise: bool = True,
    jitter: bool = False,
    circuit_breaker_key: str | None = None,
) -> Callable[[AF], AF]:
    """Async version of retry decorator.

    Args:
        Same as retry(), plus:
        circuit_breaker_key: Optional key for circuit breaker integration.
            When provided, checks circuit breaker state before execution
            and records success/failure. (December 2025)

    Returns:
        Decorated async function

    Example:
        @retry_async(max_attempts=3, jitter=True)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

        @retry_async(max_attempts=3, circuit_breaker_key="external_api")
        async def call_external_api():
            # Circuit breaker will prevent calls when circuit is open
            ...
    """
    import random

    # Circuit breaker integration (December 2025)
    breaker = None
    if circuit_breaker_key:
        try:
            from app.distributed.circuit_breaker import get_operation_breaker
            breaker = get_operation_breaker(circuit_breaker_key)
        except ImportError:
            pass

    def decorator(func: AF) -> AF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check circuit breaker before attempting
            if breaker and not breaker.can_execute():
                raise RetryableError(
                    f"Circuit breaker '{circuit_breaker_key}' is open, refusing execution"
                )

            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    # Record success with circuit breaker
                    if breaker:
                        breaker.record_success()
                    return result
                except exceptions as e:
                    last_exception = e

                    # Record failure with circuit breaker
                    if breaker:
                        breaker.record_failure()

                    # Don't retry FatalError
                    if isinstance(e, FatalError):
                        raise

                    if attempt < max_attempts:
                        # Apply jitter if enabled (±25% of delay)
                        actual_delay = current_delay
                        if jitter:
                            actual_delay = current_delay * (0.75 + random.random() * 0.5)

                        if on_retry:
                            on_retry(e, attempt)
                        else:
                            logger.warning(
                                f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                                f"Retrying in {actual_delay:.1f}s..."
                            )

                        await asyncio.sleep(actual_delay)
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
# Unified Retry Policy (December 2025)
# ============================================================================

from dataclasses import dataclass
from enum import Enum


class RetryStrategy(Enum):
    """Retry delay strategies."""
    LINEAR = "linear"  # Constant delay
    EXPONENTIAL = "exponential"  # Delay doubles each attempt
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with random jitter


@dataclass
class RetryPolicy:
    """Unified retry policy configuration.

    Use this class to define consistent retry behavior across the codebase.
    Prefer using predefined policies (DEFAULT, AGGRESSIVE, CONSERVATIVE) or
    create custom policies for specific use cases.

    Usage:
        from app.core.error_handler import RetryPolicy, DEFAULT_RETRY_POLICY

        # Use default policy
        @retry(**DEFAULT_RETRY_POLICY.to_retry_kwargs())
        def my_function():
            ...

        # Create custom policy
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
        )

        @retry(**policy.to_retry_kwargs())
        def my_function():
            ...

        # Or use the policy directly for manual retry loops
        for attempt in range(policy.max_attempts):
            try:
                return do_work()
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < policy.max_attempts - 1:
                    time.sleep(policy.get_delay(attempt))
                else:
                    raise
    """
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1  # For EXPONENTIAL_JITTER: random factor (0-1)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed).

        Args:
            attempt: Current attempt number (0 = first retry)

        Returns:
            Delay in seconds before next attempt
        """
        if self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.multiplier ** attempt)
        elif self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = self.initial_delay * (self.multiplier ** attempt)
            jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
            delay = base_delay * (1 + jitter)
        else:
            delay = self.initial_delay

        return min(delay, self.max_delay)

    def to_retry_kwargs(self) -> dict:
        """Convert policy to kwargs for the @retry decorator.

        Returns:
            Dict that can be unpacked as **kwargs to retry()
        """
        return {
            "max_attempts": self.max_attempts,
            "delay": self.initial_delay,
            "backoff": self.multiplier if self.strategy != RetryStrategy.LINEAR else 1.0,
            "max_delay": self.max_delay,
        }

    @classmethod
    def from_config(cls, config: dict) -> RetryPolicy:
        """Create policy from configuration dict.

        Args:
            config: Dict with policy parameters

        Returns:
            RetryPolicy instance
        """
        strategy = config.get("strategy", "exponential")
        if isinstance(strategy, str):
            strategy = RetryStrategy(strategy)

        return cls(
            strategy=strategy,
            max_attempts=config.get("max_attempts", 3),
            initial_delay=config.get("initial_delay", 1.0),
            multiplier=config.get("multiplier", 2.0),
            max_delay=config.get("max_delay", 60.0),
            jitter_factor=config.get("jitter_factor", 0.1),
        )


# =============================================================================
# Predefined Retry Policies (December 2025 - uses centralized RetryDefaults)
# =============================================================================

# Import centralized retry defaults
try:
    from app.config.coordination_defaults import RetryDefaults
    _retry_defaults = RetryDefaults()
except ImportError:
    _retry_defaults = None

# Default policy - standard operations
DEFAULT_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL,
    max_attempts=_retry_defaults.MAX_RETRIES if _retry_defaults else 3,
    initial_delay=_retry_defaults.BASE_DELAY if _retry_defaults else 1.0,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=_retry_defaults.MAX_DELAY if _retry_defaults else 60.0,
)

# For critical operations that need more attempts
AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    max_attempts=_retry_defaults.AGGRESSIVE_MAX_RETRIES if _retry_defaults else 5,
    initial_delay=_retry_defaults.AGGRESSIVE_BASE_DELAY if _retry_defaults else 0.5,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=30.0,
    jitter_factor=0.2,
)

# For less critical operations with longer delays
CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL,
    max_attempts=_retry_defaults.MAX_RETRIES if _retry_defaults else 3,
    initial_delay=5.0,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=300.0,  # 5 minutes
)

# For quick network operations
FAST_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.LINEAR,
    max_attempts=_retry_defaults.FAST_MAX_RETRIES if _retry_defaults else 2,
    initial_delay=_retry_defaults.FAST_BASE_DELAY if _retry_defaults else 0.5,
    multiplier=1.0,
    max_delay=_retry_defaults.FAST_MAX_DELAY if _retry_defaults else 5.0,
)

# For sync/file operations that may need circuit breaker integration
SYNC_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    max_attempts=_retry_defaults.SYNC_MAX_RETRIES if _retry_defaults else 3,
    initial_delay=_retry_defaults.SYNC_BASE_DELAY if _retry_defaults else 2.0,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=_retry_defaults.SYNC_MAX_DELAY if _retry_defaults else 30.0,
    jitter_factor=_retry_defaults.JITTER_FACTOR if _retry_defaults else 0.1,
)

# HTTP-specific policy with jitter to prevent thundering herd
HTTP_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    max_attempts=_retry_defaults.MAX_RETRIES if _retry_defaults else 3,
    initial_delay=_retry_defaults.BASE_DELAY if _retry_defaults else 1.0,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=_retry_defaults.MAX_DELAY if _retry_defaults else 60.0,
    jitter_factor=_retry_defaults.JITTER_FACTOR if _retry_defaults else 0.1,
)

# SSH operations - longer delays between retries
SSH_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    max_attempts=_retry_defaults.MAX_RETRIES if _retry_defaults else 3,
    initial_delay=2.0,
    multiplier=_retry_defaults.BACKOFF_MULTIPLIER if _retry_defaults else 2.0,
    max_delay=_retry_defaults.MAX_DELAY if _retry_defaults else 60.0,
    jitter_factor=0.15,
)

# Database operations - quick retries for lock contention
DATABASE_RETRY_POLICY = RetryPolicy(
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    max_attempts=_retry_defaults.AGGRESSIVE_MAX_RETRIES if _retry_defaults else 5,
    initial_delay=0.1,  # Fast retry for SQLite lock contention
    multiplier=2.0,
    max_delay=5.0,
    jitter_factor=0.2,
)


def with_retry_policy(policy: RetryPolicy) -> Callable[[F], F]:
    """Decorator factory that applies a RetryPolicy.

    Args:
        policy: RetryPolicy to use

    Returns:
        Decorator function

    Example:
        @with_retry_policy(AGGRESSIVE_RETRY_POLICY)
        def my_function():
            ...
    """
    return retry(**policy.to_retry_kwargs())


def with_retry_policy_async(policy: RetryPolicy) -> Callable[[AF], AF]:
    """Async version of with_retry_policy."""
    return retry_async(**policy.to_retry_kwargs())


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

    def add(self, error: Exception, context: dict | None = None) -> None:
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

    def raise_if_any(self, error_class: type[Exception] = RingRiftError) -> None:
        """Raise an exception if any errors were collected."""
        if self.errors:
            raise error_class(self.summary())
