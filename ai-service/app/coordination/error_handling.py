"""Unified error handling for coordination layer (January 2026).

This module provides decorators and utilities for consistent error handling
across the coordination layer. It reduces boilerplate and ensures consistent
logging, event emission, and retry behavior.

Usage:
    from app.coordination.error_handling import (
        handle_coordination_error,
        CoordinationErrorConfig,
    )

    # Basic usage - logs and suppresses errors
    @handle_coordination_error()
    async def my_async_handler():
        # Do work that might fail
        pass

    # With event emission and retry
    @handle_coordination_error(
        emit_event=True,
        event_type="MY_HANDLER_ERROR",
        max_retries=3,
        retry_delay=1.0,
    )
    async def my_important_handler():
        pass

    # Sync function support
    @handle_coordination_error(sync=True)
    def my_sync_function():
        pass
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic decorator typing
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CoordinationErrorConfig:
    """Configuration for error handling behavior.

    Attributes:
        component_name: Name of the component for logging (auto-detected if None)
        emit_event: Whether to emit an error event via event_router
        event_type: Event type to emit (uses default if None)
        suppress_errors: Whether to suppress errors (True) or re-raise (False)
        max_retries: Maximum retry attempts (0 = no retries)
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Backoff multiplier for retry delays
        track_errors: Whether to track error counts in a global counter
        log_level: Logging level for errors (default: ERROR)
        include_traceback: Whether to include full traceback in logs
    """

    component_name: str | None = None
    emit_event: bool = False
    event_type: str | None = None
    suppress_errors: bool = True
    max_retries: int = 0
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    track_errors: bool = True
    log_level: int = logging.ERROR
    include_traceback: bool = False


# Global error tracking (for observability)
_error_counts: dict[str, int] = {}


def get_error_counts() -> dict[str, int]:
    """Get current error counts per component."""
    return dict(_error_counts)


def reset_error_counts() -> None:
    """Reset error counts (for testing)."""
    _error_counts.clear()


def _emit_error_event(
    event_type: str,
    component: str,
    error: Exception,
    context: dict[str, Any],
) -> None:
    """Emit an error event via the event router."""
    try:
        from app.coordination.event_router import emit_event

        emit_event(event_type, {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            **context,
        })
    except ImportError:
        logger.debug(f"[ErrorHandling] event_router not available, skipping event: {event_type}")
    except Exception as e:
        logger.debug(f"[ErrorHandling] Failed to emit error event: {e}")


def handle_coordination_error(
    component_name: str | None = None,
    emit_event: bool = False,
    event_type: str | None = None,
    suppress_errors: bool = True,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    track_errors: bool = True,
    log_level: int = logging.ERROR,
    include_traceback: bool = False,
    sync: bool = False,
) -> Callable[[F], F]:
    """Decorator for unified error handling in coordination components.

    This decorator wraps a function with consistent error handling:
    - Logs errors with component context
    - Optionally emits error events
    - Optionally retries with exponential backoff
    - Tracks error counts for observability

    Args:
        component_name: Component name for logging (uses function name if None)
        emit_event: Whether to emit an error event
        event_type: Event type to emit (defaults to COORDINATION_ERROR)
        suppress_errors: Whether to suppress errors (True) or re-raise (False)
        max_retries: Maximum retry attempts (0 = no retries)
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Backoff multiplier for retry delays
        track_errors: Whether to track error counts globally
        log_level: Logging level for errors
        include_traceback: Whether to include full traceback
        sync: Set True if decorating a synchronous function

    Returns:
        Decorated function with error handling

    Examples:
        # Basic usage
        @handle_coordination_error()
        async def my_handler():
            pass

        # With retry
        @handle_coordination_error(max_retries=3, retry_delay=1.0)
        async def flaky_operation():
            pass

        # With event emission
        @handle_coordination_error(emit_event=True, event_type="SYNC_ERROR")
        async def sync_data():
            pass

        # Sync function
        @handle_coordination_error(sync=True)
        def process_locally():
            pass
    """

    def decorator(func: F) -> F:
        component = component_name or func.__name__
        error_event = event_type or "COORDINATION_ERROR"

        if sync:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return _execute_with_handling(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    component=component,
                    emit_event_flag=emit_event,
                    error_event=error_event,
                    suppress_errors=suppress_errors,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff=retry_backoff,
                    track_errors=track_errors,
                    log_level=log_level,
                    include_traceback=include_traceback,
                    is_async=False,
                )
            return sync_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _execute_with_handling_async(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    component=component,
                    emit_event_flag=emit_event,
                    error_event=error_event,
                    suppress_errors=suppress_errors,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff=retry_backoff,
                    track_errors=track_errors,
                    log_level=log_level,
                    include_traceback=include_traceback,
                )
            return async_wrapper  # type: ignore[return-value]

    return decorator


def _execute_with_handling(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    component: str,
    emit_event_flag: bool,
    error_event: str,
    suppress_errors: bool,
    max_retries: int,
    retry_delay: float,
    retry_backoff: float,
    track_errors: bool,
    log_level: int,
    include_traceback: bool,
    is_async: bool,
) -> Any:
    """Execute sync function with error handling."""
    attempt = 0
    delay = retry_delay

    while True:
        attempt += 1
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(
                e=e,
                component=component,
                attempt=attempt,
                max_retries=max_retries,
                emit_event_flag=emit_event_flag,
                error_event=error_event,
                track_errors=track_errors,
                log_level=log_level,
                include_traceback=include_traceback,
            )

            if attempt <= max_retries:
                logger.info(f"[{component}] Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
                delay *= retry_backoff
                continue

            if not suppress_errors:
                raise

            return None


async def _execute_with_handling_async(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    component: str,
    emit_event_flag: bool,
    error_event: str,
    suppress_errors: bool,
    max_retries: int,
    retry_delay: float,
    retry_backoff: float,
    track_errors: bool,
    log_level: int,
    include_traceback: bool,
) -> Any:
    """Execute async function with error handling."""
    attempt = 0
    delay = retry_delay

    while True:
        attempt += 1
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(
                e=e,
                component=component,
                attempt=attempt,
                max_retries=max_retries,
                emit_event_flag=emit_event_flag,
                error_event=error_event,
                track_errors=track_errors,
                log_level=log_level,
                include_traceback=include_traceback,
            )

            if attempt <= max_retries:
                logger.info(f"[{component}] Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(delay)
                delay *= retry_backoff
                continue

            if not suppress_errors:
                raise

            return None


def _handle_exception(
    e: Exception,
    component: str,
    attempt: int,
    max_retries: int,
    emit_event_flag: bool,
    error_event: str,
    track_errors: bool,
    log_level: int,
    include_traceback: bool,
) -> None:
    """Handle an exception with logging, tracking, and optional event emission."""
    # Track error count
    if track_errors:
        _error_counts[component] = _error_counts.get(component, 0) + 1

    # Build log message
    retry_info = f" (attempt {attempt}/{max_retries + 1})" if max_retries > 0 else ""
    log_msg = f"[{component}] Error{retry_info}: {type(e).__name__}: {e}"

    if include_traceback:
        import traceback
        log_msg += f"\n{traceback.format_exc()}"

    logger.log(log_level, log_msg)

    # Emit error event if configured
    if emit_event_flag:
        _emit_error_event(
            event_type=error_event,
            component=component,
            error=e,
            context={
                "attempt": attempt,
                "max_retries": max_retries,
            },
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "handle_coordination_error",
    "CoordinationErrorConfig",
    "get_error_counts",
    "reset_error_counts",
]
