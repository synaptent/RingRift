"""Debug and profiling utilities for development and troubleshooting.

This module provides utilities for debugging, profiling, and troubleshooting
code during development. These are designed to be low-overhead when disabled
and provide useful context when enabled.

Usage:
    from app.utils.debug_utils import timed, log_call, debug_context

    # Time a function
    @timed
    def slow_function():
        ...

    # Log function calls with arguments
    @log_call
    def important_function(arg1, arg2):
        ...

    # Add debug context
    with debug_context("processing batch", batch_id=123):
        ...
"""

from __future__ import annotations

import functools
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, List, TypeVar

from app.utils.env_config import get_bool

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Check if debug mode is enabled
DEBUG_ENABLED = get_bool("RINGRIFT_DEBUG", False)
PROFILE_ENABLED = get_bool("RINGRIFT_PROFILE", False)


def timed(func: F = None, *, threshold_ms: float = 100.0, log_level: int = logging.DEBUG) -> F:
    """Decorator to log execution time of functions.

    Args:
        func: Function to decorate
        threshold_ms: Only log if execution exceeds this threshold (in ms)
        log_level: Logging level to use

    Example:
        @timed
        def slow_function():
            ...

        @timed(threshold_ms=500)
        def very_slow_function():
            ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms >= threshold_ms:
                    logger.log(
                        log_level,
                        f"{fn.__module__}.{fn.__name__} took {elapsed_ms:.1f}ms"
                    )
        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore


def timed_async(func: F = None, *, threshold_ms: float = 100.0, log_level: int = logging.DEBUG) -> F:
    """Async version of @timed decorator."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms >= threshold_ms:
                    logger.log(
                        log_level,
                        f"{fn.__module__}.{fn.__name__} took {elapsed_ms:.1f}ms"
                    )
        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore


def log_call(func: F = None, *, include_result: bool = False, log_level: int = logging.DEBUG) -> F:
    """Decorator to log function calls with arguments.

    Only logs when RINGRIFT_DEBUG=true.

    Args:
        func: Function to decorate
        include_result: Also log the return value
        log_level: Logging level to use

    Example:
        @log_call
        def process_game(game_id, player):
            ...
        # Logs: "process_game(game_id='abc', player=1)"
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if DEBUG_ENABLED:
                # Format arguments
                arg_strs = [repr(a) for a in args]
                kwarg_strs = [f"{k}={v!r}" for k, v in kwargs.items()]
                all_args = ", ".join(arg_strs + kwarg_strs)
                logger.log(log_level, f"{fn.__name__}({all_args})")

            result = fn(*args, **kwargs)

            if DEBUG_ENABLED and include_result:
                logger.log(log_level, f"{fn.__name__} returned {result!r}")

            return result
        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore


@contextmanager
def debug_context(description: str, **context):
    """Context manager that logs entry/exit and any exceptions.

    Only logs when RINGRIFT_DEBUG=true.

    Args:
        description: Short description of the context
        **context: Key-value pairs to include in logs

    Example:
        with debug_context("processing batch", batch_id=123, size=50):
            process_batch(...)
    """
    if not DEBUG_ENABLED:
        yield
        return

    context_str = ", ".join(f"{k}={v!r}" for k, v in context.items())
    full_desc = f"{description} ({context_str})" if context_str else description

    logger.debug(f"Entering: {full_desc}")
    start = time.perf_counter()

    try:
        yield
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Failed after {elapsed_ms:.1f}ms: {full_desc} - {type(e).__name__}: {e}")
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Completed in {elapsed_ms:.1f}ms: {full_desc}")


@dataclass
class Timer:
    """Simple timer for measuring elapsed time.

    Example:
        timer = Timer()
        # ... do work ...
        print(f"Took {timer.elapsed_ms:.1f}ms")

        # Or with checkpoints
        timer = Timer()
        step1()
        timer.checkpoint("step1")
        step2()
        timer.checkpoint("step2")
        print(timer.summary())
    """
    _start: float = field(default_factory=time.perf_counter)
    _checkpoints: List[tuple] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    def checkpoint(self, name: str) -> float:
        """Record a checkpoint and return elapsed time since start."""
        elapsed = self.elapsed_ms
        self._checkpoints.append((name, elapsed))
        return elapsed

    def reset(self) -> None:
        """Reset the timer."""
        self._start = time.perf_counter()
        self._checkpoints.clear()

    def summary(self) -> str:
        """Get a summary of all checkpoints."""
        if not self._checkpoints:
            return f"Total: {self.elapsed_ms:.1f}ms"

        lines = []
        prev = 0.0
        for name, elapsed in self._checkpoints:
            delta = elapsed - prev
            lines.append(f"  {name}: {delta:.1f}ms (total: {elapsed:.1f}ms)")
            prev = elapsed

        final_delta = self.elapsed_ms - prev
        if final_delta > 0.1:
            lines.append(f"  (remaining): {final_delta:.1f}ms")

        lines.insert(0, f"Timing breakdown (total: {self.elapsed_ms:.1f}ms):")
        return "\n".join(lines)


def get_call_stack(skip: int = 1, limit: int = 10) -> List[str]:
    """Get a simplified call stack for debugging.

    Args:
        skip: Number of frames to skip from the top
        limit: Maximum number of frames to include

    Returns:
        List of "file:line in function" strings
    """
    frames = []
    for frame_info in traceback.extract_stack()[:-skip][-limit:]:
        frames.append(f"{frame_info.filename}:{frame_info.lineno} in {frame_info.name}")
    return frames


def format_exception_chain(exc: BaseException) -> str:
    """Format an exception with its full cause chain.

    Args:
        exc: Exception to format

    Returns:
        Formatted string with exception chain
    """
    lines = []
    current = exc
    while current is not None:
        lines.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return " <- ".join(lines)


def memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes, or -1 if unavailable
    """
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS, kilobytes on Linux
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes to MB
        else:
            return usage.ru_maxrss / 1024  # KB to MB
    except ImportError:
        return -1.0


def dump_state(obj: Any, name: str = "state", max_depth: int = 3) -> str:
    """Create a debug dump of an object's state.

    Args:
        obj: Object to dump
        name: Name to use in output
        max_depth: Maximum nesting depth

    Returns:
        Formatted string representation
    """
    def _format(o: Any, depth: int) -> str:
        if depth > max_depth:
            return "..."

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = [f"{k!r}: {_format(v, depth + 1)}" for k, v in list(o.items())[:20]]
            if len(o) > 20:
                items.append(f"... ({len(o) - 20} more)")
            return "{\n" + ",\n".join(f"  {i}" for i in items) + "\n}"

        if isinstance(o, (list, tuple)):
            if not o:
                return "[]" if isinstance(o, list) else "()"
            items = [_format(v, depth + 1) for v in list(o)[:20]]
            if len(o) > 20:
                items.append(f"... ({len(o) - 20} more)")
            brackets = "[]" if isinstance(o, list) else "()"
            return brackets[0] + ", ".join(items) + brackets[1]

        if hasattr(o, "__dict__"):
            attrs = {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            if attrs:
                return f"{type(o).__name__}({_format(attrs, depth + 1)})"
            return f"{type(o).__name__}()"

        return repr(o)

    return f"{name} = {_format(obj, 0)}"


__all__ = [
    "timed",
    "timed_async",
    "log_call",
    "debug_context",
    "Timer",
    "get_call_stack",
    "format_exception_chain",
    "memory_usage_mb",
    "dump_state",
    "DEBUG_ENABLED",
    "PROFILE_ENABLED",
]
