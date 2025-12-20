"""Standardized logging utilities for consistent log formatting.

This module provides utilities for creating loggers with consistent
formatting and prefixes across the codebase.

Usage:
    from app.utils.logging_utils import get_logger, LogContext

    # Get a logger with automatic module prefix
    logger = get_logger(__name__)
    logger.info("Starting process")  # Logs with module name in standard format

    # Or create a prefixed logger
    logger = get_logger("MyComponent")
    logger.info("Starting")  # "[MyComponent] Starting"

    # Use log context for structured logging
    with LogContext(logger, operation="sync", target="db"):
        logger.info("Starting sync")  # Includes context in message
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class PrefixedLogger:
    """A logger wrapper that automatically adds a prefix to messages.

    This provides a consistent `[Prefix] message` format across the codebase.
    """

    def __init__(self, logger: logging.Logger, prefix: str):
        self._logger = logger
        self._prefix = prefix

    def _format(self, msg: str) -> str:
        return f"[{self._prefix}] {msg}"

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(self._format(msg), *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(self._format(msg), *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(self._format(msg), *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(self._format(msg), *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(self._format(msg), *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(self._format(msg), *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        self._logger.log(level, self._format(msg), *args, **kwargs)

    @property
    def level(self) -> int:
        return self._logger.level

    def setLevel(self, level: int) -> None:
        self._logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)


def get_logger(name: str, *, prefixed: bool = True) -> PrefixedLogger | logging.Logger:
    """Get a logger with optional automatic prefix.

    Args:
        name: Logger name (typically __name__ or component name)
        prefixed: If True, wrap in PrefixedLogger

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Message")  # "[module.name] Message"

        logger = get_logger("MyComponent")
        logger.info("Message")  # "[MyComponent] Message"
    """
    # Extract short name from module path
    if "." in name:
        short_name = name.split(".")[-1]
    else:
        short_name = name

    # Convert snake_case to CamelCase for prefix
    if "_" in short_name:
        prefix = "".join(word.capitalize() for word in short_name.split("_"))
    else:
        prefix = short_name.capitalize() if short_name.islower() else short_name

    base_logger = logging.getLogger(name)

    if prefixed:
        return PrefixedLogger(base_logger, prefix)
    return base_logger


@dataclass
class LogContext:
    """Context manager for adding structured context to log messages.

    Example:
        logger = get_logger(__name__)
        with LogContext(logger, operation="sync", node="worker-1"):
            logger.info("Starting")  # "[Module] Starting (operation=sync, node=worker-1)"
    """

    logger: PrefixedLogger | logging.Logger
    context: dict[str, Any] = field(default_factory=dict)
    _original_prefix: str | None = field(default=None, init=False)

    def __init__(self, logger: PrefixedLogger | logging.Logger, **context):
        self.logger = logger
        self.context = context

    def __enter__(self) -> LogContext:
        if isinstance(self.logger, PrefixedLogger):
            self._original_prefix = self.logger._prefix
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            self.logger._prefix = f"{self._original_prefix} {context_str}"
        return self

    def __exit__(self, *args):
        if isinstance(self.logger, PrefixedLogger) and self._original_prefix:
            self.logger._prefix = self._original_prefix


def log_duration(
    logger: PrefixedLogger | logging.Logger,
    operation: str,
    level: int = logging.INFO,
    threshold_ms: float = 0,
) -> Callable[[F], F]:
    """Decorator to log the duration of a function.

    Args:
        logger: Logger to use
        operation: Description of the operation
        level: Log level to use
        threshold_ms: Only log if duration exceeds threshold

    Example:
        logger = get_logger(__name__)

        @log_duration(logger, "process batch")
        def process_batch(items):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms >= threshold_ms:
                    if isinstance(logger, PrefixedLogger):
                        logger.log(level, f"{operation} completed in {elapsed_ms:.1f}ms")
                    else:
                        logger.log(level, f"{operation} completed in {elapsed_ms:.1f}ms")

        return wrapper  # type: ignore

    return decorator


def log_duration_async(
    logger: PrefixedLogger | logging.Logger,
    operation: str,
    level: int = logging.INFO,
    threshold_ms: float = 0,
) -> Callable[[F], F]:
    """Async version of log_duration decorator."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms >= threshold_ms:
                    if isinstance(logger, PrefixedLogger):
                        logger.log(level, f"{operation} completed in {elapsed_ms:.1f}ms")
                    else:
                        logger.log(level, f"{operation} completed in {elapsed_ms:.1f}ms")

        return wrapper  # type: ignore

    return decorator


def configure_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    stream=None,
) -> None:
    """Configure the root logger with standard format.

    Args:
        level: Log level
        format_string: Custom format string (uses default if None)
        stream: Output stream (defaults to stderr)
    """
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        stream=stream or sys.stderr,
    )
    # basicConfig only sets level if no handlers exist, so set explicitly
    logging.getLogger().setLevel(level)


def silence_logger(name: str, level: int = logging.WARNING) -> None:
    """Silence a noisy logger by raising its level.

    Args:
        name: Logger name to silence
        level: New minimum level
    """
    logging.getLogger(name).setLevel(level)


__all__ = [
    "LogContext",
    "PrefixedLogger",
    "configure_logging",
    "get_logger",
    "log_duration",
    "log_duration_async",
    "silence_logger",
]
