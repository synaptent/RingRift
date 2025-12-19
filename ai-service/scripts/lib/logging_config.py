"""
Logging Configuration Library

Provides consistent logging setup across all training scripts:
- Structured logging with JSON support
- File and console handlers
- Log rotation
- Metrics collection

Usage:
    from scripts.lib.logging_config import setup_logging, get_logger

    setup_logging(level="INFO", log_file="logs/my_script.log")
    logger = get_logger(__name__)
    logger.info("Starting process", extra={"config": "square8_2p"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_pathname: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.include_pathname = include_pathname
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {}

        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_name:
            log_data["logger"] = record.name

        log_data["message"] = record.getMessage()

        if self.include_pathname:
            log_data["pathname"] = record.pathname
            log_data["lineno"] = record.lineno

        # Include exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        # Include configured extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"

        return super().format(record)


class MetricsLogger:
    """Collects and logs metrics for monitoring."""

    def __init__(self, name: str, log_interval: int = 60):
        """Initialize metrics logger.

        Args:
            name: Name for this metrics logger
            log_interval: Seconds between automatic metric logging
        """
        self.name = name
        self.log_interval = log_interval
        self.logger = logging.getLogger(f"metrics.{name}")
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, float] = {}
        self.last_log_time = time.time()

    def set(self, name: str, value: Any) -> None:
        """Set a metric value."""
        self.metrics[name] = value
        self._maybe_log()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value
        self._maybe_log()

    def time(self, name: str) -> "TimerContext":
        """Start a timer context manager."""
        return TimerContext(self, name)

    def record_time(self, name: str, seconds: float) -> None:
        """Record a timing measurement."""
        self.timers[name] = seconds
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Log metrics if interval has passed."""
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self.flush()
            self.last_log_time = now

    def flush(self) -> None:
        """Log all current metrics."""
        data = {
            "metrics": self.metrics.copy(),
            "counters": self.counters.copy(),
            "timers": self.timers.copy(),
        }
        self.logger.info("Metrics snapshot", extra=data)

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "metrics": self.metrics.copy(),
            "counters": self.counters.copy(),
            "timers": self.timers.copy(),
        }


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, metrics_logger: MetricsLogger, name: str):
        self.metrics_logger = metrics_logger
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimerContext":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.metrics_logger.record_time(self.name, elapsed)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    json_logs: bool = False,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        json_logs: Use JSON format for file logs
        log_to_console: Also log to console
        max_bytes: Max size before log rotation
        backup_count: Number of backup files to keep
        extra_fields: Extra fields to include in JSON logs

    Returns:
        Root logger
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)

        if json_logs:
            file_formatter = JsonFormatter(extra_fields=extra_fields)
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def get_metrics_logger(name: str, log_interval: int = 60) -> MetricsLogger:
    """Get a metrics logger with the given name."""
    return MetricsLogger(name, log_interval)


# Convenience function for scripts
def setup_script_logging(
    script_name: str,
    log_dir: Union[str, Path] = "logs",
    level: str = "INFO",
    json_logs: bool = False,
) -> logging.Logger:
    """Set up logging for a script with standard configuration.

    Args:
        script_name: Name of the script (used in log filename)
        log_dir: Directory for log files
        level: Log level
        json_logs: Use JSON format

    Returns:
        Logger for the script
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{script_name}_{timestamp}.log"

    setup_logging(
        level=level,
        log_file=log_file,
        json_logs=json_logs,
        extra_fields={"script": script_name},
    )

    return get_logger(script_name)
