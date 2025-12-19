"""Unified logging configuration for the RingRift AI service.

This module provides standardized logging setup used across all scripts.
Use this instead of calling logging.basicConfig() directly.

Usage:
    from app.core.logging_config import setup_logging, get_logger

    # In your script's main():
    setup_logging("my_script")
    logger = get_logger(__name__)

    # Or as a one-liner:
    logger = setup_logging(__name__)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Import path utilities (graceful fallback for bootstrap)
try:
    from app.utils.paths import ensure_dir, ensure_parent_dir
except ImportError:
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    def ensure_parent_dir(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

# Default log format - matches most existing scripts
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Compact format for high-volume output
COMPACT_FORMAT = "%(asctime)s [%(levelname).1s] %(message)s"

# Detailed format with file/line info
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# JSON-like format for structured logging
STRUCTURED_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been configured
_logging_configured = False


def setup_logging(
    name: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
    format_style: str = "default",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Set up standardized logging for a script or module.

    This function is idempotent - calling it multiple times with the same
    name will return the same configured logger.

    Args:
        name: Logger name. Use __name__ for module loggers, or a descriptive
              name for script entry points (e.g., "unified_ai_loop").
        level: Logging level (e.g., logging.INFO, logging.DEBUG, "DEBUG")
        format_style: One of "default", "compact", "detailed", "structured"
        log_file: Optional path to log file. If provided, logs go to file too.
        log_dir: Optional directory for log files. If provided with name but
                 no log_file, creates log_dir/{name}_{date}.log
        console: Whether to log to console (default True)
        propagate: Whether to propagate to parent loggers (default False)

    Returns:
        Configured Logger instance

    Example:
        # Basic usage
        logger = setup_logging(__name__)
        logger.info("Starting process")

        # With file logging
        logger = setup_logging("training", log_dir="logs/training")

        # Debug level with detailed format
        logger = setup_logging(__name__, level=logging.DEBUG, format_style="detailed")
    """
    global _logging_configured

    # Handle level as string
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Choose format
    format_map = {
        "default": DEFAULT_FORMAT,
        "compact": COMPACT_FORMAT,
        "detailed": DETAILED_FORMAT,
        "structured": STRUCTURED_FORMAT,
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=DATE_FORMAT)

    # Get or create logger
    logger_name = name or "ringrift"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file or log_dir:
        if log_file:
            file_path = Path(log_file)
        else:
            log_dir_path = ensure_dir(Path(log_dir))
            date_str = datetime.now().strftime("%Y%m%d")
            file_path = log_dir_path / f"{logger_name}_{date_str}.log"

        ensure_parent_dir(file_path)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Configure root logger if not already done (for third-party libs)
    if not _logging_configured:
        root = logging.getLogger()
        if not root.handlers:
            root.setLevel(logging.WARNING)  # Quieter root logger
            root_handler = logging.StreamHandler(sys.stdout)
            root_handler.setFormatter(logging.Formatter(COMPACT_FORMAT, datefmt=DATE_FORMAT))
            root.addHandler(root_handler)
        _logging_configured = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name.

    If logging hasn't been configured, sets up with defaults.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        # Logging not configured, set up defaults
        setup_logging(name)
    return logger


def configure_third_party_loggers(
    quiet: bool = True,
    verbose_packages: Optional[list[str]] = None,
) -> None:
    """Configure logging levels for common third-party packages.

    Args:
        quiet: If True, set third-party loggers to WARNING or higher
        verbose_packages: List of package names to keep at INFO level
    """
    noisy_packages = [
        "urllib3",
        "requests",
        "paramiko",
        "asyncio",
        "aiohttp",
        "websockets",
        "httpx",
        "httpcore",
        "fsspec",
        "torch",
        "transformers",
    ]

    verbose_packages = verbose_packages or []

    for package in noisy_packages:
        if package not in verbose_packages:
            logging.getLogger(package).setLevel(
                logging.WARNING if quiet else logging.INFO
            )


class LogContext:
    """Context manager for temporary log level changes.

    Usage:
        with LogContext(logger, logging.DEBUG):
            # This block has debug logging
            logger.debug("Detailed info")
        # Back to original level
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.original_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False
