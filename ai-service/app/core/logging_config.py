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

Environment Variables (December 2025):
    RINGRIFT_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
    RINGRIFT_LOG_FORMAT: Set format style (default, compact, detailed, json)
    RINGRIFT_LOG_FILE: Path to log file (optional)
    RINGRIFT_LOG_JSON: Set to "true" for JSON structured logging

Migration Guide:
    BEFORE (scattered logging setup):
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    AFTER (centralized):
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)

    Benefits:
        - Consistent formatting across all modules
        - Environment-based configuration
        - JSON structured logging support
        - Automatic third-party logger silencing
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

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


# =============================================================================
# Environment-based Configuration (December 2025)
# =============================================================================

def _get_env_log_level() -> int:
    """Get log level from environment."""
    level_str = os.environ.get("RINGRIFT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def _get_env_format_style() -> str:
    """Get format style from environment."""
    style = os.environ.get("RINGRIFT_LOG_FORMAT", "default").lower()
    if os.environ.get("RINGRIFT_LOG_JSON", "").lower() == "true":
        return "json"
    return style


def _get_env_log_file() -> Optional[str]:
    """Get log file path from environment."""
    return os.environ.get("RINGRIFT_LOG_FILE")


# Default log format - matches most existing scripts
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Compact format for high-volume output
COMPACT_FORMAT = "%(asctime)s [%(levelname).1s] %(message)s"

# Detailed format with file/line info
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# JSON-like format for structured logging (simple string format)
STRUCTURED_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging has been configured
_logging_configured = False


# =============================================================================
# JSON Formatter (December 2025)
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing by log aggregators
    like ELK, Loki, or CloudWatch.

    Output format:
        {"timestamp": "2025-12-19T10:30:00", "level": "INFO", "logger": "app.training", "message": "...", ...}
    """

    def __init__(
        self,
        include_extra: bool = True,
        include_exception: bool = True,
    ):
        super().__init__()
        self.include_extra = include_extra
        self.include_exception = include_exception

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.pathname:
            log_obj["file"] = f"{record.filename}:{record.lineno}"

        # Add extra fields (anything passed via extra={...} in log calls)
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "taskName", "message",
                }:
                    try:
                        json.dumps(value)  # Check serializable
                        log_obj[key] = value
                    except (TypeError, ValueError):
                        log_obj[key] = str(value)

        # Add exception info
        if self.include_exception and record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def create_formatter(format_style: str) -> logging.Formatter:
    """Create a formatter based on style.

    Args:
        format_style: One of "default", "compact", "detailed", "json", "structured"

    Returns:
        Configured Formatter instance
    """
    if format_style == "json":
        return JSONFormatter()

    format_map = {
        "default": DEFAULT_FORMAT,
        "compact": COMPACT_FORMAT,
        "detailed": DETAILED_FORMAT,
        "structured": STRUCTURED_FORMAT,
    }
    log_format = format_map.get(format_style, DEFAULT_FORMAT)
    return logging.Formatter(log_format, datefmt=DATE_FORMAT)


def setup_logging(
    name: Optional[str] = None,
    level: Optional[Union[int, str]] = None,
    format_style: Optional[str] = None,
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

    # Apply environment-based defaults (December 2025)
    if level is None:
        level = _get_env_log_level()
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_style is None:
        format_style = _get_env_format_style()

    if log_file is None:
        log_file = _get_env_log_file()

    # Create formatter using centralized helper
    formatter = create_formatter(format_style)

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

    def __enter__(self) -> logging.Logger:
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.logger.setLevel(self.original_level)


# =============================================================================
# Convenience Functions (December 2025)
# =============================================================================

def quick_setup(name: str = "ringrift", level: int = logging.INFO) -> logging.Logger:
    """Quick one-liner logging setup for simple scripts.

    Usage:
        from app.core.logging_config import quick_setup
        logger = quick_setup(__name__)
        logger.info("Ready to go")
    """
    return setup_logging(name, level=level)


def production_logger(name: str) -> logging.Logger:
    """Get a production-ready logger with JSON output.

    Suitable for containerized deployments where logs are parsed by
    log aggregators (ELK, Loki, CloudWatch).

    Usage:
        from app.core.logging_config import production_logger
        logger = production_logger(__name__)
    """
    return setup_logging(name, format_style="json")


def script_logger(
    script_name: str,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Get a logger configured for long-running scripts.

    Logs to both console and file, with detailed format.

    Usage:
        from app.core.logging_config import script_logger
        logger = script_logger("unified_ai_loop", log_dir="logs/training")
    """
    if log_dir is None:
        log_dir = Path("logs") / script_name
    return setup_logging(
        script_name,
        format_style="detailed",
        log_dir=log_dir,
    )


def add_context(logger: logging.Logger, **context: Any) -> logging.LoggerAdapter:
    """Create a logger adapter with extra context fields.

    When using JSON logging, these fields appear in every log entry.

    Usage:
        from app.core.logging_config import get_logger, add_context
        logger = get_logger(__name__)
        ctx_logger = add_context(logger, config_key="square8_2p", node="gpu-01")
        ctx_logger.info("Training started")  # Includes config_key and node in JSON
    """
    return logging.LoggerAdapter(logger, context)
