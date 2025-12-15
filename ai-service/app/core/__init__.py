"""Core shared infrastructure for the RingRift AI service.

This package provides standardized utilities used across all scripts:
- logging_config: Unified logging setup
- error_handler: Retry decorators, error recovery, emergency halt
"""

from app.core.logging_config import setup_logging, get_logger
from app.core.error_handler import (
    retry,
    retry_async,
    with_emergency_halt_check,
    RingRiftError,
    RetryableError,
    FatalError,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "retry",
    "retry_async",
    "with_emergency_halt_check",
    "RingRiftError",
    "RetryableError",
    "FatalError",
]
