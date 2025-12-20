"""Core shared infrastructure for the RingRift AI service.

This package provides standardized utilities used across all scripts:
- logging_config: Unified logging setup
- error_handler: Retry decorators, error recovery, emergency halt
- shutdown: Graceful shutdown coordination (December 2025)
"""

from app.core.error_handler import (
    FatalError,
    RetryableError,
    RingRiftError,
    retry,
    retry_async,
    with_emergency_halt_check,
)
from app.core.logging_config import get_logger, setup_logging
from app.core.shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    is_shutting_down,
    on_shutdown,
    request_shutdown,
    shutdown_scope,
)
from app.core.tasks import (
    TaskInfo,
    TaskManager,
    TaskState,
    background_task,
    get_task_manager,
)

__all__ = [
    "FatalError",
    "RetryableError",
    "RingRiftError",
    # Shutdown (December 2025)
    "ShutdownManager",
    "TaskInfo",
    "TaskManager",
    "TaskState",
    # Background tasks (December 2025)
    "background_task",
    "get_logger",
    "get_shutdown_manager",
    "get_task_manager",
    "is_shutting_down",
    "on_shutdown",
    "request_shutdown",
    # Error handling
    "retry",
    "retry_async",
    # Logging
    "setup_logging",
    "shutdown_scope",
    "with_emergency_halt_check",
]
