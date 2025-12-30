"""Base class for AutoSyncDaemon mixins.

December 2025: Created to consolidate common patterns across the 4 sync mixins:
- SyncEventMixin
- SyncPushMixin
- SyncPullMixin
- SyncEphemeralMixin

Provides:
- Protocol defining expected interface from main class
- Common abstract methods (emit events)
- Utility methods for error handling
- Shared type hints
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.coordination.sync_strategies import AutoSyncConfig, SyncStats
    from app.distributed.circuit_breaker import CircuitBreaker
    from app.distributed.cluster_manifest import ClusterManifest

logger = logging.getLogger(__name__)


@runtime_checkable
class AutoSyncDaemonProtocol(Protocol):
    """Protocol defining the interface expected from AutoSyncDaemon.

    All sync mixins expect the main class to implement these attributes
    and methods. This protocol documents the contract explicitly.
    """

    # Core configuration
    config: AutoSyncConfig
    node_id: str

    # Runtime state
    _running: bool
    _subscribed: bool
    _is_ephemeral: bool
    _is_broadcast: bool

    # Statistics
    _stats: SyncStats
    _events_processed: int
    _errors_count: int
    _last_error: str

    # Infrastructure
    _circuit_breaker: CircuitBreaker | None
    _cluster_manifest: ClusterManifest | None

    # Ephemeral mode state
    _urgent_sync_pending: dict[str, float]
    _pending_games: list[dict[str, Any]]

    # Core sync methods
    async def _sync_all(self) -> None:
        """Execute full sync cycle."""
        ...

    async def _sync_to_peer(self, node_id: str) -> bool:
        """Sync to a specific peer node."""
        ...

    def _validate_database_completeness(self, db_path: Path) -> tuple[bool, str]:
        """Validate database has complete game data."""
        ...


@dataclass
class SyncError:
    """Represents a sync error with context.

    Attributes:
        error_type: Category of error (network, timeout, validation, etc.)
        message: Human-readable error message
        target_node: Node that sync was targeting (if applicable)
        db_path: Database path involved (if applicable)
        timestamp: When the error occurred
        recoverable: Whether this error type is typically recoverable
    """

    error_type: str
    message: str
    target_node: str = ""
    db_path: str = ""
    timestamp: float = field(default_factory=time.time)
    recoverable: bool = True

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        target_node: str = "",
        db_path: str = "",
    ) -> SyncError:
        """Create SyncError from an exception.

        Args:
            exc: The exception that was raised
            target_node: Target node if applicable
            db_path: Database path if applicable

        Returns:
            SyncError instance with appropriate error_type
        """
        exc_type = type(exc).__name__

        # Classify by exception type
        if "Timeout" in exc_type or "timeout" in str(exc).lower():
            error_type = "timeout"
            recoverable = True
        elif "Connection" in exc_type or "SSH" in str(exc):
            error_type = "network"
            recoverable = True
        elif "sqlite" in str(exc).lower() or "database" in str(exc).lower():
            error_type = "database"
            recoverable = False
        elif "Permission" in str(exc) or "Access" in str(exc):
            error_type = "permission"
            recoverable = False
        else:
            error_type = "unknown"
            recoverable = True

        return cls(
            error_type=error_type,
            message=str(exc),
            target_node=target_node,
            db_path=db_path,
            recoverable=recoverable,
        )


class SyncMixinBase(ABC):
    """Base class for AutoSyncDaemon mixins.

    Provides common functionality used across all sync mixins:
    - Error tracking utilities
    - Event emission stubs
    - Logging helpers

    Subclasses should declare their specific required attributes as type hints.
    """

    # Type hints for attributes from main class - subclasses can reference these
    config: AutoSyncConfig
    node_id: str
    _stats: SyncStats
    _running: bool
    _events_processed: int
    _errors_count: int
    _last_error: str
    _circuit_breaker: CircuitBreaker | None

    # Logging prefix for consistent log messages
    LOG_PREFIX: str = "[AutoSyncDaemon]"

    def _record_error(self, error: str | Exception, target_node: str = "") -> SyncError:
        """Record an error and update statistics.

        Args:
            error: Error message or exception
            target_node: Target node if applicable

        Returns:
            SyncError instance with error details
        """
        if isinstance(error, Exception):
            sync_error = SyncError.from_exception(error, target_node=target_node)
        else:
            sync_error = SyncError(
                error_type="unknown",
                message=str(error),
                target_node=target_node,
            )

        # Update counters
        self._errors_count += 1
        self._last_error = sync_error.message

        # Log based on severity
        if sync_error.recoverable:
            logger.warning(f"{self.LOG_PREFIX} {sync_error.error_type} error: {sync_error.message}")
        else:
            logger.error(f"{self.LOG_PREFIX} {sync_error.error_type} error: {sync_error.message}")

        return sync_error

    def _record_event_processed(self) -> None:
        """Increment the events processed counter."""
        self._events_processed += 1

    def _log_info(self, message: str) -> None:
        """Log an info message with standard prefix."""
        logger.info(f"{self.LOG_PREFIX} {message}")

    def _log_debug(self, message: str) -> None:
        """Log a debug message with standard prefix."""
        logger.debug(f"{self.LOG_PREFIX} {message}")

    def _log_warning(self, message: str) -> None:
        """Log a warning message with standard prefix."""
        logger.warning(f"{self.LOG_PREFIX} {message}")

    def _log_error(self, message: str) -> None:
        """Log an error message with standard prefix."""
        logger.error(f"{self.LOG_PREFIX} {message}")

    async def _retry_with_backoff(
        self,
        async_func: Any,
        *args: Any,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        backoff_multiplier: float | None = None,
        non_retryable_errors: tuple[str, ...] = ("Connection refused", "No route to host"),
        operation_name: str = "operation",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an async function with exponential backoff retry.

        December 2025: Extracted from sync_push_mixin.py to consolidate
        retry logic across all sync mixins.

        Args:
            async_func: Async function to execute
            *args: Positional arguments for the function
            max_retries: Max retry attempts (default from RetryDefaults)
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            backoff_multiplier: Multiplier for exponential backoff
            non_retryable_errors: Error substrings that should not trigger retry
            operation_name: Name for logging purposes
            **kwargs: Keyword arguments for the function

        Returns:
            Result dict from the function, or error dict on failure
        """
        import asyncio
        import random

        # Get retry defaults
        try:
            from app.config.coordination_defaults import RetryDefaults
            max_retries = max_retries if max_retries is not None else RetryDefaults.SYNC_MAX_RETRIES
            base_delay = base_delay if base_delay is not None else RetryDefaults.SYNC_BASE_DELAY
            max_delay = max_delay if max_delay is not None else RetryDefaults.SYNC_MAX_DELAY
            backoff_multiplier = backoff_multiplier if backoff_multiplier is not None else RetryDefaults.BACKOFF_MULTIPLIER
        except ImportError:
            max_retries = max_retries if max_retries is not None else 3
            base_delay = base_delay if base_delay is not None else 2.0
            max_delay = max_delay if max_delay is not None else 30.0
            backoff_multiplier = backoff_multiplier if backoff_multiplier is not None else 2.0

        last_result: dict[str, Any] | None = None

        for attempt in range(max_retries):
            try:
                result = await async_func(*args, **kwargs)

                # Check if result indicates success
                if isinstance(result, dict):
                    if result.get("success"):
                        if attempt > 0:
                            self._log_info(f"{operation_name} succeeded on attempt {attempt + 1}")
                        return result
                    last_result = result
                    error = str(result.get("error", "Unknown"))
                else:
                    # Non-dict result treated as success
                    return {"success": True, "result": result}

            except Exception as e:
                error = str(e)
                last_result = {"success": False, "error": error}

            # Check if we should retry
            if any(non_ret in error for non_ret in non_retryable_errors):
                self._log_debug(f"Not retrying {operation_name}: {error}")
                break

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                # Add jitter (+/-10%)
                jitter = delay * 0.1 * (random.random() * 2 - 1)
                delay = delay + jitter

                self._log_debug(
                    f"{operation_name} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.1f}s: {error}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_result and max_retries > 1:
            self._log_warning(
                f"{operation_name} failed after {max_retries} attempts: "
                f"{last_result.get('error', 'Unknown')}"
            )

        return last_result or {"success": False, "error": "No result"}

    # Abstract methods - centralized declarations for all mixins
    @abstractmethod
    async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
        """Emit DATA_SYNC_FAILED event for sync failure.

        Args:
            target_node: Node the sync was targeting
            db_path: Database path that failed to sync
            error: Error message describing the failure
        """
        raise NotImplementedError

    @abstractmethod
    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        """Emit SYNC_STALLED event for timeout situations.

        Args:
            target_node: Node the sync was targeting
            timeout_seconds: How long the operation waited before timing out
            data_type: Type of data being synced (game, model, etc.)
            retry_count: Number of retries attempted
        """
        raise NotImplementedError


# Convenience type alias for type checking
SyncDaemonLike = AutoSyncDaemonProtocol


def validate_sync_daemon_protocol(obj: Any) -> bool:
    """Check if an object implements the AutoSyncDaemonProtocol.

    Args:
        obj: Object to validate

    Returns:
        True if object has all required attributes and methods
    """
    required_attrs = [
        "config",
        "node_id",
        "_running",
        "_stats",
        "_events_processed",
        "_errors_count",
    ]
    required_methods = [
        "_sync_all",
        "_sync_to_peer",
    ]

    for attr in required_attrs:
        if not hasattr(obj, attr):
            return False

    for method in required_methods:
        if not hasattr(obj, method) or not callable(getattr(obj, method)):
            return False

    return True
