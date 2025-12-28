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
