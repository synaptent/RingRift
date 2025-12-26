"""RingRift Exception Hierarchy - Typed exceptions for better error handling.

.. deprecated:: December 2025
    This module is deprecated. Use `app.errors` instead:

        from app.errors import (
            RingRiftError,
            ResourceError,
            SyncError,
            TrainingError,
            # ... etc
        )

    This module will be removed in a future version.

Legacy Usage (deprecated):
    from app.core.exceptions import (
        RingRiftError,
        ResourceError,
        SyncError,
        DaemonError,
        NetworkError,
        TrainingError,
    )

    try:
        sync_data_to_host(host)
    except NetworkError as e:
        logger.error(f"Network error during sync: {e}")
        # Can retry
    except SyncError as e:
        logger.error(f"Sync logic error: {e}")
        # Can't retry, need to fix

Benefits:
- Clearer error messages in logs
- Ability to catch specific error types
- Better retry logic (retry NetworkError, don't retry ValidationError)
- Prevents hiding bugs with broad exception catches
"""

from __future__ import annotations


# =============================================================================
# Base Exception
# =============================================================================

class RingRiftError(Exception):
    """Base exception for all RingRift-specific errors.

    All custom exceptions should inherit from this to allow
    catching all RingRift errors with `except RingRiftError`.
    """

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Resource Errors
# =============================================================================

class ResourceError(RingRiftError):
    """Error related to system resources (GPU, memory, disk, etc.).

    Use this for:
    - Out of memory errors
    - Disk full errors
    - GPU not available
    - CPU overload
    """
    pass


class MemoryError(ResourceError):
    """Out of memory error."""
    pass


class DiskError(ResourceError):
    """Disk-related error (full, I/O error, etc.)."""
    pass


class GPUError(ResourceError):
    """GPU-related error (not available, out of VRAM, etc.)."""
    pass


# =============================================================================
# Sync Errors
# =============================================================================

class SyncError(RingRiftError):
    """Error during data synchronization.

    Use this for:
    - Failed file transfers
    - Data integrity issues
    - Sync protocol errors
    """
    pass


class SyncTimeoutError(SyncError):
    """Sync operation timed out."""
    pass


class SyncConflictError(SyncError):
    """Conflicting data during sync (e.g., file modified on both sides)."""
    pass


class SyncIntegrityError(SyncError):
    """Data integrity check failed after sync."""
    pass


# =============================================================================
# Network Errors
# =============================================================================

class NetworkError(RingRiftError):
    """Error during network operations.

    Use this for:
    - Connection failures
    - SSH errors
    - HTTP errors
    - P2P communication errors

    These are typically retryable.
    """
    pass


class ConnectionError(NetworkError):
    """Failed to connect to remote host."""
    pass


class SSHError(NetworkError):
    """SSH-specific error."""
    pass


class HTTPError(NetworkError):
    """HTTP request error."""

    def __init__(self, message: str, status_code: int | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class P2PError(NetworkError):
    """P2P mesh network error."""
    pass


# =============================================================================
# Daemon Errors
# =============================================================================

class DaemonError(RingRiftError):
    """Error in daemon lifecycle or operation.

    Use this for:
    - Daemon startup failures
    - Daemon crash handling
    - Daemon configuration errors
    """
    pass


class DaemonStartupError(DaemonError):
    """Daemon failed to start."""
    pass


class DaemonCrashError(DaemonError):
    """Daemon crashed unexpectedly."""
    pass


class DaemonConfigError(DaemonError):
    """Invalid daemon configuration."""
    pass


# =============================================================================
# Training Errors
# =============================================================================

class TrainingError(RingRiftError):
    """Error during model training.

    Use this for:
    - Training data issues
    - Model convergence problems
    - Checkpoint loading errors
    """
    pass


class DataQualityError(TrainingError):
    """Training data quality too low."""
    pass


class ModelLoadError(TrainingError):
    """Failed to load model checkpoint."""
    pass


class ConvergenceError(TrainingError):
    """Model failed to converge during training."""
    pass


class RegressionError(TrainingError):
    """Model performance regressed."""

    def __init__(self, message: str, severity: str = "moderate", **kwargs):
        super().__init__(message, **kwargs)
        self.severity = severity


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(RingRiftError):
    """Error during validation.

    Use this for:
    - Input validation failures
    - Schema mismatches
    - Configuration validation
    """
    pass


class ConfigValidationError(ValidationError):
    """Configuration validation failed."""
    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    pass


# =============================================================================
# Event System Errors
# =============================================================================

class EventError(RingRiftError):
    """Error in event system.

    Use this for:
    - Event routing failures
    - Event handler errors
    - Event bus issues
    """
    pass


class EventRoutingError(EventError):
    """Event could not be routed."""
    pass


class EventHandlerError(EventError):
    """Event handler raised an error."""
    pass


# =============================================================================
# Selfplay Errors
# =============================================================================

class SelfplayError(RingRiftError):
    """Error during selfplay game generation.

    Use this for:
    - Game engine errors
    - AI decision errors
    - Game state inconsistencies
    """
    pass


class GameEngineError(SelfplayError):
    """Game engine raised an error."""
    pass


class ParityError(SelfplayError):
    """TypeScript/Python parity check failed."""
    pass


# =============================================================================
# Cluster Errors
# =============================================================================

class ClusterError(RingRiftError):
    """Error in cluster operations.

    Use this for:
    - Leader election issues
    - Node coordination failures
    - Cluster state inconsistencies
    """
    pass


class LeaderElectionError(ClusterError):
    """Leader election failed."""
    pass


class NodeUnreachableError(ClusterError):
    """Cluster node is unreachable."""
    pass


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Base
    "RingRiftError",
    # Resources
    "ResourceError",
    "MemoryError",
    "DiskError",
    "GPUError",
    # Sync
    "SyncError",
    "SyncTimeoutError",
    "SyncConflictError",
    "SyncIntegrityError",
    # Network
    "NetworkError",
    "ConnectionError",
    "SSHError",
    "HTTPError",
    "P2PError",
    # Daemon
    "DaemonError",
    "DaemonStartupError",
    "DaemonCrashError",
    "DaemonConfigError",
    # Training
    "TrainingError",
    "DataQualityError",
    "ModelLoadError",
    "ConvergenceError",
    "RegressionError",
    # Validation
    "ValidationError",
    "ConfigValidationError",
    "SchemaValidationError",
    # Events
    "EventError",
    "EventRoutingError",
    "EventHandlerError",
    # Selfplay
    "SelfplayError",
    "GameEngineError",
    "ParityError",
    # Cluster
    "ClusterError",
    "LeaderElectionError",
    "NodeUnreachableError",
]
