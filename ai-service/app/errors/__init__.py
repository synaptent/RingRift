"""Unified error handling module for RingRift AI Service.

This module provides a consolidated exception hierarchy for all RingRift
components. Use these exceptions instead of bare Exception or ValueError
for better error categorization and retry logic.

Usage:
    from app.errors import (
        RingRiftError,
        ResourceError,
        NetworkError,
        TrainingError,
        SyncError,
        DaemonError,
    )

    try:
        await sync_data_to_host(host)
    except NetworkError as e:
        logger.warning(f"Network error (retryable): {e}")
        await retry_with_backoff()
    except SyncError as e:
        logger.error(f"Sync logic error (not retryable): {e}")
        raise

Error Categories:
    - RingRiftError: Base for all RingRift errors
    - ResourceError: GPU, memory, disk issues (GPUError, MemoryError, DiskError)
    - NetworkError: Connection issues (SSHError, HTTPError, P2PError)
    - SyncError: Data sync issues (SyncTimeoutError, SyncIntegrityError)
    - TrainingError: Training issues (DataQualityError, ModelLoadError)
    - DaemonError: Daemon lifecycle (DaemonStartupError, DaemonCrashError)
    - ValidationError: Input/output validation
    - ConfigurationError: Configuration problems

Retry Guidelines:
    - NetworkError: Usually retryable with exponential backoff
    - ResourceError: May be retryable after waiting for resources
    - SyncError: Depends on type (timeout yes, integrity no)
    - TrainingError: Usually not retryable without intervention
    - ValidationError: Never retryable (fix the data)
    - ConfigurationError: Never retryable (fix the config)

December 2025: Created as canonical error module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# =============================================================================
# Error Codes (for structured logging and alerting)
# =============================================================================

class ErrorCode(Enum):
    """Error codes for structured logging and alerting."""

    # Resource errors (1xx)
    RESOURCE_EXHAUSTED = 100
    GPU_NOT_AVAILABLE = 101
    GPU_OOM = 102
    DISK_FULL = 103
    MEMORY_OOM = 104

    # Network errors (2xx)
    CONNECTION_FAILED = 200
    CONNECTION_TIMEOUT = 201
    SSH_AUTH_FAILED = 202
    SSH_COMMAND_FAILED = 203
    HTTP_CLIENT_ERROR = 204
    HTTP_SERVER_ERROR = 205
    P2P_MESH_ERROR = 206
    P2P_LEADER_UNAVAILABLE = 207

    # Sync errors (3xx)
    SYNC_TIMEOUT = 300
    SYNC_CONFLICT = 301
    SYNC_INTEGRITY_FAILED = 302
    SYNC_MANIFEST_MISMATCH = 303

    # Training errors (4xx)
    DATA_QUALITY_LOW = 400
    MODEL_LOAD_FAILED = 401
    CONVERGENCE_FAILED = 402
    CHECKPOINT_CORRUPT = 403
    TRAINING_INTERRUPTED = 404
    TRAINING_ERROR = 405
    CHECKPOINT_ERROR = 406
    MODEL_VERSIONING_ERROR = 407
    DATA_LOAD_ERROR = 408
    EVALUATION_ERROR = 409
    SELFPLAY_ERROR = 410

    # Daemon errors (5xx)
    DAEMON_START_FAILED = 500
    DAEMON_CRASHED = 501
    DAEMON_CONFIG_INVALID = 502
    DAEMON_DEPENDENCY_FAILED = 503

    # Validation errors (6xx)
    VALIDATION_FAILED = 600
    SCHEMA_MISMATCH = 601
    PARITY_FAILED = 602

    # Configuration errors (7xx)
    CONFIG_MISSING = 700
    CONFIG_INVALID = 701
    CONFIG_TYPE_ERROR = 702

    # System errors (8xx)
    EMERGENCY_HALT = 800
    RETRYABLE = 801
    NON_RETRYABLE = 802

    # Unknown
    UNKNOWN = 999


# =============================================================================
# Base Exception
# =============================================================================

class RingRiftError(Exception):
    """Base exception for all RingRift-specific errors.

    All custom exceptions should inherit from this to allow
    catching all RingRift errors with `except RingRiftError`.

    Attributes:
        message: Human-readable error message
        code: ErrorCode enum for structured handling
        details: Additional context as key-value pairs
        retryable: Whether this error is typically retryable
    """

    code: ErrorCode = ErrorCode.UNKNOWN
    retryable: bool = False

    def __init__(
        self,
        message: str,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}
        if retryable is not None:
            self.retryable = retryable

    def __str__(self) -> str:
        parts = [self.message]
        if self.code is not None:
            try:
                if isinstance(self.code, ErrorCode):
                    if self.code != ErrorCode.UNKNOWN:
                        parts.append(f"[{self.code.name}]")
                else:
                    # Defensive: handle string codes gracefully
                    parts.append(f"[{self.code}]")
            except (AttributeError, ValueError, TypeError):
                # Ultimate fallback: just skip the code
                pass
        if self.details:
            try:
                detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
                parts.append(f"({detail_str})")
            except (AttributeError, TypeError):
                pass
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        try:
            if isinstance(self.code, ErrorCode):
                code_value = self.code.value
                code_name = self.code.name
            else:
                code_value = self.code
                code_name = str(self.code)
        except (AttributeError, ValueError, TypeError):
            code_value = "UNKNOWN"
            code_name = "UNKNOWN"

        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": code_value,
            "code_name": code_name,
            "retryable": self.retryable,
            "details": self.details or {},
        }


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
    code = ErrorCode.RESOURCE_EXHAUSTED
    retryable = True  # Often retryable after waiting


class GPUError(ResourceError):
    """GPU-related error (not available, out of VRAM, etc.)."""
    code = ErrorCode.GPU_NOT_AVAILABLE


class GPUOutOfMemoryError(GPUError):
    """GPU ran out of memory."""
    code = ErrorCode.GPU_OOM


class DiskError(ResourceError):
    """Disk-related error (full, I/O error, etc.)."""
    code = ErrorCode.DISK_FULL


class DiskSpaceError(DiskError):
    """Insufficient disk space for operation.

    This is a specialized disk error for out-of-space conditions.
    Used by disk_utils.ensure_disk_space() to prevent data loss
    when disk fills up.

    Attributes:
        path: The path where disk space was checked
        available_bytes: Available space in bytes
        required_bytes: Required space in bytes
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        available_bytes: int | None = None,
        required_bytes: int | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # Build details dict
        details = context or {}
        if path is not None:
            details["path"] = path
        if available_bytes is not None:
            details["available_bytes"] = available_bytes
        if required_bytes is not None:
            details["required_bytes"] = required_bytes

        super().__init__(message, details=details, **kwargs)
        self.path = path
        self.available_bytes = available_bytes
        self.required_bytes = required_bytes


class MemoryExhaustedError(ResourceError):
    """System ran out of memory."""
    code = ErrorCode.MEMORY_OOM


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
    code = ErrorCode.CONNECTION_FAILED
    retryable = True


class ConnectionError(NetworkError):
    """Failed to connect to remote host."""
    code = ErrorCode.CONNECTION_FAILED


class ConnectionTimeoutError(NetworkError):
    """Connection timed out."""
    code = ErrorCode.CONNECTION_TIMEOUT


class SSHError(NetworkError):
    """SSH-specific error."""
    code = ErrorCode.SSH_COMMAND_FAILED


class SSHAuthError(SSHError):
    """SSH authentication failed."""
    code = ErrorCode.SSH_AUTH_FAILED
    retryable = False  # Auth errors usually not retryable


class HTTPError(NetworkError):
    """HTTP request error."""
    code = ErrorCode.HTTP_CLIENT_ERROR

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        if status_code:
            self.details["status_code"] = status_code
            if status_code >= 500:
                self.code = ErrorCode.HTTP_SERVER_ERROR


class P2PError(NetworkError):
    """P2P mesh network error."""
    code = ErrorCode.P2P_MESH_ERROR


class P2PLeaderUnavailableError(P2PError):
    """P2P leader node is unavailable."""
    code = ErrorCode.P2P_LEADER_UNAVAILABLE


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
    code = ErrorCode.SYNC_TIMEOUT
    retryable = True


class SyncTimeoutError(SyncError):
    """Sync operation timed out."""
    code = ErrorCode.SYNC_TIMEOUT


class SyncConflictError(SyncError):
    """Conflicting data during sync (e.g., file modified on both sides)."""
    code = ErrorCode.SYNC_CONFLICT
    retryable = False  # Need manual resolution


class SyncIntegrityError(SyncError):
    """Data integrity check failed after sync."""
    code = ErrorCode.SYNC_INTEGRITY_FAILED
    retryable = False  # Data is corrupted


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
    code = ErrorCode.TRAINING_ERROR
    retryable = False  # Usually need intervention


class DataQualityError(TrainingError):
    """Training data quality too low."""
    code = ErrorCode.DATA_QUALITY_LOW


class InvalidGameError(DataQualityError):
    """Game data integrity violation.

    Raised when a game cannot be stored or finalized due to data integrity
    issues, such as missing move data. This is a critical safeguard to
    prevent useless games from polluting training databases.

    Attributes:
        game_id: The ID of the invalid game
        move_count: Number of moves in the game
    """

    def __init__(
        self,
        message: str,
        game_id: str | None = None,
        move_count: int | None = None,
        **kwargs: Any,
    ):
        details = kwargs.pop("details", {}) or {}
        if game_id:
            details["game_id"] = game_id
        if move_count is not None:
            details["move_count"] = move_count
        super().__init__(message, details=details, **kwargs)
        self.game_id = game_id
        self.move_count = move_count


class ModelLoadError(TrainingError):
    """Failed to load model checkpoint."""
    code = ErrorCode.MODEL_LOAD_FAILED


class CheckpointCorruptError(ModelLoadError):
    """Model checkpoint is corrupted."""
    code = ErrorCode.CHECKPOINT_CORRUPT


class ConvergenceError(TrainingError):
    """Model failed to converge during training."""
    code = ErrorCode.CONVERGENCE_FAILED


class CheckpointError(TrainingError):
    """Error during checkpoint save/load operations."""
    code = ErrorCode.CHECKPOINT_ERROR


class ModelVersioningError(CheckpointError):
    """Error in model versioning or compatibility."""
    code = ErrorCode.MODEL_VERSIONING_ERROR


class DataLoadError(TrainingError):
    """Error loading training data."""
    code = ErrorCode.DATA_LOAD_ERROR


class EvaluationError(TrainingError):
    """Error during model evaluation."""
    code = ErrorCode.EVALUATION_ERROR


class SelfplayError(TrainingError):
    """Error during self-play data generation."""
    code = ErrorCode.SELFPLAY_ERROR


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
    code = ErrorCode.DAEMON_CRASHED
    retryable = True  # Daemons can often be restarted


class DaemonStartupError(DaemonError):
    """Daemon failed to start."""
    code = ErrorCode.DAEMON_START_FAILED


class DaemonCrashError(DaemonError):
    """Daemon crashed unexpectedly."""
    code = ErrorCode.DAEMON_CRASHED


class DaemonConfigError(DaemonError):
    """Invalid daemon configuration."""
    code = ErrorCode.DAEMON_CONFIG_INVALID
    retryable = False  # Fix config first


class DaemonDependencyError(DaemonError):
    """Daemon dependency not available."""
    code = ErrorCode.DAEMON_DEPENDENCY_FAILED


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(RingRiftError):
    """Input or output validation failed.

    Use this for:
    - Schema validation failures
    - Data format errors
    - Parity check failures
    """
    code = ErrorCode.VALIDATION_FAILED
    retryable = False  # Fix the data


class SchemaError(ValidationError):
    """Data doesn't match expected schema."""
    code = ErrorCode.SCHEMA_MISMATCH


class ParityError(ValidationError):
    """TS/Python parity check failed."""
    code = ErrorCode.PARITY_FAILED


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(RingRiftError):
    """Configuration problem.

    Use this for:
    - Missing required config
    - Invalid config values
    - Config type mismatches
    """
    code = ErrorCode.CONFIG_INVALID
    retryable = False  # Fix the config


class ConfigMissingError(ConfigurationError):
    """Required configuration is missing."""
    code = ErrorCode.CONFIG_MISSING


class ConfigTypeError(ConfigurationError):
    """Configuration value has wrong type."""
    code = ErrorCode.CONFIG_TYPE_ERROR


# =============================================================================
# System Errors
# =============================================================================

class EmergencyHaltError(RingRiftError):
    """Emergency halt triggered to stop all training and selfplay.

    Used when a critical issue is detected that requires immediate
    cessation of all training and selfplay operations.
    """
    code = ErrorCode.EMERGENCY_HALT
    retryable = False  # Requires manual intervention


class RetryableError(RingRiftError):
    """Error that can be retried (network issues, transient failures).

    Use this for errors where a retry may succeed, such as:
    - Network timeouts
    - SSH connection drops
    - Temporary resource unavailability
    """
    code = ErrorCode.RETRYABLE
    retryable = True


class NonRetryableError(RingRiftError):
    """Error that should not be retried.

    Alias: FatalError
    Use when retry would be futile.
    """
    code = ErrorCode.NON_RETRYABLE
    retryable = False


# Backwards compatibility aliases
FatalError = NonRetryableError
RecoverableError = RetryableError


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Codes
    "ErrorCode",
    # Base
    "RingRiftError",
    # Resource
    "ResourceError",
    "GPUError",
    "GPUOutOfMemoryError",
    "DiskError",
    "DiskSpaceError",
    "MemoryExhaustedError",
    # Network
    "NetworkError",
    "ConnectionError",
    "ConnectionTimeoutError",
    "SSHError",
    "SSHAuthError",
    "HTTPError",
    "P2PError",
    "P2PLeaderUnavailableError",
    # Sync
    "SyncError",
    "SyncTimeoutError",
    "SyncConflictError",
    "SyncIntegrityError",
    # Training
    "TrainingError",
    "DataQualityError",
    "InvalidGameError",
    "ModelLoadError",
    "CheckpointCorruptError",
    "ConvergenceError",
    "ModelVersioningError",
    "CheckpointError",
    "DataLoadError",
    "EvaluationError",
    "SelfplayError",
    # Daemon
    "DaemonError",
    "DaemonStartupError",
    "DaemonCrashError",
    "DaemonConfigError",
    "DaemonDependencyError",
    # Validation
    "ValidationError",
    "SchemaError",
    "ParityError",
    # Configuration
    "ConfigurationError",
    "ConfigMissingError",
    "ConfigTypeError",
    # System
    "EmergencyHaltError",
    "RetryableError",
    "NonRetryableError",
    # Aliases
    "FatalError",
    "RecoverableError",
]
