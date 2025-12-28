"""
RingRift Error Hierarchy

Unified exception hierarchy for consistent error handling across the codebase.
All custom exceptions inherit from RingRiftError for easy catching and filtering.

Usage:
    from app.errors import RulesViolationError, AIFallbackError

    try:
        engine.apply_move(state, move)
    except RulesViolationError as e:
        logger.warning(f"Invalid move: {e.message}, rule: {e.rule_ref}")
"""

from typing import Any

__all__ = [
    # AI errors
    "AIError",
    "AIFallbackError",
    "AITimeoutError",
    "CheckpointError",
    "CheckpointIncompatibleError",
    "ClusterError",
    "ConfigurationError",
    "DataLoadError",
    "DataQualityError",
    # Database errors
    "DatabaseError",
    "DatasetError",
    "DiskSpaceError",
    "EmergencyHaltError",
    "EvaluationError",
    # Aliases
    "FatalError",
    # Infrastructure errors
    "InfrastructureError",
    "InvalidGameError",
    "InvalidMoveError",
    "InvalidStateError",
    "LifecycleError",
    "ModelLoadError",
    "ModelVersioningError",
    "NoHealthyWorkersError",
    "NonRetryableError",
    "OutOfMemoryError",
    "ParityError",
    "RegressionDetectedError",
    # Resource errors
    "ResourceError",
    # Retry/recovery errors
    "RetryableError",
    # Base error
    "RingRiftError",
    # Game rules errors
    "RulesViolationError",
    "SSHError",
    "SelfplayError",
    "StorageError",
    "SyncError",
    # Training errors
    "TrainingError",
    # Validation errors
    "ValidationError",
]


class RingRiftError(Exception):
    """Base exception for all RingRift errors.

    Attributes:
        code: Machine-readable error code for categorization
        message: Human-readable error description
        context: Additional context for debugging
    """
    code: str = "RINGRIFT_ERROR"

    def __init__(
        self,
        message: str,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{self.code}] {self.message} ({ctx})"
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


# =============================================================================
# Game Rules Errors
# =============================================================================


class RulesViolationError(RingRiftError):
    """Invalid move per game rules.

    Raised when a move violates the canonical game rules. The rule_ref
    field can reference specific rules from RULES_CANONICAL_SPEC.md.

    Attributes:
        rule_ref: Reference to specific rule (e.g., "RR-CANON-R062")
    """
    code: str = "RULES_VIOLATION"

    def __init__(
        self,
        message: str,
        rule_ref: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        self.rule_ref = rule_ref
        if rule_ref:
            self.context["rule_ref"] = rule_ref


class InvalidStateError(RingRiftError):
    """Corrupted or unexpected game state.

    Raised when the game state is in an invalid configuration that
    should not be possible through normal gameplay.
    """
    code: str = "INVALID_STATE"


class InvalidMoveError(RingRiftError):
    """Move that cannot be applied to current state.

    Raised when a move is structurally valid but cannot be applied
    to the current game state (e.g., wrong player, wrong phase).
    """
    code: str = "INVALID_MOVE"


# =============================================================================
# AI Errors
# =============================================================================


class AIError(RingRiftError):
    """Base class for AI-related errors."""
    code: str = "AI_ERROR"


class AIFallbackError(AIError):
    """AI failed and used fallback move selection.

    Raised when the primary AI strategy fails and a fallback
    (e.g., random move) is used instead.

    Attributes:
        original_error: The exception that triggered the fallback
        fallback_method: Description of fallback used (e.g., "random")
    """
    code: str = "AI_FALLBACK"

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        fallback_method: str = "random",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        self.original_error = original_error
        self.fallback_method = fallback_method
        self.context["fallback_method"] = fallback_method
        if original_error:
            self.context["original_error"] = str(original_error)


class AITimeoutError(AIError):
    """AI search exceeded time limit.

    Raised when AI search takes longer than allocated time.
    """
    code: str = "AI_TIMEOUT"

    def __init__(
        self,
        message: str,
        time_limit_ms: int | None = None,
        actual_time_ms: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if time_limit_ms:
            self.context["time_limit_ms"] = time_limit_ms
        if actual_time_ms:
            self.context["actual_time_ms"] = actual_time_ms


class ModelLoadError(AIError):
    """Failed to load AI model checkpoint.

    Raised when a neural network checkpoint cannot be loaded,
    either due to file issues or architecture mismatch.
    """
    code: str = "MODEL_LOAD_ERROR"

    def __init__(
        self,
        message: str,
        model_path: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if model_path:
            self.context["model_path"] = model_path


class CheckpointIncompatibleError(ModelLoadError):
    """Checkpoint architecture doesn't match current model.

    Raised when attempting to load a checkpoint saved with
    different model architecture parameters.
    """
    code: str = "CHECKPOINT_INCOMPATIBLE"

    def __init__(
        self,
        message: str,
        saved_hash: str | None = None,
        current_hash: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if saved_hash:
            self.context["saved_architecture_hash"] = saved_hash
        if current_hash:
            self.context["current_architecture_hash"] = current_hash


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(RingRiftError):
    """Base class for training-related errors."""
    code: str = "TRAINING_ERROR"


class DatasetError(TrainingError):
    """Error loading or processing training dataset."""
    code: str = "DATASET_ERROR"

    def __init__(
        self,
        message: str,
        dataset_path: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if dataset_path:
            self.context["dataset_path"] = dataset_path


class RegressionDetectedError(TrainingError):
    """Model regression detected during evaluation.

    Raised when a newly trained model performs worse than
    the baseline by a significant margin.
    """
    code: str = "REGRESSION_DETECTED"

    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        baseline_value: float | None = None,
        current_value: float | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if metric_name:
            self.context["metric_name"] = metric_name
        if baseline_value is not None:
            self.context["baseline_value"] = baseline_value
        if current_value is not None:
            self.context["current_value"] = current_value


# =============================================================================
# Infrastructure Errors
# =============================================================================


class InfrastructureError(RingRiftError):
    """Base class for infrastructure-related errors."""
    code: str = "INFRASTRUCTURE_ERROR"


class StorageError(InfrastructureError):
    """Error accessing storage backend (S3, GCS, local)."""
    code: str = "STORAGE_ERROR"


class ClusterError(InfrastructureError):
    """Error in distributed cluster operations."""
    code: str = "CLUSTER_ERROR"


class NoHealthyWorkersError(ClusterError):
    """No healthy workers available in cluster."""
    code: str = "NO_HEALTHY_WORKERS"


class SyncError(InfrastructureError):
    """Error syncing data between nodes."""
    code: str = "SYNC_ERROR"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(RingRiftError):
    """Base class for validation errors."""
    code: str = "VALIDATION_ERROR"


class ConfigurationError(ValidationError):
    """Invalid configuration."""
    code: str = "CONFIGURATION_ERROR"


class ParityError(ValidationError):
    """Parity check failed between implementations.

    Raised when Python and TypeScript implementations produce
    different results for the same input.
    """
    code: str = "PARITY_ERROR"

    def __init__(
        self,
        message: str,
        python_result: Any | None = None,
        typescript_result: Any | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if python_result is not None:
            self.context["python_result"] = str(python_result)
        if typescript_result is not None:
            self.context["typescript_result"] = str(typescript_result)


# =============================================================================
# Retry and Recovery Errors
# =============================================================================


class RetryableError(RingRiftError):
    """Error that can be retried (network issues, transient failures).

    Use this for errors where a retry may succeed, such as:
    - Network timeouts
    - SSH connection drops
    - Temporary resource unavailability
    """
    code: str = "RETRYABLE_ERROR"


class NonRetryableError(RingRiftError):
    """Error that should not be retried.

    Alias for FatalError - use when retry would be futile.
    """
    code: str = "NON_RETRYABLE_ERROR"


# Aliases for backwards compatibility with fault_tolerance.py
FatalError = NonRetryableError
RecoverableError = RetryableError
NonRecoverableError = NonRetryableError


class EmergencyHaltError(RingRiftError):
    """Raised when emergency halt is detected.

    Used to stop all training and selfplay loops when
    a critical issue is detected.
    """
    code: str = "EMERGENCY_HALT"


class SSHError(RetryableError):
    """SSH connection or command execution error.

    Raised when SSH commands fail or connections drop.
    """
    code: str = "SSH_ERROR"

    def __init__(
        self,
        message: str,
        host: str | None = None,
        exit_code: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if host:
            self.context["host"] = host
        if exit_code is not None:
            self.context["exit_code"] = exit_code


class DatabaseError(RingRiftError):
    """Database access error.

    Raised for SQLite, PostgreSQL, or other database failures.
    """
    code: str = "DATABASE_ERROR"

    def __init__(
        self,
        message: str,
        db_path: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if db_path:
            self.context["db_path"] = db_path


# =============================================================================
# Resource Errors
# =============================================================================


class ResourceError(RingRiftError):
    """Base class for resource-related errors.

    Raised when system resources are exhausted or unavailable.
    """
    code: str = "RESOURCE_ERROR"


class OutOfMemoryError(ResourceError):
    """GPU or system memory exhausted."""
    code: str = "OUT_OF_MEMORY"


class DiskSpaceError(ResourceError):
    """Insufficient disk space."""
    code: str = "DISK_SPACE_ERROR"


# =============================================================================
# Data Quality Errors
# =============================================================================


class DataQualityError(TrainingError):
    """Data quality issue detected.

    Raised when training data has quality problems like:
    - NaN/Inf values
    - Corrupted samples
    - Too many duplicates
    """
    code: str = "DATA_QUALITY_ERROR"


class InvalidGameError(DataQualityError):
    """Game data integrity violation.

    Raised when a game cannot be stored or finalized due to data integrity
    issues, such as missing move data. This is a critical safeguard to
    prevent useless games from polluting training databases.
    """
    code: str = "INVALID_GAME_ERROR"

    def __init__(
        self,
        message: str,
        game_id: str | None = None,
        move_count: int | None = None,
        quality_score: float | None = None,
        samples_affected: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if game_id:
            self.context["game_id"] = game_id
        if move_count is not None:
            self.context["move_count"] = move_count
        if quality_score is not None:
            self.context["quality_score"] = quality_score
        if samples_affected is not None:
            self.context["samples_affected"] = samples_affected


class LifecycleError(TrainingError):
    """Model lifecycle error.

    Raised for invalid stage transitions, model not found during
    promotion, or archive/delete failures.
    """
    code: str = "LIFECYCLE_ERROR"

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        current_stage: str | None = None,
        target_stage: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context=context)
        if model_id:
            self.context["model_id"] = model_id
        if current_stage:
            self.context["current_stage"] = current_stage
        if target_stage:
            self.context["target_stage"] = target_stage


class CheckpointError(TrainingError):
    """Checkpoint-related failure.

    Raised when checkpoint operations fail (corrupted file,
    incompatible version, save failure).
    """
    code: str = "CHECKPOINT_ERROR"


class ModelVersioningError(CheckpointError):
    """Model versioning failure.

    Raised when model version operations fail (version mismatch,
    checksum mismatch, legacy checkpoint issues).

    NOTE: Subclasses (VersionMismatchError, ChecksumMismatchError, etc.)
    remain in app.training.model_versioning for specialized handling.
    """
    code: str = "MODEL_VERSIONING_ERROR"


class EvaluationError(TrainingError):
    """Evaluation-related failure.

    Raised when evaluation operations fail (game errors, timeout,
    baseline comparison issues).
    """
    code: str = "EVALUATION_ERROR"


class SelfplayError(TrainingError):
    """Selfplay generation failure.

    Raised when selfplay operations fail (worker crash, data corruption,
    resource exhaustion).
    """
    code: str = "SELFPLAY_ERROR"


class DataLoadError(TrainingError):
    """Data loading failure.

    Raised when data loading operations fail (I/O error, corrupt data,
    invalid format).
    """
    code: str = "DATA_LOAD_ERROR"


# =============================================================================
# Recoverable/Non-Recoverable Training Errors
# =============================================================================

# Aliases for training-specific error patterns
RecoverableError = RetryableError
NonRecoverableError = NonRetryableError
FatalError = NonRetryableError  # More intuitive name for critical failures
