"""Comprehensive tests for the RingRift error hierarchy.

Tests cover:
- Base RingRiftError instantiation and attributes
- String representation formatting
- to_dict() serialization
- Each major error subclass
- Special attributes on subclasses (rule_ref, model_path, host, etc.)
- Inheritance chains
- Aliases (FatalError, RecoverableError, NonRecoverableError)
"""

import pytest

from app.errors import (
    # Base error
    RingRiftError,
    # Game rules errors
    RulesViolationError,
    InvalidStateError,
    InvalidMoveError,
    # AI errors
    AIError,
    AIFallbackError,
    AITimeoutError,
    ModelLoadError,
    CheckpointIncompatibleError,
    # Training errors
    TrainingError,
    DatasetError,
    RegressionDetectedError,
    DataQualityError,
    InvalidGameError,
    LifecycleError,
    CheckpointError,
    ModelVersioningError,
    EvaluationError,
    SelfplayError,
    DataLoadError,
    # Infrastructure errors
    InfrastructureError,
    StorageError,
    ClusterError,
    NoHealthyWorkersError,
    SyncError,
    # Validation errors
    ValidationError,
    ConfigurationError,
    ParityError,
    # Retry/recovery errors
    RetryableError,
    NonRetryableError,
    EmergencyHaltError,
    SSHError,
    DatabaseError,
    # Resource errors
    ResourceError,
    OutOfMemoryError,
    DiskSpaceError,
    # Aliases
    FatalError,
    RecoverableError,
    NonRecoverableError,
)


class TestRingRiftErrorBase:
    """Tests for the base RingRiftError class."""

    def test_basic_instantiation(self):
        """Test basic error instantiation with just a message."""
        error = RingRiftError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code == "RINGRIFT_ERROR"
        assert error.context == {}

    def test_custom_code(self):
        """Test instantiation with a custom error code."""
        error = RingRiftError("Test error", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"
        assert error.message == "Test error"

    def test_context_provided(self):
        """Test instantiation with context dictionary."""
        ctx = {"key": "value", "count": 42}
        error = RingRiftError("Test error", context=ctx)
        assert error.context == {"key": "value", "count": 42}

    def test_context_defaults_to_empty_dict(self):
        """Test that context defaults to empty dict if None."""
        error = RingRiftError("Test", context=None)
        assert error.context == {}

    def test_exception_inheritance(self):
        """Test that RingRiftError inherits from Exception."""
        error = RingRiftError("Test")
        assert isinstance(error, Exception)

    def test_args_set_correctly(self):
        """Test that the exception args are set correctly."""
        error = RingRiftError("Test message")
        assert error.args == ("Test message",)


class TestRingRiftErrorStringRepresentation:
    """Tests for __str__ formatting."""

    def test_str_without_context(self):
        """Test string representation without context."""
        error = RingRiftError("Test message")
        assert str(error) == "[RINGRIFT_ERROR] Test message"

    def test_str_with_context(self):
        """Test string representation with context."""
        error = RingRiftError("Test message", context={"key": "value"})
        assert str(error) == "[RINGRIFT_ERROR] Test message (key=value)"

    def test_str_with_multiple_context_items(self):
        """Test string representation with multiple context items."""
        error = RingRiftError("Error", context={"a": 1, "b": 2})
        result = str(error)
        assert "[RINGRIFT_ERROR] Error" in result
        assert "a=1" in result
        assert "b=2" in result

    def test_str_with_custom_code(self):
        """Test string representation with custom code."""
        error = RingRiftError("Test", code="MY_CODE")
        assert str(error) == "[MY_CODE] Test"


class TestRingRiftErrorToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        error = RingRiftError("Test message")
        d = error.to_dict()
        assert d == {
            "code": "RINGRIFT_ERROR",
            "message": "Test message",
            "context": {},
        }

    def test_to_dict_with_context(self):
        """Test to_dict with context."""
        error = RingRiftError("Test", context={"key": "value"})
        d = error.to_dict()
        assert d["context"] == {"key": "value"}

    def test_to_dict_with_custom_code(self):
        """Test to_dict with custom code."""
        error = RingRiftError("Test", code="CUSTOM")
        d = error.to_dict()
        assert d["code"] == "CUSTOM"


class TestRulesViolationError:
    """Tests for RulesViolationError."""

    def test_default_code(self):
        """Test default error code."""
        error = RulesViolationError("Invalid move")
        assert error.code == "RULES_VIOLATION"

    def test_rule_ref_attribute(self):
        """Test rule_ref attribute is stored."""
        error = RulesViolationError("Invalid move", rule_ref="RR-CANON-R062")
        assert error.rule_ref == "RR-CANON-R062"

    def test_rule_ref_in_context(self):
        """Test rule_ref is added to context."""
        error = RulesViolationError("Invalid move", rule_ref="RR-CANON-R062")
        assert error.context.get("rule_ref") == "RR-CANON-R062"

    def test_rule_ref_none(self):
        """Test rule_ref can be None."""
        error = RulesViolationError("Invalid move")
        assert error.rule_ref is None
        assert "rule_ref" not in error.context

    def test_inheritance(self):
        """Test inheritance chain."""
        error = RulesViolationError("Test")
        assert isinstance(error, RingRiftError)

    def test_with_additional_context(self):
        """Test rule_ref combined with other context."""
        error = RulesViolationError(
            "Invalid move",
            rule_ref="RR-CANON-R062",
            context={"player": 1, "move": "A1"},
        )
        assert error.context["rule_ref"] == "RR-CANON-R062"
        assert error.context["player"] == 1
        assert error.context["move"] == "A1"


class TestInvalidStateError:
    """Tests for InvalidStateError."""

    def test_default_code(self):
        error = InvalidStateError("Corrupted state")
        assert error.code == "INVALID_STATE"

    def test_inheritance(self):
        error = InvalidStateError("Test")
        assert isinstance(error, RingRiftError)


class TestInvalidMoveError:
    """Tests for InvalidMoveError."""

    def test_default_code(self):
        error = InvalidMoveError("Wrong player turn")
        assert error.code == "INVALID_MOVE"

    def test_inheritance(self):
        error = InvalidMoveError("Test")
        assert isinstance(error, RingRiftError)


class TestAIError:
    """Tests for AIError base class."""

    def test_default_code(self):
        error = AIError("AI computation failed")
        assert error.code == "AI_ERROR"

    def test_inheritance(self):
        error = AIError("Test")
        assert isinstance(error, RingRiftError)


class TestAIFallbackError:
    """Tests for AIFallbackError."""

    def test_default_code(self):
        error = AIFallbackError("Using fallback")
        assert error.code == "AI_FALLBACK"

    def test_original_error_attribute(self):
        """Test original_error attribute is stored."""
        original = ValueError("original problem")
        error = AIFallbackError("Fallback used", original_error=original)
        assert error.original_error is original

    def test_fallback_method_default(self):
        """Test fallback_method defaults to 'random'."""
        error = AIFallbackError("Fallback used")
        assert error.fallback_method == "random"

    def test_fallback_method_custom(self):
        """Test custom fallback_method."""
        error = AIFallbackError("Fallback used", fallback_method="heuristic")
        assert error.fallback_method == "heuristic"

    def test_fallback_method_in_context(self):
        """Test fallback_method is added to context."""
        error = AIFallbackError("Fallback used", fallback_method="greedy")
        assert error.context["fallback_method"] == "greedy"

    def test_original_error_in_context(self):
        """Test original_error is added to context as string."""
        original = RuntimeError("computation failed")
        error = AIFallbackError("Fallback used", original_error=original)
        assert "computation failed" in error.context["original_error"]

    def test_inheritance(self):
        error = AIFallbackError("Test")
        assert isinstance(error, AIError)
        assert isinstance(error, RingRiftError)


class TestAITimeoutError:
    """Tests for AITimeoutError."""

    def test_default_code(self):
        error = AITimeoutError("Search timed out")
        assert error.code == "AI_TIMEOUT"

    def test_time_limit_in_context(self):
        """Test time_limit_ms is added to context."""
        error = AITimeoutError("Timeout", time_limit_ms=1000)
        assert error.context["time_limit_ms"] == 1000

    def test_actual_time_in_context(self):
        """Test actual_time_ms is added to context."""
        error = AITimeoutError("Timeout", actual_time_ms=1500)
        assert error.context["actual_time_ms"] == 1500

    def test_both_times_in_context(self):
        """Test both time values in context."""
        error = AITimeoutError("Timeout", time_limit_ms=1000, actual_time_ms=1500)
        assert error.context["time_limit_ms"] == 1000
        assert error.context["actual_time_ms"] == 1500

    def test_inheritance(self):
        error = AITimeoutError("Test")
        assert isinstance(error, AIError)


class TestModelLoadError:
    """Tests for ModelLoadError."""

    def test_default_code(self):
        error = ModelLoadError("Failed to load model")
        assert error.code == "MODEL_LOAD_ERROR"

    def test_model_path_in_context(self):
        """Test model_path is added to context."""
        error = ModelLoadError("Failed", model_path="/path/to/model.pth")
        assert error.context["model_path"] == "/path/to/model.pth"

    def test_model_path_none(self):
        """Test model_path can be None."""
        error = ModelLoadError("Failed")
        assert "model_path" not in error.context

    def test_inheritance(self):
        error = ModelLoadError("Test")
        assert isinstance(error, AIError)


class TestCheckpointIncompatibleError:
    """Tests for CheckpointIncompatibleError."""

    def test_default_code(self):
        error = CheckpointIncompatibleError("Architecture mismatch")
        assert error.code == "CHECKPOINT_INCOMPATIBLE"

    def test_saved_hash_in_context(self):
        """Test saved_hash is added to context."""
        error = CheckpointIncompatibleError("Mismatch", saved_hash="abc123")
        assert error.context["saved_architecture_hash"] == "abc123"

    def test_current_hash_in_context(self):
        """Test current_hash is added to context."""
        error = CheckpointIncompatibleError("Mismatch", current_hash="def456")
        assert error.context["current_architecture_hash"] == "def456"

    def test_both_hashes_in_context(self):
        """Test both hashes in context."""
        error = CheckpointIncompatibleError(
            "Mismatch", saved_hash="abc123", current_hash="def456"
        )
        assert error.context["saved_architecture_hash"] == "abc123"
        assert error.context["current_architecture_hash"] == "def456"

    def test_inheritance(self):
        """Test inherits from ModelLoadError."""
        error = CheckpointIncompatibleError("Test")
        assert isinstance(error, ModelLoadError)
        assert isinstance(error, AIError)


class TestTrainingError:
    """Tests for TrainingError base class."""

    def test_default_code(self):
        error = TrainingError("Training failed")
        assert error.code == "TRAINING_ERROR"

    def test_inheritance(self):
        error = TrainingError("Test")
        assert isinstance(error, RingRiftError)


class TestDatasetError:
    """Tests for DatasetError."""

    def test_default_code(self):
        error = DatasetError("Dataset loading failed")
        assert error.code == "DATASET_ERROR"

    def test_dataset_path_in_context(self):
        """Test dataset_path is added to context."""
        error = DatasetError("Failed", dataset_path="/data/train.npz")
        assert error.context["dataset_path"] == "/data/train.npz"

    def test_inheritance(self):
        error = DatasetError("Test")
        assert isinstance(error, TrainingError)


class TestRegressionDetectedError:
    """Tests for RegressionDetectedError."""

    def test_default_code(self):
        error = RegressionDetectedError("Model regressed")
        assert error.code == "REGRESSION_DETECTED"

    def test_metric_name_in_context(self):
        """Test metric_name is added to context."""
        error = RegressionDetectedError("Regressed", metric_name="win_rate")
        assert error.context["metric_name"] == "win_rate"

    def test_baseline_value_in_context(self):
        """Test baseline_value is added to context."""
        error = RegressionDetectedError("Regressed", baseline_value=0.85)
        assert error.context["baseline_value"] == 0.85

    def test_current_value_in_context(self):
        """Test current_value is added to context."""
        error = RegressionDetectedError("Regressed", current_value=0.75)
        assert error.context["current_value"] == 0.75

    def test_all_metrics_in_context(self):
        """Test all metric values in context."""
        error = RegressionDetectedError(
            "Regressed",
            metric_name="win_rate",
            baseline_value=0.85,
            current_value=0.75,
        )
        assert error.context["metric_name"] == "win_rate"
        assert error.context["baseline_value"] == 0.85
        assert error.context["current_value"] == 0.75

    def test_zero_values_included(self):
        """Test that zero values are included in context."""
        error = RegressionDetectedError("Regressed", baseline_value=0.0, current_value=0.0)
        assert error.context["baseline_value"] == 0.0
        assert error.context["current_value"] == 0.0

    def test_inheritance(self):
        error = RegressionDetectedError("Test")
        assert isinstance(error, TrainingError)


class TestInfrastructureError:
    """Tests for InfrastructureError base class."""

    def test_default_code(self):
        error = InfrastructureError("Infrastructure failure")
        assert error.code == "INFRASTRUCTURE_ERROR"

    def test_inheritance(self):
        error = InfrastructureError("Test")
        assert isinstance(error, RingRiftError)


class TestStorageError:
    """Tests for StorageError."""

    def test_default_code(self):
        error = StorageError("S3 access failed")
        assert error.code == "STORAGE_ERROR"

    def test_inheritance(self):
        error = StorageError("Test")
        assert isinstance(error, InfrastructureError)


class TestClusterError:
    """Tests for ClusterError."""

    def test_default_code(self):
        error = ClusterError("Cluster operation failed")
        assert error.code == "CLUSTER_ERROR"

    def test_inheritance(self):
        error = ClusterError("Test")
        assert isinstance(error, InfrastructureError)


class TestNoHealthyWorkersError:
    """Tests for NoHealthyWorkersError."""

    def test_default_code(self):
        error = NoHealthyWorkersError("No workers available")
        assert error.code == "NO_HEALTHY_WORKERS"

    def test_inheritance(self):
        """Test inherits from ClusterError."""
        error = NoHealthyWorkersError("Test")
        assert isinstance(error, ClusterError)
        assert isinstance(error, InfrastructureError)


class TestSyncError:
    """Tests for SyncError."""

    def test_default_code(self):
        error = SyncError("Sync failed")
        assert error.code == "SYNC_ERROR"

    def test_inheritance(self):
        error = SyncError("Test")
        assert isinstance(error, InfrastructureError)


class TestValidationError:
    """Tests for ValidationError base class."""

    def test_default_code(self):
        error = ValidationError("Validation failed")
        assert error.code == "VALIDATION_ERROR"

    def test_inheritance(self):
        error = ValidationError("Test")
        assert isinstance(error, RingRiftError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_default_code(self):
        error = ConfigurationError("Invalid config")
        assert error.code == "CONFIGURATION_ERROR"

    def test_inheritance(self):
        error = ConfigurationError("Test")
        assert isinstance(error, ValidationError)


class TestParityError:
    """Tests for ParityError."""

    def test_default_code(self):
        error = ParityError("Parity check failed")
        assert error.code == "PARITY_ERROR"

    def test_python_result_in_context(self):
        """Test python_result is added to context."""
        error = ParityError("Mismatch", python_result=[1, 2, 3])
        assert error.context["python_result"] == "[1, 2, 3]"

    def test_typescript_result_in_context(self):
        """Test typescript_result is added to context."""
        error = ParityError("Mismatch", typescript_result={"key": "value"})
        assert error.context["typescript_result"] == "{'key': 'value'}"

    def test_both_results_in_context(self):
        """Test both results in context."""
        error = ParityError("Mismatch", python_result=42, typescript_result=43)
        assert error.context["python_result"] == "42"
        assert error.context["typescript_result"] == "43"

    def test_inheritance(self):
        error = ParityError("Test")
        assert isinstance(error, ValidationError)


class TestRetryableError:
    """Tests for RetryableError."""

    def test_default_code(self):
        error = RetryableError("Network timeout")
        assert error.code == "RETRYABLE_ERROR"

    def test_inheritance(self):
        error = RetryableError("Test")
        assert isinstance(error, RingRiftError)


class TestNonRetryableError:
    """Tests for NonRetryableError."""

    def test_default_code(self):
        error = NonRetryableError("Fatal error")
        assert error.code == "NON_RETRYABLE_ERROR"

    def test_inheritance(self):
        error = NonRetryableError("Test")
        assert isinstance(error, RingRiftError)


class TestEmergencyHaltError:
    """Tests for EmergencyHaltError."""

    def test_default_code(self):
        error = EmergencyHaltError("Emergency halt triggered")
        assert error.code == "EMERGENCY_HALT"

    def test_inheritance(self):
        error = EmergencyHaltError("Test")
        assert isinstance(error, RingRiftError)


class TestSSHError:
    """Tests for SSHError."""

    def test_default_code(self):
        error = SSHError("SSH connection failed")
        assert error.code == "SSH_ERROR"

    def test_host_in_context(self):
        """Test host is added to context."""
        error = SSHError("Connection failed", host="worker-1")
        assert error.context["host"] == "worker-1"

    def test_exit_code_in_context(self):
        """Test exit_code is added to context."""
        error = SSHError("Command failed", exit_code=127)
        assert error.context["exit_code"] == 127

    def test_exit_code_zero_included(self):
        """Test that exit_code=0 is included."""
        error = SSHError("Unexpected success", exit_code=0)
        assert error.context["exit_code"] == 0

    def test_both_in_context(self):
        """Test both host and exit_code in context."""
        error = SSHError("Failed", host="node-1", exit_code=1)
        assert error.context["host"] == "node-1"
        assert error.context["exit_code"] == 1

    def test_inheritance(self):
        """Test inherits from RetryableError."""
        error = SSHError("Test")
        assert isinstance(error, RetryableError)


class TestDatabaseError:
    """Tests for DatabaseError."""

    def test_default_code(self):
        error = DatabaseError("Database access failed")
        assert error.code == "DATABASE_ERROR"

    def test_db_path_in_context(self):
        """Test db_path is added to context."""
        error = DatabaseError("Failed", db_path="/data/games.db")
        assert error.context["db_path"] == "/data/games.db"

    def test_inheritance(self):
        error = DatabaseError("Test")
        assert isinstance(error, RingRiftError)


class TestResourceError:
    """Tests for ResourceError base class."""

    def test_default_code(self):
        error = ResourceError("Resource exhausted")
        assert error.code == "RESOURCE_ERROR"

    def test_inheritance(self):
        error = ResourceError("Test")
        assert isinstance(error, RingRiftError)


class TestOutOfMemoryError:
    """Tests for OutOfMemoryError."""

    def test_default_code(self):
        error = OutOfMemoryError("GPU OOM")
        assert error.code == "OUT_OF_MEMORY"

    def test_inheritance(self):
        error = OutOfMemoryError("Test")
        assert isinstance(error, ResourceError)


class TestDiskSpaceError:
    """Tests for DiskSpaceError."""

    def test_default_code(self):
        error = DiskSpaceError("Disk full")
        assert error.code == "DISK_SPACE_ERROR"

    def test_inheritance(self):
        error = DiskSpaceError("Test")
        assert isinstance(error, ResourceError)


class TestDataQualityError:
    """Tests for DataQualityError."""

    def test_default_code(self):
        error = DataQualityError("Data quality issue")
        assert error.code == "DATA_QUALITY_ERROR"

    def test_inheritance(self):
        error = DataQualityError("Test")
        assert isinstance(error, TrainingError)


class TestInvalidGameError:
    """Tests for InvalidGameError."""

    def test_default_code(self):
        error = InvalidGameError("Game data integrity violation")
        assert error.code == "INVALID_GAME_ERROR"

    def test_game_id_in_context(self):
        """Test game_id is added to context."""
        error = InvalidGameError("Invalid", game_id="game-123")
        assert error.context["game_id"] == "game-123"

    def test_move_count_in_context(self):
        """Test move_count is added to context."""
        error = InvalidGameError("Invalid", move_count=0)
        assert error.context["move_count"] == 0

    def test_quality_score_in_context(self):
        """Test quality_score is added to context."""
        error = InvalidGameError("Invalid", quality_score=0.3)
        assert error.context["quality_score"] == 0.3

    def test_samples_affected_in_context(self):
        """Test samples_affected is added to context."""
        error = InvalidGameError("Invalid", samples_affected=50)
        assert error.context["samples_affected"] == 50

    def test_all_attributes_in_context(self):
        """Test all optional attributes in context."""
        error = InvalidGameError(
            "Invalid",
            game_id="game-456",
            move_count=10,
            quality_score=0.5,
            samples_affected=100,
        )
        assert error.context["game_id"] == "game-456"
        assert error.context["move_count"] == 10
        assert error.context["quality_score"] == 0.5
        assert error.context["samples_affected"] == 100

    def test_inheritance(self):
        """Test inherits from DataQualityError."""
        error = InvalidGameError("Test")
        assert isinstance(error, DataQualityError)
        assert isinstance(error, TrainingError)


class TestLifecycleError:
    """Tests for LifecycleError."""

    def test_default_code(self):
        error = LifecycleError("Invalid stage transition")
        assert error.code == "LIFECYCLE_ERROR"

    def test_model_id_in_context(self):
        """Test model_id is added to context."""
        error = LifecycleError("Invalid", model_id="model-v1")
        assert error.context["model_id"] == "model-v1"

    def test_current_stage_in_context(self):
        """Test current_stage is added to context."""
        error = LifecycleError("Invalid", current_stage="training")
        assert error.context["current_stage"] == "training"

    def test_target_stage_in_context(self):
        """Test target_stage is added to context."""
        error = LifecycleError("Invalid", target_stage="production")
        assert error.context["target_stage"] == "production"

    def test_all_attributes_in_context(self):
        """Test all stage attributes in context."""
        error = LifecycleError(
            "Invalid transition",
            model_id="model-v2",
            current_stage="evaluation",
            target_stage="production",
        )
        assert error.context["model_id"] == "model-v2"
        assert error.context["current_stage"] == "evaluation"
        assert error.context["target_stage"] == "production"

    def test_inheritance(self):
        error = LifecycleError("Test")
        assert isinstance(error, TrainingError)


class TestCheckpointError:
    """Tests for CheckpointError."""

    def test_default_code(self):
        error = CheckpointError("Checkpoint corrupted")
        assert error.code == "CHECKPOINT_ERROR"

    def test_inheritance(self):
        error = CheckpointError("Test")
        assert isinstance(error, TrainingError)


class TestModelVersioningError:
    """Tests for ModelVersioningError."""

    def test_default_code(self):
        error = ModelVersioningError("Version mismatch")
        assert error.code == "MODEL_VERSIONING_ERROR"

    def test_inheritance(self):
        """Test inherits from CheckpointError."""
        error = ModelVersioningError("Test")
        assert isinstance(error, CheckpointError)
        assert isinstance(error, TrainingError)


class TestEvaluationError:
    """Tests for EvaluationError."""

    def test_default_code(self):
        error = EvaluationError("Evaluation failed")
        assert error.code == "EVALUATION_ERROR"

    def test_inheritance(self):
        error = EvaluationError("Test")
        assert isinstance(error, TrainingError)


class TestSelfplayError:
    """Tests for SelfplayError."""

    def test_default_code(self):
        error = SelfplayError("Selfplay generation failed")
        assert error.code == "SELFPLAY_ERROR"

    def test_inheritance(self):
        error = SelfplayError("Test")
        assert isinstance(error, TrainingError)


class TestDataLoadError:
    """Tests for DataLoadError."""

    def test_default_code(self):
        error = DataLoadError("Data loading failed")
        assert error.code == "DATA_LOAD_ERROR"

    def test_inheritance(self):
        error = DataLoadError("Test")
        assert isinstance(error, TrainingError)


class TestErrorAliases:
    """Tests for backward-compatible error aliases."""

    def test_fatal_error_alias(self):
        """Test FatalError is alias for NonRetryableError."""
        assert FatalError is NonRetryableError
        error = FatalError("Fatal")
        assert error.code == "NON_RETRYABLE_ERROR"

    def test_recoverable_error_alias(self):
        """Test RecoverableError is alias for RetryableError."""
        assert RecoverableError is RetryableError
        error = RecoverableError("Recoverable")
        assert error.code == "RETRYABLE_ERROR"

    def test_non_recoverable_error_alias(self):
        """Test NonRecoverableError is alias for NonRetryableError."""
        assert NonRecoverableError is NonRetryableError


class TestInheritanceChains:
    """Tests to verify correct inheritance chains."""

    def test_ai_error_chain(self):
        """Test AI error inheritance chain."""
        # AIError -> RingRiftError
        assert issubclass(AIError, RingRiftError)
        # AIFallbackError -> AIError
        assert issubclass(AIFallbackError, AIError)
        # AITimeoutError -> AIError
        assert issubclass(AITimeoutError, AIError)
        # ModelLoadError -> AIError
        assert issubclass(ModelLoadError, AIError)
        # CheckpointIncompatibleError -> ModelLoadError -> AIError
        assert issubclass(CheckpointIncompatibleError, ModelLoadError)
        assert issubclass(CheckpointIncompatibleError, AIError)

    def test_training_error_chain(self):
        """Test training error inheritance chain."""
        # TrainingError -> RingRiftError
        assert issubclass(TrainingError, RingRiftError)
        # DatasetError -> TrainingError
        assert issubclass(DatasetError, TrainingError)
        # RegressionDetectedError -> TrainingError
        assert issubclass(RegressionDetectedError, TrainingError)
        # DataQualityError -> TrainingError
        assert issubclass(DataQualityError, TrainingError)
        # InvalidGameError -> DataQualityError -> TrainingError
        assert issubclass(InvalidGameError, DataQualityError)
        # LifecycleError -> TrainingError
        assert issubclass(LifecycleError, TrainingError)
        # CheckpointError -> TrainingError
        assert issubclass(CheckpointError, TrainingError)
        # ModelVersioningError -> CheckpointError -> TrainingError
        assert issubclass(ModelVersioningError, CheckpointError)
        # EvaluationError -> TrainingError
        assert issubclass(EvaluationError, TrainingError)
        # SelfplayError -> TrainingError
        assert issubclass(SelfplayError, TrainingError)
        # DataLoadError -> TrainingError
        assert issubclass(DataLoadError, TrainingError)

    def test_infrastructure_error_chain(self):
        """Test infrastructure error inheritance chain."""
        # InfrastructureError -> RingRiftError
        assert issubclass(InfrastructureError, RingRiftError)
        # StorageError -> InfrastructureError
        assert issubclass(StorageError, InfrastructureError)
        # ClusterError -> InfrastructureError
        assert issubclass(ClusterError, InfrastructureError)
        # NoHealthyWorkersError -> ClusterError -> InfrastructureError
        assert issubclass(NoHealthyWorkersError, ClusterError)
        # SyncError -> InfrastructureError
        assert issubclass(SyncError, InfrastructureError)

    def test_validation_error_chain(self):
        """Test validation error inheritance chain."""
        # ValidationError -> RingRiftError
        assert issubclass(ValidationError, RingRiftError)
        # ConfigurationError -> ValidationError
        assert issubclass(ConfigurationError, ValidationError)
        # ParityError -> ValidationError
        assert issubclass(ParityError, ValidationError)

    def test_resource_error_chain(self):
        """Test resource error inheritance chain."""
        # ResourceError -> RingRiftError
        assert issubclass(ResourceError, RingRiftError)
        # OutOfMemoryError -> ResourceError
        assert issubclass(OutOfMemoryError, ResourceError)
        # DiskSpaceError -> ResourceError
        assert issubclass(DiskSpaceError, ResourceError)

    def test_ssh_error_chain(self):
        """Test SSH error inheritance chain."""
        # SSHError -> RetryableError -> RingRiftError
        assert issubclass(SSHError, RetryableError)
        assert issubclass(SSHError, RingRiftError)


class TestErrorCatching:
    """Tests to verify errors can be caught at various levels."""

    def test_catch_by_base_class(self):
        """Test catching errors by base class."""
        with pytest.raises(RingRiftError):
            raise AIError("Test")

        with pytest.raises(RingRiftError):
            raise TrainingError("Test")

    def test_catch_ai_errors(self):
        """Test catching all AI errors with AIError."""
        with pytest.raises(AIError):
            raise AIFallbackError("Test")

        with pytest.raises(AIError):
            raise AITimeoutError("Test")

        with pytest.raises(AIError):
            raise ModelLoadError("Test")

    def test_catch_training_errors(self):
        """Test catching all training errors with TrainingError."""
        with pytest.raises(TrainingError):
            raise DatasetError("Test")

        with pytest.raises(TrainingError):
            raise RegressionDetectedError("Test")

        with pytest.raises(TrainingError):
            raise InvalidGameError("Test")

    def test_catch_infrastructure_errors(self):
        """Test catching all infrastructure errors."""
        with pytest.raises(InfrastructureError):
            raise StorageError("Test")

        with pytest.raises(InfrastructureError):
            raise ClusterError("Test")

        with pytest.raises(InfrastructureError):
            raise NoHealthyWorkersError("Test")

    def test_catch_retryable_errors(self):
        """Test catching retryable errors including SSHError."""
        with pytest.raises(RetryableError):
            raise SSHError("Test")

        with pytest.raises(RecoverableError):  # Alias
            raise SSHError("Test")
