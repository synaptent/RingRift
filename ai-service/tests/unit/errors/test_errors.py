"""Unit tests for app.errors module.

Tests the unified error handling infrastructure for RingRift AI Service.
"""

import pytest

from app.errors import (
    # Codes
    ErrorCode,
    # Base
    RingRiftError,
    # Resource
    ResourceError,
    GPUError,
    GPUOutOfMemoryError,
    DiskError,
    DiskSpaceError,
    MemoryExhaustedError,
    # Network
    NetworkError,
    ConnectionError,
    ConnectionTimeoutError,
    SSHError,
    SSHAuthError,
    HTTPError,
    P2PError,
    P2PLeaderUnavailableError,
    # Sync
    SyncError,
    SyncTimeoutError,
    SyncConflictError,
    SyncIntegrityError,
    # Training
    TrainingError,
    DataQualityError,
    InvalidGameError,
    ModelLoadError,
    CheckpointCorruptError,
    ConvergenceError,
    ModelVersioningError,
    # Daemon
    DaemonError,
    DaemonStartupError,
    DaemonCrashError,
    DaemonConfigError,
    DaemonDependencyError,
    # Validation
    ValidationError,
    SchemaError,
    ParityError,
    # Configuration
    ConfigurationError,
    ConfigMissingError,
    ConfigTypeError,
    # System
    EmergencyHaltError,
    RetryableError,
    NonRetryableError,
    # Aliases
    FatalError,
    RecoverableError,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_resource_error_codes(self):
        """Test resource-related error codes (1xx)."""
        assert ErrorCode.RESOURCE_EXHAUSTED.value == 100
        assert ErrorCode.GPU_NOT_AVAILABLE.value == 101
        assert ErrorCode.GPU_OOM.value == 102
        assert ErrorCode.DISK_FULL.value == 103
        assert ErrorCode.MEMORY_OOM.value == 104

    def test_network_error_codes(self):
        """Test network-related error codes (2xx)."""
        assert ErrorCode.CONNECTION_FAILED.value == 200
        assert ErrorCode.CONNECTION_TIMEOUT.value == 201
        assert ErrorCode.SSH_AUTH_FAILED.value == 202
        assert ErrorCode.SSH_COMMAND_FAILED.value == 203
        assert ErrorCode.HTTP_CLIENT_ERROR.value == 204
        assert ErrorCode.HTTP_SERVER_ERROR.value == 205
        assert ErrorCode.P2P_MESH_ERROR.value == 206
        assert ErrorCode.P2P_LEADER_UNAVAILABLE.value == 207

    def test_sync_error_codes(self):
        """Test sync-related error codes (3xx)."""
        assert ErrorCode.SYNC_TIMEOUT.value == 300
        assert ErrorCode.SYNC_CONFLICT.value == 301
        assert ErrorCode.SYNC_INTEGRITY_FAILED.value == 302
        assert ErrorCode.SYNC_MANIFEST_MISMATCH.value == 303

    def test_training_error_codes(self):
        """Test training-related error codes (4xx)."""
        assert ErrorCode.DATA_QUALITY_LOW.value == 400
        assert ErrorCode.MODEL_LOAD_FAILED.value == 401
        assert ErrorCode.CONVERGENCE_FAILED.value == 402
        assert ErrorCode.CHECKPOINT_CORRUPT.value == 403
        assert ErrorCode.TRAINING_INTERRUPTED.value == 404

    def test_daemon_error_codes(self):
        """Test daemon-related error codes (5xx)."""
        assert ErrorCode.DAEMON_START_FAILED.value == 500
        assert ErrorCode.DAEMON_CRASHED.value == 501
        assert ErrorCode.DAEMON_CONFIG_INVALID.value == 502
        assert ErrorCode.DAEMON_DEPENDENCY_FAILED.value == 503

    def test_validation_error_codes(self):
        """Test validation-related error codes (6xx)."""
        assert ErrorCode.VALIDATION_FAILED.value == 600
        assert ErrorCode.SCHEMA_MISMATCH.value == 601
        assert ErrorCode.PARITY_FAILED.value == 602

    def test_config_error_codes(self):
        """Test configuration-related error codes (7xx)."""
        assert ErrorCode.CONFIG_MISSING.value == 700
        assert ErrorCode.CONFIG_INVALID.value == 701
        assert ErrorCode.CONFIG_TYPE_ERROR.value == 702

    def test_system_error_codes(self):
        """Test system-related error codes (8xx)."""
        assert ErrorCode.EMERGENCY_HALT.value == 800
        assert ErrorCode.RETRYABLE.value == 801
        assert ErrorCode.NON_RETRYABLE.value == 802

    def test_unknown_code(self):
        """Test unknown error code."""
        assert ErrorCode.UNKNOWN.value == 999


class TestRingRiftError:
    """Tests for RingRiftError base class."""

    def test_basic_creation(self):
        """Test basic error creation."""
        err = RingRiftError("Something went wrong")
        assert err.message == "Something went wrong"
        assert err.code == ErrorCode.UNKNOWN
        assert err.details == {}
        assert err.retryable is False

    def test_with_code(self):
        """Test error with specific code."""
        err = RingRiftError("GPU error", code=ErrorCode.GPU_NOT_AVAILABLE)
        assert err.code == ErrorCode.GPU_NOT_AVAILABLE

    def test_with_details(self):
        """Test error with details."""
        err = RingRiftError("Failed", details={"host": "node-1", "attempt": 3})
        assert err.details == {"host": "node-1", "attempt": 3}

    def test_with_retryable(self):
        """Test error with retryable flag."""
        err = RingRiftError("Temp failure", retryable=True)
        assert err.retryable is True

    def test_str_basic(self):
        """Test __str__ for basic error."""
        err = RingRiftError("Something went wrong")
        assert str(err) == "Something went wrong"

    def test_str_with_code(self):
        """Test __str__ includes code name."""
        err = RingRiftError("GPU failed", code=ErrorCode.GPU_NOT_AVAILABLE)
        assert "[GPU_NOT_AVAILABLE]" in str(err)

    def test_str_with_details(self):
        """Test __str__ includes details."""
        err = RingRiftError("Failed", details={"host": "node-1"})
        assert "(host=node-1)" in str(err)

    def test_to_dict(self):
        """Test to_dict serialization."""
        err = RingRiftError(
            "Test error",
            code=ErrorCode.SYNC_TIMEOUT,
            details={"host": "node-1"},
            retryable=True,
        )
        d = err.to_dict()
        assert d["error"] == "RingRiftError"
        assert d["message"] == "Test error"
        assert d["code"] == 300
        assert d["code_name"] == "SYNC_TIMEOUT"
        assert d["retryable"] is True
        assert d["details"] == {"host": "node-1"}

    def test_is_exception(self):
        """Test that RingRiftError is an Exception."""
        err = RingRiftError("Test")
        assert isinstance(err, Exception)
        with pytest.raises(RingRiftError):
            raise err


class TestResourceErrors:
    """Tests for resource-related errors."""

    def test_resource_error_defaults(self):
        """Test ResourceError default values."""
        err = ResourceError("Resource exhausted")
        assert err.code == ErrorCode.RESOURCE_EXHAUSTED
        assert err.retryable is True  # Resources often free up

    def test_gpu_error(self):
        """Test GPUError."""
        err = GPUError("No GPU available")
        assert err.code == ErrorCode.GPU_NOT_AVAILABLE
        assert err.retryable is True  # Inherited from ResourceError

    def test_gpu_oom_error(self):
        """Test GPUOutOfMemoryError."""
        err = GPUOutOfMemoryError("CUDA OOM")
        assert err.code == ErrorCode.GPU_OOM

    def test_disk_error(self):
        """Test DiskError."""
        err = DiskError("Disk full")
        assert err.code == ErrorCode.DISK_FULL

    def test_disk_space_error(self):
        """Test DiskSpaceError with extra fields."""
        err = DiskSpaceError(
            "Insufficient space",
            path="/data",
            available_bytes=1000000,
            required_bytes=5000000,
        )
        assert err.path == "/data"
        assert err.available_bytes == 1000000
        assert err.required_bytes == 5000000
        assert err.details["path"] == "/data"
        assert err.details["available_bytes"] == 1000000
        assert err.details["required_bytes"] == 5000000

    def test_disk_space_error_with_context(self):
        """Test DiskSpaceError with additional context."""
        err = DiskSpaceError(
            "Insufficient space",
            path="/data",
            context={"operation": "export"},
        )
        assert err.details["path"] == "/data"
        assert err.details["operation"] == "export"

    def test_memory_exhausted_error(self):
        """Test MemoryExhaustedError."""
        err = MemoryExhaustedError("OOM")
        assert err.code == ErrorCode.MEMORY_OOM


class TestNetworkErrors:
    """Tests for network-related errors."""

    def test_network_error_defaults(self):
        """Test NetworkError default values."""
        err = NetworkError("Connection failed")
        assert err.code == ErrorCode.CONNECTION_FAILED
        assert err.retryable is True

    def test_connection_error(self):
        """Test ConnectionError."""
        err = ConnectionError("Cannot connect")
        assert err.code == ErrorCode.CONNECTION_FAILED

    def test_connection_timeout_error(self):
        """Test ConnectionTimeoutError."""
        err = ConnectionTimeoutError("Timed out")
        assert err.code == ErrorCode.CONNECTION_TIMEOUT

    def test_ssh_error(self):
        """Test SSHError."""
        err = SSHError("SSH command failed")
        assert err.code == ErrorCode.SSH_COMMAND_FAILED

    def test_ssh_auth_error(self):
        """Test SSHAuthError."""
        err = SSHAuthError("Auth failed")
        assert err.code == ErrorCode.SSH_AUTH_FAILED
        assert err.retryable is False  # Auth errors need manual fix

    def test_http_error_client(self):
        """Test HTTPError for client errors."""
        err = HTTPError("Not found", status_code=404)
        assert err.status_code == 404
        assert err.code == ErrorCode.HTTP_CLIENT_ERROR
        assert err.details["status_code"] == 404

    def test_http_error_server(self):
        """Test HTTPError for server errors."""
        err = HTTPError("Server error", status_code=500)
        assert err.code == ErrorCode.HTTP_SERVER_ERROR

    def test_http_error_502(self):
        """Test HTTPError for 502 (server error)."""
        err = HTTPError("Bad gateway", status_code=502)
        assert err.code == ErrorCode.HTTP_SERVER_ERROR

    def test_http_error_no_status(self):
        """Test HTTPError without status code."""
        err = HTTPError("Unknown HTTP error")
        assert err.status_code is None
        assert err.code == ErrorCode.HTTP_CLIENT_ERROR

    def test_p2p_error(self):
        """Test P2PError."""
        err = P2PError("Mesh error")
        assert err.code == ErrorCode.P2P_MESH_ERROR

    def test_p2p_leader_unavailable_error(self):
        """Test P2PLeaderUnavailableError."""
        err = P2PLeaderUnavailableError("No leader")
        assert err.code == ErrorCode.P2P_LEADER_UNAVAILABLE


class TestSyncErrors:
    """Tests for sync-related errors."""

    def test_sync_error_defaults(self):
        """Test SyncError default values."""
        err = SyncError("Sync failed")
        assert err.code == ErrorCode.SYNC_TIMEOUT
        assert err.retryable is True

    def test_sync_timeout_error(self):
        """Test SyncTimeoutError."""
        err = SyncTimeoutError("Timed out")
        assert err.code == ErrorCode.SYNC_TIMEOUT

    def test_sync_conflict_error(self):
        """Test SyncConflictError."""
        err = SyncConflictError("Conflict")
        assert err.code == ErrorCode.SYNC_CONFLICT
        assert err.retryable is False  # Needs manual resolution

    def test_sync_integrity_error(self):
        """Test SyncIntegrityError."""
        err = SyncIntegrityError("Checksum mismatch")
        assert err.code == ErrorCode.SYNC_INTEGRITY_FAILED
        assert err.retryable is False  # Data is corrupted


class TestTrainingErrors:
    """Tests for training-related errors."""

    def test_training_error_defaults(self):
        """Test TrainingError default values."""
        err = TrainingError("Training failed")
        assert err.code == ErrorCode.TRAINING_INTERRUPTED
        assert err.retryable is False

    def test_data_quality_error(self):
        """Test DataQualityError."""
        err = DataQualityError("Quality too low")
        assert err.code == ErrorCode.DATA_QUALITY_LOW

    def test_invalid_game_error(self):
        """Test InvalidGameError with extra fields."""
        err = InvalidGameError(
            "Game has no moves",
            game_id="abc-123",
            move_count=0,
        )
        assert err.game_id == "abc-123"
        assert err.move_count == 0
        assert err.details["game_id"] == "abc-123"
        assert err.details["move_count"] == 0

    def test_invalid_game_error_minimal(self):
        """Test InvalidGameError without extra fields."""
        err = InvalidGameError("Game invalid")
        assert err.game_id is None
        assert err.move_count is None

    def test_model_load_error(self):
        """Test ModelLoadError."""
        err = ModelLoadError("Failed to load")
        assert err.code == ErrorCode.MODEL_LOAD_FAILED

    def test_checkpoint_corrupt_error(self):
        """Test CheckpointCorruptError."""
        err = CheckpointCorruptError("Corrupt checkpoint")
        assert err.code == ErrorCode.CHECKPOINT_CORRUPT

    def test_convergence_error(self):
        """Test ConvergenceError."""
        err = ConvergenceError("Did not converge")
        assert err.code == ErrorCode.CONVERGENCE_FAILED

    def test_model_versioning_error(self):
        """Test ModelVersioningError."""
        err = ModelVersioningError("Version mismatch")
        assert err.code == ErrorCode.MODEL_LOAD_FAILED


class TestDaemonErrors:
    """Tests for daemon-related errors."""

    def test_daemon_error_defaults(self):
        """Test DaemonError default values."""
        err = DaemonError("Daemon crashed")
        assert err.code == ErrorCode.DAEMON_CRASHED
        assert err.retryable is True  # Daemons can restart

    def test_daemon_startup_error(self):
        """Test DaemonStartupError."""
        err = DaemonStartupError("Failed to start")
        assert err.code == ErrorCode.DAEMON_START_FAILED

    def test_daemon_crash_error(self):
        """Test DaemonCrashError."""
        err = DaemonCrashError("Unexpected crash")
        assert err.code == ErrorCode.DAEMON_CRASHED

    def test_daemon_config_error(self):
        """Test DaemonConfigError."""
        err = DaemonConfigError("Invalid config")
        assert err.code == ErrorCode.DAEMON_CONFIG_INVALID
        assert err.retryable is False  # Fix config first

    def test_daemon_dependency_error(self):
        """Test DaemonDependencyError."""
        err = DaemonDependencyError("Missing dependency")
        assert err.code == ErrorCode.DAEMON_DEPENDENCY_FAILED


class TestValidationErrors:
    """Tests for validation-related errors."""

    def test_validation_error_defaults(self):
        """Test ValidationError default values."""
        err = ValidationError("Validation failed")
        assert err.code == ErrorCode.VALIDATION_FAILED
        assert err.retryable is False

    def test_schema_error(self):
        """Test SchemaError."""
        err = SchemaError("Schema mismatch")
        assert err.code == ErrorCode.SCHEMA_MISMATCH

    def test_parity_error(self):
        """Test ParityError."""
        err = ParityError("TS/Python parity failed")
        assert err.code == ErrorCode.PARITY_FAILED


class TestConfigurationErrors:
    """Tests for configuration-related errors."""

    def test_configuration_error_defaults(self):
        """Test ConfigurationError default values."""
        err = ConfigurationError("Config invalid")
        assert err.code == ErrorCode.CONFIG_INVALID
        assert err.retryable is False

    def test_config_missing_error(self):
        """Test ConfigMissingError."""
        err = ConfigMissingError("Config not found")
        assert err.code == ErrorCode.CONFIG_MISSING

    def test_config_type_error(self):
        """Test ConfigTypeError."""
        err = ConfigTypeError("Wrong type")
        assert err.code == ErrorCode.CONFIG_TYPE_ERROR


class TestSystemErrors:
    """Tests for system-related errors."""

    def test_emergency_halt_error(self):
        """Test EmergencyHaltError."""
        err = EmergencyHaltError("Critical issue")
        assert err.code == ErrorCode.EMERGENCY_HALT
        assert err.retryable is False

    def test_retryable_error(self):
        """Test RetryableError."""
        err = RetryableError("Transient failure")
        assert err.code == ErrorCode.RETRYABLE
        assert err.retryable is True

    def test_non_retryable_error(self):
        """Test NonRetryableError."""
        err = NonRetryableError("Fatal failure")
        assert err.code == ErrorCode.NON_RETRYABLE
        assert err.retryable is False


class TestBackwardCompatAliases:
    """Tests for backward compatibility aliases."""

    def test_fatal_error_alias(self):
        """Test FatalError is alias for NonRetryableError."""
        assert FatalError is NonRetryableError
        err = FatalError("Fatal")
        assert err.code == ErrorCode.NON_RETRYABLE

    def test_recoverable_error_alias(self):
        """Test RecoverableError is alias for RetryableError."""
        assert RecoverableError is RetryableError
        err = RecoverableError("Recoverable")
        assert err.code == ErrorCode.RETRYABLE


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """Test all error classes inherit from RingRiftError."""
        error_classes = [
            ResourceError, GPUError, GPUOutOfMemoryError, DiskError,
            DiskSpaceError, MemoryExhaustedError, NetworkError,
            ConnectionError, ConnectionTimeoutError, SSHError, SSHAuthError,
            HTTPError, P2PError, P2PLeaderUnavailableError, SyncError,
            SyncTimeoutError, SyncConflictError, SyncIntegrityError,
            TrainingError, DataQualityError, InvalidGameError, ModelLoadError,
            CheckpointCorruptError, ConvergenceError, ModelVersioningError,
            DaemonError, DaemonStartupError, DaemonCrashError,
            DaemonConfigError, DaemonDependencyError, ValidationError,
            SchemaError, ParityError, ConfigurationError, ConfigMissingError,
            ConfigTypeError, EmergencyHaltError, RetryableError, NonRetryableError,
        ]
        for cls in error_classes:
            err = cls("Test")
            assert isinstance(err, RingRiftError), f"{cls.__name__} should inherit from RingRiftError"
            assert isinstance(err, Exception), f"{cls.__name__} should inherit from Exception"

    def test_gpu_error_hierarchy(self):
        """Test GPU error inheritance chain."""
        err = GPUOutOfMemoryError("OOM")
        assert isinstance(err, GPUError)
        assert isinstance(err, ResourceError)
        assert isinstance(err, RingRiftError)

    def test_ssh_error_hierarchy(self):
        """Test SSH error inheritance chain."""
        err = SSHAuthError("Auth failed")
        assert isinstance(err, SSHError)
        assert isinstance(err, NetworkError)
        assert isinstance(err, RingRiftError)

    def test_invalid_game_error_hierarchy(self):
        """Test InvalidGameError inheritance chain."""
        err = InvalidGameError("Invalid")
        assert isinstance(err, DataQualityError)
        assert isinstance(err, TrainingError)
        assert isinstance(err, RingRiftError)


class TestExceptionCatching:
    """Tests for catching errors at different levels."""

    def test_catch_base_error(self):
        """Test catching all RingRift errors."""
        with pytest.raises(RingRiftError):
            raise GPUOutOfMemoryError("OOM")

    def test_catch_resource_error(self):
        """Test catching resource errors."""
        with pytest.raises(ResourceError):
            raise GPUOutOfMemoryError("OOM")

    def test_catch_network_error(self):
        """Test catching network errors."""
        with pytest.raises(NetworkError):
            raise SSHAuthError("Auth failed")

    def test_catch_specific_error(self):
        """Test catching specific error type."""
        with pytest.raises(GPUOutOfMemoryError):
            raise GPUOutOfMemoryError("OOM")

    def test_not_catch_unrelated_error(self):
        """Test that unrelated errors aren't caught."""
        with pytest.raises(GPUError):
            try:
                raise GPUError("GPU error")
            except NetworkError:
                pytest.fail("Should not catch GPUError as NetworkError")


class TestRetryDecisions:
    """Tests for retry decision making based on error types."""

    def test_retryable_errors(self):
        """Test errors that should be retried."""
        retryable = [
            ResourceError("Resource"),
            NetworkError("Network"),
            SyncError("Sync"),
            DaemonError("Daemon"),
            RetryableError("Retryable"),
        ]
        for err in retryable:
            assert err.retryable is True, f"{type(err).__name__} should be retryable"

    def test_non_retryable_errors(self):
        """Test errors that should not be retried."""
        non_retryable = [
            SSHAuthError("Auth"),
            SyncConflictError("Conflict"),
            SyncIntegrityError("Integrity"),
            TrainingError("Training"),
            DaemonConfigError("Config"),
            ValidationError("Validation"),
            ConfigurationError("Config"),
            EmergencyHaltError("Emergency"),
            NonRetryableError("Fatal"),
        ]
        for err in non_retryable:
            assert err.retryable is False, f"{type(err).__name__} should not be retryable"

    def test_override_retryable(self):
        """Test overriding default retryable flag."""
        # Make normally non-retryable error retryable
        err = ValidationError("Temp validation", retryable=True)
        assert err.retryable is True

        # Make normally retryable error non-retryable
        err = NetworkError("Permanent failure", retryable=False)
        assert err.retryable is False
