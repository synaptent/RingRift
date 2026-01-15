"""Comprehensive tests for the RingRift error hierarchy.

Tests cover:
- Base RingRiftError instantiation and attributes
- String representation formatting
- to_dict() serialization
- Each major error subclass
- Special attributes on subclasses (game_id, status_code, host, etc.)
- Inheritance chains
- ErrorCode enum
- Aliases (FatalError, RecoverableError)
"""

import pytest

from app.errors import (
    # Codes
    ErrorCode,
    # Base error
    RingRiftError,
    # Resource errors
    ResourceError,
    GPUError,
    GPUOutOfMemoryError,
    DiskError,
    DiskSpaceError,
    MemoryExhaustedError,
    # Network errors
    NetworkError,
    ConnectionError,
    ConnectionTimeoutError,
    SSHError,
    SSHAuthError,
    HTTPError,
    P2PError,
    P2PLeaderUnavailableError,
    # Sync errors
    SyncError,
    SyncTimeoutError,
    SyncConflictError,
    SyncIntegrityError,
    # Training errors
    TrainingError,
    DataQualityError,
    InvalidGameError,
    ModelLoadError,
    CheckpointCorruptError,
    ConvergenceError,
    ModelVersioningError,
    # Daemon errors
    DaemonError,
    DaemonStartupError,
    DaemonCrashError,
    DaemonConfigError,
    DaemonDependencyError,
    # Validation errors
    ValidationError,
    SchemaError,
    ParityError,
    # Configuration errors
    ConfigurationError,
    ConfigMissingError,
    ConfigTypeError,
    # System errors
    EmergencyHaltError,
    RetryableError,
    NonRetryableError,
    # Aliases
    FatalError,
    RecoverableError,
)


class TestErrorCode:
    """Tests for the ErrorCode enum."""

    def test_resource_error_codes(self):
        """Test resource-related error codes exist and have correct values."""
        assert ErrorCode.RESOURCE_EXHAUSTED.value == 100
        assert ErrorCode.GPU_NOT_AVAILABLE.value == 101
        assert ErrorCode.GPU_OOM.value == 102
        assert ErrorCode.DISK_FULL.value == 103
        assert ErrorCode.MEMORY_OOM.value == 104

    def test_network_error_codes(self):
        """Test network-related error codes exist and have correct values."""
        assert ErrorCode.CONNECTION_FAILED.value == 200
        assert ErrorCode.CONNECTION_TIMEOUT.value == 201
        assert ErrorCode.SSH_AUTH_FAILED.value == 202
        assert ErrorCode.SSH_COMMAND_FAILED.value == 203
        assert ErrorCode.HTTP_CLIENT_ERROR.value == 204
        assert ErrorCode.HTTP_SERVER_ERROR.value == 205
        assert ErrorCode.P2P_MESH_ERROR.value == 206
        assert ErrorCode.P2P_LEADER_UNAVAILABLE.value == 207

    def test_sync_error_codes(self):
        """Test sync-related error codes."""
        assert ErrorCode.SYNC_TIMEOUT.value == 300
        assert ErrorCode.SYNC_CONFLICT.value == 301
        assert ErrorCode.SYNC_INTEGRITY_FAILED.value == 302
        assert ErrorCode.SYNC_MANIFEST_MISMATCH.value == 303

    def test_training_error_codes(self):
        """Test training-related error codes."""
        assert ErrorCode.DATA_QUALITY_LOW.value == 400
        assert ErrorCode.MODEL_LOAD_FAILED.value == 401
        assert ErrorCode.CONVERGENCE_FAILED.value == 402
        assert ErrorCode.CHECKPOINT_CORRUPT.value == 403
        assert ErrorCode.TRAINING_INTERRUPTED.value == 404

    def test_daemon_error_codes(self):
        """Test daemon-related error codes."""
        assert ErrorCode.DAEMON_START_FAILED.value == 500
        assert ErrorCode.DAEMON_CRASHED.value == 501
        assert ErrorCode.DAEMON_CONFIG_INVALID.value == 502
        assert ErrorCode.DAEMON_DEPENDENCY_FAILED.value == 503

    def test_validation_error_codes(self):
        """Test validation-related error codes."""
        assert ErrorCode.VALIDATION_FAILED.value == 600
        assert ErrorCode.SCHEMA_MISMATCH.value == 601
        assert ErrorCode.PARITY_FAILED.value == 602

    def test_config_error_codes(self):
        """Test configuration-related error codes."""
        assert ErrorCode.CONFIG_MISSING.value == 700
        assert ErrorCode.CONFIG_INVALID.value == 701
        assert ErrorCode.CONFIG_TYPE_ERROR.value == 702

    def test_system_error_codes(self):
        """Test system-related error codes."""
        assert ErrorCode.EMERGENCY_HALT.value == 800
        assert ErrorCode.RETRYABLE.value == 801
        assert ErrorCode.NON_RETRYABLE.value == 802

    def test_unknown_code(self):
        """Test unknown error code."""
        assert ErrorCode.UNKNOWN.value == 999


class TestRingRiftErrorBase:
    """Tests for the base RingRiftError class."""

    def test_basic_instantiation(self):
        """Test basic error instantiation with just a message."""
        error = RingRiftError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.UNKNOWN
        assert error.details == {}
        assert error.retryable is False

    def test_custom_code(self):
        """Test instantiation with a custom error code."""
        error = RingRiftError("Test error", code=ErrorCode.GPU_OOM)
        assert error.code == ErrorCode.GPU_OOM
        assert error.message == "Test error"

    def test_details_provided(self):
        """Test instantiation with details dictionary."""
        details = {"key": "value", "count": 42}
        error = RingRiftError("Test error", details=details)
        assert error.details == {"key": "value", "count": 42}

    def test_details_defaults_to_empty_dict(self):
        """Test that details defaults to empty dict if None."""
        error = RingRiftError("Test", details=None)
        assert error.details == {}

    def test_retryable_override(self):
        """Test that retryable can be overridden."""
        error = RingRiftError("Test", retryable=True)
        assert error.retryable is True

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

    def test_str_without_details_unknown_code(self):
        """Test string representation without details and unknown code."""
        error = RingRiftError("Test message")
        # Unknown code is not included in output
        assert str(error) == "Test message"

    def test_str_with_known_code(self):
        """Test string representation with known code."""
        error = RingRiftError("Test message", code=ErrorCode.GPU_OOM)
        assert "[GPU_OOM]" in str(error)
        assert "Test message" in str(error)

    def test_str_with_details(self):
        """Test string representation with details."""
        error = RingRiftError("Error", code=ErrorCode.GPU_OOM, details={"gpu": 0})
        result = str(error)
        assert "Error" in result
        assert "gpu=0" in result

    def test_str_with_multiple_details(self):
        """Test string representation with multiple detail items."""
        error = RingRiftError("Error", code=ErrorCode.DISK_FULL, details={"a": 1, "b": 2})
        result = str(error)
        assert "a=1" in result
        assert "b=2" in result


class TestRingRiftErrorToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        error = RingRiftError("Test message")
        d = error.to_dict()
        assert d["error"] == "RingRiftError"
        assert d["message"] == "Test message"
        assert d["code"] == ErrorCode.UNKNOWN.value
        assert d["code_name"] == "UNKNOWN"
        assert d["retryable"] is False
        assert d["details"] == {}

    def test_to_dict_with_details(self):
        """Test to_dict with details."""
        error = RingRiftError("Test", details={"key": "value"})
        d = error.to_dict()
        assert d["details"] == {"key": "value"}

    def test_to_dict_with_custom_code(self):
        """Test to_dict with custom code."""
        error = RingRiftError("Test", code=ErrorCode.GPU_OOM)
        d = error.to_dict()
        assert d["code"] == ErrorCode.GPU_OOM.value
        assert d["code_name"] == "GPU_OOM"

    def test_to_dict_preserves_subclass_name(self):
        """Test to_dict preserves subclass name."""
        error = GPUError("GPU issue")
        d = error.to_dict()
        assert d["error"] == "GPUError"


class TestResourceErrors:
    """Tests for Resource error classes."""

    def test_resource_error_defaults(self):
        """Test ResourceError default attributes."""
        error = ResourceError("Resource exhausted")
        assert error.code == ErrorCode.RESOURCE_EXHAUSTED
        assert error.retryable is True

    def test_gpu_error_defaults(self):
        """Test GPUError default attributes."""
        error = GPUError("GPU not available")
        assert error.code == ErrorCode.GPU_NOT_AVAILABLE
        assert isinstance(error, ResourceError)

    def test_gpu_oom_error_defaults(self):
        """Test GPUOutOfMemoryError default attributes."""
        error = GPUOutOfMemoryError("Out of VRAM")
        assert error.code == ErrorCode.GPU_OOM
        assert isinstance(error, GPUError)

    def test_disk_error_defaults(self):
        """Test DiskError default attributes."""
        error = DiskError("Disk full")
        assert error.code == ErrorCode.DISK_FULL
        assert isinstance(error, ResourceError)

    def test_memory_exhausted_error_defaults(self):
        """Test MemoryExhaustedError default attributes."""
        error = MemoryExhaustedError("OOM")
        assert error.code == ErrorCode.MEMORY_OOM
        assert isinstance(error, ResourceError)


class TestDiskSpaceError:
    """Tests for DiskSpaceError special attributes."""

    def test_disk_space_error_basic(self):
        """Test DiskSpaceError basic instantiation."""
        error = DiskSpaceError("Disk full")
        assert error.code == ErrorCode.DISK_FULL
        assert isinstance(error, DiskError)

    def test_disk_space_error_with_path(self):
        """Test DiskSpaceError with path attribute."""
        error = DiskSpaceError("Disk full", path="/data")
        assert error.path == "/data"
        assert error.details["path"] == "/data"

    def test_disk_space_error_with_bytes(self):
        """Test DiskSpaceError with byte counts."""
        error = DiskSpaceError(
            "Disk full",
            available_bytes=100000,
            required_bytes=500000,
        )
        assert error.available_bytes == 100000
        assert error.required_bytes == 500000
        assert error.details["available_bytes"] == 100000
        assert error.details["required_bytes"] == 500000

    def test_disk_space_error_all_attributes(self):
        """Test DiskSpaceError with all attributes."""
        error = DiskSpaceError(
            "Disk full",
            path="/var/data",
            available_bytes=1024,
            required_bytes=2048,
        )
        assert error.path == "/var/data"
        assert error.available_bytes == 1024
        assert error.required_bytes == 2048


class TestNetworkErrors:
    """Tests for Network error classes."""

    def test_network_error_defaults(self):
        """Test NetworkError default attributes."""
        error = NetworkError("Connection failed")
        assert error.code == ErrorCode.CONNECTION_FAILED
        assert error.retryable is True

    def test_connection_error_defaults(self):
        """Test ConnectionError default attributes."""
        error = ConnectionError("Failed to connect")
        assert error.code == ErrorCode.CONNECTION_FAILED
        assert isinstance(error, NetworkError)

    def test_connection_timeout_error_defaults(self):
        """Test ConnectionTimeoutError default attributes."""
        error = ConnectionTimeoutError("Timed out")
        assert error.code == ErrorCode.CONNECTION_TIMEOUT
        assert isinstance(error, NetworkError)

    def test_ssh_error_defaults(self):
        """Test SSHError default attributes."""
        error = SSHError("SSH command failed")
        assert error.code == ErrorCode.SSH_COMMAND_FAILED
        assert isinstance(error, NetworkError)

    def test_ssh_auth_error_defaults(self):
        """Test SSHAuthError default attributes."""
        error = SSHAuthError("Authentication failed")
        assert error.code == ErrorCode.SSH_AUTH_FAILED
        assert error.retryable is False  # Auth errors not retryable
        assert isinstance(error, SSHError)

    def test_p2p_error_defaults(self):
        """Test P2PError default attributes."""
        error = P2PError("Mesh error")
        assert error.code == ErrorCode.P2P_MESH_ERROR
        assert isinstance(error, NetworkError)

    def test_p2p_leader_unavailable_defaults(self):
        """Test P2PLeaderUnavailableError default attributes."""
        error = P2PLeaderUnavailableError("Leader down")
        assert error.code == ErrorCode.P2P_LEADER_UNAVAILABLE
        assert isinstance(error, P2PError)


class TestHTTPError:
    """Tests for HTTPError special attributes."""

    def test_http_error_basic(self):
        """Test HTTPError basic instantiation."""
        error = HTTPError("HTTP request failed")
        assert error.code == ErrorCode.HTTP_CLIENT_ERROR
        assert isinstance(error, NetworkError)

    def test_http_error_with_client_status_code(self):
        """Test HTTPError with 4xx status code."""
        error = HTTPError("Not found", status_code=404)
        assert error.status_code == 404
        assert error.details["status_code"] == 404
        assert error.code == ErrorCode.HTTP_CLIENT_ERROR

    def test_http_error_with_server_status_code(self):
        """Test HTTPError with 5xx status code sets server error code."""
        error = HTTPError("Server error", status_code=500)
        assert error.status_code == 500
        assert error.code == ErrorCode.HTTP_SERVER_ERROR

    def test_http_error_502_is_server_error(self):
        """Test HTTPError with 502 sets server error code."""
        error = HTTPError("Bad gateway", status_code=502)
        assert error.code == ErrorCode.HTTP_SERVER_ERROR


class TestSyncErrors:
    """Tests for Sync error classes."""

    def test_sync_error_defaults(self):
        """Test SyncError default attributes."""
        error = SyncError("Sync failed")
        assert error.code == ErrorCode.SYNC_TIMEOUT
        assert error.retryable is True

    def test_sync_timeout_error_defaults(self):
        """Test SyncTimeoutError default attributes."""
        error = SyncTimeoutError("Timed out")
        assert error.code == ErrorCode.SYNC_TIMEOUT
        assert isinstance(error, SyncError)

    def test_sync_conflict_error_defaults(self):
        """Test SyncConflictError default attributes."""
        error = SyncConflictError("Conflict detected")
        assert error.code == ErrorCode.SYNC_CONFLICT
        assert error.retryable is False  # Need manual resolution
        assert isinstance(error, SyncError)

    def test_sync_integrity_error_defaults(self):
        """Test SyncIntegrityError default attributes."""
        error = SyncIntegrityError("Integrity check failed")
        assert error.code == ErrorCode.SYNC_INTEGRITY_FAILED
        assert error.retryable is False  # Data is corrupted
        assert isinstance(error, SyncError)


class TestTrainingErrors:
    """Tests for Training error classes."""

    def test_training_error_defaults(self):
        """Test TrainingError default attributes."""
        error = TrainingError("Training failed")
        assert error.code == ErrorCode.TRAINING_ERROR
        assert error.retryable is False

    def test_data_quality_error_defaults(self):
        """Test DataQualityError default attributes."""
        error = DataQualityError("Data quality too low")
        assert error.code == ErrorCode.DATA_QUALITY_LOW
        assert isinstance(error, TrainingError)

    def test_model_load_error_defaults(self):
        """Test ModelLoadError default attributes."""
        error = ModelLoadError("Failed to load model")
        assert error.code == ErrorCode.MODEL_LOAD_FAILED
        assert isinstance(error, TrainingError)

    def test_checkpoint_corrupt_error_defaults(self):
        """Test CheckpointCorruptError default attributes."""
        error = CheckpointCorruptError("Checkpoint corrupted")
        assert error.code == ErrorCode.CHECKPOINT_CORRUPT
        assert isinstance(error, ModelLoadError)

    def test_convergence_error_defaults(self):
        """Test ConvergenceError default attributes."""
        error = ConvergenceError("Failed to converge")
        assert error.code == ErrorCode.CONVERGENCE_FAILED
        assert isinstance(error, TrainingError)

    def test_model_versioning_error_defaults(self):
        """Test ModelVersioningError default attributes."""
        error = ModelVersioningError("Version mismatch")
        assert error.code == ErrorCode.MODEL_VERSIONING_ERROR
        assert isinstance(error, TrainingError)


class TestInvalidGameError:
    """Tests for InvalidGameError special attributes."""

    def test_invalid_game_error_basic(self):
        """Test InvalidGameError basic instantiation."""
        error = InvalidGameError("Game data invalid")
        assert error.code == ErrorCode.DATA_QUALITY_LOW
        assert isinstance(error, DataQualityError)

    def test_invalid_game_error_with_game_id(self):
        """Test InvalidGameError with game_id attribute."""
        error = InvalidGameError("Invalid", game_id="game-123")
        assert error.game_id == "game-123"
        assert error.details["game_id"] == "game-123"

    def test_invalid_game_error_with_move_count(self):
        """Test InvalidGameError with move_count attribute."""
        error = InvalidGameError("Invalid", move_count=0)
        assert error.move_count == 0
        assert error.details["move_count"] == 0

    def test_invalid_game_error_all_attributes(self):
        """Test InvalidGameError with all attributes."""
        error = InvalidGameError(
            "Invalid game",
            game_id="game-456",
            move_count=10,
        )
        assert error.game_id == "game-456"
        assert error.move_count == 10


class TestDaemonErrors:
    """Tests for Daemon error classes."""

    def test_daemon_error_defaults(self):
        """Test DaemonError default attributes."""
        error = DaemonError("Daemon failed")
        assert error.code == ErrorCode.DAEMON_CRASHED
        assert error.retryable is True  # Daemons can be restarted

    def test_daemon_startup_error_defaults(self):
        """Test DaemonStartupError default attributes."""
        error = DaemonStartupError("Failed to start")
        assert error.code == ErrorCode.DAEMON_START_FAILED
        assert isinstance(error, DaemonError)

    def test_daemon_crash_error_defaults(self):
        """Test DaemonCrashError default attributes."""
        error = DaemonCrashError("Crashed")
        assert error.code == ErrorCode.DAEMON_CRASHED
        assert isinstance(error, DaemonError)

    def test_daemon_config_error_defaults(self):
        """Test DaemonConfigError default attributes."""
        error = DaemonConfigError("Invalid config")
        assert error.code == ErrorCode.DAEMON_CONFIG_INVALID
        assert error.retryable is False  # Fix config first
        assert isinstance(error, DaemonError)

    def test_daemon_dependency_error_defaults(self):
        """Test DaemonDependencyError default attributes."""
        error = DaemonDependencyError("Dependency not available")
        assert error.code == ErrorCode.DAEMON_DEPENDENCY_FAILED
        assert isinstance(error, DaemonError)


class TestValidationErrors:
    """Tests for Validation error classes."""

    def test_validation_error_defaults(self):
        """Test ValidationError default attributes."""
        error = ValidationError("Validation failed")
        assert error.code == ErrorCode.VALIDATION_FAILED
        assert error.retryable is False  # Fix the data

    def test_schema_error_defaults(self):
        """Test SchemaError default attributes."""
        error = SchemaError("Schema mismatch")
        assert error.code == ErrorCode.SCHEMA_MISMATCH
        assert isinstance(error, ValidationError)

    def test_parity_error_defaults(self):
        """Test ParityError default attributes."""
        error = ParityError("Parity check failed")
        assert error.code == ErrorCode.PARITY_FAILED
        assert isinstance(error, ValidationError)


class TestConfigurationErrors:
    """Tests for Configuration error classes."""

    def test_configuration_error_defaults(self):
        """Test ConfigurationError default attributes."""
        error = ConfigurationError("Invalid config")
        assert error.code == ErrorCode.CONFIG_INVALID
        assert error.retryable is False  # Fix the config

    def test_config_missing_error_defaults(self):
        """Test ConfigMissingError default attributes."""
        error = ConfigMissingError("Config missing")
        assert error.code == ErrorCode.CONFIG_MISSING
        assert isinstance(error, ConfigurationError)

    def test_config_type_error_defaults(self):
        """Test ConfigTypeError default attributes."""
        error = ConfigTypeError("Wrong type")
        assert error.code == ErrorCode.CONFIG_TYPE_ERROR
        assert isinstance(error, ConfigurationError)


class TestSystemErrors:
    """Tests for System error classes."""

    def test_emergency_halt_error_defaults(self):
        """Test EmergencyHaltError default attributes."""
        error = EmergencyHaltError("Emergency halt triggered")
        assert error.code == ErrorCode.EMERGENCY_HALT
        assert error.retryable is False  # Requires intervention

    def test_retryable_error_defaults(self):
        """Test RetryableError default attributes."""
        error = RetryableError("Transient failure")
        assert error.code == ErrorCode.RETRYABLE
        assert error.retryable is True

    def test_non_retryable_error_defaults(self):
        """Test NonRetryableError default attributes."""
        error = NonRetryableError("Fatal error")
        assert error.code == ErrorCode.NON_RETRYABLE
        assert error.retryable is False


class TestErrorAliases:
    """Tests for backward-compatible error aliases."""

    def test_fatal_error_alias(self):
        """Test FatalError is alias for NonRetryableError."""
        assert FatalError is NonRetryableError
        error = FatalError("Fatal")
        assert error.code == ErrorCode.NON_RETRYABLE
        assert error.retryable is False

    def test_recoverable_error_alias(self):
        """Test RecoverableError is alias for RetryableError."""
        assert RecoverableError is RetryableError
        error = RecoverableError("Recoverable")
        assert error.code == ErrorCode.RETRYABLE
        assert error.retryable is True


class TestInheritanceChains:
    """Tests to verify correct inheritance chains."""

    def test_resource_error_chain(self):
        """Test resource error inheritance chain."""
        # ResourceError -> RingRiftError
        assert issubclass(ResourceError, RingRiftError)
        # GPUError -> ResourceError
        assert issubclass(GPUError, ResourceError)
        # GPUOutOfMemoryError -> GPUError -> ResourceError
        assert issubclass(GPUOutOfMemoryError, GPUError)
        # DiskError -> ResourceError
        assert issubclass(DiskError, ResourceError)
        # DiskSpaceError -> DiskError -> ResourceError
        assert issubclass(DiskSpaceError, DiskError)
        # MemoryExhaustedError -> ResourceError
        assert issubclass(MemoryExhaustedError, ResourceError)

    def test_network_error_chain(self):
        """Test network error inheritance chain."""
        # NetworkError -> RingRiftError
        assert issubclass(NetworkError, RingRiftError)
        # ConnectionError -> NetworkError
        assert issubclass(ConnectionError, NetworkError)
        # ConnectionTimeoutError -> NetworkError
        assert issubclass(ConnectionTimeoutError, NetworkError)
        # SSHError -> NetworkError
        assert issubclass(SSHError, NetworkError)
        # SSHAuthError -> SSHError -> NetworkError
        assert issubclass(SSHAuthError, SSHError)
        # HTTPError -> NetworkError
        assert issubclass(HTTPError, NetworkError)
        # P2PError -> NetworkError
        assert issubclass(P2PError, NetworkError)
        # P2PLeaderUnavailableError -> P2PError -> NetworkError
        assert issubclass(P2PLeaderUnavailableError, P2PError)

    def test_sync_error_chain(self):
        """Test sync error inheritance chain."""
        # SyncError -> RingRiftError
        assert issubclass(SyncError, RingRiftError)
        # SyncTimeoutError -> SyncError
        assert issubclass(SyncTimeoutError, SyncError)
        # SyncConflictError -> SyncError
        assert issubclass(SyncConflictError, SyncError)
        # SyncIntegrityError -> SyncError
        assert issubclass(SyncIntegrityError, SyncError)

    def test_training_error_chain(self):
        """Test training error inheritance chain."""
        # TrainingError -> RingRiftError
        assert issubclass(TrainingError, RingRiftError)
        # DataQualityError -> TrainingError
        assert issubclass(DataQualityError, TrainingError)
        # InvalidGameError -> DataQualityError -> TrainingError
        assert issubclass(InvalidGameError, DataQualityError)
        # ModelLoadError -> TrainingError
        assert issubclass(ModelLoadError, TrainingError)
        # CheckpointCorruptError -> ModelLoadError -> TrainingError
        assert issubclass(CheckpointCorruptError, ModelLoadError)
        # ConvergenceError -> TrainingError
        assert issubclass(ConvergenceError, TrainingError)
        # ModelVersioningError -> TrainingError
        assert issubclass(ModelVersioningError, TrainingError)

    def test_daemon_error_chain(self):
        """Test daemon error inheritance chain."""
        # DaemonError -> RingRiftError
        assert issubclass(DaemonError, RingRiftError)
        # DaemonStartupError -> DaemonError
        assert issubclass(DaemonStartupError, DaemonError)
        # DaemonCrashError -> DaemonError
        assert issubclass(DaemonCrashError, DaemonError)
        # DaemonConfigError -> DaemonError
        assert issubclass(DaemonConfigError, DaemonError)
        # DaemonDependencyError -> DaemonError
        assert issubclass(DaemonDependencyError, DaemonError)

    def test_validation_error_chain(self):
        """Test validation error inheritance chain."""
        # ValidationError -> RingRiftError
        assert issubclass(ValidationError, RingRiftError)
        # SchemaError -> ValidationError
        assert issubclass(SchemaError, ValidationError)
        # ParityError -> ValidationError
        assert issubclass(ParityError, ValidationError)

    def test_configuration_error_chain(self):
        """Test configuration error inheritance chain."""
        # ConfigurationError -> RingRiftError
        assert issubclass(ConfigurationError, RingRiftError)
        # ConfigMissingError -> ConfigurationError
        assert issubclass(ConfigMissingError, ConfigurationError)
        # ConfigTypeError -> ConfigurationError
        assert issubclass(ConfigTypeError, ConfigurationError)


class TestErrorCatching:
    """Tests to verify errors can be caught at various levels."""

    def test_catch_by_base_class(self):
        """Test catching errors by base class."""
        with pytest.raises(RingRiftError):
            raise ResourceError("Test")

        with pytest.raises(RingRiftError):
            raise NetworkError("Test")

        with pytest.raises(RingRiftError):
            raise TrainingError("Test")

    def test_catch_resource_errors(self):
        """Test catching all resource errors with ResourceError."""
        with pytest.raises(ResourceError):
            raise GPUError("Test")

        with pytest.raises(ResourceError):
            raise GPUOutOfMemoryError("Test")

        with pytest.raises(ResourceError):
            raise DiskError("Test")

    def test_catch_network_errors(self):
        """Test catching all network errors with NetworkError."""
        with pytest.raises(NetworkError):
            raise SSHError("Test")

        with pytest.raises(NetworkError):
            raise HTTPError("Test")

        with pytest.raises(NetworkError):
            raise P2PError("Test")

    def test_catch_training_errors(self):
        """Test catching all training errors with TrainingError."""
        with pytest.raises(TrainingError):
            raise DataQualityError("Test")

        with pytest.raises(TrainingError):
            raise ModelLoadError("Test")

        with pytest.raises(TrainingError):
            raise InvalidGameError("Test")

    def test_catch_daemon_errors(self):
        """Test catching all daemon errors with DaemonError."""
        with pytest.raises(DaemonError):
            raise DaemonStartupError("Test")

        with pytest.raises(DaemonError):
            raise DaemonCrashError("Test")

        with pytest.raises(DaemonError):
            raise DaemonConfigError("Test")


class TestRetryableVsNonRetryable:
    """Tests to verify retryable flag is correctly set."""

    def test_retryable_errors(self):
        """Test that retryable errors have retryable=True."""
        retryable_classes = [
            ResourceError,
            NetworkError,
            SyncError,
            DaemonError,
            RetryableError,
        ]
        for cls in retryable_classes:
            error = cls("Test")
            assert error.retryable is True, f"{cls.__name__} should be retryable"

    def test_non_retryable_errors(self):
        """Test that non-retryable errors have retryable=False."""
        non_retryable_classes = [
            TrainingError,
            ValidationError,
            ConfigurationError,
            NonRetryableError,
            EmergencyHaltError,
            SyncConflictError,
            SyncIntegrityError,
            SSHAuthError,
            DaemonConfigError,
        ]
        for cls in non_retryable_classes:
            error = cls("Test")
            assert error.retryable is False, f"{cls.__name__} should not be retryable"
