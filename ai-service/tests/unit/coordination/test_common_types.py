"""Tests for common_types module.

December 2025 - Tests for canonical coordination type definitions.
"""

import pytest

from app.coordination.common_types import (
    # Re-exported from contracts
    CoordinatorStatus,
    HealthCheckResult,
    # Enums
    SyncPriority,
    TransportState,
    # Dataclasses
    SyncResult,
    TransportConfig,
    # Exception
    TransportError,
    # Backward compat aliases
    RUNNING,
    STOPPED,
    ERROR,
)


# =============================================================================
# SyncPriority Tests
# =============================================================================


class TestSyncPriority:
    """Tests for SyncPriority enum."""

    def test_priority_values(self):
        """Test priority numeric values."""
        assert SyncPriority.LOW.value == 1
        assert SyncPriority.NORMAL.value == 5
        assert SyncPriority.HIGH.value == 10
        assert SyncPriority.URGENT.value == 20
        assert SyncPriority.CRITICAL.value == 50

    def test_priority_ordering(self):
        """Test priorities can be compared via value."""
        assert SyncPriority.LOW.value < SyncPriority.NORMAL.value
        assert SyncPriority.NORMAL.value < SyncPriority.HIGH.value
        assert SyncPriority.HIGH.value < SyncPriority.URGENT.value
        assert SyncPriority.URGENT.value < SyncPriority.CRITICAL.value

    def test_all_priorities_exist(self):
        """Test all expected priority levels exist."""
        expected = {"LOW", "NORMAL", "HIGH", "URGENT", "CRITICAL"}
        actual = {p.name for p in SyncPriority}
        assert actual == expected


# =============================================================================
# SyncResult Tests
# =============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = SyncResult(success=True)
        assert result.success is True
        assert result.files_synced == 0
        assert result.bytes_transferred == 0
        assert result.duration_seconds == 0.0
        assert result.errors == []
        assert result.source == ""
        assert result.destination == ""

    def test_full_initialization(self):
        """Test initialization with all fields."""
        result = SyncResult(
            success=True,
            files_synced=10,
            bytes_transferred=1024,
            duration_seconds=5.5,
            errors=["warn1"],
            source="node-a",
            destination="node-b",
        )
        assert result.files_synced == 10
        assert result.bytes_transferred == 1024
        assert result.duration_seconds == 5.5
        assert result.source == "node-a"
        assert result.destination == "node-b"

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = SyncResult(
            success=True,
            files_synced=5,
            bytes_transferred=500,
            duration_seconds=2.0,
            errors=["e1", "e2"],
            source="src",
            destination="dst",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["files_synced"] == 5
        assert d["bytes_transferred"] == 500
        assert d["duration_seconds"] == 2.0
        assert d["errors"] == ["e1", "e2"]
        assert d["source"] == "src"
        assert d["destination"] == "dst"

    def test_has_errors_false(self):
        """Test has_errors property when no errors."""
        result = SyncResult(success=True)
        assert result.has_errors is False

    def test_has_errors_true(self):
        """Test has_errors property when errors exist."""
        result = SyncResult(success=False, errors=["failed"])
        assert result.has_errors is True

    def test_failed_result(self):
        """Test failed sync result."""
        result = SyncResult(success=False, errors=["Connection timeout"])
        assert result.success is False
        assert result.has_errors is True


# =============================================================================
# TransportState Tests
# =============================================================================


class TestTransportState:
    """Tests for TransportState enum."""

    def test_state_values(self):
        """Test state string values."""
        assert TransportState.CLOSED.value == "closed"
        assert TransportState.OPEN.value == "open"
        assert TransportState.HALF_OPEN.value == "half_open"

    def test_all_states_exist(self):
        """Test all expected states exist."""
        expected = {"CLOSED", "OPEN", "HALF_OPEN"}
        actual = {s.name for s in TransportState}
        assert actual == expected

    def test_state_from_value(self):
        """Test creating state from string value."""
        assert TransportState("closed") == TransportState.CLOSED
        assert TransportState("open") == TransportState.OPEN
        assert TransportState("half_open") == TransportState.HALF_OPEN


# =============================================================================
# TransportConfig Tests
# =============================================================================


class TestTransportConfig:
    """Tests for TransportConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransportConfig()
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout_seconds == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransportConfig(
            timeout_seconds=60.0,
            max_retries=5,
            retry_delay_seconds=2.0,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout_seconds=120.0,
        )
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.circuit_breaker_threshold == 10
        assert config.circuit_breaker_timeout_seconds == 120.0


# =============================================================================
# TransportError Tests
# =============================================================================


class TestTransportError:
    """Tests for TransportError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = TransportError("Connection failed")
        assert error.message == "Connection failed"
        assert error.transport_type == "unknown"
        assert error.is_retryable is True
        assert error.details == {}

    def test_full_error(self):
        """Test error with all fields."""
        error = TransportError(
            message="Timeout",
            transport_type="ssh",
            is_retryable=False,
            details={"host": "node-1", "port": 22},
        )
        assert error.message == "Timeout"
        assert error.transport_type == "ssh"
        assert error.is_retryable is False
        assert error.details == {"host": "node-1", "port": 22}

    def test_str_representation(self):
        """Test string representation."""
        error = TransportError("Failed", transport_type="http")
        assert str(error) == "TransportError(http): Failed"

    def test_is_exception(self):
        """Test that TransportError is an Exception."""
        error = TransportError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Test error can be raised and caught."""
        with pytest.raises(TransportError) as exc_info:
            raise TransportError("Test error", transport_type="test")
        assert exc_info.value.message == "Test error"
        assert exc_info.value.transport_type == "test"


# =============================================================================
# Re-exported Types Tests
# =============================================================================


class TestReExportedTypes:
    """Tests for types re-exported from contracts.py."""

    def test_coordinator_status_available(self):
        """Test CoordinatorStatus is available."""
        assert CoordinatorStatus.RUNNING is not None
        assert CoordinatorStatus.STOPPED is not None
        assert CoordinatorStatus.ERROR is not None

    def test_health_check_result_available(self):
        """Test HealthCheckResult is available."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
        )
        assert result.healthy is True


# =============================================================================
# Backward Compatibility Aliases Tests
# =============================================================================


class TestBackwardCompatAliases:
    """Tests for backward compatibility aliases."""

    def test_running_alias(self):
        """Test RUNNING alias."""
        assert RUNNING == CoordinatorStatus.RUNNING

    def test_stopped_alias(self):
        """Test STOPPED alias."""
        assert STOPPED == CoordinatorStatus.STOPPED

    def test_error_alias(self):
        """Test ERROR alias."""
        assert ERROR == CoordinatorStatus.ERROR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
