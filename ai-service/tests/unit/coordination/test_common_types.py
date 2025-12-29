"""Tests for common_types module.

December 2025 - Tests for canonical coordination type definitions.
This module tests the re-exports from common_types.py which provides
a unified import point for types from their canonical locations.
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
    TransportResult,
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
    """Tests for SyncPriority enum.

    Note: Values are from sync_constants.py (the canonical source).
    """

    def test_priority_values(self):
        """Test priority numeric values."""
        assert SyncPriority.BACKGROUND.value == 10
        assert SyncPriority.LOW.value == 25
        assert SyncPriority.NORMAL.value == 50
        assert SyncPriority.HIGH.value == 75
        assert SyncPriority.CRITICAL.value == 100

    def test_priority_ordering(self):
        """Test priorities can be compared via value."""
        assert SyncPriority.BACKGROUND.value < SyncPriority.LOW.value
        assert SyncPriority.LOW.value < SyncPriority.NORMAL.value
        assert SyncPriority.NORMAL.value < SyncPriority.HIGH.value
        assert SyncPriority.HIGH.value < SyncPriority.CRITICAL.value

    def test_all_priorities_exist(self):
        """Test all expected priority levels exist."""
        expected = {"BACKGROUND", "LOW", "NORMAL", "HIGH", "CRITICAL"}
        actual = {p.name for p in SyncPriority}
        assert actual == expected

    def test_comparison_operators(self):
        """Test that SyncPriority supports < and > operators."""
        assert SyncPriority.LOW < SyncPriority.NORMAL
        assert SyncPriority.HIGH > SyncPriority.NORMAL


# =============================================================================
# SyncResult Tests
# =============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass.

    Note: Fields are from sync_constants.py (the canonical source).
    """

    def test_default_values(self):
        """Test default initialization."""
        result = SyncResult(success=True, source="src", dest="dst", host="node")
        assert result.success is True
        assert result.source == "src"
        assert result.dest == "dst"
        assert result.host == "node"
        assert result.files_synced == 0
        assert result.bytes_transferred == 0
        assert result.duration_seconds == 0.0
        assert result.error is None

    def test_full_initialization(self):
        """Test initialization with common fields."""
        result = SyncResult(
            success=True,
            source="node-a",
            dest="node-b",
            host="node-a",
            files_synced=10,
            bytes_transferred=1024,
            duration_seconds=5.5,
        )
        assert result.files_synced == 10
        assert result.bytes_transferred == 1024
        assert result.duration_seconds == 5.5
        assert result.source == "node-a"
        assert result.dest == "node-b"

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = SyncResult(
            success=True,
            source="src",
            dest="dst",
            host="host",
            files_synced=5,
            bytes_transferred=500,
            duration_seconds=2.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["files_synced"] == 5
        assert d["bytes_transferred"] == 500
        assert d["duration_seconds"] == 2.0
        assert d["source"] == "src"
        assert d["dest"] == "dst"

    def test_failed_result(self):
        """Test failed sync result."""
        result = SyncResult(
            success=False,
            source="a",
            dest="b",
            host="a",
            error="Connection timeout"
        )
        assert result.success is False
        assert result.error == "Connection timeout"

    def test_factory_methods(self):
        """Test SyncResult factory methods."""
        # Test success factory
        success = SyncResult.success_result(
            source="a",
            dest="b",
            host="a",
            bytes_transferred=100,
            files_synced=1,
            duration_seconds=1.0,
        )
        assert success.success is True
        assert success.bytes_transferred == 100

        # Test failure factory
        failure = SyncResult.failure_result(
            source="a",
            dest="b",
            host="a",
            error="Failed",
        )
        assert failure.success is False
        assert failure.error == "Failed"


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
# TransportResult Tests
# =============================================================================


class TestTransportResult:
    """Tests for TransportResult dataclass."""

    def test_default_values(self):
        """Test default values for successful result."""
        result = TransportResult(success=True)
        assert result.success is True
        assert result.transport_used == ""
        assert result.error is None
        assert result.latency_ms == 0.0
        assert result.bytes_transferred == 0
        assert result.data is None
        assert result.metadata == {}

    def test_full_result(self):
        """Test result with all fields populated."""
        result = TransportResult(
            success=True,
            transport_used="ssh",
            error=None,
            latency_ms=150.5,
            bytes_transferred=1024,
            data={"status": "ok"},
            metadata={"host": "node-1"},
        )
        assert result.success is True
        assert result.transport_used == "ssh"
        assert result.latency_ms == 150.5
        assert result.bytes_transferred == 1024
        assert result.data == {"status": "ok"}
        assert result.metadata == {"host": "node-1"}

    def test_failed_result_gets_default_error(self):
        """Test that failed results get a default error message."""
        result = TransportResult(success=False)
        assert result.success is False
        assert result.error == "Unknown error"

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = TransportResult(
            success=True,
            transport_used="http",
            latency_ms=100.0,
            bytes_transferred=500,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["transport_used"] == "http"
        assert d["latency_ms"] == 100.0
        assert d["bytes_transferred"] == 500


# =============================================================================
# TransportError Tests
# =============================================================================


class TestTransportError:
    """Tests for TransportError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = TransportError("Connection failed")
        assert error.message == "Connection failed"
        assert error.transport == ""
        assert error.target == ""
        assert error.cause is None

    def test_full_error(self):
        """Test error with all fields."""
        cause = ValueError("underlying issue")
        error = TransportError(
            message="Timeout",
            transport="ssh",
            target="node-1",
            cause=cause,
        )
        assert error.message == "Timeout"
        assert error.transport == "ssh"
        assert error.target == "node-1"
        assert error.cause is cause

    def test_str_representation(self):
        """Test string representation."""
        error = TransportError("Failed", transport="http", target="host:8080")
        assert str(error) == "Failed | transport=http | target=host:8080"

    def test_is_exception(self):
        """Test that TransportError is an Exception."""
        error = TransportError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised(self):
        """Test error can be raised and caught."""
        with pytest.raises(TransportError) as exc_info:
            raise TransportError("Test error", transport="test")
        assert exc_info.value.message == "Test error"
        assert exc_info.value.transport == "test"


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
