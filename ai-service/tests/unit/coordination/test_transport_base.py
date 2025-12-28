"""Tests for TransportBase - Unified transport infrastructure.

Created: December 28, 2025
Purpose: Test circuit breaker, timeout handling, and state persistence in TransportBase

Tests cover:
- TransportState enum values and transitions
- TransportResult dataclass validation
- TransportError exception formatting
- CircuitBreakerConfig factory methods
- TimeoutConfig factory methods
- TargetStatus circuit breaker state
- TransportBase circuit breaker methods
- TransportBase timeout execution
- TransportBase state persistence
- TransportBase health_check
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.transport_base import (
    CircuitBreakerConfig,
    TargetStatus,
    TimeoutConfig,
    TransportBase,
    TransportError,
    TransportResult,
    TransportState,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class ConcreteTransport(TransportBase):
    """Concrete implementation for testing the abstract base class."""

    async def transfer(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> TransportResult:
        """Minimal transfer implementation for testing."""
        # Return a successful result by default
        return TransportResult(
            success=True,
            source=str(source),
            destination=str(destination),
            message="Test transfer completed",
        )


@pytest.fixture
def transport():
    """Create a basic transport instance."""
    return ConcreteTransport()


@pytest.fixture
def transport_with_state(tmp_path):
    """Create a transport with state persistence."""
    state_file = tmp_path / "transport_state.json"
    return ConcreteTransport(state_path=state_file)


# =============================================================================
# TransportState Tests
# =============================================================================


class TestTransportState:
    """Tests for TransportState enum."""

    def test_closed_state_value(self):
        """CLOSED state should have value 'closed'."""
        assert TransportState.CLOSED.value == "closed"

    def test_open_state_value(self):
        """OPEN state should have value 'open'."""
        assert TransportState.OPEN.value == "open"

    def test_half_open_state_value(self):
        """HALF_OPEN state should have value 'half_open'."""
        assert TransportState.HALF_OPEN.value == "half_open"

    def test_all_states_enumerable(self):
        """All states should be enumerable."""
        states = list(TransportState)
        assert len(states) == 3
        assert TransportState.CLOSED in states
        assert TransportState.OPEN in states
        assert TransportState.HALF_OPEN in states


# =============================================================================
# TransportResult Tests
# =============================================================================


class TestTransportResult:
    """Tests for TransportResult dataclass."""

    def test_success_result(self):
        """Successful result should have correct defaults."""
        result = TransportResult(success=True)
        assert result.success is True
        assert result.error is None
        assert result.transport_used == ""
        assert result.latency_ms == 0.0
        assert result.bytes_transferred == 0
        assert result.metadata == {}

    def test_failure_result_requires_error(self):
        """Failed result should have error message."""
        result = TransportResult(success=False)
        assert result.success is False
        assert result.error == "Unknown error"  # Set by __post_init__

    def test_failure_result_with_custom_error(self):
        """Failed result can have custom error."""
        result = TransportResult(success=False, error="Connection refused")
        assert result.error == "Connection refused"

    def test_result_with_metadata(self):
        """Result can include metadata."""
        result = TransportResult(
            success=True,
            transport_used="rsync",
            latency_ms=150.0,
            bytes_transferred=1024,
            metadata={"compression": "gzip"},
        )
        assert result.transport_used == "rsync"
        assert result.latency_ms == 150.0
        assert result.bytes_transferred == 1024
        assert result.metadata["compression"] == "gzip"


# =============================================================================
# TransportError Tests
# =============================================================================


class TestTransportError:
    """Tests for TransportError exception."""

    def test_error_message_only(self):
        """Error with message only."""
        error = TransportError(message="Connection failed")
        assert str(error) == "Connection failed"

    def test_error_with_transport(self):
        """Error with transport name."""
        error = TransportError(message="Timeout", transport="rsync")
        assert "Timeout" in str(error)
        assert "transport=rsync" in str(error)

    def test_error_with_target(self):
        """Error with target."""
        error = TransportError(message="Failed", target="host-1")
        assert "Failed" in str(error)
        assert "target=host-1" in str(error)

    def test_error_full(self):
        """Error with all fields."""
        cause = ValueError("Bad value")
        error = TransportError(
            message="Transfer failed",
            transport="scp",
            target="remote-host",
            cause=cause,
        )
        msg = str(error)
        assert "Transfer failed" in msg
        assert "scp" in msg
        assert "remote-host" in msg
        assert error.cause is cause

    def test_error_is_exception(self):
        """TransportError should be an Exception subclass."""
        error = TransportError(message="Error")
        assert isinstance(error, Exception)

        # Should be raisable
        with pytest.raises(TransportError):
            raise error


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Default config values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 300.0
        assert config.half_open_max_calls == 1

    def test_aggressive_factory(self):
        """Aggressive config for unreliable targets."""
        config = CircuitBreakerConfig.aggressive()
        assert config.failure_threshold == 2
        assert config.recovery_timeout == 60.0

    def test_patient_factory(self):
        """Patient config for occasionally-failing targets."""
        config = CircuitBreakerConfig.patient()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 600.0


# =============================================================================
# TimeoutConfig Tests
# =============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_default_values(self):
        """Default timeout values."""
        config = TimeoutConfig()
        assert config.connect_timeout == 30
        assert config.operation_timeout == 180
        assert config.http_timeout == 30

    def test_fast_factory(self):
        """Fast config for responsive targets."""
        config = TimeoutConfig.fast()
        assert config.connect_timeout == 10
        assert config.operation_timeout == 60
        assert config.http_timeout == 15

    def test_slow_factory(self):
        """Slow config for slow networks."""
        config = TimeoutConfig.slow()
        assert config.connect_timeout == 60
        assert config.operation_timeout == 600
        assert config.http_timeout == 120


# =============================================================================
# TransportBase Initialization Tests
# =============================================================================


class TestTransportBaseInit:
    """Tests for TransportBase initialization."""

    def test_default_name(self):
        """Name defaults to class name."""
        transport = ConcreteTransport()
        assert transport.name == "ConcreteTransport"

    def test_custom_name(self):
        """Custom name is used."""
        transport = ConcreteTransport(name="MyTransport")
        assert transport.name == "MyTransport"

    def test_default_timeout_config(self):
        """Default timeout config is applied."""
        transport = ConcreteTransport()
        assert transport.connect_timeout == 30
        assert transport.operation_timeout == 180
        assert transport.http_timeout == 30

    def test_custom_timeout_config(self):
        """Custom timeout config is applied."""
        config = TimeoutConfig.fast()
        transport = ConcreteTransport(timeout_config=config)
        assert transport.connect_timeout == 10
        assert transport.operation_timeout == 60

    def test_initial_metrics(self):
        """Initial metrics are zero."""
        transport = ConcreteTransport()
        assert transport._total_operations == 0
        assert transport._total_successes == 0
        assert transport._total_failures == 0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_new_target_allows_attempt(self, transport):
        """New target should allow attempts."""
        assert transport.can_attempt("new-host") is True

    def test_circuit_closed_after_success(self, transport):
        """Circuit should be closed after success."""
        transport.record_success("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.CLOSED

    def test_failures_under_threshold_dont_open(self, transport):
        """Failures under threshold don't open circuit."""
        # Default threshold is 3
        transport.record_failure("host-1")
        transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.CLOSED
        assert transport.can_attempt("host-1") is True

    def test_failures_at_threshold_open_circuit(self, transport):
        """Failures at threshold open the circuit."""
        # Default threshold is 3
        transport.record_failure("host-1")
        transport.record_failure("host-1")
        transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.OPEN

    def test_open_circuit_blocks_attempts(self, transport):
        """Open circuit should block attempts."""
        # Open the circuit
        for _ in range(3):
            transport.record_failure("host-1")
        assert transport.can_attempt("host-1") is False

    def test_circuit_transitions_to_half_open(self, transport):
        """Circuit should transition to half-open after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            transport.record_failure("host-1")

        # Mock time to simulate recovery timeout passing
        with patch("time.time") as mock_time:
            # Initial failure time
            mock_time.return_value = 1000.0
            status = transport._target_status["host-1"]
            status.last_failure_time = 1000.0

            # After recovery timeout (default 300s)
            mock_time.return_value = 1400.0  # 400 seconds later
            assert transport.can_attempt("host-1") is True
            assert transport.get_circuit_state("host-1") == TransportState.HALF_OPEN

    def test_success_closes_half_open_circuit(self, transport):
        """Success during half-open should close the circuit."""
        # Get circuit to half-open state
        status = transport._get_or_create_status("host-1")
        status.state = TransportState.HALF_OPEN

        # Record success
        transport.record_success("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.CLOSED

    def test_failure_reopens_half_open_circuit(self, transport):
        """Failure during half-open should reopen the circuit."""
        # Get circuit to half-open state
        status = transport._get_or_create_status("host-1")
        status.state = TransportState.HALF_OPEN

        # Record failure
        transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.OPEN

    def test_success_resets_failure_count(self, transport):
        """Success should reset failure count."""
        transport.record_failure("host-1")
        transport.record_failure("host-1")
        assert transport._target_status["host-1"].failure_count == 2

        transport.record_success("host-1")
        assert transport._target_status["host-1"].failure_count == 0

    def test_reset_circuit(self, transport):
        """Reset should clear circuit state."""
        for _ in range(3):
            transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.OPEN

        transport.reset_circuit("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.CLOSED

    def test_reset_all_circuits(self, transport):
        """Reset all should clear all circuits."""
        for _ in range(3):
            transport.record_failure("host-1")
            transport.record_failure("host-2")

        transport.reset_all_circuits()
        assert len(transport._target_status) == 0

    def test_get_all_circuit_states(self, transport):
        """Get all states returns copy of states."""
        transport.record_success("host-1")
        transport.record_failure("host-2")

        states = transport.get_all_circuit_states()
        assert "host-1" in states
        assert "host-2" in states
        # Should be a copy
        assert states is not transport._target_status

    def test_metrics_updated_on_operations(self, transport):
        """Metrics should be updated on operations."""
        transport.record_success("host-1")
        transport.record_success("host-1")
        transport.record_failure("host-2")

        assert transport._total_operations == 3
        assert transport._total_successes == 2
        assert transport._total_failures == 1


# =============================================================================
# Timeout Tests
# =============================================================================


class TestTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self, transport):
        """Successful operation completes within timeout."""

        async def fast_op():
            return "result"

        result = await transport.execute_with_timeout(fast_op(), timeout=5.0)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_raises_on_timeout(self, transport):
        """Timeout raises TransportError."""

        async def slow_op():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_timeout(slow_op(), timeout=0.01)

        assert "timed out" in str(exc_info.value).lower() or "ConcreteTransport" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_execute_with_timeout_uses_default(self, transport):
        """Default timeout from config is used."""
        transport.operation_timeout = 0.01

        async def slow_op():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(TransportError):
            await transport.execute_with_timeout(slow_op())

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, transport):
        """Successful operation on first try."""

        async def op():
            return "success"

        result = await transport.execute_with_retry(
            operation=lambda: op(),
            target="host-1",
            max_retries=3,
        )
        assert result == "success"
        assert transport._total_successes == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_on_failure(self, transport):
        """Failed operations are retried."""
        attempts = 0

        async def failing_then_success():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Fail")
            return "success"

        result = await transport.execute_with_retry(
            operation=lambda: failing_then_success(),
            target="host-1",
            max_retries=3,
            backoff_base=0.01,  # Fast backoff for testing
        )
        assert result == "success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_fails_after_max_retries(self, transport):
        """Raises error after max retries exhausted."""

        async def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_retry(
                operation=lambda: always_fail(),
                target="host-1",
                max_retries=2,
                backoff_base=0.01,
            )

        assert "3 attempts failed" in str(exc_info.value)  # 0 + 2 retries = 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_respects_circuit_breaker(self, transport):
        """Retry stops if circuit is open."""
        # Open the circuit first
        for _ in range(3):
            transport.record_failure("host-1")

        async def op():
            return "never"

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_retry(
                operation=lambda: op(),
                target="host-1",
                max_retries=3,
            )

        assert "Circuit open" in str(exc_info.value)


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence."""

    def test_load_state_empty_on_no_path(self, transport):
        """No state path returns empty dict."""
        result = transport._load_state()
        assert result == {}

    def test_load_state_empty_on_missing_file(self, transport_with_state):
        """Missing file returns empty dict."""
        result = transport_with_state._load_state()
        assert result == {}

    def test_save_and_load_state(self, transport_with_state):
        """State can be saved and loaded."""
        data = {"key": "value", "count": 42}
        assert transport_with_state._save_state(data) is True

        loaded = transport_with_state._load_state()
        assert loaded == data

    def test_save_state_creates_parent_dirs(self, tmp_path):
        """Save creates parent directories if needed."""
        state_file = tmp_path / "subdir" / "deep" / "state.json"
        transport = ConcreteTransport(state_path=state_file)

        assert transport._save_state({"test": True}) is True
        assert state_file.exists()

    def test_save_state_returns_false_on_no_path(self, transport):
        """Save returns False with no path."""
        result = transport._save_state({"data": "test"})
        assert result is False

    def test_load_state_handles_corrupted_json(self, transport_with_state):
        """Corrupted JSON returns empty dict."""
        # Write invalid JSON
        transport_with_state._state_path.write_text("not valid json {")

        result = transport_with_state._load_state()
        assert result == {}


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_idle(self, transport):
        """Health check with no targets."""
        health = transport.health_check()
        assert health["healthy"] is True
        assert health["status"] == "idle"
        assert health["details"]["total_targets"] == 0

    def test_health_check_healthy(self, transport):
        """Health check with all circuits closed."""
        transport.record_success("host-1")
        transport.record_success("host-2")

        health = transport.health_check()
        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert health["details"]["closed_circuits"] == 2
        assert health["details"]["open_circuits"] == 0

    def test_health_check_degraded(self, transport):
        """Health check with some circuits open."""
        transport.record_success("host-1")
        for _ in range(3):
            transport.record_failure("host-2")

        health = transport.health_check()
        assert health["healthy"] is True
        assert health["status"] == "degraded"
        assert health["details"]["open_circuits"] == 1
        assert health["details"]["closed_circuits"] == 1

    def test_health_check_unhealthy(self, transport):
        """Health check with all circuits open."""
        for _ in range(3):
            transport.record_failure("host-1")
            transport.record_failure("host-2")

        health = transport.health_check()
        assert health["healthy"] is False
        assert health["status"] == "unhealthy"
        assert health["details"]["open_circuits"] == 2

    def test_health_check_includes_metrics(self, transport):
        """Health check includes operation metrics."""
        transport.record_success("host-1")
        transport.record_success("host-1")
        transport.record_failure("host-1")

        health = transport.health_check()
        details = health["details"]
        assert details["total_operations"] == 3
        assert details["total_successes"] == 2
        assert details["total_failures"] == 1
        assert details["success_rate"] == pytest.approx(0.667, abs=0.01)

    def test_health_check_includes_transport_name(self, transport):
        """Health check includes transport name."""
        health = transport.health_check()
        assert health["details"]["transport_name"] == "ConcreteTransport"


# =============================================================================
# Custom Config Tests
# =============================================================================


class TestCustomConfig:
    """Tests for custom configuration."""

    def test_custom_failure_threshold(self):
        """Custom failure threshold is respected."""
        config = CircuitBreakerConfig(failure_threshold=5)
        transport = ConcreteTransport(circuit_breaker_config=config)

        # 4 failures should not open circuit
        for _ in range(4):
            transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.CLOSED

        # 5th failure should open
        transport.record_failure("host-1")
        assert transport.get_circuit_state("host-1") == TransportState.OPEN

    def test_custom_recovery_timeout(self):
        """Custom recovery timeout is respected."""
        config = CircuitBreakerConfig(recovery_timeout=60.0)
        transport = ConcreteTransport(circuit_breaker_config=config)

        # Open circuit
        for _ in range(3):
            transport.record_failure("host-1")

        with patch("time.time") as mock_time:
            # Set failure time
            status = transport._target_status["host-1"]
            status.last_failure_time = 1000.0

            # 30 seconds later - still blocked
            mock_time.return_value = 1030.0
            assert transport.can_attempt("host-1") is False

            # 70 seconds later - should allow
            mock_time.return_value = 1070.0
            assert transport.can_attempt("host-1") is True
