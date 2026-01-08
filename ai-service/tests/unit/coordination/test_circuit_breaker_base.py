"""Tests for circuit_breaker_base.py.

Sprint 17.9 / Jan 2026: Comprehensive test coverage for circuit breaker infrastructure.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.circuit_breaker_base import (
    CircuitBreakerBase,
    CircuitConfig,
    CircuitDataBase,
    CircuitState,
    CircuitStatusBase,
)


# =============================================================================
# Test fixtures and helpers
# =============================================================================


class ConcreteCircuitBreaker(CircuitBreakerBase):
    """Concrete implementation for testing the abstract base class."""

    def _create_circuit_data(self) -> CircuitDataBase:
        return CircuitDataBase()

    def _create_status(self, target: str, circuit: CircuitDataBase) -> CircuitStatusBase:
        return CircuitStatusBase(
            target=target,
            state=circuit.state,
            failure_count=circuit.failure_count,
            success_count=circuit.success_count,
            last_failure_time=circuit.last_failure_time,
            last_success_time=circuit.last_success_time,
            opened_at=circuit.opened_at,
            half_open_at=circuit.half_open_at,
            recovery_timeout=self._compute_jittered_timeout(circuit),
            consecutive_opens=circuit.consecutive_opens,
        )


@pytest.fixture
def default_config() -> CircuitConfig:
    """Create default config with no jitter for predictable tests."""
    return CircuitConfig(jitter_factor=0.0)


@pytest.fixture
def cb(default_config: CircuitConfig) -> ConcreteCircuitBreaker:
    """Create a circuit breaker with default config."""
    return ConcreteCircuitBreaker(config=default_config)


# =============================================================================
# CircuitConfig tests
# =============================================================================


class TestCircuitConfig:
    """Tests for CircuitConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are set correctly."""
        config = CircuitConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 1
        assert config.half_open_max_calls == 1
        assert config.backoff_multiplier == 2.0
        assert config.max_backoff == 300.0
        assert config.jitter_factor == 0.1
        assert config.emit_events is True
        assert config.operation_type == "default"

    def test_validation_failure_threshold(self) -> None:
        """Test that failure_threshold must be > 0."""
        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            CircuitConfig(failure_threshold=0)

    def test_validation_recovery_timeout(self) -> None:
        """Test that recovery_timeout must be > 0."""
        with pytest.raises(ValueError, match="recovery_timeout must be > 0"):
            CircuitConfig(recovery_timeout=0)

    def test_validation_success_threshold(self) -> None:
        """Test that success_threshold must be > 0."""
        with pytest.raises(ValueError, match="success_threshold must be > 0"):
            CircuitConfig(success_threshold=0)

    def test_validation_half_open_max_calls(self) -> None:
        """Test that half_open_max_calls must be > 0."""
        with pytest.raises(ValueError, match="half_open_max_calls must be > 0"):
            CircuitConfig(half_open_max_calls=0)

    def test_validation_backoff_multiplier(self) -> None:
        """Test that backoff_multiplier must be >= 1.0."""
        with pytest.raises(ValueError, match="backoff_multiplier must be >= 1.0"):
            CircuitConfig(backoff_multiplier=0.5)

    def test_validation_max_backoff(self) -> None:
        """Test that max_backoff must be > 0."""
        with pytest.raises(ValueError, match="max_backoff must be > 0"):
            CircuitConfig(max_backoff=0)

    def test_validation_jitter_factor_low(self) -> None:
        """Test that jitter_factor must be >= 0."""
        with pytest.raises(ValueError, match="jitter_factor must be between 0 and 1"):
            CircuitConfig(jitter_factor=-0.1)

    def test_validation_jitter_factor_high(self) -> None:
        """Test that jitter_factor must be <= 1."""
        with pytest.raises(ValueError, match="jitter_factor must be between 0 and 1"):
            CircuitConfig(jitter_factor=1.5)

    def test_for_transport_http(self) -> None:
        """Test HTTP transport config."""
        config = CircuitConfig.for_transport("http")
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.operation_type == "http"

    def test_for_transport_ssh(self) -> None:
        """Test SSH transport config."""
        config = CircuitConfig.for_transport("ssh")
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.operation_type == "ssh"

    def test_for_transport_rsync(self) -> None:
        """Test rsync transport config."""
        config = CircuitConfig.for_transport("rsync")
        assert config.failure_threshold == 2
        assert config.recovery_timeout == 90.0
        assert config.operation_type == "rsync"

    def test_for_transport_p2p(self) -> None:
        """Test P2P transport config."""
        config = CircuitConfig.for_transport("p2p")
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 45.0
        assert config.operation_type == "p2p"

    def test_for_transport_unknown(self) -> None:
        """Test unknown transport uses defaults with custom operation_type."""
        config = CircuitConfig.for_transport("unknown")
        assert config.failure_threshold == 5  # Default
        assert config.recovery_timeout == 60.0  # Default
        assert config.operation_type == "unknown"


# =============================================================================
# CircuitDataBase tests
# =============================================================================


class TestCircuitDataBase:
    """Tests for CircuitDataBase dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        data = CircuitDataBase()
        assert data.state == CircuitState.CLOSED
        assert data.failure_count == 0
        assert data.success_count == 0
        assert data.last_failure_time is None
        assert data.last_success_time is None
        assert data.opened_at is None
        assert data.half_open_at is None
        assert data.half_open_calls == 0
        assert data.consecutive_opens == 0
        assert data.jitter_offset == 0.0


# =============================================================================
# CircuitStatusBase tests
# =============================================================================


class TestCircuitStatusBase:
    """Tests for CircuitStatusBase dataclass."""

    def test_time_since_open_when_not_open(self) -> None:
        """Test time_since_open returns None when not opened."""
        status = CircuitStatusBase(
            target="test",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            opened_at=None,
            half_open_at=None,
            recovery_timeout=60.0,
        )
        assert status.time_since_open is None

    def test_time_since_open_when_open(self) -> None:
        """Test time_since_open returns correct value."""
        opened_time = time.time() - 10.0  # 10 seconds ago
        status = CircuitStatusBase(
            target="test",
            state=CircuitState.OPEN,
            failure_count=5,
            success_count=0,
            last_failure_time=opened_time,
            last_success_time=None,
            opened_at=opened_time,
            half_open_at=None,
            recovery_timeout=60.0,
        )
        assert status.time_since_open is not None
        assert 9.9 < status.time_since_open < 11.0

    def test_time_until_recovery_when_closed(self) -> None:
        """Test time_until_recovery returns 0 when closed."""
        status = CircuitStatusBase(
            target="test",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            opened_at=None,
            half_open_at=None,
            recovery_timeout=60.0,
        )
        assert status.time_until_recovery == 0.0

    def test_time_until_recovery_when_open(self) -> None:
        """Test time_until_recovery returns remaining time."""
        opened_time = time.time() - 10.0  # 10 seconds ago
        status = CircuitStatusBase(
            target="test",
            state=CircuitState.OPEN,
            failure_count=5,
            success_count=0,
            last_failure_time=opened_time,
            last_success_time=None,
            opened_at=opened_time,
            half_open_at=None,
            recovery_timeout=60.0,
        )
        # 60 - 10 = ~50 seconds
        assert 49.0 < status.time_until_recovery < 51.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        status = CircuitStatusBase(
            target="node-1",
            state=CircuitState.OPEN,
            failure_count=5,
            success_count=2,
            last_failure_time=1000.0,
            last_success_time=900.0,
            opened_at=990.0,
            half_open_at=None,
            recovery_timeout=60.0,
            consecutive_opens=2,
        )
        d = status.to_dict()
        assert d["target"] == "node-1"
        assert d["state"] == "open"
        assert d["failure_count"] == 5
        assert d["success_count"] == 2
        assert d["recovery_timeout"] == 60.0
        assert d["consecutive_opens"] == 2


# =============================================================================
# CircuitBreakerBase tests - Core functionality
# =============================================================================


class TestCircuitBreakerBaseCore:
    """Tests for core circuit breaker functionality."""

    def test_initial_state_is_closed(self, cb: ConcreteCircuitBreaker) -> None:
        """Test that new circuits start closed."""
        assert cb.get_state("test") == CircuitState.CLOSED
        assert cb.can_execute("test") is True

    def test_can_execute_when_closed(self, cb: ConcreteCircuitBreaker) -> None:
        """Test can_execute returns True when closed."""
        assert cb.can_execute("test") is True

    def test_record_success_increments_counter(self, cb: ConcreteCircuitBreaker) -> None:
        """Test record_success increments counter."""
        cb.record_success("test")
        status = cb.get_status("test")
        assert status.success_count == 1

    def test_record_failure_increments_counter(self, cb: ConcreteCircuitBreaker) -> None:
        """Test record_failure increments counter."""
        cb.record_failure("test")
        status = cb.get_status("test")
        assert status.failure_count == 1

    def test_circuit_opens_after_threshold(self, cb: ConcreteCircuitBreaker) -> None:
        """Test circuit opens after failure threshold reached."""
        for _ in range(5):  # Default threshold is 5
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN
        assert cb.can_execute("test") is False

    def test_circuit_remains_closed_below_threshold(
        self, cb: ConcreteCircuitBreaker
    ) -> None:
        """Test circuit stays closed below threshold."""
        for _ in range(4):  # Just below threshold of 5
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.CLOSED
        assert cb.can_execute("test") is True

    def test_success_resets_failure_count(self, cb: ConcreteCircuitBreaker) -> None:
        """Test success resets failure count when closed."""
        cb.record_failure("test")
        cb.record_failure("test")
        cb.record_success("test")
        status = cb.get_status("test")
        assert status.failure_count == 0

    def test_can_execute_false_when_open(self, cb: ConcreteCircuitBreaker) -> None:
        """Test can_execute returns False when open."""
        for _ in range(5):
            cb.record_failure("test")
        assert cb.can_execute("test") is False


# =============================================================================
# CircuitBreakerBase tests - Half-open state
# =============================================================================


class TestCircuitBreakerHalfOpen:
    """Tests for half-open state transitions."""

    def test_transition_to_half_open_after_timeout(self) -> None:
        """Test circuit transitions to half-open after recovery timeout.

        Note: The transition is triggered by can_execute(), not get_state().
        """
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=0.1,
            backoff_multiplier=1.0,  # Disable backoff for predictable timing
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open the circuit
        for _ in range(3):
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

        # Wait for recovery timeout (0.1s * 1^1 = 0.1s)
        time.sleep(0.15)

        # can_execute() triggers the transition to half-open
        # and returns True for the first half-open call
        assert cb.can_execute("test") is True
        assert cb.get_state("test") == CircuitState.HALF_OPEN

    def test_can_execute_in_half_open_limited(self) -> None:
        """Test that half-open allows limited calls."""
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,  # Disable backoff for predictable timing
            half_open_max_calls=2,
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure("test")
        time.sleep(0.1)

        # First two calls should succeed (triggers transition and counts)
        assert cb.can_execute("test") is True
        assert cb.can_execute("test") is True
        # Third call should fail (max_calls=2)
        assert cb.can_execute("test") is False

    def test_success_in_half_open_closes_circuit(self) -> None:
        """Test success in half-open closes the circuit."""
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,  # Disable backoff for predictable timing
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure("test")
        time.sleep(0.1)

        # Trigger transition to half-open via can_execute
        assert cb.can_execute("test") is True
        assert cb.get_state("test") == CircuitState.HALF_OPEN

        # Record success - closes circuit
        cb.record_success("test")
        assert cb.get_state("test") == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self) -> None:
        """Test failure in half-open re-opens the circuit."""
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,  # Disable backoff for predictable timing
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open and wait for half-open
        for _ in range(3):
            cb.record_failure("test")
        time.sleep(0.1)

        # Trigger transition to half-open via can_execute
        assert cb.can_execute("test") is True
        assert cb.get_state("test") == CircuitState.HALF_OPEN

        # Record failure in half-open - reopens circuit
        cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN


# =============================================================================
# CircuitBreakerBase tests - Backoff and jitter
# =============================================================================


class TestCircuitBreakerBackoff:
    """Tests for exponential backoff behavior."""

    def test_consecutive_opens_tracked(self) -> None:
        """Test that consecutive opens are tracked when circuit reopens."""
        config = CircuitConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            backoff_multiplier=2.0,
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # First open
        cb.record_failure("test")
        cb.record_failure("test")
        status1 = cb.get_status("test")
        assert status1.consecutive_opens == 1
        assert status1.recovery_timeout == 2.0  # base * multiplier^1

    def test_reset_clears_consecutive_opens(self) -> None:
        """Test that reset() clears consecutive opens counter."""
        config = CircuitConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            backoff_multiplier=2.0,
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure("test")
        cb.record_failure("test")
        assert cb.get_status("test").consecutive_opens == 1

        # Reset clears the counter
        cb.reset("test")
        assert cb.get_status("test").consecutive_opens == 0

    def test_max_backoff_caps_timeout(self) -> None:
        """Test that max_backoff caps the recovery timeout."""
        config = CircuitConfig(
            failure_threshold=2,
            recovery_timeout=100.0,
            backoff_multiplier=10.0,
            max_backoff=200.0,
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Simulate many consecutive opens
        cb._circuits["test"] = CircuitDataBase(
            state=CircuitState.OPEN,
            consecutive_opens=5,  # Would be 100 * 10^5 without cap
            opened_at=time.time(),
        )

        status = cb.get_status("test")
        assert status.recovery_timeout == 200.0  # Capped at max_backoff


# =============================================================================
# CircuitBreakerBase tests - Status and summary
# =============================================================================


class TestCircuitBreakerStatus:
    """Tests for status and summary methods."""

    def test_get_status_creates_circuit_if_missing(
        self, cb: ConcreteCircuitBreaker
    ) -> None:
        """Test get_status creates circuit data for new target."""
        status = cb.get_status("new-target")
        assert status.target == "new-target"
        assert status.state == CircuitState.CLOSED
        assert status.failure_count == 0

    def test_get_all_states(self, cb: ConcreteCircuitBreaker) -> None:
        """Test get_all_states returns all tracked circuits."""
        cb.record_failure("node-1")
        cb.record_failure("node-2")
        cb.record_failure("node-2")

        states = cb.get_all_states()
        assert "node-1" in states
        assert "node-2" in states
        assert states["node-1"].failure_count == 1
        assert states["node-2"].failure_count == 2

    def test_get_open_circuits(self, cb: ConcreteCircuitBreaker) -> None:
        """Test get_open_circuits returns only open circuits."""
        # Open node-1
        for _ in range(5):
            cb.record_failure("node-1")

        # Keep node-2 closed
        cb.record_failure("node-2")

        open_circuits = cb.get_open_circuits()
        assert "node-1" in open_circuits
        assert "node-2" not in open_circuits

    def test_get_summary(self, cb: ConcreteCircuitBreaker) -> None:
        """Test get_summary returns correct counts."""
        # Open node-1
        for _ in range(5):
            cb.record_failure("node-1")

        # Keep node-2 and node-3 closed
        cb.record_failure("node-2")
        cb.record_success("node-3")

        summary = cb.get_summary()
        assert summary["total_targets"] == 3
        assert summary["open"] == 1
        assert summary["closed"] == 2
        assert summary["half_open"] == 0
        assert "node-1" in summary["open_targets"]


# =============================================================================
# CircuitBreakerBase tests - Reset operations
# =============================================================================


class TestCircuitBreakerReset:
    """Tests for reset operations."""

    def test_reset_single_circuit(self, cb: ConcreteCircuitBreaker) -> None:
        """Test resetting a single circuit."""
        # Open the circuit
        for _ in range(5):
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

        # Reset it
        cb.reset("test")
        assert cb.get_state("test") == CircuitState.CLOSED
        status = cb.get_status("test")
        assert status.failure_count == 0

    def test_reset_all(self, cb: ConcreteCircuitBreaker) -> None:
        """Test resetting all circuits."""
        # Open multiple circuits
        for node in ["node-1", "node-2", "node-3"]:
            for _ in range(5):
                cb.record_failure(node)

        assert len(cb.get_open_circuits()) == 3

        # Reset all
        cb.reset_all()
        assert len(cb.get_open_circuits()) == 0

    def test_force_open(self, cb: ConcreteCircuitBreaker) -> None:
        """Test force_open opens a closed circuit."""
        assert cb.get_state("test") == CircuitState.CLOSED
        cb.force_open("test")
        assert cb.get_state("test") == CircuitState.OPEN

    def test_force_close(self, cb: ConcreteCircuitBreaker) -> None:
        """Test force_close closes an open circuit."""
        for _ in range(5):
            cb.record_failure("test")
        assert cb.get_state("test") == CircuitState.OPEN

        cb.force_close("test")
        assert cb.get_state("test") == CircuitState.CLOSED

    def test_force_reset_clears_all_counters(self, cb: ConcreteCircuitBreaker) -> None:
        """Test force_reset clears all counters and state."""
        # Build up some state
        for _ in range(5):
            cb.record_failure("test")
        cb.record_success("test")

        cb.force_reset("test")
        status = cb.get_status("test")
        assert status.state == CircuitState.CLOSED
        assert status.failure_count == 0
        assert status.success_count == 0
        assert status.consecutive_opens == 0


# =============================================================================
# CircuitBreakerBase tests - TTL decay
# =============================================================================


class TestCircuitBreakerDecay:
    """Tests for TTL-based circuit decay."""

    def test_decay_old_circuits_resets_stale_open(self) -> None:
        """Test that decay_old_circuits resets old open circuits."""
        config = CircuitConfig(failure_threshold=3, jitter_factor=0.0)
        cb = ConcreteCircuitBreaker(config=config)

        # Open circuit with old timestamp
        cb._circuits["test"] = CircuitDataBase(
            state=CircuitState.OPEN,
            failure_count=5,
            opened_at=time.time() - 7200,  # 2 hours ago
        )

        result = cb.decay_old_circuits(ttl_seconds=3600)  # 1 hour TTL
        assert "test" in result["decayed"]
        assert cb.get_state("test") == CircuitState.CLOSED

    def test_decay_old_circuits_keeps_recent_open(self) -> None:
        """Test that decay_old_circuits keeps recently opened circuits."""
        # Use recovery_timeout and max_backoff longer than 30 minutes so
        # get_state() doesn't trigger HALF_OPEN transition.
        # Note: backoff is capped at max_backoff, so both must be set.
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=3600.0,  # 1 hour - longer than 30 min test age
            max_backoff=7200.0,  # 2 hours - must be >= recovery_timeout
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config)

        # Open circuit with recent timestamp (30 minutes ago)
        cb._circuits["test"] = CircuitDataBase(
            state=CircuitState.OPEN,
            failure_count=5,
            opened_at=time.time() - 1800,  # 30 minutes ago
        )

        result = cb.decay_old_circuits(ttl_seconds=3600)  # 1 hour TTL
        assert "test" not in result["decayed"]
        # Circuit should still be OPEN (not decayed, not transitioned to HALF_OPEN
        # because recovery_timeout=3600 > 1800 seconds)
        assert cb.get_state("test") == CircuitState.OPEN


# =============================================================================
# CircuitBreakerBase tests - Callback
# =============================================================================


class TestCircuitBreakerCallback:
    """Tests for state change callback."""

    def test_callback_called_on_open(self) -> None:
        """Test callback is called when circuit opens."""
        callback = MagicMock()
        config = CircuitConfig(failure_threshold=3, jitter_factor=0.0)
        cb = ConcreteCircuitBreaker(config=config, on_state_change=callback)

        for _ in range(3):
            cb.record_failure("test")

        callback.assert_called_once_with("test", CircuitState.CLOSED, CircuitState.OPEN)

    def test_callback_called_on_close(self) -> None:
        """Test callback is called when circuit closes."""
        callback = MagicMock()
        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,  # Disable backoff for predictable timing
            jitter_factor=0.0,
        )
        cb = ConcreteCircuitBreaker(config=config, on_state_change=callback)

        # Open circuit
        for _ in range(3):
            cb.record_failure("test")
        callback.reset_mock()

        # Wait for half-open transition
        time.sleep(0.1)

        # Trigger transition to HALF_OPEN via can_execute
        assert cb.can_execute("test") is True
        assert cb.get_state("test") == CircuitState.HALF_OPEN

        # Note: _check_recovery does NOT notify, so callback wasn't called for OPEN->HALF_OPEN
        # But we want to verify the HALF_OPEN->CLOSED callback
        callback.reset_mock()

        # Record success - should transition to CLOSED and call callback
        cb.record_success("test")
        callback.assert_called_with("test", CircuitState.HALF_OPEN, CircuitState.CLOSED)

    def test_callback_not_called_on_same_state(self, cb: ConcreteCircuitBreaker) -> None:
        """Test callback not called when state doesn't change."""
        callback = MagicMock()
        cb._on_state_change_callback = callback

        # Record a failure (stays closed)
        cb.record_failure("test")
        callback.assert_not_called()

    def test_callback_error_is_logged_not_raised(self) -> None:
        """Test callback errors are caught and logged."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        config = CircuitConfig(failure_threshold=3, jitter_factor=0.0)
        cb = ConcreteCircuitBreaker(config=config, on_state_change=callback)

        # Should not raise
        for _ in range(3):
            cb.record_failure("test")

        # Callback was called despite error
        callback.assert_called_once()


# =============================================================================
# CircuitBreakerBase tests - Preemptive failures
# =============================================================================


class TestCircuitBreakerPreemptive:
    """Tests for preemptive (gossip) failure handling."""

    def test_preemptive_failure_increments_count(
        self, cb: ConcreteCircuitBreaker
    ) -> None:
        """Test preemptive failure increments failure count."""
        cb.record_failure("test", preemptive=True)
        status = cb.get_status("test")
        assert status.failure_count == 1

    def test_preemptive_failure_does_not_update_time(
        self, cb: ConcreteCircuitBreaker
    ) -> None:
        """Test preemptive failure doesn't update last_failure_time."""
        cb.record_failure("test", preemptive=True)
        status = cb.get_status("test")
        assert status.last_failure_time is None

    def test_preemptive_and_regular_failures_combined(
        self, cb: ConcreteCircuitBreaker
    ) -> None:
        """Test that preemptive and regular failures combine to open circuit."""
        # 3 preemptive + 2 regular = 5 (threshold)
        for _ in range(3):
            cb.record_failure("test", preemptive=True)
        for _ in range(2):
            cb.record_failure("test", preemptive=False)

        assert cb.get_state("test") == CircuitState.OPEN


# =============================================================================
# CircuitBreakerBase tests - Thread safety
# =============================================================================


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_lock_is_reentrant(self, cb: ConcreteCircuitBreaker) -> None:
        """Test that the internal lock is reentrant."""
        # This tests that get_summary (which calls get_open_circuits) works
        # Both methods acquire the lock, so it must be reentrant
        for _ in range(5):
            cb.record_failure("test")

        # Should not deadlock
        summary = cb.get_summary()
        assert summary["open"] == 1
