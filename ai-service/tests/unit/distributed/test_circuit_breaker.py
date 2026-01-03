"""Tests for circuit breaker pattern implementation.

These tests verify:
1. CircuitState enum and CircuitStatus dataclass
2. CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
3. Failure threshold and recovery timeout behavior
4. Exponential backoff with jitter
5. Context managers and decorators
6. CircuitBreakerRegistry and FallbackChain
7. Thread safety
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CircuitStatus,
    FallbackChain,
    format_circuit_status,
    get_adaptive_timeout,
    get_circuit_registry,
    get_host_breaker,
    get_operation_breaker,
    get_training_breaker,
    set_host_breaker_callback,
    with_circuit_breaker,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self):
        """CircuitState should have correct string values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_state_enum_members(self):
        """CircuitState should have exactly 3 members."""
        assert len(CircuitState) == 3


class TestCircuitStatus:
    """Tests for CircuitStatus dataclass."""

    def test_status_creation(self):
        """CircuitStatus should be created with all fields."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            half_open_at=None,
        )
        assert status.target == "host1"
        assert status.state == CircuitState.CLOSED
        assert status.failure_count == 0
        assert status.success_count == 5

    def test_time_since_open_when_closed(self):
        """time_since_open should be None when circuit not opened."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            opened_at=None,
            half_open_at=None,
        )
        assert status.time_since_open is None

    def test_time_since_open_when_open(self):
        """time_since_open should return elapsed seconds when open."""
        opened_at = time.time() - 10.0
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=opened_at,
            half_open_at=None,
        )
        assert status.time_since_open is not None
        assert 9.5 < status.time_since_open < 11.0

    def test_to_dict(self):
        """to_dict should return all fields as dict."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=123.456,
            last_success_time=None,
            opened_at=100.0,
            half_open_at=None,
            consecutive_opens=2,
        )
        d = status.to_dict()
        assert d["target"] == "host1"
        assert d["state"] == "open"
        assert d["failure_count"] == 3
        assert d["consecutive_opens"] == 2


class TestCircuitBreakerBasics:
    """Basic CircuitBreaker functionality tests."""

    def test_default_state_is_closed(self):
        """New circuit should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_can_execute_when_closed(self):
        """can_execute should return True when CLOSED."""
        breaker = CircuitBreaker()
        assert breaker.can_execute("host1") is True

    def test_record_success_increments_count(self):
        """record_success should increment success_count."""
        breaker = CircuitBreaker()
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_record_failure_increments_count(self):
        """record_failure should increment failure_count."""
        breaker = CircuitBreaker()
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_success_resets_failure_count(self):
        """Success should reset failure_count to 0 in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_status("host1").failure_count == 2

        breaker.record_success("host1")
        assert breaker.get_status("host1").failure_count == 0


class TestCircuitBreakerStateTransitions:
    """Tests for circuit state transitions."""

    def test_opens_after_threshold_failures(self):
        """Circuit should OPEN after failure_threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        for i in range(3):
            assert breaker.get_state("host1") == CircuitState.CLOSED
            breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_blocks_requests_when_open(self):
        """can_execute should return False when OPEN."""
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN
        assert breaker.can_execute("host1") is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after recovery_timeout."""
        # Use longer timeout to account for exponential backoff (2^1 = 2x base)
        # With consecutive_opens=1, timeout = 0.1 * 2 = 0.2s + jitter
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            backoff_multiplier=1.0,  # Disable exponential backoff for this test
            jitter_factor=0.0,  # Disable jitter for predictable timing
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        time.sleep(0.15)

        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        """HALF_OPEN should allow up to half_open_max_calls."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)

        # Should allow 2 calls
        assert breaker.can_execute("host1") is True
        assert breaker.can_execute("host1") is True
        # Third should be blocked
        assert breaker.can_execute("host1") is False

    def test_success_in_half_open_closes_circuit(self):
        """Success in HALF_OPEN should close the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        breaker.record_success("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN should reopen the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_consecutive_opens_tracked(self):
        """consecutive_opens should increment on each open."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        # First open
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 1

        # Wait and transition to half-open
        time.sleep(0.1)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        # Fail again (reopens)
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 2

    def test_consecutive_opens_reset_on_success(self):
        """consecutive_opens should reset after successful recovery."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        breaker.record_failure("host1")
        time.sleep(0.1)
        # Transition to half-open and fail again
        breaker.can_execute("host1")  # Triggers half-open check
        breaker.record_failure("host1")
        time.sleep(0.1)

        assert breaker.get_status("host1").consecutive_opens >= 2

        # Trigger half-open and recover
        breaker.can_execute("host1")
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 0
        assert status.state == CircuitState.CLOSED


class TestCircuitBreakerReset:
    """Tests for circuit reset functionality."""

    def test_reset_returns_to_closed(self):
        """reset should return circuit to CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        breaker.reset("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_reset_clears_failure_count(self):
        """reset should clear failure count."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        breaker.reset("host1")
        status = breaker.get_status("host1")
        assert status.failure_count == 0

    def test_reset_all_clears_all_circuits(self):
        """reset_all should clear all tracked circuits."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")
        breaker.record_failure("host2")

        breaker.reset_all()

        assert breaker.get_state("host1") == CircuitState.CLOSED
        assert breaker.get_state("host2") == CircuitState.CLOSED

    def test_force_open(self):
        """force_open should immediately open circuit."""
        breaker = CircuitBreaker()
        assert breaker.get_state("host1") == CircuitState.CLOSED

        breaker.force_open("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_force_close(self):
        """force_close should immediately close circuit."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        breaker.force_close("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED


class TestContextManagers:
    """Tests for context manager functionality."""

    def test_protected_sync_records_success(self):
        """protected_sync should record success on normal exit."""
        breaker = CircuitBreaker()

        with breaker.protected_sync("host1"):
            pass  # Simulated successful operation

        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_protected_sync_records_failure(self):
        """protected_sync should record failure on exception."""
        breaker = CircuitBreaker()

        with pytest.raises(ValueError):
            with breaker.protected_sync("host1"):
                raise ValueError("Test error")

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_protected_sync_raises_when_open(self):
        """protected_sync should raise CircuitOpenError when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            with breaker.protected_sync("host1"):
                pass

    @pytest.mark.asyncio
    async def test_protected_async_records_success(self):
        """protected should record success on normal exit."""
        breaker = CircuitBreaker()

        async with breaker.protected("host1"):
            await asyncio.sleep(0.01)

        status = breaker.get_status("host1")
        assert status.success_count == 1

    @pytest.mark.asyncio
    async def test_protected_async_records_failure(self):
        """protected should record failure on exception."""
        breaker = CircuitBreaker()

        with pytest.raises(ValueError):
            async with breaker.protected("host1"):
                raise ValueError("Test error")

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    @pytest.mark.asyncio
    async def test_protected_async_raises_when_open(self):
        """protected should raise CircuitOpenError when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            async with breaker.protected("host1"):
                pass


class TestExecuteMethods:
    """Tests for execute and execute_async methods."""

    def test_execute_returns_result(self):
        """execute should return function result on success."""
        breaker = CircuitBreaker()
        result = breaker.execute("host1", lambda: 42)
        assert result == 42

    def test_execute_records_success(self):
        """execute should record success."""
        breaker = CircuitBreaker()
        breaker.execute("host1", lambda: "ok")

        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_execute_records_failure(self):
        """execute should record failure on exception."""
        breaker = CircuitBreaker()

        def failing_func():
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            breaker.execute("host1", failing_func)

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_execute_with_fallback(self):
        """execute should call fallback when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        result = breaker.execute(
            "host1",
            lambda: "primary",
            fallback=lambda: "fallback",
        )
        assert result == "fallback"

    def test_execute_raises_without_fallback(self):
        """execute should raise CircuitOpenError when open without fallback."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            breaker.execute("host1", lambda: "primary")

    @pytest.mark.asyncio
    async def test_execute_async_returns_result(self):
        """execute_async should return coroutine result."""
        breaker = CircuitBreaker()

        async def async_func():
            return 42

        result = await breaker.execute_async("host1", async_func)
        assert result == 42


class TestStateChangeCallback:
    """Tests for state change callback functionality."""

    def test_callback_on_open(self):
        """Callback should be called when circuit opens."""
        callback = MagicMock()
        breaker = CircuitBreaker(failure_threshold=1, on_state_change=callback)

        breaker.record_failure("host1")

        callback.assert_called_once_with(
            "host1", CircuitState.CLOSED, CircuitState.OPEN
        )

    def test_callback_on_close(self):
        """Callback should be called when circuit closes."""
        callback = MagicMock()
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            on_state_change=callback,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        breaker.record_failure("host1")
        time.sleep(0.1)
        breaker.can_execute("host1")  # Trigger half-open
        breaker.record_success("host1")

        # Should have been called for CLOSED->OPEN and HALF_OPEN->CLOSED
        assert callback.call_count >= 2

    def test_callback_exception_does_not_crash(self):
        """Callback exception should not affect circuit operation."""
        def bad_callback(*args):
            raise RuntimeError("Callback error")

        breaker = CircuitBreaker(failure_threshold=1, on_state_change=bad_callback)

        # Should not raise
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_instance_returns_singleton(self):
        """get_instance should return the same instance."""
        reg1 = CircuitBreakerRegistry.get_instance()
        reg2 = CircuitBreakerRegistry.get_instance()
        assert reg1 is reg2

    def test_get_breaker_returns_same_breaker(self):
        """get_breaker should return same breaker for same operation type."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get_breaker("ssh")
        b2 = registry.get_breaker("ssh")
        assert b1 is b2

    def test_get_breaker_creates_different_breakers(self):
        """get_breaker should return different breakers for different types."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get_breaker("ssh")
        b2 = registry.get_breaker("http")
        assert b1 is not b2

    def test_get_timeout_normal(self):
        """get_timeout should return default when circuit closed."""
        registry = CircuitBreakerRegistry()
        timeout = registry.get_timeout("ssh", "host1", 60.0)
        assert timeout == 60.0

    def test_get_timeout_half_open(self):
        """get_timeout should return shorter timeout in half-open."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get_breaker("ssh")

        # Force to half-open
        breaker.force_open("host1")
        breaker._circuits["host1"].state = CircuitState.HALF_OPEN

        timeout = registry.get_timeout("ssh", "host1", 60.0)
        assert timeout < 60.0  # Should be reduced

    def test_get_all_open_circuits(self):
        """get_all_open_circuits should return only non-closed circuits."""
        registry = CircuitBreakerRegistry()

        ssh_breaker = registry.get_breaker("ssh")
        ssh_breaker.force_open("host1")

        http_breaker = registry.get_breaker("http")
        http_breaker.record_success("host2")  # Keep closed

        open_circuits = registry.get_all_open_circuits()
        assert "ssh" in open_circuits
        assert "host1" in open_circuits["ssh"]
        assert "http" not in open_circuits


class TestGlobalBreakers:
    """Tests for global circuit breaker functions."""

    def test_get_host_breaker_returns_breaker(self):
        """get_host_breaker should return a CircuitBreaker."""
        breaker = get_host_breaker()
        assert isinstance(breaker, CircuitBreaker)

    def test_get_training_breaker_returns_breaker(self):
        """get_training_breaker should return a CircuitBreaker."""
        breaker = get_training_breaker()
        assert isinstance(breaker, CircuitBreaker)

    def test_get_operation_breaker_returns_breaker(self):
        """get_operation_breaker should return a CircuitBreaker."""
        breaker = get_operation_breaker("test_op")
        assert isinstance(breaker, CircuitBreaker)

    def test_get_circuit_registry_returns_registry(self):
        """get_circuit_registry should return CircuitBreakerRegistry."""
        registry = get_circuit_registry()
        assert isinstance(registry, CircuitBreakerRegistry)

    def test_get_adaptive_timeout(self):
        """get_adaptive_timeout should return appropriate timeout."""
        timeout = get_adaptive_timeout("ssh", "host1", 60.0)
        assert isinstance(timeout, float)


class TestFormatCircuitStatus:
    """Tests for format_circuit_status function."""

    def test_format_closed_status(self):
        """format_circuit_status should show checkmark for closed."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            half_open_at=None,
        )
        formatted = format_circuit_status(status)
        assert "✓" in formatted
        assert "host1" in formatted
        assert "closed" in formatted

    def test_format_open_status(self):
        """format_circuit_status should show X for open."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time(),
            half_open_at=None,
        )
        formatted = format_circuit_status(status)
        assert "✗" in formatted
        assert "failures=3" in formatted

    def test_format_half_open_status(self):
        """format_circuit_status should show half-circle for half-open."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.HALF_OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time() - 60,
            half_open_at=time.time(),
        )
        formatted = format_circuit_status(status)
        assert "◐" in formatted
        assert "half_open" in formatted


class TestFallbackChain:
    """Tests for FallbackChain class."""

    def test_add_operation_returns_self(self):
        """add_operation should return self for chaining."""
        chain = FallbackChain()
        result = chain.add_operation("op1", lambda: None, 10.0)
        assert result is chain

    def test_remaining_budget_initial(self):
        """remaining_budget should equal total_timeout initially."""
        chain = FallbackChain(total_timeout=100.0)
        assert chain.remaining_budget == 100.0

    @pytest.mark.asyncio
    async def test_execute_returns_first_success(self):
        """execute should return result from first successful operation."""
        chain = FallbackChain(total_timeout=10.0)

        async def op1(host, timeout):
            return "result1"

        chain.add_operation("op1", op1, 5.0)

        result = await chain.execute(host="host1")
        assert result == "result1"

    @pytest.mark.asyncio
    async def test_execute_skips_open_circuits(self):
        """execute should skip operations with open circuits."""
        chain = FallbackChain(total_timeout=10.0)

        # Open the circuit for op1
        registry = get_circuit_registry()
        breaker = registry.get_breaker("test_skip_open")
        breaker.force_open("host1")

        async def op1(host, timeout):
            return "should_not_reach"

        async def op2(host, timeout):
            return "fallback_result"

        chain.add_operation("test_skip_open", op1, 5.0)
        chain.add_operation("test_skip_fallback", op2, 5.0)

        result = await chain.execute(host="host1")
        assert result == "fallback_result"


class TestWithCircuitBreakerDecorator:
    """Tests for with_circuit_breaker decorator."""

    def test_decorator_allows_when_closed(self):
        """Decorated function should execute when circuit is closed."""
        @with_circuit_breaker("test_decorator_closed")
        def my_func(host: str):
            return f"success for {host}"

        result = my_func(host="test_host")
        assert result == "success for test_host"

    def test_decorator_blocks_when_open(self):
        """Decorated function should raise when circuit is open."""
        # First, open the circuit
        breaker = get_operation_breaker("test_decorator_open")
        breaker.force_open("block_host")

        @with_circuit_breaker("test_decorator_open")
        def my_func(host: str):
            return "should not reach"

        with pytest.raises(CircuitOpenError):
            my_func(host="block_host")

    def test_decorator_records_success(self):
        """Decorated function should record success."""
        @with_circuit_breaker("test_decorator_success")
        def my_func(host: str):
            return "ok"

        my_func(host="success_host")

        breaker = get_operation_breaker("test_decorator_success")
        status = breaker.get_status("success_host")
        assert status.success_count >= 1

    def test_decorator_records_failure(self):
        """Decorated function should record failure on exception."""
        @with_circuit_breaker("test_decorator_failure")
        def my_func(host: str):
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            my_func(host="fail_host")

        breaker = get_operation_breaker("test_decorator_failure")
        status = breaker.get_status("fail_host")
        assert status.failure_count >= 1

    @pytest.mark.asyncio
    async def test_decorator_async(self):
        """Decorator should work with async functions."""
        @with_circuit_breaker("test_decorator_async")
        async def async_func(host: str):
            await asyncio.sleep(0.01)
            return "async success"

        result = await async_func(host="async_host")
        assert result == "async success"

    def test_decorator_with_custom_host_param(self):
        """Decorator should accept custom host parameter name."""
        @with_circuit_breaker("test_custom_param", host_param="hostname")
        def my_func(hostname: str):
            return f"success for {hostname}"

        result = my_func(hostname="custom_host")
        assert result == "success for custom_host"


class TestMultipleTargets:
    """Tests for handling multiple targets."""

    def test_independent_circuits_per_target(self):
        """Each target should have independent circuit state."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open circuit for host1
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        # host1 should be open
        assert breaker.get_state("host1") == CircuitState.OPEN

        # host2 should still be closed
        assert breaker.get_state("host2") == CircuitState.CLOSED
        assert breaker.can_execute("host2") is True

    def test_get_all_states_returns_all_targets(self):
        """get_all_states should return status for all tracked targets."""
        breaker = CircuitBreaker()

        breaker.record_success("host1")
        breaker.record_failure("host2")
        breaker.record_success("host3")

        states = breaker.get_all_states()

        assert "host1" in states
        assert "host2" in states
        assert "host3" in states
        assert len(states) == 3


# =============================================================================
# Test Escalation Tiers (Phase 15.1.8 - January 2026)
# =============================================================================


class TestEscalationTiers:
    """Tests for escalation tier functionality.

    Phase 15.1.8: Instead of circuits staying permanently open after
    max_consecutive_opens, they enter escalation tiers with progressively
    longer recovery periods.
    """

    def test_tier_zero_below_max_opens(self):
        """Tier should be 0 when consecutive_opens < max_consecutive_opens."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        # Open the circuit 4 times (below max of 5)
        for i in range(4):
            breaker.record_failure("host1")
            breaker.record_failure("host1")
            breaker.force_reset("host1")

        # Should still be tier 0
        circuit = breaker._get_or_create_circuit("host1")
        tier = breaker._get_escalation_tier(circuit)
        assert tier == 0

    def test_tier_progression_with_consecutive_opens(self):
        """Tier should increase as consecutive_opens exceeds max."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")

        # At exactly max_consecutive_opens
        circuit.consecutive_opens = 5
        assert breaker._get_escalation_tier(circuit) == 1

        # 5 more opens -> tier 2
        circuit.consecutive_opens = 10
        assert breaker._get_escalation_tier(circuit) == 2

        # 5 more opens -> tier 3
        circuit.consecutive_opens = 15
        assert breaker._get_escalation_tier(circuit) == 3

        # Should max out at tier 3
        circuit.consecutive_opens = 100
        assert breaker._get_escalation_tier(circuit) == 3

    def test_tier_config_values(self):
        """Each tier should have correct wait and probe_interval values."""
        breaker = CircuitBreaker()

        # Tier 0: 1 min wait, 10s probe
        config0 = breaker._get_tier_config(0)
        assert config0["wait"] == 60
        assert config0["probe_interval"] == 10

        # Tier 1: 5 min wait, 30s probe
        config1 = breaker._get_tier_config(1)
        assert config1["wait"] == 300
        assert config1["probe_interval"] == 30

        # Tier 2: 15 min wait, 60s probe
        config2 = breaker._get_tier_config(2)
        assert config2["wait"] == 900
        assert config2["probe_interval"] == 60

        # Tier 3: 1 hour wait, 5 min probe
        config3 = breaker._get_tier_config(3)
        assert config3["wait"] == 3600
        assert config3["probe_interval"] == 300

    def test_tier_config_clamped_to_max(self):
        """Tier config should clamp to max tier for out-of-range values."""
        breaker = CircuitBreaker()

        config_high = breaker._get_tier_config(100)
        config_max = breaker._get_tier_config(3)

        assert config_high == config_max

    def test_probe_timing_respects_tier_wait(self):
        """Should not probe until tier wait period has passed."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time()  # Just entered tier

        # Should not probe - haven't waited 5 minutes yet
        assert breaker._should_probe_in_tier(circuit) is False

    def test_probe_timing_allows_after_wait(self):
        """Should allow probe after tier wait period."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time() - 400  # 400s ago (> 300s wait)
        circuit.last_probe_at = None  # Never probed

        # Should allow probe - waited long enough
        assert breaker._should_probe_in_tier(circuit) is True

    def test_probe_interval_respected(self):
        """Should respect probe_interval between probes."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time() - 400  # Past wait period
        circuit.last_probe_at = time.time() - 10  # Probed 10s ago

        # Tier 1 has 30s probe interval - should not allow yet
        assert breaker._should_probe_in_tier(circuit) is False

        # After 30s - should allow
        circuit.last_probe_at = time.time() - 35
        assert breaker._should_probe_in_tier(circuit) is True

    def test_escalate_updates_tier(self):
        """_escalate should update tier and enter time."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 10  # Should be tier 2

        breaker._escalate(circuit)

        assert circuit.escalation_tier == 2
        assert circuit.escalation_entered_at is not None
        assert time.time() - circuit.escalation_entered_at < 1.0

    def test_escalate_only_updates_on_tier_change(self):
        """_escalate should not update entered_at if tier unchanged."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 6
        circuit.escalation_tier = 1
        original_entered_at = time.time() - 100
        circuit.escalation_entered_at = original_entered_at

        breaker._escalate(circuit)

        # Tier unchanged, entered_at should not change
        assert circuit.escalation_tier == 1
        assert circuit.escalation_entered_at == original_entered_at

    def test_get_time_until_next_probe_during_wait(self):
        """Should return remaining wait time during tier wait period."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time() - 100  # 100s ago

        # Tier 1 wait is 300s, so 200s remaining
        remaining = breaker._get_time_until_next_probe(circuit)
        assert remaining is not None
        assert 195 < remaining < 205  # Allow some tolerance

    def test_get_time_until_next_probe_during_interval(self):
        """Should return remaining probe interval time."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time() - 400  # Past wait
        circuit.last_probe_at = time.time() - 10  # 10s ago

        # Tier 1 probe interval is 30s, so 20s remaining
        remaining = breaker._get_time_until_next_probe(circuit)
        assert remaining is not None
        assert 15 < remaining < 25  # Allow some tolerance

    def test_get_time_until_next_probe_ready_now(self):
        """Should return 0 when probe is allowed immediately."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 5
        circuit.escalation_tier = 1
        circuit.escalation_entered_at = time.time() - 400  # Past wait
        circuit.last_probe_at = time.time() - 100  # Long ago

        remaining = breaker._get_time_until_next_probe(circuit)
        assert remaining == 0.0

    def test_successful_recovery_resets_escalation(self):
        """Successful probe should reset circuit and escalation state."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        # Put circuit in escalation by directly setting state
        # (normal failures would need state transitions to increment consecutive_opens)
        circuit = breaker._get_or_create_circuit("host1")
        circuit.state = CircuitState.OPEN
        circuit.consecutive_opens = 10  # In escalation tier 2
        circuit.escalation_tier = 2
        circuit.escalation_entered_at = time.time() - 1000

        assert circuit.consecutive_opens >= 5
        assert circuit.escalation_tier == 2

        # Reset the circuit (simulates successful recovery)
        breaker.force_reset("host1")

        circuit = breaker._get_or_create_circuit("host1")
        assert circuit.consecutive_opens == 0
        assert circuit.escalation_tier == 0
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_max_tier_still_allows_probing(self):
        """Even at max tier, probing should still be allowed after intervals."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        circuit = breaker._get_or_create_circuit("host1")
        circuit.consecutive_opens = 100  # Way past max
        circuit.escalation_tier = 3  # Max tier
        circuit.escalation_entered_at = time.time() - 10000  # Long ago
        circuit.last_probe_at = time.time() - 500  # Past 5 min interval

        # Should still allow probing at max tier
        assert breaker._should_probe_in_tier(circuit) is True

    def test_status_includes_escalation_info(self):
        """CircuitStatus should include escalation information."""
        breaker = CircuitBreaker(failure_threshold=2, max_consecutive_opens=5)

        # Put into escalation by directly setting state
        # (consecutive_opens only increments on state transitions,
        # not from repeated failures)
        circuit = breaker._get_or_create_circuit("host1")
        circuit.state = CircuitState.OPEN
        circuit.consecutive_opens = 10  # In escalation tier 2
        circuit.escalation_tier = 2
        circuit.escalation_entered_at = time.time() - 100

        status = breaker.get_status("host1")

        assert status.escalation_tier > 0
        assert status.is_escalated is True
