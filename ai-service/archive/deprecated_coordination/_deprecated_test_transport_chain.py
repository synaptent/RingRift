"""Tests for TransportChain multi-transport failover coordinator.

Comprehensive test suite covering:
- FailoverMode enum values and string conversion
- TransportMetrics dataclass properties
- ChainResult dataclass factory methods
- TransportChain initialization, validation, and transfer methods
- Circuit breaker integration
- Metrics tracking
- Health check aggregation
- Priority management
- Callbacks

December 2025: Created for transport_chain.py test coverage.
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
from app.coordination.transport_chain import (
    ChainResult,
    FailoverMode,
    TransportChain,
    TransportMetrics,
    create_adaptive_chain,
    create_parallel_chain,
    create_sequential_chain,
)


# =============================================================================
# Mock Transport Classes
# =============================================================================


class MockTransport(TransportBase):
    """Mock transport for testing chain coordination."""

    def __init__(
        self,
        name: str = "MockTransport",
        success: bool = True,
        latency_ms: float = 10.0,
        error_message: str | None = None,
        raise_exception: bool = False,
        can_attempt_result: bool = True,
    ):
        super().__init__(name=name)
        self._success = success
        self._latency_ms = latency_ms
        self._error_message = error_message
        self._raise_exception = raise_exception
        self._can_attempt_override = can_attempt_result
        self.transfer_calls: list[tuple[Any, Any, str]] = []

    def can_attempt(self, target: str) -> bool:
        """Override can_attempt for testing."""
        if not self._can_attempt_override:
            return False
        return super().can_attempt(target)

    async def transfer(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> TransportResult:
        """Mock transfer operation."""
        self.transfer_calls.append((source, destination, target))

        # Simulate some latency
        await asyncio.sleep(self._latency_ms / 1000.0)

        if self._raise_exception:
            raise TransportError(
                message=self._error_message or "Mock exception",
                transport=self.name,
                target=target,
            )

        return TransportResult(
            success=self._success,
            transport_used=self.name,
            error=self._error_message if not self._success else None,
            latency_ms=self._latency_ms,
        )


# =============================================================================
# FailoverMode Enum Tests
# =============================================================================


class TestFailoverModeEnum:
    """Tests for FailoverMode enum."""

    def test_sequential_value(self):
        """Test SEQUENTIAL enum value."""
        assert FailoverMode.SEQUENTIAL.value == "sequential"

    def test_parallel_value(self):
        """Test PARALLEL enum value."""
        assert FailoverMode.PARALLEL.value == "parallel"

    def test_priority_value(self):
        """Test PRIORITY enum value."""
        assert FailoverMode.PRIORITY.value == "priority"

    def test_adaptive_value(self):
        """Test ADAPTIVE enum value."""
        assert FailoverMode.ADAPTIVE.value == "adaptive"

    def test_all_modes_exist(self):
        """Test all expected modes exist."""
        expected_modes = {"SEQUENTIAL", "PARALLEL", "PRIORITY", "ADAPTIVE"}
        actual_modes = {m.name for m in FailoverMode}
        assert actual_modes == expected_modes

    def test_string_conversion(self):
        """Test enum to string conversion."""
        assert str(FailoverMode.SEQUENTIAL.value) == "sequential"
        assert str(FailoverMode.PARALLEL.value) == "parallel"

    def test_enum_from_value(self):
        """Test creating enum from string value."""
        mode = FailoverMode("sequential")
        assert mode == FailoverMode.SEQUENTIAL


# =============================================================================
# TransportMetrics Dataclass Tests
# =============================================================================


class TestTransportMetrics:
    """Tests for TransportMetrics dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        metrics = TransportMetrics(transport_name="test")
        assert metrics.transport_name == "test"
        assert metrics.total_attempts == 0
        assert metrics.successes == 0
        assert metrics.failures == 0
        assert metrics.timeouts == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.last_attempt_time == 0.0
        assert metrics.last_success_time == 0.0
        assert metrics.priority_score == 1.0

    def test_success_rate_no_attempts(self):
        """Test success_rate returns 0 with no attempts."""
        metrics = TransportMetrics(transport_name="test")
        assert metrics.success_rate == 0.0

    def test_success_rate_with_attempts(self):
        """Test success_rate calculation."""
        metrics = TransportMetrics(
            transport_name="test",
            total_attempts=10,
            successes=8,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_all_successful(self):
        """Test success_rate with 100% success."""
        metrics = TransportMetrics(
            transport_name="test",
            total_attempts=5,
            successes=5,
        )
        assert metrics.success_rate == 1.0

    def test_success_rate_all_failed(self):
        """Test success_rate with 0% success."""
        metrics = TransportMetrics(
            transport_name="test",
            total_attempts=5,
            successes=0,
            failures=5,
        )
        assert metrics.success_rate == 0.0

    def test_avg_latency_no_successes(self):
        """Test avg_latency_ms returns 0 with no successes."""
        metrics = TransportMetrics(transport_name="test")
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_with_successes(self):
        """Test avg_latency_ms calculation."""
        metrics = TransportMetrics(
            transport_name="test",
            successes=4,
            total_latency_ms=100.0,
        )
        assert metrics.avg_latency_ms == 25.0

    def test_avg_latency_single_success(self):
        """Test avg_latency_ms with single success."""
        metrics = TransportMetrics(
            transport_name="test",
            successes=1,
            total_latency_ms=50.0,
        )
        assert metrics.avg_latency_ms == 50.0


# =============================================================================
# ChainResult Dataclass Tests
# =============================================================================


class TestChainResult:
    """Tests for ChainResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ChainResult(success=True)
        assert result.success is True
        assert result.result is None
        assert result.transport_used == ""
        assert result.transports_attempted == []
        assert result.total_latency_ms == 0.0
        assert result.error is None

    def test_from_transport_result_success(self):
        """Test from_transport_result with successful result."""
        transport_result = TransportResult(
            success=True,
            transport_used="TestTransport",
            latency_ms=25.0,
        )
        chain_result = ChainResult.from_transport_result(
            result=transport_result,
            transports_attempted=["T1", "T2"],
            total_latency=100.0,
        )

        assert chain_result.success is True
        assert chain_result.result == transport_result
        assert chain_result.transport_used == "TestTransport"
        assert chain_result.transports_attempted == ["T1", "T2"]
        assert chain_result.total_latency_ms == 100.0
        assert chain_result.error is None

    def test_from_transport_result_failure(self):
        """Test from_transport_result with failed result."""
        transport_result = TransportResult(
            success=False,
            transport_used="FailedTransport",
            error="Connection refused",
        )
        chain_result = ChainResult.from_transport_result(
            result=transport_result,
            transports_attempted=["T1"],
            total_latency=50.0,
        )

        assert chain_result.success is False
        assert chain_result.error == "Connection refused"
        assert chain_result.transport_used == "FailedTransport"

    def test_failure_factory_method(self):
        """Test failure class method."""
        chain_result = ChainResult.failure(
            error="All transports failed",
            transports_attempted=["T1", "T2", "T3"],
            total_latency=500.0,
        )

        assert chain_result.success is False
        assert chain_result.error == "All transports failed"
        assert chain_result.transports_attempted == ["T1", "T2", "T3"]
        assert chain_result.total_latency_ms == 500.0
        assert chain_result.result is None
        assert chain_result.transport_used == ""


# =============================================================================
# TransportChain Initialization Tests
# =============================================================================


class TestTransportChainInit:
    """Tests for TransportChain initialization."""

    def test_requires_at_least_one_transport(self):
        """Test that empty transport list raises ValueError."""
        with pytest.raises(ValueError, match="At least one transport"):
            TransportChain([])

    def test_single_transport_init(self):
        """Test initialization with single transport."""
        transport = MockTransport("T1")
        chain = TransportChain([transport])
        assert chain.transport_names == ["T1"]

    def test_multiple_transports_init(self):
        """Test initialization with multiple transports."""
        transports = [
            MockTransport("T1"),
            MockTransport("T2"),
            MockTransport("T3"),
        ]
        chain = TransportChain(transports)
        assert chain.transport_names == ["T1", "T2", "T3"]

    def test_default_failover_mode(self):
        """Test default failover mode is SEQUENTIAL."""
        chain = TransportChain([MockTransport()])
        assert chain._failover_mode == FailoverMode.SEQUENTIAL

    def test_custom_failover_mode(self):
        """Test custom failover mode."""
        chain = TransportChain(
            [MockTransport()],
            failover_mode=FailoverMode.PARALLEL,
        )
        assert chain._failover_mode == FailoverMode.PARALLEL

    def test_metrics_initialized_per_transport(self):
        """Test metrics are created for each transport."""
        transports = [MockTransport("A"), MockTransport("B")]
        chain = TransportChain(transports)

        assert "A" in chain._transport_metrics
        assert "B" in chain._transport_metrics
        assert chain._transport_metrics["A"].transport_name == "A"
        assert chain._transport_metrics["B"].transport_name == "B"


class TestTransportNames:
    """Tests for transport_names property."""

    def test_transport_names_single(self):
        """Test transport_names with single transport."""
        chain = TransportChain([MockTransport("Single")])
        assert chain.transport_names == ["Single"]

    def test_transport_names_multiple(self):
        """Test transport_names with multiple transports."""
        transports = [
            MockTransport("First"),
            MockTransport("Second"),
            MockTransport("Third"),
        ]
        chain = TransportChain(transports)
        assert chain.transport_names == ["First", "Second", "Third"]

    def test_transport_names_preserves_order(self):
        """Test transport_names preserves insertion order."""
        transports = [MockTransport(f"T{i}") for i in range(5)]
        chain = TransportChain(transports)
        assert chain.transport_names == [f"T{i}" for i in range(5)]


# =============================================================================
# TransportChain Transfer Tests - Sequential Mode
# =============================================================================


class TestSequentialTransfer:
    """Tests for sequential failover transfer."""

    @pytest.mark.asyncio
    async def test_first_transport_succeeds(self):
        """Test first transport succeeds immediately."""
        t1 = MockTransport("T1", success=True)
        t2 = MockTransport("T2", success=True)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.SEQUENTIAL)

        result = await chain.transfer("/src", "/dst", "target-host")

        assert result.success is True
        assert result.transport_used == "T1"
        assert result.transports_attempted == ["T1"]
        assert len(t1.transfer_calls) == 1
        assert len(t2.transfer_calls) == 0

    @pytest.mark.asyncio
    async def test_first_fails_second_succeeds(self):
        """Test sequential failover: first fails, second succeeds."""
        t1 = MockTransport("T1", success=False, error_message="Connection refused")
        t2 = MockTransport("T2", success=True)
        chain = TransportChain([t1, t2])

        result = await chain.transfer("/src", "/dst", "target-host")

        assert result.success is True
        assert result.transport_used == "T2"
        assert result.transports_attempted == ["T1", "T2"]

    @pytest.mark.asyncio
    async def test_all_transports_fail(self):
        """Test all transports failing returns error."""
        t1 = MockTransport("T1", success=False, error_message="Error 1")
        t2 = MockTransport("T2", success=False, error_message="Error 2")
        t3 = MockTransport("T3", success=False, error_message="Error 3")
        chain = TransportChain([t1, t2, t3])

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is False
        assert result.error == "Error 3"  # Last error preserved
        assert result.transports_attempted == ["T1", "T2", "T3"]

    @pytest.mark.asyncio
    async def test_transport_raises_exception(self):
        """Test handling of transport exceptions."""
        t1 = MockTransport("T1", raise_exception=True, error_message="Exception!")
        t2 = MockTransport("T2", success=True)
        chain = TransportChain([t1, t2])

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        assert result.transport_used == "T2"
        assert result.transports_attempted == ["T1", "T2"]

    @pytest.mark.asyncio
    async def test_skip_transport_with_open_circuit(self):
        """Test skipping transport with open circuit breaker."""
        t1 = MockTransport("T1", can_attempt_result=False)
        t2 = MockTransport("T2", success=True)
        chain = TransportChain([t1, t2])

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        assert result.transport_used == "T2"
        # T1 not attempted because circuit is open
        assert result.transports_attempted == ["T2"]
        assert len(t1.transfer_calls) == 0

    @pytest.mark.asyncio
    async def test_total_latency_tracked(self):
        """Test total latency is tracked correctly."""
        t1 = MockTransport("T1", success=False, latency_ms=20.0)
        t2 = MockTransport("T2", success=True, latency_ms=30.0)
        chain = TransportChain([t1, t2])

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        assert result.total_latency_ms >= 40.0  # At least sum of latencies


# =============================================================================
# TransportChain Transfer Tests - Parallel Mode
# =============================================================================


class TestParallelTransfer:
    """Tests for parallel transfer mode."""

    @pytest.mark.asyncio
    async def test_parallel_first_success_wins(self):
        """Test parallel mode uses first successful result."""
        # Fast successful transport
        t1 = MockTransport("Fast", success=True, latency_ms=10.0)
        # Slow successful transport
        t2 = MockTransport("Slow", success=True, latency_ms=100.0)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.PARALLEL)

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        # Fast transport should win
        assert result.transport_used == "Fast"

    @pytest.mark.asyncio
    async def test_parallel_all_fail(self):
        """Test parallel mode when all transports fail."""
        t1 = MockTransport("T1", success=False, latency_ms=10.0)
        t2 = MockTransport("T2", success=False, latency_ms=20.0)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.PARALLEL)

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is False
        assert "All parallel transports failed" in result.error
        assert set(result.transports_attempted) == {"T1", "T2"}

    @pytest.mark.asyncio
    async def test_parallel_no_available_transports(self):
        """Test parallel mode with no available transports."""
        t1 = MockTransport("T1", can_attempt_result=False)
        t2 = MockTransport("T2", can_attempt_result=False)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.PARALLEL)

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is False
        assert "No transports available" in result.error


# =============================================================================
# TransportChain Transfer Tests - Priority Mode
# =============================================================================


class TestPriorityTransfer:
    """Tests for priority-based transfer mode."""

    @pytest.mark.asyncio
    async def test_priority_mode_uses_highest_score(self):
        """Test priority mode tries highest score first."""
        t1 = MockTransport("Low", success=True)
        t2 = MockTransport("High", success=True)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.PRIORITY)

        # Set priority scores
        chain.set_transport_priority("High", 10.0)
        chain.set_transport_priority("Low", 1.0)

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        # High priority should be tried first
        assert result.transport_used == "High"

    @pytest.mark.asyncio
    async def test_priority_failover_to_lower(self):
        """Test failover to lower priority transport."""
        t1 = MockTransport("Low", success=True)
        t2 = MockTransport("High", success=False)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.PRIORITY)

        chain.set_transport_priority("High", 10.0)
        chain.set_transport_priority("Low", 1.0)

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        assert result.transport_used == "Low"


# =============================================================================
# TransportChain Transfer Tests - Adaptive Mode
# =============================================================================


class TestAdaptiveTransfer:
    """Tests for adaptive transfer mode."""

    @pytest.mark.asyncio
    async def test_adaptive_favors_higher_success_rate(self):
        """Test adaptive mode prefers higher success rate."""
        t1 = MockTransport("Reliable", success=True)
        t2 = MockTransport("Unreliable", success=True)
        chain = TransportChain([t1, t2], failover_mode=FailoverMode.ADAPTIVE)

        # Simulate metrics history
        chain._transport_metrics["Reliable"].total_attempts = 10
        chain._transport_metrics["Reliable"].successes = 9
        chain._transport_metrics["Unreliable"].total_attempts = 10
        chain._transport_metrics["Unreliable"].successes = 3

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        # Should try reliable first due to higher success rate
        assert result.transport_used == "Reliable"


# =============================================================================
# Circuit Breaker Integration Tests
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_shared_circuit_blocks_all_transports(self):
        """Test shared circuit breaker can block entire chain."""
        t1 = MockTransport("T1")
        chain = TransportChain([t1])

        # Open the shared circuit for target
        chain._shared_status["target-host"] = TargetStatus(
            state=TransportState.OPEN,
            failure_count=5,
            last_failure_time=time.time(),
        )

        result = await chain.transfer("/src", "/dst", "target-host")

        assert result.success is False
        assert "Circuit open" in result.error
        assert result.transports_attempted == []

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test shared circuit opens after consecutive failures."""
        t1 = MockTransport("T1", success=False)
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=300.0)
        chain = TransportChain([t1], circuit_breaker_config=config)

        # First failure
        await chain.transfer("/src", "/dst", "target")
        # Second failure - should open circuit
        await chain.transfer("/src", "/dst", "target")

        # Check circuit state
        status = chain._shared_status.get("target")
        assert status is not None
        assert status.failure_count >= 2

    @pytest.mark.asyncio
    async def test_circuit_half_open_after_recovery_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        t1 = MockTransport("T1")
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        chain = TransportChain([t1], circuit_breaker_config=config)

        # Open the circuit
        chain._shared_status["target"] = TargetStatus(
            state=TransportState.OPEN,
            failure_count=3,
            last_failure_time=time.time() - 1.0,  # 1 second ago
        )

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should be able to attempt now (half-open)
        can_attempt = chain._can_attempt_target("target")
        assert can_attempt is True

    @pytest.mark.asyncio
    async def test_success_resets_circuit(self):
        """Test successful transfer resets circuit state."""
        t1 = MockTransport("T1", success=True)
        chain = TransportChain([t1])

        # Pre-set some failure state
        chain._shared_status["target"] = TargetStatus(
            state=TransportState.HALF_OPEN,
            failure_count=3,
        )

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        status = chain._shared_status["target"]
        assert status.state == TransportState.CLOSED
        assert status.failure_count == 0


# =============================================================================
# Metrics Tracking Tests
# =============================================================================


class TestMetricsTracking:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self):
        """Test metrics are updated on successful transfer."""
        t1 = MockTransport("T1", success=True, latency_ms=25.0)
        chain = TransportChain([t1])

        await chain.transfer("/src", "/dst", "target")

        metrics = chain._transport_metrics["T1"]
        assert metrics.total_attempts == 1
        assert metrics.successes == 1
        assert metrics.failures == 0
        assert metrics.total_latency_ms >= 25.0
        assert metrics.last_attempt_time > 0
        assert metrics.last_success_time > 0

    @pytest.mark.asyncio
    async def test_metrics_updated_on_failure(self):
        """Test metrics are updated on failed transfer."""
        t1 = MockTransport("T1", success=False)
        t2 = MockTransport("T2", success=True)
        chain = TransportChain([t1, t2])

        await chain.transfer("/src", "/dst", "target")

        metrics_t1 = chain._transport_metrics["T1"]
        assert metrics_t1.total_attempts == 1
        assert metrics_t1.failures == 1
        assert metrics_t1.successes == 0

    @pytest.mark.asyncio
    async def test_timeout_tracked_as_timeout(self):
        """Test timeouts are tracked separately."""
        t1 = MockTransport("T1")
        # Set very short timeout
        chain = TransportChain(
            [t1],
            timeout_config=TimeoutConfig(operation_timeout=0.001),
        )

        # Make transfer take longer than timeout
        async def slow_transfer(*args, **kwargs):
            await asyncio.sleep(1.0)
            return TransportResult(success=True)

        t1.transfer = slow_transfer

        # The transfer should timeout
        result = await chain.transfer("/src", "/dst", "target")

        # Check timeout was tracked
        metrics = chain._transport_metrics["T1"]
        assert metrics.timeouts >= 1


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_basic_structure(self):
        """Test health check returns correct structure."""
        t1 = MockTransport("T1")
        t1.health_check = MagicMock(return_value={"healthy": True})
        chain = TransportChain([t1])

        health = chain.health_check()

        assert "healthy" in health
        assert "status" in health
        assert "message" in health
        assert "details" in health
        assert "transports" in health["details"]
        assert "failover_mode" in health["details"]

    def test_health_check_all_transports_healthy(self):
        """Test health status when all transports healthy."""
        t1 = MockTransport("T1")
        t2 = MockTransport("T2")
        t1.health_check = MagicMock(return_value={"healthy": True})
        t2.health_check = MagicMock(return_value={"healthy": True})
        chain = TransportChain([t1, t2])

        health = chain.health_check()

        assert health["healthy"] is True
        assert health["status"] == "healthy"
        assert "2/2 transports healthy" in health["message"]

    def test_health_check_some_transports_unhealthy(self):
        """Test degraded status when some transports unhealthy."""
        t1 = MockTransport("T1")
        t2 = MockTransport("T2")
        t1.health_check = MagicMock(return_value={"healthy": True})
        t2.health_check = MagicMock(return_value={"healthy": False})
        chain = TransportChain([t1, t2])

        health = chain.health_check()

        assert health["healthy"] is True  # At least one healthy
        assert health["status"] == "degraded"
        assert "1/2 transports healthy" in health["message"]

    def test_health_check_all_transports_unhealthy(self):
        """Test unhealthy status when all transports unhealthy."""
        t1 = MockTransport("T1")
        t2 = MockTransport("T2")
        t1.health_check = MagicMock(return_value={"healthy": False})
        t2.health_check = MagicMock(return_value={"healthy": False})
        chain = TransportChain([t1, t2])

        health = chain.health_check()

        assert health["healthy"] is False
        assert health["status"] == "unhealthy"

    def test_health_check_includes_success_rate(self):
        """Test health check includes overall success rate."""
        t1 = MockTransport("T1")
        t1.health_check = MagicMock(return_value={"healthy": True})
        chain = TransportChain([t1])

        # Simulate some operations
        chain._transport_metrics["T1"].total_attempts = 10
        chain._transport_metrics["T1"].successes = 8

        health = chain.health_check()

        assert "overall_success_rate" in health["details"]
        assert health["details"]["overall_success_rate"] == 0.8

    def test_health_check_includes_circuit_states(self):
        """Test health check includes shared circuit states."""
        t1 = MockTransport("T1")
        t1.health_check = MagicMock(return_value={"healthy": True})
        chain = TransportChain([t1])

        chain._shared_status["target-1"] = TargetStatus(state=TransportState.OPEN)
        chain._shared_status["target-2"] = TargetStatus(state=TransportState.CLOSED)

        health = chain.health_check()

        assert "shared_circuits" in health["details"]
        assert len(health["details"]["shared_circuits"]) == 2


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for get_statistics method."""

    def test_statistics_basic_structure(self):
        """Test statistics returns correct structure."""
        t1 = MockTransport("T1")
        chain = TransportChain([t1])

        stats = chain.get_statistics()

        assert "failover_mode" in stats
        assert "transports" in stats
        assert "shared_circuits" in stats
        assert "open_circuits" in stats

    def test_statistics_per_transport(self):
        """Test statistics includes per-transport metrics."""
        t1 = MockTransport("T1")
        chain = TransportChain([t1])

        # Simulate activity
        chain._transport_metrics["T1"].total_attempts = 5
        chain._transport_metrics["T1"].successes = 4

        stats = chain.get_statistics()

        assert "T1" in stats["transports"]
        assert stats["transports"]["T1"]["attempts"] == 5
        assert stats["transports"]["T1"]["successes"] == 4
        assert stats["transports"]["T1"]["success_rate"] == 0.8

    def test_reset_statistics(self):
        """Test reset_statistics clears all metrics."""
        t1 = MockTransport("T1")
        t1.reset_all_circuits = MagicMock()
        chain = TransportChain([t1])

        # Add some state
        chain._transport_metrics["T1"].total_attempts = 10
        chain._shared_status["target"] = TargetStatus()

        chain.reset_statistics()

        assert chain._transport_metrics["T1"].total_attempts == 0
        assert len(chain._shared_status) == 0


# =============================================================================
# Priority Management Tests
# =============================================================================


class TestPriorityManagement:
    """Tests for priority management methods."""

    def test_set_transport_priority(self):
        """Test setting transport priority."""
        chain = TransportChain([MockTransport("T1")])

        chain.set_transport_priority("T1", 5.0)

        assert chain._transport_metrics["T1"].priority_score == 5.0

    def test_set_priority_unknown_transport(self):
        """Test setting priority for unknown transport is safe."""
        chain = TransportChain([MockTransport("T1")])

        # Should not raise
        chain.set_transport_priority("NonExistent", 5.0)

    def test_boost_transport(self):
        """Test boosting transport priority."""
        chain = TransportChain([MockTransport("T1")])
        chain._transport_metrics["T1"].priority_score = 2.0

        chain.boost_transport("T1", factor=1.5)

        assert chain._transport_metrics["T1"].priority_score == 3.0

    def test_penalize_transport(self):
        """Test penalizing transport priority."""
        chain = TransportChain([MockTransport("T1")])
        chain._transport_metrics["T1"].priority_score = 2.0

        chain.penalize_transport("T1", factor=0.5)

        assert chain._transport_metrics["T1"].priority_score == 1.0


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for success/failure callbacks."""

    @pytest.mark.asyncio
    async def test_on_success_callback_called(self):
        """Test on_success callback is called on successful transfer."""
        t1 = MockTransport("T1", success=True)
        chain = TransportChain([t1])

        callback_data = []
        chain.on_success(lambda name, target, latency: callback_data.append((name, target, latency)))

        await chain.transfer("/src", "/dst", "target")

        assert len(callback_data) == 1
        assert callback_data[0][0] == "T1"
        assert callback_data[0][1] == "target"

    @pytest.mark.asyncio
    async def test_on_failure_callback_called(self):
        """Test on_failure callback is called on exception."""
        t1 = MockTransport("T1", raise_exception=True, error_message="Failed!")
        chain = TransportChain([t1])

        callback_data = []
        chain.on_failure(lambda name, target, exc: callback_data.append((name, target, str(exc))))

        await chain.transfer("/src", "/dst", "target")

        assert len(callback_data) == 1
        assert callback_data[0][0] == "T1"
        assert callback_data[0][1] == "target"

    @pytest.mark.asyncio
    async def test_callback_exception_ignored(self):
        """Test callback exceptions don't break transfer."""
        t1 = MockTransport("T1", success=True)
        chain = TransportChain([t1])

        def bad_callback(*args):
            raise RuntimeError("Callback failed!")

        chain.on_success(bad_callback)

        # Should not raise despite bad callback
        result = await chain.transfer("/src", "/dst", "target")
        assert result.success is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_sequential_chain(self):
        """Test create_sequential_chain factory."""
        transports = [MockTransport("T1"), MockTransport("T2")]
        chain = create_sequential_chain(transports)

        assert chain._failover_mode == FailoverMode.SEQUENTIAL
        assert len(chain._transports) == 2

    def test_create_parallel_chain(self):
        """Test create_parallel_chain factory."""
        transports = [MockTransport("T1"), MockTransport("T2")]
        chain = create_parallel_chain(transports)

        assert chain._failover_mode == FailoverMode.PARALLEL

    def test_create_adaptive_chain(self):
        """Test create_adaptive_chain factory."""
        transports = [MockTransport("T1"), MockTransport("T2")]
        chain = create_adaptive_chain(transports)

        assert chain._failover_mode == FailoverMode.ADAPTIVE


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complex scenarios."""

    @pytest.mark.asyncio
    async def test_full_failover_chain(self):
        """Test complete failover scenario with multiple failures."""
        t1 = MockTransport("Primary", success=False)
        t2 = MockTransport("Secondary", raise_exception=True, error_message="Network error")
        t3 = MockTransport("Tertiary", success=True)
        chain = TransportChain([t1, t2, t3])

        result = await chain.transfer("/src", "/dst", "target")

        assert result.success is True
        assert result.transport_used == "Tertiary"
        assert result.transports_attempted == ["Primary", "Secondary", "Tertiary"]

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_transfers(self):
        """Test metrics accumulate across multiple transfers."""
        t1 = MockTransport("T1", success=True)
        chain = TransportChain([t1])

        # Multiple transfers
        for _ in range(5):
            await chain.transfer("/src", "/dst", "target")

        metrics = chain._transport_metrics["T1"]
        assert metrics.total_attempts == 5
        assert metrics.successes == 5

    @pytest.mark.asyncio
    async def test_different_targets_tracked_separately(self):
        """Test circuit state tracked separately per target."""
        t1 = MockTransport("T1", success=True)
        chain = TransportChain([t1])

        await chain.transfer("/src", "/dst", "target-1")
        await chain.transfer("/src", "/dst", "target-2")

        # Both targets should have state
        assert "target-1" in chain._shared_status
        assert "target-2" in chain._shared_status
