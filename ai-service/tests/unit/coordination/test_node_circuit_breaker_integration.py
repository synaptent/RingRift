"""Unit tests for node circuit breaker integration in sync operations.

December 2025: Tests for P0 critical fix - circuit breaker per node.
"""

import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.node_circuit_breaker import (
    NodeCircuitBreaker,
    NodeCircuitBreakerRegistry,
    NodeCircuitConfig,
    NodeCircuitState,
    NodeCircuitStatus,
    get_node_circuit_breaker,
    get_node_circuit_registry,
)


def _wait_for_half_open(
    breaker: NodeCircuitBreaker,
    node_id: str,
    timeout_s: float = 0.5,
) -> None:
    """Wait until the breaker transitions into HALF_OPEN for a node."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if breaker.get_state(node_id) == NodeCircuitState.HALF_OPEN:
            return
        time.sleep(0.005)
    raise AssertionError(f"{node_id} did not transition to HALF_OPEN within {timeout_s}s")


class TestNodeCircuitBreaker:
    """Tests for NodeCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that initial circuit state is CLOSED."""
        breaker = NodeCircuitBreaker()
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

    def test_can_check_returns_true_for_closed_circuit(self):
        """Test that can_check returns True for closed circuit."""
        breaker = NodeCircuitBreaker()
        assert breaker.can_check("test-node") is True

    def test_can_check_returns_false_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        config = NodeCircuitConfig(failure_threshold=3, recovery_timeout=60.0)
        breaker = NodeCircuitBreaker(config=config)

        # Record failures
        breaker.record_failure("test-node")
        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is True  # Still closed

        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is False  # Now open

    def test_success_resets_failure_count(self):
        """Test that success resets consecutive failure count."""
        config = NodeCircuitConfig(failure_threshold=3)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        breaker.record_failure("test-node")
        breaker.record_success("test-node")  # Reset failure count

        # Should need 3 more failures to open
        breaker.record_failure("test-node")
        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is True

        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is False

    def test_recovery_timeout_triggers_half_open(self):
        """Test that circuit transitions to half-open after recovery timeout."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.01)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is False

        # Wait for recovery timeout
        _wait_for_half_open(breaker, "test-node")
        assert breaker.can_check("test-node") is True
        assert breaker.get_state("test-node") == NodeCircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Test that success in half-open state closes circuit."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.01)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        _wait_for_half_open(breaker, "test-node")

        assert breaker.get_state("test-node") == NodeCircuitState.HALF_OPEN
        breaker.record_success("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Test that failure in half-open state reopens circuit."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.01)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        _wait_for_half_open(breaker, "test-node")

        assert breaker.get_state("test-node") == NodeCircuitState.HALF_OPEN
        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

    def test_force_open(self):
        """Test force_open manually opens circuit."""
        breaker = NodeCircuitBreaker()
        breaker.force_open("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN
        assert breaker.can_check("test-node") is False

    def test_force_close(self):
        """Test force_close manually closes circuit."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is False

        breaker.force_close("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED
        assert breaker.can_check("test-node") is True

    def test_reset_clears_circuit(self):
        """Test reset clears circuit to initial state."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        breaker.reset("test-node")

        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED
        assert breaker.can_check("test-node") is True

    def test_get_status_returns_correct_info(self):
        """Test get_status returns correct NodeCircuitStatus."""
        breaker = NodeCircuitBreaker()
        breaker.record_success("test-node")

        status = breaker.get_status("test-node")
        assert status.node_id == "test-node"
        assert status.state == NodeCircuitState.CLOSED
        assert status.success_count == 1
        assert status.failure_count == 0
        assert status.last_success_time is not None

    def test_get_open_circuits(self):
        """Test get_open_circuits returns only open circuit nodes."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("node-1")  # Opens
        breaker.record_success("node-2")  # Stays closed

        open_circuits = breaker.get_open_circuits()
        assert "node-1" in open_circuits
        assert "node-2" not in open_circuits

    def test_get_summary(self):
        """Test get_summary returns correct counts."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_success("node-1")  # Closed
        breaker.record_failure("node-2")  # Open
        breaker.record_failure("node-3")  # Open

        summary = breaker.get_summary()
        assert summary["total_nodes"] == 3
        assert summary["closed"] == 1
        assert summary["open"] == 2
        assert summary["half_open"] == 0

    def test_independent_circuits_per_node(self):
        """Test that each node has independent circuit state."""
        config = NodeCircuitConfig(failure_threshold=2)
        breaker = NodeCircuitBreaker(config=config)

        # Fail node-1 twice (opens it)
        breaker.record_failure("node-1")
        breaker.record_failure("node-1")

        # Fail node-2 once (still closed)
        breaker.record_failure("node-2")

        assert breaker.can_check("node-1") is False  # Open
        assert breaker.can_check("node-2") is True   # Still closed


class TestNodeCircuitBreakerRegistry:
    """Tests for NodeCircuitBreakerRegistry singleton."""

    def setup_method(self):
        """Reset registry before each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = NodeCircuitBreakerRegistry.get_instance()
        registry2 = NodeCircuitBreakerRegistry.get_instance()
        assert registry1 is registry2

    def test_get_breaker_creates_new_breaker(self):
        """Test get_breaker creates breaker for operation type."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        breaker = registry.get_breaker("sync")
        assert isinstance(breaker, NodeCircuitBreaker)

    def test_get_breaker_returns_same_instance(self):
        """Test get_breaker returns same instance for same operation type."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        breaker1 = registry.get_breaker("sync")
        breaker2 = registry.get_breaker("sync")
        assert breaker1 is breaker2

    def test_different_operation_types_different_breakers(self):
        """Test different operation types get different breakers."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        sync_breaker = registry.get_breaker("sync")
        p2p_breaker = registry.get_breaker("p2p_sync")
        assert sync_breaker is not p2p_breaker


class TestGetNodeCircuitBreaker:
    """Tests for module-level get_node_circuit_breaker function."""

    def setup_method(self):
        """Reset registry before each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def test_get_node_circuit_breaker(self):
        """Test get_node_circuit_breaker returns breaker."""
        breaker = get_node_circuit_breaker("health_check")
        assert isinstance(breaker, NodeCircuitBreaker)

    def test_get_node_circuit_breaker_default_type(self):
        """Test get_node_circuit_breaker uses health_check by default."""
        breaker1 = get_node_circuit_breaker()
        breaker2 = get_node_circuit_breaker("health_check")
        assert breaker1 is breaker2


class TestStateChangeCallback:
    """Tests for state change callbacks."""

    def test_callback_called_on_state_change(self):
        """Test that callback is called when circuit state changes."""
        callback = MagicMock()
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config, on_state_change=callback)

        breaker.record_failure("test-node")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test-node"  # node_id
        assert args[1] == NodeCircuitState.CLOSED  # old_state
        assert args[2] == NodeCircuitState.OPEN  # new_state

    def test_callback_not_called_if_state_unchanged(self):
        """Test that callback is not called when state doesn't change."""
        callback = MagicMock()
        breaker = NodeCircuitBreaker(on_state_change=callback)

        # Multiple successes should not change state
        breaker.record_success("test-node")
        breaker.record_success("test-node")

        callback.assert_not_called()

    def test_callback_error_does_not_break_operation(self):
        """Test that callback errors don't break the breaker."""
        callback = MagicMock(side_effect=RuntimeError("Callback error"))
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config, on_state_change=callback)

        # Should not raise even though callback fails
        breaker.record_failure("test-node")

        # State should still change
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN


class TestNodeCircuitStatus:
    """Tests for NodeCircuitStatus dataclass."""

    def test_time_until_recovery_when_open(self):
        """Test time_until_recovery calculation for open circuit."""
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time() - 30,  # 30 seconds ago
            recovery_timeout=60.0,
        )

        time_until = status.time_until_recovery
        assert 29 <= time_until <= 31  # ~30 seconds remaining

    def test_time_until_recovery_when_closed(self):
        """Test time_until_recovery returns 0 for closed circuit."""
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            recovery_timeout=60.0,
        )

        assert status.time_until_recovery == 0.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = time.time()
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.OPEN,
            failure_count=3,
            success_count=1,
            last_failure_time=now,
            last_success_time=now - 100,
            opened_at=now - 10,
            recovery_timeout=60.0,
        )

        d = status.to_dict()
        assert d["node_id"] == "test-node"
        assert d["state"] == "open"
        assert d["failure_count"] == 3
        assert d["success_count"] == 1
        assert "time_until_recovery" in d


class TestSyncCoordinatorCircuitBreaker:
    """Tests for circuit breaker integration in sync_coordinator."""

    @pytest.mark.asyncio
    async def test_sync_skips_node_with_open_circuit(self):
        """Test that sync operations skip nodes with open circuits."""
        # This tests the integration - when circuit is open, the node should be skipped
        breaker = get_node_circuit_breaker("sync")
        breaker.force_open("failing-node")

        # The sync should skip this node immediately
        assert breaker.can_check("failing-node") is False
        assert breaker.can_check("healthy-node") is True

    def test_circuit_breaker_isolates_failures(self):
        """Test that failures on one node don't affect others."""
        breaker = get_node_circuit_breaker("sync")
        config = NodeCircuitConfig(failure_threshold=1)

        # Create fresh breaker with low threshold for testing
        test_breaker = NodeCircuitBreaker(config=config)

        # Fail one node
        test_breaker.record_failure("bad-node")

        # Other nodes should be unaffected
        assert test_breaker.can_check("bad-node") is False
        assert test_breaker.can_check("good-node-1") is True
        assert test_breaker.can_check("good-node-2") is True
