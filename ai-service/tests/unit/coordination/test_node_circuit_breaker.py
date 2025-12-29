"""Tests for node_circuit_breaker.py.

December 29, 2025: Comprehensive test coverage for per-node circuit breaker.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.coordination.node_circuit_breaker import (
    NodeCircuitBreaker,
    NodeCircuitBreakerRegistry,
    NodeCircuitConfig,
    NodeCircuitState,
    NodeCircuitStatus,
    get_node_circuit_breaker,
    get_node_circuit_registry,
)


class TestNodeCircuitState:
    """Tests for NodeCircuitState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert NodeCircuitState.CLOSED.value == "closed"
        assert NodeCircuitState.OPEN.value == "open"
        assert NodeCircuitState.HALF_OPEN.value == "half_open"

    def test_state_count(self):
        """Test there are exactly 3 states."""
        assert len(NodeCircuitState) == 3


class TestNodeCircuitConfig:
    """Tests for NodeCircuitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NodeCircuitConfig()
        assert config.failure_threshold == 5  # default from env or 5
        assert config.recovery_timeout == 60.0  # default from env or 60
        assert config.success_threshold == 1
        assert config.emit_events is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = NodeCircuitConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=3,
            emit_events=False,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.success_threshold == 3
        assert config.emit_events is False

    def test_validation_failure_threshold(self):
        """Test validation rejects invalid failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be > 0"):
            NodeCircuitConfig(failure_threshold=0)

    def test_validation_recovery_timeout(self):
        """Test validation rejects invalid recovery_timeout."""
        with pytest.raises(ValueError, match="recovery_timeout must be > 0"):
            NodeCircuitConfig(recovery_timeout=0)

    def test_env_var_loading(self):
        """Test configuration loads from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_P2P_NODE_CIRCUIT_FAILURE_THRESHOLD": "10",
            "RINGRIFT_P2P_NODE_CIRCUIT_RECOVERY_TIMEOUT": "120.0",
        }):
            config = NodeCircuitConfig()
            assert config.failure_threshold == 10
            assert config.recovery_timeout == 120.0


class TestNodeCircuitStatus:
    """Tests for NodeCircuitStatus dataclass."""

    def test_creation(self):
        """Test status creation."""
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
        assert status.node_id == "test-node"
        assert status.state == NodeCircuitState.CLOSED
        assert status.success_count == 5

    def test_time_until_recovery_closed(self):
        """Test time_until_recovery for closed circuit."""
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            opened_at=None,
            recovery_timeout=60.0,
        )
        assert status.time_until_recovery == 0.0

    def test_time_until_recovery_open(self):
        """Test time_until_recovery for open circuit."""
        opened_time = time.time() - 30  # Opened 30 seconds ago
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.OPEN,
            failure_count=5,
            success_count=0,
            last_failure_time=opened_time,
            last_success_time=None,
            opened_at=opened_time,
            recovery_timeout=60.0,
        )
        # Should be approximately 30 seconds remaining
        assert 25 < status.time_until_recovery < 35

    def test_to_dict(self):
        """Test to_dict serialization."""
        status = NodeCircuitStatus(
            node_id="test-node",
            state=NodeCircuitState.OPEN,
            failure_count=5,
            success_count=0,
            last_failure_time=1000.0,
            last_success_time=None,
            opened_at=1000.0,
            recovery_timeout=60.0,
        )
        d = status.to_dict()
        assert d["node_id"] == "test-node"
        assert d["state"] == "open"
        assert d["failure_count"] == 5
        assert "time_until_recovery" in d


class TestNodeCircuitBreaker:
    """Tests for NodeCircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test new nodes start with closed circuit."""
        breaker = NodeCircuitBreaker()
        state = breaker.get_state("test-node")
        assert state == NodeCircuitState.CLOSED

    def test_can_check_closed_circuit(self):
        """Test can_check returns True for closed circuit."""
        breaker = NodeCircuitBreaker()
        assert breaker.can_check("test-node") is True

    def test_record_success_resets_failure_count(self):
        """Test success resets failure count."""
        breaker = NodeCircuitBreaker()
        breaker.record_failure("test-node")
        breaker.record_failure("test-node")
        breaker.record_success("test-node")

        status = breaker.get_status("test-node")
        assert status.failure_count == 0
        assert status.success_count == 1

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = NodeCircuitConfig(failure_threshold=3)
        breaker = NodeCircuitBreaker(config=config)

        # Record failures up to threshold
        breaker.record_failure("test-node")
        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

    def test_open_circuit_blocks_checks(self):
        """Test open circuit blocks health checks."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        assert breaker.can_check("test-node") is False

    def test_half_open_after_recovery_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.get_state("test-node") == NodeCircuitState.HALF_OPEN

    def test_half_open_allows_check(self):
        """Test half-open circuit allows checks."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        time.sleep(0.15)

        assert breaker.can_check("test-node") is True

    def test_half_open_closes_on_success(self):
        """Test half-open circuit closes on success."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        time.sleep(0.15)

        # Should be half-open
        assert breaker.get_state("test-node") == NodeCircuitState.HALF_OPEN

        breaker.record_success("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Test half-open circuit reopens on failure."""
        config = NodeCircuitConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        time.sleep(0.15)

        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

    def test_get_all_states(self):
        """Test get_all_states returns all tracked nodes."""
        breaker = NodeCircuitBreaker()

        breaker.record_success("node-1")
        breaker.record_success("node-2")
        breaker.record_failure("node-3")

        states = breaker.get_all_states()
        assert len(states) == 3
        assert "node-1" in states
        assert "node-2" in states
        assert "node-3" in states

    def test_get_open_circuits(self):
        """Test get_open_circuits returns only open nodes."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_success("healthy-node")
        breaker.record_failure("unhealthy-node")

        open_circuits = breaker.get_open_circuits()
        assert "unhealthy-node" in open_circuits
        assert "healthy-node" not in open_circuits

    def test_get_summary(self):
        """Test get_summary returns correct counts."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_success("healthy-1")
        breaker.record_success("healthy-2")
        breaker.record_failure("unhealthy-1")

        summary = breaker.get_summary()
        assert summary["total_nodes"] == 3
        assert summary["closed"] == 2
        assert summary["open"] == 1
        assert summary["half_open"] == 0
        assert "unhealthy-1" in summary["open_nodes"]

    def test_reset_node(self):
        """Test resetting a specific node."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

        breaker.reset("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

    def test_reset_all(self):
        """Test resetting all nodes."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("node-1")
        breaker.record_failure("node-2")

        breaker.reset_all()

        assert breaker.get_state("node-1") == NodeCircuitState.CLOSED
        assert breaker.get_state("node-2") == NodeCircuitState.CLOSED

    def test_force_open(self):
        """Test forcing circuit open."""
        breaker = NodeCircuitBreaker()

        breaker.force_open("test-node")
        assert breaker.get_state("test-node") == NodeCircuitState.OPEN

    def test_force_close(self):
        """Test forcing circuit closed."""
        config = NodeCircuitConfig(failure_threshold=1)
        breaker = NodeCircuitBreaker(config=config)

        breaker.record_failure("test-node")
        breaker.force_close("test-node")

        assert breaker.get_state("test-node") == NodeCircuitState.CLOSED

    def test_state_change_callback(self):
        """Test state change callback is called."""
        config = NodeCircuitConfig(failure_threshold=1)
        callback = MagicMock()
        breaker = NodeCircuitBreaker(config=config, on_state_change=callback)

        breaker.record_failure("test-node")

        callback.assert_called_once_with(
            "test-node",
            NodeCircuitState.CLOSED,
            NodeCircuitState.OPEN,
        )

    def test_state_change_callback_error_handling(self):
        """Test callback errors are handled gracefully."""
        config = NodeCircuitConfig(failure_threshold=1)
        callback = MagicMock(side_effect=Exception("Callback error"))
        breaker = NodeCircuitBreaker(config=config, on_state_change=callback)

        # Should not raise despite callback error
        breaker.record_failure("test-node")


class TestNodeCircuitBreakerRegistry:
    """Tests for NodeCircuitBreakerRegistry class."""

    def setup_method(self):
        """Reset singleton before each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NodeCircuitBreakerRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Test singleton instance creation."""
        registry1 = NodeCircuitBreakerRegistry.get_instance()
        registry2 = NodeCircuitBreakerRegistry.get_instance()
        assert registry1 is registry2

    def test_reset_instance(self):
        """Test singleton reset."""
        registry1 = NodeCircuitBreakerRegistry.get_instance()
        NodeCircuitBreakerRegistry.reset_instance()
        registry2 = NodeCircuitBreakerRegistry.get_instance()
        assert registry1 is not registry2

    def test_get_breaker_creates_new(self):
        """Test get_breaker creates new breaker for operation type."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        breaker = registry.get_breaker("health_check")
        assert isinstance(breaker, NodeCircuitBreaker)

    def test_get_breaker_returns_same(self):
        """Test get_breaker returns same instance for same type."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        breaker1 = registry.get_breaker("health_check")
        breaker2 = registry.get_breaker("health_check")
        assert breaker1 is breaker2

    def test_get_breaker_different_types(self):
        """Test different operation types get different breakers."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        breaker1 = registry.get_breaker("health_check")
        breaker2 = registry.get_breaker("gossip")
        assert breaker1 is not breaker2

    def test_get_all_summaries(self):
        """Test get_all_summaries returns all operation types."""
        registry = NodeCircuitBreakerRegistry.get_instance()
        registry.get_breaker("health_check")
        registry.get_breaker("gossip")

        summaries = registry.get_all_summaries()
        assert "health_check" in summaries
        assert "gossip" in summaries


class TestModuleLevelFunctions:
    """Tests for module-level accessor functions."""

    def setup_method(self):
        """Reset global state before each test."""
        NodeCircuitBreakerRegistry.reset_instance()
        # Reset module-level _registry
        import app.coordination.node_circuit_breaker as module
        module._registry = None

    def teardown_method(self):
        """Reset global state after each test."""
        NodeCircuitBreakerRegistry.reset_instance()
        import app.coordination.node_circuit_breaker as module
        module._registry = None

    def test_get_node_circuit_breaker_default(self):
        """Test get_node_circuit_breaker with default type."""
        breaker = get_node_circuit_breaker()
        assert isinstance(breaker, NodeCircuitBreaker)

    def test_get_node_circuit_breaker_custom_type(self):
        """Test get_node_circuit_breaker with custom type."""
        breaker = get_node_circuit_breaker("gossip")
        assert isinstance(breaker, NodeCircuitBreaker)

    def test_get_node_circuit_breaker_same_instance(self):
        """Test same breaker returned for same type."""
        breaker1 = get_node_circuit_breaker("health_check")
        breaker2 = get_node_circuit_breaker("health_check")
        assert breaker1 is breaker2

    def test_get_node_circuit_registry(self):
        """Test get_node_circuit_registry returns registry."""
        registry = get_node_circuit_registry()
        assert isinstance(registry, NodeCircuitBreakerRegistry)
