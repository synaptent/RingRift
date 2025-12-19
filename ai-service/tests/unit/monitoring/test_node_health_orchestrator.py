"""Tests for NodeHealthOrchestrator (node health monitoring).

Tests cover:
- NodeHealthState enum
- NodeHealth and ClusterHealthStats dataclasses
- NodeHealthOrchestrator event handling and state management
- Module functions (wire_health_events, get_health_orchestrator, reset_health_orchestrator)
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# =============================================================================
# Test NodeHealthState Enum
# =============================================================================

class TestNodeHealthState:
    """Tests for NodeHealthState enum."""

    def test_healthy_value(self):
        """Test HEALTHY state value."""
        from app.monitoring.node_health_orchestrator import NodeHealthState
        assert NodeHealthState.HEALTHY.value == "healthy"

    def test_unhealthy_value(self):
        """Test UNHEALTHY state value."""
        from app.monitoring.node_health_orchestrator import NodeHealthState
        assert NodeHealthState.UNHEALTHY.value == "unhealthy"

    def test_recovering_value(self):
        """Test RECOVERING state value."""
        from app.monitoring.node_health_orchestrator import NodeHealthState
        assert NodeHealthState.RECOVERING.value == "recovering"

    def test_unknown_value(self):
        """Test UNKNOWN state value."""
        from app.monitoring.node_health_orchestrator import NodeHealthState
        assert NodeHealthState.UNKNOWN.value == "unknown"


# =============================================================================
# Test NodeHealth Dataclass
# =============================================================================

class TestNodeHealth:
    """Tests for NodeHealth dataclass."""

    def test_create_with_required_fields(self):
        """Test creating health with required fields."""
        from app.monitoring.node_health_orchestrator import NodeHealth

        health = NodeHealth(node_name="gh200-a")
        assert health.node_name == "gh200-a"

    def test_default_values(self):
        """Test default values are set correctly."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        health = NodeHealth(node_name="test")

        assert health.node_ip == ""
        assert health.state == NodeHealthState.UNKNOWN
        assert health.last_health_check_time == 0.0
        assert health.last_state_change_time == 0.0
        assert health.consecutive_failures == 0
        assert health.consecutive_successes == 0
        assert health.last_error == ""
        assert health.recovery_attempts == 0
        assert health.last_recovery_time == 0.0
        assert health.gpu_utilization == 0.0
        assert health.memory_used_percent == 0.0
        assert health.disk_used_percent == 0.0
        assert health.active_constraints == []

    def test_all_fields(self):
        """Test creating health with all fields."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        health = NodeHealth(
            node_name="gh200-b",
            node_ip="10.0.0.2",
            state=NodeHealthState.HEALTHY,
            last_health_check_time=1000.0,
            last_state_change_time=900.0,
            consecutive_failures=0,
            consecutive_successes=5,
            last_error="",
            recovery_attempts=1,
            last_recovery_time=800.0,
            gpu_utilization=75.0,
            memory_used_percent=60.0,
            disk_used_percent=45.0,
            active_constraints=[],
        )

        assert health.node_name == "gh200-b"
        assert health.node_ip == "10.0.0.2"
        assert health.state == NodeHealthState.HEALTHY
        assert health.consecutive_successes == 5


# =============================================================================
# Test ClusterHealthStats Dataclass
# =============================================================================

class TestClusterHealthStats:
    """Tests for ClusterHealthStats dataclass."""

    def test_default_values(self):
        """Test default values for cluster stats."""
        from app.monitoring.node_health_orchestrator import ClusterHealthStats

        stats = ClusterHealthStats()

        assert stats.total_nodes == 0
        assert stats.healthy_nodes == 0
        assert stats.unhealthy_nodes == 0
        assert stats.recovering_nodes == 0
        assert stats.unknown_nodes == 0
        assert stats.total_health_checks == 0
        assert stats.total_failures == 0
        assert stats.total_recoveries == 0
        assert stats.total_recovery_failures == 0
        assert stats.last_activity_time == 0.0

    def test_all_fields(self):
        """Test creating stats with all fields."""
        from app.monitoring.node_health_orchestrator import ClusterHealthStats

        stats = ClusterHealthStats(
            total_nodes=10,
            healthy_nodes=8,
            unhealthy_nodes=1,
            recovering_nodes=1,
            unknown_nodes=0,
            total_health_checks=1000,
            total_failures=50,
            total_recoveries=5,
            total_recovery_failures=1,
            last_activity_time=time.time(),
        )

        assert stats.total_nodes == 10
        assert stats.healthy_nodes == 8
        assert stats.unhealthy_nodes == 1
        assert stats.total_recoveries == 5


# =============================================================================
# Test NodeHealthOrchestrator
# =============================================================================

class TestNodeHealthOrchestrator:
    """Tests for NodeHealthOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealthOrchestrator,
            reset_health_orchestrator,
        )

        reset_health_orchestrator()
        orch = NodeHealthOrchestrator()
        yield orch
        reset_health_orchestrator()

    def test_initialization_defaults(self, orchestrator):
        """Test default initialization."""
        assert orchestrator.node_stale_threshold_seconds == 300.0
        assert orchestrator.max_history_per_node == 100
        assert orchestrator.recovery_cooldown_seconds == 60.0
        assert not orchestrator._subscribed

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        from app.monitoring.node_health_orchestrator import NodeHealthOrchestrator

        orch = NodeHealthOrchestrator(
            node_stale_threshold_seconds=600.0,
            max_history_per_node=50,
            recovery_cooldown_seconds=120.0,
        )

        assert orch.node_stale_threshold_seconds == 600.0
        assert orch.max_history_per_node == 50
        assert orch.recovery_cooldown_seconds == 120.0

    def test_get_or_create_node_new(self, orchestrator):
        """Test getting/creating a new node."""
        node = orchestrator._get_or_create_node("gh200-a", "10.0.0.1")
        assert node.node_name == "gh200-a"
        assert node.node_ip == "10.0.0.1"
        assert "gh200-a" in orchestrator._nodes

    def test_get_or_create_node_existing(self, orchestrator):
        """Test getting an existing node."""
        node1 = orchestrator._get_or_create_node("gh200-a", "10.0.0.1")
        node1.consecutive_successes = 5
        node2 = orchestrator._get_or_create_node("gh200-a")
        assert node2.consecutive_successes == 5
        assert node1 is node2

    def test_on_node_unhealthy(self, orchestrator):
        """Test handling node unhealthy event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "node_name": "gh200-a",
            "node_ip": "10.0.0.1",
            "error": "GPU memory exhausted",
            "gpu_utilization": 100.0,
        })

        orchestrator._on_node_unhealthy(event)

        node = orchestrator._nodes["gh200-a"]
        assert node.state == NodeHealthState.UNHEALTHY
        assert node.last_error == "GPU memory exhausted"
        assert node.consecutive_failures == 1
        assert node.gpu_utilization == 100.0
        assert orchestrator._total_failures == 1

    def test_on_node_recovered(self, orchestrator):
        """Test handling node recovered event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        # First make node unhealthy
        orchestrator._nodes["gh200-a"] = orchestrator._get_or_create_node("gh200-a")
        orchestrator._nodes["gh200-a"].state = NodeHealthState.UNHEALTHY
        orchestrator._nodes["gh200-a"].consecutive_failures = 5
        orchestrator._nodes["gh200-a"].active_constraints = ["GPU_HIGH"]

        event = MockEvent(payload={
            "node_name": "gh200-a",
        })

        orchestrator._on_node_recovered(event)

        node = orchestrator._nodes["gh200-a"]
        assert node.state == NodeHealthState.HEALTHY
        assert node.consecutive_failures == 0
        assert node.consecutive_successes == 1
        assert node.last_error == ""
        assert node.active_constraints == []

    def test_on_health_check_passed_from_unknown(self, orchestrator):
        """Test health check passed transitions from unknown to healthy."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={"node_name": "new-node"})

        orchestrator._on_health_check_passed(event)

        node = orchestrator._nodes["new-node"]
        assert node.state == NodeHealthState.HEALTHY
        assert orchestrator._total_health_checks == 1

    def test_on_health_check_passed_from_unhealthy_needs_multiple(self, orchestrator):
        """Test that unhealthy requires 3 consecutive successes to become healthy."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        # Setup unhealthy node
        node = orchestrator._get_or_create_node("test-node")
        node.state = NodeHealthState.UNHEALTHY

        event = MockEvent(payload={"node_name": "test-node"})

        # First two checks don't change state
        orchestrator._on_health_check_passed(event)
        assert node.state == NodeHealthState.UNHEALTHY
        assert node.consecutive_successes == 1

        orchestrator._on_health_check_passed(event)
        assert node.state == NodeHealthState.UNHEALTHY
        assert node.consecutive_successes == 2

        # Third check transitions to healthy
        orchestrator._on_health_check_passed(event)
        assert node.state == NodeHealthState.HEALTHY
        assert node.consecutive_successes == 3

    def test_on_health_check_failed(self, orchestrator):
        """Test handling health check failed event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "node_name": "failing-node",
            "error": "Connection refused",
        })

        # First failure
        orchestrator._on_health_check_failed(event)
        node = orchestrator._nodes["failing-node"]
        assert node.consecutive_failures == 1
        assert node.state != NodeHealthState.UNHEALTHY  # Not yet

        # Second failure marks unhealthy
        orchestrator._on_health_check_failed(event)
        assert node.consecutive_failures == 2
        assert node.state == NodeHealthState.UNHEALTHY
        assert orchestrator._total_failures == 2

    def test_on_resource_constraint(self, orchestrator):
        """Test handling resource constraint event."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "node_name": "constrained-node",
            "constraint": "GPU_MEMORY_HIGH",
            "memory_percent": 95.0,
            "gpu_utilization": 98.0,
        })

        orchestrator._on_resource_constraint(event)

        node = orchestrator._nodes["constrained-node"]
        assert "GPU_MEMORY_HIGH" in node.active_constraints
        assert node.memory_used_percent == 95.0
        assert node.gpu_utilization == 98.0

    def test_on_resource_constraint_multiple(self, orchestrator):
        """Test handling multiple constraints on same node."""
        @dataclass
        class MockEvent:
            payload: dict

        orchestrator._on_resource_constraint(MockEvent(payload={
            "node_name": "test",
            "constraint": "GPU_HIGH",
        }))
        orchestrator._on_resource_constraint(MockEvent(payload={
            "node_name": "test",
            "constraint": "MEMORY_HIGH",
        }))

        node = orchestrator._nodes["test"]
        assert len(node.active_constraints) == 2
        assert "GPU_HIGH" in node.active_constraints
        assert "MEMORY_HIGH" in node.active_constraints

    def test_on_recovery_initiated(self, orchestrator):
        """Test handling recovery initiated event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={"node_name": "recovering-node"})

        orchestrator._on_recovery_initiated(event)

        node = orchestrator._nodes["recovering-node"]
        assert node.state == NodeHealthState.RECOVERING
        assert node.recovery_attempts == 1
        assert node.last_recovery_time > 0

    def test_on_recovery_completed(self, orchestrator):
        """Test handling recovery completed event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        # Setup recovering node
        node = orchestrator._get_or_create_node("test-node")
        node.state = NodeHealthState.RECOVERING
        node.consecutive_failures = 5

        event = MockEvent(payload={"node_name": "test-node"})

        orchestrator._on_recovery_completed(event)

        assert node.state == NodeHealthState.HEALTHY
        assert node.consecutive_failures == 0
        assert node.consecutive_successes == 1
        assert orchestrator._total_recoveries == 1

    def test_on_recovery_failed(self, orchestrator):
        """Test handling recovery failed event."""
        from app.monitoring.node_health_orchestrator import NodeHealthState

        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "node_name": "failed-node",
            "error": "GPU driver crash",
        })

        orchestrator._on_recovery_failed(event)

        node = orchestrator._nodes["failed-node"]
        assert node.state == NodeHealthState.UNHEALTHY
        assert node.last_error == "GPU driver crash"
        assert orchestrator._total_recovery_failures == 1

    def test_add_to_history(self, orchestrator):
        """Test adding entries to node history."""
        orchestrator._add_to_history("node1", "test_event", {"value": 123})
        orchestrator._add_to_history("node1", "test_event", {"value": 456})

        assert len(orchestrator._node_history["node1"]) == 2
        assert orchestrator._node_history["node1"][0]["value"] == 123

    def test_history_trimming(self):
        """Test that history is trimmed to max_history_per_node."""
        from app.monitoring.node_health_orchestrator import NodeHealthOrchestrator

        orch = NodeHealthOrchestrator(max_history_per_node=5)

        for i in range(10):
            orch._add_to_history("test", "event", {"value": i})

        assert len(orch._node_history["test"]) == 5
        values = [e["value"] for e in orch._node_history["test"]]
        assert values == [5, 6, 7, 8, 9]

    def test_get_node_health(self, orchestrator):
        """Test getting health for a specific node."""
        from app.monitoring.node_health_orchestrator import NodeHealth

        orchestrator._nodes["gh200-a"] = NodeHealth(
            node_name="gh200-a",
            consecutive_successes=10,
        )

        health = orchestrator.get_node_health("gh200-a")
        assert health is not None
        assert health.consecutive_successes == 10

        assert orchestrator.get_node_health("nonexistent") is None

    def test_get_healthy_nodes(self, orchestrator):
        """Test getting healthy nodes."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["healthy1"] = NodeHealth(
            node_name="healthy1",
            state=NodeHealthState.HEALTHY,
        )
        orchestrator._nodes["healthy2"] = NodeHealth(
            node_name="healthy2",
            state=NodeHealthState.HEALTHY,
        )
        orchestrator._nodes["unhealthy"] = NodeHealth(
            node_name="unhealthy",
            state=NodeHealthState.UNHEALTHY,
        )

        healthy = orchestrator.get_healthy_nodes()
        assert len(healthy) == 2
        assert all(n.state == NodeHealthState.HEALTHY for n in healthy)

    def test_get_unhealthy_nodes(self, orchestrator):
        """Test getting unhealthy nodes."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["healthy"] = NodeHealth(
            node_name="healthy",
            state=NodeHealthState.HEALTHY,
        )
        orchestrator._nodes["unhealthy"] = NodeHealth(
            node_name="unhealthy",
            state=NodeHealthState.UNHEALTHY,
        )

        unhealthy = orchestrator.get_unhealthy_nodes()
        assert len(unhealthy) == 1
        assert unhealthy[0].node_name == "unhealthy"

    def test_get_recovering_nodes(self, orchestrator):
        """Test getting recovering nodes."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["recovering"] = NodeHealth(
            node_name="recovering",
            state=NodeHealthState.RECOVERING,
        )
        orchestrator._nodes["healthy"] = NodeHealth(
            node_name="healthy",
            state=NodeHealthState.HEALTHY,
        )

        recovering = orchestrator.get_recovering_nodes()
        assert len(recovering) == 1
        assert recovering[0].node_name == "recovering"

    def test_get_node_history(self, orchestrator):
        """Test getting node history."""
        orchestrator._add_to_history("node1", "event", {"a": 1})
        orchestrator._add_to_history("node2", "event", {"b": 2})

        # Get all
        history = orchestrator.get_node_history()
        assert "node1" in history
        assert "node2" in history

        # Get specific
        history = orchestrator.get_node_history("node1")
        assert "node1" in history
        assert "node2" not in history

        # Get nonexistent
        history = orchestrator.get_node_history("nonexistent")
        assert history["nonexistent"] == []

    def test_get_stats(self, orchestrator):
        """Test getting aggregate statistics."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        now = time.time()

        orchestrator._nodes["healthy"] = NodeHealth(
            node_name="healthy",
            state=NodeHealthState.HEALTHY,
            last_health_check_time=now - 10,
        )
        orchestrator._nodes["unhealthy"] = NodeHealth(
            node_name="unhealthy",
            state=NodeHealthState.UNHEALTHY,
            last_state_change_time=now - 5,
        )
        orchestrator._nodes["recovering"] = NodeHealth(
            node_name="recovering",
            state=NodeHealthState.RECOVERING,
        )
        orchestrator._total_health_checks = 100
        orchestrator._total_failures = 10
        orchestrator._total_recoveries = 3
        orchestrator._total_recovery_failures = 1

        stats = orchestrator.get_stats()

        assert stats.total_nodes == 3
        assert stats.healthy_nodes == 1
        assert stats.unhealthy_nodes == 1
        assert stats.recovering_nodes == 1
        assert stats.total_health_checks == 100
        assert stats.total_failures == 10
        assert stats.total_recoveries == 3
        assert stats.total_recovery_failures == 1

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["test"] = NodeHealth(
            node_name="test",
            state=NodeHealthState.UNHEALTHY,
        )

        status = orchestrator.get_status()

        assert status["subscribed"] is False
        assert status["total_nodes"] == 1
        assert "test" in status["node_names"]
        assert "test" in status["unhealthy_node_names"]

    def test_should_attempt_recovery_success(self, orchestrator):
        """Test recovery should be attempted for unhealthy node."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["test"] = NodeHealth(
            node_name="test",
            state=NodeHealthState.UNHEALTHY,
            last_recovery_time=time.time() - 120,  # Well past cooldown
        )

        assert orchestrator.should_attempt_recovery("test") is True

    def test_should_attempt_recovery_cooldown(self, orchestrator):
        """Test recovery respects cooldown period."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["test"] = NodeHealth(
            node_name="test",
            state=NodeHealthState.UNHEALTHY,
            last_recovery_time=time.time() - 10,  # Within cooldown
        )

        assert orchestrator.should_attempt_recovery("test") is False

    def test_should_attempt_recovery_not_unhealthy(self, orchestrator):
        """Test no recovery for healthy nodes."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealth,
            NodeHealthState,
        )

        orchestrator._nodes["test"] = NodeHealth(
            node_name="test",
            state=NodeHealthState.HEALTHY,
        )

        assert orchestrator.should_attempt_recovery("test") is False

    def test_should_attempt_recovery_nonexistent(self, orchestrator):
        """Test no recovery for nonexistent nodes."""
        assert orchestrator.should_attempt_recovery("nonexistent") is False

    def test_subscribe_to_events_success(self, orchestrator):
        """Test successful event subscription."""
        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            result = orchestrator.subscribe_to_events()

        assert result is True
        assert orchestrator._subscribed is True
        assert mock_bus.subscribe.call_count == 8

    def test_unsubscribe(self, orchestrator):
        """Test event unsubscription."""
        mock_bus = MagicMock()
        orchestrator._subscribed = True

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orchestrator.unsubscribe()

        assert orchestrator._subscribed is False
        assert mock_bus.unsubscribe.call_count == 8


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.monitoring.node_health_orchestrator import reset_health_orchestrator

        reset_health_orchestrator()
        yield
        reset_health_orchestrator()

    def test_wire_health_events(self):
        """Test wiring health events."""
        from app.monitoring.node_health_orchestrator import (
            wire_health_events,
            get_health_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_health_events()

        assert orch is not None
        assert get_health_orchestrator() is orch
        assert orch._subscribed is True

    def test_wire_health_events_singleton(self):
        """Test that wire_health_events returns same instance."""
        from app.monitoring.node_health_orchestrator import wire_health_events

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch1 = wire_health_events()
            orch2 = wire_health_events()

        assert orch1 is orch2

    def test_get_health_orchestrator_none(self):
        """Test get_health_orchestrator returns None when not wired."""
        from app.monitoring.node_health_orchestrator import get_health_orchestrator

        assert get_health_orchestrator() is None

    def test_reset_health_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.monitoring.node_health_orchestrator import (
            wire_health_events,
            get_health_orchestrator,
            reset_health_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            wire_health_events()

        assert get_health_orchestrator() is not None

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            reset_health_orchestrator()

        assert get_health_orchestrator() is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestHealthIntegration:
    """Integration tests for health orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.monitoring.node_health_orchestrator import reset_health_orchestrator

        reset_health_orchestrator()
        yield
        reset_health_orchestrator()

    def test_full_health_lifecycle(self):
        """Test full node health lifecycle from healthy through recovery."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealthOrchestrator,
            NodeHealthState,
        )

        @dataclass
        class MockEvent:
            payload: dict

        orch = NodeHealthOrchestrator(recovery_cooldown_seconds=0)

        node_name = "gh200-test"

        # Node starts unknown, becomes healthy on first check
        orch._on_health_check_passed(MockEvent(payload={"node_name": node_name}))
        assert orch._nodes[node_name].state == NodeHealthState.HEALTHY

        # Multiple health check failures
        for _ in range(3):
            orch._on_health_check_failed(MockEvent(payload={
                "node_name": node_name,
                "error": "Connection timeout",
            }))

        assert orch._nodes[node_name].state == NodeHealthState.UNHEALTHY
        assert orch.should_attempt_recovery(node_name) is True

        # Recovery initiated
        orch._on_recovery_initiated(MockEvent(payload={"node_name": node_name}))
        assert orch._nodes[node_name].state == NodeHealthState.RECOVERING

        # Recovery completed
        orch._on_recovery_completed(MockEvent(payload={"node_name": node_name}))
        assert orch._nodes[node_name].state == NodeHealthState.HEALTHY

        stats = orch.get_stats()
        assert stats.total_recoveries == 1

    def test_cluster_status_tracking(self):
        """Test tracking multiple nodes in cluster."""
        from app.monitoring.node_health_orchestrator import (
            NodeHealthOrchestrator,
            NodeHealthState,
        )

        @dataclass
        class MockEvent:
            payload: dict

        orch = NodeHealthOrchestrator()

        nodes = ["gh200-a", "gh200-b", "gh200-c", "mac-studio"]

        # All nodes healthy
        for node in nodes:
            orch._on_health_check_passed(MockEvent(payload={"node_name": node}))

        stats = orch.get_stats()
        assert stats.total_nodes == 4
        assert stats.healthy_nodes == 4

        # One node fails
        orch._on_health_check_failed(MockEvent(payload={"node_name": "gh200-c", "error": "Error"}))
        orch._on_health_check_failed(MockEvent(payload={"node_name": "gh200-c", "error": "Error"}))

        stats = orch.get_stats()
        assert stats.healthy_nodes == 3
        assert stats.unhealthy_nodes == 1
        assert len(orch.get_unhealthy_nodes()) == 1
