"""Tests for HealthCoordinator.

January 3, 2026 - Sprint 13: Comprehensive test coverage for unified P2P health coordination.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from scripts.p2p.health_coordinator import (
    CircuitBreakerSummary,
    ClusterHealthState,
    DaemonHealthSummary,
    GossipHealthSummary,
    HealthCoordinator,
    OverallHealthLevel,
    RecoveryAction,
    get_health_coordinator,
    reset_health_coordinator,
)
from scripts.p2p.leader_election import QuorumHealthLevel


class TestOverallHealthLevel:
    """Tests for OverallHealthLevel enum."""

    def test_ordering(self):
        """Test health levels are ordered correctly."""
        assert OverallHealthLevel.CRITICAL < OverallHealthLevel.DEGRADED
        assert OverallHealthLevel.DEGRADED < OverallHealthLevel.WARNING
        assert OverallHealthLevel.WARNING < OverallHealthLevel.HEALTHY

    def test_values(self):
        """Test enum values."""
        assert OverallHealthLevel.CRITICAL.value == "critical"
        assert OverallHealthLevel.HEALTHY.value == "healthy"


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_values(self):
        """Test enum values."""
        assert RecoveryAction.RESTART_P2P.value == "restart_p2p"
        assert RecoveryAction.TRIGGER_ELECTION.value == "trigger_election"
        assert RecoveryAction.NONE.value == "none"


class TestDaemonHealthSummary:
    """Tests for DaemonHealthSummary dataclass."""

    def test_defaults(self):
        """Test default values."""
        summary = DaemonHealthSummary()
        assert summary.running_count == 0
        assert summary.failed_count == 0
        assert summary.total_count == 0
        assert summary.critical_failed == []

    def test_failure_rate(self):
        """Test failure rate calculation."""
        summary = DaemonHealthSummary(running_count=8, failed_count=2, total_count=10)
        assert summary.failure_rate == 0.2

    def test_failure_rate_zero_total(self):
        """Test failure rate with zero total."""
        summary = DaemonHealthSummary()
        assert summary.failure_rate == 0.0

    def test_is_healthy(self):
        """Test is_healthy property."""
        healthy = DaemonHealthSummary(running_count=10, failed_count=0, total_count=10)
        assert healthy.is_healthy is True

        unhealthy = DaemonHealthSummary(running_count=5, failed_count=5, total_count=10)
        assert unhealthy.is_healthy is False

    def test_is_healthy_with_critical_failed(self):
        """Test is_healthy with critical failures."""
        summary = DaemonHealthSummary(
            running_count=9, failed_count=1, total_count=10, critical_failed=["auto_sync"]
        )
        assert summary.is_healthy is False


class TestGossipHealthSummary:
    """Tests for GossipHealthSummary dataclass."""

    def test_defaults(self):
        """Test default values."""
        summary = GossipHealthSummary()
        assert summary.total_peers == 0
        assert summary.healthy_peers == 0
        assert summary.peers_in_backoff == []

    def test_healthy_ratio(self):
        """Test healthy ratio calculation."""
        summary = GossipHealthSummary(total_peers=10, healthy_peers=8)
        assert summary.healthy_ratio == 0.8

    def test_healthy_ratio_zero_peers(self):
        """Test healthy ratio with zero peers."""
        summary = GossipHealthSummary()
        assert summary.healthy_ratio == 1.0


class TestCircuitBreakerSummary:
    """Tests for CircuitBreakerSummary dataclass."""

    def test_defaults(self):
        """Test default values."""
        summary = CircuitBreakerSummary()
        assert summary.total_circuits == 0
        assert summary.open_circuits == []

    def test_open_ratio(self):
        """Test open ratio calculation."""
        summary = CircuitBreakerSummary(
            total_circuits=10, closed_count=7, open_count=2, half_open_count=1
        )
        assert summary.open_ratio == 0.2

    def test_is_cascade_risk_high_ratio(self):
        """Test cascade risk detection with high open ratio."""
        summary = CircuitBreakerSummary(total_circuits=10, open_count=6)
        assert summary.is_cascade_risk is True

    def test_is_cascade_risk_recent_opens(self):
        """Test cascade risk detection with recent opens."""
        summary = CircuitBreakerSummary(
            total_circuits=10,
            open_count=2,
            recently_opened=["node1", "node2", "node3", "node4"],
        )
        assert summary.is_cascade_risk is True

    def test_no_cascade_risk(self):
        """Test no cascade risk."""
        summary = CircuitBreakerSummary(
            total_circuits=10, open_count=1, recently_opened=["node1"]
        )
        assert summary.is_cascade_risk is False


class TestClusterHealthState:
    """Tests for ClusterHealthState dataclass."""

    def test_defaults(self):
        """Test default values."""
        state = ClusterHealthState()
        assert state.overall_health == OverallHealthLevel.HEALTHY
        assert state.overall_score == 1.0
        assert state.quorum_health == QuorumHealthLevel.HEALTHY

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = ClusterHealthState(
            node_id="test-node",
            overall_health=OverallHealthLevel.DEGRADED,
            overall_score=0.65,
            alive_peers=5,
            total_peers=10,
        )
        d = state.to_dict()

        assert d["overall_health"] == "degraded"
        assert d["overall_score"] == 0.65
        assert d["cluster"]["node_id"] == "test-node"
        assert d["cluster"]["alive_peers"] == 5


class TestHealthCoordinator:
    """Tests for HealthCoordinator class."""

    @pytest.fixture
    def coordinator(self):
        """Create a fresh HealthCoordinator for each test."""
        return HealthCoordinator(node_id="test-node")

    def test_init(self, coordinator):
        """Test initialization."""
        assert coordinator._node_id == "test-node"
        assert coordinator._gossip_tracker is None
        assert coordinator._node_circuit_breaker is None

    def test_set_node_id(self, coordinator):
        """Test setting node ID."""
        coordinator.set_node_id("new-node")
        assert coordinator._node_id == "new-node"

    def test_update_cluster_state(self, coordinator):
        """Test updating cluster state."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.DEGRADED,
            is_leader=True,
            leader_id="test-node",
            alive_peers=5,
            total_peers=10,
        )

        assert coordinator._quorum_health == QuorumHealthLevel.DEGRADED
        assert coordinator._is_leader is True
        assert coordinator._leader_id == "test-node"
        assert coordinator._alive_peers == 5
        assert coordinator._total_peers == 10

    def test_update_daemon_health(self, coordinator):
        """Test updating daemon health."""
        summary = DaemonHealthSummary(running_count=10, total_count=12)
        coordinator.update_daemon_health(summary)

        assert coordinator._daemon_health.running_count == 10

    def test_get_cluster_health_basic(self, coordinator):
        """Test basic health state retrieval."""
        state = coordinator.get_cluster_health()

        assert isinstance(state, ClusterHealthState)
        assert state.node_id == "test-node"
        assert state.overall_health == OverallHealthLevel.HEALTHY

    def test_get_cluster_health_cached(self, coordinator):
        """Test health state caching."""
        state1 = coordinator.get_cluster_health()
        state2 = coordinator.get_cluster_health()

        # Should return same cached instance
        assert state1 is state2

    def test_get_cluster_health_force_refresh(self, coordinator):
        """Test forced health state refresh."""
        state1 = coordinator.get_cluster_health()
        state2 = coordinator.get_cluster_health(force_refresh=True)

        # Should return different instances
        assert state1 is not state2

    def test_health_score_healthy(self, coordinator):
        """Test health score calculation for healthy cluster."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, alive_peers=10, total_peers=10
        )

        state = coordinator.get_cluster_health(force_refresh=True)
        assert state.overall_score >= 0.85
        assert state.overall_health == OverallHealthLevel.HEALTHY

    def test_health_score_degraded(self, coordinator):
        """Test health score calculation for degraded cluster."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.DEGRADED, alive_peers=5, total_peers=10
        )
        coordinator.update_daemon_health(
            DaemonHealthSummary(running_count=7, failed_count=3, total_count=10)
        )

        state = coordinator.get_cluster_health(force_refresh=True)
        assert state.overall_health in (OverallHealthLevel.WARNING, OverallHealthLevel.DEGRADED)

    def test_health_score_critical(self, coordinator):
        """Test health score calculation for critical cluster."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.LOST, alive_peers=2, total_peers=10
        )
        coordinator.update_daemon_health(
            DaemonHealthSummary(running_count=3, failed_count=7, total_count=10)
        )

        state = coordinator.get_cluster_health(force_refresh=True)
        assert state.overall_health == OverallHealthLevel.CRITICAL

    def test_collect_gossip_health_no_tracker(self, coordinator):
        """Test gossip health collection without tracker."""
        summary = coordinator._collect_gossip_health()
        assert summary.total_peers == 0

    def test_collect_gossip_health_with_tracker(self, coordinator):
        """Test gossip health collection with mock tracker.

        Jan 3, 2026 Sprint 13: Updated to mock the new get_health_summary() API.
        """
        from scripts.p2p.gossip_protocol import GossipHealthSummary as TrackerHealthSummary

        # Create mock tracker with new public API
        mock_tracker = MagicMock()

        # Mock the new get_health_summary() API (Sprint 13)
        mock_summary = TrackerHealthSummary(
            failure_counts={"peer1": 0, "peer2": 2, "peer3": 5},
            last_success={"peer1": time.time(), "peer2": time.time()},
            suspected_peers=["peer3"],
            stale_peers=[],
            total_tracked_peers=3,
            failure_threshold=3,
        )
        mock_tracker.get_health_summary.return_value = mock_summary
        mock_tracker.should_skip_peer = lambda p: p == "peer3"
        mock_tracker.get_backoff_seconds = lambda p: 16.0 if p == "peer3" else 0.0

        coordinator.set_gossip_tracker(mock_tracker)
        summary = coordinator._collect_gossip_health()

        assert summary.total_peers == 3
        assert summary.suspected_peers == 1
        assert "peer3" in summary.peers_in_backoff
        assert summary.max_backoff_seconds == 16.0

    def test_collect_circuit_health_no_breaker(self, coordinator):
        """Test circuit health collection without breaker."""
        summary = coordinator._collect_circuit_health()
        assert summary.total_circuits == 0

    def test_collect_circuit_health_with_breaker(self, coordinator):
        """Test circuit health collection with mock breaker."""
        from app.coordination.node_circuit_breaker import NodeCircuitState

        mock_circuit_data = MagicMock()
        mock_circuit_data.state = NodeCircuitState.OPEN
        mock_circuit_data.opened_at = time.time() - 60  # Opened 1 min ago

        mock_breaker = MagicMock()
        mock_breaker._circuits = {
            "node1": mock_circuit_data,
            "node2": MagicMock(state=NodeCircuitState.CLOSED),
        }

        coordinator.set_node_circuit_breaker(mock_breaker)
        summary = coordinator._collect_circuit_health()

        assert summary.total_circuits == 2
        assert summary.open_count == 1
        assert summary.closed_count == 1
        assert "node1" in summary.open_circuits
        assert "node1" in summary.recently_opened


class TestRecoveryActions:
    """Tests for recovery action logic."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with extended uptime."""
        coord = HealthCoordinator(node_id="test-node")
        # Simulate running for 60 seconds (past grace period)
        coord._start_time = time.time() - 60
        return coord

    def test_recovery_action_healthy(self, coordinator):
        """Test no action needed for healthy cluster."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, alive_peers=10, total_peers=10
        )

        action = coordinator.get_recovery_action()
        assert action == RecoveryAction.NONE

    def test_recovery_action_quorum_lost(self, coordinator):
        """Test election trigger on quorum loss."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.LOST, alive_peers=2, total_peers=10
        )

        action = coordinator.get_recovery_action()
        assert action == RecoveryAction.TRIGGER_ELECTION

    def test_recovery_action_cascade_risk(self, coordinator):
        """Test partition healing on cascade risk."""
        from app.coordination.node_circuit_breaker import NodeCircuitState

        # Create mock circuit breaker with cascade risk
        mock_circuits = {}
        for i in range(6):
            mock_circuit = MagicMock()
            mock_circuit.state = NodeCircuitState.OPEN
            mock_circuit.opened_at = time.time() - 30
            mock_circuits[f"node{i}"] = mock_circuit

        for i in range(4):
            mock_circuit = MagicMock()
            mock_circuit.state = NodeCircuitState.CLOSED
            mock_circuits[f"healthy{i}"] = mock_circuit

        mock_breaker = MagicMock()
        mock_breaker._circuits = mock_circuits

        coordinator.set_node_circuit_breaker(mock_breaker)
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, alive_peers=10, total_peers=10
        )

        action = coordinator.get_recovery_action()
        assert action == RecoveryAction.HEAL_PARTITIONS

    def test_recovery_action_cooldown(self, coordinator):
        """Test recovery cooldown prevents rapid re-triggering."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.LOST, alive_peers=2, total_peers=10
        )

        # First action
        action1 = coordinator.get_recovery_action()
        assert action1 == RecoveryAction.TRIGGER_ELECTION

        # Second action should be blocked by cooldown
        action2 = coordinator.get_recovery_action()
        assert action2 == RecoveryAction.NONE


class TestElectionTrigger:
    """Tests for election trigger logic."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with extended uptime."""
        coord = HealthCoordinator(node_id="test-node")
        coord._start_time = time.time() - 60
        return coord

    def test_should_trigger_election_grace_period(self, coordinator):
        """Test no election during grace period."""
        coordinator._start_time = time.time()  # Just started

        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, leader_id=None
        )

        assert coordinator.should_trigger_election() is False

    def test_should_trigger_election_no_leader(self, coordinator):
        """Test election triggered when no leader."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, leader_id=None
        )

        assert coordinator.should_trigger_election() is True

    def test_should_trigger_election_quorum_minimum(self, coordinator):
        """Test election triggered at minimum quorum."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.MINIMUM, leader_id="other-node", is_leader=False
        )

        assert coordinator.should_trigger_election() is True

    def test_should_not_trigger_when_leader(self, coordinator):
        """Test no election triggered when we're leader."""
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.MINIMUM, leader_id="test-node", is_leader=True
        )

        assert coordinator.should_trigger_election() is False


class TestHealthCheck:
    """Tests for health check integration."""

    def test_health_check_healthy(self):
        """Test health check result for healthy cluster."""
        coordinator = HealthCoordinator(node_id="test-node")
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.HEALTHY, alive_peers=10, total_peers=10
        )

        result = coordinator.health_check()

        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert "overall_score" in result["details"]

    def test_health_check_unhealthy(self):
        """Test health check result for unhealthy cluster."""
        coordinator = HealthCoordinator(node_id="test-node")
        coordinator.update_cluster_state(
            quorum_health=QuorumHealthLevel.LOST, alive_peers=1, total_peers=10
        )
        coordinator.update_daemon_health(
            DaemonHealthSummary(running_count=2, failed_count=8, total_count=10)
        )

        result = coordinator.health_check()

        assert result["healthy"] is False
        assert result["status"] == "critical"


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_health_coordinator()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_health_coordinator()

    def test_get_health_coordinator(self):
        """Test getting singleton instance."""
        coord1 = get_health_coordinator()
        coord2 = get_health_coordinator()

        assert coord1 is coord2

    def test_reset_health_coordinator(self):
        """Test resetting singleton."""
        coord1 = get_health_coordinator()
        reset_health_coordinator()
        coord2 = get_health_coordinator()

        assert coord1 is not coord2


class TestHealthChangeCallback:
    """Tests for health change callbacks."""

    def test_callback_on_health_change(self):
        """Test callback is triggered on health level change."""
        coordinator = HealthCoordinator(node_id="test-node")
        callback_called = []

        def on_change(state):
            callback_called.append(state.overall_health)

        coordinator.register_health_change_callback(on_change)

        # Initial healthy state
        coordinator.get_cluster_health(force_refresh=True)

        # Change to critical
        coordinator.update_cluster_state(quorum_health=QuorumHealthLevel.LOST)
        coordinator.update_daemon_health(
            DaemonHealthSummary(running_count=2, failed_count=8, total_count=10)
        )
        coordinator.get_cluster_health(force_refresh=True)

        assert len(callback_called) == 1
        assert callback_called[0] == OverallHealthLevel.CRITICAL

    def test_callback_error_handling(self):
        """Test callback errors don't break coordinator."""
        coordinator = HealthCoordinator(node_id="test-node")

        def bad_callback(state):
            raise ValueError("Callback error")

        coordinator.register_health_change_callback(bad_callback)

        # Should not raise
        coordinator.get_cluster_health(force_refresh=True)
        coordinator.update_cluster_state(quorum_health=QuorumHealthLevel.LOST)
        coordinator.get_cluster_health(force_refresh=True)
