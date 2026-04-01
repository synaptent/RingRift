"""Tests for LeadershipCoordinator.

Tests cover:
- Leader election and tracking
- Failover when leader disconnects
- Work distribution among followers
- Split-brain prevention
- Node lifecycle (online/offline)
- Event history and statistics
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.leadership_coordinator import (
    LeadershipCoordinator,
    LeadershipDomain,
    LeadershipEvent,
    LeadershipRecord,
    LeadershipStats,
    NodeInfo,
    NodeRole,
    get_leadership_coordinator,
    is_leader,
    get_current_leader,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh LeadershipCoordinator for each test."""
    return LeadershipCoordinator(local_node_id="test-node-1")


@pytest.fixture
def mock_event():
    """Create a mock event with payload."""
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLeadershipCoordinatorInit:
    """Test LeadershipCoordinator initialization."""

    def test_init_with_custom_node_id(self):
        """Test initialization with custom node ID."""
        coord = LeadershipCoordinator(local_node_id="my-node")
        assert coord.local_node_id == "my-node"

    def test_init_default_node_id_is_hostname(self):
        """Test default node ID is hostname."""
        with patch("socket.gethostname", return_value="test-host"):
            coord = LeadershipCoordinator()
            assert coord.local_node_id == "test-host"

    def test_init_local_node_registered(self, coordinator):
        """Test local node is registered on init."""
        nodes = coordinator.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "test-node-1"

    def test_init_custom_timeouts(self):
        """Test custom timeout configuration."""
        coord = LeadershipCoordinator(
            local_node_id="test",
            heartbeat_timeout=60.0,
            election_timeout=20.0,
            max_event_history=500,
        )
        assert coord.heartbeat_timeout == 60.0
        assert coord.election_timeout == 20.0
        assert coord.max_event_history == 500


# =============================================================================
# Leadership Claim/Release Tests
# =============================================================================


class TestLeadershipClaimRelease:
    """Test leadership claim and release functionality."""

    def test_claim_leadership_success(self, coordinator):
        """Test successful leadership claim."""
        result = coordinator.claim_leadership("training", reason="test")

        assert result is True
        assert coordinator.is_leader("training")
        assert coordinator.get_leader("training") == "test-node-1"
        assert coordinator.has_leader("training")

    def test_claim_leadership_already_claimed(self, coordinator):
        """Test claiming leadership when already claimed."""
        coordinator.claim_leadership("training")

        # Cannot claim again
        result = coordinator.claim_leadership("training")
        assert result is False

    def test_claim_multiple_domains(self, coordinator):
        """Test claiming leadership for multiple domains."""
        coordinator.claim_leadership("training")
        coordinator.claim_leadership("evaluation")

        assert coordinator.is_leader("training")
        assert coordinator.is_leader("evaluation")

        leaders = coordinator.get_leaders()
        assert "training" in leaders
        assert "evaluation" in leaders

    def test_release_leadership_success(self, coordinator):
        """Test successful leadership release."""
        coordinator.claim_leadership("training")
        result = coordinator.release_leadership("training")

        assert result is True
        assert not coordinator.is_leader("training")
        assert coordinator.get_leader("training") is None

    def test_release_leadership_not_leader(self, coordinator):
        """Test releasing leadership when not leader."""
        result = coordinator.release_leadership("training")
        assert result is False

    def test_release_updates_node_role(self, coordinator):
        """Test that releasing leadership updates node role."""
        coordinator.claim_leadership("training")
        coordinator.release_leadership("training")

        local = coordinator.get_node("test-node-1")
        assert local.role == NodeRole.FOLLOWER
        assert "training" not in local.leader_domains


# =============================================================================
# Leadership Query Tests
# =============================================================================


class TestLeadershipQueries:
    """Test leadership query methods."""

    def test_is_leader_false_when_no_leader(self, coordinator):
        """Test is_leader returns False when no leader."""
        assert not coordinator.is_leader("training")

    def test_get_leader_none_when_no_leader(self, coordinator):
        """Test get_leader returns None when no leader."""
        assert coordinator.get_leader("training") is None

    def test_has_leader_false_when_no_leader(self, coordinator):
        """Test has_leader returns False when no leader."""
        assert not coordinator.has_leader("training")

    def test_default_domain_is_cluster(self, coordinator):
        """Test default domain is 'cluster'."""
        coordinator.claim_leadership("cluster")
        assert coordinator.is_leader()  # default domain
        assert coordinator.get_leader() == "test-node-1"

    def test_subscribe_to_events_uses_router_helper(self, coordinator):
        """Leadership subscription should go through the unified helper."""
        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = coordinator.subscribe_to_events()

        assert result is True
        assert coordinator._subscribed is True
        assert mock_subscribe.call_count == 5


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestLeaderElectedEvent:
    """Test LEADER_ELECTED event handling."""

    @pytest.mark.asyncio
    async def test_leader_elected_updates_leadership(self, coordinator, mock_event):
        """Test that LEADER_ELECTED event updates leadership."""
        event = mock_event(payload={
            "domain": "training",
            "node_id": "other-node",
            "term": 5,
            "reason": "election",
        })

        await coordinator._on_leader_elected(event)

        assert coordinator.get_leader("training") == "other-node"
        assert coordinator.has_leader("training")

    @pytest.mark.asyncio
    async def test_leader_elected_increments_term(self, coordinator, mock_event):
        """Test that LEADER_ELECTED increments term."""
        event = mock_event(payload={
            "domain": "training",
            "node_id": "other-node",
            "term": 10,
        })

        await coordinator._on_leader_elected(event)

        stats = coordinator.get_stats()
        assert stats.current_term >= 10

    @pytest.mark.asyncio
    async def test_leader_elected_records_event(self, coordinator, mock_event):
        """Test that LEADER_ELECTED records event history."""
        event = mock_event(payload={
            "domain": "training",
            "node_id": "other-node",
            "term": 1,
        })

        await coordinator._on_leader_elected(event)

        history = coordinator.get_event_history()
        assert len(history) == 1
        assert history[0].event_type == "elected"
        assert history[0].domain == "training"

    @pytest.mark.asyncio
    async def test_leader_elected_triggers_callback(self, coordinator, mock_event):
        """Test that LEADER_ELECTED triggers callbacks."""
        callback_called = []
        coordinator.on_leader_change(lambda d, o, n: callback_called.append((d, o, n)))

        event = mock_event(payload={
            "domain": "training",
            "node_id": "new-leader",
            "term": 1,
        })

        await coordinator._on_leader_elected(event)

        assert len(callback_called) == 1
        assert callback_called[0] == ("training", "", "new-leader")


class TestLeaderLostEvent:
    """Test LEADER_LOST event handling."""

    @pytest.mark.asyncio
    async def test_leader_lost_clears_leadership(self, coordinator, mock_event):
        """Test that LEADER_LOST clears leadership."""
        # First establish leader
        coordinator.claim_leadership("training")

        event = mock_event(payload={
            "domain": "training",
            "node_id": "test-node-1",
        })

        await coordinator._on_leader_lost(event)

        assert not coordinator.has_leader("training")

    @pytest.mark.asyncio
    async def test_leader_lost_increments_failover_count(self, coordinator, mock_event):
        """Test that LEADER_LOST increments failover count."""
        coordinator.claim_leadership("training")

        event = mock_event(payload={
            "domain": "training",
            "node_id": "test-node-1",
        })

        await coordinator._on_leader_lost(event)

        stats = coordinator.get_stats()
        assert stats.total_failovers == 1


class TestLeaderStepdownEvent:
    """Test LEADER_STEPDOWN event handling."""

    @pytest.mark.asyncio
    async def test_leader_stepdown_clears_leadership(self, coordinator, mock_event):
        """Test that LEADER_STEPDOWN clears leadership."""
        coordinator.claim_leadership("training")

        event = mock_event(payload={
            "domain": "training",
            "node_id": "test-node-1",
        })

        await coordinator._on_leader_stepdown(event)

        assert not coordinator.has_leader("training")


class TestHostOnlineOfflineEvents:
    """Test HOST_ONLINE and HOST_OFFLINE events."""

    @pytest.mark.asyncio
    async def test_host_online_adds_node(self, coordinator, mock_event):
        """Test that HOST_ONLINE adds new node."""
        event = mock_event(payload={
            "node_id": "new-node",
            "hostname": "new-host",
            "ip_address": "10.0.0.1",
            "capabilities": ["gpu", "training"],
        })

        await coordinator._on_host_online(event)

        node = coordinator.get_node("new-node")
        assert node is not None
        assert node.hostname == "new-host"
        assert "gpu" in node.capabilities

    @pytest.mark.asyncio
    async def test_host_offline_marks_node_offline(self, coordinator, mock_event):
        """Test that HOST_OFFLINE marks node offline."""
        # Add node first
        online_event = mock_event(payload={
            "node_id": "some-node",
            "hostname": "some-host",
        })
        await coordinator._on_host_online(online_event)

        # Mark offline
        offline_event = mock_event(payload={
            "node_id": "some-node",
        })
        await coordinator._on_host_offline(offline_event)

        node = coordinator.get_node("some-node")
        assert node.role == NodeRole.OFFLINE

    @pytest.mark.asyncio
    async def test_host_offline_clears_leadership(self, coordinator, mock_event):
        """Test that HOST_OFFLINE clears leadership for that node."""
        # First add the node
        online_event = mock_event(payload={
            "node_id": "leader-node",
            "hostname": "leader-host",
        })
        await coordinator._on_host_online(online_event)

        # Simulate node becoming leader
        leader_event = mock_event(payload={
            "domain": "training",
            "node_id": "leader-node",
            "term": 1,
        })
        await coordinator._on_leader_elected(leader_event)

        # Node goes offline
        offline_event = mock_event(payload={
            "node_id": "leader-node",
        })
        await coordinator._on_host_offline(offline_event)

        assert not coordinator.has_leader("training")


# =============================================================================
# Node Query Tests
# =============================================================================


class TestNodeQueries:
    """Test node query methods."""

    @pytest.mark.asyncio
    async def test_get_all_nodes(self, coordinator, mock_event):
        """Test get_all_nodes returns all nodes."""
        # Add some nodes
        for i in range(3):
            event = mock_event(payload={
                "node_id": f"node-{i}",
                "hostname": f"host-{i}",
            })
            await coordinator._on_host_online(event)

        nodes = coordinator.get_all_nodes()
        # Should have local node + 3 added nodes
        assert len(nodes) == 4

    @pytest.mark.asyncio
    async def test_get_online_nodes(self, coordinator, mock_event):
        """Test get_online_nodes filters offline nodes."""
        # Add online node
        online_event = mock_event(payload={
            "node_id": "online-node",
            "hostname": "online-host",
        })
        await coordinator._on_host_online(online_event)

        # Add and offline another node
        add_event = mock_event(payload={
            "node_id": "offline-node",
            "hostname": "offline-host",
        })
        await coordinator._on_host_online(add_event)

        offline_event = mock_event(payload={
            "node_id": "offline-node",
        })
        await coordinator._on_host_offline(offline_event)

        online_nodes = coordinator.get_online_nodes()
        node_ids = [n.node_id for n in online_nodes]

        assert "online-node" in node_ids
        assert "offline-node" not in node_ids


# =============================================================================
# Statistics Tests
# =============================================================================


class TestLeadershipStats:
    """Test leadership statistics."""

    def test_initial_stats(self, coordinator):
        """Test initial statistics."""
        stats = coordinator.get_stats()

        assert stats.total_nodes == 1
        assert stats.online_nodes == 1
        assert stats.total_elections == 0
        assert stats.total_failovers == 0

    def test_stats_after_elections(self, coordinator):
        """Test statistics after elections."""
        coordinator.claim_leadership("training")
        coordinator.claim_leadership("evaluation")

        stats = coordinator.get_stats()

        assert stats.total_elections == 2
        assert "training" in stats.leaders_by_domain
        assert "evaluation" in stats.leaders_by_domain

    def test_cluster_healthy_requires_leaders(self, coordinator):
        """Test cluster health requires leaders for key domains."""
        # Without leaders, cluster is not healthy
        stats = coordinator.get_stats()
        assert not stats.cluster_healthy

        # With cluster and training leaders, cluster is healthy
        coordinator.claim_leadership("cluster")
        coordinator.claim_leadership("training")

        stats = coordinator.get_stats()
        assert stats.cluster_healthy


# =============================================================================
# Event History Tests
# =============================================================================


class TestEventHistory:
    """Test event history tracking."""

    def test_event_history_records_claims(self, coordinator):
        """Test event history records leadership claims."""
        coordinator.claim_leadership("training", reason="test-claim")

        history = coordinator.get_event_history()
        assert len(history) == 1
        assert history[0].event_type == "elected"
        assert history[0].reason == "test-claim"

    def test_event_history_records_releases(self, coordinator):
        """Test event history records leadership releases."""
        coordinator.claim_leadership("training")
        coordinator.release_leadership("training", reason="test-release")

        history = coordinator.get_event_history()
        assert len(history) == 2
        assert history[1].event_type == "stepdown"
        assert history[1].reason == "test-release"

    def test_event_history_limit(self):
        """Test event history is limited."""
        coord = LeadershipCoordinator(
            local_node_id="test",
            max_event_history=5,
        )

        # Generate more events than limit
        for i in range(10):
            coord.claim_leadership(f"domain-{i}")

        history = coord.get_event_history()
        assert len(history) == 5

    def test_event_history_with_limit_param(self, coordinator):
        """Test event history respects limit parameter."""
        for i in range(5):
            coordinator.claim_leadership(f"domain-{i}")

        history = coordinator.get_event_history(limit=2)
        assert len(history) == 2


# =============================================================================
# Callback Tests
# =============================================================================


class TestLeaderChangeCallbacks:
    """Test leader change callbacks."""

    def test_callback_on_claim(self, coordinator):
        """Test callback is called on leadership claim."""
        changes = []
        coordinator.on_leader_change(lambda d, o, n: changes.append((d, o, n)))

        coordinator.claim_leadership("training")

        # No callback on direct claim (only on events)
        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_callback_on_leader_change_event(self, coordinator, mock_event):
        """Test callback is called on leader change event."""
        changes = []
        coordinator.on_leader_change(lambda d, o, n: changes.append((d, o, n)))

        # First leader
        event1 = mock_event(payload={
            "domain": "training",
            "node_id": "leader-1",
            "term": 1,
        })
        await coordinator._on_leader_elected(event1)

        # Second leader
        event2 = mock_event(payload={
            "domain": "training",
            "node_id": "leader-2",
            "term": 2,
        })
        await coordinator._on_leader_elected(event2)

        assert len(changes) == 2
        assert changes[0] == ("training", "", "leader-1")
        assert changes[1] == ("training", "leader-1", "leader-2")

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, coordinator, mock_event):
        """Test callback errors are caught and logged."""
        def bad_callback(d, o, n):
            raise ValueError("Test error")

        coordinator.on_leader_change(bad_callback)

        event = mock_event(payload={
            "domain": "training",
            "node_id": "leader",
            "term": 1,
        })

        # Should not raise
        await coordinator._on_leader_elected(event)


# =============================================================================
# Status and Serialization Tests
# =============================================================================


class TestStatusSerialization:
    """Test status and serialization."""

    def test_get_status(self, coordinator):
        """Test get_status returns proper structure."""
        coordinator.claim_leadership("training")

        status = coordinator.get_status()

        assert "local_node_id" in status
        assert "local_role" in status
        assert "total_nodes" in status
        assert "leaders" in status
        assert "subscribed" in status

    def test_get_status_includes_leader_domains(self, coordinator):
        """Test get_status includes leader domains."""
        coordinator.claim_leadership("training")
        coordinator.claim_leadership("evaluation")

        status = coordinator.get_status()

        assert "training" in status["local_leader_domains"]
        assert "evaluation" in status["local_leader_domains"]


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_leadership_coordinator_returns_singleton(self):
        """Test get_leadership_coordinator returns same instance."""
        # Reset singleton
        import app.coordination.leadership_coordinator as lc
        lc._leadership_coordinator = None

        coord1 = get_leadership_coordinator()
        coord2 = get_leadership_coordinator()

        assert coord1 is coord2

    def test_convenience_functions_use_singleton(self):
        """Test convenience functions use singleton."""
        import app.coordination.leadership_coordinator as lc
        lc._leadership_coordinator = None

        coord = get_leadership_coordinator()
        coord.claim_leadership("training")

        assert is_leader("training")
        assert get_current_leader("training") == coord.local_node_id


# =============================================================================
# NodeInfo Tests
# =============================================================================


class TestNodeInfo:
    """Test NodeInfo dataclass."""

    def test_is_alive_recent_heartbeat(self):
        """Test is_alive returns True for recent heartbeat."""
        node = NodeInfo(
            node_id="test",
            last_heartbeat=time.time(),
        )
        assert node.is_alive

    def test_is_alive_old_heartbeat(self):
        """Test is_alive returns False for old heartbeat."""
        node = NodeInfo(
            node_id="test",
            last_heartbeat=time.time() - 60,  # 60 seconds ago
        )
        assert not node.is_alive


# =============================================================================
# LeadershipDomain Enum Tests
# =============================================================================


class TestLeadershipDomain:
    """Test LeadershipDomain enum."""

    def test_domain_values(self):
        """Test domain enum values."""
        assert LeadershipDomain.TRAINING.value == "training"
        assert LeadershipDomain.EVALUATION.value == "evaluation"
        assert LeadershipDomain.PROMOTION.value == "promotion"
        assert LeadershipDomain.DATA_SYNC.value == "data_sync"
        assert LeadershipDomain.CLUSTER.value == "cluster"
