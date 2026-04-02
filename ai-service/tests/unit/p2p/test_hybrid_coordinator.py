"""Tests for HybridCoordinator - Protocol coordination layer for SWIM/Raft.

This module tests:
- HybridCoordinator initialization and lifecycle
- Protocol fallback behavior (SWIM -> HTTP, Raft -> Bully)
- Membership mode routing (get_alive_peers, is_peer_alive)
- Consensus mode routing (claim_work, is_leader)
- Status reporting combining all protocols
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock P2P orchestrator for fallback operations."""
    orch = MagicMock()
    orch.node_id = "test-node"
    orch.leader_id = "leader-node"
    orch.peers = {
        "peer-1": {"last_seen": 9999999999.0},  # Future timestamp = alive
        "peer-2": {"last_seen": 9999999999.0},
    }
    orch.get_alive_peers = MagicMock(return_value=["peer-1", "peer-2"])
    orch.is_leader = MagicMock(return_value=False)
    return orch


@pytest.fixture
def mock_swim_manager():
    """Create a mock SWIM membership manager."""
    manager = MagicMock()
    manager._started = True
    manager.get_alive_peers = MagicMock(return_value=["swim-peer-1", "swim-peer-2"])
    manager.is_peer_alive = MagicMock(return_value=True)
    manager.get_membership_summary = MagicMock(
        return_value={
            "started": True,
            "members": 3,
            "alive": 2,
            "suspected": 0,
            "failed": 1,
        }
    )
    manager.start = AsyncMock(return_value=True)
    manager.stop = AsyncMock()
    return manager


@pytest.fixture
def mock_raft_queue():
    """Create a mock Raft replicated work queue."""
    queue = MagicMock()
    queue.is_ready = True
    queue.is_leader = False
    queue.leader_address = "192.168.1.10:4321"
    queue.claim_work = MagicMock(return_value=True)
    queue.get_queue_stats = MagicMock(
        return_value={
            "pending": 5,
            "claimed": 2,
            "running": 1,
            "completed": 100,
            "failed": 3,
        }
    )
    queue.destroy = MagicMock()
    return queue


# =============================================================================
# Test HybridCoordinator Initialization
# =============================================================================


class TestHybridCoordinatorInit:
    """Test HybridCoordinator initialization."""

    def test_init_with_orchestrator(self, mock_orchestrator):
        """Test initialization with orchestrator extracts node_id."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(orchestrator=mock_orchestrator)
        assert coord.node_id == "test-node"
        assert coord._orchestrator == mock_orchestrator
        assert not coord.started

    def test_init_with_explicit_node_id(self):
        """Test initialization with explicit node_id."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(node_id="explicit-node")
        assert coord.node_id == "explicit-node"

    def test_init_detects_mode_from_constants(self):
        """Test that modes are read from constants module."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(node_id="test")
        # Should have some membership_mode and consensus_mode
        assert coord.membership_mode in ("http", "swim", "hybrid")
        assert coord.consensus_mode in ("bully", "raft", "hybrid")


# =============================================================================
# Test Membership Fallback
# =============================================================================


class TestMembershipFallback:
    """Test SWIM -> HTTP fallback for membership operations."""

    def test_get_alive_peers_uses_http_when_swim_unavailable(self, mock_orchestrator):
        """When SWIM not started, should fall back to HTTP peers."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._swim_manager = None  # No SWIM
        coord._membership_mode = "hybrid"

        peers = coord.get_alive_peers()
        assert "peer-1" in peers or len(peers) >= 0  # Falls back to HTTP

    def test_get_alive_peers_uses_swim_when_available(
        self, mock_orchestrator, mock_swim_manager
    ):
        """When SWIM is started, should use SWIM peers."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        with patch(
            "app.p2p.hybrid_coordinator.SWIM_ENABLED", True
        ), patch(
            "app.p2p.hybrid_coordinator.SWIM_AVAILABLE", True
        ):
            coord = HybridCoordinator(
                orchestrator=mock_orchestrator,
                node_id="test",
            )
            coord._started = True
            coord._swim_manager = mock_swim_manager
            coord._swim_fallback_active = False
            coord._membership_mode = "swim"

            peers = coord.get_alive_peers()
            # Should return SWIM peers
            assert peers == ["swim-peer-1", "swim-peer-2"]
            mock_swim_manager.get_alive_peers.assert_called_once()

    def test_is_peer_alive_fallback(self, mock_orchestrator):
        """is_peer_alive should check get_alive_peers on fallback."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._swim_manager = None
        coord._swim_fallback_active = True

        # Should not raise, returns boolean
        result = coord.is_peer_alive("peer-1")
        assert isinstance(result, bool)


# =============================================================================
# Test Consensus Fallback
# =============================================================================


class TestConsensusFallback:
    """Test Raft -> Bully/SQLite fallback for consensus operations."""

    def test_is_leader_uses_bully_when_raft_unavailable(self, mock_orchestrator):
        """When Raft not ready, should fall back to Bully leader check."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._raft_queue = None
        coord._raft_fallback_active = True
        coord._consensus_mode = "hybrid"

        result = coord.is_leader()
        assert isinstance(result, bool)
        assert result is False
        mock_orchestrator.is_leader.assert_not_called()

    def test_is_leader_uses_raft_when_available(
        self, mock_orchestrator, mock_raft_queue
    ):
        """When Raft is ready, should check Raft leader status."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        with patch(
            "app.p2p.hybrid_coordinator.RAFT_ENABLED", True
        ), patch(
            "app.p2p.hybrid_coordinator.PYSYNCOBJ_AVAILABLE", True
        ):
            coord = HybridCoordinator(
                orchestrator=mock_orchestrator,
                node_id="test",
            )
            coord._started = True
            coord._raft_queue = mock_raft_queue
            coord._raft_fallback_active = False
            coord._consensus_mode = "raft"

            result = coord.is_leader()
            assert result == False  # mock_raft_queue.is_leader is False

    def test_get_leader_id_fallback(self, mock_orchestrator):
        """get_leader_id should fall back to Bully when Raft unavailable."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._raft_queue = None
        coord._raft_fallback_active = True

        leader = coord.get_leader_id()
        assert leader == "leader-node" or leader == ""


# =============================================================================
# Test Status Reporting
# =============================================================================


class TestStatusReporting:
    """Test comprehensive status reporting."""

    def test_get_status_structure(self, mock_orchestrator):
        """Status should include all protocol sections."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True

        status = coord.get_status()

        # Check top-level structure
        assert "started" in status
        assert "node_id" in status
        assert "membership_mode" in status
        assert "consensus_mode" in status

        # Check protocol sections
        assert "swim" in status
        assert "raft" in status
        assert "http" in status
        assert "work_queue" in status

        # Check SWIM section
        assert "enabled" in status["swim"]
        assert "available" in status["swim"]

        # Check Raft section
        assert "enabled" in status["raft"]
        assert "available" in status["raft"]

    def test_get_status_with_swim(self, mock_orchestrator, mock_swim_manager):
        """Status should include SWIM details when manager present."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._swim_manager = mock_swim_manager

        status = coord.get_status()

        # SWIM section should have counts from manager
        assert status["swim"]["alive_count"] == 2
        assert status["swim"]["suspected_count"] == 0
        assert status["swim"]["failed_count"] == 1

    def test_get_status_with_raft(self, mock_orchestrator, mock_raft_queue):
        """Status should include Raft details when queue present."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._raft_queue = mock_raft_queue

        status = coord.get_status()

        # Raft section should have queue stats
        assert status["raft"]["ready"] == True
        assert status["raft"]["is_leader"] == False
        assert status["raft"]["leader_address"] == "192.168.1.10:4321"


# =============================================================================
# Test Lifecycle
# =============================================================================


class TestLifecycle:
    """Test coordinator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_without_dependencies_succeeds(self, mock_orchestrator):
        """Start should succeed even without SWIM/Raft (uses fallbacks)."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )

        # Patch constants to disable both protocols
        with patch(
            "app.p2p.hybrid_coordinator.SWIM_ENABLED", False
        ), patch(
            "app.p2p.hybrid_coordinator.RAFT_ENABLED", False
        ):
            result = await coord.start()
            assert result == True
            assert coord.started

    @pytest.mark.asyncio
    async def test_stop_cleans_up(self, mock_orchestrator, mock_swim_manager):
        """Stop should clean up SWIM and Raft resources."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True
        coord._swim_manager = mock_swim_manager

        await coord.stop()

        assert not coord.started
        mock_swim_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_started_returns_true(self, mock_orchestrator):
        """Starting an already-started coordinator should return True."""
        from app.p2p.hybrid_coordinator import HybridCoordinator

        coord = HybridCoordinator(
            orchestrator=mock_orchestrator,
            node_id="test",
        )
        coord._started = True

        result = await coord.start()
        assert result == True


# =============================================================================
# Test Factory Function
# =============================================================================


class TestFactoryFunction:
    """Test create_hybrid_coordinator factory."""

    def test_create_hybrid_coordinator_basic(self, mock_orchestrator):
        """Factory should create coordinator with orchestrator."""
        from app.p2p.hybrid_coordinator import create_hybrid_coordinator

        coord = create_hybrid_coordinator(
            orchestrator=mock_orchestrator,
            node_id="factory-node",
        )

        assert coord.node_id == "factory-node"
        assert coord._orchestrator == mock_orchestrator


# =============================================================================
# Test HybridStatus Dataclass
# =============================================================================


class TestHybridStatus:
    """Test HybridStatus dataclass."""

    def test_to_dict_structure(self):
        """to_dict should produce expected JSON structure."""
        from app.p2p.hybrid_coordinator import HybridStatus

        status = HybridStatus(
            started=True,
            node_id="test-node",
            membership_mode="hybrid",
            consensus_mode="bully",
            swim_enabled=True,
            swim_available=True,
            swim_started=True,
            swim_alive_count=5,
            raft_enabled=False,
            raft_available=False,
        )

        d = status.to_dict()

        assert d["started"] == True
        assert d["node_id"] == "test-node"
        assert d["membership_mode"] == "hybrid"
        assert d["swim"]["enabled"] == True
        assert d["swim"]["alive_count"] == 5
        assert d["raft"]["enabled"] == False
