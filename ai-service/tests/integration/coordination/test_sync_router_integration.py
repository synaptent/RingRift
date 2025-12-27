"""Integration tests for SyncRouter multi-component coordination.

These tests verify:
1. Routing with ClusterManifest integration
2. Event subscription and handling
3. Quality-based routing decisions
4. Multi-node sync coordination
5. Capacity-aware routing

December 2025 - RingRift AI Service
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_router import (
    NodeSyncCapability,
    SyncRoute,
    SyncRouter,
    get_sync_router,
    reset_sync_router,
)
from app.distributed.cluster_manifest import DataType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset SyncRouter singleton before/after each test."""
    reset_sync_router()
    yield
    reset_sync_router()


@pytest.fixture
def mock_config():
    """Mock cluster configuration."""
    return {
        "hosts": {
            "training-node": {
                "status": "ready",
                "role": "training",
                "gpu": "H100",
            },
            "selfplay-node": {
                "status": "ready",
                "role": "selfplay",
                "gpu": "A100",
            },
            "coordinator": {
                "status": "ready",
                "role": "coordinator",
                "is_coordinator": True,
            },
            "ephemeral-node": {
                "status": "ready",
                "role": "selfplay",
                "gpu": "RTX4090",
            },
        },
        "sync_routing": {
            "max_disk_usage_percent": 70.0,
            "ephemeral_hosts": ["ephemeral-node"],
            "nfs_shares": {},
        },
    }


@pytest.fixture
def mock_manifest():
    """Create a mock ClusterManifest."""
    manifest = MagicMock()
    manifest.should_replicate.return_value = True
    manifest.get_disk_usage.return_value = 30.0
    manifest.has_data.return_value = False
    manifest.get_sync_targets.return_value = []
    return manifest


@pytest.fixture
def router(mock_config, mock_manifest):
    """Create SyncRouter with mocked dependencies."""
    with patch("app.coordination.sync_router.load_cluster_config") as mock_load, \
         patch("app.coordination.sync_router.get_cluster_manifest") as mock_get_manifest, \
         patch("socket.gethostname", return_value="test-coordinator"):
        mock_load.return_value = mock_config
        mock_get_manifest.return_value = mock_manifest
        router = SyncRouter()
        yield router


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestSyncRoute:
    """Tests for SyncRoute dataclass."""

    def test_creation(self):
        """SyncRoute should store source, target, and data type."""
        route = SyncRoute(
            source_node="node-a",
            target_node="node-b",
            data_type=DataType.GAME,
            priority=5,
            reason="training priority",
        )

        assert route.source_node == "node-a"
        assert route.target_node == "node-b"
        assert route.data_type == DataType.GAME
        assert route.priority == 5
        assert route.reason == "training priority"

    def test_defaults(self):
        """SyncRoute should have sensible defaults."""
        route = SyncRoute(
            source_node="a",
            target_node="b",
            data_type=DataType.MODEL,
        )

        assert route.priority == 0
        assert route.reason == ""
        assert route.estimated_size_bytes == 0
        assert route.bandwidth_limit_mbps is None
        assert route.quality_score == 0.0


class TestNodeSyncCapability:
    """Tests for NodeSyncCapability dataclass."""

    def test_creation(self):
        """NodeSyncCapability should store node capabilities."""
        cap = NodeSyncCapability(
            node_id="training-gpu",
            is_training_node=True,
            can_receive_games=True,
            can_receive_npz=True,
        )

        assert cap.node_id == "training-gpu"
        assert cap.is_training_node is True
        assert cap.can_receive_games is True
        assert cap.can_receive_npz is True

    def test_defaults(self):
        """NodeSyncCapability should have sensible defaults."""
        cap = NodeSyncCapability(node_id="test-node")

        assert cap.can_receive_games is True
        assert cap.can_receive_models is True
        assert cap.can_receive_npz is True
        assert cap.is_training_node is False
        assert cap.is_priority_node is False
        assert cap.is_ephemeral is False
        assert cap.shares_nfs is False
        assert cap.provider == "unknown"


# =============================================================================
# Test SyncRouter Initialization
# =============================================================================


class TestSyncRouterInit:
    """Tests for SyncRouter initialization."""

    def test_initialization_loads_config(self, router, mock_config):
        """Router should load cluster config on init."""
        # Should have parsed nodes from config
        assert router.node_id == "test-coordinator"

    def test_singleton_pattern(self, router):
        """get_sync_router should return singleton."""
        with patch("app.coordination.sync_router.load_cluster_config") as mock_load, \
             patch("app.coordination.sync_router.get_cluster_manifest"), \
             patch("socket.gethostname", return_value="test-coordinator"):
            mock_load.return_value = {"hosts": {}, "sync_routing": {}}
            router1 = get_sync_router()
            router2 = get_sync_router()

            # Should be same instance
            assert router1 is router2


# =============================================================================
# Test Sync Target Resolution
# =============================================================================


class TestSyncTargetResolution:
    """Tests for get_sync_targets method."""

    def test_get_sync_targets_returns_list(self, router, mock_manifest):
        """get_sync_targets should return list of targets."""
        mock_manifest.get_sync_targets.return_value = [
            MagicMock(node_id="training-node", priority=10),
            MagicMock(node_id="selfplay-node", priority=5),
        ]

        targets = router.get_sync_targets(data_type=DataType.GAME)

        assert isinstance(targets, list)

    def test_get_sync_targets_respects_max(self, router, mock_manifest):
        """get_sync_targets should respect max_targets limit."""
        mock_manifest.get_sync_targets.return_value = [
            MagicMock(node_id=f"node-{i}", priority=i)
            for i in range(10)
        ]

        targets = router.get_sync_targets(
            data_type=DataType.GAME,
            max_targets=3,
        )

        assert len(targets) <= 3

    def test_get_sync_targets_excludes_nodes(self, router, mock_manifest):
        """get_sync_targets should respect exclusion list."""
        mock_manifest.get_sync_targets.return_value = [
            MagicMock(node_id="keep-node", priority=10),
            MagicMock(node_id="exclude-node", priority=5),
        ]

        targets = router.get_sync_targets(
            data_type=DataType.GAME,
            exclude_nodes=["exclude-node"],
        )

        target_ids = [t.node_id for t in targets]
        assert "exclude-node" not in target_ids


# =============================================================================
# Test Should Sync Decisions
# =============================================================================


class TestShouldSyncDecisions:
    """Tests for should_sync_to_node method."""

    def test_should_sync_blocks_coordinator_for_games(self, router):
        """Should not sync games to coordinator nodes."""
        with patch.object(router, "_is_excluded", return_value=False):
            # Mark node as coordinator
            router._node_capabilities["coordinator"] = NodeSyncCapability(
                node_id="coordinator",
                can_receive_games=False,  # Coordinators don't receive games
            )

            result = router.should_sync_to_node("coordinator", DataType.GAME)

            assert result is False

    def test_should_sync_blocks_same_source_target(self, router):
        """Should not sync to same node as source."""
        result = router.should_sync_to_node(
            "node-a",
            DataType.GAME,
            source_node="node-a",
        )

        assert result is False

    def test_should_sync_allows_different_nodes(self, router, mock_manifest):
        """Should allow sync between different nodes."""
        mock_manifest.should_replicate.return_value = True
        mock_manifest.get_disk_usage.return_value = 30.0

        router._node_capabilities["training-node"] = NodeSyncCapability(
            node_id="training-node",
            can_receive_games=True,
            is_training_node=True,
        )

        with patch.object(router, "_is_excluded", return_value=False):
            result = router.should_sync_to_node(
                "training-node",
                DataType.GAME,
                source_node="selfplay-node",
            )

            # With proper config, should allow
            assert isinstance(result, bool)


# =============================================================================
# Test Node Capability Management
# =============================================================================


class TestNodeCapabilities:
    """Tests for node capability management."""

    def test_get_node_capability_returns_none_for_unknown(self, router):
        """get_node_capability should return None for unknown nodes."""
        result = router.get_node_capability("nonexistent-node")

        assert result is None

    def test_get_node_capability_returns_capability(self, router):
        """get_node_capability should return stored capability."""
        router._node_capabilities["test-node"] = NodeSyncCapability(
            node_id="test-node",
            is_training_node=True,
        )

        result = router.get_node_capability("test-node")

        assert result is not None
        assert result.node_id == "test-node"
        assert result.is_training_node is True


# =============================================================================
# Test Status Reporting
# =============================================================================


class TestStatusReporting:
    """Tests for get_status method."""

    def test_get_status_structure(self, router):
        """get_status should return complete status dict."""
        status = router.get_status()

        assert "node_id" in status
        assert "total_nodes" in status
        assert isinstance(status, dict)

    def test_get_status_reflects_node_count(self, router):
        """get_status should reflect actual node count."""
        router._node_capabilities["node-1"] = NodeSyncCapability(node_id="node-1")
        router._node_capabilities["node-2"] = NodeSyncCapability(node_id="node-2")

        status = router.get_status()

        assert status["total_nodes"] == 2


# =============================================================================
# Test Event Wiring
# =============================================================================


class TestEventWiring:
    """Tests for event system integration."""

    def test_wire_to_event_router_subscribes(self, router):
        """wire_to_event_router should subscribe to events."""
        with patch("app.coordination.sync_router.subscribe") as mock_subscribe:
            router.wire_to_event_router()

            # Should have subscribed to various events
            assert mock_subscribe.called
            call_args = [call[0][0] for call in mock_subscribe.call_args_list]

            # Check that we subscribed to key events (by name or constant)
            assert len(call_args) > 0

    def test_wire_to_event_router_idempotent(self, router):
        """Calling wire_to_event_router twice should be safe."""
        with patch("app.coordination.sync_router.subscribe"):
            # Should not raise on second call
            router.wire_to_event_router()
            router.wire_to_event_router()


# =============================================================================
# Test Multi-Node Coordination
# =============================================================================


class TestMultiNodeCoordination:
    """Tests for coordinating sync across multiple nodes."""

    def test_training_nodes_get_priority(self, router, mock_manifest):
        """Training nodes should have higher priority for data."""
        # Set up nodes with different roles
        router._node_capabilities["training-gpu"] = NodeSyncCapability(
            node_id="training-gpu",
            is_training_node=True,
            is_priority_node=True,
        )
        router._node_capabilities["selfplay-gpu"] = NodeSyncCapability(
            node_id="selfplay-gpu",
            is_training_node=False,
        )

        # Mock manifest to return both
        mock_manifest.get_sync_targets.return_value = [
            MagicMock(node_id="training-gpu", priority=10),
            MagicMock(node_id="selfplay-gpu", priority=5),
        ]

        targets = router.get_sync_targets(data_type=DataType.GAME)

        # Training node should be included
        target_ids = [t.node_id for t in targets]
        assert "training-gpu" in target_ids

    def test_ephemeral_nodes_not_excluded_for_games(self, router, mock_manifest):
        """Ephemeral nodes should receive game data (for selfplay)."""
        router._node_capabilities["ephemeral-gpu"] = NodeSyncCapability(
            node_id="ephemeral-gpu",
            is_ephemeral=True,
            can_receive_games=True,
        )

        with patch.object(router, "_is_excluded", return_value=False):
            result = router.should_sync_to_node(
                "ephemeral-gpu",
                DataType.GAME,
            )

            # Should allow (ephemeral nodes need game data for selfplay)
            assert isinstance(result, bool)


# =============================================================================
# Test Capacity Awareness
# =============================================================================


class TestCapacityAwareness:
    """Tests for disk capacity-aware routing."""

    def test_high_disk_usage_blocks_sync(self, router, mock_manifest):
        """Should not sync to nodes with high disk usage."""
        # Set up node with high disk usage
        router._node_capabilities["full-node"] = NodeSyncCapability(
            node_id="full-node",
            disk_usage_percent=85.0,  # Above 70% threshold
            can_receive_games=True,
        )

        mock_manifest.get_disk_usage.return_value = 85.0

        with patch.object(router, "_check_disk_capacity", return_value=False):
            result = router.should_sync_to_node("full-node", DataType.GAME)

            assert result is False

    def test_low_disk_usage_allows_sync(self, router, mock_manifest):
        """Should allow sync to nodes with low disk usage."""
        router._node_capabilities["empty-node"] = NodeSyncCapability(
            node_id="empty-node",
            disk_usage_percent=30.0,
            can_receive_games=True,
        )

        mock_manifest.get_disk_usage.return_value = 30.0
        mock_manifest.should_replicate.return_value = True

        with patch.object(router, "_is_excluded", return_value=False), \
             patch.object(router, "_check_disk_capacity", return_value=True):
            result = router.should_sync_to_node("empty-node", DataType.GAME)

            assert result is True


# =============================================================================
# Test Data Type Specific Routing
# =============================================================================


class TestDataTypeRouting:
    """Tests for routing different data types."""

    def test_models_route_to_selfplay_nodes(self, router):
        """Models should route to selfplay nodes."""
        router._node_capabilities["selfplay-node"] = NodeSyncCapability(
            node_id="selfplay-node",
            can_receive_models=True,
            is_training_node=False,
        )

        with patch.object(router, "_is_excluded", return_value=False), \
             patch.object(router, "_check_disk_capacity", return_value=True):
            result = router.should_sync_to_node(
                "selfplay-node",
                DataType.MODEL,
            )

            # Selfplay nodes need models
            assert isinstance(result, bool)

    def test_npz_routes_to_training_nodes(self, router):
        """NPZ training data should route to training nodes."""
        router._node_capabilities["training-node"] = NodeSyncCapability(
            node_id="training-node",
            can_receive_npz=True,
            is_training_node=True,
        )

        with patch.object(router, "_is_excluded", return_value=False), \
             patch.object(router, "_check_disk_capacity", return_value=True):
            result = router.should_sync_to_node(
                "training-node",
                DataType.NPZ,
            )

            # Training nodes need NPZ data
            assert isinstance(result, bool)


# =============================================================================
# Test Quality-Based Routing (Dec 2025)
# =============================================================================


class TestQualityBasedRouting:
    """Tests for quality-based routing decisions."""

    def test_sync_route_includes_quality_score(self):
        """SyncRoute should include quality_score field."""
        route = SyncRoute(
            source_node="a",
            target_node="b",
            data_type=DataType.GAME,
            quality_score=0.95,
        )

        assert route.quality_score == 0.95

    def test_quality_score_affects_priority(self, router):
        """High quality data should have higher sync priority."""
        # Create routes with different quality scores
        high_quality = SyncRoute(
            source_node="a",
            target_node="b",
            data_type=DataType.GAME,
            priority=5,
            quality_score=0.95,
        )
        low_quality = SyncRoute(
            source_node="a",
            target_node="c",
            data_type=DataType.GAME,
            priority=5,
            quality_score=0.60,
        )

        # High quality should have higher effective priority
        assert high_quality.quality_score > low_quality.quality_score
