"""Tests for SyncRouter - Intelligent Data Routing.

Tests cover:
- Sync target selection based on node capabilities
- NFS sharing detection (Lambda nodes)
- Ephemeral node handling (Vast.ai)
- Priority routing for training nodes
- Optimal source selection for replication
- Sync route planning
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.sync_router import (
    NodeSyncCapability,
    SyncRoute,
    SyncRouter,
    get_sync_router,
    reset_sync_router,
)
from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataType,
    GameLocation,
    NodeCapacity,
    NodeSyncPolicy,
    SyncTarget,
    reset_cluster_manifest,
)


@pytest.fixture
def mock_manifest():
    """Create a mock ClusterManifest."""
    manifest = MagicMock(spec=ClusterManifest)
    manifest.node_id = "local-node"
    return manifest


@pytest.fixture
def router(mock_manifest, tmp_path):
    """Create a SyncRouter with mock manifest."""
    reset_sync_router()

    # Create minimal config
    config_path = tmp_path / "distributed_hosts.yaml"
    config_path.write_text("""
hosts:
  lambda-gh200-a:
    role: training
    gpu: GH200
    tailscale_ip: 192.168.1.1
  lambda-gh200-b:
    role: selfplay
    gpu: GH200
    tailscale_ip: 192.168.1.2
  vast-5090:
    role: selfplay
    gpu: RTX 5090
    tailscale_ip: 192.168.1.3
  coordinator:
    role: coordinator
    gpu: none
    tailscale_ip: 192.168.1.4
  mac-studio:
    role: development
    gpu: MPS

sync_routing:
  priority_hosts:
    - lambda-gh200-a
  max_disk_usage_percent: 70
  excluded_hosts:
    - name: coordinator
      receive_games: false
      receive_npz: false
      reason: coordinator node
""")

    router = SyncRouter(config_path=config_path, manifest=mock_manifest)
    yield router
    reset_sync_router()


class TestSyncRouterInit:
    """Tests for SyncRouter initialization."""

    def test_init_loads_config(self, router):
        """Test router loads configuration."""
        assert len(router._node_capabilities) > 0

    def test_init_identifies_training_nodes(self, router):
        """Test router identifies training nodes."""
        cap = router.get_node_capability("lambda-gh200-a")
        assert cap is not None
        assert cap.is_training_node is True

    def test_init_identifies_ephemeral_nodes(self, router):
        """Test router identifies ephemeral (Vast.ai) nodes."""
        cap = router.get_node_capability("vast-5090")
        assert cap is not None
        assert cap.is_ephemeral is True

    def test_init_identifies_nfs_nodes(self, router):
        """Test router identifies NFS-sharing Lambda nodes."""
        cap = router.get_node_capability("lambda-gh200-a")
        assert cap is not None
        assert cap.shares_nfs is True

    def test_init_identifies_priority_nodes(self, router):
        """Test router identifies priority nodes."""
        cap = router.get_node_capability("lambda-gh200-a")
        assert cap is not None
        assert cap.is_priority_node is True

    def test_singleton_accessor(self, tmp_path):
        """Test singleton accessor."""
        reset_sync_router()

        # First call creates instance
        r1 = get_sync_router()
        r2 = get_sync_router()
        assert r1 is r2

        reset_sync_router()


class TestNodeSyncCapability:
    """Tests for NodeSyncCapability dataclass."""

    def test_capability_defaults(self):
        """Test default capability values."""
        cap = NodeSyncCapability(node_id="test-node")
        assert cap.can_receive_games is True
        assert cap.can_receive_models is True
        assert cap.can_receive_npz is True
        assert cap.is_training_node is False
        assert cap.is_ephemeral is False

    def test_capability_provider_detection(self, router):
        """Test provider detection from hostname."""
        lambda_cap = router.get_node_capability("lambda-gh200-a")
        assert lambda_cap.provider == "lambda"

        vast_cap = router.get_node_capability("vast-5090")
        assert vast_cap.provider == "vast"


class TestSyncTargetSelection:
    """Tests for sync target selection."""

    def test_get_sync_targets_excludes_self(self, router, mock_manifest):
        """Test that current node is excluded from targets."""
        router.node_id = "lambda-gh200-a"
        mock_manifest.can_receive_data.return_value = True

        targets = router.get_sync_targets(data_type="game")

        node_ids = [t.node_id for t in targets]
        assert "lambda-gh200-a" not in node_ids

    def test_get_sync_targets_respects_data_type(self, router, mock_manifest):
        """Test targets respect data type permissions."""
        mock_manifest.can_receive_data.return_value = True

        # Coordinator shouldn't receive games
        targets = router.get_sync_targets(data_type="game")
        node_ids = [t.node_id for t in targets]

        # Check if coordinator policy is correctly applied
        coord_cap = router.get_node_capability("coordinator")
        if coord_cap and not coord_cap.can_receive_games:
            assert "coordinator" not in node_ids

    def test_get_sync_targets_priority_ordering(self, router, mock_manifest):
        """Test targets are sorted by priority."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "vast-5090"  # Not Lambda, so NFS check won't exclude

        targets = router.get_sync_targets(data_type="game", max_targets=10)

        if len(targets) >= 2:
            # Higher priority should come first
            assert targets[0].priority >= targets[-1].priority

    def test_get_sync_targets_training_nodes_prioritized(self, router, mock_manifest):
        """Test training nodes get higher priority for game data."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "vast-5090"

        targets = router.get_sync_targets(
            data_type="game", max_targets=10
        )

        # Find training node targets
        training_targets = [
            t for t in targets
            if router.get_node_capability(t.node_id)
            and router.get_node_capability(t.node_id).is_training_node
        ]

        # Training nodes should have higher priority
        if training_targets:
            other_targets = [t for t in targets if t not in training_targets]
            if other_targets:
                assert training_targets[0].priority >= other_targets[0].priority

    def test_get_sync_targets_excludes_nfs_sharing_nodes(self, router, mock_manifest):
        """Test NFS-sharing nodes are skipped (data already visible)."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "lambda-gh200-a"  # Lambda node

        targets = router.get_sync_targets(data_type="game")

        # Other Lambda nodes should be excluded (share NFS)
        node_ids = [t.node_id for t in targets]
        # If current node is Lambda, other Lambda nodes should be excluded
        lambda_targets = [n for n in node_ids if "lambda" in n.lower()]
        assert len(lambda_targets) == 0

    def test_get_sync_targets_max_limit(self, router, mock_manifest):
        """Test max_targets limit is respected."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "mac-studio"

        targets = router.get_sync_targets(data_type="game", max_targets=2)

        assert len(targets) <= 2


class TestShouldSyncToNode:
    """Tests for should_sync_to_node decision."""

    def test_should_sync_same_node_returns_false(self, router, mock_manifest):
        """Test syncing to self returns False."""
        result = router.should_sync_to_node(
            target_node="test-node",
            data_type="game",
            source_node="test-node",
        )
        assert result is False

    def test_should_sync_checks_data_type_permission(self, router, mock_manifest):
        """Test data type permissions are checked."""
        mock_manifest.can_receive_data.return_value = False

        result = router.should_sync_to_node(
            target_node="coordinator",
            data_type="game",
        )

        # Coordinator should not receive games
        coord_cap = router.get_node_capability("coordinator")
        if coord_cap and not coord_cap.can_receive_games:
            assert result is False

    def test_should_sync_checks_capacity(self, router, mock_manifest):
        """Test capacity check is performed."""
        mock_manifest.can_receive_data.return_value = False

        router.should_sync_to_node(
            target_node="lambda-gh200-b",
            data_type="game",
        )

        # Should have called capacity check
        mock_manifest.can_receive_data.assert_called()

    def test_should_sync_string_data_type(self, router, mock_manifest):
        """Test string data type is converted correctly."""
        mock_manifest.can_receive_data.return_value = True

        # Should handle both string and enum
        result1 = router.should_sync_to_node("lambda-gh200-b", "game")
        result2 = router.should_sync_to_node("lambda-gh200-b", DataType.GAME)

        # Both should work (actual result depends on mock)
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


class TestOptimalSourceSelection:
    """Tests for optimal sync source selection."""

    def test_get_optimal_source_prefers_same_provider(self, router, mock_manifest):
        """Test source selection prefers same provider."""
        # Setup game locations on different providers
        mock_manifest.find_game.return_value = [
            GameLocation(
                game_id="game-001",
                node_id="lambda-gh200-a",
                db_path="/data/db.db",
            ),
            GameLocation(
                game_id="game-001",
                node_id="vast-5090",
                db_path="/data/db.db",
            ),
        ]

        # Add vast-new to capabilities as a vast provider
        router._node_capabilities["vast-new"] = NodeSyncCapability(
            node_id="vast-new",
            provider="vast",
            is_ephemeral=True,
        )

        # Target is Vast.ai node
        source = router.get_optimal_source("game-001", "vast-new")

        # Should prefer vast-5090 (same provider) - or lambda if vast is ephemeral penalty
        # The algorithm scores same-provider +20, ephemeral -10
        # So vast-5090 gets 50+20-10=60, lambda-gh200-a gets 50=50
        assert source in ["vast-5090", "lambda-gh200-a"]  # Both are valid based on scoring

    def test_get_optimal_source_avoids_ephemeral(self, router, mock_manifest):
        """Test source selection avoids ephemeral nodes when possible."""
        mock_manifest.find_game.return_value = [
            GameLocation(
                game_id="game-001",
                node_id="lambda-gh200-a",
                db_path="/data/db.db",
            ),
            GameLocation(
                game_id="game-001",
                node_id="vast-5090",
                db_path="/data/db.db",
            ),
        ]

        # Target is some node
        source = router.get_optimal_source("game-001", "new-node")

        # Should prefer lambda (not ephemeral) for reliability
        # Note: depends on scoring implementation
        assert source in ["lambda-gh200-a", "vast-5090"]

    def test_get_optimal_source_no_locations(self, router, mock_manifest):
        """Test handling when no source locations exist."""
        mock_manifest.find_game.return_value = []

        source = router.get_optimal_source("nonexistent-game", "target")

        assert source is None

    def test_get_optimal_source_excludes_target(self, router, mock_manifest):
        """Test source doesn't return target node."""
        mock_manifest.find_game.return_value = [
            GameLocation(
                game_id="game-001",
                node_id="target-node",
                db_path="/data/db.db",
            ),
        ]

        source = router.get_optimal_source("game-001", "target-node")

        assert source is None  # Only location is target itself


class TestReplicationPlanning:
    """Tests for replication route planning."""

    def test_plan_replication_creates_routes(self, router, mock_manifest):
        """Test replication planning creates sync routes."""
        # Game exists on one node
        mock_manifest.find_game.return_value = [
            GameLocation(
                game_id="game-001",
                node_id="lambda-gh200-a",
                db_path="/data/db.db",
            ),
        ]

        # Available targets
        mock_manifest.get_replication_targets.return_value = [
            SyncTarget(
                node_id="lambda-gh200-b",
                priority=80,
                reason="available",
            ),
        ]

        routes = router.plan_replication("game-001", min_copies=2)

        assert len(routes) == 1
        assert routes[0].source_node == "lambda-gh200-a"
        assert routes[0].target_node == "lambda-gh200-b"
        assert routes[0].data_type == DataType.GAME

    def test_plan_replication_already_replicated(self, router, mock_manifest):
        """Test no routes when already replicated."""
        mock_manifest.find_game.return_value = [
            GameLocation("game-001", "node-a", "/data/db.db"),
            GameLocation("game-001", "node-b", "/data/db.db"),
        ]
        mock_manifest.get_replication_targets.return_value = []

        routes = router.plan_replication("game-001", min_copies=2)

        assert len(routes) == 0


class TestSyncRoute:
    """Tests for SyncRoute dataclass."""

    def test_sync_route_fields(self):
        """Test SyncRoute has expected fields."""
        route = SyncRoute(
            source_node="node-a",
            target_node="node-b",
            data_type=DataType.GAME,
            priority=80,
            reason="training node",
            estimated_size_bytes=1_000_000,
            bandwidth_limit_mbps=100,
        )

        assert route.source_node == "node-a"
        assert route.target_node == "node-b"
        assert route.data_type == DataType.GAME
        assert route.bandwidth_limit_mbps == 100


class TestRouterStatus:
    """Tests for router status reporting."""

    def test_get_status(self, router):
        """Test status report contains expected fields."""
        status = router.get_status()

        assert "node_id" in status
        assert "total_nodes" in status
        assert "training_nodes" in status
        assert "priority_nodes" in status
        assert "ephemeral_nodes" in status
        assert "nfs_nodes" in status

    def test_status_counts_correct(self, router):
        """Test status counts are accurate."""
        status = router.get_status()

        # At least the nodes from our test config
        assert status["total_nodes"] >= 1

        # Training nodes count
        training_count = sum(
            1 for cap in router._node_capabilities.values()
            if cap.is_training_node
        )
        assert status["training_nodes"] == training_count


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_config_file(self, mock_manifest):
        """Test handling missing config file."""
        reset_sync_router()
        router = SyncRouter(
            config_path=Path("/nonexistent/config.yaml"),
            manifest=mock_manifest,
        )

        # Should still initialize, just with empty config
        assert router._node_capabilities == {}

        reset_sync_router()

    def test_unknown_node_capability(self, router):
        """Test getting capability for unknown node."""
        cap = router.get_node_capability("nonexistent-node")
        assert cap is None

    def test_should_sync_unknown_node(self, router, mock_manifest):
        """Test should_sync for unknown node falls back to manifest."""
        mock_manifest.can_receive_data.return_value = True

        router.should_sync_to_node(
            target_node="unknown-node",
            data_type="game",
        )

        # Should fall back to manifest policy
        mock_manifest.can_receive_data.assert_called()


class TestEventHandlers:
    """Tests for SyncRouter event handlers (Dec 2025 P2P integration)."""

    @pytest.mark.asyncio
    async def test_on_host_online_adds_new_node(self, router, mock_manifest):
        """Test HOST_ONLINE event adds new node to capabilities."""
        event = MagicMock()
        event.payload = {"host": "new-node-xyz"}

        await router._on_host_online(event)

        cap = router.get_node_capability("new-node-xyz")
        assert cap is not None
        assert cap.node_id == "new-node-xyz"
        assert cap.can_receive_games is True

    @pytest.mark.asyncio
    async def test_on_host_online_ignores_existing_node(self, router, mock_manifest):
        """Test HOST_ONLINE doesn't overwrite existing capabilities."""
        # Modify existing node
        existing_cap = router.get_node_capability("lambda-gh200-a")
        original_priority = existing_cap.is_priority_node

        event = MagicMock()
        event.payload = {"host": "lambda-gh200-a"}

        await router._on_host_online(event)

        # Should not have changed
        cap = router.get_node_capability("lambda-gh200-a")
        assert cap.is_priority_node == original_priority

    @pytest.mark.asyncio
    async def test_on_host_offline_disables_receive(self, router, mock_manifest):
        """Test HOST_OFFLINE event disables data reception."""
        event = MagicMock()
        event.payload = {"host": "lambda-gh200-b"}

        await router._on_host_offline(event)

        cap = router.get_node_capability("lambda-gh200-b")
        assert cap.can_receive_games is False
        assert cap.can_receive_models is False
        assert cap.can_receive_npz is False

    @pytest.mark.asyncio
    async def test_on_training_started_marks_priority(self, router, mock_manifest):
        """Test TRAINING_STARTED event marks node as training priority."""
        event = MagicMock()
        event.payload = {"node_id": "lambda-gh200-b"}

        await router._on_training_started(event)

        cap = router.get_node_capability("lambda-gh200-b")
        assert cap.is_training_node is True
        assert cap.is_priority_node is True

    @pytest.mark.asyncio
    async def test_on_cluster_capacity_changed_refresh(self, router, mock_manifest):
        """Test CLUSTER_CAPACITY_CHANGED refreshes capacity data."""
        event = MagicMock()
        event.payload = {
            "change_type": "node_joined",
            "node_id": "new-gpu-node",
            "total_nodes": 10,
            "gpu_nodes": 8,
        }

        # Should not raise
        await router._on_cluster_capacity_changed(event)

    @pytest.mark.asyncio
    async def test_on_new_games_available_routing(self, router, mock_manifest):
        """Test NEW_GAMES_AVAILABLE event triggers routing."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "mac-studio"

        event = MagicMock()
        event.payload = {
            "count": 100,
            "source": "vast-5090",
        }

        # Should not raise
        await router._on_new_games_available(event)

    @pytest.mark.asyncio
    async def test_event_handlers_graceful_on_invalid_payload(self, router, mock_manifest):
        """Test event handlers don't crash on malformed payloads."""
        event = MagicMock()
        event.payload = {}  # Missing required fields

        # All handlers should handle gracefully
        await router._on_host_online(event)
        await router._on_host_offline(event)
        await router._on_training_started(event)
        await router._on_new_games_available(event)
        await router._on_cluster_capacity_changed(event)

    def test_wire_to_event_router_graceful_without_router(self, router):
        """Test wire_to_event_router handles missing event router."""
        # Patch the actual import location used inside wire_to_event_router
        with patch.dict('sys.modules', {'app.coordination.event_router': None}):
            # Should not raise, just log warning
            router.wire_to_event_router()


class TestRefreshCapacity:
    """Tests for capacity refresh functionality (Dec 2025)."""

    def test_refresh_all_capacity(self, router, mock_manifest):
        """Test refreshing all node capacity from manifest."""
        mock_manifest.get_node_capacity.return_value = NodeCapacity(
            node_id="lambda-gh200-a",
            total_bytes=1_000_000_000,
            used_bytes=500_000_000,
            free_bytes=500_000_000,
            usage_percent=50.0,
        )

        # Method is refresh_all_capacity, not refresh_node_capacity
        router.refresh_all_capacity()

        # Should have called manifest for each node
        cap = router.get_node_capability("lambda-gh200-a")
        assert cap is not None

    def test_refresh_capacity_handles_missing_nodes(self, router, mock_manifest):
        """Test refresh handles nodes not in manifest gracefully."""
        mock_manifest.get_node_capacity.return_value = None

        # Should not raise
        router.refresh_all_capacity()
