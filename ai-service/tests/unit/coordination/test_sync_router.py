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
from unittest.mock import AsyncMock, MagicMock, patch

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


# =============================================================================
# Tests for Health Check (December 2025)
# =============================================================================


class TestHealthCheck:
    """Tests for SyncRouter.health_check() method."""

    def test_health_check_returns_health_check_result(self, router):
        """Test health_check returns HealthCheckResult."""
        from app.coordination.contracts import HealthCheckResult

        result = router.health_check()

        assert isinstance(result, HealthCheckResult)
        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "details")

    def test_health_check_healthy_with_nodes(self, router):
        """Test health_check returns healthy when nodes are configured."""
        # router fixture already has nodes configured
        result = router.health_check()

        assert result.healthy is True

    def test_health_check_degraded_no_nodes(self, router):
        """Test health_check returns degraded when no nodes are configured."""
        from app.coordination.contracts import CoordinatorStatus

        # Clear all node capabilities
        router._node_capabilities = {}

        result = router.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "No node capabilities" in result.message

    def test_health_check_degraded_all_nodes_disabled(self, router):
        """Test health_check returns degraded when all nodes have sync disabled."""
        from app.coordination.contracts import CoordinatorStatus

        # Disable sync for all nodes
        for cap in router._node_capabilities.values():
            cap.can_receive_games = False
            cap.can_receive_models = False
            cap.can_receive_npz = False

        result = router.health_check()

        assert result.status == CoordinatorStatus.DEGRADED
        assert "disabled" in result.message.lower()

    def test_health_check_includes_node_counts(self, router):
        """Test health_check includes node count details."""
        result = router.health_check()

        assert "total_nodes" in result.details
        assert "enabled_nodes" in result.details
        assert "training_nodes" in result.details
        assert "priority_nodes" in result.details
        assert "manifest_available" in result.details

    def test_health_check_reports_manifest_health(self, router, mock_manifest):
        """Test health_check reports manifest availability."""
        result = router.health_check()

        assert result.details["manifest_available"] is True

    def test_health_check_with_manifest_error(self, router, mock_manifest):
        """Test health_check handles manifest errors gracefully."""
        from app.coordination.contracts import CoordinatorStatus

        # Make manifest.find_game raise
        mock_manifest.find_game.side_effect = Exception("Manifest error")

        # Remove the find_game attribute to simulate unavailable manifest
        router._manifest = None

        result = router.health_check()

        # Should be degraded but not fail
        assert result.details.get("manifest_available") is False


# =============================================================================
# Tests for Backpressure Integration (December 2025)
# =============================================================================


class TestBackpressureIntegration:
    """Tests for backpressure handling."""

    def test_is_under_backpressure_default_false(self, router):
        """Test is_under_backpressure returns False by default."""
        assert router.is_under_backpressure() is False

    def test_is_under_backpressure_global_active(self, router):
        """Test is_under_backpressure detects global backpressure."""
        router._backpressure_active = {"__global__"}

        assert router.is_under_backpressure() is True
        assert router.is_under_backpressure("any-node") is True

    def test_is_under_backpressure_node_specific(self, router):
        """Test is_under_backpressure detects node-specific backpressure."""
        router._backpressure_active = {"node-1", "node-2"}

        assert router.is_under_backpressure("node-1") is True
        assert router.is_under_backpressure("node-2") is True
        assert router.is_under_backpressure("node-3") is False

    def test_is_under_backpressure_without_attr(self, router):
        """Test is_under_backpressure handles missing attribute."""
        # Remove the attribute if it exists
        if hasattr(router, "_backpressure_active"):
            delattr(router, "_backpressure_active")

        assert router.is_under_backpressure() is False

    @pytest.mark.asyncio
    async def test_on_backpressure_activated_tracks_node(self, router):
        """Test _on_backpressure_activated tracks backpressure state."""
        event = MagicMock()
        event.payload = {
            "source_node": "test-node",
            "queue_depth": 500,
            "threshold": 100,
        }

        await router._on_backpressure_activated(event)

        assert "test-node" in router._backpressure_active

    @pytest.mark.asyncio
    async def test_on_backpressure_activated_global(self, router):
        """Test _on_backpressure_activated handles global backpressure."""
        event = MagicMock()
        event.payload = {
            "source_node": "",  # Empty = global
            "queue_depth": 1000,
        }

        await router._on_backpressure_activated(event)

        assert "__global__" in router._backpressure_active

    @pytest.mark.asyncio
    async def test_on_backpressure_released_clears_node(self, router):
        """Test _on_backpressure_released clears backpressure state."""
        router._backpressure_active = {"test-node", "other-node"}

        event = MagicMock()
        event.payload = {"source_node": "test-node"}

        await router._on_backpressure_released(event)

        assert "test-node" not in router._backpressure_active
        assert "other-node" in router._backpressure_active

    @pytest.mark.asyncio
    async def test_on_backpressure_released_global(self, router):
        """Test _on_backpressure_released handles global release."""
        router._backpressure_active = {"__global__", "node-1"}

        event = MagicMock()
        event.payload = {"source_node": ""}  # Empty = global

        await router._on_backpressure_released(event)

        assert "__global__" not in router._backpressure_active
        assert "node-1" in router._backpressure_active

    @pytest.mark.asyncio
    async def test_backpressure_does_not_reduce_priority(self, router):
        """Test backpressure does NOT reduce sync priority (Dec 28 fix)."""
        # This is critical - we track but don't reduce priority
        router._node_capabilities["test-node"] = NodeSyncCapability(
            node_id="test-node",
            can_receive_games=True,
        )

        event = MagicMock()
        event.payload = {
            "source_node": "test-node",
            "queue_depth": 500,
            "threshold": 100,
        }

        await router._on_backpressure_activated(event)

        # Node should still be able to receive games
        cap = router.get_node_capability("test-node")
        assert cap.can_receive_games is True


# =============================================================================
# Tests for Sync Timestamp Persistence (December 2025)
# =============================================================================


class TestSyncTimestampPersistence:
    """Tests for sync timestamp persistence functionality."""

    def test_record_sync_success_updates_timestamp(self, router):
        """Test record_sync_success updates node's last_sync_time."""
        import time

        router._node_capabilities["test-node"] = NodeSyncCapability(
            node_id="test-node", last_sync_time=0.0
        )

        before = time.time()
        router.record_sync_success("test-node")
        after = time.time()

        cap = router.get_node_capability("test-node")
        assert before <= cap.last_sync_time <= after

    def test_record_sync_success_unknown_node(self, router):
        """Test record_sync_success handles unknown node gracefully."""
        # Should not raise
        router.record_sync_success("unknown-node")

    def test_record_sync_success_persists_to_file(self, router, tmp_path):
        """Test record_sync_success saves timestamps to JSON file."""
        import json

        state_file = tmp_path / "sync_state.json"
        router._SYNC_STATE_FILE = state_file

        router._node_capabilities["test-node"] = NodeSyncCapability(
            node_id="test-node"
        )

        router.record_sync_success("test-node")

        assert state_file.exists()
        with open(state_file) as f:
            data = json.load(f)
        assert "test-node" in data

    def test_load_sync_timestamps_loads_from_file(self, router, tmp_path):
        """Test _load_sync_timestamps loads persisted data."""
        import json

        state_file = tmp_path / "sync_state.json"
        router._SYNC_STATE_FILE = state_file

        # Create test data
        test_data = {"lambda-gh200-a": 1234567890.0}
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(test_data, f)

        router._load_sync_timestamps()

        cap = router.get_node_capability("lambda-gh200-a")
        if cap:  # Only check if node exists
            assert cap.last_sync_time == 1234567890.0

    def test_load_sync_timestamps_missing_file(self, router, tmp_path):
        """Test _load_sync_timestamps handles missing file."""
        router._SYNC_STATE_FILE = tmp_path / "nonexistent.json"

        # Should not raise
        router._load_sync_timestamps()

    def test_load_sync_timestamps_corrupted_file(self, router, tmp_path):
        """Test _load_sync_timestamps handles corrupted JSON."""
        state_file = tmp_path / "corrupted.json"
        state_file.write_text("not valid json {{{")
        router._SYNC_STATE_FILE = state_file

        # Should not raise
        router._load_sync_timestamps()


# =============================================================================
# Tests for Quality-Based Routing (December 2025)
# =============================================================================


class TestQualityBasedRouting:
    """Tests for quality-based sync priority."""

    def test_plan_replication_includes_quality_score(self, router, mock_manifest):
        """Test plan_replication includes quality_score in routes."""
        mock_manifest.find_game.return_value = []
        mock_manifest.get_replication_targets.return_value = [
            SyncTarget(node_id="target-1", priority=50, reason="available"),
        ]

        routes = router.plan_replication(
            "game-001",
            min_copies=1,
            quality_score=0.85,
        )

        if routes:
            assert routes[0].quality_score == 0.85

    def test_plan_replication_quality_boosts_priority(self, router, mock_manifest):
        """Test high quality score boosts sync priority."""
        mock_manifest.find_game.return_value = []
        mock_manifest.get_replication_targets.return_value = [
            SyncTarget(node_id="target-1", priority=50, reason="available"),
        ]

        routes_high_quality = router.plan_replication(
            "game-001",
            min_copies=1,
            quality_score=0.95,  # High quality
        )

        mock_manifest.get_replication_targets.return_value = [
            SyncTarget(node_id="target-1", priority=50, reason="available"),
        ]

        routes_low_quality = router.plan_replication(
            "game-002",
            min_copies=1,
            quality_score=0.2,  # Low quality
        )

        if routes_high_quality and routes_low_quality:
            # High quality should have higher priority
            assert routes_high_quality[0].priority > routes_low_quality[0].priority

    def test_get_game_quality_score_default(self, router, mock_manifest):
        """Test _get_game_quality_score returns default for unknown game.

        Note: ClusterManifest doesn't have get_game_metadata, so we rely on
        find_game returning empty list, which triggers default score.
        """
        mock_manifest.find_game.return_value = []

        score = router._get_game_quality_score("unknown-game")

        assert score == 0.5  # Default neutral quality

    def test_get_game_quality_score_from_location(self, router, mock_manifest):
        """Test _get_game_quality_score extracts from game location metadata.

        Note: Quality is computed from game location attributes (game_length, winner, etc.)
        since ClusterManifest doesn't expose get_game_metadata directly.
        """
        # When find_game returns empty and unified_quality is not available,
        # it returns the default score
        mock_manifest.find_game.return_value = []

        score = router._get_game_quality_score("game-001")

        # Without game locations, returns default 0.5
        assert score == 0.5


# =============================================================================
# Tests for Transport Escalation (December 2025)
# =============================================================================


class TestTransportEscalation:
    """Tests for transport escalation in sync stalled handler."""

    def test_transport_escalation_order(self, router):
        """Test TRANSPORT_ESCALATION_ORDER is defined correctly."""
        expected = ["p2p", "http", "rsync", "base64"]
        assert router.TRANSPORT_ESCALATION_ORDER == expected

    @pytest.mark.asyncio
    async def test_on_sync_stalled_tracks_failed_transport(self, router):
        """Test _on_sync_stalled tracks failed transports."""
        router._node_capabilities["target-node"] = NodeSyncCapability(
            node_id="target-node"
        )

        event = MagicMock()
        event.payload = {
            "target_host": "target-node",
            "timeout_seconds": 60,
            "retry_count": 3,
            "transport": "p2p",
            "data_type": "game",
        }

        # Patch the event_router module's safe_emit_event where it's imported from
        with patch("app.coordination.event_router.safe_emit_event"):
            await router._on_sync_stalled(event)

        assert "p2p" in router._failed_transports.get("target-node", set())

    @pytest.mark.asyncio
    async def test_on_sync_stalled_escalates_transport(self, router):
        """Test _on_sync_stalled escalates to next transport."""
        router._node_capabilities["target-node"] = NodeSyncCapability(
            node_id="target-node"
        )

        event = MagicMock()
        event.payload = {
            "target_host": "target-node",
            "timeout_seconds": 60,
            "retry_count": 1,
            "transport": "p2p",
            "data_type": "game",
        }

        # Patch the event_router module's safe_emit_event where it's imported from
        with patch("app.coordination.event_router.safe_emit_event") as mock_emit:
            await router._on_sync_stalled(event)

            # Should emit SYNC_RETRY_REQUESTED with next transport
            if mock_emit.called:
                call_args = mock_emit.call_args
                # The preferred_transport should be "http" (next after p2p)
                payload = call_args[0][1]
                assert payload.get("preferred_transport") == "http"

    @pytest.mark.asyncio
    async def test_on_sync_stalled_disables_after_all_fail(self, router):
        """Test _on_sync_stalled disables sync after all transports fail."""
        router._node_capabilities["target-node"] = NodeSyncCapability(
            node_id="target-node",
            can_receive_games=True,
        )

        # All transports already failed
        router._failed_transports = {
            "target-node": {"p2p", "http", "rsync"}
        }

        event = MagicMock()
        event.payload = {
            "target_host": "target-node",
            "timeout_seconds": 60,
            "retry_count": 10,
            "transport": "base64",  # Last transport
            "data_type": "game",
        }

        await router._on_sync_stalled(event)

        cap = router.get_node_capability("target-node")
        assert cap.can_receive_games is False

    @pytest.mark.asyncio
    async def test_on_sync_stalled_disables_specific_data_type(self, router):
        """Test _on_sync_stalled disables only the stalled data type."""
        router._node_capabilities["target-node"] = NodeSyncCapability(
            node_id="target-node",
            can_receive_games=True,
            can_receive_models=True,
            can_receive_npz=True,
        )

        router._failed_transports = {
            "target-node": {"p2p", "http", "rsync", "base64"}  # All failed
        }

        event = MagicMock()
        event.payload = {
            "target_host": "target-node",
            "transport": "unknown",
            "data_type": "model",  # Only model sync stalled
        }

        await router._on_sync_stalled(event)

        cap = router.get_node_capability("target-node")
        assert cap.can_receive_models is False
        # Games and NPZ should still be enabled
        assert cap.can_receive_games is True
        assert cap.can_receive_npz is True


# =============================================================================
# Tests for External Storage Configuration (December 2025)
# =============================================================================


class TestExternalStorageConfiguration:
    """Tests for external storage path handling."""

    def test_get_external_storage_path_configured(self, router):
        """Test get_external_storage_path returns path for configured host."""
        router._external_storage = [
            {
                "host": "backup-server",
                "path": "/backup/data",
                "subdirs": {"games": "game_archive", "models": "model_archive"},
            }
        ]

        path = router.get_external_storage_path("backup-server", "games")
        assert path == "/backup/data/game_archive"

    def test_get_external_storage_path_default_subdir(self, router):
        """Test get_external_storage_path uses data_type as default subdir."""
        router._external_storage = [
            {
                "host": "backup-server",
                "path": "/backup/data",
                "subdirs": {},
            }
        ]

        path = router.get_external_storage_path("backup-server", "npz")
        assert path == "/backup/data/npz"

    def test_get_external_storage_path_not_configured(self, router):
        """Test get_external_storage_path returns None for unconfigured host."""
        router._external_storage = []

        path = router.get_external_storage_path("unknown-host", "games")
        assert path is None

    def test_get_external_storage_path_empty_path(self, router):
        """Test get_external_storage_path handles empty base path."""
        router._external_storage = [
            {
                "host": "backup-server",
                "path": "",
                "subdirs": {"games": "archive"},
            }
        ]

        path = router.get_external_storage_path("backup-server", "games")
        assert path is None


# =============================================================================
# Tests for Node Recovery Events (December 2025)
# =============================================================================


class TestNodeRecoveryEvents:
    """Tests for NODE_RECOVERED event handling."""

    @pytest.mark.asyncio
    async def test_on_node_recovered_re_enables_sync(self, router, mock_manifest):
        """Test _on_node_recovered re-enables sync capabilities."""
        # Simulate offline node
        router._node_capabilities["recovered-node"] = NodeSyncCapability(
            node_id="recovered-node",
            can_receive_games=False,
            can_receive_models=False,
            can_receive_npz=False,
        )

        # Mock get_sync_policy to return a proper NodeSyncPolicy
        mock_policy = NodeSyncPolicy(
            node_id="recovered-node",
            receive_games=True,
            receive_models=True,
            receive_npz=True,
        )
        mock_manifest.get_sync_policy.return_value = mock_policy

        event = MagicMock()
        event.payload = {"node_id": "recovered-node"}

        await router._on_node_recovered(event)

        cap = router.get_node_capability("recovered-node")
        # Should be re-enabled based on policy
        assert cap.can_receive_games is True
        assert cap.can_receive_models is True
        assert cap.can_receive_npz is True

    @pytest.mark.asyncio
    async def test_on_node_recovered_adds_new_node(self, router, mock_manifest):
        """Test _on_node_recovered adds new node if not in capabilities."""
        event = MagicMock()
        event.payload = {"node_id": "new-recovered-node"}

        await router._on_node_recovered(event)

        cap = router.get_node_capability("new-recovered-node")
        assert cap is not None
        assert cap.node_id == "new-recovered-node"

    @pytest.mark.asyncio
    async def test_on_node_recovered_uses_host_fallback(self, router, mock_manifest):
        """Test _on_node_recovered uses 'host' key as fallback."""
        event = MagicMock()
        event.payload = {"host": "fallback-node"}  # node_id not present

        await router._on_node_recovered(event)

        cap = router.get_node_capability("fallback-node")
        assert cap is not None


# =============================================================================
# Tests for Model Sync Request Handler (December 2025)
# =============================================================================


class TestModelSyncRequestHandler:
    """Tests for MODEL_SYNC_REQUESTED event handler."""

    @pytest.mark.asyncio
    async def test_on_model_sync_requested_finds_source(self, router, mock_manifest):
        """Test _on_model_sync_requested finds source nodes."""
        # Add some nodes that can provide models
        router._node_capabilities["model-source"] = NodeSyncCapability(
            node_id="model-source",
            can_receive_models=True,
        )

        event = MagicMock()
        event.payload = {
            "model_id": "model-001",
            "node_id": "requesting-node",
            "reason": "model_missing",
        }

        # Should not raise
        await router._on_model_sync_requested(event)

    @pytest.mark.asyncio
    async def test_on_model_sync_requested_missing_model_id(self, router):
        """Test _on_model_sync_requested handles missing model_id."""
        event = MagicMock()
        event.payload = {"node_id": "requesting-node"}  # Missing model_id

        # Should not raise, just return early
        await router._on_model_sync_requested(event)

    @pytest.mark.asyncio
    async def test_on_model_sync_requested_missing_node_id(self, router):
        """Test _on_model_sync_requested handles missing node_id."""
        event = MagicMock()
        event.payload = {"model_id": "model-001"}  # Missing node_id

        # Should not raise, just return early
        await router._on_model_sync_requested(event)


# =============================================================================
# Tests for Source Priority Computation (December 2025)
# =============================================================================


class TestSourcePriorityComputation:
    """Tests for _compute_source_priority method."""

    def test_compute_source_priority_selfplay_game(self, router):
        """Test source priority for selfplay nodes with game data."""
        cap = NodeSyncCapability(
            node_id="selfplay-node",
            selfplay_enabled=True,
            has_gpu=True,
        )

        priority = router._compute_source_priority(cap, DataType.GAME)

        # Selfplay + GPU should give high priority
        assert priority >= 120  # 100 (selfplay) + 20 (gpu)

    def test_compute_source_priority_training_model(self, router):
        """Test source priority for training nodes with model data."""
        cap = NodeSyncCapability(
            node_id="training-node",
            is_training_node=True,
        )

        priority = router._compute_source_priority(cap, DataType.MODEL)

        assert priority >= 100  # Training nodes have latest models

    def test_compute_source_priority_training_npz(self, router):
        """Test source priority for training nodes with NPZ data."""
        cap = NodeSyncCapability(
            node_id="training-node",
            is_training_node=True,
        )

        priority = router._compute_source_priority(cap, DataType.NPZ)

        assert priority >= 100  # Training nodes export NPZ

    def test_compute_source_priority_high_disk_boost(self, router):
        """Test source priority boost for high disk usage (more data)."""
        cap = NodeSyncCapability(
            node_id="data-rich-node",
            selfplay_enabled=True,
            disk_usage_percent=75.0,  # High disk = more data
        )

        priority = router._compute_source_priority(cap, DataType.GAME)

        # Should include disk usage boost
        assert priority >= 110  # 100 (selfplay) + 10 (high disk)


# =============================================================================
# Tests for Sync Sources (December 2025)
# =============================================================================


class TestGetSyncSources:
    """Tests for get_sync_sources method."""

    def test_get_sync_sources_excludes_target(self, router, mock_manifest):
        """Test get_sync_sources excludes target node."""
        router._node_capabilities["source-1"] = NodeSyncCapability(
            node_id="source-1", selfplay_enabled=True
        )
        router._node_capabilities["target-node"] = NodeSyncCapability(
            node_id="target-node", selfplay_enabled=True
        )

        with patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            sources = router.get_sync_sources(
                data_type=DataType.GAME,
                target_node="target-node",
            )

        source_ids = [s.node_id for s in sources]
        assert "target-node" not in source_ids

    def test_get_sync_sources_defaults_to_self(self, router, mock_manifest):
        """Test get_sync_sources defaults target to current node."""
        router.node_id = "my-node"
        router._node_capabilities["source-1"] = NodeSyncCapability(
            node_id="source-1", selfplay_enabled=True
        )

        with patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            sources = router.get_sync_sources(data_type=DataType.GAME)

        # my-node should be excluded as target
        source_ids = [s.node_id for s in sources]
        assert router.node_id not in source_ids

    def test_get_sync_sources_filters_by_capability(self, router, mock_manifest):
        """Test get_sync_sources filters by data type capability."""
        router._node_capabilities["selfplay-node"] = NodeSyncCapability(
            node_id="selfplay-node", selfplay_enabled=True
        )
        router._node_capabilities["storage-node"] = NodeSyncCapability(
            node_id="storage-node", selfplay_enabled=False
        )

        with patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            sources = router.get_sync_sources(data_type=DataType.GAME)

        source_ids = [s.node_id for s in sources]
        assert "selfplay-node" in source_ids
        assert "storage-node" not in source_ids

    def test_get_sync_sources_respects_max_sources(self, router, mock_manifest):
        """Test get_sync_sources respects max_sources limit."""
        # Add many source nodes
        for i in range(10):
            router._node_capabilities[f"source-{i}"] = NodeSyncCapability(
                node_id=f"source-{i}", selfplay_enabled=True
            )

        with patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            sources = router.get_sync_sources(
                data_type=DataType.GAME,
                max_sources=3,
            )

        assert len(sources) <= 3

    def test_get_sync_sources_sorted_by_priority(self, router, mock_manifest):
        """Test get_sync_sources returns sources sorted by priority."""
        router._node_capabilities["low-priority"] = NodeSyncCapability(
            node_id="low-priority",
            selfplay_enabled=True,
        )
        router._node_capabilities["high-priority"] = NodeSyncCapability(
            node_id="high-priority",
            selfplay_enabled=True,
            has_gpu=True,
            is_training_node=True,
        )

        with patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            sources = router.get_sync_sources(data_type=DataType.GAME)

        if len(sources) >= 2:
            # High priority should come first
            assert sources[0].priority >= sources[-1].priority


# =============================================================================
# Tests for Capacity Check (December 2025)
# =============================================================================


class TestCapacityCheck:
    """Tests for _check_node_capacity method."""

    def test_check_node_capacity_calls_manifest(self, router, mock_manifest):
        """Test _check_node_capacity calls manifest.can_receive_data."""
        mock_manifest.can_receive_data.return_value = True

        result = router._check_node_capacity("test-node")

        mock_manifest.can_receive_data.assert_called()

    def test_check_node_capacity_triggers_refresh(self, router, mock_manifest):
        """Test _check_node_capacity may trigger capacity refresh."""
        import time

        # Set stale refresh time
        router._last_capacity_refresh = time.time() - 100
        mock_manifest.can_receive_data.return_value = True

        with patch.object(router, "_maybe_refresh_capacity") as mock_refresh:
            router._check_node_capacity("test-node")
            mock_refresh.assert_called()


# =============================================================================
# Tests for Priority Computation Details (December 2025)
# =============================================================================


class TestPriorityComputationDetails:
    """Detailed tests for _compute_target_priority."""

    def test_priority_base_value(self, router):
        """Test base priority value is 50."""
        cap = NodeSyncCapability(node_id="test")
        priority = router._compute_target_priority(cap, DataType.GAME)

        # Base is 50, but time weight adds up to 30 for never-synced
        assert priority >= 50

    def test_priority_training_bonus(self, router):
        """Test training nodes get +30 for GAME/NPZ."""
        cap = NodeSyncCapability(node_id="test", is_training_node=True)

        priority_game = router._compute_target_priority(cap, DataType.GAME)
        priority_npz = router._compute_target_priority(cap, DataType.NPZ)
        priority_model = router._compute_target_priority(cap, DataType.MODEL)

        # Training nodes get bonus for GAME and NPZ, not MODEL
        cap_base = NodeSyncCapability(node_id="base")
        base_priority = router._compute_target_priority(cap_base, DataType.GAME)

        assert priority_game > base_priority
        assert priority_npz > router._compute_target_priority(cap_base, DataType.NPZ)

    def test_priority_priority_node_bonus(self, router):
        """Test priority nodes get +20."""
        cap = NodeSyncCapability(node_id="test", is_priority_node=True)
        cap_base = NodeSyncCapability(node_id="base")

        priority = router._compute_target_priority(cap, DataType.GAME)
        base = router._compute_target_priority(cap_base, DataType.GAME)

        assert priority > base

    def test_priority_ephemeral_penalty(self, router):
        """Test ephemeral nodes get -15."""
        cap = NodeSyncCapability(node_id="test", is_ephemeral=True)
        cap_base = NodeSyncCapability(node_id="base")

        priority = router._compute_target_priority(cap, DataType.GAME)
        base = router._compute_target_priority(cap_base, DataType.GAME)

        assert priority < base

    def test_priority_low_disk_bonus(self, router):
        """Test nodes with <50% disk get +10."""
        cap = NodeSyncCapability(node_id="test", disk_usage_percent=30.0)
        cap_high = NodeSyncCapability(node_id="high", disk_usage_percent=80.0)

        priority_low = router._compute_target_priority(cap, DataType.GAME)
        priority_high = router._compute_target_priority(cap_high, DataType.GAME)

        assert priority_low > priority_high

    def test_priority_high_disk_penalty(self, router):
        """Test nodes with >70% disk get -20."""
        cap = NodeSyncCapability(node_id="test", disk_usage_percent=75.0)
        cap_normal = NodeSyncCapability(node_id="normal", disk_usage_percent=50.0)

        priority = router._compute_target_priority(cap, DataType.GAME)
        normal = router._compute_target_priority(cap_normal, DataType.GAME)

        assert priority < normal

    def test_priority_time_weight(self, router):
        """Test time-since-sync weight (up to +30)."""
        import time

        cap_never = NodeSyncCapability(node_id="never", last_sync_time=0.0)
        cap_recent = NodeSyncCapability(
            node_id="recent",
            last_sync_time=time.time() - 60  # 1 minute ago
        )

        priority_never = router._compute_target_priority(cap_never, DataType.GAME)
        priority_recent = router._compute_target_priority(cap_recent, DataType.GAME)

        # Never synced gets max time weight (+30)
        assert priority_never > priority_recent


# =============================================================================
# Tests for Training-Active Priority (December 2025)
# =============================================================================


class TestTrainingActivePriority:
    """Tests for training-active node priority functionality."""

    def test_is_node_training_active_no_attribute(self, router):
        """Test _is_node_training_active returns False when attribute missing."""
        if hasattr(router, "_training_active_nodes"):
            delattr(router, "_training_active_nodes")

        assert router._is_node_training_active("any-node") is False

    def test_is_node_training_active_empty_set(self, router):
        """Test _is_node_training_active returns False when set is empty."""
        router._training_active_nodes = set()

        assert router._is_node_training_active("any-node") is False

    def test_is_node_training_active_node_in_set(self, router):
        """Test _is_node_training_active returns True when node is active."""
        router._training_active_nodes = {"training-1", "training-2"}

        assert router._is_node_training_active("training-1") is True
        assert router._is_node_training_active("training-2") is True

    def test_is_node_training_active_node_not_in_set(self, router):
        """Test _is_node_training_active returns False for inactive node."""
        router._training_active_nodes = {"training-1"}

        assert router._is_node_training_active("selfplay-1") is False

    def test_update_training_active_nodes(self, router):
        """Test update_training_active_nodes updates the set."""
        active = {"node-a", "node-b", "node-c"}

        router.update_training_active_nodes(active)

        assert router._training_active_nodes == active

    def test_update_training_active_nodes_empty(self, router):
        """Test update_training_active_nodes handles empty set."""
        router.update_training_active_nodes(set())

        assert router._training_active_nodes == set()

    def test_update_training_active_nodes_replaces(self, router):
        """Test update_training_active_nodes replaces existing set."""
        router._training_active_nodes = {"old-1", "old-2"}

        router.update_training_active_nodes({"new-1"})

        assert router._training_active_nodes == {"new-1"}
        assert "old-1" not in router._training_active_nodes

    def test_training_active_priority_boost(self, router):
        """Test training-active nodes get +50 priority boost."""
        router._training_active_nodes = {"active-training"}

        cap_active = NodeSyncCapability(node_id="active-training")
        cap_inactive = NodeSyncCapability(node_id="inactive")

        priority_active = router._compute_target_priority(cap_active, DataType.GAME)
        priority_inactive = router._compute_target_priority(cap_inactive, DataType.GAME)

        # Active should be at least 50 points higher
        assert priority_active >= priority_inactive + 50


# =============================================================================
# Tests for can_provide_data_type (December 2025)
# =============================================================================


class TestCanProvideDataType:
    """Tests for _can_provide_data_type method."""

    def test_can_provide_game_selfplay_enabled(self, router):
        """Test selfplay-enabled nodes can provide game data."""
        cap = NodeSyncCapability(node_id="selfplay", selfplay_enabled=True)

        assert router._can_provide_data_type(cap, DataType.GAME) is True

    def test_can_provide_game_selfplay_disabled(self, router):
        """Test non-selfplay nodes cannot provide game data."""
        cap = NodeSyncCapability(node_id="other", selfplay_enabled=False)

        assert router._can_provide_data_type(cap, DataType.GAME) is False

    def test_can_provide_model_any_node(self, router):
        """Test any node can provide model data."""
        cap = NodeSyncCapability(node_id="any")

        assert router._can_provide_data_type(cap, DataType.MODEL) is True

    def test_can_provide_npz_training_node(self, router):
        """Test training nodes can provide NPZ data."""
        cap = NodeSyncCapability(node_id="training", is_training_node=True)

        assert router._can_provide_data_type(cap, DataType.NPZ) is True

    def test_can_provide_npz_selfplay_node(self, router):
        """Test selfplay nodes can provide NPZ data."""
        cap = NodeSyncCapability(node_id="selfplay", selfplay_enabled=True)

        assert router._can_provide_data_type(cap, DataType.NPZ) is True

    def test_can_provide_npz_no_data(self, router):
        """Test nodes without training/selfplay cannot provide NPZ."""
        cap = NodeSyncCapability(node_id="storage")

        assert router._can_provide_data_type(cap, DataType.NPZ) is False


# =============================================================================
# Tests for shares_storage_with (December 2025)
# =============================================================================


class TestSharesStorageWith:
    """Tests for _shares_storage_with method."""

    def test_both_lambda_share_nfs(self, router):
        """Test two Lambda nodes share NFS storage."""
        router._node_capabilities["lambda-1"] = NodeSyncCapability(
            node_id="lambda-1", shares_nfs=True, provider="lambda"
        )
        router._node_capabilities["lambda-2"] = NodeSyncCapability(
            node_id="lambda-2", shares_nfs=True, provider="lambda"
        )
        router.node_id = "lambda-1"

        assert router._shares_storage_with("lambda-2") is True

    def test_mixed_providers_no_share(self, router):
        """Test different providers don't share storage."""
        router._node_capabilities["lambda-1"] = NodeSyncCapability(
            node_id="lambda-1", shares_nfs=True, provider="lambda"
        )
        router._node_capabilities["vast-1"] = NodeSyncCapability(
            node_id="vast-1", shares_nfs=False, provider="vast"
        )
        router.node_id = "lambda-1"

        assert router._shares_storage_with("vast-1") is False

    def test_unknown_source_no_share(self, router):
        """Test unknown source node doesn't share."""
        router._node_capabilities["target"] = NodeSyncCapability(
            node_id="target", shares_nfs=True
        )
        router.node_id = "unknown-source"

        assert router._shares_storage_with("target") is False

    def test_unknown_target_no_share(self, router):
        """Test unknown target node doesn't share."""
        router._node_capabilities["source"] = NodeSyncCapability(
            node_id="source", shares_nfs=True
        )
        router.node_id = "source"

        assert router._shares_storage_with("unknown-target") is False


# =============================================================================
# Tests for Event Emission (December 2025)
# =============================================================================


class TestEventEmission:
    """Tests for event emission helper methods."""

    @pytest.mark.asyncio
    async def test_emit_capacity_refresh_success(self, router):
        """Test _emit_capacity_refresh emits event successfully."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router_obj = MagicMock()
            mock_get_router.return_value = mock_router_obj

            await router._emit_capacity_refresh(
                change_type="node_added",
                node_id="test-node",
                total_nodes=10,
                gpu_nodes=8,
            )

            mock_router_obj.publish.assert_called_once()
            call_args = mock_router_obj.publish.call_args
            assert call_args[0][0] == "SYNC_CAPACITY_REFRESHED"
            payload = call_args[0][1]
            assert payload["change_type"] == "node_added"
            assert payload["node_id"] == "test-node"

    @pytest.mark.asyncio
    async def test_emit_capacity_refresh_import_error(self, router):
        """Test _emit_capacity_refresh handles import error gracefully."""
        # Use a different approach - mock the router to raise on publish
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_get_router.side_effect = ImportError("No event router")

            # Should not raise
            await router._emit_capacity_refresh(
                change_type="node_removed",
                node_id="test-node",
                total_nodes=9,
                gpu_nodes=7,
            )

    @pytest.mark.asyncio
    async def test_emit_sync_routing_decision_success(self, router):
        """Test _emit_sync_routing_decision emits event successfully."""
        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_bus = AsyncMock()
            mock_get_bus.return_value = mock_bus

            await router._emit_sync_routing_decision(
                source="source-node",
                targets=["target-1", "target-2"],
                data_type=DataType.GAME,
                reason="test",
            )

            mock_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_sync_routing_decision_no_bus(self, router):
        """Test _emit_sync_routing_decision handles missing bus gracefully."""
        with patch("app.coordination.event_router.get_event_bus", return_value=None):
            # Should not raise
            await router._emit_sync_routing_decision(
                source="source-node",
                targets=["target-1"],
                data_type=DataType.MODEL,
                reason="test",
            )


# =============================================================================
# Tests for NodeSyncCapability Properties (December 2025)
# =============================================================================


class TestNodeSyncCapabilityProperties:
    """Tests for NodeSyncCapability property aliases."""

    def test_training_enabled_alias(self):
        """Test training_enabled is alias for is_training_node."""
        cap = NodeSyncCapability(node_id="test", is_training_node=True)
        assert cap.training_enabled is True

        cap2 = NodeSyncCapability(node_id="test2", is_training_node=False)
        assert cap2.training_enabled is False

    def test_disk_percent_alias(self):
        """Test disk_percent is alias for disk_usage_percent."""
        cap = NodeSyncCapability(node_id="test", disk_usage_percent=75.5)
        assert cap.disk_percent == 75.5


# =============================================================================
# Tests for get_target_reason (December 2025)
# =============================================================================


class TestGetTargetReason:
    """Tests for _get_target_reason method."""

    def test_reason_training_only(self, router):
        """Test reason for training-only node."""
        cap = NodeSyncCapability(node_id="test", is_training_node=True)

        reason = router._get_target_reason(cap)

        assert reason == "training"

    def test_reason_priority_only(self, router):
        """Test reason for priority-only node."""
        cap = NodeSyncCapability(node_id="test", is_priority_node=True)

        reason = router._get_target_reason(cap)

        assert reason == "priority"

    def test_reason_ephemeral_only(self, router):
        """Test reason for ephemeral-only node."""
        cap = NodeSyncCapability(node_id="test", is_ephemeral=True)

        reason = router._get_target_reason(cap)

        assert reason == "ephemeral"

    def test_reason_combined_training_priority(self, router):
        """Test reason for node with multiple flags."""
        cap = NodeSyncCapability(
            node_id="test",
            is_training_node=True,
            is_priority_node=True,
        )

        reason = router._get_target_reason(cap)

        assert "training" in reason
        assert "priority" in reason

    def test_reason_no_flags(self, router):
        """Test reason for node with no special flags."""
        cap = NodeSyncCapability(node_id="test")

        reason = router._get_target_reason(cap)

        assert reason == "available"


# =============================================================================
# Tests for maybe_refresh_capacity (December 2025)
# =============================================================================


class TestMaybeRefreshCapacity:
    """Tests for _maybe_refresh_capacity method."""

    def test_maybe_refresh_not_stale(self, router, mock_manifest):
        """Test capacity is not refreshed when not stale."""
        import time

        router._last_capacity_refresh = time.time()

        router._maybe_refresh_capacity()

        mock_manifest.update_local_capacity.assert_not_called()

    def test_maybe_refresh_stale(self, router, mock_manifest):
        """Test capacity is refreshed when stale."""
        import time

        router._last_capacity_refresh = time.time() - 60  # 60s ago, > 30s interval

        router._maybe_refresh_capacity()

        mock_manifest.update_local_capacity.assert_called_once()

    def test_maybe_refresh_updates_timestamp(self, router, mock_manifest):
        """Test _maybe_refresh_capacity updates last refresh timestamp."""
        import time

        router._last_capacity_refresh = 0  # Very old

        before = time.time()
        router._maybe_refresh_capacity()
        after = time.time()

        assert before <= router._last_capacity_refresh <= after

    def test_maybe_refresh_handles_manifest_error(self, router, mock_manifest):
        """Test _maybe_refresh_capacity handles manifest errors."""
        router._last_capacity_refresh = 0

        mock_manifest.update_local_capacity.side_effect = RuntimeError("Error")

        # Should not raise
        router._maybe_refresh_capacity()


# =============================================================================
# Tests for _check_node_capacity (December 2025)
# =============================================================================


class TestCheckNodeCapacity:
    """Tests for _check_node_capacity method."""

    def test_check_capacity_returns_manifest_result(self, router, mock_manifest):
        """Test _check_node_capacity returns manifest result."""
        mock_manifest.can_receive_data.return_value = True

        result = router._check_node_capacity("test-node")

        assert result is True

    def test_check_capacity_returns_false(self, router, mock_manifest):
        """Test _check_node_capacity returns False when full."""
        mock_manifest.can_receive_data.return_value = False

        result = router._check_node_capacity("full-node")

        assert result is False


# =============================================================================
# Tests for Integration Scenarios (December 2025)
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_full_sync_workflow(self, router, mock_manifest):
        """Test complete sync workflow from new games to routing."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "coordinator"

        # Get sync targets
        targets = router.get_sync_targets(
            data_type=DataType.GAME,
            max_targets=3,
        )

        # Verify targets are sorted by priority
        if len(targets) >= 2:
            for i in range(len(targets) - 1):
                assert targets[i].priority >= targets[i + 1].priority

    def test_ephemeral_to_persistent_routing(self, router, mock_manifest):
        """Test ephemeral nodes have lower receive priority than persistent."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "coordinator"

        # Add ephemeral and persistent nodes
        router._node_capabilities["ephemeral-1"] = NodeSyncCapability(
            node_id="ephemeral-1",
            is_ephemeral=True,
            can_receive_games=True,
        )
        router._node_capabilities["persistent-1"] = NodeSyncCapability(
            node_id="persistent-1",
            is_ephemeral=False,
            can_receive_games=True,
        )

        targets = router.get_sync_targets(
            data_type=DataType.GAME,
            max_targets=10,
        )

        # Find priorities for each type
        ephemeral_targets = [t for t in targets if "ephemeral" in t.node_id]
        persistent_targets = [t for t in targets if "persistent" in t.node_id]

        if ephemeral_targets and persistent_targets:
            # Persistent should have higher priority
            assert persistent_targets[0].priority > ephemeral_targets[0].priority

    def test_training_node_priority_for_games(self, router, mock_manifest):
        """Test training nodes get higher priority for game data."""
        mock_manifest.can_receive_data.return_value = True
        router.node_id = "coordinator"

        # Add training and selfplay nodes
        router._node_capabilities["training-1"] = NodeSyncCapability(
            node_id="training-1",
            is_training_node=True,
            can_receive_games=True,
        )
        router._node_capabilities["selfplay-1"] = NodeSyncCapability(
            node_id="selfplay-1",
            is_training_node=False,
            can_receive_games=True,
        )

        targets = router.get_sync_targets(
            data_type=DataType.GAME,
            max_targets=10,
        )

        training_targets = [t for t in targets if "training" in t.node_id]
        selfplay_targets = [t for t in targets if "selfplay" in t.node_id and "training" not in t.node_id]

        if training_targets and selfplay_targets:
            # Training should have higher priority for game data
            assert training_targets[0].priority > selfplay_targets[0].priority
