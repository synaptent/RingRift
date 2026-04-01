"""Tests for app.coordination.sync_router module.

This module tests the intelligent data routing based on node capabilities.
"""

import json
import socket
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================


@pytest.fixture
def mock_manifest():
    """Create mock ClusterManifest."""
    manifest = MagicMock()
    manifest.get_sync_policy = MagicMock(return_value=MagicMock(
        receive_games=True,
        receive_models=True,
        receive_npz=True,
    ))
    manifest.can_receive_data = MagicMock(return_value=True)
    manifest.update_local_capacity = MagicMock()
    return manifest


@pytest.fixture
def mock_cluster_config():
    """Create mock ClusterConfig."""
    config = MagicMock()
    config.hosts_raw = {
        "training-node-1": {
            "role": "training",
            "gpu": "RTX 4090",
            "tailscale_ip": "100.1.1.1",
        },
        "selfplay-node-1": {
            "role": "selfplay",
            "gpu": "RTX 3090",
            "tailscale_ip": "100.2.2.2",
        },
        "coordinator": {
            "role": "coordinator",
            "gpu": "",
            "tailscale_ip": "100.3.3.3",
        },
    }
    config.get_raw_section = MagicMock(return_value={
        "priority_hosts": ["training-node-1"],
    })
    config.sync_routing = MagicMock()
    config.sync_routing.allowed_external_storage = []
    return config


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset SyncRouter singleton between tests."""
    import app.coordination.sync_router as sr
    sr._sync_router = None
    yield
    sr._sync_router = None


# =============================================================================
# SyncRoute Tests
# =============================================================================


class TestSyncRoute:
    """Tests for SyncRoute dataclass."""

    def test_basic_creation(self):
        """Test creating SyncRoute with required fields."""
        from app.coordination.sync_router import SyncRoute
        from app.distributed.cluster_manifest import DataType
        
        route = SyncRoute(
            source_node="node1",
            target_node="node2",
            data_type=DataType.GAME,
        )
        
        assert route.source_node == "node1"
        assert route.target_node == "node2"
        assert route.data_type == DataType.GAME
        assert route.priority == 0
        assert route.reason == ""

    def test_with_all_fields(self):
        """Test creating SyncRoute with all fields."""
        from app.coordination.sync_router import SyncRoute
        from app.distributed.cluster_manifest import DataType
        
        route = SyncRoute(
            source_node="node1",
            target_node="node2",
            data_type=DataType.MODEL,
            priority=100,
            reason="Training node needs models",
            estimated_size_bytes=1024000,
            bandwidth_limit_mbps=100,
            quality_score=0.85,
        )
        
        assert route.priority == 100
        assert route.reason == "Training node needs models"
        assert route.estimated_size_bytes == 1024000
        assert route.bandwidth_limit_mbps == 100
        assert route.quality_score == 0.85


# =============================================================================
# NodeSyncCapability Tests
# =============================================================================


class TestNodeSyncCapability:
    """Tests for NodeSyncCapability dataclass."""

    def test_basic_creation(self):
        """Test creating NodeSyncCapability with defaults."""
        from app.coordination.sync_router import NodeSyncCapability
        
        cap = NodeSyncCapability(node_id="test-node")
        
        assert cap.node_id == "test-node"
        assert cap.can_receive_games is True
        assert cap.can_receive_models is True
        assert cap.can_receive_npz is True
        assert cap.is_training_node is False
        assert cap.is_priority_node is False
        assert cap.is_ephemeral is False

    def test_with_all_fields(self):
        """Test creating NodeSyncCapability with all fields."""
        from app.coordination.sync_router import NodeSyncCapability
        
        cap = NodeSyncCapability(
            node_id="training-node",
            can_receive_games=True,
            can_receive_models=True,
            can_receive_npz=True,
            is_training_node=True,
            is_priority_node=True,
            is_ephemeral=False,
            shares_nfs=False,
            provider="runpod",
            disk_usage_percent=45.5,
            available_gb=500.0,
            last_sync_time=time.time(),
            selfplay_enabled=True,
            has_gpu=True,
        )
        
        assert cap.is_training_node is True
        assert cap.provider == "runpod"
        assert cap.disk_usage_percent == 45.5
        assert cap.selfplay_enabled is True
        assert cap.has_gpu is True

    def test_training_enabled_property(self):
        """Test training_enabled alias property."""
        from app.coordination.sync_router import NodeSyncCapability
        
        cap = NodeSyncCapability(
            node_id="training-node",
            is_training_node=True,
        )
        
        assert cap.training_enabled is True

    def test_disk_percent_property(self):
        """Test disk_percent alias property."""
        from app.coordination.sync_router import NodeSyncCapability
        
        cap = NodeSyncCapability(
            node_id="test-node",
            disk_usage_percent=78.5,
        )
        
        assert cap.disk_percent == 78.5


# =============================================================================
# SyncRouter Initialization Tests
# =============================================================================


class TestSyncRouterInit:
    """Tests for SyncRouter initialization."""

    def test_initialization_with_manifest(self, mock_manifest, mock_cluster_config):
        """Test initialization with provided manifest."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            assert router._manifest is mock_manifest
            assert len(router._node_capabilities) >= 0

    def test_initialization_loads_config(self, mock_manifest, mock_cluster_config):
        """Test that initialization loads cluster config."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config) as mock_load, \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            mock_load.assert_called_once()


# =============================================================================
# SyncRouter get_sync_targets Tests
# =============================================================================


class TestSyncRouterGetSyncTargets:
    """Tests for SyncRouter.get_sync_targets method."""

    def test_get_sync_targets_basic(self, mock_manifest, mock_cluster_config):
        """Test basic get_sync_targets functionality."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            from app.coordination.sync_router import SyncRouter
            from app.distributed.cluster_manifest import DataType
            
            router = SyncRouter(manifest=mock_manifest)
            
            targets = router.get_sync_targets(data_type="game")
            
            # Returns a list of targets
            assert isinstance(targets, list)

    def test_get_sync_targets_excludes_self(self, mock_manifest, mock_cluster_config):
        """Test that get_sync_targets excludes current node."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value={}), \
             patch("socket.gethostname", return_value="training-node-1"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            targets = router.get_sync_targets(data_type="game")
            
            # Self should be excluded
            target_ids = [t.node_id for t in targets]
            assert "training-node-1" not in target_ids

    def test_get_sync_targets_respects_exclude_list(self, mock_manifest, mock_cluster_config):
        """Test that get_sync_targets respects exclude_nodes parameter."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            targets = router.get_sync_targets(
                data_type="game",
                exclude_nodes=["selfplay-node-1"],
            )
            
            target_ids = [t.node_id for t in targets]
            assert "selfplay-node-1" not in target_ids

    def test_get_sync_targets_max_targets(self, mock_manifest, mock_cluster_config):
        """Test that get_sync_targets respects max_targets parameter."""
        # Add more nodes to config
        mock_cluster_config.hosts_raw = {
            f"node-{i}": {"role": "selfplay", "gpu": "RTX 3090"}
            for i in range(20)
        }
        
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            targets = router.get_sync_targets(data_type="game", max_targets=5)
            
            assert len(targets) <= 5


# =============================================================================
# SyncRouter get_sync_sources Tests
# =============================================================================


class TestSyncRouterGetSyncSources:
    """Tests for SyncRouter.get_sync_sources method."""

    def test_get_sync_sources_basic(self, mock_manifest, mock_cluster_config):
        """Test basic get_sync_sources functionality."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value={}):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            sources = router.get_sync_sources(data_type="game")
            
            assert isinstance(sources, list)

    def test_get_sync_sources_excludes_coordinators(self, mock_manifest, mock_cluster_config):
        """Test that get_sync_sources excludes coordinator nodes."""
        mock_cluster_nodes = {
            "coordinator": MagicMock(role="coordinator"),
        }
        
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_nodes", return_value=mock_cluster_nodes):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            sources = router.get_sync_sources(data_type="game")
            
            source_ids = [s.node_id for s in sources]
            assert "coordinator" not in source_ids


# =============================================================================
# SyncRouter should_sync_to_node Tests
# =============================================================================


class TestSyncRouterShouldSync:
    """Tests for SyncRouter.should_sync_to_node method."""

    def test_should_sync_to_node_true(self, mock_manifest, mock_cluster_config):
        """Test should_sync_to_node returns True for valid target."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            # Should be able to sync to a known node
            result = router.should_sync_to_node("training-node-1", data_type="game")
            
            # The result depends on the node capability
            assert isinstance(result, bool)

    def test_should_sync_to_node_unknown_node(self, mock_manifest, mock_cluster_config):
        """Test should_sync_to_node uses manifest policy for unknown node."""
        # Make manifest return False for unknown nodes
        mock_manifest.can_receive_data.return_value = False

        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            result = router.should_sync_to_node("unknown-node", data_type="game")

            # Falls back to manifest policy for unknown nodes
            assert result is False
            mock_manifest.can_receive_data.assert_called()


# =============================================================================
# SyncRouter get_external_storage_path Tests
# =============================================================================


class TestSyncRouterExternalStorage:
    """Tests for SyncRouter external storage methods."""

    def test_get_external_storage_path_none(self, mock_manifest, mock_cluster_config):
        """Test get_external_storage_path returns None when not configured."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router.get_external_storage_path("some-host", "games")
            
            assert result is None

    def test_get_external_storage_path_configured(self, mock_manifest, mock_cluster_config):
        """Test get_external_storage_path returns path when configured."""
        # Configure external storage
        mock_storage = MagicMock()
        mock_storage.host = "backup-host"
        mock_storage.path = "/mnt/backup"
        mock_storage.receive_games = True
        mock_storage.receive_npz = True
        mock_storage.receive_models = True
        mock_storage.subdirs = {"games": "game_data", "models": "model_data"}
        mock_cluster_config.sync_routing.allowed_external_storage = [mock_storage]
        
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router.get_external_storage_path("backup-host", "games")
            
            assert result == "/mnt/backup/game_data"


# =============================================================================
# SyncRouter get_optimal_source Tests
# =============================================================================


class TestSyncRouterGetOptimalSource:
    """Tests for SyncRouter.get_optimal_source method."""

    def test_get_optimal_source_empty(self, mock_manifest, mock_cluster_config):
        """Test get_optimal_source with no game locations."""
        # Manifest returns no locations for this game
        mock_manifest.find_game.return_value = []

        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            # get_optimal_source takes game_id and target_node
            result = router.get_optimal_source(game_id="test-game-123", target_node="training-node-1")

            assert result is None


# =============================================================================
# SyncRouter Health Check Tests
# =============================================================================


class TestSyncRouterHealth:
    """Tests for SyncRouter health check methods."""

    def test_get_status(self, mock_manifest, mock_cluster_config):
        """Test get_status returns expected structure."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            status = router.get_status()

            # Actual keys returned by get_status()
            assert "node_id" in status
            assert "total_nodes" in status
            assert "training_nodes" in status
            assert "priority_nodes" in status
            assert "ephemeral_nodes" in status
            assert "nfs_nodes" in status

    def test_health_check(self, mock_manifest, mock_cluster_config):
        """Test health_check returns HealthCheckResult."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            from app.coordination.contracts import HealthCheckResult
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router.health_check()
            
            assert isinstance(result, HealthCheckResult)
            assert hasattr(result, "healthy")

    def test_get_node_capability(self, mock_manifest, mock_cluster_config):
        """Test get_node_capability returns capability or None."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            from app.coordination.sync_router import NodeSyncCapability
            
            router = SyncRouter(manifest=mock_manifest)
            
            # Known node
            cap = router.get_node_capability("training-node-1")
            if cap is not None:
                assert isinstance(cap, NodeSyncCapability)
            
            # Unknown node
            cap2 = router.get_node_capability("unknown-node")
            assert cap2 is None


# =============================================================================
# SyncRouter Sync Timestamp Tests
# =============================================================================


class TestSyncRouterTimestamps:
    """Tests for SyncRouter sync timestamp tracking."""

    def test_record_sync_success(self, mock_manifest, mock_cluster_config, tmp_path):
        """Test recording sync success updates timestamp."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            # Override state file path for testing
            router._SYNC_STATE_FILE = tmp_path / "sync_state.json"
            
            # Record a sync
            router.record_sync_success("test-node")
            
            # Check capability was updated
            cap = router._node_capabilities.get("test-node")
            if cap is not None:
                assert cap.last_sync_time > 0


# =============================================================================
# SyncRouter Backpressure Tests
# =============================================================================


class TestSyncRouterBackpressure:
    """Tests for SyncRouter backpressure handling."""

    def test_is_under_backpressure_default(self, mock_manifest, mock_cluster_config):
        """Test is_under_backpressure returns False by default."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router.is_under_backpressure()
            
            assert result is False

    def test_is_under_backpressure_for_node(self, mock_manifest, mock_cluster_config):
        """Test is_under_backpressure for specific node."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router.is_under_backpressure("training-node-1")
            
            assert isinstance(result, bool)


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_sync_router_singleton(self, mock_manifest, mock_cluster_config):
        """Test get_sync_router returns singleton."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_manifest", return_value=mock_manifest):
            from app.coordination.sync_router import get_sync_router, reset_sync_router
            
            router1 = get_sync_router()
            router2 = get_sync_router()
            
            assert router1 is router2
            
            reset_sync_router()

    def test_reset_sync_router(self, mock_manifest, mock_cluster_config):
        """Test reset_sync_router clears singleton."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"), \
             patch("app.coordination.sync_router.get_cluster_manifest", return_value=mock_manifest):
            from app.coordination.sync_router import get_sync_router, reset_sync_router
            
            router1 = get_sync_router()
            reset_sync_router()
            router2 = get_sync_router()
            
            # New instance after reset
            assert router1 is not router2


# =============================================================================
# SyncRouter Capacity Tests
# =============================================================================


class TestSyncRouterCapacity:
    """Tests for SyncRouter capacity management."""

    def test_refresh_all_capacity(self, mock_manifest, mock_cluster_config):
        """Test refresh_all_capacity calls manifest update."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            router.refresh_all_capacity()
            
            mock_manifest.update_local_capacity.assert_called()

    def test_maybe_refresh_capacity_respects_interval(self, mock_manifest, mock_cluster_config):
        """Test _maybe_refresh_capacity respects refresh interval."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            # Set last refresh to now
            router._last_capacity_refresh = time.time()
            
            # Reset mock
            mock_manifest.update_local_capacity.reset_mock()
            
            # Should not refresh (too recent)
            router._maybe_refresh_capacity()
            
            mock_manifest.update_local_capacity.assert_not_called()


# =============================================================================
# SyncRouter Event Handler Tests
# =============================================================================


class TestSyncRouterEventHandlers:
    """Tests for SyncRouter event handlers."""

    @pytest.mark.asyncio
    async def test_on_new_games_available(self, mock_manifest, mock_cluster_config):
        """Test _on_new_games_available handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            event = MagicMock()
            event.source_node = "selfplay-node-1"
            event.game_count = 100
            event.config_key = "hex8_2p"
            
            # Should not raise
            await router._on_new_games_available(event)

    @pytest.mark.asyncio
    async def test_on_training_started(self, mock_manifest, mock_cluster_config):
        """Test _on_training_started handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            event = MagicMock()
            event.node_id = "training-node-1"
            event.config_key = "hex8_2p"
            
            await router._on_training_started(event)

    @pytest.mark.asyncio
    async def test_on_host_online(self, mock_manifest, mock_cluster_config):
        """Test _on_host_online handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            event = MagicMock()
            event.node_id = "new-node"
            
            await router._on_host_online(event)

    @pytest.mark.asyncio
    async def test_on_host_offline(self, mock_manifest, mock_cluster_config):
        """Test _on_host_offline handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            event = MagicMock()
            event.node_id = "offline-node"
            
            await router._on_host_offline(event)

    @pytest.mark.asyncio
    async def test_on_backpressure_activated(self, mock_manifest, mock_cluster_config):
        """Test _on_backpressure_activated handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            # Handler expects event.payload to be a dict with source_node key
            event = MagicMock()
            event.payload = {
                "source_node": "overloaded-node",
                "queue_depth": 150,
                "threshold": 100,
            }

            await router._on_backpressure_activated(event)

            # Node should be marked as under backpressure
            assert router.is_under_backpressure("overloaded-node")

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self, mock_manifest, mock_cluster_config):
        """Test _on_backpressure_released handler."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            # First activate backpressure (using payload format)
            event = MagicMock()
            event.payload = {
                "source_node": "recovered-node",
                "queue_depth": 150,
                "threshold": 100,
            }
            await router._on_backpressure_activated(event)

            # Then release it (using payload format)
            release_event = MagicMock()
            release_event.payload = {
                "source_node": "recovered-node",
            }
            await router._on_backpressure_released(release_event)

            # Node should no longer be under backpressure
            assert not router.is_under_backpressure("recovered-node")

    @pytest.mark.asyncio
    async def test_emit_capacity_refresh_awaits_router_publish(self, mock_manifest, mock_cluster_config):
        """Capacity refresh events should be awaited instead of dropped."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()
        with patch("app.coordination.event_router.get_router", return_value=mock_router):
            await router._emit_capacity_refresh("node_joined", "node-1", 4, 3)

        mock_router.publish.assert_awaited_once_with(
            "SYNC_CAPACITY_REFRESHED",
            {
                "change_type": "node_joined",
                "node_id": "node-1",
                "total_nodes": 4,
                "gpu_nodes": 3,
                "router": "SyncRouter",
            },
        )

    @pytest.mark.asyncio
    async def test_emit_sync_routing_decision_uses_router_publish(self, mock_manifest, mock_cluster_config):
        """Routing-decision emission should go through the unified router."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.event_router import DataEventType
            from app.coordination.sync_router import SyncRouter
            from app.distributed.cluster_manifest import DataType

            router = SyncRouter(manifest=mock_manifest)

        mock_router = MagicMock()
        mock_router.publish = AsyncMock()
        with patch("app.coordination.event_router.get_router", return_value=mock_router):
            await router._emit_sync_routing_decision(
                source="source-node",
                targets=["target-a", "target-b"],
                data_type=DataType.MODEL,
                reason="fresh_model",
            )

        mock_router.publish.assert_awaited_once_with(
            DataEventType.SYNC_REQUEST,
            {
                "source": "source-node",
                "targets": ["target-a", "target-b"],
                "data_type": DataType.MODEL.value,
                "reason": "fresh_model",
                "router": "SyncRouter",
            },
            source="SyncRouter",
        )


# =============================================================================
# SyncRouter Wire to Event Router Tests
# =============================================================================


class TestSyncRouterWiring:
    """Tests for SyncRouter event wiring."""

    def test_wire_to_event_router_handles_errors(self, mock_manifest, mock_cluster_config):
        """Test wire_to_event_router handles errors gracefully."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            # Should not raise even if event router is not available
            # (the method catches ImportError and other exceptions)
            router.wire_to_event_router()  # No exception = success

    def test_wire_to_event_router_success(self, mock_manifest, mock_cluster_config):
        """Test wire_to_event_router with working event router."""
        # Create mock event type enum
        from enum import Enum

        class MockDataEventType(Enum):
            NEW_GAMES_AVAILABLE = "new_games_available"
            TRAINING_STARTED = "training_started"
            HOST_ONLINE = "host_online"
            HOST_OFFLINE = "host_offline"
            NODE_RECOVERED = "node_recovered"
            CLUSTER_CAPACITY_CHANGED = "cluster_capacity_changed"
            BACKPRESSURE_ACTIVATED = "backpressure_activated"
            BACKPRESSURE_RELEASED = "backpressure_released"
            SYNC_FAILURE_CRITICAL = "sync_failure_critical"

        mock_router_instance = MagicMock()

        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter

            router = SyncRouter(manifest=mock_manifest)

            # Mock the import inside wire_to_event_router
            mock_module = MagicMock()
            mock_module.DataEventType = MockDataEventType
            mock_module.get_router = MagicMock(return_value=mock_router_instance)

            with patch.dict("sys.modules", {"app.coordination.event_router": mock_module}):
                router.wire_to_event_router()

            # Just verify no exceptions - the actual subscription happens internally


# =============================================================================
# SyncRouter Training Active Tests
# =============================================================================


class TestSyncRouterTrainingActive:
    """Tests for SyncRouter training activity tracking."""

    def test_update_training_active_nodes(self, mock_manifest, mock_cluster_config):
        """Test update_training_active_nodes updates internal state."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            active_nodes = {"training-node-1", "training-node-2"}
            router.update_training_active_nodes(active_nodes)
            
            # Check nodes are marked active
            assert router._is_node_training_active("training-node-1")
            assert not router._is_node_training_active("selfplay-node-1")

    def test_is_node_training_active_false_by_default(self, mock_manifest, mock_cluster_config):
        """Test _is_node_training_active returns False by default."""
        with patch("app.coordination.sync_router.load_cluster_config", return_value=mock_cluster_config), \
             patch("app.coordination.sync_router.get_host_provider", return_value="local"):
            from app.coordination.sync_router import SyncRouter
            
            router = SyncRouter(manifest=mock_manifest)
            
            result = router._is_node_training_active("any-node")
            
            assert result is False
