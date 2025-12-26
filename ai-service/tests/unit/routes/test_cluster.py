"""Tests for app.routes.cluster - Cluster Monitoring API Routes.

This module tests the FastAPI cluster endpoints including:
- GET /api/cluster/status - Cluster health summary
- GET /api/cluster/sync/status - Sync daemon status
- GET /api/cluster/manifest - Manifest inspection
- POST /api/cluster/sync/trigger - Manual sync trigger
- GET /api/cluster/nodes - Node inventory list
- GET /api/cluster/nodes/{node_id} - Specific node details
- GET /api/cluster/health - Quick health check
- GET /api/cluster/config - Cluster configuration
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from app.routes.cluster import (
    ClusterStatusResponse,
    ManifestSummaryResponse,
    NodeInventoryResponse,
    SyncStatusResponse,
    SyncTriggerRequest,
    SyncTriggerResponse,
    router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_p2p_status():
    """Create mock P2P status response."""
    return {
        "node_id": "test-node",
        "is_leader": True,
        "leader_id": "test-node",
        "uptime": 3600.0,
        "job_count": 5,
        "peers": [
            {"id": "peer-1", "alive": True, "last_seen": 1234567890},
            {"id": "peer-2", "alive": True, "last_seen": 1234567891},
            {"id": "peer-3", "alive": False, "last_seen": 1234567850},
        ],
    }


@pytest.fixture
def mock_manifest():
    """Create mock ClusterManifest."""
    manifest = MagicMock()
    manifest.node_id = "test-node"
    manifest.get_cluster_stats.return_value = {
        "total_games": 10000,
        "total_models": 12,
        "total_npz_files": 50,
        "games_by_config": {
            "hex8_2p": 5000,
            "square8_2p": 3000,
            "square8_4p": 2000,
        },
        "under_replicated_count": 5,
        "nodes_with_data": 3,
    }
    manifest.get_all_node_ids.return_value = ["test-node", "peer-1", "peer-2"]

    # Create mock node inventory
    inventory = MagicMock()
    inventory.game_count = 5000
    inventory.model_count = 12
    inventory.npz_count = 25
    inventory.capacity = MagicMock(usage_percent=45.0, free_bytes=500 * 1024**3)
    manifest.get_node_inventory.return_value = inventory

    return manifest


@pytest.fixture
def mock_sync_daemon():
    """Create mock AutoSyncDaemon."""
    daemon = MagicMock()
    daemon.get_status.return_value = {
        "node_id": "test-node",
        "running": True,
        "pending_games": 10,
        "stats": {
            "last_sync_time": 1234567890.0,
            "games_synced": 1000,
            "models_synced": 5,
            "bytes_transferred": 5 * 1024**3,
            "recent_errors": ["Error 1", "Error 2"],
        },
    }
    daemon._running = True
    return daemon


@pytest.fixture
def test_client():
    """Create a test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestClusterStatusResponse:
    """Tests for ClusterStatusResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = ClusterStatusResponse(
            node_id="test-node",
            is_leader=True,
            alive_peers=2,
            total_peers=3,
            uptime_seconds=3600.0,
            cluster_healthy=True,
        )
        assert response.node_id == "test-node"
        assert response.is_leader is True
        assert response.alive_peers == 2
        assert response.total_peers == 3

    def test_optional_leader_id(self):
        """Should allow optional leader_id."""
        response = ClusterStatusResponse(
            node_id="test-node",
            is_leader=False,
            leader_id="other-node",
            alive_peers=2,
            total_peers=3,
            uptime_seconds=3600.0,
            cluster_healthy=True,
        )
        assert response.leader_id == "other-node"

    def test_default_job_count(self):
        """Should default job_count to 0."""
        response = ClusterStatusResponse(
            node_id="test-node",
            is_leader=True,
            alive_peers=2,
            total_peers=3,
            uptime_seconds=3600.0,
            cluster_healthy=True,
        )
        assert response.job_count == 0


class TestSyncStatusResponse:
    """Tests for SyncStatusResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = SyncStatusResponse(
            node_id="test-node",
            running=True,
        )
        assert response.node_id == "test-node"
        assert response.running is True

    def test_default_values(self):
        """Should have correct default values."""
        response = SyncStatusResponse(
            node_id="test-node",
            running=True,
        )
        assert response.last_sync_time == 0
        assert response.pending_syncs == 0
        assert response.games_synced == 0
        assert response.models_synced == 0
        assert response.bytes_transferred == 0
        assert response.errors == []


class TestManifestSummaryResponse:
    """Tests for ManifestSummaryResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = ManifestSummaryResponse(
            node_id="test-node",
            total_games=10000,
            total_models=12,
            total_npz_files=50,
        )
        assert response.node_id == "test-node"
        assert response.total_games == 10000
        assert response.total_models == 12
        assert response.total_npz_files == 50

    def test_default_dicts(self):
        """Should have correct default values."""
        response = ManifestSummaryResponse(
            node_id="test-node",
            total_games=10000,
            total_models=12,
            total_npz_files=50,
        )
        assert response.games_by_config == {}
        assert response.under_replicated_count == 0
        assert response.nodes_with_data == 0


class TestNodeInventoryResponse:
    """Tests for NodeInventoryResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = NodeInventoryResponse(
            node_id="test-node",
        )
        assert response.node_id == "test-node"

    def test_default_values(self):
        """Should have correct default values."""
        response = NodeInventoryResponse(
            node_id="test-node",
        )
        assert response.game_count == 0
        assert response.model_count == 0
        assert response.npz_count == 0
        assert response.disk_usage_percent is None
        assert response.disk_free_gb is None
        assert response.is_training is False
        assert response.is_selfplaying is False
        assert response.last_seen == 0


class TestSyncTriggerRequest:
    """Tests for SyncTriggerRequest model."""

    def test_optional_filters(self):
        """Should allow optional filters."""
        request = SyncTriggerRequest(
            board_type="hex8",
            num_players=2,
            priority=True,
        )
        assert request.board_type == "hex8"
        assert request.num_players == 2
        assert request.priority is True

    def test_default_priority(self):
        """Should default priority to False."""
        request = SyncTriggerRequest()
        assert request.priority is False


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestGetClusterStatusEndpoint:
    """Tests for GET /api/cluster/status endpoint."""

    def test_get_status_healthy_cluster(self, test_client, mock_p2p_status):
        """Should return healthy status with P2P running."""
        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/status")
            assert response.status_code == 200

            data = response.json()
            assert data["node_id"] == "test-node"
            assert data["is_leader"] is True
            assert data["leader_id"] == "test-node"
            assert data["alive_peers"] == 2
            assert data["total_peers"] == 3
            assert data["uptime_seconds"] == 3600.0
            assert data["job_count"] == 5
            assert data["cluster_healthy"] is True

    def test_get_status_p2p_down(self, test_client):
        """Should return degraded status when P2P not running."""
        with patch("app.routes.cluster._get_p2p_status", return_value=None):
            response = test_client.get("/cluster/status")
            assert response.status_code == 200

            data = response.json()
            assert data["is_leader"] is False
            assert data["leader_id"] is None
            assert data["alive_peers"] == 0
            assert data["total_peers"] == 0
            assert data["cluster_healthy"] is False

    def test_get_status_no_alive_peers(self, test_client, mock_p2p_status):
        """Should mark unhealthy when no alive peers."""
        mock_p2p_status["peers"] = [
            {"id": "peer-1", "alive": False},
            {"id": "peer-2", "alive": False},
        ]

        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/status")
            assert response.status_code == 200

            data = response.json()
            assert data["alive_peers"] == 0
            assert data["cluster_healthy"] is False

    def test_get_status_no_leader(self, test_client, mock_p2p_status):
        """Should mark unhealthy when no leader elected."""
        mock_p2p_status["leader_id"] = None

        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/status")
            assert response.status_code == 200

            data = response.json()
            assert data["cluster_healthy"] is False


class TestGetSyncStatusEndpoint:
    """Tests for GET /api/cluster/sync/status endpoint."""

    def test_get_sync_status_running(self, test_client, mock_sync_daemon):
        """Should return sync daemon status."""
        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.get("/cluster/sync/status")
            assert response.status_code == 200

            data = response.json()
            assert data["node_id"] == "test-node"
            assert data["running"] is True
            assert data["last_sync_time"] == 1234567890.0
            assert data["pending_syncs"] == 10
            assert data["games_synced"] == 1000
            assert data["models_synced"] == 5
            assert data["bytes_transferred"] == 5 * 1024**3
            assert len(data["errors"]) == 2

    def test_get_sync_status_not_running(self, test_client):
        """Should return not running when daemon unavailable."""
        with patch("app.routes.cluster._get_sync_daemon", return_value=None):
            response = test_client.get("/cluster/sync/status")
            assert response.status_code == 200

            data = response.json()
            assert data["running"] is False

    def test_get_sync_status_limits_errors(self, test_client, mock_sync_daemon):
        """Should limit recent errors to 5."""
        mock_sync_daemon.get_status.return_value["stats"]["recent_errors"] = [
            f"Error {i}" for i in range(10)
        ]

        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.get("/cluster/sync/status")
            assert response.status_code == 200

            data = response.json()
            assert len(data["errors"]) == 5


class TestGetManifestSummaryEndpoint:
    """Tests for GET /api/cluster/manifest endpoint."""

    def test_get_manifest_summary(self, test_client, mock_manifest):
        """Should return manifest summary."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            response = test_client.get("/cluster/manifest")
            assert response.status_code == 200

            data = response.json()
            assert data["node_id"] == "test-node"
            assert data["total_games"] == 10000
            assert data["total_models"] == 12
            assert data["total_npz_files"] == 50
            assert len(data["games_by_config"]) == 3
            assert data["under_replicated_count"] == 5
            assert data["nodes_with_data"] == 3

    def test_get_manifest_not_available(self, test_client):
        """Should return empty summary when manifest not available."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=None):
            response = test_client.get("/cluster/manifest")
            assert response.status_code == 200

            data = response.json()
            assert data["total_games"] == 0
            assert data["total_models"] == 0
            assert data["total_npz_files"] == 0


class TestListNodesEndpoint:
    """Tests for GET /api/cluster/nodes endpoint."""

    def test_list_nodes(self, test_client, mock_manifest, mock_p2p_status):
        """Should list all nodes with inventory."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
                response = test_client.get("/cluster/nodes")
                assert response.status_code == 200

                data = response.json()
                assert len(data) == 3
                assert data[0]["node_id"] in ["test-node", "peer-1", "peer-2"]
                assert data[0]["game_count"] == 5000
                assert data[0]["model_count"] == 12
                assert data[0]["npz_count"] == 25

    def test_list_nodes_no_manifest(self, test_client):
        """Should return empty list when manifest not available."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=None):
            response = test_client.get("/cluster/nodes")
            assert response.status_code == 200

            data = response.json()
            assert data == []

    def test_list_nodes_disk_usage(self, test_client, mock_manifest, mock_p2p_status):
        """Should include disk usage when available."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
                response = test_client.get("/cluster/nodes")
                assert response.status_code == 200

                data = response.json()
                assert data[0]["disk_usage_percent"] == 45.0
                assert abs(data[0]["disk_free_gb"] - 500.0) < 0.1

    def test_list_nodes_no_capacity(self, test_client, mock_manifest, mock_p2p_status):
        """Should handle nodes without capacity info."""
        inventory = mock_manifest.get_node_inventory.return_value
        inventory.capacity = None

        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
                response = test_client.get("/cluster/nodes")
                assert response.status_code == 200

                data = response.json()
                assert data[0]["disk_usage_percent"] is None
                assert data[0]["disk_free_gb"] is None


class TestGetNodeDetailsEndpoint:
    """Tests for GET /api/cluster/nodes/{node_id} endpoint."""

    def test_get_node_details(self, test_client, mock_manifest):
        """Should return specific node details."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            response = test_client.get("/cluster/nodes/test-node")
            assert response.status_code == 200

            data = response.json()
            assert data["node_id"] == "test-node"
            assert data["game_count"] == 5000
            assert data["model_count"] == 12
            assert data["npz_count"] == 25
            assert data["disk_usage_percent"] == 45.0

    def test_get_node_details_not_found(self, test_client, mock_manifest):
        """Should return 404 when node not found."""
        mock_manifest.get_node_inventory.side_effect = Exception("Node not found")

        with patch("app.routes.cluster._get_cluster_manifest", return_value=mock_manifest):
            response = test_client.get("/cluster/nodes/nonexistent")
            assert response.status_code == 404

    def test_get_node_details_manifest_not_available(self, test_client):
        """Should return 503 when manifest not available."""
        with patch("app.routes.cluster._get_cluster_manifest", return_value=None):
            response = test_client.get("/cluster/nodes/test-node")
            assert response.status_code == 503


class TestTriggerSyncEndpoint:
    """Tests for POST /api/cluster/sync/trigger endpoint."""

    async def test_trigger_sync_success(self, test_client, mock_sync_daemon):
        """Should trigger sync successfully."""
        # Mock the asyncio methods
        import asyncio
        mock_sync_daemon.start = MagicMock(return_value=asyncio.Future())
        mock_sync_daemon.start.return_value.set_result(None)
        mock_sync_daemon.trigger_sync = MagicMock(return_value=asyncio.Future())
        mock_sync_daemon.trigger_sync.return_value.set_result(None)

        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.post("/cluster/sync/trigger")
            assert response.status_code == 200

            data = response.json()
            assert data["triggered"] is True
            assert "successfully" in data["message"]

    async def test_trigger_sync_with_filters(self, test_client, mock_sync_daemon):
        """Should include filters in message."""
        # Mock the asyncio methods
        import asyncio
        mock_sync_daemon.start = MagicMock(return_value=asyncio.Future())
        mock_sync_daemon.start.return_value.set_result(None)
        mock_sync_daemon.trigger_sync = MagicMock(return_value=asyncio.Future())
        mock_sync_daemon.trigger_sync.return_value.set_result(None)

        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.post(
                "/cluster/sync/trigger",
                json={"board_type": "hex8", "num_players": 2, "priority": True},
            )
            assert response.status_code == 200

            data = response.json()
            assert "hex8" in data["message"]
            assert "2p" in data["message"]

    def test_trigger_sync_daemon_not_available(self, test_client):
        """Should return failure when daemon not available."""
        with patch("app.routes.cluster._get_sync_daemon", return_value=None):
            response = test_client.post("/cluster/sync/trigger")
            assert response.status_code == 200

            data = response.json()
            assert data["triggered"] is False
            assert "not available" in data["message"]

    def test_trigger_sync_daemon_error(self, test_client, mock_sync_daemon):
        """Should handle daemon errors gracefully."""
        mock_sync_daemon.trigger_sync.side_effect = RuntimeError("Sync failed")

        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.post("/cluster/sync/trigger")
            assert response.status_code == 200

            data = response.json()
            assert data["triggered"] is False
            assert "failed" in data["message"]

    def test_trigger_sync_starts_daemon(self, test_client, mock_sync_daemon):
        """Should start daemon if not running."""
        mock_sync_daemon._running = False

        with patch("app.routes.cluster._get_sync_daemon", return_value=mock_sync_daemon):
            response = test_client.post("/cluster/sync/trigger")
            assert response.status_code == 200

            mock_sync_daemon.start.assert_called_once()


class TestClusterHealthCheckEndpoint:
    """Tests for GET /api/cluster/health endpoint."""

    def test_health_check_healthy(self, test_client, mock_p2p_status):
        """Should return 200 for healthy cluster."""
        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["leader"] == "test-node"
            assert data["alive_peers"] == 2

    def test_health_check_p2p_down(self, test_client):
        """Should return 503 when P2P not running."""
        with patch("app.routes.cluster._get_p2p_status", return_value=None):
            response = test_client.get("/cluster/health")
            assert response.status_code == 503
            assert "P2P daemon not running" in response.json()["detail"]

    def test_health_check_no_peers(self, test_client, mock_p2p_status):
        """Should return 503 when no alive peers."""
        mock_p2p_status["peers"] = [
            {"id": "peer-1", "alive": False},
        ]

        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/health")
            assert response.status_code == 503
            assert "No alive peers" in response.json()["detail"]

    def test_health_check_no_leader(self, test_client, mock_p2p_status):
        """Should return 503 when no leader elected."""
        mock_p2p_status["leader_id"] = None

        with patch("app.routes.cluster._get_p2p_status", return_value=mock_p2p_status):
            response = test_client.get("/cluster/health")
            assert response.status_code == 503
            assert "No leader elected" in response.json()["detail"]


class TestGetClusterConfigEndpoint:
    """Tests for GET /api/cluster/config endpoint."""

    def test_get_config_success(self, test_client):
        """Should return cluster configuration."""
        config_data = {
            "hosts": {
                "node-1": {"gpu": "H100"},
                "node-2": {"gpu": "A100"},
            },
            "sync_routing": {
                "priority_hosts": ["node-1"],
                "max_disk_usage_percent": 70,
            },
        }

        mock_file = mock_open(read_data="")
        with patch("builtins.open", mock_file):
            with patch("yaml.safe_load", return_value=config_data):
                with patch("pathlib.Path.exists", return_value=True):
                    response = test_client.get("/cluster/config")
                    assert response.status_code == 200

                    data = response.json()
                    assert data["configured"] is True
                    assert data["host_count"] == 2
                    assert len(data["hosts"]) == 2
                    assert data["priority_hosts"] == ["node-1"]
                    assert data["max_disk_usage_percent"] == 70

    def test_get_config_not_configured(self, test_client):
        """Should handle missing config file."""
        with patch("pathlib.Path.exists", return_value=False):
            response = test_client.get("/cluster/config")
            assert response.status_code == 200

            data = response.json()
            assert data["configured"] is False
            assert data["hosts"] == []

    def test_get_config_error(self, test_client):
        """Should handle config loading errors."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=IOError("Permission denied")):
                response = test_client.get("/cluster/config")
                assert response.status_code == 200

                data = response.json()
                assert data["configured"] is False
                assert "error" in data


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_p2p_status_success(self):
        """Should fetch P2P status successfully."""
        from app.routes.cluster import _get_p2p_status

        mock_response = MagicMock()
        mock_response.read.return_value.decode.return_value = json.dumps(
            {"node_id": "test", "is_leader": True}
        )
        mock_response.__enter__ = lambda self: self
        mock_response.__exit__ = lambda self, *args: None

        with patch("urllib.request.urlopen", return_value=mock_response):
            status = _get_p2p_status()
            assert status["node_id"] == "test"
            assert status["is_leader"] is True

    def test_get_p2p_status_timeout(self):
        """Should handle timeout gracefully."""
        from app.routes.cluster import _get_p2p_status

        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            status = _get_p2p_status()
            assert status is None

    def test_get_p2p_status_connection_error(self):
        """Should handle connection errors gracefully."""
        from app.routes.cluster import _get_p2p_status

        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            status = _get_p2p_status()
            assert status is None

    def test_get_cluster_manifest_import_error(self):
        """Should handle import errors gracefully."""
        from app.routes.cluster import _get_cluster_manifest

        # Patch the import at the point where it's used in the function
        with patch("app.distributed.cluster_manifest.get_cluster_manifest", side_effect=ImportError):
            manifest = _get_cluster_manifest()
            assert manifest is None

    def test_get_sync_daemon_import_error(self):
        """Should handle import errors gracefully."""
        from app.routes.cluster import _get_sync_daemon

        # Patch the import at the point where it's used in the function
        with patch("app.coordination.auto_sync_daemon.get_auto_sync_daemon", side_effect=ImportError):
            daemon = _get_sync_daemon()
            assert daemon is None
