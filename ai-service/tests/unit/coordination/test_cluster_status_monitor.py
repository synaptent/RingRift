"""Tests for cluster_status_monitor.py - Cluster health monitoring.

December 2025 - Critical path tests for cluster status monitoring.

Tests cover:
- NodeStatus dataclass fields and defaults
- ClusterStatus dataclass and aggregation
- ClusterMonitor initialization
- Host configuration loading
- Status querying (mocked SSH)
- Async status querying
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.cluster_status_monitor import (
    NodeStatus,
    ClusterStatus,
    ClusterMonitor,
)


class TestNodeStatus:
    """Tests for NodeStatus dataclass."""

    def test_node_status_defaults(self):
        """Test that NodeStatus has correct default values."""
        status = NodeStatus(host_name="test-host")

        assert status.host_name == "test-host"
        assert status.reachable is False
        assert status.response_time_ms == 0.0
        assert status.game_counts == {}
        assert status.total_games == 0
        assert status.training_active is False
        assert status.training_processes == []
        assert status.disk_usage_percent == 0.0
        assert status.disk_free_gb == 0.0
        assert status.disk_total_gb == 0.0
        assert status.error is None
        assert status.role == "unknown"

    def test_node_status_with_values(self):
        """Test NodeStatus with custom values."""
        status = NodeStatus(
            host_name="gpu-node-1",
            reachable=True,
            response_time_ms=25.5,
            game_counts={"hex8_2p": 100, "square8_4p": 50},
            total_games=150,
            training_active=True,
            disk_usage_percent=45.0,
            disk_free_gb=100.0,
            disk_total_gb=200.0,
            gpu="H100",
            role="training",
        )

        assert status.host_name == "gpu-node-1"
        assert status.reachable is True
        assert status.response_time_ms == 25.5
        assert status.game_counts["hex8_2p"] == 100
        assert status.total_games == 150
        assert status.training_active is True
        assert status.gpu == "H100"

    def test_node_status_error_field(self):
        """Test NodeStatus error tracking."""
        status = NodeStatus(
            host_name="failed-host",
            reachable=False,
            error="SSH connection refused",
        )

        assert status.reachable is False
        assert status.error == "SSH connection refused"

    def test_node_status_gpu_metrics(self):
        """Test GPU metrics fields."""
        status = NodeStatus(
            host_name="gpu-node",
            gpu_utilization_percent=85.0,
            gpu_memory_used_gb=70.0,
            gpu_memory_total_gb=80.0,
        )

        assert status.gpu_utilization_percent == 85.0
        assert status.gpu_memory_used_gb == 70.0
        assert status.gpu_memory_total_gb == 80.0


class TestClusterStatus:
    """Tests for ClusterStatus dataclass."""

    def test_cluster_status_defaults(self):
        """Test ClusterStatus default values."""
        status = ClusterStatus()

        assert status.total_nodes == 0
        assert status.active_nodes == 0
        assert status.unreachable_nodes == 0
        assert status.total_games == 0
        assert status.games_by_config == {}
        assert status.nodes_training == 0
        assert status.total_training_processes == 0
        assert status.avg_disk_usage == 0.0
        assert status.nodes == {}
        assert status.errors == []
        assert isinstance(status.timestamp, datetime)

    def test_cluster_status_healthy_nodes_property(self):
        """Test healthy_nodes is an alias for active_nodes."""
        status = ClusterStatus(active_nodes=5)
        assert status.healthy_nodes == 5

    def test_cluster_status_with_nodes(self):
        """Test ClusterStatus with node data."""
        node1 = NodeStatus(host_name="node-1", reachable=True, total_games=100)
        node2 = NodeStatus(host_name="node-2", reachable=True, total_games=50)
        node3 = NodeStatus(host_name="node-3", reachable=False)

        status = ClusterStatus(
            total_nodes=3,
            active_nodes=2,
            unreachable_nodes=1,
            total_games=150,
            nodes={"node-1": node1, "node-2": node2, "node-3": node3},
        )

        assert status.total_nodes == 3
        assert status.active_nodes == 2
        assert status.total_games == 150
        assert len(status.nodes) == 3
        assert status.nodes["node-1"].total_games == 100

    def test_cluster_status_query_metadata(self):
        """Test ClusterStatus query metadata fields."""
        status = ClusterStatus(
            query_duration_seconds=2.5,
            errors=["node-1: timeout", "node-2: connection refused"],
        )

        assert status.query_duration_seconds == 2.5
        assert len(status.errors) == 2
        assert "timeout" in status.errors[0]


class TestClusterMonitorInit:
    """Tests for ClusterMonitor initialization."""

    def test_init_with_defaults(self):
        """Test ClusterMonitor initialization with defaults."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        assert monitor.ssh_timeout == 15
        assert monitor.parallel is True

    def test_init_with_custom_timeout(self):
        """Test ClusterMonitor with custom SSH timeout."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(ssh_timeout=30)

        assert monitor.ssh_timeout == 30

    def test_init_with_parallel_disabled(self):
        """Test ClusterMonitor with parallel queries disabled."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        assert monitor.parallel is False

    def test_init_loads_hosts(self):
        """Test that initialization loads hosts configuration."""
        mock_config = {
            "hosts": {
                "node-1": {"status": "ready", "ssh_host": "10.0.0.1"},
                "node-2": {"status": "ready", "ssh_host": "10.0.0.2"},
            }
        }

        with patch("app.coordination.cluster_status_monitor.HAS_YAML", True):
            with patch("builtins.open", MagicMock()):
                with patch(
                    "yaml.safe_load",
                    return_value=mock_config,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        # Mock the config loading
                        with patch.object(ClusterMonitor, "_load_hosts") as mock_load:
                            monitor = ClusterMonitor()
                            mock_load.assert_called_once()


class TestClusterMonitorHosts:
    """Tests for host-related methods."""

    def test_get_active_hosts(self):
        """Test get_active_hosts returns only active/ready hosts."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "node-1": {"status": "ready"},
            "node-2": {"status": "active"},
            "node-3": {"status": "offline"},
            "node-4": {"status": "maintenance"},
            "node-5": {"status": "ready"},
        }

        active = monitor.get_active_hosts()

        assert "node-1" in active
        assert "node-2" in active
        assert "node-5" in active
        assert "node-3" not in active
        assert "node-4" not in active
        assert len(active) == 3

    def test_get_active_hosts_empty(self):
        """Test get_active_hosts with no active hosts."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "node-1": {"status": "offline"},
            "node-2": {"status": "maintenance"},
        }

        active = monitor.get_active_hosts()
        assert active == []


class TestClusterMonitorNodeStatus:
    """Tests for node status querying."""

    def test_get_node_status_unreachable(self):
        """Test get_node_status for unreachable node."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "test-node": {
                "status": "ready",
                "ssh_host": "10.0.0.1",
                "role": "selfplay",
            }
        }

        # Mock connectivity check to fail
        with patch.object(monitor, "_check_connectivity", return_value=False):
            status = monitor.get_node_status("test-node")

        assert status.host_name == "test-node"
        assert status.reachable is False
        assert status.error == "Host unreachable"
        assert status.role == "selfplay"

    def test_get_node_status_reachable(self):
        """Test get_node_status for reachable node."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "test-node": {
                "status": "ready",
                "ssh_host": "10.0.0.1",
                "role": "training",
                "gpu": "A100",
            }
        }
        monitor.game_discovery = None  # Disable game discovery

        # Mock connectivity check to succeed
        with patch.object(monitor, "_check_connectivity", return_value=True):
            # Mock _run_ssh_command for various status checks
            with patch.object(
                monitor,
                "_run_ssh_command",
                return_value=(0, "50\n60.0\n200.0", ""),
            ):
                status = monitor.get_node_status(
                    "test-node",
                    include_game_counts=False,
                    include_training_status=False,
                    include_disk_usage=False,
                    include_gpu_metrics=False,
                    include_sync_status=False,
                )

        assert status.host_name == "test-node"
        assert status.reachable is True
        assert status.role == "training"
        assert status.gpu == "A100"


class TestClusterMonitorClusterStatus:
    """Tests for cluster-wide status querying."""

    def test_get_cluster_status_aggregation(self):
        """Test that get_cluster_status correctly aggregates node data."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        monitor._hosts = {
            "node-1": {"status": "ready"},
            "node-2": {"status": "ready"},
        }

        node1_status = NodeStatus(
            host_name="node-1",
            reachable=True,
            total_games=100,
            game_counts={"hex8_2p": 60, "square8_2p": 40},
            training_active=True,
            training_processes=[{"pid": 123}],
            disk_usage_percent=40.0,
            disk_free_gb=100.0,
            disk_total_gb=200.0,
        )
        node2_status = NodeStatus(
            host_name="node-2",
            reachable=True,
            total_games=50,
            game_counts={"hex8_2p": 30, "square8_4p": 20},
            training_active=False,
            disk_usage_percent=60.0,
            disk_free_gb=50.0,
            disk_total_gb=150.0,
        )

        def mock_get_node_status(host_name, **kwargs):
            if host_name == "node-1":
                return node1_status
            return node2_status

        with patch.object(monitor, "get_node_status", side_effect=mock_get_node_status):
            status = monitor.get_cluster_status()

        assert status.total_nodes == 2
        assert status.active_nodes == 2
        assert status.total_games == 150
        assert status.games_by_config["hex8_2p"] == 90
        assert status.games_by_config["square8_2p"] == 40
        assert status.games_by_config["square8_4p"] == 20
        assert status.nodes_training == 1
        assert status.total_training_processes == 1
        assert status.avg_disk_usage == 50.0  # (40 + 60) / 2
        assert status.total_disk_free_gb == 150.0

    def test_get_cluster_status_with_unreachable_nodes(self):
        """Test cluster status with some unreachable nodes."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        monitor._hosts = {
            "node-1": {"status": "ready"},
            "node-2": {"status": "ready"},
            "node-3": {"status": "ready"},
        }

        def mock_get_node_status(host_name, **kwargs):
            if host_name == "node-2":
                return NodeStatus(
                    host_name="node-2",
                    reachable=False,
                    error="Connection refused",
                )
            return NodeStatus(
                host_name=host_name,
                reachable=True,
                total_games=50,
            )

        with patch.object(monitor, "get_node_status", side_effect=mock_get_node_status):
            status = monitor.get_cluster_status()

        assert status.total_nodes == 3
        assert status.active_nodes == 2
        assert status.unreachable_nodes == 1
        assert "Connection refused" in status.errors[0]

    def test_get_cluster_status_custom_hosts(self):
        """Test cluster status with custom host list."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        monitor._hosts = {
            "node-1": {"status": "ready"},
            "node-2": {"status": "ready"},
            "node-3": {"status": "ready"},
        }

        def mock_get_node_status(host_name, **kwargs):
            return NodeStatus(host_name=host_name, reachable=True)

        with patch.object(monitor, "get_node_status", side_effect=mock_get_node_status):
            status = monitor.get_cluster_status(hosts=["node-1", "node-3"])

        assert status.total_nodes == 2
        assert "node-1" in status.nodes
        assert "node-3" in status.nodes
        assert "node-2" not in status.nodes


class TestClusterMonitorAsync:
    """Tests for async status methods."""

    @pytest.mark.asyncio
    async def test_get_node_status_async(self):
        """Test async node status retrieval."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "test-node": {"status": "ready", "role": "training"}
        }

        expected_status = NodeStatus(
            host_name="test-node",
            reachable=True,
            role="training",
        )

        with patch.object(monitor, "get_node_status", return_value=expected_status):
            status = await monitor.get_node_status_async("test-node")

        assert status.host_name == "test-node"
        assert status.reachable is True

    @pytest.mark.asyncio
    async def test_get_cluster_status_async(self):
        """Test async cluster status retrieval."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        monitor._hosts = {
            "node-1": {"status": "ready"},
            "node-2": {"status": "ready"},
        }

        async def mock_get_node_status_async(host_name, **kwargs):
            return NodeStatus(
                host_name=host_name,
                reachable=True,
                total_games=50,
            )

        with patch.object(
            monitor,
            "get_node_status_async",
            side_effect=mock_get_node_status_async,
        ):
            status = await monitor.get_cluster_status_async()

        assert status.total_nodes == 2
        assert status.active_nodes == 2
        assert status.total_games == 100

    @pytest.mark.asyncio
    async def test_async_run_ssh_command_timeout(self):
        """Test async SSH command with timeout."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(ssh_timeout=1)

        monitor._hosts = {
            "slow-node": {
                "ssh_host": "10.0.0.1",
                "ssh_user": "ubuntu",
                "ssh_key": "~/.ssh/id_test",
            }
        }

        # This test just verifies the method exists and handles the call
        # Actual SSH execution would require integration testing
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
            mock_exec.return_value = mock_proc

            returncode, stdout, stderr = await monitor._async_run_ssh_command(
                "slow-node",
                "echo test",
                timeout=5,
            )

            mock_exec.assert_called_once()


class TestClusterMonitorParallel:
    """Tests for parallel query execution."""

    def test_parallel_query_enabled_by_default(self):
        """Test that parallel querying is enabled by default."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        assert monitor.parallel is True

    def test_parallel_query_can_be_disabled(self):
        """Test that parallel querying can be disabled."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        assert monitor.parallel is False


class TestClusterMonitorIntegration:
    """Integration tests for ClusterMonitor (mocked SSH)."""

    def test_full_status_flow(self):
        """Test complete status retrieval flow."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        monitor._hosts = {
            "gpu-node-1": {"status": "ready", "role": "training", "gpu": "H100"},
            "gpu-node-2": {"status": "ready", "role": "selfplay", "gpu": "A100"},
        }
        monitor.game_discovery = None

        # Mock all external calls
        with patch.object(monitor, "_check_connectivity", return_value=True):
            with patch.object(
                monitor,
                "_run_ssh_command",
                return_value=(0, "", ""),
            ):
                status = monitor.get_cluster_status(
                    include_game_counts=False,
                    include_training_status=False,
                    include_disk_usage=False,
                    include_sync_status=False,
                )

        assert status.total_nodes == 2
        assert status.active_nodes == 2
        assert status.query_duration_seconds > 0
        assert len(status.errors) == 0
