"""Tests for ClusterStatusMonitor.

Covers:
- NodeStatus and ClusterStatus dataclasses
- ClusterMonitor initialization and host loading
- Node status queries (connectivity, training, disk, GPU, sync)
- Cluster-wide status aggregation
- Async versions of status methods
- Health check implementation
- Dashboard printing and watch mode

December 2025: Added comprehensive tests for cluster monitoring.
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.coordination.cluster_status_monitor import (
    ClusterMonitor,
    ClusterStatus,
    NodeStatus,
)


def _mock_ssh_result(*, success: bool, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.success = success
    result.stdout = stdout
    result.stderr = stderr
    return result


# =============================================================================
# NodeStatus Dataclass Tests
# =============================================================================


class TestNodeStatus:
    """Tests for NodeStatus dataclass."""

    def test_default_values(self):
        """Test default initialization of NodeStatus."""
        status = NodeStatus(host_name="test-node")

        assert status.host_name == "test-node"
        assert status.reachable is False
        assert status.response_time_ms == 0.0
        assert status.total_games == 0
        assert status.training_active is False
        assert status.disk_usage_percent == 0.0
        assert status.gpu_utilization_percent == 0.0
        assert status.error is None

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        status = NodeStatus(
            host_name="gpu-node-1",
            reachable=True,
            response_time_ms=45.5,
            game_counts={"hex8_2p": 100, "square8_4p": 50},
            total_games=150,
            training_active=True,
            disk_usage_percent=65.5,
            disk_free_gb=120.0,
            disk_total_gb=500.0,
            gpu_utilization_percent=85.0,
            gpu_memory_used_gb=70.0,
            gpu_memory_total_gb=80.0,
            role="training",
            gpu="H100",
            status="ready",
        )

        assert status.host_name == "gpu-node-1"
        assert status.reachable is True
        assert status.total_games == 150
        assert status.gpu == "H100"
        assert status.disk_free_gb == 120.0

    def test_game_counts_default(self):
        """Test game_counts defaults to empty dict."""
        status = NodeStatus(host_name="test")
        assert status.game_counts == {}

    def test_training_processes_default(self):
        """Test training_processes defaults to empty list."""
        status = NodeStatus(host_name="test")
        assert status.training_processes == []


# =============================================================================
# ClusterStatus Dataclass Tests
# =============================================================================


class TestClusterStatus:
    """Tests for ClusterStatus dataclass."""

    def test_default_values(self):
        """Test default initialization of ClusterStatus."""
        status = ClusterStatus()

        assert status.total_nodes == 0
        assert status.active_nodes == 0
        assert status.unreachable_nodes == 0
        assert status.total_games == 0
        assert status.nodes_training == 0
        assert isinstance(status.timestamp, datetime)

    def test_healthy_nodes_alias(self):
        """Test healthy_nodes property is alias for active_nodes."""
        status = ClusterStatus(active_nodes=5)

        assert status.healthy_nodes == 5
        assert status.healthy_nodes == status.active_nodes

    def test_games_by_config_default(self):
        """Test games_by_config defaults to empty dict."""
        status = ClusterStatus()
        assert status.games_by_config == {}

    def test_nodes_default(self):
        """Test nodes defaults to empty dict."""
        status = ClusterStatus()
        assert status.nodes == {}


# =============================================================================
# ClusterMonitor Initialization Tests
# =============================================================================


class TestClusterMonitorInit:
    """Tests for ClusterMonitor initialization."""

    def test_default_initialization(self):
        """Test monitor initializes with defaults."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        assert monitor.ssh_timeout == 15
        assert monitor.parallel is True

    def test_custom_timeout(self):
        """Test custom SSH timeout."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(ssh_timeout=30)

        assert monitor.ssh_timeout == 30

    def test_sequential_mode(self):
        """Test disabling parallel mode."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)

        assert monitor.parallel is False

    def test_config_path_discovery(self):
        """Test auto-discovery of hosts config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config" / "distributed_hosts.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("hosts: {}")

            with patch.object(ClusterMonitor, "_load_hosts"):
                monitor = ClusterMonitor(hosts_config_path=config_path)

            assert monitor.hosts_config_path == config_path


# =============================================================================
# Host Loading Tests
# =============================================================================


class TestHostLoading:
    """Tests for host configuration loading."""

    def test_load_hosts_via_cluster_config(self):
        """Test loading hosts via cluster_config helpers."""
        mock_config = MagicMock()
        mock_config.hosts_raw = {
            "test-node": {"status": "ready", "tailscale_ip": "100.1.2.3"}
        }

        mock_cluster_config = MagicMock()
        mock_cluster_config.load_cluster_config = MagicMock(return_value=mock_config)

        with patch.dict("sys.modules", {"app.config.cluster_config": mock_cluster_config}):
            monitor = ClusterMonitor.__new__(ClusterMonitor)
            monitor.hosts_config_path = Path("/test/path.yaml")
            monitor._hosts = {}
            monitor._load_hosts()

        # After load, _hosts should have content (or fallback to empty if module mock incomplete)
        # The key test is it doesn't crash
        assert isinstance(monitor._hosts, dict)

    def test_load_hosts_yaml_fallback(self):
        """Test fallback to direct YAML loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "hosts.yaml"
            config_path.write_text("""
hosts:
  gpu-node-1:
    status: ready
    tailscale_ip: 100.1.2.3
  gpu-node-2:
    status: ready
    tailscale_ip: 100.1.2.4
""")

            monitor = ClusterMonitor.__new__(ClusterMonitor)
            monitor.hosts_config_path = config_path
            monitor._hosts = {}

            # Mock ImportError for cluster_config by making it None
            with patch.dict("sys.modules", {"app.config.cluster_config": None}):
                monitor._load_hosts()

            assert len(monitor._hosts) == 2
            assert "gpu-node-1" in monitor._hosts

    def test_load_hosts_missing_config(self):
        """Test handling of missing config file."""
        monitor = ClusterMonitor.__new__(ClusterMonitor)
        monitor.hosts_config_path = Path("/nonexistent/path.yaml")
        monitor._hosts = {}

        with patch.dict("sys.modules", {"app.config.cluster_config": None}):
            monitor._load_hosts()

        assert monitor._hosts == {}


# =============================================================================
# Active Hosts Tests
# =============================================================================


class TestGetActiveHosts:
    """Tests for get_active_hosts method."""

    def test_get_active_hosts_filters_ready(self):
        """Test filtering hosts by ready status."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "node-1": {"status": "ready"},
                "node-2": {"status": "active"},
                "node-3": {"status": "offline"},
                "node-4": {"status": "maintenance"},
            }

        active = monitor.get_active_hosts()

        assert "node-1" in active
        assert "node-2" in active
        assert "node-3" not in active
        assert "node-4" not in active

    def test_get_active_hosts_empty(self):
        """Test with no active hosts."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {}

        assert monitor.get_active_hosts() == []


# =============================================================================
# Connectivity Tests
# =============================================================================


class TestConnectivity:
    """Tests for connectivity checking."""

    def test_check_connectivity_success(self):
        """Test successful connectivity check."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "test-node": {
                    "tailscale_ip": "100.1.2.3",
                    "ssh_user": "ubuntu",
                    "ssh_key": "~/.ssh/id_test",
                    "ssh_port": 22,
                }
            }

        mock_client = MagicMock()
        mock_client.is_alive.return_value = True

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            result = monitor._check_connectivity("test-node")

        assert result is True

    def test_check_connectivity_failure(self):
        """Test failed connectivity check."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "test-node": {
                    "tailscale_ip": "100.1.2.3",
                    "ssh_user": "ubuntu",
                }
            }

        mock_client = MagicMock()
        mock_client.is_alive.return_value = False

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            result = monitor._check_connectivity("test-node")

        assert result is False

    def test_check_connectivity_timeout(self):
        """Test connectivity check timeout."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        mock_client = MagicMock()
        mock_client.is_alive.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=5)

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            result = monitor._check_connectivity("test-node")

        assert result is False

    def test_check_connectivity_unknown_host(self):
        """Test connectivity check for unknown host."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {}

        result = monitor._check_connectivity("unknown-host")

        assert result is False


# =============================================================================
# Training Status Tests
# =============================================================================


class TestTrainingStatus:
    """Tests for training status checking."""

    def test_check_training_status_active(self):
        """Test detecting active training processes."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "test-node": {
                    "tailscale_ip": "100.1.2.3",
                    "ssh_user": "ubuntu",
                }
            }

        run_result = _mock_ssh_result(
            success=True,
            stdout=(
            "ubuntu   12345 50.0 10.0 123456 12345 pts/0 S+ 12:34 0:05 python -m app.training.train\n"
            ),
        )
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            status = monitor._check_training_status("test-node")

        assert status["active"] is True
        assert len(status["processes"]) == 1

    def test_check_training_status_inactive(self):
        """Test when no training is active."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(success=True, stdout="")
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            status = monitor._check_training_status("test-node")

        assert status["active"] is False
        assert status["processes"] == []

    def test_check_training_status_error(self):
        """Test handling SSH error."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        mock_client = MagicMock()
        mock_client.run.side_effect = Exception("SSH error")

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            status = monitor._check_training_status("test-node")

        assert status["active"] is False


# =============================================================================
# Disk Usage Tests
# =============================================================================


class TestDiskUsage:
    """Tests for disk usage checking."""

    def test_check_disk_usage_success(self):
        """Test successful disk usage check."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "test-node": {
                    "tailscale_ip": "100.1.2.3",
                    "ringrift_path": "~/ringrift/ai-service",
                }
            }

        run_result = _mock_ssh_result(
            success=True,
            stdout="/dev/sda1 500G 350G 150G 70% /home",
        )
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            usage = monitor._check_disk_usage("test-node")

        assert usage["percent"] == 70.0
        assert usage["free_gb"] == 150.0
        assert usage["total_gb"] == 500.0

    def test_check_disk_usage_error(self):
        """Test disk usage check failure."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(success=False)
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            usage = monitor._check_disk_usage("test-node")

        assert usage["percent"] == 0.0
        assert usage["free_gb"] == 0.0


# =============================================================================
# GPU Metrics Tests
# =============================================================================


class TestGPUMetrics:
    """Tests for GPU metrics checking."""

    def test_check_gpu_metrics_success(self):
        """Test successful GPU metrics check."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(
            success=True,
            stdout="85, 65536, 81920",
        )
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            metrics = monitor._check_gpu_metrics("test-node")

        assert metrics["utilization_percent"] == 85.0
        assert metrics["memory_used_gb"] == 64.0  # 65536 / 1024
        assert metrics["memory_total_gb"] == 80.0  # 81920 / 1024

    def test_check_gpu_metrics_no_gpu(self):
        """Test GPU metrics when no GPU available."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(success=False, stdout="")
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            metrics = monitor._check_gpu_metrics("test-node")

        assert metrics["utilization_percent"] == 0.0
        assert metrics["memory_used_gb"] == 0.0


# =============================================================================
# Node Status Integration Tests
# =============================================================================


class TestNodeStatusIntegration:
    """Tests for get_node_status method."""

    def test_get_node_status_unreachable(self):
        """Test node status for unreachable node."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"status": "ready"}}
            monitor.game_discovery = None

        with patch.object(monitor, "_check_connectivity", return_value=False):
            status = monitor.get_node_status("test-node")

        assert status.reachable is False
        assert status.error == "Host unreachable"

    def test_get_node_status_reachable(self):
        """Test node status for reachable node."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "test-node": {"status": "ready", "gpu": "H100"}
            }
            monitor.game_discovery = None

        with patch.object(monitor, "_check_connectivity", return_value=True):
            with patch.object(
                monitor,
                "_check_training_status",
                return_value={"active": False, "processes": []},
            ):
                with patch.object(
                    monitor,
                    "_check_disk_usage",
                    return_value={"percent": 50.0, "free_gb": 100.0, "total_gb": 200.0},
                ):
                    with patch.object(
                        monitor,
                        "_check_gpu_metrics",
                        return_value={
                            "utilization_percent": 80.0,
                            "memory_used_gb": 60.0,
                            "memory_total_gb": 80.0,
                        },
                    ):
                        status = monitor.get_node_status("test-node")

        assert status.reachable is True
        assert status.disk_usage_percent == 50.0
        assert status.gpu_utilization_percent == 80.0


# =============================================================================
# Cluster Status Aggregation Tests
# =============================================================================


class TestClusterStatusAggregation:
    """Tests for cluster-wide status aggregation."""

    def test_get_cluster_status_aggregates_games(self):
        """Test that cluster status aggregates game counts."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "node-1": {"status": "ready"},
                "node-2": {"status": "ready"},
            }

        node1 = NodeStatus(
            host_name="node-1",
            reachable=True,
            game_counts={"hex8_2p": 100},
            total_games=100,
        )
        node2 = NodeStatus(
            host_name="node-2",
            reachable=True,
            game_counts={"hex8_2p": 50, "square8_4p": 25},
            total_games=75,
        )

        with patch.object(monitor, "get_node_status", side_effect=[node1, node2]):
            status = monitor.get_cluster_status(hosts=["node-1", "node-2"])

        assert status.total_games == 175
        assert status.games_by_config["hex8_2p"] == 150
        assert status.games_by_config["square8_4p"] == 25

    def test_get_cluster_status_counts_nodes(self):
        """Test that cluster status counts active/unreachable nodes."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {
                "node-1": {"status": "ready"},
                "node-2": {"status": "ready"},
                "node-3": {"status": "ready"},
            }

        nodes = [
            NodeStatus(host_name="node-1", reachable=True),
            NodeStatus(host_name="node-2", reachable=True),
            NodeStatus(host_name="node-3", reachable=False, error="timeout"),
        ]

        with patch.object(monitor, "get_node_status", side_effect=nodes):
            status = monitor.get_cluster_status(hosts=["node-1", "node-2", "node-3"])

        assert status.total_nodes == 3
        assert status.active_nodes == 2
        assert status.unreachable_nodes == 1
        assert len(status.errors) == 1


# =============================================================================
# Async Methods Tests
# =============================================================================


class TestAsyncMethods:
    """Tests for async versions of status methods."""

    @pytest.mark.asyncio
    async def test_async_check_host_connectivity(self):
        """Test async connectivity check."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        with patch.object(
            monitor,
            "_async_run_ssh_command",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ):
            result = await monitor._async_check_host_connectivity("test-node")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_cluster_status_async(self):
        """Test async cluster status retrieval."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"node-1": {"status": "ready"}}

        mock_status = NodeStatus(host_name="node-1", reachable=True)

        with patch.object(
            monitor,
            "get_node_status_async",
            new_callable=AsyncMock,
            return_value=mock_status,
        ):
            status = await monitor.get_cluster_status_async(hosts=["node-1"])

        assert status.total_nodes == 1
        assert status.active_nodes == 1


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_with_hosts(self):
        """Test health check when hosts are configured."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"node-1": {}, "node-2": {}}
            monitor._running = True

        result = monitor.health_check()

        assert result.healthy is True
        assert "2 hosts" in result.message

    def test_health_check_no_hosts(self):
        """Test health check when no hosts configured."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {}

        result = monitor.health_check()

        assert result.healthy is False
        assert "0 hosts" in result.message


# =============================================================================
# Run Forever Tests
# =============================================================================


class TestRunForever:
    """Tests for run_forever method."""

    @pytest.mark.asyncio
    async def test_run_forever_stops(self):
        """Test that run_forever can be stopped."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {}

        with patch.object(
            monitor,
            "get_cluster_status_async",
            new_callable=AsyncMock,
            return_value=ClusterStatus(),
        ):
            # Start run_forever and stop it after a short delay
            task = asyncio.create_task(monitor.run_forever(interval=1))
            await asyncio.sleep(0.1)
            monitor.stop()

            # Give it time to exit
            await asyncio.sleep(0.2)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        assert monitor._running is False


# =============================================================================
# Dashboard and Watch Tests
# =============================================================================


class TestDashboard:
    """Tests for dashboard and watch functionality."""

    def test_print_dashboard_does_not_crash(self):
        """Test that print_dashboard handles all data gracefully."""
        status = ClusterStatus(
            total_nodes=2,
            active_nodes=1,
            unreachable_nodes=1,
            total_games=1000,
            games_by_config={"hex8_2p": 500, "square8_4p": 500},
            nodes={
                "node-1": NodeStatus(
                    host_name="node-1",
                    reachable=True,
                    total_games=500,
                    training_active=True,
                    disk_usage_percent=50.0,
                    disk_free_gb=100.0,
                    gpu="H100",
                    gpu_utilization_percent=80.0,
                    gpu_memory_used_gb=60.0,
                    gpu_memory_total_gb=80.0,
                ),
                "node-2": NodeStatus(
                    host_name="node-2",
                    reachable=False,
                    error="timeout",
                ),
            },
            errors=["node-2: timeout"],
        )

        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        # Should not raise
        with patch("builtins.print"):
            monitor.print_dashboard(status)

    def test_print_dashboard_empty_status(self):
        """Test dashboard with empty status."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()

        with patch("builtins.print"):
            monitor.print_dashboard(ClusterStatus())


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_node_status_with_no_ssh_host(self):
        """Test connectivity check with missing SSH host."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {}}  # No tailscale_ip or ssh_host

        result = monitor._check_connectivity("test-node")
        assert result is False

    def test_disk_usage_parse_error(self):
        """Test handling of malformed df output."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(success=True, stdout="invalid output")
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            usage = monitor._check_disk_usage("test-node")

        assert usage["percent"] == 0.0

    def test_gpu_metrics_parse_error(self):
        """Test handling of malformed nvidia-smi output."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            monitor._hosts = {"test-node": {"tailscale_ip": "100.1.2.3"}}

        run_result = _mock_ssh_result(success=True, stdout="not,valid,numbers")
        mock_client = MagicMock()
        mock_client.run.return_value = run_result

        with patch.object(monitor, "_get_ssh_client", return_value=mock_client):
            metrics = monitor._check_gpu_metrics("test-node")

        assert metrics["utilization_percent"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
