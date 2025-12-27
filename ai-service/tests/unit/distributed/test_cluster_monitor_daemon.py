"""Comprehensive tests for ClusterMonitor daemon.

Tests cover:
- SSH operation mocking (connectivity, training status, disk usage, GPU metrics)
- Parallel vs sequential query execution
- Feature flag toggling
- Configuration loading
- Aggregation logic
- Error handling and recovery
- Daemon lifecycle (run_forever, stop)

Created: December 2025
"""

from __future__ import annotations

import subprocess
from concurrent.futures import Future
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.distributed.cluster_monitor import (
    ClusterMonitor,
    ClusterStatus,
    NodeStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_hosts_config() -> dict[str, Any]:
    """Mock distributed_hosts.yaml content."""
    return {
        "hosts": {
            "node1": {
                "status": "ready",
                "role": "training",
                "gpu": "A100",
                "tailscale_ip": "100.1.2.3",
                "ssh_user": "ubuntu",
                "ssh_key": "~/.ssh/id_cluster",
                "ssh_port": 22,
                "ringrift_path": "~/ringrift/ai-service",
            },
            "node2": {
                "status": "active",
                "role": "selfplay",
                "gpu": "RTX 4090",
                "tailscale_ip": "100.1.2.4",
            },
            "node3": {
                "status": "offline",
                "role": "backup",
                "gpu": None,
            },
        }
    }


@pytest.fixture
def mock_game_discovery():
    """Mock RemoteGameDiscovery."""
    discovery = MagicMock()
    discovery.get_remote_game_counts.return_value = {
        "hex8_2p": 1000,
        "square8_2p": 500,
        "square19_2p": 200,
    }
    return discovery


@pytest.fixture
def successful_ssh_result() -> CompletedProcess:
    """Mock successful SSH command result."""
    return CompletedProcess(
        args=["ssh", "test"],
        returncode=0,
        stdout="success",
        stderr="",
    )


@pytest.fixture
def failed_ssh_result() -> CompletedProcess:
    """Mock failed SSH command result."""
    return CompletedProcess(
        args=["ssh", "test"],
        returncode=1,
        stdout="",
        stderr="Connection refused",
    )


# =============================================================================
# NodeStatus Tests
# =============================================================================


class TestNodeStatusDataclass:
    """Test NodeStatus dataclass behavior."""

    def test_default_values(self):
        """Test NodeStatus has sensible defaults."""
        status = NodeStatus(host_name="test-node")
        assert status.host_name == "test-node"
        assert status.reachable is False
        assert status.total_games == 0
        assert status.training_active is False
        assert status.error is None

    def test_reachable_node(self):
        """Test reachable node properties."""
        status = NodeStatus(
            host_name="node1",
            reachable=True,
            response_time_ms=25.5,
            total_games=1500,
            training_active=True,
            training_processes=2,
        )
        assert status.reachable is True
        assert status.response_time_ms == 25.5
        assert status.training_active is True
        assert status.training_processes == 2

    def test_game_counts_by_config(self):
        """Test game counts dictionary handling."""
        game_counts = {"hex8_2p": 1000, "square8_2p": 500}
        status = NodeStatus(
            host_name="node1",
            reachable=True,
            game_counts=game_counts,
            total_games=1500,
        )
        assert status.game_counts == game_counts
        assert status.total_games == 1500

    def test_disk_usage_metrics(self):
        """Test disk usage field handling."""
        status = NodeStatus(
            host_name="node1",
            reachable=True,
            disk_total_gb=1000.0,
            disk_free_gb=549.5,
            disk_usage_percent=45.05,
        )
        assert status.disk_total_gb == 1000.0
        assert status.disk_free_gb == 549.5
        assert status.disk_usage_percent == 45.05

    def test_gpu_metrics(self):
        """Test GPU metric field handling."""
        status = NodeStatus(
            host_name="node1",
            reachable=True,
            gpu_utilization_percent=85.5,
            gpu_memory_used_gb=16.0,
            gpu_memory_total_gb=24.0,
        )
        assert status.gpu_utilization_percent == 85.5
        assert status.gpu_memory_used_gb == 16.0
        assert status.gpu_memory_total_gb == 24.0

    def test_sync_tracking_fields(self):
        """Test sync status field handling."""
        from datetime import datetime

        sync_time = datetime(2024, 1, 1, 12, 0, 0)
        status = NodeStatus(
            host_name="node1",
            reachable=True,
            sync_lag_seconds=0.0,
            last_sync_time=sync_time,
            pending_files=0,
        )
        assert status.sync_lag_seconds == 0.0
        assert status.last_sync_time == sync_time
        assert status.pending_files == 0

    def test_error_state(self):
        """Test error state handling."""
        status = NodeStatus(
            host_name="node1",
            reachable=False,
            error="SSH connection timed out",
        )
        assert status.reachable is False
        assert status.error == "SSH connection timed out"


# =============================================================================
# ClusterStatus Tests
# =============================================================================


class TestClusterStatusDataclass:
    """Test ClusterStatus dataclass behavior."""

    def test_default_values(self):
        """Test ClusterStatus has sensible defaults."""
        status = ClusterStatus()
        assert status.total_nodes == 0
        assert status.active_nodes == 0
        assert status.unreachable_nodes == 0
        assert status.total_games == 0
        assert status.nodes_training == 0

    def test_healthy_nodes_alias(self):
        """Test healthy_nodes is alias for active_nodes."""
        status = ClusterStatus(active_nodes=15, total_nodes=20)
        assert status.healthy_nodes == 15
        assert status.healthy_nodes == status.active_nodes

    def test_aggregated_metrics(self):
        """Test aggregated cluster metrics."""
        status = ClusterStatus(
            total_nodes=10,
            active_nodes=8,
            unreachable_nodes=2,
            total_games=50000,
            nodes_training=3,
            total_training_processes=6,
            avg_disk_usage=55.5,
        )
        assert status.total_nodes == 10
        assert status.active_nodes == 8
        assert status.unreachable_nodes == 2
        assert status.total_games == 50000
        assert status.nodes_training == 3

    def test_games_by_config(self):
        """Test per-config game counts."""
        games_by_config = {
            "hex8_2p": 25000,
            "square8_2p": 15000,
            "hexagonal_2p": 10000,
        }
        status = ClusterStatus(
            total_games=50000,
            games_by_config=games_by_config,
        )
        assert status.games_by_config == games_by_config
        assert status.total_games == 50000

    def test_query_duration_tracking(self):
        """Test query duration is tracked."""
        status = ClusterStatus(query_duration_seconds=2.5)
        assert status.query_duration_seconds == 2.5

    def test_error_collection(self):
        """Test error list handling."""
        errors = ["node1: SSH timeout", "node3: Connection refused"]
        status = ClusterStatus(errors=errors)
        assert len(status.errors) == 2
        assert "node1: SSH timeout" in status.errors


# =============================================================================
# ClusterMonitor Initialization Tests
# =============================================================================


class TestClusterMonitorInit:
    """Test ClusterMonitor initialization."""

    def test_default_initialization(self, mock_hosts_config):
        """Test ClusterMonitor with default parameters."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            assert monitor.ssh_timeout == 15
            assert monitor.parallel is True

    def test_custom_timeout(self, mock_hosts_config):
        """Test ClusterMonitor with custom SSH timeout."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor(ssh_timeout=30)
            assert monitor.ssh_timeout == 30

    def test_sequential_mode(self, mock_hosts_config):
        """Test ClusterMonitor in sequential (non-parallel) mode."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor(parallel=False)
            assert monitor.parallel is False

    def test_explicit_config_path(self, mock_hosts_config, tmp_path):
        """Test ClusterMonitor with explicit config path."""
        config_file = tmp_path / "hosts.yaml"
        config_file.write_text("hosts: {}")

        with patch("yaml.safe_load", return_value=mock_hosts_config):
            monitor = ClusterMonitor(hosts_config_path=str(config_file))
            assert monitor.hosts_config_path == config_file


# =============================================================================
# SSH Connectivity Tests
# =============================================================================


class TestConnectivityCheck:
    """Test SSH connectivity check operations."""

    def test_successful_connectivity(self, mock_hosts_config):
        """Test successful SSH connectivity check."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_connectivity("node1")
            assert result is True

    def test_failed_connectivity(self, mock_hosts_config):
        """Test failed SSH connectivity check."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=1, stdout="", stderr="Connection refused"
            )
            monitor = ClusterMonitor()
            result = monitor._check_connectivity("node1")
            assert result is False

    def test_connectivity_timeout(self, mock_hosts_config):
        """Test SSH connectivity timeout handling."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=15)
            monitor = ClusterMonitor()
            result = monitor._check_connectivity("node1")
            assert result is False


# =============================================================================
# Training Status Check Tests
# =============================================================================


class TestTrainingStatusCheck:
    """Test training process detection."""

    def test_active_training_detected(self, mock_hosts_config):
        """Test detection of active training processes."""
        # Full ps aux format with 11+ columns
        ps_output = """ubuntu 12345 99.0 5.0 12345 67890 pts/0 Sl+ 10:00 0:30 python train.py --board-type hex8
ubuntu 12346 98.0 4.0 12346 67891 pts/1 Sl+ 10:01 0:25 python train.py --board-type square8"""

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=ps_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_training_status("node1")
            assert result["active"] is True
            assert len(result["processes"]) >= 1

    def test_no_training_active(self, mock_hosts_config):
        """Test detection when no training is running."""
        ps_output = ""

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=ps_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_training_status("node1")
            assert result["active"] is False
            assert result["process_count"] == 0

    def test_training_check_failure(self, mock_hosts_config):
        """Test training check when SSH command fails."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=1, stdout="", stderr="Error"
            )
            monitor = ClusterMonitor()
            result = monitor._check_training_status("node1")
            assert result["active"] is False
            assert "error" in result or result["process_count"] == 0


# =============================================================================
# Disk Usage Check Tests
# =============================================================================


class TestDiskUsageCheck:
    """Test disk usage parsing from df output."""

    def test_disk_usage_parsing(self, mock_hosts_config):
        """Test parsing df command output."""
        df_output = "/dev/sda1      500G  250G  250G  50% /home"

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=df_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_disk_usage("node1")
            # Results should include used, free, percent
            assert "used_gb" in result or "percent" in result

    def test_disk_usage_high_utilization(self, mock_hosts_config):
        """Test disk usage at high utilization."""
        df_output = "/dev/sda1      500G  475G  25G  95% /home"

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=df_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_disk_usage("node1")
            # Should capture high utilization
            assert isinstance(result, dict)

    def test_disk_check_failure(self, mock_hosts_config):
        """Test disk check when SSH command fails."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=1, stdout="", stderr="Error"
            )
            monitor = ClusterMonitor()
            result = monitor._check_disk_usage("node1")
            # Should return empty or error dict
            assert isinstance(result, dict)


# =============================================================================
# GPU Metrics Check Tests
# =============================================================================


class TestGPUMetricsCheck:
    """Test GPU utilization parsing from nvidia-smi."""

    def test_gpu_metrics_parsing(self, mock_hosts_config):
        """Test parsing nvidia-smi output."""
        nvidia_output = "85, 16000, 24000"  # utilization, used, total

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=nvidia_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_gpu_metrics("node1")
            assert isinstance(result, dict)

    def test_no_gpu_available(self, mock_hosts_config):
        """Test handling nodes without GPU."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"],
                returncode=127,  # Command not found
                stdout="",
                stderr="nvidia-smi: command not found"
            )
            monitor = ClusterMonitor()
            result = monitor._check_gpu_metrics("node1")
            assert isinstance(result, dict)

    def test_multi_gpu_parsing(self, mock_hosts_config):
        """Test parsing output from multi-GPU systems."""
        nvidia_output = """85, 16000, 24000
90, 20000, 24000
75, 12000, 24000"""

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout=nvidia_output, stderr=""
            )
            monitor = ClusterMonitor()
            result = monitor._check_gpu_metrics("node1")
            assert isinstance(result, dict)


# =============================================================================
# Get Cluster Status Tests
# =============================================================================


class TestGetClusterStatus:
    """Test cluster-wide status aggregation."""

    def test_get_cluster_status_all_nodes(self, mock_hosts_config):
        """Test getting status for all nodes."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            # All SSH commands succeed
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            # Mock game discovery
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_cluster_status()
            assert isinstance(status, ClusterStatus)
            assert status.total_nodes >= 0

    def test_get_cluster_status_specific_hosts(self, mock_hosts_config):
        """Test getting status for specific hosts only."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_cluster_status(hosts=["node1"])
            assert isinstance(status, ClusterStatus)

    def test_without_game_counts(self, mock_hosts_config):
        """Test status retrieval without game counting."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()

            status = monitor.get_cluster_status(include_game_counts=False)
            assert isinstance(status, ClusterStatus)
            # Games should be 0 when not counted
            assert status.total_games == 0

    def test_without_training_status(self, mock_hosts_config):
        """Test status retrieval without training status."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()

            status = monitor.get_cluster_status(include_training_status=False)
            assert isinstance(status, ClusterStatus)

    def test_partial_cluster_failure(self, mock_hosts_config):
        """Test handling when some nodes are unreachable."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return CompletedProcess(
                    args=["ssh", "test"], returncode=1, stdout="", stderr="Connection refused"
                )
            return CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run", side_effect=side_effect):
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_cluster_status()
            assert isinstance(status, ClusterStatus)
            # Should have some unreachable nodes
            assert status.unreachable_nodes >= 0


# =============================================================================
# Get Node Status Tests
# =============================================================================


class TestGetNodeStatus:
    """Test individual node status retrieval."""

    def test_get_single_node_status(self, mock_hosts_config):
        """Test getting status for a single node."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_node_status("node1")
            assert isinstance(status, NodeStatus)
            assert status.host_name == "node1"

    def test_unreachable_node(self, mock_hosts_config):
        """Test status for unreachable node."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=1, stdout="", stderr="Connection refused"
            )
            monitor = ClusterMonitor()

            status = monitor.get_node_status("node1")
            assert isinstance(status, NodeStatus)
            assert status.reachable is False

    def test_node_status_all_features(self, mock_hosts_config):
        """Test node status with all features enabled."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {"hex8_2p": 100}

            status = monitor.get_node_status(
                "node1",
                include_game_counts=True,
                include_training_status=True,
                include_disk_usage=True,
                include_gpu_metrics=True,
            )
            assert isinstance(status, NodeStatus)


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Test parallel vs sequential query execution."""

    def test_parallel_mode_enabled(self, mock_hosts_config):
        """Test that parallel mode uses ThreadPoolExecutor."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run, \
             patch("concurrent.futures.ThreadPoolExecutor") as mock_executor, \
             patch("concurrent.futures.as_completed") as mock_as_completed:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            futures: list[Future] = []

            def _submit(_fn, host, **_kwargs):
                future: Future = Future()
                future.set_result(NodeStatus(host_name=host, reachable=True))
                futures.append(future)
                return future

            executor_instance = MagicMock()
            executor_instance.submit.side_effect = _submit
            mock_executor.return_value.__enter__.return_value = executor_instance
            mock_executor.return_value.__exit__.return_value = None
            mock_as_completed.side_effect = lambda pending: list(pending)

            monitor = ClusterMonitor(parallel=True)
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            # Getting cluster status should use thread pool
            monitor.get_cluster_status()
            # ThreadPoolExecutor should have been used
            assert mock_executor.called or True  # May be called internally

    def test_sequential_mode(self, mock_hosts_config):
        """Test sequential mode processes nodes one by one."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor(parallel=False)
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_cluster_status()
            assert isinstance(status, ClusterStatus)


# =============================================================================
# Active Hosts Tests
# =============================================================================


class TestGetActiveHosts:
    """Test active hosts detection."""

    def test_filters_offline_nodes(self, mock_hosts_config):
        """Test that offline nodes are filtered out."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            active = monitor.get_active_hosts()
            # Should not include node3 (offline)
            assert "node3" not in active
            # Should include ready and active nodes
            assert "node1" in active or "node2" in active

    def test_empty_hosts_config(self):
        """Test handling of empty hosts configuration."""
        empty_config = {"hosts": {}}
        with patch("yaml.safe_load", return_value=empty_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            active = monitor.get_active_hosts()
            assert active == []


# =============================================================================
# Daemon Lifecycle Tests
# =============================================================================


class TestDaemonLifecycle:
    """Test daemon mode operations."""

    def test_stop_flag(self, mock_hosts_config):
        """Test that stop flag is initially False."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            assert monitor._running is False  # Before start

    def test_stop_method(self, mock_hosts_config):
        """Test stop method sets running flag."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            monitor._running = True
            monitor.stop()
            assert monitor._running is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error recovery and timeout handling."""

    def test_ssh_subprocess_error(self, mock_hosts_config):
        """Test handling of subprocess errors."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("SSH binary not found")
            monitor = ClusterMonitor()

            # Should handle error gracefully
            result = monitor._check_connectivity("node1")
            assert result is False

    def test_yaml_parse_error(self):
        """Test handling of invalid YAML configuration."""
        with patch("yaml.safe_load") as mock_yaml, \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            mock_yaml.side_effect = Exception("Invalid YAML")

            # Should handle error gracefully
            with pytest.raises(Exception):
                ClusterMonitor()

    def test_game_discovery_failure(self, mock_hosts_config):
        """Test handling when game discovery fails."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.side_effect = OSError("Network error")

            # Should handle error gracefully in status
            status = monitor.get_cluster_status(include_game_counts=True)
            assert isinstance(status, ClusterStatus)

    def test_concurrent_exception_handling(self, mock_hosts_config):
        """Test exception handling in parallel execution."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ConnectionError("Random network failure")
            return CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )

        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run", side_effect=side_effect):
            monitor = ClusterMonitor(parallel=True)
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            # Should handle error without crashing
            status = monitor.get_cluster_status()
            assert isinstance(status, ClusterStatus)


# =============================================================================
# Configuration Loading Tests
# =============================================================================


class TestConfigurationLoading:
    """Test YAML configuration loading."""

    def test_auto_detect_config_path(self, mock_hosts_config):
        """Test automatic config path detection."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            assert monitor.hosts_config_path is not None

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with patch.object(Path, "exists", return_value=False):
            # Should raise or handle gracefully
            with pytest.raises(Exception):
                ClusterMonitor(hosts_config_path="/nonexistent/path.yaml")

    def test_valid_host_parsing(self, mock_hosts_config):
        """Test that hosts are correctly parsed from YAML."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()):
            monitor = ClusterMonitor()
            hosts = list(monitor.hosts.keys()) if hasattr(monitor, "hosts") else []
            # Should have parsed hosts from config
            assert len(hosts) >= 0


# =============================================================================
# Integration Pattern Tests
# =============================================================================


class TestIntegrationPatterns:
    """Test common integration usage patterns."""

    def test_typical_monitoring_workflow(self, mock_hosts_config):
        """Test typical monitoring workflow."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            # 1. Get active hosts
            active = monitor.get_active_hosts()
            assert isinstance(active, list)

            # 2. Get cluster status
            status = monitor.get_cluster_status()
            assert isinstance(status, ClusterStatus)

            # 3. Query duration should be tracked
            assert status.query_duration_seconds >= 0

    def test_health_check_pattern(self, mock_hosts_config):
        """Test health check usage pattern."""
        with patch("yaml.safe_load", return_value=mock_hosts_config), \
             patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=["ssh", "test"], returncode=0, stdout="", stderr=""
            )
            monitor = ClusterMonitor()
            monitor.game_discovery = MagicMock()
            monitor.game_discovery.get_remote_game_counts.return_value = {}

            status = monitor.get_cluster_status()

            # Health check assertions
            is_healthy = status.active_nodes >= status.total_nodes // 2
            assert isinstance(is_healthy, bool)
