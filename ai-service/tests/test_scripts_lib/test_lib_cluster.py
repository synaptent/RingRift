"""Tests for scripts/lib/cluster.py module.

Tests cover:
- ClusterNode SSH operations
- ClusterManager multi-node operations
- ClusterAutomation discovery and management
- VastNodeManager instance management
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import subprocess
import json
import time

from scripts.lib.cluster import (
    ClusterNode,
    ClusterManager,
    ClusterAutomation,
    VastNodeManager,
    NodeStatus,
    NodeHealth,
    GPUInfo,
    CommandResult,
    CommandError,
    get_cluster,
    get_automation,
)


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_success_bool(self):
        result = CommandResult(
            success=True,
            stdout="output",
            stderr="",
            exit_code=0,
            duration_seconds=0.5,
            node="test",
            command="echo hello",
        )
        assert bool(result) is True

    def test_failure_bool(self):
        result = CommandResult(
            success=False,
            stdout="",
            stderr="error",
            exit_code=1,
            duration_seconds=0.5,
            node="test",
            command="bad command",
        )
        assert bool(result) is False

    def test_output_property_stdout(self):
        result = CommandResult(
            success=True,
            stdout="  output  ",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            node="test",
            command="test",
        )
        assert result.output == "output"

    def test_output_property_stderr_fallback(self):
        result = CommandResult(
            success=False,
            stdout="",
            stderr="  error message  ",
            exit_code=1,
            duration_seconds=0.1,
            node="test",
            command="test",
        )
        assert result.output == "error message"


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_memory_free(self):
        gpu = GPUInfo(
            index=0,
            name="RTX 3090",
            memory_total_mb=24576,
            memory_used_mb=8192,
            utilization_percent=50,
            temperature_c=65,
        )
        assert gpu.memory_free_mb == 16384

    def test_memory_utilization_percent(self):
        gpu = GPUInfo(
            index=0,
            name="RTX 3090",
            memory_total_mb=24576,
            memory_used_mb=12288,
            utilization_percent=50,
            temperature_c=65,
        )
        assert abs(gpu.memory_utilization_percent - 50.0) < 0.1

    def test_memory_utilization_zero_total(self):
        gpu = GPUInfo(
            index=0,
            name="Unknown",
            memory_total_mb=0,
            memory_used_mb=0,
            utilization_percent=0,
            temperature_c=0,
        )
        assert gpu.memory_utilization_percent == 0


class TestNodeHealth:
    """Tests for NodeHealth dataclass."""

    def test_total_gpu_memory(self):
        health = NodeHealth(
            status=NodeStatus.HEALTHY,
            gpus=[
                GPUInfo(0, "GPU0", 24576, 0, 0, 50),
                GPUInfo(1, "GPU1", 24576, 0, 0, 50),
            ],
        )
        assert health.total_gpu_memory_gb == 48.0

    def test_avg_gpu_utilization(self):
        health = NodeHealth(
            status=NodeStatus.HEALTHY,
            gpus=[
                GPUInfo(0, "GPU0", 24576, 0, 80, 50),
                GPUInfo(1, "GPU1", 24576, 0, 60, 50),
            ],
        )
        assert health.avg_gpu_utilization == 70.0

    def test_avg_gpu_utilization_no_gpus(self):
        health = NodeHealth(status=NodeStatus.UNKNOWN)
        assert health.avg_gpu_utilization == 0


class TestClusterNode:
    """Tests for ClusterNode class."""

    def test_init_defaults(self):
        node = ClusterNode("test-node")
        assert node.name == "test-node"
        assert node.hostname == "test-node"
        assert node.ssh_user == "ubuntu"
        assert node.connect_timeout == 10
        assert node.command_timeout == 30

    def test_init_custom(self):
        node = ClusterNode(
            name="custom",
            hostname="192.168.1.100",
            ssh_user="admin",
            ssh_key="/path/to/key",
            connect_timeout=20,
            command_timeout=60,
        )
        assert node.hostname == "192.168.1.100"
        assert node.ssh_user == "admin"
        assert node.ssh_key == "/path/to/key"

    def test_repr(self):
        node = ClusterNode("my-node")
        assert repr(node) == "ClusterNode(my-node)"

    @patch("scripts.lib.cluster.subprocess.run")
    def test_run_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="hello",
            stderr="",
        )
        node = ClusterNode("test", hostname="localhost")
        result = node.run("echo hello")

        assert result.success is True
        assert result.stdout == "hello"
        assert result.node == "test"
        assert mock_run.called

    @patch("scripts.lib.cluster.subprocess.run")
    def test_run_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="command failed",
        )
        node = ClusterNode("test", hostname="localhost")
        result = node.run("bad command")

        assert result.success is False
        assert result.exit_code == 1

    @patch("scripts.lib.cluster.subprocess.run")
    def test_run_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
        node = ClusterNode("test", hostname="localhost")
        result = node.run("slow command", timeout=30)

        assert result.success is False
        assert "timed out" in result.stderr

    @patch("scripts.lib.cluster.subprocess.run")
    def test_run_with_check_raises(self, mock_run):
        # Mock subprocess.run to return a failed command result
        failed_result = MagicMock()
        failed_result.returncode = 1
        failed_result.stdout = ""
        failed_result.stderr = "error"
        mock_run.return_value = failed_result

        node = ClusterNode("test", hostname="localhost")

        # When check=True, it should raise CommandError
        with pytest.raises(CommandError) as excinfo:
            node.run("failing command", check=True)

        assert excinfo.value.result.exit_code == 1
        assert excinfo.value.result.success is False


class TestClusterManager:
    """Tests for ClusterManager class."""

    def test_init_default_nodes(self):
        manager = ClusterManager()
        assert len(manager.nodes) == len(ClusterManager.DEFAULT_NODES)

    def test_init_custom_nodes(self):
        custom_nodes = [
            {"name": "node1", "hostname": "192.168.1.1"},
            {"name": "node2", "hostname": "192.168.1.2"},
        ]
        manager = ClusterManager(nodes=custom_nodes)
        assert len(manager.nodes) == 2
        assert "node1" in manager.nodes
        assert "node2" in manager.nodes

    def test_get_node(self):
        manager = ClusterManager()
        # Should have default nodes
        node = manager.get_node("lambda-gh200-e")
        assert node is not None
        assert node.name == "lambda-gh200-e"

    def test_get_node_not_found(self):
        manager = ClusterManager()
        node = manager.get_node("nonexistent")
        assert node is None

    @patch.object(ClusterNode, "check_health")
    def test_get_healthy_nodes(self, mock_health):
        mock_health.return_value = NodeHealth(status=NodeStatus.HEALTHY)

        manager = ClusterManager(nodes=[
            {"name": "node1", "hostname": "host1"},
            {"name": "node2", "hostname": "host2"},
        ])

        healthy = manager.get_healthy_nodes()
        assert len(healthy) == 2

    @patch.object(ClusterNode, "run")
    def test_run_on_all(self, mock_run):
        mock_run.return_value = CommandResult(
            success=True,
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            node="test",
            command="test",
        )

        manager = ClusterManager(nodes=[
            {"name": "node1", "hostname": "host1"},
            {"name": "node2", "hostname": "host2"},
        ])

        results = manager.run_on_all("echo test")
        assert len(results) == 2
        assert all(r.success for r in results.values())


class TestVastNodeManager:
    """Tests for VastNodeManager class."""

    @patch("scripts.lib.cluster.subprocess.run")
    def test_list_instances_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"id": 123, "gpu_name": "RTX 3090"},
                {"id": 456, "gpu_name": "RTX 4090"},
            ]),
        )

        manager = VastNodeManager()
        instances = manager.list_instances()

        assert len(instances) == 2
        assert instances[0]["id"] == 123

    @patch("scripts.lib.cluster.subprocess.run")
    def test_list_instances_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="vastai error",
        )

        manager = VastNodeManager()
        instances = manager.list_instances()

        assert instances == []

    @patch("scripts.lib.cluster.subprocess.run")
    def test_get_instance_ssh(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"id": 123, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345},
            ]),
        )

        manager = VastNodeManager()
        result = manager.get_instance_ssh(123)

        assert result == ("ssh1.vast.ai", 12345)

    @patch("scripts.lib.cluster.subprocess.run")
    def test_get_instance_ssh_not_found(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([]),
        )

        manager = VastNodeManager()
        result = manager.get_instance_ssh(999)

        assert result is None


class TestClusterAutomation:
    """Tests for ClusterAutomation class."""

    def test_init(self):
        automation = ClusterAutomation()
        assert automation.cluster is not None
        assert automation.vast is not None

    @patch.object(VastNodeManager, "list_instances")
    @patch("scripts.lib.cluster.subprocess.run")
    def test_discover_all_nodes_tailscale(self, mock_run, mock_vast):
        # Mock Tailscale status
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "Peer": {
                    "abc123": {
                        "HostName": "lambda-gh200-a",
                        "TailscaleIPs": ["100.1.2.3"],
                        "Online": True,
                    },
                    "def456": {
                        "HostName": "vast-node",
                        "TailscaleIPs": ["100.4.5.6"],
                        "Online": False,
                    },
                }
            }),
        )
        # Mock Vast.ai instances (empty list to avoid interference)
        mock_vast.return_value = []

        automation = ClusterAutomation()
        nodes = automation.discover_all_nodes()

        assert "lambda-gh200-a" in nodes
        assert nodes["lambda-gh200-a"]["provider"] == "lambda"
        assert nodes["lambda-gh200-a"]["online"] is True

    @patch("urllib.request.urlopen")
    def test_check_p2p_status_running(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        automation = ClusterAutomation()
        status = automation.check_p2p_status("100.1.2.3")

        assert status == "running"

    @patch("urllib.request.urlopen")
    def test_check_p2p_status_stopped(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        automation = ClusterAutomation()
        status = automation.check_p2p_status("100.1.2.3")

        assert status == "stopped"


class TestFactoryFunctions:
    """Tests for module factory functions."""

    def test_get_cluster(self):
        cluster = get_cluster()
        assert isinstance(cluster, ClusterManager)

    def test_get_automation(self):
        automation = get_automation()
        assert isinstance(automation, ClusterAutomation)
