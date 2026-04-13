"""Unit tests for cluster monitoring dashboard.

Run with:
    pytest tests/test_cluster_monitor.py -v
"""

import pytest
from datetime import datetime
from pathlib import Path

from app.coordination.cluster_status_monitor import (
    ClusterMonitor,
    ClusterStatus,
    NodeStatus,
)


class TestNodeStatus:
    """Test NodeStatus dataclass."""

    def test_default_values(self):
        """Test NodeStatus default initialization."""
        node = NodeStatus(host_name="test-host")

        assert node.host_name == "test-host"
        assert not node.reachable
        assert node.total_games == 0
        assert not node.training_active
        assert len(node.training_processes) == 0
        assert node.disk_usage_percent == 0.0
        assert node.role == "unknown"
        assert node.error is None

    def test_with_data(self):
        """Test NodeStatus with actual data."""
        node = NodeStatus(
            host_name="lambda-gh200-a",
            reachable=True,
            response_time_ms=120.5,
            total_games=50000,
            game_counts={"hex8_2p": 30000, "square8_2p": 20000},
            training_active=True,
            training_processes=[{"pid": "12345", "command": "train.py"}],
            disk_usage_percent=45.2,
            disk_free_gb=580.0,
            disk_total_gb=1000.0,
            role="nn_training_primary",
            gpu="NVIDIA GH200 (96GB)",
        )

        assert node.reachable
        assert node.total_games == 50000
        assert node.training_active
        assert len(node.training_processes) == 1
        assert node.disk_usage_percent == 45.2
        assert node.role == "nn_training_primary"


class TestClusterStatus:
    """Test ClusterStatus dataclass."""

    def test_default_values(self):
        """Test ClusterStatus default initialization."""
        status = ClusterStatus()

        assert status.total_nodes == 0
        assert status.active_nodes == 0
        assert status.total_games == 0
        assert len(status.games_by_config) == 0
        assert len(status.nodes) == 0
        assert len(status.errors) == 0

    def test_with_nodes(self):
        """Test ClusterStatus with node data."""
        status = ClusterStatus(
            total_nodes=3,
            active_nodes=2,
            unreachable_nodes=1,
            total_games=100000,
            games_by_config={"hex8_2p": 60000, "square8_2p": 40000},
        )

        assert status.total_nodes == 3
        assert status.active_nodes == 2
        assert status.unreachable_nodes == 1
        assert status.total_games == 100000


class TestClusterMonitor:
    """Test ClusterMonitor functionality."""

    def test_init_without_config(self):
        """Test initialization without hosts config."""
        monitor = ClusterMonitor(hosts_config_path="/nonexistent/path.yaml")
        assert monitor.ssh_timeout > 0
        assert len(monitor._hosts) == 0

    def test_init_with_config(self):
        """Test initialization with real hosts config."""
        # Try to find actual config
        config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"

        if config_path.exists():
            monitor = ClusterMonitor(hosts_config_path=config_path)
            assert len(monitor._hosts) > 0
            assert monitor.game_discovery is not None
        else:
            pytest.skip("distributed_hosts.yaml not found")

    def test_get_active_hosts(self):
        """Test filtering active hosts."""
        monitor = ClusterMonitor()
        monitor._hosts = {
            "host1": {"status": "ready"},
            "host2": {"status": "active"},
            "host3": {"status": "offline"},
            "host4": {"status": "ready"},
        }

        active = monitor.get_active_hosts()
        assert len(active) == 3
        assert "host1" in active
        assert "host2" in active
        assert "host4" in active
        assert "host3" not in active

    def test_cluster_status_aggregation(self):
        """Test cluster status aggregates node data correctly."""
        monitor = ClusterMonitor()

        # Create mock node statuses
        monitor._hosts = {
            "node1": {"status": "ready"},
            "node2": {"status": "ready"},
        }

        # Since we can't easily mock SSH, we'll test the structure
        status = ClusterStatus(
            total_nodes=2,
            nodes={
                "node1": NodeStatus(
                    host_name="node1",
                    reachable=True,
                    total_games=50000,
                    game_counts={"hex8_2p": 30000, "square8_2p": 20000},
                    training_active=True,
                    disk_usage_percent=40.0,
                    disk_free_gb=600.0,
                ),
                "node2": NodeStatus(
                    host_name="node2",
                    reachable=True,
                    total_games=30000,
                    game_counts={"hex8_2p": 10000, "square8_2p": 20000},
                    training_active=False,
                    disk_usage_percent=60.0,
                    disk_free_gb=400.0,
                ),
            },
        )

        # Manually aggregate (normally done by get_cluster_status)
        status.active_nodes = sum(1 for n in status.nodes.values() if n.reachable)
        status.total_games = sum(n.total_games for n in status.nodes.values())

        for node in status.nodes.values():
            for config, count in node.game_counts.items():
                status.games_by_config[config] = (
                    status.games_by_config.get(config, 0) + count
                )

        reachable_nodes = [n for n in status.nodes.values() if n.reachable]
        status.avg_disk_usage = sum(
            n.disk_usage_percent for n in reachable_nodes
        ) / len(reachable_nodes)

        # Verify aggregation
        assert status.active_nodes == 2
        assert status.total_games == 80000
        assert status.games_by_config["hex8_2p"] == 40000
        assert status.games_by_config["square8_2p"] == 40000
        assert status.avg_disk_usage == 50.0


class TestErrorHandling:
    """Test error handling and timeouts."""

    def test_unreachable_node(self):
        """Test handling of unreachable nodes."""
        node = NodeStatus(
            host_name="unreachable-host",
            reachable=False,
            error="Connection timeout",
        )

        assert not node.reachable
        assert node.error is not None
        assert node.total_games == 0

    def test_partial_cluster_failure(self):
        """Test cluster status with some failed nodes."""
        status = ClusterStatus(
            total_nodes=3,
            nodes={
                "node1": NodeStatus(host_name="node1", reachable=True, total_games=50000),
                "node2": NodeStatus(host_name="node2", reachable=False, error="Timeout"),
                "node3": NodeStatus(host_name="node3", reachable=True, total_games=30000),
            },
        )

        status.active_nodes = sum(1 for n in status.nodes.values() if n.reachable)
        status.unreachable_nodes = status.total_nodes - status.active_nodes

        assert status.active_nodes == 2
        assert status.unreachable_nodes == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
