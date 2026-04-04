"""Tests for app/distributed/cluster_monitor.py - cluster monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta
import warnings
from unittest.mock import MagicMock, patch

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from app.distributed.cluster_monitor import (
        ClusterMonitor,
        ClusterStatus,
        NodeStatus,
    )


class TestNodeStatus:
    """Tests for NodeStatus dataclass."""

    def test_basic_creation(self):
        """Test creating NodeStatus with host name."""
        status = NodeStatus(host_name="test-node")
        assert status.host_name == "test-node"
        assert status.reachable is False  # Default
        assert status.total_games == 0  # Default

    def test_default_values(self):
        """Test default values are set correctly."""
        status = NodeStatus(host_name="node1")
        assert status.reachable is False
        assert status.response_time_ms == 0.0
        assert status.game_counts == {}
        assert status.total_games == 0
        assert status.training_active is False
        assert status.training_processes == []
        assert status.disk_usage_percent == 0.0
        assert status.disk_free_gb == 0.0
        assert status.disk_total_gb == 0.0
        assert status.gpu_utilization_percent == 0.0
        assert status.role == "unknown"
        assert status.status == "unknown"
        assert status.error is None

    def test_all_fields(self):
        """Test NodeStatus with all fields populated."""
        now = datetime.now()
        status = NodeStatus(
            host_name="training-node",
            reachable=True,
            response_time_ms=45.5,
            game_counts={"hex8_2p": 1000, "square8_2p": 500},
            total_games=1500,
            training_active=True,
            training_processes=[{"pid": 1234, "command": "train.py"}],
            disk_usage_percent=75.5,
            disk_free_gb=100.0,
            disk_total_gb=400.0,
            memory_usage_percent=60.0,
            cpu_percent=80.0,
            gpu_utilization_percent=95.0,
            gpu_memory_used_gb=40.0,
            gpu_memory_total_gb=80.0,
            last_sync_time=now,
            sync_lag_seconds=30.0,
            pending_files=5,
            role="training",
            gpu="A100",
            status="active",
            error=None,
            last_check=now,
        )
        assert status.host_name == "training-node"
        assert status.reachable is True
        assert status.response_time_ms == 45.5
        assert status.total_games == 1500
        assert status.training_active is True
        assert len(status.training_processes) == 1
        assert status.disk_usage_percent == 75.5
        assert status.gpu_utilization_percent == 95.0
        assert status.role == "training"
        assert status.gpu == "A100"

    def test_game_counts_mutable(self):
        """Test game_counts dict is mutable."""
        status = NodeStatus(host_name="node1")
        status.game_counts["hex8_2p"] = 100
        assert status.game_counts["hex8_2p"] == 100

    def test_error_tracking(self):
        """Test error field stores messages."""
        status = NodeStatus(host_name="node1", error="Connection refused")
        assert status.error == "Connection refused"


class TestClusterStatus:
    """Tests for ClusterStatus dataclass."""

    def test_basic_creation(self):
        """Test creating ClusterStatus with defaults."""
        status = ClusterStatus()
        assert status.total_nodes == 0
        assert status.active_nodes == 0
        assert status.unreachable_nodes == 0
        assert status.total_games == 0
        assert status.nodes == {}
        assert status.errors == []

    def test_with_node_data(self):
        """Test ClusterStatus with node data."""
        node1 = NodeStatus(host_name="node1", reachable=True, total_games=500)
        node2 = NodeStatus(host_name="node2", reachable=True, total_games=300)
        node3 = NodeStatus(host_name="node3", reachable=False, error="Timeout")

        status = ClusterStatus(
            total_nodes=3,
            active_nodes=2,
            unreachable_nodes=1,
            total_games=800,
            games_by_config={"hex8_2p": 500, "square8_2p": 300},
            nodes={"node1": node1, "node2": node2, "node3": node3},
        )
        assert status.total_nodes == 3
        assert status.active_nodes == 2
        assert status.unreachable_nodes == 1
        assert status.total_games == 800
        assert len(status.nodes) == 3
        assert status.nodes["node1"].total_games == 500

    def test_resource_aggregates(self):
        """Test resource aggregate fields."""
        status = ClusterStatus(
            avg_disk_usage=65.0,
            total_disk_free_gb=1000.0,
            total_disk_capacity_gb=4000.0,
        )
        assert status.avg_disk_usage == 65.0
        assert status.total_disk_free_gb == 1000.0
        assert status.total_disk_capacity_gb == 4000.0

    def test_query_metadata(self):
        """Test query metadata fields."""
        status = ClusterStatus(
            query_duration_seconds=2.5,
            errors=["Node X unreachable", "SSH timeout on Y"],
        )
        assert status.query_duration_seconds == 2.5
        assert len(status.errors) == 2

    def test_timestamp_default(self):
        """Test timestamp defaults to now."""
        before = datetime.now()
        status = ClusterStatus()
        after = datetime.now()
        assert before <= status.timestamp <= after


class TestClusterMonitor:
    """Tests for ClusterMonitor class."""

    def test_init_defaults(self):
        """Test ClusterMonitor initialization with defaults."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor()
            assert monitor.ssh_timeout == 15
            assert monitor.parallel is True

    def test_init_custom_timeout(self):
        """Test ClusterMonitor with custom timeout."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(ssh_timeout=30)
            assert monitor.ssh_timeout == 30

    def test_init_sequential_mode(self):
        """Test ClusterMonitor with parallel disabled."""
        with patch.object(ClusterMonitor, "_load_hosts"):
            monitor = ClusterMonitor(parallel=False)
            assert monitor.parallel is False


class TestNodeStatusEdgeCases:
    """Tests for edge cases in NodeStatus."""

    def test_zero_disk_metrics(self):
        """Test handling of zero disk metrics."""
        status = NodeStatus(
            host_name="node1",
            disk_usage_percent=0.0,
            disk_free_gb=0.0,
            disk_total_gb=0.0,
        )
        assert status.disk_usage_percent == 0.0

    def test_negative_response_time(self):
        """Test negative response time (edge case)."""
        # This shouldn't happen but should be handled
        status = NodeStatus(host_name="node1", response_time_ms=-1.0)
        assert status.response_time_ms == -1.0

    def test_large_game_counts(self):
        """Test handling of large game counts."""
        status = NodeStatus(host_name="node1", total_games=1_000_000)
        assert status.total_games == 1_000_000

    def test_multiple_training_processes(self):
        """Test multiple training processes."""
        processes = [
            {"pid": 1234, "command": "train hex8"},
            {"pid": 2345, "command": "train square8"},
            {"pid": 3456, "command": "train hexagonal"},
        ]
        status = NodeStatus(
            host_name="node1",
            training_active=True,
            training_processes=processes,
        )
        assert len(status.training_processes) == 3


class TestClusterStatusAggregation:
    """Tests for ClusterStatus aggregation logic."""

    def test_games_by_config_aggregation(self):
        """Test games are aggregated by config."""
        status = ClusterStatus(
            games_by_config={
                "hex8_2p": 1000,
                "hex8_3p": 500,
                "square8_2p": 2000,
                "square19_2p": 300,
            }
        )
        assert status.games_by_config["hex8_2p"] == 1000
        assert status.games_by_config["square8_2p"] == 2000
        assert sum(status.games_by_config.values()) == 3800

    def test_training_node_count(self):
        """Test training node count."""
        status = ClusterStatus(
            nodes_training=5,
            total_training_processes=12,
        )
        assert status.nodes_training == 5
        assert status.total_training_processes == 12


class TestNodeStatusSyncTracking:
    """Tests for sync tracking in NodeStatus."""

    def test_sync_lag_calculation(self):
        """Test sync lag tracking."""
        last_sync = datetime.now() - timedelta(minutes=5)
        status = NodeStatus(
            host_name="node1",
            last_sync_time=last_sync,
            sync_lag_seconds=300.0,  # 5 minutes
        )
        assert status.sync_lag_seconds == 300.0
        assert status.last_sync_time == last_sync

    def test_pending_files_tracking(self):
        """Test pending files count."""
        status = NodeStatus(host_name="node1", pending_files=42)
        assert status.pending_files == 42

    def test_no_sync_info(self):
        """Test node with no sync info."""
        status = NodeStatus(host_name="node1")
        assert status.last_sync_time is None
        assert status.sync_lag_seconds == 0.0
        assert status.pending_files == 0


class TestClusterStatusTimestamp:
    """Tests for timestamp handling."""

    def test_custom_timestamp(self):
        """Test setting custom timestamp."""
        ts = datetime(2025, 12, 25, 12, 0, 0)
        status = ClusterStatus(timestamp=ts)
        assert status.timestamp == ts

    def test_timestamp_comparison(self):
        """Test timestamp comparison for freshness."""
        old_status = ClusterStatus(timestamp=datetime.now() - timedelta(hours=1))
        new_status = ClusterStatus()
        assert new_status.timestamp > old_status.timestamp


class TestClusterStatusErrors:
    """Tests for error tracking in ClusterStatus."""

    def test_empty_errors(self):
        """Test empty errors list."""
        status = ClusterStatus()
        assert status.errors == []

    def test_multiple_errors(self):
        """Test multiple errors."""
        status = ClusterStatus(
            errors=[
                "Node A: Connection refused",
                "Node B: SSH timeout",
                "Node C: Permission denied",
            ]
        )
        assert len(status.errors) == 3
        assert "Connection refused" in status.errors[0]

    def test_errors_mutable(self):
        """Test errors list is mutable."""
        status = ClusterStatus()
        status.errors.append("New error")
        assert len(status.errors) == 1
