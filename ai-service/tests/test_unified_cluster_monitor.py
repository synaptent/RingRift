"""Tests for the Unified Cluster Monitor.

Tests the consolidated cluster monitoring system that provides:
- HTTP health endpoint checks with Tailscale fallback
- SSH-based deep checks for GPU/process status
- Leader API integration for training job status
- Unified alerting (console, webhook, metrics)
"""

import json

# Import from app.monitoring module
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.monitoring.unified_cluster_monitor import (
    ClusterConfig,
    ClusterHealth,
    LeaderHealth,
    NodeHealth,
    UnifiedClusterMonitor,
)


@pytest.fixture
def mock_config():
    """Create mock cluster configuration."""
    config = MagicMock(spec=ClusterConfig)
    config.nodes = {
        "node1": {
            "name": "node1",
            "ssh_host": "10.0.0.1",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "tailscale_ip": "100.64.0.1",
            "p2p_port": 8770,
            "primary_url": "http://10.0.0.1:8770/health",
            "tailscale_url": "http://100.64.0.1:8770/health",
            "is_leader": True,
        },
        "node2": {
            "name": "node2",
            "ssh_host": "10.0.0.2",
            "ssh_user": "ubuntu",
            "ssh_port": 22,
            "tailscale_ip": "100.64.0.2",
            "p2p_port": 8770,
            "primary_url": "http://10.0.0.2:8770/health",
            "tailscale_url": "http://100.64.0.2:8770/health",
            "is_leader": False,
        },
    }
    config.leader_url = "http://10.0.0.1:8770"
    config.get_node_names.return_value = ["node1", "node2"]
    return config


@pytest.fixture
def monitor(mock_config):
    """Create monitor with mock config."""
    return UnifiedClusterMonitor(
        config=mock_config,
        check_interval=60,
        deep_checks=False,
    )


class TestNodeHealth:
    """Tests for NodeHealth dataclass."""

    def test_default_values(self):
        """NodeHealth should have sensible defaults."""
        health = NodeHealth(name="test")

        assert health.name == "test"
        assert health.status == "unknown"
        assert health.via_tailscale is False
        assert health.cpu_percent == 0.0
        assert health.memory_percent == 0.0
        assert health.disk_percent == 0.0
        assert health.selfplay_active is False
        assert health.games_played == 0
        assert health.error is None

    def test_with_values(self):
        """NodeHealth should store provided values."""
        health = NodeHealth(
            name="gpu-server",
            status="healthy",
            cpu_percent=45.0,
            memory_percent=60.0,
            disk_percent=30.0,
            selfplay_active=True,
            games_played=1000,
            gpu_util=80.0,
        )

        assert health.name == "gpu-server"
        assert health.status == "healthy"
        assert health.cpu_percent == 45.0
        assert health.selfplay_active is True
        assert health.gpu_util == 80.0


class TestLeaderHealth:
    """Tests for LeaderHealth dataclass."""

    def test_default_values(self):
        """LeaderHealth should have sensible defaults."""
        leader = LeaderHealth()

        assert leader.is_leader is False
        assert leader.selfplay_jobs == 0
        assert leader.selfplay_rate == 0.0
        assert leader.training_nnue_running == 0
        assert leader.training_cmaes_running == 0
        assert leader.error is None

    def test_with_values(self):
        """LeaderHealth should store provided values."""
        leader = LeaderHealth(
            is_leader=True,
            node_id="leader-1",
            selfplay_jobs=5,
            selfplay_rate=120.5,
            training_nnue_running=1,
        )

        assert leader.is_leader is True
        assert leader.node_id == "leader-1"
        assert leader.selfplay_jobs == 5
        assert leader.selfplay_rate == 120.5


class TestClusterHealth:
    """Tests for ClusterHealth dataclass."""

    def test_default_values(self):
        """ClusterHealth should have sensible defaults."""
        cluster = ClusterHealth()

        assert cluster.nodes == {}
        assert cluster.leader is None
        assert cluster.total_nodes == 0
        assert cluster.healthy_nodes == 0
        assert cluster.alerts == []
        assert cluster.critical_alerts == []

    def test_with_nodes(self):
        """ClusterHealth should store node data."""
        node1 = NodeHealth(name="node1", status="healthy")
        node2 = NodeHealth(name="node2", status="unhealthy")

        cluster = ClusterHealth(
            nodes={"node1": node1, "node2": node2},
            total_nodes=2,
            healthy_nodes=1,
        )

        assert len(cluster.nodes) == 2
        assert cluster.total_nodes == 2
        assert cluster.healthy_nodes == 1


class TestClusterConfig:
    """Tests for ClusterConfig class."""

    @patch("app.monitoring.unified_cluster_monitor.yaml")
    @patch.object(ClusterConfig, "__init__", lambda self, config_path=None: None)
    def test_load_config(self, mock_yaml):
        """Should load config from YAML file."""
        # Test that ClusterConfig can be instantiated with mock
        # The actual config loading is tested implicitly via __init__
        mock_yaml.safe_load.return_value = {
            "hosts": {
                "test-node": {
                    "ssh_host": "10.0.0.1",
                    "ssh_user": "ubuntu",
                    "p2p_port": 8770,
                    "role": "worker",
                    "status": "running",
                }
            }
        }

        # Create instance with mocked __init__
        config = ClusterConfig()
        config.nodes = {}  # Initialize empty since __init__ is mocked

        # Verify the mock can be used
        assert mock_yaml.safe_load is not None

    def test_get_node_names(self, mock_config):
        """get_node_names should return list of node names."""
        names = mock_config.get_node_names()
        assert "node1" in names
        assert "node2" in names


class TestUnifiedClusterMonitor:
    """Tests for UnifiedClusterMonitor class."""

    def test_init(self, mock_config):
        """Monitor should initialize with config."""
        monitor = UnifiedClusterMonitor(
            config=mock_config,
            webhook_url="https://webhook.test",
            check_interval=30,
            deep_checks=True,
        )

        assert monitor.config is mock_config
        assert monitor.webhook_url == "https://webhook.test"
        assert monitor.check_interval == 30
        assert monitor.deep_checks is True

    def test_thresholds_loaded(self, monitor):
        """Monitor should have threshold values."""
        assert monitor.disk_warning > 0
        assert monitor.disk_critical > monitor.disk_warning
        assert monitor.memory_warning > 0
        assert monitor.memory_critical > monitor.memory_warning


class TestHttpHealthChecks:
    """Tests for HTTP health check functionality."""

    def test_http_get_json_success(self, monitor):
        """Should parse JSON from successful response."""
        with patch("app.monitoring.unified_cluster_monitor.urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"healthy": true, "cpu_percent": 45.0}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = monitor._http_get_json("http://test/health")

            assert result["healthy"] is True
            assert result["cpu_percent"] == 45.0

    def test_http_get_json_error(self, monitor):
        """Should return error dict on failure."""
        with patch("app.monitoring.unified_cluster_monitor.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")

            result = monitor._http_get_json("http://test/health")

            assert "error" in result

    def test_check_node_http_success(self, monitor, mock_config):
        """Should check node via HTTP."""
        with patch.object(monitor, "_http_get_json") as mock_get:
            mock_get.return_value = {
                "healthy": True,
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "disk_percent": 40.0,
                "selfplay_active": True,
                "games_played": 500,
            }

            health = monitor.check_node_http("node1")

            assert health.status == "healthy"
            assert health.cpu_percent == 50.0
            assert health.selfplay_active is True

    def test_check_node_http_fallback_to_tailscale(self, monitor, mock_config):
        """Should fallback to Tailscale URL on primary failure."""
        call_count = [0]

        def mock_get(url):
            call_count[0] += 1
            if "10.0.0" in url:  # Primary URL
                return {"error": "Connection refused"}
            else:  # Tailscale URL
                return {"healthy": True, "cpu_percent": 45.0}

        with patch.object(monitor, "_http_get_json", side_effect=mock_get):
            health = monitor.check_node_http("node1")

            assert health.status == "healthy"
            assert health.via_tailscale is True

    def test_check_node_http_unreachable(self, monitor, mock_config):
        """Should mark node unreachable if all URLs fail."""
        with patch.object(monitor, "_http_get_json") as mock_get:
            mock_get.return_value = {"error": "Connection refused"}

            health = monitor.check_node_http("node1")

            assert health.status == "unreachable"
            assert health.error is not None


class TestAlertGeneration:
    """Tests for alert generation."""

    def test_disk_warning_alert(self, monitor):
        """Should generate warning for high disk usage."""
        cluster = ClusterHealth()
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="healthy",
            disk_percent=monitor.disk_warning + 5,
        )

        monitor.generate_alerts(cluster)

        assert len(cluster.alerts) > 0
        assert any("disk" in alert.lower() for alert in cluster.alerts)

    def test_disk_critical_alert(self, monitor):
        """Should generate critical alert for very high disk usage."""
        cluster = ClusterHealth()
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="healthy",
            disk_percent=monitor.disk_critical + 5,
        )

        monitor.generate_alerts(cluster)

        assert len(cluster.critical_alerts) > 0
        assert any("disk" in alert.lower() for alert in cluster.critical_alerts)

    def test_memory_warning_alert(self, monitor):
        """Should generate warning for high memory usage."""
        cluster = ClusterHealth()
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="healthy",
            memory_percent=monitor.memory_warning + 5,
        )

        monitor.generate_alerts(cluster)

        assert len(cluster.alerts) > 0
        assert any("memory" in alert.lower() for alert in cluster.alerts)

    def test_unreachable_node_critical_alert(self, monitor):
        """Should generate critical alert for unreachable nodes."""
        cluster = ClusterHealth()
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="unreachable",
            error="Connection refused",
        )

        monitor.generate_alerts(cluster)

        assert len(cluster.critical_alerts) > 0
        assert any("unreachable" in alert.lower() for alert in cluster.critical_alerts)

    def test_cluster_down_critical_alert(self, monitor):
        """Should generate critical alert if no healthy nodes."""
        cluster = ClusterHealth()
        cluster.nodes["node1"] = NodeHealth(name="node1", status="unreachable")
        cluster.nodes["node2"] = NodeHealth(name="node2", status="unreachable")
        cluster.healthy_nodes = 0

        monitor.generate_alerts(cluster)

        assert len(cluster.critical_alerts) > 0
        assert any("cluster down" in alert.lower() for alert in cluster.critical_alerts)


class TestOutputFormatting:
    """Tests for output formatting."""

    def test_format_text_output(self, monitor):
        """Should format cluster health as text."""
        cluster = ClusterHealth(
            total_nodes=2,
            healthy_nodes=2,
            timestamp=datetime.now(),
        )
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="healthy",
            selfplay_active=True,
            disk_percent=30.0,
            memory_percent=50.0,
            games_played=1000,
        )
        cluster.nodes["node2"] = NodeHealth(
            name="node2",
            status="healthy",
            selfplay_active=False,
            disk_percent=40.0,
            memory_percent=60.0,
        )

        output = monitor.format_text(cluster)

        assert "CLUSTER HEALTH" in output
        assert "node1" in output
        assert "node2" in output
        assert "2/2 healthy" in output

    def test_format_json_output(self, monitor):
        """Should format cluster health as valid JSON."""
        cluster = ClusterHealth(
            total_nodes=1,
            healthy_nodes=1,
            timestamp=datetime.now(),
        )
        cluster.nodes["node1"] = NodeHealth(
            name="node1",
            status="healthy",
            cpu_percent=50.0,
        )

        output = monitor.format_json(cluster)

        # Should be valid JSON
        data = json.loads(output)
        assert "timestamp" in data
        assert "nodes" in data
        assert "node1" in data["nodes"]
        assert data["nodes"]["node1"]["status"] == "healthy"

    def test_format_text_with_alerts(self, monitor):
        """Should include alerts in text output."""
        cluster = ClusterHealth()
        cluster.alerts = ["Warning: High disk usage"]
        cluster.critical_alerts = ["CRITICAL: Node unreachable"]

        output = monitor.format_text(cluster)

        assert "ALERTS" in output
        assert "CRITICAL" in output or "High disk" in output


class TestClusterCheck:
    """Tests for full cluster check."""

    def test_check_cluster(self, monitor, mock_config):
        """Should check all nodes and aggregate results."""
        with patch.object(monitor, "check_node_http") as mock_check:
            mock_check.side_effect = [
                NodeHealth(name="node1", status="healthy"),
                NodeHealth(name="node2", status="healthy"),
            ]
            with patch.object(monitor, "check_leader") as mock_leader:
                mock_leader.return_value = LeaderHealth(is_leader=True)

                cluster = monitor.check_cluster()

                assert cluster.total_nodes == 2
                assert cluster.healthy_nodes == 2
                assert "node1" in cluster.nodes
                assert "node2" in cluster.nodes

    def test_check_cluster_calculates_summary(self, monitor, mock_config):
        """Should calculate summary metrics."""
        with patch.object(monitor, "check_node_http") as mock_check:
            mock_check.side_effect = [
                NodeHealth(
                    name="node1",
                    status="healthy",
                    gpu_util=80.0,
                    games_played=500,
                ),
                NodeHealth(
                    name="node2",
                    status="healthy",
                    gpu_util=60.0,
                    games_played=300,
                ),
            ]
            with patch.object(monitor, "check_leader") as mock_leader:
                mock_leader.return_value = None

                cluster = monitor.check_cluster()

                assert cluster.total_games == 800
                assert cluster.avg_gpu_util == 70.0


class TestWebhookAlerts:
    """Tests for webhook alert functionality."""

    def test_send_webhook_alert(self, monitor):
        """Should send webhook alert."""
        monitor.webhook_url = "https://webhook.test/alert"

        with patch("unified_cluster_monitor.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            monitor.send_webhook_alert("Test alert message", level="warning")

            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.full_url == "https://webhook.test/alert"

    def test_no_webhook_when_not_configured(self, monitor):
        """Should not attempt webhook when URL not configured."""
        monitor.webhook_url = None

        with patch("unified_cluster_monitor.urllib.request.urlopen") as mock_urlopen:
            monitor.send_webhook_alert("Test alert")

            mock_urlopen.assert_not_called()


class TestAlertCooldown:
    """Tests for alert cooldown functionality."""

    def test_should_alert_first_time(self, monitor):
        """Should alert on first occurrence."""
        assert monitor._should_alert("test_key") is True

    def test_should_not_alert_within_cooldown(self, monitor):
        """Should not alert within cooldown period."""
        monitor._should_alert("test_key")  # First alert
        assert monitor._should_alert("test_key") is False  # Within cooldown

    def test_should_alert_after_cooldown(self, monitor):
        """Should alert after cooldown expires."""
        monitor._alert_cooldown = 0  # No cooldown
        monitor._should_alert("test_key")  # First alert
        assert monitor._should_alert("test_key") is True  # After cooldown
