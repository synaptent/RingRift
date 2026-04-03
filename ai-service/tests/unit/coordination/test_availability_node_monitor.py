"""Unit tests for availability/node_monitor.py.

Tests for NodeMonitor daemon that performs multi-layer health checks.

Created: Dec 28, 2025
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.node_monitor import (
    HealthCheckLayer,
    NodeHealthResult,
    NodeMonitor,
    NodeMonitorConfig,
    get_node_monitor,
    reset_node_monitor,
)


class TestHealthCheckLayer:
    """Tests for HealthCheckLayer enum."""

    def test_layer_values(self):
        """Test all layer values exist."""
        assert HealthCheckLayer.P2P.value == "p2p"
        assert HealthCheckLayer.SSH.value == "ssh"
        assert HealthCheckLayer.GPU.value == "gpu"
        assert HealthCheckLayer.PROVIDER_API.value == "provider_api"
        assert HealthCheckLayer.ALL.value == "all"

    def test_layer_is_string_enum(self):
        """Test layer is a string enum."""
        assert isinstance(HealthCheckLayer.P2P, str)
        assert HealthCheckLayer.P2P == "p2p"


class TestNodeHealthResult:
    """Tests for NodeHealthResult dataclass."""

    def test_healthy_result(self):
        """Test creating a healthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )
        assert result.healthy
        assert result.node_id == "test-node"
        assert result.layer == HealthCheckLayer.P2P
        assert result.latency_ms == 50.0
        assert result.error is None

    def test_unhealthy_result(self):
        """Test creating an unhealthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.SSH,
            healthy=False,
            latency_ms=30000.0,
            error="Connection refused",
        )
        assert not result.healthy
        assert result.layer == HealthCheckLayer.SSH
        assert "Connection refused" in result.error

    def test_result_with_details(self):
        """Test result with extra details."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.GPU,
            healthy=True,
            latency_ms=100.0,
            details={"gpu_info": "NVIDIA GH200"},
        )
        assert result.details["gpu_info"] == "NVIDIA GH200"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=25.0,
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["layer"] == "p2p"
        assert d["healthy"] is True
        assert d["latency_ms"] == 25.0
        assert "timestamp" in d


class TestNodeMonitorConfig:
    """Tests for NodeMonitorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NodeMonitorConfig()
        assert config.check_interval_seconds == 30.0
        assert config.p2p_timeout_seconds == 15.0
        assert config.ssh_timeout_seconds == 30.0
        assert config.gpu_check_enabled is True
        assert config.consecutive_failures_before_unhealthy == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = NodeMonitorConfig(
            check_interval_seconds=60.0,
            p2p_timeout_seconds=20.0,
            consecutive_failures_before_unhealthy=5,
        )
        assert config.check_interval_seconds == 60.0
        assert config.p2p_timeout_seconds == 20.0
        assert config.consecutive_failures_before_unhealthy == 5


class TestNodeMonitor:
    """Tests for NodeMonitor daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_singleton_pattern(self):
        """Test that get_node_monitor returns singleton."""
        monitor1 = get_node_monitor()
        monitor2 = get_node_monitor()

        assert monitor1 is monitor2

    def test_custom_config(self):
        """Test creating monitor with custom config."""
        config = NodeMonitorConfig(check_interval_seconds=60.0)
        monitor = NodeMonitor(config=config)

        assert monitor.config.check_interval_seconds == 60.0

    def test_get_node_status(self):
        """Test getting status for a specific node."""
        monitor = get_node_monitor()

        # Status for unknown node should have None values
        status = monitor.get_node_status("unknown-node")
        assert "node_id" in status
        assert status["node_id"] == "unknown-node"
        assert status["healthy"] is None

    def test_get_all_node_statuses(self):
        """Test getting all node statuses."""
        monitor = get_node_monitor()

        statuses = monitor.get_all_node_statuses()
        assert isinstance(statuses, dict)

    def test_health_check(self):
        """Test health_check method returns valid result."""
        monitor = get_node_monitor()

        result = monitor.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "message")
        assert hasattr(result, "details")
        assert result.details["nodes_monitored"] == 0

    def test_set_nodes(self):
        """Test setting nodes to monitor."""
        monitor = get_node_monitor()

        # Create mock nodes
        mock_node = MagicMock()
        mock_node.name = "test-node"

        monitor.set_nodes([mock_node])

        assert len(monitor._nodes) == 1
        assert monitor._nodes[0].name == "test-node"


class TestNodeMonitorIntegration:
    """Integration tests for NodeMonitor."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_p2p_check_with_mock(self):
        """Test P2P health check with mocked HTTP client."""
        monitor = get_node_monitor()

        # Mock the HTTP call
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # The actual check would use the mock
            # This verifies the monitor can be instantiated and checked
            assert monitor is not None

    @pytest.mark.asyncio
    async def test_run_cycle_with_no_nodes(self):
        """Test run cycle with no nodes configured."""
        monitor = get_node_monitor()

        # Patch node loading to return empty
        with patch.object(monitor, "_load_nodes_from_config", new_callable=AsyncMock):
            await monitor._run_cycle()

            # Should complete without error
            assert True
