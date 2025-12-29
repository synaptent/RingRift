"""Unit tests for node_monitor module.

Tests the multi-layer node health monitoring daemon.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.node_monitor import (
    HealthCheckLayer,
    NodeHealthResult,
    NodeMonitor,
    NodeMonitorConfig,
)


# =============================================================================
# HealthCheckLayer Tests
# =============================================================================


class TestHealthCheckLayer:
    """Tests for HealthCheckLayer enum."""

    def test_has_p2p(self):
        """HealthCheckLayer should have P2P."""
        assert HealthCheckLayer.P2P.value == "p2p"

    def test_has_ssh(self):
        """HealthCheckLayer should have SSH."""
        assert HealthCheckLayer.SSH.value == "ssh"

    def test_has_gpu(self):
        """HealthCheckLayer should have GPU."""
        assert HealthCheckLayer.GPU.value == "gpu"

    def test_has_provider_api(self):
        """HealthCheckLayer should have PROVIDER_API."""
        assert HealthCheckLayer.PROVIDER_API.value == "provider_api"

    def test_has_all(self):
        """HealthCheckLayer should have ALL."""
        assert HealthCheckLayer.ALL.value == "all"

    def test_is_string_enum(self):
        """HealthCheckLayer should be string enum."""
        assert isinstance(HealthCheckLayer.P2P, str)
        assert HealthCheckLayer.P2P == "p2p"


# =============================================================================
# NodeHealthResult Tests
# =============================================================================


class TestNodeHealthResult:
    """Tests for NodeHealthResult dataclass."""

    def test_create_healthy_result(self):
        """Should create a healthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=15.5,
        )
        assert result.node_id == "test-node"
        assert result.layer == HealthCheckLayer.P2P
        assert result.healthy is True
        assert result.latency_ms == 15.5
        assert result.error is None

    def test_create_unhealthy_result(self):
        """Should create an unhealthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.SSH,
            healthy=False,
            latency_ms=0.0,
            error="Connection refused",
        )
        assert result.healthy is False
        assert result.error == "Connection refused"

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.GPU,
            healthy=True,
            latency_ms=50.0,
            details={"gpu_memory": 48},
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["layer"] == "gpu"
        assert d["healthy"] is True
        assert d["latency_ms"] == 50.0
        assert d["details"] == {"gpu_memory": 48}
        assert "timestamp" in d

    def test_default_timestamp(self):
        """Should have default timestamp."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=10.0,
        )
        assert isinstance(result.timestamp, datetime)

    def test_default_details(self):
        """Should have empty default details."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=10.0,
        )
        assert result.details == {}


# =============================================================================
# NodeMonitorConfig Tests
# =============================================================================


class TestNodeMonitorConfig:
    """Tests for NodeMonitorConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = NodeMonitorConfig()
        assert config.check_interval_seconds == 30
        assert config.p2p_timeout_seconds == 15.0
        assert config.ssh_timeout_seconds == 30.0
        assert config.gpu_check_enabled is True
        assert config.provider_check_enabled is True
        assert config.consecutive_failures_before_unhealthy == 3
        assert config.consecutive_failures_before_recovery == 5

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = NodeMonitorConfig(
            check_interval_seconds=60,
            p2p_timeout_seconds=20.0,
            gpu_check_enabled=False,
            consecutive_failures_before_unhealthy=5,
        )
        assert config.check_interval_seconds == 60
        assert config.p2p_timeout_seconds == 20.0
        assert config.gpu_check_enabled is False
        assert config.consecutive_failures_before_unhealthy == 5


# =============================================================================
# NodeMonitor Tests
# =============================================================================


class TestNodeMonitorInit:
    """Tests for NodeMonitor initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        monitor = NodeMonitor()
        assert monitor.config is not None
        assert isinstance(monitor.config, NodeMonitorConfig)

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = NodeMonitorConfig(p2p_timeout_seconds=10.0)
        monitor = NodeMonitor(config=config)
        assert monitor.config.p2p_timeout_seconds == 10.0

    def test_init_with_nodes(self):
        """Should accept initial node list."""
        mock_node = MagicMock()
        mock_node.name = "test-node"
        monitor = NodeMonitor(nodes=[mock_node])
        assert len(monitor._nodes) == 1

    def test_init_empty_failure_counts(self):
        """Should start with empty failure counts."""
        monitor = NodeMonitor()
        assert monitor._failure_counts == {}

    def test_init_empty_health_history(self):
        """Should start with empty health history."""
        monitor = NodeMonitor()
        assert monitor._health_history == {}

    def test_get_daemon_name(self):
        """Should return correct daemon name."""
        monitor = NodeMonitor()
        assert monitor._get_daemon_name() == "NodeMonitor"

    def test_get_default_config(self):
        """Should return NodeMonitorConfig."""
        monitor = NodeMonitor()
        config = monitor._get_default_config()
        assert isinstance(config, NodeMonitorConfig)


class TestNodeMonitorSetNodes:
    """Tests for set_nodes method."""

    def test_sets_nodes_list(self):
        """Should update nodes list."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"

        monitor.set_nodes([mock_node])

        assert len(monitor._nodes) == 1

    def test_initializes_failure_counts(self):
        """Should initialize failure counts for new nodes."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"

        monitor.set_nodes([mock_node])

        assert monitor._failure_counts["test-node"] == 0

    def test_initializes_health_history(self):
        """Should initialize health history for new nodes."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"

        monitor.set_nodes([mock_node])

        assert monitor._health_history["test-node"] == []

    def test_preserves_existing_counts(self):
        """Should preserve existing failure counts."""
        monitor = NodeMonitor()
        monitor._failure_counts["existing-node"] = 5

        mock_node = MagicMock()
        mock_node.name = "new-node"
        monitor.set_nodes([mock_node])

        assert monitor._failure_counts["existing-node"] == 5


class TestNodeMonitorHealthHistory:
    """Tests for health history tracking."""

    def test_empty_history_for_new_nodes(self):
        """New nodes should have empty history."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"
        monitor.set_nodes([mock_node])

        assert monitor._health_history["test-node"] == []

    def test_history_is_per_node(self):
        """History should be tracked per node."""
        monitor = NodeMonitor()
        mock_node1 = MagicMock()
        mock_node1.name = "node-1"
        mock_node2 = MagicMock()
        mock_node2.name = "node-2"

        monitor.set_nodes([mock_node1, mock_node2])

        assert "node-1" in monitor._health_history
        assert "node-2" in monitor._health_history
        assert monitor._health_history["node-1"] is not monitor._health_history["node-2"]


class TestNodeMonitorFailureCounts:
    """Tests for failure count tracking."""

    def test_counts_start_at_zero(self):
        """Failure counts should start at zero."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"
        monitor.set_nodes([mock_node])

        assert monitor._failure_counts["test-node"] == 0

    def test_can_increment_counts(self):
        """Should be able to increment failure counts."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"
        monitor.set_nodes([mock_node])

        monitor._failure_counts["test-node"] += 1
        assert monitor._failure_counts["test-node"] == 1

    def test_can_reset_counts(self):
        """Should be able to reset failure counts."""
        monitor = NodeMonitor()
        mock_node = MagicMock()
        mock_node.name = "test-node"
        monitor.set_nodes([mock_node])

        monitor._failure_counts["test-node"] = 5
        monitor._failure_counts["test-node"] = 0
        assert monitor._failure_counts["test-node"] == 0


class TestNodeMonitorRunCycle:
    """Tests for _run_cycle method."""

    @pytest.mark.asyncio
    async def test_handles_empty_nodes(self):
        """Should handle empty node list gracefully."""
        monitor = NodeMonitor()
        # Should not raise
        await monitor._run_cycle()


class TestNodeMonitorP2PPort:
    """Tests for P2P port configuration."""

    def test_default_p2p_port(self):
        """Should have default P2P port from config."""
        from app.config.ports import P2P_DEFAULT_PORT

        config = NodeMonitorConfig()
        assert config.p2p_port == P2P_DEFAULT_PORT

    def test_custom_p2p_port(self):
        """Should accept custom P2P port."""
        config = NodeMonitorConfig(p2p_port=9999)
        assert config.p2p_port == 9999


class TestNodeMonitorConsecutiveFailures:
    """Tests for consecutive failure thresholds."""

    def test_failures_before_unhealthy_default(self):
        """Default consecutive failures before unhealthy should be 3."""
        config = NodeMonitorConfig()
        assert config.consecutive_failures_before_unhealthy == 3

    def test_failures_before_recovery_default(self):
        """Default consecutive failures before recovery should be 5."""
        config = NodeMonitorConfig()
        assert config.consecutive_failures_before_recovery == 5

    def test_custom_failure_thresholds(self):
        """Should accept custom failure thresholds."""
        config = NodeMonitorConfig(
            consecutive_failures_before_unhealthy=5,
            consecutive_failures_before_recovery=10,
        )
        assert config.consecutive_failures_before_unhealthy == 5
        assert config.consecutive_failures_before_recovery == 10


class TestNodeMonitorTimeouts:
    """Tests for timeout configuration."""

    def test_p2p_timeout_default(self):
        """Default P2P timeout should be 15 seconds."""
        config = NodeMonitorConfig()
        assert config.p2p_timeout_seconds == 15.0

    def test_ssh_timeout_default(self):
        """Default SSH timeout should be 30 seconds."""
        config = NodeMonitorConfig()
        assert config.ssh_timeout_seconds == 30.0

    def test_custom_timeouts(self):
        """Should accept custom timeouts."""
        config = NodeMonitorConfig(
            p2p_timeout_seconds=10.0,
            ssh_timeout_seconds=20.0,
        )
        assert config.p2p_timeout_seconds == 10.0
        assert config.ssh_timeout_seconds == 20.0


class TestNodeMonitorFeatureFlags:
    """Tests for feature flags."""

    def test_gpu_check_enabled_default(self):
        """GPU check should be enabled by default."""
        config = NodeMonitorConfig()
        assert config.gpu_check_enabled is True

    def test_provider_check_enabled_default(self):
        """Provider check should be enabled by default."""
        config = NodeMonitorConfig()
        assert config.provider_check_enabled is True

    def test_can_disable_gpu_check(self):
        """Should be able to disable GPU check."""
        config = NodeMonitorConfig(gpu_check_enabled=False)
        assert config.gpu_check_enabled is False

    def test_can_disable_provider_check(self):
        """Should be able to disable provider check."""
        config = NodeMonitorConfig(provider_check_enabled=False)
        assert config.provider_check_enabled is False
