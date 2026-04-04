"""Tests for swim_adapter.py module.

Comprehensive tests for SWIM-based membership management:
- SwimBootstrapConfig dataclass
- SwimConfig dataclass
- SwimMembershipManager initialization and configuration
- Factory methods and configuration loading
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.p2p.swim_adapter import (
    SWIM_AVAILABLE,
    SWIM_FAILURE_TIMEOUT,
    SWIM_INDIRECT_PING_COUNT,
    SWIM_PING_INTERVAL,
    SWIM_SUSPICION_TIMEOUT,
    SwimBootstrapConfig,
    SwimConfig,
    SwimMembershipManager,
)


# =============================================================================
# SwimBootstrapConfig Tests
# =============================================================================


class TestSwimBootstrapConfig:
    """Tests for SwimBootstrapConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = SwimBootstrapConfig()
        assert config.max_attempts == 5
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.seed_rotation is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = SwimBootstrapConfig(
            max_attempts=10,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            backoff_multiplier=1.5,
            seed_rotation=False,
        )
        assert config.max_attempts == 10
        assert config.initial_delay_seconds == 2.0
        assert config.max_delay_seconds == 60.0
        assert config.backoff_multiplier == 1.5
        assert config.seed_rotation is False

    def test_exponential_backoff_calculation(self):
        """Verify exponential backoff formula with config values."""
        config = SwimBootstrapConfig(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            max_delay_seconds=30.0,
        )
        # Delay for attempt N = initial * (multiplier ^ N), capped at max
        delays = []
        for attempt in range(5):
            delay = min(
                config.initial_delay_seconds * (config.backoff_multiplier ** attempt),
                config.max_delay_seconds,
            )
            delays.append(delay)

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]


# =============================================================================
# SwimConfig Tests
# =============================================================================


class TestSwimConfig:
    """Tests for SwimConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = SwimConfig()
        assert config.bind_host == "0.0.0.0"
        # bind_port comes from SWIM_PORT constant
        assert isinstance(config.bind_port, int)
        assert config.failure_timeout == SWIM_FAILURE_TIMEOUT
        assert config.suspicion_timeout == SWIM_SUSPICION_TIMEOUT
        assert config.ping_interval == SWIM_PING_INTERVAL
        assert config.ping_request_group_size == SWIM_INDIRECT_PING_COUNT
        assert config.max_transmissions == 10
        assert config.seeds == []
        assert isinstance(config.bootstrap, SwimBootstrapConfig)

    def test_custom_values(self):
        """Should accept custom values."""
        config = SwimConfig(
            bind_host="127.0.0.1",
            bind_port=9000,
            failure_timeout=10.0,
            suspicion_timeout=5.0,
            ping_interval=2.0,
            seeds=[("192.168.1.1", 9000), ("192.168.1.2", 9000)],
        )
        assert config.bind_host == "127.0.0.1"
        assert config.bind_port == 9000
        assert config.failure_timeout == 10.0
        assert len(config.seeds) == 2

    def test_embedded_bootstrap_config(self):
        """Should allow custom bootstrap config."""
        bootstrap = SwimBootstrapConfig(max_attempts=10)
        config = SwimConfig(bootstrap=bootstrap)
        assert config.bootstrap.max_attempts == 10


# =============================================================================
# SwimMembershipManager Tests
# =============================================================================


class TestSwimMembershipManagerInit:
    """Tests for SwimMembershipManager initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        manager = SwimMembershipManager(node_id="test-node")
        assert manager.node_id == "test-node"
        assert isinstance(manager.config, SwimConfig)
        assert manager._started is False
        assert manager._swim is None

    def test_init_custom_port(self):
        """Should accept custom port."""
        manager = SwimMembershipManager(node_id="test-node", bind_port=9999)
        assert manager.config.bind_port == 9999

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = SwimConfig(
            failure_timeout=10.0,
            seeds=[("192.168.1.1", 7947)],
        )
        manager = SwimMembershipManager(node_id="test-node", config=config)
        assert manager.config.failure_timeout == 10.0
        assert len(manager.config.seeds) == 1

    def test_init_with_callbacks(self):
        """Should accept callback functions."""
        alive_cb = MagicMock()
        failed_cb = MagicMock()

        manager = SwimMembershipManager(
            node_id="test-node",
            on_member_alive=alive_cb,
            on_member_failed=failed_cb,
        )

        assert manager.on_member_alive is alive_cb
        assert manager.on_member_failed is failed_cb


class TestSwimMembershipManagerFactory:
    """Tests for SwimMembershipManager factory methods."""

    def test_from_distributed_hosts_with_cluster_config(self):
        """Should use cluster_config when available."""
        mock_nodes = {
            "node-1": MagicMock(best_ip="10.0.0.1", is_active=True),
            "node-2": MagicMock(best_ip="10.0.0.2", is_active=True),
            "node-3": MagicMock(best_ip="10.0.0.3", is_active=True),
        }

        with patch("app.p2p.swim_adapter.HAS_CLUSTER_CONFIG", True), \
             patch("app.p2p.swim_adapter.get_p2p_voters", return_value=["node-1", "node-2", "node-3"]), \
             patch("app.p2p.swim_adapter.get_cluster_nodes", return_value=mock_nodes):

            manager = SwimMembershipManager.from_distributed_hosts(
                node_id="node-1",
                bind_port=7947,
            )

        # Should have seeds for other nodes (not self)
        seeds = manager.config.seeds
        assert len(seeds) >= 2
        # Seeds should be tuples of (host, port)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in seeds)
        # Self should be excluded
        assert ("10.0.0.1", 7947) not in seeds

    def test_from_distributed_hosts_yaml_fallback(self):
        """Should fall back to YAML when cluster_config unavailable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
hosts:
  node-1:
    ssh_host: 192.168.1.1
    voter: true
    p2p_enabled: true
  node-2:
    ssh_host: 192.168.1.2
    voter: true
    p2p_enabled: true
  node-3:
    ssh_host: 192.168.1.3
    voter: false
    p2p_enabled: true
""")
            config_path = Path(f.name)

        try:
            with patch("app.p2p.swim_adapter.HAS_CLUSTER_CONFIG", False):
                manager = SwimMembershipManager.from_distributed_hosts(
                    node_id="node-1",
                    config_path=config_path,
                    bind_port=7947,
                )

            # Should have seed for voter node-2 (not self node-1)
            seeds = manager.config.seeds
            assert ("192.168.1.2", 7947) in seeds
            assert ("192.168.1.1", 7947) not in seeds  # Self excluded
        finally:
            config_path.unlink()

    def test_from_distributed_hosts_missing_file(self):
        """Should return manager with empty seeds for missing file."""
        with patch("app.p2p.swim_adapter.HAS_CLUSTER_CONFIG", False):
            manager = SwimMembershipManager.from_distributed_hosts(
                node_id="node-1",
                config_path=Path("/nonexistent/config.yaml"),
            )

        assert manager.config.seeds == []


class TestSwimMembershipManagerState:
    """Tests for SwimMembershipManager state management."""

    def test_initial_state(self):
        """Should start in not-started state."""
        manager = SwimMembershipManager(node_id="test-node")
        assert manager._started is False
        assert manager._swim is None
        assert manager._members == {}

    def test_is_healthy_when_not_started(self):
        """Should return False when not started."""
        manager = SwimMembershipManager(node_id="test-node")
        assert manager.is_healthy() is False

    def test_get_health_status_when_not_started(self):
        """Should return appropriate status when not started."""
        manager = SwimMembershipManager(node_id="test-node")
        status = manager.get_health_status()

        assert status["healthy"] is False
        assert status["started"] is False
        assert "reason" in status

    def test_get_health_status_swim_not_available(self):
        """Should indicate swim-p2p not installed."""
        manager = SwimMembershipManager(node_id="test-node")

        with patch("app.p2p.swim_adapter.SWIM_AVAILABLE", False):
            status = manager.get_health_status()

        assert status["healthy"] is False
        assert status["swim_available"] is False
        assert "not installed" in status["reason"]

    def test_get_alive_peers_when_not_started(self):
        """Should return empty list when not started."""
        manager = SwimMembershipManager(node_id="test-node")
        assert manager.get_alive_peers() == []


class TestSwimMembershipManagerCallbacks:
    """Tests for callback handling."""

    def test_handle_member_alive_callback(self):
        """Should call on_member_alive callback."""
        callback = MagicMock()
        manager = SwimMembershipManager(
            node_id="test-node",
            on_member_alive=callback,
        )

        # Create mock member
        mock_member = MagicMock()
        mock_member.id = "other-node"

        manager._handle_member_alive(mock_member)

        callback.assert_called_once_with("other-node")
        assert manager._members["other-node"] == "alive"

    def test_handle_member_failed_callback(self):
        """Failed members are tracked locally without firing liveness callbacks."""
        callback = MagicMock()
        manager = SwimMembershipManager(
            node_id="test-node",
            on_member_failed=callback,
        )

        mock_member = MagicMock()
        mock_member.id = "failed-node"

        manager._handle_member_failed(mock_member)

        callback.assert_not_called()
        assert manager._members["failed-node"] == "failed"

    def test_callback_exception_handling(self):
        """Should handle exceptions in callbacks gracefully."""

        def bad_callback(node_id):
            raise ValueError("Callback error")

        manager = SwimMembershipManager(
            node_id="test-node",
            on_member_alive=bad_callback,
        )

        mock_member = MagicMock()
        mock_member.id = "other-node"

        # Should not raise
        manager._handle_member_alive(mock_member)

        # State should still be updated
        assert manager._members["other-node"] == "alive"


# =============================================================================
# SWIM Availability Tests
# =============================================================================


class TestSwimAvailability:
    """Tests for SWIM package availability handling."""

    def test_swim_available_constant(self):
        """SWIM_AVAILABLE should be a boolean."""
        assert isinstance(SWIM_AVAILABLE, bool)

    @pytest.mark.asyncio
    async def test_start_without_swim(self):
        """Should fail gracefully when swim-p2p not installed."""
        manager = SwimMembershipManager(node_id="test-node")

        with patch("app.p2p.swim_adapter.SWIM_AVAILABLE", False):
            result = await manager.start()

        assert result is False
        assert manager._started is False


# =============================================================================
# Seed Rotation Tests
# =============================================================================


class TestSeedRotation:
    """Tests for seed rotation during bootstrap."""

    def test_seed_rotation_logic(self):
        """Verify seed rotation algorithm."""
        seeds = [("10.0.0.1", 7947), ("10.0.0.2", 7947), ("10.0.0.3", 7947)]

        # Attempt 0: no rotation
        rotated_0 = seeds[0:] + seeds[:0]
        assert rotated_0 == seeds

        # Attempt 1: rotate by 1
        rotation = 1 % len(seeds)
        rotated_1 = seeds[rotation:] + seeds[:rotation]
        assert rotated_1[0] == ("10.0.0.2", 7947)

        # Attempt 2: rotate by 2
        rotation = 2 % len(seeds)
        rotated_2 = seeds[rotation:] + seeds[:rotation]
        assert rotated_2[0] == ("10.0.0.3", 7947)

        # Attempt 3: wraps around
        rotation = 3 % len(seeds)
        rotated_3 = seeds[rotation:] + seeds[:rotation]
        assert rotated_3[0] == ("10.0.0.1", 7947)
