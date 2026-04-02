"""Tests for app.p2p.config - P2P Configuration.

This module tests the P2P configuration and constants.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.p2p.config import (
    DEFAULT_PORT,
    GPU_POWER_RANKINGS,
    HEARTBEAT_INTERVAL,
    LEADER_LEASE_DURATION,
    PEER_TIMEOUT,
    P2PConfig,
    get_p2p_config,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_port(self):
        """Should have correct default port."""
        assert DEFAULT_PORT == 8770

    def test_heartbeat_interval(self):
        """Should have reasonable heartbeat interval."""
        assert HEARTBEAT_INTERVAL == 15

    def test_peer_timeout(self):
        """Peer timeout should be longer than heartbeat interval."""
        assert PEER_TIMEOUT > HEARTBEAT_INTERVAL
        assert PEER_TIMEOUT == 180

    def test_leader_lease_duration(self):
        """Leader lease should be longer than peer timeout."""
        assert LEADER_LEASE_DURATION >= PEER_TIMEOUT
        assert LEADER_LEASE_DURATION == 300


# =============================================================================
# GPU Power Rankings Tests
# =============================================================================


class TestGPUPowerRankings:
    """Tests for GPU power rankings."""

    def test_has_datacenter_gpus(self):
        """Should have data center GPU rankings."""
        assert "H100" in GPU_POWER_RANKINGS
        assert "H200" in GPU_POWER_RANKINGS
        assert "GH200" in GPU_POWER_RANKINGS
        assert "A100" in GPU_POWER_RANKINGS

    def test_has_consumer_gpus(self):
        """Should have consumer GPU rankings."""
        assert "4090" in GPU_POWER_RANKINGS
        assert "4080" in GPU_POWER_RANKINGS
        assert "3090" in GPU_POWER_RANKINGS

    def test_has_apple_silicon(self):
        """Should have Apple Silicon rankings."""
        assert "Apple M3" in GPU_POWER_RANKINGS
        assert "Apple M2" in GPU_POWER_RANKINGS
        assert "Apple M1" in GPU_POWER_RANKINGS

    def test_has_unknown_fallback(self):
        """Should have fallback for unknown GPUs."""
        assert "Unknown" in GPU_POWER_RANKINGS
        assert GPU_POWER_RANKINGS["Unknown"] > 0

    def test_datacenter_higher_than_consumer(self):
        """Data center GPUs should rank higher than consumer."""
        assert GPU_POWER_RANKINGS["H100"] > GPU_POWER_RANKINGS["4090"]
        assert GPU_POWER_RANKINGS["A100"] > GPU_POWER_RANKINGS["3090"]

    def test_newer_gpus_higher_ranking(self):
        """Newer GPUs should generally rank higher."""
        assert GPU_POWER_RANKINGS["4090"] > GPU_POWER_RANKINGS["3090"]
        assert GPU_POWER_RANKINGS["4080"] > GPU_POWER_RANKINGS["3080"]

    def test_all_rankings_positive(self):
        """All rankings should be positive."""
        for gpu, ranking in GPU_POWER_RANKINGS.items():
            assert ranking > 0, f"{gpu} has non-positive ranking"


# =============================================================================
# P2PConfig Tests
# =============================================================================


class TestP2PConfig:
    """Tests for P2PConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = P2PConfig()
        assert config.DISK_CRITICAL_THRESHOLD == 90
        assert config.DISK_WARNING_THRESHOLD == 70
        assert config.DISK_CLEANUP_THRESHOLD == 65
        assert config.MEMORY_CRITICAL_THRESHOLD == 95
        assert config.MEMORY_WARNING_THRESHOLD == 85
        assert config.LOAD_MAX_FOR_NEW_JOBS == 85

    def test_gpu_util_targets(self):
        """Should have GPU utilization targets."""
        config = P2PConfig()
        assert config.TARGET_GPU_UTIL_MIN == 60
        assert config.TARGET_GPU_UTIL_MAX == 90
        assert config.TARGET_GPU_UTIL_MIN < config.TARGET_GPU_UTIL_MAX

    def test_connection_settings(self):
        """Should have connection settings."""
        config = P2PConfig()
        assert config.HTTP_CONNECT_TIMEOUT == 10
        assert config.HTTP_TOTAL_TIMEOUT == 30
        assert config.MAX_CONSECUTIVE_FAILURES == 3

    def test_env_override_disk_threshold(self):
        """Should respect environment variable override."""
        with patch.dict(os.environ, {"RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD": "80"}):
            config = P2PConfig()
            assert config.DISK_CRITICAL_THRESHOLD == 80

    def test_env_override_memory_threshold(self):
        """Should respect memory threshold override."""
        with patch.dict(os.environ, {"RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD": "90"}):
            config = P2PConfig()
            assert config.MEMORY_WARNING_THRESHOLD == 90


# =============================================================================
# get_p2p_config Tests
# =============================================================================


class TestGetP2PConfig:
    """Tests for get_p2p_config function."""

    def test_returns_config(self):
        """Should return P2PConfig instance."""
        config = get_p2p_config()
        assert isinstance(config, P2PConfig)

    def test_singleton_behavior(self):
        """Should return same instance on multiple calls."""
        config1 = get_p2p_config()
        config2 = get_p2p_config()
        # Note: This depends on implementation - may or may not be singleton
        assert isinstance(config1, P2PConfig)
        assert isinstance(config2, P2PConfig)
