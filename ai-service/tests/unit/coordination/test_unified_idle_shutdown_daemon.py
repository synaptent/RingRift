"""Tests for UnifiedIdleShutdownDaemon.

Tests cover:
- IdleShutdownConfig dataclass (defaults, provider-specific, env overrides)
- NodeIdleStatus dataclass (idle duration, is_idle property)
- UnifiedIdleShutdownDaemon initialization
- Factory functions for backward compatibility
- Provider-specific behavior
"""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_idle_shutdown_daemon import (
    IdleShutdownConfig,
    NodeIdleStatus,
    PROVIDER_DEFAULTS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return IdleShutdownConfig(
        enabled=True,
        idle_threshold_seconds=600,
        min_nodes_to_retain=2,
        provider_name="test",
    )


@pytest.fixture
def node_status():
    """Create a test node status."""
    return NodeIdleStatus(
        instance_id="inst-123",
        instance_name="test-node",
        provider="vast",
        ip_address="10.0.0.1",
        gpu_name="RTX 4090",
        gpu_utilization=5.0,
        cost_per_hour=0.50,
    )


# =============================================================================
# PROVIDER_DEFAULTS Tests
# =============================================================================


class TestProviderDefaults:
    """Test provider-specific default configurations."""

    def test_lambda_defaults_exist(self):
        """Test Lambda provider defaults are defined."""
        assert "lambda" in PROVIDER_DEFAULTS
        defaults = PROVIDER_DEFAULTS["lambda"]
        assert defaults["idle_threshold_seconds"] == 1800  # 30 minutes
        assert defaults["env_prefix"] == "LAMBDA"

    def test_vast_defaults_exist(self):
        """Test Vast provider defaults are defined."""
        assert "vast" in PROVIDER_DEFAULTS
        defaults = PROVIDER_DEFAULTS["vast"]
        # Jan 2026: Updated from 900s to 1800s (Sprint 3.5)
        assert defaults["idle_threshold_seconds"] == 1800  # 30 minutes
        assert defaults["min_nodes_to_retain"] == 0

    def test_runpod_defaults_exist(self):
        """Test RunPod provider defaults are defined."""
        assert "runpod" in PROVIDER_DEFAULTS
        defaults = PROVIDER_DEFAULTS["runpod"]
        assert defaults["idle_threshold_seconds"] == 1200  # 20 minutes

    def test_vultr_defaults_exist(self):
        """Test Vultr provider defaults are defined."""
        assert "vultr" in PROVIDER_DEFAULTS


# =============================================================================
# IdleShutdownConfig Tests
# =============================================================================


class TestIdleShutdownConfig:
    """Test IdleShutdownConfig dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        config = IdleShutdownConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 60
        assert config.idle_threshold_seconds == 1800
        assert config.idle_utilization_threshold == 10.0
        assert config.min_nodes_to_retain == 1
        assert config.drain_period_seconds == 300
        assert config.dry_run is False

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = IdleShutdownConfig(
            enabled=False,
            idle_threshold_seconds=600,
            min_nodes_to_retain=5,
            dry_run=True,
            provider_name="custom",
        )
        assert config.enabled is False
        assert config.idle_threshold_seconds == 600
        assert config.min_nodes_to_retain == 5
        assert config.dry_run is True
        assert config.provider_name == "custom"

    def test_for_provider_lambda(self):
        """Test factory method for Lambda provider."""
        config = IdleShutdownConfig.for_provider("lambda")
        assert config.idle_threshold_seconds == 1800
        assert config.provider_name == "lambda"

    def test_for_provider_vast(self):
        """Test factory method for Vast provider."""
        config = IdleShutdownConfig.for_provider("vast")
        # Jan 2026: Updated from 900s to 1800s (Sprint 3.5)
        assert config.idle_threshold_seconds == 1800
        assert config.min_nodes_to_retain == 0
        assert config.provider_name == "vast"

    def test_for_provider_runpod(self):
        """Test factory method for RunPod provider."""
        config = IdleShutdownConfig.for_provider("runpod")
        assert config.idle_threshold_seconds == 1200
        assert config.provider_name == "runpod"

    def test_for_provider_case_insensitive(self):
        """Test provider name is case-insensitive."""
        config1 = IdleShutdownConfig.for_provider("VAST")
        config2 = IdleShutdownConfig.for_provider("vast")
        assert config1.idle_threshold_seconds == config2.idle_threshold_seconds

    def test_for_provider_unknown(self):
        """Test factory method for unknown provider uses base defaults."""
        config = IdleShutdownConfig.for_provider("unknown_provider")
        assert config.provider_name == "unknown_provider"
        # Uses base class defaults when provider not found
        assert config.enabled is True

    def test_env_override(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            "VAST_IDLE_THRESHOLD": "1200",
            "VAST_MIN_NODES": "3",
            "VAST_IDLE_DRY_RUN": "true",
        }):
            config = IdleShutdownConfig.for_provider("vast")
            assert config.idle_threshold_seconds == 1200
            assert config.min_nodes_to_retain == 3
            assert config.dry_run is True

    def test_env_enabled_override(self):
        """Test enabled flag can be overridden via env."""
        with patch.dict(os.environ, {"LAMBDA_IDLE_ENABLED": "false"}):
            config = IdleShutdownConfig.for_provider("lambda")
            assert config.enabled is False


# =============================================================================
# NodeIdleStatus Tests
# =============================================================================


class TestNodeIdleStatus:
    """Test NodeIdleStatus dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        status = NodeIdleStatus(
            instance_id="inst-1",
            instance_name="node-1",
            provider="vast",
            ip_address="10.0.0.1",
        )
        assert status.ssh_port == 22
        assert status.gpu_utilization == 0.0
        assert status.idle_since == 0.0
        assert status.status == "unknown"

    def test_custom_values(self, node_status):
        """Test initialization with custom values."""
        assert node_status.instance_id == "inst-123"
        assert node_status.gpu_name == "RTX 4090"
        assert node_status.cost_per_hour == 0.50

    def test_idle_duration_seconds_not_idle(self):
        """Test idle duration when not idle."""
        status = NodeIdleStatus(
            instance_id="inst-1",
            instance_name="node-1",
            provider="vast",
            ip_address="10.0.0.1",
            idle_since=0.0,
        )
        assert status.idle_duration_seconds == 0.0

    def test_idle_duration_seconds_when_idle(self):
        """Test idle duration when idle."""
        # Set idle_since to 60 seconds ago
        idle_start = time.time() - 60
        status = NodeIdleStatus(
            instance_id="inst-1",
            instance_name="node-1",
            provider="vast",
            ip_address="10.0.0.1",
            idle_since=idle_start,
        )
        # Should be approximately 60 seconds
        assert 59 <= status.idle_duration_seconds <= 62

    def test_is_idle_property_false(self):
        """Test is_idle property when not idle."""
        status = NodeIdleStatus(
            instance_id="inst-1",
            instance_name="node-1",
            provider="vast",
            ip_address="10.0.0.1",
            idle_since=0.0,
        )
        assert status.is_idle is False

    def test_is_idle_property_true(self):
        """Test is_idle property when idle."""
        status = NodeIdleStatus(
            instance_id="inst-1",
            instance_name="node-1",
            provider="vast",
            ip_address="10.0.0.1",
            idle_since=time.time() - 100,
        )
        assert status.is_idle is True


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for backward compatibility."""

    def test_create_lambda_idle_daemon_import(self):
        """Test lambda factory can be imported."""
        from app.coordination.unified_idle_shutdown_daemon import (
            create_lambda_idle_daemon,
        )
        assert callable(create_lambda_idle_daemon)

    def test_create_vast_idle_daemon_import(self):
        """Test vast factory can be imported."""
        from app.coordination.unified_idle_shutdown_daemon import (
            create_vast_idle_daemon,
        )
        assert callable(create_vast_idle_daemon)

    def test_create_runpod_idle_daemon_import(self):
        """Test runpod factory can be imported."""
        from app.coordination.unified_idle_shutdown_daemon import (
            create_runpod_idle_daemon,
        )
        assert callable(create_runpod_idle_daemon)


# =============================================================================
# UnifiedIdleShutdownDaemon Tests
# =============================================================================


class TestUnifiedIdleShutdownDaemonInit:
    """Test UnifiedIdleShutdownDaemon initialization."""

    def test_init_with_mock_provider(self):
        """Test initialization with mock provider."""
        from app.coordination.unified_idle_shutdown_daemon import (
            UnifiedIdleShutdownDaemon,
        )

        mock_provider = MagicMock()
        mock_provider.provider_type = "test"

        daemon = UnifiedIdleShutdownDaemon(
            provider=mock_provider,
            config=IdleShutdownConfig(provider_name="test"),
        )
        assert daemon is not None

    def test_init_with_default_config(self):
        """Test initialization uses default config if not provided."""
        from app.coordination.unified_idle_shutdown_daemon import (
            UnifiedIdleShutdownDaemon,
        )

        mock_provider = MagicMock()
        mock_provider.provider_type = "vast"

        daemon = UnifiedIdleShutdownDaemon(provider=mock_provider)
        assert daemon.config is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the daemon."""

    def test_config_isolation(self):
        """Test each config is independent."""
        config1 = IdleShutdownConfig(idle_threshold_seconds=100)
        config2 = IdleShutdownConfig(idle_threshold_seconds=200)
        assert config1.idle_threshold_seconds == 100
        assert config2.idle_threshold_seconds == 200

    def test_provider_defaults_are_readonly(self):
        """Test PROVIDER_DEFAULTS are not mutated."""
        original_vast = PROVIDER_DEFAULTS["vast"]["idle_threshold_seconds"]
        _ = IdleShutdownConfig.for_provider("vast")
        assert PROVIDER_DEFAULTS["vast"]["idle_threshold_seconds"] == original_vast
