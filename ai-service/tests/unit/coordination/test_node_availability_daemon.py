"""Unit tests for node_availability.daemon module.

Tests the NodeAvailabilityDaemon that syncs cloud provider state with config.
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.daemon import (
    DaemonStats,
    NodeAvailabilityConfig,
    NodeAvailabilityDaemon,
)


# =============================================================================
# NodeAvailabilityConfig Tests
# =============================================================================


class TestNodeAvailabilityConfig:
    """Tests for NodeAvailabilityConfig dataclass."""

    def test_default_check_interval(self):
        """Should have 5 minute default check interval."""
        config = NodeAvailabilityConfig()
        assert config.check_interval_seconds == 300.0

    def test_default_dry_run(self):
        """Should have dry_run enabled by default (safe mode)."""
        config = NodeAvailabilityConfig()
        assert config.dry_run is True

    def test_default_grace_period(self):
        """Should have 60 second grace period."""
        config = NodeAvailabilityConfig()
        assert config.grace_period_seconds == 60.0

    def test_default_provider_toggles(self):
        """All providers should be enabled by default."""
        config = NodeAvailabilityConfig()
        assert config.vast_enabled is True
        assert config.lambda_enabled is True
        assert config.runpod_enabled is True
        assert config.vultr_enabled is True
        assert config.hetzner_enabled is True

    def test_default_auto_update_voters(self):
        """Should not auto-update P2P voters by default."""
        config = NodeAvailabilityConfig()
        assert config.auto_update_voters is False

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = NodeAvailabilityConfig(
            check_interval_seconds=120.0,
            dry_run=False,
            grace_period_seconds=30.0,
            vast_enabled=False,
            auto_update_voters=True,
        )
        assert config.check_interval_seconds == 120.0
        assert config.dry_run is False
        assert config.grace_period_seconds == 30.0
        assert config.vast_enabled is False
        assert config.auto_update_voters is True

    def test_from_env_with_values(self):
        """Should read configuration from environment."""
        with patch.dict(os.environ, {
            "RINGRIFT_NODE_AVAILABILITY_ENABLED": "1",
            "RINGRIFT_NODE_AVAILABILITY_DRY_RUN": "false",
            "RINGRIFT_NODE_AVAILABILITY_INTERVAL": "120",
            "RINGRIFT_NODE_AVAILABILITY_GRACE_PERIOD": "45",
            "RINGRIFT_NODE_AVAILABILITY_VAST": "false",
            "RINGRIFT_NODE_AVAILABILITY_AUTO_VOTERS": "true",
        }):
            config = NodeAvailabilityConfig.from_env()
        assert config.enabled is True
        assert config.dry_run is False
        assert config.check_interval_seconds == 120.0
        assert config.grace_period_seconds == 45.0
        assert config.vast_enabled is False
        assert config.auto_update_voters is True


# =============================================================================
# DaemonStats Tests
# =============================================================================


class TestDaemonStats:
    """Tests for DaemonStats dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        stats = DaemonStats()
        assert stats.cycles_completed == 0
        assert stats.last_cycle_time is None
        assert stats.last_cycle_duration_seconds == 0.0
        assert stats.total_updates == 0
        assert stats.nodes_updated == 0
        assert stats.dry_run_updates == 0

    def test_default_provider_dicts(self):
        """Should have empty provider dictionaries."""
        stats = DaemonStats()
        assert stats.provider_checks == {}
        assert stats.provider_errors == {}

    def test_record_cycle(self):
        """Should record cycle completion."""
        stats = DaemonStats()
        stats.record_cycle(2.5)
        assert stats.cycles_completed == 1
        assert stats.last_cycle_time is not None
        assert stats.last_cycle_duration_seconds == 2.5

    def test_record_multiple_cycles(self):
        """Should track multiple cycles."""
        stats = DaemonStats()
        stats.record_cycle(1.0)
        stats.record_cycle(2.0)
        stats.record_cycle(1.5)
        assert stats.cycles_completed == 3
        assert stats.last_cycle_duration_seconds == 1.5

    def test_record_provider_check_success(self):
        """Should record successful provider checks."""
        stats = DaemonStats()
        stats.record_provider_check("vast", success=True)
        assert stats.provider_checks["vast"] == 1
        assert stats.provider_errors.get("vast", 0) == 0

    def test_record_provider_check_failure(self):
        """Should record failed provider checks."""
        stats = DaemonStats()
        stats.record_provider_check("lambda", success=False)
        assert stats.provider_checks["lambda"] == 1
        assert stats.provider_errors["lambda"] == 1

    def test_record_multiple_provider_checks(self):
        """Should accumulate provider check counts."""
        stats = DaemonStats()
        stats.record_provider_check("vast", success=True)
        stats.record_provider_check("vast", success=True)
        stats.record_provider_check("vast", success=False)
        assert stats.provider_checks["vast"] == 3
        assert stats.provider_errors["vast"] == 1

    def test_record_update(self):
        """Should record update results."""
        stats = DaemonStats()
        mock_result = MagicMock()
        mock_result.update_count = 5
        mock_result.dry_run = False
        stats.record_update(mock_result)
        assert stats.total_updates == 1
        assert stats.nodes_updated == 5
        assert stats.dry_run_updates == 0

    def test_record_dry_run_update(self):
        """Should track dry run updates."""
        stats = DaemonStats()
        mock_result = MagicMock()
        mock_result.update_count = 3
        mock_result.dry_run = True
        stats.record_update(mock_result)
        assert stats.total_updates == 1
        assert stats.nodes_updated == 3
        assert stats.dry_run_updates == 1


# =============================================================================
# NodeAvailabilityDaemon Tests
# =============================================================================


class TestNodeAvailabilityDaemonInit:
    """Tests for NodeAvailabilityDaemon initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        daemon = NodeAvailabilityDaemon()
        assert daemon.config is not None
        assert isinstance(daemon.config, NodeAvailabilityConfig)

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = NodeAvailabilityConfig(dry_run=False)
        daemon = NodeAvailabilityDaemon(config=config)
        assert daemon.config.dry_run is False

    def test_init_has_stats(self):
        """Should initialize with empty stats."""
        daemon = NodeAvailabilityDaemon()
        assert daemon._stats is not None
        assert daemon._stats.cycles_completed == 0

    def test_init_has_config_updater(self):
        """Should initialize config updater."""
        daemon = NodeAvailabilityDaemon()
        assert daemon._config_updater is not None


class TestNodeAvailabilityDaemonMethods:
    """Tests for NodeAvailabilityDaemon methods."""

    def test_get_daemon_name(self):
        """Should return correct daemon name."""
        daemon = NodeAvailabilityDaemon()
        assert daemon.name == "NodeAvailabilityDaemon"

    def test_get_default_config(self):
        """Default daemon config should be a NodeAvailabilityConfig."""
        daemon = NodeAvailabilityDaemon()
        config = daemon.config
        assert isinstance(config, NodeAvailabilityConfig)
