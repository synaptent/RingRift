"""Tests for sync_strategies module (December 2025).

This module tests the sync strategy definitions and configuration classes
used by AutoSyncDaemon.

Coverage:
- SyncStrategy constants
- AutoSyncConfig dataclass and from_config_file()
- SyncStats tracking and backward-compatible aliases
- MIN_MOVES_PER_GAME constants
"""

from __future__ import annotations

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSyncStrategy:
    """Tests for SyncStrategy enum-like class."""

    def test_strategy_values_exist(self):
        """All strategy values should be defined."""
        from app.coordination.sync_strategies import SyncStrategy

        assert SyncStrategy.HYBRID == "hybrid"
        assert SyncStrategy.EPHEMERAL == "ephemeral"
        assert SyncStrategy.BROADCAST == "broadcast"
        assert SyncStrategy.PULL == "pull"
        assert SyncStrategy.AUTO == "auto"

    def test_strategy_values_are_strings(self):
        """All strategy values should be strings."""
        from app.coordination.sync_strategies import SyncStrategy

        strategies = [
            SyncStrategy.HYBRID,
            SyncStrategy.EPHEMERAL,
            SyncStrategy.BROADCAST,
            SyncStrategy.PULL,
            SyncStrategy.AUTO,
        ]
        for strategy in strategies:
            assert isinstance(strategy, str)

    def test_strategy_values_are_unique(self):
        """All strategy values should be unique."""
        from app.coordination.sync_strategies import SyncStrategy

        strategies = [
            SyncStrategy.HYBRID,
            SyncStrategy.EPHEMERAL,
            SyncStrategy.BROADCAST,
            SyncStrategy.PULL,
            SyncStrategy.AUTO,
        ]
        assert len(strategies) == len(set(strategies))


class TestMinMoveConstants:
    """Tests for MIN_MOVES_PER_GAME constants."""

    def test_min_moves_dict_exists(self):
        """MIN_MOVES_PER_GAME dictionary should exist."""
        from app.coordination.sync_strategies import MIN_MOVES_PER_GAME

        assert isinstance(MIN_MOVES_PER_GAME, dict)
        assert len(MIN_MOVES_PER_GAME) > 0

    def test_default_min_moves_exists(self):
        """DEFAULT_MIN_MOVES should exist."""
        from app.coordination.sync_strategies import DEFAULT_MIN_MOVES

        assert isinstance(DEFAULT_MIN_MOVES, int)
        assert DEFAULT_MIN_MOVES > 0

    def test_min_moves_for_known_configs(self):
        """MIN_MOVES_PER_GAME should have entries for common configs."""
        from app.coordination.sync_strategies import MIN_MOVES_PER_GAME

        # Check hex8 configs
        assert ("hex8", 2) in MIN_MOVES_PER_GAME
        assert ("hex8", 3) in MIN_MOVES_PER_GAME
        assert ("hex8", 4) in MIN_MOVES_PER_GAME

        # Check square8 configs
        assert ("square8", 2) in MIN_MOVES_PER_GAME

        # Check square19 configs
        assert ("square19", 2) in MIN_MOVES_PER_GAME

    def test_min_moves_values_are_positive(self):
        """All MIN_MOVES values should be positive integers."""
        from app.coordination.sync_strategies import MIN_MOVES_PER_GAME

        for key, value in MIN_MOVES_PER_GAME.items():
            assert isinstance(value, int), f"Value for {key} should be int"
            assert value > 0, f"Value for {key} should be positive"

    def test_min_moves_increases_with_players(self):
        """MIN_MOVES should generally increase with player count."""
        from app.coordination.sync_strategies import MIN_MOVES_PER_GAME

        # For hex8
        assert MIN_MOVES_PER_GAME[("hex8", 2)] < MIN_MOVES_PER_GAME[("hex8", 3)]
        assert MIN_MOVES_PER_GAME[("hex8", 3)] < MIN_MOVES_PER_GAME[("hex8", 4)]


class TestAutoSyncConfig:
    """Tests for AutoSyncConfig dataclass."""

    def test_default_config_creation(self):
        """Should create config with default values."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()

        assert config.enabled is True
        assert config.interval_seconds == 60
        assert config.gossip_interval_seconds == 30
        assert config.max_concurrent_syncs == 4
        assert config.min_games_to_sync == 10

    def test_config_strategy_default(self):
        """Default strategy should be AUTO."""
        from app.coordination.sync_strategies import AutoSyncConfig, SyncStrategy

        config = AutoSyncConfig()
        assert config.strategy == SyncStrategy.AUTO

    def test_config_strategy_can_be_set(self):
        """Strategy can be set explicitly."""
        from app.coordination.sync_strategies import AutoSyncConfig, SyncStrategy

        config = AutoSyncConfig(strategy=SyncStrategy.EPHEMERAL)
        assert config.strategy == SyncStrategy.EPHEMERAL

    def test_config_exclude_hosts_default(self):
        """Exclude hosts should default to empty list."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()
        assert isinstance(config.exclude_hosts, list)
        assert len(config.exclude_hosts) == 0

    def test_config_disk_thresholds(self):
        """Disk usage thresholds should have defaults."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()
        assert config.max_disk_usage_percent == 70.0
        assert config.target_disk_usage_percent == 60.0

    def test_config_quality_filter_settings(self):
        """Quality filter settings should have defaults."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()
        assert config.quality_filter_enabled is True
        assert config.min_quality_for_sync >= 0.0
        assert config.min_quality_for_sync <= 1.0
        assert config.quality_sample_size > 0

    def test_config_ephemeral_settings(self):
        """Ephemeral-specific settings should have defaults."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()
        assert config.ephemeral_poll_seconds == 5
        assert config.ephemeral_write_through is True
        assert config.ephemeral_write_through_timeout > 0
        assert config.ephemeral_wal_enabled is True

    def test_config_broadcast_high_priority_configs(self):
        """Broadcast high priority configs should have defaults."""
        from app.coordination.sync_strategies import AutoSyncConfig

        config = AutoSyncConfig()
        assert isinstance(config.broadcast_high_priority_configs, list)
        assert len(config.broadcast_high_priority_configs) > 0
        assert "hex8_2p" in config.broadcast_high_priority_configs

    def test_from_config_file_without_file(self):
        """from_config_file should work with missing config."""
        from app.coordination.sync_strategies import AutoSyncConfig

        # Mock load_cluster_config to raise error - patch where it's imported
        with patch(
            "app.config.cluster_config.load_cluster_config",
            side_effect=OSError("File not found"),
        ):
            config = AutoSyncConfig.from_config_file()

        # Should return default config
        assert config.enabled is True
        assert config.interval_seconds == 60

    def test_from_config_file_with_mock_config(self):
        """from_config_file should load values from cluster config."""
        from app.coordination.sync_strategies import AutoSyncConfig

        # Create mock cluster config
        mock_cluster_cfg = MagicMock()
        mock_cluster_cfg.sync_routing.max_disk_usage_percent = 80.0
        mock_cluster_cfg.sync_routing.target_disk_usage_percent = 65.0
        mock_cluster_cfg.auto_sync.enabled = True
        mock_cluster_cfg.auto_sync.interval_seconds = 120
        mock_cluster_cfg.auto_sync.gossip_interval_seconds = 45
        mock_cluster_cfg.auto_sync.exclude_hosts = ["test-host"]
        mock_cluster_cfg.auto_sync.skip_nfs_sync = False
        mock_cluster_cfg.auto_sync.max_concurrent_syncs = 8
        mock_cluster_cfg.auto_sync.min_games_to_sync = 20
        mock_cluster_cfg.auto_sync.bandwidth_limit_mbps = 50
        mock_cluster_cfg.hosts_raw = {}

        with patch(
            "app.config.cluster_config.load_cluster_config",
            return_value=mock_cluster_cfg,
        ):
            config = AutoSyncConfig.from_config_file()

        assert config.max_disk_usage_percent == 80.0
        assert config.interval_seconds == 120
        assert config.max_concurrent_syncs == 8

    def test_from_config_file_excludes_coordinator_by_role(self):
        """from_config_file should exclude coordinator nodes by role."""
        from app.coordination.sync_strategies import AutoSyncConfig

        mock_cluster_cfg = MagicMock()
        mock_cluster_cfg.sync_routing.max_disk_usage_percent = 70.0
        mock_cluster_cfg.sync_routing.target_disk_usage_percent = 60.0
        mock_cluster_cfg.auto_sync.enabled = True
        mock_cluster_cfg.auto_sync.interval_seconds = 60
        mock_cluster_cfg.auto_sync.gossip_interval_seconds = 30
        mock_cluster_cfg.auto_sync.exclude_hosts = []
        mock_cluster_cfg.auto_sync.skip_nfs_sync = True
        mock_cluster_cfg.auto_sync.max_concurrent_syncs = 4
        mock_cluster_cfg.auto_sync.min_games_to_sync = 10
        mock_cluster_cfg.auto_sync.bandwidth_limit_mbps = 20
        mock_cluster_cfg.hosts_raw = {
            "coordinator-node": {"role": "coordinator"},
            "worker-node": {"role": "worker"},
        }

        with patch(
            "app.config.cluster_config.load_cluster_config",
            return_value=mock_cluster_cfg,
        ):
            config = AutoSyncConfig.from_config_file()

        assert "coordinator-node" in config.exclude_hosts
        assert "worker-node" not in config.exclude_hosts

    def test_from_config_file_excludes_by_hostname_pattern(self):
        """from_config_file should exclude coordinator nodes by hostname pattern."""
        from app.coordination.sync_strategies import AutoSyncConfig

        mock_cluster_cfg = MagicMock()
        mock_cluster_cfg.sync_routing.max_disk_usage_percent = 70.0
        mock_cluster_cfg.sync_routing.target_disk_usage_percent = 60.0
        mock_cluster_cfg.auto_sync.enabled = True
        mock_cluster_cfg.auto_sync.interval_seconds = 60
        mock_cluster_cfg.auto_sync.gossip_interval_seconds = 30
        mock_cluster_cfg.auto_sync.exclude_hosts = []
        mock_cluster_cfg.auto_sync.skip_nfs_sync = True
        mock_cluster_cfg.auto_sync.max_concurrent_syncs = 4
        mock_cluster_cfg.auto_sync.min_games_to_sync = 10
        mock_cluster_cfg.auto_sync.bandwidth_limit_mbps = 20
        mock_cluster_cfg.hosts_raw = {
            "mac-studio": {},
            "local-mac": {},
            "vast-worker": {},
        }

        with patch(
            "app.config.cluster_config.load_cluster_config",
            return_value=mock_cluster_cfg,
        ):
            config = AutoSyncConfig.from_config_file()

        assert "mac-studio" in config.exclude_hosts
        assert "local-mac" in config.exclude_hosts
        assert "vast-worker" not in config.exclude_hosts

    def test_from_config_file_allows_coordinator_with_external_storage(self):
        """Coordinators with external storage should NOT be excluded."""
        from app.coordination.sync_strategies import AutoSyncConfig

        mock_cluster_cfg = MagicMock()
        mock_cluster_cfg.sync_routing.max_disk_usage_percent = 70.0
        mock_cluster_cfg.sync_routing.target_disk_usage_percent = 60.0
        mock_cluster_cfg.auto_sync.enabled = True
        mock_cluster_cfg.auto_sync.interval_seconds = 60
        mock_cluster_cfg.auto_sync.gossip_interval_seconds = 30
        mock_cluster_cfg.auto_sync.exclude_hosts = []
        mock_cluster_cfg.auto_sync.skip_nfs_sync = True
        mock_cluster_cfg.auto_sync.max_concurrent_syncs = 4
        mock_cluster_cfg.auto_sync.min_games_to_sync = 10
        mock_cluster_cfg.auto_sync.bandwidth_limit_mbps = 20
        mock_cluster_cfg.hosts_raw = {
            "mac-studio": {"use_external_storage": True},
        }

        with patch(
            "app.config.cluster_config.load_cluster_config",
            return_value=mock_cluster_cfg,
        ):
            config = AutoSyncConfig.from_config_file()

        # Should NOT be excluded because it has external storage
        assert "mac-studio" not in config.exclude_hosts


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_stats_creation(self):
        """Should create stats with default values."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()

        assert stats.games_synced == 0
        assert stats.databases_skipped_quality == 0
        assert stats.databases_quality_checked == 0

    def test_stats_inherits_from_base(self):
        """SyncStats should inherit from SyncDaemonStats."""
        from app.coordination.sync_strategies import SyncStats
        from app.coordination.daemon_stats import SyncDaemonStats

        stats = SyncStats()
        assert isinstance(stats, SyncDaemonStats)

    def test_backward_compat_total_syncs(self):
        """total_syncs alias should work."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.operations_attempted = 42

        assert stats.total_syncs == 42

    def test_backward_compat_successful_syncs(self):
        """successful_syncs alias should work."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.syncs_completed = 30

        assert stats.successful_syncs == 30

    def test_backward_compat_failed_syncs(self):
        """failed_syncs alias should work."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.syncs_failed = 5

        assert stats.failed_syncs == 5

    def test_backward_compat_bytes_transferred(self):
        """bytes_transferred alias should work."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.bytes_synced = 1024 * 1024

        assert stats.bytes_transferred == 1024 * 1024

    def test_backward_compat_last_sync_time(self):
        """last_sync_time alias should work."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.last_check_time = 1234567890.0

        assert stats.last_sync_time == 1234567890.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all SyncStats fields."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.games_synced = 100
        stats.databases_skipped_quality = 5
        stats.databases_quality_checked = 50
        stats.syncs_completed = 10
        stats.syncs_failed = 2
        stats.bytes_synced = 2048

        result = stats.to_dict()

        assert result["games_synced"] == 100
        assert result["databases_skipped_quality"] == 5
        assert result["databases_quality_checked"] == 50
        assert result["successful_syncs"] == 10
        assert result["failed_syncs"] == 2
        assert result["bytes_transferred"] == 2048

    def test_to_dict_includes_verification_stats(self):
        """to_dict should include verification stats."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        stats.databases_verified = 25
        stats.databases_verification_failed = 3
        stats.last_verification_time = 1234567890.0

        result = stats.to_dict()

        assert result["databases_verified"] == 25
        assert result["databases_verification_failed"] == 3
        assert result["last_verification_time"] == 1234567890.0

    def test_is_healthy_inherited(self):
        """is_healthy should be inherited from base class."""
        from app.coordination.sync_strategies import SyncStats

        stats = SyncStats()
        # Base class default should be healthy
        assert stats.is_healthy() is True

        # Simulate failures - must also set operations_attempted > 0
        # because is_healthy() returns True early if operations_attempted == 0
        stats.operations_attempted = 10
        stats.consecutive_failures = 10
        # is_healthy checks consecutive_failures >= max_failures (default 5)
        assert stats.is_healthy() is False


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_exist(self):
        """All items in __all__ should be importable."""
        from app.coordination.sync_strategies import __all__

        import app.coordination.sync_strategies as module

        for name in __all__:
            assert hasattr(module, name), f"Missing export: {name}"

    def test_expected_exports(self):
        """Expected exports should be available."""
        from app.coordination.sync_strategies import (
            SyncStrategy,
            AutoSyncConfig,
            SyncStats,
            MIN_MOVES_PER_GAME,
            DEFAULT_MIN_MOVES,
        )

        # Just verify they can be imported
        assert SyncStrategy is not None
        assert AutoSyncConfig is not None
        assert SyncStats is not None
        assert MIN_MOVES_PER_GAME is not None
        assert DEFAULT_MIN_MOVES is not None
