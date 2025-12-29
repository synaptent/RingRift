"""Unit tests for memory_monitor_daemon module.

Tests the GPU VRAM and process memory monitoring daemon.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.memory_monitor_daemon import (
    MemoryMonitorConfig,
    MemoryMonitorDaemon,
    MemoryThresholds,
    _env_bool,
    _env_float,
    _env_int,
)


# =============================================================================
# Environment Helper Tests
# =============================================================================


class TestEnvFloat:
    """Tests for _env_float helper."""

    def test_returns_default_when_not_set(self):
        """Should return default when env var not set."""
        result = _env_float("NONEXISTENT_VAR", 0.75)
        assert result == 0.75

    def test_returns_env_value_when_set(self):
        """Should return env value when set."""
        with patch.dict(os.environ, {"RINGRIFT_MEMORY_GPU_WARNING": "0.80"}):
            result = _env_float("GPU_WARNING", 0.75)
        assert result == 0.80


class TestEnvInt:
    """Tests for _env_int helper."""

    def test_returns_default_when_not_set(self):
        """Should return default when env var not set."""
        result = _env_int("NONEXISTENT_VAR", 100)
        assert result == 100

    def test_returns_env_value_when_set(self):
        """Should return env value when set."""
        with patch.dict(os.environ, {"RINGRIFT_MEMORY_CHECK_INTERVAL": "60"}):
            result = _env_int("CHECK_INTERVAL", 30)
        assert result == 60


class TestEnvBool:
    """Tests for _env_bool helper."""

    def test_returns_default_when_not_set(self):
        """Should return default when env var not set."""
        result = _env_bool("NONEXISTENT_VAR", True)
        assert result is True

    def test_true_values(self):
        """Should recognize true values."""
        for val in ["true", "1", "yes"]:
            with patch.dict(os.environ, {"RINGRIFT_MEMORY_TEST": val}):
                result = _env_bool("TEST", False)
            assert result is True, f"Failed for value: {val}"


# =============================================================================
# MemoryThresholds Tests
# =============================================================================


class TestMemoryThresholds:
    """Tests for MemoryThresholds dataclass."""

    def test_default_gpu_thresholds(self):
        """Should have sensible GPU defaults."""
        thresholds = MemoryThresholds()
        assert thresholds.gpu_warning == 0.75  # 75%
        assert thresholds.gpu_critical == 0.85  # 85%

    def test_default_ram_thresholds(self):
        """Should have sensible RAM defaults."""
        thresholds = MemoryThresholds()
        assert thresholds.ram_warning == 0.80  # 80%
        assert thresholds.ram_critical == 0.90  # 90%

    def test_default_process_rss_threshold(self):
        """Should have 32GB RSS threshold."""
        thresholds = MemoryThresholds()
        expected = 32 * 1024 * 1024 * 1024  # 32GB
        assert thresholds.process_rss_critical_bytes == expected

    def test_default_sigkill_grace_period(self):
        """Should have 60s grace period."""
        thresholds = MemoryThresholds()
        assert thresholds.sigkill_grace_period == 60.0

    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        thresholds = MemoryThresholds(
            gpu_warning=0.70,
            gpu_critical=0.80,
        )
        assert thresholds.gpu_warning == 0.70
        assert thresholds.gpu_critical == 0.80


# =============================================================================
# MemoryMonitorConfig Tests
# =============================================================================


class TestMemoryMonitorConfig:
    """Tests for MemoryMonitorConfig dataclass."""

    def test_default_enabled(self):
        """Should be enabled by default."""
        config = MemoryMonitorConfig()
        assert config.enabled is True

    def test_default_check_interval(self):
        """Should have 30s check interval by default."""
        config = MemoryMonitorConfig()
        assert config.check_interval_seconds == 30.0

    def test_default_kill_enabled(self):
        """Should have kill enabled by default."""
        config = MemoryMonitorConfig()
        assert config.kill_enabled is True

    def test_default_event_cooldown(self):
        """Should have 60s event cooldown by default."""
        config = MemoryMonitorConfig()
        assert config.event_cooldown_seconds == 60.0

    def test_default_monitors_enabled(self):
        """All monitors should be enabled by default."""
        config = MemoryMonitorConfig()
        assert config.monitor_gpu is True
        assert config.monitor_ram is True
        assert config.monitor_processes is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = MemoryMonitorConfig(
            enabled=False,
            check_interval_seconds=60.0,
            kill_enabled=False,
            monitor_gpu=False,
        )
        assert config.enabled is False
        assert config.check_interval_seconds == 60.0
        assert config.kill_enabled is False
        assert config.monitor_gpu is False

    def test_from_env_method(self):
        """Should have from_env class method."""
        assert hasattr(MemoryMonitorConfig, "from_env")
        config = MemoryMonitorConfig.from_env()
        assert isinstance(config, MemoryMonitorConfig)


# =============================================================================
# MemoryMonitorDaemon Tests
# =============================================================================


class TestMemoryMonitorDaemonInit:
    """Tests for MemoryMonitorDaemon initialization."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        MemoryMonitorDaemon.reset_instance()
        daemon = MemoryMonitorDaemon()
        assert daemon._config is not None
        assert isinstance(daemon._config, MemoryMonitorConfig)
        MemoryMonitorDaemon.reset_instance()

    def test_init_custom_config(self):
        """Should accept custom config."""
        MemoryMonitorDaemon.reset_instance()
        config = MemoryMonitorConfig(check_interval_seconds=60.0)
        daemon = MemoryMonitorDaemon(config=config)
        assert daemon._config.check_interval_seconds == 60.0
        MemoryMonitorDaemon.reset_instance()

    def test_is_singleton(self):
        """Should be a singleton."""
        MemoryMonitorDaemon.reset_instance()

        daemon1 = MemoryMonitorDaemon.get_instance()
        daemon2 = MemoryMonitorDaemon.get_instance()
        assert daemon1 is daemon2

        MemoryMonitorDaemon.reset_instance()

    def test_get_instance_class_method(self):
        """Should have get_instance class method."""
        MemoryMonitorDaemon.reset_instance()

        daemon = MemoryMonitorDaemon.get_instance()
        assert isinstance(daemon, MemoryMonitorDaemon)

        MemoryMonitorDaemon.reset_instance()


class TestMemoryMonitorDaemonState:
    """Tests for MemoryMonitorDaemon state tracking."""

    def test_initial_last_event_time_zero(self):
        """Last event time should start at 0."""
        MemoryMonitorDaemon.reset_instance()
        daemon = MemoryMonitorDaemon()
        assert daemon._last_memory_pressure_event == 0.0
        MemoryMonitorDaemon.reset_instance()


class TestMemoryMonitorDaemonThresholds:
    """Tests for threshold validation."""

    def test_gpu_warning_less_than_critical(self):
        """GPU warning should be less than critical."""
        thresholds = MemoryThresholds()
        assert thresholds.gpu_warning < thresholds.gpu_critical

    def test_ram_warning_less_than_critical(self):
        """RAM warning should be less than critical."""
        thresholds = MemoryThresholds()
        assert thresholds.ram_warning < thresholds.ram_critical


class TestMemoryMonitorDaemonGracePeriod:
    """Tests for grace period configuration."""

    def test_positive_grace_period(self):
        """Grace period should be positive."""
        thresholds = MemoryThresholds()
        assert thresholds.sigkill_grace_period > 0


class TestMemoryMonitorDaemonProcessRss:
    """Tests for process RSS configuration."""

    def test_rss_threshold_is_bytes(self):
        """RSS threshold should be in bytes."""
        thresholds = MemoryThresholds()
        # 32GB in bytes = 32 * 1024^3
        assert thresholds.process_rss_critical_bytes == 32 * 1024 * 1024 * 1024

    def test_rss_threshold_is_positive(self):
        """RSS threshold should be positive."""
        thresholds = MemoryThresholds()
        assert thresholds.process_rss_critical_bytes > 0


class TestMemoryMonitorDaemonEventCooldown:
    """Tests for event cooldown."""

    def test_event_cooldown_positive(self):
        """Event cooldown should be positive."""
        config = MemoryMonitorConfig()
        assert config.event_cooldown_seconds > 0
