"""Tests for ProgressWatchdogDaemon.

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.progress_watchdog_daemon import (
    CANONICAL_CONFIGS,
    ConfigProgress,
    ProgressWatchdogConfig,
    ProgressWatchdogDaemon,
    get_progress_watchdog,
)
from app.coordination.protocols import CoordinatorStatus


class TestProgressWatchdogConfig:
    """Tests for ProgressWatchdogConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProgressWatchdogConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 3600  # 1 hour
        assert config.min_elo_velocity == 0.5
        assert config.stall_threshold_hours == 6.0
        assert config.recovery_action == "boost_selfplay"
        assert config.boost_multiplier == 2.0
        assert config.max_recovery_attempts == 4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProgressWatchdogConfig(
            check_interval_seconds=1800,
            min_elo_velocity=1.0,
            stall_threshold_hours=3.0,
        )
        assert config.check_interval_seconds == 1800
        assert config.min_elo_velocity == 1.0
        assert config.stall_threshold_hours == 3.0

    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_PROGRESS_ENABLED": "0",
            "RINGRIFT_PROGRESS_INTERVAL": "7200",
            "RINGRIFT_PROGRESS_MIN_VELOCITY": "0.25",
            "RINGRIFT_PROGRESS_STALL_HOURS": "12",
        }):
            config = ProgressWatchdogConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 7200
            assert config.min_elo_velocity == 0.25
            assert config.stall_threshold_hours == 12.0


class TestConfigProgress:
    """Tests for ConfigProgress dataclass."""

    def test_default_values(self):
        """Test default values."""
        progress = ConfigProgress(config_key="hex8_2p")
        assert progress.config_key == "hex8_2p"
        assert progress.last_elo == 1500.0
        assert progress.current_elo == 1500.0
        assert progress.velocity == 0.0
        assert progress.stall_start_time == 0.0
        assert progress.is_stalled is False

    def test_is_stalled(self):
        """Test stall detection."""
        progress = ConfigProgress(config_key="hex8_2p")
        assert progress.is_stalled is False

        progress.stall_start_time = time.time()
        assert progress.is_stalled is True

    def test_stall_duration(self):
        """Test stall duration calculation."""
        progress = ConfigProgress(config_key="hex8_2p")
        assert progress.stall_duration_hours == 0.0

        # Set stall start to 2 hours ago
        progress.stall_start_time = time.time() - (2 * 3600)
        duration = progress.stall_duration_hours
        assert 1.9 < duration < 2.1

    def test_reset_recovery_counter(self):
        """Test recovery counter reset after 24h."""
        progress = ConfigProgress(config_key="hex8_2p")
        progress.recovery_attempts = 3
        progress.last_recovery_time = time.time() - 86400 - 1  # 24h ago

        progress.reset_recovery_counter_if_needed()
        assert progress.recovery_attempts == 0


class TestProgressWatchdogDaemon:
    """Tests for ProgressWatchdogDaemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with default config."""
        ProgressWatchdogDaemon.reset_instance()
        config = ProgressWatchdogConfig()
        return ProgressWatchdogDaemon(config=config)

    def test_initialization(self, daemon):
        """Test daemon initialization."""
        assert daemon.config is not None
        assert len(daemon._progress) == len(CANONICAL_CONFIGS)
        assert daemon._total_stalls_detected == 0
        assert daemon._total_recoveries_triggered == 0

    def test_singleton(self):
        """Test singleton pattern."""
        ProgressWatchdogDaemon.reset_instance()
        d1 = get_progress_watchdog()
        d2 = get_progress_watchdog()
        assert d1 is d2
        ProgressWatchdogDaemon.reset_instance()

    def test_daemon_name(self, daemon):
        """Test daemon name."""
        assert daemon._get_daemon_name() == "ProgressWatchdog"

    def test_health_check_not_running(self, daemon):
        """Test health check when not running."""
        health = daemon.health_check()
        assert health.healthy is False
        assert health.status == CoordinatorStatus.STOPPED
        assert "not running" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self, daemon):
        """Test health check when running."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING

        health = daemon.health_check()
        assert health.healthy is True
        assert "making progress" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_with_stalled_configs(self, daemon):
        """Test health check with stalled configs."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING

        # Simulate a stalled config
        daemon._progress["hex8_2p"].stall_start_time = time.time() - (7 * 3600)

        health = daemon.health_check()
        assert health.healthy is True  # Daemon is healthy, config is stalled
        assert "stalled" in health.message.lower()
        assert "hex8_2p" in health.details["stalled_configs"]

    def test_get_stalled_configs(self, daemon):
        """Test getting list of stalled configs."""
        assert daemon.get_stalled_configs() == []

        # Simulate stall
        daemon._progress["square8_3p"].stall_start_time = time.time() - (7 * 3600)

        stalled = daemon.get_stalled_configs()
        assert "square8_3p" in stalled

    def test_get_progress_summary(self, daemon):
        """Test getting progress summary."""
        summary = daemon.get_progress_summary()
        assert len(summary) == len(CANONICAL_CONFIGS)
        for config_key in CANONICAL_CONFIGS:
            assert config_key in summary
            assert "velocity" in summary[config_key]
            assert "elo" in summary[config_key]
            assert "stalled" in summary[config_key]

    def test_get_status(self, daemon):
        """Test get_status includes progress details."""
        daemon._total_stalls_detected = 5
        daemon._total_recoveries_triggered = 3

        status = daemon.get_status()
        assert "progress" in status
        assert status["total_stalls_detected"] == 5
        assert status["total_recoveries_triggered"] == 3


class TestProgressWatchdogDaemonAsync:
    """Async tests for ProgressWatchdogDaemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for async tests."""
        ProgressWatchdogDaemon.reset_instance()
        config = ProgressWatchdogConfig()
        return ProgressWatchdogDaemon(config=config)

    @pytest.mark.asyncio
    async def test_get_elo_velocity_fallback(self, daemon):
        """Test Elo velocity calculation fallback."""
        velocity = await daemon._get_elo_velocity("hex8_2p")
        # Without database, should return 0.0
        assert velocity == 0.0

    @pytest.mark.asyncio
    async def test_check_config_progress_no_stall(self, daemon):
        """Test check_config_progress when making progress."""
        # Mock velocity to show progress
        with patch.object(daemon, "_get_elo_velocity", new_callable=AsyncMock) as mock:
            mock.return_value = 1.0  # Good velocity
            await daemon._check_config_progress("hex8_2p")

            progress = daemon._progress["hex8_2p"]
            assert progress.velocity == 1.0
            assert progress.is_stalled is False

    @pytest.mark.asyncio
    async def test_check_config_progress_starts_stall(self, daemon):
        """Test check_config_progress when stall starts."""
        with patch.object(daemon, "_get_elo_velocity", new_callable=AsyncMock) as mock:
            mock.return_value = 0.0  # No progress
            await daemon._check_config_progress("hex8_2p")

            progress = daemon._progress["hex8_2p"]
            assert progress.is_stalled is True

    @pytest.mark.asyncio
    async def test_trigger_recovery(self, daemon):
        """Test recovery triggering."""
        progress = daemon._progress["hex8_2p"]
        progress.stall_start_time = time.time() - (7 * 3600)  # 7h stall

        with patch("app.coordination.progress_watchdog_daemon.emit_event", new_callable=AsyncMock) as mock_emit:
            await daemon._trigger_recovery("hex8_2p", progress)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "PROGRESS_STALL_DETECTED"
            assert call_args[0][1]["config_key"] == "hex8_2p"

        assert progress.recovery_attempts == 1
        assert daemon._total_stalls_detected == 1
        assert daemon._total_recoveries_triggered == 1

    @pytest.mark.asyncio
    async def test_max_recovery_attempts(self, daemon):
        """Test max recovery attempts limit."""
        progress = daemon._progress["hex8_2p"]
        progress.stall_start_time = time.time() - (7 * 3600)
        progress.recovery_attempts = daemon.config.max_recovery_attempts

        with patch("app.coordination.progress_watchdog_daemon.emit_event", new_callable=AsyncMock) as mock_emit:
            await daemon._trigger_recovery("hex8_2p", progress)

            # Should not emit event due to max attempts
            mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle(self, daemon):
        """Test full run cycle."""
        with patch.object(daemon, "_check_config_progress", new_callable=AsyncMock) as mock:
            await daemon._run_cycle()

            # Should check all 12 configs
            assert mock.call_count == len(CANONICAL_CONFIGS)
