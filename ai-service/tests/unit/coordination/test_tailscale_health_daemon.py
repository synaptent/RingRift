"""Unit tests for TailscaleHealthDaemon.

December 2025: Created for comprehensive test coverage.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.tailscale_health_daemon import (
    TailscaleHealthConfig,
    TailscaleHealthDaemon,
    TailscaleState,
    TailscaleStatus,
    create_tailscale_health_daemon,
    get_tailscale_health_daemon,
)


# =============================================================================
# TailscaleStatus Tests
# =============================================================================


class TestTailscaleStatus:
    """Tests for TailscaleStatus enum."""

    def test_has_expected_statuses(self):
        """Should have expected status values."""
        assert TailscaleStatus.CONNECTED.value == "connected"
        assert TailscaleStatus.DISCONNECTED.value == "disconnected"
        assert TailscaleStatus.STARTING.value == "starting"
        assert TailscaleStatus.NEEDS_AUTH.value == "needs_auth"
        assert TailscaleStatus.UNKNOWN.value == "unknown"
        assert TailscaleStatus.NOT_INSTALLED.value == "not_installed"


# =============================================================================
# TailscaleState Tests
# =============================================================================


class TestTailscaleState:
    """Tests for TailscaleState dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        state = TailscaleState()
        assert state.status == TailscaleStatus.UNKNOWN
        assert state.tailscale_ip is None
        assert state.hostname is None
        assert state.backend_state is None
        assert state.is_online is False
        assert state.is_running is False
        assert state.last_check_time == 0.0
        assert state.last_connected_time == 0.0
        assert state.recovery_attempts == 0
        assert state.last_recovery_time == 0.0
        assert state.error_message is None

    def test_custom_values(self):
        """Should accept custom values."""
        state = TailscaleState(
            status=TailscaleStatus.CONNECTED,
            tailscale_ip="100.64.0.1",
            hostname="my-node",
            is_online=True,
            is_running=True,
        )
        assert state.status == TailscaleStatus.CONNECTED
        assert state.tailscale_ip == "100.64.0.1"
        assert state.hostname == "my-node"
        assert state.is_online is True
        assert state.is_running is True


# =============================================================================
# TailscaleHealthConfig Tests
# =============================================================================


class TestTailscaleHealthConfig:
    """Tests for TailscaleHealthConfig dataclass."""

    def test_default_values(self):
        """Should have expected defaults."""
        config = TailscaleHealthConfig()
        assert config.check_interval_seconds == 30.0
        assert config.health_check_timeout_seconds == 10.0
        assert config.max_recovery_attempts == 3
        assert config.recovery_cooldown_seconds == 300.0
        assert config.use_userspace_networking is True
        assert config.tailscale_state_dir == "/var/lib/tailscale"
        assert config.tailscale_socket_dir == "/var/run/tailscale"
        assert config.report_to_p2p is True
        assert config.p2p_status_endpoint == "http://localhost:8770/tailscale_health"

    def test_custom_values(self):
        """Should accept custom values."""
        config = TailscaleHealthConfig(
            check_interval_seconds=60.0,
            max_recovery_attempts=5,
            use_userspace_networking=False,
        )
        assert config.check_interval_seconds == 60.0
        assert config.max_recovery_attempts == 5
        assert config.use_userspace_networking is False

    def test_from_env(self):
        """Should load config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_TAILSCALE_CHECK_INTERVAL": "45.0",
            "RINGRIFT_TAILSCALE_MAX_RECOVERY": "5",
            "RINGRIFT_TAILSCALE_USERSPACE": "false",
            "RINGRIFT_TAILSCALE_REPORT_P2P": "false",
        }):
            config = TailscaleHealthConfig.from_env()
            assert config.check_interval_seconds == 45.0
            assert config.max_recovery_attempts == 5
            assert config.use_userspace_networking is False
            assert config.report_to_p2p is False


# =============================================================================
# TailscaleHealthDaemon Tests
# =============================================================================


class TestTailscaleHealthDaemon:
    """Tests for TailscaleHealthDaemon class."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with test config."""
        TailscaleHealthDaemon.reset_instance()
        config = TailscaleHealthConfig(
            check_interval_seconds=1.0,
            recovery_cooldown_seconds=0.0,
            report_to_p2p=False,
        )
        return TailscaleHealthDaemon(config=config)

    def test_initialization(self, daemon):
        """Should initialize with config."""
        assert daemon._ts_config is not None
        assert daemon._state is not None
        assert daemon._recovery_in_progress is False

    def test_get_state(self, daemon):
        """Should return current state."""
        state = daemon.get_state()
        assert isinstance(state, TailscaleState)
        assert state.status == TailscaleStatus.UNKNOWN

    def test_get_tailscale_ip(self, daemon):
        """Should return tailscale IP."""
        assert daemon.get_tailscale_ip() is None
        daemon._state.tailscale_ip = "100.64.0.1"
        assert daemon.get_tailscale_ip() == "100.64.0.1"

    def test_health_check_disconnected(self, daemon):
        """Should report unhealthy when disconnected."""
        daemon._state.status = TailscaleStatus.DISCONNECTED
        health = daemon.health_check()
        assert health.healthy is False
        assert "disconnected" in health.message.lower()

    def test_health_check_connected(self, daemon):
        """Should report healthy when connected."""
        daemon._state.status = TailscaleStatus.CONNECTED
        daemon._state.tailscale_ip = "100.64.0.1"
        health = daemon.health_check()
        assert health.healthy is True
        assert "connected" in health.message.lower()
        assert health.details["tailscale_ip"] == "100.64.0.1"

    def test_get_event_subscriptions(self, daemon):
        """Should have no event subscriptions (polling daemon)."""
        subs = daemon._get_event_subscriptions()
        assert subs == {}


class TestTailscaleHealthDaemonAsync:
    """Async tests for TailscaleHealthDaemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for async tests."""
        TailscaleHealthDaemon.reset_instance()
        config = TailscaleHealthConfig(
            check_interval_seconds=1.0,
            recovery_cooldown_seconds=0.0,
            report_to_p2p=False,
        )
        return TailscaleHealthDaemon(config=config)

    @pytest.mark.asyncio
    async def test_check_tailscale_not_installed(self, daemon):
        """Should detect Tailscale not installed."""
        with patch("shutil.which", return_value=None):
            await daemon._check_tailscale_status()
            assert daemon._state.status == TailscaleStatus.NOT_INSTALLED
            assert daemon._state.is_online is False

    @pytest.mark.asyncio
    async def test_check_tailscaled_not_running(self, daemon):
        """Should detect tailscaled not running."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            with patch.object(daemon, "_is_tailscaled_running", new_callable=AsyncMock) as mock:
                mock.return_value = False
                await daemon._check_tailscale_status()
                assert daemon._state.status == TailscaleStatus.DISCONNECTED
                assert daemon._state.is_running is False

    @pytest.mark.asyncio
    async def test_check_tailscale_connected(self, daemon):
        """Should parse connected status correctly."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            with patch.object(daemon, "_is_tailscaled_running", new_callable=AsyncMock) as mock_running:
                mock_running.return_value = True
                with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.return_value = {
                        "returncode": 0,
                        "stdout": '{"BackendState": "Running", "Self": {"TailscaleIPs": ["100.64.0.1"], "HostName": "test-node"}}',
                        "stderr": "",
                    }
                    await daemon._check_tailscale_status()
                    assert daemon._state.status == TailscaleStatus.CONNECTED
                    assert daemon._state.is_online is True
                    assert daemon._state.tailscale_ip == "100.64.0.1"
                    assert daemon._state.hostname == "test-node"

    @pytest.mark.asyncio
    async def test_check_tailscale_needs_login(self, daemon):
        """Should detect needs login state."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            with patch.object(daemon, "_is_tailscaled_running", new_callable=AsyncMock) as mock_running:
                mock_running.return_value = True
                with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.return_value = {
                        "returncode": 0,
                        "stdout": '{"BackendState": "NeedsLogin", "Self": {}}',
                        "stderr": "",
                    }
                    await daemon._check_tailscale_status()
                    assert daemon._state.status == TailscaleStatus.NEEDS_AUTH

    @pytest.mark.asyncio
    async def test_check_tailscale_starting(self, daemon):
        """Should detect starting state."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            with patch.object(daemon, "_is_tailscaled_running", new_callable=AsyncMock) as mock_running:
                mock_running.return_value = True
                with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.return_value = {
                        "returncode": 0,
                        "stdout": '{"BackendState": "Starting"}',
                        "stderr": "",
                    }
                    await daemon._check_tailscale_status()
                    assert daemon._state.status == TailscaleStatus.STARTING

    @pytest.mark.asyncio
    async def test_check_tailscale_timeout(self, daemon):
        """Should handle status timeout."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            with patch.object(daemon, "_is_tailscaled_running", new_callable=AsyncMock) as mock_running:
                mock_running.return_value = True
                with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.side_effect = asyncio.TimeoutError()
                    await daemon._check_tailscale_status()
                    assert daemon._state.status == TailscaleStatus.UNKNOWN
                    assert "timed out" in daemon._state.error_message.lower()

    @pytest.mark.asyncio
    async def test_recovery_already_in_progress(self, daemon):
        """Should skip recovery if already in progress."""
        daemon._recovery_in_progress = True
        result = await daemon._attempt_recovery()
        assert result is False

    @pytest.mark.asyncio
    async def test_recovery_cooldown_active(self, daemon):
        """Should respect recovery cooldown."""
        daemon._ts_config.recovery_cooldown_seconds = 300.0
        daemon._state.last_recovery_time = 1.0  # Recent attempt
        import time
        daemon._state.last_recovery_time = time.time()

        result = await daemon._attempt_recovery()
        assert result is False

    @pytest.mark.asyncio
    async def test_recovery_max_attempts_reached(self, daemon):
        """Should stop after max recovery attempts."""
        daemon._state.recovery_attempts = daemon._ts_config.max_recovery_attempts

        result = await daemon._attempt_recovery()
        assert result is False

    @pytest.mark.asyncio
    async def test_recovery_success_tailscale_up(self, daemon):
        """Should recover via tailscale up."""
        with patch.object(daemon, "_try_tailscale_up", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await daemon._attempt_recovery()
            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_fallback_to_restart(self, daemon):
        """Should fall back to restart if up fails."""
        with patch.object(daemon, "_try_tailscale_up", new_callable=AsyncMock) as mock_up:
            mock_up.side_effect = [False, True]  # Fail first, succeed after restart
            with patch.object(daemon, "_restart_tailscaled", new_callable=AsyncMock) as mock_restart:
                mock_restart.return_value = True
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await daemon._attempt_recovery()
                    assert result is True
                    assert mock_restart.called

    @pytest.mark.asyncio
    async def test_is_tailscaled_running_success(self, daemon):
        """Should detect running tailscaled."""
        with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock:
            mock.return_value = {"returncode": 0, "stdout": "1234", "stderr": ""}
            result = await daemon._is_tailscaled_running()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_tailscaled_running_not_found(self, daemon):
        """Should detect not running tailscaled."""
        with patch.object(daemon, "_run_command", new_callable=AsyncMock) as mock:
            mock.return_value = {"returncode": 1, "stdout": "", "stderr": ""}
            result = await daemon._is_tailscaled_running()
            assert result is False

    @pytest.mark.asyncio
    async def test_run_command_success(self, daemon):
        """Should run command successfully."""
        with patch("asyncio.create_subprocess_exec") as mock_create:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
            mock_proc.returncode = 0
            mock_create.return_value = mock_proc

            result = await daemon._run_command(["echo", "test"])
            assert result["returncode"] == 0
            assert result["stdout"] == "output"

    @pytest.mark.asyncio
    async def test_run_command_timeout(self, daemon):
        """Should handle command timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_create:
            mock_proc = MagicMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_create.return_value = mock_proc

            result = await daemon._run_command(["sleep", "100"], timeout=0.1)
            assert result["returncode"] == -1
            assert "timed out" in result["stderr"].lower()

    @pytest.mark.asyncio
    async def test_run_command_background(self, daemon):
        """Should run command in background."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            result = await daemon._run_command(["sleep", "100"], background=True)
            assert result["returncode"] == 0
            mock_popen.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_healthy(self, daemon):
        """Should complete run cycle when healthy."""
        with patch.object(daemon, "_check_tailscale_status", new_callable=AsyncMock) as mock_check:
            daemon._state.status = TailscaleStatus.CONNECTED
            await daemon._run_cycle()
            mock_check.assert_called_once()
            assert daemon._stats.cycles_completed == 1

    @pytest.mark.asyncio
    async def test_run_cycle_unhealthy_triggers_recovery(self, daemon):
        """Should trigger recovery when unhealthy."""
        with patch.object(daemon, "_check_tailscale_status", new_callable=AsyncMock):
            daemon._state.status = TailscaleStatus.DISCONNECTED
            with patch.object(daemon, "_attempt_recovery", new_callable=AsyncMock) as mock_recover:
                mock_recover.return_value = False
                await daemon._run_cycle()
                mock_recover.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_error_handling(self, daemon):
        """Should handle errors in run cycle."""
        with patch.object(daemon, "_check_tailscale_status", new_callable=AsyncMock) as mock_check:
            mock_check.side_effect = Exception("Test error")
            await daemon._run_cycle()
            assert daemon._stats.errors_count == 1
            assert "Test error" in daemon._stats.last_error


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tailscale_health_daemon(self):
        """Should create new daemon instance."""
        TailscaleHealthDaemon.reset_instance()
        daemon = create_tailscale_health_daemon()
        assert isinstance(daemon, TailscaleHealthDaemon)

    def test_get_tailscale_health_daemon_singleton(self):
        """Should return singleton instance."""
        TailscaleHealthDaemon.reset_instance()
        d1 = get_tailscale_health_daemon()
        d2 = get_tailscale_health_daemon()
        assert d1 is d2
        TailscaleHealthDaemon.reset_instance()


# =============================================================================
# Parse Status JSON Tests
# =============================================================================


class TestParseStatusJson:
    """Tests for _parse_status_json method."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for tests."""
        TailscaleHealthDaemon.reset_instance()
        return TailscaleHealthDaemon()

    def test_parse_running_status(self, daemon):
        """Should parse Running backend state."""
        data = {
            "BackendState": "Running",
            "Self": {
                "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"],
                "HostName": "my-node",
            },
        }
        daemon._parse_status_json(data)
        assert daemon._state.status == TailscaleStatus.CONNECTED
        assert daemon._state.is_online is True
        assert daemon._state.tailscale_ip == "100.64.0.1"
        assert daemon._state.hostname == "my-node"

    def test_parse_needs_login(self, daemon):
        """Should parse NeedsLogin backend state."""
        data = {"BackendState": "NeedsLogin"}
        daemon._parse_status_json(data)
        assert daemon._state.status == TailscaleStatus.NEEDS_AUTH
        assert daemon._state.is_online is False

    def test_parse_starting(self, daemon):
        """Should parse Starting backend state."""
        data = {"BackendState": "Starting"}
        daemon._parse_status_json(data)
        assert daemon._state.status == TailscaleStatus.STARTING
        assert daemon._state.is_online is False

    def test_parse_unknown_state(self, daemon):
        """Should treat unknown states as disconnected."""
        data = {"BackendState": "Stopped"}
        daemon._parse_status_json(data)
        assert daemon._state.status == TailscaleStatus.DISCONNECTED
        assert daemon._state.is_online is False

    def test_parse_no_self_info(self, daemon):
        """Should handle missing Self info."""
        data = {"BackendState": "Running"}
        daemon._parse_status_json(data)
        assert daemon._state.status == TailscaleStatus.CONNECTED
        assert daemon._state.tailscale_ip is None
        assert daemon._state.hostname is None

    def test_parse_empty_tailscale_ips(self, daemon):
        """Should handle empty TailscaleIPs list."""
        data = {
            "BackendState": "Running",
            "Self": {"TailscaleIPs": [], "HostName": "test"},
        }
        daemon._parse_status_json(data)
        assert daemon._state.tailscale_ip is None
        assert daemon._state.hostname == "test"
