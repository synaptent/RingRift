"""Tests for P2PRecoveryDaemon.

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.p2p_recovery_daemon import (
    P2PRecoveryConfig,
    P2PRecoveryDaemon,
    get_p2p_recovery_daemon,
)
from app.coordination.protocols import CoordinatorStatus


class TestP2PRecoveryConfig:
    """Tests for P2PRecoveryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = P2PRecoveryConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 60
        assert config.health_endpoint == "http://localhost:8770/status"
        assert config.max_consecutive_failures == 3
        assert config.restart_cooldown_seconds == 300  # 5 minutes
        assert config.health_timeout_seconds == 10.0
        assert config.min_alive_peers == 3
        assert config.startup_grace_seconds == 30

    def test_custom_values(self):
        """Test custom configuration values."""
        config = P2PRecoveryConfig(
            check_interval_seconds=120,
            max_consecutive_failures=5,
            restart_cooldown_seconds=600,
        )
        assert config.check_interval_seconds == 120
        assert config.max_consecutive_failures == 5
        assert config.restart_cooldown_seconds == 600

    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_P2P_RECOVERY_ENABLED": "0",
            "RINGRIFT_P2P_RECOVERY_INTERVAL": "180",
            "RINGRIFT_P2P_HEALTH_ENDPOINT": "http://localhost:9999/health",
            "RINGRIFT_P2P_MAX_FAILURES": "5",
            "RINGRIFT_P2P_RESTART_COOLDOWN": "600",
            "RINGRIFT_P2P_MIN_PEERS": "5",
        }):
            config = P2PRecoveryConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 180
            assert config.health_endpoint == "http://localhost:9999/health"
            assert config.max_consecutive_failures == 5
            assert config.restart_cooldown_seconds == 600
            assert config.min_alive_peers == 5


class TestP2PRecoveryDaemon:
    """Tests for P2PRecoveryDaemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with default config."""
        P2PRecoveryDaemon.reset_instance()
        config = P2PRecoveryConfig(startup_grace_seconds=0)
        return P2PRecoveryDaemon(config=config)

    def test_initialization(self, daemon):
        """Test daemon initialization."""
        assert daemon.config is not None
        assert daemon._consecutive_failures == 0
        assert daemon._last_restart_time == 0.0
        assert daemon._total_restarts == 0
        assert daemon._was_unhealthy is False

    def test_singleton(self):
        """Test singleton pattern."""
        P2PRecoveryDaemon.reset_instance()
        d1 = get_p2p_recovery_daemon()
        d2 = get_p2p_recovery_daemon()
        assert d1 is d2
        P2PRecoveryDaemon.reset_instance()

    def test_daemon_name(self, daemon):
        """Test daemon name."""
        assert daemon._get_daemon_name() == "P2PRecovery"

    def test_health_check_not_running(self, daemon):
        """Test health check when not running."""
        health = daemon.health_check()
        assert health.healthy is False
        assert health.status == CoordinatorStatus.STOPPED
        assert "not running" in health.message.lower()

    def test_health_check_running(self, daemon):
        """Test health check when running."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING

        health = daemon.health_check()
        assert health.healthy is True
        assert "healthy" in health.message.lower()

    def test_health_check_p2p_unhealthy(self, daemon):
        """Test health check when P2P is unhealthy."""
        daemon._running = True
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._consecutive_failures = daemon.config.max_consecutive_failures

        health = daemon.health_check()
        assert health.healthy is True  # Daemon is healthy, P2P is not
        assert "unhealthy" in health.message.lower()
        assert health.details["p2p_healthy"] is False

    def test_is_p2p_healthy(self, daemon):
        """Test is_p2p_healthy helper."""
        assert daemon.is_p2p_healthy() is True

        daemon._consecutive_failures = daemon.config.max_consecutive_failures
        assert daemon.is_p2p_healthy() is False

    def test_can_restart_first_time(self, daemon):
        """Test that first restart is always allowed."""
        assert daemon._can_restart() is True

    def test_can_restart_cooldown(self, daemon):
        """Test restart cooldown enforcement."""
        daemon._last_restart_time = time.time()
        assert daemon._can_restart() is False

        # After cooldown expires
        daemon._last_restart_time = time.time() - daemon.config.restart_cooldown_seconds - 1
        assert daemon._can_restart() is True

    def test_get_cooldown_remaining(self, daemon):
        """Test cooldown remaining calculation."""
        assert daemon._get_cooldown_remaining() == 0

        daemon._last_restart_time = time.time()
        remaining = daemon._get_cooldown_remaining()
        assert remaining > 0
        assert remaining <= daemon.config.restart_cooldown_seconds

    def test_get_status(self, daemon):
        """Test get_status includes P2P details."""
        daemon._total_restarts = 3
        daemon._consecutive_failures = 2

        status = daemon.get_status()
        assert "p2p_status" in status
        assert status["p2p_status"]["total_restarts"] == 3
        assert status["p2p_status"]["consecutive_failures"] == 2


class TestP2PRecoveryDaemonAsync:
    """Async tests for P2PRecoveryDaemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for async tests."""
        P2PRecoveryDaemon.reset_instance()
        config = P2PRecoveryConfig(startup_grace_seconds=0)
        return P2PRecoveryDaemon(config=config)

    @pytest.fixture
    def mock_aiohttp_success(self):
        """Create a properly configured aiohttp mock for successful responses."""
        async def mock_json():
            return {
                "alive_peers": 5,
                "leader_id": "node-1",
                "role": "leader",
            }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = mock_json

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        return mock_session

    @pytest.mark.asyncio
    async def test_check_p2p_health_success(self, daemon, mock_aiohttp_success):
        """Test successful P2P health check."""
        with patch("aiohttp.ClientSession", return_value=mock_aiohttp_success):
            is_healthy, status = await daemon._check_p2p_health()

            assert is_healthy is True
            assert status["alive_peers"] == 5
            assert status["leader_id"] == "node-1"

    @pytest.mark.asyncio
    async def test_check_p2p_health_too_few_peers(self, daemon):
        """Test P2P health check with too few peers."""
        async def mock_json():
            return {
                "alive_peers": 1,  # Below min_alive_peers
                "leader_id": "node-1",
                "role": "leader",
            }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = mock_json

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            is_healthy, status = await daemon._check_p2p_health()

            assert is_healthy is False
            assert status["alive_peers"] == 1

    @pytest.mark.asyncio
    async def test_check_p2p_health_no_leader(self, daemon):
        """Test P2P health check with no leader."""
        async def mock_json():
            return {
                "alive_peers": 5,
                "leader_id": None,  # No leader
                "role": "candidate",
            }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = mock_json

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            is_healthy, status = await daemon._check_p2p_health()

            assert is_healthy is False
            assert status["leader_id"] is None

    @pytest.mark.asyncio
    async def test_check_p2p_health_http_error(self, daemon):
        """Test P2P health check with HTTP error."""
        mock_response = MagicMock()
        mock_response.status = 500

        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            is_healthy, status = await daemon._check_p2p_health()

            assert is_healthy is False
            assert "error" in status

    @pytest.mark.asyncio
    async def test_check_p2p_health_timeout(self, daemon):
        """Test P2P health check with timeout."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            is_healthy, status = await daemon._check_p2p_health()

            assert is_healthy is False
            assert status["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_run_cycle_healthy(self, daemon):
        """Test run cycle when P2P is healthy."""
        daemon._consecutive_failures = 2  # Had some failures

        with patch.object(daemon, "_check_p2p_health", new_callable=AsyncMock) as mock:
            mock.return_value = (True, {"alive_peers": 5, "leader_id": "node-1"})
            await daemon._run_cycle()

            # Failures should be reset
            assert daemon._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_cycle_unhealthy_increment(self, daemon):
        """Test run cycle increments failures when unhealthy."""
        with patch.object(daemon, "_check_p2p_health", new_callable=AsyncMock) as mock:
            mock.return_value = (False, {"error": "timeout"})
            await daemon._run_cycle()

            assert daemon._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_cycle_triggers_restart(self, daemon):
        """Test run cycle triggers restart after max failures."""
        daemon._consecutive_failures = daemon.config.max_consecutive_failures - 1

        with patch.object(daemon, "_check_p2p_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = (False, {"error": "timeout"})

            with patch.object(daemon, "_restart_p2p", new_callable=AsyncMock) as mock_restart:
                await daemon._run_cycle()

                mock_restart.assert_called_once()
                assert daemon._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_cycle_respects_cooldown(self, daemon):
        """Test run cycle respects restart cooldown."""
        daemon._consecutive_failures = daemon.config.max_consecutive_failures - 1
        daemon._last_restart_time = time.time()  # Just restarted

        with patch.object(daemon, "_check_p2p_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = (False, {"error": "timeout"})

            with patch.object(daemon, "_restart_p2p", new_callable=AsyncMock) as mock_restart:
                await daemon._run_cycle()

                # Should not restart due to cooldown
                mock_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_grace_period(self, daemon):
        """Test run cycle respects startup grace period."""
        # Create daemon with grace period
        P2PRecoveryDaemon.reset_instance()
        config = P2PRecoveryConfig(startup_grace_seconds=300)
        daemon_with_grace = P2PRecoveryDaemon(config=config)

        with patch.object(daemon_with_grace, "_check_p2p_health", new_callable=AsyncMock) as mock_health:
            await daemon_with_grace._run_cycle()

            # Should skip check during grace period
            mock_health.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_p2p(self, daemon):
        """Test P2P restart process."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock(pid=12345)

                await daemon._restart_p2p()

                # Should have killed existing process
                mock_run.assert_called_once()
                assert "pkill" in mock_run.call_args[0][0]

        assert daemon._total_restarts == 1
        assert daemon._last_restart_time > 0

    @pytest.mark.asyncio
    async def test_emit_restart_event(self, daemon):
        """Test restart event emission doesn't crash."""
        daemon._consecutive_failures = 3
        daemon._total_restarts = 2

        # Just verify the method runs without error
        # Event emission may fail if infrastructure isn't available
        await daemon._emit_restart_event()

    @pytest.mark.asyncio
    async def test_emit_recovery_event(self, daemon):
        """Test recovery event emission doesn't crash."""
        status = {"alive_peers": 5, "leader_id": "node-1"}

        # Just verify the method runs without error
        # Event emission may fail if infrastructure isn't available
        await daemon._emit_recovery_event(status)

    @pytest.mark.asyncio
    async def test_recovery_event_after_unhealthy(self, daemon):
        """Test recovery event is emitted after being unhealthy."""
        daemon._was_unhealthy = True

        with patch.object(daemon, "_check_p2p_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = (True, {"alive_peers": 5, "leader_id": "node-1"})

            with patch.object(daemon, "_emit_recovery_event", new_callable=AsyncMock) as mock_emit:
                await daemon._run_cycle()

                mock_emit.assert_called_once()
                assert daemon._was_unhealthy is False
