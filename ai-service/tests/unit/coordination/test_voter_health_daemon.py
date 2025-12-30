"""Tests for VoterHealthMonitorDaemon.

December 30, 2025: Tests for voter health monitoring critical for 48h autonomous operation.

Coverage:
- VoterHealthConfig: Configuration dataclass and env loading
- VoterHealthState: Per-voter state tracking
- VoterHealthMonitorDaemon: Main daemon class
  - Multi-transport probing (P2P HTTP, Tailscale, SSH)
  - Quorum status tracking and events
  - Health check reporting
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.voter_health_daemon import (
    VoterHealthConfig,
    VoterHealthMonitorDaemon,
    VoterHealthState,
    get_voter_health_daemon,
    reset_voter_health_daemon,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset singleton before and after each test."""
    reset_voter_health_daemon()
    yield
    reset_voter_health_daemon()


class TestVoterHealthConfig:
    """Tests for VoterHealthConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VoterHealthConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 30
        assert config.consecutive_failures_before_offline == 2
        assert config.p2p_timeout_seconds == 5.0
        assert config.tailscale_timeout_seconds == 10.0
        assert config.ssh_timeout_seconds == 15.0
        assert config.enable_ssh_fallback is True
        assert config.quorum_size == 4
        assert config.quorum_warning_threshold == 5
        assert config.startup_grace_seconds == 30
        assert config.p2p_port == 8770

    def test_from_env_default(self) -> None:
        """Test from_env with no environment variables set."""
        with patch.dict("os.environ", {}, clear=True):
            config = VoterHealthConfig.from_env()
            assert config.check_interval_seconds == 30
            assert config.consecutive_failures_before_offline == 2

    def test_from_env_custom_values(self) -> None:
        """Test from_env with custom environment variables."""
        env = {
            "RINGRIFT_VOTER_HEALTH_ENABLED": "0",
            "RINGRIFT_VOTER_HEALTH_INTERVAL": "60",
            "RINGRIFT_VOTER_HEALTH_FAILURES": "3",
            "RINGRIFT_VOTER_HEALTH_SSH_FALLBACK": "0",
            "RINGRIFT_P2P_PORT": "9999",
        }
        with patch.dict("os.environ", env, clear=True):
            config = VoterHealthConfig.from_env()
            assert config.enabled is False
            assert config.check_interval_seconds == 60
            assert config.consecutive_failures_before_offline == 3
            assert config.enable_ssh_fallback is False
            assert config.p2p_port == 9999

    def test_from_env_invalid_values(self) -> None:
        """Test from_env with invalid values (should use defaults)."""
        env = {
            "RINGRIFT_VOTER_HEALTH_INTERVAL": "not_a_number",
            "RINGRIFT_VOTER_HEALTH_FAILURES": "invalid",
            "RINGRIFT_P2P_PORT": "bad",
        }
        with patch.dict("os.environ", env, clear=True):
            config = VoterHealthConfig.from_env()
            # Should use defaults on parse failure
            assert config.check_interval_seconds == 30
            assert config.consecutive_failures_before_offline == 2
            assert config.p2p_port == 8770


class TestVoterHealthState:
    """Tests for VoterHealthState dataclass."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = VoterHealthState(voter_id="test-voter")
        assert state.voter_id == "test-voter"
        assert state.tailscale_ip == ""
        assert state.ssh_host == ""
        assert state.is_online is True
        assert state.consecutive_failures == 0
        assert state.last_successful_transport == "unknown"
        assert state.failure_reason == ""
        assert state.total_checks == 0
        assert state.total_failures == 0

    def test_custom_values(self) -> None:
        """Test state with custom values."""
        state = VoterHealthState(
            voter_id="voter-1",
            tailscale_ip="100.1.2.3",
            ssh_host="user@host:22",
            is_online=False,
            consecutive_failures=3,
            failure_reason="all_transports_failed",
        )
        assert state.voter_id == "voter-1"
        assert state.tailscale_ip == "100.1.2.3"
        assert state.ssh_host == "user@host:22"
        assert state.is_online is False
        assert state.consecutive_failures == 3
        assert state.failure_reason == "all_transports_failed"


class TestVoterHealthMonitorDaemon:
    """Tests for VoterHealthMonitorDaemon class."""

    # =========================================================================
    # Initialization
    # =========================================================================

    def test_init_default_config(self) -> None:
        """Test daemon initialization with default config."""
        daemon = VoterHealthMonitorDaemon()
        assert daemon.name == "VoterHealthMonitor"
        assert daemon.config.enabled is True
        assert daemon._voter_states == {}
        assert daemon._had_quorum is True

    def test_init_custom_config(self) -> None:
        """Test daemon initialization with custom config."""
        config = VoterHealthConfig(check_interval_seconds=10, quorum_size=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        assert daemon.config.check_interval_seconds == 10
        assert daemon.config.quorum_size == 3

    @pytest.mark.asyncio
    async def test_load_voters(self) -> None:
        """Test voter loading from cluster config."""
        mock_voters = ["voter-1", "voter-2", "voter-3"]
        mock_config = MagicMock()
        mock_config.hosts_raw = {
            "voter-1": {"tailscale_ip": "100.1.1.1", "ssh_host": "user@host1:22"},
            "voter-2": {"tailscale_ip": "100.1.1.2", "ssh_host": "user@host2:22"},
            "voter-3": {"tailscale_ip": "100.1.1.3", "ssh_host": ""},
        }

        daemon = VoterHealthMonitorDaemon()

        with patch(
            "app.config.cluster_config.get_p2p_voters", return_value=mock_voters
        ), patch(
            "app.config.cluster_config.load_cluster_config", return_value=mock_config
        ):
            daemon._load_voters()

        assert len(daemon._voter_states) == 3
        assert "voter-1" in daemon._voter_states
        assert daemon._voter_ips["voter-1"] == "100.1.1.1"
        assert daemon._voter_ssh["voter-1"] == "user@host1:22"

    # =========================================================================
    # Probing
    # =========================================================================

    @pytest.mark.asyncio
    async def test_probe_voter_p2p_success(self) -> None:
        """Test voter probe succeeds via P2P HTTP."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_ips["voter-1"] = "100.1.1.1"

        with patch.object(daemon, "_check_p2p_reachable", return_value=True):
            is_reachable, transport = await daemon._probe_voter("voter-1")

        assert is_reachable is True
        assert transport == "p2p_http"

    @pytest.mark.asyncio
    async def test_probe_voter_tailscale_fallback(self) -> None:
        """Test voter probe falls back to Tailscale ping."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_ips["voter-1"] = "100.1.1.1"

        with patch.object(daemon, "_check_p2p_reachable", return_value=False), patch.object(
            daemon, "_check_tailscale_reachable", return_value=True
        ):
            is_reachable, transport = await daemon._probe_voter("voter-1")

        assert is_reachable is True
        assert transport == "tailscale_ping"

    @pytest.mark.asyncio
    async def test_probe_voter_ssh_fallback(self) -> None:
        """Test voter probe falls back to SSH."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_ips["voter-1"] = "100.1.1.1"
        daemon._voter_ssh["voter-1"] = "user@host:22"

        with patch.object(daemon, "_check_p2p_reachable", return_value=False), patch.object(
            daemon, "_check_tailscale_reachable", return_value=False
        ), patch.object(daemon, "_check_ssh_reachable", return_value=True):
            is_reachable, transport = await daemon._probe_voter("voter-1")

        assert is_reachable is True
        assert transport == "ssh"

    @pytest.mark.asyncio
    async def test_probe_voter_all_fail(self) -> None:
        """Test voter probe when all transports fail."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_ips["voter-1"] = "100.1.1.1"
        daemon._voter_ssh["voter-1"] = "user@host:22"

        with patch.object(daemon, "_check_p2p_reachable", return_value=False), patch.object(
            daemon, "_check_tailscale_reachable", return_value=False
        ), patch.object(daemon, "_check_ssh_reachable", return_value=False):
            is_reachable, transport = await daemon._probe_voter("voter-1")

        assert is_reachable is False
        assert transport == "all_transports_failed"

    @pytest.mark.asyncio
    async def test_probe_voter_ssh_disabled(self) -> None:
        """Test voter probe with SSH fallback disabled."""
        config = VoterHealthConfig(enable_ssh_fallback=False)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_ips["voter-1"] = "100.1.1.1"
        daemon._voter_ssh["voter-1"] = "user@host:22"

        with patch.object(daemon, "_check_p2p_reachable", return_value=False), patch.object(
            daemon, "_check_tailscale_reachable", return_value=False
        ), patch.object(daemon, "_check_ssh_reachable", return_value=True) as ssh_mock:
            is_reachable, _ = await daemon._probe_voter("voter-1")

        assert is_reachable is False
        ssh_mock.assert_not_called()  # SSH should not be tried

    # =========================================================================
    # State Updates
    # =========================================================================

    @pytest.mark.asyncio
    async def test_probe_and_update_success(self) -> None:
        """Test probe_and_update on successful probe."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states["voter-1"] = VoterHealthState(
            voter_id="voter-1",
            is_online=True,
            consecutive_failures=1,
        )

        with patch.object(daemon, "_probe_voter", return_value=(True, "p2p_http")):
            await daemon._probe_and_update("voter-1")

        state = daemon._voter_states["voter-1"]
        assert state.is_online is True
        assert state.consecutive_failures == 0
        assert state.last_successful_transport == "p2p_http"
        assert state.total_checks == 1

    @pytest.mark.asyncio
    async def test_probe_and_update_failure_not_offline(self) -> None:
        """Test probe_and_update on failure (not enough to go offline)."""
        config = VoterHealthConfig(consecutive_failures_before_offline=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states["voter-1"] = VoterHealthState(
            voter_id="voter-1",
            is_online=True,
            consecutive_failures=0,
        )

        with patch.object(daemon, "_probe_voter", return_value=(False, "timeout")):
            await daemon._probe_and_update("voter-1")

        state = daemon._voter_states["voter-1"]
        assert state.is_online is True  # Still online
        assert state.consecutive_failures == 1
        assert state.total_failures == 1

    @pytest.mark.asyncio
    async def test_probe_and_update_goes_offline(self) -> None:
        """Test probe_and_update when voter goes offline."""
        config = VoterHealthConfig(consecutive_failures_before_offline=2)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states["voter-1"] = VoterHealthState(
            voter_id="voter-1",
            is_online=True,
            consecutive_failures=1,  # One more failure will trigger offline
        )

        with patch.object(daemon, "_probe_voter", return_value=(False, "all_transports_failed")):
            with patch.object(daemon, "_emit_voter_offline", new_callable=AsyncMock) as emit_mock:
                await daemon._probe_and_update("voter-1")

        state = daemon._voter_states["voter-1"]
        assert state.is_online is False
        assert state.consecutive_failures == 2
        emit_mock.assert_called_once_with("voter-1", "all_transports_failed")

    @pytest.mark.asyncio
    async def test_probe_and_update_comes_online(self) -> None:
        """Test probe_and_update when offline voter comes back online."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states["voter-1"] = VoterHealthState(
            voter_id="voter-1",
            is_online=False,  # Was offline
            consecutive_failures=5,
        )

        with patch.object(daemon, "_probe_voter", return_value=(True, "tailscale_ping")):
            with patch.object(daemon, "_emit_voter_online", new_callable=AsyncMock) as emit_mock:
                await daemon._probe_and_update("voter-1")

        state = daemon._voter_states["voter-1"]
        assert state.is_online is True
        assert state.consecutive_failures == 0
        emit_mock.assert_called_once_with("voter-1", "tailscale_ping")

    # =========================================================================
    # Quorum Status
    # =========================================================================

    def test_check_quorum_healthy(self) -> None:
        """Test quorum check when healthy."""
        config = VoterHealthConfig(quorum_size=3, quorum_warning_threshold=4)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
            "v4": VoterHealthState(voter_id="v4", is_online=True),
            "v5": VoterHealthState(voter_id="v5", is_online=True),
        }
        daemon._had_quorum = True

        daemon._check_quorum_status()

        assert daemon._stats_extra["voters_online"] == 5
        assert daemon._stats_extra["voters_offline"] == 0
        assert daemon._had_quorum is True

    def test_check_quorum_at_risk(self) -> None:
        """Test quorum check when at risk (marginal)."""
        config = VoterHealthConfig(quorum_size=3, quorum_warning_threshold=4)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
            "v4": VoterHealthState(voter_id="v4", is_online=True),
            "v5": VoterHealthState(voter_id="v5", is_online=False),
        }
        daemon._had_quorum = True
        daemon._quorum_at_risk_emitted = False

        with patch.object(daemon, "_emit_quorum_at_risk") as emit_mock:
            daemon._check_quorum_status()

        emit_mock.assert_called_once_with(4, 5)
        assert daemon._quorum_at_risk_emitted is True

    def test_check_quorum_lost(self) -> None:
        """Test quorum check when quorum is lost."""
        config = VoterHealthConfig(quorum_size=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=False),
            "v4": VoterHealthState(voter_id="v4", is_online=False),
        }
        daemon._had_quorum = True  # Previously had quorum

        with patch.object(daemon, "_emit_quorum_lost") as emit_mock:
            daemon._check_quorum_status()

        emit_mock.assert_called_once_with(2, 4)
        assert daemon._had_quorum is False
        assert daemon._stats_extra["quorum_lost_count"] == 1

    def test_check_quorum_restored(self) -> None:
        """Test quorum check when quorum is restored."""
        config = VoterHealthConfig(quorum_size=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
            "v4": VoterHealthState(voter_id="v4", is_online=False),
        }
        daemon._had_quorum = False  # Previously lost quorum

        with patch.object(daemon, "_emit_quorum_restored") as emit_mock:
            daemon._check_quorum_status()

        emit_mock.assert_called_once_with(3, 4)
        assert daemon._had_quorum is True
        assert daemon._stats_extra["quorum_restored_count"] == 1

    # =========================================================================
    # Health Check
    # =========================================================================

    def test_health_check_not_running(self) -> None:
        """Test health check when daemon is not running."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = False

        health = daemon.health_check()

        assert health.healthy is False
        assert "not running" in health.message

    def test_health_check_with_quorum(self) -> None:
        """Test health check when quorum is healthy."""
        config = VoterHealthConfig(quorum_size=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._running = True
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
            "v4": VoterHealthState(voter_id="v4", is_online=True),
        }

        health = daemon.health_check()

        assert health.healthy is True
        assert "4/4" in health.message
        assert "healthy" in health.message.lower()

    def test_health_check_quorum_at_risk(self) -> None:
        """Test health check when quorum is at risk."""
        config = VoterHealthConfig(quorum_size=3, quorum_warning_threshold=4)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._running = True
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
            "v4": VoterHealthState(voter_id="v4", is_online=True),
            "v5": VoterHealthState(voter_id="v5", is_online=False),
        }

        health = daemon.health_check()

        assert health.healthy is True  # Still healthy, just at risk
        assert "at risk" in health.message.lower()

    def test_health_check_no_quorum(self) -> None:
        """Test health check when quorum is lost."""
        config = VoterHealthConfig(quorum_size=3)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._running = True
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=True),
            "v3": VoterHealthState(voter_id="v3", is_online=False),
            "v4": VoterHealthState(voter_id="v4", is_online=False),
        }

        health = daemon.health_check()

        assert health.healthy is False
        assert "QUORUM LOST" in health.message

    # =========================================================================
    # Status Methods
    # =========================================================================

    def test_get_status_details(self) -> None:
        """Test get_status_details returns correct structure."""
        config = VoterHealthConfig(quorum_size=2)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "v1": VoterHealthState(
                voter_id="v1",
                is_online=True,
                last_successful_transport="p2p_http",
            ),
            "v2": VoterHealthState(
                voter_id="v2",
                is_online=False,
                failure_reason="timeout",
            ),
        }

        details = daemon._get_status_details()

        assert len(details["online_voters"]) == 1
        assert len(details["offline_voters"]) == 1
        assert details["quorum_size"] == 2
        assert details["has_quorum"] is False
        assert "timeout" in str(details["offline_voters"][0]["failure_reason"])

    def test_get_voter_state(self) -> None:
        """Test get_voter_state returns correct state."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states["voter-1"] = VoterHealthState(
            voter_id="voter-1", is_online=True
        )

        state = daemon.get_voter_state("voter-1")
        assert state is not None
        assert state.voter_id == "voter-1"

        state_missing = daemon.get_voter_state("nonexistent")
        assert state_missing is None

    def test_get_online_voters(self) -> None:
        """Test get_online_voters returns correct list."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=False),
            "v3": VoterHealthState(voter_id="v3", is_online=True),
        }

        online = daemon.get_online_voters()
        assert len(online) == 2
        assert "v1" in online
        assert "v3" in online
        assert "v2" not in online

    def test_get_offline_voters(self) -> None:
        """Test get_offline_voters returns correct list."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=True),
            "v2": VoterHealthState(voter_id="v2", is_online=False),
            "v3": VoterHealthState(voter_id="v3", is_online=False),
        }

        offline = daemon.get_offline_voters()
        assert len(offline) == 2
        assert "v2" in offline
        assert "v3" in offline
        assert "v1" not in offline

    # =========================================================================
    # Run Cycle
    # =========================================================================

    @pytest.mark.asyncio
    async def test_run_cycle_during_grace_period(self) -> None:
        """Test run_cycle skips during startup grace period."""
        config = VoterHealthConfig(startup_grace_seconds=60)
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._startup_time = time.time()  # Just started
        daemon._voter_states = {"v1": VoterHealthState(voter_id="v1")}

        with patch.object(daemon, "_probe_and_update", new_callable=AsyncMock) as probe_mock:
            await daemon._run_cycle()

        probe_mock.assert_not_called()  # Should skip during grace period

    @pytest.mark.asyncio
    async def test_run_cycle_no_voters(self) -> None:
        """Test run_cycle loads voters if none configured."""
        daemon = VoterHealthMonitorDaemon()
        daemon._startup_time = time.time() - 100  # Past grace period

        with patch.object(daemon, "_load_voters") as load_mock:
            await daemon._run_cycle()

        load_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_probes_all_voters(self) -> None:
        """Test run_cycle probes all voters."""
        daemon = VoterHealthMonitorDaemon()
        daemon._startup_time = time.time() - 100  # Past grace period
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1"),
            "v2": VoterHealthState(voter_id="v2"),
        }

        with patch.object(daemon, "_probe_and_update", new_callable=AsyncMock) as probe_mock:
            with patch.object(daemon, "_check_quorum_status"):
                await daemon._run_cycle()

        assert probe_mock.call_count == 2


class TestSingletonAccessors:
    """Tests for singleton accessor functions."""

    def test_get_voter_health_daemon_singleton(self) -> None:
        """Test get_voter_health_daemon returns singleton."""
        daemon1 = get_voter_health_daemon()
        daemon2 = get_voter_health_daemon()
        assert daemon1 is daemon2

    def test_reset_voter_health_daemon(self) -> None:
        """Test reset_voter_health_daemon creates new instance."""
        daemon1 = get_voter_health_daemon()
        reset_voter_health_daemon()
        daemon2 = get_voter_health_daemon()
        assert daemon1 is not daemon2


class TestEventEmission:
    """Tests for event emission methods."""

    @pytest.mark.asyncio
    async def test_emit_voter_offline(self) -> None:
        """Test VOTER_OFFLINE event emission."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states["v1"] = VoterHealthState(
            voter_id="v1",
            last_seen=time.time() - 60,
            consecutive_failures=3,
        )

        with patch("app.coordination.voter_health_daemon.emit_data_event") as emit_mock:
            await daemon._emit_voter_offline("v1", "timeout")

        emit_mock.assert_called_once()
        call_kwargs = emit_mock.call_args[1]
        assert call_kwargs["voter_id"] == "v1"
        assert call_kwargs["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_emit_voter_online(self) -> None:
        """Test VOTER_ONLINE event emission."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states["v1"] = VoterHealthState(
            voter_id="v1",
            last_seen=time.time() - 120,
        )

        with patch("app.coordination.voter_health_daemon.emit_data_event") as emit_mock:
            await daemon._emit_voter_online("v1", "p2p_http")

        emit_mock.assert_called_once()
        call_kwargs = emit_mock.call_args[1]
        assert call_kwargs["voter_id"] == "v1"
        assert call_kwargs["transport"] == "p2p_http"

    def test_emit_quorum_lost(self) -> None:
        """Test QUORUM_LOST event emission."""
        daemon = VoterHealthMonitorDaemon()
        daemon._voter_states = {
            "v1": VoterHealthState(voter_id="v1", is_online=False),
            "v2": VoterHealthState(voter_id="v2", is_online=False),
        }

        with patch("app.coordination.voter_health_daemon.emit_data_event") as emit_mock:
            daemon._emit_quorum_lost(2, 5)

        emit_mock.assert_called_once()
        call_kwargs = emit_mock.call_args[1]
        assert call_kwargs["online_voters"] == 2
        assert call_kwargs["total_voters"] == 5

    def test_emit_quorum_restored(self) -> None:
        """Test QUORUM_RESTORED event emission."""
        daemon = VoterHealthMonitorDaemon()

        with patch("app.coordination.voter_health_daemon.emit_data_event") as emit_mock:
            daemon._emit_quorum_restored(4, 5)

        emit_mock.assert_called_once()
        call_kwargs = emit_mock.call_args[1]
        assert call_kwargs["online_voters"] == 4
        assert call_kwargs["total_voters"] == 5

    def test_emit_quorum_at_risk(self) -> None:
        """Test QUORUM_AT_RISK event emission."""
        daemon = VoterHealthMonitorDaemon()

        with patch("app.coordination.voter_health_daemon.emit_data_event") as emit_mock:
            daemon._emit_quorum_at_risk(4, 7)

        emit_mock.assert_called_once()
        call_kwargs = emit_mock.call_args[1]
        assert call_kwargs["online_voters"] == 4
        assert call_kwargs["total_voters"] == 7
        assert "margin" in call_kwargs
