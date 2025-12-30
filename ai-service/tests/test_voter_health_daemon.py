"""Tests for VoterHealthMonitorDaemon.

Covers:
- VoterHealthConfig dataclass and env loading
- VoterHealthState dataclass
- VoterHealthMonitorDaemon lifecycle (start/stop)
- Multi-transport probing (P2P HTTP → Tailscale → SSH)
- VOTER_OFFLINE/VOTER_ONLINE event emission
- QUORUM_LOST/QUORUM_RESTORED/QUORUM_AT_RISK event emission
- Health check implementation
- Singleton pattern (get/reset)

December 30, 2025: Created for voter health monitoring daemon tests.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.voter_health_daemon import (
    VoterHealthConfig,
    VoterHealthMonitorDaemon,
    VoterHealthState,
    get_voter_health_daemon,
    reset_voter_health_daemon,
)


# =============================================================================
# VoterHealthConfig Tests
# =============================================================================


class TestVoterHealthConfig:
    """Tests for VoterHealthConfig dataclass."""

    def test_default_values(self):
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

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VoterHealthConfig(
            enabled=False,
            check_interval_seconds=60,
            consecutive_failures_before_offline=3,
            p2p_timeout_seconds=10.0,
            tailscale_timeout_seconds=20.0,
            ssh_timeout_seconds=30.0,
            enable_ssh_fallback=False,
            quorum_size=5,
            quorum_warning_threshold=6,
            startup_grace_seconds=60,
            p2p_port=8771,
        )

        assert config.enabled is False
        assert config.check_interval_seconds == 60
        assert config.consecutive_failures_before_offline == 3
        assert config.p2p_timeout_seconds == 10.0
        assert config.tailscale_timeout_seconds == 20.0
        assert config.ssh_timeout_seconds == 30.0
        assert config.enable_ssh_fallback is False
        assert config.quorum_size == 5
        assert config.quorum_warning_threshold == 6
        assert config.p2p_port == 8771

    def test_from_env_enabled_false(self):
        """Test loading enabled=false from environment."""
        with patch.dict(os.environ, {"RINGRIFT_VOTER_HEALTH_ENABLED": "0"}):
            config = VoterHealthConfig.from_env()
            assert config.enabled is False

    def test_from_env_interval(self):
        """Test loading check_interval_seconds from environment."""
        with patch.dict(os.environ, {"RINGRIFT_VOTER_HEALTH_INTERVAL": "45"}):
            config = VoterHealthConfig.from_env()
            assert config.check_interval_seconds == 45

    def test_from_env_failures(self):
        """Test loading consecutive_failures_before_offline from environment."""
        with patch.dict(os.environ, {"RINGRIFT_VOTER_HEALTH_FAILURES": "5"}):
            config = VoterHealthConfig.from_env()
            assert config.consecutive_failures_before_offline == 5

    def test_from_env_ssh_fallback_disabled(self):
        """Test loading enable_ssh_fallback=false from environment."""
        with patch.dict(os.environ, {"RINGRIFT_VOTER_HEALTH_SSH_FALLBACK": "0"}):
            config = VoterHealthConfig.from_env()
            assert config.enable_ssh_fallback is False

    def test_from_env_port(self):
        """Test loading p2p_port from environment."""
        with patch.dict(os.environ, {"RINGRIFT_P2P_PORT": "8771"}):
            config = VoterHealthConfig.from_env()
            assert config.p2p_port == 8771

    def test_from_env_invalid_values_ignored(self):
        """Test that invalid env values are ignored (uses defaults)."""
        with patch.dict(os.environ, {
            "RINGRIFT_VOTER_HEALTH_INTERVAL": "not_a_number",
            "RINGRIFT_VOTER_HEALTH_FAILURES": "invalid",
        }):
            config = VoterHealthConfig.from_env()
            # Should use defaults
            assert config.check_interval_seconds == 30
            assert config.consecutive_failures_before_offline == 2


# =============================================================================
# VoterHealthState Tests
# =============================================================================


class TestVoterHealthState:
    """Tests for VoterHealthState dataclass."""

    def test_default_values(self):
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

    def test_custom_values(self):
        """Test custom state values."""
        state = VoterHealthState(
            voter_id="voter-1",
            tailscale_ip="100.64.0.1",
            ssh_host="voter1.example.com",
            is_online=False,
            consecutive_failures=3,
            last_successful_transport="p2p_http",
            failure_reason="Connection refused",
            total_checks=100,
            total_failures=10,
        )

        assert state.voter_id == "voter-1"
        assert state.tailscale_ip == "100.64.0.1"
        assert state.ssh_host == "voter1.example.com"
        assert state.is_online is False
        assert state.consecutive_failures == 3
        assert state.last_successful_transport == "p2p_http"
        assert state.failure_reason == "Connection refused"
        assert state.total_checks == 100
        assert state.total_failures == 10

    def test_last_seen_auto_initialized(self):
        """Test that last_seen is auto-initialized to current time."""
        before = time.time()
        state = VoterHealthState(voter_id="test")
        after = time.time()

        assert before <= state.last_seen <= after


# =============================================================================
# VoterHealthMonitorDaemon Lifecycle Tests
# =============================================================================


class TestVoterHealthMonitorDaemonLifecycle:
    """Tests for VoterHealthMonitorDaemon lifecycle."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    def test_initialization_with_default_config(self):
        """Test daemon initialization with default config."""
        daemon = VoterHealthMonitorDaemon()

        assert daemon._daemon_config.enabled is True
        assert daemon._daemon_config.check_interval_seconds == 30
        assert daemon.name == "VoterHealthMonitor"

    def test_initialization_with_custom_config(self):
        """Test daemon initialization with custom config."""
        config = VoterHealthConfig(check_interval_seconds=60, quorum_size=5)
        daemon = VoterHealthMonitorDaemon(config=config)

        assert daemon._daemon_config.check_interval_seconds == 60
        assert daemon._daemon_config.quorum_size == 5

    def test_singleton_pattern(self):
        """Test that get_voter_health_daemon returns singleton."""
        daemon1 = get_voter_health_daemon()
        daemon2 = get_voter_health_daemon()

        assert daemon1 is daemon2

    def test_singleton_reset(self):
        """Test that reset_voter_health_daemon clears singleton."""
        daemon1 = get_voter_health_daemon()
        reset_voter_health_daemon()
        daemon2 = get_voter_health_daemon()

        assert daemon1 is not daemon2

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test daemon start and stop."""
        daemon = VoterHealthMonitorDaemon()

        assert daemon._running is False

        await daemon.start()
        assert daemon._running is True

        await daemon.stop()
        assert daemon._running is False


# =============================================================================
# Transport Probing Tests
# =============================================================================


class TestTransportProbing:
    """Tests for multi-transport probing logic."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    @pytest.fixture
    def daemon(self):
        """Create daemon with mocked dependencies."""
        config = VoterHealthConfig(
            check_interval_seconds=30,
            startup_grace_seconds=0,  # Skip grace period for tests
        )
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {}
        daemon._voter_ips = {}
        daemon._voter_ssh = {}
        return daemon

    @pytest.mark.asyncio
    async def test_p2p_check_success(self, daemon):
        """Test P2P HTTP check succeeds."""
        with patch.object(daemon, "_check_p2p_reachable", new_callable=AsyncMock) as mock:
            mock.return_value = True

            result = await daemon._check_p2p_reachable("100.64.0.1")
            assert result is True
            mock.assert_called_once_with("100.64.0.1")

    @pytest.mark.asyncio
    async def test_probe_voter_p2p_success(self, daemon):
        """Test probe_voter succeeds with P2P."""
        with patch.object(daemon, "_check_p2p_reachable", new_callable=AsyncMock) as mock_p2p:
            mock_p2p.return_value = True

            # Initialize voter state and IPs
            daemon._voter_states["voter-1"] = VoterHealthState(voter_id="voter-1")
            daemon._voter_ips = {"voter-1": "100.64.0.1"}
            daemon._voter_ssh = {}

            is_reachable, transport = await daemon._probe_voter("voter-1")

            assert is_reachable is True
            assert transport == "p2p_http"

    @pytest.mark.asyncio
    async def test_probe_voter_fallback_to_tailscale(self, daemon):
        """Test probe_voter falls back to Tailscale when P2P fails."""
        with patch.object(daemon, "_check_p2p_reachable", new_callable=AsyncMock) as mock_p2p:
            with patch.object(daemon, "_check_tailscale_reachable", new_callable=AsyncMock) as mock_ts:
                mock_p2p.return_value = False
                mock_ts.return_value = True

                daemon._voter_states["voter-1"] = VoterHealthState(voter_id="voter-1")
                daemon._voter_ips = {"voter-1": "100.64.0.1"}
                daemon._voter_ssh = {}

                is_reachable, transport = await daemon._probe_voter("voter-1")

                assert is_reachable is True
                assert transport == "tailscale_ping"
                mock_p2p.assert_called_once()
                mock_ts.assert_called_once()

    @pytest.mark.asyncio
    async def test_probe_voter_fallback_to_ssh(self, daemon):
        """Test probe_voter falls back to SSH when P2P and Tailscale fail."""
        with patch.object(daemon, "_check_p2p_reachable", new_callable=AsyncMock) as mock_p2p:
            with patch.object(daemon, "_check_tailscale_reachable", new_callable=AsyncMock) as mock_ts:
                with patch.object(daemon, "_check_ssh_reachable", new_callable=AsyncMock) as mock_ssh:
                    mock_p2p.return_value = False
                    mock_ts.return_value = False
                    mock_ssh.return_value = True

                    daemon._voter_states["voter-1"] = VoterHealthState(voter_id="voter-1")
                    daemon._voter_ips = {"voter-1": "100.64.0.1"}
                    daemon._voter_ssh = {"voter-1": "voter1.example.com"}

                    is_reachable, transport = await daemon._probe_voter("voter-1")

                    assert is_reachable is True
                    assert transport == "ssh"
                    mock_ssh.assert_called_once()

    @pytest.mark.asyncio
    async def test_probe_voter_all_fail(self, daemon):
        """Test probe_voter returns unreachable when all transports fail."""
        with patch.object(daemon, "_check_p2p_reachable", new_callable=AsyncMock) as mock_p2p:
            with patch.object(daemon, "_check_tailscale_reachable", new_callable=AsyncMock) as mock_ts:
                with patch.object(daemon, "_check_ssh_reachable", new_callable=AsyncMock) as mock_ssh:
                    mock_p2p.return_value = False
                    mock_ts.return_value = False
                    mock_ssh.return_value = False

                    daemon._voter_states["voter-1"] = VoterHealthState(voter_id="voter-1")
                    daemon._voter_ips = {"voter-1": "100.64.0.1"}
                    daemon._voter_ssh = {"voter-1": "voter1.example.com"}

                    is_reachable, transport = await daemon._probe_voter("voter-1")

                    assert is_reachable is False
                    assert transport == "all_transports_failed"


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    @pytest.fixture
    def daemon(self):
        """Create daemon with mocked dependencies."""
        config = VoterHealthConfig(
            check_interval_seconds=30,
            startup_grace_seconds=0,
        )
        return VoterHealthMonitorDaemon(config=config)

    @pytest.mark.asyncio
    async def test_emit_voter_offline(self, daemon):
        """Test VOTER_OFFLINE event emission."""
        with patch("app.coordination.voter_health_daemon.emit_data_event") as mock_emit:
            daemon._emit_voter_offline("voter-1", "Connection refused")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0].value == "voter_offline"
            assert call_args[1]["voter_id"] == "voter-1"
            assert call_args[1]["reason"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_emit_voter_online(self, daemon):
        """Test VOTER_ONLINE event emission."""
        with patch("app.coordination.voter_health_daemon.emit_data_event") as mock_emit:
            daemon._emit_voter_online("voter-1")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0].value == "voter_online"
            assert call_args[1]["voter_id"] == "voter-1"

    @pytest.mark.asyncio
    async def test_emit_quorum_lost(self, daemon):
        """Test QUORUM_LOST event emission."""
        # Set up some offline voters so the event has the right data
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=False),
            "v3": VoterHealthState("v3", is_online=False),
        }

        with patch("app.coordination.voter_health_daemon.emit_data_event") as mock_emit:
            daemon._emit_quorum_lost(online_voters=1, total_voters=3)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0].value == "quorum_lost"
            assert call_args[1]["online_voters"] == 1
            assert call_args[1]["total_voters"] == 3

    @pytest.mark.asyncio
    async def test_emit_quorum_restored(self, daemon):
        """Test QUORUM_RESTORED event emission."""
        with patch("app.coordination.voter_health_daemon.emit_data_event") as mock_emit:
            daemon._emit_quorum_restored(online_voters=5, total_voters=6)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0].value == "quorum_restored"
            assert call_args[1]["online_voters"] == 5
            assert call_args[1]["total_voters"] == 6

    @pytest.mark.asyncio
    async def test_emit_quorum_at_risk(self, daemon):
        """Test QUORUM_AT_RISK event emission."""
        with patch("app.coordination.voter_health_daemon.emit_data_event") as mock_emit:
            daemon._emit_quorum_at_risk(online_voters=4, total_voters=5)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0].value == "quorum_at_risk"
            assert call_args[1]["online_voters"] == 4
            assert call_args[1]["total_voters"] == 5


# =============================================================================
# Quorum State Tests
# =============================================================================


class TestQuorumState:
    """Tests for quorum state tracking and transitions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    @pytest.fixture
    def daemon(self):
        """Create daemon with specific quorum config."""
        config = VoterHealthConfig(
            quorum_size=4,
            quorum_warning_threshold=5,
            startup_grace_seconds=0,
        )
        return VoterHealthMonitorDaemon(config=config)

    def test_initial_quorum_state(self, daemon):
        """Test initial quorum state is True (assume healthy)."""
        assert daemon._quorum_met is True

    def test_check_quorum_status_lost(self, daemon):
        """Test quorum lost detection."""
        # Set up 3 online voters (below quorum of 4)
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=True),
            "v3": VoterHealthState("v3", is_online=True),
            "v4": VoterHealthState("v4", is_online=False),
            "v5": VoterHealthState("v5", is_online=False),
        }
        daemon._quorum_met = True  # Previously had quorum

        with patch.object(daemon, "_emit_quorum_lost") as mock_emit:
            daemon._check_quorum_status()

            assert daemon._quorum_met is False
            mock_emit.assert_called_once()

    def test_check_quorum_status_restored(self, daemon):
        """Test quorum restored detection."""
        # Set up 4 online voters (at quorum of 4)
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=True),
            "v3": VoterHealthState("v3", is_online=True),
            "v4": VoterHealthState("v4", is_online=True),
            "v5": VoterHealthState("v5", is_online=False),
        }
        daemon._quorum_met = False  # Previously lost quorum

        with patch.object(daemon, "_emit_quorum_restored") as mock_emit:
            daemon._check_quorum_status()

            assert daemon._quorum_met is True
            mock_emit.assert_called_once()

    def test_check_quorum_status_at_risk(self, daemon):
        """Test quorum at risk detection (exactly at threshold)."""
        # Set up exactly 5 online voters (at warning threshold)
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=True),
            "v3": VoterHealthState("v3", is_online=True),
            "v4": VoterHealthState("v4", is_online=True),
            "v5": VoterHealthState("v5", is_online=True),
            "v6": VoterHealthState("v6", is_online=False),
        }
        daemon._quorum_met = True

        with patch.object(daemon, "_emit_quorum_at_risk") as mock_emit:
            daemon._check_quorum_status()

            assert daemon._quorum_met is True  # Still have quorum
            mock_emit.assert_called_once()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check implementation."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    def test_health_check_not_running(self):
        """Test health check returns not running status."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status.value == "stopped"
        assert "not running" in result.message.lower()

    def test_health_check_running_healthy(self):
        """Test health check returns healthy when running."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = True
        daemon._quorum_met = True
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
        }

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status.value == "running"
        assert "voters_online" in result.details

    def test_health_check_quorum_lost(self):
        """Test health check reflects quorum lost."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = True
        daemon._quorum_met = False
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=False),
        }

        result = daemon.health_check()

        # Daemon is still healthy, but quorum is not
        assert result.healthy is True
        assert result.details.get("quorum_met") is False


# =============================================================================
# State Transitions Tests
# =============================================================================


class TestStateTransitions:
    """Tests for voter state transitions (online/offline)."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    @pytest.fixture
    def daemon(self):
        """Create daemon for state transition tests."""
        config = VoterHealthConfig(
            consecutive_failures_before_offline=2,
            startup_grace_seconds=0,
        )
        daemon = VoterHealthMonitorDaemon(config=config)
        daemon._voter_states = {
            "voter-1": VoterHealthState(
                voter_id="voter-1",
                is_online=True,
                consecutive_failures=0,
            )
        }
        daemon._voter_ips = {"voter-1": "100.64.0.1"}
        daemon._voter_ssh = {"voter-1": "voter1.example.com"}
        return daemon

    @pytest.mark.asyncio
    async def test_voter_goes_offline_after_consecutive_failures(self, daemon):
        """Test voter goes offline after consecutive_failures_before_offline failures."""
        with patch.object(daemon, "_probe_voter", new_callable=AsyncMock) as mock_probe:
            with patch.object(daemon, "_emit_voter_offline", new_callable=AsyncMock) as mock_emit:
                mock_probe.return_value = (False, "all_transports_failed")

                # First failure
                await daemon._probe_and_update("voter-1")
                assert daemon._voter_states["voter-1"].consecutive_failures == 1
                assert daemon._voter_states["voter-1"].is_online is True

                # Second failure - should go offline
                await daemon._probe_and_update("voter-1")
                assert daemon._voter_states["voter-1"].consecutive_failures == 2
                assert daemon._voter_states["voter-1"].is_online is False
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_voter_comes_online_after_success(self, daemon):
        """Test offline voter comes online after successful probe."""
        daemon._voter_states["voter-1"].is_online = False
        daemon._voter_states["voter-1"].consecutive_failures = 5

        with patch.object(daemon, "_probe_voter", new_callable=AsyncMock) as mock_probe:
            with patch.object(daemon, "_emit_voter_online", new_callable=AsyncMock) as mock_emit:
                mock_probe.return_value = (True, "p2p_http")

                await daemon._probe_and_update("voter-1")

                assert daemon._voter_states["voter-1"].is_online is True
                assert daemon._voter_states["voter-1"].consecutive_failures == 0
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_count_resets_on_success(self, daemon):
        """Test consecutive failure count resets on success."""
        daemon._voter_states["voter-1"].consecutive_failures = 1  # 1 prior failure

        with patch.object(daemon, "_probe_voter", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = (True, "p2p_http")

            await daemon._probe_and_update("voter-1")

            assert daemon._voter_states["voter-1"].consecutive_failures == 0


# =============================================================================
# Get Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status() method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_voter_health_daemon()
        yield
        reset_voter_health_daemon()

    def test_get_status_includes_voter_details(self):
        """Test that get_status includes per-voter details."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = True
        daemon._voter_states = {
            "v1": VoterHealthState("v1", is_online=True),
            "v2": VoterHealthState("v2", is_online=False),
        }

        status = daemon.get_status()

        assert "voter_status" in status
        assert status["voter_status"]["total_voters"] == 2
        assert status["voter_status"]["online_voters"] == 1
        assert status["voter_status"]["offline_voters"] == 1

    def test_get_status_includes_quorum_info(self):
        """Test that get_status includes quorum information."""
        daemon = VoterHealthMonitorDaemon()
        daemon._running = True
        daemon._quorum_met = True

        status = daemon.get_status()

        assert "voter_status" in status
        assert status["voter_status"]["quorum_met"] is True
