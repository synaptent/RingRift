"""Tests for TailscaleChecker - Local Tailscale mesh connectivity checker.

Tests cover:
- Initialization and CLI detection
- Tailscale status parsing (mocked subprocess)
- Peer state extraction
- Config correlation and matching
- Disconnected node detection
- Error handling

December 2025 - P0 test coverage for node availability.
"""

import asyncio
import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.providers.tailscale_checker import (
    TailscaleChecker,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
)


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_tailscale_status():
    """Sample tailscale status --json output."""
    return {
        "BackendState": "Running",
        "Self": {
            "ID": "self123",
            "HostName": "local-mac",
            "TailscaleIPs": ["100.123.45.1", "fd7a::1"],
            "Online": True,
            "OS": "darwin",
        },
        "Peer": {
            "peer-abc": {
                "ID": "peer-abc",
                "HostName": "nebius-h100-1",
                "TailscaleIPs": ["100.123.45.10", "fd7a::10"],
                "Online": True,
                "OS": "linux",
            },
            "peer-def": {
                "ID": "peer-def",
                "HostName": "vast-12345",
                "TailscaleIPs": ["100.123.45.20", "fd7a::20"],
                "Online": True,
                "OS": "linux",
            },
            "peer-ghi": {
                "ID": "peer-ghi",
                "HostName": "runpod-h100",
                "TailscaleIPs": ["100.123.45.30", "fd7a::30"],
                "Online": False,  # Offline
                "OS": "linux",
            },
        },
    }


@pytest.fixture
def sample_config_hosts():
    """Sample distributed_hosts.yaml hosts section."""
    return {
        "nebius-h100-1": {
            "tailscale_ip": "100.123.45.10",
            "ssh_host": "89.169.111.139",
            "status": "ready",
        },
        "vast-12345": {
            "tailscale_ip": "100.123.45.20",
            "ssh_host": "192.168.1.100",
            "status": "ready",
        },
        "runpod-h100": {
            "tailscale_ip": "100.123.45.30",
            "ssh_host": "102.210.171.65",
            "status": "ready",
        },
        "hetzner-cpu1": {
            # No tailscale_ip - not in mesh
            "ssh_host": "162.55.1.1",
            "status": "ready",
        },
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTailscaleCheckerInit:
    """Tests for TailscaleChecker initialization."""

    def test_init_with_tailscale_installed(self):
        """Test initialization when tailscale CLI is available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/tailscale"

            checker = TailscaleChecker()

            assert checker.is_enabled
            assert checker._tailscale_path == "/usr/bin/tailscale"
            assert checker.provider_name == "tailscale"

    def test_init_without_tailscale(self):
        """Test initialization when tailscale CLI is not installed."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            checker = TailscaleChecker()

            assert not checker.is_enabled
            assert checker._tailscale_path is None

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker(timeout_seconds=30.0)

            assert checker._timeout == 30.0

    def test_provider_name(self):
        """Test provider name is correctly set."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

            assert checker.provider_name == "tailscale"


# =============================================================================
# API Availability Tests
# =============================================================================


class TestTailscaleApiAvailability:
    """Tests for checking tailscale connectivity."""

    @pytest.mark.asyncio
    async def test_api_available_when_running(self, sample_tailscale_status):
        """Test API available when tailscale is running."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_api_availability()

        assert result is True

    @pytest.mark.asyncio
    async def test_api_unavailable_when_stopped(self):
        """Test API unavailable when tailscale is stopped."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        status = {"BackendState": "Stopped"}
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_api_availability()

        assert result is False

    @pytest.mark.asyncio
    async def test_api_unavailable_on_error(self):
        """Test API unavailable when command fails."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_api_availability()

        assert result is False

    @pytest.mark.asyncio
    async def test_api_unavailable_on_timeout(self):
        """Test API unavailable on command timeout."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker(timeout_seconds=0.1)

        async def slow_communicate():
            await asyncio.sleep(1)
            return (b"", b"")

        mock_proc = MagicMock()
        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_api_availability()

        assert result is False

    @pytest.mark.asyncio
    async def test_api_unavailable_no_tailscale(self):
        """Test API unavailable when tailscale not installed."""
        with patch("shutil.which", return_value=None):
            checker = TailscaleChecker()

        result = await checker.check_api_availability()

        assert result is False


# =============================================================================
# Instance State Tests
# =============================================================================


class TestTailscaleInstanceStates:
    """Tests for getting peer instance states."""

    @pytest.mark.asyncio
    async def test_get_instance_states_success(self, sample_tailscale_status):
        """Test successful retrieval of peer states."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

        # Should get self + 3 peers = 4 instances
        assert len(instances) == 4

        # Check self is included
        self_inst = next(i for i in instances if i.hostname == "local-mac")
        assert self_inst.state == ProviderInstanceState.RUNNING
        assert self_inst.tailscale_ip == "100.123.45.1"

        # Check online peer
        online = next(i for i in instances if i.hostname == "nebius-h100-1")
        assert online.state == ProviderInstanceState.RUNNING
        assert online.tailscale_ip == "100.123.45.10"

        # Check offline peer
        offline = next(i for i in instances if i.hostname == "runpod-h100")
        assert offline.state == ProviderInstanceState.STOPPED
        assert offline.tailscale_ip == "100.123.45.30"

    @pytest.mark.asyncio
    async def test_get_instance_states_disabled(self):
        """Test returns empty when checker is disabled."""
        with patch("shutil.which", return_value=None):
            checker = TailscaleChecker()

        instances = await checker.get_instance_states()

        assert instances == []

    @pytest.mark.asyncio
    async def test_get_instance_states_command_failure(self):
        """Test handles command failure gracefully."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b"", b"tailscale not running")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

        assert instances == []
        assert checker._last_error is not None

    @pytest.mark.asyncio
    async def test_get_instance_states_json_error(self):
        """Test handles invalid JSON gracefully."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"not valid json", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

        assert instances == []
        assert "JSON parse error" in checker._last_error

    @pytest.mark.asyncio
    async def test_get_instance_states_no_peers(self):
        """Test handles status with no peers."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        status = {
            "BackendState": "Running",
            "Self": {
                "HostName": "local-mac",
                "TailscaleIPs": ["100.123.45.1"],
                "Online": True,
            },
            "Peer": {},
        }

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

        # Should only get self
        assert len(instances) == 1
        assert instances[0].hostname == "local-mac"


# =============================================================================
# Peer Parsing Tests
# =============================================================================


class TestPeerParsing:
    """Tests for parsing individual peer info."""

    def test_parse_peer_info_online(self):
        """Test parsing online peer."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        peer_data = {
            "HostName": "test-peer",
            "TailscaleIPs": ["100.1.2.3", "fd7a::1"],
            "Online": True,
        }

        result = checker._parse_peer_info(peer_data, is_self=False)

        assert result is not None
        assert result.hostname == "test-peer"
        assert result.state == ProviderInstanceState.RUNNING
        assert result.tailscale_ip == "100.1.2.3"
        assert result.provider == "tailscale"

    def test_parse_peer_info_offline(self):
        """Test parsing offline peer."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        peer_data = {
            "HostName": "offline-peer",
            "TailscaleIPs": ["100.1.2.4"],
            "Online": False,
        }

        result = checker._parse_peer_info(peer_data, is_self=False)

        assert result is not None
        assert result.state == ProviderInstanceState.STOPPED

    def test_parse_peer_info_self(self):
        """Test parsing self (always online)."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        peer_data = {
            "HostName": "local-node",
            "TailscaleIPs": ["100.1.2.5"],
        }

        result = checker._parse_peer_info(peer_data, is_self=True)

        assert result is not None
        assert result.state == ProviderInstanceState.RUNNING

    def test_parse_peer_info_no_hostname(self):
        """Test parsing peer without hostname returns None."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        peer_data = {
            "TailscaleIPs": ["100.1.2.6"],
            "Online": True,
        }

        result = checker._parse_peer_info(peer_data, is_self=False)

        assert result is None

    def test_parse_peer_info_ipv4_preference(self):
        """Test IPv4 address is preferred over IPv6."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        peer_data = {
            "HostName": "dual-stack",
            "TailscaleIPs": ["fd7a::1234", "100.1.2.7"],  # IPv6 first
            "Online": True,
        }

        result = checker._parse_peer_info(peer_data, is_self=False)

        assert result is not None
        assert result.tailscale_ip == "100.1.2.7"  # IPv4 selected


# =============================================================================
# Correlation Tests
# =============================================================================


class TestCorrelation:
    """Tests for correlating peers with config hosts."""

    def test_correlate_by_tailscale_ip(self, sample_config_hosts):
        """Test correlation by tailscale_ip match."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-abc",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                tailscale_ip="100.123.45.10",
                hostname="different-hostname",
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name == "nebius-h100-1"

    def test_correlate_by_hostname_exact(self, sample_config_hosts):
        """Test correlation by exact hostname match."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-xyz",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                hostname="runpod-h100",  # Exact match
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name == "runpod-h100"

    def test_correlate_by_hostname_case_insensitive(self, sample_config_hosts):
        """Test correlation is case insensitive."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-xyz",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                hostname="RunPod-H100",  # Different case
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name == "runpod-h100"

    def test_correlate_by_partial_hostname(self, sample_config_hosts):
        """Test correlation by partial hostname match."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-xyz",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                hostname="vast-12345-gpu",  # Contains "vast-12345"
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name == "vast-12345"

    def test_correlate_no_match(self, sample_config_hosts):
        """Test instance with no match keeps node_name None."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-unknown",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                hostname="unknown-host",
                tailscale_ip="100.99.99.99",
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name is None

    def test_correlate_already_set(self, sample_config_hosts):
        """Test already correlated instances are not changed."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        instances = [
            InstanceInfo(
                instance_id="peer-xyz",
                state=ProviderInstanceState.RUNNING,
                provider="tailscale",
                hostname="runpod-h100",
                node_name="already-set",  # Pre-set
            )
        ]

        result = checker.correlate_with_config(instances, sample_config_hosts)

        assert result[0].node_name == "already-set"


# =============================================================================
# Disconnected Node Detection Tests
# =============================================================================


class TestDisconnectedNodes:
    """Tests for detecting disconnected nodes."""

    @pytest.mark.asyncio
    async def test_get_disconnected_finds_offline(
        self, sample_tailscale_status, sample_config_hosts
    ):
        """Test finds nodes that are offline in Tailscale."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            disconnected = await checker.get_disconnected_nodes(sample_config_hosts)

        # runpod-h100 is offline in sample_tailscale_status
        assert "runpod-h100" in disconnected
        # nebius-h100-1 and vast-12345 are online
        assert "nebius-h100-1" not in disconnected
        assert "vast-12345" not in disconnected

    @pytest.mark.asyncio
    async def test_get_disconnected_ignores_no_tailscale_ip(
        self, sample_tailscale_status, sample_config_hosts
    ):
        """Test ignores nodes without tailscale_ip."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            disconnected = await checker.get_disconnected_nodes(sample_config_hosts)

        # hetzner-cpu1 has no tailscale_ip, should not be in disconnected
        assert "hetzner-cpu1" not in disconnected

    @pytest.mark.asyncio
    async def test_get_disconnected_ignores_retired(
        self, sample_tailscale_status
    ):
        """Test ignores already retired nodes."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        config_hosts = {
            "runpod-h100": {
                "tailscale_ip": "100.123.45.30",
                "status": "retired",  # Already retired
            },
        }

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            disconnected = await checker.get_disconnected_nodes(config_hosts)

        # Should not report already retired node
        assert disconnected == []

    @pytest.mark.asyncio
    async def test_get_disconnected_empty_on_error(self, sample_config_hosts):
        """Test returns empty list on API error."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            disconnected = await checker.get_disconnected_nodes(sample_config_hosts)

        # Should return empty on error (conservative)
        assert disconnected == []


# =============================================================================
# Online Node Detection Tests
# =============================================================================


class TestOnlineNodes:
    """Tests for detecting online nodes."""

    @pytest.mark.asyncio
    async def test_get_online_nodes(
        self, sample_tailscale_status, sample_config_hosts
    ):
        """Test finds online nodes that match config."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(json.dumps(sample_tailscale_status).encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            online = await checker.get_online_nodes(sample_config_hosts)

        # nebius-h100-1 and vast-12345 are online and in config
        assert "nebius-h100-1" in online
        assert "vast-12345" in online
        # runpod-h100 is offline
        assert "runpod-h100" not in online


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Tests for status reporting."""

    def test_get_status_enabled(self):
        """Test status when enabled."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        status = checker.get_status()

        assert status["provider"] == "tailscale"
        assert status["enabled"] is True
        assert status["last_check"] is None
        assert status["last_error"] is None

    def test_get_status_disabled(self):
        """Test status when disabled."""
        with patch("shutil.which", return_value=None):
            checker = TailscaleChecker()

        status = checker.get_status()

        assert status["enabled"] is False

    def test_get_status_after_check(self, sample_tailscale_status):
        """Test status after successful check."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        # Simulate successful check
        checker._last_check = datetime(2025, 12, 29, 12, 0, 0)
        checker._last_error = None

        status = checker.get_status()

        assert status["last_check"] == "2025-12-29T12:00:00"
        assert status["last_error"] is None

    def test_get_status_after_error(self):
        """Test status after error."""
        with patch("shutil.which", return_value="/usr/bin/tailscale"):
            checker = TailscaleChecker()

        checker._last_error = "Command timed out"

        status = checker.get_status()

        assert status["last_error"] == "Command timed out"
