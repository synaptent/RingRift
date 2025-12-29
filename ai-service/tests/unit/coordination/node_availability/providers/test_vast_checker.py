"""Tests for VastChecker - Vast.ai node availability state checker.

Tests cover:
- Initialization and API key detection
- CLI command execution (mocked)
- Instance state parsing
- Config correlation and matching
- Terminated instance detection
- Error handling

December 2025 - P0 test coverage for node availability.
"""

import asyncio
import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.providers.vast_checker import (
    VastChecker,
    VAST_STATE_MAP,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
)


class TestVastCheckerInit:
    """Tests for VastChecker initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        checker = VastChecker(api_key="test-key-123")

        assert checker._api_key == "test-key-123"
        assert checker.is_enabled

    def test_init_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"VAST_API_KEY": "env-key-456"}):
            checker = VastChecker()

            assert checker._api_key == "env-key-456"
            assert checker.is_enabled

    def test_init_from_config_file(self):
        """Test initialization from config file."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "vast_api_key" in p

                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = "file-key-789\n"

                    checker = VastChecker()

                    assert checker._api_key == "file-key-789"

    def test_init_no_api_key(self):
        """Test initialization with no API key available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False

                checker = VastChecker()

                assert not checker.is_enabled

    def test_provider_name(self):
        """Test provider name."""
        checker = VastChecker(api_key="test")

        assert checker.provider_name == "vast"


class TestVastCheckerStateMappings:
    """Tests for state mapping constants."""

    def test_running_state(self):
        """Test 'running' maps to RUNNING."""
        assert VAST_STATE_MAP["running"] == ProviderInstanceState.RUNNING

    def test_loading_state(self):
        """Test 'loading' maps to STARTING."""
        assert VAST_STATE_MAP["loading"] == ProviderInstanceState.STARTING

    def test_exited_state(self):
        """Test 'exited' maps to STOPPED."""
        assert VAST_STATE_MAP["exited"] == ProviderInstanceState.STOPPED

    def test_destroying_state(self):
        """Test 'destroying' maps to STOPPING."""
        assert VAST_STATE_MAP["destroying"] == ProviderInstanceState.STOPPING

    def test_created_state(self):
        """Test 'created' maps to STARTING."""
        assert VAST_STATE_MAP["created"] == ProviderInstanceState.STARTING


class TestVastCheckerApiAvailability:
    """Tests for API availability checking."""

    @pytest.mark.asyncio
    async def test_api_available_no_key(self):
        """Test API not available when no key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = VastChecker()

                result = await checker.check_api_availability()

                assert result is False

    @pytest.mark.asyncio
    async def test_api_available_cli_found(self):
        """Test API available when CLI works."""
        checker = VastChecker(api_key="test-key")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await checker.check_api_availability()

            assert result is True

    @pytest.mark.asyncio
    async def test_api_available_cli_not_found(self):
        """Test API not available when CLI missing."""
        checker = VastChecker(api_key="test-key")

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            result = await checker.check_api_availability()

            assert result is False


class TestVastCheckerGetInstanceStates:
    """Tests for get_instance_states method."""

    @pytest.mark.asyncio
    async def test_get_states_disabled(self):
        """Test returns empty when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = VastChecker()

                instances = await checker.get_instance_states()

                assert instances == []

    @pytest.mark.asyncio
    async def test_get_states_success(self):
        """Test successful instance state retrieval."""
        checker = VastChecker(api_key="test-key")

        mock_output = json.dumps([
            {
                "id": 12345,
                "actual_status": "running",
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 22001,
                "public_ipaddr": "1.2.3.4",
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "gpu_ram": 24576,  # MB
            },
            {
                "id": 67890,
                "actual_status": "loading",
                "ssh_host": "ssh2.vast.ai",
                "ssh_port": 22002,
                "public_ipaddr": "5.6.7.8",
                "gpu_name": "RTX 3090",
                "num_gpus": 2,
                "gpu_ram": 24576,
            },
        ])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

            assert len(instances) == 2
            assert instances[0].instance_id == "12345"
            assert instances[0].state == ProviderInstanceState.RUNNING
            assert instances[0].ssh_host == "ssh1.vast.ai"
            assert instances[0].ssh_port == 22001
            assert instances[0].gpu_type == "RTX 4090"
            assert instances[0].gpu_count == 1

            assert instances[1].instance_id == "67890"
            assert instances[1].state == ProviderInstanceState.STARTING
            assert instances[1].gpu_count == 2

    @pytest.mark.asyncio
    async def test_get_states_cli_error(self):
        """Test handles CLI errors gracefully."""
        checker = VastChecker(api_key="test-key")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

            assert instances == []
            assert checker._last_error is not None

    @pytest.mark.asyncio
    async def test_get_states_json_error(self):
        """Test handles invalid JSON gracefully."""
        checker = VastChecker(api_key="test-key")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not valid json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            instances = await checker.get_instance_states()

            assert instances == []
            assert "JSON" in checker._last_error or "parse" in checker._last_error


class TestVastCheckerCorrelation:
    """Tests for correlate_with_config method."""

    def test_correlate_by_id(self):
        """Test correlation by instance ID in node name."""
        checker = VastChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="12345",
                state=ProviderInstanceState.RUNNING,
                provider="vast",
                ssh_host="ssh1.vast.ai",
                ssh_port=22001,
            ),
        ]

        config_hosts = {
            "vast-12345-rtx4090": {
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 22001,
            },
            "runpod-h100": {
                "ssh_host": "other.host",
                "ssh_port": 22,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "vast-12345-rtx4090"

    def test_correlate_by_ssh(self):
        """Test correlation by SSH host/port."""
        checker = VastChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="99999",
                state=ProviderInstanceState.RUNNING,
                provider="vast",
                ssh_host="ssh1.vast.ai",
                ssh_port=22001,
            ),
        ]

        config_hosts = {
            "vast-custom-node": {
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 22001,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "vast-custom-node"

    def test_correlate_no_match(self):
        """Test correlation when no match found."""
        checker = VastChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="99999",
                state=ProviderInstanceState.RUNNING,
                provider="vast",
                ssh_host="unknown.host",
                ssh_port=99999,
            ),
        ]

        config_hosts = {
            "vast-12345": {
                "ssh_host": "ssh1.vast.ai",
                "ssh_port": 22001,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name is None


class TestVastCheckerTerminatedInstances:
    """Tests for get_terminated_instances method."""

    @pytest.mark.asyncio
    async def test_find_terminated(self):
        """Test finding terminated instances."""
        checker = VastChecker(api_key="test-key")

        # Mock current active instances
        mock_output = json.dumps([
            {"id": 12345, "actual_status": "running"},
        ])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))

        config_hosts = {
            "vast-12345": {"status": "ready"},
            "vast-67890": {"status": "ready"},  # Not in API response
            "vast-11111": {"status": "retired"},  # Already retired
            "runpod-h100": {"status": "ready"},  # Different provider
        }

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            terminated = await checker.get_terminated_instances(config_hosts)

            assert "vast-67890" in terminated
            assert "vast-12345" not in terminated
            assert "vast-11111" not in terminated  # Already retired
            assert "runpod-h100" not in terminated  # Different provider

    @pytest.mark.asyncio
    async def test_find_terminated_api_failure(self):
        """Test handles API failure gracefully."""
        checker = VastChecker(api_key="test-key")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error"))

        config_hosts = {
            "vast-12345": {"status": "ready"},
        }

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            terminated = await checker.get_terminated_instances(config_hosts)

            # Should return empty on API failure (conservative)
            assert terminated == []
