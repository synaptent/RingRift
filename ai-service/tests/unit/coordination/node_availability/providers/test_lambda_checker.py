"""Tests for LambdaChecker - Lambda Labs node availability state checker.

Tests cover:
- Initialization and API key detection
- HTTP API requests (mocked)
- Instance state parsing
- Config correlation and matching
- Terminated instance detection
- Error handling

December 2025 - P0 test coverage for node availability.
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.node_availability.providers.lambda_checker import (
    LambdaChecker,
    LAMBDA_STATE_MAP,
    LAMBDA_API_BASE,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
)


class TestLambdaCheckerInit:
    """Tests for LambdaChecker initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        checker = LambdaChecker(api_key="test-key-123")

        assert checker._api_key == "test-key-123"
        assert checker.is_enabled

    def test_init_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "env-key-456"}):
            checker = LambdaChecker()

            assert checker._api_key == "env-key-456"
            assert checker.is_enabled

    def test_init_from_key_file(self):
        """Test initialization from key file."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LAMBDA_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "lambda_api_key" in p

                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = "file-key\n"

                    checker = LambdaChecker()

                    assert checker._api_key == "file-key"

    def test_init_no_api_key(self):
        """Test initialization with no API key available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LAMBDA_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False

                checker = LambdaChecker()

                assert not checker.is_enabled

    def test_provider_name(self):
        """Test provider name."""
        checker = LambdaChecker(api_key="test")

        assert checker.provider_name == "lambda"


class TestLambdaCheckerStateMappings:
    """Tests for state mapping constants."""

    def test_active_state(self):
        """Test 'active' maps to RUNNING."""
        assert LAMBDA_STATE_MAP["active"] == ProviderInstanceState.RUNNING

    def test_booting_state(self):
        """Test 'booting' maps to STARTING."""
        assert LAMBDA_STATE_MAP["booting"] == ProviderInstanceState.STARTING

    def test_unhealthy_state(self):
        """Test 'unhealthy' maps to UNKNOWN."""
        assert LAMBDA_STATE_MAP["unhealthy"] == ProviderInstanceState.UNKNOWN

    def test_terminated_state(self):
        """Test 'terminated' maps to TERMINATED."""
        assert LAMBDA_STATE_MAP["terminated"] == ProviderInstanceState.TERMINATED


class TestLambdaCheckerApiAvailability:
    """Tests for API availability checking."""

    @pytest.mark.asyncio
    async def test_api_available_no_key(self):
        """Test API not available when no key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LAMBDA_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = LambdaChecker()

                result = await checker.check_api_availability()

                assert result is False

    @pytest.mark.asyncio
    async def test_api_available_success(self):
        """Test API available when request succeeds."""
        checker = LambdaChecker(api_key="test-key")

        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(checker, "_get_session", return_value=mock_session):
            result = await checker.check_api_availability()

            assert result is True

    @pytest.mark.asyncio
    async def test_api_available_failure(self):
        """Test API not available when request fails."""
        checker = LambdaChecker(api_key="test-key")

        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(checker, "_get_session", return_value=mock_session):
            result = await checker.check_api_availability()

            assert result is False


class TestLambdaCheckerGetInstanceStates:
    """Tests for get_instance_states method."""

    @pytest.mark.asyncio
    async def test_get_states_disabled(self):
        """Test returns empty when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LAMBDA_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = LambdaChecker()

                instances = await checker.get_instance_states()

                assert instances == []

    @pytest.mark.asyncio
    async def test_get_states_success(self):
        """Test successful instance state retrieval."""
        checker = LambdaChecker(api_key="test-key")

        mock_response_data = {
            "data": [
                {
                    "id": "inst-123",
                    "name": "lambda-gh200-1",
                    "status": "active",
                    "ip": "192.168.1.100",
                    "instance_type": {
                        "name": "gpu_1x_gh200",
                        "specs": {"gpus": 1},
                    },
                },
                {
                    "id": "inst-456",
                    "name": "lambda-h100-2",
                    "status": "booting",
                    "ip": None,
                    "instance_type": {
                        "name": "gpu_1x_h100_sxm5",
                        "specs": {"gpus": 1},
                    },
                },
            ]
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert len(instances) == 2
            assert instances[0].instance_id == "inst-123"
            assert instances[0].state == ProviderInstanceState.RUNNING
            assert instances[0].hostname == "lambda-gh200-1"
            assert instances[0].public_ip == "192.168.1.100"
            assert instances[0].gpu_type == "gpu_1x_gh200"

            assert instances[1].instance_id == "inst-456"
            assert instances[1].state == ProviderInstanceState.STARTING
            assert instances[1].public_ip is None

    @pytest.mark.asyncio
    async def test_get_states_api_error(self):
        """Test handles API errors gracefully."""
        checker = LambdaChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert instances == []
            assert checker._last_error is not None


class TestLambdaCheckerCorrelation:
    """Tests for correlate_with_config method."""

    def test_correlate_by_hostname(self):
        """Test correlation by hostname."""
        checker = LambdaChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="inst-123",
                state=ProviderInstanceState.RUNNING,
                provider="lambda",
                hostname="lambda-gh200-1",
                public_ip="192.168.1.100",
            ),
        ]

        config_hosts = {
            "lambda-gh200-1": {
                "ssh_host": "192.168.1.100",
            },
            "runpod-h100": {
                "ssh_host": "other.host",
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "lambda-gh200-1"

    def test_correlate_by_ip(self):
        """Test correlation by IP address."""
        checker = LambdaChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="inst-123",
                state=ProviderInstanceState.RUNNING,
                provider="lambda",
                hostname="different-name",
                public_ip="192.168.1.100",
            ),
        ]

        config_hosts = {
            "lambda-gh200-training": {
                "ssh_host": "192.168.1.100",
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "lambda-gh200-training"

    def test_correlate_by_tailscale_ip(self):
        """Test correlation by Tailscale IP."""
        checker = LambdaChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="inst-123",
                state=ProviderInstanceState.RUNNING,
                provider="lambda",
                hostname="some-name",
                public_ip="100.100.1.1",
            ),
        ]

        config_hosts = {
            "lambda-gh200-5": {
                "ssh_host": "other.host",
                "tailscale_ip": "100.100.1.1",
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "lambda-gh200-5"


class TestLambdaCheckerTerminatedInstances:
    """Tests for get_terminated_instances method."""

    @pytest.mark.asyncio
    async def test_find_terminated(self):
        """Test finding terminated instances."""
        checker = LambdaChecker(api_key="test-key")

        mock_response_data = {
            "data": [
                {"id": "inst-123", "status": "active", "ip": "192.168.1.100"},
            ]
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        config_hosts = {
            "lambda-gh200-1": {"ssh_host": "192.168.1.100", "status": "ready"},
            "lambda-gh200-2": {"ssh_host": "192.168.1.200", "status": "ready"},
            "lambda-gh200-3": {"ssh_host": "192.168.1.300", "status": "retired"},
            "runpod-h100": {"ssh_host": "other.host", "status": "ready"},
        }

        with patch.object(checker, "_get_session", return_value=mock_session):
            terminated = await checker.get_terminated_instances(config_hosts)

            assert "lambda-gh200-2" in terminated
            assert "lambda-gh200-1" not in terminated
            assert "lambda-gh200-3" not in terminated  # Already retired
            assert "runpod-h100" not in terminated  # Different provider


class TestLambdaCheckerClose:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test close when session exists."""
        checker = LambdaChecker(api_key="test-key")

        mock_session = AsyncMock()
        checker._http_session = mock_session

        await checker.close()

        mock_session.close.assert_called_once()
        assert checker._http_session is None

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test close when no session exists."""
        checker = LambdaChecker(api_key="test-key")

        # Should not raise
        await checker.close()
