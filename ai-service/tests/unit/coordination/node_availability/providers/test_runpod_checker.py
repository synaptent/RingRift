"""Tests for RunPodChecker - RunPod node availability state checker.

Tests cover:
- Initialization and API key detection
- GraphQL API requests (mocked)
- Pod state parsing
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

from app.coordination.node_availability.providers.runpod_checker import (
    RunPodChecker,
    RUNPOD_STATE_MAP,
    RUNPOD_API_URL,
    PODS_QUERY,
)
from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
)


class TestRunPodCheckerInit:
    """Tests for RunPodChecker initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        checker = RunPodChecker(api_key="test-key-123")

        assert checker._api_key == "test-key-123"
        assert checker.is_enabled

    def test_init_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "env-key-456"}):
            checker = RunPodChecker()

            assert checker._api_key == "env-key-456"
            assert checker.is_enabled

    def test_init_from_config_file(self):
        """Test initialization from config file."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "runpod/config.toml" in p

                config_content = 'apikey = "file-key-789"\n'
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=iter([config_content]))
                mock_file.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", return_value=mock_file):
                    checker = RunPodChecker()

                    assert checker._api_key == "file-key-789"

    def test_init_from_config_file_double_quotes(self):
        """Test config file parsing with double quotes."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "runpod/config.toml" in p

                config_content = 'apikey = "double-quoted-key"\n'
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=iter([config_content]))
                mock_file.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", return_value=mock_file):
                    checker = RunPodChecker()

                    assert checker._api_key == "double-quoted-key"

    def test_init_from_config_file_single_quotes(self):
        """Test config file parsing with single quotes."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "runpod/config.toml" in p

                config_content = "apikey = 'single-quoted-key'\n"
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=iter([config_content]))
                mock_file.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", return_value=mock_file):
                    checker = RunPodChecker()

                    assert checker._api_key == "single-quoted-key"

    def test_init_no_api_key(self):
        """Test initialization with no API key available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)

            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False

                checker = RunPodChecker()

                assert not checker.is_enabled

    def test_provider_name(self):
        """Test provider name."""
        checker = RunPodChecker(api_key="test")

        assert checker.provider_name == "runpod"


class TestRunPodCheckerStateMappings:
    """Tests for state mapping constants."""

    def test_running_state(self):
        """Test 'RUNNING' maps to RUNNING."""
        assert RUNPOD_STATE_MAP["RUNNING"] == ProviderInstanceState.RUNNING

    def test_starting_state(self):
        """Test 'STARTING' maps to STARTING."""
        assert RUNPOD_STATE_MAP["STARTING"] == ProviderInstanceState.STARTING

    def test_stopped_state(self):
        """Test 'STOPPED' maps to STOPPED."""
        assert RUNPOD_STATE_MAP["STOPPED"] == ProviderInstanceState.STOPPED

    def test_terminated_state(self):
        """Test 'TERMINATED' maps to TERMINATED."""
        assert RUNPOD_STATE_MAP["TERMINATED"] == ProviderInstanceState.TERMINATED

    def test_exited_state(self):
        """Test 'EXITED' maps to STOPPED."""
        assert RUNPOD_STATE_MAP["EXITED"] == ProviderInstanceState.STOPPED

    def test_created_state(self):
        """Test 'CREATED' maps to STARTING."""
        assert RUNPOD_STATE_MAP["CREATED"] == ProviderInstanceState.STARTING

    def test_api_url(self):
        """Test API URL constant."""
        assert RUNPOD_API_URL == "https://api.runpod.io/graphql"

    def test_pods_query_contains_essential_fields(self):
        """Test pods query contains essential fields."""
        assert "id" in PODS_QUERY
        assert "name" in PODS_QUERY
        assert "desiredStatus" in PODS_QUERY
        assert "ports" in PODS_QUERY
        assert "gpus" in PODS_QUERY


class TestRunPodCheckerApiAvailability:
    """Tests for API availability checking."""

    @pytest.mark.asyncio
    async def test_api_available_no_key(self):
        """Test API not available when no key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = RunPodChecker()

                result = await checker.check_api_availability()

                assert result is False

    @pytest.mark.asyncio
    async def test_api_available_success(self):
        """Test API available when request succeeds."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"myself": {"id": "123"}}})
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            result = await checker.check_api_availability()

            assert result is True

    @pytest.mark.asyncio
    async def test_api_available_with_errors(self):
        """Test API not available when response has errors."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"errors": [{"message": "Auth failed"}]})
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            result = await checker.check_api_availability()

            assert result is False

    @pytest.mark.asyncio
    async def test_api_available_non_200(self):
        """Test API not available when status is not 200."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            result = await checker.check_api_availability()

            assert result is False


class TestRunPodCheckerGetInstanceStates:
    """Tests for get_instance_states method."""

    @pytest.mark.asyncio
    async def test_get_states_disabled(self):
        """Test returns empty when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_API_KEY", None)
            with patch("os.path.exists", return_value=False):
                checker = RunPodChecker()

                instances = await checker.get_instance_states()

                assert instances == []

    @pytest.mark.asyncio
    async def test_get_states_success(self):
        """Test successful pod state retrieval."""
        checker = RunPodChecker(api_key="test-key")

        mock_response_data = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-123",
                            "name": "runpod-h100",
                            "desiredStatus": "RUNNING",
                            "runtime": {
                                "uptimeInSeconds": 3600,
                                "ports": [
                                    {
                                        "ip": "192.168.1.100",
                                        "isIpPublic": True,
                                        "privatePort": 22,
                                        "publicPort": 30178,
                                    }
                                ],
                                "gpus": [
                                    {"id": "gpu0", "gpuUtilPercent": 50, "memoryUtilPercent": 40}
                                ],
                            },
                            "machine": {
                                "gpuDisplayName": "NVIDIA H100 80GB HBM3",
                                "cpuCount": 32,
                                "memoryTotal": 256000,
                            },
                        },
                        {
                            "id": "pod-456",
                            "name": "runpod-a100-1",
                            "desiredStatus": "STARTING",
                            "runtime": None,
                            "machine": {
                                "gpuDisplayName": "NVIDIA A100 80GB PCIe",
                                "cpuCount": 16,
                                "memoryTotal": 128000,
                            },
                        },
                    ]
                }
            }
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert len(instances) == 2

            assert instances[0].instance_id == "pod-123"
            assert instances[0].state == ProviderInstanceState.RUNNING
            assert instances[0].hostname == "runpod-h100"
            assert instances[0].public_ip == "192.168.1.100"
            assert instances[0].ssh_port == 30178
            assert instances[0].gpu_type == "NVIDIA H100 80GB HBM3"

            assert instances[1].instance_id == "pod-456"
            assert instances[1].state == ProviderInstanceState.STARTING
            assert instances[1].hostname == "runpod-a100-1"
            assert instances[1].public_ip is None

    @pytest.mark.asyncio
    async def test_get_states_api_error(self):
        """Test handles API errors gracefully."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert instances == []
            assert checker._last_error is not None

    @pytest.mark.asyncio
    async def test_get_states_graphql_error(self):
        """Test handles GraphQL errors gracefully."""
        checker = RunPodChecker(api_key="test-key")

        mock_response_data = {
            "errors": [{"message": "Unauthorized"}]
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert instances == []
            assert "Unauthorized" in checker._last_error

    @pytest.mark.asyncio
    async def test_get_states_unknown_status(self):
        """Test handles unknown pod status."""
        checker = RunPodChecker(api_key="test-key")

        mock_response_data = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-123",
                            "name": "runpod-test",
                            "desiredStatus": "UNKNOWN_STATUS",
                            "runtime": None,
                            "machine": {},
                        }
                    ]
                }
            }
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(checker, "_get_session", return_value=mock_session):
            instances = await checker.get_instance_states()

            assert len(instances) == 1
            assert instances[0].state == ProviderInstanceState.UNKNOWN


class TestRunPodCheckerCorrelation:
    """Tests for correlate_with_config method."""

    def test_correlate_by_name(self):
        """Test correlation by exact pod name."""
        checker = RunPodChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="pod-123",
                state=ProviderInstanceState.RUNNING,
                provider="runpod",
                hostname="runpod-h100",
                public_ip="192.168.1.100",
                ssh_port=30178,
            ),
        ]

        config_hosts = {
            "runpod-h100": {
                "ssh_host": "192.168.1.100",
                "ssh_port": 30178,
            },
            "vast-12345": {
                "ssh_host": "other.host",
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "runpod-h100"

    def test_correlate_by_pattern(self):
        """Test correlation by pattern matching."""
        checker = RunPodChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="pod-123",
                state=ProviderInstanceState.RUNNING,
                provider="runpod",
                hostname="my-h100-training-pod",
                public_ip="192.168.1.100",
                ssh_port=30178,
            ),
        ]

        config_hosts = {
            "runpod-h100": {
                "ssh_host": "192.168.1.100",
                "ssh_port": 30178,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "runpod-h100"

    def test_correlate_by_ssh(self):
        """Test correlation by SSH host/port."""
        checker = RunPodChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="pod-123",
                state=ProviderInstanceState.RUNNING,
                provider="runpod",
                hostname="different-name",
                ssh_host="192.168.1.100",
                ssh_port=30178,
            ),
        ]

        config_hosts = {
            "runpod-custom-node": {
                "ssh_host": "192.168.1.100",
                "ssh_port": 30178,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name == "runpod-custom-node"

    def test_correlate_no_match(self):
        """Test correlation when no match found."""
        checker = RunPodChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="pod-123",
                state=ProviderInstanceState.RUNNING,
                provider="runpod",
                hostname="unknown-pod",
                ssh_host="unknown.host",
                ssh_port=99999,
            ),
        ]

        config_hosts = {
            "runpod-h100": {
                "ssh_host": "192.168.1.100",
                "ssh_port": 30178,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        assert result[0].node_name is None

    def test_correlate_ignores_non_runpod(self):
        """Test correlation ignores non-runpod nodes."""
        checker = RunPodChecker(api_key="test-key")

        instances = [
            InstanceInfo(
                instance_id="pod-123",
                state=ProviderInstanceState.RUNNING,
                provider="runpod",
                hostname="vast-12345",  # Pod named like vast
                ssh_host="192.168.1.100",
                ssh_port=22,
            ),
        ]

        config_hosts = {
            "vast-12345": {  # Non-runpod node
                "ssh_host": "192.168.1.100",
                "ssh_port": 22,
            },
            "runpod-h100": {
                "ssh_host": "10.0.0.1",
                "ssh_port": 30178,
            },
        }

        result = checker.correlate_with_config(instances, config_hosts)

        # Should not match vast node even if hostname matches
        assert result[0].node_name is None


class TestRunPodCheckerTerminatedInstances:
    """Tests for get_terminated_instances method."""

    @pytest.mark.asyncio
    async def test_find_terminated(self):
        """Test finding terminated instances."""
        checker = RunPodChecker(api_key="test-key")

        mock_response_data = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-123",
                            "name": "runpod-h100",
                            "desiredStatus": "RUNNING",
                            "runtime": {
                                "ports": [
                                    {"ip": "192.168.1.100", "isIpPublic": True, "privatePort": 22, "publicPort": 30178}
                                ]
                            },
                            "machine": {},
                        }
                    ]
                }
            }
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        config_hosts = {
            "runpod-h100": {"ssh_host": "192.168.1.100", "status": "ready"},
            "runpod-a100-1": {"ssh_host": "192.168.1.200", "status": "ready"},
            "runpod-a100-2": {"ssh_host": "192.168.1.201", "status": "retired"},
            "vast-12345": {"ssh_host": "other.host", "status": "ready"},
        }

        with patch.object(checker, "_get_session", return_value=mock_session):
            terminated = await checker.get_terminated_instances(config_hosts)

            assert "runpod-a100-1" in terminated
            assert "runpod-h100" not in terminated
            assert "runpod-a100-2" not in terminated  # Already retired
            assert "vast-12345" not in terminated  # Different provider

    @pytest.mark.asyncio
    async def test_find_terminated_by_ip(self):
        """Test terminated detection falls back to IP check."""
        checker = RunPodChecker(api_key="test-key")

        mock_response_data = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-123",
                            "name": "different-name",  # Different name
                            "desiredStatus": "RUNNING",
                            "runtime": {
                                "ports": [
                                    {"ip": "192.168.1.100", "isIpPublic": True, "privatePort": 22, "publicPort": 30178}
                                ]
                            },
                            "machine": {},
                        }
                    ]
                }
            }
        }

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        config_hosts = {
            "runpod-h100": {"ssh_host": "192.168.1.100", "status": "ready"},  # IP matches
            "runpod-a100-1": {"ssh_host": "192.168.1.200", "status": "ready"},  # Not found
        }

        with patch.object(checker, "_get_session", return_value=mock_session):
            terminated = await checker.get_terminated_instances(config_hosts)

            assert "runpod-a100-1" in terminated
            assert "runpod-h100" not in terminated  # IP match keeps it active

    @pytest.mark.asyncio
    async def test_find_terminated_api_failure(self):
        """Test handles API failure gracefully."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Error")
        mock_session.post.return_value.__aenter__.return_value = mock_response

        config_hosts = {
            "runpod-h100": {"status": "ready"},
        }

        with patch.object(checker, "_get_session", return_value=mock_session):
            terminated = await checker.get_terminated_instances(config_hosts)

            # Should return empty on API failure (conservative)
            assert terminated == []


class TestRunPodCheckerClose:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test close when session exists."""
        checker = RunPodChecker(api_key="test-key")

        mock_session = AsyncMock()
        checker._http_session = mock_session

        await checker.close()

        mock_session.close.assert_called_once()
        assert checker._http_session is None

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test close when no session exists."""
        checker = RunPodChecker(api_key="test-key")

        # Should not raise
        await checker.close()
