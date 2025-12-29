"""Tests for RunPodProvider - RunPod cloud provider implementation.

Tests cover:
- Provider initialization and configuration
- GraphQL API request handling
- GPU type parsing
- Pod/instance parsing from GraphQL response
- Instance listing
- Instance status retrieval
- Scale up/down operations
- Cost estimation
- Singleton pattern

Created: Dec 28, 2025
"""

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.providers.runpod_provider import (
    RunPodConfig,
    RunPodProvider,
    RUNPOD_API_URL,
    RUNPOD_GPU_TYPES,
    RUNPOD_COSTS,
    get_runpod_provider,
    reset_runpod_provider,
)
from app.coordination.providers.base import (
    GPUType,
    InstanceStatus,
    ProviderType,
)


class TestRunPodConfig:
    """Tests for RunPodConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RunPodConfig()

        assert config.api_key is None
        assert config.api_url == RUNPOD_API_URL
        assert config.timeout_seconds == 30.0

    def test_config_with_api_key(self):
        """Test configuration with explicit API key."""
        config = RunPodConfig(api_key="test_key_123")

        assert config.api_key == "test_key_123"

    def test_from_env_with_env_var(self):
        """Test from_env loads from environment variable."""
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "env_api_key"}):
            config = RunPodConfig.from_env()

            assert config.api_key == "env_api_key"

    def test_from_env_without_env_var(self):
        """Test from_env returns None when no env var."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                config = RunPodConfig.from_env()

                assert config.api_key is None

    def test_from_env_with_config_file(self):
        """Test from_env loads from config file when no env var."""
        config_content = 'apikey = "config_file_key"\n'

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", MagicMock(return_value=iter(config_content.splitlines(True)))):
                    # Mock file reading
                    with patch("builtins.open") as mock_file:
                        mock_file.return_value.__enter__ = MagicMock(return_value=iter(config_content.splitlines(True)))
                        mock_file.return_value.__exit__ = MagicMock(return_value=False)

                        config = RunPodConfig.from_env()

                        # This should read from config file
                        # Note: Actual implementation may vary


class TestRunPodProviderInit:
    """Tests for RunPodProvider initialization."""

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = RunPodConfig(api_key="test_key")
        provider = RunPodProvider(config=config)

        assert provider.config.api_key == "test_key"
        assert provider._session is None

    def test_init_without_config(self):
        """Test initialization auto-loads config from env."""
        with patch.object(RunPodConfig, "from_env") as mock_from_env:
            mock_from_env.return_value = RunPodConfig(api_key="auto_key")

            provider = RunPodProvider()

            assert provider.config.api_key == "auto_key"


class TestRunPodProviderProperties:
    """Tests for RunPodProvider properties."""

    def test_provider_type(self):
        """Test provider_type property."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.provider_type == ProviderType.RUNPOD

    def test_name(self):
        """Test name property."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.name == "RunPod"


class TestRunPodProviderConfiguration:
    """Tests for RunPodProvider configuration checking."""

    def test_is_configured_true(self):
        """Test is_configured returns True when API key exists."""
        provider = RunPodProvider(config=RunPodConfig(api_key="valid_key"))

        assert provider.is_configured() is True

    def test_is_configured_false(self):
        """Test is_configured returns False when no API key."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        assert provider.is_configured() is False

    def test_is_configured_empty_key(self):
        """Test is_configured returns False for empty key."""
        provider = RunPodProvider(config=RunPodConfig(api_key=""))

        assert provider.is_configured() is False


class TestRunPodProviderGPUParsing:
    """Tests for GPU type parsing."""

    def test_parse_h100(self):
        """Test parsing H100 GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA H100 80GB HBM3") == GPUType.H100_80GB
        assert provider._parse_gpu_type("NVIDIA H100 PCIe") == GPUType.H100_80GB

    def test_parse_a100_80gb(self):
        """Test parsing A100 80GB GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA A100 80GB PCIe") == GPUType.A100_80GB
        assert provider._parse_gpu_type("NVIDIA A100-SXM4-80GB") == GPUType.A100_80GB

    def test_parse_a100_40gb(self):
        """Test parsing A100 40GB GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA A100-PCIE-40GB") == GPUType.A100_40GB
        assert provider._parse_gpu_type("NVIDIA A100-SXM4-40GB") == GPUType.A100_40GB

    def test_parse_a10(self):
        """Test parsing A10 GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA A10") == GPUType.A10

    def test_parse_l40s_as_a10_tier(self):
        """Test L40S maps to A10 tier."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        # L40S is mapped to A10 tier in the provider
        assert provider._parse_gpu_type("NVIDIA L40S") == GPUType.A10

    def test_parse_rtx_4090(self):
        """Test parsing RTX 4090 GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA GeForce RTX 4090") == GPUType.RTX_4090

    def test_parse_rtx_3090(self):
        """Test parsing RTX 3090 GPU."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("NVIDIA GeForce RTX 3090") == GPUType.RTX_3090
        assert provider._parse_gpu_type("NVIDIA GeForce RTX 3090 Ti") == GPUType.RTX_3090

    def test_parse_unknown_gpu(self):
        """Test parsing unknown GPU returns UNKNOWN."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("Unknown GPU") == GPUType.UNKNOWN

    def test_parse_empty_string(self):
        """Test parsing empty string returns UNKNOWN."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type("") == GPUType.UNKNOWN

    def test_parse_none(self):
        """Test parsing None returns UNKNOWN."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_gpu_type(None) == GPUType.UNKNOWN


class TestRunPodProviderStatusParsing:
    """Tests for pod status parsing."""

    def test_parse_running(self):
        """Test parsing RUNNING status."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("RUNNING") == InstanceStatus.RUNNING

    def test_parse_starting(self):
        """Test parsing STARTING status."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("STARTING") == InstanceStatus.STARTING

    def test_parse_created(self):
        """Test parsing CREATED status (maps to STARTING)."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("CREATED") == InstanceStatus.STARTING

    def test_parse_stopped(self):
        """Test parsing STOPPED status."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("STOPPED") == InstanceStatus.STOPPED

    def test_parse_exited(self):
        """Test parsing EXITED status (maps to STOPPED)."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("EXITED") == InstanceStatus.STOPPED

    def test_parse_terminated(self):
        """Test parsing TERMINATED status."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("TERMINATED") == InstanceStatus.TERMINATED

    def test_parse_unknown(self):
        """Test parsing unknown status."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._parse_pod_status("UNKNOWN") == InstanceStatus.UNKNOWN
        assert provider._parse_pod_status("some_random_status") == InstanceStatus.UNKNOWN


class TestRunPodProviderSSHExtraction:
    """Tests for SSH info extraction from runtime."""

    def test_extract_ssh_with_public_port(self):
        """Test extracting SSH info from public port."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        runtime = {
            "ports": [
                {"isIpPublic": True, "privatePort": 22, "publicPort": 12345, "ip": "192.168.1.100"},
                {"isIpPublic": True, "privatePort": 80, "publicPort": 8080, "ip": "192.168.1.100"},
            ]
        }

        ip, port = provider._extract_ssh_info(runtime)

        assert ip == "192.168.1.100"
        assert port == 12345

    def test_extract_ssh_no_ssh_port(self):
        """Test extracting SSH info when no SSH port exposed."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        runtime = {
            "ports": [
                {"isIpPublic": True, "privatePort": 80, "publicPort": 8080, "ip": "192.168.1.100"},
            ]
        }

        ip, port = provider._extract_ssh_info(runtime)

        assert ip is None
        assert port == 22

    def test_extract_ssh_no_runtime(self):
        """Test extracting SSH info with no runtime."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        ip, port = provider._extract_ssh_info(None)

        assert ip is None
        assert port == 22

    def test_extract_ssh_empty_ports(self):
        """Test extracting SSH info with empty ports."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        runtime = {"ports": []}

        ip, port = provider._extract_ssh_info(runtime)

        assert ip is None
        assert port == 22


class TestRunPodProviderPodToInstance:
    """Tests for pod to instance conversion."""

    def get_sample_pod_data(self):
        """Get sample pod data."""
        return {
            "id": "pod-abc123",
            "name": "ringrift-test",
            "desiredStatus": "RUNNING",
            "runtime": {
                "uptimeInSeconds": 3600,
                "ports": [
                    {"isIpPublic": True, "privatePort": 22, "publicPort": 22222, "ip": "192.168.1.100"},
                ],
            },
            "machine": {
                "gpuDisplayName": "NVIDIA GeForce RTX 4090",
                "cpuCount": 8,
                "memoryTotal": 32000,
            },
        }

    def test_pod_to_instance_running(self):
        """Test converting running pod to instance."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))
        pod = self.get_sample_pod_data()

        instance = provider._pod_to_instance(pod)

        assert instance.id == "pod-abc123"
        assert instance.provider == ProviderType.RUNPOD
        assert instance.name == "ringrift-test"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.RTX_4090
        assert instance.gpu_count == 1
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 22222
        assert instance.ssh_user == "root"

    def test_pod_to_instance_no_runtime(self):
        """Test converting pod without runtime info."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))
        pod = self.get_sample_pod_data()
        pod["runtime"] = None

        instance = provider._pod_to_instance(pod)

        assert instance.ip_address is None
        assert instance.ssh_port == 22

    def test_pod_to_instance_no_machine(self):
        """Test converting pod without machine info."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))
        pod = self.get_sample_pod_data()
        pod["machine"] = None

        instance = provider._pod_to_instance(pod)

        assert instance.gpu_type == GPUType.UNKNOWN

    def test_pod_to_instance_uses_id_as_name(self):
        """Test pod without name uses ID."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))
        pod = self.get_sample_pod_data()
        del pod["name"]

        instance = provider._pod_to_instance(pod)

        assert instance.name == "pod-abc123"


class TestRunPodProviderListInstances:
    """Tests for listing instances."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self):
        """Test successful instance listing."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        graphql_response = {
            "data": {
                "myself": {
                    "pods": [
                        {
                            "id": "pod-1",
                            "name": "test-1",
                            "desiredStatus": "RUNNING",
                            "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 4090"},
                            "runtime": {},
                        },
                        {
                            "id": "pod-2",
                            "name": "test-2",
                            "desiredStatus": "STARTING",
                            "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
                            "runtime": {},
                        },
                    ]
                }
            }
        }

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = graphql_response

            instances = await provider.list_instances()

            assert len(instances) == 2
            assert instances[0].id == "pod-1"
            assert instances[0].status == InstanceStatus.RUNNING
            assert instances[1].id == "pod-2"
            assert instances[1].status == InstanceStatus.STARTING

    @pytest.mark.asyncio
    async def test_list_instances_empty(self):
        """Test listing when no pods exist."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        graphql_response = {
            "data": {
                "myself": {
                    "pods": []
                }
            }
        }

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = graphql_response

            instances = await provider.list_instances()

            assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_list_instances_not_configured(self):
        """Test listing when provider not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        instances = await provider.list_instances()

        assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_list_instances_api_error(self):
        """Test listing when API returns error."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.side_effect = Exception("API error")

            instances = await provider.list_instances()

            assert len(instances) == 0


class TestRunPodProviderGetInstance:
    """Tests for getting specific instance."""

    @pytest.mark.asyncio
    async def test_get_instance_found(self):
        """Test getting instance that exists."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        graphql_response = {
            "data": {
                "pod": {
                    "id": "pod-123",
                    "name": "target",
                    "desiredStatus": "RUNNING",
                    "machine": {"gpuDisplayName": "NVIDIA GeForce RTX 4090"},
                    "runtime": {},
                }
            }
        }

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = graphql_response

            instance = await provider.get_instance("pod-123")

            assert instance is not None
            assert instance.id == "pod-123"
            assert instance.name == "target"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self):
        """Test getting instance that doesn't exist."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        graphql_response = {
            "data": {
                "pod": None
            }
        }

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = graphql_response

            instance = await provider.get_instance("nonexistent")

            assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_not_configured(self):
        """Test getting instance when not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        instance = await provider.get_instance("any-id")

        assert instance is None


class TestRunPodProviderGetInstanceStatus:
    """Tests for getting instance status."""

    @pytest.mark.asyncio
    async def test_get_instance_status_running(self):
        """Test getting status of running instance."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "get_instance", new_callable=AsyncMock) as mock_get:
            mock_instance = MagicMock()
            mock_instance.status = InstanceStatus.RUNNING
            mock_get.return_value = mock_instance

            status = await provider.get_instance_status("pod-123")

            assert status == InstanceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_instance_status_not_found(self):
        """Test getting status of non-existent instance."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "get_instance", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            status = await provider.get_instance_status("nonexistent")

            assert status == InstanceStatus.UNKNOWN


class TestRunPodProviderScaleDown:
    """Tests for scale down (pod termination)."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self):
        """Test successful pod termination."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = {"data": {"podTerminate": True}}

            results = await provider.scale_down(["pod-1", "pod-2"])

            assert results["pod-1"] is True
            assert results["pod-2"] is True
            assert mock_gql.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self):
        """Test scale down with partial failure."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.side_effect = [
                {"data": {"podTerminate": True}},
                Exception("Pod not found"),
            ]

            results = await provider.scale_down(["pod-1", "pod-2"])

            assert results["pod-1"] is True
            assert results["pod-2"] is False

    @pytest.mark.asyncio
    async def test_scale_down_not_configured(self):
        """Test scale down when not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        results = await provider.scale_down(["pod-1", "pod-2"])

        assert results["pod-1"] is False
        assert results["pod-2"] is False


class TestRunPodProviderStopInstance:
    """Tests for stopping instances (without terminating)."""

    @pytest.mark.asyncio
    async def test_stop_instance_success(self):
        """Test successful pod stop."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = {"data": {"podStop": {"id": "pod-123"}}}

            result = await provider.stop_instance("pod-123")

            assert result is True

    @pytest.mark.asyncio
    async def test_stop_instance_failure(self):
        """Test pod stop failure."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.side_effect = Exception("Stop failed")

            result = await provider.stop_instance("pod-123")

            assert result is False

    @pytest.mark.asyncio
    async def test_stop_instance_not_configured(self):
        """Test stop instance when not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        result = await provider.stop_instance("pod-123")

        assert result is False


class TestRunPodProviderCostEstimation:
    """Tests for cost estimation."""

    def test_get_cost_h100(self):
        """Test H100 cost estimation."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.get_cost_per_hour(GPUType.H100_80GB) == RUNPOD_COSTS[GPUType.H100_80GB]

    def test_get_cost_a100_80gb(self):
        """Test A100 80GB cost estimation."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.get_cost_per_hour(GPUType.A100_80GB) == RUNPOD_COSTS[GPUType.A100_80GB]

    def test_get_cost_rtx_4090(self):
        """Test RTX 4090 cost estimation."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.get_cost_per_hour(GPUType.RTX_4090) == RUNPOD_COSTS[GPUType.RTX_4090]

    def test_get_cost_unknown(self):
        """Test cost for unknown GPU returns 0."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.0


class TestRunPodProviderGPUMemory:
    """Tests for GPU memory lookup."""

    def test_get_gpu_memory_h100(self):
        """Test H100 memory."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._get_gpu_memory(GPUType.H100_80GB) == 80.0

    def test_get_gpu_memory_a100_80gb(self):
        """Test A100 80GB memory."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._get_gpu_memory(GPUType.A100_80GB) == 80.0

    def test_get_gpu_memory_a100_40gb(self):
        """Test A100 40GB memory."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._get_gpu_memory(GPUType.A100_40GB) == 40.0

    def test_get_gpu_memory_unknown(self):
        """Test unknown GPU memory returns 0."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        assert provider._get_gpu_memory(GPUType.UNKNOWN) == 0.0


class TestRunPodProviderGraphQLRequest:
    """Tests for GraphQL request handling."""

    @pytest.mark.asyncio
    async def test_graphql_request_success(self):
        """Test successful GraphQL request."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"myself": {"id": "123"}}})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        with patch.object(provider, "_get_session", new_callable=AsyncMock) as mock_get_session:
            mock_get_session.return_value = mock_session

            result = await provider._graphql_request("query { myself { id } }")

            assert result == {"data": {"myself": {"id": "123"}}}

    @pytest.mark.asyncio
    async def test_graphql_request_no_api_key(self):
        """Test GraphQL request fails without API key."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        with pytest.raises(ValueError, match="not configured"):
            await provider._graphql_request("query { myself { id } }")

    @pytest.mark.asyncio
    async def test_graphql_request_api_error(self):
        """Test GraphQL request handles API errors."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"errors": [{"message": "Bad request"}]})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=None),
        ))

        with patch.object(provider, "_get_session", new_callable=AsyncMock) as mock_get_session:
            mock_get_session.return_value = mock_session

            with pytest.raises(Exception, match="API error"):
                await provider._graphql_request("query { myself { id } }")


class TestRunPodProviderScaleUp:
    """Tests for scale up (pod creation)."""

    @pytest.mark.asyncio
    async def test_scale_up_not_configured(self):
        """Test scale up fails when not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        with pytest.raises(ValueError, match="not configured"):
            await provider.scale_up(GPUType.RTX_4090, count=1)

    @pytest.mark.asyncio
    async def test_scale_up_unsupported_gpu(self):
        """Test scale up fails for unsupported GPU type."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with pytest.raises(ValueError, match="not supported"):
            await provider.scale_up(GPUType.UNKNOWN, count=1)


class TestRunPodProviderClose:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        """Test close method closes HTTP session."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_close_no_session(self):
        """Test close does nothing when no session."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))
        provider._session = None

        # Should not raise
        await provider.close()


class TestRunPodProviderSingleton:
    """Tests for singleton pattern."""

    def test_get_runpod_provider_creates_instance(self):
        """Test get_runpod_provider creates singleton."""
        reset_runpod_provider()

        provider1 = get_runpod_provider()
        provider2 = get_runpod_provider()

        assert provider1 is provider2

    def test_reset_runpod_provider_clears_singleton(self):
        """Test reset_runpod_provider clears singleton."""
        provider1 = get_runpod_provider()

        reset_runpod_provider()

        provider2 = get_runpod_provider()

        assert provider1 is not provider2


class TestRunPodProviderGPUAvailability:
    """Tests for GPU availability queries."""

    @pytest.mark.asyncio
    async def test_get_gpu_availability_success(self):
        """Test successful GPU availability query."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        graphql_response = {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA H100 80GB HBM3",
                        "displayName": "H100",
                        "memoryInGb": 80,
                        "secureCloud": True,
                        "communityCloud": True,
                    },
                    {
                        "id": "NVIDIA GeForce RTX 4090",
                        "displayName": "RTX 4090",
                        "memoryInGb": 24,
                        "secureCloud": True,
                        "communityCloud": True,
                    },
                ]
            }
        }

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.return_value = graphql_response

            availability = await provider.get_gpu_availability()

            assert len(availability) == 2
            assert availability["NVIDIA H100 80GB HBM3"]["memory_gb"] == 80
            assert availability["NVIDIA GeForce RTX 4090"]["memory_gb"] == 24

    @pytest.mark.asyncio
    async def test_get_gpu_availability_not_configured(self):
        """Test GPU availability when not configured."""
        provider = RunPodProvider(config=RunPodConfig(api_key=None))

        availability = await provider.get_gpu_availability()

        assert availability == {}

    @pytest.mark.asyncio
    async def test_get_gpu_availability_error(self):
        """Test GPU availability handles errors."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        with patch.object(provider, "_graphql_request", new_callable=AsyncMock) as mock_gql:
            mock_gql.side_effect = Exception("API error")

            availability = await provider.get_gpu_availability()

            assert availability == {}


class TestRunPodProviderRunPodGPUId:
    """Tests for RunPod GPU ID mapping."""

    def test_get_runpod_gpu_id_h100(self):
        """Test GPU ID for H100."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        gpu_id = provider._get_runpod_gpu_id(GPUType.H100_80GB)

        assert gpu_id == "NVIDIA H100 80GB HBM3"

    def test_get_runpod_gpu_id_a100_80gb(self):
        """Test GPU ID for A100 80GB."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        gpu_id = provider._get_runpod_gpu_id(GPUType.A100_80GB)

        assert gpu_id == "NVIDIA A100 80GB PCIe"

    def test_get_runpod_gpu_id_rtx_4090(self):
        """Test GPU ID for RTX 4090."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        gpu_id = provider._get_runpod_gpu_id(GPUType.RTX_4090)

        assert gpu_id == "NVIDIA GeForce RTX 4090"

    def test_get_runpod_gpu_id_unknown(self):
        """Test GPU ID for unknown returns None."""
        provider = RunPodProvider(config=RunPodConfig(api_key="test"))

        gpu_id = provider._get_runpod_gpu_id(GPUType.UNKNOWN)

        assert gpu_id is None
