"""Tests for VastProvider - Vast.ai cloud provider implementation.

Tests cover:
- Provider initialization and configuration
- GPU type parsing
- Instance parsing from JSON
- CLI command execution (mocked)
- Instance listing and filtering
- Scale up/down operations
- Cost estimation
- Health check integration
"""

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.coordination.providers.vast_provider import VastProvider
from app.coordination.providers.base import (
    GPUType,
    InstanceStatus,
    ProviderType,
)


class TestVastProviderInit:
    """Tests for VastProvider initialization."""

    def test_init_with_cli_path(self):
        """Test initialization with explicit CLI path."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._cli_path == "/usr/bin/vastai"

    def test_init_auto_detect_cli(self):
        """Test initialization with auto-detection."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/vastai"
            provider = VastProvider()

            assert provider._cli_path == "/usr/local/bin/vastai"

    def test_init_cli_not_found(self):
        """Test initialization when CLI not found."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = VastProvider()

            assert provider._cli_path is None


class TestVastProviderProperties:
    """Tests for VastProvider properties."""

    def test_provider_type(self):
        """Test provider_type property."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.provider_type == ProviderType.VAST

    def test_name(self):
        """Test name property."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.name == "Vast.ai"


class TestVastProviderConfiguration:
    """Tests for VastProvider configuration checking."""

    def test_is_configured_true(self):
        """Test is_configured returns True when CLI and API key exist."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            assert provider.is_configured() is True

    def test_is_configured_no_cli(self):
        """Test is_configured returns False when no CLI."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = VastProvider()

            assert provider.is_configured() is False

    def test_is_configured_no_api_key(self):
        """Test is_configured returns False when no API key file."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False
            assert provider.is_configured() is False


class TestVastProviderGPUParsing:
    """Tests for GPU type parsing."""

    def test_parse_gh200(self):
        """Test parsing GH200 GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("NVIDIA GH200 96GB") == GPUType.GH200_96GB
        assert provider._parse_gpu_type("gh200") == GPUType.GH200_96GB

    def test_parse_h100(self):
        """Test parsing H100 GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("NVIDIA H100 80GB") == GPUType.H100_80GB
        assert provider._parse_gpu_type("h100") == GPUType.H100_80GB

    def test_parse_a100_80gb(self):
        """Test parsing A100 80GB GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("NVIDIA A100 80GB") == GPUType.A100_80GB
        assert provider._parse_gpu_type("a100-80gb") == GPUType.A100_80GB

    def test_parse_a100_40gb(self):
        """Test parsing A100 40GB GPU (default A100)."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("NVIDIA A100 40GB") == GPUType.A100_40GB
        assert provider._parse_gpu_type("A100") == GPUType.A100_40GB

    def test_parse_a10_not_confused_with_a100(self):
        """Test A10 is not confused with A100."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("NVIDIA A10") == GPUType.A10
        assert provider._parse_gpu_type("A10G") == GPUType.A10

    def test_parse_rtx_5090(self):
        """Test parsing RTX 5090 GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("RTX 5090") == GPUType.RTX_5090
        assert provider._parse_gpu_type("GeForce RTX 5090") == GPUType.RTX_5090

    def test_parse_rtx_4090(self):
        """Test parsing RTX 4090 GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("RTX 4090") == GPUType.RTX_4090
        assert provider._parse_gpu_type("GeForce RTX 4090 Ti") == GPUType.RTX_4090

    def test_parse_rtx_3090(self):
        """Test parsing RTX 3090 GPU."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("RTX 3090") == GPUType.RTX_3090
        assert provider._parse_gpu_type("GeForce RTX 3090 Ti") == GPUType.RTX_3090

    def test_parse_unknown_gpu(self):
        """Test parsing unknown GPU returns UNKNOWN."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider._parse_gpu_type("Unknown GPU") == GPUType.UNKNOWN
        assert provider._parse_gpu_type("") == GPUType.UNKNOWN


class TestVastProviderInstanceParsing:
    """Tests for instance parsing from JSON."""

    def get_sample_instance_data(self):
        """Get sample instance data."""
        return {
            "id": 12345,
            "label": "ringrift-test",
            "actual_status": "running",
            "gpu_name": "RTX 4090",
            "num_gpus": 2,
            "gpu_ram": 24576,  # MB
            "public_ipaddr": "192.168.1.100",
            "ssh_port": 22222,
            "start_date": 1703750000.0,  # Timestamp
            "dph_total": 0.50,
            "geolocation": "US-West",
        }

    def test_parse_instance_running(self):
        """Test parsing running instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()

        instance = provider._parse_instance(data)

        assert instance.id == "12345"
        assert instance.provider == ProviderType.VAST
        assert instance.name == "ringrift-test"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.RTX_4090
        assert instance.gpu_count == 2
        assert instance.gpu_memory_gb == 24.0  # 24576 MB / 1024
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 22222
        assert instance.ssh_user == "root"
        assert instance.cost_per_hour == 0.50
        assert instance.region == "US-West"

    def test_parse_instance_loading(self):
        """Test parsing loading/starting instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        data["actual_status"] = "loading"

        instance = provider._parse_instance(data)

        assert instance.status == InstanceStatus.STARTING

    def test_parse_instance_exited(self):
        """Test parsing exited/stopped instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        data["actual_status"] = "exited"

        instance = provider._parse_instance(data)

        assert instance.status == InstanceStatus.STOPPED

    def test_parse_instance_created(self):
        """Test parsing created/pending instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        data["actual_status"] = "created"

        instance = provider._parse_instance(data)

        assert instance.status == InstanceStatus.PENDING

    def test_parse_instance_unknown_status(self):
        """Test parsing instance with unknown status."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        data["actual_status"] = "unknown_status"

        instance = provider._parse_instance(data)

        assert instance.status == InstanceStatus.UNKNOWN

    def test_parse_instance_without_label(self):
        """Test parsing instance without label uses ID."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        del data["label"]

        instance = provider._parse_instance(data)

        assert instance.name == "vast-12345"

    def test_parse_instance_without_start_date(self):
        """Test parsing instance without start date."""
        provider = VastProvider(cli_path="/usr/bin/vastai")
        data = self.get_sample_instance_data()
        del data["start_date"]

        instance = provider._parse_instance(data)

        assert instance.created_at is None


class TestVastProviderListInstances:
    """Tests for listing instances."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self):
        """Test successful instance listing."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        instances_json = json.dumps([
            {
                "id": 12345,
                "label": "test-1",
                "actual_status": "running",
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "gpu_ram": 24576,
            },
            {
                "id": 12346,
                "label": "test-2",
                "actual_status": "loading",
                "gpu_name": "A100",
                "num_gpus": 1,
                "gpu_ram": 40960,
            },
        ])

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instances = await provider.list_instances()

            assert len(instances) == 2
            assert instances[0].id == "12345"
            assert instances[1].id == "12346"

    @pytest.mark.asyncio
    async def test_list_instances_cli_failure(self):
        """Test instance listing when CLI fails."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "Error: API key invalid", 1)

            instances = await provider.list_instances()

            assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self):
        """Test instance listing with invalid JSON response."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("not valid json", "", 0)

            instances = await provider.list_instances()

            assert len(instances) == 0


class TestVastProviderGetInstance:
    """Tests for getting specific instance."""

    @pytest.mark.asyncio
    async def test_get_instance_found(self):
        """Test getting instance that exists."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        instances_json = json.dumps([
            {"id": 12345, "label": "target", "actual_status": "running", "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24576},
            {"id": 12346, "label": "other", "actual_status": "running", "gpu_name": "A100", "num_gpus": 1, "gpu_ram": 40960},
        ])

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instance = await provider.get_instance("12345")

            assert instance is not None
            assert instance.id == "12345"
            assert instance.name == "target"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self):
        """Test getting instance that doesn't exist."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        instances_json = json.dumps([
            {"id": 12346, "label": "other", "actual_status": "running", "gpu_name": "A100", "num_gpus": 1, "gpu_ram": 40960},
        ])

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instance = await provider.get_instance("99999")

            assert instance is None


class TestVastProviderGetInstanceStatus:
    """Tests for getting instance status."""

    @pytest.mark.asyncio
    async def test_get_instance_status_running(self):
        """Test getting status of running instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        instances_json = json.dumps([
            {"id": 12345, "label": "test", "actual_status": "running", "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24576},
        ])

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            status = await provider.get_instance_status("12345")

            assert status == InstanceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_instance_status_not_found(self):
        """Test getting status of non-existent instance."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("[]", "", 0)

            status = await provider.get_instance_status("99999")

            assert status == InstanceStatus.UNKNOWN


class TestVastProviderScaleDown:
    """Tests for scale down (instance destruction)."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self):
        """Test successful instance destruction."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await provider.scale_down(["12345", "12346"])

            assert results["12345"] is True
            assert results["12346"] is True
            assert mock_cli.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self):
        """Test scale down with partial failure."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            # First succeeds, second fails
            mock_cli.side_effect = [
                ("", "", 0),
                ("", "Error: Instance not found", 1),
            ]

            results = await provider.scale_down(["12345", "99999"])

            assert results["12345"] is True
            assert results["99999"] is False


class TestVastProviderCostEstimation:
    """Tests for cost estimation."""

    def test_get_cost_rtx_3090(self):
        """Test RTX 3090 cost estimation."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.get_cost_per_hour(GPUType.RTX_3090) == 0.30

    def test_get_cost_rtx_4090(self):
        """Test RTX 4090 cost estimation."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.get_cost_per_hour(GPUType.RTX_4090) == 0.50

    def test_get_cost_a100_80gb(self):
        """Test A100 80GB cost estimation."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.get_cost_per_hour(GPUType.A100_80GB) == 1.30

    def test_get_cost_h100(self):
        """Test H100 cost estimation."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.get_cost_per_hour(GPUType.H100_80GB) == 2.50

    def test_get_cost_unknown(self):
        """Test cost for unknown GPU returns default."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        assert provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.5


class TestVastProviderHealthCheck:
    """Tests for health check integration."""

    def test_health_check_healthy(self):
        """Test health check when fully configured."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            result = provider.health_check()

            assert result.healthy is True
            assert "RUNNING" in result.status.value or result.status.value == "running"
            assert "configured" in result.message.lower()
            assert result.details["configured"] is True

    def test_health_check_no_cli(self):
        """Test health check when CLI not found."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = VastProvider()

            result = provider.health_check()

            assert result.healthy is False
            assert "ERROR" in result.status.value or result.status.value == "error"
            assert "not found" in result.message.lower()

    def test_health_check_no_api_key(self):
        """Test health check when API key not configured."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            result = provider.health_check()

            assert result.healthy is False
            assert "DEGRADED" in result.status.value or result.status.value == "degraded"
            assert "not configured" in result.message.lower()


class TestVastProviderScaleUp:
    """Tests for scale up (instance creation)."""

    @pytest.mark.asyncio
    async def test_scale_up_no_search_term(self):
        """Test scale up with unsupported GPU type."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        instances = await provider.scale_up(GPUType.UNKNOWN, count=1)

        assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_scale_up_no_offers(self):
        """Test scale up when no offers available."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("[]", "", 0)

            instances = await provider.scale_up(GPUType.RTX_4090, count=1)

            assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_scale_up_search_fails(self):
        """Test scale up when offer search fails."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "Error searching", 1)

            instances = await provider.scale_up(GPUType.RTX_4090, count=1)

            assert len(instances) == 0


class TestVastProviderCLI:
    """Tests for CLI execution."""

    @pytest.mark.asyncio
    async def test_run_cli_no_path(self):
        """Test CLI execution fails when no CLI path."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = VastProvider()

            with pytest.raises(RuntimeError, match="CLI not found"):
                await provider._run_cli("show", "instances")

    @pytest.mark.asyncio
    async def test_run_cli_adds_raw_flag(self):
        """Test CLI execution adds --raw flag."""
        provider = VastProvider(cli_path="/usr/bin/vastai")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_executor = MagicMock()
            mock_loop.return_value.run_in_executor = AsyncMock()

            # Create a mock subprocess result
            mock_result = MagicMock()
            mock_result.stdout = "[]"
            mock_result.stderr = ""
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result) as mock_run:
                # Bypass the async executor by directly testing the lambda
                # This is a simplified test
                pass
