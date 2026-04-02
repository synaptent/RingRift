"""Tests for VastProvider cloud provider implementation.

Tests cover:
- Provider initialization and properties
- Configuration detection (CLI and API key)
- CLI execution and output parsing
- Instance parsing from JSON
- Listing, scaling up/down operations
- Cost estimation
- Health check reporting
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.coordination.providers.base import (
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)
from app.coordination.providers.vast_provider import VastProvider


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def vast_provider():
    """Create VastProvider with mocked CLI path."""
    with patch("shutil.which", return_value="/usr/local/bin/vastai"):
        return VastProvider()


@pytest.fixture
def vast_provider_no_cli():
    """Create VastProvider with no CLI available."""
    with patch("shutil.which", return_value=None):
        return VastProvider()


@pytest.fixture
def sample_instance_data():
    """Sample Vast.ai instance JSON data."""
    return {
        "id": 12345,
        "label": "ringrift-training-1",
        "actual_status": "running",
        "gpu_name": "RTX 4090",
        "num_gpus": 2,
        "gpu_ram": 24576,  # 24 GB in MB
        "public_ipaddr": "192.168.1.100",
        "ssh_port": 30022,
        "start_date": 1704067200.0,  # 2024-01-01 00:00:00 UTC
        "dph_total": 0.48,
        "geolocation": "US-East",
    }


@pytest.fixture
def sample_offer_data():
    """Sample Vast.ai offer JSON data."""
    return {
        "id": 99999,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "reliability": 0.98,
        "dph_total": 0.45,
    }


# ===========================================================================
# Initialization Tests
# ===========================================================================


class TestVastProviderInit:
    """Test VastProvider initialization."""

    def test_init_with_auto_detect_cli(self):
        """Test CLI is auto-detected via shutil.which."""
        with patch("shutil.which", return_value="/usr/local/bin/vastai") as mock_which:
            provider = VastProvider()

            mock_which.assert_called_once_with("vastai")
            assert provider._cli_path == "/usr/local/bin/vastai"

    def test_init_with_explicit_cli_path(self):
        """Test explicit CLI path overrides auto-detection."""
        provider = VastProvider(cli_path="/custom/path/vastai")

        assert provider._cli_path == "/custom/path/vastai"

    def test_init_with_no_cli_available(self, vast_provider_no_cli):
        """Test behavior when CLI is not found."""
        assert vast_provider_no_cli._cli_path is None


class TestVastProviderProperties:
    """Test VastProvider properties."""

    def test_provider_type(self, vast_provider):
        """Test provider_type returns VAST."""
        assert vast_provider.provider_type == ProviderType.VAST

    def test_name(self, vast_provider):
        """Test name returns 'Vast.ai'."""
        assert vast_provider.name == "Vast.ai"


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestVastProviderConfiguration:
    """Test VastProvider configuration detection."""

    def test_is_configured_true(self, vast_provider):
        """Test is_configured when CLI and API key exist."""
        with patch("os.path.exists", return_value=True):
            assert vast_provider.is_configured() is True

    def test_is_configured_no_cli(self, vast_provider_no_cli):
        """Test is_configured when CLI is missing."""
        assert vast_provider_no_cli.is_configured() is False

    def test_is_configured_no_api_key(self, vast_provider):
        """Test is_configured when API key file is missing."""
        with patch("os.path.exists", return_value=False):
            assert vast_provider.is_configured() is False

    def test_is_configured_checks_api_key_path(self, vast_provider):
        """Test that is_configured checks correct API key path."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            vast_provider.is_configured()

            # Should check ~/.vast_api_key
            mock_exists.assert_called()
            call_args = mock_exists.call_args[0][0]
            assert call_args.endswith(".vast_api_key")


# ===========================================================================
# Instance Parsing Tests
# ===========================================================================


class TestVastProviderParseInstance:
    """Test VastProvider._parse_instance()."""

    def test_parse_running_instance(self, vast_provider, sample_instance_data):
        """Test parsing a running instance."""
        instance = vast_provider._parse_instance(sample_instance_data)

        assert instance.id == "12345"
        assert instance.provider == ProviderType.VAST
        assert instance.name == "ringrift-training-1"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.RTX_4090
        assert instance.gpu_count == 2
        assert instance.gpu_memory_gb == 24.0  # 24576 MB / 1024
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 30022
        assert instance.ssh_user == "root"
        assert instance.cost_per_hour == 0.48
        assert instance.region == "US-East"

    def test_parse_instance_status_mapping(self, vast_provider):
        """Test status mapping for various states."""
        status_cases = [
            ("running", InstanceStatus.RUNNING),
            ("loading", InstanceStatus.STARTING),
            ("exited", InstanceStatus.STOPPED),
            ("created", InstanceStatus.PENDING),
            ("unknown", InstanceStatus.UNKNOWN),
            ("other", InstanceStatus.UNKNOWN),
        ]

        for vast_status, expected in status_cases:
            data = {"id": 1, "actual_status": vast_status}
            instance = vast_provider._parse_instance(data)
            assert instance.status == expected, f"Status {vast_status} should map to {expected}"

    def test_parse_instance_with_missing_fields(self, vast_provider):
        """Test parsing with minimal data."""
        data = {"id": 123}
        instance = vast_provider._parse_instance(data)

        assert instance.id == "123"
        assert instance.name == "vast-123"  # Default name
        assert instance.gpu_type == GPUType.UNKNOWN
        assert instance.gpu_count == 1

    def test_parse_instance_without_label(self, vast_provider):
        """Test parsing creates default name from ID."""
        data = {"id": 999, "actual_status": "running"}
        instance = vast_provider._parse_instance(data)

        assert instance.name == "vast-999"

    def test_parse_instance_gpu_types(self, vast_provider):
        """Test various GPU name parsing."""
        gpu_cases = [
            ("RTX 3090", GPUType.RTX_3090),
            ("NVIDIA RTX 4090", GPUType.RTX_4090),
            ("GeForce RTX 5090", GPUType.RTX_5090),
            ("A100-SXM4-80GB", GPUType.A100_80GB),
            ("A100-SXM4-40GB", GPUType.A100_40GB),
            ("H100", GPUType.H100_80GB),
            ("Unknown GPU", GPUType.UNKNOWN),
        ]

        for gpu_name, expected in gpu_cases:
            data = {"id": 1, "gpu_name": gpu_name}
            instance = vast_provider._parse_instance(data)
            assert instance.gpu_type == expected, f"{gpu_name} should parse to {expected}"


# ===========================================================================
# CLI Execution Tests
# ===========================================================================


class TestVastProviderCLI:
    """Test VastProvider CLI execution."""

    @pytest.mark.asyncio
    async def test_run_cli_success(self, vast_provider):
        """Test successful CLI execution."""
        mock_result = MagicMock()
        mock_result.stdout = '{"status": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.vast_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            stdout, stderr, rc = await vast_provider._run_cli("show", "instances")

            assert stdout == '{"status": "ok"}'
            assert stderr == ""
            assert rc == 0

    @pytest.mark.asyncio
    async def test_run_cli_adds_raw_flag(self, vast_provider):
        """Test that --raw flag is added to commands."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.vast_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            await vast_provider._run_cli("show", "instances")

            call_args = mock_run.await_args.args[0]
            assert "--raw" in call_args

    @pytest.mark.asyncio
    async def test_run_cli_error(self, vast_provider):
        """Test CLI execution with error."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error: API key invalid"
        mock_result.returncode = 1

        with patch(
            "app.coordination.providers.vast_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            stdout, stderr, rc = await vast_provider._run_cli("show", "instances")

            assert rc == 1
            assert "API key invalid" in stderr

    @pytest.mark.asyncio
    async def test_run_cli_no_cli_path(self, vast_provider_no_cli):
        """Test CLI execution when CLI not available."""
        with pytest.raises(RuntimeError, match="vastai CLI not found"):
            await vast_provider_no_cli._run_cli("show", "instances")


# ===========================================================================
# List Instances Tests
# ===========================================================================


class TestVastProviderListInstances:
    """Test VastProvider.list_instances()."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self, vast_provider, sample_instance_data):
        """Test successful instance listing."""
        instances_json = json.dumps([sample_instance_data])

        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instances = await vast_provider.list_instances()

            assert len(instances) == 1
            assert instances[0].id == "12345"
            mock_cli.assert_called_once_with("show", "instances")

    @pytest.mark.asyncio
    async def test_list_instances_empty(self, vast_provider):
        """Test listing when no instances exist."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("[]", "", 0)

            instances = await vast_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_cli_error(self, vast_provider):
        """Test listing when CLI fails."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "API error", 1)

            instances = await vast_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self, vast_provider):
        """Test listing with invalid JSON response."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("not valid json", "", 0)

            instances = await vast_provider.list_instances()

            assert instances == []


# ===========================================================================
# Scale Up Tests
# ===========================================================================


class TestVastProviderScaleUp:
    """Test VastProvider.scale_up()."""

    @pytest.mark.asyncio
    async def test_scale_up_success(self, vast_provider, sample_offer_data, sample_instance_data):
        """Test successful scale up."""
        offers_json = json.dumps([sample_offer_data])
        create_response = json.dumps({"new_contract": 12345})
        instance_json = json.dumps([sample_instance_data])

        call_count = 0
        async def mock_run_cli(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Search offers
                return (offers_json, "", 0)
            elif call_count == 2:
                # Create instance
                return (create_response, "", 0)
            else:
                # Get instance
                return (instance_json, "", 0)

        with patch.object(vast_provider, "_run_cli", side_effect=mock_run_cli):
            instances = await vast_provider.scale_up(GPUType.RTX_4090, count=1)

            assert len(instances) == 1
            assert instances[0].id == "12345"

    @pytest.mark.asyncio
    async def test_scale_up_no_offers(self, vast_provider):
        """Test scale up when no offers available."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("[]", "", 0)

            instances = await vast_provider.scale_up(GPUType.RTX_4090, count=1)

            assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_unsupported_gpu(self, vast_provider):
        """Test scale up with unsupported GPU type."""
        instances = await vast_provider.scale_up(GPUType.CPU_ONLY, count=1)

        assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_search_fails(self, vast_provider):
        """Test scale up when offer search fails."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "Search failed", 1)

            instances = await vast_provider.scale_up(GPUType.RTX_4090, count=1)

            assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_gpu_mapping(self, vast_provider):
        """Test GPU type to search term mapping."""
        gpu_cases = [
            (GPUType.RTX_3090, "RTX_3090"),
            (GPUType.RTX_4090, "RTX_4090"),
            (GPUType.A100_40GB, "A100"),
            (GPUType.H100_80GB, "H100"),
        ]

        for gpu_type, expected_search in gpu_cases:
            with patch.object(vast_provider, "_run_cli") as mock_cli:
                mock_cli.return_value = ("[]", "", 0)
                await vast_provider.scale_up(gpu_type, count=1)

                call_args = mock_cli.call_args[0]
                # Find the gpu_name argument
                gpu_arg = [arg for arg in call_args if arg.startswith("gpu_name=")]
                if gpu_arg:
                    assert expected_search in gpu_arg[0]


# ===========================================================================
# Scale Down Tests
# ===========================================================================


class TestVastProviderScaleDown:
    """Test VastProvider.scale_down()."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self, vast_provider):
        """Test successful instance termination."""
        with patch.object(vast_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await vast_provider.scale_down(["12345", "67890"])

            assert results == {"12345": True, "67890": True}
            assert mock_cli.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self, vast_provider):
        """Test scale down with partial failure."""
        call_count = 0
        async def mock_run_cli(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", "", 0)  # Success
            else:
                return ("", "Instance not found", 1)  # Failure

        with patch.object(vast_provider, "_run_cli", side_effect=mock_run_cli):
            results = await vast_provider.scale_down(["12345", "99999"])

            assert results["12345"] is True
            assert results["99999"] is False

    @pytest.mark.asyncio
    async def test_scale_down_empty_list(self, vast_provider):
        """Test scale down with empty list."""
        results = await vast_provider.scale_down([])

        assert results == {}


# ===========================================================================
# Cost Tests
# ===========================================================================


class TestVastProviderCost:
    """Test VastProvider cost estimation."""

    def test_get_cost_per_hour_known_gpu(self, vast_provider):
        """Test cost lookup for known GPU types."""
        assert vast_provider.get_cost_per_hour(GPUType.RTX_3090) == 0.30
        assert vast_provider.get_cost_per_hour(GPUType.RTX_4090) == 0.50
        assert vast_provider.get_cost_per_hour(GPUType.A100_80GB) == 1.30
        assert vast_provider.get_cost_per_hour(GPUType.H100_80GB) == 2.50

    def test_get_cost_per_hour_unknown_gpu(self, vast_provider):
        """Test cost lookup for unknown GPU returns default."""
        assert vast_provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.5
        assert vast_provider.get_cost_per_hour(GPUType.CPU_ONLY) == 0.5


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestVastProviderHealthCheck:
    """Test VastProvider.health_check()."""

    def test_health_check_healthy(self, vast_provider):
        """Test health check when CLI available and configured."""
        with patch("os.path.exists", return_value=True):
            result = vast_provider.health_check()

            assert result.healthy is True
            assert "CLI available" in result.message

    def test_health_check_no_cli(self, vast_provider_no_cli):
        """Test health check when CLI missing."""
        result = vast_provider_no_cli.health_check()

        assert result.healthy is False
        assert "CLI not found" in result.message

    def test_health_check_no_api_key(self, vast_provider):
        """Test health check when API key missing."""
        with patch("os.path.exists", return_value=False):
            result = vast_provider.health_check()

            assert result.healthy is False
            assert "API key not configured" in result.message
