"""Tests for HetznerProvider cloud provider implementation.

Tests cover:
- Provider initialization and properties
- Configuration detection (CLI and API token)
- CLI execution with token handling
- Instance parsing from JSON
- Listing, scaling up/down operations
- Cost estimation
- CPU-only behavior
- Health check reporting
"""

import json
import os
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.providers.base import (
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)
from app.coordination.providers.hetzner_provider import (
    HetznerProvider,
    HETZNER_PLANS,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def hetzner_provider():
    """Create HetznerProvider with mocked CLI and token."""
    with patch("shutil.which", return_value="/usr/local/bin/hcloud"):
        with patch.dict(os.environ, {"HCLOUD_TOKEN": "test-token"}):
            return HetznerProvider()


@pytest.fixture
def hetzner_provider_cli_only():
    """Create HetznerProvider with CLI but no token."""
    with patch("shutil.which", return_value="/usr/local/bin/hcloud"):
        with patch.dict(os.environ, {}, clear=True):
            return HetznerProvider()


@pytest.fixture
def hetzner_provider_token_only():
    """Create HetznerProvider with token but no CLI."""
    with patch("shutil.which", return_value=None):
        return HetznerProvider(token="test-token")


@pytest.fixture
def hetzner_provider_unconfigured():
    """Create HetznerProvider with neither CLI nor token."""
    with patch("shutil.which", return_value=None):
        with patch.dict(os.environ, {}, clear=True):
            return HetznerProvider()


@pytest.fixture
def sample_server_data():
    """Sample Hetzner server JSON data."""
    return {
        "id": 12345,
        "name": "ringrift-cpu-1",
        "status": "running",
        "server_type": {"name": "ccx33"},
        "public_net": {
            "ipv4": {"ip": "192.168.1.100"},
            "ipv6": {"ip": "2a01:4f8:c010:1234::1"},
        },
        "created": "2024-01-01T12:00:00+00:00",
        "datacenter": {"name": "fsn1-dc14"},
        "labels": {"environment": True, "project": True},
    }


# ===========================================================================
# Initialization Tests
# ===========================================================================


class TestHetznerProviderInit:
    """Test HetznerProvider initialization."""

    def test_init_with_env_token(self):
        """Test token read from environment variable."""
        with patch("shutil.which", return_value=None):
            with patch.dict(os.environ, {"HCLOUD_TOKEN": "env-token"}):
                provider = HetznerProvider()

                assert provider._token == "env-token"

    def test_init_with_explicit_token(self):
        """Test explicit token overrides environment."""
        with patch("shutil.which", return_value=None):
            with patch.dict(os.environ, {"HCLOUD_TOKEN": "env-token"}):
                provider = HetznerProvider(token="explicit-token")

                assert provider._token == "explicit-token"

    def test_init_detects_cli(self):
        """Test CLI detection via shutil.which."""
        with patch("shutil.which", return_value="/usr/local/bin/hcloud") as mock_which:
            provider = HetznerProvider()

            mock_which.assert_called_once_with("hcloud")
            assert provider._cli_path == "/usr/local/bin/hcloud"

    def test_init_no_cli(self):
        """Test behavior when CLI not found."""
        with patch("shutil.which", return_value=None):
            provider = HetznerProvider()

            assert provider._cli_path is None


class TestHetznerProviderProperties:
    """Test HetznerProvider properties."""

    def test_provider_type(self, hetzner_provider):
        """Test provider_type returns HETZNER."""
        assert hetzner_provider.provider_type == ProviderType.HETZNER

    def test_name(self, hetzner_provider):
        """Test name returns 'Hetzner'."""
        assert hetzner_provider.name == "Hetzner"


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestHetznerProviderConfiguration:
    """Test HetznerProvider configuration detection."""

    def test_is_configured_with_token(self, hetzner_provider_token_only):
        """Test is_configured when token available."""
        assert hetzner_provider_token_only.is_configured() is True

    def test_is_configured_with_cli(self, hetzner_provider_cli_only):
        """Test is_configured when CLI available."""
        assert hetzner_provider_cli_only.is_configured() is True

    def test_is_configured_with_both(self, hetzner_provider):
        """Test is_configured when both available."""
        assert hetzner_provider.is_configured() is True

    def test_is_configured_unconfigured(self, hetzner_provider_unconfigured):
        """Test is_configured when neither available."""
        assert hetzner_provider_unconfigured.is_configured() is False


# ===========================================================================
# HETZNER_PLANS Tests
# ===========================================================================


class TestHetznerPlans:
    """Test HETZNER_PLANS configuration."""

    def test_plans_structure(self):
        """Test plan data has correct structure."""
        for plan_name, (vcpus, ram_gb, cost) in HETZNER_PLANS.items():
            assert isinstance(plan_name, str)
            assert isinstance(vcpus, int)
            assert isinstance(ram_gb, int)
            assert isinstance(cost, float)
            assert vcpus > 0
            assert ram_gb > 0
            assert cost > 0

    def test_known_plans_exist(self):
        """Test expected plan types exist."""
        expected = ["cx22", "cx32", "ccx33", "ccx53"]
        for plan in expected:
            assert plan in HETZNER_PLANS


# ===========================================================================
# Instance Parsing Tests
# ===========================================================================


class TestHetznerProviderParseInstance:
    """Test HetznerProvider._parse_instance()."""

    def test_parse_running_instance(self, hetzner_provider, sample_server_data):
        """Test parsing a running server."""
        instance = hetzner_provider._parse_instance(sample_server_data)

        assert instance.id == "12345"
        assert instance.provider == ProviderType.HETZNER
        assert instance.name == "ringrift-cpu-1"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.CPU_ONLY
        assert instance.gpu_count == 0
        assert instance.gpu_memory_gb == 0
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 22
        assert instance.ssh_user == "root"
        assert instance.cost_per_hour == 0.056  # ccx33 cost
        assert instance.region == "fsn1-dc14"

    def test_parse_instance_status_mapping(self, hetzner_provider):
        """Test status mapping for various states."""
        status_cases = [
            ("initializing", InstanceStatus.PENDING),
            ("starting", InstanceStatus.STARTING),
            ("running", InstanceStatus.RUNNING),
            ("stopping", InstanceStatus.STOPPING),
            ("off", InstanceStatus.STOPPED),
            ("deleting", InstanceStatus.TERMINATED),
            ("unknown", InstanceStatus.UNKNOWN),
        ]

        for hetzner_status, expected in status_cases:
            data = {"id": 1, "status": hetzner_status}
            instance = hetzner_provider._parse_instance(data)
            assert instance.status == expected, f"Status {hetzner_status} should map to {expected}"

    def test_parse_instance_with_missing_fields(self, hetzner_provider):
        """Test parsing with minimal data."""
        data = {"id": 123}
        instance = hetzner_provider._parse_instance(data)

        assert instance.id == "123"
        assert instance.name == ""
        assert instance.gpu_type == GPUType.CPU_ONLY
        assert instance.ip_address is None

    def test_parse_instance_unknown_server_type(self, hetzner_provider):
        """Test parsing with unknown server type uses defaults."""
        data = {
            "id": 123,
            "server_type": {"name": "unknown-type"},
        }
        instance = hetzner_provider._parse_instance(data)

        # Should use default cost
        assert instance.cost_per_hour == 0.01

    def test_parse_instance_datetime(self, hetzner_provider, sample_server_data):
        """Test datetime parsing."""
        instance = hetzner_provider._parse_instance(sample_server_data)

        assert instance.created_at is not None
        assert isinstance(instance.created_at, datetime)

    def test_parse_instance_labels_to_tags(self, hetzner_provider, sample_server_data):
        """Test labels are converted to tags."""
        instance = hetzner_provider._parse_instance(sample_server_data)

        assert "environment" in instance.tags
        assert "project" in instance.tags


# ===========================================================================
# CLI Execution Tests
# ===========================================================================


class TestHetznerProviderCLI:
    """Test HetznerProvider CLI execution."""

    @pytest.mark.asyncio
    async def test_run_cli_success(self, hetzner_provider):
        """Test successful CLI execution."""
        mock_result = MagicMock()
        mock_result.stdout = '{"status": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.hetzner_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            stdout, stderr, rc = await hetzner_provider._run_cli("server", "list")

            assert stdout == '{"status": "ok"}'
            assert stderr == ""
            assert rc == 0

    @pytest.mark.asyncio
    async def test_run_cli_adds_json_flag(self, hetzner_provider):
        """Test that -o json flag is added to commands."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.hetzner_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            await hetzner_provider._run_cli("server", "list")

            call_args = mock_run.await_args.args[0]
            assert "-o" in call_args
            assert "json" in call_args

    @pytest.mark.asyncio
    async def test_run_cli_passes_token_in_env(self, hetzner_provider):
        """Test that token is passed in environment."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.hetzner_provider.async_subprocess_run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            await hetzner_provider._run_cli("server", "list")

            call_kwargs = mock_run.await_args.kwargs
            assert "HCLOUD_TOKEN" in call_kwargs["env"]

    @pytest.mark.asyncio
    async def test_run_cli_no_cli_path(self, hetzner_provider_token_only):
        """Test CLI execution when CLI not available."""
        with pytest.raises(RuntimeError, match="hcloud CLI not found"):
            await hetzner_provider_token_only._run_cli("server", "list")


# ===========================================================================
# List Instances Tests
# ===========================================================================


class TestHetznerProviderListInstances:
    """Test HetznerProvider.list_instances()."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self, hetzner_provider, sample_server_data):
        """Test successful instance listing."""
        servers_json = json.dumps([sample_server_data])

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (servers_json, "", 0)

            instances = await hetzner_provider.list_instances()

            assert len(instances) == 1
            assert instances[0].id == "12345"
            mock_cli.assert_called_once_with("server", "list")

    @pytest.mark.asyncio
    async def test_list_instances_empty(self, hetzner_provider):
        """Test listing when no servers exist."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("[]", "", 0)

            instances = await hetzner_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_cli_error(self, hetzner_provider):
        """Test listing when CLI fails."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "API error", 1)

            instances = await hetzner_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self, hetzner_provider):
        """Test listing with invalid JSON response."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("not valid json", "", 0)

            instances = await hetzner_provider.list_instances()

            assert instances == []


# ===========================================================================
# Get Instance Tests
# ===========================================================================


class TestHetznerProviderGetInstance:
    """Test HetznerProvider.get_instance()."""

    @pytest.mark.asyncio
    async def test_get_instance_success(self, hetzner_provider, sample_server_data):
        """Test getting specific instance."""
        server_json = json.dumps(sample_server_data)

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (server_json, "", 0)

            instance = await hetzner_provider.get_instance("12345")

            assert instance is not None
            assert instance.id == "12345"
            mock_cli.assert_called_once_with("server", "describe", "12345")

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, hetzner_provider):
        """Test getting non-existent instance."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "Server not found", 1)

            instance = await hetzner_provider.get_instance("99999")

            assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_invalid_json(self, hetzner_provider):
        """Test getting instance with invalid response."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("invalid json", "", 0)

            instance = await hetzner_provider.get_instance("12345")

            assert instance is None


# ===========================================================================
# Scale Up Tests
# ===========================================================================


class TestHetznerProviderScaleUp:
    """Test HetznerProvider.scale_up()."""

    @pytest.mark.asyncio
    async def test_scale_up_success(self, hetzner_provider, sample_server_data):
        """Test successful scale up."""
        create_response = json.dumps({"server": sample_server_data})

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (create_response, "", 0)

            instances = await hetzner_provider.scale_up(GPUType.CPU_ONLY, count=1)

            assert len(instances) == 1
            assert instances[0].id == "12345"

    @pytest.mark.asyncio
    async def test_scale_up_ignores_gpu_type(self, hetzner_provider, sample_server_data):
        """Test that GPU type is ignored (Hetzner is CPU only)."""
        create_response = json.dumps({"server": sample_server_data})

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (create_response, "", 0)

            # Should work even with GPU type specified
            instances = await hetzner_provider.scale_up(GPUType.RTX_4090, count=1)

            assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_scale_up_with_region(self, hetzner_provider, sample_server_data):
        """Test scale up with specific region."""
        create_response = json.dumps({"server": sample_server_data})

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (create_response, "", 0)

            await hetzner_provider.scale_up(GPUType.CPU_ONLY, count=1, region="nbg1")

            call_args = mock_cli.call_args[0]
            assert "--datacenter" in call_args
            assert "nbg1" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_default_location(self, hetzner_provider, sample_server_data):
        """Test scale up uses default Falkenstein location."""
        create_response = json.dumps({"server": sample_server_data})

        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = (create_response, "", 0)

            await hetzner_provider.scale_up(GPUType.CPU_ONLY, count=1)

            call_args = mock_cli.call_args[0]
            assert "--location" in call_args
            assert "fsn1" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_create_fails(self, hetzner_provider):
        """Test scale up when create fails."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "Quota exceeded", 1)

            instances = await hetzner_provider.scale_up(GPUType.CPU_ONLY, count=1)

            assert instances == []


# ===========================================================================
# Scale Down Tests
# ===========================================================================


class TestHetznerProviderScaleDown:
    """Test HetznerProvider.scale_down()."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self, hetzner_provider):
        """Test successful server deletion."""
        with patch.object(hetzner_provider, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await hetzner_provider.scale_down(["12345", "67890"])

            assert results == {"12345": True, "67890": True}
            assert mock_cli.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self, hetzner_provider):
        """Test scale down with partial failure."""
        call_count = 0
        async def mock_run_cli(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", "", 0)  # Success
            else:
                return ("", "Server not found", 1)  # Failure

        with patch.object(hetzner_provider, "_run_cli", side_effect=mock_run_cli):
            results = await hetzner_provider.scale_down(["12345", "99999"])

            assert results["12345"] is True
            assert results["99999"] is False

    @pytest.mark.asyncio
    async def test_scale_down_empty_list(self, hetzner_provider):
        """Test scale down with empty list."""
        results = await hetzner_provider.scale_down([])

        assert results == {}


# ===========================================================================
# Cost Tests
# ===========================================================================


class TestHetznerProviderCost:
    """Test HetznerProvider cost estimation."""

    def test_get_cost_per_hour(self, hetzner_provider):
        """Test cost returns ccx33 default."""
        # Hetzner doesn't have GPUs, always returns CPU server cost
        cost = hetzner_provider.get_cost_per_hour(GPUType.RTX_4090)

        assert cost == 0.056  # ccx33 cost

    def test_get_cost_per_hour_cpu(self, hetzner_provider):
        """Test cost for CPU type."""
        cost = hetzner_provider.get_cost_per_hour(GPUType.CPU_ONLY)

        assert cost == 0.056


# ===========================================================================
# GPU Availability Tests
# ===========================================================================


class TestHetznerProviderGPU:
    """Test HetznerProvider GPU-related methods."""

    @pytest.mark.asyncio
    async def test_get_available_gpus_returns_cpu(self, hetzner_provider):
        """Test get_available_gpus returns CPU_ONLY."""
        gpus = await hetzner_provider.get_available_gpus()

        assert gpus == {GPUType.CPU_ONLY: 100}


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestHetznerProviderHealthCheck:
    """Test HetznerProvider.health_check()."""

    def test_health_check_healthy(self, hetzner_provider):
        """Test health check when configured."""
        result = hetzner_provider.health_check()

        assert result.healthy is True
        assert "configured" in result.message.lower()

    def test_health_check_unconfigured(self, hetzner_provider_unconfigured):
        """Test health check when not configured."""
        result = hetzner_provider_unconfigured.health_check()

        assert result.healthy is False
        assert "not found" in result.message.lower() or "token" in result.message.lower()

    def test_health_check_token_only(self, hetzner_provider_token_only):
        """Test health check with token only."""
        result = hetzner_provider_token_only.health_check()

        assert result.healthy is True
        assert result.details["has_token"] is True

    def test_health_check_cli_only(self, hetzner_provider_cli_only):
        """Test health check with CLI only."""
        result = hetzner_provider_cli_only.health_check()

        assert result.healthy is True
        assert result.details["cli_path"] is not None
