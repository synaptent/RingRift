"""Tests for Vultr cloud provider.

Coverage for app/coordination/providers/vultr_provider.py (~302 LOC).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def mock_cli_path():
    """Mock vultr-cli path."""
    return "/usr/local/bin/vultr-cli"


@pytest.fixture
def sample_instance_json():
    """Sample Vultr instance JSON response."""
    return {
        "id": "abc-123-def",
        "label": "ringrift-a100_40gb-0",
        "status": "active",
        "plan": "vcg-a100-1c-6g-4vram",
        "main_ip": "192.168.1.100",
        "date_created": "2025-12-28T10:30:00Z",
        "region": "ewr",
        "tags": {"project": "ringrift"},
    }


@pytest.fixture
def sample_instances_list_json(sample_instance_json):
    """Sample list of instances."""
    return {
        "instances": [
            sample_instance_json,
            {
                "id": "abc-456-ghi",
                "label": "ringrift-h100-0",
                "status": "pending",
                "plan": "vcg-h100-1c-80g",
                "main_ip": None,
                "date_created": "2025-12-28T11:00:00Z",
                "region": "lax",
                "tags": {},
            },
        ]
    }


@pytest.fixture
def sample_ssh_keys_json():
    """Sample SSH keys list."""
    return {
        "ssh_keys": [
            {"id": "ssh-key-123", "name": "ringrift-cluster", "ssh_key": "ssh-rsa ..."},
            {"id": "ssh-key-456", "name": "other-key", "ssh_key": "ssh-rsa ..."},
        ]
    }


@pytest.fixture
def provider_with_cli():
    """Create VultrProvider with CLI."""
    from app.coordination.providers.vultr_provider import VultrProvider
    with patch.object(VultrProvider, "_find_cli", return_value="/usr/bin/vultr-cli"):
        return VultrProvider()


@pytest.fixture
def provider_with_explicit_cli():
    """Create VultrProvider with explicit CLI path."""
    from app.coordination.providers.vultr_provider import VultrProvider
    return VultrProvider(cli_path="/custom/path/vultr-cli")


# ============================================================================
# TestVultrGPUPlans - Module-level constants
# ============================================================================


class TestVultrGPUPlans:
    """Test VULTR_GPU_PLANS constant."""

    def test_plans_dictionary_exists(self):
        """VULTR_GPU_PLANS is a dictionary."""
        from app.coordination.providers.vultr_provider import VULTR_GPU_PLANS
        assert isinstance(VULTR_GPU_PLANS, dict)

    def test_plans_has_a100_types(self):
        """A100 GPU types are present."""
        from app.coordination.providers.vultr_provider import VULTR_GPU_PLANS
        assert "vcg-a100-1c-6g-4vram" in VULTR_GPU_PLANS
        assert "vcg-a100-2c-12g-8vram" in VULTR_GPU_PLANS
        assert "vcg-a100-3c-24g-16vram" in VULTR_GPU_PLANS

    def test_plans_has_h100_type(self):
        """H100 GPU type is present."""
        from app.coordination.providers.vultr_provider import VULTR_GPU_PLANS
        assert "vcg-h100-1c-80g" in VULTR_GPU_PLANS

    def test_plan_tuple_format(self):
        """Each plan has (gpu_type, memory_gb, cost) format."""
        from app.coordination.providers.vultr_provider import VULTR_GPU_PLANS
        for plan_name, plan_info in VULTR_GPU_PLANS.items():
            assert isinstance(plan_info, tuple), f"{plan_name} not a tuple"
            assert len(plan_info) == 3, f"{plan_name} has {len(plan_info)} elements"
            gpu_type, memory_gb, cost = plan_info
            assert isinstance(gpu_type, GPUType), f"{plan_name} gpu_type not GPUType"
            assert isinstance(memory_gb, int), f"{plan_name} memory not int"
            assert isinstance(cost, float), f"{plan_name} cost not float"


class TestDefaultSSHKey:
    """Test DEFAULT_SSH_KEY constant."""

    def test_default_ssh_key_value(self):
        """DEFAULT_SSH_KEY has expected value."""
        from app.coordination.providers.vultr_provider import DEFAULT_SSH_KEY
        assert DEFAULT_SSH_KEY == "ringrift-cluster"


# ============================================================================
# TestVultrProviderInit - Constructor
# ============================================================================


class TestVultrProviderInit:
    """Test VultrProvider initialization."""

    def test_init_with_explicit_cli_path(self):
        """Provider uses explicit CLI path."""
        from app.coordination.providers.vultr_provider import VultrProvider
        provider = VultrProvider(cli_path="/custom/vultr-cli")
        assert provider._cli_path == "/custom/vultr-cli"

    def test_init_auto_detects_cli(self):
        """Provider auto-detects CLI path."""
        from app.coordination.providers.vultr_provider import VultrProvider
        with patch.object(VultrProvider, "_find_cli", return_value="/auto/vultr-cli"):
            provider = VultrProvider()
            assert provider._cli_path == "/auto/vultr-cli"

    def test_init_ssh_key_id_none(self):
        """SSH key ID starts as None."""
        from app.coordination.providers.vultr_provider import VultrProvider
        with patch.object(VultrProvider, "_find_cli", return_value=None):
            provider = VultrProvider()
            assert provider._ssh_key_id is None


# ============================================================================
# TestVultrProviderProperties - Provider properties
# ============================================================================


class TestVultrProviderProperties:
    """Test VultrProvider properties."""

    def test_provider_type(self, provider_with_cli):
        """provider_type returns VULTR."""
        assert provider_with_cli.provider_type == ProviderType.VULTR

    def test_name(self, provider_with_cli):
        """name returns 'Vultr'."""
        assert provider_with_cli.name == "Vultr"

    def test_inherits_from_cloud_provider(self, provider_with_cli):
        """VultrProvider inherits from CloudProvider."""
        assert isinstance(provider_with_cli, CloudProvider)


# ============================================================================
# TestVultrProviderFindCli - CLI detection
# ============================================================================


class TestVultrProviderFindCli:
    """Test _find_cli() method."""

    def test_find_cli_via_which(self):
        """Find CLI via shutil.which."""
        from app.coordination.providers.vultr_provider import VultrProvider

        def mock_exists(self):
            """Mock Path.exists - returns True only for the which() result."""
            return str(self) == "/usr/bin/vultr-cli"

        with patch("shutil.which", return_value="/usr/bin/vultr-cli"):
            with patch.object(Path, "exists", mock_exists):
                provider = VultrProvider.__new__(VultrProvider)
                result = provider._find_cli()
                assert result == "/usr/bin/vultr-cli"

    def test_find_cli_home_local(self):
        """Find CLI in ~/.local/bin."""
        from app.coordination.providers.vultr_provider import VultrProvider

        def mock_exists(self):
            """Mock Path.exists - returns True for home .local path."""
            return ".local/bin/vultr-cli" in str(self)

        with patch("shutil.which", return_value=None):
            with patch.object(Path, "exists", mock_exists):
                provider = VultrProvider.__new__(VultrProvider)
                result = provider._find_cli()
                assert result is not None
                assert ".local/bin/vultr-cli" in result

    def test_find_cli_not_found(self):
        """Return None when CLI not found."""
        from app.coordination.providers.vultr_provider import VultrProvider
        with patch("shutil.which", return_value=None):
            with patch.object(Path, "exists", return_value=False):
                provider = VultrProvider.__new__(VultrProvider)
                result = provider._find_cli()
                assert result is None


# ============================================================================
# TestVultrProviderIsConfigured - Configuration check
# ============================================================================


class TestVultrProviderIsConfigured:
    """Test is_configured() method."""

    def test_configured_with_cli_and_config(self):
        """Provider is configured with CLI and config file."""
        from app.coordination.providers.vultr_provider import VultrProvider
        with patch.object(VultrProvider, "_find_cli", return_value="/usr/bin/vultr-cli"):
            provider = VultrProvider()
            with patch.object(Path, "exists", return_value=True):
                assert provider.is_configured() is True

    def test_not_configured_without_cli(self):
        """Provider is not configured without CLI."""
        from app.coordination.providers.vultr_provider import VultrProvider
        provider = VultrProvider(cli_path=None)
        provider._cli_path = None
        assert provider.is_configured() is False

    def test_not_configured_without_config_file(self):
        """Provider is not configured without config file."""
        from app.coordination.providers.vultr_provider import VultrProvider
        with patch.object(VultrProvider, "_find_cli", return_value="/usr/bin/vultr-cli"):
            provider = VultrProvider()
            with patch.object(Path, "exists", return_value=False):
                assert provider.is_configured() is False


# ============================================================================
# TestVultrProviderRunCli - CLI execution
# ============================================================================


class TestVultrProviderRunCli:
    """Test _run_cli() method."""

    @pytest.mark.asyncio
    async def test_run_cli_success(self, provider_with_explicit_cli):
        """CLI execution returns stdout, stderr, returncode."""
        mock_result = MagicMock()
        mock_result.stdout = '{"instances": []}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "app.coordination.providers.vultr_provider.async_subprocess_run",
            new=AsyncMock(return_value=mock_result),
        ):
            stdout, stderr, rc = await provider_with_explicit_cli._run_cli("instance", "list")

        assert stdout == '{"instances": []}'
        assert stderr == ""
        assert rc == 0

    @pytest.mark.asyncio
    async def test_run_cli_adds_json_output(self, provider_with_explicit_cli):
        """CLI adds --output json flag."""
        mock_result = MagicMock()
        mock_result.stdout = "{}"
        mock_result.stderr = ""
        mock_result.returncode = 0

        mock_run = AsyncMock(return_value=mock_result)
        with patch(
            "app.coordination.providers.vultr_provider.async_subprocess_run",
            new=mock_run,
        ):
            await provider_with_explicit_cli._run_cli("instance", "list")

        call_args = mock_run.await_args.args[0]
        assert "--output" in call_args
        assert "json" in call_args

    @pytest.mark.asyncio
    async def test_run_cli_no_cli_path_raises(self):
        """CLI raises RuntimeError if vultr-cli not found."""
        from app.coordination.providers.vultr_provider import VultrProvider
        provider = VultrProvider(cli_path=None)
        provider._cli_path = None

        with pytest.raises(RuntimeError, match="vultr-cli not found"):
            await provider._run_cli("instance", "list")

    @pytest.mark.asyncio
    async def test_run_cli_handles_error(self, provider_with_explicit_cli):
        """CLI handles command failure."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error: unauthorized"
        mock_result.returncode = 1

        with patch(
            "app.coordination.providers.vultr_provider.async_subprocess_run",
            new=AsyncMock(return_value=mock_result),
        ):
            stdout, stderr, rc = await provider_with_explicit_cli._run_cli("instance", "list")

        assert rc == 1
        assert "unauthorized" in stderr


# ============================================================================
# TestVultrProviderParseInstance - Instance parsing
# ============================================================================


class TestVultrProviderParseInstance:
    """Test _parse_instance() method."""

    def test_parse_running_instance(self, provider_with_cli, sample_instance_json):
        """Parse running instance."""
        instance = provider_with_cli._parse_instance(sample_instance_json)

        assert instance.id == "abc-123-def"
        assert instance.provider == ProviderType.VULTR
        assert instance.name == "ringrift-a100_40gb-0"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.A100_40GB
        assert instance.gpu_count == 1
        assert instance.gpu_memory_gb == 20
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 22
        assert instance.ssh_user == "root"
        assert instance.region == "ewr"

    def test_parse_status_mapping(self, provider_with_cli):
        """Parse different status values."""
        statuses = {
            "pending": InstanceStatus.PENDING,
            "active": InstanceStatus.RUNNING,
            "suspended": InstanceStatus.STOPPED,
            "resizing": InstanceStatus.STARTING,
            "unknown_status": InstanceStatus.UNKNOWN,
        }

        for vultr_status, expected_status in statuses.items():
            data = {"id": "test-id", "status": vultr_status}
            instance = provider_with_cli._parse_instance(data)
            assert instance.status == expected_status, f"Failed for {vultr_status}"

    def test_parse_gpu_from_plan(self, provider_with_cli):
        """Parse GPU type from plan."""
        data = {
            "id": "test-id",
            "status": "active",
            "plan": "vcg-h100-1c-80g",
        }
        instance = provider_with_cli._parse_instance(data)
        assert instance.gpu_type == GPUType.H100_80GB
        assert instance.gpu_memory_gb == 80

    def test_parse_unknown_plan(self, provider_with_cli):
        """Unknown plan returns UNKNOWN GPU type."""
        data = {
            "id": "test-id",
            "status": "active",
            "plan": "unknown-plan",
        }
        instance = provider_with_cli._parse_instance(data)
        assert instance.gpu_type == GPUType.UNKNOWN

    def test_parse_cost_from_plan(self, provider_with_cli, sample_instance_json):
        """Parse cost from plan."""
        from app.coordination.providers.vultr_provider import VULTR_GPU_PLANS
        instance = provider_with_cli._parse_instance(sample_instance_json)
        expected_cost = VULTR_GPU_PLANS["vcg-a100-1c-6g-4vram"][2]
        assert instance.cost_per_hour == expected_cost

    def test_parse_datetime(self, provider_with_cli, sample_instance_json):
        """Parse created timestamp."""
        instance = provider_with_cli._parse_instance(sample_instance_json)
        assert instance.created_at is not None
        assert instance.created_at.year == 2025
        assert instance.created_at.month == 12
        assert instance.created_at.day == 28

    def test_parse_tags(self, provider_with_cli, sample_instance_json):
        """Parse tags."""
        instance = provider_with_cli._parse_instance(sample_instance_json)
        assert instance.tags == {"project": "ringrift"}

    def test_parse_raw_data_preserved(self, provider_with_cli, sample_instance_json):
        """Raw data is preserved."""
        instance = provider_with_cli._parse_instance(sample_instance_json)
        assert instance.raw_data == sample_instance_json


# ============================================================================
# TestVultrProviderListInstances - List instances
# ============================================================================


class TestVultrProviderListInstances:
    """Test list_instances() method."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self, provider_with_cli, sample_instances_list_json):
        """List instances returns parsed instances."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps(sample_instances_list_json), "", 0)

            instances = await provider_with_cli.list_instances()

            assert len(instances) == 2
            assert instances[0].id == "abc-123-def"
            assert instances[1].id == "abc-456-ghi"

    @pytest.mark.asyncio
    async def test_list_instances_empty(self, provider_with_cli):
        """List instances returns empty list."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instances": []}), "", 0)

            instances = await provider_with_cli.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_cli_failure(self, provider_with_cli):
        """List instances handles CLI failure."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "API error", 1)

            instances = await provider_with_cli.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self, provider_with_cli):
        """List instances handles invalid JSON."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("not valid json", "", 0)

            instances = await provider_with_cli.list_instances()

            assert instances == []


# ============================================================================
# TestVultrProviderGetInstance - Get specific instance
# ============================================================================


class TestVultrProviderGetInstance:
    """Test get_instance() method."""

    @pytest.mark.asyncio
    async def test_get_instance_success(self, provider_with_cli, sample_instance_json):
        """Get instance by ID."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)

            instance = await provider_with_cli.get_instance("abc-123-def")

            assert instance is not None
            assert instance.id == "abc-123-def"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, provider_with_cli):
        """Get instance returns None for missing ID."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "instance not found", 1)

            instance = await provider_with_cli.get_instance("nonexistent")

            assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_invalid_json(self, provider_with_cli):
        """Get instance handles invalid JSON."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("invalid", "", 0)

            instance = await provider_with_cli.get_instance("abc-123")

            assert instance is None


# ============================================================================
# TestVultrProviderGetSSHKeyId - SSH key lookup
# ============================================================================


class TestVultrProviderGetSSHKeyId:
    """Test _get_ssh_key_id() method."""

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_cached(self, provider_with_cli):
        """Returns cached SSH key ID."""
        provider_with_cli._ssh_key_id = "cached-key-id"

        result = await provider_with_cli._get_ssh_key_id()

        assert result == "cached-key-id"

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_found(self, provider_with_cli, sample_ssh_keys_json):
        """Finds SSH key by name."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps(sample_ssh_keys_json), "", 0)

            result = await provider_with_cli._get_ssh_key_id()

            assert result == "ssh-key-123"
            assert provider_with_cli._ssh_key_id == "ssh-key-123"

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_not_found(self, provider_with_cli):
        """Returns None when key not found."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"ssh_keys": []}), "", 0)

            result = await provider_with_cli._get_ssh_key_id()

            assert result is None

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_cli_error(self, provider_with_cli):
        """Returns None on CLI error."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "error", 1)

            result = await provider_with_cli._get_ssh_key_id()

            assert result is None


# ============================================================================
# TestVultrProviderScaleUp - Create instances
# ============================================================================


class TestVultrProviderScaleUp:
    """Test scale_up() method."""

    @pytest.mark.asyncio
    async def test_scale_up_single_instance(self, provider_with_cli, sample_instance_json):
        """Create a single instance."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value="ssh-key-123"):
                instances = await provider_with_cli.scale_up(GPUType.A100_40GB, count=1)

        assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_scale_up_multiple_instances(self, provider_with_cli, sample_instance_json):
        """Create multiple instances."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value=None):
                instances = await provider_with_cli.scale_up(GPUType.A100_40GB, count=3)

        assert len(instances) == 3

    @pytest.mark.asyncio
    async def test_scale_up_unsupported_gpu(self, provider_with_cli):
        """Scale up returns empty for unsupported GPU type."""
        instances = await provider_with_cli.scale_up(GPUType.RTX_3090, count=1)
        assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_with_region(self, provider_with_cli, sample_instance_json):
        """Scale up with specific region."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value=None):
                await provider_with_cli.scale_up(GPUType.A100_40GB, count=1, region="lax")

        call_args = mock_cli.call_args[0]
        assert "--region" in call_args
        assert "lax" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_default_region(self, provider_with_cli, sample_instance_json):
        """Scale up uses default region."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value=None):
                await provider_with_cli.scale_up(GPUType.A100_40GB, count=1)

        call_args = mock_cli.call_args[0]
        assert "--region" in call_args
        assert "ewr" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_with_ssh_key(self, provider_with_cli, sample_instance_json):
        """Scale up includes SSH key when available."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = (json.dumps({"instance": sample_instance_json}), "", 0)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value="key-123"):
                await provider_with_cli.scale_up(GPUType.A100_40GB, count=1)

        call_args = mock_cli.call_args[0]
        assert "--ssh-keys" in call_args
        assert "key-123" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_handles_failure(self, provider_with_cli):
        """Scale up handles creation failure."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "quota exceeded", 1)
            with patch.object(provider_with_cli, "_get_ssh_key_id", return_value=None):
                instances = await provider_with_cli.scale_up(GPUType.A100_40GB, count=1)

        assert instances == []


# ============================================================================
# TestVultrProviderScaleDown - Terminate instances
# ============================================================================


class TestVultrProviderScaleDown:
    """Test scale_down() method."""

    @pytest.mark.asyncio
    async def test_scale_down_single_instance(self, provider_with_cli):
        """Delete a single instance."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await provider_with_cli.scale_down(["abc-123"])

            assert results == {"abc-123": True}

    @pytest.mark.asyncio
    async def test_scale_down_multiple_instances(self, provider_with_cli):
        """Delete multiple instances."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await provider_with_cli.scale_down(["abc-123", "def-456"])

            assert results == {"abc-123": True, "def-456": True}

    @pytest.mark.asyncio
    async def test_scale_down_handles_failure(self, provider_with_cli):
        """Scale down handles deletion failure."""
        with patch.object(provider_with_cli, "_run_cli") as mock_cli:
            mock_cli.return_value = ("", "instance locked", 1)

            results = await provider_with_cli.scale_down(["abc-123"])

            assert results == {"abc-123": False}

    @pytest.mark.asyncio
    async def test_scale_down_mixed_results(self, provider_with_cli):
        """Scale down handles mixed success/failure."""
        call_count = [0]
        
        async def mock_run_cli(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("", "", 0)  # Success
            else:
                return ("", "error", 1)  # Failure

        with patch.object(provider_with_cli, "_run_cli", side_effect=mock_run_cli):
            results = await provider_with_cli.scale_down(["abc-123", "def-456"])

        assert results["abc-123"] is True
        assert results["def-456"] is False


# ============================================================================
# TestVultrProviderCost - Cost calculation
# ============================================================================


class TestVultrProviderCost:
    """Test get_cost_per_hour() method."""

    def test_cost_a100_40gb(self, provider_with_cli):
        """Cost for A100 40GB."""
        cost = provider_with_cli.get_cost_per_hour(GPUType.A100_40GB)
        assert cost == 0.62

    def test_cost_a100_80gb(self, provider_with_cli):
        """Cost for A100 80GB."""
        cost = provider_with_cli.get_cost_per_hour(GPUType.A100_80GB)
        assert cost == 2.48

    def test_cost_h100(self, provider_with_cli):
        """Cost for H100."""
        cost = provider_with_cli.get_cost_per_hour(GPUType.H100_80GB)
        assert cost == 3.99

    def test_cost_unknown_gpu(self, provider_with_cli):
        """Cost for unknown GPU returns 0."""
        cost = provider_with_cli.get_cost_per_hour(GPUType.RTX_3090)
        assert cost == 0.0


# ============================================================================
# TestVultrProviderHealthCheck - Health monitoring
# ============================================================================


class TestVultrProviderHealthCheck:
    """Test health_check() method."""

    def test_health_check_configured(self, provider_with_cli):
        """Health check passes when configured."""
        with patch.object(Path, "exists", return_value=True):
            result = provider_with_cli.health_check()

        assert result.healthy is True
        assert "configured" in result.message

    def test_health_check_no_cli(self):
        """Health check fails without CLI."""
        from app.coordination.providers.vultr_provider import VultrProvider
        provider = VultrProvider(cli_path=None)
        provider._cli_path = None

        result = provider.health_check()

        assert result.healthy is False
        assert "not found" in result.message

    def test_health_check_no_config_file(self, provider_with_cli):
        """Health check fails without config file."""
        with patch.object(Path, "exists", return_value=False):
            result = provider_with_cli.health_check()

        assert result.healthy is False
        assert "config" in result.message.lower()

    def test_health_check_returns_details(self, provider_with_cli):
        """Health check includes details."""
        with patch.object(Path, "exists", return_value=True):
            result = provider_with_cli.health_check()

        assert "cli_path" in result.details
        assert "configured" in result.details
