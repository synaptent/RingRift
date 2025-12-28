"""Tests for VultrProvider cloud provider implementation.

Tests cover:
- Provider initialization and CLI detection
- Configuration checking
- Instance parsing from JSON
- CLI command execution (mocked)
- Instance listing and retrieval
- Scale up/down operations
- SSH key management
- Cost estimation
- Health check integration
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.providers.vultr_provider import (
    VultrProvider,
    VULTR_GPU_PLANS,
    DEFAULT_SSH_KEY,
)
from app.coordination.providers.base import (
    GPUType,
    InstanceStatus,
    ProviderType,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def vultr_provider():
    """Create VultrProvider with mocked CLI path."""
    with patch.object(VultrProvider, "_find_cli", return_value="/usr/local/bin/vultr-cli"):
        return VultrProvider()


@pytest.fixture
def vultr_provider_no_cli():
    """Create VultrProvider with no CLI available."""
    with patch.object(VultrProvider, "_find_cli", return_value=None):
        return VultrProvider()


@pytest.fixture
def sample_instance_data():
    """Sample Vultr instance JSON data."""
    return {
        "id": "abc123-def456",
        "label": "ringrift-training-1",
        "status": "active",
        "plan": "vcg-a100-1c-6g-4vram",
        "main_ip": "192.168.1.100",
        "date_created": "2024-12-27T10:30:00Z",
        "region": "ewr",
        "tags": {"env": "production"},
    }


@pytest.fixture
def sample_ssh_key_data():
    """Sample Vultr SSH key JSON data."""
    return {
        "ssh_keys": [
            {"id": "ssh-key-123", "name": "ringrift-cluster"},
            {"id": "ssh-key-456", "name": "other-key"},
        ]
    }


# ===========================================================================
# Initialization Tests
# ===========================================================================


class TestVultrProviderInit:
    """Test VultrProvider initialization."""

    def test_init_with_explicit_cli_path(self):
        """Test initialization with explicit CLI path."""
        provider = VultrProvider(cli_path="/custom/path/vultr-cli")

        assert provider._cli_path == "/custom/path/vultr-cli"

    def test_init_with_auto_detect(self):
        """Test initialization auto-detects CLI."""
        with patch.object(VultrProvider, "_find_cli", return_value="/usr/bin/vultr-cli"):
            provider = VultrProvider()

            assert provider._cli_path == "/usr/bin/vultr-cli"

    def test_init_cli_not_found(self):
        """Test initialization when CLI not found."""
        with patch.object(VultrProvider, "_find_cli", return_value=None):
            provider = VultrProvider()

            assert provider._cli_path is None

    def test_init_ssh_key_id_starts_none(self, vultr_provider):
        """Test SSH key ID starts as None."""
        assert vultr_provider._ssh_key_id is None


class TestVultrProviderFindCLI:
    """Test VultrProvider._find_cli() method."""

    def test_find_cli_uses_explicit_path(self):
        """Test explicit CLI path is used without calling _find_cli."""
        provider = VultrProvider(cli_path="/custom/vultr-cli")
        assert provider._cli_path == "/custom/vultr-cli"

    def test_find_cli_returns_none_when_not_found(self):
        """Test _find_cli returns None when CLI not available anywhere."""
        with patch.object(VultrProvider, "_find_cli", return_value=None):
            provider = VultrProvider()
            assert provider._cli_path is None


# ===========================================================================
# Properties Tests
# ===========================================================================


class TestVultrProviderProperties:
    """Test VultrProvider properties."""

    def test_provider_type(self, vultr_provider):
        """Test provider_type returns VULTR."""
        assert vultr_provider.provider_type == ProviderType.VULTR

    def test_name(self, vultr_provider):
        """Test name returns 'Vultr'."""
        assert vultr_provider.name == "Vultr"


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestVultrProviderConfiguration:
    """Test VultrProvider configuration checking."""

    def test_is_configured_true(self, vultr_provider):
        """Test is_configured when CLI and config file exist."""
        with patch.object(Path, "exists", return_value=True):
            assert vultr_provider.is_configured() is True

    def test_is_configured_no_cli(self, vultr_provider_no_cli):
        """Test is_configured when CLI missing."""
        assert vultr_provider_no_cli.is_configured() is False

    def test_is_configured_no_config_file(self, vultr_provider):
        """Test is_configured when config file missing."""
        with patch.object(Path, "exists", return_value=False):
            assert vultr_provider.is_configured() is False


# ===========================================================================
# CLI Execution Tests
# ===========================================================================


class TestVultrProviderCLI:
    """Test VultrProvider CLI execution."""

    @pytest.mark.asyncio
    async def test_run_cli_success(self, vultr_provider):
        """Test successful CLI execution."""
        mock_result = MagicMock()
        mock_result.stdout = '{"status": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            stdout, stderr, rc = await vultr_provider._run_cli("instance", "list")

            assert stdout == '{"status": "ok"}'
            assert stderr == ""
            assert rc == 0

    @pytest.mark.asyncio
    async def test_run_cli_adds_json_output(self, vultr_provider):
        """Test that --output json flag is added to commands."""
        mock_result = MagicMock()
        mock_result.stdout = "[]"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await vultr_provider._run_cli("instance", "list")

            call_args = mock_run.call_args[0][0]
            assert "--output" in call_args
            assert "json" in call_args

    @pytest.mark.asyncio
    async def test_run_cli_error(self, vultr_provider):
        """Test CLI execution with error."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Error: API key invalid"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            stdout, stderr, rc = await vultr_provider._run_cli("instance", "list")

            assert rc == 1
            assert "API key invalid" in stderr

    @pytest.mark.asyncio
    async def test_run_cli_no_cli_path(self, vultr_provider_no_cli):
        """Test CLI execution when CLI not available."""
        with pytest.raises(RuntimeError, match="vultr-cli not found"):
            await vultr_provider_no_cli._run_cli("instance", "list")


# ===========================================================================
# Instance Parsing Tests
# ===========================================================================


class TestVultrProviderParseInstance:
    """Test VultrProvider._parse_instance()."""

    def test_parse_active_instance(self, vultr_provider, sample_instance_data):
        """Test parsing an active instance."""
        instance = vultr_provider._parse_instance(sample_instance_data)

        assert instance.id == "abc123-def456"
        assert instance.provider == ProviderType.VULTR
        assert instance.name == "ringrift-training-1"
        assert instance.status == InstanceStatus.RUNNING
        assert instance.gpu_type == GPUType.A100_40GB
        assert instance.gpu_count == 1
        assert instance.gpu_memory_gb == 20  # A100 20GB
        assert instance.ip_address == "192.168.1.100"
        assert instance.ssh_port == 22
        assert instance.ssh_user == "root"
        assert instance.region == "ewr"

    def test_parse_instance_status_mapping(self, vultr_provider):
        """Test status mapping for various states."""
        status_cases = [
            ("pending", InstanceStatus.PENDING),
            ("active", InstanceStatus.RUNNING),
            ("suspended", InstanceStatus.STOPPED),
            ("resizing", InstanceStatus.STARTING),
            ("unknown", InstanceStatus.UNKNOWN),
        ]

        for vultr_status, expected in status_cases:
            data = {"id": "test", "status": vultr_status}
            instance = vultr_provider._parse_instance(data)
            assert instance.status == expected, f"Status {vultr_status} should map to {expected}"

    def test_parse_instance_gpu_plans(self, vultr_provider):
        """Test GPU type parsing from plan IDs."""
        plan_cases = [
            ("vcg-a100-1c-6g-4vram", GPUType.A100_40GB, 20, 0.62),
            ("vcg-a100-2c-12g-8vram", GPUType.A100_40GB, 40, 1.24),
            ("vcg-a100-3c-24g-16vram", GPUType.A100_80GB, 80, 2.48),
            ("vcg-h100-1c-80g", GPUType.H100_80GB, 80, 3.99),
            ("unknown-plan", GPUType.UNKNOWN, 0, 0.0),
        ]

        for plan, expected_gpu, expected_mem, expected_cost in plan_cases:
            data = {"id": "test", "plan": plan}
            instance = vultr_provider._parse_instance(data)
            assert instance.gpu_type == expected_gpu, f"Plan {plan} should have GPU {expected_gpu}"
            assert instance.gpu_memory_gb == expected_mem
            assert instance.cost_per_hour == expected_cost

    def test_parse_instance_with_date(self, vultr_provider, sample_instance_data):
        """Test parsing instance with date_created."""
        instance = vultr_provider._parse_instance(sample_instance_data)

        assert instance.created_at is not None
        assert instance.created_at.year == 2024
        assert instance.created_at.month == 12
        assert instance.created_at.day == 27

    def test_parse_instance_without_date(self, vultr_provider):
        """Test parsing instance without date_created."""
        data = {"id": "test"}
        instance = vultr_provider._parse_instance(data)

        assert instance.created_at is None

    def test_parse_instance_minimal_data(self, vultr_provider):
        """Test parsing with minimal data."""
        data = {"id": "test-123"}
        instance = vultr_provider._parse_instance(data)

        assert instance.id == "test-123"
        assert instance.name == ""
        assert instance.gpu_type == GPUType.UNKNOWN
        assert instance.ip_address is None


# ===========================================================================
# List Instances Tests
# ===========================================================================


class TestVultrProviderListInstances:
    """Test VultrProvider.list_instances()."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self, vultr_provider, sample_instance_data):
        """Test successful instance listing."""
        instances_json = json.dumps({"instances": [sample_instance_data]})

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instances = await vultr_provider.list_instances()

            assert len(instances) == 1
            assert instances[0].id == "abc123-def456"
            mock_cli.assert_called_once_with("instance", "list")

    @pytest.mark.asyncio
    async def test_list_instances_empty(self, vultr_provider):
        """Test listing when no instances exist."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ('{"instances": []}', "", 0)

            instances = await vultr_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_cli_error(self, vultr_provider):
        """Test listing when CLI fails."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "API error", 1)

            instances = await vultr_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self, vultr_provider):
        """Test listing with invalid JSON response."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("not valid json", "", 0)

            instances = await vultr_provider.list_instances()

            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_multiple(self, vultr_provider, sample_instance_data):
        """Test listing multiple instances."""
        instance2 = sample_instance_data.copy()
        instance2["id"] = "xyz789"
        instance2["label"] = "ringrift-training-2"

        instances_json = json.dumps({"instances": [sample_instance_data, instance2]})

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instances_json, "", 0)

            instances = await vultr_provider.list_instances()

            assert len(instances) == 2
            assert instances[0].id == "abc123-def456"
            assert instances[1].id == "xyz789"


# ===========================================================================
# Get Instance Tests
# ===========================================================================


class TestVultrProviderGetInstance:
    """Test VultrProvider.get_instance()."""

    @pytest.mark.asyncio
    async def test_get_instance_success(self, vultr_provider, sample_instance_data):
        """Test getting an existing instance."""
        instance_json = json.dumps({"instance": sample_instance_data})

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instance_json, "", 0)

            instance = await vultr_provider.get_instance("abc123-def456")

            assert instance is not None
            assert instance.id == "abc123-def456"
            mock_cli.assert_called_once_with("instance", "get", "abc123-def456")

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, vultr_provider):
        """Test getting non-existent instance."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "Instance not found", 1)

            instance = await vultr_provider.get_instance("nonexistent")

            assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_invalid_json(self, vultr_provider):
        """Test getting instance with invalid JSON."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("not json", "", 0)

            instance = await vultr_provider.get_instance("abc123")

            assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_direct_data_format(self, vultr_provider, sample_instance_data):
        """Test get_instance with direct data format (no instance wrapper)."""
        instance_json = json.dumps(sample_instance_data)

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (instance_json, "", 0)

            instance = await vultr_provider.get_instance("abc123-def456")

            assert instance is not None
            assert instance.id == "abc123-def456"


# ===========================================================================
# SSH Key Tests
# ===========================================================================


class TestVultrProviderSSHKey:
    """Test VultrProvider SSH key management."""

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_success(self, vultr_provider, sample_ssh_key_data):
        """Test getting SSH key ID."""
        keys_json = json.dumps(sample_ssh_key_data)

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (keys_json, "", 0)

            key_id = await vultr_provider._get_ssh_key_id()

            assert key_id == "ssh-key-123"
            mock_cli.assert_called_once_with("ssh-key", "list")

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_cached(self, vultr_provider):
        """Test SSH key ID is cached after first fetch."""
        vultr_provider._ssh_key_id = "cached-key-id"

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            key_id = await vultr_provider._get_ssh_key_id()

            assert key_id == "cached-key-id"
            mock_cli.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_not_found(self, vultr_provider):
        """Test when default SSH key not found."""
        keys_json = json.dumps({"ssh_keys": [{"id": "other", "name": "other-key"}]})

        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (keys_json, "", 0)

            key_id = await vultr_provider._get_ssh_key_id()

            assert key_id is None

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_cli_error(self, vultr_provider):
        """Test SSH key lookup when CLI fails."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "API error", 1)

            key_id = await vultr_provider._get_ssh_key_id()

            assert key_id is None

    @pytest.mark.asyncio
    async def test_get_ssh_key_id_invalid_json(self, vultr_provider):
        """Test SSH key lookup with invalid JSON."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("not json", "", 0)

            key_id = await vultr_provider._get_ssh_key_id()

            assert key_id is None


# ===========================================================================
# Scale Up Tests
# ===========================================================================


class TestVultrProviderScaleUp:
    """Test VultrProvider.scale_up()."""

    @pytest.mark.asyncio
    async def test_scale_up_success(self, vultr_provider, sample_instance_data):
        """Test successful scale up."""
        create_response = json.dumps({"instance": sample_instance_data})

        with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = "ssh-key-123"

            with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (create_response, "", 0)

                instances = await vultr_provider.scale_up(GPUType.A100_40GB, count=1)

                assert len(instances) == 1
                assert instances[0].id == "abc123-def456"

    @pytest.mark.asyncio
    async def test_scale_up_multiple(self, vultr_provider, sample_instance_data):
        """Test scaling up multiple instances."""
        with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = "ssh-key-123"

            call_count = 0
            async def mock_run_cli(*args):
                nonlocal call_count
                call_count += 1
                instance = sample_instance_data.copy()
                instance["id"] = f"instance-{call_count}"
                return (json.dumps({"instance": instance}), "", 0)

            with patch.object(vultr_provider, "_run_cli", side_effect=mock_run_cli):
                instances = await vultr_provider.scale_up(GPUType.A100_40GB, count=3)

                assert len(instances) == 3

    @pytest.mark.asyncio
    async def test_scale_up_unsupported_gpu(self, vultr_provider):
        """Test scale up with unsupported GPU type."""
        instances = await vultr_provider.scale_up(GPUType.RTX_3090, count=1)

        assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_unknown_gpu(self, vultr_provider):
        """Test scale up with UNKNOWN GPU type."""
        instances = await vultr_provider.scale_up(GPUType.UNKNOWN, count=1)

        assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_create_fails(self, vultr_provider):
        """Test scale up when instance creation fails."""
        with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = "ssh-key-123"

            with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "Creation failed", 1)

                instances = await vultr_provider.scale_up(GPUType.A100_40GB, count=1)

                assert instances == []

    @pytest.mark.asyncio
    async def test_scale_up_without_ssh_key(self, vultr_provider, sample_instance_data):
        """Test scale up without SSH key found."""
        create_response = json.dumps({"instance": sample_instance_data})

        with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = None

            with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (create_response, "", 0)

                instances = await vultr_provider.scale_up(GPUType.A100_40GB, count=1)

                assert len(instances) == 1
                # Verify --ssh-keys not in args
                call_args = mock_cli.call_args[0]
                assert "--ssh-keys" not in call_args

    @pytest.mark.asyncio
    async def test_scale_up_with_region(self, vultr_provider, sample_instance_data):
        """Test scale up with custom region."""
        create_response = json.dumps({"instance": sample_instance_data})

        with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = None

            with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (create_response, "", 0)

                await vultr_provider.scale_up(GPUType.A100_40GB, count=1, region="lax")

                call_args = mock_cli.call_args[0]
                assert "--region" in call_args
                assert "lax" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_gpu_plan_mapping(self, vultr_provider):
        """Test GPU type to plan mapping."""
        gpu_plan_cases = [
            (GPUType.A100_40GB, "vcg-a100-1c-6g-4vram"),
            (GPUType.A100_80GB, "vcg-a100-3c-24g-16vram"),
            (GPUType.H100_80GB, "vcg-h100-1c-80g"),
        ]

        for gpu_type, expected_plan in gpu_plan_cases:
            with patch.object(vultr_provider, "_get_ssh_key_id", new_callable=AsyncMock, return_value=None):
                with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                    mock_cli.return_value = ("", "Simulating plan check", 1)

                    await vultr_provider.scale_up(gpu_type, count=1)

                    call_args = mock_cli.call_args[0]
                    assert "--plan" in call_args
                    plan_idx = call_args.index("--plan")
                    assert call_args[plan_idx + 1] == expected_plan


# ===========================================================================
# Scale Down Tests
# ===========================================================================


class TestVultrProviderScaleDown:
    """Test VultrProvider.scale_down()."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self, vultr_provider):
        """Test successful instance termination."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "", 0)

            results = await vultr_provider.scale_down(["inst-1", "inst-2"])

            assert results == {"inst-1": True, "inst-2": True}
            assert mock_cli.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self, vultr_provider):
        """Test scale down with partial failure."""
        call_count = 0
        async def mock_run_cli(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("", "", 0)  # Success
            else:
                return ("", "Instance not found", 1)  # Failure

        with patch.object(vultr_provider, "_run_cli", side_effect=mock_run_cli):
            results = await vultr_provider.scale_down(["inst-1", "inst-2"])

            assert results["inst-1"] is True
            assert results["inst-2"] is False

    @pytest.mark.asyncio
    async def test_scale_down_empty_list(self, vultr_provider):
        """Test scale down with empty list."""
        results = await vultr_provider.scale_down([])

        assert results == {}

    @pytest.mark.asyncio
    async def test_scale_down_cli_command(self, vultr_provider):
        """Test scale down uses correct CLI command."""
        with patch.object(vultr_provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = ("", "", 0)

            await vultr_provider.scale_down(["test-instance"])

            mock_cli.assert_called_once_with("instance", "delete", "test-instance")


# ===========================================================================
# Cost Tests
# ===========================================================================


class TestVultrProviderCost:
    """Test VultrProvider cost estimation."""

    def test_get_cost_per_hour_a100_40gb(self, vultr_provider):
        """Test A100 40GB cost."""
        assert vultr_provider.get_cost_per_hour(GPUType.A100_40GB) == 0.62

    def test_get_cost_per_hour_a100_80gb(self, vultr_provider):
        """Test A100 80GB cost."""
        assert vultr_provider.get_cost_per_hour(GPUType.A100_80GB) == 2.48

    def test_get_cost_per_hour_h100(self, vultr_provider):
        """Test H100 cost."""
        assert vultr_provider.get_cost_per_hour(GPUType.H100_80GB) == 3.99

    def test_get_cost_per_hour_unknown(self, vultr_provider):
        """Test unknown GPU returns 0."""
        assert vultr_provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.0

    def test_get_cost_per_hour_unsupported(self, vultr_provider):
        """Test unsupported GPU returns 0."""
        assert vultr_provider.get_cost_per_hour(GPUType.RTX_4090) == 0.0


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestVultrProviderHealthCheck:
    """Test VultrProvider.health_check()."""

    def test_health_check_healthy(self, vultr_provider):
        """Test health check when CLI available and configured."""
        with patch.object(Path, "exists", return_value=True):
            result = vultr_provider.health_check()

            assert result.healthy is True
            assert "CLI available" in result.message
            assert result.details["configured"] is True

    def test_health_check_no_cli(self, vultr_provider_no_cli):
        """Test health check when CLI missing."""
        result = vultr_provider_no_cli.health_check()

        assert result.healthy is False
        assert "not found" in result.message
        assert result.details["cli_path"] is None

    def test_health_check_no_config(self, vultr_provider):
        """Test health check when config file missing."""
        with patch.object(Path, "exists", return_value=False):
            result = vultr_provider.health_check()

            assert result.healthy is False
            assert "config file not found" in result.message
            assert result.details["configured"] is False

    def test_health_check_status_values(self, vultr_provider, vultr_provider_no_cli):
        """Test health check returns correct status values."""
        from app.coordination.protocols import CoordinatorStatus

        # Healthy case
        with patch.object(Path, "exists", return_value=True):
            result = vultr_provider.health_check()
            assert result.status == CoordinatorStatus.RUNNING

        # No CLI case
        result = vultr_provider_no_cli.health_check()
        assert result.status == CoordinatorStatus.ERROR

        # No config case
        with patch.object(Path, "exists", return_value=False):
            result = vultr_provider.health_check()
            assert result.status == CoordinatorStatus.DEGRADED


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestVultrProviderConstants:
    """Test VultrProvider constants."""

    def test_vultr_gpu_plans_structure(self):
        """Test VULTR_GPU_PLANS has expected structure."""
        for plan_id, info in VULTR_GPU_PLANS.items():
            assert isinstance(plan_id, str)
            assert len(info) == 3
            assert isinstance(info[0], GPUType)
            assert isinstance(info[1], (int, float))  # gpu_memory_gb
            assert isinstance(info[2], float)  # cost_per_hour

    def test_vultr_gpu_plans_has_a100(self):
        """Test VULTR_GPU_PLANS includes A100 plans."""
        a100_plans = [p for p in VULTR_GPU_PLANS if "a100" in p.lower()]
        assert len(a100_plans) >= 1

    def test_vultr_gpu_plans_has_h100(self):
        """Test VULTR_GPU_PLANS includes H100 plans."""
        h100_plans = [p for p in VULTR_GPU_PLANS if "h100" in p.lower()]
        assert len(h100_plans) >= 1

    def test_default_ssh_key_name(self):
        """Test default SSH key name."""
        assert DEFAULT_SSH_KEY == "ringrift-cluster"
