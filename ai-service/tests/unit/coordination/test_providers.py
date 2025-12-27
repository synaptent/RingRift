"""Tests for cloud provider abstraction (app/coordination/providers/).

December 2025: Added as part of test coverage initiative for provider abstraction
(5 modules, ~1,400 LOC).
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)


# =============================================================================
# ProviderType Enum Tests
# =============================================================================


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_all_providers_defined(self):
        """All expected providers are defined."""
        assert ProviderType.LAMBDA is not None
        assert ProviderType.VULTR is not None
        assert ProviderType.VAST is not None
        assert ProviderType.HETZNER is not None
        assert ProviderType.RUNPOD is not None

    def test_provider_count(self):
        """Correct number of providers."""
        assert len(ProviderType) == 5


# =============================================================================
# GPUType Enum Tests
# =============================================================================


class TestGPUType:
    """Tests for GPUType enum and parsing."""

    def test_all_gpu_types_defined(self):
        """All expected GPU types are defined."""
        # Consumer
        assert GPUType.RTX_3090 is not None
        assert GPUType.RTX_4090 is not None
        assert GPUType.RTX_5090 is not None
        # Data center
        assert GPUType.A10 is not None
        assert GPUType.A100_40GB is not None
        assert GPUType.A100_80GB is not None
        assert GPUType.H100_80GB is not None
        assert GPUType.GH200_96GB is not None
        # Other
        assert GPUType.CPU_ONLY is not None
        assert GPUType.UNKNOWN is not None

    def test_from_string_gh200(self):
        """Parse GH200 variants."""
        assert GPUType.from_string("GH200 96GB") == GPUType.GH200_96GB
        assert GPUType.from_string("NVIDIA GH200") == GPUType.GH200_96GB
        assert GPUType.from_string("gh200") == GPUType.GH200_96GB

    def test_from_string_h100(self):
        """Parse H100 variants."""
        assert GPUType.from_string("H100 80GB") == GPUType.H100_80GB
        assert GPUType.from_string("NVIDIA H100 PCIe") == GPUType.H100_80GB
        assert GPUType.from_string("h100") == GPUType.H100_80GB

    def test_from_string_a100(self):
        """Parse A100 variants."""
        assert GPUType.from_string("A100 80GB PCIe") == GPUType.A100_80GB
        assert GPUType.from_string("NVIDIA A100-80") == GPUType.A100_80GB
        assert GPUType.from_string("A100 40GB SXM") == GPUType.A100_40GB
        assert GPUType.from_string("a100") == GPUType.A100_40GB  # Default to 40GB

    def test_from_string_a10(self):
        """Parse A10 (not A100)."""
        assert GPUType.from_string("A10 24GB") == GPUType.A10
        assert GPUType.from_string("NVIDIA A10") == GPUType.A10

    def test_from_string_rtx(self):
        """Parse RTX variants."""
        assert GPUType.from_string("RTX 5090") == GPUType.RTX_5090
        assert GPUType.from_string("GeForce RTX 4090") == GPUType.RTX_4090
        assert GPUType.from_string("NVIDIA RTX 3090 Ti") == GPUType.RTX_3090

    def test_from_string_unknown(self):
        """Unknown GPU types return UNKNOWN."""
        assert GPUType.from_string("Unknown GPU") == GPUType.UNKNOWN
        assert GPUType.from_string("Tesla V100") == GPUType.UNKNOWN
        assert GPUType.from_string("") == GPUType.UNKNOWN


# =============================================================================
# InstanceStatus Enum Tests
# =============================================================================


class TestInstanceStatus:
    """Tests for InstanceStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses are defined."""
        assert InstanceStatus.PENDING is not None
        assert InstanceStatus.STARTING is not None
        assert InstanceStatus.RUNNING is not None
        assert InstanceStatus.STOPPING is not None
        assert InstanceStatus.STOPPED is not None
        assert InstanceStatus.TERMINATED is not None
        assert InstanceStatus.ERROR is not None
        assert InstanceStatus.UNKNOWN is not None


# =============================================================================
# Instance Dataclass Tests
# =============================================================================


class TestInstance:
    """Tests for Instance dataclass."""

    def test_minimal_instance(self):
        """Create instance with minimal required fields."""
        inst = Instance(
            id="i-123",
            provider=ProviderType.VAST,
            name="test-instance",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
        )
        assert inst.id == "i-123"
        assert inst.provider == ProviderType.VAST
        assert inst.status == InstanceStatus.RUNNING
        assert inst.gpu_count == 1  # Default

    def test_full_instance(self):
        """Create instance with all fields."""
        now = datetime.now()
        inst = Instance(
            id="i-456",
            provider=ProviderType.LAMBDA,
            name="gpu-node",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.H100_80GB,
            gpu_count=8,
            gpu_memory_gb=80.0,
            ip_address="192.168.1.100",
            ssh_port=22,
            ssh_user="ubuntu",
            created_at=now,
            cost_per_hour=2.50,
            region="us-east-1",
            tags={"project": "ringrift"},
            raw_data={"custom": "data"},
        )
        assert inst.gpu_count == 8
        assert inst.cost_per_hour == 2.50
        assert inst.tags["project"] == "ringrift"

    def test_is_running_property(self):
        """is_running requires RUNNING status AND IP address."""
        # Running with IP
        inst = Instance(
            id="i-1",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
            ip_address="1.2.3.4",
        )
        assert inst.is_running is True

        # Running without IP
        inst2 = Instance(
            id="i-2",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
        )
        assert inst2.is_running is False

        # Stopped with IP
        inst3 = Instance(
            id="i-3",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.STOPPED,
            gpu_type=GPUType.RTX_4090,
            ip_address="1.2.3.4",
        )
        assert inst3.is_running is False

    def test_ssh_host_property(self):
        """ssh_host returns user@ip format."""
        inst = Instance(
            id="i-1",
            provider=ProviderType.LAMBDA,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.H100_80GB,
            ip_address="10.0.0.1",
            ssh_user="ubuntu",
        )
        assert inst.ssh_host == "ubuntu@10.0.0.1"

    def test_ssh_host_no_ip(self):
        """ssh_host returns empty string if no IP."""
        inst = Instance(
            id="i-1",
            provider=ProviderType.LAMBDA,
            name="test",
            status=InstanceStatus.PENDING,
            gpu_type=GPUType.H100_80GB,
        )
        assert inst.ssh_host == ""


# =============================================================================
# CloudProvider Abstract Base Tests
# =============================================================================


class MockProvider(CloudProvider):
    """Mock provider for testing base class behavior."""

    def __init__(self, instances: list[Instance] | None = None):
        self._instances = instances or []

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VAST

    @property
    def name(self) -> str:
        return "Mock Provider"

    def is_configured(self) -> bool:
        return True

    async def list_instances(self) -> list[Instance]:
        return self._instances

    async def get_instance(self, instance_id: str) -> Instance | None:
        for inst in self._instances:
            if inst.id == instance_id:
                return inst
        return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        inst = await self.get_instance(instance_id)
        return inst.status if inst else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        return []

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        return {iid: True for iid in instance_ids}

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        costs = {
            GPUType.RTX_4090: 0.50,
            GPUType.H100_80GB: 2.50,
        }
        return costs.get(gpu_type, 0.0)


class TestCloudProvider:
    """Tests for CloudProvider abstract base class."""

    @pytest.fixture
    def provider_with_instances(self) -> MockProvider:
        """Create mock provider with test instances."""
        return MockProvider([
            Instance(
                id="i-1",
                provider=ProviderType.VAST,
                name="node-1",
                status=InstanceStatus.RUNNING,
                gpu_type=GPUType.RTX_4090,
                gpu_count=1,
                ip_address="1.1.1.1",
                cost_per_hour=0.50,
            ),
            Instance(
                id="i-2",
                provider=ProviderType.VAST,
                name="node-2",
                status=InstanceStatus.RUNNING,
                gpu_type=GPUType.RTX_4090,
                gpu_count=2,
                ip_address="2.2.2.2",
                cost_per_hour=1.00,
            ),
            Instance(
                id="i-3",
                provider=ProviderType.VAST,
                name="node-3",
                status=InstanceStatus.STOPPED,
                gpu_type=GPUType.H100_80GB,
                gpu_count=1,
                cost_per_hour=0.0,  # Stopped = no cost
            ),
        ])

    @pytest.mark.asyncio
    async def test_get_available_gpus(self, provider_with_instances):
        """get_available_gpus counts only running instances."""
        gpus = await provider_with_instances.get_available_gpus()
        # 1 + 2 = 3 RTX 4090s from running instances
        assert gpus[GPUType.RTX_4090] == 3
        # H100 is stopped, shouldn't be counted
        assert GPUType.H100_80GB not in gpus

    @pytest.mark.asyncio
    async def test_get_total_cost_per_hour(self, provider_with_instances):
        """get_total_cost_per_hour sums running instance costs."""
        cost = await provider_with_instances.get_total_cost_per_hour()
        # 0.50 + 1.00 = 1.50 (stopped instance not counted)
        assert cost == 1.50

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, provider_with_instances):
        """health_check returns False for non-running instances."""
        stopped_inst = await provider_with_instances.get_instance("i-3")
        assert stopped_inst is not None
        result = await provider_with_instances.health_check(stopped_inst)
        assert result is False


# =============================================================================
# Provider-Specific Tests (imported modules)
# =============================================================================


class TestVastProvider:
    """Tests for Vast.ai provider implementation."""

    def test_import(self):
        """VastProvider can be imported."""
        from app.coordination.providers.vast_provider import VastProvider
        assert VastProvider is not None

    def test_instantiation(self):
        """VastProvider can be instantiated."""
        from app.coordination.providers.vast_provider import VastProvider
        provider = VastProvider()
        assert provider.provider_type == ProviderType.VAST
        assert provider.name == "Vast.ai"


class TestLambdaProvider:
    """Tests for Lambda Labs provider implementation."""

    def test_import(self):
        """LambdaProvider can be imported."""
        from app.coordination.providers.lambda_provider import LambdaProvider
        assert LambdaProvider is not None

    def test_instantiation(self):
        """LambdaProvider can be instantiated."""
        from app.coordination.providers.lambda_provider import LambdaProvider
        provider = LambdaProvider()
        assert provider.provider_type == ProviderType.LAMBDA
        assert provider.name == "Lambda"  # Note: Just "Lambda", not "Lambda Labs"


class TestVultrProvider:
    """Tests for Vultr provider implementation."""

    def test_import(self):
        """VultrProvider can be imported."""
        from app.coordination.providers.vultr_provider import VultrProvider
        assert VultrProvider is not None

    def test_instantiation(self):
        """VultrProvider can be instantiated."""
        from app.coordination.providers.vultr_provider import VultrProvider
        provider = VultrProvider()
        assert provider.provider_type == ProviderType.VULTR
        assert provider.name == "Vultr"


class TestHetznerProvider:
    """Tests for Hetzner provider implementation."""

    def test_import(self):
        """HetznerProvider can be imported."""
        from app.coordination.providers.hetzner_provider import HetznerProvider
        assert HetznerProvider is not None

    def test_instantiation(self):
        """HetznerProvider can be instantiated."""
        from app.coordination.providers.hetzner_provider import HetznerProvider
        provider = HetznerProvider()
        assert provider.provider_type == ProviderType.HETZNER
        assert provider.name == "Hetzner"


# =============================================================================
# Provider __init__.py Re-exports Tests
# =============================================================================


class TestProviderExports:
    """Tests for provider module re-exports."""

    def test_base_exports_available(self):
        """Base types are available from __init__."""
        from app.coordination.providers import (
            CloudProvider,
            GPUType,
            Instance,
            InstanceStatus,
            ProviderType,
        )
        # All base types should be importable
        assert CloudProvider is not None
        assert GPUType is not None
        assert Instance is not None
        assert InstanceStatus is not None
        assert ProviderType is not None

    def test_get_provider_function(self):
        """get_provider() function is available."""
        from app.coordination.providers import get_provider, ProviderType

        # Get Vast provider via get_provider
        provider = get_provider(ProviderType.VAST)
        assert provider is not None
        assert provider.provider_type == ProviderType.VAST

    def test_get_all_providers_function(self):
        """get_all_providers() function returns list."""
        from app.coordination.providers import get_all_providers

        providers = get_all_providers()
        # Returns list (may be empty if providers not configured)
        assert isinstance(providers, list)

    def test_reset_providers_function(self):
        """reset_providers() clears the cache."""
        from app.coordination.providers import get_provider, reset_providers, ProviderType

        # Get a provider to populate cache
        p1 = get_provider(ProviderType.VAST)

        # Reset and get again
        reset_providers()
        p2 = get_provider(ProviderType.VAST)

        # Should be different instances after reset
        assert p1 is not p2
