"""Base class and types for cloud provider abstraction.

This module defines the abstract interface that all cloud providers must implement,
plus common data types for instances, GPUs, and status.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class ProviderType(Enum):
    """Supported cloud provider types."""
    LAMBDA = auto()
    VULTR = auto()
    VAST = auto()
    HETZNER = auto()
    RUNPOD = auto()


class GPUType(Enum):
    """Common GPU types across providers."""
    # NVIDIA Consumer
    RTX_3090 = "rtx_3090"
    RTX_4090 = "rtx_4090"
    RTX_5090 = "rtx_5090"

    # NVIDIA Data Center
    A10 = "a10"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100_80GB = "h100_80gb"
    GH200_96GB = "gh200_96gb"

    # CPU only
    CPU_ONLY = "cpu"

    # Unknown
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> "GPUType":
        """Parse GPU type from provider-specific string."""
        s_lower = s.lower()
        if "gh200" in s_lower:
            return cls.GH200_96GB
        if "h100" in s_lower:
            return cls.H100_80GB
        if "a100" in s_lower:
            if "80" in s_lower:
                return cls.A100_80GB
            return cls.A100_40GB
        if "a10" in s_lower and "a100" not in s_lower:
            return cls.A10
        if "5090" in s_lower:
            return cls.RTX_5090
        if "4090" in s_lower:
            return cls.RTX_4090
        if "3090" in s_lower:
            return cls.RTX_3090
        return cls.UNKNOWN


class InstanceStatus(Enum):
    """Instance lifecycle status."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class Instance:
    """Represents a cloud instance across any provider."""
    id: str
    provider: ProviderType
    name: str
    status: InstanceStatus
    gpu_type: GPUType
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0
    ip_address: str | None = None
    ssh_port: int = 22
    ssh_user: str = "root"
    created_at: datetime | None = None
    cost_per_hour: float = 0.0
    region: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        """Check if instance is running and accessible."""
        return self.status == InstanceStatus.RUNNING and self.ip_address is not None

    @property
    def ssh_host(self) -> str:
        """Get SSH connection string."""
        if not self.ip_address:
            return ""
        return f"{self.ssh_user}@{self.ip_address}"


class CloudProvider(ABC):
    """Abstract base class for cloud provider integrations.

    Each provider implementation must handle:
    - Authentication (API keys, CLI tools)
    - Instance listing and status
    - Instance creation (scale up)
    - Instance termination (scale down)
    - Cost tracking
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type enum."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured (API keys, CLI, etc)."""
        ...

    @abstractmethod
    async def list_instances(self) -> list[Instance]:
        """List all instances from this provider.

        Returns:
            List of Instance objects
        """
        ...

    @abstractmethod
    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get a specific instance by ID.

        Args:
            instance_id: Provider-specific instance ID

        Returns:
            Instance if found, None otherwise
        """
        ...

    @abstractmethod
    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get current status of an instance.

        Args:
            instance_id: Provider-specific instance ID

        Returns:
            Current instance status
        """
        ...

    @abstractmethod
    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new instances.

        Args:
            gpu_type: Type of GPU required
            count: Number of instances to create
            region: Preferred region (provider-specific)
            name_prefix: Prefix for instance names

        Returns:
            List of created Instance objects
        """
        ...

    @abstractmethod
    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate instances.

        Args:
            instance_ids: List of instance IDs to terminate

        Returns:
            Dict mapping instance_id -> success boolean
        """
        ...

    @abstractmethod
    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for a GPU type.

        Args:
            gpu_type: Type of GPU

        Returns:
            Cost in USD per hour
        """
        ...

    async def get_available_gpus(self) -> dict[GPUType, int]:
        """Get available GPU capacity from this provider.

        Returns:
            Dict mapping GPU type -> available count
        """
        # Default: inspect running instances
        instances = await self.list_instances()
        gpu_counts: dict[GPUType, int] = {}
        for inst in instances:
            if inst.is_running:
                gpu_counts[inst.gpu_type] = gpu_counts.get(inst.gpu_type, 0) + inst.gpu_count
        return gpu_counts

    async def get_total_cost_per_hour(self) -> float:
        """Calculate total hourly cost of all running instances.

        Returns:
            Total cost in USD per hour
        """
        instances = await self.list_instances()
        return sum(inst.cost_per_hour for inst in instances if inst.is_running)

    async def health_check(self, instance: Instance) -> bool:
        """Check if an instance is healthy and responsive.

        Default implementation tries SSH connection.

        Args:
            instance: Instance to check

        Returns:
            True if healthy
        """
        if not instance.is_running:
            return False

        import asyncio
        import subprocess

        try:
            # Quick SSH check
            # Dec 2025: Use get_running_loop() in async context
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ssh", "-o", "ConnectTimeout=5",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes",
                        f"{instance.ssh_user}@{instance.ip_address}",
                        "echo ok"
                    ],
                    capture_output=True,
                    timeout=10
                )
            )
            return result.returncode == 0
        except Exception:
            return False
