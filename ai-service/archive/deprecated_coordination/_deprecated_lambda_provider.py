"""Lambda Labs cloud provider implementation.

DEPRECATED (December 2025): Lambda Labs account permanently terminated.
This module is retained for historical reference only. All Lambda nodes
have been removed from the cluster. Use Vast.ai, RunPod, or Nebius instead.

Original description:
Lambda provides GPU cloud instances with GH200, H100, A100, and A10 GPUs.
Uses the Lambda Cloud API for instance management.

Configuration:
    Environment variable: LAMBDA_API_KEY
    Or direct initialization with api_key parameter
"""

import warnings

warnings.warn(
    "Lambda Labs account terminated Dec 2025. "
    "lambda_provider.py is deprecated. Use VastProvider, RunPodProvider, or NebiusProvider.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)

# Lambda instance types (as of Dec 2025)
LAMBDA_INSTANCE_TYPES = {
    # type: (gpu_type, gpu_count, gpu_memory_gb, cost_per_hour)
    "gpu_1x_a10": (GPUType.A10, 1, 24, 0.75),
    "gpu_1x_a100": (GPUType.A100_40GB, 1, 40, 1.29),
    "gpu_1x_a100_sxm4": (GPUType.A100_80GB, 1, 80, 1.69),
    "gpu_1x_h100_pcie": (GPUType.H100_80GB, 1, 80, 2.49),
    "gpu_1x_gh200": (GPUType.GH200_96GB, 1, 96, 2.99),
    "gpu_2x_h100_sxm5": (GPUType.H100_80GB, 2, 160, 4.98),
    "gpu_8x_a100_80gb_sxm4": (GPUType.A100_80GB, 8, 640, 12.99),
    "gpu_8x_h100_sxm5": (GPUType.H100_80GB, 8, 640, 19.92),
}

API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"


class LambdaProvider(CloudProvider):
    """Lambda Labs cloud provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize Lambda provider.

        Args:
            api_key: Lambda API key (reads from LAMBDA_API_KEY if not specified)
        """
        self._api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        self._session = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LAMBDA

    @property
    def name(self) -> str:
        return "Lambda"

    def is_configured(self) -> bool:
        """Check if API key is available."""
        return bool(self._api_key)

    async def _get_session(self):
        """Get aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {self._api_key}"}
                )
            except ImportError:
                raise RuntimeError("aiohttp required for Lambda provider")
        return self._session

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request to Lambda Cloud."""
        session = await self._get_session()
        url = f"{API_BASE_URL}{endpoint}"

        async with session.request(method, url, json=data) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                logger.error(f"Lambda API error {resp.status}: {text}")
                return {"error": text}
            return await resp.json()

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Lambda instance JSON into Instance object."""
        status_map = {
            "active": InstanceStatus.RUNNING,
            "booting": InstanceStatus.STARTING,
            "unhealthy": InstanceStatus.ERROR,
            "terminated": InstanceStatus.TERMINATED,
        }
        status = status_map.get(data.get("status", ""), InstanceStatus.UNKNOWN)

        # Get instance type info
        instance_type = data.get("instance_type", {}).get("name", "")
        type_info = LAMBDA_INSTANCE_TYPES.get(
            instance_type,
            (GPUType.UNKNOWN, 1, 0, 0.0)
        )

        return Instance(
            id=data.get("id", ""),
            provider=ProviderType.LAMBDA,
            name=data.get("name", ""),
            status=status,
            gpu_type=type_info[0],
            gpu_count=type_info[1],
            gpu_memory_gb=type_info[2],
            ip_address=data.get("ip"),
            ssh_port=22,
            ssh_user="ubuntu",
            created_at=None,  # Lambda API doesn't expose creation time
            cost_per_hour=type_info[3],
            region=data.get("region", {}).get("name", ""),
            tags={},
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Lambda instances."""
        if not self._api_key:
            logger.warning("Lambda API key not configured")
            return []

        response = await self._api_request("GET", "/instances")

        if "error" in response:
            return []

        instances = response.get("data", [])
        return [self._parse_instance(inst) for inst in instances]

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get specific instance by ID."""
        response = await self._api_request("GET", f"/instances/{instance_id}")

        if "error" in response:
            return None

        data = response.get("data")
        return self._parse_instance(data) if data else None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get instance status."""
        instance = await self.get_instance(instance_id)
        return instance.status if instance else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new Lambda instances."""
        # Map GPU type to Lambda instance type
        type_map = {
            GPUType.A10: "gpu_1x_a10",
            GPUType.A100_40GB: "gpu_1x_a100",
            GPUType.A100_80GB: "gpu_1x_a100_sxm4",
            GPUType.H100_80GB: "gpu_1x_h100_pcie",
            GPUType.GH200_96GB: "gpu_1x_gh200",
        }

        instance_type = type_map.get(gpu_type)
        if not instance_type:
            logger.error(f"No Lambda instance type for GPU: {gpu_type}")
            return []

        # Check availability
        avail_response = await self._api_request("GET", "/instance-types")
        if "error" in avail_response:
            return []

        available = avail_response.get("data", {})
        type_info = available.get(instance_type, {})
        regions = type_info.get("regions_with_capacity_available", [])

        if not regions:
            logger.warning(f"No capacity for {instance_type}")
            return []

        # Use specified region or first available
        target_region = region
        if not target_region and regions:
            target_region = regions[0].get("name")

        created = []
        for i in range(count):
            name = f"{name_prefix}-{gpu_type.value}-{i}"

            response = await self._api_request("POST", "/instance-operations/launch", {
                "region_name": target_region,
                "instance_type_name": instance_type,
                "name": name,
                "ssh_key_names": ["ringrift-cluster"],  # Must exist in Lambda account
            })

            if "error" in response:
                continue

            instance_ids = response.get("data", {}).get("instance_ids", [])
            for inst_id in instance_ids:
                instance = await self.get_instance(inst_id)
                if instance:
                    created.append(instance)
                    logger.info(f"Created Lambda instance: {inst_id}")

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate Lambda instances."""
        results = {}

        for instance_id in instance_ids:
            response = await self._api_request(
                "POST",
                "/instance-operations/terminate",
                {"instance_ids": [instance_id]}
            )
            success = "error" not in response

            if success:
                logger.info(f"Terminated Lambda instance: {instance_id}")
            else:
                logger.error(f"Failed to terminate {instance_id}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for GPU type."""
        cost_map = {
            GPUType.A10: 0.75,
            GPUType.A100_40GB: 1.29,
            GPUType.A100_80GB: 1.69,
            GPUType.H100_80GB: 2.49,
            GPUType.GH200_96GB: 2.99,
        }
        return cost_map.get(gpu_type, 0.0)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
