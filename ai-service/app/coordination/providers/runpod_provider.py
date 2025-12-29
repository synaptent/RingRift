"""RunPod cloud provider integration.

Implements the CloudProvider interface for RunPod GPU cloud.
Uses GraphQL API for pod management.

API Documentation: https://docs.runpod.io/api/graphql

Created: Dec 28, 2025
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)

# RunPod GraphQL API endpoint
RUNPOD_API_URL = "https://api.runpod.io/graphql"


@dataclass
class RunPodConfig:
    """Configuration for RunPod provider."""

    api_key: str | None = None
    api_url: str = RUNPOD_API_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "RunPodConfig":
        """Load configuration from environment variables.

        Checks RUNPOD_API_KEY env var first, then RunPod config files.
        """
        api_key = os.environ.get("RUNPOD_API_KEY")

        # Check config files if not in env
        if not api_key:
            config_files = [
                os.path.expanduser("~/.runpod/config.toml"),
                os.path.expanduser("~/.config/runpod/config.toml"),
            ]
            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        with open(config_file) as f:
                            for line in f:
                                if line.strip().startswith("apikey"):
                                    parts = line.split("=", 1)
                                    if len(parts) == 2:
                                        key = parts[1].strip().strip('"').strip("'")
                                        if key:
                                            api_key = key
                                            break
                        if api_key:
                            break
                    except OSError:
                        pass

        return cls(api_key=api_key)


# GPU type mappings for RunPod
RUNPOD_GPU_TYPES = {
    # Parse GPU display names to our types
    "NVIDIA H100 80GB HBM3": GPUType.H100_80GB,
    "NVIDIA H100 PCIe": GPUType.H100_80GB,
    "NVIDIA A100 80GB PCIe": GPUType.A100_80GB,
    "NVIDIA A100-SXM4-80GB": GPUType.A100_80GB,
    "NVIDIA A100-PCIE-40GB": GPUType.A100_40GB,
    "NVIDIA A100-SXM4-40GB": GPUType.A100_40GB,
    "NVIDIA A10": GPUType.A10,
    "NVIDIA L40S": GPUType.A10,  # Map L40S to A10 tier
    "NVIDIA GeForce RTX 4090": GPUType.RTX_4090,
    "NVIDIA GeForce RTX 3090": GPUType.RTX_3090,
    "NVIDIA GeForce RTX 3090 Ti": GPUType.RTX_3090,
}

# Cost per hour (approximate, check RunPod pricing for current rates)
RUNPOD_COSTS = {
    GPUType.H100_80GB: 3.89,
    GPUType.A100_80GB: 1.89,
    GPUType.A100_40GB: 1.19,
    GPUType.A10: 0.76,
    GPUType.RTX_4090: 0.69,
    GPUType.RTX_3090: 0.44,
}

# GraphQL query for listing pods
PODS_QUERY = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      runtime {
        uptimeInSeconds
        ports {
          ip
          isIpPublic
          privatePort
          publicPort
        }
        gpus {
          id
          gpuUtilPercent
          memoryUtilPercent
        }
      }
      machine {
        gpuDisplayName
        cpuCount
        memoryTotal
      }
    }
  }
}
"""

# GraphQL query for a specific pod
POD_QUERY = """
query getPod($podId: String!) {
  pod(input: { podId: $podId }) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      ports {
        ip
        isIpPublic
        privatePort
        publicPort
      }
      gpus {
        id
        gpuUtilPercent
        memoryUtilPercent
      }
    }
    machine {
      gpuDisplayName
      cpuCount
      memoryTotal
    }
  }
}
"""

# GraphQL mutation for creating a pod
CREATE_POD_MUTATION = """
mutation createPod(
  $name: String!,
  $gpuTypeId: String!,
  $cloudType: String!,
  $gpuCount: Int!,
  $volumeInGb: Int!,
  $containerDiskInGb: Int!,
  $dockerArgs: String,
  $imageName: String!
) {
  podFindAndDeployOnDemand(
    input: {
      name: $name,
      gpuTypeId: $gpuTypeId,
      cloudType: $cloudType,
      gpuCount: $gpuCount,
      volumeInGb: $volumeInGb,
      containerDiskInGb: $containerDiskInGb,
      dockerArgs: $dockerArgs,
      imageName: $imageName
    }
  ) {
    id
    name
    desiredStatus
  }
}
"""

# GraphQL mutation for stopping a pod
STOP_POD_MUTATION = """
mutation stopPod($podId: String!) {
  podStop(input: { podId: $podId }) {
    id
    desiredStatus
  }
}
"""

# GraphQL mutation for terminating a pod
TERMINATE_POD_MUTATION = """
mutation terminatePod($podId: String!) {
  podTerminate(input: { podId: $podId })
}
"""


class RunPodProvider(CloudProvider):
    """RunPod cloud provider implementation.

    Requires RUNPOD_API_KEY environment variable or config file.

    Example:
        provider = RunPodProvider()
        if provider.is_configured():
            instances = await provider.list_instances()
    """

    def __init__(self, config: RunPodConfig | None = None):
        self.config = config or RunPodConfig.from_env()
        self._session: Any = None  # aiohttp.ClientSession

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.RUNPOD

    @property
    def name(self) -> str:
        return "RunPod"

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.config.api_key)

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp

                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                )
            except ImportError:
                logger.warning("aiohttp not available, RunPod provider disabled")
                return None
        return self._session

    async def _graphql_request(
        self,
        query: str,
        variables: dict | None = None,
    ) -> dict[str, Any]:
        """Make a GraphQL API request."""
        if not self.config.api_key:
            raise ValueError("RunPod API key not configured")

        session = await self._get_session()
        if not session:
            raise ValueError("HTTP session not available")

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            async with session.post(self.config.api_url, json=payload) as resp:
                data = await resp.json()

                if resp.status != 200:
                    error_text = data.get("errors", str(data))
                    raise Exception(f"RunPod API error ({resp.status}): {error_text}")

                if "errors" in data:
                    raise Exception(f"RunPod GraphQL error: {data['errors']}")

                return data

        except Exception as e:
            if "aiohttp" in str(type(e).__module__):
                raise Exception(f"RunPod API request failed: {e}")
            raise

    def _parse_pod_status(self, desired_status: str) -> InstanceStatus:
        """Parse RunPod pod status to enum."""
        status_map = {
            "RUNNING": InstanceStatus.RUNNING,
            "STARTING": InstanceStatus.STARTING,
            "STOPPED": InstanceStatus.STOPPED,
            "TERMINATED": InstanceStatus.TERMINATED,
            "EXITED": InstanceStatus.STOPPED,
            "CREATED": InstanceStatus.STARTING,
        }
        return status_map.get(desired_status, InstanceStatus.UNKNOWN)

    def _parse_gpu_type(self, gpu_display_name: str) -> GPUType:
        """Parse RunPod GPU display name to GPU type."""
        if not gpu_display_name:
            return GPUType.UNKNOWN

        # Check exact matches first
        if gpu_display_name in RUNPOD_GPU_TYPES:
            return RUNPOD_GPU_TYPES[gpu_display_name]

        # Fall back to GPUType.from_string for partial matches
        return GPUType.from_string(gpu_display_name)

    def _extract_ssh_info(self, runtime: dict | None) -> tuple[str | None, int]:
        """Extract SSH host and port from runtime info.

        Returns:
            Tuple of (ip_address, ssh_port)
        """
        if not runtime:
            return None, 22

        ports = runtime.get("ports", [])
        for port in ports:
            if port.get("isIpPublic") and port.get("privatePort") == 22:
                return port.get("ip"), port.get("publicPort", 22)

        return None, 22

    def _pod_to_instance(self, pod: dict) -> Instance:
        """Convert a RunPod pod response to an Instance."""
        machine = pod.get("machine") or {}
        runtime = pod.get("runtime") or {}

        gpu_display_name = machine.get("gpuDisplayName", "")
        gpu_type = self._parse_gpu_type(gpu_display_name)

        ip_address, ssh_port = self._extract_ssh_info(runtime)

        return Instance(
            id=pod["id"],
            provider=ProviderType.RUNPOD,
            name=pod.get("name", pod["id"]),
            status=self._parse_pod_status(pod.get("desiredStatus", "UNKNOWN")),
            gpu_type=gpu_type,
            gpu_count=1,  # RunPod pods are typically single-GPU
            gpu_memory_gb=self._get_gpu_memory(gpu_type),
            ip_address=ip_address,
            ssh_port=ssh_port,
            ssh_user="root",
            cost_per_hour=self.get_cost_per_hour(gpu_type),
            raw_data=pod,
        )

    async def list_instances(self) -> list[Instance]:
        """List all RunPod pods."""
        if not self.is_configured():
            logger.warning("RunPod provider not configured")
            return []

        try:
            data = await self._graphql_request(PODS_QUERY)
            pods = data.get("data", {}).get("myself", {}).get("pods", [])

            instances = []
            for pod in pods:
                instance = self._pod_to_instance(pod)
                instances.append(instance)

            return instances

        except Exception as e:
            logger.error(f"Failed to list RunPod pods: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get a specific pod by ID."""
        if not self.is_configured():
            return None

        try:
            data = await self._graphql_request(
                POD_QUERY,
                variables={"podId": instance_id},
            )
            pod = data.get("data", {}).get("pod")

            if not pod:
                return None

            return self._pod_to_instance(pod)

        except Exception as e:
            logger.error(f"Failed to get RunPod pod {instance_id}: {e}")
            return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get current status of a pod."""
        instance = await self.get_instance(instance_id)
        return instance.status if instance else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new RunPod pods.

        Note: RunPod uses GPU type IDs instead of region for placement.
        The region parameter is ignored; RunPod auto-selects datacenter.
        """
        if not self.is_configured():
            raise ValueError("RunPod provider not configured")

        # Map GPU type to RunPod GPU ID
        gpu_id = self._get_runpod_gpu_id(gpu_type)
        if not gpu_id:
            raise ValueError(f"GPU type {gpu_type} not supported on RunPod")

        instances = []

        for i in range(count):
            try:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                name = f"{name_prefix}-{gpu_type.value}-{timestamp}-{i}"

                data = await self._graphql_request(
                    CREATE_POD_MUTATION,
                    variables={
                        "name": name,
                        "gpuTypeId": gpu_id,
                        "cloudType": "SECURE",  # or "COMMUNITY"
                        "gpuCount": 1,
                        "volumeInGb": 50,
                        "containerDiskInGb": 20,
                        "dockerArgs": "",
                        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                    },
                )

                pod = data.get("data", {}).get("podFindAndDeployOnDemand")
                if pod:
                    # Wait for pod to be ready
                    instance = await self._wait_for_instance(pod["id"], timeout=300)
                    if instance:
                        instances.append(instance)

            except Exception as e:
                logger.error(f"Failed to create RunPod pod: {e}")

        return instances

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate RunPod pods."""
        if not self.is_configured():
            return {inst_id: False for inst_id in instance_ids}

        results = {}

        for pod_id in instance_ids:
            try:
                await self._graphql_request(
                    TERMINATE_POD_MUTATION,
                    variables={"podId": pod_id},
                )
                results[pod_id] = True
                logger.info(f"Terminated RunPod pod: {pod_id}")

            except Exception as e:
                logger.error(f"Failed to terminate RunPod pod {pod_id}: {e}")
                results[pod_id] = False

        return results

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a RunPod pod (without terminating)."""
        if not self.is_configured():
            return False

        try:
            await self._graphql_request(
                STOP_POD_MUTATION,
                variables={"podId": instance_id},
            )
            logger.info(f"Stopped RunPod pod: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop RunPod pod {instance_id}: {e}")
            return False

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for a GPU type."""
        return RUNPOD_COSTS.get(gpu_type, 0.0)

    def _get_gpu_memory(self, gpu_type: GPUType) -> float:
        """Get GPU memory in GB."""
        memory_map = {
            GPUType.H100_80GB: 80.0,
            GPUType.A100_80GB: 80.0,
            GPUType.A100_40GB: 40.0,
            GPUType.A10: 24.0,
            GPUType.RTX_4090: 24.0,
            GPUType.RTX_3090: 24.0,
        }
        return memory_map.get(gpu_type, 0.0)

    def _get_runpod_gpu_id(self, gpu_type: GPUType) -> str | None:
        """Get RunPod GPU type ID for a GPU type."""
        # RunPod GPU type IDs (check RunPod API for current values)
        gpu_id_map = {
            GPUType.H100_80GB: "NVIDIA H100 80GB HBM3",
            GPUType.A100_80GB: "NVIDIA A100 80GB PCIe",
            GPUType.A100_40GB: "NVIDIA A100-PCIE-40GB",
            GPUType.A10: "NVIDIA A10",
            GPUType.RTX_4090: "NVIDIA GeForce RTX 4090",
            GPUType.RTX_3090: "NVIDIA GeForce RTX 3090",
        }
        return gpu_id_map.get(gpu_type)

    async def _wait_for_instance(
        self,
        instance_id: str,
        timeout: int = 300,
    ) -> Instance | None:
        """Wait for a pod to be ready."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            instance = await self.get_instance(instance_id)
            if instance and instance.status == InstanceStatus.RUNNING:
                return instance

            await asyncio.sleep(10)

        logger.warning(f"Timeout waiting for RunPod pod {instance_id}")
        return await self.get_instance(instance_id)

    async def get_gpu_availability(self) -> dict[str, dict]:
        """Get available GPU types and their stock status.

        Returns a dict mapping GPU type names to availability info.
        """
        if not self.is_configured():
            return {}

        query = """
        query {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
          }
        }
        """

        try:
            data = await self._graphql_request(query)
            gpu_types = data.get("data", {}).get("gpuTypes", [])

            availability = {}
            for gpu in gpu_types:
                availability[gpu["id"]] = {
                    "display_name": gpu.get("displayName", ""),
                    "memory_gb": gpu.get("memoryInGb", 0),
                    "secure_cloud": gpu.get("secureCloud", False),
                    "community_cloud": gpu.get("communityCloud", False),
                }

            return availability

        except Exception as e:
            logger.error(f"Failed to get RunPod GPU availability: {e}")
            return {}

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None


# Singleton instance
_runpod_provider: RunPodProvider | None = None


def get_runpod_provider() -> RunPodProvider:
    """Get the singleton RunPod provider instance."""
    global _runpod_provider
    if _runpod_provider is None:
        _runpod_provider = RunPodProvider()
    return _runpod_provider


def reset_runpod_provider() -> None:
    """Reset the singleton (for testing)."""
    global _runpod_provider
    _runpod_provider = None
