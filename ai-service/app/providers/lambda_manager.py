"""Lambda Labs cloud provider manager.

DEPRECATED (December 2025): Lambda Labs account permanently terminated.
This module is retained for historical reference only. All Lambda nodes
have been removed from the cluster. Use Vast.ai, RunPod, or Nebius instead.

Original description:
Manages Lambda Labs GPU instances via their REST API.
Supports GH200, H100, A10, and other GPU types.

API Documentation: https://cloud.lambdalabs.com/api/v1
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Lambda Labs account terminated Dec 2025. "
    "lambda_manager.py is deprecated. Use other providers (Vast, RunPod, Nebius).",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml

from app.providers.base import (
    HealthCheckResult,
    InstanceState,
    Provider,
    ProviderInstance,
    ProviderManager,
)

logger = logging.getLogger(__name__)


# API endpoints
LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


@dataclass
class LambdaInstanceType:
    """Lambda instance type configuration."""

    name: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_count: int
    memory_gb: int
    hourly_cost: float


# Known Lambda instance types
LAMBDA_INSTANCE_TYPES = {
    "gpu_1x_gh200": LambdaInstanceType(
        name="gpu_1x_gh200",
        gpu_type="GH200",
        gpu_count=1,
        gpu_memory_gb=96,
        cpu_count=72,
        memory_gb=480,
        hourly_cost=2.49,
    ),
    "gpu_1x_h100_pcie": LambdaInstanceType(
        name="gpu_1x_h100_pcie",
        gpu_type="H100 PCIe",
        gpu_count=1,
        gpu_memory_gb=80,
        cpu_count=26,
        memory_gb=200,
        hourly_cost=2.49,
    ),
    "gpu_2x_h100_sxm5": LambdaInstanceType(
        name="gpu_2x_h100_sxm5",
        gpu_type="H100 SXM5",
        gpu_count=2,
        gpu_memory_gb=160,
        cpu_count=52,
        memory_gb=400,
        hourly_cost=4.98,
    ),
    "gpu_1x_a10": LambdaInstanceType(
        name="gpu_1x_a10",
        gpu_type="A10",
        gpu_count=1,
        gpu_memory_gb=24,
        cpu_count=30,
        memory_gb=200,
        hourly_cost=0.75,
    ),
}


def _load_api_key() -> str | None:
    """Load Lambda API key from environment or config file."""
    # Try environment variable first
    if api_key := os.getenv("LAMBDA_API_KEY"):
        return api_key

    # Try config file
    config_path = Path.home() / ".lambda_cloud" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("api_key")
        except Exception as e:
            logger.warning(f"Failed to load Lambda config: {e}")

    return None


def _parse_instance_state(api_state: str) -> InstanceState:
    """Convert Lambda API state to unified state."""
    state_map = {
        "active": InstanceState.RUNNING,
        "booting": InstanceState.STARTING,
        "unhealthy": InstanceState.ERROR,
        "terminated": InstanceState.TERMINATED,
    }
    return state_map.get(api_state.lower(), InstanceState.UNKNOWN)


class LambdaManager(ProviderManager):
    """Manage Lambda Labs instances via API.

    Usage:
        manager = LambdaManager()
        instances = await manager.list_instances()

        for inst in instances:
            health = await manager.check_health(inst)
            print(f"{inst.name}: {health.message}")
    """

    provider = Provider.LAMBDA

    def __init__(self, api_key: str | None = None):
        """Initialize Lambda manager.

        Args:
            api_key: Lambda API key. If not provided, loads from
                     LAMBDA_API_KEY env var or ~/.lambda_cloud/config.yaml
        """
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            logger.warning("Lambda API key not configured")

        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            auth = aiohttp.BasicAuth(self.api_key, "") if self.api_key else None
            self._session = aiohttp.ClientSession(auth=auth)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict[str, Any] | None:
        """Make API request to Lambda."""
        if not self.api_key:
            logger.error("Lambda API key not configured")
            return None

        session = await self._get_session()
        url = f"{LAMBDA_API_BASE}{endpoint}"

        try:
            async with session.request(method, url, json=data) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"Lambda API error {resp.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"Lambda API request failed: {e}")
            return None

    async def list_instances(self) -> list[ProviderInstance]:
        """List all Lambda instances."""
        result = await self._api_request("GET", "/instances")
        if not result:
            return []

        instances = []
        for inst_data in result.get("data", []):
            instance = self._parse_instance(inst_data)
            if instance:
                instances.append(instance)

        logger.info(f"Lambda: found {len(instances)} instances")
        return instances

    async def get_instance(self, instance_id: str) -> ProviderInstance | None:
        """Get details of a specific instance."""
        result = await self._api_request("GET", f"/instances/{instance_id}")
        if not result:
            return None

        return self._parse_instance(result.get("data", {}))

    def _parse_instance(self, data: dict[str, Any]) -> ProviderInstance | None:
        """Parse API response into ProviderInstance."""
        try:
            instance_id = data.get("id", "")
            name = data.get("name", instance_id)
            instance_type = data.get("instance_type", {})
            type_name = instance_type.get("name", "")

            # Get type info
            type_info = LAMBDA_INSTANCE_TYPES.get(type_name)

            # Parse IP addresses
            ip = data.get("ip", "")

            return ProviderInstance(
                instance_id=instance_id,
                provider=Provider.LAMBDA,
                name=name,
                public_ip=ip if ip else None,
                state=_parse_instance_state(data.get("status", "unknown")),
                gpu_type=type_info.gpu_type if type_info else instance_type.get("description", ""),
                gpu_count=type_info.gpu_count if type_info else 1,
                gpu_memory_gb=type_info.gpu_memory_gb if type_info else 0,
                cpu_count=type_info.cpu_count if type_info else 0,
                memory_gb=type_info.memory_gb if type_info else 0,
                hourly_cost=type_info.hourly_cost if type_info else 0.0,
                metadata={
                    "ssh_user": "ubuntu",
                    "ssh_key": "~/.ssh/id_cluster",
                    "instance_type": type_name,
                    "region": data.get("region", {}).get("name", ""),
                    "hostname": data.get("hostname", ""),
                },
            )
        except Exception as e:
            logger.error(f"Failed to parse Lambda instance: {e}")
            return None

    async def check_health(self, instance: ProviderInstance) -> HealthCheckResult:
        """Check health of a Lambda instance."""
        # First check SSH
        ssh_result = await self.check_ssh_connectivity(instance)
        if not ssh_result.healthy:
            return ssh_result

        # Then check P2P daemon
        p2p_result = await self.check_p2p_health(instance)
        if not p2p_result.healthy:
            return p2p_result

        # Then check Tailscale
        ts_result = await self.check_tailscale(instance)

        # Aggregate results
        all_healthy = ssh_result.healthy and p2p_result.healthy and ts_result.healthy
        return HealthCheckResult(
            healthy=all_healthy,
            check_type="composite",
            message="All checks passed" if all_healthy else "Some checks failed",
            details={
                "ssh": ssh_result.healthy,
                "p2p": p2p_result.healthy,
                "tailscale": ts_result.healthy,
            },
        )

    async def reboot_instance(self, instance_id: str) -> bool:
        """Reboot a Lambda instance via API."""
        result = await self._api_request(
            "POST",
            "/instance-operations/restart",
            {"instance_ids": [instance_id]},
        )
        if result:
            logger.info(f"Lambda: rebooting instance {instance_id}")
            return True
        return False

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Lambda instance via API."""
        result = await self._api_request(
            "POST",
            "/instance-operations/terminate",
            {"instance_ids": [instance_id]},
        )
        if result:
            logger.info(f"Lambda: terminated instance {instance_id}")
            return True
        return False

    async def launch_instance(self, config: dict[str, Any]) -> str | None:
        """Launch a new Lambda instance.

        Args:
            config: Launch configuration with keys:
                - instance_type_name: e.g., "gpu_1x_gh200"
                - region_name: e.g., "us-west-1"
                - ssh_key_names: list of SSH key names
                - name: optional instance name
                - file_system_names: optional list of filesystem names

        Returns:
            Instance ID if successful, None otherwise.
        """
        launch_data = {
            "instance_type_name": config.get("instance_type_name", "gpu_1x_gh200"),
            "region_name": config.get("region_name", "us-west-1"),
            "ssh_key_names": config.get("ssh_key_names", []),
        }

        if name := config.get("name"):
            launch_data["name"] = name
        if fs_names := config.get("file_system_names"):
            launch_data["file_system_names"] = fs_names

        result = await self._api_request("POST", "/instance-operations/launch", launch_data)
        if result and result.get("data", {}).get("instance_ids"):
            instance_id = result["data"]["instance_ids"][0]
            logger.info(f"Lambda: launched instance {instance_id}")
            return instance_id
        return None

    async def get_available_instance_types(self) -> list[dict[str, Any]]:
        """Get list of available instance types."""
        result = await self._api_request("GET", "/instance-types")
        if not result:
            return []
        return result.get("data", {})

    async def get_ssh_keys(self) -> list[dict[str, Any]]:
        """Get list of registered SSH keys."""
        result = await self._api_request("GET", "/ssh-keys")
        if not result:
            return []
        return result.get("data", [])

    async def restart_p2p_daemon(self, instance: ProviderInstance) -> bool:
        """Restart P2P daemon on instance via SSH."""
        cmd = """
cd ~/ringrift/ai-service && \
pkill -f 'app.p2p.orchestrator' 2>/dev/null; \
sleep 2; \
source venv/bin/activate && \
nohup python -m app.p2p.orchestrator > logs/p2p.log 2>&1 &
echo "P2P daemon restarted"
"""
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0 and "restarted" in stdout:
            logger.info(f"Lambda: restarted P2P on {instance.name}")
            return True
        logger.error(f"Lambda: failed to restart P2P on {instance.name}: {stderr}")
        return False

    async def restart_tailscale(self, instance: ProviderInstance) -> bool:
        """Restart Tailscale daemon on instance."""
        cmd = "sudo systemctl restart tailscaled && sleep 2 && tailscale status"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0:
            logger.info(f"Lambda: restarted Tailscale on {instance.name}")
            return True
        logger.error(f"Lambda: failed to restart Tailscale on {instance.name}: {stderr}")
        return False

    async def get_tailscale_ip(self, instance: ProviderInstance) -> str | None:
        """Get Tailscale IP of instance."""
        cmd = "tailscale ip -4 2>/dev/null"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=15)
        if code == 0 and stdout.strip():
            ip = stdout.strip().split("\n")[0]
            if ip.startswith("100."):
                return ip
        return None

    async def deploy_ssh_key(
        self,
        instance: ProviderInstance,
        public_key: str,
    ) -> bool:
        """Deploy SSH public key to instance."""
        # Escape the key for shell
        escaped_key = public_key.replace("'", "'\"'\"'")
        cmd = f"""
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo '{escaped_key}' >> ~/.ssh/authorized_keys
sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
echo "Key deployed"
"""
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0 and "deployed" in stdout:
            logger.info(f"Lambda: deployed SSH key to {instance.name}")
            return True
        logger.error(f"Lambda: failed to deploy SSH key to {instance.name}: {stderr}")
        return False


async def test_lambda_api():
    """Test Lambda API connectivity."""
    manager = LambdaManager()

    print("Testing Lambda API...")
    instances = await manager.list_instances()

    if instances:
        print(f"\nFound {len(instances)} instances:")
        for inst in instances:
            print(f"  {inst}")
    else:
        print("No instances found (or API error)")

    await manager.close()


if __name__ == "__main__":
    asyncio.run(test_lambda_api())
