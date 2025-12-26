"""Hetzner cloud provider implementation.

Hetzner provides CPU-only cloud servers at low cost.
Uses hcloud CLI or REST API for instance management.

Configuration:
    Environment variable: HCLOUD_TOKEN
    Or config file: ~/.config/hcloud/cli.toml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)

# Hetzner server types (CPU only)
HETZNER_PLANS = {
    # type: (vCPUs, RAM GB, cost_per_hour)
    "cx22": (2, 4, 0.007),
    "cx32": (4, 8, 0.014),
    "cx42": (8, 16, 0.028),
    "cx52": (16, 32, 0.056),
    "ccx13": (2, 8, 0.014),  # Dedicated CPU
    "ccx23": (4, 16, 0.028),
    "ccx33": (8, 32, 0.056),
    "ccx43": (16, 64, 0.112),
    "ccx53": (32, 128, 0.224),
    "ccx63": (48, 192, 0.336),
}


class HetznerProvider(CloudProvider):
    """Hetzner cloud provider (CPU only)."""

    def __init__(self, token: str | None = None):
        """Initialize Hetzner provider.

        Args:
            token: Hetzner API token (reads from HCLOUD_TOKEN if not specified)
        """
        self._token = token or os.environ.get("HCLOUD_TOKEN")
        self._cli_path = shutil.which("hcloud")

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.HETZNER

    @property
    def name(self) -> str:
        return "Hetzner"

    def is_configured(self) -> bool:
        """Check if hcloud CLI is available and configured."""
        return bool(self._token) or bool(self._cli_path)

    async def _run_cli(self, *args: str) -> tuple[str, str, int]:
        """Run hcloud CLI command."""
        if not self._cli_path:
            raise RuntimeError("hcloud CLI not found")

        cmd = [self._cli_path] + list(args) + ["-o", "json"]

        env = os.environ.copy()
        if self._token:
            env["HCLOUD_TOKEN"] = self._token

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, env=env
            )
        )

        return result.stdout, result.stderr, result.returncode

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Hetzner server JSON into Instance object."""
        status_map = {
            "initializing": InstanceStatus.PENDING,
            "starting": InstanceStatus.STARTING,
            "running": InstanceStatus.RUNNING,
            "stopping": InstanceStatus.STOPPING,
            "off": InstanceStatus.STOPPED,
            "deleting": InstanceStatus.TERMINATED,
        }
        status = status_map.get(data.get("status", ""), InstanceStatus.UNKNOWN)

        # Get server type info
        server_type = data.get("server_type", {}).get("name", "")
        plan_info = HETZNER_PLANS.get(server_type, (1, 1, 0.01))

        # Get IP address
        ip = None
        public_net = data.get("public_net", {})
        if public_net.get("ipv4"):
            ip = public_net["ipv4"].get("ip")

        return Instance(
            id=str(data.get("id", "")),
            provider=ProviderType.HETZNER,
            name=data.get("name", ""),
            status=status,
            gpu_type=GPUType.CPU_ONLY,
            gpu_count=0,
            gpu_memory_gb=0,
            ip_address=ip,
            ssh_port=22,
            ssh_user="root",
            created_at=datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
            if data.get("created") else None,
            cost_per_hour=plan_info[2],
            region=data.get("datacenter", {}).get("name", ""),
            tags={label: "true" for label in data.get("labels", {})},
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Hetzner servers."""
        stdout, stderr, rc = await self._run_cli("server", "list")

        if rc != 0:
            logger.error(f"hcloud server list failed: {stderr}")
            return []

        try:
            servers = json.loads(stdout)
            return [self._parse_instance(srv) for srv in servers]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hcloud output: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get specific server by ID."""
        stdout, stderr, rc = await self._run_cli("server", "describe", instance_id)

        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            return self._parse_instance(data)
        except json.JSONDecodeError:
            return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get server status."""
        instance = await self.get_instance(instance_id)
        return instance.status if instance else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new CPU servers (GPU type ignored - Hetzner is CPU only)."""
        # Use a reasonable default server type for training
        server_type = "ccx33"  # 8 dedicated vCPUs, 32GB RAM

        created = []
        for i in range(count):
            name = f"{name_prefix}-cpu-{i}"

            args = [
                "server", "create",
                "--type", server_type,
                "--image", "ubuntu-22.04",
                "--name", name,
            ]

            if region:
                args.extend(["--datacenter", region])
            else:
                args.extend(["--location", "fsn1"])  # Falkenstein

            stdout, stderr, rc = await self._run_cli(*args)

            if rc != 0:
                logger.error(f"Failed to create Hetzner server: {stderr}")
                continue

            try:
                data = json.loads(stdout)
                server_data = data.get("server", data)
                instance = self._parse_instance(server_data)
                created.append(instance)
                logger.info(f"Created Hetzner server: {instance.id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse create response: {e}")

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Delete servers."""
        results = {}

        for instance_id in instance_ids:
            stdout, stderr, rc = await self._run_cli("server", "delete", instance_id)
            success = rc == 0

            if success:
                logger.info(f"Deleted Hetzner server: {instance_id}")
            else:
                logger.error(f"Failed to delete {instance_id}: {stderr}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost (CPU only, uses default server type)."""
        # Hetzner doesn't have GPUs, return CPU server cost
        return HETZNER_PLANS.get("ccx33", (0, 0, 0.056))[2]

    async def get_available_gpus(self) -> dict[GPUType, int]:
        """Hetzner has no GPUs."""
        return {GPUType.CPU_ONLY: 100}  # Effectively unlimited CPU
