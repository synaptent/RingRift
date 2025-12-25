"""Vultr cloud provider implementation.

Uses the vultr-cli command-line tool for instance management.
Requires vultr-cli to be installed and configured with API key.

Configuration:
    ~/.vultr-cli.yaml:
        api-key: YOUR_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)

# Vultr GPU instance plans (as of Dec 2025)
VULTR_GPU_PLANS = {
    # plan_id: (gpu_type, gpu_memory_gb, cost_per_hour)
    "vcg-a100-1c-6g-4vram": (GPUType.A100_40GB, 20, 0.62),  # A100 20GB
    "vcg-a100-2c-12g-8vram": (GPUType.A100_40GB, 40, 1.24),  # A100 40GB
    "vcg-a100-3c-24g-16vram": (GPUType.A100_80GB, 80, 2.48),  # A100 80GB
    "vcg-h100-1c-80g": (GPUType.H100_80GB, 80, 3.99),  # H100 80GB
}

# Default SSH key name (must be added to Vultr account)
DEFAULT_SSH_KEY = "ringrift-cluster"


class VultrProvider(CloudProvider):
    """Vultr cloud provider using vultr-cli."""

    def __init__(self, cli_path: str | None = None):
        """Initialize Vultr provider.

        Args:
            cli_path: Path to vultr-cli binary (auto-detected if not specified)
        """
        self._cli_path = cli_path or self._find_cli()
        self._ssh_key_id: str | None = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VULTR

    @property
    def name(self) -> str:
        return "Vultr"

    def _find_cli(self) -> str | None:
        """Find vultr-cli binary."""
        # Check common locations
        paths = [
            shutil.which("vultr-cli"),
            str(Path.home() / ".local/bin/vultr-cli"),
            "/usr/local/bin/vultr-cli",
        ]
        for path in paths:
            if path and Path(path).exists():
                return path
        return None

    def is_configured(self) -> bool:
        """Check if vultr-cli is available and configured."""
        if not self._cli_path:
            return False

        # Check if config file exists
        config_path = Path.home() / ".vultr-cli.yaml"
        return config_path.exists()

    async def _run_cli(self, *args: str) -> tuple[str, str, int]:
        """Run vultr-cli command.

        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        if not self._cli_path:
            raise RuntimeError("vultr-cli not found")

        cmd = [self._cli_path] + list(args) + ["--output", "json"]

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        )

        return result.stdout, result.stderr, result.returncode

    def _parse_instance(self, data: dict) -> Instance:
        """Parse Vultr instance JSON into Instance object."""
        # Parse status
        status_map = {
            "pending": InstanceStatus.PENDING,
            "active": InstanceStatus.RUNNING,
            "suspended": InstanceStatus.STOPPED,
            "resizing": InstanceStatus.STARTING,
        }
        status = status_map.get(data.get("status", ""), InstanceStatus.UNKNOWN)

        # Parse GPU type from plan
        plan = data.get("plan", "")
        gpu_info = VULTR_GPU_PLANS.get(plan, (GPUType.UNKNOWN, 0, 0.0))

        return Instance(
            id=data.get("id", ""),
            provider=ProviderType.VULTR,
            name=data.get("label", ""),
            status=status,
            gpu_type=gpu_info[0],
            gpu_count=1,
            gpu_memory_gb=gpu_info[1],
            ip_address=data.get("main_ip"),
            ssh_port=22,
            ssh_user="root",
            created_at=datetime.fromisoformat(data["date_created"].replace("Z", "+00:00"))
            if data.get("date_created") else None,
            cost_per_hour=gpu_info[2],
            region=data.get("region", ""),
            tags=data.get("tags", {}),
            raw_data=data,
        )

    async def list_instances(self) -> list[Instance]:
        """List all Vultr instances."""
        stdout, stderr, rc = await self._run_cli("instance", "list")

        if rc != 0:
            logger.error(f"vultr-cli instance list failed: {stderr}")
            return []

        try:
            data = json.loads(stdout)
            instances = data.get("instances", [])
            return [self._parse_instance(inst) for inst in instances]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vultr-cli output: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get specific instance by ID."""
        stdout, stderr, rc = await self._run_cli("instance", "get", instance_id)

        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            return self._parse_instance(data.get("instance", data))
        except (json.JSONDecodeError, KeyError):
            return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get instance status."""
        instance = await self.get_instance(instance_id)
        return instance.status if instance else InstanceStatus.UNKNOWN

    async def _get_ssh_key_id(self) -> str | None:
        """Get SSH key ID from Vultr account."""
        if self._ssh_key_id:
            return self._ssh_key_id

        stdout, stderr, rc = await self._run_cli("ssh-key", "list")
        if rc != 0:
            return None

        try:
            data = json.loads(stdout)
            keys = data.get("ssh_keys", [])
            for key in keys:
                if key.get("name") == DEFAULT_SSH_KEY:
                    self._ssh_key_id = key.get("id")
                    return self._ssh_key_id
        except json.JSONDecodeError:
            pass

        return None

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new GPU instances."""
        # Map GPU type to Vultr plan
        plan_map = {
            GPUType.A100_40GB: "vcg-a100-1c-6g-4vram",  # A100 20GB (cheapest)
            GPUType.A100_80GB: "vcg-a100-3c-24g-16vram",
            GPUType.H100_80GB: "vcg-h100-1c-80g",
        }

        plan = plan_map.get(gpu_type)
        if not plan:
            logger.error(f"No Vultr plan for GPU type: {gpu_type}")
            return []

        # Get SSH key
        ssh_key_id = await self._get_ssh_key_id()

        created = []
        for i in range(count):
            label = f"{name_prefix}-{gpu_type.value}-{i}"

            args = [
                "instance", "create",
                "--plan", plan,
                "--region", region or "ewr",  # Default to US East
                "--os", "2136",  # Ubuntu 22.04 x64
                "--label", label,
            ]

            if ssh_key_id:
                args.extend(["--ssh-keys", ssh_key_id])

            stdout, stderr, rc = await self._run_cli(*args)

            if rc != 0:
                logger.error(f"Failed to create instance: {stderr}")
                continue

            try:
                data = json.loads(stdout)
                instance = self._parse_instance(data.get("instance", data))
                created.append(instance)
                logger.info(f"Created Vultr instance: {instance.id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse create response: {e}")

        return created

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate instances."""
        results = {}

        for instance_id in instance_ids:
            stdout, stderr, rc = await self._run_cli("instance", "delete", instance_id)
            success = rc == 0

            if success:
                logger.info(f"Terminated Vultr instance: {instance_id}")
            else:
                logger.error(f"Failed to terminate {instance_id}: {stderr}")

            results[instance_id] = success

        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for GPU type."""
        cost_map = {
            GPUType.A100_40GB: 0.62,  # A100 20GB actually
            GPUType.A100_80GB: 2.48,
            GPUType.H100_80GB: 3.99,
        }
        return cost_map.get(gpu_type, 0.0)
