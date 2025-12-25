"""Hetzner Cloud provider manager.

Manages Hetzner Cloud servers via the hcloud CLI.
Primarily used for CPU-only selfplay instances.

Requires: hcloud CLI installed and configured
    https://github.com/hetznercloud/cli
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from app.providers.base import (
    HealthCheckResult,
    InstanceState,
    Provider,
    ProviderInstance,
    ProviderManager,
)

logger = logging.getLogger(__name__)


def _parse_server_state(api_state: str) -> InstanceState:
    """Convert Hetzner API state to unified state."""
    state_map = {
        "running": InstanceState.RUNNING,
        "starting": InstanceState.STARTING,
        "stopping": InstanceState.STOPPING,
        "off": InstanceState.STOPPED,
        "deleting": InstanceState.TERMINATED,
        "rebuilding": InstanceState.STARTING,
        "migrating": InstanceState.STARTING,
    }
    return state_map.get(api_state.lower(), InstanceState.UNKNOWN)


# Known Hetzner server types
HETZNER_SERVER_TYPES = {
    "cx22": {"cpu": 2, "memory_gb": 4, "hourly_cost": 0.006},
    "cx32": {"cpu": 4, "memory_gb": 8, "hourly_cost": 0.012},
    "cx42": {"cpu": 8, "memory_gb": 16, "hourly_cost": 0.024},
    "cx52": {"cpu": 16, "memory_gb": 32, "hourly_cost": 0.048},
    "cpx11": {"cpu": 2, "memory_gb": 2, "hourly_cost": 0.007},
    "cpx21": {"cpu": 3, "memory_gb": 4, "hourly_cost": 0.011},
    "cpx31": {"cpu": 4, "memory_gb": 8, "hourly_cost": 0.017},
    "cpx41": {"cpu": 8, "memory_gb": 16, "hourly_cost": 0.030},
    "cpx51": {"cpu": 16, "memory_gb": 32, "hourly_cost": 0.055},
}


class HetznerManager(ProviderManager):
    """Manage Hetzner Cloud servers via hcloud CLI.

    Usage:
        manager = HetznerManager()
        servers = await manager.list_instances()

        for server in servers:
            health = await manager.check_health(server)
            print(f"{server.name}: {health.message}")
    """

    provider = Provider.HETZNER

    def __init__(self):
        """Initialize Hetzner manager."""
        self._hcloud_available: bool | None = None

    async def _check_hcloud_available(self) -> bool:
        """Check if hcloud CLI is available."""
        if self._hcloud_available is not None:
            return self._hcloud_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "hcloud", "version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._hcloud_available = proc.returncode == 0
        except FileNotFoundError:
            self._hcloud_available = False

        if not self._hcloud_available:
            logger.warning("hcloud CLI not available")

        return self._hcloud_available

    async def _run_hcloud(self, *args: str) -> dict[str, Any] | list | None:
        """Run hcloud command and return JSON output."""
        if not await self._check_hcloud_available():
            return None

        cmd = ["hcloud", *args, "-o", "json"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.error(f"hcloud error: {stderr.decode()}")
                return None

            return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            logger.error("hcloud command timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hcloud output: {e}")
            return None
        except Exception as e:
            logger.error(f"hcloud command failed: {e}")
            return None

    async def _run_hcloud_action(self, *args: str) -> bool:
        """Run hcloud command that doesn't return JSON."""
        if not await self._check_hcloud_available():
            return False

        cmd = ["hcloud", *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                logger.error(f"hcloud action error: {stderr.decode()}")
                return False

            return True
        except Exception as e:
            logger.error(f"hcloud action failed: {e}")
            return False

    async def list_instances(self) -> list[ProviderInstance]:
        """List all Hetzner servers."""
        result = await self._run_hcloud("server", "list")
        if not result:
            return []

        instances = []
        for server_data in result:
            instance = self._parse_server(server_data)
            if instance:
                instances.append(instance)

        logger.info(f"Hetzner: found {len(instances)} servers")
        return instances

    async def get_instance(self, instance_id: str) -> ProviderInstance | None:
        """Get details of a specific server."""
        result = await self._run_hcloud("server", "describe", instance_id)
        if not result:
            return None
        return self._parse_server(result)

    def _parse_server(self, data: dict[str, Any]) -> ProviderInstance | None:
        """Parse hcloud output into ProviderInstance."""
        try:
            server_id = str(data.get("id", ""))
            name = data.get("name", server_id)

            # Get IP addresses
            public_net = data.get("public_net", {})
            public_ip = public_net.get("ipv4", {}).get("ip")

            # Get server type info
            server_type = data.get("server_type", {})
            type_name = server_type.get("name", "")
            type_info = HETZNER_SERVER_TYPES.get(type_name, {})

            # Look for Tailscale IP in labels
            labels = data.get("labels", {})
            tailscale_ip = labels.get("tailscale_ip")

            return ProviderInstance(
                instance_id=server_id,
                provider=Provider.HETZNER,
                name=name,
                public_ip=public_ip,
                tailscale_ip=tailscale_ip,
                state=_parse_server_state(data.get("status", "unknown")),
                cpu_count=type_info.get("cpu", server_type.get("cores", 0)),
                memory_gb=type_info.get("memory_gb", server_type.get("memory", 0)),
                hourly_cost=type_info.get("hourly_cost", 0.0),
                metadata={
                    "ssh_user": "root",
                    "ssh_key": "~/.ssh/id_cluster",
                    "server_type": type_name,
                    "datacenter": data.get("datacenter", {}).get("name", ""),
                    "labels": labels,
                },
            )
        except Exception as e:
            logger.error(f"Failed to parse Hetzner server: {e}")
            return None

    async def check_health(self, instance: ProviderInstance) -> HealthCheckResult:
        """Check health of a Hetzner server."""
        # Check SSH
        ssh_result = await self.check_ssh_connectivity(instance)
        if not ssh_result.healthy:
            return ssh_result

        # Check P2P daemon
        p2p_result = await self.check_p2p_health(instance)
        if not p2p_result.healthy:
            return p2p_result

        # Check Tailscale
        ts_result = await self.check_tailscale(instance)

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
        """Reboot a Hetzner server."""
        success = await self._run_hcloud_action("server", "reboot", instance_id)
        if success:
            logger.info(f"Hetzner: rebooting server {instance_id}")
        return success

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate (delete) a Hetzner server."""
        success = await self._run_hcloud_action("server", "delete", instance_id)
        if success:
            logger.info(f"Hetzner: deleted server {instance_id}")
        return success

    async def launch_instance(self, config: dict[str, Any]) -> str | None:
        """Launch a new Hetzner server.

        Args:
            config: Launch configuration with keys:
                - name: Server name (required)
                - server_type: e.g., "cpx51" (default)
                - image: e.g., "ubuntu-22.04" (default)
                - location: e.g., "fsn1", "nbg1", "hel1"
                - ssh_keys: list of SSH key names
                - labels: dict of labels

        Returns:
            Server ID if successful, None otherwise.
        """
        name = config.get("name")
        if not name:
            logger.error("Hetzner: server name required")
            return None

        args = [
            "server", "create",
            "--name", name,
            "--type", config.get("server_type", "cpx51"),
            "--image", config.get("image", "ubuntu-22.04"),
        ]

        if location := config.get("location"):
            args.extend(["--location", location])

        for ssh_key in config.get("ssh_keys", []):
            args.extend(["--ssh-key", ssh_key])

        for key, value in config.get("labels", {}).items():
            args.extend(["--label", f"{key}={value}"])

        result = await self._run_hcloud(*args)
        if result and isinstance(result, dict):
            server_id = str(result.get("server", {}).get("id", ""))
            if server_id:
                logger.info(f"Hetzner: created server {server_id}")
                return server_id

        return None

    async def power_on(self, instance_id: str) -> bool:
        """Power on a server."""
        return await self._run_hcloud_action("server", "poweron", instance_id)

    async def power_off(self, instance_id: str) -> bool:
        """Power off a server (hard shutdown)."""
        return await self._run_hcloud_action("server", "poweroff", instance_id)

    async def rebuild_server(
        self,
        instance_id: str,
        image: str = "ubuntu-22.04",
    ) -> bool:
        """Rebuild a server with a fresh image."""
        success = await self._run_hcloud_action(
            "server", "rebuild",
            instance_id,
            "--image", image,
        )
        if success:
            logger.info(f"Hetzner: rebuilding server {instance_id} with {image}")
        return success

    async def add_label(
        self,
        instance_id: str,
        key: str,
        value: str,
    ) -> bool:
        """Add/update a label on a server."""
        return await self._run_hcloud_action(
            "server", "add-label",
            instance_id,
            f"{key}={value}",
        )

    async def restart_p2p_daemon(self, instance: ProviderInstance) -> bool:
        """Restart P2P daemon on server via SSH."""
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
            logger.info(f"Hetzner: restarted P2P on {instance.name}")
            return True
        logger.error(f"Hetzner: failed to restart P2P on {instance.name}: {stderr}")
        return False

    async def restart_tailscale(self, instance: ProviderInstance) -> bool:
        """Restart Tailscale daemon on server."""
        cmd = "systemctl restart tailscaled && sleep 2 && tailscale status"
        code, stdout, stderr = await self.run_ssh_command(instance, cmd, timeout=30)
        if code == 0:
            logger.info(f"Hetzner: restarted Tailscale on {instance.name}")
            return True
        logger.error(f"Hetzner: failed to restart Tailscale on {instance.name}: {stderr}")
        return False


async def test_hetzner():
    """Test Hetzner CLI connectivity."""
    manager = HetznerManager()

    print("Testing Hetzner hcloud CLI...")
    servers = await manager.list_instances()

    if servers:
        print(f"\nFound {len(servers)} servers:")
        for server in servers:
            print(f"  {server}")
    else:
        print("No servers found (or hcloud not configured)")


if __name__ == "__main__":
    asyncio.run(test_hetzner())
