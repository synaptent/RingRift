"""RunPod State Checker (December 2025).

Queries RunPod GraphQL API to get current pod states and maps them to
ProviderInstanceState enum for distributed_hosts.yaml synchronization.

RunPod API Documentation: https://docs.runpod.io/api/graphql

RunPod State Mappings:
- "RUNNING" -> RUNNING (ready)
- "STARTING" -> STARTING (setup)
- "STOPPED" -> STOPPED (offline)
- "TERMINATED" -> TERMINATED (retired)
- "EXITED" -> STOPPED (offline)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    StateChecker,
)

logger = logging.getLogger(__name__)

# RunPod state mappings
RUNPOD_STATE_MAP: dict[str, ProviderInstanceState] = {
    "RUNNING": ProviderInstanceState.RUNNING,
    "STARTING": ProviderInstanceState.STARTING,
    "STOPPED": ProviderInstanceState.STOPPED,
    "TERMINATED": ProviderInstanceState.TERMINATED,
    "EXITED": ProviderInstanceState.STOPPED,
    "CREATED": ProviderInstanceState.STARTING,
}

RUNPOD_API_URL = "https://api.runpod.io/graphql"

# GraphQL query for pods
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


class RunPodChecker(StateChecker):
    """State checker for RunPod pods.

    Uses RunPod GraphQL API to query pod states.
    Requires RUNPOD_API_KEY environment variable.

    Instance correlation with distributed_hosts.yaml:
    - Matches by pod name pattern (runpod-*)
    - Falls back to IP address matching
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize RunPod checker.

        Args:
            api_key: RunPod API key (uses env if None)
        """
        super().__init__("runpod")
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self._http_session = None

        # Check for API key in config files if not in env
        if not self._api_key:
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
                                    # Parse: apikey = "..."
                                    parts = line.split("=", 1)
                                    if len(parts) == 2:
                                        key = parts[1].strip().strip('"').strip("'")
                                        if key:
                                            self._api_key = key
                                            break
                        if self._api_key:
                            break
                    except OSError:
                        pass

        if not self._api_key:
            self.disable("No RUNPOD_API_KEY found")

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._http_session is None:
            try:
                import aiohttp
                self._http_session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError:
                logger.warning("aiohttp not available, RunPod checker disabled")
                self.disable("aiohttp not installed")
                return None
        return self._http_session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def check_api_availability(self) -> bool:
        """Check if RunPod API is accessible."""
        if not self._api_key:
            return False

        session = await self._get_session()
        if not session:
            return False

        try:
            async with session.post(
                RUNPOD_API_URL,
                json={"query": "query { myself { id } }"}
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                return "errors" not in data
        except Exception as e:
            logger.warning(f"RunPod API check failed: {e}")
            return False

    async def get_instance_states(self) -> list[InstanceInfo]:
        """Query RunPod for all pod states.

        Returns:
            List of InstanceInfo for all RunPod pods.
        """
        if not self.is_enabled:
            return []

        session = await self._get_session()
        if not session:
            return []

        try:
            async with session.post(
                RUNPOD_API_URL,
                json={"query": PODS_QUERY}
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self._last_error = f"API error {resp.status}: {error_text}"
                    logger.error(f"RunPod API error: {self._last_error}")
                    return []

                data = await resp.json()
                if "errors" in data:
                    self._last_error = str(data["errors"])
                    logger.error(f"RunPod GraphQL error: {self._last_error}")
                    return []

                self._last_check = datetime.now()
                self._last_error = None

                instances = []
                pods = data.get("data", {}).get("myself", {}).get("pods", [])

                for pod in pods:
                    pod_id = pod.get("id", "")
                    pod_name = pod.get("name", "")
                    desired_status = pod.get("desiredStatus", "UNKNOWN")
                    state = RUNPOD_STATE_MAP.get(desired_status, ProviderInstanceState.UNKNOWN)

                    # Extract runtime info
                    runtime = pod.get("runtime") or {}
                    ports = runtime.get("ports", [])
                    public_ip = None
                    ssh_port = 22

                    for port in ports:
                        if port.get("isIpPublic") and port.get("privatePort") == 22:
                            public_ip = port.get("ip")
                            ssh_port = port.get("publicPort", 22)
                            break

                    # Extract machine info
                    machine = pod.get("machine") or {}

                    info = InstanceInfo(
                        instance_id=pod_id,
                        state=state,
                        provider="runpod",
                        hostname=pod_name,
                        public_ip=public_ip,
                        ssh_host=public_ip,
                        ssh_port=ssh_port,
                        gpu_type=machine.get("gpuDisplayName"),
                        gpu_count=machine.get("gpuCount", 1),
                        gpu_vram_gb=machine.get("memoryInGb", 0),
                        raw_data=pod,
                    )

                    instances.append(info)

                logger.debug(f"Found {len(instances)} RunPod pods")
                return instances

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error querying RunPod pods: {e}")
            return []

    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        """Match RunPod pods to node names in config.

        Matching strategies:
        1. Node name matches pod name
        2. SSH host/port matches
        """
        # Build lookup maps from config
        name_to_node: dict[str, str] = {}
        ssh_to_node: dict[tuple[str, int], str] = {}

        for node_name, host_config in config_hosts.items():
            if not node_name.startswith("runpod-"):
                continue

            name_to_node[node_name] = node_name

            # Build SSH lookup
            ssh_host = host_config.get("ssh_host")
            ssh_port = host_config.get("ssh_port", 22)
            if ssh_host:
                ssh_to_node[(ssh_host, ssh_port)] = node_name

        # Correlate instances
        for instance in instances:
            # Try name match first
            pod_name = instance.hostname or ""
            if pod_name in name_to_node:
                instance.node_name = name_to_node[pod_name]
                continue

            # Try pattern matching (runpod-h100, runpod-a100-1, etc.)
            for node_name in name_to_node.keys():
                # Check if pod name contains key parts of node name
                node_suffix = node_name.replace("runpod-", "")
                if node_suffix.lower() in pod_name.lower():
                    instance.node_name = node_name
                    break

            if instance.node_name:
                continue

            # Try SSH match
            ssh_key = (instance.ssh_host, instance.ssh_port)
            if ssh_key in ssh_to_node:
                instance.node_name = ssh_to_node[ssh_key]

        return instances

    async def get_terminated_instances(
        self,
        config_hosts: dict[str, dict],
    ) -> list[str]:
        """Find nodes in config that are no longer in RunPod.

        Args:
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            List of node names that appear terminated (not in API response).
        """
        # Get current pods
        instances = await self.get_instance_states()
        active_names = {inst.hostname for inst in instances if inst.hostname}
        active_ips = {inst.public_ip for inst in instances if inst.public_ip}

        # Find runpod nodes in config that aren't in API response
        terminated = []
        for node_name, host_config in config_hosts.items():
            if not node_name.startswith("runpod-"):
                continue

            # Check if node is active by name
            if node_name in active_names:
                continue

            # Check if node is active by IP
            ssh_host = host_config.get("ssh_host")
            if ssh_host and ssh_host in active_ips:
                continue

            # Node not found in active pods
            current_status = host_config.get("status", "")
            if current_status not in ("retired", "offline"):
                terminated.append(node_name)

        return terminated
