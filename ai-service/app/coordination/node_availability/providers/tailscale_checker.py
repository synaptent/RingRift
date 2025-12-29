"""Tailscale State Checker (December 2025).

Queries local Tailscale status to get peer connectivity states and maps them to
ProviderInstanceState enum for distributed_hosts.yaml synchronization.

Unlike cloud provider checkers (Vast, Lambda, RunPod), this checker:
1. Runs on the local node and checks the full mesh peer list
2. Detects nodes that are "running" per cloud API but disconnected from Tailscale
3. Provides actual connectivity status vs. just instance state

This enables detection of "masked failures" where:
- Cloud provider API says instance is running
- SSH gateway works
- But Tailscale inside the container is dead

Tailscale State Mappings:
- Online=true -> RUNNING (ready)
- Online=false -> STOPPED (offline)
- Not in mesh -> UNKNOWN (may be provider-terminated or just not Tailscale-connected)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime
from typing import Any, Optional

from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    StateChecker,
)

logger = logging.getLogger(__name__)


class TailscaleChecker(StateChecker):
    """State checker for Tailscale mesh connectivity.

    Queries `tailscale status --json` to get current peer states.
    Unlike cloud provider checkers, this checks actual network connectivity.

    Instance correlation with distributed_hosts.yaml:
    - Matches by tailscale_ip field
    - Falls back to hostname matching
    """

    def __init__(self, timeout_seconds: float = 10.0):
        """Initialize Tailscale checker.

        Args:
            timeout_seconds: Timeout for tailscale status command
        """
        super().__init__("tailscale")
        self._timeout = timeout_seconds
        self._tailscale_path: Optional[str] = None

        # Check if tailscale is installed
        self._tailscale_path = shutil.which("tailscale")
        if not self._tailscale_path:
            self.disable("Tailscale not installed")

    async def check_api_availability(self) -> bool:
        """Check if tailscale CLI is available and connected.

        Returns:
            True if tailscale CLI works and we're connected to the network.
        """
        if not self._tailscale_path:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )

            if proc.returncode != 0:
                return False

            # Parse to check BackendState
            data = json.loads(stdout.decode())
            backend_state = data.get("BackendState", "")
            return backend_state == "Running"

        except asyncio.TimeoutError:
            logger.warning("tailscale status check timed out")
            return False
        except json.JSONDecodeError:
            logger.warning("Failed to parse tailscale status JSON")
            return False
        except FileNotFoundError:
            logger.warning("tailscale CLI not found")
            return False
        except OSError as e:
            logger.warning(f"tailscale CLI OS error: {e}")
            return False

    async def get_instance_states(self) -> list[InstanceInfo]:
        """Query Tailscale for all peer states.

        Returns:
            List of InstanceInfo for all Tailscale peers in the mesh.
        """
        if not self.is_enabled:
            return []

        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )

            if proc.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                self._last_error = error_msg
                logger.error(f"tailscale status failed: {error_msg}")
                return []

            self._last_check = datetime.now()
            self._last_error = None

            # Parse JSON response
            data = json.loads(stdout.decode())
            instances = []

            # Get self info first
            self_info = data.get("Self", {})
            if self_info:
                instances.append(self._parse_peer_info(self_info, is_self=True))

            # Get all peers
            peers = data.get("Peer", {})
            for peer_id, peer_data in peers.items():
                info = self._parse_peer_info(peer_data, is_self=False)
                if info:
                    instances.append(info)

            logger.debug(f"Found {len(instances)} Tailscale peers")
            return instances

        except json.JSONDecodeError as e:
            self._last_error = f"JSON parse error: {e}"
            logger.error(f"Failed to parse tailscale output: {e}")
            return []
        except asyncio.TimeoutError:
            self._last_error = "Command timed out"
            logger.error("tailscale status timed out")
            return []
        except OSError as e:
            self._last_error = f"OS error: {e}"
            logger.error(f"OS error querying Tailscale: {e}")
            return []
        except (KeyError, ValueError, TypeError) as e:
            self._last_error = f"Data parse error: {e}"
            logger.error(f"Error parsing Tailscale response: {e}")
            return []

    def _parse_peer_info(
        self,
        peer_data: dict[str, Any],
        is_self: bool = False,
    ) -> Optional[InstanceInfo]:
        """Parse peer info from tailscale status JSON.

        Args:
            peer_data: Peer data from tailscale status --json
            is_self: True if this is the local node (Self key)

        Returns:
            InstanceInfo or None if parsing failed
        """
        try:
            # Get hostname for ID
            hostname = peer_data.get("HostName", "")
            if not hostname:
                return None

            # Determine state from Online flag
            # Note: Self is always online, peers have Online field
            if is_self:
                is_online = True
            else:
                is_online = peer_data.get("Online", False)

            state = (
                ProviderInstanceState.RUNNING
                if is_online
                else ProviderInstanceState.STOPPED
            )

            # Get Tailscale IP (first IPv4)
            tailscale_ips = peer_data.get("TailscaleIPs", [])
            tailscale_ip = None
            for ip in tailscale_ips:
                if "." in ip:  # IPv4
                    tailscale_ip = ip
                    break

            # Create instance info
            return InstanceInfo(
                instance_id=hostname,  # Use hostname as ID for Tailscale peers
                state=state,
                provider="tailscale",
                node_name=None,  # Will be set by correlate_with_config
                tailscale_ip=tailscale_ip,
                hostname=hostname,
                raw_data=peer_data,
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse peer: {e}")
            return None

    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        """Match Tailscale peers to node names in config.

        Matching strategies:
        1. tailscale_ip matches config's tailscale_ip field
        2. Hostname matches node name (case-insensitive)
        3. Hostname contains node name (for vast-123 -> vast-123-gpu style)
        """
        # Build lookup maps from config
        ip_to_node: dict[str, str] = {}
        hostname_to_node: dict[str, str] = {}

        for node_name, host_config in config_hosts.items():
            # Match by tailscale_ip
            ts_ip = host_config.get("tailscale_ip")
            if ts_ip:
                ip_to_node[ts_ip] = node_name

            # Build hostname lookup (lowercase for case-insensitive matching)
            hostname_to_node[node_name.lower()] = node_name

        # Correlate instances
        for instance in instances:
            # Skip if already correlated
            if instance.node_name:
                continue

            # Try IP match first (most reliable)
            if instance.tailscale_ip and instance.tailscale_ip in ip_to_node:
                instance.node_name = ip_to_node[instance.tailscale_ip]
                continue

            # Try exact hostname match
            if instance.hostname:
                hostname_lower = instance.hostname.lower()
                if hostname_lower in hostname_to_node:
                    instance.node_name = hostname_to_node[hostname_lower]
                    continue

                # Try partial match (hostname contains node name)
                for node_lower, node_name in hostname_to_node.items():
                    if node_lower in hostname_lower:
                        instance.node_name = node_name
                        break

        return instances

    async def get_disconnected_nodes(
        self,
        config_hosts: dict[str, dict],
    ) -> list[str]:
        """Find nodes in config that are offline in Tailscale.

        This detects "masked failures" where a cloud instance is running
        but Tailscale inside it is dead.

        Args:
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            List of node names that have tailscale_ip but are offline.
        """
        # Get current peers
        instances = await self.get_instance_states()

        # Conservative: if we couldn't get status, don't report disconnections
        if self._last_error is not None:
            logger.warning(f"Cannot check disconnections: {self._last_error}")
            return []

        # Build set of online Tailscale IPs
        online_ips = {
            inst.tailscale_ip
            for inst in instances
            if inst.tailscale_ip and inst.state == ProviderInstanceState.RUNNING
        }

        # Find nodes with tailscale_ip that aren't online
        disconnected = []
        for node_name, host_config in config_hosts.items():
            ts_ip = host_config.get("tailscale_ip")
            if not ts_ip:
                continue  # Node doesn't use Tailscale

            status = host_config.get("status", "")
            if status == "retired":
                continue  # Already marked as retired

            if ts_ip not in online_ips:
                disconnected.append(node_name)

        return disconnected

    async def get_online_nodes(
        self,
        config_hosts: dict[str, dict],
    ) -> list[str]:
        """Find nodes in config that are online in Tailscale.

        Args:
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            List of node names that are connected and online.
        """
        instances = await self.get_instance_states()

        # Correlate with config
        instances = self.correlate_with_config(instances, config_hosts)

        # Return nodes that are online and have a node_name
        return [
            inst.node_name
            for inst in instances
            if inst.node_name and inst.state == ProviderInstanceState.RUNNING
        ]
