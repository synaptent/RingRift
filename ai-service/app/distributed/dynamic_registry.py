"""Dynamic Host Registry - Auto-updates host IPs and handles transient failures.

This module provides:
1. Soft/hard offline detection (degraded vs dead)
2. Self-registration endpoint for nodes to announce IP changes
3. Vast.ai API integration for automatic IP updates
4. Optional YAML config writeback

Usage:
    from app.distributed.dynamic_registry import DynamicHostRegistry

    registry = DynamicHostRegistry()

    # Node self-registration (call from each node)
    registry.register_node("vast-5090-quad", "211.72.13.202", 45875)

    # Get current host config (uses dynamic IP if available)
    host = registry.get_host("vast-5090-quad")

    # Check health state
    state = registry.get_node_state("vast-5090-quad")  # "online", "degraded", "offline"
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Import secrets handling utilities
try:
    from app.utils.secrets import SecretString, mask_secret
except ImportError:
    # Fallback if utils not available
    class SecretString:  # type: ignore[no-redef]
        def __init__(self, value: str):
            self._value = value
        def get_value(self) -> str:
            return self._value
        def __str__(self) -> str:
            if not self._value:
                return "[empty]"
            return "***" + self._value[-4:] if len(self._value) > 4 else "****"
        def __bool__(self) -> bool:
            return bool(self._value)

    def mask_secret(value: str | None, visible_chars: int = 4) -> str:
        if not value:
            return "[empty]"
        if len(value) <= visible_chars:
            return "*" * len(value)
        return "***" + value[-visible_chars:]

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    """Health state of a node."""
    ONLINE = "online"           # Responding normally
    DEGRADED = "degraded"       # Temporary failures, still trying
    OFFLINE = "offline"         # Confirmed offline after multiple failures
    UNKNOWN = "unknown"         # Never contacted


# Try to load from unified config, with fallback defaults
try:
    from app.config.unified_config import get_config
    _config = get_config()
    DEGRADED_THRESHOLD = _config.distributed.degraded_failure_threshold
    OFFLINE_THRESHOLD = _config.distributed.offline_failure_threshold
    RECOVERY_CHECKS = _config.distributed.recovery_success_threshold
    VAST_API_CHECK_INTERVAL = _config.distributed.vast_api_check_interval_seconds
    AWS_API_CHECK_INTERVAL = _config.distributed.aws_api_check_interval_seconds
    TAILSCALE_CHECK_INTERVAL = _config.distributed.tailscale_check_interval_seconds
except ImportError:
    # Fallback defaults if config not available
    DEGRADED_THRESHOLD = 2          # Failures before degraded
    OFFLINE_THRESHOLD = 5           # Failures before offline
    RECOVERY_CHECKS = 2             # Successful checks to go from degraded to online
    VAST_API_CHECK_INTERVAL = 300   # Check Vast API every 5 minutes
    AWS_API_CHECK_INTERVAL = 300    # Check AWS (CLI) every 5 minutes
    TAILSCALE_CHECK_INTERVAL = 120  # Check Tailscale status every 2 minutes

STATE_FILE = "logs/p2p_orchestrator/dynamic_registry.json"


@dataclass
class DynamicNodeInfo:
    """Dynamic information about a node."""
    node_id: str
    static_host: str            # Original host from YAML
    static_port: int            # Original port from YAML
    dynamic_host: str | None = None   # Self-registered or API-discovered host
    dynamic_port: int | None = None   # Self-registered or API-discovered port
    state: NodeState = NodeState.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_registration_time: float = 0.0
    failure_reason: str | None = None
    vast_instance_id: str | None = None   # For Vast.ai API queries
    aws_instance_id: str | None = None    # For AWS CLI queries (ec2 instance id)
    aws_region: str | None = None         # Optional explicit region for the instance id
    tailscale_ip: str | None = None       # Discovered mesh IP (100.x) when available

    @property
    def effective_host(self) -> str:
        """Get the current effective host (dynamic if available)."""
        return self.dynamic_host or self.static_host

    @property
    def effective_port(self) -> int:
        """Get the current effective port (dynamic if available)."""
        return self.dynamic_port or self.static_port


class DynamicHostRegistry:
    """Registry that tracks dynamic host information and health states."""

    def __init__(self, config_path: str | None = None):
        self._nodes: dict[str, DynamicNodeInfo] = {}
        self._lock = threading.RLock()
        self._config_path = config_path
        self._state_file = Path(__file__).parent.parent.parent / STATE_FILE
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        # Store API key as SecretString to prevent accidental logging
        _api_key = os.environ.get("VAST_API_KEY")
        self._vast_api_key: SecretString | None = SecretString(_api_key) if _api_key else None
        self._last_vast_check = 0.0
        self._last_aws_check = 0.0
        self._last_tailscale_check = 0.0

        # Load static config
        self._load_static_config()

        # Load persisted dynamic state
        self._load_state()

    def _load_static_config(self) -> None:
        """Load hosts from YAML config."""
        try:
            from app.distributed.hosts import load_remote_hosts
            hosts = load_remote_hosts(self._config_path)

            with self._lock:
                for name, host_config in hosts.items():
                    if name not in self._nodes:
                        self._nodes[name] = DynamicNodeInfo(
                            node_id=name,
                            static_host=host_config.ssh_host,
                            static_port=host_config.ssh_port,
                        )

                        # Extract Vast instance ID if present
                        props = host_config.properties
                        if "vast_instance_id" in props:
                            self._nodes[name].vast_instance_id = props["vast_instance_id"]
                        if "aws_instance_id" in props:
                            self._nodes[name].aws_instance_id = props["aws_instance_id"]
                        if "aws_region" in props:
                            self._nodes[name].aws_region = str(props["aws_region"]).strip() or None
                        if "tailscale_ip" in props:
                            self._nodes[name].tailscale_ip = props["tailscale_ip"]
                        elif name.startswith("vast-"):
                            # Try to extract from comments in YAML
                            for _key, val in props.items():
                                if isinstance(val, str) and "Instance ID:" in val:
                                    # Parse "Instance ID: 28654132" from comments
                                    parts = val.split("Instance ID:")
                                    if len(parts) > 1:
                                        instance_id = parts[1].strip().split()[0]
                                        self._nodes[name].vast_instance_id = instance_id
                    else:
                        # Update static config but preserve dynamic state
                        self._nodes[name].static_host = host_config.ssh_host
                        self._nodes[name].static_port = host_config.ssh_port

            logger.info(f"Loaded {len(self._nodes)} hosts into dynamic registry")
        except (ImportError, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to load static config: {e}")

    def _load_state(self) -> None:
        """Load persisted dynamic state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    data = json.load(f)

                with self._lock:
                    for node_id, node_data in data.get("nodes", {}).items():
                        if node_id in self._nodes:
                            node = self._nodes[node_id]
                            node.dynamic_host = node_data.get("dynamic_host")
                            node.dynamic_port = node_data.get("dynamic_port")
                            node.vast_instance_id = node_data.get("vast_instance_id")
                            node.aws_instance_id = node_data.get("aws_instance_id")
                            node.aws_region = node_data.get("aws_region")
                            node.tailscale_ip = node_data.get("tailscale_ip")
                            # Don't restore state - re-check on startup
                            node.state = NodeState.UNKNOWN
                            node.consecutive_failures = 0
                            node.consecutive_successes = 0

                logger.info(f"Loaded dynamic registry state from {self._state_file}")
        except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load registry state: {e}")

    def _save_state(self) -> None:
        """Persist dynamic state to disk."""
        try:
            with self._lock:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "nodes": {
                        node_id: {
                            "dynamic_host": node.dynamic_host,
                            "dynamic_port": node.dynamic_port,
                            "vast_instance_id": node.vast_instance_id,
                            "aws_instance_id": node.aws_instance_id,
                            "aws_region": node.aws_region,
                            "tailscale_ip": node.tailscale_ip,
                            "state": node.state.value,
                            "last_success_time": node.last_success_time,
                        }
                        for node_id, node in self._nodes.items()
                    }
                }

            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to save registry state: {e}")

    def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        vast_instance_id: str | None = None,
        tailscale_ip: str | None = None,
    ) -> bool:
        """Register or update a node's dynamic address.

        Called by nodes to announce their current IP (e.g., after Vast restart).

        Args:
            node_id: Node identifier (e.g., "vast-5090-quad")
            host: Current IP address
            port: Current SSH port
            vast_instance_id: Optional Vast.ai instance ID

        Returns:
            True if registration was accepted
        """
        with self._lock:
            if node_id not in self._nodes:
                # New node - create entry
                self._nodes[node_id] = DynamicNodeInfo(
                    node_id=node_id,
                    static_host=host,
                    static_port=port,
                )

            node = self._nodes[node_id]
            old_host = node.effective_host
            old_port = node.effective_port

            node.dynamic_host = host
            node.dynamic_port = port
            node.last_registration_time = time.time()

            if vast_instance_id:
                node.vast_instance_id = vast_instance_id
            if tailscale_ip:
                node.tailscale_ip = tailscale_ip

            # If IP changed, reset failure counters
            if host != old_host or port != old_port:
                node.consecutive_failures = 0
                node.state = NodeState.UNKNOWN
                logger.info(f"Node {node_id} updated: {old_host}:{old_port} -> {host}:{port}")

            self._save_state()
            return True

    def record_check_result(
        self,
        node_id: str,
        success: bool,
        failure_reason: str | None = None,
    ) -> NodeState:
        """Record result of a health check and update node state.

        Args:
            node_id: Node identifier
            success: Whether the check succeeded
            failure_reason: Optional reason for failure

        Returns:
            New node state
        """
        with self._lock:
            if node_id not in self._nodes:
                return NodeState.UNKNOWN

            node = self._nodes[node_id]
            node.last_check_time = time.time()

            if success:
                node.consecutive_failures = 0
                node.consecutive_successes += 1
                node.last_success_time = time.time()
                node.failure_reason = None

                # Transition to online
                if node.state == NodeState.DEGRADED:
                    if node.consecutive_successes >= RECOVERY_CHECKS:
                        node.state = NodeState.ONLINE
                        logger.info(f"Node {node_id} recovered: degraded -> online")
                elif node.state in (NodeState.UNKNOWN, NodeState.OFFLINE):
                    node.state = NodeState.ONLINE
                    logger.info(f"Node {node_id} is now online")
            else:
                node.consecutive_successes = 0
                node.consecutive_failures += 1
                node.last_failure_time = time.time()
                node.failure_reason = failure_reason

                # Transition through states
                if node.consecutive_failures >= OFFLINE_THRESHOLD:
                    if node.state != NodeState.OFFLINE:
                        node.state = NodeState.OFFLINE
                        logger.warning(f"Node {node_id} is now offline after {node.consecutive_failures} failures")
                elif node.consecutive_failures >= DEGRADED_THRESHOLD and node.state == NodeState.ONLINE:
                    node.state = NodeState.DEGRADED
                    logger.warning(f"Node {node_id} is degraded: {failure_reason}")

            return node.state

    def get_node_state(self, node_id: str) -> NodeState:
        """Get current state of a node."""
        with self._lock:
            if node_id in self._nodes:
                return self._nodes[node_id].state
            return NodeState.UNKNOWN

    def get_effective_address(self, node_id: str) -> tuple[str, int] | None:
        """Get current effective address for a node.

        Returns:
            Tuple of (host, port) or None if node not found
        """
        with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                return (node.effective_host, node.effective_port)
            return None

    def get_online_nodes(self) -> list[str]:
        """Get list of nodes that are online or degraded (still usable)."""
        with self._lock:
            return [
                node_id for node_id, node in self._nodes.items()
                if node.state in (NodeState.ONLINE, NodeState.DEGRADED)
            ]

    def get_all_nodes_status(self) -> dict[str, dict[str, Any]]:
        """Get status summary of all nodes."""
        with self._lock:
            return {
                node_id: {
                    "state": node.state.value,
                    "effective_host": node.effective_host,
                    "effective_port": node.effective_port,
                    "consecutive_failures": node.consecutive_failures,
                    "last_success": datetime.fromtimestamp(node.last_success_time).isoformat() if node.last_success_time else None,
                    "failure_reason": node.failure_reason,
                }
                for node_id, node in self._nodes.items()
            }

    async def update_vast_ips(self) -> int:
        """Query Vast.ai API and update IPs for Vast instances.

        Returns:
            Number of nodes updated
        """
        # Rate limit API calls
        if time.time() - self._last_vast_check < VAST_API_CHECK_INTERVAL:
            return 0

        self._last_vast_check = time.time()

        if not self._vast_api_key:
            return await self._update_vast_ips_via_cli()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Use get_value() to get actual API key - SecretString prevents accidental logging
                headers = {"Authorization": f"Bearer {self._vast_api_key.get_value()}"}
                async with session.get(
                    "https://console.vast.ai/api/v0/instances/",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Vast API returned {resp.status}")
                        return 0

                    data = await resp.json()
        except ImportError:
            logger.warning("aiohttp not installed, can't query Vast API")
            return 0
        except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to query Vast API: {e}")
            return 0

        updated = 0
        instances = data.get("instances", [])

        with self._lock:
            # Build map of instance_id -> node_id
            instance_to_node = {}
            for node_id, node in self._nodes.items():
                if node.vast_instance_id:
                    instance_to_node[node.vast_instance_id] = node_id

            for instance in instances:
                instance_id = str(instance.get("id"))
                if instance_id not in instance_to_node:
                    continue

                node_id = instance_to_node[instance_id]
                node = self._nodes[node_id]

                # Get current IP and port from Vast
                ssh_host = instance.get("ssh_host")
                ssh_port = instance.get("ssh_port")

                if (ssh_host and ssh_port
                        and (ssh_host != node.dynamic_host or ssh_port != node.dynamic_port)):
                    logger.info(f"Vast API: {node_id} IP updated to {ssh_host}:{ssh_port}")
                    node.dynamic_host = ssh_host
                    node.dynamic_port = ssh_port
                    node.consecutive_failures = 0
                    node.state = NodeState.UNKNOWN
                    updated += 1

        if updated:
            self._save_state()

        return updated

    async def _update_vast_ips_via_cli(self) -> int:
        """Fallback Vast.ai discovery via `vastai` CLI when VAST_API_KEY is not set."""
        try:
            import asyncio

            def _run() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["vastai", "show", "instances", "--raw"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            result = await asyncio.to_thread(_run)
            if result.returncode != 0:
                return 0

            try:
                instances = json.loads(result.stdout or "[]")
            except (json.JSONDecodeError, TypeError, ValueError):
                return 0

            updated = 0
            with self._lock:
                instance_to_node = {
                    str(node.vast_instance_id): node_id
                    for node_id, node in self._nodes.items()
                    if node.vast_instance_id
                }
                for inst in instances:
                    instance_id = str(inst.get("id") or "")
                    if not instance_id or instance_id not in instance_to_node:
                        continue
                    ssh_host = inst.get("ssh_host")
                    ssh_port = inst.get("ssh_port")
                    if not ssh_host or not ssh_port:
                        continue
                    node_id = instance_to_node[instance_id]
                    node = self._nodes[node_id]
                    if ssh_host != node.dynamic_host or ssh_port != node.dynamic_port:
                        logger.info(f"Vast CLI: {node_id} IP updated to {ssh_host}:{ssh_port}")
                        node.dynamic_host = ssh_host
                        node.dynamic_port = ssh_port
                        node.consecutive_failures = 0
                        node.state = NodeState.UNKNOWN
                        updated += 1

            if updated:
                self._save_state()
            return updated
        except FileNotFoundError:
            return 0
        except (TimeoutError, OSError, json.JSONDecodeError):
            return 0

    async def update_aws_ips(self) -> int:
        """Query AWS (via `aws` CLI) and update IPs for configured EC2 instances.

        Nodes must define `aws_instance_id` in distributed_hosts.yaml properties.

        Returns:
            Number of nodes updated
        """
        # Rate limit calls.
        if time.time() - self._last_aws_check < AWS_API_CHECK_INTERVAL:
            return 0
        self._last_aws_check = time.time()

        instance_ids: list[str] = []
        with self._lock:
            for node in self._nodes.values():
                if node.aws_instance_id:
                    instance_ids.append(str(node.aws_instance_id))

        if not instance_ids:
            return 0

        try:
            import asyncio
            import re

            region_env = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or ""
            ).strip()

            # Group instance IDs by region so mixed-region clusters can still
            # refresh IPs reliably without depending on a single default region.
            by_region: dict[str, list[str]] = {}
            id_to_node: dict[str, str] = {}
            with self._lock:
                for node_id, node in self._nodes.items():
                    if not node.aws_instance_id:
                        continue
                    instance_id = str(node.aws_instance_id)
                    region = (node.aws_region or region_env or "").strip()
                    by_region.setdefault(region, []).append(instance_id)
                    id_to_node[instance_id] = node_id

            def _extract_invalid_ids(stderr: str) -> list[str]:
                # Common AWS CLI error when instances are terminated/deleted:
                #   InvalidInstanceID.NotFound ... instance IDs 'i-...' do not exist
                if not stderr:
                    return []
                if "InvalidInstanceID" not in stderr and "InvalidInstanceID.NotFound" not in stderr:
                    return []
                return re.findall(r"i-[0-9a-fA-F]{8,32}", stderr)

            async def _describe_instances(region: str, instance_ids: list[str]) -> dict[str, Any] | None:
                cmd = [
                    "aws",
                    "ec2",
                    "describe-instances",
                    "--instance-ids",
                    *instance_ids,
                    "--output",
                    "json",
                ]
                if region:
                    cmd.extend(["--region", region])

                def _run() -> subprocess.CompletedProcess[str]:
                    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                result = await asyncio.to_thread(_run)
                if result.returncode != 0:
                    invalid = _extract_invalid_ids(result.stderr or "")
                    if invalid:
                        # Caller will retry without invalid IDs.
                        return {"__invalid_instance_ids": invalid, "__stderr": result.stderr or ""}
                    logger.warning(
                        "AWS CLI describe-instances failed "
                        f"(region={region or 'default'}, ids={len(instance_ids)}): "
                        f"{(result.stderr or '').strip()[:200]}"
                    )
                    return None

                try:
                    return json.loads(result.stdout or "{}")
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None

            updated = 0
            for region, region_instance_ids in by_region.items():
                remaining = list(dict.fromkeys(region_instance_ids))  # stable de-dupe
                payload: dict[str, Any] | None = None
                invalid_ids: list[str] = []

                # Retry once if terminated instance IDs cause the bulk query to fail.
                for _attempt in range(2):
                    if not remaining:
                        break
                    response = await _describe_instances(region, remaining)
                    if response is None:
                        payload = None
                        break
                    invalid = response.get("__invalid_instance_ids") if isinstance(response, dict) else None
                    if invalid:
                        invalid_ids = [str(i) for i in invalid if str(i)]
                        remaining = [i for i in remaining if i not in set(invalid_ids)]
                        continue
                    payload = response
                    break

                if invalid_ids:
                    with self._lock:
                        for bad_id in invalid_ids:
                            node_id = id_to_node.get(bad_id)
                            if not node_id:
                                continue
                            self.record_check_result(node_id, False, failure_reason="aws_instance_not_found")

                if not payload:
                    continue

                reservations = payload.get("Reservations") or []
                with self._lock:
                    for reservation in reservations:
                        instances = reservation.get("Instances") or []
                        for inst in instances:
                            instance_id = str(inst.get("InstanceId") or "")
                            if not instance_id or instance_id not in id_to_node:
                                continue
                            state = (inst.get("State") or {}).get("Name") or ""
                            public_ip = inst.get("PublicIpAddress") or ""
                            node_id = id_to_node[instance_id]
                            node = self._nodes.get(node_id)
                            if not node:
                                continue

                            # Mark as non-healthy if instance isn't running; still allow
                            # the registry to recover automatically if it comes back.
                            if state and state != "running":
                                self.record_check_result(
                                    node_id, False, failure_reason=f"aws_state_{state}"
                                )
                                continue
                            if not public_ip:
                                self.record_check_result(node_id, False, failure_reason="aws_no_public_ip")
                                continue

                            self.record_check_result(node_id, True)
                            if public_ip != node.dynamic_host:
                                logger.info(
                                    f"AWS CLI: {node_id} IP updated to {public_ip}:{node.static_port} "
                                    f"(region={region or 'default'})"
                                )
                                node.dynamic_host = public_ip
                                node.dynamic_port = node.static_port
                                node.consecutive_failures = 0
                                node.state = NodeState.UNKNOWN
                                updated += 1

            if updated:
                self._save_state()
            return updated
        except FileNotFoundError:
            return 0
        except (TimeoutError, OSError, json.JSONDecodeError):
            return 0

    async def update_tailscale_ips(self) -> int:
        """Discover and record Tailscale IPs for nodes via `tailscale status --json`.

        This is best-effort and safe: it only fills/updates `tailscale_ip` fields
        and does not overwrite `dynamic_host` (which may be a public SSH address).

        Returns:
            Number of nodes whose tailscale_ip changed.
        """
        if time.time() - self._last_tailscale_check < TAILSCALE_CHECK_INTERVAL:
            return 0
        self._last_tailscale_check = time.time()

        try:
            import asyncio

            def _run() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["tailscale", "status", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

            result = await asyncio.to_thread(_run)
            if result.returncode != 0:
                return 0

            payload = json.loads(result.stdout or "{}")
        except FileNotFoundError:
            return 0
        except (TimeoutError, OSError, json.JSONDecodeError):
            return 0

        def _normalize_name(name: str) -> str:
            name = (name or "").strip().lower()
            if not name:
                return ""
            if "." in name:
                name = name.split(".", 1)[0]
            return name

        def _first_ipv4(ips: Any) -> str | None:
            if not isinstance(ips, list):
                return None
            for ip in ips:
                if isinstance(ip, str) and ip and ":" not in ip:
                    return ip.strip()
            return None

        name_to_ip: dict[str, str] = {}

        try:
            self_entry = payload.get("Self") or {}
            self_ip = _first_ipv4(self_entry.get("TailscaleIPs"))
            self_name = _normalize_name(self_entry.get("HostName") or self_entry.get("DNSName") or "")
            if self_ip and self_name:
                name_to_ip[self_name] = self_ip
        except (KeyError, AttributeError, TypeError):
            pass

        peers = payload.get("Peer") or {}
        if isinstance(peers, dict):
            for peer in peers.values():
                if not isinstance(peer, dict):
                    continue
                ip = _first_ipv4(peer.get("TailscaleIPs"))
                if not ip:
                    continue
                host = _normalize_name(peer.get("HostName") or "")
                dns = _normalize_name(peer.get("DNSName") or "")
                if host:
                    name_to_ip[host] = ip
                if dns:
                    name_to_ip[dns] = ip

        updated = 0
        with self._lock:
            for node_id, node in self._nodes.items():
                key = _normalize_name(node_id)
                ip = name_to_ip.get(key)
                if not ip:
                    continue
                if ip != node.tailscale_ip:
                    node.tailscale_ip = ip
                    updated += 1

        if updated:
            self._save_state()
        return updated

    def update_yaml_config(self) -> bool:
        """Write current dynamic addresses back to YAML config.

        This is optional and only updates hosts where dynamic != static.
        Creates a backup before modifying.

        Returns:
            True if config was updated
        """
        try:
            import yaml

            from app.distributed.hosts import CONFIG_FILE_PATH, get_ai_service_dir

            config_path = get_ai_service_dir() / CONFIG_FILE_PATH
            if not config_path.exists():
                return False

            # Create backup
            backup_path = config_path.with_suffix(".yaml.bak")
            with open(config_path) as f:
                original_content = f.read()
            with open(backup_path, "w") as f:
                f.write(original_content)

            # Load and update
            config = yaml.safe_load(original_content)

            updated = False
            with self._lock:
                for node_id, node in self._nodes.items():
                    if node_id not in config.get("hosts", {}):
                        continue

                    if node.dynamic_host and node.dynamic_host != node.static_host:
                        config["hosts"][node_id]["ssh_host"] = node.dynamic_host
                        updated = True

                    if node.dynamic_port and node.dynamic_port != node.static_port:
                        config["hosts"][node_id]["ssh_port"] = node.dynamic_port
                        updated = True

            if updated:
                # Add timestamp comment
                config["# Last updated by dynamic_registry"] = datetime.now().isoformat()

                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                logger.info("Updated YAML config with dynamic addresses")
                return True

            return False

        except (OSError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to update YAML config: {e}")
            return False


# Global registry instance
_registry: DynamicHostRegistry | None = None


def get_registry() -> DynamicHostRegistry:
    """Get or create the global dynamic host registry."""
    global _registry
    if _registry is None:
        _registry = DynamicHostRegistry()
    return _registry
