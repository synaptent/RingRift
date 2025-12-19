"""Unified Node Inventory - Multi-CLI Discovery for RingRift Cluster.

Discovers nodes from ALL sources (Vast, Tailscale, Lambda, Hetzner) and maintains
a unified registry. Designed to run as part of the P2P orchestrator leader loop.

Discovery Sources:
- Vast CLI: `vastai show instances --raw`
- Tailscale CLI: `tailscale status --json`
- Lambda: HTTP probe to known IPs from distributed_hosts.yaml
- Hetzner CLI: `hcloud server list --output json` (if available)

Usage:
    from app.coordination.unified_inventory import UnifiedInventory, get_inventory

    inventory = get_inventory()
    await inventory.discover_all()  # Run discovery
    idle_nodes = inventory.get_idle_nodes(gpu_threshold=10)

Integration with P2P orchestrator:
    - Leader runs discovery every 60 seconds
    - Idle detection runs every 30 seconds
    - Work is auto-assigned to idle nodes
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.utils.yaml_utils import safe_load_yaml

from app.utils.paths import AI_SERVICE_ROOT

from app.utils.env_config import env

logger = logging.getLogger(__name__)
DISTRIBUTED_HOSTS_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
CLUSTER_NODES_PATH = AI_SERVICE_ROOT / "config" / "cluster_nodes.env"

# Discovery configuration (from centralized env_config)
DISCOVERY_INTERVAL = env.discovery_interval
IDLE_CHECK_INTERVAL = env.idle_check_interval
IDLE_GPU_THRESHOLD = env.idle_gpu_threshold
AUTO_ASSIGN_ENABLED = env.auto_assign_enabled

# Tailscale CGNAT range
TAILSCALE_CGNAT_PREFIX = "100."

# GPU role mapping (from vast_p2p_sync.py)
GPU_ROLES = {
    "RTX 3070": "gpu_selfplay",
    "RTX 3060": "gpu_selfplay",
    "RTX 3060 Ti": "cpu_selfplay",
    "RTX 2060S": "gpu_selfplay",
    "RTX 2060 SUPER": "gpu_selfplay",
    "RTX 2080 Ti": "gpu_selfplay",
    "RTX 4060 Ti": "gpu_selfplay",
    "RTX 4080S": "nn_training_primary",
    "RTX 4080 SUPER": "nn_training_primary",
    "RTX 5070": "nn_training_primary",
    "RTX 5080": "nn_training_primary",
    "RTX 5090": "nn_training_primary",
    "A10": "nn_training_primary",
    "A40": "nn_training_primary",
    "A100": "nn_training_primary",
    "H100": "nn_training_primary",
    "GH200": "nn_training_primary",
}


@dataclass
class DiscoveredNode:
    """Node discovered from external sources."""
    node_id: str
    host: str  # IP or hostname
    port: int = 8770  # P2P port
    source: str = "unknown"  # vast, tailscale, lambda, hetzner

    # Connection info
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = "root"
    tailscale_ip: str = ""

    # Hardware info
    gpu_name: str = ""
    num_gpus: int = 1
    memory_gb: int = 0
    vcpus: int = 0

    # Status
    status: str = "unknown"  # running, offline, etc.
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0

    # Metadata
    role: str = "selfplay"
    vast_instance_id: str = ""
    last_seen: float = field(default_factory=time.time)
    p2p_healthy: bool = False
    retired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "source": self.source,
            "tailscale_ip": self.tailscale_ip,
            "gpu_name": self.gpu_name,
            "status": self.status,
            "gpu_percent": self.gpu_percent,
            "selfplay_jobs": self.selfplay_jobs,
            "role": self.role,
        }


class UnifiedInventory:
    """Discovers and tracks nodes from all sources."""

    def __init__(self):
        self._nodes: Dict[str, DiscoveredNode] = {}
        self._lock = asyncio.Lock()
        self._last_discovery: float = 0.0
        self._distributed_hosts: Dict[str, Any] = {}
        self._cluster_nodes: Dict[str, str] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load configuration files."""
        # Load distributed_hosts.yaml
        config = safe_load_yaml(DISTRIBUTED_HOSTS_PATH, default={}, log_errors=True)
        if config:
            self._distributed_hosts = config.get("hosts", {})
            logger.info(f"Loaded {len(self._distributed_hosts)} hosts from distributed_hosts.yaml")

        # Load cluster_nodes.env
        if CLUSTER_NODES_PATH.exists():
            try:
                with open(CLUSTER_NODES_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            self._cluster_nodes[key.strip()] = value.strip().strip('"')
                logger.info(f"Loaded cluster_nodes.env with {len(self._cluster_nodes)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cluster_nodes.env: {e}")

    async def discover_all(self) -> Dict[str, DiscoveredNode]:
        """Run all discovery methods in parallel and merge results."""
        self._last_discovery = time.time()

        # Run all discovery methods concurrently
        results = await asyncio.gather(
            self._discover_vast(),
            self._discover_tailscale(),
            self._discover_lambda(),
            self._discover_hetzner(),
            return_exceptions=True
        )

        # Collect all discovered nodes
        all_nodes: List[DiscoveredNode] = []
        source_names = ["vast", "tailscale", "lambda", "hetzner"]

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Discovery from {source_names[i]} failed: {result}")
            elif result:
                all_nodes.extend(result)
                logger.info(f"Discovered {len(result)} nodes from {source_names[i]}")

        # Merge and deduplicate
        async with self._lock:
            merged = self._merge_nodes(all_nodes)
            self._nodes = merged

        logger.info(f"Unified inventory: {len(self._nodes)} total nodes")
        return self._nodes.copy()

    def _merge_nodes(self, nodes: List[DiscoveredNode]) -> Dict[str, DiscoveredNode]:
        """Merge nodes from multiple sources, preferring most detailed info."""
        merged: Dict[str, DiscoveredNode] = {}

        for node in nodes:
            key = node.node_id.lower()

            # Also check by tailscale IP for deduplication
            if node.tailscale_ip:
                ip_key = f"ip:{node.tailscale_ip}"
                if ip_key in merged:
                    # Merge with existing
                    existing = merged[ip_key]
                    node = self._merge_two_nodes(existing, node)
                merged[ip_key] = node

            if key in merged:
                # Merge with existing
                existing = merged[key]
                node = self._merge_two_nodes(existing, node)

            merged[key] = node

        # Remove IP-based keys (keep only node_id keys)
        return {k: v for k, v in merged.items() if not k.startswith("ip:")}

    def _merge_two_nodes(self, existing: DiscoveredNode, new: DiscoveredNode) -> DiscoveredNode:
        """Merge two nodes, preferring more complete information."""
        # Keep newer status info
        if new.last_seen > existing.last_seen:
            existing.gpu_percent = new.gpu_percent
            existing.cpu_percent = new.cpu_percent
            existing.selfplay_jobs = new.selfplay_jobs
            existing.training_jobs = new.training_jobs
            existing.p2p_healthy = new.p2p_healthy
            existing.status = new.status
            existing.last_seen = new.last_seen

        # Fill in missing info
        if not existing.tailscale_ip and new.tailscale_ip:
            existing.tailscale_ip = new.tailscale_ip
        if not existing.gpu_name and new.gpu_name:
            existing.gpu_name = new.gpu_name
        if existing.memory_gb == 0 and new.memory_gb > 0:
            existing.memory_gb = new.memory_gb
        if not existing.ssh_host and new.ssh_host:
            existing.ssh_host = new.ssh_host
            existing.ssh_port = new.ssh_port

        return existing

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    async def _discover_vast(self) -> List[DiscoveredNode]:
        """Discover nodes from Vast.ai CLI."""
        nodes: List[DiscoveredNode] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "vastai", "show", "instances", "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.debug(f"vastai CLI error: {stderr.decode()}")
                return nodes

            instances = json.loads(stdout.decode())

            for inst in instances:
                if inst.get("actual_status") != "running":
                    continue

                instance_id = inst.get("id", 0)
                gpu_name = inst.get("gpu_name", "Unknown")
                num_gpus = inst.get("num_gpus", 1) or 1

                nodes.append(DiscoveredNode(
                    node_id=f"vast-{instance_id}",
                    host=inst.get("ssh_host", ""),
                    port=8770,
                    source="vast",
                    ssh_host=inst.get("ssh_host", ""),
                    ssh_port=inst.get("ssh_port", 22),
                    ssh_user="root",
                    gpu_name=gpu_name,
                    num_gpus=num_gpus,
                    memory_gb=int((inst.get("cpu_ram", 0) or 0) / 1024),
                    vcpus=int(inst.get("cpu_cores_effective", 0) or 0),
                    status="running",
                    role=GPU_ROLES.get(gpu_name, "gpu_selfplay"),
                    vast_instance_id=str(instance_id),
                ))

        except FileNotFoundError:
            logger.debug("vastai CLI not installed")
        except asyncio.TimeoutError:
            logger.warning("vastai CLI timed out")
        except Exception as e:
            logger.warning(f"Vast discovery failed: {e}")

        return nodes

    async def _discover_tailscale(self) -> List[DiscoveredNode]:
        """Discover nodes from Tailscale CLI."""
        nodes: List[DiscoveredNode] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "tailscale", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

            if proc.returncode != 0:
                logger.debug(f"tailscale CLI error: {stderr.decode()}")
                return nodes

            status = json.loads(stdout.decode())
            peers = status.get("Peer", {})

            for peer_id, peer_info in peers.items():
                # Get Tailscale IP (prefer the 100.x.x.x address)
                tailscale_ips = peer_info.get("TailscaleIPs", [])
                ts_ip = ""
                for ip in tailscale_ips:
                    if ip.startswith(TAILSCALE_CGNAT_PREFIX):
                        ts_ip = ip
                        break

                if not ts_ip:
                    continue

                # Get hostname as node_id
                hostname = peer_info.get("HostName", "").lower()
                if not hostname:
                    continue

                # Check if online
                online = peer_info.get("Online", False)
                if not online:
                    continue

                # Cross-reference with distributed_hosts.yaml for additional info
                host_info = self._find_host_by_tailscale_ip(ts_ip)
                gpu_name = ""
                memory_gb = 0
                role = "selfplay"

                if host_info:
                    gpu_name = host_info.get("gpu", "")
                    memory_gb = host_info.get("memory_gb", 0)
                    role = host_info.get("role", "selfplay")

                # Map common hostnames to node IDs
                node_id = self._hostname_to_node_id(hostname, ts_ip)

                nodes.append(DiscoveredNode(
                    node_id=node_id,
                    host=ts_ip,
                    port=8770,
                    source="tailscale",
                    tailscale_ip=ts_ip,
                    gpu_name=gpu_name,
                    memory_gb=memory_gb,
                    status="online" if online else "offline",
                    role=role,
                ))

        except FileNotFoundError:
            logger.debug("tailscale CLI not installed")
        except asyncio.TimeoutError:
            logger.warning("tailscale CLI timed out")
        except Exception as e:
            logger.warning(f"Tailscale discovery failed: {e}")

        return nodes

    async def _discover_lambda(self) -> List[DiscoveredNode]:
        """Discover Lambda nodes by probing known IPs from config."""
        nodes: List[DiscoveredNode] = []

        # Get Lambda nodes from distributed_hosts.yaml
        lambda_hosts = {
            name: info for name, info in self._distributed_hosts.items()
            if name.startswith("lambda-")
        }

        # Probe each host
        probe_tasks = []
        for name, info in lambda_hosts.items():
            ts_ip = info.get("tailscale_ip", "")
            if ts_ip:
                probe_tasks.append(self._probe_p2p_health(name, ts_ip, info))

        if probe_tasks:
            results = await asyncio.gather(*probe_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, DiscoveredNode):
                    nodes.append(result)

        return nodes

    async def _probe_p2p_health(self, node_id: str, host: str, info: Dict[str, Any]) -> Optional[DiscoveredNode]:
        """Probe a node's P2P health endpoint."""
        try:
            import urllib.request

            url = f"http://{host}:8770/health"
            loop = asyncio.get_event_loop()

            def fetch():
                with urllib.request.urlopen(url, timeout=5) as resp:
                    return json.loads(resp.read().decode())

            health = await asyncio.wait_for(loop.run_in_executor(None, fetch), timeout=10)

            return DiscoveredNode(
                node_id=node_id,
                host=host,
                port=8770,
                source="lambda",
                tailscale_ip=host if host.startswith(TAILSCALE_CGNAT_PREFIX) else "",
                gpu_name=info.get("gpu", "") or health.get("gpu_name", ""),
                memory_gb=info.get("memory_gb", 0),
                status="healthy",
                gpu_percent=float(health.get("gpu_percent", 0) or 0),
                cpu_percent=float(health.get("cpu_percent", 0) or 0),
                selfplay_jobs=int(health.get("selfplay_jobs", 0) or 0),
                training_jobs=int(health.get("training_jobs", 0) or 0),
                role=info.get("role", "selfplay"),
                p2p_healthy=True,
            )
        except Exception as e:
            logger.debug(f"Failed to probe {node_id} at {host}: {e}")
            return None

    async def _discover_hetzner(self) -> List[DiscoveredNode]:
        """Discover nodes from Hetzner Cloud CLI."""
        nodes: List[DiscoveredNode] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "hcloud", "server", "list", "--output", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.debug(f"hcloud CLI error: {stderr.decode()}")
                return nodes

            servers = json.loads(stdout.decode())

            for server in servers:
                if server.get("status") != "running":
                    continue

                name = server.get("name", "")
                server_type = server.get("server_type", {}).get("name", "")

                # Get public IP
                public_net = server.get("public_net", {})
                ipv4 = public_net.get("ipv4", {}).get("ip", "")

                if not ipv4:
                    continue

                nodes.append(DiscoveredNode(
                    node_id=f"hetzner-{name}",
                    host=ipv4,
                    port=8770,
                    source="hetzner",
                    ssh_host=ipv4,
                    ssh_port=22,
                    status="running",
                    role="cpu_cmaes" if "cx" in server_type else "selfplay",
                ))

        except FileNotFoundError:
            logger.debug("hcloud CLI not installed")
        except asyncio.TimeoutError:
            logger.warning("hcloud CLI timed out")
        except Exception as e:
            logger.warning(f"Hetzner discovery failed: {e}")

        return nodes

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_host_by_tailscale_ip(self, ts_ip: str) -> Optional[Dict[str, Any]]:
        """Find host info from distributed_hosts.yaml by Tailscale IP."""
        for name, info in self._distributed_hosts.items():
            if info.get("tailscale_ip") == ts_ip:
                return info
            if info.get("ssh_host") == ts_ip:
                return info
        return None

    def _hostname_to_node_id(self, hostname: str, ts_ip: str) -> str:
        """Convert Tailscale hostname to canonical node_id."""
        hostname = hostname.lower().replace(".tail", "").replace(".ts.net", "")

        # Check if we have a mapping in distributed_hosts
        for name, info in self._distributed_hosts.items():
            if info.get("tailscale_ip") == ts_ip:
                return name

        # Common patterns
        if "gh200" in hostname:
            # Extract gh200 letter suffix if present
            for letter in "abcdefghijklmnop":
                if f"gh200{letter}" in hostname or f"gh200-{letter}" in hostname:
                    return f"lambda-gh200-{letter}"
            return f"lambda-{hostname}"

        if "h100" in hostname:
            return f"lambda-{hostname}"

        if "a10" in hostname:
            return f"lambda-{hostname}"

        return hostname

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_all_nodes(self) -> Dict[str, DiscoveredNode]:
        """Get all discovered nodes."""
        return self._nodes.copy()

    def get_node(self, node_id: str) -> Optional[DiscoveredNode]:
        """Get a specific node by ID."""
        return self._nodes.get(node_id.lower())

    def get_idle_nodes(self, gpu_threshold: float = IDLE_GPU_THRESHOLD) -> List[DiscoveredNode]:
        """Get nodes with low GPU utilization and no running jobs."""
        idle = []
        for node in self._nodes.values():
            if node.retired:
                continue
            # Consider idle if GPU < threshold and no selfplay/training jobs
            if node.gpu_percent < gpu_threshold and node.selfplay_jobs == 0 and node.training_jobs == 0:
                # Skip CPU-only nodes or nodes without GPU info
                if node.gpu_name and "CPU" not in node.gpu_name.upper():
                    idle.append(node)
        return idle

    def get_nodes_by_source(self, source: str) -> List[DiscoveredNode]:
        """Get nodes discovered from a specific source."""
        return [n for n in self._nodes.values() if n.source == source]

    def get_healthy_nodes(self) -> List[DiscoveredNode]:
        """Get nodes that are healthy and can accept work."""
        return [n for n in self._nodes.values() if n.p2p_healthy and not n.retired]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of inventory status."""
        by_source = {}
        idle_count = 0
        healthy_count = 0

        for node in self._nodes.values():
            by_source[node.source] = by_source.get(node.source, 0) + 1
            if node.gpu_percent < IDLE_GPU_THRESHOLD and node.selfplay_jobs == 0:
                idle_count += 1
            if node.p2p_healthy and not node.retired:
                healthy_count += 1

        return {
            "total_nodes": len(self._nodes),
            "by_source": by_source,
            "idle_nodes": idle_count,
            "healthy_nodes": healthy_count,
            "last_discovery": self._last_discovery,
            "discovery_interval": DISCOVERY_INTERVAL,
        }


# Singleton instance
_inventory: Optional[UnifiedInventory] = None


def get_inventory() -> UnifiedInventory:
    """Get the singleton UnifiedInventory instance."""
    global _inventory
    if _inventory is None:
        _inventory = UnifiedInventory()
    return _inventory
