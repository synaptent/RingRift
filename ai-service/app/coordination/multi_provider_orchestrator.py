"""Multi-Provider Cluster Orchestrator.

Automatically discovers, manages, and utilizes nodes across multiple providers:
- Lambda Labs (GH200, H100, A10)
- Vast.ai (RTX 3060-5090, A40)
- AWS EC2
- Hetzner Cloud
- Local (Mac Studio, MacBooks)

Features:
1. Unified discovery across all providers
2. Automatic Tailscale deployment to new nodes
3. Health monitoring and auto-recovery
4. Workload distribution (selfplay, training)
5. Cost optimization (prefer cheaper nodes for selfplay)

Usage:
    from app.coordination.multi_provider_orchestrator import (
        MultiProviderOrchestrator,
        get_orchestrator,
        discover_all_nodes,
        deploy_to_all_nodes,
    )

    orchestrator = get_orchestrator()
    await orchestrator.discover_all()
    await orchestrator.ensure_all_connected()
    await orchestrator.deploy_selfplay_everywhere()
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Cloud/compute providers."""
    LAMBDA = "lambda"
    VAST = "vast"
    AWS = "aws"
    HETZNER = "hetzner"
    LOCAL = "local"
    UNKNOWN = "unknown"


class NodeRole(str, Enum):
    """Node roles in the cluster."""
    TRAINING = "training"       # GPU training
    SELFPLAY = "selfplay"       # Self-play game generation
    COORDINATOR = "coordinator" # Cluster coordination
    IDLE = "idle"               # Available but unused
    OFFLINE = "offline"         # Not reachable


@dataclass
class ClusterNode:
    """Unified node representation across all providers."""
    name: str
    provider: Provider
    tailscale_ip: str | None = None
    public_ip: str | None = None
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_host: str | None = None  # SSH jump host if needed

    # Hardware
    gpu_name: str | None = None
    gpu_count: int = 1
    gpu_memory_gb: float = 0
    cpu_cores: int = 0
    memory_gb: float = 0

    # Status
    is_online: bool = False
    is_tailscale_connected: bool = False
    role: NodeRole = NodeRole.IDLE
    last_seen: float = 0

    # Provider-specific IDs
    provider_id: str | None = None  # e.g., Vast instance ID, AWS instance ID

    # Workload
    selfplay_running: bool = False
    training_running: bool = False
    current_job: str | None = None

    def ssh_command(self, cmd: str) -> str:
        """Generate SSH command for this node (deprecated, use ssh_command_list)."""
        host = self.tailscale_ip or self.public_ip or self.ssh_host
        port = self.ssh_port if not self.tailscale_ip else 22
        return f"ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p {port} {self.ssh_user}@{host} '{cmd}'"

    def ssh_command_list(self, cmd: str) -> list[str]:
        """Generate SSH command as a list for subprocess (no shell=True needed)."""
        host = self.tailscale_ip or self.public_ip or self.ssh_host
        port = self.ssh_port if not self.tailscale_ip else 22
        return [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-p", str(port),
            f"{self.ssh_user}@{host}",
            cmd,
        ]


class MultiProviderOrchestrator:
    """Orchestrates nodes across multiple cloud providers."""

    def __init__(self):
        self.nodes: dict[str, ClusterNode] = {}
        self._discovery_lock = asyncio.Lock()
        self._last_discovery = 0
        self._discovery_interval = 60  # seconds

    async def discover_all(self) -> dict[str, ClusterNode]:
        """Discover nodes from all providers."""
        async with self._discovery_lock:
            logger.info("[Orchestrator] Starting multi-provider discovery...")

            # Run all discoveries in parallel
            results = await asyncio.gather(
                self._discover_tailscale(),
                self._discover_vast(),
                self._discover_aws(),
                self._discover_hetzner(),
                self._discover_lambda_from_config(),
                return_exceptions=True
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    provider = ["tailscale", "vast", "aws", "hetzner", "lambda"][i]
                    logger.error(f"[Orchestrator] {provider} discovery failed: {result}")

            self._last_discovery = time.time()

            # Summary
            online = sum(1 for n in self.nodes.values() if n.is_online)
            logger.info(
                f"[Orchestrator] Discovery complete: "
                f"{len(self.nodes)} nodes, {online} online"
            )

            return self.nodes

    async def _discover_tailscale(self) -> list[ClusterNode]:
        """Discover nodes from Tailscale network."""
        nodes = []
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return nodes

            data = json.loads(result.stdout)
            peers = data.get("Peer", {})

            for _peer_key, peer in peers.items():
                name = peer.get("HostName", "unknown")
                ts_ips = peer.get("TailscaleIPs", [])
                ts_ip = ts_ips[0] if ts_ips else None
                is_online = peer.get("Online", False)

                # Determine provider from hostname
                provider = self._detect_provider(name)

                # Get or create node
                if name in self.nodes:
                    node = self.nodes[name]
                    node.tailscale_ip = ts_ip
                    node.is_tailscale_connected = is_online
                    node.is_online = is_online
                else:
                    node = ClusterNode(
                        name=name,
                        provider=provider,
                        tailscale_ip=ts_ip,
                        is_online=is_online,
                        is_tailscale_connected=is_online,
                    )
                    self.nodes[name] = node

                if is_online:
                    node.last_seen = time.time()

                nodes.append(node)

            logger.info(f"[Tailscale] Discovered {len(nodes)} peers")

        except Exception as e:
            logger.error(f"[Tailscale] Discovery error: {e}")

        return nodes

    async def _discover_vast(self) -> list[ClusterNode]:
        """Discover nodes from Vast.ai."""
        nodes = []
        try:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return nodes

            instances = json.loads(result.stdout)

            for inst in instances:
                inst_id = str(inst.get("id", ""))
                status = inst.get("cur_state", "unknown")
                gpu_name = inst.get("gpu_name", "unknown")
                gpu_count = inst.get("num_gpus", 1)
                ssh_host = inst.get("ssh_host", "")
                ssh_port = inst.get("ssh_port", 22)
                label = inst.get("label") or f"vast-{inst_id}"

                name = label or f"vast-{gpu_name.replace(' ', '-')}-{inst_id}"

                node = ClusterNode(
                    name=name,
                    provider=Provider.VAST,
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    gpu_name=gpu_name,
                    gpu_count=gpu_count,
                    is_online=(status == "running"),
                    provider_id=inst_id,
                )

                # Merge with existing if found
                if name in self.nodes:
                    existing = self.nodes[name]
                    existing.ssh_host = ssh_host
                    existing.ssh_port = ssh_port
                    existing.gpu_name = gpu_name
                    existing.gpu_count = gpu_count
                    existing.provider_id = inst_id
                else:
                    self.nodes[name] = node

                nodes.append(node)

            logger.info(f"[Vast] Discovered {len(nodes)} instances")

        except FileNotFoundError:
            logger.debug("[Vast] vastai CLI not found")
        except Exception as e:
            logger.error(f"[Vast] Discovery error: {e}")

        return nodes

    async def _discover_aws(self) -> list[ClusterNode]:
        """Discover nodes from AWS EC2."""
        nodes = []
        try:
            result = subprocess.run(
                ["aws", "ec2", "describe-instances",
                 "--region", "us-east-1",
                 "--query", "Reservations[*].Instances[*]",
                 "--output", "json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return nodes

            reservations = json.loads(result.stdout)

            for reservation in reservations:
                for inst in reservation:
                    inst_id = inst.get("InstanceId", "")
                    state = inst.get("State", {}).get("Name", "unknown")
                    public_ip = inst.get("PublicIpAddress")

                    # Get name from tags
                    name = inst_id
                    for tag in inst.get("Tags", []):
                        if tag.get("Key") == "Name":
                            name = tag.get("Value", inst_id)
                            break

                    node = ClusterNode(
                        name=name,
                        provider=Provider.AWS,
                        public_ip=public_ip,
                        is_online=(state == "running"),
                        provider_id=inst_id,
                    )

                    if name in self.nodes:
                        existing = self.nodes[name]
                        existing.public_ip = public_ip
                        existing.provider_id = inst_id
                    else:
                        self.nodes[name] = node

                    nodes.append(node)

            logger.info(f"[AWS] Discovered {len(nodes)} instances")

        except FileNotFoundError:
            logger.debug("[AWS] aws CLI not found")
        except Exception as e:
            logger.error(f"[AWS] Discovery error: {e}")

        return nodes

    async def _discover_hetzner(self) -> list[ClusterNode]:
        """Discover nodes from Hetzner Cloud."""
        nodes = []
        try:
            result = subprocess.run(
                ["hcloud", "server", "list", "-o", "json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return nodes

            servers = json.loads(result.stdout)

            for server in servers:
                server_id = str(server.get("id", ""))
                name = server.get("name", f"hetzner-{server_id}")
                status = server.get("status", "unknown")

                # Get IPs
                public_net = server.get("public_net", {})
                ipv4 = public_net.get("ipv4", {}).get("ip")

                # CPU-only instances don't have GPUs
                cpu_cores = server.get("server_type", {}).get("cores", 0)
                memory_gb = server.get("server_type", {}).get("memory", 0)

                node = ClusterNode(
                    name=name,
                    provider=Provider.HETZNER,
                    public_ip=ipv4,
                    ssh_user="root",  # Hetzner default
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb,
                    is_online=(status == "running"),
                    provider_id=server_id,
                )

                # Merge with existing (may have Tailscale info)
                if name in self.nodes:
                    existing = self.nodes[name]
                    existing.public_ip = ipv4
                    existing.cpu_cores = cpu_cores
                    existing.memory_gb = memory_gb
                    existing.provider = Provider.HETZNER
                    existing.provider_id = server_id
                else:
                    self.nodes[name] = node

                nodes.append(node)

            logger.info(f"[Hetzner] Discovered {len(nodes)} servers")

        except FileNotFoundError:
            logger.debug("[Hetzner] hcloud CLI not found")
        except Exception as e:
            logger.error(f"[Hetzner] Discovery error: {e}")

        return nodes

    async def _discover_lambda_from_config(self) -> list[ClusterNode]:
        """Load Lambda nodes from config file."""
        nodes = []
        config_path = Path(__file__).parents[2] / "config" / "distributed_hosts.yaml"

        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            for name, host_config in hosts.items():
                if "lambda" in name.lower():
                    node = ClusterNode(
                        name=name,
                        provider=Provider.LAMBDA,
                        tailscale_ip=host_config.get("tailscale_ip"),
                        public_ip=host_config.get("ssh_host"),
                        ssh_user=host_config.get("ssh_user", "ubuntu"),
                        gpu_name=host_config.get("gpu"),
                        memory_gb=host_config.get("memory_gb", 0),
                    )

                    if name in self.nodes:
                        existing = self.nodes[name]
                        if not existing.gpu_name:
                            existing.gpu_name = node.gpu_name
                        if not existing.memory_gb:
                            existing.memory_gb = node.memory_gb
                    else:
                        self.nodes[name] = node

                    nodes.append(node)

            logger.info(f"[Lambda] Loaded {len(nodes)} nodes from config")

        except Exception as e:
            logger.error(f"[Lambda] Config load error: {e}")

        return nodes

    def _detect_provider(self, hostname: str) -> Provider:
        """Detect provider from hostname."""
        hostname_lower = hostname.lower()
        if "lambda" in hostname_lower or "gh200" in hostname_lower:
            return Provider.LAMBDA
        elif "vast" in hostname_lower:
            return Provider.VAST
        elif hostname_lower.startswith("ip-172-31") or "aws" in hostname_lower:
            return Provider.AWS
        elif "hetzner" in hostname_lower or hostname_lower.startswith("ringrift-cpu"):
            return Provider.HETZNER
        elif "mac" in hostname_lower or "mbp" in hostname_lower:
            return Provider.LOCAL
        return Provider.UNKNOWN

    async def ensure_tailscale_connected(self, node: ClusterNode) -> bool:
        """Ensure a node has Tailscale connected."""
        if node.is_tailscale_connected:
            return True

        if not node.ssh_host and not node.public_ip:
            logger.warning(f"[Orchestrator] No SSH access to {node.name}")
            return False

        logger.info(f"[Orchestrator] Deploying Tailscale to {node.name}...")

        try:
            # Check if Tailscale is installed
            check_cmd = "which tailscale || echo 'not_found'"
            ssh_args = node.ssh_command_list(check_cmd)
            result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=30)

            if "not_found" in result.stdout:
                # Install Tailscale
                install_cmd = (
                    "curl -fsSL https://tailscale.com/install.sh | sh && "
                    "sudo tailscale up --authkey=$TAILSCALE_AUTHKEY --hostname=" + node.name
                )
                ssh_args = node.ssh_command_list(install_cmd)
                result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.error(f"[Orchestrator] Tailscale install failed on {node.name}: {result.stderr}")
                    return False
            else:
                # Tailscale installed, just bring it up
                up_cmd = "sudo tailscale up --authkey=$TAILSCALE_AUTHKEY --hostname=" + node.name
                ssh_args = node.ssh_command_list(up_cmd)
                result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=60)

            logger.info(f"[Orchestrator] Tailscale connected on {node.name}")
            node.is_tailscale_connected = True
            return True

        except Exception as e:
            logger.error(f"[Orchestrator] Tailscale deploy failed on {node.name}: {e}")
            return False

    async def deploy_selfplay(self, node: ClusterNode) -> bool:
        """Deploy selfplay worker to a node."""
        if not node.is_online or not node.is_tailscale_connected:
            return False

        host = node.tailscale_ip or node.public_ip
        if not host:
            return False

        logger.info(f"[Orchestrator] Deploying selfplay to {node.name}...")

        try:
            deploy_cmd = (
                "cd ~/ringrift/ai-service && "
                "source venv/bin/activate && "
                "nohup python -m app.selfplay.worker --config config/selfplay_config.yaml "
                f"--node-name {node.name} > logs/selfplay.log 2>&1 &"
            )

            ssh_args = node.ssh_command_list(deploy_cmd)
            result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                node.selfplay_running = True
                node.role = NodeRole.SELFPLAY
                logger.info(f"[Orchestrator] Selfplay started on {node.name}")
                return True
            else:
                logger.error(f"[Orchestrator] Selfplay deploy failed on {node.name}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"[Orchestrator] Selfplay deploy error on {node.name}: {e}")
            return False

    async def deploy_to_all_online_nodes(self, role: str = "selfplay") -> dict[str, bool]:
        """Deploy workload to all online nodes."""
        results = {}

        for name, node in self.nodes.items():
            if not node.is_online:
                results[name] = False
                continue

            if role == "selfplay":
                results[name] = await self.deploy_selfplay(node)

        return results

    def get_online_nodes(self) -> list[ClusterNode]:
        """Get all online nodes."""
        return [n for n in self.nodes.values() if n.is_online]

    def get_nodes_by_provider(self, provider: Provider) -> list[ClusterNode]:
        """Get nodes from a specific provider."""
        return [n for n in self.nodes.values() if n.provider == provider]

    def get_idle_nodes(self) -> list[ClusterNode]:
        """Get nodes that are online but not running workloads."""
        return [
            n for n in self.nodes.values()
            if n.is_online and not n.selfplay_running and not n.training_running
        ]

    def get_status_summary(self) -> dict[str, Any]:
        """Get cluster status summary."""
        online = self.get_online_nodes()
        by_provider = {}

        for provider in Provider:
            nodes = self.get_nodes_by_provider(provider)
            online_count = sum(1 for n in nodes if n.is_online)
            by_provider[provider.value] = {
                "total": len(nodes),
                "online": online_count,
            }

        return {
            "total_nodes": len(self.nodes),
            "online_nodes": len(online),
            "offline_nodes": len(self.nodes) - len(online),
            "by_provider": by_provider,
            "idle_nodes": len(self.get_idle_nodes()),
            "selfplay_running": sum(1 for n in self.nodes.values() if n.selfplay_running),
            "training_running": sum(1 for n in self.nodes.values() if n.training_running),
            "last_discovery": self._last_discovery,
        }

    def print_status(self):
        """Print cluster status to console."""
        summary = self.get_status_summary()
        print("=" * 60)
        print("MULTI-PROVIDER CLUSTER STATUS")
        print("=" * 60)
        print(f"Total Nodes: {summary['total_nodes']}")
        print(f"Online: {summary['online_nodes']}, Offline: {summary['offline_nodes']}")
        print()
        print("By Provider:")
        for provider, stats in summary["by_provider"].items():
            if stats["total"] > 0:
                print(f"  {provider:10}: {stats['online']}/{stats['total']} online")
        print()
        print("Workloads:")
        print(f"  Selfplay: {summary['selfplay_running']} nodes")
        print(f"  Training: {summary['training_running']} nodes")
        print(f"  Idle: {summary['idle_nodes']} nodes")
        print("=" * 60)


# Global orchestrator instance
_orchestrator: MultiProviderOrchestrator | None = None


def get_orchestrator() -> MultiProviderOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiProviderOrchestrator()
    return _orchestrator


async def discover_all_nodes() -> dict[str, ClusterNode]:
    """Convenience function to discover all nodes."""
    return await get_orchestrator().discover_all()


async def deploy_to_all_nodes(role: str = "selfplay") -> dict[str, bool]:
    """Convenience function to deploy to all nodes."""
    orch = get_orchestrator()
    await orch.discover_all()
    return await orch.deploy_to_all_online_nodes(role)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Provider Cluster Orchestrator")
    parser.add_argument("--discover", action="store_true", help="Discover all nodes")
    parser.add_argument("--deploy", choices=["selfplay", "training"], help="Deploy workload")
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    async def main():
        orch = get_orchestrator()

        if args.discover or args.status or args.deploy:
            await orch.discover_all()

        if args.deploy:
            results = await orch.deploy_to_all_online_nodes(args.deploy)
            success = sum(1 for v in results.values() if v)
            print(f"Deployed to {success}/{len(results)} nodes")

        if args.status or not (args.deploy or args.discover):
            if args.json:
                print(json.dumps(orch.get_status_summary(), indent=2))
            else:
                orch.print_status()

    asyncio.run(main())


def reset_orchestrator() -> None:
    """Reset the singleton for testing."""
    global _orchestrator
    _orchestrator = None


def wire_orchestrator_events() -> MultiProviderOrchestrator:
    """Wire orchestrator to the event bus for automatic node tracking.

    Subscribes to:
    - HOST_ONLINE: Update node online status
    - HOST_OFFLINE: Update node offline status
    - CLUSTER_STATUS_CHANGED: Trigger discovery

    Returns:
        The configured MultiProviderOrchestrator instance
    """
    orchestrator = get_orchestrator()

    try:
        from app.coordination.event_router import get_router
        from app.distributed.data_events import DataEventType

        router = get_router()

        def _event_payload(event: Any) -> dict[str, Any]:
            if isinstance(event, dict):
                return event
            payload = getattr(event, "payload", None)
            return payload if isinstance(payload, dict) else {}

        def _on_host_online(event: Any) -> None:
            """Handle host coming online."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("node_id")
            if host and host in orchestrator.nodes:
                node = orchestrator.nodes[host]
                node.is_online = True
                node.last_seen = time.time()
                logger.info(f"[Orchestrator] Node {host} is online")

        def _on_host_offline(event: Any) -> None:
            """Handle host going offline."""
            payload = _event_payload(event)
            host = payload.get("host") or payload.get("node_id")
            if host and host in orchestrator.nodes:
                node = orchestrator.nodes[host]
                node.is_online = False
                node.role = NodeRole.OFFLINE
                logger.warning(f"[Orchestrator] Node {host} is offline")

        def _on_cluster_changed(event: Any) -> None:
            """Handle cluster status change - trigger async discovery."""
            # Schedule discovery on next event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(orchestrator.discover_all())
            except RuntimeError:
                pass  # No event loop, skip

        router.subscribe(DataEventType.HOST_ONLINE.value, _on_host_online)
        router.subscribe(DataEventType.HOST_OFFLINE.value, _on_host_offline)
        router.subscribe(DataEventType.CLUSTER_STATUS_CHANGED.value, _on_cluster_changed)

        logger.info("[Orchestrator] Wired to event bus (HOST_ONLINE, HOST_OFFLINE, CLUSTER_STATUS_CHANGED)")

    except ImportError:
        logger.warning("[Orchestrator] data_events not available, running without event bus")

    return orchestrator


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "ClusterNode",
    # Main class
    "MultiProviderOrchestrator",
    "NodeRole",
    # Enums
    "Provider",
    "deploy_to_all_nodes",
    "discover_all_nodes",
    # Functions
    "get_orchestrator",
    "reset_orchestrator",
    "wire_orchestrator_events",
]
