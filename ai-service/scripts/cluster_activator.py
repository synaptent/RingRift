#!/usr/bin/env python3
"""Cluster Activator - Auto-activate all nodes in the cluster.

December 2025 - Automated cluster node activation for long-term unattended operation.

This script:
1. Reads distributed_hosts.yaml for node configuration
2. Checks health status of all nodes via SSH and P2P health endpoints
3. Activates nodes that are not responding:
   - SSH activation for Lambda/Hetzner nodes
   - Vast.ai API relaunch for Vast instances
   - Vultr API for Vultr instances
4. Emits node_activated events for tracking

Usage:
    # Check cluster status (dry run)
    python scripts/cluster_activator.py --check

    # Activate all nodes
    python scripts/cluster_activator.py --activate

    # Activate only specific provider
    python scripts/cluster_activator.py --activate --provider vast

    # Continuous monitoring mode (check every 5 minutes)
    python scripts/cluster_activator.py --watch --interval 300
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Add ai-service to path for imports
AI_SERVICE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SSH_FAILED = "ssh_failed"
    P2P_FAILED = "p2p_failed"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class ProviderType(Enum):
    """Cloud provider types."""
    LAMBDA = "lambda"
    VAST = "vast"
    VULTR = "vultr"
    HETZNER = "hetzner"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_key: str = "~/.ssh/id_cluster"
    ssh_port: int = 22
    tailscale_ip: str | None = None
    ringrift_path: str = "~/ringrift/ai-service"
    venv_activate: str = "source ~/ringrift/ai-service/venv/bin/activate"
    provider: ProviderType = ProviderType.UNKNOWN
    status: str = "unknown"
    p2p_voter: bool = False
    worker_port: int = 8766
    vast_instance_id: str | None = None
    vultr_id: str | None = None
    aws_instance_id: str | None = None
    gpu: str = ""
    memory_gb: int = 0
    role: str = ""
    raw_config: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, name: str, config: dict) -> "NodeConfig":
        """Create NodeConfig from YAML configuration."""
        # Determine provider from name or config
        provider = ProviderType.UNKNOWN
        if name.startswith("lambda-"):
            provider = ProviderType.LAMBDA
        elif name.startswith("vast-") or "vast_instance_id" in config:
            provider = ProviderType.VAST
        elif name.startswith("vultr-") or "vultr_id" in config:
            provider = ProviderType.VULTR
        elif name.startswith("hetzner-"):
            provider = ProviderType.HETZNER
        elif name in ("mac-studio", "mbp-new"):
            provider = ProviderType.LOCAL

        return cls(
            name=name,
            ssh_host=config.get("ssh_host", ""),
            ssh_user=config.get("ssh_user", "ubuntu"),
            ssh_key=config.get("ssh_key", "~/.ssh/id_cluster"),
            ssh_port=config.get("ssh_port", 22),
            tailscale_ip=config.get("tailscale_ip"),
            ringrift_path=config.get("ringrift_path", "~/ringrift/ai-service"),
            venv_activate=config.get("venv_activate", ""),
            provider=provider,
            status=config.get("status", "unknown"),
            p2p_voter=config.get("p2p_voter", False),
            worker_port=config.get("worker_port", 8766),
            vast_instance_id=config.get("vast_instance_id"),
            vultr_id=config.get("vultr_id"),
            aws_instance_id=config.get("aws_instance_id"),
            gpu=config.get("gpu", ""),
            memory_gb=config.get("memory_gb", 0),
            role=config.get("role", ""),
            raw_config=config,
        )


@dataclass
class NodeHealthResult:
    """Result of a health check for a node."""
    node: NodeConfig
    status: NodeStatus
    ssh_ok: bool = False
    p2p_ok: bool = False
    latency_ms: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)


class ClusterActivator:
    """Activates and monitors cluster nodes."""

    def __init__(
        self,
        config_path: str | None = None,
        ssh_timeout: int = 10,
        p2p_timeout: int = 5,
    ):
        """Initialize the cluster activator.

        Args:
            config_path: Path to distributed_hosts.yaml
            ssh_timeout: SSH connection timeout in seconds
            p2p_timeout: P2P health check timeout in seconds
        """
        self.config_path = config_path or str(AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml")
        self.ssh_timeout = ssh_timeout
        self.p2p_timeout = p2p_timeout

        self._nodes: dict[str, NodeConfig] = {}
        self._health_cache: dict[str, NodeHealthResult] = {}
        self._providers: dict[ProviderType, Any] = {}

    def load_config(self) -> dict[str, NodeConfig]:
        """Load node configuration from YAML file.

        Returns:
            Dict mapping node name to NodeConfig
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        nodes = {}

        # Load hosts section
        hosts = config.get("hosts", {})
        for name, node_config in hosts.items():
            nodes[name] = NodeConfig.from_yaml(name, node_config)

        # Load vultr section
        vultr_section = config.get("vultr", {})
        for name, node_config in vultr_section.items():
            nodes[name] = NodeConfig.from_yaml(name, node_config)

        self._nodes = nodes
        logger.info(f"Loaded {len(nodes)} nodes from {self.config_path}")
        return nodes

    async def check_ssh_health(self, node: NodeConfig) -> tuple[bool, float, str]:
        """Check SSH connectivity to a node.

        Args:
            node: Node configuration

        Returns:
            Tuple of (success, latency_ms, error_message)
        """
        if not node.ssh_host:
            return False, 0.0, "No SSH host configured"

        # Build SSH command
        ssh_key = Path(node.ssh_key).expanduser()
        ssh_args = [
            "ssh",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
        ]

        if ssh_key.exists():
            ssh_args.extend(["-i", str(ssh_key)])

        if node.ssh_port != 22:
            ssh_args.extend(["-p", str(node.ssh_port)])

        ssh_args.extend([f"{node.ssh_user}@{node.ssh_host}", "echo ok"])

        start_time = time.time()
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_args,
                    capture_output=True,
                    timeout=self.ssh_timeout + 5,
                    text=True,
                )
            )
            latency_ms = (time.time() - start_time) * 1000

            if result.returncode == 0:
                return True, latency_ms, ""
            else:
                return False, latency_ms, result.stderr.strip() or "SSH failed"

        except subprocess.TimeoutExpired:
            return False, (time.time() - start_time) * 1000, "SSH timeout"
        except Exception as e:
            return False, (time.time() - start_time) * 1000, str(e)

    async def check_p2p_health(self, node: NodeConfig) -> tuple[bool, str]:
        """Check P2P daemon health via HTTP endpoint.

        Args:
            node: Node configuration

        Returns:
            Tuple of (success, error_message)
        """
        # Use tailscale IP if available, otherwise SSH host
        host = node.tailscale_ip or node.ssh_host
        if not host:
            return False, "No host for P2P check"

        url = f"http://{host}:{node.worker_port}/health"

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.p2p_timeout)) as resp:
                    if resp.status == 200:
                        return True, ""
                    else:
                        return False, f"HTTP {resp.status}"
        except ImportError:
            # Fall back to curl
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["curl", "-s", "--connect-timeout", str(self.p2p_timeout), url],
                        capture_output=True,
                        timeout=self.p2p_timeout + 2,
                    )
                )
                return result.returncode == 0, "" if result.returncode == 0 else "curl failed"
            except Exception as e:
                return False, str(e)
        except Exception as e:
            return False, str(e)

    async def check_node_health(self, node: NodeConfig) -> NodeHealthResult:
        """Check overall health of a node.

        Args:
            node: Node configuration

        Returns:
            NodeHealthResult with health status
        """
        # Check if node is marked as terminated
        if node.status == "terminated":
            return NodeHealthResult(
                node=node,
                status=NodeStatus.TERMINATED,
                error="Node marked as terminated",
            )

        # Check SSH first
        ssh_ok, latency_ms, ssh_error = await self.check_ssh_health(node)

        # Check P2P if SSH is OK
        p2p_ok = False
        p2p_error = ""
        if ssh_ok and node.p2p_voter:
            p2p_ok, p2p_error = await self.check_p2p_health(node)

        # Determine overall status
        if ssh_ok and (p2p_ok or not node.p2p_voter):
            status = NodeStatus.HEALTHY
            error = ""
        elif not ssh_ok:
            status = NodeStatus.SSH_FAILED
            error = ssh_error
        else:
            status = NodeStatus.P2P_FAILED
            error = p2p_error

        result = NodeHealthResult(
            node=node,
            status=status,
            ssh_ok=ssh_ok,
            p2p_ok=p2p_ok,
            latency_ms=latency_ms,
            error=error,
        )

        self._health_cache[node.name] = result
        return result

    async def check_all_nodes(
        self,
        provider_filter: ProviderType | None = None,
        skip_terminated: bool = True,
    ) -> list[NodeHealthResult]:
        """Check health of all nodes.

        Args:
            provider_filter: Only check nodes from this provider
            skip_terminated: Skip nodes marked as terminated

        Returns:
            List of NodeHealthResult for all checked nodes
        """
        if not self._nodes:
            self.load_config()

        nodes_to_check = []
        for node in self._nodes.values():
            if skip_terminated and node.status == "terminated":
                continue
            if provider_filter and node.provider != provider_filter:
                continue
            nodes_to_check.append(node)

        logger.info(f"Checking {len(nodes_to_check)} nodes...")

        # Run health checks in parallel
        tasks = [self.check_node_health(node) for node in nodes_to_check]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(NodeHealthResult(
                    node=nodes_to_check[i],
                    status=NodeStatus.UNKNOWN,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    async def activate_node_ssh(self, node: NodeConfig) -> bool:
        """Activate a node via SSH (wake up P2P daemon).

        Args:
            node: Node configuration

        Returns:
            True if activation succeeded
        """
        if not node.ssh_host:
            logger.warning(f"Cannot activate {node.name}: no SSH host")
            return False

        # Build SSH command to start P2P daemon
        ssh_key = Path(node.ssh_key).expanduser()
        activate_cmd = (
            f"cd {node.ringrift_path} && "
            f"{node.venv_activate} && "
            f"nohup python -m app.distributed.p2p_worker --port {node.worker_port} "
            f"> logs/p2p_worker.log 2>&1 &"
        )

        ssh_args = [
            "ssh",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]

        if ssh_key.exists():
            ssh_args.extend(["-i", str(ssh_key)])

        if node.ssh_port != 22:
            ssh_args.extend(["-p", str(node.ssh_port)])

        ssh_args.extend([f"{node.ssh_user}@{node.ssh_host}", activate_cmd])

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ssh_args,
                    capture_output=True,
                    timeout=30,
                )
            )

            if result.returncode == 0:
                logger.info(f"Activated {node.name} via SSH")
                await self._emit_node_activated(node)
                return True
            else:
                logger.warning(f"Failed to activate {node.name}: SSH returned {result.returncode}")
                return False

        except Exception as e:
            logger.error(f"Failed to activate {node.name}: {e}")
            return False

    async def activate_vast_node(self, node: NodeConfig) -> bool:
        """Activate a Vast.ai node via API.

        Args:
            node: Node configuration

        Returns:
            True if activation succeeded
        """
        if not node.vast_instance_id:
            logger.warning(f"Cannot activate {node.name}: no Vast instance ID")
            return False

        try:
            from app.coordination.providers.vast_provider import VastProvider
            provider = VastProvider()

            if not provider.is_configured():
                logger.error("Vast.ai provider not configured (missing VAST_API_KEY)")
                return False

            # Start the instance
            success = await provider.start_instance(node.vast_instance_id)

            if success:
                logger.info(f"Started Vast.ai instance {node.vast_instance_id} for {node.name}")
                await self._emit_node_activated(node)
                return True
            else:
                logger.warning(f"Failed to start Vast.ai instance {node.vast_instance_id}")
                return False

        except ImportError:
            logger.warning("VastProvider not available")
            return False
        except Exception as e:
            logger.error(f"Failed to activate Vast.ai node {node.name}: {e}")
            return False

    async def activate_vultr_node(self, node: NodeConfig) -> bool:
        """Activate a Vultr node via API.

        Args:
            node: Node configuration

        Returns:
            True if activation succeeded
        """
        if not node.vultr_id:
            logger.warning(f"Cannot activate {node.name}: no Vultr ID")
            return False

        try:
            from app.coordination.providers.vultr_provider import VultrProvider
            provider = VultrProvider()

            if not provider.is_configured():
                logger.error("Vultr provider not configured (missing VULTR_API_KEY)")
                return False

            # Start the instance
            success = await provider.start_instance(node.vultr_id)

            if success:
                logger.info(f"Started Vultr instance {node.vultr_id} for {node.name}")
                await self._emit_node_activated(node)
                return True
            else:
                logger.warning(f"Failed to start Vultr instance {node.vultr_id}")
                return False

        except ImportError:
            logger.warning("VultrProvider not available")
            return False
        except Exception as e:
            logger.error(f"Failed to activate Vultr node {node.name}: {e}")
            return False

    async def activate_node(self, node: NodeConfig) -> bool:
        """Activate a node using the appropriate method for its provider.

        Args:
            node: Node configuration

        Returns:
            True if activation succeeded
        """
        if node.provider == ProviderType.VAST:
            return await self.activate_vast_node(node)
        elif node.provider == ProviderType.VULTR:
            return await self.activate_vultr_node(node)
        else:
            # Try SSH activation for Lambda, Hetzner, and unknown providers
            return await self.activate_node_ssh(node)

    async def activate_unhealthy_nodes(
        self,
        health_results: list[NodeHealthResult],
        dry_run: bool = False,
    ) -> dict[str, bool]:
        """Activate all unhealthy nodes.

        Args:
            health_results: Results from check_all_nodes
            dry_run: If True, only log what would be done

        Returns:
            Dict mapping node name to activation success
        """
        activation_results = {}

        unhealthy = [r for r in health_results if r.status in (
            NodeStatus.SSH_FAILED,
            NodeStatus.P2P_FAILED,
            NodeStatus.UNHEALTHY,
        )]

        if not unhealthy:
            logger.info("All nodes are healthy, no activation needed")
            return activation_results

        logger.info(f"Found {len(unhealthy)} unhealthy nodes to activate")

        for result in unhealthy:
            node = result.node
            if dry_run:
                logger.info(f"[DRY RUN] Would activate {node.name} ({node.provider.value})")
                activation_results[node.name] = True
            else:
                success = await self.activate_node(node)
                activation_results[node.name] = success

        return activation_results

    async def _emit_node_activated(self, node: NodeConfig) -> None:
        """Emit node_activated event for tracking.

        Args:
            node: Activated node configuration
        """
        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEvent, DataEventType

            bus = get_event_bus()
            if bus:
                event = DataEvent(
                    event_type=DataEventType.NODE_ACTIVATED,
                    payload={
                        "node_name": node.name,
                        "provider": node.provider.value,
                        "ssh_host": node.ssh_host,
                        "gpu": node.gpu,
                        "timestamp": time.time(),
                    },
                    source="cluster_activator",
                )
                await bus.publish(event)
                logger.debug(f"Emitted NODE_ACTIVATED event for {node.name}")
        except Exception as e:
            logger.debug(f"Failed to emit node_activated event: {e}")

    def print_status_report(self, results: list[NodeHealthResult]) -> None:
        """Print a formatted status report.

        Args:
            results: Health check results to report
        """
        # Group by status
        by_status: dict[NodeStatus, list[NodeHealthResult]] = {}
        for result in results:
            by_status.setdefault(result.status, []).append(result)

        print("\n" + "=" * 60)
        print("CLUSTER STATUS REPORT")
        print("=" * 60)

        # Summary
        total = len(results)
        healthy = len(by_status.get(NodeStatus.HEALTHY, []))
        print(f"\nTotal nodes checked: {total}")
        print(f"Healthy: {healthy} ({healthy/total*100:.1f}%)" if total > 0 else "Healthy: 0")
        print()

        # By status
        status_order = [
            NodeStatus.HEALTHY,
            NodeStatus.SSH_FAILED,
            NodeStatus.P2P_FAILED,
            NodeStatus.UNHEALTHY,
            NodeStatus.TERMINATED,
            NodeStatus.UNKNOWN,
        ]

        for status in status_order:
            nodes = by_status.get(status, [])
            if not nodes:
                continue

            print(f"\n{status.value.upper()} ({len(nodes)}):")
            print("-" * 40)
            for result in sorted(nodes, key=lambda r: r.node.name):
                node = result.node
                extra = ""
                if result.latency_ms > 0:
                    extra = f" ({result.latency_ms:.0f}ms)"
                if result.error:
                    extra += f" - {result.error[:40]}"
                print(f"  {node.name:30} {node.provider.value:8}{extra}")

        print("\n" + "=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Activate and monitor cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check cluster status (dry run)",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate unhealthy nodes",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous monitoring mode",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval for watch mode (seconds)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["lambda", "vast", "vultr", "hetzner"],
        help="Only check/activate nodes from this provider",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to distributed_hosts.yaml",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--include-terminated",
        action="store_true",
        help="Include terminated nodes in checks",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create activator
    activator = ClusterActivator(config_path=args.config)
    activator.load_config()

    # Parse provider filter
    provider_filter = None
    if args.provider:
        provider_filter = ProviderType(args.provider)

    if args.watch:
        # Continuous monitoring mode
        logger.info(f"Starting watch mode with {args.interval}s interval")
        while True:
            try:
                results = await activator.check_all_nodes(
                    provider_filter=provider_filter,
                    skip_terminated=not args.include_terminated,
                )
                activator.print_status_report(results)

                if args.activate:
                    await activator.activate_unhealthy_nodes(results)

                await asyncio.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Interrupted, exiting")
                break
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                await asyncio.sleep(60)

    else:
        # One-shot mode
        results = await activator.check_all_nodes(
            provider_filter=provider_filter,
            skip_terminated=not args.include_terminated,
        )
        activator.print_status_report(results)

        if args.activate:
            activation_results = await activator.activate_unhealthy_nodes(
                results,
                dry_run=args.check,
            )
            if activation_results:
                print("\nActivation Results:")
                for name, success in activation_results.items():
                    status = "SUCCESS" if success else "FAILED"
                    print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
