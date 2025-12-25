"""P2P Auto-Deployer - Ensures P2P network runs on all cluster nodes.

This module solves the critical gap where P2P orchestrator is not consistently
deployed across all cluster nodes. It:

1. Periodically checks which nodes are missing P2P
2. Auto-deploys P2P to nodes that need it
3. Reports P2P coverage metrics
4. Integrates with UnifiedNodeHealthDaemon for recovery

Root Cause Analysis:
- Manual deployment required (`deploy_cluster_resilience.py --apply`)
- No periodic remediation when P2P crashes
- Lambda nodes lack automatic startup scripts
- Vast.ai onstart.sh exists but may not run consistently

Solution:
- Periodic health checks for P2P on all known nodes
- SSH-based deployment using existing setup scripts
- Integration with node health daemon for holistic monitoring

Usage:
    # Standalone
    python -m app.coordination.p2p_auto_deployer

    # As part of UnifiedNodeHealthDaemon
    deployer = P2PAutoDeployer(health_orchestrator)
    await deployer.check_and_deploy()
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class P2PDeploymentConfig:
    """Configuration for P2P auto-deployment."""

    # Check intervals
    check_interval_seconds: float = 300.0  # Check every 5 minutes

    # Deployment settings
    deployment_timeout_seconds: float = 120.0
    max_concurrent_deployments: int = 5
    retry_count: int = 2
    retry_delay_seconds: float = 30.0

    # Health check
    p2p_port: int = 8770
    health_check_timeout_seconds: float = 10.0

    # Paths
    hosts_config_path: str = "config/distributed_hosts.yaml"
    deploy_script_path: str = "deploy/deploy_cluster_resilience.py"
    setup_script_path: str = "deploy/scripts/setup-systemd-p2p.sh"

    # Coverage thresholds
    min_coverage_percent: float = 90.0  # Alert if < 90% have P2P

    # Excluded nodes (coordinator, local dev machines)
    excluded_nodes: list[str] = field(
        default_factory=lambda: ["mac-studio", "mbp-new", "mbp-64gb", "mbp-16gb"]
    )


@dataclass
class P2PDeploymentResult:
    """Result of a P2P deployment attempt."""

    node_id: str
    success: bool
    method: str  # "ssh_direct", "deploy_script", "already_running"
    message: str
    duration_seconds: float


@dataclass
class P2PCoverageReport:
    """Report on P2P coverage across cluster."""

    total_nodes: int
    nodes_with_p2p: int
    nodes_without_p2p: int
    unreachable_nodes: int
    coverage_percent: float
    nodes_needing_deployment: list[str]
    timestamp: float = field(default_factory=time.time)


class P2PAutoDeployer:
    """Automatically deploys P2P orchestrator to all cluster nodes.

    This solves the fundamental issue of P2P not being consistently
    deployed across the cluster by:

    1. Reading node inventory from distributed_hosts.yaml
    2. Checking P2P health on each node
    3. Deploying P2P to nodes missing it
    4. Reporting coverage metrics
    """

    def __init__(
        self,
        config: P2PDeploymentConfig | None = None,
        health_orchestrator: Any | None = None,
    ):
        """Initialize the auto-deployer.

        Args:
            config: Deployment configuration
            health_orchestrator: Optional HealthCheckOrchestrator for node state
        """
        self.config = config or P2PDeploymentConfig()
        self.health_orchestrator = health_orchestrator

        self._running = False
        self._last_check = 0.0
        self._deployment_history: list[P2PDeploymentResult] = []
        self._coverage_history: list[P2PCoverageReport] = []

        # Load hosts config
        self._hosts: dict[str, dict[str, Any]] = {}
        self._load_hosts_config()

    def _load_hosts_config(self) -> None:
        """Load hosts from distributed_hosts.yaml."""
        config_path = ROOT / self.config.hosts_config_path

        if not config_path.exists():
            logger.warning(f"Hosts config not found: {config_path}")
            return

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            self._hosts = data.get("hosts", {})
            logger.info(f"Loaded {len(self._hosts)} hosts from config")

        except Exception as e:
            logger.error(f"Failed to load hosts config: {e}")

    def _get_deployable_hosts(self) -> dict[str, dict[str, Any]]:
        """Get hosts that should have P2P deployed.

        Excludes:
        - Local dev machines
        - Nodes marked as retired
        - Nodes without SSH access
        """
        deployable = {}

        for host_id, host_info in self._hosts.items():
            # Skip excluded nodes
            if host_id in self.config.excluded_nodes:
                continue

            # Skip retired/offline nodes
            status = host_info.get("status", "")
            if status in ("retired", "offline", "terminated"):
                continue

            # Must have SSH access
            ssh_host = host_info.get("ssh_host") or host_info.get("tailscale_ip")
            if not ssh_host:
                continue

            deployable[host_id] = host_info

        return deployable

    async def check_p2p_health(self, host_id: str, host_info: dict[str, Any]) -> bool:
        """Check if P2P is running on a node.

        Args:
            host_id: Node identifier
            host_info: Node configuration from hosts.yaml

        Returns:
            True if P2P is healthy, False otherwise
        """
        # Use health orchestrator if available
        if self.health_orchestrator:
            health = self.health_orchestrator.get_node_health(host_id)
            if health and health.p2p_healthy:
                return True

        # Otherwise check directly via HTTP
        tailscale_ip = host_info.get("tailscale_ip")
        ssh_host = host_info.get("ssh_host")

        for host in [tailscale_ip, ssh_host]:
            if not host:
                continue

            url = f"http://{host}:{self.config.p2p_port}/health"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    "--connect-timeout", str(int(self.config.health_check_timeout_seconds)),
                    url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.health_check_timeout_seconds + 5,
                )

                if stdout.decode().strip() == "200":
                    return True

            except Exception as e:
                logger.debug(f"P2P health check failed for {host_id} via {host}: {e}")

        return False

    async def deploy_p2p_to_node(
        self,
        host_id: str,
        host_info: dict[str, Any],
    ) -> P2PDeploymentResult:
        """Deploy P2P to a single node.

        Uses SSH to run the setup script on the remote node.

        Args:
            host_id: Node identifier
            host_info: Node configuration

        Returns:
            Deployment result
        """
        start_time = time.time()

        ssh_host = host_info.get("tailscale_ip") or host_info.get("ssh_host")
        ssh_user = host_info.get("ssh_user", "ubuntu")
        ssh_key = host_info.get("ssh_key", "~/.ssh/id_cluster")
        ringrift_path = host_info.get("ringrift_path", "~/ringrift/ai-service")

        if not ssh_host:
            return P2PDeploymentResult(
                node_id=host_id,
                success=False,
                method="none",
                message="No SSH host available",
                duration_seconds=time.time() - start_time,
            )

        # Build SSH command
        ssh_key_expanded = os.path.expanduser(ssh_key)
        ssh_args = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={int(self.config.deployment_timeout_seconds)}",
            "-i", ssh_key_expanded,
            f"{ssh_user}@{ssh_host}",
        ]

        # Remote command to start P2P
        # We'll use the direct Python approach since it's more reliable
        remote_cmd = f"""
cd {ringrift_path}
# Kill any existing P2P
pkill -f p2p_orchestrator 2>/dev/null || true
sleep 2
# Start P2P orchestrator
nohup python3 scripts/p2p_orchestrator.py \\
    --node-id {host_id} \\
    --port {self.config.p2p_port} \\
    >> logs/p2p_orchestrator.log 2>&1 &
sleep 3
# Verify it started
if pgrep -f p2p_orchestrator > /dev/null; then
    echo "P2P_STARTED"
else
    echo "P2P_FAILED"
fi
"""

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_args, remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.deployment_timeout_seconds,
            )

            output = stdout.decode().strip()

            if "P2P_STARTED" in output:
                logger.info(f"Successfully deployed P2P to {host_id}")
                return P2PDeploymentResult(
                    node_id=host_id,
                    success=True,
                    method="ssh_direct",
                    message="P2P started successfully",
                    duration_seconds=time.time() - start_time,
                )
            else:
                return P2PDeploymentResult(
                    node_id=host_id,
                    success=False,
                    method="ssh_direct",
                    message=f"P2P failed to start: {output[-200:]}",
                    duration_seconds=time.time() - start_time,
                )

        except asyncio.TimeoutError:
            return P2PDeploymentResult(
                node_id=host_id,
                success=False,
                method="ssh_direct",
                message="SSH connection timeout",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return P2PDeploymentResult(
                node_id=host_id,
                success=False,
                method="ssh_direct",
                message=f"Deployment error: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def check_and_deploy(self) -> P2PCoverageReport:
        """Check P2P coverage and deploy to nodes missing it.

        Returns:
            Coverage report after deployment attempts
        """
        logger.info("=" * 50)
        logger.info("P2P AUTO-DEPLOYER: Checking cluster coverage")
        logger.info("=" * 50)

        self._last_check = time.time()

        # Get deployable hosts
        hosts = self._get_deployable_hosts()
        logger.info(f"Checking {len(hosts)} nodes for P2P health")

        # Check health of each node
        nodes_with_p2p: list[str] = []
        nodes_without_p2p: list[str] = []
        unreachable: list[str] = []

        # Check all nodes in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(10)

        async def check_node(host_id: str, host_info: dict) -> tuple[str, str]:
            async with semaphore:
                try:
                    is_healthy = await self.check_p2p_health(host_id, host_info)
                    return (host_id, "healthy" if is_healthy else "missing")
                except Exception:
                    return (host_id, "unreachable")

        tasks = [
            check_node(host_id, host_info)
            for host_id, host_info in hosts.items()
        ]
        results = await asyncio.gather(*tasks)

        for host_id, status in results:
            if status == "healthy":
                nodes_with_p2p.append(host_id)
            elif status == "missing":
                nodes_without_p2p.append(host_id)
            else:
                unreachable.append(host_id)

        logger.info(f"P2P Status: {len(nodes_with_p2p)} healthy, "
                   f"{len(nodes_without_p2p)} missing, "
                   f"{len(unreachable)} unreachable")

        # Deploy to nodes missing P2P
        deployments: list[P2PDeploymentResult] = []

        if nodes_without_p2p:
            logger.info(f"Deploying P2P to {len(nodes_without_p2p)} nodes: {nodes_without_p2p}")

            deploy_semaphore = asyncio.Semaphore(self.config.max_concurrent_deployments)

            async def deploy_with_limit(host_id: str) -> P2PDeploymentResult:
                async with deploy_semaphore:
                    return await self.deploy_p2p_to_node(host_id, hosts[host_id])

            deploy_tasks = [deploy_with_limit(h) for h in nodes_without_p2p]
            deployments = await asyncio.gather(*deploy_tasks)

            # Log results
            successful = [d for d in deployments if d.success]
            failed = [d for d in deployments if not d.success]

            logger.info(f"Deployment results: {len(successful)} succeeded, {len(failed)} failed")

            for d in failed:
                logger.warning(f"  Failed: {d.node_id} - {d.message}")

            # Update counts after deployment
            for d in successful:
                nodes_with_p2p.append(d.node_id)
                nodes_without_p2p.remove(d.node_id)

        # Calculate final coverage
        total = len(hosts)
        coverage_percent = (len(nodes_with_p2p) / total * 100) if total > 0 else 0

        report = P2PCoverageReport(
            total_nodes=total,
            nodes_with_p2p=len(nodes_with_p2p),
            nodes_without_p2p=len(nodes_without_p2p),
            unreachable_nodes=len(unreachable),
            coverage_percent=coverage_percent,
            nodes_needing_deployment=nodes_without_p2p + unreachable,
        )

        self._coverage_history.append(report)
        self._deployment_history.extend(deployments)

        # Log summary
        logger.info("=" * 50)
        logger.info(f"P2P COVERAGE: {report.coverage_percent:.1f}% "
                   f"({report.nodes_with_p2p}/{report.total_nodes} nodes)")
        if report.nodes_needing_deployment:
            logger.warning(f"Nodes still needing P2P: {report.nodes_needing_deployment}")
        logger.info("=" * 50)

        return report

    async def run_daemon(self) -> None:
        """Run as a continuous daemon."""
        self._running = True

        logger.info("P2P Auto-Deployer starting...")
        logger.info(f"Check interval: {self.config.check_interval_seconds}s")
        logger.info(f"Min coverage threshold: {self.config.min_coverage_percent}%")

        while self._running:
            try:
                report = await self.check_and_deploy()

                # Alert if coverage below threshold
                if report.coverage_percent < self.config.min_coverage_percent:
                    logger.error(
                        f"P2P coverage below threshold! "
                        f"{report.coverage_percent:.1f}% < {self.config.min_coverage_percent}%"
                    )

                await asyncio.sleep(self.config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"P2P Auto-Deployer error: {e}")
                await asyncio.sleep(60)

        logger.info("P2P Auto-Deployer stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

    def get_latest_coverage(self) -> P2PCoverageReport | None:
        """Get the most recent coverage report."""
        return self._coverage_history[-1] if self._coverage_history else None


async def main() -> None:
    """Run P2P auto-deployer standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    deployer = P2PAutoDeployer()

    try:
        # Single check or daemon mode based on args
        import argparse
        parser = argparse.ArgumentParser(description="P2P Auto-Deployer")
        parser.add_argument("--daemon", action="store_true", help="Run as daemon")
        parser.add_argument("--interval", type=float, default=300.0, help="Check interval")
        args = parser.parse_args()

        if args.daemon:
            deployer.config.check_interval_seconds = args.interval
            await deployer.run_daemon()
        else:
            report = await deployer.check_and_deploy()
            print(f"\nCoverage: {report.coverage_percent:.1f}%")
            print(f"Nodes with P2P: {report.nodes_with_p2p}/{report.total_nodes}")
            if report.nodes_needing_deployment:
                print(f"Nodes needing deployment: {report.nodes_needing_deployment}")

    except KeyboardInterrupt:
        deployer.stop()


if __name__ == "__main__":
    asyncio.run(main())
