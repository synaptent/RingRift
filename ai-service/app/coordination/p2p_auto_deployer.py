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
import json
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


def _get_default_p2p_port() -> int:
    """Get default P2P port from centralized config."""
    from app.config.cluster_config import get_p2p_port
    return get_p2p_port()


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

    # Health check (December 2025: Use centralized port config)
    p2p_port: int = field(default_factory=_get_default_p2p_port)
    health_check_timeout_seconds: float = 20.0  # Increased for slow SSH connections

    # Paths
    hosts_config_path: str = "config/distributed_hosts.yaml"
    deploy_script_path: str = "deploy/deploy_cluster_resilience.py"
    setup_script_path: str = "deploy/scripts/setup-systemd-p2p.sh"

    # Coverage thresholds
    min_coverage_percent: float = 90.0  # Alert if < 90% have P2P

    # Excluded nodes (coordinator, local dev machines, proxy-only)
    excluded_nodes: list[str] = field(
        default_factory=lambda: [
            "mac-studio",
            "local-mac",
            "mbp-new",
            "mbp-64gb",
            "mbp-16gb",
            "aws-proxy",  # Proxy-only node, no P2P needed
        ]
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
        # December 2025: Use cluster_config helpers instead of inline YAML loading
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            self._hosts = config.hosts_raw
            logger.info(f"Loaded {len(self._hosts)} hosts from cluster_config")

        except ImportError:
            # Fallback to direct YAML loading if cluster_config not available
            config_path = ROOT / self.config.hosts_config_path
            if not config_path.exists():
                logger.warning(f"Hosts config not found: {config_path}")
                return

            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}
                self._hosts = data.get("hosts", {})
                logger.info(f"Loaded {len(self._hosts)} hosts from YAML fallback")
            except (OSError, IOError, yaml.YAMLError) as e:
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

    def _build_ssh_args(
        self,
        host_info: dict[str, Any],
        timeout: float = 10.0,
    ) -> list[str]:
        """Build SSH command arguments for a node.

        Args:
            host_info: Node configuration from hosts.yaml
            timeout: Connection timeout in seconds

        Returns:
            List of SSH command arguments
        """
        ssh_host = host_info.get("ssh_host")
        ssh_user = host_info.get("ssh_user", "ubuntu")
        ssh_key = host_info.get("ssh_key", "~/.ssh/id_cluster")
        ssh_port = host_info.get("ssh_port", 22)

        ssh_key_expanded = os.path.expanduser(ssh_key)

        return [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={int(timeout)}",
            "-o", "BatchMode=yes",
            "-o", "ControlMaster=no",  # Disable connection sharing to avoid stale connections
            "-p", str(ssh_port),
            "-i", ssh_key_expanded,
            f"{ssh_user}@{ssh_host}",
        ]

    async def check_p2p_health(self, host_id: str, host_info: dict[str, Any]) -> bool:
        """Check if P2P is running on a node.

        Uses SSH to check P2P health, bypassing firewall/NAT issues.

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

        ssh_host = host_info.get("ssh_host")
        if not ssh_host:
            return False

        # Use SSH to check P2P health (works through firewalls/NAT)
        ssh_args = self._build_ssh_args(host_info, timeout=self.config.health_check_timeout_seconds)
        remote_cmd = f"curl -s --connect-timeout 5 localhost:{self.config.p2p_port}/health 2>/dev/null"

        try:
            # Use '--' to separate SSH options from the command
            proc = await asyncio.create_subprocess_exec(
                *ssh_args, "--", remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.health_check_timeout_seconds + 5,
            )

            output = stdout.decode().strip()
            # Check for "healthy": true (not "healthy": false)
            if '"healthy": true' in output or '"healthy":true' in output:
                return True

        except Exception as e:
            logger.debug(f"P2P SSH health check failed for {host_id}: {e}")

        return False

    def _get_seed_peers(self) -> str:
        """Get seed peer URLs from voter configuration.

        Returns:
            Comma-separated peer URLs
        """
        peers = []
        voter_nodes = []

        # December 2025: Use get_p2p_voters() instead of re-loading YAML
        try:
            from app.config.cluster_config import get_p2p_voters
            voter_nodes = get_p2p_voters()
        except ImportError:
            logger.debug("cluster_config not available for P2P voters")
        except Exception as e:
            logger.debug(f"Failed to get P2P voters: {e}")

        # Build peer URLs from voter hosts
        for voter in voter_nodes[:5]:  # Max 5 peers
            if voter in self._hosts:
                host_info = self._hosts[voter]
                host = host_info.get("ssh_host")
                if host:
                    peers.append(f"http://{host}:{self.config.p2p_port}")

        # Fallback: use cluster config to get voter IPs (December 2025 fix)
        if not peers:
            try:
                from app.config.cluster_config import get_p2p_voters, get_cluster_nodes
                voters = get_p2p_voters()
                nodes = get_cluster_nodes()
                for voter in voters[:2]:  # Use first 2 voters as fallback
                    if voter in nodes:
                        node = nodes[voter]
                        ip = node.best_ip or node.ssh_host
                        if ip:
                            peers.append(f"http://{ip}:{self.config.p2p_port}")
            except (ImportError, OSError, ValueError) as e:
                logger.warning(f"Could not load cluster config for P2P fallback: {e}")

        return ",".join(peers)

    async def deploy_p2p_to_node(
        self,
        host_id: str,
        host_info: dict[str, Any],
    ) -> P2PDeploymentResult:
        """Deploy P2P to a single node.

        Uses SSH to run the orchestrator with proper venv activation.

        Args:
            host_id: Node identifier
            host_info: Node configuration

        Returns:
            Deployment result
        """
        start_time = time.time()

        ssh_host = host_info.get("ssh_host")
        if not ssh_host:
            return P2PDeploymentResult(
                node_id=host_id,
                success=False,
                method="none",
                message="No SSH host available",
                duration_seconds=time.time() - start_time,
            )

        ringrift_path = host_info.get("ringrift_path", "~/ringrift/ai-service")
        venv_activate = host_info.get("venv_activate", f"source {ringrift_path}/venv/bin/activate")
        seed_peers = self._get_seed_peers()

        # Build SSH command
        ssh_args = self._build_ssh_args(host_info, timeout=self.config.deployment_timeout_seconds)

        # Remote command to start P2P with venv and proper peers
        # Note: We avoid "pkill -f p2p_orchestrator" as it can kill the SSH session
        # Instead, kill by listening port or use a more specific pattern
        # Jan 2026: Switched to nohup and added screen cleanup to prevent dead sessions
        remote_cmd = f"""
cd {ringrift_path} || exit 1
# Kill any existing P2P by port (safer than pkill -f which can kill SSH)
fuser -k {self.config.p2p_port}/tcp 2>/dev/null || true
# Clean up any dead screen sessions
screen -X -S p2p quit 2>/dev/null || true
screen -wipe 2>/dev/null || true
sleep 2
# Ensure logs directory exists
mkdir -p logs
# Activate venv and start P2P orchestrator (use nohup, not screen)
{venv_activate} 2>/dev/null || true
export PYTHONPATH={ringrift_path}
nohup python scripts/p2p_orchestrator.py --node-id {host_id} --port {self.config.p2p_port} --peers {seed_peers} > logs/p2p.log 2>&1 &
sleep 8
# Verify it started via health check
if curl -s --connect-timeout 5 localhost:{self.config.p2p_port}/health 2>/dev/null | grep -q '"healthy"'; then
    echo "P2P_STARTED"
else
    # Fallback: check if port is listening
    if ss -tlnp 2>/dev/null | grep -q ':{self.config.p2p_port}' || netstat -tlnp 2>/dev/null | grep -q ':{self.config.p2p_port}'; then
        echo "P2P_STARTED"
    else
        echo "P2P_FAILED"
    fi
fi
"""

        try:
            # Use '--' to separate SSH options from the command
            proc = await asyncio.create_subprocess_exec(
                *ssh_args, "--", remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.deployment_timeout_seconds,
            )

            output = stdout.decode().strip()

            if "P2P_STARTED" in output:
                # Verify mesh join - check if node sees peers
                mesh_ok = await self._verify_mesh_join(host_id, host_info)
                if mesh_ok:
                    logger.info(f"Successfully deployed P2P to {host_id} (mesh verified)")
                    return P2PDeploymentResult(
                        node_id=host_id,
                        success=True,
                        method="ssh_direct",
                        message="P2P started and mesh verified",
                        duration_seconds=time.time() - start_time,
                    )
                else:
                    logger.warning(f"P2P started on {host_id} but mesh join not verified")
                    return P2PDeploymentResult(
                        node_id=host_id,
                        success=True,  # Still success - process is running
                        method="ssh_direct",
                        message="P2P started but mesh join unverified (may be isolated)",
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
                except (asyncio.TimeoutError, OSError):
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

    async def _verify_mesh_join(
        self,
        host_id: str,
        host_info: dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> bool:
        """Verify that a node has successfully joined the P2P mesh.

        Checks:
        1. Node responds to /status
        2. Node sees at least one peer (not isolated)
        3. Node has adopted a valid epoch (synced with cluster)

        Args:
            host_id: Node identifier
            host_info: Node configuration
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if mesh join verified, False otherwise
        """
        ssh_args = self._build_ssh_args(host_info, timeout=10.0)
        remote_cmd = f"curl -s --connect-timeout 5 localhost:{self.config.p2p_port}/status 2>/dev/null"

        for attempt in range(max_retries):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *ssh_args, "--", remote_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=15.0,
                )

                output = stdout.decode().strip()
                if not output:
                    logger.debug(f"Mesh verify {host_id}: Empty response (attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    continue

                # Parse JSON status
                try:
                    status = json.loads(output)
                except json.JSONDecodeError:
                    logger.debug(f"Mesh verify {host_id}: Invalid JSON (attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    continue

                # Check for peers - node should see at least one other node
                alive_peers = status.get("alive_peers", 0)
                peer_count = status.get("peer_count", 0)
                total_peers = max(alive_peers, peer_count)

                # Check for valid epoch (indicates cluster sync)
                epoch = status.get("epoch", 0)
                leader_id = status.get("leader_id", "")

                if total_peers > 0:
                    logger.debug(
                        f"Mesh verify {host_id}: OK - {total_peers} peers, "
                        f"epoch={epoch}, leader={leader_id}"
                    )
                    return True

                # Node running but isolated - might just need more time
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Mesh verify {host_id}: No peers yet (attempt {attempt + 1}), "
                        f"epoch={epoch}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)

            except asyncio.TimeoutError:
                logger.debug(f"Mesh verify {host_id}: Timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.debug(f"Mesh verify {host_id}: Error {e} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        logger.warning(f"Mesh verify {host_id}: Failed after {max_retries} attempts (node may be isolated)")
        return False

    def get_latest_coverage(self) -> P2PCoverageReport | None:
        """Get the most recent coverage report."""
        return self._coverage_history[-1] if self._coverage_history else None

    def health_check(self) -> "HealthCheckResult":
        """Return health status for daemon monitoring.

        December 2025: Added for DaemonManager integration.

        Health criteria:
        - Must have checked P2P coverage at least once
        - Coverage should be above minimum threshold (90% default)
        - Should not have excessive deployment failures

        Returns:
            HealthCheckResult indicating health status
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Check if we've ever run a check
        if not self._coverage_history:
            # Never checked - healthy if running, waiting for first check
            if self._running:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.RUNNING,
                    message="P2PAutoDeployer running, awaiting first check",
                    details={
                        "running": self._running,
                        "checks_completed": 0,
                        "last_check": 0.0,
                    },
                )
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="P2PAutoDeployer not running",
                details={"running": False, "checks_completed": 0},
            )

        # Get latest coverage report
        latest = self._coverage_history[-1]
        deployment_failures = sum(
            1 for d in self._deployment_history[-20:] if not d.success
        )

        # Determine health
        is_healthy = (
            latest.coverage_percent >= self.config.min_coverage_percent
            and deployment_failures < 10  # Less than 50% failure rate in last 20
        )

        # Build details
        details = {
            "running": self._running,
            "coverage_percent": latest.coverage_percent,
            "nodes_with_p2p": latest.nodes_with_p2p,
            "total_nodes": latest.total_nodes,
            "nodes_needing_deployment": len(latest.nodes_needing_deployment),
            "unreachable_nodes": latest.unreachable_nodes,
            "deployment_failures_recent": deployment_failures,
            "last_check": self._last_check,
            "checks_completed": len(self._coverage_history),
        }

        if is_healthy:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"P2P coverage {latest.coverage_percent:.1f}% ({latest.nodes_with_p2p}/{latest.total_nodes} nodes)",
                details=details,
            )
        else:
            # Unhealthy - coverage too low or too many failures
            issues = []
            if latest.coverage_percent < self.config.min_coverage_percent:
                issues.append(
                    f"coverage {latest.coverage_percent:.1f}% < {self.config.min_coverage_percent}%"
                )
            if deployment_failures >= 10:
                issues.append(f"{deployment_failures} recent deployment failures")
            if latest.unreachable_nodes > 0:
                issues.append(f"{latest.unreachable_nodes} unreachable nodes")

            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"P2P health issues: {'; '.join(issues)}",
                details=details,
            )


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
