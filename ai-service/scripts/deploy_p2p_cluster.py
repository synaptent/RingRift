#!/usr/bin/env python3
"""Cluster-wide P2P Orchestrator Deployment Script.

Deploys p2p_orchestrator to all nodes with p2p_enabled: true in distributed_hosts.yaml.

Features:
- Reads distributed_hosts.yaml to discover P2P-enabled nodes
- Syncs latest config/distributed_hosts.yaml to each node
- Kills existing p2p_orchestrator processes
- Starts p2p_orchestrator with correct --node-id
- Verifies health on port 8770
- Parallel deployment (max 5 concurrent by default)
- Dry-run mode for testing
- Node filtering support
- Detailed success/failure reporting

Usage:
    # Deploy to all P2P-enabled nodes
    python scripts/deploy_p2p_cluster.py

    # Dry run (preview actions)
    python scripts/deploy_p2p_cluster.py --dry-run

    # Deploy to specific nodes
    python scripts/deploy_p2p_cluster.py --nodes runpod-h100,vast-29129529

    # Skip config sync (use existing config on nodes)
    python scripts/deploy_p2p_cluster.py --skip-config-sync

    # Verbose output
    python scripts/deploy_p2p_cluster.py -v

Exit codes:
    0 - All deployments successful
    1 - Some deployments failed
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

from app.core.ssh import SSHClient, SSHConfig, get_ssh_client

logger = logging.getLogger(__name__)


# Constants
P2P_PORT = 8770
DEPLOYMENT_TIMEOUT = 120
MAX_CONCURRENT = 5
CONFIG_SYNC_TIMEOUT = 60
HEALTH_CHECK_TIMEOUT = 10
HEALTH_CHECK_RETRIES = 3
HEALTH_CHECK_RETRY_DELAY = 2.0

ROOT = Path(__file__).resolve().parent.parent
HOSTS_CONFIG = ROOT / "config" / "distributed_hosts.yaml"


@dataclass
class DeploymentResult:
    """Result of a single node deployment."""

    node_id: str
    success: bool
    stage: str  # "config_sync", "kill_existing", "start_p2p", "health_check"
    message: str
    duration_seconds: float
    config_synced: bool = False
    process_killed: bool = False
    p2p_started: bool = False
    p2p_healthy: bool = False
    error: str | None = None


@dataclass
class ClusterDeploymentSummary:
    """Summary of cluster-wide deployment."""

    total_nodes: int
    successful: int
    failed: int
    skipped: int
    unreachable: int
    results: list[DeploymentResult]
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (%)."""
        if self.total_nodes == 0:
            return 0.0
        return (self.successful / self.total_nodes) * 100.0


class P2PClusterDeployer:
    """Deploys P2P orchestrator across the entire cluster."""

    def __init__(
        self,
        dry_run: bool = False,
        skip_config_sync: bool = False,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        """Initialize deployer.

        Args:
            dry_run: If True, preview actions without executing
            skip_config_sync: If True, skip syncing distributed_hosts.yaml
            max_concurrent: Max concurrent deployments
        """
        self.dry_run = dry_run
        self.skip_config_sync = skip_config_sync
        self.max_concurrent = max_concurrent

        self.hosts_config: dict[str, Any] = {}
        self.p2p_voters: list[str] = []

    def load_hosts_config(self) -> None:
        """Load hosts configuration from distributed_hosts.yaml."""
        if not HOSTS_CONFIG.exists():
            raise FileNotFoundError(f"Config not found: {HOSTS_CONFIG}")

        try:
            with open(HOSTS_CONFIG) as f:
                data = yaml.safe_load(f) or {}

            self.hosts_config = data.get("hosts", {})
            self.p2p_voters = data.get("p2p_voters", [])

            logger.info(f"Loaded {len(self.hosts_config)} hosts from config")
            logger.info(f"P2P voters: {self.p2p_voters}")

        except Exception as e:
            raise RuntimeError(f"Failed to load hosts config: {e}") from e

    def get_p2p_enabled_nodes(self, node_filter: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Get nodes with p2p_enabled: true.

        Args:
            node_filter: Optional list of node IDs to deploy to (None = all)

        Returns:
            Dict of node_id -> node_config for P2P-enabled nodes
        """
        p2p_nodes = {}

        for node_id, node_config in self.hosts_config.items():
            # Check if P2P is enabled
            if not node_config.get("p2p_enabled", False):
                continue

            # Apply node filter if provided
            if node_filter and node_id not in node_filter:
                continue

            # Skip nodes with no SSH access
            ssh_host = node_config.get("ssh_host") or node_config.get("tailscale_ip")
            if not ssh_host:
                logger.warning(f"Node {node_id} has no SSH host configured, skipping")
                continue

            # Skip retired/terminated nodes
            status = node_config.get("status", "")
            if status in ("retired", "terminated", "offline"):
                logger.info(f"Node {node_id} status={status}, skipping")
                continue

            p2p_nodes[node_id] = node_config

        return p2p_nodes

    async def sync_config_to_node(
        self,
        node_id: str,
        client: SSHClient,
        timeout: int = CONFIG_SYNC_TIMEOUT,
    ) -> tuple[bool, str]:
        """Sync distributed_hosts.yaml to remote node.

        Args:
            node_id: Node identifier
            client: SSH client for the node
            timeout: Sync timeout in seconds

        Returns:
            Tuple of (success, message)
        """
        if self.dry_run:
            return True, "[DRY-RUN] Would sync config"

        if self.skip_config_sync:
            return True, "Config sync skipped (--skip-config-sync)"

        logger.debug(f"[{node_id}] Syncing distributed_hosts.yaml...")

        # Get remote path
        ringrift_path = client.config.work_dir or "~/ringrift/ai-service"
        remote_config_path = f"{ringrift_path}/config/distributed_hosts.yaml"

        # Use scp to copy config
        result = client.scp_to(
            str(HOSTS_CONFIG),
            remote_config_path,
            timeout=timeout,
        )

        if result.success:
            return True, "Config synced successfully"
        else:
            return False, f"Config sync failed: {result.stderr}"

    async def kill_existing_p2p(
        self,
        node_id: str,
        client: SSHClient,
    ) -> tuple[bool, str]:
        """Kill existing p2p_orchestrator processes on node.

        Also kills supervisor and keepalive processes that might restart the orchestrator.
        Uses SIGTERM first, then SIGKILL as fallback if processes don't terminate.

        Args:
            node_id: Node identifier
            client: SSH client for the node

        Returns:
            Tuple of (success, message)
        """
        if self.dry_run:
            return True, "[DRY-RUN] Would kill existing p2p_orchestrator"

        logger.debug(f"[{node_id}] Killing existing P2P processes (orchestrator, supervisor, keepalive, systemd)...")

        # Step 1: Stop systemd service if it exists (prevents restarts)
        # Use kill mode to force immediate stop without waiting for graceful shutdown
        # Bypass circuit breaker since we must execute this for deployment
        cmd_stop_systemd = (
            "systemctl kill --signal=SIGKILL ringrift-p2p 2>/dev/null || true; "
            "systemctl stop ringrift-p2p 2>/dev/null || true; "
            "systemctl disable ringrift-p2p 2>/dev/null || true"
        )
        await client.run_async(cmd_stop_systemd, timeout=60, bypass_circuit_breaker=True)

        # Step 2: Kill supervisor and keepalive to prevent restarts
        cmd_kill_managers = (
            "pkill -9 -f 'p2p_supervisor' 2>/dev/null || true; "
            "pkill -9 -f 'p2p_keepalive' 2>/dev/null || true; "
            "sleep 1"
        )
        await client.run_async(cmd_kill_managers, timeout=30, bypass_circuit_breaker=True)

        # Step 3: Try graceful SIGTERM for orchestrator, wait 5 seconds
        cmd_sigterm = (
            "pkill -f 'python.*p2p_orchestrator' || true; "
            "sleep 5; "
            "pgrep -f 'python.*p2p_orchestrator' | wc -l"
        )

        result = await client.run_async(cmd_sigterm, timeout=60, bypass_circuit_breaker=True)

        if not result.success:
            return False, f"Failed to kill processes: {result.stderr}"

        # Check if processes were killed
        remaining = result.stdout.strip()
        if remaining == "0":
            return True, "Killed existing processes (if any)"

        # Step 4: Processes still running - use SIGKILL
        logger.warning(f"[{node_id}] {remaining} processes still running after SIGTERM, using SIGKILL...")

        cmd_sigkill = (
            "pkill -9 -f 'python.*p2p_orchestrator' || true; "
            "sleep 3; "
            "pgrep -f 'python.*p2p_orchestrator' | wc -l"
        )

        result = await client.run_async(cmd_sigkill, timeout=30, bypass_circuit_breaker=True)

        if not result.success:
            return False, f"Failed to SIGKILL processes: {result.stderr}"

        remaining = result.stdout.strip()
        if remaining == "0":
            return True, "Killed existing processes with SIGKILL"
        else:
            return False, f"{remaining} processes still running after SIGKILL"

    async def start_p2p_orchestrator(
        self,
        node_id: str,
        client: SSHClient,
    ) -> tuple[bool, str]:
        """Start p2p_orchestrator on node.

        Args:
            node_id: Node identifier
            client: SSH client for the node

        Returns:
            Tuple of (success, message)
        """
        if self.dry_run:
            return True, f"[DRY-RUN] Would start p2p_orchestrator --node-id {node_id}"

        logger.debug(f"[{node_id}] Starting p2p_orchestrator...")

        # Build command to start p2p_orchestrator
        ringrift_path = client.config.work_dir or "~/ringrift/ai-service"
        venv_activate = client.config.venv_activate or f"source {ringrift_path}/venv/bin/activate"
        log_dir = f"{ringrift_path}/logs"

        cmd = (
            f"mkdir -p {log_dir} && "
            f"cd {ringrift_path} && "
            f"{venv_activate} && "
            f"nohup python -m app.p2p.p2p_orchestrator --node-id {node_id} "
            f"> {log_dir}/p2p_orchestrator.log 2>&1 & "
            f"echo $!"
        )

        result = await client.run_async(cmd, timeout=30)

        if not result.success:
            return False, f"Failed to start: {result.stderr}"

        # Extract PID
        pid = result.stdout.strip()
        if pid and pid.isdigit():
            return True, f"Started with PID {pid}"
        else:
            return False, f"No PID returned: {result.stdout}"

    async def check_p2p_health(
        self,
        node_id: str,
        node_config: dict[str, Any],
        retries: int = HEALTH_CHECK_RETRIES,
    ) -> tuple[bool, str]:
        """Check if p2p_orchestrator is responding on port 8770.

        Args:
            node_id: Node identifier
            node_config: Node configuration
            retries: Number of health check retries

        Returns:
            Tuple of (healthy, message)
        """
        if self.dry_run:
            return True, "[DRY-RUN] Would check health on port 8770"

        # Get node IP
        ssh_host = node_config.get("ssh_host") or node_config.get("tailscale_ip")
        if not ssh_host:
            return False, "No SSH host configured"

        # Extract IP (handle user@host format)
        if "@" in ssh_host:
            ssh_host = ssh_host.split("@", 1)[1]

        logger.debug(f"[{node_id}] Checking P2P health at {ssh_host}:{P2P_PORT}")

        # Retry health check with backoff
        last_error = "Unknown error"
        for attempt in range(retries):
            if attempt > 0:
                await asyncio.sleep(HEALTH_CHECK_RETRY_DELAY)

            try:
                # Try to fetch /status endpoint
                import urllib.request

                url = f"http://{ssh_host}:{P2P_PORT}/status"
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/json")

                with urllib.request.urlopen(req, timeout=HEALTH_CHECK_TIMEOUT) as response:
                    data = json.loads(response.read().decode())
                    status = data.get("status")

                    if status in ("follower", "candidate", "leader"):
                        return True, f"Healthy ({status})"
                    else:
                        return False, f"Unexpected status: {status}"

            except Exception as e:
                last_error = str(e)
                logger.debug(f"[{node_id}] Health check attempt {attempt+1}/{retries} failed: {e}")
                continue

        return False, f"Health check failed after {retries} attempts: {last_error}"

    async def deploy_to_node(
        self,
        node_id: str,
        node_config: dict[str, Any],
    ) -> DeploymentResult:
        """Deploy P2P orchestrator to a single node.

        Args:
            node_id: Node identifier
            node_config: Node configuration from hosts.yaml

        Returns:
            DeploymentResult with detailed status
        """
        start_time = time.time()
        stage = "init"
        error = None

        logger.info(f"[{node_id}] Starting deployment...")

        try:
            # Create SSH client with extended timeouts for high-latency connections
            config = SSHConfig(
                host=node_config.get("ssh_host", ""),
                port=node_config.get("ssh_port", 22),
                user=node_config.get("ssh_user", "root"),
                key_path=node_config.get("ssh_key"),
                work_dir=node_config.get("ringrift_path", "~/ringrift/ai-service"),
                venv_activate=node_config.get("venv_activate"),
                tailscale_ip=node_config.get("tailscale_ip"),
                connect_timeout=30,  # Increased for high-latency connections
                server_alive_interval=10,
                server_alive_count_max=3,
            )
            client = SSHClient(config)

            # Stage 1: Sync config
            stage = "config_sync"
            config_ok, config_msg = await self.sync_config_to_node(node_id, client)
            if not config_ok:
                raise RuntimeError(config_msg)

            # Stage 2: Kill existing processes
            stage = "kill_existing"
            kill_ok, kill_msg = await self.kill_existing_p2p(node_id, client)
            if not kill_ok:
                raise RuntimeError(kill_msg)

            # Stage 3: Start p2p_orchestrator
            stage = "start_p2p"
            start_ok, start_msg = await self.start_p2p_orchestrator(node_id, client)
            if not start_ok:
                raise RuntimeError(start_msg)

            # Stage 4: Health check
            stage = "health_check"
            health_ok, health_msg = await self.check_p2p_health(node_id, node_config)
            if not health_ok:
                raise RuntimeError(health_msg)

            # Success!
            duration = time.time() - start_time
            logger.info(f"[{node_id}] Deployment successful in {duration:.1f}s")

            return DeploymentResult(
                node_id=node_id,
                success=True,
                stage="completed",
                message="Deployment successful",
                duration_seconds=duration,
                config_synced=config_ok,
                process_killed=kill_ok,
                p2p_started=start_ok,
                p2p_healthy=health_ok,
            )

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            logger.error(f"[{node_id}] Deployment failed at {stage}: {error}")

            return DeploymentResult(
                node_id=node_id,
                success=False,
                stage=stage,
                message=f"Failed at {stage}",
                duration_seconds=duration,
                error=error,
            )

    async def deploy_to_cluster(
        self,
        node_filter: list[str] | None = None,
    ) -> ClusterDeploymentSummary:
        """Deploy P2P orchestrator to all eligible nodes.

        Args:
            node_filter: Optional list of node IDs to deploy to

        Returns:
            ClusterDeploymentSummary with results
        """
        start_time = time.time()

        # Load config
        self.load_hosts_config()

        # Get P2P-enabled nodes
        p2p_nodes = self.get_p2p_enabled_nodes(node_filter)

        if not p2p_nodes:
            logger.warning("No P2P-enabled nodes found")
            return ClusterDeploymentSummary(
                total_nodes=0,
                successful=0,
                failed=0,
                skipped=0,
                unreachable=0,
                results=[],
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Deploying to {len(p2p_nodes)} nodes (max {self.max_concurrent} concurrent)...")

        if self.dry_run:
            logger.info("[DRY-RUN] Preview mode - no changes will be made")

        # Create deployment tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def deploy_with_semaphore(node_id: str, node_config: dict[str, Any]) -> DeploymentResult:
            async with semaphore:
                return await self.deploy_to_node(node_id, node_config)

        tasks = [
            deploy_with_semaphore(node_id, node_config)
            for node_id, node_config in p2p_nodes.items()
        ]

        # Execute deployments
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Compile summary
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        duration = time.time() - start_time

        summary = ClusterDeploymentSummary(
            total_nodes=len(p2p_nodes),
            successful=successful,
            failed=failed,
            skipped=0,
            unreachable=0,
            results=results,
            duration_seconds=duration,
        )

        return summary


def print_summary(summary: ClusterDeploymentSummary, verbose: bool = False) -> None:
    """Print deployment summary.

    Args:
        summary: Deployment summary
        verbose: If True, print detailed results for each node
    """
    print("\n" + "=" * 70)
    print("P2P CLUSTER DEPLOYMENT SUMMARY")
    print("=" * 70)
    print(f"Total nodes:       {summary.total_nodes}")
    print(f"Successful:        {summary.successful} ({summary.success_rate:.1f}%)")
    print(f"Failed:            {summary.failed}")
    print(f"Duration:          {summary.duration_seconds:.1f}s")
    print("=" * 70)

    # Group results by status
    success_nodes = [r for r in summary.results if r.success]
    failed_nodes = [r for r in summary.results if not r.success]

    if success_nodes:
        print(f"\nSUCCESSFUL ({len(success_nodes)} nodes):")
        for result in success_nodes:
            status = "✓"
            if verbose:
                print(f"  {status} {result.node_id:25s} {result.duration_seconds:5.1f}s  {result.message}")
            else:
                print(f"  {status} {result.node_id}")

    if failed_nodes:
        print(f"\nFAILED ({len(failed_nodes)} nodes):")
        for result in failed_nodes:
            status = "✗"
            print(f"  {status} {result.node_id:25s} [{result.stage}] {result.error or result.message}")

    print("")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy P2P orchestrator to cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node IDs to deploy to (default: all P2P-enabled nodes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--skip-config-sync",
        action="store_true",
        help="Skip syncing distributed_hosts.yaml to nodes",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent deployments (default: {MAX_CONCURRENT})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse node filter
    node_filter = None
    if args.nodes:
        node_filter = [n.strip() for n in args.nodes.split(",")]
        logger.info(f"Filtering to nodes: {node_filter}")

    # Create deployer
    deployer = P2PClusterDeployer(
        dry_run=args.dry_run,
        skip_config_sync=args.skip_config_sync,
        max_concurrent=args.max_concurrent,
    )

    try:
        # Deploy to cluster
        summary = await deployer.deploy_to_cluster(node_filter=node_filter)

        # Print results
        print_summary(summary, verbose=args.verbose)

        # Return exit code
        if summary.failed > 0:
            return 1
        return 0

    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
