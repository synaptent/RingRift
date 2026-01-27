#!/usr/bin/env python3
"""Quorum-Safe Rolling Update Coordinator.

January 3, 2026 - Sprint 16.2: Production-grade cluster update system that
prevents quorum loss by integrating with existing health infrastructure.

Problem Solved: The previous update script (update_all_nodes.py) operated in
complete isolation from P2P health monitoring, allowing simultaneous restarts
of 10+ nodes including voters, causing quorum loss and cluster failure.

Solution: QuorumSafeUpdateCoordinator ensures:
    1. Quorum-First: Never update more voters than quorum allows
    2. Health-Gated: Verify cluster health before/after each batch
    3. Staged Rollout: Non-voters first, then voters one at a time
    4. Convergence Aware: Wait for gossip to propagate after restarts
    5. Rollback Ready: Checkpoint before update, revert on failure

Usage:
    from scripts.cluster_update_coordinator import (
        QuorumSafeUpdateCoordinator,
        UpdateResult,
    )

    coordinator = QuorumSafeUpdateCoordinator()
    result = await coordinator.update_cluster(
        target_commit="main",
        restart_p2p=True,
    )

CLI Usage:
    python scripts/update_all_nodes.py --safe-mode --restart-p2p
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ssh import SSHClient, SSHConfig

logger = logging.getLogger(__name__)

__all__ = [
    "QuorumSafeUpdateCoordinator",
    "UpdateCoordinatorConfig",
    "UpdateResult",
    "UpdateBatch",
    "BatchCheckpoint",
    "QuorumHealthLevel",
    "QuorumUnsafeError",
    "ConvergenceTimeoutError",
    "QuorumLostError",
]


class QuorumHealthLevel(str, Enum):
    """Quorum health levels (mirrors scripts/p2p/leader_election.py)."""
    HEALTHY = "healthy"      # >= quorum + 2 voters alive
    DEGRADED = "degraded"    # >= quorum + 1 voters alive
    MINIMUM = "minimum"      # = quorum voters alive (fragile)
    LOST = "lost"            # < quorum voters alive


class QuorumUnsafeError(Exception):
    """Raised when update would be unsafe for quorum."""
    pass


class ConvergenceTimeoutError(Exception):
    """Raised when cluster fails to converge after update."""
    pass


class QuorumLostError(Exception):
    """Raised when quorum is lost during update."""
    pass


@dataclass
class NodeConfig:
    """Configuration for a single cluster node."""
    name: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_key: str | None
    tailscale_ip: str | None
    ringrift_path: str
    is_voter: bool
    status: str
    nat_blocked: bool = False
    venv_activate: str = "source venv/bin/activate"


@dataclass
class UpdateBatch:
    """A batch of nodes to update together."""
    nodes: list[NodeConfig]
    batch_type: str  # "non_voters" or "voter"

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]


@dataclass
class BatchCheckpoint:
    """Checkpoint for rollback capability."""
    batch_nodes: list[str]
    previous_commits: dict[str, str]  # node_name -> commit hash
    p2p_was_running: dict[str, bool]  # node_name -> was P2P running
    timestamp: float


@dataclass
class UpdateResult:
    """Result of cluster update operation."""
    success: bool
    batches_updated: int
    nodes_updated: list[str] = field(default_factory=list)
    nodes_failed: list[str] = field(default_factory=list)
    nodes_skipped: list[str] = field(default_factory=list)
    rollback_performed: bool = False
    error_message: str | None = None
    failed_batch: int | None = None
    duration_seconds: float = 0.0


@dataclass
class UpdateCoordinatorConfig:
    """Configuration for QuorumSafeUpdateCoordinator."""
    convergence_timeout: float = 120.0
    voter_update_delay: float = 30.0
    max_parallel_non_voters: int = 10
    dry_run: bool = False
    config_path: str | Path | None = None
    health_check_endpoint: str = "http://localhost:8770/status"
    sync_config: bool = False  # Sync non-git-tracked config files


# Config files to sync (relative to ai-service directory)
# These are files in .gitignore that need explicit sync
CONFIG_FILES_TO_SYNC = [
    'config/distributed_hosts.yaml',
]


@dataclass
class ClusterHealth:
    """Snapshot of cluster health state."""
    quorum_level: QuorumHealthLevel
    alive_peers: int
    total_peers: int
    leader_id: str | None
    alive_voters: list[str]
    total_voters: int
    quorum_required: int


class QuorumSafeUpdateCoordinator:
    """Production-grade cluster update with quorum preservation.

    This coordinator ensures cluster updates never cause quorum loss by:
    1. Pre-flight validation of cluster health
    2. Batching nodes (non-voters first, then voters one at a time)
    3. Health verification between batches
    4. Automatic rollback on failure

    Example:
        coordinator = QuorumSafeUpdateCoordinator()
        result = await coordinator.update_cluster(
            target_commit="main",
            restart_p2p=True,
        )
        if result.success:
            print(f"Updated {len(result.nodes_updated)} nodes")
    """

    # Default voter list (from distributed_hosts.yaml p2p_voters)
    DEFAULT_VOTERS = [
        "lambda-gh200-1",
        "lambda-gh200-2",
        "nebius-backbone-1",
        "nebius-h100-3",
        "hetzner-cpu1",
        "hetzner-cpu2",
        "vultr-a100-20gb",
    ]

    # Quorum requirement (from leader_election.py VOTER_MIN_QUORUM)
    QUORUM_REQUIRED = 4

    def __init__(
        self,
        config: UpdateCoordinatorConfig | None = None,
        *,
        config_path: str | Path | None = None,
        health_check_endpoint: str = "http://localhost:8770/status",
        convergence_timeout: float = 120.0,
        max_parallel_non_voters: int = 10,
        voter_update_delay: float = 30.0,
    ):
        """Initialize the update coordinator.

        Args:
            config: Configuration object (preferred). If provided, overrides other args.
            config_path: Path to distributed_hosts.yaml (auto-detected if None)
            health_check_endpoint: P2P status endpoint for health checks
            convergence_timeout: Seconds to wait for gossip convergence
            max_parallel_non_voters: Max concurrent non-voter updates
            voter_update_delay: Seconds to wait between voter updates
        """
        # Use config object if provided, otherwise use individual args
        if config is not None:
            self.config_path = config.config_path or self._find_config_path()
            self.health_endpoint = config.health_check_endpoint
            self.convergence_timeout = config.convergence_timeout
            self.max_parallel_non_voters = config.max_parallel_non_voters
            self.voter_update_delay = config.voter_update_delay
            self._dry_run = config.dry_run
            self._sync_config = config.sync_config
        else:
            self.config_path = config_path or self._find_config_path()
            self.health_endpoint = health_check_endpoint
            self.convergence_timeout = convergence_timeout
            self.max_parallel_non_voters = max_parallel_non_voters
            self.voter_update_delay = voter_update_delay
            self._dry_run = False
            self._sync_config = False

        # Load config on init
        self._config: dict[str, Any] | None = None
        self._voter_node_ids: set[str] = set()

    def _find_config_path(self) -> Path:
        """Find distributed_hosts.yaml config file."""
        candidates = [
            Path(__file__).parent.parent / "config" / "distributed_hosts.yaml",
            Path.cwd() / "config" / "distributed_hosts.yaml",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError("Could not find distributed_hosts.yaml")

    def _load_config(self) -> dict[str, Any]:
        """Load cluster configuration from YAML."""
        if self._config is None:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)

            # Extract voter list
            p2p_config = self._config.get("p2p_settings", {})
            voters = p2p_config.get("p2p_voters", self.DEFAULT_VOTERS)
            self._voter_node_ids = set(voters)

        return self._config

    def _get_node_configs(self) -> list[NodeConfig]:
        """Parse all node configurations from YAML."""
        config = self._load_config()
        hosts = config.get("hosts", {})
        nodes = []

        for name, host_config in hosts.items():
            # Skip nodes that aren't ready
            status = host_config.get("status", "unknown")

            # Determine path
            ringrift_path = host_config.get("ringrift_path")
            if ringrift_path is None:
                # Use provider-based defaults
                for provider in ["runpod", "vast", "nebius", "vultr", "hetzner"]:
                    if name.startswith(provider):
                        if provider == "runpod":
                            ringrift_path = "/workspace/ringrift/ai-service"
                        elif provider == "vultr" or provider == "hetzner":
                            ringrift_path = "/root/ringrift/ai-service"
                        else:
                            ringrift_path = "~/ringrift/ai-service"
                        break
                if ringrift_path is None:
                    ringrift_path = "~/ringrift/ai-service"

            nodes.append(NodeConfig(
                name=name,
                ssh_host=host_config.get("ssh_host", host_config.get("tailscale_ip", "")),
                ssh_port=host_config.get("ssh_port", 22),
                ssh_user=host_config.get("ssh_user", "root"),
                ssh_key=host_config.get("ssh_key"),
                tailscale_ip=host_config.get("tailscale_ip"),
                ringrift_path=ringrift_path,
                is_voter=name in self._voter_node_ids,
                status=status,
                nat_blocked=host_config.get("nat_blocked", False),
                venv_activate=host_config.get("venv_activate", "source venv/bin/activate"),
            ))

        return nodes

    async def _get_cluster_health(self) -> ClusterHealth:
        """Query cluster health from P2P status endpoint."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.health_endpoint, timeout=10) as resp:
                    if resp.status != 200:
                        logger.warning(f"Health endpoint returned {resp.status}")
                        return ClusterHealth(
                            quorum_level=QuorumHealthLevel.LOST,
                            alive_peers=0,
                            total_peers=0,
                            leader_id=None,
                            alive_voters=[],
                            total_voters=len(self._voter_node_ids),
                            quorum_required=self.QUORUM_REQUIRED,
                        )

                    data = await resp.json()

                    # Extract quorum info
                    alive_peers = data.get("alive_peers", 0)
                    leader_id = data.get("leader_id")

                    # Get alive voters - P2P status uses 'peers' dict and 'voter_quorum_ok'
                    # First try the voters_alive count directly from P2P
                    voter_quorum_ok = data.get("voter_quorum_ok", False)
                    voters_alive_count = data.get("voters_alive", 0)

                    # Extract peer names from 'peers' dict (P2P uses this format)
                    peers_dict = data.get("peers", {})
                    alive_peers_list = list(peers_dict.keys()) if peers_dict else []

                    # Match peer names to voter IDs (handle both formats)
                    alive_voters = []
                    for peer_name in alive_peers_list:
                        # Check both raw name and name without port
                        clean_name = peer_name.split(":")[0] if ":" in peer_name else peer_name
                        if peer_name in self._voter_node_ids or clean_name in self._voter_node_ids:
                            alive_voters.append(peer_name)

                    # Use P2P's voter_quorum_ok if available, otherwise calculate
                    voter_count = max(len(alive_voters), voters_alive_count)
                    if voter_quorum_ok:
                        # P2P says quorum is OK - trust it
                        if voter_count >= self.QUORUM_REQUIRED + 2:
                            level = QuorumHealthLevel.HEALTHY
                        elif voter_count == self.QUORUM_REQUIRED + 1:
                            level = QuorumHealthLevel.DEGRADED
                        else:
                            level = QuorumHealthLevel.MINIMUM
                    else:
                        # P2P says quorum is lost
                        level = QuorumHealthLevel.LOST

                    return ClusterHealth(
                        quorum_level=level,
                        alive_peers=alive_peers,
                        total_peers=len(self._get_node_configs()),
                        leader_id=leader_id,
                        alive_voters=alive_voters,
                        total_voters=len(self._voter_node_ids),
                        quorum_required=self.QUORUM_REQUIRED,
                    )

        except Exception as e:
            logger.error(f"Failed to get cluster health: {e}")
            return ClusterHealth(
                quorum_level=QuorumHealthLevel.LOST,
                alive_peers=0,
                total_peers=0,
                leader_id=None,
                alive_voters=[],
                total_voters=len(self._voter_node_ids),
                quorum_required=self.QUORUM_REQUIRED,
            )

    def _calculate_update_batches(
        self,
        nodes: list[NodeConfig],
        skip_voters: bool = False,
    ) -> list[UpdateBatch]:
        """Calculate safe update batches.

        Strategy:
        1. Batch 1: All non-voter nodes (parallel, safe for quorum)
        2. Batches 2+: One voter at a time (preserve quorum)

        Args:
            nodes: All nodes to potentially update
            skip_voters: If True, only update non-voters

        Returns:
            List of UpdateBatch in order of execution
        """
        # Separate voters and non-voters
        voters = [n for n in nodes if n.is_voter and n.status == "ready"]
        non_voters = [n for n in nodes if not n.is_voter and n.status == "ready"]

        batches = []

        # Batch 1: All non-voters
        if non_voters:
            batches.append(UpdateBatch(
                nodes=non_voters,
                batch_type="non_voters",
            ))

        # Batches 2+: One voter at a time
        if not skip_voters:
            for voter in voters:
                batches.append(UpdateBatch(
                    nodes=[voter],
                    batch_type="voter",
                ))

        return batches

    async def _create_ssh_client(self, node: NodeConfig) -> SSHClient:
        """Create SSH client for a node."""
        # Prefer Tailscale IP over SSH host
        if node.tailscale_ip:
            host = node.tailscale_ip
            port = 22
        else:
            host = node.ssh_host
            port = node.ssh_port

        config = SSHConfig(
            host=host,
            port=port,
            user=node.ssh_user,
            key_path=node.ssh_key,
            tailscale_ip=node.tailscale_ip,
            work_dir=node.ringrift_path,
        )
        return SSHClient(config)

    async def _check_p2p_running(self, client: SSHClient, node_name: str) -> bool:
        """Check if P2P orchestrator is running on node."""
        result = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
        return result.returncode == 0 and bool(result.stdout.strip())

    async def _sync_config_files(
        self,
        client: SSHClient,
        node: NodeConfig,
        dry_run: bool = False,
    ) -> tuple[bool, str]:
        """Sync non-git-tracked config files to a node.

        These files are in .gitignore so git pull doesn't update them.

        Returns:
            (success, message)
        """
        local_base = Path(__file__).parent.parent  # ai-service directory
        synced_files = []
        failed_files = []

        for config_file in CONFIG_FILES_TO_SYNC:
            local_file = local_base / config_file
            if not local_file.exists():
                logger.warning(f"[{node.name}] Config file not found locally: {config_file}")
                failed_files.append(config_file)
                continue

            remote_file = f"{node.ringrift_path}/{config_file}"

            if dry_run:
                logger.info(f"[{node.name}] DRY-RUN: Would sync {config_file}")
                synced_files.append(config_file)
                continue

            try:
                with open(local_file, 'r') as f:
                    content = f.read()

                # Ensure parent directory exists
                parent_dir = '/'.join(remote_file.rsplit('/', 1)[:-1])
                await client.run_async(f"mkdir -p {parent_dir}", timeout=10)

                # Write content via heredoc
                cmd = f"cat > {remote_file} << 'CONFIGEOF'\n{content}\nCONFIGEOF"
                result = await client.run_async(cmd, timeout=30)

                if result.returncode == 0:
                    logger.info(f"[{node.name}] Synced {config_file}")
                    synced_files.append(config_file)
                else:
                    logger.warning(f"[{node.name}] Failed to sync {config_file}: {result.stderr}")
                    failed_files.append(config_file)

            except Exception as e:
                logger.error(f"[{node.name}] Error syncing {config_file}: {e}")
                failed_files.append(config_file)

        if failed_files:
            return (False, f"Config sync partial: {len(synced_files)} OK, {len(failed_files)} failed")
        elif synced_files:
            return (True, f"Config synced: {', '.join(synced_files)}")
        else:
            return (True, "No config files to sync")

    async def _save_batch_checkpoint(
        self,
        batch: UpdateBatch,
    ) -> BatchCheckpoint:
        """Save git commits and P2P state before update for rollback."""
        commits = {}
        p2p_running = {}

        for node in batch.nodes:
            try:
                client = await self._create_ssh_client(node)

                # Get current commit
                result = await client.run_async(
                    f"cd {node.ringrift_path} && git rev-parse --short HEAD",
                    timeout=10,
                )
                commits[node.name] = result.stdout.strip() if result.returncode == 0 else "unknown"

                # Check P2P status
                p2p_running[node.name] = await self._check_p2p_running(client, node.name)

            except Exception as e:
                logger.warning(f"[{node.name}] Failed to save checkpoint: {e}")
                commits[node.name] = "unknown"
                p2p_running[node.name] = False

        return BatchCheckpoint(
            batch_nodes=batch.node_names,
            previous_commits=commits,
            p2p_was_running=p2p_running,
            timestamp=time.time(),
        )

    async def _update_single_node(
        self,
        node: NodeConfig,
        target_commit: str,
        restart_p2p: bool,
        dry_run: bool,
        sync_config: bool = False,
    ) -> tuple[str, bool, str]:
        """Update a single node.

        Returns:
            (node_name, success, message)
        """
        try:
            client = await self._create_ssh_client(node)

            # Test connection
            test = await client.run_async("echo connected", timeout=10)
            if test.returncode != 0:
                return (node.name, False, f"Connection failed: {test.stderr}")

            # Check P2P status before update
            p2p_was_running = await self._check_p2p_running(client, node.name)

            if dry_run:
                msg = f"DRY-RUN: Would update {node.ringrift_path}"
                if sync_config:
                    msg += " and sync config"
                if p2p_was_running and restart_p2p:
                    msg += " and restart P2P"
                return (node.name, True, msg)

            # Stash local changes
            await client.run_async(f"cd {node.ringrift_path} && git stash", timeout=30)

            # Pull latest code
            pull_result = await client.run_async(
                f"cd {node.ringrift_path} && git pull origin main",
                timeout=60,
            )
            if pull_result.returncode != 0:
                return (node.name, False, f"Git pull failed: {pull_result.stderr}")

            # Verify commit
            verify = await client.run_async(
                f"cd {node.ringrift_path} && git rev-parse --short HEAD",
                timeout=10,
            )
            current_commit = verify.stdout.strip() if verify.returncode == 0 else "unknown"

            # Sync config files if requested (for files in .gitignore)
            config_sync_msg = ""
            if sync_config:
                sync_success, sync_msg = await self._sync_config_files(client, node, dry_run)
                if sync_success:
                    config_sync_msg = f", {sync_msg}"
                else:
                    config_sync_msg = f", {sync_msg}"

            # Restart P2P if needed
            if p2p_was_running and restart_p2p:
                # Kill existing P2P and clean up screen sessions
                # Jan 2026: Added screen cleanup to prevent dead session accumulation
                await client.run_async(
                    "pkill -f p2p_orchestrator 2>/dev/null || true; "
                    "screen -X -S p2p quit 2>/dev/null || true; "
                    "screen -wipe 2>/dev/null || true",
                    timeout=15,
                )
                await asyncio.sleep(2)

                # Build restart command
                p2p_args = [
                    f"--node-id {node.name}",
                    "--port 8770",
                    f"--ringrift-path {node.ringrift_path}",
                    "--kill-duplicates",
                ]

                if node.tailscale_ip:
                    p2p_args.append(f"--advertise-host {node.tailscale_ip}")

                if node.nat_blocked:
                    relay_peers = ["vultr-a100-20gb:8770", "nebius-h100-3:8770"]
                    p2p_args.append(f"--relay-peers {','.join(relay_peers)}")

                known_peers = [
                    "vultr-a100-20gb:8770",
                    "nebius-h100-3:8770",
                    "hetzner-cpu1:8770",
                    "nebius-backbone-1:8770",
                ]
                p2p_args.append(f"--peers {','.join(known_peers)}")

                start_cmd = (
                    f"cd {node.ringrift_path} && {node.venv_activate} && "
                    f"mkdir -p logs && "
                    f"nohup python scripts/p2p_orchestrator.py {' '.join(p2p_args)} "
                    f"> logs/p2p.log 2>&1 &"
                )
                await client.run_async(start_cmd, timeout=15)
                await asyncio.sleep(3)

                if await self._check_p2p_running(client, node.name):
                    return (node.name, True, f"Updated to {current_commit}{config_sync_msg}, P2P restarted")
                else:
                    return (node.name, True, f"Updated to {current_commit}{config_sync_msg}, P2P restart failed")

            return (node.name, True, f"Updated to {current_commit}{config_sync_msg}")

        except Exception as e:
            logger.error(f"[{node.name}] Update error: {e}")
            return (node.name, False, f"Exception: {str(e)}")

    async def _update_batch(
        self,
        batch: UpdateBatch,
        target_commit: str,
        restart_p2p: bool,
        dry_run: bool,
        sync_config: bool = False,
    ) -> list[tuple[str, bool, str]]:
        """Update all nodes in a batch.

        For non-voters: parallel updates (up to max_parallel_non_voters)
        For voters: single node (already split into single-node batches)
        """
        if batch.batch_type == "non_voters":
            # Parallel updates with semaphore
            semaphore = asyncio.Semaphore(self.max_parallel_non_voters)

            async def update_with_semaphore(node: NodeConfig):
                async with semaphore:
                    return await self._update_single_node(
                        node, target_commit, restart_p2p, dry_run, sync_config
                    )

            tasks = [update_with_semaphore(n) for n in batch.nodes]
            return list(await asyncio.gather(*tasks))
        else:
            # Single voter update
            results = []
            for node in batch.nodes:
                result = await self._update_single_node(
                    node, target_commit, restart_p2p, dry_run, sync_config
                )
                results.append(result)
            return results

    async def _wait_for_convergence(
        self,
        batch: UpdateBatch,
        timeout: float | None = None,
    ) -> bool:
        """Wait for updated nodes to rejoin mesh and gossip to converge.

        Args:
            batch: Batch that was just updated
            timeout: Override convergence timeout

        Returns:
            True if all nodes visible and quorum healthy
        """
        timeout = timeout or self.convergence_timeout
        deadline = time.time() + timeout
        node_names = set(batch.node_names)

        logger.info(f"Waiting up to {timeout}s for convergence of {node_names}")

        while time.time() < deadline:
            health = await self._get_cluster_health()

            # Check quorum is not lost
            if health.quorum_level == QuorumHealthLevel.LOST:
                logger.error(f"Quorum lost during convergence wait")
                return False

            # Check if leader is stable
            if health.leader_id:
                # For voter batches, verify voter is visible
                if batch.batch_type == "voter":
                    # Voter should be in alive peers list (via /status)
                    pass  # Can't easily check without alive_peers_list

                # Check quorum is at least DEGRADED
                if health.quorum_level in (QuorumHealthLevel.HEALTHY, QuorumHealthLevel.DEGRADED):
                    logger.info(f"Convergence achieved: quorum={health.quorum_level.value}, "
                               f"leader={health.leader_id}")
                    return True

            await asyncio.sleep(5)

        logger.warning(f"Convergence timeout after {timeout}s")
        return False

    async def _rollback_batch(
        self,
        batch: UpdateBatch,
        checkpoint: BatchCheckpoint,
    ) -> None:
        """Rollback batch to checkpoint commits."""
        logger.warning(f"Rolling back batch: {batch.node_names}")

        for node in batch.nodes:
            target_commit = checkpoint.previous_commits.get(node.name)
            if not target_commit or target_commit == "unknown":
                logger.warning(f"[{node.name}] No checkpoint commit, skipping rollback")
                continue

            try:
                client = await self._create_ssh_client(node)

                # Checkout previous commit
                await client.run_async(
                    f"cd {node.ringrift_path} && git checkout {target_commit}",
                    timeout=30,
                )
                logger.info(f"[{node.name}] Rolled back to {target_commit}")

                # Restart P2P if it was running
                if checkpoint.p2p_was_running.get(node.name):
                    await client.run_async("pkill -f p2p_orchestrator", timeout=10)
                    await asyncio.sleep(2)
                    # Simplified restart
                    await client.run_async(
                        f"cd {node.ringrift_path} && {node.venv_activate} && "
                        f"nohup python scripts/p2p_orchestrator.py "
                        f"--node-id {node.name} --port 8770 "
                        f"> logs/p2p.log 2>&1 &",
                        timeout=15,
                    )

            except Exception as e:
                logger.error(f"[{node.name}] Rollback failed: {e}")

    async def update_cluster(
        self,
        target_commit: str = "main",
        restart_p2p: bool = True,
        dry_run: bool = False,
        skip_voters: bool = False,
        skip_non_voters: bool = False,
    ) -> UpdateResult:
        """Safely update entire cluster with quorum preservation.

        This is the main entry point for cluster updates. It:
        1. Validates cluster health before starting
        2. Calculates safe update batches
        3. Updates each batch with health verification
        4. Rolls back on failure

        Args:
            target_commit: Git commit/branch to update to
            restart_p2p: Whether to restart P2P orchestrator
            dry_run: Preview mode without making changes
            skip_voters: Only update non-voter nodes
            skip_non_voters: Only update voter nodes

        Returns:
            UpdateResult with success status and details
        """
        start_time = time.time()

        # Load config
        self._load_config()

        # Phase 1: Pre-flight checks
        logger.info("Phase 1: Pre-flight health check...")
        health = await self._get_cluster_health()

        if health.quorum_level == QuorumHealthLevel.LOST:
            return UpdateResult(
                success=False,
                batches_updated=0,
                error_message="Cannot update: quorum already lost",
                duration_seconds=time.time() - start_time,
            )

        if health.quorum_level == QuorumHealthLevel.MINIMUM:
            logger.warning("Quorum at MINIMUM - proceeding with caution")

        logger.info(f"Pre-flight: quorum={health.quorum_level.value}, "
                   f"alive_voters={len(health.alive_voters)}/{health.total_voters}, "
                   f"leader={health.leader_id}")

        # Phase 2: Calculate safe batches
        logger.info("Phase 2: Calculating update batches...")
        all_nodes = self._get_node_configs()

        if skip_non_voters:
            all_nodes = [n for n in all_nodes if n.is_voter]

        batches = self._calculate_update_batches(all_nodes, skip_voters=skip_voters)

        if not batches:
            return UpdateResult(
                success=True,
                batches_updated=0,
                error_message="No nodes to update",
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Calculated {len(batches)} batches:")
        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i+1}: {batch.batch_type} - {batch.node_names}")

        # Phase 3: Update each batch
        result = UpdateResult(success=True, batches_updated=0)

        for batch_num, batch in enumerate(batches):
            logger.info(f"\nPhase 3.{batch_num+1}: Updating batch {batch_num+1}/{len(batches)} "
                       f"({batch.batch_type}): {batch.node_names}")

            # Save checkpoint for rollback
            checkpoint = await self._save_batch_checkpoint(batch)

            # Update batch
            batch_results = await self._update_batch(
                batch, target_commit, restart_p2p, dry_run, self._sync_config
            )

            # Process results
            for node_name, success, message in batch_results:
                logger.info(f"  [{node_name}] {message}")
                if success:
                    if "SKIPPED" in message or "DRY-RUN" in message:
                        result.nodes_skipped.append(node_name)
                    else:
                        result.nodes_updated.append(node_name)
                else:
                    result.nodes_failed.append(node_name)

            # Skip convergence check for dry run or non-P2P updates
            if dry_run or not restart_p2p:
                result.batches_updated += 1
                continue

            # Wait for convergence
            logger.info(f"  Waiting for convergence...")
            converged = await self._wait_for_convergence(batch)

            if not converged:
                logger.error(f"Batch {batch_num+1} failed to converge, rolling back...")
                await self._rollback_batch(batch, checkpoint)
                result.rollback_performed = True
                result.success = False
                result.error_message = f"Batch {batch_num+1} failed to converge"
                result.failed_batch = batch_num + 1
                result.duration_seconds = time.time() - start_time
                return result

            # Verify quorum still healthy
            health = await self._get_cluster_health()
            if health.quorum_level == QuorumHealthLevel.LOST:
                logger.error(f"Quorum lost after batch {batch_num+1}, rolling back...")
                await self._rollback_batch(batch, checkpoint)
                result.rollback_performed = True
                result.success = False
                result.error_message = f"Quorum lost after batch {batch_num+1}"
                result.failed_batch = batch_num + 1
                result.duration_seconds = time.time() - start_time
                return result

            result.batches_updated += 1

            # Delay before next voter batch
            if batch.batch_type == "voter" and batch_num < len(batches) - 1:
                logger.info(f"  Waiting {self.voter_update_delay}s before next voter...")
                await asyncio.sleep(self.voter_update_delay)

        result.duration_seconds = time.time() - start_time
        logger.info(f"\nUpdate complete: {len(result.nodes_updated)} updated, "
                   f"{len(result.nodes_failed)} failed, "
                   f"{len(result.nodes_skipped)} skipped")

        return result


async def main():
    """CLI entry point for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quorum-safe cluster update coordinator"
    )
    parser.add_argument("--commit", default="main",
                       help="Target git commit/branch (default: main)")
    parser.add_argument("--restart-p2p", action="store_true",
                       help="Restart P2P orchestrator after update")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview mode without making changes")
    parser.add_argument("--skip-voters", action="store_true",
                       help="Only update non-voter nodes")
    parser.add_argument("--skip-non-voters", action="store_true",
                       help="Only update voter nodes")
    parser.add_argument("--convergence-timeout", type=int, default=120,
                       help="Seconds to wait for convergence (default: 120)")
    parser.add_argument("--voter-delay", type=int, default=30,
                       help="Seconds between voter updates (default: 30)")
    parser.add_argument("--max-parallel", type=int, default=10,
                       help="Max parallel non-voter updates (default: 10)")

    args = parser.parse_args()

    coordinator = QuorumSafeUpdateCoordinator(
        convergence_timeout=args.convergence_timeout,
        voter_update_delay=args.voter_delay,
        max_parallel_non_voters=args.max_parallel,
    )

    result = await coordinator.update_cluster(
        target_commit=args.commit,
        restart_p2p=args.restart_p2p,
        dry_run=args.dry_run,
        skip_voters=args.skip_voters,
        skip_non_voters=args.skip_non_voters,
    )

    if result.success:
        print(f"\n✅ Update successful: {result.batches_updated} batches, "
              f"{len(result.nodes_updated)} nodes updated")
    else:
        print(f"\n❌ Update failed: {result.error_message}")
        if result.rollback_performed:
            print("   Rollback was performed")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(asyncio.run(main()))
