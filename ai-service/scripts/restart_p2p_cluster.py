#!/usr/bin/env python3
"""Coordinated P2P cluster restart with quorum preservation.

Jan 7, 2026: Created to fix cluster fragmentation issues. Restarts P2P nodes
in a safe order to preserve voter quorum:

1. Kill duplicate processes on all nodes
2. Restart non-voter nodes first (parallel)
3. Restart voter nodes one-by-one
4. Verify cluster convergence

Usage:
    # Dry run (preview actions)
    python scripts/restart_p2p_cluster.py --dry-run

    # Full restart
    python scripts/restart_p2p_cluster.py

    # Skip duplicate killing
    python scripts/restart_p2p_cluster.py --skip-kill-duplicates

    # Restart only voters
    python scripts/restart_p2p_cluster.py --voters-only
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.ssh import SSHClient, SSHConfig
from app.config.cluster_config import (
    get_cluster_nodes,
    get_p2p_voters,
    ClusterNode,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Node path mappings by provider
PATH_MAPPINGS = {
    'runpod': '/workspace/ringrift/ai-service',
    'vast': '~/ringrift/ai-service',
    'nebius': '~/ringrift/ai-service',
    'vultr': '/root/ringrift/ai-service',
    'hetzner': '/root/ringrift/ai-service',
    'lambda': '~/ringrift/ai-service',
}


def get_node_path(node_name: str, node: ClusterNode) -> str:
    """Get the ringrift path for a node."""
    if hasattr(node, 'ringrift_path') and node.ringrift_path:
        return node.ringrift_path

    for provider in PATH_MAPPINGS:
        if node_name.startswith(provider):
            return PATH_MAPPINGS[provider]

    return '~/ringrift/ai-service'


def get_ssh_config(node: ClusterNode) -> SSHConfig:
    """Build SSH config from cluster node."""
    # Prefer Tailscale IP for reliability
    host = node.tailscale_ip or node.ssh_host or node.name
    port = node.ssh_port or 22
    user = node.ssh_user or 'root'

    return SSHConfig(
        host=host,
        port=port,
        user=user,
        connect_timeout=30,
        command_timeout=60,
    )


async def kill_duplicate_p2p(node_name: str, node: ClusterNode, dry_run: bool = False) -> bool:
    """Kill duplicate P2P processes on a node.

    Returns True if successful (or dry run), False on error.
    """
    node_path = get_node_path(node_name, node)
    ssh_config = get_ssh_config(node)

    if dry_run:
        logger.info(f"[DRY RUN] Would kill duplicates on {node_name}")
        return True

    client = None
    try:
        client = SSHClient(ssh_config)
        # Kill all p2p_orchestrator processes and clean up screen sessions
        # Jan 2026: Added screen cleanup to prevent dead session accumulation
        result = await client.run_async(
            "pkill -f p2p_orchestrator 2>/dev/null || true; "
            "screen -X -S p2p quit 2>/dev/null || true; "
            "screen -wipe 2>/dev/null || true",
            timeout=15
        )
        logger.info(f"[{node_name}] Killed P2P processes and cleaned up screens")
        await asyncio.sleep(2)  # Allow processes to terminate
        return True
    except Exception as e:
        logger.error(f"[{node_name}] Failed to kill duplicates: {e}")
        return False
    finally:
        if client:
            client.close()


async def restart_p2p_node(
    node_name: str,
    node: ClusterNode,
    known_peers: list[str],
    dry_run: bool = False
) -> bool:
    """Restart P2P on a single node.

    Returns True if successful (or dry run), False on error.
    """
    node_path = get_node_path(node_name, node)
    ssh_config = get_ssh_config(node)

    if dry_run:
        logger.info(f"[DRY RUN] Would restart P2P on {node_name}")
        return True

    client = None
    try:
        client = SSHClient(ssh_config)

        # Determine venv activation
        venv_activate = "source .venv/bin/activate 2>/dev/null || true"

        # Build P2P arguments
        p2p_args = [
            f"--node-id {node_name}",
            "--port 8770",
            f"--ringrift-path {node_path}",
            "--kill-duplicates",
        ]

        # Add advertise-host for Tailscale
        if node.tailscale_ip:
            p2p_args.append(f"--advertise-host {node.tailscale_ip}")

        # Add bootstrap peers
        if known_peers:
            p2p_args.append(f"--peers {','.join(known_peers)}")

        p2p_args_str = ' '.join(p2p_args)

        start_cmd = (
            f"cd {node_path} && {venv_activate} && "
            f"mkdir -p logs && "
            f"nohup python scripts/p2p_orchestrator.py {p2p_args_str} "
            f"> logs/p2p.log 2>&1 &"
        )

        result = await client.run_async(start_cmd, timeout=15)

        if result.returncode != 0:
            logger.warning(f"[{node_name}] P2P restart may have failed: {result.stderr}")
            return False

        # Verify it started
        await asyncio.sleep(3)
        verify = await client.run_async("pgrep -f p2p_orchestrator", timeout=10)
        if verify.returncode == 0 and verify.stdout.strip():
            logger.info(f"[{node_name}] P2P restarted (PID: {verify.stdout.strip().split()[0]})")
            return True
        else:
            logger.warning(f"[{node_name}] P2P process not found after restart")
            return False

    except Exception as e:
        logger.error(f"[{node_name}] Failed to restart P2P: {e}")
        return False
    finally:
        if client:
            client.close()


async def check_cluster_convergence(
    coordinator_host: str = "localhost",
    timeout: float = 120.0,
    min_alive_peers: int = 5
) -> dict[str, Any]:
    """Check if cluster has converged to a consistent state.

    Returns dict with:
        - converged: bool
        - leader_id: str or None
        - alive_peers: int
        - voter_quorum_ok: bool
    """
    import aiohttp

    start = time.time()

    while (time.time() - start) < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{coordinator_host}:8770/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        alive = status.get('alive_peers', 0)
                        leader = status.get('leader_id')
                        quorum_ok = status.get('voter_quorum_ok', False)

                        logger.info(
                            f"Cluster status: leader={leader}, "
                            f"alive={alive}, quorum_ok={quorum_ok}"
                        )

                        if leader and alive >= min_alive_peers and quorum_ok:
                            return {
                                'converged': True,
                                'leader_id': leader,
                                'alive_peers': alive,
                                'voter_quorum_ok': quorum_ok,
                            }
        except Exception as e:
            logger.debug(f"Convergence check failed: {e}")

        await asyncio.sleep(10)

    return {
        'converged': False,
        'leader_id': None,
        'alive_peers': 0,
        'voter_quorum_ok': False,
    }


async def restart_p2p_cluster(
    kill_duplicates: bool = True,
    restart_voters_last: bool = True,
    verify_convergence: bool = True,
    dry_run: bool = False,
    voters_only: bool = False,
) -> bool:
    """Coordinated P2P cluster restart.

    Args:
        kill_duplicates: Kill duplicate processes before restart
        restart_voters_last: Restart voters after non-voters for safety
        verify_convergence: Wait for cluster convergence after restart
        dry_run: Log actions without executing
        voters_only: Only restart voter nodes

    Returns:
        True if cluster converged successfully, False otherwise
    """
    logger.info("=" * 60)
    logger.info("P2P Cluster Restart")
    logger.info("=" * 60)

    # Load cluster config
    nodes = get_cluster_nodes()
    voters = set(get_p2p_voters())

    logger.info(f"Total nodes: {len(nodes)}")
    logger.info(f"Voters: {voters}")

    # Build bootstrap peer list from voters
    bootstrap_peers = []
    for voter_name in voters:
        node = nodes.get(voter_name)
        if node and node.tailscale_ip:
            bootstrap_peers.append(f"{node.tailscale_ip}:8770")
    logger.info(f"Bootstrap peers: {bootstrap_peers[:5]}")

    # Separate voters and non-voters
    voter_nodes = {n: nodes[n] for n in voters if n in nodes}
    non_voter_nodes = {n: v for n, v in nodes.items() if n not in voters}

    # Filter to active nodes only
    def is_active(node: ClusterNode) -> bool:
        status = getattr(node, 'status', 'active')
        return status in ('active', 'ready', None, '')

    voter_nodes = {n: v for n, v in voter_nodes.items() if is_active(v)}
    non_voter_nodes = {n: v for n, v in non_voter_nodes.items() if is_active(v)}

    logger.info(f"Active voters: {len(voter_nodes)}")
    logger.info(f"Active non-voters: {len(non_voter_nodes)}")

    # Phase 1: Kill duplicates on all nodes
    if kill_duplicates:
        logger.info("\n--- Phase 1: Killing duplicate processes ---")
        all_nodes = {**voter_nodes, **non_voter_nodes}
        kill_tasks = [
            kill_duplicate_p2p(name, node, dry_run)
            for name, node in all_nodes.items()
        ]
        await asyncio.gather(*kill_tasks, return_exceptions=True)
        logger.info("Duplicate killing complete")
        await asyncio.sleep(5)  # Let processes fully terminate

    # Phase 2: Restart non-voters in parallel
    if not voters_only and non_voter_nodes:
        logger.info("\n--- Phase 2: Restarting non-voter nodes (parallel) ---")
        restart_tasks = [
            restart_p2p_node(name, node, bootstrap_peers, dry_run)
            for name, node in non_voter_nodes.items()
        ]
        results = await asyncio.gather(*restart_tasks, return_exceptions=True)
        success = sum(1 for r in results if r is True)
        logger.info(f"Non-voters: {success}/{len(non_voter_nodes)} restarted successfully")
        await asyncio.sleep(10)  # Let non-voters stabilize

    # Phase 3: Restart voters one-by-one
    logger.info("\n--- Phase 3: Restarting voter nodes (sequential) ---")
    voter_success = 0
    for name, node in voter_nodes.items():
        logger.info(f"Restarting voter: {name}")
        if await restart_p2p_node(name, node, bootstrap_peers, dry_run):
            voter_success += 1
        # Wait between voter restarts to maintain quorum
        await asyncio.sleep(15)

    logger.info(f"Voters: {voter_success}/{len(voter_nodes)} restarted successfully")

    # Phase 4: Verify convergence
    if verify_convergence and not dry_run:
        logger.info("\n--- Phase 4: Verifying cluster convergence ---")
        result = await check_cluster_convergence(timeout=120)

        if result['converged']:
            logger.info("=" * 60)
            logger.info("CLUSTER CONVERGED SUCCESSFULLY")
            logger.info(f"  Leader: {result['leader_id']}")
            logger.info(f"  Alive peers: {result['alive_peers']}")
            logger.info(f"  Quorum OK: {result['voter_quorum_ok']}")
            logger.info("=" * 60)
            return True
        else:
            logger.error("=" * 60)
            logger.error("CLUSTER FAILED TO CONVERGE")
            logger.error("Manual intervention may be required")
            logger.error("=" * 60)
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Coordinated P2P cluster restart with quorum preservation"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview actions without executing'
    )
    parser.add_argument(
        '--skip-kill-duplicates',
        action='store_true',
        help='Skip killing duplicate processes'
    )
    parser.add_argument(
        '--voters-only',
        action='store_true',
        help='Only restart voter nodes'
    )
    parser.add_argument(
        '--skip-convergence-check',
        action='store_true',
        help='Skip waiting for cluster convergence'
    )
    args = parser.parse_args()

    success = asyncio.run(restart_p2p_cluster(
        kill_duplicates=not args.skip_kill_duplicates,
        verify_convergence=not args.skip_convergence_check,
        dry_run=args.dry_run,
        voters_only=args.voters_only,
    ))

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
