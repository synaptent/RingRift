#!/usr/bin/env python3
"""Enable SWIM/Raft protocols on cluster nodes.

This script enables SWIM and Raft by setting environment variables and
optionally restarting P2P orchestrators.

Usage:
    # Enable SWIM on all nodes (hybrid mode)
    python scripts/enable_cluster_protocols.py --swim

    # Enable Raft on voter nodes only
    python scripts/enable_cluster_protocols.py --raft --voters-only

    # Enable both and restart P2P
    python scripts/enable_cluster_protocols.py --swim --raft --restart

    # Dry run
    python scripts/enable_cluster_protocols.py --swim --raft --dry-run

December 28, 2025
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.cluster_config import get_cluster_nodes, ClusterNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# P2P voter nodes (7-node quorum from distributed_hosts.yaml)
VOTER_NODES = [
    "nebius-backbone-1",
    "nebius-h100-3",
    "hetzner-cpu1",
    "hetzner-cpu2",
    "vultr-a100-20gb",
    "lambda-gh200-1",
    "lambda-gh200-2",
]


@dataclass
class EnableResult:
    """Result of enabling protocols on a node."""
    node_name: str
    success: bool
    message: str
    swim_enabled: bool = False
    raft_enabled: bool = False
    p2p_restarted: bool = False


async def run_ssh_command(
    node: ClusterNode,
    command: str,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """Run SSH command on node."""
    ssh_host = node.best_ip
    if not ssh_host:
        return False, "No SSH host available"

    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]

    if node.ssh_key:
        key_path = Path(node.ssh_key).expanduser()
        if key_path.exists():
            ssh_args.extend(["-i", str(key_path)])

    if node.ssh_port and node.ssh_port != 22:
        ssh_args.extend(["-p", str(node.ssh_port)])

    ssh_user = node.ssh_user or "ubuntu"
    ssh_args.append(f"{ssh_user}@{ssh_host}")
    ssh_args.append(command)

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode() + stderr.decode()
        return proc.returncode == 0, output.strip()
    except asyncio.TimeoutError:
        return False, f"SSH timeout after {timeout}s"
    except Exception as e:
        return False, f"SSH error: {e}"


def get_env_file_path(node: ClusterNode) -> str:
    """Get the path to the environment file on the node."""
    ringrift_path = getattr(node, "ringrift_path", "~/ringrift/ai-service")
    # Use .env.local for node-specific config
    return f"{ringrift_path}/.env.local"


async def enable_on_node(
    node: ClusterNode,
    enable_swim: bool = False,
    enable_raft: bool = False,
    restart_p2p: bool = False,
    dry_run: bool = False,
) -> EnableResult:
    """Enable protocols on a single node."""
    node_name = node.name

    # Skip non-ready nodes
    if node.status != "ready":
        return EnableResult(
            node_name=node_name,
            success=False,
            message=f"Node not ready (status={node.status})",
        )

    if not enable_swim and not enable_raft:
        return EnableResult(
            node_name=node_name,
            success=True,
            message="Nothing to enable",
        )

    logger.info(f"[{node_name}] Enabling protocols...")

    # Build environment variables to set
    env_vars = []
    if enable_swim:
        env_vars.append("RINGRIFT_SWIM_ENABLED=true")
        env_vars.append("RINGRIFT_MEMBERSHIP_MODE=hybrid")
    if enable_raft:
        env_vars.append("RINGRIFT_RAFT_ENABLED=true")
        env_vars.append("RINGRIFT_CONSENSUS_MODE=hybrid")

    env_file = get_env_file_path(node)

    # Create command to update .env.local
    # First, remove any existing entries for these variables, then append new ones
    remove_cmd = " && ".join([
        f"sed -i'' -e '/^{var.split('=')[0]}=/d' {env_file} 2>/dev/null || true"
        for var in env_vars
    ])

    append_cmd = " && ".join([
        f"echo '{var}' >> {env_file}"
        for var in env_vars
    ])

    # Create file if doesn't exist
    full_cmd = f"touch {env_file} && {remove_cmd} && {append_cmd}"

    if dry_run:
        logger.info(f"[{node_name}] Would run: {full_cmd}")
        if restart_p2p:
            logger.info(f"[{node_name}] Would restart P2P orchestrator")
        return EnableResult(
            node_name=node_name,
            success=True,
            message="Dry run - would enable protocols",
            swim_enabled=enable_swim,
            raft_enabled=enable_raft,
            p2p_restarted=restart_p2p,
        )

    # Execute command
    success, output = await run_ssh_command(node, full_cmd, timeout=30)

    if not success:
        return EnableResult(
            node_name=node_name,
            success=False,
            message=f"Failed to update env: {output[:100]}",
        )

    logger.info(f"[{node_name}] Environment updated")

    # Optionally restart P2P
    p2p_restarted = False
    if restart_p2p:
        # Find and restart P2P orchestrator
        restart_cmd = (
            "pkill -f 'p2p_orchestrator' 2>/dev/null || true && "
            "sleep 1 && "
            f"cd {getattr(node, 'ringrift_path', '~/ringrift/ai-service')} && "
            "nohup python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &"
        )

        success, output = await run_ssh_command(node, restart_cmd, timeout=30)
        if success:
            logger.info(f"[{node_name}] P2P restarted")
            p2p_restarted = True
        else:
            logger.warning(f"[{node_name}] P2P restart failed: {output[:100]}")

    return EnableResult(
        node_name=node_name,
        success=True,
        message="Protocols enabled",
        swim_enabled=enable_swim,
        raft_enabled=enable_raft,
        p2p_restarted=p2p_restarted,
    )


async def enable_all(
    node_filter: Optional[list[str]] = None,
    enable_swim: bool = False,
    enable_raft: bool = False,
    voters_only: bool = False,
    restart_p2p: bool = False,
    dry_run: bool = False,
    max_parallel: int = 10,
) -> list[EnableResult]:
    """Enable protocols on all nodes."""
    nodes = get_cluster_nodes()

    # Filter to voters only if requested
    if voters_only:
        nodes = {name: node for name, node in nodes.items() if name in VOTER_NODES}

    # Additional filter
    if node_filter:
        nodes = {name: node for name, node in nodes.items() if name in node_filter}

    if not nodes:
        logger.error("No nodes found")
        return []

    logger.info(f"Enabling protocols on {len(nodes)} nodes (max_parallel={max_parallel})")

    sem = asyncio.Semaphore(max_parallel)

    async def enable_with_sem(node: ClusterNode) -> EnableResult:
        async with sem:
            return await enable_on_node(
                node, enable_swim, enable_raft, restart_p2p, dry_run
            )

    tasks = [enable_with_sem(node) for node in nodes.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            node_name = list(nodes.keys())[i]
            final_results.append(EnableResult(
                node_name=node_name,
                success=False,
                message=f"Exception: {result}",
            ))
        else:
            final_results.append(result)

    return final_results


def print_summary(results: list[EnableResult]) -> None:
    """Print summary."""
    print("\n" + "=" * 60)
    print("ENABLE PROTOCOLS SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r.success)
    swim_count = sum(1 for r in results if r.swim_enabled)
    raft_count = sum(1 for r in results if r.raft_enabled)
    restart_count = sum(1 for r in results if r.p2p_restarted)

    print(f"\nTotal nodes: {len(results)}")
    print(f"Successful:  {success_count}")
    print(f"SWIM enabled: {swim_count}")
    print(f"Raft enabled: {raft_count}")
    print(f"P2P restarted: {restart_count}")

    print("\nDetailed results:")
    for result in sorted(results, key=lambda r: r.node_name):
        status = "OK" if result.success else "FAILED"
        protocols = []
        if result.swim_enabled:
            protocols.append("SWIM")
        if result.raft_enabled:
            protocols.append("Raft")
        if result.p2p_restarted:
            protocols.append("restarted")
        proto_str = ", ".join(protocols) if protocols else result.message
        print(f"  {result.node_name}: [{status}] {proto_str}")

    print("\n" + "=" * 60)

    if success_count > 0:
        print("\nProtocols will take effect after P2P restart.")
        print("To manually restart P2P on a node:")
        print("  pkill -f p2p_orchestrator && python scripts/p2p_orchestrator.py")
        print("\nTo verify SWIM/Raft status:")
        print("  curl http://<node>:8770/status | jq .swim_raft")


def main():
    parser = argparse.ArgumentParser(
        description="Enable SWIM/Raft protocols on cluster nodes"
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node names",
    )
    parser.add_argument(
        "--swim",
        action="store_true",
        help="Enable SWIM protocol (faster failure detection)",
    )
    parser.add_argument(
        "--raft",
        action="store_true",
        help="Enable Raft protocol (replicated work queue)",
    )
    parser.add_argument(
        "--voters-only",
        action="store_true",
        help="Only enable on voter nodes (for Raft)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart P2P orchestrators after enabling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum parallel operations (default: 10)",
    )

    args = parser.parse_args()

    if not args.swim and not args.raft:
        logger.error("Must specify --swim and/or --raft")
        sys.exit(1)

    node_filter = None
    if args.nodes:
        node_filter = [n.strip() for n in args.nodes.split(",")]

    results = asyncio.run(enable_all(
        node_filter=node_filter,
        enable_swim=args.swim,
        enable_raft=args.raft,
        voters_only=args.voters_only,
        restart_p2p=args.restart,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    ))

    print_summary(results)

    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
