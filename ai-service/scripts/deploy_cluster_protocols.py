#!/usr/bin/env python3
"""Deploy SWIM/Raft protocol dependencies to cluster nodes.

This script installs the required dependencies for SWIM and Raft protocols
on cluster nodes, enabling improved reliability features:
- SWIM: 5s failure detection (vs 60-90s HTTP polling)
- Raft: Replicated work queue with sub-second leader failover

Usage:
    # Install on all nodes
    python scripts/deploy_cluster_protocols.py

    # Install on specific nodes
    python scripts/deploy_cluster_protocols.py --nodes nebius-h100-3,runpod-h100

    # Install SWIM only
    python scripts/deploy_cluster_protocols.py --swim-only

    # Install Raft only
    python scripts/deploy_cluster_protocols.py --raft-only

    # Dry run (preview commands)
    python scripts/deploy_cluster_protocols.py --dry-run

After installation:
    # Enable SWIM on each node
    export RINGRIFT_SWIM_ENABLED=true
    export RINGRIFT_MEMBERSHIP_MODE=swim  # or hybrid

    # Enable Raft on each node
    export RINGRIFT_RAFT_ENABLED=true
    export RINGRIFT_CONSENSUS_MODE=raft  # or hybrid

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


@dataclass
class DeployResult:
    """Result of deploying to a node."""
    node_name: str
    success: bool
    message: str
    swim_installed: bool = False
    raft_installed: bool = False


# Package requirements
SWIM_PACKAGE = "swim-p2p>=1.2.0"
RAFT_PACKAGE = "pysyncobj>=0.3.14"


async def run_ssh_command(
    node: ClusterNode,
    command: str,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """Run SSH command on node.

    Args:
        node: Cluster node
        command: Command to run
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, output)
    """
    ssh_host = node.best_ip
    if not ssh_host:
        return False, "No SSH host available"

    # Build SSH command
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]

    # Add SSH key if configured
    if node.ssh_key:
        key_path = Path(node.ssh_key).expanduser()
        if key_path.exists():
            ssh_args.extend(["-i", str(key_path)])

    # Add port if non-standard
    if node.ssh_port and node.ssh_port != 22:
        ssh_args.extend(["-p", str(node.ssh_port)])

    # Add user@host
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


async def deploy_to_node(
    node: ClusterNode,
    install_swim: bool = True,
    install_raft: bool = True,
    dry_run: bool = False,
) -> DeployResult:
    """Deploy protocol dependencies to a node.

    Args:
        node: Cluster node
        install_swim: Install SWIM package
        install_raft: Install Raft package
        dry_run: Preview commands without executing

    Returns:
        DeployResult with status
    """
    node_name = node.name

    # Skip coordinator-only nodes
    if node.role == "coordinator" and not node.is_gpu_node:
        return DeployResult(
            node_name=node_name,
            success=True,
            message="Skipped (coordinator-only)",
        )

    # Skip nodes that are not ready
    if node.status != "ready":
        return DeployResult(
            node_name=node_name,
            success=False,
            message=f"Node not ready (status={node.status})",
        )

    logger.info(f"[{node_name}] Deploying protocol dependencies...")

    # Get Python path on node
    ringrift_path = getattr(node, "ringrift_path", "~/ringrift/ai-service")
    venv_path = f"{ringrift_path}/venv/bin/pip"

    # Try venv pip first, then system pip
    pip_commands = [
        f"cd {ringrift_path} && {venv_path}",  # venv pip
        "pip3",  # system pip3
        "pip",   # fallback
    ]

    packages_to_install = []
    if install_swim:
        # Quote package spec to prevent shell interpretation of >=
        packages_to_install.append(f"'{SWIM_PACKAGE}'")
    if install_raft:
        packages_to_install.append(f"'{RAFT_PACKAGE}'")

    if not packages_to_install:
        return DeployResult(
            node_name=node_name,
            success=True,
            message="No packages to install",
        )

    packages_str = " ".join(packages_to_install)

    # First check if packages are already installed
    check_cmd = "python3 -c 'import swim; print(\"swim:\", swim.__version__)' 2>/dev/null || echo 'swim: not installed'"
    check_cmd += " && python3 -c 'import pysyncobj; print(\"pysyncobj: installed\")' 2>/dev/null || echo 'pysyncobj: not installed'"

    if dry_run:
        logger.info(f"[{node_name}] Would check: {check_cmd}")
        logger.info(f"[{node_name}] Would install: pip install {packages_str}")
        return DeployResult(
            node_name=node_name,
            success=True,
            message="Dry run - would install packages",
            swim_installed=install_swim,
            raft_installed=install_raft,
        )

    # Check current status
    success, check_output = await run_ssh_command(node, check_cmd, timeout=30)
    if success:
        logger.info(f"[{node_name}] Current status: {check_output}")

    # Try each pip path
    installed = False
    last_error = ""

    for pip_cmd in pip_commands:
        install_cmd = f"{pip_cmd} install {packages_str}"
        logger.info(f"[{node_name}] Trying: {install_cmd}")

        success, output = await run_ssh_command(node, install_cmd, timeout=120)

        if success:
            logger.info(f"[{node_name}] Installation successful")
            installed = True
            break
        else:
            last_error = output
            logger.warning(f"[{node_name}] Failed with {pip_cmd}: {output[:100]}")

    if not installed:
        return DeployResult(
            node_name=node_name,
            success=False,
            message=f"All pip paths failed: {last_error[:100]}",
        )

    # Verify installation using venv python
    verify_msgs = []
    swim_ok = False
    raft_ok = False

    # Use venv python for verification (same as where pip installed)
    venv_python = f"{ringrift_path}/venv/bin/python"

    if install_swim:
        # Try venv python first, then system python
        success, output = await run_ssh_command(
            node,
            f"{venv_python} -c 'import swim; print(swim.__version__)' 2>/dev/null || python3 -c 'import swim; print(swim.__version__)'",
            timeout=30,
        )
        swim_ok = success
        verify_msgs.append(f"SWIM: {'OK' if success else 'FAILED'}")

    if install_raft:
        success, output = await run_ssh_command(
            node,
            f"{venv_python} -c 'import pysyncobj; print(\"OK\")' 2>/dev/null || python3 -c 'import pysyncobj; print(\"OK\")'",
            timeout=30,
        )
        raft_ok = success
        verify_msgs.append(f"Raft: {'OK' if success else 'FAILED'}")

    return DeployResult(
        node_name=node_name,
        success=swim_ok or raft_ok,
        message=", ".join(verify_msgs),
        swim_installed=swim_ok,
        raft_installed=raft_ok,
    )


async def deploy_all(
    node_filter: Optional[list[str]] = None,
    install_swim: bool = True,
    install_raft: bool = True,
    dry_run: bool = False,
    max_parallel: int = 5,
) -> list[DeployResult]:
    """Deploy to all nodes in parallel.

    Args:
        node_filter: Optional list of node names to deploy to
        install_swim: Install SWIM package
        install_raft: Install Raft package
        dry_run: Preview commands without executing
        max_parallel: Maximum parallel deployments

    Returns:
        List of DeployResults
    """
    nodes = get_cluster_nodes()

    # Filter nodes
    if node_filter:
        nodes = {name: node for name, node in nodes.items() if name in node_filter}

    if not nodes:
        logger.error("No nodes found to deploy to")
        return []

    logger.info(f"Deploying to {len(nodes)} nodes (max_parallel={max_parallel})")

    # Create semaphore for parallel control
    sem = asyncio.Semaphore(max_parallel)

    async def deploy_with_sem(node: ClusterNode) -> DeployResult:
        async with sem:
            return await deploy_to_node(node, install_swim, install_raft, dry_run)

    # Deploy in parallel
    tasks = [deploy_with_sem(node) for node in nodes.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            node_name = list(nodes.keys())[i]
            final_results.append(DeployResult(
                node_name=node_name,
                success=False,
                message=f"Exception: {result}",
            ))
        else:
            final_results.append(result)

    return final_results


def print_summary(results: list[DeployResult]) -> None:
    """Print deployment summary."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r.success)
    swim_count = sum(1 for r in results if r.swim_installed)
    raft_count = sum(1 for r in results if r.raft_installed)

    print(f"\nTotal nodes: {len(results)}")
    print(f"Successful:  {success_count}")
    print(f"SWIM installed: {swim_count}")
    print(f"Raft installed: {raft_count}")

    print("\nDetailed results:")
    for result in sorted(results, key=lambda r: r.node_name):
        status = "OK" if result.success else "FAILED"
        print(f"  {result.node_name}: [{status}] {result.message}")

    print("\n" + "=" * 60)

    if success_count > 0:
        print("\nTo enable SWIM (faster failure detection):")
        print("  export RINGRIFT_SWIM_ENABLED=true")
        print("  export RINGRIFT_MEMBERSHIP_MODE=swim  # or 'hybrid'")
        print("")
        print("To enable Raft (replicated work queue):")
        print("  export RINGRIFT_RAFT_ENABLED=true")
        print("  export RINGRIFT_CONSENSUS_MODE=raft  # or 'hybrid'")
        print("")
        print("Then restart P2P orchestrators on all nodes.")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy SWIM/Raft protocol dependencies to cluster nodes"
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node names to deploy to",
    )
    parser.add_argument(
        "--swim-only",
        action="store_true",
        help="Install SWIM only (skip Raft)",
    )
    parser.add_argument(
        "--raft-only",
        action="store_true",
        help="Install Raft only (skip SWIM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum parallel deployments (default: 5)",
    )

    args = parser.parse_args()

    node_filter = None
    if args.nodes:
        node_filter = [n.strip() for n in args.nodes.split(",")]

    install_swim = not args.raft_only
    install_raft = not args.swim_only

    if not install_swim and not install_raft:
        logger.error("No packages to install (--swim-only and --raft-only conflict)")
        sys.exit(1)

    results = asyncio.run(deploy_all(
        node_filter=node_filter,
        install_swim=install_swim,
        install_raft=install_raft,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    ))

    print_summary(results)

    # Exit with error if any deployments failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
