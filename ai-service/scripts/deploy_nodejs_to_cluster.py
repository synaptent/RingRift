#!/usr/bin/env python3
"""Deploy Node.js to cluster nodes for parity gate validation.

This script installs Node.js (LTS) on cluster nodes to enable TypeScript/Python
parity validation via `npx ts-node`. Parity gates are currently failing on
cluster nodes with "pending_gate" status because `npx` is not available.

Usage:
    # Install on all nodes
    python scripts/deploy_nodejs_to_cluster.py

    # Install on specific nodes
    python scripts/deploy_nodejs_to_cluster.py --nodes nebius-h100-3,runpod-h100

    # Dry run (preview commands)
    python scripts/deploy_nodejs_to_cluster.py --dry-run

    # Skip verification after install
    python scripts/deploy_nodejs_to_cluster.py --skip-verify

After installation, parity gates will work:
    python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db

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
    node_version: Optional[str] = None
    npm_version: Optional[str] = None


# Node.js installation commands for different platforms
# Using NodeSource setup script for consistent LTS installation
# NOTE: No inline comments allowed - they break when commands are joined with &&
INSTALL_COMMANDS = {
    "debian": """curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && \
sudo apt-get install -y nodejs && \
sudo npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
    "debian_root": """curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
apt-get install -y nodejs && \
npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
    "rhel": """curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash - && \
sudo yum install -y nodejs && \
sudo npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
    "rhel_root": """curl -fsSL https://rpm.nodesource.com/setup_lts.x | bash - && \
yum install -y nodejs && \
npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
    "alpine": """apk add --no-cache nodejs npm && \
npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
    "nvm": """curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
export NVM_DIR="$HOME/.nvm" && \
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
nvm install --lts && \
nvm use --lts && \
npm install -g ts-node typescript && \
echo "Node.js installation complete" """,
}


async def run_ssh_command(
    node: ClusterNode,
    command: str,
    timeout: float = 300.0,
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


async def check_node_version(node: ClusterNode) -> tuple[Optional[str], Optional[str]]:
    """Check Node.js and npm versions on node.

    Returns:
        Tuple of (node_version, npm_version) or (None, None) if not installed
    """
    # Check node version
    success, node_out = await run_ssh_command(node, "node --version 2>/dev/null || echo 'not installed'", timeout=30)
    node_version = node_out.strip() if success and not node_out.startswith("not") else None

    # Check npm version
    success, npm_out = await run_ssh_command(node, "npm --version 2>/dev/null || echo 'not installed'", timeout=30)
    npm_version = npm_out.strip() if success and not npm_out.startswith("not") else None

    return node_version, npm_version


async def detect_os_type(node: ClusterNode) -> str:
    """Detect OS type for appropriate installation commands.

    Returns:
        One of: "debian", "debian_root", "rhel", "rhel_root", "alpine", "nvm"
    """
    # Check for sudo availability (container vs VM)
    has_sudo, _ = await run_ssh_command(node, "which sudo", timeout=10)

    # Check for Debian/Ubuntu
    success, _ = await run_ssh_command(node, "which apt-get", timeout=10)
    if success:
        return "debian" if has_sudo else "debian_root"

    # Check for RHEL/CentOS/Fedora
    success, _ = await run_ssh_command(node, "which yum", timeout=10)
    if success:
        return "rhel" if has_sudo else "rhel_root"

    # Check for Alpine (typically doesn't use sudo)
    success, _ = await run_ssh_command(node, "which apk", timeout=10)
    if success:
        return "alpine"

    # Default to nvm (works anywhere with bash)
    return "nvm"


async def deploy_to_node(
    node: ClusterNode,
    dry_run: bool = False,
    skip_verify: bool = False,
    force: bool = False,
) -> DeployResult:
    """Deploy Node.js to a node.

    Args:
        node: Cluster node
        dry_run: Preview commands without executing
        skip_verify: Skip verification after install
        force: Force reinstall even if already installed

    Returns:
        DeployResult with status
    """
    node_name = node.name

    # Skip coordinator-only nodes (they don't need Node.js for parity)
    if node.role == "coordinator" and node_name in ("mac-studio", "local-mac"):
        return DeployResult(
            node_name=node_name,
            success=True,
            message="Skipped (coordinator)",
        )

    # Skip nodes that are not ready
    if node.status != "ready":
        return DeployResult(
            node_name=node_name,
            success=False,
            message=f"Node not ready (status={node.status})",
        )

    logger.info(f"[{node_name}] Checking Node.js status...")

    # Check if already installed
    node_version, npm_version = await check_node_version(node)

    if node_version and not force:
        logger.info(f"[{node_name}] Node.js already installed: {node_version}")
        return DeployResult(
            node_name=node_name,
            success=True,
            message=f"Already installed (node {node_version})",
            node_version=node_version,
            npm_version=npm_version,
        )

    # Detect OS type
    os_type = await detect_os_type(node)
    logger.info(f"[{node_name}] Detected OS type: {os_type}")

    install_cmd = INSTALL_COMMANDS.get(os_type, INSTALL_COMMANDS["nvm"])
    # Clean up the command - just normalize whitespace since commands are already properly formatted
    install_cmd = " ".join(install_cmd.split())

    if dry_run:
        logger.info(f"[{node_name}] Would run: {install_cmd[:100]}...")
        return DeployResult(
            node_name=node_name,
            success=True,
            message=f"Dry run - would install via {os_type}",
        )

    # Run installation
    logger.info(f"[{node_name}] Installing Node.js via {os_type}...")
    success, output = await run_ssh_command(node, install_cmd, timeout=300)

    if not success:
        logger.error(f"[{node_name}] Installation failed: {output[:200]}")
        return DeployResult(
            node_name=node_name,
            success=False,
            message=f"Installation failed: {output[:100]}",
        )

    logger.info(f"[{node_name}] Installation output: {output[-200:]}")

    # Verify installation
    if not skip_verify:
        node_version, npm_version = await check_node_version(node)

        if not node_version:
            return DeployResult(
                node_name=node_name,
                success=False,
                message="Installation completed but node not found in PATH",
            )

        # Check npx specifically (required for parity gates)
        success, npx_out = await run_ssh_command(node, "npx --version 2>/dev/null", timeout=30)
        if not success:
            logger.warning(f"[{node_name}] npx not available after install")

        # Check ts-node
        success, ts_out = await run_ssh_command(node, "npx ts-node --version 2>/dev/null", timeout=30)
        ts_version = ts_out.strip() if success else None

        logger.info(f"[{node_name}] Verified: node={node_version}, npm={npm_version}, ts-node={ts_version}")

    return DeployResult(
        node_name=node_name,
        success=True,
        message=f"Installed node {node_version or 'unknown'}",
        node_version=node_version,
        npm_version=npm_version,
    )


async def deploy_all(
    node_filter: Optional[list[str]] = None,
    dry_run: bool = False,
    skip_verify: bool = False,
    force: bool = False,
    max_parallel: int = 5,
) -> list[DeployResult]:
    """Deploy to all nodes in parallel.

    Args:
        node_filter: Optional list of node names to deploy to
        dry_run: Preview commands without executing
        skip_verify: Skip verification after install
        force: Force reinstall even if already installed
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

    # Filter out coordinator-only nodes
    deployable_nodes = {
        name: node for name, node in nodes.items()
        if node.status == "ready" and name not in ("mac-studio", "local-mac")
    }

    logger.info(f"Deploying Node.js to {len(deployable_nodes)} nodes (max_parallel={max_parallel})")

    # Create semaphore for parallel control
    sem = asyncio.Semaphore(max_parallel)

    async def deploy_with_sem(node: ClusterNode) -> DeployResult:
        async with sem:
            return await deploy_to_node(node, dry_run, skip_verify, force)

    # Deploy in parallel
    tasks = [deploy_with_sem(node) for node in deployable_nodes.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            node_name = list(deployable_nodes.keys())[i]
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
    print("NODE.JS DEPLOYMENT SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r.success)
    with_version = sum(1 for r in results if r.node_version)

    print(f"\nTotal nodes: {len(results)}")
    print(f"Successful:  {success_count}")
    print(f"With Node.js: {with_version}")

    print("\nDetailed results:")
    for result in sorted(results, key=lambda r: r.node_name):
        status = "OK" if result.success else "FAILED"
        version = f" ({result.node_version})" if result.node_version else ""
        print(f"  {result.node_name}: [{status}] {result.message}{version}")

    print("\n" + "=" * 60)

    if success_count > 0:
        print("\nParity validation now available on these nodes:")
        print("  python scripts/check_ts_python_replay_parity.py --db data/games/canonical_hex8.db")
        print("")
        print("To run parity gates during selfplay:")
        print("  # No additional flags needed - parity gates auto-enable when npx is available")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Node.js to cluster nodes for parity validation"
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of node names to deploy to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification after installation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall even if already installed",
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

    results = asyncio.run(deploy_all(
        node_filter=node_filter,
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
        force=args.force,
        max_parallel=args.max_parallel,
    ))

    print_summary(results)

    # Exit with error if any deployments failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
