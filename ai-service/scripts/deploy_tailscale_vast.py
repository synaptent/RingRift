#!/usr/bin/env python3
"""
Deploy Tailscale to Vast.ai nodes for P2P mesh connectivity.

Jan 19, 2026: Part of P2P reliability improvements for CGNAT bypass.

Usage:
    # Set auth key first (get from https://login.tailscale.com/admin/settings/keys)
    export TAILSCALE_AUTH_KEY=tskey-auth-xxx

    # Deploy to all active Vast nodes
    python scripts/deploy_tailscale_vast.py

    # Deploy to specific nodes
    python scripts/deploy_tailscale_vast.py --nodes vast-28889766,vast-28918742

    # Dry run (show what would be done)
    python scripts/deploy_tailscale_vast.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.cluster_config import get_cluster_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DeployResult:
    """Result of Tailscale deployment to a node."""

    node_id: str
    success: bool
    tailscale_ip: str | None = None
    error: str | None = None
    already_installed: bool = False


def get_vast_nodes() -> list[dict[str, Any]]:
    """Get all active Vast.ai nodes from config."""
    nodes_dict = get_cluster_nodes()
    vast_nodes = []

    for node_id, node in nodes_dict.items():
        if not node_id.startswith("vast-"):
            continue

        status = getattr(node, "status", "")
        if status not in ("active", "online", "ready", ""):
            logger.debug(f"Skipping {node_id}: status={status}")
            continue

        # Convert ClusterNode to dict for compatibility
        vast_nodes.append({
            "name": node_id,
            "ssh_host": getattr(node, "ssh_host", ""),
            "ssh_port": getattr(node, "ssh_port", 22),
            "ssh_user": getattr(node, "ssh_user", "root"),
            "ssh_key": getattr(node, "ssh_key", None),
            "tailscale_ip": getattr(node, "tailscale_ip", None),
            "status": status,
        })

    return vast_nodes


def get_ssh_command(node: dict[str, Any]) -> list[str]:
    """Get SSH command for connecting to a Vast node."""
    ssh_host = node.get("ssh_host", "")
    ssh_port = node.get("ssh_port", 22)
    ssh_user = node.get("ssh_user", "root")

    if not ssh_host:
        raise ValueError(f"No SSH host for {node.get('name')}")

    cmd = [
        "ssh",
        "-o", "ConnectTimeout=30",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]

    if ssh_port != 22:
        cmd.extend(["-p", str(ssh_port)])

    # Add SSH key if specified
    ssh_key = node.get("ssh_key")
    if ssh_key:
        key_path = os.path.expanduser(ssh_key)
        if os.path.exists(key_path):
            cmd.extend(["-i", key_path])

    cmd.append(f"{ssh_user}@{ssh_host}")
    return cmd


async def deploy_to_node(
    node: dict[str, Any],
    auth_key: str,
    dry_run: bool = False,
) -> DeployResult:
    """Deploy Tailscale to a single Vast node."""
    node_id = node.get("name", "unknown")

    if dry_run:
        logger.info(f"[DRY RUN] Would deploy Tailscale to {node_id}")
        return DeployResult(node_id=node_id, success=True, error="dry_run")

    try:
        ssh_cmd = get_ssh_command(node)

        # First check if Tailscale is already installed and connected
        check_cmd = ssh_cmd + [
            "tailscale ip -4 2>/dev/null || echo 'NOT_CONNECTED'"
        ]

        logger.info(f"[{node_id}] Checking Tailscale status...")

        proc = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode().strip()

        if output and output != "NOT_CONNECTED" and not output.startswith("error"):
            # Already connected
            logger.info(f"[{node_id}] Tailscale already connected: {output}")
            return DeployResult(
                node_id=node_id,
                success=True,
                tailscale_ip=output,
                already_installed=True,
            )

        # Install/setup Tailscale
        logger.info(f"[{node_id}] Installing Tailscale...")

        # Read the setup script
        script_path = Path(__file__).parent / "setup_tailscale_vast.sh"
        if not script_path.exists():
            return DeployResult(
                node_id=node_id,
                success=False,
                error="setup_tailscale_vast.sh not found",
            )

        script_content = script_path.read_text()

        # Run the script remotely with auth key
        install_cmd = ssh_cmd + [
            f"TAILSCALE_AUTH_KEY={auth_key} bash -s"
        ]

        proc = await asyncio.create_subprocess_exec(
            *install_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=script_content.encode()),
            timeout=180,  # 3 minutes for install
        )

        output = stdout.decode()
        errors = stderr.decode()

        if proc.returncode != 0:
            logger.error(f"[{node_id}] Installation failed: {errors}")
            return DeployResult(
                node_id=node_id,
                success=False,
                error=errors[:500],
            )

        # Extract Tailscale IP from output
        tailscale_ip = None
        for line in output.split("\n"):
            if "Tailscale IP:" in line:
                tailscale_ip = line.split(":")[-1].strip()
                break

        if not tailscale_ip:
            # Try to get it directly
            get_ip_cmd = ssh_cmd + ["tailscale ip -4"]
            proc = await asyncio.create_subprocess_exec(
                *get_ip_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            tailscale_ip = stdout.decode().strip()

        logger.info(f"[{node_id}] Tailscale installed: {tailscale_ip}")
        return DeployResult(
            node_id=node_id,
            success=True,
            tailscale_ip=tailscale_ip,
        )

    except asyncio.TimeoutError:
        logger.error(f"[{node_id}] Timeout during deployment")
        return DeployResult(node_id=node_id, success=False, error="timeout")
    except Exception as e:
        logger.error(f"[{node_id}] Deployment failed: {e}")
        return DeployResult(node_id=node_id, success=False, error=str(e))


async def main():
    parser = argparse.ArgumentParser(
        description="Deploy Tailscale to Vast.ai nodes"
    )
    parser.add_argument(
        "--nodes",
        help="Comma-separated list of node IDs (default: all active Vast nodes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Max parallel deployments (default: 5)",
    )
    args = parser.parse_args()

    # Get auth key
    auth_key = os.environ.get("TAILSCALE_AUTH_KEY")
    if not auth_key and not args.dry_run:
        print("ERROR: TAILSCALE_AUTH_KEY environment variable not set")
        print("Get a reusable auth key from: https://login.tailscale.com/admin/settings/keys")
        print("Recommended settings: Reusable=Yes, Ephemeral=Yes, Tags=tag:vast")
        sys.exit(1)

    # Get target nodes
    if args.nodes:
        target_ids = set(args.nodes.split(","))
        all_nodes = get_vast_nodes()
        nodes = [n for n in all_nodes if n.get("name") in target_ids]
        if not nodes:
            print(f"ERROR: No matching nodes found for: {args.nodes}")
            print(f"Available Vast nodes: {[n.get('name') for n in all_nodes]}")
            sys.exit(1)
    else:
        nodes = get_vast_nodes()

    if not nodes:
        print("No active Vast nodes found in config")
        sys.exit(0)

    print(f"Deploying Tailscale to {len(nodes)} Vast nodes:")
    for node in nodes:
        print(f"  - {node.get('name')}")
    print()

    # Deploy in parallel with semaphore
    semaphore = asyncio.Semaphore(args.parallel)

    async def deploy_with_semaphore(node):
        async with semaphore:
            return await deploy_to_node(node, auth_key or "", args.dry_run)

    tasks = [deploy_with_semaphore(node) for node in nodes]
    results = await asyncio.gather(*tasks)

    # Print summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    already_installed = [r for r in results if r.already_installed]

    if successes:
        print(f"\n✅ Successfully deployed: {len(successes)} nodes")
        for r in successes:
            ip_info = f" ({r.tailscale_ip})" if r.tailscale_ip else ""
            status = " [already installed]" if r.already_installed else ""
            print(f"  - {r.node_id}{ip_info}{status}")

    if failures:
        print(f"\n❌ Failed: {len(failures)} nodes")
        for r in failures:
            print(f"  - {r.node_id}: {r.error}")

    # Output YAML snippet for config update
    new_ips = [r for r in successes if r.tailscale_ip and not r.already_installed]
    if new_ips:
        print("\n" + "-" * 60)
        print("Add to config/distributed_hosts.yaml:")
        print("-" * 60)
        for r in new_ips:
            print(f"# {r.node_id}:")
            print(f"#   tailscale_ip: {r.tailscale_ip}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
