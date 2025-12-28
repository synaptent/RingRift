#!/usr/bin/env python3
"""Force restart P2P orchestrators on all cluster nodes.

This script:
1. Reads nodes from config/distributed_hosts.yaml
2. SSHs to each node in parallel (max 10 concurrent)
3. Kills any existing p2p_orchestrator process
4. Starts a new P2P orchestrator that loads .env.local

Usage:
    python scripts/force_restart_p2p_cluster.py
    python scripts/force_restart_p2p_cluster.py --dry-run
    python scripts/force_restart_p2p_cluster.py --max-parallel 5
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    name: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_key: str
    ringrift_path: str
    venv_activate: str
    status: str
    role: str
    p2p_enabled: bool


def load_cluster_nodes() -> list[NodeConfig]:
    """Load node configurations from distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    nodes = []
    for name, host_config in config.get("hosts", {}).items():
        # Skip nodes that are not ready or not P2P enabled
        status = host_config.get("status", "unknown")
        p2p_enabled = host_config.get("p2p_enabled", False)
        role = host_config.get("role", "unknown")

        # Skip coordinator nodes (they have different startup procedures)
        if role == "coordinator":
            continue

        # Skip proxy nodes
        if role == "proxy" or status == "proxy_only":
            continue

        # Skip retired/offline nodes
        if status in ("retired", "offline"):
            continue

        # Skip nodes without ringrift_path
        if not host_config.get("ringrift_path"):
            continue

        # Skip nodes without P2P enabled
        if not p2p_enabled:
            continue

        # Only process ready nodes
        if status != "ready":
            continue

        nodes.append(NodeConfig(
            name=name,
            ssh_host=host_config.get("ssh_host", ""),
            ssh_port=host_config.get("ssh_port", 22),
            ssh_user=host_config.get("ssh_user", "root"),
            ssh_key=host_config.get("ssh_key", "~/.ssh/id_cluster"),
            ringrift_path=host_config.get("ringrift_path", ""),
            venv_activate=host_config.get("venv_activate", ""),
            status=status,
            role=role,
            p2p_enabled=p2p_enabled,
        ))

    return nodes


@dataclass
class RestartResult:
    """Result of a restart operation on a node."""
    node_name: str
    success: bool
    message: str
    kill_output: str = ""
    start_output: str = ""


async def restart_p2p_on_node(
    node: NodeConfig,
    dry_run: bool = False,
    timeout: int = 60,
) -> RestartResult:
    """Restart P2P orchestrator on a single node."""

    # Expand SSH key path
    ssh_key = os.path.expanduser(node.ssh_key)

    # Build SSH command base
    ssh_base = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-o", "BatchMode=yes",
        "-i", ssh_key,
        "-p", str(node.ssh_port),
        f"{node.ssh_user}@{node.ssh_host}",
    ]

    # Step 1: Kill existing P2P orchestrator process
    kill_cmd = "pkill -9 -f 'python.*p2p_orchestrator' || true"

    # Step 2: Start new P2P orchestrator with .env.local loading
    # The key challenge is that we need .env.local loaded BEFORE Python imports
    # The p2p_orchestrator.py already has _load_env_local() at the top that handles this

    # Build the start command with proper path and venv
    cd_cmd = f"cd {node.ringrift_path}"

    # Handle venv activation
    if node.venv_activate and node.venv_activate.strip() != ":":
        activate_cmd = node.venv_activate
    else:
        activate_cmd = "true"  # No-op if no venv

    # Start P2P orchestrator in background with nohup
    # The script's _load_env_local() will load .env.local automatically
    start_cmd = (
        f"nohup python scripts/p2p_orchestrator.py --node-id {node.name} "
        f"> logs/p2p_orchestrator.log 2>&1 &"
    )

    # Combine into full remote command
    full_cmd = f"{cd_cmd} && {activate_cmd} && mkdir -p logs && {kill_cmd} && sleep 2 && {start_cmd}"

    if dry_run:
        return RestartResult(
            node_name=node.name,
            success=True,
            message=f"[DRY RUN] Would execute: {full_cmd}",
        )

    try:
        # Execute the combined command
        proc = await asyncio.create_subprocess_exec(
            *ssh_base,
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return RestartResult(
                node_name=node.name,
                success=False,
                message=f"Timeout after {timeout}s",
            )

        if proc.returncode != 0:
            return RestartResult(
                node_name=node.name,
                success=False,
                message=f"SSH command failed with code {proc.returncode}",
                kill_output=stderr.decode("utf-8", errors="replace"),
            )

        return RestartResult(
            node_name=node.name,
            success=True,
            message="P2P orchestrator restarted successfully",
            start_output=stdout.decode("utf-8", errors="replace"),
        )

    except Exception as e:
        return RestartResult(
            node_name=node.name,
            success=False,
            message=f"Error: {e}",
        )


async def verify_swim_enabled(nodes: list[NodeConfig], timeout: int = 30) -> list[dict]:
    """Verify SWIM is enabled on nodes by checking their /status endpoint."""

    results = []

    async def check_node(node: NodeConfig) -> Optional[dict]:
        """Check a single node's status."""
        # Use Tailscale IP if available, otherwise SSH host
        # For verification, we need HTTP access to port 8770

        # Build SSH command to curl the local status endpoint
        ssh_key = os.path.expanduser(node.ssh_key)
        ssh_base = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-i", ssh_key,
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}",
        ]

        # Curl the local status endpoint
        curl_cmd = "curl -s http://localhost:8770/status 2>/dev/null || echo '{\"error\": \"curl failed\"}'"

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_base,
                curl_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return None

            if proc.returncode != 0:
                return None

            import json
            try:
                status = json.loads(stdout.decode("utf-8", errors="replace"))
                return {
                    "node": node.name,
                    "status": status,
                    "swim_enabled": status.get("swim_raft", {}).get("swim", {}).get("enabled", False),
                    "raft_enabled": status.get("swim_raft", {}).get("raft", {}).get("enabled", False),
                }
            except json.JSONDecodeError:
                return None

        except Exception:
            return None

    # Check nodes in parallel (but limit concurrency)
    semaphore = asyncio.Semaphore(10)

    async def check_with_semaphore(node: NodeConfig) -> Optional[dict]:
        async with semaphore:
            return await check_node(node)

    tasks = [check_with_semaphore(node) for node in nodes]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser(
        description="Force restart P2P orchestrators on all cluster nodes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum concurrent SSH connections (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per node in seconds (default: 60)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SWIM verification after restart",
    )

    args = parser.parse_args()

    print("Loading cluster configuration...")
    nodes = load_cluster_nodes()

    print(f"\nFound {len(nodes)} nodes with status='ready' and p2p_enabled=true:")
    for node in nodes:
        print(f"  - {node.name} ({node.ssh_user}@{node.ssh_host}:{node.ssh_port})")

    print(f"\nRestarting P2P orchestrators (max {args.max_parallel} concurrent)...")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def restart_with_semaphore(node: NodeConfig) -> RestartResult:
        async with semaphore:
            return await restart_p2p_on_node(node, args.dry_run, args.timeout)

    # Run all restarts in parallel
    tasks = [restart_with_semaphore(node) for node in nodes]
    results = await asyncio.gather(*tasks)

    # Print results
    print("\n" + "=" * 60)
    print("RESTART RESULTS")
    print("=" * 60)

    success_count = 0
    failed_count = 0

    for result in results:
        if result.success:
            success_count += 1
            print(f"[OK] {result.node_name}: {result.message}")
        else:
            failed_count += 1
            print(f"[FAIL] {result.node_name}: {result.message}")
            if result.kill_output:
                print(f"       Error: {result.kill_output[:200]}")

    print(f"\nSummary: {success_count} succeeded, {failed_count} failed")

    # Verify SWIM is enabled on at least 3 nodes
    if not args.dry_run and not args.skip_verify:
        print("\n" + "=" * 60)
        print("VERIFYING SWIM/RAFT STATUS")
        print("=" * 60)

        # Wait a bit for P2P to start up
        print("\nWaiting 10 seconds for P2P orchestrators to start...")
        await asyncio.sleep(10)

        print("Checking /status endpoints...")
        verification_results = await verify_swim_enabled(nodes, timeout=30)

        swim_enabled_count = sum(1 for r in verification_results if r.get("swim_enabled"))
        raft_enabled_count = sum(1 for r in verification_results if r.get("raft_enabled"))

        print(f"\nVerification results ({len(verification_results)} nodes responded):")
        for r in verification_results:
            swim = "SWIM=ON" if r.get("swim_enabled") else "SWIM=off"
            raft = "RAFT=ON" if r.get("raft_enabled") else "RAFT=off"
            print(f"  - {r['node']}: {swim}, {raft}")

        print(f"\nSWIM enabled: {swim_enabled_count} nodes")
        print(f"RAFT enabled: {raft_enabled_count} nodes")

        if swim_enabled_count >= 3:
            print("\n[PASS] At least 3 nodes have SWIM enabled")
        else:
            print(f"\n[WARN] Only {swim_enabled_count} nodes have SWIM enabled (target: 3+)")
            print("       Check .env.local on remote nodes or SWIM dependencies")


if __name__ == "__main__":
    asyncio.run(main())
