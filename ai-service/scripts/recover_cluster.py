#!/usr/bin/env python3
"""Recover P2P cluster - start P2P on all disconnected nodes.

This script is designed to be run manually or via cron when the cluster
needs recovery. It's independent of the P2P leader election and can be
run from any machine with SSH access to cluster nodes.

Usage:
    python scripts/recover_cluster.py              # Recover all disconnected nodes
    python scripts/recover_cluster.py --dry-run   # Show what would be done
    python scripts/recover_cluster.py --parallel 20  # Max 20 nodes in parallel

Jan 2026: Created to provide reliable cluster recovery independent of P2P state.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def check_p2p_health(host: str, port: int = 8770, timeout: float = 5.0) -> bool:
    """Check if a node's P2P is responding."""
    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(f"http://{host}:{port}/health") as resp:
                return resp.status == 200
    except Exception:
        return False


async def start_p2p_on_node(
    node_id: str,
    node_config: dict,
    ssh_key: str,
    dry_run: bool = False,
) -> tuple[str, bool, str]:
    """Start P2P on a remote node via SSH.

    Returns (node_id, success, message)
    """
    # Determine connection details
    ssh_host = node_config.get("ssh_host") or node_config.get("tailscale_ip")
    ssh_port = node_config.get("ssh_port", 22)
    ssh_user = node_config.get("ssh_user", "ubuntu")
    ringrift_path = node_config.get("ringrift_path", "~/ringrift/ai-service")

    # Handle host:port format
    if ssh_host and ":" in str(ssh_host):
        host_part, port_part = ssh_host.rsplit(":", 1)
        ssh_host = host_part
        try:
            ssh_port = int(port_part)
        except ValueError:
            pass

    if not ssh_host:
        return node_id, False, "No SSH host configured"

    # Check user override for RunPod/Vast.ai
    provider = node_config.get("provider", "")
    if provider in ("runpod", "vast"):
        ssh_user = "root"
        if not ringrift_path.startswith("/workspace"):
            ringrift_path = "/workspace/ringrift/ai-service"

    if dry_run:
        return node_id, True, f"DRY RUN: Would SSH to {ssh_user}@{ssh_host}:{ssh_port}"

    # Build SSH command
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
        "-i", ssh_key,
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
        f"cd {ringrift_path} && pkill -9 -f p2p_orchestrator 2>/dev/null; "
        f"sleep 2; nohup python scripts/p2p_orchestrator.py > /tmp/p2p.log 2>&1 &; "
        f"sleep 5; pgrep -f p2p_orchestrator && echo P2P_STARTED || echo P2P_FAILED",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode().strip()

        if "P2P_STARTED" in output:
            return node_id, True, "P2P started successfully"
        else:
            return node_id, False, f"P2P failed: {output or stderr.decode()[:200]}"
    except asyncio.TimeoutError:
        return node_id, False, "SSH command timed out"
    except Exception as e:
        return node_id, False, f"SSH error: {e}"


async def recover_cluster(
    config_path: str,
    max_parallel: int = 10,
    dry_run: bool = False,
    ssh_key: str = "~/.ssh/id_cluster",
) -> dict:
    """Recover all disconnected nodes in the cluster.

    Returns statistics dict.
    """
    ssh_key = os.path.expanduser(ssh_key)
    if not os.path.exists(ssh_key):
        logger.error(f"SSH key not found: {ssh_key}")
        return {"error": "SSH key not found"}

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = config.get("hosts", {})

    # Find nodes that need recovery
    nodes_to_recover = []

    for node_id, node_config in hosts.items():
        # Skip retired, proxy-only, or P2P-disabled nodes
        status = node_config.get("status", "active")
        p2p_enabled = node_config.get("p2p_enabled", True)

        if status in ("retired", "proxy_only") or not p2p_enabled:
            continue

        # Get IP for health check
        ip = node_config.get("tailscale_ip") or node_config.get("ssh_host")
        if ip and ":" in str(ip):
            ip = ip.split(":")[0]

        if not ip:
            continue

        nodes_to_recover.append((node_id, node_config, ip))

    logger.info(f"Checking {len(nodes_to_recover)} active nodes...")

    # Check which nodes are already healthy
    healthy = []
    unhealthy = []

    async def check_node(node_id: str, ip: str) -> tuple[str, bool]:
        is_healthy = await check_p2p_health(ip)
        return node_id, is_healthy

    check_tasks = [check_node(n[0], n[2]) for n in nodes_to_recover]
    check_results = await asyncio.gather(*check_tasks)

    node_lookup = {n[0]: n for n in nodes_to_recover}

    for node_id, is_healthy in check_results:
        if is_healthy:
            healthy.append(node_id)
        else:
            unhealthy.append(node_id)

    logger.info(f"Healthy: {len(healthy)}, Unhealthy: {len(unhealthy)}")

    if not unhealthy:
        logger.info("All nodes are healthy!")
        return {"healthy": len(healthy), "recovered": 0, "failed": 0}

    logger.info(f"Unhealthy nodes: {unhealthy}")

    if dry_run:
        logger.info("DRY RUN - not making changes")

    # Recover unhealthy nodes in parallel batches
    recovered = 0
    failed = 0

    semaphore = asyncio.Semaphore(max_parallel)

    async def recover_with_limit(node_id: str) -> tuple[str, bool, str]:
        async with semaphore:
            node_info = node_lookup[node_id][1]
            return await start_p2p_on_node(node_id, node_info, ssh_key, dry_run)

    tasks = [recover_with_limit(n) for n in unhealthy]
    results = await asyncio.gather(*tasks)

    for node_id, success, message in results:
        if success:
            recovered += 1
            logger.info(f"[{node_id}] {message}")
        else:
            failed += 1
            logger.warning(f"[{node_id}] {message}")

    logger.info(f"Recovery complete: {recovered} recovered, {failed} failed")

    return {
        "healthy": len(healthy),
        "recovered": recovered,
        "failed": failed,
        "unhealthy_nodes": unhealthy,
    }


def main():
    parser = argparse.ArgumentParser(description="Recover P2P cluster")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--parallel", type=int, default=10, help="Max parallel recoveries")
    parser.add_argument("--ssh-key", default="~/.ssh/id_cluster", help="SSH key path")
    parser.add_argument(
        "--config",
        default="config/distributed_hosts.yaml",
        help="Cluster config path",
    )
    args = parser.parse_args()

    # Find config path
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to ai-service
        ai_service_dir = Path(__file__).parent.parent
        config_path = ai_service_dir / config_path

    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    start = time.time()
    result = asyncio.run(recover_cluster(
        str(config_path),
        max_parallel=args.parallel,
        dry_run=args.dry_run,
        ssh_key=args.ssh_key,
    ))
    elapsed = time.time() - start

    logger.info(f"Completed in {elapsed:.1f}s")
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
