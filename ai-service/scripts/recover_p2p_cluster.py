#!/usr/bin/env python3
"""P2P Cluster Recovery Script (December 31, 2025).

Scans all configured P2P-enabled nodes, checks if P2P is running,
and restarts it on nodes where it's not responding.

Usage:
    python scripts/recover_p2p_cluster.py
    python scripts/recover_p2p_cluster.py --dry-run
    python scripts/recover_p2p_cluster.py --max-parallel 5

Can be run periodically via cron or integrated into master_loop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Node configuration from distributed_hosts.yaml."""
    name: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    ssh_key: str
    tailscale_ip: str
    ringrift_path: str
    gpu: str
    status: str


def load_p2p_nodes() -> list[NodeInfo]:
    """Load all P2P-enabled nodes from config."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    nodes = []
    for name, info in config.get("hosts", {}).items():
        if not info.get("p2p_enabled", False):
            continue
        if info.get("status") == "offline":
            continue
        # Skip local nodes
        if name in ("local-mac", "mac-studio"):
            continue

        nodes.append(NodeInfo(
            name=name,
            ssh_host=info.get("ssh_host", ""),
            ssh_port=info.get("ssh_port", 22),
            ssh_user=info.get("ssh_user", "ubuntu"),
            ssh_key=info.get("ssh_key", "~/.ssh/id_cluster"),
            tailscale_ip=info.get("tailscale_ip", ""),
            ringrift_path=info.get("ringrift_path", "~/ringrift/ai-service"),
            gpu=info.get("gpu", "unknown"),
            status=info.get("status", "unknown"),
        ))

    return nodes


async def check_node_p2p_status(node: NodeInfo, timeout: int = 15) -> dict[str, Any]:
    """Check if P2P is running on a node via SSH."""
    result = {
        "name": node.name,
        "reachable": False,
        "p2p_running": False,
        "p2p_responding": False,
        "error": None,
    }

    # Determine best IP to use
    target = node.tailscale_ip or node.ssh_host
    if not target:
        result["error"] = "No IP/host available"
        return result

    # For vast nodes, use the ssh_host:ssh_port format
    if "vast.ai" in node.ssh_host:
        ssh_cmd = [
            "ssh", "-o", f"ConnectTimeout={timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-i", node.ssh_key,
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}",
        ]
    else:
        # Use tailscale IP if available, otherwise ssh_host
        ssh_cmd = [
            "ssh", "-o", f"ConnectTimeout={timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-i", node.ssh_key,
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{target}",
        ]

    proc = None
    try:
        # Check if P2P process is running
        check_cmd = ssh_cmd + ["pgrep -f 'python.*p2p_orchestrator' >/dev/null && echo 'RUNNING' || echo 'NOT_RUNNING'"]
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout + 5,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout + 5,
        )

        if proc.returncode == 0:
            result["reachable"] = True
            output = stdout.decode().strip()
            result["p2p_running"] = "RUNNING" in output

            # If process is running, check if P2P port is responding
            if result["p2p_running"]:
                health_cmd = ssh_cmd + ["curl -s -o /dev/null -w '%{http_code}' http://localhost:8770/health 2>/dev/null || echo '000'"]
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *health_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=timeout + 5,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout + 5,
                )
                http_code = stdout.decode().strip()
                result["p2p_responding"] = http_code == "200"
        else:
            result["error"] = f"SSH failed: {stderr.decode()[:100]}"

    except asyncio.TimeoutError:
        result["error"] = "Timeout"
        # Kill the subprocess if it's still running
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass  # Already terminated
    except Exception as e:
        result["error"] = str(e)[:100]
        # Clean up on any error
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass

    return result


async def restart_p2p_on_node(node: NodeInfo, dry_run: bool = False) -> bool:
    """Restart P2P on a node via SSH."""
    logger.info(f"Restarting P2P on {node.name}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would restart P2P on {node.name}")
        return True

    target = node.tailscale_ip or node.ssh_host
    if not target:
        logger.error(f"  No IP/host available for {node.name}")
        return False

    # Build SSH command
    if "vast.ai" in node.ssh_host:
        ssh_cmd = [
            "ssh", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-i", node.ssh_key,
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{node.ssh_host}",
        ]
    else:
        ssh_cmd = [
            "ssh", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-i", node.ssh_key,
            "-p", str(node.ssh_port),
            f"{node.ssh_user}@{target}",
        ]

    # Restart command - include --advertise-host for Tailscale nodes
    # December 2025: Critical fix - without --advertise-host, nodes advertise
    # unreachable IPs causing P2P mesh fragmentation
    advertise_host_arg = ""
    if node.tailscale_ip:
        advertise_host_arg = f"--advertise-host {node.tailscale_ip} "

    # January 21, 2026: Add --managed-by-master-loop flag to indicate automated recovery
    restart_cmd = (
        f"pkill -SIGTERM -f 'python.*p2p_orchestrator' 2>/dev/null; "
        f"sleep 2; "
        f"cd {node.ringrift_path} && "
        f"nohup python scripts/p2p_orchestrator.py --node-id {node.name} "
        f"--managed-by-master-loop "
        f"{advertise_host_arg}"
        f"> logs/p2p.log 2>&1 &"
    )

    proc = None
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *ssh_cmd, restart_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=30,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode == 0:
            logger.info(f"  Successfully restarted P2P on {node.name}")
            return True
        else:
            logger.error(f"  Failed to restart P2P on {node.name}")
            return False

    except asyncio.TimeoutError:
        logger.error(f"  Timeout restarting P2P on {node.name}")
        # Kill the subprocess if it's still running
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        return False
    except Exception as e:
        logger.error(f"  Error restarting P2P on {node.name}: {e}")
        # Clean up on any error
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
        return False


async def recover_cluster(
    max_parallel: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan all P2P-enabled nodes and restart P2P where needed."""
    nodes = load_p2p_nodes()
    logger.info(f"Loaded {len(nodes)} P2P-enabled nodes from config")

    # Check status of all nodes in parallel
    semaphore = asyncio.Semaphore(max_parallel)

    async def check_with_semaphore(node: NodeInfo) -> dict[str, Any]:
        async with semaphore:
            return await check_node_p2p_status(node)

    logger.info("Checking P2P status on all nodes...")
    statuses = await asyncio.gather(*[check_with_semaphore(n) for n in nodes])

    # Categorize results
    reachable = [s for s in statuses if s["reachable"]]
    unreachable = [s for s in statuses if not s["reachable"]]
    p2p_ok = [s for s in reachable if s["p2p_responding"]]
    p2p_not_running = [s for s in reachable if not s["p2p_running"]]
    p2p_not_responding = [s for s in reachable if s["p2p_running"] and not s["p2p_responding"]]

    logger.info(f"\n=== P2P Cluster Status ===")
    logger.info(f"  Reachable nodes: {len(reachable)}/{len(nodes)}")
    logger.info(f"  P2P healthy: {len(p2p_ok)}")
    logger.info(f"  P2P not running: {len(p2p_not_running)}")
    logger.info(f"  P2P not responding: {len(p2p_not_responding)}")
    logger.info(f"  Unreachable: {len(unreachable)}")

    # Log unreachable nodes
    if unreachable:
        logger.info("\n=== Unreachable Nodes ===")
        for s in unreachable:
            logger.info(f"  {s['name']}: {s['error']}")

    # Restart P2P on nodes where it's not running or not responding
    nodes_to_restart = []
    for status in p2p_not_running + p2p_not_responding:
        node = next(n for n in nodes if n.name == status["name"])
        nodes_to_restart.append(node)

    if nodes_to_restart:
        logger.info(f"\n=== Restarting P2P on {len(nodes_to_restart)} nodes ===")
        for node in nodes_to_restart:
            logger.info(f"  - {node.name}")

        # Restart in parallel with semaphore
        async def restart_with_semaphore(node: NodeInfo) -> bool:
            async with semaphore:
                return await restart_p2p_on_node(node, dry_run)

        restart_results = await asyncio.gather(
            *[restart_with_semaphore(n) for n in nodes_to_restart]
        )
        restarted = sum(1 for r in restart_results if r)
        logger.info(f"\nRestarted P2P on {restarted}/{len(nodes_to_restart)} nodes")
    else:
        logger.info("\nNo nodes need P2P restart")

    return {
        "total_nodes": len(nodes),
        "reachable": len(reachable),
        "unreachable": len(unreachable),
        "p2p_ok": len(p2p_ok),
        "p2p_not_running": len(p2p_not_running),
        "p2p_not_responding": len(p2p_not_responding),
        "restarted": len([r for r in locals().get("restart_results", []) if r]),
    }


def main():
    parser = argparse.ArgumentParser(description="Recover P2P cluster")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually restart")
    parser.add_argument("--max-parallel", type=int, default=10, help="Max parallel checks")
    args = parser.parse_args()

    result = asyncio.run(recover_cluster(
        max_parallel=args.max_parallel,
        dry_run=args.dry_run,
    ))

    print(f"\n=== Summary ===")
    print(f"Total configured: {result['total_nodes']}")
    print(f"Reachable: {result['reachable']}")
    print(f"P2P healthy: {result['p2p_ok']}")
    print(f"Restarted: {result.get('restarted', 0)}")


if __name__ == "__main__":
    main()
