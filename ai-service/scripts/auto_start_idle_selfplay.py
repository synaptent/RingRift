#!/usr/bin/env python3
"""Auto-start selfplay on idle GPU nodes.

This script queries the P2P cluster for idle GPU nodes and starts selfplay
on them via SSH. It handles both directly reachable nodes and NAT-blocked
nodes (via their SSH gateway).

Usage:
    python scripts/auto_start_idle_selfplay.py --once    # Single run
    python scripts/auto_start_idle_selfplay.py --daemon  # Continuous mode

December 27, 2025
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    port: int
    gpu_name: str
    gpu_percent: float
    selfplay_jobs: int
    nat_blocked: bool
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: str = "root"
    ssh_key: Optional[str] = None
    ringrift_path: str = "~/ringrift/ai-service"


def load_ssh_config() -> dict:
    """Load SSH configuration from distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get("hosts", {})


def get_cluster_status(leader_url: str = "http://localhost:8770") -> list[NodeInfo]:
    """Query the P2P cluster for node status."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{leader_url}/status", timeout=10) as resp:
            data = json.load(resp)
    except Exception as e:
        logger.error(f"Failed to query cluster status: {e}")
        return []

    peers = data.get("peers", {})
    ssh_config = load_ssh_config()
    nodes = []

    for node_id, peer in peers.items():
        if not peer.get("is_cuda_gpu_node"):
            continue

        # Get SSH config if available
        node_ssh = ssh_config.get(node_id, {})

        node = NodeInfo(
            node_id=node_id,
            host=peer.get("host", ""),
            port=peer.get("port", 8770),
            gpu_name=peer.get("gpu_name", "unknown"),
            gpu_percent=peer.get("gpu_percent", 0),
            selfplay_jobs=peer.get("selfplay_jobs", 0),
            nat_blocked=peer.get("nat_blocked", False),
            ssh_host=node_ssh.get("ssh_host"),
            ssh_port=node_ssh.get("ssh_port"),
            ssh_user=node_ssh.get("ssh_user", "root"),
            ssh_key=node_ssh.get("ssh_key", "~/.ssh/id_cluster"),
            ringrift_path=node_ssh.get("ringrift_path", "~/ringrift/ai-service"),
        )
        nodes.append(node)

    return nodes


def get_idle_nodes(nodes: list[NodeInfo], gpu_threshold: float = 10.0) -> list[NodeInfo]:
    """Filter to nodes that are idle (low GPU util, no selfplay jobs)."""
    return [
        n for n in nodes
        if n.gpu_percent < gpu_threshold and n.selfplay_jobs == 0
    ]


def start_selfplay_via_http(node: NodeInfo, board_type: str, num_players: int, num_games: int) -> bool:
    """Try to start selfplay via direct HTTP request."""
    import urllib.request

    url = f"http://{node.host}:{node.port}/selfplay/start"
    data = json.dumps({
        "board_type": board_type,
        "num_players": num_players,
        "num_games": num_games,
        "engine_mode": "gumbel-mcts",
    }).encode()

    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.load(resp)
            if result.get("success"):
                logger.info(f"Started selfplay on {node.node_id} via HTTP: {board_type}_{num_players}p")
                return True
    except Exception as e:
        logger.debug(f"HTTP request to {node.node_id} failed: {e}")

    return False


def start_selfplay_via_ssh(node: NodeInfo, board_type: str, num_players: int, num_games: int) -> bool:
    """Start selfplay via SSH."""
    if not node.ssh_host or not node.ssh_port:
        logger.warning(f"No SSH config for {node.node_id}")
        return False

    ssh_key = Path(node.ssh_key).expanduser() if node.ssh_key else None

    cmd = [
        "ssh",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-p", str(node.ssh_port),
    ]

    if ssh_key and ssh_key.exists():
        cmd.extend(["-i", str(ssh_key)])

    cmd.append(f"{node.ssh_user}@{node.ssh_host}")

    # Build the remote command
    remote_cmd = f"""
cd {node.ringrift_path} && source venv/bin/activate 2>/dev/null || true
nohup python scripts/selfplay.py --board {board_type} --num-players {num_players} --num-games {num_games} --engine gumbel > /tmp/selfplay_{board_type}_{num_players}p.log 2>&1 &
echo "Started selfplay: {board_type}_{num_players}p"
"""
    cmd.append(remote_cmd)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Started selfplay on {node.node_id} via SSH: {board_type}_{num_players}p")
            return True
        else:
            logger.warning(f"SSH to {node.node_id} failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"SSH to {node.node_id} timed out")
    except Exception as e:
        logger.error(f"SSH to {node.node_id} error: {e}")

    return False


def start_selfplay_on_node(node: NodeInfo, configs: list[tuple[str, int]] = None) -> int:
    """Start selfplay on a node using the best available method.

    Returns the number of jobs successfully started.
    """
    if configs is None:
        # Default configuration mix for diverse training data
        configs = [
            ("hex8", 2),
            ("hex8", 3),
            ("square8", 2),
            ("square8", 4),
        ]

    # Scale games based on GPU power
    gpu_name = node.gpu_name.upper()
    if "H100" in gpu_name or "GH200" in gpu_name or "A100" in gpu_name:
        num_games = 1000
    elif "4090" in gpu_name or "5090" in gpu_name or "L40" in gpu_name:
        num_games = 750
    elif "3090" in gpu_name or "4080" in gpu_name or "5080" in gpu_name:
        num_games = 500
    else:
        num_games = 300

    started = 0

    for board_type, num_players in configs:
        # Try HTTP first (faster if not NAT-blocked)
        if not node.nat_blocked:
            if start_selfplay_via_http(node, board_type, num_players, num_games):
                started += 1
                continue

        # Fall back to SSH
        if start_selfplay_via_ssh(node, board_type, num_players, num_games):
            started += 1

    return started


def run_once(leader_url: str = "http://localhost:8770") -> int:
    """Run a single check and start selfplay on idle nodes.

    Returns the number of jobs started.
    """
    logger.info("Checking for idle GPU nodes...")

    nodes = get_cluster_status(leader_url)
    if not nodes:
        logger.warning("No nodes found in cluster")
        return 0

    idle_nodes = get_idle_nodes(nodes)
    if not idle_nodes:
        logger.info("No idle GPU nodes found")
        return 0

    logger.info(f"Found {len(idle_nodes)} idle GPU nodes")

    total_started = 0
    for node in idle_nodes:
        logger.info(f"Starting selfplay on {node.node_id} ({node.gpu_name})")
        started = start_selfplay_on_node(node)
        total_started += started
        if started > 0:
            logger.info(f"  Started {started} jobs")
        else:
            logger.warning(f"  Failed to start any jobs")

    return total_started


def run_daemon(leader_url: str = "http://localhost:8770", interval: int = 60):
    """Run continuously, checking for idle nodes periodically."""
    logger.info(f"Starting daemon mode (interval={interval}s)")

    while True:
        try:
            started = run_once(leader_url)
            logger.info(f"Cycle complete: {started} jobs started")
        except Exception as e:
            logger.error(f"Cycle failed: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Auto-start selfplay on idle GPU nodes")
    parser.add_argument("--leader-url", default="http://localhost:8770",
                       help="P2P leader URL for cluster status")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit")
    parser.add_argument("--daemon", action="store_true",
                       help="Run continuously")
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval in seconds (daemon mode)")

    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.leader_url, args.interval)
    else:
        # Default to single run
        started = run_once(args.leader_url)
        sys.exit(0 if started >= 0 else 1)


if __name__ == "__main__":
    main()
