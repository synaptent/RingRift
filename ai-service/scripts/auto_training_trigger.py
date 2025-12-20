#!/usr/bin/env python3
"""Auto-trigger training jobs to maximize GPU utilization.

This script finds idle GPU nodes and triggers training jobs to ensure
the cluster maintains high GPU utilization. It can be run via cron:

    */5 * * * * python3 /path/to/auto_training_trigger.py

Or as a daemon:
    python3 auto_training_trigger.py --daemon --interval 300
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

P2P_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))
TARGET_GPU_MIN = 60.0  # Target minimum GPU utilization
MAX_TRAINING_JOBS_PER_RUN = 5  # Cap training jobs per run

# Board configurations to cycle through
TRAINING_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
    ("hex8", 2),
    ("hex8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
]


def http_get(url: str, timeout: int = 15) -> dict | None:
    """Make HTTP GET request and return JSON."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return None


def http_post(url: str, data: dict, timeout: int = 15) -> dict | None:
    """Make HTTP POST request and return JSON."""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logger.debug(f"HTTP POST failed for {url}: {e}")
        return None


def get_local_health() -> dict | None:
    """Get local P2P node health."""
    return http_get(f"http://localhost:{P2P_PORT}/health")


def get_cluster_status() -> dict | None:
    """Get full cluster status from local P2P node."""
    return http_get(f"http://localhost:{P2P_PORT}/status")


def find_idle_gpu_nodes(status: dict) -> list[tuple[str, float, str]]:
    """Find GPU nodes with low utilization.

    Returns: List of (node_id, gpu_percent, gpu_name) tuples
    """
    idle_nodes = []

    # Check self
    self_info = status.get("self", {})
    if self_info.get("has_gpu") and float(self_info.get("gpu_percent", 0) or 0) < TARGET_GPU_MIN:
        training_jobs = int(self_info.get("training_jobs", 0) or 0)
        if training_jobs == 0:  # Only if not already training
            idle_nodes.append((
                self_info.get("node_id", "self"),
                float(self_info.get("gpu_percent", 0) or 0),
                self_info.get("gpu_name", "")
            ))

    # Check peers
    peers = status.get("peers", {})
    for node_id, peer in peers.items():
        if not peer.get("has_gpu"):
            continue

        # Skip retired nodes
        if peer.get("retired"):
            continue

        gpu_percent = float(peer.get("gpu_percent", 0) or 0)
        training_jobs = int(peer.get("training_jobs", 0) or 0)

        # Check if idle and not already training
        if gpu_percent < TARGET_GPU_MIN and training_jobs == 0:
            idle_nodes.append((
                node_id,
                gpu_percent,
                peer.get("gpu_name", "")
            ))

    # Sort by GPU utilization (lowest first)
    idle_nodes.sort(key=lambda x: x[1])
    return idle_nodes


def trigger_training(board_type: str, num_players: int) -> dict | None:
    """Trigger a training job via the local P2P API."""
    url = f"http://localhost:{P2P_PORT}/training/start"
    data = {
        "board_type": board_type,
        "num_players": num_players,
        "model_type": "nnue"
    }
    return http_post(url, data)


def main():
    parser = argparse.ArgumentParser(description="Auto-trigger training jobs")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in daemon mode (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually trigger training")
    args = parser.parse_args()

    def run_once():
        # Get local health first
        health = get_local_health()
        if not health:
            logger.error("Cannot connect to local P2P orchestrator")
            return 0

        role = health.get("role", "")
        node_id = health.get("node_id", "unknown")

        # Only the leader should trigger training cluster-wide
        if role != "leader":
            leader_id = health.get("leader_id")
            if leader_id:
                logger.info(f"Not leader (current leader: {leader_id}), skipping")
            else:
                logger.warning("No leader in cluster, skipping")
            return 0

        # Get full cluster status
        status = get_cluster_status()
        if not status:
            logger.error("Cannot get cluster status")
            return 0

        # Find idle GPU nodes
        idle_nodes = find_idle_gpu_nodes(status)
        if not idle_nodes:
            logger.info("No idle GPU nodes found")
            return 0

        logger.info(f"Found {len(idle_nodes)} idle GPU node(s)")
        for node_id, gpu_pct, gpu_name in idle_nodes[:5]:
            logger.info(f"  {node_id}: {gpu_pct:.1f}% GPU ({gpu_name})")

        # Trigger training jobs
        triggered = 0
        config_idx = random.randint(0, len(TRAINING_CONFIGS) - 1)

        for _ in range(min(len(idle_nodes), MAX_TRAINING_JOBS_PER_RUN)):
            board_type, num_players = TRAINING_CONFIGS[config_idx]
            config_idx = (config_idx + 1) % len(TRAINING_CONFIGS)

            if args.dry_run:
                logger.info(f"[DRY RUN] Would trigger training: {board_type} {num_players}p")
                triggered += 1
                continue

            result = trigger_training(board_type, num_players)
            if result and result.get("success"):
                worker = result.get("worker", "unknown")
                job_id = result.get("job_id", "unknown")
                logger.info(f"Triggered training: {board_type} {num_players}p -> {worker} (job: {job_id})")
                triggered += 1
            else:
                error = result.get("error", "unknown") if result else "no response"
                logger.warning(f"Failed to trigger training: {error}")
                break

        return triggered

    if args.daemon:
        logger.info(f"Starting auto-training daemon (interval: {args.interval}s)")
        while True:
            try:
                triggered = run_once()
                if triggered > 0:
                    logger.info(f"Triggered {triggered} training job(s)")
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
            time.sleep(args.interval)
    else:
        triggered = run_once()
        if triggered > 0:
            logger.info(f"Triggered {triggered} training job(s)")
        sys.exit(0 if triggered >= 0 else 1)


if __name__ == "__main__":
    main()
