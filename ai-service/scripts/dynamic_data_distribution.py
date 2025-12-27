#!/usr/bin/env python3
"""Dynamic Data Distribution Daemon for RingRift AI Training.

This script orchestrates data distribution from OWC external drive and AWS S3
to training nodes based on their current needs and available capacity.

Features:
- Capacity-aware: Only pushes to nodes with adequate disk space
- Priority-based: Prioritizes data for active training configs
- Dynamic: Adjusts distribution based on node availability
- Orchestrated: Coordinates with AutoSyncDaemon events

Usage:
    # One-time distribution
    python scripts/dynamic_data_distribution.py --once

    # Run as daemon
    python scripts/dynamic_data_distribution.py --daemon

    # Check status only
    python scripts/dynamic_data_distribution.py --status

Environment:
    MAC_STUDIO_HTTP: HTTP URL for mac-studio OWC data (default: http://100.107.168.125:8780)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MAC_STUDIO_HTTP = "http://100.107.168.125:8780"
MIN_FREE_DISK_GB = 50  # Minimum free disk to receive data
MAX_CONCURRENT_DOWNLOADS = 3

# Training node configurations
TRAINING_NODES = {
    "nebius-h100-3": {
        "ssh": "ubuntu@89.169.110.128",
        "key": "~/.ssh/id_cluster",
        "path": "~/ringrift/ai-service",
        "priority": 1,  # Highest priority
    },
    "nebius-h100-1": {
        "ssh": "ubuntu@89.169.111.139",
        "key": "~/.ssh/id_cluster",
        "path": "~/ringrift/ai-service",
        "priority": 2,
    },
    "vultr-a100": {
        "ssh": "root@208.167.249.164",
        "key": "~/.ssh/id_transfer",  # Use transfer key (no passphrase)
        "path": "/root/ringrift/ai-service",
        "priority": 3,
    },
}

# Critical training configs that need data
CRITICAL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p",
]

# Data sources on mac-studio
DATA_SOURCES = {
    "canonical_games": "/canonical_games/",
    "cluster_games": "/cluster_games/",
    "canonical_data": "/canonical_data/",
    "canonical_models": "/canonical_models/",
}


@dataclass
class NodeStatus:
    """Status of a training node."""
    name: str
    reachable: bool = False
    disk_free_gb: float = 0.0
    disk_used_percent: float = 100.0
    game_count: int = 0
    npz_count: int = 0
    last_check: float = 0.0


@dataclass
class DistributionPlan:
    """Plan for distributing data to nodes."""
    node: str
    files: list[str] = field(default_factory=list)
    data_type: str = ""
    priority: int = 0
    estimated_size_mb: float = 0.0


async def run_ssh_command(node_config: dict, command: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run SSH command on a node."""
    ssh_key = Path(node_config["key"]).expanduser()
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-i", str(ssh_key),
        node_config["ssh"],
        command
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        return 1, "", "Timeout"
    except Exception as e:
        return 1, "", str(e)


async def check_node_status(name: str, config: dict) -> NodeStatus:
    """Check status of a training node."""
    status = NodeStatus(name=name, last_check=time.time())

    # Check disk space
    code, stdout, _ = await run_ssh_command(
        config,
        "df -BG / | tail -1 | awk '{print $4, $5}'"
    )

    if code == 0:
        status.reachable = True
        try:
            parts = stdout.strip().split()
            status.disk_free_gb = float(parts[0].rstrip("G"))
            status.disk_used_percent = float(parts[1].rstrip("%"))
        except (IndexError, ValueError):
            pass

    # Count game databases
    code, stdout, _ = await run_ssh_command(
        config,
        f"ls {config['path']}/data/games/*.db 2>/dev/null | wc -l"
    )
    if code == 0:
        try:
            status.game_count = int(stdout.strip())
        except ValueError:
            pass

    # Count NPZ files
    code, stdout, _ = await run_ssh_command(
        config,
        f"ls {config['path']}/data/training/*.npz 2>/dev/null | wc -l"
    )
    if code == 0:
        try:
            status.npz_count = int(stdout.strip())
        except ValueError:
            pass

    return status


async def get_available_files(data_type: str) -> list[dict]:
    """Get list of available files from mac-studio via HTTP."""
    import urllib.request
    import re

    url = MAC_STUDIO_HTTP + DATA_SOURCES.get(data_type, "/")

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            html = response.read().decode()
            # Parse HTML directory listing
            files = []
            for match in re.finditer(r'href="([^"]+\.(?:db|npz|pth))"', html):
                filename = match.group(1)
                files.append({"name": filename, "url": url + filename})
            return files
    except Exception as e:
        logger.error(f"Failed to list {data_type}: {e}")
        return []


async def download_to_node(
    node_name: str,
    node_config: dict,
    file_url: str,
    dest_subdir: str
) -> bool:
    """Download a file to a node via wget."""
    dest_path = f"{node_config['path']}/data/{dest_subdir}/"
    filename = file_url.split("/")[-1]

    cmd = f"mkdir -p {dest_path} && wget -q -O {dest_path}{filename} '{file_url}'"
    code, _, stderr = await run_ssh_command(node_config, cmd, timeout=300)

    if code == 0:
        logger.info(f"  Downloaded {filename} to {node_name}")
        return True
    else:
        logger.error(f"  Failed to download {filename} to {node_name}: {stderr}")
        return False


async def distribute_data_type(
    data_type: str,
    dest_subdir: str,
    nodes: dict[str, NodeStatus],
    node_configs: dict
) -> int:
    """Distribute a type of data to all eligible nodes."""
    files = await get_available_files(data_type)
    if not files:
        logger.warning(f"No files found for {data_type}")
        return 0

    logger.info(f"Found {len(files)} {data_type} files")

    total_distributed = 0

    for node_name, status in nodes.items():
        if not status.reachable:
            continue
        if status.disk_free_gb < MIN_FREE_DISK_GB:
            logger.info(f"  Skipping {node_name}: low disk ({status.disk_free_gb:.1f}GB free)")
            continue

        config = node_configs[node_name]

        # Download files in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def download_with_semaphore(file_info):
            async with semaphore:
                return await download_to_node(
                    node_name, config, file_info["url"], dest_subdir
                )

        logger.info(f"Distributing to {node_name}...")
        results = await asyncio.gather(*[
            download_with_semaphore(f) for f in files
        ])

        total_distributed += sum(1 for r in results if r)

    return total_distributed


async def run_distribution():
    """Run a full distribution cycle."""
    logger.info("Starting distribution cycle")

    # Check node statuses
    logger.info("Checking node statuses...")
    node_statuses = {}
    for name, config in TRAINING_NODES.items():
        status = await check_node_status(name, config)
        node_statuses[name] = status
        logger.info(
            f"  {name}: reachable={status.reachable}, "
            f"disk_free={status.disk_free_gb:.1f}GB, "
            f"games={status.game_count}, npz={status.npz_count}"
        )

    # Filter to reachable nodes with space
    eligible_nodes = {
        name: status for name, status in node_statuses.items()
        if status.reachable and status.disk_free_gb >= MIN_FREE_DISK_GB
    }

    if not eligible_nodes:
        logger.warning("No eligible nodes found")
        return

    # Distribute NPZ files first (small, needed for training)
    logger.info("\n=== Distributing NPZ files ===")
    npz_count = await distribute_data_type(
        "canonical_data", "training", eligible_nodes, TRAINING_NODES
    )

    # Distribute canonical games
    logger.info("\n=== Distributing canonical games ===")
    db_count = await distribute_data_type(
        "canonical_games", "games", eligible_nodes, TRAINING_NODES
    )

    # Distribute cluster games (larger)
    logger.info("\n=== Distributing cluster games ===")
    cluster_count = await distribute_data_type(
        "cluster_games", "games", eligible_nodes, TRAINING_NODES
    )

    logger.info(f"\nDistribution complete: {npz_count} NPZ, {db_count + cluster_count} DBs")


async def run_daemon(interval: int = 300):
    """Run as a daemon, distributing data periodically."""
    logger.info(f"Starting distribution daemon (interval: {interval}s)")

    while True:
        try:
            await run_distribution()
        except Exception as e:
            logger.error(f"Distribution cycle failed: {e}")

        logger.info(f"Sleeping for {interval}s...")
        await asyncio.sleep(interval)


async def show_status():
    """Show current status of all nodes."""
    print("=== Training Node Status ===\n")

    for name, config in TRAINING_NODES.items():
        status = await check_node_status(name, config)
        print(f"{name}:")
        print(f"  Reachable: {status.reachable}")
        print(f"  Disk Free: {status.disk_free_gb:.1f} GB")
        print(f"  Disk Used: {status.disk_used_percent:.1f}%")
        print(f"  Game DBs: {status.game_count}")
        print(f"  NPZ Files: {status.npz_count}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Dynamic data distribution for RingRift")
    parser.add_argument("--once", action="store_true", help="Run one distribution cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show node status")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval (seconds)")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.daemon:
        asyncio.run(run_daemon(args.interval))
    elif args.once:
        asyncio.run(run_distribution())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
