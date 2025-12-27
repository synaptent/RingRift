#!/usr/bin/env python3
"""Orchestrated Data Sync for Training Nodes.

This script implements intelligent data distribution from OWC to training nodes:
- Config-aware: Only syncs data relevant to pending/active training
- Capacity-aware: Respects disk space limits on target nodes
- Bandwidth-aware: Uses rsync with bandwidth limits
- Priority-based: Most urgent configs first

The key insight: Training uses NPZ files, not raw DBs. So:
1. Keep NPZ distribution via HTTP (fast, already working)
2. For fresh exports, trigger export on OWC then distribute NPZ
3. Only sync DBs if node needs to run local exports

Usage:
    # Sync NPZ files (most common)
    python scripts/orchestrated_data_sync.py --npz-only

    # Sync specific config DBs
    python scripts/orchestrated_data_sync.py --config hex8_4p

    # Trigger export on OWC then sync NPZ
    python scripts/orchestrated_data_sync.py --export-and-sync hex8_4p

    # Full orchestration daemon
    python scripts/orchestrated_data_sync.py --daemon

December 2025: Created for intelligent data orchestration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configuration
MAC_STUDIO_HOST = "mac-studio"
OWC_DATA_PATH = "/Volumes/RingRift-Data"
RSYNC_BANDWIDTH_LIMIT = "50m"  # 50 MB/s
MIN_FREE_GB = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("orchestrated_sync")


@dataclass
class TrainingNode:
    """Training node configuration."""
    name: str
    ssh_target: str
    data_path: str
    priority: int = 10


TRAINING_NODES = [
    TrainingNode("nebius-h100-3", "ubuntu@89.169.110.128", "~/ringrift/ai-service", 1),
    TrainingNode("nebius-h100-1", "ubuntu@89.169.111.139", "~/ringrift/ai-service", 2),
    TrainingNode("vultr-a100", "root@208.167.249.164", "/root/ringrift/ai-service", 3),
]


@dataclass
class ConfigData:
    """Data availability for a config."""
    board_type: str
    num_players: int
    owc_games: int = 0
    owc_npz_path: str = ""
    owc_npz_size_mb: float = 0
    node_npz_exists: dict = field(default_factory=dict)
    node_games: dict = field(default_factory=dict)


def run_ssh(host: str, cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run SSH command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "Timeout"
    except Exception as e:
        return 1, str(e)


def run_rsync(src: str, dst: str, bwlimit: str = RSYNC_BANDWIDTH_LIMIT) -> bool:
    """Run rsync with bandwidth limit."""
    cmd = [
        "rsync", "-avz", "--progress",
        f"--bwlimit={bwlimit}",
        src, dst
    ]
    logger.info(f"rsync: {src} -> {dst}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


async def check_node_space(node: TrainingNode) -> float:
    """Get free disk space on node in GB."""
    code, output = run_ssh(node.ssh_target, "df -BG ~ | tail -1 | awk '{print $4}'")
    if code == 0:
        try:
            return float(output.rstrip("G"))
        except (ValueError, AttributeError) as e:
            # Dec 2025: Narrow exception - float() may fail on malformed output
            logger.debug(f"[check_node_space] Could not parse disk space '{output}': {e}")
    return 0


async def get_node_npz_files(node: TrainingNode) -> list[str]:
    """Get list of NPZ files on node."""
    code, output = run_ssh(
        node.ssh_target,
        f"ls {node.data_path}/data/training/*.npz 2>/dev/null | xargs -I{{}} basename {{}}"
    )
    if code == 0 and output:
        return output.split("\n")
    return []


async def trigger_owc_export(config: str) -> bool:
    """Trigger NPZ export on mac-studio for a config."""
    board_type, num_players = config.rsplit("_", 1)
    num_players = int(num_players.rstrip("p"))

    cmd = f"""
cd ~/ringrift/ai-service &&
PYTHONPATH=. ./venv/bin/python3 scripts/scheduled_npz_export.py --once --config {config}
"""
    logger.info(f"Triggering export for {config} on mac-studio...")
    code, output = run_ssh(MAC_STUDIO_HOST, cmd, timeout=600)

    if code == 0:
        logger.info(f"Export completed for {config}")
        return True
    else:
        logger.error(f"Export failed for {config}: {output}")
        return False


async def sync_npz_to_node(node: TrainingNode, config: str) -> bool:
    """Sync NPZ file for config to node."""
    npz_name = f"{config}.npz"
    src = f"{MAC_STUDIO_HOST}:{OWC_DATA_PATH}/canonical_data/{npz_name}"
    dst = f"{node.ssh_target}:{node.data_path}/data/training/"

    return run_rsync(src, dst)


async def get_pending_training_configs() -> list[str]:
    """Discover what configs have pending/active training."""
    configs_needed = []

    for node in TRAINING_NODES:
        # Check for auto_train scripts
        code, output = run_ssh(
            node.ssh_target,
            "pgrep -fa 'auto_train' 2>/dev/null | head -5"
        )
        if code == 0 and output:
            # Parse config from script name, e.g., auto_train_hex8_4p.sh
            for line in output.split("\n"):
                for config in ["hex8_2p", "hex8_3p", "hex8_4p", "square8_2p", "square8_3p", "square8_4p"]:
                    if config in line:
                        if config not in configs_needed:
                            configs_needed.append(config)

    return configs_needed


async def orchestrate_sync(configs: list[str] = None, npz_only: bool = True):
    """Main orchestration loop."""

    if not configs:
        # Auto-discover pending training configs
        configs = await get_pending_training_configs()
        if not configs:
            # Default to common configs
            configs = ["hex8_2p", "hex8_3p", "hex8_4p"]

    logger.info(f"Syncing configs: {configs}")

    for node in TRAINING_NODES:
        # Check disk space
        free_gb = await check_node_space(node)
        if free_gb < MIN_FREE_GB:
            logger.warning(f"{node.name}: Only {free_gb}GB free, skipping")
            continue

        logger.info(f"{node.name}: {free_gb}GB free, syncing...")

        # Get existing NPZ files
        existing = await get_node_npz_files(node)

        for config in configs:
            npz_name = f"{config}.npz"
            if npz_name not in existing:
                logger.info(f"  {node.name}: Missing {npz_name}, syncing...")
                await sync_npz_to_node(node, config)
            else:
                logger.info(f"  {node.name}: Has {npz_name}")


async def export_and_sync(config: str):
    """Trigger export then sync to all nodes."""
    # First export on OWC
    if await trigger_owc_export(config):
        # Then sync to all nodes
        for node in TRAINING_NODES:
            await sync_npz_to_node(node, config)


def main():
    parser = argparse.ArgumentParser(description="Orchestrated data sync for training")
    parser.add_argument("--npz-only", action="store_true", help="Only sync NPZ files")
    parser.add_argument("--config", type=str, help="Specific config to sync")
    parser.add_argument("--export-and-sync", type=str, metavar="CONFIG",
                        help="Export config on OWC then sync")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=1800, help="Daemon interval in seconds")
    args = parser.parse_args()

    if args.export_and_sync:
        asyncio.run(export_and_sync(args.export_and_sync))
    elif args.config:
        asyncio.run(orchestrate_sync([args.config]))
    elif args.daemon:
        logger.info(f"Starting orchestrated sync daemon (interval: {args.interval}s)")
        while True:
            try:
                asyncio.run(orchestrate_sync())
            except Exception as e:
                logger.error(f"Sync cycle failed: {e}")
            import time
            time.sleep(args.interval)
    else:
        asyncio.run(orchestrate_sync())


if __name__ == "__main__":
    main()
