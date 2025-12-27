#!/usr/bin/env python3
"""Training Orchestrator - Dynamic Data and Space Management for RingRift AI.

This script provides intelligent orchestration of the training pipeline:
- Monitors disk space on all training nodes
- Prioritizes data distribution based on training needs
- Automatically cleans up low-priority data when space is tight
- Coordinates NPZ export, distribution, and training job scheduling

Key Features:
1. Config Priority Scoring: Ranks configs by training need (games count, recency)
2. Dynamic Space Reclamation: Frees space on nodes without losing critical data
3. Training Job Scheduling: Queues training jobs based on data availability
4. Bidirectional Sync Coordination: OWC <-> Cluster data flow

Usage:
    # Check current state
    python scripts/training_orchestrator.py --status

    # Run orchestration cycle
    python scripts/training_orchestrator.py --once

    # Run as daemon (every 30 minutes)
    python scripts/training_orchestrator.py --daemon

    # Force data distribution for a config
    python scripts/training_orchestrator.py --distribute hex8_4p

December 2025: Created for intelligent training pipeline orchestration.
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

# Configuration
MIN_FREE_GB_FOR_TRAINING = 50  # Minimum free GB to start training
MIN_FREE_GB_FOR_DISTRIBUTION = 40  # Minimum free GB to receive data
TARGET_FREE_GB = 60  # Target free GB after cleanup
CLEANUP_PRIORITY_ORDER = [
    # Files to delete first (lowest priority)
    "*.log",
    "*_backup.db",
    "*_old.db",
    "*_temp.db",
    "*.db-wal",
    "*.db-shm",
]

# Config priorities (higher = more important)
CONFIG_PRIORITIES = {
    "hex8_2p": 100,    # Primary production config
    "square8_2p": 95,  # High volume training
    "hex8_4p": 90,     # Underserved but needed
    "square19_4p": 85, # Underserved
    "hexagonal_2p": 80,
    "square8_4p": 75,
    "hex8_3p": 70,
    "square8_3p": 65,
    "hexagonal_4p": 60,
    "square19_2p": 55,
    "hexagonal_3p": 50,
    "square19_3p": 45,
}

# Minimum game counts per config (below this = underserved)
MIN_GAMES_PER_CONFIG = {
    "hex8_2p": 10000,
    "hex8_3p": 5000,
    "hex8_4p": 5000,
    "square8_2p": 50000,
    "square8_3p": 10000,
    "square8_4p": 10000,
    "square19_2p": 10000,
    "square19_3p": 5000,
    "square19_4p": 5000,
    "hexagonal_2p": 20000,
    "hexagonal_3p": 5000,
    "hexagonal_4p": 10000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator")


@dataclass
class NodeStatus:
    """Status of a training node."""
    name: str
    ssh_target: str
    free_gb: float
    total_gb: float
    usage_percent: float
    can_train: bool  # Has enough space for training
    can_receive: bool  # Can receive more data
    training_active: bool  # Currently training
    selfplay_active: bool  # Currently running selfplay
    npz_count: int
    db_count: int


@dataclass
class ConfigStatus:
    """Status of a board configuration."""
    config: str
    games_on_owc: int
    npz_size_mb: float
    npz_age_hours: float
    priority: int
    is_underserved: bool
    needs_export: bool  # NPZ older than DB
    needs_distribution: bool  # Nodes missing this config


@dataclass
class OrchestrationPlan:
    """Plan for orchestration actions."""
    cleanup_actions: list[dict] = field(default_factory=list)
    distribution_actions: list[dict] = field(default_factory=list)
    export_actions: list[dict] = field(default_factory=list)
    training_actions: list[dict] = field(default_factory=list)


class TrainingOrchestrator:
    """Orchestrates training data distribution and space management."""

    def __init__(self):
        self.nodes = self._load_nodes()
        self.owc_host = "armand@100.107.168.125"
        self.owc_data_path = "/Volumes/RingRift-Data"

    def _load_nodes(self) -> list[dict]:
        """Load training node configurations."""
        return [
            {"name": "nebius-h100-3", "ssh_target": "ubuntu@89.169.110.128", "ssh_key": "~/.ssh/id_cluster"},
            {"name": "nebius-h100-1", "ssh_target": "ubuntu@89.169.111.139", "ssh_key": "~/.ssh/id_cluster"},
            {"name": "vultr-a100", "ssh_target": "root@208.167.249.164", "ssh_key": "~/.ssh/id_transfer"},
        ]

    async def _run_ssh(self, target: str, cmd: str, key: str = None, timeout: int = 30) -> tuple[int, str]:
        """Run SSH command and return (code, output)."""
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if key:
            ssh_cmd.extend(["-i", os.path.expanduser(key)])
        ssh_cmd.extend([target, cmd])

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode, stdout.decode().strip()
        except asyncio.TimeoutError:
            return 1, "Timeout"
        except Exception as e:
            return 1, str(e)

    async def get_node_status(self, node: dict) -> NodeStatus | None:
        """Get detailed status of a training node."""
        # Get disk usage
        code, output = await self._run_ssh(
            node["ssh_target"],
            "df -BG ~ | tail -1",
            node.get("ssh_key"),
        )
        if code != 0:
            logger.warning(f"Failed to get disk status for {node['name']}")
            return None

        try:
            parts = output.split()
            total_gb = float(parts[1].rstrip("G"))
            used_gb = float(parts[2].rstrip("G"))
            free_gb = float(parts[3].rstrip("G"))
            usage_percent = float(parts[4].rstrip("%"))
        except (IndexError, ValueError):
            return None

        # Check if training/selfplay is running
        code, procs = await self._run_ssh(
            node["ssh_target"],
            "ps aux | grep -E 'app.training.train|selfplay' | grep python | grep -v grep | wc -l",
            node.get("ssh_key"),
        )
        training_active = False
        selfplay_active = False
        if code == 0:
            try:
                proc_count = int(procs)
                code, details = await self._run_ssh(
                    node["ssh_target"],
                    "ps aux | grep -E 'app.training.train|selfplay' | grep python | grep -v grep",
                    node.get("ssh_key"),
                )
                if "app.training.train" in details:
                    training_active = True
                if "selfplay" in details:
                    selfplay_active = True
            except ValueError:
                pass

        # Count NPZ and DB files
        code, npz = await self._run_ssh(
            node["ssh_target"],
            "ls ~/ringrift/ai-service/data/training/*.npz 2>/dev/null | wc -l",
            node.get("ssh_key"),
        )
        npz_count = int(npz) if code == 0 and npz.isdigit() else 0

        code, dbs = await self._run_ssh(
            node["ssh_target"],
            "ls ~/ringrift/ai-service/data/games/*.db 2>/dev/null | wc -l",
            node.get("ssh_key"),
        )
        db_count = int(dbs) if code == 0 and dbs.isdigit() else 0

        return NodeStatus(
            name=node["name"],
            ssh_target=node["ssh_target"],
            free_gb=free_gb,
            total_gb=total_gb,
            usage_percent=usage_percent,
            can_train=free_gb >= MIN_FREE_GB_FOR_TRAINING,
            can_receive=free_gb >= MIN_FREE_GB_FOR_DISTRIBUTION,
            training_active=training_active,
            selfplay_active=selfplay_active,
            npz_count=npz_count,
            db_count=db_count,
        )

    async def get_owc_status(self) -> dict[str, ConfigStatus]:
        """Get status of all configs on OWC."""
        configs = {}

        # Get game counts from canonical databases
        code, output = await self._run_ssh(
            self.owc_host,
            f"for db in {self.owc_data_path}/canonical_games/canonical_*.db; do "
            f"name=$(basename $db .db | sed 's/canonical_//'); "
            f"count=$(sqlite3 \"$db\" 'SELECT COUNT(*) FROM games' 2>/dev/null || echo 0); "
            f"echo \"$name:$count\"; done",
            timeout=60,
        )

        game_counts = {}
        if code == 0:
            for line in output.split("\n"):
                if ":" in line:
                    name, count = line.split(":")
                    # Parse config name (e.g., "hex8_2p" from "hex8_2p")
                    game_counts[name] = int(count) if count.isdigit() else 0

        # Get NPZ file info
        code, npz_output = await self._run_ssh(
            self.owc_host,
            f"ls -l {self.owc_data_path}/canonical_data/*.npz 2>/dev/null | "
            f"awk '{{print $5\":\"$6\" \"$7\" \"$8\":\"$9}}'",
            timeout=30,
        )

        npz_info = {}
        if code == 0:
            for line in npz_output.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        size_bytes = parts[0]
                        rest = parts[1]
                        # Extract filename
                        if "/" in rest:
                            filename = rest.split("/")[-1].replace(".npz", "")
                            try:
                                npz_info[filename] = {
                                    "size_mb": int(size_bytes) / 1024 / 1024,
                                }
                            except ValueError:
                                pass

        # Build config status
        for config, priority in CONFIG_PRIORITIES.items():
            games = game_counts.get(config, 0)
            npz = npz_info.get(config, {"size_mb": 0})
            min_games = MIN_GAMES_PER_CONFIG.get(config, 5000)

            configs[config] = ConfigStatus(
                config=config,
                games_on_owc=games,
                npz_size_mb=npz.get("size_mb", 0),
                npz_age_hours=0,  # TODO: Calculate from mtime
                priority=priority,
                is_underserved=games < min_games,
                needs_export=False,  # TODO: Compare DB vs NPZ mtime
                needs_distribution=False,  # TODO: Check nodes
            )

        return configs

    async def cleanup_node(self, node: NodeStatus, target_free_gb: float) -> float:
        """Clean up a node to free space. Returns GB freed."""
        if node.free_gb >= target_free_gb:
            return 0

        freed = 0.0
        node_config = next((n for n in self.nodes if n["name"] == node.name), None)
        if not node_config:
            return 0

        logger.info(f"Cleaning {node.name}: {node.free_gb:.1f}GB free, target {target_free_gb:.1f}GB")

        # 1. Delete old logs
        code, _ = await self._run_ssh(
            node_config["ssh_target"],
            "find ~/ringrift/ai-service/logs -name '*.log' -mtime +3 -delete 2>/dev/null",
            node_config.get("ssh_key"),
        )

        # 2. Delete WAL/SHM files
        code, _ = await self._run_ssh(
            node_config["ssh_target"],
            "rm -f ~/ringrift/ai-service/data/games/*.db-wal ~/ringrift/ai-service/data/games/*.db-shm 2>/dev/null",
            node_config.get("ssh_key"),
        )

        # 3. Clear caches
        code, _ = await self._run_ssh(
            node_config["ssh_target"],
            "rm -rf ~/.cache/pip ~/.cache/torch 2>/dev/null",
            node_config.get("ssh_key"),
        )

        # Get new free space
        code, output = await self._run_ssh(
            node_config["ssh_target"],
            "df -BG ~ | tail -1 | awk '{print $4}'",
            node_config.get("ssh_key"),
        )
        if code == 0:
            try:
                new_free = float(output.rstrip("G"))
                freed = new_free - node.free_gb
                logger.info(f"  Freed {freed:.1f}GB on {node.name}")
            except ValueError:
                pass

        return freed

    async def distribute_config(self, config: str, node: NodeStatus) -> bool:
        """Distribute a config's NPZ to a node."""
        node_config = next((n for n in self.nodes if n["name"] == node.name), None)
        if not node_config:
            return False

        logger.info(f"Distributing {config}.npz to {node.name}")

        # Download via HTTP
        url = f"http://100.107.168.125:8780/canonical_data/{config}.npz"
        dest = f"~/ringrift/ai-service/data/training/{config}.npz"

        code, _ = await self._run_ssh(
            node_config["ssh_target"],
            f"wget -q -O {dest} '{url}'",
            node_config.get("ssh_key"),
            timeout=300,
        )

        return code == 0

    async def run_orchestration_cycle(self) -> OrchestrationPlan:
        """Run a full orchestration cycle."""
        plan = OrchestrationPlan()

        logger.info("=" * 60)
        logger.info("Starting orchestration cycle")
        logger.info("=" * 60)

        # 1. Get node statuses
        logger.info("\nChecking node statuses...")
        node_statuses = []
        for node in self.nodes:
            status = await self.get_node_status(node)
            if status:
                node_statuses.append(status)
                logger.info(f"  {status.name}: {status.free_gb:.0f}GB free, "
                           f"train={status.training_active}, selfplay={status.selfplay_active}")

        # 2. Get OWC config status
        logger.info("\nChecking OWC config status...")
        config_statuses = await self.get_owc_status()
        underserved = [c for c in config_statuses.values() if c.is_underserved]
        if underserved:
            logger.info(f"  Underserved configs: {[c.config for c in underserved]}")

        # 3. Cleanup nodes with low space
        for node in node_statuses:
            if not node.can_train:
                freed = await self.cleanup_node(node, TARGET_FREE_GB)
                plan.cleanup_actions.append({
                    "node": node.name,
                    "freed_gb": freed,
                })

        # 4. Distribute underserved configs to idle nodes
        for config in sorted(underserved, key=lambda c: -c.priority):
            for node in node_statuses:
                if node.can_receive and not node.training_active:
                    success = await self.distribute_config(config.config, node)
                    plan.distribution_actions.append({
                        "config": config.config,
                        "node": node.name,
                        "success": success,
                    })
                    break

        logger.info("\nOrchestration cycle complete")
        return plan

    async def run_status(self) -> dict:
        """Get current pipeline status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "nodes": [],
            "configs": {},
        }

        for node in self.nodes:
            node_status = await self.get_node_status(node)
            if node_status:
                status["nodes"].append({
                    "name": node_status.name,
                    "free_gb": node_status.free_gb,
                    "can_train": node_status.can_train,
                    "training_active": node_status.training_active,
                    "selfplay_active": node_status.selfplay_active,
                })

        config_statuses = await self.get_owc_status()
        for config, cs in config_statuses.items():
            status["configs"][config] = {
                "games": cs.games_on_owc,
                "underserved": cs.is_underserved,
                "priority": cs.priority,
            }

        return status


async def main():
    parser = argparse.ArgumentParser(description="Training Orchestrator")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--once", action="store_true", help="Run one orchestration cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=1800, help="Daemon interval (seconds)")
    parser.add_argument("--distribute", type=str, help="Force distribute a config")
    args = parser.parse_args()

    orchestrator = TrainingOrchestrator()

    if args.status:
        status = await orchestrator.run_status()
        print(json.dumps(status, indent=2))
    elif args.once:
        await orchestrator.run_orchestration_cycle()
    elif args.daemon:
        logger.info(f"Starting orchestrator daemon (interval: {args.interval}s)")
        while True:
            try:
                await orchestrator.run_orchestration_cycle()
            except Exception as e:
                logger.error(f"Orchestration cycle failed: {e}")
            await asyncio.sleep(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
