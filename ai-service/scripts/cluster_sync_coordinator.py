#!/usr/bin/env python3
"""Cluster Sync Coordinator - Unified orchestration of all sync utilities.

Coordinates and deduplicates functionality across:
- unified_data_sync.py (game data sync)
- elo_db_sync.py (ELO database sync)
- sync_models.py (model distribution)
- aria2_data_sync.py (high-performance downloads)
- gossip_sync.py (P2P replication)

Ensures no data silos by:
1. Syncing best models to all nodes
2. Aggregating game data from all nodes
3. Keeping ELO databases consistent
4. Using aria2/tailscale/cloudflare for hard-to-reach nodes

Usage:
    python scripts/cluster_sync_coordinator.py --mode full
    python scripts/cluster_sync_coordinator.py --mode models
    python scripts/cluster_sync_coordinator.py --mode games
    python scripts/cluster_sync_coordinator.py --mode elo
    python scripts/cluster_sync_coordinator.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ClusterCoord] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    name: str
    tailscale_ip: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ssh_key: str = "~/.ssh/id_cluster"
    reachable: bool = False
    last_check: float = 0.0
    selfplay_processes: int = 0
    game_count: int = 0
    model_version: str = ""
    elo_db_hash: str = ""
    transport: str = "unknown"  # tailscale, ssh, aria2


@dataclass
class ClusterState:
    """Current state of the entire cluster."""
    nodes: Dict[str, NodeStatus] = field(default_factory=dict)
    best_models: Dict[str, str] = field(default_factory=dict)  # board_type -> model_path
    total_games: int = 0
    last_sync: float = 0.0
    sync_in_progress: bool = False


class ClusterSyncCoordinator:
    """Coordinates all sync operations across the cluster."""

    def __init__(self, hosts_config: Path, sync_config: Optional[Path] = None):
        self.hosts_config = hosts_config
        self.sync_config = sync_config
        self.state = ClusterState()
        self.executor = ThreadPoolExecutor(max_workers=20)

        # Load configuration
        self._load_hosts()

    def _load_hosts(self):
        """Load host configuration from YAML."""
        if not self.hosts_config.exists():
            logger.error(f"Hosts config not found: {self.hosts_config}")
            return

        with open(self.hosts_config) as f:
            config = yaml.safe_load(f)

        hosts = config.get("hosts", {})
        for name, info in hosts.items():
            if info.get("status") == "terminated":
                continue

            self.state.nodes[name] = NodeStatus(
                name=name,
                tailscale_ip=info.get("tailscale_ip", info.get("ssh_host", "")),
                ssh_host=info.get("ssh_host", ""),
                ssh_port=info.get("ssh_port", 22),
                ssh_user=info.get("ssh_user", "ubuntu"),
                ssh_key=info.get("ssh_key", "~/.ssh/id_cluster"),
            )

        logger.info(f"Loaded {len(self.state.nodes)} nodes from config")

    def _ssh_exec(self, node: NodeStatus, cmd: str, timeout: int = 30) -> Tuple[bool, str]:
        """Execute command on node via SSH, preferring Tailscale IP."""
        # Try Tailscale first, then fall back to SSH host
        hosts_to_try = []
        if node.tailscale_ip:
            hosts_to_try.append((node.tailscale_ip, "tailscale"))
        if node.ssh_host and node.ssh_host != node.tailscale_ip:
            hosts_to_try.append((node.ssh_host, "ssh"))

        for host, transport in hosts_to_try:
            try:
                ssh_cmd = [
                    "ssh",
                    "-o", "ConnectTimeout=10",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "BatchMode=yes",
                ]
                if node.ssh_port != 22:
                    ssh_cmd.extend(["-p", str(node.ssh_port)])
                if node.ssh_key:
                    ssh_cmd.extend(["-i", os.path.expanduser(node.ssh_key)])
                ssh_cmd.extend([f"{node.ssh_user}@{host}", cmd])

                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode == 0:
                    node.transport = transport
                    return True, result.stdout.strip()

            except subprocess.TimeoutExpired:
                continue
            except Exception as e:
                continue

        return False, ""

    async def check_node_status(self, node: NodeStatus) -> bool:
        """Check status of a single node."""
        loop = asyncio.get_event_loop()

        # Check reachability
        success, output = await loop.run_in_executor(
            self.executor,
            lambda: self._ssh_exec(node, "echo ok", timeout=15)
        )

        node.reachable = success
        node.last_check = time.time()

        if not success:
            return False

        # Check selfplay processes
        success, output = await loop.run_in_executor(
            self.executor,
            lambda: self._ssh_exec(
                node,
                "ps aux | grep -E 'run_diverse_selfplay|run_gpu_selfplay' | grep -v grep | wc -l",
                timeout=15
            )
        )
        if success:
            try:
                node.selfplay_processes = int(output.strip())
            except ValueError:
                node.selfplay_processes = 0

        return True

    async def check_all_nodes(self) -> Dict[str, bool]:
        """Check status of all nodes in parallel."""
        tasks = []
        for node in self.state.nodes.values():
            tasks.append(self.check_node_status(node))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        status = {}
        for node, result in zip(self.state.nodes.values(), results):
            if isinstance(result, Exception):
                status[node.name] = False
            else:
                status[node.name] = result

        return status

    async def sync_models_to_node(self, node: NodeStatus, model_paths: Dict[str, str]) -> bool:
        """Sync best models to a specific node."""
        if not node.reachable:
            return False

        loop = asyncio.get_event_loop()

        for board_type, model_path in model_paths.items():
            if not Path(model_path).exists():
                continue

            # Use rsync for model transfer
            dest_path = f"~/ringrift/ai-service/models/{Path(model_path).name}"

            rsync_cmd = [
                "rsync", "-avz", "--progress",
                "-e", f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i {os.path.expanduser(node.ssh_key)}",
                str(model_path),
                f"{node.ssh_user}@{node.tailscale_ip or node.ssh_host}:{dest_path}"
            ]

            try:
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
                )
                if result.returncode == 0:
                    logger.info(f"Synced {board_type} model to {node.name}")
                else:
                    logger.warning(f"Failed to sync model to {node.name}: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"Error syncing to {node.name}: {e}")
                return False

        return True

    async def sync_models_cluster_wide(self, model_paths: Dict[str, str]) -> int:
        """Sync models to all reachable nodes."""
        logger.info(f"Syncing {len(model_paths)} models to cluster...")

        # First check all nodes
        await self.check_all_nodes()

        reachable = [n for n in self.state.nodes.values() if n.reachable]
        logger.info(f"Found {len(reachable)} reachable nodes")

        # Sync to all reachable nodes
        tasks = [self.sync_models_to_node(node, model_paths) for node in reachable]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        logger.info(f"Model sync complete: {success_count}/{len(reachable)} nodes updated")

        return success_count

    async def collect_games_from_node(self, node: NodeStatus, dest_dir: Path) -> int:
        """Collect game databases from a node."""
        if not node.reachable:
            return 0

        loop = asyncio.get_event_loop()
        node_dest = dest_dir / node.name
        node_dest.mkdir(parents=True, exist_ok=True)

        # Use rsync to collect game databases
        rsync_cmd = [
            "rsync", "-avz",
            "--include=*.db",
            "--include=diverse_selfplay/",
            "--include=diverse_selfplay/*.db",
            "--exclude=*",
            "-e", f"ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -i {os.path.expanduser(node.ssh_key)}",
            f"{node.ssh_user}@{node.tailscale_ip or node.ssh_host}:~/ringrift/ai-service/data/games/",
            str(node_dest) + "/"
        ]

        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
            )

            # Count collected files
            db_files = list(node_dest.glob("**/*.db"))
            return len(db_files)

        except Exception as e:
            logger.error(f"Error collecting from {node.name}: {e}")
            return 0

    async def collect_games_cluster_wide(self, dest_dir: Path) -> int:
        """Collect games from all nodes."""
        logger.info("Collecting games from all nodes...")

        await self.check_all_nodes()
        reachable = [n for n in self.state.nodes.values() if n.reachable]

        tasks = [self.collect_games_from_node(node, dest_dir) for node in reachable]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total = sum(r for r in results if isinstance(r, int))
        logger.info(f"Collected {total} database files from {len(reachable)} nodes")

        return total

    async def sync_elo_database(self) -> bool:
        """Sync ELO database to all nodes using existing elo_db_sync."""
        logger.info("Syncing ELO database cluster-wide...")

        # Use existing elo_db_sync.py script
        sync_script = AI_SERVICE_ROOT / "scripts" / "elo_db_sync.py"
        if not sync_script.exists():
            logger.error("elo_db_sync.py not found")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(sync_script), "--push", "--all-nodes"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(AI_SERVICE_ROOT),
            )

            if result.returncode == 0:
                logger.info("ELO sync successful")
                return True
            else:
                logger.warning(f"ELO sync had issues: {result.stderr[:500]}")
                return False

        except Exception as e:
            logger.error(f"ELO sync failed: {e}")
            return False

    async def run_full_sync(self) -> Dict[str, any]:
        """Run a full cluster synchronization."""
        self.state.sync_in_progress = True
        results = {
            "nodes_checked": 0,
            "nodes_reachable": 0,
            "models_synced": 0,
            "games_collected": 0,
            "elo_synced": False,
        }

        try:
            # 1. Check all nodes
            status = await self.check_all_nodes()
            results["nodes_checked"] = len(status)
            results["nodes_reachable"] = sum(1 for v in status.values() if v)

            # 2. Sync ELO database
            results["elo_synced"] = await self.sync_elo_database()

            # 3. Collect games from all nodes
            collect_dir = AI_SERVICE_ROOT / "data" / "games" / "collected"
            results["games_collected"] = await self.collect_games_cluster_wide(collect_dir)

            # 4. Find and sync best models (if any)
            models_dir = AI_SERVICE_ROOT / "models"
            if models_dir.exists():
                best_models = {}
                for model_file in models_dir.glob("*.pth"):
                    # Simple heuristic: latest model per board type
                    name = model_file.stem
                    for board in ["square8", "square19", "hexagonal", "hex"]:
                        if board in name.lower():
                            if board not in best_models or model_file.stat().st_mtime > Path(best_models[board]).stat().st_mtime:
                                best_models[board] = str(model_file)

                if best_models:
                    results["models_synced"] = await self.sync_models_cluster_wide(best_models)

            self.state.last_sync = time.time()

        finally:
            self.state.sync_in_progress = False

        return results

    def get_status_report(self) -> str:
        """Generate a status report of the cluster."""
        lines = [
            "=" * 60,
            "CLUSTER STATUS REPORT",
            "=" * 60,
            "",
        ]

        # Node summary
        total = len(self.state.nodes)
        reachable = sum(1 for n in self.state.nodes.values() if n.reachable)
        selfplay = sum(n.selfplay_processes for n in self.state.nodes.values())

        lines.append(f"Nodes: {reachable}/{total} reachable")
        lines.append(f"Total selfplay processes: {selfplay}")
        lines.append("")

        # Per-node details
        lines.append("Node Details:")
        lines.append("-" * 50)

        for name, node in sorted(self.state.nodes.items()):
            status = "OK" if node.reachable else "DOWN"
            transport = f"[{node.transport}]" if node.reachable else ""
            procs = f"{node.selfplay_processes} procs" if node.selfplay_processes > 0 else "idle"
            lines.append(f"  {name}: {status} {transport} - {procs}")

        lines.append("")
        lines.append(f"Last sync: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.state.last_sync)) if self.state.last_sync else 'Never'}")
        lines.append("=" * 60)

        return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Cluster Sync Coordinator")
    parser.add_argument(
        "--mode",
        choices=["full", "models", "games", "elo", "check"],
        default="check",
        help="Sync mode",
    )
    parser.add_argument("--status", action="store_true", help="Show cluster status")
    parser.add_argument(
        "--hosts",
        type=str,
        default="config/distributed_hosts.yaml",
        help="Hosts config file",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    hosts_path = AI_SERVICE_ROOT / args.hosts
    coordinator = ClusterSyncCoordinator(hosts_path)

    if args.status or args.mode == "check":
        await coordinator.check_all_nodes()
        print(coordinator.get_status_report())
        return

    if args.mode == "full":
        results = await coordinator.run_full_sync()
        print(f"\nSync Results:")
        print(f"  Nodes checked: {results['nodes_checked']}")
        print(f"  Nodes reachable: {results['nodes_reachable']}")
        print(f"  ELO synced: {results['elo_synced']}")
        print(f"  Games collected: {results['games_collected']}")
        print(f"  Models synced: {results['models_synced']}")

    elif args.mode == "models":
        # Just sync models
        await coordinator.check_all_nodes()
        models_dir = AI_SERVICE_ROOT / "models"
        best_models = {}
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                name = model_file.stem
                for board in ["square8", "square19", "hexagonal"]:
                    if board in name.lower():
                        best_models[board] = str(model_file)
        if best_models:
            count = await coordinator.sync_models_cluster_wide(best_models)
            print(f"Synced models to {count} nodes")
        else:
            print("No models found to sync")

    elif args.mode == "games":
        collect_dir = AI_SERVICE_ROOT / "data" / "games" / "collected"
        count = await coordinator.collect_games_cluster_wide(collect_dir)
        print(f"Collected {count} database files")

    elif args.mode == "elo":
        success = await coordinator.sync_elo_database()
        print(f"ELO sync: {'success' if success else 'failed'}")


if __name__ == "__main__":
    asyncio.run(main())
