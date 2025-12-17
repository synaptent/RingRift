#!/usr/bin/env python3
"""Cluster Sync Integration - Unified data and model synchronization.

DEPRECATED (2025-12-16): This script overlaps with cluster_sync_coordinator.py.
Use cluster_sync_coordinator.py instead. This file will be removed in a future release.
See docs/RESOURCE_MANAGEMENT.md for sync tool consolidation notes.

This script provides integrated synchronization for:
1. Model distribution across all cluster nodes
2. Training data aggregation from distributed selfplay
3. Elo database consolidation
4. Lineage tracking across distributed training

Features:
- Bidirectional model sync (push best models, pull new candidates)
- Incremental data sync with manifest deduplication
- Elo database merge with conflict resolution
- Health monitoring and auto-recovery

Usage:
    # Full sync cycle
    python scripts/cluster_sync_integration.py --full-sync

    # Models only
    python scripts/cluster_sync_integration.py --sync-models

    # Data aggregation only
    python scripts/cluster_sync_integration.py --aggregate-data

    # Continuous sync daemon
    python scripts/cluster_sync_integration.py --daemon --interval 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ClusterSync")

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ClusterNode:
    """Represents a cluster node."""
    name: str
    host: str
    user: str = "ubuntu"
    ai_service_path: str = "~/ringrift/ai-service"
    has_gpu: bool = True
    is_reachable: bool = False
    last_seen: Optional[str] = None


@dataclass
class SyncConfig:
    """Configuration for cluster sync."""
    # Node configuration
    nodes: List[ClusterNode] = field(default_factory=list)

    # Sync settings
    ssh_timeout: int = 30
    rsync_bandwidth_limit: str = "50m"  # 50 MB/s
    parallel_transfers: int = 4

    # Paths
    models_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "models")
    data_dir: Path = field(default_factory=lambda: AI_SERVICE_ROOT / "data")
    nfs_path: Optional[Path] = None  # If using NFS

    # Model patterns to sync
    model_patterns: List[str] = field(default_factory=lambda: [
        "ringrift_best_*.pth",
        "nnue_*.pt",
        "*.meta.json",
    ])

    # Data patterns to aggregate
    data_patterns: List[str] = field(default_factory=lambda: [
        "gpu_selfplay/*/*.db",
        "gpu_selfplay/*/*.jsonl",
        "games/*.db",
    ])


class ClusterSyncManager:
    """Manages cluster-wide synchronization."""

    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()

        # Try to load node config from file
        nodes_file = AI_SERVICE_ROOT / "config" / "cluster_nodes.json"
        if nodes_file.exists():
            self._load_nodes_from_file(nodes_file)
        else:
            self._discover_nodes_from_env()

        logger.info(f"ClusterSyncManager initialized with {len(self.config.nodes)} nodes")

    def _load_nodes_from_file(self, path: Path):
        """Load node configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        self.config.nodes = [
            ClusterNode(**node) for node in data.get("nodes", [])
        ]

    def _discover_nodes_from_env(self):
        """Discover nodes from environment variables."""
        # GH200 nodes
        gh200_hosts = [
            ("GH200-A", "192.222.51.162"),
            ("GH200-B", "192.222.51.167"),
            ("GH200-C", "192.222.57.162"),
            ("GH200-D", "192.222.57.178"),
        ]

        for name, host in gh200_hosts:
            self.config.nodes.append(ClusterNode(
                name=name,
                host=host,
                has_gpu=True
            ))

        # H100 node
        h100_host = os.environ.get("H100_HOST", "209.20.157.81")
        self.config.nodes.append(ClusterNode(
            name="H100",
            host=h100_host,
            has_gpu=True
        ))

    def check_node_health(self, node: ClusterNode) -> bool:
        """Check if a node is reachable and healthy."""
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", f"ConnectTimeout={self.config.ssh_timeout}",
                    "-o", "StrictHostKeyChecking=no",
                    f"{node.user}@{node.host}",
                    "echo OK && uptime"
                ],
                capture_output=True,
                text=True,
                timeout=self.config.ssh_timeout + 5
            )

            node.is_reachable = result.returncode == 0
            if node.is_reachable:
                node.last_seen = datetime.now(timezone.utc).isoformat()

            return node.is_reachable

        except Exception as e:
            logger.debug(f"Node {node.name} health check failed: {e}")
            node.is_reachable = False
            return False

    def check_all_nodes(self) -> Dict[str, bool]:
        """Check health of all nodes in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.parallel_transfers) as executor:
            futures = {
                executor.submit(self.check_node_health, node): node
                for node in self.config.nodes
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[node.name] = future.result()
                except Exception as e:
                    logger.error(f"Health check failed for {node.name}: {e}")
                    results[node.name] = False

        reachable = sum(1 for v in results.values() if v)
        logger.info(f"Node health: {reachable}/{len(results)} reachable")
        return results

    def sync_models_to_node(self, node: ClusterNode) -> bool:
        """Sync models to a specific node."""
        if not node.is_reachable:
            logger.warning(f"Skipping unreachable node: {node.name}")
            return False

        try:
            # Build rsync command
            src_patterns = []
            for pattern in self.config.model_patterns:
                src_patterns.extend(self.config.models_dir.glob(pattern))

            if not src_patterns:
                logger.info("No models to sync")
                return True

            # Create include file
            include_file = Path("/tmp/rsync_models_include.txt")
            with open(include_file, 'w') as f:
                for p in src_patterns:
                    f.write(f"+ {p.name}\n")
                f.write("- *\n")

            cmd = [
                "rsync", "-avz", "--progress",
                f"--bwlimit={self.config.rsync_bandwidth_limit}",
                f"--include-from={include_file}",
                f"{self.config.models_dir}/",
                f"{node.user}@{node.host}:{node.ai_service_path}/models/"
            ]

            logger.info(f"Syncing models to {node.name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"Rsync to {node.name} failed: {result.stderr}")
                return False

            return True

        except Exception as e:
            logger.error(f"Model sync to {node.name} failed: {e}")
            return False

    def sync_models_from_node(self, node: ClusterNode) -> bool:
        """Pull models from a specific node."""
        if not node.is_reachable:
            return False

        try:
            # Create temp dir for incoming models
            incoming_dir = self.config.models_dir / "incoming" / node.name
            incoming_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "rsync", "-avz",
                f"--bwlimit={self.config.rsync_bandwidth_limit}",
                f"{node.user}@{node.host}:{node.ai_service_path}/models/*.pth",
                f"{incoming_dir}/"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.warning(f"No models to pull from {node.name}")
                return False

            # Check for new models
            for model_path in incoming_dir.glob("*.pth"):
                target = self.config.models_dir / model_path.name
                if not target.exists():
                    logger.info(f"New model from {node.name}: {model_path.name}")
                    shutil.copy2(model_path, target)

            return True

        except Exception as e:
            logger.error(f"Model pull from {node.name} failed: {e}")
            return False

    def sync_models_all(self) -> Dict[str, bool]:
        """Sync models to/from all nodes."""
        results = {}

        # First, pull new models from all nodes
        logger.info("Pulling models from cluster...")
        for node in self.config.nodes:
            if node.is_reachable:
                results[f"{node.name}_pull"] = self.sync_models_from_node(node)

        # Then push latest models to all nodes
        logger.info("Pushing models to cluster...")
        with ThreadPoolExecutor(max_workers=self.config.parallel_transfers) as executor:
            futures = {
                executor.submit(self.sync_models_to_node, node): node
                for node in self.config.nodes
                if node.is_reachable
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[f"{node.name}_push"] = future.result()
                except Exception as e:
                    logger.error(f"Push to {node.name} failed: {e}")
                    results[f"{node.name}_push"] = False

        return results

    def aggregate_data_from_node(self, node: ClusterNode) -> Tuple[bool, int]:
        """Aggregate training data from a specific node."""
        if not node.is_reachable:
            return False, 0

        try:
            # Create local directory for this node's data
            node_data_dir = self.config.data_dir / "cluster" / node.name
            node_data_dir.mkdir(parents=True, exist_ok=True)

            total_files = 0

            for pattern in self.config.data_patterns:
                cmd = [
                    "rsync", "-avz", "--progress",
                    f"--bwlimit={self.config.rsync_bandwidth_limit}",
                    f"{node.user}@{node.host}:{node.ai_service_path}/data/{pattern}",
                    f"{node_data_dir}/"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode == 0:
                    # Count files synced
                    for line in result.stdout.split('\n'):
                        if line.endswith('.db') or line.endswith('.jsonl'):
                            total_files += 1

            logger.info(f"Aggregated {total_files} files from {node.name}")
            return True, total_files

        except Exception as e:
            logger.error(f"Data aggregation from {node.name} failed: {e}")
            return False, 0

    def aggregate_data_all(self) -> Dict[str, Tuple[bool, int]]:
        """Aggregate data from all nodes."""
        results = {}

        logger.info("Aggregating data from cluster...")

        with ThreadPoolExecutor(max_workers=self.config.parallel_transfers) as executor:
            futures = {
                executor.submit(self.aggregate_data_from_node, node): node
                for node in self.config.nodes
                if node.is_reachable
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[node.name] = future.result()
                except Exception as e:
                    logger.error(f"Aggregation from {node.name} failed: {e}")
                    results[node.name] = (False, 0)

        total_files = sum(count for _, count in results.values())
        logger.info(f"Total files aggregated: {total_files}")
        return results

    def consolidate_elo_databases(self) -> bool:
        """Consolidate Elo databases from all nodes into unified DB."""
        try:
            from scripts.unified_improvement_controller import UnifiedEloSystem

            unified_elo = UnifiedEloSystem()
            node_dbs = []

            # Collect Elo DBs from cluster data
            cluster_dir = self.config.data_dir / "cluster"
            for node_dir in cluster_dir.iterdir():
                if node_dir.is_dir():
                    for elo_db in node_dir.glob("**/elo*.db"):
                        node_dbs.append(elo_db)

            if not node_dbs:
                logger.info("No Elo databases to consolidate")
                return True

            logger.info(f"Consolidating {len(node_dbs)} Elo databases")

            import sqlite3

            for db_path in node_dbs:
                try:
                    with sqlite3.connect(db_path) as src_conn:
                        src_conn.row_factory = sqlite3.Row

                        # Migrate participants
                        for row in src_conn.execute("SELECT * FROM participants"):
                            unified_elo.register_participant(
                                participant_id=row['id'],
                                name=row['name'],
                                ai_type=row['ai_type'],
                                difficulty=row['difficulty'],
                                use_neural_net=bool(row['use_neural_net']),
                                model_id=row.get('model_id')
                            )

                        # Migrate ratings (keep highest)
                        for row in src_conn.execute("SELECT * FROM elo_ratings"):
                            current = unified_elo.get_or_create_rating(
                                row['participant_id'],
                                row['board_type'],
                                row['num_players']
                            )
                            if row['rating'] > current:
                                # Update if higher
                                with sqlite3.connect(unified_elo.db_path) as dst_conn:
                                    dst_conn.execute("""
                                        UPDATE elo_ratings
                                        SET rating = ?, games_played = games_played + ?,
                                            wins = wins + ?, losses = losses + ?
                                        WHERE participant_id = ? AND board_type = ? AND num_players = ?
                                    """, (
                                        row['rating'],
                                        row['games_played'],
                                        row['wins'],
                                        row['losses'],
                                        row['participant_id'],
                                        row['board_type'],
                                        row['num_players']
                                    ))
                                    dst_conn.commit()

                except Exception as e:
                    logger.warning(f"Failed to consolidate {db_path}: {e}")

            return True

        except Exception as e:
            logger.error(f"Elo consolidation failed: {e}")
            return False

    def full_sync(self) -> Dict[str, Any]:
        """Perform a full synchronization cycle."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": {},
            "models": {},
            "data": {},
            "elo": False
        }

        # 1. Check node health
        logger.info("=== Checking node health ===")
        results["health"] = self.check_all_nodes()

        # 2. Sync models
        logger.info("=== Syncing models ===")
        results["models"] = self.sync_models_all()

        # 3. Aggregate data
        logger.info("=== Aggregating data ===")
        results["data"] = self.aggregate_data_all()

        # 4. Consolidate Elo
        logger.info("=== Consolidating Elo databases ===")
        results["elo"] = self.consolidate_elo_databases()

        return results

    def run_daemon(self, interval_seconds: int = 300):
        """Run as a continuous sync daemon."""
        logger.info(f"Starting sync daemon, interval: {interval_seconds}s")

        try:
            while True:
                try:
                    results = self.full_sync()

                    # Log summary
                    healthy = sum(1 for v in results["health"].values() if v)
                    total_data = sum(
                        count for ok, count in results["data"].values() if ok
                    )
                    logger.info(
                        f"Sync complete: {healthy} healthy nodes, "
                        f"{total_data} data files aggregated"
                    )

                except Exception as e:
                    logger.error(f"Sync cycle failed: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted, shutting down")

    def get_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        self.check_all_nodes()

        return {
            "nodes": [
                {
                    "name": n.name,
                    "host": n.host,
                    "reachable": n.is_reachable,
                    "last_seen": n.last_seen,
                    "has_gpu": n.has_gpu
                }
                for n in self.config.nodes
            ],
            "config": {
                "models_dir": str(self.config.models_dir),
                "data_dir": str(self.config.data_dir),
                "nfs_path": str(self.config.nfs_path) if self.config.nfs_path else None
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Cluster Sync Integration")

    parser.add_argument("--full-sync", action="store_true", help="Full sync cycle")
    parser.add_argument("--sync-models", action="store_true", help="Sync models only")
    parser.add_argument("--aggregate-data", action="store_true", help="Aggregate data only")
    parser.add_argument("--consolidate-elo", action="store_true", help="Consolidate Elo DBs")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval (seconds)")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    manager = ClusterSyncManager()

    if args.status:
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        return

    if args.daemon:
        manager.run_daemon(args.interval)
        return

    if args.full_sync:
        results = manager.full_sync()
        print(json.dumps(results, indent=2, default=str))
        return

    if args.sync_models:
        manager.check_all_nodes()
        results = manager.sync_models_all()
        print(json.dumps(results, indent=2))
        return

    if args.aggregate_data:
        manager.check_all_nodes()
        results = manager.aggregate_data_all()
        print(json.dumps({k: {"success": v[0], "files": v[1]} for k, v in results.items()}, indent=2))
        return

    if args.consolidate_elo:
        success = manager.consolidate_elo_databases()
        print(f"Elo consolidation: {'success' if success else 'failed'}")
        return

    # Default: show status
    status = manager.get_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
