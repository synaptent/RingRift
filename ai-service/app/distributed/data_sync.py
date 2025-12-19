#!/usr/bin/env python3
"""
Distributed Data Synchronization Module

.. deprecated:: 2025-12-18
    This module is deprecated. Use the following replacements:

    **Sync Coordination** (cluster-wide orchestration):
        - :class:`app.coordination.sync_coordinator.SyncCoordinator`
        - Features: backpressure-aware scheduling, priority queues, host tracking

    **Low-Level Sync** (actual data transfer):
        - :class:`app.distributed.unified_data_sync.UnifiedDataSyncService`
        - Features: multi-transport failover, WAL, deduplication, circuit breaker

    **Data Discovery** (finding training data):
        - :class:`app.distributed.data_catalog.DataCatalog`
        - Features: cluster-wide discovery, quality-aware selection

    **Quality Scoring** (game quality computation):
        - :class:`app.quality.unified_quality.UnifiedQualityScorer`
        - Features: unified algorithm, sync priority computation

    **Note**: This module remains functional during the transition period.
    Scripts using `DataSyncManager` will continue to work but show deprecation warnings.
    For new code, prefer the replacement modules above.

Provides multiple transport methods for synchronizing training data and models
across cluster nodes, handling hard-to-reach instances via:
- Tailscale mesh network (direct P2P)
- Cloudflare Zero Trust tunnels (for NAT traversal)
- aria2 parallel downloads (multi-source acceleration)
- Direct HTTP/rsync (fallback)

This module ensures no data silos - all nodes have access to:
1. Best models (latest checkpoints)
2. Training data (consolidated databases)
3. ELO ratings (unified_elo.db)
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.utils.checksum_utils import compute_file_checksum

import yaml

# Emit deprecation warning on module import
warnings.warn(
    "app.distributed.data_sync is deprecated since 2025-12-18. "
    "Use app.distributed.sync_orchestrator.SyncOrchestrator for unified sync access. "
    "Migration guide:\n"
    "  - For sync coordination: use app.coordination.sync_coordinator.SyncCoordinator\n"
    "  - For low-level sync: use app.distributed.unified_data_sync.UnifiedDataSyncService\n"
    "  - For data discovery: use app.distributed.data_catalog.DataCatalog\n"
    "  - For quality scoring: use app.quality.unified_quality.UnifiedQualityScorer\n"
    "This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        check_disk_space as unified_check_disk,
        get_disk_usage as unified_get_disk_usage,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_check_disk = None
    unified_get_disk_usage = None
    RESOURCE_LIMITS = None

from app.utils.env_config import env
from app.utils.paths import AI_SERVICE_ROOT, MODELS_DIR, DATA_DIR

# Disk usage limits - 70% max enforced 2025-12-16
MAX_DISK_USAGE_PERCENT = env.max_disk_percent

# Configuration
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"
SYNC_STATE_PATH = DATA_DIR / "sync_state.json"


def check_disk_usage(path: Path = None) -> Tuple[bool, float]:
    """Check if disk has capacity for syncing.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement (70% for disk).

    Args:
        path: Path to check disk usage for. Defaults to DATA_DIR.

    Returns:
        Tuple of (has_capacity, current_usage_percent)
    """
    check_path = str(path) if path else str(DATA_DIR)

    # Use unified utilities when available
    if HAS_RESOURCE_GUARD and unified_get_disk_usage is not None:
        try:
            percent, _, _ = unified_get_disk_usage(check_path)
            has_capacity = percent < MAX_DISK_USAGE_PERCENT
            if not has_capacity:
                logger.warning(f"Disk usage {percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%")
            return has_capacity, percent
        except Exception:
            pass  # Fall through to original implementation

    # Fallback to original implementation
    try:
        usage = shutil.disk_usage(check_path)
        percent = 100.0 * usage.used / usage.total
        has_capacity = percent < MAX_DISK_USAGE_PERCENT
        if not has_capacity:
            logger.warning(f"Disk usage {percent:.1f}% exceeds limit {MAX_DISK_USAGE_PERCENT}%")
        return has_capacity, percent
    except Exception as e:
        logger.error(f"Failed to check disk usage: {e}")
        return True, 0.0  # Allow sync on error (fail open)


@dataclass
class SyncTarget:
    """A file or directory to synchronize across nodes."""
    path: str
    priority: int = 5  # 1-10, higher = more important
    sync_interval: int = 300  # seconds
    checksum: Optional[str] = None
    last_sync: float = 0


@dataclass
class NodeConfig:
    """Configuration for a cluster node."""
    name: str
    ssh_host: str
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_key: str = "~/.ssh/id_cluster"
    tailscale_ip: Optional[str] = None
    cloudflare_tunnel: Optional[str] = None
    ringrift_path: str = "~/ringrift/ai-service"
    status: str = "unknown"
    # Transports for pushing files TO nodes (aria2 is for pulling only)
    transports: List[str] = field(default_factory=lambda: ["tailscale", "cloudflare", "ssh"])


class DataSyncManager:
    """Manages data synchronization across cluster nodes."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.nodes: Dict[str, NodeConfig] = {}
        self.sync_targets: List[SyncTarget] = []
        self.sync_state: Dict = {}
        self._load_config()
        self._load_sync_state()
        self._init_sync_targets()

    def _load_config(self):
        """Load cluster configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}")
            return

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        for name, host_config in config.get("hosts", {}).items():
            if host_config.get("status") == "terminated":
                continue

            ssh_host = host_config.get("tailscale_ip") or host_config.get("ssh_host")
            self.nodes[name] = NodeConfig(
                name=name,
                ssh_host=ssh_host,
                ssh_user=host_config.get("ssh_user", "root"),
                ssh_port=host_config.get("ssh_port", 22),
                ssh_key=host_config.get("ssh_key", "~/.ssh/id_cluster"),
                tailscale_ip=host_config.get("tailscale_ip"),
                cloudflare_tunnel=host_config.get("cloudflare_tunnel"),
                ringrift_path=host_config.get("ringrift_path", "~/ringrift/ai-service"),
                status=host_config.get("status", "unknown"),
            )

    def _load_sync_state(self):
        """Load synchronization state from disk."""
        if SYNC_STATE_PATH.exists():
            try:
                with open(SYNC_STATE_PATH) as f:
                    self.sync_state = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
                self.sync_state = {}

    def _save_sync_state(self):
        """Save synchronization state to disk."""
        SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SYNC_STATE_PATH, "w") as f:
            json.dump(self.sync_state, f, indent=2)

    def _init_sync_targets(self):
        """Initialize default sync targets."""
        self.sync_targets = [
            # Critical: Best models (highest priority)
            SyncTarget(
                path="models/ringrift_best_*.pth",
                priority=10,
                sync_interval=60,
            ),
            # High: ELO database
            SyncTarget(
                path="data/unified_elo.db",
                priority=9,
                sync_interval=120,
            ),
            # Medium: Training data
            SyncTarget(
                path="data/training/*.npz",
                priority=7,
                sync_interval=300,
            ),
            # Medium: Consolidated game databases
            SyncTarget(
                path="data/games/*_consolidated.db",
                priority=6,
                sync_interval=600,
            ),
        ]

    def compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        return compute_file_checksum(path, truncate=16)

    async def sync_via_tailscale(
        self,
        node: NodeConfig,
        local_path: Path,
        remote_path: str,
        direction: str = "push"
    ) -> bool:
        """Sync file via Tailscale direct connection."""
        if not node.tailscale_ip:
            return False

        try:
            ssh_key = os.path.expanduser(node.ssh_key)
            if direction == "push":
                cmd = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -i {ssh_key} -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
                    str(local_path),
                    f"{node.ssh_user}@{node.tailscale_ip}:{remote_path}",
                ]
            else:  # pull
                cmd = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -i {ssh_key} -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
                    f"{node.ssh_user}@{node.tailscale_ip}:{remote_path}",
                    str(local_path),
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            if proc.returncode == 0:
                logger.info(f"Tailscale sync to {node.name} succeeded")
                return True
            else:
                logger.warning(f"Tailscale sync to {node.name} failed: {stderr.decode()}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Tailscale sync to {node.name} timed out")
            return False
        except Exception as e:
            logger.warning(f"Tailscale sync to {node.name} error: {e}")
            return False

    async def sync_via_aria2(
        self,
        urls: List[str],
        local_path: Path,
        max_connections: int = 16,
    ) -> bool:
        """Download file using aria2 with parallel connections from multiple sources."""
        try:
            # Create aria2 input file with all source URLs
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for url in urls:
                    f.write(f"{url}\n")
                input_file = f.name

            cmd = [
                "aria2c",
                "-x", str(max_connections),  # Max connections per server
                "-s", str(min(len(urls), 16)),  # Max concurrent downloads
                "-j", str(min(len(urls), 8)),  # Max parallel downloads
                "-i", input_file,
                "-d", str(local_path.parent),
                "-o", local_path.name,
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--check-certificate=false",
                "--timeout=60",
                "--connect-timeout=15",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

            os.unlink(input_file)

            if proc.returncode == 0:
                logger.info(f"aria2 download succeeded: {local_path}")
                return True
            else:
                logger.warning(f"aria2 download failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.warning(f"aria2 download error: {e}")
            return False

    async def sync_via_cloudflare_tunnel(
        self,
        node: NodeConfig,
        local_path: Path,
        remote_path: str,
        direction: str = "push"
    ) -> bool:
        """Sync file via Cloudflare Zero Trust tunnel."""
        if not node.cloudflare_tunnel:
            return False

        try:
            # Use cloudflared access to tunnel through Zero Trust
            tunnel_url = f"https://{node.cloudflare_tunnel}"

            if direction == "push":
                # Upload via HTTP POST
                cmd = [
                    "curl", "-X", "POST",
                    "--data-binary", f"@{local_path}",
                    f"{tunnel_url}/upload?path={remote_path}",
                    "--connect-timeout", "30",
                    "--max-time", "300",
                ]
            else:
                # Download via HTTP GET
                cmd = [
                    "curl", "-o", str(local_path),
                    f"{tunnel_url}/download?path={remote_path}",
                    "--connect-timeout", "30",
                    "--max-time", "300",
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=360)

            if proc.returncode == 0:
                logger.info(f"Cloudflare tunnel sync to {node.name} succeeded")
                return True
            else:
                logger.warning(f"Cloudflare tunnel sync failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.warning(f"Cloudflare tunnel error: {e}")
            return False

    async def sync_via_ssh(
        self,
        node: NodeConfig,
        local_path: Path,
        remote_path: str,
        direction: str = "push"
    ) -> bool:
        """Sync file via direct SSH/rsync."""
        try:
            ssh_key = os.path.expanduser(node.ssh_key)
            port_arg = f"-p {node.ssh_port}" if node.ssh_port != 22 else ""

            if direction == "push":
                cmd = [
                    "rsync", "-avz",
                    "-e", f"ssh -i {ssh_key} {port_arg} -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
                    str(local_path),
                    f"{node.ssh_user}@{node.ssh_host}:{remote_path}",
                ]
            else:
                cmd = [
                    "rsync", "-avz",
                    "-e", f"ssh -i {ssh_key} {port_arg} -o ConnectTimeout=15 -o StrictHostKeyChecking=no",
                    f"{node.ssh_user}@{node.ssh_host}:{remote_path}",
                    str(local_path),
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            if proc.returncode == 0:
                logger.info(f"SSH sync to {node.name} succeeded")
                return True
            else:
                logger.warning(f"SSH sync to {node.name} failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.warning(f"SSH sync error: {e}")
            return False

    async def sync_file_to_node(
        self,
        node: NodeConfig,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Sync a file to a node using available transports (with fallback)."""
        for transport in node.transports:
            success = False

            if transport == "tailscale" and node.tailscale_ip:
                success = await self.sync_via_tailscale(node, local_path, remote_path)
            elif transport == "cloudflare" and node.cloudflare_tunnel:
                success = await self.sync_via_cloudflare_tunnel(node, local_path, remote_path)
            elif transport == "ssh":
                success = await self.sync_via_ssh(node, local_path, remote_path)

            if success:
                return True

        logger.error(f"All transports failed for {node.name}")
        return False

    async def sync_best_models(self) -> Dict[str, bool]:
        """Sync best model checkpoints to all nodes."""
        results = {}

        # Check disk usage before syncing (70% limit enforced 2025-12-16)
        has_capacity, disk_percent = check_disk_usage()
        if not has_capacity:
            logger.warning(f"Skipping model sync - disk at {disk_percent:.1f}%")
            return {"error": f"disk_full:{disk_percent:.1f}%"}

        # Find best models
        best_models = list(MODELS_DIR.glob("ringrift_best_*.pth"))
        if not best_models:
            logger.warning("No best models found to sync")
            return results

        for model_path in best_models:
            checksum = self.compute_checksum(model_path)
            model_name = model_path.name

            for node_name, node in self.nodes.items():
                if node.status != "ready":
                    continue

                # Check if node already has this version
                state_key = f"{node_name}:{model_name}"
                if self.sync_state.get(state_key) == checksum:
                    logger.debug(f"{node_name} already has {model_name}")
                    continue

                remote_path = f"{node.ringrift_path}/models/{model_name}"
                success = await self.sync_file_to_node(node, model_path, remote_path)

                if success:
                    self.sync_state[state_key] = checksum

                results[f"{node_name}:{model_name}"] = success

        self._save_sync_state()
        return results

    async def sync_elo_database(self) -> Dict[str, bool]:
        """Sync unified ELO database to all nodes."""
        results = {}

        # Check disk usage before syncing (70% limit enforced 2025-12-16)
        has_capacity, disk_percent = check_disk_usage()
        if not has_capacity:
            logger.warning(f"Skipping ELO sync - disk at {disk_percent:.1f}%")
            return {"error": f"disk_full:{disk_percent:.1f}%"}

        elo_db = DATA_DIR / "unified_elo.db"
        if not elo_db.exists():
            logger.warning("ELO database not found")
            return results

        checksum = self.compute_checksum(elo_db)

        for node_name, node in self.nodes.items():
            if node.status != "ready":
                continue

            state_key = f"{node_name}:unified_elo.db"
            if self.sync_state.get(state_key) == checksum:
                continue

            remote_path = f"{node.ringrift_path}/data/unified_elo.db"
            success = await self.sync_file_to_node(node, elo_db, remote_path)

            if success:
                self.sync_state[state_key] = checksum

            results[node_name] = success

        self._save_sync_state()
        return results

    async def collect_training_data(self, pattern: str = "hex8*.db") -> List[Path]:
        """Collect training databases from all nodes."""
        # Check disk usage before collecting (70% limit enforced 2025-12-16)
        has_capacity, disk_percent = check_disk_usage()
        if not has_capacity:
            logger.warning(f"Skipping data collection - disk at {disk_percent:.1f}%")
            return []

        collected = []
        local_sync_dir = DATA_DIR / "games" / "synced"
        local_sync_dir.mkdir(parents=True, exist_ok=True)

        for node_name, node in self.nodes.items():
            if node.status != "ready":
                continue

            try:
                # List remote databases
                ssh_key = os.path.expanduser(node.ssh_key)
                port_arg = f"-p {node.ssh_port}" if node.ssh_port != 22 else ""

                cmd = f"ssh -i {ssh_key} {port_arg} -o ConnectTimeout=10 {node.ssh_user}@{node.ssh_host} 'ls {node.ringrift_path}/data/games/{pattern} 2>/dev/null'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    continue

                remote_files = result.stdout.strip().split("\n")
                for remote_file in remote_files:
                    if not remote_file:
                        continue

                    local_file = local_sync_dir / f"{node_name}_{Path(remote_file).name}"

                    # Pull the database
                    success = await self.sync_via_ssh(
                        node, local_file, remote_file, direction="pull"
                    )

                    if success and local_file.exists():
                        collected.append(local_file)
                        logger.info(f"Collected {remote_file} from {node_name}")

            except Exception as e:
                logger.warning(f"Failed to collect from {node_name}: {e}")

        return collected

    async def run_sync_daemon(self, interval: int = 300):
        """Run continuous sync daemon."""
        logger.info(f"Starting sync daemon with {interval}s interval")

        while True:
            try:
                # Sync best models (highest priority)
                logger.info("Syncing best models...")
                model_results = await self.sync_best_models()

                # Sync ELO database
                logger.info("Syncing ELO database...")
                elo_results = await self.sync_elo_database()

                # Log summary
                model_success = sum(1 for v in model_results.values() if v)
                elo_success = sum(1 for v in elo_results.values() if v)
                logger.info(
                    f"Sync complete: {model_success}/{len(model_results)} models, "
                    f"{elo_success}/{len(elo_results)} ELO syncs"
                )

            except Exception as e:
                logger.error(f"Sync daemon error: {e}")

            await asyncio.sleep(interval)


def get_sync_manager() -> DataSyncManager:
    """Get or create the global sync manager instance."""
    return DataSyncManager()


async def sync_all_now():
    """Perform immediate sync of all critical data."""
    manager = get_sync_manager()

    print("Syncing best models...")
    model_results = await manager.sync_best_models()
    print(f"  Models: {sum(1 for v in model_results.values() if v)}/{len(model_results)} succeeded")

    print("Syncing ELO database...")
    elo_results = await manager.sync_elo_database()
    print(f"  ELO: {sum(1 for v in elo_results.values() if v)}/{len(elo_results)} succeeded")

    return model_results, elo_results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Data Synchronization Manager")
    parser.add_argument("--sync", action="store_true", help="Run immediate sync")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300, help="Sync interval (seconds)")
    parser.add_argument("--collect", type=str, help="Collect databases matching pattern")
    args = parser.parse_args()

    if args.sync:
        asyncio.run(sync_all_now())
    elif args.daemon:
        manager = get_sync_manager()
        asyncio.run(manager.run_sync_daemon(args.interval))
    elif args.collect:
        manager = get_sync_manager()
        collected = asyncio.run(manager.collect_training_data(args.collect))
        print(f"Collected {len(collected)} databases")
    else:
        parser.print_help()
