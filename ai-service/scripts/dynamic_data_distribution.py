#!/usr/bin/env python3
"""Dynamic Data Distribution Daemon for RingRift AI Training.

This script orchestrates data distribution from OWC external drive (via HTTP)
to training nodes based on their current needs and available capacity.

Features:
- Config-driven: Loads node definitions from distributed_hosts.yaml
- Capacity-aware: Only pushes to nodes with adequate disk space
- Incremental: Skips files that already exist on target nodes
- Priority-based: Distributes to highest-priority nodes first
- Metrics: Tracks distribution statistics and emits events
- Graceful: Handles shutdown signals properly

Usage:
    # One-time distribution
    python scripts/dynamic_data_distribution.py --once

    # Run as daemon (default 5 min interval)
    python scripts/dynamic_data_distribution.py --daemon

    # Run as daemon with custom interval
    python scripts/dynamic_data_distribution.py --daemon --interval 600

    # Check status only
    python scripts/dynamic_data_distribution.py --status

    # Dry run (show what would be distributed)
    python scripts/dynamic_data_distribution.py --once --dry-run

Environment:
    DATA_SOURCE_URL: HTTP URL for OWC data server (default: http://100.107.168.125:8780)
    MIN_FREE_DISK_GB: Minimum free disk to receive data (default: 50)
    MAX_CONCURRENT_DOWNLOADS: Parallel downloads per node (default: 3)
    DISTRIBUTION_LOG_FILE: Log file path (default: logs/distribution.log)

December 2025: Created for automated training data distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configuration from environment
DATA_SOURCE_URL = os.getenv("DATA_SOURCE_URL", "http://100.107.168.125:8780")
MIN_FREE_DISK_GB = float(os.getenv("MIN_FREE_DISK_GB", "50"))
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
LOG_FILE = os.getenv("DISTRIBUTION_LOG_FILE", str(ROOT / "logs" / "distribution.log"))

# Ensure log directory exists
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("distribution")

# Data source paths on OWC
DATA_SOURCES = {
    "canonical_data": {"path": "/canonical_data/", "dest": "training", "extensions": [".npz"]},
    "canonical_games": {"path": "/canonical_games/", "dest": "games", "extensions": [".db"]},
    "cluster_games": {"path": "/cluster_games/", "dest": "games", "extensions": [".db"]},
    "canonical_models": {"path": "/canonical_models/", "dest": "models", "extensions": [".pth", ".pt"]},
}


@dataclass
class NodeConfig:
    """Configuration for a training node."""
    name: str
    ssh_target: str  # user@host
    ssh_key: str     # Path to SSH key
    data_path: str   # Remote path to ai-service
    priority: int = 10  # Lower = higher priority
    enabled: bool = True

    @classmethod
    def from_yaml_entry(cls, name: str, entry: dict) -> "NodeConfig":
        """Create from distributed_hosts.yaml entry."""
        # Handle various SSH formats
        ssh = entry.get("ssh", "")
        if not ssh:
            user = entry.get("user", "ubuntu")
            host = entry.get("host", entry.get("tailscale_ip", entry.get("public_ip", "")))
            ssh = f"{user}@{host}" if host else ""

        return cls(
            name=name,
            ssh_target=ssh,
            ssh_key=entry.get("ssh_key", entry.get("key", "~/.ssh/id_cluster")),
            data_path=entry.get("data_path", entry.get("path", "~/ringrift/ai-service")),
            priority=entry.get("priority", 10),
            enabled=entry.get("enabled", True),
        )


@dataclass
class NodeStatus:
    """Runtime status of a training node."""
    name: str
    reachable: bool = False
    disk_free_gb: float = 0.0
    disk_used_percent: float = 100.0
    game_count: int = 0
    npz_count: int = 0
    model_count: int = 0
    existing_files: set = field(default_factory=set)
    last_check: float = 0.0
    error: str = ""


@dataclass
class DistributionStats:
    """Statistics for a distribution cycle."""
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    nodes_checked: int = 0
    nodes_eligible: int = 0
    files_distributed: int = 0
    files_skipped: int = 0
    bytes_transferred: int = 0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.cycle_end - self.cycle_start if self.cycle_end else 0,
            "nodes_checked": self.nodes_checked,
            "nodes_eligible": self.nodes_eligible,
            "files_distributed": self.files_distributed,
            "files_skipped": self.files_skipped,
            "bytes_transferred": self.bytes_transferred,
            "error_count": len(self.errors),
        }


class DataDistributionDaemon:
    """Daemon for distributing training data to cluster nodes."""

    def __init__(
        self,
        data_source_url: str = DATA_SOURCE_URL,
        min_free_disk_gb: float = MIN_FREE_DISK_GB,
        max_concurrent: int = MAX_CONCURRENT_DOWNLOADS,
        dry_run: bool = False,
    ):
        self.data_source_url = data_source_url.rstrip("/")
        self.min_free_disk_gb = min_free_disk_gb
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run

        self._running = False
        self._nodes: dict[str, NodeConfig] = {}
        self._stats_history: list[DistributionStats] = []
        self._total_files_distributed = 0
        self._total_bytes_transferred = 0
        self._start_time = 0.0

        # Load node configurations
        self._load_node_configs()

    def _load_node_configs(self) -> None:
        """Load node configurations from distributed_hosts.yaml or fallback to defaults."""
        config_path = ROOT / "config" / "distributed_hosts.yaml"

        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Look for priority_hosts or training_targets
                hosts = config.get("priority_hosts", [])
                if not hosts:
                    hosts = config.get("training_targets", [])
                if not hosts:
                    hosts = config.get("hosts", {})

                if isinstance(hosts, list):
                    for i, entry in enumerate(hosts):
                        if isinstance(entry, dict):
                            name = entry.get("name", entry.get("host_id", f"node-{i}"))
                            node = NodeConfig.from_yaml_entry(name, entry)
                            if node.ssh_target and node.enabled:
                                self._nodes[name] = node
                elif isinstance(hosts, dict):
                    for name, entry in hosts.items():
                        if isinstance(entry, dict):
                            node = NodeConfig.from_yaml_entry(name, entry)
                            if node.ssh_target and node.enabled:
                                self._nodes[name] = node

                if self._nodes:
                    logger.info(f"Loaded {len(self._nodes)} nodes from {config_path}")
                    return

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Fallback to hardcoded defaults
        logger.info("Using default node configurations")
        self._nodes = {
            "nebius-h100-3": NodeConfig(
                name="nebius-h100-3",
                ssh_target="ubuntu@89.169.110.128",
                ssh_key="~/.ssh/id_cluster",
                data_path="~/ringrift/ai-service",
                priority=1,
            ),
            "nebius-h100-1": NodeConfig(
                name="nebius-h100-1",
                ssh_target="ubuntu@89.169.111.139",
                ssh_key="~/.ssh/id_cluster",
                data_path="~/ringrift/ai-service",
                priority=2,
            ),
            "vultr-a100": NodeConfig(
                name="vultr-a100",
                ssh_target="root@208.167.249.164",
                ssh_key="~/.ssh/id_transfer",
                data_path="/root/ringrift/ai-service",
                priority=3,
            ),
        }

    async def _run_ssh_command(
        self,
        node: NodeConfig,
        command: str,
        timeout: int = 30
    ) -> tuple[int, str, str]:
        """Run SSH command on a node."""
        ssh_key = Path(node.ssh_key).expanduser()
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-i", str(ssh_key),
            node.ssh_target,
            command,
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
            return 1, "", "SSH timeout"
        except Exception as e:
            return 1, "", str(e)

    async def _check_node_status(self, node: NodeConfig) -> NodeStatus:
        """Check status of a training node."""
        status = NodeStatus(name=node.name, last_check=time.time())

        # Check disk space
        code, stdout, stderr = await self._run_ssh_command(
            node,
            "df -BG / | tail -1 | awk '{print $4, $5}'"
        )

        if code != 0:
            status.error = stderr or "Connection failed"
            return status

        status.reachable = True
        try:
            parts = stdout.strip().split()
            status.disk_free_gb = float(parts[0].rstrip("G"))
            status.disk_used_percent = float(parts[1].rstrip("%"))
        except (IndexError, ValueError):
            pass

        # Get existing files in one command
        code, stdout, _ = await self._run_ssh_command(
            node,
            f"find {node.data_path}/data -name '*.db' -o -name '*.npz' -o -name '*.pth' 2>/dev/null | xargs -I{{}} basename {{}}"
        )
        if code == 0:
            status.existing_files = set(stdout.strip().split("\n")) if stdout.strip() else set()
            status.game_count = sum(1 for f in status.existing_files if f.endswith(".db"))
            status.npz_count = sum(1 for f in status.existing_files if f.endswith(".npz"))
            status.model_count = sum(1 for f in status.existing_files if f.endswith(".pth"))

        return status

    async def _get_available_files(self, source_key: str) -> list[dict]:
        """Get list of available files from HTTP server."""
        source = DATA_SOURCES.get(source_key)
        if not source:
            return []

        url = self.data_source_url + source["path"]

        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                html = response.read().decode()

            files = []
            extensions = "|".join(ext.replace(".", r"\.") for ext in source["extensions"])
            pattern = rf'href="([^"]+(?:{extensions}))"'

            for match in re.finditer(pattern, html):
                filename = match.group(1)
                files.append({
                    "name": filename,
                    "url": url + filename,
                    "dest": source["dest"],
                })

            return files

        except Exception as e:
            logger.error(f"Failed to list {source_key} from {url}: {e}")
            return []

    async def _download_file(
        self,
        node: NodeConfig,
        file_info: dict,
        stats: DistributionStats,
    ) -> bool:
        """Download a file to a node via wget."""
        dest_path = f"{node.data_path}/data/{file_info['dest']}/"
        filename = file_info["name"]

        if self.dry_run:
            logger.info(f"  [DRY-RUN] Would download {filename} to {node.name}")
            return True

        cmd = f"mkdir -p {dest_path} && wget -q -O {dest_path}{filename} '{file_info['url']}'"
        code, _, stderr = await self._run_ssh_command(node, cmd, timeout=300)

        if code == 0:
            logger.info(f"  Downloaded {filename} to {node.name}")
            stats.files_distributed += 1
            return True
        else:
            logger.error(f"  Failed to download {filename} to {node.name}: {stderr}")
            stats.errors.append(f"{node.name}:{filename}:{stderr[:100]}")
            return False

    async def _distribute_to_node(
        self,
        node: NodeConfig,
        status: NodeStatus,
        all_files: list[dict],
        stats: DistributionStats,
    ) -> int:
        """Distribute files to a single node."""
        # Filter to files not already on node
        files_to_send = [
            f for f in all_files
            if f["name"] not in status.existing_files
        ]

        skipped = len(all_files) - len(files_to_send)
        stats.files_skipped += skipped

        if not files_to_send:
            logger.info(f"  {node.name}: All {len(all_files)} files already present")
            return 0

        logger.info(f"  {node.name}: Distributing {len(files_to_send)} files ({skipped} skipped)")

        # Download with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_limit(file_info):
            async with semaphore:
                return await self._download_file(node, file_info, stats)

        results = await asyncio.gather(*[
            download_with_limit(f) for f in files_to_send
        ])

        return sum(1 for r in results if r)

    async def run_distribution_cycle(self) -> DistributionStats:
        """Run a full distribution cycle."""
        stats = DistributionStats(cycle_start=time.time())

        logger.info("=" * 60)
        logger.info("Starting distribution cycle")
        logger.info(f"Data source: {self.data_source_url}")

        # Check all node statuses
        logger.info("\nChecking node statuses...")
        node_statuses: dict[str, NodeStatus] = {}

        for name, config in sorted(self._nodes.items(), key=lambda x: x[1].priority):
            status = await self._check_node_status(config)
            node_statuses[name] = status
            stats.nodes_checked += 1

            if status.reachable:
                logger.info(
                    f"  {name}: OK - {status.disk_free_gb:.0f}GB free, "
                    f"{status.npz_count} NPZ, {status.game_count} DBs"
                )
            else:
                logger.warning(f"  {name}: UNREACHABLE - {status.error}")

        # Filter to eligible nodes
        eligible = {
            name: status for name, status in node_statuses.items()
            if status.reachable and status.disk_free_gb >= self.min_free_disk_gb
        }
        stats.nodes_eligible = len(eligible)

        if not eligible:
            logger.warning("No eligible nodes found for distribution")
            stats.cycle_end = time.time()
            return stats

        logger.info(f"\n{len(eligible)} eligible nodes for distribution")

        # Gather all available files
        all_files: list[dict] = []
        for source_key in DATA_SOURCES:
            files = await self._get_available_files(source_key)
            all_files.extend(files)
            if files:
                logger.info(f"Found {len(files)} files in {source_key}")

        if not all_files:
            logger.warning("No files available for distribution")
            stats.cycle_end = time.time()
            return stats

        # Distribute to each eligible node (in priority order)
        logger.info("\nDistributing files...")
        for name in sorted(eligible.keys(), key=lambda n: self._nodes[n].priority):
            node = self._nodes[name]
            status = eligible[name]
            await self._distribute_to_node(node, status, all_files, stats)

        stats.cycle_end = time.time()
        self._stats_history.append(stats)
        self._total_files_distributed += stats.files_distributed

        duration = stats.cycle_end - stats.cycle_start
        logger.info(
            f"\nCycle complete in {duration:.1f}s: "
            f"{stats.files_distributed} distributed, {stats.files_skipped} skipped, "
            f"{len(stats.errors)} errors"
        )

        # Emit event if available
        await self._emit_distribution_event(stats)

        return stats

    async def _emit_distribution_event(self, stats: DistributionStats) -> None:
        """Emit distribution completed event."""
        try:
            from app.coordination.event_router import emit
            await emit(
                event_type="DISTRIBUTION_COMPLETED",
                data={
                    **stats.to_dict(),
                    "timestamp": time.time(),
                    "source_url": self.data_source_url,
                },
            )
        except ImportError:
            pass  # event_router not available
        except Exception as e:
            logger.debug(f"Failed to emit event: {e}")

    async def run_daemon(self, interval: int = 300) -> None:
        """Run as a daemon, distributing data periodically."""
        self._running = True
        self._start_time = time.time()

        logger.info("=" * 60)
        logger.info("Starting Data Distribution Daemon")
        logger.info(f"  Interval: {interval}s")
        logger.info(f"  Nodes: {len(self._nodes)}")
        logger.info(f"  Min free disk: {self.min_free_disk_gb}GB")
        logger.info(f"  Data source: {self.data_source_url}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info("=" * 60)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown())

        while self._running:
            try:
                await self.run_distribution_cycle()
            except Exception as e:
                logger.error(f"Distribution cycle failed: {e}", exc_info=True)

            if self._running:
                logger.info(f"Next cycle in {interval}s...")
                await asyncio.sleep(interval)

        logger.info("Daemon stopped")

    def _shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False

    async def show_status(self) -> None:
        """Show current status of all nodes."""
        print("=" * 60)
        print("Training Node Status")
        print("=" * 60)
        print(f"Data Source: {self.data_source_url}")
        print(f"Min Free Disk: {self.min_free_disk_gb}GB")
        print()

        for name, config in sorted(self._nodes.items(), key=lambda x: x[1].priority):
            status = await self._check_node_status(config)

            state = "OK" if status.reachable else "UNREACHABLE"
            eligible = "✓" if status.disk_free_gb >= self.min_free_disk_gb else "✗"

            print(f"{name} (priority {config.priority}):")
            print(f"  Status: {state}")
            if status.reachable:
                print(f"  Disk: {status.disk_free_gb:.0f}GB free ({status.disk_used_percent:.0f}% used)")
                print(f"  Data: {status.npz_count} NPZ, {status.game_count} DBs, {status.model_count} models")
                print(f"  Eligible: {eligible}")
            else:
                print(f"  Error: {status.error}")
            print()

        # Check data source
        print("Data Source Status:")
        for source_key, source in DATA_SOURCES.items():
            files = await self._get_available_files(source_key)
            print(f"  {source_key}: {len(files)} files")

    def get_metrics(self) -> dict:
        """Get daemon metrics."""
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "total_files_distributed": self._total_files_distributed,
            "total_bytes_transferred": self._total_bytes_transferred,
            "cycles_completed": len(self._stats_history),
            "nodes_configured": len(self._nodes),
            "data_source_url": self.data_source_url,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic data distribution for RingRift AI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check current status
    python scripts/dynamic_data_distribution.py --status

    # Run one distribution cycle
    python scripts/dynamic_data_distribution.py --once

    # Run as daemon (every 5 minutes)
    python scripts/dynamic_data_distribution.py --daemon

    # Run as daemon with custom interval
    python scripts/dynamic_data_distribution.py --daemon --interval 600

    # Dry run (no actual downloads)
    python scripts/dynamic_data_distribution.py --once --dry-run
        """,
    )
    parser.add_argument("--once", action="store_true", help="Run one distribution cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show node status")
    parser.add_argument("--interval", type=int, default=300, help="Daemon interval in seconds (default: 300)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without downloading")
    parser.add_argument("--source-url", type=str, default=DATA_SOURCE_URL, help="HTTP URL for data source")
    parser.add_argument("--min-disk", type=float, default=MIN_FREE_DISK_GB, help="Minimum free disk GB")
    args = parser.parse_args()

    daemon = DataDistributionDaemon(
        data_source_url=args.source_url,
        min_free_disk_gb=args.min_disk,
        dry_run=args.dry_run,
    )

    if args.status:
        asyncio.run(daemon.show_status())
    elif args.daemon:
        asyncio.run(daemon.run_daemon(args.interval))
    elif args.once:
        asyncio.run(daemon.run_distribution_cycle())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
