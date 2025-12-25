"""Cluster Data Sync Daemon - Ensures game data is available on all eligible nodes.

This daemon runs on the leader node and pushes game databases to all cluster nodes
that have adequate storage, excluding development machines like mac-studio.

Features:
- Push-based sync (proactive, not pull-based)
- Filters by available disk space (MIN_DISK_FREE_GB)
- Excludes specified nodes (EXCLUDED_NODES)
- Both event-driven and periodic sync
- Integrates with existing transport infrastructure (SSH/rsync, aria2, P2P)

Usage:
    # Via DaemonManager
    manager = get_daemon_manager()
    await manager.start(DaemonType.CLUSTER_DATA_SYNC)

    # Via launch script
    python scripts/launch_daemons.py --daemon cluster_data_sync
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Nodes that should NEVER receive synced data (dev machines, low storage)
EXCLUDED_NODES: frozenset[str] = frozenset({
    "mac-studio",      # Local dev machine - don't fill disk
    "mbp-16gb",        # Low storage laptop
    "mbp-64gb",        # Dev machine laptop
    "aws-proxy",       # Relay-only node
})

# Minimum free disk space (GB) to be eligible for sync
MIN_DISK_FREE_GB = 50

# Sync interval (seconds) - 5 minutes
SYNC_INTERVAL_SECONDS = 300

# Bandwidth limit per transfer (KB/s) - 20 MB/s
SYNC_BANDWIDTH_LIMIT_KBPS = 20_000

# Maximum concurrent syncs
MAX_CONCURRENT_SYNCS = 3

# Database patterns to sync
SYNC_DB_PATTERNS: list[str] = [
    "canonical_*.db",
    "gumbel_*.db",
    "selfplay_*.db",
    "synced/*.db",
]

# High-priority configs (sync these first)
HIGH_PRIORITY_CONFIGS: frozenset[str] = frozenset({
    "square8_2p", "hex8_2p", "hex8_3p", "hex8_4p"
})

# P2P status endpoint
P2P_STATUS_URL = "http://localhost:8770/status"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SyncTarget:
    """A node eligible to receive synced data."""
    node_id: str
    host: str
    disk_free_gb: float
    is_nfs: bool = False  # Lambda nodes share NFS, skip rsync between them


@dataclass
class SyncResult:
    """Result of a sync operation."""
    source: str
    target: str
    success: bool
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


# =============================================================================
# Utility Functions
# =============================================================================

def get_p2p_status() -> dict[str, Any]:
    """Get cluster status from P2P orchestrator."""
    try:
        req = Request(P2P_STATUS_URL, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"Failed to get P2P status: {e}")
        return {}


def get_sync_targets() -> list[SyncTarget]:
    """Get nodes eligible to receive synced data.

    Filters:
    - Not in EXCLUDED_NODES
    - Has at least MIN_DISK_FREE_GB free
    - Not retired
    - Is reachable (recent heartbeat)
    """
    status = get_p2p_status()
    if not status:
        return []

    targets = []

    # Check self node
    self_info = status.get("self", {})
    self_node_id = self_info.get("node_id", "")

    # Check peers
    peers = status.get("peers", {})
    for node_id, info in peers.items():
        # Skip excluded nodes
        if node_id in EXCLUDED_NODES:
            logger.debug(f"Skipping excluded node: {node_id}")
            continue

        # Skip retired nodes
        if info.get("retired", False):
            logger.debug(f"Skipping retired node: {node_id}")
            continue

        # Check disk space
        disk_free = info.get("disk_free_gb", 0)
        if disk_free < MIN_DISK_FREE_GB:
            logger.debug(f"Skipping {node_id}: only {disk_free:.1f}GB free")
            continue

        # Check for stale heartbeat (>5 min old)
        last_heartbeat = info.get("last_heartbeat", 0)
        if time.time() - last_heartbeat > 300:
            logger.debug(f"Skipping {node_id}: stale heartbeat")
            continue

        # Determine if NFS-connected (Lambda nodes share storage)
        is_nfs = info.get("nfs_accessible", False)

        # Get host address
        host = info.get("host", "")
        if not host:
            continue

        targets.append(SyncTarget(
            node_id=node_id,
            host=host,
            disk_free_gb=disk_free,
            is_nfs=is_nfs,
        ))

    # Sort by disk space (push to nodes with most space first)
    targets.sort(key=lambda t: t.disk_free_gb, reverse=True)

    logger.info(f"Found {len(targets)} sync targets with adequate storage")
    return targets


def discover_local_databases() -> list[Path]:
    """Find all game databases on this node that should be synced."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "games"
    if not data_dir.exists():
        return []

    databases = []
    for pattern in SYNC_DB_PATTERNS:
        databases.extend(data_dir.glob(pattern))

    # Filter out empty databases
    databases = [db for db in databases if db.stat().st_size > 1024]

    # Sort by priority (high-priority configs first)
    def priority_key(path: Path) -> tuple[int, str]:
        name = path.stem
        for config in HIGH_PRIORITY_CONFIGS:
            if config in name:
                return (0, name)
        return (1, name)

    databases.sort(key=priority_key)

    logger.info(f"Found {len(databases)} databases to sync")
    return databases


async def sync_to_target(source: Path, target: SyncTarget) -> SyncResult:
    """Push a database to a target node using rsync.

    For NFS-connected nodes (Lambda cluster), this is a no-op since they
    share storage. For other nodes, uses rsync over SSH.
    """
    start_time = time.time()

    # NFS optimization: Lambda nodes share storage, no sync needed
    if target.is_nfs:
        logger.debug(f"Skipping sync to {target.node_id}: NFS-connected")
        return SyncResult(
            source=str(source),
            target=target.node_id,
            success=True,
            bytes_transferred=0,
            duration_seconds=0,
        )

    # Build rsync command
    target_path = f"ubuntu@{target.host}:~/ringrift/ai-service/data/games/synced/"
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        f"--bwlimit={SYNC_BANDWIDTH_LIMIT_KBPS}",
        "--timeout=300",
        "-e", "ssh -i ~/.ssh/id_cluster -o StrictHostKeyChecking=no -o ConnectTimeout=10",
        str(source),
        target_path,
    ]

    try:
        logger.info(f"Syncing {source.name} to {target.node_id}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        if proc.returncode == 0:
            # Parse rsync output for bytes transferred
            bytes_transferred = source.stat().st_size  # Approximate
            logger.info(
                f"Synced {source.name} to {target.node_id} in {duration:.1f}s"
            )
            return SyncResult(
                source=str(source),
                target=target.node_id,
                success=True,
                bytes_transferred=bytes_transferred,
                duration_seconds=duration,
            )
        else:
            error = stderr.decode().strip() if stderr else "Unknown error"
            logger.warning(f"Sync failed to {target.node_id}: {error}")
            return SyncResult(
                source=str(source),
                target=target.node_id,
                success=False,
                duration_seconds=duration,
                error=error,
            )

    except asyncio.TimeoutError:
        logger.error(f"Sync to {target.node_id} timed out")
        return SyncResult(
            source=str(source),
            target=target.node_id,
            success=False,
            duration_seconds=time.time() - start_time,
            error="Timeout",
        )
    except Exception as e:
        logger.error(f"Sync to {target.node_id} error: {e}")
        return SyncResult(
            source=str(source),
            target=target.node_id,
            success=False,
            duration_seconds=time.time() - start_time,
            error=str(e),
        )


# =============================================================================
# Main Daemon Class
# =============================================================================

class ClusterDataSyncDaemon:
    """Daemon that ensures data is available on all eligible cluster nodes.

    Runs on the leader node and periodically pushes game databases to all
    nodes with adequate storage, excluding development machines.
    """

    def __init__(self):
        self._running = False
        self._sync_count = 0
        self._last_sync_time: float = 0.0
        self._last_sync_results: list[SyncResult] = []

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            "running": self._running,
            "sync_count": self._sync_count,
            "last_sync_time": self._last_sync_time,
            "last_sync_results": [
                {
                    "source": r.source,
                    "target": r.target,
                    "success": r.success,
                    "bytes": r.bytes_transferred,
                    "duration": r.duration_seconds,
                    "error": r.error,
                }
                for r in self._last_sync_results
            ],
        }

    async def run(self) -> None:
        """Main daemon loop."""
        self._running = True
        logger.info("ClusterDataSyncDaemon started")

        # Initial sync on startup
        await self._sync_cycle()

        # Periodic sync loop
        while self._running:
            try:
                await asyncio.sleep(SYNC_INTERVAL_SECONDS)
                if self._running:
                    await self._sync_cycle()
            except asyncio.CancelledError:
                logger.info("ClusterDataSyncDaemon cancelled")
                break
            except Exception as e:
                logger.error(f"Sync cycle error: {e}")
                await asyncio.sleep(60)  # Back off on error

        self._running = False
        logger.info("ClusterDataSyncDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

    async def _sync_cycle(self) -> None:
        """One sync cycle - push data to all eligible targets."""
        logger.info("Starting sync cycle")
        self._sync_count += 1
        self._last_sync_time = time.time()

        # Get eligible targets
        targets = get_sync_targets()
        if not targets:
            logger.info("No sync targets available")
            return

        # Get databases to sync
        databases = discover_local_databases()
        if not databases:
            logger.info("No databases to sync")
            return

        # Sync each database to each target (with concurrency limit)
        results: list[SyncResult] = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYNCS)

        async def sync_with_limit(db: Path, target: SyncTarget) -> SyncResult:
            async with semaphore:
                return await sync_to_target(db, target)

        # Create all sync tasks
        tasks = []
        for db in databases:
            for target in targets:
                tasks.append(sync_with_limit(db, target))

        # Execute concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            self._last_sync_results = [
                r for r in results
                if isinstance(r, SyncResult)
            ]

            # Log summary
            successful = sum(1 for r in self._last_sync_results if r.success)
            failed = len(self._last_sync_results) - successful
            logger.info(f"Sync cycle complete: {successful} successful, {failed} failed")

    async def trigger_sync(self) -> None:
        """Trigger an immediate sync (called from event handlers)."""
        if self._running:
            await self._sync_cycle()


# =============================================================================
# Singleton Access
# =============================================================================

_daemon_instance: ClusterDataSyncDaemon | None = None


def get_cluster_data_sync_daemon() -> ClusterDataSyncDaemon:
    """Get the singleton daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = ClusterDataSyncDaemon()
    return _daemon_instance
