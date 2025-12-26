"""Cluster Data Sync Daemon - Ensures game data is available on all eligible nodes.

This daemon runs on the leader node and pushes game databases to all cluster nodes
that have adequate storage, excluding development machines like mac-studio.

Features:
- Push-based sync (proactive, not pull-based)
- Filters by available disk space (via DaemonExclusionPolicy)
- Excludes specified nodes (via DaemonExclusionPolicy)
- Both event-driven and periodic sync
- Integrates with existing transport infrastructure (SSH/rsync, aria2, P2P)

Note: As of December 2025, exclusion configuration is loaded from
coordinator_config.DaemonExclusionPolicy which reads from:
- unified_loop.yaml (auto_sync.exclude_hosts, data_aggregation.excluded_nodes)
- distributed_hosts.yaml (nfs_nodes, retired)

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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from app.config.ports import get_p2p_status_url

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Import unified exclusion policy (December 2025)
# This replaces the hardcoded EXCLUDED_NODES frozenset
from app.coordination.coordinator_config import get_exclusion_policy
from app.coordination.sync_constants import SyncResult

# Legacy alias for backwards compatibility - do not use directly
# Use get_exclusion_policy().excluded_nodes instead
EXCLUDED_NODES: frozenset[str] = frozenset()  # Deprecated - loaded dynamically

# Legacy constant for backwards compatibility - do not use directly
# Use get_exclusion_policy().min_disk_free_gb instead
MIN_DISK_FREE_GB = 50  # Deprecated - loaded dynamically

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

# P2P status endpoint (uses centralized port configuration)
P2P_STATUS_URL = get_p2p_status_url()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EligibleSyncNode:
    """A node eligible to receive synced data.

    Note: This is different from sync_constants.SyncTarget which is for SSH connection details.
    This class tracks node eligibility and capacity for sync operations.
    """
    node_id: str
    host: str
    disk_free_gb: float
    is_nfs: bool = False  # Lambda nodes share NFS, skip rsync between them


# SyncResult is now imported from sync_constants


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


def get_sync_targets() -> list["EligibleSyncNode"]:
    """Get nodes eligible to receive synced data.

    Uses DaemonExclusionPolicy from coordinator_config for filtering.

    Filters:
    - Not excluded by policy (should_exclude())
    - Has sufficient free disk space (policy.min_disk_free_gb)
    - Not retired
    - Is reachable (recent heartbeat)
    """
    status = get_p2p_status()
    if not status:
        return []

    targets = []

    # Get unified exclusion policy
    exclusion_policy = get_exclusion_policy()

    # Check peers
    peers = status.get("peers", {})
    for node_id, info in peers.items():
        # Skip excluded nodes (using unified policy)
        if exclusion_policy.should_exclude(node_id):
            logger.debug(f"Skipping excluded node: {node_id}")
            continue

        # Skip retired nodes (also checked by policy, but P2P may report this too)
        if info.get("retired", False):
            logger.debug(f"Skipping retired node: {node_id}")
            continue

        # Check disk space (using policy threshold)
        disk_free = info.get("disk_free_gb", 0)
        if disk_free < exclusion_policy.min_disk_free_gb:
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

        targets.append(EligibleSyncNode(
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


async def sync_to_target(source: Path, target: EligibleSyncNode) -> SyncResult:
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

        async def sync_with_limit(db: Path, target: EligibleSyncNode) -> SyncResult:
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
# Training Node Watcher
# =============================================================================

class TrainingNodeWatcher:
    """Detects active training and triggers priority sync.

    This class monitors for training activity across the cluster and
    ensures training nodes have fresh data before training starts.

    Features:
    - Detects training node via process monitoring
    - Triggers priority sync when training detected
    - Integrates with freshness checker for pre-training validation
    """

    def __init__(self):
        self._running = False
        self._watch_task: asyncio.Task | None = None
        self._training_nodes: set[str] = set()
        self._last_check_time: float = 0.0

    async def start(self, check_interval: float = 30.0) -> None:
        """Start watching for training activity.

        Args:
            check_interval: Seconds between checks
        """
        self._running = True
        self._watch_task = asyncio.create_task(
            self._watch_loop(check_interval)
        )
        logger.info("TrainingNodeWatcher started")

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("TrainingNodeWatcher stopped")

    async def _watch_loop(self, check_interval: float) -> None:
        """Main watch loop."""
        while self._running:
            try:
                await self._check_training_activity()
            except Exception as e:
                logger.error(f"Training watch error: {e}")

            await asyncio.sleep(check_interval)

    async def _check_training_activity(self) -> None:
        """Check for training activity across cluster."""
        self._last_check_time = time.time()

        # Get cluster status
        status = get_p2p_status()
        if not status:
            return

        training_detected = set()

        # Check peers for training activity
        peers = status.get("peers", {})
        for node_id, info in peers.items():
            # Check for training process indicators
            running_jobs = info.get("running_jobs", [])
            for job in running_jobs:
                job_type = job.get("type", "")
                if "train" in job_type.lower():
                    training_detected.add(node_id)
                    break

            # Also check process list if available
            processes = info.get("processes", [])
            for proc in processes:
                if "train" in str(proc).lower():
                    training_detected.add(node_id)
                    break

        # Detect new training nodes
        new_training = training_detected - self._training_nodes
        if new_training:
            logger.info(f"New training detected on nodes: {new_training}")
            await self._on_training_detected(new_training)

        self._training_nodes = training_detected

    async def _on_training_detected(self, nodes: set[str]) -> None:
        """Handle detection of new training activity.

        Triggers priority sync to ensure training nodes have fresh data.
        """
        try:
            daemon = get_cluster_data_sync_daemon()
            if daemon.is_running:
                logger.info(f"Triggering priority sync for training nodes: {nodes}")
                await daemon.trigger_sync()
        except Exception as e:
            logger.error(f"Failed to trigger sync for training: {e}")
            # Emit DATA_SYNC_FAILED event
            try:
                from app.distributed.data_events import emit_data_sync_failed
                from app.core.async_context import fire_and_forget
                fire_and_forget(
                    emit_data_sync_failed(
                        host=",".join(nodes) if nodes else "unknown",
                        error=str(e),
                        source="TrainingNodeWatcher.trigger_sync_for_training_nodes",
                    ),
                    error_callback=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                )
            except Exception as emit_err:
                logger.debug(f"[DataSync] Event emission failed: {emit_err}")

    def detect_local_training(self) -> bool:
        """Check if training is running locally.

        Returns:
            True if local training detected
        """
        import subprocess

        try:
            # Check for training processes
            result = subprocess.run(
                ["pgrep", "-f", "app.training.train"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"Could not check for local training process: {e}")
            return False

    async def trigger_priority_sync(self, node_id: str) -> bool:
        """Trigger priority sync for a specific training node.

        Args:
            node_id: Node that needs priority sync

        Returns:
            True if sync was triggered
        """
        try:
            daemon = get_cluster_data_sync_daemon()
            if daemon.is_running:
                logger.info(f"Priority sync requested for training node: {node_id}")
                await daemon.trigger_sync()
                return True
            return False
        except Exception as e:
            logger.error(f"Priority sync failed: {e}")
            # Emit DATA_SYNC_FAILED event
            try:
                from app.distributed.data_events import emit_data_sync_failed
                from app.core.async_context import fire_and_forget
                fire_and_forget(
                    emit_data_sync_failed(
                        host=node_id,
                        error=str(e),
                        source="TrainingNodeWatcher.trigger_priority_sync",
                    ),
                    error_callback=lambda exc: logger.debug(f"Failed to emit sync failed: {exc}"),
                )
            except Exception as emit_err:
                logger.debug(f"[DataSync] Priority sync event emission failed: {emit_err}")
            return False

    def get_training_nodes(self) -> set[str]:
        """Get set of currently detected training nodes."""
        return self._training_nodes.copy()

    @property
    def stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "running": self._running,
            "training_nodes": list(self._training_nodes),
            "last_check_time": self._last_check_time,
        }


# Singleton watcher
_training_watcher: TrainingNodeWatcher | None = None


def get_training_node_watcher() -> TrainingNodeWatcher:
    """Get the singleton TrainingNodeWatcher instance."""
    global _training_watcher
    if _training_watcher is None:
        _training_watcher = TrainingNodeWatcher()
    return _training_watcher


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
