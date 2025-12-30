"""Adaptive Resource Manager - Proactive resource management across cluster.

This module provides automatic resource monitoring and management:
1. Monitor NFS disk usage and trigger cleanup at thresholds
2. Monitor per-node disk, memory, and GPU usage
3. Pause jobs on nodes with resource pressure
4. Aggregate selfplay data from nodes to NFS
5. Integrate with the work queue to prevent overload

Usage:
    from app.coordination.adaptive_resource_manager import (
        AdaptiveResourceManager,
        get_resource_manager,
    )

    manager = get_resource_manager()
    asyncio.create_task(manager.run())
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import bandwidth limiting from cluster_config (December 2025)
try:
    from app.config.cluster_config import get_node_bandwidth_kbs
    HAS_BANDWIDTH_CONFIG = True
except ImportError:
    HAS_BANDWIDTH_CONFIG = False
    get_node_bandwidth_kbs = None


# ============================================
# Configuration (December 2025: centralized defaults)
# ============================================

# Import centralized defaults
from app.config.coordination_defaults import ResourceManagerDefaults

# Thresholds (percentage) - from centralized config
DISK_WARNING_THRESHOLD = ResourceManagerDefaults.DISK_WARNING_THRESHOLD
DISK_CRITICAL_THRESHOLD = ResourceManagerDefaults.DISK_CRITICAL_THRESHOLD
DISK_CLEANUP_THRESHOLD = ResourceManagerDefaults.DISK_CLEANUP_THRESHOLD
MEMORY_WARNING_THRESHOLD = ResourceManagerDefaults.MEMORY_WARNING_THRESHOLD
MEMORY_CRITICAL_THRESHOLD = ResourceManagerDefaults.MEMORY_CRITICAL_THRESHOLD

# Check intervals (seconds) - from centralized config
CHECK_INTERVAL = ResourceManagerDefaults.CHECK_INTERVAL  # 5 minutes
CLEANUP_COOLDOWN = ResourceManagerDefaults.CLEANUP_COOLDOWN  # 30 minutes between cleanups
AGGREGATION_INTERVAL = ResourceManagerDefaults.AGGREGATION_INTERVAL  # 10 minutes

# Cleanup settings
MIN_FILE_AGE_HOURS = 24  # Only clean files older than 24 hours
CLEANUP_BATCH_SIZE = 100  # Max files to clean per batch

# Default paths
DEFAULT_NFS_PATH = "/mnt/nfs/ringrift"
DEFAULT_DATA_PATH = "/home/ubuntu/ringrift/ai-service/data"


@dataclass
class ResourceStatus:
    """Current resource status for a node."""
    node_id: str
    timestamp: float = field(default_factory=time.time)

    # Disk
    disk_total_gb: float = 0
    disk_used_gb: float = 0
    disk_free_gb: float = 0
    disk_percent: float = 0

    # Memory
    memory_total_gb: float = 0
    memory_used_gb: float = 0
    memory_free_gb: float = 0
    memory_percent: float = 0

    # GPU (optional)
    gpu_memory_used_gb: float = 0
    gpu_memory_total_gb: float = 0
    gpu_percent: float = 0

    # Status
    is_healthy: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "disk": {
                "total_gb": self.disk_total_gb,
                "used_gb": self.disk_used_gb,
                "free_gb": self.disk_free_gb,
                "percent": self.disk_percent,
            },
            "memory": {
                "total_gb": self.memory_total_gb,
                "used_gb": self.memory_used_gb,
                "free_gb": self.memory_free_gb,
                "percent": self.memory_percent,
            },
            "gpu": {
                "used_gb": self.gpu_memory_used_gb,
                "total_gb": self.gpu_memory_total_gb,
                "percent": self.gpu_percent,
            },
            "is_healthy": self.is_healthy,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    success: bool
    files_deleted: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0


class AdaptiveResourceManager:
    """Proactive resource management across cluster."""

    def __init__(
        self,
        nfs_path: str | None = None,
        data_path: str | None = None,
        disk_threshold: float = DISK_CLEANUP_THRESHOLD,
        memory_threshold: float = MEMORY_WARNING_THRESHOLD,
    ):
        self.nfs_path = Path(nfs_path or os.environ.get("RINGRIFT_NFS_PATH", DEFAULT_NFS_PATH))
        self.data_path = Path(data_path or DEFAULT_DATA_PATH)
        self.disk_threshold = disk_threshold
        self.memory_threshold = memory_threshold

        self.running = False

        # State
        self.last_cleanup_time: float = 0
        self.last_aggregation_time: float = 0
        self.node_statuses: dict[str, ResourceStatus] = {}

        # Stats
        self.stats = {
            "cleanups_triggered": 0,
            "bytes_freed_total": 0,
            "files_deleted_total": 0,
            "aggregations_completed": 0,
            "nodes_paused": 0,
            "errors": 0,
        }

    def _get_disk_usage(self, path: Path) -> tuple[float, float, float]:
        """Get disk usage for a path. Returns (total_gb, used_gb, free_gb)."""
        try:
            stat = shutil.disk_usage(path)
            return (
                stat.total / (1024**3),
                stat.used / (1024**3),
                stat.free / (1024**3),
            )
        except OSError as e:
            # OSError covers FileNotFoundError, PermissionError, etc.
            logger.error(f"Error getting disk usage for {path}: {e}")
            return 0, 0, 0

    def _get_memory_usage(self) -> tuple[float, float, float]:
        """Get memory usage. Returns (total_gb, used_gb, free_gb)."""
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        value = int(parts[1]) / (1024**2)  # Convert KB to GB
                        meminfo[key] = value

                total = meminfo.get("MemTotal", 0)
                free = meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0) + meminfo.get("Cached", 0)
                used = total - free
                return total, used, free
        except (OSError, ValueError) as e:
            # OSError: /proc/meminfo not found (non-Linux)
            # ValueError: parsing failure
            logger.error(f"Error getting memory usage: {e}")
            return 0, 0, 0

    def _get_gpu_memory(self) -> tuple[float, float]:
        """Get GPU memory usage. Returns (used_gb, total_gb)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                total_used = 0
                total_total = 0
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        total_used += float(parts[0].strip()) / 1024  # MB to GB
                        total_total += float(parts[1].strip()) / 1024
                return total_used, total_total
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            # FileNotFoundError: nvidia-smi not installed
            # TimeoutExpired: query took too long
            # SubprocessError: execution failed
            # ValueError: float parsing failure
            logger.debug(f"GPU memory query failed: {e}")
        return 0, 0

    def get_local_status(self, node_id: str = "local") -> ResourceStatus:
        """Get resource status for the local node."""
        status = ResourceStatus(node_id=node_id)

        # Disk
        total, used, free = self._get_disk_usage(self.data_path)
        status.disk_total_gb = total
        status.disk_used_gb = used
        status.disk_free_gb = free
        status.disk_percent = (used / total * 100) if total > 0 else 0

        # Memory
        total, used, free = self._get_memory_usage()
        status.memory_total_gb = total
        status.memory_used_gb = used
        status.memory_free_gb = free
        status.memory_percent = (used / total * 100) if total > 0 else 0

        # GPU
        gpu_used, gpu_total = self._get_gpu_memory()
        status.gpu_memory_used_gb = gpu_used
        status.gpu_memory_total_gb = gpu_total
        status.gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0

        # Check thresholds
        if status.disk_percent >= DISK_CRITICAL_THRESHOLD:
            status.errors.append(f"Disk critical: {status.disk_percent:.1f}%")
            status.is_healthy = False
        elif status.disk_percent >= DISK_WARNING_THRESHOLD:
            status.warnings.append(f"Disk warning: {status.disk_percent:.1f}%")

        if status.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
            status.errors.append(f"Memory critical: {status.memory_percent:.1f}%")
            status.is_healthy = False
        elif status.memory_percent >= MEMORY_WARNING_THRESHOLD:
            status.warnings.append(f"Memory warning: {status.memory_percent:.1f}%")

        self.node_statuses[node_id] = status
        return status

    def get_nfs_status(self) -> ResourceStatus:
        """Get resource status for NFS mount."""
        status = ResourceStatus(node_id="nfs")

        if not self.nfs_path.exists():
            status.errors.append("NFS path not accessible")
            status.is_healthy = False
            return status

        total, used, free = self._get_disk_usage(self.nfs_path)
        status.disk_total_gb = total
        status.disk_used_gb = used
        status.disk_free_gb = free
        status.disk_percent = (used / total * 100) if total > 0 else 0

        if status.disk_percent >= DISK_CRITICAL_THRESHOLD:
            status.errors.append(f"NFS disk critical: {status.disk_percent:.1f}%")
            status.is_healthy = False
        elif status.disk_percent >= DISK_WARNING_THRESHOLD:
            status.warnings.append(f"NFS disk warning: {status.disk_percent:.1f}%")

        self.node_statuses["nfs"] = status
        return status

    async def cleanup_old_files(
        self,
        path: Path,
        min_age_hours: float = MIN_FILE_AGE_HOURS,
        patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> CleanupResult:
        """Clean up old files to free disk space.

        Args:
            path: Directory to clean
            min_age_hours: Only delete files older than this
            patterns: File patterns to match (e.g., ["*.jsonl", "*.tmp"])
            dry_run: If True, only report what would be deleted

        Returns:
            CleanupResult with details of the operation
        """
        result = CleanupResult(success=True)
        start_time = time.time()

        patterns = patterns or ["*.jsonl.gz", "*.jsonl", "*.tmp", "*.log"]
        min_age_seconds = min_age_hours * 3600
        now = time.time()

        try:
            files_to_delete = []

            for pattern in patterns:
                for file_path in path.rglob(pattern):
                    if not file_path.is_file():
                        continue

                    try:
                        file_age = now - file_path.stat().st_mtime
                        if file_age >= min_age_seconds:
                            files_to_delete.append(file_path)
                    except (FileNotFoundError, OSError):
                        continue

            # Sort by age (oldest first) and limit
            files_to_delete.sort(key=lambda f: f.stat().st_mtime)
            files_to_delete = files_to_delete[:CLEANUP_BATCH_SIZE]

            for file_path in files_to_delete:
                try:
                    size = file_path.stat().st_size
                    if not dry_run:
                        file_path.unlink()
                    result.files_deleted += 1
                    result.bytes_freed += size
                except OSError as e:
                    # OSError covers PermissionError, FileNotFoundError (race condition)
                    result.errors.append(f"Failed to delete {file_path}: {e}")

        except Exception as e:
            result.success = False
            result.errors.append(f"Cleanup failed: {e}")
            self.stats["errors"] += 1

        result.duration_seconds = time.time() - start_time

        if result.files_deleted > 0:
            self.stats["cleanups_triggered"] += 1
            self.stats["files_deleted_total"] += result.files_deleted
            self.stats["bytes_freed_total"] += result.bytes_freed
            self.last_cleanup_time = time.time()

            action = "Would delete" if dry_run else "Deleted"
            logger.info(
                f"{action} {result.files_deleted} files, freed {result.bytes_freed / (1024**2):.1f} MB"
            )

        return result

    async def aggregate_selfplay_data(self, source_nodes: list[str] | None = None) -> dict[str, Any]:
        """Aggregate selfplay data from nodes to NFS.

        This consolidates training data from distributed nodes to the
        central NFS storage for training.

        Args:
            source_nodes: List of node IDs to aggregate from (None = all)

        Returns:
            Aggregation result dict
        """
        result: dict[str, Any] = {
            "success": True,
            "games_aggregated": 0,
            "bytes_transferred": 0,
            "errors": [],
            "nodes_processed": [],
        }

        # Get nodes to process
        if source_nodes is None:
            source_nodes = await self._get_active_selfplay_nodes()

        if not source_nodes:
            logger.debug("No source nodes for NFS aggregation")
            return result

        # Create destination directory if needed
        nfs_selfplay_dir = self.nfs_path / "ai-service" / "data" / "games"
        nfs_selfplay_dir.mkdir(parents=True, exist_ok=True)

        # Process each node
        for node_id in source_nodes:
            try:
                node_result = await self._aggregate_from_node(node_id, nfs_selfplay_dir)
                result["nodes_processed"].append(node_id)
                result["games_aggregated"] += node_result.get("games", 0)
                result["bytes_transferred"] += node_result.get("bytes", 0)
                if node_result.get("error"):
                    result["errors"].append(f"{node_id}: {node_result['error']}")
            except Exception as e:
                logger.warning(f"Failed to aggregate from {node_id}: {e}")
                result["errors"].append(f"{node_id}: {str(e)}")

        if result["errors"]:
            result["success"] = len(result["nodes_processed"]) > 0

        self.last_aggregation_time = time.time()
        self.stats["aggregations_completed"] += 1
        self.stats["total_games_aggregated"] = self.stats.get("total_games_aggregated", 0) + result["games_aggregated"]

        if result["games_aggregated"] > 0:
            logger.info(
                f"NFS aggregation complete: {result['games_aggregated']} games, "
                f"{result['bytes_transferred'] / (1024**2):.1f} MB from {len(result['nodes_processed'])} nodes"
            )

        return result

    async def _get_active_selfplay_nodes(self) -> list[str]:
        """Get list of nodes running selfplay jobs.

        December 2025: Fixed bug - was trying to import non-existent get_cluster_monitor.
        Now uses ClusterMonitor class directly.

        December 2025 (v2): Use async version to avoid blocking the event loop.
        The sync get_cluster_status() uses concurrent.futures.as_completed() which
        blocks the asyncio event loop, causing test timeouts.
        """
        try:
            from app.coordination.cluster_status_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = await monitor.get_cluster_status_async()
            nodes: list[str] = []
            for node_id, node_info in getattr(status, "nodes", {}).items():
                has_selfplay = False
                if hasattr(node_info, "total_games"):
                    has_selfplay = getattr(node_info, "total_games", 0) > 0
                elif isinstance(node_info, dict):
                    has_selfplay = node_info.get("has_selfplay_data", False)
                    if not has_selfplay:
                        has_selfplay = node_info.get("total_games", 0) > 0
                if has_selfplay:
                    nodes.append(node_id)
            return nodes
        except ImportError:
            return []
        except Exception as e:
            logger.debug(f"Failed to get active selfplay nodes: {e}")
            return []

    async def _aggregate_from_node(
        self, node_id: str, dest_dir: Path
    ) -> dict[str, Any]:
        """Aggregate selfplay data from a single node.

        Args:
            node_id: Node identifier
            dest_dir: Destination directory on NFS

        Returns:
            Dict with keys: games, bytes, error
        """
        result: dict[str, Any] = {"games": 0, "bytes": 0, "error": None}

        try:
            from app.core.ssh import get_ssh_client

            client = get_ssh_client(node_id)

            # Find database files on the node
            find_cmd = (
                "find ~/ringrift/ai-service/data/games -name '*.db' "
                "-mmin +5 -type f 2>/dev/null || true"
            )
            find_result = client.run(find_cmd, timeout=30)

            if find_result.returncode != 0 or not find_result.stdout.strip():
                return result

            db_files = find_result.stdout.strip().split("\n")
            db_files = [f for f in db_files if f and "canonical" not in f]

            if not db_files:
                return result

            # Rsync database files to NFS
            for db_file in db_files[:10]:  # Limit batch size
                db_name = Path(db_file).name
                dest_path = dest_dir / f"{node_id}_{db_name}"

                # Skip if already exists
                if dest_path.exists():
                    continue

                # Use rsync for transfer with bandwidth limiting (December 2025)
                bwlimit_args = []
                if HAS_BANDWIDTH_CONFIG and get_node_bandwidth_kbs:
                    try:
                        bwlimit_kbs = get_node_bandwidth_kbs(client.host)
                        if bwlimit_kbs > 0:
                            bwlimit_args = [f"--bwlimit={bwlimit_kbs}"]
                    except (KeyError, ValueError):
                        pass

                rsync_cmd = [
                    "rsync", "-az", "--timeout=60",
                    *bwlimit_args,
                    f"{client.user}@{client.host}:{db_file}",
                    str(dest_path),
                ]

                proc = await asyncio.create_subprocess_exec(
                    *rsync_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                if proc.returncode == 0 and dest_path.exists():
                    result["games"] += 1
                    result["bytes"] += dest_path.stat().st_size
                else:
                    logger.debug(f"Rsync failed for {db_file}: {stderr.decode()}")

            return result

        except ImportError:
            result["error"] = "SSH module not available"
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    async def check_and_cleanup(self) -> dict[str, Any]:
        """Check resource levels and trigger cleanup if needed."""
        results = {
            "nfs_status": None,
            "local_status": None,
            "cleanup_triggered": False,
            "cleanup_result": None,
        }

        # Check NFS
        nfs_status = self.get_nfs_status()
        results["nfs_status"] = nfs_status.to_dict()

        # Check local
        local_status = self.get_local_status()
        results["local_status"] = local_status.to_dict()

        # Trigger cleanup if needed and cooldown elapsed
        cooldown_elapsed = time.time() - self.last_cleanup_time > CLEANUP_COOLDOWN

        if nfs_status.disk_percent >= self.disk_threshold and cooldown_elapsed:
            logger.warning(f"NFS disk at {nfs_status.disk_percent:.1f}%, triggering cleanup")
            cleanup_path = self.nfs_path / "ai-service" / "data" / "selfplay"
            cleanup_result = await self.cleanup_old_files(cleanup_path)
            results["cleanup_triggered"] = True
            results["cleanup_result"] = {
                "success": cleanup_result.success,
                "files_deleted": cleanup_result.files_deleted,
                "bytes_freed": cleanup_result.bytes_freed,
                "errors": cleanup_result.errors,
            }

        return results

    async def run(self) -> None:
        """Main monitoring loop."""
        self.running = True
        logger.info("Adaptive Resource Manager starting")

        while self.running:
            try:
                # Check and cleanup
                await self.check_and_cleanup()

                # Aggregate data periodically
                if time.time() - self.last_aggregation_time > AGGREGATION_INTERVAL:
                    await self.aggregate_selfplay_data()

            except Exception as e:
                logger.error(f"Resource manager error: {e}")
                self.stats["errors"] += 1

            await asyncio.sleep(CHECK_INTERVAL)

        logger.info("Adaptive Resource Manager stopped")

    def stop(self) -> None:
        """Stop the manager."""
        self.running = False

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "running": self.running,
            "last_cleanup_time": self.last_cleanup_time,
            "last_aggregation_time": self.last_aggregation_time,
            "nfs_path": str(self.nfs_path),
            "data_path": str(self.data_path),
            "disk_threshold": self.disk_threshold,
            "memory_threshold": self.memory_threshold,
            "node_statuses": {
                node_id: status.to_dict()
                for node_id, status in self.node_statuses.items()
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Check manager health for CoordinatorProtocol compliance.

        December 2025 Phase 9: Added for daemon health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        stats = self.get_stats()
        is_running = stats.get("running", False)
        has_errors = stats.get("errors", 0) > 0

        return HealthCheckResult(
            healthy=is_running or not has_errors,
            status=CoordinatorStatus.RUNNING if is_running else CoordinatorStatus.READY,
            message=f"AdaptiveResourceManager: {stats.get('cleanups_triggered', 0)} cleanups, {stats.get('aggregations_completed', 0)} aggregations",
            details=stats,
        )


# Singleton instance
_resource_manager: AdaptiveResourceManager | None = None


def get_resource_manager() -> AdaptiveResourceManager:
    """Get the singleton resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = AdaptiveResourceManager()
    return _resource_manager


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "AdaptiveResourceManager",
    "CleanupResult",
    "ResourceStatus",
    "get_resource_manager",
]
