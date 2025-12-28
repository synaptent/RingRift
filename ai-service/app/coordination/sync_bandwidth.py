"""Bandwidth-Coordinated Rsync - Prevents network contention during syncs.

This module provides bandwidth coordination for rsync operations to prevent
multiple sync daemons from saturating network links.

Features:
- Per-host bandwidth allocation
- Priority-based scheduling
- Concurrent transfer limits
- Dynamic bandwidth adjustment based on network conditions

Usage:
    from app.coordination.sync_bandwidth import (
        BandwidthCoordinatedRsync,
        TransferPriority,
        get_bandwidth_manager,
    )

    # Get bandwidth-coordinated rsync
    rsync = BandwidthCoordinatedRsync()

    # Sync with bandwidth limits
    result = await rsync.sync(
        source="/data/games/",
        dest="ubuntu@remote:/data/games/",
        host="gpu-node-1",
        priority=TransferPriority.HIGH,
    )

    # Check current allocations
    manager = get_bandwidth_manager()
    print(manager.get_status())
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Dec 2025: Import WAL sync utilities for database sync operations
from app.coordination.wal_sync_utils import (
    checkpoint_database,
    get_rsync_include_args_for_db,
)


# December 2025: Provider-specific bandwidth hints (KB/s)
# Consolidated to use cluster_config as the single source of truth.
# This re-export is for backward compatibility with existing imports.
def _get_provider_bandwidth_hints() -> dict[str, int]:
    """Get provider bandwidth hints from cluster_config.

    December 2025: Consolidated to single source of truth in cluster_config.py
    This function provides backward compatibility for existing imports.
    """
    try:
        from app.config.cluster_config import _PROVIDER_BANDWIDTH_DEFAULTS_KBS
        # Add tailscale as high-bandwidth provider (not tracked in cluster_config)
        hints = dict(_PROVIDER_BANDWIDTH_DEFAULTS_KBS)
        hints.setdefault("tailscale", 100_000)  # 100 MB/s for Tailscale mesh
        return hints
    except ImportError:
        # Fallback if cluster_config not available
        return {
            "lambda": 100_000,
            "runpod": 100_000,
            "nebius": 50_000,
            "vast": 50_000,
            "vultr": 50_000,
            "hetzner": 80_000,
            "tailscale": 100_000,
            "default": 20_000,
        }


# Backward compatibility: expose as module-level dict
# Note: For new code, use cluster_config.get_node_bandwidth_kbs() instead
PROVIDER_BANDWIDTH_HINTS = _get_provider_bandwidth_hints()


# December 2025: Import TransferPriority from canonical source
from app.coordination.types import TransferPriority

# TransferPriority is now imported from app.coordination.types
# Canonical values: CRITICAL, HIGH, NORMAL, LOW, BACKGROUND


@dataclass
class BandwidthAllocation:
    """Represents a bandwidth allocation for a transfer."""

    host: str
    priority: TransferPriority
    bwlimit_kbps: int
    allocated_at: float = field(default_factory=time.time)
    transfer_id: str = ""
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        return self.expires_at > 0 and time.time() > self.expires_at


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth management."""

    # Default limits (KB/s)
    default_bwlimit_kbps: int = 10000  # 10 MB/s default
    max_bwlimit_kbps: int = 50000  # 50 MB/s max
    min_bwlimit_kbps: int = 1000  # 1 MB/s minimum

    # Per-host limits
    per_host_limit_kbps: int = 20000  # 20 MB/s per host
    total_limit_kbps: int = 100000  # 100 MB/s total across all hosts

    # Concurrency
    max_concurrent_per_host: int = 2
    max_concurrent_total: int = 8

    # Allocation settings
    allocation_timeout_seconds: float = 3600.0  # 1 hour max allocation
    priority_multipliers: dict[TransferPriority, float] = field(default_factory=lambda: {
        TransferPriority.LOW: 0.5,
        TransferPriority.NORMAL: 1.0,
        TransferPriority.HIGH: 1.5,
        TransferPriority.CRITICAL: 2.0,
    })

    # Adaptive bandwidth settings (Phase 8: December 2025)
    enable_adaptive: bool = True
    host_bandwidth_hints: dict[str, int] = field(default_factory=dict)  # host -> KB/s


def load_host_bandwidth_hints() -> dict[str, int]:
    """Load per-host bandwidth hints from distributed_hosts.yaml.

    December 2025: Consolidated to use cluster_config.py helpers.

    Returns:
        Dictionary mapping host names to bandwidth limits in KB/s
    """
    try:
        from app.config.cluster_config import get_cluster_nodes, get_node_bandwidth_kbs

        nodes = get_cluster_nodes()
        hints = {
            name: get_node_bandwidth_kbs(name)
            for name in nodes
        }

        if hints:
            logger.debug(f"Loaded bandwidth hints for {len(hints)} hosts from cluster_config")

        return hints

    except ImportError:
        logger.debug("cluster_config not available, using empty bandwidth hints")
        return {}
    except Exception as e:
        logger.debug(f"Failed to load bandwidth hints: {e}")
        return {}


class BandwidthManager:
    """Manages bandwidth allocations across sync operations."""

    _instance: BandwidthManager | None = None

    def __init__(self, config: BandwidthConfig | None = None):
        self.config = config or BandwidthConfig()
        self._allocations: dict[str, BandwidthAllocation] = {}
        self._host_usage: dict[str, int] = {}  # host -> current KB/s
        self._host_transfers: dict[str, int] = {}  # host -> concurrent count

        # Load adaptive bandwidth hints (Phase 8)
        if self.config.enable_adaptive and not self.config.host_bandwidth_hints:
            self.config.host_bandwidth_hints = load_host_bandwidth_hints()
            if self.config.host_bandwidth_hints:
                logger.info(
                    f"Adaptive bandwidth enabled: {len(self.config.host_bandwidth_hints)} hosts"
                )
        self._lock = asyncio.Lock()
        self._allocation_counter = 0

    @classmethod
    def get_instance(cls, config: BandwidthConfig | None = None) -> BandwidthManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def request_allocation(
        self,
        host: str,
        priority: TransferPriority = TransferPriority.NORMAL,
        timeout: float = 60.0,
    ) -> BandwidthAllocation | None:
        """Request a bandwidth allocation.

        Args:
            host: Target host for transfer
            priority: Transfer priority
            timeout: Max time to wait for allocation

        Returns:
            BandwidthAllocation if granted, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            async with self._lock:
                # Clean up expired allocations
                self._cleanup_expired()

                # Check if we can allocate
                if self._can_allocate(host):
                    allocation = self._create_allocation(host, priority)
                    return allocation

            # Wait and retry
            await asyncio.sleep(1.0)

        logger.warning(f"[BandwidthManager] Timeout waiting for allocation to {host}")
        return None

    def _can_allocate(self, host: str) -> bool:
        """Check if we can allocate bandwidth to host."""
        # Check per-host concurrent limit
        host_count = self._host_transfers.get(host, 0)
        if host_count >= self.config.max_concurrent_per_host:
            return False

        # Check total concurrent limit
        total_count = sum(self._host_transfers.values())
        if total_count >= self.config.max_concurrent_total:
            return False

        # Check total bandwidth
        total_usage = sum(self._host_usage.values())
        if total_usage >= self.config.total_limit_kbps:
            return False

        return True

    def _create_allocation(
        self,
        host: str,
        priority: TransferPriority,
    ) -> BandwidthAllocation:
        """Create a new bandwidth allocation."""
        self._allocation_counter += 1
        transfer_id = f"transfer_{self._allocation_counter}_{int(time.time())}"

        # Calculate bandwidth limit - use adaptive hints if available (Phase 8)
        if self.config.enable_adaptive and host in self.config.host_bandwidth_hints:
            base_limit = self.config.host_bandwidth_hints[host]
        else:
            base_limit = self.config.per_host_limit_kbps

        multiplier = self.config.priority_multipliers.get(priority, 1.0)

        # Adjust for current host usage
        current_host_usage = self._host_usage.get(host, 0)
        host_max = self.config.host_bandwidth_hints.get(host, self.config.per_host_limit_kbps)
        available = host_max - current_host_usage

        # Apply priority multiplier but cap at available
        bwlimit = int(min(base_limit * multiplier, available, self.config.max_bwlimit_kbps))
        bwlimit = max(bwlimit, self.config.min_bwlimit_kbps)

        allocation = BandwidthAllocation(
            host=host,
            priority=priority,
            bwlimit_kbps=bwlimit,
            transfer_id=transfer_id,
            expires_at=time.time() + self.config.allocation_timeout_seconds,
        )

        # Track allocation
        self._allocations[transfer_id] = allocation
        self._host_usage[host] = self._host_usage.get(host, 0) + bwlimit
        self._host_transfers[host] = self._host_transfers.get(host, 0) + 1

        logger.debug(
            f"[BandwidthManager] Allocated {bwlimit} KB/s to {host} "
            f"(priority={priority.value}, id={transfer_id})"
        )

        return allocation

    async def release_allocation(self, allocation: BandwidthAllocation) -> None:
        """Release a bandwidth allocation."""
        async with self._lock:
            if allocation.transfer_id in self._allocations:
                del self._allocations[allocation.transfer_id]

                # Update host tracking
                host = allocation.host
                self._host_usage[host] = max(
                    0, self._host_usage.get(host, 0) - allocation.bwlimit_kbps
                )
                self._host_transfers[host] = max(
                    0, self._host_transfers.get(host, 0) - 1
                )

                logger.debug(
                    f"[BandwidthManager] Released allocation {allocation.transfer_id}"
                )

            # Dec 2025: Proactively cleanup any other expired allocations
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Clean up expired allocations."""
        expired = [
            alloc for alloc in self._allocations.values()
            if alloc.is_expired
        ]
        for alloc in expired:
            if alloc.transfer_id in self._allocations:
                del self._allocations[alloc.transfer_id]
                self._host_usage[alloc.host] = max(
                    0, self._host_usage.get(alloc.host, 0) - alloc.bwlimit_kbps
                )
                self._host_transfers[alloc.host] = max(
                    0, self._host_transfers.get(alloc.host, 0) - 1
                )
                logger.warning(
                    f"[BandwidthManager] Cleaned up expired allocation: {alloc.transfer_id}"
                )

    def get_status(self) -> dict[str, Any]:
        """Get bandwidth manager status."""
        # Dec 2025: Cleanup expired allocations on status check
        self._cleanup_expired()

        total_usage = sum(self._host_usage.values())
        total_transfers = sum(self._host_transfers.values())

        return {
            "total_usage_kbps": total_usage,
            "total_limit_kbps": self.config.total_limit_kbps,
            "usage_percent": (total_usage / self.config.total_limit_kbps * 100)
            if self.config.total_limit_kbps > 0 else 0,
            "active_transfers": total_transfers,
            "max_concurrent": self.config.max_concurrent_total,
            "per_host": {
                host: {
                    "usage_kbps": self._host_usage.get(host, 0),
                    "transfers": self._host_transfers.get(host, 0),
                    "limit_kbps": self.config.per_host_limit_kbps,
                }
                for host in set(self._host_usage.keys()) | set(self._host_transfers.keys())
            },
            "active_allocations": [
                {
                    "transfer_id": a.transfer_id,
                    "host": a.host,
                    "bwlimit_kbps": a.bwlimit_kbps,
                    "priority": a.priority.value,
                    "age_seconds": time.time() - a.allocated_at,
                }
                for a in self._allocations.values()
            ],
        }

    def health_check(self) -> "HealthCheckResult":
        """Check bandwidth manager health for daemon monitoring.

        December 2025: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with health status.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            status = self.get_status()
            total_transfers = status.get("active_transfers", 0)
            max_concurrent = status.get("max_concurrent", 0)
            usage_percent = status.get("usage_percent", 0)

            # Degraded if at capacity
            if total_transfers >= max_concurrent:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"BandwidthManager at capacity: {total_transfers}/{max_concurrent} transfers",
                    details={
                        "active_transfers": total_transfers,
                        "max_concurrent": max_concurrent,
                        "usage_percent": usage_percent,
                    },
                )

            # Healthy
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"BandwidthManager healthy: {total_transfers}/{max_concurrent} transfers, {usage_percent:.1f}% bandwidth",
                details={
                    "active_transfers": total_transfers,
                    "max_concurrent": max_concurrent,
                    "usage_percent": usage_percent,
                    "per_host": status.get("per_host", {}),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"BandwidthManager health check error: {e}",
            )


# Phase 5 (Dec 2025): Use canonical SyncResult from sync_constants
# The shared version is a superset with additional fields for metadata/state tracking
from app.coordination.sync_constants import SyncResult


class BandwidthCoordinatedRsync:
    """Rsync wrapper with bandwidth coordination."""

    def __init__(
        self,
        manager: BandwidthManager | None = None,
        rsync_path: str = "rsync",
        default_options: list[str] | None = None,
        # P11-HIGH-3: Enable checksum verification by default
        verify_checksum: bool = True,
    ):
        self.manager = manager or BandwidthManager.get_instance()
        self.rsync_path = rsync_path
        self.default_options = default_options or ["-avz", "--progress"]
        self.verify_checksum = verify_checksum

    async def sync(
        self,
        source: str,
        dest: str,
        host: str,
        priority: TransferPriority = TransferPriority.NORMAL,
        extra_options: list[str] | None = None,
        timeout: float = 3600.0,
        allocation_timeout: float = 60.0,
        verify_checksum: bool | None = None,
    ) -> SyncResult:
        """Execute bandwidth-coordinated rsync.

        Args:
            source: Source path (local or remote)
            dest: Destination path (local or remote)
            host: Host identifier for bandwidth tracking
            priority: Transfer priority
            extra_options: Additional rsync options
            timeout: Rsync execution timeout
            allocation_timeout: Max time to wait for bandwidth allocation
            verify_checksum: Use checksum verification (slower but safer).
                If None, uses instance default. (P11-HIGH-3 Dec 2025)

        Returns:
            SyncResult with transfer details
        """
        start_time = time.time()

        # Get bandwidth allocation
        allocation = await self.manager.request_allocation(
            host=host,
            priority=priority,
            timeout=allocation_timeout,
        )

        if allocation is None:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error="Failed to acquire bandwidth allocation",
                duration_seconds=time.time() - start_time,
            )

        try:
            # Build rsync command
            cmd = [self.rsync_path]
            cmd.extend(self.default_options)
            cmd.append(f"--bwlimit={allocation.bwlimit_kbps}")

            # P0.1 Dec 2025: Add --partial for resume on network glitches
            # Without this, network interruptions cause full file retransfer
            cmd.append("--partial")
            cmd.append("--partial-dir=.rsync-partial")

            # P11-HIGH-3: Add checksum verification for data integrity
            use_checksum = verify_checksum if verify_checksum is not None else self.verify_checksum
            if use_checksum:
                cmd.append("--checksum")

            if extra_options:
                cmd.extend(extra_options)

            cmd.extend([source, dest])

            logger.info(
                f"[BandwidthCoordinatedRsync] Starting sync: {source} -> {dest} "
                f"(bwlimit={allocation.bwlimit_kbps} KB/s, checksum={use_checksum})"
            )

            # Execute rsync
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SyncResult(
                    success=False,
                    source=source,
                    dest=dest,
                    host=host,
                    bwlimit_kbps=allocation.bwlimit_kbps,
                    error=f"Rsync timed out after {timeout}s",
                    duration_seconds=time.time() - start_time,
                )

            duration = time.time() - start_time
            exit_code = process.returncode or 0
            success = exit_code == 0

            # Parse bytes transferred from output
            bytes_transferred = self._parse_bytes_transferred(
                stdout.decode("utf-8", errors="replace")
            )
            effective_rate = (bytes_transferred / 1024 / duration) if duration > 0 else 0

            result = SyncResult(
                success=success,
                source=source,
                dest=dest,
                host=host,
                bytes_transferred=bytes_transferred,
                duration_seconds=duration,
                bwlimit_kbps=allocation.bwlimit_kbps,
                effective_rate_kbps=effective_rate,
                exit_code=exit_code,
                error=stderr.decode("utf-8", errors="replace") if not success else None,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

            if success:
                logger.info(
                    f"[BandwidthCoordinatedRsync] Sync complete: "
                    f"{bytes_transferred / 1024 / 1024:.1f} MB in {duration:.1f}s "
                    f"({effective_rate:.1f} KB/s effective)"
                )
            else:
                logger.warning(
                    f"[BandwidthCoordinatedRsync] Sync failed: exit_code={exit_code}"
                )

            return result

        finally:
            await self.manager.release_allocation(allocation)

    def _parse_bytes_transferred(self, output: str) -> int:
        """Parse bytes transferred from rsync output."""
        # Look for patterns like "123,456 bytes" or "123.45M"
        import re

        # Try to find "sent X bytes" pattern
        match = re.search(r'sent\s+([\d,]+)\s+bytes', output)
        if match:
            return int(match.group(1).replace(",", ""))

        # Try to find "total size is X" pattern
        match = re.search(r'total size is\s+([\d,]+)', output)
        if match:
            return int(match.group(1).replace(",", ""))

        return 0


# =============================================================================
# Singleton Access
# =============================================================================

def get_bandwidth_manager(config: BandwidthConfig | None = None) -> BandwidthManager:
    """Get the global bandwidth manager instance."""
    return BandwidthManager.get_instance(config)


def get_coordinated_rsync(
    manager: BandwidthManager | None = None,
) -> BandwidthCoordinatedRsync:
    """Get a bandwidth-coordinated rsync instance."""
    return BandwidthCoordinatedRsync(manager)


@dataclass
class BatchSyncResult:
    """Result of a batch sync operation."""

    success: bool
    source_dir: str
    dest: str
    host: str
    files_requested: int = 0
    files_transferred: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    effective_rate_kbps: float = 0.0
    errors: list = field(default_factory=list)


class BatchRsync:
    """Batch rsync operations for transferring multiple files efficiently.

    Features:
    - Uses rsync --files-from for batch transfers
    - Single connection for multiple files (more efficient than per-file)
    - Bandwidth coordination with BandwidthManager
    - Resume support via --partial

    Usage:
        batch_rsync = BatchRsync()

        result = await batch_rsync.sync_files(
            source_dir="/data/games/",
            dest="ubuntu@gpu-node:/data/games/",
            host="gpu-node",
            files=["game1.db", "game2.db", "game3.db"],
        )
    """

    def __init__(
        self,
        manager: BandwidthManager | None = None,
        rsync_path: str = "rsync",
    ):
        self.manager = manager or BandwidthManager.get_instance()
        self.rsync_path = rsync_path

    async def sync_files(
        self,
        source_dir: str,
        dest: str,
        host: str,
        files: list[str],
        priority: TransferPriority = TransferPriority.NORMAL,
        timeout: float = 3600.0,
        partial: bool = True,
    ) -> BatchSyncResult:
        """Sync multiple files in a single rsync operation.

        Args:
            source_dir: Local directory containing the files
            dest: Remote destination (user@host:/path/)
            host: Host identifier for bandwidth tracking
            files: List of filenames relative to source_dir
            priority: Transfer priority
            timeout: Total timeout for the batch
            partial: Enable partial transfers (resume support)

        Returns:
            BatchSyncResult with transfer details
        """
        import tempfile
        import os

        start_time = time.time()
        result = BatchSyncResult(
            success=False,
            source_dir=source_dir,
            dest=dest,
            host=host,
            files_requested=len(files),
        )

        if not files:
            result.success = True
            return result

        # Get bandwidth allocation
        allocation = await self.manager.request_allocation(
            host=host,
            priority=priority,
            timeout=60.0,
        )

        if allocation is None:
            result.errors.append("Failed to acquire bandwidth allocation")
            result.duration_seconds = time.time() - start_time
            return result

        try:
            # Create temp file with list of files to transfer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for filename in files:
                    f.write(filename + '\n')
                files_from_path = f.name

            try:
                # Dec 2025: Checkpoint WAL for database files before sync
                # This ensures all transactions are in the main .db file
                for filename in files:
                    if filename.endswith('.db'):
                        db_path = Path(source_dir) / filename
                        if db_path.exists():
                            checkpoint_database(str(db_path))

                # Dec 2025: Expand file list to include WAL files for databases
                # Without this, WAL transactions may be lost during sync
                expanded_files = []
                for filename in files:
                    expanded_files.append(filename)
                    if filename.endswith('.db'):
                        # Add WAL companion files if they exist
                        wal_file = filename + "-wal"
                        shm_file = filename + "-shm"
                        db_path = Path(source_dir)
                        if (db_path / wal_file).exists():
                            expanded_files.append(wal_file)
                        if (db_path / shm_file).exists():
                            expanded_files.append(shm_file)

                # Rewrite the files-from with expanded list
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    for filename in expanded_files:
                        f.write(filename + '\n')
                    files_from_path = f.name

                # Build rsync command with --files-from
                cmd = [
                    self.rsync_path,
                    "-avz",
                    "--progress",
                    f"--bwlimit={allocation.bwlimit_kbps}",
                    f"--files-from={files_from_path}",
                ]

                if partial:
                    cmd.append("--partial")

                # Ensure source_dir ends with /
                if not source_dir.endswith('/'):
                    source_dir = source_dir + '/'

                cmd.extend([source_dir, dest])

                logger.info(
                    f"[BatchRsync] Starting batch sync: {len(files)} files to {host} "
                    f"(bwlimit={allocation.bwlimit_kbps} KB/s)"
                )

                # Execute rsync
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    result.errors.append(f"Batch sync timed out after {timeout}s")
                    result.duration_seconds = time.time() - start_time
                    return result

                duration = time.time() - start_time
                exit_code = process.returncode or 0
                output = stdout.decode("utf-8", errors="replace")

                result.success = exit_code == 0
                result.duration_seconds = duration

                # Parse transfer stats
                result.bytes_transferred = self._parse_bytes(output)
                result.effective_rate_kbps = (
                    result.bytes_transferred / 1024 / duration
                ) if duration > 0 else 0

                # Count transferred vs skipped files
                result.files_transferred = output.count("\n>f")  # New files
                result.files_skipped = len(files) - result.files_transferred

                if not result.success:
                    result.errors.append(stderr.decode("utf-8", errors="replace"))
                    result.files_failed = len(files) - result.files_transferred

                logger.info(
                    f"[BatchRsync] Batch complete: {result.files_transferred}/{len(files)} files, "
                    f"{result.bytes_transferred / 1024 / 1024:.1f} MB in {duration:.1f}s"
                )

            finally:
                # Clean up temp file
                os.unlink(files_from_path)

        finally:
            await self.manager.release_allocation(allocation)

        return result

    async def sync_directory(
        self,
        source_dir: str,
        dest: str,
        host: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        priority: TransferPriority = TransferPriority.NORMAL,
        timeout: float = 3600.0,
        delete: bool = False,
    ) -> BatchSyncResult:
        """Sync a directory with include/exclude patterns.

        Args:
            source_dir: Local directory to sync
            dest: Remote destination (user@host:/path/)
            host: Host identifier for bandwidth tracking
            include_patterns: Patterns to include (e.g., ["*.npz", "*.db"])
            exclude_patterns: Patterns to exclude (e.g., ["*.tmp", "*.log"])
            priority: Transfer priority
            timeout: Total timeout
            delete: Delete extraneous files on dest

        Returns:
            BatchSyncResult with transfer details
        """
        start_time = time.time()
        result = BatchSyncResult(
            success=False,
            source_dir=source_dir,
            dest=dest,
            host=host,
        )

        # Get bandwidth allocation
        allocation = await self.manager.request_allocation(
            host=host,
            priority=priority,
            timeout=60.0,
        )

        if allocation is None:
            result.errors.append("Failed to acquire bandwidth allocation")
            result.duration_seconds = time.time() - start_time
            return result

        try:
            # Dec 2025: Checkpoint WAL for database files before sync
            # This ensures all transactions are in the main .db file
            if include_patterns and any("*.db" in p or p.endswith(".db") for p in include_patterns):
                # Find and checkpoint all .db files in source directory
                source_path = Path(source_dir)
                if source_path.is_dir():
                    for db_file in source_path.glob("*.db"):
                        checkpoint_database(str(db_file))

            # Build rsync command
            cmd = [
                self.rsync_path,
                "-avz",
                "--progress",
                f"--bwlimit={allocation.bwlimit_kbps}",
            ]

            if delete:
                cmd.append("--delete")

            # Add include patterns first
            # Dec 2025: Auto-expand *.db to include WAL files
            if include_patterns:
                expanded_patterns = []
                for pattern in include_patterns:
                    expanded_patterns.append(pattern)
                    # If pattern includes .db files, also include their WAL companions
                    if pattern == "*.db" or pattern.endswith(".db"):
                        base = pattern[:-3] if pattern.endswith(".db") else pattern[:-2]
                        expanded_patterns.append(f"{base}db-wal")
                        expanded_patterns.append(f"{base}db-shm")
                for pattern in expanded_patterns:
                    cmd.append(f"--include={pattern}")
                # Exclude everything else
                cmd.append("--exclude=*")
            elif exclude_patterns:
                for pattern in exclude_patterns:
                    cmd.append(f"--exclude={pattern}")

            # Ensure source_dir ends with /
            if not source_dir.endswith('/'):
                source_dir = source_dir + '/'

            cmd.extend([source_dir, dest])

            logger.info(
                f"[BatchRsync] Starting directory sync to {host} "
                f"(bwlimit={allocation.bwlimit_kbps} KB/s)"
            )

            # Execute rsync
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.errors.append(f"Directory sync timed out after {timeout}s")
                result.duration_seconds = time.time() - start_time
                return result

            duration = time.time() - start_time
            exit_code = process.returncode or 0
            output = stdout.decode("utf-8", errors="replace")

            result.success = exit_code == 0
            result.duration_seconds = duration
            result.bytes_transferred = self._parse_bytes(output)
            result.effective_rate_kbps = (
                result.bytes_transferred / 1024 / duration
            ) if duration > 0 else 0

            # Count files
            result.files_transferred = output.count("\n>f")

            if not result.success:
                result.errors.append(stderr.decode("utf-8", errors="replace"))

            logger.info(
                f"[BatchRsync] Directory sync complete: {result.files_transferred} files, "
                f"{result.bytes_transferred / 1024 / 1024:.1f} MB in {duration:.1f}s"
            )

        finally:
            await self.manager.release_allocation(allocation)

        return result

    def _parse_bytes(self, output: str) -> int:
        """Parse bytes transferred from rsync output."""
        import re

        match = re.search(r'sent\s+([\d,]+)\s+bytes', output)
        if match:
            return int(match.group(1).replace(",", ""))

        match = re.search(r'total size is\s+([\d,]+)', output)
        if match:
            return int(match.group(1).replace(",", ""))

        return 0


def get_batch_rsync(manager: BandwidthManager | None = None) -> BatchRsync:
    """Get a BatchRsync instance."""
    return BatchRsync(manager)


__all__ = [
    "BandwidthAllocation",
    "BandwidthConfig",
    "BandwidthCoordinatedRsync",
    "BandwidthManager",
    "BatchRsync",
    "BatchSyncResult",
    "SyncResult",
    "TransferPriority",
    "get_bandwidth_manager",
    "get_batch_rsync",
    "get_coordinated_rsync",
]
