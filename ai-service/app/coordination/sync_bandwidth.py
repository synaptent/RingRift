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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TransferPriority(Enum):
    """Priority levels for bandwidth allocation."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


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


class BandwidthManager:
    """Manages bandwidth allocations across sync operations."""

    _instance: BandwidthManager | None = None

    def __init__(self, config: BandwidthConfig | None = None):
        self.config = config or BandwidthConfig()
        self._allocations: dict[str, BandwidthAllocation] = {}
        self._host_usage: dict[str, int] = {}  # host -> current KB/s
        self._host_transfers: dict[str, int] = {}  # host -> concurrent count
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

        # Calculate bandwidth limit
        base_limit = self.config.per_host_limit_kbps
        multiplier = self.config.priority_multipliers.get(priority, 1.0)

        # Adjust for current host usage
        current_host_usage = self._host_usage.get(host, 0)
        available = self.config.per_host_limit_kbps - current_host_usage

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


@dataclass
class SyncResult:
    """Result of a bandwidth-coordinated sync operation."""

    success: bool
    source: str
    dest: str
    host: str
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    bwlimit_kbps: int = 0
    effective_rate_kbps: float = 0.0
    exit_code: int = 0
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


class BandwidthCoordinatedRsync:
    """Rsync wrapper with bandwidth coordination."""

    def __init__(
        self,
        manager: BandwidthManager | None = None,
        rsync_path: str = "rsync",
        default_options: list[str] | None = None,
    ):
        self.manager = manager or BandwidthManager.get_instance()
        self.rsync_path = rsync_path
        self.default_options = default_options or ["-avz", "--progress"]

    async def sync(
        self,
        source: str,
        dest: str,
        host: str,
        priority: TransferPriority = TransferPriority.NORMAL,
        extra_options: list[str] | None = None,
        timeout: float = 3600.0,
        allocation_timeout: float = 60.0,
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

            if extra_options:
                cmd.extend(extra_options)

            cmd.extend([source, dest])

            logger.info(
                f"[BandwidthCoordinatedRsync] Starting sync: {source} -> {dest} "
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


__all__ = [
    "BandwidthAllocation",
    "BandwidthConfig",
    "BandwidthCoordinatedRsync",
    "BandwidthManager",
    "SyncResult",
    "TransferPriority",
    "get_bandwidth_manager",
    "get_coordinated_rsync",
]
