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
import os
import tempfile
import time
from contextlib import contextmanager
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

# Dec 2025: Circuit breaker for per-host failure isolation
from app.coordination.node_circuit_breaker import (
    NodeCircuitBreaker,
    NodeCircuitConfig,
    NodeCircuitState,
)

# Dec 2025: Centralized port configuration
from app.config.ports import P2P_DEFAULT_PORT


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
    # December 2025: Added for backward compatibility with tests
    granted: bool = True
    allocation_id: str = ""
    estimated_mb: int = 0
    reason: str = ""

    def __post_init__(self):
        """Set allocation_id from transfer_id if not provided."""
        if not self.allocation_id and self.transfer_id:
            self.allocation_id = self.transfer_id

    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def bwlimit_mbps(self) -> float:
        """Get bandwidth limit in Mbps (megabits per second).

        Converts from KB/s to Mbps: KB/s * 8 / 1000 = KB/s / 125
        """
        return self.bwlimit_kbps / 125.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "host": self.host,
            "priority": self.priority.value,
            "bwlimit_kbps": self.bwlimit_kbps,
            "bwlimit_mbps": self.bwlimit_mbps,
            "allocated_at": self.allocated_at,
            "transfer_id": self.transfer_id,
            "allocation_id": self.allocation_id,
            "expires_at": self.expires_at,
            "granted": self.granted,
            "is_expired": self.is_expired,
            "estimated_mb": self.estimated_mb,
            "reason": self.reason,
        }


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth management."""

    # Default limits (KB/s)
    # Dec 2025: Increased limits for 2-3x faster sync (actual network capacity ~500 MB/s)
    default_bwlimit_kbps: int = int(os.getenv("RINGRIFT_SYNC_DEFAULT_BW_KBPS", "25000"))  # 25 MB/s default (was 10)
    max_bwlimit_kbps: int = int(os.getenv("RINGRIFT_SYNC_MAX_BW_KBPS", "150000"))  # 150 MB/s max (was 50)
    min_bwlimit_kbps: int = 1000  # 1 MB/s minimum

    # Per-host limits
    # Dec 2025: Increased for faster parallel sync
    per_host_limit_kbps: int = int(os.getenv("RINGRIFT_SYNC_PER_HOST_KBPS", "50000"))  # 50 MB/s per host (was 12.5)
    total_limit_kbps: int = int(os.getenv("RINGRIFT_SYNC_TOTAL_KBPS", "300000"))  # 300 MB/s total (was 100)

    # Concurrency
    # Dec 2025: Increased for faster parallel sync (was 3/8)
    max_concurrent_per_host: int = int(os.getenv("RINGRIFT_SYNC_MAX_PER_HOST", "5"))
    max_concurrent_total: int = int(os.getenv("RINGRIFT_SYNC_MAX_TOTAL", "12"))

    # Allocation settings
    allocation_timeout_seconds: float = 3600.0  # 1 hour max allocation
    priority_multipliers: dict[TransferPriority, float] = field(default_factory=lambda: {
        TransferPriority.BACKGROUND: 0.25,  # Lowest priority
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
    except (KeyError, ValueError, AttributeError, TypeError) as e:
        # Config structure errors: missing keys, invalid values, wrong types
        logger.debug(f"Failed to load bandwidth hints (config error): {e}")
        return {}
    except OSError as e:
        # File system errors: file not found, permission denied
        logger.debug(f"Failed to load bandwidth hints (filesystem error): {e}")
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

    # =========================================================================
    # Sync API (December 2025) - Wrapper methods for convenience functions
    # =========================================================================

    def request(
        self,
        host: str,
        estimated_mb: int,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> BandwidthAllocation:
        """Request bandwidth allocation (sync version).

        This is a synchronous wrapper for convenience functions.
        For async code, use request_allocation() instead.

        Args:
            host: Target host for the transfer.
            estimated_mb: Estimated transfer size in MB.
            priority: Transfer priority level.

        Returns:
            BandwidthAllocation with granted=True/False and allocation details.
        """
        import uuid
        transfer_id = str(uuid.uuid4())

        # Check concurrent transfer limit per host
        current_host_transfers = self._host_transfers.get(host, 0)
        if current_host_transfers >= self.config.max_concurrent_per_host:
            return BandwidthAllocation(
                host=host,
                priority=priority,
                bwlimit_kbps=0,
                transfer_id=transfer_id,
                expires_at=0.0,
                granted=False,
                allocation_id=transfer_id,
                estimated_mb=estimated_mb,
                reason=f"Max concurrent transfers per host ({self.config.max_concurrent_per_host})",
            )

        # Check total concurrent limit
        total_transfers = sum(self._host_transfers.values())
        if total_transfers >= self.config.max_concurrent_total:
            return BandwidthAllocation(
                host=host,
                priority=priority,
                bwlimit_kbps=0,
                transfer_id=transfer_id,
                expires_at=0.0,
                granted=False,
                allocation_id=transfer_id,
                estimated_mb=estimated_mb,
                reason=f"Max concurrent transfers total ({self.config.max_concurrent_total})",
            )

        # Calculate bandwidth limit based on priority and host
        if self.config.enable_adaptive and host in self.config.host_bandwidth_hints:
            base_limit = self.config.host_bandwidth_hints[host]
        else:
            base_limit = self.config.per_host_limit_kbps

        multiplier = self.config.priority_multipliers.get(priority, 1.0)

        # Adjust for current host usage
        current_host_usage = self._host_usage.get(host, 0)
        host_max = self.config.host_bandwidth_hints.get(host, self.config.per_host_limit_kbps)
        available = host_max - current_host_usage

        # Check if any bandwidth available
        if available <= 0:
            return BandwidthAllocation(
                host=host,
                priority=priority,
                bwlimit_kbps=0,
                transfer_id=transfer_id,
                expires_at=0.0,
                granted=False,
                allocation_id=transfer_id,
                estimated_mb=estimated_mb,
                reason="Insufficient bandwidth available",
            )

        # Apply priority multiplier but cap at available
        bwlimit = int(min(base_limit * multiplier, available, self.config.max_bwlimit_kbps))
        bwlimit = max(bwlimit, self.config.min_bwlimit_kbps)

        allocation = BandwidthAllocation(
            host=host,
            priority=priority,
            bwlimit_kbps=bwlimit,
            transfer_id=transfer_id,
            expires_at=time.time() + self.config.allocation_timeout_seconds,
            granted=True,
            allocation_id=transfer_id,
            estimated_mb=estimated_mb,
        )

        # Track allocation
        self._allocations[transfer_id] = allocation
        self._host_usage[host] = self._host_usage.get(host, 0) + bwlimit
        self._host_transfers[host] = self._host_transfers.get(host, 0) + 1

        return allocation

    def release(
        self,
        allocation_id: str,
        bytes_transferred: int = 0,
        duration_seconds: float = 0,
    ) -> bool:
        """Release bandwidth allocation (sync version).

        This is a synchronous wrapper for convenience functions.
        For async code, use release_allocation() instead.

        Args:
            allocation_id: ID of the allocation to release.
            bytes_transferred: Actual bytes transferred (for stats).
            duration_seconds: Transfer duration (for stats).

        Returns:
            True if allocation was found and released, False otherwise.
        """
        if allocation_id not in self._allocations:
            return False

        allocation = self._allocations[allocation_id]
        del self._allocations[allocation_id]

        # Update host tracking
        host = allocation.host
        self._host_usage[host] = max(
            0, self._host_usage.get(host, 0) - allocation.bwlimit_kbps
        )
        self._host_transfers[host] = max(
            0, self._host_transfers.get(host, 0) - 1
        )

        # Cleanup any other expired allocations
        self._cleanup_expired()

        return True

    def get_host_status(self, host: str) -> dict:
        """Get bandwidth status for a specific host.

        Args:
            host: Host to get status for.

        Returns:
            Dict with host bandwidth status including active_transfers, used_mbps, limit_mbps, etc.
        """
        # Get limit for this host, supporting prefix matching (e.g., "gh200-*")
        limit_kbps = self._get_host_limit_kbps(host)
        usage_kbps = self._host_usage.get(host, 0)
        available_kbps = limit_kbps - usage_kbps
        active_transfers = self._host_transfers.get(host, 0)

        # Build transfers list for this host
        transfers = [
            {"allocation_id": alloc.allocation_id, "priority": alloc.priority.value}
            for alloc in self._allocations.values()
            if alloc.host == host
        ]

        return {
            "host": host,
            "active_transfers": active_transfers,
            "used_kbps": usage_kbps,
            "used_mbps": usage_kbps / 125.0,  # KB/s to Mbps
            "limit_kbps": limit_kbps,
            "limit_mbps": limit_kbps / 125.0,  # KB/s to Mbps
            "available_kbps": available_kbps,
            "available_mbps": available_kbps / 125.0,  # KB/s to Mbps
            "transfers": transfers,
        }

    def _get_host_limit_kbps(self, host: str) -> int:
        """Get bandwidth limit for a host, supporting prefix matching.

        Known prefixes: gh200 -> 2500 Mbps = 312500 KB/s

        Args:
            host: Host name.

        Returns:
            Bandwidth limit in KB/s.
        """
        # Check exact match first
        if host in self.config.host_bandwidth_hints:
            return self.config.host_bandwidth_hints[host]

        # Check known prefixes (gh200 nodes get higher bandwidth)
        if host.startswith("gh200"):
            return 312500  # 2500 Mbps = 2500 * 125 KB/s

        # Default limit
        return self.config.per_host_limit_kbps

    def get_optimal_time(self, host: str, size_mb: int) -> tuple:
        """Get optimal time to transfer based on current load.

        Args:
            host: Target host.
            size_mb: Transfer size in MB.

        Returns:
            Tuple of (optimal_datetime, reason) - optimal_datetime is when transfer should start.
        """
        from datetime import datetime, timedelta

        status = self.get_host_status(host)
        available_kbps = max(status.get("available_kbps", 0), self.config.min_bwlimit_kbps)
        active_transfers = status.get("active_transfers", 0)

        # For idle hosts, transfer now
        if active_transfers == 0:
            return (datetime.now(), "Host is idle, transfer now")

        # Estimate when current transfers might complete
        # Assume each active transfer is ~50% done on average
        estimated_wait_seconds = 30 * active_transfers  # ~30s wait per active transfer

        optimal_time = datetime.now() + timedelta(seconds=estimated_wait_seconds)
        reason = f"{active_transfers} active transfers, estimated {estimated_wait_seconds}s wait"

        return (optimal_time, reason)

    def get_stats_sync(self) -> dict:
        """Get bandwidth stats (sync version).

        Returns:
            Dict with bandwidth statistics including name, status, etc.
        """
        status = self.get_status()

        return {
            "name": "BandwidthManager",
            "status": "running",
            "active_allocations": len(self._allocations),
            "history_24h": {
                "transfers_completed": 0,  # Would need history tracking
                "bytes_transferred": 0,
            },
            "host_limits": dict(self.config.host_bandwidth_hints),
            "per_host": {
                host: self.get_host_status(host)
                for host in set(self._host_transfers.keys())
            },
            **status,
        }

    def cleanup(self, max_age_days: int = 30) -> int:
        """Clean up old transfer history.

        Args:
            max_age_days: Maximum age of history to keep in days.
                         If 0, delete all history.

        Returns:
            Number of records deleted.
        """
        # For now just cleanup expired allocations
        # Real implementation would clean up historical stats
        before_count = len(self._allocations)
        self._cleanup_expired()
        after_count = len(self._allocations)
        deleted = before_count - after_count

        # If max_age_days is 0, clear all allocations
        if max_age_days == 0:
            deleted = len(self._allocations)
            self._allocations.clear()
            self._host_usage.clear()
            self._host_transfers.clear()

        return deleted


# Phase 5 (Dec 2025): Use canonical SyncResult from sync_constants
# The shared version is a superset with additional fields for metadata/state tracking
from app.coordination.sync_constants import SyncResult


class BandwidthCoordinatedRsync:
    """Rsync wrapper with bandwidth coordination and circuit breaker protection.

    December 29, 2025: Added per-host circuit breaker to prevent cascade failures
    when cluster nodes are slow or unreachable. Circuit opens after 3 consecutive
    failures (configurable) and recovers after 60 seconds.
    """

    def __init__(
        self,
        manager: BandwidthManager | None = None,
        rsync_path: str = "rsync",
        default_options: list[str] | None = None,
        # P11-HIGH-3: Enable checksum verification by default
        verify_checksum: bool = True,
        # Dec 2025: Per-host circuit breaker for failure isolation
        circuit_breaker: NodeCircuitBreaker | None = None,
        circuit_breaker_config: NodeCircuitConfig | None = None,
    ):
        self.manager = manager or BandwidthManager.get_instance()
        self.rsync_path = rsync_path
        self.default_options = default_options or ["-avz", "--progress"]
        self.verify_checksum = verify_checksum
        # Initialize circuit breaker with config or create new one
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        else:
            # Default: 3 failures to open, 60s recovery timeout
            config = circuit_breaker_config or NodeCircuitConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                success_threshold=1,  # Single success to close
            )
            self._circuit_breaker = NodeCircuitBreaker(config=config)

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
        """Execute bandwidth-coordinated rsync with circuit breaker protection.

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

        Note:
            December 29, 2025: Circuit breaker prevents cascade failures.
            If circuit is open, returns immediately with circuit_open error.
        """
        start_time = time.time()

        # Dec 2025: Check circuit breaker before attempting sync
        if not self._circuit_breaker.can_check(host):
            circuit_state = self._circuit_breaker.get_state(host)
            logger.warning(
                f"[BandwidthCoordinatedRsync] Circuit OPEN for {host}, skipping sync"
            )
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"Circuit breaker open for host {host} (state={circuit_state.name})",
                duration_seconds=time.time() - start_time,
            )

        # Get bandwidth allocation
        allocation = await self.manager.request_allocation(
            host=host,
            priority=priority,
            timeout=allocation_timeout,
        )

        if allocation is None:
            # Don't record as failure - this is a bandwidth allocation issue, not host failure
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
                # Dec 2025: Record timeout as failure for circuit breaker
                self._circuit_breaker.record_failure(host)
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

            # Dec 2025: Record circuit breaker success/failure
            if success:
                self._circuit_breaker.record_success(host)
                logger.info(
                    f"[BandwidthCoordinatedRsync] Sync complete: "
                    f"{bytes_transferred / 1024 / 1024:.1f} MB in {duration:.1f}s "
                    f"({effective_rate:.1f} KB/s effective)"
                )
            else:
                self._circuit_breaker.record_failure(host)
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

    async def sync_with_fallback(
        self,
        source: str,
        dest: str,
        host: str,
        priority: TransferPriority = TransferPriority.NORMAL,
        extra_options: list[str] | None = None,
        timeout: float = 3600.0,
        allocation_timeout: float = 60.0,
        verify_checksum: bool | None = None,
        use_base64_fallback: bool = True,
        use_chunked_fallback: bool = True,
        use_http_fallback: bool = True,
        p2p_port: int = P2P_DEFAULT_PORT,
    ) -> SyncResult:
        """Execute bandwidth-coordinated rsync with automatic fallback.

        December 2025: This method wraps `sync()` and automatically falls back
        to base64, chunked, or HTTP transfer when rsync fails with connection reset
        errors. This is useful for connections that experience "Connection reset
        by peer" errors due to firewall/proxy binary corruption.

        Fallback order:
        1. rsync with bandwidth coordination (fastest)
        2. base64 transfer (works when binary streams fail)
        3. chunked transfer (for very unstable connections)
        4. HTTP transfer via P2P endpoints (December 2025 - permanent workaround)

        Args:
            source: Source path (local or remote)
            dest: Destination path (local or remote)
            host: Host identifier for bandwidth tracking
            priority: Transfer priority
            extra_options: Additional rsync options
            timeout: Rsync execution timeout
            allocation_timeout: Max time to wait for bandwidth allocation
            verify_checksum: Use checksum verification (slower but safer)
            use_base64_fallback: Enable base64 fallback on connection reset
            use_chunked_fallback: Enable chunked fallback as last resort

        Returns:
            SyncResult with transfer details
        """
        # Try rsync first
        result = await self.sync(
            source=source,
            dest=dest,
            host=host,
            priority=priority,
            extra_options=extra_options,
            timeout=timeout,
            allocation_timeout=allocation_timeout,
            verify_checksum=verify_checksum,
        )

        if result.success:
            return result

        # Check if this is a connection reset error
        from app.distributed.resilient_transfer import CONNECTION_RESET_PATTERNS

        error_lower = (result.error or "").lower() + (result.stderr or "").lower()
        is_connection_reset = any(
            pattern in error_lower for pattern in CONNECTION_RESET_PATTERNS
        )

        if not is_connection_reset:
            # Not a connection reset - return original error
            logger.debug(
                f"[BandwidthCoordinatedRsync] sync failed but not connection reset: {result.error}"
            )
            return result

        logger.warning(
            f"[BandwidthCoordinatedRsync] Connection reset detected, attempting fallback: {result.error}"
        )

        # Try base64 fallback
        if use_base64_fallback:
            base64_result = await self._fallback_base64(source, dest, host)
            if base64_result.success:
                logger.info(
                    f"[BandwidthCoordinatedRsync] base64 fallback succeeded for {source} -> {dest}"
                )
                return base64_result
            logger.warning(f"[BandwidthCoordinatedRsync] base64 fallback failed: {base64_result.error}")

        # Try chunked fallback as last resort
        if use_chunked_fallback:
            chunked_result = await self._fallback_chunked(source, dest, host)
            if chunked_result.success:
                logger.info(
                    f"[BandwidthCoordinatedRsync] chunked fallback succeeded for {source} -> {dest}"
                )
                return chunked_result
            logger.warning(f"[BandwidthCoordinatedRsync] chunked fallback failed: {chunked_result.error}")

        # Try HTTP fallback via P2P endpoints (December 2025)
        if use_http_fallback:
            http_result = await self._fallback_http(source, dest, host, p2p_port)
            if http_result.success:
                logger.info(
                    f"[BandwidthCoordinatedRsync] HTTP fallback succeeded for {source} -> {dest}"
                )
                return http_result
            logger.warning(f"[BandwidthCoordinatedRsync] HTTP fallback failed: {http_result.error}")

        # All fallbacks failed - return original result with updated error
        result.error = (
            f"All transfer methods failed. Original rsync: {result.error}. "
            f"Fallbacks also failed."
        )
        return result

    async def _fallback_base64(self, source: str, dest: str, host: str) -> SyncResult:
        """Execute base64 transfer fallback.

        December 2025: Uses base64 encoding to avoid binary stream corruption.
        """
        import time
        import subprocess
        import os
        import base64

        start_time = time.time()

        try:
            # Parse source and dest to determine direction (push vs pull)
            # Format: "user@host:path" for remote, or just "path" for local

            if ":" in source and "@" in source.split(":")[0]:
                # Pull: remote -> local
                return await self._base64_pull(source, dest, host, start_time)
            else:
                # Push: local -> remote
                return await self._base64_push(source, dest, host, start_time)

        except Exception as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"base64 fallback error: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def _base64_push(
        self, source: str, dest: str, host: str, start_time: float
    ) -> SyncResult:
        """Push local file to remote using base64 encoding."""
        import subprocess
        import base64
        import time
        from pathlib import Path

        source_path = Path(source)
        if not source_path.exists():
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"Source file not found: {source}",
                duration_seconds=time.time() - start_time,
            )

        # Parse dest to get user@host and path
        if ":" not in dest:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error="Invalid dest format - expected user@host:path",
                duration_seconds=time.time() - start_time,
            )

        remote_target, remote_path = dest.rsplit(":", 1)

        # Read and encode file
        with open(source_path, "rb") as f:
            file_data = f.read()
        encoded_data = base64.b64encode(file_data).decode("ascii")
        file_size = len(file_data)

        # Get host config for SSH options
        try:
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            node_config = nodes.get(host)
            ssh_key = node_config.ssh_key if node_config else None
            ssh_port = node_config.ssh_port if node_config else 22
        except (ImportError, KeyError, AttributeError, ValueError, TypeError, OSError):
            # Config module unavailable, host not found, or malformed config
            ssh_key = None
            ssh_port = 22

        # Build SSH command options
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if ssh_key:
            import os
            key_path = os.path.expanduser(ssh_key)
            if os.path.exists(key_path):
                ssh_opts.extend(["-i", key_path])
        if ssh_port != 22:
            ssh_opts.extend(["-p", str(ssh_port)])

        # Ensure remote directory exists and decode file
        remote_dir = str(Path(remote_path).parent)
        decode_cmd = f"mkdir -p '{remote_dir}' && base64 -d > '{remote_path}'"

        ssh_cmd = ["ssh", *ssh_opts, remote_target, decode_cmd]

        # Pipe base64 data through SSH
        result = subprocess.run(
            ssh_cmd,
            input=encoded_data,
            capture_output=True,
            text=True,
            timeout=600,
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"base64 push failed: {result.stderr[:200]}",
                duration_seconds=duration,
            )

        return SyncResult(
            success=True,
            source=source,
            dest=dest,
            host=host,
            bytes_transferred=file_size,
            duration_seconds=duration,
            effective_rate_kbps=file_size / 1024 / duration if duration > 0 else 0,
        )

    async def _base64_pull(
        self, source: str, dest: str, host: str, start_time: float
    ) -> SyncResult:
        """Pull remote file to local using base64 encoding."""
        import subprocess
        import base64
        import time
        from pathlib import Path

        # Parse source to get user@host and path
        remote_target, remote_path = source.rsplit(":", 1)
        dest_path = Path(dest)

        # Get host config for SSH options
        try:
            from app.config.cluster_config import get_cluster_nodes
            nodes = get_cluster_nodes()
            node_config = nodes.get(host)
            ssh_key = node_config.ssh_key if node_config else None
            ssh_port = node_config.ssh_port if node_config else 22
        except (ImportError, KeyError, AttributeError, ValueError, TypeError, OSError):
            # Config module unavailable, host not found, or malformed config
            ssh_key = None
            ssh_port = 22

        # Build SSH command options
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if ssh_key:
            import os
            key_path = os.path.expanduser(ssh_key)
            if os.path.exists(key_path):
                ssh_opts.extend(["-i", key_path])
        if ssh_port != 22:
            ssh_opts.extend(["-p", str(ssh_port)])

        # Command to read and base64-encode remote file
        encode_cmd = f"base64 '{remote_path}'"
        ssh_cmd = ["ssh", *ssh_opts, remote_target, encode_cmd]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            timeout=600,
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"base64 pull failed: {result.stderr.decode('utf-8', errors='replace')[:200]}",
                duration_seconds=duration,
            )

        # Decode base64 data
        try:
            file_data = base64.b64decode(result.stdout)
        except Exception as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"base64 decode failed: {e}",
                duration_seconds=duration,
            )

        # Create parent directory and write file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(file_data)

        file_size = len(file_data)

        return SyncResult(
            success=True,
            source=source,
            dest=dest,
            host=host,
            bytes_transferred=file_size,
            duration_seconds=duration,
            effective_rate_kbps=file_size / 1024 / duration if duration > 0 else 0,
        )

    async def _fallback_chunked(self, source: str, dest: str, host: str) -> SyncResult:
        """Execute chunked transfer fallback.

        December 2025: Uses chunked transfer for very unstable connections.
        This is slower but can handle connections that fail during large transfers.
        """
        import time
        start_time = time.time()

        # Use ResilientTransfer which already has chunked implementation
        try:
            from app.distributed.resilient_transfer import (
                ResilientTransfer,
                TransferRequest,
            )
            from pathlib import Path

            # Parse source and dest to determine direction
            if ":" in source and "@" in source.split(":")[0]:
                # Pull: remote -> local
                remote_target, remote_path = source.rsplit(":", 1)
                local_path = Path(dest)

                # Extract node_id from remote_target (user@host or user@ip)
                remote_host = remote_target.split("@")[-1]

                transfer = ResilientTransfer()
                result = await transfer._transfer_via_chunked(
                    TransferRequest(
                        source_node=host,  # Use the host parameter as node_id
                        source_path=remote_path,
                        target_path=local_path,
                    )
                )

                duration = time.time() - start_time

                return SyncResult(
                    success=result.success,
                    source=source,
                    dest=dest,
                    host=host,
                    bytes_transferred=result.bytes_transferred,
                    duration_seconds=duration,
                    effective_rate_kbps=result.bytes_transferred / 1024 / duration if duration > 0 else 0,
                    error=result.error if not result.success else None,
                )
            else:
                # Push: local -> remote - not implemented in ResilientTransfer
                # Fall back to using base64 with chunking manually
                return SyncResult(
                    success=False,
                    source=source,
                    dest=dest,
                    host=host,
                    error="Chunked push not supported - use base64 fallback",
                    duration_seconds=time.time() - start_time,
                )

        except Exception as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"chunked fallback error: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def _fallback_http(
        self, source: str, dest: str, host: str, p2p_port: int = P2P_DEFAULT_PORT
    ) -> SyncResult:
        """Execute HTTP transfer fallback via P2P endpoints.

        December 2025: Uses P2P orchestrator's /files/ endpoints as permanent
        workaround for SSH "Connection reset by peer" errors on some hosts.

        Note: HTTP is only supported for PULL operations (remote -> local).
        For PUSH, this method returns an error.
        """
        import time
        import urllib.request
        import urllib.error
        from pathlib import Path

        start_time = time.time()

        try:
            # Parse source to determine direction
            if ":" in source and "@" in source.split(":")[0]:
                # Pull: remote -> local
                return await self._http_pull(source, dest, host, p2p_port, start_time)
            else:
                # Push: local -> remote - not supported via HTTP
                return SyncResult(
                    success=False,
                    source=source,
                    dest=dest,
                    host=host,
                    error="HTTP push not supported - use rsync/base64 for push operations",
                    duration_seconds=time.time() - start_time,
                )

        except Exception as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"HTTP fallback error: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def _http_pull(
        self,
        source: str,
        dest: str,
        host: str,
        p2p_port: int,
        start_time: float,
    ) -> SyncResult:
        """Pull remote file to local via HTTP from P2P endpoints."""
        import time
        import urllib.request
        import urllib.error
        from pathlib import Path

        # Parse source to get user@host and path
        remote_target, remote_path = source.rsplit(":", 1)
        dest_path = Path(dest)

        # Get host IP for HTTP request
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            node_config = nodes.get(host)
            if node_config:
                http_host = node_config.best_ip or node_config.ssh_host
            else:
                # Extract from remote_target
                http_host = remote_target.split("@")[-1]
        except (ImportError, KeyError, AttributeError, ValueError, TypeError, OSError):
            # Config module unavailable, host not found, or malformed config
            http_host = remote_target.split("@")[-1]

        # Determine endpoint from path
        if "models" in remote_path or remote_path.endswith(".pth"):
            # Extract just the filename for models
            filename = Path(remote_path).name
            url = f"http://{http_host}:{p2p_port}/files/models/{filename}"
        elif "data/games" in remote_path or remote_path.endswith(".db"):
            filename = Path(remote_path).name
            url = f"http://{http_host}:{p2p_port}/files/data/{filename}"
        elif "data/training" in remote_path or remote_path.endswith(".npz"):
            filename = Path(remote_path).name
            url = f"http://{http_host}:{p2p_port}/files/data/{filename}"
        else:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"Unsupported file type for HTTP transfer: {remote_path}",
                duration_seconds=time.time() - start_time,
            )

        # Download via HTTP
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "RingRift-SyncBandwidth/1.0")

            with urllib.request.urlopen(req, timeout=300) as response:
                # Create parent directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream to file
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        f.write(chunk)

            file_size = dest_path.stat().st_size
            duration = time.time() - start_time

            return SyncResult(
                success=True,
                source=source,
                dest=dest,
                host=host,
                bytes_transferred=file_size,
                duration_seconds=duration,
                effective_rate_kbps=file_size / 1024 / duration if duration > 0 else 0,
            )

        except urllib.error.HTTPError as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"HTTP {e.code}: {e.reason} for {url}",
                duration_seconds=time.time() - start_time,
            )
        except urllib.error.URLError as e:
            return SyncResult(
                success=False,
                source=source,
                dest=dest,
                host=host,
                error=f"HTTP connection error: {e.reason}",
                duration_seconds=time.time() - start_time,
            )


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


# =============================================================================
# Convenience Functions (December 2025 - migrated from bandwidth_manager.py)
# =============================================================================


def reset_bandwidth_manager() -> None:
    """Reset the global manager (for testing)."""
    BandwidthManager.reset_instance()


def request_bandwidth(
    host: str,
    estimated_mb: int,
    priority: TransferPriority = TransferPriority.NORMAL,
) -> BandwidthAllocation:
    """Request bandwidth allocation for a transfer."""
    return get_bandwidth_manager().request(host, estimated_mb, priority)


def release_bandwidth(
    allocation_id: str,
    bytes_transferred: int = 0,
    duration_seconds: float = 0,
) -> bool:
    """Release bandwidth allocation."""
    return get_bandwidth_manager().release(allocation_id, bytes_transferred, duration_seconds)


def get_host_bandwidth_status(host: str) -> dict:
    """Get bandwidth status for a host."""
    return get_bandwidth_manager().get_host_status(host)


def get_optimal_transfer_time(
    host: str,
    size_mb: int,
) -> tuple:
    """Get optimal time to transfer."""
    return get_bandwidth_manager().get_optimal_time(host, size_mb)


def get_bandwidth_stats() -> dict:
    """Get bandwidth management statistics (sync version)."""
    return get_bandwidth_manager().get_stats_sync()


@contextmanager
def bandwidth_allocation(
    host: str,
    estimated_mb: int,
    priority: TransferPriority = TransferPriority.NORMAL,
):
    """Context manager for bandwidth allocation.

    Usage:
        with bandwidth_allocation("gh200-a", 1000) as alloc:
            if alloc.granted:
                rsync_with_bwlimit(alloc.bwlimit_kbps)
    """
    import time
    manager = get_bandwidth_manager()
    allocation = manager.request(host, estimated_mb, priority)
    start_time = time.time()
    bytes_transferred = 0

    try:
        yield allocation
    finally:
        if allocation.granted:
            duration = time.time() - start_time
            manager.release(allocation.allocation_id, bytes_transferred, duration)


__all__ = [
    # Classes
    "BandwidthAllocation",
    "BandwidthConfig",
    "BandwidthCoordinatedRsync",
    "BandwidthManager",
    "BatchRsync",
    "BatchSyncResult",
    "SyncResult",
    "TransferPriority",
    # Factory functions
    "get_bandwidth_manager",
    "get_batch_rsync",
    "get_coordinated_rsync",
    # Convenience functions (migrated from bandwidth_manager.py Dec 2025)
    "bandwidth_allocation",
    "get_bandwidth_stats",
    "get_host_bandwidth_status",
    "get_optimal_transfer_time",
    "release_bandwidth",
    "request_bandwidth",
    "reset_bandwidth_manager",
]
