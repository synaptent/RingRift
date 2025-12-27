#!/usr/bin/env python3
"""Automated Bandwidth Management.

.. deprecated:: December 2025
    This module is superseded by :mod:`sync_bandwidth.py` which provides
    simpler, more focused bandwidth coordination. Prefer sync_bandwidth.py
    for new code. This module will be archived in Q2 2026.

This module manages network bandwidth allocation for data sync operations.
It prevents network congestion by coordinating transfers across hosts.

Extends CoordinatorBase for standardized lifecycle management and SQLitePersistenceMixin
for thread-safe database access.

Features:
- Per-host bandwidth tracking and limits
- Transfer scheduling during low-usage periods
- Rate limiting for concurrent syncs
- Integration with sync mutex
- Historical bandwidth analysis

Usage:
    from app.coordination.bandwidth_manager import (
        get_bandwidth_manager,
        request_bandwidth,
        release_bandwidth,
        get_optimal_transfer_time,
    )

    # Request bandwidth before starting transfer
    allocation = request_bandwidth("gh200-a", estimated_mb=1000)
    if allocation.granted:
        try:
            run_rsync_with_bwlimit(allocation.bwlimit_kbps)
        finally:
            release_bandwidth(allocation.allocation_id)

    # Check optimal time to transfer
    best_time = get_optimal_transfer_time("gh200-a", size_mb=5000)
"""

from __future__ import annotations

import json
import os
import socket
import threading
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorStatus,
    SQLitePersistenceMixin,
)

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_BANDWIDTH_DB = Path("/tmp/ringrift_coordination/bandwidth.db")

# Default bandwidth limits (Mbps)
DEFAULT_HOST_BANDWIDTH = {
    "lambda_h100": 1000,      # 1 Gbps
    "lambda_2xh100": 1000,    # 1 Gbps
    "gh200": 2500,            # 2.5 Gbps
    "aws": 500,               # 500 Mbps
    "default": 100,           # 100 Mbps fallback
}

# Transfer rate limits per priority
PRIORITY_BANDWIDTH_FACTOR = {
    "critical": 1.0,    # Full bandwidth
    "high": 0.8,        # 80% bandwidth
    "normal": 0.5,      # 50% bandwidth
    "low": 0.2,         # 20% bandwidth
    "background": 0.1,  # 10% bandwidth
}

# Use centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import BandwidthDefaults
    MAX_CONCURRENT_TRANSFERS = BandwidthDefaults.MAX_CONCURRENT_TRANSFERS
    MEASUREMENT_WINDOW = BandwidthDefaults.MEASUREMENT_WINDOW
except ImportError:
    # Fallback defaults
    MAX_CONCURRENT_TRANSFERS = 3
    MEASUREMENT_WINDOW = 300  # 5 minutes


class TransferPriority(Enum):
    """Priority levels for bandwidth allocation."""
    CRITICAL = "critical"   # Emergency/urgent transfers
    HIGH = "high"           # Production data
    NORMAL = "normal"       # Regular sync
    LOW = "low"             # Bulk transfers
    BACKGROUND = "background"  # Best effort


@dataclass
class BandwidthAllocation:
    """A bandwidth allocation for a transfer."""

    allocation_id: str
    host: str
    granted: bool
    bwlimit_kbps: int  # Bandwidth limit in KB/s (for rsync --bwlimit)
    priority: TransferPriority
    started_at: float
    estimated_mb: int
    reason: str = ""

    @property
    def bwlimit_mbps(self) -> float:
        return self.bwlimit_kbps / 125  # KB/s to Mbps

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "host": self.host,
            "granted": self.granted,
            "bwlimit_kbps": self.bwlimit_kbps,
            "bwlimit_mbps": round(self.bwlimit_mbps, 1),
            "priority": self.priority.value,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "estimated_mb": self.estimated_mb,
            "reason": self.reason,
        }


@dataclass
class TransferRecord:
    """Record of a completed transfer."""

    host: str
    bytes_transferred: int
    duration_seconds: float
    priority: str
    completed_at: float

    @property
    def mbps(self) -> float:
        if self.duration_seconds <= 0:
            return 0
        return (self.bytes_transferred * 8) / (self.duration_seconds * 1_000_000)


class BandwidthManager(CoordinatorBase, SQLitePersistenceMixin):
    """Manages bandwidth allocation for data transfers.

    Extends CoordinatorBase for standardized lifecycle and SQLitePersistenceMixin
    for thread-safe database access.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        host_limits: dict[str, int] | None = None,
    ):
        CoordinatorBase.__init__(self, name="BandwidthManager")
        self.host_limits = host_limits or DEFAULT_HOST_BANDWIDTH

        # Initialize SQLite persistence
        db_path = db_path or DEFAULT_BANDWIDTH_DB
        self.init_db(db_path)

        # Mark as ready (no async init needed)
        self._status = CoordinatorStatus.READY

    def _get_schema(self) -> str:
        """Get database schema SQL."""
        return '''
            -- Active bandwidth allocations
            CREATE TABLE IF NOT EXISTS allocations (
                allocation_id TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                priority TEXT NOT NULL,
                bwlimit_kbps INTEGER NOT NULL,
                estimated_mb INTEGER NOT NULL,
                holder_pid INTEGER NOT NULL,
                holder_hostname TEXT NOT NULL,
                started_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );

            -- Transfer history for bandwidth analysis
            CREATE TABLE IF NOT EXISTS transfer_history (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                bytes_transferred INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                priority TEXT NOT NULL,
                completed_at REAL NOT NULL,
                hour_of_day INTEGER NOT NULL
            );

            -- Bandwidth usage measurements
            CREATE TABLE IF NOT EXISTS bandwidth_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT NOT NULL,
                bytes_per_second REAL NOT NULL,
                measured_at REAL NOT NULL
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_alloc_host ON allocations(host);
            CREATE INDEX IF NOT EXISTS idx_history_host ON transfer_history(host);
            CREATE INDEX IF NOT EXISTS idx_history_hour ON transfer_history(hour_of_day);
            CREATE INDEX IF NOT EXISTS idx_usage_host ON bandwidth_usage(host);
        '''

    def _get_host_limit_mbps(self, host: str) -> int:
        """Get bandwidth limit for a host in Mbps."""
        # Check direct match
        if host in self.host_limits:
            return self.host_limits[host]

        # Check prefix match
        for prefix, limit in self.host_limits.items():
            if prefix in host.lower():
                return limit

        return self.host_limits.get("default", 100)

    def _get_active_allocations(self, host: str) -> list[dict[str, Any]]:
        """Get active allocations for a host."""
        conn = self._get_connection()
        now = time.time()

        # Clean up expired allocations first
        conn.execute('DELETE FROM allocations WHERE expires_at < ?', (now,))
        conn.commit()

        cursor = conn.execute(
            'SELECT * FROM allocations WHERE host = ?',
            (host,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def request(
        self,
        host: str,
        estimated_mb: int,
        priority: TransferPriority = TransferPriority.NORMAL,
        max_duration_minutes: int = 30,
    ) -> BandwidthAllocation:
        """Request bandwidth allocation for a transfer.

        Args:
            host: Target host for transfer
            estimated_mb: Estimated transfer size in MB
            priority: Transfer priority
            max_duration_minutes: Maximum expected duration

        Returns:
            BandwidthAllocation with granted/denied status and rate limit
        """
        conn = self._get_connection()
        now = time.time()
        hostname = socket.gethostname()
        pid = os.getpid()
        allocation_id = f"{host}_{pid}_{int(now)}"

        # Get host limit and active allocations
        host_limit_mbps = self._get_host_limit_mbps(host)
        active = self._get_active_allocations(host)

        # Check concurrent transfer limit
        if len(active) >= MAX_CONCURRENT_TRANSFERS:
            return BandwidthAllocation(
                allocation_id=allocation_id,
                host=host,
                granted=False,
                bwlimit_kbps=0,
                priority=priority,
                started_at=now,
                estimated_mb=estimated_mb,
                reason=f"Max concurrent transfers ({MAX_CONCURRENT_TRANSFERS}) reached for {host}",
            )

        # Calculate available bandwidth
        used_mbps = sum(a["bwlimit_kbps"] / 125 for a in active)
        available_mbps = host_limit_mbps - used_mbps

        # Apply priority factor
        priority_factor = PRIORITY_BANDWIDTH_FACTOR.get(priority.value, 0.5)
        allocated_mbps = min(available_mbps, host_limit_mbps * priority_factor)

        if allocated_mbps < 10:  # Less than 10 Mbps not worth it
            return BandwidthAllocation(
                allocation_id=allocation_id,
                host=host,
                granted=False,
                bwlimit_kbps=0,
                priority=priority,
                started_at=now,
                estimated_mb=estimated_mb,
                reason=f"Insufficient bandwidth available ({available_mbps:.1f} Mbps free)",
            )

        # Convert to KB/s for rsync --bwlimit
        bwlimit_kbps = int(allocated_mbps * 125)
        expires_at = now + (max_duration_minutes * 60)

        # Record allocation
        conn.execute(
            '''INSERT INTO allocations
               (allocation_id, host, priority, bwlimit_kbps, estimated_mb,
                holder_pid, holder_hostname, started_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (allocation_id, host, priority.value, bwlimit_kbps, estimated_mb,
             pid, hostname, now, expires_at)
        )
        conn.commit()

        return BandwidthAllocation(
            allocation_id=allocation_id,
            host=host,
            granted=True,
            bwlimit_kbps=bwlimit_kbps,
            priority=priority,
            started_at=now,
            estimated_mb=estimated_mb,
            reason=f"Allocated {allocated_mbps:.1f} Mbps",
        )

    def release(
        self,
        allocation_id: str,
        bytes_transferred: int = 0,
        duration_seconds: float = 0,
    ) -> bool:
        """Release a bandwidth allocation.

        Args:
            allocation_id: Allocation to release
            bytes_transferred: Actual bytes transferred (for analytics)
            duration_seconds: Actual duration (for analytics)

        Returns:
            True if allocation was found and released
        """
        conn = self._get_connection()

        # Get allocation info before deleting
        cursor = conn.execute(
            'SELECT host, priority FROM allocations WHERE allocation_id = ?',
            (allocation_id,)
        )
        row = cursor.fetchone()

        if not row:
            return False

        host = row["host"]
        priority = row["priority"]

        # Delete allocation
        conn.execute('DELETE FROM allocations WHERE allocation_id = ?', (allocation_id,))

        # Record transfer history if we have stats
        if bytes_transferred > 0 and duration_seconds > 0:
            hour_of_day = datetime.now().hour
            conn.execute(
                '''INSERT INTO transfer_history
                   (host, bytes_transferred, duration_seconds, priority, completed_at, hour_of_day)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (host, bytes_transferred, duration_seconds, priority, time.time(), hour_of_day)
            )

        conn.commit()
        return True

    def get_host_status(self, host: str) -> dict[str, Any]:
        """Get bandwidth status for a host."""
        host_limit = self._get_host_limit_mbps(host)
        active = self._get_active_allocations(host)

        used_mbps = sum(a["bwlimit_kbps"] / 125 for a in active)
        available_mbps = host_limit - used_mbps

        return {
            "host": host,
            "limit_mbps": host_limit,
            "used_mbps": round(used_mbps, 1),
            "available_mbps": round(available_mbps, 1),
            "active_transfers": len(active),
            "transfers": [
                {
                    "allocation_id": a["allocation_id"],
                    "priority": a["priority"],
                    "bwlimit_mbps": round(a["bwlimit_kbps"] / 125, 1),
                    "estimated_mb": a["estimated_mb"],
                }
                for a in active
            ],
        }

    def get_optimal_time(
        self,
        host: str,
        size_mb: int,
        priority: TransferPriority = TransferPriority.NORMAL,
    ) -> tuple[datetime, str]:
        """Get optimal time to schedule a transfer.

        Args:
            host: Target host
            size_mb: Transfer size in MB
            priority: Transfer priority

        Returns:
            Tuple of (optimal_time, reason)
        """
        conn = self._get_connection()
        now = datetime.now()

        # Check current availability
        status = self.get_host_status(host)
        if status["active_transfers"] == 0:
            return now, "Host currently idle - transfer now"

        # Analyze historical patterns
        cursor = conn.execute(
            '''SELECT hour_of_day, AVG(bytes_transferred / duration_seconds) as avg_rate
               FROM transfer_history
               WHERE host = ? AND completed_at > ?
               GROUP BY hour_of_day
               ORDER BY avg_rate DESC''',
            (host, time.time() - 7 * 86400)  # Last 7 days
        )

        hourly_rates = {row["hour_of_day"]: row["avg_rate"] for row in cursor.fetchall()}

        # If no history, suggest off-peak hours
        if not hourly_rates:
            # Off-peak: 2-6 AM local time
            current_hour = now.hour
            if 2 <= current_hour <= 6:
                return now, "Currently off-peak hours"

            # Find next off-peak window
            hours_until_offpeak = (2 - current_hour) % 24
            optimal = now + timedelta(hours=hours_until_offpeak)
            return optimal, "Scheduling for off-peak hours (2-6 AM)"

        # Find best hour based on historical rates
        best_hour = max(hourly_rates.keys(), key=lambda h: hourly_rates[h])
        current_hour = now.hour

        if best_hour == current_hour:
            return now, f"Current hour ({current_hour}:00) historically fastest"

        hours_until_best = (best_hour - current_hour) % 24
        optimal = now + timedelta(hours=hours_until_best)
        return optimal, f"Hour {best_hour}:00 historically fastest"

    async def get_stats(self) -> dict[str, Any]:
        """Get bandwidth management statistics.

        Implements CoordinatorBase.get_stats() interface.
        """
        # Get base stats from CoordinatorBase
        base_stats = await super().get_stats()

        conn = self._get_connection()

        # Get all active allocations
        cursor = conn.execute('SELECT host, COUNT(*) as count FROM allocations GROUP BY host')
        active_by_host = {row["host"]: row["count"] for row in cursor.fetchall()}

        # Get transfer history summary
        cursor = conn.execute(
            '''SELECT host,
                      COUNT(*) as transfer_count,
                      SUM(bytes_transferred) / 1e9 as total_gb,
                      AVG(bytes_transferred / duration_seconds) / 1e6 as avg_mbps
               FROM transfer_history
               WHERE completed_at > ?
               GROUP BY host''',
            (time.time() - 24 * 3600,)  # Last 24 hours
        )

        history_by_host = {
            row["host"]: {
                "transfers": row["transfer_count"],
                "total_gb": round(row["total_gb"] or 0, 2),
                "avg_mbps": round((row["avg_mbps"] or 0) * 8, 1),  # Convert to bits
            }
            for row in cursor.fetchall()
        }

        # Merge with bandwidth-specific stats
        base_stats.update({
            "active_allocations": active_by_host,
            "history_24h": history_by_host,
            "host_limits": self.host_limits,
        })
        return base_stats

    def get_stats_sync(self) -> dict[str, Any]:
        """Synchronous version of get_stats for CLI usage."""
        conn = self._get_connection()

        cursor = conn.execute('SELECT host, COUNT(*) as count FROM allocations GROUP BY host')
        active_by_host = {row["host"]: row["count"] for row in cursor.fetchall()}

        cursor = conn.execute(
            '''SELECT host,
                      COUNT(*) as transfer_count,
                      SUM(bytes_transferred) / 1e9 as total_gb,
                      AVG(bytes_transferred / duration_seconds) / 1e6 as avg_mbps
               FROM transfer_history
               WHERE completed_at > ?
               GROUP BY host''',
            (time.time() - 24 * 3600,)
        )

        history_by_host = {
            row["host"]: {
                "transfers": row["transfer_count"],
                "total_gb": round(row["total_gb"] or 0, 2),
                "avg_mbps": round((row["avg_mbps"] or 0) * 8, 1),
            }
            for row in cursor.fetchall()
        }

        return {
            "name": self.name,
            "status": self.status.value,
            "active_allocations": active_by_host,
            "history_24h": history_by_host,
            "host_limits": self.host_limits,
        }

    def cleanup(self, max_age_days: int = 30) -> int:
        """Clean up old transfer history."""
        conn = self._get_connection()
        cutoff = time.time() - (max_age_days * 86400)
        cursor = conn.execute('DELETE FROM transfer_history WHERE completed_at < ?', (cutoff,))
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        self._close_connection()


# Global singleton
_manager: BandwidthManager | None = None
_manager_lock = threading.RLock()


def get_bandwidth_manager(db_path: Path | None = None) -> BandwidthManager:
    """Get the global bandwidth manager singleton."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = BandwidthManager(db_path)
        return _manager


def reset_bandwidth_manager() -> None:
    """Reset the global manager (for testing)."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
        _manager = None


# Convenience functions


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


def get_host_bandwidth_status(host: str) -> dict[str, Any]:
    """Get bandwidth status for a host."""
    return get_bandwidth_manager().get_host_status(host)


def get_optimal_transfer_time(
    host: str,
    size_mb: int,
) -> tuple[datetime, str]:
    """Get optimal time to transfer."""
    return get_bandwidth_manager().get_optimal_time(host, size_mb)


def get_bandwidth_stats() -> dict[str, Any]:
    """Get bandwidth management statistics (sync version)."""
    return get_bandwidth_manager().get_stats_sync()


@contextmanager
def bandwidth_allocation(
    host: str,
    estimated_mb: int,
    priority: TransferPriority = TransferPriority.NORMAL,
) -> Generator[BandwidthAllocation]:
    """Context manager for bandwidth allocation.

    Usage:
        with bandwidth_allocation("gh200-a", 1000) as alloc:
            if alloc.granted:
                rsync_with_bwlimit(alloc.bwlimit_kbps)
    """
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


# Command-line interface

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bandwidth management")
    parser.add_argument("--status", type=str, help="Show status for host")
    parser.add_argument("--stats", action="store_true", help="Show overall stats")
    parser.add_argument("--request", nargs=2, help="Request bandwidth: <host> <mb>")
    parser.add_argument("--release", type=str, help="Release allocation ID")
    parser.add_argument("--optimal", nargs=2, help="Get optimal time: <host> <mb>")
    parser.add_argument("--priority", type=str, default="normal", help="Priority level")
    args = parser.parse_args()

    manager = get_bandwidth_manager()

    if args.status:
        print(json.dumps(manager.get_host_status(args.status), indent=2))

    elif args.stats:
        print(json.dumps(manager.get_stats_sync(), indent=2))

    elif args.request:
        host, mb = args.request
        priority = TransferPriority(args.priority)
        allocation = manager.request(host, int(mb), priority)
        print(json.dumps(allocation.to_dict(), indent=2))
        if allocation.granted:
            print(f"\nUse: rsync --bwlimit={allocation.bwlimit_kbps} ...")
            print(f"Release with: --release {allocation.allocation_id}")

    elif args.release:
        if manager.release(args.release):
            print(f"Released allocation {args.release}")
        else:
            print(f"Allocation {args.release} not found")

    elif args.optimal:
        host, mb = args.optimal
        optimal_time, reason = manager.get_optimal_time(host, int(mb))
        print(f"Optimal time: {optimal_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Reason: {reason}")

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "BandwidthAllocation",
    # Main class
    "BandwidthManager",
    # Enums
    "TransferPriority",
    "TransferRecord",
    "bandwidth_allocation",
    # Functions
    "get_bandwidth_manager",
    "get_bandwidth_stats",
    "get_host_bandwidth_status",
    "get_optimal_transfer_time",
    "release_bandwidth",
    "request_bandwidth",
    "reset_bandwidth_manager",
]
