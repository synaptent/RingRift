"""Unified cluster sync module (December 2025).

Consolidates sync-related functionality from sync_coordinator.py and sync_bandwidth.py.

This is the RECOMMENDED import path for sync functionality.

Usage:
    from app.coordination.cluster.sync import (
        SyncScheduler,
        ClusterDataStatus,
        BandwidthCoordinatedRsync,
    )
"""

from __future__ import annotations

import warnings

# Suppress deprecation warning when importing from package (December 2025)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*app\.coordination\.sync_coordinator.*")
    # Re-export from sync_coordinator (scheduling layer)
    from app.coordination.sync_coordinator import (
        SyncScheduler,
        SyncCoordinator,
        ClusterDataStatus,
        HostDataState,
        get_sync_coordinator,
        get_sync_scheduler,
        wire_sync_events,
    )

# Re-export from sync_bandwidth (execution layer)
from app.coordination.sync_bandwidth import (
    BandwidthCoordinatedRsync,
    TransferPriority,
    BandwidthAllocation,
)

# Re-export from sync_mutex (distributed locking)
from app.coordination.sync_mutex import (
    SyncMutex,
    acquire_sync_lock,
    release_sync_lock,
)

__all__ = [
    # From sync_coordinator
    "SyncScheduler",
    "SyncCoordinator",
    "ClusterDataStatus",
    "HostDataState",
    "get_sync_coordinator",
    "get_sync_scheduler",
    "wire_sync_events",
    # From sync_bandwidth
    "BandwidthCoordinatedRsync",
    "TransferPriority",
    "BandwidthAllocation",
    # From sync_mutex
    "SyncMutex",
    "acquire_sync_lock",
    "release_sync_lock",
]
