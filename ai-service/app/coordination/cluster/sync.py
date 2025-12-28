"""Unified cluster sync module (December 2025).

DEPRECATED: This is a re-export module for backward compatibility.
Import directly from the source modules instead:

    # Instead of:
    from app.coordination.cluster.sync import SyncScheduler

    # Use:
    from app.coordination.sync_coordinator import SyncScheduler
    from app.coordination.sync_bandwidth import BandwidthCoordinatedRsync
    from app.coordination.sync_mutex import SyncMutex

This module will be removed in Q2 2026.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.cluster.sync is deprecated. "
    "Import directly from app.coordination.sync_coordinator, "
    "app.coordination.sync_bandwidth, or app.coordination.sync_mutex instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
