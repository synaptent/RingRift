"""Core sync infrastructure - consolidated re-exports.

Provides unified access to data synchronization infrastructure:
- SyncFacade - unified programmatic entry point
- SyncRouter - intelligent routing based on node capabilities
- AutoSyncDaemon - automated P2P data sync
- BandwidthManager - bandwidth coordination
- Sync types and constants

Usage:
    from app.coordination.core_sync import (
        # Facade (main entry point)
        SyncFacade, get_sync_facade, SyncRequest, SyncResponse,

        # Router
        SyncRouter, get_sync_router, SyncRoute, NodeSyncCapability,

        # Daemon
        AutoSyncDaemon, get_auto_sync_daemon, AutoSyncConfig,
        SyncStrategy,

        # Bandwidth
        BandwidthManager, get_bandwidth_manager,
        BandwidthCoordinatedRsync, BatchRsync, get_batch_rsync,

        # Constants and types
        SyncState, SyncPriority, SyncDirection,
        SyncTarget, SyncResult,

        # Integrity
        compute_file_checksum, verify_checksum, check_sqlite_integrity,
    )

This module consolidates:
- sync_facade.py - SyncFacade (entry point)
- sync_router.py - SyncRouter (routing)
- auto_sync_daemon.py - AutoSyncDaemon
- sync_bandwidth.py - BandwidthManager, BatchRsync
- sync_constants.py - SyncState, SyncPriority, SyncResult
- sync_integrity.py - Checksum utilities

Created December 2025 as part of 157->15 module consolidation.
"""

from __future__ import annotations

# =============================================================================
# Sync Constants and Types (from sync_constants.py)
# =============================================================================
from app.coordination.sync_constants import (
    SyncState,
    SyncPriority,
    SyncDirection,
    SyncTarget,
    SyncResult,
)

# =============================================================================
# Sync Facade (from sync_facade.py) - Main entry point
# =============================================================================
from app.coordination.sync_facade import (
    SyncBackend,
    SyncRequest,
    SyncResponse,
    SyncFacade,
    get_sync_facade,
    reset_sync_facade,
)

# =============================================================================
# Sync Router (from sync_router.py)
# =============================================================================
from app.coordination.sync_router import (
    SyncRoute,
    NodeSyncCapability,
    SyncRouter,
    get_sync_router,
    reset_sync_router,
)

# =============================================================================
# Auto Sync Daemon (from auto_sync_daemon.py)
# =============================================================================
from app.coordination.auto_sync_daemon import (
    SyncStrategy,
    AutoSyncConfig,
    AutoSyncDaemon,
    get_auto_sync_daemon,
    reset_auto_sync_daemon,
    # Factory functions for backward compatibility
    create_ephemeral_sync_daemon,
    create_cluster_data_sync_daemon,
    create_training_sync_daemon,
    get_ephemeral_sync_daemon,
    get_cluster_data_sync_daemon,
    is_ephemeral_host,
)

# =============================================================================
# Bandwidth Management (from sync_bandwidth.py)
# =============================================================================
from app.coordination.sync_bandwidth import (
    BandwidthAllocation,
    BandwidthConfig,
    BandwidthManager,
    get_bandwidth_manager,
    BandwidthCoordinatedRsync,
    get_coordinated_rsync,
    BatchSyncResult,
    BatchRsync,
    get_batch_rsync,
)

# =============================================================================
# Sync Integrity (from sync_integrity.py)
# =============================================================================
from app.coordination.sync_integrity import (
    IntegrityCheckResult,
    IntegrityReport,
    compute_file_checksum,
    compute_db_checksum,
    verify_checksum,
    check_sqlite_integrity,
    verify_sync_integrity,
    prepare_database_for_transfer,
    atomic_file_write,
    verified_database_copy,
)

# =============================================================================
# Database Sync Manager (from database_sync_manager.py)
# =============================================================================
from app.coordination.database_sync_manager import (
    DatabaseSyncManager,
    DatabaseSyncState,
    SyncNodeInfo,
)

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # === Constants and types ===
    "SyncState",
    "SyncPriority",
    "SyncDirection",
    "SyncTarget",
    "SyncResult",
    # === Facade ===
    "SyncBackend",
    "SyncRequest",
    "SyncResponse",
    "SyncFacade",
    "get_sync_facade",
    "reset_sync_facade",
    # === Router ===
    "SyncRoute",
    "NodeSyncCapability",
    "SyncRouter",
    "get_sync_router",
    "reset_sync_router",
    # === Daemon ===
    "SyncStrategy",
    "AutoSyncConfig",
    "AutoSyncDaemon",
    "get_auto_sync_daemon",
    "reset_auto_sync_daemon",
    "create_ephemeral_sync_daemon",
    "create_cluster_data_sync_daemon",
    "create_training_sync_daemon",
    "get_ephemeral_sync_daemon",
    "get_cluster_data_sync_daemon",
    "is_ephemeral_host",
    # === Bandwidth ===
    "BandwidthAllocation",
    "BandwidthConfig",
    "BandwidthManager",
    "get_bandwidth_manager",
    "BandwidthCoordinatedRsync",
    "get_coordinated_rsync",
    "BatchSyncResult",
    "BatchRsync",
    "get_batch_rsync",
    # === Integrity ===
    "IntegrityCheckResult",
    "IntegrityReport",
    "compute_file_checksum",
    "compute_db_checksum",
    "verify_checksum",
    "check_sqlite_integrity",
    "verify_sync_integrity",
    "prepare_database_for_transfer",
    "atomic_file_write",
    "verified_database_copy",
    # === Database sync ===
    "DatabaseSyncManager",
    "DatabaseSyncState",
    "SyncNodeInfo",
]
