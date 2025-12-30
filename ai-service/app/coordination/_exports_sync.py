"""Sync-related exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates all sync-related imports in one place.
"""

# Bandwidth manager exports (canonical: sync_bandwidth.py, Dec 2025)
from app.coordination.sync_bandwidth import (
    BandwidthAllocation,
    BandwidthManager,
    TransferPriority,
    bandwidth_allocation,
    get_bandwidth_manager,
    get_bandwidth_stats,
    get_host_bandwidth_status,
    get_optimal_transfer_time,
    release_bandwidth,
    request_bandwidth,
    reset_bandwidth_manager,
)

# Sync SCHEDULER exports (unified cluster-wide data sync SCHEDULING)
# Note: This is the SCHEDULING layer - decides WHEN/WHAT to sync.
# For EXECUTION (HOW to sync), use app.distributed.sync_coordinator.SyncCoordinator
# Suppress deprecation warning when importing from package __init__ (December 2025)
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*app\.coordination\.sync_coordinator.*",
    )
    from app.coordination.sync_coordinator import (
        ClusterDataStatus,
        HostDataState,
        HostType,
        SyncAction,
        SyncPriority,
        SyncRecommendation,
        SyncScheduler,
        execute_priority_sync,
        get_cluster_data_status,
        get_next_sync_target,
        get_sync_coordinator,  # Deprecated: use get_sync_scheduler
        get_sync_recommendations,
        get_sync_scheduler,
        record_games_generated,
        record_sync_complete,
        record_sync_start,
        register_host,
        reset_sync_coordinator,  # Deprecated: use reset_sync_scheduler
        reset_sync_scheduler,
        update_host_state,
    )

# Backward-compatible alias for code that imported SyncCoordinator from coordination
CoordinationSyncCoordinator = SyncScheduler

# Sync stall handler exports (automatic failover for stalled syncs)
from app.coordination.sync_stall_handler import (
    SyncStallHandler,
    get_stall_handler,
    reset_stall_handler,
)

# SyncFacade - Unified entry point for all sync operations (December 2025)
from app.coordination.sync_facade import (
    SyncBackend,
    SyncFacade,
    SyncRequest,
    SyncResponse,
    get_sync_facade,
    reset_sync_facade,
    sync,
)

# Sync mutex exports
from app.coordination.sync_mutex import (
    SyncLockInfo,
    SyncMutex,
    acquire_sync_lock,
    get_sync_mutex,
    get_sync_stats,
    is_sync_locked,
    release_sync_lock,
    reset_sync_mutex,
    sync_lock,
    sync_lock_required,
)

# Sync Durability (December 2025 - WAL and DLQ for sync operations)
from app.coordination.sync_durability import (
    DeadLetterEntry,
    DeadLetterQueue,
    DLQStats,
    SyncStatus,
    SyncWAL,
    SyncWALEntry,
    WALStats,
    get_dlq,
    get_sync_wal,
)

# Sync Integrity (December 2025 - Checksum validation and integrity verification)
from app.coordination.sync_integrity import (
    IntegrityCheckResult,
    IntegrityReport,
    check_sqlite_integrity,
    compute_db_checksum,
    compute_file_checksum as sync_compute_file_checksum,
    verify_checksum,
    verify_sync_integrity,
)

# Sync Bloom Filter (December 2025 - Efficient P2P set membership testing)
from app.coordination.sync_bloom_filter import (
    BloomFilter,
    BloomFilterStats,
    SyncBloomFilter,
    create_event_dedup_filter,
    create_game_id_filter,
    create_model_hash_filter,
)

# Transfer verification exports (checksum verification for data integrity)
from app.coordination.transfer_verification import (
    BatchChecksum,
    QuarantineRecord,
    TransferRecord,
    TransferVerifier,
    compute_batch_checksum,
    compute_file_checksum,
    get_transfer_verifier,
    quarantine_file,
    reset_transfer_verifier,
    verify_batch,
    verify_transfer,
)

# Re-export distributed layer sync execution for convenience
try:
    from app.distributed.sync_coordinator import (
        ClusterSyncStats,
        SyncCategory,
        SyncCoordinator as DistributedSyncCoordinator,
        SyncStats,
        full_cluster_sync,
        get_elo_lookup,
        get_quality_lookup,
        sync_games,
        sync_high_quality_games,
        sync_models,
        sync_training_data,
    )
except ImportError:
    # Distributed layer may not be available in all environments
    pass

__all__ = [
    # Bandwidth
    "BandwidthAllocation",
    "BandwidthManager",
    "TransferPriority",
    "bandwidth_allocation",
    "get_bandwidth_manager",
    "get_bandwidth_stats",
    "get_host_bandwidth_status",
    "get_optimal_transfer_time",
    "release_bandwidth",
    "request_bandwidth",
    "reset_bandwidth_manager",
    # Sync Scheduler
    "ClusterDataStatus",
    "CoordinationSyncCoordinator",
    "HostDataState",
    "HostType",
    "SyncAction",
    "SyncPriority",
    "SyncRecommendation",
    "SyncScheduler",
    "execute_priority_sync",
    "get_cluster_data_status",
    "get_next_sync_target",
    "get_sync_coordinator",
    "get_sync_recommendations",
    "get_sync_scheduler",
    "record_games_generated",
    "record_sync_complete",
    "record_sync_start",
    "register_host",
    "reset_sync_coordinator",
    "reset_sync_scheduler",
    "update_host_state",
    # Stall Handler
    "SyncStallHandler",
    "get_stall_handler",
    "reset_stall_handler",
    # Facade
    "SyncBackend",
    "SyncFacade",
    "SyncRequest",
    "SyncResponse",
    "get_sync_facade",
    "reset_sync_facade",
    "sync",
    # Mutex
    "SyncLockInfo",
    "SyncMutex",
    "acquire_sync_lock",
    "get_sync_mutex",
    "get_sync_stats",
    "is_sync_locked",
    "release_sync_lock",
    "reset_sync_mutex",
    "sync_lock",
    "sync_lock_required",
    # Durability
    "DeadLetterEntry",
    "DeadLetterQueue",
    "DLQStats",
    "SyncStatus",
    "SyncWAL",
    "SyncWALEntry",
    "WALStats",
    "get_dlq",
    "get_sync_wal",
    # Integrity
    "IntegrityCheckResult",
    "IntegrityReport",
    "check_sqlite_integrity",
    "compute_db_checksum",
    "sync_compute_file_checksum",
    "verify_checksum",
    "verify_sync_integrity",
    # Bloom Filter
    "BloomFilter",
    "BloomFilterStats",
    "SyncBloomFilter",
    "create_event_dedup_filter",
    "create_game_id_filter",
    "create_model_hash_filter",
    # Transfer Verification
    "BatchChecksum",
    "QuarantineRecord",
    "TransferRecord",
    "TransferVerifier",
    "compute_batch_checksum",
    "compute_file_checksum",
    "get_transfer_verifier",
    "quarantine_file",
    "reset_transfer_verifier",
    "verify_batch",
    "verify_transfer",
    # Distributed (optional)
    "ClusterSyncStats",
    "DistributedSyncCoordinator",
    "SyncCategory",
    "SyncStats",
    "full_cluster_sync",
    "get_elo_lookup",
    "get_quality_lookup",
    "sync_games",
    "sync_high_quality_games",
    "sync_models",
    "sync_training_data",
]
