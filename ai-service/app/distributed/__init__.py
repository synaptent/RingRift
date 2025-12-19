"""
Distributed training infrastructure for local Mac cluster and cloud deployment.

This module provides:
- Worker discovery via Bonjour/mDNS
- HTTP client for distributed task execution
- Coordinator utilities for CMA-ES population evaluation
- Queue abstractions for cloud deployment (Redis, SQS)
- In-memory game collection for distributed recording
- Host configuration and memory detection for cluster operations
- Memory profiling and peak tracking utilities
"""

from .hosts import (
    HostConfig,
    HostMemoryInfo,
    load_remote_hosts,
    get_local_memory_gb,
    get_remote_memory_gb,
    detect_host_memory,
    detect_all_host_memory,
    get_eligible_hosts_for_board,
    get_high_memory_hosts,
    clear_memory_cache,
    SSHExecutor,
    get_ssh_executor,
    BOARD_MEMORY_REQUIREMENTS,
)
from .memory import (
    MemorySample,
    MemoryProfile,
    MemoryTracker,
    RemoteMemoryMonitor,
    get_current_rss_mb,
    get_peak_rss_mb,
    get_process_rss_mb,
    profile_function,
    format_memory_profile,
    write_memory_report,
)
from .discovery import (
    WorkerDiscovery,
    WorkerInfo,
    discover_workers,
    wait_for_workers,
    parse_manual_workers,
    filter_healthy_workers,
)
from .client import (
    WorkerClient,
    DistributedEvaluator,
    QueueDistributedEvaluator,
    EvaluationStats,
)
from .queue import (
    TaskQueue,
    EvalTask,
    EvalResult,
    GameReplayData,
    RedisQueue,
    SQSQueue,
    get_task_queue,
)
from .game_collector import (
    InMemoryGameCollector,
    CollectedGame,
    deserialize_game_data,
    write_games_to_db,
)
# Deprecated: cluster_coordinator imports are now lazy to avoid warnings on every import
# Use app.coordination.task_coordinator and app.coordination.orchestrator_registry instead
# These symbols are still available for backward compatibility via __getattr__
_CLUSTER_COORDINATOR_SYMBOLS = {
    "ClusterCoordinator",
    "TaskRole",
    "ProcessLimits",
    "TaskInfo",
    "check_and_abort_if_role_held",
}
from .db_utils import (
    atomic_write,
    safe_transaction,
    exclusive_db_lock,
    atomic_json_update,
    TransactionManager,
    save_state_atomically,
    load_state_safely,
)
from .health_checks import (
    HealthChecker,
    HealthSummary,
    ComponentHealth,
    get_health_summary,
    format_health_report,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitStatus,
    CircuitOpenError,
    get_host_breaker,
    get_training_breaker,
    format_circuit_status,
)
from .event_helpers import (
    has_event_bus,
    get_event_bus_safe,
    get_event_types,
    create_event,
    emit_event_safe,
    subscribe_safe,
    emit_model_promoted_safe,
    emit_training_completed_safe,
    emit_evaluation_completed_safe,
    emit_error_safe,
    emit_elo_updated_safe,
    emit_new_games_safe,
    emit_training_started_safe,
    emit_training_failed_safe,
    emit_sync,
)

# Unified data sync components
from .unified_manifest import (
    DataManifest,
    HostSyncState,
    create_manifest,
)
from .unified_data_sync import (
    UnifiedDataSyncService,
    SyncConfig,
    load_hosts_from_yaml,
)
from .manifest_replication import (
    ManifestReplicator,
    ReplicaHost,
    create_replicator_from_config,
)
from .p2p_sync_client import (
    P2PFallbackSync,
    P2PSyncClient,
)
from .content_deduplication import (
    ContentDeduplicator,
    GameFingerprint,
    DeduplicationResult,
    create_deduplicator,
)
# Unified WAL - recommended for new code
from .unified_wal import (
    UnifiedWAL,
    WALEntry,
    WALEntryType,
    WALEntryStatus,
    WALCheckpoint,
    WALStats,
    get_unified_wal,
)
# Deprecated WAL wrappers - use UnifiedWAL instead
from .ingestion_wal import (
    IngestionWAL,  # DEPRECATED: Use UnifiedWAL directly
    create_ingestion_wal,
)
from .sync_utils import (
    rsync_file,
    rsync_directory,
    rsync_push_file,
)

# Storage provider abstraction for NFS, ephemeral, and local storage
from .storage_provider import (
    StorageProvider,
    StorageProviderType,
    StorageCapabilities,
    StoragePaths,
    LambdaNFSProvider,
    VastEphemeralProvider,
    LocalStorageProvider,
    get_storage_provider,
    detect_storage_provider,
    get_optimal_transport_config,
    is_nfs_available,
    should_sync_to_node,
    get_selfplay_dir,
    get_models_dir,
    get_training_dir,
    get_scratch_dir,
)

# Sync coordinator - unified entry point for all sync operations
from .sync_coordinator import (
    SyncCoordinator,
    SyncCategory,
    SyncStats,
    ClusterSyncStats,
    sync_training_data,
    sync_models,
    sync_games,
    full_cluster_sync,
)

__all__ = [
    # Host configuration and memory detection
    "HostConfig",
    "HostMemoryInfo",
    "load_remote_hosts",
    "get_local_memory_gb",
    "get_remote_memory_gb",
    "detect_host_memory",
    "detect_all_host_memory",
    "get_eligible_hosts_for_board",
    "get_high_memory_hosts",
    "clear_memory_cache",
    "SSHExecutor",
    "get_ssh_executor",
    "BOARD_MEMORY_REQUIREMENTS",
    # Memory profiling
    "MemorySample",
    "MemoryProfile",
    "MemoryTracker",
    "RemoteMemoryMonitor",
    "get_current_rss_mb",
    "get_peak_rss_mb",
    "get_process_rss_mb",
    "profile_function",
    "format_memory_profile",
    "write_memory_report",
    # Local cluster (HTTP-based)
    "WorkerDiscovery",
    "WorkerInfo",
    "discover_workers",
    "wait_for_workers",
    "parse_manual_workers",
    "filter_healthy_workers",
    "WorkerClient",
    "DistributedEvaluator",
    "QueueDistributedEvaluator",
    "EvaluationStats",
    # Cloud deployment (queue-based)
    "TaskQueue",
    "EvalTask",
    "EvalResult",
    "GameReplayData",
    "RedisQueue",
    "SQSQueue",
    "get_task_queue",
    # Game recording
    "InMemoryGameCollector",
    "CollectedGame",
    "deserialize_game_data",
    "write_games_to_db",
    # Cluster coordination
    "ClusterCoordinator",
    "TaskRole",
    "ProcessLimits",
    "TaskInfo",
    "check_and_abort_if_role_held",
    # Database utilities
    "atomic_write",
    "safe_transaction",
    "exclusive_db_lock",
    "atomic_json_update",
    "TransactionManager",
    "save_state_atomically",
    "load_state_safely",
    # Health checks
    "HealthChecker",
    "HealthSummary",
    "ComponentHealth",
    "get_health_summary",
    "format_health_report",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitStatus",
    "CircuitOpenError",
    "get_host_breaker",
    "get_training_breaker",
    "format_circuit_status",
    # Event helpers (safe wrappers)
    "has_event_bus",
    "get_event_bus_safe",
    "get_event_types",
    "create_event",
    "emit_event_safe",
    "subscribe_safe",
    "emit_model_promoted_safe",
    "emit_training_completed_safe",
    "emit_evaluation_completed_safe",
    "emit_error_safe",
    "emit_new_games_safe",
    "emit_training_started_safe",
    "emit_training_failed_safe",
    "emit_sync",
    # Unified data sync
    "DataManifest",
    "HostSyncState",
    "create_manifest",
    "UnifiedDataSyncService",
    "SyncConfig",
    "load_hosts_from_yaml",
    # Manifest replication
    "ManifestReplicator",
    "ReplicaHost",
    "create_replicator_from_config",
    # P2P sync
    "P2PFallbackSync",
    "P2PSyncClient",
    # Content deduplication
    "ContentDeduplicator",
    "GameFingerprint",
    "DeduplicationResult",
    "create_deduplicator",
    # Unified WAL (recommended)
    "UnifiedWAL",
    "WALEntry",
    "WALEntryType",
    "WALEntryStatus",
    "WALCheckpoint",
    "WALStats",
    "get_unified_wal",
    # Deprecated WAL wrappers - use UnifiedWAL instead
    "IngestionWAL",  # DEPRECATED
    "create_ingestion_wal",
    # Sync utilities
    "rsync_file",
    "rsync_directory",
    "rsync_push_file",
    # Deprecated cluster coordination (lazy loaded)
    "ClusterCoordinator",
    "TaskRole",
    "ProcessLimits",
    "TaskInfo",
    "check_and_abort_if_role_held",
]


def __getattr__(name: str):
    """Lazy loading for deprecated cluster_coordinator symbols.

    This avoids triggering deprecation warnings on every import of app.distributed.
    The warning only fires when someone actually accesses these deprecated symbols.
    """
    if name in _CLUSTER_COORDINATOR_SYMBOLS:
        from .cluster_coordinator import (
            ClusterCoordinator,
            TaskRole,
            ProcessLimits,
            TaskInfo,
            check_and_abort_if_role_held,
        )
        symbols = {
            "ClusterCoordinator": ClusterCoordinator,
            "TaskRole": TaskRole,
            "ProcessLimits": ProcessLimits,
            "TaskInfo": TaskInfo,
            "check_and_abort_if_role_held": check_and_abort_if_role_held,
        }
        return symbols[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
