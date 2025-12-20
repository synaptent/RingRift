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

from .client import (
    DistributedEvaluator,
    EvaluationStats,
    QueueDistributedEvaluator,
    WorkerClient,
)
from .discovery import (
    WorkerDiscovery,
    WorkerInfo,
    discover_workers,
    filter_healthy_workers,
    parse_manual_workers,
    wait_for_workers,
)
from .game_collector import (
    CollectedGame,
    InMemoryGameCollector,
    deserialize_game_data,
    write_games_to_db,
)
from .hosts import (
    BOARD_MEMORY_REQUIREMENTS,
    HostConfig,
    HostMemoryInfo,
    SSHExecutor,
    clear_memory_cache,
    detect_all_host_memory,
    detect_host_memory,
    get_eligible_hosts_for_board,
    get_high_memory_hosts,
    get_local_memory_gb,
    get_remote_memory_gb,
    get_ssh_executor,
    load_remote_hosts,
)
from .memory import (
    MemoryProfile,
    MemorySample,
    MemoryTracker,
    RemoteMemoryMonitor,
    format_memory_profile,
    get_current_rss_mb,
    get_peak_rss_mb,
    get_process_rss_mb,
    profile_function,
    write_memory_report,
)
from .queue import (
    EvalResult,
    EvalTask,
    GameReplayData,
    RedisQueue,
    SQSQueue,
    TaskQueue,
    get_task_queue,
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
from .circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitStatus,
    format_circuit_status,
    get_host_breaker,
    get_training_breaker,
)
from .content_deduplication import (
    ContentDeduplicator,
    DeduplicationResult,
    GameFingerprint,
    create_deduplicator,
)
from .db_utils import (
    TransactionManager,
    atomic_json_update,
    atomic_write,
    exclusive_db_lock,
    load_state_safely,
    safe_transaction,
    save_state_atomically,
)
from .event_helpers import (
    create_event,
    emit_elo_updated_safe,
    emit_error_safe,
    emit_evaluation_completed_safe,
    emit_event_safe,
    emit_model_promoted_safe,
    emit_new_games_safe,
    emit_sync,
    emit_training_completed_safe,
    emit_training_failed_safe,
    emit_training_started_safe,
    get_event_bus_safe,
    get_event_types,
    has_event_bus,
    subscribe_safe,
)
from .health_checks import (
    ComponentHealth,
    HealthChecker,
    HealthSummary,
    format_health_report,
    get_health_summary,
)

# Deprecated WAL wrappers - use UnifiedWAL instead
from .ingestion_wal import (
    IngestionWAL,  # DEPRECATED: Use UnifiedWAL directly
    create_ingestion_wal,
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

# Storage provider abstraction for NFS, ephemeral, and local storage
from .storage_provider import (
    LambdaNFSProvider,
    LocalStorageProvider,
    StorageCapabilities,
    StoragePaths,
    StorageProvider,
    StorageProviderType,
    VastEphemeralProvider,
    detect_storage_provider,
    get_models_dir,
    get_optimal_transport_config,
    get_scratch_dir,
    get_selfplay_dir,
    get_storage_provider,
    get_training_dir,
    is_nfs_available,
    should_sync_to_node,
)

# Sync coordinator - unified entry point for all sync operations
from .sync_coordinator import (
    ClusterSyncStats,
    SyncCategory,
    SyncCoordinator,
    SyncStats,
    full_cluster_sync,
    sync_games,
    sync_models,
    sync_training_data,
)
from .sync_orchestrator import (
    FullSyncResult,
    SyncOrchestrator,
    SyncOrchestratorConfig,
    SyncOrchestratorState,
    SyncResult,
    get_sync_orchestrator,
    reset_sync_orchestrator,
)
from .sync_utils import (
    rsync_directory,
    rsync_file,
    rsync_push_file,
)
from .unified_data_sync import (
    SyncConfig,
    UnifiedDataSyncService,
    load_hosts_from_yaml,
)

# Unified data sync components
from .unified_manifest import (
    DataManifest,
    HostSyncState,
    create_manifest,
)

# Unified WAL - recommended for new code
from .unified_wal import (
    UnifiedWAL,
    WALCheckpoint,
    WALEntry,
    WALEntryStatus,
    WALEntryType,
    WALStats,
    get_unified_wal,
)

__all__ = [
    "BOARD_MEMORY_REQUIREMENTS",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CircuitStatus",
    # Cluster coordination
    "ClusterCoordinator",
    # Deprecated cluster coordination (lazy loaded)
    "ClusterCoordinator",
    "ClusterSyncStats",
    "CollectedGame",
    "ComponentHealth",
    # Content deduplication
    "ContentDeduplicator",
    # Unified data sync
    "DataManifest",
    "DeduplicationResult",
    "DistributedEvaluator",
    "EvalResult",
    "EvalTask",
    "EvaluationStats",
    "FullSyncResult",
    "GameFingerprint",
    "GameReplayData",
    # Health checks
    "HealthChecker",
    "HealthSummary",
    # Host configuration and memory detection
    "HostConfig",
    "HostMemoryInfo",
    "HostSyncState",
    # Game recording
    "InMemoryGameCollector",
    # Deprecated WAL wrappers - use UnifiedWAL instead
    "IngestionWAL",  # DEPRECATED
    "LambdaNFSProvider",
    "LocalStorageProvider",
    # Manifest replication
    "ManifestReplicator",
    "MemoryProfile",
    # Memory profiling
    "MemorySample",
    "MemoryTracker",
    # P2P sync
    "P2PFallbackSync",
    "P2PSyncClient",
    "ProcessLimits",
    "ProcessLimits",
    "QueueDistributedEvaluator",
    "RedisQueue",
    "RemoteMemoryMonitor",
    "ReplicaHost",
    "SQSQueue",
    "SSHExecutor",
    "StorageCapabilities",
    "StoragePaths",
    # Storage provider abstraction
    "StorageProvider",
    "StorageProviderType",
    "SyncCategory",
    "SyncConfig",
    # Sync coordinator
    "SyncCoordinator",
    # Sync orchestrator
    "SyncOrchestrator",
    "SyncOrchestratorConfig",
    "SyncOrchestratorState",
    "SyncResult",
    "SyncStats",
    "TaskInfo",
    "TaskInfo",
    # Cloud deployment (queue-based)
    "TaskQueue",
    "TaskRole",
    "TaskRole",
    "TransactionManager",
    "UnifiedDataSyncService",
    # Unified WAL (recommended)
    "UnifiedWAL",
    "VastEphemeralProvider",
    "WALCheckpoint",
    "WALEntry",
    "WALEntryStatus",
    "WALEntryType",
    "WALStats",
    "WorkerClient",
    # Local cluster (HTTP-based)
    "WorkerDiscovery",
    "WorkerInfo",
    "atomic_json_update",
    # Database utilities
    "atomic_write",
    "check_and_abort_if_role_held",
    "check_and_abort_if_role_held",
    "clear_memory_cache",
    "create_deduplicator",
    "create_event",
    "create_ingestion_wal",
    "create_manifest",
    "create_replicator_from_config",
    "deserialize_game_data",
    "detect_all_host_memory",
    "detect_host_memory",
    "detect_storage_provider",
    "discover_workers",
    "emit_error_safe",
    "emit_evaluation_completed_safe",
    "emit_event_safe",
    "emit_model_promoted_safe",
    "emit_new_games_safe",
    "emit_sync",
    "emit_training_completed_safe",
    "emit_training_failed_safe",
    "emit_training_started_safe",
    "exclusive_db_lock",
    "filter_healthy_workers",
    "format_circuit_status",
    "format_health_report",
    "format_memory_profile",
    "full_cluster_sync",
    "get_current_rss_mb",
    "get_eligible_hosts_for_board",
    "get_event_bus_safe",
    "get_event_types",
    "get_health_summary",
    "get_high_memory_hosts",
    "get_host_breaker",
    "get_local_memory_gb",
    "get_models_dir",
    "get_optimal_transport_config",
    "get_peak_rss_mb",
    "get_process_rss_mb",
    "get_remote_memory_gb",
    "get_scratch_dir",
    "get_selfplay_dir",
    "get_ssh_executor",
    "get_storage_provider",
    "get_sync_orchestrator",
    "get_task_queue",
    "get_training_breaker",
    "get_training_dir",
    "get_unified_wal",
    # Event helpers (safe wrappers)
    "has_event_bus",
    "is_nfs_available",
    "load_hosts_from_yaml",
    "load_remote_hosts",
    "load_state_safely",
    "parse_manual_workers",
    "profile_function",
    "reset_sync_orchestrator",
    "rsync_directory",
    # Sync utilities
    "rsync_file",
    "rsync_push_file",
    "safe_transaction",
    "save_state_atomically",
    "should_sync_to_node",
    "subscribe_safe",
    "sync_games",
    "sync_models",
    "sync_training_data",
    "wait_for_workers",
    "write_games_to_db",
    "write_memory_report",
]


def __getattr__(name: str):
    """Lazy loading for deprecated cluster_coordinator symbols.

    This avoids triggering deprecation warnings on every import of app.distributed.
    The warning only fires when someone actually accesses these deprecated symbols.
    """
    import warnings

    if name in _CLUSTER_COORDINATOR_SYMBOLS:
        warnings.warn(
            f"'{name}' is deprecated. Use the following instead:\n"
            "  - ClusterCoordinator -> app.coordination.task_coordinator.TaskCoordinator\n"
            "  - TaskRole -> app.coordination.orchestrator_registry.OrchestratorRole\n"
            "  - TaskInfo -> app.coordination.orchestrator_registry.OrchestratorInfo\n"
            "  - check_and_abort_if_role_held -> OrchestratorRegistry.acquire_role()\n"
            "See docs/CONSOLIDATION_STATUS_2025_12_19.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .cluster_coordinator import (
            ClusterCoordinator,
            ProcessLimits,
            TaskInfo,
            TaskRole,
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
