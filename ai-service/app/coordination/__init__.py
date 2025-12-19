"""Cluster coordination package for unified resource management.

Provides centralized task coordination to prevent uncontrolled task spawning.

Primary modules:
1. task_coordinator - SQLite-backed coordination with rate limiting, backpressure
2. orchestrator_registry - Role-based mutual exclusion with heartbeat liveness
3. safeguards - Circuit breakers, resource monitoring, spawn rate tracking
4. queue_monitor - Queue depth monitoring with backpressure signals
5. bandwidth_manager - Network bandwidth allocation for transfers
6. sync_mutex - Cross-process mutex for rsync operations
7. p2p_backend - REST API client for P2P orchestrator cluster
8. job_scheduler - Priority-based job scheduling with Elo curriculum
9. stage_events - Event-driven pipeline orchestration with callbacks

Usage:
    # Task coordination (canonical)
    from app.coordination import TaskCoordinator, TaskType
    coordinator = TaskCoordinator.get_instance()
    if coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")[0]:
        coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1")

    # Orchestrator role management
    from app.coordination import acquire_orchestrator_role, OrchestratorRole
    if acquire_orchestrator_role(OrchestratorRole.CLUSTER_ORCHESTRATOR):
        # This process now holds the orchestrator role
        pass

    # Backpressure checking
    from app.coordination import should_throttle_production, QueueType
    if should_throttle_production(QueueType.TRAINING_DATA):
        # Slow down or skip data production
        pass
"""

# Task coordinator exports (canonical coordination system)
from app.coordination.task_coordinator import (
    TaskCoordinator,
    TaskType,
    TaskLimits,
    TaskInfo,
    CoordinatorState,
    OrchestratorLock,
    RateLimiter,
    CoordinatedTask,
    get_coordinator,
    can_spawn,
    emergency_stop_all,
    # Resource-aware task classification
    ResourceType,
    TASK_RESOURCE_MAP,
    get_task_resource_type,
    is_gpu_task,
    is_cpu_task,
)

# Safeguards exports
from app.coordination.safeguards import (
    Safeguards,
    SafeguardConfig,
    SpawnRateTracker,
    ResourceMonitor,
    check_before_spawn,
)

# Circuit breaker - canonical location is app.distributed.circuit_breaker
from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)

# Orchestrator registry exports
from app.coordination.orchestrator_registry import (
    OrchestratorRegistry,
    OrchestratorRole,
    OrchestratorState,
    OrchestratorInfo,
    get_registry,
    acquire_orchestrator_role,
    release_orchestrator_role,
    is_orchestrator_role_available,
    orchestrator_role,
)

# Cross-process event queue exports
from app.coordination.cross_process_events import (
    CrossProcessEventQueue,
    CrossProcessEvent,
    CrossProcessEventPoller,
    get_event_queue,
    reset_event_queue,
    publish_event,
    subscribe_process,
    poll_events,
    ack_event,
    ack_events,
    bridge_to_cross_process,
)

# Host health policy exports (renamed from health_check.py for clarity)
from app.coordination.host_health_policy import (
    HealthStatus,
    check_host_health,
    is_host_healthy,
    get_healthy_hosts,
    get_health_summary,
    clear_health_cache,
    mark_host_unhealthy,
    pre_spawn_check,
)

# Sync mutex exports
from app.coordination.sync_mutex import (
    SyncMutex,
    SyncLockInfo,
    get_sync_mutex,
    reset_sync_mutex,
    acquire_sync_lock,
    release_sync_lock,
    is_sync_locked,
    get_sync_stats,
    sync_lock,
    sync_lock_required,
)

# Duration scheduler exports
from app.coordination.duration_scheduler import (
    DurationScheduler,
    TaskDurationRecord,
    ScheduledTask,
    get_scheduler,
    reset_scheduler,
    estimate_task_duration,
    record_task_completion,
    register_running_task,
    get_resource_availability,
    can_schedule_task,
)

# Queue monitor exports
from app.coordination.queue_monitor import (
    QueueMonitor,
    QueueType,
    QueueStatus,
    BackpressureLevel,
    get_queue_monitor,
    reset_queue_monitor,
    report_queue_depth,
    check_backpressure,
    should_throttle_production,
    should_stop_production,
    get_throttle_factor,
    get_queue_stats,
)

# Bandwidth manager exports
from app.coordination.bandwidth_manager import (
    BandwidthManager,
    BandwidthAllocation,
    TransferPriority,
    get_bandwidth_manager,
    reset_bandwidth_manager,
    request_bandwidth,
    release_bandwidth,
    get_host_bandwidth_status,
    get_optimal_transfer_time,
    get_bandwidth_stats,
    bandwidth_allocation,
)

# Resource targets exports (unified utilization targets)
from app.coordination.resource_targets import (
    ResourceTargetManager,
    UtilizationTargets,
    HostTargets,
    HostTier,
    get_resource_targets,
    get_host_targets,
    should_scale_up,
    should_scale_down,
    get_target_job_count,
    get_utilization_score,
    record_utilization,
    get_cluster_summary,
    set_backpressure,
    reset_resource_targets,
)

# Resource optimizer exports (PID-controlled cluster-wide optimization)
from app.coordination.resource_optimizer import (
    ResourceOptimizer,
    ResourceType,
    ScaleAction,
    NodeResources,
    ClusterState,
    OptimizationResult,
    PIDController,
    get_resource_optimizer,
    get_optimal_concurrency,
    get_cluster_utilization,
    # Note: should_scale_up/down/record_utilization also exist here
    # but we prefer the resource_targets versions for per-host decisions
)

# Sync coordinator exports (unified cluster-wide data sync management)
from app.coordination.sync_coordinator import (
    SyncCoordinator,
    HostDataState,
    HostType,
    SyncPriority,
    SyncAction,
    SyncRecommendation,
    ClusterDataStatus,
    get_sync_coordinator,
    get_cluster_data_status,
    get_sync_recommendations,
    get_next_sync_target,
    register_host,
    update_host_state,
    record_sync_start,
    record_sync_complete,
    record_games_generated,
    reset_sync_coordinator,
)

# Cluster transport layer (unified multi-transport communication)
from app.coordination.cluster_transport import (
    ClusterTransport,
    CircuitBreaker,
    NodeConfig,
    TransportResult,
    get_cluster_transport,
)

# Ephemeral data guard exports (data insurance for ephemeral hosts)
from app.coordination.ephemeral_data_guard import (
    EphemeralDataGuard,
    HostCheckpoint,
    WriteThrough,
    get_ephemeral_guard,
    reset_ephemeral_guard,
    checkpoint_games,
    ephemeral_heartbeat,
    is_host_ephemeral,
    get_evacuation_candidates,
    request_evacuation,
    queue_critical_game,
)

# Transfer verification exports (checksum verification for data integrity)
from app.coordination.transfer_verification import (
    TransferVerifier,
    TransferRecord,
    BatchChecksum,
    QuarantineRecord,
    get_transfer_verifier,
    reset_transfer_verifier,
    compute_file_checksum,
    verify_transfer,
    quarantine_file,
    verify_batch,
    compute_batch_checksum,
)

# Transaction isolation exports (ACID-like guarantees for merge operations)
from app.coordination.transaction_isolation import (
    TransactionIsolation,
    TransactionState,
    MergeOperation,
    MergeTransaction,
    get_transaction_isolation,
    reset_transaction_isolation,
    begin_merge_transaction,
    add_merge_operation,
    complete_merge_operation,
    commit_merge_transaction,
    rollback_merge_transaction,
    merge_transaction,
    get_transaction_stats,
)

# Coordination helpers (safe wrappers to reduce duplicate try/except imports)
from app.coordination.helpers import (
    # Availability checks
    has_coordination,
    has_sync_lock,
    has_bandwidth_manager,
    has_duration_scheduler,
    has_cross_process_events,
    has_resource_targets,
    # Type getters
    get_task_types,
    get_orchestrator_roles,
    get_queue_types,
    get_transfer_priorities,
    get_event_poller_class,
    # Coordinator functions
    get_coordinator_safe,
    can_spawn_safe,
    register_task_safe,
    complete_task_safe,
    fail_task_safe,
    # Orchestrator role functions
    get_registry_safe,
    acquire_role_safe,
    release_role_safe,
    has_role,
    get_role_holder,
    # Safeguard functions
    check_spawn_allowed,
    get_safeguards,
    # Convenience functions
    get_current_node_id,
    is_unified_loop_running,
    warn_if_orchestrator_running,
    # Queue backpressure functions
    should_throttle_safe,
    should_stop_safe,
    get_throttle_factor_safe,
    report_queue_depth_safe,
    # Sync mutex functions
    get_sync_lock_context,
    acquire_sync_lock_safe,
    release_sync_lock_safe,
    # Bandwidth functions
    request_bandwidth_safe,
    release_bandwidth_safe,
    get_bandwidth_context,
    # Duration scheduling functions
    can_schedule_task_safe,
    register_running_task_safe,
    record_task_completion_safe,
    estimate_duration_safe,
    # Cross-process events functions
    publish_event_safe,
    poll_events_safe,
    ack_event_safe,
    subscribe_process_safe,
    # Resource targets functions
    get_resource_targets_safe,
    get_host_targets_safe,
    get_cluster_summary_safe,
    should_scale_up_safe,
    should_scale_down_safe,
    set_backpressure_safe,
)

# P2P Backend exports (REST API client for P2P orchestrator)
from app.coordination.p2p_backend import (
    P2PBackend,
    P2PNodeInfo,
    discover_p2p_leader_url,
    get_p2p_backend,
    P2P_DEFAULT_PORT,
    P2P_HTTP_TIMEOUT,
    HAS_AIOHTTP,
)

# Job Scheduler exports (priority-based job scheduling)
from app.coordination.job_scheduler import (
    PriorityJobScheduler,
    JobPriority,
    ScheduledJob,
    get_scheduler as get_job_scheduler,
    reset_scheduler as reset_job_scheduler,
    get_config_game_counts,
    select_curriculum_config,
    get_underserved_configs,
    get_cpu_rich_hosts,
    get_gpu_rich_hosts,
    TARGET_GPU_UTILIZATION_MIN,
    TARGET_GPU_UTILIZATION_MAX,
    TARGET_CPU_UTILIZATION_MIN,
    TARGET_CPU_UTILIZATION_MAX,
    MIN_MEMORY_GB_FOR_TASKS,
    ELO_CURRICULUM_ENABLED,
)

# Stage Events exports (event-driven pipeline orchestration)
from app.coordination.stage_events import (
    StageEventBus,
    StageEvent,
    StageCompletionResult,
    StageCompletionCallback,
    get_event_bus as get_stage_event_bus,
    reset_event_bus as reset_stage_event_bus,
    create_pipeline_callbacks,
    register_standard_callbacks,
)

# Distributed Locking (Redis + file-based fallback)
from app.coordination.distributed_lock import (
    DistributedLock,
    acquire_training_lock,
    release_training_lock,
    training_lock,
)

# Coordinator Base (common patterns for coordinators/managers)
from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorProtocol,
    CoordinatorStatus,
    CoordinatorStats,
    SQLitePersistenceMixin,
    SingletonMixin,
    CallbackMixin,
    is_coordinator,
)

# Unified Event Coordinator (December 2025 - bridges all event systems)
from app.coordination.unified_event_coordinator import (
    UnifiedEventCoordinator,
    CoordinatorStats as EventCoordinatorStats,
    get_event_coordinator,
    start_coordinator as start_event_coordinator,
    stop_coordinator as stop_event_coordinator,
    get_coordinator_stats as get_event_coordinator_stats,
)

__all__ = [
    # Task Coordinator (canonical)
    "TaskCoordinator",
    "TaskType",
    "TaskLimits",
    "TaskInfo",
    "CoordinatorState",
    "OrchestratorLock",
    "RateLimiter",
    "CoordinatedTask",
    "get_coordinator",
    "can_spawn",
    "emergency_stop_all",
    # Safeguards
    "Safeguards",
    "SafeguardConfig",
    "CircuitBreaker",
    "CircuitState",
    "SpawnRateTracker",
    "ResourceMonitor",
    "check_before_spawn",
    # Orchestrator Registry
    "OrchestratorRegistry",
    "OrchestratorRole",
    "OrchestratorState",
    "OrchestratorInfo",
    "get_registry",
    "acquire_orchestrator_role",
    "release_orchestrator_role",
    "is_orchestrator_role_available",
    "orchestrator_role",
    # Cross-Process Event Queue
    "CrossProcessEventQueue",
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    "get_event_queue",
    "reset_event_queue",
    "publish_event",
    "subscribe_process",
    "poll_events",
    "ack_event",
    "ack_events",
    "bridge_to_cross_process",
    # Health Check
    "HealthStatus",
    "check_host_health",
    "is_host_healthy",
    "get_healthy_hosts",
    "get_health_summary",
    "clear_health_cache",
    "mark_host_unhealthy",
    "pre_spawn_check",
    # Sync Mutex
    "SyncMutex",
    "SyncLockInfo",
    "get_sync_mutex",
    "reset_sync_mutex",
    "acquire_sync_lock",
    "release_sync_lock",
    "is_sync_locked",
    "get_sync_stats",
    "sync_lock",
    "sync_lock_required",
    # Duration Scheduler
    "DurationScheduler",
    "TaskDurationRecord",
    "ScheduledTask",
    "get_scheduler",
    "reset_scheduler",
    "estimate_task_duration",
    "record_task_completion",
    "register_running_task",
    "get_resource_availability",
    "can_schedule_task",
    # Queue Monitor
    "QueueMonitor",
    "QueueType",
    "QueueStatus",
    "BackpressureLevel",
    "get_queue_monitor",
    "reset_queue_monitor",
    "report_queue_depth",
    "check_backpressure",
    "should_throttle_production",
    "should_stop_production",
    "get_throttle_factor",
    "get_queue_stats",
    # Bandwidth Manager
    "BandwidthManager",
    "BandwidthAllocation",
    "TransferPriority",
    "get_bandwidth_manager",
    "reset_bandwidth_manager",
    "request_bandwidth",
    "release_bandwidth",
    "get_host_bandwidth_status",
    "get_optimal_transfer_time",
    "get_bandwidth_stats",
    "bandwidth_allocation",
    # Resource Targets
    "ResourceTargetManager",
    "UtilizationTargets",
    "HostTargets",
    "HostTier",
    "get_resource_targets",
    "get_host_targets",
    "should_scale_up",
    "should_scale_down",
    "get_target_job_count",
    "get_utilization_score",
    "record_utilization",
    "get_cluster_summary",
    "set_backpressure",
    "reset_resource_targets",
    # Resource Optimizer (cluster-wide PID control)
    "ResourceOptimizer",
    "ResourceType",
    "ScaleAction",
    "NodeResources",
    "ClusterState",
    "OptimizationResult",
    "PIDController",
    "get_resource_optimizer",
    "get_optimal_concurrency",
    "get_cluster_utilization",
    # Sync Coordinator (unified data sync management)
    "SyncCoordinator",
    "HostDataState",
    "HostType",
    "SyncPriority",
    "SyncAction",
    "SyncRecommendation",
    "ClusterDataStatus",
    "get_sync_coordinator",
    "get_cluster_data_status",
    "get_sync_recommendations",
    "get_next_sync_target",
    "register_host",
    "update_host_state",
    "record_sync_start",
    "record_sync_complete",
    "record_games_generated",
    "reset_sync_coordinator",
    # Ephemeral Data Guard (data insurance for ephemeral hosts)
    "EphemeralDataGuard",
    "HostCheckpoint",
    "WriteThrough",
    "get_ephemeral_guard",
    "reset_ephemeral_guard",
    "checkpoint_games",
    "ephemeral_heartbeat",
    "is_host_ephemeral",
    "get_evacuation_candidates",
    "request_evacuation",
    "queue_critical_game",
    # Transfer Verification (checksum verification for data integrity)
    "TransferVerifier",
    "TransferRecord",
    "BatchChecksum",
    "QuarantineRecord",
    "get_transfer_verifier",
    "reset_transfer_verifier",
    "compute_file_checksum",
    "verify_transfer",
    "quarantine_file",
    "verify_batch",
    "compute_batch_checksum",
    # Transaction Isolation (ACID-like guarantees for merge operations)
    "TransactionIsolation",
    "TransactionState",
    "MergeOperation",
    "MergeTransaction",
    "get_transaction_isolation",
    "reset_transaction_isolation",
    "begin_merge_transaction",
    "add_merge_operation",
    "complete_merge_operation",
    "commit_merge_transaction",
    "rollback_merge_transaction",
    "merge_transaction",
    "get_transaction_stats",
    # Coordination helpers (safe wrappers)
    "has_coordination",
    "get_task_types",
    "get_orchestrator_roles",
    "get_queue_types",
    "get_transfer_priorities",
    "get_event_poller_class",
    "get_coordinator_safe",
    "can_spawn_safe",
    "register_task_safe",
    "complete_task_safe",
    "fail_task_safe",
    "get_registry_safe",
    "acquire_role_safe",
    "release_role_safe",
    "has_role",
    "get_role_holder",
    "check_spawn_allowed",
    "get_safeguards",
    "get_current_node_id",
    "is_unified_loop_running",
    "warn_if_orchestrator_running",
    # Additional availability checks
    "has_sync_lock",
    "has_bandwidth_manager",
    "has_duration_scheduler",
    "has_cross_process_events",
    "has_resource_targets",
    # Queue backpressure helpers
    "should_throttle_safe",
    "should_stop_safe",
    "get_throttle_factor_safe",
    "report_queue_depth_safe",
    # Sync mutex helpers
    "get_sync_lock_context",
    "acquire_sync_lock_safe",
    "release_sync_lock_safe",
    # Bandwidth helpers
    "request_bandwidth_safe",
    "release_bandwidth_safe",
    "get_bandwidth_context",
    # Duration scheduling helpers
    "can_schedule_task_safe",
    "register_running_task_safe",
    "record_task_completion_safe",
    "estimate_duration_safe",
    # Cross-process events helpers
    "publish_event_safe",
    "poll_events_safe",
    "ack_event_safe",
    "subscribe_process_safe",
    # Resource targets helpers
    "get_resource_targets_safe",
    "get_host_targets_safe",
    "get_cluster_summary_safe",
    "should_scale_up_safe",
    "should_scale_down_safe",
    "set_backpressure_safe",
    # P2P Backend (REST API client for P2P orchestrator)
    "P2PBackend",
    "P2PNodeInfo",
    "discover_p2p_leader_url",
    "get_p2p_backend",
    "P2P_DEFAULT_PORT",
    "P2P_HTTP_TIMEOUT",
    "HAS_AIOHTTP",
    # Job Scheduler (priority-based job scheduling)
    "PriorityJobScheduler",
    "JobPriority",
    "ScheduledJob",
    "get_job_scheduler",
    "reset_job_scheduler",
    "get_config_game_counts",
    "select_curriculum_config",
    "get_underserved_configs",
    "get_cpu_rich_hosts",
    "get_gpu_rich_hosts",
    "TARGET_GPU_UTILIZATION_MIN",
    "TARGET_GPU_UTILIZATION_MAX",
    "TARGET_CPU_UTILIZATION_MIN",
    "TARGET_CPU_UTILIZATION_MAX",
    "MIN_MEMORY_GB_FOR_TASKS",
    "ELO_CURRICULUM_ENABLED",
    # Stage Events (event-driven pipeline orchestration)
    "StageEventBus",
    "StageEvent",
    "StageCompletionResult",
    "StageCompletionCallback",
    "get_stage_event_bus",
    "reset_stage_event_bus",
    "create_pipeline_callbacks",
    "register_standard_callbacks",
    # Distributed Locking
    "DistributedLock",
    "acquire_training_lock",
    "release_training_lock",
    "training_lock",
    # Coordinator Base (common patterns for coordinators/managers)
    "CoordinatorBase",
    "CoordinatorProtocol",
    "CoordinatorStatus",
    "CoordinatorStats",
    "SQLitePersistenceMixin",
    "SingletonMixin",
    "CallbackMixin",
    "is_coordinator",
    # Unified Event Coordinator (bridges all event systems)
    "UnifiedEventCoordinator",
    "EventCoordinatorStats",
    "get_event_coordinator",
    "start_event_coordinator",
    "stop_event_coordinator",
    "get_event_coordinator_stats",
]
