"""Core coordination exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates core coordination: task coordinator, orchestrator registry,
queue monitor, safeguards, resource management, health, and P2P backend.
"""

# Task coordinator exports (canonical coordination system)
from app.coordination.task_coordinator import (
    TASK_RESOURCE_MAP,
    CoordinatedTask,
    CoordinatorState,
    OrchestratorLock,
    RateLimiter,
    ResourceType,
    TaskCoordinator,
    TaskInfo,
    TaskLimits,
    TaskType,
    can_spawn,
    emergency_stop_all,
    get_coordinator,
    get_task_resource_type,
    is_cpu_task,
    is_gpu_task,
)

# Orchestrator registry exports
from app.coordination.orchestrator_registry import (
    CoordinatorHealth,
    CrossCoordinatorHealthProtocol,
    OrchestratorInfo,
    OrchestratorRegistry,
    OrchestratorRole,
    OrchestratorState,
    acquire_orchestrator_role,
    auto_register_known_coordinators,
    check_cluster_health,
    get_coordinator as get_registered_coordinator,
    get_cross_coordinator_health,
    get_registered_coordinators,
    get_registry,
    is_orchestrator_role_available,
    orchestrator_role,
    register_coordinator,
    release_orchestrator_role,
    shutdown_all_coordinators as shutdown_registered_coordinators,
    unregister_coordinator,
)

# Queue monitor exports
from app.coordination.queue_monitor import (
    BackpressureLevel,
    QueueMonitor,
    QueueStatus,
    QueueType,
    check_backpressure,
    get_queue_monitor,
    get_queue_stats,
    get_throttle_factor,
    report_queue_depth,
    reset_queue_monitor,
    should_stop_production,
    should_throttle_production,
)

# Resource optimizer exports (PID-controlled cluster-wide optimization)
from app.coordination.resource_optimizer import (
    ClusterState,
    NodeResources,
    OptimizationResult,
    PIDController,
    ResourceOptimizer,
    ScaleAction,
    get_cluster_utilization,
    get_optimal_concurrency,
    get_resource_optimizer,
)

# Resource targets exports (unified utilization targets)
from app.coordination.resource_targets import (
    HostTargets,
    HostTier,
    ResourceTargetManager,
    UtilizationTargets,
    get_cluster_summary,
    get_host_targets,
    get_resource_targets,
    get_target_job_count,
    get_utilization_score,
    record_utilization,
    reset_resource_targets,
    set_backpressure,
    should_scale_down,
    should_scale_up,
)

# Safeguards exports
from app.coordination.safeguards import (
    ResourceMonitor,
    SafeguardConfig,
    Safeguards,
    SpawnRateTracker,
    check_before_spawn,
)

# Duration scheduler exports
from app.coordination.duration_scheduler import (
    DurationScheduler,
    ScheduledTask,
    TaskDurationRecord,
    can_schedule_task,
    estimate_task_duration,
    get_resource_availability,
    get_scheduler,
    record_task_completion,
    register_running_task,
    reset_scheduler,
)

# Host health policy exports
from app.coordination.host_health_policy import (
    HostHealthStatus as HealthStatus,
    check_host_health,
    clear_health_cache,
    get_health_summary,
    get_healthy_hosts,
    is_host_healthy,
    mark_host_unhealthy,
    pre_spawn_check,
)

# UnifiedHealthManager - consolidated error recovery and health management
from app.coordination.unified_health_manager import (
    ErrorRecord,
    ErrorSeverity,
    HealthStats,
    JobHealthState,
    NodeHealthState,
    PipelineState,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryConfig,
    RecoveryEvent,
    RecoveryResult,
    RecoveryStatus,
    SystemHealthConfig,
    SystemHealthLevel,
    SystemHealthScore,
    UnifiedHealthManager,
    get_health_manager,
    get_system_health_level,
    get_system_health_score,
    is_component_healthy,
    reset_health_manager,
    should_pause_pipeline,
    wire_health_events,
)

# Health Facade - unified entry point for all health operations
from app.coordination.health_facade import (
    HealthCheckOrchestrator,
    NodeHealthDetails,
    get_cluster_health_summary,
    get_degraded_nodes,
    get_health_orchestrator,
    get_healthy_nodes,
    get_node_health,
    get_node_health_monitor,
    get_offline_nodes,
    get_unhealthy_nodes,
    mark_node_retired,
)

# Ephemeral data guard exports (data insurance for ephemeral hosts)
from app.coordination.ephemeral_data_guard import (
    EphemeralDataGuard,
    HostCheckpoint,
    WriteThrough,
    checkpoint_games,
    ephemeral_heartbeat,
    get_ephemeral_guard,
    get_evacuation_candidates,
    is_host_ephemeral,
    queue_critical_game,
    request_evacuation,
    reset_ephemeral_guard,
)

# P2P Backend exports (REST API client for P2P orchestrator)
from app.coordination.p2p_backend import (
    HAS_AIOHTTP,
    P2P_DEFAULT_PORT,
    P2P_HTTP_TIMEOUT,
    P2PBackend,
    P2PNodeInfo,
    discover_p2p_leader_url,
    get_p2p_backend,
)

# Cluster transport layer (unified multi-transport communication)
from app.coordination.cluster_transport import (
    ClusterTransport,
    NodeConfig,
    TransportResult,
    get_cluster_transport,
)

# Circuit breaker - canonical location is app.distributed.circuit_breaker
from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)

# Job Scheduler exports (priority-based job scheduling)
from app.coordination.job_scheduler import (
    ELO_CURRICULUM_ENABLED,
    MIN_MEMORY_GB_FOR_TASKS,
    TARGET_CPU_UTILIZATION_MAX,
    TARGET_CPU_UTILIZATION_MIN,
    TARGET_GPU_UTILIZATION_MAX,
    TARGET_GPU_UTILIZATION_MIN,
    JobPriority,
    PriorityJobScheduler,
    ScheduledJob,
    get_config_game_counts,
    get_cpu_rich_hosts,
    get_gpu_rich_hosts,
    get_scheduler as get_job_scheduler,
    get_underserved_configs,
    reset_scheduler as reset_job_scheduler,
    select_curriculum_config,
)

__all__ = [
    # Constants
    "ELO_CURRICULUM_ENABLED",
    "HAS_AIOHTTP",
    "MIN_MEMORY_GB_FOR_TASKS",
    "P2P_DEFAULT_PORT",
    "P2P_HTTP_TIMEOUT",
    "TARGET_CPU_UTILIZATION_MAX",
    "TARGET_CPU_UTILIZATION_MIN",
    "TARGET_GPU_UTILIZATION_MAX",
    "TARGET_GPU_UTILIZATION_MIN",
    "TASK_RESOURCE_MAP",
    # Task Coordinator
    "CoordinatedTask",
    "CoordinatorState",
    "OrchestratorLock",
    "RateLimiter",
    "ResourceType",
    "TaskCoordinator",
    "TaskInfo",
    "TaskLimits",
    "TaskType",
    "can_spawn",
    "emergency_stop_all",
    "get_coordinator",
    "get_task_resource_type",
    "is_cpu_task",
    "is_gpu_task",
    # Orchestrator Registry
    "CoordinatorHealth",
    "CrossCoordinatorHealthProtocol",
    "OrchestratorInfo",
    "OrchestratorRegistry",
    "OrchestratorRole",
    "OrchestratorState",
    "acquire_orchestrator_role",
    "auto_register_known_coordinators",
    "check_cluster_health",
    "get_registered_coordinator",
    "get_cross_coordinator_health",
    "get_registered_coordinators",
    "get_registry",
    "is_orchestrator_role_available",
    "orchestrator_role",
    "register_coordinator",
    "release_orchestrator_role",
    "shutdown_registered_coordinators",
    "unregister_coordinator",
    # Queue Monitor
    "BackpressureLevel",
    "QueueMonitor",
    "QueueStatus",
    "QueueType",
    "check_backpressure",
    "get_queue_monitor",
    "get_queue_stats",
    "get_throttle_factor",
    "report_queue_depth",
    "reset_queue_monitor",
    "should_stop_production",
    "should_throttle_production",
    # Resource Optimizer
    "ClusterState",
    "NodeResources",
    "OptimizationResult",
    "PIDController",
    "ResourceOptimizer",
    "ScaleAction",
    "get_cluster_utilization",
    "get_optimal_concurrency",
    "get_resource_optimizer",
    # Resource Targets
    "HostTargets",
    "HostTier",
    "ResourceTargetManager",
    "UtilizationTargets",
    "get_cluster_summary",
    "get_host_targets",
    "get_resource_targets",
    "get_target_job_count",
    "get_utilization_score",
    "record_utilization",
    "reset_resource_targets",
    "set_backpressure",
    "should_scale_down",
    "should_scale_up",
    # Safeguards
    "ResourceMonitor",
    "SafeguardConfig",
    "Safeguards",
    "SpawnRateTracker",
    "check_before_spawn",
    # Duration Scheduler
    "DurationScheduler",
    "ScheduledTask",
    "TaskDurationRecord",
    "can_schedule_task",
    "estimate_task_duration",
    "get_resource_availability",
    "get_scheduler",
    "record_task_completion",
    "register_running_task",
    "reset_scheduler",
    # Host Health Policy
    "HealthStatus",
    "check_host_health",
    "clear_health_cache",
    "get_health_summary",
    "get_healthy_hosts",
    "is_host_healthy",
    "mark_host_unhealthy",
    "pre_spawn_check",
    # Unified Health Manager
    "ErrorRecord",
    "ErrorSeverity",
    "HealthStats",
    "JobHealthState",
    "NodeHealthState",
    "PipelineState",
    "RecoveryAction",
    "RecoveryAttempt",
    "RecoveryConfig",
    "RecoveryEvent",
    "RecoveryResult",
    "RecoveryStatus",
    "SystemHealthConfig",
    "SystemHealthLevel",
    "SystemHealthScore",
    "UnifiedHealthManager",
    "get_health_manager",
    "get_system_health_level",
    "get_system_health_score",
    "is_component_healthy",
    "reset_health_manager",
    "should_pause_pipeline",
    "wire_health_events",
    # Health Facade
    "HealthCheckOrchestrator",
    "NodeHealthDetails",
    "get_cluster_health_summary",
    "get_degraded_nodes",
    "get_health_orchestrator",
    "get_healthy_nodes",
    "get_node_health",
    "get_node_health_monitor",
    "get_offline_nodes",
    "get_unhealthy_nodes",
    "mark_node_retired",
    # Ephemeral Data Guard
    "EphemeralDataGuard",
    "HostCheckpoint",
    "WriteThrough",
    "checkpoint_games",
    "ephemeral_heartbeat",
    "get_ephemeral_guard",
    "get_evacuation_candidates",
    "is_host_ephemeral",
    "queue_critical_game",
    "request_evacuation",
    "reset_ephemeral_guard",
    # P2P Backend
    "P2PBackend",
    "P2PNodeInfo",
    "discover_p2p_leader_url",
    "get_p2p_backend",
    # Cluster Transport
    "CircuitBreaker",
    "CircuitState",
    "ClusterTransport",
    "NodeConfig",
    "TransportResult",
    "get_cluster_transport",
    # Job Scheduler
    "JobPriority",
    "PriorityJobScheduler",
    "ScheduledJob",
    "get_config_game_counts",
    "get_cpu_rich_hosts",
    "get_gpu_rich_hosts",
    "get_job_scheduler",
    "get_underserved_configs",
    "reset_job_scheduler",
    "select_curriculum_config",
]
