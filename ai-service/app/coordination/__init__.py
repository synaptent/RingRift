"""Cluster coordination package for unified resource management.

Provides centralized task coordination to prevent uncontrolled task spawning.

Primary modules:
1. task_coordinator - SQLite-backed coordination with rate limiting, backpressure
2. orchestrator_registry - Role-based mutual exclusion with heartbeat liveness
3. safeguards - Circuit breakers, resource monitoring, spawn rate tracking
4. queue_monitor - Queue depth monitoring with backpressure signals
5. bandwidth_manager - Network bandwidth allocation for transfers
6. sync_mutex - Cross-process mutex for rsync operations

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
)

# Safeguards exports
from app.coordination.safeguards import (
    Safeguards,
    SafeguardConfig,
    CircuitBreaker,
    CircuitState,
    SpawnRateTracker,
    ResourceMonitor,
    check_before_spawn,
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
]
