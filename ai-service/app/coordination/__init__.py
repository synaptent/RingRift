"""Cluster coordination package for unified resource management.

Provides centralized task coordination to prevent uncontrolled task spawning.

Two modules are available:
1. cluster_lock - Legacy coordination via file locks and JSON registry
2. task_coordinator - Enhanced coordination with SQLite, rate limiting, backpressure

Usage:
    # Simple check before spawning
    from app.coordination import can_spawn_task
    if can_spawn_task(host="node-1", task_type="selfplay"):
        # spawn task

    # Enhanced coordinator
    from app.coordination import TaskCoordinator, TaskType
    coordinator = TaskCoordinator.get_instance()
    if coordinator.can_spawn_task(TaskType.SELFPLAY, "node-1")[0]:
        coordinator.register_task(task_id, TaskType.SELFPLAY, "node-1")
"""

# Legacy cluster_lock exports
from app.coordination.cluster_lock import (
    acquire_orchestrator_lock,
    release_orchestrator_lock,
    can_spawn_task,
    register_task,
    unregister_task,
    cleanup_stale_tasks,
    get_cluster_status,
    get_host_load,
    emergency_cluster_halt,
    check_emergency_halt,
    clear_emergency_halt,
    MAX_LOAD_THRESHOLD,
    MAX_PROCESSES_PER_HOST,
)

# Enhanced task_coordinator exports
from app.coordination.task_coordinator import (
    TaskCoordinator,
    TaskType,
    TaskLimits,
    TaskInfo as EnhancedTaskInfo,
    CoordinatorState,
    OrchestratorLock,
    RateLimiter,
    CoordinatedTask,
    get_coordinator,
    can_spawn as can_spawn_enhanced,
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

# Health check exports
from app.coordination.health_check import (
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

__all__ = [
    # Legacy
    "acquire_orchestrator_lock",
    "release_orchestrator_lock",
    "can_spawn_task",
    "register_task",
    "unregister_task",
    "cleanup_stale_tasks",
    "get_cluster_status",
    "get_host_load",
    "emergency_cluster_halt",
    "check_emergency_halt",
    "clear_emergency_halt",
    "MAX_LOAD_THRESHOLD",
    "MAX_PROCESSES_PER_HOST",
    # Enhanced
    "TaskCoordinator",
    "TaskType",
    "TaskLimits",
    "EnhancedTaskInfo",
    "CoordinatorState",
    "OrchestratorLock",
    "RateLimiter",
    "CoordinatedTask",
    "get_coordinator",
    "can_spawn_enhanced",
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
]
