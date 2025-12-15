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
]
