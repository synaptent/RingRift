"""Utility and helper exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates utilities, helpers, tracing, and base classes.
"""

# Async Bridge Manager (shared executor pool)
from app.coordination.async_bridge_manager import (
    AsyncBridgeManager,
    get_bridge_manager,
    get_shared_executor,
    reset_bridge_manager,
    run_in_bridge_pool,
)

# Coordinator Base (common patterns for coordinators/managers)
from app.coordination.coordinator_base import (
    CallbackMixin,
    CoordinatorBase,
    CoordinatorProtocol,
    CoordinatorStats,
    CoordinatorStatus,
    SingletonMixin,
    SQLitePersistenceMixin,
    is_coordinator,
)

# Coordinator Configuration
from app.coordination.coordinator_config import (
    CacheConfig,
    CoordinatorConfig,
    EventBusConfig,
    HandlerResilienceConfig,
    HeartbeatConfig,
    MetricsConfig,
    OptimizationConfig,
    PipelineConfig,
    ResourceConfig,
    SelfplayConfig,
    TaskLifecycleConfig,
    get_config,
    reset_config,
    set_config,
    update_config,
    validate_config,
)

# Coordinator Persistence Layer (state snapshots and recovery)
from app.coordination.coordinator_persistence import (
    SnapshotCoordinator,
    StatePersistable,
    StatePersistenceMixin,
    StateSerializer,
    StateSnapshot,
    get_snapshot_coordinator,
    reset_snapshot_coordinator,
)

# Distributed Locking (Redis + file-based fallback)
from app.coordination.distributed_lock import (
    DistributedLock,
    acquire_training_lock,
    release_training_lock,
    training_lock,
)

# Dynamic Threshold Adjustment (adaptive thresholds)
from app.coordination.dynamic_thresholds import (
    AdjustmentStrategy,
    DynamicThreshold,
    ThresholdManager,
    ThresholdObservation,
    get_threshold_manager,
    reset_threshold_manager,
)

# Distributed Tracing
from app.coordination.tracing import (
    TraceCollector,
    TraceContext,
    TraceSpan,
    collect_trace,
    extract_trace_from_event,
    extract_trace_from_headers,
    get_trace_collector,
    get_trace_context,
    get_trace_id,
    inject_trace_into_event,
    inject_trace_into_headers,
    new_trace,
    set_trace_id,
    span,
    traced,
    with_trace,
)

# Task Decorators (lifecycle management)
from app.coordination.task_decorators import (
    TaskContext,
    coordinate_async_task,
    coordinate_task,
    get_current_task_context,
    task_context,
)

# Transaction isolation exports (ACID-like guarantees for merge operations)
from app.coordination.transaction_isolation import (
    MergeOperation,
    MergeTransaction,
    TransactionIsolation,
    TransactionState,
    add_merge_operation,
    begin_merge_transaction,
    commit_merge_transaction,
    complete_merge_operation,
    get_transaction_isolation,
    get_transaction_stats,
    merge_transaction,
    reset_transaction_isolation,
    rollback_merge_transaction,
)

# Coordination Utilities (reusable base classes)
from app.coordination.utils import (
    BoundedHistory,
    CallbackRegistry,
    HistoryEntry,
    MetricsAccumulator,
)

# Coordination helpers (safe wrappers to reduce duplicate try/except imports)
from app.coordination.helpers import (
    ack_event_safe,
    acquire_role_safe,
    acquire_sync_lock_safe,
    can_schedule_task_safe,
    can_spawn_safe,
    check_spawn_allowed,
    complete_task_safe,
    estimate_duration_safe,
    fail_task_safe,
    get_bandwidth_context,
    get_cluster_summary_safe,
    get_coordinator_safe,
    get_current_node_id,
    get_event_poller_class,
    get_host_targets_safe,
    get_orchestrator_roles,
    get_queue_types,
    get_registry_safe,
    get_resource_targets_safe,
    get_role_holder,
    get_safeguards,
    get_sync_lock_context,
    get_task_types,
    get_throttle_factor_safe,
    get_transfer_priorities,
    has_bandwidth_manager,
    has_coordination,
    has_cross_process_events,
    has_duration_scheduler,
    has_resource_targets,
    has_role,
    has_sync_lock,
    is_unified_loop_running,
    poll_events_safe,
    publish_event_safe,
    record_task_completion_safe,
    register_running_task_safe,
    register_task_safe,
    release_bandwidth_safe,
    release_role_safe,
    release_sync_lock_safe,
    report_queue_depth_safe,
    request_bandwidth_safe,
    set_backpressure_safe,
    should_scale_down_safe,
    should_scale_up_safe,
    should_stop_safe,
    should_throttle_safe,
    subscribe_process_safe,
    warn_if_orchestrator_running,
)

# Work Distribution (cluster work queue integration)
from app.coordination.work_distributor import (
    DistributedWorkConfig,
    WorkDistributor,
    distribute_evaluation,
    distribute_selfplay,
    distribute_training,
    get_work_distributor,
)

# Unified Registry
from app.coordination.unified_registry import (
    ClusterHealth,
    UnifiedRegistry,
    get_unified_registry,
    reset_unified_registry,
)

__all__ = [
    # Async Bridge
    "AsyncBridgeManager",
    "get_bridge_manager",
    "get_shared_executor",
    "reset_bridge_manager",
    "run_in_bridge_pool",
    # Coordinator Base
    "CallbackMixin",
    "CoordinatorBase",
    "CoordinatorProtocol",
    "CoordinatorStats",
    "CoordinatorStatus",
    "SingletonMixin",
    "SQLitePersistenceMixin",
    "is_coordinator",
    # Coordinator Config
    "CacheConfig",
    "CoordinatorConfig",
    "EventBusConfig",
    "HandlerResilienceConfig",
    "HeartbeatConfig",
    "MetricsConfig",
    "OptimizationConfig",
    "PipelineConfig",
    "ResourceConfig",
    "SelfplayConfig",
    "TaskLifecycleConfig",
    "get_config",
    "reset_config",
    "set_config",
    "update_config",
    "validate_config",
    # Coordinator Persistence
    "SnapshotCoordinator",
    "StatePersistable",
    "StatePersistenceMixin",
    "StateSerializer",
    "StateSnapshot",
    "get_snapshot_coordinator",
    "reset_snapshot_coordinator",
    # Distributed Lock
    "DistributedLock",
    "acquire_training_lock",
    "release_training_lock",
    "training_lock",
    # Dynamic Thresholds
    "AdjustmentStrategy",
    "DynamicThreshold",
    "ThresholdManager",
    "ThresholdObservation",
    "get_threshold_manager",
    "reset_threshold_manager",
    # Tracing
    "TraceCollector",
    "TraceContext",
    "TraceSpan",
    "collect_trace",
    "extract_trace_from_event",
    "extract_trace_from_headers",
    "get_trace_collector",
    "get_trace_context",
    "get_trace_id",
    "inject_trace_into_event",
    "inject_trace_into_headers",
    "new_trace",
    "set_trace_id",
    "span",
    "traced",
    "with_trace",
    # Task Decorators
    "TaskContext",
    "coordinate_async_task",
    "coordinate_task",
    "get_current_task_context",
    "task_context",
    # Transaction Isolation
    "MergeOperation",
    "MergeTransaction",
    "TransactionIsolation",
    "TransactionState",
    "add_merge_operation",
    "begin_merge_transaction",
    "commit_merge_transaction",
    "complete_merge_operation",
    "get_transaction_isolation",
    "get_transaction_stats",
    "merge_transaction",
    "reset_transaction_isolation",
    "rollback_merge_transaction",
    # Utils
    "BoundedHistory",
    "CallbackRegistry",
    "HistoryEntry",
    "MetricsAccumulator",
    # Helpers
    "ack_event_safe",
    "acquire_role_safe",
    "acquire_sync_lock_safe",
    "can_schedule_task_safe",
    "can_spawn_safe",
    "check_spawn_allowed",
    "complete_task_safe",
    "estimate_duration_safe",
    "fail_task_safe",
    "get_bandwidth_context",
    "get_cluster_summary_safe",
    "get_coordinator_safe",
    "get_current_node_id",
    "get_event_poller_class",
    "get_host_targets_safe",
    "get_orchestrator_roles",
    "get_queue_types",
    "get_registry_safe",
    "get_resource_targets_safe",
    "get_role_holder",
    "get_safeguards",
    "get_sync_lock_context",
    "get_task_types",
    "get_throttle_factor_safe",
    "get_transfer_priorities",
    "has_bandwidth_manager",
    "has_coordination",
    "has_cross_process_events",
    "has_duration_scheduler",
    "has_resource_targets",
    "has_role",
    "has_sync_lock",
    "is_unified_loop_running",
    "poll_events_safe",
    "publish_event_safe",
    "record_task_completion_safe",
    "register_running_task_safe",
    "register_task_safe",
    "release_bandwidth_safe",
    "release_role_safe",
    "release_sync_lock_safe",
    "report_queue_depth_safe",
    "request_bandwidth_safe",
    "set_backpressure_safe",
    "should_scale_down_safe",
    "should_scale_up_safe",
    "should_stop_safe",
    "should_throttle_safe",
    "subscribe_process_safe",
    "warn_if_orchestrator_running",
    # Work Distribution
    "ClusterHealth",
    "DistributedWorkConfig",
    "UnifiedRegistry",
    "WorkDistributor",
    "distribute_evaluation",
    "distribute_selfplay",
    "distribute_training",
    "get_unified_registry",
    "get_work_distributor",
    "reset_unified_registry",
]
