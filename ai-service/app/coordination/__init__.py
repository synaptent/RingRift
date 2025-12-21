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
# Bandwidth manager exports
from app.coordination.bandwidth_manager import (
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

# Cross-process event queue exports
from app.coordination.cross_process_events import (
    CrossProcessEvent,
    CrossProcessEventPoller,
    CrossProcessEventQueue,
    ack_event,
    ack_events,
    bridge_to_cross_process,
    get_event_queue,
    poll_events,
    publish_event,
    reset_event_queue,
    subscribe_process,
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

# Host health policy exports (renamed from health_check.py for clarity)
from app.coordination.host_health_policy import (
    HealthStatus,
    check_host_health,
    clear_health_cache,
    get_health_summary,
    get_healthy_hosts,
    is_host_healthy,
    mark_host_unhealthy,
    pre_spawn_check,
)

# Orchestrator registry exports
from app.coordination.orchestrator_registry import (
    OrchestratorInfo,
    OrchestratorRegistry,
    OrchestratorRole,
    OrchestratorState,
    acquire_orchestrator_role,
    get_registry,
    is_orchestrator_role_available,
    orchestrator_role,
    release_orchestrator_role,
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
    # ResourceType already imported from task_coordinator above
    ScaleAction,
    get_cluster_utilization,
    # Note: should_scale_up/down/record_utilization also exist here
    # but we prefer the resource_targets versions for per-host decisions
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

# Sync SCHEDULER exports (unified cluster-wide data sync SCHEDULING)
# Note: This is the SCHEDULING layer - decides WHEN/WHAT to sync.
# For EXECUTION (HOW to sync), use app.distributed.sync_coordinator.SyncCoordinator
from app.coordination.sync_coordinator import (
    ClusterDataStatus,
    # Data types
    HostDataState,
    HostType,
    SyncAction,
    SyncPriority,
    SyncRecommendation,
    # Canonical class name (December 2025)
    SyncScheduler,
    execute_priority_sync,  # Bridge to distributed layer execution
    # Functions
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
from app.coordination.task_coordinator import (
    TASK_RESOURCE_MAP,
    CoordinatedTask,
    CoordinatorState,
    OrchestratorLock,
    RateLimiter,
    # Resource-aware task classification
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

# Circuit breaker - canonical location is app.distributed.circuit_breaker
from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)

# Re-export distributed layer sync execution for convenience
# These provide the actual sync transport (aria2, SSH, P2P)
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

# Cluster transport layer (unified multi-transport communication)
# Async Bridge Manager (December 2025 - shared executor pool)
import contextlib

from app.coordination.async_bridge_manager import (
    AsyncBridgeManager,
    get_bridge_manager,
    get_shared_executor,
    reset_bridge_manager,
    run_in_bridge_pool,
)

# Async Training Bridge (async wrapper + event integration)
from app.coordination.async_training_bridge import (
    AsyncTrainingBridge,
    TrainingProgressEvent,
    async_can_train,
    async_complete_training,
    async_get_training_status,
    async_request_training,
    async_update_progress,
    get_training_bridge,
    reset_training_bridge,
)

# CacheCoordinationOrchestrator - unified cache management
from app.coordination.cache_coordination_orchestrator import (
    CacheCoordinationOrchestrator,
    CacheEntry,
    CacheStats,
    CacheStatus,
    CacheType,
    NodeCacheState,
    get_cache_orchestrator,
    invalidate_model_caches,
    register_cache,
    wire_cache_events,
)
from app.coordination.cluster_transport import (
    ClusterTransport,
    # CircuitBreaker already imported from app.distributed.circuit_breaker above
    NodeConfig,
    TransportResult,
    get_cluster_transport,
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

# Coordinator Configuration (December 2025 - centralized config)
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

# Coordinator Persistence Layer (December 2025 - state snapshots and recovery)
from app.coordination.coordinator_persistence import (
    SnapshotCoordinator,
    StatePersistable,
    StatePersistenceMixin,
    StateSerializer,
    StateSnapshot,
    get_snapshot_coordinator,
    reset_snapshot_coordinator,
)

# DaemonManager - unified lifecycle management for all background services (December 2025)
from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
    setup_signal_handlers,
)

# DataPipelineOrchestrator - unified pipeline stage coordination
from app.coordination.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    IterationRecord,
    PipelineStage,
    PipelineStats,
    StageTransition,
    get_current_pipeline_stage,
    get_pipeline_orchestrator,
    get_pipeline_status,
    wire_pipeline_events,
)

# Distributed Locking (Redis + file-based fallback)
from app.coordination.distributed_lock import (
    DistributedLock,
    acquire_training_lock,
    release_training_lock,
    training_lock,
)

# Dynamic Threshold Adjustment (December 2025 - adaptive thresholds)
from app.coordination.dynamic_thresholds import (
    AdjustmentStrategy,
    DynamicThreshold,
    ThresholdManager,
    ThresholdObservation,
    get_threshold_manager,
    reset_threshold_manager,
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

# Event Emitters (December 2025 - centralized event emission)
from app.coordination.event_emitters import (
    emit_backpressure_activated,
    emit_backpressure_released,
    emit_cache_invalidated,
    emit_evaluation_complete,
    emit_host_offline,
    emit_host_online,
    emit_node_recovered,
    # New emitters (December 2025)
    emit_optimization_triggered,
    emit_plateau_detected,
    emit_promotion_complete,
    emit_quality_updated,
    emit_regression_detected,
    emit_selfplay_complete,
    emit_sync_complete,
    emit_task_complete,
    emit_training_complete,
    emit_training_complete_sync,
    emit_training_started,
)

# Unified Event Router (December 2025 - single API for all event systems)
# Note: EventRouter provides the publishing API, EventCoordinator provides bridging daemon
from app.coordination.event_router import (
    EventSource,
    RouterEvent,
    UnifiedEventRouter,
    get_router as get_event_router,
    publish as router_publish_event,  # Alias to avoid collision with cross_process_events.publish_event
    publish_sync as publish_event_sync,
    reset_router as reset_event_router,
    subscribe as subscribe_event,
    unsubscribe as unsubscribe_event,
)

# Integration Bridge (December 2025 - C2 consolidation)
# Wires integration modules (model_lifecycle, p2p_integration, pipeline_feedback)
# to the unified event router for automatic event propagation
from app.coordination.integration_bridge import (
    reset_integration_wiring,
    wire_all_integrations,
    wire_all_integrations_sync,
    wire_model_lifecycle_events,
    wire_p2p_integration_events,
    wire_pipeline_feedback_events,
)

# Coordination helpers (safe wrappers to reduce duplicate try/except imports)
from app.coordination.helpers import (
    ack_event_safe,
    acquire_role_safe,
    acquire_sync_lock_safe,
    # Duration scheduling functions
    can_schedule_task_safe,
    can_spawn_safe,
    # Safeguard functions
    check_spawn_allowed,
    complete_task_safe,
    estimate_duration_safe,
    fail_task_safe,
    get_bandwidth_context,
    get_cluster_summary_safe,
    # Coordinator functions
    get_coordinator_safe,
    # Convenience functions
    get_current_node_id,
    get_event_poller_class,
    get_host_targets_safe,
    get_orchestrator_roles,
    get_queue_types,
    # Orchestrator role functions
    get_registry_safe,
    # Resource targets functions
    get_resource_targets_safe,
    get_role_holder,
    get_safeguards,
    # Sync mutex functions
    get_sync_lock_context,
    # Type getters
    get_task_types,
    get_throttle_factor_safe,
    get_transfer_priorities,
    has_bandwidth_manager,
    # Availability checks
    has_coordination,
    has_cross_process_events,
    has_duration_scheduler,
    has_resource_targets,
    has_role,
    has_sync_lock,
    is_unified_loop_running,
    poll_events_safe,
    # Cross-process events functions
    publish_event_safe,
    record_task_completion_safe,
    register_running_task_safe,
    register_task_safe,
    release_bandwidth_safe,
    release_role_safe,
    release_sync_lock_safe,
    report_queue_depth_safe,
    # Bandwidth functions
    request_bandwidth_safe,
    set_backpressure_safe,
    should_scale_down_safe,
    should_scale_up_safe,
    should_stop_safe,
    # Queue backpressure functions
    should_throttle_safe,
    subscribe_process_safe,
    warn_if_orchestrator_running,
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

# MetricsAnalysisOrchestrator - unified metrics analysis
from app.coordination.metrics_analysis_orchestrator import (
    AnalysisResult,
    MetricsAnalysisOrchestrator,
    MetricTracker,
    MetricType,
    analyze_metrics,
    get_metrics_orchestrator,
    record_metric,
    wire_metrics_events,
)

# Model Lifecycle Coordinator (December 2025)
from app.coordination.model_lifecycle_coordinator import (
    CacheEntry as ModelCacheEntry,
    CheckpointInfo,
    ModelLifecycleCoordinator,
    ModelLifecycleStats,
    ModelRecord,
    ModelState,
    get_model_coordinator,
    get_production_elo,
    get_production_model_id,
    wire_model_events,
)

# UnifiedHealthManager - consolidated error recovery and health management (December 2025)
from app.coordination.unified_health_manager import (
    ErrorRecord,
    ErrorSeverity,
    HealthStats,
    JobHealthState,
    NodeHealthState,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryConfig,
    RecoveryEvent,
    RecoveryResult,
    RecoveryStatus,
    UnifiedHealthManager,
    get_health_manager,
    is_component_healthy,
    reset_health_manager,
    wire_health_events,
)

# OptimizationCoordinator - unified optimization management
from app.coordination.optimization_coordinator import (
    OptimizationCoordinator,
    OptimizationRun,
    OptimizationStats,
    OptimizationType,
    get_optimization_coordinator,
    get_optimization_stats,
    trigger_cmaes,
    trigger_nas,
    wire_optimization_events,
)

# Cross-Coordinator Health Protocol (December 2025)
from app.coordination.orchestrator_registry import (
    CoordinatorHealth,
    CrossCoordinatorHealthProtocol,
    auto_register_known_coordinators,
    check_cluster_health,
    get_coordinator as get_registered_coordinator,  # Alias to avoid collision with task_coordinator.get_coordinator
    get_cross_coordinator_health,
    get_registered_coordinators,
    # Coordinator Registration (December 2025)
    register_coordinator,
    shutdown_all_coordinators as shutdown_registered_coordinators,  # Alias to avoid collision
    unregister_coordinator,
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

# ResourceMonitoringCoordinator - unified resource monitoring
from app.coordination.resource_monitoring_coordinator import (
    NodeResourceState,
    ResourceAlert,
    ResourceMonitoringCoordinator,
    ResourceStats,
    check_resource_thresholds,
    get_resource_coordinator,
    update_node_resources,
    wire_resource_events,
)

# =============================================================================
# Orchestrator Imports (December 2025 - event-driven coordination)
# =============================================================================
# SelfplayOrchestrator - unified selfplay event coordination
from app.coordination.selfplay_orchestrator import (
    SelfplayOrchestrator,
    SelfplayStats,
    SelfplayTaskInfo,
    SelfplayType,
    emit_selfplay_completion,
    get_selfplay_orchestrator,
    get_selfplay_stats,
    wire_selfplay_events,
)

# Stage Events exports (event-driven pipeline orchestration)
from app.coordination.stage_events import (
    StageCompletionCallback,
    StageCompletionResult,
    StageEvent,
    StageEventBus,
    create_pipeline_callbacks,
    get_event_bus as get_stage_event_bus,
    register_standard_callbacks,
    reset_event_bus as reset_stage_event_bus,
)

# Task Decorators (December 2025 - lifecycle management)
from app.coordination.task_decorators import (
    TaskContext,
    coordinate_async_task,
    coordinate_task,
    get_current_task_context,
    task_context,
)

# TaskLifecycleCoordinator - unified task event monitoring
from app.coordination.task_lifecycle_coordinator import (
    TaskLifecycleCoordinator,
    TaskLifecycleStats,
    TaskStatus,
    TrackedTask,
    get_active_task_count,
    get_task_lifecycle_coordinator,
    get_task_stats,
    wire_task_events,
)

# Distributed Tracing (December 2025)
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

# Training Coordination (cluster-wide training management)
from app.coordination.training_coordinator import (
    TrainingCoordinator,
    TrainingJob,
    can_train,
    get_training_coordinator,
    get_training_status,
    release_training_slot,
    request_training_slot,
    training_slot,
    update_training_progress,
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

# Unified Event Coordinator (December 2025 - bridges all event systems)
from app.coordination.unified_event_coordinator import (
    CoordinatorStats as EventCoordinatorStats,
    UnifiedEventCoordinator,
    get_coordinator_stats as get_event_coordinator_stats,
    get_event_coordinator,
    start_coordinator as start_event_coordinator,
    stop_coordinator as stop_event_coordinator,
)

# Unified Registry (December 2025 - registry facade)
from app.coordination.unified_registry import (
    ClusterHealth,
    UnifiedRegistry,
    get_unified_registry,
    reset_unified_registry,
)

# Coordination Utilities (December 2025 - reusable base classes)
from app.coordination.utils import (
    BoundedHistory,
    CallbackRegistry,
    HistoryEntry,
    MetricsAccumulator,
)

# Module-level singleton placeholders for cleanup in shutdown_all_coordinators
_selfplay_orchestrator = None
_pipeline_orchestrator = None
_task_lifecycle_coordinator = None
_optimization_coordinator = None
_metrics_orchestrator = None
_resource_coordinator = None
_cache_orchestrator = None
_event_coordinator = None


def _init_with_retry(
    name: str,
    init_func,
    max_retries: int = 3,
    base_delay: float = 0.5,
    logger=None,
) -> tuple:
    """Initialize a coordinator with retry logic.

    Args:
        name: Coordinator name for logging
        init_func: Function that returns (instance, subscribed_flag)
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        logger: Logger instance

    Returns:
        (instance, success, error_message)
    """
    import time as _time

    last_error = None

    for attempt in range(max_retries):
        try:
            instance, subscribed = init_func()

            if not subscribed:
                raise RuntimeError(f"{name} failed to subscribe to events")

            if logger:
                if attempt > 0:
                    logger.info(f"[init_with_retry] {name} succeeded on attempt {attempt + 1}")
                else:
                    logger.info(f"[initialize_all_coordinators] {name} wired")

            return (instance, True, None)

        except Exception as e:
            last_error = str(e)
            if logger:
                logger.warning(
                    f"[init_with_retry] {name} attempt {attempt + 1}/{max_retries} failed: {e}"
                )

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                _time.sleep(delay)

    if logger:
        logger.error(f"[initialize_all_coordinators] {name} failed after {max_retries} attempts")

    return (None, False, last_error)


def initialize_all_coordinators(
    auto_trigger_pipeline: bool = False,
    heartbeat_threshold: float = 60.0,
    max_retries: int = 3,
    retry_delay: float = 0.5,
    wrap_handlers: bool = True,
) -> dict:
    """Initialize all orchestrators and coordinators with event wiring (December 2025).

    This is the single entry point to bootstrap all coordination infrastructure.
    It wires all event subscriptions and returns a status dictionary.

    Features:
    - Retry logic with exponential backoff for failed subscriptions
    - Validation that subscriptions actually succeeded
    - Emits COORDINATOR_INIT_FAILED for persistent failures
    - Optionally wraps handlers with resilience (exception boundaries + timeouts)

    Args:
        auto_trigger_pipeline: If True, pipeline stages auto-trigger downstream
        heartbeat_threshold: Seconds without heartbeat to mark tasks orphaned
        max_retries: Maximum retry attempts per coordinator
        retry_delay: Base delay for exponential backoff
        wrap_handlers: If True, wrap handlers with resilience

    Returns:
        Dict with initialization status for each orchestrator
    """
    import asyncio
    import logging
    logger = logging.getLogger(__name__)

    status = {
        "selfplay": False,
        "pipeline": False,
        "task_lifecycle": False,
        "optimization": False,
        "metrics": False,
        "resources": False,
        "cache": False,
        "event_coordinator": False,
    }
    errors = {}
    instances = {}

    # Define init functions that return (instance, subscribed)
    def init_task_lifecycle():
        coord = wire_task_events(heartbeat_threshold=heartbeat_threshold)
        return (coord, coord._subscribed)

    def init_resources():
        coord = wire_resource_events()
        return (coord, coord._subscribed)

    def init_cache():
        coord = wire_cache_events()
        return (coord, coord._subscribed)

    def init_selfplay():
        coord = wire_selfplay_events()
        return (coord, coord._subscribed)

    def init_pipeline():
        coord = wire_pipeline_events(auto_trigger=auto_trigger_pipeline)
        return (coord, coord._subscribed)

    def init_optimization():
        coord = wire_optimization_events()
        return (coord, coord._subscribed)

    def init_metrics():
        coord = wire_metrics_events()
        return (coord, coord._subscribed)

    # Initialize in dependency order (foundational first)
    init_order = [
        ("task_lifecycle", init_task_lifecycle),
        ("resources", init_resources),
        ("cache", init_cache),
        ("selfplay", init_selfplay),
        ("pipeline", init_pipeline),
        ("optimization", init_optimization),
        ("metrics", init_metrics),
    ]

    for name, init_func in init_order:
        instance, success, error = _init_with_retry(
            name,
            init_func,
            max_retries=max_retries,
            base_delay=retry_delay,
            logger=logger,
        )
        status[name] = success
        if instance:
            instances[name] = instance
        if error:
            errors[name] = error

    # Wrap handlers with resilience if requested
    if wrap_handlers:
        try:
            from app.coordination.handler_resilience import make_handlers_resilient

            for name, instance in instances.items():
                make_handlers_resilient(instance, name)
            logger.debug("[initialize_all_coordinators] Wrapped handlers with resilience")
        except ImportError:
            logger.debug("[initialize_all_coordinators] handler_resilience not available")

    # Start UnifiedEventCoordinator
    try:
        stats = get_event_coordinator_stats()
        if not stats.get("is_running", False):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(start_event_coordinator())
                status["event_coordinator"] = True
            except RuntimeError:
                status["event_coordinator"] = asyncio.run(start_event_coordinator())
        else:
            status["event_coordinator"] = True
        logger.info("[initialize_all_coordinators] UnifiedEventCoordinator started")
    except Exception as e:
        logger.error(f"[initialize_all_coordinators] UnifiedEventCoordinator failed: {e}")
        errors["event_coordinator"] = str(e)

    # Emit COORDINATOR_INIT_FAILED for any failures (best effort)
    if errors:
        try:
            import time as _time

            from app.distributed.data_events import DataEvent, DataEventType, get_event_bus

            bus = get_event_bus()
            for name, error in errors.items():
                event = DataEvent(
                    event_type=DataEventType.COORDINATOR_INIT_FAILED,
                    payload={
                        "coordinator_name": name,
                        "error": error,
                        "timestamp": _time.time(),
                    },
                    source="initialize_all_coordinators",
                )
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(bus.publish(event))
                except RuntimeError:
                    asyncio.run(bus.publish(event))
        except Exception:
            pass

    # Log summary
    wired_count = sum(1 for k, v in status.items() if v and not k.startswith("_"))
    total_count = len([k for k in status if not k.startswith("_")])

    if wired_count == total_count:
        logger.info(
            f"[initialize_all_coordinators] All {total_count} orchestrators/coordinators initialized"
        )
    else:
        logger.warning(
            f"[initialize_all_coordinators] Initialized {wired_count}/{total_count} "
            f"orchestrators/coordinators. Failed: {list(errors.keys())}"
        )

    status["_errors"] = errors
    status["_instances"] = list(instances.keys())

    return status


def get_all_coordinator_status() -> dict:
    """Get unified status from all orchestrators and coordinators.

    Returns:
        Dict with status from each orchestrator
    """
    return {
        "selfplay": get_selfplay_orchestrator().get_status(),
        "pipeline": get_pipeline_orchestrator().get_status(),
        "task_lifecycle": get_task_lifecycle_coordinator().get_status(),
        "optimization": get_optimization_coordinator().get_status(),
        "metrics": get_metrics_orchestrator().get_status(),
        "resources": get_resource_coordinator().get_status(),
        "cache": get_cache_orchestrator().get_status(),
        "event_coordinator": get_event_coordinator_stats(),
    }


def get_system_health() -> dict:
    """Get aggregated system health from all coordinators (December 2025).

    This function provides a comprehensive health assessment of the entire
    coordination system, including:
    - Overall health score (0.0-1.0)
    - Per-coordinator health status
    - Critical issues that need attention
    - Handler health from resilience mixin

    Returns:
        Dict with health information:
        {
            "overall_health": 0.95,
            "status": "healthy" | "degraded" | "unhealthy",
            "coordinators": {...per-coordinator health...},
            "issues": [...list of critical issues...],
            "handler_health": {...handler metrics...},
        }
    """
    import time as _time

    issues = []
    coordinator_health = {}
    total_score = 0.0
    coordinator_count = 0

    def _get_health_score(name: str, status: dict) -> float:
        """Calculate health score for a coordinator."""
        score = 1.0
        # Note: 'issues' list is mutated via append(), no nonlocal needed

        # Check subscription status
        if not status.get("subscribed", True):
            score -= 0.3
            issues.append(f"{name}: not subscribed to events")

        # Check for paused state (pipeline)
        if status.get("paused", False):
            score -= 0.2
            issues.append(f"{name}: paused ({status.get('pause_reason', 'unknown')})")

        # Check for resource constraints
        if status.get("resource_constraints"):
            for constraint_type, constraint in status.get("resource_constraints", {}).items():
                if isinstance(constraint, dict) and constraint.get("severity") == "critical":
                    score -= 0.2
                    issues.append(f"{name}: critical {constraint_type} constraint")

        # Check for backpressure
        if status.get("backpressure_active"):
            score -= 0.1
            issues.append(f"{name}: backpressure active")

        # Check for plateaus/regressions (metrics)
        if status.get("plateaus"):
            score -= 0.1 * min(len(status["plateaus"]), 3)
            for metric in status["plateaus"][:3]:
                issues.append(f"{name}: plateau detected in {metric}")

        if status.get("regressions"):
            score -= 0.2 * min(len(status["regressions"]), 2)
            for metric in status["regressions"][:2]:
                issues.append(f"{name}: regression detected in {metric}")

        # Check for orphaned tasks
        if status.get("orphaned", 0) > 0:
            orphan_count = status["orphaned"]
            score -= 0.1 * min(orphan_count, 5)
            issues.append(f"{name}: {orphan_count} orphaned tasks")

        # Check for failed tasks
        if status.get("failed_tasks", 0) > 10:
            score -= 0.1
            issues.append(f"{name}: high failure count ({status['failed_tasks']})")

        return max(0.0, score)

    # Gather coordinator statuses with error handling
    coordinators = [
        ("selfplay", get_selfplay_orchestrator),
        ("pipeline", get_pipeline_orchestrator),
        ("task_lifecycle", get_task_lifecycle_coordinator),
        ("optimization", get_optimization_coordinator),
        ("metrics", get_metrics_orchestrator),
        ("resources", get_resource_coordinator),
        ("cache", get_cache_orchestrator),
    ]

    for name, getter in coordinators:
        try:
            status = getter().get_status()
            coordinator_health[name] = _get_health_score(name, status)
            coordinator_count += 1
            total_score += coordinator_health[name]
        except Exception as e:
            coordinator_health[name] = 0.0
            issues.append(f"{name}: failed to get status ({e})")

    # Calculate overall health
    overall_health = total_score / coordinator_count if coordinator_count > 0 else 0.0

    # Determine status string
    if overall_health >= 0.9:
        status_str = "healthy"
    elif overall_health >= 0.7:
        status_str = "degraded"
    else:
        status_str = "unhealthy"

    # Get handler health from resilience module
    handler_health = {}
    try:
        from app.coordination.handler_resilience import get_all_handler_metrics

        all_metrics = get_all_handler_metrics()
        total_invocations = sum(m.invocation_count for m in all_metrics.values())
        total_failures = sum(m.failure_count for m in all_metrics.values())
        total_timeouts = sum(m.timeout_count for m in all_metrics.values())

        handler_health = {
            "total_handlers": len(all_metrics),
            "total_invocations": total_invocations,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "success_rate": (
                (total_invocations - total_failures - total_timeouts) / total_invocations
                if total_invocations > 0 else 1.0
            ),
            "unhealthy_handlers": [
                name for name, m in all_metrics.items()
                if m.consecutive_failures >= 3
            ],
        }

        if handler_health["unhealthy_handlers"]:
            for handler in handler_health["unhealthy_handlers"]:
                issues.append(f"handler: {handler} has consecutive failures")
    except Exception:
        pass

    return {
        "overall_health": round(overall_health, 3),
        "status": status_str,
        "coordinators": coordinator_health,
        "issues": issues[:20],  # Limit to top 20 issues
        "handler_health": handler_health,
        "timestamp": _time.time(),
    }


async def shutdown_all_coordinators(
    timeout_seconds: float = 30.0,
    emit_events: bool = True,
) -> dict:
    """Gracefully shutdown all coordinators (December 2025).

    This function performs coordinated shutdown of all coordinators:
    1. Emits COORDINATOR_SHUTDOWN events to notify subscribers
    2. Drains active operations (respecting timeout)
    3. Stops coordinators in reverse dependency order
    4. Cleans up resources

    Args:
        timeout_seconds: Maximum time to wait for graceful shutdown
        emit_events: Whether to emit shutdown events

    Returns:
        Dict with shutdown status for each coordinator
    """
    import asyncio
    import logging
    import time as _time

    logger = logging.getLogger(__name__)
    logger.info("[shutdown_all_coordinators] Starting graceful shutdown...")

    status = {}
    start_time = _time.time()

    # Emit shutdown events for each coordinator
    if emit_events:
        try:
            from app.coordination.event_emitters import emit_coordinator_shutdown

            coordinators = [
                "optimization", "metrics", "pipeline", "selfplay",
                "cache", "resources", "task_lifecycle",
            ]
            for coord_name in coordinators:
                with contextlib.suppress(Exception):
                    await emit_coordinator_shutdown(
                        coordinator_name=coord_name,
                        reason="system_shutdown",
                    )
        except ImportError:
            pass

    # Shutdown in reverse dependency order
    # (High-level coordinators first, foundational last)
    shutdown_order = [
        ("optimization", get_optimization_coordinator),
        ("metrics", get_metrics_orchestrator),
        ("pipeline", get_pipeline_orchestrator),
        ("selfplay", get_selfplay_orchestrator),
        ("cache", get_cache_orchestrator),
        ("resources", get_resource_coordinator),
        ("task_lifecycle", get_task_lifecycle_coordinator),
    ]

    async def _shutdown_coordinator(name: str, getter) -> tuple:
        """Shutdown a single coordinator with timeout."""
        try:
            coord = getter()

            # Check if coordinator has shutdown method
            if hasattr(coord, 'shutdown') and asyncio.iscoroutinefunction(coord.shutdown):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.shutdown(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            # If no shutdown method, try stop
            elif hasattr(coord, 'stop') and asyncio.iscoroutinefunction(coord.stop):
                remaining = timeout_seconds - (_time.time() - start_time)
                if remaining > 0:
                    await asyncio.wait_for(coord.stop(), timeout=remaining)
                    return (name, True, None)
                return (name, False, "timeout exceeded")

            # Coordinator has no lifecycle methods - just mark as done
            return (name, True, "no lifecycle methods")

        except asyncio.TimeoutError:
            return (name, False, "shutdown timed out")
        except Exception as e:
            return (name, False, str(e))

    # Shutdown each coordinator
    for name, getter in shutdown_order:
        result = await _shutdown_coordinator(name, getter)
        status[result[0]] = {
            "success": result[1],
            "error": result[2],
        }

        if result[1]:
            logger.info(f"[shutdown_all_coordinators] {name} shutdown complete")
        else:
            logger.warning(f"[shutdown_all_coordinators] {name} shutdown failed: {result[2]}")

    # Cleanup global singletons
    try:
        global _selfplay_orchestrator, _pipeline_orchestrator, _task_lifecycle_coordinator
        global _optimization_coordinator, _metrics_orchestrator, _resource_coordinator
        global _cache_orchestrator, _event_coordinator

        _selfplay_orchestrator = None
        _pipeline_orchestrator = None
        _task_lifecycle_coordinator = None
        _optimization_coordinator = None
        _metrics_orchestrator = None
        _resource_coordinator = None
        _cache_orchestrator = None
        _event_coordinator = None
    except NameError:
        pass

    # Reset handler metrics
    try:
        from app.coordination.handler_resilience import reset_handler_metrics
        reset_handler_metrics()
    except ImportError:
        pass

    # Reset dependency graph
    try:
        from app.coordination.coordinator_dependencies import reset_dependency_graph
        reset_dependency_graph()
    except ImportError:
        pass

    total_time = _time.time() - start_time
    success_count = sum(1 for s in status.values() if s["success"])

    logger.info(
        f"[shutdown_all_coordinators] Shutdown complete: {success_count}/{len(status)} "
        f"coordinators in {total_time:.2f}s"
    )

    return {
        "status": status,
        "total_time_seconds": round(total_time, 2),
        "success_count": success_count,
        "total_count": len(status),
    }


# =============================================================================
# Coordinator Heartbeat System (December 2025)
# =============================================================================

_heartbeat_task = None
_heartbeat_running = False


async def _emit_coordinator_heartbeats(interval_seconds: float = 30.0) -> None:
    """Background task to emit heartbeats from all coordinators.

    Emits COORDINATOR_HEARTBEAT events periodically to indicate
    each coordinator is alive and functioning.
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    _heartbeat_running = True
    logger.info(f"[HeartbeatManager] Started with {interval_seconds}s interval")

    while _heartbeat_running:
        try:
            from app.coordination.event_emitters import emit_coordinator_heartbeat

            # Gather health from each coordinator
            coordinators = [
                ("selfplay", get_selfplay_orchestrator),
                ("pipeline", get_pipeline_orchestrator),
                ("task_lifecycle", get_task_lifecycle_coordinator),
                ("optimization", get_optimization_coordinator),
                ("metrics", get_metrics_orchestrator),
                ("resources", get_resource_coordinator),
                ("cache", get_cache_orchestrator),
            ]

            for name, getter in coordinators:
                try:
                    coord = getter()
                    status = coord.get_status()

                    # Extract health indicators
                    health_score = 1.0
                    if not status.get("subscribed", True):
                        health_score = 0.5
                    if status.get("paused", False):
                        health_score = 0.7
                    if status.get("backpressure_active", False):
                        health_score = 0.6

                    await emit_coordinator_heartbeat(
                        coordinator_name=name,
                        health_score=health_score,
                        active_handlers=status.get("metrics_tracked", 0)
                        if name == "metrics" else status.get("active_tasks", 0),
                        events_processed=status.get("total_invocations", 0)
                        if "total_invocations" in status else 0,
                    )
                except Exception as e:
                    logger.debug(f"[HeartbeatManager] Failed to emit heartbeat for {name}: {e}")

        except ImportError:
            logger.debug("[HeartbeatManager] event_emitters not available")
        except Exception as e:
            logger.debug(f"[HeartbeatManager] Error in heartbeat loop: {e}")

        # Wait for next interval
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break

    logger.info("[HeartbeatManager] Stopped")


def start_coordinator_heartbeats(interval_seconds: float = 30.0) -> bool:
    """Start the coordinator heartbeat background task.

    Args:
        interval_seconds: Interval between heartbeat emissions

    Returns:
        True if started successfully
    """
    import asyncio

    global _heartbeat_task

    if _heartbeat_task is not None and not _heartbeat_task.done():
        return True  # Already running

    try:
        loop = asyncio.get_running_loop()
        _heartbeat_task = loop.create_task(
            _emit_coordinator_heartbeats(interval_seconds)
        )
        return True
    except RuntimeError:
        # No event loop running
        return False


def stop_coordinator_heartbeats() -> None:
    """Stop the coordinator heartbeat background task."""
    global _heartbeat_task, _heartbeat_running

    _heartbeat_running = False

    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        _heartbeat_task = None


def is_heartbeat_running() -> bool:
    """Check if heartbeat manager is running."""
    return _heartbeat_task is not None and not _heartbeat_task.done()


__all__ = [
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
    "AdjustmentStrategy",
    "AnalysisResult",
    "AsyncBridgeManager",
    "AsyncTrainingBridge",
    "BackpressureLevel",
    "BandwidthAllocation",
    "BandwidthManager",
    "BatchChecksum",
    "BoundedHistory",
    "CacheConfig",
    "CacheCoordinationOrchestrator",
    "CacheEntry",
    "CacheStats",
    "CacheStatus",
    "CacheType",
    "CallbackMixin",
    "CallbackRegistry",
    "CheckpointInfo",
    "CircuitBreaker",
    "CircuitState",
    "ClusterDataStatus",
    "ClusterHealth",
    "ClusterState",
    "ClusterSyncStats",
    "ClusterTransport",
    "CoordinatedTask",
    "CoordinationSyncCoordinator",
    "CoordinatorBase",
    "CoordinatorConfig",
    "CoordinatorHealth",
    "CoordinatorProtocol",
    "CoordinatorState",
    "CoordinatorStats",
    "CoordinatorStatus",
    "CrossCoordinatorHealthProtocol",
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    "CrossProcessEventQueue",
    "DaemonInfo",
    "DaemonManager",
    "DaemonManagerConfig",
    "DaemonState",
    "DaemonType",
    "DataPipelineOrchestrator",
    "DistributedLock",
    "DistributedSyncCoordinator",
    "DurationScheduler",
    "DynamicThreshold",
    "EphemeralDataGuard",
    "ErrorRecord",
    "ErrorSeverity",
    "EventBusConfig",
    "EventCoordinatorStats",
    "EventSource",
    "HandlerResilienceConfig",
    "HealthStatus",
    "HealthStats",
    "HeartbeatConfig",
    "HistoryEntry",
    "HostCheckpoint",
    "HostDataState",
    "HostTargets",
    "HostTier",
    "HostType",
    "IterationRecord",
    "JobHealthState",
    "JobPriority",
    "MergeOperation",
    "MergeTransaction",
    "MetricTracker",
    "MetricType",
    "MetricsAccumulator",
    "MetricsAnalysisOrchestrator",
    "MetricsConfig",
    "ModelCacheEntry",
    "ModelLifecycleCoordinator",
    "ModelLifecycleStats",
    "ModelRecord",
    "ModelState",
    "NodeCacheState",
    "NodeConfig",
    "NodeHealthState",
    "NodeResourceState",
    "NodeResources",
    "OptimizationConfig",
    "OptimizationCoordinator",
    "OptimizationResult",
    "OptimizationRun",
    "OptimizationStats",
    "OptimizationType",
    "OrchestratorInfo",
    "OrchestratorLock",
    "OrchestratorRegistry",
    "OrchestratorRole",
    "OrchestratorState",
    "P2PBackend",
    "P2PNodeInfo",
    "PIDController",
    "PipelineConfig",
    "PipelineStage",
    "PipelineStats",
    "PriorityJobScheduler",
    "QuarantineRecord",
    "QueueMonitor",
    "QueueStatus",
    "QueueType",
    "RateLimiter",
    "RecoveryAction",
    "RecoveryAttempt",
    "RecoveryConfig",
    "RecoveryEvent",
    "RecoveryResult",
    "RecoveryStatus",
    "ResourceAlert",
    "ResourceConfig",
    "ResourceMonitor",
    "ResourceMonitoringCoordinator",
    "ResourceOptimizer",
    "ResourceStats",
    "ResourceTargetManager",
    "ResourceType",
    "RouterEvent",
    "SQLitePersistenceMixin",
    "SafeguardConfig",
    "Safeguards",
    "ScaleAction",
    "ScheduledJob",
    "ScheduledTask",
    "SelfplayConfig",
    "SelfplayOrchestrator",
    "SelfplayStats",
    "SelfplayTaskInfo",
    "SelfplayType",
    "SingletonMixin",
    "SnapshotCoordinator",
    "SpawnRateTracker",
    "StageCompletionCallback",
    "StageCompletionResult",
    "StageEvent",
    "StageEventBus",
    "StageTransition",
    "StatePersistable",
    "StatePersistenceMixin",
    "StateSerializer",
    "StateSnapshot",
    "SyncAction",
    "SyncCategory",
    "SyncLockInfo",
    "SyncMutex",
    "SyncPriority",
    "SyncRecommendation",
    "SyncScheduler",
    "SyncStats",
    "TaskContext",
    "TaskCoordinator",
    "TaskDurationRecord",
    "TaskInfo",
    "TaskLifecycleConfig",
    "TaskLifecycleCoordinator",
    "TaskLifecycleStats",
    "TaskLimits",
    "TaskStatus",
    "TaskType",
    "ThresholdManager",
    "ThresholdObservation",
    "TraceCollector",
    "TraceContext",
    "TraceSpan",
    "TrackedTask",
    "TrainingCoordinator",
    "TrainingJob",
    "TrainingProgressEvent",
    "TransactionIsolation",
    "TransactionState",
    "TransferPriority",
    "TransferRecord",
    "TransferVerifier",
    "TransportResult",
    "UnifiedEventCoordinator",
    "UnifiedEventRouter",
    "UnifiedHealthManager",
    "UnifiedRegistry",
    "UtilizationTargets",
    "WriteThrough",
    "ack_event",
    "ack_event_safe",
    "ack_events",
    "acquire_orchestrator_role",
    "acquire_role_safe",
    "acquire_sync_lock",
    "acquire_sync_lock_safe",
    "acquire_training_lock",
    "add_merge_operation",
    "analyze_metrics",
    "async_can_train",
    "async_complete_training",
    "async_get_training_status",
    "async_request_training",
    "async_update_progress",
    "auto_register_known_coordinators",
    "bandwidth_allocation",
    "begin_merge_transaction",
    "bridge_to_cross_process",
    "can_schedule_task",
    # Duration scheduling helpers
    "can_schedule_task_safe",
    "can_spawn",
    "can_spawn_safe",
    "can_train",
    "check_backpressure",
    "check_before_spawn",
    "check_cluster_health",
    "check_host_health",
    "check_resource_thresholds",
    "check_spawn_allowed",
    "checkpoint_games",
    "clear_health_cache",
    "collect_trace",
    "commit_merge_transaction",
    "complete_merge_operation",
    "complete_task_safe",
    "compute_batch_checksum",
    "compute_file_checksum",
    "coordinate_async_task",
    "coordinate_task",
    "create_pipeline_callbacks",
    "discover_p2p_leader_url",
    "emergency_stop_all",
    "emit_backpressure_activated",
    "emit_backpressure_released",
    "emit_cache_invalidated",
    "emit_evaluation_complete",
    "emit_host_offline",
    "emit_host_online",
    "emit_node_recovered",
    # New emitters (December 2025)
    "emit_optimization_triggered",
    "emit_plateau_detected",
    "emit_promotion_complete",
    "emit_quality_updated",
    "emit_regression_detected",
    "emit_selfplay_complete",
    "emit_selfplay_completion",
    "emit_sync_complete",
    "emit_task_complete",
    "emit_training_complete",
    "emit_training_complete_sync",
    # Event Emitters (December 2025)
    "emit_training_started",
    "ephemeral_heartbeat",
    "estimate_duration_safe",
    "estimate_task_duration",
    "execute_priority_sync",
    "extract_trace_from_event",
    "extract_trace_from_headers",
    "fail_task_safe",
    "full_cluster_sync",
    "get_active_task_count",
    "get_all_coordinator_status",
    "get_bandwidth_context",
    "get_bandwidth_manager",
    "get_bandwidth_stats",
    "get_bridge_manager",
    "get_cache_orchestrator",
    "get_cluster_data_status",
    "get_cluster_summary",
    "get_cluster_summary_safe",
    "get_cluster_transport",
    "get_cluster_utilization",
    "get_config",
    "get_config_game_counts",
    "get_coordinator",
    "get_coordinator_safe",
    "get_cpu_rich_hosts",
    "get_cross_coordinator_health",
    "get_current_node_id",
    "get_current_pipeline_stage",
    "get_current_task_context",
    "get_daemon_manager",
    "get_elo_lookup",
    "get_ephemeral_guard",
    "get_evacuation_candidates",
    "get_event_coordinator",
    "get_event_coordinator_stats",
    "get_event_poller_class",
    "get_event_queue",
    "get_event_router",
    "get_gpu_rich_hosts",
    "get_health_manager",
    "get_health_summary",
    "get_healthy_hosts",
    "get_host_bandwidth_status",
    "get_host_targets",
    "get_host_targets_safe",
    "get_job_scheduler",
    "get_metrics_orchestrator",
    "get_model_coordinator",
    "get_next_sync_target",
    "get_optimal_concurrency",
    "get_optimal_transfer_time",
    "get_optimization_coordinator",
    "get_optimization_stats",
    "get_orchestrator_roles",
    "get_p2p_backend",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "get_production_elo",
    "get_production_model_id",
    "get_quality_lookup",
    "get_queue_monitor",
    "get_queue_stats",
    "get_queue_types",
    "get_registered_coordinator",
    "get_registered_coordinators",
    "get_registry",
    "get_registry_safe",
    "get_resource_availability",
    "get_resource_coordinator",
    "get_resource_optimizer",
    "get_resource_targets",
    # Resource targets helpers
    "get_resource_targets_safe",
    "get_role_holder",
    "get_safeguards",
    "get_scheduler",
    "get_selfplay_orchestrator",
    "get_selfplay_stats",
    "get_shared_executor",
    "get_snapshot_coordinator",
    "get_stage_event_bus",
    "get_sync_coordinator",
    # Sync mutex helpers
    "get_sync_lock_context",
    "get_sync_mutex",
    "get_sync_recommendations",
    "get_sync_scheduler",
    "get_sync_stats",
    "get_system_health",
    "get_target_job_count",
    "get_task_lifecycle_coordinator",
    "get_task_resource_type",
    "get_task_stats",
    "get_task_types",
    "get_threshold_manager",
    "get_throttle_factor",
    "get_throttle_factor_safe",
    "get_trace_collector",
    "get_trace_context",
    "get_trace_id",
    "get_training_bridge",
    "get_training_coordinator",
    "get_training_status",
    "get_transaction_isolation",
    "get_transaction_stats",
    "get_transfer_priorities",
    "get_transfer_verifier",
    "get_underserved_configs",
    "get_unified_registry",
    "get_utilization_score",
    "has_bandwidth_manager",
    "has_coordination",
    "has_cross_process_events",
    "has_duration_scheduler",
    "has_resource_targets",
    "has_role",
    "has_sync_lock",
    "initialize_all_coordinators",
    "inject_trace_into_event",
    "inject_trace_into_headers",
    "invalidate_model_caches",
    "is_component_healthy",
    "is_coordinator",
    "is_cpu_task",
    "is_gpu_task",
    "is_heartbeat_running",
    "is_host_ephemeral",
    "is_host_healthy",
    "is_orchestrator_role_available",
    "is_sync_locked",
    "is_unified_loop_running",
    "mark_host_unhealthy",
    "merge_transaction",
    "new_trace",
    "orchestrator_role",
    "poll_events",
    "poll_events_safe",
    "pre_spawn_check",
    "publish_event",
    "publish_event_safe",
    "publish_event_sync",
    "quarantine_file",
    "queue_critical_game",
    "record_games_generated",
    "record_metric",
    "record_sync_complete",
    "record_sync_start",
    "record_task_completion",
    "record_task_completion_safe",
    "record_utilization",
    "register_cache",
    "register_coordinator",
    "register_host",
    "register_running_task",
    "register_running_task_safe",
    "register_standard_callbacks",
    "register_task_safe",
    "release_bandwidth",
    "release_bandwidth_safe",
    "release_orchestrator_role",
    "release_role_safe",
    "release_sync_lock",
    "release_sync_lock_safe",
    "release_training_lock",
    "release_training_slot",
    "report_queue_depth",
    "report_queue_depth_safe",
    "request_bandwidth",
    "request_bandwidth_safe",
    "request_evacuation",
    "request_training_slot",
    "reset_bandwidth_manager",
    "reset_bridge_manager",
    "reset_config",
    "reset_daemon_manager",
    "reset_ephemeral_guard",
    "reset_event_queue",
    "reset_event_router",
    "reset_health_manager",
    "reset_job_scheduler",
    "reset_queue_monitor",
    "reset_resource_targets",
    "reset_scheduler",
    "reset_snapshot_coordinator",
    "reset_stage_event_bus",
    "reset_sync_coordinator",
    "reset_sync_mutex",
    "reset_sync_scheduler",
    "reset_threshold_manager",
    "reset_training_bridge",
    "reset_transaction_isolation",
    "reset_transfer_verifier",
    "reset_unified_registry",
    "rollback_merge_transaction",
    "router_publish_event",
    "run_in_bridge_pool",
    "select_curriculum_config",
    "set_backpressure",
    "set_backpressure_safe",
    "set_config",
    "set_trace_id",
    "setup_signal_handlers",
    "should_scale_down",
    "should_scale_down_safe",
    "should_scale_up",
    "should_scale_up_safe",
    "should_stop_production",
    "should_stop_safe",
    "should_throttle_production",
    "should_throttle_safe",
    "shutdown_all_coordinators",
    "shutdown_registered_coordinators",
    "span",
    "start_coordinator_heartbeats",
    "start_event_coordinator",
    "stop_coordinator_heartbeats",
    "stop_event_coordinator",
    "subscribe_event",
    "subscribe_process",
    "subscribe_process_safe",
    "sync_games",
    "sync_high_quality_games",
    "sync_lock",
    "sync_lock_required",
    "sync_models",
    "sync_training_data",
    "task_context",
    "traced",
    "training_lock",
    "training_slot",
    "trigger_cmaes",
    "trigger_nas",
    "unregister_coordinator",
    "unsubscribe_event",
    "update_config",
    "update_host_state",
    "update_node_resources",
    "update_training_progress",
    "validate_config",
    "verify_batch",
    "verify_transfer",
    "warn_if_orchestrator_running",
    "wire_cache_events",
    "wire_health_events",
    "wire_metrics_events",
    "wire_model_events",
    "wire_optimization_events",
    "wire_pipeline_events",
    "wire_resource_events",
    "wire_selfplay_events",
    "wire_task_events",
    "with_trace",
]
