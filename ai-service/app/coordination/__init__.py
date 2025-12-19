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

# Sync SCHEDULER exports (unified cluster-wide data sync SCHEDULING)
# Note: This is the SCHEDULING layer - decides WHEN/WHAT to sync.
# For EXECUTION (HOW to sync), use app.distributed.sync_coordinator.SyncCoordinator
from app.coordination.sync_coordinator import (
    # Preferred names (December 2025)
    SyncScheduler,
    get_sync_scheduler,
    reset_sync_scheduler,
    # Backward-compatible names (use SyncScheduler for new code)
    SyncCoordinator as CoordinationSyncCoordinator,  # Alias to avoid collision
    get_sync_coordinator,
    reset_sync_coordinator,
    # Data types
    HostDataState,
    HostType,
    SyncPriority,
    SyncAction,
    SyncRecommendation,
    ClusterDataStatus,
    # Functions
    get_cluster_data_status,
    get_sync_recommendations,
    get_next_sync_target,
    register_host,
    update_host_state,
    record_sync_start,
    record_sync_complete,
    record_games_generated,
    execute_priority_sync,  # Bridge to distributed layer execution
)

# Re-export distributed layer sync execution for convenience
# These provide the actual sync transport (aria2, SSH, P2P)
try:
    from app.distributed.sync_coordinator import (
        SyncCoordinator as DistributedSyncCoordinator,
        SyncStats,
        ClusterSyncStats,
        SyncCategory,
        sync_training_data,
        sync_models,
        sync_games,
        sync_high_quality_games,
        full_cluster_sync,
        get_quality_lookup,
        get_elo_lookup,
    )
except ImportError:
    # Distributed layer may not be available in all environments
    pass

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

# Training Coordination (cluster-wide training management)
from app.coordination.training_coordinator import (
    TrainingCoordinator,
    TrainingJob,
    get_training_coordinator,
    request_training_slot,
    release_training_slot,
    update_training_progress,
    can_train,
    get_training_status,
    training_slot,
)

# Async Training Bridge (async wrapper + event integration)
from app.coordination.async_training_bridge import (
    AsyncTrainingBridge,
    TrainingProgressEvent,
    get_training_bridge,
    reset_training_bridge,
    async_can_train,
    async_request_training,
    async_update_progress,
    async_complete_training,
    async_get_training_status,
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

# Unified Event Router (December 2025 - single API for all event systems)
# Note: EventRouter provides the publishing API, EventCoordinator provides bridging daemon
from app.coordination.event_router import (
    UnifiedEventRouter,
    RouterEvent,
    EventSource,
    get_router as get_event_router,
    reset_router as reset_event_router,
    publish as publish_event,
    publish_sync as publish_event_sync,
    subscribe as subscribe_event,
    unsubscribe as unsubscribe_event,
)

# Distributed Tracing (December 2025)
from app.coordination.tracing import (
    TraceContext,
    TraceSpan,
    TraceCollector,
    get_trace_id,
    set_trace_id,
    get_trace_context,
    new_trace,
    with_trace,
    span,
    traced,
    inject_trace_into_event,
    extract_trace_from_event,
    inject_trace_into_headers,
    extract_trace_from_headers,
    get_trace_collector,
    collect_trace,
)

# Cross-Coordinator Health Protocol (December 2025)
from app.coordination.orchestrator_registry import (
    CoordinatorHealth,
    CrossCoordinatorHealthProtocol,
    get_cross_coordinator_health,
    check_cluster_health,
    # Coordinator Registration (December 2025)
    register_coordinator,
    unregister_coordinator,
    get_coordinator,
    get_registered_coordinators,
    shutdown_all_coordinators,
    auto_register_known_coordinators,
)

# Async Bridge Manager (December 2025 - shared executor pool)
from app.coordination.async_bridge_manager import (
    AsyncBridgeManager,
    get_bridge_manager,
    reset_bridge_manager,
    get_shared_executor,
    run_in_bridge_pool,
)

# Model Lifecycle Coordinator (December 2025)
from app.coordination.model_lifecycle_coordinator import (
    ModelLifecycleCoordinator,
    ModelState,
    ModelRecord,
    CheckpointInfo,
    CacheEntry,
    ModelLifecycleStats,
    get_model_coordinator,
    wire_model_events,
    get_production_model_id,
    get_production_elo,
)

# Task Decorators (December 2025 - lifecycle management)
from app.coordination.task_decorators import (
    TaskContext,
    coordinate_task,
    coordinate_async_task,
    task_context,
    get_current_task_context,
)

# Event Emitters (December 2025 - centralized event emission)
from app.coordination.event_emitters import (
    emit_training_started,
    emit_training_complete,
    emit_training_complete_sync,
    emit_selfplay_complete,
    emit_evaluation_complete,
    emit_promotion_complete,
    emit_sync_complete,
    emit_quality_updated,
    emit_task_complete,
    # New emitters (December 2025)
    emit_optimization_triggered,
    emit_plateau_detected,
    emit_regression_detected,
    emit_backpressure_activated,
    emit_backpressure_released,
    emit_cache_invalidated,
    emit_host_online,
    emit_host_offline,
    emit_node_recovered,
)

# Unified Registry (December 2025 - registry facade)
from app.coordination.unified_registry import (
    UnifiedRegistry,
    ClusterHealth,
    get_unified_registry,
    reset_unified_registry,
)

# =============================================================================
# Orchestrator Imports (December 2025 - event-driven coordination)
# =============================================================================

# SelfplayOrchestrator - unified selfplay event coordination
from app.coordination.selfplay_orchestrator import (
    SelfplayOrchestrator,
    SelfplayType,
    SelfplayTaskInfo,
    SelfplayStats,
    get_selfplay_orchestrator,
    wire_selfplay_events,
    emit_selfplay_completion,
    get_selfplay_stats,
)

# DataPipelineOrchestrator - unified pipeline stage coordination
from app.coordination.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    PipelineStage,
    StageTransition,
    IterationRecord,
    PipelineStats,
    get_pipeline_orchestrator,
    wire_pipeline_events,
    get_pipeline_status,
    get_current_pipeline_stage,
)

# TaskLifecycleCoordinator - unified task event monitoring
from app.coordination.task_lifecycle_coordinator import (
    TaskLifecycleCoordinator,
    TaskStatus,
    TrackedTask,
    TaskLifecycleStats,
    get_task_lifecycle_coordinator,
    wire_task_events,
    get_task_stats,
    get_active_task_count,
)

# OptimizationCoordinator - unified optimization management
from app.coordination.optimization_coordinator import (
    OptimizationCoordinator,
    OptimizationType,
    OptimizationRun,
    OptimizationStats,
    get_optimization_coordinator,
    wire_optimization_events,
    trigger_cmaes,
    trigger_nas,
    get_optimization_stats,
)

# MetricsAnalysisOrchestrator - unified metrics analysis
from app.coordination.metrics_analysis_orchestrator import (
    MetricsAnalysisOrchestrator,
    MetricType,
    MetricTracker,
    AnalysisResult,
    get_metrics_orchestrator,
    wire_metrics_events,
    record_metric,
    analyze_metrics,
)

# ResourceMonitoringCoordinator - unified resource monitoring
from app.coordination.resource_monitoring_coordinator import (
    ResourceMonitoringCoordinator,
    NodeResourceState,
    ResourceAlert,
    ResourceStats,
    get_resource_coordinator,
    wire_resource_events,
    update_node_resources,
    check_resource_thresholds,
)

# CacheCoordinationOrchestrator - unified cache management
from app.coordination.cache_coordination_orchestrator import (
    CacheCoordinationOrchestrator,
    CacheType,
    CacheStatus,
    CacheEntry,
    NodeCacheState,
    CacheStats,
    get_cache_orchestrator,
    wire_cache_events,
    register_cache,
    invalidate_model_caches,
)

# DaemonManager - unified lifecycle management for all background services (December 2025)
from app.coordination.daemon_manager import (
    DaemonManager,
    DaemonType,
    DaemonState,
    DaemonInfo,
    DaemonManagerConfig,
    get_daemon_manager,
    reset_daemon_manager,
    setup_signal_handlers,
)

# Coordinator Persistence Layer (December 2025 - state snapshots and recovery)
from app.coordination.coordinator_persistence import (
    StateSerializer,
    StateSnapshot,
    StatePersistable,
    StatePersistenceMixin,
    SnapshotCoordinator,
    get_snapshot_coordinator,
    reset_snapshot_coordinator,
)

# Coordinator Configuration (December 2025 - centralized config)
from app.coordination.coordinator_config import (
    TaskLifecycleConfig,
    SelfplayConfig,
    PipelineConfig,
    OptimizationConfig,
    MetricsConfig,
    ResourceConfig,
    CacheConfig,
    HandlerResilienceConfig,
    HeartbeatConfig,
    EventBusConfig,
    CoordinatorConfig,
    get_config,
    set_config,
    reset_config,
    update_config,
    validate_config,
)

# Dynamic Threshold Adjustment (December 2025 - adaptive thresholds)
from app.coordination.dynamic_thresholds import (
    DynamicThreshold,
    ThresholdObservation,
    AdjustmentStrategy,
    ThresholdManager,
    get_threshold_manager,
    reset_threshold_manager,
)

# Coordination Utilities (December 2025 - reusable base classes)
from app.coordination.utils import (
    BoundedHistory,
    HistoryEntry,
    MetricsAccumulator,
    CallbackRegistry,
)


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
            from app.distributed.data_events import DataEvent, DataEventType, get_event_bus
            import time as _time

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
    total_count = len([k for k in status.keys() if not k.startswith("_")])

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
        nonlocal issues

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
                try:
                    await emit_coordinator_shutdown(
                        coordinator_name=coord_name,
                        reason="system_shutdown",
                    )
                except Exception:
                    pass
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
    import time as _time

    logger = logging.getLogger(__name__)
    global _heartbeat_running

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

    global _heartbeat_task, _heartbeat_running

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
    global _heartbeat_task
    return _heartbeat_task is not None and not _heartbeat_task.done()


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
    # Sync SCHEDULER (unified data sync SCHEDULING - December 2025)
    # Preferred names (avoids collision with distributed.sync_coordinator.SyncCoordinator)
    "SyncScheduler",
    "get_sync_scheduler",
    "reset_sync_scheduler",
    # Backward-compatible names
    "CoordinationSyncCoordinator",  # Renamed to avoid collision
    "get_sync_coordinator",
    "reset_sync_coordinator",
    # Data types
    "HostDataState",
    "HostType",
    "SyncPriority",
    "SyncAction",
    "SyncRecommendation",
    "ClusterDataStatus",
    # Functions
    "get_cluster_data_status",
    "get_sync_recommendations",
    "get_next_sync_target",
    "register_host",
    "update_host_state",
    "record_sync_start",
    "record_sync_complete",
    "record_games_generated",
    "execute_priority_sync",
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
    # Training Coordination (cluster-wide training management)
    "TrainingCoordinator",
    "TrainingJob",
    "get_training_coordinator",
    "request_training_slot",
    "release_training_slot",
    "update_training_progress",
    "can_train",
    "get_training_status",
    "training_slot",
    # Async Training Bridge (async wrapper + event integration)
    "AsyncTrainingBridge",
    "TrainingProgressEvent",
    "get_training_bridge",
    "reset_training_bridge",
    "async_can_train",
    "async_request_training",
    "async_update_progress",
    "async_complete_training",
    "async_get_training_status",
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
    # Unified Event Router (December 2025)
    "UnifiedEventRouter",
    "RouterEvent",
    "EventSource",
    "get_event_router",
    "reset_event_router",
    "publish_event",
    "publish_event_sync",
    "subscribe_event",
    "unsubscribe_event",
    # Distributed Tracing (December 2025)
    "TraceContext",
    "TraceSpan",
    "TraceCollector",
    "get_trace_id",
    "set_trace_id",
    "get_trace_context",
    "new_trace",
    "with_trace",
    "span",
    "traced",
    "inject_trace_into_event",
    "extract_trace_from_event",
    "inject_trace_into_headers",
    "extract_trace_from_headers",
    "get_trace_collector",
    "collect_trace",
    # Cross-Coordinator Health Protocol (December 2025)
    "CoordinatorHealth",
    "CrossCoordinatorHealthProtocol",
    "get_cross_coordinator_health",
    "check_cluster_health",
    # Coordinator Registration (December 2025)
    "register_coordinator",
    "unregister_coordinator",
    "get_coordinator",
    "get_registered_coordinators",
    "shutdown_all_coordinators",
    "auto_register_known_coordinators",
    # Async Bridge Manager (December 2025)
    "AsyncBridgeManager",
    "get_bridge_manager",
    "reset_bridge_manager",
    "get_shared_executor",
    "run_in_bridge_pool",
    # Task Decorators (December 2025)
    "TaskContext",
    "coordinate_task",
    "coordinate_async_task",
    "task_context",
    "get_current_task_context",
    # Event Emitters (December 2025)
    "emit_training_started",
    "emit_training_complete",
    "emit_training_complete_sync",
    "emit_selfplay_complete",
    "emit_evaluation_complete",
    "emit_promotion_complete",
    "emit_sync_complete",
    "emit_quality_updated",
    "emit_task_complete",
    # New emitters (December 2025)
    "emit_optimization_triggered",
    "emit_plateau_detected",
    "emit_regression_detected",
    "emit_backpressure_activated",
    "emit_backpressure_released",
    "emit_cache_invalidated",
    "emit_host_online",
    "emit_host_offline",
    "emit_node_recovered",
    # Unified Registry (December 2025)
    "UnifiedRegistry",
    "ClusterHealth",
    "get_unified_registry",
    "reset_unified_registry",
    # ==========================================================================
    # Orchestrators (December 2025 - event-driven coordination)
    # ==========================================================================
    # SelfplayOrchestrator
    "SelfplayOrchestrator",
    "SelfplayType",
    "SelfplayTaskInfo",
    "SelfplayStats",
    "get_selfplay_orchestrator",
    "wire_selfplay_events",
    "emit_selfplay_completion",
    "get_selfplay_stats",
    # DataPipelineOrchestrator
    "DataPipelineOrchestrator",
    "PipelineStage",
    "StageTransition",
    "IterationRecord",
    "PipelineStats",
    "get_pipeline_orchestrator",
    "wire_pipeline_events",
    "get_pipeline_status",
    "get_current_pipeline_stage",
    # TaskLifecycleCoordinator
    "TaskLifecycleCoordinator",
    "TaskStatus",
    "TrackedTask",
    "TaskLifecycleStats",
    "get_task_lifecycle_coordinator",
    "wire_task_events",
    "get_task_stats",
    "get_active_task_count",
    # OptimizationCoordinator
    "OptimizationCoordinator",
    "OptimizationType",
    "OptimizationRun",
    "OptimizationStats",
    "get_optimization_coordinator",
    "wire_optimization_events",
    "trigger_cmaes",
    "trigger_nas",
    "get_optimization_stats",
    # MetricsAnalysisOrchestrator
    "MetricsAnalysisOrchestrator",
    "MetricType",
    "MetricTracker",
    "AnalysisResult",
    "get_metrics_orchestrator",
    "wire_metrics_events",
    "record_metric",
    "analyze_metrics",
    # ResourceMonitoringCoordinator
    "ResourceMonitoringCoordinator",
    "NodeResourceState",
    "ResourceAlert",
    "ResourceStats",
    "get_resource_coordinator",
    "wire_resource_events",
    "update_node_resources",
    "check_resource_thresholds",
    # CacheCoordinationOrchestrator
    "CacheCoordinationOrchestrator",
    "CacheType",
    "CacheStatus",
    "CacheEntry",
    "NodeCacheState",
    "CacheStats",
    "get_cache_orchestrator",
    "wire_cache_events",
    "register_cache",
    "invalidate_model_caches",
    # DaemonManager (December 2025)
    "DaemonManager",
    "DaemonType",
    "DaemonState",
    "DaemonInfo",
    "DaemonManagerConfig",
    "get_daemon_manager",
    "reset_daemon_manager",
    "setup_signal_handlers",
    # Unified Initialization (December 2025)
    "initialize_all_coordinators",
    "get_all_coordinator_status",
    "get_system_health",
    "shutdown_all_coordinators",
    # Coordinator Heartbeats (December 2025)
    "start_coordinator_heartbeats",
    "stop_coordinator_heartbeats",
    "is_heartbeat_running",
    # Coordinator Persistence Layer (December 2025)
    "StateSerializer",
    "StateSnapshot",
    "StatePersistable",
    "StatePersistenceMixin",
    "SnapshotCoordinator",
    "get_snapshot_coordinator",
    "reset_snapshot_coordinator",
    # Coordinator Configuration (December 2025)
    "TaskLifecycleConfig",
    "SelfplayConfig",
    "PipelineConfig",
    "OptimizationConfig",
    "MetricsConfig",
    "ResourceConfig",
    "CacheConfig",
    "HandlerResilienceConfig",
    "HeartbeatConfig",
    "EventBusConfig",
    "CoordinatorConfig",
    "get_config",
    "set_config",
    "reset_config",
    "update_config",
    "validate_config",
    # Dynamic Threshold Adjustment (December 2025)
    "DynamicThreshold",
    "ThresholdObservation",
    "AdjustmentStrategy",
    "ThresholdManager",
    "get_threshold_manager",
    "reset_threshold_manager",
    # Coordination Utilities (December 2025)
    "BoundedHistory",
    "HistoryEntry",
    "MetricsAccumulator",
    "CallbackRegistry",
]
