"""High-level orchestrator exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates all orchestrator and coordinator imports.
"""

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

# MetricsAnalysisOrchestrator - unified metrics analysis
from app.coordination.metrics_analysis_orchestrator import (
    AnalysisResult,
    MetricTracker,
    MetricType,
    MetricsAnalysisOrchestrator,
    analyze_metrics,
    get_metrics_orchestrator,
    record_metric,
    wire_metrics_events,
)

# Model Lifecycle Coordinator
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

__all__ = [
    # Cache Orchestrator
    "CacheCoordinationOrchestrator",
    "CacheEntry",
    "CacheStats",
    "CacheStatus",
    "CacheType",
    "NodeCacheState",
    "get_cache_orchestrator",
    "invalidate_model_caches",
    "register_cache",
    "wire_cache_events",
    # Pipeline Orchestrator
    "DataPipelineOrchestrator",
    "IterationRecord",
    "PipelineStage",
    "PipelineStats",
    "StageTransition",
    "get_current_pipeline_stage",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "wire_pipeline_events",
    # Metrics Orchestrator
    "AnalysisResult",
    "MetricTracker",
    "MetricType",
    "MetricsAnalysisOrchestrator",
    "analyze_metrics",
    "get_metrics_orchestrator",
    "record_metric",
    "wire_metrics_events",
    # Model Lifecycle
    "CheckpointInfo",
    "ModelCacheEntry",
    "ModelLifecycleCoordinator",
    "ModelLifecycleStats",
    "ModelRecord",
    "ModelState",
    "get_model_coordinator",
    "get_production_elo",
    "get_production_model_id",
    "wire_model_events",
    # Optimization
    "OptimizationCoordinator",
    "OptimizationRun",
    "OptimizationStats",
    "OptimizationType",
    "get_optimization_coordinator",
    "get_optimization_stats",
    "trigger_cmaes",
    "trigger_nas",
    "wire_optimization_events",
    # Resource Monitoring
    "NodeResourceState",
    "ResourceAlert",
    "ResourceMonitoringCoordinator",
    "ResourceStats",
    "check_resource_thresholds",
    "get_resource_coordinator",
    "update_node_resources",
    "wire_resource_events",
    # Selfplay
    "SelfplayOrchestrator",
    "SelfplayStats",
    "SelfplayTaskInfo",
    "SelfplayType",
    "emit_selfplay_completion",
    "get_selfplay_orchestrator",
    "get_selfplay_stats",
    "wire_selfplay_events",
    # Task Lifecycle
    "TaskLifecycleCoordinator",
    "TaskLifecycleStats",
    "TaskStatus",
    "TrackedTask",
    "get_active_task_count",
    "get_task_lifecycle_coordinator",
    "get_task_stats",
    "wire_task_events",
    # Training
    "AsyncTrainingBridge",
    "TrainingCoordinator",
    "TrainingJob",
    "TrainingProgressEvent",
    "async_can_train",
    "async_complete_training",
    "async_get_training_status",
    "async_request_training",
    "async_update_progress",
    "can_train",
    "get_training_bridge",
    "get_training_coordinator",
    "get_training_status",
    "release_training_slot",
    "request_training_slot",
    "reset_training_bridge",
    "training_slot",
    "update_training_progress",
]
