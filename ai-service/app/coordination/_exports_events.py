"""Event system exports for coordination package.

December 2025: Extracted from __init__.py to improve maintainability.
This module consolidates all event-related imports (router, emitters, cross-process).
"""

# Cross-process event queue exports (via event_router for consolidation)
from app.coordination.event_router import (
    CrossProcessEvent,
    CrossProcessEventPoller,
    CrossProcessEventQueue,
    ack_event,
    ack_events,
    bridge_to_cross_process,
    cp_poll_events as poll_events,
    cp_publish as publish_event,
    get_cross_process_queue as get_event_queue,
    reset_cross_process_queue as reset_event_queue,
    subscribe_process,
)

# Event Emitters (December 2025 - centralized event emission)
from app.coordination.event_emitters import (
    emit_backpressure_activated,
    emit_backpressure_released,
    emit_cache_invalidated,
    emit_evaluation_complete,
    emit_host_offline,
    emit_host_online,
    emit_hyperparameter_updated,
    emit_node_recovered,
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
from app.coordination.event_router import (
    CoordinatorStats as EventCoordinatorStats,
    EventSource,
    RouterEvent,
    UnifiedEventCoordinator,
    UnifiedEventRouter,
    get_coordinator_stats as get_event_coordinator_stats,
    get_event_coordinator,
    get_router as get_event_router,
    publish as router_publish_event,
    publish_sync as publish_event_sync,
    reset_router as reset_event_router,
    start_coordinator as start_event_coordinator,
    stop_coordinator as stop_event_coordinator,
    subscribe as subscribe_event,
    unsubscribe as unsubscribe_event,
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

# Integration Bridge (December 2025 - C2 consolidation)
from app.coordination.integration_bridge import (
    get_wiring_status,
    reset_integration_wiring,
    verify_integration_health,
    verify_integration_health_sync,
    wire_all_integrations,
    wire_all_integrations_sync,
    wire_model_lifecycle_events,
    wire_p2p_integration_events,
    wire_pipeline_feedback_events,
)

__all__ = [
    # Cross-process Events
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    "CrossProcessEventQueue",
    "ack_event",
    "ack_events",
    "bridge_to_cross_process",
    "get_event_queue",
    "poll_events",
    "publish_event",
    "reset_event_queue",
    "subscribe_process",
    # Event Emitters
    "emit_backpressure_activated",
    "emit_backpressure_released",
    "emit_cache_invalidated",
    "emit_evaluation_complete",
    "emit_host_offline",
    "emit_host_online",
    "emit_hyperparameter_updated",
    "emit_node_recovered",
    "emit_optimization_triggered",
    "emit_plateau_detected",
    "emit_promotion_complete",
    "emit_quality_updated",
    "emit_regression_detected",
    "emit_selfplay_complete",
    "emit_sync_complete",
    "emit_task_complete",
    "emit_training_complete",
    "emit_training_complete_sync",
    "emit_training_started",
    # Event Router
    "EventCoordinatorStats",
    "EventSource",
    "RouterEvent",
    "UnifiedEventCoordinator",
    "UnifiedEventRouter",
    "get_event_coordinator",
    "get_event_coordinator_stats",
    "get_event_router",
    "publish_event_sync",
    "reset_event_router",
    "router_publish_event",
    "start_event_coordinator",
    "stop_event_coordinator",
    "subscribe_event",
    "unsubscribe_event",
    # Stage Events
    "StageCompletionCallback",
    "StageCompletionResult",
    "StageEvent",
    "StageEventBus",
    "create_pipeline_callbacks",
    "get_stage_event_bus",
    "register_standard_callbacks",
    "reset_stage_event_bus",
    # Integration Bridge
    "get_wiring_status",
    "reset_integration_wiring",
    "verify_integration_health",
    "verify_integration_health_sync",
    "wire_all_integrations",
    "wire_all_integrations_sync",
    "wire_model_lifecycle_events",
    "wire_p2p_integration_events",
    "wire_pipeline_feedback_events",
]
