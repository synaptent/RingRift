"""Consolidated Event System (December 2025).

This module consolidates the event routing infrastructure:
- Unified Event Router (3-tier event bus unification)
- Event type mappings (Stage <-> Data <-> CrossProcess)
- Typed emit functions (70+ convenience emitters)
- Event name normalization

This is part of the 157→15 module consolidation (Phase 5).

Migration Guide:
    # Old imports (DEPRECATED - do not use event_emitters directly):
    #   from app.coordination.event_emitters import (
    #       emit_training_complete, emit_selfplay_complete, ...
    #   )
    # Use event_router for generic emit:
    #   from app.coordination.event_router import safe_emit_event, emit_event
    # Or use this module (core_events) for typed emitters:

    # Preferred imports:
    from app.coordination.core_events import (
        # Router core
        UnifiedEventRouter, get_router, publish, publish_sync,
        subscribe, unsubscribe, reset_router,

        # Event types
        DataEventType, DataEvent, EventBus,
        StageEvent, StageCompletionResult,
        CrossProcessEvent, EventSource, RouterEvent,

        # Event bus access
        get_event_bus, get_stage_event_bus, get_cross_process_queue,

        # Mappings
        STAGE_TO_DATA_EVENT_MAP, DATA_TO_STAGE_EVENT_MAP,
        DATA_TO_CROSS_PROCESS_MAP, CROSS_PROCESS_TO_DATA_MAP,
        STAGE_TO_CROSS_PROCESS_MAP,
        get_data_event_type, get_stage_event_type, get_cross_process_event_type,

        # Typed emitters
        emit_training_complete, emit_training_started,
        emit_selfplay_complete, emit_sync_complete,
        emit_evaluation_complete, emit_promotion_complete,
        emit_regression_detected, emit_quality_updated,
        # ... see __all__ for complete list

        # Normalization
        normalize_event_type, CANONICAL_EVENT_NAMES,

        # Utilities
        validate_event_flow, get_orphaned_events, get_event_stats,
    )
"""

from __future__ import annotations

import importlib
import warnings

# =============================================================================
# Re-exports from event_router.py (main routing infrastructure)
# =============================================================================

from app.coordination.event_router import (
    # Backward compatibility for unified_event_coordinator
    CoordinatorStats,
    # Cross-process events (re-exported)
    CrossProcessEvent,
    CrossProcessEventPoller,
    CrossProcessEventQueue,
    # Mapping re-exports
    DATA_TO_STAGE_EVENT_MAP,
    # Data events (re-exported)
    DataEvent,
    DataEventType,
    EventBus,
    # Source enum
    EventSource,
    RouterEvent,
    STAGE_TO_DATA_EVENT_MAP,
    # Stage events (re-exported)
    StageCompletionResult,
    StageEvent,
    UnifiedEventCoordinator,
    # Core router class
    UnifiedEventRouter,
    ack_event,
    ack_events,
    bridge_to_cross_process,
    cp_poll_events,
    cp_publish,
    # Data event emit functions (re-exported from data_events)
    emit_cluster_capacity_changed,
    emit_curriculum_advanced,
    emit_daemon_status_changed,
    emit_data_event,
    emit_data_sync_failed,
    emit_elo_velocity_changed,
    # Backward compatibility emit functions
    emit_evaluation_completed,
    emit_exploration_boost,
    emit_host_offline,
    emit_host_online,
    emit_idle_resource_detected,
    emit_leader_elected,
    emit_leader_lost,
    emit_model_promoted,
    emit_node_overloaded,
    emit_promotion_candidate,
    emit_quality_check_requested,
    emit_quality_degraded,
    emit_quality_score_updated,
    emit_selfplay_batch_completed,
    emit_selfplay_target_updated,
    emit_sync_completed,
    emit_training_completed,
    emit_training_completed_sync,
    emit_training_early_stopped,
    emit_training_failed,
    emit_training_loss_anomaly,
    emit_training_loss_trend,
    emit_training_started,
    emit_training_started_sync,
    get_coordinator_stats,
    get_cross_process_queue,
    # Bus access functions
    get_event_bus,
    get_event_coordinator,
    get_event_payload,
    get_event_stats,
    get_orphaned_events,
    # Global access
    get_router,
    get_stage_event_bus,
    # Convenience functions
    publish,
    publish_sync,
    reset_cross_process_queue,
    reset_router,
    start_coordinator,
    stop_coordinator,
    subscribe,
    subscribe_process,
    unsubscribe,
    validate_event_flow,
)

# =============================================================================
# Re-exports from event_mappings.py (event type mappings)
# =============================================================================

from app.coordination.event_mappings import (
    CROSS_PROCESS_TO_DATA_MAP,
    DATA_TO_CROSS_PROCESS_MAP,
    DATA_TO_STAGE_EVENT_MAP as _DATA_TO_STAGE_MAP_MAPPINGS,
    STAGE_TO_CROSS_PROCESS_MAP,
    STAGE_TO_DATA_EVENT_MAP as _STAGE_TO_DATA_MAP_MAPPINGS,
    get_all_event_types,
    get_cross_process_event_type,
    get_data_event_type,
    get_stage_event_type,
    is_mapped_event,
    validate_mappings,
)

# =============================================================================
# Re-exports from event_router.py (typed emit functions)
# April 2026: Migrated from deprecated event_emitters to event_router.
# The event_router module provides safe_emit_event() and emit_event() as the
# primary API. For backward compatibility, legacy emit_* names are resolved
# via __getattr__ from event_router (which re-exports from data_events).
# =============================================================================

_EVENT_EMITTER_EXPORTS = frozenset(
    {
        "emit_backpressure_activated",
        "emit_backpressure_released",
        "emit_cache_invalidated",
        "emit_coordinator_health_degraded",
        "emit_coordinator_healthy",
        "emit_coordinator_heartbeat",
        "emit_coordinator_shutdown",
        "emit_coordinator_unhealthy",
        "emit_curriculum_rebalanced",
        "emit_curriculum_updated",
        "emit_evaluation_complete",
        "emit_game_quality_score",
        "emit_handler_failed",
        "emit_handler_timeout",
        "emit_health_check_failed",
        "emit_health_check_passed",
        "emit_hyperparameter_updated",
        "emit_model_corrupted",
        "emit_new_games",
        "emit_node_recovered",
        "emit_node_unhealthy",
        "emit_optimization_triggered",
        "emit_p2p_cluster_healthy",
        "emit_p2p_cluster_unhealthy",
        "emit_p2p_node_dead",
        "emit_plateau_detected",
        "emit_promotion_complete",
        "emit_promotion_complete_sync",
        "emit_quality_updated",
        "emit_regression_detected",
        "emit_repair_completed",
        "emit_repair_failed",
        "emit_selfplay_complete",
        "emit_sync_complete",
        "emit_task_abandoned",
        "emit_task_complete",
        "emit_task_orphaned",
        "emit_training_complete",
        "emit_training_complete_sync",
        "emit_training_rollback_completed",
        "emit_training_rollback_needed",
        "emit_training_triggered",
    }
)


def _get_event_router_module():
    """Resolve the event_router module for dynamic emit function lookup."""
    return importlib.import_module("app.coordination.event_router")

# =============================================================================
# Re-exports from event_normalization.py
# =============================================================================

from app.coordination.event_normalization import (
    CANONICAL_EVENT_NAMES,
    EVENT_NAMING_GUIDELINES,
    audit_event_usage,
    get_variants,
    is_canonical,
    normalize_event_type,
    validate_event_names,
)

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # ==== Core Router ====
    "UnifiedEventRouter",
    "RouterEvent",
    "EventSource",
    "get_router",
    "publish",
    "publish_sync",
    "subscribe",
    "unsubscribe",
    "reset_router",
    # ==== Event Types ====
    "DataEventType",
    "DataEvent",
    "EventBus",
    "StageEvent",
    "StageCompletionResult",
    "CrossProcessEvent",
    "CrossProcessEventPoller",
    "CrossProcessEventQueue",
    # ==== Bus Access ====
    "get_event_bus",
    "get_stage_event_bus",
    "get_cross_process_queue",
    "reset_cross_process_queue",
    # ==== Cross-Process Functions ====
    "ack_event",
    "ack_events",
    "bridge_to_cross_process",
    "cp_poll_events",
    "cp_publish",
    "subscribe_process",
    # ==== Event Mappings ====
    "STAGE_TO_DATA_EVENT_MAP",
    "DATA_TO_STAGE_EVENT_MAP",
    "DATA_TO_CROSS_PROCESS_MAP",
    "CROSS_PROCESS_TO_DATA_MAP",
    "STAGE_TO_CROSS_PROCESS_MAP",
    "get_data_event_type",
    "get_stage_event_type",
    "get_cross_process_event_type",
    "is_mapped_event",
    "get_all_event_types",
    "validate_mappings",
    # ==== Event Normalization ====
    "CANONICAL_EVENT_NAMES",
    "EVENT_NAMING_GUIDELINES",
    "normalize_event_type",
    "is_canonical",
    "get_variants",
    "audit_event_usage",
    "validate_event_names",
    # ==== Validation & Stats ====
    "validate_event_flow",
    "get_orphaned_events",
    "get_event_stats",
    "get_event_payload",
    # ==== Typed Emit Functions (Event Router) ====
    "emit_data_event",
    "emit_cluster_capacity_changed",
    "emit_curriculum_advanced",
    "emit_daemon_status_changed",
    "emit_data_sync_failed",
    "emit_elo_velocity_changed",
    "emit_exploration_boost",
    "emit_host_offline",
    "emit_host_online",
    "emit_idle_resource_detected",
    "emit_leader_elected",
    "emit_leader_lost",
    "emit_node_overloaded",
    "emit_promotion_candidate",
    "emit_quality_check_requested",
    "emit_quality_degraded",
    "emit_quality_score_updated",
    "emit_selfplay_target_updated",
    "emit_training_early_stopped",
    "emit_training_loss_anomaly",
    "emit_training_loss_trend",
    # ==== Typed Emit Functions (Event Emitters) ====
    "emit_backpressure_activated",
    "emit_backpressure_released",
    "emit_cache_invalidated",
    "emit_coordinator_healthy",
    "emit_coordinator_unhealthy",
    "emit_coordinator_heartbeat",
    "emit_coordinator_shutdown",
    "emit_coordinator_health_degraded",
    "emit_curriculum_updated",
    "emit_curriculum_rebalanced",
    "emit_evaluation_complete",
    "emit_game_quality_score",
    "emit_handler_failed",
    "emit_handler_timeout",
    "emit_health_check_passed",
    "emit_health_check_failed",
    "emit_hyperparameter_updated",
    "emit_model_corrupted",
    "emit_new_games",
    "emit_node_recovered",
    "emit_node_unhealthy",
    "emit_optimization_triggered",
    "emit_p2p_cluster_healthy",
    "emit_p2p_cluster_unhealthy",
    "emit_p2p_node_dead",
    "emit_plateau_detected",
    "emit_promotion_complete",
    "emit_promotion_complete_sync",
    "emit_quality_updated",
    "emit_regression_detected",
    "emit_repair_completed",
    "emit_repair_failed",
    "emit_selfplay_complete",
    "emit_sync_complete",
    "emit_task_abandoned",
    "emit_task_complete",
    "emit_task_orphaned",
    "emit_training_complete",
    "emit_training_complete_sync",
    "emit_training_rollback_completed",
    "emit_training_rollback_needed",
    "emit_training_started",
    "emit_training_started_sync",
    "emit_training_triggered",
    # ==== Backward Compatibility (unified_event_coordinator) ====
    "UnifiedEventCoordinator",
    "CoordinatorStats",
    "get_event_coordinator",
    "get_coordinator_stats",
    "start_coordinator",
    "stop_coordinator",
    "emit_evaluation_completed",
    "emit_model_promoted",
    "emit_selfplay_batch_completed",
    "emit_sync_completed",
    "emit_training_completed",
    "emit_training_completed_sync",
    "emit_training_failed",
]


def __getattr__(name: str):
    """Forward legacy emitter exports via event_router.

    April 2026: Migrated from event_emitters to event_router. Tries event_router
    first (which re-exports many emit functions from data_events), then falls back
    to event_emitters compatibility shim for typed stage-event emitters that haven't
    been ported to event_router yet.
    """
    if name in _EVENT_EMITTER_EXPORTS:
        # Try event_router first (preferred source)
        router_module = _get_event_router_module()
        if hasattr(router_module, name):
            return getattr(router_module, name)
        # Fall back to event_emitters shim for stage-event typed emitters
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _emitters = importlib.import_module("app.coordination.event_emitters")
        return getattr(_emitters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
