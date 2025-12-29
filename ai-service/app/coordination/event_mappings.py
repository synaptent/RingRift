"""Centralized event type mappings for the RingRift event system.

.. deprecated:: December 2025
    This module has been consolidated into ``app.coordination.core_events``.
    Import from core_events for new code. This module remains for backward
    compatibility and will be removed in Q2 2026.

    Migration:
        # Old import (deprecated)
        from app.coordination.event_mappings import STAGE_TO_DATA_EVENT_MAP

        # New import (preferred)
        from app.coordination.core_events import STAGE_TO_DATA_EVENT_MAP

This module consolidates all event type mappings between the three event buses:
1. DataEventBus (data_events.py) - In-memory async event bus
2. StageEventBus (stage_events.py) - Pipeline stage completion events
3. CrossProcessEventQueue (cross_process_events.py) - SQLite-backed IPC

The mappings enable automatic event translation as events flow between buses.

Usage:
    from app.coordination.core_events import (
        STAGE_TO_DATA_EVENT_MAP,
        DATA_TO_CROSS_PROCESS_MAP,
        get_cross_process_event_type,
    )

Design Goals:
    1. Single source of truth for all event mappings
    2. Automatic reverse mapping generation
    3. Validation helpers for unmapped event types
    4. Documentation of event relationships

Created: December 2025
Purpose: DRY consolidation of event mappings (Phase 13+ consolidation)
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# Stage Event → Data Event Mappings
# =============================================================================

# Map StageEvent values to DataEventType values
# Used by UnifiedEventRouter to forward stage events to the data bus
# P0.2 Dec 2025: Updated to emit selfplay_complete directly instead of just new_games
STAGE_TO_DATA_EVENT_MAP: Final[dict[str, str]] = {
    # Selfplay events (P0.2: now emits selfplay_complete for feedback loop)
    "selfplay_complete": "selfplay_complete",
    "canonical_selfplay_complete": "selfplay_complete",
    "gpu_selfplay_complete": "selfplay_complete",
    # Sync events
    "sync_complete": "sync_completed",
    "cluster_sync_complete": "sync_completed",
    "model_sync_complete": "p2p_model_synced",
    # Training events
    "training_complete": "training_completed",
    "training_started": "training_started",
    "training_failed": "training_failed",
    # Evaluation events
    "evaluation_complete": "evaluation_completed",
    "shadow_tournament_complete": "evaluation_completed",
    "elo_calibration_complete": "elo_updated",
    # Promotion events
    "promotion_complete": "model_promoted",
    "tier_gating_complete": "model_promoted",
    # Optimization events
    "cmaes_complete": "cmaes_completed",
    "pbt_complete": "pbt_generation_complete",
    "nas_complete": "nas_completed",
}

# Reverse mapping: DataEventType → StageEvent
# Note: This is a many-to-one mapping reversed, so we pick representative stage event
DATA_TO_STAGE_EVENT_MAP: Final[dict[str, str]] = {
    "selfplay_complete": "selfplay_complete",  # P0.2 Dec 2025
    "new_games": "selfplay_complete",
    "sync_completed": "sync_complete",
    "p2p_model_synced": "model_sync_complete",
    "training_completed": "training_complete",
    "training_started": "training_started",
    "training_failed": "training_failed",
    "evaluation_completed": "evaluation_complete",
    "elo_updated": "elo_calibration_complete",
    "model_promoted": "promotion_complete",
    "cmaes_completed": "cmaes_complete",
    "pbt_generation_complete": "pbt_complete",
    "nas_completed": "nas_complete",
}

# =============================================================================
# Data Event → Cross-Process Event Mappings
# =============================================================================

# Map DataEventType values to CrossProcess event type strings
# CrossProcess uses UPPERCASE_SNAKE_CASE convention
DATA_TO_CROSS_PROCESS_MAP: Final[dict[str, str]] = {
    # Training events
    "training_started": "TRAINING_STARTED",
    "training_completed": "TRAINING_COMPLETED",
    "training_failed": "TRAINING_FAILED",
    "training_threshold": "TRAINING_THRESHOLD_REACHED",
    "training_progress": "TRAINING_PROGRESS",
    # Evaluation events
    "evaluation_started": "EVALUATION_STARTED",
    "evaluation_completed": "EVALUATION_COMPLETED",
    "evaluation_failed": "EVALUATION_FAILED",
    "evaluation_progress": "EVALUATION_PROGRESS",
    "elo_updated": "ELO_UPDATED",
    # Promotion events
    "model_promoted": "MODEL_PROMOTED",
    "promotion_failed": "PROMOTION_FAILED",
    "promotion_candidate": "PROMOTION_CANDIDATE",
    "promotion_started": "PROMOTION_STARTED",
    "promotion_rejected": "PROMOTION_REJECTED",
    # Data events
    "new_games": "NEW_GAMES_AVAILABLE",
    "sync_completed": "DATA_SYNC_COMPLETED",
    "sync_started": "DATA_SYNC_STARTED",
    "sync_failed": "DATA_SYNC_FAILED",
    "sync_stalled": "SYNC_STALLED",
    "sync_triggered": "SYNC_TRIGGERED",
    "data_stale": "DATA_STALE",
    "data_fresh": "DATA_FRESH",
    "game_synced": "GAME_SYNCED",
    "p2p_model_synced": "P2P_MODEL_SYNCED",
    # Quality events
    "quality_score_updated": "QUALITY_SCORE_UPDATED",
    "quality_distribution_changed": "QUALITY_DISTRIBUTION_CHANGED",
    "high_quality_data_available": "HIGH_QUALITY_DATA_AVAILABLE",
    "low_quality_data_warning": "LOW_QUALITY_DATA_WARNING",
    "quality_degraded": "QUALITY_DEGRADED",
    "quality_check_requested": "QUALITY_CHECK_REQUESTED",
    "quality_check_failed": "QUALITY_CHECK_FAILED",
    "data_quality_alert": "DATA_QUALITY_ALERT",
    # Regression events
    "regression_detected": "REGRESSION_DETECTED",
    "regression_minor": "REGRESSION_MINOR",
    "regression_moderate": "REGRESSION_MODERATE",
    "regression_severe": "REGRESSION_SEVERE",
    "regression_critical": "REGRESSION_CRITICAL",
    "regression_cleared": "REGRESSION_CLEARED",
    # Optimization events
    "cmaes_completed": "CMAES_COMPLETED",
    "cmaes_triggered": "CMAES_TRIGGERED",
    "pbt_generation_complete": "PBT_COMPLETE",
    "nas_completed": "NAS_COMPLETED",
    "nas_triggered": "NAS_TRIGGERED",
    "plateau_detected": "PLATEAU_DETECTED",
    "hyperparameter_updated": "HYPERPARAMETER_UPDATED",
    # Curriculum events
    "curriculum_updated": "CURRICULUM_UPDATED",
    "curriculum_rebalanced": "CURRICULUM_REBALANCED",
    "curriculum_advanced": "CURRICULUM_ADVANCED",
    "tier_promotion": "TIER_PROMOTION",
    "elo_significant_change": "ELO_SIGNIFICANT_CHANGE",
    "elo_velocity_changed": "ELO_VELOCITY_CHANGED",
    # Selfplay feedback events
    "selfplay_complete": "SELFPLAY_BATCH_COMPLETE",
    "selfplay_target_updated": "SELFPLAY_TARGET_UPDATED",
    "selfplay_rate_changed": "SELFPLAY_RATE_CHANGED",
    # Selfplay orchestrator events (Dec 2025 - feedback loop integration)
    "request_selfplay_queued": "REQUEST_SELFPLAY_QUEUED",
    "selfplay_budget_adjusted": "SELFPLAY_BUDGET_ADJUSTED",
    "curriculum_allocation_changed": "CURRICULUM_ALLOCATION_CHANGED",
    # Coordinator/daemon health
    "coordinator_healthy": "COORDINATOR_HEALTHY",
    "coordinator_unhealthy": "COORDINATOR_UNHEALTHY",
    "coordinator_heartbeat": "COORDINATOR_HEARTBEAT",
    "coordinator_shutdown": "COORDINATOR_SHUTDOWN",
    "daemon_started": "DAEMON_STARTED",
    "daemon_stopped": "DAEMON_STOPPED",
    "daemon_status_changed": "DAEMON_STATUS_CHANGED",
    # Host/cluster events
    "host_online": "HOST_ONLINE",
    "host_offline": "HOST_OFFLINE",
    "cluster_status_changed": "CLUSTER_STATUS_CHANGED",
    "node_unhealthy": "NODE_UNHEALTHY",
    "node_recovered": "NODE_RECOVERED",
    "node_suspect": "NODE_SUSPECT",  # Dec 2025: Node entering grace period
    "node_retired": "NODE_RETIRED",  # Dec 2025: Node explicitly retired
    "node_activated": "NODE_ACTIVATED",
    "cluster_capacity_changed": "CLUSTER_CAPACITY_CHANGED",
    "sync_capacity_refreshed": "SYNC_CAPACITY_REFRESHED",
    # Availability/provisioning events (December 28, 2025)
    "capacity_low": "CAPACITY_LOW",
    "capacity_restored": "CAPACITY_RESTORED",
    "node_provisioned": "NODE_PROVISIONED",
    "node_provision_failed": "NODE_PROVISION_FAILED",
    "budget_exceeded": "BUDGET_EXCEEDED",
    "budget_alert": "BUDGET_ALERT",
    # P2P cluster events
    "p2p_cluster_healthy": "P2P_CLUSTER_HEALTHY",
    "p2p_cluster_unhealthy": "P2P_CLUSTER_UNHEALTHY",
    "p2p_nodes_dead": "P2P_NODES_DEAD",
    "p2p_node_dead": "P2P_NODE_DEAD",  # Dec 2025: Single node dead (vs p2p_nodes_dead for batch)
    "leader_elected": "LEADER_ELECTED",
    "leader_lost": "LEADER_LOST",
    "leader_heartbeat_missing": "LEADER_HEARTBEAT_MISSING",  # P0 Dec 2025: Monitoring event
    # Work queue events
    "work_queued": "WORK_QUEUED",
    "work_claimed": "WORK_CLAIMED",
    "work_started": "WORK_STARTED",
    "work_completed": "WORK_COMPLETED",
    "work_failed": "WORK_FAILED",
    "work_retry": "WORK_RETRY",
    "work_timeout": "WORK_TIMEOUT",
    "work_cancelled": "WORK_CANCELLED",
    "job_preempted": "JOB_PREEMPTED",
    # Task lifecycle events
    "task_spawned": "TASK_SPAWNED",
    "task_heartbeat": "TASK_HEARTBEAT",
    "task_completed": "TASK_COMPLETED",
    "task_failed": "TASK_FAILED",
    "task_orphaned": "TASK_ORPHANED",
    "task_cancelled": "TASK_CANCELLED",
    "task_abandoned": "TASK_ABANDONED",  # Dec 2025: Intentionally cancelled tasks
    # Handler lifecycle events
    "handler_failed": "HANDLER_FAILED",  # Dec 2025: Event handler threw exception
    # Resource/capacity events
    "idle_resource_detected": "IDLE_RESOURCE_DETECTED",
    "backpressure_activated": "BACKPRESSURE_ACTIVATED",
    "backpressure_released": "BACKPRESSURE_RELEASED",
    "resource_constraint": "RESOURCE_CONSTRAINT",
    # Cache/registry events
    "cache_invalidated": "CACHE_INVALIDATED",
    "registry_updated": "REGISTRY_UPDATED",
    "metrics_updated": "METRICS_UPDATED",
    # Model management events
    "model_updated": "MODEL_UPDATED",
    "model_sync_requested": "MODEL_SYNC_REQUESTED",
    "model_distribution_complete": "MODEL_DISTRIBUTION_COMPLETE",
    # Database/orphan detection events
    "database_created": "DATABASE_CREATED",
    "orphan_games_detected": "ORPHAN_GAMES_DETECTED",
    "orphan_games_registered": "ORPHAN_GAMES_REGISTERED",
    # Parity validation events
    "parity_validation_started": "PARITY_VALIDATION_STARTED",
    "parity_validation_completed": "PARITY_VALIDATION_COMPLETED",
    "parity_failure_rate_changed": "PARITY_FAILURE_RATE_CHANGED",
    # December 2025: Additional cross-process events for cluster coordination
    # Model distribution events
    "model_distribution_started": "MODEL_DISTRIBUTION_STARTED",
    "model_distribution_failed": "MODEL_DISTRIBUTION_FAILED",
    # Data consolidation events
    "consolidation_started": "CONSOLIDATION_STARTED",
    "consolidation_complete": "CONSOLIDATION_COMPLETE",
    # NPZ export and combination events (December 2025 - training data pipeline)
    "npz_export_started": "NPZ_EXPORT_STARTED",
    "npz_export_complete": "NPZ_EXPORT_COMPLETE",
    "npz_combination_started": "NPZ_COMBINATION_STARTED",
    "npz_combination_complete": "NPZ_COMBINATION_COMPLETE",
    "npz_combination_failed": "NPZ_COMBINATION_FAILED",
    # Replication/repair events
    "repair_completed": "REPAIR_COMPLETED",
    "repair_failed": "REPAIR_FAILED",
    "replication_alert": "REPLICATION_ALERT",
    # Split-brain detection (P2P cluster)
    "split_brain_detected": "SPLIT_BRAIN_DETECTED",
    "split_brain_resolved": "SPLIT_BRAIN_RESOLVED",
    # Sync integrity events
    "sync_request": "SYNC_REQUEST",
    "sync_checksum_failed": "SYNC_CHECKSUM_FAILED",
    "sync_node_unreachable": "SYNC_NODE_UNREACHABLE",
    # Recovery events
    "recovery_initiated": "RECOVERY_INITIATED",
    "recovery_completed": "RECOVERY_COMPLETED",
    "recovery_failed": "RECOVERY_FAILED",
    # Node lifecycle events
    "node_terminated": "NODE_TERMINATED",
    "cluster_stall_detected": "CLUSTER_STALL_DETECTED",
    # December 28, 2025: High-impact events from coordinator audit
    # Daemon health events
    "daemon_permanently_failed": "DAEMON_PERMANENTLY_FAILED",
    "coordinator_health_degraded": "COORDINATOR_HEALTH_DEGRADED",
    "coordinator_init_failed": "COORDINATOR_INIT_FAILED",
    # Resource and disk events
    "disk_space_low": "DISK_SPACE_LOW",
    "node_overloaded": "NODE_OVERLOADED",
    "deadlock_detected": "DEADLOCK_DETECTED",
    # Model and data integrity events
    "model_corrupted": "MODEL_CORRUPTED",
    "promotion_rolled_back": "PROMOTION_ROLLED_BACK",
    # Training health events
    "training_blocked_by_quality": "TRAINING_BLOCKED_BY_QUALITY",
    "training_loss_anomaly": "TRAINING_LOSS_ANOMALY",
    "training_rollback_needed": "TRAINING_ROLLBACK_NEEDED",
    "training_rollback_completed": "TRAINING_ROLLBACK_COMPLETED",
    "training_early_stopped": "TRAINING_EARLY_STOPPED",
    # Health monitoring events
    "health_alert": "HEALTH_ALERT",
    "health_check_failed": "HEALTH_CHECK_FAILED",
    "health_check_passed": "HEALTH_CHECK_PASSED",
    # Handler events
    "handler_timeout": "HANDLER_TIMEOUT",
}

# Reverse mapping: CrossProcess → DataEventType
CROSS_PROCESS_TO_DATA_MAP: Final[dict[str, str]] = {
    v: k for k, v in DATA_TO_CROSS_PROCESS_MAP.items()
}

# =============================================================================
# Stage Event → Cross-Process Event Mappings
# =============================================================================

# Direct mapping from StageEvent to CrossProcess for efficiency
# Bypasses intermediate Data event conversion
STAGE_TO_CROSS_PROCESS_MAP: Final[dict[str, str]] = {
    # Training events
    "training_complete": "TRAINING_COMPLETED",
    "training_started": "TRAINING_STARTED",
    "training_failed": "TRAINING_FAILED",
    # Evaluation events
    "evaluation_complete": "EVALUATION_COMPLETED",
    "shadow_tournament_complete": "SHADOW_TOURNAMENT_COMPLETE",
    "elo_calibration_complete": "ELO_CALIBRATION_COMPLETE",
    # Promotion events
    "promotion_complete": "MODEL_PROMOTED",
    "tier_gating_complete": "TIER_GATING_COMPLETE",
    # Selfplay events
    "selfplay_complete": "SELFPLAY_BATCH_COMPLETE",
    "canonical_selfplay_complete": "CANONICAL_SELFPLAY_COMPLETE",
    "gpu_selfplay_complete": "GPU_SELFPLAY_COMPLETE",
    # Data/sync events
    "sync_complete": "DATA_SYNC_COMPLETED",
    "parity_validation_complete": "PARITY_VALIDATION_COMPLETE",
    "npz_export_started": "NPZ_EXPORT_STARTED",
    "npz_export_complete": "NPZ_EXPORT_COMPLETE",
    # Dec 2025: NPZ combination events (quality-weighted data combination for training)
    "npz_combination_started": "NPZ_COMBINATION_STARTED",
    "npz_combination_complete": "NPZ_COMBINATION_COMPLETE",
    "npz_combination_failed": "NPZ_COMBINATION_FAILED",
    "cluster_sync_complete": "CLUSTER_SYNC_COMPLETE",
    "model_sync_complete": "MODEL_SYNC_COMPLETE",
    # Optimization events
    "cmaes_complete": "CMAES_COMPLETE",
    "pbt_complete": "PBT_COMPLETE",
    "nas_complete": "NAS_COMPLETE",
    # Utility events
    "iteration_complete": "ITERATION_COMPLETE",
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_data_event_type(stage_event: str) -> str | None:
    """Convert a StageEvent to DataEventType.

    Args:
        stage_event: Stage event name (e.g., "training_complete")

    Returns:
        DataEventType value (e.g., "training_completed") or None if unmapped
    """
    return STAGE_TO_DATA_EVENT_MAP.get(stage_event)


def get_cross_process_event_type(
    event_type: str,
    source: str = "data",
) -> str | None:
    """Convert an event type to CrossProcess format.

    Args:
        event_type: Event type from any bus
        source: Source bus ("data" or "stage")

    Returns:
        CrossProcess event type (UPPERCASE_SNAKE_CASE) or None if unmapped
    """
    if source == "stage":
        return STAGE_TO_CROSS_PROCESS_MAP.get(event_type)
    return DATA_TO_CROSS_PROCESS_MAP.get(event_type)


def get_stage_event_type(data_event: str) -> str | None:
    """Convert a DataEventType to StageEvent.

    Args:
        data_event: Data event name (e.g., "training_completed")

    Returns:
        StageEvent value (e.g., "training_complete") or None if unmapped
    """
    return DATA_TO_STAGE_EVENT_MAP.get(data_event)


def is_mapped_event(event_type: str) -> bool:
    """Check if an event type has mappings defined.

    Args:
        event_type: Event type string (any case/format)

    Returns:
        True if the event type exists in any mapping
    """
    lower = event_type.lower()
    upper = event_type.upper()

    return (
        lower in STAGE_TO_DATA_EVENT_MAP
        or lower in DATA_TO_CROSS_PROCESS_MAP
        or lower in DATA_TO_STAGE_EVENT_MAP
        or upper in CROSS_PROCESS_TO_DATA_MAP
    )


def get_all_event_types() -> set[str]:
    """Get all known event types across all mappings.

    Returns:
        Set of all event type strings (lowercase and uppercase)
    """
    all_types: set[str] = set()

    # Stage events
    all_types.update(STAGE_TO_DATA_EVENT_MAP.keys())
    all_types.update(STAGE_TO_CROSS_PROCESS_MAP.keys())

    # Data events
    all_types.update(DATA_TO_CROSS_PROCESS_MAP.keys())
    all_types.update(DATA_TO_STAGE_EVENT_MAP.keys())

    # Cross-process events
    all_types.update(CROSS_PROCESS_TO_DATA_MAP.keys())

    return all_types


# =============================================================================
# Validation
# =============================================================================


def validate_mappings() -> list[str]:
    """Validate mapping consistency and return any warnings.

    Checks:
        1. Reverse mappings are consistent
        2. No orphaned event types
        3. Cross-process naming convention (UPPERCASE)

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings: list[str] = []

    # Check STAGE_TO_DATA reverse mapping
    for stage, data in STAGE_TO_DATA_EVENT_MAP.items():
        if data in DATA_TO_STAGE_EVENT_MAP:
            reverse_stage = DATA_TO_STAGE_EVENT_MAP[data]
            # Allow many-to-one (multiple stage events map to same data event)
            # but the reverse should map back to a valid stage event
            if reverse_stage not in STAGE_TO_DATA_EVENT_MAP:
                warnings.append(
                    f"Reverse mapping inconsistency: {data} -> {reverse_stage} "
                    f"but {reverse_stage} not in STAGE_TO_DATA_EVENT_MAP"
                )

    # Check cross-process naming convention
    for cp_event in CROSS_PROCESS_TO_DATA_MAP:
        if cp_event != cp_event.upper():
            warnings.append(
                f"CrossProcess event '{cp_event}' should be UPPERCASE"
            )

    return warnings


__all__ = [
    # Stage <-> Data mappings
    "DATA_TO_STAGE_EVENT_MAP",
    "STAGE_TO_DATA_EVENT_MAP",
    # Data <-> CrossProcess mappings
    "CROSS_PROCESS_TO_DATA_MAP",
    "DATA_TO_CROSS_PROCESS_MAP",
    # Stage -> CrossProcess mapping
    "STAGE_TO_CROSS_PROCESS_MAP",
    # Helper functions
    "get_all_event_types",
    "get_cross_process_event_type",
    "get_data_event_type",
    "get_stage_event_type",
    "is_mapped_event",
    "validate_mappings",
]
