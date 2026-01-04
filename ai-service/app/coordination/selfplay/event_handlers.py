"""Event handler documentation for SelfplayScheduler.

Sprint 17.3 (Jan 4, 2026): Documents the 32 event handlers in SelfplayScheduler.

This module provides:
1. Documentation of which events the scheduler handles
2. Event type categorization
3. Handler registry for introspection

The actual handler implementations remain in selfplay_scheduler.py until
a more thorough refactoring can be done. These handlers depend heavily on
internal scheduler state and require careful extraction.

Event Categories:
-----------------
CORE: Essential events that drive the scheduling loop
QUALITY: Data quality and training feedback
P2P: Cluster health and node management
BACKPRESSURE: Flow control and load management
VELOCITY: Elo velocity and progress tracking
ARCHITECTURE: Neural network architecture selection

Usage:
    from app.coordination.selfplay.event_handlers import (
        CORE_EVENTS,
        QUALITY_EVENTS,
        get_event_category,
    )

    # Check which category an event belongs to
    category = get_event_category("TRAINING_COMPLETED")
    # Returns: "CORE"
"""

from __future__ import annotations

from typing import Literal

# Event categories
EventCategory = Literal[
    "CORE",
    "QUALITY",
    "P2P",
    "BACKPRESSURE",
    "VELOCITY",
    "ARCHITECTURE",
]

# Core events that drive the scheduling loop
CORE_EVENTS = frozenset({
    "SELFPLAY_COMPLETE",
    "TRAINING_COMPLETED",
    "MODEL_PROMOTED",
    "SELFPLAY_TARGET_UPDATED",
    "CURRICULUM_REBALANCED",
    "SELFPLAY_RATE_CHANGED",
})

# Quality and training feedback events
QUALITY_EVENTS = frozenset({
    "QUALITY_DEGRADED",
    "TRAINING_BLOCKED_BY_QUALITY",
    "QUALITY_FEEDBACK_ADJUSTED",
    "LOW_QUALITY_DATA_WARNING",
    "REGRESSION_DETECTED",
    "OPPONENT_MASTERED",
    "TRAINING_EARLY_STOPPED",
})

# P2P cluster health events
P2P_EVENTS = frozenset({
    "NODE_UNHEALTHY",
    "NODE_RECOVERED",
    "NODE_ACTIVATED",
    "P2P_NODE_DEAD",
    "P2P_CLUSTER_UNHEALTHY",
    "P2P_CLUSTER_HEALTHY",
    "HOST_OFFLINE",
    "NODE_TERMINATED",
    "VOTER_DEMOTED",
    "VOTER_PROMOTED",
    "CIRCUIT_RESET",
    "P2P_RESTARTED",
})

# Backpressure and load management events
BACKPRESSURE_EVENTS = frozenset({
    "BACKPRESSURE_ACTIVATED",
    "BACKPRESSURE_RELEASED",
    "EVALUATION_BACKPRESSURE",
    "EVALUATION_BACKPRESSURE_RELEASED",
    "NODE_OVERLOADED",
})

# Elo velocity and progress events
VELOCITY_EVENTS = frozenset({
    "ELO_VELOCITY_CHANGED",
    "ELO_UPDATED",
    "EXPLORATION_BOOST",
    "CURRICULUM_ADVANCED",
    "ADAPTIVE_PARAMS_CHANGED",
    "PROGRESS_STALL_DETECTED",
    "PROGRESS_RECOVERED",
})

# Architecture selection events
ARCHITECTURE_EVENTS = frozenset({
    "ARCHITECTURE_WEIGHTS_UPDATED",
})

# All handled events
ALL_EVENTS = (
    CORE_EVENTS |
    QUALITY_EVENTS |
    P2P_EVENTS |
    BACKPRESSURE_EVENTS |
    VELOCITY_EVENTS |
    ARCHITECTURE_EVENTS
)


def get_event_category(event_name: str) -> EventCategory | None:
    """Get the category for an event type.

    Args:
        event_name: Event type name (e.g., "TRAINING_COMPLETED")

    Returns:
        Event category or None if not handled
    """
    if event_name in CORE_EVENTS:
        return "CORE"
    elif event_name in QUALITY_EVENTS:
        return "QUALITY"
    elif event_name in P2P_EVENTS:
        return "P2P"
    elif event_name in BACKPRESSURE_EVENTS:
        return "BACKPRESSURE"
    elif event_name in VELOCITY_EVENTS:
        return "VELOCITY"
    elif event_name in ARCHITECTURE_EVENTS:
        return "ARCHITECTURE"
    return None


def is_event_handled(event_name: str) -> bool:
    """Check if an event type is handled by SelfplayScheduler.

    Args:
        event_name: Event type name

    Returns:
        True if the scheduler has a handler for this event
    """
    return event_name in ALL_EVENTS


# Handler method names for each event (for documentation)
EVENT_HANDLERS: dict[str, str] = {
    # Core events
    "SELFPLAY_COMPLETE": "_on_selfplay_complete",
    "TRAINING_COMPLETED": "_on_training_complete",
    "MODEL_PROMOTED": "_on_promotion_complete",
    "SELFPLAY_TARGET_UPDATED": "_on_selfplay_target_updated",
    "CURRICULUM_REBALANCED": "_on_curriculum_rebalanced",
    "SELFPLAY_RATE_CHANGED": "_on_selfplay_rate_changed",
    # Quality events
    "QUALITY_DEGRADED": "_on_quality_degraded",
    "TRAINING_BLOCKED_BY_QUALITY": "_on_training_blocked_by_quality",
    "QUALITY_FEEDBACK_ADJUSTED": "_on_quality_feedback_adjusted",
    "LOW_QUALITY_DATA_WARNING": "_on_low_quality_warning",
    "REGRESSION_DETECTED": "_on_regression_detected",
    "OPPONENT_MASTERED": "_on_opponent_mastered",
    "TRAINING_EARLY_STOPPED": "_on_training_early_stopped",
    # P2P events
    "NODE_UNHEALTHY": "_on_node_unhealthy",
    "NODE_RECOVERED": "_on_node_recovered",
    "NODE_ACTIVATED": "_on_node_recovered",
    "P2P_NODE_DEAD": "_on_node_unhealthy",
    "P2P_CLUSTER_UNHEALTHY": "_on_cluster_unhealthy",
    "P2P_CLUSTER_HEALTHY": "_on_cluster_healthy",
    "HOST_OFFLINE": "_on_host_offline",
    "NODE_TERMINATED": "_on_host_offline",
    "VOTER_DEMOTED": "_on_voter_demoted",
    "VOTER_PROMOTED": "_on_voter_promoted",
    "CIRCUIT_RESET": "_on_circuit_reset",
    "P2P_RESTARTED": "_on_p2p_restarted",
    # Backpressure events
    "BACKPRESSURE_ACTIVATED": "_on_backpressure_activated",
    "BACKPRESSURE_RELEASED": "_on_backpressure_released",
    "EVALUATION_BACKPRESSURE": "_on_evaluation_backpressure",
    "EVALUATION_BACKPRESSURE_RELEASED": "_on_backpressure_released",
    "NODE_OVERLOADED": "_on_node_overloaded",
    # Velocity events
    "ELO_VELOCITY_CHANGED": "_on_elo_velocity_changed",
    "ELO_UPDATED": "_on_elo_updated",
    "EXPLORATION_BOOST": "_on_exploration_boost",
    "CURRICULUM_ADVANCED": "_on_curriculum_advanced",
    "ADAPTIVE_PARAMS_CHANGED": "_on_adaptive_params_changed",
    "PROGRESS_STALL_DETECTED": "_on_progress_stall",
    "PROGRESS_RECOVERED": "_on_progress_recovered",
    # Architecture events
    "ARCHITECTURE_WEIGHTS_UPDATED": "_on_architecture_weights_updated",
}


__all__ = [
    "EventCategory",
    "CORE_EVENTS",
    "QUALITY_EVENTS",
    "P2P_EVENTS",
    "BACKPRESSURE_EVENTS",
    "VELOCITY_EVENTS",
    "ARCHITECTURE_EVENTS",
    "ALL_EVENTS",
    "EVENT_HANDLERS",
    "get_event_category",
    "is_event_handled",
]
