"""Selfplay scheduling package.

Decomposes the selfplay_scheduler.py monolith (4,743 LOC) into focused modules:

- scheduler.py: Main SelfplayScheduler orchestration
- priority_calculator.py: Priority scoring logic
- node_allocator.py: Game allocation to nodes
- quality_budget.py: Quality tracking and Gumbel budgets
- event_handlers.py: Event handler implementations
- status_monitoring.py: Health checks and metrics

Sprint 17.3 (Jan 4, 2026): Initial decomposition phase.

Usage:
    # The main class is re-exported for backward compatibility
    from app.coordination.selfplay import SelfplayScheduler

    # Or use the original import path (deprecated)
    from app.coordination.selfplay_scheduler import SelfplayScheduler
"""

# Re-export main class for backward compatibility
# During decomposition, we still import from the monolith
from app.coordination.selfplay_scheduler import (
    SelfplayScheduler,
    get_selfplay_scheduler,
    reset_selfplay_scheduler,
)

# Types are in selfplay_priority_types.py
from app.coordination.selfplay_priority_types import (
    ConfigPriority,
    DynamicWeights,
)

from app.coordination.priority_calculator import (
    PriorityInputs,
    ClusterState,
)

# Mixin classes for decomposed functionality
from app.coordination.selfplay.status_monitoring import StatusMonitoringMixin

# Event handler documentation and registry
from app.coordination.selfplay.event_handlers import (
    CORE_EVENTS,
    QUALITY_EVENTS,
    P2P_EVENTS,
    BACKPRESSURE_EVENTS,
    VELOCITY_EVENTS,
    ARCHITECTURE_EVENTS,
    ALL_EVENTS,
    EVENT_HANDLERS,
    get_event_category,
    is_event_handled,
)

__all__ = [
    "SelfplayScheduler",
    "get_selfplay_scheduler",
    "reset_selfplay_scheduler",
    # Types
    "ConfigPriority",
    "PriorityInputs",
    "DynamicWeights",
    "ClusterState",
    # Mixins (Sprint 17.3)
    "StatusMonitoringMixin",
    # Event registry (Sprint 17.3)
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
