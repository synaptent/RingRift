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

from __future__ import annotations

from typing import TYPE_CHECKING

# Types are in selfplay_priority_types.py - these don't cause circular imports
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
from app.coordination.selfplay.quality_signal_handler import SelfplayQualitySignalMixin
from app.coordination.selfplay.velocity_mixin import SelfplayVelocityMixin
from app.coordination.selfplay.data_providers import DataProviderMixin
from app.coordination.selfplay.node_targeting import NodeTargetingMixin
from app.coordination.selfplay.idle_injection import IdleWorkInjectionMixin
from app.coordination.selfplay.architecture_tracker_mixin import ArchitectureTrackerMixin

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

# Sprint 17.9: Allocation engine extraction
from app.coordination.selfplay.allocation_engine import (
    AllocationContext,
    AllocationEngine,
    AllocationResult,
)

# January 2026: Priority boost multipliers extraction
from app.coordination.selfplay.priority_boosts import (
    get_cascade_priority,
    get_improvement_boosts,
    get_momentum_multipliers,
    get_architecture_boosts,
)

# January 2026: Allocation event emission helpers
from app.coordination.selfplay.allocation_events import (
    emit_allocation_updated,
    emit_starvation_alert,
    emit_idle_node_work_injected,
)

# Lazy import for SelfplayScheduler to avoid circular imports
# When quality_signal_handler.py imports from this package, the scheduler
# hasn't been fully loaded yet
if TYPE_CHECKING:
    from app.coordination.selfplay_scheduler import (
        SelfplayScheduler,
        get_selfplay_scheduler,
        reset_selfplay_scheduler,
    )


def __getattr__(name: str):
    """Lazy import for SelfplayScheduler and related functions."""
    if name in ("SelfplayScheduler", "get_selfplay_scheduler", "reset_selfplay_scheduler"):
        from app.coordination.selfplay_scheduler import (
            SelfplayScheduler,
            get_selfplay_scheduler,
            reset_selfplay_scheduler,
        )
        return {
            "SelfplayScheduler": SelfplayScheduler,
            "get_selfplay_scheduler": get_selfplay_scheduler,
            "reset_selfplay_scheduler": reset_selfplay_scheduler,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SelfplayScheduler",
    "get_selfplay_scheduler",
    "reset_selfplay_scheduler",
    # Types
    "ConfigPriority",
    "PriorityInputs",
    "DynamicWeights",
    "ClusterState",
    # Mixins (Sprint 17.3+)
    "StatusMonitoringMixin",
    "SelfplayQualitySignalMixin",
    "SelfplayVelocityMixin",
    "DataProviderMixin",
    "NodeTargetingMixin",
    "IdleWorkInjectionMixin",
    "ArchitectureTrackerMixin",
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
    # Allocation engine (Sprint 17.9)
    "AllocationContext",
    "AllocationEngine",
    "AllocationResult",
    # Priority boosts (Jan 2026)
    "get_cascade_priority",
    "get_improvement_boosts",
    "get_momentum_multipliers",
    "get_architecture_boosts",
    # Allocation events (Jan 2026)
    "emit_allocation_updated",
    "emit_starvation_alert",
    "emit_idle_node_work_injected",
]
