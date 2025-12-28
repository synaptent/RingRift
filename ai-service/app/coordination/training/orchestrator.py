"""Training orchestration (December 2025).

DEPRECATED: This is a re-export module for backward compatibility.
Import directly from the source modules instead:

    # Instead of:
    from app.coordination.training.orchestrator import TrainingCoordinator

    # Use:
    from app.coordination.training_coordinator import TrainingCoordinator
    from app.coordination.selfplay_orchestrator import SelfplayOrchestrator

This module will be removed in Q2 2026.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.training.orchestrator is deprecated. "
    "Import directly from app.coordination.training_coordinator or "
    "app.coordination.selfplay_orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from training_coordinator
from app.coordination.training_coordinator import (
    TrainingCoordinator,
    get_training_coordinator,
    get_training_status,
    TrainingJob,
    can_train,
    request_training_slot,
    release_training_slot,
    wire_training_events,
)

# Re-export from selfplay_orchestrator
from app.coordination.selfplay_orchestrator import (
    SelfplayOrchestrator,
    get_selfplay_orchestrator,
    get_selfplay_stats,
    is_large_board,
    get_engine_for_board,
    get_simulation_budget_for_board,
    SelfplayStats,
    SelfplayType,
    wire_selfplay_events,
)

__all__ = [
    # From training_coordinator
    "TrainingCoordinator",
    "get_training_coordinator",
    "get_training_status",
    "TrainingJob",
    "can_train",
    "request_training_slot",
    "release_training_slot",
    "wire_training_events",
    # From selfplay_orchestrator
    "SelfplayOrchestrator",
    "get_selfplay_orchestrator",
    "get_selfplay_stats",
    "is_large_board",
    "get_engine_for_board",
    "get_simulation_budget_for_board",
    "SelfplayStats",
    "SelfplayType",
    "wire_selfplay_events",
]
