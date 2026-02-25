"""Selfplay scheduler mixins extracted from selfplay_scheduler.py.

February 2026: P0 decomposition of the monolithic SelfplayScheduler (4,425 LOC)
into focused mixin modules for better maintainability.

Modules:
- engine_selection: Engine mode selection and diversity tracking
- job_targeting: Per-node job targeting and resource allocation
- event_handlers: Event subscription and handler methods
- priority: Priority calculation and architecture selection
- config_selection: Core config selection (pick_weighted_config) and boost management
- dispatch: Spawn verification and dispatch tracking
"""

from .config_selection import ConfigSelectionMixin
from .dispatch import DispatchTrackingMixin
from .engine_selection import DiversityMetrics, EngineSelectionMixin
from .event_handlers import EventHandlersMixin
from .job_targeting import JobTargetingMixin
from .priority import ArchitectureSelectionMixin, PriorityCalculatorMixin

__all__ = [
    "ArchitectureSelectionMixin",
    "ConfigSelectionMixin",
    "DispatchTrackingMixin",
    "DiversityMetrics",
    "EngineSelectionMixin",
    "EventHandlersMixin",
    "JobTargetingMixin",
    "PriorityCalculatorMixin",
]
