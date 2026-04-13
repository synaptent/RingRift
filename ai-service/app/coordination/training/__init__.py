"""Training coordination modules.

This package consolidates training-related coordination:
- orchestrator: Training and selfplay orchestration
- scheduler: Job and duration scheduling

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.training.orchestrator import TrainingCoordinator
    from app.coordination.training.scheduler import PriorityJobScheduler
"""

from __future__ import annotations

import importlib

_SUBMODULES = ("orchestrator", "scheduler")

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Resolve the documented training package submodules lazily."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
