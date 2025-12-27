"""Core coordination modules.

This package provides core coordination infrastructure:
- events: Unified event routing and emission
- tasks: Task lifecycle and coordination
- pipeline: Training pipeline orchestration

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.core.events import UnifiedEventRouter
    from app.coordination.core.tasks import TaskCoordinator
    from app.coordination.core.pipeline import DataPipelineOrchestrator
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("events", "tasks", "pipeline"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
