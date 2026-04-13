"""Canonical training coordination package exports.

This package exposes the public training coordination API directly from the
canonical modules instead of routing callers through deprecated shim
submodules.

Usage:
    from app.coordination.training import TrainingCoordinator, PriorityJobScheduler
"""

from __future__ import annotations

import importlib

_EXPORTS: dict[str, tuple[str, str]] = {
    # training_coordinator
    "TrainingCoordinator": ("app.coordination.training_coordinator", "TrainingCoordinator"),
    "get_training_coordinator": ("app.coordination.training_coordinator", "get_training_coordinator"),
    "get_training_status": ("app.coordination.training_coordinator", "get_training_status"),
    "TrainingJob": ("app.coordination.training_coordinator", "TrainingJob"),
    "can_train": ("app.coordination.training_coordinator", "can_train"),
    "request_training_slot": ("app.coordination.training_coordinator", "request_training_slot"),
    "release_training_slot": ("app.coordination.training_coordinator", "release_training_slot"),
    "wire_training_events": ("app.coordination.training_coordinator", "wire_training_events"),
    # selfplay_orchestrator
    "SelfplayOrchestrator": ("app.coordination.selfplay_orchestrator", "SelfplayOrchestrator"),
    "get_selfplay_orchestrator": ("app.coordination.selfplay_orchestrator", "get_selfplay_orchestrator"),
    "get_selfplay_stats": ("app.coordination.selfplay_orchestrator", "get_selfplay_stats"),
    "is_large_board": ("app.coordination.selfplay_orchestrator", "is_large_board"),
    "get_engine_for_board": ("app.coordination.selfplay_orchestrator", "get_engine_for_board"),
    "get_simulation_budget_for_board": ("app.coordination.selfplay_orchestrator", "get_simulation_budget_for_board"),
    "SelfplayStats": ("app.coordination.selfplay_orchestrator", "SelfplayStats"),
    "SelfplayType": ("app.coordination.selfplay_orchestrator", "SelfplayType"),
    "wire_selfplay_events": ("app.coordination.selfplay_orchestrator", "wire_selfplay_events"),
    # job_scheduler
    "PriorityJobScheduler": ("app.coordination.job_scheduler", "PriorityJobScheduler"),
    "JobPriority": ("app.coordination.job_scheduler", "JobPriority"),
    "ScheduledJob": ("app.coordination.job_scheduler", "ScheduledJob"),
    "HostDeadJobMigrator": ("app.coordination.job_scheduler", "HostDeadJobMigrator"),
    # duration_scheduler
    "DurationScheduler": ("app.coordination.duration_scheduler", "DurationScheduler"),
    "ScheduledTask": ("app.coordination.duration_scheduler", "ScheduledTask"),
    "TaskDurationRecord": ("app.coordination.duration_scheduler", "TaskDurationRecord"),
    "estimate_task_duration": ("app.coordination.duration_scheduler", "estimate_task_duration"),
    "can_schedule_task": ("app.coordination.duration_scheduler", "can_schedule_task"),
    # work_distributor
    "WorkDistributor": ("app.coordination.work_distributor", "WorkDistributor"),
    # unified_scheduler
    "UnifiedScheduler": ("app.coordination.unified_scheduler", "UnifiedScheduler"),
    "get_unified_scheduler": ("app.coordination.unified_scheduler", "get_scheduler"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve documented package exports lazily."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
