"""Training scheduling (December 2025).

DEPRECATED: This is a re-export module for backward compatibility.
Import directly from the source modules instead:

    # Instead of:
    from app.coordination.training.scheduler import PriorityJobScheduler

    # Use:
    from app.coordination.job_scheduler import PriorityJobScheduler
    from app.coordination.duration_scheduler import DurationScheduler
    from app.coordination.unified_scheduler import UnifiedScheduler

This module will be removed in Q2 2026.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "app.coordination.training.scheduler is deprecated. "
    "Import directly from app.coordination.job_scheduler, "
    "app.coordination.duration_scheduler, or app.coordination.unified_scheduler instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from job_scheduler
from app.coordination.job_scheduler import (
    PriorityJobScheduler,
    JobPriority,
    ScheduledJob,
    HostDeadJobMigrator,
)

# Re-export from duration_scheduler
from app.coordination.duration_scheduler import (
    DurationScheduler,
    ScheduledTask,
    TaskDurationRecord,
    estimate_task_duration,
    can_schedule_task,
)

# Re-export from work_distributor
from app.coordination.work_distributor import (
    WorkDistributor,
)

# Re-export from unified_scheduler
from app.coordination.unified_scheduler import (
    UnifiedScheduler,
    get_scheduler as get_unified_scheduler,
)

__all__ = [
    # From job_scheduler
    "PriorityJobScheduler",
    "JobPriority",
    "ScheduledJob",
    "HostDeadJobMigrator",
    # From duration_scheduler
    "DurationScheduler",
    "ScheduledTask",
    "TaskDurationRecord",
    "estimate_task_duration",
    "can_schedule_task",
    # From work_distributor
    "WorkDistributor",
    # From unified_scheduler
    "UnifiedScheduler",
    "get_unified_scheduler",
]
