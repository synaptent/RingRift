"""Priority-based job scheduler for the unified AI improvement loop.

This module provides priority-based job scheduling with Elo-driven curriculum
learning to ensure critical jobs run first while balancing training data.

Consolidated from:
- scripts/archive/cluster_orchestrator.py (PriorityJobScheduler, curriculum functions)

Usage:
    from app.coordination.job_scheduler import (
        PriorityJobScheduler,
        JobPriority,
        ScheduledJob,
        get_scheduler,
    )

    # Queue jobs with different priorities
    scheduler = get_scheduler()
    scheduler.schedule(ScheduledJob(
        job_type="promotion_evaluation",
        priority=JobPriority.CRITICAL,
        config={"model_id": "latest"},
        requires_gpu=True,
    ))
    scheduler.schedule(ScheduledJob(
        job_type="selfplay",
        priority=JobPriority.NORMAL,
        config={"board": "square8", "players": 2},
    ))

    # Get next job to run
    job, host = scheduler.next_job(hosts, statuses)
"""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, TypeVar

from app.config.thresholds import DISK_PRODUCTION_HALT_PERCENT

logger = logging.getLogger(__name__)

# Duration scheduler integration (December 2025 consolidation)
# Provides historical duration estimation and host availability tracking
try:
    from app.coordination.duration_scheduler import (
        estimate_task_duration,
        get_resource_availability,
        record_task_completion,
    )
    _DURATION_SCHEDULER_AVAILABLE = True
except ImportError:
    _DURATION_SCHEDULER_AVAILABLE = False
    # Fallback stubs for when duration_scheduler is not available
    def estimate_task_duration(task_type: str, config: str = "", host: str = "") -> float:
        return 3600.0  # Default 1 hour
    def get_resource_availability(host: str, task_type: str = "") -> tuple[bool, float]:
        return (True, 0.0)  # Always available
    def record_task_completion(
        task_type: str, host: str, started_at: float, completed_at: float, success: bool = True
    ) -> None:
        pass

# Resource thresholds - import from centralized thresholds (December 2025)
try:
    from app.config.thresholds import (
        CPU_CRITICAL_PERCENT,
        CPU_WARNING_PERCENT,
        ELO_UNDERSERVED_THRESHOLD as ELO_UNDERSERVED_THRESHOLD_CONFIG,
        GPU_CRITICAL_PERCENT,
        GPU_WARNING_PERCENT,
    )
    TARGET_GPU_UTILIZATION_MIN = GPU_WARNING_PERCENT
    TARGET_GPU_UTILIZATION_MAX = GPU_CRITICAL_PERCENT
    TARGET_CPU_UTILIZATION_MIN = CPU_WARNING_PERCENT
    TARGET_CPU_UTILIZATION_MAX = CPU_CRITICAL_PERCENT
    ELO_UNDERSERVED_THRESHOLD = ELO_UNDERSERVED_THRESHOLD_CONFIG
except ImportError:
    # Try coordination_defaults as fallback (December 2025)
    try:
        from app.config.coordination_defaults import UtilizationDefaults
        TARGET_GPU_UTILIZATION_MIN = UtilizationDefaults.GPU_TARGET_MIN
        TARGET_GPU_UTILIZATION_MAX = UtilizationDefaults.GPU_TARGET_MAX
        TARGET_CPU_UTILIZATION_MIN = UtilizationDefaults.CPU_TARGET_MIN
        TARGET_CPU_UTILIZATION_MAX = UtilizationDefaults.CPU_TARGET_MAX
    except ImportError:
        # Hardcoded fallback for testing/standalone use
        TARGET_GPU_UTILIZATION_MIN = 60
        TARGET_GPU_UTILIZATION_MAX = 80
        TARGET_CPU_UTILIZATION_MIN = 60
        TARGET_CPU_UTILIZATION_MAX = 80
    ELO_UNDERSERVED_THRESHOLD = 100

# Use centralized defaults for memory threshold (December 2025)
try:
    from app.config.coordination_defaults import UtilizationDefaults as _UD
    MIN_MEMORY_GB_FOR_TASKS = _UD.MIN_MEMORY_GB
except ImportError:
    MIN_MEMORY_GB_FOR_TASKS = 64  # Skip nodes with less than this to avoid OOM

# Resource optimizer PID controller integration (Phase 21.2 - Dec 2025)
try:
    from app.coordination.resource_optimizer import (
        get_resource_optimizer,
        should_scale_down,
    )
    _RESOURCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    _RESOURCE_OPTIMIZER_AVAILABLE = False
    get_resource_optimizer = None
    should_scale_down = None

# Elo curriculum configuration
ELO_CURRICULUM_ENABLED = True

# Default Elo database path
DEFAULT_ELO_DB_PATH = Path(__file__).parent.parent.parent / "data" / "unified_elo.db"


class JobPriority(IntEnum):
    """Priority levels for job scheduling.

    Lower values = higher priority.
    Critical jobs (promotion evaluation) always run first.
    """

    CRITICAL = 0  # Promotion evaluation, regression tests
    HIGH = 1  # Shadow tournaments, Elo calibration
    NORMAL = 2  # Regular selfplay, training
    LOW = 3  # Backfill, optional data collection


# P2.1 (Dec 2025): Fair allocation quotas - max GPU time any single config can use
# This prevents one config (e.g., square8_2p) from monopolizing cluster resources
DEFAULT_CONFIG_QUOTA = 0.30  # 30% max per config by default
CONFIG_QUOTAS: dict[str, float] = {
    # Can be overridden per config if needed
    # "hex8_2p": 0.40,  # Example: give hex8_2p more allocation
}

# P2.2 (Dec 2025): Starvation prevention - boost priority after N hours
STARVATION_THRESHOLD_HOURS = 4.0  # After 4 hours, boost priority

# P2.3 (Dec 2025): Cost-aware scheduling - prefer cheaper providers for low priority work
# Lower cost = better. Scale is relative (not actual $/hr)
PROVIDER_COSTS: dict[str, float] = {
    "vast": 1.0,       # Cheapest - spot instances
    "vultr": 1.5,      # vGPU, moderate cost
    "runpod": 2.0,     # On-demand GPUs
    "nebius": 2.5,     # Cloud GPUs
    "lambda": 3.0,     # Premium datacenter GPUs (when online)
    "unknown": 2.0,    # Default to middle tier
}

# P2.4 (Dec 2025): Ephemeral node optimization
# Ephemeral providers can be terminated at any time - only schedule short jobs
EPHEMERAL_PROVIDERS: set[str] = {"vast"}  # Vast.ai spot instances
MAX_EPHEMERAL_JOB_DURATION_SECONDS = 1800  # 30 minutes max for ephemeral hosts
# Job types that should NEVER run on ephemeral hosts (too important to lose)
EPHEMERAL_BLOCKED_JOB_TYPES: set[str] = {"training", "promotion", "evaluation"}


@dataclass
class ScheduledJob:
    """A job to be scheduled on the cluster."""

    job_type: str  # selfplay, tournament, training, promotion, etc.
    priority: JobPriority
    config: dict[str, Any] = field(default_factory=dict)
    host_preference: str | None = None  # Preferred host name or None
    requires_gpu: bool = False
    estimated_duration_seconds: int = 3600
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    job_id: str | None = None

    def __lt__(self, other: ScheduledJob) -> bool:
        """Enable sorting by priority."""
        return self.priority < other.priority

    def __hash__(self) -> int:
        """Hash by job_id or id if no job_id."""
        return hash(self.job_id or id(self))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "job_type": self.job_type,
            "priority": self.priority.name,
            "config": self.config,
            "host_preference": self.host_preference,
            "requires_gpu": self.requires_gpu,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "job_id": self.job_id,
        }


# Type var for host config (to avoid circular imports)
HostConfigT = TypeVar("HostConfigT")
HostStatusT = TypeVar("HostStatusT")


class PriorityJobScheduler:
    """Priority-based job scheduler for the unified AI improvement loop.

    Ensures critical jobs (promotion evaluation, regression tests) run
    first, while regular selfplay jobs fill remaining capacity.

    Features:
    - Priority-based job queue (CRITICAL > HIGH > NORMAL > LOW)
    - Host matching based on GPU requirements and capacity
    - Capacity reservation for training jobs
    - Queue statistics and monitoring
    """

    def __init__(self, max_queue_size: int = 1000):
        """Initialize the scheduler.

        Args:
            max_queue_size: Maximum number of jobs in the queue
        """
        self._queue: list[ScheduledJob] = []
        self._running: dict[str, ScheduledJob] = {}  # host_name -> job
        self._max_queue_size = max_queue_size
        self._completed_jobs: list[ScheduledJob] = []
        self._max_completed_history = 100

        # P2.1 (Dec 2025): Fair allocation tracking
        # Tracks running time per config in current window (1 hour rolling)
        self._config_allocation: dict[str, float] = {}  # config_key -> seconds
        self._allocation_window_start: float = time.time()
        self._allocation_window_seconds: float = 3600.0  # 1 hour window

        # P2.2 (Dec 2025): Starvation tracking - jobs waiting too long
        self._starvation_boosted: set[str] = set()  # job_ids that got boosted

    def schedule(self, job: ScheduledJob) -> bool:
        """Add a job to the scheduling queue.

        Args:
            job: The job to schedule

        Returns:
            True if job was added, False if queue is full
        """
        if len(self._queue) >= self._max_queue_size:
            logger.warning(
                f"Job queue full ({self._max_queue_size}), rejecting {job.job_type}"
            )
            return False

        self._queue.append(job)
        self._queue.sort()  # Sort by priority
        logger.debug(
            f"Scheduled {job.job_type} job (priority={job.priority.name}), "
            f"queue size={len(self._queue)}"
        )
        return True

    def _get_config_key(self, job: ScheduledJob) -> str:
        """Extract config key from job for quota tracking (P2.1).

        Args:
            job: The scheduled job

        Returns:
            Config key like "hex8_2p" or "unknown"
        """
        board = job.config.get("board", job.config.get("board_type", "unknown"))
        players = job.config.get("players", job.config.get("num_players", 0))
        if board == "unknown" or players == 0:
            return "unknown"
        return f"{board}_{players}p"

    def _reset_allocation_window_if_needed(self) -> None:
        """Reset allocation window if it has expired (P2.1)."""
        now = time.time()
        if now - self._allocation_window_start > self._allocation_window_seconds:
            logger.debug("[JobScheduler] Resetting allocation window")
            self._config_allocation.clear()
            self._allocation_window_start = now

    def _get_total_allocation(self) -> float:
        """Get total allocation seconds in current window."""
        return sum(self._config_allocation.values())

    def _is_over_quota(self, config_key: str) -> bool:
        """Check if a config has exceeded its allocation quota (P2.1).

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            True if config is over quota
        """
        if config_key == "unknown":
            return False  # Don't block unknown configs

        self._reset_allocation_window_if_needed()

        total = self._get_total_allocation()
        if total <= 0:
            return False  # No allocation yet

        config_alloc = self._config_allocation.get(config_key, 0.0)
        config_ratio = config_alloc / total

        # Get quota for this config (default if not specified)
        quota = CONFIG_QUOTAS.get(config_key, DEFAULT_CONFIG_QUOTA)

        is_over = config_ratio > quota
        if is_over:
            logger.debug(
                f"[JobScheduler] Config {config_key} over quota: "
                f"{config_ratio:.1%} > {quota:.1%}"
            )
        return is_over

    def _update_allocation(self, config_key: str, seconds: float) -> None:
        """Update allocation tracking when a job completes (P2.1).

        Args:
            config_key: Config key like "hex8_2p"
            seconds: Duration of the job in seconds
        """
        if config_key == "unknown":
            return
        self._reset_allocation_window_if_needed()
        self._config_allocation[config_key] = (
            self._config_allocation.get(config_key, 0.0) + seconds
        )
        logger.debug(
            f"[JobScheduler] Updated allocation for {config_key}: "
            f"{self._config_allocation[config_key]:.0f}s"
        )

    def _get_host_cost(self, host_name: str) -> float:
        """Get relative cost for a host based on provider (P2.3).

        Args:
            host_name: Name of the host

        Returns:
            Relative cost (lower = cheaper)
        """
        host_lower = host_name.lower()
        for provider, cost in PROVIDER_COSTS.items():
            if provider in host_lower:
                return cost
        return PROVIDER_COSTS.get("unknown", 2.0)

    def _is_ephemeral_host(self, host_name: str) -> bool:
        """Check if a host is ephemeral (can be terminated at any time) (P2.4).

        Args:
            host_name: Name of the host

        Returns:
            True if host is ephemeral
        """
        host_lower = host_name.lower()
        return any(provider in host_lower for provider in EPHEMERAL_PROVIDERS)

    def _is_job_suitable_for_ephemeral(self, job: ScheduledJob) -> bool:
        """Check if a job is suitable for ephemeral hosts (P2.4).

        Ephemeral hosts can be terminated at any time, so we only schedule:
        - Short-duration jobs (< 30 minutes)
        - Non-critical job types (not training, promotion, evaluation)
        - Jobs with priority NORMAL or LOW (not CRITICAL/HIGH)

        Args:
            job: The scheduled job

        Returns:
            True if job can run on ephemeral hosts
        """
        # Block critical job types
        if job.job_type in EPHEMERAL_BLOCKED_JOB_TYPES:
            return False

        # Block high priority jobs
        if job.priority < JobPriority.NORMAL:
            return False

        # Block long-running jobs
        if job.estimated_duration_seconds > MAX_EPHEMERAL_JOB_DURATION_SECONDS:
            return False

        return True

    def _apply_starvation_prevention(self) -> int:
        """Boost priority for jobs waiting too long (P2.2).

        Returns:
            Number of jobs boosted
        """
        now = time.time()
        threshold_seconds = STARVATION_THRESHOLD_HOURS * 3600
        boosted_count = 0

        for job in self._queue:
            job_id = job.job_id or str(id(job))

            # Skip if already boosted
            if job_id in self._starvation_boosted:
                continue

            # Skip if not waiting long enough
            wait_time = now - job.created_at
            if wait_time < threshold_seconds:
                continue

            # Skip if already high priority
            if job.priority <= JobPriority.HIGH:
                continue

            # Boost priority by 1 level
            old_priority = job.priority
            new_priority_value = max(0, job.priority.value - 1)
            job.priority = JobPriority(new_priority_value)
            self._starvation_boosted.add(job_id)
            boosted_count += 1

            logger.info(
                f"[JobScheduler] Starvation prevention: boosted {job.job_type} "
                f"from {old_priority.name} to {job.priority.name} "
                f"(waited {wait_time / 3600:.1f}h)"
            )

        if boosted_count > 0:
            # Re-sort queue after priority changes
            self._queue.sort()

        return boosted_count

    def next_job(
        self,
        hosts: list[Any],
        statuses: list[Any],
        *,
        host_has_gpu: Callable[[Any], bool] | None = None,
        host_get_name: Callable[[Any], str] | None = None,
        host_get_memory_gb: Callable[[Any], int] | None = None,
        status_get_cpu: Callable[[Any], float] | None = None,
        status_get_disk: Callable[[Any], float] | None = None,
        status_get_memory: Callable[[Any], float] | None = None,
        status_is_reachable: Callable[[Any], bool] | None = None,
    ) -> tuple[ScheduledJob, Any] | None:
        """Get the next job to run and the host to run it on.

        This method is designed to work with any host/status types by accepting
        optional accessor functions. If not provided, it will attempt to access
        attributes directly.

        Args:
            hosts: List of host configurations
            statuses: List of host statuses (same order as hosts)
            host_has_gpu: Function to check if host has GPU
            host_get_name: Function to get host name
            host_get_memory_gb: Function to get host memory in GB
            status_get_cpu: Function to get CPU usage percent
            status_get_disk: Function to get disk usage percent
            status_get_memory: Function to get memory usage percent
            status_is_reachable: Function to check if host is reachable

        Returns:
            (job, host) tuple or None if no suitable match
        """
        if not self._queue:
            return None

        # P2.2 (Dec 2025): Apply starvation prevention before job selection
        self._apply_starvation_prevention()

        # Default accessor functions
        def _has_gpu(h: Any) -> bool:
            if host_has_gpu:
                return host_has_gpu(h)
            return getattr(h, "has_gpu", False)

        def _get_name(h: Any) -> str:
            if host_get_name:
                return host_get_name(h)
            return getattr(h, "name", str(h))

        def _get_memory_gb(h: Any) -> int:
            if host_get_memory_gb:
                return host_get_memory_gb(h)
            return getattr(h, "memory_gb", 0)

        def _get_cpu(s: Any) -> float:
            if status_get_cpu:
                return status_get_cpu(s)
            return getattr(s, "cpu_percent", 0.0)

        def _get_disk(s: Any) -> float:
            if status_get_disk:
                return status_get_disk(s)
            return getattr(s, "disk_percent", 0.0)

        def _get_mem(s: Any) -> float:
            if status_get_memory:
                return status_get_memory(s)
            return getattr(s, "memory_percent", 0.0)

        def _is_reachable(s: Any) -> bool:
            if status_is_reachable:
                return status_is_reachable(s)
            return getattr(s, "reachable", True)

        # Build host availability map
        available_hosts: list[tuple[Any, Any]] = []
        for host, status in zip(hosts, statuses, strict=False):
            if not _is_reachable(status):
                continue
            if _get_disk(status) > DISK_PRODUCTION_HALT_PERCENT:
                continue
            if _get_mem(status) > 80:  # 80% limit enforced 2025-12-16
                continue
            # Skip low-memory hosts to avoid OOM
            mem_gb = _get_memory_gb(host)
            if mem_gb > 0 and mem_gb < MIN_MEMORY_GB_FOR_TASKS:
                continue
            available_hosts.append((host, status))

        if not available_hosts:
            return None

        # Find best match for highest priority job
        for job_idx, job in enumerate(self._queue):
            # P2.1 (Dec 2025): Skip jobs from over-quota configs (unless CRITICAL)
            config_key = self._get_config_key(job)
            if job.priority > JobPriority.CRITICAL and self._is_over_quota(config_key):
                continue

            # P2.3 (Dec 2025): Cost-aware host selection for lower priority jobs
            # Sort hosts by cost (cheapest first) for NORMAL/LOW priority work
            if job.priority >= JobPriority.NORMAL:
                hosts_to_try = sorted(
                    available_hosts,
                    key=lambda x: self._get_host_cost(_get_name(x[0]))
                )
            else:
                # For CRITICAL/HIGH priority, use default order (typically by capability)
                hosts_to_try = available_hosts

            for host, status in hosts_to_try:
                host_name = _get_name(host)

                # Check GPU requirement
                if job.requires_gpu and not _has_gpu(host):
                    continue

                # Check host preference
                if job.host_preference and host_name != job.host_preference:
                    continue

                # Check CPU capacity for selfplay jobs
                if job.job_type == "selfplay" and _get_cpu(status) > TARGET_CPU_UTILIZATION_MAX:
                    continue

                # P2.4 (Dec 2025): Block unsuitable jobs on ephemeral hosts
                if self._is_ephemeral_host(host_name) and not self._is_job_suitable_for_ephemeral(job):
                    logger.debug(
                        f"[JobScheduler] Skipping {job.job_type} on ephemeral host {host_name}: "
                        f"job not suitable (type={job.job_type}, priority={job.priority.name}, "
                        f"duration={job.estimated_duration_seconds}s)"
                    )
                    continue

                # Check duration-based availability (December 2025 consolidation)
                # Uses historical task data to avoid overloading hosts with long-running tasks
                if _DURATION_SCHEDULER_AVAILABLE:
                    is_available, _ = get_resource_availability(host_name, job.job_type)
                    if not is_available:
                        continue

                # Phase 21.2: Check PID controller for resource saturation
                # Skip LOW/NORMAL priority jobs if resources are saturated (should_scale_down)
                if _RESOURCE_OPTIMIZER_AVAILABLE and should_scale_down:
                    resource_type = "gpu" if job.requires_gpu else "cpu"
                    if job.priority >= JobPriority.NORMAL and should_scale_down(resource_type):
                        logger.debug(
                            f"[JobScheduler] Skipping {job.job_type} on {host_name}: "
                            f"PID controller recommends scaling down {resource_type}"
                        )
                        continue

                # Found a match
                self._queue.pop(job_idx)
                job.started_at = time.time()
                # Update duration estimate from historical data (December 2025 consolidation)
                job.estimated_duration_seconds = int(estimate_task_duration(
                    job.job_type,
                    str(job.config) if job.config else "",
                    host_name,
                ))
                self._running[host_name] = job
                return (job, host)

        return None

    def complete_job(self, host_name: str, success: bool = True) -> ScheduledJob | None:
        """Mark a job as completed for a host.

        Args:
            host_name: Name of the host where job completed
            success: Whether the job completed successfully

        Returns:
            The completed job, or None if no job was running on that host
        """
        if host_name in self._running:
            job = self._running.pop(host_name)
            job.completed_at = time.time()

            # Record completion to duration scheduler for historical learning (December 2025)
            if job.started_at and _DURATION_SCHEDULER_AVAILABLE:
                try:
                    record_task_completion(
                        task_type=job.job_type,
                        host=host_name,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                        success=success,
                    )
                except Exception as e:
                    logger.debug(f"Failed to record duration: {e}")

            # P2.1 (Dec 2025): Update allocation tracking for fair quota enforcement
            if job.started_at:
                duration = job.completed_at - job.started_at
                config_key = self._get_config_key(job)
                self._update_allocation(config_key, duration)

            # Keep history of completed jobs
            self._completed_jobs.append(job)
            if len(self._completed_jobs) > self._max_completed_history:
                self._completed_jobs.pop(0)

            return job
        return None

    def cancel_job(self, host_name: str) -> ScheduledJob | None:
        """Cancel a running job on a host.

        Args:
            host_name: Name of the host

        Returns:
            The cancelled job, or None
        """
        return self._running.pop(host_name, None)

    def remove_pending(self, job: ScheduledJob) -> bool:
        """Remove a pending job from the queue.

        Args:
            job: The job to remove

        Returns:
            True if job was found and removed
        """
        try:
            self._queue.remove(job)
            return True
        except ValueError:
            return False

    def get_queue_depth(self) -> int:
        """Get total number of pending jobs in the queue.

        Phase 24.2: Simple accessor for queue depth, used by IdleResourceDaemon
        for queue-aware scaling decisions.

        Returns:
            Total number of jobs waiting in the queue.
        """
        return len(self._queue)

    def get_queue_stats(self) -> dict[str, int]:
        """Get statistics about queued jobs by priority."""
        stats = {
            "total": len(self._queue),
            "running": len(self._running),
            "critical": 0,
            "high": 0,
            "normal": 0,
            "low": 0,
        }
        for job in self._queue:
            if job.priority == JobPriority.CRITICAL:
                stats["critical"] += 1
            elif job.priority == JobPriority.HIGH:
                stats["high"] += 1
            elif job.priority == JobPriority.NORMAL:
                stats["normal"] += 1
            else:
                stats["low"] += 1
        return stats

    def get_allocation_stats(self) -> dict[str, Any]:
        """Get allocation statistics for fair quota monitoring (P2.1).

        Returns:
            Dict with allocation per config and quota status
        """
        self._reset_allocation_window_if_needed()
        total = self._get_total_allocation()

        stats: dict[str, Any] = {
            "window_start": self._allocation_window_start,
            "window_seconds": self._allocation_window_seconds,
            "total_allocation_seconds": total,
            "configs": {},
            "starvation_boosted_count": len(self._starvation_boosted),
        }

        for config_key, seconds in self._config_allocation.items():
            ratio = seconds / total if total > 0 else 0.0
            quota = CONFIG_QUOTAS.get(config_key, DEFAULT_CONFIG_QUOTA)
            stats["configs"][config_key] = {
                "seconds": seconds,
                "ratio": ratio,
                "quota": quota,
                "over_quota": ratio > quota,
            }

        return stats

    def has_critical_pending(self) -> bool:
        """Check if any critical priority jobs are pending."""
        return any(j.priority == JobPriority.CRITICAL for j in self._queue)

    def has_pending(self, priority: JobPriority | None = None) -> bool:
        """Check if any jobs are pending.

        Args:
            priority: If specified, check only for jobs of this priority

        Returns:
            True if matching jobs are pending
        """
        if priority is None:
            return len(self._queue) > 0
        return any(j.priority == priority for j in self._queue)

    def get_running_jobs(self) -> dict[str, ScheduledJob]:
        """Get all currently running jobs by host name."""
        return dict(self._running)

    def get_pending_jobs(self, priority: JobPriority | None = None) -> list[ScheduledJob]:
        """Get all pending jobs, optionally filtered by priority."""
        if priority is None:
            return list(self._queue)
        return [j for j in self._queue if j.priority == priority]

    def find_preemptable_jobs(
        self,
        for_priority: JobPriority = JobPriority.CRITICAL,
    ) -> list[tuple[str, ScheduledJob]]:
        """Find running jobs that can be preempted for higher priority work.

        December 2025: Enable critical jobs to preempt lower priority work.

        Preemption rules:
        1. Only preempt jobs with priority > for_priority (higher value = lower priority)
        2. Job must have preemptable=True
        3. Job must have run for at least preemption_min_runtime_seconds
        4. Prefer preempting lowest priority jobs first

        Args:
            for_priority: Priority level that needs resources

        Returns:
            List of (host_name, job) tuples that can be preempted, sorted by
            priority descending (lowest priority first - best preemption targets)
        """
        now = time.time()
        preemptable = []

        for host_name, job in self._running.items():
            # Only preempt lower priority jobs
            if job.priority <= for_priority:
                continue

            # Check if job is preemptable
            if not job.preemptable:
                continue

            # Check minimum runtime
            if job.started_at:
                runtime = now - job.started_at
                if runtime < job.preemption_min_runtime_seconds:
                    continue

            preemptable.append((host_name, job))

        # Sort by priority descending (lowest priority = best preemption target)
        preemptable.sort(key=lambda x: x[1].priority, reverse=True)
        return preemptable

    async def preempt_for_critical_job(
        self,
        host_name: str,
        preempting_job: ScheduledJob,
    ) -> ScheduledJob | None:
        """Preempt a running job on a host to make room for a critical job.

        December 2025: Enables CRITICAL priority jobs to preempt lower priority work.

        Args:
            host_name: Host where job is running
            preempting_job: The higher-priority job that needs the resources

        Returns:
            The preempted job (requeued with same priority), or None if no job to preempt
        """
        if host_name not in self._running:
            return None

        preempted_job = self._running.pop(host_name)
        preempted_job.running_on_host = None

        # Re-queue the preempted job (keep same priority - it will run when resources free)
        requeued_job = ScheduledJob(
            job_type=preempted_job.job_type,
            priority=preempted_job.priority,
            config=preempted_job.config,
            host_preference=None,  # Clear host preference - needs new host
            requires_gpu=preempted_job.requires_gpu,
            estimated_duration_seconds=preempted_job.estimated_duration_seconds,
            job_id=f"{preempted_job.job_id or 'preempted'}_requeue",
            preemptable=preempted_job.preemptable,
            preemption_min_runtime_seconds=preempted_job.preemption_min_runtime_seconds,
        )
        self.schedule(requeued_job)

        # Emit JOB_PREEMPTED event
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus

            await get_event_bus().publish(DataEvent(
                event_type=DataEventType.JOB_PREEMPTED,
                payload={
                    "host": host_name,
                    "preempted_job_type": preempted_job.job_type,
                    "preempted_job_id": preempted_job.job_id,
                    "preempted_priority": preempted_job.priority.name,
                    "preempting_job_type": preempting_job.job_type,
                    "preempting_job_id": preempting_job.job_id,
                    "preempting_priority": preempting_job.priority.name,
                    "runtime_seconds": time.time() - (preempted_job.started_at or time.time()),
                },
                source="JobScheduler",
            ))
        except ImportError:
            pass  # Event system not available

        logger.info(
            f"[JobScheduler] Preempted {preempted_job.job_type} ({preempted_job.priority.name}) "
            f"on {host_name} for {preempting_job.job_type} ({preempting_job.priority.name})"
        )

        return preempted_job

    def reserve_capacity_for_training(
        self,
        hosts: list[Any],
        statuses: list[Any],
        reserve_percent: float = 20.0,
        *,
        host_has_gpu: Callable[[Any], bool] | None = None,
        host_get_name: Callable[[Any], str] | None = None,
        status_get_gpu: Callable[[Any], float] | None = None,
        status_is_reachable: Callable[[Any], bool] | None = None,
    ) -> list[str]:
        """Reserve GPU capacity for training on GPU hosts.

        Returns list of host names where capacity was reserved.
        """
        # Default accessor functions
        def _has_gpu(h: Any) -> bool:
            if host_has_gpu:
                return host_has_gpu(h)
            return getattr(h, "has_gpu", False)

        def _get_name(h: Any) -> str:
            if host_get_name:
                return host_get_name(h)
            return getattr(h, "name", str(h))

        def _get_gpu(s: Any) -> float:
            if status_get_gpu:
                return status_get_gpu(s)
            return getattr(s, "gpu_percent", 0.0)

        def _is_reachable(s: Any) -> bool:
            if status_is_reachable:
                return status_is_reachable(s)
            return getattr(s, "reachable", True)

        reserved = []
        for host, status in zip(hosts, statuses, strict=False):
            if not _has_gpu(host):
                continue
            if not _is_reachable(status):
                continue

            # Reserve by not scheduling selfplay jobs on this host
            # if GPU utilization is already in acceptable range for training
            if _get_gpu(status) < reserve_percent:
                reserved.append(_get_name(host))

        return reserved

    def clear_queue(self) -> int:
        """Clear all pending jobs from the queue.

        Returns:
            Number of jobs cleared
        """
        count = len(self._queue)
        self._queue.clear()
        return count

    def health_check(self) -> "HealthCheckResult":
        """Check health of the job scheduler.

        Returns:
            HealthCheckResult indicating scheduler health status.
        """
        # Import from contracts (zero-dependency module)
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        queue_stats = self.get_queue_stats()
        allocation_stats = self.get_allocation_stats()

        # Check for warning conditions
        warnings = []

        # Queue too deep?
        if queue_stats["total"] > self._max_queue_size * 0.9:
            warnings.append(f"Queue near capacity: {queue_stats['total']}/{self._max_queue_size}")

        # Starvation detected?
        if allocation_stats.get("starvation_boosted_count", 0) > 10:
            warnings.append(f"High starvation count: {allocation_stats['starvation_boosted_count']}")

        # Check for over-quota configs
        over_quota_configs = [
            cfg for cfg, data in allocation_stats.get("configs", {}).items()
            if data.get("over_quota", False)
        ]
        if len(over_quota_configs) > 3:
            warnings.append(f"Multiple configs over quota: {len(over_quota_configs)}")

        is_healthy = len(warnings) == 0
        status = CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message="; ".join(warnings) if warnings else "Scheduler healthy",
            details={
                "queue_total": queue_stats["total"],
                "running_jobs": queue_stats["running"],
                "critical_pending": queue_stats["critical"],
                "starvation_boosted": allocation_stats.get("starvation_boosted_count", 0),
                "over_quota_configs": len(over_quota_configs),
            },
        )


# Global scheduler instance
_scheduler: PriorityJobScheduler | None = None


def get_scheduler() -> PriorityJobScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PriorityJobScheduler()
    return _scheduler


# Alias for compatibility (Phase 24.2 - used by IdleResourceDaemon)
get_job_scheduler = get_scheduler


def reset_scheduler() -> None:
    """Reset the global scheduler instance."""
    global _scheduler
    _scheduler = None


# ============================================================================
# Elo-Driven Curriculum Learning
# ============================================================================


def get_config_game_counts(
    db_path: Path | None = None,
) -> dict[str, int]:
    """Get game counts per config from match history for curriculum prioritization.

    Args:
        db_path: Path to the Elo database

    Returns:
        Dict of "board_players" -> game_count
    """
    db_path = db_path or DEFAULT_ELO_DB_PATH
    if not db_path.exists():
        return {}

    try:
        # December 27, 2025: Use context manager to prevent connection leaks
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            cursor = conn.execute(
                """
                SELECT board_type, num_players, COUNT(*) as game_count
                FROM match_history
                GROUP BY board_type, num_players
            """
            )
            counts = {}
            for row in cursor.fetchall():
                key = f"{row[0]}_{row[1]}p"
                counts[key] = row[2]
            return counts
    except Exception as e:
        logger.warning(f"Failed to get game counts: {e}")
        return {}


def select_curriculum_config(
    configs: list[dict[str, Any]],
    game_counts: dict[str, int],
) -> dict[str, Any]:
    """Select next config based on curriculum learning (prioritize underserved).

    Configs with fewer games get higher priority to ensure balanced training.

    Args:
        configs: List of config dicts with 'board' and 'players' keys
        game_counts: Dict of "board_playerssp" -> game_count

    Returns:
        Selected config dict
    """
    if not configs:
        return {}

    if not game_counts:
        # No history - use round-robin
        return random.choice(configs)

    # Calculate priority score for each config (lower games = higher priority)
    scored_configs = []
    for cfg in configs:
        board = cfg.get("board", cfg.get("board_type", "square8"))
        players = cfg.get("players", cfg.get("num_players", 2))
        key = f"{board}_{players}p"
        count = game_counts.get(key, 0)
        # Inverse priority: fewer games = higher weight
        priority = 1.0 / (count + 1)
        scored_configs.append((cfg, priority))

    # Weighted random selection based on priority
    total_weight = sum(p for _, p in scored_configs)
    if total_weight <= 0:
        return random.choice(configs)

    r = random.random() * total_weight
    cumulative = 0.0
    for cfg, priority in scored_configs:
        cumulative += priority
        if r <= cumulative:
            return cfg

    return configs[-1]


def get_underserved_configs(
    configs: list[dict[str, Any]],
    game_counts: dict[str, int],
    threshold: int = ELO_UNDERSERVED_THRESHOLD,
) -> list[dict[str, Any]]:
    """Get configs that have fewer games than the threshold.

    Args:
        configs: List of config dicts
        game_counts: Dict of config keys to game counts
        threshold: Minimum games to not be considered underserved

    Returns:
        List of underserved config dicts
    """
    underserved = []
    for cfg in configs:
        board = cfg.get("board", cfg.get("board_type", "square8"))
        players = cfg.get("players", cfg.get("num_players", 2))
        key = f"{board}_{players}p"
        count = game_counts.get(key, 0)
        if count < threshold:
            underserved.append(cfg)
    return underserved


# ============================================================================
# Host Selection Helpers
# ============================================================================


def get_cpu_rich_hosts(
    hosts: list[Any],
    statuses: dict[str, Any],
    *,
    host_enabled: Callable[[Any], bool] | None = None,
    host_get_name: Callable[[Any], str] | None = None,
    host_get_cpus: Callable[[Any], int] | None = None,
    host_get_role: Callable[[Any], str] | None = None,
    host_has_expensive_gpu: Callable[[Any], bool] | None = None,
    status_get_cpu: Callable[[Any], float] | None = None,
    status_is_reachable: Callable[[Any], bool] | None = None,
) -> list[tuple[Any, Any]]:
    """Get CPU-rich hosts suitable for tournament workloads.

    Prioritizes hosts with:
    - High CPU count
    - Low current CPU utilization (< 60%)
    - Not GPU-constrained (or no expensive GPU)

    When selecting hosts for CPU-bound tasks, hosts with expensive GPUs
    (H100, GH200, A100, etc.) are deprioritized to avoid wasting GPU resources.
    Hosts with role="cpu_selfplay" are boosted in priority.

    Args:
        hosts: List of host configs
        statuses: Dict of host_name -> status
        host_enabled: Function to check if host is enabled
        host_get_name: Function to get host name
        host_get_cpus: Function to get CPU count
        host_get_role: Function to get host role
        host_has_expensive_gpu: Function to check if host has expensive GPU
        status_get_cpu: Function to get CPU usage
        status_is_reachable: Function to check reachability

    Returns:
        List of (host, status) tuples sorted by CPU priority score (descending)
    """
    # Default accessor functions
    def _enabled(h: Any) -> bool:
        if host_enabled:
            return host_enabled(h)
        return getattr(h, "enabled", True)

    def _get_name(h: Any) -> str:
        if host_get_name:
            return host_get_name(h)
        return getattr(h, "name", str(h))

    def _get_cpus(h: Any) -> int:
        if host_get_cpus:
            return host_get_cpus(h)
        # Check both direct attribute and properties dict
        cpus = getattr(h, "cpus", None)
        if cpus is None:
            props = getattr(h, "properties", {})
            cpus = props.get("cpus", 0) if props else 0
        return cpus or 0

    def _get_role(h: Any) -> str:
        if host_get_role:
            return host_get_role(h)
        props = getattr(h, "properties", {})
        return props.get("role", "") if props else ""

    def _has_expensive_gpu(h: Any) -> bool:
        if host_has_expensive_gpu:
            return host_has_expensive_gpu(h)
        # Check GPU field for expensive models
        props = getattr(h, "properties", {})
        gpu = props.get("gpu", "") if props else ""
        expensive_patterns = ["H100", "GH200", "A100", "A10", "5090", "4090"]
        return any(p in gpu for p in expensive_patterns)

    def _get_cpu(s: Any) -> float:
        if status_get_cpu:
            return status_get_cpu(s)
        return getattr(s, "cpu_percent", 0.0)

    def _is_reachable(s: Any) -> bool:
        if status_is_reachable:
            return status_is_reachable(s)
        return getattr(s, "reachable", True)

    def _cpu_priority_score(h: Any) -> float:
        """Calculate priority score for CPU task assignment.

        Higher score = better for CPU tasks.
        - Base: CPU count
        - Boost: +1000 for cpu_selfplay role (prioritize dedicated CPU hosts)
        - Penalty: -500 for expensive GPUs (save for GPU tasks)
        """
        score = float(_get_cpus(h))
        role = _get_role(h)
        if role == "cpu_selfplay":
            score += 1000  # Strongly prefer dedicated CPU hosts
        if _has_expensive_gpu(h):
            score -= 500  # Deprioritize expensive GPU hosts for CPU tasks
        return score

    cpu_hosts = []
    for host in hosts:
        if not _enabled(host):
            continue
        name = _get_name(host)
        status = statuses.get(name)
        if not status or not _is_reachable(status):
            continue
        # Prefer hosts with available CPU capacity
        if _get_cpu(status) < TARGET_CPU_UTILIZATION_MIN:
            cpu_hosts.append((host, status))

    # Sort by CPU priority score (descending) - prioritize CPU-focused hosts
    cpu_hosts.sort(key=lambda x: _cpu_priority_score(x[0]), reverse=True)
    return cpu_hosts


def get_gpu_rich_hosts(
    hosts: list[Any],
    statuses: dict[str, Any],
    *,
    host_enabled: Callable[[Any], bool] | None = None,
    host_has_gpu: Callable[[Any], bool] | None = None,
    host_get_name: Callable[[Any], str] | None = None,
    host_get_memory_gb: Callable[[Any], int] | None = None,
    status_get_gpu: Callable[[Any], float] | None = None,
    status_is_reachable: Callable[[Any], bool] | None = None,
) -> list[tuple[Any, Any]]:
    """Get GPU-rich hosts suitable for GPU selfplay and training.

    Prioritizes hosts with:
    - GPU available
    - Low GPU utilization (< 60%)

    Args:
        hosts: List of host configs
        statuses: Dict of host_name -> status
        host_enabled: Function to check if host is enabled
        host_has_gpu: Function to check if host has GPU
        host_get_name: Function to get host name
        host_get_memory_gb: Function to get memory in GB
        status_get_gpu: Function to get GPU usage
        status_is_reachable: Function to check reachability

    Returns:
        List of (host, status) tuples sorted by memory descending
    """
    # Default accessor functions
    def _enabled(h: Any) -> bool:
        if host_enabled:
            return host_enabled(h)
        return getattr(h, "enabled", True)

    def _has_gpu(h: Any) -> bool:
        if host_has_gpu:
            return host_has_gpu(h)
        return getattr(h, "has_gpu", False)

    def _get_name(h: Any) -> str:
        if host_get_name:
            return host_get_name(h)
        return getattr(h, "name", str(h))

    def _get_memory_gb(h: Any) -> int:
        if host_get_memory_gb:
            return host_get_memory_gb(h)
        return getattr(h, "memory_gb", 0)

    def _get_gpu(s: Any) -> float:
        if status_get_gpu:
            return status_get_gpu(s)
        return getattr(s, "gpu_percent", 0.0)

    def _is_reachable(s: Any) -> bool:
        if status_is_reachable:
            return status_is_reachable(s)
        return getattr(s, "reachable", True)

    gpu_hosts = []
    for host in hosts:
        if not _enabled(host) or not _has_gpu(host):
            continue
        name = _get_name(host)
        status = statuses.get(name)
        if not status or not _is_reachable(status):
            continue
        # Prefer hosts with available GPU capacity
        if _get_gpu(status) < TARGET_GPU_UTILIZATION_MIN:
            gpu_hosts.append((host, status))

    # Sort by GPU memory descending (bigger GPUs first)
    gpu_hosts.sort(key=lambda x: _get_memory_gb(x[0]), reverse=True)
    return gpu_hosts


# ============================================================================
# Host Dead → Job Migration Wiring (December 2025)
# ============================================================================


class HostDeadJobMigrator:
    """Watches for HOST_OFFLINE events and migrates jobs from dead hosts.

    When a host goes offline, any jobs running on that host are:
    1. Cancelled from the running jobs set
    2. Re-queued with elevated priority for execution on another host

    Usage:
        from app.coordination.job_scheduler import wire_host_dead_to_job_migration

        migrator = wire_host_dead_to_job_migration()
        # Now HOST_OFFLINE events will automatically trigger job migration

    Integration:
        - Subscribes to DataEventType.HOST_OFFLINE from data_events
        - Uses PriorityJobScheduler to cancel and re-queue jobs
    """

    def __init__(
        self,
        scheduler: PriorityJobScheduler | None = None,
        requeue_priority_boost: int = 1,  # Boost priority by 1 level on requeue
    ):
        """Initialize job migrator.

        Args:
            scheduler: PriorityJobScheduler to use (default: global scheduler)
            requeue_priority_boost: How many priority levels to boost requeued jobs
                                   (e.g., 1 means NORMAL -> HIGH)
        """
        self._scheduler = scheduler
        self._requeue_priority_boost = requeue_priority_boost
        self._migrations_count = 0
        self._failed_migrations = 0
        self._subscribed = False

    @property
    def scheduler(self) -> PriorityJobScheduler:
        """Get scheduler, using global singleton if not provided."""
        if self._scheduler is None:
            self._scheduler = get_scheduler()
        return self._scheduler

    def subscribe_to_host_events(self) -> bool:
        """Subscribe to HOST_OFFLINE events from the event bus.

        Returns:
            True if subscription was successful
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
            self._subscribed = True
            logger.info("[HostDeadJobMigrator] Subscribed to HOST_OFFLINE events")
            return True

        except ImportError as e:
            logger.warning(f"[HostDeadJobMigrator] Could not subscribe to events: {e}")
            return False

    def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE event by migrating jobs.

        Args:
            event: DataEvent with HOST_OFFLINE type
        """
        host = event.payload.get("host", "")
        reason = event.payload.get("reason", "unknown")

        if not host:
            logger.warning("[HostDeadJobMigrator] HOST_OFFLINE event without host name")
            return

        logger.info(f"[HostDeadJobMigrator] Host offline: {host} (reason: {reason})")

        # Migrate jobs from this host
        migrated = self.migrate_jobs_from_host(host, reason)

        if migrated > 0:
            logger.info(
                f"[HostDeadJobMigrator] Migrated {migrated} jobs from dead host {host}"
            )

    def migrate_jobs_from_host(self, host_name: str, reason: str = "") -> int:
        """Migrate all running jobs from a host.

        Args:
            host_name: Name of the dead/offline host
            reason: Reason for migration (for logging)

        Returns:
            Number of jobs migrated
        """
        running_jobs = self.scheduler.get_running_jobs()

        if host_name not in running_jobs:
            logger.debug(f"[HostDeadJobMigrator] No running jobs on host {host_name}")
            return 0

        migrated = 0

        try:
            # Cancel the job from running set
            cancelled_job = self.scheduler.cancel_job(host_name)

            if cancelled_job:
                # Boost priority for faster requeue
                new_priority = self._boost_priority(cancelled_job.priority)

                # Create re-queued job
                requeued_job = ScheduledJob(
                    job_type=cancelled_job.job_type,
                    priority=new_priority,
                    config=cancelled_job.config,
                    host_preference=None,  # Clear host preference - needs new host
                    requires_gpu=cancelled_job.requires_gpu,
                    estimated_duration_seconds=cancelled_job.estimated_duration_seconds,
                    job_id=f"{cancelled_job.job_id or 'migrated'}_requeue",
                )

                # Schedule the requeued job
                if self.scheduler.schedule(requeued_job):
                    migrated = 1
                    self._migrations_count += 1
                    logger.info(
                        f"[HostDeadJobMigrator] Requeued job {cancelled_job.job_type} "
                        f"from {host_name} with priority {new_priority.name}"
                    )
                else:
                    self._failed_migrations += 1
                    logger.error(
                        f"[HostDeadJobMigrator] Failed to requeue job from {host_name}"
                    )

        except Exception as e:
            self._failed_migrations += 1
            logger.error(f"[HostDeadJobMigrator] Error migrating job from {host_name}: {e}")

        return migrated

    def _boost_priority(self, current: JobPriority) -> JobPriority:
        """Boost job priority by configured amount.

        Args:
            current: Current job priority

        Returns:
            Boosted priority (capped at CRITICAL)
        """
        # Lower value = higher priority
        new_value = max(0, current.value - self._requeue_priority_boost)
        return JobPriority(new_value)

    def get_stats(self) -> dict[str, Any]:
        """Get migration statistics.

        Returns:
            Dict with migration stats
        """
        return {
            "subscribed": self._subscribed,
            "migrations_count": self._migrations_count,
            "failed_migrations": self._failed_migrations,
            "requeue_priority_boost": self._requeue_priority_boost,
        }


# Singleton migrator instance
_job_migrator: HostDeadJobMigrator | None = None


def wire_host_dead_to_job_migration(
    scheduler: PriorityJobScheduler | None = None,
    requeue_priority_boost: int = 1,
) -> HostDeadJobMigrator:
    """Wire HOST_OFFLINE events to automatic job migration.

    This enables automatic job migration when hosts go offline:
    - Jobs running on offline hosts are cancelled
    - Jobs are requeued with boosted priority

    Args:
        scheduler: Scheduler to use (default: global singleton)
        requeue_priority_boost: Priority boost for requeued jobs

    Returns:
        HostDeadJobMigrator instance
    """
    global _job_migrator

    if _job_migrator is None:
        _job_migrator = HostDeadJobMigrator(
            scheduler=scheduler,
            requeue_priority_boost=requeue_priority_boost,
        )
        _job_migrator.subscribe_to_host_events()

    return _job_migrator


def get_job_migrator() -> HostDeadJobMigrator | None:
    """Get the job migrator instance if wired."""
    return _job_migrator


def reset_job_migrator() -> None:
    """Reset the job migrator singleton (for testing)."""
    global _job_migrator
    _job_migrator = None


# ============================================================================
# NODE_OVERLOADED → Job Redistribution Handler (Phase 3 December 2025)
# ============================================================================


class NodeOverloadedHandler:
    """Handles NODE_OVERLOADED events by redistributing jobs from overloaded nodes.

    When a node reports critical CPU/GPU utilization:
    1. Cancel lowest-priority jobs on that node
    2. Re-queue cancelled jobs for assignment to less loaded nodes
    3. Emit metrics for monitoring overload frequency

    This closes the feedback loop: node overload → automatic job redistribution.
    """

    def __init__(
        self,
        scheduler: PriorityJobScheduler | None = None,
        max_jobs_to_migrate: int = 2,  # Max jobs to migrate per overload event
    ):
        """Initialize the handler.

        Args:
            scheduler: PriorityJobScheduler to use (default: global scheduler)
            max_jobs_to_migrate: Maximum jobs to migrate per overload event
        """
        self._scheduler = scheduler
        self.max_jobs_to_migrate = max_jobs_to_migrate

        self._subscribed = False
        self._overload_count = 0
        self._jobs_migrated = 0

    @property
    def scheduler(self) -> PriorityJobScheduler:
        """Get scheduler, using global singleton if not provided."""
        if self._scheduler is None:
            self._scheduler = get_scheduler()
        return self._scheduler

    def subscribe(self) -> bool:
        """Subscribe to NODE_OVERLOADED events.

        Returns:
            True if subscription was successful
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()

            def on_node_overloaded(event):
                """Handle NODE_OVERLOADED event."""
                self._handle_overload(event.payload)

            bus.subscribe(DataEventType.NODE_OVERLOADED, on_node_overloaded)
            self._subscribed = True
            logger.info("[NodeOverloadedHandler] Subscribed to NODE_OVERLOADED events")
            return True

        except ImportError as e:
            logger.debug(f"[NodeOverloadedHandler] Event system not available: {e}")
            return False

    def _handle_overload(self, payload: dict[str, Any]) -> None:
        """Handle a node overload event.

        Args:
            payload: Event payload with overload details
        """
        host = payload.get("host", "")
        cpu_percent = payload.get("cpu_percent", 0.0)
        gpu_percent = payload.get("gpu_percent", 0.0)
        resource_type = payload.get("resource_type", "cpu")

        if not host:
            logger.warning("[NodeOverloadedHandler] NODE_OVERLOADED event without host")
            return

        self._overload_count += 1

        logger.warning(
            f"[NodeOverloadedHandler] Overload #{self._overload_count} on {host}: "
            f"CPU={cpu_percent:.1f}%, GPU={gpu_percent:.1f}% ({resource_type})"
        )

        # Check if we have running jobs on this host
        running_jobs = self.scheduler.get_running_jobs()
        if host not in running_jobs:
            logger.debug(
                f"[NodeOverloadedHandler] No running jobs on overloaded host {host}"
            )
            return

        # Migrate jobs from overloaded host
        migrated = 0
        job = running_jobs.get(host)

        if job:
            # Cancel and re-queue with lower priority to spread load
            cancelled = self.scheduler.cancel_job(host)
            if cancelled:
                # Re-queue with same priority (let scheduler find better host)
                requeued = ScheduledJob(
                    job_type=cancelled.job_type,
                    priority=cancelled.priority,
                    config=cancelled.config,
                    host_preference=None,  # Clear host preference
                    requires_gpu=cancelled.requires_gpu,
                    estimated_duration_seconds=cancelled.estimated_duration_seconds,
                    job_id=f"{cancelled.job_id or 'migrated'}_overload",
                )

                if self.scheduler.schedule(requeued):
                    migrated += 1
                    self._jobs_migrated += 1
                    logger.info(
                        f"[NodeOverloadedHandler] Migrated {cancelled.job_type} "
                        f"from overloaded host {host}"
                    )

        if migrated > 0:
            logger.info(
                f"[NodeOverloadedHandler] Migrated {migrated} job(s) from {host}"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "subscribed": self._subscribed,
            "overload_count": self._overload_count,
            "jobs_migrated": self._jobs_migrated,
            "max_jobs_to_migrate": self.max_jobs_to_migrate,
        }

    def health_check(self) -> "HealthCheckResult":
        """Check overload handler health for daemon monitoring.

        December 2025: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with health status.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            stats = self.get_stats()
            subscribed = stats.get("subscribed", False)
            overload_count = stats.get("overload_count", 0)
            jobs_migrated = stats.get("jobs_migrated", 0)

            # Not subscribed is degraded
            if not subscribed:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message="NodeOverloadedHandler not subscribed to events",
                    details=stats,
                )

            # Healthy
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"NodeOverloadedHandler healthy: {overload_count} overloads handled, {jobs_migrated} jobs migrated",
                details=stats,
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"NodeOverloadedHandler health check error: {e}",
            )


# Singleton overload handler
_overload_handler: NodeOverloadedHandler | None = None


def wire_node_overloaded_handler(
    scheduler: PriorityJobScheduler | None = None,
) -> NodeOverloadedHandler:
    """Wire NODE_OVERLOADED events to automatic job redistribution.

    Args:
        scheduler: Scheduler to use (default: global singleton)

    Returns:
        Subscribed handler instance
    """
    global _overload_handler

    if _overload_handler is None:
        _overload_handler = NodeOverloadedHandler(scheduler=scheduler)
        _overload_handler.subscribe()

    return _overload_handler


def get_overload_handler() -> NodeOverloadedHandler | None:
    """Get the overload handler if wired."""
    return _overload_handler


def reset_overload_handler() -> None:
    """Reset the overload handler singleton (for testing)."""
    global _overload_handler
    _overload_handler = None


__all__ = [
    # Configuration
    "ELO_CURRICULUM_ENABLED",
    "ELO_UNDERSERVED_THRESHOLD",
    "MIN_MEMORY_GB_FOR_TASKS",
    "TARGET_CPU_UTILIZATION_MAX",
    "TARGET_CPU_UTILIZATION_MIN",
    "TARGET_GPU_UTILIZATION_MAX",
    "TARGET_GPU_UTILIZATION_MIN",
    # P2.1 (Dec 2025): Fair allocation quotas
    "CONFIG_QUOTAS",
    "DEFAULT_CONFIG_QUOTA",
    # P2.2 (Dec 2025): Starvation prevention
    "STARVATION_THRESHOLD_HOURS",
    # P2.3 (Dec 2025): Cost-aware scheduling
    "PROVIDER_COSTS",
    # P2.4 (Dec 2025): Ephemeral node optimization
    "EPHEMERAL_BLOCKED_JOB_TYPES",
    "EPHEMERAL_PROVIDERS",
    "MAX_EPHEMERAL_JOB_DURATION_SECONDS",
    # Job migration (December 2025)
    "HostDeadJobMigrator",
    "JobPriority",
    # Priority scheduler
    "PriorityJobScheduler",
    "ScheduledJob",
    # Curriculum learning
    "get_config_game_counts",
    # Host selection
    "get_cpu_rich_hosts",
    "get_gpu_rich_hosts",
    "get_job_migrator",
    "get_job_scheduler",  # Alias for get_scheduler (Phase 24.2)
    "get_scheduler",
    "get_underserved_configs",
    "reset_job_migrator",
    "reset_scheduler",
    "select_curriculum_config",
    "wire_host_dead_to_job_migration",
    # Node overload handling (Phase 3 December 2025)
    "NodeOverloadedHandler",
    "get_overload_handler",
    "reset_overload_handler",
    "wire_node_overloaded_handler",
]
