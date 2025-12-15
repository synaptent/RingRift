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
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Utilization targets for job scheduling
TARGET_GPU_UTILIZATION_MIN = 60  # Start more jobs if GPU below this %
TARGET_GPU_UTILIZATION_MAX = 90  # Don't start more jobs if GPU above this %
TARGET_CPU_UTILIZATION_MIN = 60  # Start more jobs if CPU below this %
TARGET_CPU_UTILIZATION_MAX = 85  # Don't start more jobs if CPU above this %
MIN_MEMORY_GB_FOR_TASKS = 64  # Skip nodes with less than this to avoid OOM

# Elo curriculum configuration
ELO_CURRICULUM_ENABLED = True
ELO_UNDERSERVED_THRESHOLD = 100  # Configs with fewer games are "underserved"

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


@dataclass
class ScheduledJob:
    """A job to be scheduled on the cluster."""

    job_type: str  # selfplay, tournament, training, promotion, etc.
    priority: JobPriority
    config: Dict[str, Any] = field(default_factory=dict)
    host_preference: Optional[str] = None  # Preferred host name or None
    requires_gpu: bool = False
    estimated_duration_seconds: int = 3600
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    job_id: Optional[str] = None

    def __lt__(self, other: "ScheduledJob") -> bool:
        """Enable sorting by priority."""
        return self.priority < other.priority

    def __hash__(self) -> int:
        """Hash by job_id or id if no job_id."""
        return hash(self.job_id or id(self))

    def to_dict(self) -> Dict[str, Any]:
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
        self._queue: List[ScheduledJob] = []
        self._running: Dict[str, ScheduledJob] = {}  # host_name -> job
        self._max_queue_size = max_queue_size
        self._completed_jobs: List[ScheduledJob] = []
        self._max_completed_history = 100

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

    def next_job(
        self,
        hosts: List[Any],
        statuses: List[Any],
        *,
        host_has_gpu: Optional[Callable[[Any], bool]] = None,
        host_get_name: Optional[Callable[[Any], str]] = None,
        host_get_memory_gb: Optional[Callable[[Any], int]] = None,
        status_get_cpu: Optional[Callable[[Any], float]] = None,
        status_get_disk: Optional[Callable[[Any], float]] = None,
        status_get_memory: Optional[Callable[[Any], float]] = None,
        status_is_reachable: Optional[Callable[[Any], bool]] = None,
    ) -> Optional[Tuple[ScheduledJob, Any]]:
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
        available_hosts: List[Tuple[Any, Any]] = []
        for host, status in zip(hosts, statuses):
            if not _is_reachable(status):
                continue
            if _get_disk(status) > 90:
                continue
            if _get_mem(status) > 90:
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
            for host, status in available_hosts:
                # Check GPU requirement
                if job.requires_gpu and not _has_gpu(host):
                    continue

                # Check host preference
                if job.host_preference and _get_name(host) != job.host_preference:
                    continue

                # Check CPU capacity for selfplay jobs
                if job.job_type == "selfplay" and _get_cpu(status) > TARGET_CPU_UTILIZATION_MAX:
                    continue

                # Found a match
                self._queue.pop(job_idx)
                job.started_at = time.time()
                self._running[_get_name(host)] = job
                return (job, host)

        return None

    def complete_job(self, host_name: str, success: bool = True) -> Optional[ScheduledJob]:
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

            # Keep history of completed jobs
            self._completed_jobs.append(job)
            if len(self._completed_jobs) > self._max_completed_history:
                self._completed_jobs.pop(0)

            return job
        return None

    def cancel_job(self, host_name: str) -> Optional[ScheduledJob]:
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

    def get_queue_stats(self) -> Dict[str, int]:
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

    def has_critical_pending(self) -> bool:
        """Check if any critical priority jobs are pending."""
        return any(j.priority == JobPriority.CRITICAL for j in self._queue)

    def has_pending(self, priority: Optional[JobPriority] = None) -> bool:
        """Check if any jobs are pending.

        Args:
            priority: If specified, check only for jobs of this priority

        Returns:
            True if matching jobs are pending
        """
        if priority is None:
            return len(self._queue) > 0
        return any(j.priority == priority for j in self._queue)

    def get_running_jobs(self) -> Dict[str, ScheduledJob]:
        """Get all currently running jobs by host name."""
        return dict(self._running)

    def get_pending_jobs(self, priority: Optional[JobPriority] = None) -> List[ScheduledJob]:
        """Get all pending jobs, optionally filtered by priority."""
        if priority is None:
            return list(self._queue)
        return [j for j in self._queue if j.priority == priority]

    def reserve_capacity_for_training(
        self,
        hosts: List[Any],
        statuses: List[Any],
        reserve_percent: float = 20.0,
        *,
        host_has_gpu: Optional[Callable[[Any], bool]] = None,
        host_get_name: Optional[Callable[[Any], str]] = None,
        status_get_gpu: Optional[Callable[[Any], float]] = None,
        status_is_reachable: Optional[Callable[[Any], bool]] = None,
    ) -> List[str]:
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
        for host, status in zip(hosts, statuses):
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


# Global scheduler instance
_scheduler: Optional[PriorityJobScheduler] = None


def get_scheduler() -> PriorityJobScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PriorityJobScheduler()
    return _scheduler


def reset_scheduler() -> None:
    """Reset the global scheduler instance."""
    global _scheduler
    _scheduler = None


# ============================================================================
# Elo-Driven Curriculum Learning
# ============================================================================


def get_config_game_counts(
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
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
        conn = sqlite3.connect(str(db_path), timeout=5.0)
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
        conn.close()
        return counts
    except Exception as e:
        logger.warning(f"Failed to get game counts: {e}")
        return {}


def select_curriculum_config(
    configs: List[Dict[str, Any]],
    game_counts: Dict[str, int],
) -> Dict[str, Any]:
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
    configs: List[Dict[str, Any]],
    game_counts: Dict[str, int],
    threshold: int = ELO_UNDERSERVED_THRESHOLD,
) -> List[Dict[str, Any]]:
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
    hosts: List[Any],
    statuses: Dict[str, Any],
    *,
    host_enabled: Optional[Callable[[Any], bool]] = None,
    host_get_name: Optional[Callable[[Any], str]] = None,
    host_get_cpus: Optional[Callable[[Any], int]] = None,
    status_get_cpu: Optional[Callable[[Any], float]] = None,
    status_is_reachable: Optional[Callable[[Any], bool]] = None,
) -> List[Tuple[Any, Any]]:
    """Get CPU-rich hosts suitable for tournament workloads.

    Prioritizes hosts with:
    - High CPU count
    - Low current CPU utilization (< 60%)
    - Not GPU-constrained (or no GPU)

    Args:
        hosts: List of host configs
        statuses: Dict of host_name -> status
        host_enabled: Function to check if host is enabled
        host_get_name: Function to get host name
        host_get_cpus: Function to get CPU count
        status_get_cpu: Function to get CPU usage
        status_is_reachable: Function to check reachability

    Returns:
        List of (host, status) tuples sorted by CPU count descending
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
        return getattr(h, "cpus", 0)

    def _get_cpu(s: Any) -> float:
        if status_get_cpu:
            return status_get_cpu(s)
        return getattr(s, "cpu_percent", 0.0)

    def _is_reachable(s: Any) -> bool:
        if status_is_reachable:
            return status_is_reachable(s)
        return getattr(s, "reachable", True)

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

    # Sort by CPU count (descending) - bigger hosts first
    cpu_hosts.sort(key=lambda x: _get_cpus(x[0]), reverse=True)
    return cpu_hosts


def get_gpu_rich_hosts(
    hosts: List[Any],
    statuses: Dict[str, Any],
    *,
    host_enabled: Optional[Callable[[Any], bool]] = None,
    host_has_gpu: Optional[Callable[[Any], bool]] = None,
    host_get_name: Optional[Callable[[Any], str]] = None,
    host_get_memory_gb: Optional[Callable[[Any], int]] = None,
    status_get_gpu: Optional[Callable[[Any], float]] = None,
    status_is_reachable: Optional[Callable[[Any], bool]] = None,
) -> List[Tuple[Any, Any]]:
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


__all__ = [
    # Priority scheduler
    "PriorityJobScheduler",
    "JobPriority",
    "ScheduledJob",
    "get_scheduler",
    "reset_scheduler",
    # Curriculum learning
    "get_config_game_counts",
    "select_curriculum_config",
    "get_underserved_configs",
    # Host selection
    "get_cpu_rich_hosts",
    "get_gpu_rich_hosts",
    # Configuration
    "TARGET_GPU_UTILIZATION_MIN",
    "TARGET_GPU_UTILIZATION_MAX",
    "TARGET_CPU_UTILIZATION_MIN",
    "TARGET_CPU_UTILIZATION_MAX",
    "MIN_MEMORY_GB_FOR_TASKS",
    "ELO_CURRICULUM_ENABLED",
    "ELO_UNDERSERVED_THRESHOLD",
]
