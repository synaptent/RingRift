"""Work Queue Monitor Daemon - Tracks work queue lifecycle events.

Subscribes to all WORK_* events and provides:
1. Queue depth tracking
2. Job latency monitoring
3. Stuck job detection (claimed but not started)
4. Node workload distribution
5. Backpressure signaling

Usage:
    from app.coordination.work_queue_monitor_daemon import (
        WorkQueueMonitorDaemon,
        get_work_queue_monitor,
    )

    monitor = WorkQueueMonitorDaemon.get_instance()
    await monitor.start()

    stats = monitor.get_queue_stats()
    print(f"Queue depth: {stats.pending_count}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config.coordination_defaults import WorkQueueMonitorDefaults
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)

BACKPRESSURE_THRESHOLD = WorkQueueMonitorDefaults.BACKPRESSURE_THRESHOLD
STUCK_JOB_THRESHOLD_SECONDS = WorkQueueMonitorDefaults.STUCK_JOB_THRESHOLD
NODE_OVERLOAD_THRESHOLD = WorkQueueMonitorDefaults.NODE_OVERLOAD_THRESHOLD
LATENCY_WINDOW_SIZE = WorkQueueMonitorDefaults.LATENCY_WINDOW_SIZE


@dataclass
class JobTracker:
    """Tracks a single job's lifecycle."""

    work_id: str
    work_type: str
    priority: int = 50
    config_key: str = ""
    queued_at: float = 0.0
    claimed_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    claimed_by: str = ""
    status: str = "pending"  # pending, claimed, running, completed, failed
    retry_count: int = 0


@dataclass
class QueueStats:
    """Statistics about the work queue."""

    pending_count: int = 0
    claimed_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    retry_count: int = 0

    # Latency metrics
    avg_latency_seconds: float = 0.0
    p95_latency_seconds: float = 0.0
    max_latency_seconds: float = 0.0

    # Per-node distribution
    jobs_per_node: dict[str, int] = field(default_factory=dict)

    # Backpressure state
    backpressure_active: bool = False

    # Stuck jobs
    stuck_job_count: int = 0


class WorkQueueMonitorDaemon(HandlerBase):
    """Daemon that monitors work queue events and provides visibility.

    Subscribes to all WORK_* events and tracks:
    - Queue depth and composition
    - Job latency (time from queued to completed)
    - Stuck jobs (claimed but not progressing)
    - Node workload distribution
    - Backpressure conditions
    """

    _event_source = "WorkQueueMonitor"

    def __init__(self) -> None:
        """Initialize the work queue monitor."""
        super().__init__(
            name="work_queue_monitor",
            cycle_interval=60.0,
        )

        # Thresholds
        self._backpressure_threshold = BACKPRESSURE_THRESHOLD
        self._stuck_job_threshold_seconds = STUCK_JOB_THRESHOLD_SECONDS
        self._node_overload_threshold = NODE_OVERLOAD_THRESHOLD
        self._latency_window_size = LATENCY_WINDOW_SIZE
        self._backpressure_check_interval = 10.0

        # Job tracking
        self._jobs: dict[str, JobTracker] = {}
        self._completed_latencies: list[float] = []  # Rolling window

        # Counters
        self._total_queued = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_retries = 0

        # Per-node tracking
        self._node_job_counts: dict[str, int] = defaultdict(int)

        # Backpressure state
        self._backpressure_active = False
        self._last_backpressure_check = 0.0

        # Lock for thread safety on job tracking
        self._jobs_lock = asyncio.Lock()

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event handlers for work queue events."""
        subscriptions = {
            "WORK_QUEUED": self._on_work_queued,
            "WORK_CLAIMED": self._on_work_claimed,
            "WORK_STARTED": self._on_work_started,
            "WORK_COMPLETED": self._on_work_completed,
            "WORK_FAILED": self._on_work_failed,
        }

        # WORK_RETRY and WORK_CANCELLED may not exist in all versions
        try:
            from app.distributed.data_events import DataEventType
            if hasattr(DataEventType, "WORK_RETRY"):
                subscriptions["WORK_RETRY"] = self._on_work_retry
            if hasattr(DataEventType, "WORK_CANCELLED"):
                subscriptions["WORK_CANCELLED"] = self._on_work_cancelled
        except ImportError:
            pass

        return subscriptions

    async def _run_cycle(self) -> None:
        """Run one monitoring cycle - check for stuck jobs."""
        await self._check_stuck_jobs()

    async def _check_stuck_jobs(self) -> None:
        """Check for jobs that are stuck (claimed but not started)."""
        now = time.time()
        stuck_jobs: list[tuple[JobTracker, float]] = []

        async with self._jobs_lock:
            for job in self._jobs.values():
                # Job is stuck if claimed but not started for too long
                if job.status == "claimed" and job.claimed_at > 0 and job.started_at == 0:
                    stuck_duration = now - job.claimed_at
                    if stuck_duration > self._stuck_job_threshold_seconds:
                        stuck_jobs.append((job, stuck_duration))

        # Emit events for stuck jobs (outside lock)
        for job, duration in stuck_jobs:
            await self._emit_stuck_job_event(job, duration)
            logger.warning(
                f"[{self.name}] Stuck job detected: {job.work_id} "
                f"claimed by {job.claimed_by} for {duration:.0f}s"
            )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_work_queued(self, event: Any) -> None:
        """Handle WORK_QUEUED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")
        if not work_id:
            return

        # Check for duplicate using dedup
        if self._is_duplicate_event(payload, key_fields=["work_id", "work_type"]):
            return

        async with self._jobs_lock:
            self._jobs[work_id] = JobTracker(
                work_id=work_id,
                work_type=payload.get("work_type", "unknown"),
                priority=payload.get("priority", 50),
                config_key=payload.get("config_key", ""),
                queued_at=time.time(),
                status="pending",
            )
            self._total_queued += 1

        self._stats.events_processed += 1
        logger.debug(f"[{self.name}] Work queued: {work_id}")
        await self._check_backpressure()

    async def _on_work_claimed(self, event: Any) -> None:
        """Handle WORK_CLAIMED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")
        claimed_by = payload.get("claimed_by", "") or payload.get("node_id", "")

        async with self._jobs_lock:
            if work_id in self._jobs:
                self._jobs[work_id].claimed_at = time.time()
                self._jobs[work_id].claimed_by = claimed_by
                self._jobs[work_id].status = "claimed"

                # Track per-node load
                if claimed_by:
                    self._node_job_counts[claimed_by] += 1
                    await self._check_node_overload(claimed_by)

        self._stats.events_processed += 1

    async def _on_work_started(self, event: Any) -> None:
        """Handle WORK_STARTED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._jobs_lock:
            if work_id in self._jobs:
                self._jobs[work_id].started_at = time.time()
                self._jobs[work_id].status = "running"

        self._stats.events_processed += 1

    async def _on_work_completed(self, event: Any) -> None:
        """Handle WORK_COMPLETED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._jobs_lock:
            if work_id in self._jobs:
                job = self._jobs[work_id]
                job.completed_at = time.time()
                job.status = "completed"
                self._total_completed += 1

                # Calculate latency
                if job.queued_at > 0:
                    latency = job.completed_at - job.queued_at
                    self._completed_latencies.append(latency)
                    # Keep rolling window
                    if len(self._completed_latencies) > self._latency_window_size:
                        self._completed_latencies.pop(0)

                # Decrement node load
                if job.claimed_by:
                    self._node_job_counts[job.claimed_by] = max(
                        0, self._node_job_counts[job.claimed_by] - 1
                    )

                # Remove from active tracking
                del self._jobs[work_id]

        self._stats.events_processed += 1
        await self._check_backpressure()

    async def _on_work_failed(self, event: Any) -> None:
        """Handle WORK_FAILED event (permanent failure)."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._jobs_lock:
            if work_id in self._jobs:
                job = self._jobs[work_id]
                job.status = "failed"
                self._total_failed += 1

                # Decrement node load
                if job.claimed_by:
                    self._node_job_counts[job.claimed_by] = max(
                        0, self._node_job_counts[job.claimed_by] - 1
                    )

                # Remove from active tracking
                del self._jobs[work_id]

        self._stats.events_processed += 1
        self._record_error(f"Work failed: {work_id}")
        logger.warning(f"[{self.name}] Work failed permanently: {work_id}")

    async def _on_work_retry(self, event: Any) -> None:
        """Handle WORK_RETRY event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._jobs_lock:
            if work_id in self._jobs:
                self._jobs[work_id].retry_count += 1
                self._jobs[work_id].status = "pending"
                self._jobs[work_id].claimed_at = 0.0
                self._jobs[work_id].started_at = 0.0
                self._total_retries += 1

        self._stats.events_processed += 1

    async def _on_work_cancelled(self, event: Any) -> None:
        """Handle WORK_CANCELLED event.

        Dec 28, 2025 - Added handler for previously orphan event.
        """
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")
        reason = payload.get("reason", "unknown")

        async with self._jobs_lock:
            if work_id in self._jobs:
                job = self._jobs[work_id]
                # Decrement node job count if job was claimed
                if job.claimed_by and job.claimed_by in self._node_job_counts:
                    self._node_job_counts[job.claimed_by] = max(
                        0, self._node_job_counts[job.claimed_by] - 1
                    )
                # Remove from active tracking
                del self._jobs[work_id]

        self._stats.events_processed += 1
        logger.debug(f"[{self.name}] Work cancelled: {work_id}, reason: {reason}")

    # =========================================================================
    # Backpressure and Overload Detection
    # =========================================================================

    async def _check_backpressure(self) -> None:
        """Check if backpressure should be activated/deactivated."""
        now = time.time()

        # Throttle checks to avoid spam
        if now - self._last_backpressure_check < self._backpressure_check_interval:
            return
        self._last_backpressure_check = now

        async with self._jobs_lock:
            pending_count = sum(1 for j in self._jobs.values() if j.status == "pending")

        threshold = self._backpressure_threshold

        if pending_count > threshold and not self._backpressure_active:
            self._backpressure_active = True
            await self._emit_backpressure_event(True, pending_count)
            logger.warning(
                f"[{self.name}] BACKPRESSURE ACTIVATED: {pending_count} pending jobs"
            )

        elif pending_count <= threshold * 0.7 and self._backpressure_active:
            self._backpressure_active = False
            await self._emit_backpressure_event(False, pending_count)
            logger.info(
                f"[{self.name}] Backpressure deactivated: {pending_count} pending jobs"
            )

    async def _check_node_overload(self, node_id: str) -> None:
        """Check if a node is overloaded."""
        job_count = self._node_job_counts.get(node_id, 0)
        if job_count > self._node_overload_threshold:
            await self._emit_node_overload_event(node_id, job_count)
            logger.warning(
                f"[{self.name}] Node overloaded: {node_id} has {job_count} jobs"
            )

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_backpressure_event(self, active: bool, queue_depth: int) -> None:
        """Emit backpressure activation/deactivation event."""
        try:
            from app.coordination.event_router import publish
            event_type = "BACKPRESSURE_ACTIVATED" if active else "BACKPRESSURE_DEACTIVATED"
            await publish(
                event_type=event_type,
                payload={
                    "active": active,
                    "queue_depth": queue_depth,
                    "threshold": self._backpressure_threshold,
                    "timestamp": time.time(),
                },
                source=self.name,
            )
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit backpressure event: {e}")

    async def _emit_node_overload_event(self, node_id: str, job_count: int) -> None:
        """Emit node overload event."""
        try:
            from app.coordination.event_router import publish

            await publish(
                event_type="NODE_OVERLOADED",
                payload={
                    "node_id": node_id,
                    "job_count": job_count,
                    "threshold": self._node_overload_threshold,
                    "timestamp": time.time(),
                },
                source=self.name,
            )
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit overload event: {e}")

    async def _emit_stuck_job_event(self, job: JobTracker, stuck_duration: float) -> None:
        """Emit stuck job detected event."""
        try:
            from app.coordination.event_router import publish

            await publish(
                event_type="STUCK_JOB_DETECTED",
                payload={
                    "work_id": job.work_id,
                    "work_type": job.work_type,
                    "claimed_by": job.claimed_by,
                    "stuck_duration_seconds": stuck_duration,
                    "threshold": self._stuck_job_threshold_seconds,
                    "timestamp": time.time(),
                },
                source=self.name,
            )
        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit stuck job event: {e}")

    # =========================================================================
    # Statistics and Status
    # =========================================================================

    def get_queue_stats(self) -> QueueStats:
        """Get current queue statistics."""
        now = time.time()

        # Count by status
        pending = 0
        claimed = 0
        running = 0

        for job in self._jobs.values():
            if job.status == "pending":
                pending += 1
            elif job.status == "claimed":
                claimed += 1
            elif job.status == "running":
                running += 1

        # Calculate latency metrics
        avg_latency = 0.0
        p95_latency = 0.0
        max_latency = 0.0

        if self._completed_latencies:
            sorted_latencies = sorted(self._completed_latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]
            max_latency = sorted_latencies[-1]

        # Count stuck jobs
        stuck_count = 0
        for job in self._jobs.values():
            if job.status == "claimed" and job.claimed_at > 0 and job.started_at == 0:
                if now - job.claimed_at > self._stuck_job_threshold_seconds:
                    stuck_count += 1

        return QueueStats(
            pending_count=pending,
            claimed_count=claimed,
            running_count=running,
            completed_count=self._total_completed,
            failed_count=self._total_failed,
            retry_count=self._total_retries,
            avg_latency_seconds=avg_latency,
            p95_latency_seconds=p95_latency,
            max_latency_seconds=max_latency,
            jobs_per_node=dict(self._node_job_counts),
            backpressure_active=self._backpressure_active,
            stuck_job_count=stuck_count,
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        stats = self.get_queue_stats()
        base_status = super().get_status()
        base_status.update({
            "pending_count": stats.pending_count,
            "running_count": stats.running_count,
            "completed_count": stats.completed_count,
            "failed_count": stats.failed_count,
            "backpressure_active": stats.backpressure_active,
            "stuck_job_count": stats.stuck_job_count,
            "avg_latency_seconds": round(stats.avg_latency_seconds, 2),
        })
        return base_status

    def health_check(self) -> HealthCheckResult:
        """Return health status with work-queue specific metrics."""
        base_result = super().health_check()
        if not base_result.healthy:
            return base_result

        stats = self.get_queue_stats()

        if stats.backpressure_active:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Backpressure active: {stats.pending_count} pending jobs",
                details=base_result.details,
            )

        if stats.stuck_job_count > 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"{stats.stuck_job_count} stuck job(s) detected",
                details=base_result.details,
            )

        return base_result


# =============================================================================
# Singleton Accessor Functions
# =============================================================================


def get_work_queue_monitor() -> WorkQueueMonitorDaemon:
    """Get the singleton WorkQueueMonitorDaemon instance."""
    return WorkQueueMonitorDaemon.get_instance()


def get_work_queue_monitor_sync() -> WorkQueueMonitorDaemon:
    """Get the singleton instance synchronously."""
    return WorkQueueMonitorDaemon.get_instance()


__all__ = [
    "WorkQueueMonitorDaemon",
    "QueueStats",
    "JobTracker",
    "get_work_queue_monitor",
    "get_work_queue_monitor_sync",
    "BACKPRESSURE_THRESHOLD",
    "STUCK_JOB_THRESHOLD_SECONDS",
    "NODE_OVERLOAD_THRESHOLD",
]
