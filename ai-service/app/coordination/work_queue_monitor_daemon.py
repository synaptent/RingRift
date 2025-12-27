"""Work Queue Monitor Daemon - Tracks work queue lifecycle events.

December 2025: Created to address critical gap in event coordination.
Previously, WORK_QUEUED, WORK_STARTED, WORK_COMPLETED, WORK_FAILED events
were emitted but had NO subscribers - making queue visibility impossible.

This daemon subscribes to all WORK_* events and provides:
1. Queue depth tracking
2. Job latency monitoring
3. Stuck job detection (claimed but not started)
4. Node workload distribution
5. Backpressure signaling

Event Subscriptions:
- WORK_QUEUED: Track new work items
- WORK_CLAIMED: Track work assignment to nodes
- WORK_STARTED: Track job execution start
- WORK_COMPLETED: Track successful completion
- WORK_FAILED: Track permanent failures
- WORK_RETRY: Track retry attempts

Events Emitted:
- BACKPRESSURE_ACTIVATED: Queue depth exceeds threshold
- BACKPRESSURE_DEACTIVATED: Queue depth recovered
- STUCK_JOB_DETECTED: Job claimed but not started for too long
- NODE_OVERLOADED: Node has too many concurrent jobs

Usage:
    from app.coordination.work_queue_monitor_daemon import (
        WorkQueueMonitorDaemon,
        get_work_queue_monitor,
    )

    # Start monitoring
    monitor = get_work_queue_monitor()
    await monitor.start()

    # Get queue statistics
    stats = monitor.get_queue_stats()
    print(f"Queue depth: {stats['pending_count']}")
    print(f"Avg latency: {stats['avg_latency_seconds']:.1f}s")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds (December 27, 2025: Centralized in coordination_defaults.py)
from app.config.coordination_defaults import WorkQueueMonitorDefaults

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


class WorkQueueMonitorDaemon:
    """Daemon that monitors work queue events and provides visibility.

    Subscribes to all WORK_* events and tracks:
    - Queue depth and composition
    - Job latency (time from queued to completed)
    - Stuck jobs (claimed but not progressing)
    - Node workload distribution
    - Backpressure conditions
    """

    def __init__(self):
        """Initialize the work queue monitor."""
        self._running = False
        self._subscribed = False

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

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start the monitor daemon."""
        if self._running:
            return True

        self._running = True
        success = await self._subscribe_to_events()

        if success:
            logger.info("[WorkQueueMonitor] Started - monitoring WORK_* events")

            # Start background monitoring loop
            asyncio.create_task(self._monitoring_loop())
        else:
            logger.warning("[WorkQueueMonitor] Started without event subscriptions")

        return success

    async def stop(self) -> None:
        """Stop the monitor daemon."""
        self._running = False
        await self._unsubscribe_from_events()
        logger.info("[WorkQueueMonitor] Stopped")

    async def _subscribe_to_events(self) -> bool:
        """Subscribe to all WORK_* events."""
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()

            # Subscribe to all work queue events
            router.subscribe(DataEventType.WORK_QUEUED.value, self._on_work_queued)
            router.subscribe(DataEventType.WORK_CLAIMED.value, self._on_work_claimed)
            router.subscribe(DataEventType.WORK_STARTED.value, self._on_work_started)
            router.subscribe(DataEventType.WORK_COMPLETED.value, self._on_work_completed)
            router.subscribe(DataEventType.WORK_FAILED.value, self._on_work_failed)

            # WORK_RETRY may not exist - check first
            if hasattr(DataEventType, "WORK_RETRY"):
                router.subscribe(DataEventType.WORK_RETRY.value, self._on_work_retry)

            self._subscribed = True
            logger.info("[WorkQueueMonitor] Subscribed to 5+ WORK_* events")
            return True

        except ImportError as e:
            logger.warning(f"[WorkQueueMonitor] data_events not available: {e}")
            return False
        except Exception as e:
            logger.error(f"[WorkQueueMonitor] Failed to subscribe: {e}")
            return False

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import get_router

            router = get_router()

            router.unsubscribe(DataEventType.WORK_QUEUED.value, self._on_work_queued)
            router.unsubscribe(DataEventType.WORK_CLAIMED.value, self._on_work_claimed)
            router.unsubscribe(DataEventType.WORK_STARTED.value, self._on_work_started)
            router.unsubscribe(DataEventType.WORK_COMPLETED.value, self._on_work_completed)
            router.unsubscribe(DataEventType.WORK_FAILED.value, self._on_work_failed)

            if hasattr(DataEventType, "WORK_RETRY"):
                router.unsubscribe(DataEventType.WORK_RETRY.value, self._on_work_retry)

            self._subscribed = False

        except Exception as e:
            logger.warning(f"[WorkQueueMonitor] Error unsubscribing: {e}")

    async def _on_work_queued(self, event: Any) -> None:
        """Handle WORK_QUEUED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")
        if not work_id:
            return

        async with self._lock:
            self._jobs[work_id] = JobTracker(
                work_id=work_id,
                work_type=payload.get("work_type", "unknown"),
                priority=payload.get("priority", 50),
                config_key=payload.get("config_key", ""),
                queued_at=time.time(),
                status="pending",
            )
            self._total_queued += 1

        logger.debug(f"[WorkQueueMonitor] Work queued: {work_id}")
        await self._check_backpressure()

    async def _on_work_claimed(self, event: Any) -> None:
        """Handle WORK_CLAIMED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")
        claimed_by = payload.get("claimed_by", "") or payload.get("node_id", "")

        async with self._lock:
            if work_id in self._jobs:
                self._jobs[work_id].claimed_at = time.time()
                self._jobs[work_id].claimed_by = claimed_by
                self._jobs[work_id].status = "claimed"

                # Track per-node load
                if claimed_by:
                    self._node_job_counts[claimed_by] += 1
                    await self._check_node_overload(claimed_by)

    async def _on_work_started(self, event: Any) -> None:
        """Handle WORK_STARTED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._lock:
            if work_id in self._jobs:
                self._jobs[work_id].started_at = time.time()
                self._jobs[work_id].status = "running"

    async def _on_work_completed(self, event: Any) -> None:
        """Handle WORK_COMPLETED event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._lock:
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
                    if len(self._completed_latencies) > LATENCY_WINDOW_SIZE:
                        self._completed_latencies.pop(0)

                # Decrement node load
                if job.claimed_by:
                    self._node_job_counts[job.claimed_by] = max(
                        0, self._node_job_counts[job.claimed_by] - 1
                    )

                # Remove from active tracking
                del self._jobs[work_id]

        await self._check_backpressure()

    async def _on_work_failed(self, event: Any) -> None:
        """Handle WORK_FAILED event (permanent failure)."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._lock:
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

        logger.warning(f"[WorkQueueMonitor] Work failed permanently: {work_id}")

    async def _on_work_retry(self, event: Any) -> None:
        """Handle WORK_RETRY event."""
        payload = event.payload if hasattr(event, "payload") else event

        work_id = payload.get("work_id", "")

        async with self._lock:
            if work_id in self._jobs:
                self._jobs[work_id].retry_count += 1
                self._jobs[work_id].status = "pending"
                self._jobs[work_id].claimed_at = 0.0
                self._jobs[work_id].started_at = 0.0
                self._total_retries += 1

    async def _check_backpressure(self) -> None:
        """Check if backpressure should be activated/deactivated."""
        now = time.time()

        # Throttle checks to avoid spam
        if now - self._last_backpressure_check < 10:
            return
        self._last_backpressure_check = now

        async with self._lock:
            pending_count = sum(1 for j in self._jobs.values() if j.status == "pending")

        was_active = self._backpressure_active

        if pending_count > BACKPRESSURE_THRESHOLD and not self._backpressure_active:
            self._backpressure_active = True
            await self._emit_backpressure_event(True, pending_count)
            logger.warning(
                f"[WorkQueueMonitor] BACKPRESSURE ACTIVATED: {pending_count} pending jobs"
            )

        elif pending_count <= BACKPRESSURE_THRESHOLD * 0.7 and self._backpressure_active:
            self._backpressure_active = False
            await self._emit_backpressure_event(False, pending_count)
            logger.info(
                f"[WorkQueueMonitor] Backpressure deactivated: {pending_count} pending jobs"
            )

    async def _check_node_overload(self, node_id: str) -> None:
        """Check if a node is overloaded."""
        job_count = self._node_job_counts.get(node_id, 0)
        if job_count > NODE_OVERLOAD_THRESHOLD:
            await self._emit_node_overload_event(node_id, job_count)
            logger.warning(
                f"[WorkQueueMonitor] Node overloaded: {node_id} has {job_count} jobs"
            )

    async def _emit_backpressure_event(self, active: bool, queue_depth: int) -> None:
        """Emit backpressure activation/deactivation event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            event_type = "BACKPRESSURE_ACTIVATED" if active else "BACKPRESSURE_DEACTIVATED"
            await router.publish(
                event_type,
                {
                    "active": active,
                    "queue_depth": queue_depth,
                    "threshold": BACKPRESSURE_THRESHOLD,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"[WorkQueueMonitor] Failed to emit backpressure event: {e}")

    async def _emit_node_overload_event(self, node_id: str, job_count: int) -> None:
        """Emit node overload event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            await router.publish(
                "NODE_OVERLOADED",
                {
                    "node_id": node_id,
                    "job_count": job_count,
                    "threshold": NODE_OVERLOAD_THRESHOLD,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"[WorkQueueMonitor] Failed to emit overload event: {e}")

    async def _emit_stuck_job_event(self, job: JobTracker, stuck_duration: float) -> None:
        """Emit stuck job detected event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            await router.publish(
                "STUCK_JOB_DETECTED",
                {
                    "work_id": job.work_id,
                    "work_type": job.work_type,
                    "claimed_by": job.claimed_by,
                    "stuck_duration_seconds": stuck_duration,
                    "threshold": STUCK_JOB_THRESHOLD_SECONDS,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug(f"[WorkQueueMonitor] Failed to emit stuck job event: {e}")

    async def _monitoring_loop(self) -> None:
        """Background loop for periodic monitoring checks."""
        while self._running:
            try:
                await self._check_stuck_jobs()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WorkQueueMonitor] Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _check_stuck_jobs(self) -> None:
        """Check for jobs that are stuck (claimed but not started)."""
        now = time.time()
        stuck_jobs: list[tuple[JobTracker, float]] = []

        async with self._lock:
            for job in self._jobs.values():
                # Job is stuck if claimed but not started for too long
                if job.status == "claimed" and job.claimed_at > 0 and job.started_at == 0:
                    stuck_duration = now - job.claimed_at
                    if stuck_duration > STUCK_JOB_THRESHOLD_SECONDS:
                        stuck_jobs.append((job, stuck_duration))

        # Emit events for stuck jobs (outside lock)
        for job, duration in stuck_jobs:
            await self._emit_stuck_job_event(job, duration)
            logger.warning(
                f"[WorkQueueMonitor] Stuck job detected: {job.work_id} "
                f"claimed by {job.claimed_by} for {duration:.0f}s"
            )

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
                if now - job.claimed_at > STUCK_JOB_THRESHOLD_SECONDS:
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
        """Get daemon status for health checks."""
        stats = self.get_queue_stats()
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "pending_count": stats.pending_count,
            "running_count": stats.running_count,
            "completed_count": stats.completed_count,
            "failed_count": stats.failed_count,
            "backpressure_active": stats.backpressure_active,
            "stuck_job_count": stats.stuck_job_count,
            "avg_latency_seconds": round(stats.avg_latency_seconds, 2),
        }

    def health_check(self) -> "HealthCheckResult":
        """Check daemon health status."""
        try:
            from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

            if not self._running:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.STOPPED,
                    message="Work queue monitor not running",
                )

            if not self._subscribed:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message="Work queue monitor not subscribed to events",
                )

            stats = self.get_queue_stats()

            # Check for issues
            if stats.stuck_job_count > 5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"{stats.stuck_job_count} stuck jobs detected",
                    details=self.get_status(),
                )

            if stats.backpressure_active:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"Backpressure active: {stats.pending_count} pending",
                    details=self.get_status(),
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Queue monitor running (pending: {stats.pending_count}, running: {stats.running_count})",
                details=self.get_status(),
            )

        except ImportError:
            return {"healthy": self._running and self._subscribed}


# Singleton instance
_monitor_instance: WorkQueueMonitorDaemon | None = None
_monitor_lock = asyncio.Lock()


async def get_work_queue_monitor() -> WorkQueueMonitorDaemon:
    """Get or create the singleton WorkQueueMonitorDaemon instance."""
    global _monitor_instance

    async with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = WorkQueueMonitorDaemon()
        return _monitor_instance


def get_work_queue_monitor_sync() -> WorkQueueMonitorDaemon:
    """Get the singleton instance synchronously (may create if not exists)."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = WorkQueueMonitorDaemon()
    return _monitor_instance


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
