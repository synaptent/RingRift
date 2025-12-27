"""Vast.ai CPU Pipeline Daemon - CPU-only preprocessing on cheap Vast instances.

This daemon manages CPU-only Vast.ai instances for data preprocessing tasks
that don't require GPU acceleration. This allows using cheap CPU instances
for export, merge, and validation operations.

Use cases:
- NPZ export from databases (CPU-bound, no GPU needed)
- Database merging and deduplication
- Data validation and quality checks
- Feature extraction preprocessing

Architecture:
    1. Monitors work queue for CPU-suitable jobs
    2. Claims jobs that can run on CPU-only instances
    3. Dispatches to available Vast.ai CPU instances
    4. Reports results back to coordinator
    5. Emits CPU_PIPELINE_JOB_COMPLETED events

Usage:
    # Via DaemonManager
    manager.register_factory(DaemonType.VAST_CPU_PIPELINE, daemon.run)

    # Standalone
    python -m app.distributed.vast_cpu_pipeline
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CPUJobType(Enum):
    """Types of CPU-suitable jobs."""

    NPZ_EXPORT = "npz_export"
    DATABASE_MERGE = "database_merge"
    DATA_VALIDATION = "data_validation"
    FEATURE_EXTRACTION = "feature_extraction"
    DATABASE_CLEANUP = "database_cleanup"


@dataclass
class CPUJob:
    """A CPU pipeline job."""

    job_id: str
    job_type: CPUJobType
    params: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    node_id: str | None = None
    success: bool = False
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class VastCpuPipelineConfig:
    """Configuration for Vast.ai CPU pipeline daemon."""

    # Polling settings
    poll_interval_seconds: float = 30.0
    max_concurrent_jobs: int = 3

    # Job settings
    job_timeout_seconds: float = 3600.0  # 1 hour
    retry_count: int = 2

    # Instance settings
    target_instance_count: int = 2
    max_hourly_cost: float = 1.0  # Maximum hourly cost per instance
    min_cpu_cores: int = 4
    min_ram_gb: int = 16

    # Event settings
    emit_events: bool = True


class VastCpuPipelineDaemon:
    """Daemon that manages CPU-only jobs on Vast.ai instances.

    Monitors the work queue for CPU-suitable jobs and dispatches them
    to available Vast.ai CPU-only instances for processing.
    """

    def __init__(self, config: VastCpuPipelineConfig | None = None):
        self.config = config or VastCpuPipelineConfig()
        self._running = False
        self._active_jobs: dict[str, CPUJob] = {}
        self._job_history: list[CPUJob] = []

    async def start(self) -> None:
        """Start the daemon."""
        logger.info("VastCpuPipelineDaemon starting...")
        self._running = True

        while self._running:
            try:
                # Check for new jobs
                await self._poll_for_jobs()

                # Check job status
                await self._check_job_status()

                await asyncio.sleep(self.config.poll_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Vast CPU pipeline loop: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

        logger.info("VastCpuPipelineDaemon stopped")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._running = False

    async def _poll_for_jobs(self) -> None:
        """Poll for CPU-suitable jobs from the work queue."""
        if len(self._active_jobs) >= self.config.max_concurrent_jobs:
            return

        try:
            # Get available CPU instances
            instances = await self._get_available_instances()
            if not instances:
                logger.debug("No CPU instances available")
                return

            # Get pending CPU jobs from queue
            jobs = await self._get_pending_cpu_jobs()
            if not jobs:
                return

            # Dispatch jobs to instances
            for job in jobs[: self.config.max_concurrent_jobs - len(self._active_jobs)]:
                instance = instances.pop(0) if instances else None
                if instance:
                    await self._dispatch_job(job, instance)

        except Exception as e:
            logger.error(f"Failed to poll for jobs: {e}")

    async def _get_available_instances(self) -> list[dict[str, Any]]:
        """Get list of available Vast.ai CPU instances.

        Returns:
            List of instance info dicts with 'id', 'host', 'ssh_port' keys.
        """
        # TODO: Integrate with Vast.ai API to get running instances
        # For now, return empty list (stub implementation)
        return []

    async def _get_pending_cpu_jobs(self) -> list[CPUJob]:
        """Get pending CPU-suitable jobs from the work queue.

        Returns:
            List of CPUJob objects ready for dispatch.
        """
        # TODO: Integrate with work queue to get pending jobs
        # For now, return empty list (stub implementation)
        return []

    async def _dispatch_job(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Dispatch a job to a Vast.ai instance."""
        logger.info(f"Dispatching job {job.job_id} ({job.job_type.value}) to instance {instance.get('id')}")

        job.started_at = time.time()
        job.node_id = instance.get("id")
        self._active_jobs[job.job_id] = job

        try:
            # Build and execute command based on job type
            if job.job_type == CPUJobType.NPZ_EXPORT:
                await self._run_npz_export(job, instance)
            elif job.job_type == CPUJobType.DATABASE_MERGE:
                await self._run_database_merge(job, instance)
            elif job.job_type == CPUJobType.DATA_VALIDATION:
                await self._run_data_validation(job, instance)
            else:
                logger.warning(f"Unknown job type: {job.job_type}")
                job.error = f"Unknown job type: {job.job_type}"

        except Exception as e:
            job.error = str(e)
            logger.error(f"Job {job.job_id} failed: {e}")

    async def _run_npz_export(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Run NPZ export job on instance."""
        # TODO: SSH to instance and run export script
        logger.info(f"Running NPZ export: {job.params}")

    async def _run_database_merge(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Run database merge job on instance."""
        # TODO: SSH to instance and run merge script
        logger.info(f"Running database merge: {job.params}")

    async def _run_data_validation(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Run data validation job on instance."""
        # TODO: SSH to instance and run validation script
        logger.info(f"Running data validation: {job.params}")

    async def _check_job_status(self) -> None:
        """Check status of active jobs and handle completions."""
        completed: list[str] = []

        for job_id, job in self._active_jobs.items():
            try:
                # Check if job timed out
                if job.started_at:
                    elapsed = time.time() - job.started_at
                    if elapsed > self.config.job_timeout_seconds:
                        job.error = "Job timed out"
                        job.success = False
                        job.completed_at = time.time()
                        completed.append(job_id)
                        continue

                # TODO: Check actual job status via SSH or API
                # For now, assume job is still running

            except Exception as e:
                logger.error(f"Failed to check job {job_id} status: {e}")

        # Clean up completed jobs
        for job_id in completed:
            job = self._active_jobs.pop(job_id)
            self._job_history.append(job)

            if self.config.emit_events:
                await self._emit_job_completed_event(job)

    async def _emit_job_completed_event(self, job: CPUJob) -> None:
        """Emit CPU_PIPELINE_JOB_COMPLETED event."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()
            if router is None:
                return

            await router.publish(
                DataEventType.CPU_PIPELINE_JOB_COMPLETED,
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "success": job.success,
                    "node_id": job.node_id,
                    "duration_seconds": (
                        (job.completed_at - job.started_at)
                        if job.completed_at and job.started_at
                        else None
                    ),
                    "error": job.error,
                    "result": job.result,
                    "timestamp": time.time(),
                },
            )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to emit job completed event: {e}")


async def run() -> None:
    """Run the daemon (entry point for DaemonManager)."""
    daemon = VastCpuPipelineDaemon()
    await daemon.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run())
