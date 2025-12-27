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

        December 2025: Queries distributed_hosts.yaml for vast.ai instances
        that are marked as CPU-capable or have no GPU.

        Returns:
            List of instance info dicts with 'id', 'host', 'ssh_port', 'ssh_user', 'ssh_key' keys.
        """
        instances = []

        try:
            # Dec 2025: Fixed import - host_config doesn't exist, use hosts module
            from app.distributed.hosts import load_remote_hosts

            hosts = load_remote_hosts()
            for name, host in hosts.items():
                # Check if it's a Vast.ai instance
                if not name.startswith("vast-") and host.provider != "vast":
                    continue

                # Check if instance is suitable for CPU work
                # (either no GPU or explicitly marked as cpu_capable)
                is_cpu_capable = (
                    getattr(host, "cpu_capable", False)
                    or not getattr(host, "has_gpu", True)
                    or host.gpu_count == 0
                )

                if is_cpu_capable and host.best_ip:
                    instances.append({
                        "id": name,
                        "host": host.best_ip,
                        "ssh_port": host.ssh_port or 22,
                        "ssh_user": host.ssh_user or "root",
                        "ssh_key": host.ssh_key,
                        "ringrift_path": host.ringrift_path or "~/ringrift/ai-service",
                    })

            # Filter to instances that are reachable (quick health check)
            reachable = []
            for inst in instances[:5]:  # Limit to first 5 to avoid delays
                try:
                    from app.core.ssh import run_ssh_command_async

                    result = await run_ssh_command_async(
                        inst["id"],
                        "echo ok",
                        timeout=5,
                    )
                    if result.returncode == 0:
                        reachable.append(inst)
                except Exception:
                    pass  # Instance not reachable

            return reachable

        except ImportError as e:
            logger.debug(f"host_config not available: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to get available instances: {e}")
            return []

    async def _get_pending_cpu_jobs(self) -> list[CPUJob]:
        """Get pending CPU-suitable jobs from the work queue.

        December 2025: Queries the work queue for jobs that match CPU capabilities.

        Returns:
            List of CPUJob objects ready for dispatch.
        """
        jobs = []

        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            if queue is None:
                return []

            # Get pending items that match CPU capabilities
            pending = queue.get_pending_items(
                capabilities=["cpu", "npz_export", "data_validation", "database_merge"],
                limit=self.config.max_concurrent_jobs,
            )

            for item in pending:
                # Map work_type to CPUJobType
                work_type = item.work_type
                try:
                    if "export" in work_type.lower() or "npz" in work_type.lower():
                        job_type = CPUJobType.NPZ_EXPORT
                    elif "merge" in work_type.lower():
                        job_type = CPUJobType.DATABASE_MERGE
                    elif "valid" in work_type.lower() or "quality" in work_type.lower():
                        job_type = CPUJobType.DATA_VALIDATION
                    elif "cleanup" in work_type.lower():
                        job_type = CPUJobType.DATABASE_CLEANUP
                    else:
                        continue  # Skip unsupported job types

                    jobs.append(CPUJob(
                        job_id=item.work_id,
                        job_type=job_type,
                        params=item.config or {},
                        priority=item.priority or 0,
                        created_at=item.created_at or time.time(),
                    ))

                except Exception as e:
                    logger.debug(f"Failed to convert work item to CPUJob: {e}")

            return jobs

        except ImportError as e:
            logger.debug(f"work_queue not available: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to get pending CPU jobs: {e}")
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
        """Run NPZ export job on instance.

        December 2025: Executes export_replay_dataset.py via SSH.
        """
        logger.info(f"Running NPZ export: {job.params}")

        try:
            from app.core.ssh import run_ssh_command_async

            ringrift_path = instance.get("ringrift_path", "~/ringrift/ai-service")

            # Build export command from job params
            board_type = job.params.get("board_type", "hex8")
            num_players = job.params.get("num_players", 2)
            output_path = job.params.get("output_path", f"data/training/{board_type}_{num_players}p.npz")
            db_path = job.params.get("db_path", "")

            cmd = (
                f"cd {ringrift_path} && "
                f"PYTHONPATH=. python scripts/export_replay_dataset.py "
                f"--board-type {board_type} --num-players {num_players} "
                f"--output {output_path}"
            )
            if db_path:
                cmd += f" --db {db_path}"
            else:
                cmd += " --use-discovery"

            # Run as background process with nohup
            bg_cmd = f"nohup {cmd} > /tmp/export_{job.job_id}.log 2>&1 & echo $!"

            result = await run_ssh_command_async(
                instance["id"],
                bg_cmd,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                job.result["pid"] = pid
                job.result["log_file"] = f"/tmp/export_{job.job_id}.log"
                logger.info(f"NPZ export started on {instance['id']} with PID {pid}")
            else:
                job.error = f"Failed to start export: {result.stderr}"

        except Exception as e:
            job.error = str(e)
            logger.error(f"NPZ export failed: {e}")

    async def _run_database_merge(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Run database merge job on instance.

        December 2025: Executes database merge via SSH.
        """
        logger.info(f"Running database merge: {job.params}")

        try:
            from app.core.ssh import run_ssh_command_async

            ringrift_path = instance.get("ringrift_path", "~/ringrift/ai-service")

            # Build merge command
            source_dbs = job.params.get("source_dbs", [])
            target_db = job.params.get("target_db", "data/games/merged.db")

            if not source_dbs:
                job.error = "No source databases specified"
                return

            sources_str = " ".join(source_dbs)
            cmd = (
                f"cd {ringrift_path} && "
                f"PYTHONPATH=. python scripts/merge_databases.py "
                f"--sources {sources_str} --target {target_db}"
            )

            bg_cmd = f"nohup {cmd} > /tmp/merge_{job.job_id}.log 2>&1 & echo $!"

            result = await run_ssh_command_async(
                instance["id"],
                bg_cmd,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                job.result["pid"] = pid
                job.result["log_file"] = f"/tmp/merge_{job.job_id}.log"
                logger.info(f"Database merge started on {instance['id']} with PID {pid}")
            else:
                job.error = f"Failed to start merge: {result.stderr}"

        except Exception as e:
            job.error = str(e)
            logger.error(f"Database merge failed: {e}")

    async def _run_data_validation(self, job: CPUJob, instance: dict[str, Any]) -> None:
        """Run data validation job on instance.

        December 2025: Executes data quality validation via SSH.
        """
        logger.info(f"Running data validation: {job.params}")

        try:
            from app.core.ssh import run_ssh_command_async

            ringrift_path = instance.get("ringrift_path", "~/ringrift/ai-service")

            # Build validation command
            db_path = job.params.get("db_path", "")
            npz_path = job.params.get("npz_path", "")

            if db_path:
                cmd = (
                    f"cd {ringrift_path} && "
                    f"PYTHONPATH=. python -m app.training.data_quality --db {db_path}"
                )
            elif npz_path:
                cmd = (
                    f"cd {ringrift_path} && "
                    f"PYTHONPATH=. python -m app.training.data_quality --npz {npz_path} --detailed"
                )
            else:
                job.error = "No db_path or npz_path specified"
                return

            bg_cmd = f"nohup {cmd} > /tmp/validation_{job.job_id}.log 2>&1 & echo $!"

            result = await run_ssh_command_async(
                instance["id"],
                bg_cmd,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                job.result["pid"] = pid
                job.result["log_file"] = f"/tmp/validation_{job.job_id}.log"
                logger.info(f"Data validation started on {instance['id']} with PID {pid}")
            else:
                job.error = f"Failed to start validation: {result.stderr}"

        except Exception as e:
            job.error = str(e)
            logger.error(f"Data validation failed: {e}")

    async def _check_job_status(self) -> None:
        """Check status of active jobs and handle completions.

        December 2025: Checks if background process is still running via SSH.
        """
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

                # Check actual job status via SSH
                pid = job.result.get("pid")
                if pid and job.node_id:
                    try:
                        from app.core.ssh import run_ssh_command_async

                        # Check if process is still running
                        result = await run_ssh_command_async(
                            job.node_id,
                            f"ps -p {pid} -o pid= 2>/dev/null || echo 'done'",
                            timeout=10,
                        )

                        if result.returncode == 0:
                            output = result.stdout.strip()
                            if output == "done" or not output:
                                # Process finished - check log for result
                                log_file = job.result.get("log_file", "")
                                if log_file:
                                    log_result = await run_ssh_command_async(
                                        job.node_id,
                                        f"tail -20 {log_file} 2>/dev/null",
                                        timeout=10,
                                    )
                                    log_output = log_result.stdout if log_result.returncode == 0 else ""

                                    # Check for success indicators
                                    if "error" in log_output.lower() or "failed" in log_output.lower():
                                        job.success = False
                                        job.error = f"Job failed - check {log_file}"
                                    else:
                                        job.success = True

                                    job.result["log_tail"] = log_output[-500:]  # Last 500 chars

                                job.completed_at = time.time()
                                completed.append(job_id)
                                logger.info(
                                    f"Job {job_id} completed on {job.node_id}: "
                                    f"success={job.success}"
                                )

                    except Exception as e:
                        logger.debug(f"Failed to check PID {pid} on {job.node_id}: {e}")

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
