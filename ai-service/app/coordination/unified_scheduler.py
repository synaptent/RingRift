"""Unified Scheduler - Cross-Orchestrator Job Management.

Routes jobs to the appropriate backend based on target node:
- Slurm: Lambda Labs nodes (GH200, H100, A10)
- Vast.ai API: Dynamic GPU instances
- P2P: Hetzner, AWS, and other nodes

This provides a single interface for job submission across all providers
while leveraging each orchestrator's strengths.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  UNIFIED SCHEDULER                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │ Slurm Backend│  │ Vast Backend │  │ P2P Backend  │      │
    │  │   (Lambda)   │  │   (Vast.ai)  │  │(Hetzner/AWS) │      │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
    │         │                 │                 │               │
    │         └─────────────────┼─────────────────┘               │
    │                           ▼                                 │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │           UNIFIED JOB QUEUE (SQLite)                 │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from app.coordination.unified_scheduler import (
        UnifiedScheduler,
        get_scheduler,
        UnifiedJob,
        Backend,
    )

    scheduler = get_scheduler()

    # Submit job - automatically routes to correct backend
    job_id = await scheduler.submit(UnifiedJob(
        name="selfplay-square8-2p",
        job_type="selfplay",
        config={"board_type": "square8", "num_players": 2},
    ))

    # Submit to specific node
    job_id = await scheduler.submit(UnifiedJob(
        name="training-run",
        job_type="training",
        target_node="gpu-node-1",
        config={"data_path": "data/training/latest"},
    ))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.p2p_backend import P2PBackend
    from app.coordination.slurm_backend import SlurmJobState, SlurmJobStatus

__all__ = [
    # Enums
    "Backend",
    "JobState",
    # Dataclasses
    "JobStatus",
    "JobType",
    "UnifiedJob",
    # Scheduler
    "UnifiedScheduler",
    "get_scheduler",
    # Convenience functions
    "submit_gpu_selfplay",
    "submit_selfplay",
    "submit_training",
]

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    """Available job backends."""
    SLURM = "slurm"
    VAST = "vast"
    P2P = "p2p"
    AUTO = "auto"  # Automatically select based on availability


class JobType(str, Enum):
    """Standard job types."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"
    TRAINING = "training"
    EVALUATION = "evaluation"
    TOURNAMENT = "tournament"
    GAUNTLET = "gauntlet"
    CUSTOM = "custom"


class JobState(str, Enum):
    """Unified job states across all backends."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# Node patterns for backend routing
# Note: Lambda Labs terminated Dec 2025, patterns kept for reference only
SLURM_PATTERNS: list[str] = []

VAST_PATTERNS = [
    "vast-*",
]

P2P_PATTERNS = [
    "ringrift-*",
    "ip-172-*",
    "hetzner-*",
    "*-cpu*",
]


@dataclass
class UnifiedJob:
    """Unified job specification across all backends."""
    name: str
    job_type: JobType = JobType.SELFPLAY

    # Targeting
    target_node: str | None = None
    target_backend: Backend = Backend.AUTO
    target_gpu_type: str | None = None  # e.g., "gh200", "h100", "rtx5090"

    # Resources
    cpus: int = 16
    memory_gb: int = 64
    gpus: int = 1
    time_limit_hours: float = 8.0

    # Job configuration
    config: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)

    # Execution
    command: str | None = None
    script_path: str | None = None
    working_dir: str | None = None

    # Priority and dependencies
    priority: int = 50  # 0-100, higher = more important
    depends_on: list[str] = field(default_factory=list)

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


@dataclass
class JobStatus:
    """Unified job status across all backends."""
    job_id: str
    unified_id: str
    backend: Backend
    state: JobState
    node: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    exit_code: int | None = None
    output: str | None = None
    error: str | None = None

    @property
    def is_running(self) -> bool:
        return self.state == JobState.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        )

    @property
    def runtime_seconds(self) -> float | None:
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


class UnifiedScheduler:
    """Unified scheduler that routes jobs to appropriate backends."""

    def __init__(
        self,
        db_path: str | None = None,
        enable_slurm: bool = True,
        enable_vast: bool = True,
        enable_p2p: bool = True,
    ):
        self.db_path = db_path or self._default_db_path()
        self.enable_slurm = enable_slurm
        self.enable_vast = enable_vast
        self.enable_p2p = enable_p2p

        # Backend instances (lazy loaded)
        self._slurm_backend = None
        self._p2p_backend = None

        # Job tracking
        self._jobs: dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

        # Initialize database
        self._init_db()

    def _default_db_path(self) -> str:
        """Get default database path."""
        base = Path(os.getenv("RINGRIFT_AI_SERVICE_DIR", "."))
        return str(base / "data" / "unified_scheduler.db")

    def _init_db(self) -> None:
        """Initialize SQLite database for job tracking."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    unified_id TEXT PRIMARY KEY,
                    backend_job_id TEXT,
                    backend TEXT,
                    name TEXT,
                    job_type TEXT,
                    state TEXT,
                    target_node TEXT,
                    config TEXT,
                    priority INTEGER,
                    created_at REAL,
                    started_at REAL,
                    finished_at REAL,
                    exit_code INTEGER,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_backend ON jobs(backend)
            """)
            conn.commit()

    @property
    def slurm(self):
        """Lazy-load Slurm backend."""
        if self._slurm_backend is None:
            from app.coordination.slurm_backend import get_slurm_backend
            self._slurm_backend = get_slurm_backend()
        return self._slurm_backend

    def _detect_backend(self, job: UnifiedJob) -> Backend:
        """Detect which backend should handle this job."""
        if job.target_backend != Backend.AUTO:
            return job.target_backend

        # Route based on target node
        if job.target_node:
            if any(fnmatch(job.target_node, p) for p in SLURM_PATTERNS):
                return Backend.SLURM
            if any(fnmatch(job.target_node, p) for p in VAST_PATTERNS):
                return Backend.VAST
            return Backend.P2P

        # Route based on GPU type preference
        if job.target_gpu_type:
            gpu = job.target_gpu_type.lower()
            if gpu in ("gh200", "h100", "a10"):
                return Backend.SLURM
            if gpu in ("rtx5090", "rtx4090", "rtx4080", "a40"):
                return Backend.VAST

        # Default routing based on job type
        if job.job_type == JobType.TRAINING:
            return Backend.SLURM  # Training prefers stable Lambda nodes
        if job.job_type == JobType.GPU_SELFPLAY:
            return Backend.SLURM  # GPU selfplay prefers high-end GPUs

        # Default to P2P for general workloads
        return Backend.P2P

    async def submit(self, job: UnifiedJob) -> str:
        """Submit a job to the appropriate backend.

        Returns the unified job ID.
        """
        backend = self._detect_backend(job)
        logger.info(f"[Scheduler] Submitting job {job.name} to {backend.value}")

        async with self._lock:
            # Record job in database
            self._record_job(job, backend)

            # Submit to appropriate backend
            if backend == Backend.SLURM:
                backend_id = await self._submit_slurm(job)
            elif backend == Backend.VAST:
                backend_id = await self._submit_vast(job)
            else:
                backend_id = await self._submit_p2p(job)

            # Update with backend job ID
            if backend_id:
                self._update_job(job.id, backend_job_id=backend_id, state=JobState.QUEUED)
                logger.info(f"[Scheduler] Job {job.id} submitted as {backend.value}:{backend_id}")
            else:
                self._update_job(job.id, state=JobState.FAILED, error="Submission failed")
                logger.error(f"[Scheduler] Job {job.id} submission failed")

            return job.id

    async def _submit_slurm(self, job: UnifiedJob) -> str | None:
        """Submit job to Slurm backend."""
        from app.coordination.slurm_backend import SlurmJob, SlurmPartition

        # Map job type to partition
        partition_map = {
            JobType.TRAINING: SlurmPartition.GPU_TRAIN,
            JobType.GPU_SELFPLAY: SlurmPartition.GPU_SELFPLAY,
            JobType.SELFPLAY: SlurmPartition.GPU_SELFPLAY,
            JobType.EVALUATION: SlurmPartition.CPU_EVAL,
        }
        partition = partition_map.get(job.job_type, SlurmPartition.GPU_SELFPLAY)

        # Build command from config if not provided
        command = job.command
        if not command and job.config:
            command = self._build_command(job)

        slurm_job = SlurmJob(
            name=job.name,
            partition=partition,
            command=command,
            script_path=job.script_path,
            nodelist=job.target_node,
            cpus_per_task=job.cpus,
            memory_gb=job.memory_gb,
            gpus=job.gpus,
            time_limit=f"{int(job.time_limit_hours)}:00:00",
            env_vars=job.env_vars,
        )

        job_id = await self.slurm.submit_job(slurm_job)
        return str(job_id) if job_id else None

    async def _submit_vast(self, job: UnifiedJob) -> str | None:
        """Submit job to Vast.ai backend."""
        # Build command
        command = job.command or self._build_command(job)
        if not command:
            return None

        # Get target instance
        target = job.target_node
        if not target:
            # Find available Vast instance
            instances = await self._get_vast_instances()
            idle = [i for i in instances if i.get("utilization", 100) < 50]
            if not idle:
                logger.warning("[Vast] No idle instances available")
                return None
            target = idle[0].get("label") or str(idle[0].get("id"))

        # Execute via SSH
        instance_id = target.replace("vast-", "").split("-")[-1]

        try:
            proc = await asyncio.create_subprocess_exec(
                "vastai", "execute", instance_id, command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                # Generate synthetic job ID
                return f"vast-{instance_id}-{int(time.time())}"
            else:
                logger.error(f"[Vast] Execute failed: {stderr.decode()}")
                return None
        except Exception as e:
            logger.error(f"[Vast] Submit error: {e}")
            return None

    async def _submit_p2p(self, job: UnifiedJob) -> str | None:
        """Submit job to P2P backend."""
        command = job.command or self._build_command(job)
        if not command:
            return None

        target = job.target_node
        if not target:
            # Find available P2P node
            # For now, return synthetic ID - real implementation would use P2P API
            logger.warning("[P2P] No target node specified, using local")
            target = "local"

        # Submit via P2P REST API or SSH
        try:
            if target == "local":
                # Run locally
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                return f"local-{proc.pid}"
            else:
                # SSH to node
                ssh_cmd = [
                    "ssh", "-o", "ConnectTimeout=10",
                    "-o", "StrictHostKeyChecking=no",
                    target, f"nohup {command} > /tmp/job_{job.id}.log 2>&1 &"
                ]
                proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if proc.returncode == 0:
                    return f"p2p-{target}-{int(time.time())}"
                return None
        except Exception as e:
            logger.error(f"[P2P] Submit error: {e}")
            return None

    async def _get_vast_instances(self) -> list[dict]:
        """Get list of Vast.ai instances."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "vastai", "show", "instances", "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return json.loads(stdout.decode())
        except Exception as e:
            logger.error(f"[Vast] List instances error: {e}")
        return []

    def _build_command(self, job: UnifiedJob) -> str | None:
        """Build command from job configuration."""
        cfg = job.config

        if job.job_type == JobType.SELFPLAY:
            output_dir = f"data/selfplay/unified_{job.id}"
            return (
                f"python3 scripts/run_self_play_soak.py "
                f"--board-type {cfg.get('board_type', 'square8')} "
                f"--num-players {cfg.get('num_players', 2)} "
                f"--num-games {cfg.get('num_games', 1000)} "
                f"--engine-mode {cfg.get('engine_mode', 'mixed')} "
                f"--difficulty-band canonical "
                f"--record-db {output_dir}/games.db "
                f"--log-jsonl {output_dir}/games.jsonl "
                f"--streaming-record"
            )

        if job.job_type == JobType.GPU_SELFPLAY:
            return (
                f"python3 scripts/run_gpu_selfplay.py "
                f"--board-type {cfg.get('board_type', 'square8')} "
                f"--num-players {cfg.get('num_players', 2)} "
                f"--num-games {cfg.get('num_games', 2000)} "
                f"--output-dir data/selfplay/unified_{job.id}"
            )

        if job.job_type == JobType.TRAINING:
            return (
                f"python3 -m app.training.train "
                f"--data-path {cfg.get('data_path', 'data/training/latest')} "
                f"--epochs {cfg.get('epochs', 100)}"
            )

        return None

    def _record_job(self, job: UnifiedJob, backend: Backend) -> None:
        """Record job in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO jobs (
                    unified_id, backend, name, job_type, state,
                    target_node, config, priority, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id,
                backend.value,
                job.name,
                job.job_type.value,
                JobState.PENDING.value,
                job.target_node,
                json.dumps(job.config),
                job.priority,
                job.created_at,
            ))
            conn.commit()

    def _update_job(
        self,
        unified_id: str,
        backend_job_id: str | None = None,
        state: JobState | None = None,
        error: str | None = None,
    ) -> None:
        """Update job in database."""
        updates = []
        params = []

        if backend_job_id:
            updates.append("backend_job_id = ?")
            params.append(backend_job_id)

        if state:
            updates.append("state = ?")
            params.append(state.value)

        if error:
            updates.append("error = ?")
            params.append(error)

        if not updates:
            return

        params.append(unified_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE unified_id = ?",
                params
            )
            conn.commit()

    async def get_status(self, unified_id: str) -> JobStatus | None:
        """Get status of a job."""
        def _fetch_status() -> JobStatus | None:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM jobs WHERE unified_id = ?",
                    (unified_id,)
                ).fetchone()

                if not row:
                    return None

                return JobStatus(
                    job_id=row["backend_job_id"] or unified_id,
                    unified_id=unified_id,
                    backend=Backend(row["backend"]),
                    state=JobState(row["state"]),
                    node=row["target_node"],
                    start_time=row["started_at"],
                    end_time=row["finished_at"],
                    exit_code=row["exit_code"],
                    error=row["error"],
                )

        return await asyncio.to_thread(_fetch_status)

    async def list_jobs(
        self,
        backend: Backend | None = None,
        state: JobState | None = None,
        limit: int = 100,
    ) -> list[JobStatus]:
        """List jobs with optional filtering."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list[Any] = []

        if backend:
            query += " AND backend = ?"
            params.append(backend.value)

        if state:
            query += " AND state = ?"
            params.append(state.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        def _fetch_jobs() -> list[JobStatus]:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, params).fetchall()

                return [
                    JobStatus(
                        job_id=row["backend_job_id"] or row["unified_id"],
                        unified_id=row["unified_id"],
                        backend=Backend(row["backend"]),
                        state=JobState(row["state"]),
                        node=row["target_node"],
                        start_time=row["started_at"],
                        end_time=row["finished_at"],
                        exit_code=row["exit_code"],
                        error=row["error"],
                    )
                    for row in rows
                ]

        return await asyncio.to_thread(_fetch_jobs)

    async def cancel(self, unified_id: str) -> bool:
        """Cancel a job."""
        status = await self.get_status(unified_id)
        if not status:
            return False

        success = False

        if status.backend == Backend.SLURM:
            try:
                job_id = int(status.job_id)
                success = await self.slurm.cancel_job(job_id)
            except (ValueError, TypeError):
                pass
        elif status.backend == Backend.VAST:
            # Vast cancellation via vastai CLI
            success = await self._cancel_vast_job(status.backend_job_id)
        else:
            # P2P cancellation via P2P backend
            success = await self._cancel_p2p_job(status.backend_job_id)

        if success:
            self._update_job(unified_id, state=JobState.CANCELLED)

        return success

    async def _cancel_vast_job(self, backend_job_id: str | None) -> bool:
        """Cancel a Vast.ai job by stopping the instance.

        Since Vast jobs are SSH commands running on instances, we stop the
        instance to cancel the job. This is more reliable than trying to
        kill specific processes.

        Args:
            backend_job_id: Job ID in format "vast-{instance_id}-{timestamp}"

        Returns:
            True if cancellation successful
        """
        instance_id = self._parse_vast_instance_id(backend_job_id)
        if not instance_id:
            logger.error(f"[Vast] Cannot parse instance ID from {backend_job_id}")
            return False

        try:
            # First try to kill processes gracefully via execute
            proc = await asyncio.create_subprocess_exec(
                "vastai", "execute", instance_id,
                "pkill -TERM -f 'selfplay|training|gauntlet' || true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"[Vast] Cancelled job on instance {instance_id}")
                return True

            # Fallback: stop the instance entirely
            logger.warning(f"[Vast] Execute failed, stopping instance {instance_id}")
            proc = await asyncio.create_subprocess_exec(
                "vastai", "stop", "instance", instance_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                logger.info(f"[Vast] Stopped instance {instance_id}")
                return True

            logger.error(f"[Vast] Failed to cancel job on instance {instance_id}")
            return False

        except Exception as e:
            logger.error(f"[Vast] Cancellation error: {e}")
            return False

    async def _cancel_p2p_job(self, backend_job_id: str | None) -> bool:
        """Cancel a P2P job via the P2P backend.

        Args:
            backend_job_id: Job ID to cancel

        Returns:
            True if cancellation successful
        """
        if not backend_job_id:
            return False

        try:
            backend = await self._get_p2p_backend()
            if not backend:
                logger.warning("[P2P] No P2P backend available for cancellation")
                return False

            result = await backend.cancel_job(backend_job_id)
            success = result.get("success", False)

            if success:
                logger.info(f"[P2P] Cancelled job {backend_job_id}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"[P2P] Cancellation failed: {error}")

            return success

        except Exception as e:
            logger.error(f"[P2P] Cancellation error: {e}")
            return False

    def _map_slurm_state(self, state: SlurmJobState) -> JobState:
        from app.coordination.slurm_backend import SlurmJobState

        if state == SlurmJobState.RUNNING:
            return JobState.RUNNING
        if state == SlurmJobState.PENDING:
            return JobState.QUEUED
        if state == SlurmJobState.COMPLETED:
            return JobState.COMPLETED
        if state == SlurmJobState.CANCELLED:
            return JobState.CANCELLED
        if state in (SlurmJobState.FAILED, SlurmJobState.TIMEOUT, SlurmJobState.NODE_FAIL):
            return JobState.FAILED
        return JobState.UNKNOWN

    def _parse_vast_instance_id(self, backend_job_id: str | None) -> str | None:
        if not backend_job_id:
            return None
        parts = backend_job_id.split("-")
        if len(parts) >= 3 and parts[0] == "vast":
            return parts[1]
        return None

    def _map_vast_instance_state(self, instance: dict[str, Any]) -> JobState:
        raw = str(
            instance.get("cur_state")
            or instance.get("actual_status")
            or instance.get("state")
            or ""
        ).lower()
        if raw in {"running", "active"}:
            return JobState.RUNNING
        if raw in {"pending", "init", "starting", "booting"}:
            return JobState.QUEUED
        if raw in {"stopped", "exited"}:
            return JobState.COMPLETED
        if raw in {"terminated", "deleted"}:
            return JobState.CANCELLED
        if raw in {"failed", "error"}:
            return JobState.FAILED
        return JobState.UNKNOWN

    def _sync_slurm_job_states(
        self,
        slurm_jobs: dict[int, SlurmJobStatus],
        stale_after_seconds: float = 120.0,
    ) -> int:
        """Sync slurm job states into the unified jobs table."""
        if not slurm_jobs:
            return 0

        now = time.time()
        slurm_state_map = {
            str(job_id): self._map_slurm_state(job.state)
            for job_id, job in slurm_jobs.items()
        }
        updates = 0
        terminal_states = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT unified_id, backend_job_id, state, created_at, started_at, finished_at
                FROM jobs
                WHERE backend = ?
                  AND backend_job_id IS NOT NULL
                  AND state IN (?, ?, ?)
                """,
                (
                    Backend.SLURM.value,
                    JobState.PENDING.value,
                    JobState.QUEUED.value,
                    JobState.RUNNING.value,
                ),
            ).fetchall()

            for row in rows:
                backend_id = row["backend_job_id"]
                current_state = row["state"]
                started_at = row["started_at"]
                finished_at = row["finished_at"]
                desired_state = slurm_state_map.get(str(backend_id))

                if desired_state:
                    update_fields = []
                    params = []

                    if desired_state.value != current_state:
                        update_fields.append("state = ?")
                        params.append(desired_state.value)

                    if desired_state == JobState.RUNNING and not started_at:
                        update_fields.append("started_at = ?")
                        params.append(now)

                    if desired_state in terminal_states and not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    if update_fields:
                        params.append(row["unified_id"])
                        conn.execute(
                            f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                            params,
                        )
                        updates += 1
                    continue

                if (
                    desired_state is None
                    and current_state != JobState.UNKNOWN.value
                    and row["created_at"]
                    and (now - float(row["created_at"])) >= stale_after_seconds
                ):
                    update_fields = ["state = ?"]
                    params = [JobState.UNKNOWN.value]

                    if not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    params.append(row["unified_id"])
                    conn.execute(
                        f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                        params,
                    )
                    updates += 1

            conn.commit()

        return updates

    def _sync_vast_job_states(
        self,
        instances: list[dict[str, Any]],
        stale_after_seconds: float = 300.0,
    ) -> int:
        """Sync Vast.ai job states into the unified jobs table."""
        if not instances:
            return 0

        now = time.time()
        instance_map: dict[str, JobState] = {}
        for inst in instances:
            inst_id = inst.get("id")
            if inst_id is None:
                inst_id = inst.get("instance_id")
            if inst_id is None:
                inst_id = inst.get("machine_id")
            if inst_id is None:
                continue
            instance_map[str(inst_id)] = self._map_vast_instance_state(inst)

        updates = 0
        terminal_states = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }
        missing_instance_ids = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT unified_id, backend_job_id, state, created_at, started_at, finished_at
                FROM jobs
                WHERE backend = ?
                  AND backend_job_id IS NOT NULL
                  AND state IN (?, ?, ?)
                """,
                (
                    Backend.VAST.value,
                    JobState.PENDING.value,
                    JobState.QUEUED.value,
                    JobState.RUNNING.value,
                ),
            ).fetchall()

            for row in rows:
                current_state = row["state"]
                started_at = row["started_at"]
                finished_at = row["finished_at"]
                instance_id = self._parse_vast_instance_id(row["backend_job_id"])
                if not instance_id:
                    missing_instance_ids += 1
                    continue

                desired_state = instance_map.get(instance_id)

                if desired_state:
                    update_fields = []
                    params = []

                    if desired_state.value != current_state:
                        update_fields.append("state = ?")
                        params.append(desired_state.value)

                    if desired_state == JobState.RUNNING and not started_at:
                        update_fields.append("started_at = ?")
                        params.append(now)

                    if desired_state in terminal_states and not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    if update_fields:
                        params.append(row["unified_id"])
                        conn.execute(
                            f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                            params,
                        )
                        updates += 1
                    continue

                if (
                    current_state != JobState.UNKNOWN.value
                    and row["created_at"]
                    and (now - float(row["created_at"])) >= stale_after_seconds
                ):
                    update_fields = ["state = ?"]
                    params = [JobState.UNKNOWN.value]

                    if not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    params.append(row["unified_id"])
                    conn.execute(
                        f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                        params,
                    )
                    updates += 1

            conn.commit()

        if missing_instance_ids:
            logger.debug(
                "[Scheduler] Vast jobs missing instance IDs: %s",
                missing_instance_ids,
            )

        return updates

    def _map_p2p_job_state(self, raw_state: str | None) -> JobState:
        state = (raw_state or "").strip().lower()
        if state in {"running", "in_progress", "active"}:
            return JobState.RUNNING
        if state in {"completed", "success", "done"}:
            return JobState.COMPLETED
        if state in {"failed", "error"}:
            return JobState.FAILED
        if state in {"cancelled", "canceled", "aborted"}:
            return JobState.CANCELLED
        if state in {"pending", "queued"}:
            return JobState.QUEUED
        return JobState.UNKNOWN

    def _sync_p2p_job_states(
        self,
        p2p_status: dict[str, Any] | None,
        p2p_history: list[dict[str, Any]] | None,
        stale_after_seconds: float = 300.0,
    ) -> int:
        """Best-effort sync for P2P jobs using leader status/history."""
        if not p2p_status and not p2p_history:
            return 0

        now = time.time()
        updates = 0
        terminal_states = {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }

        job_state_map: dict[str, JobState] = {}
        if p2p_status:
            current = p2p_status.get("current_job", {})
            if isinstance(current, dict):
                job_id = current.get("job_id") or current.get("id")
                status = current.get("status") or current.get("state")
                if job_id:
                    job_state_map[str(job_id)] = self._map_p2p_job_state(status)

        for entry in p2p_history or []:
            if not isinstance(entry, dict):
                continue
            job_id = entry.get("job_id") or entry.get("id")
            status = entry.get("status") or entry.get("state")
            if job_id:
                job_state_map[str(job_id)] = self._map_p2p_job_state(status)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT unified_id, backend_job_id, state, created_at, started_at, finished_at
                FROM jobs
                WHERE backend = ?
                  AND backend_job_id IS NOT NULL
                  AND state IN (?, ?, ?)
                """,
                (
                    Backend.P2P.value,
                    JobState.PENDING.value,
                    JobState.QUEUED.value,
                    JobState.RUNNING.value,
                ),
            ).fetchall()

            for row in rows:
                backend_job_id = row["backend_job_id"]
                current_state = row["state"]
                started_at = row["started_at"]
                finished_at = row["finished_at"]
                desired_state = job_state_map.get(str(backend_job_id))

                if desired_state:
                    update_fields = []
                    params = []

                    if desired_state.value != current_state:
                        update_fields.append("state = ?")
                        params.append(desired_state.value)

                    if desired_state == JobState.RUNNING and not started_at:
                        update_fields.append("started_at = ?")
                        params.append(now)

                    if desired_state in terminal_states and not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    if update_fields:
                        params.append(row["unified_id"])
                        conn.execute(
                            f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                            params,
                        )
                        updates += 1
                    continue

                if (
                    current_state != JobState.UNKNOWN.value
                    and row["created_at"]
                    and (now - float(row["created_at"])) >= stale_after_seconds
                ):
                    update_fields = ["state = ?"]
                    params = [JobState.UNKNOWN.value]

                    if not finished_at:
                        update_fields.append("finished_at = ?")
                        params.append(now)

                    params.append(row["unified_id"])
                    conn.execute(
                        f"UPDATE jobs SET {', '.join(update_fields)} WHERE unified_id = ?",
                        params,
                    )
                    updates += 1

            conn.commit()

        return updates

    async def _get_p2p_backend(self) -> P2PBackend | None:
        try:
            from app.coordination.p2p_backend import (
                HAS_AIOHTTP,
                get_p2p_backend_with_registry,
                get_p2p_leader_from_registry,
            )
        except (ImportError, ModuleNotFoundError):
            return None

        if not HAS_AIOHTTP:
            return None

        leader_url = (
            os.environ.get("P2P_LEADER")
            or os.environ.get("RINGRIFT_P2P_LEADER_URL")
            or get_p2p_leader_from_registry()
        )
        seed_env = os.environ.get("RINGRIFT_P2P_SEEDS", "")
        seed_urls = [s.strip() for s in seed_env.split(",") if s.strip()]

        if not leader_url and not seed_urls:
            return None

        try:
            backend = await get_p2p_backend_with_registry(
                seed_urls=seed_urls or None,
                leader_url=leader_url,
                auth_token=os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", ""),
                use_registry=True,
            )
            backend.timeout = 5.0
            return backend
        except Exception as exc:
            logger.debug("[Scheduler] P2P backend unavailable: %s", exc)
            return None

    async def sync_job_states(
        self,
        slurm_jobs: dict[int, SlurmJobStatus] | None = None,
        vast_instances: list[dict[str, Any]] | None = None,
        p2p_status: dict[str, Any] | None = None,
        p2p_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Sync backend job states into the unified jobs table."""
        updates = {"slurm": 0, "vast": 0, "p2p": 0}

        if self.enable_slurm:
            if slurm_jobs is None:
                slurm_jobs = await self.slurm.get_jobs(refresh=True)
            updates["slurm"] = self._sync_slurm_job_states(slurm_jobs)

        if self.enable_vast:
            if vast_instances is None:
                vast_instances = await self._get_vast_instances()
            updates["vast"] = self._sync_vast_job_states(vast_instances)

        if self.enable_p2p:
            if p2p_status is None or p2p_history is None:
                backend = await self._get_p2p_backend()
                if backend:
                    try:
                        if p2p_status is None:
                            p2p_status = await backend.get_cluster_status()
                        if p2p_history is None:
                            p2p_history = await backend.get_job_history(limit=50)
                    finally:
                        await backend.close()
            updates["p2p"] = self._sync_p2p_job_states(p2p_status, p2p_history)

        return updates

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get overall cluster status across all backends."""
        status = {
            "slurm": {"enabled": self.enable_slurm, "nodes": 0, "jobs": 0},
            "vast": {"enabled": self.enable_vast, "instances": 0},
            "p2p": {"enabled": self.enable_p2p, "nodes": 0},
            "jobs": {"total": 0, "running": 0, "pending": 0},
        }

        # Slurm status
        slurm_jobs = None
        if self.enable_slurm:
            from app.coordination.slurm_backend import SlurmJobState

            try:
                nodes = await self.slurm.get_nodes(refresh=True)
                jobs = await self.slurm.get_jobs(refresh=True)
                slurm_jobs = jobs
                status["slurm"]["nodes"] = len(nodes)
                status["slurm"]["jobs"] = len(jobs)
                status["slurm"]["idle_nodes"] = sum(1 for n in nodes.values() if n.is_idle)
                status["slurm"]["jobs_running"] = sum(1 for j in jobs.values() if j.is_running)
                status["slurm"]["jobs_pending"] = sum(
                    1 for j in jobs.values() if j.state == SlurmJobState.PENDING
                )
            except Exception as e:
                status["slurm"]["error"] = str(e)

        # Vast status
        vast_instances = None
        if self.enable_vast:
            try:
                instances = await self._get_vast_instances()
                vast_instances = instances
                status["vast"]["instances"] = len(instances)
                status["vast"]["running"] = sum(
                    1
                    for i in instances
                    if (i.get("cur_state") or i.get("actual_status")) == "running"
                )
                status["vast"]["jobs_running"] = status["vast"]["running"]
            except Exception as e:
                status["vast"]["error"] = str(e)

        # P2P status
        p2p_status = None
        p2p_history = None
        if self.enable_p2p:
            backend = await self._get_p2p_backend()
            if backend:
                try:
                    p2p_status = await backend.get_cluster_status()
                    nodes = p2p_status.get("nodes", []) if isinstance(p2p_status, dict) else []
                    status["p2p"]["nodes"] = len(nodes)
                    status["p2p"]["selfplay_running"] = sum(
                        int(n.get("selfplay_jobs", 0) or 0) for n in nodes
                    )
                    status["p2p"]["training_running"] = sum(
                        int(n.get("training_jobs", 0) or 0) for n in nodes
                    )
                    status["p2p"]["jobs_running"] = (
                        status["p2p"]["selfplay_running"]
                        + status["p2p"]["training_running"]
                    )
                    p2p_history = await backend.get_job_history(limit=50)
                except Exception as e:
                    status["p2p"]["error"] = str(e)
                finally:
                    await backend.close()

        # Keep unified job states aligned with backend status.
        await self.sync_job_states(
            slurm_jobs=slurm_jobs,
            vast_instances=vast_instances,
            p2p_status=p2p_status,
            p2p_history=p2p_history,
        )

        # Job counts from database
        def _fetch_job_counts() -> tuple[int, int, int]:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) as running,
                        SUM(CASE WHEN state = 'pending' OR state = 'queued' THEN 1 ELSE 0 END) as pending
                    FROM jobs
                """).fetchone()
                if row:
                    return (row[0], row[1] or 0, row[2] or 0)
                return (0, 0, 0)

        total, running, pending = await asyncio.to_thread(_fetch_job_counts)
        status["jobs"]["total"] = total
        status["jobs"]["running"] = running
        status["jobs"]["pending"] = pending

        return status

    def health_check(self) -> "HealthCheckResult":
        """Check scheduler health for DaemonManager integration.

        Returns:
            HealthCheckResult with status and metrics
        """
        from app.coordination.contracts import HealthCheckResult
        from app.coordination.protocols import CoordinatorStatus

        try:
            # Count jobs by state
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) as running,
                        SUM(CASE WHEN state = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM jobs
                    WHERE created_at > ?
                """, (time.time() - 3600,)).fetchone()  # Last hour

                total = row[0] if row else 0
                running = row[1] or 0 if row else 0
                failed = row[2] or 0 if row else 0

            # Check if backends are available
            backends_available = []
            if self.enable_slurm:
                backends_available.append("slurm")
            if self.enable_vast:
                backends_available.append("vast")
            if self.enable_p2p:
                backends_available.append("p2p")

            # Health is good if we have at least one backend and failure rate < 20%
            failure_rate = failed / total if total > 0 else 0.0
            is_healthy = len(backends_available) > 0 and failure_rate < 0.2

            return HealthCheckResult(
                healthy=is_healthy,
                status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.DEGRADED,
                message="" if is_healthy else f"High failure rate: {failure_rate:.1%}",
                details={
                    "backends": backends_available,
                    "jobs_last_hour": total,
                    "running": running,
                    "failed": failed,
                    "failure_rate": round(failure_rate, 3),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )


# Singleton instance
_scheduler: UnifiedScheduler | None = None


def get_scheduler() -> UnifiedScheduler:
    """Get the unified scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = UnifiedScheduler()
    return _scheduler


# Convenience functions

async def submit_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 1000,
    target_node: str | None = None,
    engine_mode: str = "mixed",
) -> str:
    """Submit a selfplay job."""
    scheduler = get_scheduler()
    job = UnifiedJob(
        name=f"selfplay-{board_type}-{num_players}p",
        job_type=JobType.SELFPLAY,
        target_node=target_node,
        config={
            "board_type": board_type,
            "num_players": num_players,
            "num_games": num_games,
            "engine_mode": engine_mode,
        },
    )
    return await scheduler.submit(job)


async def submit_gpu_selfplay(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 2000,
    target_node: str | None = None,
) -> str:
    """Submit a GPU selfplay job."""
    scheduler = get_scheduler()
    job = UnifiedJob(
        name=f"gpu-selfplay-{board_type}-{num_players}p",
        job_type=JobType.GPU_SELFPLAY,
        target_node=target_node,
        config={
            "board_type": board_type,
            "num_players": num_players,
            "num_games": num_games,
        },
    )
    return await scheduler.submit(job)


async def submit_training(
    data_path: str,
    model_name: str | None = None,
    epochs: int = 100,
    target_node: str | None = None,
) -> str:
    """Submit a training job."""
    scheduler = get_scheduler()
    job = UnifiedJob(
        name=f"train-{model_name or 'latest'}",
        job_type=JobType.TRAINING,
        target_node=target_node,
        time_limit_hours=24.0,
        config={
            "data_path": data_path,
            "model_name": model_name,
            "epochs": epochs,
        },
    )
    return await scheduler.submit(job)
