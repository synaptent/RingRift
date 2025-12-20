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
        target_node="lambda-gh200-f",
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
from typing import Any

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
SLURM_PATTERNS = [
    "lambda-gh200-*",
    "lambda-h100*",
    "lambda-2xh100",
    "lambda-a10",
]

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
            return (
                f"python3 scripts/run_self_play_soak.py "
                f"--board-type {cfg.get('board_type', 'square8')} "
                f"--num-players {cfg.get('num_players', 2)} "
                f"--num-games {cfg.get('num_games', 1000)} "
                f"--engine-mode {cfg.get('engine_mode', 'mixed')} "
                f"--difficulty-band canonical "
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

    async def list_jobs(
        self,
        backend: Backend | None = None,
        state: JobState | None = None,
        limit: int = 100,
    ) -> list[JobStatus]:
        """List jobs with optional filtering."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []

        if backend:
            query += " AND backend = ?"
            params.append(backend.value)

        if state:
            query += " AND state = ?"
            params.append(state.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

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
            # Vast cancellation via API
            logger.warning("[Vast] Job cancellation not implemented")
        else:
            # P2P cancellation
            logger.warning("[P2P] Job cancellation not implemented")

        if success:
            self._update_job(unified_id, state=JobState.CANCELLED)

        return success

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get overall cluster status across all backends."""
        status = {
            "slurm": {"enabled": self.enable_slurm, "nodes": 0, "jobs": 0},
            "vast": {"enabled": self.enable_vast, "instances": 0},
            "p2p": {"enabled": self.enable_p2p, "nodes": 0},
            "jobs": {"total": 0, "running": 0, "pending": 0},
        }

        # Slurm status
        if self.enable_slurm:
            try:
                nodes = await self.slurm.get_nodes(refresh=True)
                jobs = await self.slurm.get_jobs(refresh=True)
                status["slurm"]["nodes"] = len(nodes)
                status["slurm"]["jobs"] = len(jobs)
                status["slurm"]["idle_nodes"] = sum(1 for n in nodes.values() if n.is_idle)
            except Exception as e:
                status["slurm"]["error"] = str(e)

        # Vast status
        if self.enable_vast:
            try:
                instances = await self._get_vast_instances()
                status["vast"]["instances"] = len(instances)
                status["vast"]["running"] = sum(1 for i in instances if i.get("cur_state") == "running")
            except Exception as e:
                status["vast"]["error"] = str(e)

        # Job counts from database
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) as running,
                    SUM(CASE WHEN state = 'pending' OR state = 'queued' THEN 1 ELSE 0 END) as pending
                FROM jobs
            """).fetchone()
            if row:
                status["jobs"]["total"] = row[0]
                status["jobs"]["running"] = row[1] or 0
                status["jobs"]["pending"] = row[2] or 0

        return status


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
