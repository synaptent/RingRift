"""Slurm Backend for Unified Cluster Orchestration.

Provides Slurm integration for Lambda Labs nodes, enabling:
1. Job submission via sbatch
2. Queue monitoring via squeue
3. Node status via sinfo
4. Job management (cancel, status)

This backend complements the P2P orchestrator by handling Lambda nodes
through Slurm while P2P handles dynamic Vast.ai/Hetzner/AWS nodes.

Usage:
    from app.coordination.slurm_backend import (
        SlurmBackend,
        get_slurm_backend,
        SlurmJob,
        SlurmPartition,
    )

    backend = get_slurm_backend()
    job_id = await backend.submit_job(SlurmJob(
        name="gpu-selfplay",
        partition=SlurmPartition.GPU_SELFPLAY,
        script_path="/path/to/script.sh",
        nodes=1,
        cpus_per_task=16,
        memory_gb=64,
    ))
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path

__all__ = [
    # Backend
    "SlurmBackend",
    # Dataclasses
    "SlurmJob",
    # Enums
    "SlurmJobState",
    "SlurmJobStatus",
    "SlurmNode",
    "SlurmPartition",
    # Singleton
    "get_slurm_backend",
    # Convenience functions
    "submit_gpu_selfplay_job",
    "submit_selfplay_job",
    "submit_training_job",
    # Constants
    "SLURM_MANAGED_PATTERNS",
]

logger = logging.getLogger(__name__)


class SlurmPartition(str, Enum):
    """Available Slurm partitions in the RingRift cluster."""
    GPU_TRAIN = "gpu-train"
    GPU_SELFPLAY = "gpu-selfplay"
    CPU_EVAL = "cpu-eval"
    GPU_GH200 = "gpu-gh200"


class SlurmJobState(str, Enum):
    """Slurm job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass
class SlurmNode:
    """Represents a Slurm cluster node."""
    name: str
    partition: str
    state: str
    cpus: int
    memory_mb: int
    features: list[str] = field(default_factory=list)
    gres: str = ""  # GPU resources

    @property
    def is_idle(self) -> bool:
        return self.state.lower() in ("idle", "mix")

    @property
    def is_allocated(self) -> bool:
        return "alloc" in self.state.lower()

    @property
    def gpu_type(self) -> str:
        """Extract GPU type from features or gres."""
        for f in self.features:
            if f in ("gh200", "h100", "a10", "a40"):
                return f
        return "unknown"


@dataclass
class SlurmJob:
    """Job specification for Slurm submission."""
    name: str
    partition: SlurmPartition = SlurmPartition.GPU_SELFPLAY

    # Resource requirements
    nodes: int = 1
    cpus_per_task: int = 16
    memory_gb: int = 64
    gpus: int = 1
    time_limit: str = "8:00:00"

    # Script configuration
    script_path: str | None = None
    command: str | None = None  # Alternative to script_path
    working_dir: str = "/home/ubuntu/ringrift/ai-service"  # Use local storage, not NFS

    # Environment
    env_vars: dict[str, str] = field(default_factory=dict)

    # Targeting
    nodelist: str | None = None  # Specific node(s) to run on
    exclude: str | None = None   # Nodes to exclude
    features: str | None = None  # Required features (e.g., "gh200")

    # Job control
    dependency: str | None = None  # Job dependencies
    array: str | None = None       # Array job specification

    def to_sbatch_args(self) -> list[str]:
        """Convert to sbatch command-line arguments."""
        args = [
            f"--job-name={self.name}",
            f"--partition={self.partition.value}",
            f"--nodes={self.nodes}",
            f"--cpus-per-task={self.cpus_per_task}",
            f"--mem={self.memory_gb}G",
            f"--time={self.time_limit}",
            f"--chdir={self.working_dir}",
        ]

        if self.gpus > 0:
            args.append(f"--gres=gpu:{self.gpus}")

        if self.nodelist:
            args.append(f"--nodelist={self.nodelist}")

        if self.exclude:
            args.append(f"--exclude={self.exclude}")

        if self.features:
            args.append(f"--constraint={self.features}")

        if self.dependency:
            args.append(f"--dependency={self.dependency}")

        if self.array:
            args.append(f"--array={self.array}")

        return args


@dataclass
class SlurmJobStatus:
    """Status of a submitted Slurm job."""
    job_id: int
    name: str
    state: SlurmJobState
    partition: str
    node: str | None
    start_time: str | None
    run_time: str | None
    exit_code: int | None = None

    @property
    def is_running(self) -> bool:
        return self.state == SlurmJobState.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.state in (
            SlurmJobState.COMPLETED,
            SlurmJobState.FAILED,
            SlurmJobState.CANCELLED,
            SlurmJobState.TIMEOUT,
        )


# Lambda nodes that should be managed by Slurm
SLURM_MANAGED_PATTERNS = [
    "lambda-gh200-*",
    "lambda-h100*",
    "lambda-2xh100",
    "lambda-a10",
]

# Slurm controller configuration
SLURM_CONTROLLER_HOST = os.getenv("SLURM_CONTROLLER_HOST", "100.78.101.123")
SLURM_CONTROLLER_USER = os.getenv("SLURM_CONTROLLER_USER", "ubuntu")
SLURM_NFS_BASE = os.getenv("SLURM_NFS_BASE", "/home/ubuntu/ringrift/ai-service")  # Use local storage


class SlurmBackend:
    """Backend for managing Slurm jobs on the Lambda cluster."""

    def __init__(
        self,
        controller_host: str = SLURM_CONTROLLER_HOST,
        controller_user: str = SLURM_CONTROLLER_USER,
        nfs_base: str = SLURM_NFS_BASE,
        ssh_timeout: int = 10,
    ):
        self.controller_host = controller_host
        self.controller_user = controller_user
        self.nfs_base = Path(nfs_base)
        self.ssh_timeout = ssh_timeout

        # Cached state
        self._nodes: dict[str, SlurmNode] = {}
        self._jobs: dict[int, SlurmJobStatus] = {}
        self._last_refresh = 0
        self._refresh_interval = 30  # seconds

    def is_slurm_node(self, hostname: str) -> bool:
        """Check if a hostname should be managed by Slurm."""
        return any(fnmatch(hostname, pattern) for pattern in SLURM_MANAGED_PATTERNS)

    async def _ssh_command(self, cmd: str) -> tuple[int, str, str]:
        """Execute command on Slurm controller via SSH."""
        ssh_cmd = [
            "ssh",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            f"{self.controller_user}@{self.controller_host}",
            cmd,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.ssh_timeout + 5
            )
            return (
                proc.returncode or 0,
                stdout.decode().strip(),
                stderr.decode().strip(),
            )
        except asyncio.TimeoutError:
            logger.error(f"[Slurm] SSH timeout to {self.controller_host}")
            return (-1, "", "SSH timeout")
        except Exception as e:
            logger.error(f"[Slurm] SSH error: {e}")
            return (-1, "", str(e))

    async def refresh_state(self) -> None:
        """Refresh node and job state from Slurm."""
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return

        await asyncio.gather(
            self._refresh_nodes(),
            self._refresh_jobs(),
            return_exceptions=True,
        )
        self._last_refresh = now

    async def _refresh_nodes(self) -> None:
        """Refresh node information from sinfo."""
        # sinfo with custom format for easy parsing
        cmd = 'sinfo -N -h -o "%N|%P|%T|%c|%m|%f|%G"'
        rc, stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] sinfo failed: {stderr}")
            return

        nodes = {}
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                parts = line.split("|")
                if len(parts) >= 6:
                    name, partition, state, cpus, mem, features = parts[:6]
                    gres = parts[6] if len(parts) > 6 else ""

                    # Parse memory (comes as "123456" in MB)
                    mem_mb = int(mem) if mem.isdigit() else 0

                    node = SlurmNode(
                        name=name,
                        partition=partition,
                        state=state,
                        cpus=int(cpus) if cpus.isdigit() else 0,
                        memory_mb=mem_mb,
                        features=features.split(",") if features else [],
                        gres=gres,
                    )
                    nodes[name] = node
            except Exception as e:
                logger.warning(f"[Slurm] Failed to parse sinfo line: {line}: {e}")

        self._nodes = nodes
        logger.info(f"[Slurm] Refreshed {len(nodes)} nodes")

    async def _refresh_jobs(self) -> None:
        """Refresh job queue from squeue."""
        cmd = 'squeue -h -o "%i|%j|%T|%P|%N|%S|%M"'
        rc, stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] squeue failed: {stderr}")
            return

        jobs = {}
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                parts = line.split("|")
                if len(parts) >= 5:
                    job_id = int(parts[0])
                    jobs[job_id] = SlurmJobStatus(
                        job_id=job_id,
                        name=parts[1],
                        state=SlurmJobState(parts[2]) if parts[2] in SlurmJobState.__members__ else SlurmJobState.UNKNOWN,
                        partition=parts[3],
                        node=parts[4] if parts[4] else None,
                        start_time=parts[5] if len(parts) > 5 else None,
                        run_time=parts[6] if len(parts) > 6 else None,
                    )
            except Exception as e:
                logger.warning(f"[Slurm] Failed to parse squeue line: {line}: {e}")

        self._jobs = jobs
        logger.info(f"[Slurm] Refreshed {len(jobs)} jobs in queue")

    async def get_completed_job_status(self, job_id: int) -> SlurmJobStatus | None:
        """Get status of a completed job including exit code via sacct.

        Use this to get final job status after job has left the queue.
        Returns exit code and final state for jobs that have completed.
        """
        # sacct format: JobID|JobName|State|Partition|NodeList|Start|Elapsed|ExitCode
        cmd = f'sacct -j {job_id} -n -P -o "JobID,JobName,State,Partition,NodeList,Start,Elapsed,ExitCode" | head -1'
        rc, stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] sacct failed for job {job_id}: {stderr}")
            return None

        if not stdout.strip():
            logger.warning(f"[Slurm] No sacct data for job {job_id}")
            return None

        try:
            parts = stdout.strip().split("|")
            if len(parts) >= 7:
                # Parse exit code - format is "exitcode:signal" e.g. "0:0" or "1:0"
                exit_code_str = parts[7] if len(parts) > 7 else "0:0"
                exit_code = self._parse_exit_code(exit_code_str)

                # Map sacct state to SlurmJobState
                state_str = parts[2].upper()
                state = self._map_sacct_state(state_str)

                return SlurmJobStatus(
                    job_id=job_id,
                    name=parts[1],
                    state=state,
                    partition=parts[3],
                    node=parts[4] if parts[4] else None,
                    start_time=parts[5] if parts[5] else None,
                    run_time=parts[6] if parts[6] else None,
                    exit_code=exit_code,
                )
        except Exception as e:
            logger.warning(f"[Slurm] Failed to parse sacct output for job {job_id}: {e}")

        return None

    def _parse_exit_code(self, exit_code_str: str) -> int:
        """Parse Slurm exit code string (format: 'exitcode:signal')."""
        try:
            if ":" in exit_code_str:
                exit_code, signal = exit_code_str.split(":", 1)
                # If killed by signal, return 128 + signal (Unix convention)
                sig = int(signal) if signal.isdigit() else 0
                if sig > 0:
                    return 128 + sig
                return int(exit_code) if exit_code.isdigit() else 0
            return int(exit_code_str) if exit_code_str.isdigit() else 0
        except (ValueError, AttributeError):
            return 0

    def _map_sacct_state(self, state_str: str) -> SlurmJobState:
        """Map sacct state string to SlurmJobState enum."""
        # sacct states can include suffixes like "COMPLETED+" or "CANCELLED by ..."
        state_upper = state_str.split()[0].rstrip("+")

        state_map = {
            "PENDING": SlurmJobState.PENDING,
            "RUNNING": SlurmJobState.RUNNING,
            "COMPLETED": SlurmJobState.COMPLETED,
            "FAILED": SlurmJobState.FAILED,
            "CANCELLED": SlurmJobState.CANCELLED,
            "TIMEOUT": SlurmJobState.TIMEOUT,
            "NODE_FAIL": SlurmJobState.NODE_FAIL,
            "PREEMPTED": SlurmJobState.CANCELLED,
            "OUT_OF_MEMORY": SlurmJobState.FAILED,
            "DEADLINE": SlurmJobState.TIMEOUT,
        }
        return state_map.get(state_upper, SlurmJobState.UNKNOWN)

    async def get_recent_completed_jobs(
        self,
        since_hours: int = 24,
        job_name_filter: str | None = None,
    ) -> list[SlurmJobStatus]:
        """Get recently completed jobs with exit codes.

        Args:
            since_hours: Look back this many hours for completed jobs.
            job_name_filter: Optional job name prefix to filter by.

        Returns:
            List of completed job statuses with exit codes.
        """
        # Calculate start time
        start_time = f"now-{since_hours}hours"
        cmd = f'sacct -S {start_time} -n -P -o "JobID,JobName,State,Partition,NodeList,Start,Elapsed,ExitCode"'

        if job_name_filter:
            cmd += f' -n "{job_name_filter}"'

        rc, stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] sacct failed: {stderr}")
            return []

        jobs = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                parts = line.split("|")
                # Skip batch/extern sub-steps (e.g., "12345.batch", "12345.extern")
                job_id_str = parts[0]
                if "." in job_id_str:
                    continue

                job_id = int(job_id_str)
                exit_code = self._parse_exit_code(parts[7] if len(parts) > 7 else "0:0")
                state = self._map_sacct_state(parts[2])

                jobs.append(SlurmJobStatus(
                    job_id=job_id,
                    name=parts[1],
                    state=state,
                    partition=parts[3],
                    node=parts[4] if parts[4] else None,
                    start_time=parts[5] if len(parts) > 5 else None,
                    run_time=parts[6] if len(parts) > 6 else None,
                    exit_code=exit_code,
                ))
            except Exception as e:
                logger.warning(f"[Slurm] Failed to parse sacct line: {line}: {e}")

        logger.info(f"[Slurm] Found {len(jobs)} completed jobs in last {since_hours}h")
        return jobs

    async def get_nodes(self, refresh: bool = False) -> dict[str, SlurmNode]:
        """Get all Slurm nodes."""
        if refresh or not self._nodes:
            await self._refresh_nodes()
        return self._nodes

    async def get_idle_nodes(
        self,
        partition: SlurmPartition | None = None,
        gpu_type: str | None = None,
    ) -> list[SlurmNode]:
        """Get idle nodes optionally filtered by partition and GPU type."""
        await self.refresh_state()

        idle = []
        for node in self._nodes.values():
            if not node.is_idle:
                continue
            if partition and partition.value not in node.partition:
                continue
            if gpu_type and node.gpu_type != gpu_type:
                continue
            idle.append(node)

        return idle

    async def get_jobs(self, refresh: bool = False) -> dict[int, SlurmJobStatus]:
        """Get all jobs in the queue."""
        if refresh or not self._jobs:
            await self._refresh_jobs()
        return self._jobs

    async def submit_job(self, job: SlurmJob) -> int | None:
        """Submit a job to Slurm and return the job ID."""
        # Generate script if command provided
        if job.command and not job.script_path:
            script_content = self._generate_script(job)
            script_path = await self._upload_script(job.name, script_content)
            if not script_path:
                return None
            job.script_path = script_path

        if not job.script_path:
            logger.error("[Slurm] No script_path or command provided")
            return None

        # Build sbatch command
        args = job.to_sbatch_args()
        cmd = f"sbatch --parsable {' '.join(args)} {job.script_path}"

        rc, stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] sbatch failed: {stderr}")
            return None

        try:
            job_id = int(stdout.split(";")[0])  # Handle array jobs
            logger.info(f"[Slurm] Submitted job {job_id}: {job.name}")
            return job_id
        except ValueError:
            logger.error(f"[Slurm] Failed to parse job ID from: {stdout}")
            return None

    def _generate_script(self, job: SlurmJob) -> str:
        """Generate a batch script from job specification."""
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"cd {job.working_dir}",
            "",
            "# Architecture-aware venv activation",
            'ARCH="$(uname -m)"',
            'if [ "$ARCH" = "aarch64" ]; then',
            "  source /home/ubuntu/venv-arm64-local/bin/activate",
            "else",
            f"  source {job.working_dir}/venv/bin/activate",
            "fi",
            "",
            f"export PYTHONPATH={job.working_dir}",
        ]

        # Add environment variables
        for key, value in job.env_vars.items():
            lines.append(f"export {key}={value}")

        lines.append("")
        lines.append("# Execute command")

        if job.command:
            lines.append(job.command)

        return "\n".join(lines) + "\n"

    async def _upload_script(self, name: str, content: str) -> str | None:
        """Upload script to NFS and return path."""
        # Generate unique filename
        timestamp = int(time.time())
        script_name = f"{name}.{timestamp:x}.sh"
        script_path = f"{self.nfs_base}/data/slurm/jobs/{script_name}"

        # Create script via SSH using heredoc (no escaping needed)
        cmd = f"mkdir -p {self.nfs_base}/data/slurm/jobs && cat > {script_path} << 'SLURM_SCRIPT_EOF'\n{content}SLURM_SCRIPT_EOF\nchmod +x {script_path}"

        rc, _stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] Failed to upload script: {stderr}")
            return None

        return script_path

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a Slurm job."""
        cmd = f"scancel {job_id}"
        rc, _stdout, stderr = await self._ssh_command(cmd)

        if rc != 0:
            logger.error(f"[Slurm] scancel failed: {stderr}")
            return False

        logger.info(f"[Slurm] Cancelled job {job_id}")
        return True

    async def get_job_status(self, job_id: int, include_completed: bool = True) -> SlurmJobStatus | None:
        """Get status of a specific job.

        Args:
            job_id: The Slurm job ID.
            include_completed: If True and job is not in queue, check sacct
                for completed job status including exit code.

        Returns:
            Job status or None if not found.
        """
        await self._refresh_jobs()
        status = self._jobs.get(job_id)

        if status is not None:
            return status

        # Job not in queue - check sacct for completed status
        if include_completed:
            return await self.get_completed_job_status(job_id)

        return None

    async def wait_for_job(
        self,
        job_id: int,
        timeout: float = 3600,
        poll_interval: float = 10,
    ) -> SlurmJobStatus | None:
        """Wait for a job to complete.

        Returns job status with exit code when job finishes.
        """
        start = time.time()
        while time.time() - start < timeout:
            # First check if job is still in queue
            await self._refresh_jobs()
            status = self._jobs.get(job_id)

            if status and status.is_finished:
                # Job finished but may still be in queue briefly - get exit code from sacct
                completed_status = await self.get_completed_job_status(job_id)
                return completed_status or status

            if status is None:
                # Job left queue - get final status from sacct
                completed_status = await self.get_completed_job_status(job_id)
                if completed_status:
                    return completed_status

            await asyncio.sleep(poll_interval)

        return None


# Singleton instance
_slurm_backend: SlurmBackend | None = None


def get_slurm_backend() -> SlurmBackend:
    """Get the Slurm backend singleton."""
    global _slurm_backend
    if _slurm_backend is None:
        _slurm_backend = SlurmBackend()
    return _slurm_backend


# Convenience functions for common job types

async def submit_selfplay_job(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 1000,
    engine_mode: str = "mixed",
    target_node: str | None = None,
    partition: SlurmPartition = SlurmPartition.GPU_SELFPLAY,
) -> int | None:
    """Submit a selfplay job to Slurm."""
    backend = get_slurm_backend()

    timestamp = int(time.time())
    output_dir = f"data/selfplay/slurm_{board_type}_{num_players}p_{timestamp}"

    command = (
        f"python3 scripts/run_self_play_soak.py "
        f"--num-games {num_games} "
        f"--board-type {board_type} "
        f"--num-players {num_players} "
        f"--engine-mode {engine_mode} "
        f"--difficulty-band canonical "
        f"--record-db {output_dir}/games.db "
        f"--log-jsonl {output_dir}/games.jsonl "
        f"--streaming-record"
    )

    job = SlurmJob(
        name=f"selfplay-{board_type}-{num_players}p",
        partition=partition,
        command=command,
        nodelist=target_node,
        cpus_per_task=16,
        memory_gb=64,
        env_vars={
            "RINGRIFT_STRICT_NO_MOVE_INVARIANT": "1",
            "RINGRIFT_PARITY_VALIDATION": "off",
            "OMP_NUM_THREADS": "1",
        },
    )

    return await backend.submit_job(job)


async def submit_gpu_selfplay_job(
    board_type: str = "square8",
    num_players: int = 2,
    num_games: int = 2000,
    target_node: str | None = None,
    partition: SlurmPartition = SlurmPartition.GPU_SELFPLAY,
) -> int | None:
    """Submit a GPU-accelerated selfplay job to Slurm."""
    backend = get_slurm_backend()

    timestamp = int(time.time())
    output_dir = f"data/selfplay/slurm_gpu_{board_type}_{num_players}p_{timestamp}"

    command = (
        f"python3 scripts/run_gpu_selfplay.py "
        f"--board-type {board_type} "
        f"--num-players {num_players} "
        f"--num-games {num_games} "
        f"--output-dir {output_dir}"
    )

    job = SlurmJob(
        name=f"gpu-selfplay-{board_type}-{num_players}p",
        partition=partition,
        command=command,
        nodelist=target_node,
        cpus_per_task=16,
        memory_gb=64,
        gpus=1,
    )

    return await backend.submit_job(job)


async def submit_training_job(
    data_path: str,
    model_name: str,
    epochs: int = 100,
    target_node: str | None = None,
) -> int | None:
    """Submit a training job to Slurm."""
    backend = get_slurm_backend()

    command = (
        f"python3 -m app.training.train "
        f"--data-path {data_path} "
        f"--model-name {model_name} "
        f"--epochs {epochs} "
        f"--wandb"
    )

    job = SlurmJob(
        name=f"train-{model_name}",
        partition=SlurmPartition.GPU_TRAIN,
        command=command,
        nodelist=target_node,
        cpus_per_task=32,
        memory_gb=128,
        gpus=1,
        time_limit="24:00:00",
    )

    return await backend.submit_job(job)
