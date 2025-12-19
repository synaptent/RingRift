"""Orchestrator backend abstraction for different execution strategies.

This module provides a unified interface for running distributed workloads
across different backends:
- LocalBackend: Execute on the local machine
- SSHBackend: Execute via SSH on remote hosts
- P2PBackend: Execute via P2P orchestrator REST API
- SlurmBackend: Execute via Slurm on HPC clusters

All orchestrators should use this abstraction instead of implementing
their own execution logic.

Usage:
    from app.execution.backends import get_backend, BackendType

    # Get configured backend (from config)
    backend = get_backend()

    # Or explicitly choose backend type
    backend = get_backend(BackendType.SSH)

    # Run selfplay on available workers
    results = await backend.run_selfplay(
        games=100,
        board_type="square8",
        num_players=2,
    )

    # Run tournament
    results = await backend.run_tournament(
        agent_ids=["random", "heuristic"],
        games_per_pairing=20,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.execution.executor import (
    ExecutionResult,
    LocalExecutor,
    SSHExecutor,
    ExecutorPool,
)

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Available backend types."""
    LOCAL = "local"
    SSH = "ssh"
    P2P = "p2p"
    SLURM = "slurm"
    HYBRID = "hybrid"  # Local + SSH fallback


@dataclass
class WorkerStatus:
    """Status of a worker node."""
    name: str
    available: bool
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_jobs: int = 0
    last_seen: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of a distributed job."""
    job_id: str
    success: bool
    worker: str
    output: Any
    duration_seconds: float
    error: Optional[str] = None


class OrchestratorBackend(ABC):
    """Abstract base class for orchestrator backends."""

    @abstractmethod
    async def get_available_workers(self) -> List[WorkerStatus]:
        """Get list of available workers."""
        pass

    @abstractmethod
    async def run_selfplay(
        self,
        games: int,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[JobResult]:
        """Run selfplay games on available workers.

        Args:
            games: Total number of games to generate
            board_type: Board type for games
            num_players: Number of players per game
            model_path: Optional path to model weights
            **kwargs: Additional backend-specific options

        Returns:
            List of job results
        """
        pass

    @abstractmethod
    async def run_tournament(
        self,
        agent_ids: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 20,
        **kwargs,
    ) -> JobResult:
        """Run a tournament on available workers.

        Args:
            agent_ids: List of agent IDs to compete
            board_type: Board type for games
            num_players: Number of players per game
            games_per_pairing: Games per agent pairing
            **kwargs: Additional backend-specific options

        Returns:
            Job result with tournament results
        """
        pass

    @abstractmethod
    async def run_training(
        self,
        data_path: str,
        model_output_path: str,
        epochs: int = 100,
        **kwargs,
    ) -> JobResult:
        """Run training job on a worker.

        Args:
            data_path: Path to training data
            model_output_path: Path for output model
            epochs: Number of training epochs
            **kwargs: Additional training options

        Returns:
            Job result
        """
        pass

    @abstractmethod
    async def sync_models(
        self,
        model_paths: List[str],
        target_workers: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Sync models to workers.

        Args:
            model_paths: List of model paths to sync
            target_workers: Optional list of target workers (all if None)

        Returns:
            Dict of worker -> success status
        """
        pass

    @abstractmethod
    async def sync_data(
        self,
        source_workers: Optional[List[str]] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """Sync data from workers to local.

        Args:
            source_workers: Optional list of source workers (all if None)
            target_path: Local target path for synced data

        Returns:
            Dict of worker -> games synced count
        """
        pass


class LocalBackend(OrchestratorBackend):
    """Execute all jobs locally."""

    def __init__(self, working_dir: Optional[str] = None):
        self.executor = LocalExecutor(working_dir)
        self._ai_service_root = Path(__file__).parent.parent.parent

    async def get_available_workers(self) -> List[WorkerStatus]:
        """Local backend has one worker - the local machine."""
        import psutil

        return [
            WorkerStatus(
                name="local",
                available=True,
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                active_jobs=0,
            )
        ]

    async def run_selfplay(
        self,
        games: int,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[JobResult]:
        """Run selfplay locally."""
        import time
        from uuid import uuid4

        job_id = str(uuid4())[:8]
        start = time.time()

        cmd = (
            f"python scripts/run_self_play_soak.py "
            f"--games {games} "
            f"--board-type {board_type} "
            f"--num-players {num_players}"
        )
        if model_path:
            cmd += f" --model {model_path}"

        result = await self.executor.run(cmd, timeout=kwargs.get("timeout", 7200))

        return [
            JobResult(
                job_id=job_id,
                success=result.success,
                worker="local",
                output=result.stdout,
                duration_seconds=time.time() - start,
                error=result.stderr if not result.success else None,
            )
        ]

    async def run_tournament(
        self,
        agent_ids: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 20,
        **kwargs,
    ) -> JobResult:
        """Run tournament locally using the tournament framework."""
        import time
        from uuid import uuid4

        job_id = str(uuid4())[:8]
        start = time.time()

        try:
            from app.tournament import run_quick_tournament

            results = run_quick_tournament(
                agent_ids=agent_ids,
                board_type=board_type,
                num_players=num_players,
                games_per_pairing=games_per_pairing,
            )

            return JobResult(
                job_id=job_id,
                success=True,
                worker="local",
                output=results.to_dict(),
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            return JobResult(
                job_id=job_id,
                success=False,
                worker="local",
                output=None,
                duration_seconds=time.time() - start,
                error=str(e),
            )

    async def run_training(
        self,
        data_path: str,
        model_output_path: str,
        epochs: int = 100,
        **kwargs,
    ) -> JobResult:
        """Run training locally."""
        import time
        from uuid import uuid4

        job_id = str(uuid4())[:8]
        start = time.time()

        cmd = (
            f"python scripts/run_nn_training_baseline.py "
            f"--data {data_path} "
            f"--output {model_output_path} "
            f"--epochs {epochs}"
        )

        result = await self.executor.run(cmd, timeout=kwargs.get("timeout", 14400))

        return JobResult(
            job_id=job_id,
            success=result.success,
            worker="local",
            output=result.stdout,
            duration_seconds=time.time() - start,
            error=result.stderr if not result.success else None,
        )

    async def sync_models(
        self,
        model_paths: List[str],
        target_workers: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """No-op for local backend."""
        return {"local": True}

    async def sync_data(
        self,
        source_workers: Optional[List[str]] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """No-op for local backend."""
        return {"local": 0}


class SSHBackend(OrchestratorBackend):
    """Execute jobs via SSH on remote hosts."""

    def __init__(self, hosts_config_path: Optional[str] = None):
        self.pool = ExecutorPool()
        self._hosts: Dict[str, Dict] = {}
        self._load_hosts(hosts_config_path)

    def _load_hosts(self, config_path: Optional[str]) -> None:
        """Load hosts from configuration."""
        if config_path is None:
            try:
                from app.config.unified_config import get_config
                config = get_config()
                config_path = config.hosts_config_path
            except ImportError:
                config_path = "config/distributed_hosts.yaml"

        # Resolve relative path
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            ai_root = Path(__file__).parent.parent.parent
            config_path_obj = ai_root / config_path

        if not config_path_obj.exists():
            logger.warning(f"Hosts config not found: {config_path_obj}")
            return

        import yaml
        with open(config_path_obj) as f:
            data = yaml.safe_load(f) or {}

        hosts = data.get("hosts", {})
        for name, host_data in hosts.items():
            status = host_data.get("status", "ready")
            if status not in ("ready", "active"):
                continue

            self._hosts[name] = host_data
            self.pool.add_ssh(
                name=name,
                host=host_data.get("ssh_host") or host_data.get("tailscale_ip"),
                user=host_data.get("ssh_user"),
                port=host_data.get("ssh_port", 22),
                key_path=host_data.get("ssh_key"),
            )

        logger.info(f"Loaded {len(self._hosts)} SSH hosts")

    async def get_available_workers(self) -> List[WorkerStatus]:
        """Check which SSH hosts are available."""
        availability = await self.pool.check_all_available()

        workers = []
        for name, available in availability.items():
            host_data = self._hosts.get(name, {})
            workers.append(
                WorkerStatus(
                    name=name,
                    available=available,
                    metadata={
                        "gpu": host_data.get("gpu"),
                        "memory_gb": host_data.get("memory_gb"),
                        "role": host_data.get("role"),
                        "training_enabled": host_data.get("training_enabled", True),
                        "selfplay_enabled": host_data.get("selfplay_enabled", True),
                    },
                )
            )

        return workers

    async def run_selfplay(
        self,
        games: int,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[JobResult]:
        """Distribute selfplay across SSH workers."""
        import time
        from uuid import uuid4

        workers = await self.get_available_workers()
        available = [w for w in workers if w.available]

        if not available:
            logger.warning("No SSH workers available, falling back to local")
            local = LocalBackend()
            return await local.run_selfplay(games, board_type, num_players, model_path, **kwargs)

        # Selfplay is CPU-bound - prioritize hosts with low CPU utilization
        try:
            from app.coordination import get_hosts_for_cpu_tasks
            cpu_hosts = get_hosts_for_cpu_tasks(
                [w.name for w in available],
                max_cpu_util=80.0,  # Only use hosts with <80% CPU
            )
            if cpu_hosts:
                # Filter to only use low-CPU hosts for selfplay
                available = [w for w in available if w.name in cpu_hosts]
        except ImportError:
            pass  # Use all available workers

        # Distribute games across workers
        games_per_worker = max(1, games // len(available))
        results = []

        async def run_on_worker(worker: WorkerStatus, num_games: int) -> JobResult:
            job_id = str(uuid4())[:8]
            start = time.time()

            host_data = self._hosts.get(worker.name, {})
            ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")
            venv_activate = host_data.get("venv_activate", "")

            cmd = f"cd {ringrift_path} && "
            if venv_activate:
                cmd += f"{venv_activate} && "
            cmd += (
                f"python scripts/run_self_play_soak.py "
                f"--games {num_games} "
                f"--board-type {board_type} "
                f"--num-players {num_players}"
            )
            if model_path:
                cmd += f" --model {model_path}"

            result = await self.pool.run_on(
                worker.name, cmd, timeout=kwargs.get("timeout", 7200)
            )

            return JobResult(
                job_id=job_id,
                success=result.success,
                worker=worker.name,
                output=result.stdout,
                duration_seconds=time.time() - start,
                error=result.stderr if not result.success else None,
            )

        # Run on all workers in parallel
        tasks = [run_on_worker(w, games_per_worker) for w in available]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to JobResults
        job_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                job_results.append(
                    JobResult(
                        job_id=f"error-{i}",
                        success=False,
                        worker=available[i].name,
                        output=None,
                        duration_seconds=0,
                        error=str(result),
                    )
                )
            else:
                job_results.append(result)

        return job_results

    async def run_tournament(
        self,
        agent_ids: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 20,
        **kwargs,
    ) -> JobResult:
        """Run tournament on best available worker."""
        import time
        from uuid import uuid4

        workers = await self.get_available_workers()
        available = [w for w in workers if w.available]

        if not available:
            logger.warning("No SSH workers available, running locally")
            local = LocalBackend()
            return await local.run_tournament(
                agent_ids, board_type, num_players, games_per_pairing, **kwargs
            )

        # Pick best worker for tournaments (CPU-bound task)
        # Prefer non-GPU workers to keep GPUs free for training
        # Use resource-aware selection if available
        try:
            from app.coordination import get_hosts_for_cpu_tasks
            cpu_hosts = get_hosts_for_cpu_tasks(
                [w.name for w in available],
                max_cpu_util=70.0,
            )
            if cpu_hosts:
                worker = next((w for w in available if w.name == cpu_hosts[0]), available[0])
            else:
                # Fallback: prefer non-GPU workers
                non_gpu_workers = [w for w in available if not w.metadata.get("gpu")]
                worker = non_gpu_workers[0] if non_gpu_workers else available[0]
        except ImportError:
            # Fallback: prefer non-GPU workers
            non_gpu_workers = [w for w in available if not w.metadata.get("gpu")]
            worker = non_gpu_workers[0] if non_gpu_workers else available[0]

        job_id = str(uuid4())[:8]
        start = time.time()

        host_data = self._hosts.get(worker.name, {})
        ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")
        venv_activate = host_data.get("venv_activate", "")

        agents_str = ",".join(agent_ids)
        cmd = f"cd {ringrift_path} && "
        if venv_activate:
            cmd += f"{venv_activate} && "
        cmd += (
            f"python scripts/run_unified_tournament.py "
            f"--agents {agents_str} "
            f"--board-type {board_type} "
            f"--num-players {num_players} "
            f"--games {games_per_pairing}"
        )

        result = await self.pool.run_on(
            worker.name, cmd, timeout=kwargs.get("timeout", 7200)
        )

        return JobResult(
            job_id=job_id,
            success=result.success,
            worker=worker.name,
            output=result.stdout,
            duration_seconds=time.time() - start,
            error=result.stderr if not result.success else None,
        )

    async def run_training(
        self,
        data_path: str,
        model_output_path: str,
        epochs: int = 100,
        **kwargs,
    ) -> JobResult:
        """Run training on best GPU worker (lowest GPU utilization)."""
        import time
        from uuid import uuid4

        workers = await self.get_available_workers()
        gpu_workers = [
            w for w in workers
            if w.available
            and w.metadata.get("gpu")
            and w.metadata.get("training_enabled", True)  # Respect training_enabled flag
            and "NVIDIA" in str(w.metadata.get("gpu", ""))  # Prefer CUDA GPUs
        ]

        # Fallback: include MPS GPUs if no CUDA available
        if not gpu_workers:
            gpu_workers = [
                w for w in workers
                if w.available and w.metadata.get("gpu") and w.metadata.get("training_enabled", True)
            ]

        if not gpu_workers:
            logger.warning("No GPU workers available, running locally")
            local = LocalBackend()
            return await local.run_training(data_path, model_output_path, epochs, **kwargs)

        # Select GPU worker with lowest GPU utilization
        try:
            from app.coordination import get_hosts_for_gpu_tasks
            gpu_hosts = get_hosts_for_gpu_tasks(
                [w.name for w in gpu_workers],
                max_gpu_util=85.0,  # Only consider hosts with <85% GPU util
            )
            if gpu_hosts:
                worker = next((w for w in gpu_workers if w.name == gpu_hosts[0]), gpu_workers[0])
            else:
                worker = gpu_workers[0]
        except ImportError:
            worker = gpu_workers[0]
        job_id = str(uuid4())[:8]
        start = time.time()

        host_data = self._hosts.get(worker.name, {})
        ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")
        venv_activate = host_data.get("venv_activate", "")

        cmd = f"cd {ringrift_path} && "
        if venv_activate:
            cmd += f"{venv_activate} && "

        # Build training command with correct arguments
        board_type = kwargs.get("board_type", "square8")
        num_players = kwargs.get("num_players", 2)

        # Convert local paths to remote-relative paths
        # Data should be synced to remote's data/games directory
        data_filename = Path(data_path).name
        remote_data_path = f"data/games/{data_filename}"

        # Use a timestamped run directory on the remote
        import time as time_mod
        run_id = f"{board_type}_{num_players}p_{int(time_mod.time())}"
        remote_run_dir = f"runs/{run_id}"

        cmd += (
            f"python scripts/run_nn_training_baseline.py "
            f"--run-dir {remote_run_dir} "
            f"--data-path {remote_data_path} "
            f"--board {board_type} "
            f"--num-players {num_players} "
            f"--epochs {epochs}"
        )

        result = await self.pool.run_on(
            worker.name, cmd, timeout=kwargs.get("timeout", 14400)
        )

        # If training succeeded, sync the model back to local
        if result.success:
            try:
                await self._sync_model_back(
                    worker_name=worker.name,
                    remote_run_dir=remote_run_dir,
                    board_type=board_type,
                    num_players=num_players,
                )
            except Exception as e:
                logger.warning(f"Failed to sync model back from {worker.name}: {e}")

        return JobResult(
            job_id=job_id,
            success=result.success,
            worker=worker.name,
            output=result.stdout,
            duration_seconds=time.time() - start,
            error=result.stderr if not result.success else None,
        )

    async def _sync_model_back(
        self,
        worker_name: str,
        remote_run_dir: str,
        board_type: str,
        num_players: int,
    ) -> None:
        """Sync trained model back from remote worker to local.

        The training script saves models to:
        - models/<model_id>.pth (best model)
        - models/checkpoints/<model_id>/ (checkpoints)
        - runs/<run_dir>/nn_training_report.json (report)

        Args:
            worker_name: Name of the worker that trained the model
            remote_run_dir: Remote directory where the report was saved
            board_type: Board type used for training
            num_players: Number of players
        """
        host_data = self._hosts.get(worker_name, {})
        ssh_host = host_data.get("ssh_host") or host_data.get("tailscale_ip")
        ssh_user = host_data.get("ssh_user", "ubuntu")
        ssh_key = host_data.get("ssh_key")
        ssh_port = host_data.get("ssh_port", 22)
        ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")

        if not ssh_host:
            logger.warning(f"No SSH host for {worker_name}, cannot sync model back")
            return

        # Build SSH options
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if ssh_key:
            key_path = Path(ssh_key).expanduser()
            ssh_opts.extend(["-i", str(key_path)])
        if ssh_port != 22:
            ssh_opts.extend(["-p", str(ssh_port)])

        ssh_opts_str = " ".join(ssh_opts)

        # Local destination - models directory
        local_models_dir = Path(__file__).parent.parent.parent / "models"
        local_models_dir.mkdir(parents=True, exist_ok=True)

        # Config key for naming
        config_key = f"{board_type}_{num_players}p"
        board_prefix = board_type[:3] if board_type.startswith("square") else board_type[:3]
        if board_type == "square8":
            board_prefix = "sq8"
        elif board_type == "square19":
            board_prefix = "sq19"
        elif board_type == "hexagonal":
            board_prefix = "hex"

        # Find the most recent model file on remote
        # Models are saved as: <board_prefix>_<players>p_nn_baseline_<timestamp>.pth
        find_cmd = (
            f"ssh {ssh_opts_str} {ssh_user}@{ssh_host} "
            f"\"find {ringrift_path}/models -maxdepth 1 -name '{board_prefix}_{num_players}p_*.pth' "
            f"-mmin -5 -type f -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2\""
        )

        logger.info(f"[SyncBack] Finding recent model on {worker_name}")

        proc = await asyncio.create_subprocess_shell(
            find_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0 or not stdout.strip():
            logger.warning(f"[SyncBack] No recent model found on {worker_name}: {stderr.decode()}")
            return

        remote_model_path = stdout.decode().strip()
        if not remote_model_path:
            logger.warning(f"[SyncBack] Empty model path from {worker_name}")
            return

        # Rsync the model file back
        dest_path = local_models_dir / f"ringrift_{config_key}.pth"

        rsync_cmd = [
            "rsync", "-avz", "--timeout=60",
            "-e", f"ssh {ssh_opts_str}",
            f"{ssh_user}@{ssh_host}:{remote_model_path}",
            str(dest_path),
        ]

        logger.info(f"[SyncBack] Downloading {remote_model_path} -> {dest_path}")

        proc = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            logger.warning(f"[SyncBack] rsync failed: {stderr.decode()}")
            return

        logger.info(f"[SyncBack] Model synced: {dest_path}")

    async def sync_models(
        self,
        model_paths: List[str],
        target_workers: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Sync models to workers via rsync over SSH."""
        workers = await self.get_available_workers()
        if target_workers:
            workers = [w for w in workers if w.name in target_workers]

        results = {}
        for worker in workers:
            if not worker.available:
                results[worker.name] = False
                continue

            host_data = self._hosts.get(worker.name, {})
            ssh_target = f"{host_data.get('ssh_user', 'ubuntu')}@{host_data.get('ssh_host')}"
            ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")

            success = True
            for model_path in model_paths:
                cmd = f"rsync -avz {model_path} {ssh_target}:{ringrift_path}/models/"
                result = await self.pool._local.run(cmd, timeout=300)
                if not result.success:
                    success = False
                    break

            results[worker.name] = success

        return results

    async def sync_data(
        self,
        source_workers: Optional[List[str]] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """Sync data from workers via rsync."""
        if target_path is None:
            target_path = str(Path(__file__).parent.parent.parent / "data" / "games")

        workers = await self.get_available_workers()
        if source_workers:
            workers = [w for w in workers if w.name in source_workers]

        results = {}
        for worker in workers:
            if not worker.available:
                results[worker.name] = 0
                continue

            host_data = self._hosts.get(worker.name, {})
            ssh_target = f"{host_data.get('ssh_user', 'ubuntu')}@{host_data.get('ssh_host')}"
            ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")

            # Count files before
            cmd = f"ls {target_path}/*.db 2>/dev/null | wc -l"
            before = await self.pool._local.run(cmd, timeout=10)
            before_count = int(before.stdout.strip()) if before.success else 0

            # Sync
            cmd = f"rsync -avz {ssh_target}:{ringrift_path}/data/games/*.db {target_path}/"
            await self.pool._local.run(cmd, timeout=600)

            # Count after
            cmd = f"ls {target_path}/*.db 2>/dev/null | wc -l"
            after = await self.pool._local.run(cmd, timeout=10)
            after_count = int(after.stdout.strip()) if after.success else 0

            results[worker.name] = after_count - before_count

        return results


# ============================================================================
# Slurm Backend
# ============================================================================


class SlurmBackend(OrchestratorBackend):
    """Execute jobs via Slurm on a stable HPC cluster."""

    def __init__(self, config, working_dir: Optional[str] = None):
        self.config = config
        self.executor = LocalExecutor(working_dir)
        self.repo_root = self._resolve_repo_root()
        self.job_dir = self._resolve_path(config.job_dir)
        self.log_dir = self._resolve_path(config.log_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_repo_root(self) -> Path:
        shared_root = getattr(self.config, "shared_root", None)
        if shared_root:
            return Path(shared_root) / getattr(self.config, "repo_subdir", "ai-service")
        return Path(__file__).parent.parent.parent

    def _resolve_path(self, path: str) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.repo_root / resolved
        return resolved

    def _normalize_job_name(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return safe[:128] if safe else "ringrift"

    def _build_sbatch_args(
        self,
        job_name: str,
        work_type: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        overrides = overrides or {}
        args: List[str] = []

        args.extend(["--job-name", job_name])
        args.extend(["--output", str(self.log_dir / f"{job_name}.%j.out")])
        args.extend(["--error", str(self.log_dir / f"{job_name}.%j.err")])

        account = overrides.get("account", getattr(self.config, "account", None))
        if account:
            args.extend(["--account", str(account)])

        qos = overrides.get("qos", getattr(self.config, "qos", None))
        if qos:
            args.extend(["--qos", str(qos)])

        partition_key = f"partition_{work_type}"
        partition = overrides.get("partition", getattr(self.config, partition_key, None))
        if partition:
            args.extend(["--partition", str(partition)])

        time_key = f"default_time_{work_type}"
        time_limit = overrides.get("time", getattr(self.config, time_key, None))
        if time_limit:
            args.extend(["--time", str(time_limit)])

        gpus_key = f"gpus_{work_type}"
        gpus = overrides.get("gpus", getattr(self.config, gpus_key, 0))
        if gpus and int(gpus) > 0:
            args.extend(["--gres", f"gpu:{int(gpus)}"])

        cpus_key = f"cpus_{work_type}"
        cpus = overrides.get("cpus", getattr(self.config, cpus_key, None))
        if cpus:
            args.extend(["--cpus-per-task", str(cpus)])

        mem_key = f"mem_{work_type}"
        mem = overrides.get("mem", getattr(self.config, mem_key, None))
        if mem:
            args.extend(["--mem", str(mem)])

        extra_args = overrides.get("extra_sbatch_args", None) or getattr(self.config, "extra_sbatch_args", [])
        args.extend([str(a) for a in extra_args])

        return args

    def _build_job_script(self, command: str) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(self.repo_root))}",
        ]

        venv_activate = getattr(self.config, "venv_activate", None)
        if venv_activate:
            lines.append(f"source {shlex.quote(str(venv_activate))}")

        setup_commands = getattr(self.config, "setup_commands", []) or []
        for cmd in setup_commands:
            lines.append(str(cmd))

        lines.append(command)
        return "\n".join(lines) + "\n"

    async def _submit_job(
        self,
        job_name: str,
        work_type: str,
        command: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
        from uuid import uuid4

        safe_name = self._normalize_job_name(job_name)
        script_body = self._build_job_script(command)
        script_path = self.job_dir / f"{safe_name}.{str(uuid4())[:8]}.sh"
        script_path.write_text(script_body, encoding="utf-8")
        script_path.chmod(0o755)

        sbatch_args = self._build_sbatch_args(safe_name, work_type, overrides)
        cmd = f"sbatch --parsable {' '.join(shlex.quote(a) for a in sbatch_args)} {shlex.quote(str(script_path))}"
        result = await self.executor.run(cmd, timeout=30)
        if not result.success:
            return None, result.stderr.strip() or result.stdout.strip(), script_path

        job_id = result.stdout.strip().split(";")[0].strip()
        if not job_id:
            return None, "sbatch returned empty job id", script_path

        return job_id, None, script_path

    async def _get_job_status(self, job_id: str) -> Tuple[Optional[str], Optional[str]]:
        squeue_cmd = f"squeue -j {shlex.quote(job_id)} -h -o %T"
        result = await self.executor.run(squeue_cmd, timeout=10)
        if result.success:
            state = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if state:
                return state, None

        sacct_cmd = f"sacct -j {shlex.quote(job_id)} --format=State,ExitCode -n -P"
        result = await self.executor.run(sacct_cmd, timeout=10)
        if not result.success:
            return None, result.stderr.strip() or result.stdout.strip() or "sacct failed"

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            state = parts[0].strip()
            exit_code = parts[1].strip() if len(parts) > 1 else ""
            if state:
                return state, exit_code or None

        return None, "sacct returned no state"

    async def _wait_for_job(
        self,
        job_id: str,
        timeout: float,
    ) -> Tuple[str, Optional[str]]:
        import time

        start = time.time()
        poll_interval = getattr(self.config, "poll_interval_seconds", 20)

        while time.time() - start < timeout:
            state, detail = await self._get_job_status(job_id)
            if state:
                normalized = state.split("+")[0].split(":")[0]
                if normalized in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                    return normalized, detail
                if normalized in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"):
                    await asyncio.sleep(poll_interval)
                    continue

            await asyncio.sleep(poll_interval)

        return "TIMEOUT", None

    async def get_available_workers(self) -> List[WorkerStatus]:
        """Surface Slurm availability as a single logical worker."""
        result = await self.executor.run("sinfo -h -o %P", timeout=10)
        if not result.success:
            return []
        partitions = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        return [
            WorkerStatus(
                name="slurm",
                available=True,
                metadata={"partitions": partitions},
            )
        ]

    async def run_selfplay(
        self,
        games: int,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[JobResult]:
        import time

        if model_path:
            logger.warning("Slurm selfplay ignores model_path; use model aliases/config instead.")

        job_name = f"ringrift-selfplay-{board_type}-{num_players}p"
        cmd_parts = [
            "python",
            "scripts/run_self_play_soak.py",
            "--num-games",
            str(games),
            "--board-type",
            board_type,
            "--num-players",
            str(num_players),
        ]

        extra_args = kwargs.get("extra_args", [])
        cmd_parts.extend([str(a) for a in extra_args])
        command = " ".join(shlex.quote(p) for p in cmd_parts)

        job_id, error, script_path = await self._submit_job(
            job_name,
            "selfplay",
            command,
            overrides=kwargs,
        )

        if not job_id:
            return [JobResult(
                job_id="",
                success=False,
                worker="slurm",
                output={"script_path": str(script_path) if script_path else None},
                duration_seconds=0.0,
                error=error or "Failed to submit Slurm job",
            )]

        start = time.time()
        state, detail = await self._wait_for_job(job_id, timeout=kwargs.get("timeout", 7200))
        success = state == "COMPLETED"
        return [JobResult(
            job_id=job_id,
            success=success,
            worker="slurm",
            output={
                "state": state,
                "detail": detail,
                "script_path": str(script_path) if script_path else None,
                "log_stdout": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.out"),
                "log_stderr": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.err"),
            },
            duration_seconds=time.time() - start,
            error=None if success else (detail or "Slurm job failed"),
        )]

    async def run_tournament(
        self,
        agent_ids: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 20,
        **kwargs,
    ) -> JobResult:
        import time
        import json

        job_name = f"ringrift-tournament-{board_type}-{num_players}p"
        agents_json = json.dumps(agent_ids)
        command = (
            "python - <<'PY'\n"
            "from app.tournament import run_quick_tournament\n"
            "import json\n"
            f"agents = json.loads({agents_json!r})\n"
            f"results = run_quick_tournament(agents, board_type={board_type!r}, "
            f"num_players={num_players}, games_per_pairing={games_per_pairing})\n"
            "print(json.dumps(results.to_dict()))\n"
            "PY"
        )

        job_id, error, script_path = await self._submit_job(
            job_name,
            "tournament",
            command,
            overrides=kwargs,
        )

        if not job_id:
            return JobResult(
                job_id="",
                success=False,
                worker="slurm",
                output={"script_path": str(script_path) if script_path else None},
                duration_seconds=0.0,
                error=error or "Failed to submit Slurm tournament job",
            )

        start = time.time()
        state, detail = await self._wait_for_job(job_id, timeout=kwargs.get("timeout", 7200))
        success = state == "COMPLETED"
        return JobResult(
            job_id=job_id,
            success=success,
            worker="slurm",
            output={
                "state": state,
                "detail": detail,
                "script_path": str(script_path) if script_path else None,
                "log_stdout": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.out"),
                "log_stderr": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.err"),
            },
            duration_seconds=time.time() - start,
            error=None if success else (detail or "Slurm tournament failed"),
        )

    async def run_training(
        self,
        data_path: str,
        model_output_path: str,
        epochs: int = 100,
        **kwargs,
    ) -> JobResult:
        import time
        from uuid import uuid4

        board_type = kwargs.get("board_type", "square8")
        num_players = kwargs.get("num_players", 2)
        run_id = f"{board_type}_{num_players}p_{str(uuid4())[:8]}"
        run_dir = f"runs/{run_id}"

        resolved_data_path = self._resolve_path(data_path)

        cmd_parts = [
            "python",
            "scripts/run_nn_training_baseline.py",
            "--run-dir",
            run_dir,
            "--data-path",
            str(resolved_data_path),
            "--board",
            board_type,
            "--num-players",
            str(num_players),
            "--epochs",
            str(epochs),
        ]

        extra_args = kwargs.get("extra_args", [])
        cmd_parts.extend([str(a) for a in extra_args])
        command = " ".join(shlex.quote(p) for p in cmd_parts)

        job_name = f"ringrift-training-{board_type}-{num_players}p"
        job_id, error, script_path = await self._submit_job(
            job_name,
            "training",
            command,
            overrides=kwargs,
        )

        if not job_id:
            return JobResult(
                job_id="",
                success=False,
                worker="slurm",
                output={"script_path": str(script_path) if script_path else None},
                duration_seconds=0.0,
                error=error or "Failed to submit Slurm training job",
            )

        start = time.time()
        state, detail = await self._wait_for_job(job_id, timeout=kwargs.get("timeout", 14400))
        success = state == "COMPLETED"
        return JobResult(
            job_id=job_id,
            success=success,
            worker="slurm",
            output={
                "state": state,
                "detail": detail,
                "script_path": str(script_path) if script_path else None,
                "run_dir": run_dir,
                "log_stdout": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.out"),
                "log_stderr": str(self.log_dir / f"{self._normalize_job_name(job_name)}.{job_id}.err"),
            },
            duration_seconds=time.time() - start,
            error=None if success else (detail or "Slurm training failed"),
        )

    async def sync_models(
        self,
        model_paths: List[str],
        target_workers: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """No-op when shared filesystem is available."""
        return {"slurm": True}

    async def sync_data(
        self,
        source_workers: Optional[List[str]] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """No-op when shared filesystem is available."""
        return {"slurm": 0}


# ============================================================================
# P2P Work Queue Backend
# ============================================================================


class P2PBackend(OrchestratorBackend):
    """Execute jobs via P2P orchestrator work queue.

    This backend adds work items to the centralized work queue and waits
    for results. Workers pull work from the queue and report completion.
    """

    def __init__(
        self,
        leader_url: Optional[str] = None,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ):
        """Initialize P2P backend.

        Args:
            leader_url: URL of the P2P leader (auto-detect if None)
            poll_interval: Seconds between status polls
            timeout: Default timeout for work items
        """
        self.leader_url = leader_url
        self.poll_interval = poll_interval
        self.default_timeout = timeout
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get_leader_url(self) -> str:
        """Get the P2P leader URL."""
        if self.leader_url:
            return self.leader_url

        # Try to get from work queue module
        try:
            from app.coordination.work_queue import get_work_queue
            wq = get_work_queue()
            if wq and hasattr(wq, 'leader_url'):
                return wq.leader_url
        except ImportError:
            pass

        # Try common localhost port
        return "http://localhost:8770"

    async def get_available_workers(self) -> List[WorkerStatus]:
        """Get available workers from P2P cluster."""
        try:
            session = await self._get_session()
            url = f"{await self._get_leader_url()}/health"
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                # Return leader as a worker
                return [WorkerStatus(
                    name=data.get("node_id", "unknown"),
                    available=data.get("healthy", False),
                    cpu_percent=data.get("cpu_percent", 0),
                    memory_percent=data.get("memory_percent", 0),
                    active_jobs=data.get("selfplay_jobs", 0) + data.get("training_jobs", 0),
                )]
        except Exception as e:
            logger.warning(f"Failed to get P2P workers: {e}")
            return []

    async def _add_work(
        self,
        work_type: str,
        priority: int,
        config: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """Add a work item to the queue."""
        try:
            session = await self._get_session()
            url = f"{await self._get_leader_url()}/work/add"
            payload = {
                "work_type": work_type,
                "priority": priority,
                "config": config,
            }
            if timeout_seconds:
                payload["timeout_seconds"] = timeout_seconds

            async with session.post(url, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to add work: {resp.status}")
                    return None
                data = await resp.json()
                return data.get("work_id")
        except Exception as e:
            logger.error(f"Error adding work: {e}")
            return None

    async def _wait_for_completion(
        self,
        work_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Wait for a work item to complete."""
        import time
        start = time.time()
        session = await self._get_session()
        url = f"{await self._get_leader_url()}/work/status"

        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    data = await resp.json()

                    # Check if work is completed (in running list with completed status)
                    for item in data.get("running", []):
                        if item.get("work_id") == work_id:
                            status = item.get("status")
                            if status in ("completed", "failed", "timeout"):
                                return item

                    # Check if still pending
                    for item in data.get("pending", []):
                        if item.get("work_id") == work_id:
                            # Still pending, wait
                            break
                    else:
                        # Not found in pending or running - check history
                        hist_url = f"{await self._get_leader_url()}/work/history?limit=10"
                        async with session.get(hist_url, timeout=10) as hist_resp:
                            if hist_resp.status == 200:
                                hist_data = await hist_resp.json()
                                for item in hist_data.get("history", []):
                                    if item.get("work_id") == work_id:
                                        return item

            except Exception as e:
                logger.warning(f"Error polling work status: {e}")

            await asyncio.sleep(self.poll_interval)

        return {"status": "timeout", "work_id": work_id, "error": "Timeout waiting for completion"}

    async def run_selfplay(
        self,
        games: int,
        board_type: str,
        num_players: int,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[JobResult]:
        """Run selfplay via work queue."""
        import time
        from uuid import uuid4

        work_id = await self._add_work(
            work_type="selfplay",
            priority=kwargs.get("priority", 50),
            config={
                "board_type": board_type,
                "num_players": num_players,
                "games": games,
                "model_path": model_path,
            },
            timeout_seconds=kwargs.get("timeout", self.default_timeout),
        )

        if not work_id:
            return [JobResult(
                job_id=str(uuid4())[:8],
                success=False,
                worker="unknown",
                output=None,
                duration_seconds=0,
                error="Failed to add work to queue",
            )]

        start = time.time()
        result = await self._wait_for_completion(
            work_id,
            timeout=kwargs.get("timeout", self.default_timeout),
        )

        return [JobResult(
            job_id=work_id,
            success=result.get("status") == "completed",
            worker=result.get("claimed_by", "unknown"),
            output=result.get("result"),
            duration_seconds=time.time() - start,
            error=result.get("error"),
        )]

    async def run_tournament(
        self,
        agent_ids: List[str],
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 20,
        **kwargs,
    ) -> JobResult:
        """Run tournament via work queue."""
        import time
        from uuid import uuid4

        work_id = await self._add_work(
            work_type="tournament",
            priority=kwargs.get("priority", 70),
            config={
                "agent_ids": agent_ids,
                "board_type": board_type,
                "num_players": num_players,
                "games_per_pairing": games_per_pairing,
            },
            timeout_seconds=kwargs.get("timeout", 7200),
        )

        if not work_id:
            return JobResult(
                job_id=str(uuid4())[:8],
                success=False,
                worker="unknown",
                output=None,
                duration_seconds=0,
                error="Failed to add tournament work to queue",
            )

        start = time.time()
        result = await self._wait_for_completion(
            work_id,
            timeout=kwargs.get("timeout", 7200),
        )

        return JobResult(
            job_id=work_id,
            success=result.get("status") == "completed",
            worker=result.get("claimed_by", "unknown"),
            output=result.get("result"),
            duration_seconds=time.time() - start,
            error=result.get("error"),
        )

    async def run_training(
        self,
        data_path: str,
        model_output_path: str,
        epochs: int = 100,
        **kwargs,
    ) -> JobResult:
        """Run training via work queue."""
        import time
        from uuid import uuid4

        board_type = kwargs.get("board_type", "square8")
        num_players = kwargs.get("num_players", 2)

        work_id = await self._add_work(
            work_type="training",
            priority=kwargs.get("priority", 80),
            config={
                "data_path": data_path,
                "model_output_path": model_output_path,
                "epochs": epochs,
                "board_type": board_type,
                "num_players": num_players,
            },
            timeout_seconds=kwargs.get("timeout", 14400),  # 4 hours default
        )

        if not work_id:
            return JobResult(
                job_id=str(uuid4())[:8],
                success=False,
                worker="unknown",
                output=None,
                duration_seconds=0,
                error="Failed to add training work to queue",
            )

        start = time.time()
        result = await self._wait_for_completion(
            work_id,
            timeout=kwargs.get("timeout", 14400),
        )

        return JobResult(
            job_id=work_id,
            success=result.get("status") == "completed",
            worker=result.get("claimed_by", "unknown"),
            output=result.get("result"),
            duration_seconds=time.time() - start,
            error=result.get("error"),
        )

    async def sync_data(
        self,
        source_workers: Optional[List[str]] = None,
        target_path: Optional[str] = None,
    ) -> Dict[str, int]:
        """Sync data is handled by P2P orchestrator - not needed."""
        return {}

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None


# ============================================================================
# Backend Factory
# ============================================================================

_backend_instance: Optional[OrchestratorBackend] = None


def get_backend(
    backend_type: Optional[BackendType] = None,
    force_new: bool = False,
) -> OrchestratorBackend:
    """Get the configured orchestrator backend.

    Args:
        backend_type: Explicit backend type (auto-detect from config if None)
        force_new: Create new instance even if one exists

    Returns:
        Configured backend instance
    """
    global _backend_instance

    if _backend_instance is not None and not force_new:
        return _backend_instance

    # Determine backend type from config if not specified
    if backend_type is None:
        try:
            from app.config.unified_config import get_config
            config = get_config()

            backend_choice = str(getattr(config, "execution_backend", "auto") or "auto").lower()
            if backend_choice != "auto":
                try:
                    backend_type = BackendType(backend_choice)
                except ValueError:
                    logger.warning(f"Unknown execution_backend={backend_choice!r}, falling back to auto")
                    backend_type = None

            if backend_type is None:
                if getattr(config, "slurm", None) and config.slurm.enabled:
                    backend_type = BackendType.SLURM
                elif config.hosts_config_path and Path(config.hosts_config_path).exists():
                    backend_type = BackendType.SSH
                else:
                    backend_type = BackendType.LOCAL
        except ImportError:
            backend_type = BackendType.LOCAL

    # Create backend
    if backend_type == BackendType.SSH:
        _backend_instance = SSHBackend()
    elif backend_type == BackendType.P2P:
        _backend_instance = P2PBackend()
    elif backend_type == BackendType.SLURM:
        try:
            from app.config.unified_config import get_config
            config = get_config()
            _backend_instance = SlurmBackend(config.slurm)
        except Exception as exc:
            logger.warning(f"Failed to initialize Slurm backend: {exc}. Falling back to local.")
            _backend_instance = LocalBackend()
    elif backend_type == BackendType.LOCAL:
        _backend_instance = LocalBackend()
    else:
        # Default to local
        _backend_instance = LocalBackend()

    logger.info(f"Using orchestrator backend: {backend_type.value}")
    return _backend_instance
