"""Orchestrator backend abstraction for different execution strategies.

This module provides a unified interface for running distributed workloads
across different backends:
- LocalBackend: Execute on the local machine
- SSHBackend: Execute via SSH on remote hosts
- P2PBackend: Execute via P2P orchestrator REST API

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

        # Pick best worker (prefer GPU for tournaments)
        gpu_workers = [w for w in available if w.metadata.get("gpu")]
        worker = gpu_workers[0] if gpu_workers else available[0]

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
        """Run training on best GPU worker."""
        import time
        from uuid import uuid4

        workers = await self.get_available_workers()
        gpu_workers = [
            w for w in workers
            if w.available and w.metadata.get("gpu")
        ]

        if not gpu_workers:
            logger.warning("No GPU workers available, running locally")
            local = LocalBackend()
            return await local.run_training(data_path, model_output_path, epochs, **kwargs)

        worker = gpu_workers[0]
        job_id = str(uuid4())[:8]
        start = time.time()

        host_data = self._hosts.get(worker.name, {})
        ringrift_path = host_data.get("ringrift_path", "~/ringrift/ai-service")
        venv_activate = host_data.get("venv_activate", "")

        cmd = f"cd {ringrift_path} && "
        if venv_activate:
            cmd += f"{venv_activate} && "
        cmd += (
            f"python scripts/run_nn_training_baseline.py "
            f"--data {data_path} "
            f"--output {model_output_path} "
            f"--epochs {epochs}"
        )

        result = await self.pool.run_on(
            worker.name, cmd, timeout=kwargs.get("timeout", 14400)
        )

        return JobResult(
            job_id=job_id,
            success=result.success,
            worker=worker.name,
            output=result.stdout,
            duration_seconds=time.time() - start,
            error=result.stderr if not result.success else None,
        )

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

            # Check if we have remote hosts configured
            if config.hosts_config_path and Path(config.hosts_config_path).exists():
                backend_type = BackendType.SSH
            else:
                backend_type = BackendType.LOCAL
        except ImportError:
            backend_type = BackendType.LOCAL

    # Create backend
    if backend_type == BackendType.SSH:
        _backend_instance = SSHBackend()
    elif backend_type == BackendType.LOCAL:
        _backend_instance = LocalBackend()
    else:
        # Default to local
        _backend_instance = LocalBackend()

    logger.info(f"Using orchestrator backend: {backend_type.value}")
    return _backend_instance
