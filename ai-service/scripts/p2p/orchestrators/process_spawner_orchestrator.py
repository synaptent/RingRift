"""Process Spawner Orchestrator - Handles job process lifecycle.

January 2026: Created as part of Phase 3 P2POrchestrator decomposition.

Responsibilities:
- Local job process spawning (selfplay, GPU selfplay, etc.)
- Cluster-wide job management and distribution
- Job type selection and command building
- Process lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.async_context import safe_create_task
from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

# Import job type constants if available
try:
    from scripts.p2p.job_types import JobType, ClusterJob
    HAS_JOB_TYPES = True
except ImportError:
    HAS_JOB_TYPES = False
    JobType = None
    ClusterJob = None

# Import selfplay config helpers if available
try:
    from scripts.p2p.config.selfplay_job_configs import (
        SELFPLAY_ENGINE_MODES,
        GUMBEL_ENGINE_MODES,
    )
    HAS_SELFPLAY_CONFIG = True
except ImportError:
    HAS_SELFPLAY_CONFIG = False
    SELFPLAY_ENGINE_MODES = {"nn-only", "gumbel-mcts-only", "mixed"}
    GUMBEL_ENGINE_MODES = {"gumbel-mcts", "gumbel", "gumbel-mcts-only"}

# Import safeguards if available
try:
    from app.coordination.safeguards import check_before_spawn
    HAS_SAFEGUARDS = True
    _safeguards = True  # Placeholder for safeguard module
except ImportError:
    HAS_SAFEGUARDS = False
    _safeguards = None
    check_before_spawn = None

# Import Gumbel budget utilities if available
try:
    from app.ai.gumbel_common import get_adaptive_budget_for_elo
    HAS_GUMBEL_BUDGET = True
except ImportError:
    HAS_GUMBEL_BUDGET = False

    def get_adaptive_budget_for_elo(elo: float) -> int:
        """Fallback budget calculation."""
        if elo < 1200:
            return 64
        elif elo < 1400:
            return 150
        elif elo < 1600:
            return 400
        else:
            return 800


class ProcessSpawnerOrchestrator(BaseOrchestrator):
    """Orchestrator for job process spawning and lifecycle management.

    This orchestrator handles all aspects of starting and managing job processes:
    - Building commands for different job types (selfplay, GPU selfplay, etc.)
    - Spawning local processes with proper environment
    - Tracking process lifecycle and monitoring
    - Cluster-wide job distribution (leader only)

    The actual subprocess management is delegated to JobOrchestrator, but this
    orchestrator handles job type selection, command building, and lifecycle.

    Usage:
        # In P2POrchestrator.__init__:
        self.process_spawner = ProcessSpawnerOrchestrator(self)

        # Start a local job:
        job = await self.process_spawner.start_local_job(
            JobType.GPU_SELFPLAY, board_type="hex8", num_players=2
        )
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the process spawner orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Statistics
        self._local_jobs_started: int = 0
        self._local_jobs_completed: int = 0
        self._local_jobs_failed: int = 0
        self._cluster_jobs_dispatched: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "process_spawner"

    def health_check(self) -> HealthCheckResult:
        """Check the health of process spawner orchestrator.

        Returns:
            HealthCheckResult with job spawning status.
        """
        try:
            issues = []

            # Check job orchestrator availability
            jobs_orch = getattr(self._p2p, "jobs", None)
            if jobs_orch is None:
                issues.append("JobOrchestrator not available")

            # Check failure rate
            total = self._local_jobs_completed + self._local_jobs_failed
            if total > 10:
                failure_rate = self._local_jobs_failed / total
                if failure_rate > 0.3:
                    issues.append(f"High local job failure rate: {failure_rate:.0%}")

            # Check local jobs dict
            local_jobs = getattr(self._p2p, "local_jobs", None)
            if local_jobs is None:
                issues.append("local_jobs not available")

            healthy = len(issues) == 0
            message = "Process spawner healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "local_jobs_started": self._local_jobs_started,
                    "local_jobs_completed": self._local_jobs_completed,
                    "local_jobs_failed": self._local_jobs_failed,
                    "cluster_jobs_dispatched": self._cluster_jobs_dispatched,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_ai_service_path(self) -> str:
        """Get the AI service path."""
        if hasattr(self._p2p, "_get_ai_service_path"):
            return self._p2p._get_ai_service_path()
        ringrift_path = getattr(self._p2p, "ringrift_path", None)
        if ringrift_path:
            return str(Path(ringrift_path) / "ai-service")
        return ""

    def _get_script_path(self, script_name: str) -> str:
        """Get the full path to a script."""
        if hasattr(self._p2p, "_get_script_path"):
            return self._p2p._get_script_path(script_name)
        ai_service = self._get_ai_service_path()
        return str(Path(ai_service) / "scripts" / script_name)

    def _load_distributed_hosts(self) -> dict[str, Any]:
        """Load the distributed hosts configuration."""
        if hasattr(self._p2p, "_load_distributed_hosts"):
            return self._p2p._load_distributed_hosts()
        return {"hosts": {}}

    def _spawn_and_track_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cmd: list[str],
        output_dir: Path,
        log_filename: str = "run.log",
        cuda_visible_devices: str | None = None,
        safeguard_reason: str | None = None,
    ) -> tuple[Any, Any] | None:
        """Spawn and track a job using JobOrchestrator.

        Delegates to JobOrchestrator.spawn_and_track_job() if available,
        otherwise falls back to P2POrchestrator._spawn_and_track_job().
        """
        # Try JobOrchestrator first
        jobs_orch = getattr(self._p2p, "jobs", None)
        if jobs_orch is not None and hasattr(jobs_orch, "spawn_and_track_job"):
            return jobs_orch.spawn_and_track_job(
                job_id=job_id,
                job_type=job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                cmd=cmd,
                output_dir=output_dir,
                log_filename=log_filename,
                cuda_visible_devices=cuda_visible_devices,
                safeguard_reason=safeguard_reason,
            )

        # Fallback to P2POrchestrator method
        if hasattr(self._p2p, "_spawn_and_track_job"):
            return self._p2p._spawn_and_track_job(
                job_id=job_id,
                job_type=job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                cmd=cmd,
                output_dir=output_dir,
                log_filename=log_filename,
                cuda_visible_devices=cuda_visible_devices,
                safeguard_reason=safeguard_reason,
            )

        self._log_error("No job spawning method available")
        return None

    def _save_state(self) -> None:
        """Save P2P state."""
        if hasattr(self._p2p, "_save_state"):
            self._p2p._save_state()

    def _update_gpu_job_count(self, delta: int) -> None:
        """Update GPU job count."""
        if hasattr(self._p2p, "_update_gpu_job_count"):
            self._p2p._update_gpu_job_count(delta)

    # =========================================================================
    # Local Job Spawning
    # =========================================================================

    async def start_local_job(
        self,
        job_type: Any,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "gumbel-mcts",
        job_id: str | None = None,
        cuda_visible_devices: str | None = None,
        export_params: dict[str, Any] | None = None,
        simulation_budget: int | None = None,
    ) -> Any | None:
        """Start a job on the local node.

        Jan 29, 2026: Implementation moved from P2POrchestrator._start_local_job().

        SAFEGUARD: Checks coordination safeguards before spawning.

        Args:
            job_type: Type of job to start (JobType enum or string)
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players
            engine_mode: Engine mode for selfplay
            job_id: Optional job ID (auto-generated if not provided)
            cuda_visible_devices: CUDA device selection
            export_params: Parameters for DATA_EXPORT jobs
            simulation_budget: Gumbel MCTS budget (None = use tier default)

        Returns:
            ClusterJob if successful, None if blocked or failed
        """
        try:
            # SAFEGUARD: Check safeguards before spawning
            if HAS_SAFEGUARDS and _safeguards and check_before_spawn is not None:
                task_type_str = job_type.value if hasattr(job_type, "value") else str(job_type)
                allowed, reason = check_before_spawn(task_type_str, self.node_id)
                if not allowed:
                    self._log_info(f"SAFEGUARD blocked {task_type_str} on {self.node_id}: {reason}")
                    # Track blocked spawn via JobOrchestrationManager
                    job_orchestration = getattr(self._p2p, "job_orchestration", None)
                    if job_orchestration is not None:
                        job_orchestration.record_spawn_blocked(f"safeguard:{reason}")
                    return None

                # Apply backpressure delay
                try:
                    delay = _safeguards.get_delay() if hasattr(_safeguards, "get_delay") else 0
                except Exception:
                    delay = 0
                if delay > 0:
                    self._log_info(f"SAFEGUARD applying {delay:.1f}s backpressure delay")
                    await asyncio.sleep(delay)

            # Generate or validate job_id
            if job_id:
                job_id = str(job_id)
                jobs_lock = getattr(self._p2p, "jobs_lock", None)
                local_jobs = getattr(self._p2p, "local_jobs", {})
                if jobs_lock is not None:
                    with jobs_lock:
                        existing = local_jobs.get(job_id)
                else:
                    existing = local_jobs.get(job_id)
                if existing and getattr(existing, "status", None) == "running":
                    return existing
            else:
                job_id = str(uuid.uuid4())[:8]

            # Get JobType enum if needed
            if HAS_JOB_TYPES and JobType is not None:
                if isinstance(job_type, str):
                    job_type = JobType(job_type)

            # Route to appropriate handler based on job type
            job_type_val = job_type.value if hasattr(job_type, "value") else str(job_type)

            if job_type_val == "selfplay":
                return await self._start_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode
                )
            elif job_type_val == "cpu_selfplay":
                return await self._start_cpu_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode
                )
            elif job_type_val == "gpu_selfplay":
                return await self._start_gpu_selfplay_job(
                    job_id, job_type, board_type, num_players, cuda_visible_devices
                )
            elif job_type_val == "hybrid_selfplay":
                return await self._start_hybrid_selfplay_job(
                    job_id, job_type, board_type, num_players, engine_mode, cuda_visible_devices
                )
            elif job_type_val == "gumbel_selfplay":
                return await self._start_gumbel_selfplay_job(
                    job_id, job_type, board_type, num_players, simulation_budget, cuda_visible_devices
                )
            elif job_type_val == "data_export":
                return await self._start_data_export_job(
                    job_id, job_type, board_type, num_players, export_params
                )
            else:
                self._log_warning(f"Unknown job type: {job_type_val}")
                return None

        except Exception as e:
            self._log_error(f"Failed to start job: {e}")
            self._local_jobs_failed += 1
            return None

    async def _start_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
    ) -> Any | None:
        """Start a standard selfplay job."""
        # Normalize engine mode
        if engine_mode in GUMBEL_ENGINE_MODES:
            engine_mode_norm = "gumbel-mcts-only"
        elif engine_mode in SELFPLAY_ENGINE_MODES:
            engine_mode_norm = engine_mode
        else:
            engine_mode_norm = "nn-only"

        # Memory-safety defaults for large boards
        num_games = 1000
        extra_args: list[str] = []
        if board_type in ("square19", "hexagonal"):
            num_games = 200 if board_type == "square19" else 100
            extra_args.extend(["--memory-constrained"])

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "p2p",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use venv python if available
        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--verbose", "0",
            *extra_args,
        ]

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            safeguard_reason=f"selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Add process monitoring
        safe_create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "selfplay"
        ), name=f"spawner-monitor-selfplay-{job_id[:8]}")

        return job

    async def _start_cpu_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
    ) -> Any | None:
        """Start a CPU-only selfplay job."""
        # CPU-friendly engine modes
        cpu_engine_modes = {
            "descent-only", "minimax-only", "mcts-only", "heuristic-only",
            "random-only", "mixed", "nn-only", "best-vs-pool",
            "nn-vs-mcts", "nn-vs-minimax", "nn-vs-descent", "tournament-varied",
            "heuristic-vs-nn", "heuristic-vs-mcts", "random-vs-mcts",
        }
        engine_mode_norm = engine_mode if engine_mode in cpu_engine_modes else "nn-only"

        # CPU-only jobs can handle more games per batch
        num_games = 2000
        extra_args: list[str] = []
        if board_type in ("square19", "hexagonal"):
            num_games = 400 if board_type == "square19" else 200
            extra_args.extend(["--memory-constrained"])

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "p2p",
            f"{board_type}_{num_players}p_cpu",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--verbose", "0",
            *extra_args,
        ]

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            cuda_visible_devices="",  # Disable GPU
            safeguard_reason=f"cpu-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        safe_create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "cpu_selfplay"
        ), name=f"spawner-monitor-cpu-selfplay-{job_id[:8]}")

        return job

    async def _start_gpu_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a GPU selfplay job."""
        # Normalize board type
        board_arg = {
            "hex8": "hex8",
            "hex": "hex8",
            "square8": "square8",
            "square19": "square19",
            "hexagonal": "hexagonal",
        }.get(board_type, "square8")

        # Number of games per batch
        num_games = 100
        if board_arg == "square19":
            num_games = 50
        elif board_arg == "hexagonal":
            num_games = 100

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "games",
        )

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_gpu_selfplay.py"),
            "--board", board_arg,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--output-dir", str(output_dir),
        ]

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("gpu_selfplay")

        gpu_engine_mode = "gumbel-mcts"
        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=gpu_engine_mode,
            cmd=cmd,
            output_dir=output_dir,
            log_filename="gpu_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"gpu-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track GPU job count
        self._update_gpu_job_count(+1)

        # Track diversity metrics
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": gpu_engine_mode,
            })

        # Monitor GPU selfplay
        job_coord_manager = getattr(self._p2p, "job_coordination_manager", None)
        if job_coord_manager is not None:
            safe_create_task(job_coord_manager.monitor_gpu_selfplay_and_validate(
                job_id, proc, output_dir, board_type, num_players
            ), name=f"spawner-monitor-gpu-selfplay-{job_id[:8]}")

        return job

    async def _start_hybrid_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        engine_mode: str,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a hybrid CPU/GPU selfplay job."""
        # Normalize engine mode for hybrid
        hybrid_engine_modes = {
            "random-only", "heuristic-only", "mixed", "nnue-guided", "mcts",
            "gumbel-mcts-only", "maxn-only", "brs-only", "policy-only", "diverse"
        }
        nn_modes = {"nn-only", "best-vs-pool", "nn-vs-mcts", "nn-vs-minimax", "nn-vs-descent", "tournament-varied"}
        engine_mode_map = {
            "gumbel-mcts": "gumbel-mcts-only",
            "maxn": "maxn-only",
            "brs": "brs-only",
            "minimax": "minimax-only",
        }

        if engine_mode in hybrid_engine_modes:
            engine_mode_norm = engine_mode
        elif engine_mode in engine_mode_map:
            engine_mode_norm = engine_mode_map[engine_mode]
        elif engine_mode in nn_modes:
            engine_mode_norm = "nnue-guided"
        elif engine_mode in ("mcts-only", "descent-only"):
            engine_mode_norm = "mcts"
        elif engine_mode == "minimax-only":
            engine_mode_norm = "minimax-only"
        else:
            engine_mode_norm = "diverse"

        # Game counts based on board type
        num_games = 1000
        if board_type == "square19":
            num_games = 500
        elif board_type in ("hex", "hexagonal"):
            num_games = 300

        output_dir = Path(
            self._get_ai_service_path(),
            "data",
            "selfplay",
            "p2p_hybrid",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("run_self_play_soak.py"),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--log-jsonl", str(output_dir / "games.jsonl"),
            "--summary-json", str(output_dir / "summary.json"),
            "--record-db", str(output_dir / "games.db"),
            "--lean-db",
            "--engine-mode", engine_mode_norm,
            "--max-moves", "10000",
            "--verbose", "0",
        ]

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("hybrid_selfplay")

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode_norm,
            cmd=cmd,
            output_dir=output_dir,
            log_filename="hybrid_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"hybrid-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track diversity
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": engine_mode_norm,
            })

        safe_create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "hybrid_selfplay"
        ), name=f"spawner-monitor-hybrid-selfplay-{job_id[:8]}")

        return job

    async def _start_gumbel_selfplay_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        simulation_budget: int | None,
        cuda_visible_devices: str | None,
    ) -> Any | None:
        """Start a Gumbel MCTS selfplay job."""
        # Determine effective budget
        if simulation_budget is not None:
            effective_budget = simulation_budget
        else:
            # Look up config Elo and use adaptive budget
            config_key = f"{board_type}_{num_players}p"
            try:
                selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
                if selfplay_scheduler is not None and hasattr(selfplay_scheduler, "get_config_elo"):
                    config_elo = selfplay_scheduler.get_config_elo(config_key)
                else:
                    config_elo = 1200.0
                effective_budget = get_adaptive_budget_for_elo(config_elo)
                self._log_debug(f"[Gumbel] {config_key}: Elo={config_elo:.0f} -> budget={effective_budget}")
            except Exception:
                effective_budget = 150  # Fallback to bootstrap tier

        # Games based on board type and budget
        num_games = 50 if effective_budget >= 800 else 100
        if board_type == "square19":
            num_games = 10 if effective_budget >= 800 else 50
        elif board_type in ("hex", "hexagonal", "hex8"):
            num_games = 20 if effective_budget >= 800 else 100

        output_dir = Path(
            getattr(self._p2p, "ringrift_path", "."),
            "ai-service",
            "data",
            "selfplay",
            "gumbel",
            f"{board_type}_{num_players}p",
            job_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize board type for gumbel script
        board_arg = {"hex": "hexagonal", "hex8": "hex8"}.get(board_type, board_type)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        cmd = [
            python_exec,
            self._get_script_path("generate_gumbel_selfplay.py"),
            "--board", board_arg,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--simulation-budget", str(effective_budget),
            "--output-dir", str(output_dir),
            "--db", str(output_dir / "games.db"),
            "--seed", str(int(time.time() * 1000) % 2**31),
            "--allow-fresh-weights",
        ]

        # Check if GPU tree is disabled for this node
        node_config = self._load_distributed_hosts().get("hosts", {}).get(self.node_id, {})
        if not node_config.get("disable_gpu_tree", False):
            cmd.append("--use-gpu-tree")

        # GPU selection
        effective_cuda_devices = cuda_visible_devices
        if effective_cuda_devices is None or not str(effective_cuda_devices).strip():
            effective_cuda_devices = self._auto_select_gpu("gumbel_selfplay")

        result = self._spawn_and_track_job(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode="gumbel-mcts",
            cmd=cmd,
            output_dir=output_dir,
            log_filename="gumbel_run.log",
            cuda_visible_devices=effective_cuda_devices,
            safeguard_reason=f"gumbel-selfplay-{board_type}-{num_players}p",
        )
        if result is None:
            return None

        job, proc = result
        self._local_jobs_started += 1

        # Track diversity
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is not None:
            selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": "gumbel-mcts",
            })

        safe_create_task(self._monitor_selfplay_process(
            job_id, proc, output_dir, board_type, num_players, "gumbel_selfplay"
        ), name=f"spawner-monitor-gumbel-selfplay-{job_id[:8]}")

        return job

    async def _start_data_export_job(
        self,
        job_id: str,
        job_type: Any,
        board_type: str,
        num_players: int,
        export_params: dict[str, Any] | None,
    ) -> Any | None:
        """Start a data export job."""
        if not export_params:
            self._log_info("DATA_EXPORT job requires export_params")
            return None

        input_path = export_params.get("input_path")
        output_path = export_params.get("output_path")
        encoder_version = export_params.get("encoder_version", "v3")
        max_games = export_params.get("max_games", 5000)
        is_jsonl = export_params.get("is_jsonl", False)

        if not input_path or not output_path:
            self._log_info("DATA_EXPORT requires input_path and output_path")
            return None

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = Path(self._get_ai_service_path(), "venv", "bin", "python")
        python_exec = str(venv_python) if venv_python.exists() else "python3"

        if is_jsonl:
            export_script = self._get_script_path("jsonl_to_npz.py")
            cmd = [
                python_exec,
                export_script,
                "--input", str(input_path),
                "--output", str(output_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--gpu-selfplay",
                "--max-games", str(max_games),
            ]
            if encoder_version and encoder_version != "default":
                cmd.extend(["--encoder-version", encoder_version])
        else:
            export_script = self._get_script_path("export_replay_dataset.py")
            cmd = [
                python_exec,
                export_script,
                "--db", str(input_path),
                "--output", str(output_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--max-games", str(max_games),
                "--require-completed",
                "--min-moves", "10",
            ]
            if encoder_version and encoder_version != "default":
                cmd.extend(["--encoder-version", encoder_version])

        # Start export process
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_ai_service_path()
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        log_path = output_dir / f"export_{job_id}.log"
        log_handle = open(log_path, "w")  # noqa: SIM115
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=self._get_ai_service_path(),
            )
        finally:
            log_handle.close()

        # Create ClusterJob
        if HAS_JOB_TYPES and ClusterJob is not None:
            job = ClusterJob(
                job_id=job_id,
                job_type=job_type,
                node_id=self.node_id,
                board_type=board_type,
                num_players=num_players,
                engine_mode="export",
                pid=proc.pid,
                started_at=time.time(),
                status="running",
            )
        else:
            job = {
                "job_id": job_id,
                "job_type": "data_export",
                "node_id": self.node_id,
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": "export",
                "pid": proc.pid,
                "started_at": time.time(),
                "status": "running",
            }

        # Track in local_jobs
        jobs_lock = getattr(self._p2p, "jobs_lock", None)
        local_jobs = getattr(self._p2p, "local_jobs", {})
        if jobs_lock is not None:
            with jobs_lock:
                local_jobs[job_id] = job
        else:
            local_jobs[job_id] = job

        self._log_info(f"Started DATA_EXPORT job {job_id} (PID {proc.pid}): {input_path} -> {output_path}")
        self._save_state()
        self._local_jobs_started += 1

        # Track via JobOrchestrationManager
        job_orchestration = getattr(self._p2p, "job_orchestration", None)
        if job_orchestration is not None:
            job_type_val = job_type.value if hasattr(job_type, "value") else str(job_type)
            job_orchestration.record_job_started(job_type_val)

        return job

    # =========================================================================
    # GPU Selection
    # =========================================================================

    def _auto_select_gpu(self, job_type: str) -> str:
        """Auto-select a GPU device for a job.

        Args:
            job_type: Type of job for counting running jobs

        Returns:
            CUDA_VISIBLE_DEVICES string
        """
        gpu_count = 0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                gpu_count = len([line for line in out.stdout.splitlines() if line.strip()])
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, OSError):
            gpu_count = 0

        if gpu_count > 0:
            # Count running jobs of this type
            jobs_lock = getattr(self._p2p, "jobs_lock", None)
            local_jobs = getattr(self._p2p, "local_jobs", {})
            if jobs_lock is not None:
                with jobs_lock:
                    running_jobs = sum(
                        1 for j in local_jobs.values()
                        if getattr(j, "status", None) == "running"
                        and job_type.lower() in str(getattr(j, "job_type", "")).lower()
                    )
            else:
                running_jobs = 0
            return str(running_jobs % gpu_count)
        else:
            return "0"

    # =========================================================================
    # Process Monitoring
    # =========================================================================

    async def _monitor_selfplay_process(
        self,
        job_id: str,
        proc: Any,
        output_dir: Path,
        board_type: str,
        num_players: int,
        job_type_str: str,
    ) -> None:
        """Monitor a selfplay process until completion.

        Delegates to P2POrchestrator._monitor_selfplay_process() if available.
        """
        try:
            if hasattr(self._p2p, "_monitor_selfplay_process"):
                await self._p2p._monitor_selfplay_process(
                    job_id, proc, output_dir, board_type, num_players, job_type_str
                )
                self._local_jobs_completed += 1
            else:
                # Simple fallback: wait for process to complete
                await asyncio.get_event_loop().run_in_executor(None, proc.wait)
                if proc.returncode == 0:
                    self._local_jobs_completed += 1
                else:
                    self._local_jobs_failed += 1
        except Exception as e:
            self._log_error(f"Error monitoring process {job_id}: {e}")
            self._local_jobs_failed += 1

    # =========================================================================
    # Cluster Job Management (Leader Only)
    # =========================================================================

    async def manage_cluster_jobs(self) -> None:
        """Manage jobs across the cluster (leader only).

        Jan 29, 2026: Implementation moved from P2POrchestrator._manage_cluster_jobs().

        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        # Import constants lazily to avoid circular imports
        try:
            from scripts.p2p.config.p2p_constants import (
                DISK_CLEANUP_THRESHOLD,
                DISK_CRITICAL_THRESHOLD,
                DISK_WARNING_THRESHOLD,
                GPU_IDLE_RESTART_TIMEOUT,
                GPU_IDLE_THRESHOLD,
                LOAD_MAX_FOR_NEW_JOBS,
                MEMORY_CRITICAL_THRESHOLD,
                MEMORY_WARNING_THRESHOLD,
                RUNAWAY_SELFPLAY_PROCESS_THRESHOLD,
            )
        except ImportError:
            # Fallback defaults
            DISK_CLEANUP_THRESHOLD = 85
            DISK_CRITICAL_THRESHOLD = 95
            DISK_WARNING_THRESHOLD = 80
            GPU_IDLE_RESTART_TIMEOUT = 600
            GPU_IDLE_THRESHOLD = 5
            LOAD_MAX_FOR_NEW_JOBS = 90
            MEMORY_CRITICAL_THRESHOLD = 90
            MEMORY_WARNING_THRESHOLD = 80
            RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = 128

        # Import selfplay config helpers
        try:
            from scripts.p2p.config.selfplay_job_configs import (
                get_filtered_configs,
                get_unique_configs,
                get_weighted_configs,
            )
            HAS_CONFIG_HELPERS = True
        except ImportError:
            HAS_CONFIG_HELPERS = False

        self._log_info("Leader: Managing cluster jobs...")

        # Track cluster management run
        job_orchestration = getattr(self._p2p, "job_orchestration", None)
        if job_orchestration is not None:
            job_orchestration.record_cluster_management_run()

        # Gather cluster state
        # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking event
        # loop on peers_lock contention (was causing 361s manage_cluster_jobs cycles)
        _snapshot_fn = getattr(self._p2p, "_get_peers_snapshot_nonblocking", None)
        if _snapshot_fn is not None:
            alive_peers = [p for p in _snapshot_fn() if p.is_alive()]
        else:
            peers = getattr(self._p2p, "peers", {})
            alive_peers = [p for p in peers.values() if p.is_alive()]

        # Add self (use cached self_info - refreshed by heartbeat/health loops)
        # Feb 22, 2026: Removed _update_self_info_async() call here. Resource
        # detection (GPU, disk, CPU) is slow on macOS (10-30s) and was blocking
        # the event loop, preventing work/claim HTTP requests from being processed.
        # self_info is refreshed in the background by heartbeat and health loops.
        self_info = getattr(self._p2p, "self_info", None)
        all_nodes = [*alive_peers]
        if self_info is not None:
            all_nodes.append(self_info)

        # Phase 1: Handle resource warnings and cleanup
        for node in all_nodes:
            if node.disk_percent >= DISK_CLEANUP_THRESHOLD:
                self._log_info(f"{node.node_id}: Disk at {node.disk_percent:.0f}% - triggering cleanup")
                if node.node_id == self.node_id:
                    if hasattr(self._p2p, "_cleanup_local_disk"):
                        await self._p2p._cleanup_local_disk()
                else:
                    if hasattr(self._p2p, "_request_remote_cleanup"):
                        await self._p2p._request_remote_cleanup(node)
                continue

            # Load shedding
            pressure_reasons: list[str] = []
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                pressure_reasons.append("memory")
            if node.disk_percent >= DISK_WARNING_THRESHOLD:
                pressure_reasons.append("disk")

            if pressure_reasons:
                selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
                desired = selfplay_scheduler.get_target_jobs_for_node(node) if selfplay_scheduler else 0
                if node.memory_percent >= MEMORY_CRITICAL_THRESHOLD or node.disk_percent >= DISK_CRITICAL_THRESHOLD:
                    desired = 0

                if node.selfplay_jobs > desired:
                    reason = "+".join(pressure_reasons)
                    self._log_info(
                        f"{node.node_id}: Load shedding (reason={reason}) "
                        f"{node.selfplay_jobs}->{desired} selfplay jobs"
                    )
                    if node.node_id == self.node_id:
                        if hasattr(self._p2p, "_reduce_local_selfplay_jobs"):
                            await self._p2p._reduce_local_selfplay_jobs(desired, reason=reason)
                    else:
                        if hasattr(self._p2p, "_request_reduce_selfplay"):
                            await self._p2p._request_reduce_selfplay(node, desired, reason=reason)

        # Phase 1.5: Detect stuck jobs (GPU idle with running processes)
        gpu_idle_since = getattr(self._p2p, "gpu_idle_since", {})
        for node in all_nodes:
            if not node.has_gpu or node.selfplay_jobs <= 0:
                if node.node_id in gpu_idle_since:
                    del gpu_idle_since[node.node_id]
                continue

            gpu_name = (node.gpu_name or "").upper()
            is_cuda_gpu = "MPS" not in gpu_name and "APPLE" not in gpu_name
            if not is_cuda_gpu:
                continue

            if node.gpu_percent < GPU_IDLE_THRESHOLD:
                if node.node_id not in gpu_idle_since:
                    gpu_idle_since[node.node_id] = time.time()
                    self._log_info(f"{node.node_id}: GPU idle ({node.gpu_percent:.0f}%) with {node.selfplay_jobs} jobs - monitoring")
                else:
                    idle_duration = time.time() - gpu_idle_since[node.node_id]
                    if idle_duration >= GPU_IDLE_RESTART_TIMEOUT:
                        self._log_info(f"{node.node_id}: STUCK! GPU idle for {idle_duration:.0f}s with {node.selfplay_jobs} jobs")
                        self._log_info(f"{node.node_id}: Requesting job restart...")
                        if node.node_id == self.node_id:
                            if hasattr(self._p2p, "_restart_local_stuck_jobs"):
                                await self._p2p._restart_local_stuck_jobs()
                        else:
                            if hasattr(self._p2p, "_request_job_restart"):
                                await self._p2p._request_job_restart(node)
                        del gpu_idle_since[node.node_id]
            else:
                if node.node_id in gpu_idle_since:
                    del gpu_idle_since[node.node_id]

        # Phase 1.6: Detect runaway selfplay processes
        selfplay_scheduler = getattr(self._p2p, "selfplay_scheduler", None)
        for node in all_nodes:
            try:
                target = selfplay_scheduler.get_target_jobs_for_node(node) if selfplay_scheduler else 8
                dynamic_threshold = max(16, target * 3)
                runaway_threshold = (
                    int(RUNAWAY_SELFPLAY_PROCESS_THRESHOLD)
                    if int(RUNAWAY_SELFPLAY_PROCESS_THRESHOLD) > 0
                    else int(dynamic_threshold)
                )
                if int(getattr(node, "selfplay_jobs", 0) or 0) < runaway_threshold:
                    continue
            except (ValueError, AttributeError):
                continue

            self._log_info(
                f"{node.node_id}: RUNAWAY selfplay count ({node.selfplay_jobs}) "
                f">= {runaway_threshold} — requesting restart sweep"
            )
            if node.node_id == self.node_id:
                if hasattr(self._p2p, "_restart_local_stuck_jobs"):
                    await self._p2p._restart_local_stuck_jobs()
            else:
                if hasattr(self._p2p, "_request_job_restart"):
                    await self._p2p._request_job_restart(node)

        # Phase 2: Calculate desired job distribution for healthy nodes
        healthy_nodes = [n for n in all_nodes if n.is_healthy()]
        healthy_nodes.sort(key=lambda n: n.get_load_score())

        if healthy_nodes:
            load_summary = ", ".join(
                f"{n.node_id[:12]}={n.get_load_score():.0f}%"
                for n in healthy_nodes[:5]
            )
            self._log_info(f"Load balancing: {load_summary}")

        for node in healthy_nodes:
            load_score = node.get_load_score()
            if load_score >= LOAD_MAX_FOR_NEW_JOBS:
                self._log_info(f"{node.node_id}: Load {load_score:.0f}% - skipping new job starts")
                continue

            # Get hybrid job targets
            if selfplay_scheduler is None:
                continue
            hybrid_targets = selfplay_scheduler.get_hybrid_job_targets(node)
            gpu_job_target = hybrid_targets.get("gpu_jobs", 0)
            cpu_only_target = hybrid_targets.get("cpu_only_jobs", 0)
            total_target = hybrid_targets.get("total_jobs", gpu_job_target)
            target_selfplay = total_target

            # Check if node needs more jobs
            if node.selfplay_jobs >= target_selfplay:
                continue

            needed = target_selfplay - node.selfplay_jobs
            self._log_info(f"{node.node_id} needs {needed} more selfplay jobs")

            # Get configs
            if HAS_CONFIG_HELPERS:
                node_mem = int(getattr(node, "memory_gb", 0) or 0)
                filtered_configs = get_filtered_configs(node_memory_gb=node_mem)
                weighted_configs = get_weighted_configs(filtered_configs)
                unique_configs = get_unique_configs(filtered_configs)
            else:
                unique_configs = [{"board_type": "hex8", "num_players": 2, "engine_mode": "gumbel-mcts"}]
                weighted_configs = unique_configs

            jobs_to_start = min(needed, 10)

            # Calculate GPU vs CPU-only slots
            current_gpu_jobs = min(node.selfplay_jobs, gpu_job_target)
            remaining_gpu_slots = max(0, gpu_job_target - current_gpu_jobs)
            remaining_cpu_slots = max(0, cpu_only_target)
            should_use_cpu_only = (
                selfplay_scheduler.should_spawn_cpu_only_jobs(node) and cpu_only_target > 0
            )

            for i in range(jobs_to_start):
                # Determine job type based on GPU capabilities
                gpu_name = (node.gpu_name or "").upper()
                node_has_gpu = bool(node.has_gpu)

                # YAML fallback when runtime GPU detection fails
                if not node_has_gpu or not gpu_name:
                    if hasattr(self._p2p, "_check_yaml_gpu_config"):
                        yaml_has_gpu, yaml_gpu_name, _ = self._p2p._check_yaml_gpu_config(node.node_id)
                        if yaml_has_gpu:
                            node_has_gpu = True
                            if yaml_gpu_name:
                                gpu_name = yaml_gpu_name.upper()

                is_high_end_gpu = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
                is_apple_gpu = "MPS" in gpu_name or "APPLE" in gpu_name

                # GPU unavailability check
                gpu_percent = getattr(node, "gpu_percent", 0) or 0
                gpu_job_count = getattr(node, "gpu_job_count", 0) or 0
                gpu_failure_count = getattr(node, "gpu_failure_count", 0) or 0
                last_gpu_failure = getattr(node, "last_gpu_job_failure", 0) or 0
                gpu_seems_unavailable = (
                    node_has_gpu
                    and not is_apple_gpu
                    and (
                        (gpu_job_count >= 2 and gpu_percent < 1)
                        or (gpu_failure_count >= 3 and time.time() - last_gpu_failure < 300)
                    )
                )

                # Role-based job preference
                job_preference = "both"
                if hasattr(self._p2p, "_get_node_job_preference"):
                    job_preference = self._p2p._get_node_job_preference(node.node_id)

                if job_preference == "training_only":
                    continue
                elif job_preference == "cpu_only":
                    spawn_cpu_only = True
                    if remaining_cpu_slots > 0:
                        remaining_cpu_slots -= 1
                    else:
                        continue
                elif job_preference == "gpu_only" and node_has_gpu and not gpu_seems_unavailable:
                    if remaining_gpu_slots > 0:
                        remaining_gpu_slots -= 1
                        spawn_cpu_only = False
                    else:
                        continue
                else:
                    spawn_cpu_only = False
                    if remaining_gpu_slots > 0 and not gpu_seems_unavailable:
                        remaining_gpu_slots -= 1
                    elif should_use_cpu_only and remaining_cpu_slots > 0:
                        spawn_cpu_only = True
                        remaining_cpu_slots -= 1
                    elif gpu_seems_unavailable:
                        spawn_cpu_only = True

                # Determine job type
                if spawn_cpu_only:
                    job_type_enum = self._get_job_type("cpu_selfplay")
                    task_type_str = "CPU-only (hybrid mode)"
                elif node.has_gpu and is_high_end_gpu and not is_apple_gpu and not gpu_seems_unavailable:
                    import random
                    if random.random() < 0.5:
                        job_type_enum = self._get_job_type("gumbel_selfplay")
                        task_type_str = "GUMBEL (high-quality)"
                    else:
                        job_type_enum = self._get_job_type("gpu_selfplay")
                        task_type_str = "GPU (high-parity)"
                elif node.has_gpu and not is_apple_gpu and not gpu_seems_unavailable:
                    job_type_enum = self._get_job_type("hybrid_selfplay")
                    task_type_str = "HYBRID (accel)"
                else:
                    job_type_enum = self._get_job_type("selfplay")
                    task_type_str = "CPU-only"

                gpu_info = f"gpu={node.gpu_name or 'none'}, gpu%={getattr(node, 'gpu_percent', 0):.0f}" if node.has_gpu else "no-gpu"
                self._log_info(f"Assigning {task_type_str} task to {node.node_id} ({gpu_info}, load={node.get_load_score():.0f}%)")

                # Config selection
                improvement_cycle = getattr(self._p2p, "improvement_cycle_manager", None)
                if improvement_cycle and hasattr(improvement_cycle, "get_next_selfplay_config_for_node"):
                    node_gpu_power = node.gpu_power_score() if hasattr(node, "gpu_power_score") else 0
                    node_memory = int(getattr(node, "memory_gb", 0) or 0)
                    cluster_data = getattr(self._p2p, "cluster_data_manifest", None)
                    config = improvement_cycle.get_next_selfplay_config_for_node(
                        node_gpu_power=node_gpu_power,
                        node_memory_gb=node_memory,
                        cluster_data=cluster_data,
                    )
                else:
                    config_idx = i % len(unique_configs)
                    config = unique_configs[config_idx]

                # Track diversity
                if selfplay_scheduler is not None:
                    selfplay_scheduler.track_diversity(config)

                # Dispatch job
                if node.node_id == self.node_id:
                    await self.start_local_job(
                        job_type_enum,
                        board_type=config["board_type"],
                        num_players=config["num_players"],
                        engine_mode=config["engine_mode"],
                    )
                else:
                    if hasattr(self._p2p, "_request_remote_job"):
                        await self._p2p._request_remote_job(
                            node, job_type_enum,
                            board_type=config["board_type"],
                            num_players=config["num_players"],
                            engine_mode=config["engine_mode"],
                        )
                self._cluster_jobs_dispatched += 1

    def _get_job_type(self, name: str) -> Any:
        """Get a JobType enum value by name."""
        if HAS_JOB_TYPES and JobType is not None:
            name_upper = name.upper()
            for member in JobType:
                if member.name == name_upper or member.value == name:
                    return member
        return name

    # =========================================================================
    # Decentralized Local Job Management
    # =========================================================================

    async def manage_local_jobs_decentralized(self) -> int:
        """DECENTRALIZED: Each node manages its own job count based on gossip state.

        Jan 29, 2026: Extracted from P2POrchestrator._manage_local_jobs_decentralized().

        Runs on ALL nodes to ensure selfplay continues even during leader elections.
        Each node autonomously:
        1. Checks its own resource pressure (disk, memory, CPU)
        2. Uses gossip state to calculate proportional job count
        3. Starts or stops local jobs as needed

        PHASE 3 DECENTRALIZATION (Dec 2025):
        - With Serf providing reliable failure detection, we can act quickly
        - Proportional allocation based on gossip cluster capacity
        - 30-second timeout for faster leader-failure recovery

        Returns:
            Number of jobs started/stopped
        """
        import random

        # Import constants lazily to avoid circular imports
        try:
            from scripts.p2p.config.p2p_constants import (
                DISK_WARNING_THRESHOLD,
                LEADER_WORK_DISPATCH_TIMEOUT,
                LEADERLESS_TRAINING_TIMEOUT,
                MEMORY_WARNING_THRESHOLD,
            )
        except ImportError:
            DISK_WARNING_THRESHOLD = 85
            LEADER_WORK_DISPATCH_TIMEOUT = 300
            LEADERLESS_TRAINING_TIMEOUT = 180
            MEMORY_WARNING_THRESHOLD = 90

        try:
            from scripts.p2p.node_info import NodeRole
        except ImportError:
            NodeRole = None

        p2p = self._p2p
        changes = 0
        now = time.time()

        # Rate limit: check every 30 seconds (reduced from 60s for faster response)
        last_check = getattr(p2p, "_last_local_job_manage", 0)
        if now - last_check < 30:
            return 0
        p2p._last_local_job_manage = now

        # Skip if leader is managing (avoid conflicts)
        # But continue if leaderless for > 30 seconds (reduced from 60s for Serf reliability)
        # Dec 30, 2025: Also allow self-assignment if leader exists but isn't dispatching work
        if NodeRole is not None and getattr(p2p, "role", None) == NodeRole.LEADER:
            return 0  # Leader uses centralized management
        if p2p.leader_id:
            leaderless_duration = now - getattr(p2p, "last_leader_seen", now)
            work_dispatch_gap = now - getattr(p2p, "last_work_from_leader", now)

            # Defer to leader only if BOTH conditions are met:
            # 1. Leader was seen recently (alive)
            # 2. Leader has been dispatching work recently (active)
            if leaderless_duration < LEADERLESS_TRAINING_TIMEOUT:
                if work_dispatch_gap < LEADER_WORK_DISPATCH_TIMEOUT:
                    return 0  # Have a functioning leader that's actively dispatching
                else:
                    # Leader is present but not dispatching work - allow self-assignment
                    logger.info(
                        f"LOCAL: Leader present but no work dispatched in {work_dispatch_gap:.0f}s "
                        f"(timeout={LEADER_WORK_DISPATCH_TIMEOUT}s) - self-assigning"
                    )

        # Update self info
        # Feb 2026: Use async version to prevent event loop blocking
        if hasattr(p2p, "_update_self_info_async"):
            await p2p._update_self_info_async()
        elif hasattr(p2p, "_update_self_info"):
            import asyncio
            await asyncio.to_thread(p2p._update_self_info)
        node = p2p.self_info

        # Check resource pressure - don't start jobs if under pressure
        if node.disk_percent >= DISK_WARNING_THRESHOLD:
            logger.info(f"LOCAL: Disk at {node.disk_percent:.0f}% - skipping job starts")
            if hasattr(p2p, "_cleanup_local_disk"):
                await p2p._cleanup_local_disk()
            return 0

        if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
            logger.info(f"LOCAL: Memory at {node.memory_percent:.0f}% - skipping job starts")
            return 0

        # Calculate target jobs for this node (delegated to SelfplayScheduler Dec 2025)
        selfplay_scheduler = getattr(p2p, "selfplay_scheduler", None)
        if selfplay_scheduler is None:
            return 0
        target_selfplay = selfplay_scheduler.get_target_jobs_for_node(node)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)

        # Dec 30, 2025: Cluster-aware job balancing
        # Prevent job concentration on a few nodes by checking cluster average
        # Skip spawning if we're already above average + 2 to give idle nodes a chance
        try:
            # Jan 10, 2026: Copy-on-read pattern - minimize lock hold time
            peers_lock = getattr(p2p, "peers_lock", None)
            peers = getattr(p2p, "peers", {})
            if peers_lock:
                with peers_lock:
                    peers_snapshot = list(peers.values())
            else:
                peers_snapshot = list(peers.values())
            # Compute job counts outside lock (is_alive() can be slow)
            peer_job_counts = [
                int(getattr(p, "selfplay_jobs", 0) or 0)
                for p in peers_snapshot
                if p.is_alive() and hasattr(p, "selfplay_jobs")
            ]
            if peer_job_counts:
                avg_jobs = sum(peer_job_counts) / len(peer_job_counts)
                if current_jobs > avg_jobs + 2:
                    logger.info(
                        f"LOCAL: Skipping job spawn - {current_jobs} jobs > cluster avg {avg_jobs:.1f}+2 "
                        f"(letting underutilized nodes catch up)"
                    )
                    return 0
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Cluster balance check error: {e}")

        # Start jobs if below target
        if current_jobs < target_selfplay:
            needed = min(target_selfplay - current_jobs, 3)  # Max 3 per cycle
            logger.info(f"LOCAL: Starting {needed} selfplay job(s) ({current_jobs}/{target_selfplay})")

            # Dec 27, 2025: Generate batch ID and emit BATCH_SCHEDULED
            batch_id = f"selfplay_{p2p.node_id}_{int(time.time())}"
            first_config = selfplay_scheduler.pick_weighted_config(node)
            config_key = f"{first_config['board_type']}_{first_config['num_players']}p" if first_config else "mixed"
            if hasattr(p2p, "_emit_batch_scheduled"):
                await p2p._emit_batch_scheduled(
                    batch_id=batch_id,
                    batch_type="selfplay",
                    config_key=config_key,
                    job_count=needed,
                    target_nodes=[p2p.node_id],
                    reason="local_job_management",
                )

            jobs_dispatched = 0
            jobs_failed = 0

            # Jan 2026 fix: Detect GPU tier for proper job type selection
            # High-end GPUs (GH200, H100, H200, A100, 5090, 4090) get GPU_SELFPLAY/GUMBEL_SELFPLAY
            # Mid-tier GPUs get HYBRID_SELFPLAY (CPU rules + GPU eval)
            gpu_name_raw = getattr(node, "gpu_name", "") or ""
            gpu_name = gpu_name_raw.upper()
            has_gpu = bool(getattr(node, "has_gpu", False))

            # Session 17.50: YAML fallback when runtime GPU detection fails
            # Runtime detection can fail on some nodes (vGPU, containers, driver issues)
            if not has_gpu or not gpu_name:
                if hasattr(p2p, "_check_yaml_gpu_config"):
                    yaml_has_gpu, yaml_gpu_name, yaml_vram = p2p._check_yaml_gpu_config()
                    if yaml_has_gpu:
                        logger.info(
                            f"LOCAL: Runtime GPU detection failed but YAML shows GPU "
                            f"({yaml_gpu_name}, {yaml_vram}GB). Using YAML config."
                        )
                        has_gpu = True
                        gpu_name_raw = yaml_gpu_name
                        gpu_name = yaml_gpu_name.upper()

            is_high_end_gpu = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
            is_apple_gpu = "MPS" in gpu_name or "APPLE" in gpu_name

            # Log GPU tier detection for visibility
            if has_gpu and is_high_end_gpu:
                logger.info(f"LOCAL: High-end GPU detected ({gpu_name_raw}) - using GPU/Gumbel selfplay")
            elif has_gpu and not is_apple_gpu:
                logger.info(f"LOCAL: Mid-tier GPU detected ({gpu_name_raw}) - using hybrid selfplay")

            for _ in range(needed):
                try:
                    # Pick a config weighted by priority (using SelfplayScheduler manager)
                    config = selfplay_scheduler.pick_weighted_config(node)
                    if config:
                        # Jan 2026: Select job type based on GPU tier (consistent with cluster dispatch)
                        if has_gpu and is_high_end_gpu and not is_apple_gpu:
                            # High-end GPUs: 50% GUMBEL (quality) / 50% GPU_SELFPLAY (volume)
                            if random.random() < 0.5:
                                job_type = self._get_job_type("GUMBEL_SELFPLAY")
                                engine_mode = "gumbel-mcts"
                            else:
                                job_type = self._get_job_type("GPU_SELFPLAY")
                                engine_mode = "gpu"
                        elif has_gpu and not is_apple_gpu:
                            # Mid-tier GPUs: HYBRID mode for rule fidelity
                            job_type = self._get_job_type("HYBRID_SELFPLAY")
                            engine_mode = "mixed"
                        else:
                            # CPU-only or Apple MPS: CPU selfplay
                            job_type = self._get_job_type("SELFPLAY")
                            engine_mode = config.get("engine_mode", "gumbel-mcts")

                        job = await self._start_local_job_via_p2p(
                            job_type,
                            board_type=config["board_type"],
                            num_players=config["num_players"],
                            engine_mode=engine_mode,
                        )
                        if job:
                            changes += 1
                            jobs_dispatched += 1
                        else:
                            jobs_failed += 1
                except Exception as e:  # noqa: BLE001
                    logger.info(f"LOCAL: Failed to start selfplay: {e}")
                    jobs_failed += 1
                    break

            # Dec 27, 2025: Emit BATCH_DISPATCHED after loop completes
            if hasattr(p2p, "_emit_batch_dispatched"):
                await p2p._emit_batch_dispatched(
                    batch_id=batch_id,
                    batch_type="selfplay",
                    config_key=config_key,
                    jobs_dispatched=jobs_dispatched,
                    jobs_failed=jobs_failed,
                    target_nodes=[p2p.node_id],
                )

        # Stop jobs if way over target (2x or more)
        elif current_jobs > target_selfplay * 2:
            excess = current_jobs - target_selfplay
            logger.info(f"LOCAL: Reducing selfplay jobs by {excess} ({current_jobs}/{target_selfplay})")
            if hasattr(p2p, "_reduce_local_selfplay_jobs"):
                await p2p._reduce_local_selfplay_jobs(target_selfplay, reason="over_target")
            changes += excess

        if changes > 0:
            logger.info(f"LOCAL job management: {changes} change(s)")
        return changes

    async def _start_local_job_via_p2p(
        self,
        job_type: Any,
        board_type: str = "",
        num_players: int = 2,
        engine_mode: str = "",
    ) -> Any:
        """Start a local job via P2P orchestrator's _start_local_job.

        This is a thin wrapper that delegates to the P2P orchestrator's
        existing _start_local_job method (which itself delegates to this
        orchestrator's start_local_job).

        Returns:
            Job object if started, None otherwise.
        """
        p2p = self._p2p
        if hasattr(p2p, "_start_local_job"):
            return await p2p._start_local_job(
                job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
            )
        return None
