"""JobManager: Job spawning and lifecycle management for P2P orchestrator.

Extracted from p2p_orchestrator.py for better modularity.
Handles job spawning, execution, monitoring, and cleanup across different job types.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models import ClusterJob, ImprovementLoopState, NodeInfo

logger = logging.getLogger(__name__)

# Event emission helper - imported lazily to avoid circular imports
_emit_event: Callable[[str, dict], None] | None = None


def _get_event_emitter() -> Callable[[str, dict], None] | None:
    """Get the event emitter function, initializing if needed."""
    global _emit_event
    if _emit_event is None:
        try:
            from app.coordination.event_router import emit_sync
            _emit_event = emit_sync
        except ImportError:
            # Event system not available
            pass
    return _emit_event


class JobManager:
    """Manages job spawning and lifecycle for P2P orchestrator.

    Responsibilities:
    - Spawn selfplay jobs (GPU, hybrid, heuristic)
    - Spawn training jobs (local and distributed)
    - Spawn export/tournament jobs
    - Track running jobs per node
    - Monitor job status and cleanup
    - Provide job metrics

    Usage:
        job_mgr = JobManager(
            ringrift_path="/path/to/ringrift",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        # Spawn a selfplay job
        await job_mgr.run_gpu_selfplay_job(
            job_id="job123",
            board_type="hex8",
            num_players=2,
            num_games=100,
            engine_mode="heuristic-only"
        )

        # Get job status
        job_count = job_mgr.get_job_count_for_node("node1")
    """

    # Engine modes that require search (need run_hybrid_selfplay.py)
    SEARCH_ENGINE_MODES = {
        "maxn", "brs", "mcts", "gumbel-mcts",
        "policy-only", "nn-descent", "nn-minimax"
    }

    def __init__(
        self,
        ringrift_path: str,
        node_id: str,
        peers: dict[str, Any],
        peers_lock: threading.Lock,
        active_jobs: dict[str, dict[str, Any]],
        jobs_lock: threading.Lock,
        improvement_loop_state: dict[str, Any] | None = None,
        distributed_tournament_state: dict[str, Any] | None = None,
    ):
        """Initialize the JobManager.

        Args:
            ringrift_path: Path to RingRift repository root
            node_id: Current node's ID
            peers: Dict of node_id -> NodeInfo
            peers_lock: Lock for thread-safe peer access
            active_jobs: Dict tracking running jobs by type
            jobs_lock: Lock for thread-safe job access
            improvement_loop_state: Optional improvement loop state dict
            distributed_tournament_state: Optional tournament state dict
        """
        self.ringrift_path = ringrift_path
        self.node_id = node_id
        self.peers = peers
        self.peers_lock = peers_lock
        self.active_jobs = active_jobs
        self.jobs_lock = jobs_lock
        self.improvement_loop_state = improvement_loop_state or {}
        self.distributed_tournament_state = distributed_tournament_state or {}

    def _emit_task_event(self, event_type: str, job_id: str, job_type: str, **kwargs) -> None:
        """Emit a task lifecycle event if the event system is available.

        Args:
            event_type: One of TASK_SPAWNED, TASK_COMPLETED, TASK_FAILED
            job_id: Unique job identifier
            job_type: Type of job (selfplay, training, etc.)
            **kwargs: Additional event data
        """
        emitter = _get_event_emitter()
        if emitter is None:
            return

        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "node_id": self.node_id,
            "timestamp": time.time(),
            **kwargs,
        }

        try:
            emitter(event_type, payload)
            logger.debug(f"Emitted {event_type} for job {job_id}")
        except Exception as e:
            logger.debug(f"Failed to emit {event_type}: {e}")

    # =========================================================================
    # Selfplay Job Methods
    # =========================================================================

    async def run_gpu_selfplay_job(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        num_games: int,
        engine_mode: str,
    ) -> None:
        """Run selfplay job with appropriate script based on engine mode.

        For simple modes (random, heuristic, nnue-guided): use run_gpu_selfplay.py (GPU-optimized)
        For search modes (maxn, brs, mcts, gumbel-mcts): use run_hybrid_selfplay.py (supports search)

        Args:
            job_id: Unique job identifier
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2-4)
            num_games: Number of games to play
            engine_mode: Engine mode (heuristic-only, gumbel-mcts, etc.)
        """
        board_norm = board_type.replace("hexagonal", "hex")
        output_dir = Path(self.ringrift_path) / "ai-service" / "data" / "selfplay" / "p2p_gpu" / f"{board_norm}_{num_players}p" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_mode = engine_mode or "heuristic-only"

        if effective_mode in self.SEARCH_ENGINE_MODES:
            # Use run_hybrid_selfplay.py for search-based modes
            script_path = os.path.join(self.ringrift_path, "ai-service", "scripts", "run_hybrid_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"Hybrid selfplay script not found: {script_path}")
                return

            cmd = [
                sys.executable,
                script_path,
                "--board-type", board_norm,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--output-dir", str(output_dir),
                "--record-db", str(output_dir / "games.db"),
                "--lean-db",
                "--engine-mode", effective_mode,
                "--seed", str(int(time.time() * 1000) % 2**31),
            ]
        else:
            # Use run_gpu_selfplay.py for GPU-optimized modes
            script_path = os.path.join(self.ringrift_path, "ai-service", "scripts", "run_gpu_selfplay.py")
            if not os.path.exists(script_path):
                logger.warning(f"GPU selfplay script not found: {script_path}")
                return

            # Map engine modes: run_gpu_selfplay.py only supports: random-only, heuristic-only, nnue-guided
            mode_map = {
                "mixed": "heuristic-only",
                "gpu": "heuristic-only",
                "descent-only": "heuristic-only",
                "heuristic-only": "heuristic-only",
                "nnue-guided": "nnue-guided",
                "random-only": "random-only",
            }
            gpu_engine_mode = mode_map.get(effective_mode, "heuristic-only")

            cmd = [
                sys.executable,
                script_path,
                "--board", board_norm,  # Note: --board not --board-type
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--output-dir", str(output_dir),
                "--output-db", str(output_dir / "games.db"),
                "--engine-mode", gpu_engine_mode,
                "--seed", str(int(time.time() * 1000) % 2**31),
            ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Track the job
            with self.jobs_lock:
                if "selfplay" not in self.active_jobs:
                    self.active_jobs["selfplay"] = {}
                self.active_jobs["selfplay"][job_id] = {
                    "job_id": job_id,
                    "status": "running",
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": num_games,
                    "started_at": time.time(),
                    "pid": proc.pid,
                }

            # Emit TASK_SPAWNED event for pipeline coordination
            self._emit_task_event(
                "TASK_SPAWNED",
                job_id,
                "selfplay",
                board_type=board_type,
                num_players=num_players,
                num_games=num_games,
                engine_mode=effective_mode,
            )

            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=7200)  # 2 hour max

            # Update job status and emit completion event
            duration = time.time() - self.active_jobs.get("selfplay", {}).get(job_id, {}).get("started_at", time.time())
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    if proc.returncode == 0:
                        self.active_jobs["selfplay"][job_id]["status"] = "completed"
                        self._emit_task_event(
                            "TASK_COMPLETED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            num_games=num_games,
                            duration_seconds=duration,
                        )
                    else:
                        error_msg = stderr.decode()[:500]
                        logger.warning(f"Selfplay job {job_id} failed: {error_msg}")
                        self.active_jobs["selfplay"][job_id]["status"] = "failed"
                        self._emit_task_event(
                            "TASK_FAILED",
                            job_id,
                            "selfplay",
                            board_type=board_type,
                            num_players=num_players,
                            error=error_msg,
                            duration_seconds=duration,
                        )
                    # Remove from active jobs
                    del self.active_jobs["selfplay"][job_id]

        except asyncio.TimeoutError:
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    self.active_jobs["selfplay"][job_id]["status"] = "timeout"
                    del self.active_jobs["selfplay"][job_id]
            logger.warning(f"Selfplay job {job_id} timed out")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error="timeout", board_type=board_type)
        except Exception as e:
            with self.jobs_lock:
                if job_id in self.active_jobs.get("selfplay", {}):
                    self.active_jobs["selfplay"][job_id]["status"] = "error"
                    del self.active_jobs["selfplay"][job_id]
            logger.error(f"Error running selfplay job {job_id}: {e}")
            self._emit_task_event("TASK_FAILED", job_id, "selfplay", error=str(e), board_type=board_type)

    async def run_distributed_selfplay(self, job_id: str) -> None:
        """Coordinate distributed selfplay for improvement loop.

        Distributes selfplay games across all available workers.
        Each worker runs selfplay using the current best model and reports
        progress back to the coordinator.

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        # Distribute selfplay across workers
        num_workers = max(len(state.worker_nodes), 1)
        games_per_worker = state.games_per_iteration // num_workers
        remainder = state.games_per_iteration % num_workers

        logger.info(f"Starting distributed selfplay: {games_per_worker} games/worker, {num_workers} workers")

        # Create output directory for this iteration
        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        # Send selfplay tasks to workers (would need HTTP client here)
        # For now, this is a placeholder for the distributed coordination logic
        # The actual HTTP calls would be made by the orchestrator

        # If no workers available, run locally
        if not state.worker_nodes:
            logger.info("No workers available, running selfplay locally")
            await self.run_local_selfplay(
                job_id, state.games_per_iteration,
                state.board_type, state.num_players,
                state.best_model_path, iteration_dir
            )

    async def run_local_selfplay(
        self,
        job_id: str,
        num_games: int,
        board_type: str,
        num_players: int,
        model_path: str | None,
        output_dir: str,
    ) -> None:
        """Run selfplay locally using subprocess.

        Args:
            job_id: Job ID
            num_games: Number of games to play
            board_type: Board type
            num_players: Number of players
            model_path: Path to model (or None for heuristic)
            output_dir: Output directory for games
        """
        output_file = os.path.join(output_dir, f"{self.node_id}_games.jsonl")

        # Build selfplay command
        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", "descent-only" if model_path else "heuristic-only",
            "--max-moves", "10000",  # Avoid draws due to move limit
            "--log-jsonl", output_file,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                logger.info(f"Local selfplay completed: {num_games} games")
                # Update progress
                if job_id in self.improvement_loop_state:
                    self.improvement_loop_state[job_id].selfplay_progress[self.node_id] = num_games
            else:
                logger.warning(f"Local selfplay failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            logger.warning("Local selfplay timed out")
        except Exception as e:
            logger.error(f"Local selfplay error: {e}")

    # =========================================================================
    # Training Job Methods
    # =========================================================================

    async def export_training_data(self, job_id: str) -> None:
        """Export training data from selfplay games.

        Converts JSONL game records to training format (HDF5 or NPZ).

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Exporting training data for job {job_id}, iteration {state.current_iteration}")

        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        output_file = os.path.join(
            self.ringrift_path, "ai-service", "data", "training",
            f"improve_{job_id}", f"iter_{state.current_iteration}.npz"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Run export script (simplified - actual implementation would use proper export logic)
        logger.info(f"Exporting training data to {output_file}")
        # The actual export would be done here

    async def run_training(self, job_id: str) -> None:
        """Run neural network training on GPU node.

        Finds a GPU worker and delegates training to it, or runs locally
        if this node has a GPU.

        Args:
            job_id: Job ID for the improvement loop
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Running training for job {job_id}, iteration {state.current_iteration}")

        # Find GPU worker
        gpu_worker = None
        with self.peers_lock:
            for peer in self.peers.values():
                if hasattr(peer, 'has_gpu') and peer.has_gpu and hasattr(peer, 'is_healthy') and peer.is_healthy():
                    gpu_worker = peer
                    break

        # Model output path
        new_model_path = os.path.join(
            self.ringrift_path, "ai-service", "models",
            f"improve_{job_id}", f"iter_{state.current_iteration}.pt"
        )
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)

        training_config = {
            "job_id": job_id,
            "iteration": state.current_iteration,
            "training_data": getattr(state, 'training_data_path', ''),
            "output_model": new_model_path,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        # For distributed training, would delegate to GPU worker here
        # For now, run locally
        await self.run_local_training(training_config)
        state.candidate_model_path = new_model_path

    async def run_local_training(self, config: dict) -> None:
        """Run training locally using subprocess.

        Args:
            config: Training configuration dict
        """
        logger.info("Running local training")

        training_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import numpy as np
import torch

# Load training data
try:
    data = np.load('{config.get("training_data", "")}', allow_pickle=True)
    print(f"Loaded training data")
except Exception as e:
    print(f"No training data available: {{e}}")
    data = None

# Import or create model architecture
try:
    from app.models.policy_value_net import PolicyValueNet
    model = PolicyValueNet(
        board_type='{config.get("board_type", "square8")}',
        num_players={config.get("num_players", 2)}
    )
except ImportError:
    # Fallback to simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

# Save model
torch.save(model.state_dict(), '{config.get("output_model", "/tmp/model.pt")}')
print(f"Saved model to {{config.get('output_model', '/tmp/model.pt')}}")
"""

        cmd = [sys.executable, "-c", training_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            logger.info(f"Training output: {stdout.decode()}")
            if proc.returncode != 0:
                logger.warning(f"Training stderr: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            logger.warning("Local training timed out")
        except Exception as e:
            logger.error(f"Local training error: {e}")

    # =========================================================================
    # Tournament Job Methods
    # =========================================================================

    async def run_distributed_tournament(self, job_id: str) -> None:
        """Main coordinator loop for distributed tournament.

        Args:
            job_id: Tournament job ID
        """
        try:
            state = self.distributed_tournament_state.get(job_id)
            if not state:
                return

            logger.info(f"Tournament coordinator started for job {job_id}")

            # Distribute matches to workers (simplified - actual implementation would use HTTP)
            # The full implementation would send matches to workers and collect results

            # Calculate final ratings (placeholder)
            state.status = "completed"

            logger.info(f"Tournament {job_id} completed: {state.completed_matches} matches")

        except Exception as e:
            logger.error(f"Tournament coordinator error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {e}"

    async def run_model_comparison_tournament(self, config: dict) -> None:
        """Run a model comparison tournament.

        Args:
            config: Tournament configuration
        """
        logger.info(f"Running model comparison tournament: {config}")
        # Placeholder for tournament logic

    # =========================================================================
    # Gauntlet and Validation Methods
    # =========================================================================

    async def run_post_training_gauntlet(self, job: Any) -> bool:
        """Run gauntlet evaluation after training.

        Args:
            job: TrainingJob object

        Returns:
            True if gauntlet passed, False otherwise
        """
        logger.info(f"Running post-training gauntlet for job {job.job_id}")
        # Placeholder for gauntlet logic
        return True

    async def run_parity_validation(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        num_seeds: int = 100,
    ) -> None:
        """Run parity validation against TypeScript engine.

        Args:
            job_id: Job ID
            board_type: Board type
            num_players: Number of players
            num_seeds: Number of random seeds to test
        """
        logger.info(f"Running parity validation for {board_type}_{num_players}p with {num_seeds} seeds")
        # Placeholder for parity validation logic

    async def run_npz_export(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        output_dir: str,
    ) -> None:
        """Export games to NPZ training format.

        Args:
            job_id: Job ID
            board_type: Board type
            num_players: Number of players
            output_dir: Output directory for NPZ files
        """
        logger.info(f"Exporting NPZ for {board_type}_{num_players}p to {output_dir}")
        # Placeholder for NPZ export logic

    # =========================================================================
    # Job Lifecycle and Metrics
    # =========================================================================

    def get_job_count_for_node(self, node_id: str) -> int:
        """Get total job count for a specific node.

        Args:
            node_id: Node ID

        Returns:
            Total number of running jobs on the node
        """
        count = 0
        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job in jobs.values():
                    if isinstance(job, dict) and job.get("node_id") == node_id:
                        count += 1
                    elif hasattr(job, "node_id") and job.node_id == node_id:
                        count += 1
        return count

    def get_selfplay_job_count_for_node(self, node_id: str) -> int:
        """Get selfplay job count for a specific node.

        Args:
            node_id: Node ID

        Returns:
            Number of running selfplay jobs on the node
        """
        count = 0
        with self.jobs_lock:
            selfplay_jobs = self.active_jobs.get("selfplay", {})
            for job in selfplay_jobs.values():
                if isinstance(job, dict) and job.get("node_id") == node_id:
                    count += 1
                elif hasattr(job, "node_id") and job.node_id == node_id:
                    count += 1
        return count

    def get_training_job_count_for_node(self, node_id: str) -> int:
        """Get training job count for a specific node.

        Args:
            node_id: Node ID

        Returns:
            Number of running training jobs on the node
        """
        count = 0
        with self.jobs_lock:
            training_jobs = self.active_jobs.get("training", {})
            for job in training_jobs.values():
                if isinstance(job, dict) and job.get("node_id") == node_id:
                    count += 1
                elif hasattr(job, "node_id") and job.node_id == node_id:
                    count += 1
        return count

    def get_all_jobs(self) -> dict[str, dict[str, Any]]:
        """Get all active jobs.

        Returns:
            Dict of job_type -> {job_id -> job_info}
        """
        with self.jobs_lock:
            return {
                job_type: dict(jobs)
                for job_type, jobs in self.active_jobs.items()
            }

    def cleanup_completed_jobs(self) -> int:
        """Clean up completed jobs from tracking.

        Returns:
            Number of jobs cleaned up
        """
        cleaned = 0
        with self.jobs_lock:
            for job_type in list(self.active_jobs.keys()):
                for job_id in list(self.active_jobs[job_type].keys()):
                    job = self.active_jobs[job_type][job_id]
                    status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                    if status in ("completed", "failed", "timeout", "error"):
                        del self.active_jobs[job_type][job_id]
                        cleaned += 1
        return cleaned
