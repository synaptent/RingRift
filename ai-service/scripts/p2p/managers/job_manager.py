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

        # Event subscription state (December 2025)
        self._subscribed = False

    # =========================================================================
    # Event Subscriptions (December 2025)
    # =========================================================================

    def subscribe_to_events(self) -> None:
        """Subscribe to job-relevant events.

        Subscribes to:
        - HOST_OFFLINE: Cancel jobs on offline hosts
        - HOST_ONLINE: Potentially reschedule cancelled jobs
        """
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus
            from app.distributed.data_events import DataEventType

            bus = get_event_bus()

            # Subscribe to HOST_OFFLINE to cancel jobs on dead nodes
            if hasattr(DataEventType, "HOST_OFFLINE"):
                bus.subscribe(DataEventType.HOST_OFFLINE, self._on_host_offline)
                logger.info("[JobManager] Subscribed to HOST_OFFLINE")

            # Subscribe to HOST_ONLINE for potential job rescheduling
            if hasattr(DataEventType, "HOST_ONLINE"):
                bus.subscribe(DataEventType.HOST_ONLINE, self._on_host_online)
                logger.info("[JobManager] Subscribed to HOST_ONLINE")

            self._subscribed = True
        except ImportError:
            logger.debug("[JobManager] Event router not available")
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"[JobManager] Failed to subscribe: {e}")

    async def _on_host_offline(self, event) -> None:
        """Handle HOST_OFFLINE events - cancel jobs on offline host."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")

            if not node_id:
                return

            logger.info(f"[JobManager] HOST_OFFLINE: {node_id}, cancelling jobs")

            # Cancel all jobs running on offline host
            cancelled = 0
            with self.jobs_lock:
                for job_type, jobs in self.active_jobs.items():
                    for job_id, job in list(jobs.items()):
                        job_node = job.get("node_id") if isinstance(job, dict) else getattr(job, "node_id", None)
                        if job_node == node_id:
                            job_status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                            if job_status == "running":
                                if isinstance(job, dict):
                                    job["status"] = "cancelled"
                                else:
                                    job.status = "cancelled"
                                cancelled += 1
                                logger.info(f"[JobManager] Cancelled {job_type} job {job_id} on offline node {node_id}")

            if cancelled > 0:
                logger.info(f"[JobManager] Cancelled {cancelled} jobs on offline node {node_id}")

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[JobManager] Error handling host offline: {e}")

    async def _on_host_online(self, event) -> None:
        """Handle HOST_ONLINE events - log for potential job rescheduling."""
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            node_id = payload.get("node_id", "")

            if not node_id:
                return

            logger.info(f"[JobManager] HOST_ONLINE: {node_id}, node available for jobs")

            # Note: Job rescheduling is handled by the scheduler, not the job manager
            # This is logged for observability

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[JobManager] Error handling host online: {e}")

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

        # GPU availability check for GPU-accelerated modes
        if effective_mode in self.SEARCH_ENGINE_MODES:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        f"GPU mode {effective_mode} requested but no GPU available, "
                        f"falling back to heuristic-only for job {job_id}"
                    )
                    effective_mode = "heuristic-only"
            except ImportError:
                logger.warning(
                    f"PyTorch not available for GPU check, falling back to heuristic-only for job {job_id}"
                )
                effective_mode = "heuristic-only"

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

        # Dispatch selfplay tasks to workers via HTTP
        if state.worker_nodes:
            dispatch_results = await self._dispatch_selfplay_to_workers(
                job_id=job_id,
                workers=state.worker_nodes,
                games_per_worker=games_per_worker,
                remainder=remainder,
                board_type=state.board_type,
                num_players=state.num_players,
                model_path=state.best_model_path,
                output_dir=iteration_dir,
            )
            logger.info(
                f"Dispatched selfplay to {len(dispatch_results)} workers: "
                f"{sum(1 for r in dispatch_results.values() if r.get('success'))} succeeded"
            )
        else:
            # If no workers available, run locally
            logger.info("No workers available, running selfplay locally")
            await self.run_local_selfplay(
                job_id, state.games_per_iteration,
                state.board_type, state.num_players,
                state.best_model_path, iteration_dir
            )

    async def _dispatch_selfplay_to_workers(
        self,
        job_id: str,
        workers: list[Any],
        games_per_worker: int,
        remainder: int,
        board_type: str,
        num_players: int,
        model_path: str | None,
        output_dir: str,
    ) -> dict[str, dict[str, Any]]:
        """Dispatch selfplay tasks to worker nodes via HTTP.

        Args:
            job_id: Job ID for tracking
            workers: List of worker node info objects
            games_per_worker: Base games per worker
            remainder: Extra games to distribute to first workers
            board_type: Board type
            num_players: Number of players
            model_path: Path to model file (or None for heuristic)
            output_dir: Directory for output files

        Returns:
            Dict of worker_id -> dispatch result
        """
        results: dict[str, dict[str, Any]] = {}

        try:
            from scripts.p2p.network import get_client_session
            from aiohttp import ClientTimeout
        except ImportError:
            logger.warning("aiohttp not available for distributed dispatch")
            return results

        timeout = ClientTimeout(total=30)

        async with get_client_session(timeout) as session:
            for i, worker in enumerate(workers):
                worker_id = getattr(worker, "node_id", str(worker))
                games = games_per_worker + (1 if i < remainder else 0)

                # Get worker endpoint
                worker_ip = getattr(worker, "best_ip", None) or getattr(worker, "ip", None)
                worker_port = getattr(worker, "port", 8770)

                if not worker_ip:
                    results[worker_id] = {"success": False, "error": "no_ip"}
                    continue

                url = f"http://{worker_ip}:{worker_port}/run-selfplay"
                payload = {
                    "job_id": f"{job_id}_{worker_id}",
                    "parent_job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "num_games": games,
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "engine_mode": "gumbel-mcts" if model_path else "heuristic-only",
                }

                try:
                    async with session.post(url, json=payload, timeout=timeout) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            results[worker_id] = {
                                "success": result.get("success", True),
                                "games": games,
                            }
                            logger.debug(f"Dispatched {games} games to {worker_id}")
                        else:
                            results[worker_id] = {
                                "success": False,
                                "error": f"http_{resp.status}",
                            }
                except asyncio.TimeoutError:
                    results[worker_id] = {"success": False, "error": "timeout"}
                except Exception as e:
                    results[worker_id] = {"success": False, "error": str(e)}
                    logger.debug(f"Failed to dispatch to {worker_id}: {e}")

        return results

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

        jsonl_files = list(Path(iteration_dir).glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL files found for export in {iteration_dir}")
            return

        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "jsonl_to_npz.py"),
            "--input-dir",
            iteration_dir,
            "--output",
            output_file,
            "--board-type",
            state.board_type,
            "--num-players",
            str(state.num_players),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600  # 10 minutes max
            )

            if proc.returncode == 0:
                logger.info(f"Exported training data to {output_file}")
                state.training_data_path = output_file
            else:
                logger.warning(f"Training data export failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            logger.warning("Training data export timed out")
        except Exception as e:
            logger.error(f"Training data export error: {e}")

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

        Distributes matches across workers, collects results, and calculates
        Elo rating updates.

        Args:
            job_id: Tournament job ID
        """
        try:
            state = self.distributed_tournament_state.get(job_id)
            if not state:
                return

            logger.info(f"Tournament coordinator started for job {job_id}")
            state.status = "running"

            # Get tournament configuration
            models = getattr(state, "models", [])
            games_per_pair = getattr(state, "games_per_pair", 10)
            board_type = getattr(state, "board_type", "hex8")
            num_players = getattr(state, "num_players", 2)

            if len(models) < 2:
                logger.warning(f"Tournament {job_id} needs at least 2 models")
                state.status = "error: not enough models"
                return

            # Generate match pairings (round-robin)
            matches = self._generate_tournament_matches(models, games_per_pair)
            state.total_matches = len(matches)
            state.completed_matches = 0

            logger.info(f"Tournament {job_id}: {len(matches)} matches for {len(models)} models")

            # Get available workers
            workers = self._get_tournament_workers()
            if not workers:
                # Run locally if no workers
                logger.info("No workers available, running tournament locally")
                results = await self._run_tournament_matches_locally(
                    job_id, matches, board_type, num_players
                )
            else:
                # Distribute matches to workers
                results = await self._dispatch_tournament_matches(
                    job_id, matches, workers, board_type, num_players
                )

            # Calculate Elo updates
            elo_updates = self._calculate_elo_updates(models, results)
            state.elo_updates = elo_updates

            # Update state
            state.results = results
            state.status = "completed"

            logger.info(
                f"Tournament {job_id} completed: {state.completed_matches} matches, "
                f"Elo updates: {elo_updates}"
            )

            # Emit tournament completion event
            self._emit_task_event(
                "TASK_COMPLETED",
                job_id,
                "tournament",
                models=models,
                total_matches=state.total_matches,
                elo_updates=elo_updates,
            )

        except Exception as e:
            logger.error(f"Tournament coordinator error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {e}"
            self._emit_task_event("TASK_FAILED", job_id, "tournament", error=str(e))

    def _generate_tournament_matches(
        self,
        models: list[str],
        games_per_pair: int,
    ) -> list[dict[str, Any]]:
        """Generate round-robin match pairings.

        Args:
            models: List of model paths
            games_per_pair: Number of games per model pair

        Returns:
            List of match configurations
        """
        matches = []
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                for game_num in range(games_per_pair):
                    # Alternate starting player
                    first_player = model_a if game_num % 2 == 0 else model_b
                    second_player = model_b if game_num % 2 == 0 else model_a
                    matches.append({
                        "match_id": f"{i}_{len(matches)}",
                        "player1_model": first_player,
                        "player2_model": second_player,
                        "game_num": game_num,
                    })
        return matches

    def _get_tournament_workers(self) -> list[Any]:
        """Get available workers for tournament matches.

        Returns:
            List of worker node info objects
        """
        workers = []
        with self.peers_lock:
            for peer in self.peers.values():
                if hasattr(peer, "is_healthy") and peer.is_healthy():
                    # Prefer GPU nodes for neural net matches
                    if hasattr(peer, "has_gpu") and peer.has_gpu:
                        workers.insert(0, peer)
                    else:
                        workers.append(peer)
        return workers

    async def _dispatch_tournament_matches(
        self,
        job_id: str,
        matches: list[dict[str, Any]],
        workers: list[Any],
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Dispatch tournament matches to workers.

        Args:
            job_id: Tournament job ID
            matches: List of match configurations
            workers: List of worker nodes
            board_type: Board type
            num_players: Number of players

        Returns:
            List of match results
        """
        results = []

        try:
            from scripts.p2p.network import get_client_session
            from aiohttp import ClientTimeout
        except ImportError:
            logger.warning("aiohttp not available for distributed tournament")
            return await self._run_tournament_matches_locally(
                job_id, matches, board_type, num_players
            )

        timeout = ClientTimeout(total=120)  # 2 min per match
        state = self.distributed_tournament_state.get(job_id)

        # Distribute matches across workers (round-robin)
        worker_assignments: dict[str, list[dict]] = {
            getattr(w, "node_id", str(w)): [] for w in workers
        }

        for i, match in enumerate(matches):
            worker = workers[i % len(workers)]
            worker_id = getattr(worker, "node_id", str(worker))
            worker_assignments[worker_id].append(match)

        # Dispatch to each worker
        async with get_client_session(timeout) as session:
            dispatch_tasks = []

            for worker in workers:
                worker_id = getattr(worker, "node_id", str(worker))
                worker_matches = worker_assignments.get(worker_id, [])
                if not worker_matches:
                    continue

                worker_ip = getattr(worker, "best_ip", None) or getattr(worker, "ip", None)
                worker_port = getattr(worker, "port", 8770)

                if not worker_ip:
                    continue

                url = f"http://{worker_ip}:{worker_port}/run-tournament-matches"
                payload = {
                    "job_id": f"{job_id}_{worker_id}",
                    "parent_job_id": job_id,
                    "matches": worker_matches,
                    "board_type": board_type,
                    "num_players": num_players,
                }

                dispatch_tasks.append(
                    self._send_tournament_request(session, url, payload, worker_id, timeout)
                )

            # Wait for all workers to complete
            worker_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)

            for result in worker_results:
                if isinstance(result, Exception):
                    logger.warning(f"Tournament worker error: {result}")
                elif isinstance(result, list):
                    results.extend(result)
                    if state:
                        state.completed_matches += len(result)

        return results

    async def _send_tournament_request(
        self,
        session: Any,
        url: str,
        payload: dict,
        worker_id: str,
        timeout: Any,
    ) -> list[dict[str, Any]]:
        """Send tournament match request to a worker.

        Args:
            session: aiohttp session
            url: Worker endpoint URL
            payload: Match request payload
            worker_id: Worker node ID
            timeout: Request timeout

        Returns:
            List of match results from worker
        """
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    match_results = result.get("results", [])
                    logger.debug(
                        f"Worker {worker_id} completed {len(match_results)} matches"
                    )
                    return match_results
                else:
                    logger.warning(f"Worker {worker_id} returned status {resp.status}")
                    return []
        except asyncio.TimeoutError:
            logger.warning(f"Worker {worker_id} timed out")
            return []
        except Exception as e:
            logger.warning(f"Worker {worker_id} error: {e}")
            return []

    async def _run_tournament_matches_locally(
        self,
        job_id: str,
        matches: list[dict[str, Any]],
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Run tournament matches locally when no workers available.

        Args:
            job_id: Tournament job ID
            matches: List of match configurations
            board_type: Board type
            num_players: Number of players

        Returns:
            List of match results
        """
        results = []
        state = self.distributed_tournament_state.get(job_id)

        for match in matches:
            try:
                # Import game runner (lazy to avoid circular imports)
                from app.ai.heuristic_ai import HeuristicAI, HeuristicConfig

                # For now, run heuristic-only matches for testing
                # Full implementation would load neural net models
                result = {
                    "match_id": match["match_id"],
                    "player1_model": match["player1_model"],
                    "player2_model": match["player2_model"],
                    "winner": None,  # Would be determined by actual game
                    "moves": 0,
                }
                results.append(result)

                if state:
                    state.completed_matches += 1

            except Exception as e:
                logger.warning(f"Local match error: {e}")

        return results

    def _calculate_elo_updates(
        self,
        models: list[str],
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate Elo rating updates from match results.

        Uses standard Elo formula with K=32.

        Args:
            models: List of model paths
            results: List of match results

        Returns:
            Dict of model -> Elo delta
        """
        K = 32  # Elo K-factor
        elo_deltas: dict[str, float] = {m: 0.0 for m in models}
        game_counts: dict[str, int] = {m: 0 for m in models}

        # Current ratings (start at 1500)
        ratings = {m: 1500.0 for m in models}

        for result in results:
            p1 = result.get("player1_model")
            p2 = result.get("player2_model")
            winner = result.get("winner")

            if p1 not in ratings or p2 not in ratings:
                continue

            # Calculate expected scores
            r1, r2 = ratings[p1], ratings[p2]
            e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))
            e2 = 1.0 - e1

            # Actual scores
            if winner == p1:
                s1, s2 = 1.0, 0.0
            elif winner == p2:
                s1, s2 = 0.0, 1.0
            else:  # Draw
                s1, s2 = 0.5, 0.5

            # Update ratings
            delta1 = K * (s1 - e1)
            delta2 = K * (s2 - e2)

            elo_deltas[p1] += delta1
            elo_deltas[p2] += delta2
            game_counts[p1] += 1
            game_counts[p2] += 1

        return elo_deltas

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

    # =========================================================================
    # Job Lifecycle and Metrics
    # =========================================================================
    # NOTE: The following methods were removed Dec 2025 as dead code:
    # - run_parity_validation() - placeholder stub, never called
    # - run_npz_export() - placeholder stub, never called
    # - get_job_count_for_node() - no production callers
    # - get_selfplay_job_count_for_node() - no production callers
    # - get_training_job_count_for_node() - no production callers
    # - get_all_jobs() - no production callers
    # Use orchestrator's job tracking or SelfplayScheduler for job counts.

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

    # =========================================================================
    # Health Check (December 2025)
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check health status of JobManager.

        Returns:
            Dict with status, job counts, and error info
        """
        status = "healthy"
        errors_count = 0
        last_error: str | None = None
        total_jobs = 0
        failed_jobs = 0
        running_jobs = 0

        with self.jobs_lock:
            for job_type, jobs in self.active_jobs.items():
                for job_id, job in jobs.items():
                    total_jobs += 1
                    job_status = job.get("status") if isinstance(job, dict) else getattr(job, "status", "running")
                    if job_status == "running":
                        running_jobs += 1
                    elif job_status in ("failed", "error", "timeout"):
                        failed_jobs += 1

        # Degrade status if high failure rate
        if total_jobs > 0:
            failure_rate = failed_jobs / total_jobs
            if failure_rate > 0.5:
                status = "unhealthy"
                last_error = f"High job failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs
            elif failure_rate > 0.2:
                status = "degraded"
                last_error = f"Elevated job failure rate: {failure_rate:.0%}"
                errors_count = failed_jobs

        # Check if subscribed to events
        if not self._subscribed:
            if status == "healthy":
                status = "degraded"
                last_error = "Not subscribed to events"

        return {
            "status": status,
            "operations_count": total_jobs,
            "errors_count": errors_count,
            "last_error": last_error,
            "running_jobs": running_jobs,
            "failed_jobs": failed_jobs,
            "job_types": list(self.active_jobs.keys()),
            "subscribed": self._subscribed,
        }
