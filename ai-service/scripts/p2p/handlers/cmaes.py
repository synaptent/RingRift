"""CMA-ES HTTP Handlers Mixin.

Provides HTTP endpoints for distributed CMA-ES (Covariance Matrix Adaptation
Evolution Strategy) hyperparameter optimization. Enables parallel evaluation
of candidate solutions across cluster nodes.

Inherits from BaseP2PHandler for standardized response formatting.

Usage:
    class P2POrchestrator(CMAESHandlersMixin, ...):
        pass

Endpoints:
    POST /cmaes/start - Start distributed CMA-ES optimization (leader only)
    GET /cmaes/status - Get optimization run status and progress
    POST /cmaes/evaluate - Worker evaluates candidate weights
    POST /cmaes/report - Worker reports evaluation results
    POST /cmaes/stop - Stop running optimization

CMA-ES Workflow:
    1. Leader starts optimization via /cmaes/start with search space
    2. Each generation, leader samples population of candidate weights
    3. Candidates distributed to workers via /cmaes/evaluate
    4. Workers play games and report fitness via /cmaes/report
    5. Leader updates covariance matrix and samples next generation
    6. Converges to optimal hyperparameters (NNUE weights, search params)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_TOURNAMENT

if TYPE_CHECKING:
    from scripts.p2p.models import NodeRole

logger = logging.getLogger(__name__)


class CMAESHandlersMixin(BaseP2PHandler):
    """Mixin providing distributed CMA-ES HTTP handlers.

    Inherits from BaseP2PHandler for standardized response formatting
    (json_response, error_response).

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: str
    - peers: dict
    - peers_lock: threading.Lock
    - distributed_cmaes_state: dict
    - _run_distributed_cmaes() method
    - _evaluate_cmaes_weights() method
    """

    # Type hints for IDE support
    node_id: str
    role: Any  # NodeRole
    leader_id: str
    peers: dict
    peers_lock: Any
    distributed_cmaes_state: dict
    self_info: Any  # SelfInfo
    training_lock: Any  # threading.Lock
    training_jobs: dict

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cmaes_start(self, request: web.Request) -> web.Response:
        """Start a distributed CMA-ES optimization job.

        Only the leader can start distributed CMA-ES jobs.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "generations": 100,
            "population_size": 20,
            "games_per_eval": 50
        }
        """
        from scripts.p2p.models import DistributedCMAESState
        from scripts.p2p.types import NodeRole

        try:
            if self.role != NodeRole.LEADER:
                return self.error_response(
                    "Only the leader can start distributed CMA-ES",
                    status=403,
                    details={"leader_id": self.leader_id},
                )

            data = await request.json()
            job_id = f"cmaes_{uuid.uuid4().hex[:8]}"

            # Create state for this job
            state = DistributedCMAESState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                generations=data.get("generations", 100),
                population_size=data.get("population_size", 20),
                games_per_eval=data.get("games_per_eval", 50),
                status="starting",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available GPU workers
            with self.peers_lock:
                gpu_nodes = [
                    p.node_id for p in self.peers.values()
                    if p.is_healthy() and p.has_gpu
                ]
            state.worker_nodes = gpu_nodes

            if not state.worker_nodes:
                return self.error_response(
                    "No GPU workers available for CMA-ES",
                    status=503,
                )

            self.distributed_cmaes_state[job_id] = state
            state.status = "running"

            logger.info(f"Started distributed CMA-ES job {job_id} with {len(state.worker_nodes)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_cmaes(job_id))

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "workers": state.worker_nodes,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "generations": state.generations,
                    "population_size": state.population_size,
                    "games_per_eval": state.games_per_eval,
                },
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cmaes_evaluate(self, request: web.Request) -> web.Response:
        """Request evaluation of weights from workers.

        Called by the coordinator to distribute weight evaluation tasks.
        Workers respond via /cmaes/result endpoint.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            weights = data.get("weights", {})
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)

            if not job_id:
                return self.error_response("job_id required", status=400)

            # Extract evaluation parameters from request
            games_per_eval = data.get("games_per_eval", 5)
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            # Store evaluation task for local processing
            logger.info(f"Received CMA-ES evaluation request: job={job_id}, gen={generation}, idx={individual_idx}")

            # Start evaluation in background
            asyncio.create_task(self._evaluate_cmaes_weights(
                job_id, weights, generation, individual_idx,
                games_per_eval=games_per_eval, board_type=board_type, num_players=num_players
            ))

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "status": "evaluation_started",
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cmaes_status(self, request: web.Request) -> web.Response:
        """Get status of distributed CMA-ES jobs."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_cmaes_state:
                    return self.error_response("Job not found", status=404)
                state = self.distributed_cmaes_state[job_id]
                return self.json_response(state.to_dict())

            # Return all jobs
            return self.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_cmaes_state.items()
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cmaes_result(self, request: web.Request) -> web.Response:
        """Receive evaluation result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)
            fitness = data.get("fitness", 0.0)
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_cmaes_state:
                return self.error_response("Job not found", status=404)

            logger.info(f"CMA-ES result: job={job_id}, gen={generation}, idx={individual_idx}, fitness={fitness:.4f} from {worker_id}")

            # Store result - the coordinator loop will process it
            state = self.distributed_cmaes_state[job_id]
            state.last_update = time.time()

            # Store result keyed by generation and index for coordinator to collect
            result_key = f"{generation}_{individual_idx}"
            state.pending_results[result_key] = fitness

            # Update best if applicable
            if fitness > state.best_fitness:
                state.best_fitness = fitness
                state.best_weights = data.get("weights", {})

            return self.json_response({
                "success": True,
                "job_id": job_id,
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cmaes_start_auto(self, request: web.Request) -> web.Response:
        """Handle CMA-ES optimization start request.

        Uses distributed GPU CMA-ES across all cluster GPU nodes for maximum throughput.
        Falls back to local GPU CMA-ES if no remote workers available.

        January 2026: Moved from p2p_orchestrator.py to CMAESHandlersMixin.
        """
        from scripts.p2p.models import DistributedCMAESState

        try:
            data = await request.json()
            job_id = data.get("job_id")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            # Check for available GPU workers in the cluster
            gpu_workers = []
            with self.peers_lock:
                for peer in self.peers.values():
                    if peer.is_healthy() and peer.has_gpu and peer.node_id != self.node_id:
                        gpu_workers.append(peer)

            # Include self if we have GPU
            if self.self_info.has_gpu:
                gpu_workers.append(self.self_info)

            if len(gpu_workers) >= 2:
                # DISTRIBUTED MODE: Use P2P distributed CMA-ES across cluster
                logger.info(f"Starting DISTRIBUTED GPU CMA-ES with {len(gpu_workers)} workers")

                # Create distributed CMA-ES state
                cmaes_job_id = f"cmaes_auto_{job_id}_{int(time.time())}"
                state = DistributedCMAESState(
                    job_id=cmaes_job_id,
                    board_type=board_type,
                    num_players=num_players,
                    generations=100,  # More generations for better optimization
                    population_size=max(32, len(gpu_workers) * 8),  # Scale with workers
                    games_per_eval=100,  # More games for accurate fitness
                    status="running",
                    started_at=time.time(),
                    last_update=time.time(),
                    worker_nodes=[w.node_id for w in gpu_workers],
                )
                self.distributed_cmaes_state[cmaes_job_id] = state

                # Launch distributed coordinator task
                asyncio.create_task(self._run_distributed_cmaes(cmaes_job_id))

                # Track as training job
                with self.training_lock:
                    if job_id in self.training_jobs:
                        self.training_jobs[job_id].status = "running"
                        self.training_jobs[job_id].started_at = time.time()

                return self.json_response({
                    "success": True,
                    "mode": "distributed",
                    "job_id": cmaes_job_id,
                    "workers": [w.node_id for w in gpu_workers],
                })

            else:
                # LOCAL MODE: Run GPU CMA-ES on this node only
                logger.info("Starting LOCAL GPU CMA-ES (no remote workers available)")

                output_dir = os.path.join(
                    self._get_ai_service_path(), "data", "cmaes",
                    f"{board_type}_{num_players}p_auto_{int(time.time())}"
                )
                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    sys.executable,
                    os.path.join(self._get_ai_service_path(), "scripts", "run_gpu_cmaes.py"),
                    "--board", board_type,
                    "--num-players", str(num_players),
                    "--generations", "100",
                    "--population-size", "32",
                    "--games-per-eval", "100",
                    "--max-moves", "10000",
                    "--output-dir", output_dir,
                    "--multi-gpu",
                ]

                env = os.environ.copy()
                env["PYTHONPATH"] = self._get_ai_service_path()
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )

                logger.info(f"Started local GPU CMA-ES (PID {proc.pid}) for job {job_id}")
                asyncio.create_task(self._monitor_training_process(job_id, proc, output_dir))

                return self.json_response({
                    "success": True,
                    "mode": "local",
                    "pid": proc.pid,
                })

        except Exception as e:  # noqa: BLE001
            return self.error_response(str(e), status=500)
