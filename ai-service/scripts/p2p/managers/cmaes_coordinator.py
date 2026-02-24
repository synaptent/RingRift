"""CMA-ES Coordinator Manager.

January 2026 - Extracted from p2p_orchestrator.py as part of aggressive decomposition.

Coordinates distributed CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
hyperparameter optimization across cluster GPU nodes.

CMA-ES Workflow:
    1. Leader starts optimization via /cmaes/start with search space
    2. Each generation, leader samples population of candidate weights
    3. Candidates distributed to workers via /cmaes/evaluate
    4. Workers play games and report fitness via /cmaes/report
    5. Leader updates covariance matrix and samples next generation
    6. Converges to optimal hyperparameters (heuristic weights, search params)

Dependencies are injected via callbacks to avoid tight coupling to orchestrator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)

# Singleton instance
_cmaes_coordinator: CMAESCoordinator | None = None


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES coordinator."""

    ai_service_path: str = ""
    default_generations: int = 100
    default_population_size: int = 20
    default_games_per_eval: int = 50
    evaluation_timeout: float = 300.0  # 5 minutes per evaluation
    generation_timeout: float = 300.0  # 5 minutes per generation


@dataclass
class CMAESJobState:
    """State for a distributed CMA-ES job."""

    job_id: str
    board_type: str
    num_players: int
    generations: int
    population_size: int
    games_per_eval: int
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0
    current_generation: int = 0
    best_fitness: float = 0.0
    best_weights: dict[str, float] = field(default_factory=dict)
    worker_nodes: list[str] = field(default_factory=list)
    pending_results: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "generations": self.generations,
            "population_size": self.population_size,
            "games_per_eval": self.games_per_eval,
            "status": self.status,
            "started_at": self.started_at,
            "last_update": self.last_update,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "best_weights": self.best_weights,
            "worker_nodes": self.worker_nodes,
        }


@dataclass
class CMAESStats:
    """Statistics for CMA-ES coordinator."""

    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_generations: int = 0
    total_evaluations: int = 0
    best_fitness_ever: float = 0.0


class CMAESCoordinator:
    """Coordinates distributed CMA-ES hyperparameter optimization.

    Uses callback injection to interact with P2P infrastructure without
    tight coupling to the orchestrator.

    Callbacks:
        get_gpu_workers: () -> list[NodeInfo] - Get available GPU workers
        send_to_worker: (worker_id, endpoint, payload) -> bool - Send request to worker
        report_to_leader: (endpoint, payload) -> bool - Report result to leader
        get_node_role: () -> str - Get current node role ("leader" or "worker")
        get_leader_id: () -> str | None - Get current leader ID
        get_node_id: () -> str - Get current node ID
    """

    def __init__(
        self,
        config: CMAESConfig | None = None,
        *,
        get_gpu_workers: Callable[[], list[Any]] | None = None,
        send_to_worker: Callable[[str, str, dict], Any] | None = None,
        report_to_leader: Callable[[str, dict], Any] | None = None,
        get_node_role: Callable[[], str] | None = None,
        get_leader_id: Callable[[], str | None] | None = None,
        get_node_id: Callable[[], str] | None = None,
        handle_cmaes_complete: Callable[[str, int, dict], str | None] | None = None,
    ):
        self.config = config or CMAESConfig()
        self._jobs: dict[str, CMAESJobState] = {}
        self._stats = CMAESStats()
        self._eval_semaphore = asyncio.Semaphore(1)

        # Callback injections
        self._get_gpu_workers = get_gpu_workers
        self._send_to_worker = send_to_worker
        self._report_to_leader = report_to_leader
        self._get_node_role = get_node_role
        self._get_leader_id = get_leader_id
        self._get_node_id = get_node_id
        self._handle_cmaes_complete = handle_cmaes_complete

        # Auto-tuning state
        self._auto_tuners: dict[str, Any] = {}
        self._last_elo: dict[str, float] = {}
        self._weight_configs: dict[str, dict] = {}

    @property
    def jobs(self) -> dict[str, CMAESJobState]:
        """Get all CMA-ES jobs."""
        return self._jobs

    @property
    def stats(self) -> CMAESStats:
        """Get CMA-ES statistics."""
        return self._stats

    def get_job(self, job_id: str) -> CMAESJobState | None:
        """Get a specific job by ID."""
        return self._jobs.get(job_id)

    def create_job(
        self,
        job_id: str,
        board_type: str,
        num_players: int,
        generations: int | None = None,
        population_size: int | None = None,
        games_per_eval: int | None = None,
    ) -> CMAESJobState:
        """Create a new CMA-ES job."""
        state = CMAESJobState(
            job_id=job_id,
            board_type=board_type,
            num_players=num_players,
            generations=generations or self.config.default_generations,
            population_size=population_size or self.config.default_population_size,
            games_per_eval=games_per_eval or self.config.default_games_per_eval,
            status="pending",
            started_at=time.time(),
            last_update=time.time(),
        )
        self._jobs[job_id] = state
        self._stats.total_jobs += 1
        return state

    async def run_distributed_cmaes(self, job_id: str) -> None:
        """Main coordinator loop for distributed CMA-ES.

        Integrates with CMA-ES algorithm to optimize heuristic weights.
        Distributes candidate evaluation across GPU workers in the cluster.
        """
        try:
            state = self._jobs.get(job_id)
            if not state:
                return

            logger.info(f"CMA-ES coordinator started for job {job_id}")
            logger.info(
                f"Config: {state.generations} gens, pop={state.population_size}, "
                f"{state.games_per_eval} games/eval"
            )

            # Try to import CMA-ES library
            try:
                import cma
                import numpy as np
            except ImportError:
                logger.info("CMA-ES requires: pip install cma numpy")
                state.status = "error: cma not installed"
                self._stats.failed_jobs += 1
                return

            # Default heuristic weights to optimize
            weight_names = [
                "material_weight",
                "ring_count_weight",
                "stack_height_weight",
                "center_control_weight",
                "territory_weight",
                "mobility_weight",
                "line_potential_weight",
                "defensive_weight",
            ]
            default_weights = {
                "material_weight": 1.0,
                "ring_count_weight": 0.5,
                "stack_height_weight": 0.3,
                "center_control_weight": 0.4,
                "territory_weight": 0.8,
                "mobility_weight": 0.2,
                "line_potential_weight": 0.6,
                "defensive_weight": 0.3,
            }

            # Convert to vector for CMA-ES
            x0 = np.array([default_weights[n] for n in weight_names])

            # Initialize CMA-ES
            es = cma.CMAEvolutionStrategy(
                x0,
                0.5,
                {
                    "popsize": state.population_size,
                    "maxiter": state.generations,
                    "bounds": [0, 2],  # Weights between 0 and 2
                },
            )

            state.current_generation = 0
            state.status = "running"

            while not es.stop() and state.status == "running":
                state.current_generation += 1
                state.last_update = time.time()
                self._stats.total_generations += 1

                # Get candidate solutions
                solutions = es.ask()

                # Distribute evaluations across workers
                fitness_results = {}

                for idx, sol in enumerate(solutions):
                    weights = {name: float(sol[i]) for i, name in enumerate(weight_names)}

                    # Round-robin assign to workers
                    if state.worker_nodes:
                        worker_idx = idx % len(state.worker_nodes)
                        worker_id = state.worker_nodes[worker_idx]

                        # Send evaluation request to worker
                        if self._send_to_worker:
                            try:
                                await self._send_to_worker(
                                    worker_id,
                                    "/cmaes/evaluate",
                                    {
                                        "job_id": job_id,
                                        "weights": weights,
                                        "generation": state.current_generation,
                                        "individual_idx": idx,
                                        "games_per_eval": state.games_per_eval,
                                        "board_type": state.board_type,
                                        "num_players": state.num_players,
                                    },
                                )
                                self._stats.total_evaluations += 1
                            except Exception as e:  # noqa: BLE001
                                logger.error(f"Failed to send eval to {worker_id}: {e}")
                                # Fall back to local evaluation
                                fitness = await self.evaluate_weights_local(
                                    weights,
                                    state.games_per_eval,
                                    state.board_type,
                                    state.num_players,
                                )
                                fitness_results[idx] = fitness
                    else:
                        # No workers, evaluate locally
                        fitness = await self.evaluate_weights_local(
                            weights,
                            state.games_per_eval,
                            state.board_type,
                            state.num_players,
                        )
                        fitness_results[idx] = fitness

                # Wait for results with timeout
                wait_start = time.time()
                while (
                    len(fitness_results) < len(solutions)
                    and (time.time() - wait_start) < self.config.generation_timeout
                ):
                    await asyncio.sleep(1)
                    state.last_update = time.time()

                    # Check for results from workers
                    for idx in range(len(solutions)):
                        if idx in fitness_results:
                            continue
                        result_key = f"{state.current_generation}_{idx}"
                        if result_key in state.pending_results:
                            fitness_results[idx] = state.pending_results[result_key]
                            del state.pending_results[result_key]

                    # Progress logging
                    elapsed = time.time() - wait_start
                    if int(elapsed) % 30 == 0 and elapsed > 1:
                        received = len(fitness_results)
                        logger.info(
                            f"Gen {state.current_generation}: {received}/{len(solutions)} "
                            f"results received ({elapsed:.0f}s elapsed)"
                        )

                # Fill in missing results with default fitness
                fitnesses = []
                for idx in range(len(solutions)):
                    fitness = fitness_results.get(idx, 0.5)
                    fitnesses.append(-fitness)  # CMA-ES minimizes, so negate

                # Update CMA-ES
                es.tell(solutions, fitnesses)

                # Track best
                best_idx = np.argmin(fitnesses)
                if -fitnesses[best_idx] > state.best_fitness:
                    state.best_fitness = -fitnesses[best_idx]
                    state.best_weights = {
                        name: float(solutions[best_idx][i])
                        for i, name in enumerate(weight_names)
                    }

                logger.info(f"Gen {state.current_generation}: best_fitness={state.best_fitness:.4f}")

            state.status = "completed"
            self._stats.completed_jobs += 1
            logger.info(f"CMA-ES job {job_id} completed: best_fitness={state.best_fitness:.4f}")
            logger.info(f"Best weights: {state.best_weights}")

            # Update best fitness ever
            if state.best_fitness > self._stats.best_fitness_ever:
                self._stats.best_fitness_ever = state.best_fitness

            # Feed results back to improvement cycle manager
            if self._handle_cmaes_complete and state.best_weights:
                try:
                    agent_id = self._handle_cmaes_complete(
                        state.board_type, state.num_players, state.best_weights
                    )
                    logger.info(f"CMA-ES weights registered as agent: {agent_id}")

                    # Save weights to file
                    await self._save_weights(
                        state.board_type,
                        state.num_players,
                        state.best_weights,
                        state.best_fitness,
                        job_id,
                        state.current_generation,
                    )

                    # Propagate to selfplay workers
                    await self.propagate_weights(
                        state.board_type, state.num_players, state.best_weights
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to register CMA-ES weights: {e}")

        except Exception as e:  # noqa: BLE001
            import traceback

            logger.info(f"CMA-ES coordinator error: {e}")
            traceback.print_exc()
            if job_id in self._jobs:
                self._jobs[job_id].status = f"error: {e}"
            self._stats.failed_jobs += 1

    async def evaluate_weights_local(
        self,
        weights: dict,
        num_games: int,
        board_type: str,
        num_players: int,
    ) -> float:
        """Evaluate weights locally by running selfplay games."""
        try:
            async with self._eval_semaphore:
                import tempfile

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(weights, f)
                    weights_file = f.name

                ai_service_path = self.config.ai_service_path or str(
                    Path(__file__).parent.parent.parent.parent
                )
                cmd = [
                    sys.executable,
                    "-c",
                    f"""
import sys
sys.path.insert(0, '{ai_service_path}')
from app.game_engine import GameEngine
from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, BoardType, GameStatus
from app.training.generate_data import create_initial_state
import json

weights = json.load(open('{weights_file}'))
board_type = BoardType('{board_type}')
wins = 0
total = {num_games}

for i in range(total):
    state = create_initial_state(board_type, num_players={num_players})
    engine = GameEngine()

    config_candidate = AIConfig(difficulty=5, randomness=0.1, think_time=500, custom_weights=weights)
    config_baseline = AIConfig(difficulty=5, randomness=0.1, think_time=500)

    ai_candidate = HeuristicAI(1, config_candidate)
    ai_baseline = HeuristicAI(2, config_baseline)

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < 300:
        current_ai = ai_candidate if state.current_player == 1 else ai_baseline
        move = current_ai.select_move(state)
        if move is None:
            break
        state = engine.apply_move(state, move)
        move_count += 1

    if state.winner == 1:
        wins += 1
    elif state.winner is None:
        wins += 0.5  # Draw

print(wins / total)
""",
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "PYTHONPATH": ai_service_path},
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.evaluation_timeout
                )

                # Clean up temp file
                os.unlink(weights_file)

                if proc.returncode == 0:
                    return float(stdout.decode().strip())
                else:
                    logger.info(f"Local eval error: {stderr.decode()}")
                    return 0.5

        except Exception as e:  # noqa: BLE001
            logger.info(f"Local CMA-ES evaluation error: {e}")
            return 0.5

    async def evaluate_weights(
        self,
        job_id: str,
        weights: dict,
        generation: int,
        individual_idx: int,
        games_per_eval: int = 5,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> None:
        """Evaluate weights locally and report result to coordinator."""
        try:
            fitness = await self.evaluate_weights_local(
                weights, games_per_eval, board_type, num_players
            )

            logger.info(
                f"Completed local CMA-ES evaluation: job={job_id}, gen={generation}, "
                f"idx={individual_idx}, fitness={fitness:.4f}"
            )

            # If we're not the coordinator, report result back
            node_role = self._get_node_role() if self._get_node_role else "worker"
            leader_id = self._get_leader_id() if self._get_leader_id else None
            node_id = self._get_node_id() if self._get_node_id else "unknown"

            if node_role != "leader" and leader_id and self._report_to_leader:
                try:
                    await self._report_to_leader(
                        "/cmaes/result",
                        {
                            "job_id": job_id,
                            "generation": generation,
                            "individual_idx": individual_idx,
                            "fitness": fitness,
                            "weights": weights,
                            "worker_id": node_id,
                        },
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to report CMA-ES result to leader: {e}")

        except Exception as e:  # noqa: BLE001
            logger.info(f"CMA-ES evaluation error: {e}")

    async def trigger_auto_cmaes(self, board_type: str, num_players: int) -> None:
        """Automatically trigger CMA-ES optimization for a configuration."""
        try:
            job_id = f"auto_cmaes_{board_type}_{num_players}p_{int(time.time())}"
            logger.info(f"Auto-triggering CMA-ES: {job_id}")

            # Get GPU workers via callback
            gpu_workers = self._get_gpu_workers() if self._get_gpu_workers else []

            if len(gpu_workers) >= 2:
                # DISTRIBUTED MODE
                state = self.create_job(
                    job_id=job_id,
                    board_type=board_type,
                    num_players=num_players,
                    generations=100,
                    population_size=max(32, len(gpu_workers) * 8),
                    games_per_eval=100,
                )
                state.status = "running"
                state.worker_nodes = [getattr(w, "node_id", str(w)) for w in gpu_workers]
                safe_create_task(self.run_distributed_cmaes(job_id), name="cmaes-coordinator-run")
                logger.info(f"Started distributed CMA-ES with {len(gpu_workers)} workers")
            else:
                # LOCAL MODE - use GPU CMA-ES script
                ai_service_path = self.config.ai_service_path or str(
                    Path(__file__).parent.parent.parent.parent
                )
                output_dir = os.path.join(
                    ai_service_path,
                    "data",
                    "cmaes",
                    f"{board_type}_{num_players}p_auto_{int(time.time())}",
                )
                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    sys.executable,
                    os.path.join(ai_service_path, "scripts", "run_gpu_cmaes.py"),
                    "--board",
                    board_type,
                    "--num-players",
                    str(num_players),
                    "--generations",
                    "100",
                    "--population-size",
                    "32",
                    "--games-per-eval",
                    "100",
                    "--max-moves",
                    "10000",
                    "--output-dir",
                    output_dir,
                    "--multi-gpu",
                ]

                env = os.environ.copy()
                env["PYTHONPATH"] = ai_service_path
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                logger.info(f"Started local CMA-ES optimization (PID {proc.pid})")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Auto CMA-ES trigger failed: {e}")

    async def check_auto_tuning(self, config_key: str) -> None:
        """Check if CMA-ES auto-tuning should be triggered for a config.

        Monitors Elo progression and triggers hyperparameter optimization
        when the model's improvement plateaus.
        """
        if config_key not in self._auto_tuners:
            return

        try:
            # Try to get Elo from database
            try:
                from app.tournament import get_elo_database

                db = get_elo_database()
            except ImportError:
                return

            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Find best model for this config
            best_model = None
            best_elo = 1200.0  # Default starting Elo

            ai_service_path = self.config.ai_service_path or str(
                Path(__file__).parent.parent.parent.parent
            )
            models_dir = Path(ai_service_path) / "models" / "nnue"
            pattern = f"nnue_{board_type}_{num_players}p*.pt"

            for model_path in models_dir.glob(pattern):
                model_id = model_path.stem
                elo = db.get_elo(model_id)
                if elo and elo > best_elo:
                    best_elo = elo
                    best_model = model_id

            if not best_model:
                return

            # Check for plateau
            auto_tuner = self._auto_tuners[config_key]
            should_tune = auto_tuner.check_plateau(best_elo)
            self._last_elo[config_key] = best_elo

            if should_tune:
                logger.info(f"[CMA-ES] Elo plateau detected for {config_key} (Elo: {best_elo:.0f})")
                logger.info("[CMA-ES] Triggering auto hyperparameter optimization...")
                await self.trigger_auto_cmaes(board_type, num_players)

        except Exception as e:  # noqa: BLE001
            logger.info(f"[CMA-ES] Auto-tuning check error for {config_key}: {e}")

    async def propagate_weights(
        self,
        board_type: str,
        num_players: int,
        weights: dict[str, float],
    ) -> None:
        """Propagate new CMA-ES weights to selfplay workers."""
        try:
            config_key = f"{board_type}_{num_players}p"
            logger.info(f"Propagating CMA-ES weights for {config_key}")

            # Save to shared heuristic weights config
            ai_service_path = self.config.ai_service_path or str(
                Path(__file__).parent.parent.parent.parent
            )
            config_path = Path(ai_service_path) / "config" / "heuristic_weights.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            existing: dict = {}
            if config_path.exists():
                with contextlib.suppress(Exception):
                    existing = json.loads(config_path.read_text())

            existing[config_key] = {
                "weights": weights,
                "updated_at": time.time(),
            }
            config_path.write_text(json.dumps(existing, indent=2))
            logger.info(f"Updated heuristic_weights.json with {config_key} weights")

            # Track config for weight-aware selfplay scheduling
            self._weight_configs[config_key] = {
                "weights": weights,
                "updated_at": time.time(),
            }

            logger.info(f"Weight propagation complete for {config_key}")

        except Exception as e:  # noqa: BLE001
            logger.info(f"CMA-ES weight propagation error: {e}")

    async def _save_weights(
        self,
        board_type: str,
        num_players: int,
        weights: dict[str, float],
        fitness: float,
        job_id: str,
        generation: int,
    ) -> None:
        """Save CMA-ES weights to file."""
        try:
            ai_service_path = self.config.ai_service_path or str(
                Path(__file__).parent.parent.parent.parent
            )
            weights_file = (
                Path(ai_service_path)
                / "data"
                / "cmaes"
                / f"best_weights_{board_type}_{num_players}p.json"
            )
            weights_file.parent.mkdir(parents=True, exist_ok=True)

            with open(weights_file, "w") as f:
                json.dump(
                    {
                        "weights": weights,
                        "fitness": fitness,
                        "job_id": job_id,
                        "generation": generation,
                        "timestamp": time.time(),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved CMA-ES weights to {weights_file}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to save CMA-ES weights: {e}")

    def register_auto_tuner(self, config_key: str, auto_tuner: Any) -> None:
        """Register an auto-tuner for a configuration."""
        self._auto_tuners[config_key] = auto_tuner

    def get_weight_config(self, config_key: str) -> dict | None:
        """Get stored weight config for a configuration."""
        return self._weight_configs.get(config_key)


def create_cmaes_coordinator(
    config: CMAESConfig | None = None,
    **callbacks: Any,
) -> CMAESCoordinator:
    """Create a new CMAESCoordinator instance."""
    return CMAESCoordinator(config, **callbacks)


def get_cmaes_coordinator() -> CMAESCoordinator | None:
    """Get the global CMAESCoordinator instance."""
    return _cmaes_coordinator


def set_cmaes_coordinator(coordinator: CMAESCoordinator) -> None:
    """Set the global CMAESCoordinator instance."""
    global _cmaes_coordinator
    _cmaes_coordinator = coordinator


def reset_cmaes_coordinator() -> None:
    """Reset the global CMAESCoordinator instance."""
    global _cmaes_coordinator
    _cmaes_coordinator = None
