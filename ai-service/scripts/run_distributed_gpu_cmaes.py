#!/usr/bin/env python
"""Distributed GPU CMA-ES optimization across multiple machines.

This script coordinates CMA-ES optimization across a cluster of GPU machines,
distributing candidate evaluation to maximize throughput.

Architecture:
    - Coordinator: Runs CMA-ES algorithm, distributes candidates
    - Workers: GPU machines that evaluate fitness batches

Usage (Coordinator):
    python scripts/run_distributed_gpu_cmaes.py \\
        --mode coordinator \\
        --board square8 \\
        --num-players 2 \\
        --generations 100 \\
        --workers vast-quad:8766,vast-dual:8766,vast-3090:8766,lambda:8766 \\
        --output-dir logs/cmaes/distributed

Usage (Worker):
    python scripts/run_distributed_gpu_cmaes.py \\
        --mode worker \\
        --port 8766 \\
        --board square8

Workers can also be started via SSH from coordinator using --auto-start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from app.ai.gpu_batch import get_device
from app.ai.gpu_parallel_games import (
    evaluate_candidate_fitness_gpu,
    benchmark_parallel_games,
)

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_distributed_gpu_cmaes")

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_WEIGHTS = {
    "material_weight": 1.0,
    "ring_count_weight": 0.5,
    "stack_height_weight": 0.3,
    "center_control_weight": 0.4,
    "territory_weight": 0.8,
    "mobility_weight": 0.2,
    "line_potential_weight": 0.6,
    "defensive_weight": 0.3,
}

WEIGHT_NAMES = list(DEFAULT_WEIGHTS.keys())


def weights_to_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([weights.get(name, DEFAULT_WEIGHTS[name]) for name in WEIGHT_NAMES])


def vector_to_weights(vec: np.ndarray) -> Dict[str, float]:
    return {name: float(vec[i]) for i, name in enumerate(WEIGHT_NAMES)}


# =============================================================================
# Worker Node
# =============================================================================


@dataclass
class EvalTask:
    """Task for fitness evaluation."""
    task_id: str
    candidate_weights: Dict[str, float]
    baseline_weights: Dict[str, float]
    num_games: int
    board_size: int
    num_players: int
    max_moves: int


@dataclass
class EvalResult:
    """Result from fitness evaluation."""
    task_id: str
    fitness: float
    games_played: int
    elapsed_seconds: float
    error: Optional[str] = None


class GPUWorker:
    """GPU worker that evaluates fitness tasks."""

    def __init__(
        self,
        board_size: int = 8,
        num_players: int = 2,
        default_games: int = 50,
        max_moves: int = 10000,
    ):
        self.device = get_device()
        self.board_size = board_size
        self.num_players = num_players
        self.default_games = default_games
        self.max_moves = max_moves

        logger.info(f"GPU Worker initialized on {self.device}")
        logger.info(f"  Board: {board_size}x{board_size}, Players: {num_players}")

        # Warmup
        logger.info("Running GPU warmup...")
        bench = benchmark_parallel_games(
            batch_sizes=[default_games],
            board_size=board_size,
            max_moves=50,
            device=self.device,
        )
        logger.info(f"  Throughput: {bench['games_per_second'][0]:.1f} games/sec")

    def evaluate(self, task: EvalTask) -> EvalResult:
        """Evaluate a single candidate."""
        start = time.time()
        try:
            fitness = evaluate_candidate_fitness_gpu(
                candidate_weights=task.candidate_weights,
                opponent_weights=task.baseline_weights,
                num_games=task.num_games,
                board_size=task.board_size,
                num_players=task.num_players,
                max_moves=task.max_moves,
                device=self.device,
            )
            elapsed = time.time() - start
            return EvalResult(
                task_id=task.task_id,
                fitness=fitness,
                games_played=task.num_games,
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            return EvalResult(
                task_id=task.task_id,
                fitness=0.0,
                games_played=0,
                elapsed_seconds=time.time() - start,
                error=str(e),
            )


def run_worker_server(port: int, board_size: int, num_players: int):
    """Run HTTP worker server for remote evaluation."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="GPU CMA-ES Worker")
    worker = GPUWorker(board_size=board_size, num_players=num_players)

    class TaskRequest(BaseModel):
        task_id: str
        candidate_weights: Dict[str, float]
        baseline_weights: Dict[str, float]
        num_games: int = 50
        board_size: int = 8
        num_players: int = 2
        max_moves: int = 2000

    class TaskResponse(BaseModel):
        task_id: str
        fitness: float
        games_played: int
        elapsed_seconds: float
        error: Optional[str] = None

    @app.post("/evaluate", response_model=TaskResponse)
    async def evaluate_task(request: TaskRequest):
        task = EvalTask(
            task_id=request.task_id,
            candidate_weights=request.candidate_weights,
            baseline_weights=request.baseline_weights,
            num_games=request.num_games,
            board_size=request.board_size,
            num_players=request.num_players,
            max_moves=request.max_moves,
        )
        result = worker.evaluate(task)
        return TaskResponse(
            task_id=result.task_id,
            fitness=result.fitness,
            games_played=result.games_played,
            elapsed_seconds=result.elapsed_seconds,
            error=result.error,
        )

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "device": str(worker.device),
            "board_size": worker.board_size,
            "num_players": worker.num_players,
        }

    @app.get("/benchmark")
    async def run_benchmark():
        results = benchmark_parallel_games(
            batch_sizes=[32, 64, 128, 256],
            board_size=worker.board_size,
            max_moves=100,
            device=worker.device,
        )
        return {
            "batch_sizes": results["batch_size"],
            "games_per_second": results["games_per_second"],
        }

    logger.info(f"Starting worker server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


# =============================================================================
# Coordinator Node
# =============================================================================


class DistributedCoordinator:
    """Coordinate CMA-ES optimization across multiple GPU workers."""

    def __init__(
        self,
        worker_urls: List[str],
        board_size: int = 8,
        num_players: int = 2,
        games_per_eval: int = 50,
        max_moves: int = 2000,
        baseline_weights: Optional[Dict[str, float]] = None,
    ):
        self.worker_urls = worker_urls
        self.board_size = board_size
        self.num_players = num_players
        self.games_per_eval = games_per_eval
        self.max_moves = max_moves
        self.baseline_weights = baseline_weights or DEFAULT_WEIGHTS.copy()

        self.task_counter = 0
        self.total_games = 0
        self.total_time = 0.0

        logger.info(f"Distributed coordinator with {len(worker_urls)} workers")
        for url in worker_urls:
            logger.info(f"  - {url}")

    def check_workers(self) -> Dict[str, Any]:
        """Check health of all workers."""
        import requests

        status = {}
        for url in self.worker_urls:
            try:
                resp = requests.get(f"{url}/health", timeout=10)
                status[url] = resp.json() if resp.ok else {"status": "error", "code": resp.status_code}
            except Exception as e:
                status[url] = {"status": "unreachable", "error": str(e)}
        return status

    def evaluate_population(
        self,
        population: List[np.ndarray],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Evaluate population distributed across workers."""
        import requests

        start = time.time()
        fitness_scores = [0.0] * len(population)

        # Create tasks
        tasks = []
        for i, vec in enumerate(population):
            self.task_counter += 1
            task = {
                "task_id": f"gen_task_{self.task_counter}",
                "candidate_weights": vector_to_weights(vec),
                "baseline_weights": self.baseline_weights,
                "num_games": self.games_per_eval,
                "board_size": self.board_size,
                "num_players": self.num_players,
                "max_moves": self.max_moves,
            }
            tasks.append((i, task))

        # Distribute tasks across workers
        def evaluate_on_worker(args):
            idx, task, url = args
            try:
                resp = requests.post(f"{url}/evaluate", json=task, timeout=300)
                if resp.ok:
                    result = resp.json()
                    return idx, result["fitness"], None
                return idx, 0.0, f"HTTP {resp.status_code}"
            except Exception as e:
                return idx, 0.0, str(e)

        # Round-robin assignment to workers
        work_items = [
            (idx, task, self.worker_urls[idx % len(self.worker_urls)])
            for idx, task in tasks
        ]

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=len(self.worker_urls)) as executor:
            futures = [executor.submit(evaluate_on_worker, item) for item in work_items]
            for future in as_completed(futures):
                idx, fitness, error = future.result()
                if error:
                    logger.warning(f"Task {idx} failed: {error}")
                fitness_scores[idx] = fitness

        elapsed = time.time() - start
        total_games = len(population) * self.games_per_eval
        self.total_games += total_games
        self.total_time += elapsed

        stats = {
            "population_size": len(population),
            "num_workers": len(self.worker_urls),
            "total_games": total_games,
            "elapsed_seconds": elapsed,
            "games_per_second": total_games / elapsed if elapsed > 0 else 0,
            "best_fitness": max(fitness_scores),
            "mean_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
        }

        return fitness_scores, stats


def run_distributed_cmaes(
    worker_urls: List[str],
    board_type: str,
    num_players: int,
    generations: int,
    population_size: int,
    games_per_eval: int,
    output_dir: str,
    sigma: float = 0.5,
    max_moves: int = 2000,
    baseline_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run distributed CMA-ES optimization."""
    try:
        import cma
    except ImportError:
        logger.error("CMA-ES library not installed. Run: pip install cma")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    board_size = {"square8": 8, "square19": 19, "hex": 25}.get(board_type.lower(), 8)

    logger.info("=" * 60)
    logger.info("DISTRIBUTED GPU CMA-ES OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Board: {board_type} ({board_size}x{board_size})")
    logger.info(f"Players: {num_players}")
    logger.info(f"Workers: {len(worker_urls)}")
    logger.info(f"Generations: {generations}")
    logger.info(f"Population: {population_size}")
    logger.info(f"Games/eval: {games_per_eval}")
    logger.info("")

    # Initialize coordinator
    coordinator = DistributedCoordinator(
        worker_urls=worker_urls,
        board_size=board_size,
        num_players=num_players,
        games_per_eval=games_per_eval,
        max_moves=max_moves,
        baseline_weights=baseline_weights,
    )

    # Check workers
    logger.info("Checking worker health...")
    health = coordinator.check_workers()
    healthy = sum(1 for h in health.values() if h.get("status") == "healthy")
    logger.info(f"  {healthy}/{len(worker_urls)} workers healthy")
    if healthy == 0:
        logger.error("No healthy workers available!")
        return {"error": "No healthy workers"}

    # Initialize CMA-ES
    x0 = weights_to_vector(baseline_weights or DEFAULT_WEIGHTS)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma,
        {
            "popsize": population_size,
            "seed": seed,
            "verbose": -1,
            "bounds": [0.0, 5.0],
        },
    )

    best_fitness = 0.0
    best_weights = baseline_weights or DEFAULT_WEIGHTS.copy()
    history = []
    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        solutions = es.ask()

        logger.info(f"Generation {gen + 1}/{generations}")
        fitness_scores, stats = coordinator.evaluate_population(solutions)

        minimization_scores = [-max(0.001, f) for f in fitness_scores]
        es.tell(solutions, minimization_scores)

        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_weights = vector_to_weights(solutions[gen_best_idx])

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_weights = gen_best_weights.copy()

        gen_elapsed = time.time() - gen_start

        logger.info(
            f"  Best: {gen_best_fitness:.3f}, Mean: {stats['mean_fitness']:.3f}"
        )
        logger.info(
            f"  Speed: {stats['games_per_second']:.1f} g/s ({len(worker_urls)} workers)"
        )
        logger.info(f"  Overall best: {best_fitness:.3f}")
        logger.info("")

        checkpoint = {
            "generation": gen + 1,
            "fitness": gen_best_fitness,
            "best_fitness": best_fitness,
            "weights": gen_best_weights,
            "best_weights": best_weights,
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(checkpoint)

        with open(os.path.join(output_dir, f"checkpoint_gen{gen + 1:03d}.json"), "w") as f:
            json.dump(checkpoint, f, indent=2)

        if es.stop():
            logger.info("CMA-ES converged")
            break

    total_time = time.time() - start_time

    results = {
        "board_type": board_type,
        "num_players": num_players,
        "num_workers": len(worker_urls),
        "generations_completed": len(history),
        "best_fitness": best_fitness,
        "best_weights": best_weights,
        "total_time_seconds": total_time,
        "total_games": coordinator.total_games,
        "history": history,
    }

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best fitness: {best_fitness:.3f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Total games: {coordinator.total_games}")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Distributed GPU CMA-ES optimization"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["coordinator", "worker"],
        help="Run as coordinator or worker",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )

    # Coordinator options
    parser.add_argument(
        "--workers",
        type=str,
        help="Comma-separated worker URLs (e.g., http://host1:8766,http://host2:8766)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=50,
        help="Games per fitness evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/cmaes/distributed",
        help="Output directory",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="CMA-ES step size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Worker options
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Worker server port",
    )

    args = parser.parse_args()

    if args.mode == "worker":
        board_size = {"square8": 8, "square19": 19, "hex": 25}.get(args.board.lower(), 8)
        run_worker_server(args.port, board_size, args.num_players)
    else:
        if not args.workers:
            logger.error("Coordinator mode requires --workers")
            sys.exit(1)

        worker_urls = [url.strip() for url in args.workers.split(",")]
        run_distributed_cmaes(
            worker_urls=worker_urls,
            board_type=args.board,
            num_players=args.num_players,
            generations=args.generations,
            population_size=args.population_size,
            games_per_eval=args.games_per_eval,
            output_dir=args.output_dir,
            sigma=args.sigma,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
