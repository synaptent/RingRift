#!/usr/bin/env python3
"""Unified CMA-ES Optimization for RingRift Heuristic Weights.

This script consolidates ALL CMA-ES optimization modes into a single interface,
defaulting to GPU-accelerated distributed CMA-ES using EvoTorch.

CONSOLIDATED FROM:
    - run_cmaes_optimization.py (main CMA-ES, 49 weights, comprehensive)
    - run_evotorch_cmaes.py (EvoTorch GPU-native)
    - run_gpu_cmaes.py (GPU CMA-ES)
    - run_distributed_gpu_cmaes.py (distributed coordinator/worker)
    - run_iterative_cmaes.py (iterative self-improvement loop)
    - cmaes_cloud_worker.py (queue-based cloud worker)

BACKENDS:
    evotorch (default): GPU-native CMA-ES, full vectorization, 10-100x faster
    pycma: CPU-based CMA-ES (legacy fallback)

EXECUTION MODES:
    local: Single GPU/CPU on current machine
    distributed: Multi-GPU across cluster nodes via HTTP workers
    cloud: Queue-based workers (Redis/SQS) for cloud deployment
    iterative: Self-improvement loop with automatic baseline promotion

FEATURES:
    - Full 49 heuristic weights (HEURISTIC_WEIGHT_KEYS)
    - GPU-accelerated fitness evaluation via ParallelGameRunner
    - Multi-board evaluation (cross-board generalization)
    - Multi-player support (2p, 3p, 4p)
    - State pool evaluation (multi-start from mid-game positions)
    - Checkpoint/resume with generation tracking
    - Model registry integration for auto-promotion
    - NN quality gating for guided optimization
    - Opponent mode: baseline-only or baseline-plus-incumbent

Usage:
    # Default: GPU-accelerated EvoTorch on local machine
    python scripts/run_unified_cmaes.py --board square8 --num-players 2

    # Distributed across cluster (4 workers)
    python scripts/run_unified_cmaes.py --board square8 --distributed \\
        --workers http://gh200-e:8766,http://gh200-f:8766

    # Iterative self-improvement loop
    python scripts/run_unified_cmaes.py --board square8 --iterative \\
        --max-iterations 10 --improvement-threshold 0.55

    # Cloud worker mode (Redis queue)
    python scripts/run_unified_cmaes.py --mode worker --queue-backend redis

    # Legacy pycma mode with GPU fitness eval
    python scripts/run_unified_cmaes.py --board square8 --backend pycma --gpu

Requirements:
    pip install evotorch torch cma

References:
    - EvoTorch: https://docs.evotorch.ai/
    - pycma: https://github.com/CMA-ES/pycma
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# =============================================================================
# Imports and Availability Checks
# =============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    torch = None  # type: ignore

try:
    import evotorch
    from evotorch import Problem
    from evotorch.algorithms import CMAES as EvotorchCMAES
    from evotorch.logging import StdOutLogger
    EVOTORCH_AVAILABLE = True
except ImportError:
    EVOTORCH_AVAILABLE = False
    evotorch = None  # type: ignore

try:
    import cma
    PYCMA_AVAILABLE = True
except ImportError:
    PYCMA_AVAILABLE = False
    cma = None  # type: ignore

# RingRift imports
from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)
from app.rules.core import BOARD_CONFIGS, BoardType

try:
    from app.ai.gpu_parallel_games import (
        ParallelGameRunner,
        evaluate_candidate_fitness_gpu,
    )
    from app.ai.gpu_batch import clear_gpu_memory, get_device
    GPU_GAMES_AVAILABLE = True
except ImportError:
    GPU_GAMES_AVAILABLE = False
    ParallelGameRunner = None  # type: ignore

try:
    from app.ai.hybrid_gpu import HybridGPUEvaluator
    HYBRID_GPU_AVAILABLE = True
except ImportError:
    HYBRID_GPU_AVAILABLE = False
    HybridGPUEvaluator = None  # type: ignore

try:
    from app.training.eval_pools import load_state_pool
    EVAL_POOLS_AVAILABLE = True
except ImportError:
    EVAL_POOLS_AVAILABLE = False

try:
    from app.distributed.queue import TaskQueue, get_task_queue, EvalTask, EvalResult
    QUEUE_AVAILABLE = True
except ImportError:
    QUEUE_AVAILABLE = False

try:
    from app.training.cmaes_registry_integration import register_cmaes_result
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_cmaes_result = None  # type: ignore

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

NUM_WEIGHTS = len(HEURISTIC_WEIGHT_KEYS)

BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hex8": BoardType.HEX8,
    "hexagonal": BoardType.HEXAGONAL,
}

# =============================================================================
# Weight Conversion Utilities
# =============================================================================


def weights_to_array(weights: HeuristicWeights) -> np.ndarray:
    """Convert weight dict to numpy array in canonical order."""
    return np.array([weights.get(k, 0.0) for k in HEURISTIC_WEIGHT_KEYS], dtype=np.float32)


def array_to_weights(arr: np.ndarray) -> HeuristicWeights:
    """Convert numpy array to weight dict."""
    return {k: float(arr[i]) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}


def weights_to_tensor(weights: HeuristicWeights, device: str = "cpu") -> "torch.Tensor":
    """Convert weight dict to torch tensor."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for tensor conversion")
    return torch.tensor(
        [weights.get(k, 0.0) for k in HEURISTIC_WEIGHT_KEYS],
        dtype=torch.float32,
        device=device,
    )


def tensor_to_weights(tensor: "torch.Tensor") -> HeuristicWeights:
    """Convert torch tensor to weight dict."""
    values = tensor.detach().cpu().numpy()
    return {k: float(values[i]) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}


def weights_to_gpu_format(weights: HeuristicWeights) -> dict[str, float]:
    """Convert to format expected by GPU evaluator (lowercase, no WEIGHT_ prefix)."""
    return {k.replace("WEIGHT_", "").lower(): v for k, v in weights.items()}


def load_weights_from_file(path: str) -> HeuristicWeights:
    """Load weights from JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Support both flat and nested formats
    if "weights" in data:
        return data["weights"]
    if "best_weights" in data:
        return data["best_weights"]
    return data


def save_weights_to_file(weights: HeuristicWeights, path: str, metadata: dict = None):
    """Save weights to JSON file with metadata."""
    data = {
        "weights": weights,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {}),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Fitness Evaluation
# =============================================================================


def evaluate_fitness_gpu(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    board_type: str,
    num_players: int,
    games_per_eval: int,
    max_moves: int = 200,
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[float, dict[str, Any]]:
    """Evaluate fitness using GPU-accelerated parallel games.

    Uses ParallelGameRunner for maximum throughput.

    Returns:
        Tuple of (fitness_score, stats_dict)
    """
    if not GPU_GAMES_AVAILABLE:
        raise ImportError("GPU game modules not available")

    board_enum = BOARD_TYPE_MAP.get(board_type.lower())
    if not board_enum:
        raise ValueError(f"Unknown board type: {board_type}")

    board_size = BOARD_CONFIGS[board_enum].size

    # Convert weights to GPU format
    candidate_gpu = weights_to_gpu_format(candidate_weights)
    baseline_gpu = weights_to_gpu_format(baseline_weights)

    # Use the existing evaluate_candidate_fitness_gpu function
    win_rate = evaluate_candidate_fitness_gpu(
        candidate_weights=candidate_gpu,
        opponent_weights=baseline_gpu,
        num_games=games_per_eval,
        board_size=board_size,
        num_players=num_players,
        max_moves=max_moves,
        device=torch.device(device) if TORCH_AVAILABLE else None,
    )

    stats = {
        "games_completed": games_per_eval,
        "win_rate": win_rate,
        "board_type": board_type,
        "num_players": num_players,
    }

    return win_rate, stats


def evaluate_fitness_cpu(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    board_type: str,
    num_players: int,
    games_per_eval: int,
    max_moves: int = 200,
) -> tuple[float, dict[str, Any]]:
    """Evaluate fitness using CPU game simulation."""
    from app.ai.heuristic_ai import HeuristicAI
    from app.rules.default_engine import DefaultRulesEngine

    board_enum = BOARD_TYPE_MAP.get(board_type.lower())

    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    total_moves = 0

    for game_idx in range(games_per_eval):
        # Create game state
        game_state = DefaultRulesEngine.create_initial_state(
            board_type=board_enum,
            num_players=num_players,
        )

        # Create AIs
        ais = []
        for p in range(num_players):
            ai = HeuristicAI(player_number=p + 1)
            if (game_idx + p) % 2 == 0:
                ai.apply_weights(candidate_weights)
            else:
                ai.apply_weights(baseline_weights)
            ais.append(ai)

        # Play game
        moves = 0
        while not game_state.is_terminal and moves < max_moves:
            current_player = game_state.current_player
            ai = ais[current_player - 1]
            move = ai.select_move(game_state)
            if move:
                game_state = DefaultRulesEngine.apply_move(game_state, move)
            moves += 1

        total_moves += moves

        # Score
        if game_state.winner:
            if (game_idx + game_state.winner - 1) % 2 == 0:
                candidate_wins += 1
            else:
                baseline_wins += 1
        else:
            draws += 1

    total_games = candidate_wins + baseline_wins + draws
    fitness = (candidate_wins + 0.5 * draws) / max(1, total_games)

    stats = {
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "total_games": total_games,
        "avg_moves": total_moves / max(1, total_games),
    }

    return fitness, stats


def evaluate_fitness(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    config: "CMAESConfig",
) -> tuple[float, dict[str, Any]]:
    """Unified fitness evaluation dispatcher."""
    if config.use_gpu and GPU_GAMES_AVAILABLE and CUDA_AVAILABLE:
        return evaluate_fitness_gpu(
            candidate_weights=candidate_weights,
            baseline_weights=baseline_weights,
            board_type=config.board_type,
            num_players=config.num_players,
            games_per_eval=config.games_per_eval,
            max_moves=config.max_moves,
            batch_size=config.gpu_batch_size,
            device=config.device,
        )
    else:
        return evaluate_fitness_cpu(
            candidate_weights=candidate_weights,
            baseline_weights=baseline_weights,
            board_type=config.board_type,
            num_players=config.num_players,
            games_per_eval=config.games_per_eval,
            max_moves=config.max_moves,
        )


# =============================================================================
# EvoTorch Backend (GPU-Native)
# =============================================================================


def _create_ringrift_problem(config: "CMAESConfig", baseline_weights: HeuristicWeights):
    """Factory function to create EvoTorch Problem (only when EvoTorch available)."""
    if not EVOTORCH_AVAILABLE:
        raise ImportError("EvoTorch not available")

    class RingRiftOptimizationProblem(Problem):
        """EvoTorch Problem for RingRift heuristic weight optimization."""

        def __init__(self):
            self.config = config
            self.baseline_weights = baseline_weights
            self.best_fitness = 0.0
            self.best_weights = baseline_weights.copy()
            self.eval_count = 0

            # Compute bounds based on baseline weights
            # EvoTorch expects bounds as (lower_bounds, upper_bounds) tuple
            lower_bounds = []
            upper_bounds = []
            for key in HEURISTIC_WEIGHT_KEYS:
                val = baseline_weights.get(key, 0.0)
                delta = max(abs(val) * 0.5, 1.0)
                lower_bounds.append(val - delta)
                upper_bounds.append(val + delta)

            device = config.device if TORCH_AVAILABLE else "cpu"

            super().__init__(
                objective_sense="max",
                solution_length=NUM_WEIGHTS,
                initial_bounds=(lower_bounds, upper_bounds),
                dtype=torch.float32 if TORCH_AVAILABLE else np.float32,
                device=device,
            )

        def _evaluate(self, solutions) -> "torch.Tensor":
            """Evaluate solutions - handles EvoTorch Solution objects."""
            # EvoTorch can pass either a single Solution or a SolutionBatch
            # Check for SolutionBatch by looking for access_values method + checking ndim

            def get_values_tensor(sol):
                """Get the values tensor from a Solution object."""
                vals = sol.values
                if callable(vals):
                    vals = vals()
                return vals

            # Check if this is a batch by checking if access_values returns 2D tensor
            has_access = hasattr(solutions, 'access_values')
            if has_access:
                all_vals = solutions.access_values()
                if all_vals.dim() == 2:
                    # True batch
                    batch_size = all_vals.shape[0]
                    all_values = all_vals.detach().cpu().numpy()

                    for i in range(batch_size):
                        values = all_values[i]
                        weights = {k: float(values[j]) for j, k in enumerate(HEURISTIC_WEIGHT_KEYS)}
                        fitness, _stats = evaluate_fitness(
                            candidate_weights=weights,
                            baseline_weights=self.baseline_weights,
                            config=self.config,
                        )
                        self.eval_count += 1

                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_weights = weights.copy()

                        solutions[i].set_evals(torch.tensor([fitness], device=solutions.device, dtype=torch.float32))

                    return None

            # Single Solution - get values directly
            vals = get_values_tensor(solutions)
            values = vals.detach().cpu().numpy()
            weights = {k: float(values[i]) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}
            fitness, _stats = evaluate_fitness(
                candidate_weights=weights,
                baseline_weights=self.baseline_weights,
                config=self.config,
            )
            self.eval_count += 1

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_weights = weights.copy()

            solutions.set_evals(torch.tensor([fitness], device=vals.device, dtype=torch.float32))
            return None

    return RingRiftOptimizationProblem()


def run_evotorch_cmaes(config: "CMAESConfig") -> tuple[HeuristicWeights, float]:
    """Run CMA-ES using EvoTorch backend (GPU-native)."""
    if not EVOTORCH_AVAILABLE:
        raise ImportError("EvoTorch not available. Install: pip install evotorch")

    logger.info("=" * 70)
    logger.info("EvoTorch GPU-Native CMA-ES")
    logger.info("=" * 70)
    logger.info(f"Backend: EvoTorch (GPU-native algorithm)")
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Generations: {config.generations}, Population: {config.population_size}")
    logger.info(f"Games per eval: {config.games_per_eval}, Max moves: {config.max_moves}")
    logger.info(f"Device: {config.device}, GPU games: {config.use_gpu}")
    logger.info(f"Weights to optimize: {NUM_WEIGHTS}")
    logger.info("=" * 70)

    # Create output directory
    run_id = config.run_id or datetime.now().strftime("evotorch_%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create problem
    problem = _create_ringrift_problem(config, config.baseline_weights)

    # Create CMA-ES algorithm
    initial_weights = weights_to_tensor(config.baseline_weights, config.device)

    searcher = EvotorchCMAES(
        problem,
        stdev_init=config.sigma,
        popsize=config.population_size,
        center_init=initial_weights,
    )

    # Add logging
    StdOutLogger(searcher)

    start_time = time.time()

    for gen in range(config.generations):
        searcher.step()

        # Progress logging
        elapsed = time.time() - start_time
        evals = problem.eval_count
        rate = evals / elapsed if elapsed > 0 else 0

        logger.info(
            f"[Gen {gen+1}/{config.generations}] "
            f"Best: {problem.best_fitness:.4f} | "
            f"Evals: {evals} | "
            f"Rate: {rate:.1f}/s | "
            f"Elapsed: {elapsed:.0f}s"
        )

        # Checkpoint every 5 generations
        if (gen + 1) % 5 == 0:
            checkpoint = {
                "generation": gen + 1,
                "best_fitness": problem.best_fitness,
                "best_weights": problem.best_weights,
                "elapsed_seconds": elapsed,
                "evaluations": evals,
            }
            with open(output_dir / f"checkpoint_gen{gen+1:03d}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)

    # Final results
    elapsed = time.time() - start_time
    results = {
        "run_id": run_id,
        "backend": "evotorch",
        "board_type": config.board_type,
        "num_players": config.num_players,
        "generations": config.generations,
        "population_size": config.population_size,
        "games_per_eval": config.games_per_eval,
        "best_fitness": problem.best_fitness,
        "best_weights": problem.best_weights,
        "elapsed_seconds": elapsed,
        "total_evaluations": problem.eval_count,
        "baseline_weights": config.baseline_weights,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"Optimization complete in {elapsed:.1f}s")
    logger.info(f"Best fitness: {problem.best_fitness:.4f}")
    logger.info(f"Total evaluations: {problem.eval_count}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)

    return problem.best_weights, problem.best_fitness


# =============================================================================
# pycma Backend (CPU Algorithm, Optional GPU Fitness)
# =============================================================================


def run_pycma_cmaes(config: "CMAESConfig") -> tuple[HeuristicWeights, float]:
    """Run CMA-ES using pycma backend."""
    if not PYCMA_AVAILABLE:
        raise ImportError("pycma not available. Install: pip install cma")

    logger.info("=" * 70)
    logger.info("pycma CMA-ES")
    logger.info("=" * 70)
    logger.info(f"Backend: pycma (CPU algorithm, {'GPU' if config.use_gpu else 'CPU'} fitness)")
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Generations: {config.generations}, Population: {config.population_size}")
    logger.info(f"Games per eval: {config.games_per_eval}")
    logger.info("=" * 70)

    run_id = config.run_id or datetime.now().strftime("pycma_%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    x0 = weights_to_array(config.baseline_weights)

    es = cma.CMAEvolutionStrategy(
        x0,
        config.sigma,
        {
            "popsize": config.population_size,
            "maxiter": config.generations,
            "verb_disp": 1,
        },
    )

    best_weights = config.baseline_weights.copy()
    best_fitness = 0.0
    eval_count = 0

    start_time = time.time()
    generation = 0

    while not es.stop():
        generation += 1
        solutions = es.ask()

        fitnesses = []
        for sol in solutions:
            weights = array_to_weights(sol)
            fitness, _ = evaluate_fitness(weights, config.baseline_weights, config)
            eval_count += 1

            # pycma minimizes, so negate
            fitnesses.append(-fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = weights.copy()
                logger.info(f"[Gen {generation}] New best: {best_fitness:.4f}")

        es.tell(solutions, fitnesses)

        # Checkpoint
        if generation % 5 == 0:
            elapsed = time.time() - start_time
            checkpoint = {
                "generation": generation,
                "best_fitness": best_fitness,
                "best_weights": best_weights,
                "elapsed_seconds": elapsed,
            }
            with open(output_dir / f"checkpoint_gen{generation:03d}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)

    elapsed = time.time() - start_time
    results = {
        "run_id": run_id,
        "backend": "pycma",
        "generations": generation,
        "best_fitness": best_fitness,
        "best_weights": best_weights,
        "elapsed_seconds": elapsed,
        "total_evaluations": eval_count,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Optimization complete in {elapsed:.1f}s")
    logger.info(f"Best fitness: {best_fitness:.4f}")

    return best_weights, best_fitness


# =============================================================================
# Iterative Self-Improvement Loop
# =============================================================================


def run_iterative_cmaes(config: "CMAESConfig") -> tuple[HeuristicWeights, float]:
    """Run iterative CMA-ES with automatic baseline promotion.

    Similar to AlphaZero's training loop:
    1. Run CMA-ES for N generations
    2. If improvement > threshold: promote new weights as baseline
    3. Repeat until max iterations or plateau
    """
    logger.info("=" * 70)
    logger.info("Iterative CMA-ES Self-Improvement Loop")
    logger.info("=" * 70)
    logger.info(f"Max iterations: {config.max_iterations}")
    logger.info(f"Generations per iteration: {config.generations}")
    logger.info(f"Improvement threshold: {config.improvement_threshold}")
    logger.info("=" * 70)

    current_baseline = config.baseline_weights.copy()
    global_best_fitness = 0.0
    global_best_weights = current_baseline.copy()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, config.max_iterations + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"ITERATION {iteration}/{config.max_iterations}")
        logger.info(f"{'='*70}")

        # Create iteration config
        iter_config = CMAESConfig(
            backend=config.backend,
            generations=config.generations,
            population_size=config.population_size,
            sigma=config.sigma,
            games_per_eval=config.games_per_eval,
            max_moves=config.max_moves,
            board_type=config.board_type,
            num_players=config.num_players,
            device=config.device,
            use_gpu=config.use_gpu,
            gpu_batch_size=config.gpu_batch_size,
            output_dir=str(output_dir / f"iter_{iteration:03d}"),
            baseline_weights=current_baseline,
            run_id=f"iter_{iteration:03d}",
        )

        # Run CMA-ES
        if config.backend == "evotorch" and EVOTORCH_AVAILABLE:
            best_weights, best_fitness = run_evotorch_cmaes(iter_config)
        else:
            best_weights, best_fitness = run_pycma_cmaes(iter_config)

        logger.info(f"Iteration {iteration} complete: fitness = {best_fitness:.4f}")

        # Check for improvement
        if best_fitness > global_best_fitness:
            improvement = best_fitness - global_best_fitness
            global_best_fitness = best_fitness
            global_best_weights = best_weights.copy()

            if best_fitness >= config.improvement_threshold:
                logger.info(f"Promoting new baseline (fitness: {best_fitness:.4f})")
                current_baseline = best_weights.copy()

                # Save promoted weights
                save_weights_to_file(
                    best_weights,
                    str(output_dir / f"promoted_iter_{iteration:03d}.json"),
                    {"iteration": iteration, "fitness": best_fitness},
                )
        else:
            logger.info(f"No improvement (best: {global_best_fitness:.4f})")

            # Check for plateau
            if iteration >= config.plateau_generations:
                logger.info("Plateau detected, stopping early")
                break

    # Save final results
    final_results = {
        "iterations_completed": iteration,
        "global_best_fitness": global_best_fitness,
        "global_best_weights": global_best_weights,
        "final_baseline": current_baseline,
    }

    with open(output_dir / "iterative_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    return global_best_weights, global_best_fitness


# =============================================================================
# Distributed Coordinator Mode
# =============================================================================


class DistributedCoordinator:
    """Coordinate CMA-ES optimization across multiple GPU workers."""

    def __init__(
        self,
        worker_urls: list[str],
        config: "CMAESConfig",
    ):
        self.worker_urls = worker_urls
        self.config = config
        self.task_counter = 0
        self.total_games = 0
        self.total_time = 0.0

        logger.info(f"Distributed coordinator with {len(worker_urls)} workers")
        for url in worker_urls:
            logger.info(f"  - {url}")

    def check_workers(self) -> dict[str, Any]:
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

    def evaluate_candidates(
        self,
        candidates: list[HeuristicWeights],
        baseline_weights: HeuristicWeights,
    ) -> list[float]:
        """Evaluate candidates distributed across workers."""
        import requests

        start = time.time()
        fitness_scores = [0.0] * len(candidates)

        # Create tasks
        tasks = []
        for i, weights in enumerate(candidates):
            self.task_counter += 1
            task = {
                "task_id": f"task_{self.task_counter}",
                "weights": weights,
                "baseline": baseline_weights,
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
        total_games = len(candidates) * self.config.games_per_eval
        self.total_games += total_games
        self.total_time += elapsed

        logger.info(
            f"  Distributed eval: {len(candidates)} candidates, "
            f"{total_games} games, {total_games / elapsed:.1f} g/s"
        )

        return fitness_scores


def run_distributed_cmaes(config: "CMAESConfig") -> tuple[HeuristicWeights, float]:
    """Run CMA-ES with distributed workers for evaluation."""
    if not config.workers:
        raise ValueError("Distributed mode requires --workers")

    logger.info("=" * 70)
    logger.info("Distributed CMA-ES")
    logger.info("=" * 70)
    logger.info(f"Workers: {len(config.workers)}")
    for url in config.workers:
        logger.info(f"  - {url}")
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Generations: {config.generations}, Population: {config.population_size}")
    logger.info("=" * 70)

    run_id = config.run_id or datetime.now().strftime("distributed_%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinator = DistributedCoordinator(config.workers, config)

    # Check workers
    logger.info("Checking worker health...")
    health = coordinator.check_workers()
    healthy = sum(1 for h in health.values() if h.get("status") == "healthy")
    logger.info(f"  {healthy}/{len(config.workers)} workers healthy")
    if healthy == 0:
        raise RuntimeError("No healthy workers available")

    # Initialize CMA-ES
    x0 = weights_to_array(config.baseline_weights)

    if config.backend == "evotorch" and EVOTORCH_AVAILABLE:
        # Use EvoTorch with distributed evaluation
        return run_evotorch_distributed(config, coordinator)
    elif PYCMA_AVAILABLE:
        # Use pycma with distributed evaluation
        es = cma.CMAEvolutionStrategy(
            x0,
            config.sigma,
            {
                "popsize": config.population_size,
                "maxiter": config.generations,
                "verb_disp": 0,
            },
        )

        best_weights = config.baseline_weights.copy()
        best_fitness = 0.0
        start_time = time.time()
        generation = 0

        while not es.stop():
            generation += 1
            solutions = es.ask()

            # Convert to weight dicts
            candidates = [array_to_weights(sol) for sol in solutions]

            # Distributed evaluation
            fitness_scores = coordinator.evaluate_candidates(
                candidates, config.baseline_weights
            )

            # pycma minimizes
            minimization_scores = [-max(0.001, f) for f in fitness_scores]
            es.tell(solutions, minimization_scores)

            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = candidates[gen_best_idx].copy()
                logger.info(f"[Gen {generation}] New best: {best_fitness:.4f}")

            # Checkpoint
            if generation % 5 == 0:
                elapsed = time.time() - start_time
                checkpoint = {
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "best_weights": best_weights,
                    "elapsed_seconds": elapsed,
                }
                with open(output_dir / f"checkpoint_gen{generation:03d}.json", "w") as f:
                    json.dump(checkpoint, f, indent=2)

        elapsed = time.time() - start_time
        results = {
            "run_id": run_id,
            "mode": "distributed",
            "generations": generation,
            "best_fitness": best_fitness,
            "best_weights": best_weights,
            "elapsed_seconds": elapsed,
            "total_games": coordinator.total_games,
        }

        with open(output_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Distributed optimization complete in {elapsed:.1f}s")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Total games: {coordinator.total_games}")

        return best_weights, best_fitness
    else:
        raise ImportError("No CMA-ES backend available")


def run_evotorch_distributed(
    config: "CMAESConfig",
    coordinator: DistributedCoordinator,
) -> tuple[HeuristicWeights, float]:
    """Run EvoTorch CMA-ES with distributed evaluation."""
    if not EVOTORCH_AVAILABLE:
        raise ImportError("EvoTorch not available")

    # Create a custom problem that uses distributed evaluation
    class DistributedProblem(Problem):
        def __init__(self):
            self.coordinator = coordinator
            self.config = config
            self.best_fitness = 0.0
            self.best_weights = config.baseline_weights.copy()
            self.eval_count = 0

            # Compute bounds - EvoTorch expects (lower_bounds, upper_bounds) tuple
            lower_bounds = []
            upper_bounds = []
            for key in HEURISTIC_WEIGHT_KEYS:
                val = config.baseline_weights.get(key, 0.0)
                delta = max(abs(val) * 0.5, 1.0)
                lower_bounds.append(val - delta)
                upper_bounds.append(val + delta)

            super().__init__(
                objective_sense="max",
                solution_length=NUM_WEIGHTS,
                initial_bounds=(lower_bounds, upper_bounds),
                dtype=torch.float32,
                device="cpu",  # Algorithm on CPU, eval on workers
            )

        def _evaluate(self, solutions) -> "torch.Tensor":
            # Handle EvoTorch Solution objects - same pattern as local evaluation

            def get_values_tensor(sol):
                vals = sol.values
                if callable(vals):
                    vals = vals()
                return vals

            # Check if batch by checking access_values dimensionality
            has_access = hasattr(solutions, 'access_values')
            if has_access:
                all_vals = solutions.access_values()
                if all_vals.dim() == 2:
                    # True batch - gather all candidates then distribute
                    batch_size = all_vals.shape[0]
                    all_values = all_vals.detach().cpu().numpy()
                    candidates = []
                    for i in range(batch_size):
                        values = all_values[i]
                        weights = {k: float(values[j]) for j, k in enumerate(HEURISTIC_WEIGHT_KEYS)}
                        candidates.append(weights)

                    # Distributed evaluation
                    fitness_scores = self.coordinator.evaluate_candidates(
                        candidates, self.config.baseline_weights
                    )

                    self.eval_count += batch_size

                    # Track best and set evals
                    for i, fitness in enumerate(fitness_scores):
                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_weights = candidates[i].copy()
                        solutions[i].set_evals(torch.tensor([fitness], device=solutions.device, dtype=torch.float32))

                    return None

            # Single Solution
            vals = get_values_tensor(solutions)
            values = vals.detach().cpu().numpy()
            weights = {k: float(values[i]) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}
            candidates = [weights]
            fitness_scores = self.coordinator.evaluate_candidates(
                candidates, self.config.baseline_weights
            )
            self.eval_count += 1
            fitness = fitness_scores[0]
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_weights = weights.copy()
            solutions.set_evals(torch.tensor([fitness], device=vals.device, dtype=torch.float32))
            return None

    problem = DistributedProblem()
    initial_weights = weights_to_tensor(config.baseline_weights, "cpu")

    searcher = EvotorchCMAES(
        problem,
        stdev_init=config.sigma,
        popsize=config.population_size,
        center_init=initial_weights,
    )

    StdOutLogger(searcher)

    run_id = config.run_id or datetime.now().strftime("evotorch_dist_%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for gen in range(config.generations):
        searcher.step()

        elapsed = time.time() - start_time
        logger.info(
            f"[Gen {gen+1}/{config.generations}] "
            f"Best: {problem.best_fitness:.4f} | "
            f"Evals: {problem.eval_count} | "
            f"Elapsed: {elapsed:.0f}s"
        )

        if (gen + 1) % 5 == 0:
            checkpoint = {
                "generation": gen + 1,
                "best_fitness": problem.best_fitness,
                "best_weights": problem.best_weights,
                "elapsed_seconds": elapsed,
            }
            with open(output_dir / f"checkpoint_gen{gen+1:03d}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)

    elapsed = time.time() - start_time
    results = {
        "run_id": run_id,
        "mode": "distributed",
        "backend": "evotorch",
        "generations": config.generations,
        "best_fitness": problem.best_fitness,
        "best_weights": problem.best_weights,
        "elapsed_seconds": elapsed,
        "total_games": coordinator.total_games,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Distributed optimization complete in {elapsed:.1f}s")
    logger.info(f"Best fitness: {problem.best_fitness:.4f}")

    return problem.best_weights, problem.best_fitness


# =============================================================================
# Multi-Board Evaluation
# =============================================================================


def evaluate_multi_board(
    candidate_weights: HeuristicWeights,
    baseline_weights: HeuristicWeights,
    config: "CMAESConfig",
) -> tuple[float, dict[str, float]]:
    """Evaluate candidate across multiple board types for generalization.

    Returns average fitness and per-board breakdown.
    """
    if not config.eval_boards:
        # Default to single board
        fitness, stats = evaluate_fitness(candidate_weights, baseline_weights, config)
        return fitness, {config.board_type: fitness}

    per_board_fitness = {}

    for board_type in config.eval_boards:
        # Create board-specific config
        board_config = CMAESConfig(
            backend=config.backend,
            mode="local",
            generations=config.generations,
            population_size=config.population_size,
            sigma=config.sigma,
            games_per_eval=config.games_per_eval,
            max_moves=config.max_moves,
            board_type=board_type,
            num_players=config.num_players,
            device=config.device,
            use_gpu=config.use_gpu,
            gpu_batch_size=config.gpu_batch_size,
            baseline_weights=baseline_weights,
        )

        fitness, _ = evaluate_fitness(candidate_weights, baseline_weights, board_config)
        per_board_fitness[board_type] = fitness

    # Average fitness across boards
    avg_fitness = sum(per_board_fitness.values()) / len(per_board_fitness)

    return avg_fitness, per_board_fitness


# =============================================================================
# Distributed Worker Mode
# =============================================================================


def run_worker_mode(config: "CMAESConfig"):
    """Run as a distributed worker, receiving tasks from coordinator or queue."""
    if config.queue_backend and QUEUE_AVAILABLE:
        run_queue_worker(config)
    else:
        run_http_worker(config)


def run_http_worker(config: "CMAESConfig"):
    """Run HTTP worker for distributed evaluation."""
    logger.info(f"Starting HTTP worker on port {config.worker_port}")

    class WorkerHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/evaluate":
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                task = json.loads(body)

                weights = task["weights"]
                baseline = task.get("baseline", config.baseline_weights)

                fitness, stats = evaluate_fitness(
                    candidate_weights=weights,
                    baseline_weights=baseline,
                    config=config,
                )

                response = {"fitness": fitness, "stats": stats}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(404)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_error(404)

    server = HTTPServer(("0.0.0.0", config.worker_port), WorkerHandler)
    logger.info(f"Worker listening on port {config.worker_port}")
    server.serve_forever()


def run_queue_worker(config: "CMAESConfig"):
    """Run queue-based worker for cloud deployment."""
    if not QUEUE_AVAILABLE:
        raise ImportError("Queue modules not available")

    logger.info(f"Starting queue worker (backend: {config.queue_backend})")

    queue = get_task_queue(config.queue_backend)
    running = True

    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    while running:
        task = queue.get_task(timeout=5)
        if task is None:
            continue

        try:
            weights = task.weights
            baseline = task.baseline or config.baseline_weights

            fitness, stats = evaluate_fitness(
                candidate_weights=weights,
                baseline_weights=baseline,
                config=config,
            )

            result = EvalResult(
                task_id=task.task_id,
                fitness=fitness,
                stats=stats,
            )
            queue.put_result(result)

        except Exception as e:
            logger.error(f"Task failed: {e}")
            queue.put_result(EvalResult(
                task_id=task.task_id,
                fitness=0.0,
                error=str(e),
            ))

    logger.info("Worker shutdown complete")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CMAESConfig:
    """Unified configuration for CMA-ES optimization."""

    # Backend selection
    backend: str = "evotorch"  # evotorch, pycma
    mode: str = "local"  # local, distributed, worker, iterative

    # Algorithm parameters
    generations: int = 50
    population_size: int = 32
    sigma: float = 0.5
    games_per_eval: int = 20
    max_moves: int = 200

    # Game settings
    board_type: str = "square8"
    num_players: int = 2

    # Device settings
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)
    use_gpu: bool = True
    gpu_batch_size: int = 64

    # Iterative mode settings
    max_iterations: int = 10
    improvement_threshold: float = 0.55
    plateau_generations: int = 5

    # Distributed settings
    distributed: bool = False
    workers: list[str] = field(default_factory=list)
    worker_port: int = 8766
    num_actors: int = 1

    # Queue settings (cloud)
    queue_backend: str | None = None  # redis, sqs
    queue_timeout: float = 300.0

    # Output settings
    output_dir: str = "logs/cmaes"
    run_id: str | None = None

    # Baseline weights
    baseline_weights: HeuristicWeights = field(
        default_factory=lambda: dict(BASE_V1_BALANCED_WEIGHTS)
    )
    baseline_path: str | None = None

    # Resume
    resume_from: str | None = None

    # Multi-board evaluation (for cross-board generalization)
    eval_boards: list[str] = field(default_factory=list)
    multi_board: bool = False


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified CMA-ES optimization for RingRift heuristic weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: GPU-accelerated EvoTorch
  python scripts/run_unified_cmaes.py --board square8 --num-players 2

  # Distributed across workers
  python scripts/run_unified_cmaes.py --board square8 --distributed \\
      --workers http://gh200-e:8766,http://gh200-f:8766

  # Iterative self-improvement
  python scripts/run_unified_cmaes.py --board square8 --mode iterative \\
      --max-iterations 10

  # Worker mode
  python scripts/run_unified_cmaes.py --mode worker --port 8766

  # Legacy pycma backend
  python scripts/run_unified_cmaes.py --board square8 --backend pycma
        """,
    )

    # Mode and backend
    parser.add_argument("--mode", choices=["local", "distributed", "worker", "iterative"],
                        default="local")
    parser.add_argument("--backend", choices=["evotorch", "pycma"], default="evotorch")

    # Algorithm parameters
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--games-per-eval", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=200)

    # Game settings
    parser.add_argument("--board", choices=["square8", "square19", "hex8", "hexagonal"],
                        default="square8")
    parser.add_argument("--num-players", type=int, choices=[2, 3, 4], default=2)

    # Device settings
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gpu-batch-size", type=int, default=64)

    # Iterative settings
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--improvement-threshold", type=float, default=0.55)
    parser.add_argument("--plateau-generations", type=int, default=5)

    # Distributed settings
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--workers", type=str, default="")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--num-actors", type=int, default=1)

    # Queue settings
    parser.add_argument("--queue-backend", choices=["redis", "sqs"], default=None)

    # Multi-board evaluation
    parser.add_argument("--multi-board", action="store_true",
                        help="Evaluate on multiple boards for generalization")
    parser.add_argument("--eval-boards", type=str, default="",
                        help="Comma-separated boards to evaluate on (default: square8,hexagonal,square19)")

    # Output
    parser.add_argument("--output", default="logs/cmaes")
    parser.add_argument("--run-id", type=str, default=None)

    # Baseline
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    # Utility
    parser.add_argument("--check-deps", action="store_true")

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        print(f"PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}")
        print(f"CUDA: {'✓' if CUDA_AVAILABLE else '✗'}")
        print(f"EvoTorch: {'✓' if EVOTORCH_AVAILABLE else '✗'}")
        print(f"pycma: {'✓' if PYCMA_AVAILABLE else '✗'}")
        print(f"GPU Games: {'✓' if GPU_GAMES_AVAILABLE else '✗'}")
        print(f"Weights: {NUM_WEIGHTS}")
        return

    # Load baseline weights
    baseline_weights = dict(BASE_V1_BALANCED_WEIGHTS)
    if args.baseline:
        baseline_weights = load_weights_from_file(args.baseline)
        logger.info(f"Loaded baseline from: {args.baseline}")

    # Parse workers
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]

    # Parse eval boards
    if args.eval_boards:
        eval_boards = [b.strip() for b in args.eval_boards.split(",") if b.strip()]
    elif args.multi_board:
        eval_boards = ["square8", "hexagonal", "square19"]
    else:
        eval_boards = []

    # Create config
    config = CMAESConfig(
        backend=args.backend,
        mode=args.mode if not args.distributed else "distributed",
        generations=args.generations,
        population_size=args.population_size,
        sigma=args.sigma,
        games_per_eval=args.games_per_eval,
        max_moves=args.max_moves,
        board_type=args.board,
        num_players=args.num_players,
        device=args.device,
        use_gpu=not args.no_gpu,
        gpu_batch_size=args.gpu_batch_size,
        max_iterations=args.max_iterations,
        improvement_threshold=args.improvement_threshold,
        plateau_generations=args.plateau_generations,
        distributed=args.distributed,
        workers=workers,
        worker_port=args.port,
        num_actors=args.num_actors,
        queue_backend=args.queue_backend,
        output_dir=args.output,
        run_id=args.run_id,
        baseline_weights=baseline_weights,
        resume_from=args.resume_from,
        eval_boards=eval_boards,
        multi_board=args.multi_board or bool(eval_boards),
    )

    # Worker mode doesn't need CMA-ES backend - run immediately
    if config.mode == "worker":
        run_worker_mode(config)
        return

    # Select backend (only needed for non-worker modes)
    if config.backend == "evotorch" and not EVOTORCH_AVAILABLE:
        logger.warning("EvoTorch not available, falling back to pycma")
        config.backend = "pycma"

    if config.backend == "pycma" and not PYCMA_AVAILABLE:
        logger.error("No CMA-ES backend available. Install evotorch or cma.")
        sys.exit(1)

    # Run appropriate mode
    if config.mode == "distributed":
        best_weights, best_fitness = run_distributed_cmaes(config)
    elif config.mode == "iterative":
        best_weights, best_fitness = run_iterative_cmaes(config)
    elif config.backend == "evotorch":
        best_weights, best_fitness = run_evotorch_cmaes(config)
    else:
        best_weights, best_fitness = run_pycma_cmaes(config)

    # Print summary (for non-worker modes)
    if config.mode != "worker":
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info("\nTop weight changes from baseline:")
        changes = []
        for key in HEURISTIC_WEIGHT_KEYS:
            baseline = config.baseline_weights.get(key, 0.0)
            optimized = best_weights.get(key, 0.0)
            if baseline != 0:
                delta_pct = (optimized - baseline) / abs(baseline) * 100
            else:
                delta_pct = 0
            changes.append((key, baseline, optimized, delta_pct))

        # Sort by absolute change
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        for key, baseline, optimized, delta in changes[:10]:
            logger.info(f"  {key}: {baseline:.2f} -> {optimized:.2f} ({delta:+.1f}%)")
        logger.info(f"  ... and {len(changes) - 10} more weights")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
