#!/usr/bin/env python
"""Iterative CMA-ES self-play improvement pipeline.

This script implements iterative self-play improvement similar to AlphaZero's
training loop. After each CMA-ES run:

1. If improvement > threshold: promote new weights as baseline, restart
2. If plateau detected: declare convergence

Usage (local):
    python scripts/run_iterative_cmaes.py \
        --board square8 \
        --num-players 2 \
        --generations-per-iter 15 \
        --max-iterations 10 \
        --improvement-threshold 0.55 \
        --plateau-generations 5 \
        --output-dir logs/cmaes/iterative/square8_2p

Usage (distributed):
    python scripts/run_iterative_cmaes.py \
        --board square8 \
        --num-players 3 \
        --generations-per-iter 10 \
        --max-iterations 5 \
        --output-dir logs/cmaes/iterative/square8_3p_dist \
        --distributed \
        --workers http://worker1:8000,http://worker2:8000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_profile_key(num_players: int) -> str:
    """Get the profile key for trained_heuristic_profiles.json."""
    return f"heuristic_v1_{num_players}p"


def load_trained_profiles(profiles_path: str) -> Dict[str, Any]:
    """Load the trained heuristic profiles JSON."""
    if not os.path.exists(profiles_path):
        return {
            "version": "1.0.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "description": "CMA-ES optimized heuristic weight profiles",
            "profiles": {},
            "training_metadata": {},
        }
    with open(profiles_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_trained_profiles(profiles: Dict[str, Any], profiles_path: str) -> None:
    """Save the trained heuristic profiles JSON."""
    os.makedirs(os.path.dirname(profiles_path) or ".", exist_ok=True)
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def load_checkpoint_fitness(checkpoint_path: str) -> Tuple[float, Dict[str, float]]:
    """Load fitness and weights from a checkpoint file."""
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fitness", 0.0), data.get("weights", {})


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Tuple[int, str]]:
    """Find the latest checkpoint in a directory.

    Returns (generation, path) or None.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    latest_gen = -1
    latest_path = None

    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("checkpoint_gen") and name.endswith(".json")):
            continue
        try:
            gen_str = name[len("checkpoint_gen"):-len(".json")]
            gen_num = int(gen_str)
            if gen_num > latest_gen:
                latest_gen = gen_num
                latest_path = os.path.join(checkpoint_dir, name)
        except ValueError:
            continue

    return (latest_gen, latest_path) if latest_path else None


def extract_fitness_history(checkpoint_dir: str) -> List[Tuple[int, float]]:
    """Extract fitness progression from all checkpoints."""
    history = []
    if not os.path.exists(checkpoint_dir):
        return history

    for name in sorted(os.listdir(checkpoint_dir)):
        if not (name.startswith("checkpoint_gen") and name.endswith(".json")):
            continue
        try:
            gen_str = name[len("checkpoint_gen"):-len(".json")]
            gen_num = int(gen_str)
            path = os.path.join(checkpoint_dir, name)
            fitness, _ = load_checkpoint_fitness(path)
            history.append((gen_num, fitness))
        except (ValueError, json.JSONDecodeError):
            continue

    return sorted(history, key=lambda x: x[0])


def detect_plateau(
    fitness_history: List[Tuple[int, float]],
    plateau_generations: int = 5,
    improvement_threshold: float = 0.01,
) -> bool:
    """Detect if fitness has plateaued.

    Returns True if fitness hasn't improved by more than threshold
    for the last N generations.
    """
    if len(fitness_history) < plateau_generations:
        return False

    recent = fitness_history[-plateau_generations:]
    max_fitness = max(f for _, f in recent)
    min_fitness = min(f for _, f in recent)

    return (max_fitness - min_fitness) < improvement_threshold


def run_cmaes_iteration(
    iteration: int,
    board: str,
    num_players: int,
    generations: int,
    population_size: int,
    games_per_eval: int,
    state_pool_id: str,
    output_dir: str,
    baseline_path: Optional[str] = None,
    seed: int = 42,
    max_moves: int = 200,
    eval_randomness: float = 0.02,
    progress_interval_sec: int = 30,
    no_record: bool = False,
    sigma: float = 0.5,
    inject_profiles: Optional[List[str]] = None,
    # Distributed mode options
    distributed: bool = False,
    workers: Optional[str] = None,
    discover_workers: bool = False,
    min_workers: int = 1,
    mode: str = "local",
) -> Tuple[str, int]:
    """Run a single CMA-ES iteration.

    Returns (run_dir, return_code).
    """
    run_id = f"iter{iteration:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output = os.path.join(output_dir, f"iter{iteration:02d}")

    cmd = [
        sys.executable,
        "scripts/run_cmaes_optimization.py",
        "--generations", str(generations),
        "--population-size", str(population_size),
        "--games-per-eval", str(games_per_eval),
        "--board", board,
        "--eval-boards", board,
        "--eval-mode", "multi-start",
        "--state-pool-id", state_pool_id,
        "--eval-randomness", str(eval_randomness),
        "--seed", str(seed + iteration * 1000),
        "--max-moves", str(max_moves),
        "--num-players", str(num_players),
        "--progress-interval-sec", str(progress_interval_sec),
        "--sigma", str(sigma),
        "--output", run_output,
    ]

    if baseline_path and os.path.exists(baseline_path):
        cmd.extend(["--baseline", baseline_path])

    if no_record:
        cmd.append("--no-record")

    # Only inject profiles on the first iteration to seed initial population
    if inject_profiles and iteration == 1:
        cmd.extend(["--inject-profiles", ",".join(inject_profiles)])

    # Add distributed mode flags
    if distributed:
        cmd.append("--distributed")
        if workers:
            cmd.extend(["--workers", workers])
        if discover_workers:
            cmd.append("--discover-workers")
        cmd.extend(["--min-workers", str(min_workers)])

    # Add deployment mode for host selection
    if mode != "local":
        cmd.extend(["--mode", mode])

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    env = os.environ.copy()
    env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

    result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.dirname(__file__)))

    # Find the actual run directory (has timestamp in name)
    runs_dir = os.path.join(run_output, "runs")
    if os.path.exists(runs_dir):
        subdirs = sorted(os.listdir(runs_dir))
        if subdirs:
            return os.path.join(runs_dir, subdirs[-1]), result.returncode

    return run_output, result.returncode


def update_baseline(
    best_weights: Dict[str, float],
    best_fitness: float,
    iteration: int,
    run_id: str,
    board: str,
    num_players: int,
    profiles_path: str,
    output_dir: str,
) -> str:
    """Update baseline weights and save new baseline file.

    Returns path to new baseline file.
    """
    # Save iteration-specific baseline
    baseline_path = os.path.join(output_dir, f"baseline_iter{iteration:02d}.json")
    baseline_data = {
        "weights": best_weights,
        "fitness": best_fitness,
        "iteration": iteration,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    # Update trained profiles
    profiles = load_trained_profiles(profiles_path)
    profile_key = get_profile_key(num_players)

    # Round weights to 2 decimal places for cleaner JSON
    rounded_weights = {k: round(v, 2) for k, v in best_weights.items()}
    profiles["profiles"][profile_key] = rounded_weights

    if "training_metadata" not in profiles:
        profiles["training_metadata"] = {}

    profiles["training_metadata"][profile_key] = {
        "source": "Iterative CMA-ES",
        "iteration": iteration,
        "run_id": run_id,
        "board": board,
        "fitness": round(best_fitness, 3),
    }
    profiles["version"] = "1.2.0"
    profiles["updated"] = datetime.now().strftime("%Y-%m-%d")

    save_trained_profiles(profiles, profiles_path)
    print(f"Updated {profiles_path} with {profile_key}")

    return baseline_path


def get_baseline_fitness(num_players: int) -> float:
    """Get expected normalized fitness when playing against equal-strength opponents.

    CMA-ES normalizes all fitness to [0, 1] range where fitness = win rate.
    For equal-strength opponents, expected win rate = 1/n.

    - 2p: 0.50 (50% win rate)
    - 3p: 0.333 (33.3% win rate)
    - 4p: 0.25 (25% win rate)
    """
    return 1.0 / num_players


def get_default_improvement_threshold(num_players: int) -> float:
    """Get default improvement threshold for promotion.

    This is baseline + a margin that indicates meaningful improvement.
    - 2p: 0.50 + 0.05 = 0.55 (55% win rate)
    - 3p: 0.333 + 0.05 = 0.383 (38.3% win rate vs 33.3% baseline)
    - 4p: 0.25 + 0.05 = 0.30 (30% win rate vs 25% baseline)
    """
    baseline = get_baseline_fitness(num_players)
    margin = 0.05
    return baseline + margin


def run_iterative_pipeline(
    board: str,
    num_players: int,
    generations_per_iter: int,
    max_iterations: int,
    improvement_threshold: Optional[float],
    plateau_generations: int,
    population_size: int,
    games_per_eval: int,
    output_dir: str,
    profiles_path: str,
    state_pool_id: Optional[str] = None,
    seed: int = 42,
    max_moves: int = 200,
    eval_randomness: float = 0.02,
    no_record: bool = False,
    sigma: float = 0.5,
    inject_profiles: Optional[List[str]] = None,
    # Distributed mode options
    distributed: bool = False,
    workers: Optional[str] = None,
    discover_workers: bool = False,
    min_workers: int = 1,
    mode: str = "local",
) -> None:
    """Run the iterative CMA-ES pipeline."""

    # Determine state pool ID
    if state_pool_id is None:
        if num_players == 2:
            state_pool_id = "v1"
        else:
            state_pool_id = f"{num_players}p_v1"

    # Use player-count-appropriate threshold
    if improvement_threshold is None:
        improvement_threshold = get_default_improvement_threshold(num_players)

    baseline_fitness = get_baseline_fitness(num_players)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline state
    pipeline_state_path = os.path.join(output_dir, "pipeline_state.json")
    pipeline_state = {
        "board": board,
        "num_players": num_players,
        "started": datetime.now().isoformat(),
        "iterations": [],
        "converged": False,
        "final_fitness": None,
        "baseline_fitness": baseline_fitness,
        "improvement_threshold": improvement_threshold,
    }

    baseline_path: Optional[str] = None
    best_overall_fitness = baseline_fitness

    print(f"\n{'#'*60}")
    print(f"ITERATIVE CMA-ES PIPELINE")
    print(f"{'#'*60}")
    print(f"Board: {board}")
    print(f"Players: {num_players}")
    print(f"Generations per iteration: {generations_per_iter}")
    print(f"Max iterations: {max_iterations}")
    print(f"Improvement threshold: {improvement_threshold}")
    print(f"Plateau detection: {plateau_generations} generations")
    print(f"State pool: {state_pool_id}")
    print(f"Output: {output_dir}")
    if distributed:
        print(f"Distributed mode: ENABLED")
        if workers:
            print(f"  Workers: {workers}")
        if discover_workers:
            print(f"  Worker discovery: enabled")
        print(f"  Min workers: {min_workers}")
        if not no_record:
            print(f"  Game recording: enabled (games collected from workers)")
    else:
        print(f"Distributed mode: disabled (local)")
    print()

    for iteration in range(1, max_iterations + 1):
        iter_start = time.time()

        # Run CMA-ES
        run_dir, return_code = run_cmaes_iteration(
            iteration=iteration,
            board=board,
            num_players=num_players,
            generations=generations_per_iter,
            population_size=population_size,
            games_per_eval=games_per_eval,
            state_pool_id=state_pool_id,
            output_dir=output_dir,
            baseline_path=baseline_path,
            seed=seed,
            max_moves=max_moves,
            eval_randomness=eval_randomness,
            no_record=no_record,
            sigma=sigma,
            inject_profiles=inject_profiles,
            # Distributed mode
            distributed=distributed,
            workers=workers,
            discover_workers=discover_workers,
            min_workers=min_workers,
            mode=mode,
        )

        if return_code != 0:
            print(f"ERROR: CMA-ES iteration {iteration} failed with code {return_code}")
            break

        # Extract results
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        latest = find_latest_checkpoint(checkpoint_dir)

        if not latest:
            print(f"ERROR: No checkpoints found in {checkpoint_dir}")
            break

        gen, checkpoint_path = latest
        fitness, weights = load_checkpoint_fitness(checkpoint_path)

        # Check fitness history for plateau
        fitness_history = extract_fitness_history(checkpoint_dir)
        is_plateau = detect_plateau(fitness_history, plateau_generations)

        iter_duration = time.time() - iter_start

        # Record iteration
        iter_record = {
            "iteration": iteration,
            "run_dir": run_dir,
            "final_generation": gen,
            "fitness": fitness,
            "duration_sec": round(iter_duration, 1),
            "plateau_detected": is_plateau,
        }
        pipeline_state["iterations"].append(iter_record)

        print(f"\n--- Iteration {iteration} Results ---")
        print(f"Final fitness: {fitness:.3f}")
        print(f"Duration: {iter_duration/60:.1f} min")
        print(f"Plateau detected: {is_plateau}")

        # Check if we should continue
        if fitness > improvement_threshold:
            print(f"✓ Fitness {fitness:.3f} > threshold {improvement_threshold}")

            # Update baseline
            baseline_path = update_baseline(
                best_weights=weights,
                best_fitness=fitness,
                iteration=iteration,
                run_id=os.path.basename(run_dir),
                board=board,
                num_players=num_players,
                profiles_path=profiles_path,
                output_dir=output_dir,
            )
            best_overall_fitness = fitness
            print(f"New baseline saved to: {baseline_path}")

            if is_plateau:
                print("\nPlateau detected within iteration - trying one more with new baseline")
                # Continue to see if new baseline opens up improvement

        else:
            print(f"✗ Fitness {fitness:.3f} <= threshold {improvement_threshold}")
            print("No significant improvement over baseline")

            # Convergence: plateau detected or fitness near/below baseline
            convergence_floor = baseline_fitness + 0.02
            if is_plateau or fitness <= convergence_floor:
                print("\n*** CONVERGENCE REACHED ***")
                pipeline_state["converged"] = True
                pipeline_state["final_fitness"] = best_overall_fitness
                break

        # Save pipeline state
        with open(pipeline_state_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_state, f, indent=2)

    # Final summary
    pipeline_state["finished"] = datetime.now().isoformat()
    pipeline_state["final_fitness"] = best_overall_fitness

    with open(pipeline_state_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_state, f, indent=2)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {len(pipeline_state['iterations'])}")
    print(f"Final fitness: {best_overall_fitness:.3f}")
    print(f"Converged: {pipeline_state['converged']}")
    print(f"Pipeline state: {pipeline_state_path}")
    if baseline_path:
        print(f"Final baseline: {baseline_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative CMA-ES self-play improvement pipeline"
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
    parser.add_argument(
        "--generations-per-iter",
        type=int,
        default=15,
        help="CMA-ES generations per iteration",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=None,
        help="Minimum fitness (win rate) to promote as new baseline. "
             "Default: 1/n + 0.05 (2p: 0.55, 3p: 0.383, 4p: 0.30)",
    )
    parser.add_argument(
        "--plateau-generations",
        type=int,
        default=5,
        help="Generations without improvement to detect plateau",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=14,
        help="CMA-ES population size",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=20,
        help="Games per fitness evaluation",
    )
    parser.add_argument(
        "--state-pool-id",
        type=str,
        default=None,
        help="State pool ID (default: v1 for 2p, {n}p_v1 for 3/4p)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for iterations",
    )
    parser.add_argument(
        "--profiles-path",
        type=str,
        default="data/trained_heuristic_profiles.json",
        help="Path to trained profiles JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Max moves per game",
    )
    parser.add_argument(
        "--eval-randomness",
        type=float,
        default=0.02,
        help="Evaluation randomness",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help=(
            "Disable game recording. By default, all self-play games are "
            "recorded to {run_dir}/games.db for later replay and analysis."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help=(
            "Initial CMA-ES step size (sigma). Higher values explore more of "
            "the search space initially. Default: 0.5. Consider 0.8-1.0 for "
            "more exploration in early training."
        ),
    )
    parser.add_argument(
        "--inject-profiles",
        type=str,
        default=None,
        help=(
            "Comma-separated list of JSON profile file paths to inject into "
            "the initial CMA-ES population (first iteration only). These "
            "profiles will be evaluated alongside the randomly sampled "
            "candidates, helping seed the search with known-good weights from "
            "other player counts or board types."
        ),
    )

    # Distributed mode arguments
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "Enable distributed mode. Fitness evaluations will be distributed "
            "across worker nodes for parallel processing."
        ),
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of worker URLs for distributed mode "
            "(e.g., 'http://worker1:8000,http://worker2:8000')"
        ),
    )
    parser.add_argument(
        "--discover-workers",
        action="store_true",
        help="Auto-discover workers via mDNS/service discovery",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required to start (default: 1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "lan", "aws", "hybrid"],
        default="local",
        help=(
            "Deployment mode for host selection. 'local' runs locally only, "
            "'lan' uses local Mac cluster workers, 'aws' uses AWS staging "
            "workers (square8 only due to 16GB RAM limit), 'hybrid' uses both. "
            "Default: local."
        ),
    )

    args = parser.parse_args()

    # Parse inject_profiles from comma-separated string to list
    inject_profiles_list = None
    if args.inject_profiles:
        inject_profiles_list = [p.strip() for p in args.inject_profiles.split(",")]

    run_iterative_pipeline(
        board=args.board,
        num_players=args.num_players,
        generations_per_iter=args.generations_per_iter,
        max_iterations=args.max_iterations,
        improvement_threshold=args.improvement_threshold,
        plateau_generations=args.plateau_generations,
        population_size=args.population_size,
        games_per_eval=args.games_per_eval,
        output_dir=args.output_dir,
        profiles_path=args.profiles_path,
        state_pool_id=args.state_pool_id,
        seed=args.seed,
        max_moves=args.max_moves,
        eval_randomness=args.eval_randomness,
        no_record=args.no_record,
        sigma=args.sigma,
        inject_profiles=inject_profiles_list,
        # Distributed mode
        distributed=args.distributed,
        workers=args.workers,
        discover_workers=args.discover_workers,
        min_workers=args.min_workers,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
