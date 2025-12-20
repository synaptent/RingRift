#!/usr/bin/env python
"""Iterative CMA-ES self-play improvement pipeline.

This script implements iterative self-play improvement similar to AlphaZero's
training loop. After each CMA-ES run:

1. If improvement > threshold: promote new weights as baseline, restart
2. If plateau detected: declare convergence

Usage (local):
    python scripts/run_iterative_cmaes.py \\
        --board square8 \\
        --num-players 2 \\
        --generations-per-iter 15 \\
        --max-iterations 10 \\
        --improvement-threshold 0.55 \\
        --plateau-generations 5 \\
        --output-dir logs/cmaes/iterative/square8_2p

Usage (distributed):
    python scripts/run_iterative_cmaes.py \\
        --board square8 \\
        --num-players 3 \\
        --generations-per-iter 10 \\
        --max-iterations 5 \\
        --output-dir logs/cmaes/iterative/square8_3p_dist \\
        --distributed \\
        --workers http://worker1:8000,http://worker2:8000

Usage with NN quality gate:
    # Only run NN-guided optimization if NN beats heuristic by 55%+
    python scripts/run_iterative_cmaes.py \\
        --board square8 \\
        --num-players 2 \\
        --output-dir logs/cmaes/iterative/square8_2p_gated \\
        --nn-quality-gate 0.55 \\
        --nn-model models/square8_2p_best.pth \\
        --nn-gate-games 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Optional

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.training.env import get_theoretical_max_moves


def check_nn_quality_gate(
    nn_model_path: str,
    board_type: str,
    num_players: int,
    quality_threshold: float,
    num_games: int = 20,
    max_moves: int = 10000,
) -> tuple[bool, float]:
    """Run a quick tournament to check if NN model meets quality threshold.

    This gates CMA-ES optimization on NN quality to avoid optimizing heuristics
    toward a weak neural network target.

    Args:
        nn_model_path: Path to the NN model checkpoint
        board_type: Board type (square8, square19, hex)
        num_players: Number of players (2, 3, 4)
        quality_threshold: Minimum win rate required (e.g., 0.55)
        num_games: Number of games for the quality check tournament
        max_moves: Maximum moves per game

    Returns:
        Tuple of (passed: bool, actual_winrate: float)
    """
    import re

    if not os.path.exists(nn_model_path):
        print(f"[NN Quality Gate] Model not found: {nn_model_path}")
        return False, 0.0

    print(f"\n{'='*60}")
    print("NN QUALITY GATE CHECK")
    print(f"{'='*60}")
    print(f"Model: {nn_model_path}")
    print(f"Board: {board_type}, Players: {num_players}")
    print(f"Threshold: {quality_threshold:.1%}")
    print(f"Games: {num_games}")
    print()

    # Run tournament: Neural vs Heuristic
    # Use run_ai_tournament.py for consistent evaluation
    cmd = [
        sys.executable,
        "scripts/run_ai_tournament.py",
        "--p1", "Neural",
        "--p1-model", nn_model_path,
        "--p2", "Heuristic",
        "--board", board_type,
        "--games", str(num_games),
        "--max-moves", str(max_moves),
    ]

    env = os.environ.copy()
    env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        output = result.stdout + result.stderr

        # Parse win rate from output
        # Look for patterns like "P1 wins: 12/20 (60.0%)" or "Win rate: 0.60"
        winrate = 0.0

        # Try percentage pattern first
        match = re.search(r"P1.*?(\d+\.?\d*)%", output)
        if match:
            winrate = float(match.group(1)) / 100.0
        else:
            # Try fraction pattern: "12/20"
            match = re.search(r"P1.*?(\d+)/(\d+)", output)
            if match:
                wins = int(match.group(1))
                total = int(match.group(2))
                winrate = wins / max(total, 1)
            else:
                # Try decimal pattern
                match = re.search(r"Win rate:\s*(\d+\.?\d*)", output)
                if match:
                    winrate = float(match.group(1))

        passed = winrate >= quality_threshold

        print("\nResults:")
        print(f"  Neural win rate: {winrate:.1%}")
        print(f"  Threshold: {quality_threshold:.1%}")
        print(f"  Status: {'PASSED ✓' if passed else 'FAILED ✗'}")

        if not passed:
            print("\n  WARNING: NN model does not meet quality threshold.")
            print("  CMA-ES will proceed with heuristic-only optimization.")

        return passed, winrate

    except subprocess.TimeoutExpired:
        print("[NN Quality Gate] Tournament timed out")
        return False, 0.0
    except Exception as e:
        print(f"[NN Quality Gate] Error running tournament: {e}")
        return False, 0.0


def get_profile_key(board: str, num_players: int) -> str:
    """Get the board-specific profile key for trained_heuristic_profiles.json.

    Uses the canonical board×player format: heuristic_v1_{board_abbrev}_{n}p

    Parameters
    ----------
    board:
        Board type: "square8", "square19", or "hex"
    num_players:
        Number of players (2, 3, or 4)

    Returns
    -------
    str
        Profile key like "heuristic_v1_sq8_2p" or "heuristic_v1_hex_3p"
    """
    board_abbrev = {
        "square8": "sq8",
        "square19": "sq19",
        "hexagonal": "hex",
        "hex": "hex",
    }.get(board, board[:3])
    return f"heuristic_v1_{board_abbrev}_{num_players}p"


def load_trained_profiles(profiles_path: str) -> dict[str, Any]:
    """Load the trained heuristic profiles JSON."""
    if not os.path.exists(profiles_path):
        return {
            "version": "1.0.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "description": "CMA-ES optimized heuristic weight profiles",
            "profiles": {},
            "training_metadata": {},
        }
    with open(profiles_path, encoding="utf-8") as f:
        return json.load(f)


def save_trained_profiles(profiles: dict[str, Any], profiles_path: str) -> None:
    """Save the trained heuristic profiles JSON."""
    os.makedirs(os.path.dirname(profiles_path) or ".", exist_ok=True)
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def load_checkpoint_fitness(checkpoint_path: str) -> tuple[float, dict[str, float]]:
    """Load fitness and weights from a checkpoint file."""
    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fitness", 0.0), data.get("weights", {})


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str] | None:
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
            gen_str = name[len("checkpoint_gen") : -len(".json")]
            gen_num = int(gen_str)
            if gen_num > latest_gen:
                latest_gen = gen_num
                latest_path = os.path.join(checkpoint_dir, name)
        except ValueError:
            continue

    return (latest_gen, latest_path) if latest_path else None


def extract_fitness_history(checkpoint_dir: str) -> list[tuple[int, float]]:
    """Extract fitness progression from all checkpoints."""
    history = []
    if not os.path.exists(checkpoint_dir):
        return history

    for name in sorted(os.listdir(checkpoint_dir)):
        if not (name.startswith("checkpoint_gen") and name.endswith(".json")):
            continue
        try:
            gen_str = name[len("checkpoint_gen") : -len(".json")]
            gen_num = int(gen_str)
            path = os.path.join(checkpoint_dir, name)
            fitness, _ = load_checkpoint_fitness(path)
            history.append((gen_num, fitness))
        except (ValueError, json.JSONDecodeError):
            continue

    return sorted(history, key=lambda x: x[0])


def detect_plateau(
    fitness_history: list[tuple[int, float]],
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
    baseline_path: str | None = None,
    seed: int = 42,
    max_moves: int | None = None,  # Auto-calculated if not specified
    eval_randomness: float = 0.02,
    progress_interval_sec: int = 30,
    no_record: bool = False,
    sigma: float = 0.5,
    inject_profiles: list[str] | None = None,
    # GPU acceleration options
    gpu: bool = False,
    gpu_batch_size: int = 128,
    # Distributed mode options
    distributed: bool = False,
    workers: str | None = None,
    discover_workers: bool = False,
    min_workers: int = 1,
    mode: str = "local",
    selfplay_data_dir: str | None = None,
) -> tuple[str, int]:
    """Run a single CMA-ES iteration.

    Returns (run_dir, return_code).
    """
    f"iter{iteration:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output = os.path.join(output_dir, f"iter{iteration:02d}")

    cmd = [
        sys.executable,
        "scripts/run_cmaes_optimization.py",
        "--generations",
        str(generations),
        "--population-size",
        str(population_size),
        "--games-per-eval",
        str(games_per_eval),
        "--board",
        board,
        "--eval-boards",
        board,
        "--eval-mode",
        "multi-start",
        "--state-pool-id",
        state_pool_id,
        "--eval-randomness",
        str(eval_randomness),
        "--seed",
        str(seed + iteration * 1000),
        "--max-moves",
        str(max_moves),
        "--num-players",
        str(num_players),
        "--progress-interval-sec",
        str(progress_interval_sec),
        "--sigma",
        str(sigma),
        "--output",
        run_output,
    ]

    if baseline_path and os.path.exists(baseline_path):
        cmd.extend(["--baseline", baseline_path])

    if no_record:
        cmd.append("--no-record")

    # GPU acceleration flags
    if gpu:
        cmd.append("--gpu")
        cmd.extend(["--gpu-batch-size", str(gpu_batch_size)])

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

    # Pass through selfplay data directory if specified
    if selfplay_data_dir:
        cmd.extend(["--selfplay-data-dir", selfplay_data_dir])

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
    best_weights: dict[str, float],
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
    profile_key = get_profile_key(board, num_players)

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
    improvement_threshold: float | None,
    plateau_generations: int,
    population_size: int,
    games_per_eval: int,
    output_dir: str,
    profiles_path: str,
    state_pool_id: str | None = None,
    seed: int = 42,
    max_moves: int | None = None,  # Auto-calculated if not specified
    eval_randomness: float = 0.02,
    no_record: bool = False,
    sigma: float = 0.5,
    inject_profiles: list[str] | None = None,
    # GPU acceleration options
    gpu: bool = False,
    gpu_batch_size: int = 128,
    # Distributed mode options
    distributed: bool = False,
    workers: str | None = None,
    discover_workers: bool = False,
    min_workers: int = 1,
    mode: str = "local",
    selfplay_data_dir: str | None = None,
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

    baseline_path: str | None = None
    best_overall_fitness = baseline_fitness

    print(f"\n{'#'*60}")
    print("ITERATIVE CMA-ES PIPELINE")
    print(f"{'#'*60}")
    print(f"Board: {board}")
    print(f"Players: {num_players}")
    print(f"Generations per iteration: {generations_per_iter}")
    print(f"Max iterations: {max_iterations}")
    print(f"Improvement threshold: {improvement_threshold}")
    print(f"Plateau detection: {plateau_generations} generations")
    print(f"State pool: {state_pool_id}")
    print(f"Output: {output_dir}")
    if gpu:
        print(f"GPU acceleration: ENABLED (batch_size={gpu_batch_size})")
    else:
        print("GPU acceleration: disabled (CPU mode)")
    if distributed:
        print("Distributed mode: ENABLED")
        if workers:
            print(f"  Workers: {workers}")
        if discover_workers:
            print("  Worker discovery: enabled")
        print(f"  Min workers: {min_workers}")
        if not no_record:
            print("  Game recording: enabled (games collected from workers)")
    else:
        print("Distributed mode: disabled (local)")
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
            # GPU acceleration
            gpu=gpu,
            gpu_batch_size=gpu_batch_size,
            # Distributed mode
            distributed=distributed,
            workers=workers,
            discover_workers=discover_workers,
            min_workers=min_workers,
            mode=mode,
            selfplay_data_dir=selfplay_data_dir,
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
    parser = argparse.ArgumentParser(description="Iterative CMA-ES self-play improvement pipeline")
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

    # GPU acceleration arguments
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Enable GPU-accelerated game evaluation. Uses ParallelGameRunner for "
            "batched CUDA evaluation with ~400x speedup over CPU. Requires CUDA. "
            "Enabled by default when CUDA is available."
        ),
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=128,
        help="Batch size for GPU game evaluation (default: 128)",
    )
    parser.add_argument(
        "--auto-gpu",
        action="store_true",
        default=True,
        help="Automatically enable GPU if CUDA is available (default: True)",
    )
    parser.add_argument(
        "--no-auto-gpu",
        action="store_true",
        help="Disable automatic GPU detection; only use GPU if --gpu is specified",
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
    parser.add_argument(
        "--selfplay-data-dir",
        type=str,
        default=None,
        help=(
            "Path to directory containing aggregated selfplay JSONL data from "
            "distributed cluster. Passed through to underlying CMA-ES optimization "
            "script. Expected structure: subdirectories like 'random_square8_2p/' "
            "each containing 'games.jsonl'."
        ),
    )

    # NN Quality Gate arguments
    parser.add_argument(
        "--nn-quality-gate",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Enable NN quality gate: before starting CMA-ES, run a tournament "
            "between the specified NN model and heuristic baseline. If the NN "
            "win rate is below THRESHOLD (e.g., 0.55), proceed with heuristic-only "
            "optimization. Requires --nn-model to be specified. "
            "Example: --nn-quality-gate 0.55"
        ),
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        default=None,
        help=(
            "Path to the neural network model checkpoint for quality gate check. "
            "Required when --nn-quality-gate is specified."
        ),
    )
    parser.add_argument(
        "--nn-gate-games",
        type=int,
        default=20,
        help=(
            "Number of games to play for the NN quality gate tournament. "
            "More games give more accurate estimates but take longer. Default: 20."
        ),
    )

    args = parser.parse_args()

    # Auto-calculate max_moves if not specified
    if args.max_moves is None:
        args.max_moves = get_theoretical_max_moves(args.board, args.num_players)
        print(f"[Auto] max_moves={args.max_moves} for {args.board} {args.num_players}p")

    # Validate NN quality gate arguments
    if args.nn_quality_gate is not None and args.nn_model is None:
        parser.error("--nn-quality-gate requires --nn-model to be specified")

    # Run NN quality gate check if specified
    nn_gate_passed = True
    nn_gate_winrate = None
    if args.nn_quality_gate is not None and args.nn_model is not None:
        nn_gate_passed, nn_gate_winrate = check_nn_quality_gate(
            nn_model_path=args.nn_model,
            board_type=args.board,
            num_players=args.num_players,
            quality_threshold=args.nn_quality_gate,
            num_games=args.nn_gate_games,
            max_moves=args.max_moves,
        )

        if not nn_gate_passed:
            print(f"\n[NN Quality Gate] NN model failed quality check "
                  f"({nn_gate_winrate:.1%} < {args.nn_quality_gate:.1%})")
            print("[NN Quality Gate] Proceeding with heuristic-only CMA-ES optimization")
        else:
            print(f"\n[NN Quality Gate] NN model passed quality check "
                  f"({nn_gate_winrate:.1%} >= {args.nn_quality_gate:.1%})")

    # Parse inject_profiles from comma-separated string to list
    inject_profiles_list = None
    if args.inject_profiles:
        inject_profiles_list = [p.strip() for p in args.inject_profiles.split(",")]

    # Determine GPU mode: auto-detect CUDA unless --no-auto-gpu specified
    use_gpu = args.gpu
    if not use_gpu and args.auto_gpu and not args.no_auto_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                print(f"[Auto-GPU] CUDA available ({torch.cuda.get_device_name(0)}), enabling GPU acceleration")
        except ImportError:
            print("[Auto-GPU] PyTorch not available, using CPU mode")

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
        # GPU acceleration
        gpu=use_gpu,
        gpu_batch_size=args.gpu_batch_size,
        # Distributed mode
        distributed=args.distributed,
        workers=args.workers,
        discover_workers=args.discover_workers,
        min_workers=args.min_workers,
        mode=args.mode,
        selfplay_data_dir=args.selfplay_data_dir,
    )


if __name__ == "__main__":
    main()
