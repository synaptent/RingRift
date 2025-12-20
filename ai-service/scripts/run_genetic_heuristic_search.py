#!/usr/bin/env python
"""
Experimental genetic search over HeuristicAI weight profiles.

This script implements a simple genetic-style search on top of the same
HeuristicWeights representation and fitness function used by CMA-ES.
It is intended as a complementary tool for exploring the weight space
more broadly (large, structured perturbations) before or alongside
local CMA-ES refinement and plateau diagnostics.

Key characteristics:

- Individuals are HeuristicWeights dicts
  (same keys as BASE_V1_BALANCED_WEIGHTS).
- Fitness is win rate vs the balanced baseline, evaluated by
  `evaluate_fitness(candidate, baseline, ...)` from
  `scripts/run_cmaes_optimization.py`, so CMA-ES and GA share a single
  fitness definition and instrumentation (including per-individual
  wins/draws/losses and L2-distance diagnostics via an optional debug
  callback).
- Selection keeps the top-K individuals each generation (elitism).
- Mutation applies per-weight Gaussian noise with configurable sigma.
- Evaluation can use either:
  - `eval_mode="initial-only"` (default) to start games from the
    normal initial state, or
  - `eval_mode="multi-start"` with a `state_pool_id` pointing at a
    JSONL state pool loaded via `app/training/eval_pools.py`, which
    samples mid-game positions to reduce seed/opening bias.

Usage (from ai-service/):

    python scripts/run_genetic_heuristic_search.py \\
        --generations 3 \\
        --population-size 8 \\
        --games-per-eval 16 \\
        --sigma 2.0 \\
        --board square8 \\
        --seed 12345

The script prints per-generation statistics and writes the best weights
to `logs/ga/runs/<run_id>/best_weights.json` using the same JSON schema
(`{\"weights\": { ... }}`) as CMA-ES so that downstream tooling can
consume them if desired.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

# Allow imports from app/ when run as a script.
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import BoardType  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HeuristicWeights,
)
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    BOARD_NAME_TO_TYPE,
    evaluate_fitness,
    evaluate_fitness_over_boards,
    FitnessDebugCallback,
)
from app.utils.progress_reporter import (  # noqa: E402
    OptimizationProgressReporter,
)


BOARD_CHOICES = ["square8", "square19", "hex"]

# For CLI wiring we reuse the same canonical mapping as the CMA-ES script so
# that board names stay consistent across optimization harnesses.
BOARD_TYPE_MAP = BOARD_NAME_TO_TYPE


@dataclass
class Individual:
    weights: HeuristicWeights
    fitness: float | None = None


def _mutate_weights(
    parent: HeuristicWeights,
    rng: np.random.Generator,
    sigma: float,
    lo: float = 0.0,
    hi: float = 50.0,
) -> HeuristicWeights:
    """Return a mutated copy of parent weights with per-key Gaussian noise."""
    child: HeuristicWeights = {}
    for key, value in parent.items():
        noise = rng.normal(0.0, sigma)
        mutated = float(np.clip(value + noise, lo, hi))
        child[key] = mutated
    return child


def _initial_population(
    size: int,
    sigma: float,
    rng: np.random.Generator,
) -> list[Individual]:
    """Create an initial population around the balanced baseline."""
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    pop: list[Individual] = [Individual(weights=baseline)]
    while len(pop) < size:
        mutated = _mutate_weights(baseline, rng, sigma)
        pop.append(Individual(weights=mutated))
    return pop


def _evaluate_population(
    population: list[Individual],
    games_per_eval: int,
    boards: list[BoardType],
    *,
    eval_mode: str = "initial-only",
    state_pool_id: str = "v1",
    seed: int | None = None,
    eval_randomness: float = 0.0,
    generation_index: int | None = None,
    progress_reporter: OptimizationProgressReporter | None = None,
    progress_interval_sec: float = 10.0,
    enable_eval_progress: bool = True,
) -> None:
    """Evaluate a population on one or more boards and update fitness.

    Fitness values are written back to the provided population in-place.
    Per-individual diagnostics (wins/draws/losses and L2 distance to the
    baseline) are emitted via the shared :class:`FitnessDebugCallback`
    mechanism exposed by ``evaluate_fitness``.
    """
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    total_individuals = len(population)
    total_games = total_individuals * games_per_eval * max(1, len(boards))
    completed_games = 0

    def _make_debug_callback(ind_index: int) -> FitnessDebugCallback:
        def _callback(
            candidate_w: HeuristicWeights,
            baseline_w: HeuristicWeights,
            board_type: BoardType,
            stats: dict[str, Any],
            *,
            _total_individuals: int = total_individuals,
            _generation_index: int | None = generation_index,
        ) -> None:
            g = _generation_index if _generation_index is not None else -1
            wins = int(stats.get("wins", 0))
            draws = int(stats.get("draws", 0))
            losses = int(stats.get("losses", 0))
            fitness = float(stats.get("fitness", 0.0))
            games = int(stats.get("games", stats.get("games_per_eval", 0)))
            l2 = float(
                stats.get(
                    "weight_l2_to_baseline",
                    stats.get("weight_l2", 0.0),
                )
            )

            try:
                board_label = board_type.value
            except AttributeError:
                board_label = str(board_type)

            print(
                "[GA] "
                f"gen={g} ind={ind_index}/{_total_individuals} "
                f"board={board_label} "
                f"fitness={fitness:.4f} "
                f"W={wins} D={draws} L={losses} "
                f"games={games} "
                f"||w-baseline||_2={l2:.3f}"
            )

        return _callback

    for idx, ind in enumerate(population, start=1):
        debug_cb = _make_debug_callback(idx)

        if len(boards) == 1:
            board_type = boards[0]
            ind.fitness = evaluate_fitness(
                candidate_weights=ind.weights,
                baseline_weights=baseline,
                games_per_eval=games_per_eval,
                board_type=board_type,
                verbose=False,
                opponent_mode="baseline-only",
                eval_mode=eval_mode,
                state_pool_id=state_pool_id,
                eval_randomness=eval_randomness,
                seed=seed,
                debug_callback=debug_cb,
            )
            games_this_individual = games_per_eval
        else:
            # Build progress label for per-board reporters when enabled
            progress_label: str | None = None
            if enable_eval_progress and generation_index is not None:
                progress_label = f"GA gen={generation_index} ind={idx}"

            agg, _per_board = evaluate_fitness_over_boards(
                candidate_weights=ind.weights,
                baseline_weights=baseline,
                games_per_eval=games_per_eval,
                boards=boards,
                opponent_mode="baseline-only",
                max_moves=200,
                verbose=False,
                eval_mode=eval_mode,
                state_pool_id=state_pool_id,
                seed=seed,
                eval_randomness=eval_randomness,
                debug_callback=debug_cb,
                progress_label=progress_label,
                progress_interval_sec=progress_interval_sec,
                enable_eval_progress=enable_eval_progress,
            )
            ind.fitness = agg
            games_this_individual = games_per_eval * len(boards)

        completed_games += games_this_individual
        remaining_games = max(0, total_games - completed_games)
        print(
            f"    progress: individual {idx}/{total_individuals}, "
            f"games {completed_games}/{total_games} "
            f"({remaining_games} remaining in generation)"
        )

        # Time-based progress reporting via the shared reporter
        if progress_reporter is not None:
            if progress_reporter is not None:
                progress_reporter.record_candidate(
                    candidate_idx=idx,
                    fitness=ind.fitness,
                    games_played=games_this_individual,
                )


def _select_elites(
    population: list[Individual],
    elite_count: int,
) -> list[Individual]:
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness or 0.0,
        reverse=True,
    )
    return sorted_pop[:elite_count]


def _next_generation(
    elites: list[Individual],
    population_size: int,
    sigma: float,
    rng: np.random.Generator,
) -> list[Individual]:
    next_pop: list[Individual] = []

    # Always carry the single best individual forward unchanged (elitism).
    if elites:
        next_pop.append(Individual(weights=dict(elites[0].weights)))

    # Fill the rest of the population with mutated copies of elites.
    while len(next_pop) < population_size:
        parent = elites[rng.integers(0, len(elites))]
        child_weights = _mutate_weights(parent.weights, rng, sigma)
        next_pop.append(Individual(weights=child_weights))

    return next_pop


def _save_best_weights(
    best: Individual,
    run_dir: str,
    generation: int,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "best_weights.json")
    payload = {
        "generation": generation,
        "fitness": best.fitness,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": best.weights,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental genetic search over HeuristicAI weights",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of generations to run (default: 3).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=8,
        help="Population size per generation (default: 8).",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=3,
        help="Number of elites to keep each generation (default: 3).",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=16,
        help="Games per individual evaluation (default: 16).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Mutation sigma for per-weight Gaussian noise (default: 2.0).",
    )
    parser.add_argument(
        "--board",
        type=str,
        choices=BOARD_CHOICES,
        default="square8",
        help="Board type to use for evaluation (default: square8).",
    )
    parser.add_argument(
        "--eval-boards",
        type=str,
        default=None,
        help=(
            "Comma-separated list of board types to evaluate on, e.g. "
            "'square8' (default), "
            "'square8,square19', or "
            "'square8,square19,hex'. "
            "If omitted, defaults to the single board from --board."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/ga",
        help="Base output directory for GA runs (default: logs/ga).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Logical run identifier; defaults to timestamp if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for population/mutation (default: 12345).",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["initial-only", "multi-start"],
        default="multi-start",
        help=(
            "Evaluation mode for GA fitness: same options as CMA-ES "
            "('initial-only' or 'multi-start'); default is 'multi-start', "
            "which evaluates from a pooled set of mid-game states."
        ),
    )
    parser.add_argument(
        "--state-pool-id",
        type=str,
        default="v1",
        help=("State pool identifier when using eval-mode=multi-start " "(default: v1)."),
    )
    parser.add_argument(
        "--eval-randomness",
        type=float,
        default=0.0,
        help=(
            "Optional randomness parameter for heuristic evaluation. "
            "0.0 (default) keeps evaluation deterministic; "
            "small positive values enable controlled stochastic tie-breaking."
        ),
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=10.0,
        help=(
            "Minimum seconds between optimisation progress log lines "
            "(default: 10.0). This is forwarded to the shared "
            "OptimizationProgressReporter."
        ),
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help=("Disable optimisation-level progress logging for GA runs, " "leaving only per-individual debug output."),
    )
    parser.add_argument(
        "--disable-eval-progress",
        action="store_true",
        help=(
            "Disable per-board evaluation progress reporters during "
            "multi-board GA runs, relying solely on the outer "
            "optimisation-level reporter. By default per-board "
            "evaluation progress is enabled when using multiple boards."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.population_size <= 0:
        raise SystemExit("population-size must be positive")
    if args.elite_count <= 0 or args.elite_count > args.population_size:
        raise SystemExit("elite-count must be in [1, population-size]")

    # Derive the list of boards to evaluate on. When --eval-boards is omitted,
    # we default to the single board selected via --board to preserve the
    # historical single-board behaviour.
    if args.eval_boards is None:
        eval_boards_str = args.board
    else:
        eval_boards_str = args.eval_boards

    raw_names = [name.strip() for name in eval_boards_str.split(",") if name.strip()]
    if not raw_names:
        raise SystemExit("At least one board must be specified in --eval-boards")

    boards: list[BoardType] = []
    for name in raw_names:
        try:
            boards.append(BOARD_NAME_TO_TYPE[name])
        except KeyError:
            raise SystemExit(f"Unknown board name in --eval-boards: {name!r}")

    rng = np.random.default_rng(args.seed)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("ga_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    print("=== Genetic Heuristic Search ===")
    print(f"Run id:           {run_id}")
    print(f"Run directory:    {run_dir}")
    print(f"Generations:      {args.generations}")
    print(f"Population size:  {args.population_size}")
    print(f"Elite count:      {args.elite_count}")
    print(f"Games per eval:   {args.games_per_eval}")
    print(f"Sigma:            {args.sigma}")
    print(f"Board:            {args.board}")
    boards_label = ",".join(b.value for b in boards)
    print(f"Eval boards:      {boards_label}")
    if len(boards) > 1:
        print("NOTE: Aggregate GA fitness is the mean across eval boards.")
    print(f"Seed:             {args.seed}")
    print(f"Eval randomness:  {args.eval_randomness}")
    print()

    population = _initial_population(args.population_size, args.sigma, rng)

    # Initialize progress reporter for time-based progress output, unless
    # explicitly disabled via CLI. This keeps default behaviour (~10s
    # heartbeats) while allowing very quiet runs when needed.
    progress_reporter: OptimizationProgressReporter | None
    if args.disable_progress:
        progress_reporter = None
    else:
        progress_reporter = OptimizationProgressReporter(
            total_generations=args.generations,
            candidates_per_generation=args.population_size,
            report_interval_sec=args.progress_interval_sec,
        )

    best_overall: Individual | None = None

    for gen in range(1, args.generations + 1):
        if progress_reporter is not None:
            progress_reporter.start_generation(gen)

        _evaluate_population(
            population,
            args.games_per_eval,
            boards,
            eval_mode=args.eval_mode,
            state_pool_id=args.state_pool_id,
            seed=args.seed,
            eval_randomness=args.eval_randomness,
            generation_index=gen,
            progress_reporter=progress_reporter,
            progress_interval_sec=args.progress_interval_sec,
            enable_eval_progress=not args.disable_eval_progress,
        )

        fitnesses = [ind.fitness or 0.0 for ind in population]
        mean_f = float(np.mean(fitnesses))
        std_f = float(np.std(fitnesses))
        max_f = float(np.max(fitnesses))

        elites = _select_elites(population, args.elite_count)
        best = elites[0]

        print(f"  mean={mean_f:.4f}, std={std_f:.4f}, " f"best={max_f:.4f}")

        # Report generation completion with statistics
        if progress_reporter is not None:
            progress_reporter.finish_generation(
                mean_fitness=mean_f,
                best_fitness=max_f,
                std_fitness=std_f,
            )

        # Track global best
        if best_overall is None or (best.fitness or 0.0) > (best_overall.fitness or 0.0):
            best_overall = Individual(
                weights=dict(best.weights),
                fitness=best.fitness,
            )
            _save_best_weights(best_overall, run_dir, generation=gen)

        # Prepare next generation
        if gen < args.generations:
            population = _next_generation(
                elites,
                args.population_size,
                args.sigma,
                rng,
            )

    # Emit final optimization summary
    if progress_reporter is not None:
        progress_reporter.finish()

    if best_overall is not None:
        print()
        print(
            f"Best overall fitness: {best_overall.fitness:.4f} "
            f"(saved to {os.path.join(run_dir, 'best_weights.json')})"
        )
    else:
        print("No individuals evaluated; nothing saved.")


if __name__ == "__main__":
    main()
