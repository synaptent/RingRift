#!/usr/bin/env python
"""Sanity check for multi-board, multi-start heuristic evaluation.

This script evaluates:
- baseline vs baseline (BASE_V1_BALANCED_WEIGHTS vs itself)
- zero vs baseline (all heuristic weights set to 0.0 vs baseline)

using the canonical multi-board, multi-start configuration from
app.training.env.build_training_eval_kwargs.

It is intended as a quick diagnostic to confirm that:
- fitness is not structurally pinned at 0.5 for baseline vs baseline
- clearly bad weights (zero) score significantly worse than the baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure imports work when running from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
)
from app.models import BoardType  # type: ignore  # noqa: E402
from app.training.env import (  # type: ignore  # noqa: E402
    build_training_eval_kwargs,
)
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    evaluate_fitness,
    evaluate_fitness_over_boards,
)


def _format_per_board(per_board: Dict[BoardType, float]) -> str:
    """Return a compact string mapping board names to fitness values."""
    label_by_board = {
        BoardType.SQUARE8: "Square8",
        BoardType.SQUARE19: "Square19",
        BoardType.HEXAGONAL: "Hex",
    }
    parts = []
    for board, value in per_board.items():
        label = label_by_board.get(board, str(board))
        parts.append(f"{label}: {value:.3f}")
    return "{ " + ", ".join(parts) + " }"


def _run_profile(
    label: str,
    candidate_weights: Dict[str, float],
    baseline_weights: Dict[str, float],
    boards: List[BoardType],
    *,
    games_per_eval: int,
    max_moves: int,
    eval_mode: str,
    state_pool_id: str,
    eval_randomness: float,
    seed: int | None,
    per_board_mode: bool,
    verbose: bool,
) -> Tuple[float, Dict[BoardType, float]]:
    """Run evaluation for a single profile.

    Optionally emit per-board diagnostics and timings.
    """
    if not boards:
        raise SystemExit("No boards selected for evaluation")

    print(
        f"[{label}] games_per_eval={games_per_eval}, "
        f"max_moves={max_moves}, eval_mode={eval_mode}, "
        f"state_pool_id={state_pool_id!r}, seed={seed}",
        flush=True,
    )

    if per_board_mode:
        per_board_fitness: Dict[BoardType, float] = {}
        start_all = time.perf_counter()
        for idx, board in enumerate(boards):
            print(
                f"[{label}]   Board {idx + 1}/{len(boards)}: {board.name}",
                flush=True,
            )
            board_seed = None if seed is None else seed + idx * 10_000
            start = time.perf_counter()
            fitness = evaluate_fitness(
                candidate_weights=candidate_weights,
                baseline_weights=baseline_weights,
                games_per_eval=games_per_eval,
                board_type=board,
                verbose=verbose,
                opponent_mode="baseline-only",
                max_moves=max_moves,
                eval_mode=eval_mode,
                state_pool_id=state_pool_id,
                eval_randomness=eval_randomness,
                seed=board_seed,
            )
            elapsed = time.perf_counter() - start
            per_board_fitness[board] = float(fitness)
            print(
                f"[{label}]   -> {board.name} fitness={fitness:.3f} "
                f"(elapsed={elapsed:.1f}s)",
                flush=True,
            )

        total_elapsed = time.perf_counter() - start_all
        aggregate = float(
            sum(per_board_fitness.values()) / float(len(per_board_fitness))
        )
        print(
            f"[{label}] aggregate over {len(boards)} boards = {aggregate:.3f} "
            f"(total elapsed={total_elapsed:.1f}s)",
            flush=True,
        )
        return aggregate, per_board_fitness

    # Single call over all boards using the shared helper.
    print(
        f"[{label}] Evaluating over {len(boards)} boards via "
        "evaluate_fitness_over_boards(...)",
        flush=True,
    )
    start = time.perf_counter()
    aggregate, per_board_map = evaluate_fitness_over_boards(
        candidate_weights=candidate_weights,
        baseline_weights=baseline_weights,
        games_per_eval=games_per_eval,
        boards=boards,
        opponent_mode="baseline-only",
        max_moves=max_moves,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        eval_randomness=eval_randomness,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    print(
        f"[{label}] aggregate={aggregate:.3f} over {len(boards)} boards "
        f"(elapsed={elapsed:.1f}s)",
        flush=True,
    )
    return aggregate, per_board_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check multi-board, multi-start heuristic evaluation."
        ),
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=8,
        help=(
            "Number of games per board for each evaluation "
            "(default: 8). Smaller values run faster but are noisier."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for evaluation (threaded into per-board seeds).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help=(
            "Override max_moves for evaluation "
            "(default comes from DEFAULT_TRAINING_EVAL_CONFIG)."
        ),
    )
    parser.add_argument(
        "--boards",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of boards to evaluate, using "
            "names: square8,square19,hex. Defaults to all boards from "
            "DEFAULT_TRAINING_EVAL_CONFIG."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["initial-only", "multi-start"],
        default=None,
        help=(
            "Override eval_mode for evaluation. Defaults to the value in "
            "DEFAULT_TRAINING_EVAL_CONFIG."
        ),
    )
    parser.add_argument(
        "--per-board",
        action="store_true",
        help=(
            "Evaluate each board sequentially and print per-board timings "
            "using evaluate_fitness(...) instead of the aggregate helper."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Pass verbose=True into the evaluation harness to print W/D/L "
            "and avg_moves diagnostics per board."
        ),
    )
    args = parser.parse_args()

    # Define profiles
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)
    zero = {key: 0.0 for key in HEURISTIC_WEIGHT_KEYS}

    # Build canonical multi-board, multi-start evaluation kwargs.
    eval_cfg = build_training_eval_kwargs(
        games_per_eval=args.games_per_eval,
        eval_randomness=0.0,
        seed=args.seed,
    )

    boards: List[BoardType] = list(eval_cfg["boards"])
    if args.boards:
        name_to_board = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex": BoardType.HEXAGONAL,
        }
        requested: List[BoardType] = []
        for raw in args.boards.split(","):
            name = raw.strip().lower()
            if not name:
                continue
            if name not in name_to_board:
                raise SystemExit(
                    f"Unknown board name {raw!r} in --boards "
                    "(expected square8,square19,hex)"
                )
            requested.append(name_to_board[name])
        if not requested:
            raise SystemExit("No valid boards selected via --boards")
        # Preserve order from default config but filter to requested
        boards = [b for b in boards if b in requested]
        if not boards:
            raise SystemExit(
                "No overlap between --boards selection and "
                "DEFAULT_TRAINING_EVAL_CONFIG boards"
            )

    eval_mode = args.eval_mode or eval_cfg["eval_mode"]
    if args.max_moves is not None:
        max_moves = args.max_moves
    else:
        max_moves = eval_cfg["max_moves"]
    state_pool_id = eval_cfg["state_pool_id"]
    eval_randomness = eval_cfg["eval_randomness"]
    seed = eval_cfg["seed"]

    print("Multi-board, multi-start heuristic evaluation sanity check")
    print("----------------------------------------------------------")
    print(
        f"Config: games_per_eval={args.games_per_eval}, "
        f"eval_mode={eval_mode}, state_pool_id={state_pool_id!r}, "
        f"eval_randomness={eval_randomness}, seed={seed}, "
        f"max_moves={max_moves}",
        flush=True,
    )
    print(f"Boards: {[b.name for b in boards]}")
    print(f"Per-board mode: {args.per_board}, verbose: {args.verbose}")
    print()

    # Baseline vs baseline
    agg_bb, per_board_bb = _run_profile(
        "Baseline vs baseline",
        candidate_weights=baseline,
        baseline_weights=baseline,
        boards=boards,
        games_per_eval=args.games_per_eval,
        max_moves=max_moves,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        eval_randomness=eval_randomness,
        seed=seed,
        per_board_mode=args.per_board,
        verbose=args.verbose,
    )
    print()

    # Zero vs baseline
    agg_zb, per_board_zb = _run_profile(
        "Zero vs baseline",
        candidate_weights=zero,
        baseline_weights=baseline,
        boards=boards,
        games_per_eval=args.games_per_eval,
        max_moves=max_moves,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        eval_randomness=eval_randomness,
        seed=seed,
        per_board_mode=args.per_board,
        verbose=args.verbose,
    )
    print()

    # Final compact summary
    print("Summary")
    print("=======")
    print("Baseline vs baseline:")
    print(f"  aggregate fitness: {agg_bb:.3f}")
    print(f"  per-board: {_format_per_board(per_board_bb)}")
    print()
    print("Zero vs baseline:")
    print(f"  aggregate fitness: {agg_zb:.3f}")
    print(f"  per-board: {_format_per_board(per_board_zb)}")


if __name__ == "__main__":
    main()