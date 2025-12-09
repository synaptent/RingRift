#!/usr/bin/env python
"""Cross-board candidate-vs-baseline parity gate for AI models.

This script is a lightweight wrapper around ``evaluate_ai_models.py`` that:

- Runs a small evaluation matrix of candidate vs baseline AIs across boards.
- Aggregates win-rate + Wilson CI metrics per board.
- Emits a compact JSON report with a single ``overall_pass`` flag.

It is intended to be used as an *additional* safety check in promotion
pipelines: before a candidate is considered for tier-specific gating
(``run_tier_gate.py``) and perf budgets, we can sanity-check that it is not
clearly worse than the current baseline on the boards we care about.

The implementation focuses on core decision logic and JSON shape so that unit
tests can exercise the gate without running expensive tournaments. The CLI
path can be used in ad-hoc workflows and CI once wired into the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the evaluation harness and board helpers.
from scripts.evaluate_ai_models import (  # type: ignore[import]  # noqa: E402
    BOARD_TYPE_MAP,
    EvaluationResults,
    SUPPORTED_AI_TYPES,
    format_results_json,
    run_evaluation,
)


@dataclass(frozen=True)
class MatrixSpec:
    """Single evaluation matrix entry (board + game count).

    For now we only support 2-player games, consistent with the current
    ``run_evaluation`` helper and the square-8 2-player tier pipeline.
    """

    name: str
    board_key: str
    games: int
    max_moves: int


def _default_matrix(games: int, max_moves: int) -> List[MatrixSpec]:
    """Build the default evaluation matrix for gating.

    The default keeps the focus on the square8 2-player setting used by the
    tiered ladder, while remaining extensible if we later want to add
    square19 / hexagonal entries.
    """
    return [
        MatrixSpec(
            name="square8_2p",
            board_key="square8",
            games=games,
            max_moves=max_moves,
        ),
    ]


def _run_matrix(
    spec: MatrixSpec,
    *,
    player1_type: str,
    player2_type: str,
    seed: int | None,
    checkpoint: str | None,
    checkpoint2: str | None,
    cmaes_weights: str | None,
    minimax_depth: int,
) -> Dict[str, Any]:
    """Run a single candidate-vs-baseline evaluation matrix."""
    board_type = BOARD_TYPE_MAP[spec.board_key]

    results: EvaluationResults = run_evaluation(
        player1_type=player1_type,
        player2_type=player2_type,
        num_games=spec.games,
        board_type=board_type,
        seed=seed,
        checkpoint_path=checkpoint,
        checkpoint_path2=checkpoint2,
        cmaes_weights_path=cmaes_weights,
        minimax_depth=minimax_depth,
        max_moves_per_game=spec.max_moves,
        verbose=False,
    )

    return format_results_json(results)


def _evaluate_promotion(
    formatted_by_matrix: Mapping[str, Mapping[str, Any]],
    *,
    min_ci_lower_bound: float = 0.5,
) -> Dict[str, Any]:
    """Compute promotion decision from formatted evaluation results.

    Args:
        formatted_by_matrix: Mapping from matrix name to the dictionary
            returned by ``format_results_json`` for that matrix.
        min_ci_lower_bound: Minimum acceptable lower bound of the 95% Wilson
            confidence interval for the candidate win rate. Values below this
            threshold are treated as evidence that the candidate is weaker
            than baseline for that matrix.

    Returns:
        A dictionary with:
          - overall_pass: True iff all matrices pass.
          - thresholds: {min_ci_lower_bound}.
          - worst_case_ci_lower_bound: smallest lower CI bound observed.
          - matrices: per-matrix summaries with:
              - board
              - games
              - player1_win_rate
              - player1_win_rate_ci95
              - piece_advantage_p1
              - passes
    """
    matrices_summary: Dict[str, Any] = {}
    overall_pass = True
    worst_ci_lower = 1.0

    for name, payload in formatted_by_matrix.items():
        cfg = payload.get("config", {})
        res = payload.get("results", {})

        win_rate = float(res.get("player1_win_rate", 0.0))
        ci = res.get("player1_win_rate_ci95", [0.0, 0.0])
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            ci = [0.0, 0.0]
        ci_lo = float(ci[0])
        ci_hi = float(ci[1])
        piece_adv = float(res.get("piece_advantage_p1", 0.0))

        # Simple non-inferiority gate: candidate must have CI lower bound
        # above the configured threshold for this matrix.
        passes = ci_lo >= min_ci_lower_bound
        overall_pass = overall_pass and passes
        worst_ci_lower = min(worst_ci_lower, ci_lo)

        matrices_summary[name] = {
            "board": cfg.get("board", name),
            "games": cfg.get("games"),
            "player1_win_rate": win_rate,
            "player1_win_rate_ci95": [ci_lo, ci_hi],
            "piece_advantage_p1": piece_adv,
            "passes": passes,
        }

    # If we had no matrices at all, be conservative and fail the gate.
    if not matrices_summary:
        overall_pass = False
        worst_ci_lower = 0.0

    return {
        "overall_pass": overall_pass,
        "thresholds": {
            "min_ci_lower_bound": min_ci_lower_bound,
        },
        "worst_case_ci_lower_bound": worst_ci_lower,
        "matrices": matrices_summary,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the parity promotion gate."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a small candidate-vs-baseline evaluation matrix across one "
            "or more boards and emit a promotion gate JSON summary."
        ),
    )

    parser.add_argument(
        "--player1",
        type=str,
        choices=SUPPORTED_AI_TYPES,
        required=True,
        help="AI type for the candidate (Player 1 in reporting).",
    )
    parser.add_argument(
        "--player2",
        type=str,
        choices=SUPPORTED_AI_TYPES,
        required=True,
        help="AI type for the baseline (Player 2 in reporting).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for the candidate (player1).",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="Optional checkpoint path for the baseline (player2).",
    )
    parser.add_argument(
        "--cmaes-weights",
        type=str,
        default=None,
        help="Optional CMA-ES weights path (for cmaes_heuristic).",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=3,
        help="Minimax search depth when using the minimax AI type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed (default: 1).",
    )
    parser.add_argument(
        "--games-per-matrix",
        type=int,
        default=200,
        help="Number of games to play for each board matrix (default: 200).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game before declaring a draw (default: 200).",
    )
    parser.add_argument(
        "--min-ci-lower-bound",
        type=float,
        default=0.5,
        help=(
            "Minimum acceptable lower bound of the 95%% CI for the candidate " "win rate on each matrix (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--boards",
        type=str,
        nargs="*",
        choices=list(BOARD_TYPE_MAP.keys()),
        default=["square8"],
        help=("Boards to include in the evaluation matrix. Defaults to " "['square8']."),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the aggregated JSON report.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    # Build matrix specs from requested boards.
    matrix_specs: List[MatrixSpec] = []
    for board_key in args.boards:
        if board_key not in BOARD_TYPE_MAP:
            raise SystemExit(f"Unsupported board: {board_key}")
        matrix_specs.append(
            MatrixSpec(
                name=f"{board_key}_2p",
                board_key=board_key,
                games=args.games_per_matrix,
                max_moves=args.max_moves,
            ),
        )

    # Fall back to the default matrix if none were provided.
    if not matrix_specs:
        matrix_specs = _default_matrix(
            games=args.games_per_matrix,
            max_moves=args.max_moves,
        )

    formatted_by_matrix: MutableMapping[str, Dict[str, Any]] = {}
    for spec in matrix_specs:
        payload = _run_matrix(
            spec,
            player1_type=args.player1,
            player2_type=args.player2,
            seed=args.seed,
            checkpoint=args.checkpoint,
            checkpoint2=args.checkpoint2,
            cmaes_weights=args.cmaes_weights,
            minimax_depth=args.minimax_depth,
        )
        formatted_by_matrix[spec.name] = payload

    gate_summary = _evaluate_promotion(
        formatted_by_matrix,
        min_ci_lower_bound=args.min_ci_lower_bound,
    )

    report: Dict[str, Any] = {
        "config": {
            "player1": args.player1,
            "player2": args.player2,
            "checkpoint": args.checkpoint,
            "checkpoint2": args.checkpoint2,
            "boards": args.boards,
            "games_per_matrix": args.games_per_matrix,
            "max_moves": args.max_moves,
            "seed": args.seed,
        },
        "gate": gate_summary,
    }

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Wrote parity promotion gate report to {output_path}")
    else:
        # Print a concise human summary to stdout.
        print("Parity promotion gate summary:")
        print(json.dumps(report["gate"], indent=2, sort_keys=True))

    return 0 if gate_summary.get("overall_pass") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
