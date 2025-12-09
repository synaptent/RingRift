#!/usr/bin/env python
"""Multi-board plateau diagnostics for heuristic CMA-ES / GA.

This script reuses the shared training evaluation harness to probe a small
set of heuristic weight profiles against the balanced baseline under a
configuration that matches the canonical 2-player training preset:

- multi-board evaluation on Square8, Square19, and Hexagonal boards;
- eval_mode="multi-start" from the "v1" mid/late-game state pools; and
- a small, non-zero eval_randomness to break ties without drowning signal.

For each candidate profile (baseline, zero, 5x scaled baseline, and a
handful of near-baseline perturbations) the script prints a compact
per-board summary:

    candidate=zero board=square19 l2=33.8 fitness=0.812 W=13 D=0 L=3

It also writes the probed weight profiles to ``logs/plateau_probe/`` so
that :mod:`scripts.diagnose_policy_equivalence` can compare action-level
policies over the same state pools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

# Ensure app.* imports resolve when run from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import BoardType  # type: ignore  # noqa: E402
from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
    HEURISTIC_WEIGHT_KEYS,
    HeuristicWeights,
)
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    BOARD_NAME_TO_TYPE,
    evaluate_fitness_over_boards,
)


ProbeResult = Dict[str, Any]


def _compute_l2(
    candidate: HeuristicWeights,
    baseline: HeuristicWeights,
) -> float:
    """Compute L2 distance using the canonical HEURISTIC_WEIGHT_KEYS order."""

    try:
        cand_vec = np.array(
            [float(candidate[k]) for k in HEURISTIC_WEIGHT_KEYS],
            dtype=float,
        )
        base_vec = np.array(
            [float(baseline[k]) for k in HEURISTIC_WEIGHT_KEYS],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return 0.0
    return float(np.linalg.norm(cand_vec - base_vec))


def _parse_board_list(raw: str) -> List[BoardType]:
    """Parse a comma-separated list of board names into BoardType values."""
    names = [name.strip().lower() for name in raw.split(",") if name.strip()]
    if not names:
        raise SystemExit("At least one board must be specified via --boards")

    boards: List[BoardType] = []
    for name in names:
        try:
            boards.append(BOARD_NAME_TO_TYPE[name])
        except KeyError:
            raise SystemExit(
                f"Unknown board name in --boards: {name!r} " "(expected one of: square8,square19,hex)",
            )
    return boards


def _run_probes_for_candidate(
    label: str,
    candidate: HeuristicWeights,
    baseline: HeuristicWeights,
    *,
    boards: List[BoardType],
    games_per_eval: int,
    eval_mode: str,
    state_pool_id: str,
    eval_randomness: float,
    seed: int,
    max_moves: int = 200,
) -> List[ProbeResult]:
    """Evaluate one candidate profile against the baseline on all boards.

    This function routes all evaluation through
    :func:`evaluate_fitness_over_boards` so that it stays aligned with the
    multi-board, multi-start configuration used by the CMA-ES training
    harness.
    """

    per_board_stats: Dict[BoardType, Dict[str, Any]] = {}

    def _debug_callback(
        _candidate_w: HeuristicWeights,
        _baseline_w: HeuristicWeights,
        board_type: BoardType,
        stats: Dict[str, Any],
    ) -> None:
        # One callback invocation per candidate/board; record the latest
        # stats for this board.
        per_board_stats[board_type] = dict(stats)

    # Run the shared evaluation helper. We primarily rely on the per-board
    # debug callback payloads for W/D/L and weight_l2 diagnostics.
    _, per_board_fitness = evaluate_fitness_over_boards(
        candidate_weights=candidate,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        boards=boards,
        opponent_mode="baseline-only",
        max_moves=max_moves,
        verbose=False,
        debug_hook=None,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        seed=seed,
        eval_randomness=eval_randomness,
        debug_callback=_debug_callback,
        progress_label=f"plateau-probe | profile={label}",
    )

    weight_l2_default = _compute_l2(candidate, baseline)
    rows: List[ProbeResult] = []

    for board in boards:
        stats = per_board_stats.get(board, {})
        fitness = float(
            stats.get("fitness", per_board_fitness.get(board, 0.0)),
        )
        wins = int(stats.get("wins", 0))
        draws = int(stats.get("draws", 0))
        losses = int(stats.get("losses", 0))
        weight_l2 = float(
            stats.get(
                "weight_l2_to_baseline",
                stats.get("weight_l2", weight_l2_default),
            ),
        )

        row: ProbeResult = {
            "profile": label,
            "board": board,
            "board_name": str(getattr(board, "value", board)),
            "eval_mode": eval_mode,
            "games_per_eval": games_per_eval,
            "state_pool_id": state_pool_id,
            "eval_randomness": eval_randomness,
            "fitness": fitness,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "weight_l2": weight_l2,
        }
        rows.append(row)

    return rows


def _format_row(row: ProbeResult) -> str:
    """Return a formatted, per-board table row string."""

    w = row["wins"]
    d = row["draws"]
    losses = row["losses"]
    return (
        f"candidate={row['profile']} "
        f"board={row['board_name']} "
        f"l2={row['weight_l2']:5.3f} "
        f"fitness={row['fitness']:5.3f} "
        f"W={w} D={d} L={losses}"
    )


def _export_weights(
    out_dir: str,
    profiles: Mapping[str, HeuristicWeights],
) -> List[str]:
    """Write weight JSON files for each profile and return their paths."""

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for name, weights in profiles.items():
        payload = {
            "weights": weights,
            "meta": {
                "id": name,
                "source": "probe_plateau_diagnostics",
            },
        }
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        written.append(path)
    return written


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Probe heuristic CMA-ES/GA plateau behaviour using the shared " "multi-board evaluation harness."),
    )
    parser.add_argument(
        "--boards",
        type=str,
        default="square8,square19,hex",
        help=("Comma-separated list of boards to probe " "(default: 'square8,square19,hex')."),
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=16,
        help=("Games per candidate evaluation on each board " "(default: 16, matching DEFAULT_TRAINING_EVAL_CONFIG)."),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["initial-only", "multi-start"],
        default="multi-start",
        help=(
            "Evaluation mode: 'initial-only' uses the empty starting "
            "position; 'multi-start' samples positions from a state pool. "
            "Default: 'multi-start'."
        ),
    )
    parser.add_argument(
        "--state-pool-id",
        type=str,
        default="v1",
        help=("Identifier for the evaluation state pool when using " "eval-mode=multi-start (default: 'v1')."),
    )
    parser.add_argument(
        "--eval-randomness",
        type=float,
        default=0.02,
        help=(
            "Randomness parameter forwarded to HeuristicAI during "
            "evaluation. Default 0.02 matches the 2p training preset; "
            "use 0.0 for fully deterministic evaluation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=("Base RNG seed forwarded to the evaluation harness " "(default: 12345)."),
    )
    parser.add_argument(
        "--near-count",
        type=int,
        default=5,
        help=("Number of near-baseline random candidates to generate " "(default: 5)."),
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help=("Maximum moves per self-play game before declaring a draw " "(default: 200)."),
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="logs/plateau_probe",
        help=("Directory to write probed weight profiles as JSON " "(default: logs/plateau_probe)."),
    )
    args = parser.parse_args(argv)

    boards = _parse_board_list(args.boards)
    baseline: HeuristicWeights = dict(BASE_V1_BALANCED_WEIGHTS)

    zero: HeuristicWeights = {k: 0.0 for k in HEURISTIC_WEIGHT_KEYS}
    scaled5: HeuristicWeights = {k: float(baseline[k] * 5.0) for k in HEURISTIC_WEIGHT_KEYS}

    rng = np.random.default_rng(args.seed)
    near_profiles: List[Tuple[str, HeuristicWeights]] = []
    for i in range(args.near_count):
        deltas = rng.normal(
            loc=0.0,
            scale=1.0,
            size=len(HEURISTIC_WEIGHT_KEYS),
        )
        weights: HeuristicWeights = {}
        for idx, key in enumerate(HEURISTIC_WEIGHT_KEYS):
            weights[key] = float(baseline[key] + deltas[idx])
        near_profiles.append((f"near_{i}", weights))

    # Export weight profiles for downstream policy-equivalence diagnostics.
    profiles_to_export: Dict[str, HeuristicWeights] = {"baseline": baseline}
    profiles_to_export["zero"] = zero
    profiles_to_export["scaled5"] = scaled5
    for name, weights in near_profiles:
        profiles_to_export[name] = weights

    written_paths = _export_weights(args.export_dir, profiles_to_export)
    print("Exported weight profiles:")
    for path in written_paths:
        print(f"  {path}")

    print()
    print("=== Multi-board heuristic plateau probes ===")
    boards_label = ",".join(getattr(b, "value", str(b)) for b in boards)
    print(
        "candidate board       l2    fitness  W  D  L  "
        f"(boards={boards_label}, "
        f"mode={args.eval_mode}, "
        f"games={args.games_per_eval}, "
        f"state_pool_id={args.state_pool_id}, "
        f"eval_randomness={args.eval_randomness})",
    )

    # Order of candidates: baseline, zero, scaled5, then near-baseline set.
    candidates: List[Tuple[str, HeuristicWeights]] = [
        ("baseline", baseline),
        ("zero", zero),
        ("scaled5", scaled5),
        *near_profiles,
    ]

    for name, weights in candidates:
        rows = _run_probes_for_candidate(
            label=name,
            candidate=weights,
            baseline=baseline,
            boards=boards,
            games_per_eval=args.games_per_eval,
            eval_mode=args.eval_mode,
            state_pool_id=args.state_pool_id,
            eval_randomness=args.eval_randomness,
            seed=args.seed,
            max_moves=args.max_moves,
        )
        for row in rows:
            print(_format_row(row))


if __name__ == "__main__":  # pragma: no cover
    main()
