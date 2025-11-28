#!/usr/bin/env python
"""Focused diagnostics for the CMA-ES / GA 0.5 plateau on Square8.

This script reuses the existing evaluate_fitness(...) harness to run a small
set of controlled probes:

- baseline vs baseline
- zero weights vs baseline
- scaled baseline (5x) vs baseline
- several near-baseline random perturbations vs baseline

It prints a compact table of fitness and W/D/L patterns for each profile
under the classic initial-only evaluation regime, and optionally under a
multi-start regime using a Square8 state pool. It also writes the probed
weight profiles to logs/plateau_probe/ so that
scripts/diagnose_policy_equivalence.py can compare action-level policies.
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
    evaluate_fitness,
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


def _run_single_probe(
    label: str,
    candidate: HeuristicWeights,
    baseline: HeuristicWeights,
    *,
    games_per_eval: int,
    eval_mode: str,
    state_pool_id: str | None,
    seed: int,
    max_moves: int = 200,
) -> ProbeResult:
    """Evaluate one candidate profile against the baseline on Square8."""

    stats: Dict[str, Any] = {}

    def _hook(h: Dict[str, Any]) -> None:
        stats.update(h)

    fitness = evaluate_fitness(
        candidate_weights=candidate,
        baseline_weights=baseline,
        games_per_eval=games_per_eval,
        board_type=BoardType.SQUARE8,
        verbose=True,
        opponent_mode="baseline-only",
        max_moves=max_moves,
        debug_hook=_hook,
        eval_mode=eval_mode,
        state_pool_id=state_pool_id,
        eval_randomness=0.0,
        seed=seed,
    )

    if not stats:
        # Fallback in case the debug hook is not invoked for some reason.
        stats = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "games_per_eval": games_per_eval,
            "weight_l2": _compute_l2(candidate, baseline),
        }

    result: ProbeResult = {
        "profile": label,
        "eval_mode": eval_mode,
        "games_per_eval": games_per_eval,
        "fitness": float(fitness),
        "wins": int(stats["wins"]),
        "draws": int(stats["draws"]),
        "losses": int(stats["losses"]),
        "weight_l2": float(stats["weight_l2"]),
    }
    return result


def _format_row(row: ProbeResult, comment: str) -> str:
    """Return a formatted table row string."""

    w = row["wins"]
    d = row["draws"]
    losses = row["losses"]
    return (
        f"{row['profile']:<10} "
        f"{row['weight_l2']:7.3f} "
        f"{row['fitness']:7.3f} "
        f"{w:02d}-{d:02d}-{losses:02d} "
        f"{comment}"
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
        description=(
            "Probe CMA-ES/GA 0.5 plateau behaviour on Square8 "
            "using evaluate_fitness."
        ),
    )
    parser.add_argument(
        "--games-initial",
        type=int,
        default=8,
        help="Games per evaluation for initial-only probes (default: 8).",
    )
    parser.add_argument(
        "--games-multistart",
        type=int,
        default=16,
        help=(
            "Games per evaluation for multi-start probes when enabled "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base RNG seed forwarded to evaluate_fitness (default: 12345).",
    )
    parser.add_argument(
        "--near-count",
        type=int,
        default=5,
        help=(
            "Number of near-baseline random candidates to generate "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--run-multistart",
        action="store_true",
        help=(
            "Also run probes under eval_mode=multi-start using the Square8 "
            "state pool (state_pool_id='v1')."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="logs/plateau_probe",
        help=(
            "Directory to write probed weight profiles as JSON "
            "(default: logs/plateau_probe)."
        ),
    )
    args = parser.parse_args(argv)

    baseline: HeuristicWeights = dict(BASE_V1_BALANCED_WEIGHTS)

    zero: HeuristicWeights = {k: 0.0 for k in HEURISTIC_WEIGHT_KEYS}
    scaled5: HeuristicWeights = {
        k: float(baseline[k] * 5.0) for k in HEURISTIC_WEIGHT_KEYS
    }

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
    print("=== Initial-only evaluation (Square8, baseline-only) ===")
    print("Profile     l2       fitness  W-D-L  comment")

    initial_results: List[ProbeResult] = []
    # Baseline vs baseline
    initial_results.append(
        _run_single_probe(
            "baseline",
            baseline,
            baseline,
            games_per_eval=args.games_initial,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=args.seed,
        ),
    )
    # Zero weights vs baseline
    initial_results.append(
        _run_single_probe(
            "zero",
            zero,
            baseline,
            games_per_eval=args.games_initial,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=args.seed,
        ),
    )
    # Scaled baseline vs baseline
    initial_results.append(
        _run_single_probe(
            "scaled5",
            scaled5,
            baseline,
            games_per_eval=args.games_initial,
            eval_mode="initial-only",
            state_pool_id=None,
            seed=args.seed,
        ),
    )
    # Near-baseline random candidates vs baseline
    for name, weights in near_profiles:
        initial_results.append(
            _run_single_probe(
                name,
                weights,
                baseline,
                games_per_eval=args.games_initial,
                eval_mode="initial-only",
                state_pool_id=None,
                seed=args.seed,
            ),
        )

    comments: Dict[str, str] = {
        "baseline": "baseline vs baseline",
        "zero": "zero vs baseline",
        "scaled5": "5x baseline vs baseline",
    }
    for i in range(args.near_count):
        comments[f"near_{i}"] = f"near-baseline #{i} vs baseline"

    for row in initial_results:
        comment = comments.get(row["profile"], "")
        print(_format_row(row, comment))

    if args.run_multistart:
        print()
        print(
            "=== Multi-start evaluation (Square8, state_pool_id='v1') ===",
        )
        print("Profile     l2       fitness  W-D-L  comment")
        multi_results: List[ProbeResult] = []
        for name, weights in [
            ("baseline", baseline),
            ("zero", zero),
            ("scaled5", scaled5),
        ]:
            multi_results.append(
                _run_single_probe(
                    name,
                    weights,
                    baseline,
                    games_per_eval=args.games_multistart,
                    eval_mode="multi-start",
                    state_pool_id="v1",
                    seed=args.seed,
                ),
            )
        for name, weights in near_profiles:
            multi_results.append(
                _run_single_probe(
                    name,
                    weights,
                    baseline,
                    games_per_eval=args.games_multistart,
                    eval_mode="multi-start",
                    state_pool_id="v1",
                    seed=args.seed,
                ),
            )
        for row in multi_results:
            comment = comments.get(row["profile"], "") + " (multi-start)"
            print(_format_row(row, comment))


if __name__ == "__main__":  # pragma: no cover
    main()