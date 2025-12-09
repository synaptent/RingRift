#!/usr/bin/env python
"""Tier performance benchmark CLI for RingRift difficulty tiers.

This script runs a small per-move latency benchmark for a given
difficulty tier (for example D4/D6/D8 on square8 2p) using the canonical
training environment and ladder configuration.

It is intended for local tuning and manual verification of perf budgets,
not as a CI gate. CI-facing assertions live in ai-service/tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure app/ is importable when running as a script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.perf_budgets import (  # noqa: E402
    TIER_PERF_BUDGETS,
    TierPerfBudget,
    get_tier_perf_budget,
)
from app.training.tier_perf_benchmark import (  # noqa: E402
    TierPerfResult,
    run_tier_perf_benchmark,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for tier perf benchmarking."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a small per-move latency benchmark for a difficulty tier " "(for example D4/D6/D8 on square8 2p)."
        ),
    )

    parser.add_argument(
        "--tier",
        required=True,
        help=("Tier identifier (for example D4, D6, D8 or " "D4_SQ8_2P / D6_SQ8_2P / D8_SQ8_2P)."),
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=4,
        help=("Number of self-play games to run (default: 4). " "Each game samples up to --moves-per-game moves."),
    )

    parser.add_argument(
        "--moves-per-game",
        type=int,
        default=16,
        help=(
            "Maximum number of moves to sample per game (default: 16). "
            "Only the first N moves are used for latency measurement."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for reproducible benchmarks (default: 1).",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=("Optional path to write a JSON summary containing raw metrics " "and PASS/FAIL flags."),
    )

    return parser.parse_args(argv)


def _eval_budget(result: TierPerfResult) -> Dict[str, Any]:
    """Evaluate whether the benchmark result is within configured budgets."""
    within_avg = result.average_ms <= result.budget.max_avg_move_ms
    within_p95 = result.p95_ms <= result.budget.max_p95_move_ms
    overall_pass = within_avg and within_p95
    return {
        "within_avg": within_avg,
        "within_p95": within_p95,
        "overall_pass": overall_pass,
    }


def _print_human_summary(
    result: TierPerfResult,
    budget_eval: Dict[str, Any],
) -> None:
    """Print a concise human-readable summary of the benchmark."""
    budget: TierPerfBudget = result.budget

    print()
    print("=" * 72)
    print("TIER PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Tier:          {budget.tier_name}")
    print(f"Difficulty:    D{budget.difficulty}")
    board_str = f"{budget.board_type.value}, {budget.num_players} players"
    print(f"Board:         {board_str}")
    print(f"Samples:       {result.num_samples} moves")
    print()
    print("Observed latency (ms):")
    print(f"  average:     {result.average_ms:.1f} ms")
    print(f"  p95:         {result.p95_ms:.1f} ms")
    print()
    print("Budget (ms):")
    print(f"  max average: {budget.max_avg_move_ms:.1f} ms")
    print(f"  max p95:     {budget.max_p95_move_ms:.1f} ms")
    print()
    print("Evaluation:")
    avg_status = "PASS" if budget_eval["within_avg"] else "FAIL"
    p95_status = "PASS" if budget_eval["within_p95"] else "FAIL"
    overall = "PASS" if budget_eval["overall_pass"] else "FAIL"
    print(f"  average:     {avg_status}")
    print(f"  p95:         {p95_status}")
    print()
    print(f"Overall:       {overall}")
    print("=" * 72)
    print()

    if budget.notes:
        print("Notes:")
        print(budget.notes)
        print()


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the tier perf benchmark CLI."""
    args = parse_args(argv)

    # Normalise tier name to the keys used by TIER_PERF_BUDGETS.
    tier_key = args.tier.upper()
    if tier_key not in TIER_PERF_BUDGETS:
        # Allow bare difficulty names by mapping D4 -> D4_SQ8_2P etc when
        # such an entry exists.
        if tier_key.startswith("D") and tier_key[1:].isdigit():
            # Best-effort: try to resolve via get_tier_perf_budget to
            # surface a clearer error if still unknown.
            try:
                _ = get_tier_perf_budget(tier_key)
            except KeyError as exc:
                available = ", ".join(sorted(TIER_PERF_BUDGETS.keys()))
                msg = f"Unknown tier '{args.tier}'. " f"Available perf-budgeted tiers: {available}"
                raise SystemExit(msg) from exc
        else:
            available = ", ".join(sorted(TIER_PERF_BUDGETS.keys()))
            msg = f"Unknown tier '{args.tier}'. " f"Available perf-budgeted tiers: {available}"
            raise SystemExit(msg)

    result: TierPerfResult = run_tier_perf_benchmark(
        tier_name=tier_key,
        num_games=args.num_games,
        moves_per_game=args.moves_per_game,
        seed=args.seed,
    )
    budget_eval = _eval_budget(result)
    _print_human_summary(result, budget_eval)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload: Dict[str, Any] = {
            "tier_name": result.tier_name,
            "difficulty": result.budget.difficulty,
            "board_type": result.budget.board_type.value,
            "num_players": result.budget.num_players,
            "metrics": {
                "average_ms": result.average_ms,
                "p95_ms": result.p95_ms,
            },
            "budget": {
                "max_avg_move_ms": result.budget.max_avg_move_ms,
                "max_p95_move_ms": result.budget.max_p95_move_ms,
            },
            "evaluation": budget_eval,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"JSON summary written to: {output_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
