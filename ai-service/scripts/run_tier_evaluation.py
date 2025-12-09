#!/usr/bin/env python
"""Tier-aware difficulty evaluation CLI for RingRift.

This script evaluates a candidate AI configuration against the
tier-specific opponents and thresholds defined in
``app.training.tier_eval_config`` using the canonical RingRiftEnv
from ``app.training.env``.

It is intended for both local experimentation and CI gating.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Ensure app/ is importable when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.training.tier_eval_config import (  # noqa: E402
    TIER_EVAL_CONFIGS,
    get_tier_config,
)
from app.training.tier_eval_runner import (  # noqa: E402
    run_tier_evaluation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for tier evaluation."""
    parser = argparse.ArgumentParser(
        description=("Evaluate a candidate AI configuration for a difficulty tier."),
    )

    parser.add_argument(
        "--tier",
        required=True,
        help="Tier identifier (e.g. D2, D4, D6, D8).",
    )

    parser.add_argument(
        "--candidate-config",
        required=True,
        help=(
            "Identifier for the candidate configuration under test "
            "(for example a heuristic profile id, model path, or "
            "short name)."
        ),
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help=("Override the number of games per opponent. Defaults to " "the tier profile value."),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for reproducible evaluations.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary file.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=("Optional profile variant name. Reserved for future " "multi-profile tier configs."),
    )

    return parser.parse_args()


def _print_human_summary(result: Any) -> None:
    """Print a concise human-readable summary of the evaluation."""
    print()
    print("=" * 72)
    print("TIER EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Tier:      {result.tier_name}")
    print(f"Candidate: {result.candidate_id}")
    board_str = f"{result.board_type.value}, {result.num_players} players"
    print(f"Board:     {board_str}")
    print(f"Games:     {result.total_games}")
    print()

    print("Matchups:")
    for m in result.matchups:
        win_rate_pct = m.win_rate * 100.0
        header = f"  vs {m.opponent_id} " f"(difficulty {m.opponent_difficulty}, " f"ai={m.opponent_ai_type})"
        print(header)
        line = f"    W/D/L: {m.wins} / {m.draws} / {m.losses} " f"(win-rate: {win_rate_pct:.1f}%)"
        print(line)

    print()
    print("Gates:")
    for name, passed in result.criteria.items():
        if passed is None:
            status = "N/A"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"

        metric_key = name.replace("pass_", "")
        metric_val = result.metrics.get(metric_key, None)
        if isinstance(metric_val, float):
            suffix = f" (observed={metric_val:.3f})"
        else:
            suffix = ""
        print(f"  {name}: {status}{suffix}")

    print()
    overall = "PASS" if result.overall_pass else "FAIL"
    print(f"Overall: {overall}")
    print("=" * 72)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    tier_name = args.tier.upper()
    if tier_name not in TIER_EVAL_CONFIGS:
        available = ", ".join(sorted(TIER_EVAL_CONFIGS.keys()))
        msg = f"Unknown tier '{args.tier}'. " f"Available tiers: {available}"
        raise SystemExit(msg)

    tier_config = get_tier_config(tier_name)

    result = run_tier_evaluation(
        tier_config=tier_config,
        candidate_id=args.candidate_config,
        seed=args.seed,
        num_games_override=args.num_games,
    )

    _print_human_summary(result)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        data = result.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"\nJSON summary written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
