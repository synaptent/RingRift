#!/usr/bin/env python
"""
Lightweight diagnostics for the HeuristicAI fitness landscape.

This script samples a handful of extreme and random weight profiles and
evaluates them against the canonical balanced baseline using the same
`evaluate_fitness` helper that CMA-ES uses. It is intended to answer
questions such as:

- Do obviously bad or heavily scaled weights perform noticeably worse
  than the baseline on the current evaluation harness?
- Does the baseline look stronger on some boards than others?
- How much variation in fitness do we see from random profiles?

Usage (from ai-service/):

    python scripts/diagnose_heuristic_landscape.py

The script prints a small table of fitness values per profile and board
type. It is deliberately conservative in games_per_eval so that a quick
run completes in a reasonable amount of time.
"""

from __future__ import annotations

import os
import random

# Allow imports from app/ when run as a script.
import sys
from collections.abc import Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS, HeuristicWeights  # type: ignore
from app.models import BoardType  # type: ignore
from scripts.run_cmaes_optimization import evaluate_fitness  # type: ignore


def _make_zero_weights() -> HeuristicWeights:
    return dict.fromkeys(BASE_V1_BALANCED_WEIGHTS.keys(), 0.0)


def _make_scaled_weights(scale: float) -> HeuristicWeights:
    return {k: v * scale for k, v in BASE_V1_BALANCED_WEIGHTS.items()}


def _make_random_weights(seed: int, low: float = -10.0, high: float = 30.0) -> HeuristicWeights:
    rng = random.Random(seed)
    return {k: rng.uniform(low, high) for k in BASE_V1_BALANCED_WEIGHTS}


def _evaluate_profiles(
    profiles: dict[str, HeuristicWeights],
    boards: Iterable[BoardType],
    games_per_eval: int,
) -> None:
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)

    print(f"Evaluating {len(profiles)} profiles against baseline on {len(list(boards))} boards")
    print(f"Games per eval: {games_per_eval}")
    print()

    header = f"{'profile':<24} {'board':<10} {'fitness':>8}"
    print(header)
    print("-" * len(header))

    for name, weights in profiles.items():
        for board in boards:
            fitness = evaluate_fitness(
                candidate_weights=weights,
                baseline_weights=baseline,
                games_per_eval=games_per_eval,
                board_type=board,
                verbose=False,
                opponent_mode="baseline-only",
            )
            print(f"{name:<24} {board.value:<10} {fitness:8.4f}")


def main() -> None:
    # Modest number of games per eval so the script can complete quickly.
    games_per_eval = 16

    profiles: dict[str, HeuristicWeights] = {
        "baseline": dict(BASE_V1_BALANCED_WEIGHTS),
        "all_zero": _make_zero_weights(),
        "scaled_5x": _make_scaled_weights(5.0),
        "scaled_0.2x": _make_scaled_weights(0.2),
        "random_seed_1": _make_random_weights(seed=1),
        "random_seed_2": _make_random_weights(seed=2),
    }

    boards = [BoardType.SQUARE8, BoardType.SQUARE19, BoardType.HEXAGONAL]

    _evaluate_profiles(profiles, boards, games_per_eval)


if __name__ == "__main__":
    main()
