#!/usr/bin/env python
"""Smoke tests for heuristic CMA-ES-style heuristic weight optimisation."""

from __future__ import annotations

import os
import sys
from typing import List


# Ensure app/ is importable when running tests directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.training.train import (  # noqa: E402
    run_cmaes_heuristic_optimization,
)


def test_cmaes_heuristic_smoke(monkeypatch) -> None:
    """Fast, deterministic smoke test for the CMA-ES heuristic harness.

    The real evaluation of heuristic candidates is expensive because it plays
    full games via the eval-pool harness. For unit tests we stub
    ``evaluate_heuristic_candidate`` so that:

    * Fitness is computed from the candidate vector only.
    * No game engine or eval-pool IO is involved.
    """
    calls: list[float] = []

    def fake_evaluate_heuristic_candidate(
        tier_spec,
        base_profile_id,
        keys,
        candidate_vector,
        rng_seed,
        games_per_candidate=None,
    ):
        # Simple convex fitness surface: optimum at the zero vector.
        fitness = -sum(float(v) ** 2 for v in candidate_vector)
        result = {
            "games_played": 1,
            "results": {"wins": 0, "losses": 0, "draws": 1},
            "margins": {
                "ring_margin_mean": 0.0,
                "territory_margin_mean": 0.0,
            },
        }
        calls.append(fitness)
        return fitness, result

    # Stub out the expensive evaluation harness.
    monkeypatch.setattr(
        "app.training.train.evaluate_heuristic_candidate",
        fake_evaluate_heuristic_candidate,
    )

    report = run_cmaes_heuristic_optimization(
        tier_id="sq8_heuristic_baseline_v1",
        base_profile_id="heuristic_v1_balanced",
        generations=2,
        population_size=4,
        rng_seed=42,
        games_per_candidate=1,
    )

    assert report["run_type"] == "heuristic_cmaes_square8"
    assert report["generations"] == 2
    assert report["population_size"] == 4
    assert report["best"]["fitness"] == max(calls)
    assert len(report["history"]) == 2