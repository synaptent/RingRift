"""Integration tests for evaluation randomness in the CMA-ES harness.

These tests ensure that the shared fitness evaluator in
`scripts/run_cmaes_optimization.py` remains deterministic given a fixed
seed, both when eval_randomness is 0.0 (fully deterministic evaluation)
and when eval_randomness is a small positive value (controlled stochastic
tie-breaking).
"""

from __future__ import annotations

import os
import sys

# Ensure app package and training scripts are importable when running tests
# directly (mirrors pattern used in other ai-service tests).
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.ai.heuristic_weights import (  # type: ignore  # noqa: E402
    BASE_V1_BALANCED_WEIGHTS,
)
from app.models import BoardType  # type: ignore  # noqa: E402
from scripts.run_cmaes_optimization import (  # type: ignore  # noqa: E402
    evaluate_fitness,
)


def test_eval_randomness_zero_is_deterministic() -> None:
    """eval_randomness=0.0 must be deterministic for a fixed seed."""
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)

    f1 = evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=4,
        board_type=BoardType.SQUARE8,
        opponent_mode="baseline-only",
        max_moves=20,
        verbose=False,
        eval_mode="initial-only",
        state_pool_id=None,
        eval_randomness=0.0,
        seed=12345,
    )
    f2 = evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=4,
        board_type=BoardType.SQUARE8,
        opponent_mode="baseline-only",
        max_moves=20,
        verbose=False,
        eval_mode="initial-only",
        state_pool_id=None,
        eval_randomness=0.0,
        seed=12345,
    )

    assert f1 == f2


def test_eval_randomness_nonzero_is_seed_deterministic() -> None:
    """eval_randomness>0.0 must still be deterministic for a fixed seed."""
    baseline = dict(BASE_V1_BALANCED_WEIGHTS)

    f1 = evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=4,
        board_type=BoardType.SQUARE8,
        opponent_mode="baseline-only",
        max_moves=20,
        verbose=False,
        eval_mode="initial-only",
        state_pool_id=None,
        eval_randomness=0.05,
        seed=999,
    )
    f2 = evaluate_fitness(
        candidate_weights=baseline,
        baseline_weights=baseline,
        games_per_eval=4,
        board_type=BoardType.SQUARE8,
        opponent_mode="baseline-only",
        max_moves=20,
        verbose=False,
        eval_mode="initial-only",
        state_pool_id=None,
        eval_randomness=0.05,
        seed=999,
    )

    assert f1 == f2