"""Minimal smoke test for multi-start evaluation in the CMA-ES harness.

This test verifies that the multi-start evaluation path in
`scripts.run_cmaes_optimization.evaluate_fitness` can be exercised using a
temporary Square8 state pool, and that baseline-vs-baseline evaluation
produces a sensible fitness value in [0.0, 1.0].

It does **not** attempt to be an exhaustive statistical test; the goal is to
ensure that the wiring between:

- the eval_pools loader
- the multi-start evaluation loop
- and the CLI-facing `eval_mode` / `state_pool_id` parameters

is functional and deterministic for a small number of games.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure app.* imports resolve when running tests directly under ai-service/
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import contextlib

from app.ai.heuristic_weights import (  # type: ignore
    BASE_V1_BALANCED_WEIGHTS,
)
from app.models import (  # type: ignore
    BoardType,
    GameState,
)
from app.training import (  # type: ignore
    eval_pools,
)
from app.training.env import (  # type: ignore
    RingRiftEnv,
)
from scripts.run_cmaes_optimization import (  # type: ignore
    evaluate_fitness,
)


def _write_temp_state_pool(states: list[GameState]) -> str:
    """Write a small JSONL state pool to a temporary file.

    Returns
    -------
    str
        The filesystem path to the created JSONL file.
    """
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)

    with open(path, "w", encoding="utf-8") as f:
        for state in states:
            # Use model_dump_json to mirror run_self_play_soak behaviour.
            payload = state.model_dump_json()  # type: ignore[attr-defined]
            f.write(payload)
            f.write("\n")

    return path


@pytest.mark.parametrize("games_per_eval", [2])
def test_multi_start_baseline_vs_baseline_square8(games_per_eval: int) -> None:
    """Baseline-vs-baseline evaluation via multi-start should be well-behaved.

    The exact numeric fitness is not asserted here (it may depend on the
    particular mid-game positions sampled), but for identical weights on both
    sides we at least require that:

    - evaluation runs without raising, and
    - the resulting fitness lies in [0.0, 1.0].
    """
    # Build a tiny pool of Square8 GameStates using the shared training env.
    env = RingRiftEnv(BoardType.SQUARE8)
    base_state = env.reset()
    states: list[GameState] = [base_state]

    # Take a single legal move (if available) to obtain a non-trivial mid-game
    # snapshot as a second entry in the pool.
    moves = env.legal_moves()
    if moves:
        move = moves[0]
        next_state, _reward, done, _info = env.step(move)
        # Only use ACTIVE states as pool entries to match run_self_play_soak.
        if not done and next_state.game_status == base_state.game_status:
            states.append(next_state)

    pool_path = _write_temp_state_pool(states)

    # Patch POOL_PATHS so that (SQUARE8, "test") resolves to our temp file.
    key = (BoardType.SQUARE8, "test")
    old_mapping = dict(eval_pools.POOL_PATHS)
    try:
        eval_pools.POOL_PATHS[key] = pool_path

        baseline = dict(BASE_V1_BALANCED_WEIGHTS)

        fitness = evaluate_fitness(
            candidate_weights=baseline,
            baseline_weights=baseline,
            games_per_eval=games_per_eval,
            board_type=BoardType.SQUARE8,
            opponent_mode="baseline-only",
            max_moves=20,
            verbose=False,
            eval_mode="multi-start",
            state_pool_id="test",
        )

        assert 0.0 <= fitness <= 1.0
    finally:
        # Restore the original mapping and remove the temp file.
        eval_pools.POOL_PATHS.clear()
        eval_pools.POOL_PATHS.update(old_mapping)
        with contextlib.suppress(OSError):
            os.remove(pool_path)
