"""AI plateau progress tests for Python GameEngine.

These tests start from TS-exported plateau snapshots for
square8 / 2 AI seeds (1 and 18), hydrate them into Python GameState
instances, and then:

* run the Python GameEngine forward using a simple policy
  (first legal move for the current player),
* assert the S-invariant (markers + collapsed + eliminated) is
  never decreasing, and
* assert we do not observe a long active stall (>= 8 consecutive
  no-op steps in terms of state hash and S) within a generous
  bound of follow-up actions.

This mirrors the TS-side stall regression tests and provides a
lightweight AI-level parity check for plateau behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure app package is importable when running tests directly.
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.board_manager import BoardManager  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.models.core import GameState  # noqa: E402

from tests.parity.test_ts_seed_plateau_snapshot_parity import (  # noqa: E402
    SEED1_SNAPSHOT,
    SEED18_SNAPSHOT,
    _build_game_state_from_snapshot,
)


def _load_plateau_state(path: Path) -> GameState:
    """Load a TS plateau ComparableSnapshot JSON and hydrate GameState."""
    with path.open("r", encoding="utf-8") as f:
        snapshot: dict[str, Any] = json.load(f)

    return _build_game_state_from_snapshot(snapshot)


@pytest.mark.parametrize("fixture_path", [SEED18_SNAPSHOT, SEED1_SNAPSHOT])
def test_ai_progress_from_plateau_snapshots(fixture_path: Path) -> None:
    """Python GameEngine should not regress S or stall from plateau states.

    For each plateau snapshot (seed 18 and seed 1):

    1. Hydrate a Python GameState via the same builder used for
       snapshot parity.
    2. Compute initial S-invariant via BoardManager.compute_progress_snapshot.
    3. Step the Python GameEngine forward, always choosing the first
       legal move for the current player, up to a generous bound
       (MAX_STEPS).
    4. At each step:
       - Assert S is non-decreasing.
       - Track (S, zobrist_hash) stagnation; if we see >= 8 consecutive
         active steps with unchanged S and zobrist_hash, treat this as
         a stall regression.
    5. At the end, assert that final S is at least the initial S and
       that either the game has completed or we made observable
       progress (S increased at least once).
    """

    if not fixture_path.exists():
        pytest.skip(
            "TS plateau snapshot fixture not found. Run the TS exporters "
            "(ExportSeed18Snapshot/ExportSeed1Snapshot Jest tests) first."
        )

    state = _load_plateau_state(fixture_path)

    # Initial invariant snapshot
    snap = BoardManager.compute_progress_snapshot(state)
    initial_S = snap.S

    # Use zobrist_hash when available to detect exact board repeats.
    last_hash = BoardManager.hash_game_state(state)
    last_S = initial_S
    stagnant_steps = 0
    S_increased_at_least_once = False

    MAX_STEPS = 200

    for _ in range(MAX_STEPS):
        if state.game_status.name.lower() not in {"active", "waiting"}:
            break

        current_player = state.current_player
        moves = GameEngine.get_valid_moves(state, current_player)

        if not moves:
            # No legal moves for current player; rely on GameEngine's
            # internal end-of-turn / forced-elimination logic to
            # resolve via apply_move in real usage. For this test,
            # treat it as no further progress possible.
            break

        move = moves[0]
        state = GameEngine.apply_move(state, move)

        new_snap = BoardManager.compute_progress_snapshot(state)
        new_S = new_snap.S
        new_hash = BoardManager.hash_game_state(state)

        # S-invariant must be non-decreasing.
        assert new_S >= last_S, (
            "Python AI plateau regression: S decreased from "
            f"{last_S} to {new_S} starting from plateau fixture "
            f"{fixture_path.name}"
        )

        if new_S > last_S:
            S_increased_at_least_once = True

        if (
            state.game_status.name.lower() == "active"
            and new_S == last_S
            and new_hash == last_hash
        ):
            stagnant_steps += 1
        else:
            stagnant_steps = 0

        assert stagnant_steps < 8, (
            "Python AI plateau regression: observed >= 8 consecutive "
            "no-op active steps (no change in S or hash) from plateau "
            f"fixture {fixture_path.name}"
        )

        last_S = new_S
        last_hash = new_hash

    # Final sanity: S should not have decreased overall, and if we
    # didn't finish, we should at least have made some progress.
    final_snap = BoardManager.compute_progress_snapshot(state)
    assert final_snap.S >= initial_S

    if state.game_status.name.lower() == "active":
        assert S_increased_at_least_once, (
            "Python AI plateau regression: game remained active after "
            f"{MAX_STEPS} steps from plateau fixture {fixture_path.name} "
            "without any increase in S."
        )
