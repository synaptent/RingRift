from __future__ import annotations

"""Regression for ACTIVE/MOVEMENT no-move invariant with placements only.

This test guards the scenario captured in
``ai-service/logs/invariant_failures/active_no_moves_p1_1764007977.json``.

In that snapshot, after a long sequence of line and territory work, the
Python GameEngine reached a shape with:

- ``game_status == ACTIVE``
- ``current_phase == MOVEMENT``
- ``current_player == 1``
- player 1 still having rings in hand (legal ring placements exist)
- but no legal movement/capture/forced-elimination moves

The original strict invariant only consulted phase-specific interactive
moves via ``get_valid_moves(state, player)`` together with forced
elimination. Because placements are not exposed in MOVEMENT, the engine
wrongly concluded that player 1 had *no legal actions* and raised a
RuntimeError.

We resolved this by aligning the invariant with the TS TurnEngine
"hasValidActions" contract: any ACTIVE state is considered valid as long
as the current player has *some* global action available:

- a ring placement, movement, or capture, or
- a forced-elimination decision when they still control stacks.

This regression test replays the recorded snapshot under strict mode and
asserts that re-applying the recorded move no longer triggers the
invariant, even though the MOVEMENT phase itself exposes no moves for
player 1. The invariant is now satisfied because global placements are
available.
"""

import json
import os
import sys

import pytest

# Ensure app package is importable when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import GameState, Move, GameStatus  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402


SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "logs",
    "invariant_failures",
    "active_no_moves_p1_1764007977.json",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SNAPSHOT_PATH),
    reason=(
        "Invariant-failure snapshot not found; "
        "run strict soak to regenerate"
    ),
)
def test_movement_placements_only_invariant_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-apply the recorded line/territory sequence under strict mode.

    Historically this snapshot represented an ACTIVE / MOVEMENT state for
    player 1 where:

    - phase-specific interactive moves (movement + capture) were empty, and
    - no forced-elimination moves were available, but
    - global ring placements were still legal (player 1 had rings in hand
      and valid placement targets).

    The earlier strict invariant looked only at ``get_valid_moves`` for the
    current phase plus forced elimination, so it treated this as a dead
    state and raised. After generalising the invariant to count any global
    action (placements, movements, captures, or forced elimination),
    re-applying the same move from the recorded state should no longer
    raise.
    """

    # Enable the strict invariant both via environment and the module flag
    monkeypatch.setenv("RINGRIFT_STRICT_NO_MOVE_INVARIANT", "1")
    import app.game_engine as game_engine  # local import to patch module flag

    monkeypatch.setattr(
        game_engine,
        "STRICT_NO_MOVE_INVARIANT",
        True,
        raising=False,
    )

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_payload = payload["state"]
    move_payload = payload["move"]

    state = GameState.model_validate(state_payload)
    move = Move.model_validate(move_payload)

    # Sanity-check the recorded bad shape matches expectations.
    assert state.game_status == GameStatus.ACTIVE
    assert state.current_phase.value == "movement"
    assert state.current_player == 1

    # From the snapshot diagnostics we expect no interactive MOVEMENT moves
    # or forced eliminations for player 1, but at least one legal placement.
    phase_moves = GameEngine.get_valid_moves(state, 1)
    placements = GameEngine._get_ring_placement_moves(state, 1)  # type: ignore[attr-defined]
    forced = GameEngine._get_forced_elimination_moves(  # type: ignore[attr-defined]
        state,
        1,
    )

    assert not phase_moves
    assert not forced
    assert placements, (
        "Expected at least one legal ring placement for P1"
    )

    # Under the generalised invariant, this call should *not* raise a
    # RuntimeError about an ACTIVE state with no legal actions, because
    # global placements are available.
    next_state = GameEngine.apply_move(state, move)

    # As an extra guard, ensure that any ACTIVE state we reach exposes at
    # least one global action for the current player from the snapshot's
    # perspective as well (placements, movements, captures, or forced elim).
    if next_state.game_status == GameStatus.ACTIVE:
        legal_phase = GameEngine.get_valid_moves(
            next_state,
            next_state.current_player,
        )
        placements_next = GameEngine._get_ring_placement_moves(  # type: ignore[attr-defined]
            next_state,
            next_state.current_player,
        )
        forced_next = GameEngine._get_forced_elimination_moves(  # type: ignore[attr-defined]
            next_state,
            next_state.current_player,
        )
        assert (
            legal_phase or placements_next or forced_next
        ), (
            "Regression: ACTIVE state with no global legal actions "
            "for current_player"
        )
