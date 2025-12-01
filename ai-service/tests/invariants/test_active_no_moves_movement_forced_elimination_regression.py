from __future__ import annotations

"""Regression for ACTIVE/MOVEMENT no-move invariant with forced elimination.

This test guards the scenario captured in
``ai-service/logs/invariant_failures/active_no_moves_p1_1763999745.json``
where the Python GameEngine previously raised the strict invariant after a
territory-processing / elimination step:

- ``game_status == ACTIVE``
- ``current_phase == MOVEMENT``
- ``current_player == 1``
- ``get_valid_moves(state, 1)`` returned ``[]``
- but ``_get_forced_elimination_moves(state, 1)`` was non-empty.

Under the compact rules / TS TurnEngine semantics, a blocked player who
still controls stacks always has at least one legal *action*: they may be
forced to pay a self-elimination cost rather than being left in a dead
state. The strict invariant has therefore been generalised to treat
forced-elimination availability as satisfying the "ACTIVE state has a
legal action" contract.

This test replays the recorded snapshot under strict-invariant mode and
asserts that:

- ``GameEngine.apply_move`` no longer raises the invariant RuntimeError,
  and
- any resulting ACTIVE state for the next player exposes either
  interactive moves or at least one forced-elimination move.
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
from app.rules import global_actions as ga  # noqa: E402


SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "logs",
    "invariant_failures",
    "active_no_moves_p1_1763999745.json",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SNAPSHOT_PATH),
    reason=(
        "Invariant-failure snapshot not found; "
        "run strict soak to regenerate"
    ),
)
def test_movement_forced_elimination_invariant_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-apply the recorded elimination/territory sequence under strict mode.

    Historically this snapshot represented an ACTIVE / MOVEMENT state for
    player 1 with no interactive moves but at least one available
    forced-elimination move. The earlier strict invariant only consulted
    ``get_valid_moves`` and therefore treated this as a violation.

    After generalising the invariant to also count forced-elimination
    availability, re-applying the same move from the recorded state should
    no longer raise, and any resulting ACTIVE state must expose at least one
    interactive move *or* forced elimination for the current player.
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

    # Sanity-check that we are starting from the recorded bad shape.
    assert state.game_status == GameStatus.ACTIVE
    assert state.current_phase.value == "movement"

    # Under the generalised invariant, this call should *not* raise a
    # RuntimeError about an ACTIVE state with no legal actions.
    next_state = GameEngine.apply_move(state, move)

    # As an extra guard, ensure that any ACTIVE state we reach obeys the
    # invariant from both the legacy "legal moves" view and the R200
    # global-actions view.
    if next_state.game_status == GameStatus.ACTIVE:
        legal = GameEngine.get_valid_moves(
            next_state,
            next_state.current_player,
        )
        get_forced_elim = GameEngine._get_forced_elimination_moves
        forced = get_forced_elim(
            next_state,
            next_state.current_player,
        )
        summary = ga.global_legal_actions_summary(
            next_state,
            next_state.current_player,
        )

        # INV-ACTIVE-NO-MOVES (R200–R203, ANM-SCEN-01):
        # ACTIVE state must expose at least one global action.
        assert (
            legal or forced or summary.has_global_placement_action
        ), (
            "Regression: ACTIVE state with no interactive moves, no forced "
            "elimination, and no global placements for current_player"
        )

        assert summary.has_turn_material is True

        # Explicit ANM predicate should be false after the regression fix.
        assert not ga.is_anm_state(next_state)
