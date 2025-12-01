from __future__ import annotations

"""Regression test for ACTIVE/LINE_PROCESSING no-move invariant.

This test guards against the historical bug where the Python GameEngine
could leave the game in an ACTIVE state with current_phase ==
LINE_PROCESSING and no legal moves for current_player, diverging from the
TS TurnLogic semantics (which always advance out of line_processing when
no further line decisions remain).

We re-use a concrete invariant-failure snapshot captured during strict
self-play soaks (see ai-service/logs/invariant_failures). From that
snapshot we hydrate the recorded GameState and triggering Move, then
re-apply the move via GameEngine.apply_move with the strict invariant
flag enabled.

The regression is considered fixed when this re-application no longer
raises a RuntimeError and does not leave an ACTIVE state with no legal
moves for current_player.
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
    "active_no_moves_p2_1763996214.json",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SNAPSHOT_PATH),
    reason="Invariant-failure snapshot not found; run strict soak to regenerate",
)
def test_line_processing_invariant_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-apply the recorded line-processing move under strict invariant.

    Historically this snapshot represented an ACTIVE / LINE_PROCESSING
    state for player 2 with no legal moves, which violated the shared
    TS TurnLogic invariant. After fixing GameEngine._update_phase to
    advance through _advance_to_line_processing, re-applying the same
    move from the recorded state should no longer produce an invariant
    violation.
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

    # Hydrate models from the Pydantic v2-compatible dumps stored by the
    # invariant logger in GameEngine.apply_move.
    state = GameState.model_validate(state_payload)
    move = Move.model_validate(move_payload)

    # Sanity-check that we are starting from the recorded bad shape.
    assert state.game_status == GameStatus.ACTIVE
    assert state.current_phase.value == "line_processing"

    # Under the regression fix, this call should *not* raise a RuntimeError
    # about an ACTIVE state with no legal moves. If it does, the invariant
    # is still being violated for this scenario.
    next_state = GameEngine.apply_move(state, move)

    # As an extra guard, ensure that any ACTIVE state we reach obeys both the
    # phase-local and global ANM invariants.
    if next_state.game_status == GameStatus.ACTIVE:
        curr_player = next_state.current_player
        legal = GameEngine.get_valid_moves(next_state, curr_player)
        assert (
            legal
        ), (
            "Regression: ACTIVE state with no legal moves after "
            "re-applying snapshot move"
        )

        # INV-ACTIVE-NO-MOVES / INV-PHASE-CONSISTENCY (ANM-SCEN-05):
        # current player in LINE_PROCESSING must have turn-material and at
        # least one global action; ANM(state) must be false.
        summary = ga.global_legal_actions_summary(next_state, curr_player)
        assert summary.has_turn_material is True
        assert ga.is_anm_state(next_state) is False
