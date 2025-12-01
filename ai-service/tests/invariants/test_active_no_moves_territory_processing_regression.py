from __future__ import annotations

"""Regression for ACTIVE/TERRITORY_PROCESSING no-move invariant.

This test guards the territory-processing snapshot captured in
``ai-service/logs/invariant_failures/active_no_moves_p1_1764000947.json``.

Historically, after a PROCESS_TERRITORY_REGION decision for player 1, the
Python GameEngine could leave the state in:

- ``game_status == ACTIVE``
- ``current_phase == TERRITORY_PROCESSING``
- ``current_player == 1``
- ``get_valid_moves(state, 1) == []`` and no forced-elimination moves

even though player 1 had **no material at all** (no stacks, no rings in
hand) while player 2 still had stacks and placements available. Under the
shared TS `advanceTurnAndPhase` / backend `TurnEngine` semantics, all
territory-processing decisions for the moving player are followed by a
turn rotation to the next player with material; we should never end a
territory step with a player who cannot act.

We fixed this by teaching ``GameEngine._update_phase`` to re-evaluate
territory-processing moves after each PROCESS_TERRITORY_REGION /
TERRITORY_CLAIM / CHOOSE_TERRITORY_OPTION decision, and call
``_end_turn`` when none remain. That brings the Python phase sequencer
back in line with the TS lane.

This regression test replays the recorded snapshot under strict-invariant
mode and asserts that re-applying the recorded territory-processing move
no longer yields an ACTIVE / TERRITORY_PROCESSING state for a player with
no actions.
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
    "active_no_moves_p1_1764000947.json",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SNAPSHOT_PATH),
    reason=(
        "Invariant-failure snapshot not found; "
        "run strict soak to regenerate"
    ),
)
def test_territory_processing_invariant_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-apply the recorded territory-processing move under strict invariant.

    Under the fixed semantics, this call must not re-produce an ACTIVE
    / TERRITORY_PROCESSING state for a player who has neither interactive
    moves nor forced-elimination moves available.
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
    assert state.current_phase.value == "territory_processing"

    # Under the regression fix, this call should *not* raise a RuntimeError
    # about an ACTIVE state with no legal actions.
    next_state = GameEngine.apply_move(state, move)

    # As an extra guard, ensure that any ACTIVE state we reach obeys the
    # no-dead-action invariant from both the legacy and global-actions views.
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

        # INV-ACTIVE-NO-MOVES / INV-PHASE-CONSISTENCY (ANM-SCEN-04):
        # current player in TERRITORY_PROCESSING must have at least one
        # global action; ANM(state) must be false.
        assert (
            legal or forced
        ), (
            "Regression: ACTIVE state with neither legal moves nor "
            "forced eliminations"
        )
        assert summary.has_turn_material is True
        assert ga.is_anm_state(next_state) is False
