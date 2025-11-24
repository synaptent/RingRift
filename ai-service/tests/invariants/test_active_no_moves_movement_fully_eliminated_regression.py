from __future__ import annotations

"""Regression for ACTIVE/MOVEMENT no-move invariant with fully eliminated player.

This test guards the scenario captured in
``ai-service/logs/invariant_failures/active_no_moves_p1_1764002534.json``.

In that snapshot, after a long chain-capture and territory-processing
sequence, the Python GameEngine had rotated the turn into a shape with:

- ``game_status == ACTIVE``
- ``current_phase == MOVEMENT``
- ``current_player == 1``
- player 1 having **no stacks and no rings in hand** (fully eliminated)
- player 2 still having stacks and rings in hand
- ``get_valid_moves(state, 1) == []``
- ``_get_forced_elimination_moves(state, 1) == []``

Under the TS TurnEngine semantics, a player with no material (no stacks and
no rings in hand) is skipped entirely during turn rotation; they should
never be left as the active player in an ACTIVE game state. The Python
strict no-move invariant now defends against this class of bug by
performing a final `_end_turn` attempt for fully eliminated players before
raising.

This regression test replays the recorded snapshot under strict-invariant
mode and asserts that:

- ``GameEngine.apply_move`` no longer raises the invariant RuntimeError for
  this shape, and
- any resulting ACTIVE state exposes at least one interactive move or forced
  elimination for the *current* player, and
- a fully eliminated player (no stacks, no rings in hand) is not left as
  the active player in an ACTIVE state.
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
    "active_no_moves_p1_1764002534.json",
)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(SNAPSHOT_PATH),
    reason="Invariant-failure snapshot not found; run strict soak to regenerate",
)
def test_movement_fully_eliminated_player_invariant_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-apply the recorded move under strict mode.

    Historically this snapshot left player 1 as the active player in an
    ACTIVE / MOVEMENT state despite them having no stacks and no rings in
    hand, while player 2 still had material. The strict no-move invariant
    fired because there were no interactive moves or forced eliminations
    available for player 1.

    After hardening the invariant to perform a final `_end_turn` for fully
    eliminated players before declaring failure, re-applying the same move
    from the recorded "bad" state should:

    - not raise a RuntimeError, and
    - either terminate the game or rotate to a player who has at least one
      interactive move or forced-elimination move.
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

    # Player 1 should have no stacks and no rings in hand in the snapshot.
    p1 = next(p for p in state.players if p.player_number == 1)
    p1_stacks = [
        s for s in state.board.stacks.values()
        if s.controlling_player == 1
    ]
    assert not p1_stacks
    assert p1.rings_in_hand == 0

    # Under the hardened invariant, this call should *not* raise a
    # RuntimeError about an ACTIVE state with no legal actions.
    next_state = GameEngine.apply_move(state, move)

    if next_state.game_status == GameStatus.ACTIVE:
        # The current player must have at least one interactive move or
        # forced-elimination move.
        legal = GameEngine.get_valid_moves(
            next_state,
            next_state.current_player,
        )
        forced = GameEngine._get_forced_elimination_moves(  # type: ignore[attr-defined]
            next_state,
            next_state.current_player,
        )
        assert (
            legal or forced
        ), "Regression: ACTIVE state with neither legal moves nor forced eliminations"

        # A fully eliminated player must not remain the active player in an
        # ACTIVE state.
        p1_after = next(
            p for p in next_state.players if p.player_number == 1
        )
        p1_after_stacks = [
            s for s in next_state.board.stacks.values()
            if s.controlling_player == 1
        ]
        if not p1_after_stacks and p1_after.rings_in_hand == 0:
            assert (
                next_state.current_player != 1
            ), "Regression: fully eliminated player kept as current_player in ACTIVE state"
