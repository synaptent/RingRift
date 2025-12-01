from __future__ import annotations

"""ANM and termination invariant smoke tests.

These tests construct small synthetic states to exercise:

- INV-ACTIVE-NO-MOVES
- INV-ANM-TURN-MATERIAL-SKIP
- INV-TERMINATION / INV-ELIMINATION-MONOTONIC
"""

import os
import sys

import pytest

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    GameStatus,
    Player,
    Position,
    RingStack,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402
from app.rules.global_actions import (  # noqa: E402
    apply_forced_elimination_for_player,
)
from tests.rules.helpers import (  # noqa: E402
    _make_base_game_state,
    _make_place_ring_move,
)


# INV-ACTIVE-NO-MOVES (R200–R203)
def test_strict_invariant_raises_on_synthetic_anm_state() -> None:
    """Synthetic ANM state triggers strict invariant."""

    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_player = 1

    state.board.stacks.clear()
    state.board.markers.clear()
    state.board.collapsed_spaces.clear()
    state.board.size = 0

    player = state.players[0]
    player.rings_in_hand = 1

    assert ga.has_turn_material(state, 1) is True
    assert ga.is_anm_state(state) is True

    dummy_move = _make_place_ring_move(player=1, x=0, y=0)
    checker = GameEngine._assert_active_player_has_legal_action
    with pytest.raises(RuntimeError):
        checker(state, dummy_move)


# INV-ANM-TURN-MATERIAL-SKIP (R201)
def test_end_turn_skips_fully_eliminated_player() -> None:
    """_end_turn skips players with no turn material in ACTIVE games."""

    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE

    p1, p2 = state.players
    state.players.append(
        Player(
            id="p3",
            username="p3",
            type="human",
            playerNumber=3,
            isReady=True,
            timeRemaining=p1.time_remaining,
            aiDifficulty=None,
            ringsInHand=1,
            eliminatedRings=0,
            territorySpaces=0,
        )
    )

    p1.rings_in_hand = 0
    state.board.stacks.clear()
    state.current_player = 1

    GameEngine._end_turn(state)

    assert state.current_player in (2, 3)
    assert ga.has_turn_material(state, state.current_player) is True
    assert ga.is_anm_state(state) is False


# INV-TERMINATION / INV-ELIMINATION-MONOTONIC (R191, R207)
def test_forced_elimination_chain_is_monotone_and_finite() -> None:
    """Forced-elimination-only chain is monotone and finite."""

    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_player = 1

    player = state.players[0]
    player.rings_in_hand = 0

    board = state.board
    board.size = 1
    pos = Position(x=0, y=0)
    key = pos.to_key()

    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    stack = RingStack(
        position=pos,
        rings=[1, 1, 1],
        stackHeight=3,
        capHeight=3,
        controllingPlayer=1,
    )
    board.stacks[key] = stack

    total_before = state.total_rings_eliminated
    steps = 0

    while steps < 10:
        outcome = apply_forced_elimination_for_player(state, 1)
        if outcome is None:
            break

        assert outcome.eliminated_player == 1
        assert outcome.eliminated_count >= 1
        assert state.total_rings_eliminated > total_before

        total_before = state.total_rings_eliminated
        steps += 1

    assert steps > 0
    assert apply_forced_elimination_for_player(state, 1) is None