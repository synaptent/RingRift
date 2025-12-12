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
    GamePhase,
    GameStatus,
    MoveType,
    Player,
    Position,
    RingStack,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402
from app.rules.global_actions import (  # noqa: E402
    apply_forced_elimination_for_player,
)
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from tests.rules.helpers import (  # noqa: E402
    _make_base_game_state,
    _make_place_ring_move,
)
from tests.parity.test_anm_global_actions_parity import (  # noqa: E402
    make_anm_scen01_movement_no_moves_but_fe_available,
    make_anm_scen02_movement_placements_only,
    make_anm_scen03_movement_current_player_fully_eliminated,
    make_anm_scen06_global_stalemate_bare_board,
)


# ARCHIVED TEST: test_strict_invariant_raises_on_synthetic_anm_state
# Removed 2025-12-07
#
# INV-ACTIVE-NO-MOVES (R200-R203)
#
# This test expected _assert_active_player_has_legal_action to raise RuntimeError
# for a synthetic ANM state (player with material but no valid placement positions).
# However, the current (correct) invariant implementation also checks for phase
# requirements (see GameEngine._assert_active_player_has_legal_action lines 1378-1393).
#
# When a player has rings_in_hand but all positions are collapsed, the engine
# returns a NO_PLACEMENT_ACTION phase requirement. This means the state is NOT
# an invariant violation - hosts are expected to emit the bookkeeping move.
#
# The invariant only raises when:
# 1. No interactive moves exist AND
# 2. No phase requirement exists
#
# The test's synthetic state always has a phase requirement (NO_PLACEMENT_ACTION),
# so the invariant is correctly satisfied.


# INV-ANM-TURN-MATERIAL-SKIP (RR-CANON-R201)
#
# Per canonical rules (RR-CANON-R201):
# - A player has "turn-material" if they control stacks OR have rings in hand
# - A player is "permanently eliminated" if they have NO rings anywhere
#   (no controlled stacks, no rings in hand, no buried rings)
# - Only PERMANENTLY ELIMINATED players are skipped in turn rotation
# - Players without turn-material but with buried rings still get turns
#   (they may use recovery slides in MOVEMENT phase)
#
# This test verifies that when P1 has no stacks and no rings in hand,
# they are skipped (in this test P1 is fully eliminated with no buried rings).
def test_end_turn_skips_fully_eliminated_player() -> None:
    """_end_turn skips permanently eliminated players (no rings anywhere)."""

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
    pos = Position(x=3, y=3)
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

    # Collapse all positions except the stack position to trap it.
    # This forces the player to use forced elimination as their only action.
    for x in range(8):
        for y in range(8):
            if x == pos.x and y == pos.y:
                continue
            collapse_pos = Position(x=x, y=y)
            board.collapsed_spaces[collapse_pos.to_key()] = 2  # Enemy territory

    total_before = state.total_rings_eliminated
    steps = 0

    while steps < 10:
        outcome = apply_forced_elimination_for_player(state, 1)
        if outcome is None:
            break

        assert outcome.eliminated_player == 1
        assert outcome.eliminated_count >= 1
        assert state.total_rings_eliminated > total_before

        if state.game_status == GameStatus.ACTIVE:
            assert ga.is_anm_state(state) is False

        total_before = state.total_rings_eliminated
        steps += 1

    assert steps > 0
    assert apply_forced_elimination_for_player(state, 1) is None


def test_is_anm_state_basic_logical_invariants() -> None:
    """Basic logical properties of is_anm_state and global action summary."""

    # If a player has no turn material, ANM(state) must be false.
    state = make_anm_scen03_movement_current_player_fully_eliminated()
    player = state.current_player
    summary = ga.global_legal_actions_summary(state, player)

    assert summary.has_turn_material is False
    assert ga.is_anm_state(state) is False

    # If any global action exists (placement / phase-local / FE), ANM must be
    # false.
    state2 = make_anm_scen02_movement_placements_only()
    player2 = state2.current_player
    summary2 = ga.global_legal_actions_summary(state2, player2)

    assert (
        summary2.has_global_placement_action
        or summary2.has_phase_local_interactive_move
        or summary2.has_forced_elimination_action
    )
    assert ga.is_anm_state(state2) is False


def test_global_stalemate_is_terminal_not_anm() -> None:
    """Only bare-board global stalemate may have an empty global surface."""

    state = make_anm_scen06_global_stalemate_bare_board()

    for player in state.players:
        summary = ga.global_legal_actions_summary(state, player.player_number)
        assert summary.has_turn_material is True
        assert summary.has_global_placement_action is False
        assert summary.has_phase_local_interactive_move is False
        assert summary.has_forced_elimination_action is False

    assert ga.is_anm_state(state) is False

    GameEngine._check_victory(state)  # type: ignore[attr-defined]

    assert state.game_status != GameStatus.ACTIVE
    assert state.current_phase == GamePhase.GAME_OVER
    assert state.winner in {p.player_number for p in state.players}


def test_no_movement_action_sequence_does_not_hit_anm() -> None:
    """NO_MOVEMENT_ACTION bookkeeping does not leave ANM ACTIVE states."""

    state = make_anm_scen02_movement_placements_only()
    player = state.current_player

    # Starting shape is not ANM by construction.
    assert ga.is_anm_state(state) is False

    engine = DefaultRulesEngine(mutator_first=False, skip_shadow_contracts=True)
    moves = engine.get_valid_moves(state, player)

    assert moves, "Expected at least one bookkeeping move"

    no_move_actions = [
        m for m in moves if m.type == MoveType.NO_MOVEMENT_ACTION
    ]
    assert len(no_move_actions) == 1

    next_state = engine.apply_move(state, no_move_actions[0])

    if next_state.game_status == GameStatus.ACTIVE:
        assert ga.is_anm_state(next_state) is False