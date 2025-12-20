from __future__ import annotations

"""
Forced-elimination semantics parity tests.

These tests mirror the TS globalActions.shared.test.ts expectations for
forced elimination:

- hasForcedEliminationAction is true exactly when the player controls at
  least one stack but has no legal placement, movement, or capture.
- Enumeration of forced-elimination options covers all controlled stacks
  with the correct capHeight / stackHeight metadata.
- apply_forced_elimination_for_player selects the stack with the smallest
  positive capHeight (falling back to the first stack when all caps are
  zero) and increases total_rings_eliminated in normal cases.
"""

import os
import sys
from typing import Dict, List, Tuple, TYPE_CHECKING

import pytest

# Ensure app package is importable when running tests directly from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    GamePhase,
    GameStatus,
    MoveType,
    Position,
    RingStack,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402
from tests.rules.helpers import _make_base_game_state  # noqa: E402

if TYPE_CHECKING:
    from app.models import GameState  # noqa: F401


def _block_all_positions_except(
    state: "GameState",
    allowed: list[Position],
) -> None:
    """Collapse all positions except the given ones to block movement/capture.

    This mirrors the helper used in
    tests/parity/test_forced_elimination_sequences_parity.py but keeps the
    implementation local so these tests remain self-contained.
    """
    board = state.board
    allowed_keys = {p.to_key() for p in allowed}
    size = board.size or 0

    for x in range(size):
        for y in range(size):
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key in allowed_keys:
                continue
            if key in board.stacks:
                # Preserve explicit stack cells so we still detect stacks as
                # turn material for forced-elimination checks.
                continue
            board.collapsed_spaces[key] = 2  # Treat as opponent territory


def _enumerate_forced_elimination_options(
    state: "GameState",
    player: int,
) -> dict[str, tuple[int, int]]:
    """Return {pos_key: (cap_height, stack_height)} for all FE candidates.

    This uses GameEngine._get_forced_elimination_moves as the Python analogue
    of TS enumerateForcedEliminationOptions, then projects stack metadata
    from the board.
    """
    moves = GameEngine._get_forced_elimination_moves(state, player)
    options: dict[str, tuple[int, int]] = {}
    board = state.board

    for move in moves:
        assert move.to is not None
        key = move.to.to_key()
        stack = board.stacks.get(key)
        assert stack is not None
        options[key] = (stack.cap_height, stack.stack_height)

    return options


def _build_mixed_caps_forced_elimination_state() -> tuple[
    "GameState",
    Position,
    list[Position],
]:
    """Blocked state where P1 has stacks with mixed positive capHeights.

    Geometry:

    - Three stacks for P1 at distinct positions with capHeights 3, 2, 1.
    - All other cells collapsed so no movement/capture is legal.
    - P1 has rings_in_hand == 0 so no placements are available.

    Expected semantics:

    - has_forced_elimination_action(state, 1) is True.
    - GameEngine._get_forced_elimination_moves surfaces all three stacks.
    - apply_forced_elimination_for_player chooses the stack with capHeight 1.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    # Use FORCED_ELIMINATION phase per RR-CANON-R070 - FE is its own phase
    state.current_phase = GamePhase.FORCED_ELIMINATION
    state.current_player = 1

    # Remove placements for player 1.
    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()

    # Three stacks with different capHeights; all controlled by player 1.
    pos_high = Position(x=0, y=0)   # capHeight 3
    pos_mid = Position(x=2, y=2)    # capHeight 2
    pos_low = Position(x=4, y=4)    # capHeight 1 (smallest positive)

    board.stacks[pos_high.to_key()] = RingStack(
        position=pos_high,
        rings=[1, 1, 1],
        stackHeight=3,
        capHeight=3,
        controllingPlayer=1,
    )
    board.stacks[pos_mid.to_key()] = RingStack(
        position=pos_mid,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    board.stacks[pos_low.to_key()] = RingStack(
        position=pos_low,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    # Trap all stacks by collapsing every other position on the board; this
    # prevents any legal movement or capture for player 1.
    _block_all_positions_except(state, [pos_high, pos_mid, pos_low])

    # Ensure global engine caches don't leak between tests.
    GameEngine.clear_cache()

    return state, pos_low, [pos_high, pos_mid, pos_low]


def _build_all_zero_caps_forced_elimination_state() -> tuple[
    "GameState",
    Position,
    list[Position],
]:
    """Blocked state where P1 has stacks whose capHeight metadata is zero.

    This models the degenerate situation covered by the TS tests where
    capHeight metadata is missing or zero but stackHeight is positive; forced
    elimination should still be available and should fall back to the first
    candidate stack. We only assert on the reported eliminatedCount, not on
    total_rings_eliminated, to mirror TS semantics for such legacy shapes.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    # Use FORCED_ELIMINATION phase per RR-CANON-R070 - FE is its own phase
    state.current_phase = GamePhase.FORCED_ELIMINATION
    state.current_player = 1

    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()

    first_pos = Position(x=1, y=1)
    second_pos = Position(x=5, y=5)

    # Deliberately inconsistent capHeight metadata (0 despite non-empty stack).
    board.stacks[first_pos.to_key()] = RingStack(
        position=first_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=0,
        controllingPlayer=1,
    )
    board.stacks[second_pos.to_key()] = RingStack(
        position=second_pos,
        rings=[1, 1, 1],
        stackHeight=3,
        capHeight=0,
        controllingPlayer=1,
    )

    _block_all_positions_except(state, [first_pos, second_pos])
    GameEngine.clear_cache()

    return state, first_pos, [first_pos, second_pos]


def test_forced_elimination_mixed_caps_enumeration_and_selection() -> None:
    """Positive FE case A: mixed capHeights with no other actions.

    - has_forced_elimination_action is True.
    - Enumeration covers all stacks with correct (capHeight, stackHeight).
    - get_valid_moves surfaces only FORCED_ELIMINATION moves.
    - apply_forced_elimination_for_player chooses smallest positive capHeight
      and increases total_rings_eliminated.
    """
    state, expected_pos, stack_positions = (
        _build_mixed_caps_forced_elimination_state()
    )
    player = state.current_player

    assert ga.has_forced_elimination_action(state, player) is True

    # Enumeration via the engine helper mirrors TS
    # enumerateForcedEliminationOptions.
    options = _enumerate_forced_elimination_options(state, player)
    expected_keys = {p.to_key() for p in stack_positions}
    assert set(options.keys()) == expected_keys

    # Cap / stack heights in options must match board metadata.
    for key in expected_keys:
        stack = state.board.stacks[key]
        cap_height, stack_height = options[key]
        assert cap_height == stack.cap_height
        assert stack_height == stack.stack_height

    # get_valid_moves should surface FORCED_ELIMINATION actions (plus no-action markers).
    # Filter out no-action bookkeeping moves to check only FE moves.
    all_moves = GameEngine.get_valid_moves(state, player)
    moves = [m for m in all_moves if m.type not in (
        MoveType.NO_MOVEMENT_ACTION,
        MoveType.NO_PLACEMENT_ACTION,
    )]
    assert moves, "Expected forced-elimination moves for blocked player"
    assert all(m.type == MoveType.FORCED_ELIMINATION for m in moves)

    move_targets = {m.to.to_key() for m in moves if m.to is not None}
    assert move_targets == expected_keys

    # apply_forced_elimination_for_player auto-selects the smallest
    # positive capHeight.
    before_total = state.total_rings_eliminated
    outcome = ga.apply_forced_elimination_for_player(state, player)
    assert outcome is not None
    assert outcome.eliminated_from == expected_pos
    assert outcome.eliminated_count >= 1
    assert state.total_rings_eliminated >= before_total + 1


def test_forced_elimination_all_caps_zero_falls_back_to_first_stack() -> None:
    """Positive FE case B: all caps zero; selection falls back to first stack.

    In this degenerate metadata case we only assert that:

    - forced elimination is available;
    - the first candidate stack is chosen; and
    - outcome.eliminated_count is at least 1.
    """
    state, expected_pos, stack_positions = (
        _build_all_zero_caps_forced_elimination_state()
    )
    player = state.current_player

    assert ga.has_forced_elimination_action(state, player) is True

    options = _enumerate_forced_elimination_options(state, player)
    expected_keys = {p.to_key() for p in stack_positions}
    assert set(options.keys()) == expected_keys
    assert all(cap == 0 for cap, _ in options.values())

    outcome = ga.apply_forced_elimination_for_player(state, player)
    assert outcome is not None
    assert outcome.eliminated_from == expected_pos
    assert outcome.eliminated_count >= 1


def test_forced_elimination_not_available_when_global_actions_exist() -> None:
    """Negative FE case: stacks plus placements/moves ⇒ no FE."""
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    # Use MOVEMENT phase to check for normal moves
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    # Player 1 has rings in hand and a central stack that can move or place.
    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 3  # Placements available.

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()

    center = Position(x=3, y=3)
    board.stacks[center.to_key()] = RingStack(
        position=center,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )

    GameEngine.clear_cache()

    # Preconditions: player has turn material and at least one legal action.
    assert ga.has_turn_material(state, 1) is True
    moves = GameEngine.get_valid_moves(state, 1)
    # Should have real moves (not just no-action markers) since player has a movable stack
    real_moves = [m for m in moves if m.type not in (
        MoveType.NO_MOVEMENT_ACTION,
        MoveType.NO_PLACEMENT_ACTION,
    )]
    assert real_moves, "Expected at least one legal move"

    # Under R072/R100/R205, forced elimination must not be available here.
    assert ga.has_forced_elimination_action(state, 1) is False
    assert GameEngine._get_forced_elimination_moves(state, 1) == []


if __name__ == "__main__":
    pytest.main()