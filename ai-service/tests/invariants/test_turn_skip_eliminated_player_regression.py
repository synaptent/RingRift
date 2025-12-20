from __future__ import annotations

"""Regression for turn rotation with recovery-eligible players.

This test documents the scenario captured in
``ai-service/logs/selfplay/failures/failure_120_no_legal_moves_for_current_player.json``.

In that snapshot, after Player 2's move_stack from (5,5) to (4,6), the Python
GameEngine had rotated the turn to Player 1 who has:

- **no stacks** (all rings buried in opponent stacks)
- **no rings in hand**

Under the revised RingRift rules (RR-CANON-R201), players are only skipped if
they have NO RINGS ANYWHERE (not controlled, not buried, not in hand). A player
with buried rings may be recovery-eligible (RR-CANON-R110) and therefore must
NOT be skipped - they should receive turns with recovery moves available.

This test validates that:
1. Players with buried rings (but no turn-material) still receive turns
2. Such players get NO_MOVEMENT_ACTION or recovery moves in their MOVEMENT phase
3. Only players with NO rings anywhere are skipped during turn rotation
"""

import json
import os
import sys
from typing import Dict, Any, Optional

import pytest

# Ensure app package is importable when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (
    GameState, Move, Position, MoveType, GamePhase, GameStatus, BoardType,
    BoardState, Player, RingStack, TimeControl
)
from app.game_engine import GameEngine
from app.board_manager import BoardManager
from app.rules import global_actions as ga


FAILURE_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "logs",
    "selfplay",
    "failures",
    "failure_120_no_legal_moves_for_current_player.json",
)


def describe_state(state: GameState, label: str = "") -> dict[str, Any]:
    """Return diagnostic info for a game state."""
    p1 = next((p for p in state.players if p.player_number == 1), None)
    p2 = next((p for p in state.players if p.player_number == 2), None)

    p1_stacks = [k for k, s in state.board.stacks.items() if s.controlling_player == 1]
    p2_stacks = [k for k, s in state.board.stacks.items() if s.controlling_player == 2]

    return {
        "label": label,
        "current_player": state.current_player,
        "current_phase": state.current_phase.value if state.current_phase else None,
        "game_status": state.game_status.value if state.game_status else None,
        "p1_rings_in_hand": p1.rings_in_hand if p1 else None,
        "p1_stacks": len(p1_stacks),
        "p1_eliminated": p1.eliminated_rings if p1 else None,
        "p2_rings_in_hand": p2.rings_in_hand if p2 else None,
        "p2_stacks": len(p2_stacks),
        "p2_eliminated": p2.eliminated_rings if p2 else None,
        "p1_has_turn_material": ga.has_turn_material(state, 1) if p1 else None,
        "p2_has_turn_material": ga.has_turn_material(state, 2) if p2 else None,
    }


@pytest.mark.skipif(
    not os.path.exists(FAILURE_SNAPSHOT_PATH),
    reason="Failure snapshot not found; run self-play soak to regenerate",
)
def test_turn_rotation_keeps_recovery_eligible_player_from_snapshot() -> None:
    """Load the failure snapshot and verify recovery-eligible behavior.

    This test validates the revised rules (RR-CANON-R201):
    - The snapshot shows current_player=1 in MOVEMENT phase
    - Player 1 has no stacks and no rings in hand
    - Under revised rules, if P1 has buried rings, they are NOT skipped

    After calling _end_turn, P1 should NOT be skipped if they have buried rings.
    """
    with open(FAILURE_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state = GameState.model_validate(payload["state"])

    diag = describe_state(state, "snapshot")
    print(f"\nSnapshot state diagnostics: {json.dumps(diag, indent=2)}")

    assert state.game_status == GameStatus.ACTIVE, "Game should be ACTIVE"
    assert state.current_player == 1, "current_player is 1"

    # Verify Player 1 has no material (no stacks, no rings in hand)
    p1 = next(p for p in state.players if p.player_number == 1)
    p1_stacks = [s for s in state.board.stacks.values() if s.controlling_player == 1]
    assert len(p1_stacks) == 0, "P1 should have no stacks"
    assert p1.rings_in_hand == 0, "P1 should have no rings in hand"

    # Check if P1 has buried rings
    p1_has_buried_rings = any(
        1 in stack.rings
        for stack in state.board.stacks.values()
        if stack.controlling_player != 1
    )

    test_state = state.model_copy(deep=True)
    GameEngine._end_turn(test_state)

    result_diag = describe_state(test_state, "after_end_turn")
    print(f"After _end_turn: {json.dumps(result_diag, indent=2)}")

    if p1_has_buried_rings:
        # Per RR-CANON-R201: P1 has rings (buried), so should NOT be skipped
        # They get a turn with NO_MOVEMENT_ACTION or recovery moves
        assert test_state.current_player == 1, (
            f"P1 has buried rings and should NOT be skipped, got P{test_state.current_player}"
        )
    else:
        # P1 has no rings anywhere (permanently eliminated), should be skipped
        assert test_state.current_player == 2, (
            f"P1 is permanently eliminated and should be skipped, got P{test_state.current_player}"
        )


def test_turn_rotation_keeps_player_with_buried_rings_synthetic() -> None:
    """Synthetic test: verify turn rotation does NOT skip players with buried rings.

    This test creates a synthetic state where:
    - Player 1 has no stacks and no rings in hand BUT has buried rings
    - Player 2 has stacks and rings in hand
    - When rotating from P2's TERRITORY_PROCESSING, P1 should receive the turn (NOT skipped)

    Per RR-CANON-R201, players are only skipped if they have NO rings anywhere.
    """
    from datetime import datetime

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "3,3": RingStack(
                position=Position(x=3, y=3),
                controlling_player=2,
                rings=[2, 2],
                stack_height=2,
                cap_height=2,
            ),
            "4,4": RingStack(
                position=Position(x=4, y=4),
                controlling_player=2,
                rings=[2, 1, 2],  # P1 ring buried in middle
                stack_height=3,
                cap_height=2,
            ),
        },
        markers={},
        collapsed_spaces={},
        eliminated_rings={"1": 2, "2": 1},
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id="p1",
            username="Player 1",
            type="ai",
            player_number=1,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=0,  # No rings in hand
            eliminated_rings=2,
            territory_spaces=0,
        ),
        Player(
            id="p2",
            username="Player 2",
            type="ai",
            player_number=2,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=10,  # Has rings in hand
            eliminated_rings=1,
            territory_spaces=0,
        ),
    ]

    time_control = TimeControl(initial_time=600000, increment=0, type="standard")
    now = datetime.now()

    # Set up P2's turn in TERRITORY_PROCESSING - next _end_turn will rotate to P1
    state = GameState(
        id="synthetic-turn-skip-test",
        board_type=BoardType.SQUARE8,
        rng_seed=42,
        board=board,
        players=players,
        current_phase=GamePhase.TERRITORY_PROCESSING,  # P2's turn ending
        current_player=2,  # P2 is finishing their turn
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=2,
        total_rings_in_play=18 * 2,
        total_rings_eliminated=3,
        victory_threshold=5,
        territory_victory_threshold=15,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
    )

    # Verify preconditions for P1
    p1_stacks = BoardManager.get_player_stacks(state.board, 1)
    assert len(p1_stacks) == 0, "P1 should have no stacks"
    assert players[0].rings_in_hand == 0, "P1 should have no rings in hand"

    # Verify P1 has buried rings
    p1_buried = any(
        1 in stack.rings
        for stack in state.board.stacks.values()
        if stack.controlling_player != 1
    )
    assert p1_buried, "P1 should have buried rings for this test"

    # Call _end_turn from P2's TERRITORY_PROCESSING - should rotate to P1
    test_state = state.model_copy(deep=True)

    GameEngine._end_turn(test_state)

    # Per RR-CANON-R201: P1 has rings (buried), should NOT be skipped
    # P1 should be the current player after rotation
    assert test_state.current_player == 1, (
        f"P1 has buried rings and should NOT be skipped. "
        f"After rotating from P2's turn, got P{test_state.current_player}"
    )

    # Per RR-CANON-R073: ALL players start in RING_PLACEMENT phase without exception.
    # Players with ringsInHand == 0 will emit no_placement_action and proceed to movement,
    # but they MUST enter ring_placement first - no phase skipping is allowed.
    assert test_state.current_phase == GamePhase.RING_PLACEMENT, (
        f"P1 MUST start in RING_PLACEMENT phase (per RR-CANON-R073 - no phase skipping). "
        f"Got {test_state.current_phase}"
    )


def test_turn_rotation_skips_permanently_eliminated_player_synthetic() -> None:
    """Synthetic test: verify turn rotation DOES skip permanently eliminated players.

    This test creates a synthetic state where:
    - Player 1 has no stacks, no rings in hand, and NO buried rings
    - Player 2 has stacks and rings in hand
    - After calling _end_turn from P1, turn should go to P2

    Per RR-CANON-R201, players with no rings anywhere are permanently eliminated
    and must be skipped.
    """
    from datetime import datetime

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "3,3": RingStack(
                position=Position(x=3, y=3),
                controlling_player=2,
                rings=[2, 2],  # Only P2 rings
                stack_height=2,
                cap_height=2,
            ),
            "4,4": RingStack(
                position=Position(x=4, y=4),
                controlling_player=2,
                rings=[2, 2, 2],  # Only P2 rings - NO P1 buried rings
                stack_height=3,
                cap_height=2,
            ),
        },
        markers={},
        collapsed_spaces={},
        eliminated_rings={"1": 5, "2": 1},  # P1 has 5 eliminated (none anywhere else)
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id="p1",
            username="Player 1",
            type="ai",
            player_number=1,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=0,  # No rings in hand
            eliminated_rings=5,
            territory_spaces=0,
        ),
        Player(
            id="p2",
            username="Player 2",
            type="ai",
            player_number=2,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=10,
            eliminated_rings=1,
            territory_spaces=0,
        ),
    ]

    time_control = TimeControl(initial_time=600000, increment=0, type="standard")
    now = datetime.now()

    state = GameState(
        id="synthetic-permanent-elimination-test",
        board_type=BoardType.SQUARE8,
        rng_seed=42,
        board=board,
        players=players,
        current_phase=GamePhase.MOVEMENT,
        current_player=1,  # P1 is current but permanently eliminated
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=2,
        total_rings_in_play=18 * 2,
        total_rings_eliminated=6,
        victory_threshold=5,
        territory_victory_threshold=15,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
    )

    # Verify preconditions
    p1_stacks = BoardManager.get_player_stacks(state.board, 1)
    assert len(p1_stacks) == 0, "P1 should have no stacks"
    assert players[0].rings_in_hand == 0, "P1 should have no rings in hand"

    # Verify P1 has NO buried rings (permanently eliminated)
    p1_has_any_ring = any(
        1 in stack.rings
        for stack in state.board.stacks.values()
    )
    assert not p1_has_any_ring, "P1 should have NO rings anywhere for this test"

    # Call _end_turn - P1 should be skipped (permanently eliminated)
    test_state = state.model_copy(deep=True)
    initial_player = test_state.current_player

    GameEngine._end_turn(test_state)

    # Per RR-CANON-R201: P1 has NO rings anywhere, should be skipped
    assert test_state.current_player == 2, (
        f"P1 is permanently eliminated (no rings) and should be skipped. "
        f"Initial: {initial_player}, Final: {test_state.current_player}"
    )

    # P2 should start in RING_PLACEMENT since they have rings in hand
    assert test_state.current_phase == GamePhase.RING_PLACEMENT, (
        f"P2 should start in RING_PLACEMENT, got {test_state.current_phase}"
    )

    # Verify P2 has valid moves
    p2_moves = GameEngine.get_valid_moves(test_state, 2)
    assert len(p2_moves) > 0, "P2 should have valid moves"


# ARCHIVED TEST: test_fully_eliminated_player_not_left_as_current
# Removed 2025-12-07
#
# This test expected _end_turn to skip players without turn material, but the
# current (correct) behavior is that fully-eliminated players are NOT skipped.
# Per the GameEngine._end_turn() documentation: "Do not skip fully-eliminated
# players; even seats with no stacks and no rings in hand must still traverse
# all phases and record no-action moves."
#
# The invariant being tested (ACTIVE state should never have a fully eliminated
# current_player) is not enforced by _end_turn - instead, hosts must emit
# NO_PLACEMENT_ACTION and other bookkeeping moves for players without material.
# The _assert_active_player_has_legal_action invariant accounts for this by
# checking for phase requirements (bookkeeping moves) in addition to interactive
# moves.
