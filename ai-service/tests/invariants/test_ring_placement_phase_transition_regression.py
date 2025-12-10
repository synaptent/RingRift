from __future__ import annotations

"""Regression tests for RING_PLACEMENT phase transition when no rings in hand.

These tests validate the fix for the bug where games terminated with
"no_legal_moves_for_current_player" when the player was in RING_PLACEMENT
phase with 0 rings in hand but had stacks that could move/capture.

Per RingRift rules, placement is optional - when a player has stacks on the
board but cannot place (0 rings in hand or no valid placement spots), they
should automatically advance to movement. The Python engine must expose
movement/capture moves in this situation.

Failure scenario (prior to fix):
- Game phase: RING_PLACEMENT
- Player has 0 rings in hand
- Player has stacks on the board with valid movements
- get_valid_moves() returned [] because:
  - _get_ring_placement_moves() requires rings_in_hand > 0
  - _get_skip_placement_moves() requires rings_in_hand > 0
- Result: game terminated incorrectly

After fix:
- When phase == RING_PLACEMENT and no placement/skip moves available,
  get_valid_moves() checks for movement/capture moves and returns those.
"""

import os
import sys
from datetime import datetime
from typing import List

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


def create_ring_placement_zero_rings_state() -> GameState:
    """Create a state where player is in RING_PLACEMENT with 0 rings but has stacks."""
    # Player 1 has a stack at (3,3) that can move
    # Ensure there's space around it for movement
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "3,3": RingStack(
                position=Position(x=3, y=3),
                controlling_player=1,
                rings=[1],
                stack_height=1,
                cap_height=1,
            ),
            "6,6": RingStack(
                position=Position(x=6, y=6),
                controlling_player=2,
                rings=[2],
                stack_height=1,
                cap_height=1,
            ),
        },
        collapsed_spaces={},
        markers={},
        eliminated_rings={"1": 0, "2": 0},
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id="p1", username="P1", type="ai", player_number=1,
            is_ready=True, time_remaining=600000, ai_difficulty=5,
            rings_in_hand=0,  # Key: 0 rings in hand
            eliminated_rings=0, territory_spaces=0,
        ),
        Player(
            id="p2", username="P2", type="ai", player_number=2,
            is_ready=True, time_remaining=600000, ai_difficulty=5,
            rings_in_hand=10, eliminated_rings=0, territory_spaces=0,
        ),
    ]

    time_control = TimeControl(initial_time=600000, increment=0, type="standard")
    now = datetime.now()

    return GameState(
        id="ring-placement-test",
        board_type=BoardType.SQUARE8,
        rng_seed=42,
        board=board,
        players=players,
        current_phase=GamePhase.RING_PLACEMENT,  # Key: RING_PLACEMENT phase
        current_player=1,
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=2,
        total_rings_in_play=36,
        total_rings_eliminated=0,
        victory_threshold=18,  # RR-CANON-R061: ringsPerPlayer
        territory_victory_threshold=33,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
    )


class TestRingPlacementPhaseTransition:
    """Test that movement is available when placement isn't possible.

    ARCHIVED TEST: test_ring_placement_zero_rings_exposes_movement_moves
    - Removed 2025-12-07: This test expected get_valid_moves() to cross-phase
      return MOVEMENT moves when in RING_PLACEMENT phase with 0 rings. The
      current (correct) behavior is that get_valid_moves() returns phase-
      appropriate moves only. The phase machine handles transitions via
      NO_PLACEMENT_ACTION bookkeeping moves, not cross-phase fallbacks in
      get_valid_moves(). See GameEngine._end_turn() documentation.
    """

    def test_ring_placement_with_rings_still_exposes_placement(self):
        """When in RING_PLACEMENT with rings available, placement is exposed."""
        state = create_ring_placement_zero_rings_state()

        # Modify: give player 1 some rings
        for p in state.players:
            if p.player_number == 1:
                p.rings_in_hand = 5

        moves = GameEngine.get_valid_moves(state, 1)

        # Should have placement moves
        placement_moves = [m for m in moves if m.type == MoveType.PLACE_RING]
        assert len(placement_moves) > 0, (
            "Should have placement moves when rings_in_hand > 0"
        )

        # Should also have skip_placement (since player has stacks)
        skip_moves = [m for m in moves if m.type == MoveType.SKIP_PLACEMENT]
        assert len(skip_moves) > 0, (
            "Should have skip_placement when rings_in_hand > 0 and player has stacks"
        )

    def test_no_stacks_no_rings_returns_empty(self):
        """When player has no stacks and no rings, no moves available."""
        state = create_ring_placement_zero_rings_state()

        # Remove player 1's stack
        del state.board.stacks["3,3"]

        moves = GameEngine.get_valid_moves(state, 1)

        # With no stacks and no rings, no moves possible
        # (This might trigger forced elimination or termination check elsewhere)
        assert len(moves) == 0, (
            "Player with no stacks and no rings should have no moves"
        )


class TestRingPlacementInvariant:
    """Test the invariant: RING_PLACEMENT + stacks → must have some action.

    ARCHIVED TESTS:
    - test_invariant_ring_placement_with_stacks_has_action (removed 2025-12-07):
      Expected get_valid_moves() to cross-phase return movement moves. Current
      correct behavior uses phase-strict move generation with NO_PLACEMENT_ACTION
      bookkeeping moves for phase transitions.

    - test_various_rings_in_hand_values (removed 2025-12-07):
      Same root cause - tested cross-phase movement fallback which is not the
      current (correct) architecture.
    """

    pass  # All tests in this class were archived - class kept for documentation


class TestRingPlacementEdgeCases:
    """Test edge cases for RING_PLACEMENT phase transition.

    ARCHIVED TEST: test_blocked_stack_still_has_forced_elimination (removed 2025-12-07)
    - This test expected FORCED_ELIMINATION moves to be returned in RING_PLACEMENT
      phase. The current (correct) behavior is that FE moves are only surfaced in
      the dedicated FORCED_ELIMINATION phase. The phase machine transitions players
      to FORCED_ELIMINATION phase via GameEngine._end_turn() when appropriate.
      See RR-CANON-R070/R072/R100/R204 in the rules specification.
    """

    pass  # All tests in this class were archived - class kept for documentation
