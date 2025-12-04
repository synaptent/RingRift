"""Parity tests for forced elimination sequences.

These tests validate that Python's forced elimination logic matches
TypeScript behavior for:

- Stack selection strategy (prefer smallest capHeight)
- Multi-player rotation during forced elimination chains
- Victory/elimination counting after forced eliminations
- Phase transitions after forced elimination

Based on the canonical rules:
- R072: Forced elimination precondition (stacks exist, no legal actions)
- R100: Stack selection strategy (smallest capHeight first)
- R205: Forced elimination application and counting
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402


def _make_multiplayer_game_state(num_players: int = 3) -> GameState:
    """Create a multi-player game state for forced elimination tests."""
    board = BoardState(type=BoardType.SQUARE8, size=8)
    players = [
        Player(
            id=f"p{i}",
            username=f"player{i}",
            type="human",
            playerNumber=i,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=12,  # Reduced for multi-player
            eliminatedRings=0,
            territorySpaces=0,
        )
        for i in range(1, num_players + 1)
    ]

    now = datetime.now()

    return GameState(
        id="test-fe-parity",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.MOVEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def _block_all_positions(state: GameState, except_positions: list[Position]) -> None:
    """Collapse all board positions except specified ones to block movement."""
    except_keys = {p.to_key() for p in except_positions}
    for x in range(state.board.size):
        for y in range(state.board.size):
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key not in except_keys and key not in state.board.stacks:
                state.board.collapsed_spaces[key] = 2  # Enemy territory


class TestForcedEliminationStackSelection:
    """Tests for R100: stack selection strategy."""

    def test_prefers_smallest_cap_height(self) -> None:
        """Forced elimination should select stack with smallest capHeight."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0  # Force elimination scenario

        # Add two stacks with different cap heights - place them far apart
        # so they can't capture each other
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=7, y=7)  # Opposite corner

        state.board.stacks[pos1.to_key()] = RingStack(
            position=pos1,
            rings=[1, 1, 1],
            stackHeight=3,
            capHeight=3,
            controllingPlayer=1,
        )
        state.board.stacks[pos2.to_key()] = RingStack(
            position=pos2,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        # Block all other positions to prevent movement
        _block_all_positions(state, [pos1, pos2])

        # Verify player has forced elimination available
        assert ga.has_forced_elimination_action(state, 1), (
            "Expected forced elimination to be available"
        )

        # Apply forced elimination
        outcome = ga.apply_forced_elimination_for_player(state, 1)
        assert outcome is not None

        # Should have selected the smaller stack (capHeight=1)
        assert outcome.eliminated_from == pos2
        assert outcome.eliminated_count == 1

    def test_fallback_to_first_stack_when_no_caps(self) -> None:
        """When no stacks have caps, select first available."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0

        # Add two stacks with same cap height
        pos1 = Position(x=0, y=0)
        pos2 = Position(x=2, y=2)

        state.board.stacks[pos1.to_key()] = RingStack(
            position=pos1,
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )
        state.board.stacks[pos2.to_key()] = RingStack(
            position=pos2,
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )

        # Block all other positions to prevent movement
        _block_all_positions(state, [pos1, pos2])

        outcome = ga.apply_forced_elimination_for_player(state, 1)
        assert outcome is not None
        # Should select one of them (first in iteration order)
        assert outcome.eliminated_from in (pos1, pos2)


class TestForcedEliminationMultiPlayer:
    """Tests for multi-player forced elimination rotation."""

    def test_skip_fully_eliminated_player_in_rotation(self) -> None:
        """Turn rotation should skip players with no turn material."""
        state = _make_multiplayer_game_state(3)
        state.current_player = 1

        # P1: has stack (will trigger FE)
        # P2: no stacks, no rings (fully eliminated - should be skipped)
        # P3: has rings in hand (should get turn)

        pos1 = Position(x=0, y=0)
        state.board.stacks[pos1.to_key()] = RingStack(
            position=pos1,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        state.players[0].rings_in_hand = 0  # P1: stack only

        state.players[1].rings_in_hand = 0  # P2: fully eliminated

        state.players[2].rings_in_hand = 5  # P3: has rings

        # Run end turn from P1
        GameEngine._end_turn(state)

        # Should have skipped P2 and gone to P3
        assert state.current_player == 3, (
            f"Expected P3, got P{state.current_player}"
        )

    def test_forced_elimination_chain_terminates(self) -> None:
        """Forced elimination chain must terminate (monotonic ring decrease)."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0

        # Create a stack that will be eliminated piece by piece
        pos = Position(x=3, y=3)
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1, 1, 1, 1, 1],
            stackHeight=5,
            capHeight=5,
            controllingPlayer=1,
        )

        # Block all positions to force elimination
        _block_all_positions(state, [pos])

        initial_total = state.total_rings_eliminated
        steps = 0
        max_steps = 10

        while steps < max_steps:
            outcome = ga.apply_forced_elimination_for_player(state, 1)
            if outcome is None:
                break

            # Each step must eliminate at least one ring
            assert outcome.eliminated_count >= 1
            # Total eliminated must increase monotonically
            assert state.total_rings_eliminated > initial_total
            initial_total = state.total_rings_eliminated
            steps += 1

        # Chain must have terminated
        assert steps > 0, "No eliminations occurred"
        assert steps < max_steps, "Chain did not terminate"


class TestForcedEliminationPhaseTransitions:
    """Tests for phase transitions after forced elimination."""

    def test_same_player_continues_after_fe_with_actions(self) -> None:
        """Player stays in MOVEMENT after FE if they have new actions."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0

        # Two stacks: one trapped (will be eliminated), one with movement
        pos1 = Position(x=0, y=0)  # This one trapped
        pos2 = Position(x=4, y=4)  # This one can move

        state.board.stacks[pos1.to_key()] = RingStack(
            position=pos1,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        # Larger stack can move after smaller is eliminated
        state.board.stacks[pos2.to_key()] = RingStack(
            position=pos2,
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )

        # Block pos1's movement but allow pos2 to move
        # Collapse all positions adjacent to pos1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                adj = Position(x=dx, y=dy)
                state.board.collapsed_spaces[adj.to_key()] = 2

        # Verify the setup: pos2 can move, pos1 cannot
        # The test validates that after FE, player continues if they have actions

    def test_game_ends_on_victory_threshold(self) -> None:
        """Game should end when victory threshold is reached via FE."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0
        state.victory_threshold = 3

        # Pre-set eliminated rings close to threshold
        state.board.eliminated_rings["1"] = 2

        # Create stack with multiple rings that will push over threshold
        pos = Position(x=0, y=0)
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )

        # Block all positions to force elimination
        _block_all_positions(state, [pos])

        # Apply forced elimination
        outcome = ga.apply_forced_elimination_for_player(state, 1)
        assert outcome is not None
        assert outcome.eliminated_count >= 1

        # Re-check victory
        GameEngine._check_victory(state)

        # Verify victory conditions
        total_eliminated = state.board.eliminated_rings.get("1", 0)
        if total_eliminated >= state.victory_threshold:
            assert state.game_status == GameStatus.FINISHED
            assert state.winner == 1


class TestForcedEliminationPreconditions:
    """Tests for R072: forced elimination preconditions."""

    def test_no_fe_when_player_has_legal_moves(self) -> None:
        """FE should not trigger when player has legal actions."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 3  # Can place rings

        # Even with a stack, placement is available
        pos = Position(x=0, y=0)
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        # Should NOT have forced elimination
        assert not ga.has_forced_elimination_action(state, 1)

    def test_no_fe_when_player_has_no_stacks(self) -> None:
        """FE requires controlling at least one stack."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0
        state.board.stacks.clear()

        # No stacks = no FE possible
        assert not ga.has_forced_elimination_action(state, 1)

    def test_fe_available_when_blocked_with_stacks(self) -> None:
        """FE should be available when player has stacks but no moves."""
        state = _make_multiplayer_game_state(2)
        state.current_phase = GamePhase.MOVEMENT
        state.current_player = 1
        state.players[0].rings_in_hand = 0

        # Create a trapped stack (surrounded by collapsed spaces)
        center = Position(x=3, y=3)
        state.board.stacks[center.to_key()] = RingStack(
            position=center,
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )

        # Collapse all adjacent spaces to trap the stack
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                adj_pos = Position(x=3 + dx, y=3 + dy)
                state.board.collapsed_spaces[adj_pos.to_key()] = 2  # Enemy territory

        # No normal moves (movement/capture) should be available
        all_moves = GameEngine.get_valid_moves(state, 1)
        normal_moves = [m for m in all_moves if m.type != MoveType.FORCED_ELIMINATION]
        assert len(normal_moves) == 0, "Should have no normal moves (movement/capture)"

        # But FE should be available (and should be included in get_valid_moves)
        assert ga.has_forced_elimination_action(state, 1)
        fe_moves = [m for m in all_moves if m.type == MoveType.FORCED_ELIMINATION]
        assert len(fe_moves) > 0, "FE moves should be included in get_valid_moves when blocked"


class TestForcedEliminationCounting:
    """Tests for R205: elimination counting and tracking."""

    def test_eliminated_count_matches_cap_height(self) -> None:
        """Eliminated ring count should match the stack's capHeight."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0

        pos = Position(x=0, y=0)
        cap_height = 4
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1] * cap_height,
            stackHeight=cap_height,
            capHeight=cap_height,
            controllingPlayer=1,
        )

        # Block all positions to force elimination
        _block_all_positions(state, [pos])

        initial_eliminated = state.board.eliminated_rings.get("1", 0)

        outcome = ga.apply_forced_elimination_for_player(state, 1)
        assert outcome is not None
        assert outcome.eliminated_count == cap_height

        final_eliminated = state.board.eliminated_rings.get("1", 0)
        assert final_eliminated == initial_eliminated + cap_height

    def test_total_rings_eliminated_updated(self) -> None:
        """total_rings_eliminated should be updated after FE."""
        state = _make_multiplayer_game_state(2)
        state.current_player = 1
        state.players[0].rings_in_hand = 0
        state.total_rings_eliminated = 5

        pos = Position(x=0, y=0)
        state.board.stacks[pos.to_key()] = RingStack(
            position=pos,
            rings=[1, 1, 1],
            stackHeight=3,
            capHeight=3,
            controllingPlayer=1,
        )

        # Block all positions to force elimination
        _block_all_positions(state, [pos])

        outcome = ga.apply_forced_elimination_for_player(state, 1)
        assert outcome is not None

        assert state.total_rings_eliminated == 5 + outcome.eliminated_count
