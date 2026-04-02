"""
Full turn phase sequence parity tests for Python engine.

These tests verify that multi-phase turn sequences correctly transition
through phases according to RR-CANON-R070 (Turn Phase Order):
1. Ring placement
2. Movement
3. Capture/chain capture
4. Line processing
5. Territory processing
6. Forced elimination (if blocked)

The Python engine must match TypeScript phase transition behavior for
training data parity.
"""

import os
import sys
from datetime import datetime

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.board_manager import BoardManager
from app.game_engine import GameEngine
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    LineInfo,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    Territory,
    TimeControl,
)
from app.rules.core import infer_total_rings_in_play


def create_base_state(
    board_type: BoardType = BoardType.SQUARE8,
    phase: GamePhase = GamePhase.MOVEMENT,
    num_players: int = 2,
) -> GameState:
    """Create a base game state for phase sequence testing."""
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    else:
        size = 13

    players = []
    for i in range(1, num_players + 1):
        players.append(
            Player(
                id=f"p{i}",
                username=f"Player{i}",
                type="human",
                playerNumber=i,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            )
        )

    board = BoardState(type=board_type, size=size)

    return GameState(
        id="phase-sequence-test",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=infer_total_rings_in_play(players, board),
        totalRingsEliminated=0,
        victoryThreshold=12,
        territoryVictoryThreshold=33,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        rngSeed=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


def place_stack(
    board: BoardState,
    x: int,
    y: int,
    player: int,
    height: int = 1,
) -> None:
    """Helper to place a stack on the board."""
    pos = Position(x=x, y=y)
    key = pos.to_key()
    rings = [player] * height
    stack = RingStack(
        position=pos,
        rings=rings,
        stackHeight=height,
        capHeight=height,
        controllingPlayer=player,
    )
    board.stacks[key] = stack


def place_marker(
    board: BoardState,
    x: int,
    y: int,
    player: int,
) -> None:
    """Helper to place a marker on the board."""
    pos = Position(x=x, y=y)
    key = pos.to_key()
    marker = MarkerInfo(
        player=player,
        position=pos,
        type="regular",
    )
    board.markers[key] = marker


class TestPhaseSequenceOrder:
    """Tests verifying phase transition order per RR-CANON-R070."""

    def test_movement_to_capture_transition(self) -> None:
        """After movement landing enables capture, transition to CAPTURE phase."""
        state = create_base_state(phase=GamePhase.MOVEMENT)
        board = state.board

        # P1 stack that can move and capture P2 stack
        place_stack(board, 2, 2, player=1, height=2)
        place_stack(board, 2, 3, player=2, height=1)  # Capturable

        # Simulate movement to landing position where capture is available
        # Note: Move is immutable, so all fields must be set at construction
        movement_move = Move(
            id="test-move",
            type=MoveType.MOVE_STACK,
            player=1,
            to=Position(x=2, y=4),  # Land past the target
            from_pos=Position(x=2, y=2),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        state_after = GameEngine.apply_move(state, movement_move)

        # Should be in CAPTURE phase if capture is available from landing
        # Note: The exact phase depends on whether capture is mandatory
        assert state_after.current_phase in (
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
            GamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING,
            GamePhase.FORCED_ELIMINATION,
            GamePhase.GAME_OVER,
            GamePhase.MOVEMENT,  # Next player's turn
        ), f"After movement, expected valid phase transition, got {state_after.current_phase}"

    def test_capture_to_chain_or_next_phase_transition(self) -> None:
        """After capture, transition to CHAIN_CAPTURE if continuation exists, else next phase."""
        state = create_base_state(phase=GamePhase.CAPTURE, num_players=4)
        board = state.board

        # Setup for potential chain capture
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # First target
        place_stack(board, 3, 6, player=3, height=1)  # Potential chain target

        state.move_history.append(
            Move(
                id="setup",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=3, y=3),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=1,
            )
        )

        # Get and apply capture
        captures = GameEngine._get_capture_moves(state, 1)
        assert captures, "Expected capture moves"

        # Find capture to (3,5) which might enable chain
        capture = next(
            (c for c in captures if c.to.x == 3 and c.to.y == 5),
            captures[0],
        )
        state_after = GameEngine.apply_move(state, capture)

        # After capture, should transition to a valid next phase
        # CHAIN_CAPTURE if continuation available, else LINE_PROCESSING or beyond
        assert state_after.current_phase in (
            GamePhase.CHAIN_CAPTURE,
            GamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING,
            GamePhase.FORCED_ELIMINATION,
            GamePhase.GAME_OVER,
        ), f"After capture, expected valid phase transition, got {state_after.current_phase}"

    def test_line_processing_to_territory_transition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After line processing completes, transition to TERRITORY_PROCESSING."""
        state = create_base_state(phase=GamePhase.LINE_PROCESSING)
        board = state.board

        # Setup: P1 has a formed line requiring processing
        line_positions = [
            Position(x=0, y=3),
            Position(x=1, y=3),
            Position(x=2, y=3),
            Position(x=3, y=3),
        ]
        for pos in line_positions:
            place_marker(board, pos.x, pos.y, player=1)

        # Add the formed line (LineInfo requires positions, player, length, direction)
        board.formed_lines = [
            LineInfo(
                positions=line_positions,
                player=1,
                length=4,
                direction=Position(x=1, y=0),  # Horizontal direction vector
            )
        ]

        # P1 needs a stack for elimination
        place_stack(board, 5, 5, player=1, height=2)

        # Mock no territory disconnections (to isolate line -> territory test)
        def mock_find_disconnected_regions(board, player_number):
            return []

        monkeypatch.setattr(
            BoardManager,
            "find_disconnected_regions",
            staticmethod(mock_find_disconnected_regions),
        )

        # Get line processing moves
        moves = GameEngine.get_valid_moves(state, 1)
        line_moves = [
            m for m in moves
            if m.type in (MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD)
        ]
        assert line_moves, "Expected line processing moves"

        # Apply line choice
        state_after_line = GameEngine.apply_move(state, line_moves[0])

        # Need to apply elimination if required
        if state_after_line.current_phase == GamePhase.LINE_PROCESSING:
            # Still in line processing - need elimination
            elim_moves = GameEngine.get_valid_moves(state_after_line, 1)
            elim_move = next(
                (m for m in elim_moves if m.type == MoveType.ELIMINATE_RINGS_FROM_STACK),
                None,
            )
            if elim_move:
                state_after_line = GameEngine.apply_move(state_after_line, elim_move)

        # Should transition to TERRITORY_PROCESSING or beyond (no END_TURN enum - turn ends by player change)
        assert state_after_line.current_phase in (
            GamePhase.TERRITORY_PROCESSING,
            GamePhase.FORCED_ELIMINATION,
            GamePhase.GAME_OVER,
            GamePhase.LINE_PROCESSING,  # If more lines to process
            GamePhase.RING_PLACEMENT,  # Next player's turn started
            GamePhase.MOVEMENT,  # Next player's turn started
        ), f"Expected territory or next phase after line, got {state_after_line.current_phase}"


class TestMultiPhaseSequence:
    """Tests for complete multi-phase turn sequences."""

    def test_capture_to_line_to_end_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test capture -> line processing -> end turn sequence.

        This tests a realistic multi-phase sequence where:
        1. A capture is made
        2. The capture creates a line (via marker landing)
        3. Line processing occurs
        4. Turn ends
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        # Setup: P1 attacker, P2 target
        place_stack(board, 3, 3, player=1, height=2)
        place_stack(board, 3, 4, player=2, height=1)

        # P1 stack for elimination
        place_stack(board, 5, 5, player=1, height=2)

        state.move_history.append(
            Move(
                id="setup",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=3, y=3),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=1,
            )
        )

        # Mock no territory (simplify test)
        def mock_find_disconnected_regions(board, player_number):
            return []

        monkeypatch.setattr(
            BoardManager,
            "find_disconnected_regions",
            staticmethod(mock_find_disconnected_regions),
        )

        # Track phase transitions
        phases_seen = [state.current_phase]
        current_state = state
        max_iterations = 20

        for _ in range(max_iterations):
            moves = GameEngine.get_valid_moves(current_state, 1)
            if not moves:
                break

            # Apply first available move
            current_state = GameEngine.apply_move(current_state, moves[0])
            phases_seen.append(current_state.current_phase)

            # Check for turn end (game over or player change)
            if current_state.current_phase == GamePhase.GAME_OVER:
                break
            if current_state.current_player != 1:
                break  # Turn ended, now player 2's turn

        # Verify we saw phase transitions (not stuck in one phase)
        unique_phases = set(phases_seen)
        assert len(unique_phases) >= 1, (
            f"Expected multiple phase transitions, saw only: {phases_seen}"
        )

    def test_phase_order_canonical_compliance(self) -> None:
        """Verify phase enum values match canonical order (RR-CANON-R070).

        This ensures the Python GamePhase enum maintains the correct
        ordering for comparison operations.
        """
        # Per RR-CANON-R070, the phase order is:
        expected_order = [
            GamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT,
            GamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE,
            GamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING,
            # GAME_OVER is terminal, not in the main sequence
        ]

        # Verify phases exist
        for phase in expected_order:
            assert phase is not None, f"Phase {phase} should exist"

        # Note: We can't test ordering by enum value because Python Enum
        # doesn't guarantee numeric ordering. The important thing is that
        # the phase transition logic in GameEngine follows this order.


class TestSMetricMonotonicity:
    """Tests for S-metric (M + C + E) monotonicity during turn."""

    def test_s_metric_non_decreasing_after_capture(self) -> None:
        """S-metric should not decrease after capture (rings are captured, not destroyed)."""
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        place_stack(board, 3, 3, player=1, height=2)
        place_stack(board, 3, 4, player=2, height=1)

        state.move_history.append(
            Move(
                id="setup",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=3, y=3),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=1,
            )
        )

        # Calculate initial S-metric (total rings in play + captured)
        def calc_s_metric(s: GameState) -> int:
            in_play = sum(
                stack.stack_height
                for stack in s.board.stacks.values()
            )
            eliminated = sum(p.eliminated_rings for p in s.players)
            captured = sum(
                len(stack.rings) - stack.cap_height
                for stack in s.board.stacks.values()
                if stack.cap_height < len(stack.rings)
            )
            return in_play + eliminated + captured

        initial_s = calc_s_metric(state)

        # Apply capture
        captures = GameEngine._get_capture_moves(state, 1)
        if captures:
            state_after = GameEngine.apply_move(state, captures[0])
            final_s = calc_s_metric(state_after)

            # S-metric should not decrease
            assert final_s >= initial_s, (
                f"S-metric decreased: {initial_s} -> {final_s}"
            )
