import sys
import os
from datetime import datetime
from typing import Optional

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.models import (
    GameState,
    BoardType,
    BoardState,
    GamePhase,
    GameStatus,
    TimeControl,
    Player,
    Position,
    RingStack,
    MarkerInfo,
    Move,
    MoveType,
)
from app.game_engine import GameEngine

def create_base_state(
    board_type: BoardType = BoardType.SQUARE8,
    phase: GamePhase = GamePhase.MOVEMENT,
) -> GameState:
    """Create a base game state for chain capture testing."""
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    else:
        size = 13  # Canonical hex: size=13, radius=12

    return GameState(
        id="chain-capture-parity",
        boardType=board_type,
        board=BoardState(type=board_type, size=size),
        players=[
            Player(
                id="p1",
                username="Player1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="Player2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
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
    z: int | None = None,
) -> None:
    """Helper to place a stack on the board."""
    pos = Position(x=x, y=y, z=z)
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

def create_setup_move(
    player: int,
    position: Position,
    move_type: MoveType = MoveType.PLACE_RING,
) -> Move:
    """Create a synthetic setup move for testing."""
    return Move(
        id="setup",
        type=move_type,
        player=player,
        to=position,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )

class TestChainCapturePhaseFix:
    def test_chain_capture_phase_transition(self) -> None:
        """
        Verify that after the first segment of a multi-segment capture,
        the game phase transitions to CHAIN_CAPTURE (not CAPTURE).
        """
        # Use SQUARE19 as per the failing case
        state = create_base_state(board_type=BoardType.SQUARE19)
        board = state.board

        # Setup a chain capture scenario:
        # P1 stack at (2,2) height 2
        # P2 stack at (2,3) height 1 (first target)
        # P2 stack at (2,5) height 1 (second target)
        # Empty cells at (2,4) and (2,6) for landing
        place_stack(board, 2, 2, player=1, height=2)
        place_stack(board, 2, 3, player=2, height=1)
        place_stack(board, 2, 5, player=2, height=1)

        # Add a synthetic placement move to enable capture detection
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Ensure we are in MOVEMENT or CAPTURE phase initially (depending on how we got here)
        # But GameEngine._get_capture_moves works regardless of phase if we call it directly.
        # Let's assume we are in MOVEMENT phase and about to capture.
        state.current_phase = GamePhase.MOVEMENT

        # Get initial capture moves
        initial_captures = GameEngine._get_capture_moves(state, 1)
        assert len(initial_captures) > 0, "Expected at least one capture move"

        # Find the capture over (2,3) landing at (2,4)
        first_capture = None
        for move in initial_captures:
            if (
                move.capture_target
                and move.capture_target.x == 2
                and move.capture_target.y == 3
                and move.to.x == 2
                and move.to.y == 4
            ):
                first_capture = move
                break

        assert first_capture is not None, "Expected capture over (2,3) to (2,4)"

        # Apply first capture
        state_after_first = GameEngine.apply_move(state, first_capture)

        # Verify phase is CHAIN_CAPTURE
        print(f"DEBUG: Phase after first capture: {state_after_first.current_phase}")
        assert state_after_first.current_phase == GamePhase.CHAIN_CAPTURE, (
            f"Expected phase CHAIN_CAPTURE, got {state_after_first.current_phase}"
        )

        # Verify we have continuation moves
        continuation_caps = GameEngine._get_capture_moves(state_after_first, 1)
        assert len(continuation_caps) > 0, "Expected continuation captures"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])