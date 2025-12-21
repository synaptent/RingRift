"""
Regression tests for recovery slide (RR-CANON-R110–R115).

This suite focuses on two invariants that are critical for canonical
self-play and replay parity:

1) Zobrist hash must remain consistent after RECOVERY_SLIDE so move caching
   cannot return stale movement surfaces.
2) Buried-ring extraction cost must increment elimination counters so the
   S-invariant and victory thresholds remain TS-aligned.
"""

import os
import sys
from datetime import datetime

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.core.zobrist import ZobristHash
from app.game_engine import GameEngine
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)


def test_recovery_stack_strike_updates_elimination_and_zobrist_hash() -> None:
    fixed_ts = datetime(2020, 1, 1, 0, 0, 0)

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            # Destination stack to strike: single ring, should be removed by strike.
            "2,2": RingStack(
                position=Position(x=2, y=2),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            ),
            # Opponent-controlled stack containing a buried ring for player 2 (cost extraction).
            "0,0": RingStack(
                position=Position(x=0, y=0),
                rings=[2, 1],  # bottom->top; player 2 ring is buried
                stackHeight=2,
                capHeight=1,
                controllingPlayer=1,
            ),
        },
        markers={
            # Player 2 marker adjacent to the destination stack.
            "2,1": MarkerInfo(
                player=2,
                position=Position(x=2, y=1),
                type="regular",
            )
        },
        collapsedSpaces={},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )

    players = [
        Player(
            id="p1",
            username="P1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="P2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    state = GameState(
        id="recovery-stack-strike-hash",
        boardType=BoardType.SQUARE8,
        rngSeed=1,
        board=board,
        players=players,
        currentPhase=GamePhase.MOVEMENT,
        currentPlayer=2,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="rapid"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=fixed_ts,
        lastMoveAt=fixed_ts,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        mustMoveFromStackKey=None,
        chainCaptureState=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )

    recovery = Move(  # type: ignore[call-arg]
        **{
            "id": "recovery-strike",
            "type": MoveType.RECOVERY_SLIDE,
            "player": 2,
            "from": Position(x=2, y=1),
            "to": Position(x=2, y=2),
            "recoveryMode": "stack_strike",
            "extractionStacks": ("0,0",),
            "timestamp": fixed_ts,
            "thinkTime": 0,
            "moveNumber": 1,
        }
    )

    next_state = GameEngine.apply_move(state, recovery, trace_mode=True)

    # Marker is consumed in stack_strike mode.
    assert "2,1" not in next_state.board.markers
    # Target stack is struck and removed (single ring).
    assert "2,2" not in next_state.board.stacks
    # Cost extraction removes the buried ring of player 2 from the extraction stack.
    assert next_state.board.stacks["0,0"].rings == [1]

    # Elimination accounting:
    # - +1 for the struck ring (credited to player 2)
    # - +1 for the extracted buried ring cost (self-elimination, credited to player 2)
    assert next_state.total_rings_eliminated == 2
    assert next_state.board.eliminated_rings.get("2") == 2
    assert next(p for p in next_state.players if p.player_number == 2).eliminated_rings == 2

    # Zobrist hash must match full recomputation after recovery.
    assert next_state.zobrist_hash == ZobristHash().compute_initial_hash(next_state)

