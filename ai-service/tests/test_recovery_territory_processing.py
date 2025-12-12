from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.game_engine import GameEngine  # noqa: E402
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
    Territory,
    TimeControl,
)


def _make_player(player_number: int, rings_in_hand: int = 0) -> Player:
    return Player(
        id=f"p{player_number}",
        username=f"P{player_number}",
        type="human",
        playerNumber=player_number,
        isReady=True,
        timeRemaining=600,
        aiDifficulty=None,
        ringsInHand=rings_in_hand,
        eliminatedRings=0,
        territorySpaces=0,
    )


def _make_time_control() -> TimeControl:
    return TimeControl(initialTime=600, increment=0, type="blitz")


def test_eliminate_rings_from_stack_recovery_context_extracts_buried_ring() -> None:
    """ELIMINATE_RINGS_FROM_STACK with eliminationContext=recovery extracts a buried ring.

    Mirrors TS EliminationAggregate('recovery') semantics:
    - remove exactly one buried ring belonging to the acting player
    - the stack need not be controlled by that player
    """
    now = datetime.now(timezone.utc)
    pos = Position(x=3, y=3)
    pos_key = pos.to_key()

    # Python stack rings are bottom -> top. P2 controls the stack, P1 has a buried ring.
    stack = RingStack(
        position=pos,
        rings=[1, 2, 2],
        stackHeight=3,
        capHeight=2,
        controllingPlayer=2,
    )

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={pos_key: stack},
        markers={},
        collapsedSpaces={},
        territories={},
        formedLines=[],
        eliminatedRings={},
    )

    state = GameState(  # type: ignore[call-arg]
        id="recovery-territory-elim",
        boardType=BoardType.SQUARE8,
        board=board,
        players=[_make_player(1, rings_in_hand=0), _make_player(2, rings_in_hand=0)],
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=[],
        timeControl=_make_time_control(),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=3,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
    )

    move = Move(  # type: ignore[call-arg]
        id="elim-recovery",
        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
        player=1,
        to=pos,
        eliminationContext="recovery",
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    next_state = GameEngine.apply_move(state, move)

    # P1's buried ring is removed; P2 still controls the stack.
    next_stack = next_state.board.stacks.get(pos_key)
    assert next_stack is not None
    assert next_stack.rings == [2, 2]
    assert next_stack.controlling_player == 2
    assert next_stack.stack_height == 2
    assert next_stack.cap_height == 2

    # Elimination is credited to the acting player.
    p1_after = next(p for p in next_state.players if p.player_number == 1)
    assert p1_after.eliminated_rings == 1
    assert next_state.total_rings_eliminated == 1


def test_territory_processing_moves_surface_recovery_elimination_targets() -> None:
    """After processing a territory region via a recovery slide, the self-elimination
    decision surface uses recovery extraction targets (RR-CANON-R114).
    """
    now = datetime.now(timezone.utc)
    region_space = Position(x=0, y=0)
    processed_region = Territory(
        spaces=[region_space],
        controlling_player=1,
        is_disconnected=True,
    )

    target_pos = Position(x=3, y=3)
    target_key = target_pos.to_key()

    # Recovery target: stack controlled by P2 but containing a buried ring of P1.
    target_stack = RingStack(
        position=target_pos,
        rings=[1, 2, 2],
        stackHeight=3,
        capHeight=2,
        controllingPlayer=2,
    )

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={target_key: target_stack},
        markers={},
        collapsedSpaces={},
        territories={},
        formedLines=[],
        eliminatedRings={},
    )

    recovery_slide = Move(  # type: ignore[call-arg]
        id="recovery-slide",
        type=MoveType.RECOVERY_SLIDE,
        player=1,
        to=Position(x=1, y=1),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )
    processed_region_move = Move(  # type: ignore[call-arg]
        id="choose-territory",
        type=MoveType.CHOOSE_TERRITORY_OPTION,
        player=1,
        to=region_space,
        disconnectedRegions=(processed_region,),
        timestamp=now,
        thinkTime=0,
        moveNumber=2,
    )

    state = GameState(  # type: ignore[call-arg]
        id="recovery-territory-pending-elim",
        boardType=BoardType.SQUARE8,
        board=board,
        players=[_make_player(1, rings_in_hand=0), _make_player(2, rings_in_hand=0)],
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=[recovery_slide, processed_region_move],
        timeControl=_make_time_control(),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=3,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
    )

    GameEngine.clear_cache()
    moves = GameEngine.get_valid_moves(state, 1)
    assert moves, "Expected recovery-context territory elimination moves"
    assert all(m.type == MoveType.ELIMINATE_RINGS_FROM_STACK for m in moves)
    assert all(getattr(m, "elimination_context", None) == "recovery" for m in moves)
    assert any(m.to and m.to.to_key() == target_key for m in moves)


def test_recovery_context_region_prerequisite_allows_buried_ring_outside_region() -> None:
    """RR-CANON-R114: recovery-context territory regions are processable when a buried-ring
    extraction target exists outside the region (even if not controlled).
    """
    now = datetime.now(timezone.utc)
    region_space = Position(x=0, y=0)
    region = Territory(spaces=[region_space], controlling_player=1, is_disconnected=True)

    outside_stack_pos = Position(x=3, y=3)
    outside_stack_key = outside_stack_pos.to_key()
    outside_stack = RingStack(
        position=outside_stack_pos,
        rings=[1, 2, 2],
        stackHeight=3,
        capHeight=2,
        controllingPlayer=2,
    )

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={outside_stack_key: outside_stack},
        markers={},
        collapsedSpaces={},
        territories={},
        formedLines=[],
        eliminatedRings={},
    )

    recovery_slide = Move(  # type: ignore[call-arg]
        id="recovery-slide",
        type=MoveType.RECOVERY_SLIDE,
        player=1,
        to=Position(x=1, y=1),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    state = GameState(  # type: ignore[call-arg]
        id="recovery-territory-prereq",
        boardType=BoardType.SQUARE8,
        board=board,
        players=[_make_player(1, rings_in_hand=0), _make_player(2, rings_in_hand=0)],
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=[recovery_slide],
        timeControl=_make_time_control(),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=3,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
    )

    assert GameEngine._can_process_disconnected_region(state, region, 1)
