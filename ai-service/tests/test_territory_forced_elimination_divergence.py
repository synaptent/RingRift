from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from app.rules.mutators.territory import TerritoryMutator  # noqa: E402


def _make_player(player_number: int, rings_in_hand: int) -> Player:
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


def _make_forced_elimination_state() -> tuple[GameState, Move]:
    """State where P1 eliminates from their own stack and P2 is blocked.

    After the ELIMINATE_RINGS_FROM_STACK move, GameEngine._update_phase
    delegates to _end_turn, which may apply an extra forced elimination for
    the next player via _perform_forced_elimination_for_player.
    """
    now = datetime.utcnow()

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "5,5": RingStack(
                position=Position(x=5, y=5),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            ),
            "0,0": RingStack(
                position=Position(x=0, y=0),
                rings=[2],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=2,
            ),
        },
        markers={},
        collapsedSpaces={
            # Surround the P2 stack so it has no legal non-capture moves.
            "1,0": 2,
            "0,1": 2,
            "1,1": 2,
        },
        eliminatedRings={},
    )

    players = [
        _make_player(1, rings_in_hand=0),
        _make_player(2, rings_in_hand=0),
    ]

    state = GameState(  # type: ignore[call-arg]
        id="territory-forced-elim",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=[],
        timeControl=_make_time_control(),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=2,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
    )

    move = Move(  # type: ignore[call-arg]
        id="eliminate-5,5",
        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
        player=1,
        to=Position(x=5, y=5),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    return state, move


def _make_non_forced_elimination_state() -> tuple[GameState, Move]:
    """State where P1 eliminates and the next player has material plus
    legal actions, so no host-level forced elimination occurs.
    """
    now = datetime.utcnow()

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "5,5": RingStack(
                position=Position(x=5, y=5),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            ),
        },
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    players = [
        _make_player(1, rings_in_hand=0),
        # P2 has rings in hand but no stacks; _end_turn rotates to them in
        # RING_PLACEMENT without invoking forced elimination.
        _make_player(2, rings_in_hand=5),
    ]

    state = GameState(  # type: ignore[call-arg]
        id="territory-no-forced-elim",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=[],
        timeControl=_make_time_control(),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=6,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
    )

    move = Move(  # type: ignore[call-arg]
        id="eliminate-5,5",
        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
        player=1,
        to=Position(x=5, y=5),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    return state, move


def test_territory_forced_elimination_divergence_pattern() -> None:
    """Reproduce the divergence pattern between TerritoryMutator and the
    canonical GameEngine.apply_move when host-level forced elimination
    occurs after an ELIMINATE_RINGS_FROM_STACK move.
    """
    state, move = _make_forced_elimination_state()

    next_via_engine = GameEngine.apply_move(state, move)

    mut_state = state.model_copy(deep=True)
    TerritoryMutator().apply(mut_state, move)

    # Explicit elimination on the acting player's stack must match exactly.
    assert (
        mut_state.board.stacks.get("5,5")
        == next_via_engine.board.stacks.get("5,5")
    )

    # Canonical path performs an additional forced elimination on the next
    # blocked player, so it must eliminate strictly more rings overall.
    assert (
        next_via_engine.total_rings_eliminated
        > mut_state.total_rings_eliminated
    )

    # That extra elimination should affect the next player's stack, which is
    # removed entirely on the canonical path but untouched in the mutator path.
    assert mut_state.board.stacks.get("0,0") is not None
    assert next_via_engine.board.stacks.get("0,0") is None


def test_default_engine_no_forced_elim_strict_contract() -> None:
    """When no host-level forced elimination occurs after an explicit
    ELIMINATE_RINGS_FROM_STACK decision, DefaultRulesEngine should continue
    to enforce the strict TerritoryMutator contract.
    """
    state, move = _make_non_forced_elimination_state()

    canonical = GameEngine.apply_move(state, move)

    engine = DefaultRulesEngine()
    next_state = engine.apply_move(state, move)

    # No escape hatch should fire; the strict per-move contract ensures that
    # the mutator path matches the canonical GameEngine result exactly.
    assert next_state == canonical


def test_default_engine_forced_elim_escape_hatch() -> None:
    """When host-level forced elimination occurs during _end_turn after an
    ELIMINATE_RINGS_FROM_STACK move, DefaultRulesEngine.apply_move must:
    - not raise a RuntimeError about TerritoryMutator divergence, and
    - still return the canonical GameEngine.apply_move result.
    """
    state, move = _make_forced_elimination_state()

    canonical = GameEngine.apply_move(state, move)

    engine = DefaultRulesEngine()
    next_state = engine.apply_move(state, move)

    assert next_state == canonical