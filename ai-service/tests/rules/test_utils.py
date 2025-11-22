import os
import sys
from datetime import datetime

# Ensure `app.*` imports resolve when running pytest from ai-service/
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
    TimeControl,
)


def _make_base_game_state() -> GameState:
    """Create a minimal, square8 game state for rules tests.

    This helper centralises the common square8 two-player seed used across
    mutator and DefaultRulesEngine equivalence tests. It mirrors the
    alias-style field names from the TS JSON shape (boardType, currentPhase,
    etc.) so it can be safely reused in parity-oriented scenarios.
    """
    board = BoardState(type=BoardType.SQUARE8, size=8)
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    now = datetime.now()

    return GameState(
        id="test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def _make_place_ring_move(
    player: int,
    x: int,
    y: int,
    placement_count: int | None = None,
    placed_on_stack: bool | None = None,
) -> Move:
    """Construct a PLACE_RING move with sensible defaults.

    This mirrors the helper previously in test_mutators and is reused by
    multiple rules tests to keep move construction consistent with the TS
    fixtures (placementCount / placedOnStack fields).
    """
    now = datetime.now()
    kwargs: dict = {}
    if placement_count is not None:
        kwargs["placementCount"] = placement_count
    if placed_on_stack is not None:
        kwargs["placedOnStack"] = placed_on_stack

    return Move(
        id="m1",
        type=MoveType.PLACE_RING,
        player=player,
        to=Position(x=x, y=y),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
        **kwargs,
    )
