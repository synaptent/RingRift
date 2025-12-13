from datetime import datetime

from app.game_engine import GameEngine
from app.models import (
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


def _make_base_state(*, move_history: list[Move]) -> GameState:
    now = datetime.now()
    players = [
        Player(
            id="p1",
            username="P1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=1,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="P2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=1,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    # Create an eligible self-elimination target (single-colour, height > 1)
    # so that territory-processing follow-ups can surface ELIMINATE_RINGS_FROM_STACK.
    stack_pos = Position(x=0, y=0)
    stack = RingStack(
        position=stack_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={stack_pos.to_key(): stack},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    return GameState(
        id="move-cache-territory-regression",
        boardType=BoardType.SQUARE8,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=1,
        moveHistory=move_history,
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        # Force a stable zobrist hash so the move cache key depends on the
        # metadata we intend to test (last move signature), not on hash
        # recomputation.
        zobristHash=123,
        rulesOptions=None,
    )


def test_move_cache_does_not_hide_territory_followup_elimination_moves() -> None:
    """Regression guard for cached territory-processing decision surfaces."""
    GameEngine.clear_cache()

    now = datetime.now()
    empty_surface = _make_base_state(
        move_history=[
            Move(
                id="m1",
                type=MoveType.PROCESS_LINE,
                player=1,
                to=Position(x=1, y=1),
                timestamp=now,
                thinkTime=0,
                moveNumber=1,
            )
        ]
    )

    followup_surface = _make_base_state(
        move_history=[
            Move(
                id="m1",
                type=MoveType.CHOOSE_TERRITORY_OPTION,
                player=1,
                to=Position(x=1, y=1),
                timestamp=now,
                thinkTime=0,
                moveNumber=1,
            )
        ]
    )

    assert GameEngine.get_valid_moves(empty_surface, 1) == []

    followups = GameEngine.get_valid_moves(followup_surface, 1)
    assert any(m.type == MoveType.ELIMINATE_RINGS_FROM_STACK for m in followups)
