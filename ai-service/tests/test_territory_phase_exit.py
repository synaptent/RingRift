from datetime import datetime

from app.game_engine import GameEngine, PhaseRequirementType
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
    Territory,
    TimeControl,
)


def test_eliminate_rings_from_stack_does_not_silently_exit_territory_processing() -> None:
    """
    Regression: after an ELIMINATE_RINGS_FROM_STACK decision in territory_processing,
    Python must NOT rotate to the next player when no further territory decisions remain.

    Instead, it should remain in territory_processing so the host can record an
    explicit NO_TERRITORY_ACTION bookkeeping move (RR-CANON-R075/R204), mirroring TS.
    """

    now = datetime.now()

    board = BoardState(type=BoardType.SQUARE8, size=8)
    # Two active players with stacks but no disconnected regions/marker borders.
    board.stacks = {
        "0,0": RingStack(
            position=Position(x=0, y=0),
            rings=[2, 2],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=2,
        ),
        "1,1": RingStack(
            position=Position(x=1, y=1),
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        ),
    }

    players = [
        Player(
            id="p1",
            username="P1",
            type="ai",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            aiDifficulty=None,
            ringsInHand=10,
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
            aiDifficulty=None,
            ringsInHand=10,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    choose_region = Move(
        id="choose-region-1",
        type=MoveType.CHOOSE_TERRITORY_OPTION,
        player=2,
        to=Position(x=2, y=2),
        disconnectedRegions=(
            Territory(
                spaces=[Position(x=2, y=2)],
                controllingPlayer=2,
                isDisconnected=True,
            ),
        ),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    state = GameState(
        id="territory-phase-exit-test",
        boardType=BoardType.SQUARE8,
        rngSeed=123,
        board=board,
        players=players,
        currentPhase=GamePhase.TERRITORY_PROCESSING,
        currentPlayer=2,
        moveHistory=[choose_region],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=999,
        territoryVictoryThreshold=999,
    )

    eliminate = Move(
        id="elim-1",
        type=MoveType.ELIMINATE_RINGS_FROM_STACK,
        player=2,
        to=Position(x=0, y=0),
        eliminationContext="territory",
        timestamp=now,
        thinkTime=0,
        moveNumber=2,
    )

    after_elim = GameEngine.apply_move(state, eliminate, trace_mode=True)
    assert after_elim.current_phase == GamePhase.TERRITORY_PROCESSING
    assert after_elim.current_player == 2

    # No further interactive territory decisions exist, so the host must emit NO_TERRITORY_ACTION.
    assert GameEngine.get_valid_moves(after_elim, 2) == []
    req = GameEngine.get_phase_requirement(after_elim, 2)
    assert req is not None
    assert req.type == PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED

    bookkeeping = GameEngine.synthesize_bookkeeping_move(req, after_elim)
    after_bookkeeping = GameEngine.apply_move(after_elim, bookkeeping, trace_mode=True)
    assert after_bookkeeping.current_phase == GamePhase.RING_PLACEMENT
    assert after_bookkeeping.current_player == 1

