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
    Territory,
    TimeControl,
)
from app.rules.core import infer_total_rings_in_play


def test_eliminate_rings_from_stack_ends_turn_when_territory_complete() -> None:
    """
    After an ELIMINATE_RINGS_FROM_STACK decision in territory_processing, if no
    further territory decisions remain, the engine must end the turn and rotate
    to the next player's ring_placement (RR-CANON-R075/R204).

    NO_TERRITORY_ACTION is reserved for the case where the player entered
    territory_processing and had *no* eligible territory decisions at all.
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
        totalRingsInPlay=infer_total_rings_in_play(players, board),
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
    assert after_elim.current_phase == GamePhase.RING_PLACEMENT
    assert after_elim.current_player == 1
