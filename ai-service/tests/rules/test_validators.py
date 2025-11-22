import pytest
from datetime import datetime
from app.models import (
    GameState, Move, MoveType, GamePhase, BoardType, BoardState, Player,
    TimeControl, Position
)
from app.rules.validators.placement import PlacementValidator
from app.rules.validators.movement import MovementValidator
from app.rules.validators.capture import CaptureValidator
from app.rules.validators.line import LineValidator
from app.rules.validators.territory import TerritoryValidator

@pytest.fixture
def empty_game_state():
    return GameState(
        id="test",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=[
            Player(
                id="p1", username="p1", type="human", playerNumber=1,
                isReady=True, timeRemaining=60, ringsInHand=18,
                eliminatedRings=0, territorySpaces=0
            ),
            Player(
                id="p2", username="p2", type="human", playerNumber=2,
                isReady=True, timeRemaining=60, ringsInHand=18,
                eliminatedRings=0, territorySpaces=0
            )
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus="active",
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10
    )


def test_placement_validator(empty_game_state):
    validator = PlacementValidator()

    # Valid placement
    move = Move(
        id="m1", type=MoveType.PLACE_RING, player=1,
        to=Position(x=0, y=0), timestamp=datetime.now(), thinkTime=0,
        moveNumber=1
    )
    assert validator.validate(empty_game_state, move) is True

    # Invalid phase
    empty_game_state.current_phase = GamePhase.MOVEMENT
    assert validator.validate(empty_game_state, move) is False
    empty_game_state.current_phase = GamePhase.RING_PLACEMENT

    # Invalid player
    move_wrong_player = move.model_copy(update={"player": 2})
    assert validator.validate(empty_game_state, move_wrong_player) is False


def test_movement_validator(empty_game_state):
    validator = MovementValidator()
    empty_game_state.current_phase = GamePhase.MOVEMENT

    # Setup stack
    from app.models import RingStack
    pos = Position(x=0, y=0)
    stack = RingStack(
        position=pos, rings=[1], stackHeight=1, capHeight=1,
        controllingPlayer=1
    )
    empty_game_state.board.stacks[pos.to_key()] = stack

    # Valid move
    move = Move(
        id="m1", type=MoveType.MOVE_STACK, player=1,
        from_pos=pos, to=Position(x=0, y=2),
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move) is True

    # Invalid distance (too short)
    # Same pos
    move_short = move.model_copy(update={"to": Position(x=0, y=0)})
    assert validator.validate(empty_game_state, move_short) is False


def test_capture_validator(empty_game_state):
    validator = CaptureValidator()
    empty_game_state.current_phase = GamePhase.MOVEMENT

    # Setup attacker and target
    from app.models import RingStack
    attacker_pos = Position(x=0, y=0)
    target_pos = Position(x=0, y=2)
    landing_pos = Position(x=0, y=3)

    attacker = RingStack(
        position=attacker_pos, rings=[1, 1], stackHeight=2, capHeight=2,
        controllingPlayer=1
    )
    target = RingStack(
        position=target_pos, rings=[2], stackHeight=1, capHeight=1,
        controllingPlayer=2
    )

    empty_game_state.board.stacks[attacker_pos.to_key()] = attacker
    empty_game_state.board.stacks[target_pos.to_key()] = target

    # Valid capture
    move = Move(
        id="m1", type=MoveType.OVERTAKING_CAPTURE, player=1,
        from_pos=attacker_pos, to=landing_pos, capture_target=target_pos,
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move) is True


def test_line_validator(empty_game_state):
    validator = LineValidator()
    empty_game_state.current_phase = GamePhase.LINE_PROCESSING

    # Setup line
    from app.models import LineInfo
    line = LineInfo(
        positions=[Position(x=0, y=i) for i in range(5)],
        player=1, length=5, direction=Position(x=0, y=1)
    )
    empty_game_state.board.formed_lines.append(line)

    # Valid line process
    move = Move(
        id="m1", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),  # Dummy pos
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move) is True
    
    # No lines
    empty_game_state.board.formed_lines = []
    assert validator.validate(empty_game_state, move) is False


def test_territory_validator(empty_game_state):
    validator = TerritoryValidator()
    empty_game_state.current_phase = GamePhase.TERRITORY_PROCESSING
 
    # Setup disconnected territory
    from app.models import Territory
    territory = Territory(
        spaces=[Position(x=0, y=0)],
        controllingPlayer=1,
        isDisconnected=True,
    )
    empty_game_state.board.territories["t1"] = territory
 
    # Valid territory process
    move = Move(
        id="m1",
        type=MoveType.PROCESS_TERRITORY_REGION,
        player=1,
        to=Position(x=0, y=0),  # Dummy pos
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )
    assert validator.validate(empty_game_state, move) is True
     
    # No disconnected territory
    empty_game_state.board.territories = {}
    assert validator.validate(empty_game_state, move) is False
 
 
def test_no_dead_placement_matches_movement_semantics(empty_game_state):
    """
    For any PLACE_RING move accepted by the engine's ring placement
    generator, if applying that move leads to MOVEMENT for the same
    player, then GameEngine.get_valid_moves must return at least one
    legal move (movement or capture) in that state.
    """
    from app.game_engine import GameEngine
 
    state = empty_game_state
    player = state.current_player
 
    placement_moves = GameEngine._get_ring_placement_moves(state, player)
    assert placement_moves, "Expected at least one legal placement move"
 
    for placement in placement_moves:
        next_state = GameEngine.apply_move(state, placement)
 
        # Only check the invariant when the engine actually advances to
        # MOVEMENT; some placements may skip directly to line/territory
        # processing.
        if next_state.current_phase != GamePhase.MOVEMENT:
            continue
 
        assert next_state.must_move_from_stack_key == placement.to.to_key()
 
        moves = GameEngine.get_valid_moves(next_state, player)
        assert moves, (
            "Placement accepted by no-dead-placement led to MOVEMENT "
            "phase with no legal moves"
        )