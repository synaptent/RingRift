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

    # Valid line process with line_index=0 (first line)
    move = Move(
        id="m1", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),  # Dummy pos
        line_index=0,  # Required field for TS parity
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move) is True

    # No lines - line_index 0 is now out of bounds
    empty_game_state.board.formed_lines = []
    assert validator.validate(empty_game_state, move) is False


def test_line_validator_line_index_required(empty_game_state):
    """Test that line_index is required for line processing moves (RR-CANON parity)."""
    validator = LineValidator()
    empty_game_state.current_phase = GamePhase.LINE_PROCESSING

    from app.models import LineInfo
    line = LineInfo(
        positions=[Position(x=0, y=i) for i in range(4)],
        player=1, length=4, direction=Position(x=0, y=1)
    )
    empty_game_state.board.formed_lines.append(line)

    # Missing line_index should fail
    move_no_index = Move(
        id="m1", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
        # line_index intentionally omitted
    )
    assert validator.validate(empty_game_state, move_no_index) is False


def test_line_validator_line_index_bounds(empty_game_state):
    """Test line_index bounds checking mirrors TS validateProcessLine."""
    validator = LineValidator()
    empty_game_state.current_phase = GamePhase.LINE_PROCESSING

    from app.models import LineInfo
    line = LineInfo(
        positions=[Position(x=0, y=i) for i in range(4)],
        player=1, length=4, direction=Position(x=0, y=1)
    )
    empty_game_state.board.formed_lines.append(line)

    # Negative index should fail
    move_negative = Move(
        id="m1", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        line_index=-1,
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_negative) is False

    # Index beyond formed_lines length should fail
    move_out_of_bounds = Move(
        id="m2", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        line_index=5,  # Only 1 line exists
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_out_of_bounds) is False

    # Exactly at boundary (len=1, index=1) should fail
    move_at_boundary = Move(
        id="m3", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        line_index=1,  # Only 1 line exists (index 0)
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_at_boundary) is False


def test_line_validator_line_ownership(empty_game_state):
    """Test that line_index must reference a line owned by the moving player."""
    validator = LineValidator()
    empty_game_state.current_phase = GamePhase.LINE_PROCESSING

    from app.models import LineInfo
    # Line owned by player 2
    line_p2 = LineInfo(
        positions=[Position(x=0, y=i) for i in range(4)],
        player=2, length=4, direction=Position(x=0, y=1)
    )
    # Line owned by player 1
    line_p1 = LineInfo(
        positions=[Position(x=1, y=i) for i in range(4)],
        player=1, length=4, direction=Position(x=0, y=1)
    )
    empty_game_state.board.formed_lines = [line_p2, line_p1]

    # Player 1 trying to process player 2's line (index 0) should fail
    move_wrong_owner = Move(
        id="m1", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        line_index=0,  # Points to line_p2 owned by player 2
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_wrong_owner) is False

    # Player 1 processing their own line (index 1) should succeed
    move_correct_owner = Move(
        id="m2", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=1, y=0),
        line_index=1,  # Points to line_p1 owned by player 1
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_correct_owner) is True


def test_line_validator_multiple_lines(empty_game_state):
    """Test line_index correctly identifies the target line in multi-line scenarios."""
    validator = LineValidator()
    empty_game_state.current_phase = GamePhase.LINE_PROCESSING

    from app.models import LineInfo
    # Multiple lines for player 1
    lines = [
        LineInfo(
            positions=[Position(x=i, y=0) for i in range(4)],
            player=1, length=4, direction=Position(x=1, y=0)
        ),
        LineInfo(
            positions=[Position(x=0, y=i) for i in range(4)],
            player=1, length=4, direction=Position(x=0, y=1)
        ),
        LineInfo(
            positions=[Position(x=i, y=i) for i in range(4)],
            player=1, length=4, direction=Position(x=1, y=1)
        ),
    ]
    empty_game_state.board.formed_lines = lines

    # Each valid index should work
    for idx in range(3):
        move = Move(
            id=f"m{idx}", type=MoveType.PROCESS_LINE, player=1,
            to=lines[idx].positions[0],
            line_index=idx,
            timestamp=datetime.now(), thinkTime=0, moveNumber=1
        )
        assert validator.validate(empty_game_state, move) is True, f"Index {idx} should be valid"

    # Index 3 should fail (only 3 lines exist)
    move_invalid = Move(
        id="m3", type=MoveType.PROCESS_LINE, player=1,
        to=Position(x=0, y=0),
        line_index=3,
        timestamp=datetime.now(), thinkTime=0, moveNumber=1
    )
    assert validator.validate(empty_game_state, move_invalid) is False


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