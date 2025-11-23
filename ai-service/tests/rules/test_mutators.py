from datetime import datetime
import os
import sys

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    LineInfo,
    Territory,
    TimeControl,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.mutators.placement import PlacementMutator  # noqa: E402
from app.rules.mutators.movement import MovementMutator  # noqa: E402
from app.rules.mutators.capture import CaptureMutator  # noqa: E402
from app.rules.mutators.line import LineMutator  # noqa: E402
from app.rules.mutators.territory import TerritoryMutator  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402

from tests.rules.test_utils import (
    _make_base_game_state,
    _make_place_ring_move,
)


def test_placement_mutator_empty_cell_multi_ring() -> None:
    """Placing multiple rings on an empty cell updates stack + rings_in_hand."""
    state = _make_base_game_state()
    move = _make_place_ring_move(player=1, x=0, y=0, placement_count=3)

    before_rings = state.players[0].rings_in_hand

    PlacementMutator().apply(state, move)

    pos_key = move.to.to_key()
    stack = state.board.stacks[pos_key]

    # Rings and stack geometry
    assert stack.rings == [1, 1, 1]
    assert stack.stack_height == 3
    assert stack.cap_height == 3
    assert stack.controlling_player == 1

    # rings_in_hand decremented but never below zero
    assert state.players[0].rings_in_hand == max(0, before_rings - 3)


def test_placement_mutator_on_existing_stack_appends_and_recomputes_cap() -> None:
    """Placing on an existing stack appends rings and recomputes capHeight."""
    state = _make_base_game_state()
    pos = Position(x=1, y=1)
    pos_key = pos.to_key()

    # Existing stack: bottom [2, 2, 1] (top = 1, capHeight = 1)
    existing = RingStack(
        position=pos,
        rings=[2, 2, 1],
        stackHeight=3,
        capHeight=1,
        controllingPlayer=1,
    )
    state.board.stacks[pos_key] = existing

    move = _make_place_ring_move(
        player=1,
        x=pos.x,
        y=pos.y,
        placement_count=1,
        placed_on_stack=True,
    )

    PlacementMutator().apply(state, move)

    stack = state.board.stacks[pos_key]
    # New ring appended on top
    assert stack.rings == [2, 2, 1, 1]
    assert stack.stack_height == 4
    # Cap is now the run of 1s on top: length 2
    assert stack.cap_height == 2
    assert stack.controlling_player == 1


def test_placement_mutator_does_not_touch_markers_or_collapsed_spaces() -> None:
    """PlacementMutator delegates pure stack logic to GameEngine._apply_place_ring.

    Board markers and collapsed spaces should be left unchanged; these are
    managed by movement/capture/line/territory logic instead.
    """
    state = _make_base_game_state()
    pos = Position(x=2, y=2)
    pos_key = pos.to_key()

    # Pre-populate a marker and a collapsed space elsewhere on the board.
    marker_pos = Position(x=3, y=3)
    marker_key = marker_pos.to_key()
    state.board.markers[marker_key] = MarkerInfo(
        player=1,
        position=marker_pos,
        type="regular",
    )

    collapsed_pos = Position(x=4, y=4)
    collapsed_key = collapsed_pos.to_key()
    state.board.collapsed_spaces[collapsed_key] = 1

    move = _make_place_ring_move(
        player=1,
        x=pos.x,
        y=pos.y,
        placement_count=1,
    )
    PlacementMutator().apply(state, move)

    # Marker and collapsed spaces unaffected
    assert marker_key in state.board.markers
    assert collapsed_key in state.board.collapsed_spaces

    # Placement still created/updated a stack at the target position
    assert pos_key in state.board.stacks


def test_placement_mutator_matches_game_engine_for_place_ring_board_and_players(
) -> None:
    """PlacementMutator semantics match GameEngine._apply_place_ring.

    We compare only board + players fields, leaving orchestration concerns
    (move history, phase transitions, hashes, etc.) to GameEngine.
    """
    base_state = _make_base_game_state()
    move = _make_place_ring_move(
        player=1,
        x=0,
        y=0,
        placement_count=2,
    )

    # Baseline: canonical GameEngine path
    engine_next = GameEngine.apply_move(base_state, move)

    # Mutator path: apply to a fresh copy of the original state
    mutator_state = _make_base_game_state()
    PlacementMutator().apply(mutator_state, move)

    # Board stacks and markers/collapsed spaces should match exactly
    assert mutator_state.board.stacks == engine_next.board.stacks
    assert mutator_state.board.markers == engine_next.board.markers
    assert (
        mutator_state.board.collapsed_spaces
        == engine_next.board.collapsed_spaces
    )
    assert (
        mutator_state.board.eliminated_rings
        == engine_next.board.eliminated_rings
    )

    # Player ring counts + elimination/territory metadata should match
    assert mutator_state.players == engine_next.players


def test_movement_mutator_matches_game_engine_for_move_stack_board_and_players(
) -> None:
    """MovementMutator semantics match GameEngine._apply_move_stack.

    As with placement, we compare only board + players fields, leaving
    orchestration concerns (move history, phase transitions, hashes, etc.)
    to GameEngine.apply_move.
    """
    # Shared initial state with a single stack for player 1 at (0, 0).
    base_state = _make_base_game_state()
    mutator_state = _make_base_game_state()

    from_pos = Position(x=0, y=0)
    from_key = from_pos.to_key()

    stack = RingStack(
        position=from_pos,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )
    base_state.board.stacks[from_key] = stack
    mutator_state.board.stacks[from_key] = stack.model_copy()

    # Simple non-capture move two steps up the file.
    to_pos = Position(x=0, y=2)
    now = datetime.now()
    move = Move(
        id="m-move",
        type=MoveType.MOVE_STACK,
        player=1,
        from_pos=from_pos,
        to=to_pos,
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    # Baseline: canonical GameEngine path
    engine_next = GameEngine.apply_move(base_state, move)

    # Mutator path: apply to a fresh copy of the original state
    MovementMutator().apply(mutator_state, move)

    # Board-level equivalence
    assert mutator_state.board.stacks == engine_next.board.stacks
    assert mutator_state.board.markers == engine_next.board.markers
    assert (
        mutator_state.board.collapsed_spaces
        == engine_next.board.collapsed_spaces
    )
    assert (
        mutator_state.board.eliminated_rings
        == engine_next.board.eliminated_rings
    )

    # Player metadata equivalence
    assert mutator_state.players == engine_next.players


def test_capture_mutator_matches_game_engine_for_overtaking_capture(
) -> None:
    """CaptureMutator semantics match GameEngine._apply_chain_capture.

    We construct a simple overtaking capture segment and assert that running
    the canonical GameEngine.apply_move path and the CaptureMutator path on
    identical starting states yields the same board + player side-effects.
    """
    base_state = _make_base_game_state()

    attacker_pos = Position(x=0, y=0)
    attacker_key = attacker_pos.to_key()
    target_pos = Position(x=0, y=2)
    target_key = target_pos.to_key()
    landing_pos = Position(x=0, y=5)

    # Attacker: height 2, cap 2, owned by player 1.
    attacker_stack = RingStack(
        position=attacker_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    # Target: single-ring stack for player 2 with smaller cap.
    target_stack = RingStack(
        position=target_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )

    base_state.board.stacks[attacker_key] = attacker_stack
    base_state.board.stacks[target_key] = target_stack

    # Mutator path starts from a deep copy of the same initial state.
    mutator_state = base_state.model_copy(deep=True)

    now = datetime.now()
    move = Move(
        id="m-cap",
        type=MoveType.OVERTAKING_CAPTURE,
        player=1,
        from_pos=attacker_pos,
        to=landing_pos,
        capture_target=target_pos,
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    engine_next = GameEngine.apply_move(base_state, move)
    CaptureMutator().apply(mutator_state, move)

    # Board-level equivalence
    assert mutator_state.board.stacks == engine_next.board.stacks
    assert mutator_state.board.markers == engine_next.board.markers
    assert (
        mutator_state.board.collapsed_spaces
        == engine_next.board.collapsed_spaces
    )
    assert (
        mutator_state.board.eliminated_rings
        == engine_next.board.eliminated_rings
    )

    # Player metadata equivalence
    assert mutator_state.players == engine_next.players


def test_line_mutator_matches_game_engine_for_process_line_board_and_players(
) -> None:
    """LineMutator semantics match GameEngine._apply_line_formation.

    We use a synthetic line via BoardManager monkeypatching, mirroring
    tests/parity/test_line_and_territory_scenario_parity.py, and verify
    that LineMutator and GameEngine.apply_move agree on board + players.
    """
    state = _make_base_game_state()
    state.current_phase = GamePhase.LINE_PROCESSING

    required_len = 3 if state.board.type == BoardType.SQUARE8 else 4
    line_positions = [Position(x=i, y=0) for i in range(required_len)]

    synthetic_line = LineInfo(
        positions=line_positions,
        player=1,
        length=len(line_positions),
        direction=Position(x=1, y=0),
    )

    orig_find_all_lines = BoardManager.find_all_lines

    try:
        BoardManager.find_all_lines = staticmethod(  # type: ignore[assignment]
            lambda board: [synthetic_line]
        )

        line_moves = GameEngine._get_line_processing_moves(state, 1)
        assert line_moves, "Expected at least one line-processing move"
        move = line_moves[0]

        engine_next = GameEngine.apply_move(state, move)
        mutator_state = state.model_copy(deep=True)
        LineMutator().apply(mutator_state, move)

        # Board-level equivalence
        assert mutator_state.board.stacks == engine_next.board.stacks
        assert mutator_state.board.markers == engine_next.board.markers
        assert (
            mutator_state.board.collapsed_spaces
            == engine_next.board.collapsed_spaces
        )
        assert (
            mutator_state.board.eliminated_rings
            == engine_next.board.eliminated_rings
        )

        # Player metadata equivalence
        assert mutator_state.players == engine_next.players

    finally:
        BoardManager.find_all_lines = orig_find_all_lines  # type: ignore[assignment]


def test_territory_mutator_matches_game_engine_for_process_region_board_and_players(
) -> None:
    """TerritoryMutator semantics match GameEngine._apply_territory_claim.

    We construct a minimal disconnected-region scenario, monkeypatching
    BoardManager helpers, and assert that TerritoryMutator and
    GameEngine.apply_move agree on board + players.
    """
    state = _make_base_game_state()
    state.current_phase = GamePhase.TERRITORY_PROCESSING

    board = state.board
    region_pos = Position(x=5, y=5)
    region_key = region_pos.to_key()

    # P2 stack inside the region.
    region_stack = RingStack(
        position=region_pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )
    board.stacks[region_key] = region_stack

    # P1 stack outside the region to satisfy self-elimination prerequisite.
    outside_pos = Position(x=7, y=7)
    outside_key = outside_pos.to_key()
    p1_rings = [1, 1]
    outside_stack = RingStack(
        position=outside_pos,
        rings=p1_rings,
        stackHeight=len(p1_rings),
        capHeight=len(p1_rings),
        controllingPlayer=1,
    )
    board.stacks[outside_key] = outside_stack

    region_territory = Territory(
        spaces=[region_pos],
        controllingPlayer=1,
        isDisconnected=True,
    )

    orig_find_regions = BoardManager.find_disconnected_regions
    orig_get_border = BoardManager.get_border_marker_positions

    try:
        BoardManager.find_disconnected_regions = staticmethod(  # type: ignore[assignment]
            lambda b, moving_player: [region_territory]
        )
        BoardManager.get_border_marker_positions = staticmethod(  # type: ignore[assignment]
            lambda spaces, b: []
        )

        territory_moves = GameEngine._get_territory_processing_moves(
            state, 1
        )
        assert territory_moves, "Expected at least one territory move"
        move = territory_moves[0]

        engine_next = GameEngine.apply_move(state, move)
        mutator_state = state.model_copy(deep=True)
        TerritoryMutator().apply(mutator_state, move)

        # Board-level equivalence
        assert mutator_state.board.stacks == engine_next.board.stacks
        assert mutator_state.board.markers == engine_next.board.markers
        assert (
            mutator_state.board.collapsed_spaces
            == engine_next.board.collapsed_spaces
        )
        assert (
            mutator_state.board.eliminated_rings
            == engine_next.board.eliminated_rings
        )

        # Player metadata equivalence
        assert mutator_state.players == engine_next.players

    finally:
        BoardManager.find_disconnected_regions = (  # type: ignore[assignment]
            orig_find_regions
        )
        BoardManager.get_border_marker_positions = (  # type: ignore[assignment]
            orig_get_border
        )
