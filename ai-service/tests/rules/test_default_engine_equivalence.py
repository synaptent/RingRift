import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from app.training.env import RingRiftEnv  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402

from tests.rules.test_utils import _make_base_game_state


@pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
def test_default_engine_apply_move_matches_game_engine_for_place_ring(
    board_type: BoardType,
) -> None:
    """DefaultRulesEngine.apply_move stays in lockstep with GameEngine.apply_move.

    This harness focuses on PLACE_RING moves, where we ultimately want the
    rules engine to delegate mutation to PlacementMutator while preserving
    the orchestration semantics of GameEngine.apply_move.

    For now DefaultRulesEngine.apply_move simply defers to GameEngine, so
    this test acts as a guardrail for future refactors.
    """
    env = RingRiftEnv(board_type)
    state = env.reset()

    # Select a concrete PLACE_RING move from the canonical move generator.
    moves = GameEngine.get_valid_moves(state, state.current_player)
    place_moves = [m for m in moves if m.type == MoveType.PLACE_RING]
    assert place_moves, "Expected at least one PLACE_RING move"
    move = place_moves[0]

    engine = DefaultRulesEngine()

    next_via_engine = GameEngine.apply_move(state, move)
    next_via_rules = engine.apply_move(state, move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping should also stay aligned.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status


@pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
def test_default_engine_apply_move_matches_game_engine_for_move_stack(
    board_type: BoardType,
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for MOVE_STACK.

    This mirrors the mutator-level MOVE_STACK shadow contract by asserting
    that the full DefaultRulesEngine.apply_move orchestration remains in
    lockstep with GameEngine.apply_move for a concrete MOVE_STACK move
    discovered via RingRiftEnv.
    """
    env = RingRiftEnv(board_type)
    state = env.reset()
    engine = DefaultRulesEngine()

    # Advance until a MOVE_STACK move exists.
    for _ in range(20):
        moves = GameEngine.get_valid_moves(state, state.current_player)
        move_stack_moves = [m for m in moves if m.type == MoveType.MOVE_STACK]
        if move_stack_moves:
            move = move_stack_moves[0]
            break

        placement_or_skip = [
            m
            for m in moves
            if m.type in (MoveType.PLACE_RING, MoveType.SKIP_PLACEMENT)
        ]
        assert placement_or_skip
        state = GameEngine.apply_move(state, placement_or_skip[0])
    else:
        raise AssertionError("Failed to find a MOVE_STACK move within 20 plies")

    next_via_engine = GameEngine.apply_move(state, move)
    next_via_rules = engine.apply_move(state, move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping should also stay aligned.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status

def test_default_engine_apply_move_matches_game_engine_for_overtaking_capture_synthetic(
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for a simple capture.

    This is a synthetic overtaking-capture segment mirroring the
    CaptureMutator unit test. We assert that DefaultRulesEngine.apply_move
    (with its CaptureMutator shadow contract) stays in full-state lockstep
    with GameEngine.apply_move.
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

    engine = DefaultRulesEngine()

    next_via_engine = GameEngine.apply_move(base_state, move)
    next_via_rules = engine.apply_move(base_state, move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping and chain-capture state should align.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status
    assert next_via_rules.chain_capture_state == next_via_engine.chain_capture_state


@pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
def test_default_engine_apply_move_matches_game_engine_for_chain_capture_continuation_env(
    board_type: BoardType,
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for continuation segments.

    Starting from a fresh RingRiftEnv game, advance until an initial
    OVERTAKING_CAPTURE is available, apply it via GameEngine to enter a
    chain-capture state, then require that applying a CONTINUE_CAPTURE_SEGMENT
    move via DefaultRulesEngine and GameEngine yields identical next states.
    """
    env = RingRiftEnv(board_type)
    state = env.reset()

    # First, find and apply an initial overtaking capture to enter
    # chain-capture state.
    first_capture: Move | None = None

    for _ in range(300):
        moves = GameEngine.get_valid_moves(state, state.current_player)
        capture_moves = [m for m in moves if m.type == MoveType.OVERTAKING_CAPTURE]
        if capture_moves:
            first_capture = capture_moves[0]
            break

        progression_moves = [
            m
            for m in moves
            if m.type
            in (
                MoveType.PLACE_RING,
                MoveType.SKIP_PLACEMENT,
                MoveType.MOVE_STACK,
            )
        ]
        assert progression_moves, (
            "Expected at least one non-capture move to advance the game "
            "while searching for an initial capture"
        )
        state = GameEngine.apply_move(state, progression_moves[0])
    else:
        raise AssertionError(
            "Failed to find an initial OVERTAKING_CAPTURE within 300 plies"
        )

    assert first_capture is not None

    # Apply the initial capture via the canonical engine to get a
    # chain-capture state.
    state_after_first = GameEngine.apply_move(state, first_capture)
    assert state_after_first.chain_capture_state is not None, (
        "Expected chain_capture_state to be populated after initial capture"
    )

    # From this state, enumerate continuation options.
    continuation_moves = [
        m
        for m in GameEngine.get_valid_moves(
            state_after_first,
            state_after_first.current_player,
        )
        if m.type == MoveType.CONTINUE_CAPTURE_SEGMENT
    ]
    assert continuation_moves, (
        "Expected at least one CONTINUE_CAPTURE_SEGMENT move after initial "
        "capture"
    )
    continuation_move = continuation_moves[0]

    engine = DefaultRulesEngine()

    next_via_engine = GameEngine.apply_move(state_after_first, continuation_move)
    next_via_rules = engine.apply_move(state_after_first, continuation_move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping and chain-capture state should align.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status
    assert next_via_rules.chain_capture_state == next_via_engine.chain_capture_state


def test_default_engine_apply_move_matches_game_engine_for_process_line_synthetic(
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for PROCESS_LINE.

    We construct a synthetic line for player 1 by monkeypatching
    BoardManager.find_all_lines, obtain a canonical PROCESS_LINE move via
    GameEngine._get_line_processing_moves, and assert that applying this move
    via DefaultRulesEngine and GameEngine yields identical next states.
    """
    state = _make_base_game_state()
    state.current_phase = GamePhase.LINE_PROCESSING

    required_len = 3 if state.board.type == BoardType.SQUARE8 else 4
    line_positions = [Position(x=i, y=0) for i in range(required_len)]

    # Import LineInfo lazily to keep imports focused at the top of the file.
    from app.models import LineInfo  # noqa: WPS433,E402

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

        engine = DefaultRulesEngine()

        next_via_engine = GameEngine.apply_move(state, move)
        next_via_rules = engine.apply_move(state, move)

        # Board-level equivalence
        assert next_via_rules.board.stacks == next_via_engine.board.stacks
        assert next_via_rules.board.markers == next_via_engine.board.markers
        assert (
            next_via_rules.board.collapsed_spaces
            == next_via_engine.board.collapsed_spaces
        )
        assert (
            next_via_rules.board.eliminated_rings
            == next_via_engine.board.eliminated_rings
        )

        # Player metadata equivalence
        assert next_via_rules.players == next_via_engine.players

        # Turn/phase/victory bookkeeping should also stay aligned (though
        # line-processing moves are currently phase-preserving).
        assert next_via_rules.current_player == next_via_engine.current_player
        assert next_via_rules.current_phase == next_via_engine.current_phase
        assert next_via_rules.game_status == next_via_engine.game_status

    finally:
        BoardManager.find_all_lines = orig_find_all_lines  # type: ignore[assignment]


def test_default_engine_apply_move_matches_game_engine_for_process_territory_region_synthetic(
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for PROCESS_TERRITORY_REGION.

    We construct a minimal disconnected-region scenario (mirroring the
    TerritoryMutator unit test), monkeypatch BoardManager helpers, and
    assert that DefaultRulesEngine.apply_move and GameEngine.apply_move
    agree on the resulting state for a PROCESS_TERRITORY_REGION move.
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

    # Import Territory lazily to keep top-level imports concise.
    from app.models import Territory  # noqa: WPS433,E402

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

        territory_moves = GameEngine._get_territory_processing_moves(state, 1)
        assert territory_moves, "Expected at least one territory move"

        process_region_moves = [
            m for m in territory_moves if m.type == MoveType.PROCESS_TERRITORY_REGION
        ]
        assert process_region_moves, (
            "Expected at least one PROCESS_TERRITORY_REGION move from "
            "_get_territory_processing_moves"
        )
        move = process_region_moves[0]

        engine = DefaultRulesEngine()

        next_via_engine = GameEngine.apply_move(state, move)
        next_via_rules = engine.apply_move(state, move)

        # Board-level equivalence
        assert next_via_rules.board.stacks == next_via_engine.board.stacks
        assert next_via_rules.board.markers == next_via_engine.board.markers
        assert (
            next_via_rules.board.collapsed_spaces
            == next_via_engine.board.collapsed_spaces
        )
        assert (
            next_via_rules.board.eliminated_rings
            == next_via_engine.board.eliminated_rings
        )

        # Player metadata equivalence
        assert next_via_rules.players == next_via_engine.players

        # Turn/phase/victory bookkeeping should also stay aligned.
        assert next_via_rules.current_player == next_via_engine.current_player
        assert next_via_rules.current_phase == next_via_engine.current_phase
        assert next_via_rules.game_status == next_via_engine.game_status

    finally:
        BoardManager.find_disconnected_regions = (  # type: ignore[assignment]
            orig_find_regions
        )
        BoardManager.get_border_marker_positions = (  # type: ignore[assignment]
            orig_get_border
        )


def test_default_engine_apply_move_matches_game_engine_for_eliminate_rings_from_stack_synthetic(
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for ELIMINATE_RINGS_FROM_STACK.

    We build a simple board where player 1 controls a single capped stack,
    obtain an ELIMINATE_RINGS_FROM_STACK move from
    GameEngine._get_territory_processing_moves, and assert that applying it
    via DefaultRulesEngine and GameEngine yields identical states.
    """
    state = _make_base_game_state()
    state.current_phase = GamePhase.TERRITORY_PROCESSING

    pos = Position(x=3, y=3)
    pos_key = pos.to_key()
    rings = [1, 1, 1]
    stack = RingStack(
        position=pos,
        rings=rings,
        stackHeight=len(rings),
        capHeight=len(rings),
        controllingPlayer=1,
    )
    state.board.stacks[pos_key] = stack

    territory_moves = GameEngine._get_territory_processing_moves(state, 1)
    assert territory_moves, "Expected at least one territory move"

    eliminate_moves = [
        m for m in territory_moves if m.type == MoveType.ELIMINATE_RINGS_FROM_STACK
    ]
    assert eliminate_moves, "Expected an ELIMINATE_RINGS_FROM_STACK move"
    move = eliminate_moves[0]

    engine = DefaultRulesEngine()

    next_via_engine = GameEngine.apply_move(state, move)
    next_via_rules = engine.apply_move(state, move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping should also stay aligned.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status


@pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
def test_default_engine_apply_move_matches_game_engine_for_overtaking_capture_env(
    board_type: BoardType,
) -> None:
    """DefaultRulesEngine.apply_move matches GameEngine for env-driven captures.

    Starting from a fresh RingRiftEnv game, advance deterministically until
    a legal OVERTAKING_CAPTURE exists, then assert that applying that move
    via DefaultRulesEngine and GameEngine yields identical next states.

    This complements the synthetic scenario above by exercising the capture
    shadow contract under realistic orchestration and phase transitions.
    """
    env = RingRiftEnv(board_type)
    state = env.reset()
    engine = DefaultRulesEngine()

    move = None

    # Advance until we find an overtaking capture opportunity.
    for _ in range(200):
        moves = GameEngine.get_valid_moves(state, state.current_player)

        capture_moves = [m for m in moves if m.type == MoveType.OVERTAKING_CAPTURE]
        if capture_moves:
            move = capture_moves[0]
            break

        # Otherwise, advance the game using a simple deterministic policy
        # (place ring / skip placement / move stack) to explore the tree.
        progression_moves = [
            m
            for m in moves
            if m.type
            in (
                MoveType.PLACE_RING,
                MoveType.SKIP_PLACEMENT,
                MoveType.MOVE_STACK,
            )
        ]
        assert (
            progression_moves
        ), "Expected at least one non-capture move to advance the game"
        state = GameEngine.apply_move(state, progression_moves[0])
    else:
        raise AssertionError(
            "Failed to find an OVERTAKING_CAPTURE move within 200 plies"
        )

    assert move is not None

    next_via_engine = GameEngine.apply_move(state, move)
    next_via_rules = engine.apply_move(state, move)

    # Board-level equivalence
    assert next_via_rules.board.stacks == next_via_engine.board.stacks
    assert next_via_rules.board.markers == next_via_engine.board.markers
    assert (
        next_via_rules.board.collapsed_spaces
        == next_via_engine.board.collapsed_spaces
    )
    assert (
        next_via_rules.board.eliminated_rings
        == next_via_engine.board.eliminated_rings
    )

    # Player metadata equivalence
    assert next_via_rules.players == next_via_engine.players

    # Turn/phase/victory bookkeeping and chain-capture state should align.
    assert next_via_rules.current_player == next_via_engine.current_player
    assert next_via_rules.current_phase == next_via_engine.current_phase
    assert next_via_rules.game_status == next_via_engine.game_status
    assert next_via_rules.chain_capture_state == next_via_engine.chain_capture_state
