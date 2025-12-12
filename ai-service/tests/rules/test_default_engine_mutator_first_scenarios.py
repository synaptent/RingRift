import os
import sys
from datetime import datetime

import pytest

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    BoardType,
    GamePhase,
    GameState,
    Move,
    MoveType,
    Position,
    RingStack,
    Territory,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.default_engine import DefaultRulesEngine  # noqa: E402
from app.training.env import RingRiftEnv  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402
from tests.rules.helpers import _make_base_game_state  # noqa: E402


@pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
def test_mutator_first_env_smoke_for_place_ring_and_move_stack(
    board_type: BoardType,
) -> None:
    """Simple env-driven smoke test for mutator-first mode.

    This test ensures that, for basic PLACE_RING and MOVE_STACK moves
    discovered via RingRiftEnv, DefaultRulesEngine constructed with
    ``mutator_first=True``:

    - Does not raise RuntimeError from its mutator-first shadow path.
    - Returns a state that matches ``GameEngine.apply_move`` on all
      board/turn bookkeeping fields we care about.
    """
    env = RingRiftEnv(board_type)
    state = env.reset()
    engine = DefaultRulesEngine(mutator_first=True)

    # --- First: a concrete PLACE_RING move ---
    moves = GameEngine.get_valid_moves(state, state.current_player)
    place_moves = [m for m in moves if m.type == MoveType.PLACE_RING]
    assert place_moves, "Expected at least one PLACE_RING move from env"
    place_move = place_moves[0]

    via_engine = GameEngine.apply_move(state, place_move)
    via_rules = engine.apply_move(state, place_move)

    assert via_rules.board.stacks == via_engine.board.stacks
    assert via_rules.board.markers == via_engine.board.markers
    assert via_rules.board.collapsed_spaces == via_engine.board.collapsed_spaces
    assert via_rules.board.eliminated_rings == via_engine.board.eliminated_rings
    assert via_rules.players == via_engine.players
    assert via_rules.current_player == via_engine.current_player
    assert via_rules.current_phase == via_engine.current_phase
    assert via_rules.game_status == via_engine.game_status

    # --- Then: advance deterministically until a MOVE_STACK is found ---
    state = via_engine
    for _ in range(30):
        moves = GameEngine.get_valid_moves(state, state.current_player)
        stack_moves = [m for m in moves if m.type == MoveType.MOVE_STACK]
        if stack_moves:
            move_stack = stack_moves[0]
            via_engine = GameEngine.apply_move(state, move_stack)
            via_rules = engine.apply_move(state, move_stack)

            assert via_rules.board.stacks == via_engine.board.stacks
            assert via_rules.board.markers == via_engine.board.markers
            assert (
                via_rules.board.collapsed_spaces
                == via_engine.board.collapsed_spaces
            )
            assert (
                via_rules.board.eliminated_rings
                == via_engine.board.eliminated_rings
            )
            assert via_rules.players == via_engine.players
            assert via_rules.current_player == via_engine.current_player
            assert via_rules.current_phase == via_engine.current_phase
            assert via_rules.game_status == via_engine.game_status
            break

        # Advance using placement / skip moves if no MOVE_STACK yet.
        progression_moves = [
            m
            for m in moves
            if m.type
            in (
                MoveType.PLACE_RING,
                MoveType.SKIP_PLACEMENT,
            )
        ]
        assert progression_moves, (
            "Expected at least one placement/skip move while searching "
            "for a MOVE_STACK in mutator-first smoke test"
        )
        state = GameEngine.apply_move(state, progression_moves[0])
    else:
        pytest.skip("Could not find a MOVE_STACK move within 30 plies")


def test_mutator_first_choose_territory_option_synthetic() -> None:
    """Mutator-first territory processing matches canonical semantics.

    This mirrors the synthetic territory scenario from
    ``test_default_engine_apply_move_matches_game_engine_for_choose_territory_option_synthetic``,
    but runs with ``mutator_first=True`` enabled to ensure the full
    mutator-driven orchestration path agrees with ``GameEngine.apply_move``
    for CHOOSE_TERRITORY_OPTION moves (including the downstream
    forced-elimination and territory bookkeeping).
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

        territory_moves = GameEngine._get_territory_processing_moves(state, 1)
        assert territory_moves, "Expected at least one territory move"

        process_region_moves = [m for m in territory_moves if m.type == MoveType.CHOOSE_TERRITORY_OPTION]
        assert process_region_moves, (
            "Expected at least one CHOOSE_TERRITORY_OPTION move from "
            "_get_territory_processing_moves in mutator-first test"
        )
        move = process_region_moves[0]

        engine = DefaultRulesEngine(mutator_first=True)

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
        assert (
            next_via_rules.current_player
            == next_via_engine.current_player
        )
        assert (
            next_via_rules.current_phase
            == next_via_engine.current_phase
        )
        assert next_via_rules.game_status == next_via_engine.game_status

    finally:
        BoardManager.find_disconnected_regions = (  # type: ignore[assignment]
            orig_find_regions
        )
        BoardManager.get_border_marker_positions = (  # type: ignore[assignment]
            orig_get_border
        )
