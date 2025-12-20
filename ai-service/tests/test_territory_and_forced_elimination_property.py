from __future__ import annotations

import os
import sys
from typing import List, TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure app package is importable when running tests directly from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    GamePhase,
    GameStatus,
    Position,
    RingStack,
    MoveType,
    Move,
)
from app.models.core import MarkerInfo  # noqa: E402
from app.game_engine import GameEngine  # noqa: E402
from app.board_manager import BoardManager  # noqa: E402
from app.rules import global_actions as ga  # noqa: E402
from tests.rules.helpers import _make_base_game_state  # noqa: E402

if TYPE_CHECKING:
    from app.models import GameState  # noqa: F401


def _build_q23_region_state(
    internal_heights: List[int],
    outside_height: int,
) -> tuple["GameState", List[Position], int]:
    """Construct a Q23-style 2×2 disconnected region on square8 for P1.

    Geometry mirrors the TS territoryProcessing.property harness but fixes
    the region position; randomness comes from internal stack heights and the
    outside self-elimination stack height.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.TERRITORY_PROCESSING
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()
    board.markers.clear()

    # Fixed interior 2×2 region with a one-cell margin inside the 8×8 board.
    region_positions = [
        Position(x=2, y=2),
        Position(x=3, y=2),
        Position(x=2, y=3),
        Position(x=3, y=3),
    ]

    heights = internal_heights[:4]
    assert len(heights) == len(region_positions)

    expected_internal = 0
    for pos_, height in zip(region_positions, heights):
        board.stacks[pos_.to_key()] = RingStack(
            position=pos_,
            rings=[2] * height,
            stackHeight=height,
            capHeight=height,
            controllingPlayer=2,
        )
        expected_internal += height

    # Simple rectangular ring of markers one cell around the region, matching
    # the Q23 mini-region fixtures used on the TS side.
    border_coords: List[tuple[int, int]] = []
    for x in range(1, 5):
        border_coords.append((x, 1))
        border_coords.append((x, 4))
    for y in range(2, 4):
        border_coords.append((1, y))
        border_coords.append((4, y))
    for x, y in border_coords:
        marker_pos = Position(x=x, y=y)
        key = marker_pos.to_key()
        board.markers[key] = MarkerInfo(
            player=1,
            position=marker_pos,
            type="regular",
        )

    # Outside stack for Player 1 used for the self-elimination prerequisite.
    outside_pos = Position(x=0, y=0)
    outside_key = outside_pos.to_key()
    board.stacks[outside_key] = RingStack(
        position=outside_pos,
        rings=[1] * outside_height,
        stackHeight=outside_height,
        capHeight=outside_height,
        controllingPlayer=1,
    )

    # Reset player statistics to a known baseline.
    p1 = next(p for p in state.players if p.player_number == 1)
    p2 = next(p for p in state.players if p.player_number == 2)
    p1.rings_in_hand = 0
    p2.rings_in_hand = 0
    p1.territory_spaces = 0
    p2.territory_spaces = 0
    p1.eliminated_rings = 0
    p2.eliminated_rings = 0
    state.total_rings_eliminated = 0

    return state, region_positions, expected_internal


@given(
    internal_heights=st.lists(
        st.integers(min_value=1, max_value=3),
        min_size=4,
        max_size=4,
    ),
    # RR-CANON-R082/R145: Any controlled stack is eligible, including height-1
    outside_height=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=25, deadline=None)
def test_territory_processing_q23_region_property(
    internal_heights: List[int],
    outside_height: int,
) -> None:
    """Territory-processing invariants for random Q23-style 2×2 regions.

    For random internal stack heights and outside self-elimination stack
    height, GameEngine._get_territory_processing_moves must surface exactly
    one PROCESS_TERRITORY_REGION move and _apply_territory_claim must:
    - remove all stacks inside the region,
    - collapse region spaces to Player 1,
    - credit all internal eliminations to Player 1, and
    - keep collapsed-space counts monotone.

    Note: Per RR-CANON-R082/R145, any controlled stack (including height-1
    standalone rings) is eligible for territory processing self-elimination.
    """
    state, region_positions, expected_internal = _build_q23_region_state(
        internal_heights,
        outside_height,
    )
    board = state.board

    p1 = next(p for p in state.players if p.player_number == 1)
    initial_territory = p1.territory_spaces
    initial_eliminated = p1.eliminated_rings
    initial_total = state.total_rings_eliminated
    initial_collapsed = len(board.collapsed_spaces)

    terr_moves = GameEngine._get_territory_processing_moves(state, 1)
    assert terr_moves, "expected territory-processing moves"

    # Filter to only territory option moves (exclude SKIP_TERRITORY_PROCESSING which is
    # a valid optional choice per RR-CANON-R075 but not what we're testing here).
    option_moves = [
        m for m in terr_moves
        if m.type in (MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION)
    ]
    assert option_moves, "expected at least one territory option move"

    for move in option_moves:
        assert move.disconnected_regions
        region = list(move.disconnected_regions)[0]
        region_spaces = list(region.spaces)
        # Region geometry must match the constructed Q23-style region, but the
        # ordering of spaces is not significant.
        assert sorted(region_spaces, key=lambda p: (p.x, p.y)) == sorted(
            region_positions,
            key=lambda p: (p.x, p.y),
        )
        assert move.to in region_spaces

    move = option_moves[0]
    GameEngine._apply_territory_claim(state, move)

    p1_after = next(p for p in state.players if p.player_number == 1)
    board_after = state.board

    # Region stacks removed and collapsed to Player 1.
    for pos_ in region_positions:
        key = pos_.to_key()
        assert board_after.stacks.get(key) is None
        assert board_after.collapsed_spaces.get(key) == 1

    # TerritorySpaces increases by the region plus border size.
    delta_territory = p1_after.territory_spaces - initial_territory
    # Reconstruct the border ring used in _build_q23_region_state so we do not
    # need to change its return type.
    border_coords: List[tuple[int, int]] = []
    for x in range(1, 5):
        border_coords.append((x, 1))
        border_coords.append((x, 4))
    for y in range(2, 4):
        border_coords.append((1, y))
        border_coords.append((4, y))
    expected_territory_gain = len(region_positions) + len(border_coords)
    assert delta_territory == expected_territory_gain

    # Internal eliminations credited consistently.
    delta_elim_p1 = p1_after.eliminated_rings - initial_eliminated
    delta_total = state.total_rings_eliminated - initial_total
    assert delta_elim_p1 == expected_internal
    assert delta_total == expected_internal

    # Collapsed spaces are monotone.
    assert len(board_after.collapsed_spaces) >= initial_collapsed


def test_territory_and_forced_elimination_moves_rejected_outside_phases(
) -> None:
    """Territory/FE moves must not be applied outside their dedicated phases.

    This asserts that the engine-level phase→MoveType invariant matches the
    TS-side [PHASE_MOVE_INVARIANT] contract for:

    - PROCESS_TERRITORY_REGION moves (territory_processing only).
    - FORCED_ELIMINATION moves (forced_elimination phase only).
    """
    from datetime import datetime

    # Base square8 state for two players.
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_player = 1

    now = datetime.now()

    # PROCESS_TERRITORY_REGION is only valid during TERRITORY_PROCESSING.
    bad_territory_move = Move(
        id="bad-territory",
        type=MoveType.PROCESS_TERRITORY_REGION,
        player=1,
        timestamp=now,
        think_time=0,
        move_number=0,
    )

    state.current_phase = GamePhase.MOVEMENT
    with pytest.raises(RuntimeError) as terr_exc:
        GameEngine.apply_move(state, bad_territory_move)
    assert "Phase/move invariant violated" in str(terr_exc.value)

    # FORCED_ELIMINATION is only valid during the FORCED_ELIMINATION phase.
    bad_fe_move = Move(
        id="bad-fe",
        type=MoveType.FORCED_ELIMINATION,
        player=1,
        timestamp=now,
        think_time=0,
        move_number=1,
    )

    state.current_phase = GamePhase.MOVEMENT
    with pytest.raises(RuntimeError) as fe_exc:
        GameEngine.apply_move(state, bad_fe_move)
    assert "Phase/move invariant violated" in str(fe_exc.value)


def test_no_territory_action_then_forced_elimination_phase_transition(
) -> None:
    """ANM / FE gating after territory_processing for square8 2p.

    Construct a square8 state where:

    - Player 1 controls at least one stack on the board.
    - The entire turn history for Player 1 consists only of NO_*_ACTION moves
      (no interactive actions in any phase).
    - The last move is NO_TERRITORY_ACTION in TERRITORY_PROCESSING.

    After applying that move, the Python phase machine must:

    - Transition into the FORCED_ELIMINATION phase for Player 1, and
    - Offer only FORCED_ELIMINATION moves, mirroring the TS
      onTerritoryProcessingComplete + post-move FSM semantics.
    """
    from datetime import datetime

    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_player = 1
    state.current_phase = GamePhase.TERRITORY_PROCESSING

    board = state.board

    # Set up a stack for Player 1 so the ANM/FE gating logic applies.
    # Place a blocked stack (surrounded by collapsed spaces) to ensure
    # no movement or capture actions are available.
    p1_stack_pos = Position(x=3, y=3)
    board.stacks[p1_stack_pos.to_key()] = RingStack(
        position=p1_stack_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )

    # Block surrounding cells to prevent movement/capture options.
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            block_pos = Position(x=3 + dx, y=3 + dy)
            board.collapsed_spaces[block_pos.to_key()] = 2

    # Set rings_in_hand to 0 so player has no placements available.
    # This ensures the ANM/FE gating triggers properly.
    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    # Ensure Player 1 has at least one stack on the board.
    assert any(s.controlling_player == 1 for s in board.stacks.values())

    # Clear any existing history and seed the turn with bookkeeping NO_* moves.
    state.move_history = []
    now = datetime.now()

    def _mk_no_move(mtype: MoveType, num: int) -> Move:
        return Move(
            id=f"no-{mtype.value}-{num}",
            type=mtype,
            player=1,
            timestamp=now,
            think_time=0,
            move_number=num,
        )

    # Visit placement, movement, and line phases with forced no-ops only.
    state.move_history.append(_mk_no_move(MoveType.NO_PLACEMENT_ACTION, 0))
    state.move_history.append(_mk_no_move(MoveType.NO_MOVEMENT_ACTION, 1))
    state.move_history.append(_mk_no_move(MoveType.NO_LINE_ACTION, 2))

    # Sanity: the engine-side helper must treat this turn as having had no
    # interactive actions so far.
    from app.rules import phase_machine as pm  # noqa: E402

    assert pm.compute_had_any_action_this_turn(state) is False

    # Final bookkeeping move in TERRITORY_PROCESSING.
    no_territory = _mk_no_move(MoveType.NO_TERRITORY_ACTION, 3)

    # Apply the move via the full engine, which will delegate phase/turn
    # transitions to the shared phase_machine.advance_phases helper.
    new_state = GameEngine.apply_move(state, no_territory)

    # The next phase for Player 1 must be FORCED_ELIMINATION.
    assert new_state.current_phase == GamePhase.FORCED_ELIMINATION
    assert new_state.current_player == 1

    # In the forced_elimination phase, only FORCED_ELIMINATION moves are legal.
    fe_moves = GameEngine.get_valid_moves(new_state, new_state.current_player)
    assert fe_moves
    assert all(m.type == MoveType.FORCED_ELIMINATION for m in fe_moves)
