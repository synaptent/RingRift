from __future__ import annotations

import os
import sys
from typing import List, TYPE_CHECKING

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
    outside_height=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=25)
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

    for move in terr_moves:
        # All surfaced moves must be PROCESS_TERRITORY_REGION decisions.
        assert move.type == MoveType.PROCESS_TERRITORY_REGION
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

    move = terr_moves[0]
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


@given(
    cap_heights=st.lists(
        st.integers(min_value=0, max_value=4),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=25)
def test_territory_processing_elimination_surface_property(
    cap_heights: List[int],
) -> None:
    """Elimination decision surface for TERRITORY_PROCESSING.

    When no disconnected regions are eligible but the player controls stacks,
    _get_territory_processing_moves must surface one
    ELIMINATE_RINGS_FROM_STACK move per stack with positive capHeight.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.TERRITORY_PROCESSING
    state.current_player = 1

    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()
    board.markers.clear()

    base_positions = [
        Position(x=1, y=1),
        Position(x=2, y=2),
        Position(x=3, y=3),
        Position(x=4, y=4),
    ]
    positions = base_positions[: len(cap_heights)]

    for pos_, height in zip(positions, cap_heights):
        board.stacks[pos_.to_key()] = RingStack(
            position=pos_,
            rings=[1] * max(1, height),
            stackHeight=max(1, height),
            capHeight=height,
            controllingPlayer=1,
        )

    # Stub disconnected-region detection so that no regions are reported;
    # this isolates the elimination-only branch of
    # GameEngine._get_territory_processing_moves.
    orig_find_disconnected = BoardManager.find_disconnected_regions
    try:
        BoardManager.find_disconnected_regions = staticmethod(
            lambda board, player_number: []
        )
        moves = GameEngine._get_territory_processing_moves(state, 1)

        # All surfaced moves must be ELIMINATE_RINGS_FROM_STACK.
        assert all(m.type == MoveType.ELIMINATE_RINGS_FROM_STACK for m in moves)

        expected_keys = {
            key
            for key, stack in board.stacks.items()
            if stack.controlling_player == 1 and stack.cap_height > 0
        }
        target_keys = {m.to.to_key() for m in moves if m.to is not None}
        assert target_keys == expected_keys
    finally:
        BoardManager.find_disconnected_regions = orig_find_disconnected


def _block_all_positions_except(
    state: "GameState",
    allowed: List[Position],
) -> None:
    """Collapse all non-stack cells to block movement or capture for
    the player."""
    board = state.board
    allowed_keys = {p.to_key() for p in allowed}
    size = board.size or 0

    for x in range(size):
        for y in range(size):
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key in allowed_keys:
                continue
            if key in board.stacks:
                continue
            board.collapsed_spaces[key] = 2


@given(
    cap_heights=st.lists(
        st.integers(min_value=1, max_value=4),
        min_size=2,
        max_size=4,
    )
)
@settings(max_examples=25)
def test_forced_elimination_min_cap_property(
    cap_heights: List[int],
) -> None:
    """Forced-elimination selection prefers smallest positive capHeight.

    For random blocked states where Player 1 controls several pure-cap stacks,
    has_forced_elimination_action must hold and
    apply_forced_elimination_for_player must eliminate from one of the stacks
    with minimum capHeight.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    p1 = next(p for p in state.players if p.player_number == 1)
    p1.rings_in_hand = 0

    board = state.board
    board.stacks.clear()
    board.collapsed_spaces.clear()

    # Place stacks for Player 1 at fixed distinct positions on the board.
    base_positions = [
        Position(x=0, y=0),
        Position(x=2, y=2),
        Position(x=4, y=4),
        Position(x=6, y=6),
    ]
    positions = base_positions[: len(cap_heights)]

    for pos_, height in zip(positions, cap_heights):
        board.stacks[pos_.to_key()] = RingStack(
            position=pos_,
            rings=[1] * height,
            stackHeight=height,
            capHeight=height,
            controllingPlayer=1,
        )

    _block_all_positions_except(state, positions)
    GameEngine.clear_cache()

    # Preconditions: player is blocked with stacks but has no other actions.
    assert ga.has_forced_elimination_action(state, 1) is True

    moves = GameEngine._get_forced_elimination_moves(state, 1)
    assert moves, "expected forced-elimination moves"

    keys_from_moves = {m.to.to_key() for m in moves if m.to is not None}
    keys_from_board = {p.to_key() for p in positions}
    assert keys_from_moves == keys_from_board

    # The public get_valid_moves surface should also expose only
    # FORCED_ELIMINATION actions targeting exactly these stacks.
    gvm_moves = GameEngine.get_valid_moves(state, 1)
    assert gvm_moves, "expected forced-elimination moves from get_valid_moves"
    assert all(m.type == MoveType.FORCED_ELIMINATION for m in gvm_moves)
    gvm_targets = {m.to.to_key() for m in gvm_moves if m.to is not None}
    assert gvm_targets == keys_from_board

    before_total = state.total_rings_eliminated
    outcome = ga.apply_forced_elimination_for_player(state, 1)
    assert outcome is not None
    assert outcome.eliminated_from is not None
    eliminated_key = outcome.eliminated_from.to_key()

    min_cap = min(cap_heights)
    candidate_keys = {
        pos.to_key()
        for pos, height in zip(positions, cap_heights)
        if height == min_cap
    }
    assert eliminated_key in candidate_keys

    after_total = state.total_rings_eliminated
    assert after_total >= before_total + 1
    assert outcome.eliminated_count >= 1
