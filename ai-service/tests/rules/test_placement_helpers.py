import os
import sys

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # noqa: E402
    Position,
    RingStack,
)
from app.game_engine import GameEngine  # noqa: E402
from app.rules.placement import (  # noqa: E402
    PlacementContextPy,
    validate_placement_on_board_py,
    evaluate_skip_placement_eligibility_py,
    apply_place_ring_py,
)
from app.rules.core import BOARD_CONFIGS  # noqa: E402
from tests.rules.helpers import (  # noqa: E402
    _make_base_game_state,
    _make_place_ring_move,
)


def test_validate_placement_on_board_py_accepts_basic_empty_cell() -> None:
    """
    Structurally legal placement on an empty cell is accepted.

    Mirrors TS PlacementAggregate.validatePlacementOnBoard semantics for
    a simple square8 start position with rings available.
    """
    state = _make_base_game_state()
    board = state.board
    board_type = board.type
    board_config = BOARD_CONFIGS[board_type]

    player = state.current_player
    player_obj = next(p for p in state.players if p.player_number == player)

    ctx = PlacementContextPy(
        board_type=board_type,
        player=player,
        rings_in_hand=player_obj.rings_in_hand,
        rings_per_player_cap=board_config.rings_per_player,
    )

    pos = Position(x=0, y=0)
    result = validate_placement_on_board_py(board, pos, 1, ctx)

    assert result.valid is True
    assert result.max_placement_count >= 1


def test_validate_placement_on_board_py_respects_ring_cap() -> None:
    """
    Per-player ring cap prevents further placements once reached.

    Aligns helper semantics with TS ring-cap enforcement
    (RR-CANON-R020).
    """
    state = _make_base_game_state()
    board = state.board
    board_type = board.type
    board_config = BOARD_CONFIGS[board_type]

    player = state.current_player
    per_player_cap = board_config.rings_per_player

    pos = Position(x=1, y=1)
    pos_key = pos.to_key()

    # Stack containing exactly cap rings for this player.
    rings = [player] * per_player_cap
    stack = RingStack(
        position=pos,
        rings=rings,
        stackHeight=len(rings),
        capHeight=len(rings),
        controllingPlayer=player,
    )
    board.stacks[pos_key] = stack

    # Player still has rings in hand, but own-colour rings in play
    # already reach the per-player cap.
    player_obj = next(p for p in state.players if p.player_number == player)
    player_obj.rings_in_hand = 1

    ctx = PlacementContextPy(
        board_type=board_type,
        player=player,
        rings_in_hand=player_obj.rings_in_hand,
        rings_per_player_cap=per_player_cap,
    )

    result = validate_placement_on_board_py(board, pos, 1, ctx)

    assert result.valid is False
    assert result.max_placement_count == 0
    assert result.error_code == "NO_RINGS_AVAILABLE"


def test_skip_eligibility_requires_controlled_stack() -> None:
    """
    Skip-placement is ineligible when the player controls no stacks.

    Mirrors TS evaluateSkipPlacementEligibility NO_CONTROLLED_STACKS
    path.
    """
    state = _make_base_game_state()
    board = state.board
    board.stacks.clear()

    player = state.current_player
    result = evaluate_skip_placement_eligibility_py(state, player)

    assert result.eligible is False
    assert result.code == "NO_CONTROLLED_STACKS"


def test_skip_eligibility_true_when_stack_has_action() -> None:
    """
    Skip-placement is eligible when a controlled stack has a legal
    action.
    """
    state = _make_base_game_state()
    board = state.board

    player = state.current_player
    pos = Position(x=0, y=0)
    pos_key = pos.to_key()

    stack = RingStack(
        position=pos,
        rings=[player],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=player,
    )
    board.stacks[pos_key] = stack

    result = evaluate_skip_placement_eligibility_py(state, player)

    assert result.eligible is True
    assert result.code is None


def test_skip_eligibility_rejects_when_no_rings_in_hand() -> None:
    """
    Skip-placement is not allowed when the player has zero rings in hand.
    `no_placement_action` must be recorded instead.
    """
    state = _make_base_game_state()
    board = state.board

    player = state.current_player
    pos = Position(x=0, y=0)
    pos_key = pos.to_key()

    stack = RingStack(
        position=pos,
        rings=[player],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=player,
    )
    board.stacks[pos_key] = stack

    # Exhaust rings in hand for the active player
    for p in state.players:
        if p.player_number == player:
            p.rings_in_hand = 0

    result = evaluate_skip_placement_eligibility_py(state, player)

    assert result.eligible is False
    assert result.code == "NO_RINGS_IN_HAND"


def test_skip_eligibility_rejects_when_no_legal_actions(monkeypatch) -> None:
    """
    Skip-placement is rejected when stacks exist but have no actions.

    Uses a monkeypatch of
    GameEngine._has_any_legal_move_or_capture_from_on_board to force
    the NO_LEGAL_ACTIONS path.
    """
    state = _make_base_game_state()
    board = state.board

    player = state.current_player
    pos = Position(x=0, y=0)
    pos_key = pos.to_key()

    stack = RingStack(
        position=pos,
        rings=[player],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=player,
    )
    board.stacks[pos_key] = stack

    monkeypatch.setattr(
        GameEngine,
        "_has_any_legal_move_or_capture_from_on_board",
        staticmethod(lambda *args, **kwargs: False),
        raising=False,
    )

    result = evaluate_skip_placement_eligibility_py(state, player)

    assert result.eligible is False
    assert result.code == "NO_LEGAL_ACTIONS"


def test_apply_place_ring_py_matches_game_engine() -> None:
    """
    apply_place_ring_py produces same board+players as GameEngine.
    """
    base_state = _make_base_game_state()
    helper_state = _make_base_game_state()

    move = _make_place_ring_move(
        player=base_state.current_player,
        x=0,
        y=0,
        placement_count=2,
    )

    engine_next = GameEngine.apply_move(base_state, move)

    outcome = apply_place_ring_py(helper_state, move)
    next_state = outcome.next_state

    assert next_state.board.stacks == engine_next.board.stacks
    assert next_state.board.markers == engine_next.board.markers
    assert (
        next_state.board.collapsed_spaces
        == engine_next.board.collapsed_spaces
    )
    assert (
        next_state.board.eliminated_rings
        == engine_next.board.eliminated_rings
    )

    assert next_state.players == engine_next.players

    expected_applied = move.placement_count or 1
    assert outcome.applied_count == expected_applied
