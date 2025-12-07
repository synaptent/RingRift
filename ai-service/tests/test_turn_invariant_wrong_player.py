import os

import pytest

from app.game_engine import GameEngine
from app.models.core import Move, MoveType, Position
from app.training.env import RingRiftEnv


@pytest.mark.skipif(
    os.environ.get("RINGRIFT_SKIP_PHASE_INVARIANT", "0") in {"1", "true", "yes", "on"},
    reason="Test requires phase/player invariant enforcement to be enabled",
)
def test_apply_move_rejects_wrong_player():
    """Ensure apply_move enforces move.player == current_player."""
    env = RingRiftEnv(board_type="square8", num_players=2)
    state = env.reset()
    assert state.current_player == 1

    bad_move = Move(
        id="mis-attributed",
        type=MoveType.PLACE_RING,
        player=2,  # wrong actor
        to=Position(x=0, y=0),
        timestamp=state.last_move_at,
        thinkTime=0,
        moveNumber=1,
    )

    try:
        GameEngine.apply_move(state, bad_move, trace_mode=True)
    except ValueError as exc:
        assert "current player" in str(exc)
    else:
        raise AssertionError("apply_move should reject moves by non-current player")
