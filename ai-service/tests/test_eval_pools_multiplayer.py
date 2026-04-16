"""Tests for multiplayer evaluation pool loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models import BoardType
from app.training.eval_pools import POOL_PATHS, load_state_pool


def test_load_square19_3p_pool_from_fixture(tmp_path: Path):
    """Verify that a configured 3-player Square19 pool loads with the expected player count."""
    pool_id = "3p_v1"
    board_type = BoardType.SQUARE19
    pool_path = tmp_path / "square19_3p_pool.jsonl"
    pool_path.write_text('{"fixture": 1}\n{"fixture": 2}\n', encoding="utf-8")

    states = []
    for _ in range(2):
        state = MagicMock()
        state.board_type = board_type
        state.players = [object(), object(), object()]
        states.append(state)

    patched_paths = {(board_type, pool_id): str(pool_path)}
    with patch.dict(POOL_PATHS, patched_paths, clear=False):
        with patch("app.training.eval_pools.GameState") as mock_game_state:
            mock_game_state.model_validate_json.side_effect = states
            loaded_states = load_state_pool(board_type, pool_id=pool_id, num_players=3)

    assert len(loaded_states) == 2
    for state in loaded_states:
        assert state.board_type == board_type, f"Expected {board_type}, got {state.board_type}"
        assert len(state.players) == 3, f"Expected 3 players, got {len(state.players)}"
