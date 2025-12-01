"""Tests for loading 3-player and 4-player evaluation pools."""
import os
import pytest
from app.models import BoardType
from app.training.eval_pools import load_state_pool, POOL_PATHS


def test_load_square19_3p_pool():
    """Verify that the 3-player Square19 pool can be loaded and has correct player counts."""
    pool_id = "3p_v1"
    board_type = BoardType.SQUARE19
    
    # Ensure the file exists (skip test if pool hasn't been generated yet)
    path = POOL_PATHS.get((board_type, pool_id))
    if not path or not os.path.exists(path):
        pytest.skip(f"Pool file not found: {path}. Run generation command in data/eval_pools/square19_3p/README.md")

    states = load_state_pool(board_type, pool_id=pool_id, num_players=3)
    assert len(states) > 0, "Pool should contain at least one state"
    
    for state in states:
        assert state.board_type == board_type, f"Expected {board_type}, got {state.board_type}"
        assert len(state.players) == 3, f"Expected 3 players, got {len(state.players)}"


def test_load_hex_4p_pool():
    """Verify that the 4-player Hex pool can be loaded and has correct player counts."""
    pool_id = "4p_v1"
    board_type = BoardType.HEXAGONAL
    
    # Ensure the file exists (skip test if pool hasn't been generated yet)
    path = POOL_PATHS.get((board_type, pool_id))
    if not path or not os.path.exists(path):
        pytest.skip(f"Pool file not found: {path}. Run generation command in data/eval_pools/hex_4p/README.md")

    states = load_state_pool(board_type, pool_id=pool_id, num_players=4)
    assert len(states) > 0, "Pool should contain at least one state"
    
    for state in states:
        assert state.board_type == board_type, f"Expected {board_type}, got {state.board_type}"
        assert len(state.players) == 4, f"Expected 4 players, got {len(state.players)}"