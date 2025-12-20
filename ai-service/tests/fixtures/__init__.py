"""Test fixtures for RingRift AI service.

This package contains reusable test fixtures for different board types
and game scenarios.
"""

from tests.fixtures.hex_fixtures import (
    # Coordinate class
    HexCoord,
    # Symmetry operations
    D6_ROTATIONS,
    D6_REFLECTIONS,
    D6_SYMMETRIES,
    apply_symmetry_to_board,
    apply_symmetry_to_state,
    verify_d6_symmetry,
    # Factory functions
    create_hex_game_state,
    create_hex_board_with_stacks,
    create_hex_training_sample,
    # Utility functions
    hex_center_position,
    hex_corner_positions,
    hex_edge_midpoints,
    # Pytest fixtures
    empty_hex_state,
    empty_hex8_state,
    hex_state_with_center_stack,
    hex_state_symmetric,
    hex_training_sample,
    hex_coord_origin,
    hex_coords_ring_1,
)
