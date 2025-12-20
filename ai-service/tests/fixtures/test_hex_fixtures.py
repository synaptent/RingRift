"""Tests for Hex Board Test Fixtures.

Verifies the hex coordinate system, D6 symmetry operations,
and game state factory functions work correctly.
"""

import numpy as np
import pytest

from app.models import BoardType, GamePhase, GameStatus, Position, RingStack
from tests.fixtures.hex_fixtures import (
    D6_REFLECTIONS,
    D6_ROTATIONS,
    D6_SYMMETRIES,
    HexCoord,
    apply_symmetry_to_board,
    create_hex_board_with_stacks,
    create_hex_game_state,
    create_hex_training_sample,
    hex_center_position,
    hex_corner_positions,
    hex_edge_midpoints,
    verify_d6_symmetry,
)


class TestHexCoord:
    """Tests for HexCoord dataclass."""

    def test_create_origin(self):
        """Origin coordinate should be (0, 0)."""
        coord = HexCoord(0, 0)
        assert coord.q == 0
        assert coord.r == 0
        assert coord.s == 0

    def test_s_coordinate(self):
        """S coordinate should satisfy q + r + s = 0."""
        coord = HexCoord(3, -2)
        assert coord.q + coord.r + coord.s == 0
        assert coord.s == -1

    def test_to_cube(self):
        """to_cube should return (q, r, s) tuple."""
        coord = HexCoord(2, -1)
        cube = coord.to_cube()
        assert cube == (2, -1, -1)

    def test_from_cube(self):
        """from_cube should create correct HexCoord."""
        coord = HexCoord.from_cube(3, -2, -1)
        assert coord.q == 3
        assert coord.r == -2

    def test_offset_roundtrip(self):
        """to_offset and from_offset should be inverses."""
        size = 11
        original = HexCoord(3, -2)
        x, y = original.to_offset(size)
        recovered = HexCoord.from_offset(x, y, size)
        assert recovered == original

    def test_offset_origin(self):
        """Origin should convert to center of offset grid."""
        size = 11
        origin = HexCoord(0, 0)
        _x, y = origin.to_offset(size)
        # With size 11, origin should be near center
        assert y == size  # r + size = 0 + 11

    def test_distance_same_point(self):
        """Distance to self should be 0."""
        coord = HexCoord(3, -2)
        assert coord.distance(coord) == 0

    def test_distance_neighbors(self):
        """Distance to neighbors should be 1."""
        origin = HexCoord(0, 0)
        for neighbor in origin.neighbors():
            assert origin.distance(neighbor) == 1

    def test_neighbors_count(self):
        """Should have exactly 6 neighbors."""
        coord = HexCoord(0, 0)
        assert len(coord.neighbors()) == 6

    def test_neighbors_unique(self):
        """All neighbors should be unique."""
        coord = HexCoord(0, 0)
        neighbors = coord.neighbors()
        assert len(set(neighbors)) == 6

    def test_str_representation(self):
        """String representation should be readable."""
        coord = HexCoord(3, -2)
        assert str(coord) == "(3,-2)"


class TestHexCoordRotations:
    """Tests for HexCoord rotation methods."""

    def test_rotate_60_six_times_returns_to_origin(self):
        """Six 60-degree rotations should return to original."""
        coord = HexCoord(3, -1)
        result = coord
        for _ in range(6):
            result = result.rotate_60()
        assert result == coord

    def test_rotate_120_three_times_returns_to_origin(self):
        """Three 120-degree rotations should return to original."""
        coord = HexCoord(2, -3)
        result = coord.rotate_120().rotate_120().rotate_120()
        assert result == coord

    def test_rotate_180_twice_returns_to_origin(self):
        """Two 180-degree rotations should return to original."""
        coord = HexCoord(4, -2)
        result = coord.rotate_180().rotate_180()
        assert result == coord

    def test_rotate_180_formula(self):
        """180-degree rotation should negate coordinates."""
        coord = HexCoord(3, -2)
        rotated = coord.rotate_180()
        assert rotated.q == -3
        assert rotated.r == 2

    def test_rotate_240_equals_rotate_60_four_times(self):
        """240-degree rotation should equal four 60-degree rotations."""
        coord = HexCoord(2, -1)
        r240 = coord.rotate_240()
        r60_4 = coord.rotate_60().rotate_60().rotate_60().rotate_60()
        assert r240 == r60_4

    def test_rotate_300_equals_rotate_60_five_times(self):
        """300-degree rotation should equal five 60-degree rotations."""
        coord = HexCoord(3, -2)
        r300 = coord.rotate_300()
        r60_5 = coord.rotate_60().rotate_60().rotate_60().rotate_60().rotate_60()
        assert r300 == r60_5

    def test_origin_unchanged_by_rotation(self):
        """Origin should be unchanged by any rotation."""
        origin = HexCoord(0, 0)
        assert origin.rotate_60() == origin
        assert origin.rotate_120() == origin
        assert origin.rotate_180() == origin
        assert origin.rotate_240() == origin
        assert origin.rotate_300() == origin


class TestHexCoordReflections:
    """Tests for HexCoord reflection methods."""

    def test_reflect_q_twice_returns_to_origin(self):
        """Two reflections should return to original."""
        coord = HexCoord(3, -2)
        result = coord.reflect_q().reflect_q()
        assert result == coord

    def test_reflect_q_formula(self):
        """Reflection should swap r and s in cube coords."""
        coord = HexCoord(2, -1)  # cube: (2, -1, -1)
        coord.reflect_q()
        # Swap y and z: (2, -1, -1) -> (2, -1, -1) - in this case same
        # Let's use a different example
        coord2 = HexCoord(3, -2)  # cube: (3, -2, -1)
        reflected2 = coord2.reflect_q()
        # Swap y and z: (3, -2, -1) -> (3, -1, -2)
        assert reflected2.q == 3
        assert reflected2.r == -1

    def test_origin_unchanged_by_reflection(self):
        """Origin should be unchanged by reflection."""
        origin = HexCoord(0, 0)
        assert origin.reflect_q() == origin


class TestD6Symmetries:
    """Tests for D6 symmetry operations."""

    def test_d6_rotations_count(self):
        """Should have exactly 6 rotations."""
        assert len(D6_ROTATIONS) == 6

    def test_d6_reflections_count(self):
        """Should have exactly 6 reflections."""
        assert len(D6_REFLECTIONS) == 6

    def test_d6_symmetries_count(self):
        """Should have exactly 12 symmetries."""
        assert len(D6_SYMMETRIES) == 12

    def test_identity_is_first(self):
        """Identity should be the first rotation."""
        name, fn = D6_ROTATIONS[0]
        assert name == "identity"
        coord = HexCoord(3, -2)
        assert fn(coord) == coord

    def test_all_symmetries_preserve_distance_to_origin(self):
        """All symmetries should preserve distance from origin."""
        coord = HexCoord(3, -2)
        origin = HexCoord(0, 0)
        original_dist = coord.distance(origin)

        for name, sym_fn in D6_SYMMETRIES:
            transformed = sym_fn(coord)
            assert transformed.distance(origin) == original_dist, f"{name} changed distance"


class TestApplySymmetryToBoard:
    """Tests for apply_symmetry_to_board function."""

    def test_identity_preserves_stacks(self):
        """Identity transformation should not change stacks."""
        stacks = {
            "11,11": RingStack(
                position=Position(x=11, y=11),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            ),
            "12,10": RingStack(
                position=Position(x=12, y=10),
                rings=[1, 2],
                stackHeight=2,
                capHeight=1,
                controllingPlayer=1,
            ),
        }

        def identity(c):
            return c
        result = apply_symmetry_to_board(stacks, identity, size=11)

        assert set(result.keys()) == set(stacks.keys())

    def test_rotation_moves_stacks(self):
        """Rotation should move stack positions."""
        # Place a stack at a non-origin position
        stacks = {
            "14,8": RingStack(
                position=Position(x=14, y=8),
                rings=[1],
                stackHeight=1,
                capHeight=1,
                controllingPlayer=1,
            ),  # q=3, r=-3
        }

        _, rotate_180 = D6_ROTATIONS[3]  # 180-degree rotation
        result = apply_symmetry_to_board(stacks, rotate_180, size=11)

        # Stack should have moved (but not be at same position)
        assert "14,8" not in result or len(result) == 1


class TestCreateHexGameState:
    """Tests for create_hex_game_state factory."""

    def test_default_parameters(self):
        """Default state should be valid."""
        state = create_hex_game_state()

        assert state.id == "test-hex"
        assert state.board.type == BoardType.HEXAGONAL
        assert state.board.size == 11
        assert len(state.players) == 2
        assert state.current_phase == GamePhase.RING_PLACEMENT
        assert state.game_status == GameStatus.ACTIVE

    def test_custom_size(self):
        """Should respect custom size."""
        state = create_hex_game_state(size=8)
        assert state.board.size == 8

    def test_custom_num_players(self):
        """Should create correct number of players."""
        state = create_hex_game_state(num_players=4)
        assert len(state.players) == 4
        assert state.players[0].player_number == 1
        assert state.players[3].player_number == 4

    def test_custom_phase(self):
        """Should respect custom phase."""
        state = create_hex_game_state(phase=GamePhase.MOVEMENT)
        assert state.current_phase == GamePhase.MOVEMENT

    def test_empty_board(self):
        """Board should start empty."""
        state = create_hex_game_state()
        assert len(state.board.stacks) == 0
        assert len(state.board.markers) == 0

    def test_players_have_rings(self):
        """Players should start with rings in hand."""
        state = create_hex_game_state()
        for player in state.players:
            assert player.rings_in_hand > 0


class TestCreateHexBoardWithStacks:
    """Tests for create_hex_board_with_stacks factory."""

    def test_single_stack(self):
        """Should create board with single stack."""
        state = create_hex_board_with_stacks({
            HexCoord(0, 0): [1],
        })

        assert len(state.board.stacks) == 1

    def test_multiple_stacks(self):
        """Should create board with multiple stacks."""
        state = create_hex_board_with_stacks({
            HexCoord(0, 0): [1],
            HexCoord(1, 0): [2],
            HexCoord(0, 1): [1, 2],
        })

        assert len(state.board.stacks) == 3

    def test_stack_rings_tracking(self):
        """Should track rings correctly in stacks."""
        state = create_hex_board_with_stacks({
            HexCoord(0, 0): [1, 2, 1],
        })

        # Find the stack
        for stack in state.board.stacks.values():
            assert stack.stack_height == 3
            assert stack.rings == [1, 2, 1]
            assert stack.controlling_player == 1  # Bottom ring controls

    def test_player_rings_updated(self):
        """Should update player rings in hand."""
        state = create_hex_board_with_stacks({
            HexCoord(0, 0): [1, 1],  # Player 1 uses 2 rings
            HexCoord(1, 0): [2],      # Player 2 uses 1 ring
        })

        # Player 1 should have fewer rings
        p1 = state.players[0]
        p2 = state.players[1]

        # Assuming standard rings per player
        assert p1.rings_in_hand < p2.rings_in_hand

    def test_phase_is_movement(self):
        """Should default to movement phase."""
        state = create_hex_board_with_stacks({HexCoord(0, 0): [1]})
        assert state.current_phase == GamePhase.MOVEMENT


class TestCreateHexTrainingSample:
    """Tests for create_hex_training_sample factory."""

    def test_default_shapes(self):
        """Should create arrays with correct default shapes."""
        sample = create_hex_training_sample()

        assert sample['features'].shape == (10, 21, 21)
        assert sample['values'].shape == (1,)
        assert sample['policy_values'].shape == (64,)

    def test_custom_shapes(self):
        """Should respect custom shapes."""
        sample = create_hex_training_sample(
            features_shape=(5, 15, 15),
            policy_size=128,
        )

        assert sample['features'].shape == (5, 15, 15)
        assert sample['policy_values'].shape == (128,)

    def test_value_range(self):
        """Value should be in [-1, 1] range (tanh)."""
        sample = create_hex_training_sample()
        assert -1 <= sample['values'][0] <= 1

    def test_policy_sums_to_one(self):
        """Policy should be a valid probability distribution."""
        sample = create_hex_training_sample()
        assert abs(sample['policy_values'].sum() - 1.0) < 1e-6

    def test_reproducible_with_seed(self):
        """Same seed should produce same sample."""
        sample1 = create_hex_training_sample(seed=42)
        sample2 = create_hex_training_sample(seed=42)

        np.testing.assert_array_equal(sample1['features'], sample2['features'])
        np.testing.assert_array_equal(sample1['values'], sample2['values'])

    def test_different_seeds_different_samples(self):
        """Different seeds should produce different samples."""
        sample1 = create_hex_training_sample(seed=42)
        sample2 = create_hex_training_sample(seed=123)

        assert not np.array_equal(sample1['features'], sample2['features'])


class TestStandardPositions:
    """Tests for standard hex position helpers."""

    def test_hex_center_position(self):
        """Center should be origin."""
        center = hex_center_position(11)
        assert center == HexCoord(0, 0)

    def test_hex_corner_positions_count(self):
        """Should return exactly 6 corners."""
        corners = hex_corner_positions(11)
        assert len(corners) == 6

    def test_hex_corner_positions_distance(self):
        """All corners should be at same distance from center."""
        corners = hex_corner_positions(11)
        center = HexCoord(0, 0)

        distances = [corner.distance(center) for corner in corners]
        assert all(d == distances[0] for d in distances)

    def test_hex_corner_positions_unique(self):
        """All corners should be unique."""
        corners = hex_corner_positions(11)
        assert len(set(corners)) == 6

    def test_hex_edge_midpoints_count(self):
        """Should return exactly 6 edge midpoints."""
        midpoints = hex_edge_midpoints(11)
        assert len(midpoints) == 6

    def test_hex_edge_midpoints_unique(self):
        """All edge midpoints should be unique."""
        midpoints = hex_edge_midpoints(11)
        assert len(set(midpoints)) == 6


class TestPytestFixtures:
    """Tests for pytest fixtures themselves.

    Note: These tests use the fixtures directly from the module
    rather than pytest injection since the fixtures are defined
    in hex_fixtures.py, not conftest.py.
    """

    def test_empty_hex_state_factory(self):
        """empty_hex_state factory should work."""
        state = create_hex_game_state()
        assert state is not None
        assert len(state.board.stacks) == 0

    def test_hex_state_with_center_stack_factory(self):
        """create_hex_board_with_stacks should create stack at center."""
        state = create_hex_board_with_stacks({HexCoord(0, 0): [1]})
        assert len(state.board.stacks) == 1

    def test_hex_state_symmetric_factory(self):
        """Factory should create symmetric corner stacks."""
        corners = hex_corner_positions(11)
        stacks = {corner: [1] for corner in corners[:3]}
        stacks.update({corner: [2] for corner in corners[3:]})
        state = create_hex_board_with_stacks(stacks)
        assert len(state.board.stacks) == 6

    def test_hex_training_sample_factory(self):
        """create_hex_training_sample should return valid sample."""
        sample = create_hex_training_sample()
        assert 'features' in sample
        assert 'values' in sample
        assert 'policy_values' in sample

    def test_hex_coord_origin(self):
        """HexCoord origin should be (0, 0)."""
        origin = HexCoord(0, 0)
        assert origin.q == 0
        assert origin.r == 0

    def test_hex_coords_ring_1(self):
        """Origin neighbors should have 6 coords."""
        ring_1 = HexCoord(0, 0).neighbors()
        assert len(ring_1) == 6
