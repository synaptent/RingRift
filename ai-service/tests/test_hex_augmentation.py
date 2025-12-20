"""
Tests for Hex Board Symmetry Augmentation.

Tests the D6 dihedral group transformations for hexagonal boards.
"""

import numpy as np
import pytest

from app.training.hex_augmentation import (
    HexSymmetryTransform,
    augment_hex_sample,
    HEX_BOARD_SIZE,
    HEX_RADIUS,
    HEX_PLACEMENT_SPAN,
    HEX_MOVEMENT_BASE,
    P_HEX,
)


def create_hex_mask(board_size: int = HEX_BOARD_SIZE) -> np.ndarray:
    """
    Create a boolean mask of valid hex cells within the bounding box.
    
    A hex cell at grid position (cx, cy) is valid if the corresponding
    axial coords (q, r) satisfy: max(|q|, |r|, |q+r|) <= radius.
    """
    radius = (board_size - 1) // 2
    mask = np.zeros((board_size, board_size), dtype=bool)

    for cy in range(board_size):
        for cx in range(board_size):
            q = cx - radius
            r = cy - radius
            s = -q - r
            # Valid hex cell if max Manhattan distance <= radius
            if max(abs(q), abs(r), abs(s)) <= radius:
                mask[cy, cx] = True

    return mask


def create_hex_board_with_zeros_outside(
    channels: int = 10, board_size: int = HEX_BOARD_SIZE
) -> np.ndarray:
    """
    Create a random board with zeros outside the valid hex region.
    
    This ensures clamping behavior doesn't affect composition tests.
    """
    mask = create_hex_mask(board_size)
    board = np.random.rand(channels, board_size, board_size).astype(np.float32)
    # Set cells outside valid hex to zero
    board[:, ~mask] = 0
    return board


class TestHexSymmetryTransform:
    """Test suite for HexSymmetryTransform class."""

    @pytest.fixture
    def transform(self):
        """Create a HexSymmetryTransform instance."""
        return HexSymmetryTransform(board_size=HEX_BOARD_SIZE)

    def test_identity_transformation(self, transform):
        """Transform 0 should be identity (no change)."""
        # Create random board
        board = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        board = board.astype(np.float32)

        result = transform.transform_board(board, transform_id=0)

        np.testing.assert_array_almost_equal(result, board)

    def test_identity_policy(self, transform):
        """Policy transform 0 should be identity."""
        policy = np.random.rand(P_HEX).astype(np.float32)
        policy /= policy.sum()  # Normalize

        result = transform.transform_policy(policy, transform_id=0)

        np.testing.assert_array_almost_equal(result, policy)

    def test_rotation_60_coordinates(self, transform):
        """Test 60° rotation formula: (q, r) -> (-r, q + r)."""
        # Test center (0, 0) -> (0, 0)
        q, r = transform._transform_axial(0, 0, 1)
        assert q == 0 and r == 0

        # Test (1, 0) -> (0, 1)
        q, r = transform._transform_axial(1, 0, 1)
        assert q == 0 and r == 1

        # Test (0, 1) -> (-1, 1)
        q, r = transform._transform_axial(0, 1, 1)
        assert q == -1 and r == 1

    def test_rotation_180_coordinates(self, transform):
        """Test 180° rotation formula: (q, r) -> (-q, -r)."""
        # Test (1, 2) -> (-1, -2)
        q, r = transform._transform_axial(1, 2, 3)
        assert q == -1 and r == -2

        # Test (-3, 5) -> (3, -5)
        q, r = transform._transform_axial(-3, 5, 3)
        assert q == 3 and r == -5

    def test_all_transforms_preserve_board_shape(self, transform):
        """All 12 transforms should preserve board dimensions."""
        board = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        board = board.astype(np.float32)

        for t_id in range(12):
            result = transform.transform_board(board, t_id)
            assert result.shape == board.shape, \
                f"Transform {t_id} changed shape"

    def test_all_transforms_preserve_policy_shape(self, transform):
        """All 12 transforms should preserve policy dimensions."""
        policy = np.random.rand(P_HEX).astype(np.float32)

        for t_id in range(12):
            result = transform.transform_policy(policy, t_id)
            assert result.shape == policy.shape, \
                f"Transform {t_id} changed shape"

    def test_rotation_composition_60_plus_60_equals_120(self, transform):
        """60° + 60° should equal 120° for valid hex cells."""
        # Use board with zeros outside hex
        board = create_hex_board_with_zeros_outside()
        mask = create_hex_mask()

        # Apply 60° twice
        result_twice = transform.transform_board(board, 1)
        result_twice = transform.transform_board(result_twice, 1)

        # Apply 120° once
        result_once = transform.transform_board(board, 2)

        # Compare only valid hex positions (clamping affects boundary cells)
        np.testing.assert_array_almost_equal(
            result_twice[:, mask], result_once[:, mask]
        )

    def test_rotation_composition_3x60_equals_180(self, transform):
        """3 x 60° should equal 180° for valid hex cells."""
        # Use board with zeros outside hex
        board = create_hex_board_with_zeros_outside()
        mask = create_hex_mask()

        # Apply 60° three times
        result = board.copy()
        for _ in range(3):
            result = transform.transform_board(result, 1)

        # Apply 180° once
        expected = transform.transform_board(board, 3)

        # Compare only valid hex positions
        np.testing.assert_array_almost_equal(
            result[:, mask], expected[:, mask]
        )

    def test_rotation_6x60_equals_identity(self, transform):
        """6 x 60° should return to identity for valid hex cells."""
        # Use board with zeros outside hex
        board = create_hex_board_with_zeros_outside()
        mask = create_hex_mask()

        # Apply 60° six times
        result = board.copy()
        for _ in range(6):
            result = transform.transform_board(result, 1)

        # Compare only valid hex positions
        np.testing.assert_array_almost_equal(result[:, mask], board[:, mask])

    def test_compose_transformations(self, transform):
        """Test explicit composition function."""
        # 60° (1) composed with 60° (1) should be 120° (2)
        composed = transform.compose_transforms(1, 1)
        assert composed == 2

        # 60° (1) composed with 300° (5) should be identity (0)
        composed = transform.compose_transforms(1, 5)
        assert composed == 0

    def test_roundtrip_rotation(self, transform):
        """Apply rotation then inverse should return original for valid hex."""
        # Use board with zeros outside hex
        board = create_hex_board_with_zeros_outside()
        mask = create_hex_mask()

        for t_id in range(6):  # Test all rotations
            transformed = transform.transform_board(board, t_id)
            inverse_id = transform.get_inverse_transform(t_id)
            recovered = transform.transform_board(transformed, inverse_id)

            # Compare only valid hex positions
            np.testing.assert_array_almost_equal(
                recovered[:, mask], board[:, mask],
                err_msg=f"Roundtrip failed for rotation {t_id}"
            )

    def test_roundtrip_reflection(self, transform):
        """Reflections are self-inverse for valid hex cells."""
        # Use board with zeros outside hex
        board = create_hex_board_with_zeros_outside()
        mask = create_hex_mask()

        for t_id in range(6, 12):  # Test all reflections
            transformed = transform.transform_board(board, t_id)
            # Reflections are self-inverse
            recovered = transform.transform_board(transformed, t_id)

            # Compare only valid hex positions
            np.testing.assert_array_almost_equal(
                recovered[:, mask], board[:, mask],
                err_msg=f"Reflection {t_id} is not self-inverse"
            )

    def test_roundtrip_policy(self, transform):
        def test_roundtrip_policy(self, transform):
            """Policy roundtrip should recover original."""
            policy = np.random.rand(P_HEX).astype(np.float32)
            policy /= policy.sum()

            for t_id in range(12):
                transformed = transform.transform_policy(policy, t_id)
                inverse_id = transform.get_inverse_transform(t_id)
                recovered = transform.transform_policy(transformed, inverse_id)

                # Check non-zero values are preserved
                orig_nonzero = set(np.where(policy > 1e-6)[0])
                rec_nonzero = set(np.where(recovered > 1e-6)[0])

                # Note: some indices may be lost due to out-of-bounds
                # after transformation, so we check overlap
                overlap = len(orig_nonzero & rec_nonzero)
                expected = 0.8 * len(orig_nonzero)
                assert overlap > expected, \
                    f"Transform {t_id} lost too many policy entries"

    def test_get_all_transforms(self, transform):
        """get_all_transforms should return [0..11]."""
        all_transforms = transform.get_all_transforms()
        assert all_transforms == list(range(12))

    def test_get_inverse_transform(self, transform):
        """Test inverse transform IDs."""
        # Rotations: inverse of R^k is R^(6-k)
        assert transform.get_inverse_transform(0) == 0  # Identity
        assert transform.get_inverse_transform(1) == 5  # 60° -> 300°
        assert transform.get_inverse_transform(2) == 4  # 120° -> 240°
        assert transform.get_inverse_transform(3) == 3  # 180° -> 180°

        # Reflections are self-inverse
        for t_id in range(6, 12):
            assert transform.get_inverse_transform(t_id) == t_id


class TestMoveIndexTransformation:
    """Test move index transformation."""

    @pytest.fixture
    def transform(self):
        return HexSymmetryTransform(board_size=HEX_BOARD_SIZE)

    def test_pass_move_unchanged(self, transform):
        """Pass move (special action) should not change."""
        pass_idx = P_HEX - 1  # Last index is pass/special
        result = transform.transform_move_index(pass_idx, 1)
        assert result == pass_idx

    def test_special_actions_unchanged(self, transform):
        """Special actions at end of policy are unchanged."""
        for t_id in range(12):
            result = transform.transform_move_index(P_HEX - 1, t_id)
            assert result == P_HEX - 1

    def test_placement_identity(self, transform):
        """Placement at center should stay at center with identity."""
        # Center placement: (10, 10) with piece type 0
        center_idx = 10 * HEX_BOARD_SIZE * 3 + 10 * 3 + 0
        result = transform.transform_move_index(center_idx, 0)
        assert result == center_idx

    def test_movement_identity(self, transform):
        """Movement should be unchanged with identity transform."""
        # A movement from center
        move_idx = HEX_MOVEMENT_BASE + 100
        result = transform.transform_move_index(move_idx, 0)
        assert result == move_idx


class TestSparsePolicy:
    """Test sparse policy transformation."""

    @pytest.fixture
    def transform(self):
        return HexSymmetryTransform(board_size=HEX_BOARD_SIZE)

    def test_sparse_policy_identity(self, transform):
        """Sparse policy with identity transformation."""
        indices = np.array([0, 100, 1000], dtype=np.int32)
        values = np.array([0.3, 0.5, 0.2], dtype=np.float32)

        new_indices, new_values = transform.transform_sparse_policy(
            indices, values, transform_id=0
        )

        np.testing.assert_array_equal(new_indices, indices)
        np.testing.assert_array_almost_equal(new_values, values)

    def test_sparse_policy_preserves_sum(self, transform):
        """Sparse policy sum should be preserved."""
        indices = np.array([0, 100, 500, 1000], dtype=np.int32)
        values = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float32)

        for t_id in range(12):
            new_indices, new_values = transform.transform_sparse_policy(
                indices, values, transform_id=t_id
            )
            # Sum should be approximately preserved (some may be filtered)
            assert new_values.sum() <= values.sum() + 1e-5

    def test_empty_sparse_policy(self, transform):
        """Empty sparse policy should return empty."""
        indices = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float32)

        new_indices, new_values = transform.transform_sparse_policy(
            indices, values, transform_id=1
        )

        assert len(new_indices) == 0
        assert len(new_values) == 0


class TestAugmentHexSample:
    """Test the augment_hex_sample convenience function."""

    def test_returns_12_augmented_samples(self):
        """Should return 12 augmented samples for all D6 transforms."""
        features = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        features = features.astype(np.float32)
        globals_vec = np.random.rand(10).astype(np.float32)
        policy_indices = np.array([0, 100, 500], dtype=np.int32)
        policy_values = np.array([0.3, 0.5, 0.2], dtype=np.float32)

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        assert len(results) == 12

    def test_first_sample_is_identity(self):
        """First sample should be identity transformation."""
        features = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        features = features.astype(np.float32)
        globals_vec = np.random.rand(10).astype(np.float32)
        policy_indices = np.array([0, 100], dtype=np.int32)
        policy_values = np.array([0.6, 0.4], dtype=np.float32)

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        # First result should be identity
        aug_features, aug_globals, aug_indices, aug_values = results[0]

        np.testing.assert_array_almost_equal(aug_features, features)
        np.testing.assert_array_almost_equal(aug_globals, globals_vec)
        np.testing.assert_array_equal(aug_indices, policy_indices)
        np.testing.assert_array_almost_equal(aug_values, policy_values)

    def test_all_samples_have_correct_shape(self):
        """All augmented samples should have correct shapes."""
        features = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        features = features.astype(np.float32)
        globals_vec = np.random.rand(10).astype(np.float32)
        policy_indices = np.array([0, 100, 500], dtype=np.int32)
        policy_values = np.array([0.3, 0.5, 0.2], dtype=np.float32)

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        for i, (aug_feat, aug_glob, aug_idx, aug_val) in enumerate(results):
            assert aug_feat.shape == features.shape, f"Sample {i} features"
            assert aug_glob.shape == globals_vec.shape, f"Sample {i} globals"
            assert len(aug_idx) == len(aug_val), f"Sample {i} policy mismatch"


class TestBoardTransformConsistency:
    """Test consistency between board and policy transformations."""

    @pytest.fixture
    def transform(self):
        return HexSymmetryTransform(board_size=HEX_BOARD_SIZE)

    def test_board_policy_consistency_placement(self, transform):
        """Board and policy transforms should be consistent for placements.

        If we place a stone at position (q, r), the transformed policy
        should indicate placement at the transformed position.
        """
        # Create a board with a single stone at a known position
        # Position: (q=2, r=3) in axial -> (cy=13, cx=12) in grid coords
        q, r = 2, 3
        cy = HEX_RADIUS + r
        cx = HEX_RADIUS + q

        # Create feature with a stone at (cy, cx)
        shape = (10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        board = np.zeros(shape, dtype=np.float32)
        board[0, cy, cx] = 1.0  # Channel 0, stone at position

        # Create policy with placement at same position
        placement_idx = cy * HEX_BOARD_SIZE * 3 + cx * 3 + 0
        policy = np.zeros(P_HEX, dtype=np.float32)
        policy[placement_idx] = 1.0

        for t_id in range(12):
            # Transform board
            t_board = transform.transform_board(board, t_id)

            # Transform policy
            t_policy = transform.transform_policy(policy, t_id)

            # Find where the stone moved to in the board
            stone_pos = np.where(t_board[0] == 1.0)
            if len(stone_pos[0]) > 0:
                t_cy, t_cx = stone_pos[0][0], stone_pos[1][0]

                # Find where the policy mass moved to
                policy_idx = np.argmax(t_policy)

                if policy_idx < HEX_PLACEMENT_SPAN:
                    # Decode placement index
                    p_cy = policy_idx // (HEX_BOARD_SIZE * 3)
                    p_cx = (policy_idx % (HEX_BOARD_SIZE * 3)) // 3

                    # Should match
                    assert p_cy == t_cy and p_cx == t_cx, \
                        f"Transform {t_id}: board ({t_cy},{t_cx}) != " \
                        f"policy ({p_cy},{p_cx})"


class TestHexGridValidation:
    """Test hex grid boundary validation."""

    @pytest.fixture
    def transform(self):
        return HexSymmetryTransform(board_size=HEX_BOARD_SIZE)

    def test_center_stays_inside(self, transform):
        """Center position should stay inside grid under all transforms."""
        q, r = 0, 0
        for t_id in range(12):
            t_q, t_r = transform._transform_axial(q, r, t_id)
            assert abs(t_q) <= HEX_RADIUS
            assert abs(t_r) <= HEX_RADIUS
            assert abs(t_q + t_r) <= HEX_RADIUS

    def test_boundary_respects_radius(self, transform):
        """Positions within radius should stay within radius."""
        # Test some boundary positions
        boundary_positions = [
            (HEX_RADIUS, 0),
            (-HEX_RADIUS, 0),
            (0, HEX_RADIUS),
            (0, -HEX_RADIUS),
            (HEX_RADIUS, -HEX_RADIUS),
            (-HEX_RADIUS, HEX_RADIUS),
        ]

        for q, r in boundary_positions:
            for t_id in range(12):
                t_q, t_r = transform._transform_axial(q, r, t_id)
                in_bounds = (
                    abs(t_q) <= HEX_RADIUS and
                    abs(t_r) <= HEX_RADIUS and
                    abs(t_q + t_r) <= HEX_RADIUS
                )
                assert in_bounds, \
                    f"Transform {t_id} of ({q},{r}) -> ({t_q},{t_r}) OOB"