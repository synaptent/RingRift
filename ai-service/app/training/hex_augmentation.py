"""
Hex Board Symmetry Augmentation for Training Data.

Implements the dihedral group D6: 12 transformations
(6 rotations + 6 reflections) for hexagonal boards
using axial coordinates (q, r).

The canonical hex board has radius N=12 (469 cells)
embedded in a 25x25 bounding box.

Rotation formulas (clockwise, axial coordinates):
- 0°: (q, r) -> (q, r) [identity]
- 60° CW: (q, r) -> (-r, q + r)
- 120° CW: (q, r) -> (-q - r, q)
- 180°: (q, r) -> (-q, -r)
- 240° CW: (q, r) -> (r, -q - r)
- 300° CW: (q, r) -> (q + r, -q)

Reflection (swap q and r, reflection across s-axis):
- S: (q, r) -> (r, q)

The 12 D6 transformations are:
- 0-5: Rotations R^k for k = 0..5
- 6-11: Reflections S * R^k for k = 0..5
"""

import numpy as np
from typing import List, Tuple, Optional

# Hex board constants (match neural_net.py)
HEX_BOARD_SIZE = 25
HEX_RADIUS = 12

# Number of hex directions for movement encoding
NUM_HEX_DIRS = 6
HEX_MAX_DIST = 24

# Policy layout constants (from neural_net.py)
HEX_PLACEMENT_SPAN = HEX_BOARD_SIZE * HEX_BOARD_SIZE * 3  # 1875
HEX_MOVEMENT_BASE = HEX_PLACEMENT_SPAN
HEX_MOVEMENT_SPAN = (
    HEX_BOARD_SIZE * HEX_BOARD_SIZE * NUM_HEX_DIRS * HEX_MAX_DIST
)  # 90000
HEX_SPECIAL_BASE = HEX_MOVEMENT_BASE + HEX_MOVEMENT_SPAN  # 91875
P_HEX = HEX_SPECIAL_BASE + 1  # 91876

# Hex directions in canonical (dq, dr) form (matching neural_net.HEX_DIRS)
HEX_DIRS = [
    (1, 0),    # +q
    (0, 1),    # +r
    (-1, 1),   # -q + r
    (-1, 0),   # -q
    (0, -1),   # -r
    (1, -1),   # +q - r
]


class HexSymmetryTransform:
    """
    Applies D6 symmetry transformations to hex board features and policies.

    The 12 transformations are indexed as:
    - 0-5: Rotations (0°, 60°, 120°, 180°, 240°, 300° clockwise)
    - 6-11: Reflections (rotation k composed with swap reflection, k=0..5)

    All reflections are self-inverse.
    For rotations, the inverse of R^k is R^(6-k).
    """

    def __init__(self, board_size: int = HEX_BOARD_SIZE):
        self.board_size = board_size
        self.radius = (board_size - 1) // 2  # For 25x25, radius = 12

        # Precompute transformation index mappings for efficiency
        self._build_index_maps()
        self._build_direction_maps()

    @staticmethod
    def get_all_transforms() -> List[int]:
        """Returns list of all 12 transformation IDs [0, 1, ..., 11]."""
        return list(range(12))

    def _axial_to_canonical(self, q: int, r: int) -> Tuple[int, int]:
        """Convert axial coords (q, r) to canonical grid coords (cx, cy)."""
        return q + self.radius, r + self.radius

    def _canonical_to_axial(self, cx: int, cy: int) -> Tuple[int, int]:
        """Convert canonical grid coords (cx, cy) to axial coords (q, r)."""
        return cx - self.radius, cy - self.radius

    def _transform_axial(
        self, q: int, r: int, transform_id: int
    ) -> Tuple[int, int]:
        """
        Apply transformation to axial coordinates.

        Rotations (clockwise):
        - 0: identity: (q, r) → (q, r)
        - 1: 60° CW: (q, r) → (-r, q + r)
        - 2: 120° CW: (q, r) → (-q - r, q)
        - 3: 180°: (q, r) → (-q, -r)
        - 4: 240° CW: (q, r) → (r, -q - r)
        - 5: 300° CW: (q, r) → (q + r, -q)

        Reflections (apply rotation k, then swap q and r):
        - 6: S*R^0: (q, r) → (r, q)
        - 7: S*R^1: (q, r) → (q + r, -r)
        - 8: S*R^2: (q, r) → (q, -q - r)
        - 9: S*R^3: (q, r) → (-r, -q)
        - 10: S*R^4: (q, r) → (-q - r, r)
        - 11: S*R^5: (q, r) → (-q, q + r)
        """
        # Apply rotation first
        rot = transform_id % 6

        if rot == 0:
            qr, rr = q, r
        elif rot == 1:
            qr, rr = -r, q + r
        elif rot == 2:
            qr, rr = -q - r, q
        elif rot == 3:
            qr, rr = -q, -r
        elif rot == 4:
            qr, rr = r, -q - r
        elif rot == 5:
            qr, rr = q + r, -q
        else:
            qr, rr = q, r

        # Apply reflection if transform_id >= 6
        if transform_id >= 6:
            qr, rr = rr, qr  # Swap q and r (reflection across s-axis)

        return qr, rr

    def _inverse_transform_axial(
        self, q: int, r: int, transform_id: int
    ) -> Tuple[int, int]:
        """
        Apply inverse transformation to axial coordinates.

        For rotations R^k, inverse is R^(6-k) mod 6.
        For reflections S*R^k, they are self-inverse: (S*R^k)^(-1) = S*R^k.
        """
        if transform_id >= 6:
            # Reflection: apply same transformation (self-inverse)
            return self._transform_axial(q, r, transform_id)
        else:
            # Pure rotation: inverse is R^(6-k) mod 6
            if transform_id == 0:
                inverse_rot = 0
            else:
                inverse_rot = 6 - transform_id
            return self._transform_axial(q, r, inverse_rot)

    def _build_index_maps(self):
        """
        Precompute index mapping arrays for each transformation.

        For transform t, self._board_maps[t] is a tuple (cy_map, cx_map)
        such that:
        output[:, cy', cx'] = input[:, cy_map[cy', cx'], cx_map[cy', cx']]

        This allows efficient transformation of feature tensors using
        numpy advanced indexing.
        """
        self._board_maps = {}

        for t in range(12):
            size = self.board_size
            cy_map = np.zeros((size, size), dtype=np.int32)
            cx_map = np.zeros((size, size), dtype=np.int32)

            for cy_out in range(self.board_size):
                for cx_out in range(self.board_size):
                    # For output position (cx_out, cy_out), find input position
                    q_out, r_out = self._canonical_to_axial(cx_out, cy_out)
                    q_in, r_in = self._inverse_transform_axial(q_out, r_out, t)
                    cx_in, cy_in = self._axial_to_canonical(q_in, r_in)

                    # Clamp to valid range (out-of-bounds reads 0)
                    cx_in = max(0, min(self.board_size - 1, cx_in))
                    cy_in = max(0, min(self.board_size - 1, cy_in))

                    cy_map[cy_out, cx_out] = cy_in
                    cx_map[cy_out, cx_out] = cx_in

            self._board_maps[t] = (cy_map, cx_map)

    def _build_direction_maps(self):
        """
        Precompute how each hex direction transforms under each transformation.

        For movement encoding, we need to know how direction indices map.
        HEX_DIRS[d] gives (dq, dr) for direction d.
        Under transformation t, direction d becomes direction d' such that
        HEX_DIRS[d'] is the transformed (dq, dr).
        """
        self._dir_maps = {}

        for t in range(12):
            dir_map = np.zeros(NUM_HEX_DIRS, dtype=np.int32)

            for d in range(NUM_HEX_DIRS):
                dq, dr = HEX_DIRS[d]
                # Transform the direction vector
                dq_t, dr_t = self._transform_axial(dq, dr, t)

                # Find which direction index this corresponds to
                try:
                    d_new = HEX_DIRS.index((dq_t, dr_t))
                except ValueError:
                    # Direction not in canonical set - shouldn't happen
                    # but handle gracefully
                    d_new = d

                dir_map[d] = d_new

            self._dir_maps[t] = dir_map

    def transform_board(
        self, board: np.ndarray, transform_id: int
    ) -> np.ndarray:
        """
        Transform a board feature tensor.

        Args:
            board: Feature tensor of shape (C, H, W) where H = W = board_size
            transform_id: Transformation index 0-11

        Returns:
            Transformed tensor of same shape
        """
        if transform_id == 0:
            return board.copy()

        cy_map, cx_map = self._board_maps[transform_id]

        # Apply transformation using advanced indexing
        # output[:, y, x] = input[:, cy_map[y, x], cx_map[y, x]]
        return board[:, cy_map, cx_map]

    def transform_policy(
        self, policy: np.ndarray, transform_id: int
    ) -> np.ndarray:
        """
        Transform a policy probability/logit vector.

        Args:
            policy: Policy vector of length P_HEX (91876)
            transform_id: Transformation index 0-11

        Returns:
            Transformed policy vector of same length
        """
        if transform_id == 0:
            return policy.copy()

        output = np.zeros_like(policy)

        # Transform placement indices (0 to HEX_PLACEMENT_SPAN-1)
        for idx in range(HEX_PLACEMENT_SPAN):
            new_idx = self._transform_placement_index(idx, transform_id)
            if 0 <= new_idx < HEX_PLACEMENT_SPAN:
                output[new_idx] = policy[idx]

        # Transform movement indices (HEX_MOVEMENT_BASE to HEX_SPECIAL_BASE-1)
        for idx in range(HEX_MOVEMENT_BASE, HEX_SPECIAL_BASE):
            new_idx = self._transform_movement_index(idx, transform_id)
            if HEX_MOVEMENT_BASE <= new_idx < HEX_SPECIAL_BASE:
                output[new_idx] = policy[idx]

        # Special index (skip) is unchanged
        if len(policy) > HEX_SPECIAL_BASE:
            output[HEX_SPECIAL_BASE] = policy[HEX_SPECIAL_BASE]

        return output

    def _transform_placement_index(self, idx: int, transform_id: int) -> int:
        """Transform a placement policy index."""
        # Decode: idx = (cy * board_size + cx) * 3 + count_idx
        count_idx = idx % 3
        pos_idx = idx // 3
        cy = pos_idx // self.board_size
        cx = pos_idx % self.board_size

        # Transform position
        q, r = self._canonical_to_axial(cx, cy)
        q_t, r_t = self._transform_axial(q, r, transform_id)
        cx_t, cy_t = self._axial_to_canonical(q_t, r_t)

        # Check bounds
        if not (0 <= cx_t < self.board_size and 0 <= cy_t < self.board_size):
            return -1

        # Encode new index
        new_pos_idx = cy_t * self.board_size + cx_t
        return new_pos_idx * 3 + count_idx

    def _transform_movement_index(self, idx: int, transform_id: int) -> int:
        """Transform a movement policy index."""
        # Decode: HEX_MOVEMENT_BASE + from * (6*20) + dir * 20 + (dist-1)
        offset = idx - HEX_MOVEMENT_BASE

        dist_idx = offset % HEX_MAX_DIST
        offset //= HEX_MAX_DIST

        dir_idx = offset % NUM_HEX_DIRS
        offset //= NUM_HEX_DIRS

        from_idx = offset
        from_cy = from_idx // self.board_size
        from_cx = from_idx % self.board_size

        # Transform from position
        q, r = self._canonical_to_axial(from_cx, from_cy)
        q_t, r_t = self._transform_axial(q, r, transform_id)
        from_cx_t, from_cy_t = self._axial_to_canonical(q_t, r_t)

        # Transform direction
        dir_idx_t = self._dir_maps[transform_id][dir_idx]

        # Check bounds
        size = self.board_size
        if not (0 <= from_cx_t < size and 0 <= from_cy_t < size):
            return -1

        # Encode new index
        new_from_idx = from_cy_t * self.board_size + from_cx_t
        return (
            HEX_MOVEMENT_BASE
            + new_from_idx * (NUM_HEX_DIRS * HEX_MAX_DIST)
            + dir_idx_t * HEX_MAX_DIST
            + dist_idx
        )

    def transform_move_index(
        self,
        move_idx: int,
        transform_id: int,
        board_size: Optional[int] = None,
    ) -> int:
        """
        Transform a single move index.

        Args:
            move_idx: Policy index for the move
            transform_id: Transformation index 0-11
            board_size: Board size (unused, kept for API compatibility)

        Returns:
            Transformed move index, or -1 if out of bounds
        """
        if transform_id == 0:
            return move_idx

        if move_idx < 0:
            return -1

        if move_idx < HEX_PLACEMENT_SPAN:
            return self._transform_placement_index(move_idx, transform_id)
        elif move_idx < HEX_SPECIAL_BASE:
            return self._transform_movement_index(move_idx, transform_id)
        elif move_idx == HEX_SPECIAL_BASE:
            return move_idx  # Skip is unchanged
        else:
            return -1

    def transform_sparse_policy(
        self,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        transform_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a sparse policy representation.

        Args:
            policy_indices: Array of move indices
            policy_values: Array of corresponding values/probabilities
            transform_id: Transformation index 0-11

        Returns:
            Tuple of (transformed_indices, values) with invalid indices
            filtered out
        """
        if transform_id == 0:
            return policy_indices.copy(), policy_values.copy()

        new_indices = np.array([
            self.transform_move_index(int(idx), transform_id)
            for idx in policy_indices
        ], dtype=policy_indices.dtype)

        # Filter out invalid indices
        valid_mask = new_indices >= 0
        return new_indices[valid_mask], policy_values[valid_mask]

    def get_inverse_transform(self, transform_id: int) -> int:
        """
        Get the inverse transformation ID.

        For rotations R^k, inverse is R^(6-k) mod 6.
        For reflections S*R^k, inverse is S*R^k (self-inverse).
        """
        if transform_id >= 6:
            return transform_id  # Reflections are self-inverse
        elif transform_id == 0:
            return 0
        else:
            return 6 - transform_id

    def compose_transforms(self, t1: int, t2: int) -> int:
        """
        Compute the composition of two transformations: t2 ∘ t1
        (apply t1 first, then t2).

        This is useful for verifying rotation composition properties.

        Returns the transformation ID equivalent to applying t1 then t2.
        """
        # Apply t1 to a reference point, then apply t2
        # We use the group structure of D6 to compute this

        # Decompose into rotation and reflection components
        r1, s1 = t1 % 6, t1 >= 6
        r2, s2 = t2 % 6, t2 >= 6

        if not s1 and not s2:
            # Both rotations: R^r2 * R^r1 = R^(r1+r2)
            return (r1 + r2) % 6
        elif s1 and not s2:
            # R^r2 * S * R^r1 = S * R^(-r2) * R^r1 = S * R^(r1-r2)
            return 6 + (r1 - r2) % 6
        elif not s1 and s2:
            # S * R^r2 * R^r1 = S * R^(r1+r2)
            return 6 + (r1 + r2) % 6
        else:
            # S * R^r2 * S * R^r1 = R^(-r2) * R^r1 = R^(r1-r2)
            return (r1 - r2) % 6


def augment_hex_sample(
    features: np.ndarray,
    globals_vec: np.ndarray,
    policy_indices: np.ndarray,
    policy_values: np.ndarray,
    transform: Optional[HexSymmetryTransform] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply all 12 D6 transformations to a single training sample.

    Args:
        features: Board features tensor (C, H, W)
        globals_vec: Global features vector
        policy_indices: Sparse policy move indices
        policy_values: Sparse policy values
        transform: HexSymmetryTransform instance (created if not provided)

    Returns:
        List of 12 tuples, each containing
        (features, globals, policy_indices, policy_values)
    """
    if transform is None:
        transform = HexSymmetryTransform()

    augmented = []
    for t in transform.get_all_transforms():
        aug_features = transform.transform_board(features, t)
        aug_indices, aug_values = transform.transform_sparse_policy(
            policy_indices, policy_values, t
        )
        # Globals don't change under symmetry transformations
        augmented.append((
            aug_features, globals_vec.copy(), aug_indices, aug_values
        ))

    return augmented
