"""
Square Board Symmetry Augmentation for Training Data.

Implements the dihedral group D4: 8 transformations
(4 rotations + 4 reflections) for square boards.

Rotation formulas (clockwise, grid coordinates):
- 0°: (x, y) -> (x, y) [identity]
- 90° CW: (x, y) -> (N-1-y, x)
- 180°: (x, y) -> (N-1-x, N-1-y)
- 270° CW: (x, y) -> (y, N-1-x)

Reflections:
- Horizontal flip: (x, y) -> (N-1-x, y)
- Vertical flip: (x, y) -> (x, N-1-y)
- Diagonal (main): (x, y) -> (y, x)
- Anti-diagonal: (x, y) -> (N-1-y, N-1-x)

The 8 D4 transformations are indexed as:
- 0: Identity
- 1: 90° CW rotation
- 2: 180° rotation
- 3: 270° CW rotation
- 4: Horizontal flip
- 5: Vertical flip
- 6: Main diagonal flip (transpose)
- 7: Anti-diagonal flip
"""


import numpy as np

# Square board movement directions (8 directions - Moore neighborhood)
SQUARE_DIRS = [
    (1, 0),    # right
    (1, 1),    # down-right
    (0, 1),    # down
    (-1, 1),   # down-left
    (-1, 0),   # left
    (-1, -1),  # up-left
    (0, -1),   # up
    (1, -1),   # up-right
]
NUM_SQUARE_DIRS = 8


class SquareSymmetryTransform:
    """
    Applies D4 symmetry transformations to square board features and policies.

    The 8 transformations are:
    - 0: Identity
    - 1: 90° CW rotation
    - 2: 180° rotation
    - 3: 270° CW rotation
    - 4: Horizontal flip (reflect across vertical axis)
    - 5: Vertical flip (reflect across horizontal axis)
    - 6: Main diagonal flip (transpose)
    - 7: Anti-diagonal flip
    """

    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        self._build_index_maps()
        self._build_direction_maps()

        # Policy layout depends on board size
        self._compute_policy_layout()

    def _compute_policy_layout(self):
        """Compute policy index ranges based on board size."""
        n = self.board_size
        max_dist = n - 1

        # Placement: 3 * N * N (ring counts 1-3 at each position)
        self.placement_span = 3 * n * n

        # Movement: N * N * 8 * (N-1)
        # from_idx * (8 * max_dist) + dir_idx * max_dist + dist_idx
        self.movement_base = self.placement_span
        self.movement_span = n * n * NUM_SQUARE_DIRS * max_dist
        self.max_dist = max_dist

        # Special actions start after movement
        self.special_base = self.movement_base + self.movement_span
        self.policy_size = self.special_base + 100  # Room for special actions

    @staticmethod
    def get_all_transforms() -> list[int]:
        """Returns list of all 8 transformation IDs [0, 1, ..., 7]."""
        return list(range(8))

    def _transform_coords(
        self, x: int, y: int, transform_id: int
    ) -> tuple[int, int]:
        """
        Apply transformation to grid coordinates.

        Rotations (clockwise):
        - 0: identity: (x, y) -> (x, y)
        - 1: 90° CW: (x, y) -> (N-1-y, x)
        - 2: 180°: (x, y) -> (N-1-x, N-1-y)
        - 3: 270° CW: (x, y) -> (y, N-1-x)

        Reflections:
        - 4: H-flip: (x, y) -> (N-1-x, y)
        - 5: V-flip: (x, y) -> (x, N-1-y)
        - 6: Main diag: (x, y) -> (y, x)
        - 7: Anti-diag: (x, y) -> (N-1-y, N-1-x)
        """
        n = self.board_size - 1

        if transform_id == 0:
            return x, y
        elif transform_id == 1:  # 90° CW
            return n - y, x
        elif transform_id == 2:  # 180°
            return n - x, n - y
        elif transform_id == 3:  # 270° CW
            return y, n - x
        elif transform_id == 4:  # H-flip
            return n - x, y
        elif transform_id == 5:  # V-flip
            return x, n - y
        elif transform_id == 6:  # Main diagonal
            return y, x
        elif transform_id == 7:  # Anti-diagonal
            return n - y, n - x
        else:
            return x, y

    def _inverse_transform_coords(
        self, x: int, y: int, transform_id: int
    ) -> tuple[int, int]:
        """
        Apply inverse transformation to grid coordinates.

        Inverse mappings:
        - 0 (identity): inverse is 0
        - 1 (90° CW): inverse is 3 (270° CW)
        - 2 (180°): inverse is 2 (180°)
        - 3 (270° CW): inverse is 1 (90° CW)
        - 4-7 (reflections): all are self-inverse
        """
        inverse_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
        return self._transform_coords(x, y, inverse_map[transform_id])

    def _build_index_maps(self):
        """
        Precompute index mapping arrays for each transformation.

        For transform t, self._board_maps[t] is a tuple (y_map, x_map)
        such that:
        output[:, y', x'] = input[:, y_map[y', x'], x_map[y', x']]
        """
        self._board_maps = {}

        for t in range(8):
            size = self.board_size
            y_map = np.zeros((size, size), dtype=np.int32)
            x_map = np.zeros((size, size), dtype=np.int32)

            for y_out in range(size):
                for x_out in range(size):
                    # For output position (x_out, y_out), find input position
                    x_in, y_in = self._inverse_transform_coords(x_out, y_out, t)

                    # Clamp to valid range
                    x_in = max(0, min(size - 1, x_in))
                    y_in = max(0, min(size - 1, y_in))

                    y_map[y_out, x_out] = y_in
                    x_map[y_out, x_out] = x_in

            self._board_maps[t] = (y_map, x_map)

    def _build_direction_maps(self):
        """
        Precompute how each direction transforms under each transformation.

        For movement encoding, we need to know how direction indices map.
        SQUARE_DIRS[d] gives (dx, dy) for direction d.
        Under transformation t, direction d becomes direction d' such that
        SQUARE_DIRS[d'] is the transformed (dx, dy).
        """
        self._dir_maps = {}

        for t in range(8):
            dir_map = np.zeros(NUM_SQUARE_DIRS, dtype=np.int32)

            for d in range(NUM_SQUARE_DIRS):
                dx, dy = SQUARE_DIRS[d]

                # Transform direction vector
                # For rotations, we use the same transformation
                # For reflections, we need to apply to direction too
                dx_t, dy_t = self._transform_direction(dx, dy, t)

                # Find which direction index this corresponds to
                try:
                    d_new = SQUARE_DIRS.index((dx_t, dy_t))
                except ValueError:
                    # Shouldn't happen for valid D4 transforms
                    d_new = d

                dir_map[d] = d_new

            self._dir_maps[t] = dir_map

    def _transform_direction(
        self, dx: int, dy: int, transform_id: int
    ) -> tuple[int, int]:
        """Transform a direction vector under D4 transformation."""
        # Direction transforms same as position but without translation
        if transform_id == 0:
            return dx, dy
        elif transform_id == 1:  # 90° CW
            return -dy, dx
        elif transform_id == 2:  # 180°
            return -dx, -dy
        elif transform_id == 3:  # 270° CW
            return dy, -dx
        elif transform_id == 4:  # H-flip
            return -dx, dy
        elif transform_id == 5:  # V-flip
            return dx, -dy
        elif transform_id == 6:  # Main diagonal
            return dy, dx
        elif transform_id == 7:  # Anti-diagonal
            return -dy, -dx
        else:
            return dx, dy

    def transform_board(
        self, board: np.ndarray, transform_id: int
    ) -> np.ndarray:
        """
        Transform a board feature tensor.

        Args:
            board: Feature tensor of shape (C, H, W) where H = W = board_size
            transform_id: Transformation index 0-7

        Returns:
            Transformed tensor of same shape
        """
        if transform_id == 0:
            return board.copy()

        y_map, x_map = self._board_maps[transform_id]

        # Apply transformation using advanced indexing
        return board[:, y_map, x_map]

    def transform_policy(
        self, policy: np.ndarray, transform_id: int
    ) -> np.ndarray:
        """
        Transform a policy probability/logit vector.

        Args:
            policy: Policy vector
            transform_id: Transformation index 0-7

        Returns:
            Transformed policy vector of same length
        """
        if transform_id == 0:
            return policy.copy()

        output = np.zeros_like(policy)

        # Transform placement indices
        for idx in range(min(len(policy), self.placement_span)):
            new_idx = self._transform_placement_index(idx, transform_id)
            if 0 <= new_idx < len(output):
                output[new_idx] = policy[idx]

        # Transform movement indices
        end_movement = min(len(policy), self.special_base)
        for idx in range(self.movement_base, end_movement):
            new_idx = self._transform_movement_index(idx, transform_id)
            if self.movement_base <= new_idx < len(output):
                output[new_idx] = policy[idx]

        # Copy special indices unchanged
        if len(policy) > self.special_base:
            output[self.special_base:] = policy[self.special_base:]

        return output

    def _transform_placement_index(self, idx: int, transform_id: int) -> int:
        """Transform a placement policy index."""
        # Decode: idx = (y * board_size + x) * 3 + count_idx
        n = self.board_size
        count_idx = idx % 3
        pos_idx = idx // 3
        y = pos_idx // n
        x = pos_idx % n

        # Transform position
        x_t, y_t = self._transform_coords(x, y, transform_id)

        # Check bounds
        if not (0 <= x_t < n and 0 <= y_t < n):
            return -1

        # Encode new index
        new_pos_idx = y_t * n + x_t
        return new_pos_idx * 3 + count_idx

    def _transform_movement_index(self, idx: int, transform_id: int) -> int:
        """Transform a movement policy index."""
        n = self.board_size
        max_dist = self.max_dist

        # Decode: movement_base + from_idx * (8 * max_dist) + dir * max_dist + dist_idx
        offset = idx - self.movement_base

        dist_idx = offset % max_dist
        offset //= max_dist

        dir_idx = offset % NUM_SQUARE_DIRS
        offset //= NUM_SQUARE_DIRS

        from_idx = offset
        from_y = from_idx // n
        from_x = from_idx % n

        # Transform from position
        from_x_t, from_y_t = self._transform_coords(from_x, from_y, transform_id)

        # Transform direction
        dir_idx_t = self._dir_maps[transform_id][dir_idx]

        # Check bounds
        if not (0 <= from_x_t < n and 0 <= from_y_t < n):
            return -1

        # Encode new index
        new_from_idx = from_y_t * n + from_x_t
        return (
            self.movement_base
            + new_from_idx * (NUM_SQUARE_DIRS * max_dist)
            + dir_idx_t * max_dist
            + dist_idx
        )

    def transform_move_index(self, move_idx: int, transform_id: int) -> int:
        """
        Transform a single move index.

        Args:
            move_idx: Policy index for the move
            transform_id: Transformation index 0-7

        Returns:
            Transformed move index, or -1 if out of bounds
        """
        if transform_id == 0:
            return move_idx

        if move_idx < 0:
            return -1

        if move_idx < self.placement_span:
            return self._transform_placement_index(move_idx, transform_id)
        elif move_idx < self.special_base:
            return self._transform_movement_index(move_idx, transform_id)
        else:
            return move_idx  # Special actions unchanged

    def transform_sparse_policy(
        self,
        policy_indices: np.ndarray,
        policy_values: np.ndarray,
        transform_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a sparse policy representation.

        Args:
            policy_indices: Array of move indices
            policy_values: Array of corresponding values/probabilities
            transform_id: Transformation index 0-7

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

        For rotations: 0->0, 1->3, 2->2, 3->1
        Reflections are self-inverse: 4->4, 5->5, 6->6, 7->7
        """
        inverse_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
        return inverse_map[transform_id]


def augment_square_sample(
    features: np.ndarray,
    globals_vec: np.ndarray,
    policy_indices: np.ndarray,
    policy_values: np.ndarray,
    transform: SquareSymmetryTransform | None = None,
    board_size: int = 8,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Apply all 8 D4 transformations to a single training sample.

    Args:
        features: Board features tensor (C, H, W)
        globals_vec: Global features vector
        policy_indices: Sparse policy move indices
        policy_values: Sparse policy values
        transform: SquareSymmetryTransform instance (created if not provided)
        board_size: Board size (default 8)

    Returns:
        List of 8 tuples, each containing
        (features, globals, policy_indices, policy_values)
    """
    if transform is None:
        transform = SquareSymmetryTransform(board_size)

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
