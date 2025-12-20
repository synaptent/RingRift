"""
Numba JIT-compiled evaluation functions for HeuristicAI performance optimization.

These functions are compiled to native machine code for significant speedups
in the most performance-critical evaluation loops.
"""


import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide fallback no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


# Pre-computed direction offsets for different board types
# Square board: 8 directions (N, NE, E, SE, S, SW, W, NW)
SQUARE_DIRECTIONS = np.array(
    [
        [0, -1],  # N
        [1, -1],  # NE
        [1, 0],  # E
        [1, 1],  # SE
        [0, 1],  # S
        [-1, 1],  # SW
        [-1, 0],  # W
        [-1, -1],  # NW
    ],
    dtype=np.int32,
)

# Hex board: 6 directions (axial coordinates)
HEX_DIRECTIONS = np.array(
    [
        [1, 0],  # E
        [1, -1],  # NE
        [0, -1],  # NW
        [-1, 0],  # W
        [-1, 1],  # SW
        [0, 1],  # SE
    ],
    dtype=np.int32,
)


@njit(cache=True)
def compute_offset_square(q: int, r: int, dir_idx: int, distance: int) -> tuple[int, int]:
    """Compute offset position for square board."""
    dq = SQUARE_DIRECTIONS[dir_idx, 0]
    dr = SQUARE_DIRECTIONS[dir_idx, 1]
    return q + dq * distance, r + dr * distance


@njit(cache=True)
def compute_offset_hex(q: int, r: int, dir_idx: int, distance: int) -> tuple[int, int]:
    """Compute offset position for hex board."""
    dq = HEX_DIRECTIONS[dir_idx, 0]
    dr = HEX_DIRECTIONS[dir_idx, 1]
    return q + dq * distance, r + dr * distance


@njit(cache=True)
def is_valid_square_coord(q: int, r: int, size: int) -> bool:
    """Check if coordinates are valid for square board."""
    return 0 <= q < size and 0 <= r < size


@njit(cache=True)
def is_valid_hex_coord(q: int, r: int, radius: int) -> bool:
    """Check if coordinates are valid for hex board (axial coordinates)."""
    # For hex board with radius R, valid coords satisfy: |q| <= R, |r| <= R, |q+r| <= R
    return abs(q) <= radius and abs(r) <= radius and abs(q + r) <= radius


@njit(cache=True)
def encode_key(q: int, r: int) -> int:
    """Encode (q, r) into a single integer key for fast lookup."""
    # Offset to handle negative coordinates (hex boards)
    # Using 16 bits per coordinate allows range [-32768, 32767]
    return ((q + 32768) << 16) | (r + 32768)


@njit(cache=True)
def decode_key(key: int) -> tuple[int, int]:
    """Decode integer key back to (q, r)."""
    q = (key >> 16) - 32768
    r = (key & 0xFFFF) - 32768
    return q, r


@njit(cache=True)
def evaluate_line_potential_numba(
    marker_positions: np.ndarray,  # Shape (n, 2) - q, r coordinates
    all_marker_set: np.ndarray,  # Set of all marker keys (encoded)
    board_type_is_hex: bool,
    board_size: int,
    weight_two: float,
    weight_three: float,
    weight_four: float,
) -> float:
    """
    JIT-compiled line potential evaluation.

    Args:
        marker_positions: Array of (q, r) coordinates for player's markers
        all_marker_set: Array of encoded keys for ALL markers (for lookup)
        board_type_is_hex: True for hex board, False for square
        board_size: Board size (side length for square, radius for hex)
        weight_two: Weight for 2-in-a-row
        weight_three: Weight for 3-in-a-row
        weight_four: Weight for 4-in-a-row

    Returns:
        Total line potential score
    """
    score = 0.0
    num_markers = len(marker_positions)
    num_directions = 6 if board_type_is_hex else 8

    # Create a set for fast marker lookup (using sorted array + binary search)
    marker_key_set = np.empty(len(all_marker_set), dtype=np.int64)
    for i in range(len(all_marker_set)):
        marker_key_set[i] = all_marker_set[i]
    marker_key_set.sort()

    for i in range(num_markers):
        q = marker_positions[i, 0]
        r = marker_positions[i, 1]

        for dir_idx in range(num_directions):
            # Check positions 1, 2, 3 steps away
            if board_type_is_hex:
                q2, r2 = compute_offset_hex(q, r, dir_idx, 1)
                if not is_valid_hex_coord(q2, r2, board_size):
                    continue
            else:
                q2, r2 = compute_offset_square(q, r, dir_idx, 1)
                if not is_valid_square_coord(q2, r2, board_size):
                    continue

            key2 = encode_key(q2, r2)
            # Binary search for key2 in marker_key_set
            if not _binary_search(marker_key_set, key2):
                continue

            # Found 2 in a row
            score += weight_two

            # Check for 3 in a row
            if board_type_is_hex:
                q3, r3 = compute_offset_hex(q, r, dir_idx, 2)
                if not is_valid_hex_coord(q3, r3, board_size):
                    continue
            else:
                q3, r3 = compute_offset_square(q, r, dir_idx, 2)
                if not is_valid_square_coord(q3, r3, board_size):
                    continue

            key3 = encode_key(q3, r3)
            if not _binary_search(marker_key_set, key3):
                continue

            # Found 3 in a row
            score += weight_three

            # Check for 4 in a row
            if board_type_is_hex:
                q4, r4 = compute_offset_hex(q, r, dir_idx, 3)
                if not is_valid_hex_coord(q4, r4, board_size):
                    continue
            else:
                q4, r4 = compute_offset_square(q, r, dir_idx, 3)
                if not is_valid_square_coord(q4, r4, board_size):
                    continue

            key4 = encode_key(q4, r4)
            if not _binary_search(marker_key_set, key4):
                continue

            # Found 4 in a row
            score += weight_four

    return score


@njit(cache=True)
def _binary_search(sorted_arr: np.ndarray, value: int) -> bool:
    """Binary search for value in sorted array."""
    left = 0
    right = len(sorted_arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_arr[mid] == value:
            return True
        elif sorted_arr[mid] < value:
            left = mid + 1
        else:
            right = mid - 1

    return False


@njit(cache=True)
def compute_territory_connectivity_numba(
    territory_positions: np.ndarray,  # Shape (n, 2) - q, r coordinates
    board_type_is_hex: bool,
) -> int:
    """
    Count connected pairs in territory for connectivity scoring.

    Returns the number of adjacent pairs (useful for territory cohesion scoring).
    """
    n = len(territory_positions)
    if n == 0:
        return 0

    # Build set of territory keys
    territory_keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        territory_keys[i] = encode_key(territory_positions[i, 0], territory_positions[i, 1])
    territory_keys.sort()

    num_directions = 6 if board_type_is_hex else 8
    connected_pairs = 0

    for i in range(n):
        q = territory_positions[i, 0]
        r = territory_positions[i, 1]

        for dir_idx in range(num_directions):
            if board_type_is_hex:
                nq, nr = compute_offset_hex(q, r, dir_idx, 1)
            else:
                nq, nr = compute_offset_square(q, r, dir_idx, 1)

            neighbor_key = encode_key(nq, nr)
            if _binary_search(territory_keys, neighbor_key):
                connected_pairs += 1

    # Each pair counted twice, so divide by 2
    return connected_pairs // 2


@njit(cache=True)
def manhattan_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    """Manhattan distance for square board."""
    return abs(q2 - q1) + abs(r2 - r1)


@njit(cache=True)
def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex distance (axial coordinates)."""
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2


@njit(cache=True)
def compute_centroid_distance_sum_numba(
    positions: np.ndarray,  # Shape (n, 2)
    board_type_is_hex: bool,
) -> float:
    """
    Compute sum of distances from each position to centroid.
    Useful for evaluating marker/territory spread.
    """
    n = len(positions)
    if n == 0:
        return 0.0

    # Compute centroid
    cq = 0.0
    cr = 0.0
    for i in range(n):
        cq += positions[i, 0]
        cr += positions[i, 1]
    cq /= n
    cr /= n

    # Round to nearest integer for distance calculation
    cq_int = round(cq)
    cr_int = round(cr)

    total_dist = 0.0
    for i in range(n):
        q = positions[i, 0]
        r = positions[i, 1]
        if board_type_is_hex:
            total_dist += hex_distance(q, r, cq_int, cr_int)
        else:
            total_dist += manhattan_distance(q, r, cq_int, cr_int)

    return total_dist


@njit(cache=True, parallel=True)
def batch_evaluate_positions_numba(
    candidate_positions: np.ndarray,  # Shape (m, 2) - positions to evaluate
    existing_markers: np.ndarray,  # Shape (n, 2) - existing player markers
    all_markers_keys: np.ndarray,  # All marker keys for lookup
    board_type_is_hex: bool,
    board_size: int,
    weight_two: float,
    weight_three: float,
    weight_four: float,
) -> np.ndarray:
    """
    Batch evaluate line potential for multiple candidate positions.

    This is useful for move selection - evaluate many positions at once.
    Returns array of scores for each candidate position.
    """
    m = len(candidate_positions)
    scores = np.empty(m, dtype=np.float64)

    for idx in prange(m):
        # Create combined marker array (existing + candidate)
        n_existing = len(existing_markers)
        combined = np.empty((n_existing + 1, 2), dtype=np.int32)
        for i in range(n_existing):
            combined[i, 0] = existing_markers[i, 0]
            combined[i, 1] = existing_markers[i, 1]
        combined[n_existing, 0] = candidate_positions[idx, 0]
        combined[n_existing, 1] = candidate_positions[idx, 1]

        # Add candidate to marker set
        new_key = encode_key(candidate_positions[idx, 0], candidate_positions[idx, 1])
        all_markers_extended = np.empty(len(all_markers_keys) + 1, dtype=np.int64)
        for i in range(len(all_markers_keys)):
            all_markers_extended[i] = all_markers_keys[i]
        all_markers_extended[len(all_markers_keys)] = new_key

        scores[idx] = evaluate_line_potential_numba(
            combined,
            all_markers_extended,
            board_type_is_hex,
            board_size,
            weight_two,
            weight_three,
            weight_four,
        )

    return scores


# Helper function to convert Python data structures to Numba-compatible arrays
def prepare_marker_arrays(markers, player_id):
    """
    Convert marker list to numpy arrays for Numba functions.

    Args:
        markers: Dict mapping position keys to MarkerInfo objects
        player_id: Player ID (player number) to filter markers for

    Returns:
        Tuple of (player_marker_positions, all_marker_keys)
    """
    player_positions = []
    all_keys = []

    for key, marker in markers.items():
        # Parse key "q,r" format
        parts = key.split(",")
        q, r = int(parts[0]), int(parts[1])
        all_keys.append(encode_key(q, r))

        if marker.player == player_id:
            player_positions.append([q, r])

    player_arr = np.array(player_positions, dtype=np.int32) if player_positions else np.empty((0, 2), dtype=np.int32)
    keys_arr = np.array(all_keys, dtype=np.int64) if all_keys else np.empty(0, dtype=np.int64)

    return player_arr, keys_arr


def prepare_territory_array(territories, player_id):
    """
    Convert territory dict to numpy array for Numba functions.

    Args:
        territories: Dict mapping position keys to owner IDs
        player_id: Player ID to filter territories for

    Returns:
        numpy array of (q, r) positions
    """
    positions = []
    for key, owner in territories.items():
        if owner == player_id:
            parts = key.split(",")
            q, r = int(parts[0]), int(parts[1])
            positions.append([q, r])

    return np.array(positions, dtype=np.int32) if positions else np.empty((0, 2), dtype=np.int32)
