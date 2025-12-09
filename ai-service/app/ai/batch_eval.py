"""
NumPy-based batch evaluation for processing multiple positions at once.

This module provides vectorized evaluation that processes all candidate moves
in parallel using NumPy operations, significantly faster than iterating
through moves individually on large boards.

Key optimizations:
1. Convert board state to NumPy arrays once
2. Represent moves as array indices
3. Compute features for all positions in parallel
4. Use broadcasting for weight application
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .lightweight_state import LightweightState

# Board type constants
BOARD_SQUARE8 = 0
BOARD_SQUARE19 = 1
BOARD_HEX = 2


class BoardArrays:
    """
    NumPy array representation of board state for fast batch operations.

    Arrays are indexed by flattened position index for O(1) access.
    """
    __slots__ = [
        'board_type', 'board_size', 'num_positions',
        'stack_owner', 'stack_height', 'marker_owner',
        'is_collapsed', 'territory_owner',
        'player_rings_in_hand', 'player_eliminated', 'player_territory',
        'position_to_idx', 'idx_to_position',
        'center_mask', 'neighbor_indices',
        'victory_rings', 'victory_territory',
    ]

    def __init__(self, board_size: int, board_type: int):
        self.board_type = board_type
        self.board_size = board_size

        # Calculate number of positions based on board type
        if board_type == BOARD_HEX:
            # Hex board with radius = size - 1
            radius = board_size - 1
            self.num_positions = 3 * radius * (radius + 1) + 1
        else:
            # Square board
            self.num_positions = board_size * board_size

        # Position mapping
        self.position_to_idx: Dict[str, int] = {}
        self.idx_to_position: List[str] = []

        # Board state arrays (initialized to 0 = empty/neutral)
        self.stack_owner = np.zeros(self.num_positions, dtype=np.int8)
        self.stack_height = np.zeros(self.num_positions, dtype=np.int8)
        self.marker_owner = np.zeros(self.num_positions, dtype=np.int8)
        self.is_collapsed = np.zeros(self.num_positions, dtype=np.bool_)
        self.territory_owner = np.zeros(self.num_positions, dtype=np.int8)

        # Player state arrays (indexed by player number, 0 unused)
        self.player_rings_in_hand = np.zeros(5, dtype=np.int16)  # Up to 4 players
        self.player_eliminated = np.zeros(5, dtype=np.int16)
        self.player_territory = np.zeros(5, dtype=np.int16)

        # Pre-computed masks
        self.center_mask = np.zeros(self.num_positions, dtype=np.bool_)

        # Neighbor indices for each position (for mobility calculation)
        # -1 means no neighbor (edge/invalid)
        max_neighbors = 8 if board_type != BOARD_HEX else 6
        self.neighbor_indices = np.full(
            (self.num_positions, max_neighbors), -1, dtype=np.int32
        )

        # Victory conditions
        self.victory_rings = 19
        self.victory_territory = 33

        # Build position mappings and neighbor indices
        self._build_position_mappings()
        self._build_center_mask()
        self._build_neighbor_indices()

    def _build_position_mappings(self):
        """Build bidirectional position key <-> index mappings."""
        idx = 0

        if self.board_type == BOARD_HEX:
            radius = self.board_size - 1
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    z = -x - y
                    if abs(x) <= radius and abs(y) <= radius and abs(z) <= radius:
                        key = f"{x},{y}"
                        self.position_to_idx[key] = idx
                        self.idx_to_position.append(key)
                        idx += 1
        else:
            for y in range(self.board_size):
                for x in range(self.board_size):
                    key = f"{x},{y}"
                    self.position_to_idx[key] = idx
                    self.idx_to_position.append(key)
                    idx += 1

    def _build_center_mask(self):
        """Build mask for center positions."""
        if self.board_type == BOARD_SQUARE8:
            center_keys = [
                "3,3", "3,4", "4,3", "4,4",
                "2,2", "2,3", "2,4", "2,5",
                "3,2", "3,5", "4,2", "4,5",
                "5,2", "5,3", "5,4", "5,5",
            ]
        elif self.board_type == BOARD_SQUARE19:
            center_keys = [f"{x},{y}" for x in range(7, 12) for y in range(7, 12)]
        else:  # Hex
            center_keys = [
                "0,0",
                "1,0", "0,1", "-1,1", "-1,0", "0,-1", "1,-1",
                "2,0", "1,1", "0,2", "-1,2", "-2,2", "-2,1",
                "-2,0", "-1,-1", "0,-2", "1,-2", "2,-2", "2,-1",
            ]

        for key in center_keys:
            if key in self.position_to_idx:
                self.center_mask[self.position_to_idx[key]] = True

    def _build_neighbor_indices(self):
        """Pre-compute neighbor indices for each position."""
        if self.board_type == BOARD_HEX:
            # Hex directions (axial coordinates)
            directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        else:
            # Square directions (8-connected)
            directions = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),          (1, 0),
                (-1, 1),  (0, 1), (1, 1),
            ]

        for key, idx in self.position_to_idx.items():
            parts = key.split(",")
            x, y = int(parts[0]), int(parts[1])

            for d_idx, (dx, dy) in enumerate(directions):
                nx, ny = x + dx, y + dy
                neighbor_key = f"{nx},{ny}"

                if neighbor_key in self.position_to_idx:
                    self.neighbor_indices[idx, d_idx] = self.position_to_idx[neighbor_key]

    @classmethod
    def from_lightweight_state(cls, state: 'LightweightState') -> 'BoardArrays':
        """Create BoardArrays from a LightweightState."""
        # Determine board type
        board_type_str = state.board_type.value
        if board_type_str == "square8":
            board_type = BOARD_SQUARE8
            board_size = 8
        elif board_type_str == "square19":
            board_type = BOARD_SQUARE19
            board_size = 19
        else:
            board_type = BOARD_HEX
            board_size = state.board_size

        arrays = cls(board_size, board_type)

        # Populate stack data
        for key, stack in state.stacks.items():
            if key in arrays.position_to_idx:
                idx = arrays.position_to_idx[key]
                arrays.stack_owner[idx] = stack.controlling_player
                arrays.stack_height[idx] = stack.stack_height

        # Populate marker data
        for key, marker in state.markers.items():
            if key in arrays.position_to_idx:
                idx = arrays.position_to_idx[key]
                arrays.marker_owner[idx] = marker.player

        # Populate collapsed spaces
        for key in state.collapsed_spaces:
            if key in arrays.position_to_idx:
                idx = arrays.position_to_idx[key]
                arrays.is_collapsed[idx] = True

        # Populate territory
        for key, owner in state.territories.items():
            if key in arrays.position_to_idx:
                idx = arrays.position_to_idx[key]
                arrays.territory_owner[idx] = owner

        # Populate player data
        for pnum, player in state.players.items():
            if 1 <= pnum <= 4:
                arrays.player_rings_in_hand[pnum] = player.rings_in_hand
                arrays.player_eliminated[pnum] = player.eliminated_rings
                arrays.player_territory[pnum] = player.territory_spaces

        arrays.victory_rings = state.victory_rings
        arrays.victory_territory = state.victory_territory

        return arrays

    def copy(self) -> 'BoardArrays':
        """Create a shallow copy with copied arrays."""
        new = BoardArrays.__new__(BoardArrays)
        new.board_type = self.board_type
        new.board_size = self.board_size
        new.num_positions = self.num_positions
        new.position_to_idx = self.position_to_idx  # Shared (immutable)
        new.idx_to_position = self.idx_to_position  # Shared (immutable)
        new.center_mask = self.center_mask  # Shared (immutable)
        new.neighbor_indices = self.neighbor_indices  # Shared (immutable)
        new.victory_rings = self.victory_rings
        new.victory_territory = self.victory_territory

        # Copy mutable arrays
        new.stack_owner = self.stack_owner.copy()
        new.stack_height = self.stack_height.copy()
        new.marker_owner = self.marker_owner.copy()
        new.is_collapsed = self.is_collapsed.copy()
        new.territory_owner = self.territory_owner.copy()
        new.player_rings_in_hand = self.player_rings_in_hand.copy()
        new.player_eliminated = self.player_eliminated.copy()
        new.player_territory = self.player_territory.copy()

        return new

    def update_from_lightweight_state(self, state: 'LightweightState') -> None:
        """
        Update arrays in-place from a LightweightState.

        This is faster than creating a new BoardArrays when the board
        geometry hasn't changed (same board type and size).

        Args:
            state: LightweightState to update from
        """
        # Reset mutable arrays to zero
        self.stack_owner.fill(0)
        self.stack_height.fill(0)
        self.marker_owner.fill(0)
        self.is_collapsed.fill(False)
        self.territory_owner.fill(0)
        self.player_rings_in_hand.fill(0)
        self.player_eliminated.fill(0)
        self.player_territory.fill(0)

        # Populate stack data
        for key, stack in state.stacks.items():
            if key in self.position_to_idx:
                idx = self.position_to_idx[key]
                self.stack_owner[idx] = stack.controlling_player
                self.stack_height[idx] = stack.stack_height

        # Populate marker data
        for key, marker in state.markers.items():
            if key in self.position_to_idx:
                idx = self.position_to_idx[key]
                self.marker_owner[idx] = marker.player

        # Populate collapsed spaces
        for key in state.collapsed_spaces:
            if key in self.position_to_idx:
                idx = self.position_to_idx[key]
                self.is_collapsed[idx] = True

        # Populate territory
        for key, owner in state.territories.items():
            if key in self.position_to_idx:
                idx = self.position_to_idx[key]
                self.territory_owner[idx] = owner

        # Populate player data
        for pnum, player in state.players.items():
            if 1 <= pnum <= 4:
                self.player_rings_in_hand[pnum] = player.rings_in_hand
                self.player_eliminated[pnum] = player.eliminated_rings
                self.player_territory[pnum] = player.territory_spaces

        self.victory_rings = state.victory_rings
        self.victory_territory = state.victory_territory

    def reset_for_reuse(self) -> None:
        """Reset all mutable arrays to zero for reuse from pool."""
        self.stack_owner.fill(0)
        self.stack_height.fill(0)
        self.marker_owner.fill(0)
        self.is_collapsed.fill(False)
        self.territory_owner.fill(0)
        self.player_rings_in_hand.fill(0)
        self.player_eliminated.fill(0)
        self.player_territory.fill(0)


def batch_evaluate_positions(
    base_arrays: BoardArrays,
    moves: List[Tuple[str, str, int, int]],  # (from_key, to_key, player, move_type)
    player_number: int,
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Evaluate multiple moves in batch using vectorized operations.

    Args:
        base_arrays: Base board state as NumPy arrays
        moves: List of (from_key, to_key, player, move_type) tuples
            move_type: 0=place_ring, 1=move_stack, 2=capture
        player_number: Player to evaluate for
        weights: Weight dictionary

    Returns:
        NumPy array of scores for each move
    """
    num_moves = len(moves)
    if num_moves == 0:
        return np.array([], dtype=np.float64)

    # Extract weights
    w_stack = weights.get('WEIGHT_STACK_CONTROL', 10.0)
    w_no_stacks = weights.get('WEIGHT_NO_STACKS_PENALTY', -50.0)
    w_single_stack = weights.get('WEIGHT_SINGLE_STACK_PENALTY', -10.0)
    w_territory = weights.get('WEIGHT_TERRITORY', 15.0)
    w_rings = weights.get('WEIGHT_RINGS_IN_HAND', 3.0)
    w_center = weights.get('WEIGHT_CENTER_CONTROL', 8.0)
    w_eliminated = weights.get('WEIGHT_ELIMINATED_RINGS', 20.0)
    w_marker = weights.get('WEIGHT_MARKER_COUNT', 2.0)
    w_victory = weights.get('WEIGHT_VICTORY_PROXIMITY', 25.0)
    w_victory_bonus = weights.get('WEIGHT_VICTORY_THRESHOLD_BONUS', 30.0)

    # Pre-compute base features that don't change
    base_my_stacks = np.sum(base_arrays.stack_owner == player_number)
    base_opp_stacks = np.sum((base_arrays.stack_owner > 0) & (base_arrays.stack_owner != player_number))
    base_my_markers = np.sum(base_arrays.marker_owner == player_number)
    base_opp_markers = np.sum((base_arrays.marker_owner > 0) & (base_arrays.marker_owner != player_number))
    base_my_center = np.sum((base_arrays.stack_owner == player_number) & base_arrays.center_mask)
    base_opp_center = np.sum((base_arrays.stack_owner > 0) & (base_arrays.stack_owner != player_number) & base_arrays.center_mask)
    base_my_territory = np.sum(base_arrays.territory_owner == player_number)
    base_opp_territory = np.sum((base_arrays.territory_owner > 0) & (base_arrays.territory_owner != player_number))

    # Initialize score arrays
    scores = np.zeros(num_moves, dtype=np.float64)

    # Delta arrays for each move
    delta_my_stacks = np.zeros(num_moves, dtype=np.int32)
    delta_opp_stacks = np.zeros(num_moves, dtype=np.int32)
    delta_my_markers = np.zeros(num_moves, dtype=np.int32)
    delta_my_center = np.zeros(num_moves, dtype=np.int32)
    delta_opp_center = np.zeros(num_moves, dtype=np.int32)
    delta_rings_in_hand = np.zeros(num_moves, dtype=np.int32)

    # Process each move to compute deltas
    for i, (from_key, to_key, move_player, move_type) in enumerate(moves):
        to_idx = base_arrays.position_to_idx.get(to_key, -1)
        from_idx = base_arrays.position_to_idx.get(from_key, -1) if from_key else -1

        if to_idx < 0:
            continue

        is_my_move = (move_player == player_number)

        if move_type == 0:  # place_ring
            # New stack at to_key
            if base_arrays.stack_owner[to_idx] == 0:
                # Creating new stack
                if is_my_move:
                    delta_my_stacks[i] += 1
                    delta_rings_in_hand[i] -= 1
                    if base_arrays.center_mask[to_idx]:
                        delta_my_center[i] += 1
                else:
                    delta_opp_stacks[i] += 1
                    if base_arrays.center_mask[to_idx]:
                        delta_opp_center[i] += 1
            else:
                # Adding to existing stack - may change control
                old_owner = base_arrays.stack_owner[to_idx]
                if is_my_move:
                    delta_rings_in_hand[i] -= 1
                    if old_owner != player_number:
                        # Taking control
                        delta_my_stacks[i] += 1
                        delta_opp_stacks[i] -= 1
                        if base_arrays.center_mask[to_idx]:
                            delta_my_center[i] += 1
                            delta_opp_center[i] -= 1

        elif move_type == 1:  # move_stack
            if from_idx >= 0:
                # Leave marker at from position
                if base_arrays.marker_owner[from_idx] == 0:
                    if is_my_move:
                        delta_my_markers[i] += 1

                # Stack moves from -> to
                if base_arrays.stack_owner[to_idx] == 0:
                    # Moving to empty - stack count unchanged
                    if base_arrays.center_mask[from_idx] and not base_arrays.center_mask[to_idx]:
                        if is_my_move:
                            delta_my_center[i] -= 1
                    elif not base_arrays.center_mask[from_idx] and base_arrays.center_mask[to_idx]:
                        if is_my_move:
                            delta_my_center[i] += 1

        elif move_type == 2:  # capture
            if from_idx >= 0:
                # Leave marker at from position
                if base_arrays.marker_owner[from_idx] == 0:
                    if is_my_move:
                        delta_my_markers[i] += 1

                # Capture: our stack takes over target stack
                target_owner = base_arrays.stack_owner[to_idx]
                if target_owner > 0 and target_owner != move_player:
                    if is_my_move:
                        # We capture opponent's stack
                        delta_opp_stacks[i] -= 1
                        if base_arrays.center_mask[to_idx]:
                            delta_opp_center[i] -= 1

    # Compute final features
    final_my_stacks = base_my_stacks + delta_my_stacks
    final_opp_stacks = base_opp_stacks + delta_opp_stacks
    final_my_markers = base_my_markers + delta_my_markers
    final_my_center = base_my_center + delta_my_center
    final_opp_center = base_opp_center + delta_opp_center
    final_rings = base_arrays.player_rings_in_hand[player_number] + delta_rings_in_hand

    # Stack control score
    scores += (final_my_stacks - final_opp_stacks) * w_stack
    scores += np.where(final_my_stacks == 0, w_no_stacks, 0)
    scores += np.where(final_my_stacks == 1, w_single_stack, 0)

    # Territory score (unchanged by most moves)
    scores += (base_my_territory - base_opp_territory) * w_territory

    # Rings in hand score
    opp_rings = 0
    opp_count = 0
    for pnum in range(1, 5):
        if pnum != player_number and base_arrays.player_rings_in_hand[pnum] > 0:
            opp_rings += base_arrays.player_rings_in_hand[pnum]
            opp_count += 1
    avg_opp_rings = opp_rings / max(1, opp_count)
    scores += (final_rings - avg_opp_rings) * w_rings

    # Center control score
    scores += (final_my_center - final_opp_center) * w_center

    # Eliminated rings score
    scores += base_arrays.player_eliminated[player_number] * w_eliminated

    # Marker count score
    scores += (final_my_markers - base_opp_markers) * w_marker

    # Victory proximity score
    rings_needed = max(0, base_arrays.victory_rings - base_arrays.player_eliminated[player_number])
    territory_needed = max(0, base_arrays.victory_territory - base_arrays.player_territory[player_number])

    if rings_needed <= 3 or territory_needed <= 5:
        scores += w_victory_bonus

    if rings_needed > 0:
        scores += (1.0 / rings_needed) * 10.0 * w_victory
    if territory_needed > 0:
        scores += (1.0 / territory_needed) * 5.0 * w_victory

    return scores


def prepare_moves_for_batch(
    moves: List,  # List of Move objects
    position_to_idx: Dict[str, int],
) -> List[Tuple[str, str, int, int]]:
    """
    Convert Move objects to tuples for batch processing.

    Args:
        moves: List of Move objects
        position_to_idx: Position key to index mapping

    Returns:
        List of (from_key, to_key, player, move_type) tuples
    """
    result = []

    for move in moves:
        to_key = move.to.to_key()
        from_key = move.from_pos.to_key() if move.from_pos else ""
        player = move.player

        # Determine move type
        move_type_str = move.type.value if hasattr(move.type, 'value') else str(move.type)

        if move_type_str == "place_ring":
            move_type = 0
        elif move_type_str in ("move_stack", "recovery_slide"):
            # recovery_slide is a marker movement (RR-CANON-R110–R115)
            move_type = 1
        elif move_type_str in ("overtaking_capture", "continue_capture_segment", "chain_capture"):
            move_type = 2
        else:
            # Skip non-board-changing moves (line processing, etc.)
            move_type = -1

        result.append((from_key, to_key, player, move_type))

    return result


# BoardArrays pool for reuse
class BoardArraysPool:
    """
    Pool of BoardArrays for reuse across evaluations.

    This avoids the overhead of allocating and garbage collecting
    BoardArrays objects on each move evaluation. Arrays are cached
    by board geometry (type + size) for fast lookup.
    """
    _instances: Dict[str, 'BoardArraysPool'] = {}

    def __init__(self, board_size: int, board_type: int, pool_size: int = 4):
        self.board_size = board_size
        self.board_type = board_type
        self.pool_size = pool_size

        # Pre-allocate pool of BoardArrays
        self._available: List[BoardArrays] = []
        self._in_use: List[BoardArrays] = []

        # Create initial pool
        for _ in range(pool_size):
            arrays = BoardArrays(board_size, board_type)
            self._available.append(arrays)

    @classmethod
    def get_pool(cls, board_size: int, board_type: int) -> 'BoardArraysPool':
        """Get or create a pool for the given board geometry."""
        key = f"{board_type}_{board_size}"
        if key not in cls._instances:
            cls._instances[key] = cls(board_size, board_type)
        return cls._instances[key]

    def acquire(self) -> BoardArrays:
        """
        Acquire a BoardArrays from the pool.

        Returns an available arrays object, or creates a new one if
        pool is exhausted.
        """
        if self._available:
            arrays = self._available.pop()
        else:
            # Pool exhausted - create new (will grow pool)
            arrays = BoardArrays(self.board_size, self.board_type)

        self._in_use.append(arrays)
        return arrays

    def release(self, arrays: BoardArrays) -> None:
        """
        Release a BoardArrays back to the pool.

        The arrays are reset for reuse.
        """
        if arrays in self._in_use:
            self._in_use.remove(arrays)

        # Reset and return to available pool
        arrays.reset_for_reuse()

        # Only keep up to pool_size in available
        if len(self._available) < self.pool_size:
            self._available.append(arrays)
        # Otherwise let it be garbage collected

    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            'available': len(self._available),
            'in_use': len(self._in_use),
            'pool_size': self.pool_size,
        }


def get_or_update_board_arrays(
    state: 'LightweightState',
    cached_arrays: Optional[BoardArrays] = None,
) -> BoardArrays:
    """
    Get BoardArrays for a state, reusing cached arrays if compatible.

    This is the primary entry point for lazy BoardArrays reuse.
    If cached_arrays is provided and compatible (same board geometry),
    it will be updated in-place. Otherwise, a new BoardArrays is created.

    Args:
        state: LightweightState to convert
        cached_arrays: Optional previously used BoardArrays to reuse

    Returns:
        BoardArrays populated from the state
    """
    # Determine board type
    board_type_str = state.board_type.value
    if board_type_str == "square8":
        board_type = BOARD_SQUARE8
        board_size = 8
    elif board_type_str == "square19":
        board_type = BOARD_SQUARE19
        board_size = 19
    else:
        board_type = BOARD_HEX
        board_size = state.board_size

    # Check if we can reuse cached arrays
    if (cached_arrays is not None and
            cached_arrays.board_type == board_type and
            cached_arrays.board_size == board_size):
        cached_arrays.update_from_lightweight_state(state)
        return cached_arrays

    # Create new arrays
    return BoardArrays.from_lightweight_state(state)
