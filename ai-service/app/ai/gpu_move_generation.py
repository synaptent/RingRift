"""GPU move generation for parallel games.

This module provides move generation functions for the GPU parallel games
system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R12 refactoring.

Move generation per RR-CANON rules:
- R090-R092: Movement moves
- R100-R103: Capture moves
- R110-R115: Recovery slide moves
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .gpu_game_types import GamePhase, MoveType

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState


# =============================================================================
# Batch Move Generation
# =============================================================================


@dataclass
class BatchMoves:
    """Batch of candidate moves for parallel games.

    Moves are stored as index tensors for efficient GPU operations.
    """

    # Move tensors: (total_moves,)
    game_idx: torch.Tensor     # Which game each move belongs to
    move_type: torch.Tensor    # MoveType enum
    from_y: torch.Tensor       # Source position (or target for placement)
    from_x: torch.Tensor
    to_y: torch.Tensor         # Destination position
    to_x: torch.Tensor

    # Indexing
    moves_per_game: torch.Tensor  # (batch_size,) count of moves per game
    move_offsets: torch.Tensor    # (batch_size,) cumulative offset

    total_moves: int
    device: torch.device


def _empty_batch_moves(batch_size: int, device: torch.device) -> BatchMoves:
    """Return empty BatchMoves structure."""
    return BatchMoves(
        game_idx=torch.tensor([], dtype=torch.int32, device=device),
        move_type=torch.tensor([], dtype=torch.int8, device=device),
        from_y=torch.tensor([], dtype=torch.int32, device=device),
        from_x=torch.tensor([], dtype=torch.int32, device=device),
        to_y=torch.tensor([], dtype=torch.int32, device=device),
        to_x=torch.tensor([], dtype=torch.int32, device=device),
        moves_per_game=torch.zeros(batch_size, dtype=torch.int32, device=device),
        move_offsets=torch.zeros(batch_size, dtype=torch.int32, device=device),
        total_moves=0,
        device=device,
    )


def generate_placement_moves_batch(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Generate all valid placement moves for active games.

    Per RingRift rules, placement is valid on:
    - Empty positions (stack_owner == 0): can place 1-3 rings
    - Occupied positions (stack_owner > 0): can place exactly 1 ring on top

    Placement is NOT valid on:
    - Collapsed spaces (is_collapsed == True)
    - Positions at max stack height (5) - can't exceed max height

    Note: The GPU engine simplifies by allowing placement on ANY non-collapsed
    position below max height. The placement count (1 vs 1-3) is handled during
    move selection and application, not during move generation.

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for (optional)

    Returns:
        BatchMoves with all valid placements
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size

    # Find all valid placement positions per game:
    # - Must not be collapsed
    # - Must not contain a marker (placement never occurs onto an existing marker)
    # - Must not be at max stack height (5) - can't place on full stacks
    # - Game must be active
    # valid_positions: (batch_size, board_size, board_size) bool
    valid_positions = (
        (~state.is_collapsed)
        & (state.marker_owner == 0)
        & (state.stack_height < 5)  # Max stack height is 5 per RR-CANON rules
        & active_mask.view(-1, 1, 1)
    )

    # Get indices of all valid positions
    game_idx, y_idx, x_idx = torch.where(valid_positions)

    total_moves = len(game_idx)

    # Count moves per game for indexing
    moves_per_game = valid_positions.view(batch_size, -1).sum(dim=1)
    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), moves_per_game[:-1]]),
        dim=0
    )

    return BatchMoves(
        game_idx=game_idx.int(),
        move_type=torch.full((total_moves,), MoveType.PLACEMENT, dtype=torch.int8, device=device),
        from_y=y_idx.int(),
        from_x=x_idx.int(),
        to_y=y_idx.int(),  # For placements, to == from (the placement cell)
        to_x=x_idx.int(),
        moves_per_game=moves_per_game.int(),
        move_offsets=move_offsets.int(),
        total_moves=total_moves,
        device=device,
    )


# =============================================================================
# Movement Move Generation (RR-CANON-R090-R092)
# =============================================================================


# Pre-computed direction vectors for 8-directional movement
# Shape: (8, 2) for (dy, dx) pairs
DIRECTIONS = torch.tensor([
    [-1, 0], [-1, 1], [0, 1], [1, 1],
    [1, 0], [1, -1], [0, -1], [-1, -1]
], dtype=torch.int32)

# Device-cached directions to avoid repeated .to(device) calls
_DIRECTIONS_CACHE: dict = {}


def _get_directions(device: torch.device) -> torch.Tensor:
    """Get directions tensor for a specific device (cached)."""
    key = str(device)
    if key not in _DIRECTIONS_CACHE:
        _DIRECTIONS_CACHE[key] = DIRECTIONS.to(device)
    return _DIRECTIONS_CACHE[key]


def _validate_paths_vectorized_fast(
    state: BatchGameState,
    game_indices: torch.Tensor,  # (N,)
    from_y: torch.Tensor,        # (N,)
    from_x: torch.Tensor,        # (N,)
    to_y: torch.Tensor,          # (N,)
    to_x: torch.Tensor,          # (N,)
    max_path_len: int,
) -> torch.Tensor:
    """Fully vectorized path validation.

    Validates all paths in parallel using tensor operations.
    Per RR-CANON-R091, checks that no stacks block intermediate path cells.

    Args:
        state: BatchGameState
        game_indices: (N,) game index for each candidate move
        from_y, from_x: (N,) origin positions
        to_y, to_x: (N,) destination positions
        max_path_len: Maximum path length to check

    Returns:
        Boolean tensor (N,) - True if path is valid
    """
    device = state.device
    N = game_indices.shape[0]

    if N == 0:
        return torch.tensor([], dtype=torch.bool, device=device)

    # Compute direction vectors and distances
    dy = torch.sign(to_y - from_y)  # (N,)
    dx = torch.sign(to_x - from_x)  # (N,)
    dist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))  # (N,)

    # Generate step indices: (N, max_path_len-1) for steps 1 to max_path_len-1
    # We exclude step 0 (origin) and step dist (destination)
    steps = torch.arange(1, max_path_len, device=device).unsqueeze(0)  # (1, max_path_len-1)

    # Compute path cell coordinates for all moves at all steps
    # path_y[i, s] = from_y[i] + dy[i] * (s+1) for step s+1
    path_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps  # (N, max_path_len-1)
    path_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps  # (N, max_path_len-1)

    # Create mask for valid intermediate steps (step < dist for each move)
    # step s corresponds to distance s+1 from origin
    valid_step_mask = steps < dist.unsqueeze(1)  # (N, max_path_len-1)

    # Clamp coordinates to valid board range for indexing
    board_size = state.board_size
    path_y_clamped = torch.clamp(path_y, 0, board_size - 1).long()
    path_x_clamped = torch.clamp(path_x, 0, board_size - 1).long()
    game_idx_expanded = game_indices.unsqueeze(1).expand(-1, max_path_len - 1).long()

    # Look up stack owners at all path cells
    # Use advanced indexing to get (N, max_path_len-1) tensor of owners
    path_owners = state.stack_owner[game_idx_expanded, path_y_clamped, path_x_clamped]

    # Check for collapsed cells too
    path_collapsed = state.is_collapsed[game_idx_expanded, path_y_clamped, path_x_clamped]

    # A path cell is blocked if it has a stack (owner != 0) or is collapsed
    path_blocked = (path_owners != 0) | path_collapsed

    # Apply the valid step mask - only check cells that are actual intermediate steps
    # Set blocked to False for steps >= dist (those aren't part of the path)
    path_blocked = path_blocked & valid_step_mask

    # Path is valid if NO intermediate cells are blocked
    valid = ~path_blocked.any(dim=1)  # (N,)

    return valid


def generate_movement_moves_batch_vectorized(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Fully vectorized movement move generation.

    This replaces the Python-loop version with tensor operations for GPU speedup.

    Per RR-CANON-R090-R092:
    - Move in straight line (8 directions: N, NE, E, SE, S, SW, W, NW)
    - Distance must be >= stack height at origin
    - Cannot pass through ANY stacks on intermediate cells
    - Cannot land on ANY stacks
    - Once a ray hits a blocker (collapsed or stack), all further cells in that direction are invalid

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for

    Returns:
        BatchMoves with all valid movement moves
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    directions = _get_directions(device)

    # === Step 1: Find all player stacks across all games ===
    player_expanded = state.current_player.view(-1, 1, 1).expand(-1, board_size, board_size)

    player_stacks = (
        (state.stack_owner == player_expanded) &
        (state.stack_height > 0) &
        active_mask.view(-1, 1, 1)
    )

    # Handle must_move_from constraint
    has_constraint = (state.must_move_from_y >= 0)

    if has_constraint.any():
        y_grid = torch.arange(board_size, device=device).view(1, -1, 1).expand(batch_size, -1, board_size)
        x_grid = torch.arange(board_size, device=device).view(1, 1, -1).expand(batch_size, board_size, -1)

        must_y = state.must_move_from_y.view(-1, 1, 1).expand(-1, board_size, board_size)
        must_x = state.must_move_from_x.view(-1, 1, 1).expand(-1, board_size, board_size)

        constraint_mask = (y_grid == must_y) & (x_grid == must_x)
        constraint_applies = has_constraint.view(-1, 1, 1).expand(-1, board_size, board_size)
        player_stacks = player_stacks & (~constraint_applies | constraint_mask)

    stack_game_idx, stack_y, stack_x = torch.where(player_stacks)
    n_stacks = stack_game_idx.shape[0]

    if n_stacks == 0:
        return _empty_batch_moves(batch_size, device)

    stack_heights = state.stack_height[stack_game_idx, stack_y, stack_x]
    max_dist = board_size - 1
    n_dirs = 8

    # === Step 2: Compute ray-blocking distances for each (stack, direction) ===
    # For each stack and direction, find the first distance at which we hit a blocker
    # (out-of-bounds, collapsed, or occupied cell). Moves beyond that are invalid.

    # Shape: (N_stacks, 8)
    stack_game_idx_dir = stack_game_idx.unsqueeze(1).expand(-1, n_dirs).long()
    stack_y_dir = stack_y.unsqueeze(1).expand(-1, n_dirs)
    stack_x_dir = stack_x.unsqueeze(1).expand(-1, n_dirs)

    dir_dy = directions[:, 0].view(1, n_dirs)
    dir_dx = directions[:, 1].view(1, n_dirs)

    # For each (stack, direction), compute blocking distance
    # Start with max_dist + 1 (no blocker found)
    blocking_dist = torch.full((n_stacks, n_dirs), max_dist + 1, dtype=torch.int32, device=device)

    # Check each distance step to find first blocker
    # Shape after expansion: (N_stacks, 8, max_dist)
    steps = torch.arange(1, max_dist + 1, device=device)
    check_y = stack_y_dir.unsqueeze(2) + dir_dy.unsqueeze(2) * steps  # (N_stacks, 8, max_dist)
    check_x = stack_x_dir.unsqueeze(2) + dir_dx.unsqueeze(2) * steps

    # Out of bounds check
    out_of_bounds = (check_y < 0) | (check_y >= board_size) | (check_x < 0) | (check_x >= board_size)

    # Clamp for safe indexing
    check_y_safe = torch.clamp(check_y, 0, board_size - 1).long()
    check_x_safe = torch.clamp(check_x, 0, board_size - 1).long()
    game_idx_exp = stack_game_idx_dir.unsqueeze(2).expand(-1, -1, max_dist)

    # Check collapsed and occupied
    is_collapsed_check = state.is_collapsed[game_idx_exp, check_y_safe, check_x_safe]
    is_occupied_check = state.stack_owner[game_idx_exp, check_y_safe, check_x_safe] != 0

    # Cell is blocking if: out of bounds, collapsed, or occupied
    is_blocking = out_of_bounds | is_collapsed_check | is_occupied_check  # (N_stacks, 8, max_dist)

    # Find first blocking distance for each (stack, direction)
    # Use argmax on the blocking mask - it returns first True index
    # If no blocker, argmax returns 0, so we need to handle that
    has_any_blocker = is_blocking.any(dim=2)  # (N_stacks, 8)

    # For rays with blockers, find the first blocking step
    # argmax returns the index of first True, but we need to add 1 because steps start at 1
    first_blocker_idx = is_blocking.to(torch.int32).argmax(dim=2)  # (N_stacks, 8)
    blocking_dist = torch.where(
        has_any_blocker,
        first_blocker_idx + 1,  # +1 because step index 0 means distance 1
        torch.tensor(max_dist + 1, device=device, dtype=torch.int32)
    )

    # === Step 3: Generate valid moves ===
    # A move is valid if: distance >= stack_height AND distance < blocking_dist

    # Expand for all distances
    stack_heights_exp = stack_heights.unsqueeze(1).expand(-1, n_dirs)  # (N_stacks, 8)

    # For each (stack, direction), valid distances are [stack_height, blocking_dist)
    # Generate all candidate moves
    distances_tensor = torch.arange(1, max_dist + 1, device=device).view(1, 1, -1)  # (1, 1, max_dist)

    # Expand all to (N_stacks, 8, max_dist)
    stack_heights_full = stack_heights_exp.unsqueeze(2).expand(-1, -1, max_dist)
    blocking_dist_full = blocking_dist.unsqueeze(2).expand(-1, -1, max_dist)
    distances_full = distances_tensor.expand(n_stacks, n_dirs, -1)

    # Valid move mask
    valid_mask = (distances_full >= stack_heights_full) & (distances_full < blocking_dist_full)

    # Also need to check that destination is in bounds and not collapsed/occupied
    dest_y = stack_y_dir.unsqueeze(2) + dir_dy.unsqueeze(2) * distances_tensor
    dest_x = stack_x_dir.unsqueeze(2) + dir_dx.unsqueeze(2) * distances_tensor

    dest_in_bounds = (
        (dest_y >= 0) & (dest_y < board_size) &
        (dest_x >= 0) & (dest_x < board_size)
    )

    dest_y_safe = torch.clamp(dest_y, 0, board_size - 1).long()
    dest_x_safe = torch.clamp(dest_x, 0, board_size - 1).long()

    dest_not_collapsed = ~state.is_collapsed[game_idx_exp, dest_y_safe, dest_x_safe]
    dest_not_occupied = state.stack_owner[game_idx_exp, dest_y_safe, dest_x_safe] == 0

    valid_mask = valid_mask & dest_in_bounds & dest_not_collapsed & dest_not_occupied

    # === Step 4: Extract valid moves ===
    valid_indices = torch.where(valid_mask.reshape(-1))[0]

    if valid_indices.numel() == 0:
        return _empty_batch_moves(batch_size, device)

    # Flatten source tensors
    game_idx_flat = stack_game_idx_dir.unsqueeze(2).expand(-1, -1, max_dist).reshape(-1)
    from_y_flat = stack_y_dir.unsqueeze(2).expand(-1, -1, max_dist).reshape(-1)
    from_x_flat = stack_x_dir.unsqueeze(2).expand(-1, -1, max_dist).reshape(-1)
    to_y_flat = dest_y.reshape(-1)
    to_x_flat = dest_x.reshape(-1)

    # Extract valid moves
    final_game_idx = game_idx_flat[valid_indices].int()
    final_from_y = from_y_flat[valid_indices].int()
    final_from_x = from_x_flat[valid_indices].int()
    final_to_y = to_y_flat[valid_indices].int()
    final_to_x = to_x_flat[valid_indices].int()

    total_moves = valid_indices.numel()

    # Count moves per game
    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    ones = torch.ones(total_moves, dtype=torch.int32, device=device)
    moves_per_game.scatter_add_(0, final_game_idx.long(), ones)

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device, dtype=torch.int32), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=final_game_idx,
        move_type=torch.full((total_moves,), MoveType.MOVEMENT, dtype=torch.int8, device=device),
        from_y=final_from_y,
        from_x=final_from_x,
        to_y=final_to_y,
        to_x=final_to_x,
        moves_per_game=moves_per_game,
        move_offsets=move_offsets,
        total_moves=total_moves,
        device=device,
    )


def _validate_paths_vectorized(
    state: BatchGameState,
    game_indices: torch.Tensor,
    from_positions: torch.Tensor,
    to_positions: torch.Tensor,
    players: torch.Tensor,
) -> torch.Tensor:
    """Validate movement paths in a semi-vectorized manner.

    Per RR-CANON-R091, checks that no stacks (own OR opponent) block the
    intermediate path cells. The destination is checked separately in
    generate_movement_moves_batch.

    This is a hybrid approach: we iterate over moves but use tensor indexing
    for path cell lookups to reduce Python overhead.

    Args:
        state: BatchGameState
        game_indices: (N,) game index for each candidate move
        from_positions: (N, 2) [y, x] origin positions
        to_positions: (N, 2) [y, x] destination positions
        players: (N,) player number for each move (unused - all stacks block)

    Returns:
        Boolean tensor (N,) - True if path is valid (no stacks on intermediate cells)
    """
    device = state.device
    N = game_indices.shape[0]

    if N == 0:
        return torch.tensor([], dtype=torch.bool, device=device)

    valid = torch.ones(N, dtype=torch.bool, device=device)

    # Process in chunks for better memory efficiency
    # For each move, we need to check all intermediate cells along the path
    for i in range(N):
        g = game_indices[i].item()
        from_y, from_x = from_positions[i, 0].item(), from_positions[i, 1].item()
        to_y, to_x = to_positions[i, 0].item(), to_positions[i, 1].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = int(max(abs(to_y - from_y), abs(to_x - from_x)))

        # Check each INTERMEDIATE path cell (excluding destination)
        # Per RR-CANON-R091: intermediate cells must contain no stack (any owner)
        for step in range(1, dist):  # Exclude destination (dist)
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            cell_owner = state.stack_owner[g, check_y, check_x].item()

            # ANY stack blocks the path (own or opponent)
            if cell_owner != 0:
                valid[i] = False
                break

    return valid


def generate_movement_moves_batch(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Generate all valid non-capture movement moves for active games.

    Per RR-CANON-R090-R092:
    - Move in straight line (8 directions: N, NE, E, SE, S, SW, W, NW)
    - Distance must be >= stack height at origin
    - Cannot pass through ANY stacks (own or opponent) on intermediate cells
    - Cannot land on ANY stacks (landing on opponent is capture, not movement)

    This function delegates to the fully vectorized implementation for GPU performance.
    Set RINGRIFT_GPU_MOVEMENT_LEGACY=1 to use the old Python-loop implementation.

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for

    Returns:
        BatchMoves with all valid movement moves
    """
    if os.environ.get("RINGRIFT_GPU_MOVEMENT_LEGACY", "0") == "1":
        return _generate_movement_moves_batch_legacy(state, active_mask)
    return generate_movement_moves_batch_vectorized(state, active_mask)


def _generate_movement_moves_batch_legacy(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Legacy Python-loop based movement generation.

    Kept for debugging and comparison. Use generate_movement_moves_batch_vectorized
    for production GPU workloads.
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size

    # 8 directions: (dy, dx)
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    candidate_game_idx = []
    candidate_from = []
    candidate_to = []
    candidate_player = []

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        player = state.current_player[g].item()
        must_y = int(state.must_move_from_y[g].item())
        must_x = int(state.must_move_from_x[g].item())

        my_stacks = (state.stack_owner[g] == player)
        stack_positions = torch.nonzero(my_stacks, as_tuple=False)

        for pos_idx in range(stack_positions.shape[0]):
            from_y = stack_positions[pos_idx, 0].item()
            from_x = stack_positions[pos_idx, 1].item()
            if must_y >= 0 and (from_y != must_y or from_x != must_x):
                continue
            stack_height = state.stack_height[g, from_y, from_x].item()

            if stack_height <= 0:
                continue

            for dy, dx in directions:
                for dist in range(stack_height, board_size):
                    to_y = from_y + dy * dist
                    to_x = from_x + dx * dist

                    if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                        break

                    if state.is_collapsed[g, to_y, to_x].item():
                        break

                    dest_owner = state.stack_owner[g, to_y, to_x].item()
                    if dest_owner != 0:
                        break

                    candidate_game_idx.append(g)
                    candidate_from.append([from_y, from_x])
                    candidate_to.append([to_y, to_x])
                    candidate_player.append(player)

    if not candidate_game_idx:
        return _empty_batch_moves(batch_size, device)

    game_idx_t = torch.tensor(candidate_game_idx, dtype=torch.int64, device=device)
    from_t = torch.tensor(candidate_from, dtype=torch.int64, device=device)
    to_t = torch.tensor(candidate_to, dtype=torch.int64, device=device)
    torch.tensor(candidate_player, dtype=torch.int64, device=device)

    # Use fully vectorized fast path validation (avoids .item() calls)
    valid_mask = _validate_paths_vectorized_fast(
        state, game_idx_t,
        from_t[:, 0], from_t[:, 1],  # from_y, from_x
        to_t[:, 0], to_t[:, 1],      # to_y, to_x
        max_path_len=board_size
    )
    valid_indices = torch.where(valid_mask)[0]

    if valid_indices.numel() == 0:
        return _empty_batch_moves(batch_size, device)

    valid_game_idx = game_idx_t[valid_indices].int()
    valid_from_y = from_t[valid_indices, 0].int()
    valid_from_x = from_t[valid_indices, 1].int()
    valid_to_y = to_t[valid_indices, 0].int()
    valid_to_x = to_t[valid_indices, 1].int()

    total_moves = valid_indices.numel()

    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    ones = torch.ones(total_moves, dtype=torch.int32, device=device)
    moves_per_game.scatter_add_(0, valid_game_idx.long(), ones)

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device, dtype=torch.int32), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=valid_game_idx,
        move_type=torch.full((total_moves,), MoveType.MOVEMENT, dtype=torch.int8, device=device),
        from_y=valid_from_y,
        from_x=valid_from_x,
        to_y=valid_to_y,
        to_x=valid_to_x,
        moves_per_game=moves_per_game,
        move_offsets=move_offsets,
        total_moves=total_moves,
        device=device,
    )


# =============================================================================
# Capture Move Generation (RR-CANON-R100-R103)
# =============================================================================


def generate_capture_moves_batch(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Generate all valid capture moves for active games.

    Per RR-CANON-R100-R103:
    - Capture by "overtaking": choose (from, target, landing) where landing is empty beyond target
      and attacker cap_height >= target cap_height (target may be any owner, including self).
    - Move in straight line, distance >= stack height (total height, not cap)
    - Path must be clear of ANY stacks (no passing through own or opponent stacks)
    - This generator stores only (from -> landing); the implicit target is the first stack along the ray.

    This function delegates to the fully vectorized implementation for GPU performance.
    Set RINGRIFT_GPU_CAPTURE_LEGACY=1 to use the old Python-loop implementation.

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for

    Returns:
        BatchMoves with all valid capture moves
    """
    if os.environ.get("RINGRIFT_GPU_CAPTURE_LEGACY", "0") == "1":
        return _generate_capture_moves_batch_legacy(state, active_mask)
    return generate_capture_moves_batch_vectorized(state, active_mask)


def generate_capture_moves_batch_vectorized(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Fully vectorized capture move generation.

    Per RR-CANON-R100-R103:
    - Find target: first stack along ray where my_cap_height >= target_cap_height
    - Landing: empty cell beyond target at distance >= my_height
    - Path from origin to target must be clear (no stacks or collapsed)
    - Path from target to landing must be clear (no stacks or collapsed beyond target)

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for

    Returns:
        BatchMoves with all valid capture moves
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    directions = _get_directions(device)
    max_dist = board_size - 1
    n_dirs = 8

    # === Step 1: Find all player stacks ===
    player_expanded = state.current_player.view(-1, 1, 1).expand(-1, board_size, board_size)
    player_stacks = (
        (state.stack_owner == player_expanded) &
        (state.stack_height > 0) &
        active_mask.view(-1, 1, 1)
    )

    # Handle must_move_from constraint
    has_constraint = (state.must_move_from_y >= 0)
    if has_constraint.any():
        y_grid = torch.arange(board_size, device=device).view(1, -1, 1).expand(batch_size, -1, board_size)
        x_grid = torch.arange(board_size, device=device).view(1, 1, -1).expand(batch_size, board_size, -1)
        must_y = state.must_move_from_y.view(-1, 1, 1).expand(-1, board_size, board_size)
        must_x = state.must_move_from_x.view(-1, 1, 1).expand(-1, board_size, board_size)
        constraint_mask = (y_grid == must_y) & (x_grid == must_x)
        constraint_applies = has_constraint.view(-1, 1, 1).expand(-1, board_size, board_size)
        player_stacks = player_stacks & (~constraint_applies | constraint_mask)

    stack_game_idx, stack_y, stack_x = torch.where(player_stacks)
    n_stacks = stack_game_idx.shape[0]

    if n_stacks == 0:
        return _empty_batch_moves(batch_size, device)

    # Get stack properties
    stack_heights = state.stack_height[stack_game_idx, stack_y, stack_x]
    stack_cap_heights = state.cap_height[stack_game_idx, stack_y, stack_x]

    # === Step 2: For each (stack, direction), find target and compute captures ===
    # Expand for directions: (N_stacks, 8)
    stack_game_idx_dir = stack_game_idx.unsqueeze(1).expand(-1, n_dirs).long()
    stack_y_dir = stack_y.unsqueeze(1).expand(-1, n_dirs)
    stack_x_dir = stack_x.unsqueeze(1).expand(-1, n_dirs)
    stack_heights_dir = stack_heights.unsqueeze(1).expand(-1, n_dirs)
    stack_cap_heights_dir = stack_cap_heights.unsqueeze(1).expand(-1, n_dirs)

    dir_dy = directions[:, 0].view(1, n_dirs)
    dir_dx = directions[:, 1].view(1, n_dirs)

    # Compute cells along each ray: (N_stacks, 8, max_dist)
    steps = torch.arange(1, max_dist + 1, device=device)
    ray_y = stack_y_dir.unsqueeze(2) + dir_dy.unsqueeze(2) * steps
    ray_x = stack_x_dir.unsqueeze(2) + dir_dx.unsqueeze(2) * steps
    game_idx_exp = stack_game_idx_dir.unsqueeze(2).expand(-1, -1, max_dist)

    # Check bounds
    in_bounds = (ray_y >= 0) & (ray_y < board_size) & (ray_x >= 0) & (ray_x < board_size)

    # Safe indexing
    ray_y_safe = torch.clamp(ray_y, 0, board_size - 1).long()
    ray_x_safe = torch.clamp(ray_x, 0, board_size - 1).long()

    # Get cell properties along rays
    ray_collapsed = state.is_collapsed[game_idx_exp, ray_y_safe, ray_x_safe]
    ray_owner = state.stack_owner[game_idx_exp, ray_y_safe, ray_x_safe]
    ray_cap_height = state.cap_height[game_idx_exp, ray_y_safe, ray_x_safe]

    # Find first blocker (collapsed or OOB) along each ray
    is_blocker = ~in_bounds | ray_collapsed
    blocker_cumsum = torch.cumsum(is_blocker.to(torch.int32), dim=2)
    before_blocker = blocker_cumsum == 0  # True for cells before any blocker

    # Find cells with ANY stacks - per RR-CANON-R101: "Self-capture is legal: target may be owned by P."
    # Self-captures are valid (can overtake own stacks if cap_height allows)
    has_stack = (ray_owner != 0) & before_blocker

    # Find target: first stack along ray where my_cap_height >= target_cap_height
    my_cap_exp = stack_cap_heights_dir.unsqueeze(2).expand(-1, -1, max_dist)
    valid_target = has_stack & (my_cap_exp >= ray_cap_height)

    # Find first valid target distance for each (stack, direction)
    # Use argmax - but need to handle case where no valid target exists
    has_any_target = valid_target.any(dim=2)  # (N_stacks, 8)
    first_target_idx = valid_target.to(torch.int32).argmax(dim=2)  # (N_stacks, 8)
    target_dist = first_target_idx + 1  # Convert index to distance

    # Also need to check that there's no stack before the target that we can't capture
    # (a stack with target_cap > my_cap blocks the ray even if there's a valid target beyond)
    invalid_stack = has_stack & (my_cap_exp < ray_cap_height)
    invalid_stack_cumsum = torch.cumsum(invalid_stack.to(torch.int32), dim=2)

    # Target is only valid if no invalid stack exists before it
    target_idx_exp = first_target_idx.unsqueeze(2)  # (N_stacks, 8, 1)
    step_indices = torch.arange(max_dist, device=device).view(1, 1, -1)  # (1, 1, max_dist)
    before_target = step_indices < target_idx_exp
    invalid_before_target = (invalid_stack_cumsum * before_target.to(torch.int32)).sum(dim=2) > 0

    # Final valid target mask
    has_valid_target = has_any_target & ~invalid_before_target  # (N_stacks, 8)

    # === Step 3: For (stack, direction) pairs with valid targets, enumerate landings ===
    # Flatten to process only valid (stack, direction) pairs
    valid_sd_indices = torch.where(has_valid_target.reshape(-1))[0]

    if valid_sd_indices.numel() == 0:
        return _empty_batch_moves(batch_size, device)

    # Extract valid (stack, direction) data
    n_valid_sd = valid_sd_indices.numel()
    sd_game_idx = stack_game_idx_dir.reshape(-1)[valid_sd_indices]
    sd_from_y = stack_y_dir.reshape(-1)[valid_sd_indices]
    sd_from_x = stack_x_dir.reshape(-1)[valid_sd_indices]
    sd_height = stack_heights_dir.reshape(-1)[valid_sd_indices]
    sd_target_dist = target_dist.reshape(-1)[valid_sd_indices]
    sd_dir_dy = dir_dy.expand(n_stacks, -1).reshape(-1)[valid_sd_indices]
    sd_dir_dx = dir_dx.expand(n_stacks, -1).reshape(-1)[valid_sd_indices]

    # Landing distance constraints per RR-CANON-R100-R103:
    # - Landing must be past target: landing_dist >= target_dist + 1
    # - Landing distance must be >= stack_height (overtaking rule)
    # So min_landing = max(stack_height, target_dist + 1)
    # There is NO upper bound - captures can land anywhere >= min_landing
    sd_min_landing = torch.maximum(sd_height, sd_target_dist + 1)

    # Expand for all possible landing distances: (n_valid_sd, max_dist)
    landing_dists = torch.arange(1, max_dist + 1, device=device).view(1, -1)  # (1, max_dist)

    # Expand sd data: (n_valid_sd, max_dist)
    sd_game_idx_exp = sd_game_idx.unsqueeze(1).expand(-1, max_dist)
    sd_from_y_exp = sd_from_y.unsqueeze(1).expand(-1, max_dist)
    sd_from_x_exp = sd_from_x.unsqueeze(1).expand(-1, max_dist)
    sd_target_dist.unsqueeze(1).expand(-1, max_dist)
    sd_min_landing_exp = sd_min_landing.unsqueeze(1).expand(-1, max_dist)
    sd_dir_dy_exp = sd_dir_dy.unsqueeze(1).expand(-1, max_dist)
    sd_dir_dx_exp = sd_dir_dx.unsqueeze(1).expand(-1, max_dist)

    # Compute landing positions
    landing_y = sd_from_y_exp + sd_dir_dy_exp * landing_dists
    landing_x = sd_from_x_exp + sd_dir_dx_exp * landing_dists

    # Filter 1: landing_dist >= min_landing (no upper bound)
    # This ensures: landing > target AND landing >= stack_height
    valid_landing_dist = landing_dists >= sd_min_landing_exp

    # Filter 2: landing in bounds
    landing_in_bounds = (
        (landing_y >= 0) & (landing_y < board_size) &
        (landing_x >= 0) & (landing_x < board_size)
    )

    # Safe indexing for landing
    landing_y_safe = torch.clamp(landing_y, 0, board_size - 1).long()
    landing_x_safe = torch.clamp(landing_x, 0, board_size - 1).long()

    # Filter 3: landing not collapsed
    landing_collapsed = state.is_collapsed[sd_game_idx_exp.long(), landing_y_safe, landing_x_safe]
    landing_not_collapsed = ~landing_collapsed

    # Filter 4: landing is empty
    landing_owner = state.stack_owner[sd_game_idx_exp.long(), landing_y_safe, landing_x_safe]
    landing_empty = landing_owner == 0

    # Combine filters so far
    valid_mask = valid_landing_dist & landing_in_bounds & landing_not_collapsed & landing_empty

    # === Step 4: Path validation from target+1 to landing-1 ===
    # For each candidate landing, check that cells between target and landing are clear
    # This is cells at distances (target_dist+1) to (landing_dist-1)

    # Compute path cells for all (sd, landing_dist) combinations
    # We need to check if any cell at dist in (target_dist, landing_dist) is blocked
    # Use the ray data we already computed

    # Get ray data for these (stack, direction) pairs
    # valid_sd_indices maps to (stack_idx * n_dirs + dir_idx)
    stack_idx = valid_sd_indices // n_dirs
    dir_idx = valid_sd_indices % n_dirs

    # For each valid (sd, landing), check path
    # Path is blocked if any cell at dist in [target_dist+1, landing_dist) has stack or collapsed
    # Use the ray_owner and ray_collapsed we computed earlier

    # Index into ray data: (n_valid_sd, max_dist)
    ray_owner_sd = ray_owner[stack_idx, dir_idx, :]  # (n_valid_sd, max_dist)
    ray_collapsed_sd = ray_collapsed[stack_idx, dir_idx, :]
    ray_in_bounds_sd = in_bounds[stack_idx, dir_idx, :]

    # For each candidate (n_valid_sd, max_dist landing), check path
    # Path cell at step s (0-indexed) corresponds to distance s+1
    # We need to check steps where: target_dist < step+1 < landing_dist
    # i.e., steps where: target_dist-1 < step < landing_dist-1
    # i.e., steps in range [target_dist, landing_dist-1)

    step_indices_2d = torch.arange(max_dist, device=device).view(1, 1, -1).expand(n_valid_sd, max_dist, -1)
    target_dist_3d = sd_target_dist.view(-1, 1, 1).expand(-1, max_dist, max_dist)
    landing_dist_3d = landing_dists.view(1, -1, 1).expand(n_valid_sd, -1, max_dist)

    # Mask for steps that are in the path (between target and landing)
    in_path = (step_indices_2d >= target_dist_3d) & (step_indices_2d < landing_dist_3d - 1)

    # Check if path cells are blocked (have stack or collapsed or OOB)
    ray_owner_3d = ray_owner_sd.unsqueeze(1).expand(-1, max_dist, -1)
    ray_collapsed_3d = ray_collapsed_sd.unsqueeze(1).expand(-1, max_dist, -1)
    ray_in_bounds_3d = ray_in_bounds_sd.unsqueeze(1).expand(-1, max_dist, -1)

    path_cell_blocked = (ray_owner_3d != 0) | ray_collapsed_3d | ~ray_in_bounds_3d
    path_blocked_at_step = path_cell_blocked & in_path

    # Path is blocked if any step in path is blocked
    path_blocked = path_blocked_at_step.any(dim=2)  # (n_valid_sd, max_dist)

    # === Step 5: Blocking distance for landings ===
    # In legacy, if we hit a collapsed or occupied landing, we break
    # But we only iterate starting from min_landing, so cells before don't count as blockers
    # Only consider cells at distance >= min_landing as potential blockers
    landing_invalid = ~(landing_in_bounds & landing_not_collapsed & landing_empty)
    # Mask out cells before min_landing - they don't count as blockers
    landing_invalid_in_range = landing_invalid & (landing_dists >= sd_min_landing_exp)
    landing_invalid_cumsum = torch.cumsum(landing_invalid_in_range.to(torch.int32), dim=1)
    before_first_invalid_landing = landing_invalid_cumsum == 0

    # Combine all filters
    valid_mask = valid_mask & ~path_blocked & before_first_invalid_landing

    # === Step 6: Extract valid captures ===
    valid_indices = torch.where(valid_mask.reshape(-1))[0]

    if valid_indices.numel() == 0:
        return _empty_batch_moves(batch_size, device)

    # Flatten and extract
    game_idx_flat = sd_game_idx_exp.reshape(-1)
    from_y_flat = sd_from_y_exp.reshape(-1)
    from_x_flat = sd_from_x_exp.reshape(-1)
    to_y_flat = landing_y.reshape(-1)
    to_x_flat = landing_x.reshape(-1)

    final_game_idx = game_idx_flat[valid_indices].int()
    final_from_y = from_y_flat[valid_indices].int()
    final_from_x = from_x_flat[valid_indices].int()
    final_to_y = to_y_flat[valid_indices].int()
    final_to_x = to_x_flat[valid_indices].int()

    total_moves = valid_indices.numel()

    # Count moves per game
    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    ones = torch.ones(total_moves, dtype=torch.int32, device=device)
    moves_per_game.scatter_add_(0, final_game_idx.long(), ones)

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device, dtype=torch.int32), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=final_game_idx,
        move_type=torch.full((total_moves,), MoveType.CAPTURE, dtype=torch.int8, device=device),
        from_y=final_from_y,
        from_x=final_from_x,
        to_y=final_to_y,
        to_x=final_to_x,
        moves_per_game=moves_per_game,
        move_offsets=move_offsets,
        total_moves=total_moves,
        device=device,
    )


def _generate_capture_moves_batch_legacy(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Legacy Python-loop based capture generation.

    Kept for debugging and comparison.
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size

    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    all_game_idx = []
    all_from_y = []
    all_from_x = []
    all_to_y = []
    all_to_x = []

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        player = state.current_player[g].item()
        must_y = int(state.must_move_from_y[g].item())
        must_x = int(state.must_move_from_x[g].item())

        my_stacks = (state.stack_owner[g] == player)
        stack_positions = torch.nonzero(my_stacks, as_tuple=False)

        for pos_idx in range(stack_positions.shape[0]):
            from_y = stack_positions[pos_idx, 0].item()
            from_x = stack_positions[pos_idx, 1].item()
            if must_y >= 0 and (from_y != must_y or from_x != must_x):
                continue
            my_height = state.stack_height[g, from_y, from_x].item()
            my_cap_height = state.cap_height[g, from_y, from_x].item()

            if my_height <= 0:
                continue

            for dy, dx in directions:
                target_y = None
                target_dist = 0

                for step in range(1, board_size):
                    check_y = from_y + dy * step
                    check_x = from_x + dx * step

                    if not (0 <= check_y < board_size and 0 <= check_x < board_size):
                        break

                    if state.is_collapsed[g, check_y, check_x].item():
                        break

                    cell_owner = state.stack_owner[g, check_y, check_x].item()
                    if cell_owner != 0:
                        target_cap = state.cap_height[g, check_y, check_x].item()
                        if my_cap_height >= target_cap:
                            target_y = check_y
                            target_dist = step
                        break

                if target_y is None:
                    continue

                min_landing_dist = max(my_height, target_dist + 1)

                for landing_dist in range(min_landing_dist, board_size):
                    landing_y = from_y + dy * landing_dist
                    landing_x = from_x + dx * landing_dist

                    if not (0 <= landing_y < board_size and 0 <= landing_x < board_size):
                        break

                    if state.is_collapsed[g, landing_y, landing_x].item():
                        break

                    path_clear = True
                    for step in range(target_dist + 1, landing_dist):
                        check_y = from_y + dy * step
                        check_x = from_x + dx * step
                        if state.stack_owner[g, check_y, check_x].item() != 0:
                            path_clear = False
                            break
                        if state.is_collapsed[g, check_y, check_x].item():
                            path_clear = False
                            break

                    if not path_clear:
                        break

                    landing_owner = state.stack_owner[g, landing_y, landing_x].item()
                    if landing_owner != 0:
                        break

                    all_game_idx.append(g)
                    all_from_y.append(from_y)
                    all_from_x.append(from_x)
                    all_to_y.append(landing_y)
                    all_to_x.append(landing_x)

    total_moves = len(all_game_idx)

    if total_moves == 0:
        return _empty_batch_moves(batch_size, device)

    game_idx_t = torch.tensor(all_game_idx, dtype=torch.int32, device=device)
    from_y_t = torch.tensor(all_from_y, dtype=torch.int32, device=device)
    from_x_t = torch.tensor(all_from_x, dtype=torch.int32, device=device)
    to_y_t = torch.tensor(all_to_y, dtype=torch.int32, device=device)
    to_x_t = torch.tensor(all_to_x, dtype=torch.int32, device=device)

    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for g in all_game_idx:
        moves_per_game[g] += 1

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=game_idx_t,
        move_type=torch.full((total_moves,), MoveType.CAPTURE, dtype=torch.int8, device=device),
        from_y=from_y_t,
        from_x=from_x_t,
        to_y=to_y_t,
        to_x=to_x_t,
        moves_per_game=moves_per_game,
        move_offsets=move_offsets,
        total_moves=total_moves,
        device=device,
    )


def generate_chain_capture_moves_from_position(
    state: BatchGameState,
    game_idx: int,
    from_y: int,
    from_x: int,
) -> list[tuple[int, int]]:
    """Generate all valid chain capture moves from a specific position.

    Used for chain capture continuation per RR-CANON-R103:
    - After executing an 'overtaking_capture' segment, if additional legal capture
      segments exist from the new landing position, the chain must continue.

    This function checks captures only from the specified position, not all stacks.
    Uses cap_height comparison per RR-CANON-R101.

    Optimized 2025-12-13: Pre-extract numpy arrays to avoid .item() calls.

    Args:
        state: Current batch game state
        game_idx: Game index in batch
        from_y: Row position of the stack to check captures from
        from_x: Column position of the stack to check captures from

    Returns:
        List of (landing_y, landing_x) positions for valid capture segments
    """
    board_size = state.board_size

    # Pre-extract game state as numpy to avoid repeated .item() calls
    player = int(state.current_player[game_idx].item())
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()
    cap_height_np = state.cap_height[game_idx].cpu().numpy()
    is_collapsed_np = state.is_collapsed[game_idx].cpu().numpy()

    # Verify we control this stack
    if stack_owner_np[from_y, from_x] != player:
        return []

    my_height = int(stack_height_np[from_y, from_x])
    my_cap_height = int(cap_height_np[from_y, from_x])
    if my_height <= 0:
        return []

    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    captures: list[tuple[int, int]] = []

    for dy, dx in directions:
        # Step 1: Find the first stack along this ray (implicit target).
        target_y = None
        target_x = None
        target_dist = 0

        for step in range(1, board_size):
            check_y = from_y + dy * step
            check_x = from_x + dx * step

            if not (0 <= check_y < board_size and 0 <= check_x < board_size):
                break

            if is_collapsed_np[check_y, check_x]:
                break

            cell_owner = stack_owner_np[check_y, check_x]
            if cell_owner != 0:
                # Per RR-CANON-R101: "Self-capture is legal: target may be owned by P."
                # Check cap_height comparison for any stack (own or enemy)
                target_cap = cap_height_np[check_y, check_x]
                if my_cap_height >= target_cap:
                    target_y = check_y
                    target_x = check_x
                    target_dist = step
                # Any stack (own or enemy) stops the search along this ray.
                break

        if target_y is None or target_x is None:
            continue

        # Step 2: Enumerate legal landing positions strictly beyond target.
        min_landing_dist = max(my_height, target_dist + 1)

        for landing_dist in range(min_landing_dist, board_size):
            landing_y = from_y + dy * landing_dist
            landing_x = from_x + dx * landing_dist

            if not (0 <= landing_y < board_size and 0 <= landing_x < board_size):
                break

            if is_collapsed_np[landing_y, landing_x]:
                break

            # Ensure the path from target -> landing is clear (no stacks, no collapsed spaces).
            path_clear = True
            for step in range(target_dist + 1, landing_dist):
                check_y = from_y + dy * step
                check_x = from_x + dx * step
                if stack_owner_np[check_y, check_x] != 0:
                    path_clear = False
                    break
                if is_collapsed_np[check_y, check_x]:
                    path_clear = False
                    break

            if not path_clear:
                break

            # Landing must be empty (markers are allowed).
            if stack_owner_np[landing_y, landing_x] != 0:
                break

            captures.append((landing_y, landing_x))

    return captures


def apply_single_chain_capture(
    state: BatchGameState,
    game_idx: int,
    from_y: int,
    from_x: int,
    to_y: int,
    to_x: int,
) -> tuple[int, int]:
    """Apply a single capture move for chain capture continuation.

    Per RR-CANON-R101/R102/R103 (overtaking capture):
    - Move the attacking stack from ``from`` to the landing cell ``(to_y,to_x)``.
    - The implicit target is the first stack between ``from`` and landing.
    - Pop the target's top ring and append it to the bottom of the attacking stack.
    - Process marker interactions along the path as in movement (R092), including
      the landing marker removal + cap-elimination cost.

    Optimized 2025-12-13: Pre-extract numpy arrays to avoid .item() calls in loops.

    Args:
        state: BatchGameState to modify
        game_idx: Game index in batch
        from_y, from_x: Origin position
        to_y, to_x: Landing position

    Returns:
        (new_y, new_x) landing position for potential chain continuation
    """
    # Pre-extract game slice as numpy for efficient reading (avoid .item() calls)
    player = int(state.current_player[game_idx].item())
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()
    cap_height_np = state.cap_height[game_idx].cpu().numpy()
    marker_owner_np = state.marker_owner[game_idx].cpu().numpy()
    is_collapsed_np = state.is_collapsed[game_idx].cpu().numpy()

    # Capture move representation:
    # - (from -> landing) is passed in as (to_y, to_x)
    # - The target stack is implicit as the first stack along the ray
    attacker_height = int(stack_height_np[from_y, from_x])
    attacker_cap_height = int(cap_height_np[from_y, from_x])

    dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
    dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
    dist = max(abs(to_y - from_y), abs(to_x - from_x))

    # Find target BEFORE recording to history (December 2025 bugfix)
    # December 2025 BUG FIX: Use LIVE state tensor for target finding, not the numpy
    # snapshot. The snapshot was taken at function entry, but in chain captures, the
    # previous capture may have already eliminated the target. Using the live tensor
    # ensures we find the CURRENT first stack along the ray.
    target_y = None
    target_x = None
    for step in range(1, dist):
        check_y = from_y + dy * step
        check_x = from_x + dx * step
        if state.stack_owner[game_idx, check_y, check_x].item() != 0:
            target_y = check_y
            target_x = check_x
            break

    if target_y is None or target_x is None:
        # No valid target found - this should not happen if generate_chain_capture_moves_from_position
        # is working correctly. Log warning and return without recording invalid move.
        import logging
        logging.warning(
            f"apply_single_chain_capture: No target found from ({from_y},{from_x}) to ({to_y},{to_x}) "
            f"in game {game_idx}. This indicates a bug in chain capture generation."
        )
        # Still move the stack as a fallback, but don't record as capture
        state.stack_height[game_idx, to_y, to_x] = attacker_height
        state.stack_owner[game_idx, to_y, to_x] = player
        state.cap_height[game_idx, to_y, to_x] = min(attacker_cap_height, attacker_height)
        state.stack_height[game_idx, from_y, from_x] = 0
        state.stack_owner[game_idx, from_y, from_x] = 0
        state.cap_height[game_idx, from_y, from_x] = 0
        state.marker_owner[game_idx, from_y, from_x] = player
        return to_y, to_x

    # Record in history AFTER verifying target exists (December 2025 bugfix)
    # Chain captures use CONTINUE_CAPTURE_SEGMENT and CHAIN_CAPTURE phase
    mc = int(state.move_count[game_idx].item())
    if mc < state.max_history_moves:
        state.move_history[game_idx, mc, 0] = MoveType.CONTINUE_CAPTURE_SEGMENT
        state.move_history[game_idx, mc, 1] = player
        state.move_history[game_idx, mc, 2] = from_y
        state.move_history[game_idx, mc, 3] = from_x
        state.move_history[game_idx, mc, 4] = to_y
        state.move_history[game_idx, mc, 5] = to_x
        state.move_history[game_idx, mc, 6] = GamePhase.CHAIN_CAPTURE
        # December 2025: Added capture_target columns for canonical export
        state.move_history[game_idx, mc, 7] = target_y
        state.move_history[game_idx, mc, 8] = target_x
    state.move_count[game_idx] += 1

    # Process markers along the full path excluding the implicit target cell.
    for step in range(1, dist):
        check_y = from_y + dy * step
        check_x = from_x + dx * step
        if check_y == target_y and check_x == target_x:
            continue

        marker_owner_val = marker_owner_np[check_y, check_x]
        if marker_owner_val == 0:
            continue
        if marker_owner_val != player:
            state.marker_owner[game_idx, check_y, check_x] = player
            continue

        # Own marker on intermediate cell: collapse to territory.
        state.marker_owner[game_idx, check_y, check_x] = 0
        if not is_collapsed_np[check_y, check_x]:
            state.is_collapsed[game_idx, check_y, check_x] = True
            state.territory_owner[game_idx, check_y, check_x] = player
            state.territory_count[game_idx, player] += 1

    # Landing marker: remove any marker and pay cap-elimination cost.
    dest_marker = marker_owner_np[to_y, to_x]
    landing_ring_cost = 1 if dest_marker != 0 else 0
    if landing_ring_cost:
        state.marker_owner[game_idx, to_y, to_x] = 0
        state.eliminated_rings[game_idx, player] += 1
        state.rings_caused_eliminated[game_idx, player] += 1

    # Pop top ring from the implicit target and append to the bottom of attacker.
    target_owner = int(stack_owner_np[target_y, target_x])
    target_height = int(stack_height_np[target_y, target_x])
    target_cap_height = int(cap_height_np[target_y, target_x])

    state.marker_owner[game_idx, target_y, target_x] = 0

    # December 2025: BUG FIX - When capturing the target's entire cap, ownership
    # transfers to the opponent.
    new_target_height = max(0, target_height - 1)
    state.stack_height[game_idx, target_y, target_x] = new_target_height

    # Check if target's cap was fully captured
    target_cap_fully_captured = target_cap_height <= 1  # Cap will be 0 after -1

    if new_target_height <= 0:
        state.stack_owner[game_idx, target_y, target_x] = 0
        state.cap_height[game_idx, target_y, target_x] = 0
        # BUG FIX 2025-12-20: Clear buried_at and decrement buried_rings when stack eliminated
        for p in range(1, state.num_players + 1):
            if state.buried_at[game_idx, p, target_y, target_x].item():
                state.buried_at[game_idx, p, target_y, target_x] = False
                state.buried_rings[game_idx, p] -= 1
    elif target_cap_fully_captured:
        # Cap captured, ownership transfers to opponent
        opponent = 1 if target_owner == 2 else 2
        state.stack_owner[game_idx, target_y, target_x] = opponent
        state.cap_height[game_idx, target_y, target_x] = new_target_height
        # BUG FIX 2025-12-20: If opponent had buried ring here, it's now exposed as cap
        if state.buried_at[game_idx, opponent, target_y, target_x].item():
            state.buried_at[game_idx, opponent, target_y, target_x] = False
            state.buried_rings[game_idx, opponent] -= 1
    else:
        # Cap not fully captured, defender keeps ownership
        new_target_cap = target_cap_height - 1
        if new_target_cap <= 0:
            new_target_cap = 1
        if new_target_cap > new_target_height:
            new_target_cap = new_target_height
        state.cap_height[game_idx, target_y, target_x] = new_target_cap

    if target_owner != 0 and target_owner != player:
        state.buried_rings[game_idx, target_owner] += 1
        # December 2025: Track buried ring position for recovery extraction
        state.buried_at[game_idx, target_owner, to_y, to_x] = True

    # December 2025: BUG FIX - When landing marker eliminates the attacker's entire cap,
    # ownership transfers to the target's original owner.
    new_height = attacker_height + 1 - landing_ring_cost
    state.stack_height[game_idx, to_y, to_x] = new_height

    # Check if landing cost eliminated entire cap
    cap_fully_eliminated = landing_ring_cost >= attacker_cap_height
    # December 2025: Check if attacker has buried rings (opponent's rings under their cap)
    attacker_has_buried = attacker_cap_height < attacker_height
    buried_count = attacker_height - attacker_cap_height

    if state.num_players == 2 and cap_fully_eliminated and attacker_has_buried:
        # December 2025: BUG FIX - When cap is eliminated AND attacker has buried
        # rings, ownership transfers to the opponent (who owns those buried rings).
        # The remaining stack: captured ring (bottom) + buried opponent rings (now cap)
        opponent = 1 if player == 2 else 2
        new_owner = opponent
        new_cap = buried_count
        # The buried rings are now exposed - clear buried tracking
        if state.buried_at[game_idx, opponent, to_y, to_x].item():
            state.buried_at[game_idx, opponent, to_y, to_x] = False
            state.buried_rings[game_idx, opponent] -= 1
    elif cap_fully_eliminated:
        # Ownership transfers to target owner, new cap is all remaining rings
        new_owner = target_owner
        new_cap = new_height
    elif target_owner == player and attacker_cap_height == attacker_height:
        # SELF-CAPTURE without buried rings:
        # Per RR-CANON-R101/R102, captured ring goes to bottom of stack.
        # If attacker has no buried rings (cap == height), and target is same color,
        # the entire resulting stack is same color, so cap = new_height.
        new_owner = player
        new_cap = new_height
        if new_cap <= 0:
            new_cap = 1
    else:
        # ENEMY CAPTURE or SELF-CAPTURE with buried rings:
        # Captured ring goes to bottom, doesn't extend the cap sequence from top.
        new_owner = player
        new_cap = attacker_cap_height - landing_ring_cost
        if new_cap <= 0:
            new_cap = 1
        if new_cap > new_height:
            new_cap = new_height

    state.stack_owner[game_idx, to_y, to_x] = new_owner
    state.cap_height[game_idx, to_y, to_x] = new_cap

    # Clear origin stack and leave departure marker.
    state.stack_height[game_idx, from_y, from_x] = 0
    state.stack_owner[game_idx, from_y, from_x] = 0
    state.cap_height[game_idx, from_y, from_x] = 0
    state.marker_owner[game_idx, from_y, from_x] = player

    # December 2025: Move buried_at tracking from origin to landing
    for p in range(1, state.num_players + 1):
        if state.buried_at[game_idx, p, from_y, from_x]:
            state.buried_at[game_idx, p, to_y, to_x] = True
            state.buried_at[game_idx, p, from_y, from_x] = False

    state.capture_chain_depth[game_idx] += 1

    return to_y, to_x


def apply_single_initial_capture(
    state: BatchGameState,
    game_idx: int,
    from_y: int,
    from_x: int,
    to_y: int,
    to_x: int,
) -> tuple[int, int]:
    """Apply a single INITIAL capture (not chain capture) from a position.

    This is similar to apply_single_chain_capture but uses:
    - MoveType.OVERTAKING_CAPTURE (not CONTINUE_CAPTURE_SEGMENT)
    - GamePhase.CAPTURE (not CHAIN_CAPTURE)

    Used for post-movement captures where we need to record the correct phase.

    December 2025: Added for post-movement capture parity with CPU phase machine.

    Args:
        state: BatchGameState to modify
        game_idx: Game index in batch
        from_y, from_x: Origin position
        to_y, to_x: Landing position

    Returns:
        (new_y, new_x) landing position for potential chain continuation
    """
    player = int(state.current_player[game_idx].item())
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()
    cap_height_np = state.cap_height[game_idx].cpu().numpy()
    marker_owner_np = state.marker_owner[game_idx].cpu().numpy()
    is_collapsed_np = state.is_collapsed[game_idx].cpu().numpy()

    attacker_height = int(stack_height_np[from_y, from_x])
    attacker_cap_height = int(cap_height_np[from_y, from_x])

    dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
    dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
    dist = max(abs(to_y - from_y), abs(to_x - from_x))

    # Find target
    target_y = None
    target_x = None
    for step in range(1, dist):
        check_y = from_y + dy * step
        check_x = from_x + dx * step
        if stack_owner_np[check_y, check_x] != 0:
            target_y = check_y
            target_x = check_x
            break

    if target_y is None or target_x is None:
        import logging
        logging.warning(
            f"apply_single_initial_capture: No target found from ({from_y},{from_x}) to ({to_y},{to_x})"
        )
        return to_y, to_x

    # Record in history with OVERTAKING_CAPTURE and CAPTURE phase
    mc = int(state.move_count[game_idx].item())
    if mc < state.max_history_moves:
        state.move_history[game_idx, mc, 0] = MoveType.OVERTAKING_CAPTURE
        state.move_history[game_idx, mc, 1] = player
        state.move_history[game_idx, mc, 2] = from_y
        state.move_history[game_idx, mc, 3] = from_x
        state.move_history[game_idx, mc, 4] = to_y
        state.move_history[game_idx, mc, 5] = to_x
        state.move_history[game_idx, mc, 6] = GamePhase.CAPTURE
        state.move_history[game_idx, mc, 7] = target_y
        state.move_history[game_idx, mc, 8] = target_x
    state.move_count[game_idx] += 1

    # Process markers along path (same as chain capture)
    for step in range(1, dist):
        check_y = from_y + dy * step
        check_x = from_x + dx * step
        if check_y == target_y and check_x == target_x:
            continue

        marker_owner_val = marker_owner_np[check_y, check_x]
        if marker_owner_val == 0:
            continue
        if marker_owner_val != player:
            state.marker_owner[game_idx, check_y, check_x] = player
            continue
        state.marker_owner[game_idx, check_y, check_x] = 0
        if not is_collapsed_np[check_y, check_x]:
            state.is_collapsed[game_idx, check_y, check_x] = True
            state.territory_owner[game_idx, check_y, check_x] = player
            state.territory_count[game_idx, player] += 1

    # Landing marker handling
    dest_marker = marker_owner_np[to_y, to_x]
    landing_ring_cost = 1 if dest_marker != 0 else 0
    if landing_ring_cost:
        state.marker_owner[game_idx, to_y, to_x] = 0
        state.eliminated_rings[game_idx, player] += 1
        state.rings_caused_eliminated[game_idx, player] += 1

    # Pop top ring from target
    target_owner = int(stack_owner_np[target_y, target_x])
    target_height = int(stack_height_np[target_y, target_x])
    target_cap_height = int(cap_height_np[target_y, target_x])

    state.marker_owner[game_idx, target_y, target_x] = 0

    # December 2025: BUG FIX - When capturing the target's entire cap, ownership
    # transfers to the opponent.
    new_target_height = max(0, target_height - 1)
    state.stack_height[game_idx, target_y, target_x] = new_target_height

    # Check if target's cap was fully captured
    target_cap_fully_captured = target_cap_height <= 1  # Cap will be 0 after -1

    if new_target_height <= 0:
        state.stack_owner[game_idx, target_y, target_x] = 0
        state.cap_height[game_idx, target_y, target_x] = 0
        # BUG FIX 2025-12-20: Clear buried_at and decrement buried_rings when stack eliminated
        for p in range(1, state.num_players + 1):
            if state.buried_at[game_idx, p, target_y, target_x].item():
                state.buried_at[game_idx, p, target_y, target_x] = False
                state.buried_rings[game_idx, p] -= 1
    elif target_cap_fully_captured:
        # Cap captured, ownership transfers to opponent
        opponent = 1 if target_owner == 2 else 2
        state.stack_owner[game_idx, target_y, target_x] = opponent
        state.cap_height[game_idx, target_y, target_x] = new_target_height
        # BUG FIX 2025-12-20: If opponent had buried ring here, it's now exposed as cap
        if state.buried_at[game_idx, opponent, target_y, target_x].item():
            state.buried_at[game_idx, opponent, target_y, target_x] = False
            state.buried_rings[game_idx, opponent] -= 1
    else:
        # Cap not fully captured, defender keeps ownership
        new_target_cap = max(1, min(target_cap_height - 1, new_target_height))
        state.cap_height[game_idx, target_y, target_x] = new_target_cap

    if target_owner != 0 and target_owner != player:
        state.buried_rings[game_idx, target_owner] += 1
        # December 2025: Track buried ring position for recovery extraction
        state.buried_at[game_idx, target_owner, to_y, to_x] = True

    # Set up landing stack
    # December 2025: BUG FIX - When landing marker eliminates the attacker's entire cap,
    # ownership transfers to the target's original owner.
    new_height = attacker_height + 1 - landing_ring_cost
    state.stack_height[game_idx, to_y, to_x] = new_height

    # Check if landing cost eliminated entire cap
    cap_fully_eliminated = landing_ring_cost >= attacker_cap_height
    # December 2025: Check if attacker has buried rings (opponent's rings under their cap)
    attacker_has_buried = attacker_cap_height < attacker_height
    buried_count = attacker_height - attacker_cap_height

    if state.num_players == 2 and cap_fully_eliminated and attacker_has_buried:
        # December 2025: BUG FIX - When cap is eliminated AND attacker has buried
        # rings, ownership transfers to the opponent (who owns those buried rings).
        # The remaining stack: captured ring (bottom) + buried opponent rings (now cap)
        opponent = 1 if player == 2 else 2
        new_owner = opponent
        new_cap = buried_count
        # The buried rings are now exposed - clear buried tracking
        if state.buried_at[game_idx, opponent, to_y, to_x].item():
            state.buried_at[game_idx, opponent, to_y, to_x] = False
            state.buried_rings[game_idx, opponent] -= 1
    elif cap_fully_eliminated:
        # Ownership transfers to target owner, new cap is all remaining rings
        new_owner = target_owner
        new_cap = new_height
    elif target_owner == player and attacker_cap_height == attacker_height:
        # SELF-CAPTURE without buried rings:
        # Per RR-CANON-R101/R102, captured ring goes to bottom of stack.
        # If attacker has no buried rings (cap == height), and target is same color,
        # the entire resulting stack is same color, so cap = new_height.
        new_owner = player
        new_cap = new_height
        if new_cap <= 0:
            new_cap = 1
    else:
        # ENEMY CAPTURE or SELF-CAPTURE with buried rings:
        # Captured ring goes to bottom, doesn't extend the cap sequence from top.
        new_owner = player
        new_cap = max(1, min(attacker_cap_height - landing_ring_cost, new_height))

    state.stack_owner[game_idx, to_y, to_x] = new_owner
    state.cap_height[game_idx, to_y, to_x] = new_cap

    # Clear origin
    state.stack_height[game_idx, from_y, from_x] = 0
    state.stack_owner[game_idx, from_y, from_x] = 0
    state.cap_height[game_idx, from_y, from_x] = 0
    state.marker_owner[game_idx, from_y, from_x] = player

    # December 2025: Move buried_at tracking from origin to landing
    for p in range(1, state.num_players + 1):
        if state.buried_at[game_idx, p, from_y, from_x]:
            state.buried_at[game_idx, p, to_y, to_x] = True
            state.buried_at[game_idx, p, from_y, from_x] = False

    # Mark as being in a capture chain (for subsequent chain captures)
    # After the initial capture, we transition to CHAIN_CAPTURE phase for any
    # subsequent captures, matching CPU phase machine behavior.
    state.in_capture_chain[game_idx] = True
    state.capture_chain_depth[game_idx] = 1
    state.current_phase[game_idx] = GamePhase.CHAIN_CAPTURE

    return to_y, to_x


# =============================================================================
# Recovery Slide Move Generation (RR-CANON-R110-R115)
# =============================================================================


def generate_recovery_moves_batch(
    state: BatchGameState,
    active_mask: torch.Tensor | None = None,
) -> BatchMoves:
    """Generate all valid recovery slide moves for eligible players.

    Per RR-CANON-R110-R115:
    - Player must have no controlled stacks
    - Player must have at least one marker on the board
    - Player must have buried rings (can afford the recovery cost)
    - Recovery eligibility is independent of rings in hand; players with rings
      in hand may reach recovery by recording skip_placement and then using
      recovery in movement (RR-CANON-R110).
    - Recovery slides a marker to an adjacent empty cell
    - "Line mode": slide completes a line of markers (preferred)
    - "Fallback mode": any adjacent slide if no line-forming recovery slide exists
      anywhere on the board (costs 1 buried ring)
    - Stack-strike (RR-CANON-R112(b2)): allow sliding onto an adjacent stack,
      sacrificing the marker to eliminate that stack's top ring.

    NOTE: This GPU implementation does not currently enforce the full
    fallback-class gate ("no line-forming recovery slide exists anywhere");
    it surfaces both empty-cell and stack-strike recovery options whenever the
    player is recovery-eligible.

    Optimized 2025-12-13: Pre-extract numpy arrays to avoid .item() calls.

    Args:
        state: Current batch game state
        active_mask: Mask of games to generate moves for

    Returns:
        BatchMoves with all valid recovery slide moves
    """
    if active_mask is None:
        active_mask = state.get_active_mask()

    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size

    # Line-length threshold per RR-CANON-R112:
    # - square8 2p: 4
    # - square8 3-4p: 3
    # - square19 / hex (GPU board_size != canonical embedding): 4
    num_players = int(getattr(state, "num_players", 2) or 2)
    required_line_length = 4 if board_size != 8 else (4 if num_players == 2 else 3)

    # 8 directions for sliding (Moore neighborhood)
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    # Unique line axes (direction + opposite) for line-formation checks.
    line_axes = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    # Pre-extract state arrays as numpy once to avoid repeated .item() calls
    active_mask_np = active_mask.cpu().numpy()
    current_player_np = state.current_player.cpu().numpy()
    stack_owner_np = state.stack_owner.cpu().numpy()
    stack_height_np = state.stack_height.cpu().numpy()
    marker_owner_np = state.marker_owner.cpu().numpy()
    territory_owner_np = state.territory_owner.cpu().numpy()
    buried_rings_np = state.buried_rings.cpu().numpy()
    is_collapsed_np = state.is_collapsed.cpu().numpy() if hasattr(state, 'is_collapsed') else None

    def _is_line_forming_recovery_slide_np(
        g: int,
        player: int,
        from_y: int,
        from_x: int,
        to_y: int,
        to_x: int,
    ) -> bool:
        """Return True when sliding marker to (to_y,to_x) forms a legal line.

        Uses numpy arrays instead of tensor .item() calls.
        """
        for dy, dx in line_axes:
            length = 1  # include destination marker

            # Forward direction
            y = to_y + dy
            x = to_x + dx
            while 0 <= y < board_size and 0 <= x < board_size:
                if is_collapsed_np is not None and is_collapsed_np[g, y, x]:
                    break
                if stack_owner_np[g, y, x] != 0:
                    break
                marker = marker_owner_np[g, y, x]
                if y == from_y and x == from_x:
                    marker = 0
                if marker != player:
                    break
                length += 1
                y += dy
                x += dx

            # Backward direction
            y = to_y - dy
            x = to_x - dx
            while 0 <= y < board_size and 0 <= x < board_size:
                if is_collapsed_np is not None and is_collapsed_np[g, y, x]:
                    break
                if stack_owner_np[g, y, x] != 0:
                    break
                marker = marker_owner_np[g, y, x]
                if y == from_y and x == from_x:
                    marker = 0
                if marker != player:
                    break
                length += 1
                y -= dy
                x -= dx

            if length >= required_line_length:
                return True

        return False

    all_game_idx = []
    all_from_y = []
    all_from_x = []
    all_to_y = []
    all_to_x = []

    for g in range(batch_size):
        if not active_mask_np[g]:
            continue

        player = int(current_player_np[g])

        # Check recovery eligibility per RR-CANON-R110:
        # 1. No controlled stacks
        has_stacks = (stack_owner_np[g] == player).any()
        if has_stacks:
            continue

        # 2. Has markers on board
        my_markers = (marker_owner_np[g] == player)
        marker_positions = np.argwhere(my_markers)
        if marker_positions.shape[0] == 0:
            continue

        # 3. Has buried rings (can afford recovery cost)
        buried_rings = buried_rings_np[g, player]
        if buried_rings <= 0:
            continue

        # Player is eligible for recovery.
        line_moves: list[tuple[int, int, int, int]] = []
        fallback_moves: list[tuple[int, int, int, int]] = []
        stack_strike_moves: list[tuple[int, int, int, int]] = []

        for pos_idx in range(marker_positions.shape[0]):
            from_y = int(marker_positions[pos_idx, 0])
            from_x = int(marker_positions[pos_idx, 1])

            for dy, dx in directions:
                to_y = from_y + dy
                to_x = from_x + dx

                # Check bounds
                if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                    continue

                # Target may be either empty cell or adjacent stack (stack-strike)
                if is_collapsed_np is not None and is_collapsed_np[g, to_y, to_x]:
                    continue

                has_stack = stack_height_np[g, to_y, to_x] > 0 and stack_owner_np[g, to_y, to_x] > 0

                if has_stack:
                    # Stack-strike: destination is a stack
                    if marker_owner_np[g, to_y, to_x] != 0:
                        continue
                    if territory_owner_np[g, to_y, to_x] != 0:
                        continue
                    stack_strike_moves.append((from_y, from_x, to_y, to_x))
                else:
                    # Empty-cell slide: destination must be empty
                    if stack_owner_np[g, to_y, to_x] != 0:
                        continue
                    if marker_owner_np[g, to_y, to_x] != 0:
                        continue
                    if territory_owner_np[g, to_y, to_x] != 0:
                        continue

                    if _is_line_forming_recovery_slide_np(
                        g,
                        player,
                        from_y,
                        from_x,
                        to_y,
                        to_x,
                    ):
                        line_moves.append((from_y, from_x, to_y, to_x))
                    else:
                        fallback_moves.append((from_y, from_x, to_y, to_x))

        selected_moves: list[tuple[int, int, int, int]]
        if line_moves:
            selected_moves = line_moves
        else:
            selected_moves = fallback_moves + stack_strike_moves

        for from_y, from_x, to_y, to_x in selected_moves:
            all_game_idx.append(g)
            all_from_y.append(from_y)
            all_from_x.append(from_x)
            all_to_y.append(to_y)
            all_to_x.append(to_x)

    total_moves = len(all_game_idx)

    if total_moves == 0:
        return BatchMoves(
            game_idx=torch.tensor([], dtype=torch.int32, device=device),
            move_type=torch.tensor([], dtype=torch.int8, device=device),
            from_y=torch.tensor([], dtype=torch.int32, device=device),
            from_x=torch.tensor([], dtype=torch.int32, device=device),
            to_y=torch.tensor([], dtype=torch.int32, device=device),
            to_x=torch.tensor([], dtype=torch.int32, device=device),
            moves_per_game=torch.zeros(batch_size, dtype=torch.int32, device=device),
            move_offsets=torch.zeros(batch_size, dtype=torch.int32, device=device),
            total_moves=0,
            device=device,
        )

    game_idx_t = torch.tensor(all_game_idx, dtype=torch.int32, device=device)
    from_y_t = torch.tensor(all_from_y, dtype=torch.int32, device=device)
    from_x_t = torch.tensor(all_from_x, dtype=torch.int32, device=device)
    to_y_t = torch.tensor(all_to_y, dtype=torch.int32, device=device)
    to_x_t = torch.tensor(all_to_x, dtype=torch.int32, device=device)

    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for g in all_game_idx:
        moves_per_game[g] += 1

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=game_idx_t,
        move_type=torch.full((total_moves,), MoveType.RECOVERY_SLIDE, dtype=torch.int8, device=device),
        from_y=from_y_t,
        from_x=from_x_t,
        to_y=to_y_t,
        to_x=to_x_t,
        moves_per_game=moves_per_game,
        move_offsets=move_offsets,
        total_moves=total_moves,
        device=device,
    )


__all__ = [
    'DIRECTIONS',
    'BatchMoves',
    '_empty_batch_moves',
    '_generate_capture_moves_batch_legacy',
    '_generate_movement_moves_batch_legacy',
    '_validate_paths_vectorized',
    '_validate_paths_vectorized_fast',
    'apply_single_chain_capture',
    'apply_single_initial_capture',
    'generate_capture_moves_batch',
    'generate_capture_moves_batch_vectorized',
    'generate_chain_capture_moves_from_position',
    'generate_movement_moves_batch',
    'generate_movement_moves_batch_vectorized',
    'generate_placement_moves_batch',
    'generate_recovery_moves_batch',
]
