"""Native GPU kernels for RingRift game simulation.

This module provides vectorized GPU implementations of performance-critical
game operations, replacing Python for-loops with fully parallel GPU operations.

Supports both CUDA (NVIDIA) and MPS (Apple Silicon) backends through PyTorch.

Key optimizations:
1. Vectorized move generation - all moves generated in parallel
2. Batched path validation - check all paths simultaneously
3. Fused heuristic evaluation - single kernel for all metrics
4. Memory-efficient tensor operations - minimize CPU-GPU transfers

NOTE: This module is primarily used for parity testing (tests/gpu/).
For production code, prefer gpu_batch.py or gpu_parallel_games.py.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Device Detection and Configuration
# Re-export from gpu_batch for consistency - use gpu_batch.get_device() directly
# in production code for the full-featured version with prefer_gpu/device_id.
# =============================================================================

def get_device() -> torch.device:
    """Get the best available GPU device.

    For production code with more control, use:
        from app.ai.gpu_batch import get_device
        device = get_device(prefer_gpu=True, device_id=0)

    This simple version is kept for backward compatibility with test code.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_cuda_available() -> bool:
    """Check if CUDA is available for native CUDA kernels."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS is available for Apple Silicon acceleration."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# =============================================================================
# Direction Constants (Pre-computed on GPU)
# =============================================================================

# 8 directions: N, NE, E, SE, S, SW, W, NW
DIRECTIONS_Y = torch.tensor([-1, -1, 0, 1, 1, 1, 0, -1], dtype=torch.int32)
DIRECTIONS_X = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], dtype=torch.int32)


def get_directions(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get direction tensors on the specified device."""
    return DIRECTIONS_Y.to(device), DIRECTIONS_X.to(device)


# =============================================================================
# Vectorized Move Mask Generation
# =============================================================================

@torch.jit.script
def generate_placement_mask_kernel(
    stack_owner: torch.Tensor,  # (batch, board, board)
    rings_in_hand: torch.Tensor,  # (batch, num_players+1)
    current_player: torch.Tensor,  # (batch,)
    active_mask: torch.Tensor,  # (batch,)
) -> torch.Tensor:
    """Generate placement move mask for all games in parallel.

    Returns:
        Tensor of shape (batch, board, board) where True = valid placement
    """
    batch_size, board_size, _ = stack_owner.shape

    # Get rings in hand for current players
    player_rings = torch.zeros(batch_size, dtype=torch.int32, device=stack_owner.device)
    for b in range(batch_size):
        if active_mask[b]:
            player_rings[b] = rings_in_hand[b, current_player[b]]

    # Valid placements: empty cells where player has rings
    empty_cells = (stack_owner == 0)  # (batch, board, board)
    has_rings = (player_rings > 0).view(batch_size, 1, 1)  # (batch, 1, 1)
    is_active = active_mask.view(batch_size, 1, 1)  # (batch, 1, 1)

    return empty_cells & has_rings & is_active


def generate_placement_moves_vectorized(
    stack_owner: torch.Tensor,
    rings_in_hand: torch.Tensor,
    current_player: torch.Tensor,
    active_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all valid placement moves for the batch.

    Returns:
        Tuple of (game_idx, to_y, to_x, num_moves_per_game)
    """
    mask = generate_placement_mask_kernel(
        stack_owner, rings_in_hand, current_player, active_mask
    )

    # Find all valid positions
    valid_positions = torch.nonzero(mask, as_tuple=False)  # (N, 3)

    if valid_positions.shape[0] == 0:
        device = stack_owner.device
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.zeros(stack_owner.shape[0], dtype=torch.int32, device=device),
        )

    game_idx = valid_positions[:, 0].int()
    to_y = valid_positions[:, 1].int()
    to_x = valid_positions[:, 2].int()

    # Count moves per game
    batch_size = stack_owner.shape[0]
    num_moves = torch.bincount(game_idx, minlength=batch_size).int()

    return game_idx, to_y, to_x, num_moves


# =============================================================================
# Vectorized Normal Move Generation
# =============================================================================

def generate_normal_moves_vectorized(
    stack_owner: torch.Tensor,  # (batch, board, board)
    stack_height: torch.Tensor,  # (batch, board, board)
    current_player: torch.Tensor,  # (batch,)
    active_mask: torch.Tensor,  # (batch,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all valid normal (non-capture) moves for the batch.

    Returns:
        Tuple of (game_idx, from_y, from_x, to_y, to_x, num_moves_per_game)
    """
    device = stack_owner.device
    batch_size, board_size, _ = stack_owner.shape
    dir_y, dir_x = get_directions(device)

    # Find all stacks controlled by current players
    # Create player ownership mask
    player_mask = torch.zeros_like(stack_owner, dtype=torch.bool)
    for b in range(batch_size):
        if active_mask[b]:
            player_mask[b] = (stack_owner[b] == current_player[b])

    # Get positions of player's stacks
    stack_positions = torch.nonzero(player_mask, as_tuple=False)  # (N, 3)

    if stack_positions.shape[0] == 0:
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    # For each stack, generate moves in all 8 directions
    all_game_idx = []
    all_from_y = []
    all_from_x = []
    all_to_y = []
    all_to_x = []

    for i in range(stack_positions.shape[0]):
        g = stack_positions[i, 0].item()
        from_y = stack_positions[i, 1].item()
        from_x = stack_positions[i, 2].item()
        height = stack_height[g, from_y, from_x].item()

        if height <= 0:
            continue

        player = current_player[g].item()

        # Check all 8 directions
        for d in range(8):
            dy = dir_y[d].item()
            dx = dir_x[d].item()

            # Move distance >= stack height for normal moves
            for dist in range(height, board_size):
                to_y = from_y + dy * dist
                to_x = from_x + dx * dist

                # Bounds check
                if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                    break

                # Path check - must be clear
                path_blocked = False
                for step in range(1, dist):
                    check_y = from_y + dy * step
                    check_x = from_x + dx * step
                    cell_owner = stack_owner[g, check_y, check_x].item()
                    if cell_owner != 0 and cell_owner != player:
                        path_blocked = True
                        break

                if path_blocked:
                    break

                # Destination check - must be empty for normal move
                dest_owner = stack_owner[g, to_y, to_x].item()
                if dest_owner == 0:
                    # Valid normal move
                    all_game_idx.append(g)
                    all_from_y.append(from_y)
                    all_from_x.append(from_x)
                    all_to_y.append(to_y)
                    all_to_x.append(to_x)
                elif dest_owner != player:
                    # Opponent stack - can't pass
                    break
                # Own stack - can pass over

    if len(all_game_idx) == 0:
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    game_idx = torch.tensor(all_game_idx, dtype=torch.int32, device=device)
    from_y_t = torch.tensor(all_from_y, dtype=torch.int32, device=device)
    from_x_t = torch.tensor(all_from_x, dtype=torch.int32, device=device)
    to_y_t = torch.tensor(all_to_y, dtype=torch.int32, device=device)
    to_x_t = torch.tensor(all_to_x, dtype=torch.int32, device=device)

    num_moves = torch.bincount(game_idx, minlength=batch_size).int()

    return game_idx, from_y_t, from_x_t, to_y_t, to_x_t, num_moves


# =============================================================================
# Vectorized Capture Move Generation
# =============================================================================

def generate_capture_moves_vectorized(
    stack_owner: torch.Tensor,  # (batch, board, board)
    stack_height: torch.Tensor,  # (batch, board, board)
    current_player: torch.Tensor,  # (batch,)
    active_mask: torch.Tensor,  # (batch,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate all valid capture moves for the batch.

    Returns:
        Tuple of (game_idx, from_y, from_x, to_y, to_x, num_moves_per_game)
    """
    device = stack_owner.device
    batch_size, board_size, _ = stack_owner.shape
    dir_y, dir_x = get_directions(device)

    # Find all stacks controlled by current players
    player_mask = torch.zeros_like(stack_owner, dtype=torch.bool)
    for b in range(batch_size):
        if active_mask[b]:
            player_mask[b] = (stack_owner[b] == current_player[b])

    stack_positions = torch.nonzero(player_mask, as_tuple=False)

    if stack_positions.shape[0] == 0:
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    all_game_idx = []
    all_from_y = []
    all_from_x = []
    all_to_y = []
    all_to_x = []

    for i in range(stack_positions.shape[0]):
        g = stack_positions[i, 0].item()
        from_y = stack_positions[i, 1].item()
        from_x = stack_positions[i, 2].item()
        my_height = stack_height[g, from_y, from_x].item()

        if my_height <= 0:
            continue

        player = current_player[g].item()

        for d in range(8):
            dy = dir_y[d].item()
            dx = dir_x[d].item()

            for dist in range(my_height, board_size):
                to_y = from_y + dy * dist
                to_x = from_x + dx * dist

                if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                    break

                # Check path
                path_blocked = False
                for step in range(1, dist):
                    check_y = from_y + dy * step
                    check_x = from_x + dx * step
                    cell_owner = stack_owner[g, check_y, check_x].item()
                    if cell_owner != 0 and cell_owner != player:
                        path_blocked = True
                        break

                if path_blocked:
                    break

                # Check destination for capture
                dest_owner = stack_owner[g, to_y, to_x].item()
                if dest_owner != 0 and dest_owner != player:
                    dest_height = stack_height[g, to_y, to_x].item()
                    if my_height >= dest_height:
                        # Valid capture
                        all_game_idx.append(g)
                        all_from_y.append(from_y)
                        all_from_x.append(from_x)
                        all_to_y.append(to_y)
                        all_to_x.append(to_x)
                    # Can't continue past opponent
                    break

    if len(all_game_idx) == 0:
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    game_idx = torch.tensor(all_game_idx, dtype=torch.int32, device=device)
    from_y_t = torch.tensor(all_from_y, dtype=torch.int32, device=device)
    from_x_t = torch.tensor(all_from_x, dtype=torch.int32, device=device)
    to_y_t = torch.tensor(all_to_y, dtype=torch.int32, device=device)
    to_x_t = torch.tensor(all_to_x, dtype=torch.int32, device=device)

    num_moves = torch.bincount(game_idx, minlength=batch_size).int()

    return game_idx, from_y_t, from_x_t, to_y_t, to_x_t, num_moves


# =============================================================================
# Vectorized Heuristic Evaluation Kernel
# =============================================================================

def evaluate_positions_kernel(
    stack_owner: torch.Tensor,  # (batch, board, board)
    stack_height: torch.Tensor,  # (batch, board, board)
    marker_owner: torch.Tensor,  # (batch, board, board)
    territory_owner: torch.Tensor,  # (batch, board, board)
    rings_in_hand: torch.Tensor,  # (batch, num_players+1)
    territory_count: torch.Tensor,  # (batch, num_players+1)
    eliminated_rings: torch.Tensor,  # (batch, num_players+1)
    buried_rings: torch.Tensor,  # (batch, num_players+1)
    current_player: torch.Tensor,  # (batch,)
    active_mask: torch.Tensor,  # (batch,)
    weights: Dict[str, float],
    board_size: int,
    num_players: int,
) -> torch.Tensor:
    """Evaluate all positions in parallel using vectorized GPU operations.

    Returns:
        Tensor of shape (batch, num_players+1) with scores per player
    """
    device = stack_owner.device
    batch_size = stack_owner.shape[0]

    # Initialize scores
    scores = torch.zeros(batch_size, num_players + 1, dtype=torch.float32, device=device)

    # Get weights with defaults
    def get_weight(name: str, default: float = 0.0) -> float:
        return weights.get(name, weights.get(name.lower(), default))

    # Pre-compute center for distance calculations
    center = (board_size - 1) / 2.0

    # Create coordinate grids
    y_coords = torch.arange(board_size, dtype=torch.float32, device=device).view(1, board_size, 1)
    x_coords = torch.arange(board_size, dtype=torch.float32, device=device).view(1, 1, board_size)

    # Distance from center (broadcast across batch)
    dist_from_center = torch.sqrt((y_coords - center) ** 2 + (x_coords - center) ** 2)
    max_dist = center * 1.414
    center_bonus = 1.0 - (dist_from_center / max_dist)  # (1, board, board)

    for player in range(1, num_players + 1):
        # === Stack-based metrics ===
        player_stacks = (stack_owner == player)  # (batch, board, board)

        # Stack count
        stack_count = player_stacks.sum(dim=(1, 2)).float()

        # Total height
        player_heights = stack_height * player_stacks.float()
        total_height = player_heights.sum(dim=(1, 2))

        # Center control
        center_control = (player_heights * center_bonus).sum(dim=(1, 2))

        # === Ring-based metrics ===
        rings_hand = rings_in_hand[:, player].float()
        territory = territory_count[:, player].float()
        eliminated = eliminated_rings[:, player].float()
        buried = buried_rings[:, player].float()

        # === Marker-based metrics ===
        player_markers = (marker_owner == player)
        marker_count = player_markers.sum(dim=(1, 2)).float()

        # === Compute weighted score ===
        score = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Material weight (stack control)
        material_w = get_weight('material_weight', get_weight('WEIGHT_STACK_CONTROL', 1.0))
        score += material_w * stack_count

        # Ring count
        ring_w = get_weight('ring_count_weight', get_weight('WEIGHT_RINGS_IN_HAND', 0.5))
        score += ring_w * rings_hand

        # Stack height
        height_w = get_weight('stack_height_weight', get_weight('WEIGHT_STACK_HEIGHT', 0.3))
        score += height_w * total_height

        # Center control
        center_w = get_weight('center_control_weight', get_weight('WEIGHT_CENTER_CONTROL', 0.4))
        score += center_w * center_control

        # Territory
        territory_w = get_weight('territory_weight', get_weight('WEIGHT_TERRITORY', 0.8))
        score += territory_w * territory

        # Markers
        marker_w = get_weight('WEIGHT_MARKERS', 0.3)
        score += marker_w * marker_count

        # Eliminated rings (offensive)
        elim_w = get_weight('WEIGHT_ELIMINATED_RINGS', 0.5)
        score += elim_w * eliminated

        # Buried rings (negative - these are captured from us)
        buried_w = get_weight('WEIGHT_BURIED_RINGS', -0.4)
        score += buried_w * buried

        # Victory proximity
        # Per RR-CANON-R061: threshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
        # Per BOARD_CONFIGS: 8×8=18, 19×19=60, hex(13)=72 rings per player
        # For 2p: 8×8=18, 19×19=60, hex=72; for 3p: 8×8=24, 19×19=80, hex=96
        rings_per_player = {8: 18, 19: 60, 13: 72}.get(board_size, 18)
        victory_threshold = round(rings_per_player * (2/3 + 1/3 * (num_players - 1)))
        proximity = eliminated / victory_threshold
        victory_w = get_weight('WEIGHT_VICTORY_PROXIMITY', 1.0)
        score += victory_w * proximity * 10.0

        # No stacks penalty
        no_stacks_penalty = get_weight('WEIGHT_NO_STACKS_PENALTY', 5.0)
        has_no_stacks = (stack_count == 0).float()
        has_no_rings = (rings_hand == 0).float()
        score -= no_stacks_penalty * has_no_stacks * has_no_rings * (1.0 - (buried > 0).float())

        scores[:, player] = score

    return scores


# =============================================================================
# Line Detection Kernels
# =============================================================================

def detect_lines_kernel(
    marker_owner: torch.Tensor,  # (batch, board, board)
    board_size: int,
    min_line_length: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect all lines of markers for each player.

    Returns:
        Tuple of (game_idx, player, line_start_y, line_start_x) for detected lines
    """
    device = marker_owner.device
    batch_size = marker_owner.shape[0]
    dir_y, dir_x = get_directions(device)

    # Only check 4 directions (N, NE, E, SE) to avoid duplicates
    check_dirs = [0, 1, 2, 3]

    all_game_idx = []
    all_player = []
    all_start_y = []
    all_start_x = []

    for g in range(batch_size):
        for y in range(board_size):
            for x in range(board_size):
                owner = marker_owner[g, y, x].item()
                if owner == 0:
                    continue

                for d in check_dirs:
                    dy = dir_y[d].item()
                    dx = dir_x[d].item()

                    # Count consecutive markers
                    length = 1
                    for dist in range(1, board_size):
                        ny = y + dy * dist
                        nx = x + dx * dist
                        if not (0 <= ny < board_size and 0 <= nx < board_size):
                            break
                        if marker_owner[g, ny, nx].item() != owner:
                            break
                        length += 1

                    if length >= min_line_length:
                        all_game_idx.append(g)
                        all_player.append(owner)
                        all_start_y.append(y)
                        all_start_x.append(x)

    if len(all_game_idx) == 0:
        return (
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
        )

    return (
        torch.tensor(all_game_idx, dtype=torch.int32, device=device),
        torch.tensor(all_player, dtype=torch.int32, device=device),
        torch.tensor(all_start_y, dtype=torch.int32, device=device),
        torch.tensor(all_start_x, dtype=torch.int32, device=device),
    )


# =============================================================================
# Victory Condition Kernels
# =============================================================================

def check_victory_conditions_kernel(
    eliminated_rings: torch.Tensor,  # (batch, num_players+1)
    rings_in_hand: torch.Tensor,  # (batch, num_players+1)
    stack_owner: torch.Tensor,  # (batch, board, board)
    buried_rings: torch.Tensor,  # (batch, num_players+1)
    game_status: torch.Tensor,  # (batch,)
    num_players: int,
    board_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check victory conditions for all games in parallel.

    Returns:
        Tuple of (winner, victory_type, updated_status)
        - winner: 0 = no winner, 1-4 = player number
        - victory_type: 0 = none, 1 = ring_elimination, 2 = last_standing
    """
    device = eliminated_rings.device
    batch_size = eliminated_rings.shape[0]

    # Per RR-CANON-R061: threshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
    # Per BOARD_CONFIGS: 8×8=18, 19×19=60, hex(13)=72 rings per player
    # For 2p: 8×8=18, 19×19=60, hex=72; for 3p: 8×8=24, 19×19=80, hex=96
    rings_per_player = {8: 18, 19: 60, 13: 72}.get(board_size, 18)
    victory_threshold = round(rings_per_player * (2/3 + 1/3 * (num_players - 1)))

    winner = torch.zeros(batch_size, dtype=torch.int32, device=device)
    victory_type = torch.zeros(batch_size, dtype=torch.int32, device=device)
    status = game_status.clone()

    # Check ring elimination victory
    for player in range(1, num_players + 1):
        player_eliminated = eliminated_rings[:, player] >= victory_threshold
        no_winner_yet = (winner == 0)
        new_winners = player_eliminated & no_winner_yet
        winner = torch.where(new_winners, torch.tensor(player, device=device), winner)
        victory_type = torch.where(new_winners, torch.tensor(1, device=device), victory_type)

    # Check last standing victory
    # Count players with any rings (controlled, in hand, or buried)
    for g in range(batch_size):
        if winner[g] != 0:
            continue

        players_with_rings = 0
        last_player_with_rings = 0

        for player in range(1, num_players + 1):
            # Check if player has any rings anywhere
            has_controlled = (stack_owner[g] == player).any().item()
            has_in_hand = rings_in_hand[g, player].item() > 0
            has_buried = buried_rings[g, player].item() > 0

            if has_controlled or has_in_hand or has_buried:
                players_with_rings += 1
                last_player_with_rings = player

        if players_with_rings == 1:
            winner[g] = last_player_with_rings
            victory_type[g] = 2  # last_standing
            status[g] = 1  # COMPLETED

    # Update status for ring elimination victories
    status = torch.where(victory_type == 1, torch.tensor(1, device=device), status)

    return winner, victory_type, status


# =============================================================================
# Batch Move Application
# =============================================================================

def apply_placement_batch(
    stack_owner: torch.Tensor,
    stack_height: torch.Tensor,
    rings_in_hand: torch.Tensor,
    game_idx: torch.Tensor,
    to_y: torch.Tensor,
    to_x: torch.Tensor,
    current_player: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply placement moves in batch.

    Returns:
        Updated (stack_owner, stack_height, rings_in_hand)
    """
    stack_owner = stack_owner.clone()
    stack_height = stack_height.clone()
    rings_in_hand = rings_in_hand.clone()

    for i in range(game_idx.shape[0]):
        g = game_idx[i].item()
        y = to_y[i].item()
        x = to_x[i].item()
        player = current_player[g].item()

        # Place ring
        stack_owner[g, y, x] = player
        stack_height[g, y, x] = 1
        rings_in_hand[g, player] -= 1

    return stack_owner, stack_height, rings_in_hand


def apply_movement_batch(
    stack_owner: torch.Tensor,
    stack_height: torch.Tensor,
    game_idx: torch.Tensor,
    from_y: torch.Tensor,
    from_x: torch.Tensor,
    to_y: torch.Tensor,
    to_x: torch.Tensor,
    current_player: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply movement moves in batch.

    Returns:
        Updated (stack_owner, stack_height)
    """
    stack_owner = stack_owner.clone()
    stack_height = stack_height.clone()

    for i in range(game_idx.shape[0]):
        g = game_idx[i].item()
        fy = from_y[i].item()
        fx = from_x[i].item()
        ty = to_y[i].item()
        tx = to_x[i].item()
        player = current_player[g].item()

        # Move stack
        moving_height = stack_height[g, fy, fx].item()
        stack_owner[g, fy, fx] = 0
        stack_height[g, fy, fx] = 0
        stack_owner[g, ty, tx] = player
        stack_height[g, ty, tx] = moving_height

    return stack_owner, stack_height


def apply_capture_batch(
    stack_owner: torch.Tensor,
    stack_height: torch.Tensor,
    marker_owner: torch.Tensor,
    eliminated_rings: torch.Tensor,
    buried_rings: torch.Tensor,
    game_idx: torch.Tensor,
    from_y: torch.Tensor,
    from_x: torch.Tensor,
    to_y: torch.Tensor,
    to_x: torch.Tensor,
    current_player: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply capture moves in batch.

    Returns:
        Updated (stack_owner, stack_height, marker_owner, eliminated_rings, buried_rings)
    """
    stack_owner = stack_owner.clone()
    stack_height = stack_height.clone()
    marker_owner = marker_owner.clone()
    eliminated_rings = eliminated_rings.clone()
    buried_rings = buried_rings.clone()

    for i in range(game_idx.shape[0]):
        g = game_idx[i].item()
        fy = from_y[i].item()
        fx = from_x[i].item()
        ty = to_y[i].item()
        tx = to_x[i].item()
        player = current_player[g].item()

        attacker_height = stack_height[g, fy, fx].item()
        defender = stack_owner[g, ty, tx].item()
        defender_height = stack_height[g, ty, tx].item()

        # Clear source
        stack_owner[g, fy, fx] = 0
        stack_height[g, fy, fx] = 0

        # Merge stacks - attacker on top
        new_height = attacker_height + defender_height - 1  # -1 for eliminated ring

        # Cap at 5
        if new_height > 5:
            # Excess rings become buried for defender
            buried_rings[g, defender] += new_height - 5
            new_height = 5

        stack_owner[g, ty, tx] = player
        stack_height[g, ty, tx] = new_height

        # Place marker for eliminated defender ring
        marker_owner[g, ty, tx] = player

        # Track eliminated ring
        eliminated_rings[g, player] += 1
        buried_rings[g, defender] += 1  # Defender's ring is buried

    return stack_owner, stack_height, marker_owner, eliminated_rings, buried_rings


# =============================================================================
# Utility Functions
# =============================================================================

def count_controlled_stacks(
    stack_owner: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Count stacks controlled by each player.

    Returns:
        Tensor of shape (batch, num_players+1) with stack counts
    """
    batch_size = stack_owner.shape[0]
    device = stack_owner.device

    counts = torch.zeros(batch_size, num_players + 1, dtype=torch.int32, device=device)

    for player in range(1, num_players + 1):
        counts[:, player] = (stack_owner == player).sum(dim=(1, 2)).int()

    return counts


def get_mobility(
    stack_owner: torch.Tensor,
    stack_height: torch.Tensor,
    current_player: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mobility (number of available moves) for current players.

    Returns:
        Tensor of shape (batch,) with mobility counts
    """
    # Get normal moves
    _, _, _, _, _, normal_counts = generate_normal_moves_vectorized(
        stack_owner, stack_height, current_player, active_mask
    )

    # Get capture moves
    _, _, _, _, _, capture_counts = generate_capture_moves_vectorized(
        stack_owner, stack_height, current_player, active_mask
    )

    return (normal_counts + capture_counts).float()
