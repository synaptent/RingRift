"""GPU-accelerated parallel game simulation for CMA-ES and selfplay.

This module enables running multiple RingRift games in parallel using GPU
acceleration for move evaluation. Key use cases:

1. CMA-ES fitness evaluation: Run 10-100+ games per candidate in parallel
2. Selfplay data generation: Generate training data 10x faster
3. Tournament evaluation: Run many tournament games concurrently

Architecture:
- Maintains batch of game states on GPU
- Vectorized move generation and application
- Batch neural network / heuristic evaluation
- Efficient memory management with game recycling

Performance targets:
- 10-100 games/sec on RTX 3090 (vs 1 game/sec CPU)
- 50-500 games/sec on A100 / RTX 5090
"""

from __future__ import annotations

import gc
import os
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .gpu_batch import get_device, clear_gpu_memory
from .shadow_validation import (
    ShadowValidator, create_shadow_validator,
    StateValidator, create_state_validator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MPS Compatibility Helpers
# =============================================================================


def get_int_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate integer dtype for device.

    MPS (Apple Silicon) doesn't support int16 with index_put_(accumulate=True),
    so we use int32 on MPS and int16 elsewhere for memory efficiency.
    """
    if device.type == "mps":
        return torch.int32
    return torch.int16


# =============================================================================
# Vectorized Selection Utilities
# =============================================================================


def select_moves_vectorized(
    moves: "BatchMoves",
    active_mask: torch.Tensor,
    board_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select one move per game using vectorized softmax sampling.

    This replaces the per-game Python loop anti-pattern with fully vectorized
    tensor operations. Key optimizations:
    - No .item() calls (avoids GPU→CPU sync)
    - All scoring done in parallel across all moves
    - Segment-wise softmax using scatter operations
    - Batch multinomial sampling

    Args:
        moves: BatchMoves containing flattened moves for all games
        active_mask: (batch_size,) bool tensor of games to process
        board_size: Board dimension for center-bias calculation
        temperature: Softmax temperature (higher = more random)

    Returns:
        (batch_size,) tensor of selected local move indices per game
    """
    device = moves.device
    batch_size = active_mask.shape[0]

    # Initialize output: -1 for games with no moves
    selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    if moves.total_moves == 0:
        return selected

    # Compute center-bias scores for ALL moves at once (vectorized)
    center = board_size // 2
    max_dist = center * 2.0

    # Use to_y/to_x for destination-based scoring (captures/movements)
    # For placements, from_y/from_x == to_y/to_x, so this works universally
    dist_to_center = (
        (moves.to_y.float() - center).abs() +
        (moves.to_x.float() - center).abs()
    )

    # Score: higher for closer to center + random noise for stochasticity
    noise = torch.rand(moves.total_moves, device=device) * 2.0
    scores = (max_dist - dist_to_center) / temperature + noise

    # Create game assignment for each move
    game_idx = moves.game_idx.long()

    # Segment-wise softmax: compute max per game for numerical stability
    # Use scatter_reduce to get max score per game
    neg_inf = torch.full((batch_size,), float('-inf'), device=device)
    max_per_game = neg_inf.scatter_reduce(0, game_idx, scores, reduce='amax')

    # Subtract max for numerical stability (only for games with moves)
    scores_stable = scores - max_per_game[game_idx]

    # Compute exp(scores)
    exp_scores = torch.exp(scores_stable)

    # Sum exp per game using scatter_add
    sum_per_game = torch.zeros(batch_size, device=device)
    sum_per_game.scatter_add_(0, game_idx, exp_scores)

    # Compute probabilities (exp / sum)
    probs = exp_scores / (sum_per_game[game_idx] + 1e-10)

    # Sample from each game's probability distribution
    # Strategy: cumsum within each game, compare to uniform random
    # This is a vectorized segment-wise multinomial

    # Compute cumsum per game using segment operations
    # Check if already sorted (common case: moves generated game-by-game)
    is_sorted = (game_idx[1:] >= game_idx[:-1]).all() if moves.total_moves > 1 else True

    if is_sorted:
        # Fast path: moves already sorted by game_idx
        sorted_indices = torch.arange(moves.total_moves, device=device)
        sorted_game_idx = game_idx
        sorted_probs = probs
    else:
        # Need to sort
        sorted_indices = torch.argsort(game_idx)
        sorted_game_idx = game_idx[sorted_indices]
        sorted_probs = probs[sorted_indices]

    # Cumsum all probs
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)

    # Subtract cumsum at game boundaries to get per-game cumsum
    # Find where games change
    game_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
    game_starts[1:] = torch.searchsorted(sorted_game_idx, torch.arange(1, batch_size, device=device))

    # Get cumsum value at start of each game
    cumsum_at_start = torch.zeros(batch_size, device=device)
    cumsum_at_start[1:] = cumsum_probs[game_starts[1:] - 1]

    # Per-game cumsum = global cumsum - cumsum at game start
    per_game_cumsum = cumsum_probs - cumsum_at_start[sorted_game_idx]

    # Generate one random value per game
    rand_vals = torch.rand(batch_size, device=device)

    # For each move, check if this is the first move where cumsum > rand
    # Mask: rand < cumsum (this move or later could be selected)
    exceeds_rand = per_game_cumsum > rand_vals[sorted_game_idx]

    # Find first exceeding index per game
    # Set non-exceeding to large value, then take min per game
    large_val = moves.total_moves + 1
    indices_or_large = torch.where(
        exceeds_rand,
        torch.arange(moves.total_moves, device=device, dtype=torch.float32),
        torch.full((moves.total_moves,), float(large_val), device=device)
    )

    # Get min index per game (first exceeding)
    # Use float32 for scatter_reduce_ compatibility on MPS backend
    first_exceed_f = torch.full((batch_size,), float(large_val), dtype=torch.float32, device=device)
    first_exceed_f.scatter_reduce_(0, sorted_game_idx, indices_or_large, reduce='amin')
    first_exceed = first_exceed_f.long()

    # Convert global sorted index to local index within game
    # local_idx = global_sorted_idx - game_start
    has_moves = moves.moves_per_game > 0
    valid_selection = has_moves & active_mask & (first_exceed < large_val)

    # Map sorted indices back to original indices
    original_indices = sorted_indices[first_exceed.clamp(0, moves.total_moves - 1)]

    # Compute local index: original_idx - move_offset for that game
    local_idx = original_indices - moves.move_offsets

    selected[valid_selection] = local_idx[valid_selection]

    # Clamp to valid range
    selected = torch.clamp(selected, min=0)

    return selected


def select_moves_heuristic(
    moves: "BatchMoves",
    state: "BatchGameState",
    active_mask: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select one move per game using heuristic scoring.

    This is an enhanced version of select_moves_vectorized that uses
    game state features to score moves, providing better move quality
    for training data generation.

    Features used for scoring:
    - Center distance (closer = better)
    - Capture value (for captures: target stack height)
    - Adjacency bonus (moves near own stacks)
    - Line potential (moves that extend lines)

    Uses symmetric scoring to avoid P1/P2 bias.

    Args:
        moves: BatchMoves containing flattened moves for all games
        state: BatchGameState for feature extraction
        active_mask: (batch_size,) bool tensor of games to process
        weights: Optional weight dict for feature importance
        temperature: Softmax temperature (higher = more random)

    Returns:
        (batch_size,) tensor of selected local move indices per game
    """
    device = moves.device
    batch_size = active_mask.shape[0]
    board_size = state.board_size

    # Initialize output: -1 for games with no moves
    selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    if moves.total_moves == 0:
        return selected

    # Default weights for heuristic features
    if weights is None:
        weights = {
            "center": 3.0,
            "capture_value": 5.0,
            "adjacency": 2.0,
            "line_potential": 1.5,
            "noise": 1.0,
        }

    center = board_size // 2
    max_dist = center * 2.0

    # === Feature 1: Center distance ===
    dist_to_center = (
        (moves.to_y.float() - center).abs() +
        (moves.to_x.float() - center).abs()
    )
    center_score = (max_dist - dist_to_center) * weights.get("center", 3.0)

    # === Feature 2: Capture value (for capture moves) ===
    # Look up target stack height at destination
    game_idx = moves.game_idx.long()
    to_y = moves.to_y.long()
    to_x = moves.to_x.long()

    target_heights = state.stack_height[game_idx, to_y, to_x].float()
    # Only count for captures (move_type == MoveType.CAPTURE)
    is_capture = moves.move_type == 3  # MoveType.CAPTURE = 3
    capture_score = torch.where(
        is_capture,
        target_heights * weights.get("capture_value", 5.0),
        torch.zeros_like(target_heights)
    )

    # === Feature 3: Adjacency to own stacks ===
    # Count adjacent friendly stacks at destination
    current_players = state.current_player[game_idx]

    # Pad stack_owner for boundary-safe adjacency check
    so_padded = torch.nn.functional.pad(
        state.stack_owner.float(), (1, 1, 1, 1), value=0
    )

    # Check 4 neighbors (up, down, left, right) for same owner
    # Offset by 1 due to padding
    adj_up = so_padded[game_idx, to_y, to_x + 1] == current_players.float()
    adj_down = so_padded[game_idx, to_y + 2, to_x + 1] == current_players.float()
    adj_left = so_padded[game_idx, to_y + 1, to_x] == current_players.float()
    adj_right = so_padded[game_idx, to_y + 1, to_x + 2] == current_players.float()

    adjacent_count = adj_up.float() + adj_down.float() + adj_left.float() + adj_right.float()
    adjacency_score = adjacent_count * weights.get("adjacency", 2.0)

    # === Feature 4: Line potential (simplified) ===
    # Check for 2+ in a row potential at destination
    # Check horizontal and vertical lines
    # Index by game_idx to get the board for each move's game
    own_stacks = (state.stack_owner[game_idx] == current_players.view(-1, 1, 1))
    own_stacks_padded = torch.nn.functional.pad(own_stacks.float(), (1, 1, 1, 1), value=0)

    # own_stacks_padded has shape (num_moves, board+2, board+2) - one board per move
    # Use move indices to index the first dimension
    move_idx = torch.arange(moves.total_moves, device=device)

    # Horizontal line: check left and right neighbors
    h_left = own_stacks_padded[move_idx, to_y + 1, to_x]
    h_right = own_stacks_padded[move_idx, to_y + 1, to_x + 2]
    h_line = h_left + h_right

    # Vertical line: check up and down neighbors
    v_up = own_stacks_padded[move_idx, to_y, to_x + 1]
    v_down = own_stacks_padded[move_idx, to_y + 2, to_x + 1]
    v_line = v_up + v_down

    line_score = (h_line + v_line) * weights.get("line_potential", 1.5)

    # === Combine all scores ===
    noise = torch.rand(moves.total_moves, device=device) * weights.get("noise", 1.0)
    scores = center_score + capture_score + adjacency_score + line_score + noise

    # Use the same segment-wise softmax sampling as select_moves_vectorized
    neg_inf = torch.full((batch_size,), float('-inf'), device=device)
    max_per_game = neg_inf.scatter_reduce(0, game_idx, scores, reduce='amax')

    scores_stable = scores - max_per_game[game_idx]
    exp_scores = torch.exp(scores_stable / temperature)

    sum_per_game = torch.zeros(batch_size, device=device)
    sum_per_game.scatter_add_(0, game_idx, exp_scores)

    probs = exp_scores / (sum_per_game[game_idx] + 1e-10)

    # Segment-wise multinomial sampling
    is_sorted = (game_idx[1:] >= game_idx[:-1]).all() if moves.total_moves > 1 else True

    if is_sorted:
        sorted_indices = torch.arange(moves.total_moves, device=device)
        sorted_game_idx = game_idx
        sorted_probs = probs
    else:
        sorted_indices = torch.argsort(game_idx)
        sorted_game_idx = game_idx[sorted_indices]
        sorted_probs = probs[sorted_indices]

    cumsum_probs = torch.cumsum(sorted_probs, dim=0)

    game_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
    game_starts[1:] = torch.searchsorted(sorted_game_idx, torch.arange(1, batch_size, device=device))

    cumsum_at_start = torch.zeros(batch_size, device=device)
    cumsum_at_start[1:] = cumsum_probs[game_starts[1:] - 1]

    per_game_cumsum = cumsum_probs - cumsum_at_start[sorted_game_idx]

    rand_vals = torch.rand(batch_size, device=device)
    exceeds_rand = per_game_cumsum > rand_vals[sorted_game_idx]

    large_val = moves.total_moves + 1
    indices_or_large = torch.where(
        exceeds_rand,
        torch.arange(moves.total_moves, device=device, dtype=torch.float32),
        torch.full((moves.total_moves,), float(large_val), device=device)
    )

    first_exceed_f = torch.full((batch_size,), float(large_val), dtype=torch.float32, device=device)
    first_exceed_f.scatter_reduce_(0, sorted_game_idx, indices_or_large, reduce='amin')
    first_exceed = first_exceed_f.long()

    has_moves = moves.moves_per_game > 0
    valid_selection = has_moves & active_mask & (first_exceed < large_val)

    original_indices = sorted_indices[first_exceed.clamp(0, moves.total_moves - 1)]
    local_idx = original_indices - moves.move_offsets

    selected[valid_selection] = local_idx[valid_selection]
    selected = torch.clamp(selected, min=0)

    return selected


def apply_capture_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply capture moves in a vectorized manner.

    This applies selected capture moves for multiple games simultaneously,
    minimizing Python loops and .item() calls.

    Note: Some operations (like path marker flipping) still require iteration
    due to variable-length paths. This is a known limitation documented in
    GPU_PIPELINE_ROADMAP.md Section 2.2 (Irregular Data Access Patterns).

    Args:
        state: BatchGameState to modify
        selected_local_idx: (batch_size,) local move indices
        moves: BatchMoves containing capture moves
        active_mask: (batch_size,) bool tensor of games with captures to apply
    """
    device = state.device
    batch_size = state.batch_size

    # Identify games that have moves to apply
    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    # Compute global indices for selected moves
    global_idx = moves.move_offsets + selected_local_idx
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    # Get move data for all selected moves at once
    selected_from_y = moves.from_y[global_idx]
    selected_from_x = moves.from_x[global_idx]
    selected_to_y = moves.to_y[global_idx]
    selected_to_x = moves.to_x[global_idx]

    # Get current players for active games
    current_players = state.current_player

    # Apply moves game by game (some operations require iteration due to variable paths)
    # This is the minimal iteration - just for path processing
    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = int(state.move_count[g].item())
        # Clear must-move constraint after the first movement/capture action
        # following a placement (RR-CANON-R090).
        state.must_move_from_y[g] = -1
        state.must_move_from_x[g] = -1

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.CAPTURE
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
        state.move_count[g] += 1

        # Get attacker stack info at origin.
        attacker_height = int(state.stack_height[g, from_y, from_x].item())
        attacker_cap_height = int(state.cap_height[g, from_y, from_x].item())

        # Capture move representation:
        # - (from -> landing) is stored in BatchMoves
        # - The target stack is implicit as the first stack along the ray
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        target_y = None
        target_x = None
        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            if state.stack_owner[g, check_y, check_x].item() != 0:
                target_y = check_y
                target_x = check_x
                break

        if target_y is None or target_x is None:
            # Defensive fallback: treat this as a movement to landing.
            state.stack_height[g, to_y, to_x] = attacker_height
            state.stack_owner[g, to_y, to_x] = player
            state.cap_height[g, to_y, to_x] = min(attacker_cap_height, attacker_height)
            state.stack_height[g, from_y, from_x] = 0
            state.stack_owner[g, from_y, from_x] = 0
            state.cap_height[g, from_y, from_x] = 0
            state.marker_owner[g, from_y, from_x] = player
            continue

        # Process markers along the full path (RR-CANON-R102 delegates to R092),
        # excluding the implicit target stack cell.
        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            if check_y == target_y and check_x == target_x:
                continue

            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner == 0:
                continue
            if marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player
                continue

            # Own marker on intermediate cell: collapse to territory.
            state.marker_owner[g, check_y, check_x] = 0
            if not state.is_collapsed[g, check_y, check_x].item():
                state.is_collapsed[g, check_y, check_x] = True
                state.territory_owner[g, check_y, check_x] = player
                state.territory_count[g, player] += 1

        # Landing marker interaction (RR-CANON-R102): remove any marker on landing
        # (do not collapse), then eliminate the top ring of the moving stack's cap.
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 1 if dest_marker != 0 else 0
        if landing_ring_cost:
            state.marker_owner[g, to_y, to_x] = 0
            state.eliminated_rings[g, player] += 1
            state.rings_caused_eliminated[g, player] += 1

        # Pop the top ring from the implicit target and append it to the bottom
        # of the attacking stack (RR-CANON-R102). We do not store full ring
        # sequences on GPU; we approximate by updating stack/cap metadata and
        # tracking captured rings via buried_rings.
        target_owner = int(state.stack_owner[g, target_y, target_x].item())
        target_height = int(state.stack_height[g, target_y, target_x].item())
        target_cap_height = int(state.cap_height[g, target_y, target_x].item())

        # Target cell should not contain a marker; clear defensively.
        state.marker_owner[g, target_y, target_x] = 0

        new_target_height = max(0, target_height - 1)
        state.stack_height[g, target_y, target_x] = new_target_height
        if new_target_height <= 0:
            state.stack_owner[g, target_y, target_x] = 0
            state.cap_height[g, target_y, target_x] = 0
        else:
            new_target_cap = target_cap_height - 1
            if new_target_cap <= 0:
                new_target_cap = 1
            if new_target_cap > new_target_height:
                new_target_cap = new_target_height
            state.cap_height[g, target_y, target_x] = new_target_cap

        # Track captured ring as "buried" for the ring's owner (when capturing an opponent).
        if target_owner != 0 and target_owner != player:
            state.buried_rings[g, target_owner] += 1

        # Move attacker to landing and apply net height change:
        # +1 captured ring (to bottom) - landing marker elimination cost.
        new_height = attacker_height + 1 - landing_ring_cost
        state.stack_height[g, to_y, to_x] = new_height
        state.stack_owner[g, to_y, to_x] = player

        new_cap = attacker_cap_height - landing_ring_cost
        if new_cap <= 0:
            new_cap = 1
        if new_cap > new_height:
            new_cap = new_height
        state.cap_height[g, to_y, to_x] = new_cap

        # Clear origin stack and leave a departure marker (RR-CANON-R092).
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0
        state.marker_owner[g, from_y, from_x] = player


def apply_movement_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply movement moves in a vectorized manner.

    Similar to capture moves but without defender elimination.
    Still requires iteration for path marker processing.
    """
    device = state.device
    batch_size = state.batch_size

    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    global_idx = moves.move_offsets + selected_local_idx
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    selected_from_y = moves.from_y[global_idx]
    selected_from_x = moves.from_x[global_idx]
    selected_to_y = moves.to_y[global_idx]
    selected_to_x = moves.to_x[global_idx]

    current_players = state.current_player
    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = int(state.move_count[g].item())
        state.must_move_from_y[g] = -1
        state.must_move_from_x[g] = -1

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.MOVEMENT
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
        state.move_count[g] += 1

        moving_height = state.stack_height[g, from_y, from_x].item()
        moving_cap_height = state.cap_height[g, from_y, from_x].item()

        # Process markers along path (RR-CANON-R092):
        # - Flip opponent markers along the path
        # - Collapse own markers on intermediate cells to territory
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner == 0:
                continue
            if marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player
                continue

            # Own marker on intermediate cell: collapse to territory.
            state.marker_owner[g, check_y, check_x] = 0
            if not state.is_collapsed[g, check_y, check_x].item():
                state.is_collapsed[g, check_y, check_x] = True
                state.territory_owner[g, check_y, check_x] = player
                state.territory_count[g, player] += 1

        # Handle landing on ANY marker (own or opponent):
        # remove the marker (do not collapse), then eliminate the top cap ring.
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 1 if dest_marker != 0 else 0
        if landing_ring_cost:
            state.marker_owner[g, to_y, to_x] = 0

        # Movement cannot land on stacks; destination is guaranteed empty by move generation.
        new_height = moving_height - landing_ring_cost

        # Track eliminated ring from landing cost
        if landing_ring_cost > 0:
            current_elim = state.eliminated_rings[g, player].item()
            state.eliminated_rings[g, player] = current_elim + landing_ring_cost
            # Player eliminates their own ring for landing cost (self-elimination counts for victory)
            state.rings_caused_eliminated[g, player] += landing_ring_cost

        # Update destination
        final_height = max(0, new_height)
        if final_height <= 0:
            # Landing cost can eliminate the final ring; the destination remains
            # empty (but may still become collapsed if the move landed on a marker).
            state.stack_height[g, to_y, to_x] = 0
            state.stack_owner[g, to_y, to_x] = 0
            state.cap_height[g, to_y, to_x] = 0
        else:
            state.stack_height[g, to_y, to_x] = final_height
            state.stack_owner[g, to_y, to_x] = player
            # Best-effort cap update (GPU does not track ring colors beyond capHeight metadata).
            new_cap = moving_cap_height - landing_ring_cost
            if new_cap <= 0:
                new_cap = 1
            if new_cap > final_height:
                new_cap = final_height
            state.cap_height[g, to_y, to_x] = new_cap

        # Clear origin
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0
        # Leave a marker on the departure space (RR-CANON-R092).
        state.marker_owner[g, from_y, from_x] = player


def apply_recovery_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply recovery slide moves in a fully vectorized manner.

    Optimized 2025-12-13: Eliminated Python loops and .item() calls.
    """
    device = state.device

    has_selection = (selected_local_idx >= 0) & active_mask & (moves.moves_per_game > 0)

    if not has_selection.any():
        return

    game_indices = torch.where(has_selection)[0]
    n_games = game_indices.shape[0]

    global_idx = moves.move_offsets[game_indices] + selected_local_idx[game_indices]
    global_idx = torch.clamp(global_idx, 0, max(0, moves.total_moves - 1))

    from_y = moves.from_y[global_idx].long()
    from_x = moves.from_x[global_idx].long()
    to_y = moves.to_y[global_idx].long()
    to_x = moves.to_x[global_idx].long()

    players = state.current_player[game_indices]

    # Record in history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = MoveType.RECOVERY_SLIDE
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    state.move_count[game_indices] += 1

    # Clear source marker
    state.marker_owner[game_indices, from_y, from_x] = 0

    # Check destination for stack-strike vs normal recovery
    dest_height = state.stack_height[game_indices, to_y, to_x]
    dest_owner = state.stack_owner[game_indices, to_y, to_x]
    is_stack_strike = (dest_height > 0) & (dest_owner > 0)

    # Handle stack-strike recovery (RR-CANON-R112(b2))
    if is_stack_strike.any():
        ss_games = game_indices[is_stack_strike]
        ss_to_y = to_y[is_stack_strike]
        ss_to_x = to_x[is_stack_strike]
        ss_players = players[is_stack_strike]
        ss_dest_owner = dest_owner[is_stack_strike]
        ss_dest_height = dest_height[is_stack_strike]
        ss_old_cap = state.cap_height[ss_games, ss_to_y, ss_to_x]

        # Update eliminated rings tracking
        ones = torch.ones(ss_games.shape[0], dtype=state.eliminated_rings.dtype, device=device)
        state.eliminated_rings.index_put_(
            (ss_games, ss_dest_owner.long()),
            ones,
            accumulate=True
        )
        state.rings_caused_eliminated.index_put_(
            (ss_games, ss_players.long()),
            ones,
            accumulate=True
        )

        # Update stack
        new_height = torch.clamp(ss_dest_height - 1, min=0)
        new_cap = torch.clamp(ss_old_cap - 1, min=1)
        new_cap = torch.minimum(new_cap, new_height)
        new_cap = torch.where(new_height > 0, new_cap, torch.zeros_like(new_cap))

        state.stack_height[ss_games, ss_to_y, ss_to_x] = new_height.to(state.stack_height.dtype)
        is_cleared = new_height == 0
        state.stack_owner[ss_games[is_cleared], ss_to_y[is_cleared], ss_to_x[is_cleared]] = 0
        state.cap_height[ss_games, ss_to_y, ss_to_x] = new_cap.to(state.cap_height.dtype)

    # Handle normal recovery slide
    is_normal = ~is_stack_strike
    if is_normal.any():
        nr_games = game_indices[is_normal]
        nr_to_y = to_y[is_normal]
        nr_to_x = to_x[is_normal]
        nr_players = players[is_normal]
        state.marker_owner[nr_games, nr_to_y, nr_to_x] = nr_players.to(state.marker_owner.dtype)

    # Deduct buried ring cost - only if player has buried rings
    current_buried = state.buried_rings[game_indices, players.long()]
    has_buried = current_buried > 0
    if has_buried.any():
        hb_games = game_indices[has_buried]
        hb_players = players[has_buried].long()
        neg_ones = torch.full((hb_games.shape[0],), -1, dtype=state.buried_rings.dtype, device=device)
        pos_ones = torch.ones(hb_games.shape[0], dtype=state.eliminated_rings.dtype, device=device)

        state.buried_rings.index_put_((hb_games, hb_players), neg_ones, accumulate=True)
        # Self-elimination for buried ring extraction (RR-CANON-R114/R115)
        state.eliminated_rings.index_put_((hb_games, hb_players), pos_ones, accumulate=True)
        state.rings_caused_eliminated.index_put_((hb_games, hb_players), pos_ones, accumulate=True)


def apply_no_action_moves_batch(
    state: "BatchGameState",
    mask: torch.Tensor,
) -> None:
    """Record a NO_ACTION move for each masked active game.

    This is used to avoid silent phase progression when a player has no
    legal action in an interactive phase (RR-CANON-R075).

    Optimized 2025-12-13: Eliminated Python loops and .item() calls.
    """
    active_mask = mask & state.get_active_mask()
    if not active_mask.any():
        return

    game_indices = torch.where(active_mask)[0]
    move_idx = state.move_count[game_indices]
    players = state.current_player[game_indices]

    # Record in history for games with space
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = MoveType.NO_ACTION
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = -1
        state.move_history[hist_games, hist_move_idx, 3] = -1
        state.move_history[hist_games, hist_move_idx, 4] = -1
        state.move_history[hist_games, hist_move_idx, 5] = -1

    state.move_count[game_indices] += 1


# =============================================================================
# Constants
# =============================================================================

class GameStatus(IntEnum):
    """Game status codes for batch tracking."""
    ACTIVE = 0
    COMPLETED = 1
    DRAW = 2
    MAX_MOVES = 3


class MoveType(IntEnum):
    """Move type codes for batch move representation."""
    PLACEMENT = 0
    MOVEMENT = 1
    CAPTURE = 2
    LINE_FORMATION = 3
    TERRITORY_CLAIM = 4
    SKIP = 5
    NO_ACTION = 6  # For phases with no available action
    RECOVERY_SLIDE = 7  # Recovery move for players without turn material


class GamePhase(IntEnum):
    """Game phase codes for turn FSM.

    Per RR-CANON, each turn flows through phases:
    RING_PLACEMENT -> MOVEMENT -> LINE_PROCESSING -> TERRITORY_PROCESSING -> END_TURN
    """
    RING_PLACEMENT = 0      # Place ring from hand (if available)
    MOVEMENT = 1            # Move stack (movement/capture/recovery)
    LINE_PROCESSING = 2     # Check and convert lines to territory
    TERRITORY_PROCESSING = 3  # Calculate enclosed territory
    END_TURN = 4            # Advance to next player


# =============================================================================
# Batch Game State
# =============================================================================


@dataclass
class BatchGameState:
    """Batched game state representation for parallel simulation.

    All tensors have shape (batch_size, ...) and are stored on GPU.
    """

    # Board state: (batch_size, board_size, board_size)
    stack_owner: torch.Tensor      # 0=empty, 1-4=player
    stack_height: torch.Tensor     # 0-5 (total rings in stack)
    cap_height: torch.Tensor       # 0-5 (consecutive top rings of owner's color per RR-CANON-R101)
    marker_owner: torch.Tensor     # 0=none, 1-4=player
    territory_owner: torch.Tensor  # 0=neutral, 1-4=player
    is_collapsed: torch.Tensor     # bool

    # Player state: (batch_size, num_players)
    rings_in_hand: torch.Tensor
    territory_count: torch.Tensor
    is_eliminated: torch.Tensor    # bool
    eliminated_rings: torch.Tensor # Rings LOST BY this player (not for victory check)
    buried_rings: torch.Tensor     # Rings buried in stacks (captured but not removed)
    rings_caused_eliminated: torch.Tensor  # Rings CAUSED TO BE ELIMINATED BY this player (RR-CANON-R060)

    # Game metadata: (batch_size,)
    current_player: torch.Tensor   # 1-4
    current_phase: torch.Tensor    # GamePhase enum
    move_count: torch.Tensor
    game_status: torch.Tensor      # GameStatus enum
    winner: torch.Tensor           # 0=none, 1-4=player
    swap_offered: torch.Tensor     # bool: whether swap_sides (pie rule) was offered to P2

    # Per-turn movement constraint (RR-CANON-R090):
    # When a placement occurs, the subsequent movement/capture must originate
    # from that exact stack. -1 indicates "no constraint".
    must_move_from_y: torch.Tensor  # int16 (batch_size,)
    must_move_from_x: torch.Tensor  # int16 (batch_size,)

    # LPS tracking (RR-CANON-R172): tensor mirrors of GameState fields.
    # We track a full-round cycle over all non-permanently-eliminated players.
    lps_round_index: torch.Tensor  # int32 (batch_size,)
    lps_current_round_first_player: torch.Tensor  # int8 (batch_size,) 0=unset
    lps_current_round_seen_mask: torch.Tensor  # bool (batch_size, num_players+1)
    lps_current_round_real_action_mask: torch.Tensor  # bool (batch_size, num_players+1)
    lps_exclusive_player_for_completed_round: torch.Tensor  # int8 (batch_size,) 0=none
    lps_consecutive_exclusive_rounds: torch.Tensor  # int16 (batch_size,)
    lps_consecutive_exclusive_player: torch.Tensor  # int8 (batch_size,) 0=none

    # Move history: (batch_size, max_moves, 6) - [move_type, player, from_y, from_x, to_y, to_x]
    # -1 indicates unused slot
    move_history: torch.Tensor
    max_history_moves: int

    # LPS configuration: consecutive exclusive rounds required for victory.
    lps_rounds_required: int

    # Configuration
    device: torch.device
    batch_size: int
    board_size: int
    num_players: int

    @classmethod
    def create_batch(
        cls,
        batch_size: int,
        board_size: int = 8,
        num_players: int = 2,
        device: Optional[torch.device] = None,
        max_history_moves: int = 500,
        lps_rounds_required: int = 3,
        rings_per_player: Optional[int] = None,
        board_type: Optional[str] = None,
    ) -> "BatchGameState":
        """Create a batch of initialized game states.

        Args:
            batch_size: Number of parallel games
            board_size: Board dimension (8, 19) or hex embedding (25)
            num_players: Number of players (2-4)
            device: GPU device (auto-detected if None)
            max_history_moves: Maximum moves to track in history
            rings_per_player: Starting rings per player (None = board default)
            board_type: Board type string ("square8", "square19", "hexagonal")
                        If "hexagonal", marks out-of-bounds cells as collapsed.

        Returns:
            Initialized BatchGameState with all games ready to start
        """
        if device is None:
            device = get_device()

        # Get appropriate int dtype for device (int32 on MPS, int16 elsewhere)
        int_dtype = get_int_dtype(device)

        # Initialize board tensors
        shape_board = (batch_size, board_size, board_size)
        shape_players = (batch_size, num_players + 1)  # +1 for 1-indexed players

        # Starting rings per player based on board size (per RR-CANON-R020)
        # square8: 18, square19: 72, hexagonal: 96
        # Allow override for ablation studies
        if rings_per_player is not None:
            starting_rings = rings_per_player
        else:
            # Board size -> default rings mapping.
            #
            # NOTE: GPU hex kernels use a 25×25 embedding (radius-12 -> 2r+1),
            # so the canonical hex supply must map to board_size=25 as well.
            starting_rings = {8: 18, 19: 72, 13: 96, 25: 96}.get(board_size, 18)

        rings = torch.zeros(shape_players, dtype=int_dtype, device=device)
        rings[:, 1:num_players+1] = starting_rings

        # Move history: (batch_size, max_moves, 6) - [move_type, player, from_y, from_x, to_y, to_x]
        # Initialize with -1 to indicate unused slots
        move_history = torch.full(
            (batch_size, max_history_moves, 6),
            -1,
            dtype=int_dtype,
            device=device,
        )

        batch = cls(
            stack_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            stack_height=torch.zeros(shape_board, dtype=torch.int8, device=device),
            cap_height=torch.zeros(shape_board, dtype=torch.int8, device=device),
            marker_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            territory_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            is_collapsed=torch.zeros(shape_board, dtype=torch.bool, device=device),
            rings_in_hand=rings,
            territory_count=torch.zeros(shape_players, dtype=int_dtype, device=device),
            is_eliminated=torch.zeros(shape_players, dtype=torch.bool, device=device),
            eliminated_rings=torch.zeros(shape_players, dtype=int_dtype, device=device),
            buried_rings=torch.zeros(shape_players, dtype=int_dtype, device=device),
            rings_caused_eliminated=torch.zeros(shape_players, dtype=int_dtype, device=device),
            current_player=torch.ones(batch_size, dtype=torch.int8, device=device),
            current_phase=torch.zeros(batch_size, dtype=torch.int8, device=device),  # RING_PLACEMENT
            move_count=torch.zeros(batch_size, dtype=torch.int32, device=device),
            game_status=torch.zeros(batch_size, dtype=torch.int8, device=device),
            winner=torch.zeros(batch_size, dtype=torch.int8, device=device),
            swap_offered=torch.zeros(batch_size, dtype=torch.bool, device=device),
            must_move_from_y=torch.full(
                (batch_size,), -1, dtype=int_dtype, device=device
            ),
            must_move_from_x=torch.full(
                (batch_size,), -1, dtype=int_dtype, device=device
            ),
            lps_round_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            lps_current_round_first_player=torch.zeros(
                batch_size, dtype=torch.int8, device=device
            ),
            lps_current_round_seen_mask=torch.zeros(
                shape_players, dtype=torch.bool, device=device
            ),
            lps_current_round_real_action_mask=torch.zeros(
                shape_players, dtype=torch.bool, device=device
            ),
            lps_exclusive_player_for_completed_round=torch.zeros(
                batch_size, dtype=torch.int8, device=device
            ),
            lps_consecutive_exclusive_rounds=torch.zeros(
                batch_size, dtype=int_dtype, device=device
            ),
            lps_consecutive_exclusive_player=torch.zeros(
                batch_size, dtype=torch.int8, device=device
            ),
            move_history=move_history,
            max_history_moves=max_history_moves,
            lps_rounds_required=int(lps_rounds_required),
            device=device,
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
        )

        # Mark hex out-of-bounds cells as collapsed (GPU parity fix)
        # Hex boards are embedded in a square bounding box (e.g., 25x25 for radius-12).
        # Only cells satisfying max(|q|, |r|, |q+r|) <= radius are valid.
        # All other cells must be marked as collapsed to prevent invalid placements.
        if board_type and board_type.lower() in ("hexagonal", "hex"):
            center = board_size // 2
            radius = center  # Hex radius = half the bounding box size (e.g., 12 for 25x25)

            # Create hex validity mask
            for row in range(board_size):
                for col in range(board_size):
                    # Convert to axial coordinates (q, r) centered at origin
                    q = col - center
                    r = row - center
                    # Check if outside hex radius using axial distance formula
                    if max(abs(q), abs(r), abs(q + r)) > radius:
                        # Mark as collapsed (out of bounds) for all games in batch
                        batch.is_collapsed[:, row, col] = True

            # Count valid cells for logging
            valid_cells = (~batch.is_collapsed[0]).sum().item()
            logger.info(
                f"Hex board initialized: {board_size}x{board_size} embedding, "
                f"radius={radius}, valid_cells={valid_cells} (expected ~{3*radius**2 + 3*radius + 1})"
            )

        return batch

    @classmethod
    def from_single_game(
        cls,
        game_state: "GameState",
        device: Optional[torch.device] = None,
    ) -> "BatchGameState":
        """Convert a single CPU GameState to a BatchGameState with batch_size=1.

        This method enables direct comparison between CPU and GPU implementations
        by converting the canonical CPU representation to GPU tensor format.

        Args:
            game_state: A single GameState from the CPU implementation
            device: GPU device (auto-detected if None)

        Returns:
            BatchGameState with batch_size=1 containing the converted state
        """
        from app.models import GameState, BoardType, GamePhase as CPUGamePhase

        if device is None:
            device = get_device()

        # Determine board size from board type (Pydantic converts to snake_case)
        board_type = game_state.board_type
        board_size = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEXAGONAL: 13,  # Hex uses 13 for radius calculation
        }.get(board_type, 8)

        num_players = len(game_state.players)
        batch_size = 1

        # Create empty batch state
        batch = cls.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
        )

        # Copy board state (Pydantic converts to snake_case for access)
        # Note: Hex boards use 3D coordinates (x,y,z), square boards use 2D (x,y)
        is_hex = board_type == BoardType.HEXAGONAL

        # mustMoveFromStackKey (TS) -> must_move_from_{y,x} (GPU)
        must_key = getattr(game_state, "must_move_from_stack_key", None)
        if isinstance(must_key, str) and must_key:
            try:
                parts = [int(p) for p in must_key.split(",")]
                if len(parts) >= 2 and not is_hex:
                    batch.must_move_from_x[0] = parts[0]
                    batch.must_move_from_y[0] = parts[1]
            except Exception:
                pass

        for key, stack in game_state.board.stacks.items():
            coords = list(map(int, key.split(",")))
            if is_hex:
                # Hex: convert cube coords to array index (skip for now - GPU doesn't support hex)
                continue
            x, y = coords[0], coords[1]
            if 0 <= x < board_size and 0 <= y < board_size:
                batch.stack_owner[0, y, x] = stack.controlling_player
                batch.stack_height[0, y, x] = len(stack.rings)
                batch.cap_height[0, y, x] = stack.cap_height

        for key, marker in game_state.board.markers.items():
            coords = list(map(int, key.split(",")))
            if is_hex:
                continue  # GPU doesn't support hex yet
            x, y = coords[0], coords[1]
            if 0 <= x < board_size and 0 <= y < board_size:
                # Handle both int (legacy) and MarkerInfo (current) marker values
                player = marker.player if hasattr(marker, 'player') else marker
                batch.marker_owner[0, y, x] = player

        for key, player in game_state.board.collapsed_spaces.items():
            coords = list(map(int, key.split(",")))
            if is_hex:
                continue  # GPU doesn't support hex yet
            x, y = coords[0], coords[1]
            if 0 <= x < board_size and 0 <= y < board_size:
                batch.territory_owner[0, y, x] = player
                batch.is_collapsed[0, y, x] = True

        # Copy player state
        for i, player in enumerate(game_state.players):
            player_num = i + 1
            batch.rings_in_hand[0, player_num] = player.rings_in_hand
            batch.eliminated_rings[0, player_num] = player.eliminated_rings
            batch.territory_count[0, player_num] = player.territory_spaces

        # Copy game metadata
        batch.current_player[0] = game_state.current_player

        # Map CPU GamePhase to GPU GamePhase (IntEnum)
        # Note: GPU uses simplified phase model, map all to closest equivalent
        phase_map = {
            CPUGamePhase.RING_PLACEMENT: 0,
            CPUGamePhase.MOVEMENT: 1,
            CPUGamePhase.CAPTURE: 1,  # Map to movement
            CPUGamePhase.CHAIN_CAPTURE: 1,  # Map to movement
            CPUGamePhase.LINE_PROCESSING: 2,
            CPUGamePhase.TERRITORY_PROCESSING: 3,
            CPUGamePhase.FORCED_ELIMINATION: 3,  # Map to territory processing
            CPUGamePhase.GAME_OVER: 4,
        }
        batch.current_phase[0] = phase_map.get(game_state.current_phase, 0)

        return batch

    def to_game_state(self, game_idx: int) -> "GameState":
        """Convert a single game from this batch back to a CPU GameState.

        This is used for shadow validation - comparing GPU-generated moves
        against the canonical CPU rules engine.

        Args:
            game_idx: Index of game in batch to extract

        Returns:
            CPU GameState that can be passed to GameEngine for validation
        """
        from datetime import datetime
        from app.models import (
            GameState, BoardState, BoardType, Player, TimeControl,
            RingStack, Position, MarkerInfo, GamePhase as CPUGamePhase,
            GameStatus as CPUGameStatus
        )

        # Map GPU board_size back to BoardType
        # Note: hex uses 25x25 embedding (radius 12 -> 2*12+1 = 25)
        board_type_map = {
            8: BoardType.SQUARE8,
            19: BoardType.SQUARE19,
            13: BoardType.HEXAGONAL,
            25: BoardType.HEXAGONAL,  # Hex 25x25 embedding
        }
        board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)

        # Hex boards: convert GPU grid coords (row, col in 25x25) to axial (x, y)
        # CPU hex uses size=13 (radius), GPU uses size=25 (embedding)
        is_hex = board_type == BoardType.HEXAGONAL
        hex_center = self.board_size // 2 if is_hex else 0  # 12 for 25x25
        cpu_board_size = 13 if is_hex else self.board_size

        def grid_to_cpu_coords(row: int, col: int):
            """Convert GPU grid coords to CPU format."""
            if is_hex:
                # GPU grid -> axial: (x, y) = (col - center, row - center)
                ax = col - hex_center
                ay = row - hex_center
                az = -ax - ay  # Cube constraint
                return ax, ay, az
            return col, row, None

        # Build stacks dict from GPU tensors
        stacks = {}
        for row in range(self.board_size):
            for col in range(self.board_size):
                owner = self.stack_owner[game_idx, row, col].item()
                height = self.stack_height[game_idx, row, col].item()
                if owner > 0 and height > 0:
                    ax, ay, az = grid_to_cpu_coords(row, col)
                    if is_hex:
                        key = f"{ax},{ay},{az}"
                        pos = Position(x=ax, y=ay, z=az)
                    else:
                        key = f"{ax},{ay}"
                        pos = Position(x=ax, y=ay)
                    # Reconstruct rings list (simplified - all same owner)
                    rings = [owner] * height
                    cap = self.cap_height[game_idx, row, col].item()
                    stacks[key] = RingStack(
                        position=pos,
                        rings=rings,
                        stackHeight=height,
                        capHeight=cap,
                        controllingPlayer=owner,
                    )

        # Build markers dict
        markers = {}
        for row in range(self.board_size):
            for col in range(self.board_size):
                marker_player = self.marker_owner[game_idx, row, col].item()
                if marker_player > 0:
                    ax, ay, az = grid_to_cpu_coords(row, col)
                    if is_hex:
                        key = f"{ax},{ay},{az}"
                        pos = Position(x=ax, y=ay, z=az)
                    else:
                        key = f"{ax},{ay}"
                        pos = Position(x=ax, y=ay)
                    markers[key] = MarkerInfo(
                        player=marker_player,
                        position=pos,
                        type="regular",
                    )

        # Build collapsed_spaces dict
        # For hex, skip out-of-bounds collapsed cells (they are just embedding padding)
        collapsed_spaces = {}
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_collapsed[game_idx, row, col].item():
                    ax, ay, az = grid_to_cpu_coords(row, col)
                    # For hex, skip out-of-bounds cells (embedding padding)
                    if is_hex and max(abs(ax), abs(ay), abs(ax + ay)) > hex_center:
                        continue  # This is embedding padding, not a real collapsed cell
                    territory_player = self.territory_owner[game_idx, row, col].item()
                    if is_hex:
                        key = f"{ax},{ay},{az}"
                    else:
                        key = f"{ax},{ay}"
                    collapsed_spaces[key] = territory_player

        # Build board state
        board = BoardState(
            type=board_type,
            size=cpu_board_size,
            stacks=stacks,
            markers=markers,
            collapsedSpaces=collapsed_spaces,
        )

        # Build players list
        players = []
        for p in range(1, self.num_players + 1):
            players.append(Player(
                id=f"gpu_player_{p}",
                username=f"GPU Player {p}",
                type="ai",
                playerNumber=p,
                isReady=True,
                timeRemaining=600000,
                ringsInHand=self.rings_in_hand[game_idx, p].item(),
                eliminatedRings=self.eliminated_rings[game_idx, p].item(),
                territorySpaces=self.territory_count[game_idx, p].item(),
            ))

        # Map GPU phase to CPU phase
        gpu_phase = self.current_phase[game_idx].item()
        phase_map = {
            0: CPUGamePhase.RING_PLACEMENT,
            1: CPUGamePhase.MOVEMENT,
            2: CPUGamePhase.LINE_PROCESSING,
            3: CPUGamePhase.TERRITORY_PROCESSING,
            4: CPUGamePhase.GAME_OVER,
        }
        current_phase = phase_map.get(gpu_phase, CPUGamePhase.MOVEMENT)

        # Determine game status
        gpu_status = self.game_status[game_idx].item()
        if gpu_status == GameStatus.ACTIVE:
            game_status = CPUGameStatus.ACTIVE
        else:
            game_status = CPUGameStatus.COMPLETED

        now = datetime.now()

        # Compute canonical victory thresholds for shadow validation.
        # Per RR-CANON-R061/R062-v2 these depend on the board type and player count.
        from app.rules.core import (
            get_rings_per_player,
            get_territory_victory_minimum,
            get_territory_victory_threshold,
            get_victory_threshold,
        )

        rings_per_player = get_rings_per_player(board_type)
        victory_threshold = get_victory_threshold(board_type, self.num_players)
        territory_victory_threshold = get_territory_victory_threshold(board_type)
        # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
        territory_victory_minimum = get_territory_victory_minimum(board_type, self.num_players)
        total_rings_in_play = rings_per_player * self.num_players
        total_rings_eliminated = int(
            self.eliminated_rings[game_idx, 1 : self.num_players + 1].sum().item()
        )

        # LPS tracking (RR-CANON-R172) for shadow validation / debugging.
        lps_seen = self.lps_current_round_seen_mask[game_idx].cpu()
        lps_real = self.lps_current_round_real_action_mask[game_idx].cpu()
        lps_actor_mask = {
            int(p): bool(lps_real[p].item())
            for p in range(1, self.num_players + 1)
            if bool(lps_seen[p].item())
        }
        lps_first = int(self.lps_current_round_first_player[game_idx].item())
        lps_first_player = lps_first if lps_first > 0 else None
        lps_exclusive = int(self.lps_exclusive_player_for_completed_round[game_idx].item())
        lps_exclusive_player = lps_exclusive if lps_exclusive > 0 else None
        lps_consecutive_player_raw = int(
            self.lps_consecutive_exclusive_player[game_idx].item()
        )
        lps_consecutive_player = (
            lps_consecutive_player_raw if lps_consecutive_player_raw > 0 else None
        )

        # Convert must_move_from_y/x to must_move_from_stack_key for CPU validation
        # This constrains CPU move generation to the same stack the GPU is constrained to
        must_move_from_stack_key = None
        must_y = int(self.must_move_from_y[game_idx].item())
        must_x = int(self.must_move_from_x[game_idx].item())
        if must_y >= 0 and must_x >= 0:
            # Convert GPU grid coords to CPU stack key format
            ax, ay, az = grid_to_cpu_coords(must_y, must_x)
            if is_hex:
                must_move_from_stack_key = f"{ax},{ay},{az}"
            else:
                must_move_from_stack_key = f"{ax},{ay}"

        return GameState(
            id=f"gpu_game_{game_idx}",
            boardType=board_type,
            board=board,
            players=players,
            currentPhase=current_phase,
            currentPlayer=self.current_player[game_idx].item(),
            moveHistory=[],  # Not reconstructed for validation
            timeControl=TimeControl(initialTime=600000, increment=0, type="standard"),
            spectators=[],
            gameStatus=game_status,
            winner=self.winner[game_idx].item() if self.winner[game_idx].item() > 0 else None,
            createdAt=now,
            lastMoveAt=now,
            isRated=False,
            maxPlayers=self.num_players,
            totalRingsInPlay=total_rings_in_play,
            totalRingsEliminated=total_rings_eliminated,
            victoryThreshold=victory_threshold,
            territoryVictoryThreshold=territory_victory_threshold,
            territoryVictoryMinimum=territory_victory_minimum,
            lpsRoundIndex=int(self.lps_round_index[game_idx].item()),
            lpsCurrentRoundActorMask=lps_actor_mask,
            lpsExclusivePlayerForCompletedRound=lps_exclusive_player,
            lpsCurrentRoundFirstPlayer=lps_first_player,
            lpsConsecutiveExclusiveRounds=int(
                self.lps_consecutive_exclusive_rounds[game_idx].item()
            ),
            lpsRoundsRequired=int(self.lps_rounds_required),
            lpsConsecutiveExclusivePlayer=lps_consecutive_player,
            mustMoveFromStackKey=must_move_from_stack_key,
        )

    def get_active_mask(self) -> torch.Tensor:
        """Get mask of games that are still active.

        Returns:
            Boolean tensor (batch_size,) - True for active games
        """
        return self.game_status == GameStatus.ACTIVE

    def count_active(self) -> int:
        """Count number of active games."""
        return self.get_active_mask().sum().item()

    def to_feature_tensor(self, history_length: int = 4) -> torch.Tensor:
        """Convert batch state to neural network input features.

        Args:
            history_length: Number of history planes (not used for single state)

        Returns:
            Feature tensor (batch_size, channels, board_size, board_size)
        """
        # Simple feature encoding:
        # - Plane 0-3: Stack owner one-hot (players 1-4)
        # - Plane 4: Stack height normalized
        # - Plane 5-8: Marker owner one-hot
        # - Plane 9-12: Territory owner one-hot
        # - Plane 13: Current player encoding

        num_channels = 14
        features = torch.zeros(
            self.batch_size, num_channels, self.board_size, self.board_size,
            dtype=torch.float32, device=self.device
        )

        # Stack owner one-hot
        for p in range(1, self.num_players + 1):
            features[:, p-1] = (self.stack_owner == p).float()

        # Stack height (normalized)
        features[:, 4] = self.stack_height.float() / 5.0

        # Marker owner one-hot
        for p in range(1, self.num_players + 1):
            features[:, 5 + p - 1] = (self.marker_owner == p).float()

        # Territory owner one-hot
        for p in range(1, self.num_players + 1):
            features[:, 9 + p - 1] = (self.territory_owner == p).float()

        # Current player broadcast
        for i in range(self.batch_size):
            features[i, 13] = self.current_player[i].float() / self.num_players

        return features

    def extract_move_history(self, game_idx: int) -> List[Dict[str, Any]]:
        """Extract move history for a single game as list of dicts.

        Args:
            game_idx: Index of game in batch

        Returns:
            List of move dicts in format compatible with training data:
            [{'type': str, 'player': int, 'to': {'x': int, 'y': int}, ...}, ...]
        """
        moves = []
        num_moves = min(self.move_count[game_idx].item(), self.max_history_moves)

        # Move type names for output - must match MoveType enum values
        # from app/models/core.py for compatibility with jsonl_to_npz.py
        # MoveType.PLACE_RING = "place_ring", MoveType.MOVE_STACK = "move_stack"
        move_type_names = {
            MoveType.PLACEMENT: "place_ring",
            MoveType.MOVEMENT: "move_stack",
            MoveType.CAPTURE: "overtaking_capture",
            MoveType.LINE_FORMATION: "line_formation",
            MoveType.TERRITORY_CLAIM: "territory_claim",
            MoveType.SKIP: "skip_capture",
            MoveType.RECOVERY_SLIDE: "recovery_slide",
        }

        for i in range(num_moves):
            history_row = self.move_history[game_idx, i].cpu().tolist()
            move_type_code, player, from_y, from_x, to_y, to_x = history_row

            if move_type_code < 0:  # -1 indicates unused slot
                break

            # Skip NO_ACTION moves - they don't represent real game moves
            if move_type_code == MoveType.NO_ACTION:
                continue

            move_dict = {
                "type": move_type_names.get(move_type_code, f"unknown_{move_type_code}"),
                "player": int(player),
            }

            # Add positions based on move type
            if move_type_code == MoveType.PLACEMENT:
                move_dict["to"] = {"x": int(to_x), "y": int(to_y)}
            else:
                if from_x >= 0 and from_y >= 0:
                    move_dict["from"] = {"x": int(from_x), "y": int(from_y)}
                if to_x >= 0 and to_y >= 0:
                    move_dict["to"] = {"x": int(to_x), "y": int(to_y)}

            moves.append(move_dict)

        return moves

    def derive_victory_type(self, game_idx: int, max_moves: int) -> Tuple[str, Optional[str]]:
        """Derive victory type for a single game.

        Based on final game state, determines the victory type following
        GAME_RECORD_SPEC.md categories:
        - ring_elimination: Elimination threshold reached
        - territory: Territory threshold reached
        - timeout: Max moves reached (draw/stalemate)
        - lps: Last-player-standing (RR-CANON-R172)
        - stalemate: Draw by other means

        Args:
            game_idx: Index of game in batch
            max_moves: Maximum moves limit used

        Returns:
            Tuple of (victory_type, stalemate_tiebreaker or None)
        """
        status = self.game_status[game_idx].item()
        winner = self.winner[game_idx].item()
        move_count = self.move_count[game_idx].item()

        # Check for max moves reached
        if status == GameStatus.MAX_MOVES or move_count >= max_moves:
            if winner == 0:
                return ("timeout", None)
            # Winner by tiebreaker
            return ("stalemate", self._determine_tiebreaker(game_idx))

        # Check for draw
        if status == GameStatus.DRAW or winner == 0:
            return ("stalemate", self._determine_tiebreaker(game_idx))

        # Check victory conditions
        if winner > 0:
            from app.models import BoardType
            from app.rules.core import (
                get_territory_victory_minimum,
                get_victory_threshold,
            )

            board_type_map = {
                8: BoardType.SQUARE8,
                19: BoardType.SQUARE19,
                13: BoardType.HEXAGONAL,
            }
            board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)
            ring_elimination_threshold = get_victory_threshold(
                board_type,
                self.num_players,
            )
            # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
            territory_victory_minimum = get_territory_victory_minimum(board_type, self.num_players)

            # Territory victory per RR-CANON-R062-v2 (dual condition)
            player_territory = self.territory_count[game_idx, winner].item()
            total_territory = sum(
                self.territory_count[game_idx, p].item()
                for p in range(1, self.num_players + 1)
            )
            opponents_territory = total_territory - player_territory
            if (
                player_territory >= territory_victory_minimum
                and player_territory > opponents_territory
            ):
                return ("territory", None)

            # Ring-elimination victory (RR-CANON-R170/R061)
            if (
                self.rings_caused_eliminated[game_idx, winner].item()
                >= ring_elimination_threshold
            ):
                return ("ring_elimination", None)

            # Last-player-standing (RR-CANON-R172) via round tracker.
            lps_required = int(getattr(self, "lps_rounds_required", 3) or 3)
            if (
                self.lps_consecutive_exclusive_player[game_idx].item() == winner
                and self.lps_consecutive_exclusive_rounds[game_idx].item()
                >= lps_required
            ):
                return ("lps", None)

            # Defensive fallback for unexpected end conditions.
            return ("ring_elimination", None)

        return ("unknown", None)

    def _determine_tiebreaker(self, game_idx: int) -> str:
        """Determine tiebreaker used for stalemate/timeout games.

        Returns one of: 'territory', 'eliminated_rings', 'markers', 'last_actor'
        """
        # Check if territory counts differ
        territory_counts = self.territory_count[game_idx, 1:self.num_players+1].cpu().tolist()
        if len(set(territory_counts)) > 1:
            return "territory"

        # Check eliminated rings
        eliminated_counts = self.eliminated_rings[game_idx, 1:self.num_players+1].cpu().tolist()
        if len(set(eliminated_counts)) > 1:
            return "eliminated_rings"

        # Check marker counts
        marker_counts = []
        for p in range(1, self.num_players + 1):
            marker_counts.append((self.marker_owner[game_idx] == p).sum().item())
        if len(set(marker_counts)) > 1:
            return "markers"

        # Default to last_actor
        return "last_actor"


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


def generate_placement_moves_batch(
    state: BatchGameState,
    active_mask: Optional[torch.Tensor] = None,
) -> BatchMoves:
    """Generate all valid placement moves for active games.

    Per RingRift rules, placement is valid on:
    - Empty positions (stack_owner == 0): can place 1-3 rings
    - Occupied positions (stack_owner > 0): can place exactly 1 ring on top

    Placement is NOT valid on:
    - Collapsed spaces (is_collapsed == True)

    Note: The GPU engine simplifies by allowing placement on ANY non-collapsed
    position. The placement count (1 vs 1-3) is handled during move selection
    and application, not during move generation.

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
    board_size = state.board_size

    # Find all valid placement positions per game:
    # - Must not be collapsed
    # - Must not contain a marker (placement never occurs onto an existing marker)
    # - Game must be active
    # valid_positions: (batch_size, board_size, board_size) bool
    valid_positions = (
        (~state.is_collapsed)
        & (state.marker_owner == 0)
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
    active_mask: Optional[torch.Tensor] = None,
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
    directions = DIRECTIONS.to(device)

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
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

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
    active_mask: Optional[torch.Tensor] = None,
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
    import os
    if os.environ.get("RINGRIFT_GPU_MOVEMENT_LEGACY", "0") == "1":
        return _generate_movement_moves_batch_legacy(state, active_mask)
    return generate_movement_moves_batch_vectorized(state, active_mask)


def _generate_movement_moves_batch_legacy(
    state: BatchGameState,
    active_mask: Optional[torch.Tensor] = None,
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
    player_t = torch.tensor(candidate_player, dtype=torch.int64, device=device)

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
    active_mask: Optional[torch.Tensor] = None,
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
    import os
    if os.environ.get("RINGRIFT_GPU_CAPTURE_LEGACY", "0") == "1":
        return _generate_capture_moves_batch_legacy(state, active_mask)
    return generate_capture_moves_batch_vectorized(state, active_mask)


def generate_capture_moves_batch_vectorized(
    state: BatchGameState,
    active_mask: Optional[torch.Tensor] = None,
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
    directions = DIRECTIONS.to(device)
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

    # Find cells with stacks (potential targets) that are before any blocker
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

    # Min landing distance = max(stack_height, target_dist + 1)
    sd_min_landing = torch.maximum(sd_height, sd_target_dist + 1)

    # Expand for all possible landing distances: (n_valid_sd, max_dist)
    landing_dists = torch.arange(1, max_dist + 1, device=device).view(1, -1)  # (1, max_dist)

    # Expand sd data: (n_valid_sd, max_dist)
    sd_game_idx_exp = sd_game_idx.unsqueeze(1).expand(-1, max_dist)
    sd_from_y_exp = sd_from_y.unsqueeze(1).expand(-1, max_dist)
    sd_from_x_exp = sd_from_x.unsqueeze(1).expand(-1, max_dist)
    sd_target_dist_exp = sd_target_dist.unsqueeze(1).expand(-1, max_dist)
    sd_min_landing_exp = sd_min_landing.unsqueeze(1).expand(-1, max_dist)
    sd_dir_dy_exp = sd_dir_dy.unsqueeze(1).expand(-1, max_dist)
    sd_dir_dx_exp = sd_dir_dx.unsqueeze(1).expand(-1, max_dist)

    # Compute landing positions
    landing_y = sd_from_y_exp + sd_dir_dy_exp * landing_dists
    landing_x = sd_from_x_exp + sd_dir_dx_exp * landing_dists

    # Filter 1: landing_dist >= min_landing_dist
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
    active_mask: Optional[torch.Tensor] = None,
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
                target_y, target_x = None, None
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
                            target_y, target_x = check_y, check_x
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
) -> List[Tuple[int, int]]:
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

    captures: List[Tuple[int, int]] = []

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
                target_cap = cap_height_np[check_y, check_x]
                if my_cap_height >= target_cap:
                    target_y = check_y
                    target_x = check_x
                    target_dist = step
                # Any stack stops the search along this ray.
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
) -> Tuple[int, int]:
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
    mc = int(state.move_count[game_idx].item())
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()
    cap_height_np = state.cap_height[game_idx].cpu().numpy()
    marker_owner_np = state.marker_owner[game_idx].cpu().numpy()
    is_collapsed_np = state.is_collapsed[game_idx].cpu().numpy()

    # Record in history
    if mc < state.max_history_moves:
        state.move_history[game_idx, mc, 0] = MoveType.CAPTURE
        state.move_history[game_idx, mc, 1] = player
        state.move_history[game_idx, mc, 2] = from_y
        state.move_history[game_idx, mc, 3] = from_x
        state.move_history[game_idx, mc, 4] = to_y
        state.move_history[game_idx, mc, 5] = to_x
    state.move_count[game_idx] += 1

    # Capture move representation:
    # - (from -> landing) is passed in as (to_y, to_x)
    # - The target stack is implicit as the first stack along the ray
    attacker_height = int(stack_height_np[from_y, from_x])
    attacker_cap_height = int(cap_height_np[from_y, from_x])

    dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
    dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
    dist = max(abs(to_y - from_y), abs(to_x - from_x))

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
        # Defensive fallback: treat as movement.
        state.stack_height[game_idx, to_y, to_x] = attacker_height
        state.stack_owner[game_idx, to_y, to_x] = player
        state.cap_height[game_idx, to_y, to_x] = min(attacker_cap_height, attacker_height)
        state.stack_height[game_idx, from_y, from_x] = 0
        state.stack_owner[game_idx, from_y, from_x] = 0
        state.cap_height[game_idx, from_y, from_x] = 0
        state.marker_owner[game_idx, from_y, from_x] = player
        return to_y, to_x

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

    new_target_height = max(0, target_height - 1)
    state.stack_height[game_idx, target_y, target_x] = new_target_height
    if new_target_height <= 0:
        state.stack_owner[game_idx, target_y, target_x] = 0
        state.cap_height[game_idx, target_y, target_x] = 0
    else:
        new_target_cap = target_cap_height - 1
        if new_target_cap <= 0:
            new_target_cap = 1
        if new_target_cap > new_target_height:
            new_target_cap = new_target_height
        state.cap_height[game_idx, target_y, target_x] = new_target_cap

    if target_owner != 0 and target_owner != player:
        state.buried_rings[game_idx, target_owner] += 1

    new_height = attacker_height + 1 - landing_ring_cost
    state.stack_height[game_idx, to_y, to_x] = new_height
    state.stack_owner[game_idx, to_y, to_x] = player

    new_cap = attacker_cap_height - landing_ring_cost
    if new_cap <= 0:
        new_cap = 1
    if new_cap > new_height:
        new_cap = new_height
    state.cap_height[game_idx, to_y, to_x] = new_cap

    # Clear origin stack and leave departure marker.
    state.stack_height[game_idx, from_y, from_x] = 0
    state.stack_owner[game_idx, from_y, from_x] = 0
    state.cap_height[game_idx, from_y, from_x] = 0
    state.marker_owner[game_idx, from_y, from_x] = player

    return to_y, to_x


# =============================================================================
# Recovery Slide Move Generation (RR-CANON-R110-R115)
# =============================================================================


def generate_recovery_moves_batch(
    state: BatchGameState,
    active_mask: Optional[torch.Tensor] = None,
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


# =============================================================================
# Batch Move Application
# =============================================================================


def apply_placement_moves_batch_vectorized(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized placement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    y = moves.from_y[global_indices].long()
    x = moves.from_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Gather destination state
    dest_owner = state.stack_owner[game_indices, y, x]
    dest_height = state.stack_height[game_indices, y, x]
    dest_cap = state.cap_height[game_indices, y, x]

    # Record move history (for games with history space)
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_y = y[history_mask]
        hist_x = x[history_mask]
        hist_players = players[history_mask]
        hist_move_type = move_type[history_mask]

        # Cast to match move_history dtype (int16)
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = hist_move_type.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = hist_players.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = hist_y.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = hist_x.to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = hist_y.to(hist_dtype)  # to_y same for placement
        state.move_history[hist_games, hist_move_idx, 5] = hist_x.to(hist_dtype)  # to_x same for placement

    # Compute new values based on destination state
    is_empty = dest_height <= 0
    is_same_owner = dest_owner == players

    # New height: 1 if empty, else dest_height + 1 (clamped to 127)
    new_height = torch.where(is_empty, torch.ones_like(dest_height), torch.clamp(dest_height + 1, max=127))

    # New cap_height: 1 if empty or different owner, else dest_cap + 1 (clamped to 127)
    new_cap = torch.where(
        is_empty | ~is_same_owner,
        torch.ones_like(dest_cap),
        torch.clamp(dest_cap + 1, max=127)
    )

    # Update stack state (cast to match dtypes)
    state.stack_owner[game_indices, y, x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, y, x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, y, x] = new_cap.to(state.cap_height.dtype)

    # Handle buried rings for opponent stacks - vectorized with index_put_
    is_opponent = ~is_empty & (dest_owner != 0) & (dest_owner != players)
    if is_opponent.any():
        opp_games = game_indices[is_opponent]
        opp_owners = dest_owner[is_opponent].long()
        opp_ones = torch.ones(opp_games.shape[0], dtype=state.buried_rings.dtype, device=device)
        state.buried_rings.index_put_(
            (opp_games, opp_owners),
            opp_ones,
            accumulate=True
        )

    # Update rings_in_hand - vectorized with index_put_
    neg_ones = torch.full((n_games,), -1, dtype=state.rings_in_hand.dtype, device=device)
    state.rings_in_hand.index_put_(
        (game_indices, players.long()),
        neg_ones,
        accumulate=True
    )

    # Update must_move_from (cast to match dtype)
    state.must_move_from_y[game_indices] = y.to(state.must_move_from_y.dtype)
    state.must_move_from_x[game_indices] = x.to(state.must_move_from_x.dtype)

    # Advance move counter
    state.move_count[game_indices] += 1


def _apply_placement_moves_batch_legacy(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Legacy Python-loop based placement application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        y = moves.from_y[global_idx].item()
        x = moves.from_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = y
            state.move_history[g, move_idx, 3] = x
            state.move_history[g, move_idx, 4] = y
            state.move_history[g, move_idx, 5] = x

        dest_owner = int(state.stack_owner[g, y, x].item())
        dest_height = int(state.stack_height[g, y, x].item())
        dest_cap = int(state.cap_height[g, y, x].item())

        if dest_height <= 0:
            state.stack_owner[g, y, x] = player
            state.stack_height[g, y, x] = 1
            state.cap_height[g, y, x] = 1
        else:
            new_height = min(127, dest_height + 1)
            state.stack_owner[g, y, x] = player
            state.stack_height[g, y, x] = new_height
            if dest_owner == player:
                state.cap_height[g, y, x] = min(127, dest_cap + 1)
            else:
                state.cap_height[g, y, x] = 1

            if dest_owner not in (0, player):
                state.buried_rings[g, dest_owner] += 1
        state.rings_in_hand[g, player] -= 1
        state.must_move_from_y[g] = y
        state.must_move_from_x[g] = x
        state.move_count[g] += 1


def apply_placement_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Apply selected placement moves to batch state (in-place).

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_PLACEMENT_LEGACY", "0") == "1":
        _apply_placement_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_placement_moves_batch_vectorized(state, move_indices, moves)


def apply_movement_moves_batch_vectorized(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized movement move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    from_y = moves.from_y[global_indices].long()
    from_x = moves.from_x[global_indices].long()
    to_y = moves.to_y[global_indices].long()
    to_x = moves.to_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Record move history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = move_type[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    # Get moving stack info
    moving_height = state.stack_height[game_indices, from_y, from_x]
    moving_cap_height = state.cap_height[game_indices, from_y, from_x]

    # Compute direction and distance for each move
    dy = torch.sign(to_y - from_y)
    dx = torch.sign(to_x - from_x)
    dist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))
    max_dist = dist.max().item() if n_games > 0 else 0

    # Process markers along path (flip opposing markers)
    # Create path positions for all games: (n_games, max_dist-1)
    if max_dist > 1:
        steps = torch.arange(1, max_dist, device=device).view(1, -1)  # (1, max_dist-1)
        path_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps  # (n_games, max_dist-1)
        path_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps

        # Mask for valid path positions (step < dist, so we don't include destination)
        valid_path = steps < dist.unsqueeze(1)

        # Clamp for safe indexing
        path_y_safe = torch.clamp(path_y, 0, board_size - 1).long()
        path_x_safe = torch.clamp(path_x, 0, board_size - 1).long()

        # Get marker owners along path
        game_indices_exp = game_indices.unsqueeze(1).expand(-1, max_dist - 1)
        path_marker_owners = state.marker_owner[game_indices_exp, path_y_safe, path_x_safe]

        # Find opponent markers to flip (not 0 and not player)
        players_exp = players.unsqueeze(1).expand(-1, max_dist - 1)
        is_opponent_marker = (path_marker_owners != 0) & (path_marker_owners != players_exp) & valid_path

        # Flip opponent markers
        if is_opponent_marker.any():
            flip_games = game_indices_exp[is_opponent_marker]
            flip_y = path_y_safe[is_opponent_marker]
            flip_x = path_x_safe[is_opponent_marker]
            flip_players = players_exp[is_opponent_marker]
            state.marker_owner[flip_games, flip_y, flip_x] = flip_players.to(state.marker_owner.dtype)

    # Handle landing on own marker (collapse cost)
    dest_marker = state.marker_owner[game_indices, to_y, to_x]
    landing_on_own_marker = dest_marker == players
    landing_ring_cost = landing_on_own_marker.int()

    if landing_on_own_marker.any():
        collapse_games = game_indices[landing_on_own_marker]
        collapse_y = to_y[landing_on_own_marker]
        collapse_x = to_x[landing_on_own_marker]
        state.is_collapsed[collapse_games, collapse_y, collapse_x] = True
        state.marker_owner[collapse_games, collapse_y, collapse_x] = 0

    # Clear origin
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Set destination
    new_height = torch.clamp(moving_height - landing_ring_cost, min=1)
    new_cap_height = torch.clamp(moving_cap_height - landing_ring_cost, min=1)
    state.stack_owner[game_indices, to_y, to_x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, to_y, to_x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, to_y, to_x] = new_cap_height.to(state.cap_height.dtype)

    # Advance turn
    state.move_count[game_indices] += 1
    new_player = (players % state.num_players) + 1
    state.current_player[game_indices] = new_player.to(state.current_player.dtype)


def _apply_movement_moves_batch_legacy(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Legacy Python-loop based movement application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        from_y = moves.from_y[global_idx].item()
        from_x = moves.from_x[global_idx].item()
        to_y = moves.to_y[global_idx].item()
        to_x = moves.to_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

        moving_height = state.stack_height[g, from_y, from_x].item()
        moving_cap_height = state.cap_height[g, from_y, from_x].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 0
        if dest_marker == player:
            landing_ring_cost = 1
            state.is_collapsed[g, to_y, to_x] = True
            state.marker_owner[g, to_y, to_x] = 0

        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0

        new_height = max(1, moving_height - landing_ring_cost)
        new_cap_height = max(1, moving_cap_height - landing_ring_cost)
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = new_height
        state.cap_height[g, to_y, to_x] = new_cap_height

        state.move_count[g] += 1
        state.current_player[g] = (player % state.num_players) + 1


def apply_movement_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Apply selected movement moves to batch state (in-place).

    Per RR-CANON-R090-R092:
    - Stack moves from origin to destination
    - Origin becomes empty
    - Destination gets merged stack (if own stack) or new stack
    - Markers along path: flip on pass, collapse cost on landing

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_MOVEMENT_APPLY_LEGACY", "0") == "1":
        _apply_movement_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_movement_moves_batch_vectorized(state, move_indices, moves)


def apply_capture_moves_batch_vectorized(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Vectorized capture move application.

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    active_mask = state.get_active_mask()

    # Filter to games with valid moves
    has_moves = moves.moves_per_game > 0
    valid_local_idx = move_indices < moves.moves_per_game
    process_mask = active_mask & has_moves & valid_local_idx

    if not process_mask.any():
        return

    # Get game indices to process
    game_indices = torch.where(process_mask)[0]
    n_games = game_indices.shape[0]

    # Compute global move indices
    global_indices = moves.move_offsets[game_indices] + move_indices[game_indices]

    # Gather move data
    from_y = moves.from_y[global_indices].long()
    from_x = moves.from_x[global_indices].long()
    to_y = moves.to_y[global_indices].long()
    to_x = moves.to_x[global_indices].long()
    move_type = moves.move_type[global_indices]
    players = state.current_player[game_indices]

    # Record move history
    move_idx = state.move_count[game_indices]
    history_mask = move_idx < state.max_history_moves
    if history_mask.any():
        hist_games = game_indices[history_mask]
        hist_move_idx = move_idx[history_mask].long()
        hist_dtype = state.move_history.dtype
        state.move_history[hist_games, hist_move_idx, 0] = move_type[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 1] = players[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 2] = from_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 3] = from_x[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 4] = to_y[history_mask].to(hist_dtype)
        state.move_history[hist_games, hist_move_idx, 5] = to_x[history_mask].to(hist_dtype)

    # Get attacker stack info
    attacker_height = state.stack_height[game_indices, from_y, from_x]
    attacker_cap_height = state.cap_height[game_indices, from_y, from_x]

    # Get defender info
    defender_owner = state.stack_owner[game_indices, to_y, to_x]
    defender_height = state.stack_height[game_indices, to_y, to_x]

    # Compute direction and distance for path processing
    dy = torch.sign(to_y - from_y)
    dx = torch.sign(to_x - from_x)
    dist = torch.maximum(torch.abs(to_y - from_y), torch.abs(to_x - from_x))
    max_dist = dist.max().item() if n_games > 0 else 0

    # Process markers along path (flip opposing markers)
    if max_dist > 1:
        steps = torch.arange(1, max_dist, device=device).view(1, -1)
        path_y = from_y.unsqueeze(1) + dy.unsqueeze(1) * steps
        path_x = from_x.unsqueeze(1) + dx.unsqueeze(1) * steps

        valid_path = steps < dist.unsqueeze(1)

        path_y_safe = torch.clamp(path_y, 0, board_size - 1).long()
        path_x_safe = torch.clamp(path_x, 0, board_size - 1).long()

        game_indices_exp = game_indices.unsqueeze(1).expand(-1, max_dist - 1)
        path_marker_owners = state.marker_owner[game_indices_exp, path_y_safe, path_x_safe]

        players_exp = players.unsqueeze(1).expand(-1, max_dist - 1)
        is_opponent_marker = (path_marker_owners != 0) & (path_marker_owners != players_exp) & valid_path

        if is_opponent_marker.any():
            flip_games = game_indices_exp[is_opponent_marker]
            flip_y = path_y_safe[is_opponent_marker]
            flip_x = path_x_safe[is_opponent_marker]
            flip_players = players_exp[is_opponent_marker]
            state.marker_owner[flip_games, flip_y, flip_x] = flip_players.to(state.marker_owner.dtype)

    # Eliminate defender's ring (update tracking tensors) - vectorized using index_put_
    ones = torch.ones(n_games, dtype=state.eliminated_rings.dtype, device=device)
    state.eliminated_rings.index_put_(
        (game_indices, defender_owner.long()),
        ones,
        accumulate=True
    )
    state.rings_caused_eliminated.index_put_(
        (game_indices, players.long()),
        ones,
        accumulate=True
    )

    # Clear origin
    state.stack_owner[game_indices, from_y, from_x] = 0
    state.stack_height[game_indices, from_y, from_x] = 0
    state.cap_height[game_indices, from_y, from_x] = 0

    # Merge stacks at destination
    new_height = torch.clamp(attacker_height + defender_height - 1, max=5)
    new_cap_height = torch.clamp(attacker_cap_height, max=5)
    state.stack_owner[game_indices, to_y, to_x] = players.to(state.stack_owner.dtype)
    state.stack_height[game_indices, to_y, to_x] = new_height.to(state.stack_height.dtype)
    state.cap_height[game_indices, to_y, to_x] = new_cap_height.to(state.cap_height.dtype)

    # Place marker for attacker
    state.marker_owner[game_indices, to_y, to_x] = players.to(state.marker_owner.dtype)

    # Advance turn
    state.move_count[game_indices] += 1
    new_player = (players % state.num_players) + 1
    state.current_player[game_indices] = new_player.to(state.current_player.dtype)


def _apply_capture_moves_batch_legacy(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Legacy Python-loop based capture application.

    Kept for debugging and comparison.
    """
    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        from_y = moves.from_y[global_idx].item()
        from_x = moves.from_x[global_idx].item()
        to_y = moves.to_y[global_idx].item()
        to_x = moves.to_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

        attacker_height = state.stack_height[g, from_y, from_x].item()
        attacker_cap_height = state.cap_height[g, from_y, from_x].item()

        defender_owner = state.stack_owner[g, to_y, to_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        state.eliminated_rings[g, defender_owner] += 1
        state.rings_caused_eliminated[g, player] += 1

        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0
        state.cap_height[g, from_y, from_x] = 0

        new_height = attacker_height + defender_height - 1
        new_cap_height = attacker_cap_height
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(5, new_height)
        state.cap_height[g, to_y, to_x] = min(5, new_cap_height)

        state.marker_owner[g, to_y, to_x] = player

        state.move_count[g] += 1
        state.current_player[g] = (player % state.num_players) + 1


def apply_capture_moves_batch(
    state: BatchGameState,
    move_indices: torch.Tensor,
    moves: BatchMoves,
) -> None:
    """Apply selected capture moves to batch state (in-place).

    Per RR-CANON-R100-R103:
    - Attacker moves onto defender stack
    - Defender's top ring is eliminated
    - Stacks merge (attacker on top)
    - Control transfers to attacker

    Args:
        state: BatchGameState to modify
        move_indices: (batch_size,) index into moves for each game's selected move
        moves: BatchMoves containing all candidate moves
    """
    if os.environ.get("RINGRIFT_GPU_CAPTURE_APPLY_LEGACY", "0") == "1":
        _apply_capture_moves_batch_legacy(state, move_indices, moves)
    else:
        apply_capture_moves_batch_vectorized(state, move_indices, moves)


# =============================================================================
# Line Detection and Processing (RR-CANON-R120-R122)
# =============================================================================


@dataclass
class DetectedLine:
    """A detected marker line with metadata for processing."""
    positions: List[Tuple[int, int]]  # All marker positions in the line
    length: int                        # Total length of line
    is_overlength: bool               # True if len > required_length
    direction: Tuple[int, int]        # Direction vector (dy, dx)


def get_required_line_length(board_size: int, num_players: int) -> int:
    """Get required line length per RR-CANON-R120.

    Args:
        board_size: Board dimension
        num_players: Number of players

    Returns:
        Required line length (3 or 4)
    """
    # square8 (8x8) with 3-4 players uses line length 3, all others use 4
    if board_size == 8 and num_players >= 3:
        return 3
    return 4


def detect_lines_vectorized(
    state: BatchGameState,
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized line detection returning positions mask and line count.

    Per RR-CANON-R120: A line is a sequence of consecutive MARKERS (not stacks)
    of the same player, with length >= required_length.

    This is a fast vectorized implementation for MCTS. It returns:
    1. A mask of positions that are part of valid lines
    2. A count of positions per game that are in lines (proxy for line count)

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        Tuple of:
        - in_line_mask: (batch_size, board_size, board_size) bool tensor
        - line_position_counts: (batch_size,) int tensor with count of line positions
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    required_length = get_required_line_length(board_size, state.num_players)

    if game_mask is None:
        game_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Per RR-CANON-R120: Lines are formed by MARKERS, not stacks
    # A marker at (y,x) can be part of a line only if no stack is there
    player_markers = (
        (state.marker_owner == player) &
        (state.stack_owner == 0) &
        game_mask.view(-1, 1, 1)
    )

    # Output mask: positions that are part of any line
    in_line_mask = torch.zeros_like(player_markers)

    # Check all 4 directions
    # For each direction, we check if there's a sequence of required_length markers

    # Direction 1: Horizontal (dy=0, dx=1)
    # Use cumsum trick: if we have N consecutive markers, positions i to i+N-1
    # all get marked. We do this by checking windows of size required_length.

    # Note: Using explicit OR + assignment instead of |= due to MPS device quirk
    # Compute horizontal windows
    if board_size >= required_length:
        # Sum of required_length consecutive horizontal positions
        markers_float = player_markers.float()

        # Horizontal: sum along x-axis
        cumsum_h = markers_float.cumsum(dim=2)
        # Window sum at position x is cumsum[x+req-1] - cumsum[x-1]
        # But we need to handle edge cases
        padded_cumsum_h = torch.cat([
            torch.zeros(batch_size, board_size, 1, device=device),
            cumsum_h
        ], dim=2)
        # window_sum[y, x] = cumsum_h[y, x+req-1] - cumsum_h[y, x-1]
        #                  = padded_cumsum_h[y, x+req] - padded_cumsum_h[y, x]
        # This gives sum of markers from x to x+req-1
        window_sums_h = padded_cumsum_h[:, :, required_length:] - padded_cumsum_h[:, :, :-required_length]
        # window_sums_h shape: (batch, board_size, board_size - req + 1)
        # A window is a complete line if sum == required_length
        complete_windows_h = (window_sums_h == required_length)
        # Mark all positions in complete windows
        for offset in range(required_length):
            # Position x is in a complete window if any window starting at x-offset is complete
            # Window starting at x covers positions x to x+req-1
            # So position p is covered by windows starting at max(0, p-req+1) to p
            if offset < complete_windows_h.shape[2]:
                start = offset
                end = board_size - required_length + offset + 1
                in_line_mask[:, :, start:end] = in_line_mask[:, :, start:end] | complete_windows_h

        # Vertical: sum along y-axis
        cumsum_v = markers_float.cumsum(dim=1)
        padded_cumsum_v = torch.cat([
            torch.zeros(batch_size, 1, board_size, device=device),
            cumsum_v
        ], dim=1)
        window_sums_v = padded_cumsum_v[:, required_length:, :] - padded_cumsum_v[:, :-required_length, :]
        complete_windows_v = (window_sums_v == required_length)
        for offset in range(required_length):
            if offset < complete_windows_v.shape[1]:
                start = offset
                end = board_size - required_length + offset + 1
                in_line_mask[:, start:end, :] = in_line_mask[:, start:end, :] | complete_windows_v

        # Diagonal (dy=1, dx=1): Need to handle diagonals
        # Create shifted versions and compare
        # A diagonal line at (y,x) requires markers at (y+i, x+i) for i in 0..req-1
        diag_mask = torch.ones(batch_size, board_size, board_size, dtype=torch.bool, device=device)
        for i in range(required_length):
            # Shifted marker mask: markers_float shifted by (i, i)
            if i == 0:
                shifted = player_markers
            else:
                shifted = torch.zeros_like(player_markers)
                shifted[:, :-i, :-i] = player_markers[:, i:, i:]
            diag_mask = diag_mask & shifted

        # diag_mask[y,x] = True if there's a diagonal line starting at (y,x)
        # Valid starting positions: y <= board_size - req, x <= board_size - req
        diag_mask[:, board_size - required_length + 1:, :] = False
        diag_mask[:, :, board_size - required_length + 1:] = False

        # Mark all positions covered by diagonal lines
        valid_region = board_size - required_length + 1
        for i in range(required_length):
            y_start, y_end = i, valid_region + i
            x_start, x_end = i, valid_region + i
            in_line_mask[:, y_start:y_end, x_start:x_end] = \
                in_line_mask[:, y_start:y_end, x_start:x_end] | diag_mask[:, :valid_region, :valid_region]

        # Anti-diagonal (dy=1, dx=-1): markers at (y+i, x-i) for i in 0..req-1
        anti_diag_mask = torch.ones(batch_size, board_size, board_size, dtype=torch.bool, device=device)
        for i in range(required_length):
            if i == 0:
                shifted = player_markers
            else:
                shifted = torch.zeros_like(player_markers)
                # Shift: target[y, x] = source[y+i, x-i]
                # So we need: shifted[:, :-i, i:] = markers[:, i:, :-i]
                shifted[:, :-i, i:] = player_markers[:, i:, :-i]
            anti_diag_mask = anti_diag_mask & shifted

        # Valid starting positions: y <= board_size - req, x >= req - 1
        anti_diag_mask[:, board_size - required_length + 1:, :] = False
        anti_diag_mask[:, :, :required_length - 1] = False

        # Mark all positions covered by anti-diagonal lines
        valid_y = board_size - required_length + 1
        valid_x_start = required_length - 1
        for i in range(required_length):
            # Anti-diag line starting at (y,x) covers (y+i, x-i)
            y_start, y_end = i, valid_y + i
            x_start, x_end = valid_x_start - i, board_size - i
            in_line_mask[:, y_start:y_end, x_start:x_end] = \
                in_line_mask[:, y_start:y_end, x_start:x_end] | anti_diag_mask[:, :valid_y, valid_x_start:]

    # Count line positions per game
    line_position_counts = in_line_mask.view(batch_size, -1).sum(dim=1)

    return in_line_mask, line_position_counts


def has_lines_batch_vectorized(
    state: BatchGameState,
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast vectorized check for whether a player has any lines.

    This is the fastest path for MCTS victory checking - just returns
    a boolean tensor indicating which games have lines for this player.

    Args:
        state: Current batch game state
        player: Player number to check
        game_mask: Mask of games to check (optional)

    Returns:
        (batch_size,) bool tensor - True if player has at least one line
    """
    _, line_counts = detect_lines_vectorized(state, player, game_mask)
    return line_counts > 0


def detect_lines_with_metadata(
    state: BatchGameState,
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> List[List[DetectedLine]]:
    """Detect lines with full metadata including overlength status.

    Per RR-CANON-R120: A line for player P is a maximal sequence of positions
    where each position contains a MARKER of P (no stacks, no collapsed spaces).

    Returns structured line data including whether each line is overlength,
    enabling proper Option 1/2 handling per RR-CANON-R122.

    Optimized 2025-12-13: Use vectorized early-exit and numpy for iteration.

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        List of lists of DetectedLine objects, one list per game
    """
    batch_size = state.batch_size
    board_size = state.board_size
    num_players = state.num_players

    required_length = get_required_line_length(board_size, num_players)

    if game_mask is None:
        game_mask = torch.ones(batch_size, dtype=torch.bool, device=state.device)

    lines_per_game: List[List[DetectedLine]] = [[] for _ in range(batch_size)]

    # Early exit: use vectorized detection to quickly identify games WITH lines
    _, line_counts = detect_lines_vectorized(state, player, game_mask)
    games_with_lines = (line_counts > 0) & game_mask

    if not games_with_lines.any():
        return lines_per_game

    # Get marker mask for all games at once (batch, H, W)
    player_markers_batch = (state.marker_owner == player) & (state.stack_owner == 0)

    # 4 directions to check for lines: horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # Only process games that have lines
    games_to_check = games_with_lines.nonzero(as_tuple=True)[0].tolist()

    for g in games_to_check:
        # Get this game's markers as numpy for faster iteration
        player_markers = player_markers_batch[g].cpu().numpy()

        # Track which positions have been assigned to a line using numpy array
        # (faster than Python set for dense coordinate checks)
        assigned = np.zeros((board_size, board_size), dtype=np.bool_)

        # Check each direction for lines
        for dy, dx in directions:
            for start_y in range(board_size):
                for start_x in range(board_size):
                    if assigned[start_y, start_x]:
                        continue
                    if not player_markers[start_y, start_x]:
                        continue

                    # Trace line in this direction
                    line_positions = [(start_y, start_x)]
                    y, x = start_y + dy, start_x + dx

                    while 0 <= y < board_size and 0 <= x < board_size:
                        if player_markers[y, x] and not assigned[y, x]:
                            line_positions.append((y, x))
                            y, x = y + dy, x + dx
                        else:
                            break

                    # If line meets required length, record it
                    if len(line_positions) >= required_length:
                        for pos_y, pos_x in line_positions:
                            assigned[pos_y, pos_x] = True

                        lines_per_game[g].append(DetectedLine(
                            positions=line_positions,
                            length=len(line_positions),
                            is_overlength=(len(line_positions) > required_length),
                            direction=(dy, dx),
                        ))

    return lines_per_game


def detect_lines_batch(
    state: BatchGameState,
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> List[List[Tuple[int, int]]]:
    """Detect lines of consecutive same-owner MARKERS for a player.

    Per RR-CANON-R120: A line for player P is a maximal sequence of positions
    where each position contains a MARKER of P (no stacks, no collapsed spaces).

    Line length requirement is player-count-aware:
    - square8 (board_size=8) with 2 players: 4 consecutive markers
    - square8 (board_size=8) with 3-4 players: 3 consecutive markers
    - square19 (board_size=19): 4 consecutive markers (all player counts)
    - hexagonal (board_size=13): 4 consecutive markers (all player counts)

    IMPORTANT: Per RR-CANON-R120, lines are formed by MARKERS, not stacks.
    A position with a stack cannot be part of a marker line.

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        List of lists of (y, x) tuples, one per game, containing all line positions
    """
    # Use the metadata version and flatten to just positions
    lines_with_meta = detect_lines_with_metadata(state, player, game_mask)

    lines_per_game = []
    for game_lines in lines_with_meta:
        all_positions = []
        for line in game_lines:
            all_positions.extend(line.positions)
        lines_per_game.append(all_positions)

    return lines_per_game


def _eliminate_one_ring_from_any_stack(
    state: BatchGameState,
    game_idx: int,
    player: int,
) -> bool:
    """Eliminate one ring from any controlled stack.

    Per RR-CANON-R122: Any controlled stack is eligible for line elimination,
    including height-1 standalone rings.

    Optimized 2025-12-13: Use numpy to find eligible stack without .item() calls.

    Args:
        state: BatchGameState to modify
        game_idx: Game index
        player: Player performing elimination

    Returns:
        True if elimination was performed, False if no eligible stack found
    """
    # Use numpy to find first eligible stack (avoids .item() per cell)
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()

    # Find cells where this player has stacks with height > 0
    eligible = (stack_owner_np == player) & (stack_height_np > 0)
    positions = np.argwhere(eligible)

    if len(positions) == 0:
        return False

    # Take first eligible position
    y, x = int(positions[0, 0]), int(positions[0, 1])
    stack_height = int(stack_height_np[y, x])

    # Eliminate one ring from top
    state.stack_height[game_idx, y, x] = stack_height - 1
    state.eliminated_rings[game_idx, player] += 1
    # Player eliminates their own ring for line collapse cost
    state.rings_caused_eliminated[game_idx, player] += 1

    # If stack is now empty, clear ownership
    if stack_height - 1 == 0:
        state.stack_owner[game_idx, y, x] = 0

    return True


def process_lines_batch(
    state: BatchGameState,
    game_mask: Optional[torch.Tensor] = None,
    option2_probability: float = 0.3,
) -> None:
    """Process formed marker lines for all players (in-place).

    Per RR-CANON-R121-R122:
    - Lines are formed by MARKERS (not stacks)
    - Exact-length lines: Collapse all markers, pay one ring elimination
    - Overlength lines (len > required): Player chooses Option 1 or Option 2
      - Option 1: Collapse ALL markers, pay one ring elimination
      - Option 2: Collapse exactly required_length markers, NO elimination cost

    For GPU training, we implement probabilistic Option 1/2 selection for overlength
    lines to expose the AI to both strategies.

    Optimized 2025-12-13: Pre-extract numpy arrays, batch random generation.

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
        option2_probability: Probability of choosing Option 2 for overlength lines
                            (default 0.3 - prefer Option 1 for more territory)
    """
    batch_size = state.batch_size
    board_size = state.board_size
    device = state.device

    if game_mask is None:
        game_mask = state.get_active_mask()

    required_length = get_required_line_length(board_size, state.num_players)

    # Pre-generate random values for option selection (max 100 lines per batch is generous)
    # This avoids per-line .item() calls for random decisions
    max_lines_estimate = 100
    random_vals = torch.rand(max_lines_estimate, device=device).cpu().numpy()
    random_idx = 0

    for p in range(1, state.num_players + 1):
        lines_with_meta = detect_lines_with_metadata(state, p, game_mask)

        for g in range(batch_size):
            if not game_mask[g]:
                continue

            game_lines = lines_with_meta[g]
            if not game_lines:
                continue

            # Pre-extract stack_owner for this game as numpy for faster checks
            stack_owner_np = state.stack_owner[g].cpu().numpy()

            # Process each line individually per RR-CANON-R121
            for line in game_lines:
                positions_to_collapse = line.positions

                if line.is_overlength:
                    # Per RR-CANON-R122 Case 2: Overlength line - Option 1 or Option 2
                    # Use pre-generated random value (avoid .item() call)
                    use_option2 = random_vals[random_idx % max_lines_estimate] < option2_probability
                    random_idx += 1

                    if use_option2:
                        # Option 2: Collapse exactly required_length markers, NO elimination
                        # Per RR-CANON-R122: Player can choose which markers to collapse
                        # For training variety, randomly select which subset to collapse
                        all_positions = line.positions
                        if len(all_positions) > required_length:
                            # Randomly select which required_length positions to collapse
                            indices = torch.randperm(len(all_positions), device=device)[:required_length]
                            indices = indices.sort().values  # Keep line order for determinism
                            positions_to_collapse = [all_positions[i] for i in indices.tolist()]
                        else:
                            positions_to_collapse = all_positions[:required_length]
                        # No elimination cost for Option 2
                    else:
                        # Option 1: Collapse ALL markers, pay one ring elimination
                        positions_to_collapse = line.positions
                        # Check if player can pay elimination cost - use numpy
                        if (stack_owner_np == p).any():
                            _eliminate_one_ring_from_any_stack(state, g, p)
                        # If no stack available, still collapse (per RR-CANON-R122 interpretation B)
                else:
                    # Exact-length line: Must pay elimination cost
                    # Per RR-CANON-R122 Case 1: len == requiredLen
                    # Use numpy for check (avoid .item())
                    if (stack_owner_np == p).any():
                        _eliminate_one_ring_from_any_stack(state, g, p)

                # Collapse the selected markers to territory
                for (y, x) in positions_to_collapse:
                    # Remove marker and convert to territory (collapsed space)
                    state.marker_owner[g, y, x] = 0
                    state.territory_owner[g, y, x] = p
                    state.is_collapsed[g, y, x] = True
                    state.territory_count[g, p] += 1


# =============================================================================
# Territory Processing (RR-CANON-R140-R146)
# =============================================================================


def _find_eligible_territory_cap(
    state: BatchGameState,
    game_idx: int,
    player: int,
    excluded_positions: Optional[set] = None,
) -> Optional[Tuple[int, int, int]]:
    """Find an eligible stack for territory self-elimination.

    Per RR-CANON-R145: Eligible targets for territory processing are:
    - Multicolor stacks (player controls but other colors buried)
    - Single-color stacks of height > 1

    Height-1 standalone rings are NOT eligible for territory elimination.

    Optimized 2025-12-13: Use numpy to find eligible stack without .item() calls.

    Args:
        state: BatchGameState
        game_idx: Game index
        player: Player performing territory processing
        excluded_positions: Set of (y, x) positions to exclude (e.g., in region)

    Returns:
        Tuple of (y, x, cap_height) or None if no eligible stack
    """
    board_size = state.board_size
    if excluded_positions is None:
        excluded_positions = set()

    # Use numpy to find eligible stacks (avoids .item() per cell)
    stack_owner_np = state.stack_owner[game_idx].cpu().numpy()
    stack_height_np = state.stack_height[game_idx].cpu().numpy()

    # Per RR-CANON-R145: Height-1 standalone rings are NOT eligible
    # Eligible: player owns stack AND height > 1
    eligible = (stack_owner_np == player) & (stack_height_np > 1)

    # Apply exclusions if any
    if excluded_positions:
        for y, x in excluded_positions:
            if 0 <= y < board_size and 0 <= x < board_size:
                eligible[y, x] = False

    positions = np.argwhere(eligible)

    if len(positions) == 0:
        return None

    # Take first eligible position
    y, x = int(positions[0, 0]), int(positions[0, 1])
    height = int(stack_height_np[y, x])

    return (y, x, height)


def _find_all_regions(
    state: BatchGameState,
    game_idx: int,
) -> List[Set[Tuple[int, int]]]:
    """Find all maximal connected regions of non-collapsed cells (R140).

    Uses BFS to discover all connected regions of non-collapsed cells.
    A region is a maximal set of non-collapsed cells where each cell is connected
    to at least one other cell in the region via 4-connectivity.

    Optimized 2025-12-13: Use deque for O(1) queue ops and numpy for visited tracking.

    Args:
        state: BatchGameState
        game_idx: Game index

    Returns:
        List of regions, where each region is a set of (y, x) positions
    """
    board_size = state.board_size
    g = game_idx

    # Non-collapsed cells are those that are not territory (collapsed spaces)
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()

    # Use numpy array for visited tracking (faster than Python list of lists)
    visited = np.zeros((board_size, board_size), dtype=np.bool_)
    regions = []

    # Pre-compute direction offsets
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for start_y in range(board_size):
        for start_x in range(board_size):
            if visited[start_y, start_x] or not non_collapsed[start_y, start_x]:
                continue

            # BFS to find connected region using deque for O(1) popleft
            region = set()
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()  # O(1) instead of O(n) with list.pop(0)
                region.add((y, x))

                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < board_size and 0 <= nx < board_size:
                        if not visited[ny, nx] and non_collapsed[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if region:
                regions.append(region)

    return regions


def _is_physically_disconnected(
    state: BatchGameState,
    game_idx: int,
    region: Set[Tuple[int, int]],
) -> Tuple[bool, Optional[int]]:
    """Check if a region is physically disconnected per R141.

    A region R is physically disconnected if every path from any cell in R to
    any non-collapsed cell outside R must cross:
    - Collapsed spaces (any color), and/or
    - Board edge (off-board), and/or
    - Markers belonging to exactly ONE player B (the border color)

    Optimized 2025-12-13: Use numpy arrays for membership checks.

    Args:
        state: BatchGameState
        game_idx: Game index
        region: Set of (y, x) positions in the region

    Returns:
        Tuple of (is_disconnected, border_player) where border_player is the
        single player B whose markers form the barrier (or None if not disconnected)
    """
    board_size = state.board_size
    g = game_idx

    # Pre-extract all numpy arrays at once
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()
    marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

    # Create numpy mask for region membership (faster than set for dense checks)
    region_mask = np.zeros((board_size, board_size), dtype=np.bool_)
    for y, x in region:
        region_mask[y, x] = True

    # Find outside_non_collapsed using numpy vectorized operation
    outside_mask = non_collapsed & ~region_mask
    outside_positions = np.argwhere(outside_mask)

    # If no cells outside region, region spans entire non-collapsed board
    if len(outside_positions) == 0:
        return (False, None)

    # Convert to set for O(1) lookup in later checks
    outside_non_collapsed = set(map(tuple, outside_positions))

    # BFS from region boundary
    blocking_marker_players = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    region_boundary = set()
    for y, x in region:
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < board_size and 0 <= nx < board_size:
                if not region_mask[ny, nx]:
                    region_boundary.add((y, x))
                    break
            else:
                # Edge of board - this boundary touches edge
                region_boundary.add((y, x))

    # Check what separates region from outside
    # Use numpy array for visited tracking (faster than set for dense boards)
    visited = np.copy(region_mask)
    queue = list(region_boundary)
    can_reach_outside = False

    for y, x in queue:
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Off-board - counts as barrier
            if not (0 <= ny < board_size and 0 <= nx < board_size):
                continue

            if visited[ny, nx]:
                continue

            # Collapsed space - counts as barrier
            if not non_collapsed[ny, nx]:
                continue

            # Non-collapsed cell outside region
            if outside_mask[ny, nx]:
                # Can we reach it directly, or is there a marker barrier?
                cell_marker_owner = 0
                if marker_owner_np is not None:
                    cell_marker_owner = marker_owner_np[ny, nx]

                if cell_marker_owner > 0:
                    blocking_marker_players.add(int(cell_marker_owner))
                else:
                    # Empty cell or stack - we can reach outside without marker barrier
                    can_reach_outside = True

    # If we can reach outside directly (without crossing markers), not disconnected
    if can_reach_outside and not blocking_marker_players:
        return (False, None)

    # If blocking markers belong to multiple players, not physically disconnected
    if len(blocking_marker_players) > 1:
        return (False, None)

    # If exactly one player's markers form the barrier
    if len(blocking_marker_players) == 1:
        border_player = blocking_marker_players.pop()
        return (True, border_player)

    # Edge case: region is isolated by collapsed spaces and/or board edges only
    # This is physically disconnected with no border player
    # For territory purposes, any player who can claim it may do so
    return (True, None)


def _is_color_disconnected(
    state: BatchGameState,
    game_idx: int,
    region: Set[Tuple[int, int]],
) -> bool:
    """Check if a region is color-disconnected per R142.

    R is color-disconnected if RegionColors is a strict subset of ActiveColors.
    - ActiveColors: players with at least one ring anywhere on the board (any stack)
    - RegionColors: players controlling at least one stack (by top ring) in R

    Empty regions (no stacks) have RegionColors = ∅, which is always a strict
    subset of any non-empty ActiveColors, so they satisfy color-disconnection.

    Optimized 2025-12-13: Use numpy to compute colors without .item() calls.

    Args:
        state: BatchGameState
        game_idx: Game index
        region: Set of (y, x) positions in the region

    Returns:
        True if region is color-disconnected (eligible for processing)
    """
    g = game_idx

    # Use numpy to avoid .item() calls in board scan
    stack_owner_np = state.stack_owner[g].cpu().numpy()
    stack_height_np = state.stack_height[g].cpu().numpy()

    # Compute ActiveColors: unique owners with height > 0 across entire board
    active_mask = (stack_owner_np > 0) & (stack_height_np > 0)
    active_colors = set(stack_owner_np[active_mask].tolist())

    # If no active colors (empty board), no territory processing possible
    if not active_colors:
        return False

    # Compute RegionColors: players controlling stacks in the region
    region_colors = set()
    for y, x in region:
        owner = stack_owner_np[y, x]
        height = stack_height_np[y, x]
        if owner > 0 and height > 0:
            region_colors.add(int(owner))

    # R is color-disconnected if RegionColors ⊂ ActiveColors (strict subset)
    # This means RegionColors != ActiveColors AND RegionColors ⊆ ActiveColors
    # Empty set is always a strict subset of non-empty set

    is_strict_subset = region_colors < active_colors  # Python set comparison
    return is_strict_subset


def compute_territory_batch(
    state: BatchGameState,
    game_mask: Optional[torch.Tensor] = None,
) -> None:
    """Compute and update territory claims (in-place).

    Per RR-CANON-R140-R146:
    - R140: Find all maximal regions of non-collapsed cells
    - R141: Check physical disconnection (all blocking markers belong to ONE player)
    - R142: Check color-disconnection (RegionColors ⊂ ActiveColors)
    - R143: Self-elimination prerequisite (player must have eligible cap outside)
    - R145: Region collapse and elimination (collapse interior + border markers)

    This implementation correctly handles:
    1. Regions divided by collapsed spaces or single-color marker lines
    2. The single-color boundary requirement (R141)
    3. The color-disconnection criterion (R142)

    Cap eligibility is checked per RR-CANON-R145: height-1 standalone rings
    are NOT eligible for territory elimination cost.

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
    """
    batch_size = state.batch_size
    board_size = state.board_size

    if game_mask is None:
        game_mask = state.get_active_mask()

    for g in range(batch_size):
        if not game_mask[g]:
            continue

        # R140: Find all maximal regions of non-collapsed cells
        all_regions = _find_all_regions(state, g)

        # If only one region, no territory processing possible
        # (entire non-collapsed board is connected)
        if len(all_regions) <= 1:
            continue

        # Pre-extract game arrays as numpy to avoid .item() calls in loops
        # (Optimized 2025-12-13)
        stack_height_np = state.stack_height[g].cpu().numpy()
        stack_owner_np = state.stack_owner[g].cpu().numpy()
        is_collapsed_np = state.is_collapsed[g].cpu().numpy()
        marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None

        # Process each region
        # Track which regions have been processed to avoid double-processing
        processed_regions = set()

        # Iterate until no more regions can be processed
        # (processing a region may create new disconnected regions)
        max_iterations = 10  # Safety limit
        for iteration in range(max_iterations):
            found_processable = False

            for region_idx, region in enumerate(all_regions):
                if region_idx in processed_regions:
                    continue

                # R141: Check physical disconnection
                is_disconnected, border_player = _is_physically_disconnected(state, g, region)
                if not is_disconnected:
                    continue

                # R142: Check color-disconnection
                if not _is_color_disconnected(state, g, region):
                    continue

                # Region is both physically and color-disconnected
                # Determine which player can process it

                # For each player who could claim this territory
                for player in range(1, state.num_players + 1):
                    # Get positions in the region
                    region_positions = region

                    # R143: Find eligible cap for elimination (outside region)
                    eligible_cap = _find_eligible_territory_cap(
                        state, g, player, excluded_positions=region_positions
                    )

                    if eligible_cap is None:
                        continue

                    # Player can process this region
                    cap_y, cap_x, cap_height = eligible_cap

                    # R145: Process the region
                    # 1. Collapse interior (all cells in region become territory)
                    territory_count = 0
                    for y, x in region:
                        # Remove any stacks - use numpy for reading
                        sh = int(stack_height_np[y, x])
                        if sh > 0:
                            # Track which player lost the rings - use numpy
                            so = int(stack_owner_np[y, x])
                            if so > 0:
                                state.eliminated_rings[g, so] += sh
                            # Player processing territory CAUSED these eliminations (for victory)
                            state.rings_caused_eliminated[g, player] += sh
                            state.stack_height[g, y, x] = 0
                            state.stack_owner[g, y, x] = 0
                            # Update local numpy copy for consistency
                            stack_height_np[y, x] = 0
                            stack_owner_np[y, x] = 0

                        # Collapse the cell - use numpy for reading
                        if not is_collapsed_np[y, x]:
                            state.is_collapsed[g, y, x] = True
                            state.territory_owner[g, y, x] = player
                            territory_count += 1
                            is_collapsed_np[y, x] = True

                    # 2. Collapse border markers of single border color B (if applicable)
                    if border_player is not None and marker_owner_np is not None:
                        # Convert region to set once for O(1) lookup
                        region_set = set(region) if not isinstance(region, set) else region
                        # Find and collapse border markers - use numpy for reading
                        for y in range(board_size):
                            for x in range(board_size):
                                if marker_owner_np[y, x] == border_player:
                                    # Check if this marker is on the boundary of region
                                    is_boundary = False
                                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        ny, nx = y + dy, x + dx
                                        if 0 <= ny < board_size and 0 <= nx < board_size:
                                            if (ny, nx) in region_set:
                                                is_boundary = True
                                                break

                                    if is_boundary and not is_collapsed_np[y, x]:
                                        state.is_collapsed[g, y, x] = True
                                        state.territory_owner[g, y, x] = player
                                        state.marker_owner[g, y, x] = 0
                                        territory_count += 1
                                        is_collapsed_np[y, x] = True
                                        marker_owner_np[y, x] = 0

                    # 4. Mandatory self-elimination (eliminate cap)
                    state.stack_height[g, cap_y, cap_x] = 0
                    state.stack_owner[g, cap_y, cap_x] = 0
                    stack_height_np[cap_y, cap_x] = 0
                    stack_owner_np[cap_y, cap_x] = 0
                    # Player eliminates own rings for territory cap cost
                    state.eliminated_rings[g, player] += cap_height
                    # Player CAUSED these eliminations (self-elimination counts for victory)
                    state.rings_caused_eliminated[g, player] += cap_height

                    # Update territory count
                    state.territory_count[g, player] += territory_count

                    processed_regions.add(region_idx)
                    found_processable = True
                    break  # Move to next region

                if found_processable:
                    break

            if not found_processable:
                break

            # Recompute regions after processing (new disconnections may appear)
            all_regions = _find_all_regions(state, g)
            processed_regions = set()  # Reset since region indices changed
            # Re-extract numpy arrays since state changed
            stack_height_np = state.stack_height[g].cpu().numpy()
            stack_owner_np = state.stack_owner[g].cpu().numpy()
            is_collapsed_np = state.is_collapsed[g].cpu().numpy()
            marker_owner_np = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None


# =============================================================================
# Batch Heuristic Evaluation
# =============================================================================


def evaluate_positions_batch(
    state: BatchGameState,
    weights: Dict[str, float],
) -> torch.Tensor:
    """Evaluate all positions using comprehensive heuristic scoring.

    Implements all 45 heuristic weights from BASE_V1_BALANCED_WEIGHTS to match
    the CPU HeuristicAI evaluation. Weights are organized into categories:

    Core Position Weights:
    - WEIGHT_STACK_CONTROL: Number of controlled stacks
    - WEIGHT_STACK_HEIGHT: Total ring height on controlled stacks
    - WEIGHT_CAP_HEIGHT: Summed cap height (capture power)
    - WEIGHT_TERRITORY: Territory count
    - WEIGHT_RINGS_IN_HAND: Rings available to place
    - WEIGHT_CENTER_CONTROL: Stacks near board center
    - WEIGHT_ADJACENCY: Adjacency bonuses for stack clusters

    Threat/Defense Weights:
    - WEIGHT_OPPONENT_THREAT: Opponent line threats
    - WEIGHT_MOBILITY: Available movement options
    - WEIGHT_ELIMINATED_RINGS: Rings eliminated by this player
    - WEIGHT_VULNERABILITY: Exposure to capture

    Line/Victory Weights:
    - WEIGHT_LINE_POTENTIAL: 2/3/4-in-a-row patterns
    - WEIGHT_VICTORY_PROXIMITY: Distance to victory threshold
    - WEIGHT_LINE_CONNECTIVITY: Connected line structures

    Advanced Weights (v1.1+):
    - WEIGHT_MARKER_COUNT: Board markers controlled
    - WEIGHT_OVERTAKE_POTENTIAL: Capture opportunities
    - WEIGHT_TERRITORY_CLOSURE: Enclosed territory potential
    - WEIGHT_TERRITORY_SAFETY: Protected territory
    - WEIGHT_STACK_MOBILITY: Per-stack movement freedom
    - WEIGHT_OPPONENT_VICTORY_THREAT: Opponent's progress to victory
    - WEIGHT_FORCED_ELIMINATION_RISK: Risk of forced elimination
    - WEIGHT_LPS_ACTION_ADVANTAGE: Last Player Standing advantage
    - WEIGHT_MULTI_LEADER_THREAT: Multiple opponents with lead

    Penalty/Bonus Weights (v1.1 refactor):
    - WEIGHT_NO_STACKS_PENALTY: Massive penalty for no controlled stacks
    - WEIGHT_SINGLE_STACK_PENALTY: Penalty for single stack vulnerability
    - WEIGHT_STACK_DIVERSITY_BONUS: Spread vs concentration
    - WEIGHT_SAFE_MOVE_BONUS: Bonus for safe moves available
    - WEIGHT_NO_SAFE_MOVES_PENALTY: Penalty for no safe moves
    - WEIGHT_VICTORY_THRESHOLD_BONUS: Near-victory bonus

    Line Pattern Weights:
    - WEIGHT_TWO_IN_ROW, WEIGHT_THREE_IN_ROW, WEIGHT_FOUR_IN_ROW
    - WEIGHT_CONNECTED_NEIGHBOR, WEIGHT_GAP_POTENTIAL
    - WEIGHT_BLOCKED_STACK_PENALTY

    Swap/Opening Weights (v1.2-v1.4):
    - WEIGHT_SWAP_OPENING_CENTER, WEIGHT_SWAP_OPENING_ADJACENCY
    - WEIGHT_SWAP_OPENING_HEIGHT, WEIGHT_SWAP_CORNER_PENALTY
    - WEIGHT_SWAP_EDGE_BONUS, WEIGHT_SWAP_DIAGONAL_BONUS
    - WEIGHT_SWAP_OPENING_STRENGTH, WEIGHT_SWAP_EXPLORATION_TEMPERATURE

    Recovery Weights (v1.5):
    - WEIGHT_RECOVERY_POTENTIAL, WEIGHT_RECOVERY_ELIGIBILITY
    - WEIGHT_BURIED_RING_VALUE, WEIGHT_RECOVERY_THREAT

    Args:
        state: BatchGameState to evaluate
        weights: Heuristic weight dictionary (can use either old 8-weight format
                 or full 45-weight format; missing weights use defaults)

    Returns:
        Tensor of scores (batch_size, num_players) for each player
    """
    device = state.device
    batch_size = state.batch_size
    board_size = state.board_size
    num_players = state.num_players
    center = board_size // 2

    scores = torch.zeros(batch_size, num_players + 1, dtype=torch.float32, device=device)

    # Pre-compute center distance matrix
    y_coords = torch.arange(board_size, device=device).view(-1, 1).expand(board_size, board_size)
    x_coords = torch.arange(board_size, device=device).view(1, -1).expand(board_size, board_size)
    center_dist = ((y_coords - center).abs() + (x_coords - center).abs()).float()
    max_dist = center_dist.max()
    center_bonus = (max_dist - center_dist) / max_dist  # 1.0 at center, 0.0 at corners

    # Canonical victory thresholds (RR-CANON-R061/R062-v2).
    # Keep this in sync with app.rules.core.BOARD_CONFIGS.
    from app.models import BoardType
    from app.rules.core import get_territory_victory_minimum, get_victory_threshold

    board_type_map = {
        8: BoardType.SQUARE8,
        19: BoardType.SQUARE19,
        13: BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(board_size, BoardType.SQUARE8)
    # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
    territory_victory_minimum = get_territory_victory_minimum(board_type, num_players)
    ring_victory_threshold = get_victory_threshold(board_type, num_players)

    # Weight mapping: support both old 8-weight format and new 45-weight format
    def get_weight(new_key: str, old_key: str = None, default: float = 0.0) -> float:
        """Get weight from dict, checking both new and old key formats."""
        if new_key in weights:
            return weights[new_key]
        if old_key and old_key in weights:
            return weights[old_key]
        return default

    for p in range(1, num_players + 1):
        # === CORE POSITION METRICS ===
        player_stacks = (state.stack_owner == p)
        stack_count = player_stacks.sum(dim=(1, 2)).float()

        # Total rings on controlled stacks
        player_heights = state.stack_height * player_stacks.int()
        total_ring_count = player_heights.sum(dim=(1, 2)).float()

        # Cap height (sum of stack heights, reflects capture power)
        cap_height = total_ring_count.clone()  # Same as ring count for now

        # Rings in hand
        rings_in_hand = state.rings_in_hand[:, p].float()

        # Territory count
        territory = state.territory_count[:, p].float()

        # Center control: weighted sum of positions near center
        center_control = (center_bonus.unsqueeze(0) * player_stacks.float()).sum(dim=(1, 2))

        # === STACK HEIGHT METRICS ===
        # Average stack height
        avg_height = total_ring_count / (stack_count + 1e-6)

        # Tall stacks bonus (height 3+)
        tall_stacks = ((state.stack_height >= 3) & player_stacks).sum(dim=(1, 2)).float()

        # === ADJACENCY METRICS ===
        # Count adjacent pairs of controlled stacks (vectorized)
        # Check horizontal adjacency: player_stacks[:, :, :-1] AND player_stacks[:, :, 1:]
        player_stacks_float = player_stacks.float()
        horizontal_adj = (player_stacks_float[:, :, :-1] * player_stacks_float[:, :, 1:]).sum(dim=(1, 2))
        # Check vertical adjacency: player_stacks[:, :-1, :] AND player_stacks[:, 1:, :]
        vertical_adj = (player_stacks_float[:, :-1, :] * player_stacks_float[:, 1:, :]).sum(dim=(1, 2))
        adjacency_score = horizontal_adj + vertical_adj

        # === MARKER METRICS ===
        marker_count = (state.marker_owner == p).sum(dim=(1, 2)).float()

        # === ELIMINATED RINGS METRICS ===
        eliminated_rings = state.eliminated_rings[:, p].float()

        # === BURIED RINGS METRICS ===
        buried_rings = state.buried_rings[:, p].float()

        # === MOBILITY METRICS (simplified) ===
        # Approximate mobility by stack count and territory
        mobility = stack_count * 4.0 + territory * 0.5  # Each stack ~4 moves avg

        # === LINE POTENTIAL METRICS (VECTORIZED) ===
        # Track 2/3/4-in-a-row patterns using tensor operations instead of per-game loops
        # This provides ~10x speedup over the naive O(batch * board^2 * directions) approach

        # HORIZONTAL patterns: check consecutive columns
        # 2-in-row: stack at (y, x) AND stack at (y, x+1)
        h2 = (player_stacks_float[:, :, :-1] * player_stacks_float[:, :, 1:]).sum(dim=(1, 2))
        # 3-in-row: (y, x) AND (y, x+1) AND (y, x+2)
        h3 = (player_stacks_float[:, :, :-2] * player_stacks_float[:, :, 1:-1] * player_stacks_float[:, :, 2:]).sum(dim=(1, 2))
        # 4-in-row
        h4 = (player_stacks_float[:, :, :-3] * player_stacks_float[:, :, 1:-2] * player_stacks_float[:, :, 2:-1] * player_stacks_float[:, :, 3:]).sum(dim=(1, 2))

        # VERTICAL patterns: check consecutive rows
        v2 = (player_stacks_float[:, :-1, :] * player_stacks_float[:, 1:, :]).sum(dim=(1, 2))
        v3 = (player_stacks_float[:, :-2, :] * player_stacks_float[:, 1:-1, :] * player_stacks_float[:, 2:, :]).sum(dim=(1, 2))
        v4 = (player_stacks_float[:, :-3, :] * player_stacks_float[:, 1:-2, :] * player_stacks_float[:, 2:-1, :] * player_stacks_float[:, 3:, :]).sum(dim=(1, 2))

        # DIAGONAL patterns (down-right): check (y,x), (y+1,x+1), etc.
        d1_2 = (player_stacks_float[:, :-1, :-1] * player_stacks_float[:, 1:, 1:]).sum(dim=(1, 2))
        d1_3 = (player_stacks_float[:, :-2, :-2] * player_stacks_float[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, 2:]).sum(dim=(1, 2))
        d1_4 = (player_stacks_float[:, :-3, :-3] * player_stacks_float[:, 1:-2, 1:-2] * player_stacks_float[:, 2:-1, 2:-1] * player_stacks_float[:, 3:, 3:]).sum(dim=(1, 2))

        # ANTI-DIAGONAL patterns (down-left): check (y,x), (y+1,x-1), etc.
        d2_2 = (player_stacks_float[:, :-1, 1:] * player_stacks_float[:, 1:, :-1]).sum(dim=(1, 2))
        d2_3 = (player_stacks_float[:, :-2, 2:] * player_stacks_float[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, :-2]).sum(dim=(1, 2))
        d2_4 = (player_stacks_float[:, :-3, 3:] * player_stacks_float[:, 1:-2, 2:-1] * player_stacks_float[:, 2:-1, 1:-2] * player_stacks_float[:, 3:, :-3]).sum(dim=(1, 2))

        # Sum across all directions
        two_in_row = h2 + v2 + d1_2 + d2_2
        three_in_row = h3 + v3 + d1_3 + d2_3
        four_in_row = h4 + v4 + d1_4 + d2_4

        # Connected neighbors: count of adjacent pairs (same as two_in_row)
        connected_neighbors = two_in_row

        # Gap potential: simplified - check patterns like [stack, empty, stack]
        # Horizontal gaps: stack at x, empty at x+1, stack at x+2
        empty_cells = (state.stack_owner == 0).float()
        h_gap = (player_stacks_float[:, :, :-2] * empty_cells[:, :, 1:-1] * player_stacks_float[:, :, 2:]).sum(dim=(1, 2))
        v_gap = (player_stacks_float[:, :-2, :] * empty_cells[:, 1:-1, :] * player_stacks_float[:, 2:, :]).sum(dim=(1, 2))
        d1_gap = (player_stacks_float[:, :-2, :-2] * empty_cells[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, 2:]).sum(dim=(1, 2))
        d2_gap = (player_stacks_float[:, :-2, 2:] * empty_cells[:, 1:-1, 1:-1] * player_stacks_float[:, 2:, :-2]).sum(dim=(1, 2))
        gap_potential = (h_gap + v_gap + d1_gap + d2_gap) * 0.5

        # === OPPONENT THREAT METRICS (VECTORIZED) ===
        opponent_threat = torch.zeros(batch_size, device=device)
        opponent_victory_threat = torch.zeros(batch_size, device=device)
        blocking_score = torch.zeros(batch_size, device=device)

        for opponent in range(1, num_players + 1):
            if opponent == p:
                continue

            opp_stacks = (state.stack_owner == opponent).float()
            opp_territory = state.territory_count[:, opponent].float()
            opp_eliminated = state.eliminated_rings[:, opponent].float()

            # Victory proximity threat
            opp_territory_progress = opp_territory / territory_victory_minimum
            opp_elim_progress = opp_eliminated / ring_victory_threshold
            opponent_victory_threat += torch.max(opp_territory_progress, opp_elim_progress)

            # Vectorized line threat detection (same as line potential but for opponent)
            # HORIZONTAL opponent lines
            opp_h2 = (opp_stacks[:, :, :-1] * opp_stacks[:, :, 1:]).sum(dim=(1, 2))
            opp_h3 = (opp_stacks[:, :, :-2] * opp_stacks[:, :, 1:-1] * opp_stacks[:, :, 2:]).sum(dim=(1, 2))
            opp_h4 = (opp_stacks[:, :, :-3] * opp_stacks[:, :, 1:-2] * opp_stacks[:, :, 2:-1] * opp_stacks[:, :, 3:]).sum(dim=(1, 2))

            # VERTICAL opponent lines
            opp_v2 = (opp_stacks[:, :-1, :] * opp_stacks[:, 1:, :]).sum(dim=(1, 2))
            opp_v3 = (opp_stacks[:, :-2, :] * opp_stacks[:, 1:-1, :] * opp_stacks[:, 2:, :]).sum(dim=(1, 2))
            opp_v4 = (opp_stacks[:, :-3, :] * opp_stacks[:, 1:-2, :] * opp_stacks[:, 2:-1, :] * opp_stacks[:, 3:, :]).sum(dim=(1, 2))

            # DIAGONAL opponent lines (down-right)
            opp_d1_2 = (opp_stacks[:, :-1, :-1] * opp_stacks[:, 1:, 1:]).sum(dim=(1, 2))
            opp_d1_3 = (opp_stacks[:, :-2, :-2] * opp_stacks[:, 1:-1, 1:-1] * opp_stacks[:, 2:, 2:]).sum(dim=(1, 2))
            opp_d1_4 = (opp_stacks[:, :-3, :-3] * opp_stacks[:, 1:-2, 1:-2] * opp_stacks[:, 2:-1, 2:-1] * opp_stacks[:, 3:, 3:]).sum(dim=(1, 2))

            # ANTI-DIAGONAL opponent lines (down-left)
            opp_d2_2 = (opp_stacks[:, :-1, 1:] * opp_stacks[:, 1:, :-1]).sum(dim=(1, 2))
            opp_d2_3 = (opp_stacks[:, :-2, 2:] * opp_stacks[:, 1:-1, 1:-1] * opp_stacks[:, 2:, :-2]).sum(dim=(1, 2))
            opp_d2_4 = (opp_stacks[:, :-3, 3:] * opp_stacks[:, 1:-2, 2:-1] * opp_stacks[:, 2:-1, 1:-2] * opp_stacks[:, 3:, :-3]).sum(dim=(1, 2))

            # Weight threats by line length (longer = more dangerous)
            opp_two = opp_h2 + opp_v2 + opp_d1_2 + opp_d2_2
            opp_three = opp_h3 + opp_v3 + opp_d1_3 + opp_d2_3
            opp_four = opp_h4 + opp_v4 + opp_d1_4 + opp_d2_4
            opponent_threat += opp_two * 1.0 + opp_three * 1.5 + opp_four * 2.0

            # Blocking score: count our stacks adjacent to opponent stacks
            # Horizontal blocking
            block_h = (player_stacks_float[:, :, :-1] * opp_stacks[:, :, 1:]).sum(dim=(1, 2))
            block_h += (player_stacks_float[:, :, 1:] * opp_stacks[:, :, :-1]).sum(dim=(1, 2))
            # Vertical blocking
            block_v = (player_stacks_float[:, :-1, :] * opp_stacks[:, 1:, :]).sum(dim=(1, 2))
            block_v += (player_stacks_float[:, 1:, :] * opp_stacks[:, :-1, :]).sum(dim=(1, 2))
            blocking_score += block_h + block_v

        # === VULNERABILITY METRICS (VECTORIZED) ===
        # Check how many of our stacks could be captured by taller adjacent opponent stacks
        # Create padded versions for neighbor checking (pad with 0s)
        heights = state.stack_height.float()
        owners = state.stack_owner

        # Opponent mask: owned by non-zero and not by player p
        opponent_mask = (owners != 0) & (owners != p)
        opponent_heights = heights * opponent_mask.float()

        # For each of 4 directions, check if opponent neighbor is >= our height
        # Pad player_stacks and opponent_heights to handle boundary
        ps_pad = torch.nn.functional.pad(player_stacks_float, (1, 1, 1, 1), value=0)
        oh_pad = torch.nn.functional.pad(opponent_heights, (1, 1, 1, 1), value=0)
        h_pad = torch.nn.functional.pad(heights, (1, 1, 1, 1), value=0)
        om_pad = torch.nn.functional.pad(opponent_mask.float(), (1, 1, 1, 1), value=0)
        own_pad = torch.nn.functional.pad((owners != 0).float(), (1, 1, 1, 1), value=0)

        # Our stack positions (in padded coords, offset by 1)
        # Check each direction: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
        # For vulnerability: opponent neighbor height >= our height at (y,x)
        vuln_up = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, :-2, 1:-1] *
                   (oh_pad[:, :-2, 1:-1] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_down = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 2:, 1:-1] *
                     (oh_pad[:, 2:, 1:-1] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_left = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, :-2] *
                     (oh_pad[:, 1:-1, :-2] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vuln_right = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, 2:] *
                      (oh_pad[:, 1:-1, 2:] >= h_pad[:, 1:-1, 1:-1]).float()).sum(dim=(1, 2))
        vulnerability = vuln_up + vuln_down + vuln_left + vuln_right

        # Blocked stacks: count adjacent occupied cells per stack, blocked if >= 3
        adj_up = ps_pad[:, 1:-1, 1:-1] * own_pad[:, :-2, 1:-1]
        adj_down = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 2:, 1:-1]
        adj_left = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 1:-1, :-2]
        adj_right = ps_pad[:, 1:-1, 1:-1] * own_pad[:, 1:-1, 2:]
        adj_count = adj_up + adj_down + adj_left + adj_right
        blocked_stacks = (adj_count >= 3).float().sum(dim=(1, 2))

        # === OVERTAKE POTENTIAL (VECTORIZED) ===
        # Count opponent stacks we could capture (our taller stacks adjacent to shorter opponent)
        # Our height > opponent neighbor height
        over_up = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, :-2, 1:-1] *
                   (h_pad[:, 1:-1, 1:-1] > oh_pad[:, :-2, 1:-1]).float()).sum(dim=(1, 2))
        over_down = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 2:, 1:-1] *
                     (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 2:, 1:-1]).float()).sum(dim=(1, 2))
        over_left = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, :-2] *
                     (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 1:-1, :-2]).float()).sum(dim=(1, 2))
        over_right = (ps_pad[:, 1:-1, 1:-1] * om_pad[:, 1:-1, 2:] *
                      (h_pad[:, 1:-1, 1:-1] > oh_pad[:, 1:-1, 2:]).float()).sum(dim=(1, 2))
        overtake_potential = over_up + over_down + over_left + over_right

        # === TERRITORY METRICS (VECTORIZED) ===
        # Territory closure: territory cells adjacent to our stacks
        player_territory = (state.territory_owner == p).float()
        pt_pad = torch.nn.functional.pad(player_territory, (1, 1, 1, 1), value=0)

        # For each territory cell, check if any adjacent cell has our stack
        # Neighbor our-stack adjacency
        terr_adj_up = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, :-2, 1:-1]
        terr_adj_down = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 2:, 1:-1]
        terr_adj_left = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 1:-1, :-2]
        terr_adj_right = pt_pad[:, 1:-1, 1:-1] * ps_pad[:, 1:-1, 2:]
        territory_adj = terr_adj_up + terr_adj_down + terr_adj_left + terr_adj_right
        territory_closure = territory_adj.sum(dim=(1, 2)) * 0.5
        territory_safety = territory_closure  # Same metric for now

        # === STACK MOBILITY ===
        # Per-stack movement freedom (simplified)
        stack_mobility = stack_count * 3.0  # Avg 3 directions per stack

        # === VICTORY PROXIMITY ===
        # How close to winning (normalized 0-1)
        territory_progress = territory / territory_victory_minimum
        elim_progress = eliminated_rings / ring_victory_threshold
        victory_proximity = torch.max(territory_progress, elim_progress)

        # === FORCED ELIMINATION RISK ===
        # Risk of being forced to eliminate (few stacks, surrounded)
        forced_elim_risk = torch.where(
            stack_count <= 2,
            vulnerability * 2.0,
            vulnerability * 0.5
        )

        # === LPS ACTION ADVANTAGE ===
        # Bonus for having more moves available than opponents
        lps_advantage = mobility / (mobility.sum() / num_players + 1e-6)

        # === MULTI-LEADER THREAT ===
        # Multiple opponents ahead
        multi_leader = opponent_victory_threat / (num_players - 1 + 1e-6)

        # === RECOVERY METRICS ===
        # Per RR-CANON-R110: eligible iff controls no stacks, has a marker,
        # and has at least one buried ring. Rings in hand do not affect eligibility.
        has_buried = buried_rings > 0
        no_controlled = stack_count == 0
        has_markers = marker_count > 0
        recovery_eligible = has_buried & no_controlled & has_markers
        recovery_potential = buried_rings * recovery_eligible.float()

        # === PENALTY/BONUS FLAGS ===
        no_stacks_flag = (stack_count == 0).float()
        single_stack_flag = (stack_count == 1).float()
        stack_diversity = (stack_count - adjacency_score / 2).clamp(min=0)  # Spread bonus
        near_victory = (victory_proximity > 0.8).float()

        # === COMBINE ALL COMPONENTS ===
        # Core position weights
        score = torch.zeros(batch_size, device=device)
        score += stack_count * get_weight("WEIGHT_STACK_CONTROL", "material_weight", 9.39)
        score += total_ring_count * get_weight("WEIGHT_STACK_HEIGHT", "ring_count_weight", 6.81)
        score += cap_height * get_weight("WEIGHT_CAP_HEIGHT", None, 4.82)
        score += territory * get_weight("WEIGHT_TERRITORY", "territory_weight", 8.66)
        score += rings_in_hand * get_weight("WEIGHT_RINGS_IN_HAND", None, 5.17)
        score += center_control * get_weight("WEIGHT_CENTER_CONTROL", "center_control_weight", 2.28)
        score += adjacency_score * get_weight("WEIGHT_ADJACENCY", None, 1.57)

        # Threat/defense weights
        score -= opponent_threat * get_weight("WEIGHT_OPPONENT_THREAT", None, 6.11)
        score += mobility * get_weight("WEIGHT_MOBILITY", "mobility_weight", 5.31)
        score += eliminated_rings * get_weight("WEIGHT_ELIMINATED_RINGS", None, 13.12)
        score -= vulnerability * get_weight("WEIGHT_VULNERABILITY", None, 9.32)

        # Line/victory weights
        line_potential = (
            two_in_row * get_weight("WEIGHT_TWO_IN_ROW", None, 4.25) +
            three_in_row * get_weight("WEIGHT_THREE_IN_ROW", None, 2.13) +
            four_in_row * get_weight("WEIGHT_FOUR_IN_ROW", None, 4.36)
        )
        score += line_potential * get_weight("WEIGHT_LINE_POTENTIAL", "line_potential_weight", 7.24)
        score += victory_proximity * get_weight("WEIGHT_VICTORY_PROXIMITY", None, 20.94)
        score += connected_neighbors * get_weight("WEIGHT_CONNECTED_NEIGHBOR", None, 2.21)
        score += gap_potential * get_weight("WEIGHT_GAP_POTENTIAL", None, 0.03)

        # Advanced weights
        score += marker_count * get_weight("WEIGHT_MARKER_COUNT", None, 3.76)
        score += overtake_potential * get_weight("WEIGHT_OVERTAKE_POTENTIAL", None, 5.96)
        score += territory_closure * get_weight("WEIGHT_TERRITORY_CLOSURE", None, 11.56)
        score += connected_neighbors * get_weight("WEIGHT_LINE_CONNECTIVITY", None, 5.65)
        score += territory_safety * get_weight("WEIGHT_TERRITORY_SAFETY", None, 2.83)
        score += stack_mobility * get_weight("WEIGHT_STACK_MOBILITY", None, 1.11)

        # Opponent threat weights
        score -= opponent_victory_threat * get_weight("WEIGHT_OPPONENT_VICTORY_THREAT", None, 5.21)
        score -= forced_elim_risk * get_weight("WEIGHT_FORCED_ELIMINATION_RISK", None, 2.89)
        score += lps_advantage * get_weight("WEIGHT_LPS_ACTION_ADVANTAGE", None, 0.99)
        score -= multi_leader * get_weight("WEIGHT_MULTI_LEADER_THREAT", None, 1.03)

        # Penalty/bonus weights
        score -= no_stacks_flag * get_weight("WEIGHT_NO_STACKS_PENALTY", None, 51.02)
        score -= single_stack_flag * get_weight("WEIGHT_SINGLE_STACK_PENALTY", None, 10.53)
        score += stack_diversity * get_weight("WEIGHT_STACK_DIVERSITY_BONUS", None, -0.74)
        score -= blocked_stacks * get_weight("WEIGHT_BLOCKED_STACK_PENALTY", None, 4.57)
        score += near_victory * get_weight("WEIGHT_VICTORY_THRESHOLD_BONUS", None, 998.52)

        # Defensive bonus (backward compat)
        score += blocking_score * get_weight("defensive_weight", None, 0.3)

        # Recovery weights (v1.5)
        score += recovery_potential * get_weight("WEIGHT_RECOVERY_POTENTIAL", None, 6.0)
        score += recovery_eligible.float() * get_weight("WEIGHT_RECOVERY_ELIGIBILITY", None, 8.0)
        score += buried_rings * get_weight("WEIGHT_BURIED_RING_VALUE", None, 3.0)

        # === ELIMINATION PENALTY ===
        # Player with no stacks, no rings in hand, and no buried rings is eliminated
        has_material = (stack_count > 0) | (rings_in_hand > 0) | (buried_rings > 0)
        score = torch.where(
            ~has_material,
            score - 10000.0,  # Massive penalty for permanent elimination
            score
        )

        scores[:, p] = score

    return scores


# =============================================================================
# Parallel Game Runner
# =============================================================================


class ParallelGameRunner:
    """GPU-accelerated parallel game simulation.

    Runs multiple games simultaneously using batch operations on GPU.
    Supports different AI configurations per game for CMA-ES evaluation.

    Example:
        runner = ParallelGameRunner(batch_size=64, device="cuda")

        # Run games with specific heuristic weights
        results = runner.run_games(
            weights_per_game=[weights1, weights2, ...],  # length 64
            max_moves=500,
        )

        # Results contain win/loss/draw for each game
    """

    def __init__(
        self,
        batch_size: int = 64,
        board_size: int = 8,
        num_players: int = 2,
        device: Optional[torch.device] = None,
        shadow_validation: bool = False,
        shadow_sample_rate: float = 0.05,
        shadow_threshold: float = 0.001,
        state_validation: bool = False,
        state_sample_rate: float = 0.01,
        state_threshold: float = 0.001,
        swap_enabled: bool = False,
        lps_victory_rounds: Optional[int] = None,
        rings_per_player: Optional[int] = None,
        board_type: Optional[str] = None,
        use_heuristic_selection: bool = False,
        weight_noise: float = 0.0,
    ):
        """Initialize parallel game runner.

        Args:
            batch_size: Number of games to run in parallel
            board_size: Board dimension
            num_players: Number of players per game
            device: GPU device (auto-detected if None)
            shadow_validation: Enable shadow validation against CPU rules (move generation)
            shadow_sample_rate: Fraction of moves to validate (0.0-1.0)
            shadow_threshold: Maximum divergence rate before halt
            state_validation: Enable CPU oracle mode (state validation)
            state_sample_rate: Fraction of states to validate (0.0-1.0)
            state_threshold: Maximum state divergence rate before halt
            swap_enabled: Enable pie rule (swap_sides) for 2-player games (RR-CANON R180-R184)
            lps_victory_rounds: Number of consecutive rounds one player must have exclusive
                               real actions to win via LPS (None = board default, respects env vars)
            rings_per_player: Starting rings per player (None = board default, respects env vars)
            board_type: Board type string ("square8", "square19", "hexagonal") for proper
                       initialization. If "hexagonal", marks out-of-bounds cells as collapsed.
            use_heuristic_selection: When True, use heuristic-based move selection instead of
                       center-bias. Provides better move quality but slightly slower.
            weight_noise: Multiplicative noise factor (0.0-1.0) for heuristic weights.
                       Each weight is multiplied by a random factor in [1-noise, 1+noise].
                       This increases training diversity by making each game use slightly
                       different evaluation. Default 0.0 (no noise).
        """
        self.batch_size = batch_size
        self.board_size = board_size
        self.num_players = num_players
        self.swap_enabled = swap_enabled
        self.board_type = board_type
        self.use_heuristic_selection = use_heuristic_selection
        self.weight_noise = weight_noise
        self.use_policy_selection = False
        self.policy_model: Optional["RingRiftNNUEWithPolicy"] = None
        # Default LPS victory rounds to 3 if not specified
        self.lps_victory_rounds = lps_victory_rounds if lps_victory_rounds is not None else 3
        self.rings_per_player = rings_per_player

        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Pre-allocate state buffer
        self.state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=self.device,
            rings_per_player=rings_per_player,
            board_type=board_type,
            lps_rounds_required=self.lps_victory_rounds,
        )

        # Shadow validation for GPU/CPU parity checking (Phase 2 - move generation)
        self.shadow_validator: Optional[ShadowValidator] = None
        if shadow_validation:
            self.shadow_validator = create_shadow_validator(
                sample_rate=shadow_sample_rate,
                threshold=shadow_threshold,
                enabled=True,
            )
            logger.info(
                f"Shadow validation enabled: sample_rate={shadow_sample_rate}, "
                f"threshold={shadow_threshold}"
            )

        # State validation for CPU oracle mode (A1 - state parity)
        self.state_validator: Optional[StateValidator] = None
        if state_validation:
            self.state_validator = create_state_validator(
                sample_rate=state_sample_rate,
                threshold=state_threshold,
                enabled=True,
            )
            logger.info(
                f"State validation (CPU oracle) enabled: sample_rate={state_sample_rate}, "
                f"threshold={state_threshold}"
            )

        # Statistics
        self._games_completed = 0
        self._total_moves = 0
        self._total_time = 0.0

        logger.info(
            f"ParallelGameRunner initialized: {batch_size} games, "
            f"{board_size}x{board_size} board, {num_players} players, "
            f"device={self.device}"
        )

    def reset_games(self) -> None:
        """Reset all games to initial state."""
        self.state = BatchGameState.create_batch(
            batch_size=self.batch_size,
            board_size=self.board_size,
            num_players=self.num_players,
            device=self.device,
            rings_per_player=self.rings_per_player,
            board_type=self.board_type,
        )

    def load_policy_model(self, model_path: Optional[str] = None) -> bool:
        """Load policy model for policy-based move selection.

        Args:
            model_path: Path to policy model checkpoint. If None, uses default.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            from .nnue_policy import RingRiftNNUEWithPolicy
            from ..models import BoardType

            if model_path is None:
                board_type_str = self.board_type or "square8"
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..",
                    "models", "nnue", f"nnue_policy_{board_type_str}_{self.num_players}p.pt"
                )
                model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                logger.debug(f"Policy model not found at {model_path}")
                return False

            # Load checkpoint (weights_only=False for our trusted checkpoints with numpy scalars)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            hidden_dim = checkpoint.get("hidden_dim", 256)
            num_hidden_layers = checkpoint.get("num_hidden_layers", 2)

            # Determine board type
            board_type_str = self.board_type or "square8"
            board_type = BoardType(board_type_str)

            self.policy_model = RingRiftNNUEWithPolicy(
                board_type=board_type,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )
            self.policy_model.load_state_dict(checkpoint["model_state_dict"])
            self.policy_model.to(self.device)
            self.policy_model.eval()
            self.use_policy_selection = True

            logger.info(f"ParallelGameRunner: Loaded policy model from {model_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load policy model: {e}")
            return False

    def _select_moves(
        self,
        moves: "BatchMoves",
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select moves using configured selection strategy.

        Selection priority:
        1. Policy-based selection (if policy model loaded)
        2. Heuristic-based selection (if use_heuristic_selection=True)
        3. Fast center-bias selection (default)
        """
        if self.use_policy_selection and self.policy_model is not None:
            return self._select_moves_policy(moves, active_mask)
        elif self.use_heuristic_selection:
            return select_moves_heuristic(
                moves, self.state, active_mask, temperature=1.0
            )
        else:
            return select_moves_vectorized(
                moves, active_mask, self.board_size, temperature=1.0
            )

    def _select_moves_policy(
        self,
        moves: "BatchMoves",
        active_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Select moves using policy network scores (batched).

        Uses vectorized operations for efficient batch inference:
        1. Extract features for all active games in parallel
        2. Run single batch forward pass through policy model
        3. Score and sample all moves using vectorized operations

        Falls back to center-bias if policy evaluation fails.
        """
        device = moves.device
        batch_size = active_mask.shape[0]

        # Initialize output: -1 for games with no moves
        selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

        if moves.total_moves == 0 or self.policy_model is None:
            return selected

        try:
            from ..models import BoardType
            from .nnue import get_feature_dim

            board_type_str = self.board_type or "square8"
            board_type = BoardType(board_type_str)

            # Get active game indices that have moves
            active_indices = torch.where(active_mask)[0]
            moves_per_game = moves.moves_per_game[active_indices]
            games_with_moves = active_indices[moves_per_game > 0]

            if len(games_with_moves) == 0:
                return selected

            # === Batched Feature Extraction ===
            # Extract features for all active games in one pass using vectorized ops
            feature_dim = get_feature_dim(board_type)
            num_active = len(games_with_moves)
            features_batch = self._extract_features_batched(
                games_with_moves, board_type, feature_dim, device
            )

            if features_batch is None:
                raise RuntimeError("Batched feature extraction failed")

            # === Batched Policy Inference ===
            with torch.no_grad():
                _, from_logits_batch, to_logits_batch = self.policy_model(
                    features_batch, return_policy=True
                )
                # from_logits_batch: (num_active, H*W)
                # to_logits_batch: (num_active, H*W)

            # === Vectorized Move Scoring ===
            # Score all moves across all games in parallel
            center = self.board_size // 2
            center_idx = center * self.board_size + center
            num_positions = self.board_size * self.board_size

            for local_idx, g_idx in enumerate(games_with_moves):
                g = g_idx.item()
                move_start = moves.move_offsets[g].item()
                move_count = moves.moves_per_game[g].item()

                if move_count == 0:
                    continue

                # Get from/to positions for all moves of this game
                from_y = moves.from_y[move_start:move_start + move_count]
                from_x = moves.from_x[move_start:move_start + move_count]
                to_y = moves.to_y[move_start:move_start + move_count]
                to_x = moves.to_x[move_start:move_start + move_count]

                # Compute flat indices vectorized (use center for negative coords)
                from_idx = torch.where(
                    from_y >= 0,
                    from_y * self.board_size + from_x,
                    torch.full_like(from_y, center_idx)
                ).long()
                to_idx = torch.where(
                    to_y >= 0,
                    to_y * self.board_size + to_x,
                    torch.full_like(to_y, center_idx)
                ).long()

                # Clamp indices to valid range
                from_idx = from_idx.clamp(0, num_positions - 1)
                to_idx = to_idx.clamp(0, num_positions - 1)

                # Get logits for this game
                from_logits = from_logits_batch[local_idx]
                to_logits = to_logits_batch[local_idx]

                # Compute move scores vectorized
                move_scores = from_logits[from_idx] + to_logits[to_idx]

                # Sample move with temperature
                probs = torch.softmax(move_scores / temperature, dim=0)
                selected_local = torch.multinomial(probs, 1).item()
                selected[g] = selected_local

        except Exception as e:
            logger.debug(f"Policy selection failed, falling back to center-bias: {e}")
            return select_moves_vectorized(
                moves, active_mask, self.board_size, temperature=temperature
            )

        return selected

    def _extract_features_batched(
        self,
        game_indices: torch.Tensor,
        board_type: "BoardType",
        feature_dim: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Extract NNUE features for multiple games using vectorized operations.

        This is much more efficient than the per-game extraction as it uses
        batch tensor operations instead of Python loops.

        Args:
            game_indices: Tensor of game indices to extract features for
            board_type: Board type enum
            feature_dim: Expected feature dimension
            device: Target device

        Returns:
            Tensor of shape (num_games, feature_dim) or None on failure
        """
        try:
            num_games = len(game_indices)
            board_size = self.board_size
            num_positions = board_size * board_size

            # Initialize features tensor
            features = torch.zeros(
                (num_games, feature_dim), dtype=torch.float32, device=device
            )

            # Get state tensors for selected games
            current_player = self.state.current_player[game_indices]  # (N,)
            current_player = torch.where(
                current_player < 1,
                torch.ones_like(current_player),
                current_player
            )

            # Extract game state slices: (N, H, W)
            stack_owner = self.state.stack_owner[game_indices]
            stack_height = self.state.stack_height[game_indices]
            territory_owner = self.state.territory_owner[game_indices]

            # Create position indices (flat)
            y_coords = torch.arange(board_size, device=device).view(-1, 1).expand(
                board_size, board_size
            )
            x_coords = torch.arange(board_size, device=device).view(1, -1).expand(
                board_size, board_size
            )
            pos_indices = (y_coords * board_size + x_coords).flatten()  # (H*W,)

            # Process each player's perspective
            # For each game g and position (y,x), we compute:
            #   plane_offset = (owner - current_player[g]) % 4
            # Then set appropriate feature planes

            for g_local in range(num_games):
                cp = current_player[g_local].item()

                # Ring and stack features
                owner_slice = stack_owner[g_local].flatten()  # (H*W,)
                height_slice = stack_height[g_local].flatten()  # (H*W,)

                # Find occupied positions
                occupied = (owner_slice > 0) & (height_slice > 0)
                occupied_idx = torch.where(occupied)[0]

                if len(occupied_idx) > 0:
                    owners = owner_slice[occupied_idx]
                    heights = height_slice[occupied_idx]
                    positions = pos_indices[occupied_idx]

                    # Compute plane offsets (rotate perspective)
                    plane_offsets = ((owners - cp) % self.num_players).long()

                    # Set ring features (planes 0-3)
                    ring_indices = plane_offsets * num_positions + positions
                    valid_ring = ring_indices < feature_dim
                    features[g_local].scatter_(
                        0, ring_indices[valid_ring], torch.ones_like(ring_indices[valid_ring], dtype=torch.float32)
                    )

                    # Set stack height features (planes 4-7)
                    stack_indices = (4 + plane_offsets) * num_positions + positions
                    valid_stack = stack_indices < feature_dim
                    heights_scaled = torch.clamp(heights.float() / 5.0, 0.0, 1.0)
                    features[g_local].scatter_(
                        0, stack_indices[valid_stack], heights_scaled[valid_stack]
                    )

                # Territory features (planes 8-11)
                territory_slice = territory_owner[g_local].flatten()
                territory_occupied = territory_slice > 0
                territory_idx = torch.where(territory_occupied)[0]

                if len(territory_idx) > 0:
                    territory_owners = territory_slice[territory_idx]
                    territory_positions = pos_indices[territory_idx]

                    territory_offsets = ((territory_owners - cp) % self.num_players).long()
                    territory_plane_indices = (8 + territory_offsets) * num_positions + territory_positions
                    valid_territory = territory_plane_indices < feature_dim
                    features[g_local].scatter_(
                        0,
                        territory_plane_indices[valid_territory],
                        torch.ones_like(
                            territory_plane_indices[valid_territory],
                            dtype=torch.float32,
                        ),
                    )

            return features

        except Exception as e:
            logger.debug(f"Batched feature extraction failed: {e}")
            return None

    def _extract_features_for_game(
        self,
        game_idx: int,
        board_type: "BoardType",
    ) -> Optional["np.ndarray"]:
        """Extract NNUE features from batch state for a single game.

        This is a simplified implementation that extracts features game-by-game.
        A more efficient implementation would batch this extraction.
        """
        try:
            import numpy as np
            from .nnue import get_feature_dim

            feature_dim = get_feature_dim(board_type)
            features = np.zeros(feature_dim, dtype=np.float32)
            board_size = self.board_size
            num_positions = board_size * board_size

            current_player = self.state.current_player[game_idx].item()
            if current_player < 1:
                current_player = 1

            # Extract stack ownership for each player (simplified)
            # Planes 0-3: Ring presence, 4-7: Stack presence, 8-11: Territory
            for y in range(board_size):
                for x in range(board_size):
                    pos_idx = y * board_size + x
                    owner = self.state.stack_owner[game_idx, y, x].item()
                    height = self.state.stack_height[game_idx, y, x].item()

                    if owner > 0 and height > 0:
                        # Rotate perspective so current player is always plane 0
                        plane_offset = ((owner - current_player) % self.num_players)

                        # Set ring and stack features
                        ring_plane = plane_offset * num_positions + pos_idx
                        stack_plane = (4 + plane_offset) * num_positions + pos_idx

                        if ring_plane < feature_dim:
                            features[ring_plane] = 1.0
                        if stack_plane < feature_dim:
                            features[stack_plane] = min(float(height) / 5.0, 1.0)

                    # Territory
                    territory_owner = self.state.territory_owner[game_idx, y, x].item()
                    if territory_owner > 0:
                        plane_offset = ((territory_owner - current_player) % self.num_players)
                        territory_plane = (8 + plane_offset) * num_positions + pos_idx
                        if territory_plane < feature_dim:
                            features[territory_plane] = 1.0

            return features

        except Exception as e:
            logger.debug(f"Feature extraction failed for game {game_idx}: {e}")
            return None

    @torch.no_grad()
    def run_games(
        self,
        weights_list: Optional[List[Dict[str, float]]] = None,
        max_moves: int = 500,
        callback: Optional[Callable[[int, BatchGameState], None]] = None,
    ) -> Dict[str, Any]:
        """Run all games to completion.

        Args:
            weights_list: List of weight dicts (one per game) or None for default
            max_moves: Maximum moves before declaring draw
            callback: Optional callback(move_num, state) after each batch move

        Returns:
            Dictionary with:
                - winners: List of winner player numbers (0 for draw)
                - move_counts: List of move counts per game
                - game_lengths: List of game durations
        """
        self.reset_games()
        start_time = time.perf_counter()

        # Use default weights if not provided (with optional noise for diversity)
        if weights_list is None:
            weights_list = self._generate_weights_list()

        # In this runner, a single call to ``_step_games`` advances one phase
        # (ring_placement, movement, line_processing, territory_processing, end_turn).
        # ``max_moves`` is interpreted as a per-game *move record* limit (i.e.
        # ``BatchGameState.move_count``), not a phase-step limit. To prevent
        # pathological stalls where ``move_count`` does not advance, we also
        # apply a conservative safety cap on phase steps.
        phase_steps = 0
        max_phase_steps = max_moves * 20

        while self.state.count_active() > 0 and phase_steps < max_phase_steps:
            active_mask = self.state.get_active_mask()

            # Enforce per-game move limit based on the recorded move_count.
            reached_limit = active_mask & (self.state.move_count >= max_moves)
            if reached_limit.any():
                self.state.game_status[reached_limit] = GameStatus.MAX_MOVES

            if self.state.count_active() == 0:
                break

            # Generate and apply moves for all remaining active games.
            self._step_games(weights_list)

            phase_steps += 1
            if callback:
                callback(phase_steps, self.state)

            # Check for game completion.
            self._check_victory_conditions()

        # Mark any remaining active games as max-moves timeouts.
        active_mask = self.state.get_active_mask()
        self.state.game_status[active_mask] = GameStatus.MAX_MOVES

        elapsed = time.perf_counter() - start_time
        self._games_completed += self.batch_size
        self._total_moves += self.state.move_count.sum().item()
        self._total_time += elapsed

        # Extract move histories and victory types for each game
        move_histories = []
        victory_types = []
        stalemate_tiebreakers = []

        for g in range(self.batch_size):
            move_histories.append(self.state.extract_move_history(g))
            vtype, tiebreaker = self.state.derive_victory_type(g, max_moves)
            victory_types.append(vtype)
            stalemate_tiebreakers.append(tiebreaker)

        # Build results
        results = {
            "winners": self.state.winner.cpu().tolist(),
            "move_counts": self.state.move_count.cpu().tolist(),
            "status": self.state.game_status.cpu().tolist(),
            "move_histories": move_histories,
            "victory_types": victory_types,
            "stalemate_tiebreakers": stalemate_tiebreakers,
            "elapsed_seconds": elapsed,
            "games_per_second": self.batch_size / elapsed,
        }

        # Add validation reports if enabled
        if self.shadow_validator:
            results["shadow_validation"] = self.shadow_validator.get_report()
        if self.state_validator:
            results["state_validation"] = self.state_validator.get_report()

        return results

    def get_validation_reports(self) -> Dict[str, Any]:
        """Get validation reports from both shadow and state validators.

        Returns:
            Dictionary with validation reports and combined status.
        """
        reports = {}

        if self.shadow_validator:
            reports["shadow_validation"] = self.shadow_validator.get_report()

        if self.state_validator:
            reports["state_validation"] = self.state_validator.get_report()

        # Compute combined status
        all_passed = True
        if self.shadow_validator and self.shadow_validator.stats.divergence_rate > self.shadow_validator.threshold:
            all_passed = False
        if self.state_validator and self.state_validator.stats.divergence_rate > self.state_validator.threshold:
            all_passed = False

        reports["combined_status"] = "PASS" if all_passed else "FAIL"
        return reports

    def reset_validation_stats(self) -> None:
        """Reset all validation statistics."""
        if self.shadow_validator:
            self.shadow_validator.reset_stats()
        if self.state_validator:
            self.state_validator.reset_stats()

    def _step_games(self, weights_list: List[Dict[str, float]]) -> None:
        """Execute one phase step for all active games using full rules FSM.

        Phase flow per turn (per RR-CANON):
        RING_PLACEMENT -> MOVEMENT -> LINE_PROCESSING -> TERRITORY_PROCESSING -> END_TURN

        Each call to _step_games processes ONE phase for all active games,
        then advances to the next phase.
        """
        active_mask = self.state.get_active_mask()

        if not active_mask.any():
            return

        # Snapshot phases at the start of the step so a single call processes
        # at most one phase per game. (If we re-read current_phase after each
        # handler, games that advance phases could be processed multiple times
        # per call, which breaks RR-CANON-R172 round-based LPS timing and makes
        # tests non-deterministic.)
        phase_snapshot = self.state.current_phase.clone()

        device = self.device
        batch_size = self.batch_size

        # Process each phase type separately based on current phase
        # Games may be in different phases, so we handle each group

        # PHASE: RING_PLACEMENT (0)
        placement_mask = active_mask & (phase_snapshot == GamePhase.RING_PLACEMENT)
        if placement_mask.any():
            self._step_placement_phase(placement_mask, weights_list)

        # PHASE: MOVEMENT (1)
        movement_mask = active_mask & (phase_snapshot == GamePhase.MOVEMENT)
        if movement_mask.any():
            self._step_movement_phase(movement_mask, weights_list)

        # PHASE: LINE_PROCESSING (2)
        line_mask = active_mask & (phase_snapshot == GamePhase.LINE_PROCESSING)
        if line_mask.any():
            self._step_line_phase(line_mask)

        # PHASE: TERRITORY_PROCESSING (3)
        territory_mask = active_mask & (phase_snapshot == GamePhase.TERRITORY_PROCESSING)
        if territory_mask.any():
            self._step_territory_phase(territory_mask)

        # PHASE: END_TURN (4)
        end_turn_mask = active_mask & (phase_snapshot == GamePhase.END_TURN)
        if end_turn_mask.any():
            self._step_end_turn_phase(end_turn_mask)

    def _validate_placement_moves_sample(
        self,
        moves: "BatchMoves",
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of placement moves against CPU rules.

        Called when shadow_validator is enabled. Samples games probabilistically
        and validates GPU-generated moves against canonical CPU implementation.
        """
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Extract GPU moves for this game
            move_start = moves.move_offsets[g].item()
            move_count = moves.moves_per_game[g].item()

            if move_count == 0:
                continue

            gpu_positions = []
            for i in range(move_count):
                idx = move_start + i
                # Placement moves store position in from_y, from_x (target position)
                row = moves.from_y[idx].item()
                col = moves.from_x[idx].item()

                # Convert GPU grid coords to CPU format
                # For hex boards (25x25 embedding): convert to axial coords
                # CPU axial (x, y) = GPU grid (col - center, row - center)
                if self.board_type and self.board_type.lower() in ("hexagonal", "hex"):
                    center = self.board_size // 2  # 12 for 25x25
                    x = col - center
                    y = row - center
                else:
                    # Square boards: grid coords match directly
                    x = col
                    y = row

                gpu_positions.append((x, y))

            # Convert to CPU state and validate
            cpu_state = self.state.to_game_state(g)
            player = self.state.current_player[g].item()

            self.shadow_validator.validate_placement_moves(
                gpu_positions, cpu_state, player
            )

    def _validate_movement_moves_sample(
        self,
        movement_moves: "BatchMoves",
        capture_moves: "BatchMoves",
        mask: torch.Tensor,
    ) -> None:
        """Shadow validate a sample of movement/capture moves against CPU rules.

        Called when shadow_validator is enabled. Validates both movement and
        capture move generation against canonical CPU implementation.
        """
        if self.shadow_validator is None:
            return

        game_indices = torch.where(mask)[0].tolist()

        for g in game_indices:
            if not self.shadow_validator.should_validate():
                continue

            # Hex coordinate conversion helper
            is_hex = self.board_type and self.board_type.lower() in ("hexagonal", "hex")
            hex_center = self.board_size // 2 if is_hex else 0

            def to_cpu_coords(row: int, col: int):
                """Convert GPU grid to CPU coords."""
                if is_hex:
                    return col - hex_center, row - hex_center
                return col, row

            # Extract GPU movement moves
            move_start = movement_moves.move_offsets[g].item()
            move_count = movement_moves.moves_per_game[g].item()

            gpu_movement = []
            for i in range(move_count):
                idx = move_start + i
                from_row = movement_moves.from_y[idx].item()
                from_col = movement_moves.from_x[idx].item()
                to_row = movement_moves.to_y[idx].item()
                to_col = movement_moves.to_x[idx].item()
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_movement.append(((from_x, from_y), (to_x, to_y)))

            # Extract GPU capture moves
            cap_start = capture_moves.move_offsets[g].item()
            cap_count = capture_moves.moves_per_game[g].item()

            gpu_captures = []
            for i in range(cap_count):
                idx = cap_start + i
                from_row = capture_moves.from_y[idx].item()
                from_col = capture_moves.from_x[idx].item()
                to_row = capture_moves.to_y[idx].item()
                to_col = capture_moves.to_x[idx].item()
                # Convert to CPU format
                from_x, from_y = to_cpu_coords(from_row, from_col)
                to_x, to_y = to_cpu_coords(to_row, to_col)
                gpu_captures.append(((from_x, from_y), (to_x, to_y)))

            # Convert to CPU state and validate
            cpu_state = self.state.to_game_state(g)
            player = self.state.current_player[g].item()

            if gpu_movement:
                self.shadow_validator.validate_movement_moves(
                    gpu_movement, cpu_state, player
                )

            if gpu_captures:
                self.shadow_validator.validate_capture_moves(
                    gpu_captures, cpu_state, player
                )

    def _player_has_real_action_gpu(self, g: int, player: int) -> bool:
        """Return True if player has any RR-CANON-R172 real action in game g.

        Real actions are:
        - any legal placement (approximated here as ``rings_in_hand > 0``), OR
        - any legal non-capture movement or overtaking capture.

        Recovery and forced elimination do NOT count as real actions.
        """
        # Placement: treat any remaining rings in hand as a real action.
        if self.state.rings_in_hand[g, player].item() > 0:
            return True

        # Without controlled stacks, there is no movement/capture.
        if not (self.state.stack_owner[g] == player).any().item():
            return False

        prev_player = int(self.state.current_player[g].item())
        self.state.current_player[g] = player
        try:
            single_mask = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            single_mask[g] = True
            movement_moves = generate_movement_moves_batch(self.state, single_mask)
            capture_moves = generate_capture_moves_batch(self.state, single_mask)
            return bool(
                (movement_moves.moves_per_game[g].item() > 0)
                or (capture_moves.moves_per_game[g].item() > 0)
            )
        finally:
            self.state.current_player[g] = prev_player

    def _maybe_apply_lps_victory_at_turn_start(
        self,
        mask: torch.Tensor,
        player_has_rings: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply RR-CANON-R172 LPS victory at the start of a player's turn.

        This is called after updating round tracking in ``ring_placement``.
        We only run the expensive real-action check when a candidate has
        already achieved the required consecutive exclusive rounds.
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        lps_required = self.lps_victory_rounds
        candidate_mask = (
            active_mask
            & (self.state.lps_consecutive_exclusive_rounds >= lps_required)
            & (self.state.lps_consecutive_exclusive_player > 0)
            & (
                self.state.current_player
                == self.state.lps_consecutive_exclusive_player
            )
        )
        if not candidate_mask.any():
            return

        if player_has_rings is None:
            player_has_rings = self._compute_player_ring_status_batch()

        for g in torch.where(candidate_mask)[0].tolist():
            if self.state.game_status[g].item() != GameStatus.ACTIVE:
                continue

            candidate = int(self.state.lps_consecutive_exclusive_player[g].item())
            if candidate <= 0:
                continue

            if not self._player_has_real_action_gpu(g, candidate):
                continue

            others_have_real = False
            for pid in range(1, self.num_players + 1):
                if pid == candidate:
                    continue
                if not bool(player_has_rings[g, pid].item()):
                    continue
                if self._player_has_real_action_gpu(g, pid):
                    others_have_real = True
                    break

            if others_have_real:
                continue

            self.state.winner[g] = candidate
            self.state.game_status[g] = GameStatus.COMPLETED

    def _update_lps_round_tracking_for_current_player(
        self,
        mask: torch.Tensor,
    ) -> None:
        """Update LPS round tracking (RR-CANON-R172) for games in mask.

        Mirrors the CPU/TS approach:
        - Track the first player of the current round.
        - Mark each non-permanently-eliminated player as "seen" once per round.
        - Record whether that player had any real action available at the start
          of their turn (placements count; recovery does not).
        - When cycling back to the first player, finalize the previous round and
          update consecutive-exclusive round counters.
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        player_has_rings = self._compute_player_ring_status_batch()
        current = self.state.current_player
        first = self.state.lps_current_round_first_player

        first_has_rings = torch.gather(
            player_has_rings,
            dim=1,
            index=first.unsqueeze(1).long(),
        ).squeeze(1)

        starting_new_cycle = active_mask & ((first == 0) | (~first_has_rings))

        round_has_entries = self.state.lps_current_round_seen_mask.any(dim=1)
        finalize_round = (
            active_mask
            & (~starting_new_cycle)
            & (current == first)
            & round_has_entries
        )

        if starting_new_cycle.any():
            idx = starting_new_cycle
            self.state.lps_round_index[idx] += 1
            self.state.lps_current_round_first_player[idx] = current[idx]
            self.state.lps_current_round_seen_mask[idx] = False
            self.state.lps_current_round_real_action_mask[idx] = False
            self.state.lps_exclusive_player_for_completed_round[idx] = 0

            # Reset consecutive tracking only if the prior exclusive player
            # also dropped out (per TS semantics).
            excl = self.state.lps_consecutive_exclusive_player
            excl_has_rings = torch.gather(
                player_has_rings,
                dim=1,
                index=excl.unsqueeze(1).long(),
            ).squeeze(1)
            reset_consecutive = idx & (~excl_has_rings)
            if reset_consecutive.any():
                self.state.lps_consecutive_exclusive_rounds[reset_consecutive] = 0
                self.state.lps_consecutive_exclusive_player[reset_consecutive] = 0

        if finalize_round.any():
            idx = finalize_round
            eligible = player_has_rings & self.state.lps_current_round_seen_mask
            real_action_players = eligible & self.state.lps_current_round_real_action_mask

            true_counts = real_action_players.sum(dim=1).to(torch.int16)
            exclusive_pid = torch.argmax(real_action_players.to(torch.int8), dim=1).to(torch.int8)
            exclusive_pid = torch.where(
                true_counts == 1,
                exclusive_pid,
                torch.zeros_like(exclusive_pid),
            )

            self.state.lps_exclusive_player_for_completed_round[idx] = exclusive_pid[idx]

            has_exclusive = idx & (exclusive_pid != 0)
            same_exclusive = has_exclusive & (
                exclusive_pid == self.state.lps_consecutive_exclusive_player
            )
            if same_exclusive.any():
                self.state.lps_consecutive_exclusive_rounds[same_exclusive] += 1

            diff_exclusive = has_exclusive & (~same_exclusive)
            if diff_exclusive.any():
                self.state.lps_consecutive_exclusive_player[diff_exclusive] = exclusive_pid[diff_exclusive]
                self.state.lps_consecutive_exclusive_rounds[diff_exclusive] = 1

            no_exclusive = idx & (exclusive_pid == 0)
            if no_exclusive.any():
                self.state.lps_consecutive_exclusive_rounds[no_exclusive] = 0
                self.state.lps_consecutive_exclusive_player[no_exclusive] = 0

            # Start a new round from the current (first) player.
            self.state.lps_round_index[idx] += 1
            self.state.lps_current_round_first_player[idx] = current[idx]
            self.state.lps_current_round_seen_mask[idx] = False
            self.state.lps_current_round_real_action_mask[idx] = False

        # Record that the current player started their turn in this round.
        game_indices = torch.where(active_mask)[0]
        player_indices = current[game_indices].long()
        self.state.lps_current_round_seen_mask[game_indices, player_indices] = True

        # Record initial real-action availability from placements only.
        # Movement/capture availability is filled in during MOVEMENT phase.
        rings_for_current = torch.gather(
            self.state.rings_in_hand,
            dim=1,
            index=current.unsqueeze(1).long(),
        ).squeeze(1)
        has_placement = rings_for_current > 0
        self.state.lps_current_round_real_action_mask[
            game_indices, player_indices
        ] |= has_placement[game_indices]

        # Apply LPS victory at the start of the candidate's next turn.
        self._maybe_apply_lps_victory_at_turn_start(active_mask, player_has_rings)

    def _step_placement_phase(
        self,
        mask: torch.Tensor,
        weights_list: List[Dict[str, float]],
    ) -> None:
        """Handle RING_PLACEMENT phase for games in mask.

        Per RR-CANON-R073, every turn begins in ``ring_placement``. If the current
        player has a legal placement they may place (we generate a single-ring
        placement for simplicity). Otherwise they proceed to ``movement``.

        Note: Placement is part of the player's turn; player rotation happens in
        ``END_TURN``.
        """
        # RR-CANON-R172: update round tracking at the start of the player's
        # turn (in ring_placement). LPS victory is applied here when eligible.
        self._update_lps_round_tracking_for_current_player(mask)

        # Some games may have ended by LPS; drop them from this phase step.
        mask = mask & self.state.get_active_mask()
        if not mask.any():
            return

        # Check which games have rings to place (vectorized).
        current_players = self.state.current_player  # (batch_size,)
        rings_for_current_player = torch.gather(
            self.state.rings_in_hand,
            dim=1,
            index=current_players.unsqueeze(1).long()
        ).squeeze(1)

        # If the player is recovery-eligible, allow movement so they can take
        # a recovery action (RR-CANON-R110): skipping placement is permitted
        # even when rings remain in hand.
        player_expanded = current_players.view(self.batch_size, 1, 1).expand_as(self.state.stack_owner)
        controls_stack = (self.state.stack_owner == player_expanded).any(dim=(1, 2))
        has_marker = (self.state.marker_owner == player_expanded).any(dim=(1, 2))
        buried_for_current = torch.gather(
            self.state.buried_rings,
            dim=1,
            index=current_players.unsqueeze(1).long(),
        ).squeeze(1)
        recovery_eligible = mask & (~controls_stack) & has_marker & (buried_for_current > 0)

        games_with_rings = mask & (rings_for_current_player > 0) & (~recovery_eligible)

        # Games WITH rings: generate and apply placement moves
        if games_with_rings.any():
            moves = generate_placement_moves_batch(self.state, games_with_rings)

            # Shadow validation: validate move generation against CPU
            self._validate_placement_moves_sample(moves, games_with_rings)

            if moves.total_moves > 0:
                # Use configured move selection strategy
                selected = self._select_moves(moves, games_with_rings)
                apply_placement_moves_batch(self.state, selected, moves)

        # Advance to MOVEMENT for this player's turn regardless of whether a
        # placement occurred (no legal placements, no rings in hand, or a
        # strategic skip to enable recovery).
        self.state.current_phase[mask] = GamePhase.MOVEMENT

    def _step_movement_phase(
        self,
        mask: torch.Tensor,
        weights_list: List[Dict[str, float]],
    ) -> None:
        """Handle MOVEMENT phase for games in mask.

        Generate both non-capture movement and capture moves,
        select the best one, and apply it.

        Refactored 2025-12-11 to use vectorized selection and application:
        - select_moves_vectorized() for parallel move selection
        - apply_*_moves_vectorized() for batched move application
        - Reduces per-game Python loops and .item() calls
        - See GPU_PIPELINE_ROADMAP.md Section 2.1 (False Parallelism) for context
        """
        # Check which games have stacks to move (vectorized)
        # Create mask for each player slot and check ownership
        current_players = self.state.current_player  # (batch_size,)

        # Expand current_player to match stack_owner shape for comparison
        # stack_owner shape: (batch_size, board_size, board_size)
        player_expanded = current_players.view(self.batch_size, 1, 1).expand_as(self.state.stack_owner)
        has_any_stack = (self.state.stack_owner == player_expanded).any(dim=(1, 2))
        has_stacks = mask & has_any_stack

        games_with_stacks = has_stacks
        games_without_stacks = mask & ~has_stacks

        # Games WITHOUT stacks: check for recovery moves
        if games_without_stacks.any():
            recovery_moves = generate_recovery_moves_batch(self.state, games_without_stacks)

            # Identify games with recovery moves (vectorized)
            games_with_recovery = games_without_stacks & (recovery_moves.moves_per_game > 0)
            games_no_recovery = games_without_stacks & (recovery_moves.moves_per_game == 0)

            # Apply recovery moves using configured selection strategy
            if games_with_recovery.any():
                selected_recovery = self._select_moves(recovery_moves, games_with_recovery)
                apply_recovery_moves_vectorized(
                    self.state, selected_recovery, recovery_moves, games_with_recovery
                )

            # No movement/capture/recovery action is available for the current player.
            # Do not treat this as an immediate stalemate (RR-CANON-R173 is global and
            # evaluated only for bare-board terminality); instead, record an explicit
            # NO_ACTION move and allow turn/round machinery (including LPS) to proceed.
            if games_no_recovery.any():
                apply_no_action_moves_batch(self.state, games_no_recovery)

        # Games WITH stacks: generate movement and capture moves
        if games_with_stacks.any():
            # Generate non-capture movement moves
            movement_moves = generate_movement_moves_batch(self.state, games_with_stacks)

            # Generate capture moves
            capture_moves = generate_capture_moves_batch(self.state, games_with_stacks)

            # RR-CANON-R172: movement/capture availability counts as a "real action"
            # for LPS purposes; recovery does not. Record availability at the
            # start of MOVEMENT (before applying any move).
            has_real_action = (movement_moves.moves_per_game > 0) | (capture_moves.moves_per_game > 0)
            lps_game_indices = torch.where(games_with_stacks)[0]
            if len(lps_game_indices) > 0:
                lps_players = current_players[lps_game_indices].long()
                self.state.lps_current_round_real_action_mask[
                    lps_game_indices, lps_players
                ] |= has_real_action[lps_game_indices]

            # Shadow validation: validate move generation against CPU
            self._validate_movement_moves_sample(movement_moves, capture_moves, games_with_stacks)

            # Identify which games have captures (prefer captures per RR-CANON)
            # Per RR-CANON-R103: After executing a capture, if additional legal captures
            # exist from the new landing position, the chain MUST continue.
            games_with_captures = games_with_stacks & (capture_moves.moves_per_game > 0)
            games_movement_only = games_with_stacks & (capture_moves.moves_per_game == 0) & (movement_moves.moves_per_game > 0)
            games_no_action = games_with_stacks & (capture_moves.moves_per_game == 0) & (movement_moves.moves_per_game == 0)

            # Apply capture moves with chain capture support (RR-CANON-R103)
            if games_with_captures.any():
                selected_captures = self._select_moves(capture_moves, games_with_captures)
                apply_capture_moves_batch(
                    self.state, selected_captures, capture_moves
                )

                # Track landing positions for chain capture continuation
                # Pre-extract data in batch to minimize .item() calls
                game_indices = torch.where(games_with_captures)[0]

                if game_indices.numel() > 0:
                    # Batch extract local indices and compute global indices
                    local_indices = selected_captures[game_indices]
                    valid_local = local_indices >= 0

                    # Compute global indices for valid selections
                    offsets = capture_moves.move_offsets[game_indices]
                    global_indices = offsets + local_indices
                    valid_global = valid_local & (global_indices < capture_moves.total_moves)

                    # Get landing positions for all valid captures
                    clamped_global = global_indices.clamp(0, max(0, capture_moves.total_moves - 1))
                    landing_y_batch = capture_moves.to_y[clamped_global]
                    landing_x_batch = capture_moves.to_x[clamped_global]

                    # Convert to numpy for efficient iteration
                    game_indices_np = game_indices.cpu().numpy()
                    valid_global_np = valid_global.cpu().numpy()
                    landing_y_np = landing_y_batch.cpu().numpy()
                    landing_x_np = landing_x_batch.cpu().numpy()

                    # Process chain captures for each game
                    for idx, g in enumerate(game_indices_np):
                        if not valid_global_np[idx]:
                            continue

                        landing_y = int(landing_y_np[idx])
                        landing_x = int(landing_x_np[idx])

                        # Chain capture loop: continue capturing from landing position
                        # per RR-CANON-R103 (mandatory chain captures)
                        max_chain_depth = 10  # Safety limit to prevent infinite loops
                        chain_depth = 0

                        while chain_depth < max_chain_depth:
                            chain_depth += 1

                            # Generate captures from current landing position only
                            chain_captures = generate_chain_capture_moves_from_position(
                                self.state, int(g), landing_y, landing_x
                            )

                            if not chain_captures:
                                # No more captures available from this position
                                break

                            # Select a chain capture (use first available for simplicity)
                            # In training, randomizing might be better but for correctness
                            # any valid chain is acceptable
                            to_y, to_x = chain_captures[0]

                            # Apply the chain capture
                            new_y, new_x = apply_single_chain_capture(
                                self.state, int(g), landing_y, landing_x, to_y, to_x
                            )

                            # Update landing position for next iteration
                            landing_y, landing_x = new_y, new_x

            # Apply movement moves for games without captures
            if games_movement_only.any():
                selected_movements = self._select_moves(movement_moves, games_movement_only)
                apply_movement_moves_batch(
                    self.state, selected_movements, movement_moves
                )

            if games_no_action.any():
                apply_no_action_moves_batch(self.state, games_no_action)

        # After movement, advance to LINE_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.LINE_PROCESSING

    def _step_line_phase(self, mask: torch.Tensor) -> None:
        """Handle LINE_PROCESSING phase for games in mask.

        Detect lines and convert them to territory markers.

        Per RR-CANON-R121-R122: Process all eligible lines for the current player.
        After line processing, check for new lines formed by territory collapse
        (cascade processing per RR-CANON-R144).
        """
        process_lines_batch(self.state, mask)

        # After line processing, advance to TERRITORY_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.TERRITORY_PROCESSING

    def _step_territory_phase(self, mask: torch.Tensor) -> None:
        """Handle TERRITORY_PROCESSING phase for games in mask.

        Calculate enclosed territory using flood-fill.

        Per RR-CANON-R144-R145: Territory processing may create conditions for
        new lines (e.g., markers that were separated now form a line due to
        territory collapse removing blocking pieces). In this case, we need to
        return to LINE_PROCESSING phase for cascade handling.

        SIMPLIFICATION (GPU training): We implement a limited cascade check.
        Full cascade would iteratively process line->territory->line until stable.
        For training efficiency, we do one round of cascade check.
        """
        # Process territory claims
        compute_territory_batch(self.state, mask)

        # Cascade check: Did territory processing create new marker lines?
        # This can happen if territory collapse removes stacks that were blocking
        # marker alignment, or if markers from captured stacks now form lines.
        cascade_games = self._check_for_new_lines(mask)

        if cascade_games.any():
            # Games with new lines go back to LINE_PROCESSING
            self.state.current_phase[cascade_games] = GamePhase.LINE_PROCESSING
            # Games without new lines advance to END_TURN
            no_cascade = mask & ~cascade_games
            self.state.current_phase[no_cascade] = GamePhase.END_TURN
        else:
            # No cascade needed, all games advance to END_TURN
            self.state.current_phase[mask] = GamePhase.END_TURN

    def _check_for_new_lines(self, mask: torch.Tensor) -> torch.Tensor:
        """Check which games have new marker lines after territory processing.

        Used for cascade detection per RR-CANON-R144.

        Refactored 2025-12-13 for vectorized efficiency:
        - Use detect_lines_vectorized which returns line counts
        - O(1) check via count > 0

        Args:
            mask: Games to check

        Returns:
            Boolean tensor indicating which games have new lines
        """
        has_new_lines = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Check each player's lines across all masked games at once
        for p in range(1, self.num_players + 1):
            # detect_lines_vectorized returns (in_line_mask, line_counts)
            _, line_counts = detect_lines_vectorized(self.state, p, mask)
            # Games with line_count > 0 have lines
            has_new_lines = has_new_lines | (line_counts > 0)

        return has_new_lines

    def _step_end_turn_phase(self, mask: torch.Tensor) -> None:
        """Handle END_TURN phase for games in mask.

        Rotate to next player and reset phase to RING_PLACEMENT.

        Per updated rules: Players are only permanently eliminated if they have
        NO rings anywhere (no controlled stacks, no buried rings, no rings in hand).
        Players with only buried rings still get turns and can use recovery moves.

        Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
        NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action
        and proceed to movement, but they MUST enter ring_placement first.

        Refactored 2025-12-11 for vectorized player rotation:
        - Precompute player elimination status for all players in all games
        - Use vectorized rotation with fallback for eliminated player skipping
        """
        # NOTE: move_count is incremented in the vectorized move application functions
        # (apply_capture_moves_vectorized, apply_movement_moves_vectorized, etc.)
        # when recording moves to history, so we don't increment here.

        # Precompute player elimination status for all games and players
        # Shape: (batch_size, num_players+1) - player_has_rings[g, p] = True if player p has rings in game g
        player_has_rings = self._compute_player_ring_status_batch()

        # Vectorized player rotation with eliminated player skipping
        # For most games (2-player with no eliminations), this is a simple increment
        current_players = self.state.current_player.clone()  # (batch_size,)

        # Start with simple rotation: (current % num_players) + 1
        next_players = (current_players % self.num_players) + 1

        # For games where the next player is eliminated, find the next non-eliminated player
        # This handles the uncommon case where we need to skip eliminated players
        for skip_round in range(self.num_players):
            # Check which games have an eliminated next player
            # Use gather to check player_has_rings[g, next_players[g]]
            next_player_has_rings = torch.gather(
                player_has_rings,
                dim=1,
                index=next_players.unsqueeze(1).long()
            ).squeeze(1)

            # Games where next player is eliminated AND we're in the mask
            needs_skip = mask & ~next_player_has_rings

            if not needs_skip.any():
                break

            # Rotate eliminated players to next candidate
            next_players[needs_skip] = (next_players[needs_skip] % self.num_players) + 1

        # Apply the computed next players
        self.state.current_player[mask] = next_players[mask]

        # Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
        # NO PHASE SKIPPING - this is a core invariant for parity with TS/Python engines.
        self.state.current_phase[mask] = GamePhase.RING_PLACEMENT
        self.state.must_move_from_y[mask] = -1
        self.state.must_move_from_x[mask] = -1

        # Swap sides (pie rule) check for 2-player games (RR-CANON R180-R184)
        # Offered to P2 immediately after P1's first complete turn
        if self.num_players == 2 and self.swap_enabled:
            self._check_and_apply_swap_sides(mask)

    def _check_and_apply_swap_sides(self, mask: torch.Tensor) -> None:
        """Check for swap_sides eligibility and mark it as offered.

        Per RR-CANON R180-R184: The pie rule allows P2 to swap colours/seats
        with P1 immediately after P1's first complete turn.

        Conditions for swap eligibility:
        1. 2-player game (already checked by caller)
        2. Current player is now P2
        3. Swap not already offered in this game
        4. P1 has completed at least one full turn (has moves in history)

        IMPORTANT (GPU parity):
        The canonical engines implement swap_sides as an *identity swap* that
        does not change board ownership or per-seat counters (ringsInHand,
        eliminatedRings, territorySpaces). GPU self-play does not currently
        record explicit swap_sides moves in its coarse move_history format, so
        applying a semantic swap here would create non-replayable traces.

        Until GPU move history is upgraded to represent swap_sides explicitly,
        we treat the pie rule as "offered but always declined" and only set
        swap_offered for observability/debugging.
        """
        # Identify games where swap should be offered:
        # - In mask (just completed END_TURN)
        # - Current player is now P2
        # - Swap not already offered
        is_p2_turn = self.state.current_player == 2
        not_yet_offered = ~self.state.swap_offered
        swap_eligible = mask & is_p2_turn & not_yet_offered

        if not swap_eligible.any():
            return

        # Mark swap as offered for these games (regardless of acceptance)
        self.state.swap_offered[swap_eligible] = True
        return

    def _compute_player_ring_status_batch(self) -> torch.Tensor:
        """Compute which players have any rings in each game (vectorized).

        Returns:
            Boolean tensor of shape (batch_size, num_players+1) where
            result[g, p] = True if player p has any rings in game g.
            Index 0 is unused (players are 1-indexed).

        A player has rings if ANY of:
        - rings_in_hand[g, p] > 0
        - Any cell where stack_owner[g, y, x] == p (controlled stacks)
        - buried_rings[g, p] > 0
        """
        device = self.device
        batch_size = self.batch_size
        num_players = self.num_players

        # Initialize result tensor
        has_rings = torch.zeros(batch_size, num_players + 1, dtype=torch.bool, device=device)

        for p in range(1, num_players + 1):
            # Check rings in hand
            has_in_hand = self.state.rings_in_hand[:, p] > 0

            # Check controlled stacks (any cell where stack_owner == p)
            has_controlled = (self.state.stack_owner == p).any(dim=(1, 2))

            # Check buried rings
            has_buried = self.state.buried_rings[:, p] > 0

            # Player has rings if any of the above is true
            has_rings[:, p] = has_in_hand | has_controlled | has_buried

        return has_rings

    def _player_has_any_rings_gpu(self, g: int, player: int) -> bool:
        """Check if a player has any rings anywhere (controlled, buried, or in hand).

        A player with no rings anywhere is permanently eliminated.
        A player who has rings (even if only buried) is NOT permanently eliminated
        and should still get turns (they may have recovery moves available).

        Args:
            g: Game index in batch
            player: Player number (1-indexed)

        Returns:
            True if player has any rings anywhere, False if permanently eliminated
        """
        # Check rings in hand
        if self.state.rings_in_hand[g, player].item() > 0:
            return True

        # Check controlled stacks (rings in stacks we control)
        has_controlled_stacks = (self.state.stack_owner[g] == player).any().item()
        if has_controlled_stacks:
            return True

        # Check buried rings (rings in opponent-controlled stacks)
        # buried_rings uses 1-indexed players (shape is batch_size x num_players+1)
        if self.state.buried_rings[g, player].item() > 0:
            return True

        return False

    def _apply_single_capture(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single capture move for game g at global index move_idx.

        Per RR-CANON-R100-R103:
        - Attacker moves onto defender stack
        - Defender's top ring is eliminated
        - Stacks merge (attacker on top)
        - Control transfers to attacker
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        move_count = state.move_count[g].item()
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x

        # Get moving stack info
        attacker_height = state.stack_height[g, from_y, from_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()
        defender_owner = state.stack_owner[g, to_y, to_x].item()

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                # Flip opponent marker to our color
                state.marker_owner[g, check_y, check_x] = player

        # Eliminate defender's top ring
        defender_eliminated = 1
        defender_new_height = max(0, defender_height - defender_eliminated)

        # Track elimination
        if defender_owner > 0:
            # Defender LOSES the ring
            state.eliminated_rings[g, defender_owner] += defender_eliminated
            # Attacker (player) CAUSED the elimination (for victory check)
            state.rings_caused_eliminated[g, player] += defender_eliminated

        # Clear attacker origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        # Place merged stack at destination (attacker on top)
        new_height = attacker_height + defender_new_height
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(5, new_height)  # Cap at 5

    def _apply_single_movement(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single movement move for game g at global index move_idx.

        Per RR-CANON-R090-R092:
        - Stack moves from origin to destination
        - Origin becomes empty
        - Destination gets merged stack (if own stack) or new stack
        - Markers along path: flip on pass, collapse cost on landing
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        move_count = state.move_count[g].item()
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x

        # Get moving stack info
        moving_height = state.stack_height[g, from_y, from_x].item()

        # Process markers along path (simplified - flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):  # Don't flip at destination yet
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                # Flip opponent marker to our color
                state.marker_owner[g, check_y, check_x] = player

        # Handle landing on own marker (collapse cost)
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 0
        if dest_marker == player:
            # Landing on own marker costs 1 ring (cap elimination)
            landing_ring_cost = 1
            state.is_collapsed[g, to_y, to_x] = True
            state.marker_owner[g, to_y, to_x] = 0  # Marker consumed

        # Clear origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        # Handle destination
        dest_owner = state.stack_owner[g, to_y, to_x].item()
        dest_height = state.stack_height[g, to_y, to_x].item()

        if dest_owner == 0:
            # Landing on empty
            new_height = moving_height - landing_ring_cost
            state.stack_owner[g, to_y, to_x] = player
            state.stack_height[g, to_y, to_x] = max(1, new_height)
        elif dest_owner == player:
            # Merging with own stack
            new_height = dest_height + moving_height - landing_ring_cost
            state.stack_height[g, to_y, to_x] = min(5, new_height)  # Cap at 5

    def _apply_single_recovery(self, g: int, move_idx: int, moves: BatchMoves) -> None:
        """Apply a single recovery slide move for game g at global index move_idx.

        Per RR-CANON-R110-R115:
        - Recovery slide: marker moves to adjacent empty cell
        - Costs 1 buried ring (deducted from buried_rings)
        - Origin marker is cleared, destination gets marker
        - Player gains recovery attempt toward un-burying rings

        Per RR-CANON-R114 (Recovery Cascade):
        - After recovery move completes, check for line formation
        - If line formed, process it (collapse markers, eliminate ring)
        - After line processing, check for territory claims
        - Cascade continues until no new lines are formed
        """
        state = self.state
        from_y = moves.from_y[move_idx].item()
        from_x = moves.from_x[move_idx].item()
        to_y = moves.to_y[move_idx].item()
        to_x = moves.to_x[move_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[move_idx].item()

        # Record move in history
        move_count = state.move_count[g].item()
        if move_count < state.max_history_moves:
            state.move_history[g, move_count, 0] = move_type
            state.move_history[g, move_count, 1] = player
            state.move_history[g, move_count, 2] = from_y
            state.move_history[g, move_count, 3] = from_x
            state.move_history[g, move_count, 4] = to_y
            state.move_history[g, move_count, 5] = to_x

        # Move marker from origin to destination
        state.marker_owner[g, from_y, from_x] = 0  # Clear origin
        state.marker_owner[g, to_y, to_x] = player  # Place at destination

        # Deduct recovery cost: 1 buried ring
        # In the canonical rules, recovery costs rings from the buried pool
        # NOTE: buried_rings is 1-indexed (shape: batch_size, num_players + 1)
        current_buried = state.buried_rings[g, player].item()
        if current_buried > 0:
            state.buried_rings[g, player] = current_buried - 1

        # Recovery Cascade per RR-CANON-R114:
        # After recovery move, the marker slide could form a line, which triggers
        # line processing, which could trigger territory claims, and so on.
        self._process_recovery_cascade(g, player)

    def _process_recovery_cascade(self, g: int, player: int, max_iterations: int = 5) -> None:
        """Process line formation and territory claims after a recovery move.

        Per RR-CANON-R114: After a recovery move, check if a line was formed.
        If so, process the line (collapse markers, eliminate ring from any stack).
        Then check for territory claims. This cascade continues until stable.

        Args:
            g: Game index
            player: Player who made the recovery move
            max_iterations: Safety limit to prevent infinite loops (default 5)
        """
        # Create a single-game mask for this game
        single_game_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        single_game_mask[g] = True

        for iteration in range(max_iterations):
            # Check for lines for the current player using vectorized detection
            _, line_counts = detect_lines_vectorized(self.state, player, single_game_mask)

            if line_counts[g].item() == 0:
                # No lines formed, we're done with cascade
                break

            # Process lines (collapse markers, eliminate rings)
            process_lines_batch(self.state, single_game_mask)

            # After line processing, check for territory claims
            compute_territory_batch(self.state, single_game_mask)

            # Continue loop to check if territory processing created new lines
            # (e.g., by removing markers that were blocking a line)

    def _select_best_moves(
        self,
        moves: BatchMoves,
        weights_list: List[Dict[str, float]],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select the best move for each game using heuristic evaluation.

        This is a simplified version - a full implementation would:
        1. Apply each candidate move to a temporary state
        2. Evaluate the resulting position
        3. Select the move with best score

        For now, we select randomly with bias toward center positions.
        """
        batch_size = self.batch_size
        device = self.device

        # Simple selection: prefer center positions
        selected = torch.zeros(batch_size, dtype=torch.int64, device=device)

        center = self.board_size // 2

        for g in range(batch_size):
            if not active_mask[g] or moves.moves_per_game[g] == 0:
                continue

            # Get moves for this game
            start_idx = moves.move_offsets[g]
            end_idx = start_idx + moves.moves_per_game[g]

            game_moves_y = moves.from_y[start_idx:end_idx]
            game_moves_x = moves.from_x[start_idx:end_idx]

            # Score by distance to center (lower is better -> invert for softmax)
            dist_to_center = (
                (game_moves_y.float() - center).abs() +
                (game_moves_x.float() - center).abs()
            )

            # Convert to scores: higher score for closer to center
            max_dist = center * 2  # Maximum possible Manhattan distance
            scores = (max_dist - dist_to_center) + torch.rand_like(dist_to_center) * 2.0

            # Softmax selection with temperature=1.0 for stochasticity
            probs = torch.softmax(scores, dim=0)
            best_local_idx = torch.multinomial(probs, 1).item()
            selected[g] = best_local_idx

        return selected

    def _check_victory_conditions(self) -> None:
        """Check and update victory conditions for all games.

        Implements canonical rules:
        - RR-CANON-R170: Ring-elimination victory (eliminatedRingsTotal >= victoryThreshold)
        - RR-CANON-R171: Territory-control victory (dual condition: threshold AND dominance)
        - RR-CANON-R172: Last-player-standing (round-based exclusive real actions)

        Victory thresholds per RR-CANON-R061/R062-v2:
        - victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
        - territoryVictoryMinimum = floor(totalSpaces / numPlayers) + 1 [plus dominance check]
        """
        active_mask = self.state.get_active_mask()

        # Canonical thresholds depend on board type and player count (RR-CANON-R061/R062).
        from app.models import BoardType
        from app.rules.core import get_territory_victory_minimum, get_victory_threshold

        board_type_map = {
            8: BoardType.SQUARE8,
            19: BoardType.SQUARE19,
            13: BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)
        ring_elimination_threshold = get_victory_threshold(board_type, self.num_players)
        # Per RR-CANON-R062-v2: Use player-count-aware minimum threshold
        territory_victory_minimum = get_territory_victory_minimum(board_type, self.num_players)

        for p in range(1, self.num_players + 1):
            # Check ring-elimination victory (RR-CANON-R170)
            # Per RR-CANON-R060/R170: Player wins when they have CAUSED >= victoryThreshold
            # rings to be eliminated (includes self-elimination via lines, territory, etc.)
            # rings_caused_eliminated[:, p] tracks rings that player p CAUSED to be eliminated
            ring_elimination_victory = self.state.rings_caused_eliminated[:, p] >= ring_elimination_threshold

            # Check territory victory per RR-CANON-R062-v2 (dual condition)
            # Condition 1: Territory >= floor(totalSpaces / numPlayers) + 1
            player_territory = self.state.territory_count[:, p]
            meets_threshold = player_territory >= territory_victory_minimum

            # Condition 2: Territory > sum of all opponents' territory
            total_territory = self.state.territory_count[:, 1:self.num_players + 1].sum(dim=1)
            opponents_territory = total_territory - player_territory
            dominates_opponents = player_territory > opponents_territory

            # Victory requires BOTH conditions
            territory_victory = meets_threshold & dominates_opponents

            # RR-CANON-R172 (LPS) is applied at turn start via the LPS round
            # tracker (see _update_lps_round_tracking_for_current_player).
            victory_mask = active_mask & (ring_elimination_victory | territory_victory)
            self.state.winner[victory_mask] = p
            self.state.game_status[victory_mask] = GameStatus.COMPLETED

    def _check_stalemate(self, mask: torch.Tensor) -> None:
        """Check for stalemate condition (no valid moves for current player).

        Per RR-CANON-R175 (implied): If the current player has no valid moves
        and cannot make any progress (no placement, no movement, no recovery),
        then the game ends in a stalemate (draw) or tiebreaker by stack count.

        This is called during MOVEMENT phase when a player has neither:
        - Controlled stacks to move
        - Rings in hand to place
        - Recovery moves available

        Stalemate resolution per canonical rules:
        - Winner determined by highest total stack height
        - Ties result in draw
        """
        active_mask = mask & self.state.get_active_mask()
        if not active_mask.any():
            return

        for g in range(self.batch_size):
            if not active_mask[g]:
                continue

            player = self.state.current_player[g].item()

            # Check if player has any stacks
            has_stacks = (self.state.stack_owner[g] == player).any().item()

            # Check if player has rings in hand
            has_rings_in_hand = self.state.rings_in_hand[g, player].item() > 0

            # Check if player has markers and buried rings (recovery eligible)
            # Note: buried_rings uses 1-indexed players (shape is batch_size x num_players+1)
            has_markers = (self.state.marker_owner[g] == player).any().item()
            has_buried = self.state.buried_rings[g, player].item() > 0

            # If player has no possible actions, it's a stalemate
            if not has_stacks and not has_rings_in_hand and not (has_markers and has_buried):
                # Determine winner by stack height (stalemate tiebreaker)
                best_player = 0
                best_total_height = -1
                is_tie = False

                for p in range(1, self.num_players + 1):
                    p_stacks = (self.state.stack_owner[g] == p)
                    p_total_height = (self.state.stack_height[g] * p_stacks.float()).sum().item()

                    if p_total_height > best_total_height:
                        best_total_height = p_total_height
                        best_player = p
                        is_tie = False
                    elif p_total_height == best_total_height and p_total_height > 0:
                        is_tie = True

                # Set winner (0 = draw if tied)
                if is_tie or best_total_height == 0:
                    self.state.winner[g] = 0  # Draw
                else:
                    self.state.winner[g] = best_player

                self.state.game_status[g] = GameStatus.COMPLETED

    def _default_weights(self) -> Dict[str, float]:
        """Default heuristic weights."""
        return {
            "stack_count": 1.0,
            "territory_count": 2.0,
            "rings_penalty": 0.1,
            "center_control": 0.3,
        }

    def _apply_weight_noise(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply multiplicative noise to weights for training diversity.

        Each weight is multiplied by a random factor in [1-noise, 1+noise].

        Args:
            weights: Base weights dictionary

        Returns:
            New weights dictionary with noise applied
        """
        if self.weight_noise <= 0:
            return weights.copy()

        import random
        noisy_weights = {}
        for key, value in weights.items():
            # Multiplicative noise: value * uniform(1-noise, 1+noise)
            noise_factor = 1.0 + random.uniform(-self.weight_noise, self.weight_noise)
            noisy_weights[key] = value * noise_factor
        return noisy_weights

    def _generate_weights_list(self) -> List[Dict[str, float]]:
        """Generate per-game weights with optional noise.

        Returns:
            List of weight dictionaries, one per game in batch.
        """
        base_weights = self._default_weights()
        if self.weight_noise <= 0:
            # No noise - all games use same weights
            return [base_weights] * self.batch_size
        else:
            # Each game gets unique noisy weights
            return [self._apply_weight_noise(base_weights) for _ in range(self.batch_size)]

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            "games_completed": self._games_completed,
            "total_moves": self._total_moves,
            "total_time_seconds": self._total_time,
            "games_per_second": (
                self._games_completed / self._total_time
                if self._total_time > 0 else 0
            ),
            "moves_per_second": (
                self._total_moves / self._total_time
                if self._total_time > 0 else 0
            ),
        }

    def get_shadow_validation_report(self) -> Optional[Dict[str, Any]]:
        """Get shadow validation statistics if enabled.

        Returns:
            Validation report dict if shadow validation enabled, None otherwise.
            Report includes:
                - total_validations: Number of moves validated
                - total_divergences: Number of divergences detected
                - divergence_rate: Divergence rate (0.0-1.0)
                - threshold: Configured threshold
                - status: "PASS" or "FAIL"
                - by_move_type: Breakdown by move type
        """
        if self.shadow_validator is None:
            return None
        return self.shadow_validator.get_report()


# =============================================================================
# CMA-ES Integration
# =============================================================================


def evaluate_candidate_fitness_gpu(
    candidate_weights: Dict[str, float],
    opponent_weights: Dict[str, float],
    num_games: int = 10,
    board_size: int = 8,
    num_players: int = 2,
    max_moves: int = 500,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate CMA-ES candidate fitness using GPU parallel games.

    Runs multiple games between candidate and opponent, returns win rate.

    Args:
        candidate_weights: Heuristic weights for candidate
        opponent_weights: Heuristic weights for opponent
        num_games: Number of games to play
        board_size: Board dimension
        num_players: Number of players
        max_moves: Max moves per game
        device: GPU device

    Returns:
        Win rate (0.0 to 1.0) for candidate
    """
    runner = ParallelGameRunner(
        batch_size=num_games,
        board_size=board_size,
        num_players=num_players,
        device=device,
    )

    # Alternate who plays first
    weights_list = []
    for i in range(num_games):
        if i % 2 == 0:
            weights_list.append(candidate_weights)  # Candidate is P1
        else:
            weights_list.append(opponent_weights)   # Opponent is P1

    results = runner.run_games(weights_list=weights_list, max_moves=max_moves)

    # Count wins for candidate
    wins = 0
    for i, winner in enumerate(results["winners"]):
        if i % 2 == 0:  # Candidate was P1
            if winner == 1:
                wins += 1
        else:  # Candidate was P2
            if winner == 2:
                wins += 1

    win_rate = wins / num_games

    logger.debug(
        f"GPU fitness evaluation: {wins}/{num_games} wins ({win_rate:.1%}), "
        f"{results['games_per_second']:.1f} games/sec"
    )

    return win_rate


def benchmark_parallel_games(
    batch_sizes: List[int] = [1, 8, 32, 64, 128, 256],
    board_size: int = 8,
    max_moves: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Benchmark parallel game simulation performance.

    Args:
        batch_sizes: List of batch sizes to test
        board_size: Board dimension
        max_moves: Max moves per game
        device: GPU device

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "batch_size": [],
        "games_per_second": [],
        "moves_per_second": [],
        "elapsed_seconds": [],
    }

    for batch_size in batch_sizes:
        runner = ParallelGameRunner(
            batch_size=batch_size,
            board_size=board_size,
            device=device,
        )

        # Warmup
        runner.run_games(max_moves=10)

        # Benchmark
        game_results = runner.run_games(max_moves=max_moves)

        results["batch_size"].append(batch_size)
        results["games_per_second"].append(game_results["games_per_second"])
        results["moves_per_second"].append(
            sum(game_results["move_counts"]) / game_results["elapsed_seconds"]
        )
        results["elapsed_seconds"].append(game_results["elapsed_seconds"])

        logger.info(
            f"Batch {batch_size}: {game_results['games_per_second']:.1f} games/sec"
        )

    return results
