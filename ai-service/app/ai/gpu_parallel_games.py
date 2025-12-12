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
import logging
import time
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
    # Sort moves by game_idx to enable cumsum
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

    # Record moves in history (vectorized where possible)
    move_counts = state.move_count.clone()

    # Apply moves game by game (some operations require iteration due to variable paths)
    # This is the minimal iteration - just for path processing
    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = move_counts[g].item()

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.CAPTURE
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
            state.move_count[g] += 1

        # Get stack info
        attacker_height = state.stack_height[g, from_y, from_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()
        defender_owner = state.stack_owner[g, to_y, to_x].item()

        # Process markers along path
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        # Eliminate defender's top ring
        defender_new_height = max(0, defender_height - 1)

        # Track elimination
        if defender_owner > 0:
            current_elim = state.eliminated_rings[g, defender_owner].item()
            state.eliminated_rings[g, defender_owner] = current_elim + 1

        # Merge stacks
        if defender_new_height > 0:
            new_height = attacker_height + defender_new_height
            state.stack_height[g, to_y, to_x] = new_height
        else:
            state.stack_height[g, to_y, to_x] = attacker_height

        state.stack_owner[g, to_y, to_x] = player

        # Clear origin
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0


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
    move_counts = state.move_count.clone()

    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = move_counts[g].item()

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

        # Process markers along path
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        # Handle landing on own marker
        dest_marker = state.marker_owner[g, to_y, to_x].item()
        landing_ring_cost = 0
        if dest_marker == player:
            landing_ring_cost = 1
            state.is_collapsed[g, to_y, to_x] = True
            state.marker_owner[g, to_y, to_x] = 0

        # Handle landing on own stack (merge)
        dest_height = state.stack_height[g, to_y, to_x].item()
        dest_owner = state.stack_owner[g, to_y, to_x].item()

        if dest_owner == player and dest_height > 0:
            new_height = moving_height + dest_height - landing_ring_cost
        else:
            new_height = moving_height - landing_ring_cost

        # Track eliminated ring from landing cost
        if landing_ring_cost > 0:
            current_elim = state.eliminated_rings[g, player].item()
            state.eliminated_rings[g, player] = current_elim + landing_ring_cost

        # Update destination
        state.stack_height[g, to_y, to_x] = max(1, new_height)
        state.stack_owner[g, to_y, to_x] = player

        # Clear origin
        state.stack_height[g, from_y, from_x] = 0
        state.stack_owner[g, from_y, from_x] = 0


def apply_recovery_moves_vectorized(
    state: "BatchGameState",
    selected_local_idx: torch.Tensor,
    moves: "BatchMoves",
    active_mask: torch.Tensor,
) -> None:
    """Apply recovery slide moves in a vectorized manner."""
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
    move_counts = state.move_count.clone()

    game_indices = torch.where(has_selection)[0]

    for g in game_indices.tolist():
        from_y = selected_from_y[g].item()
        from_x = selected_from_x[g].item()
        to_y = selected_to_y[g].item()
        to_x = selected_to_x[g].item()
        player = current_players[g].item()
        mc = move_counts[g].item()

        # Record in history
        if mc < state.max_history_moves:
            state.move_history[g, mc, 0] = MoveType.RECOVERY_SLIDE
            state.move_history[g, mc, 1] = player
            state.move_history[g, mc, 2] = from_y
            state.move_history[g, mc, 3] = from_x
            state.move_history[g, mc, 4] = to_y
            state.move_history[g, mc, 5] = to_x
            state.move_count[g] += 1

        # Move marker
        state.marker_owner[g, from_y, from_x] = 0
        state.marker_owner[g, to_y, to_x] = player

        # Deduct buried ring cost
        current_buried = state.buried_rings[g, player].item()
        if current_buried > 0:
            state.buried_rings[g, player] = current_buried - 1


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
    stack_height: torch.Tensor     # 0-5
    marker_owner: torch.Tensor     # 0=none, 1-4=player
    territory_owner: torch.Tensor  # 0=neutral, 1-4=player
    is_collapsed: torch.Tensor     # bool

    # Player state: (batch_size, num_players)
    rings_in_hand: torch.Tensor
    territory_count: torch.Tensor
    is_eliminated: torch.Tensor    # bool
    eliminated_rings: torch.Tensor # Rings eliminated by this player
    buried_rings: torch.Tensor     # Rings buried in stacks (captured but not removed)

    # Game metadata: (batch_size,)
    current_player: torch.Tensor   # 1-4
    current_phase: torch.Tensor    # GamePhase enum
    move_count: torch.Tensor
    game_status: torch.Tensor      # GameStatus enum
    winner: torch.Tensor           # 0=none, 1-4=player

    # Move history: (batch_size, max_moves, 6) - [move_type, player, from_y, from_x, to_y, to_x]
    # -1 indicates unused slot
    move_history: torch.Tensor
    max_history_moves: int

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
    ) -> "BatchGameState":
        """Create a batch of initialized game states.

        Args:
            batch_size: Number of parallel games
            board_size: Board dimension (8, 19)
            num_players: Number of players (2-4)
            device: GPU device (auto-detected if None)
            max_history_moves: Maximum moves to track in history

        Returns:
            Initialized BatchGameState with all games ready to start
        """
        if device is None:
            device = get_device()

        # Initialize board tensors
        shape_board = (batch_size, board_size, board_size)
        shape_players = (batch_size, num_players + 1)  # +1 for 1-indexed players

        # Starting rings per player based on board size (per RR-CANON-R020)
        # square8: 18, square19: 72, hexagonal: 96
        starting_rings = {8: 18, 19: 72, 13: 96}.get(board_size, 18)

        rings = torch.zeros(shape_players, dtype=torch.int16, device=device)
        rings[:, 1:num_players+1] = starting_rings

        # Move history: (batch_size, max_moves, 6) - [move_type, player, from_y, from_x, to_y, to_x]
        # Initialize with -1 to indicate unused slots
        move_history = torch.full(
            (batch_size, max_history_moves, 6),
            -1,
            dtype=torch.int16,
            device=device,
        )

        return cls(
            stack_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            stack_height=torch.zeros(shape_board, dtype=torch.int8, device=device),
            marker_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            territory_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            is_collapsed=torch.zeros(shape_board, dtype=torch.bool, device=device),
            rings_in_hand=rings,
            territory_count=torch.zeros(shape_players, dtype=torch.int16, device=device),
            is_eliminated=torch.zeros(shape_players, dtype=torch.bool, device=device),
            eliminated_rings=torch.zeros(shape_players, dtype=torch.int16, device=device),
            buried_rings=torch.zeros(shape_players, dtype=torch.int16, device=device),
            current_player=torch.ones(batch_size, dtype=torch.int8, device=device),
            current_phase=torch.zeros(batch_size, dtype=torch.int8, device=device),  # RING_PLACEMENT
            move_count=torch.zeros(batch_size, dtype=torch.int32, device=device),
            game_status=torch.zeros(batch_size, dtype=torch.int8, device=device),
            winner=torch.zeros(batch_size, dtype=torch.int8, device=device),
            move_history=move_history,
            max_history_moves=max_history_moves,
            device=device,
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
        )

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
        for key, stack in game_state.board.stacks.items():
            x, y = map(int, key.split(","))
            if 0 <= x < board_size and 0 <= y < board_size:
                batch.stack_owner[0, y, x] = stack.controlling_player
                batch.stack_height[0, y, x] = len(stack.rings)

        for key, marker in game_state.board.markers.items():
            x, y = map(int, key.split(","))
            if 0 <= x < board_size and 0 <= y < board_size:
                # Handle both int (legacy) and MarkerInfo (current) marker values
                player = marker.player if hasattr(marker, 'player') else marker
                batch.marker_owner[0, y, x] = player

        for key, player in game_state.board.collapsed_spaces.items():
            x, y = map(int, key.split(","))
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
        board_type_map = {
            8: BoardType.SQUARE8,
            19: BoardType.SQUARE19,
            13: BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)

        # Build stacks dict from GPU tensors
        stacks = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                owner = self.stack_owner[game_idx, y, x].item()
                height = self.stack_height[game_idx, y, x].item()
                if owner > 0 and height > 0:
                    key = f"{x},{y}"
                    # Reconstruct rings list (simplified - all same owner)
                    rings = [owner] * height
                    stacks[key] = RingStack(
                        position=Position(x=x, y=y),
                        rings=rings,
                        stackHeight=height,
                        capHeight=height,  # Simplified
                        controllingPlayer=owner,
                    )

        # Build markers dict
        markers = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                marker_player = self.marker_owner[game_idx, y, x].item()
                if marker_player > 0:
                    key = f"{x},{y}"
                    markers[key] = MarkerInfo(
                        player=marker_player,
                        position=Position(x=x, y=y),
                        type="regular",
                    )

        # Build collapsed_spaces dict
        collapsed_spaces = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.is_collapsed[game_idx, y, x].item():
                    territory_player = self.territory_owner[game_idx, y, x].item()
                    key = f"{x},{y}"
                    collapsed_spaces[key] = territory_player

        # Build board state
        board = BoardState(
            type=board_type,
            size=self.board_size,
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
        elif gpu_status == GameStatus.VICTORY:
            game_status = CPUGameStatus.COMPLETED
        else:
            game_status = CPUGameStatus.ACTIVE

        now = datetime.now()

        # Compute canonical victory thresholds for shadow validation.
        # Per RR-CANON-R061/R062 these depend on the board type and player count.
        from app.rules.core import (
            get_rings_per_player,
            get_territory_victory_threshold,
            get_victory_threshold,
        )

        rings_per_player = get_rings_per_player(board_type)
        victory_threshold = get_victory_threshold(board_type, self.num_players)
        territory_threshold = get_territory_victory_threshold(board_type)
        total_rings_in_play = rings_per_player * self.num_players
        total_rings_eliminated = int(
            self.eliminated_rings[game_idx, 1 : self.num_players + 1].sum().item()
        )

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
            territoryVictoryThreshold=territory_threshold,
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

        # Move type names for output
        move_type_names = {
            MoveType.PLACEMENT: "ring_placement",
            MoveType.MOVEMENT: "movement",
            MoveType.CAPTURE: "capture",
            MoveType.LINE_FORMATION: "line_formation",
            MoveType.TERRITORY_CLAIM: "territory_claim",
            MoveType.SKIP: "skip",
        }

        for i in range(num_moves):
            history_row = self.move_history[game_idx, i].cpu().tolist()
            move_type_code, player, from_y, from_x, to_y, to_x = history_row

            if move_type_code < 0:  # -1 indicates unused slot
                break

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
        - ring_elimination: All opponents eliminated
        - territory: Territory threshold reached
        - timeout: Max moves reached (draw/stalemate)
        - lps: No valid moves available
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
            # Check territory victory threshold (per RR-CANON-R062)
            # Territory threshold = floor(totalSpaces / 2) + 1
            total_spaces = self.board_size * self.board_size
            territory_threshold = total_spaces // 2 + 1  # 33 for 8x8, 181 for 19x19
            if self.territory_count[game_idx, winner].item() >= territory_threshold:
                return ("territory", None)

            # Check if opponent has no stacks (elimination)
            opponents_have_stacks = False
            for p in range(1, self.num_players + 1):
                if p != winner:
                    if (self.stack_owner[game_idx] == p).any():
                        opponents_have_stacks = True
                        break

            if not opponents_have_stacks:
                return ("ring_elimination", None)

            # Default to ring_elimination if winner but unclear reason
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

    A placement is valid on any empty position (stack_owner == 0).

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

    # Find all empty positions per game
    # empty_positions: (batch_size, board_size, board_size) bool
    empty_positions = (state.stack_owner == 0) & active_mask.view(-1, 1, 1)

    # Get indices of all empty positions
    game_idx, y_idx, x_idx = torch.where(empty_positions)

    total_moves = len(game_idx)

    # Count moves per game for indexing
    moves_per_game = empty_positions.view(batch_size, -1).sum(dim=1)
    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), moves_per_game[:-1]]),
        dim=0
    )

    return BatchMoves(
        game_idx=game_idx.int(),
        move_type=torch.full((total_moves,), MoveType.PLACEMENT, dtype=torch.int8, device=device),
        from_y=y_idx.int(),
        from_x=x_idx.int(),
        to_y=torch.zeros(total_moves, dtype=torch.int32, device=device),
        to_x=torch.zeros(total_moves, dtype=torch.int32, device=device),
        moves_per_game=moves_per_game.int(),
        move_offsets=move_offsets.int(),
        total_moves=total_moves,
        device=device,
    )


# =============================================================================
# Movement Move Generation (RR-CANON-R090-R092)
# =============================================================================


def _validate_paths_vectorized(
    state: BatchGameState,
    game_indices: torch.Tensor,
    from_positions: torch.Tensor,
    to_positions: torch.Tensor,
    players: torch.Tensor,
) -> torch.Tensor:
    """Validate movement paths in a semi-vectorized manner.

    Checks that no opponent stacks block the path from origin to destination.
    This is a hybrid approach: we iterate over moves but use tensor indexing
    for path cell lookups to reduce Python overhead.

    Args:
        state: BatchGameState
        game_indices: (N,) game index for each candidate move
        from_positions: (N, 2) [y, x] origin positions
        to_positions: (N, 2) [y, x] destination positions
        players: (N,) player number for each move

    Returns:
        Boolean tensor (N,) - True if path is valid (no opponent blocking)
    """
    device = state.device
    N = game_indices.shape[0]

    if N == 0:
        return torch.tensor([], dtype=torch.bool, device=device)

    valid = torch.ones(N, dtype=torch.bool, device=device)

    # Process in chunks for better memory efficiency
    # For each move, we need to check all cells along the path
    for i in range(N):
        g = game_indices[i].item()
        from_y, from_x = from_positions[i, 0].item(), from_positions[i, 1].item()
        to_y, to_x = to_positions[i, 0].item(), to_positions[i, 1].item()
        player = players[i].item()

        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        # Check each path cell (including destination)
        for step in range(1, dist + 1):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            cell_owner = state.stack_owner[g, check_y, check_x].item()

            # Opponent stack blocks the path
            if cell_owner != 0 and cell_owner != player:
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
    - Cannot pass through or land on opponent stacks (those are captures)
    - Can pass through/land on empty spaces or own stacks

    Implementation uses a two-phase approach:
    1. Generate all candidate moves based on position/height constraints
    2. Validate paths to filter out moves blocked by opponent stacks

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

    # 8 directions: (dy, dx)
    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    # Phase 1: Generate all candidate moves (without path validation)
    # This reduces Python loop overhead by deferring path checks
    candidate_game_idx = []
    candidate_from = []
    candidate_to = []
    candidate_player = []

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        player = state.current_player[g].item()

        # Find all stacks controlled by current player
        my_stacks = (state.stack_owner[g] == player)
        stack_positions = torch.nonzero(my_stacks, as_tuple=False)

        for pos_idx in range(stack_positions.shape[0]):
            from_y = stack_positions[pos_idx, 0].item()
            from_x = stack_positions[pos_idx, 1].item()
            stack_height = state.stack_height[g, from_y, from_x].item()

            if stack_height <= 0:
                continue

            # Try each direction
            for dy, dx in directions:
                # Move distance must be >= stack height
                for dist in range(stack_height, board_size):
                    to_y = from_y + dy * dist
                    to_x = from_x + dx * dist

                    # Check bounds
                    if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                        break

                    # Destination must be empty or own stack (not opponent)
                    # Opponent destination is a capture, not movement
                    dest_owner = state.stack_owner[g, to_y, to_x].item()
                    if dest_owner != 0 and dest_owner != player:
                        # This direction leads to capture territory
                        # But we still need to check shorter distances
                        continue

                    # Add as candidate (path validation deferred)
                    candidate_game_idx.append(g)
                    candidate_from.append([from_y, from_x])
                    candidate_to.append([to_y, to_x])
                    candidate_player.append(player)

    if not candidate_game_idx:
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

    # Convert candidates to tensors
    game_idx_t = torch.tensor(candidate_game_idx, dtype=torch.int64, device=device)
    from_t = torch.tensor(candidate_from, dtype=torch.int64, device=device)
    to_t = torch.tensor(candidate_to, dtype=torch.int64, device=device)
    player_t = torch.tensor(candidate_player, dtype=torch.int64, device=device)

    # Phase 2: Validate paths
    valid_mask = _validate_paths_vectorized(state, game_idx_t, from_t, to_t, player_t)

    # Filter to valid moves only
    valid_indices = torch.where(valid_mask)[0]

    if valid_indices.numel() == 0:
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

    # Extract valid moves
    valid_game_idx = game_idx_t[valid_indices].int()
    valid_from_y = from_t[valid_indices, 0].int()
    valid_from_x = from_t[valid_indices, 1].int()
    valid_to_y = to_t[valid_indices, 0].int()
    valid_to_x = to_t[valid_indices, 1].int()

    total_moves = valid_indices.numel()

    # Count moves per game using scatter_add
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
    - Capture by "overtaking": move onto opponent stack with equal or greater height
    - Move in straight line, distance >= stack height
    - Captures merge stacks (attacker rings on top)
    - Defender's top ring is eliminated

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

        # Find all stacks controlled by current player
        my_stacks = (state.stack_owner[g] == player)
        stack_positions = torch.nonzero(my_stacks, as_tuple=False)

        for pos_idx in range(stack_positions.shape[0]):
            from_y = stack_positions[pos_idx, 0].item()
            from_x = stack_positions[pos_idx, 1].item()
            my_height = state.stack_height[g, from_y, from_x].item()

            if my_height <= 0:
                continue

            for dy, dx in directions:
                # Move distance must be >= stack height
                for dist in range(my_height, board_size):
                    to_y = from_y + dy * dist
                    to_x = from_x + dx * dist

                    if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                        break

                    # Check path is clear (can pass through empty or own stacks)
                    path_clear = True
                    for step in range(1, dist):  # Don't check destination
                        check_y = from_y + dy * step
                        check_x = from_x + dx * step
                        cell_owner = state.stack_owner[g, check_y, check_x].item()

                        # Path blocked by opponent stack
                        if cell_owner != 0 and cell_owner != player:
                            path_clear = False
                            break

                    if not path_clear:
                        break

                    # Check destination for capture
                    dest_owner = state.stack_owner[g, to_y, to_x].item()
                    if dest_owner != 0 and dest_owner != player:
                        # Opponent stack - check if we can capture
                        dest_height = state.stack_height[g, to_y, to_x].item()
                        if my_height >= dest_height:
                            # Valid capture!
                            all_game_idx.append(g)
                            all_from_y.append(from_y)
                            all_from_x.append(from_x)
                            all_to_y.append(to_y)
                            all_to_x.append(to_x)
                        # Cannot continue past opponent stack
                        break

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

    Args:
        state: Current batch game state
        game_idx: Game index in batch
        from_y: Row position of the stack to check captures from
        from_x: Column position of the stack to check captures from

    Returns:
        List of (to_y, to_x) destination positions for valid captures
    """
    board_size = state.board_size
    player = state.current_player[game_idx].item()

    # Verify we control this stack
    stack_owner = state.stack_owner[game_idx, from_y, from_x].item()
    if stack_owner != player:
        return []

    my_height = state.stack_height[game_idx, from_y, from_x].item()
    if my_height <= 0:
        return []

    directions = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    captures = []

    for dy, dx in directions:
        # Move distance must be >= stack height
        for dist in range(my_height, board_size):
            to_y = from_y + dy * dist
            to_x = from_x + dx * dist

            if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                break

            # Check path is clear (can pass through empty or own stacks)
            path_clear = True
            for step in range(1, dist):  # Don't check destination
                check_y = from_y + dy * step
                check_x = from_x + dx * step
                cell_owner = state.stack_owner[game_idx, check_y, check_x].item()

                # Path blocked by opponent stack
                if cell_owner != 0 and cell_owner != player:
                    path_clear = False
                    break

            if not path_clear:
                break

            # Check destination for capture
            dest_owner = state.stack_owner[game_idx, to_y, to_x].item()
            if dest_owner != 0 and dest_owner != player:
                # Opponent stack - check if we can capture
                dest_height = state.stack_height[game_idx, to_y, to_x].item()
                if my_height >= dest_height:
                    # Valid capture!
                    captures.append((to_y, to_x))
                # Cannot continue past opponent stack
                break

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

    Per RR-CANON-R100-R103:
    - Attacker moves onto defender stack
    - Defender's top ring is eliminated
    - Stacks merge (attacker on top)
    - Path markers are flipped to attacker's color

    Args:
        state: BatchGameState to modify
        game_idx: Game index in batch
        from_y, from_x: Origin position
        to_y, to_x: Destination position

    Returns:
        (new_y, new_x) landing position for potential chain continuation
    """
    player = state.current_player[game_idx].item()
    mc = state.move_count[game_idx].item()

    # Record in history
    if mc < state.max_history_moves:
        state.move_history[game_idx, mc, 0] = MoveType.CAPTURE
        state.move_history[game_idx, mc, 1] = player
        state.move_history[game_idx, mc, 2] = from_y
        state.move_history[game_idx, mc, 3] = from_x
        state.move_history[game_idx, mc, 4] = to_y
        state.move_history[game_idx, mc, 5] = to_x

    # Get stack info
    attacker_height = state.stack_height[game_idx, from_y, from_x].item()
    defender_height = state.stack_height[game_idx, to_y, to_x].item()
    defender_owner = state.stack_owner[game_idx, to_y, to_x].item()

    # Process markers along path
    dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
    dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
    dist = max(abs(to_y - from_y), abs(to_x - from_x))

    for step in range(1, dist):
        check_y = from_y + dy * step
        check_x = from_x + dx * step
        marker_owner = state.marker_owner[game_idx, check_y, check_x].item()
        if marker_owner != 0 and marker_owner != player:
            state.marker_owner[game_idx, check_y, check_x] = player

    # Eliminate defender's top ring
    defender_new_height = max(0, defender_height - 1)

    # Track elimination
    if defender_owner > 0:
        current_elim = state.eliminated_rings[game_idx, defender_owner].item()
        state.eliminated_rings[game_idx, defender_owner] = current_elim + 1

    # Merge stacks
    if defender_new_height > 0:
        new_height = attacker_height + defender_new_height
        state.stack_height[game_idx, to_y, to_x] = new_height
    else:
        state.stack_height[game_idx, to_y, to_x] = attacker_height

    state.stack_owner[game_idx, to_y, to_x] = player

    # Clear origin
    state.stack_height[game_idx, from_y, from_x] = 0
    state.stack_owner[game_idx, from_y, from_x] = 0

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
      may choose recovery over placement (RR-CANON-R110).
    - Recovery slides a marker to an adjacent empty cell
    - "Line mode": slide completes a line of markers (preferred)
    - "Fallback mode": any adjacent slide if no line possible (costs 1 buried ring)

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

    # 8 directions for sliding (Moore neighborhood)
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

        # Check recovery eligibility per RR-CANON-R110:
        # 1. No controlled stacks
        has_stacks = (state.stack_owner[g] == player).any().item()
        if has_stacks:
            continue

        # 2. Has markers on board
        my_markers = (state.marker_owner[g] == player)
        marker_positions = torch.nonzero(my_markers, as_tuple=False)
        if marker_positions.shape[0] == 0:
            continue

        # 3. Has buried rings (can afford recovery cost)
        buried_rings = state.buried_rings[g, player].item()
        if buried_rings <= 0:
            continue

        # Player is eligible for recovery - generate slide moves
        # For simplified GPU implementation, we generate all adjacent slides
        # (both line-completing and fallback moves are valid)
        for pos_idx in range(marker_positions.shape[0]):
            from_y = marker_positions[pos_idx, 0].item()
            from_x = marker_positions[pos_idx, 1].item()

            for dy, dx in directions:
                to_y = from_y + dy
                to_x = from_x + dx

                # Check bounds
                if not (0 <= to_y < board_size and 0 <= to_x < board_size):
                    continue

                # Target must be empty (no stack, no marker, no territory)
                if state.stack_owner[g, to_y, to_x].item() != 0:
                    continue
                if state.marker_owner[g, to_y, to_x].item() != 0:
                    continue
                if state.territory_owner[g, to_y, to_x].item() != 0:
                    continue

                # Valid recovery slide!
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
    # Get selected moves for each game
    # move_indices[g] is the local move index for game g
    # We need to convert to global index: move_offsets[g] + move_indices[g]

    batch_size = state.batch_size
    active_mask = state.get_active_mask()

    for g in range(batch_size):
        if not active_mask[g]:
            continue

        if moves.moves_per_game[g] == 0:
            continue

        # Get the selected move for this game
        local_idx = move_indices[g].item()
        if local_idx >= moves.moves_per_game[g]:
            continue

        global_idx = moves.move_offsets[g] + local_idx

        y = moves.from_y[global_idx].item()
        x = moves.from_x[global_idx].item()
        player = state.current_player[g].item()
        move_type = moves.move_type[global_idx].item()

        # Record move in history before incrementing move_count
        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            # Format: [move_type, player, from_y, from_x, to_y, to_x]
            # For placement, from and to are same position
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = y
            state.move_history[g, move_idx, 3] = x
            state.move_history[g, move_idx, 4] = y  # to_y same as from for placement
            state.move_history[g, move_idx, 5] = x  # to_x same as from for placement

        # Apply placement
        state.stack_owner[g, y, x] = player
        state.stack_height[g, y, x] = 1
        state.rings_in_hand[g, player] -= 1

        # Advance turn
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

        # Record move in history
        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

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

        # Advance turn
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

        # Record move in history
        move_idx = state.move_count[g].item()
        if move_idx < state.max_history_moves:
            state.move_history[g, move_idx, 0] = move_type
            state.move_history[g, move_idx, 1] = player
            state.move_history[g, move_idx, 2] = from_y
            state.move_history[g, move_idx, 3] = from_x
            state.move_history[g, move_idx, 4] = to_y
            state.move_history[g, move_idx, 5] = to_x

        # Get attacking stack info
        attacker_height = state.stack_height[g, from_y, from_x].item()

        # Get defender info
        defender_owner = state.stack_owner[g, to_y, to_x].item()
        defender_height = state.stack_height[g, to_y, to_x].item()

        # Process markers along path (flip opposing markers)
        dy = 0 if to_y == from_y else (1 if to_y > from_y else -1)
        dx = 0 if to_x == from_x else (1 if to_x > from_x else -1)
        dist = max(abs(to_y - from_y), abs(to_x - from_x))

        for step in range(1, dist):
            check_y = from_y + dy * step
            check_x = from_x + dx * step
            marker_owner = state.marker_owner[g, check_y, check_x].item()
            if marker_owner != 0 and marker_owner != player:
                state.marker_owner[g, check_y, check_x] = player

        # Eliminate defender's top ring
        state.eliminated_rings[g, player] += 1

        # Clear origin
        state.stack_owner[g, from_y, from_x] = 0
        state.stack_height[g, from_y, from_x] = 0

        # Merge stacks at destination
        # Defender loses 1 ring (eliminated), attacker stacks on top
        new_height = attacker_height + defender_height - 1
        state.stack_owner[g, to_y, to_x] = player
        state.stack_height[g, to_y, to_x] = min(5, new_height)

        # Place marker for attacker (capture marker)
        state.marker_owner[g, to_y, to_x] = player

        # Advance turn
        state.move_count[g] += 1
        state.current_player[g] = (player % state.num_players) + 1


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

    # 4 directions to check for lines: horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for g in range(batch_size):
        if not game_mask[g]:
            continue

        # Per RR-CANON-R120: Lines are formed by MARKERS, not stacks
        player_markers = (state.marker_owner[g] == player) & (state.stack_owner[g] == 0)

        # Track which positions have been assigned to a line
        assigned = set()

        # Check each direction for lines
        for dy, dx in directions:
            for start_y in range(board_size):
                for start_x in range(board_size):
                    if (start_y, start_x) in assigned:
                        continue
                    if not player_markers[start_y, start_x]:
                        continue

                    # Trace line in this direction
                    line_positions = [(start_y, start_x)]
                    y, x = start_y + dy, start_x + dx

                    while 0 <= y < board_size and 0 <= x < board_size:
                        if player_markers[y, x] and (y, x) not in assigned:
                            line_positions.append((y, x))
                            y, x = y + dy, x + dx
                        else:
                            break

                    # If line meets required length, record it
                    if len(line_positions) >= required_length:
                        for pos in line_positions:
                            assigned.add(pos)

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

    Args:
        state: BatchGameState to modify
        game_idx: Game index
        player: Player performing elimination

    Returns:
        True if elimination was performed, False if no eligible stack found
    """
    board_size = state.board_size

    for y in range(board_size):
        for x in range(board_size):
            if state.stack_owner[game_idx, y, x].item() == player:
                stack_height = state.stack_height[game_idx, y, x].item()
                if stack_height > 0:
                    # Eliminate one ring from top
                    state.stack_height[game_idx, y, x] = stack_height - 1
                    state.eliminated_rings[game_idx, player] += 1

                    # If stack is now empty, clear ownership
                    if stack_height - 1 == 0:
                        state.stack_owner[game_idx, y, x] = 0

                    return True
    return False


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

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
        option2_probability: Probability of choosing Option 2 for overlength lines
                            (default 0.3 - prefer Option 1 for more territory)
    """
    batch_size = state.batch_size
    board_size = state.board_size

    if game_mask is None:
        game_mask = state.get_active_mask()

    required_length = get_required_line_length(board_size, state.num_players)

    for p in range(1, state.num_players + 1):
        lines_with_meta = detect_lines_with_metadata(state, p, game_mask)

        for g in range(batch_size):
            if not game_mask[g]:
                continue

            game_lines = lines_with_meta[g]
            if not game_lines:
                continue

            # Process each line individually per RR-CANON-R121
            for line in game_lines:
                positions_to_collapse = line.positions

                if line.is_overlength:
                    # Per RR-CANON-R122 Case 2: Overlength line - Option 1 or Option 2
                    # Use probabilistic selection for training variety
                    use_option2 = (torch.rand(1, device=state.device).item() < option2_probability)

                    if use_option2:
                        # Option 2: Collapse exactly required_length markers, NO elimination
                        # Per RR-CANON-R122: Player can choose which markers to collapse
                        # For training variety, randomly select which subset to collapse
                        all_positions = line.positions
                        if len(all_positions) > required_length:
                            # Randomly select which required_length positions to collapse
                            indices = torch.randperm(len(all_positions), device=state.device)[:required_length]
                            indices = indices.sort().values  # Keep line order for determinism
                            positions_to_collapse = [all_positions[i] for i in indices.tolist()]
                        else:
                            positions_to_collapse = all_positions[:required_length]
                        # No elimination cost for Option 2
                    else:
                        # Option 1: Collapse ALL markers, pay one ring elimination
                        positions_to_collapse = line.positions
                        # Check if player can pay elimination cost
                        player_stacks = (state.stack_owner[g] == p)
                        if player_stacks.any().item():
                            _eliminate_one_ring_from_any_stack(state, g, p)
                        # If no stack available, still collapse (per RR-CANON-R122 interpretation B)
                else:
                    # Exact-length line: Must pay elimination cost
                    # Per RR-CANON-R122 Case 1: len == requiredLen
                    player_stacks = (state.stack_owner[g] == p)
                    if player_stacks.any().item():
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

    for y in range(board_size):
        for x in range(board_size):
            if (y, x) in excluded_positions:
                continue

            owner = state.stack_owner[game_idx, y, x].item()
            height = state.stack_height[game_idx, y, x].item()

            if owner != player or height <= 0:
                continue

            # Per RR-CANON-R145: Height-1 standalone rings are NOT eligible
            # Eligible targets are:
            # (i) Mixed-colour stack - player controls but other colors buried
            # (ii) Single-colour stack of height > 1
            #
            # In simplified GPU representation, we don't track ring colors per position,
            # only controlling player. So we use height > 1 as the eligibility criterion.
            # This is a simplification that may need refinement for full parity.
            if height > 1:
                return (y, x, height)  # Return full height as "cap" (simplified)

    return None


def _find_all_regions(
    state: BatchGameState,
    game_idx: int,
) -> List[Set[Tuple[int, int]]]:
    """Find all maximal connected regions of non-collapsed cells (R140).

    Uses union-find/BFS to discover all connected regions of non-collapsed cells.
    A region is a maximal set of non-collapsed cells where each cell is connected
    to at least one other cell in the region via 4-connectivity.

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

    visited = [[False] * board_size for _ in range(board_size)]
    regions = []

    for start_y in range(board_size):
        for start_x in range(board_size):
            if visited[start_y][start_x] or not non_collapsed[start_y, start_x]:
                continue

            # BFS to find connected region
            region = set()
            queue = [(start_y, start_x)]
            visited[start_y][start_x] = True

            while queue:
                y, x = queue.pop(0)
                region.add((y, x))

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < board_size and 0 <= nx < board_size:
                        if not visited[ny][nx] and non_collapsed[ny, nx]:
                            visited[ny][nx] = True
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

    # Find all non-collapsed cells outside the region
    non_collapsed = ~state.is_collapsed[g].cpu().numpy()

    outside_non_collapsed = set()
    for y in range(board_size):
        for x in range(board_size):
            if (y, x) not in region and non_collapsed[y, x]:
                outside_non_collapsed.add((y, x))

    # If no cells outside region, region spans entire non-collapsed board
    # This is a degenerate case - not physically disconnected in meaningful sense
    if not outside_non_collapsed:
        return (False, None)

    # Try to reach from region to outside using only:
    # - Non-collapsed cells (stacks, markers, empty)
    # Track which player's markers we cross

    # Get marker ownership for all cells
    marker_owner = state.marker_owner[g].cpu().numpy() if hasattr(state, 'marker_owner') else None
    stack_owner = state.stack_owner[g].cpu().numpy()

    # For each cell in region, try BFS to reach outside
    # Blocking cells are: collapsed spaces, board edge
    # We need to track if all blocking markers belong to single player

    blocking_marker_players = set()

    # BFS from region boundary
    region_boundary = set()
    for y, x in region:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < board_size and 0 <= nx < board_size:
                if (ny, nx) not in region:
                    region_boundary.add((y, x))
                    break
            else:
                # Edge of board - this boundary touches edge
                region_boundary.add((y, x))

    # Check what separates region from outside
    # Try to reach outside_non_collapsed from region
    visited = set(region)
    queue = list(region_boundary)
    can_reach_outside = False

    for y, x in queue:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            # Off-board - counts as barrier
            if not (0 <= ny < board_size and 0 <= nx < board_size):
                continue

            if (ny, nx) in visited:
                continue

            # Collapsed space - counts as barrier
            if not non_collapsed[ny, nx]:
                continue

            # Non-collapsed cell outside region
            if (ny, nx) in outside_non_collapsed:
                # Can we reach it directly, or is there a marker barrier?
                # Check if this cell is a marker
                cell_marker_owner = 0
                if marker_owner is not None:
                    cell_marker_owner = marker_owner[ny, nx]

                if cell_marker_owner > 0:
                    blocking_marker_players.add(cell_marker_owner)
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

    Args:
        state: BatchGameState
        game_idx: Game index
        region: Set of (y, x) positions in the region

    Returns:
        True if region is color-disconnected (eligible for processing)
    """
    g = game_idx
    board_size = state.board_size

    # Compute ActiveColors: players with at least one ring on the board
    # A player has rings if they control any stack (stack_owner == player and height > 0)
    # or have rings buried in mixed stacks (simplified: check stack_owner > 0)
    active_colors = set()
    for y in range(board_size):
        for x in range(board_size):
            owner = state.stack_owner[g, y, x].item()
            height = state.stack_height[g, y, x].item()
            if owner > 0 and height > 0:
                active_colors.add(owner)

    # If no active colors (empty board), no territory processing possible
    if not active_colors:
        return False

    # Compute RegionColors: players controlling stacks in the region
    region_colors = set()
    for y, x in region:
        owner = state.stack_owner[g, y, x].item()
        height = state.stack_height[g, y, x].item()
        if owner > 0 and height > 0:
            region_colors.add(owner)

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
                        # Remove any stacks
                        stack_height = state.stack_height[g, y, x].item()
                        if stack_height > 0:
                            state.eliminated_rings[g, player] += stack_height
                            state.stack_height[g, y, x] = 0
                            state.stack_owner[g, y, x] = 0

                        # Collapse the cell
                        if not state.is_collapsed[g, y, x]:
                            state.is_collapsed[g, y, x] = True
                            state.territory_owner[g, y, x] = player
                            territory_count += 1

                    # 2. Collapse border markers of single border color B (if applicable)
                    if border_player is not None and hasattr(state, 'marker_owner'):
                        # Find and collapse border markers
                        for y in range(board_size):
                            for x in range(board_size):
                                if state.marker_owner[g, y, x].item() == border_player:
                                    # Check if this marker is on the boundary of region
                                    is_boundary = False
                                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        ny, nx = y + dy, x + dx
                                        if 0 <= ny < board_size and 0 <= nx < board_size:
                                            if (ny, nx) in region:
                                                is_boundary = True
                                                break

                                    if is_boundary and not state.is_collapsed[g, y, x]:
                                        state.is_collapsed[g, y, x] = True
                                        state.territory_owner[g, y, x] = player
                                        state.marker_owner[g, y, x] = 0
                                        territory_count += 1

                    # 4. Mandatory self-elimination (eliminate cap)
                    state.stack_height[g, cap_y, cap_x] = 0
                    state.stack_owner[g, cap_y, cap_x] = 0
                    state.eliminated_rings[g, player] += cap_height

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

    # Canonical victory thresholds (RR-CANON-R061/R062).
    # Keep this in sync with app.rules.core.BOARD_CONFIGS.
    from app.models import BoardType
    from app.rules.core import get_territory_victory_threshold, get_victory_threshold

    board_type_map = {
        8: BoardType.SQUARE8,
        19: BoardType.SQUARE19,
        13: BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(board_size, BoardType.SQUARE8)
    territory_victory_threshold = get_territory_victory_threshold(board_type)
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

        # === LINE POTENTIAL METRICS ===
        # Track 2/3/4-in-a-row patterns
        two_in_row = torch.zeros(batch_size, device=device)
        three_in_row = torch.zeros(batch_size, device=device)
        four_in_row = torch.zeros(batch_size, device=device)
        connected_neighbors = torch.zeros(batch_size, device=device)
        gap_potential = torch.zeros(batch_size, device=device)

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, D1, D2

        for g in range(batch_size):
            my_stacks = player_stacks[g]
            t2, t3, t4, conn, gaps = 0.0, 0.0, 0.0, 0.0, 0.0

            for dy, dx in directions:
                for start_y in range(board_size):
                    for start_x in range(board_size):
                        if not my_stacks[start_y, start_x]:
                            continue

                        # Count consecutive stacks
                        count = 1
                        y, x = start_y + dy, start_x + dx
                        while 0 <= y < board_size and 0 <= x < board_size:
                            if my_stacks[y, x]:
                                count += 1
                                conn += 1.0  # Connected neighbor
                                y, x = y + dy, x + dx
                            else:
                                # Check for gap (empty followed by our stack)
                                y2, x2 = y + dy, x + dx
                                if (0 <= y2 < board_size and 0 <= x2 < board_size and
                                    state.stack_owner[g, y, x] == 0 and my_stacks[y2, x2]):
                                    gaps += 0.5  # Gap that could be filled
                                break

                        if count == 2:
                            t2 += 1.0
                        elif count == 3:
                            t3 += 1.0
                        elif count >= 4:
                            t4 += 1.0

            two_in_row[g] = t2
            three_in_row[g] = t3
            four_in_row[g] = t4
            connected_neighbors[g] = conn
            gap_potential[g] = gaps

        # === OPPONENT THREAT METRICS ===
        opponent_threat = torch.zeros(batch_size, device=device)
        opponent_victory_threat = torch.zeros(batch_size, device=device)
        blocking_score = torch.zeros(batch_size, device=device)

        for opponent in range(1, num_players + 1):
            if opponent == p:
                continue

            opp_stacks = (state.stack_owner == opponent)
            opp_territory = state.territory_count[:, opponent].float()
            opp_eliminated = state.eliminated_rings[:, opponent].float()

            # Victory proximity threat
            opp_territory_progress = opp_territory / territory_victory_threshold
            opp_elim_progress = opp_eliminated / ring_victory_threshold
            opponent_victory_threat += torch.max(opp_territory_progress, opp_elim_progress)

            for g in range(batch_size):
                opp_g = opp_stacks[g]
                threat = 0.0
                block = 0.0

                for dy, dx in directions:
                    for start_y in range(board_size):
                        for start_x in range(board_size):
                            if not opp_g[start_y, start_x]:
                                continue

                            # Count opponent consecutive stacks
                            count = 1
                            y, x = start_y + dy, start_x + dx
                            while 0 <= y < board_size and 0 <= x < board_size:
                                if opp_g[y, x]:
                                    count += 1
                                    y, x = y + dy, x + dx
                                else:
                                    break

                            if count >= 2:
                                threat += count * 0.5

                            # Check if we're blocking this line
                            if player_stacks[g, start_y, start_x]:
                                block += 1.0

                opponent_threat[g] += threat
                blocking_score[g] += block

        # === VULNERABILITY METRICS ===
        # Check how many of our stacks could be captured
        vulnerability = torch.zeros(batch_size, device=device)
        blocked_stacks = torch.zeros(batch_size, device=device)

        for g in range(batch_size):
            ps = player_stacks[g]
            vuln = 0.0
            blocked = 0.0
            for y in range(board_size):
                for x in range(board_size):
                    if not ps[y, x]:
                        continue
                    my_height = state.stack_height[g, y, x].item()
                    # Check all neighbors for taller opponent stacks
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < board_size and 0 <= nx < board_size:
                            neighbor_owner = state.stack_owner[g, ny, nx].item()
                            if neighbor_owner != 0 and neighbor_owner != p:
                                neighbor_height = state.stack_height[g, ny, nx].item()
                                if neighbor_height >= my_height:
                                    vuln += 1.0
                    # Check if stack is blocked (all directions occupied)
                    adj_count = 0
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < board_size and 0 <= nx < board_size:
                            if state.stack_owner[g, ny, nx] != 0:
                                adj_count += 1
                    if adj_count >= 3:
                        blocked += 1.0

            vulnerability[g] = vuln
            blocked_stacks[g] = blocked

        # === OVERTAKE POTENTIAL ===
        # Count enemy stacks we could capture (our taller stacks adjacent to theirs)
        overtake_potential = torch.zeros(batch_size, device=device)
        for g in range(batch_size):
            ps = player_stacks[g]
            overtake = 0.0
            for y in range(board_size):
                for x in range(board_size):
                    if not ps[y, x]:
                        continue
                    my_height = state.stack_height[g, y, x].item()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < board_size and 0 <= nx < board_size:
                            neighbor_owner = state.stack_owner[g, ny, nx].item()
                            if neighbor_owner != 0 and neighbor_owner != p:
                                neighbor_height = state.stack_height[g, ny, nx].item()
                                if my_height > neighbor_height:
                                    overtake += 1.0
            overtake_potential[g] = overtake

        # === TERRITORY METRICS ===
        # Territory closure: territory cells adjacent to our stacks
        territory_closure = torch.zeros(batch_size, device=device)
        territory_safety = torch.zeros(batch_size, device=device)
        for g in range(batch_size):
            closure = 0.0
            safety = 0.0
            for y in range(board_size):
                for x in range(board_size):
                    if state.territory_owner[g, y, x] == p:
                        # Check adjacency to our stacks
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < board_size and 0 <= nx < board_size:
                                if state.stack_owner[g, ny, nx] == p:
                                    closure += 0.5
                                    safety += 0.5
            territory_closure[g] = closure
            territory_safety[g] = safety

        # === STACK MOBILITY ===
        # Per-stack movement freedom (simplified)
        stack_mobility = stack_count * 3.0  # Avg 3 directions per stack

        # === VICTORY PROXIMITY ===
        # How close to winning (normalized 0-1)
        territory_progress = territory / territory_victory_threshold
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
        """
        self.batch_size = batch_size
        self.board_size = board_size
        self.num_players = num_players

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
        )

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

        # Use default weights if not provided
        if weights_list is None:
            weights_list = [self._default_weights()] * self.batch_size

        move_num = 0

        while self.state.count_active() > 0 and move_num < max_moves:
            # Generate and apply moves for all active games
            self._step_games(weights_list)

            move_num += 1

            if callback:
                callback(move_num, self.state)

            # Check for game completion
            self._check_victory_conditions()

        # Mark remaining active games as draws (max moves)
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

        device = self.device
        batch_size = self.batch_size

        # Process each phase type separately based on current phase
        # Games may be in different phases, so we handle each group

        # PHASE: RING_PLACEMENT (0)
        placement_mask = active_mask & (self.state.current_phase == GamePhase.RING_PLACEMENT)
        if placement_mask.any():
            self._step_placement_phase(placement_mask, weights_list)

        # PHASE: MOVEMENT (1)
        movement_mask = active_mask & (self.state.current_phase == GamePhase.MOVEMENT)
        if movement_mask.any():
            self._step_movement_phase(movement_mask, weights_list)

        # PHASE: LINE_PROCESSING (2)
        line_mask = active_mask & (self.state.current_phase == GamePhase.LINE_PROCESSING)
        if line_mask.any():
            self._step_line_phase(line_mask)

        # PHASE: TERRITORY_PROCESSING (3)
        territory_mask = active_mask & (self.state.current_phase == GamePhase.TERRITORY_PROCESSING)
        if territory_mask.any():
            self._step_territory_phase(territory_mask)

        # PHASE: END_TURN (4)
        end_turn_mask = active_mask & (self.state.current_phase == GamePhase.END_TURN)
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
                y = moves.from_y[idx].item()
                x = moves.from_x[idx].item()
                gpu_positions.append((y, x))

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

            # Extract GPU movement moves
            move_start = movement_moves.move_offsets[g].item()
            move_count = movement_moves.moves_per_game[g].item()

            gpu_movement = []
            for i in range(move_count):
                idx = move_start + i
                from_y = movement_moves.from_y[idx].item()
                from_x = movement_moves.from_x[idx].item()
                to_y = movement_moves.to_y[idx].item()
                to_x = movement_moves.to_x[idx].item()
                gpu_movement.append(((from_y, from_x), (to_y, to_x)))

            # Extract GPU capture moves
            cap_start = capture_moves.move_offsets[g].item()
            cap_count = capture_moves.moves_per_game[g].item()

            gpu_captures = []
            for i in range(cap_count):
                idx = cap_start + i
                from_y = capture_moves.from_y[idx].item()
                from_x = capture_moves.from_x[idx].item()
                to_y = capture_moves.to_y[idx].item()
                to_x = capture_moves.to_x[idx].item()
                gpu_captures.append(((from_y, from_x), (to_y, to_x)))

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

    def _step_placement_phase(
        self,
        mask: torch.Tensor,
        weights_list: List[Dict[str, float]],
    ) -> None:
        """Handle RING_PLACEMENT phase for games in mask.

        Refactored 2025-12-11 for vectorized player-based indexing.
        """
        # Check which games have rings to place (vectorized)
        # rings_in_hand shape: (batch_size, num_players + 1)
        # current_player shape: (batch_size,)
        # Use gather to get rings_in_hand[g, current_player[g]] for each game
        current_players = self.state.current_player  # (batch_size,)
        rings_for_current_player = torch.gather(
            self.state.rings_in_hand,
            dim=1,
            index=current_players.unsqueeze(1).long()
        ).squeeze(1)

        has_rings = mask & (rings_for_current_player > 0)
        games_with_rings = has_rings

        # Games WITH rings: generate and apply placement moves
        if games_with_rings.any():
            moves = generate_placement_moves_batch(self.state, games_with_rings)

            # Shadow validation: validate move generation against CPU
            self._validate_placement_moves_sample(moves, games_with_rings)

            if moves.total_moves > 0:
                selected = self._select_best_moves(moves, weights_list, games_with_rings)
                apply_placement_moves_batch(self.state, selected, moves)

        # After placement, advance to MOVEMENT phase
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

            # Apply recovery moves using vectorized selection
            if games_with_recovery.any():
                selected_recovery = select_moves_vectorized(
                    recovery_moves, games_with_recovery, self.board_size
                )
                apply_recovery_moves_vectorized(
                    self.state, selected_recovery, recovery_moves, games_with_recovery
                )

            # Check for stalemate in games that had no stacks AND no recovery moves
            if games_no_recovery.any():
                self._check_stalemate(games_no_recovery)

        # Games WITH stacks: generate movement and capture moves
        if games_with_stacks.any():
            # Generate non-capture movement moves
            movement_moves = generate_movement_moves_batch(self.state, games_with_stacks)

            # Generate capture moves
            capture_moves = generate_capture_moves_batch(self.state, games_with_stacks)

            # Shadow validation: validate move generation against CPU
            self._validate_movement_moves_sample(movement_moves, capture_moves, games_with_stacks)

            # Identify which games have captures (prefer captures per RR-CANON)
            # Per RR-CANON-R103: After executing a capture, if additional legal captures
            # exist from the new landing position, the chain MUST continue.
            games_with_captures = games_with_stacks & (capture_moves.moves_per_game > 0)
            games_movement_only = games_with_stacks & (capture_moves.moves_per_game == 0) & (movement_moves.moves_per_game > 0)

            # Apply capture moves with chain capture support (RR-CANON-R103)
            if games_with_captures.any():
                selected_captures = select_moves_vectorized(
                    capture_moves, games_with_captures, self.board_size
                )
                apply_capture_moves_vectorized(
                    self.state, selected_captures, capture_moves, games_with_captures
                )

                # Track landing positions for chain capture continuation
                # For each game that had a capture, check if more captures are available
                game_indices = torch.where(games_with_captures)[0]

                for g in game_indices.tolist():
                    local_idx = selected_captures[g].item()
                    if local_idx < 0:
                        continue

                    global_idx = (capture_moves.move_offsets[g] + local_idx).item()
                    if global_idx >= capture_moves.total_moves:
                        continue

                    # Get landing position from the capture we just applied
                    landing_y = capture_moves.to_y[global_idx].item()
                    landing_x = capture_moves.to_x[global_idx].item()

                    # Chain capture loop: continue capturing from landing position
                    # per RR-CANON-R103 (mandatory chain captures)
                    max_chain_depth = 10  # Safety limit to prevent infinite loops
                    chain_depth = 0

                    while chain_depth < max_chain_depth:
                        chain_depth += 1

                        # Generate captures from current landing position only
                        chain_captures = generate_chain_capture_moves_from_position(
                            self.state, g, landing_y, landing_x
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
                            self.state, g, landing_y, landing_x, to_y, to_x
                        )

                        # Update landing position for next iteration
                        landing_y, landing_x = new_y, new_x

            # Apply movement moves for games without captures
            if games_movement_only.any():
                selected_movements = select_moves_vectorized(
                    movement_moves, games_movement_only, self.board_size
                )
                apply_movement_moves_vectorized(
                    self.state, selected_movements, movement_moves, games_movement_only
                )

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

        Refactored 2025-12-11 for batch efficiency:
        - Call detect_lines_batch once per player (not per game)
        - Accumulate results across all players

        Args:
            mask: Games to check

        Returns:
            Boolean tensor indicating which games have new lines
        """
        has_new_lines = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Check each player's lines across all masked games at once
        for p in range(1, self.num_players + 1):
            # detect_lines_batch returns List[List[positions]] for all games
            lines_by_game = detect_lines_batch(self.state, p, mask)

            # Check which games have lines for this player
            for g in range(self.batch_size):
                if mask[g] and lines_by_game[g]:
                    has_new_lines[g] = True

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
            state.eliminated_rings[g, defender_owner] += defender_eliminated

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
            # Check for lines for the current player
            lines = detect_lines_batch(self.state, player, single_game_mask)

            if not lines[g]:
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
        - RR-CANON-R171: Territory-control victory (territorySpaces >= territoryVictoryThreshold)
        - RR-CANON-R172: Last-player-standing (only player with real actions)

        Victory thresholds per RR-CANON-R061/R062:
        - victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
        - territoryVictoryThreshold = floor(totalSpaces / 2) + 1
        """
        active_mask = self.state.get_active_mask()

        # Canonical thresholds depend on board type and player count (RR-CANON-R061/R062).
        from app.models import BoardType
        from app.rules.core import get_territory_victory_threshold, get_victory_threshold

        board_type_map = {
            8: BoardType.SQUARE8,
            19: BoardType.SQUARE19,
            13: BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(self.board_size, BoardType.SQUARE8)
        ring_elimination_threshold = get_victory_threshold(board_type, self.num_players)
        territory_victory_threshold = get_territory_victory_threshold(board_type)

        for p in range(1, self.num_players + 1):
            # Check ring-elimination victory (RR-CANON-R170)
            # Player wins when they've eliminated >= victoryThreshold rings
            ring_elimination_victory = self.state.eliminated_rings[:, p] >= ring_elimination_threshold

            # Check territory victory (RR-CANON-R171)
            territory_victory = self.state.territory_count[:, p] >= territory_victory_threshold

            # Check last-player-standing (RR-CANON-R172)
            # Player wins if they're the only one with controlled stacks or rings in hand
            has_stacks = (self.state.stack_owner == p).any(dim=(1, 2))
            has_rings_in_hand = self.state.rings_in_hand[:, p] > 0
            has_turn_material = has_stacks | has_rings_in_hand

            # Check if any other player has turn material
            others_have_turn_material = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            for other_p in range(1, self.num_players + 1):
                if other_p != p:
                    other_has_stacks = (self.state.stack_owner == other_p).any(dim=(1, 2))
                    other_has_rings = self.state.rings_in_hand[:, other_p] > 0
                    others_have_turn_material |= (other_has_stacks | other_has_rings)

            # Last player standing: this player has material, no others do
            last_player_standing = has_turn_material & ~others_have_turn_material

            # Update winners - any victory condition triggers win
            victory_mask = active_mask & (ring_elimination_victory | territory_victory | last_player_standing)
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
