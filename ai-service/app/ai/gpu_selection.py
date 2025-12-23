"""GPU move selection utilities for parallel games.

This module provides vectorized move selection functions for the GPU parallel
games system. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R20 refactoring.

Key functions:
- select_moves_vectorized: Fast random sampling with center bias
- select_moves_heuristic: Feature-based scoring for better move quality
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .gpu_game_types import MoveType

if TYPE_CHECKING:
    from .gpu_move_generation import BatchMoves
    from .gpu_parallel_games import BatchGameState


def select_moves_vectorized(
    moves: BatchMoves,
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

    # NOTE: Do NOT clamp to 0 here! Games with no valid selection should keep -1.
    # The apply functions check for valid indices >= 0.
    # Bug fixed 2025-12-20: clamping -1 to 0 caused movements to be recorded
    # for games that should only have captures (or vice versa).

    return selected


# Key mappings from full profile format to simplified keys
# Full profiles use WEIGHT_* keys, simplified uses lowercase keys
_FULL_PROFILE_KEY_MAP = {
    "center": "WEIGHT_CENTER_CONTROL",
    "capture_value": "WEIGHT_OVERTAKE_POTENTIAL",
    "adjacency": "WEIGHT_ADJACENCY",
    "line_potential": "WEIGHT_LINE_POTENTIAL",
    "noise": None,  # No equivalent in full profiles
}


def _get_weight(weights: dict, key: str, default: float) -> float:
    """Get weight from dict supporting both simplified and full profile formats."""
    # Try simplified key first
    if key in weights:
        return weights[key]
    # Map to full profile key
    full_key = _FULL_PROFILE_KEY_MAP.get(key)
    if full_key and full_key in weights:
        return weights[full_key]
    return default


def select_moves_heuristic(
    moves: BatchMoves,
    state: BatchGameState,
    active_mask: torch.Tensor,
    weights_list: list[dict[str, float]] | None = None,
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
        weights_list: Optional per-game weights list for feature importance
                     (one dict per game, already resolved to current player's weights)
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
    default_weights = {
        "center": 3.0,
        "capture_value": 5.0,
        "adjacency": 2.0,
        "line_potential": 1.5,
        "noise": 1.0,
    }

    # Build per-game weight tensors for vectorized scoring
    game_idx = moves.game_idx.long()

    if weights_list is not None:
        # Per-game weights: create tensors indexed by game using _get_weight helper
        # to support both simplified keys and full profile WEIGHT_* keys
        center_weights = torch.tensor(
            [_get_weight(weights_list[g], "center", 3.0) for g in range(batch_size)],
            device=device
        )
        capture_weights = torch.tensor(
            [_get_weight(weights_list[g], "capture_value", 5.0) for g in range(batch_size)],
            device=device
        )
        adjacency_weights = torch.tensor(
            [_get_weight(weights_list[g], "adjacency", 2.0) for g in range(batch_size)],
            device=device
        )
        line_weights = torch.tensor(
            [_get_weight(weights_list[g], "line_potential", 1.5) for g in range(batch_size)],
            device=device
        )
        noise_weights = torch.tensor(
            [_get_weight(weights_list[g], "noise", 1.0) for g in range(batch_size)],
            device=device
        )
        # Index by game to get per-move weights
        center_weight_per_move = center_weights[game_idx]
        capture_weight_per_move = capture_weights[game_idx]
        adjacency_weight_per_move = adjacency_weights[game_idx]
        line_weight_per_move = line_weights[game_idx]
        noise_weight_per_move = noise_weights[game_idx]
    else:
        # Uniform weights across all moves
        center_weight_per_move = default_weights["center"]
        capture_weight_per_move = default_weights["capture_value"]
        adjacency_weight_per_move = default_weights["adjacency"]
        line_weight_per_move = default_weights["line_potential"]
        noise_weight_per_move = default_weights["noise"]

    center = board_size // 2
    max_dist = center * 2.0

    # === Feature 1: Center distance ===
    dist_to_center = (
        (moves.to_y.float() - center).abs() +
        (moves.to_x.float() - center).abs()
    )
    center_score = (max_dist - dist_to_center) * center_weight_per_move

    # === Feature 2: Capture value (for capture moves) ===
    # Look up target stack height at destination
    to_y = moves.to_y.long()
    to_x = moves.to_x.long()

    target_heights = state.stack_height[game_idx, to_y, to_x].float()
    # Only count for captures (move_type == MoveType.CAPTURE)
    is_capture = moves.move_type == MoveType.CAPTURE
    capture_score = torch.where(
        is_capture,
        target_heights * capture_weight_per_move,
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
    adjacency_score = adjacent_count * adjacency_weight_per_move

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

    line_score = (h_line + v_line) * line_weight_per_move

    # === Combine all scores ===
    noise = torch.rand(moves.total_moves, device=device) * noise_weight_per_move
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

    # NOTE: Do NOT clamp to 0 here! Games with no valid selection should keep -1.
    # Bug fixed 2025-12-20: see select_moves_vectorized for details.

    return selected


class PolicyData:
    """Container for move selection policy data.

    Stores the probability distribution over candidate moves for a batch of games.
    Used to capture policy targets during GPU selfplay for training.
    """

    __slots__ = ('game_idx', 'move_type', 'from_y', 'from_x', 'to_y', 'to_x',
                 'scores', 'probabilities', 'moves_per_game', 'move_offsets',
                 'selected_local_idx')

    def __init__(
        self,
        game_idx: torch.Tensor,
        move_type: torch.Tensor,
        from_y: torch.Tensor,
        from_x: torch.Tensor,
        to_y: torch.Tensor,
        to_x: torch.Tensor,
        scores: torch.Tensor,
        probabilities: torch.Tensor,
        moves_per_game: torch.Tensor,
        move_offsets: torch.Tensor,
        selected_local_idx: torch.Tensor,
    ):
        self.game_idx = game_idx
        self.move_type = move_type
        self.from_y = from_y
        self.from_x = from_x
        self.to_y = to_y
        self.to_x = to_x
        self.scores = scores
        self.probabilities = probabilities
        self.moves_per_game = moves_per_game
        self.move_offsets = move_offsets
        self.selected_local_idx = selected_local_idx

    def extract_for_game(self, game_idx: int) -> dict:
        """Extract policy data for a single game as a dict.

        Returns dict with:
        - candidates: list of {move_type, from, to, score, probability}
        - selected_idx: index of selected move in candidates list
        """
        # Find moves belonging to this game
        mask = self.game_idx == game_idx
        if not mask.any():
            return {"candidates": [], "selected_idx": -1}

        # Get indices for this game's moves
        indices = mask.nonzero(as_tuple=True)[0]
        offset = self.move_offsets[game_idx].item()

        candidates = []
        for i, global_idx in enumerate(indices.tolist()):
            candidates.append({
                "move_type": int(self.move_type[global_idx].item()),
                "from_y": int(self.from_y[global_idx].item()),
                "from_x": int(self.from_x[global_idx].item()),
                "to_y": int(self.to_y[global_idx].item()),
                "to_x": int(self.to_x[global_idx].item()),
                "score": float(self.scores[global_idx].item()),
                "probability": float(self.probabilities[global_idx].item()),
            })

        selected_idx = int(self.selected_local_idx[game_idx].item())

        return {
            "candidates": candidates,
            "selected_idx": selected_idx,
        }

    def extract_batch(self, active_mask: torch.Tensor) -> list[dict | None]:
        """Extract policy data for all active games.

        Returns list of dicts (one per game), None for inactive games.
        """
        batch_size = active_mask.shape[0]
        result = []
        for g in range(batch_size):
            if active_mask[g]:
                result.append(self.extract_for_game(g))
            else:
                result.append(None)
        return result


def select_moves_heuristic_with_policy(
    moves: "BatchMoves",
    state: "BatchGameState",
    active_mask: torch.Tensor,
    weights_list: list[dict[str, float]] | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, PolicyData]:
    """Select moves using heuristic scoring and return full policy distribution.

    Same as select_moves_heuristic but also returns PolicyData containing
    the softmax probabilities for all candidate moves. This enables capturing
    policy targets during GPU selfplay for training.

    Args:
        moves: BatchMoves containing flattened moves for all games
        state: BatchGameState for feature extraction
        active_mask: (batch_size,) bool tensor of games to process
        weights_list: Optional per-game weights list for feature importance
                     (one dict per game, already resolved to current player's weights)
        temperature: Softmax temperature (higher = more random)

    Returns:
        Tuple of:
        - (batch_size,) tensor of selected local move indices per game
        - PolicyData containing scores and probabilities for all moves
    """
    device = moves.device
    batch_size = active_mask.shape[0]
    board_size = state.board_size

    # Initialize output: -1 for games with no moves
    selected = torch.full((batch_size,), -1, dtype=torch.int64, device=device)

    # Initialize empty policy data for edge case
    if moves.total_moves == 0:
        empty_policy = PolicyData(
            game_idx=torch.tensor([], dtype=torch.long, device=device),
            move_type=torch.tensor([], dtype=torch.int8, device=device),
            from_y=torch.tensor([], dtype=torch.int16, device=device),
            from_x=torch.tensor([], dtype=torch.int16, device=device),
            to_y=torch.tensor([], dtype=torch.int16, device=device),
            to_x=torch.tensor([], dtype=torch.int16, device=device),
            scores=torch.tensor([], dtype=torch.float32, device=device),
            probabilities=torch.tensor([], dtype=torch.float32, device=device),
            moves_per_game=moves.moves_per_game,
            move_offsets=moves.move_offsets,
            selected_local_idx=selected,
        )
        return selected, empty_policy

    # Default weights for heuristic features
    default_weights = {
        "center": 3.0,
        "capture_value": 5.0,
        "adjacency": 2.0,
        "line_potential": 1.5,
        "noise": 1.0,
    }

    # Build per-game weight tensors for vectorized scoring
    game_idx = moves.game_idx.long()

    if weights_list is not None:
        # Per-game weights: create tensors indexed by game using _get_weight helper
        # to support both simplified keys and full profile WEIGHT_* keys
        center_weights = torch.tensor(
            [_get_weight(weights_list[g], "center", 3.0) for g in range(batch_size)],
            device=device
        )
        capture_weights = torch.tensor(
            [_get_weight(weights_list[g], "capture_value", 5.0) for g in range(batch_size)],
            device=device
        )
        adjacency_weights = torch.tensor(
            [_get_weight(weights_list[g], "adjacency", 2.0) for g in range(batch_size)],
            device=device
        )
        line_weights = torch.tensor(
            [_get_weight(weights_list[g], "line_potential", 1.5) for g in range(batch_size)],
            device=device
        )
        noise_weights = torch.tensor(
            [_get_weight(weights_list[g], "noise", 1.0) for g in range(batch_size)],
            device=device
        )
        # Index by game to get per-move weights
        center_weight_per_move = center_weights[game_idx]
        capture_weight_per_move = capture_weights[game_idx]
        adjacency_weight_per_move = adjacency_weights[game_idx]
        line_weight_per_move = line_weights[game_idx]
        noise_weight_per_move = noise_weights[game_idx]
    else:
        # Uniform weights across all moves
        center_weight_per_move = default_weights["center"]
        capture_weight_per_move = default_weights["capture_value"]
        adjacency_weight_per_move = default_weights["adjacency"]
        line_weight_per_move = default_weights["line_potential"]
        noise_weight_per_move = default_weights["noise"]

    center = board_size // 2
    max_dist = center * 2.0

    # === Feature 1: Center distance ===
    dist_to_center = (
        (moves.to_y.float() - center).abs() +
        (moves.to_x.float() - center).abs()
    )
    center_score = (max_dist - dist_to_center) * center_weight_per_move

    # === Feature 2: Capture value (for capture moves) ===
    to_y = moves.to_y.long()
    to_x = moves.to_x.long()

    target_heights = state.stack_height[game_idx, to_y, to_x].float()
    is_capture = moves.move_type == MoveType.CAPTURE
    capture_score = torch.where(
        is_capture,
        target_heights * capture_weight_per_move,
        torch.zeros_like(target_heights)
    )

    # === Feature 3: Adjacency to own stacks ===
    current_players = state.current_player[game_idx]
    so_padded = torch.nn.functional.pad(
        state.stack_owner.float(), (1, 1, 1, 1), value=0
    )

    adj_up = so_padded[game_idx, to_y, to_x + 1] == current_players.float()
    adj_down = so_padded[game_idx, to_y + 2, to_x + 1] == current_players.float()
    adj_left = so_padded[game_idx, to_y + 1, to_x] == current_players.float()
    adj_right = so_padded[game_idx, to_y + 1, to_x + 2] == current_players.float()

    adjacent_count = adj_up.float() + adj_down.float() + adj_left.float() + adj_right.float()
    adjacency_score = adjacent_count * adjacency_weight_per_move

    # === Feature 4: Line potential ===
    own_stacks = (state.stack_owner[game_idx] == current_players.view(-1, 1, 1))
    own_stacks_padded = torch.nn.functional.pad(own_stacks.float(), (1, 1, 1, 1), value=0)

    move_idx = torch.arange(moves.total_moves, device=device)

    h_left = own_stacks_padded[move_idx, to_y + 1, to_x]
    h_right = own_stacks_padded[move_idx, to_y + 1, to_x + 2]
    h_line = h_left + h_right

    v_up = own_stacks_padded[move_idx, to_y, to_x + 1]
    v_down = own_stacks_padded[move_idx, to_y + 2, to_x + 1]
    v_line = v_up + v_down

    line_score = (h_line + v_line) * line_weight_per_move

    # === Combine all scores (without noise for policy - noise added separately) ===
    base_scores = center_score + capture_score + adjacency_score + line_score

    # Add noise for selection but keep base_scores for policy
    noise = torch.rand(moves.total_moves, device=device) * noise_weight_per_move
    scores = base_scores + noise

    # Segment-wise softmax
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

    # Create PolicyData with all move information
    policy_data = PolicyData(
        game_idx=game_idx,
        move_type=moves.move_type,
        from_y=moves.from_y,
        from_x=moves.from_x,
        to_y=moves.to_y,
        to_x=moves.to_x,
        scores=base_scores,  # Without noise for cleaner policy
        probabilities=probs,
        moves_per_game=moves.moves_per_game,
        move_offsets=moves.move_offsets,
        selected_local_idx=selected,
    )

    return selected, policy_data


__all__ = [
    'select_moves_heuristic',
    'select_moves_heuristic_with_policy',
    'select_moves_vectorized',
    'PolicyData',
]
