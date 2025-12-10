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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .gpu_batch import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)


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

    # Game metadata: (batch_size,)
    current_player: torch.Tensor   # 1-4
    move_count: torch.Tensor
    game_status: torch.Tensor      # GameStatus enum
    winner: torch.Tensor           # 0=none, 1-4=player

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
    ) -> "BatchGameState":
        """Create a batch of initialized game states.

        Args:
            batch_size: Number of parallel games
            board_size: Board dimension (8, 19)
            num_players: Number of players (2-4)
            device: GPU device (auto-detected if None)

        Returns:
            Initialized BatchGameState with all games ready to start
        """
        if device is None:
            device = get_device()

        # Initialize board tensors
        shape_board = (batch_size, board_size, board_size)
        shape_players = (batch_size, num_players + 1)  # +1 for 1-indexed players

        # Starting rings per player based on board size
        starting_rings = {8: 19, 19: 50}.get(board_size, 19)

        rings = torch.zeros(shape_players, dtype=torch.int16, device=device)
        rings[:, 1:num_players+1] = starting_rings

        return cls(
            stack_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            stack_height=torch.zeros(shape_board, dtype=torch.int8, device=device),
            marker_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            territory_owner=torch.zeros(shape_board, dtype=torch.int8, device=device),
            is_collapsed=torch.zeros(shape_board, dtype=torch.bool, device=device),
            rings_in_hand=rings,
            territory_count=torch.zeros(shape_players, dtype=torch.int16, device=device),
            is_eliminated=torch.zeros(shape_players, dtype=torch.bool, device=device),
            current_player=torch.ones(batch_size, dtype=torch.int8, device=device),
            move_count=torch.zeros(batch_size, dtype=torch.int32, device=device),
            game_status=torch.zeros(batch_size, dtype=torch.int8, device=device),
            winner=torch.zeros(batch_size, dtype=torch.int8, device=device),
            device=device,
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
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

        # Apply placement
        state.stack_owner[g, y, x] = player
        state.stack_height[g, y, x] = 1
        state.rings_in_hand[g, player] -= 1

        # Advance turn
        state.move_count[g] += 1
        state.current_player[g] = (player % state.num_players) + 1


# =============================================================================
# Batch Heuristic Evaluation
# =============================================================================


def evaluate_positions_batch(
    state: BatchGameState,
    weights: Dict[str, float],
) -> torch.Tensor:
    """Evaluate all positions using heuristic scoring.

    Args:
        state: BatchGameState to evaluate
        weights: Heuristic weight dictionary

    Returns:
        Tensor of scores (batch_size, num_players) for each player
    """
    device = state.device
    batch_size = state.batch_size
    num_players = state.num_players

    scores = torch.zeros(batch_size, num_players + 1, dtype=torch.float32, device=device)

    for p in range(1, num_players + 1):
        # Stack count
        my_stacks = (state.stack_owner == p).sum(dim=(1, 2)).float()

        # Territory
        my_territory = state.territory_count[:, p].float()

        # Rings in hand (negative - want to place them)
        my_rings = state.rings_in_hand[:, p].float()

        # Compute score
        w = weights
        scores[:, p] = (
            my_stacks * w.get("stack_count", 1.0)
            + my_territory * w.get("territory_count", 2.0)
            - my_rings * w.get("rings_penalty", 0.1)
        )

        # Elimination penalty
        scores[:, p] = torch.where(
            my_stacks == 0,
            scores[:, p] - 1000.0,
            scores[:, p]
        )

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
    ):
        """Initialize parallel game runner.

        Args:
            batch_size: Number of games to run in parallel
            board_size: Board dimension
            num_players: Number of players per game
            device: GPU device (auto-detected if None)
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

        return {
            "winners": self.state.winner.cpu().tolist(),
            "move_counts": self.state.move_count.cpu().tolist(),
            "status": self.state.game_status.cpu().tolist(),
            "elapsed_seconds": elapsed,
            "games_per_second": self.batch_size / elapsed,
        }

    def _step_games(self, weights_list: List[Dict[str, float]]) -> None:
        """Execute one move for all active games."""
        active_mask = self.state.get_active_mask()

        if not active_mask.any():
            return

        # Generate moves
        moves = generate_placement_moves_batch(self.state, active_mask)

        if moves.total_moves == 0:
            return

        # Select best move for each game based on heuristic
        selected_moves = self._select_best_moves(moves, weights_list, active_mask)

        # Apply selected moves
        apply_placement_moves_batch(self.state, selected_moves, moves)

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

            # Score by distance to center (lower is better)
            dist_to_center = (
                (game_moves_y.float() - center).abs() +
                (game_moves_x.float() - center).abs()
            )

            # Add small random noise for variety
            dist_to_center = dist_to_center + torch.rand_like(dist_to_center) * 0.5

            # Select move with minimum distance
            best_local_idx = dist_to_center.argmin()
            selected[g] = best_local_idx

        return selected

    def _check_victory_conditions(self) -> None:
        """Check and update victory conditions for all games."""
        active_mask = self.state.get_active_mask()

        for p in range(1, self.num_players + 1):
            # Check territory victory (e.g., 33 for 8x8)
            victory_threshold = 33 if self.board_size == 8 else 100
            territory_victory = self.state.territory_count[:, p] >= victory_threshold

            # Check if only this player has stacks (elimination victory)
            has_stacks = (self.state.stack_owner == p).any(dim=(1, 2))
            others_have_stacks = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            for other_p in range(1, self.num_players + 1):
                if other_p != p:
                    others_have_stacks |= (self.state.stack_owner == other_p).any(dim=(1, 2))

            elimination_victory = has_stacks & ~others_have_stacks

            # Update winners
            victory_mask = active_mask & (territory_victory | elimination_victory)
            self.state.winner[victory_mask] = p
            self.state.game_status[victory_mask] = GameStatus.COMPLETED

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
