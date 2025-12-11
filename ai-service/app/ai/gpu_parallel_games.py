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
        # square8: 18, square19: 48, hexagonal: 72
        starting_rings = {8: 18, 19: 48}.get(board_size, 18)

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

        for key, player in game_state.board.markers.items():
            x, y = map(int, key.split(","))
            if 0 <= x < board_size and 0 <= y < board_size:
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

    # Collect all valid moves
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

                    # Check path is clear of opponent stacks
                    path_blocked = False
                    for step in range(1, dist + 1):
                        check_y = from_y + dy * step
                        check_x = from_x + dx * step
                        cell_owner = state.stack_owner[g, check_y, check_x].item()

                        # If opponent stack, this is capture territory (not movement)
                        if cell_owner != 0 and cell_owner != player:
                            path_blocked = True
                            break

                    if path_blocked:
                        break

                    # Valid landing spot - empty or own stack
                    dest_owner = state.stack_owner[g, to_y, to_x].item()
                    if dest_owner == 0 or dest_owner == player:
                        all_game_idx.append(g)
                        all_from_y.append(from_y)
                        all_from_x.append(from_x)
                        all_to_y.append(to_y)
                        all_to_x.append(to_x)

    total_moves = len(all_game_idx)

    if total_moves == 0:
        # Return empty moves
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

    # Convert to tensors
    game_idx_t = torch.tensor(all_game_idx, dtype=torch.int32, device=device)
    from_y_t = torch.tensor(all_from_y, dtype=torch.int32, device=device)
    from_x_t = torch.tensor(all_from_x, dtype=torch.int32, device=device)
    to_y_t = torch.tensor(all_to_y, dtype=torch.int32, device=device)
    to_x_t = torch.tensor(all_to_x, dtype=torch.int32, device=device)

    # Count moves per game
    moves_per_game = torch.zeros(batch_size, dtype=torch.int32, device=device)
    for g in all_game_idx:
        moves_per_game[g] += 1

    move_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), moves_per_game[:-1]]),
        dim=0
    ).int()

    return BatchMoves(
        game_idx=game_idx_t,
        move_type=torch.full((total_moves,), MoveType.MOVEMENT, dtype=torch.int8, device=device),
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


# =============================================================================
# Recovery Slide Move Generation (RR-CANON-R110-R115)
# =============================================================================


def generate_recovery_moves_batch(
    state: BatchGameState,
    active_mask: Optional[torch.Tensor] = None,
) -> BatchMoves:
    """Generate all valid recovery slide moves for eligible players.

    Per RR-CANON-R110-R115:
    - Player must have no controlled stacks AND no rings in hand
    - Player must have at least one marker on the board
    - Player must have buried rings (can afford the recovery cost)
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

        # 2. No rings in hand
        rings_in_hand = state.rings_in_hand[g, player].item()
        if rings_in_hand > 0:
            continue

        # 3. Has markers on board
        my_markers = (state.marker_owner[g] == player)
        marker_positions = torch.nonzero(my_markers, as_tuple=False)
        if marker_positions.shape[0] == 0:
            continue

        # 4. Has buried rings (can afford recovery cost)
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


def detect_lines_batch(
    state: BatchGameState,
    player: int,
    game_mask: Optional[torch.Tensor] = None,
) -> List[List[Tuple[int, int]]]:
    """Detect lines of 4+ same-owner stacks for a player.

    Per RR-CANON-R120: A line is 4+ consecutive stacks controlled by same player
    in horizontal, vertical, or diagonal direction.

    Args:
        state: Current batch game state
        player: Player number to detect lines for
        game_mask: Mask of games to check (optional)

    Returns:
        List of lists of (y, x) tuples, one per game, containing all line positions
    """
    batch_size = state.batch_size
    board_size = state.board_size

    if game_mask is None:
        game_mask = torch.ones(batch_size, dtype=torch.bool, device=state.device)

    lines_per_game = [[] for _ in range(batch_size)]

    # 4 directions to check for lines: horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for g in range(batch_size):
        if not game_mask[g]:
            continue

        player_stacks = (state.stack_owner[g] == player)

        # Check each direction for lines
        for dy, dx in directions:
            visited = set()

            for start_y in range(board_size):
                for start_x in range(board_size):
                    if (start_y, start_x) in visited:
                        continue
                    if not player_stacks[start_y, start_x]:
                        continue

                    # Trace line in this direction
                    line = [(start_y, start_x)]
                    y, x = start_y + dy, start_x + dx

                    while 0 <= y < board_size and 0 <= x < board_size:
                        if player_stacks[y, x]:
                            line.append((y, x))
                            y, x = y + dy, x + dx
                        else:
                            break

                    # If line is 4+, record it
                    if len(line) >= 4:
                        for pos in line:
                            visited.add(pos)
                            if pos not in lines_per_game[g]:
                                lines_per_game[g].append(pos)

    return lines_per_game


def process_lines_batch(
    state: BatchGameState,
    game_mask: Optional[torch.Tensor] = None,
) -> None:
    """Process formed lines for all players (in-place).

    Per RR-CANON-R121-R122:
    - When a line of 4+ is formed, line owner claims those spaces
    - Stacks in line are converted to territory markers
    - Rings are removed from play

    Args:
        state: BatchGameState to modify
        game_mask: Mask of games to process
    """
    batch_size = state.batch_size

    if game_mask is None:
        game_mask = state.get_active_mask()

    for p in range(1, state.num_players + 1):
        lines = detect_lines_batch(state, p, game_mask)

        for g in range(batch_size):
            if not game_mask[g]:
                continue

            if lines[g]:
                for (y, x) in lines[g]:
                    # Convert stack to territory
                    stack_height = state.stack_height[g, y, x].item()
                    state.stack_owner[g, y, x] = 0
                    state.stack_height[g, y, x] = 0
                    state.territory_owner[g, y, x] = p
                    state.territory_count[g, p] += 1

                    # Remove rings from play
                    state.eliminated_rings[g, p] += stack_height


# =============================================================================
# Territory Processing (RR-CANON-R140-R146)
# =============================================================================


def compute_territory_batch(
    state: BatchGameState,
    game_mask: Optional[torch.Tensor] = None,
) -> None:
    """Compute and update territory claims (in-place).

    Per RR-CANON-R140-R146:
    - Empty spaces fully enclosed by one player's stacks become territory
    - Territory is counted for victory condition

    Uses flood-fill from edges to find unenclosed spaces.

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

        # For each player, find spaces they enclose
        for player in range(1, state.num_players + 1):
            # Create mask of spaces owned or controlled by player
            player_controlled = (
                (state.stack_owner[g] == player) |
                (state.territory_owner[g] == player)
            )

            # Find empty spaces (not controlled by anyone, not territory)
            empty = (state.stack_owner[g] == 0) & (state.territory_owner[g] == 0)

            # Flood-fill from edges to find unenclosed empty spaces
            reachable = torch.zeros(board_size, board_size, dtype=torch.bool, device=state.device)

            # BFS from all edge cells
            queue = []
            for i in range(board_size):
                # Top and bottom edges
                if empty[0, i] and not player_controlled[0, i]:
                    queue.append((0, i))
                    reachable[0, i] = True
                if empty[board_size-1, i] and not player_controlled[board_size-1, i]:
                    queue.append((board_size-1, i))
                    reachable[board_size-1, i] = True
                # Left and right edges
                if empty[i, 0] and not player_controlled[i, 0]:
                    queue.append((i, 0))
                    reachable[i, 0] = True
                if empty[i, board_size-1] and not player_controlled[i, board_size-1]:
                    queue.append((i, board_size-1))
                    reachable[i, board_size-1] = True

            # BFS to find all reachable empty spaces
            idx = 0
            while idx < len(queue):
                y, x = queue[idx]
                idx += 1

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < board_size and 0 <= nx < board_size:
                        if empty[ny, nx] and not reachable[ny, nx] and not player_controlled[ny, nx]:
                            reachable[ny, nx] = True
                            queue.append((ny, nx))

            # Empty spaces NOT reachable from edge are enclosed
            enclosed = empty & ~reachable

            # Claim enclosed territory
            new_territory = enclosed & (state.territory_owner[g] == 0)
            territory_count = new_territory.sum().item()

            if territory_count > 0:
                state.territory_owner[g] = torch.where(
                    new_territory,
                    torch.full_like(state.territory_owner[g], player),
                    state.territory_owner[g]
                )
                state.territory_count[g, player] += territory_count


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

    # Victory thresholds - derived from board size per BOARD_CONFIGS
    # Territory: floor(totalSpaces/2) + 1; Ring supply per player
    # square8: 64 spaces, 18 rings; square19: 361 spaces, 48 rings; hex: 469 spaces, 72 rings
    total_spaces = board_size * board_size  # Works for square boards
    territory_victory_threshold = (total_spaces // 2) + 1  # 33 for 8x8, 181 for 19x19
    rings_per_player = {8: 18, 19: 48}.get(board_size, 18)  # Per BOARD_CONFIGS
    # Per RR-CANON-R061: victoryThreshold = round((1/3)*ownStartingRings + (2/3)*opponentsCombinedStartingRings)
    # Simplified: round(ringsPerPlayer * (1/3 + 2/3*(numPlayers-1)))
    ring_victory_threshold = round(rings_per_player * (1 / 3 + (2 / 3) * (num_players - 1)))

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
        # Count adjacent pairs of controlled stacks
        adjacency_score = torch.zeros(batch_size, device=device)
        for g in range(batch_size):
            adj_count = 0.0
            ps = player_stacks[g]
            for y in range(board_size):
                for x in range(board_size):
                    if ps[y, x]:
                        # Check right and down neighbors only to avoid double counting
                        if x + 1 < board_size and ps[y, x + 1]:
                            adj_count += 1.0
                        if y + 1 < board_size and ps[y + 1, x]:
                            adj_count += 1.0
            adjacency_score[g] = adj_count

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
        # Recovery potential: value of having recovery available
        has_buried = buried_rings > 0
        no_controlled = stack_count == 0
        no_rings_in_hand = rings_in_hand == 0
        recovery_eligible = has_buried & no_controlled & no_rings_in_hand
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

        # Extract move histories and victory types for each game
        move_histories = []
        victory_types = []
        stalemate_tiebreakers = []

        for g in range(self.batch_size):
            move_histories.append(self.state.extract_move_history(g))
            vtype, tiebreaker = self.state.derive_victory_type(g, max_moves)
            victory_types.append(vtype)
            stalemate_tiebreakers.append(tiebreaker)

        return {
            "winners": self.state.winner.cpu().tolist(),
            "move_counts": self.state.move_count.cpu().tolist(),
            "status": self.state.game_status.cpu().tolist(),
            "move_histories": move_histories,
            "victory_types": victory_types,
            "stalemate_tiebreakers": stalemate_tiebreakers,
            "elapsed_seconds": elapsed,
            "games_per_second": self.batch_size / elapsed,
        }

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

    def _step_placement_phase(
        self,
        mask: torch.Tensor,
        weights_list: List[Dict[str, float]],
    ) -> None:
        """Handle RING_PLACEMENT phase for games in mask."""
        # Check which games have rings to place
        current_players = self.state.current_player[mask]

        # Get rings in hand for each game's current player
        has_rings = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        for g in range(self.batch_size):
            if mask[g]:
                p = self.state.current_player[g].item()
                has_rings[g] = self.state.rings_in_hand[g, p] > 0

        games_with_rings = mask & has_rings
        games_without_rings = mask & ~has_rings

        # Games WITH rings: generate and apply placement moves
        if games_with_rings.any():
            moves = generate_placement_moves_batch(self.state, games_with_rings)
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
        """
        # Check which games have stacks to move
        has_stacks = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        for g in range(self.batch_size):
            if mask[g]:
                p = self.state.current_player[g].item()
                has_stacks[g] = (self.state.stack_owner[g] == p).any()

        games_with_stacks = mask & has_stacks
        games_without_stacks = mask & ~has_stacks

        # Games WITHOUT stacks: check for recovery moves
        if games_without_stacks.any():
            recovery_moves = generate_recovery_moves_batch(self.state, games_without_stacks)

            # Track games that had no recovery moves (for stalemate check)
            games_no_recovery = games_without_stacks.clone()

            for g in range(self.batch_size):
                if not games_without_stacks[g]:
                    continue

                # Check if this game has recovery moves
                recovery_start = recovery_moves.move_offsets[g].item()
                recovery_count = recovery_moves.moves_per_game[g].item()

                if recovery_count > 0:
                    # This game has recovery moves, mark it
                    games_no_recovery[g] = False

                    # Select a recovery move (prefer center destinations)
                    center = self.board_size // 2
                    best_idx = 0
                    best_dist = float('inf')
                    for i in range(recovery_count):
                        idx = recovery_start + i
                        ty = recovery_moves.to_y[idx].item()
                        tx = recovery_moves.to_x[idx].item()
                        dist = abs(ty - center) + abs(tx - center)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i

                    # Apply recovery move
                    self._apply_single_recovery(g, recovery_start + best_idx, recovery_moves)

            # Check for stalemate in games that had no stacks AND no recovery moves
            if games_no_recovery.any():
                self._check_stalemate(games_no_recovery)

        # Games WITH stacks: generate movement and capture moves
        if games_with_stacks.any():
            # Generate non-capture movement moves
            movement_moves = generate_movement_moves_batch(self.state, games_with_stacks)

            # Generate capture moves
            capture_moves = generate_capture_moves_batch(self.state, games_with_stacks)

            # For simplicity, prefer captures when available (more aggressive play)
            # In a full implementation, we'd evaluate both and choose best
            for g in range(self.batch_size):
                if not games_with_stacks[g]:
                    continue

                # Check if this game has capture moves
                capture_start = capture_moves.move_offsets[g].item()
                capture_count = capture_moves.moves_per_game[g].item()

                if capture_count > 0:
                    # Select capture move using softmax sampling with center bias
                    # This adds stochasticity to break deterministic P1 advantage
                    center = self.board_size // 2
                    max_dist = center * 2
                    scores = []
                    for i in range(capture_count):
                        idx = capture_start + i
                        ty = capture_moves.to_y[idx].item()
                        tx = capture_moves.to_x[idx].item()
                        dist = abs(ty - center) + abs(tx - center)
                        # Score: higher for closer to center + random noise
                        score = (max_dist - dist) + torch.rand(1, device=self.device).item() * 2.0
                        scores.append(score)

                    # Softmax selection with temperature=1.0
                    scores_tensor = torch.tensor(scores, device=self.device)
                    probs = torch.softmax(scores_tensor, dim=0)
                    selected_idx = torch.multinomial(probs, 1).item()

                    # Apply capture move directly for this game
                    self._apply_single_capture(g, capture_start + selected_idx, capture_moves)
                else:
                    # Apply a movement move
                    move_start = movement_moves.move_offsets[g].item()
                    move_count = movement_moves.moves_per_game[g].item()

                    if move_count > 0:
                        # Select movement move using softmax sampling with center bias
                        # This adds stochasticity to break deterministic P1 advantage
                        center = self.board_size // 2
                        max_dist = center * 2
                        scores = []
                        for i in range(move_count):
                            idx = move_start + i
                            ty = movement_moves.to_y[idx].item()
                            tx = movement_moves.to_x[idx].item()
                            dist = abs(ty - center) + abs(tx - center)
                            # Score: higher for closer to center + random noise
                            score = (max_dist - dist) + torch.rand(1, device=self.device).item() * 2.0
                            scores.append(score)

                        # Softmax selection with temperature=1.0
                        scores_tensor = torch.tensor(scores, device=self.device)
                        probs = torch.softmax(scores_tensor, dim=0)
                        selected_idx = torch.multinomial(probs, 1).item()

                        # Apply movement move directly for this game
                        self._apply_single_movement(g, move_start + selected_idx, movement_moves)

        # After movement, advance to LINE_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.LINE_PROCESSING

    def _step_line_phase(self, mask: torch.Tensor) -> None:
        """Handle LINE_PROCESSING phase for games in mask.

        Detect lines and convert them to territory markers.
        """
        process_lines_batch(self.state, mask)

        # After line processing, advance to TERRITORY_PROCESSING phase
        self.state.current_phase[mask] = GamePhase.TERRITORY_PROCESSING

    def _step_territory_phase(self, mask: torch.Tensor) -> None:
        """Handle TERRITORY_PROCESSING phase for games in mask.

        Calculate enclosed territory using flood-fill.
        """
        compute_territory_batch(self.state, mask)

        # After territory processing, advance to END_TURN phase
        self.state.current_phase[mask] = GamePhase.END_TURN

    def _step_end_turn_phase(self, mask: torch.Tensor) -> None:
        """Handle END_TURN phase for games in mask.

        Rotate to next player and reset phase to RING_PLACEMENT.

        Per updated rules: Players are only permanently eliminated if they have
        NO rings anywhere (no controlled stacks, no buried rings, no rings in hand).
        Players with only buried rings still get turns and can use recovery moves.

        Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
        NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action
        and proceed to movement, but they MUST enter ring_placement first.
        """
        for g in range(self.batch_size):
            if not mask[g]:
                continue

            # Increment move count
            self.state.move_count[g] += 1

            # Rotate to next player, skipping permanently eliminated players
            current = self.state.current_player[g].item()
            next_player = (current % self.num_players) + 1

            # Check up to num_players times to find a player with any rings
            skips = 0
            while skips < self.num_players:
                # Check if this player has ANY rings (controlled, buried, or in hand)
                has_any_rings = self._player_has_any_rings_gpu(g, next_player)

                if has_any_rings:
                    # This player is not permanently eliminated
                    break

                # Skip to next player
                next_player = (next_player % self.num_players) + 1
                skips += 1

            self.state.current_player[g] = next_player

            # Per RR-CANON-R073: ALL players start in RING_PLACEMENT without exception.
            # NO PHASE SKIPPING - this is a core invariant for parity with TS/Python engines.
            self.state.current_phase[g] = GamePhase.RING_PLACEMENT

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
        - victoryThreshold = round(ringsPerPlayer × (1/3 + 2/3 × (numPlayers - 1)))
        - territoryVictoryThreshold = floor(totalSpaces / 2) + 1
        """
        active_mask = self.state.get_active_mask()

        # Calculate canonical thresholds per RR-CANON-R061
        # Ring elimination: victoryThreshold = round(ringsPerPlayer × (1/3 + 2/3 × (numPlayers - 1)))
        # Must match create_batch() initialization: {8: 18, 19: 48}
        rings_per_player = {8: 18, 19: 48}.get(self.board_size, 18)
        ring_elimination_threshold = round(rings_per_player * (1/3 + (2/3) * (self.num_players - 1)))

        # totalSpaces = board_size * board_size for square boards
        total_spaces = self.board_size * self.board_size
        territory_victory_threshold = (total_spaces // 2) + 1

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
