"""GPU batch game state for parallel games.

This module provides the BatchGameState class for GPU-accelerated parallel
game simulation. Extracted from gpu_parallel_games.py for modularity.

December 2025: Extracted as part of R21 refactoring.

BatchGameState is the core data structure that holds:
- Board state tensors (stacks, markers, territory)
- Player state tensors (rings, elimination status)
- Game metadata (phase, turn, winner)
- Move history for training data

All tensors have shape (batch_size, ...) and are stored on GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .gpu_batch import get_device
from .gpu_game_types import (
    GamePhase,
    GameStatus,
    MoveType,
)

if TYPE_CHECKING:
    from app.models import GameState

logger = logging.getLogger(__name__)


# =============================================================================
# CPU to GPU MoveType Mapping
# =============================================================================

# Mapping from CPU MoveType string values to GPU MoveType integer values
# CPU MoveType is a string enum, GPU MoveType is an IntEnum
# December 2025: Updated for canonical parity with phase-specific types
_CPU_TO_GPU_MOVE_TYPE = {
    # Placement moves
    'place_ring': MoveType.PLACEMENT,
    'skip_placement': MoveType.SKIP_PLACEMENT,
    'no_placement_action': MoveType.NO_PLACEMENT_ACTION,
    # Movement moves
    'move_stack': MoveType.MOVEMENT,
    'move_ring': MoveType.MOVEMENT,
    'build_stack': MoveType.MOVEMENT,
    'no_movement_action': MoveType.NO_MOVEMENT_ACTION,
    # Capture moves (canonical types)
    'overtaking_capture': MoveType.OVERTAKING_CAPTURE,
    'continue_capture_segment': MoveType.CONTINUE_CAPTURE_SEGMENT,
    'skip_capture': MoveType.SKIP_CAPTURE,
    'chain_capture': MoveType.OVERTAKING_CAPTURE,  # Legacy alias
    # Line moves
    'process_line': MoveType.LINE_FORMATION,
    'choose_line_reward': MoveType.CHOOSE_LINE_OPTION,
    'no_line_action': MoveType.NO_LINE_ACTION,
    'line_formation': MoveType.LINE_FORMATION,
    'choose_line_option': MoveType.CHOOSE_LINE_OPTION,
    # Territory moves
    'process_territory_region': MoveType.TERRITORY_CLAIM,
    'skip_territory_processing': MoveType.NO_TERRITORY_ACTION,
    'no_territory_action': MoveType.NO_TERRITORY_ACTION,
    'territory_claim': MoveType.TERRITORY_CLAIM,
    'choose_territory_option': MoveType.CHOOSE_TERRITORY_OPTION,
    # Recovery moves
    'recovery_slide': MoveType.RECOVERY_SLIDE,
    'skip_recovery': MoveType.SKIP_RECOVERY,
    # Other
    'swap_sides': MoveType.SKIP,
    'eliminate_rings_from_stack': MoveType.SKIP,
    'forced_elimination': MoveType.FORCED_ELIMINATION,
}


def _cpu_move_type_to_gpu(cpu_move_type) -> int:
    """Convert CPU MoveType to GPU MoveType integer value."""
    if hasattr(cpu_move_type, 'value'):
        # It's an enum, get the string value
        value = cpu_move_type.value
    else:
        value = cpu_move_type

    gpu_type = _CPU_TO_GPU_MOVE_TYPE.get(value, MoveType.SKIP)
    return int(gpu_type)


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

    # Capture chain tracking (December 2025 - canonical phases)
    # Track if game is in a capture chain sequence (for CHAIN_CAPTURE phase)
    in_capture_chain: torch.Tensor  # bool (batch_size,)
    capture_chain_depth: torch.Tensor  # int16 (batch_size,) - 0 = no chain

    # Forced elimination detection (December 2025 - RR-CANON-R160)
    # Track if player took a "real" action this turn (placement, movement, capture)
    # If turn ends with no real action and player has stacks → FORCED_ELIMINATION
    turn_had_real_action: torch.Tensor  # bool (batch_size,)

    # Buried ring position tracking (December 2025 - recovery fix)
    # Track which board positions contain each player's buried rings.
    # This enables recovery to correctly decrement stack height when extracting.
    # Shape: (batch_size, num_players + 1, board_size, board_size)
    buried_at: torch.Tensor  # bool - True if player has a buried ring at this position

    # LPS tracking (RR-CANON-R172): tensor mirrors of GameState fields.
    # We track a full-round cycle over all non-permanently-eliminated players.
    lps_round_index: torch.Tensor  # int32 (batch_size,)
    lps_current_round_first_player: torch.Tensor  # int8 (batch_size,) 0=unset
    lps_current_round_seen_mask: torch.Tensor  # bool (batch_size, num_players+1)
    lps_current_round_real_action_mask: torch.Tensor  # bool (batch_size, num_players+1)
    lps_exclusive_player_for_completed_round: torch.Tensor  # int8 (batch_size,) 0=none
    lps_consecutive_exclusive_rounds: torch.Tensor  # int16 (batch_size,)
    lps_consecutive_exclusive_player: torch.Tensor  # int8 (batch_size,) 0=none

    # Move history: (batch_size, max_moves, 9)
    # Columns: [move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x]
    # -1 indicates unused slot or N/A (e.g., capture_target for non-capture moves)
    # Phase column enables canonical export with phase tracking (December 2025)
    # Capture target columns enable canonical export of capture moves (December 2025)
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
        device: torch.device | None = None,
        max_history_moves: int = 500,
        lps_rounds_required: int = 3,
        rings_per_player: int | None = None,
        board_type: str | None = None,
    ) -> BatchGameState:
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

        # Determine starting rings based on board type or size
        if rings_per_player is None:
            if board_type == "hexagonal":
                rings_per_player = 96  # Canonical hex ring count
            elif board_type == "square19":
                rings_per_player = 72  # 19x19 ring count
            elif board_size <= 8:
                rings_per_player = 18
            elif board_size <= 13:
                rings_per_player = 24
            else:
                rings_per_player = 36

        # Create tensors
        state = cls(
            # Board state
            stack_owner=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            stack_height=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            cap_height=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            marker_owner=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            territory_owner=torch.zeros((batch_size, board_size, board_size), dtype=torch.int8, device=device),
            is_collapsed=torch.zeros((batch_size, board_size, board_size), dtype=torch.bool, device=device),

            # Player state (index 0 unused, 1-4 for players)
            # Note: rings_in_hand uses int32 because MPS requires it for index_put_ with accumulate=True
            rings_in_hand=torch.zeros((batch_size, num_players + 1), dtype=torch.int32, device=device),
            territory_count=torch.zeros((batch_size, num_players + 1), dtype=torch.int32, device=device),
            is_eliminated=torch.zeros((batch_size, num_players + 1), dtype=torch.bool, device=device),
            eliminated_rings=torch.zeros((batch_size, num_players + 1), dtype=torch.int32, device=device),
            buried_rings=torch.zeros((batch_size, num_players + 1), dtype=torch.int32, device=device),
            rings_caused_eliminated=torch.zeros((batch_size, num_players + 1), dtype=torch.int32, device=device),

            # Game metadata
            current_player=torch.ones(batch_size, dtype=torch.int8, device=device),
            current_phase=torch.full((batch_size,), GamePhase.RING_PLACEMENT, dtype=torch.int8, device=device),
            move_count=torch.zeros(batch_size, dtype=torch.int32, device=device),
            game_status=torch.full((batch_size,), GameStatus.ACTIVE, dtype=torch.int8, device=device),
            winner=torch.zeros(batch_size, dtype=torch.int8, device=device),
            swap_offered=torch.zeros(batch_size, dtype=torch.bool, device=device),

            # Movement constraints
            must_move_from_y=torch.full((batch_size,), -1, dtype=torch.int16, device=device),
            must_move_from_x=torch.full((batch_size,), -1, dtype=torch.int16, device=device),

            # Capture chain tracking (December 2025 - canonical phases)
            in_capture_chain=torch.zeros(batch_size, dtype=torch.bool, device=device),
            capture_chain_depth=torch.zeros(batch_size, dtype=torch.int16, device=device),

            # Forced elimination detection (December 2025 - RR-CANON-R160)
            turn_had_real_action=torch.zeros(batch_size, dtype=torch.bool, device=device),

            # Buried ring position tracking (December 2025 - recovery fix)
            buried_at=torch.zeros((batch_size, num_players + 1, board_size, board_size), dtype=torch.bool, device=device),

            # LPS tracking tensors (RR-CANON-R172)
            lps_round_index=torch.zeros(batch_size, dtype=torch.int32, device=device),
            lps_current_round_first_player=torch.zeros(batch_size, dtype=torch.int8, device=device),
            lps_current_round_seen_mask=torch.zeros((batch_size, num_players + 1), dtype=torch.bool, device=device),
            lps_current_round_real_action_mask=torch.zeros((batch_size, num_players + 1), dtype=torch.bool, device=device),
            lps_exclusive_player_for_completed_round=torch.zeros(batch_size, dtype=torch.int8, device=device),
            lps_consecutive_exclusive_rounds=torch.zeros(batch_size, dtype=torch.int16, device=device),
            lps_consecutive_exclusive_player=torch.zeros(batch_size, dtype=torch.int8, device=device),

            # Move history (9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x)
            move_history=torch.full((batch_size, max_history_moves, 9), -1, dtype=torch.int16, device=device),
            max_history_moves=max_history_moves,

            # LPS configuration
            lps_rounds_required=lps_rounds_required,

            # Configuration
            device=device,
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
        )

        # Initialize rings per player
        for p in range(1, num_players + 1):
            state.rings_in_hand[:, p] = rings_per_player

        # Mark hex out-of-bounds cells as collapsed if using hexagonal board
        if board_type == "hexagonal":
            # Hexagonal board uses a 25x25 grid with corners cut off
            # Standard hex board has radius 8, so cells > distance 8 from center are OOB
            center = board_size // 2
            for r in range(board_size):
                for c in range(board_size):
                    # Axial distance from center for hex grid
                    dr = r - center
                    dc = c - center
                    # Hex distance
                    hex_dist = max(abs(dr), abs(dc), abs(dr + dc))
                    if hex_dist > 8:
                        state.is_collapsed[:, r, c] = True

        return state

    @classmethod
    def from_single_game(
        cls,
        game_state: GameState,
        device: torch.device | None = None,
        max_history_moves: int = 500,
        lps_rounds_required: int = 3,
    ) -> BatchGameState:
        """Create a BatchGameState from a single CPU GameState.

        Args:
            game_state: CPU GameState to convert
            device: GPU device (auto-detected if None)
            max_history_moves: Maximum moves to track in history

        Returns:
            BatchGameState with batch_size=1 matching the input state
        """
        if device is None:
            device = get_device()

        # Delegate to from_game_states for the actual conversion
        return cls.from_game_states([game_state], device, max_history_moves, lps_rounds_required)

    @classmethod
    def from_game_states(
        cls,
        game_states: list[GameState],
        device: torch.device | None = None,
        max_history_moves: int = 500,
        lps_rounds_required: int = 3,
    ) -> BatchGameState:
        """Create a BatchGameState from multiple CPU GameStates.

        Args:
            game_states: List of CPU GameStates to convert
            device: GPU device (auto-detected if None)
            max_history_moves: Maximum moves to track in history

        Returns:
            BatchGameState with batch_size=len(game_states)
        """
        if device is None:
            device = get_device()

        if not game_states:
            raise ValueError("game_states list cannot be empty")

        from app.models import BoardType
        from app.rules.core import get_rings_per_player

        batch_size = len(game_states)
        first_game = game_states[0]
        board_size = first_game.board.size
        # Get num_players from max_players attribute or count active players
        num_players = first_game.max_players if hasattr(first_game, 'max_players') else len(first_game.players)

        # Detect board type string from enum
        board_type_str = None
        board_type_enum = first_game.board_type if hasattr(first_game, 'board_type') else first_game.board.type
        if board_type_enum == BoardType.HEXAGONAL:
            board_type_str = "hexagonal"
        elif board_type_enum == BoardType.SQUARE19:
            board_type_str = "square19"
        else:
            board_type_str = "square8"

        # Get rings per player from rules
        rings_per_player = get_rings_per_player(board_type_enum)

        # Create empty batch
        batch = cls.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
            max_history_moves=max_history_moves,
            lps_rounds_required=lps_rounds_required,
            rings_per_player=rings_per_player,
            board_type=board_type_str,
        )

        def parse_position_key(key: str) -> tuple[int, int]:
            """Parse 'row,col' position key to (row, col) tuple."""
            parts = key.split(',')
            return int(parts[0]), int(parts[1])

        # Copy state from each game
        for g, game_state in enumerate(game_states):
            board = game_state.board

            # Copy stacks from dictionary
            for pos_key, stack in board.stacks.items():
                row, col = parse_position_key(pos_key)
                batch.stack_owner[g, row, col] = stack.controlling_player
                batch.stack_height[g, row, col] = stack.stack_height
                batch.cap_height[g, row, col] = stack.cap_height

            # Copy collapsed spaces
            for pos_key, territory_owner in board.collapsed_spaces.items():
                row, col = parse_position_key(pos_key)
                batch.is_collapsed[g, row, col] = True
                if territory_owner and territory_owner > 0:
                    batch.territory_owner[g, row, col] = territory_owner
                    batch.territory_count[g, territory_owner] += 1

            # Copy markers if present
            for pos_key, marker_info in board.markers.items():
                row, col = parse_position_key(pos_key)
                if hasattr(marker_info, 'player'):
                    batch.marker_owner[g, row, col] = marker_info.player
                elif hasattr(marker_info, 'owner'):
                    batch.marker_owner[g, row, col] = marker_info.owner

            # Copy territories if present
            for pos_key, territory in board.territories.items():
                row, col = parse_position_key(pos_key)
                if hasattr(territory, 'owner') and territory.owner > 0:
                    batch.territory_owner[g, row, col] = territory.owner

            # Copy player state - players is a list, find by player_number
            for player in game_state.players:
                p = player.player_number
                if 1 <= p <= num_players:
                    batch.rings_in_hand[g, p] = player.rings_in_hand
                    # Check for is_eliminated attribute (may not exist in all versions)
                    if hasattr(player, 'is_eliminated'):
                        batch.is_eliminated[g, p] = player.is_eliminated
                    # eliminated_rings may not exist in older states
                    if hasattr(player, 'eliminated_rings'):
                        batch.eliminated_rings[g, p] = player.eliminated_rings
                    if hasattr(player, 'buried_rings'):
                        batch.buried_rings[g, p] = player.buried_rings
                    if hasattr(player, 'rings_caused_eliminated'):
                        batch.rings_caused_eliminated[g, p] = player.rings_caused_eliminated
                    if hasattr(player, 'territory_spaces'):
                        batch.territory_count[g, p] = player.territory_spaces

            # Copy game metadata
            batch.current_player[g] = game_state.current_player
            batch.move_count[g] = len(game_state.move_history)

            # Map phase from CPU to GPU enum (December 2025: canonical phase mapping)
            from app.models import GamePhase as CPUGamePhase
            phase_map = {
                CPUGamePhase.RING_PLACEMENT: GamePhase.RING_PLACEMENT,
                CPUGamePhase.MOVEMENT: GamePhase.MOVEMENT,
                CPUGamePhase.CAPTURE: GamePhase.CAPTURE,
                CPUGamePhase.CHAIN_CAPTURE: GamePhase.CHAIN_CAPTURE,
                CPUGamePhase.LINE_PROCESSING: GamePhase.LINE_PROCESSING,
                CPUGamePhase.TERRITORY_PROCESSING: GamePhase.TERRITORY_PROCESSING,
                CPUGamePhase.FORCED_ELIMINATION: GamePhase.FORCED_ELIMINATION,
                CPUGamePhase.GAME_OVER: GamePhase.GAME_OVER,
            }
            batch.current_phase[g] = phase_map.get(game_state.current_phase, GamePhase.RING_PLACEMENT)

            # Game status
            if game_state.winner:
                batch.game_status[g] = GameStatus.COMPLETED
                batch.winner[g] = game_state.winner
            else:
                batch.game_status[g] = GameStatus.ACTIVE

            # Copy must_move_from constraint if present
            if hasattr(game_state, 'must_move_from') and game_state.must_move_from:
                batch.must_move_from_y[g] = game_state.must_move_from[0]
                batch.must_move_from_x[g] = game_state.must_move_from[1]

            # Copy LPS tracking state if present
            if hasattr(game_state, 'lps_round_index'):
                batch.lps_round_index[g] = game_state.lps_round_index
            if hasattr(game_state, 'lps_current_round_first_player') and game_state.lps_current_round_first_player:
                batch.lps_current_round_first_player[g] = game_state.lps_current_round_first_player
            if hasattr(game_state, 'lps_current_round_seen'):
                for p in game_state.lps_current_round_seen:
                    batch.lps_current_round_seen_mask[g, p] = True
            if hasattr(game_state, 'lps_current_round_real_action'):
                for p in game_state.lps_current_round_real_action:
                    batch.lps_current_round_real_action_mask[g, p] = True
            if hasattr(game_state, 'lps_exclusive_player_for_completed_round') and game_state.lps_exclusive_player_for_completed_round:
                batch.lps_exclusive_player_for_completed_round[g] = game_state.lps_exclusive_player_for_completed_round
            if hasattr(game_state, 'lps_consecutive_exclusive_rounds'):
                batch.lps_consecutive_exclusive_rounds[g] = game_state.lps_consecutive_exclusive_rounds
            if hasattr(game_state, 'lps_consecutive_exclusive_player') and game_state.lps_consecutive_exclusive_player:
                batch.lps_consecutive_exclusive_player[g] = game_state.lps_consecutive_exclusive_player

            # Copy move history
            for i, move in enumerate(game_state.move_history[:max_history_moves]):
                # Convert CPU MoveType (string enum) to GPU MoveType (int enum)
                batch.move_history[g, i, 0] = _cpu_move_type_to_gpu(move.type)
                batch.move_history[g, i, 1] = move.player
                if move.from_pos:
                    batch.move_history[g, i, 2] = move.from_pos.x if hasattr(move.from_pos, 'x') else move.from_pos[0]
                    batch.move_history[g, i, 3] = move.from_pos.y if hasattr(move.from_pos, 'y') else move.from_pos[1]
                # Move uses 'to' field, not 'to_pos'
                if move.to:
                    batch.move_history[g, i, 4] = move.to.x if hasattr(move.to, 'x') else move.to[0]
                    batch.move_history[g, i, 5] = move.to.y if hasattr(move.to, 'y') else move.to[1]
                # Copy phase from move if present (column 6) - December 2025: canonical mapping
                if hasattr(move, 'phase') and move.phase is not None:
                    # Map CPU phase to GPU phase (canonical phases)
                    from app.models import GamePhase as CPUGamePhase
                    move_phase_map = {
                        CPUGamePhase.RING_PLACEMENT: GamePhase.RING_PLACEMENT,
                        CPUGamePhase.MOVEMENT: GamePhase.MOVEMENT,
                        CPUGamePhase.CAPTURE: GamePhase.CAPTURE,
                        CPUGamePhase.CHAIN_CAPTURE: GamePhase.CHAIN_CAPTURE,
                        CPUGamePhase.LINE_PROCESSING: GamePhase.LINE_PROCESSING,
                        CPUGamePhase.TERRITORY_PROCESSING: GamePhase.TERRITORY_PROCESSING,
                        CPUGamePhase.FORCED_ELIMINATION: GamePhase.FORCED_ELIMINATION,
                        CPUGamePhase.GAME_OVER: GamePhase.GAME_OVER,
                    }
                    batch.move_history[g, i, 6] = move_phase_map.get(move.phase, GamePhase.RING_PLACEMENT)

        return batch

    def to_game_state(self, game_idx: int) -> GameState:
        """Convert a single game from the batch back to a CPU GameState.

        Args:
            game_idx: Index of the game in the batch

        Returns:
            CPU GameState matching the batch state for shadow validation
        """
        from datetime import datetime

        from app.models import (
            BoardState,
            BoardType,
            GamePhase as CPUGamePhase,
            GameState,
            GameStatus as CPUGameStatus,
            MarkerInfo,
            Player,
            Position,
            RingStack,
            TimeControl,
        )
        from app.rules.core import get_rings_per_player

        # Determine board type from size
        if self.board_size == 19:
            board_type = BoardType.SQUARE19
        elif self.board_size == 25:
            board_type = BoardType.HEXAGONAL
        else:
            board_type = BoardType.SQUARE8

        rings_per_player = get_rings_per_player(board_type)

        # Build stacks dict from GPU tensors
        stacks: dict[str, RingStack] = {}
        markers: dict[str, MarkerInfo] = {}
        collapsed_spaces: dict[str, int] = {}

        for row in range(self.board_size):
            for col in range(self.board_size):
                y, x = row, col  # Tensor coords
                pos_key = f"{col},{row}"  # CPU uses x,y format for keys

                # Check for stack
                stack_owner = self.stack_owner[game_idx, y, x].item()
                stack_height = self.stack_height[game_idx, y, x].item()
                if stack_owner > 0 and stack_height > 0:
                    cap_h = self.cap_height[game_idx, y, x].item()
                    # Reconstruct rings - cap rings on top, buried on bottom
                    rings = []
                    buried_count = stack_height - cap_h
                    # Bottom rings (buried from opponent) - approximation
                    for _ in range(buried_count):
                        rings.append(3 - stack_owner if stack_owner <= 2 else 1)
                    # Top rings (owner's color)
                    for _ in range(cap_h):
                        rings.append(stack_owner)

                    stacks[pos_key] = RingStack(
                        position=Position(x=col, y=row),
                        rings=rings,
                        stackHeight=stack_height,
                        capHeight=cap_h,
                        controllingPlayer=stack_owner,
                    )

                # Check for marker
                marker_owner = self.marker_owner[game_idx, y, x].item()
                if marker_owner > 0:
                    markers[pos_key] = MarkerInfo(
                        player=marker_owner,
                        position=Position(x=col, y=row),
                        type="regular",
                    )

                # Check for collapsed space
                if self.is_collapsed[game_idx, y, x].item():
                    terr_owner = self.territory_owner[game_idx, y, x].item()
                    if terr_owner > 0:
                        collapsed_spaces[pos_key] = terr_owner

        # Create BoardState
        board = BoardState(
            type=board_type,
            size=self.board_size,
            stacks=stacks,
            markers=markers,
            collapsedSpaces=collapsed_spaces,
        )

        # Create Player objects
        players: list[Player] = []
        total_rings_eliminated = 0
        for p in range(1, self.num_players + 1):
            elim_rings = int(self.eliminated_rings[game_idx, p].item())
            total_rings_eliminated += elim_rings
            players.append(
                Player(
                    id=f"player-{p}",
                    username=f"Player{p}",
                    type="ai",
                    playerNumber=p,
                    isReady=True,
                    timeRemaining=600000,  # 10 min default
                    ringsInHand=int(self.rings_in_hand[game_idx, p].item()),
                    eliminatedRings=elim_rings,
                    territorySpaces=0,  # Not tracked in GPU state
                )
            )

        # Map GPU phase to CPU phase (December 2025: canonical phases)
        gpu_phase_val = int(self.current_phase[game_idx].item())
        phase_map = {
            GamePhase.RING_PLACEMENT.value: CPUGamePhase.RING_PLACEMENT,
            GamePhase.MOVEMENT.value: CPUGamePhase.MOVEMENT,
            GamePhase.LINE_PROCESSING.value: CPUGamePhase.LINE_PROCESSING,
            GamePhase.TERRITORY_PROCESSING.value: CPUGamePhase.TERRITORY_PROCESSING,
            GamePhase.END_TURN.value: CPUGamePhase.MOVEMENT,  # Legacy fallback
            GamePhase.CAPTURE.value: CPUGamePhase.CAPTURE,
            GamePhase.CHAIN_CAPTURE.value: CPUGamePhase.CHAIN_CAPTURE,
            # Recovery is a movement-phase action in canonical rules.
            GamePhase.RECOVERY.value: CPUGamePhase.MOVEMENT,
            GamePhase.FORCED_ELIMINATION.value: CPUGamePhase.FORCED_ELIMINATION,
            GamePhase.GAME_OVER.value: CPUGamePhase.GAME_OVER,
        }
        cpu_phase = phase_map.get(gpu_phase_val, CPUGamePhase.RING_PLACEMENT)

        # Map GPU game status to CPU status
        gpu_status = int(self.game_status[game_idx].item())
        status_map = {
            GameStatus.ACTIVE.value: CPUGameStatus.ACTIVE,
            GameStatus.COMPLETED.value: CPUGameStatus.COMPLETED,
        }
        cpu_status = status_map.get(gpu_status, CPUGameStatus.ACTIVE)

        # Compute total rings in play
        total_rings_in_play = 0
        for stack in stacks.values():
            total_rings_in_play += len(stack.rings)

        # Current timestamp for metadata
        now = datetime.now()

        # Create GameState with all required fields
        winner_val = int(self.winner[game_idx].item())
        game_state = GameState(
            id=f"gpu-game-{game_idx}",
            boardType=board_type,
            board=board,
            players=players,
            currentPhase=cpu_phase,
            currentPlayer=int(self.current_player[game_idx].item()),
            moveHistory=[],
            timeControl=TimeControl(initialTime=600000, increment=0, type="none"),
            gameStatus=cpu_status,
            winner=winner_val if winner_val > 0 else None,
            createdAt=now,
            lastMoveAt=now,
            isRated=False,
            maxPlayers=self.num_players,
            totalRingsInPlay=total_rings_in_play,
            totalRingsEliminated=total_rings_eliminated,
            victoryThreshold=rings_per_player,
            territoryVictoryThreshold=50,  # Default 50%
        )

        # Set must_move_from constraint via stack key
        mmy = self.must_move_from_y[game_idx].item()
        mmx = self.must_move_from_x[game_idx].item()
        if mmy >= 0 and mmx >= 0:
            game_state.must_move_from_stack_key = f"{mmx},{mmy}"

        # Set LPS tracking state
        game_state.lps_round_index = int(self.lps_round_index[game_idx].item())
        fp = self.lps_current_round_first_player[game_idx].item()
        game_state.lps_current_round_first_player = int(fp) if fp > 0 else None

        # LPS actor mask (dict[int, bool])
        lps_mask: dict[int, bool] = {}
        for p in range(1, self.num_players + 1):
            if self.lps_current_round_real_action_mask[game_idx, p].item():
                lps_mask[p] = True
        game_state.lps_current_round_actor_mask = lps_mask

        excl = self.lps_exclusive_player_for_completed_round[game_idx].item()
        game_state.lps_exclusive_player_for_completed_round = int(excl) if excl > 0 else None

        return game_state

    def get_active_mask(self) -> torch.Tensor:
        """Return a boolean mask of games still in progress.

        Returns:
            (batch_size,) bool tensor, True for active games
        """
        return self.game_status == GameStatus.ACTIVE

    def count_active(self) -> int:
        """Return the number of games still in progress."""
        return int(self.get_active_mask().sum().item())

    def to_feature_tensor(self, history_length: int = 4) -> torch.Tensor:
        """Convert batch state to neural network input tensor.

        Creates a multi-channel representation suitable for CNN input:
        - Channels 0-4: Current player stack ownership (one-hot)
        - Channels 5-9: Stack heights (normalized)
        - Channels 10-14: Marker ownership
        - Channels 15-19: Territory ownership
        - Channel 20: Collapsed cells
        - Channels 21+: Historical board states

        Args:
            history_length: Number of historical states to include

        Returns:
            (batch_size, channels, board_size, board_size) float tensor
        """
        device = self.device
        bs = self.batch_size
        bz = self.board_size
        np = self.num_players

        # Base channels without history
        base_channels = 5 + 5 + 5 + 5 + 1  # 21 channels
        total_channels = base_channels + history_length * 5  # Add history

        features = torch.zeros((bs, total_channels, bz, bz), dtype=torch.float32, device=device)

        # Stack ownership (one-hot encoded)
        for p in range(np + 1):
            features[:, p, :, :] = (self.stack_owner == p).float()

        # Stack heights (normalized to 0-1)
        features[:, 5, :, :] = self.stack_height.float() / 5.0

        # Cap heights (normalized)
        features[:, 6, :, :] = self.cap_height.float() / 5.0

        # Marker ownership
        for p in range(np + 1):
            features[:, 10 + p, :, :] = (self.marker_owner == p).float()

        # Territory ownership
        for p in range(np + 1):
            features[:, 15 + p, :, :] = (self.territory_owner == p).float()

        # Collapsed cells
        features[:, 20, :, :] = self.is_collapsed.float()

        return features

    def extract_move_history(self, game_idx: int) -> list[dict[str, Any]]:
        """Extract move history for a single game as a list of dictionaries.

        Args:
            game_idx: Index of the game in the batch

        Returns:
            List of move dictionaries with keys: move_type, player, from_pos, to_pos, phase
        """
        moves = []
        for i in range(self.max_history_moves):
            move_type = self.move_history[game_idx, i, 0].item()
            if move_type < 0:
                break

            player = self.move_history[game_idx, i, 1].item()
            from_y = self.move_history[game_idx, i, 2].item()
            from_x = self.move_history[game_idx, i, 3].item()
            to_y = self.move_history[game_idx, i, 4].item()
            to_x = self.move_history[game_idx, i, 5].item()
            phase = self.move_history[game_idx, i, 6].item()

            move = {
                "move_type": MoveType(move_type).name,
                "player": player,
                "from_pos": (from_y, from_x) if from_y >= 0 else None,
                "to_pos": (to_y, to_x) if to_y >= 0 else None,
                "phase": GamePhase(phase).name if phase >= 0 else None,
            }
            moves.append(move)

        return moves

    def derive_victory_type(self, game_idx: int, max_moves: int) -> tuple[str, str | None]:
        """Derive the victory type and tiebreaker for a finished game.

        This analyzes the final game state to determine how the winner won,
        which is needed for accurate training data labeling.

        Args:
            game_idx: Index of the game in the batch
            max_moves: Maximum moves allowed (for stalemate detection)

        Returns:
            Tuple of (victory_type, tiebreaker_used) where:
            - victory_type: "territory", "elimination", "line", "stalemate", "lps", etc.
            - tiebreaker_used: "territory", "eliminations", "markers", etc. or None
        """
        from app.models import BoardType
        from app.rules.core import get_territory_victory_minimum, get_victory_threshold

        # Map board size to board type
        if self.board_size == 19:
            board_type = BoardType.SQUARE19
        elif self.board_size == 25:
            board_type = BoardType.HEXAGONAL
        else:
            board_type = BoardType.SQUARE8

        territory_threshold = get_territory_victory_minimum(board_type, self.num_players)
        elim_threshold = get_victory_threshold(board_type, self.num_players)

        winner = self.winner[game_idx].item()
        if winner <= 0:
            return "in_progress", None

        # Check for LPS victory (RR-CANON-R172)
        if self.lps_consecutive_exclusive_rounds[game_idx].item() >= self.lps_rounds_required:
            lps_winner = self.lps_consecutive_exclusive_player[game_idx].item()
            if lps_winner == winner:
                return "lps", None

        # Check territory victory
        if self.territory_count[game_idx, winner].item() >= territory_threshold:
            return "territory", None

        # Check elimination victory
        if self.rings_caused_eliminated[game_idx, winner].item() >= elim_threshold:
            return "elimination", None

        # Check if stalemate (move count at max)
        if self.move_count[game_idx].item() >= max_moves:
            tiebreaker = self._determine_tiebreaker(game_idx)
            return "stalemate", tiebreaker

        # Check line victory (need to look at board state)
        # detect_lines_vectorized returns (in_line_mask, line_position_counts)
        # where line_position_counts > 0 means the player has a line
        from app.ai.gpu_line_detection import detect_lines_vectorized
        try:
            _, line_counts = detect_lines_vectorized(self, winner)
            if line_counts[game_idx].item() > 0:
                return "line", None
        except Exception:
            # Line detection may fail on edge cases, fall through to unknown
            pass

        # Default fallback
        return "unknown", None

    def _determine_tiebreaker(self, game_idx: int) -> str:
        """Determine which tiebreaker was used for a stalemate.

        Args:
            game_idx: Index of the game in the batch

        Returns:
            Tiebreaker type: "territory", "eliminations", "markers", "last_actor"
        """
        winner = self.winner[game_idx].item()
        if winner <= 0:
            return "none"

        # Check territory advantage
        max_territory = 0
        territory_leader = 0
        for p in range(1, self.num_players + 1):
            t = self.territory_count[game_idx, p].item()
            if t > max_territory:
                max_territory = t
                territory_leader = p
        if territory_leader == winner and max_territory > 0:
            # Check if other players have same territory
            others_same = False
            for p in range(1, self.num_players + 1):
                if p != winner and self.territory_count[game_idx, p].item() == max_territory:
                    others_same = True
                    break
            if not others_same:
                return "territory"

        # Check eliminations advantage
        max_elims = 0
        elim_leader = 0
        for p in range(1, self.num_players + 1):
            e = self.rings_caused_eliminated[game_idx, p].item()
            if e > max_elims:
                max_elims = e
                elim_leader = p
        if elim_leader == winner and max_elims > 0:
            others_same = False
            for p in range(1, self.num_players + 1):
                if p != winner and self.rings_caused_eliminated[game_idx, p].item() == max_elims:
                    others_same = True
                    break
            if not others_same:
                return "eliminations"

        # Check marker counts
        marker_counts = []
        for p in range(1, self.num_players + 1):
            marker_counts.append((self.marker_owner[game_idx] == p).sum().item())
        if len(set(marker_counts)) > 1:
            return "markers"

        # Default to last_actor
        return "last_actor"


__all__ = [
    'BatchGameState',
]
