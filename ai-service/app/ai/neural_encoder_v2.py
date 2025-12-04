"""Enhanced neural network state encoder for RingRift (V2).

This module implements the 42-channel board tensor encoding and 28-feature
global vector from the Neural AI Architecture document. It's designed for
multiplayer (2-4 players) and supports square and hex board types.

Channel Layout (42 channels for 4-player):
├── Stack Ownership (4 channels): stack_top_owner[p] = 1.0 if player p controls
├── Stack Height (6 channels): bucketed heights [1], [2], [3], [4-5], [6-8], [9+]
├── Cap Height (6 channels): bucketed heights for cap_height
├── Buried Rings (4 channels): has_buried_ring[p] for each player
├── Markers (4 channels): marker_owner[p] for each player
├── Collapsed Territory (4 channels): territory_owner[p] for each player
├── Valid Moves Mask (1 channel): can_move_from[x,y]
├── Capture Targets Mask (1 channel): can_capture_at[x,y]
├── Line Threat Mask (4 channels): proximity to completing 5-line per player
├── Disconnection Risk Mask (4 channels): region disconnection risk per player
└── Misc (4 channels): reserved for future use

Global Features (28 features):
├── Per-player (4 × 5 = 20 features):
│   ├── rings_in_hand[p] / MAX_RINGS
│   ├── rings_on_board[p] / MAX_RINGS
│   ├── eliminated_rings[p] / MAX_RINGS
│   ├── territory_spaces[p] / BOARD_SIZE
│   └── num_stacks_controlled[p] / MAX_STACKS
├── Game state (5 features):
│   ├── move_number / MAX_MOVES
│   └── current_player_one_hot[4]
└── Board type (3 features):
    └── board_type_one_hot[3]
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List

from ..models import GameState, BoardState, BoardType, Position
from ..rules.geometry import BoardGeometry


# Constants
MAX_PLAYERS = 4
MAX_RINGS = 27  # Per player: 27 rings
MAX_STACKS = 100  # Reasonable upper bound
MAX_MOVES = 500
BOARD_SIZE_8 = 8
BOARD_SIZE_19 = 19
HEX_BOARD_SIZE = 21  # Canonical hex bounding box (radius 10)

# Height bucket boundaries
HEIGHT_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, float('inf'))]


def _height_to_bucket(height: int) -> int:
    """Convert stack height to bucket index (0-5)."""
    for i, (low, high) in enumerate(HEIGHT_BUCKETS):
        if low <= height <= high:
            return i
    return 5  # Max bucket


def _infer_board_size(board: BoardState) -> int:
    """Infer the canonical 2D board_size for CNN feature tensors."""
    if board.type == BoardType.SQUARE8:
        return 8
    if board.type == BoardType.SQUARE19:
        return 19
    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
        return 2 * radius + 1
    return board.size


def _pos_from_key(pos_key: str) -> Position:
    """Parse a BoardState dict key like 'x,y' or 'x,y,z' into a Position."""
    parts = pos_key.split(",")
    if len(parts) == 2:
        x, y = int(parts[0]), int(parts[1])
        return Position(x=x, y=y)
    if len(parts) == 3:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return Position(x=x, y=y, z=z)
    raise ValueError(f"Invalid position key: {pos_key!r}")


def _to_canonical_xy(board: BoardState, pos: Position) -> Tuple[int, int]:
    """Map a Position to canonical (cx, cy) in [0, board_size)."""
    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return pos.x, pos.y
    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
        return pos.x + radius, pos.y + radius
    return pos.x, pos.y


class NeuralEncoderV2:
    """Enhanced state encoder producing 42-channel board tensor + 28-feature global vector.

    This encoder is designed for training and inference with the RingRiftCNN_V2
    architecture. It supports 2-4 players and all board types.

    Usage:
        encoder = NeuralEncoderV2(num_players=3)
        board_tensor, global_features = encoder.encode(game_state)
        # board_tensor: [42, H, W] numpy array
        # global_features: [28] numpy array
    """

    def __init__(
        self,
        num_players: int = 4,
        include_move_masks: bool = True,
        include_line_threats: bool = True,
        include_disconnect_risk: bool = True,
    ):
        """Initialize the encoder.

        Args:
            num_players: Number of players (2-4). Channels are always 4-player
                         compatible, unused player channels are zeroed.
            include_move_masks: Whether to compute valid move masks (slower).
            include_line_threats: Whether to compute line threat features.
            include_disconnect_risk: Whether to compute disconnection risk.
        """
        self.num_players = min(num_players, MAX_PLAYERS)
        self.include_move_masks = include_move_masks
        self.include_line_threats = include_line_threats
        self.include_disconnect_risk = include_disconnect_risk

        # Channel layout
        self._channel_offset = {
            'stack_ownership': 0,       # 4 channels (0-3)
            'stack_height': 4,          # 6 channels (4-9)
            'cap_height': 10,           # 6 channels (10-15)
            'buried_rings': 16,         # 4 channels (16-19)
            'markers': 20,              # 4 channels (20-23)
            'collapsed_territory': 24,  # 4 channels (24-27)
            'valid_moves': 28,          # 1 channel (28)
            'capture_targets': 29,      # 1 channel (29)
            'line_threat': 30,          # 4 channels (30-33)
            'disconnect_risk': 34,      # 4 channels (34-37)
            'misc': 38,                 # 4 channels (38-41)
        }
        self.num_channels = 42
        self.num_global_features = 28

    def encode(
        self,
        game_state: GameState,
        current_player_pov: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode game state to neural network input tensors.

        Args:
            game_state: The game state to encode.
            current_player_pov: Player perspective (1-indexed). If None, uses
                               game_state.current_player.

        Returns:
            Tuple of (board_tensor, global_features):
                - board_tensor: [42, H, W] float32 numpy array
                - global_features: [28] float32 numpy array
        """
        board = game_state.board
        board_size = _infer_board_size(board)
        pov_player = current_player_pov or game_state.current_player

        # Initialize tensors
        features = np.zeros(
            (self.num_channels, board_size, board_size), dtype=np.float32
        )

        # Encode each feature group
        self._encode_stacks(features, board, board_size)
        self._encode_markers(features, board, board_size)
        self._encode_collapsed_territory(features, board, board_size)

        if self.include_move_masks:
            self._encode_move_masks(features, game_state, board_size)

        if self.include_line_threats:
            self._encode_line_threats(features, board, board_size)

        if self.include_disconnect_risk:
            self._encode_disconnect_risk(features, game_state, board_size)

        # Encode global features
        global_features = self._encode_global_features(game_state, board_size)

        return features, global_features

    def _encode_stacks(
        self,
        features: np.ndarray,
        board: BoardState,
        board_size: int,
    ) -> None:
        """Encode stack ownership, height, cap height, and buried rings."""
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            # Stack ownership (1-indexed player -> 0-indexed channel)
            owner = stack.controlling_player
            if 1 <= owner <= MAX_PLAYERS:
                features[self._channel_offset['stack_ownership'] + owner - 1, cx, cy] = 1.0

            # Stack height (bucketed)
            height = stack.stack_height
            bucket = _height_to_bucket(height)
            features[self._channel_offset['stack_height'] + bucket, cx, cy] = 1.0

            # Cap height (bucketed) - rings owned by controlling player at top
            cap_height = getattr(stack, 'cap_height', height)  # Fallback to full height
            cap_bucket = _height_to_bucket(cap_height)
            features[self._channel_offset['cap_height'] + cap_bucket, cx, cy] = 1.0

            # Buried rings - check if any player has rings below the cap
            if hasattr(stack, 'rings') and stack.rings:
                cap_owner = stack.controlling_player
                for ring in stack.rings[:-1]:  # All rings except top
                    ring_owner = getattr(ring, 'player', cap_owner)
                    if ring_owner != cap_owner and 1 <= ring_owner <= MAX_PLAYERS:
                        features[
                            self._channel_offset['buried_rings'] + ring_owner - 1,
                            cx, cy
                        ] = 1.0

    def _encode_markers(
        self,
        features: np.ndarray,
        board: BoardState,
        board_size: int,
    ) -> None:
        """Encode marker ownership."""
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            owner = marker.player
            if 1 <= owner <= MAX_PLAYERS:
                features[self._channel_offset['markers'] + owner - 1, cx, cy] = 1.0

    def _encode_collapsed_territory(
        self,
        features: np.ndarray,
        board: BoardState,
        board_size: int,
    ) -> None:
        """Encode collapsed territory ownership."""
        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if 1 <= owner <= MAX_PLAYERS:
                features[
                    self._channel_offset['collapsed_territory'] + owner - 1,
                    cx, cy
                ] = 1.0

    def _encode_move_masks(
        self,
        features: np.ndarray,
        game_state: GameState,
        board_size: int,
    ) -> None:
        """Encode valid move and capture target masks.

        Note: This is expensive as it requires computing legal moves.
        Consider caching or skipping during training.
        """
        board = game_state.board
        current_player = game_state.current_player

        # Valid moves mask: positions where current player can move from
        for pos_key, stack in board.stacks.items():
            if stack.controlling_player != current_player:
                continue

            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if 0 <= cx < board_size and 0 <= cy < board_size:
                features[self._channel_offset['valid_moves'], cx, cy] = 1.0

        # Capture targets mask: enemy stacks that can be captured
        for pos_key, stack in board.stacks.items():
            if stack.controlling_player == current_player:
                continue

            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if 0 <= cx < board_size and 0 <= cy < board_size:
                features[self._channel_offset['capture_targets'], cx, cy] = 1.0

    def _encode_line_threats(
        self,
        features: np.ndarray,
        board: BoardState,
        board_size: int,
    ) -> None:
        """Encode line threat proximity for each player.

        For each marker, compute a threat score based on adjacent same-color
        markers (potential to form lines).
        """
        # Count adjacent markers per position per player
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            owner = marker.player
            if not (1 <= owner <= MAX_PLAYERS):
                continue

            # Count adjacent same-color markers
            neighbors = BoardGeometry.get_adjacent_positions(
                pos, board.type, board.size
            )
            same_color_count = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(npos, board.type, board.size):
                    continue
                n_key = npos.to_key()
                n_marker = board.markers.get(n_key)
                if n_marker and n_marker.player == owner:
                    same_color_count += 1

            # Normalize to [0, 1] - more neighbors = higher threat
            max_neighbors = 6 if board.type == BoardType.HEXAGONAL else 8
            threat_score = min(same_color_count / (max_neighbors / 2), 1.0)
            features[
                self._channel_offset['line_threat'] + owner - 1,
                cx, cy
            ] = threat_score

    def _encode_disconnect_risk(
        self,
        features: np.ndarray,
        game_state: GameState,
        board_size: int,
    ) -> None:
        """Encode disconnection risk for territory regions.

        Simplified: mark territory cells that are on the border or have
        few same-color neighbors.
        """
        board = game_state.board

        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if not (1 <= owner <= MAX_PLAYERS):
                continue

            # Count adjacent same-territory cells
            neighbors = BoardGeometry.get_adjacent_positions(
                pos, board.type, board.size
            )
            same_territory_count = 0
            total_valid = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(npos, board.type, board.size):
                    continue
                total_valid += 1
                n_key = npos.to_key()
                n_owner = board.collapsed_spaces.get(n_key)
                if n_owner == owner:
                    same_territory_count += 1

            # Risk is higher when fewer neighbors are same territory
            if total_valid > 0:
                connectivity = same_territory_count / total_valid
                risk = 1.0 - connectivity
                features[
                    self._channel_offset['disconnect_risk'] + owner - 1,
                    cx, cy
                ] = risk

    def _encode_global_features(
        self,
        game_state: GameState,
        board_size: int,
    ) -> np.ndarray:
        """Encode global features vector (28 features).

        Layout:
        - [0-19]: Per-player features (4 players × 5 features)
        - [20-23]: Current player one-hot (4)
        - [24]: Move number / MAX_MOVES
        - [25-27]: Board type one-hot (3)
        """
        features = np.zeros(self.num_global_features, dtype=np.float32)

        # Per-player features
        for i, player in enumerate(game_state.players):
            if i >= MAX_PLAYERS:
                break
            base_idx = i * 5

            # Rings in hand (normalized)
            features[base_idx + 0] = player.rings_in_hand / MAX_RINGS

            # Rings on board (count stacks controlled by this player)
            board = game_state.board
            rings_on_board = 0
            stacks_controlled = 0
            for stack in board.stacks.values():
                if stack.controlling_player == player.player_number:
                    stacks_controlled += 1
                    rings_on_board += stack.stack_height
            features[base_idx + 1] = min(rings_on_board / MAX_RINGS, 1.0)

            # Eliminated rings (normalized)
            features[base_idx + 2] = player.eliminated_rings / MAX_RINGS

            # Territory spaces (normalized)
            territory_count = sum(
                1 for owner in board.collapsed_spaces.values()
                if owner == player.player_number
            )
            features[base_idx + 3] = territory_count / (board_size * board_size)

            # Stacks controlled (normalized)
            features[base_idx + 4] = stacks_controlled / MAX_STACKS

        # Current player one-hot (1-indexed to 0-indexed)
        current = game_state.current_player - 1
        if 0 <= current < MAX_PLAYERS:
            features[20 + current] = 1.0

        # Move number (normalized)
        move_number = getattr(game_state, 'move_number', 0)
        if move_number is None:
            move_number = 0
        features[24] = min(move_number / MAX_MOVES, 1.0)

        # Board type one-hot
        board_type = game_state.board.type
        if board_type == BoardType.SQUARE8:
            features[25] = 1.0
        elif board_type == BoardType.SQUARE19:
            features[26] = 1.0
        elif board_type == BoardType.HEXAGONAL:
            features[27] = 1.0

        return features

    def get_channel_indices(self, feature_name: str) -> Tuple[int, int]:
        """Get the start and end channel indices for a feature group.

        Args:
            feature_name: One of 'stack_ownership', 'stack_height', 'cap_height',
                         'buried_rings', 'markers', 'collapsed_territory',
                         'valid_moves', 'capture_targets', 'line_threat',
                         'disconnect_risk', 'misc'.

        Returns:
            Tuple of (start_idx, end_idx) - end is exclusive.
        """
        start = self._channel_offset[feature_name]
        # Find next feature start
        feature_names = list(self._channel_offset.keys())
        idx = feature_names.index(feature_name)
        if idx < len(feature_names) - 1:
            end = self._channel_offset[feature_names[idx + 1]]
        else:
            end = self.num_channels
        return start, end


def encode_state_v2(
    game_state: GameState,
    num_players: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to encode a game state with default settings.

    Args:
        game_state: The game state to encode.
        num_players: Number of players (for normalization).

    Returns:
        Tuple of (board_tensor, global_features).
    """
    encoder = NeuralEncoderV2(num_players=num_players)
    return encoder.encode(game_state)


def encode_batch_v2(
    game_states: List[GameState],
    num_players: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a batch of game states.

    Args:
        game_states: List of game states to encode.
        num_players: Number of players (for normalization).

    Returns:
        Tuple of (board_tensors, global_features):
            - board_tensors: [B, 42, H, W] numpy array
            - global_features: [B, 28] numpy array
    """
    if not game_states:
        return np.zeros((0, 42, 8, 8), dtype=np.float32), np.zeros((0, 28), dtype=np.float32)

    encoder = NeuralEncoderV2(num_players=num_players)

    # Get board size from first state
    board_size = _infer_board_size(game_states[0].board)

    batch_boards = np.zeros(
        (len(game_states), encoder.num_channels, board_size, board_size),
        dtype=np.float32
    )
    batch_globals = np.zeros(
        (len(game_states), encoder.num_global_features),
        dtype=np.float32
    )

    for i, state in enumerate(game_states):
        board_tensor, global_features = encoder.encode(state)
        batch_boards[i] = board_tensor
        batch_globals[i] = global_features

    return batch_boards, batch_globals
