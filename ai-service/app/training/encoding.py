"""
Encoding utilities for RingRift training data.

This module provides state and action encoding for both square and hexagonal
boards. The HexStateEncoder class handles the conversion of hex game states
to feature tensors suitable for the HexNeuralNet model.
"""

from typing import List, Tuple, Union, Optional
import numpy as np

from app.models import (
    Move,
    BoardState,
    GameState,
    BoardType,
    Position,
    GamePhase,
)
from app.ai.neural_net import (
    NeuralNetAI,
    ActionEncoderHex,
    INVALID_MOVE_INDEX,
    HEX_BOARD_SIZE,
    P_HEX,
    _to_canonical_xy,
    _from_canonical_xy,
    _pos_from_key,
)
from app.rules.geometry import BoardGeometry
from app.rules.core import get_rings_per_player
from app.ai.game_state_utils import infer_num_players, select_threat_opponent


def encode_legal_moves(
    moves: List[Move],
    neural_net: NeuralNetAI,
    board_context: Union[BoardState, GameState],
) -> List[int]:
    """
    Encode a list of legal moves into their policy indices for the given
    board context.

    The returned indices live in the fixed MAX_N×MAX_N policy head used by
    NeuralNetAI. Moves that cannot be encoded (INVALID_MOVE_INDEX) are
    filtered out.
    """
    encoded_moves: List[int] = []
    for m in moves:
        idx = neural_net.encode_move(m, board_context)
        if idx != INVALID_MOVE_INDEX:
            encoded_moves.append(idx)
    return encoded_moves


def encode_hex_legal_moves(
    moves: List[Move],
    encoder: ActionEncoderHex,
    board: BoardState,
) -> List[int]:
    """
    Encode a list of legal moves for hex boards using ActionEncoderHex.

    The returned indices live in the P_HEX policy head (91,876 actions).
    Moves that cannot be encoded (INVALID_MOVE_INDEX) are filtered out.
    """
    encoded_moves: List[int] = []
    for m in moves:
        idx = encoder.encode_move(m, board)
        if idx != INVALID_MOVE_INDEX:
            encoded_moves.append(idx)
    return encoded_moves


class HexStateEncoder:
    """
    State encoder for hexagonal boards.

    Converts hex game states to feature tensors suitable for HexNeuralNet.
    The encoder outputs feature tensors of shape (C, H, W) where:
    - C = 10 channels (stacks, markers, collapsed, liberties, line potential)
    - H = W = 25 (for canonical N=12 hex board, embedded in 25x25 grid)

    Coordinate Systems:
    - Axial coordinates (q, r): Used in game state, q+r+s=0 with s=-q-r
    - Canonical grid (cx, cy): 0-indexed grid for CNN, where:
        cx = q + radius
        cy = r + radius
      For radius=12, this maps q,r ∈ [-12,12] to cx,cy ∈ [0,24]

    Feature Channels:
        0: Current player stacks (height normalized to [0,1])
        1: Opponent stacks (height normalized)
        2: Current player markers
        3: Opponent markers
        4: Current player collapsed spaces (territory)
        5: Opponent collapsed spaces
        6: Current player liberties (normalized empty neighbors)
        7: Opponent liberties
        8: Current player line potential (friendly marker neighbors)
        9: Opponent line potential

    Global Features (10 dims):
        0-4: Compressed one-hot game phase encoding
             (ring_placement, movement, capture, line_processing,
              territory_processing + forced_elimination)
        5: Current player rings in hand (normalized)
        6: Opponent rings in hand (normalized)
        7: Current player eliminated rings (normalized)
        8: Opponent eliminated rings (normalized)
        9: Current turn indicator (always 1.0)
    """

    # Canonical hex board dimensions
    BOARD_SIZE = HEX_BOARD_SIZE  # 25
    RADIUS = (HEX_BOARD_SIZE - 1) // 2  # 12
    NUM_CHANNELS = 10
    NUM_GLOBAL_FEATURES = 10
    POLICY_SIZE = P_HEX  # 91,876

    def __init__(self, board_size: int = HEX_BOARD_SIZE):
        """
        Initialize the hex state encoder.

        Args:
            board_size: Size of the canonical grid (2*radius + 1).
                        Default is 25 for radius=12 hex boards.
        """
        self.board_size = board_size
        self.radius = (board_size - 1) // 2
        self.action_encoder = ActionEncoderHex(
            board_size=board_size, policy_size=P_HEX
        )

        # Precompute valid hex cell mask for the 25x25 grid
        # Only cells where |q| + |r| + |s| <= 2*radius are valid
        self._valid_mask = self._build_valid_mask()

    def _build_valid_mask(self) -> np.ndarray:
        """
        Build a mask of valid hex cells within the canonical grid.

        Returns:
            Boolean array of shape (board_size, board_size) where True
            indicates a valid hex cell.
        """
        mask = np.zeros((self.board_size, self.board_size), dtype=bool)
        for cy in range(self.board_size):
            for cx in range(self.board_size):
                q = cx - self.radius
                r = cy - self.radius
                s = -q - r
                in_bounds = (
                    abs(q) <= self.radius
                    and abs(r) <= self.radius
                    and abs(s) <= self.radius
                )
                if in_bounds:
                    mask[cy, cx] = True
        return mask

    def get_valid_mask(self) -> np.ndarray:
        """
        Get the valid hex cell mask for masking CNN outputs.

        Returns:
            Boolean array of shape (board_size, board_size)
        """
        return self._valid_mask.copy()

    def get_valid_mask_tensor(self) -> np.ndarray:
        """
        Get valid mask as float tensor suitable for neural network masking.

        Returns:
            Float array of shape (1, board_size, board_size) with 1.0 for
            valid cells and 0.0 for padding.
        """
        return self._valid_mask.astype(np.float32)[np.newaxis, :, :]

    def axial_to_canonical(self, q: int, r: int) -> Tuple[int, int]:
        """
        Convert axial coordinates (q, r) to canonical grid coords (cx, cy).

        Args:
            q: Axial q coordinate
            r: Axial r coordinate

        Returns:
            Tuple of (cx, cy) canonical grid coordinates
        """
        return q + self.radius, r + self.radius

    def canonical_to_axial(self, cx: int, cy: int) -> Tuple[int, int]:
        """
        Convert canonical grid coords (cx, cy) to axial coordinates (q, r).

        Args:
            cx: Canonical x coordinate
            cy: Canonical y coordinate

        Returns:
            Tuple of (q, r) axial coordinates
        """
        return cx - self.radius, cy - self.radius

    def encode_position(
        self, pos: Position, board: BoardState
    ) -> Optional[Tuple[int, int]]:
        """
        Encode a game Position to canonical grid coordinates.

        Args:
            pos: Position with x, y (and optional z) coordinates
            board: Board state for context

        Returns:
            Tuple of (cx, cy) or None if position is invalid
        """
        cx, cy = _to_canonical_xy(board, pos)
        if 0 <= cx < self.board_size and 0 <= cy < self.board_size:
            return cx, cy
        return None

    def decode_position(
        self, cx: int, cy: int, board: BoardState
    ) -> Optional[Position]:
        """
        Decode canonical grid coordinates to a game Position.

        Args:
            cx: Canonical x coordinate
            cy: Canonical y coordinate
            board: Board state for context

        Returns:
            Position object or None if coordinates are invalid
        """
        return _from_canonical_xy(board, cx, cy)

    def encode(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a hex game state to feature tensors.

        Args:
            state: The game state to encode

        Returns:
            Tuple of (board_features, global_features):
            - board_features: shape (10, board_size, board_size)
            - global_features: shape (10,)
        """
        board = state.board

        if board.type != BoardType.HEXAGONAL:
            raise ValueError(
                f"HexStateEncoder requires HEXAGONAL board type, "
                f"got {board.type}"
            )

        # Initialize feature tensor
        features = np.zeros(
            (self.NUM_CHANNELS, self.board_size, self.board_size),
            dtype=np.float32,
        )

        current_player = state.current_player

        # Channel 0/1: Stacks (height normalized)
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            coords = self.encode_position(pos, board)
            if coords is None:
                continue
            cx, cy = coords

            val = min(stack.stack_height / 5.0, 1.0)
            if stack.controlling_player == current_player:
                features[0, cy, cx] = val
            else:
                features[1, cy, cx] = val

        # Channel 2/3: Markers
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            coords = self.encode_position(pos, board)
            if coords is None:
                continue
            cx, cy = coords

            if marker.player == current_player:
                features[2, cy, cx] = 1.0
            else:
                features[3, cy, cx] = 1.0

        # Channel 4/5: Collapsed spaces (territory)
        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            coords = self.encode_position(pos, board)
            if coords is None:
                continue
            cx, cy = coords

            if owner == current_player:
                features[4, cy, cx] = 1.0
            else:
                features[5, cy, cx] = 1.0

        # Channel 6/7: Liberties (normalized empty neighbor count)
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            coords = self.encode_position(pos, board)
            if coords is None:
                continue
            cx, cy = coords

            neighbors = BoardGeometry.get_adjacent_positions(
                pos, board.type, board.size
            )
            liberties = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos, board.type, board.size
                ):
                    continue
                n_key = npos.to_key()
                not_stack = n_key not in board.stacks
                not_collapsed = n_key not in board.collapsed_spaces
                if not_stack and not_collapsed:
                    liberties += 1

            # Hex has max 6 neighbors
            val = min(liberties / 6.0, 1.0)
            if stack.controlling_player == current_player:
                features[6, cy, cx] = val
            else:
                features[7, cy, cx] = val

        # Channel 8/9: Line potential (same-color marker neighbors)
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            coords = self.encode_position(pos, board)
            if coords is None:
                continue
            cx, cy = coords

            neighbors = BoardGeometry.get_adjacent_positions(
                pos, board.type, board.size
            )
            neighbor_count = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos, board.type, board.size
                ):
                    continue
                n_key = npos.to_key()
                neighbor_marker = board.markers.get(n_key)
                if (
                    neighbor_marker is not None
                    and neighbor_marker.player == marker.player
                ):
                    neighbor_count += 1

            # Normalize by half of max neighbors (3 typical for line building)
            val = min(neighbor_count / 3.0, 1.0)
            if marker.player == current_player:
                features[8, cy, cx] = val
            else:
                features[9, cy, cx] = val

        # Global features
        globals_vec = np.zeros(self.NUM_GLOBAL_FEATURES, dtype=np.float32)

        # Phase one-hot (indices 0-4).
        # We currently encode the 7 canonical phases into 5 slots by
        # compressing `forced_elimination` into the same slot as
        # `territory_processing`. This keeps the global feature size
        # stable for existing models while still distinguishing the
        # territory/cleanup portion of the turn from placement/movement/
        # capture phases.
        phase_index_map = {
            GamePhase.RING_PLACEMENT: 0,
            GamePhase.MOVEMENT: 1,
            GamePhase.CAPTURE: 2,
            GamePhase.LINE_PROCESSING: 3,
            GamePhase.TERRITORY_PROCESSING: 4,
            GamePhase.FORCED_ELIMINATION: 4,
        }
        phase_idx = phase_index_map.get(state.current_phase)
        if phase_idx is not None and 0 <= phase_idx < 5:
            globals_vec[phase_idx] = 1.0

        # Rings info (indices 5-8)
        my_player = next(
            (p for p in state.players if p.player_number == current_player),
            None,
        )
        opp_player = None
        num_players = infer_num_players(state)
        if num_players <= 2:
            opp_player = next(
                (
                    p
                    for p in state.players
                    if p.player_number != current_player
                ),
                None,
            )
        else:
            threat_pid = select_threat_opponent(state, current_player)
            if threat_pid is not None:
                opp_player = next(
                    (
                        p
                        for p in state.players
                        if p.player_number == threat_pid
                    ),
                    None,
                )
            if opp_player is None:
                opp_player = next(
                    (
                        p
                        for p in state.players
                        if p.player_number != current_player
                    ),
                    None,
                )

        if my_player:
            rings_per_player = float(get_rings_per_player(state.board.type))
            globals_vec[5] = my_player.rings_in_hand / rings_per_player
            globals_vec[7] = my_player.eliminated_rings / rings_per_player

        if opp_player:
            rings_per_player = float(get_rings_per_player(state.board.type))
            globals_vec[6] = opp_player.rings_in_hand / rings_per_player
            globals_vec[8] = opp_player.eliminated_rings / rings_per_player

        # Turn indicator (index 9)
        globals_vec[9] = 1.0

        return features, globals_vec

    def encode_with_history(
        self,
        state: GameState,
        history_frames: List[np.ndarray],
        history_length: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a game state with historical feature frames.

        Args:
            state: Current game state
            history_frames: List of previous frame features (newest last)
            history_length: Number of history frames to include

        Returns:
            Tuple of (stacked_features, global_features):
            - stacked_features: shape (10 * (history_length + 1), H, W)
            - global_features: shape (10,)
        """
        features, globals_vec = self.encode(state)

        # Build history list (newest first)
        hist = history_frames[::-1][:history_length]

        # Pad with zeros if not enough history
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))

        # Stack: current + history
        stack = np.concatenate([features] + hist, axis=0)

        return stack, globals_vec

    def encode_move(self, move: Move, board: BoardState) -> int:
        """
        Encode a move to a policy index.

        Args:
            move: The move to encode
            board: Board state for context

        Returns:
            Policy index in [0, P_HEX) or INVALID_MOVE_INDEX (-1) if invalid
        """
        return self.action_encoder.encode_move(move, board)

    def decode_move(self, index: int, state: GameState) -> Optional[Move]:
        """
        Decode a policy index to a move.

        Args:
            index: Policy index in [0, P_HEX)
            state: Game state for context

        Returns:
            Move object or None if index is invalid
        """
        return self.action_encoder.decode_move(index, state)

    def encode_legal_moves(
        self, moves: List[Move], board: BoardState
    ) -> List[int]:
        """
        Encode a list of legal moves.

        Args:
            moves: List of legal moves
            board: Board state for context

        Returns:
            List of valid policy indices
        """
        return encode_hex_legal_moves(moves, self.action_encoder, board)

    def create_policy_target(
        self,
        move_indices: List[int],
        move_probs: List[float],
    ) -> np.ndarray:
        """
        Create a dense policy target vector from sparse move probabilities.

        Args:
            move_indices: List of policy indices
            move_probs: List of probabilities corresponding to indices

        Returns:
            Dense policy vector of shape (P_HEX,)
        """
        policy = np.zeros(self.POLICY_SIZE, dtype=np.float32)
        for idx, prob in zip(move_indices, move_probs):
            if 0 <= idx < self.POLICY_SIZE:
                policy[idx] = prob
        return policy

    def create_sparse_policy_target(
        self,
        move_indices: List[int],
        move_probs: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sparse policy target arrays.

        Args:
            move_indices: List of policy indices
            move_probs: List of probabilities corresponding to indices

        Returns:
            Tuple of (indices_array, values_array)
        """
        valid_indices = []
        valid_probs = []
        for idx, prob in zip(move_indices, move_probs):
            if 0 <= idx < self.POLICY_SIZE:
                valid_indices.append(idx)
                valid_probs.append(prob)

        return (
            np.array(valid_indices, dtype=np.int32),
            np.array(valid_probs, dtype=np.float32),
        )


def detect_board_type_from_features(features: np.ndarray) -> BoardType:
    """
    Detect board type from feature tensor shape.

    Args:
        features: Feature tensor of shape (C, H, W) or (N, C, H, W)

    Returns:
        Detected BoardType

    Raises:
        ValueError: If shape cannot be mapped to a known board type
    """
    if features.ndim == 4:
        # Batch dimension
        _, _, h, w = features.shape
    elif features.ndim == 3:
        _, h, w = features.shape
    else:
        raise ValueError(
            f"Expected 3D or 4D tensor, got shape {features.shape}"
        )

    if h == w == 8:
        return BoardType.SQUARE8
    elif h == w == 19:
        return BoardType.SQUARE19
    elif h == w == 25:
        return BoardType.HEXAGONAL
    else:
        raise ValueError(
            f"Cannot detect board type from spatial size {h}x{w}. "
            f"Expected 8x8, 19x19, or 25x25."
        )


def get_encoder_for_board_type(
    board_type: BoardType,
) -> Union[HexStateEncoder, None]:
    """
    Get the appropriate encoder for a board type.

    Args:
        board_type: The board type

    Returns:
        HexStateEncoder for HEXAGONAL boards, None for square boards
        (square boards use NeuralNetAI directly)
    """
    if board_type == BoardType.HEXAGONAL:
        return HexStateEncoder()
    return None
