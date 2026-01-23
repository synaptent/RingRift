"""
Encoding utilities for RingRift training data.

This module provides state and action encoding for both square and hexagonal
boards. The HexStateEncoder class handles the conversion of hex game states
to feature tensors suitable for the HexNeuralNet model.
"""


from __future__ import annotations

import numpy as np

from app.ai.game_state_utils import (
    infer_num_players,
    infer_rings_per_player,
    select_threat_opponent,
)
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    INVALID_MOVE_INDEX,
    P_HEX,
    POLICY_SIZE_HEX8,
    ActionEncoderHex,
    NeuralNetAI,
    _from_canonical_xy,
    _pos_from_key,
    _to_canonical_xy,
)
from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    Move,
    Position,
)
from app.rules.geometry import BoardGeometry


def encode_legal_moves(
    moves: list[Move],
    neural_net: NeuralNetAI,
    board_context: BoardState | GameState,
) -> list[int]:
    """
    Encode a list of legal moves into their policy indices for the given
    board context.

    The returned indices live in the fixed MAX_N×MAX_N policy head used by
    NeuralNetAI. Moves that cannot be encoded (INVALID_MOVE_INDEX) are
    filtered out.
    """
    encoded_moves: list[int] = []
    for m in moves:
        idx = neural_net.encode_move(m, board_context)
        if idx != INVALID_MOVE_INDEX:
            encoded_moves.append(idx)
    return encoded_moves


def encode_hex_legal_moves(
    moves: list[Move],
    encoder: ActionEncoderHex,
    board: BoardState,
) -> list[int]:
    """
    Encode a list of legal moves for hex boards using ActionEncoderHex.

    The returned indices live in the P_HEX policy head (91,876 actions).
    Moves that cannot be encoded (INVALID_MOVE_INDEX) are filtered out.
    """
    encoded_moves: list[int] = []
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

    Global Features (20 dims):
        0-4: Compressed one-hot game phase encoding
             (ring_placement, movement, capture, line_processing,
              territory_processing + forced_elimination)
        5: Current player rings in hand (normalized)
        6: Opponent rings in hand (normalized)
        7: Current player eliminated rings (normalized)
        8: Opponent eliminated rings (normalized)
        9: Current turn indicator (always 1.0)
        10: Is 3-player game (1.0 if 3p, else 0.0)
        11: Is 4-player game (1.0 if 4p, else 0.0)
        12: Current player territory count (normalized)
        13: Opponent territory count (normalized)
        14: Current player marker count (normalized)
        15: Opponent marker count (normalized)
        16: Current player stack count (normalized)
        17: Opponent stack count (normalized)
        18: Game progress (move_number / 200, capped at 1.0) [feature_version=1]
            Chain-capture flag [feature_version>=2]
        19: Reserved (always 0.0) [feature_version=1]
            Forced-elimination flag [feature_version>=2]
    """

    # Canonical hex board dimensions
    BOARD_SIZE = HEX_BOARD_SIZE  # 25
    RADIUS = (HEX_BOARD_SIZE - 1) // 2  # 12
    NUM_CHANNELS = 10
    NUM_GLOBAL_FEATURES = 20  # Match NeuralNetAI for model compatibility
    POLICY_SIZE = P_HEX  # 91,876

    def __init__(
        self,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        feature_version: int = 1,
    ):
        """
        Initialize the hex state encoder.

        Args:
            board_size: Size of the canonical grid (2*radius + 1).
                        Default is 25 for radius=12 hex boards.
                        Use 9 for hex8 (radius=4) boards.
            policy_size: Size of the policy head. Default is P_HEX (91,876).
                        Use POLICY_SIZE_HEX8 (4,500) for hex8 boards.
            feature_version: Feature encoding version for global feature layout.
            feature_version: Feature encoding version for global feature layout.
        """
        self.board_size = board_size
        self.radius = (board_size - 1) // 2
        self.policy_size = policy_size
        self.feature_version = int(feature_version)
        self.action_encoder = ActionEncoderHex(
            board_size=board_size, policy_size=policy_size
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

    def axial_to_canonical(self, q: int, r: int) -> tuple[int, int]:
        """
        Convert axial coordinates (q, r) to canonical grid coords (cx, cy).

        Args:
            q: Axial q coordinate
            r: Axial r coordinate

        Returns:
            Tuple of (cx, cy) canonical grid coordinates
        """
        return q + self.radius, r + self.radius

    def canonical_to_axial(self, cx: int, cy: int) -> tuple[int, int]:
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
    ) -> tuple[int, int] | None:
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
    ) -> Position | None:
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

    def encode_state(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a hex game state to feature tensors.

        This is the canonical API method name used by the export pipeline.
        See also: encode() which is an alias for backwards compatibility.

        Args:
            state: The game state to encode

        Returns:
            Tuple of (board_features, global_features):
            - board_features: shape (10, board_size, board_size)
            - global_features: shape (20,)
        """
        board = state.board

        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            raise ValueError(
                f"HexStateEncoder requires HEXAGONAL or HEX8 board type, "
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
            GamePhase.CHAIN_CAPTURE: 2,  # Sub-phase of CAPTURE, shares index
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

        rings_per_player = float(infer_rings_per_player(state))
        if my_player:
            globals_vec[5] = my_player.rings_in_hand / rings_per_player
            globals_vec[7] = my_player.eliminated_rings / rings_per_player

        if opp_player:
            globals_vec[6] = opp_player.rings_in_hand / rings_per_player
            globals_vec[8] = opp_player.eliminated_rings / rings_per_player

        # Turn indicator (index 9)
        globals_vec[9] = 1.0

        # Extended global features (indices 10-19) for model compatibility
        # Player count indicators
        globals_vec[10] = 1.0 if num_players == 3 else 0.0
        globals_vec[11] = 1.0 if num_players == 4 else 0.0

        # Territory counts (normalized by board cell count ~397 for hex)
        max_cells = 397.0  # Approximate valid cells in hex board
        my_territory = sum(
            1 for owner in board.collapsed_spaces.values()
            if owner == current_player
        )
        opp_territory = sum(
            1 for owner in board.collapsed_spaces.values()
            if owner != current_player
        )
        globals_vec[12] = min(my_territory / max_cells, 1.0)
        globals_vec[13] = min(opp_territory / max_cells, 1.0)

        # Marker counts (normalized by typical max ~50)
        max_markers = 50.0
        my_markers = sum(
            1 for m in board.markers.values()
            if m.player == current_player
        )
        opp_markers = sum(
            1 for m in board.markers.values()
            if m.player != current_player
        )
        globals_vec[14] = min(my_markers / max_markers, 1.0)
        globals_vec[15] = min(opp_markers / max_markers, 1.0)

        # Stack counts (normalized by typical max ~30)
        max_stacks = 30.0
        my_stacks = sum(
            1 for s in board.stacks.values()
            if s.controlling_player == current_player
        )
        opp_stacks = sum(
            1 for s in board.stacks.values()
            if s.controlling_player != current_player
        )
        globals_vec[16] = min(my_stacks / max_stacks, 1.0)
        globals_vec[17] = min(opp_stacks / max_stacks, 1.0)

        if self.feature_version >= 2:
            globals_vec[18] = (
                1.0 if state.current_phase == GamePhase.CHAIN_CAPTURE else 0.0
            )
            globals_vec[19] = (
                1.0
                if state.current_phase == GamePhase.FORCED_ELIMINATION
                else 0.0
            )
        else:
            # Game progress (move number normalized by expected game length)
            move_number = getattr(state, 'turn_number', 0)
            if move_number == 0 and hasattr(state, 'move_history'):
                move_number = len(state.move_history)
            globals_vec[18] = min(move_number / 200.0, 1.0)
            # Reserved (index 19) - already 0.0 from initialization

        return features, globals_vec

    def encode(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode state with frame stacking for inference.

        Since the model expects history-stacked features (40 channels = 10 × 4),
        but inference only has the current state, we replicate the current
        frame to fill all history slots. This is a reasonable approximation
        for single-frame inference.

        Returns:
            - stacked_features: shape (40, H, W) - 4 copies of current frame
            - global_features: shape (20,)
        """
        features, globals_vec = self.encode_state(state)
        # Stack current frame 4 times to match training format (history_length=3)
        stacked = np.concatenate([features] * 4, axis=0)
        return stacked, globals_vec

    def encode_with_history(
        self,
        state: GameState,
        history_frames: list[np.ndarray],
        history_length: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a game state with historical feature frames.

        Args:
            state: Current game state
            history_frames: List of previous frame features (newest last)
            history_length: Number of history frames to include

        Returns:
            Tuple of (stacked_features, global_features):
            - stacked_features: shape (10 * (history_length + 1), H, W)
            - global_features: shape (20,)
        """
        features, globals_vec = self.encode_state(state)

        # Build history list (newest first)
        hist = history_frames[::-1][:history_length]

        # Pad with zeros if not enough history
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))

        # Stack: current + history
        stack = np.concatenate([features, *hist], axis=0)

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

    def decode_move(self, index: int, state: GameState) -> Move | None:
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
        self, moves: list[Move], board: BoardState
    ) -> list[int]:
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
        move_indices: list[int],
        move_probs: list[float],
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
        for idx, prob in zip(move_indices, move_probs, strict=False):
            if 0 <= idx < self.POLICY_SIZE:
                policy[idx] = prob
        return policy

    def create_sparse_policy_target(
        self,
        move_indices: list[int],
        move_probs: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
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
        for idx, prob in zip(move_indices, move_probs, strict=False):
            if 0 <= idx < self.POLICY_SIZE:
                valid_indices.append(idx)
                valid_probs.append(prob)

        return (
            np.array(valid_indices, dtype=np.int32),
            np.array(valid_probs, dtype=np.float32),
        )


class HexStateEncoderV3:
    """
    V3 State encoder for hexagonal boards with enhanced feature channels.

    Produces 16 base channels (vs 10 in V2) to support HexNeuralNet_v3
    spatial policy architecture. With 4 frame stacking (history_length=3),
    this produces 64 total input channels.

    Feature Channels (16 total):
        0-1: Player stacks (height normalized to [0,1]) - same as V2
        2-3: Player markers - same as V2
        4-5: Collapsed spaces (territory) - same as V2
        6-7: Liberties (normalized empty neighbors) - same as V2
        8-9: Line potential (friendly marker neighbors) - same as V2
        10-11: Contestation (cells with both friendly & enemy neighbors)
        12-13: Stack height gradient (sum of neighboring stack heights)
        14-15: Ring placement validity (can place ring here, normalized)

    Global Features (20 dims for V3):
        0-4: Game phase one-hot
        5-8: Ring counts (current/opponent rings_in_hand, eliminated_rings)
        9: Turn indicator
        10-14: Board stats (marker/stack totals and ratios)
        15-16: Chain-capture / forced-elimination flags (feature_version>=2)
        17-19: Reserved for future use
    """

    BOARD_SIZE = HEX_BOARD_SIZE  # 25
    RADIUS = (HEX_BOARD_SIZE - 1) // 2  # 12
    NUM_CHANNELS = 16  # 16 base channels for V3
    NUM_GLOBAL_FEATURES = 20  # Extended global features for V3
    POLICY_SIZE = P_HEX  # 91,876

    def __init__(
        self,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
        feature_version: int = 1,
    ):
        """Initialize the V3 hex state encoder.

        Args:
            board_size: Size of the canonical grid (2*radius + 1).
                        Default is 25 for radius=12 hex boards.
                        Use 9 for hex8 (radius=4) boards.
            policy_size: Size of the policy head. Default is P_HEX (91,876).
                        Use POLICY_SIZE_HEX8 (4,500) for hex8 boards.
        """
        self.board_size = board_size
        self.radius = (board_size - 1) // 2
        self.policy_size = policy_size
        self.feature_version = int(feature_version)
        self.action_encoder = ActionEncoderHex(
            board_size=board_size, policy_size=policy_size
        )
        self._valid_mask = self._build_valid_mask()

    def _build_valid_mask(self) -> np.ndarray:
        """Build a mask of valid hex cells within the canonical grid."""
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
        """Get the valid hex cell mask for masking CNN outputs."""
        return self._valid_mask.copy()

    def get_valid_mask_tensor(self) -> np.ndarray:
        """Get valid mask as float tensor for neural network masking."""
        return self._valid_mask.astype(np.float32)

    def encode_position(
        self, pos: Position, board: BoardState
    ) -> tuple[int, int] | None:
        """Convert axial position to canonical grid coordinates."""
        q, r = pos.x, pos.y
        cx = q + self.radius
        cy = r + self.radius
        if 0 <= cx < self.board_size and 0 <= cy < self.board_size and self._valid_mask[cy, cx]:
            return (cx, cy)
        return None

    def encode_state(
        self, state: GameState
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a hex game state to feature tensors.

        Returns:
            Tuple of (board_features, global_features):
            - board_features: shape (16, board_size, board_size)
            - global_features: shape (20,)
        """
        board = state.board

        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            raise ValueError(
                f"HexStateEncoderV3 requires HEXAGONAL or HEX8 board type, "
                f"got {board.type}"
            )

        features = np.zeros(
            (self.NUM_CHANNELS, self.board_size, self.board_size),
            dtype=np.float32,
        )

        current_player = state.current_player

        # === CHANNELS 0-9: Same as V2 ===

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
                if n_key not in board.stacks and n_key not in board.collapsed_spaces:
                    liberties += 1

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
                if neighbor_marker and neighbor_marker.player == marker.player:
                    neighbor_count += 1

            val = min(neighbor_count / 3.0, 1.0)
            if marker.player == current_player:
                features[8, cy, cx] = val
            else:
                features[9, cy, cx] = val

        # === CHANNELS 10-15: V3 Enhanced Features ===

        # Channel 10/11: Contestation (cells with both friendly & enemy neighbors)
        # Identifies strategic contested zones on the board
        for cy in range(self.board_size):
            for cx in range(self.board_size):
                if not self._valid_mask[cy, cx]:
                    continue
                q = cx - self.radius
                r = cy - self.radius
                pos = Position(x=q, y=r)

                neighbors = BoardGeometry.get_adjacent_positions(
                    pos, board.type, board.size
                )
                friendly_neighbors = 0
                enemy_neighbors = 0
                for npos in neighbors:
                    if not BoardGeometry.is_within_bounds(
                        npos, board.type, board.size
                    ):
                        continue
                    n_key = npos.to_key()
                    # Check stacks
                    n_stack = board.stacks.get(n_key)
                    if n_stack:
                        if n_stack.controlling_player == current_player:
                            friendly_neighbors += 1
                        else:
                            enemy_neighbors += 1
                    # Check markers
                    n_marker = board.markers.get(n_key)
                    if n_marker:
                        if n_marker.player == current_player:
                            friendly_neighbors += 1
                        else:
                            enemy_neighbors += 1

                # Contestation score: high when both friendly and enemy present
                if friendly_neighbors > 0 and enemy_neighbors > 0:
                    features[10, cy, cx] = min(friendly_neighbors / 3.0, 1.0)
                    features[11, cy, cx] = min(enemy_neighbors / 3.0, 1.0)

        # Channel 12/13: Stack height gradient (sum of neighboring heights)
        for cy in range(self.board_size):
            for cx in range(self.board_size):
                if not self._valid_mask[cy, cx]:
                    continue
                q = cx - self.radius
                r = cy - self.radius
                pos = Position(x=q, y=r)

                neighbors = BoardGeometry.get_adjacent_positions(
                    pos, board.type, board.size
                )
                friendly_height_sum = 0.0
                enemy_height_sum = 0.0
                for npos in neighbors:
                    if not BoardGeometry.is_within_bounds(
                        npos, board.type, board.size
                    ):
                        continue
                    n_key = npos.to_key()
                    n_stack = board.stacks.get(n_key)
                    if n_stack:
                        height = n_stack.stack_height / 5.0
                        if n_stack.controlling_player == current_player:
                            friendly_height_sum += height
                        else:
                            enemy_height_sum += height

                # Normalize by max possible (6 neighbors * 1.0 height)
                features[12, cy, cx] = min(friendly_height_sum / 6.0, 1.0)
                features[13, cy, cx] = min(enemy_height_sum / 6.0, 1.0)

        # Channel 14/15: Ring placement validity
        # Indicates cells where rings can legally be placed
        # Note: pos_key_this was unused - removed
        for cy in range(self.board_size):
            for cx in range(self.board_size):
                if not self._valid_mask[cy, cx]:
                    continue
                q = cx - self.radius
                r = cy - self.radius
                pos = Position(x=q, y=r)
                pos_key = pos.to_key()

                # Cell is valid for ring placement if:
                # - No stack exists
                # - No collapsed space
                # - Within bounds (already checked by valid_mask)
                can_place = (
                    pos_key not in board.stacks
                    and pos_key not in board.collapsed_spaces
                )

                if can_place:
                    # Check neighbor accessibility for strategic value
                    neighbors = BoardGeometry.get_adjacent_positions(
                        pos, board.type, board.size
                    )
                    accessible = sum(
                        1
                        for npos in neighbors
                        if BoardGeometry.is_within_bounds(
                            npos, board.type, board.size
                        )
                        and npos.to_key() not in board.collapsed_spaces
                    )
                    # Channel 14: valid placement cell (1.0 if can place)
                    features[14, cy, cx] = 1.0
                    # Channel 15: accessibility score (normalized by max neighbors)
                    features[15, cy, cx] = accessible / 6.0

        # === GLOBAL FEATURES (20 dims for V3) ===
        globals_vec = np.zeros(self.NUM_GLOBAL_FEATURES, dtype=np.float32)

        # Phase one-hot (indices 0-4)
        phase_index_map = {
            GamePhase.RING_PLACEMENT: 0,
            GamePhase.MOVEMENT: 1,
            GamePhase.CAPTURE: 2,
            GamePhase.CHAIN_CAPTURE: 2,  # Sub-phase of CAPTURE, shares index
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
                (p for p in state.players if p.player_number != current_player),
                None,
            )
        else:
            threat_pid = select_threat_opponent(state, current_player)
            if threat_pid is not None:
                opp_player = next(
                    (p for p in state.players if p.player_number == threat_pid),
                    None,
                )
            if opp_player is None:
                opp_player = next(
                    (p for p in state.players if p.player_number != current_player),
                    None,
                )

        rings_per_player = float(infer_rings_per_player(state))
        if my_player:
            globals_vec[5] = my_player.rings_in_hand / rings_per_player
            globals_vec[7] = my_player.eliminated_rings / rings_per_player
        if opp_player:
            globals_vec[6] = opp_player.rings_in_hand / rings_per_player
            globals_vec[8] = opp_player.eliminated_rings / rings_per_player

        # Turn indicator (index 9)
        globals_vec[9] = 1.0

        # Extended features (indices 10-19) - V3 additions
        # Index 10: Total markers on board (normalized)
        total_markers = len(board.markers)
        globals_vec[10] = min(total_markers / 50.0, 1.0)

        # Index 11: Total stacks on board (normalized)
        total_stacks = len(board.stacks)
        globals_vec[11] = min(total_stacks / 30.0, 1.0)

        # Index 12: Territory control ratio
        my_territory = sum(
            1 for owner in board.collapsed_spaces.values()
            if owner == current_player
        )
        total_territory = len(board.collapsed_spaces)
        if total_territory > 0:
            globals_vec[12] = my_territory / total_territory

        # Index 13: Marker count ratio
        my_markers = sum(
            1 for m in board.markers.values()
            if m.player == current_player
        )
        if total_markers > 0:
            globals_vec[13] = my_markers / total_markers

        # Index 14: Stack control ratio
        my_stacks = sum(
            1 for s in board.stacks.values()
            if s.controlling_player == current_player
        )
        if total_stacks > 0:
            globals_vec[14] = my_stacks / total_stacks

        if self.feature_version >= 2:
            globals_vec[15] = (
                1.0 if state.current_phase == GamePhase.CHAIN_CAPTURE else 0.0
            )
            globals_vec[16] = (
                1.0
                if state.current_phase == GamePhase.FORCED_ELIMINATION
                else 0.0
            )

        # Indices 17-19: Reserved (zeros for now)

        return features, globals_vec

    def encode(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode state with frame stacking for inference.

        Since the model expects history-stacked features (64 channels = 16 × 4),
        but inference only has the current state, we replicate the current
        frame to fill all history slots. This is a reasonable approximation
        for single-frame inference.

        Returns:
            - stacked_features: shape (64, H, W) - 4 copies of current frame
            - global_features: shape (20,)
        """
        features, globals_vec = self.encode_state(state)
        # Stack current frame 4 times to match training format (history_length=3)
        stacked = np.concatenate([features] * 4, axis=0)
        return stacked, globals_vec

    def encode_with_history(
        self,
        state: GameState,
        history_frames: list[np.ndarray],
        history_length: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode state with history frame stacking.

        Returns:
            - stacked_features: shape (16 * (history_length + 1), H, W) = (64, 25, 25)
            - global_features: shape (20,)
        """
        features, globals_vec = self.encode_state(state)

        # Get history frames (most recent first)
        hist = history_frames[::-1][:history_length]

        # Pad with zeros if not enough history
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))

        # Stack: current + history
        stack = np.concatenate([features, *hist], axis=0)

        return stack, globals_vec

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Encode a move to a policy index."""
        return self.action_encoder.encode_move(move, board)

    def decode_move(self, index: int, state: GameState) -> Move | None:
        """Decode a policy index to a move."""
        return self.action_encoder.decode_move(index, state)

    def encode_legal_moves(
        self, moves: list[Move], board: BoardState
    ) -> list[int]:
        """Encode a list of legal moves."""
        return encode_hex_legal_moves(moves, self.action_encoder, board)

    def create_policy_target(
        self,
        move_indices: list[int],
        move_probs: list[float],
    ) -> np.ndarray:
        """Create a dense policy target vector."""
        policy = np.zeros(self.POLICY_SIZE, dtype=np.float32)
        for idx, prob in zip(move_indices, move_probs, strict=False):
            if 0 <= idx < self.POLICY_SIZE:
                policy[idx] = prob
        return policy

    def create_sparse_policy_target(
        self,
        move_indices: list[int],
        move_probs: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sparse policy target arrays."""
        valid_indices = []
        valid_probs = []
        for idx, prob in zip(move_indices, move_probs, strict=False):
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
    elif h == w == 9:
        return BoardType.HEX8
    elif h == w == 19:
        return BoardType.SQUARE19
    elif h == w == 25:
        return BoardType.HEXAGONAL
    else:
        raise ValueError(
            f"Cannot detect board type from spatial size {h}x{w}. "
            f"Expected 8x8, 9x9, 19x19, or 25x25."
        )


def get_encoder_for_board_type(
    board_type: BoardType,
    version: str = "v2",
    feature_version: int = 1,
) -> HexStateEncoder | HexStateEncoderV3 | None:
    """
    Get the appropriate encoder for a board type.

    Args:
        board_type: The board type
        version: Encoder version ("v2" or "v3"). V3 produces 16 channels
                 (64 with frame stacking) for HexNeuralNet_v3.
        feature_version: Feature encoding version for global feature layout.

    Returns:
        HexStateEncoder/HexStateEncoderV3 for HEXAGONAL/HEX8 boards, None for square boards
        (square boards use NeuralNetAI directly)
    """
    if board_type == BoardType.HEXAGONAL:
        if version.lower() == "v3":
            return HexStateEncoderV3(feature_version=feature_version)
        return HexStateEncoder(feature_version=feature_version)
    elif board_type == BoardType.HEX8:
        # Hex8: radius=4, board_size=9, policy_size=4500
        if version.lower() == "v3":
            return HexStateEncoderV3(
                board_size=HEX8_BOARD_SIZE,
                policy_size=POLICY_SIZE_HEX8,
                feature_version=feature_version,
            )
        return HexStateEncoder(
            board_size=HEX8_BOARD_SIZE,
            policy_size=POLICY_SIZE_HEX8,
            feature_version=feature_version,
        )
    elif board_type in (BoardType.SQUARE8, BoardType.SQUARE19):
        board_size = 8 if board_type == BoardType.SQUARE8 else 19
        return SquareStateEncoder(
            board_type=board_type,
            board_size=board_size,
            feature_version=feature_version,
        )
    return None


class SquareStateEncoder:
    """
    State encoder for square boards (SQUARE8 and SQUARE19).

    Converts square game states to feature tensors suitable for RingRiftCNN.
    The encoder outputs feature tensors of shape (C, H, W) where:
    - C = 14 channels
    - H = W = board_size (8 or 19)

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
        10: Cap presence - current player
        11: Cap presence - opponent
        12: Valid board position mask
        13: Reserved (zeros)

    Global Features (20 dims):
        Same layout as HexStateEncoder for model compatibility.
    """

    NUM_CHANNELS = 14
    NUM_GLOBAL_FEATURES = 20

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        board_size: int = 8,
        feature_version: int = 1,
    ):
        """
        Initialize the square state encoder.

        Args:
            board_type: Board type (SQUARE8 or SQUARE19)
            board_size: Size of the board (8 or 19)
            feature_version: Feature encoding version for global feature layout.
        """
        self.board_type = board_type
        self.board_size = board_size
        self.feature_version = int(feature_version)

    def encode_state(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a square game state to feature tensors.

        Args:
            state: The game state to encode

        Returns:
            Tuple of (board_features, global_features):
            - board_features: shape (14, board_size, board_size)
            - global_features: shape (20,)
        """
        board = state.board

        if board.type not in (BoardType.SQUARE8, BoardType.SQUARE19):
            raise ValueError(
                f"SquareStateEncoder requires SQUARE8 or SQUARE19 board type, "
                f"got {board.type}"
            )

        board_size = self.board_size
        features = np.zeros((self.NUM_CHANNELS, board_size, board_size), dtype=np.float32)
        current_player = state.current_player

        # Channel 0/1: Stacks (height normalized)
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            val = min(stack.stack_height / 5.0, 1.0)
            if stack.controlling_player == current_player:
                features[0, cx, cy] = val
            else:
                features[1, cx, cy] = val

        # Channel 2/3: Markers
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if marker.player == current_player:
                features[2, cx, cy] = 1.0
            else:
                features[3, cx, cy] = 1.0

        # Channel 4/5: Collapsed spaces (territory)
        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if owner == current_player:
                features[4, cx, cy] = 1.0
            else:
                features[5, cx, cy] = 1.0

        # Channel 6/7: Liberties
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(pos, board.type, board.size)
            liberties = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(npos, board.type, board.size):
                    continue
                n_key = npos.to_key()
                if n_key in board.stacks or n_key in board.collapsed_spaces:
                    continue
                liberties += 1

            max_libs = 8.0  # Square boards have up to 8 neighbors
            val = min(liberties / max_libs, 1.0)
            if stack.controlling_player == current_player:
                features[6, cx, cy] = val
            else:
                features[7, cx, cy] = val

        # Channel 8/9: Line potential (markers with friendly neighbors)
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(pos, board.type, board.size)
            friendly_neighbors = 0
            for npos in neighbors:
                n_key = npos.to_key()
                nm = board.markers.get(n_key)
                if nm and nm.player == marker.player:
                    friendly_neighbors += 1

            val = min(friendly_neighbors / 8.0, 1.0)
            if marker.player == current_player:
                features[8, cx, cy] = val
            else:
                features[9, cx, cy] = val

        # Channel 10/11: Cap presence
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if getattr(stack, 'has_cap', False):
                if stack.controlling_player == current_player:
                    features[10, cx, cy] = 1.0
                else:
                    features[11, cx, cy] = 1.0

        # Channel 12: Valid board position mask
        features[12, :, :] = 1.0

        # Channel 13: Reserved (zeros)

        # Compute global features
        global_features = self._compute_global_features(state)

        return features, global_features

    def _compute_global_features(self, state: GameState) -> np.ndarray:
        """Compute global (non-spatial) features for the game state."""
        globals_vec = np.zeros(self.NUM_GLOBAL_FEATURES, dtype=np.float32)

        # Game phase encoding (channels 0-4)
        phase = state.current_phase
        if phase == GamePhase.RING_PLACEMENT:
            globals_vec[0] = 1.0
        elif phase == GamePhase.MOVEMENT:
            globals_vec[1] = 1.0
        elif phase == GamePhase.CAPTURE:
            globals_vec[2] = 1.0
        elif phase == GamePhase.LINE_PROCESSING:
            globals_vec[3] = 1.0
        elif phase in (GamePhase.TERRITORY_PROCESSING, GamePhase.FORCED_ELIMINATION):
            globals_vec[4] = 1.0

        # Player statistics
        num_players = infer_num_players(state)
        rings_per_player = infer_rings_per_player(state)
        current_player = state.current_player
        threat_opponent = select_threat_opponent(state, num_players)

        # Get player objects
        my_player = next(
            (p for p in state.players if p.player_number == current_player), None
        )
        opp_player = next(
            (p for p in state.players if p.player_number == threat_opponent), None
        )
        if opp_player is None:
            opp_player = next(
                (p for p in state.players if p.player_number != current_player), None
            )

        # Rings in hand (channel 5-6)
        my_rings = my_player.rings_in_hand if my_player else rings_per_player
        opp_rings = opp_player.rings_in_hand if opp_player else rings_per_player
        globals_vec[5] = my_rings / max(rings_per_player, 1)
        globals_vec[6] = opp_rings / max(rings_per_player, 1)

        # Eliminated rings (channel 7-8)
        my_elim = my_player.eliminated_rings if my_player else 0
        opp_elim = opp_player.eliminated_rings if opp_player else 0
        globals_vec[7] = my_elim / max(rings_per_player, 1)
        globals_vec[8] = opp_elim / max(rings_per_player, 1)

        # Current turn indicator (channel 9)
        globals_vec[9] = 1.0

        # Player count indicators (channel 10-11)
        globals_vec[10] = 1.0 if num_players == 3 else 0.0
        globals_vec[11] = 1.0 if num_players == 4 else 0.0

        # Territory and piece counts (channels 12-17)
        board = state.board
        my_territory = sum(1 for p, o in board.collapsed_spaces.items() if o == current_player)
        opp_territory = sum(1 for p, o in board.collapsed_spaces.items() if o == threat_opponent)
        my_markers = sum(1 for p, m in board.markers.items() if m.player == current_player)
        opp_markers = sum(1 for p, m in board.markers.items() if m.player == threat_opponent)
        my_stacks = sum(1 for p, s in board.stacks.items() if s.controlling_player == current_player)
        opp_stacks = sum(1 for p, s in board.stacks.items() if s.controlling_player == threat_opponent)

        max_cells = self.board_size * self.board_size
        globals_vec[12] = my_territory / max_cells
        globals_vec[13] = opp_territory / max_cells
        globals_vec[14] = my_markers / max_cells
        globals_vec[15] = opp_markers / max_cells
        globals_vec[16] = my_stacks / max_cells
        globals_vec[17] = opp_stacks / max_cells

        # Feature version specific (channels 18-19)
        if self.feature_version >= 2:
            globals_vec[18] = 1.0 if getattr(state, 'chain_capture_in_progress', False) else 0.0
            globals_vec[19] = 1.0 if phase == GamePhase.FORCED_ELIMINATION else 0.0
        else:
            # Safely get move number with fallback (same logic as HexStateEncoder)
            move_number = getattr(state, 'move_number', 0)
            if move_number == 0:
                move_number = getattr(state, 'turn_number', 0)
            if move_number == 0 and hasattr(state, 'move_history'):
                move_number = len(state.move_history) if state.move_history else 0
            globals_vec[18] = min(move_number / 200.0, 1.0)
            globals_vec[19] = 0.0

        return globals_vec

    def encode(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode state with frame stacking for inference.

        Since the model expects history-stacked features (56 channels = 14 × 4),
        but inference only has the current state, we replicate the current
        frame to fill all history slots. This is a reasonable approximation
        for single-frame inference.

        Returns:
            - stacked_features: shape (56, H, W) - 4 copies of current frame
            - global_features: shape (20,)
        """
        features, globals_vec = self.encode_state(state)
        # Stack current frame 4 times to match training format (history_length=3)
        stacked = np.concatenate([features] * 4, axis=0)
        return stacked, globals_vec
