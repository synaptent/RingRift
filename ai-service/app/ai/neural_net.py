"""
Neural Network AI implementation for RingRift
Uses a simple feedforward neural network for move evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Union
import random
from datetime import datetime

from .base import BaseAI
from ..models import (
    GameState,
    Move,
    BoardType,
    Position,
    BoardState,
)
from ..rules.geometry import BoardGeometry


INVALID_MOVE_INDEX = -1
MAX_N = 19  # Canonical maximum side length for policy encoding (19x19 grid)


def _infer_board_size(board: Union[BoardState, GameState]) -> int:
    """
    Infer the canonical 2D board_size for CNN feature tensors.

    For SQUARE8: 8
    For SQUARE19: 19
    For HEXAGONAL: 2 * radius + 1, where radius = board.size - 1

    The returned value is the height/width of the (10, board_size, board_size)
    feature planes used by the CNN. Raises if the logical size exceeds MAX_N
    for square boards.
    """
    # Allow a GameState to be passed directly
    if isinstance(board, GameState):
        board = board.board

    board_type = board.type

    if board_type == BoardType.SQUARE8:
        return 8
    if board_type == BoardType.SQUARE19:
        return 19
    if board_type == BoardType.HEXAGONAL:
        radius = board.size - 1
        return 2 * radius + 1

    # Defensive fallback: use board.size but guard against unsupported sizes
    size = getattr(board, "size", 8)
    if size > MAX_N:
        raise ValueError(
            f"Unsupported board size {size}; MAX_N={MAX_N} is the current "
            "canonical maximum."
        )
    return int(size)


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


def _to_canonical_xy(board: BoardState, pos: Position) -> tuple[int, int]:
    """
    Map a logical Position on this board into canonical (cx, cy) in
    [0, board_size) × [0, board_size), where board_size depends on
    board.type and board.size.

    For SQUARE8/SQUARE19: return (pos.x, pos.y) directly.

    For HEXAGONAL:
      - Interpret pos.(x,y,z) as cube/axial coords where x,y lie in
        [-radius, radius].
      - Let radius = board.size - 1.
      - Map x → cx = x + radius, y → cy = y + radius.
      - Return (cx, cy).
    """
    # We still allow callers to pass a GameState into _infer_board_size, but
    # here we require a BoardState to access geometry metadata consistently.
    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return pos.x, pos.y

    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
        cx = pos.x + radius
        cy = pos.y + radius
        return cx, cy

    # Fallback: treat as generic square coordinates.
    return pos.x, pos.y


def _from_canonical_xy(
    board: BoardState,
    cx: int,
    cy: int,
) -> Optional[Position]:
    """
    Inverse of _to_canonical_xy.

    Returns a Position instance whose coordinates lie on this board, or None
    if (cx, cy) is outside [0, board_size) × [0, board_size).

    For HEXAGONAL:
      - radius = board.size - 1
      - x = cx - radius, y = cy - radius, z = -x - y
    """
    board_size = _infer_board_size(board)
    if not (0 <= cx < board_size and 0 <= cy < board_size):
        return None

    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return Position(x=cx, y=cy)

    if board.type == BoardType.HEXAGONAL:
        radius = board.size - 1
        x = cx - radius
        y = cy - radius
        z = -x - y
        return Position(x=x, y=y, z=z)

    # Fallback generic square position.
    return Position(x=cx, y=cy)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class RingRiftCNN(nn.Module):
    def __init__(
        self, board_size=8, in_channels=10, global_features=10,
        num_res_blocks=10, num_filters=128, history_length=3
    ):
        super(RingRiftCNN, self).__init__()
        self.board_size = board_size
        
        # Input channels = base_channels * (history_length + 1)
        # Base channels = 10
        # Default history length = 3 (Current + 3 Previous)
        #
        # State Encoding (10 channels):
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        self.total_in_channels = in_channels * (history_length + 1)

        # Initial convolution
        self.conv1 = nn.Conv2d(
            self.total_in_channels, num_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Adaptive Pooling to handle variable board sizes (e.g. 8x8, 19x19)
        # We pool to a fixed 4x4 grid before flattening, ensuring the FC layer
        # input size is constant.
        # This allows the same model architecture to process different board
        # sizes, though retraining/finetuning is recommended for drastic size
        # changes.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        # Input size is now num_filters * 4 * 4 (fixed)
        conv_out_size = num_filters * 4 * 4
        self.fc1 = nn.Linear(conv_out_size + global_features, 256)
        self.dropout = nn.Dropout(0.3)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        
        # Policy head
        # We use a large fixed size to accommodate up to 19x19 boards.
        # For smaller boards, we mask the invalid logits during
        # inference/training.
        # Max size 19x19: ~55,000
        self.policy_size = 55000
        self.policy_head = nn.Linear(256, self.policy_size)

    def forward(self, x, globals):
        x = self.relu(self.bn1(self.conv1(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate global features
        x = torch.cat((x, globals), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        value = self.tanh(self.value_head(x))  # Output between -1 and 1
        policy = self.policy_head(x)  # Logits for CrossEntropyLoss

        return value, policy

    def forward_single(
        self, feature: np.ndarray, globals_vec: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Convenience method for single-sample inference.
        Returns (value, policy_logits).
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).float().to(
                next(self.parameters()).device
            )
            g = torch.from_numpy(globals_vec[None, ...]).float().to(
                next(self.parameters()).device
            )
            v, p = self.forward(x, g)
        return float(v.item()), p.cpu().numpy()[0]


class NeuralNetAI(BaseAI):
    """AI that uses a CNN to evaluate positions"""
    
    def __init__(self, player_number: int, config: Any):
        super().__init__(player_number, config)
        # Initialize model
        # Channels:
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        # Hint for tools that need the current spatial dimension (e.g. training
        # data augmentation). The encoder derives the true size from the
        # GameState/BoardState via _infer_board_size and keeps this field
        # updated at runtime.
        self.board_size = 8
        self.history_length = 3
        # Dict[str, List[np.ndarray]] - Keyed by game_id
        self.game_history = {}
        
        # Device detection
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"NeuralNetAI using device: {self.device}")
        
        # Initialize with default 8x8, but architecture is now adaptive
        self.model = RingRiftCNN(
            board_size=self.board_size, in_channels=10, global_features=10,
            num_res_blocks=10, num_filters=128,
            history_length=self.history_length
        )
        self.model.to(self.device)

        # Load weights if available
        import os
        # Use absolute path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models/ringrift_v1.pth")
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(
                        model_path, map_location=self.device, weights_only=True
                    )
                )
                self.model.eval()
            except RuntimeError as e:
                # Architecture mismatch
                print(
                    f"Could not load model (architecture mismatch): {e}. "
                    "Starting with fresh weights."
                )
                self.model.eval()
            except Exception as e:
                print(
                    f"Could not load model (error): {e}. "
                    "Starting with fresh weights."
                )
                self.model.eval()
        else:
            # No model found, start fresh (silent)
            self.model.eval()

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using neural network evaluation
        """
        self.simulate_thinking(min_ms=300, max_ms=1000)
        
        # Update history for the current game state
        current_features, _ = self._extract_features(game_state)
        game_id = game_state.id
        
        if game_id not in self.game_history:
            self.game_history[game_id] = []
            
        # Append current state to history
        # We only append if it's a new state (simple check: diff from last)
        # Or just append always? select_move is called once per turn.
        # But we might be called multiple times for same state if retrying?
        # Let's assume we append.
        self.game_history[game_id].append(current_features)
        
        # Keep only needed history (history_length + 1 for current)
        # Actually we need history_length previous states.
        # So we keep history_length + 1 (current) + maybe more?
        # We just need the last few.
        max_hist = self.history_length + 1
        if len(self.game_history[game_id]) > max_hist:
            self.game_history[game_id] = self.game_history[game_id][-max_hist:]

        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None
            
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
        else:
            # Batch evaluation
            next_states = []
            moves_list = []
            
            for move in valid_moves:
                next_states.append(
                    self.rules_engine.apply_move(game_state, move)
                )
                moves_list.append(move)
            
            # Construct stacked inputs for all next states
            # For each next_state, the history is:
            # [current_state, prev1, prev2, ...] (from self.game_history)
            # So the stack for next_state is:
            # next_state + current_state + prev1 + prev2 ...
            
            # Get the base history (current + previous)
            # self.game_history[game_id] contains [...prev2, prev1, current]
            # Reverse to get [current, prev1, prev2...]
            base_history = self.game_history[game_id][::-1]
            
            # Pad if necessary
            while len(base_history) < self.history_length:
                base_history.append(np.zeros_like(current_features))
            
            # Trim to history_length
            base_history = base_history[:self.history_length]
            
            # Now construct batch
            batch_stacks = []
            batch_globals = []
            
            for ns in next_states:
                ns_features, ns_globals = self._extract_features(ns)
                
                # Stack: [ns_features, base_history[0], base_history[1]...]
                stack_list = [ns_features] + base_history
                # Concatenate along channel dim (0)
                stack = np.concatenate(stack_list, axis=0)
                
                batch_stacks.append(stack)
                batch_globals.append(ns_globals)
            
            # Convert to tensor
            tensor_input = torch.FloatTensor(
                np.array(batch_stacks)
            ).to(self.device)
            globals_input = torch.FloatTensor(
                np.array(batch_globals)
            ).to(self.device)
            
            # Evaluate batch
            values, _ = self.evaluate_batch(
                next_states,
                tensor_input=tensor_input,
                globals_input=globals_input
            )
            
            # Find best move
            best_idx = np.argmax(values)
            selected = moves_list[best_idx]
            
        return selected

    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate position using neural network.
        """
        # Note: This method doesn't support history injection easily unless
        # we pass it. If called from outside select_move, it might lack
        # history context. We'll assume it uses the stored history for the
        # game_id if available.
        values, _ = self.evaluate_batch([game_state])
        return values[0] if values else 0.0

    def evaluate_batch(
        self,
        game_states: list[GameState],
        tensor_input: Optional[torch.Tensor] = None,
        globals_input: Optional[torch.Tensor] = None,
    ) -> tuple[list[float], np.ndarray]:
        """
        Evaluate a batch of game states.

        All states in a batch must share the same board.type and board.size so
        that the stacked feature tensors have a consistent spatial shape. This
        invariant is enforced at runtime and a ValueError is raised if it is
        violated.
        """
        if not game_states and tensor_input is None:
            empty_policy = np.zeros(
                (0, self.model.policy_size), dtype=np.float32
            )
            return [], empty_policy

        # Enforce homogeneous board geometry within a batch.
        if game_states:
            first_board = game_states[0].board
            first_type = first_board.type
            first_size = first_board.size
            for state in game_states[1:]:
                if (
                    state.board.type != first_type or
                    state.board.size != first_size
                ):
                    raise ValueError(
                        "NeuralNetAI.evaluate_batch requires all game_states "
                        "in a batch to share the same board.type and "
                        f"board.size; got {first_type}/{first_size} and "
                        f"{state.board.type}/{state.board.size}."
                    )
            # Cache the canonical spatial dimension for downstream tools.
            self.board_size = _infer_board_size(first_board)

        if tensor_input is None:
            # Fallback: construct inputs from states, using stored history.
            batch_stacks: list[np.ndarray] = []
            batch_globals: list[np.ndarray] = []

            for state in game_states:
                features, globals_vec = self._extract_features(state)

                # Try to get history
                game_id = state.id
                history: list[np.ndarray] = []
                if game_id in self.game_history:
                    # History list is [oldest, ..., newest]; we want
                    # newest-first for stacking.
                    hist_list = self.game_history[game_id][::-1]
                    history = hist_list[: self.history_length]

                # Pad history
                while len(history) < self.history_length:
                    history.append(np.zeros_like(features))

                # Stack: [current, hist1, hist2...]
                stack_list = [features] + history
                stack = np.concatenate(stack_list, axis=0)

                batch_stacks.append(stack)
                batch_globals.append(globals_vec)

            tensor_input = torch.FloatTensor(
                np.array(batch_stacks)
            ).to(self.device)
            globals_input = torch.FloatTensor(
                np.array(batch_globals)
            ).to(self.device)

        assert globals_input is not None

        with torch.no_grad():
            values, policy_logits = self.model(tensor_input, globals_input)

            # Apply softmax to logits to get probabilities for MCTS / Descent.
            policy_probs = torch.softmax(policy_logits, dim=1)

        return (
            values.cpu().numpy().flatten().tolist(),
            policy_probs.cpu().numpy(),
        )

    def encode_state_for_model(
        self,
        game_state: GameState,
        history_frames: list[np.ndarray],
        history_length: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (stacked_features[C,H,W], globals[10]) compatible with
        RingRiftCNN.
        history_frames: most recent feature frames for this game_id,
        newest last.
        """
        features, globals_vec = self._extract_features(game_state)
        # newest-first
        hist = history_frames[::-1][:history_length]
        while len(hist) < history_length:
            hist.append(np.zeros_like(features))
        stack = np.concatenate([features] + hist, axis=0)
        return stack, globals_vec

    def encode_move(
        self,
        move: Move,
        board_context: Union[BoardState, GameState, int],
    ) -> int:
        """
        Encode a move into a policy index.

        The encoding uses a fixed MAX_N × MAX_N = 19 × 19 canonical grid so
        that a single policy head can serve 8×8, 19×19, and hexagonal boards.
        Moves that require coordinates outside this canonical grid return
        INVALID_MOVE_INDEX and are simply omitted from the policy.

        For backward compatibility, board_context may be an integer board_size
        (e.g. 8 or 19). In that case we treat coordinates as already expressed
        in the canonical 2D frame for a square board and bypass BoardState-
        based mapping.
        """
        board: Optional[BoardState] = None

        # Normalise context to a BoardState when possible.
        if isinstance(board_context, GameState):
            board = board_context.board
        elif isinstance(board_context, BoardState):
            board = board_context
        elif isinstance(board_context, int):
            # Legacy callers (tests, older tooling) pass a raw board_size.
            # We do not need the size here, because we only ever check against
            # the canonical MAX_N=19 grid.
            board = None
        else:
            raise TypeError(
                f"Unsupported board_context type for encode_move: "
                f"{type(board_context)!r}"
            )

        # Pre-compute layout constants from MAX_N to avoid hard-coded offsets.
        placement_span = 3 * MAX_N * MAX_N            # 0..1082
        movement_base = placement_span                # 1083
        movement_span = MAX_N * MAX_N * (8 * (MAX_N - 1))
        line_base = movement_base + movement_span     # 53067
        line_span = MAX_N * MAX_N * 4
        territory_base = line_base + line_span        # 54511
        skip_index = territory_base + MAX_N * MAX_N   # 54872

        # Placement: 0..1082 (3 * 19 * 19)
        if move.type == "place_ring":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                # Legacy integer board_size path (square boards).
                cx, cy = move.to.x, move.to.y

            # Guard against boards larger than MAX_N×MAX_N.
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX

            # Index = (y * MAX_N + x) * 3 + (count - 1)
            pos_idx = cy * MAX_N + cx
            count_idx = (move.placement_count or 1) - 1
            return pos_idx * 3 + count_idx

        # Movement: 1083..53066
        if move.type in [
            "move_stack",
            "move_ring",
            "overtaking_capture",
            "chain_capture",
            "continue_capture_segment",
        ]:
            # Base = 1083 (3 * 19 * 19)
            # Index = Base + (from_y * MAX_N + from_x) * (8 * (MAX_N-1)) +
            #         (dir_idx * (MAX_N-1)) + (dist - 1)
            if not move.from_pos:
                return INVALID_MOVE_INDEX

            if board is not None:
                cfx, cfy = _to_canonical_xy(board, move.from_pos)
                ctx, cty = _to_canonical_xy(board, move.to)
            else:
                # Legacy integer board_size path (square boards).
                cfx, cfy = move.from_pos.x, move.from_pos.y
                ctx, cty = move.to.x, move.to.y

            # If either endpoint lies outside the canonical 19×19 grid, this
            # move cannot be represented in the fixed policy head.
            if not (
                0 <= cfx < MAX_N and 0 <= cfy < MAX_N and
                0 <= ctx < MAX_N and 0 <= cty < MAX_N
            ):
                return INVALID_MOVE_INDEX

            from_idx = cfy * MAX_N + cfx

            dx = ctx - cfx
            dy = cty - cfy

            # For square boards we use Chebyshev distance. For hex boards, the
            # canonical 2D embedding is a translation of axial coordinates, so
            # dx/dy are preserved and we can continue to use Chebyshev here as
            # long as encode/decode remain symmetric.
            dist = max(abs(dx), abs(dy))
            if dist == 0:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist if dist > 0 else 0
            dir_y = dy // dist if dist > 0 else 0

            dirs = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ]
            try:
                dir_idx = dirs.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 8-direction scheme.
                return INVALID_MOVE_INDEX

            max_dist = MAX_N - 1
            return (
                movement_base +
                from_idx * (8 * max_dist) +
                dir_idx * max_dist +
                (dist - 1)
            )

        # Line: 53067..54510
        if move.type == "line_formation":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                cx, cy = move.to.x, move.to.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            # We currently ignore direction and always use dir_idx = 0, but
            # keep the 4-way slot layout for backward compatibility.
            return line_base + pos_idx * 4

        # Territory: 54511..54871
        if move.type == "territory_claim":
            if board is not None:
                cx, cy = _to_canonical_xy(board, move.to)
            else:
                cx, cy = move.to.x, move.to.y
            if not (0 <= cx < MAX_N and 0 <= cy < MAX_N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * MAX_N + cx
            return territory_base + pos_idx

        # Skip placement: single terminal index
        if move.type == "skip_placement":
            return skip_index

        return INVALID_MOVE_INDEX

    def decode_move(self, index: int, game_state: GameState) -> Optional[Move]:
        """
        Decode a policy index into a Move.

        The inverse of encode_move, using the same MAX_N × MAX_N canonical
        grid. If the decoded coordinates fall outside the legal geometry of
        game_state.board, this returns None.
        """
        board = game_state.board

        # Pre-compute layout constants from MAX_N to align with encode_move.
        placement_span = 3 * MAX_N * MAX_N            # 0..1082
        movement_base = placement_span                # 1083
        movement_span = MAX_N * MAX_N * (8 * (MAX_N - 1))
        line_base = movement_base + movement_span     # 53067
        line_span = MAX_N * MAX_N * 4
        territory_base = line_base + line_span        # 54511
        skip_index = territory_base + MAX_N * MAX_N   # 54872

        if index < 0 or index >= self.model.policy_size:
            return None

        # Placement
        if index < placement_span:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(
                pos, board.type, board.size
            ):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "place_ring",
                "player": game_state.current_player,
                "to": to_payload,
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Movement
        if index < line_base:
            max_dist = MAX_N - 1
            offset = index - movement_base

            dist_idx = offset % max_dist
            dist = dist_idx + 1
            offset //= max_dist

            dir_idx = offset % 8
            offset //= 8

            from_idx = offset
            cfy = from_idx // MAX_N
            cfx = from_idx % MAX_N

            from_pos = _from_canonical_xy(board, cfx, cfy)
            if from_pos is None or not BoardGeometry.is_within_bounds(
                from_pos, board.type, board.size
            ):
                return None

            dirs = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ]
            dx, dy = dirs[dir_idx]

            ctx = cfx + dx * dist
            cty = cfy + dy * dist
            to_pos = _from_canonical_xy(board, ctx, cty)
            if to_pos is None or not BoardGeometry.is_within_bounds(
                to_pos, board.type, board.size
            ):
                return None

            from_payload: dict[str, int] = {"x": from_pos.x, "y": from_pos.y}
            if from_pos.z is not None:
                from_payload["z"] = from_pos.z

            to_payload: dict[str, int] = {"x": to_pos.x, "y": to_pos.y}
            if to_pos.z is not None:
                to_payload["z"] = to_pos.z

            move_data = {
                "id": "decoded",
                "type": "move_stack",
                "player": game_state.current_player,
                "from": from_payload,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Line formation
        if line_base <= index < territory_base:
            offset = index - line_base
            pos_idx = offset // 4  # Ignore dir_idx for now
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(
                pos, board.type, board.size
            ):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "line_formation",
                "player": game_state.current_player,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Territory claim
        if index < skip_index:
            pos_idx = index - territory_base
            cy = pos_idx // MAX_N
            cx = pos_idx % MAX_N

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(
                pos, board.type, board.size
            ):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": "territory_claim",
                "player": game_state.current_player,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # Skip placement
        if index == skip_index:
            move_data = {
                "id": "decoded",
                "type": "skip_placement",
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        return None

    def _extract_features(
        self,
        game_state: GameState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert game state to feature tensor for CNN and global features.

        Returns:
            (board_features, global_features)

        The board_features tensor has shape
        (10, board_size, board_size), where board_size is derived from the
        logical board via _infer_board_size so that this encoder works for
        8×8, 19×19, and hexagonal boards.
        """
        board = game_state.board
        # Derive spatial dimension from logical board geometry and keep a hint
        # for components (e.g. training augmentation) that still need to know
        # the current spatial dimension.
        board_size = _infer_board_size(board)
        self.board_size = board_size

        # Board features: 10 channels
        # 0: My stacks (height normalized)
        # 1: Opponent stacks (height normalized)
        # 2: My markers
        # 3: Opponent markers
        # 4: My collapsed spaces
        # 5: Opponent collapsed spaces
        # 6: My liberties
        # 7: Opponent liberties
        # 8: My line potential
        # 9: Opponent line potential
        features = np.zeros(
            (10, board_size, board_size), dtype=np.float32
        )

        is_hex = board.type == BoardType.HEXAGONAL

        # --- Stacks: channels 0/1 ---
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                # Robust to any stray off-board keys.
                continue

            val = min(stack.stack_height / 5.0, 1.0)  # Normalize height
            if stack.controlling_player == game_state.current_player:
                features[0, cx, cy] = val
            else:
                features[1, cx, cy] = val

        # --- Markers: channels 2/3 ---
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if marker.player == game_state.current_player:
                features[2, cx, cy] = 1.0
            else:
                features[3, cx, cy] = 1.0

        # --- Collapsed spaces: channels 4/5 ---
        for pos_key, owner in board.collapsed_spaces.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            if owner == game_state.current_player:
                features[4, cx, cy] = 1.0
            else:
                features[5, cx, cy] = 1.0

        # --- Liberties: channels 6/7 ---
        # Simple approximation based on adjacency; uses BoardGeometry so that
        # hex and square boards share the same logic.
        for pos_key, stack in board.stacks.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(
                pos,
                board.type,
                board.size,
            )
            liberties = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = npos.to_key()
                if (
                    n_key in board.stacks or
                    n_key in board.collapsed_spaces
                ):
                    continue
                liberties += 1

            max_libs = 6.0 if is_hex else 8.0
            val = min(liberties / max_libs, 1.0)
            if stack.controlling_player == game_state.current_player:
                features[6, cx, cy] = val
            else:
                features[7, cx, cy] = val

        # --- Line potential: channels 8/9 ---
        # Simplified: markers with neighbours of same colour.
        for pos_key, marker in board.markers.items():
            try:
                pos = _pos_from_key(pos_key)
            except ValueError:
                continue

            cx, cy = _to_canonical_xy(board, pos)
            if not (0 <= cx < board_size and 0 <= cy < board_size):
                continue

            neighbors = BoardGeometry.get_adjacent_positions(
                pos,
                board.type,
                board.size,
            )
            neighbor_count = 0
            for npos in neighbors:
                if not BoardGeometry.is_within_bounds(
                    npos,
                    board.type,
                    board.size,
                ):
                    continue
                n_key = npos.to_key()
                neighbor_marker = board.markers.get(n_key)
                if (
                    neighbor_marker is not None and
                    neighbor_marker.player == marker.player
                ):
                    neighbor_count += 1

            max_neighbors = 6.0 if is_hex else 8.0
            val = min(neighbor_count / (max_neighbors / 2.0), 1.0)
            if marker.player == game_state.current_player:
                features[8, cx, cy] = val
            else:
                features[9, cx, cy] = val

        # --- Global features: 10 dims ---
        # Phase (5), Rings in hand (2), Eliminated rings (2), Turn (1)
        globals = np.zeros(10, dtype=np.float32)

        # Phase one-hot
        phases = [
            "ring_placement",
            "movement",
            "capture",
            "line_processing",
            "territory_processing",
        ]
        try:
            phase_idx = phases.index(game_state.current_phase.value)
            globals[phase_idx] = 1.0
        except ValueError:
            pass

        # Rings info
        my_player = next(
            (
                p
                for p in game_state.players
                if p.player_number == game_state.current_player
            ),
            None,
        )
        opp_player = next(
            (
                p
                for p in game_state.players
                if p.player_number != game_state.current_player
            ),
            None,
        )

        if my_player:
            globals[5] = my_player.rings_in_hand / 20.0
            globals[7] = my_player.eliminated_rings / 20.0

        if opp_player:
            globals[6] = opp_player.rings_in_hand / 20.0
            globals[8] = opp_player.eliminated_rings / 20.0

        # Is it my turn? (always yes for current_player perspective)
        globals[9] = 1.0

        return features, globals

    # _evaluate_move_with_net is deprecated
