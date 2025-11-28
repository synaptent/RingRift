"""
Neural Network AI implementation for RingRift
Uses a simple feedforward neural network for move evaluation
"""

import logging

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Union
from datetime import datetime

from .base import BaseAI
from ..models import (
    GameState,
    Move,
    BoardType,
    Position,
    BoardState,
    MoveType,
)
from ..rules.geometry import BoardGeometry


INVALID_MOVE_INDEX = -1
MAX_N = 19  # Canonical maximum side length for policy encoding (19x19 grid)

# Hex-specific canonical geometry and policy layout constants.
#
# The canonical competitive hex board has radius N = 10, which yields
# 3N^2 + 3N + 1 = 331 cells. We embed this hex into a fixed 21×21
# bounding box (2N + 1 on each axis) using _to_canonical_xy /
# _from_canonical_xy, and define a dedicated hex action space of size
# P_HEX = 54_244 as documented in AI_ARCHITECTURE.md.
HEX_BOARD_SIZE = 21
HEX_MAX_DIST = HEX_BOARD_SIZE - 1  # 20 distance buckets (1..20)

# Canonical axial directions in the 2D embedding.
HEX_DIRS = [
    (1, 0),   # +q
    (0, 1),   # +r
    (-1, 1),  # -q + r
    (-1, 0),  # -q
    (0, -1),  # -r
    (1, -1),  # +q - r
]
NUM_HEX_DIRS = len(HEX_DIRS)

# Layout spans for the hex policy head (see AI_ARCHITECTURE.md):
#
# Placements: 21 × 21 × 3 = 1_323
# Movement/capture: 21 × 21 × 6 × 20 = 52_920
# Special: 1 (skip_placement)
# Total: P_HEX = 54_244
HEX_PLACEMENT_SPAN = HEX_BOARD_SIZE * HEX_BOARD_SIZE * 3
HEX_MOVEMENT_BASE = HEX_PLACEMENT_SPAN
HEX_MOVEMENT_SPAN = (
    HEX_BOARD_SIZE
    * HEX_BOARD_SIZE
    * NUM_HEX_DIRS
    * HEX_MAX_DIST
)
HEX_SPECIAL_BASE = HEX_MOVEMENT_BASE + HEX_MOVEMENT_SPAN
P_HEX = HEX_SPECIAL_BASE + 1


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


logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class RingRiftCNN(nn.Module):
    """
    CNN architecture for RingRift board evaluation.

    This model uses a ResNet-style backbone with adaptive pooling to handle
    variable board sizes (8x8, 19x19, 21x21 hex).

    Architecture Version:
        v1.0.0 - Initial architecture with 10 residual blocks, 128 filters,
                 adaptive pooling to 4x4, policy head size 55000.
    """

    # Architecture version for checkpoint compatibility checking
    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 10,
        global_features: int = 10,
        num_res_blocks: int = 10,
        num_filters: int = 128,
        history_length: int = 3,
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
        import os

        disable_mps = bool(
            os.environ.get("RINGRIFT_DISABLE_MPS")
            or os.environ.get("PYTORCH_MPS_DISABLE")
        )
        force_cpu = bool(os.environ.get("RINGRIFT_FORCE_CPU"))

        if (
            torch.backends.mps.is_available()
            and not disable_mps
            and not force_cpu
        ):
            self.device = torch.device("mps")
        elif torch.cuda.is_available() and not force_cpu:
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

        # Load weights if available. When config.nn_model_id is provided, use
        # it as the logical model identifier (e.g. "ringrift_v1" or
        # "v1-nn-heuristic-5") and resolve to ``<base_dir>/models/<id>.pth``.
        # Otherwise we fall back to the historical default "ringrift_v1.pth".
        import os
        # Use absolute path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_id = getattr(self.config, "nn_model_id", None)
        if not model_id:
            model_id = "ringrift_v1"

        model_filename = f"{model_id}.pth"
        model_path = os.path.join(base_dir, "models", model_filename)
        
        import os as os_mod
        if os_mod.path.exists(model_path):
            self._load_model_checkpoint(model_path)
        else:
            # No model found, start fresh (silent)
            logger.info(f"No model found at {model_path}, using fresh weights")
            self.model.eval()

    def _load_model_checkpoint(self, model_path: str) -> None:
        """
        Load model checkpoint with version validation.

        Uses the model versioning system when available, falls back to
        direct state_dict loading for legacy checkpoints with explicit
        error handling.
        """
        try:
            # Try to use versioned loading first
            from ..training.model_versioning import (
                ModelVersionManager,
                VersionMismatchError,
                ChecksumMismatchError,
                LegacyCheckpointError,
            )

            manager = ModelVersionManager(default_device=self.device)

            try:
                # Try strict loading first
                state_dict, metadata = manager.load_checkpoint(
                    model_path,
                    strict=True,
                    expected_version=RingRiftCNN.ARCHITECTURE_VERSION,
                    expected_class="RingRiftCNN",
                    verify_checksum=True,
                    device=self.device,
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info(
                    f"Loaded versioned model from {model_path} "
                    f"(version: {metadata.architecture_version})"
                )
                return

            except LegacyCheckpointError:
                # Legacy checkpoint - load with warning
                logger.warning(
                    f"Loading legacy checkpoint without versioning: "
                    f"{model_path}. Consider migrating to versioned format."
                )
                state_dict, _ = manager.load_checkpoint(
                    model_path,
                    strict=False,
                    verify_checksum=False,
                    device=self.device,
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                return

            except VersionMismatchError as e:
                # Version mismatch - FAIL EXPLICITLY instead of silent fallback
                logger.error(
                    f"ARCHITECTURE VERSION MISMATCH - Cannot load!\n"
                    f"  Checkpoint: {e.checkpoint_version}\n"
                    f"  Expected: {e.current_version}\n"
                    f"  Path: {model_path}\n"
                    f"  This is a P0 error. The checkpoint is incompatible "
                    f"with the current model architecture."
                )
                raise  # Re-raise to prevent silent fallback

            except ChecksumMismatchError as e:
                # Integrity failure - FAIL EXPLICITLY
                logger.error(
                    f"CHECKPOINT INTEGRITY FAILURE - File may be corrupted!\n"
                    f"  Path: {model_path}\n"
                    f"  Expected checksum: {e.expected[:16]}...\n"
                    f"  Actual checksum: {e.actual[:16]}..."
                )
                raise  # Re-raise to prevent silent fallback

        except ImportError:
            # model_versioning not available, fall back to direct loading
            logger.warning(
                "model_versioning module not available, "
                "using legacy loading"
            )
            self._load_legacy_checkpoint(model_path)

    def _load_legacy_checkpoint(self, model_path: str) -> None:
        """
        Legacy checkpoint loading with explicit error handling.

        This is used when the versioning module is not available or
        for backwards compatibility with existing code paths.
        """
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume it's a direct state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loaded legacy model from {model_path}")

        except RuntimeError as e:
            # Architecture mismatch - FAIL EXPLICITLY
            error_msg = str(e)
            if "size mismatch" in error_msg or "Missing key" in error_msg:
                logger.error(
                    f"ARCHITECTURE MISMATCH - Cannot load checkpoint!\n"
                    f"  Path: {model_path}\n"
                    f"  Error: {e}\n"
                    f"  This indicates the checkpoint was saved with a "
                    f"different model architecture. Silent fallback to "
                    f"fresh weights is DISABLED to prevent training bugs."
                )
                raise RuntimeError(
                    f"Architecture mismatch loading {model_path}: {e}. "
                    f"Silent fallback is disabled. Either use a compatible "
                    f"checkpoint or explicitly start with fresh weights."
                ) from e
            else:
                raise

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using neural network evaluation.
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

        # Optional exploration/randomisation based on configured randomness
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
        else:
            # Batch evaluation
            next_states: list[GameState] = []
            moves_list: list[Move] = []

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
            base_history = base_history[: self.history_length]

            # Now construct batch
            batch_stacks: list[np.ndarray] = []
            batch_globals: list[np.ndarray] = []

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
                globals_input=globals_input,
            )

            # Find best move
            best_idx = int(np.argmax(values))
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


class ActionEncoderHex:
    """Hex-only action encoder for the canonical N=10 board.

    The concrete layout matches the design in AI_ARCHITECTURE.md:

      * Spatial frame: 21×21 canonical hex bounding box.
      * Placements (0 .. HEX_PLACEMENT_SPAN-1):
          index = (cy * 21 + cx) * 3 + (count - 1)
        where count ∈ {1,2,3} is the number of rings placed.

      * Movement / capture (HEX_MOVEMENT_BASE .. HEX_SPECIAL_BASE-1):
          index = HEX_MOVEMENT_BASE
                  + from_idx * (6 * 20)
                  + dir_idx * 20
                  + (dist - 1)
        where from_idx = from_cy * 21 + from_cx, dir_idx ∈ [0,6),
        dist ∈ [1,20]. This shared layout is used for MOVE_STACK,
        OVERTAKING_CAPTURE, and CONTINUE_CAPTURE_SEGMENT.

      * Special (HEX_SPECIAL_BASE):
          SKIP_PLACEMENT sentinel.

    Any decoded index that maps to a canonical cell outside the true hex
    (331-cell) region is treated as invalid and returns None.
    """

    def __init__(
        self,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = 0,
    ) -> None:
        # Spatial dimension of the hex bounding box (2N+1 for side N).
        self.board_size = board_size
        # Hex-specific action space dimension.
        self.policy_size = policy_size or P_HEX

    def _encode_canonical_xy(
        self,
        board: BoardState,
        pos: Position,
    ) -> Optional[tuple[int, int]]:
        """Return (cx, cy) in [0, 21)×[0, 21) or None if off-grid."""
        cx, cy = _to_canonical_xy(board, pos)
        if not (0 <= cx < HEX_BOARD_SIZE and 0 <= cy < HEX_BOARD_SIZE):
            return None
        return cx, cy

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Map a hex move into a [0, policy_size) index.

        This encoder is only valid for BoardType.HEXAGONAL boards of the
        canonical radius (size == 11). For any non-hex geometry or
        geometries that do not match the canonical frame, the move is
        treated as unrepresentable and INVALID_MOVE_INDEX is returned.
        """
        if board.type != BoardType.HEXAGONAL:
            return INVALID_MOVE_INDEX

        # Defensive guard: only support the canonical N=10 hex for now.
        if _infer_board_size(board) != HEX_BOARD_SIZE:
            return INVALID_MOVE_INDEX

        # --- Placements ---
        if move.type == MoveType.PLACE_RING:
            canon = self._encode_canonical_xy(board, move.to)
            if canon is None:
                return INVALID_MOVE_INDEX
            cx, cy = canon

            pos_idx = cy * HEX_BOARD_SIZE + cx
            count = move.placement_count or 1
            if count < 1 or count > 3:
                return INVALID_MOVE_INDEX

            return pos_idx * 3 + (count - 1)

        # --- Movement / capture ---
        if move.type in (
            MoveType.MOVE_STACK,
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
        ):
            if move.from_pos is None:
                return INVALID_MOVE_INDEX

            from_canon = self._encode_canonical_xy(board, move.from_pos)
            to_canon = self._encode_canonical_xy(board, move.to)
            if from_canon is None or to_canon is None:
                return INVALID_MOVE_INDEX

            from_cx, from_cy = from_canon
            to_cx, to_cy = to_canon

            dx = to_cx - from_cx
            dy = to_cy - from_cy
            dist = max(abs(dx), abs(dy))
            if dist == 0 or dist > HEX_MAX_DIST:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist
            dir_y = dy // dist
            try:
                dir_idx = HEX_DIRS.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 6-direction scheme.
                return INVALID_MOVE_INDEX

            from_idx = from_cy * HEX_BOARD_SIZE + from_cx
            return (
                HEX_MOVEMENT_BASE
                + from_idx * (NUM_HEX_DIRS * HEX_MAX_DIST)
                + dir_idx * HEX_MAX_DIST
                + (dist - 1)
            )

        # --- Special ---
        if move.type == MoveType.SKIP_PLACEMENT:
            return HEX_SPECIAL_BASE

        return INVALID_MOVE_INDEX

    def decode_move(
        self,
        index: int,
        game_state: GameState,
    ) -> Optional[Move]:
        """Inverse of encode_move for hex boards.

        Returns a Move instance for valid indices whose endpoints lie on the
        true hex board, or None if the index is out of range or maps off
        the playable hex.
        """
        board = game_state.board

        if board.type != BoardType.HEXAGONAL:
            return None
        if _infer_board_size(board) != HEX_BOARD_SIZE:
            return None
        if index < 0 or index >= self.policy_size:
            return None

        # --- Placements ---
        if index < HEX_PLACEMENT_SPAN:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // HEX_BOARD_SIZE
            cx = pos_idx % HEX_BOARD_SIZE

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
                "type": MoveType.PLACE_RING,
                "player": game_state.current_player,
                "to": to_payload,
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Movement / capture ---
        if index < HEX_SPECIAL_BASE:
            offset = index - HEX_MOVEMENT_BASE

            dist_idx = offset % HEX_MAX_DIST
            dist = dist_idx + 1
            offset //= HEX_MAX_DIST

            dir_idx = offset % NUM_HEX_DIRS
            offset //= NUM_HEX_DIRS

            from_idx = offset
            from_cy = from_idx // HEX_BOARD_SIZE
            from_cx = from_idx % HEX_BOARD_SIZE

            from_pos = _from_canonical_xy(board, from_cx, from_cy)
            if from_pos is None or not BoardGeometry.is_within_bounds(
                from_pos, board.type, board.size
            ):
                return None

            dx, dy = HEX_DIRS[dir_idx]
            to_cx = from_cx + dx * dist
            to_cy = from_cy + dy * dist

            to_pos = _from_canonical_xy(board, to_cx, to_cy)
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
                "type": MoveType.MOVE_STACK,
                "player": game_state.current_player,
                "from": from_payload,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Special ---
        if index == HEX_SPECIAL_BASE:
            move_data = {
                "id": "decoded",
                "type": MoveType.SKIP_PLACEMENT,
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        return None


class HexNeuralNet(nn.Module):
    """Hex-specific CNN for the canonical N=10 hex board.

    This model mirrors the ResNet-style structure of :class:`RingRiftCNN`
    but operates on a fixed [C_hex, 21, 21] spatial frame and exposes a
    hex-only policy head of size :data:`P_HEX`.

    Architecture Version:
        v1.0.0 - Initial hex architecture with 8 residual blocks, 128 filters,
                 masked global average pooling, policy head size P_HEX=54244.

    Design (see AI_ARCHITECTURE.md, HexNeuralNet section):

      * Backbone: initial 3×3 conv → BN → ReLU, followed by ``num_res_blocks``
        residual blocks with 3×3 convolutions (stride 1, padding 1).
      * Value head:
          - 1×1 conv → BN → ReLU producing a single-channel map;
          - masked global average pooling over the 21×21 hex frame using
            ``hex_mask`` when provided;
          - concatenation with the ``global_features`` vector; and
          - a small MLP + tanh to produce a scalar in [-1, 1].
      * Policy head:
          - 1×1 conv → BN → ReLU on the shared backbone features;
          - flatten over [H, W]; and
          - a linear layer to :data:`P_HEX` logits.
    """

    # Architecture version for checkpoint compatibility checking
    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(
        self,
        in_channels: int,
        global_features: int,
        num_res_blocks: int = 8,
        num_filters: int = 128,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = P_HEX,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.board_size = board_size
        # Hex-only policy dimension (P_HEX).
        self.policy_size = policy_size

        # Shared backbone (spatial conv trunk + residual blocks).
        self.conv1 = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Value head: 1×1 conv → BN → ReLU → masked global avg pool → MLP.
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        # Pooled value features (1) + global_features.
        self.value_fc1 = nn.Linear(1 + global_features, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

        # Policy head: 1×1 conv → BN → ReLU → flatten → linear to P_HEX.
        self.policy_conv = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(num_filters)
        self.policy_fc = nn.Linear(
            num_filters * board_size * board_size, self.policy_size
        )

    def _masked_global_avg_pool(
        self,
        x: torch.Tensor,
        hex_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform masked global average pooling over [H, W].

        Args:
          x: [B, 1, H, W] tensor from the value head conv.
          hex_mask: optional [B, 1, H, W] mask (1 = valid hex cell,
            0 = padding). If None, falls back to unmasked mean.
        """
        if hex_mask is None:
            return x.mean(dim=(2, 3))  # [B, 1]

        # Ensure mask is float and on the same device.
        mask = hex_mask.to(dtype=x.dtype, device=x.device)
        masked = x * mask
        # Sum over spatial dims.
        num = masked.sum(dim=(2, 3))  # [B, 1]
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return num / denom

    def forward(
        self,
        x: torch.Tensor,
        globals: torch.Tensor,
        hex_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
          x: [B, C_hex, H, W] feature tensor on the hex bounding box.
          globals: [B, global_features] dense global features.
          hex_mask: optional [B, 1, H, W] mask (1 = real hex cell,
            0 = padding) for masked pooling in the value head.

        Returns:
          (value, policy_logits):
            - value: [B, 1] tensor in [-1, 1].
            - policy_logits: [B, P_HEX] unnormalised logits.
        """
        # Backbone trunk.
        out = self.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head.
        v = self.value_conv(out)
        v = self.relu(self.value_bn(v))
        v_pooled = self._masked_global_avg_pool(v, hex_mask)  # [B, 1]
        # Concatenate pooled value features with globals.
        v_cat = torch.cat([v_pooled, globals], dim=1)
        v_hidden = self.relu(self.value_fc1(v_cat))
        v_out = self.tanh(self.value_fc2(v_hidden))  # [B, 1]

        # Policy head.
        p = self.policy_conv(out)
        p = self.relu(self.policy_bn(p))
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)
        p_logits = self.policy_fc(p_flat)  # [B, P_HEX]

        return v_out, p_logits
