"""NNUE Adapter for MCTS/Gumbel Search Engines.

This module provides adapters that allow NNUE models (value-only or with policy)
to be used with search algorithms that expect the standard NeuralNetAI interface.

There are two approaches:
1. Direct: Use RingRiftNNUEWithPolicy directly - it already has policy heads
2. Adapter: NNUEMCTSAdapter wraps value-only NNUE and derives policy from values

The adapter derives policy logits by evaluating child positions:
    policy_logit(move) = (child_value - parent_value) / temperature

This is slower than a learned policy head but works for any value-only model.

Usage:
    from app.ai.nnue_adapter import (
        NNUEMCTSAdapter,
        create_nnue_for_search,
        NNUEWithPolicyAdapter,
    )

    # For value-only NNUE - derives policy from position evaluations
    nnue_model = load_nnue_model(...)  # RingRiftNNUE
    adapter = NNUEMCTSAdapter(nnue_model, board_type=BoardType.HEX8)
    gumbel_engine = GumbelSearchEngine(neural_net=adapter)

    # For NNUE with policy head - uses native policy
    nnue_policy = load_nnue_policy_model(...)  # RingRiftNNUEWithPolicy
    adapter = NNUEWithPolicyAdapter(nnue_policy, board_type=BoardType.HEX8)
    gumbel_engine = GumbelSearchEngine(neural_net=adapter)

    # Auto-detect and create appropriate adapter
    adapter = create_nnue_for_search(model_path, board_type)

December 2025: Created for Unified AI Evaluation Architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.models import BoardType, GameState, Move, Position

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.ai.nnue import RingRiftNNUE
    from app.ai.nnue_policy import RingRiftNNUEWithPolicy


# =============================================================================
# Constants
# =============================================================================

# Default temperature for value-to-policy conversion
# Higher = more exploration, lower = more exploitation
DEFAULT_POLICY_TEMPERATURE = 1.0

# Logit clipping to prevent extreme values
MIN_LOGIT = -10.0
MAX_LOGIT = 10.0

# Invalid move index (matches neural_net constants)
INVALID_MOVE_INDEX = -1


# =============================================================================
# Value-to-Policy Adapter (for value-only NNUE)
# =============================================================================


@dataclass
class PolicyFromValueConfig:
    """Configuration for deriving policy from value evaluations."""

    temperature: float = DEFAULT_POLICY_TEMPERATURE
    use_softmax: bool = True
    min_logit: float = MIN_LOGIT
    max_logit: float = MAX_LOGIT
    batch_child_eval: bool = True  # Batch child position evaluations


class NNUEMCTSAdapter:
    """Adapt value-only NNUE to work with MCTS/Gumbel interfaces.

    This adapter allows value-only NNUE models (RingRiftNNUE) to be used
    with search algorithms that expect policy logits.

    Policy derivation strategy:
    - For each legal move, compute the value of the resulting position
    - Convert value differences to logits: logit = (child_value - parent_value) / T
    - Apply softmax to get probability distribution

    This is slower than a learned policy head but works for any value model.
    For better performance, use RingRiftNNUEWithPolicy directly via
    NNUEWithPolicyAdapter.

    Attributes:
        nnue: The underlying value-only NNUE model
        board_type: Board type for move encoding
        config: Policy derivation configuration
        num_players: Number of players (for value head selection)
    """

    ADAPTER_VERSION = "1.0.0"

    def __init__(
        self,
        nnue_model: "RingRiftNNUE",
        board_type: BoardType,
        num_players: int = 2,
        config: PolicyFromValueConfig | None = None,
    ):
        """Initialize the adapter.

        Args:
            nnue_model: Value-only NNUE model (RingRiftNNUE)
            board_type: Board type for move encoding
            num_players: Number of players in the game
            config: Policy derivation configuration
        """
        self.nnue = nnue_model
        self.board_type = board_type
        self.num_players = num_players
        self.config = config or PolicyFromValueConfig()

        # Cache device from model
        try:
            self._device = next(nnue_model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        # Get action encoder for move encoding
        self._action_encoder = self._get_action_encoder()

        logger.info(
            f"NNUEMCTSAdapter initialized: board={board_type.value}, "
            f"players={num_players}, device={self._device}"
        )

    def _get_action_encoder(self):
        """Get the appropriate action encoder for the board type."""
        from app.ai.neural_net import get_action_encoder

        return get_action_encoder(self.board_type)

    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self._device

    def evaluate_batch(
        self,
        states: list[GameState],
        value_head: int | None = None,
    ) -> tuple[list[float], np.ndarray]:
        """Evaluate batch of states with value and derived policy.

        This is the main interface expected by GumbelMCTSAI and other search
        algorithms. Returns both values and policy logits.

        Args:
            states: List of game states to evaluate
            value_head: Player perspective for value (1-indexed)

        Returns:
            Tuple of (values, policies):
            - values: List of value estimates per state
            - policies: np.ndarray of policy logits (batch, policy_size)
        """
        if not states:
            return [], np.array([])

        batch_size = len(states)

        # Get values from NNUE
        values = self._evaluate_values(states, value_head)

        # Derive policy from value differences
        policies = self._derive_policies(states, values, value_head)

        return values, policies

    def _evaluate_values(
        self,
        states: list[GameState],
        value_head: int | None = None,
    ) -> list[float]:
        """Evaluate position values using NNUE.

        Args:
            states: List of game states
            value_head: Player perspective (1-indexed)

        Returns:
            List of value estimates [-1, 1]
        """
        from app.ai.nnue import extract_features_from_gamestate

        # Extract features for all states
        player_numbers = [
            (value_head or state.current_player) for state in states
        ]

        features_list = [
            extract_features_from_gamestate(state, pn)
            for state, pn in zip(states, player_numbers, strict=False)
        ]
        features = np.stack(features_list, axis=0)

        # Evaluate with NNUE
        self.nnue.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self._device)
            values_tensor = self.nnue.forward(x)

            # Handle multi-player value heads
            if values_tensor.dim() > 1 and values_tensor.shape[1] > 1:
                # Select value for the relevant player
                values = []
                for i, state in enumerate(states):
                    player_idx = (value_head or state.current_player) - 1
                    values.append(float(values_tensor[i, player_idx].item()))
            else:
                values = values_tensor.squeeze(-1).cpu().numpy().tolist()

        return values

    def _derive_policies(
        self,
        states: list[GameState],
        parent_values: list[float],
        value_head: int | None = None,
    ) -> np.ndarray:
        """Derive policy logits from value evaluations.

        For each legal move in each state, evaluates the resulting position
        and computes a logit based on the value improvement.

        Args:
            states: List of game states
            parent_values: Value estimates for parent positions
            value_head: Player perspective

        Returns:
            np.ndarray of policy logits (batch, policy_size)
        """
        from app.ai.neural_net import get_policy_size_for_board
        from app.rules.game_engine import GameEngine

        policy_size = get_policy_size_for_board(self.board_type)
        batch_size = len(states)

        # Initialize with very negative values (masked moves)
        policies = np.full((batch_size, policy_size), self.config.min_logit)

        for batch_idx, (state, parent_value) in enumerate(
            zip(states, parent_values, strict=False)
        ):
            legal_moves = GameEngine.get_legal_moves(state)
            if not legal_moves:
                continue

            # Batch evaluate all child positions
            if self.config.batch_child_eval and len(legal_moves) > 1:
                child_states = []
                move_indices = []

                for move in legal_moves:
                    try:
                        child_state = GameEngine.apply_move(state, move)
                        child_states.append(child_state)

                        # Encode move to policy index
                        idx = self._action_encoder.encode_move(move, state.board)
                        if idx != INVALID_MOVE_INDEX and 0 <= idx < policy_size:
                            move_indices.append((len(child_states) - 1, idx))
                    except (ValueError, RuntimeError) as e:
                        # Invalid move - skip
                        logger.debug(f"Skipping invalid move: {e}")
                        continue

                if child_states:
                    child_values = self._evaluate_values(child_states, value_head)

                    for child_idx, move_idx in move_indices:
                        # Compute logit: better moves get higher logits
                        advantage = child_values[child_idx] - parent_value
                        logit = advantage / self.config.temperature
                        logit = max(self.config.min_logit, min(self.config.max_logit, logit))
                        policies[batch_idx, move_idx] = logit

            else:
                # Sequential evaluation (for small move sets)
                for move in legal_moves:
                    try:
                        child_state = GameEngine.apply_move(state, move)
                        child_values = self._evaluate_values([child_state], value_head)
                        child_value = child_values[0]

                        idx = self._action_encoder.encode_move(move, state.board)
                        if idx != INVALID_MOVE_INDEX and 0 <= idx < policy_size:
                            advantage = child_value - parent_value
                            logit = advantage / self.config.temperature
                            logit = max(self.config.min_logit, min(self.config.max_logit, logit))
                            policies[batch_idx, idx] = logit
                    except (ValueError, RuntimeError):
                        continue

        return policies

    def encode_move(self, move: Move, board: Any) -> int:
        """Encode a move to a policy index.

        Args:
            move: Move to encode
            board: Board context

        Returns:
            Policy index or INVALID_MOVE_INDEX
        """
        return self._action_encoder.encode_move(move, board)

    def get_policy_logits(self, game_state: GameState) -> dict[Move, float]:
        """Get policy logits for all legal moves.

        This is an alternative interface used by some search algorithms.

        Args:
            game_state: Current game state

        Returns:
            Dictionary mapping moves to logits
        """
        from app.rules.game_engine import GameEngine

        values, policies = self.evaluate_batch([game_state])
        parent_value = values[0] if values else 0.0

        legal_moves = GameEngine.get_legal_moves(game_state)
        move_logits = {}

        for move in legal_moves:
            idx = self.encode_move(move, game_state.board)
            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policies[0]):
                move_logits[move] = float(policies[0, idx])
            else:
                # Fallback: evaluate move directly
                try:
                    child_state = GameEngine.apply_move(game_state, move)
                    child_values = self._evaluate_values([child_state])
                    advantage = child_values[0] - parent_value
                    move_logits[move] = advantage / self.config.temperature
                except (ValueError, RuntimeError):
                    move_logits[move] = 0.0

        return move_logits

    def get_value(self, game_state: GameState) -> float:
        """Get value estimate for a position.

        Args:
            game_state: Current game state

        Returns:
            Value estimate in [-1, 1]
        """
        values = self._evaluate_values([game_state])
        return values[0] if values else 0.0


# =============================================================================
# Native Policy Adapter (for NNUE with policy head)
# =============================================================================


class NNUEWithPolicyAdapter:
    """Adapter for NNUE models with native policy heads.

    This adapter wraps RingRiftNNUEWithPolicy to provide the interface
    expected by MCTS/Gumbel search algorithms. Unlike NNUEMCTSAdapter,
    this uses the model's learned policy head directly, which is faster.

    The model outputs from/to position heatmaps which are combined to
    score each legal move.
    """

    ADAPTER_VERSION = "1.0.0"

    def __init__(
        self,
        nnue_policy_model: "RingRiftNNUEWithPolicy",
        board_type: BoardType,
        num_players: int = 2,
    ):
        """Initialize the adapter.

        Args:
            nnue_policy_model: NNUE model with policy head
            board_type: Board type for move encoding
            num_players: Number of players
        """
        self.nnue = nnue_policy_model
        self.board_type = board_type
        self.num_players = num_players

        try:
            self._device = next(nnue_policy_model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        self._action_encoder = self._get_action_encoder()
        self.board_size = self.nnue.board_size

        logger.info(
            f"NNUEWithPolicyAdapter initialized: board={board_type.value}, "
            f"board_size={self.board_size}, device={self._device}"
        )

    def _get_action_encoder(self):
        """Get the appropriate action encoder for the board type."""
        from app.ai.neural_net import get_action_encoder

        return get_action_encoder(self.board_type)

    @property
    def device(self) -> torch.device:
        """Device the model is on."""
        return self._device

    def evaluate_batch(
        self,
        states: list[GameState],
        value_head: int | None = None,
    ) -> tuple[list[float], np.ndarray]:
        """Evaluate batch with value and native policy.

        Args:
            states: List of game states
            value_head: Player perspective (1-indexed)

        Returns:
            Tuple of (values, policies)
        """
        from app.ai.nnue import extract_features_from_gamestate
        from app.ai.neural_net import get_policy_size_for_board
        from app.rules.game_engine import GameEngine

        if not states:
            return [], np.array([])

        batch_size = len(states)
        policy_size = get_policy_size_for_board(self.board_type)

        # Extract features
        player_numbers = [
            (value_head or state.current_player) for state in states
        ]
        features_list = [
            extract_features_from_gamestate(state, pn)
            for state, pn in zip(states, player_numbers, strict=False)
        ]
        features = np.stack(features_list, axis=0)

        # Forward pass with policy
        self.nnue.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self._device)
            value, from_logits, to_logits = self.nnue.forward(x, return_policy=True)

            # Extract values
            values = value.squeeze(-1).cpu().numpy().tolist()

            # Convert from/to logits to move-level policy
            from_np = from_logits.cpu().numpy()  # (batch, H*W)
            to_np = to_logits.cpu().numpy()  # (batch, H*W)

        # Build policy from from/to logits
        policies = np.full((batch_size, policy_size), MIN_LOGIT)

        for batch_idx, state in enumerate(states):
            legal_moves = GameEngine.get_legal_moves(state)

            for move in legal_moves:
                # Get from/to positions
                from_pos = self._get_from_position(move)
                to_pos = self._get_to_position(move)

                # Combine logits
                logit = 0.0
                if from_pos is not None:
                    from_idx = self._position_to_index(from_pos)
                    if 0 <= from_idx < len(from_np[batch_idx]):
                        logit += from_np[batch_idx, from_idx]

                if to_pos is not None:
                    to_idx = self._position_to_index(to_pos)
                    if 0 <= to_idx < len(to_np[batch_idx]):
                        logit += to_np[batch_idx, to_idx]

                # Encode to policy index
                idx = self._action_encoder.encode_move(move, state.board)
                if idx != INVALID_MOVE_INDEX and 0 <= idx < policy_size:
                    policies[batch_idx, idx] = logit

        return values, policies

    def _get_from_position(self, move: Move) -> Position | None:
        """Extract from position from a move."""
        # Handle different move types
        if hasattr(move, "from_pos") and move.from_pos is not None:
            return move.from_pos
        if hasattr(move, "position"):
            return move.position
        return None

    def _get_to_position(self, move: Move) -> Position | None:
        """Extract to position from a move."""
        if hasattr(move, "to") and move.to is not None:
            return move.to
        if hasattr(move, "position") and not hasattr(move, "from_pos"):
            return move.position
        return None

    def _position_to_index(self, pos: Position) -> int:
        """Convert a position to a flattened index.

        Args:
            pos: Position with row/col

        Returns:
            Flattened index for the board
        """
        if hasattr(pos, "row") and hasattr(pos, "col"):
            return pos.row * self.board_size + pos.col
        return -1

    def encode_move(self, move: Move, board: Any) -> int:
        """Encode a move to a policy index."""
        return self._action_encoder.encode_move(move, board)

    def get_policy_logits(self, game_state: GameState) -> dict[Move, float]:
        """Get policy logits for all legal moves."""
        from app.rules.game_engine import GameEngine

        _, policies = self.evaluate_batch([game_state])
        legal_moves = GameEngine.get_legal_moves(game_state)

        move_logits = {}
        for move in legal_moves:
            idx = self.encode_move(move, game_state.board)
            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policies[0]):
                move_logits[move] = float(policies[0, idx])
            else:
                move_logits[move] = 0.0

        return move_logits

    def get_value(self, game_state: GameState) -> float:
        """Get value estimate for a position."""
        values, _ = self.evaluate_batch([game_state])
        return values[0] if values else 0.0


# =============================================================================
# Factory Functions
# =============================================================================


def create_nnue_for_search(
    model_path: str | Path,
    board_type: BoardType,
    num_players: int = 2,
    device: str = "cuda",
    prefer_policy: bool = True,
) -> NNUEMCTSAdapter | NNUEWithPolicyAdapter:
    """Create an NNUE adapter suitable for MCTS/Gumbel search.

    Automatically detects whether the model has a policy head and creates
    the appropriate adapter.

    Args:
        model_path: Path to the NNUE checkpoint
        board_type: Board type for the model
        num_players: Number of players
        device: Device to load model on
        prefer_policy: If True and model has policy, use native policy

    Returns:
        Either NNUEWithPolicyAdapter (if model has policy) or NNUEMCTSAdapter
    """
    from app.ai.unified_loader import ModelArchitecture, UnifiedModelLoader

    loader = UnifiedModelLoader()
    loaded = loader.load(str(model_path), board_type)

    if loaded.architecture == ModelArchitecture.NNUE_WITH_POLICY and prefer_policy:
        logger.info(f"Using native policy adapter for {model_path}")
        return NNUEWithPolicyAdapter(
            nnue_policy_model=loaded.model,
            board_type=board_type,
            num_players=num_players,
        )
    elif loaded.architecture in (
        ModelArchitecture.NNUE_VALUE_ONLY,
        ModelArchitecture.NNUE_WITH_POLICY,
    ):
        logger.info(f"Using value-derived policy adapter for {model_path}")
        return NNUEMCTSAdapter(
            nnue_model=loaded.model,
            board_type=board_type,
            num_players=num_players,
        )
    else:
        raise ValueError(
            f"Model at {model_path} is not an NNUE model "
            f"(detected: {loaded.architecture})"
        )


def load_nnue_with_policy(
    model_path: str | Path,
    board_type: BoardType,
    num_players: int = 2,
    device: str = "cuda",
) -> NNUEWithPolicyAdapter:
    """Load an NNUE model with policy head for search.

    Args:
        model_path: Path to NNUE+Policy checkpoint
        board_type: Board type
        num_players: Number of players
        device: Device to load on

    Returns:
        NNUEWithPolicyAdapter ready for MCTS/Gumbel

    Raises:
        ValueError: If model doesn't have policy head
    """
    from app.ai.unified_loader import ModelArchitecture, UnifiedModelLoader

    loader = UnifiedModelLoader()
    loaded = loader.load(str(model_path), board_type)

    if loaded.architecture != ModelArchitecture.NNUE_WITH_POLICY:
        raise ValueError(
            f"Model at {model_path} does not have a policy head "
            f"(detected: {loaded.architecture}). "
            "Use create_nnue_for_search() for value-only models."
        )

    return NNUEWithPolicyAdapter(
        nnue_policy_model=loaded.model,
        board_type=board_type,
        num_players=num_players,
    )


def wrap_nnue_value_only(
    nnue_model: "RingRiftNNUE",
    board_type: BoardType,
    num_players: int = 2,
    temperature: float = DEFAULT_POLICY_TEMPERATURE,
) -> NNUEMCTSAdapter:
    """Wrap a value-only NNUE model for search.

    Use this when you already have a loaded RingRiftNNUE model
    and want to use it with MCTS/Gumbel.

    Args:
        nnue_model: Already-loaded RingRiftNNUE model
        board_type: Board type
        num_players: Number of players
        temperature: Temperature for value-to-policy conversion

    Returns:
        NNUEMCTSAdapter ready for MCTS/Gumbel
    """
    config = PolicyFromValueConfig(temperature=temperature)
    return NNUEMCTSAdapter(
        nnue_model=nnue_model,
        board_type=board_type,
        num_players=num_players,
        config=config,
    )
