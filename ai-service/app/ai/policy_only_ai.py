"""Policy-Only AI implementation for RingRift.

This module implements a simple neural network policy-only agent that selects
moves based solely on the neural network's policy output without any search.
This provides an extremely fast baseline for evaluation and calibration.

The agent performs a single forward pass through the neural network and
samples from the resulting policy distribution (or takes argmax with low
temperature). This is useful for:

- Fast baseline evaluation of neural network quality
- Rapid selfplay data generation (for behavior cloning)
- Calibrating NN policy strength independent of search
- Comparison with search-based methods (MCTS, Descent)

The policy temperature controls exploration vs exploitation:
- temperature <= 0.01: Greedy (argmax)
- temperature = 1.0: Sample from policy (default)
- temperature > 1.0: More uniform sampling
"""

from __future__ import annotations

import logging

import numpy as np

from ..models import AIConfig, BoardType, GameState, Move
from .base import BaseAI
from .neural_net import INVALID_MOVE_INDEX, NeuralNetAI

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature scaling.

    Args:
        x: Input logits array.
        temperature: Temperature for scaling. Lower = more peaked.

    Returns:
        Probability distribution over actions.
    """
    # Handle potential overflow by subtracting max
    x_scaled = x / temperature
    x_scaled = x_scaled - np.max(x_scaled)
    exp_x = np.exp(x_scaled)
    return exp_x / np.sum(exp_x)


class PolicyOnlyAI(BaseAI):
    """AI that uses only neural network policy output without search.

    Extremely fast (single forward pass) but typically weaker than
    search-based methods. The strength depends entirely on the quality
    of the neural network policy head.

    Supports both Full NN and NNUE policy models (Dec 2025 - Phase 3):
        - Set use_nnue_policy=True in AIConfig to use NNUE instead of Full NN
        - NNUE models are smaller (~5-10MB vs ~30-100MB) and faster
        - Falls back to uniform policy if neither NN is available

    Attributes:
        neural_net: The loaded neural network for policy evaluation.
        nnue_policy: The loaded NNUE policy model (alternative to neural_net).
        temperature: Temperature for policy softmax sampling.
        board_type: The board type this AI is configured for.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType = BoardType.SQUARE8,
    ) -> None:
        """Initialize the Policy-Only AI.

        Args:
            player_number: The player number this AI controls (1-based).
            config: AI configuration including nn_model_id and policy_temperature.
            board_type: The board type for encoding moves.

        Raises:
            RuntimeError: If neural network cannot be loaded and allow_fresh_weights
                is False.
        """
        import os

        super().__init__(player_number, config)

        self.board_type = board_type
        self.temperature = config.policy_temperature or 1.0

        # Dec 2025 - Phase 3: NNUE policy support
        nnue_env = os.environ.get("RINGRIFT_POLICY_ONLY_USE_NNUE", "").lower() in (
            "1", "true", "yes", "on"
        )
        self.use_nnue_policy: bool = getattr(config, 'use_nnue_policy', False) or nnue_env
        self.nnue_policy: "RingRiftNNUEWithPolicy | None" = None  # type: ignore

        # Log expected encoder configuration for debugging
        try:
            from app.training.encoder_registry import get_encoder_config
            # Handle both BoardType enum and string arguments
            board_type_name = (
                self.board_type.name if hasattr(self.board_type, "name")
                else str(self.board_type).upper()
            )
            for version in ["v2", "v3"]:
                enc_config = get_encoder_config(self.board_type, version)
                logger.debug(
                    f"PolicyOnlyAI: {board_type_name} {version} expects "
                    f"{enc_config.in_channels}ch ({enc_config.encoder_type})"
                )
        except ImportError:
            pass  # Registry not available

        # Load neural network (NNUE or Full NN)
        self.neural_net: NeuralNetAI | None = None

        if self.use_nnue_policy:
            # Try to load NNUE policy model first
            self._load_nnue_policy(config)
            if self.nnue_policy is not None:
                logger.info(
                    f"PolicyOnlyAI(player={player_number}): using NNUE policy "
                    f"(temp={self.temperature})"
                )
                return  # NNUE loaded successfully

        # Fall back to Full NN
        try:
            self.neural_net = NeuralNetAI(player_number, config, board_type=self.board_type)
            logger.info(
                f"PolicyOnlyAI(player={player_number}): loaded neural network "
                f"(model={config.nn_model_id}, temp={self.temperature})"
            )
        except Exception as e:
            if not config.allow_fresh_weights:
                raise RuntimeError(
                    f"PolicyOnlyAI requires a neural network but failed to load: {e}"
                ) from e
            logger.warning(
                f"PolicyOnlyAI(player={player_number}): failed to load neural net "
                f"({e}), will use uniform policy"
            )
            self.neural_net = None

    def _load_nnue_policy(self, config: AIConfig) -> None:
        """Load NNUE policy model if available."""
        try:
            import os
            from .nnue_policy import RingRiftNNUEWithPolicy
            from app.utils.torch_utils import safe_load_checkpoint

            # Try to find NNUE policy model
            num_players = getattr(config, 'num_players', 2)
            # Handle both BoardType enum and string arguments
            board_type_str = (
                self.board_type.value if hasattr(self.board_type, "value")
                else str(self.board_type).lower()
            )
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "models", "nnue", f"nnue_policy_{board_type_str}_{num_players}p.pt"
            )
            model_path = os.path.normpath(model_path)

            if os.path.exists(model_path):
                checkpoint = safe_load_checkpoint(model_path, map_location="cpu", warn_on_unsafe=False)

                # Handle versioned checkpoints
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    hidden_dim = checkpoint.get('hidden_dim', 128)
                    num_hidden_layers = checkpoint.get('num_hidden_layers', 2)
                else:
                    state_dict = checkpoint
                    hidden_dim = 128
                    num_hidden_layers = 2

                model = RingRiftNNUEWithPolicy(
                    board_type=self.board_type,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                model.load_state_dict(state_dict)
                model.eval()

                self.nnue_policy = model
                logger.debug(f"PolicyOnlyAI: Loaded NNUE policy from {model_path}")
            else:
                logger.debug(f"PolicyOnlyAI: No NNUE policy model at {model_path}")

        except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.debug(f"PolicyOnlyAI: Could not load NNUE policy: {e}")

    def _get_nnue_policy_scores(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> np.ndarray:
        """Get policy scores using NNUE policy model.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves to score.

        Returns:
            Array of scores for each move (higher = better).
        """
        try:
            import torch
            from .nnue import extract_features_from_gamestate

            # Extract NNUE features
            features = extract_features_from_gamestate(game_state)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Forward pass through NNUE
            with torch.no_grad():
                _, policy = self.nnue_policy(features_tensor, return_policy=True)

            if policy is None:
                return np.ones(len(valid_moves))

            policy_vec = policy[0].numpy()

            # Map valid moves to policy indices
            scores = []
            for move in valid_moves:
                idx = self._encode_move_for_nnue(move, game_state.board)
                if idx is not None and 0 <= idx < len(policy_vec):
                    score = float(policy_vec[idx])
                else:
                    score = 1e-8  # Small non-zero for unmapped moves
                scores.append(score)

            return np.array(scores, dtype=np.float32)

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f"PolicyOnlyAI: NNUE policy evaluation failed ({e})")
            return np.ones(len(valid_moves))

    def _encode_move_for_nnue(self, move: Move, board: "Board") -> int | None:  # type: ignore
        """Encode a move to NNUE policy index."""
        try:
            # Use the same encoding as the NNUE policy head
            from .nnue_policy import encode_move_for_policy

            return encode_move_for_policy(move, board)
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
            # Fallback: linear move index based on position
            if hasattr(move, 'to') and move.to is not None:
                to_pos = move.to
                if hasattr(to_pos, 'x') and hasattr(to_pos, 'y'):
                    grid_size = getattr(board, 'size', 8)
                    return to_pos.x * grid_size + to_pos.y
            return None

    def select_move(self, game_state: GameState) -> Move | None:
        """Select a move based on neural network policy output.

        Performs a single forward pass through the neural network and
        samples from the resulting policy distribution.

        Args:
            game_state: Current game state.

        Returns:
            Selected move or None if no valid moves exist.
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            self.move_count += 1
            return valid_moves[0]

        # Check for swap decision first
        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # Get move scores from policy
        move_scores = self._get_policy_scores(game_state, valid_moves)

        # Select move based on temperature
        selected_move = self._sample_from_policy(valid_moves, move_scores)

        self.move_count += 1
        return selected_move

    def _get_policy_scores(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> np.ndarray:
        """Get policy scores for each valid move.

        Evaluation priority (Dec 2025 - Phase 3):
            1. NNUE policy if use_nnue_policy=True and model loaded
            2. Full NN policy if neural_net loaded
            3. Uniform policy fallback

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves to score.

        Returns:
            Array of scores for each move (higher = better).
        """
        # NNUE policy path (Dec 2025 - Phase 3)
        if self.nnue_policy is not None:
            return self._get_nnue_policy_scores(game_state, valid_moves)

        if self.neural_net is None:
            # Uniform policy when no neural net available
            return np.ones(len(valid_moves))

        try:
            # Ensure model is initialized for this board type before extracting features
            # This sets up the correct encoder (hex vs square) based on the model architecture
            from .game_state_utils import infer_num_players
            self.neural_net._ensure_model_initialized(
                game_state.board.type,
                num_players=infer_num_players(game_state),
            )

            # Update history for the current game state (critical for model accuracy)
            # The neural network was trained with history, so we need to track it
            current_features, _ = self.neural_net._extract_features(game_state)
            game_id = game_state.id

            if game_id not in self.neural_net.game_history:
                self.neural_net.game_history[game_id] = []

            # Append current state to history
            self.neural_net.game_history[game_id].append(current_features)

            # Keep only needed history (history_length + 1 for current)
            max_hist = self.neural_net.history_length + 1
            if len(self.neural_net.game_history[game_id]) > max_hist:
                self.neural_net.game_history[game_id] = self.neural_net.game_history[game_id][-max_hist:]

            # Single forward pass
            _, policy = self.neural_net.evaluate_batch([game_state])

            if policy.size == 0:
                return np.ones(len(valid_moves))

            policy_vec = policy[0]

            # Map valid moves to policy indices and extract scores
            scores = []
            for move in valid_moves:
                idx = self.neural_net.encode_move(move, game_state.board)
                if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy_vec):
                    score = float(policy_vec[idx])
                else:
                    # Small non-zero score for moves not in policy (shouldn't happen)
                    score = 1e-8
                scores.append(score)

            return np.array(scores, dtype=np.float32)

        except Exception as e:
            logger.warning(f"PolicyOnlyAI: policy evaluation failed ({e})")
            return np.ones(len(valid_moves))

    def _sample_from_policy(
        self,
        valid_moves: list[Move],
        scores: np.ndarray,
    ) -> Move:
        """Sample a move from the policy distribution.

        Args:
            valid_moves: List of valid moves.
            scores: Policy scores for each move.

        Returns:
            Selected move.
        """
        # Avoid division by zero
        if np.sum(scores) <= 0:
            scores = np.ones_like(scores)

        # Greedy selection for very low temperature
        if self.temperature <= 0.01:
            return valid_moves[np.argmax(scores)]

        # Temperature-scaled softmax sampling
        # Use log-scores to apply temperature in log-space (more stable)
        log_scores = np.log(np.maximum(scores, 1e-10))
        probs = softmax(log_scores, self.temperature)

        # Sample using per-instance RNG for reproducibility
        cumsum = np.cumsum(probs)
        r = self.rng.random()
        idx = np.searchsorted(cumsum, r)
        idx = min(idx, len(valid_moves) - 1)  # Safety clamp

        return valid_moves[idx]

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using neural network value head.

        Args:
            game_state: Current game state.

        Returns:
            Value estimate from the neural network (or 0.0 if unavailable).
        """
        if self.neural_net is None:
            return 0.0

        try:
            values, _ = self.neural_net.evaluate_batch([game_state])
            return values[0] if values else 0.0
        except (RuntimeError, TypeError, IndexError):
            return 0.0

    def get_policy_distribution(
        self,
        game_state: GameState,
    ) -> dict[str, float]:
        """Get the full policy distribution over valid moves.

        Useful for debugging and training data extraction.

        Args:
            game_state: Current game state.

        Returns:
            Dictionary mapping move string representations to probabilities.
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return {}

        scores = self._get_policy_scores(game_state, valid_moves)

        # Normalize to probabilities
        if self.temperature <= 0.01:
            # For greedy, return one-hot
            probs = np.zeros_like(scores)
            probs[np.argmax(scores)] = 1.0
        else:
            log_scores = np.log(np.maximum(scores, 1e-10))
            probs = softmax(log_scores, self.temperature)

        return {str(move): float(prob) for move, prob in zip(valid_moves, probs, strict=False)}

    def reset_for_new_game(self, *, rng_seed: int | None = None) -> None:
        """Reset state for a new game.

        Args:
            rng_seed: Optional new RNG seed.
        """
        super().reset_for_new_game(rng_seed=rng_seed)
        # No additional state to reset for policy-only

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PolicyOnlyAI(player={self.player_number}, "
            f"temp={self.temperature}, "
            f"model={self.config.nn_model_id})"
        )
