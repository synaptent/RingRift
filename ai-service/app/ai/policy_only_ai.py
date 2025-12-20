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
from typing import Optional, List, Dict, Any, cast

import numpy as np

from .base import BaseAI
from .neural_net import NeuralNetAI, INVALID_MOVE_INDEX
from ..models import GameState, Move, AIConfig, BoardType

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

    Attributes:
        neural_net: The loaded neural network for policy evaluation.
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
        super().__init__(player_number, config)

        self.board_type = board_type
        self.temperature = config.policy_temperature or 1.0

        # Load neural network
        self.neural_net: Optional[NeuralNetAI] = None
        try:
            self.neural_net = NeuralNetAI(player_number, config)
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

    def select_move(self, game_state: GameState) -> Optional[Move]:
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
        valid_moves: List[Move],
    ) -> np.ndarray:
        """Get policy scores for each valid move.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves to score.

        Returns:
            Array of scores for each move (higher = better).
        """
        if self.neural_net is None:
            # Uniform policy when no neural net available
            return np.ones(len(valid_moves))

        try:
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
        valid_moves: List[Move],
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
        except Exception:
            return 0.0

    def get_policy_distribution(
        self,
        game_state: GameState,
    ) -> Dict[str, float]:
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

        return {str(move): float(prob) for move, prob in zip(valid_moves, probs)}

    def reset_for_new_game(self, *, rng_seed: Optional[int] = None) -> None:
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
