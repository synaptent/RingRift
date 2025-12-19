"""Energy-Based Move Optimization (EBMO) AI Agent.

A novel AI that uses gradient descent on continuous action embeddings
at inference time to find optimal moves.

Unlike traditional AI approaches:
- Policy networks: softmax -> sample/argmax
- MCTS: tree search -> visit counts
- Minimax: depth-limited search

EBMO:
1. Encodes the state once
2. Runs gradient descent on action embeddings to minimize energy
3. Projects optimized embedding to nearest legal move

Usage:
    from app.ai.ebmo_ai import EBMO_AI
    from app.models import AIConfig

    config = AIConfig(difficulty=5)
    ai = EBMO_AI(player_number=1, config=config)
    move = ai.select_move(game_state)
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseAI
from .ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    ActionFeatureExtractor,
    load_ebmo_model,
)
from ..models import AIConfig, GameState, Move

logger = logging.getLogger(__name__)

# Environment variable for model path
EBMO_MODEL_PATH_ENV = "RINGRIFT_EBMO_MODEL_PATH"
EBMO_DEFAULT_MODEL_PATH = "models/ebmo/ebmo_square8_best.pt"


class EBMO_AI(BaseAI):
    """Energy-Based Move Optimization AI.

    Uses gradient descent on continuous action embeddings to find
    moves that minimize the learned energy function.

    Key algorithm:
    1. Encode game state -> state embedding (done once)
    2. For each restart:
       a. Initialize action embedding from random legal move
       b. Run gradient descent: a' = a - lr * grad_a E(s, a)
       c. Periodically project to legal move manifold
       d. Select final move with lowest energy
    3. Return best move across all restarts
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        model_path: Optional[str] = None,
        ebmo_config: Optional[EBMOConfig] = None,
    ) -> None:
        """Initialize EBMO AI.

        Args:
            player_number: Player number this AI controls (1-based)
            config: AI configuration
            model_path: Path to trained EBMO model (optional)
            ebmo_config: EBMO-specific configuration (optional)
        """
        super().__init__(player_number, config)

        # EBMO configuration
        self.ebmo_config = ebmo_config or EBMOConfig()

        # Override from AIConfig if available
        self._load_config_from_ai_config(config)

        # Device selection
        self.device = self._select_device()

        # Load or create network
        self.network: Optional[EBMONetwork] = None
        self._model_loaded = False

        if model_path:
            self._load_model(model_path)
        else:
            # Try environment variable
            env_path = os.environ.get(EBMO_MODEL_PATH_ENV)
            if env_path and Path(env_path).exists():
                self._load_model(env_path)
            elif Path(EBMO_DEFAULT_MODEL_PATH).exists():
                self._load_model(EBMO_DEFAULT_MODEL_PATH)
            else:
                # Create fresh network (for testing/untrained play)
                logger.warning(
                    "[EBMO_AI] No trained model found, using fresh weights. "
                    "Performance will be random until trained."
                )
                self.network = EBMONetwork(self.ebmo_config)
                self.network.to(self.device)
                self.network.eval()

        # Feature extractor
        self.feature_extractor = ActionFeatureExtractor(self.ebmo_config.board_size)

        # Inference statistics
        self._total_moves = 0
        self._total_optim_steps = 0

        # Selfplay integration: temperature for exploration
        self.temperature: float = 1.0

        # Training feedback: last move's root value and policy
        self.last_root_value: Optional[float] = None
        self.last_root_policy: Optional[Dict[int, float]] = None

    def _load_config_from_ai_config(self, config: AIConfig) -> None:
        """Extract EBMO parameters from AIConfig.extra."""
        extra = getattr(config, 'extra', {}) or {}

        # Optimization parameters
        if 'ebmo_optim_steps' in extra:
            self.ebmo_config.optim_steps = int(extra['ebmo_optim_steps'])
        if 'ebmo_optim_lr' in extra:
            self.ebmo_config.optim_lr = float(extra['ebmo_optim_lr'])
        if 'ebmo_restarts' in extra:
            self.ebmo_config.num_restarts = int(extra['ebmo_restarts'])
        if 'ebmo_temperature' in extra:
            self.ebmo_config.projection_temperature = float(extra['ebmo_temperature'])

    def _select_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self, path: str) -> None:
        """Load trained EBMO model."""
        try:
            self.network, info = load_ebmo_model(path, self.device, self.ebmo_config)
            self.network.eval()
            self._model_loaded = True
            logger.info(f"[EBMO_AI] Loaded model from {path}")
        except Exception as e:
            logger.error(f"[EBMO_AI] Failed to load model from {path}: {e}")
            # Fall back to fresh network
            self.network = EBMONetwork(self.ebmo_config)
            self.network.to(self.device)
            self.network.eval()

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select best move using energy-based optimization.

        Algorithm:
        1. Get all legal moves
        2. Encode state once
        3. Multi-restart gradient descent on action embeddings
        4. Return move with lowest final energy

        Args:
            game_state: Current game state

        Returns:
            Selected move or None if no legal moves
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            self.move_count += 1
            return valid_moves[0]

        # Check for random move (exploration during training)
        if self.should_pick_random_move():
            self.move_count += 1
            return self.get_random_element(valid_moves)

        # Check for swap opportunity
        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # Run EBMO optimization
        best_move = self._optimize_for_move(game_state, valid_moves)

        self.move_count += 1
        self._total_moves += 1

        return best_move

    def _optimize_for_move(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> Move:
        """Run gradient-based optimization to find best move.

        Args:
            game_state: Current game state
            valid_moves: List of legal moves

        Returns:
            Best move found by optimization
        """
        with torch.no_grad():
            # Encode state once
            state_embed = self.network.encode_state_from_game(
                game_state,
                self.player_number,
                self.device,
            )

            # Encode all legal moves for projection
            legal_embeddings = self._encode_legal_moves(valid_moves)

        # Multi-restart optimization
        best_move = valid_moves[0]
        best_energy = float('inf')

        for restart in range(self.ebmo_config.num_restarts):
            # Initialize from random legal move
            init_idx = self.rng.randint(0, len(valid_moves) - 1)
            init_move = valid_moves[init_idx]

            # Optimize action embedding
            optimized_embed, final_energy = self._optimize_action_embedding(
                state_embed,
                init_move,
                legal_embeddings,
            )

            # Find nearest legal move
            nearest_move = self._find_nearest_move(
                optimized_embed,
                valid_moves,
                legal_embeddings,
            )

            # Evaluate actual energy of nearest move
            with torch.no_grad():
                move_features = self.feature_extractor.extract_tensor(
                    [nearest_move],
                    self.device,
                )
                move_embed = self.network.encode_action(move_features)
                actual_energy = self.network.compute_energy(
                    state_embed.unsqueeze(0),
                    move_embed,
                ).item()

            if actual_energy < best_energy:
                best_energy = actual_energy
                best_move = nearest_move

        return best_move

    def _encode_legal_moves(self, moves: List[Move]) -> torch.Tensor:
        """Encode all legal moves to embeddings.

        Args:
            moves: List of legal moves

        Returns:
            (N, action_embed_dim) tensor
        """
        with torch.no_grad():
            return self.network.encode_moves(moves, self.device)

    def _optimize_action_embedding(
        self,
        state_embed: torch.Tensor,
        init_move: Move,
        legal_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Run gradient descent on action embedding.

        Args:
            state_embed: Encoded state (state_embed_dim,)
            init_move: Initial move for embedding initialization
            legal_embeddings: (N, action_embed_dim) legal move embeddings

        Returns:
            (optimized_embedding, final_energy) tuple
        """
        # Initialize from the initial move
        init_features = self.feature_extractor.extract_tensor([init_move], self.device)

        with torch.no_grad():
            action_embed = self.network.encode_action(init_features).squeeze(0)

        # Make embedding require gradients
        action_embed = action_embed.clone().detach().requires_grad_(True)

        # Optimizer for action embedding only
        optimizer = torch.optim.Adam([action_embed], lr=self.ebmo_config.optim_lr)

        # State embed for repeated use (no grad needed)
        state_for_energy = state_embed.detach()

        # Gradient descent loop
        for step in range(self.ebmo_config.optim_steps):
            optimizer.zero_grad()

            # Compute energy
            energy = self.network.compute_energy(
                state_for_energy.unsqueeze(0),
                action_embed.unsqueeze(0),
            )

            # Backward pass
            energy.backward()

            # Update embedding
            optimizer.step()

            # Periodically project to legal manifold
            if (step + 1) % self.ebmo_config.project_every_n_steps == 0:
                with torch.no_grad():
                    action_embed.data = self._soft_project(
                        action_embed.data,
                        legal_embeddings,
                    )

            self._total_optim_steps += 1

        # Final energy
        with torch.no_grad():
            final_energy = self.network.compute_energy(
                state_for_energy.unsqueeze(0),
                action_embed.unsqueeze(0),
            ).item()

        return action_embed.detach(), final_energy

    def _soft_project(
        self,
        action_embed: torch.Tensor,
        legal_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Soft projection toward legal move manifold.

        Uses softmax-weighted combination of nearby legal embeddings.

        Args:
            action_embed: (action_embed_dim,) current embedding
            legal_embeddings: (N, action_embed_dim) legal embeddings

        Returns:
            (action_embed_dim,) projected embedding
        """
        # Compute distances to all legal moves
        distances = torch.cdist(
            action_embed.unsqueeze(0),
            legal_embeddings,
        ).squeeze(0)  # (N,)

        # Softmax weights (lower distance = higher weight)
        temperature = self.ebmo_config.projection_temperature
        weights = F.softmax(-distances / temperature, dim=0)  # (N,)

        # Weighted combination
        projected = (weights.unsqueeze(1) * legal_embeddings).sum(dim=0)

        return projected

    def _find_nearest_move(
        self,
        action_embed: torch.Tensor,
        valid_moves: List[Move],
        legal_embeddings: torch.Tensor,
    ) -> Move:
        """Find legal move nearest to embedding.

        Args:
            action_embed: (action_embed_dim,) optimized embedding
            valid_moves: List of legal moves
            legal_embeddings: (N, action_embed_dim) corresponding embeddings

        Returns:
            Nearest legal move
        """
        with torch.no_grad():
            distances = torch.cdist(
                action_embed.unsqueeze(0),
                legal_embeddings,
            ).squeeze(0)

            nearest_idx = distances.argmin().item()

        return valid_moves[nearest_idx]

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using average energy of top moves.

        Higher energy = worse position for current player.
        We return negative energy so that higher = better (standard convention).

        Args:
            game_state: Game state to evaluate

        Returns:
            Position evaluation (higher = better for this AI)
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return -10000.0  # No moves = very bad

        with torch.no_grad():
            # Encode state
            state_embed = self.network.encode_state_from_game(
                game_state,
                self.player_number,
                self.device,
            )

            # Encode all moves
            move_embeddings = self._encode_legal_moves(valid_moves)

            # Compute energies for all moves
            state_batch = state_embed.unsqueeze(0).expand(len(valid_moves), -1)
            energies = self.network.compute_energy(state_batch, move_embeddings)

            # Return negative of minimum energy (best move's quality)
            min_energy = energies.min().item()

        return -min_energy

    def get_move_energies(
        self,
        game_state: GameState,
        moves: Optional[List[Move]] = None,
    ) -> Dict[str, float]:
        """Get energy values for moves (for debugging/analysis).

        Args:
            game_state: Current game state
            moves: Moves to evaluate (defaults to all legal moves)

        Returns:
            Dict mapping move description to energy
        """
        if moves is None:
            moves = self.get_valid_moves(game_state)

        if not moves:
            return {}

        with torch.no_grad():
            state_embed = self.network.encode_state_from_game(
                game_state,
                self.player_number,
                self.device,
            )

            move_embeddings = self._encode_legal_moves(moves)
            state_batch = state_embed.unsqueeze(0).expand(len(moves), -1)
            energies = self.network.compute_energy(state_batch, move_embeddings)

        result = {}
        for move, energy in zip(moves, energies.tolist()):
            key = f"{move.type.value}"
            if move.from_pos:
                key += f"_from_{move.from_pos.x},{move.from_pos.y}"
            if move.to:
                key += f"_to_{move.to.x},{move.to.y}"
            result[key] = energy

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get AI statistics.

        Returns:
            Dict with EBMO-specific stats
        """
        return {
            "type": "EBMO",
            "player": self.player_number,
            "difficulty": self.config.difficulty,
            "model_loaded": self._model_loaded,
            "device": str(self.device),
            "total_moves": self._total_moves,
            "total_optim_steps": self._total_optim_steps,
            "avg_steps_per_move": (
                self._total_optim_steps / max(self._total_moves, 1)
            ),
            "config": {
                "optim_steps": self.ebmo_config.optim_steps,
                "optim_lr": self.ebmo_config.optim_lr,
                "num_restarts": self.ebmo_config.num_restarts,
                "projection_temp": self.ebmo_config.projection_temperature,
            },
        }

    def reset_for_new_game(self, *, rng_seed: Optional[int] = None) -> None:
        """Reset for a new game."""
        super().reset_for_new_game(rng_seed=rng_seed)
        # Reset training feedback
        self.last_root_value = None
        self.last_root_policy = None
        # Optionally reset stats per game
        # self._total_moves = 0
        # self._total_optim_steps = 0

    def seed(self, seed_value: int) -> None:
        """Set random seed for reproducibility.

        Used by selfplay workers for deterministic game generation.

        Args:
            seed_value: Random seed value
        """
        self.rng_seed = int(seed_value)
        self.rng = random.Random(self.rng_seed)
        np.random.seed(self.rng_seed)
        # Also seed torch for any stochastic operations
        torch.manual_seed(self.rng_seed)

    def get_policy_distribution(self, game_state: GameState) -> Dict[int, float]:
        """Get policy distribution over moves for training.

        Converts energy-based scores to probability distribution.
        Lower energy = higher probability.

        Args:
            game_state: Current game state

        Returns:
            Dict mapping move indices to probabilities
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return {}

        with torch.no_grad():
            state_embed = self.network.encode_state_from_game(
                game_state,
                self.player_number,
                self.device,
            )

            move_embeddings = self._encode_legal_moves(valid_moves)
            state_batch = state_embed.unsqueeze(0).expand(len(valid_moves), -1)
            energies = self.network.compute_energy(state_batch, move_embeddings)

            # Convert to probabilities: lower energy = higher prob
            # Use temperature for sharpness control
            logits = -energies / max(self.temperature, 0.01)
            probs = F.softmax(logits, dim=0)

        # Map to move indices (simplified: use sequential indices)
        policy = {}
        for i, prob in enumerate(probs.tolist()):
            policy[i] = prob

        return policy


# =============================================================================
# Factory function for AI creation
# =============================================================================


def create_ebmo_ai(
    player_number: int,
    config: AIConfig,
    model_path: Optional[str] = None,
) -> EBMO_AI:
    """Factory function to create EBMO AI.

    Args:
        player_number: Player number (1-based)
        config: AI configuration
        model_path: Path to trained model

    Returns:
        EBMO_AI instance
    """
    return EBMO_AI(player_number, config, model_path)

EBMOAI = EBMO_AI


__all__ = [
    "EBMO_AI",
    "EBMOAI",
    "create_ebmo_ai",
    "EBMO_MODEL_PATH_ENV",
    "EBMO_DEFAULT_MODEL_PATH",
]
