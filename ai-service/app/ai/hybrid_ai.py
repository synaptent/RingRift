#!/usr/bin/env python3
"""Hybrid CNN-GNN AI player for RingRift.

This module provides a hybrid architecture AI that combines:
- CNN backbone for local pattern recognition
- GNN refinement for territory connectivity understanding

Usage:
    from app.ai.hybrid_ai import HybridAI, create_hybrid_ai

    ai = create_hybrid_ai(
        player_number=1,
        model_path="models/hybrid_hex8_2p/hybrid_policy_best.pt",
    )
    move = ai.select_move(state)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from app.models import AIConfig, BoardType
from app.game_engine import GameEngine
from app.ai.base import BaseAI
from app.ai.canonical_move_encoding import encode_move_for_board

if TYPE_CHECKING:
    from app.models import GameState, Move

logger = logging.getLogger(__name__)


class HybridAI(BaseAI):
    """Hybrid CNN-GNN AI player.

    Uses a trained hybrid policy network that combines CNN for local patterns
    with GNN for connectivity understanding.

    Inherits from BaseAI to comply with factory contract, providing:
    - RNG seeding and reproducibility
    - Rules engine integration
    - Standard AI interface (select_move, evaluate_position)
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        model_path: str | Path | None = None,
        device: str = "cpu",
        temperature: float = 1.0,  # Note: 1.0 recommended; lower values cause underflow on MOVEMENT/CHAIN_CAPTURE phases
    ):
        """Initialize Hybrid AI.

        Args:
            player_number: Player number (1-4)
            config: AI configuration
            model_path: Path to trained model checkpoint
            device: Device to use (cpu, cuda, mps)
            temperature: Softmax temperature for action selection
        """
        super().__init__(player_number, config)
        self.device = device
        self.temperature = temperature
        self.model = None
        self.action_space_size = 4132  # Default for hex8
        self.board_size = 9  # Default for hex8
        self.history_length = 4

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str | Path):
        """Load trained hybrid model."""
        from app.ai.neural_net.hybrid_cnn_gnn import HybridPolicyNet, HAS_PYG as HAS_HYBRID_PYG

        from app.utils.torch_utils import safe_load_checkpoint
        ckpt = safe_load_checkpoint(model_path, map_location=self.device)

        # Extract model config from checkpoint
        self.action_space_size = ckpt.get("action_space_size", 4132)
        self.board_size = ckpt.get("board_size", 9)

        # Infer in_channels from the first conv layer weights
        first_conv_key = "cnn.input_conv.0.weight"
        if first_conv_key in ckpt["model_state_dict"]:
            in_channels = ckpt["model_state_dict"][first_conv_key].shape[1]
        else:
            in_channels = ckpt.get("in_channels", 40)

        # Calculate base channels and history length for encoding
        # 40 channels = 10 base × 4 history (HexStateEncoder)
        # 64 channels = 16 base × 4 history (HexStateEncoderV3)
        if in_channels == 64:
            self._encoder_version = "v3"
            self._base_channels = 16
        else:
            self._encoder_version = "v2"
            self._base_channels = 10

        # Detect if checkpoint was trained with fallback GNN (without PyG)
        state_dict = ckpt["model_state_dict"]
        uses_fallback_gnn = any(k.startswith("gnn.fallback") for k in state_dict.keys())

        self.model = HybridPolicyNet(
            in_channels=in_channels,
            global_features=ckpt.get("global_features", 20),
            hidden_channels=ckpt.get("hidden_channels", 128),
            cnn_blocks=ckpt.get("cnn_blocks", 6),
            gnn_layers=ckpt.get("gnn_layers", 3),
            board_size=self.board_size,
            action_space_size=self.action_space_size,
            num_players=ckpt.get("num_players", 2),
            is_hex=ckpt.get("is_hex", True),
            dropout=ckpt.get("dropout", 0.0),
        )

        # If checkpoint uses fallback but we have PyG, force fallback mode
        if uses_fallback_gnn and HAS_HYBRID_PYG:
            logger.info("Checkpoint uses fallback GNN, disabling PyG layers for compatibility")
            self.model.gnn.enabled = False
            self.model.gnn.fallback = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            )

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Loaded Hybrid model: val_acc={ckpt.get('val_acc', 0):.4f}, "
            f"action_space={self.action_space_size}, in_channels={in_channels}"
        )

    def _encode_state(self, state: "GameState") -> tuple[np.ndarray, np.ndarray]:
        """Encode game state to feature tensors.

        Returns:
            Tuple of (features, global_features)
        """
        # Use the correct encoder based on model's in_channels
        # 40 channels -> HexStateEncoder (10 base × 4 history)
        # 64 channels -> HexStateEncoderV3 (16 base × 4 history)
        encoder_version = getattr(self, '_encoder_version', 'v2')
        base_channels = getattr(self, '_base_channels', 10)

        if encoder_version == "v3":
            from app.training.encoding import HexStateEncoderV3
            encoder = HexStateEncoderV3(board_size=self.board_size)
        else:
            from app.training.encoding import HexStateEncoder
            encoder = HexStateEncoder(board_size=self.board_size)

        features, encoder_globals = encoder.encode_state(state)

        # Training data has 4 history frames, all with content
        # At inference we only have current state, so replicate across all frames
        C, H, W = features.shape
        total_channels = base_channels * self.history_length

        full_features = np.zeros((total_channels, H, W), dtype=np.float32)
        # Fill all 4 frame slots with current state (better than zeros)
        for frame_idx in range(self.history_length):
            start = frame_idx * base_channels
            full_features[start:start + base_channels] = features

        # Use encoder globals if available, otherwise build our own
        if encoder_globals is not None and len(encoder_globals) >= 20:
            global_features = encoder_globals[:20].astype(np.float32)
        else:
            global_features = np.zeros(20, dtype=np.float32)
            turn_number = len(state.move_history) if hasattr(state, 'move_history') else 0
            global_features[0] = turn_number / 100.0
            global_features[1] = self.player_number / 4.0
            global_features[2] = (state.current_player or 1) / 4.0

            if hasattr(state, 'players') and state.players:
                for i, player in enumerate(state.players[:4]):
                    if hasattr(player, 'rings_remaining'):
                        global_features[7 + i] = (player.rings_remaining or 0) / 10.0

        return full_features, global_features

    def select_move(self, state: "GameState") -> "Move | None":
        """Select move using hybrid policy.

        Args:
            state: Current game state

        Returns:
            Selected move or None if no legal moves
        """
        legal_moves = GameEngine.get_valid_moves(state, self.player_number)

        if not legal_moves:
            req = GameEngine.get_phase_requirement(state, self.player_number)
            if req:
                return GameEngine.synthesize_bookkeeping_move(req, state)
            return None

        if self.model is None:
            return np.random.choice(legal_moves)

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Encode state
        features, global_features = self._encode_state(state)
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        globals_t = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            policy_logits, _ = self.model(features_t, globals_t)

        # Apply temperature and get probabilities
        logits = policy_logits[0] / self.temperature
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Get probability for each legal move
        move_probs = []
        valid_moves = []
        for move in legal_moves:
            action_idx = encode_move_for_board(move, state)
            if 0 <= action_idx < len(probs):
                move_probs.append(probs[action_idx])
                valid_moves.append(move)
            else:
                move_probs.append(1e-6)
                valid_moves.append(move)

        if not valid_moves:
            return legal_moves[0]

        # Normalize and select
        weights = np.array(move_probs, dtype=np.float64)
        weights_sum = weights.sum()

        if weights_sum < 1e-10:
            logger.warning("Hybrid policy has near-zero probability, using uniform")
            idx = np.random.randint(len(valid_moves))
        else:
            weights = weights / weights_sum
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
            idx = np.random.choice(len(valid_moves), p=weights)

        return valid_moves[idx]

    def get_move(self, state: "GameState") -> "Move | None":
        """Alias for select_move."""
        return self.select_move(state)

    def get_value(self, state: "GameState") -> float:
        """Get value estimate for current state."""
        if self.model is None:
            return 0.0

        features, global_features = self._encode_state(state)
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        globals_t = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, value = self.model(features_t, globals_t)

        return value[0, self.player_number - 1].item()

    def evaluate_position(self, game_state: "GameState") -> float:
        """Evaluate the current position from this AI's perspective.

        Required by BaseAI contract. Delegates to get_value() which
        returns the neural network's value estimate.

        Args:
            game_state: Current game state

        Returns:
            Evaluation score (positive = good for this AI)
        """
        return self.get_value(game_state)


def create_hybrid_ai(
    player_number: int,
    model_path: str | Path | None = None,
    config: AIConfig | None = None,
    device: str = "cpu",
    **kwargs,
) -> HybridAI:
    """Factory function to create Hybrid AI.

    Args:
        player_number: Player number (1-4)
        model_path: Path to trained model (default: best hex8 model)
        config: AI configuration
        device: Device to use
        **kwargs: Additional HybridAI parameters

    Returns:
        Configured HybridAI instance
    """
    if config is None:
        config = AIConfig(difficulty=6)

    if model_path is None:
        default_path = Path("models/hybrid_hex8_2p/hybrid_policy_best.pt")
        if default_path.exists():
            model_path = default_path
        else:
            logger.warning("No Hybrid model found, using untrained network")

    return HybridAI(
        player_number=player_number,
        config=config,
        model_path=model_path,
        device=device,
        **kwargs,
    )


__all__ = ["HybridAI", "create_hybrid_ai"]
