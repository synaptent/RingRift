"""CAGE AI Agent: Constraint-Aware Graph Energy-Based Move Optimization.

A novel AI that uses:
1. Graph neural networks for board representation
2. Energy-based move optimization with legality constraints
3. Primal-dual optimization to stay on legal move manifold

Usage:
    from app.ai.cage_ai import CAGE_AI
    from app.models import AIConfig

    config = AIConfig(difficulty=5)
    ai = CAGE_AI(player_number=1, config=config)
    move = ai.select_move(game_state)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..models import AIConfig, GameState, Move
from .base import BaseAI
from .cage_network import (
    CAGEConfig,
    CAGENetwork,
    board_to_graph,
)
from .ebmo_network import ActionFeatureExtractor

logger = logging.getLogger(__name__)

CAGE_MODEL_PATH_ENV = "RINGRIFT_CAGE_MODEL_PATH"
CAGE_DEFAULT_MODEL_PATH = "models/cage/cage_best.pt"


class CAGE_AI(BaseAI):
    """Constraint-Aware Graph Energy-Based Move Optimization AI.

    Uses graph neural networks and primal-dual optimization to find
    moves that minimize energy while respecting legality constraints.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        model_path: str | None = None,
        cage_config: CAGEConfig | None = None,
    ) -> None:
        super().__init__(player_number, config)

        self.cage_config = cage_config or CAGEConfig()
        self.device = self._select_device()

        # Load or create network
        self.network: CAGENetwork | None = None
        self._model_loaded = False

        if model_path:
            self._load_model(model_path)
        else:
            env_path = os.environ.get(CAGE_MODEL_PATH_ENV)
            if env_path and Path(env_path).exists():
                self._load_model(env_path)
            elif Path(CAGE_DEFAULT_MODEL_PATH).exists():
                self._load_model(CAGE_DEFAULT_MODEL_PATH)
            else:
                logger.warning(
                    "[CAGE_AI] No trained model found, using fresh weights."
                )
                self.network = CAGENetwork(self.cage_config)
                self.network.to(self.device)
                self.network.eval()

        # Action feature extractor
        self.feature_extractor = ActionFeatureExtractor(self.cage_config.board_size)

        # Statistics
        self._total_moves = 0

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self, path: str) -> None:
        from app.utils.torch_utils import safe_load_checkpoint
        try:
            checkpoint = safe_load_checkpoint(path, map_location=str(self.device), warn_on_unsafe=False)
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                if isinstance(cfg, dict):
                    self.cage_config = CAGEConfig(**{k: v for k, v in cfg.items() if hasattr(CAGEConfig, k)})
                else:
                    self.cage_config = cfg
            self.network = CAGENetwork(self.cage_config)
            self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.network.to(self.device)
            self.network.eval()
            self._model_loaded = True
            logger.info(f"[CAGE_AI] Loaded model from {path}")
        except Exception as e:
            logger.error(f"[CAGE_AI] Failed to load model: {e}")
            self.network = CAGENetwork(self.cage_config)
            self.network.to(self.device)
            self.network.eval()

    def select_move(self, game_state: GameState) -> Move | None:
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            self.move_count += 1
            return valid_moves[0]

        if self.should_pick_random_move():
            self.move_count += 1
            return self.get_random_element(valid_moves)

        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # Run CAGE optimization
        best_move = self._optimize_for_move(game_state, valid_moves)

        self.move_count += 1
        self._total_moves += 1

        return best_move

    def _optimize_for_move(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> Move:
        """Find best move using graph-based energy optimization."""
        with torch.no_grad():
            # Convert board to graph
            node_feat, edge_index, edge_attr = board_to_graph(
                game_state, self.player_number, self.cage_config.board_size
            )
            node_feat = node_feat.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)

            # Encode graph
            _, graph_embed = self.network.encode_graph(
                node_feat, edge_index, edge_attr
            )
            graph_embed = graph_embed.squeeze(0)

            # Encode all legal moves
            action_features = self.feature_extractor.extract_tensor(
                valid_moves, self.device
            )
            action_embeds = self.network.encode_action(action_features)

        # Run primal-dual optimization
        best_idx, _best_energy = self.network.primal_dual_optimize(
            graph_embed, action_embeds, self.cage_config.optim_steps
        )

        return valid_moves[best_idx]

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using minimum energy of best move."""
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return -10000.0

        with torch.no_grad():
            node_feat, edge_index, edge_attr = board_to_graph(
                game_state, self.player_number, self.cage_config.board_size
            )
            node_feat = node_feat.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)

            _, graph_embed = self.network.encode_graph(
                node_feat, edge_index, edge_attr
            )

            action_features = self.feature_extractor.extract_tensor(
                valid_moves, self.device
            )
            action_embeds = self.network.encode_action(action_features)

            graph_batch = graph_embed.expand(len(valid_moves), -1)
            energies = self.network.compute_energy(graph_batch, action_embeds)

            min_energy = energies.min().item()

        return -min_energy

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "CAGE",
            "player": self.player_number,
            "difficulty": self.config.difficulty,
            "model_loaded": self._model_loaded,
            "device": str(self.device),
            "total_moves": self._total_moves,
        }


def create_cage_ai(
    player_number: int,
    config: AIConfig,
    model_path: str | None = None,
) -> CAGE_AI:
    return CAGE_AI(player_number, config, model_path)


CAGEAI = CAGE_AI

__all__ = [
    "CAGEAI",
    "CAGE_AI",
    "create_cage_ai",
]
