"""Universal AI Wrapper for RingRift.

This module provides a wrapper that can play games using ANY model architecture
(NNUE, CNN, Hex, experimental). The wrapper automatically detects the model type
and uses the appropriate play strategy.

Usage:
    from app.ai.unified_loader import UnifiedModelLoader
    from app.ai.universal_ai import UniversalAI

    loader = UnifiedModelLoader()
    loaded = loader.load("models/my_model.pth", BoardType.SQUARE8)
    ai = UniversalAI(player_number=1, config=config, loaded_model=loaded)
    move = ai.select_move(game_state)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from app.ai.base import BaseAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.unified_loader import (
    LoadedModel,
    ModelArchitecture,
    UnifiedModelLoader,
)
from app.models import AIConfig, BoardType, GameState, Move

if TYPE_CHECKING:
    from app.ai.mutable import MutableGameState

logger = logging.getLogger(__name__)


class UniversalAI(BaseAI):
    """Universal AI wrapper that can play games using ANY model architecture.

    Wraps any loaded model (NNUE, CNN, Hex) with the appropriate play strategy
    based on the model's output signature:

    - Value-only models (NNUE): Use MinimaxAI with neural evaluation
    - Policy+Value models: Use direct policy selection or MCTS

    The wrapper automatically handles:
    - Feature extraction for different architectures
    - Move encoding/decoding for policy heads
    - Perspective rotation for correct player evaluation
    - Fallback to heuristic on any model failure
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        loaded_model: LoadedModel | None = None,
        checkpoint_path: str | Path | None = None,
        board_type: BoardType | None = None,
        num_players: int = 2,
        use_mcts: bool = False,
        mcts_simulations: int = 800,  # AlphaZero-quality (was 100)
        policy_temperature: float = 0.1,
    ):
        """Initialize the UniversalAI.

        Args:
            player_number: Player number this AI controls (1-based).
            config: AI configuration.
            loaded_model: Pre-loaded model from UnifiedModelLoader.
            checkpoint_path: Path to load model from (if loaded_model is None).
            board_type: Board type for the game.
            num_players: Number of players in the game.
            use_mcts: Whether to use MCTS for policy models.
            mcts_simulations: Number of MCTS simulations (if use_mcts=True).
            policy_temperature: Temperature for policy sampling.
        """
        super().__init__(player_number, config)

        self.board_type = board_type or config.board_type or BoardType.SQUARE8
        self.num_players = num_players
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations
        self.policy_temperature = policy_temperature

        # Load model if not provided
        if loaded_model is None and checkpoint_path is not None:
            loader = UnifiedModelLoader()
            loaded_model = loader.load(
                checkpoint_path,
                board_type=self.board_type,
                num_players=num_players,
                allow_fresh=True,
            )

        self.loaded_model = loaded_model

        # Fallback heuristic AI (always available)
        self._fallback_ai = HeuristicAI(player_number, config)

        # Feature encoder (lazy initialized)
        self._encoder = None

    def _get_encoder(self):
        """Get the appropriate feature encoder for this board type and model architecture.

        Uses the centralized get_encoder_for_board_type factory to ensure consistent
        encoder selection across the codebase.
        """
        if self._encoder is not None:
            return self._encoder

        try:
            from app.training.encoding import get_encoder_for_board_type

            # Determine encoder version from model architecture
            arch = self.loaded_model.architecture if self.loaded_model else None
            if arch in (ModelArchitecture.HEX_V3, ModelArchitecture.HEX_V3_LITE):
                version = "v3"
            else:
                version = "v2"

            # Use centralized factory to get the correct encoder
            self._encoder = get_encoder_for_board_type(
                self.board_type,
                version=version,
                feature_version=1,
            )

            if self._encoder is None:
                raise ValueError(f"No encoder available for board type {self.board_type}")

        except Exception as e:
            logger.warning(f"Failed to get encoder: {e}, using NeuralNetAI fallback")
            # Fallback to NeuralNetAI which has built-in encoding
            from app.ai.neural_net import NeuralNetAI

            # Create a minimal NeuralNetAI just for encoding
            self._encoder = NeuralNetAI(
                self.player_number,
                self.config,
                board_type=self.board_type,
            )

        return self._encoder

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move for the current game state.

        Uses the appropriate strategy based on model architecture:
        - NNUE: Minimax search with neural evaluation
        - CNN/Hex: Direct policy or MCTS

        Falls back to heuristic AI on any failure.
        """
        if self.loaded_model is None:
            return self._fallback_ai.select_move(game_state)

        arch = self.loaded_model.architecture

        try:
            if arch == ModelArchitecture.NNUE_VALUE_ONLY:
                move = self._select_move_minimax(game_state)
            elif arch == ModelArchitecture.NNUE_WITH_POLICY:
                move = self._select_move_nnue_policy(game_state)
            else:
                # CNN/Hex models with full policy
                move = self._select_move_cnn_policy(game_state)

            if move is not None:
                self.move_count += 1
                return move

        except Exception as e:
            logger.warning(f"UniversalAI: model failed, using fallback: {e}")

        # Fallback to heuristic
        move = self._fallback_ai.select_move(game_state)
        if move is not None:
            self.move_count += 1
        return move

    def _select_move_minimax(self, game_state: GameState) -> Move | None:
        """Select move using minimax with NNUE evaluation."""
        from app.ai.minimax_ai import MinimaxAI

        # Create minimax AI with custom evaluator
        minimax = MinimaxAI(self.player_number, self.config)

        # Inject custom NNUE evaluation
        model = self.loaded_model.model
        device = self.loaded_model.device
        board_type = self.board_type

        original_evaluate = minimax._evaluate_mutable

        def custom_evaluate(state: "MutableGameState") -> float:
            try:
                from app.ai.nnue import extract_features_from_mutable

                features = extract_features_from_mutable(state, self.player_number)
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    value = model(features_tensor)
                    # Scale to centipawn range
                    return float(value.item()) * 10000
            except (RuntimeError, ValueError, TypeError, ImportError):
                return original_evaluate(state)

        minimax._evaluate_mutable = custom_evaluate

        return minimax.select_move(game_state)

    def _select_move_nnue_policy(self, game_state: GameState) -> Move | None:
        """Select move using NNUE policy head."""
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]

        model = self.loaded_model.model
        device = self.loaded_model.device

        try:
            from app.ai.nnue_policy import extract_features_for_policy

            features = extract_features_for_policy(game_state, self.board_type)
            features_tensor = torch.tensor(
                features, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                value, from_logits, to_logits = model(features_tensor)

            # Score valid moves based on from/to positions
            from_logits = from_logits.cpu().numpy()[0]
            to_logits = to_logits.cpu().numpy()[0]

            move_scores = []
            for move in valid_moves:
                # Extract from/to positions from move
                from_idx, to_idx = self._get_move_positions(move)
                if from_idx >= 0 and to_idx >= 0:
                    score = from_logits[from_idx] + to_logits[to_idx]
                else:
                    score = 0.0
                move_scores.append(score)

            # Temperature-scaled selection
            return self._sample_move(valid_moves, np.array(move_scores))

        except Exception as e:
            logger.debug(f"NNUE policy failed: {e}")
            # Fall through to minimax
            return self._select_move_minimax(game_state)

    def _select_move_cnn_policy(self, game_state: GameState) -> Move | None:
        """Select move using CNN policy head."""
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]

        model = self.loaded_model.model
        device = self.loaded_model.device

        try:
            encoder = self._get_encoder()

            # Encode state
            if hasattr(encoder, "encode"):
                features, globals_vec = encoder.encode(game_state)
            else:
                # Legacy encoder
                features = encoder.extract(game_state)
                globals_vec = np.zeros(20)

            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(
                    0
                )
                g = torch.tensor(globals_vec, dtype=torch.float32, device=device).unsqueeze(
                    0
                )

                # Handle different model signatures
                if hasattr(model, "forward"):
                    # Check if model expects globals (Hex/CNN v2+ models)
                    import inspect
                    sig = inspect.signature(model.forward)
                    params = list(sig.parameters.keys())

                    # Feb 2026: Check for v5_heavy models that expect heuristics
                    heuristics_tensor = None
                    if hasattr(model, 'num_heuristics') and getattr(model, 'num_heuristics', 0) > 0:
                        try:
                            num_h = model.num_heuristics
                            player = game_state.current_player if hasattr(game_state, 'current_player') else 1
                            if num_h >= 49:
                                from app.training.fast_heuristic_features import extract_full_heuristic_features
                                h = extract_full_heuristic_features(game_state, player, normalize=True)
                            else:
                                from app.training.fast_heuristic_features import extract_heuristic_features
                                h = extract_heuristic_features(game_state, player, normalize=True)
                            heuristics_tensor = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)
                        except Exception as e:
                            logger.debug(f"Failed to extract heuristics: {e}")

                    if len(params) >= 2 and params[1] in ('globals', 'globals_vec', 'g', 'globals_'):
                        # Model expects globals as second argument
                        if heuristics_tensor is not None and 'heuristics' in params:
                            output = model(x, g, heuristics_tensor)
                        else:
                            output = model(x, g)
                    else:
                        # Legacy model without globals
                        output = model(x)

                    if isinstance(output, tuple) and len(output) >= 2:
                        value, policy = output[0], output[1]
                    else:
                        # Model only returns value
                        return self._fallback_ai.select_move(game_state)
                else:
                    return self._fallback_ai.select_move(game_state)

                policy = policy.cpu().numpy()[0]

            # Score valid moves
            move_scores = []
            for move in valid_moves:
                idx = self._encode_move(move, encoder)
                if 0 <= idx < len(policy):
                    move_scores.append(policy[idx])
                else:
                    move_scores.append(-1e10)

            return self._sample_move(valid_moves, np.array(move_scores))

        except Exception as e:
            logger.debug(f"CNN policy failed: {e}")
            return self._fallback_ai.select_move(game_state)

    def _sample_move(
        self, moves: list[Move], scores: np.ndarray
    ) -> Move | None:
        """Sample a move based on scores with temperature."""
        if len(moves) == 0:
            return None

        if self.policy_temperature <= 0.01:
            # Greedy
            best_idx = int(np.argmax(scores))
        else:
            # Softmax sampling
            scores = scores / self.policy_temperature
            scores = scores - np.max(scores)  # Stability
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)

            # Handle NaN
            if np.any(np.isnan(probs)):
                probs = np.ones(len(moves)) / len(moves)

            best_idx = np.random.choice(len(moves), p=probs)

        return moves[best_idx]

    def _get_move_positions(self, move: Move) -> tuple[int, int]:
        """Extract from/to board indices from a move."""
        # This depends on move type - simplified version
        try:
            if hasattr(move, "from_pos") and hasattr(move, "to_pos"):
                from_idx = self._pos_to_idx(move.from_pos)
                to_idx = self._pos_to_idx(move.to_pos)
                return from_idx, to_idx
            elif hasattr(move, "position"):
                idx = self._pos_to_idx(move.position)
                return idx, idx
        except AttributeError:
            pass
        return -1, -1

    def _pos_to_idx(self, pos) -> int:
        """Convert position to board index."""
        if hasattr(pos, "row") and hasattr(pos, "col"):
            board_size = 8 if self.board_type == BoardType.SQUARE8 else 19
            return pos.row * board_size + pos.col
        return -1

    def _encode_move(self, move: Move, encoder) -> int:
        """Encode a move to policy index."""
        try:
            if hasattr(encoder, "encode_move"):
                return encoder.encode_move(move)
            elif hasattr(encoder, "move_to_index"):
                return encoder.move_to_index(move)
        except AttributeError:
            pass
        return -1

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using the loaded model."""
        if self.loaded_model is None:
            return self._fallback_ai.evaluate_position(game_state)

        try:
            model = self.loaded_model.model
            device = self.loaded_model.device
            arch = self.loaded_model.architecture

            if arch in (
                ModelArchitecture.NNUE_VALUE_ONLY,
                ModelArchitecture.NNUE_WITH_POLICY,
            ):
                from app.ai.nnue import extract_features_from_gamestate

                features = extract_features_from_gamestate(
                    game_state, self.board_type
                )
                features_tensor = torch.tensor(
                    features, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    if arch == ModelArchitecture.NNUE_WITH_POLICY:
                        value, _, _ = model(features_tensor)
                    else:
                        value = model(features_tensor)
                    return float(value.item())
            else:
                # CNN models
                encoder = self._get_encoder()
                if hasattr(encoder, "encode"):
                    features, _ = encoder.encode(game_state)
                else:
                    features = encoder.extract(game_state)

                x = torch.tensor(
                    features, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    output = model(x)
                    if isinstance(output, tuple):
                        value = output[0]
                    else:
                        value = output

                    # Handle multiplayer value
                    if value.shape[-1] > 1:
                        player_idx = self.player_number - 1
                        return float(value[0, player_idx].item())
                    return float(value[0, 0].item())

        except Exception as e:
            logger.debug(f"Evaluation failed: {e}")

        return self._fallback_ai.evaluate_position(game_state)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        player_number: int,
        board_type: BoardType,
        num_players: int = 2,
        **kwargs,
    ) -> "UniversalAI":
        """Factory method to create UniversalAI from a checkpoint path."""
        loader = UnifiedModelLoader()
        loaded = loader.load(checkpoint_path, board_type, num_players)

        config = AIConfig(
            difficulty=5,
            board_type=board_type,
        )

        return cls(
            player_number=player_number,
            config=config,
            loaded_model=loaded,
            board_type=board_type,
            num_players=num_players,
            **kwargs,
        )

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        player_number: int,
        board_type: BoardType | None = None,
        num_players: int = 2,
        **kwargs,
    ) -> "UniversalAI":
        """Create UniversalAI from a model path, inferring config from checkpoint."""
        return cls.from_checkpoint(
            model_path, player_number, board_type or BoardType.SQUARE8, num_players, **kwargs
        )
