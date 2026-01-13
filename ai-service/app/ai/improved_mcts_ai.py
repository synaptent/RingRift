"""Improved MCTS AI for RingRift.

This module provides an AI player that uses the advanced ImprovedMCTS algorithm
featuring PUCT exploration, progressive widening, virtual loss for parallelization,
transposition tables, and tree reuse between moves.

Usage:
    from app.ai.improved_mcts_ai import ImprovedMCTSAI
    from app.models import AIConfig

    config = AIConfig(difficulty=8, think_time=10000)
    ai = ImprovedMCTSAI(player_number=1, config=config)
    move = ai.select_move(game_state)

The AI can be created via the factory:
    ai = AIFactory.create(AIType.IMPROVED_MCTS, player_number=1, config=config)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.ai.base import BaseAI
from app.mcts import GameState as MCTSGameState
from app.mcts import ImprovedMCTS, MCTSConfig, NeuralNetworkInterface
from app.models import AIConfig, GameState, Move

if TYPE_CHECKING:
    from app.ai.nnue_policy import RingRiftNNUEWithPolicy

logger = logging.getLogger(__name__)


class RingRiftGameStateAdapter(MCTSGameState):
    """Adapter that makes RingRift GameState compatible with ImprovedMCTS.

    This adapter wraps a RingRift GameState and provides the interface
    required by ImprovedMCTS (get_legal_moves, apply_move, is_terminal, etc.).
    """

    def __init__(
        self,
        game_state: GameState,
        rules_engine: Any,
        player_number: int,
        move_to_index: dict[str, int],
        index_to_move: dict[int, Move],
    ):
        """Initialize the adapter.

        Args:
            game_state: The RingRift game state to wrap
            rules_engine: The rules engine for move generation
            player_number: The current AI's player number
            move_to_index: Mapping from move keys to action indices
            index_to_move: Mapping from action indices back to moves
        """
        self._game_state = game_state
        self._rules_engine = rules_engine
        self._player_number = player_number
        self._move_to_index = move_to_index
        self._index_to_move = index_to_move
        self._cached_legal_moves: list[int] | None = None
        self._cached_hash: str | None = None

    @property
    def game_state(self) -> GameState:
        """Access the underlying RingRift game state."""
        return self._game_state

    def get_legal_moves(self) -> list[int]:
        """Get list of legal moves as action indices."""
        if self._cached_legal_moves is not None:
            return self._cached_legal_moves

        current_player = self._game_state.current_player_number
        moves = self._rules_engine.get_valid_moves(self._game_state, current_player)

        action_indices = []
        for move in moves:
            key = self._move_key(move)
            if key in self._move_to_index:
                action_indices.append(self._move_to_index[key])
            else:
                # Assign new index for unseen move
                idx = len(self._move_to_index)
                self._move_to_index[key] = idx
                self._index_to_move[idx] = move
                action_indices.append(idx)

        self._cached_legal_moves = action_indices
        return action_indices

    def apply_move(self, move: int) -> RingRiftGameStateAdapter:
        """Apply move and return new state."""
        if move not in self._index_to_move:
            raise ValueError(f"Unknown move index: {move}")

        actual_move = self._index_to_move[move]
        new_state = self._rules_engine.apply_move(self._game_state, actual_move)

        return RingRiftGameStateAdapter(
            new_state,
            self._rules_engine,
            self._player_number,
            self._move_to_index,
            self._index_to_move,
        )

    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        return self._game_state.status.value != "in_progress"

    def get_outcome(self, player: int) -> float:
        """Get outcome for player (-1, 0, or 1)."""
        if not self.is_terminal():
            return 0.0

        # Check winner
        winner = getattr(self._game_state, "winner", None)
        if winner is None:
            return 0.0  # Draw

        if winner == self._player_number:
            return 1.0
        else:
            return -1.0

    def current_player(self) -> int:
        """Get current player (0-indexed for MCTS)."""
        return self._game_state.current_player_number - 1

    def hash(self) -> str:
        """Get unique hash for state."""
        if self._cached_hash is not None:
            return self._cached_hash

        # Use board hash plus player turn
        board = self._game_state.board
        key_parts = [
            f"t{self._game_state.current_player_number}",
            f"p{self._game_state.phase.value}",
        ]

        # Add stack positions
        for pos_key, stack in sorted(board.stacks.items()):
            if stack.rings:
                rings_str = ",".join(str(r.owner) for r in stack.rings)
                key_parts.append(f"{pos_key}:{rings_str}")

        self._cached_hash = "|".join(key_parts)
        return self._cached_hash

    def _move_key(self, move: Move) -> str:
        """Generate a unique key for a move."""
        parts = [move.type.value]
        if move.from_pos:
            parts.append(f"f{move.from_pos.to_key()}")
        if move.to:
            parts.append(f"t{move.to.to_key()}")
        if move.ring_index is not None:
            parts.append(f"r{move.ring_index}")
        return ":".join(parts)


class RingRiftNNAdapter(NeuralNetworkInterface):
    """Adapter that makes RingRift neural networks compatible with ImprovedMCTS.

    Wraps a RingRiftNNUEWithPolicy model and provides the evaluate() interface
    required by ImprovedMCTS.
    """

    def __init__(
        self,
        model: RingRiftNNUEWithPolicy | None,
        move_to_index: dict[str, int],
        index_to_move: dict[int, Move],
        rules_engine: Any,
        fallback_evaluator: BaseAI | None = None,
        model_path: str | None = None,
        board_type: Any = None,
    ):
        """Initialize the adapter.

        Args:
            model: Optional neural network model
            move_to_index: Mapping from move keys to action indices
            index_to_move: Mapping from action indices back to moves
            rules_engine: Rules engine for fallback evaluation
            fallback_evaluator: Heuristic evaluator for when NN is unavailable
            model_path: Path to model checkpoint (for encoder auto-detection)
            board_type: Board type for encoder selection
        """
        self._model = model
        self._move_to_index = move_to_index
        self._index_to_move = index_to_move
        self._rules_engine = rules_engine
        self._fallback = fallback_evaluator
        self._model_path = model_path
        self._board_type = board_type
        self._encoder = None  # Lazy-loaded encoder

    def evaluate(self, state: MCTSGameState) -> tuple[list[float], float]:
        """Evaluate state with neural network.

        Returns:
            (policy, value) where policy is list of move probabilities
        """
        if not isinstance(state, RingRiftGameStateAdapter):
            raise TypeError("Expected RingRiftGameStateAdapter")

        game_state = state.game_state
        legal_moves = state.get_legal_moves()

        if not legal_moves:
            return [], 0.0

        # Try neural network evaluation
        if self._model is not None:
            try:
                policy, value = self._nn_evaluate(game_state, legal_moves)
                return policy, value
            except Exception as e:
                logger.debug(f"NN evaluation failed, using fallback: {e}")

        # Fallback to heuristic
        return self._heuristic_evaluate(game_state, legal_moves, state)

    def _nn_evaluate(
        self, game_state: GameState, legal_moves: list[int]
    ) -> tuple[list[float], float]:
        """Evaluate using neural network."""
        import torch

        # Lazy-load encoder based on model metadata (January 2026 fix)
        if self._encoder is None:
            if self._model_path:
                from app.training.encoder_registry import get_encoder_for_model
                self._encoder = get_encoder_for_model(
                    self._model_path,
                    board_type=self._board_type,
                )
            else:
                # Fallback to board-type-based encoder if no model path
                from app.training.encoding import get_encoder_for_board_type
                self._encoder = get_encoder_for_board_type(
                    self._board_type or "hex8",
                    version="v2",
                    feature_version=1,
                )

        features = self._encoder.encode_state(game_state)

        # Run inference
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = self._model(x)

            # Convert to numpy
            policy_probs = torch.softmax(policy_logits, dim=-1).squeeze().numpy()
            value_scalar = float(value.squeeze())

        # Map to action indices
        max_idx = max(legal_moves) + 1 if legal_moves else 0
        policy = [0.0] * max_idx

        # Distribute probability to legal moves
        total = 0.0
        for idx in legal_moves:
            if idx < len(policy_probs):
                policy[idx] = float(policy_probs[idx])
                total += policy[idx]

        # Normalize
        if total > 0:
            for idx in legal_moves:
                policy[idx] /= total
        else:
            # Uniform distribution
            uniform = 1.0 / len(legal_moves)
            for idx in legal_moves:
                policy[idx] = uniform

        return policy, value_scalar

    def _heuristic_evaluate(
        self,
        game_state: GameState,
        legal_moves: list[int],
        state: RingRiftGameStateAdapter,
    ) -> tuple[list[float], float]:
        """Evaluate using heuristic fallback."""
        # Uniform policy over legal moves
        max_idx = max(legal_moves) + 1 if legal_moves else 0
        policy = [0.0] * max_idx

        uniform = 1.0 / len(legal_moves) if legal_moves else 0.0
        for idx in legal_moves:
            policy[idx] = uniform

        # Get value from fallback evaluator
        value = 0.0
        if self._fallback is not None:
            try:
                raw_value = self._fallback.evaluate_position(game_state)
                # Normalize to [-1, 1] range
                value = max(-1.0, min(1.0, raw_value / 1000.0))
            except (ValueError, TypeError):
                pass

        return policy, value


class ImprovedMCTSAI(BaseAI):
    """AI player using advanced MCTS with PUCT, progressive widening, etc.

    Features:
    - PUCT exploration (AlphaZero-style)
    - Progressive widening for large action spaces
    - Virtual loss for parallel search
    - Transposition tables for position caching
    - Tree reuse between consecutive moves
    - Dirichlet noise at root for exploration

    Configuration via AIConfig:
    - difficulty: Controls search depth and simulation count
    - think_time: Maximum search time in milliseconds
    - use_neural_net: Whether to use neural network evaluation
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        *,
        mcts_config: MCTSConfig | None = None,
    ):
        """Initialize the Improved MCTS AI.

        Args:
            player_number: The player number this AI controls (1-indexed)
            config: AI configuration
            mcts_config: Optional custom MCTS configuration
        """
        super().__init__(player_number, config)

        # Action space mappings (populated during search)
        self._move_to_index: dict[str, int] = {}
        self._index_to_move: dict[int, Move] = {}

        # Configure MCTS based on difficulty/think_time
        if mcts_config is None:
            mcts_config = self._default_mcts_config()

        self._mcts_config = mcts_config
        self._mcts: ImprovedMCTS | None = None
        self._nn_adapter: RingRiftNNAdapter | None = None
        self._fallback_evaluator: BaseAI | None = None
        self._nn_model_path: str | None = None  # Path to loaded NN model
        self._nn_board_type: Any = None  # Board type for encoder selection

        logger.info(
            f"ImprovedMCTSAI initialized: player={player_number}, "
            f"sims={mcts_config.num_simulations}, cpuct={mcts_config.cpuct}"
        )

    def _default_mcts_config(self) -> MCTSConfig:
        """Create default MCTS config based on AI config."""
        difficulty = self.config.difficulty
        think_time = self.config.think_time or 5000

        # Scale simulations with difficulty and think time
        base_sims = 50 * difficulty
        time_factor = think_time / 5000.0
        num_simulations = int(base_sims * time_factor)
        num_simulations = max(100, min(2000, num_simulations))

        # Adjust exploration based on difficulty
        cpuct = 1.5 if difficulty <= 5 else 1.25

        return MCTSConfig(
            num_simulations=num_simulations,
            cpuct=cpuct,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25 if difficulty <= 6 else 0.1,
            use_progressive_widening=True,
            pw_alpha=0.5,
            pw_beta=0.5,
            use_transposition_table=True,
            tt_max_size=50000,
            tree_reuse=True,
            fpu_reduction=0.25,
        )

    def _get_or_create_mcts(self) -> ImprovedMCTS:
        """Get or create the MCTS instance with proper network adapter."""
        if self._mcts is not None:
            return self._mcts

        # Create fallback evaluator
        if self._fallback_evaluator is None:
            from app.ai.heuristic_ai import HeuristicAI

            self._fallback_evaluator = HeuristicAI(self.player_number, self.config)

        # Try to load neural network if enabled
        nn_model = None
        if self.config.use_neural_net:
            nn_model = self._load_neural_network()

        # Create network adapter with encoder info (January 2026 fix)
        self._nn_adapter = RingRiftNNAdapter(
            model=nn_model,
            move_to_index=self._move_to_index,
            index_to_move=self._index_to_move,
            rules_engine=self.rules_engine,
            fallback_evaluator=self._fallback_evaluator,
            model_path=self._nn_model_path,
            board_type=self._nn_board_type,
        )

        self._mcts = ImprovedMCTS(self._nn_adapter, self._mcts_config)
        return self._mcts

    def _load_neural_network(self) -> RingRiftNNUEWithPolicy | None:
        """Attempt to load the neural network model."""
        try:
            import os

            from app.ai.nnue_policy import RingRiftNNUEWithPolicy
            from app.models import BoardType
            from app.utils.torch_utils import safe_load_checkpoint

            # Default to square8 2P for now
            board_type = BoardType.SQUARE8
            num_players = 2

            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "nnue",
                f"nnue_policy_{board_type.value}_{num_players}p.pt",
            )
            model_path = os.path.normpath(model_path)

            if os.path.exists(model_path):
                checkpoint = safe_load_checkpoint(model_path, map_location="cpu")

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    hidden_dim = checkpoint.get("hidden_dim", 128)
                    num_hidden_layers = checkpoint.get("num_hidden_layers", 2)
                else:
                    state_dict = checkpoint
                    hidden_dim = 128
                    num_hidden_layers = 2

                model = RingRiftNNUEWithPolicy(
                    board_type=board_type,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                model.load_state_dict(state_dict)
                model.eval()

                # Store model path and board type for encoder selection (January 2026 fix)
                self._nn_model_path = model_path
                self._nn_board_type = board_type

                logger.info("Loaded NNUE policy model for ImprovedMCTSAI")
                return model

        except Exception as e:
            logger.warning(f"Failed to load neural network: {e}")

        return None

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using MCTS search.

        Args:
            game_state: Current game state

        Returns:
            The selected move, or None if no valid moves
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            self.move_count += 1
            return valid_moves[0]

        # Check for swap decision
        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # Random move check
        if self.should_pick_random_move():
            self.move_count += 1
            return self.get_random_element(valid_moves)

        # Run MCTS search
        mcts = self._get_or_create_mcts()

        # Create state adapter
        state_adapter = RingRiftGameStateAdapter(
            game_state,
            self.rules_engine,
            self.player_number,
            self._move_to_index,
            self._index_to_move,
        )

        # Ensure legal moves are populated
        state_adapter.get_legal_moves()

        # Run search
        best_action = mcts.search(state_adapter, add_noise=True)

        if best_action is not None and best_action in self._index_to_move:
            self.move_count += 1
            return self._index_to_move[best_action]

        # Fallback to first valid move
        logger.warning("MCTS returned no valid action, using first legal move")
        self.move_count += 1
        return valid_moves[0]

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using the neural network or heuristic fallback.

        Args:
            game_state: Current game state

        Returns:
            Evaluation score (positive = good for this AI)
        """
        if self._fallback_evaluator is None:
            from app.ai.heuristic_ai import HeuristicAI

            self._fallback_evaluator = HeuristicAI(self.player_number, self.config)

        return self._fallback_evaluator.evaluate_position(game_state)

    def get_search_statistics(self) -> dict[str, Any]:
        """Get statistics from the last MCTS search.

        Returns:
            Dict with root_visits, root_value, top_moves, etc.
        """
        if self._mcts is None:
            return {}
        return self._mcts.get_search_statistics()

    def clear_search_tree(self) -> None:
        """Clear the MCTS search tree for a new game."""
        if self._mcts is not None:
            self._mcts.root = None
            if self._mcts.transposition_table:
                self._mcts.transposition_table.clear()

        # Clear action mappings
        self._move_to_index.clear()
        self._index_to_move.clear()

    def reset_for_new_game(self, *, rng_seed: int | None = None) -> None:
        """Reset for a new game, clearing search tree."""
        super().reset_for_new_game(rng_seed=rng_seed)
        self.clear_search_tree()
