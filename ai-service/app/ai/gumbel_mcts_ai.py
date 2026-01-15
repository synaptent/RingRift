"""Gumbel MCTS AI implementation for RingRift.

.. note::
    Consider using `GumbelSearchEngine` from `app.ai.gumbel_search_engine`
    for new code. It provides a unified interface to all Gumbel variants.

    Example:
        from app.ai.gumbel_search_engine import GumbelSearchEngine
        engine = GumbelSearchEngine.for_play(neural_net)
        move = engine.search(game_state)

This module implements Gumbel AlphaZero-style MCTS with Sequential Halving
for more sample-efficient search compared to standard MCTS.

Key innovations from the Gumbel AlphaZero paper:
1. **Gumbel-Top-K sampling**: Sample k actions without replacement at the root
   using Gumbel noise added to policy logits. This focuses computation on the
   most promising actions while maintaining exploration.

2. **Sequential Halving**: Divide the simulation budget across log2(k) phases,
   progressively halving the number of candidate actions. This is more
   efficient than uniform allocation across all actions.

3. **Completed Q-values**: Use a principled estimate of action values that
   accounts for visit count asymmetry between actions.

GPU Acceleration (default enabled):
- Batches all neural network evaluations across simulation phases for 5-50x speedup
- Instead of evaluating one state per simulation, collects all states and runs batch NN inference
- Automatic fallback to sequential evaluation if no GPU available
- Control via RINGRIFT_GPU_GUMBEL_DISABLE=1 environment variable
- Shadow validation available via RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1

References:
- Danihelka et al. "Policy improvement by planning with Gumbel" (2022)
- https://arxiv.org/abs/2104.06303
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ModelVersioningError
from ..models import AIConfig, BoardType, GameState, Move
from ..rules.mutable_state import MoveUndo, MutableGameState
from .base import BaseAI
from .heuristic_ai import HeuristicAI
from .neural_net import INVALID_MOVE_INDEX, NeuralNetAI


def _infer_num_players(game_state: GameState) -> int:
    """Infer number of active players from game state."""
    if hasattr(game_state, 'num_players'):
        return int(game_state.num_players)
    # Fallback: count players in markers/stacks
    if hasattr(game_state, 'board') and game_state.board:
        players = set()
        for m in getattr(game_state.board, 'markers', {}).values():
            if hasattr(m, 'player'):
                players.add(m.player)
        return max(2, len(players))
    return 2


if TYPE_CHECKING:
    import torch
    from .tensor_gumbel_tree import GPUGumbelMCTS

logger = logging.getLogger(__name__)

# Environment variable controls for GPU acceleration
_GPU_GUMBEL_DISABLE = os.environ.get("RINGRIFT_GPU_GUMBEL_DISABLE", "").lower() in (
    "1", "true", "yes", "on"
)
_GPU_GUMBEL_SHADOW_VALIDATE = os.environ.get("RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE", "").lower() in (
    "1", "true", "yes", "on"
)
# GPU tree shadow validation rate (0.0 = disabled, 0.05 = 5% of searches validated)
_GPU_TREE_SHADOW_RATE = float(os.environ.get("RINGRIFT_GPU_TREE_SHADOW_RATE", "0.0"))

# Normalization scale for heuristic scores to [-1, 1]
# HeuristicAI returns scores in range [-100000, 100000] for terminal states
# and typically [-1000, 1000] for non-terminal positions
_HEURISTIC_NORMALIZATION_SCALE = 1000.0

# Import unified data structures from gumbel_common
# Re-exported here for backward compatibility with existing imports
from .gumbel_common import (
    GumbelAction,
    GumbelNode,
    LeafEvalRequest,
    GUMBEL_DEFAULT_BUDGET,
    GUMBEL_DEFAULT_K,
    GUMBEL_DEFAULT_C_VISIT,
    GUMBEL_DEFAULT_C_PUCT,
    get_adaptive_c_visit,
    get_adaptive_c_puct,
)

# Note: GumbelAction, GumbelNode, LeafEvalRequest are now defined in gumbel_common.py
# Import them from there for new code. Imports from this module still work for compatibility.


class LeafEvaluationBuffer:
    """Collects leaf states for batched NN evaluation.

    Instead of evaluating each leaf state individually during tree simulation,
    this buffer collects states and evaluates them in a single batch, providing
    5-50x speedup on GPU.
    """

    def __init__(self, neural_net: NeuralNetAI | None, max_batch_size: int = 256):
        self.neural_net = neural_net
        self.max_batch_size = max_batch_size
        self.pending: list[LeafEvalRequest] = []

    def add(self, request: LeafEvalRequest) -> None:
        """Add a leaf state to the buffer."""
        self.pending.append(request)

    def should_flush(self) -> bool:
        """Check if buffer is full."""
        return len(self.pending) >= self.max_batch_size

    def __len__(self) -> int:
        return len(self.pending)

    def flush(self, value_head: int | None = None) -> list[tuple[int, int, float]]:
        """Evaluate all pending states in batch.

        Args:
            value_head: If not None, indicates per-player value head is being used
                       (3+ player game), so perspective flipping should be skipped.

        Returns:
            List of (action_idx, simulation_idx, value) tuples.
        """
        if not self.pending or self.neural_net is None:
            result = [(r.action_idx, r.simulation_idx, 0.0) for r in self.pending]
            self.pending.clear()
            return result

        # Extract states for batch evaluation
        states = [req.game_state for req in self.pending]

        try:
            values, _ = self.neural_net.evaluate_batch(states, value_head=value_head)
        except (RuntimeError, ValueError, AttributeError) as e:
            error_msg = str(e).lower()
            # Check if this is a CUDA error that might be recoverable
            if "cuda" in error_msg or "device" in error_msg or "gpu" in error_msg:
                try:
                    from .gpu_batch import clear_gpu_memory, recover_cuda_device
                    logger.warning(f"Batch evaluation CUDA error, attempting recovery: {e}")
                    clear_gpu_memory()
                    if recover_cuda_device():
                        # Retry after recovery
                        try:
                            values, _ = self.neural_net.evaluate_batch(states, value_head=value_head)
                        except Exception as retry_err:
                            logger.warning(f"Retry after CUDA recovery failed: {retry_err}")
                            values = [0.0] * len(states)
                    else:
                        values = [0.0] * len(states)
                except ImportError:
                    logger.warning(f"Batch evaluation failed (no recovery available): {e}")
                    values = [0.0] * len(states)
            else:
                logger.warning(f"Batch evaluation failed: {e}")
                values = [0.0] * len(states)

        # Build result with perspective flipping.
        # For 2-player games (value_head is None), flip for opponent perspective.
        # For 3+ player games (value_head is set), the value is already from
        # our perspective via the value_head selection, so no flip is needed.
        use_multiplayer_heads = value_head is not None
        results = []
        for i, req in enumerate(self.pending):
            value = float(values[i]) if i < len(values) else 0.0
            if req.is_opponent_perspective and not use_multiplayer_heads:
                value = -value
            results.append((req.action_idx, req.simulation_idx, value))

        self.pending.clear()
        return results


class GumbelMCTSAI(BaseAI):
    """Gumbel MCTS AI with Sequential Halving for sample-efficient search.

    This AI combines neural network policy/value evaluation with Gumbel-based
    action sampling and Sequential Halving for budget allocation.

    Compared to standard MCTS:
    - More sample-efficient (better use of limited simulations)
    - Focuses computation on promising actions
    - Better theoretical guarantees for action selection

    GPU Acceleration:
        When GPU is available, batches all neural network evaluations within
        each Sequential Halving phase for 5-50x speedup. The GPU path provides:

        - **Rules parity**: Game states are generated using the canonical rules
          engine (MutableGameState.from_immutable, make_move) - identical to CPU.

        - **Evaluation parity**: The same neural network evaluate_batch() is
          called with batched states vs sequential states. No approximations.

        - **Move selection parity**: Shadow validation (5% sample) confirms
          batch and sequential paths produce identical NN outputs.

        Control via environment variables:
        - RINGRIFT_GPU_GUMBEL_DISABLE=1: Force sequential (CPU) evaluation
        - RINGRIFT_GPU_GUMBEL_SHADOW_VALIDATE=1: Enable parity validation

    Attributes:
        neural_net: Neural network for policy/value evaluation.
        num_sampled_actions: Number of actions for Gumbel-Top-K (m).
        simulation_budget: Total simulation budget (n).
        c_puct: Exploration constant for tree traversal.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        board_type: BoardType = BoardType.SQUARE8,
    ) -> None:
        """Initialize Gumbel MCTS AI.

        Args:
            player_number: The player number this AI controls.
            config: AI configuration including nn_model_id and gumbel parameters.
            board_type: Board type for move encoding.

        Raises:
            RuntimeError: If neural network cannot be loaded.
        """
        super().__init__(player_number, config)

        self.board_type = board_type

        # Gumbel MCTS parameters
        self.num_sampled_actions = config.gumbel_num_sampled_actions or 16
        self.simulation_budget = config.gumbel_simulation_budget or 150
        self.c_puct = 1.5  # Base exploration constant for tree policy
        # Adaptive c_puct parameters (Dec 2025: Phase 1 quick win)
        # Early game: high exploration, Late game: exploitation
        self.c_puct_early = 2.0  # More exploration in opening
        self.c_puct_mid = 1.5    # Balanced in midgame
        self.c_puct_late = 0.8   # More exploitation in endgame
        self.c_puct_early_threshold = 15   # Moves before switching to mid
        self.c_puct_late_threshold = 40    # Moves before switching to late

        # Store search results for training data extraction
        self._last_search_actions: list[GumbelAction] | None = None
        self._last_search_stats: dict | None = None  # Rich stats from GPU tree search

        # GPU acceleration state (lazy initialized)
        self._gpu_batch_enabled: bool = not _GPU_GUMBEL_DISABLE
        self._gpu_available: bool | None = None  # None = not yet checked
        self._gpu_device: torch.device | None = None

        # Shadow validation for GPU batch vs sequential parity checking
        self._shadow_validate: bool = _GPU_GUMBEL_SHADOW_VALIDATE
        self._shadow_divergence_count: int = 0
        self._shadow_total_checks: int = 0

        # GPU tree shadow validation: compare GPU tree search vs CPU sequential halving
        # Validates training data quality by ensuring policy distributions match
        self._gpu_tree_shadow_rate: float = _GPU_TREE_SHADOW_RATE
        self._gpu_tree_divergence_count: int = 0
        self._gpu_tree_total_checks: int = 0

        # Batch size for GPU evaluation (tune based on GPU memory)
        self._gpu_batch_size: int = getattr(config, 'gpu_batch_size', 128)

        # Hybrid NN + Heuristic evaluation configuration (RR-CANON-HYBRID-001)
        self._heuristic_blend_alpha: float | None = getattr(
            config, 'heuristic_blend_alpha', None
        )
        self._heuristic_fallback_enabled: bool = getattr(
            config, 'heuristic_fallback_enabled', True
        )
        # Lazy-initialized heuristic evaluator (only created when needed)
        self._heuristic_ai: HeuristicAI | None = None

        # Self-play mode with Dirichlet noise for exploration (AlphaZero-style)
        self.self_play: bool = getattr(config, 'self_play', False)
        self.root_dirichlet_alpha: float | None = getattr(
            config, 'root_dirichlet_alpha', None
        )
        self.root_noise_fraction: float = float(
            getattr(config, 'root_noise_fraction', None) or 0.25
        )

        # Load neural network (required for Gumbel MCTS)
        self.neural_net: NeuralNetAI | None = None
        try:
            self.neural_net = NeuralNetAI(player_number, config)
            logger.info(
                f"GumbelMCTSAI(player={player_number}): loaded neural network "
                f"(model={config.nn_model_id}, m={self.num_sampled_actions}, "
                f"budget={self.simulation_budget}, gpu_batch={self._gpu_batch_enabled})"
            )
        except (RuntimeError, FileNotFoundError, OSError, ValueError) as e:
            if not config.allow_fresh_weights:
                raise RuntimeError(
                    f"GumbelMCTSAI requires a neural network but failed to load: {e}"
                ) from e
            logger.warning(f"GumbelMCTSAI: failed to load neural net ({e})")
            self.neural_net = None

        # GPU Tree-based Gumbel MCTS (experimental - for 10-20x speedup)
        # When enabled, uses fully GPU-accelerated tree search instead of
        # Python-based Sequential Halving. See tensor_gumbel_tree.py.
        self._use_gpu_tree: bool = getattr(config, 'use_gpu_tree', False)
        self._gpu_gumbel_mcts: "GPUGumbelMCTS | None" = None  # Lazy initialized

        if self._use_gpu_tree:
            logger.info(
                f"GumbelMCTSAI(player={player_number}): GPU tree mode enabled "
                f"(target 10-20x speedup)"
            )

    def _get_value_head(self, game_state: GameState) -> int | None:
        """Get the value head index for multi-player games.

        For 3+ player games, we need to use the correct value head for this AI's
        player perspective. For 2-player games, we can use the default (value[0])
        since values are symmetrically negated.

        Returns:
            Player index (0-indexed) for 3+ player games, None for 2-player games.
        """
        num_players = _infer_num_players(game_state)
        nn_supports_mp = (
            self.neural_net is not None
            and getattr(self.neural_net, "num_players", 4) >= 2
        )
        if nn_supports_mp and num_players > 2:
            return self.player_number - 1  # Convert to 0-indexed
        return None

    def _get_adaptive_budget(self, game_state: GameState) -> int:
        """Get adaptive simulation budget based on game phase.

        Early game positions are simpler and need fewer simulations.
        Mid/late game positions with more complex tactics get full budget.

        Minimum budget is 800 simulations to ensure quality search.

        Args:
            game_state: Current game state.

        Returns:
            Adaptive simulation budget (minimum 800).
        """
        base_budget = max(self.simulation_budget, 800)  # Minimum 800 sims

        # Estimate move number from game state or AI's own move count
        move_number = getattr(game_state, 'move_count', None) or self.move_count

        # Also factor in board occupancy for better phase estimation
        board_size = 64  # default
        if hasattr(game_state, 'board') and game_state.board:
            if hasattr(game_state.board, 'width') and hasattr(game_state.board, 'height'):
                board_size = game_state.board.width * game_state.board.height
            elif hasattr(game_state.board, 'cells'):
                board_size = len(game_state.board.cells)

        # Early game: fewer sims needed (positions are simpler)
        # Scale thresholds by board size (larger boards = longer games)
        early_threshold = max(5, board_size // 10)  # ~6 for 64 cells, ~36 for 361
        mid_threshold = max(15, board_size // 4)    # ~16 for 64 cells, ~90 for 361

        if move_number < early_threshold:
            # Very early game: 50% budget (minimum 800)
            return max(800, base_budget // 2)
        elif move_number < mid_threshold:
            # Early-mid game: 75% budget (minimum 800)
            return max(800, (base_budget * 3) // 4)
        else:
            # Mid-late game: full budget for complex tactics
            return base_budget

    def _ensure_heuristic_evaluator(self) -> HeuristicAI | None:
        """Lazily initialize heuristic evaluator if needed.

        Returns:
            HeuristicAI instance if hybrid evaluation or fallback is enabled,
            None otherwise.
        """
        if self._heuristic_ai is not None:
            return self._heuristic_ai

        # Only create if blending or fallback is enabled
        if self._heuristic_blend_alpha is not None or self._heuristic_fallback_enabled:
            try:
                # Use light mode for faster heuristic evaluation in hybrid mode
                from copy import copy
                heuristic_config = copy(self.config)
                heuristic_config.heuristic_eval_mode = "light"
                self._heuristic_ai = HeuristicAI(self.player_number, heuristic_config)
                logger.debug(
                    f"GumbelMCTSAI: initialized heuristic evaluator for hybrid eval "
                    f"(blend_alpha={self._heuristic_blend_alpha})"
                )
            except (RuntimeError, ValueError, AttributeError, ImportError) as e:
                logger.warning(f"GumbelMCTSAI: failed to create heuristic evaluator: {e}")
                return None

        return self._heuristic_ai

    def _default_dirichlet_alpha(self, board_type: BoardType) -> float:
        """Return a board-specific default Dirichlet alpha.

        Smaller alpha produces more peaked noise distributions, which is
        appropriate for games with more legal moves.
        """
        if board_type in (BoardType.SQUARE19,):
            return 0.15  # More moves → sharper noise
        elif board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            return 0.2
        return 0.3  # Default for square8

    def _get_adaptive_cpuct(self, move_number: int | None = None) -> float:
        """Get adaptive c_puct based on game phase.

        This is a Phase 1 quick win for 2000+ Elo (Dec 2025).
        Estimated impact: +10-20 Elo.

        Early game benefits from more exploration (higher c_puct):
        - Many opening possibilities, exploration finds good lines
        - Less risk from tactical errors

        Late game benefits from exploitation (lower c_puct):
        - Winning lines are clearer, focus search on best moves
        - Tactical precision matters more

        Args:
            move_number: Current move number in the game. If None, uses
                self.move_count as fallback.

        Returns:
            c_puct value appropriate for the game phase:
            - 2.0 for early game (moves 0-14)
            - 1.5 for midgame (moves 15-39)
            - 0.8 for late game (moves 40+)
        """
        if move_number is None:
            move_number = self.move_count

        if move_number < self.c_puct_early_threshold:
            return self.c_puct_early
        elif move_number < self.c_puct_late_threshold:
            return self.c_puct_mid
        else:
            return self.c_puct_late

    def _apply_dirichlet_noise(
        self,
        policy_logits: np.ndarray,
        board_type: BoardType,
    ) -> np.ndarray:
        """Apply Dirichlet noise to policy logits for self-play exploration.

        This adds stochastic exploration at the root during self-play,
        following AlphaZero methodology. The noise helps discover moves
        that the policy might otherwise miss.

        Args:
            policy_logits: Array of log-probabilities for each valid move.
            board_type: Board type for determining alpha parameter.

        Returns:
            Noised policy logits.
        """
        if not self.self_play or len(policy_logits) <= 1:
            return policy_logits

        if self.root_noise_fraction <= 0:
            return policy_logits

        # Determine alpha for Dirichlet distribution
        alpha = self.root_dirichlet_alpha or self._default_dirichlet_alpha(board_type)
        epsilon = self.root_noise_fraction

        # Generate Dirichlet noise (using numpy, seeded from self.rng)
        # Re-seed numpy's random from our RNG for reproducibility
        np_rng = np.random.default_rng(self.rng.randint(0, 2**32 - 1))
        noise = np_rng.dirichlet([alpha] * len(policy_logits))

        # Convert logits to probabilities, apply noise, convert back
        # Softmax: p = exp(logits) / sum(exp(logits))
        logits_shifted = policy_logits - np.max(policy_logits)  # Numerical stability
        probs = np.exp(logits_shifted)
        probs = probs / (probs.sum() + 1e-10)

        # Mix original priors with noise: p' = (1 - ε)p + εn
        noised_probs = (1 - epsilon) * probs + epsilon * noise
        noised_probs = np.clip(noised_probs, 1e-10, 1.0)
        noised_probs = noised_probs / noised_probs.sum()

        # Convert back to logits
        noised_logits = np.log(noised_probs)

        return noised_logits

    def _normalize_heuristic_score(self, raw_score: float) -> float:
        """Normalize heuristic score to [-1, 1] range using tanh.

        Args:
            raw_score: Raw heuristic score (typically in range [-1000, 1000],
                      or ±100000 for terminal states).

        Returns:
            Normalized score in [-1, 1] range.
        """
        return math.tanh(raw_score / _HEURISTIC_NORMALIZATION_SCALE)

    def _evaluate_leaf_hybrid(
        self,
        mstate: MutableGameState,
        is_opponent_perspective: bool,
    ) -> float:
        """Evaluate a leaf node using hybrid NN + heuristic blending.

        When heuristic_blend_alpha is set, combines NN value with normalized
        heuristic score. Otherwise uses pure NN with optional heuristic fallback.

        Args:
            mstate: Mutable game state at the leaf node.
            is_opponent_perspective: True if state's current player is opponent.

        Returns:
            Value estimate in [-1, 1] from the root player's perspective.
        """
        sim_state = mstate.to_immutable()
        num_players = _infer_num_players(sim_state)

        # For 3+ players with per-player value heads, we get the value directly
        # from our player's perspective, so no negation is needed.
        use_multiplayer_heads = (
            num_players > 2
            and self.neural_net is not None
            and getattr(self.neural_net, "num_players", 4) >= num_players
        )

        # Try NN evaluation first
        nn_value: float | None = None
        if self.neural_net is not None:
            try:
                value_head = self._get_value_head(sim_state)
                values, _ = self.neural_net.evaluate_batch(
                    [sim_state], value_head=value_head
                )
                if values:
                    nn_value = float(values[0])
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.debug(f"GumbelMCTSAI: NN evaluation failed: {e}")

        # Compute heuristic value if needed for blending or fallback
        heuristic_value: float | None = None
        need_heuristic = (
            self._heuristic_blend_alpha is not None or
            (nn_value is None and self._heuristic_fallback_enabled)
        )

        if need_heuristic:
            heuristic_ai = self._ensure_heuristic_evaluator()
            if heuristic_ai is not None:
                try:
                    # Evaluate from our player's perspective
                    heuristic_ai.player_number = self.player_number
                    raw_score = heuristic_ai.evaluate_position(sim_state)
                    heuristic_value = self._normalize_heuristic_score(raw_score)
                except (RuntimeError, ValueError, AttributeError) as e:
                    logger.debug(f"GumbelMCTSAI: heuristic evaluation failed: {e}")

        # Compute final value
        if nn_value is not None and heuristic_value is not None:
            # Blend NN and heuristic
            alpha = (
                self._heuristic_blend_alpha
                if self._heuristic_blend_alpha is not None
                else 1.0
            )
            value = alpha * nn_value + (1.0 - alpha) * heuristic_value
        elif nn_value is not None:
            # Pure NN
            value = nn_value
        elif heuristic_value is not None:
            # Pure heuristic fallback
            value = heuristic_value
        else:
            # No evaluation available
            value = 0.0

        # For 2-player games, flip value for opponent perspective.
        # For 3+ player games with per-player value heads, the value is already
        # from our perspective (via value_head selection), so no flip needed.
        if is_opponent_perspective and not use_multiplayer_heads:
            value = -value

        return value

    def select_move(self, game_state: GameState) -> Move | None:
        """Select best move using Gumbel MCTS with Sequential Halving.

        Args:
            game_state: Current game state.

        Returns:
            Selected move or None if no valid moves.
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

        # Run Gumbel MCTS
        best_move = self._gumbel_mcts_search(game_state, valid_moves)

        self.move_count += 1
        return best_move

    def _gumbel_mcts_search(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> Move:
        """Run Gumbel MCTS search with Sequential Halving.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves.

        Returns:
            Best move according to search.
        """
        # Use GPU tree-based search if enabled (experimental 10-20x speedup)
        if self._use_gpu_tree:
            try:
                return self._gumbel_mcts_search_gpu_tree(game_state, valid_moves)
            except (RuntimeError, ValueError, AttributeError, ModelVersioningError):
                # Use logger.exception() for safe exception logging (avoids __str__ crash)
                logger.exception("GPU tree search failed, falling back to CPU")
                # Fall through to CPU implementation

        # Get policy logits from neural network
        policy_logits = self._get_policy_logits(game_state, valid_moves)

        # Step 1: Gumbel-Top-K sampling
        actions = self._gumbel_top_k_sample(valid_moves, policy_logits)

        if len(actions) == 1:
            # Single action - store it with full visit count
            actions[0].visit_count = 1
            self._last_search_actions = actions
            return actions[0].move

        # Step 2: Sequential Halving
        best_action = self._sequential_halving(game_state, actions)

        # Store all actions with their visit counts for training data extraction
        self._last_search_actions = actions

        return best_action.move

    def _gumbel_mcts_search_gpu_tree(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> Move:
        """Run GPU tree-based Gumbel MCTS search.

        This method uses fully GPU-accelerated tensor tree operations
        for 10-20x speedup over the Python-based Sequential Halving.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves.

        Returns:
            Best move according to GPU tree search.
        """
        from .tensor_gumbel_tree import GPUGumbelMCTS, GPUGumbelMCTSConfig

        # Lazy initialize GPU Gumbel MCTS
        if self._gpu_gumbel_mcts is None:
            # Ensure GPU availability is checked before initialization
            # (self._gpu_available starts as None, which is falsy)
            self._ensure_gpu_available()

            # Get eval_mode from AIConfig, default to "hybrid" for balanced speed/quality
            eval_mode = getattr(self.config, 'gpu_tree_eval_mode', 'hybrid')

            # Use minimum budget of 64 for GPU tree search (Sequential Halving needs some budget)
            # Note: Jan 2026 - Lowered from 800 to 64 to fix selfplay stall. Budget 800 was
            # causing games to take >2 min each. Budget 150 (commonly requested) is reasonable.
            effective_budget = max(self.simulation_budget, 64)

            gpu_config = GPUGumbelMCTSConfig(
                num_sampled_actions=self.num_sampled_actions,
                simulation_budget=effective_budget,
                max_nodes=1024,
                max_actions=min(256, len(valid_moves) * 2),
                max_rollout_depth=10,
                eval_mode=eval_mode,  # "heuristic", "nn", or "hybrid"
                device="cuda" if self._gpu_available else "cpu",
            )
            self._gpu_gumbel_mcts = GPUGumbelMCTS(gpu_config)
            logger.info(
                f"GumbelMCTSAI: initialized GPU tree search "
                f"(budget={gpu_config.simulation_budget}, eval_mode={eval_mode}, "
                f"device={gpu_config.device})"
            )

        # Run GPU tree search with rich statistics for training
        best_move, policy_dict, search_stats = self._gpu_gumbel_mcts.search_with_stats(
            game_state,
            self.neural_net,
            valid_moves,
        )

        # Store rich search stats for training data extraction
        self._last_search_stats = search_stats.to_json_dict() if search_stats else None

        # GPU tree shadow validation: compare against CPU sequential halving
        if self._gpu_tree_shadow_rate > 0 and np.random.random() < self._gpu_tree_shadow_rate:
            self._gpu_tree_shadow_validate(game_state, valid_moves, best_move, policy_dict)

        # Convert policy dict to GumbelAction list for training data extraction
        # (maintains compatibility with existing training data extraction)
        self._last_search_actions = []
        for move in valid_moves:
            move_key = self._gpu_gumbel_mcts._move_to_key(move)
            prob = policy_dict.get(move_key, 0.0)
            action = GumbelAction(
                move=move,
                policy_logit=np.log(max(prob, 1e-10)),
                gumbel_noise=0.0,
                perturbed_value=0.0,
                visit_count=int(prob * self.simulation_budget),
                total_value=0.0,
            )
            self._last_search_actions.append(action)

        return best_move

    def _get_policy_logits(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> np.ndarray:
        """Get policy logits for valid moves from neural network.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves.

        Returns:
            Array of log-probabilities for each valid move.
        """
        if self.neural_net is None:
            # Uniform logits when no neural net
            return np.zeros(len(valid_moves))

        try:
            value_head = self._get_value_head(game_state)
            _, policy = self.neural_net.evaluate_batch(
                [game_state], value_head=value_head
            )

            if policy.size == 0:
                return np.zeros(len(valid_moves))

            policy_vec = policy[0]

            logits = []
            for move in valid_moves:
                idx = self.neural_net.encode_move(move, game_state.board)
                if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy_vec):
                    # Convert probability to logit (log-odds)
                    prob = max(float(policy_vec[idx]), 1e-10)
                    logit = np.log(prob)
                else:
                    logit = -10.0  # Low logit for unrecognized moves

                logits.append(logit)

            policy_logits = np.array(logits, dtype=np.float32)

            # Apply Dirichlet noise for self-play exploration at root
            policy_logits = self._apply_dirichlet_noise(
                policy_logits, self.board_type
            )

            return policy_logits

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"GumbelMCTSAI: policy evaluation failed ({e})")
            return np.zeros(len(valid_moves))

    def _gumbel_top_k_sample(
        self,
        valid_moves: list[Move],
        policy_logits: np.ndarray,
    ) -> list[GumbelAction]:
        """Sample top-k actions using Gumbel-Top-K.

        This samples k actions without replacement by adding Gumbel noise
        to the policy logits and taking the top k.

        Args:
            valid_moves: List of valid moves.
            policy_logits: Log-probabilities for each move.

        Returns:
            List of GumbelAction objects for the top-k actions.
        """
        k = min(self.num_sampled_actions, len(valid_moves))

        # Generate Gumbel(0,1) noise using inverse CDF method
        # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
        seed = int(self.rng.randrange(0, 2**32 - 1))
        np_rng = np.random.default_rng(seed)
        uniform = np_rng.uniform(1e-10, 1.0 - 1e-10, size=len(valid_moves))
        gumbel_noise = -np.log(-np.log(uniform))

        # Perturb logits with Gumbel noise
        perturbed = policy_logits + gumbel_noise

        # Select top-k indices
        top_k_indices = np.argsort(perturbed)[-k:][::-1]

        # Create GumbelAction objects
        actions = []
        for idx in top_k_indices:
            action = GumbelAction(
                move=valid_moves[idx],
                policy_logit=float(policy_logits[idx]),
                gumbel_noise=float(gumbel_noise[idx]),
                perturbed_value=float(perturbed[idx]),
            )
            actions.append(action)

        return actions

    # =========================================================================
    # GPU Batch Evaluation Methods
    # =========================================================================

    def _ensure_gpu_available(self) -> bool:
        """Lazily check if GPU is available for batch evaluation.

        Returns:
            True if GPU is available and neural network supports batch evaluation.
        """
        if self._gpu_available is not None:
            return self._gpu_available

        if not self._gpu_batch_enabled:
            self._gpu_available = False
            logger.debug("GumbelMCTSAI: GPU batching disabled via environment variable")
            return False

        if self.neural_net is None:
            self._gpu_available = False
            return False

        try:
            from .gpu_batch import get_device

            self._gpu_device = get_device(prefer_gpu=True)
            self._gpu_available = self._gpu_device.type in ('cuda', 'mps')

            if self._gpu_available:
                logger.info(
                    f"GumbelMCTSAI: GPU batch evaluation enabled on {self._gpu_device} "
                    f"(batch_size={self._gpu_batch_size})"
                )
            else:
                logger.debug(
                    f"GumbelMCTSAI: No GPU available (device={self._gpu_device.type}), "
                    "using sequential evaluation"
                )
        except (RuntimeError, ImportError, AttributeError) as e:
            logger.warning(f"GumbelMCTSAI: GPU check failed, using sequential: {e}")
            self._gpu_available = False

        return self._gpu_available

    def _simulate_actions_batched(
        self,
        game_state: GameState,
        actions: list[GumbelAction],
        sims_per_action: int,
    ) -> None:
        """Simulate all actions in a batch for GPU efficiency.

        Instead of evaluating one state at a time, this method:
        1. Collects all (action, simulation) state pairs
        2. Batch evaluates all non-terminal states via neural network
        3. Aggregates results back to each action

        This provides significant speedup on GPU by amortizing kernel launch
        overhead and maximizing memory bandwidth utilization.

        Args:
            game_state: Current game state.
            actions: List of actions to simulate.
            sims_per_action: Number of simulations per action.
        """
        if self.neural_net is None:
            # Fallback to sequential if no neural network
            for action in actions:
                value_sum = self._simulate_action(game_state, action.move, sims_per_action)
                action.visit_count += sims_per_action
                action.total_value += value_sum
            return

        # Collect all states that need neural network evaluation
        # Format: (action_idx, sim_state, needs_flip)
        evaluation_requests: list[tuple[int, GameState, bool]] = []
        terminal_values: dict[int, list[float]] = {i: [] for i in range(len(actions))}

        # Check if we're using per-player value heads (3+ players)
        num_players = _infer_num_players(game_state)
        use_multiplayer_heads = (
            num_players > 2
            and self.neural_net is not None
            and getattr(self.neural_net, "num_players", 4) >= num_players
        )

        for action_idx, action in enumerate(actions):
            mstate = MutableGameState.from_immutable(game_state)
            undo = mstate.make_move(action.move)
            sim_state = mstate.to_immutable()

            for _ in range(sims_per_action):
                if sim_state.game_status == "completed":
                    # Terminal state - compute value directly
                    winner = sim_state.winner
                    if winner == self.player_number:
                        value = 1.0
                    elif winner is None:
                        value = 0.0  # Draw
                    else:
                        value = -1.0
                    terminal_values[action_idx].append(value)
                else:
                    # Non-terminal - add to batch
                    # needs_flip only applies for 2-player games
                    needs_flip = (
                        not use_multiplayer_heads
                        and sim_state.current_player != self.player_number
                    )
                    evaluation_requests.append((action_idx, sim_state, needs_flip))

            mstate.unmake_move(undo)

        # Add terminal values to actions
        for action_idx, values in terminal_values.items():
            if values:
                actions[action_idx].visit_count += len(values)
                actions[action_idx].total_value += sum(values)

        # Batch evaluate all non-terminal states
        if evaluation_requests:
            states = [req[1] for req in evaluation_requests]

            # Split into chunks if batch is too large
            batch_size = self._gpu_batch_size
            all_values: list[float] = []

            # Get value_head from first state (all states have same num_players)
            value_head = self._get_value_head(states[0]) if states else None

            for i in range(0, len(states), batch_size):
                batch_states = states[i:i + batch_size]
                try:
                    values, _ = self.neural_net.evaluate_batch(
                        batch_states, value_head=value_head
                    )
                    all_values.extend(values if values else [0.0] * len(batch_states))
                except (RuntimeError, ValueError, AttributeError) as e:
                    error_msg = str(e).lower()
                    # Check if this is a recoverable CUDA error
                    if "cuda" in error_msg or "device" in error_msg or "gpu" in error_msg:
                        try:
                            from .gpu_batch import clear_gpu_memory, recover_cuda_device
                            logger.warning(f"GumbelMCTSAI: CUDA error, attempting recovery: {e}")
                            clear_gpu_memory()
                            if recover_cuda_device():
                                try:
                                    values, _ = self.neural_net.evaluate_batch(
                                        batch_states, value_head=value_head
                                    )
                                    all_values.extend(values if values else [0.0] * len(batch_states))
                                    continue  # Success after recovery
                                except Exception as retry_err:
                                    logger.warning(f"Retry after recovery failed: {retry_err}")
                        except ImportError:
                            pass
                    logger.warning(f"GumbelMCTSAI: Batch evaluation failed: {e}")
                    all_values.extend([0.0] * len(batch_states))

            # Shadow validation if enabled
            if self._shadow_validate and len(evaluation_requests) > 0:
                self._shadow_validate_batch(evaluation_requests, all_values)

            # Aggregate values back to actions
            # For 2-player games, flip value if it's the opponent's turn.
            # For 3+ player games with per-player value heads, no flip needed.
            for i, (action_idx, _, needs_flip) in enumerate(evaluation_requests):
                value = all_values[i] if i < len(all_values) else 0.0
                if needs_flip:
                    value = -value
                actions[action_idx].visit_count += 1
                actions[action_idx].total_value += value

    def _shadow_validate_batch(
        self,
        requests: list[tuple[int, GameState, bool]],
        batch_values: list[float],
    ) -> None:
        """Validate batch NN results against sequential evaluation (5% sample).

        This validates that batching neural network calls produces identical results
        to sequential calls. This catches:
        - Numerical precision differences between batch and sequential inference
        - Any bugs in batch aggregation logic

        Note: Rules parity is maintained by construction since both paths use
        the canonical rules engine (MutableGameState.from_immutable, make_move).
        This validation focuses on NN evaluation consistency.

        Args:
            requests: List of (action_idx, game_state, needs_flip) tuples
            batch_values: Values returned from batched NN evaluation
        """
        import random

        if self.neural_net is None:
            return

        sample_rate = 0.05  # Check 5% of batch
        for i, (action_idx, state, needs_flip) in enumerate(requests):
            if random.random() > sample_rate:
                continue

            self._shadow_total_checks += 1

            # Get batch result
            batch_value = batch_values[i] if i < len(batch_values) else 0.0

            # Compute sequential result using identical code path
            try:
                value_head = self._get_value_head(state)
                seq_values, _ = self.neural_net.evaluate_batch(
                    [state], value_head=value_head
                )
                seq_value = seq_values[0] if seq_values else 0.0
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.debug(f"GumbelMCTSAI: Shadow validation NN call failed: {e}")
                continue

            # Check for divergence using both absolute and relative thresholds
            abs_diff = abs(batch_value - seq_value)

            # For values near zero, use absolute threshold
            # For larger values, use relative threshold
            is_divergent = False
            divergence_str = ""
            if abs(seq_value) < 0.01:
                # Near-zero values: use absolute threshold of 0.001
                if abs_diff > 0.001:
                    is_divergent = True
                    divergence_str = f"abs_diff={abs_diff:.6f}"
            else:
                # Larger values: use 1% relative threshold
                rel_diff = abs_diff / abs(seq_value)
                if rel_diff > 0.01:
                    is_divergent = True
                    divergence_str = f"rel_diff={rel_diff:.1%}"

            if is_divergent:
                self._shadow_divergence_count += 1
                logger.warning(
                    f"GumbelMCTSAI: Batch/sequential NN divergence ({divergence_str}): "
                    f"batch={batch_value:.6f}, seq={seq_value:.6f}, "
                    f"action_idx={action_idx}, needs_flip={needs_flip}"
                )

                # Log detailed state info for debugging (first 3 divergences only)
                if self._shadow_divergence_count <= 3:
                    logger.warning(
                        f"  State: player={state.current_player}, "
                        f"status={state.game_status}, move_count={state.move_count}"
                    )

        # Periodic stats logging
        if self._shadow_total_checks > 0 and self._shadow_total_checks % 100 == 0:
            rate = self._shadow_divergence_count / self._shadow_total_checks
            logger.info(
                f"GumbelMCTSAI: Shadow validation stats: "
                f"{self._shadow_divergence_count}/{self._shadow_total_checks} divergences ({rate:.2%})"
            )

    def get_shadow_validation_stats(self) -> dict[str, Any]:
        """Get shadow validation statistics.

        Returns:
            Dictionary with:
            - total_checks: Number of validation checks performed
            - divergences: Number of divergences detected
            - divergence_rate: Percentage of checks that diverged
            - shadow_enabled: Whether shadow validation is enabled
        """
        rate = 0.0
        if self._shadow_total_checks > 0:
            rate = self._shadow_divergence_count / self._shadow_total_checks

        return {
            "total_checks": self._shadow_total_checks,
            "divergences": self._shadow_divergence_count,
            "divergence_rate": rate,
            "shadow_enabled": self._shadow_validate,
        }

    def _gpu_tree_shadow_validate(
        self,
        game_state: GameState,
        valid_moves: list[Move],
        gpu_move: Move,
        gpu_policy: dict[str, float],
    ) -> None:
        """Validate GPU tree search against CPU sequential halving.

        Compares the policy distribution from GPU tree search against
        CPU-based sequential halving to ensure training data quality parity.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves.
            gpu_move: Best move selected by GPU tree.
            gpu_policy: Policy distribution from GPU tree.
        """
        self._gpu_tree_total_checks += 1

        try:
            # Run CPU sequential halving for comparison
            policy_logits = self._get_policy_logits(game_state, valid_moves)
            actions = self._gumbel_top_k_sample(valid_moves, policy_logits)

            if len(actions) <= 1:
                return  # Single action, no comparison needed

            # Reset action stats and run sequential halving
            for a in actions:
                a.visit_count = 0
                a.total_value = 0.0
            cpu_best = self._sequential_halving(game_state, actions)

            # Build CPU policy from visit counts
            cpu_policy = {}
            total_visits = sum(a.visit_count for a in actions if a.visit_count > 0)
            if total_visits > 0:
                for a in actions:
                    if a.visit_count > 0:
                        move_key = self._gpu_gumbel_mcts._move_to_key(a.move)
                        cpu_policy[move_key] = a.visit_count / total_visits

            # Compare policies using KL divergence and max probability difference
            divergence_detected = False
            divergence_info = []

            # Check if best moves match
            cpu_move_key = self._gpu_gumbel_mcts._move_to_key(cpu_best.move)
            gpu_move_key = self._gpu_gumbel_mcts._move_to_key(gpu_move)
            if cpu_move_key != gpu_move_key:
                divergence_info.append(f"move_mismatch: GPU={gpu_move_key}, CPU={cpu_move_key}")
                divergence_detected = True

            # Check policy distribution similarity (max absolute difference)
            all_keys = set(gpu_policy.keys()) | set(cpu_policy.keys())
            max_diff = 0.0
            for key in all_keys:
                gpu_prob = gpu_policy.get(key, 0.0)
                cpu_prob = cpu_policy.get(key, 0.0)
                diff = abs(gpu_prob - cpu_prob)
                max_diff = max(max_diff, diff)

            # Tolerate 5% policy difference (due to stochastic simulation)
            POLICY_TOLERANCE = 0.05
            if max_diff > POLICY_TOLERANCE:
                divergence_info.append(f"policy_diff: max_diff={max_diff:.3f} > {POLICY_TOLERANCE}")
                divergence_detected = True

            if divergence_detected:
                self._gpu_tree_divergence_count += 1
                logger.warning(
                    f"GumbelMCTSAI: GPU tree / CPU parity divergence: {', '.join(divergence_info)}"
                )
                # Log detailed info for first few divergences
                if self._gpu_tree_divergence_count <= 3:
                    logger.warning(
                        f"  GPU policy top-3: {sorted(gpu_policy.items(), key=lambda x: -x[1])[:3]}"
                    )
                    logger.warning(
                        f"  CPU policy top-3: {sorted(cpu_policy.items(), key=lambda x: -x[1])[:3]}"
                    )

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning(f"GumbelMCTSAI: GPU tree shadow validation error: {e}")

        # Periodic stats logging
        if self._gpu_tree_total_checks > 0 and self._gpu_tree_total_checks % 20 == 0:
            rate = self._gpu_tree_divergence_count / self._gpu_tree_total_checks
            logger.info(
                f"GumbelMCTSAI: GPU tree shadow validation: "
                f"{self._gpu_tree_divergence_count}/{self._gpu_tree_total_checks} divergences ({rate:.2%})"
            )

    def get_gpu_tree_validation_stats(self) -> dict[str, Any]:
        """Get GPU tree shadow validation statistics.

        Returns:
            Dictionary with:
            - total_checks: Number of GPU tree validation checks performed
            - divergences: Number of divergences detected
            - divergence_rate: Percentage of checks that diverged
            - shadow_rate: Configured shadow validation rate
        """
        rate = 0.0
        if self._gpu_tree_total_checks > 0:
            rate = self._gpu_tree_divergence_count / self._gpu_tree_total_checks

        return {
            "total_checks": self._gpu_tree_total_checks,
            "divergences": self._gpu_tree_divergence_count,
            "divergence_rate": rate,
            "shadow_rate": self._gpu_tree_shadow_rate,
        }

    def validate_move_parity(
        self,
        game_state: GameState,
        valid_moves: list[Move],
    ) -> tuple[bool, str | None]:
        """Validate that GPU batch and CPU sequential paths select the same move.

        This is the ultimate parity check - it runs the full search with both
        batch (GPU) and sequential (CPU) evaluation and compares the selected moves.

        Note: This is expensive and should only be used for debugging/validation,
        not during normal gameplay or training.

        Args:
            game_state: Current game state.
            valid_moves: List of valid moves.

        Returns:
            Tuple of (moves_match: bool, difference_info: Optional[str])
        """
        if len(valid_moves) <= 1:
            return True, None

        # Get policy logits (same for both paths)
        policy_logits = self._get_policy_logits(game_state, valid_moves)

        # Sample actions (use same seed for both paths)
        import copy
        rng_state = copy.deepcopy(self.rng.getstate()) if hasattr(self.rng, 'getstate') else None

        # Run with GPU batch path
        original_gpu_available = self._gpu_available
        self._gpu_available = True
        actions_gpu = self._gumbel_top_k_sample(valid_moves, policy_logits)
        if len(actions_gpu) > 1:
            # Reset action stats for fresh comparison
            for a in actions_gpu:
                a.visit_count = 0
                a.total_value = 0.0
            best_gpu = self._sequential_halving(game_state, actions_gpu)
        else:
            best_gpu = actions_gpu[0] if actions_gpu else None

        # Restore RNG and run with CPU sequential path
        if rng_state is not None:
            self.rng.setstate(rng_state)

        self._gpu_available = False
        actions_cpu = self._gumbel_top_k_sample(valid_moves, policy_logits)
        if len(actions_cpu) > 1:
            for a in actions_cpu:
                a.visit_count = 0
                a.total_value = 0.0
            best_cpu = self._sequential_halving(game_state, actions_cpu)
        else:
            best_cpu = actions_cpu[0] if actions_cpu else None

        # Restore original GPU state
        self._gpu_available = original_gpu_available

        # Compare moves
        if best_gpu is None and best_cpu is None:
            return True, None

        if best_gpu is None or best_cpu is None:
            return False, f"GPU={best_gpu}, CPU={best_cpu}"

        if str(best_gpu.move) == str(best_cpu.move):
            return True, None

        return False, (
            f"GPU selected {best_gpu.move} (Q={best_gpu.mean_value:.4f}, visits={best_gpu.visit_count}), "
            f"CPU selected {best_cpu.move} (Q={best_cpu.mean_value:.4f}, visits={best_cpu.visit_count})"
        )

    def _sequential_halving(
        self,
        game_state: GameState,
        actions: list[GumbelAction],
    ) -> GumbelAction:
        """Run Sequential Halving to find the best action.

        Progressively halves the number of candidate actions, allocating
        simulation budget evenly across phases.

        GPU Acceleration:
            When GPU is available and not disabled via RINGRIFT_GPU_GUMBEL_DISABLE,
            uses batch evaluation to evaluate all simulations for all actions in
            each phase with a single neural network forward pass, providing 3-5x
            speedup.

        Args:
            game_state: Current game state.
            actions: List of candidate actions from Gumbel-Top-K.

        Returns:
            Best action after Sequential Halving.
        """
        m = len(actions)
        if m == 1:
            return actions[0]

        # Check GPU availability (lazy initialization)
        self._ensure_gpu_available()  # Side effect: sets self._gpu_available

        # Use batched version if GPU is available and not disabled
        if (
            self._gpu_available
            and self.neural_net is not None
            and not _GPU_GUMBEL_DISABLE
        ):
            try:
                return self._sequential_halving_batched(game_state, actions)
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning(
                    f"GPU batched sequential halving failed, falling back to CPU: {e}"
                )
                # Fall through to sequential version

        # Sequential (CPU) version - use adaptive budget based on game phase
        adaptive_budget = self._get_adaptive_budget(game_state)
        num_phases = int(np.ceil(np.log2(m)))
        budget_per_phase = adaptive_budget // max(num_phases, 1)

        remaining = list(actions)

        for _phase in range(num_phases):
            if len(remaining) == 1:
                break

            # Allocate budget evenly across remaining actions
            sims_per_action = max(1, budget_per_phase // len(remaining))

            # Use tree-based search for proper MCTS exploration
            for action in remaining:
                value_sum = self._simulate_action(
                    game_state, action.move, sims_per_action
                )
                action.visit_count += sims_per_action
                action.total_value += value_sum

            # Sort by completed Q-value and keep top half
            # Session 17.24: Use adaptive c_visit for phase/board-aware exploration
            max_visits = max(a.visit_count for a in remaining)
            board_type_str = getattr(game_state.board_type, "value", None) if hasattr(game_state, "board_type") else None
            move_num = len(game_state.move_history) if hasattr(game_state, "move_history") else 0
            num_legal = len(game_state.valid_moves) if hasattr(game_state, "valid_moves") else 0
            adaptive_c = get_adaptive_c_visit(
                board_type=board_type_str,
                num_legal_moves=num_legal,
                move_number=move_num,
            )
            remaining.sort(key=lambda a: a.completed_q(max_visits, c_visit=adaptive_c), reverse=True)
            remaining = remaining[: max(1, len(remaining) // 2)]

        return remaining[0]

    def _sequential_halving_batched(
        self,
        game_state: GameState,
        actions: list[GumbelAction],
    ) -> GumbelAction:
        """Run Sequential Halving with batched GPU evaluation.

        Collects all leaf evaluations across all simulations for all actions
        in each phase, then evaluates them in a single batch for 3-5x speedup.

        Args:
            game_state: Current game state.
            actions: List of candidate actions from Gumbel-Top-K.

        Returns:
            Best action after Sequential Halving.
        """
        m = len(actions)
        if m == 1:
            return actions[0]

        # Use adaptive budget based on game phase
        adaptive_budget = self._get_adaptive_budget(game_state)
        num_phases = int(np.ceil(np.log2(m)))
        budget_per_phase = adaptive_budget // max(num_phases, 1)
        remaining = list(actions)
        value_head = self._get_value_head(game_state)

        # Create evaluation buffer
        eval_buffer = LeafEvaluationBuffer(
            self.neural_net, max_batch_size=256
        )

        for _phase in range(num_phases):
            if len(remaining) == 1:
                break

            sims_per_action = max(1, budget_per_phase // len(remaining))

            # Storage for collected values: action_idx -> list of values
            action_values: dict[int, list[float]] = {
                i: [] for i in range(len(remaining))
            }

            # Phase 1: Collect all leaf states across all actions and simulations
            for action_idx, action in enumerate(remaining):
                mstate = MutableGameState.from_immutable(game_state)
                undo = mstate.make_move(action.move)

                # Handle terminal states immediately
                if mstate.is_game_over():
                    winner = mstate.winner
                    if winner == self.player_number:
                        v = 1.0
                    elif winner is None:
                        v = 0.0
                    else:
                        v = -1.0
                    action_values[action_idx] = [v] * sims_per_action
                    mstate.unmake_move(undo)
                    continue

                # Run simulations, collecting leaf states
                for sim_idx in range(sims_per_action):
                    sim_mstate = MutableGameState.from_immutable(mstate.to_immutable())
                    leaf_state, is_terminal, terminal_value, is_opponent = (
                        self._collect_leaf_state(sim_mstate)
                    )

                    if is_terminal:
                        action_values[action_idx].append(terminal_value)
                    else:
                        # Add to batch buffer
                        eval_buffer.add(LeafEvalRequest(
                            game_state=leaf_state,
                            is_opponent_perspective=is_opponent,
                            action_idx=action_idx,
                            simulation_idx=sim_idx,
                        ))

                mstate.unmake_move(undo)

            # Phase 2: Batch evaluate all collected leaf states
            if len(eval_buffer) > 0:
                results = eval_buffer.flush(value_head)
                for a_idx, s_idx, value in results:
                    action_values[a_idx].append(value)

            # Phase 3: Update action statistics
            for action_idx, action in enumerate(remaining):
                values = action_values[action_idx]
                action.visit_count += len(values)
                action.total_value += sum(values)

            # Phase 4: Sort by completed Q-value and keep top half
            # Session 17.24: Use adaptive c_visit for phase/board-aware exploration
            max_visits = max(a.visit_count for a in remaining)
            board_type_str = getattr(game_state.board_type, "value", None) if hasattr(game_state, "board_type") else None
            move_num = len(game_state.move_history) if hasattr(game_state, "move_history") else 0
            num_legal = len(game_state.valid_moves) if hasattr(game_state, "valid_moves") else 0
            adaptive_c = get_adaptive_c_visit(
                board_type=board_type_str,
                num_legal_moves=num_legal,
                move_number=move_num,
            )
            remaining.sort(key=lambda a: a.completed_q(max_visits, c_visit=adaptive_c), reverse=True)
            remaining = remaining[: max(1, len(remaining) // 2)]

        return remaining[0]

    def _collect_leaf_state(
        self,
        mstate: MutableGameState,
        max_depth: int = 10,
    ) -> tuple[GameState, bool, float, bool]:
        """Traverse tree to collect a leaf state for batch evaluation.

        Similar to _run_tree_simulation but returns the leaf state instead
        of immediately evaluating it.

        Args:
            mstate: Mutable game state (will be modified in-place).
            max_depth: Maximum depth to search.

        Returns:
            Tuple of (leaf_state, is_terminal, terminal_value, is_opponent_perspective)
        """
        depth = 0

        # Traverse tree until leaf (simplified - no tree reuse for batching)
        while depth < max_depth and not mstate.is_game_over():
            valid_moves = self.rules_engine.get_valid_moves(
                mstate.to_immutable(), mstate.current_player
            )

            if not valid_moves:
                break

            # Random move selection for exploration (simplified for batching)
            move = self.rng.choice(valid_moves)
            mstate.make_move(move)
            depth += 1

        # Return leaf state info
        if mstate.is_game_over():
            winner = mstate.winner
            if winner == self.player_number:
                value = 1.0
            elif winner is None:
                value = 0.0
            else:
                value = -1.0
            return mstate.to_immutable(), True, value, False

        is_opponent = mstate.current_player != self.player_number
        return mstate.to_immutable(), False, 0.0, is_opponent

    def _simulate_action(
        self,
        game_state: GameState,
        action: Move,
        num_sims: int,
    ) -> float:
        """Simulate an action and return cumulative value.

        Runs MCTS-style tree simulations from the state after taking the action.
        Uses proper tree search with selection, expansion, evaluation, and backup.

        Args:
            game_state: Current game state.
            action: Action to simulate.
            num_sims: Number of simulations to run.

        Returns:
            Sum of values from all simulations.
        """
        # Create mutable state for simulation
        mstate = MutableGameState.from_immutable(game_state)

        # Apply the action
        undo = mstate.make_move(action)

        # Check if terminal after action
        if mstate.is_game_over():
            winner = mstate.winner
            if winner == self.player_number:
                value = 1.0
            elif winner is None:
                value = 0.0
            else:
                value = -1.0
            mstate.unmake_move(undo)
            return value * num_sims

        # Create root node for tree search from this position
        root = GumbelNode(
            move=action,
            parent=None,
            prior=1.0,
            to_move_is_root=(mstate.current_player == self.player_number),
        )

        total_value = 0.0

        # Run tree simulations
        for _ in range(num_sims):
            # Clone state for each simulation (tree simulation modifies state)
            sim_mstate = MutableGameState.from_immutable(mstate.to_immutable())
            value = self._run_tree_simulation(sim_mstate, root, max_depth=10)
            total_value += value

        # Unmake the action
        mstate.unmake_move(undo)

        return total_value

    def _run_tree_simulation(
        self,
        mstate: MutableGameState,
        root: GumbelNode,
        max_depth: int = 10,
    ) -> float:
        """Run a single tree simulation from the current state.

        Uses PUCT-style selection for internal nodes and neural network
        evaluation at leaves.

        Args:
            mstate: Mutable game state (will be modified in-place).
            root: Root node of the search tree.
            max_depth: Maximum depth to search.

        Returns:
            Value estimate from the simulation.
        """
        path: list[tuple[GumbelNode, MoveUndo]] = []
        node = root
        depth = 0

        # Check if using multiplayer value heads (3+ players)
        sim_state = mstate.to_immutable()
        num_players = _infer_num_players(sim_state)
        use_multiplayer_heads = (
            num_players > 2
            and self.neural_net is not None
            and getattr(self.neural_net, "num_players", 4) >= num_players
        )

        # Selection phase - traverse tree until leaf
        while depth < max_depth and not mstate.is_game_over():
            valid_moves = self.rules_engine.get_valid_moves(
                mstate.to_immutable(), mstate.current_player
            )

            if not valid_moves:
                break

            # Check if node is fully expanded
            unexpanded = [m for m in valid_moves if str(m) not in node.children]

            if unexpanded:
                # Expansion - add new child
                move = self.rng.choice(unexpanded)
                undo = mstate.make_move(move)

                child = GumbelNode(
                    move=move,
                    parent=node,
                    prior=1.0 / len(valid_moves),
                    to_move_is_root=(mstate.current_player == self.player_number),
                )
                node.children[str(move)] = child
                path.append((child, undo))
                node = child
                break
            else:
                # Selection - pick best child by PUCT
                best_move = self._select_puct_move(node, valid_moves)
                undo = mstate.make_move(best_move)

                child = node.children.get(str(best_move))
                if child is None:
                    # Shouldn't happen but handle gracefully
                    break

                path.append((child, undo))
                node = child
                depth += 1

        # Evaluation phase
        if mstate.is_game_over():
            # Terminal state evaluation (exact values)
            winner = mstate.winner
            if winner == self.player_number:
                value = 1.0
            elif winner is None:
                value = 0.0
            else:
                value = -1.0
        else:
            # Non-terminal: use hybrid NN + heuristic evaluation
            is_opponent = mstate.current_player != self.player_number
            value = self._evaluate_leaf_hybrid(mstate, is_opponent)

        # Backpropagation - update all nodes in path
        current_value = value
        for child_node, undo in reversed(path):
            child_node.visit_count += 1
            child_node.total_value += current_value

            # For 2-player games, flip value for opponent nodes.
            # For 3+ player games with per-player value heads, the value is already
            # from our perspective, so no flip is needed during backpropagation.
            if not use_multiplayer_heads:
                if child_node.parent and (
                    child_node.to_move_is_root
                    != getattr(child_node.parent, "to_move_is_root", True)
                ):
                    current_value = -current_value

            # Unmake move
            mstate.unmake_move(undo)

        return value

    def _select_puct_move(
        self,
        node: GumbelNode,
        valid_moves: list[Move],
    ) -> Move:
        """Select move using PUCT formula.

        Args:
            node: Current node with children.
            valid_moves: Valid moves from current position.

        Returns:
            Best move according to PUCT.
        """
        total_visits = node.visit_count + 1

        def puct_score(move: Move) -> float:
            child = node.children.get(str(move))
            if child is None:
                return float("inf")  # Prefer unexplored

            q = child.mean_value
            # Flip Q for opponent nodes
            if not child.to_move_is_root:
                q = -q

            # Use adaptive c_puct based on game phase (Dec 2025)
            adaptive_cpuct = self._get_adaptive_cpuct()
            u = adaptive_cpuct * child.prior * math.sqrt(total_visits) / (
                1 + child.visit_count
            )
            return q + u

        return max(valid_moves, key=puct_score)

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using hybrid NN + heuristic blending.

        Args:
            game_state: Current game state.

        Returns:
            Value estimate (blended NN + heuristic if configured, or pure NN/heuristic).
        """
        # Terminal state check
        if game_state.game_status == "completed":
            if game_state.winner == self.player_number:
                return 1.0
            elif game_state.winner is not None:
                return -1.0
            return 0.0

        # Try NN evaluation first
        nn_value: float | None = None
        if self.neural_net is not None:
            try:
                value_head = self._get_value_head(game_state)
                values, _ = self.neural_net.evaluate_batch(
                    [game_state], value_head=value_head
                )
                nn_value = values[0] if values else None
            except (RuntimeError, ValueError, IndexError, AttributeError):
                pass

        # Compute heuristic value if needed
        heuristic_value: float | None = None
        need_heuristic = (
            self._heuristic_blend_alpha is not None or
            (nn_value is None and self._heuristic_fallback_enabled)
        )

        if need_heuristic:
            heuristic_ai = self._ensure_heuristic_evaluator()
            if heuristic_ai is not None:
                try:
                    heuristic_ai.player_number = self.player_number
                    raw_score = heuristic_ai.evaluate_position(game_state)
                    heuristic_value = self._normalize_heuristic_score(raw_score)
                except (RuntimeError, ValueError, AttributeError):
                    pass

        # Compute final value
        if nn_value is not None and heuristic_value is not None:
            alpha = (
                self._heuristic_blend_alpha
                if self._heuristic_blend_alpha is not None
                else 1.0
            )
            return alpha * nn_value + (1.0 - alpha) * heuristic_value
        elif nn_value is not None:
            return nn_value
        elif heuristic_value is not None:
            return heuristic_value
        return 0.0

    def reset_for_new_game(self, *, rng_seed: int | None = None) -> None:
        """Reset state for a new game.

        Args:
            rng_seed: Optional new RNG seed.
        """
        super().reset_for_new_game(rng_seed=rng_seed)
        # No persistent tree to reset in Gumbel MCTS (tree is built fresh each move)

    def get_visit_distribution(self) -> tuple[list[Move], list[float]]:
        """Extract normalized visit count distribution from the last search.

        Returns a tuple of (moves, visit_probabilities) representing the
        Gumbel MCTS policy based on visit counts from Sequential Halving.
        This can be used as soft policy targets during training.

        Returns:
            Tuple of (list of moves, list of visit probabilities) where
            probabilities sum to 1.0. Returns ([], []) if no search has
            been performed.
        """
        if self._last_search_actions is None or not self._last_search_actions:
            return [], []

        # Filter actions with non-zero visits
        visited_actions = [a for a in self._last_search_actions if a.visit_count > 0]
        if not visited_actions:
            return [], []

        total_visits = sum(a.visit_count for a in visited_actions)
        if total_visits == 0:
            return [], []

        moves: list[Move] = []
        probs: list[float] = []

        for action in visited_actions:
            moves.append(action.move)
            probs.append(action.visit_count / total_visits)

        return moves, probs

    def get_search_stats(self) -> dict | None:
        """Get rich search statistics from the last GPU tree search.

        Returns Q-values, visit counts, uncertainty, prior policy, and other
        statistics useful for auxiliary training targets.

        Returns:
            Dict with search statistics, or None if no GPU tree search
            was performed (CPU sequential halving doesn't produce stats).

        Example return value:
            {
                "q_values": {"0,1->2,3": 0.45, "1,2->3,4": 0.52, ...},
                "visit_counts": {"0,1->2,3": 15, "1,2->3,4": 22, ...},
                "search_depth": 4,
                "uncertainty": 0.12,
                "prior_policy": {"0,1->2,3": 0.25, ...},
                "root_value": 0.35,
                "nodes_explored": 128,
                "total_simulations": 150
            }
        """
        return self._last_search_stats

    def __repr__(self) -> str:
        """String representation."""
        gpu_status = "enabled" if self._gpu_batch_enabled else "disabled"
        if self._gpu_available is not None:
            gpu_status = f"active:{self._gpu_device.type}" if self._gpu_available else "unavailable"
        return (
            f"GumbelMCTSAI(player={self.player_number}, "
            f"m={self.num_sampled_actions}, "
            f"budget={self.simulation_budget}, "
            f"model={self.config.nn_model_id}, "
            f"gpu={gpu_status})"
        )


def create_gumbel_mcts(
    board_type: str | BoardType,
    num_players: int = 2,
    num_sampled_actions: int = 16,
    simulation_budget: int = 800,
    neural_net: NeuralNetAI | None = None,
    player_number: int = 1,
    gpu_simulation: bool = False,
    model_path: str | None = None,
) -> GumbelMCTSAI:
    """Factory function to create Gumbel MCTS with optional GPU acceleration.

    This is the recommended way to create a Gumbel MCTS instance. It provides:
    - Easy configuration with sensible defaults
    - Optional GPU simulation acceleration (2-3x speedup)
    - Automatic neural network loading from path

    Args:
        board_type: Board type string ("square8", "hex8", etc.) or enum.
        num_players: Number of players (2, 3, or 4).
        num_sampled_actions: K for Gumbel-Top-K sampling (default 16).
        simulation_budget: Total simulations per move (default 800).
        neural_net: Pre-loaded neural network, or None to create fresh.
        player_number: Player number this AI represents.
        gpu_simulation: Enable GPU simulation acceleration.
                       When True, uses GumbelMCTSGPU for faster playouts.
        model_path: Path to model file for neural network.

    Returns:
        GumbelMCTSAI instance (or GumbelMCTSGPU if gpu_simulation=True).

    Example:
        # Standard MCTS with GPU NN batching
        mcts = create_gumbel_mcts("square8", num_players=2)

        # Full GPU acceleration for maximum speed
        mcts = create_gumbel_mcts(
            "square8",
            num_players=2,
            gpu_simulation=True,
            simulation_budget=800,
        )
    """
    from ..models import AIConfig, AIType

    if isinstance(board_type, str):
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(board_type.lower(), BoardType.SQUARE8)
    else:
        board_type_enum = board_type

    config = AIConfig(
        ai_type=AIType.GUMBEL_MCTS,
        difficulty=7,
        gumbel_num_sampled_actions=num_sampled_actions,
        gumbel_simulation_budget=simulation_budget,
        nn_model_id=model_path,
        allow_fresh_weights=True,  # Don't fail if no model
    )

    if gpu_simulation:
        # Use GPU-accelerated version with full GPU simulation
        from .gumbel_mcts_gpu import GumbelMCTSGPU

        mcts = GumbelMCTSGPU(
            player_number=player_number,
            config=config,
            board_type=board_type_enum,
            neural_net=neural_net,
            gpu_simulation=True,
        )
    else:
        # Standard MCTS with GPU NN batching only
        mcts = GumbelMCTSAI(
            player_number=player_number,
            config=config,
            board_type=board_type_enum,
        )
        if neural_net is not None:
            mcts.neural_net = neural_net

    return mcts
