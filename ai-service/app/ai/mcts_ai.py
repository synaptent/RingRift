"""MCTS AI implementation for RingRift.

This module implements a Monte Carlo Tree Search (MCTS) agent used by the
Python AI service. It supports both immutable (legacy) and mutable
make/unmake search modes, gated by ``config.use_incremental_search``.

When incremental search is enabled (the default), :class:`MCTSAI` operates
on :class:`MutableGameState` and a lightweight node representation to
reduce allocation overhead. The legacy path uses full :class:`GameState`
clones for backwards‑compatible behaviour and debugging.

GPU Acceleration (default enabled):
    - Rollout position evaluation uses GPU heuristic evaluator for 5-20x speedup
    - Full rule parity maintained (CPU rules engine, only evaluation is GPU)
    - Automatic fallback to CPU if no GPU available
    - Control via RINGRIFT_GPU_MCTS_DISABLE=1 environment variable
    - Shadow validation available via RINGRIFT_GPU_MCTS_SHADOW_VALIDATE=1
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any, cast, List, Tuple, TYPE_CHECKING
import math
import time

import numpy as np
import psutil

from .bounded_transposition_table import BoundedTranspositionTable
from .game_state_utils import infer_num_players
from .heuristic_ai import HeuristicAI
from ..models import GameState, Move, MoveType, AIConfig, BoardType, GamePhase
from ..rules.mutable_state import MutableGameState, MoveUndo
from ..utils.memory_config import MemoryConfig

# Lazy imports for neural network components to avoid loading torch when not needed
# These are only imported when difficulty >= 6 (neural MCTS tiers)
if TYPE_CHECKING:
    import torch
    from .async_nn_eval import AsyncNeuralBatcher
    from .neural_net import (
        NeuralNetAI,
        ActionEncoderHex,
        HexNeuralNet_v2,
    )
    from .nnue_policy import RingRiftNNUEWithPolicy

logger = logging.getLogger(__name__)

# GPU acceleration environment variable controls
_GPU_MCTS_DISABLE = os.environ.get("RINGRIFT_GPU_MCTS_DISABLE", "").lower() in (
    "1", "true", "yes", "on"
)
_GPU_MCTS_SHADOW_VALIDATE = os.environ.get("RINGRIFT_GPU_MCTS_SHADOW_VALIDATE", "").lower() in (
    "1", "true", "yes", "on"
)

# Module-level cache for NNUE policy models to avoid reloading per MCTSAI instance
# Key: (board_type.value, num_players) -> RingRiftNNUEWithPolicy model
_NNUE_POLICY_CACHE: Dict[Tuple[str, int], Any] = {}
_NNUE_POLICY_CACHE_LOCK = None  # Lazy init threading lock


def _get_cached_nnue_policy(board_type: BoardType, num_players: int) -> Optional[Any]:
    """Get cached NNUE policy model or load and cache it."""
    global _NNUE_POLICY_CACHE_LOCK

    cache_key = (board_type.value, num_players)

    # Fast path - already cached
    if cache_key in _NNUE_POLICY_CACHE:
        return _NNUE_POLICY_CACHE[cache_key]

    # Lazy init lock
    if _NNUE_POLICY_CACHE_LOCK is None:
        import threading
        _NNUE_POLICY_CACHE_LOCK = threading.Lock()

    with _NNUE_POLICY_CACHE_LOCK:
        # Double-check after acquiring lock
        if cache_key in _NNUE_POLICY_CACHE:
            return _NNUE_POLICY_CACHE[cache_key]

        try:
            import torch
            from .nnue_policy import RingRiftNNUEWithPolicy
            import re

            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "models", "nnue", f"nnue_policy_{board_type.value}_{num_players}p.pt"
            )
            model_path = os.path.normpath(model_path)

            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

                # Handle versioned checkpoints (with model_state_dict key)
                # and legacy direct state_dict format
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Use checkpoint metadata for hidden_dim/layers if available
                    hidden_dim = checkpoint.get('hidden_dim', 128)
                    num_hidden_layers = checkpoint.get('num_hidden_layers', 2)
                else:
                    # Legacy format: direct state_dict
                    state_dict = checkpoint
                    # Extract hidden dim from checkpoint weights
                    hidden_dim = 128
                    num_hidden_layers = 2
                    if isinstance(state_dict, dict):
                        for key in state_dict.keys():
                            match = re.match(r"fc1\.weight", key)
                            if match and hasattr(state_dict[key], "shape"):
                                hidden_dim = state_dict[key].shape[0]
                                break
                        num_fc_keys = sum(1 for k in state_dict.keys() if k.startswith("fc") and k.endswith(".weight"))
                        if num_fc_keys >= 2:
                            num_hidden_layers = num_fc_keys - 1

                model = RingRiftNNUEWithPolicy(
                    board_type=board_type,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                model.load_state_dict(state_dict)
                model.eval()

                _NNUE_POLICY_CACHE[cache_key] = model
                logger.info(
                    f"NNUE Policy Cache: Loaded model for {board_type.value}_{num_players}p "
                    f"(hidden={hidden_dim}, layers={num_hidden_layers})"
                )
                return model
            else:
                # Mark as None to avoid repeated load attempts
                _NNUE_POLICY_CACHE[cache_key] = None
                logger.debug(f"NNUE Policy Cache: No model at {model_path}")
                return None

        except Exception as e:
            logger.warning(f"NNUE Policy Cache: Failed to load model: {e}")
            _NNUE_POLICY_CACHE[cache_key] = None
            return None


def _pos_key(pos: Optional[Any]) -> Optional[str]:
    if pos is None:
        return None
    to_key = getattr(pos, "to_key", None)
    if callable(to_key):
        return cast(str, to_key())
    x = getattr(pos, "x", None)
    y = getattr(pos, "y", None)
    z = getattr(pos, "z", None)
    if x is None or y is None:
        return None
    return f"{x},{y},{z}" if z is not None else f"{x},{y}"


def _pos_seq_key(seq: Optional[Tuple[Any, ...]]) -> Optional[Tuple[str, ...]]:
    if not seq:
        return None
    return tuple(k for k in (_pos_key(p) for p in seq) if k is not None)


def _move_key(move: Move) -> tuple:
    """Return a stable, hashable key for comparing AI moves.

    Purpose:
        Used to avoid conflating distinct moves that share the same (type, to)
        shape but differ in semantics (e.g. multi-ring place_ring placementCount,
        choose_line_option collapsed_markers segments).

    Notes:
        This key intentionally excludes timing metadata (timestamp/think_time/
        move_number) so that host-synthesized bookkeeping moves remain comparable
        across regenerations.
    """
    move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
    return (
        move_type,
        int(move.player),
        _pos_key(getattr(move, "from_pos", None)),
        _pos_key(getattr(move, "to", None)),
        _pos_key(getattr(move, "capture_target", None)),
        getattr(move, "placement_count", None),
        getattr(move, "placed_on_stack", None),
        getattr(move, "line_index", None),
        _pos_seq_key(getattr(move, "collapsed_markers", None)),
        _pos_seq_key(getattr(move, "collapse_positions", None)),
        tuple(getattr(move, "extraction_stacks", None) or ()),
        getattr(move, "recovery_option", None),
        getattr(move, "recovery_mode", None),
        getattr(move, "elimination_context", None),
        _pos_seq_key(getattr(move, "capture_chain", None)),
        tuple(getattr(move, "overtaken_rings", None) or ()),
    )


def _moves_match(m1: Move, m2: Move) -> bool:
    """Check if two moves match by semantic identity (not timing metadata)."""
    if m1.type != m2.type or m1.player != m2.player:
        return False
    if m1.from_pos != m2.from_pos:
        return False
    if m1.to != m2.to:
        return False
    if m1.capture_target != m2.capture_target:
        return False
    if m1.placement_count != m2.placement_count:
        return False
    if m1.placed_on_stack != m2.placed_on_stack:
        return False
    if m1.line_index != m2.line_index:
        return False
    if m1.collapsed_markers != m2.collapsed_markers:
        return False
    if m1.collapse_positions != m2.collapse_positions:
        return False
    if m1.extraction_stacks != m2.extraction_stacks:
        return False
    if m1.recovery_option != m2.recovery_option:
        return False
    if m1.recovery_mode != m2.recovery_mode:
        return False
    if m1.elimination_context != m2.elimination_context:
        return False
    if m1.capture_chain != m2.capture_chain:
        return False
    if m1.overtaken_rings != m2.overtaken_rings:
        return False
    return True


class MCTSNode:
    """MCTS tree node for legacy (immutable) search.

    Each node owns its own :class:`GameState` snapshot. This path trades
    memory for simplicity and is kept primarily for A/B testing against
    the incremental implementation and for debugging.
    """

    def __init__(
        self,
        game_state: GameState,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Move] = None,
    ) -> None:
        self.game_state: GameState = game_state
        self.parent: Optional["MCTSNode"] = parent
        self.move: Optional[Move] = move
        self.children: List["MCTSNode"] = []
        self.wins = 0
        self.visits = 0
        self.amaf_wins = 0
        self.amaf_visits = 0
        self.untried_moves: List[Move] = []
        self.prior = 0.0
        self.policy_map: Dict[str, float] = {}
        # Whether the side to move at this node is the root player (AI player).
        # Populated by MCTSAI during tree construction/traversal.
        self.to_move_is_root: bool = True

    def uct_select_child(
        self,
        *,
        c_puct: float = 1.0,
        rave_k: float = 1000.0,
        fpu_reduction: float = 0.0,
    ) -> "MCTSNode":
        """Select child using PUCT formula with RAVE."""
        # PUCT = Q + c_puct * P * sqrt(N) / (1 + n)
        # RAVE: Value = (1 - beta) * Q + beta * AMAF

        def puct_value(child: "MCTSNode") -> float:
            parent_is_root = bool(getattr(self, "to_move_is_root", True))
            child_is_root = bool(getattr(child, "to_move_is_root", parent_is_root))
            flip = parent_is_root != child_is_root

            # Q(s,a) in the parent's perspective (root vs opponent coalition).
            parent_q = self.wins / self.visits if self.visits > 0 else 0.0
            if child.visits == 0:
                q_value = parent_q - float(fpu_reduction)
            else:
                q_value = child.wins / child.visits
                if flip:
                    q_value = -q_value

            # AMAF (RAVE) value, also in the parent's perspective.
            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
                if flip:
                    amaf_value = -amaf_value

            beta = 0.0
            if rave_k > 0:
                beta = math.sqrt(float(rave_k) / (3 * self.visits + float(rave_k)))

            combined_value = (1 - beta) * q_value + beta * amaf_value

            # Prior probability P(s, a)
            prior = getattr(child, "prior", 1.0 / len(self.children))

            u_value = (
                c_puct * prior * math.sqrt(self.visits) / (1 + child.visits)
            )
            return combined_value + u_value

        return max(self.children, key=puct_value)

    def add_child(
        self,
        move: Move,
        game_state: GameState,
        prior: Optional[float] = None,
    ) -> "MCTSNode":
        """Add a new child node."""
        child = MCTSNode(game_state, parent=self, move=move)
        if prior is not None:
            child.prior = prior
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(
        self,
        result: float,
        played_moves: Optional[List[Move]] = None,
    ) -> None:
        """Update node stats."""
        self.visits += 1
        self.wins += result

        if played_moves and self.move:
            # Update AMAF stats if this node's move was played in the
            # simulation. Check if self.move is in played_moves.
            for m in played_moves:
                if _moves_match(m, self.move):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


class MCTSNodeLite:
    """Lightweight MCTS tree node for incremental (mutable) search.

    This variant does **not** store a :class:`GameState`; the caller
    maintains a single mutable state and uses make/unmake to traverse
    the tree. This dramatically reduces memory usage while preserving
    behaviour.
    """
    __slots__ = [
        'parent', 'move', 'children', 'wins', 'visits',
        'amaf_wins', 'amaf_visits', 'untried_moves', 'prior', 'policy_map',
        'to_move_is_root',
    ]

    def __init__(
        self,
        parent: Optional["MCTSNodeLite"] = None,
        move: Optional[Move] = None,
        to_move_is_root: bool = True,
    ):
        self.parent: Optional["MCTSNodeLite"] = parent
        self.move: Optional[Move] = move
        self.children: List["MCTSNodeLite"] = []
        self.wins = 0.0
        self.visits = 0
        self.amaf_wins = 0.0
        self.amaf_visits = 0
        self.untried_moves: List[Move] = []
        self.prior = 0.0
        self.policy_map: Dict[str, float] = {}
        self.to_move_is_root: bool = bool(to_move_is_root)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0

    def uct_select_child(
        self,
        *,
        c_puct: float = 1.0,
        rave_k: float = 1000.0,
        fpu_reduction: float = 0.0,
    ) -> "MCTSNodeLite":
        """Select child using PUCT formula with RAVE."""
        def puct_value(child: "MCTSNodeLite") -> float:
            parent_is_root = bool(getattr(self, "to_move_is_root", True))
            child_is_root = bool(getattr(child, "to_move_is_root", parent_is_root))
            flip = parent_is_root != child_is_root

            parent_q = self.wins / self.visits if self.visits > 0 else 0.0
            if child.visits == 0:
                q_value = parent_q - float(fpu_reduction)
            else:
                q_value = child.wins / child.visits
                if flip:
                    q_value = -q_value

            if child.amaf_visits == 0:
                amaf_value = 0.0
            else:
                amaf_value = child.amaf_wins / child.amaf_visits
                if flip:
                    amaf_value = -amaf_value

            beta = 0.0
            if rave_k > 0:
                beta = math.sqrt(float(rave_k) / (3 * self.visits + float(rave_k)))
            combined_value = (1 - beta) * q_value + beta * amaf_value

            num_children = max(1, len(self.children))
            prior = child.prior if child.prior > 0 else 1.0 / num_children
            sqrt_visits = math.sqrt(self.visits)
            u_value = c_puct * prior * sqrt_visits / (1 + child.visits)

            return combined_value + u_value

        return max(self.children, key=puct_value)

    def add_child(
        self,
        move: Move,
        prior: Optional[float] = None,
        to_move_is_root: Optional[bool] = None,
    ) -> "MCTSNodeLite":
        """Add a new child node."""
        child = MCTSNodeLite(
            parent=self,
            move=move,
            to_move_is_root=(
                bool(to_move_is_root)
                if to_move_is_root is not None
                else bool(getattr(self, "to_move_is_root", True))
            ),
        )
        if prior is not None:
            child.prior = prior
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(
        self,
        result: float,
        played_moves: Optional[List[Move]] = None
    ) -> None:
        """Update node stats."""
        self.visits += 1
        self.wins += result

        if played_moves and self.move:
            for m in played_moves:
                if _moves_match(m, self.move):
                    self.amaf_visits += 1
                    self.amaf_wins += result
                    break


class DynamicBatchSizer:
    """Dynamically adjusts MCTS batch size based on available memory.

    This class monitors system memory and calculates optimal batch sizes
    for MCTS simulations, allowing more simulations when memory permits
    and gracefully reducing when memory is constrained.
    """

    def __init__(
        self,
        memory_config: Optional[MemoryConfig] = None,
        batch_size_min: int = 100,
        batch_size_max: int = 1600,
        memory_safety_margin: float = 0.8,
        node_size_estimate: int = 500,
    ):
        """Initialize the dynamic batch sizer.

        Args:
            memory_config: Memory configuration for limits
            batch_size_min: Minimum batch size (default: 100)
            batch_size_max: Maximum batch size (default: 1600)
            memory_safety_margin: Only use this fraction of available budget
                (default: 0.8 = 80%)
            node_size_estimate: Estimated bytes per MCTS node (default: 500)
        """
        self.memory_config = memory_config or MemoryConfig.from_env()
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.memory_safety_margin = memory_safety_margin
        self.node_size_estimate = node_size_estimate

        # Track actual memory usage to refine estimates
        self._memory_samples: List[Tuple[int, int]] = []
        self._last_batch_size = batch_size_max
        self._adjustment_count = 0

    def get_optimal_batch_size(self, current_node_count: int = 0) -> int:
        """Calculate optimal batch size based on available memory.

        Args:
            current_node_count: Current number of nodes in the tree

        Returns:
            Optimal batch size within configured limits
        """
        # Get available memory
        mem_info = psutil.virtual_memory()
        available_bytes = mem_info.available

        # Get inference budget from config
        inference_budget = self.memory_config.get_inference_limit_bytes()

        # Apply safety margin
        safe_budget = int(inference_budget * self.memory_safety_margin)

        # Use minimum of available memory and safe budget
        usable_memory = min(available_bytes, safe_budget)

        # Estimate memory already used by current tree
        current_tree_memory = current_node_count * self.node_size_estimate

        # Calculate remaining memory for new nodes
        remaining_memory = max(0, usable_memory - current_tree_memory)

        # Calculate how many new nodes we can create
        # Each batch iteration may create up to batch_size new nodes
        max_new_nodes = remaining_memory // self.node_size_estimate

        # Clamp to configured limits
        optimal = max(
            self.batch_size_min, min(self.batch_size_max, max_new_nodes)
        )

        # Log if batch size changes significantly
        if abs(optimal - self._last_batch_size) >= self.batch_size_min // 2:
            self._adjustment_count += 1
            logger.info(
                "Dynamic batch size adjusted: %d -> %d (available=%.2fMB, "
                "budget=%.2fMB, tree_nodes=%d, adjustments=%d)",
                self._last_batch_size,
                optimal,
                available_bytes / (1024**2),
                safe_budget / (1024**2),
                current_node_count,
                self._adjustment_count,
            )
            self._last_batch_size = optimal

        return optimal

    def record_memory_sample(self, node_count: int) -> None:
        """Record a memory sample to refine node size estimates.

        Args:
            node_count: Current number of nodes in the tree
        """
        # Get current process memory
        process = psutil.Process()
        memory_bytes = process.memory_info().rss

        self._memory_samples.append((node_count, memory_bytes))

        # Keep only recent samples (last 100)
        if len(self._memory_samples) > 100:
            self._memory_samples = self._memory_samples[-100:]

        # Refine estimate if we have enough samples
        if len(self._memory_samples) >= 10:
            self._refine_node_size_estimate()

    def _refine_node_size_estimate(self) -> None:
        """Refine node size estimate based on memory samples."""
        if len(self._memory_samples) < 2:
            return

        # Calculate average memory increase per node
        samples = sorted(self._memory_samples, key=lambda x: x[0])

        # Use linear regression-like approach
        deltas = []
        for i in range(1, len(samples)):
            node_delta = samples[i][0] - samples[i - 1][0]
            mem_delta = samples[i][1] - samples[i - 1][1]

            if node_delta > 0:
                deltas.append(mem_delta / node_delta)

        if deltas:
            # Use median to avoid outliers
            deltas.sort()
            median_idx = len(deltas) // 2
            estimated = int(deltas[median_idx])

            # Only update if estimate is reasonable
            if 100 <= estimated <= 2000:
                old_estimate = self.node_size_estimate
                self.node_size_estimate = estimated

                if abs(self.node_size_estimate - old_estimate) > 50:
                    logger.debug(
                        "Refined MCTS node size estimate: %d -> %d bytes",
                        old_estimate,
                        self.node_size_estimate,
                    )

    def stats(self) -> dict:
        """Return current stats.

        Returns:
            Dictionary with current batch size, estimates, and adjustment count
        """
        return {
            "current_batch_size": self._last_batch_size,
            "node_size_estimate": self.node_size_estimate,
            "batch_size_min": self.batch_size_min,
            "batch_size_max": self.batch_size_max,
            "memory_safety_margin": self.memory_safety_margin,
            "adjustment_count": self._adjustment_count,
            "sample_count": len(self._memory_samples),
        }


@dataclass
class _EvalBatchLegacy:
    leaves: List[Tuple[MCTSNode, GameState, List[Move]]]
    states: List[GameState]
    cached_results: List[Tuple[int, float, Any]]
    uncached_indices: List[int]
    uncached_states: List[GameState]
    use_hex_nn: bool


@dataclass
class _EvalBatchIncremental:
    leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]]
    states: List[GameState]
    cached_results: List[Tuple[int, float, Any]]
    uncached_indices: List[int]
    uncached_states: List[GameState]
    use_hex_nn: bool


class MCTSAI(HeuristicAI):
    """Monte Carlo Tree Search AI with neural network evaluation.

    MCTSAI combines MCTS with neural network value/policy estimates for
    stronger play. It supports both legacy (immutable) and incremental
    (mutable) search modes and reuses parts of :class:`HeuristicAI` for
    rollout policies.

    Configuration overview (:class:`AIConfig` / related fields):

    - ``use_incremental_search``: When ``True`` (default), uses the
      :class:`MutableGameState` make/unmake MCTS path
      (:meth:`_search_incremental`); when ``False``, uses the legacy
      immutable path (:meth:`_search_legacy`).
    - ``think_time``: Per‑move wall‑clock budget in milliseconds. When
      positive, both legacy and incremental searches treat this strictly
      as a search‑time limit; otherwise a difficulty‑scaled default
      (roughly 1.0s–5.5s) is used. No additional UX delay is added.
    - ``randomness``: Routed through :meth:`BaseAI.should_pick_random_move`
      to allow occasional random move selection before invoking MCTS.
    - ``nn_model_id`` / related NN config: Passed into
      :class:`NeuralNetAI` to select which checkpoint and architecture
      are used for value/policy heads. When no model is available,
      the agent falls back to heuristic rollouts.

    Memory and batching:
        Transposition tables for value/policy caches are sized using
        :class:`MemoryConfig` (either injected via ``memory_config`` or
        derived from the environment). Optional dynamic batch sizing for
        leaf evaluation is controlled by ``enable_dynamic_batching`` and
        an optional :class:`DynamicBatchSizer` instance, allowing MCTS to
        adapt its batch sizes to available memory.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        memory_config: Optional[MemoryConfig] = None,
        dynamic_sizer: Optional[DynamicBatchSizer] = None,
        enable_dynamic_batching: bool = False,
    ):
        """Initialize a new MCTS AI instance.

        Args:
            player_number: The player number (1 or 2).
            config: AI configuration for this instance.
            memory_config: Memory configuration for bounded structures.
            dynamic_sizer: Optional dynamic batch sizer for memory‑aware
                batching.
            enable_dynamic_batching: Enable dynamic batching (default:
                ``False`` for backward compatibility). If ``True`` and
                ``dynamic_sizer`` is ``None``, a default :class:`DynamicBatchSizer`
                will be created.
        """
        super().__init__(player_number, config)

        # Configuration option for incremental search (make/unmake pattern)
        self.use_incremental_search = getattr(
            config, 'use_incremental_search', True
        )

        # Neural network evaluation gating (D6+ when enabled)
        # Priority:
        # - Explicit AIConfig.use_neural_net when provided
        # - RINGRIFT_DISABLE_NEURAL_NET env var can globally disable NN usage
        # - Default: enabled for difficulty >= 6 (D6+ are neural MCTS tiers)
        disable_nn_env = os.environ.get("RINGRIFT_DISABLE_NEURAL_NET", "").lower() in {
            "1", "true", "yes", "on",
        }
        self.require_neural_net = os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_nn_config = getattr(config, "use_neural_net", None)
        # Neural MCTS is enabled at D6+ by default, can be explicitly disabled
        should_use_neural = (
            config.difficulty >= 6 and
            (use_nn_config if use_nn_config is not None else True) and
            not disable_nn_env
        )
        # Also allow explicit opt-in at any difficulty via use_neural_net=True
        if use_nn_config is True and not disable_nn_env:
            should_use_neural = True

        # Try to load neural net for evaluation when enabled
        self.neural_net: Optional["NeuralNetAI"] = None
        if should_use_neural:
            try:
                from .neural_net import NeuralNetAI  # Lazy import
                self.neural_net = NeuralNetAI(player_number, config)
                logger.info(
                    f"MCTSAI(player={player_number}, difficulty={config.difficulty}): "
                    "neural evaluation enabled"
                )
            except Exception as e:
                if self.require_neural_net:
                    raise
                logger.warning(f"Failed to load neural net for MCTS: {e}")
                self.neural_net = None
        else:
            logger.debug(
                f"MCTSAI(player={player_number}, difficulty={config.difficulty}): "
                "using heuristic evaluation (neural disabled)"
            )

        # Thread-safe NN batcher (also used for async leaf evaluation).
        self.nn_batcher: Optional["AsyncNeuralBatcher"] = None
        if self.neural_net is not None:
            from .async_nn_eval import AsyncNeuralBatcher  # Lazy import
            self.nn_batcher = AsyncNeuralBatcher(self.neural_net)

        # Optional vector value-head selection for multi-player search.
        # When enabled, callers can request a specific NeuralNetAI value head
        # (e.g. per-player utility) instead of always using head 0.
        vector_env = os.environ.get("RINGRIFT_VECTOR_VALUE_HEAD", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.use_vector_value_head: bool = bool(
            getattr(config, "use_vector_value_head", False)
        ) or vector_env

        # Optional async NN leaf evaluation to overlap CPU tree traversal with
        # GPU inference. Enabled via env var and only when a non-CPU device is used.
        async_env = os.environ.get("RINGRIFT_MCTS_ASYNC_NN_EVAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_async_nn_eval: bool = False
        self._hex_eval_executor: Optional[ThreadPoolExecutor] = None
        if async_env and self.neural_net is not None:
            dev = getattr(self.neural_net, "device", "cpu")
            dev_type = (
                dev if isinstance(dev, str) else getattr(dev, "type", "cpu")
            )
            if dev_type != "cpu":
                self.enable_async_nn_eval = True
                self._hex_eval_executor = ThreadPoolExecutor(max_workers=1)

        # Optional hex-specific encoder and network (used for hex boards).
        self.hex_encoder: Optional["ActionEncoderHex"] = None
        self.hex_model: Optional["HexNeuralNet_v2"] = None
        if self.neural_net is not None:
            try:
                # Lazy imports for hex-specific components
                from .neural_net import (
                    ActionEncoderHex,
                    HexNeuralNet_v2,
                    get_memory_tier,
                )
                # V2 models use larger input channels for richer features.
                self.hex_encoder = ActionEncoderHex()
                memory_tier = get_memory_tier()
                if memory_tier == "low":
                    from .neural_net import HexNeuralNet_v2_Lite
                    self.hex_model = HexNeuralNet_v2_Lite(
                        in_channels=12,
                        global_features=20,
                    )
                else:
                    self.hex_model = HexNeuralNet_v2(
                        in_channels=14,
                        global_features=20,
                    )
            except Exception:
                self.hex_encoder = None
                self.hex_model = None

        # Memory configuration for bounded structures
        self.memory_config = memory_config or MemoryConfig.from_env()

        # Transposition table to cache neural network evaluations
        # Key: state_hash, Value: (value, policy)
        # Entry size estimate: ~400 bytes (MCTS stores larger policies)
        tt_limit = self.memory_config.get_transposition_table_limit_bytes()
        self.transposition_table = BoundedTranspositionTable.from_memory_limit(
            tt_limit,
            entry_size_estimate=400,
        )

        # Dynamic batch sizing (optional)
        self.enable_dynamic_batching = enable_dynamic_batching
        if enable_dynamic_batching:
            self.dynamic_sizer = dynamic_sizer or DynamicBatchSizer(
                memory_config=self.memory_config
            )
        else:
            self.dynamic_sizer = dynamic_sizer

        # Lightweight tree root for incremental search
        self.last_root_lite: Optional[MCTSNodeLite] = None

        # Store the search root BEFORE selecting best child (for training)
        # This allows extraction of visit count distributions for soft policy targets
        self._training_root: Optional[MCTSNode] = None
        self._training_root_lite: Optional[MCTSNodeLite] = None

        # Self-play exploration controls (AlphaZero-style).
        # These are enabled only when AIConfig.self_play is True.
        self.self_play: bool = bool(getattr(config, "self_play", False))
        self.root_dirichlet_alpha: Optional[float] = getattr(
            config, "root_dirichlet_alpha", None
        )
        self.root_noise_fraction: float = float(
            getattr(config, "root_noise_fraction", None) or 0.25
        )
        self.temperature_override: Optional[float] = getattr(
            config, "temperature", None
        )
        self.temperature_cutoff_moves: Optional[int] = getattr(
            config, "temperature_cutoff_moves", None
        )
        self._dirichlet_applied_this_search: bool = False

        # NNUE policy model for move priors (used when no neural net or as fallback)
        # Similar to MinimaxAI's policy ordering but integrated into MCTS priors
        self.nnue_policy_model: Optional["RingRiftNNUEWithPolicy"] = None
        self._pending_nnue_policy_init: bool = False

        # Policy temperature for NNUE priors - higher values flatten the distribution
        # to reduce the influence of low-confidence policy predictions.
        # Tuned via A/B testing: temp=1.5 with mix=0.5 gives 57% vs 30% baseline.
        self.policy_temperature: float = getattr(config, "policy_temperature", 1.5)

        # Prior uniform mix - blends policy priors with uniform distribution.
        # 0.0 = pure policy, 1.0 = pure uniform, 0.5 = 50% policy + 50% uniform.
        # Helps when policy accuracy is low (< 20%). Tuned via grid search.
        self.prior_uniform_mix: float = getattr(config, "prior_uniform_mix", 0.5)

        # Enable NNUE policy priors by default when no neural net is available
        # or explicitly via use_nnue_policy_priors config
        use_nnue_policy_config = getattr(config, "use_nnue_policy_priors", None)
        if use_nnue_policy_config is True:
            self._pending_nnue_policy_init = True
        elif use_nnue_policy_config is None and self.neural_net is None:
            # Auto-enable NNUE policy as fallback when no neural net
            self._pending_nnue_policy_init = True

        if self._pending_nnue_policy_init:
            logger.debug(
                f"MCTSAI(player={player_number}): "
                "NNUE policy priors will be initialized on first move"
            )

        # GPU acceleration for rollout evaluation (default enabled)
        # Uses GPU heuristic evaluator for batch position evaluation
        self._gpu_enabled: bool = not _GPU_MCTS_DISABLE
        self._gpu_available: Optional[bool] = None  # None = not yet checked
        self._gpu_device: Optional["torch.device"] = None
        self._gpu_evaluator: Optional[Any] = None  # GPUHeuristicEvaluator
        self._board_size: Optional[int] = None
        self._board_type_cached: Optional[BoardType] = None
        self._num_players_cached: Optional[int] = None

    def get_last_search_root(self) -> Optional[MCTSNode]:
        """Return the root node from the most recent legacy search.

        This is useful for extracting MCTS visit count distributions
        as soft policy targets during training data generation.

        Returns:
            The MCTSNode root from the last legacy search, or None if
            incremental search was used or no search has been performed.
        """
        return self._training_root

    def get_last_search_root_lite(self) -> Optional[MCTSNodeLite]:
        """Return the root node from the most recent incremental search.

        This is useful for extracting MCTS visit count distributions
        as soft policy targets during training data generation.

        Returns:
            The MCTSNodeLite root from the last incremental search, or None if
            legacy search was used or no search has been performed.
        """
        return self._training_root_lite

    def get_visit_distribution(
        self,
    ) -> Tuple[List[Move], List[float]]:
        """Extract normalized visit count distribution from the last search.

        Returns a tuple of (moves, visit_probabilities) representing the
        MCTS policy based on visit counts. This can be used as soft policy
        targets during training for richer learning signal.

        Returns:
            Tuple of (list of moves, list of visit probabilities) where
            probabilities sum to 1.0. Returns ([], []) if no search has
            been performed or the root has no children.
        """
        # Try incremental root first (default search mode)
        if self._training_root_lite is not None:
            return self._extract_visit_dist_lite(self._training_root_lite)

        # Fall back to legacy root
        if self._training_root is not None:
            return self._extract_visit_dist_legacy(self._training_root)

        return [], []

    def _extract_visit_dist_legacy(
        self, root: MCTSNode
    ) -> Tuple[List[Move], List[float]]:
        """Extract visit distribution from legacy MCTSNode root."""
        if not root.children:
            return [], []

        total_visits = sum(c.visits for c in root.children)
        if total_visits == 0:
            return [], []

        moves: List[Move] = []
        probs: List[float] = []

        for child in root.children:
            if child.move is not None and child.visits > 0:
                moves.append(child.move)
                probs.append(child.visits / total_visits)

        return moves, probs

    def _extract_visit_dist_lite(
        self, root: MCTSNodeLite
    ) -> Tuple[List[Move], List[float]]:
        """Extract visit distribution from incremental MCTSNodeLite root."""
        if not root.children:
            return [], []

        total_visits = sum(c.visits for c in root.children)
        if total_visits == 0:
            return [], []

        moves: List[Move] = []
        probs: List[float] = []

        for child in root.children:
            if child.move is not None and child.visits > 0:
                moves.append(child.move)
                probs.append(child.visits / total_visits)

        return moves, probs

    # ------------------------------------------------------------------
    # Dynamic PUCT / FPU / RAVE tuning (strength-focused).
    # ------------------------------------------------------------------

    def _normalized_entropy(self, priors: List[float]) -> float:
        """Return normalized Shannon entropy of priors in [0, 1]."""
        if not priors:
            return 0.0
        total = float(sum(priors))
        if total <= 0.0:
            return 1.0
        if len(priors) <= 1:
            return 0.0
        inv_total = 1.0 / total
        ent = 0.0
        for p in priors:
            if p <= 0.0:
                continue
            pn = float(p) * inv_total
            ent -= pn * math.log(pn)
        denom = math.log(len(priors))
        if denom <= 0.0:
            return 0.0
        return max(0.0, min(1.0, ent / denom))

    def _dynamic_c_puct(self, parent_visits: int, priors: List[float]) -> float:
        """Compute a dynamic exploration constant based on priors + visits."""
        entropy = self._normalized_entropy(priors)
        visit_term = min(1.0, math.log1p(max(0, int(parent_visits))) / 6.0)

        # Conservative baseline; allow more exploration for high-entropy priors.
        base = 1.0
        cpuct = base + 0.8 * entropy + 0.4 * visit_term
        return float(max(0.25, min(4.0, cpuct)))

    def _rave_k_for_node(self, parent_visits: int, priors: List[float]) -> float:
        """Compute an effective RAVE k that tapers with visits/difficulty."""
        entropy = self._normalized_entropy(priors)

        # Higher difficulties rely more on NN priors; taper RAVE sooner.
        diff = int(getattr(self.config, "difficulty", 5))
        difficulty_scale = max(0.2, 1.0 - 0.12 * max(0, diff - 5))

        visit_scale = 1.0 / (1.0 + max(0, int(parent_visits)) / 200.0)
        entropy_scale = 0.5 + 0.5 * entropy

        base_k = 1000.0
        return float(max(0.0, base_k * difficulty_scale * visit_scale * entropy_scale))

    def _fpu_reduction_for_phase(self, phase: GamePhase) -> float:
        """Phase-aware First-Play Urgency reduction (larger => less widening)."""
        phase_map = {
            GamePhase.RING_PLACEMENT: 0.05,
            GamePhase.MOVEMENT: 0.10,
            GamePhase.CAPTURE: 0.12,
            GamePhase.CHAIN_CAPTURE: 0.12,
            GamePhase.LINE_PROCESSING: 0.16,
            GamePhase.TERRITORY_PROCESSING: 0.20,
            GamePhase.FORCED_ELIMINATION: 0.22,
        }
        return float(phase_map.get(phase, 0.10))

    def _puct_params_for_node(
        self,
        node: Any,
        phase: GamePhase,
    ) -> tuple[float, float, float]:
        children = getattr(node, "children", None) or []
        if not children:
            priors: List[float] = []
        else:
            uniform = 1.0 / max(1, len(children))
            priors = [
                float(getattr(c, "prior", 0.0) or 0.0) or uniform for c in children
            ]

        visits = int(getattr(node, "visits", 0) or 0)
        c_puct = self._dynamic_c_puct(visits, priors)
        rave_k = self._rave_k_for_node(visits, priors)
        fpu_reduction = self._fpu_reduction_for_phase(phase)
        return c_puct, rave_k, fpu_reduction

    # ------------------------------------------------------------------
    # NNUE Policy Model Support
    # ------------------------------------------------------------------

    def _init_nnue_policy_model(self, board_type: BoardType, num_players: int) -> None:
        """Initialize NNUE policy model for move priors.

        This provides policy priors when no neural network is available,
        or can be used as a faster alternative for early MCTS expansions.

        Uses module-level cache to avoid reloading the model for each MCTSAI instance.
        """
        if not self._pending_nnue_policy_init:
            return

        self._pending_nnue_policy_init = False

        # Use cached model loading to avoid repeated disk I/O
        self.nnue_policy_model = _get_cached_nnue_policy(board_type, num_players)

    def _compute_nnue_policy(
        self,
        moves: List[Move],
        state: GameState,
    ) -> Dict[str, float]:
        """Compute policy priors using NNUE policy model.

        Returns a dict mapping move string -> probability (normalized).
        """
        import torch  # Lazy import
        from .nnue_policy import pos_to_flat_index
        from .nnue_features import extract_features_from_gamestate, get_board_size

        if not moves or self.nnue_policy_model is None:
            return {}

        # Extract features from state
        board_type = state.board.type
        board_size = get_board_size(board_type)
        current_player = state.current_player or self.player_number

        features = extract_features_from_gamestate(state, current_player)
        features_tensor = torch.from_numpy(features[None, ...]).float()

        # Get policy logits
        with torch.no_grad():
            _, from_logits, to_logits = self.nnue_policy_model(features_tensor, return_policy=True)
            from_logits = from_logits[0].numpy()  # Shape: (H*W,)
            to_logits = to_logits[0].numpy()  # Shape: (H*W,)

        # Score each move
        center = board_size // 2
        center_idx = center * board_size + center
        move_scores: Dict[str, float] = {}
        max_score = float('-inf')

        for move in moves:
            from_pos = getattr(move, 'from_pos', None)
            if from_pos is None:
                from_idx = center_idx
            else:
                from_idx = pos_to_flat_index(from_pos, board_size, board_type)

            to_pos = getattr(move, 'to', None)
            if to_pos is None:
                to_pos = from_pos
            if to_pos is None:
                to_idx = center_idx
            else:
                to_idx = pos_to_flat_index(to_pos, board_size, board_type)

            score = float(from_logits[from_idx]) + float(to_logits[to_idx])
            move_scores[str(move)] = score
            if score > max_score:
                max_score = score

        # Convert to probabilities using softmax with temperature
        if not move_scores:
            return {}

        # Apply temperature: higher temp -> flatter distribution
        # This reduces the influence of low-confidence policy predictions
        temperature = self.policy_temperature

        # Subtract max for numerical stability then softmax with temperature
        exp_scores = {}
        total_exp = 0.0
        for key, score in move_scores.items():
            exp_val = math.exp((score - max_score) / temperature)
            exp_scores[key] = exp_val
            total_exp += exp_val

        # Normalize
        if total_exp > 0:
            for key in exp_scores:
                exp_scores[key] /= total_exp

        # Mix with uniform distribution to hedge against low-accuracy policy
        mix = self.prior_uniform_mix
        if mix > 0 and len(exp_scores) > 0:
            uniform = 1.0 / len(exp_scores)
            for key in exp_scores:
                exp_scores[key] = (1.0 - mix) * exp_scores[key] + mix * uniform

        return exp_scores

    def clear_search_tree(self) -> None:
        """Clear cached search tree nodes to free memory.

        Call this between games during self-play or soak tests to prevent
        memory accumulation from retained MCTS trees. The training roots
        and incremental roots are cleared, allowing garbage collection
        of the full tree structures.
        """
        self._training_root = None
        self._training_root_lite = None
        self.last_root_lite = None

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select the best move using MCTS."""
        move, _ = self.select_move_and_policy(game_state)
        return move

    def select_move_and_policy(
        self,
        game_state: GameState,
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select the best move using MCTS and return the policy distribution.

        Returns a tuple of (Best move, Policy distribution).

        Routes to incremental (make/unmake) or legacy (immutable) search
        based on the use_incremental_search configuration.
        """
        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None, None

        # Initialize NNUE policy model on first move (lazy initialization)
        if self._pending_nnue_policy_init:
            num_players = infer_num_players(game_state)
            self._init_nnue_policy_model(game_state.board.type, num_players)

        # Initialize GPU for rollout evaluation (lazy initialization)
        if self._gpu_enabled and self._gpu_available is None:
            self._ensure_gpu_initialized(game_state)

        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            policy = {str(swap_move): 1.0}
            return swap_move, policy

        valid_moves = [
            m for m in valid_moves if m.type != MoveType.SWAP_SIDES
        ]

        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0) for m in valid_moves
            }
            return selected, policy

        # Route to incremental or legacy search based on config
        if self.use_incremental_search:
            return self._search_incremental(game_state, valid_moves)
        else:
            return self._search_legacy(game_state, valid_moves)

    # =========================================================================
    # Legacy Search Methods (Immutable State)
    # =========================================================================

    def _search_legacy(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Legacy MCTS search using immutable state cloning via apply_move().

        This is the original implementation preserved for backward
        compatibility and A/B testing against the new incremental search.
        """
        # MCTS parameters - time limit based on difficulty
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            time_limit = 1.0 + (self.config.difficulty * 0.5)

        # Reset self-play root noise flag for this search.
        self._dirichlet_applied_this_search = False

        # Tree Reuse: Check if we have a subtree for the current state
        root: Optional[MCTSNode] = None
        if hasattr(self, "last_root") and self.last_root is not None:
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                for child in self.last_root.children:
                    if _moves_match(child.move, last_move):
                        root = child
                        root.parent = None
                        break

        # CRITICAL: Validate reused subtree is compatible with current phase.
        # Phase transitions (e.g., line_processing → territory_processing)
        # happen atomically during apply_move, so a reused subtree may have
        # children/untried_moves from a different phase. If the valid_moves
        # don't match what the reused tree expects, discard it and start fresh.
        if root is not None:
            valid_move_types = {m.type for m in valid_moves}
            child_move_types = {c.move.type for c in root.children if c.move is not None}
            untried_move_types = {m.type for m in root.untried_moves}
            tree_move_types = child_move_types | untried_move_types

            # If the tree's move types don't overlap with valid moves,
            # the tree is stale (phase transition occurred) - discard it.
            if tree_move_types and not (tree_move_types & valid_move_types):
                root = None  # Discard stale tree

        if root is None:
            root = MCTSNode(game_state)
            root.untried_moves = list(valid_moves)
        else:
            # CRITICAL: When reusing a subtree, the cached children/untried moves
            # may be stale. Filter both lists against the current valid moves to
            # prevent illegal actions from surviving a phase transition.
            valid_move_keys = {_move_key(m) for m in valid_moves}
            valid_move_strings = {str(m) for m in valid_moves}

            root.children = [
                child
                for child in root.children
                if child.move is not None
                and _move_key(child.move) in valid_move_keys
            ]
            root.untried_moves = [
                m
                for m in root.untried_moves
                if _move_key(m) in valid_move_keys
            ]
            if root.policy_map:
                root.policy_map = {
                    k: v for k, v in root.policy_map.items()
                    if k in valid_move_strings
                }

            if not root.children and not root.untried_moves:
                root = None

        if root is None:
            root = MCTSNode(game_state)
            root.untried_moves = list(valid_moves)

        # Progressive widening on large boards benefits from NN priors at the root.
        # Seed them once up-front so early expansions focus on top-prior moves.
        self._maybe_seed_root_priors(root, game_state)
        root.to_move_is_root = game_state.current_player == self.player_number

        board_type = game_state.board.type
        end_time = time.time() + time_limit
        default_batch_size = self._default_leaf_batch_size()
        node_count = 1

        pending_batch: Optional[_EvalBatchLegacy] = None
        pending_future: Optional[Future] = None

        # MCTS implementation with PUCT
        while time.time() < end_time:
            # If a previous async NN evaluation finished, incorporate it now.
            if pending_future is not None and pending_future.done():
                assert pending_batch is not None
                self._finish_leaf_evaluation_legacy(pending_batch, pending_future)
                pending_batch = None
                pending_future = None

            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                batch_size = self.dynamic_sizer.get_optimal_batch_size(
                    node_count
                )
            else:
                batch_size = default_batch_size

            leaves: List[Tuple[MCTSNode, GameState, List[Move]]] = []

            for _ in range(batch_size):
                node = root
                # CRITICAL: Use current game_state, NOT the stale state stored in tree.
                # Phase transitions occur atomically during apply_move, so a reused
                # tree's stored game_state may be from a different phase than the
                # actual current state. Using the current state ensures we start
                # from the correct phase.
                state = game_state
                played_moves: List[Move] = []

                # Selection
                while node.children and (
                    (not node.untried_moves)
                    or (not self._can_expand_node(node, board_type))
                ):
                    node.to_move_is_root = state.current_player == self.player_number
                    c_puct, rave_k, fpu_red = self._puct_params_for_node(
                        node,
                        state.current_phase,
                    )
                    node = node.uct_select_child(
                        c_puct=c_puct,
                        rave_k=rave_k,
                        fpu_reduction=fpu_red,
                    )
                    if node.move is not None:
                        state = self.rules_engine.apply_move(state, node.move)
                        played_moves.append(node.move)

                # Expansion
                if node.untried_moves and self._can_expand_node(node, board_type):
                    m = self._select_untried_move(node, board_type)
                    state = self.rules_engine.apply_move(state, m)

                    prior = None
                    m_key = str(m)
                    if m_key in node.policy_map:
                        prior = node.policy_map[m_key]

                    child = node.add_child(m, state, prior=prior)
                    child.to_move_is_root = state.current_player == self.player_number
                    node = child
                    node_count += 1
                    played_moves.append(m)

                leaves.append((node, state, played_moves))

                if time.time() >= end_time:
                    break

            if not leaves:
                break

            if (
                self.enable_async_nn_eval
                and self.neural_net is not None
            ):
                # Avoid overlapping model usage: finish any pending batch first.
                if pending_future is not None:
                    assert pending_batch is not None
                    self._finish_leaf_evaluation_legacy(pending_batch, pending_future)
                    pending_batch = None
                pending_future = None

                pending_batch, pending_future = self._prepare_leaf_evaluation_legacy(leaves)
                if pending_future is None:
                    self._finish_leaf_evaluation_legacy(pending_batch, None)
                    pending_batch = None
            else:
                # Synchronous evaluation Phase
                self._evaluate_leaves_legacy(leaves, root)

        # Ensure any pending eval is incorporated before selecting best move.
        if pending_future is not None and pending_batch is not None:
            self._finish_leaf_evaluation_legacy(pending_batch, pending_future)

        # Record memory sample if dynamic batching is enabled
        if self.enable_dynamic_batching and self.dynamic_sizer is not None:
            self.dynamic_sizer.record_memory_sample(node_count)

        self._log_stats()

        # Store root for training (visit count extraction) BEFORE selecting best child
        self._training_root = root
        self._training_root_lite = None  # Clear incremental root

        # Select best move based on visits
        return self._select_best_move_legacy(root, valid_moves, game_state)

    def _evaluate_leaves_legacy(
        self,
        leaves: List[Tuple[MCTSNode, GameState, List[Move]]],
        root: MCTSNode,
    ) -> None:
        """Evaluate leaf nodes using neural network or heuristic rollout."""
        if self.neural_net:
            try:
                states = [leaf[1] for leaf in leaves]

                cached_results: List[Tuple[int, float, Any]] = []
                uncached_indices: List[int] = []
                uncached_states: List[GameState] = []

                for i, state in enumerate(states):
                    state_hash = state.zobrist_hash or 0
                    cached = self.transposition_table.get(state_hash)
                    if cached is not None:
                        cached_results.append((i, cached[0], cached[1]))
                    else:
                        uncached_indices.append(i)
                        uncached_states.append(state)

                values: List[float] = [0.0] * len(states)
                policies: List[Any] = [None] * len(states)

                for idx, val, pol in cached_results:
                    values[idx] = val
                    policies[idx] = pol

                use_hex_nn = (
                    self.hex_model is not None
                    and self.hex_encoder is not None
                    and states
                    and states[0].board.type == BoardType.HEXAGONAL
                )
                use_vector_head = (
                    self.use_vector_value_head
                    and not use_hex_nn
                    and bool(states)
                    and infer_num_players(states[0]) > 2
                )
                value_head = (self.player_number - 1) if use_vector_head else None

                if uncached_states:
                    if use_hex_nn:
                        eval_values, eval_policies = self._evaluate_hex_batch(
                            uncached_states
                        )
                    else:
                        eval_values, eval_policies = (
                            self.nn_batcher.evaluate(
                                uncached_states,
                                value_head=value_head,
                            )
                            if self.nn_batcher is not None
                            else self.neural_net.evaluate_batch(
                                uncached_states,
                                value_head=value_head,
                            )
                        )

                    for j, orig_idx in enumerate(uncached_indices):
                        values[orig_idx] = eval_values[j]
                        policies[orig_idx] = eval_policies[j]

                        state_hash = uncached_states[j].zobrist_hash or 0
                        self.transposition_table.put(
                            state_hash,
                            (eval_values[j], eval_policies[j]),
                        )

                # Process results
                for i in range(len(leaves)):
                    value = values[i]
                    policy = policies[i]
                    node, state, played_moves = leaves[i]

                    if policy is None:
                        continue

                    self._update_node_policy_legacy(
                        node, state, policy, bool(use_hex_nn)
                    )

                    # Backpropagation: store values from the perspective of the
                    # side-to-move at each node (root player vs opponent coalition).
                    #
                    # For 2p this reduces to standard negamax; for 3p/4p we only
                    # flip sign when the turn switches between the root player
                    # and any opponent (coalition), not on every ply.
                    players_by_depth = [m.player for m in played_moves]
                    players_by_depth.append(state.current_player)
                    side_is_root_by_depth = [
                        (p == self.player_number) for p in players_by_depth
                    ]

                    raw_val = float(value) if value is not None else 0.0
                    if use_vector_head:
                        current_val = (
                            raw_val
                            if state.current_player == self.player_number
                            else -raw_val
                        )
                    else:
                        current_val = raw_val
                    depth_idx = len(played_moves)
                    curr_node: Optional[MCTSNode] = node

                    while curr_node is not None:
                        curr_node.update(current_val, played_moves)
                        parent = curr_node.parent
                        if parent is None:
                            break
                        parent_depth = depth_idx - 1
                        if (
                            side_is_root_by_depth[depth_idx]
                            != side_is_root_by_depth[parent_depth]
                        ):
                            current_val = -current_val
                        curr_node = parent
                        depth_idx = parent_depth

                return

            except Exception:
                # Common causes: missing/incompatible NN checkpoints. Degrade
                # to heuristic rollouts instead of failing the whole request.
                if self.require_neural_net:
                    logger.error(
                        "MCTS neural evaluation failed with RINGRIFT_REQUIRE_NEURAL_NET=1; "
                        "raising instead of falling back to heuristic rollouts",
                        exc_info=True,
                    )
                    raise
                logger.warning(
                    "MCTS neural evaluation failed; falling back to heuristic rollouts",
                    exc_info=True,
                )
                self.neural_net = None

        # Fallback to Heuristic Rollout
        for node, state, played_moves in leaves:
            result = self._heuristic_rollout_legacy(state)
            players_by_depth = [m.player for m in played_moves]
            players_by_depth.append(state.current_player)
            side_is_root_by_depth = [
                (p == self.player_number) for p in players_by_depth
            ]

            # Rollout evaluation returns a root-perspective value; convert it
            # to the leaf side-to-move (root vs coalition).
            current_val = (
                float(result)
                if state.current_player == self.player_number
                else -float(result)
            )
            depth_idx = len(played_moves)
            curr_node: Optional[MCTSNode] = node
            while curr_node is not None:
                curr_node.update(current_val, played_moves)
                parent = curr_node.parent
                if parent is None:
                    break
                parent_depth = depth_idx - 1
                if (
                    side_is_root_by_depth[depth_idx]
                    != side_is_root_by_depth[parent_depth]
                ):
                    current_val = -current_val
                curr_node = parent
                depth_idx = parent_depth

    def _prepare_leaf_evaluation_legacy(
        self,
        leaves: List[Tuple[MCTSNode, GameState, List[Move]]],
    ) -> tuple[_EvalBatchLegacy, Optional[Future]]:
        states = [leaf[1] for leaf in leaves]

        cached_results: List[Tuple[int, float, Any]] = []
        uncached_indices: List[int] = []
        uncached_states: List[GameState] = []

        for i, state in enumerate(states):
            state_hash = state.zobrist_hash or 0
            cached = self.transposition_table.get(state_hash)
            if cached is not None:
                cached_results.append((i, cached[0], cached[1]))
            else:
                uncached_indices.append(i)
                uncached_states.append(state)

        use_hex_nn = (
            self.hex_model is not None
            and self.hex_encoder is not None
            and states
            and states[0].board.type == BoardType.HEXAGONAL
        )
        use_vector_head = (
            self.use_vector_value_head
            and not use_hex_nn
            and bool(states)
            and infer_num_players(states[0]) > 2
        )
        value_head = (self.player_number - 1) if use_vector_head else None

        future: Optional[Future] = None
        if uncached_states:
            if use_hex_nn:
                if self._hex_eval_executor is not None:
                    future = self._hex_eval_executor.submit(
                        self._evaluate_hex_batch, uncached_states
                    )
                else:
                    fut: Future = Future()
                    fut.set_result(self._evaluate_hex_batch(uncached_states))
                    future = fut
            else:
                if self.nn_batcher is not None:
                    future = self.nn_batcher.submit(
                        uncached_states,
                        value_head=value_head,
                    )
                else:
                    fut = Future()
                    fut.set_result(
                        self.neural_net.evaluate_batch(  # type: ignore[union-attr]
                            uncached_states,
                            value_head=value_head,
                        )
                    )
                    future = fut

        batch = _EvalBatchLegacy(
            leaves=leaves,
            states=states,
            cached_results=cached_results,
            uncached_indices=uncached_indices,
            uncached_states=uncached_states,
            use_hex_nn=bool(use_hex_nn),
        )
        return batch, future

    def _finish_leaf_evaluation_legacy(
        self,
        batch: _EvalBatchLegacy,
        future: Optional[Future],
    ) -> None:
        states = batch.states
        use_vector_head = (
            self.use_vector_value_head
            and not batch.use_hex_nn
            and bool(states)
            and infer_num_players(states[0]) > 2
        )
        values: List[float] = [0.0] * len(states)
        policies: List[Any] = [None] * len(states)

        for idx, val, pol in batch.cached_results:
            values[idx] = val
            policies[idx] = pol

        try:
            if future is not None:
                eval_values, eval_policies = future.result()
                for j, orig_idx in enumerate(batch.uncached_indices):
                    values[orig_idx] = eval_values[j]
                    policies[orig_idx] = eval_policies[j]

                    state_hash = batch.uncached_states[j].zobrist_hash or 0
                    self.transposition_table.put(
                        state_hash,
                        (eval_values[j], eval_policies[j]),
                    )

            for i in range(len(batch.leaves)):
                value = values[i]
                policy = policies[i]
                node, state, played_moves = batch.leaves[i]

                if policy is None:
                    continue

                self._update_node_policy_legacy(
                    node, state, policy, bool(batch.use_hex_nn)
                )

                players_by_depth = [m.player for m in played_moves]
                players_by_depth.append(state.current_player)
                side_is_root_by_depth = [
                    (p == self.player_number) for p in players_by_depth
                ]

                raw_val = float(value) if value is not None else 0.0
                if use_vector_head:
                    current_val = (
                        raw_val
                        if state.current_player == self.player_number
                        else -raw_val
                    )
                else:
                    current_val = raw_val
                depth_idx = len(played_moves)
                curr_node: Optional[MCTSNode] = node

                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    parent = curr_node.parent
                    if parent is None:
                        break
                    parent_depth = depth_idx - 1
                    if (
                        side_is_root_by_depth[depth_idx]
                        != side_is_root_by_depth[parent_depth]
                    ):
                        current_val = -current_val
                    curr_node = parent
                    depth_idx = parent_depth
        except Exception:
            if self.require_neural_net:
                logger.error(
                    "Async MCTS neural evaluation failed with RINGRIFT_REQUIRE_NEURAL_NET=1; "
                    "raising instead of falling back to heuristic rollouts",
                    exc_info=True,
                )
                raise
            logger.warning(
                "Async MCTS neural evaluation failed; falling back to heuristic rollouts",
                exc_info=True,
            )
            self.neural_net = None
            for node, state, played_moves in batch.leaves:
                result = self._heuristic_rollout_legacy(state)
                players_by_depth = [m.player for m in played_moves]
                players_by_depth.append(state.current_player)
                side_is_root_by_depth = [
                    (p == self.player_number) for p in players_by_depth
                ]

                current_val = (
                    float(result)
                    if state.current_player == self.player_number
                    else -float(result)
                )
                depth_idx = len(played_moves)
                curr_node: Optional[MCTSNode] = node
                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    parent = curr_node.parent
                    if parent is None:
                        break
                    parent_depth = depth_idx - 1
                    if (
                        side_is_root_by_depth[depth_idx]
                        != side_is_root_by_depth[parent_depth]
                    ):
                        current_val = -current_val
                    curr_node = parent
                    depth_idx = parent_depth

    def _prepare_leaf_evaluation_incremental(
        self,
        leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]],
        mutable_state: MutableGameState,
    ) -> tuple[_EvalBatchIncremental, Optional[Future]]:
        states: List[GameState] = []
        for node, path_undos, played_moves in leaves:
            for undo in path_undos:
                mutable_state.make_move(undo.move)
            immutable = mutable_state.to_immutable()
            states.append(immutable)
            for undo in reversed(path_undos):
                mutable_state.unmake_move(undo)

        cached_results: List[Tuple[int, float, Any]] = []
        uncached_indices: List[int] = []
        uncached_states: List[GameState] = []

        for i, state in enumerate(states):
            state_hash = state.zobrist_hash or 0
            cached = self.transposition_table.get(state_hash)
            if cached is not None:
                cached_results.append((i, cached[0], cached[1]))
            else:
                uncached_indices.append(i)
                uncached_states.append(state)

        use_hex_nn = (
            self.hex_model is not None
            and self.hex_encoder is not None
            and states
            and states[0].board.type == BoardType.HEXAGONAL
        )
        use_vector_head = (
            self.use_vector_value_head
            and not use_hex_nn
            and bool(states)
            and infer_num_players(states[0]) > 2
        )
        value_head = (self.player_number - 1) if use_vector_head else None

        future: Optional[Future] = None
        if uncached_states:
            if use_hex_nn:
                if self._hex_eval_executor is not None:
                    future = self._hex_eval_executor.submit(
                        self._evaluate_hex_batch, uncached_states
                    )
                else:
                    fut: Future = Future()
                    fut.set_result(self._evaluate_hex_batch(uncached_states))
                    future = fut
            else:
                if self.nn_batcher is not None:
                    future = self.nn_batcher.submit(
                        uncached_states,
                        value_head=value_head,
                    )
                else:
                    fut = Future()
                    fut.set_result(
                        self.neural_net.evaluate_batch(  # type: ignore[union-attr]
                            uncached_states,
                            value_head=value_head,
                        )
                    )
                    future = fut

        batch = _EvalBatchIncremental(
            leaves=leaves,
            states=states,
            cached_results=cached_results,
            uncached_indices=uncached_indices,
            uncached_states=uncached_states,
            use_hex_nn=bool(use_hex_nn),
        )
        return batch, future

    def _finish_leaf_evaluation_incremental(
        self,
        batch: _EvalBatchIncremental,
        future: Optional[Future],
        mutable_state: MutableGameState,
    ) -> None:
        states = batch.states
        use_vector_head = (
            self.use_vector_value_head
            and not batch.use_hex_nn
            and bool(states)
            and infer_num_players(states[0]) > 2
        )
        values: List[float] = [0.0] * len(states)
        policies: List[Any] = [None] * len(states)

        for idx, val, pol in batch.cached_results:
            values[idx] = val
            policies[idx] = pol

        try:
            if future is not None:
                eval_values, eval_policies = future.result()
                for j, orig_idx in enumerate(batch.uncached_indices):
                    values[orig_idx] = eval_values[j]
                    policies[orig_idx] = eval_policies[j]

                    state_hash = batch.uncached_states[j].zobrist_hash or 0
                    self.transposition_table.put(
                        state_hash,
                        (eval_values[j], eval_policies[j]),
                    )

            for i, (node, path_undos, played_moves) in enumerate(batch.leaves):
                value = values[i]
                policy = policies[i]
                state = states[i]

                if policy is not None:
                    self._update_node_policy_lite(
                        node, state, policy, bool(batch.use_hex_nn)
                    )

                players_by_depth = [u.prev_player for u in path_undos]
                players_by_depth.append(state.current_player)
                side_is_root_by_depth = [
                    (p == self.player_number) for p in players_by_depth
                ]

                raw_val = float(value) if value is not None else 0.0
                if use_vector_head:
                    current_val = (
                        raw_val
                        if state.current_player == self.player_number
                        else -raw_val
                    )
                else:
                    current_val = raw_val
                depth_idx = len(path_undos)
                curr_node: Optional[MCTSNodeLite] = node

                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    parent = curr_node.parent
                    if parent is None:
                        break
                    parent_depth = depth_idx - 1
                    if (
                        side_is_root_by_depth[depth_idx]
                        != side_is_root_by_depth[parent_depth]
                    ):
                        current_val = -current_val
                    curr_node = parent
                    depth_idx = parent_depth
        except Exception:
            if self.require_neural_net:
                logger.error(
                    "Async MCTS neural evaluation failed with RINGRIFT_REQUIRE_NEURAL_NET=1; "
                    "raising instead of falling back to heuristic rollouts",
                    exc_info=True,
                )
                raise
            logger.warning(
                "Async MCTS neural evaluation failed; falling back to heuristic rollouts",
                exc_info=True,
            )
            self.neural_net = None
            for node, path_undos, played_moves in batch.leaves:
                for undo in path_undos:
                    mutable_state.make_move(undo.move)
                state = mutable_state.to_immutable()
                for undo in reversed(path_undos):
                    mutable_state.unmake_move(undo)

                result = self._heuristic_rollout_legacy(state)
                players_by_depth = [m.player for m in played_moves]
                players_by_depth.append(state.current_player)
                side_is_root_by_depth = [
                    (p == self.player_number) for p in players_by_depth
                ]

                current_val = (
                    float(result)
                    if state.current_player == self.player_number
                    else -float(result)
                )
                depth_idx = len(played_moves)
                curr_node: Optional[MCTSNodeLite] = node
                while curr_node is not None:
                    curr_node.update(current_val, played_moves)
                    parent = curr_node.parent
                    if parent is None:
                        break
                    parent_depth = depth_idx - 1
                    if (
                        side_is_root_by_depth[depth_idx]
                        != side_is_root_by_depth[parent_depth]
                    ):
                        current_val = -current_val
                    curr_node = parent
                    depth_idx = parent_depth

    def _evaluate_hex_batch(
        self, states: List[GameState]
    ) -> Tuple[List[float], List[Any]]:
        """Evaluate a batch of hex board states."""
        import torch  # Lazy import
        from .neural_net import HexNeuralNet_v2  # Lazy import for cast

        feature_batches = []
        globals_batches = []
        nn = self.neural_net  # type assertion for Pylance
        assert nn is not None
        for s in states:
            feats, globs = nn._extract_features(s)
            feature_batches.append(feats)
            globals_batches.append(globs)

        tensor_input = torch.FloatTensor(np.array(feature_batches))
        globals_input = torch.FloatTensor(np.array(globals_batches))

        hex_model = cast(HexNeuralNet_v2, self.hex_model)
        with torch.no_grad():
            values_tensor, policy_logits = hex_model(
                tensor_input, globals_input, hex_mask=None
            )
            policy_probs = torch.softmax(policy_logits, dim=1)

        values_np = values_tensor.detach().cpu().numpy()
        if values_np.ndim == 2:
            eval_values = values_np[:, 0].astype(np.float32).tolist()
        else:
            eval_values = values_np.reshape(values_np.shape[0]).astype(np.float32).tolist()
        eval_policies = policy_probs.cpu().numpy()

        return eval_values, list(eval_policies)

    def _update_node_policy_legacy(
        self,
        node: MCTSNode,
        state: GameState,
        policy: Any,
        use_hex_nn: bool,
    ) -> None:
        """Update node policy priors from neural network output."""
        from .neural_net import INVALID_MOVE_INDEX  # Lazy import

        # Use the host-level RulesEngine surface so that bookkeeping moves
        # (no_*_action / forced_elimination) are surfaced when required.
        valid_moves_state = self.rules_engine.get_valid_moves(
            state,
            state.current_player,
        )
        if not valid_moves_state:
            return

        existing_child_keys = (
            {_move_key(c.move) for c in node.children if c.move is not None}
            if node.children
            else set()
        )
        node.policy_map = {}

        total_prob = 0.0
        for move in valid_moves_state:
            if use_hex_nn and self.hex_encoder is not None:
                idx = self.hex_encoder.encode_move(move, state.board)
            elif self.neural_net is not None:
                idx = self.neural_net.encode_move(move, state.board)
            else:
                idx = INVALID_MOVE_INDEX

            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy):
                prob = float(cast(Any, policy)[idx])
                node.policy_map[str(move)] = prob
                total_prob += prob

        if total_prob > 0:
            for move_key in node.policy_map:
                node.policy_map[move_key] /= total_prob
        elif self.nnue_policy_model is not None:
            # Fallback to NNUE policy when neural network policy unavailable
            nnue_policy = self._compute_nnue_policy(valid_moves_state, state)
            if nnue_policy:
                node.policy_map = nnue_policy
            else:
                uniform = 1.0 / len(valid_moves_state)
                for move in valid_moves_state:
                    node.policy_map[str(move)] = uniform
        else:
            uniform = 1.0 / len(valid_moves_state)
            for move in valid_moves_state:
                node.policy_map[str(move)] = uniform

        # Keep untried moves disjoint from already-expanded children (important
        # for root prior seeding + tree reuse, and for move types where
        # (type,to) is not unique, e.g. multi-ring placements or choose-line
        # segments).
        node.untried_moves = [
            m for m in valid_moves_state if _move_key(m) not in existing_child_keys
        ]

        # Apply Dirichlet noise only at root during self-play.
        self._maybe_apply_root_dirichlet_noise(node, state.board.type)

        # Order untried moves by priors when policy_map is available.
        # For progressive widening this focuses on top-prior moves; for other
        # boards it still benefits from policy guidance during expansion.
        if node.policy_map:
            node.untried_moves.sort(
                key=lambda m: node.policy_map.get(str(m), 0.0),
                reverse=True,
            )

    def _heuristic_rollout_legacy(self, state: GameState) -> float:
        """Perform heuristic-guided rollout simulation."""
        rollout_depth = 3
        rollout_state = state

        for _ in range(rollout_depth):
            if rollout_state.game_status == "completed":
                break

            moves = self.rules_engine.get_valid_moves(
                rollout_state,
                rollout_state.current_player,
            )
            if not moves:
                break

            weights = self._compute_move_weights(moves, rollout_state)
            selected_move = self.rng.choices(moves, weights=weights, k=1)[0]
            rollout_state = self.rules_engine.apply_move(
                rollout_state,
                selected_move,
            )

        return self.evaluate_position(rollout_state)

    def _compute_move_weights(
        self, moves: List[Move], state: GameState
    ) -> List[float]:
        """Compute weights for move selection in rollout."""
        weights = []
        for m in moves:
            w = 1.0
            if m.type == "territory_claim":
                w = 100.0
            elif m.type == "line_formation":
                w = 50.0
            elif m.type == "chain_capture":
                w = 20.0
            elif m.type == "overtaking_capture":
                w = 10.0
            elif m.type == "move_stack":
                to_key = m.to.to_key()
                if to_key in state.board.stacks:
                    w = 5.0
                else:
                    w = 2.0
            elif m.type == "place_ring":
                w = 1.5
            weights.append(w)
        return weights

    def _select_best_move_legacy(
        self,
        root: MCTSNode,
        valid_moves: List[Move],
        game_state: GameState,
    ) -> Tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select best move from legacy tree based on visit counts."""
        # Build a set of valid move signatures for efficient lookup.
        # Include player attribute to catch stale moves from tree reuse where
        # the acting player or their resources (e.g., rings_in_hand) changed.
        valid_moves_set = {_move_key(m) for m in valid_moves}

        # Filter children to only those with currently valid moves.
        # This is critical for tree reuse scenarios where children were
        # expanded when a different player had the turn or when player
        # resources have changed (e.g., rings_in_hand depleted).
        valid_children = [
            c for c in root.children
            if c.move is not None and _move_key(c.move) in valid_moves_set
        ]

        if valid_children:
            total_visits = sum(c.visits for c in valid_children)
            policy: Dict[str, float] = {}
            if total_visits > 0:
                for child in valid_children:
                    policy[str(child.move)] = child.visits / total_visits
            else:
                uniform = 1.0 / len(valid_children)
                for child in valid_children:
                    policy[str(child.move)] = uniform

            best_child = max(valid_children, key=lambda c: c.visits)
            selected_child = best_child
            if self.self_play:
                temperature = self._get_selfplay_temperature(game_state)
                selected_child = self._sample_child_by_temperature(
                    valid_children, temperature
                )
            selected = selected_child.move

            self.last_root = selected_child
            self.last_root.parent = None

            # Extra validation: ensure selected move matches a valid move.
            # This should always pass given the filtering above, but we keep
            # it as a safety net.
            if selected is not None:
                is_valid = _move_key(selected) in valid_moves_set
                if not is_valid:
                    logger.warning(
                        "MCTS legacy selected invalid move %s "
                        "(not in valid_moves), falling back to random",
                        selected.type if selected else None,
                    )
                    selected = self.get_random_element(valid_moves)
                    policy = {
                        str(m): (1.0 if m == selected else 0.0)
                        for m in valid_moves
                    }
                    self.last_root = None  # Discard stale tree
        else:
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0)
                for m in valid_moves
            }

        self.move_count += 1
        return selected, policy

    # =========================================================================
    # Incremental Search Methods (Make/Unmake Pattern)
    # =========================================================================

    def _search_incremental(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Incremental MCTS search using make/unmake on MutableGameState.

        This provides significant speedup by avoiding object allocation
        overhead during tree search. State is modified in-place and restored
        using MoveUndo tokens.
        """
        # MCTS parameters - time limit based on difficulty
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            time_limit = 1.0 + (self.config.difficulty * 0.5)

        # Reset self-play root noise flag for this search.
        self._dirichlet_applied_this_search = False

        # Create mutable state once for the entire search
        mutable_state = MutableGameState.from_immutable(game_state)
        board_type = mutable_state.board_type

        # Tree Reuse: Check if we have a subtree for the current state
        root: Optional[MCTSNodeLite] = None
        if self.last_root_lite is not None:
            if game_state.move_history:
                last_move = game_state.move_history[-1]
                for child in self.last_root_lite.children:
                    if child.move is not None and _moves_match(child.move, last_move):
                        root = child
                        root.parent = None
                        break

        # CRITICAL: Validate reused subtree is compatible with current phase.
        # Phase transitions (e.g., line_processing → territory_processing)
        # happen atomically during apply_move, so a reused subtree may have
        # children/untried_moves from a different phase. If the valid_moves
        # don't match what the reused tree expects, discard it and start fresh.
        if root is not None:
            # Check if any child moves match the current valid_moves
            valid_move_types = {m.type for m in valid_moves}
            child_move_types = {c.move.type for c in root.children if c.move is not None}
            untried_move_types = {m.type for m in root.untried_moves}
            tree_move_types = child_move_types | untried_move_types

            # If the tree's move types don't overlap with valid moves,
            # the tree is stale (phase transition occurred) - discard it.
            if tree_move_types and not (tree_move_types & valid_move_types):
                root = None  # Discard stale tree

        if root is None:
            root = MCTSNodeLite()
            root.untried_moves = list(valid_moves)
        else:
            # CRITICAL: When reusing a subtree, prune stale children and untried
            # moves against the current valid move set.
            valid_move_keys = {_move_key(m) for m in valid_moves}
            valid_move_strings = {str(m) for m in valid_moves}

            root.children = [
                child
                for child in root.children
                if child.move is not None
                and _move_key(child.move) in valid_move_keys
            ]
            root.untried_moves = [
                m
                for m in root.untried_moves
                if _move_key(m) in valid_move_keys
            ]
            if root.policy_map:
                root.policy_map = {
                    k: v for k, v in root.policy_map.items()
                    if k in valid_move_strings
                }

            if not root.children and not root.untried_moves:
                root = None

        if root is None:
            root = MCTSNodeLite()
            root.untried_moves = list(valid_moves)

        # Seed NN priors at root for large boards to align with progressive widening.
        self._maybe_seed_root_priors(root, game_state)
        root.to_move_is_root = game_state.current_player == self.player_number

        end_time = time.time() + time_limit
        default_batch_size = self._default_leaf_batch_size()
        node_count = 1

        pending_batch: Optional[_EvalBatchIncremental] = None
        pending_future: Optional[Future] = None

        # MCTS implementation with PUCT using make/unmake
        while time.time() < end_time:
            if pending_future is not None and pending_future.done():
                assert pending_batch is not None
                self._finish_leaf_evaluation_incremental(
                    pending_batch, pending_future, mutable_state
                )
                pending_batch = None
                pending_future = None

            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                batch_size = self.dynamic_sizer.get_optimal_batch_size(
                    node_count
                )
            else:
                batch_size = default_batch_size

            # Collect leaves for batch evaluation
            leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]] = []

            for _ in range(batch_size):
                # Selection phase with make_move tracking
                node, path_undos, played_moves = self._select_with_mutable(
                    root, mutable_state
                )

                # Expansion phase
                if node.untried_moves and self._can_expand_node(node, board_type):
                    m = self._select_untried_move(node, board_type)
                    undo = mutable_state.make_move(m)
                    path_undos.append(undo)

                    prior = node.policy_map.get(str(m))
                    child = node.add_child(
                        m,
                        prior=prior,
                        to_move_is_root=(
                            mutable_state.current_player == self.player_number
                        ),
                    )
                    node = child
                    node_count += 1
                    played_moves.append(m)

                leaves.append((node, path_undos, played_moves))

                # Restore state for next iteration in batch
                for undo in reversed(path_undos):
                    mutable_state.unmake_move(undo)

                if time.time() >= end_time:
                    break

            if not leaves:
                break

            if (
                self.enable_async_nn_eval
                and self.neural_net is not None
            ):
                if pending_future is not None:
                    assert pending_batch is not None
                    self._finish_leaf_evaluation_incremental(
                        pending_batch, pending_future, mutable_state
                    )
                    pending_batch = None
                    pending_future = None

                pending_batch, pending_future = (
                    self._prepare_leaf_evaluation_incremental(
                        leaves, mutable_state
                    )
                )
                if pending_future is None:
                    self._finish_leaf_evaluation_incremental(
                        pending_batch, None, mutable_state
                    )
                    pending_batch = None
            else:
                # Evaluation Phase - replay paths and evaluate
                self._evaluate_leaves_incremental(
                    leaves, mutable_state, root
                )

        # Ensure any pending eval is incorporated before final stats/policy.
        if pending_future is not None and pending_batch is not None:
            self._finish_leaf_evaluation_incremental(
                pending_batch, pending_future, mutable_state
            )
            pending_batch = None
            pending_future = None

        # Record memory sample if dynamic batching is enabled
        if self.enable_dynamic_batching and self.dynamic_sizer is not None:
            self.dynamic_sizer.record_memory_sample(node_count)

        self._log_stats()

        # Store root for training (visit count extraction) BEFORE selecting best child
        self._training_root_lite = root
        self._training_root = None  # Clear legacy root

        # Select best move based on visits
        return self._select_best_move_incremental(root, valid_moves, game_state)

    def _select_with_mutable(
        self,
        node: MCTSNodeLite,
        mutable_state: MutableGameState,
    ) -> Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]:
        """Traverse tree using make_move, tracking path for unmake.

        Returns:
            Tuple of (selected node, undo tokens, played moves)
        """
        path_undos: List[MoveUndo] = []
        played_moves: List[Move] = []
        board_type = mutable_state.board_type

        # Selection - traverse to leaf
        while node.children and (
            not node.untried_moves
            or not self._can_expand_node(node, board_type)
        ):
            node.to_move_is_root = mutable_state.current_player == self.player_number
            c_puct, rave_k, fpu_red = self._puct_params_for_node(
                node,
                mutable_state.current_phase,
            )
            node = node.uct_select_child(
                c_puct=c_puct,
                rave_k=rave_k,
                fpu_reduction=fpu_red,
            )
            if node.move is not None:
                undo = mutable_state.make_move(node.move)
                path_undos.append(undo)
                played_moves.append(node.move)

        return node, path_undos, played_moves

    def _evaluate_leaves_incremental(
        self,
        leaves: List[Tuple[MCTSNodeLite, List[MoveUndo], List[Move]]],
        mutable_state: MutableGameState,
        root: MCTSNodeLite,
    ) -> None:
        """Evaluate leaf nodes using make/unmake pattern.

        For each leaf, we replay the path to reach that state,
        evaluate, then unmake to return to root.
        """
        if self.neural_net:
            try:
                # Batch evaluation - collect states
                states: List[GameState] = []
                for node, path_undos, played_moves in leaves:
                    # Replay path to reach this leaf
                    for undo in path_undos:
                        mutable_state.make_move(undo.move)

                    # Convert to immutable for evaluation
                    immutable = mutable_state.to_immutable()
                    states.append(immutable)

                    # Unmake to return to root
                    for undo in reversed(path_undos):
                        mutable_state.unmake_move(undo)

                # Check transposition table
                cached_results: List[Tuple[int, float, Any]] = []
                uncached_indices: List[int] = []
                uncached_states: List[GameState] = []

                for i, state in enumerate(states):
                    state_hash = state.zobrist_hash or 0
                    cached = self.transposition_table.get(state_hash)
                    if cached is not None:
                        cached_results.append((i, cached[0], cached[1]))
                    else:
                        uncached_indices.append(i)
                        uncached_states.append(state)

                values: List[float] = [0.0] * len(states)
                policies: List[Any] = [None] * len(states)

                for idx, val, pol in cached_results:
                    values[idx] = val
                    policies[idx] = pol

                use_hex_nn = (
                    self.hex_model is not None
                    and self.hex_encoder is not None
                    and states
                    and states[0].board.type == BoardType.HEXAGONAL
                )
                use_vector_head = (
                    self.use_vector_value_head
                    and not use_hex_nn
                    and bool(states)
                    and infer_num_players(states[0]) > 2
                )
                value_head = (self.player_number - 1) if use_vector_head else None

                if uncached_states:
                    if use_hex_nn:
                        eval_values, eval_policies = self._evaluate_hex_batch(
                            uncached_states
                        )
                    else:
                        eval_values, eval_policies = (
                            self.nn_batcher.evaluate(
                                uncached_states,
                                value_head=value_head,
                            )
                            if self.nn_batcher is not None
                            else self.neural_net.evaluate_batch(
                                uncached_states,
                                value_head=value_head,
                            )
                        )

                    for j, orig_idx in enumerate(uncached_indices):
                        values[orig_idx] = eval_values[j]
                        policies[orig_idx] = eval_policies[j]

                        state_hash = uncached_states[j].zobrist_hash or 0
                        self.transposition_table.put(
                            state_hash,
                            (eval_values[j], eval_policies[j]),
                        )

                # Process results and backpropagate
                for i, (node, path_undos, played_moves) in enumerate(leaves):
                    value = values[i]
                    policy = policies[i]
                    state = states[i]

                    if policy is not None:
                        self._update_node_policy_lite(
                            node, state, policy, bool(use_hex_nn)
                        )

                    # Backpropagation: store values from the perspective of the
                    # side-to-move at each node (root player vs opponent coalition).
                    players_by_depth = [u.prev_player for u in path_undos]
                    players_by_depth.append(state.current_player)
                    side_is_root_by_depth = [
                        (p == self.player_number) for p in players_by_depth
                    ]

                    raw_val = float(value) if value is not None else 0.0
                    if use_vector_head:
                        current_val = (
                            raw_val
                            if state.current_player == self.player_number
                            else -raw_val
                        )
                    else:
                        current_val = raw_val
                    depth_idx = len(path_undos)
                    curr_node: Optional[MCTSNodeLite] = node

                    while curr_node is not None:
                        curr_node.update(current_val, played_moves)
                        parent = curr_node.parent
                        if parent is None:
                            break
                        parent_depth = depth_idx - 1
                        if (
                            side_is_root_by_depth[depth_idx]
                            != side_is_root_by_depth[parent_depth]
                        ):
                            current_val = -current_val
                        curr_node = parent
                        depth_idx = parent_depth

                return

            except Exception:
                logger.warning(
                    "MCTS neural evaluation failed; falling back to heuristic rollouts",
                    exc_info=True,
                )
                if self.require_neural_net:
                    logger.error(
                        "RINGRIFT_REQUIRE_NEURAL_NET=1 set; raising instead of falling back",
                        exc_info=True,
                    )
                    raise
                self.neural_net = None

        # Fallback to Heuristic Rollout with make/unmake
        for node, path_undos, played_moves in leaves:
            # Replay path to reach this leaf
            for undo in path_undos:
                mutable_state.make_move(undo.move)

            leaf_player = mutable_state.current_player

            # Perform rollout
            result = self._heuristic_rollout_mutable(mutable_state)

            # Unmake to return to root
            for undo in reversed(path_undos):
                mutable_state.unmake_move(undo)

            # Convert the resulting root-perspective rollout value into the
            # leaf side-to-move (root vs coalition) perspective.
            players_by_depth = [u.prev_player for u in path_undos]
            players_by_depth.append(leaf_player)
            side_is_root_by_depth = [
                (p == self.player_number) for p in players_by_depth
            ]

            current_val = (
                float(result)
                if leaf_player == self.player_number
                else -float(result)
            )
            depth_idx = len(path_undos)
            curr_node: Optional[MCTSNodeLite] = node
            while curr_node is not None:
                curr_node.update(current_val, played_moves)
                parent = curr_node.parent
                if parent is None:
                    break
                parent_depth = depth_idx - 1
                if (
                    side_is_root_by_depth[depth_idx]
                    != side_is_root_by_depth[parent_depth]
                ):
                    current_val = -current_val
                curr_node = parent
                depth_idx = parent_depth

    def _heuristic_rollout_mutable(
        self, mutable_state: MutableGameState
    ) -> float:
        """Perform heuristic-guided rollout simulation using make/unmake.

        All rollout moves are made and then unmade to preserve state.
        """
        rollout_depth = 3
        rollout_undos: List[MoveUndo] = []

        for _ in range(rollout_depth):
            if mutable_state.is_game_over():
                break

            # Get valid moves from immutable conversion
            immutable = mutable_state.to_immutable()
            moves = self.rules_engine.get_valid_moves(
                immutable,
                mutable_state.current_player,
            )
            if not moves:
                break

            weights = self._compute_move_weights_mutable(moves, mutable_state)
            selected_move = self.rng.choices(moves, weights=weights, k=1)[0]
            undo = mutable_state.make_move(selected_move)
            rollout_undos.append(undo)

        # Evaluate at rollout end
        result = self._evaluate_mutable(mutable_state)

        # Unmake all rollout moves
        for undo in reversed(rollout_undos):
            mutable_state.unmake_move(undo)

        return result

    def _compute_move_weights_mutable(
        self, moves: List[Move], state: MutableGameState
    ) -> List[float]:
        """Compute weights for move selection in mutable rollout."""
        weights = []
        for m in moves:
            w = 1.0
            if m.type == "territory_claim":
                w = 100.0
            elif m.type == "line_formation":
                w = 50.0
            elif m.type == "chain_capture":
                w = 20.0
            elif m.type == "overtaking_capture":
                w = 10.0
            elif m.type == "move_stack":
                to_key = m.to.to_key()
                if to_key in state.stacks:
                    w = 5.0
                else:
                    w = 2.0
            elif m.type == "place_ring":
                w = 1.5
            weights.append(w)
        return weights

    def _ensure_gpu_initialized(self, game_state: Optional[GameState] = None) -> bool:
        """Lazily initialize GPU resources. Returns True if GPU available.

        GPU acceleration for MCTS rollout evaluation provides 5-20x speedup
        by using vectorized heuristic computation on GPU.

        Args:
            game_state: Optional game state to detect board configuration from.

        Returns:
            True if GPU is available and initialized.
        """
        if self._gpu_available is not None:
            return self._gpu_available

        if not self._gpu_enabled:
            self._gpu_available = False
            logger.debug("MCTSAI: GPU disabled via RINGRIFT_GPU_MCTS_DISABLE")
            return False

        try:
            from .gpu_batch import get_device, GPUHeuristicEvaluator

            self._gpu_device = get_device(prefer_gpu=True)
            self._gpu_available = self._gpu_device.type in ('cuda', 'mps')

            if self._gpu_available and game_state is not None:
                # Detect board configuration
                self._board_type_cached = game_state.board_type
                self._num_players_cached = len(game_state.players)

                board_size_map = {
                    BoardType.SQUARE8: 8,
                    BoardType.SQUARE19: 19,
                    BoardType.HEXAGONAL: 25,
                }
                self._board_size = board_size_map.get(self._board_type_cached, 8)

                # Initialize GPU heuristic evaluator
                self._gpu_evaluator = GPUHeuristicEvaluator(
                    device=self._gpu_device,
                    board_size=self._board_size,
                    num_players=self._num_players_cached,
                )
                logger.info(
                    f"MCTSAI: GPU acceleration enabled on {self._gpu_device} "
                    f"(board={self._board_size}x{self._board_size})"
                )
            elif not self._gpu_available:
                logger.info(
                    f"MCTSAI: No GPU available (device={self._gpu_device.type}), "
                    "using CPU evaluation"
                )
        except Exception as e:
            logger.warning(f"MCTSAI: GPU initialization failed, using CPU: {e}")
            self._gpu_available = False

        return self._gpu_available

    def _evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate MutableGameState for MCTS rollout.

        GPU Acceleration:
            When GPU is available, uses GPU heuristic evaluator for faster
            position evaluation. Falls back to CPU when GPU unavailable.
            Control via RINGRIFT_GPU_MCTS_DISABLE=1 environment variable.
        """
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0
            elif winner is not None:
                return -100000.0
            else:
                return 0.0

        immutable = state.to_immutable()

        # GPU evaluation path
        if self._gpu_available and self._gpu_evaluator is not None:
            try:
                from .hybrid_gpu import batch_game_states_to_gpu

                # Single-state batch evaluation
                gpu_state = batch_game_states_to_gpu(
                    [immutable],
                    self._gpu_device,
                    self._board_size or 8,
                )
                scores_tensor = self._gpu_evaluator.evaluate_batch(
                    gpu_state, self.player_number
                )
                return float(scores_tensor[0].cpu().item())
            except Exception as e:
                # Fall back to CPU on GPU error
                logger.debug(f"MCTSAI: GPU eval failed, using CPU: {e}")

        # CPU fallback
        return self.evaluate_position(immutable)

    def _update_node_policy_lite(
        self,
        node: MCTSNodeLite,
        state: GameState,
        policy: Any,
        use_hex_nn: bool,
    ) -> None:
        """Update lite node policy priors from neural network output."""
        from .neural_net import INVALID_MOVE_INDEX  # Lazy import

        valid_moves_state = self.rules_engine.get_valid_moves(
            state,
            state.current_player,
        )
        if not valid_moves_state:
            return

        existing_child_keys = (
            {_move_key(c.move) for c in node.children if c.move is not None}
            if node.children
            else set()
        )
        node.policy_map = {}

        total_prob = 0.0
        for move in valid_moves_state:
            if use_hex_nn and self.hex_encoder is not None:
                idx = self.hex_encoder.encode_move(move, state.board)
            elif self.neural_net is not None:
                idx = self.neural_net.encode_move(move, state.board)
            else:
                idx = INVALID_MOVE_INDEX

            if idx != INVALID_MOVE_INDEX and 0 <= idx < len(policy):
                prob = float(cast(Any, policy)[idx])
                node.policy_map[str(move)] = prob
                total_prob += prob

        if total_prob > 0:
            for move_key in node.policy_map:
                node.policy_map[move_key] /= total_prob
        elif self.nnue_policy_model is not None:
            # Fallback to NNUE policy when neural network policy unavailable
            nnue_policy = self._compute_nnue_policy(valid_moves_state, state)
            if nnue_policy:
                node.policy_map = nnue_policy
            else:
                uniform = 1.0 / len(valid_moves_state)
                for move in valid_moves_state:
                    node.policy_map[str(move)] = uniform
        else:
            uniform = 1.0 / len(valid_moves_state)
            for move in valid_moves_state:
                node.policy_map[str(move)] = uniform

        # Keep untried moves disjoint from already-expanded children (important
        # for root prior seeding + tree reuse, and for move types where
        # (type,to) is not unique, e.g. multi-ring placements or choose-line
        # segments).
        node.untried_moves = [
            m for m in valid_moves_state if _move_key(m) not in existing_child_keys
        ]

        # Apply Dirichlet noise only at root during self-play.
        self._maybe_apply_root_dirichlet_noise(node, state.board.type)

        # Order untried moves by priors when policy_map is available.
        # For progressive widening this focuses on top-prior moves; for other
        # boards it still benefits from policy guidance during expansion.
        if node.policy_map:
            node.untried_moves.sort(
                key=lambda m: node.policy_map.get(str(m), 0.0),
                reverse=True,
            )

    def _select_best_move_incremental(
        self,
        root: MCTSNodeLite,
        valid_moves: List[Move],
        game_state: GameState,
    ) -> Tuple[Optional[Move], Optional[Dict[str, float]]]:
        """Select best move from incremental tree based on visit counts."""
        # Build a set of valid move signatures for efficient lookup.
        # Include player attribute to catch stale moves from tree reuse where
        # the acting player or their resources (e.g., rings_in_hand) changed.
        valid_moves_set = {_move_key(m) for m in valid_moves}

        # Filter children to only those with currently valid moves.
        # This is critical for tree reuse scenarios where children were
        # expanded when a different player had the turn or when player
        # resources have changed (e.g., rings_in_hand depleted).
        valid_children = [
            c for c in root.children
            if c.move is not None and _move_key(c.move) in valid_moves_set
        ]

        if valid_children:
            total_visits = sum(c.visits for c in valid_children)
            policy: Dict[str, float] = {}
            if total_visits > 0:
                for child in valid_children:
                    if child.move is not None:
                        policy[str(child.move)] = child.visits / total_visits
            else:
                uniform = 1.0 / len(valid_children)
                for child in valid_children:
                    if child.move is not None:
                        policy[str(child.move)] = uniform

            best_child = max(valid_children, key=lambda c: c.visits)
            selected_child = best_child
            if self.self_play:
                temperature = self._get_selfplay_temperature(game_state)
                selected_child = self._sample_child_by_temperature(
                    valid_children, temperature
                )
            selected = selected_child.move

            self.last_root_lite = selected_child
            self.last_root_lite.parent = None

            # Extra validation: ensure selected move matches a valid move.
            # This should always pass given the filtering above, but we keep
            # it as a safety net.
            if selected is not None:
                is_valid = _move_key(selected) in valid_moves_set
                if not is_valid:
                    logger.warning(
                        "MCTS incremental selected invalid move %s "
                        "(not in valid_moves), falling back to random",
                        selected.type if selected else None,
                    )
                    selected = self.get_random_element(valid_moves)
                    policy = {
                        str(m): (1.0 if m == selected else 0.0)
                        for m in valid_moves
                    }
                    self.last_root_lite = None  # Discard stale tree
        else:
            selected = self.get_random_element(valid_moves)
            policy = {
                str(m): (1.0 if m == selected else 0.0)
                for m in valid_moves
            }

        self.move_count += 1
        return selected, policy

    # =========================================================================
    # Shared Utility Methods
    # =========================================================================

    def _default_dirichlet_alpha(self, board_type: BoardType) -> float:
        """Return a conservative board-specific Dirichlet alpha."""
        if board_type == BoardType.SQUARE8:
            return 0.3
        # Large action spaces should use smaller alpha.
        return 0.15

    def _maybe_apply_root_dirichlet_noise(
        self,
        node: Any,
        board_type: BoardType,
    ) -> None:
        """Mix Dirichlet noise into root priors for self-play only."""
        if not self.self_play or self._dirichlet_applied_this_search:
            return
        if getattr(node, "parent", None) is not None:
            return
        if not getattr(node, "policy_map", None):
            return

        keys = list(node.policy_map.keys())
        if len(keys) <= 1 or self.root_noise_fraction <= 0:
            self._dirichlet_applied_this_search = True
            return

        alpha = self.root_dirichlet_alpha or self._default_dirichlet_alpha(board_type)
        epsilon = self.root_noise_fraction

        # Seed numpy from the per-instance RNG for reproducibility.
        seed = int(self.rng.randrange(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        noise = rng.dirichlet([alpha] * len(keys))

        for i, key in enumerate(keys):
            prior = float(node.policy_map[key])
            node.policy_map[key] = (1.0 - epsilon) * prior + epsilon * float(noise[i])

        total = float(sum(node.policy_map.values()))
        if total > 0:
            for key in node.policy_map:
                node.policy_map[key] /= total

        self._dirichlet_applied_this_search = True

    def _get_selfplay_temperature(self, game_state: GameState) -> float:
        """Return temperature for self-play root move sampling."""
        if self.temperature_override is not None:
            return float(self.temperature_override)

        board_type = game_state.board.type
        cutoff = self.temperature_cutoff_moves
        if cutoff is None:
            if board_type == BoardType.SQUARE8:
                cutoff = 24
            else:
                cutoff = 40

        move_index = len(game_state.move_history)
        if move_index < cutoff:
            return 1.0
        if move_index < cutoff * 2:
            return 0.5
        return 0.1

    def _default_leaf_batch_size(self) -> int:
        """Choose a default leaf batch size for NN evaluation.

        This is a throughput knob only; correctness is unaffected. Callers can
        override via RINGRIFT_MCTS_LEAF_BATCH_SIZE.
        """
        env_val = os.environ.get("RINGRIFT_MCTS_LEAF_BATCH_SIZE")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

        if not self.neural_net:
            return 8

        device = getattr(self.neural_net, "device", "cpu")
        dev_str = device if isinstance(device, str) else getattr(device, "type", "cpu")
        if dev_str == "cuda":
            return 32
        if dev_str == "mps":
            return 16
        if dev_str == "cpu":
            return 8
        return 16

    def _maybe_seed_root_priors(self, root: Any, game_state: GameState) -> None:
        """Seed NN priors at the root.

        Progressive widening relies on high-quality priors to select the
        initial children. For square19/hex boards, evaluate the root once
        so early expansions focus on top-prior moves and reused roots regain
        consistent priors.

        When no neural network is available but NNUE policy is loaded, uses
        NNUE policy priors as a lightweight alternative. This applies to ALL
        board types (including square8) to benefit from policy guidance even
        without progressive widening.
        """
        board_type = game_state.board.type
        use_progressive = self._use_progressive_widening(board_type)

        # For non-progressive-widening boards, only seed if NNUE policy is available
        # (no neural net and NNUE model loaded)
        if not use_progressive:
            if self.neural_net or self.nnue_policy_model is None:
                return

        existing_map = getattr(root, "policy_map", None)
        if isinstance(existing_map, dict) and existing_map:
            return

        # Fallback to NNUE policy when no neural net is available
        if not self.neural_net:
            if self.nnue_policy_model is not None:
                try:
                    valid_moves = self.rules_engine.get_valid_moves(
                        game_state, game_state.current_player
                    )
                    nnue_policy = self._compute_nnue_policy(valid_moves, game_state)
                    if nnue_policy:
                        root.policy_map = nnue_policy
                        root.untried_moves = list(valid_moves)
                        # Sort by priors
                        root.untried_moves.sort(
                            key=lambda m: root.policy_map.get(str(m), 0.0),
                            reverse=True,
                        )
                        # Update existing children
                        for child in getattr(root, "children", []):
                            move = getattr(child, "move", None)
                            if move is None:
                                continue
                            prior = root.policy_map.get(str(move))
                            if prior is not None:
                                child.prior = float(prior)
                        # Compute entropy for logging (higher = more uniform)
                        max_p = max(nnue_policy.values()) if nnue_policy else 0.0
                        min_p = min(nnue_policy.values()) if nnue_policy else 0.0
                        logger.debug(
                            f"Seeded NNUE root priors for {board_type.value}: "
                            f"{len(nnue_policy)} moves, temp={self.policy_temperature:.1f}, "
                            f"max_prior={max_p:.3f}, min_prior={min_p:.4f}"
                        )
                except Exception:
                    logger.debug("Failed to seed NNUE root priors", exc_info=True)
            return

        try:
            use_hex_nn = (
                self.hex_model is not None
                and self.hex_encoder is not None
                and board_type == BoardType.HEXAGONAL
            )
            use_vector_head = (
                self.use_vector_value_head
                and not use_hex_nn
                and infer_num_players(game_state) > 2
            )
            value_head = (self.player_number - 1) if use_vector_head else None

            if use_hex_nn:
                eval_values, eval_policies = self._evaluate_hex_batch([game_state])
                policy_vec = eval_policies[0]
                value = float(eval_values[0]) if eval_values else 0.0
            else:
                eval_values, policy_batch = (
                    self.nn_batcher.evaluate(
                        [game_state],
                        value_head=value_head,
                    )
                    if self.nn_batcher is not None
                    else self.neural_net.evaluate_batch(
                        [game_state],
                        value_head=value_head,
                    )
                )
                policy_vec = policy_batch[0]
                value = float(eval_values[0]) if eval_values else 0.0

            if isinstance(root, MCTSNode):
                self._update_node_policy_legacy(
                    root, game_state, policy_vec, bool(use_hex_nn)
                )
            else:
                self._update_node_policy_lite(
                    cast(MCTSNodeLite, root),
                    game_state,
                    policy_vec,
                    bool(use_hex_nn),
                )

            # Update priors on existing children (tree reuse).
            for child in getattr(root, "children", []):
                move = getattr(child, "move", None)
                if move is None:
                    continue
                prior = getattr(root, "policy_map", {}).get(str(move))
                if prior is not None:
                    child.prior = float(prior)

            # Cache root eval for TT reuse.
            state_hash = game_state.zobrist_hash or 0
            if self.transposition_table.get(state_hash) is None:
                self.transposition_table.put(state_hash, (value, policy_vec))
        except Exception:
            logger.debug("Failed to seed root priors", exc_info=True)

    def _sample_child_by_temperature(
        self,
        children: List[Any],
        temperature: float,
    ) -> Any:
        """Sample a child proportional to visits^1/temperature."""
        if temperature <= 0 or len(children) == 1:
            return max(children, key=lambda c: c.visits)

        visits = np.array(
            [max(0.0, float(c.visits)) for c in children],
            dtype=np.float64,
        )
        if visits.sum() <= 0:
            probs = np.ones_like(visits) / len(visits)
        else:
            probs = visits / visits.sum()

        if temperature != 1.0:
            probs = probs ** (1.0 / float(temperature))
            p_sum = probs.sum()
            if p_sum > 0:
                probs /= p_sum

        idx = self.rng.choices(
            list(range(len(children))),
            weights=probs.tolist(),
            k=1,
        )[0]
        return children[idx]

    # ------------------------------------------------------------------
    # Progressive widening (large boards only).
    # ------------------------------------------------------------------

    def _use_progressive_widening(self, board_type: BoardType) -> bool:
        return board_type in (BoardType.SQUARE19, BoardType.HEXAGONAL)

    def _max_children_allowed(self, visits: int, board_type: BoardType) -> int:
        if not self._use_progressive_widening(board_type):
            return 1_000_000_000

        # Conservative defaults; tune per-board in future slices.
        min_children = 8 if board_type == BoardType.SQUARE19 else 10
        c = 2.0
        alpha = 0.5
        v = max(1, int(visits))
        return max(min_children, int(c * (v**alpha)))

    def _can_expand_node(self, node: Any, board_type: BoardType) -> bool:
        if not self._use_progressive_widening(board_type):
            return True
        visits = int(getattr(node, "visits", 0))
        children = getattr(node, "children", [])
        return len(children) < self._max_children_allowed(visits, board_type)

    def _select_untried_move(self, node: Any, board_type: BoardType) -> Move:
        """Pick the next untried move for expansion.

        When policy_map is available (from neural net or NNUE policy), selects
        the highest-probability move. Otherwise falls back to random selection.
        """
        moves: List[Move] = list(getattr(node, "untried_moves", []))
        if not moves:
            raise ValueError("No untried moves to select")
        # Use policy-guided selection when policy_map is available
        policy_map = getattr(node, "policy_map", None)
        if isinstance(policy_map, dict) and policy_map:
            return max(moves, key=lambda m: policy_map.get(str(m), 0.0))
        return cast(Move, self.get_random_element(moves))

    def _log_stats(self) -> None:
        """Log transposition table and dynamic sizer stats."""
        if logger.isEnabledFor(logging.DEBUG):
            tt_stats = self.transposition_table.stats()
            logger.debug(
                "MCTS transposition table stats: entries=%d/%d, "
                "hits=%d, misses=%d, hit_rate=%.2f%%, evictions=%d, "
                "est_memory=%.2fMB",
                tt_stats["entries"],
                tt_stats["max_entries"],
                tt_stats["hits"],
                tt_stats["misses"],
                tt_stats["hit_rate"] * 100,
                tt_stats["evictions"],
                tt_stats["estimated_memory_mb"],
            )

            if self.enable_dynamic_batching and self.dynamic_sizer is not None:
                ds_stats = self.dynamic_sizer.stats()
                logger.debug(
                    "MCTS dynamic batch sizer stats: batch_size=%d, "
                    "node_estimate=%d bytes, adjustments=%d",
                    ds_stats["current_batch_size"],
                    ds_stats["node_size_estimate"],
                    ds_stats["adjustment_count"],
                )

    def clear_tree(self) -> None:
        """Clear the MCTS tree to free memory.

        This should be called after move selection if memory is constrained.
        """
        if hasattr(self, "last_root"):
            self.last_root = None
