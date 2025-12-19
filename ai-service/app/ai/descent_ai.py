"""
Descent AI implementation for RingRift.

This agent implements a Descent / UBFM‑style tree search (inspired by
“A Simple AlphaZero” – arXiv:2008.01188v4) over a shared rules engine.

When ``config.use_incremental_search`` is True (the default), DescentAI uses
the make/unmake pattern on ``MutableGameState`` for faster search by
avoiding repeated object allocation. When False, it falls back to the
legacy immutable state cloning via ``apply_move()``. Both modes are kept
for A/B testing and for backwards‑compatible behaviour.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import time
from enum import Enum

import numpy as np

from .base import BaseAI
from .bounded_transposition_table import BoundedTranspositionTable
from .game_state_utils import infer_num_players, victory_progress_for_player
from ..models import GameState, Move, MoveType, AIConfig, BoardType
from ..rules.mutable_state import MutableGameState
from ..utils.memory_config import MemoryConfig

# Lazy imports for neural network components to avoid loading torch when not needed
if TYPE_CHECKING:
    from .async_nn_eval import AsyncNeuralBatcher
    from .neural_net import NeuralNetAI, ActionEncoderHex
    from .evaluation_provider import HeuristicEvaluator

# Optional GPU heuristic evaluation - try but don't fail if unavailable
GPU_HEURISTIC_AVAILABLE = False
GPUHeuristicEvaluator = None  # type: ignore
GPUBoardState = None  # type: ignore

logger = logging.getLogger(__name__)

# Maximum search depth to prevent stack overflow in degenerate cases (e.g.,
# transposition table cycles or extremely long game sequences). When this
# depth is reached, we return a heuristic evaluation instead of continuing
# the descent.
MAX_SEARCH_DEPTH = 500


class NodeStatus(Enum):
    HEURISTIC = 0
    PROVEN_WIN = 1
    PROVEN_LOSS = 2
    DRAW = 3


class DescentAI(BaseAI):
    """AI that uses a Descent / UBFM‑style tree search.

    DescentAI incrementally extends the most promising sequence of actions
    towards terminal states, using bounded transposition tables and an
    optional neural network backend for value/policy estimates. It supports
    both immutable (legacy) and mutable (make/unmake) search modes, gated
    by ``config.use_incremental_search``.

    Configuration overview (:class:`AIConfig` / related fields):

    - ``use_incremental_search``: When ``True`` (default), uses the
      :class:`MutableGameState` make/unmake path; when ``False``, falls
      back to the legacy immutable path.
    - ``think_time``: Per‑move wall‑clock budget in milliseconds. When
      set to a positive value, both legacy and incremental paths treat
      this purely as a **search time limit**; when unset or non‑positive,
      a difficulty‑scaled default is used (roughly 0.1s–2.0s).
    - ``randomness``: Passed through to :meth:`BaseAI.should_pick_random_move`
      for occasional random move selection before entering the Descent
      loop.
    - ``nn_model_id`` / related NN config: Threaded into
      :class:`NeuralNetAI` to control which checkpoint and architecture
      are used for value/policy estimates when available. When no model
      can be loaded, the agent safely degrades to purely heuristic search.

    Memory configuration:
        Transposition table sizing is controlled by :class:`MemoryConfig`,
        which is either provided explicitly via ``memory_config`` or
        derived from the environment. This determines the effective cap on
        stored nodes and thus memory usage during deep searches.
    """

    def __init__(
        self,
        player_number: int,
        config: AIConfig,
        memory_config: Optional[MemoryConfig] = None,
    ):
        super().__init__(player_number, config)

        # Determine whether neural-network-backed evaluation should be used.
        # Priority:
        # - Explicit AIConfig.use_neural_net when provided.
        # - RINGRIFT_DISABLE_NEURAL_NET env var can globally disable NN usage.
        disable_nn_env = os.environ.get("RINGRIFT_DISABLE_NEURAL_NET", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.require_neural_net = os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_nn_config = getattr(config, "use_neural_net", None)
        # Default to True (preserve existing behaviour) when unset.
        self.use_neural_net: bool = bool(
            (use_nn_config if use_nn_config is not None else True) and not disable_nn_env
        )

        # Try to load neural net for evaluation when enabled.
        # NeuralNetAI now uses lazy initialization to select the correct
        # board-specific model (RingRiftCNN_MPS for square, HexNeuralNet for hex)
        # when it first sees a game state.
        self.neural_net: Optional["NeuralNetAI"] = None
        self.hex_encoder: Optional["ActionEncoderHex"] = None

        if self.use_neural_net:
            try:
                from .neural_net import NeuralNetAI  # Lazy import
                self.neural_net = NeuralNetAI(player_number, config)
            except Exception:
                if self.require_neural_net:
                    raise
                # On any failure, degrade gracefully to heuristic-only search.
                logger.warning(
                    "DescentAI failed to initialize NeuralNetAI; "
                    "falling back to heuristic-only search",
                    exc_info=True,
                )
                self.neural_net = None
                self.use_neural_net = False

        # Hex-specific action encoder (used for move index calculation on hex boards)
        if self.neural_net is not None:
            from .neural_net import ActionEncoderHex  # Lazy import
            self.hex_encoder = ActionEncoderHex()
        # Per-instance neural batcher for safe, synchronous batched evaluation.
        self.nn_batcher: Optional["AsyncNeuralBatcher"] = None
        if self.neural_net is not None:
            from .async_nn_eval import AsyncNeuralBatcher  # Lazy import
            self.nn_batcher = AsyncNeuralBatcher(self.neural_net)
        # Optional async NN evaluation to overlap CPU expansion with background
        # inference. Enabled via env var and only when a non-CPU device is used.
        async_env = os.environ.get("RINGRIFT_DESCENT_ASYNC_NN_EVAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enable_async_nn_eval: bool = False
        if async_env and self.neural_net is not None:
            dev = getattr(self.neural_net, "device", "cpu")
            dev_type = dev if isinstance(dev, str) else getattr(dev, "type", "cpu")
            if dev_type != "cpu":
                self.enable_async_nn_eval = True

        # Memory configuration for bounded structures
        self.memory_config = memory_config or MemoryConfig.from_env()

        # GPU heuristic evaluator for batch fallback (when NN unavailable)
        self.gpu_heuristic: Optional["GPUHeuristicEvaluator"] = None
        self.use_gpu_heuristic = os.environ.get("RINGRIFT_DESCENT_GPU_HEURISTIC", "").lower() in {
            "1", "true", "yes", "on"
        }
        if self.use_gpu_heuristic and GPU_HEURISTIC_AVAILABLE and not self.neural_net:
            try:
                self.gpu_heuristic = GPUHeuristicEvaluator(device=get_device())
                logger.info("DescentAI initialized with GPU heuristic evaluator")
            except Exception:
                logger.warning("Failed to initialize GPU heuristic evaluator", exc_info=True)
                self.gpu_heuristic = None

        # FastGeometry-backed heuristic evaluator for single-position fallback
        # (when neural net is unavailable). Uses pre-computed geometry tables
        # for ~3-5x faster evaluation than naive implementations.
        self._heuristic_evaluator: Optional["HeuristicEvaluator"] = None
        self._use_heuristic_fallback = os.environ.get(
            "RINGRIFT_DESCENT_HEURISTIC_FALLBACK", "true"
        ).lower() in {"1", "true", "yes", "on"}
        if self._use_heuristic_fallback and not self.neural_net:
            try:
                from .evaluation_provider import HeuristicEvaluator, EvaluatorConfig
                eval_config = EvaluatorConfig(
                    eval_mode="light",  # Use light mode for fast Descent fallback
                    difficulty=config.difficulty,
                )
                self._heuristic_evaluator = HeuristicEvaluator(
                    player_number=player_number,
                    config=eval_config,
                )
                logger.info("DescentAI initialized with FastGeometry heuristic evaluator")
            except Exception:
                logger.warning(
                    "Failed to initialize heuristic evaluator; "
                    "falling back to simple material difference",
                    exc_info=True,
                )
                self._heuristic_evaluator = None

        # Transposition table to store values with bounded memory
        # Key: state_hash, Value: (value, children_values, status)
        # Entry size estimate: ~10000 bytes per entry.
        # Chain captures create many children per node, each storing
        # (Move, value) tuples. Conservative estimate to prevent OOM.
        tt_limit = self.memory_config.get_transposition_table_limit_bytes()
        self.transposition_table = BoundedTranspositionTable.from_memory_limit(
            tt_limit,
            entry_size_estimate=10000,
        )

        # Progressive widening parameters for large boards. We store only a
        # bounded tail of unexpanded moves to keep TT entries small.
        self._pw_max_remaining_store: int = 512

        # Search log for Tree Learning
        # List of (features, value)
        # Only populated when collect_training_data is True to prevent
        # memory leaks in inference-only scenarios.
        self.collect_training_data: bool = False
        self.search_log: List[Tuple[Any, float]] = []

        # Configuration option for incremental search (make/unmake pattern)
        self.use_incremental_search: bool = getattr(
            config, 'use_incremental_search', True
        )

        # Optional vector value-head selection for multi-player evaluation.
        # When enabled, Descent can request a specific NeuralNetAI value head
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

        # Optional uncertainty-aware child selection (UCB-style) for Descent.
        # This encourages exploration of under-visited children in deep search.
        ucb_env = os.environ.get("RINGRIFT_DESCENT_UCB", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.use_uncertainty_selection: bool = bool(
            getattr(config, "use_uncertainty_selection", False)
        ) or ucb_env
        self.uncertainty_ucb_c: float = 0.25
        cfg_c = getattr(config, "uncertainty_ucb_c", None)
        if cfg_c is not None:
            try:
                self.uncertainty_ucb_c = float(cfg_c)
            except Exception:
                pass
        env_c = os.environ.get("RINGRIFT_DESCENT_UCB_C", "").strip()
        if env_c:
            try:
                self.uncertainty_ucb_c = float(env_c)
            except Exception:
                pass
        if self.uncertainty_ucb_c <= 0:
            self.use_uncertainty_selection = False

    def _default_nn_batch_size(self) -> int:
        """Default NN batch size for Descent leaf evaluation.

        Callers can override via RINGRIFT_DESCENT_LEAF_BATCH_SIZE.
        """
        env_val = os.environ.get("RINGRIFT_DESCENT_LEAF_BATCH_SIZE")
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

    # ------------------------------------------------------------------
    # Progressive widening helpers (large boards only).
    # ------------------------------------------------------------------

    def _use_progressive_widening(self, board_type: BoardType) -> bool:
        return board_type in (BoardType.SQUARE19, BoardType.HEXAGONAL)

    def _max_children_allowed(self, visits: int, board_type: BoardType) -> int:
        if not self._use_progressive_widening(board_type):
            return 1_000_000_000
        min_children = 8 if board_type == BoardType.SQUARE19 else 10
        c = 2.0
        alpha = 0.5
        v = max(1, int(visits))
        return max(min_children, int(c * (v**alpha)))

    def _select_child_key(
        self,
        children_values: Dict[str, Any],
        *,
        parent_visits: int,
        maximizing: bool,
    ) -> str:
        """Select a child move key to descend into.

        Default behaviour matches the legacy greedy selection:
        - maximize (root-to-move): pick highest value, tie-break by policy prob.
        - minimize (opponent-to-move): pick lowest value.

        When ``self.use_uncertainty_selection`` is enabled, use an
        optimism/pessimism bound similar to UCB:
        - maximize: value + c * sqrt(log(N) / (n + 1))
        - minimize: value - c * sqrt(log(N) / (n + 1))
        where n is per-child visit count tracked in the TT entry.
        """
        if not children_values:
            raise ValueError("No children to select from")

        def _prob(data: Any) -> float:
            if isinstance(data, tuple) and len(data) > 2:
                try:
                    return float(data[2])
                except Exception:
                    return 0.0
            return 0.0

        def _visits(data: Any) -> int:
            if isinstance(data, tuple) and len(data) > 3:
                try:
                    return int(data[3])
                except Exception:
                    return 0
            return 0

        if not self.use_uncertainty_selection:
            if maximizing:
                return max(
                    children_values.items(),
                    key=lambda item: (float(item[1][1]), _prob(item[1])),
                )[0]
            return min(
                children_values.items(),
                key=lambda item: (float(item[1][1]), -_prob(item[1])),
            )[0]

        log_term = math.log(max(2.0, float(parent_visits) + 1.0))
        c = float(self.uncertainty_ucb_c)

        def _score(item: tuple[str, Any]) -> tuple[float, float]:
            _key, data = item
            try:
                mean = float(data[1])
            except Exception:
                mean = 0.0
            bonus = c * math.sqrt(log_term / float(_visits(data) + 1))
            if maximizing:
                return (mean + bonus, _prob(data))
            return (mean - bonus, -_prob(data))

        if maximizing:
            return max(children_values.items(), key=_score)[0]
        return min(children_values.items(), key=_score)[0]

    def _unpack_tt_entry(
        self, entry: Any
    ) -> tuple[float, Dict[str, Any], NodeStatus, list[tuple[Move, float]], int]:
        """Normalize TT entries across legacy/progressive formats."""
        current_val = float(entry[0])
        children_values: Dict[str, Any] = entry[1]
        status = entry[2] if len(entry) >= 3 else NodeStatus.HEURISTIC
        remaining_moves: list[tuple[Move, float]] = (
            entry[3] if len(entry) >= 4 else []
        )
        visits = int(entry[4]) if len(entry) >= 5 else 0
        return current_val, children_values, status, remaining_moves, visits

    def get_search_data(self) -> List[Tuple[Any, float]]:
        """Retrieve and clear the accumulated search log.

        Note:
            For training, set ``collect_training_data=True`` (or call
            :meth:`enable_training_data_collection`) before running
            :meth:`select_move` to enable search‑data collection.
        """
        data = self.search_log
        self.search_log = []
        return data

    def enable_training_data_collection(self, enabled: bool = True) -> None:
        """Enable or disable search‑data collection for training.

        When disabled (the default), :attr:`search_log` is not populated,
        preventing memory accumulation in inference‑only scenarios.

        Args:
            enabled: Whether to collect training data during search.
        """
        self.collect_training_data = enabled
        if not enabled:
            # Clear any existing data when disabling
            self.search_log.clear()

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using Descent search.

        Routes to incremental (make/unmake) or legacy (immutable) search
        based on the use_incremental_search configuration.
        """
        # No simulated thinking for Descent, we use the time for search.

        # Get all valid moves for this AI player via the rules engine
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        valid_moves = [
            m for m in valid_moves if m.type != MoveType.SWAP_SIDES
        ]

        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
            # get_random_element only returns None when the list is empty, but
            # we guard for type-checkers even though we already checked above.
            if selected is None:
                return None
            return selected

        # Route to incremental or legacy search based on config
        if self.use_incremental_search:
            return self._select_move_incremental(game_state, valid_moves)
        else:
            return self._select_move_legacy(game_state, valid_moves)

    def _select_move_legacy(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> Optional[Move]:
        """Legacy search using immutable state cloning via apply_move().

        This is the original implementation preserved for backward
        compatibility and A/B testing against the new incremental search.
        """
        # Descent parameters
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            # Default time limit based on difficulty
            # Difficulty 1: 0.1s, Difficulty 10: 2.0s
            time_limit = 0.1 + (self.config.difficulty * 0.2)

        end_time = time.time() + time_limit

        # Run Descent iterations until the deadline or until the root is
        # proven solved.
        iterations = 0
        while time.time() < end_time:
            self._descent_iteration(
                game_state,
                depth=0,
                deadline=end_time,
            )
            iterations += 1

            # Completion: Stop if root is solved
            state_key = self._get_state_key(game_state)
            entry = self.transposition_table.get(state_key)
            if entry is not None:
                _, _, status, _, _ = self._unpack_tt_entry(entry)
                if status in (
                    NodeStatus.PROVEN_WIN,
                    NodeStatus.PROVEN_LOSS,
                ):
                    break

        # Select best move from root
        state_key = self._get_state_key(game_state)
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            _, children_values, _, _, _ = self._unpack_tt_entry(entry)

            if children_values:
                if game_state.current_player == self.player_number:
                    best_move_key = max(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                else:
                    best_move_key = min(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                best_move = children_values[best_move_key][0]
                # Validate the move is still legal for the current state.
                # The transposition table may contain stale moves from hash
                # collisions or previous games.
                if best_move in valid_moves:
                    return best_move
                # Fall through to random fallback if TT move is invalid

        # Log transposition table stats at end of search
        if logger.isEnabledFor(logging.DEBUG):
            stats = self.transposition_table.stats()
            logger.debug(
                "Descent search completed: iterations=%d, tt_entries=%d, "
                "tt_hits=%d, tt_misses=%d, tt_evictions=%d, hit_rate=%.2f%%",
                iterations,
                stats["entries"],
                stats["hits"],
                stats["misses"],
                stats["evictions"],
                stats["hit_rate"] * 100,
            )

        # Fallback if something went wrong or no search happened. Use the
        # per-instance RNG for reproducible behaviour under a fixed seed.
        fallback = self.get_random_element(valid_moves)
        if fallback is None:
            return None
        return fallback

    def _select_move_incremental(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> Optional[Move]:
        """Incremental search using make/unmake on MutableGameState.

        This provides significant speedup by avoiding object allocation
        overhead during tree search. State is modified in-place and restored
        using MoveUndo tokens.
        """
        # Descent parameters
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            # Default time limit based on difficulty
            # Difficulty 1: 0.1s, Difficulty 10: 2.0s
            time_limit = 0.1 + (self.config.difficulty * 0.2)

        end_time = time.time() + time_limit

        # Create mutable state once for the entire search
        mutable_state = MutableGameState.from_immutable(game_state)

        # Run Descent iterations until the deadline or until the root is
        # proven solved.
        iterations = 0
        while time.time() < end_time:
            self._descent_iteration_mutable(
                mutable_state,
                depth=0,
                deadline=end_time,
            )
            iterations += 1

            # Completion: Stop if root is solved
            state_key = mutable_state.zobrist_hash
            entry = self.transposition_table.get(state_key)
            if entry is not None:
                _, _, status, _, _ = self._unpack_tt_entry(entry)
                if status in (
                    NodeStatus.PROVEN_WIN,
                    NodeStatus.PROVEN_LOSS,
                ):
                    break

        # Select best move from root
        state_key = mutable_state.zobrist_hash
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            _, children_values, _, _, _ = self._unpack_tt_entry(entry)

            if children_values:
                if mutable_state.current_player == self.player_number:
                    best_move_key = max(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                else:
                    best_move_key = min(
                        children_values.items(),
                        key=lambda x: x[1][1],
                    )[0]
                best_move = children_values[best_move_key][0]
                # Validate the move is still legal for the current state.
                # The transposition table may contain stale moves from hash
                # collisions or previous games.
                if best_move in valid_moves:
                    return best_move
                # Fall through to random fallback if TT move is invalid

        # Log transposition table stats at end of search
        if logger.isEnabledFor(logging.DEBUG):
            stats = self.transposition_table.stats()
            logger.debug(
                "Descent search (incremental) completed: iterations=%d, "
                "tt_entries=%d, tt_hits=%d, tt_misses=%d, tt_evictions=%d, "
                "hit_rate=%.2f%%",
                iterations,
                stats["entries"],
                stats["hits"],
                stats["misses"],
                stats["evictions"],
                stats["hit_rate"] * 100,
            )

        # Fallback if something went wrong or no search happened. Use the
        # per-instance RNG for reproducible behaviour under a fixed seed.
        fallback = self.get_random_element(valid_moves)
        if fallback is None:
            return None
        return fallback

    def _descent_iteration(
        self,
        state: GameState,
        depth: int = 0,
        deadline: Optional[float] = None,
    ) -> float:
        """
        Perform one iteration of Descent search.

        Recursively selects the best child until a terminal position or
        timeout is reached, then backpropagates values to the root.
        """
        # Global time budget: if we have reached or exceeded the deadline,
        # return a heuristic value for the current node without further
        # descent or expansion.
        if deadline is not None and time.time() >= deadline:
            if state.game_status == "completed":
                return self._calculate_terminal_value(state, depth)
            return self.evaluate_position(state)

        # Check if terminal
        if state.game_status == "completed":
            return self._calculate_terminal_value(state, depth)

        # Depth guard to prevent stack overflow in degenerate cases
        if depth >= MAX_SEARCH_DEPTH:
            return self.evaluate_position(state)

        state_key = self._get_state_key(state)

        # Check if state is in transposition table
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            current_val, children_values, status, remaining_moves, visits = (
                self._unpack_tt_entry(entry)
            )
            visits += 1

            # If proven, stop searching this branch
            if status != NodeStatus.HEURISTIC:
                return current_val

            # Progressive widening: expand one additional child on large boards.
            if (
                self._use_progressive_widening(state.board.type)
                and remaining_moves
                and len(children_values)
                < self._max_children_allowed(visits, state.board.type)
            ):
                if deadline is None or time.time() < deadline:
                    next_move, next_prob = remaining_moves.pop(0)
                    next_state = self.rules_engine.apply_move(state, next_move)
                    if next_state.game_status == "completed":
                        if next_state.winner == self.player_number:
                            next_val = 1.0
                        elif next_state.winner is not None:
                            next_val = -1.0
                        else:
                            next_val = 0.0
                    else:
                        next_val = self.evaluate_position(next_state)

                    children_values[str(next_move)] = (
                        next_move,
                        next_val,
                        float(next_prob),
                        0,
                    )
                    # Refresh current value after widening.
                    if state.current_player == self.player_number:
                        current_val = max(v[1] for v in children_values.values())
                    else:
                        current_val = min(v[1] for v in children_values.values())

            # Select best child to descend
            if not children_values:
                return current_val
            maximizing = state.current_player == self.player_number
            best_move_key = self._select_child_key(
                children_values,
                parent_visits=visits,
                maximizing=maximizing,
            )

            best_move = children_values[best_move_key][0]

            # Descend using the canonical rules engine
            next_state = self.rules_engine.apply_move(state, best_move)
            val = self._descent_iteration(
                next_state,
                depth + 1,
                deadline=deadline,
            )

            # Update child value and per-child visit count.
            old_data = children_values[best_move_key]
            prob = 0.0
            child_visits = 0
            if isinstance(old_data, tuple):
                if len(old_data) > 2:
                    try:
                        prob = float(old_data[2])
                    except Exception:
                        prob = 0.0
                if len(old_data) > 3:
                    try:
                        child_visits = int(old_data[3])
                    except Exception:
                        child_visits = 0
            children_values[best_move_key] = (best_move, val, prob, child_visits + 1)

            # Update current node value and status
            if state.current_player == self.player_number:
                new_best_val = max(v[1] for v in children_values.values())

                # Check for proven status.
                # If any child is PROVEN_WIN (1.0), we are PROVEN_WIN.
                # If ALL children are PROVEN_LOSS (-1.0), we are PROVEN_LOSS.
                # Note: 1.0/-1.0 are reserved for proven outcomes.
                if any(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                elif all(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                else:
                    new_status = NodeStatus.HEURISTIC
            else:
                new_best_val = min(v[1] for v in children_values.values())

                # If any child is PROVEN_LOSS (-1.0), we are PROVEN_LOSS
                # (opponent wins). If ALL children are PROVEN_WIN (1.0),
                # we are PROVEN_WIN (opponent loses).
                if any(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                elif all(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                else:
                    new_status = NodeStatus.HEURISTIC

            self.transposition_table.put(
                state_key,
                (new_best_val, children_values, new_status, remaining_moves, visits),
            )

            # Log update (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                features, _ = self.neural_net._extract_features(state)
                self.search_log.append((features, new_best_val))

            return new_best_val

        else:
            # Expand node using host-level move generation (includes
            # bookkeeping moves when required).
            valid_moves = self.rules_engine.get_valid_moves(
                state,
                state.current_player,
            )

            if not valid_moves:
                return 0.0

            # Get policy if available
            move_probs: Dict[str, float] = {}
            if self.neural_net:
                try:
                    from .neural_net import INVALID_MOVE_INDEX  # Lazy import
                    _, policy_batch = (
                        self.nn_batcher.evaluate([state])
                        if self.nn_batcher is not None
                        else self.neural_net.evaluate_batch([state])
                    )
                    policy_probs = policy_batch[0]

                    use_hex_encoder = (
                        self.hex_encoder is not None
                        and state.board.type == BoardType.HEXAGONAL
                    )

                    total_prob = 0.0
                    for m in valid_moves:
                        idx = (
                            self.hex_encoder.encode_move(m, state.board)  # type: ignore[union-attr]
                            if use_hex_encoder
                            else self.neural_net.encode_move(m, state.board)
                        )
                        if (
                            idx != INVALID_MOVE_INDEX
                            and 0 <= idx < len(policy_probs)
                        ):
                            p = float(policy_probs[idx])
                            move_probs[str(m)] = p
                            total_prob += p

                    if total_prob > 0:
                        for move_key in move_probs:
                            move_probs[move_key] /= total_prob
                    else:
                        uniform = 1.0 / len(valid_moves)
                        for m in valid_moves:
                            move_probs[str(m)] = uniform
                except Exception:
                    # Fallback if NN fails entirely (e.g. missing weights)
                    pass

            # Progressive widening: on large boards, only expand top-K moves
            # by prior initially and store the rest for later widening.
            use_pw = self._use_progressive_widening(state.board.type)
            ordered_moves = list(valid_moves)
            remaining_moves: list[tuple[Move, float]] = []
            if use_pw and move_probs:
                ordered_moves.sort(
                    key=lambda m: move_probs.get(str(m), 0.0),
                    reverse=True,
                )

            initial_k = (
                min(
                    self._max_children_allowed(1, state.board.type),
                    len(ordered_moves),
                )
                if use_pw
                else len(ordered_moves)
            )
            expand_moves = ordered_moves[:initial_k]
            if use_pw:
                tail = ordered_moves[initial_k:initial_k + self._pw_max_remaining_store]
                remaining_moves = [
                    (m, float(move_probs.get(str(m), 0.0))) for m in tail
                ]

            children_values = {}
            if state.current_player == self.player_number:
                best_val = float("-inf")
            else:
                best_val = float("inf")

            expanded_children: List[Tuple[Move, float, Optional[float]]] = []
            non_terminal_states: List[GameState] = []

            for move in expand_moves:
                next_state = self.rules_engine.apply_move(state, move)

                # If we are out of time, stop expanding and return the
                # current best_val so far.
                if deadline is not None and time.time() >= deadline:
                    break

                move_key = str(move)
                prob = move_probs.get(move_key, 0.0)

                # Evaluate leaf (batch non-terminal children).
                if next_state.game_status == "completed":
                    if next_state.winner == self.player_number:
                        val = 1.0
                    elif next_state.winner is not None:
                        val = -1.0
                    else:
                        val = 0.0
                    expanded_children.append((move, prob, val))
                else:
                    expanded_children.append((move, prob, None))
                    non_terminal_states.append(next_state)

            non_terminal_values = (
                self._batch_evaluate_positions(non_terminal_states)
                if non_terminal_states
                else []
            )

            nt_idx = 0
            for move, prob, val in expanded_children:
                if val is None:
                    if nt_idx < len(non_terminal_values):
                        val = non_terminal_values[nt_idx]
                    else:
                        # Defensive fallback: evaluate individually.
                        val = self.evaluate_position(non_terminal_states[nt_idx])
                    nt_idx += 1

                move_key = str(move)
                children_values[move_key] = (move, val, prob, 0)

                if state.current_player == self.player_number:
                    best_val = max(best_val, val)
                else:
                    best_val = min(best_val, val)

            # Determine status
            status = NodeStatus.HEURISTIC
            if best_val == 1.0 and state.current_player == self.player_number:
                status = NodeStatus.PROVEN_WIN
            elif (
                best_val == -1.0
                and state.current_player != self.player_number
            ):
                status = NodeStatus.PROVEN_LOSS

            self.transposition_table.put(
                state_key,
                (best_val, children_values, status, remaining_moves, 1),
            )

            # Log initial visit (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                features, _ = self.neural_net._extract_features(state)
                self.search_log.append((features, best_val))

            return best_val

    # =========================================================================
    # Incremental Search Methods (Make/Unmake Pattern)
    # =========================================================================

    def _descent_iteration_mutable(
        self,
        state: MutableGameState,
        depth: int = 0,
        deadline: Optional[float] = None,
    ) -> float:
        """
        Perform one iteration of Descent search using make/unmake pattern.

        This is the incremental version that avoids object allocation by
        modifying MutableGameState in-place and using MoveUndo tokens.
        """
        # Global time budget: if we have reached or exceeded the deadline,
        # return a heuristic value for the current node without further
        # descent or expansion.
        if deadline is not None and time.time() >= deadline:
            if state.is_game_over():
                return self._calculate_terminal_value_mutable(state, depth)
            return self._evaluate_mutable(state)

        # Check if terminal
        if state.is_game_over():
            return self._calculate_terminal_value_mutable(state, depth)

        # Depth guard to prevent stack overflow in degenerate cases
        if depth >= MAX_SEARCH_DEPTH:
            return self._evaluate_mutable(state)

        state_key = state.zobrist_hash

        # Check if state is in transposition table
        entry = self.transposition_table.get(state_key)
        if entry is not None:
            current_val, children_values, status, remaining_moves, visits = (
                self._unpack_tt_entry(entry)
            )
            visits += 1

            # If proven, stop searching this branch
            if status != NodeStatus.HEURISTIC:
                return current_val

            # Progressive widening: expand one additional child on large boards.
            if (
                self._use_progressive_widening(state.board_type)
                and remaining_moves
                and len(children_values)
                < self._max_children_allowed(visits, state.board_type)
            ):
                if deadline is None or time.time() < deadline:
                    next_move, next_prob = remaining_moves.pop(0)
                    undo_pw = state.make_move(next_move)
                    if state.is_game_over():
                        winner = state.get_winner()
                        if winner == self.player_number:
                            next_val = 1.0
                        elif winner is not None:
                            next_val = -1.0
                        else:
                            next_val = 0.0
                    else:
                        next_val = self._evaluate_mutable(state)
                    state.unmake_move(undo_pw)

                    children_values[str(next_move)] = (
                        next_move,
                        next_val,
                        float(next_prob),
                        0,
                    )
                    if state.current_player == self.player_number:
                        current_val = max(v[1] for v in children_values.values())
                    else:
                        current_val = min(v[1] for v in children_values.values())

            # Select best child to descend
            if not children_values:
                return current_val
            maximizing = state.current_player == self.player_number
            best_move_key = self._select_child_key(
                children_values,
                parent_visits=visits,
                maximizing=maximizing,
            )

            best_move = children_values[best_move_key][0]

            # Descend using make/unmake pattern
            undo = state.make_move(best_move)
            val = self._descent_iteration_mutable(
                state,
                depth + 1,
                deadline=deadline,
            )
            state.unmake_move(undo)

            # Update child value and per-child visit count.
            old_data = children_values[best_move_key]
            prob = 0.0
            child_visits = 0
            if isinstance(old_data, tuple):
                if len(old_data) > 2:
                    try:
                        prob = float(old_data[2])
                    except Exception:
                        prob = 0.0
                if len(old_data) > 3:
                    try:
                        child_visits = int(old_data[3])
                    except Exception:
                        child_visits = 0
            children_values[best_move_key] = (best_move, val, prob, child_visits + 1)

            # Update current node value and status
            if state.current_player == self.player_number:
                new_best_val = max(v[1] for v in children_values.values())

                # Check for proven status.
                # If any child is PROVEN_WIN (1.0), we are PROVEN_WIN.
                # If ALL children are PROVEN_LOSS (-1.0), we are PROVEN_LOSS.
                # Note: 1.0/-1.0 are reserved for proven outcomes.
                if any(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                elif all(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                else:
                    new_status = NodeStatus.HEURISTIC
            else:
                new_best_val = min(v[1] for v in children_values.values())

                # If any child is PROVEN_LOSS (-1.0), we are PROVEN_LOSS
                # (opponent wins). If ALL children are PROVEN_WIN (1.0),
                # we are PROVEN_WIN (opponent loses).
                if any(v[1] == -1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_LOSS
                elif all(v[1] == 1.0 for v in children_values.values()):
                    new_status = NodeStatus.PROVEN_WIN
                else:
                    new_status = NodeStatus.HEURISTIC

            self.transposition_table.put(
                state_key,
                (new_best_val, children_values, new_status, remaining_moves, visits),
            )

            # Log update (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                immutable = state.to_immutable()
                features, _ = self.neural_net._extract_features(immutable)
                self.search_log.append((features, new_best_val))

            return new_best_val

        else:
            # Expand node - get valid moves via conversion to immutable
            immutable = state.to_immutable()
            valid_moves = self.rules_engine.get_valid_moves(
                immutable,
                state.current_player,
            )

            if not valid_moves:
                return 0.0

            # Get policy if available
            move_probs: Dict[str, float] = {}
            if self.neural_net:
                try:
                    from .neural_net import INVALID_MOVE_INDEX  # Lazy import
                    _, policy_batch = (
                        self.nn_batcher.evaluate([immutable])
                        if self.nn_batcher is not None
                        else self.neural_net.evaluate_batch([immutable])
                    )
                    policy_probs = policy_batch[0]

                    use_hex_encoder = (
                        self.hex_encoder is not None
                        and state.board_type == BoardType.HEXAGONAL
                    )

                    total_prob = 0.0
                    for m in valid_moves:
                        idx = (
                            self.hex_encoder.encode_move(m, immutable.board)  # type: ignore[union-attr]
                            if use_hex_encoder
                            else self.neural_net.encode_move(m, immutable.board)
                        )
                        if (
                            idx != INVALID_MOVE_INDEX
                            and 0 <= idx < len(policy_probs)
                        ):
                            p = float(policy_probs[idx])
                            move_probs[str(m)] = p
                            total_prob += p

                    if total_prob > 0:
                        for move_key in move_probs:
                            move_probs[move_key] /= total_prob
                    else:
                        uniform = 1.0 / len(valid_moves)
                        for m in valid_moves:
                            move_probs[str(m)] = uniform
                except Exception:
                    # Fallback if NN fails entirely (e.g. missing weights)
                    pass

            # Progressive widening on large boards.
            use_pw = self._use_progressive_widening(state.board_type)
            ordered_moves = list(valid_moves)
            remaining_moves: list[tuple[Move, float]] = []
            if use_pw and move_probs:
                ordered_moves.sort(
                    key=lambda m: move_probs.get(str(m), 0.0),
                    reverse=True,
                )

            initial_k = (
                min(
                    self._max_children_allowed(1, state.board_type),
                    len(ordered_moves),
                )
                if use_pw
                else len(ordered_moves)
            )
            expand_moves = ordered_moves[:initial_k]
            if use_pw:
                tail = ordered_moves[initial_k:initial_k + self._pw_max_remaining_store]
                remaining_moves = [
                    (m, float(move_probs.get(str(m), 0.0))) for m in tail
                ]

            children_values = {}
            if state.current_player == self.player_number:
                best_val = float("-inf")
            else:
                best_val = float("inf")

            if not self.neural_net:
                # Preserve existing heuristic path when NN is unavailable.
                for move in expand_moves:
                    undo = state.make_move(move)

                    # If we are out of time, stop expanding and return the
                    # current best_val so far.
                    if deadline is not None and time.time() >= deadline:
                        state.unmake_move(undo)
                        break

                    # Evaluate leaf
                    if state.is_game_over():
                        winner = state.get_winner()
                        if winner == self.player_number:
                            val = 1.0
                        elif winner is not None:
                            val = -1.0
                        else:
                            val = 0.0
                    else:
                        val = self._evaluate_mutable(state)

                    state.unmake_move(undo)

                    move_key = str(move)
                    prob = move_probs.get(move_key, 0.0)
                    children_values[move_key] = (move, val, prob, 0)

                    if state.current_player == self.player_number:
                        best_val = max(best_val, val)
                    else:
                        best_val = min(best_val, val)
            else:
                expanded_children: List[Tuple[Move, float, Optional[float]]] = []
                non_terminal_states: List[GameState] = []

                for move in expand_moves:
                    undo = state.make_move(move)

                    # If we are out of time, stop expanding and return the
                    # current best_val so far.
                    if deadline is not None and time.time() >= deadline:
                        state.unmake_move(undo)
                        break

                    move_key = str(move)
                    prob = move_probs.get(move_key, 0.0)

                    # Evaluate leaf (batch non-terminal children).
                    if state.is_game_over():
                        winner = state.get_winner()
                        if winner == self.player_number:
                            val = 1.0
                        elif winner is not None:
                            val = -1.0
                        else:
                            val = 0.0
                        expanded_children.append((move, prob, val))
                    else:
                        immutable_child = state.to_immutable()
                        expanded_children.append((move, prob, None))
                        non_terminal_states.append(immutable_child)

                    state.unmake_move(undo)

                non_terminal_values = (
                    self._batch_evaluate_positions(non_terminal_states)
                    if non_terminal_states
                    else []
                )

                nt_idx = 0
                for move, prob, val in expanded_children:
                    if val is None:
                        if nt_idx < len(non_terminal_values):
                            val = non_terminal_values[nt_idx]
                        else:
                            val = self.evaluate_position(non_terminal_states[nt_idx])
                        nt_idx += 1

                    move_key = str(move)
                    children_values[move_key] = (move, val, prob, 0)

                    if state.current_player == self.player_number:
                        best_val = max(best_val, val)
                    else:
                        best_val = min(best_val, val)

            # Determine status
            status = NodeStatus.HEURISTIC
            if best_val == 1.0 and state.current_player == self.player_number:
                status = NodeStatus.PROVEN_WIN
            elif (
                best_val == -1.0
                and state.current_player != self.player_number
            ):
                status = NodeStatus.PROVEN_LOSS

            self.transposition_table.put(
                state_key,
                (best_val, children_values, status, remaining_moves, 1),
            )

            # Log initial visit (only if collecting training data)
            if self.collect_training_data and self.neural_net:
                features, _ = self.neural_net._extract_features(immutable)
                self.search_log.append((features, best_val))

            return best_val

    def _calculate_terminal_value_mutable(
        self, state: MutableGameState, depth: int
    ) -> float:
        """Calculate terminal value for mutable state.

        Includes bonuses and discount for tie-breaking.
        """
        base_val = 0.0
        winner = state.get_winner()
        if winner == self.player_number:
            base_val = 1.0
        elif winner is not None:
            base_val = -1.0
        else:
            # Draw
            return 0.0

        # Bonuses for tie-breaking metrics (Territory, Eliminated, Markers)
        # Territory
        territory_count = 0
        for p_id in state.collapsed_spaces.values():
            if p_id == self.player_number:
                territory_count += 1

        # Eliminated
        player_state = state.players.get(self.player_number)
        eliminated_count = player_state.eliminated_rings if player_state else 0

        # Markers
        marker_count = 0
        for m in state.markers.values():
            if m.player == self.player_number:
                marker_count += 1

        # Normalize bonuses to be small (max ~0.05 total). This ensures they
        # act as tie-breakers and don't override the win/loss signal.
        bonus = (
            (territory_count * 0.001)
            + (eliminated_count * 0.001)
            + (marker_count * 0.0001)
        )

        if base_val > 0:
            val = base_val + bonus
        else:
            # If losing, we still prefer having more territory/rings as
            # tie-breakers.
            val = base_val + bonus

        # Discount for depth (fewest moves to win).
        # Win fast (val > 0): val * gamma^depth decreases with depth.
        # Lose slow (val < 0): val * gamma^depth increases (towards 0)
        # with depth.
        gamma = 0.99
        discounted_val = val * (gamma ** depth)

        # Ensure we don't exceed bounds or flip sign.
        if base_val > 0:
            return max(0.001, min(1.0, discounted_val))
        elif base_val < 0:
            return max(-1.0, min(-0.001, discounted_val))
        return 0.0

    def _evaluate_mutable(self, state: MutableGameState) -> float:
        """Evaluate MutableGameState using neural net or heuristic.

        Converts to immutable for neural net evaluation, or uses
        FastGeometry-backed heuristic when available.
        """
        val = 0.0
        num_players = len(state.players)
        if self.neural_net:
            immutable = state.to_immutable()
            val = self.evaluate_position(immutable)
        elif self._heuristic_evaluator is not None:
            # Use FastGeometry-backed heuristic evaluator (requires immutable state)
            try:
                immutable = state.to_immutable()
                val = self._heuristic_fallback_eval(immutable)
            except Exception:
                # Fall through to simple heuristic
                val = self._simple_mutable_eval(state, num_players)
        else:
            # Simple material difference fallback for MutableGameState
            val = self._simple_mutable_eval(state, num_players)

        # Clamp value to (-0.99, 0.99) to reserve 1.0/-1.0 for proven
        # terminal states.
        return max(-0.99, min(0.99, val))

    def _simple_mutable_eval(self, state: MutableGameState, num_players: int) -> float:
        """Simple material evaluation directly on MutableGameState."""
        player_state = state.players.get(self.player_number)
        my_elim = player_state.eliminated_rings if player_state else 0

        opp_elims = [
            ps.eliminated_rings
            for pid, ps in state.players.items()
            if pid != self.player_number
        ]
        if num_players <= 2:
            opp_elim = sum(opp_elims)
            return (my_elim - opp_elim) * 0.05
        else:
            # Multi-player Paranoid reduction: compare victory progress
            # (max of territory/elimination/LPS proximity) against the
            # leading opponent.
            my_prog = victory_progress_for_player(state, self.player_number)
            opp_prog = max(
                (
                    victory_progress_for_player(state, pid)
                    for pid in state.players.keys()
                    if pid != self.player_number
                ),
                default=0.0,
            )
            return my_prog - opp_prog

    def _batch_evaluate_positions(
        self, game_states: List[GameState]
    ) -> List[float]:
        """Batch-evaluate immutable states with NN when available.

        Returns values in this agent's fixed perspective. NeuralNetAI encodes
        features relative to each state's ``current_player``; for Paranoid
        root-vs-coalition reductions (3p/4p) we therefore negate values when
        evaluating any non-root-to-move state (treating all opponents as a
        single minimizing coalition).
        """
        if not game_states:
            return []

        if not self.neural_net:
            # Try GPU batch heuristic if available
            if self.gpu_heuristic is not None and len(game_states) >= 4:
                try:
                    return self._gpu_batch_heuristic_eval(game_states)
                except Exception:
                    logger.warning("GPU heuristic batch eval failed, falling back to CPU", exc_info=True)
            return [self.evaluate_position(s) for s in game_states]

        try:
            use_vector_head = (
                self.use_vector_value_head
                and bool(game_states)
                and infer_num_players(game_states[0]) > 2
            )
            value_head = (self.player_number - 1) if use_vector_head else None

            if self.nn_batcher and self.enable_async_nn_eval:
                batch_size = int(self._default_nn_batch_size())
                step = max(1, batch_size)
                max_pending = 2

                adjusted: List[float] = []
                pending: list[tuple[List[GameState], Any]] = []

                def _drain_one() -> None:
                    chunk, fut = pending.pop(0)
                    values, _policy = fut.result()
                    for val, st in zip(values, chunk):
                        v = float(val)
                        if (not use_vector_head) and (
                            st.current_player != self.player_number
                        ):
                            v = -v
                        adjusted.append(max(-0.99, min(0.99, v)))

                for i in range(0, len(game_states), step):
                    chunk = game_states[i:i + step]
                    pending.append(
                        (
                            chunk,
                            self.nn_batcher.submit(chunk, value_head=value_head),
                        )
                    )
                    if len(pending) >= max_pending:
                        _drain_one()

                while pending:
                    _drain_one()

                return adjusted

            if self.nn_batcher:
                values, _ = self.nn_batcher.evaluate(
                    game_states,
                    value_head=value_head,
                )
            else:
                values, _ = self.neural_net.evaluate_batch(
                    game_states,
                    value_head=value_head,
                )

            adjusted: List[float] = []
            for val, st in zip(values, game_states):
                v = float(val)
                if (not use_vector_head) and (
                    st.current_player != self.player_number
                ):
                    v = -v
                adjusted.append(max(-0.99, min(0.99, v)))
            return adjusted
        except Exception:
            if self.require_neural_net:
                logger.error(
                    "DescentAI neural batch evaluation failed with RINGRIFT_REQUIRE_NEURAL_NET=1; "
                    "raising instead of falling back to heuristic evaluation",
                    exc_info=True,
                )
                raise
            logger.warning(
                "DescentAI neural batch evaluation failed; falling back to heuristic evaluation",
                exc_info=True,
            )
            self.neural_net = None
            self.hex_encoder = None
            self.nn_batcher = None
            return [self.evaluate_position(s) for s in game_states]

    def _gpu_batch_heuristic_eval(self, game_states: List[GameState]) -> List[float]:
        """Batch-evaluate positions using full-parity GPU heuristic.

        Uses the 49-feature evaluate_positions_batch for CPU heuristic parity,
        rather than the simplified GPUHeuristicEvaluator.

        Converts game states to BatchGameState format and evaluates in parallel.
        Returns values adjusted to this agent's perspective.
        """
        # Import here to avoid circular deps at module load
        from .gpu_parallel_games import BatchGameState, evaluate_positions_batch
        from .heuristic_weights import get_weights_for_player_count
        from .gpu_batch import get_device

        # Convert states to BatchGameState for full-parity evaluation
        device = get_device() if self.gpu_heuristic is None else self.gpu_heuristic.device
        batch_state = BatchGameState.from_game_states(game_states, device=device)

        # Get weights for current player count
        num_players = len(game_states[0].players) if game_states else 2
        weights = get_weights_for_player_count(num_players)

        # Full 49-feature GPU evaluation - returns (batch_size, num_players+1) tensor
        scores_tensor = evaluate_positions_batch(batch_state, weights)

        # Convert to list and adjust perspective
        adjusted: List[float] = []
        for i, st in enumerate(game_states):
            # Get score for this player from the batch result
            val = float(scores_tensor[i, self.player_number].item())
            # Negate if opponent to move (paranoid perspective)
            if st.current_player != self.player_number:
                val = -val
            # Clamp to avoid extreme values
            adjusted.append(max(-0.99, min(0.99, val / 100.0)))  # Normalize heuristic range

        return adjusted

    # =========================================================================
    # Legacy Search Methods (Immutable State)
    # =========================================================================

    def _get_state_key(self, state: GameState) -> int:
        """Generate a unique key for the game state using Zobrist hashing"""
        if state.zobrist_hash is not None:
            return state.zobrist_hash
        # Fallback if hash is missing (shouldn't happen with updated engine)
        from .zobrist import ZobristHash
        return ZobristHash().compute_initial_hash(state)

    def _calculate_terminal_value(self, state: GameState, depth: int) -> float:
        """Calculate terminal value with bonuses and discount"""
        base_val = 0.0
        if state.winner == self.player_number:
            base_val = 1.0
        elif state.winner is not None:
            base_val = -1.0
        else:
            # Draw
            return 0.0

        # Bonuses for tie-breaking metrics (Territory, Eliminated, Markers)
        # Bonuses for tie-breaking metrics (Territory, Eliminated, Markers)
        # Territory
        territory_count = 0
        for p_id in state.board.collapsed_spaces.values():
            if p_id == self.player_number:
                territory_count += 1

        # Eliminated
        eliminated_count = state.board.eliminated_rings.get(
            str(self.player_number),
            0,
        )

        # Markers
        marker_count = 0
        for m in state.board.markers.values():
            if m.player == self.player_number:
                marker_count += 1

        # Normalize bonuses to be small (max ~0.05 total). This ensures they
        # act as tie-breakers and don't override the win/loss signal.
        bonus = (
            (territory_count * 0.001)
            + (eliminated_count * 0.001)
            + (marker_count * 0.0001)
        )

        if base_val > 0:
            val = base_val + bonus
        else:
            # If losing, we still prefer having more territory/rings as
            # tie-breakers.
            val = base_val + bonus

        # Discount for depth (fewest moves to win).
        # Win fast (val > 0): val * gamma^depth decreases with depth.
        # Lose slow (val < 0): val * gamma^depth increases (towards 0)
        # with depth.
        gamma = 0.99
        discounted_val = val * (gamma ** depth)

        # Ensure we don't exceed bounds or flip sign.
        if base_val > 0:
            return max(0.001, min(1.0, discounted_val))
        elif base_val < 0:
            return max(-1.0, min(-0.001, discounted_val))
        return 0.0

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using neural net or heuristic"""
        val = 0.0
        if self.neural_net:
            try:
                use_vector_head = (
                    self.use_vector_value_head
                    and infer_num_players(game_state) > 2
                )
                value_head = (self.player_number - 1) if use_vector_head else None

                if self.nn_batcher:
                    values, _ = self.nn_batcher.evaluate(
                        [game_state],
                        value_head=value_head,
                    )
                    val = float(values[0]) if values else 0.0
                else:
                    values, _ = self.neural_net.evaluate_batch(
                        [game_state],
                        value_head=value_head,
                    )
                    val = float(values[0]) if values else 0.0

                # Default path: NeuralNetAI encodes features relative to the state's
                # current_player, so convert to this agent's fixed perspective by
                # negating opponent-to-move states.
                #
                # When using a vector value head we assume the selected head already
                # represents this agent's utility, so no turn-based negation is
                # applied here.
                if (not use_vector_head) and (
                    game_state.current_player != self.player_number
                ):
                    val = -val
            except Exception:
                if self.require_neural_net:
                    logger.error(
                        "DescentAI neural evaluation failed with RINGRIFT_REQUIRE_NEURAL_NET=1; "
                        "raising instead of falling back to heuristic evaluation",
                        exc_info=True,
                    )
                    raise
                logger.warning(
                    "DescentAI neural evaluation failed; falling back to heuristic evaluation",
                    exc_info=True,
                )
                self.neural_net = None
                self.hex_encoder = None
                self.nn_batcher = None
                # Fall through to the heuristic evaluation below.
                val = self._heuristic_fallback_eval(game_state)
        else:
            # Heuristic fallback using FastGeometry-backed evaluator when available
            val = self._heuristic_fallback_eval(game_state)

        # Clamp value to (-0.99, 0.99) to reserve 1.0/-1.0 for proven
        # terminal states.
        return max(-0.99, min(0.99, val))

    def _heuristic_fallback_eval(self, game_state: GameState) -> float:
        """Evaluate position using FastGeometry-backed heuristic or simple material diff.

        Uses HeuristicEvaluator when available (with FastGeometry for pre-computed
        geometry tables), otherwise falls back to simple eliminated rings difference.

        Returns:
            Evaluation score in range (-1.0, 1.0), positive favoring this player.
        """
        # Use FastGeometry-backed evaluator when available
        if self._heuristic_evaluator is not None:
            try:
                # HeuristicEvaluator.evaluate returns a score in heuristic units
                # Normalize to (-1, 1) range for tree search compatibility
                raw_score = self._heuristic_evaluator.evaluate(game_state)
                # Clamp and normalize: typical heuristic scores range ~(-100, 100)
                return max(-0.99, min(0.99, raw_score / 100.0))
            except Exception:
                # Fall through to simple heuristic on any failure
                pass

        # Simple material difference fallback
        my_elim = game_state.board.eliminated_rings.get(
            str(self.player_number),
            0,
        )

        opp_elim = 0
        for pid, count in game_state.board.eliminated_rings.items():
            if int(pid) != self.player_number:
                opp_elim += count

        return (my_elim - opp_elim) * 0.05
