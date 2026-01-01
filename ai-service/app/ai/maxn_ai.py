"""Max-N Minimax AI implementation for RingRift.

Max-N is a multi-player extension of minimax that models each player as
self-interested (maximizing their own score) rather than paranoid (assuming
all opponents collude against us).

Algorithm:
- Each node returns a score vector (one score per player)
- At each node, the current player picks the child that maximizes
  their component of the score vector
- Terminal nodes return the evaluation for all players

This is more realistic than Paranoid for games where opponents act
independently to maximize their own scores.

Note: Classic Max-N does not support alpha-beta pruning, though
shallow pruning techniques exist (not implemented here).

GPU Acceleration (default enabled):
- Leaf position evaluation is batched and run on GPU for 10-50x speedup
- Uses CPU rules engine for full parity (no GPU move generation)
- Automatic fallback to CPU if no GPU available
- Control via RINGRIFT_GPU_MAXN_DISABLE=1 environment variable
- Shadow validation available via RINGRIFT_GPU_MAXN_SHADOW_VALIDATE=1
"""

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ..models import AIConfig, BoardType, GameState, Move
from ..rules.mutable_state import MutableGameState
from .bounded_transposition_table import BoundedTranspositionTable
from .heuristic_ai import HeuristicAI
from .zobrist import ZobristHash

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Environment variable controls
_GPU_MAXN_DISABLE = os.environ.get("RINGRIFT_GPU_MAXN_DISABLE", "").lower() in (
    "1", "true", "yes", "on"
)
_GPU_MAXN_SHADOW_VALIDATE = os.environ.get("RINGRIFT_GPU_MAXN_SHADOW_VALIDATE", "").lower() in (
    "1", "true", "yes", "on"
)


class MaxNAI(HeuristicAI):
    """AI using Max-N search for multiplayer games.

    Max-N assumes each player maximizes their own score independently,
    which is more realistic than Paranoid search when opponents don't
    coordinate against you.

    The evaluation function returns a score vector where each element
    represents how good the position is for that player.

    GPU Acceleration (default enabled):
        - Batch evaluates leaf positions on GPU for 10-50x speedup
        - Full rule parity via CPU rules engine (only evaluation is GPU)
        - Automatic fallback to CPU if GPU unavailable
        - Control: RINGRIFT_GPU_MAXN_DISABLE=1 to disable
        - Shadow validation: RINGRIFT_GPU_MAXN_SHADOW_VALIDATE=1 to enable parity checks

    Neural Network Evaluation (optional, Dec 2025):
        - Set use_neural_net=True in AIConfig to enable
        - Uses per-player value heads from full NN or NNUE
        - Falls back to heuristic if NN unavailable
        - Control: RINGRIFT_MAXN_USE_NEURAL_NET=1 to force enable
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)
        self.transposition_table: BoundedTranspositionTable = (
            BoundedTranspositionTable(max_entries=50000)  # Smaller - no pruning
        )
        self.zobrist: ZobristHash = ZobristHash()
        self.start_time: float = 0.0
        self.time_limit: float = 0.0
        self.nodes_visited: int = 0

        # Cache number of players (detected on first call)
        self._num_players: int | None = None

        # GPU acceleration state (lazy initialized)
        self._gpu_enabled: bool = not _GPU_MAXN_DISABLE
        self._gpu_available: bool | None = None  # None = not yet checked
        self._gpu_device: torch.device | None = None
        self._gpu_evaluator: Any | None = None  # GPUHeuristicEvaluator

        # Leaf buffer for batched GPU evaluation
        # Stores (immutable_state, state_hash) - immutable to avoid mutation during search
        self._leaf_buffer: list[tuple[GameState, int]] = []
        self._leaf_results: dict[int, dict[int, float]] = {}  # hash -> {player: score}
        self._gpu_batch_size: int = getattr(config, 'gpu_batch_size', 64)
        self._gpu_min_batch: int = getattr(config, 'gpu_min_batch', 4)

        # Board configuration (detected on first call)
        self._board_size: int | None = None
        self._board_type: BoardType | None = None

        # Shadow validation for GPU/CPU parity checking
        self._shadow_validator: Any | None = None
        if _GPU_MAXN_SHADOW_VALIDATE:
            self._init_shadow_validator()

        # Neural network evaluation support (Dec 2025 - Phase 2)
        # Enables using Full NN or NNUE for position evaluation
        nn_env = os.environ.get("RINGRIFT_MAXN_USE_NEURAL_NET", "").lower() in (
            "1", "true", "yes", "on"
        )
        self.use_neural_net: bool = getattr(config, 'use_neural_net', False) or nn_env
        self._neural_net: Any | None = None  # NeuralNetAI (lazy loaded)
        self._nnue_evaluator: Any | None = None  # RingRiftNNUEWithPolicy (lazy loaded)
        self._multi_player_nnue: Any | None = None  # MultiPlayerNNUE (lazy loaded)
        self._nn_initialized: bool = False

        # Turn-based depth and branching optimization (Dec 2025)
        # Turn-based depth counts by player changes, not individual moves
        self.turn_based_depth: bool = getattr(config, 'turn_based_depth', True)
        # Max moves to consider at each node (None = all moves)
        self.max_branching_factor: int | None = getattr(config, 'max_branching_factor', None)
        # Skip search for single-move positions (forced moves)
        self.forced_move_extension: bool = getattr(config, 'forced_move_extension', True)

        logger.debug(
            f"MaxNAI(player={player_number}, difficulty={config.difficulty}, "
            f"gpu_enabled={self._gpu_enabled}, use_neural_net={self.use_neural_net}, "
            f"turn_based_depth={self.turn_based_depth})"
        )

    def _get_max_depth(self) -> int:
        """Get maximum search depth based on difficulty.

        Max-N is more expensive than alpha-beta (no pruning), so use
        shallower depths for equivalent difficulty.
        """
        if self.config.difficulty >= 9:
            return 4
        elif self.config.difficulty >= 7:
            return 3
        elif self.config.difficulty >= 4:
            return 2
        else:
            return 1

    # =========================================================================
    # GPU Acceleration Methods
    # =========================================================================

    def _ensure_gpu_initialized(self) -> bool:
        """Lazily initialize GPU resources. Returns True if GPU available."""
        if self._gpu_available is not None:
            return self._gpu_available

        if not self._gpu_enabled:
            self._gpu_available = False
            logger.debug("MaxNAI: GPU disabled via environment variable")
            return False

        try:
            from .gpu_batch import get_device

            self._gpu_device = get_device(prefer_gpu=True)
            self._gpu_available = self._gpu_device.type in ('cuda', 'mps')

            if self._gpu_available:
                # Note: Using full-parity 49-feature GPU heuristic (evaluate_positions_batch)
                # instead of simplified GPUHeuristicEvaluator for CPU parity
                logger.info(
                    f"MaxNAI: GPU acceleration enabled on {self._gpu_device} "
                    f"(full 49-feature parity, batch_size={self._gpu_batch_size})"
                )
            else:
                logger.info(
                    f"MaxNAI: No GPU available (device={self._gpu_device.type}), "
                    "using CPU evaluation"
                )
        except Exception as e:
            logger.warning(f"MaxNAI: GPU initialization failed, using CPU: {e}")
            self._gpu_available = False

        return self._gpu_available

    def _detect_board_config(self, game_state: GameState) -> None:
        """Detect board configuration from first game state."""
        if self._board_type is not None:
            return

        self._board_type = game_state.board_type
        self._num_players = len(game_state.players)

        # Map board type to size
        board_size_map = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEXAGONAL: 25,
        }
        self._board_size = board_size_map.get(self._board_type, 8)

        logger.debug(
            f"MaxNAI: Detected board={self._board_type}, "
            f"size={self._board_size}, players={self._num_players}"
        )

    def _init_shadow_validator(self) -> None:
        """Initialize shadow validator for GPU/CPU parity checking."""
        try:
            from .shadow_validation import ShadowValidator
            self._shadow_validator = ShadowValidator(
                sample_rate=0.05,  # 5% of evaluations
                threshold=0.01,   # 1% max divergence (higher tolerance for float diffs)
                halt_on_threshold=False,  # Log warnings but don't halt
            )
            logger.info("MaxNAI: Shadow validation enabled for GPU/CPU parity")
        except Exception as e:
            logger.warning(f"MaxNAI: Shadow validator init failed: {e}")
            self._shadow_validator = None

    def _clear_leaf_buffer(self) -> None:
        """Clear leaf buffer and results for new search."""
        self._leaf_buffer.clear()
        self._leaf_results.clear()

    def _flush_leaf_buffer(self) -> None:
        """GPU batch-evaluate all buffered leaf positions for all players.

        This is the core GPU acceleration: instead of evaluating each position
        individually with CPU heuristics, we batch them for parallel GPU evaluation.

        Uses the full 49-feature GPU heuristic (evaluate_positions_batch) for
        CPU parity, rather than the simplified GPUHeuristicEvaluator.
        """
        if not self._leaf_buffer or not self._gpu_available:
            return

        try:
            from .gpu_parallel_games import BatchGameState
            from .gpu_heuristic import evaluate_positions_batch
            from .heuristic_weights import get_weights_for_player_count

            # States are already immutable (converted when added to buffer)
            immutable_states = [state for state, _ in self._leaf_buffer]
            hashes = [h for _, h in self._leaf_buffer]

            # Create BatchGameState for full-parity evaluation
            batch_state = BatchGameState.from_game_states(
                immutable_states,
                device=self._gpu_device,
            )

            # Get weights for current player count (uses CMA-ES optimized weights)
            num_players = self._num_players or 2
            weights = get_weights_for_player_count(num_players)

            # Full 49-feature GPU evaluation - returns (batch_size, num_players+1) tensor
            scores_tensor = evaluate_positions_batch(batch_state, weights)

            # Store results indexed by state hash
            for i, state_hash in enumerate(hashes):
                player_scores: dict[int, float] = {}
                for player_num in range(1, num_players + 1):
                    player_scores[player_num] = float(scores_tensor[i, player_num].item())
                self._leaf_results[state_hash] = player_scores

            # Shadow validation if enabled
            if self._shadow_validator is not None:
                self._validate_batch(immutable_states, hashes)

            logger.debug(f"MaxNAI: GPU batch evaluated {len(self._leaf_buffer)} leaves (full parity)")

        except Exception as e:
            logger.warning(f"MaxNAI: GPU batch flush failed, falling back to CPU: {e}")
            import traceback
            logger.debug(f"MaxNAI: Traceback: {traceback.format_exc()}")
            # Fall back to CPU evaluation for this batch
            for state, state_hash in self._leaf_buffer:
                mutable = MutableGameState.from_immutable(state)
                self._leaf_results[state_hash] = self._evaluate_all_players_cpu(mutable)

        finally:
            self._leaf_buffer.clear()

    def _validate_batch(
        self,
        states: list[GameState],
        hashes: list[int],
    ) -> None:
        """Validate GPU results against CPU for a sample of positions."""
        if self._shadow_validator is None:
            return

        import random
        sample_rate = 0.05  # Check 5% of batch

        for _i, (state, state_hash) in enumerate(zip(states, hashes, strict=False)):
            if random.random() > sample_rate:
                continue

            # Get GPU result
            gpu_result = self._leaf_results.get(state_hash, {})

            # Compute CPU result
            mutable = MutableGameState.from_immutable(state)
            cpu_result = self._evaluate_all_players_cpu(mutable)

            # Check for significant divergence (> 1% relative difference)
            for player_num in gpu_result:
                gpu_score = gpu_result.get(player_num, 0.0)
                cpu_score = cpu_result.get(player_num, 0.0)

                if abs(cpu_score) > 0.01:  # Avoid division by zero
                    divergence = abs(gpu_score - cpu_score) / abs(cpu_score)
                    if divergence > 0.01:  # 1% tolerance
                        logger.warning(
                            f"MaxNAI: GPU/CPU eval divergence for player {player_num}: "
                            f"GPU={gpu_score:.2f}, CPU={cpu_score:.2f} ({divergence:.1%})"
                        )

    def _evaluate_all_players_cpu(self, state: MutableGameState) -> dict[int, float]:
        """CPU fallback for evaluating position for all players."""
        # Check for terminal state
        if state.is_game_over():
            winner = state.get_winner()
            scores = {}
            for p in range(1, (self._num_players or 2) + 1):
                if winner == p:
                    scores[p] = 100000.0
                elif winner is not None:
                    scores[p] = -100000.0
                else:
                    scores[p] = 0.0
            return scores

        # Use heuristic evaluation for each player's perspective
        immutable = state.to_immutable()
        scores = {}
        original_player = self.player_number

        for p in range(1, (self._num_players or 2) + 1):
            self.player_number = p
            scores[p] = self.evaluate_position(immutable)

        # Restore original player
        self.player_number = original_player
        return scores

    def _ensure_nn_initialized(self) -> bool:
        """Lazily initialize neural network resources. Returns True if NN available."""
        if self._nn_initialized:
            return (
                self._neural_net is not None
                or self._nnue_evaluator is not None
                or self._multi_player_nnue is not None
            )

        self._nn_initialized = True

        if not self.use_neural_net:
            return False

        num_players = self._num_players or 2

        # For multiplayer (3+), prefer MultiPlayerNNUE for true per-player scores
        if num_players >= 3 and self._board_type is not None:
            try:
                from .nnue import MultiPlayerNNUE
                self._multi_player_nnue = MultiPlayerNNUE(
                    num_players=num_players,
                    board_type=self._board_type,
                    hidden_dim=256,
                    num_hidden_layers=2,
                )
                logger.info(
                    f"MaxNAI: Loaded MultiPlayerNNUE for {num_players}p "
                    f"board_type={self._board_type.value}"
                )
                return True
            except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
                logger.debug(f"MaxNAI: Could not load MultiPlayerNNUE: {e}")

        # Try to load Full NN
        try:
            from .neural_net import NeuralNetAI
            self._neural_net = NeuralNetAI(self.player_number, self.config)
            logger.info(
                f"MaxNAI: Loaded Full NN for player {self.player_number}"
            )
            return True
        except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.debug(f"MaxNAI: Could not load Full NN: {e}")

        # Fall back to NNUE (scalar output, will be converted to per-player)
        try:
            from .nnue_policy import RingRiftNNUEWithPolicy
            if self._board_type is not None:
                self._nnue_evaluator = RingRiftNNUEWithPolicy(
                    board_type=self._board_type,
                    hidden_dim=128,
                    num_hidden_layers=2,
                )
                logger.info(
                    f"MaxNAI: Loaded NNUE for board_type={self._board_type.value}"
                )
                return True
        except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.debug(f"MaxNAI: Could not load NNUE: {e}")

        logger.warning("MaxNAI: Neural network requested but not available, using heuristic")
        return False

    def _evaluate_all_players_nn(self, state: MutableGameState) -> dict[int, float] | None:
        """Evaluate position for ALL players using neural network.

        Returns None if NN not available (caller should fall back to heuristic).

        Evaluation priority:
            1. MultiPlayerNNUE - true per-player values (best for MaxN/BRS)
            2. Full NN - per-player value head
            3. NNUE - single value converted to per-player perspective
        """
        if not self._ensure_nn_initialized():
            return None

        immutable = state.to_immutable()
        num_players = self._num_players or 2

        # MultiPlayerNNUE: Direct per-player scores (best quality)
        if self._multi_player_nnue is not None:
            try:
                from .nnue import extract_features_from_gamestate
                features = extract_features_from_gamestate(immutable, state.current_player)
                per_player_values = self._multi_player_nnue.forward_single(features)
                scores = {}
                for p in range(1, num_players + 1):
                    if p - 1 < len(per_player_values):
                        # Scale from [-1, 1] to heuristic range [-100, 100]
                        scores[p] = float(per_player_values[p - 1]) * 100.0
                    else:
                        scores[p] = 0.0
                return scores
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"MaxNAI: MultiPlayerNNUE evaluation failed: {e}")
                # Fall through to other NN types

        if self._neural_net is not None:
            # Full NN: Get per-player value estimates
            try:
                # Use vector value head for multiplayer
                values, _ = self._neural_net.evaluate_batch(
                    [immutable],
                    value_head=None,  # Get all value heads
                )
                if values is not None and len(values) > 0:
                    val = values[0]
                    # If val is a vector (multiplayer), extract per-player
                    if hasattr(val, '__iter__') and not isinstance(val, (int, float)):
                        scores = {}
                        for p in range(1, num_players + 1):
                            if p - 1 < len(val):
                                scores[p] = float(val[p - 1]) * 100.0  # Scale to heuristic range
                            else:
                                scores[p] = 0.0
                        return scores
                    else:
                        # Scalar value - convert to per-player
                        v = float(val)
                        scores = {}
                        current_player = state.current_player
                        for p in range(1, num_players + 1):
                            if p == current_player:
                                scores[p] = v * 100.0
                            else:
                                scores[p] = -v * 100.0 / (num_players - 1)
                        return scores
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"MaxNAI: NN evaluation failed: {e}")
                return None

        if self._nnue_evaluator is not None:
            # NNUE: Single value, convert to per-player perspective
            try:
                # NNUE evaluates from current player's perspective
                val = self._nnue_evaluator.evaluate(immutable)
                v = float(val) if val is not None else 0.0
                scores = {}
                current_player = state.current_player
                for p in range(1, num_players + 1):
                    if p == current_player:
                        scores[p] = v * 100.0
                    else:
                        scores[p] = -v * 100.0 / (num_players - 1)
                return scores
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"MaxNAI: NNUE evaluation failed: {e}")
                return None

        return None

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using Max-N search.

        Returns:
            The move that maximizes this player's score component
            in the resulting score vector.
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Detect board configuration and number of players
        self._detect_board_config(game_state)
        if self._num_players is None:
            self._num_players = len(game_state.players)

        # Initialize GPU on first use (after board config detected)
        if self._gpu_enabled and self._gpu_available is None:
            self._ensure_gpu_initialized()

        # Initialize search parameters
        self.start_time = time.time()
        if self.config.think_time is not None and self.config.think_time > 0:
            self.time_limit = self.config.think_time / 1000.0
        else:
            self.time_limit = 0.3 + (self.config.difficulty * 0.15)
        self.nodes_visited = 0

        # Clear caches for new search
        self.transposition_table.clear()
        self._clear_leaf_buffer()

        max_depth = self._get_max_depth()
        mutable_state = MutableGameState.from_immutable(game_state)

        best_move = valid_moves[0]
        float('-inf')

        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break

            current_best_move = None
            current_best_score = float('-inf')

            for move in valid_moves:
                if time.time() - self.start_time > self.time_limit:
                    break

                undo = mutable_state.make_move(move)
                score_vector = self._maxn_search(mutable_state, depth - 1)
                mutable_state.unmake_move(undo)

                # Extract our score from the vector
                my_score = score_vector[self.player_number]

                if my_score > current_best_score:
                    current_best_score = my_score
                    current_best_move = move

            # Flush any remaining GPU leaf buffer after each depth
            if self._gpu_available:
                self._flush_leaf_buffer()

            if current_best_move:
                best_move = current_best_move

        # Final flush for any remaining leaves
        if self._gpu_available:
            self._flush_leaf_buffer()

        return best_move

    def _maxn_search(
        self,
        state: MutableGameState,
        depth: int
    ) -> dict[int, float]:
        """Recursive Max-N search.

        Args:
            state: Current game state (mutable)
            depth: Remaining search depth

        Returns:
            Dictionary mapping player_number -> score
        """
        self.nodes_visited += 1

        # Time check
        if self.nodes_visited % 500 == 0 and time.time() - self.start_time > self.time_limit:
            return self._evaluate_all_players(state)

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            return self._evaluate_all_players(state)

        # Transposition table lookup
        state_hash = state.zobrist_hash
        entry = self.transposition_table.get(state_hash)
        if entry is not None and entry.get('depth', 0) >= depth:
            return entry['scores']

        current_player = state.current_player
        immutable = state.to_immutable()
        valid_moves = self.rules_engine.get_valid_moves(immutable, current_player)

        if not valid_moves:
            return self._evaluate_all_players(state)

        # Forced move extension (Dec 2025): Single move = play without depth cost
        if self.forced_move_extension and len(valid_moves) == 1:
            undo = state.make_move(valid_moves[0])
            result = self._maxn_search(state, depth)  # Don't decrement depth
            state.unmake_move(undo)
            return result

        # Apply max branching factor limit (Dec 2025)
        # On large boards with chain captures, there can be 100,000+ moves.
        # Use heuristic ordering to sample the most promising moves.
        if self.max_branching_factor is not None and len(valid_moves) > self.max_branching_factor:
            valid_moves = self._order_moves_for_branching(valid_moves, self.max_branching_factor)

        # Current player picks move that maximizes their own score
        best_scores: dict[int, float] | None = None
        best_my_score = float('-inf')

        for move in valid_moves:
            if time.time() - self.start_time > self.time_limit:
                break

            player_before = current_player
            undo = state.make_move(move)
            player_after = state.current_player

            # Turn-based depth (Dec 2025): Only decrement when player changes
            if self.turn_based_depth:
                next_depth = depth if player_after == player_before else depth - 1
            else:
                next_depth = depth - 1

            child_scores = self._maxn_search(state, next_depth)
            state.unmake_move(undo)

            my_score = child_scores.get(current_player, 0.0)
            if my_score > best_my_score:
                best_my_score = my_score
                best_scores = child_scores

        if best_scores is None:
            best_scores = self._evaluate_all_players(state)

        # Cache result
        self.transposition_table.put(state_hash, {
            'scores': best_scores,
            'depth': depth
        })

        return best_scores

    def _order_moves_for_branching(self, moves: list, limit: int) -> list:
        """Order moves by heuristic priority and return top N for sampling.

        Args:
            moves: List of valid moves
            limit: Maximum moves to return

        Returns:
            Top N moves ordered by priority (captures > attacks > placements)

        Dec 2025: Handles extremely high branching factor scenarios
        (100,000+ chain capture sequences on large boards).
        """
        from app.rules.move_types import MoveType

        # Score moves by type priority
        scored_moves = []
        for move in moves:
            score = 0.0
            move_type = move.get('type') if isinstance(move, dict) else getattr(move, 'type', None)

            # Prioritize captures (most impactful moves)
            if move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT,
                            'OVERTAKING_CAPTURE', 'CONTINUE_CAPTURE_SEGMENT'):
                score = 100.0
            # Then attacks/movements
            elif move_type in (MoveType.MOVE_STACK, 'MOVE_STACK'):
                score = 50.0
            # Then placements
            elif move_type in (MoveType.PLACE_RING, 'PLACE_RING'):
                score = 25.0
            # Everything else
            else:
                score = 10.0

            scored_moves.append((score, move))

        # Sort by score descending and take top N
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves[:limit]]

    def _evaluate_all_players(self, state: MutableGameState) -> dict[int, float]:
        """Evaluate position for ALL players.

        Returns a score vector where each player's score reflects
        how good the position is for them.

        Evaluation priority (Dec 2025):
            1. Neural network (Full NN or NNUE) if use_neural_net=True
            2. GPU heuristic batch evaluation if available
            3. CPU heuristic fallback

        GPU Acceleration:
            When GPU is available, positions are buffered and batch-evaluated
            for 10-50x speedup. Results are cached in transposition table.
            Falls back to CPU evaluation if GPU unavailable.
        """
        # Check for terminal state (always CPU - fast path)
        if state.is_game_over():
            winner = state.get_winner()
            scores = {}
            for p in range(1, (self._num_players or 2) + 1):
                if winner == p:
                    scores[p] = 100000.0
                elif winner is not None:
                    scores[p] = -100000.0
                else:
                    scores[p] = 0.0
            return scores

        state_hash = state.zobrist_hash

        # Check if we already have cached GPU results
        if state_hash in self._leaf_results:
            return self._leaf_results[state_hash]

        # Neural network path (Dec 2025 - Phase 2)
        if self.use_neural_net:
            nn_result = self._evaluate_all_players_nn(state)
            if nn_result is not None:
                return nn_result
            # Fall through to heuristic if NN unavailable

        # GPU path: buffer for batch evaluation
        if self._gpu_available:
            # Convert to immutable BEFORE adding to buffer (avoid mutation during search)
            immutable_state = state.to_immutable()
            self._leaf_buffer.append((immutable_state, state_hash))

            # If buffer is full, flush to GPU now
            if len(self._leaf_buffer) >= self._gpu_batch_size:
                self._flush_leaf_buffer()
                # Check if result is now available
                if state_hash in self._leaf_results:
                    return self._leaf_results[state_hash]

            # Buffer not full yet - use quick CPU estimate
            # This ensures we always return a valid score
            return self._evaluate_all_players_cpu(state)

        # CPU fallback path
        return self._evaluate_all_players_cpu(state)


class BRSAI(HeuristicAI):
    """AI using Best-Reply Search (BRS) for multiplayer games.

    BRS is a simplified multi-player search where each player
    plays their greedy best reply. It's essentially multi-player
    1-ply lookahead iterated for a few rounds.

    Much faster than Max-N but less accurate for deep tactical play.

    Neural Network Evaluation (optional, Dec 2025):
        - Set use_neural_net=True in AIConfig to enable
        - Uses Full NN or NNUE for position evaluation
        - Falls back to heuristic if NN unavailable
        - Control: RINGRIFT_BRS_USE_NEURAL_NET=1 to force enable
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)
        self.start_time: float = 0.0
        self.time_limit: float = 0.0
        self.nodes_visited: int = 0
        self._num_players: int | None = None
        self._board_type: BoardType | None = None

        # Neural network evaluation support (Dec 2025 - Phase 2)
        nn_env = os.environ.get("RINGRIFT_BRS_USE_NEURAL_NET", "").lower() in (
            "1", "true", "yes", "on"
        )
        self.use_neural_net: bool = getattr(config, 'use_neural_net', False) or nn_env
        self._neural_net: Any | None = None  # NeuralNetAI (lazy loaded)
        self._nnue_evaluator: Any | None = None  # RingRiftNNUEWithPolicy (lazy loaded)
        self._multi_player_nnue: Any | None = None  # MultiPlayerNNUE (lazy loaded)
        self._nn_initialized: bool = False

        # Turn-based depth and branching optimization (Dec 2025)
        # BRS already uses "rounds" which is conceptually turn-based
        # Max moves to consider when finding best reply (None = all moves)
        self.max_branching_factor: int | None = getattr(config, 'max_branching_factor', None)
        # Skip search for single-move positions (forced moves)
        self.forced_move_extension: bool = getattr(config, 'forced_move_extension', True)

        logger.debug(
            f"BRSAI(player={player_number}, difficulty={config.difficulty}, "
            f"use_neural_net={self.use_neural_net}, "
            f"max_branching_factor={self.max_branching_factor})"
        )

    def _get_lookahead_rounds(self) -> int:
        """Get number of BRS rounds based on difficulty."""
        if self.config.difficulty >= 7:
            return 3
        elif self.config.difficulty >= 4:
            return 2
        else:
            return 1

    def _ensure_nn_initialized(self) -> bool:
        """Lazily initialize neural network resources. Returns True if NN available."""
        if self._nn_initialized:
            return (
                self._neural_net is not None
                or self._nnue_evaluator is not None
                or self._multi_player_nnue is not None
            )

        self._nn_initialized = True

        if not self.use_neural_net:
            return False

        num_players = self._num_players or 2

        # For multiplayer (3+), prefer MultiPlayerNNUE for true per-player scores
        if num_players >= 3 and self._board_type is not None:
            try:
                from .nnue import MultiPlayerNNUE
                self._multi_player_nnue = MultiPlayerNNUE(
                    num_players=num_players,
                    board_type=self._board_type,
                    hidden_dim=256,
                    num_hidden_layers=2,
                )
                logger.info(
                    f"BRSAI: Loaded MultiPlayerNNUE for {num_players}p "
                    f"board_type={self._board_type.value}"
                )
                return True
            except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
                logger.debug(f"BRSAI: Could not load MultiPlayerNNUE: {e}")

        # Try to load Full NN
        try:
            from .neural_net import NeuralNetAI
            self._neural_net = NeuralNetAI(self.player_number, self.config)
            logger.info(
                f"BRSAI: Loaded Full NN for player {self.player_number}"
            )
            return True
        except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.debug(f"BRSAI: Could not load Full NN: {e}")

        # Fall back to NNUE (scalar output, will be converted to per-player)
        try:
            from .nnue_policy import RingRiftNNUEWithPolicy
            if self._board_type is not None:
                self._nnue_evaluator = RingRiftNNUEWithPolicy(
                    board_type=self._board_type,
                    hidden_dim=128,
                    num_hidden_layers=2,
                )
                logger.info(
                    f"BRSAI: Loaded NNUE for board_type={self._board_type.value}"
                )
                return True
        except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.debug(f"BRSAI: Could not load NNUE: {e}")

        logger.warning("BRSAI: Neural network requested but not available, using heuristic")
        return False

    def _evaluate_for_me_nn(self, state: MutableGameState) -> float | None:
        """Evaluate position using neural network. Returns None if unavailable.

        Evaluation priority:
            1. MultiPlayerNNUE - true per-player values (best for BRS)
            2. Full NN - per-player value head
            3. NNUE - single value (current player perspective)
        """
        if not self._ensure_nn_initialized():
            return None

        immutable = state.to_immutable()

        # MultiPlayerNNUE: Direct per-player scores (best quality)
        if self._multi_player_nnue is not None:
            try:
                from .nnue import extract_features_from_gamestate
                features = extract_features_from_gamestate(immutable, state.current_player)
                # Use evaluate_for_player to get this player's score directly
                score = self._multi_player_nnue.evaluate_for_player(features, self.player_number)
                # Scale from [-1, 1] to heuristic range [-100, 100]
                return score * 100.0
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"BRSAI: MultiPlayerNNUE evaluation failed: {e}")
                # Fall through to other NN types

        if self._neural_net is not None:
            try:
                values, _ = self._neural_net.evaluate_batch([immutable])
                if values is not None and len(values) > 0:
                    val = values[0]
                    # For multiplayer, extract this player's value
                    if hasattr(val, '__iter__') and not isinstance(val, (int, float)):
                        if self.player_number - 1 < len(val):
                            return float(val[self.player_number - 1]) * 100.0
                    return float(val) * 100.0
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"BRSAI: NN evaluation failed: {e}")
                return None

        if self._nnue_evaluator is not None:
            try:
                val = self._nnue_evaluator.evaluate(immutable)
                return float(val) * 100.0 if val is not None else None
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"BRSAI: NNUE evaluation failed: {e}")
                return None

        return None

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using Best-Reply Search.

        For each candidate move, simulates future rounds where each
        player plays their greedy best response.
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Detect number of players and board type
        if self._num_players is None:
            self._num_players = len(game_state.players)
        if self._board_type is None and hasattr(game_state.board, 'board_type'):
            self._board_type = game_state.board.board_type

        # Initialize search parameters
        self.start_time = time.time()
        if self.config.think_time is not None and self.config.think_time > 0:
            self.time_limit = self.config.think_time / 1000.0
        else:
            self.time_limit = 0.2 + (self.config.difficulty * 0.1)
        self.nodes_visited = 0

        rounds = self._get_lookahead_rounds()
        mutable_state = MutableGameState.from_immutable(game_state)

        best_move = valid_moves[0]
        best_score = float('-inf')

        for move in valid_moves:
            if time.time() - self.start_time > self.time_limit:
                break

            # Apply my move
            undo = mutable_state.make_move(move)

            # Simulate future rounds with greedy best replies
            final_score = self._simulate_brs_rounds(mutable_state, rounds - 1)

            mutable_state.unmake_move(undo)

            if final_score > best_score:
                best_score = final_score
                best_move = move

        return best_move

    def _simulate_brs_rounds(
        self,
        state: MutableGameState,
        remaining_rounds: int
    ) -> float:
        """Simulate BRS rounds with greedy best replies.

        Args:
            state: Current game state (mutable)
            remaining_rounds: Number of rounds to simulate

        Returns:
            Evaluation score for original player after simulation

        Dec 2025 optimizations for high branching factor:
        - Forced move extension: single-move positions don't consume a round
        - Max branching factor: limit moves evaluated when finding best reply
        - Turn-based round counting: only decrement when player changes
        """
        self.nodes_visited += 1

        if remaining_rounds == 0 or state.is_game_over():
            return self._evaluate_for_me(state)

        # Each player plays their best reply in turn order
        current_player = state.current_player
        immutable = state.to_immutable()
        valid_moves = self.rules_engine.get_valid_moves(immutable, current_player)

        if not valid_moves:
            return self._evaluate_for_me(state)

        # Forced move extension (Dec 2025): If only one legal move, play it
        # without consuming a round. This accelerates forced phase sequences.
        if self.forced_move_extension and len(valid_moves) == 1:
            move = valid_moves[0]
            undo = state.make_move(move)
            player_after = state.current_player
            # Don't decrement round if player didn't change (still same player's turn)
            next_rounds = remaining_rounds if player_after == current_player else remaining_rounds - 1
            result = self._simulate_brs_rounds(state, next_rounds)
            state.unmake_move(undo)
            return result

        # Apply max branching factor limit (Dec 2025)
        # On large boards with chain captures, there can be 100,000+ moves.
        # We sample the most promising moves based on quick heuristic ordering.
        moves_to_evaluate = valid_moves
        if self.max_branching_factor is not None and len(valid_moves) > self.max_branching_factor:
            # Order moves by quick heuristic (captures first, then by type)
            moves_to_evaluate = self._order_moves_for_sampling(
                valid_moves, state, self.max_branching_factor
            )

        # Find greedy best move for current player
        best_move = None
        best_score = float('-inf')

        for move in moves_to_evaluate:
            if time.time() - self.start_time > self.time_limit:
                break

            undo = state.make_move(move)

            # Evaluate from current player's perspective
            temp_player = self.player_number
            self.player_number = current_player
            score = self._evaluate_for_me(state)
            self.player_number = temp_player

            state.unmake_move(undo)

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            return self._evaluate_for_me(state)

        # Apply best reply and continue
        undo = state.make_move(best_move)
        player_after = state.current_player
        # Turn-based round counting: only decrement when player changes
        next_rounds = remaining_rounds if player_after == current_player else remaining_rounds - 1
        result = self._simulate_brs_rounds(state, next_rounds)
        state.unmake_move(undo)

        return result

    def _order_moves_for_sampling(
        self,
        moves: list,
        state: MutableGameState,
        limit: int
    ) -> list:
        """Order moves by heuristic priority and return top N for sampling.

        Args:
            moves: List of valid moves
            state: Current game state
            limit: Maximum moves to return

        Returns:
            Top N moves ordered by priority (captures > attacks > placements)
        """
        from app.rules.move_types import MoveType

        # Score moves by type priority
        scored_moves = []
        for move in moves:
            score = 0.0
            move_type = move.get('type') if isinstance(move, dict) else getattr(move, 'type', None)

            # Prioritize captures (most impactful moves)
            if move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT,
                            'OVERTAKING_CAPTURE', 'CONTINUE_CAPTURE_SEGMENT'):
                score = 100.0
            # Then attacks/movements
            elif move_type in (MoveType.MOVE_STACK, 'MOVE_STACK'):
                score = 50.0
            # Then placements
            elif move_type in (MoveType.PLACE_RING, 'PLACE_RING'):
                score = 25.0
            # Everything else
            else:
                score = 10.0

            scored_moves.append((score, move))

        # Sort by score descending and take top N
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves[:limit]]

    def _evaluate_for_me(self, state: MutableGameState) -> float:
        """Evaluate position from this AI's perspective.

        Evaluation priority (Dec 2025):
            1. Neural network (Full NN or NNUE) if use_neural_net=True
            2. Heuristic evaluation fallback
        """
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0
            elif winner is not None:
                return -100000.0
            return 0.0

        # Neural network path (Dec 2025 - Phase 2)
        if self.use_neural_net:
            nn_result = self._evaluate_for_me_nn(state)
            if nn_result is not None:
                return nn_result
            # Fall through to heuristic if NN unavailable

        immutable = state.to_immutable()
        return self.evaluate_position(immutable)
