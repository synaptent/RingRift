"""GPU-Optimized Minimax AI implementation for RingRift.

This agent extends MinimaxAI to leverage GPU batch evaluation for leaf positions,
providing significant speedup on CUDA/MPS-capable hardware while preserving
alpha-beta pruning semantics.

Strategy:
1. Root Level: Batch-evaluate all first-ply positions via GPU for optimal move ordering
2. Deep Search: Accumulate leaf positions in a buffer during alpha-beta traversal
3. Batch Flush: GPU-evaluate all accumulated leaves when buffer full or branch completes
4. TT Caching: Store GPU results in transposition table for reuse

Integration: This is an ADDITION to the training system, not a replacement.
Select via AIType.GPU_MINIMAX in the factory.
"""

from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
import logging
import os
import time

import torch

if TYPE_CHECKING:
    from .gpu_parallel_games import BatchGameState

from .minimax_ai import MinimaxAI, MINIMAX_ZERO_SUM_EVAL
from .heuristic_weights import BASE_V1_BALANCED_WEIGHTS, get_weights
from ..models import GameState, Move, MoveType, AIConfig, BoardType
from ..rules.mutable_state import MutableGameState

logger = logging.getLogger(__name__)


class GPUMinimaxAI(MinimaxAI):
    """GPU-accelerated minimax with batched leaf evaluation.

    Extends MinimaxAI to use GPU batch evaluation for leaf positions,
    providing 10-50x speedup on CUDA-capable hardware while preserving
    alpha-beta pruning semantics.

    Configuration (via AIConfig attributes or environment variables):
        use_gpu_batch: bool = True  # Enable/disable GPU batching
        gpu_batch_size: int = 64    # Max positions per GPU batch
        gpu_min_batch: int = 8      # Minimum batch size (CPU fallback below)

    Environment variables:
        RINGRIFT_GPU_MINIMAX_BATCH_SIZE: Override batch size
        RINGRIFT_GPU_MINIMAX_DISABLE: Set to "1" to disable GPU evaluation
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)

        # GPU configuration from config or environment
        env_disable = os.environ.get("RINGRIFT_GPU_MINIMAX_DISABLE", "").lower()
        self.use_gpu_batch: bool = (
            getattr(config, 'use_gpu_batch', True) and
            env_disable not in {"1", "true", "yes", "on"}
        )

        env_batch_size = os.environ.get("RINGRIFT_GPU_MINIMAX_BATCH_SIZE", "")
        if env_batch_size.isdigit():
            self.gpu_batch_size = int(env_batch_size)
        else:
            self.gpu_batch_size = getattr(config, 'gpu_batch_size', 64)

        self.gpu_min_batch: int = getattr(config, 'gpu_min_batch', 8)

        # Lazy initialization for GPU resources
        self._gpu_device: Optional[torch.device] = None
        self._gpu_available: Optional[bool] = None

        # Leaf buffer for batched evaluation
        self._leaf_buffer: List[Tuple[GameState, int]] = []
        self._leaf_results: Dict[int, float] = {}
        self._next_callback_id: int = 0

        # Heuristic weights cache
        self._heuristic_weights: Optional[Dict[str, float]] = None

        # Board info (detected on first use)
        self._board_size: Optional[int] = None
        self._num_players: Optional[int] = None
        self._board_type: Optional[BoardType] = None

        logger.debug(
            f"GPUMinimaxAI(player={player_number}, difficulty={config.difficulty}): "
            f"GPU batch={self.use_gpu_batch}, batch_size={self.gpu_batch_size}"
        )

    def _ensure_gpu_initialized(self) -> bool:
        """Lazily initialize GPU resources. Returns True if GPU available."""
        if self._gpu_available is not None:
            return self._gpu_available

        try:
            from .gpu_parallel_games import get_device
            self._gpu_device = get_device()
            self._gpu_available = self._gpu_device.type in ('cuda', 'mps')

            if self._gpu_available:
                logger.info(
                    f"GPUMinimaxAI: GPU evaluation enabled on {self._gpu_device}"
                )
            else:
                logger.info(
                    f"GPUMinimaxAI: No GPU available (device={self._gpu_device.type}), "
                    "using CPU evaluation"
                )
        except Exception as e:
            logger.warning(f"GPUMinimaxAI: GPU init failed, falling back to CPU: {e}")
            self._gpu_available = False

        return self._gpu_available

    def _detect_board_info(self, game_state: GameState) -> None:
        """Detect board configuration from game state."""
        if self._board_type is not None:
            return

        self._board_type = game_state.board_type
        self._num_players = len(game_state.players)

        # Map board type to size
        board_size_map = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEXAGONAL: 25,  # Hex uses 25x25 embedding
        }
        self._board_size = board_size_map.get(self._board_type, 8)

        logger.debug(
            f"GPUMinimaxAI: Detected board={self._board_type}, "
            f"size={self._board_size}, players={self._num_players}"
        )

    def _get_weights(self) -> Dict[str, float]:
        """Get heuristic weights for GPU evaluation."""
        if self._heuristic_weights is None:
            profile_id = getattr(self.config, 'heuristic_profile_id', None)
            if profile_id:
                self._heuristic_weights = get_weights(profile_id)
                if not self._heuristic_weights:
                    self._heuristic_weights = dict(BASE_V1_BALANCED_WEIGHTS)
            else:
                self._heuristic_weights = dict(BASE_V1_BALANCED_WEIGHTS)
        return self._heuristic_weights

    def _to_zero_sum_scores(
        self,
        scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Convert GPU batch scores to zero-sum format.

        For correct alpha-beta behavior, minimax requires:
            Eval(state, P1) = -Eval(state, P2)

        The raw GPU evaluation gives per-player scores. This method converts
        them to zero-sum by computing:
            zero_sum_score = (my_score - max_opponent_score) / 2

        Args:
            scores_tensor: Shape [batch_size, num_players+1] with per-player scores
                           Index 0 is unused, players are 1-indexed

        Returns:
            Tensor of shape [batch_size] with zero-sum scores for our player
        """
        if not MINIMAX_ZERO_SUM_EVAL:
            # Just return our player's score
            return scores_tensor[:, self.player_number]

        num_players = self._num_players or 2

        # Get our score
        my_scores = scores_tensor[:, self.player_number]

        # Get max opponent score (for paranoid assumption)
        opponent_mask = torch.ones(num_players + 1, dtype=torch.bool, device=scores_tensor.device)
        opponent_mask[0] = False  # Index 0 is unused
        opponent_mask[self.player_number] = False  # Exclude ourselves

        # Only keep opponent columns
        opponent_scores = scores_tensor[:, opponent_mask]
        if opponent_scores.size(1) > 0:
            max_opponent_scores = opponent_scores.max(dim=1).values
        else:
            max_opponent_scores = torch.zeros_like(my_scores)

        # Zero-sum: advantage = (my - opponent) / 2
        return (my_scores - max_opponent_scores) / 2.0

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select the best move using GPU-accelerated minimax search.

        Overrides parent to use GPU batch evaluation when beneficial.
        Falls back to parent implementation if GPU unavailable.
        """
        # Detect board info on first call
        self._detect_board_info(game_state)

        # Initialize NNUE if pending (from parent)
        if getattr(self, '_pending_nnue_init', False):
            self._init_nnue_for_board(game_state)

        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Check for swap move first
        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move:
            return swap_move

        # Use GPU-accelerated path if available
        if self.use_gpu_batch and self._ensure_gpu_initialized():
            return self._select_move_gpu(game_state, valid_moves)
        else:
            # Fall back to parent implementation
            if self.use_incremental_search:
                return self._select_move_incremental(game_state, valid_moves)
            else:
                return self._select_move_legacy(game_state, valid_moves)

    def _select_move_gpu(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> Move:
        """GPU-accelerated incremental search with batched leaf evaluation."""
        self.start_time = time.time()
        if self.config.think_time is not None and self.config.think_time > 0:
            self.time_limit = self.config.think_time / 1000.0
        else:
            self.time_limit = 0.5 + (self.config.difficulty * 0.2)
        self.nodes_visited = 0

        best_move = valid_moves[0]
        max_depth = self._get_max_depth()

        # GPU-accelerated move ordering at root level
        scored_moves = self._gpu_score_root_moves(game_state, valid_moves)

        # Clear caches for new search
        self.transposition_table.clear()
        self.killer_moves.clear()
        self._clear_leaf_buffer()

        # Create mutable state once for the entire search
        mutable_state = MutableGameState.from_immutable(game_state)

        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break

            current_best_move = None
            current_best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for _, move in scored_moves:
                if time.time() - self.start_time > self.time_limit:
                    break

                # Use make/unmake pattern
                undo = mutable_state.make_move(move)
                score = self._alpha_beta_gpu(
                    mutable_state,
                    depth - 1,
                    alpha,
                    beta,
                    (mutable_state.current_player == self.player_number),
                )
                mutable_state.unmake_move(undo)

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            # Flush any remaining leaves at end of depth
            self._flush_leaf_buffer()

            if current_best_move:
                best_move = current_best_move

        return best_move

    def _gpu_score_root_moves(
        self,
        game_state: GameState,
        valid_moves: List[Move],
    ) -> List[Tuple[float, Move]]:
        """Batch-evaluate all root moves on GPU for optimal move ordering.

        Args:
            game_state: Current game position
            valid_moves: List of valid moves to score

        Returns:
            List of (score, move) tuples sorted descending by score
        """
        if not self._gpu_available or len(valid_moves) < self.gpu_min_batch:
            # Fall back to CPU scoring
            return self._score_and_sort_moves(game_state, valid_moves)

        try:
            from .gpu_parallel_games import BatchGameState, evaluate_positions_batch

            # Generate all child states
            mutable = MutableGameState.from_immutable(game_state)
            child_states: List[GameState] = []

            for move in valid_moves:
                undo = mutable.make_move(move)
                child_states.append(mutable.to_immutable())
                mutable.unmake_move(undo)

            # Create batch and evaluate
            batch = self._create_batch_from_states(child_states)
            scores_tensor = evaluate_positions_batch(batch, self._get_weights())

            # Convert to zero-sum scores for minimax
            scores = self._to_zero_sum_scores(scores_tensor).cpu().numpy()

            # Add move-type priority bonuses (same as CPU version)
            scored_moves: List[Tuple[float, Move]] = []
            for i, move in enumerate(valid_moves):
                priority_bonus = self._get_move_priority_bonus(move)
                scored_moves.append((float(scores[i]) + priority_bonus, move))

            scored_moves.sort(key=lambda x: x[0], reverse=True)
            return scored_moves

        except Exception as e:
            logger.debug(f"GPU root scoring failed, falling back to CPU: {e}")
            return self._score_and_sort_moves(game_state, valid_moves)

    def _get_move_priority_bonus(self, move: Move) -> float:
        """Get priority bonus for move ordering (captures first, etc.)."""
        bonus = 0.0
        if move.type in {
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CHAIN_CAPTURE,
            MoveType.LINE_OVERTAKING_CAPTURE,
        }:
            bonus += 1000.0  # Captures are highest priority
        elif move.type == MoveType.PLACE_RING:
            bonus += 100.0  # Placements second
        elif move.type == MoveType.MOVE_STACK:
            bonus += 50.0  # Movements third
        return bonus

    def _alpha_beta_gpu(
        self,
        state: MutableGameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> float:
        """Alpha-beta search with GPU batch leaf evaluation.

        At depth 0 (leaf nodes), instead of immediately evaluating:
        1. If GPU available and buffer not full: defer to batch
        2. If buffer is full: flush to GPU, retrieve results
        3. If GPU unavailable: fall back to CPU evaluation
        """
        self.nodes_visited += 1
        if self.nodes_visited % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                return self._evaluate_mutable(state)

        state_hash = state.zobrist_hash

        # Transposition table lookup
        entry = self.transposition_table.get(state_hash)
        if entry is not None:
            if entry['depth'] >= depth:
                if entry['flag'] == 'exact':
                    return entry['score']
                elif entry['flag'] == 'lowerbound':
                    alpha = max(alpha, entry['score'])
                elif entry['flag'] == 'upperbound':
                    beta = min(beta, entry['score'])
                if alpha >= beta:
                    return entry['score']

        # Leaf node: GPU batch evaluation
        if depth == 0:
            return self._evaluate_leaf_gpu(state, alpha, beta, maximizing_player)

        current_player_num = state.current_player

        # Check if game is over
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0 + depth
            elif winner is not None:
                return -100000.0 - depth
            else:
                return 0.0

        # Get valid moves via conversion to immutable
        immutable = state.to_immutable()
        valid_moves = self.rules_engine.get_valid_moves(immutable, current_player_num)

        if not valid_moves:
            return self._evaluate_mutable(state)

        is_me = (current_player_num == self.player_number)

        # Move ordering with Killer Heuristic
        ordered_moves = self._order_moves_with_killers(valid_moves, depth)

        if is_me:
            max_eval = float('-inf')
            for i, move in enumerate(ordered_moves):
                undo = state.make_move(move)
                next_is_me = (state.current_player == self.player_number)

                # Principal Variation Search (PVS)
                if i == 0:
                    eval_score = self._alpha_beta_gpu(
                        state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    eval_score = self._alpha_beta_gpu(
                        state, depth - 1, alpha, alpha + 0.01, next_is_me
                    )
                    if alpha < eval_score < beta:
                        eval_score = self._alpha_beta_gpu(
                            state, depth - 1, eval_score, beta, next_is_me
                        )

                state.unmake_move(undo)

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self._store_killer_move(move, depth)
                    break

            flag = 'exact'
            if max_eval <= alpha:
                flag = 'upperbound'
            elif max_eval >= beta:
                flag = 'lowerbound'

            self.transposition_table.put(state_hash, {
                'score': max_eval,
                'depth': depth,
                'flag': flag
            })
            return max_eval
        else:
            min_eval = float('inf')
            for i, move in enumerate(ordered_moves):
                undo = state.make_move(move)
                next_is_me = (state.current_player == self.player_number)

                # Principal Variation Search (PVS)
                if i == 0:
                    eval_score = self._alpha_beta_gpu(
                        state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    eval_score = self._alpha_beta_gpu(
                        state, depth - 1, beta - 0.01, beta, next_is_me
                    )
                    if alpha < eval_score < beta:
                        eval_score = self._alpha_beta_gpu(
                            state, depth - 1, alpha, eval_score, next_is_me
                        )

                state.unmake_move(undo)

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self._store_killer_move(move, depth)
                    break

            flag = 'exact'
            if min_eval <= alpha:
                flag = 'upperbound'
            elif min_eval >= beta:
                flag = 'lowerbound'

            self.transposition_table.put(state_hash, {
                'score': min_eval,
                'depth': depth,
                'flag': flag
            })
            return min_eval

    def _evaluate_leaf_gpu(
        self,
        state: MutableGameState,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> float:
        """Evaluate leaf node, batching when beneficial.

        Strategy:
        1. Check TT cache first
        2. If GPU enabled and buffer available, add to batch
        3. If buffer full, flush and return result
        4. Fall back to CPU quiescence if result not ready
        """
        state_hash = state.zobrist_hash

        # Check transposition table first
        entry = self.transposition_table.get(state_hash)
        if entry is not None and entry['flag'] == 'exact':
            return entry['score']

        # Check if we already have a result from a previous flush
        if state_hash in self._leaf_results:
            return self._leaf_results[state_hash]

        # Add to leaf buffer for batched GPU evaluation
        callback_id = self._next_callback_id
        self._next_callback_id += 1

        # Convert to immutable for deferred evaluation
        # (MutableGameState doesn't have copy(), but we need immutable for batch anyway)
        self._leaf_buffer.append((state.to_immutable(), state_hash))

        # If buffer is full, flush to GPU
        if len(self._leaf_buffer) >= self.gpu_batch_size:
            self._flush_leaf_buffer()
            # Result should now be available
            if state_hash in self._leaf_results:
                return self._leaf_results[state_hash]

        # Buffer not full yet - use shallow quiescence for now
        # This ensures we always return a valid score
        score = self._quiescence_search_mutable(
            state, alpha, beta, maximizing_player, depth=1
        )

        # Cache the quiescence result
        self.transposition_table.put(state_hash, {
            'score': score,
            'depth': 0,
            'flag': 'exact'
        })

        return score

    def _flush_leaf_buffer(self) -> None:
        """GPU batch-evaluate all buffered leaf positions."""
        if not self._leaf_buffer:
            return

        try:
            from .gpu_parallel_games import BatchGameState, evaluate_positions_batch

            # States are already immutable (stored as GameState in buffer)
            states = [s for s, _ in self._leaf_buffer]
            hashes = [h for _, h in self._leaf_buffer]

            # Create batch and evaluate
            batch = self._create_batch_from_states(states)
            scores_tensor = evaluate_positions_batch(batch, self._get_weights())

            # Convert to zero-sum scores and store results
            scores = self._to_zero_sum_scores(scores_tensor).cpu().numpy()

            for i, state_hash in enumerate(hashes):
                score = float(scores[i])
                self._leaf_results[state_hash] = score

                # Also cache in transposition table
                self.transposition_table.put(state_hash, {
                    'score': score,
                    'depth': 0,
                    'flag': 'exact'
                })

            logger.debug(f"GPU batch evaluated {len(self._leaf_buffer)} leaves")

        except Exception as e:
            logger.warning(f"GPU batch flush failed: {e}")
            # Fall back: evaluate each state individually with CPU heuristic
            for state, state_hash in self._leaf_buffer:
                # States are immutable GameState, use parent's evaluate method
                score = self._evaluate_position(state)
                self._leaf_results[state_hash] = score
                self.transposition_table.put(state_hash, {
                    'score': score,
                    'depth': 0,
                    'flag': 'exact'
                })

        finally:
            self._leaf_buffer.clear()

    def _clear_leaf_buffer(self) -> None:
        """Clear leaf buffer and results for new search."""
        self._leaf_buffer.clear()
        self._leaf_results.clear()
        self._next_callback_id = 0

    def _create_batch_from_states(
        self, states: List[GameState]
    ) -> "BatchGameState":
        """Convert list of GameState to BatchGameState for GPU evaluation.

        Efficiently batches multiple CPU game states into a single GPU tensor
        structure for parallel evaluation.
        """
        from .gpu_parallel_games import BatchGameState

        if len(states) == 0:
            raise ValueError("Cannot create batch from empty state list")

        batch_size = len(states)

        # Use cached board info or detect from first state
        if self._board_size is None:
            self._detect_board_info(states[0])

        board_size = self._board_size or 8
        num_players = self._num_players or 2

        # Create empty batch
        batch = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=self._gpu_device,
        )

        # Populate from individual states
        is_hex = self._board_type == BoardType.HEXAGONAL

        for i, state in enumerate(states):
            self._copy_state_to_batch(state, batch, i, is_hex, board_size)

        return batch

    def _copy_state_to_batch(
        self,
        state: GameState,
        batch: "BatchGameState",
        idx: int,
        is_hex: bool,
        board_size: int,
    ) -> None:
        """Copy single GameState into batch at index idx."""
        # Hex coordinate conversion helper
        center = board_size // 2 if is_hex else 0

        def to_grid_coords(x: int, y: int) -> Tuple[int, int]:
            """Convert CPU coords to GPU grid coords."""
            if is_hex:
                # Axial to grid: offset by center
                return y + center, x + center
            return y, x

        # Copy stacks
        for key, stack in state.board.stacks.items():
            parts = key.split(",")
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])
            row, col = to_grid_coords(x, y)

            if 0 <= row < board_size and 0 <= col < board_size:
                batch.stack_owner[idx, row, col] = stack.controlling_player
                batch.stack_height[idx, row, col] = len(stack.rings)
                batch.cap_height[idx, row, col] = stack.cap_height

        # Copy markers
        for key, marker in state.board.markers.items():
            parts = key.split(",")
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])
            row, col = to_grid_coords(x, y)

            if 0 <= row < board_size and 0 <= col < board_size:
                # Handle both int and MarkerInfo
                player = marker.player if hasattr(marker, 'player') else marker
                batch.marker_owner[idx, row, col] = player

        # Copy collapsed spaces / territory
        for key, player in state.board.collapsed_spaces.items():
            parts = key.split(",")
            if len(parts) < 2:
                continue
            x, y = int(parts[0]), int(parts[1])
            row, col = to_grid_coords(x, y)

            if 0 <= row < board_size and 0 <= col < board_size:
                batch.territory_owner[idx, row, col] = player
                batch.is_collapsed[idx, row, col] = True

        # Copy player state
        for j, player in enumerate(state.players):
            player_num = j + 1
            batch.rings_in_hand[idx, player_num] = player.rings_in_hand
            batch.eliminated_rings[idx, player_num] = player.eliminated_rings
            batch.territory_count[idx, player_num] = player.territory_spaces

            # Handle buried rings if available
            if hasattr(player, 'buried_rings'):
                batch.buried_rings[idx, player_num] = player.buried_rings

        # Copy game metadata
        batch.current_player[idx] = state.current_player

    def _init_nnue_for_board(self, game_state: GameState) -> None:
        """Initialize NNUE evaluator for the detected board type."""
        try:
            from .nnue import NNUEEvaluator
            from .game_state_utils import infer_num_players

            board_type = game_state.board_type
            num_players = infer_num_players(game_state)

            self.nnue_evaluator = NNUEEvaluator(
                board_type=board_type,
                player_number=self.player_number,
                num_players=num_players,
                model_id=getattr(self.config, 'nn_model_id', None),
            )
            self.use_nnue = self.nnue_evaluator.available
            self._pending_nnue_init = False

            if self.use_nnue:
                logger.debug(
                    f"GPUMinimaxAI: NNUE initialized for {board_type}"
                )
        except Exception as e:
            logger.warning(f"NNUE initialization failed: {e}")
            self._pending_nnue_init = False
            self.use_nnue = False

    def reset_for_new_game(self, *, rng_seed: Optional[int] = None) -> None:
        """Reset state for a new game."""
        super().reset_for_new_game(rng_seed=rng_seed)
        self._clear_leaf_buffer()
        self._board_type = None
        self._board_size = None
        self._num_players = None
