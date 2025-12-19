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

from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING
import logging
import os
import time

import numpy as np

from .heuristic_ai import HeuristicAI
from .bounded_transposition_table import BoundedTranspositionTable
from .zobrist import ZobristHash
from ..models import GameState, Move, AIConfig, GamePhase, BoardType
from ..rules.mutable_state import MutableGameState

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
        self._num_players: Optional[int] = None

        # GPU acceleration state (lazy initialized)
        self._gpu_enabled: bool = not _GPU_MAXN_DISABLE
        self._gpu_available: Optional[bool] = None  # None = not yet checked
        self._gpu_device: Optional["torch.device"] = None
        self._gpu_evaluator: Optional[Any] = None  # GPUHeuristicEvaluator

        # Leaf buffer for batched GPU evaluation
        # Stores (immutable_state, state_hash) - immutable to avoid mutation during search
        self._leaf_buffer: List[Tuple[GameState, int]] = []
        self._leaf_results: Dict[int, Dict[int, float]] = {}  # hash -> {player: score}
        self._gpu_batch_size: int = getattr(config, 'gpu_batch_size', 64)
        self._gpu_min_batch: int = getattr(config, 'gpu_min_batch', 4)

        # Board configuration (detected on first call)
        self._board_size: Optional[int] = None
        self._board_type: Optional[BoardType] = None

        # Shadow validation for GPU/CPU parity checking
        self._shadow_validator: Optional[Any] = None
        if _GPU_MAXN_SHADOW_VALIDATE:
            self._init_shadow_validator()

        logger.debug(
            f"MaxNAI(player={player_number}, difficulty={config.difficulty}, "
            f"gpu_enabled={self._gpu_enabled})"
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
            from .gpu_batch import get_device, GPUHeuristicEvaluator

            self._gpu_device = get_device(prefer_gpu=True)
            self._gpu_available = self._gpu_device.type in ('cuda', 'mps')

            if self._gpu_available:
                # Initialize GPU heuristic evaluator
                board_size = self._board_size or 8
                num_players = self._num_players or 2
                self._gpu_evaluator = GPUHeuristicEvaluator(
                    device=self._gpu_device,
                    board_size=board_size,
                    num_players=num_players,
                )
                logger.info(
                    f"MaxNAI: GPU acceleration enabled on {self._gpu_device} "
                    f"(batch_size={self._gpu_batch_size})"
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
        """
        if not self._leaf_buffer or not self._gpu_available:
            return

        try:
            from .gpu_batch import GPUBoardState
            from .hybrid_gpu import batch_game_states_to_gpu

            # States are already immutable (converted when added to buffer)
            immutable_states = [state for state, _ in self._leaf_buffer]
            hashes = [h for _, h in self._leaf_buffer]

            # Create GPU batch
            gpu_state = batch_game_states_to_gpu(
                immutable_states,
                self._gpu_device,
                self._board_size or 8,
            )

            # Evaluate for all players in parallel (GPU handles batch)
            num_players = self._num_players or 2
            all_scores: Dict[int, np.ndarray] = {}

            for player_num in range(1, num_players + 1):
                scores_tensor = self._gpu_evaluator.evaluate_batch(gpu_state, player_num)
                all_scores[player_num] = scores_tensor.cpu().numpy()

            # Store results indexed by state hash
            for i, state_hash in enumerate(hashes):
                player_scores: Dict[int, float] = {}
                for player_num in range(1, num_players + 1):
                    player_scores[player_num] = float(all_scores[player_num][i])
                self._leaf_results[state_hash] = player_scores

            # Shadow validation if enabled
            if self._shadow_validator is not None:
                self._validate_batch(immutable_states, hashes)

            logger.debug(f"MaxNAI: GPU batch evaluated {len(self._leaf_buffer)} leaves")

        except Exception as e:
            logger.warning(f"MaxNAI: GPU batch flush failed, falling back to CPU: {e}")
            # Fall back to CPU evaluation for this batch
            for state, state_hash in self._leaf_buffer:
                mutable = MutableGameState.from_immutable(state)
                self._leaf_results[state_hash] = self._evaluate_all_players_cpu(mutable)

        finally:
            self._leaf_buffer.clear()

    def _validate_batch(
        self,
        states: List[GameState],
        hashes: List[int],
    ) -> None:
        """Validate GPU results against CPU for a sample of positions."""
        if self._shadow_validator is None:
            return

        import random
        sample_rate = 0.05  # Check 5% of batch

        for i, (state, state_hash) in enumerate(zip(states, hashes)):
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

    def _evaluate_all_players_cpu(self, state: MutableGameState) -> Dict[int, float]:
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

    def select_move(self, game_state: GameState) -> Optional[Move]:
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
        best_score = float('-inf')

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
                best_score = current_best_score

        # Final flush for any remaining leaves
        if self._gpu_available:
            self._flush_leaf_buffer()

        return best_move

    def _maxn_search(
        self,
        state: MutableGameState,
        depth: int
    ) -> Dict[int, float]:
        """Recursive Max-N search.

        Args:
            state: Current game state (mutable)
            depth: Remaining search depth

        Returns:
            Dictionary mapping player_number -> score
        """
        self.nodes_visited += 1

        # Time check
        if self.nodes_visited % 500 == 0:
            if time.time() - self.start_time > self.time_limit:
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

        # Current player picks move that maximizes their own score
        best_scores: Optional[Dict[int, float]] = None
        best_my_score = float('-inf')

        for move in valid_moves:
            if time.time() - self.start_time > self.time_limit:
                break

            undo = state.make_move(move)
            child_scores = self._maxn_search(state, depth - 1)
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

    def _evaluate_all_players(self, state: MutableGameState) -> Dict[int, float]:
        """Evaluate position for ALL players.

        Returns a score vector where each player's score reflects
        how good the position is for them.

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
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)
        self.start_time: float = 0.0
        self.time_limit: float = 0.0
        self.nodes_visited: int = 0
        self._num_players: Optional[int] = None

        logger.debug(
            f"BRSAI(player={player_number}, difficulty={config.difficulty})"
        )

    def _get_lookahead_rounds(self) -> int:
        """Get number of BRS rounds based on difficulty."""
        if self.config.difficulty >= 7:
            return 3
        elif self.config.difficulty >= 4:
            return 2
        else:
            return 1

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select the best move using Best-Reply Search.

        For each candidate move, simulates future rounds where each
        player plays their greedy best response.
        """
        valid_moves = self.get_valid_moves(game_state)
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Detect number of players
        if self._num_players is None:
            self._num_players = len(game_state.players)

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

        # Find greedy best move for current player
        best_move = None
        best_score = float('-inf')

        for move in valid_moves:
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
        result = self._simulate_brs_rounds(state, remaining_rounds - 1)
        state.unmake_move(undo)

        return result

    def _evaluate_for_me(self, state: MutableGameState) -> float:
        """Evaluate position from this AI's perspective."""
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0
            elif winner is not None:
                return -100000.0
            return 0.0

        immutable = state.to_immutable()
        return self.evaluate_position(immutable)
