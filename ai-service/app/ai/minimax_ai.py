"""Minimax AI implementation for RingRift.

This agent uses depth‑limited minimax with alpha‑beta pruning and optional
quiescence search. It supports both immutable (legacy) and mutable
make/unmake search modes, gated by ``config.use_incremental_search``.

In all modes, ``config.think_time`` (when set) is interpreted strictly as
an upper bound on wall‑clock search time per move. No artificial delay is
introduced once search has completed.

Neural Network Integration (D4+):
When ``config.use_neural_net`` is True and ``config.difficulty >= 4``,
MinimaxAI uses NNUE (Efficiently Updatable Neural Network) evaluation
instead of the hand-crafted heuristic. This provides stronger positional
evaluation while maintaining fast CPU inference suitable for alpha-beta
search.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ..models import AIConfig, BoardType, GamePhase, GameState, Move, MoveType
from ..rules.legacy.move_type_aliases import convert_legacy_move_type
from ..rules.mutable_state import MutableGameState
from .bounded_transposition_table import BoundedTranspositionTable
from .game_state_utils import infer_num_players
from .heuristic_ai import HeuristicAI
from .zobrist import ZobristHash

# Lazy imports for neural network components to avoid loading torch when not needed
# These are only imported when use_neural_net=True (D4+ difficulty)
if TYPE_CHECKING:
    from .nnue import NNUEEvaluator
    from .nnue_policy import RingRiftNNUEWithPolicy

# Environment variable to enable zero-sum evaluation for minimax.
# When enabled, evaluation computes (my_eval - opponent_eval) / 2 which
# ensures Eval(P1) = -Eval(P2), required for correct alpha-beta behavior.
# This doubles evaluation cost but fixes the non-zero-sum evaluation bug.
MINIMAX_ZERO_SUM_EVAL = os.getenv(
    'RINGRIFT_MINIMAX_ZERO_SUM_EVAL', 'true'
).lower() in ('true', '1', 'yes')

logger = logging.getLogger(__name__)

# Interactive phases where player actions (and thus quiescence search) are valid.
# Non-interactive phases (LINE_PROCESSING, TERRITORY_PROCESSING, FORCED_ELIMINATION)
# are bookkeeping phases where get_valid_moves would return inappropriate move types.
_INTERACTIVE_PHASES = frozenset({
    GamePhase.RING_PLACEMENT,
    GamePhase.MOVEMENT,
    GamePhase.CAPTURE,
    GamePhase.CHAIN_CAPTURE,
})


class MinimaxAI(HeuristicAI):
    """AI that uses minimax with alpha‑beta pruning.

    When ``config.use_incremental_search`` is True (the default),
    :class:`MinimaxAI` uses the make/unmake pattern on
    :class:`MutableGameState` for faster search by avoiding object
    allocation overhead. When False, it falls back to the legacy
    immutable state cloning via :meth:`RulesEngine.apply_move`.

    Difficulty and depth:
        The :attr:`AIConfig.difficulty` field controls the maximum search
        depth via :meth:`_get_max_depth`:

        - difficulty >= 9 → depth 5
        - difficulty >= 7 → depth 4
        - difficulty >= 4 → depth 3
        - otherwise      → depth 2

        Higher difficulty therefore yields deeper search, subject to the
        global time budget enforced by ``config.think_time``.
    """

    def __init__(self, player_number: int, config: AIConfig) -> None:
        super().__init__(player_number, config)
        # Use bounded tables to prevent memory leaks
        self.transposition_table: BoundedTranspositionTable = (
            BoundedTranspositionTable(max_entries=100000)
        )
        # Killer moves table: [depth][move_index] -> Move
        # We store up to 2 killer moves per depth
        self.killer_moves: BoundedTranspositionTable = (
            BoundedTranspositionTable(max_entries=10000)
        )
        self.zobrist: ZobristHash = ZobristHash()
        # Wall‑clock search bookkeeping, populated at the start of each
        # select_move call and used by both legacy and incremental paths.
        self.start_time: float = 0.0
        self.time_limit: float = 0.0
        self.nodes_visited: int = 0
        # Configuration option for incremental search
        self.use_incremental_search: bool = getattr(
            config, 'use_incremental_search', True
        )

        # NNUE neural evaluation (D4+ when enabled)
        # Priority:
        # - Explicit AIConfig.use_neural_net when provided
        # - RINGRIFT_DISABLE_NEURAL_NET env var can globally disable NN usage
        # - Default: enabled for difficulty >= 4
        disable_nn_env = os.environ.get("RINGRIFT_DISABLE_NEURAL_NET", "").lower() in {
            "1", "true", "yes", "on",
        }
        use_nn_config = getattr(config, "use_neural_net", None)
        # NNUE is enabled at D4+ by default, can be explicitly disabled
        should_use_nnue = (
            config.difficulty >= 4 and
            (use_nn_config if use_nn_config is not None else True) and
            not disable_nn_env
        )

        self.nnue_evaluator: NNUEEvaluator | None = None
        self.use_nnue: bool = False

        if should_use_nnue:
            try:
                # Board type will be determined on first evaluation
                # For now, use a placeholder that gets set in select_move
                self._pending_nnue_init = True
                logger.debug(
                    f"MinimaxAI(player={player_number}, difficulty={config.difficulty}): "
                    "NNUE evaluation will be initialized on first move"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize NNUE: {e}")
                self._pending_nnue_init = False
        else:
            self._pending_nnue_init = False
            logger.debug(
                f"MinimaxAI(player={player_number}, difficulty={config.difficulty}): "
                "using heuristic evaluation"
            )

        # Policy model for move ordering (optional, D4+ when enabled)
        # The policy head predicts which moves are likely good, enabling
        # better move ordering for more effective alpha-beta pruning.
        self.policy_model: RingRiftNNUEWithPolicy | None = None
        self.use_policy_ordering: bool = False
        self._pending_policy_init: bool = False

        use_policy_config = getattr(config, "use_policy_ordering", None)
        should_use_policy = (
            config.difficulty >= 4 and
            (use_policy_config if use_policy_config is not None else True) and
            not disable_nn_env
        )

        if should_use_policy:
            self._pending_policy_init = True
            logger.debug(
                f"MinimaxAI(player={player_number}): "
                "Policy move ordering will be initialized on first move"
            )

    def _init_policy_model(self, board_type: BoardType, num_players: int) -> None:
        """Initialize policy model for move ordering."""
        if not self._pending_policy_init:
            return

        self._pending_policy_init = False

        try:
            from .nnue_policy import RingRiftNNUEWithPolicy

            # Try to load policy model checkpoint
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "models", "nnue", f"nnue_policy_{board_type.value}_{num_players}p.pt"
            )
            model_path = os.path.normpath(model_path)

            if os.path.exists(model_path):
                # Use safe loading utility - tries weights_only=True first,
                # falls back to full loading for legacy checkpoints with metadata
                from app.utils.torch_utils import safe_load_checkpoint
                checkpoint = safe_load_checkpoint(model_path, map_location="cpu", warn_on_unsafe=False)
                state_dict = checkpoint
                hidden_dim = 256
                num_hidden_layers = 2

                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    hidden_dim = int(checkpoint.get("hidden_dim") or hidden_dim)
                    num_hidden_layers = int(checkpoint.get("num_hidden_layers") or num_hidden_layers)

                if isinstance(state_dict, dict):
                    try:
                        accumulator_weight = state_dict.get("accumulator.weight")
                        if accumulator_weight is not None and hasattr(accumulator_weight, "shape"):
                            hidden_dim = int(accumulator_weight.shape[0])
                    except (AttributeError, TypeError, ValueError, IndexError):
                        pass

                    try:
                        import re

                        layer_indices = set()
                        for key in state_dict:
                            match = re.match(r"hidden\.(\d+)\.weight$", key)
                            if match:
                                layer_indices.add(int(match.group(1)))
                        if layer_indices:
                            num_hidden_layers = len(layer_indices)
                    except (TypeError, AttributeError, ValueError):
                        pass

                self.policy_model = RingRiftNNUEWithPolicy(
                    board_type=board_type,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                if not isinstance(state_dict, dict):
                    raise TypeError(f"Unexpected policy checkpoint: {type(state_dict).__name__}")
                self.policy_model.load_state_dict(state_dict)
                self.policy_model.eval()
                self.use_policy_ordering = True
                logger.info(
                    f"MinimaxAI: Loaded policy model from {model_path}"
                )
            else:
                logger.debug(
                    f"MinimaxAI: No policy model found at {model_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to load policy model: {e}")

    def evaluate_position(self, game_state: GameState) -> float:
        """
        Zero-sum evaluation for minimax search.

        For correct alpha-beta behavior, evaluation must satisfy:
            Eval(state, P1) = -Eval(state, P2)

        The base HeuristicAI evaluation is NOT zero-sum because many features
        only consider "my" perspective (e.g., my_rings_in_hand, my_markers).
        This causes minimax to make suboptimal decisions at depth because
        opponent maximizing THEIR score != minimizing MY score.

        This override computes:
            zero_sum_eval = (my_eval - opponent_eval) / 2

        This doubles evaluation cost but ensures correct minimax behavior.
        Can be disabled via RINGRIFT_MINIMAX_ZERO_SUM_EVAL=false for testing.
        """
        # Check for game over first (these are always zero-sum)
        if game_state.game_status == "completed":
            if game_state.winner == self.player_number:
                return 100000.0
            elif game_state.winner is not None:
                return -100000.0
            else:
                return 0.0

        if not MINIMAX_ZERO_SUM_EVAL:
            # Use base evaluation (for A/B testing)
            return super().evaluate_position(game_state)

        # Compute raw eval from my perspective
        my_eval = super().evaluate_position(game_state)

        # Compute raw eval from opponent's perspective
        # For 2-player games, simply switch player number
        # For 3-4 player paranoid search, we take max of all opponents
        original_player = self.player_number

        # Determine all opponent player numbers
        opponent_players = [
            p.player_number
            for p in game_state.players
            if p.player_number != self.player_number
        ]

        if not opponent_players:
            return my_eval

        # In paranoid search, all opponents form a coalition against us.
        # We use the MAX of opponent evaluations (worst case for us).
        max_opponent_eval = float('-inf')
        for opp_num in opponent_players:
            self.player_number = opp_num
            opp_eval = super().evaluate_position(game_state)
            max_opponent_eval = max(max_opponent_eval, opp_eval)

        # Restore original player
        self.player_number = original_player

        # Zero-sum: my advantage = my eval - opponent's best eval
        # Divided by 2 to keep values in similar range
        return (my_eval - max_opponent_eval) / 2.0

    def select_move(self, game_state: GameState) -> Move | None:
        """Select the best move using minimax search.

        This method runs iterative‑deepening minimax up to a difficulty‑
        dependent depth (see :meth:`_get_max_depth`), subject to a global
        wall‑clock budget derived from ``config.think_time``. When
        ``think_time`` is not set or non‑positive, a difficulty‑scaled
        default budget is used instead. No additional sleeps or UX‑style
        delays are added on top of the search time.

        Args:
            game_state: Current game state.

        Returns:
            The selected :class:`Move`, or ``None`` if there are no legal
            moves for this player.
        """
        # Get all valid moves via the canonical rules engine
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return None

        swap_move = self.maybe_select_swap_move(game_state, valid_moves)
        if swap_move is not None:
            self.move_count += 1
            return swap_move

        # If we decide not to swap, remove the meta-move so search does not
        # mis-evaluate it under the old seat identity.
        valid_moves = [
            m for m in valid_moves if m.type != MoveType.SWAP_SIDES
        ]

        # Multi-player semantics:
        # For 3p/4p, MinimaxAI uses a "Paranoid" reduction:
        # - This AI is the sole maximizer.
        # - All other players are treated as a minimizing coalition.
        num_players = infer_num_players(game_state)

        # Lazy NNUE initialization on first move (needs board type from game state)
        if getattr(self, '_pending_nnue_init', False):
            self._pending_nnue_init = False
            try:
                from .nnue import NNUEEvaluator  # Lazy import
                board_type = game_state.board.type
                self.nnue_evaluator = NNUEEvaluator(
                    board_type=board_type,
                    player_number=self.player_number,
                    num_players=num_players,
                    model_id=getattr(self.config, 'nn_model_id', None),
                    allow_fresh=bool(
                        getattr(self.config, "allow_fresh_weights", False)
                    ),
                )
                self.use_nnue = self.nnue_evaluator.available
                if self.use_nnue:
                    logger.info(
                        f"MinimaxAI(player={self.player_number}): NNUE evaluation enabled "
                        f"for {board_type.value}"
                    )
                else:
                    logger.debug(
                        f"MinimaxAI(player={self.player_number}): NNUE model not available, "
                        "falling back to heuristic"
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize NNUE evaluator: {e}")
                self.use_nnue = False

        # Lazy policy model initialization on first move
        if getattr(self, '_pending_policy_init', False):
            board_type = game_state.board.type
            self._init_policy_model(board_type, num_players)

        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = self.get_random_element(valid_moves)
        else:
            # Route to incremental or legacy search based on config
            if self.use_incremental_search:
                selected = self._select_move_incremental(
                    game_state, valid_moves
                )
            else:
                selected = self._select_move_legacy(game_state, valid_moves)

        self.move_count += 1
        return selected

    def _select_move_legacy(
        self, game_state: GameState, valid_moves: list[Move]
    ) -> Move:
        """Legacy search using immutable state cloning via apply_move().

        This is the original implementation preserved for backward
        compatibility and A/B testing against the new incremental search.
        """
        # Iterative Deepening
        self.start_time = time.time()
        # Use think_time (ms) as an explicit wall-clock budget when
        # provided, falling back to a difficulty-scaled default.
        if (
            self.config.think_time is not None
            and self.config.think_time > 0
        ):
            self.time_limit = self.config.think_time / 1000.0
        else:
            # 0.5s to 2.5s based on difficulty.
            self.time_limit = 0.5 + (self.config.difficulty * 0.2)
        self.nodes_visited = 0

        best_move = valid_moves[0]
        max_depth = self._get_max_depth()

        # Sort moves by heuristic score for better pruning
        scored_moves = self._score_and_sort_moves(game_state, valid_moves)

        # Clear transposition table for new search
        self.transposition_table.clear()
        self.killer_moves.clear()

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

                # Use apply_move (which now returns a new state).
                next_state = self.rules_engine.apply_move(game_state, move)
                score = self._minimax(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    (next_state.current_player == self.player_number),
                )

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move

                alpha = max(alpha, score)
                if beta <= alpha:
                    break

            if current_best_move:
                best_move = current_best_move

        return best_move

    def _select_move_incremental(
        self, game_state: GameState, valid_moves: list[Move]
    ) -> Move:
        """Incremental search using make/unmake on MutableGameState.

        This provides 10-50x speedup by avoiding object allocation overhead
        during tree search. The MutableGameState is modified in-place and
        restored using MoveUndo tokens.
        """
        # Iterative Deepening
        self.start_time = time.time()
        if (
            self.config.think_time is not None
            and self.config.think_time > 0
        ):
            self.time_limit = self.config.think_time / 1000.0
        else:
            self.time_limit = 0.5 + (self.config.difficulty * 0.2)
        self.nodes_visited = 0

        best_move = valid_moves[0]
        max_depth = self._get_max_depth()

        # Sort moves by heuristic score for better pruning
        scored_moves = self._score_and_sort_moves(game_state, valid_moves)

        # Clear transposition table for new search
        self.transposition_table.clear()
        self.killer_moves.clear()

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
                score = self._alpha_beta_mutable(
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

            if current_best_move:
                best_move = current_best_move

        return best_move

    def _get_max_depth(self) -> int:
        """Get maximum search depth based on difficulty setting."""
        if self.config.difficulty >= 9:
            return 5
        elif self.config.difficulty >= 7:
            return 4
        elif self.config.difficulty >= 4:
            return 3
        else:
            return 2

    def _canonical_move_type(self, move: Move) -> str:
        """Normalize legacy aliases to canonical move type strings."""
        raw_type = move.type.value if hasattr(move.type, "value") else str(move.type)
        return convert_legacy_move_type(raw_type, warn=False)

    def _score_and_sort_moves(
        self, game_state: GameState, valid_moves: list[Move]
    ) -> list[tuple]:
        """Score and sort moves for better alpha-beta pruning.

        Uses 1-ply lookahead and move-type priority bonuses.
        """
        scored_moves = []
        for move in valid_moves:
            # Priority bonus for "noisy" moves
            priority_bonus = 0.0
            move_type = self._canonical_move_type(move)
            if move_type in ("choose_territory_option", "eliminate_rings_from_stack"):
                priority_bonus = 10000.0
            elif move_type in ("process_line", "choose_line_option"):
                priority_bonus = 5000.0
            elif move_type == "continue_capture_segment":
                priority_bonus = 2000.0
            elif move_type == "overtaking_capture":
                priority_bonus = 1000.0
            elif move_type == "recovery_slide":
                # RR-CANON-R110–R115: Recovery is tactical (frees buried rings)
                priority_bonus = 800.0

            next_state = self.rules_engine.apply_move(game_state, move)
            score = self.evaluate_position(next_state)
            scored_moves.append((score + priority_bonus, move))

        # Sort descending
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves

    def _get_state_hash(self, game_state: GameState) -> int:
        """Generate a unique hash for the game state using Zobrist hashing"""
        if game_state.zobrist_hash is not None:
            return game_state.zobrist_hash
        return self.zobrist.compute_initial_hash(game_state)

    def _minimax(
        self,
        game_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool
    ) -> float:
        """
        Minimax recursive function with Paranoid algorithm support
        """
        self.nodes_visited += 1
        # Check timeout every 100 nodes for faster response to time limits
        if self.nodes_visited % 100 == 0 and time.time() - self.start_time > self.time_limit:
            # Return heuristic to unwind safely, but result will be
            # discarded by outer loop check
            return self.evaluate_position(game_state)

        state_hash = self._get_state_hash(game_state)
        entry = self.transposition_table.get(state_hash)
        if entry is not None and entry['depth'] >= depth:
            if entry['flag'] == 'exact':
                return entry['score']
            elif entry['flag'] == 'lowerbound':
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == 'upperbound':
                beta = min(beta, entry['score'])
            if alpha >= beta:
                return entry['score']

        if depth == 0:
            # Use Quiescence Search at leaf nodes
            score = self._quiescence_search(
                game_state,
                alpha,
                beta,
                (game_state.current_player == self.player_number),
                depth=3
            )
            self.transposition_table.put(state_hash, {
                'score': score,
                'depth': depth,
                'flag': 'exact'
            })
            return score

        current_player_num = game_state.current_player

        # Check if game is over
        if game_state.game_status == "completed":
            # If I won, return huge score. If I lost, return huge negative.
            if game_state.winner == self.player_number:
                return 100000.0 + depth  # Prefer faster wins
            elif game_state.winner is not None:
                return -100000.0 - depth  # Prefer slower losses
            else:
                return 0.0  # Draw

        # Use the host-level RulesEngine surface so that when there are no
        # interactive moves but a bookkeeping move is required, it is still
        # surfaced as a legal action.
        valid_moves = self.rules_engine.get_valid_moves(
            game_state,
            current_player_num,
        )

        if not valid_moves:
            # If no moves, it's a terminal state
            # (loss for current player usually, or draw)
            # In RingRift, no moves usually means loss if it's your turn?
            # Or just pass? The engine handles pass logic, so if
            # get_valid_moves returns empty, it's likely game over.
            # Let's evaluate.
            return self.evaluate_position(game_state)

        # Determine if the CURRENT player in the simulation is ME or OPPONENT
        is_me = (current_player_num == self.player_number)

        # Move ordering with Killer Heuristic
        # 1. Killer moves
        # 2. Captures/Noisy moves
        # 3. History/Others

        ordered_moves = []
        killer_moves_at_depth = self.killer_moves.get(depth) or []

        # Separate killer moves
        killers = []
        others = []

        for move in valid_moves:
            is_killer = False
            for k in killer_moves_at_depth:
                # Simple equality check for moves
                if (move.type == k.type and
                    move.to.x == k.to.x and move.to.y == k.to.y and
                    ((move.from_pos is None and k.from_pos is None) or
                     (move.from_pos and k.from_pos and
                      move.from_pos.x == k.from_pos.x and
                      move.from_pos.y == k.from_pos.y))):
                    is_killer = True
                    break

            if is_killer:
                killers.append(move)
            else:
                others.append(move)

        # Sort others by priority (captures first)
        others.sort(
            key=lambda m: 1
            if self._canonical_move_type(m) in {
                "overtaking_capture",
                "continue_capture_segment",
                "process_line",
                "choose_line_option",
                "choose_territory_option",
                "eliminate_rings_from_stack",
            }
            else 0,
            reverse=True,
        )

        ordered_moves = killers + others

        if is_me:
            # Maximizing player (Me)
            max_eval = float('-inf')
            for i, move in enumerate(ordered_moves):
                next_state = self.rules_engine.apply_move(game_state, move)
                # Determine who is next
                next_is_me = (next_state.current_player == self.player_number)

                # Principal Variation Search (PVS)
                if i == 0:
                    # Search first move with full window
                    eval = self._minimax(
                        next_state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    # Search subsequent moves with null window
                    eval = self._minimax(
                        next_state, depth - 1, alpha, alpha + 0.01, next_is_me
                    )
                    if alpha < eval < beta:
                        # If it fails high, re-search with full window
                        eval = self._minimax(
                            next_state, depth - 1, eval, beta, next_is_me
                        )

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Beta cutoff - store killer move
                    killer_list = self.killer_moves.get(depth) or []
                    # Add if not already present
                    if move not in killer_list:
                        killer_list.insert(0, move)
                        # Keep only 2 recent killer moves
                        if len(killer_list) > 2:
                            killer_list.pop()
                        self.killer_moves.put(depth, killer_list)
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
            # Opponent turn (Minimizing my score)
            min_eval = float('inf')
            for i, move in enumerate(ordered_moves):
                next_state = self.rules_engine.apply_move(game_state, move)
                # Check who is next
                next_is_me = (next_state.current_player == self.player_number)

                # Principal Variation Search (PVS)
                if i == 0:
                    # Search first move with full window
                    eval = self._minimax(
                        next_state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    # Search subsequent moves with null window
                    eval = self._minimax(
                        next_state, depth - 1, beta - 0.01, beta, next_is_me
                    )
                    if alpha < eval < beta:
                        # If it fails low, re-search with full window
                        eval = self._minimax(
                            next_state, depth - 1, alpha, eval, next_is_me
                        )

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    # Alpha cutoff - store killer move
                    killer_list = self.killer_moves.get(depth) or []
                    if move not in killer_list:
                        killer_list.insert(0, move)
                        if len(killer_list) > 2:
                            killer_list.pop()
                        self.killer_moves.put(depth, killer_list)
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

    # =========================================================================
    # Mutable State Search Methods (Incremental Search)
    # =========================================================================

    def _alpha_beta_mutable(
        self,
        state: MutableGameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool
    ) -> float:
        """
        Alpha-beta search using make/unmake pattern on MutableGameState.

        This is the core of the incremental search implementation,
        providing significant speedup by avoiding object allocation
        during tree traversal.
        """
        self.nodes_visited += 1
        # Check timeout every 100 nodes for faster response to time limits
        if self.nodes_visited % 100 == 0 and time.time() - self.start_time > self.time_limit:
            return self._evaluate_mutable(state)

        state_hash = state.zobrist_hash
        entry = self.transposition_table.get(state_hash)
        if entry is not None and entry['depth'] >= depth:
            if entry['flag'] == 'exact':
                return entry['score']
            elif entry['flag'] == 'lowerbound':
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == 'upperbound':
                beta = min(beta, entry['score'])
            if alpha >= beta:
                return entry['score']

        if depth == 0:
            score = self._quiescence_search_mutable(
                state,
                alpha,
                beta,
                (state.current_player == self.player_number),
                depth=3,
            )
            self.transposition_table.put(state_hash, {
                'score': score,
                'depth': depth,
                'flag': 'exact'
            })
            return score

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

        # Get valid moves via conversion to immutable (move generation)
        immutable = state.to_immutable()
        valid_moves = self.rules_engine.get_valid_moves(
            immutable,
            current_player_num,
        )

        if not valid_moves:
            return self._evaluate_mutable(state)

        is_me = (current_player_num == self.player_number)

        # Move ordering with Killer Heuristic (and policy if available)
        ordered_moves = self._order_moves_with_killers(valid_moves, depth, state)

        if is_me:
            max_eval = float('-inf')
            for i, move in enumerate(ordered_moves):
                undo = state.make_move(move)
                next_is_me = (state.current_player == self.player_number)

                # Principal Variation Search (PVS)
                if i == 0:
                    eval_score = self._alpha_beta_mutable(
                        state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    eval_score = self._alpha_beta_mutable(
                        state, depth - 1, alpha, alpha + 0.01, next_is_me
                    )
                    if alpha < eval_score < beta:
                        eval_score = self._alpha_beta_mutable(
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
                    eval_score = self._alpha_beta_mutable(
                        state, depth - 1, alpha, beta, next_is_me
                    )
                else:
                    eval_score = self._alpha_beta_mutable(
                        state, depth - 1, beta - 0.01, beta, next_is_me
                    )
                    if alpha < eval_score < beta:
                        eval_score = self._alpha_beta_mutable(
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

    def _quiescence_search_mutable(
        self,
        state: MutableGameState,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        depth: int = 3
    ) -> float:
        """
        Quiescence search using make/unmake pattern on MutableGameState.

        Explores noisy moves (captures, line formations) to mitigate the
        horizon effect without the overhead of immutable state cloning.
        """
        # Ensure max/min tracking matches the actual side-to-move; this matters
        # for phases where a player may take multiple moves in a row.
        maximizing_player = (state.current_player == self.player_number)
        stand_pat = self._evaluate_mutable(state)

        if (
            self.time_limit > 0
            and (time.time() - self.start_time) > self.time_limit
        ):
            return stand_pat

        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat

        if depth <= 0:
            return stand_pat

        # Non-interactive phases (line_processing, territory_processing, etc.)
        # have no valid player moves - return static evaluation.
        if state.current_phase not in _INTERACTIVE_PHASES:
            return stand_pat

        # Get noisy moves
        current_player_num = state.current_player
        immutable = state.to_immutable()
        all_moves = self.rules_engine.get_valid_moves(
            immutable,
            current_player_num,
        )
        noisy_moves = [
            m for m in all_moves
            if self._canonical_move_type(m) in {
                "overtaking_capture",
                "continue_capture_segment",
                "process_line",
                "choose_line_option",
                "choose_territory_option",
                "eliminate_rings_from_stack",
            }
        ]

        if not noisy_moves:
            return stand_pat

        scored_moves = self._score_noisy_moves(noisy_moves)
        is_me = (current_player_num == self.player_number)

        if is_me:
            for _, move in scored_moves:
                undo = state.make_move(move)
                next_is_me = (state.current_player == self.player_number)
                score = self._quiescence_search_mutable(
                    state,
                    alpha,
                    beta,
                    next_is_me,
                    depth - 1,
                )
                state.unmake_move(undo)

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha
        else:
            for _, move in scored_moves:
                undo = state.make_move(move)
                next_is_me = (state.current_player == self.player_number)
                score = self._quiescence_search_mutable(
                    state, alpha, beta, next_is_me, depth - 1
                )
                state.unmake_move(undo)

                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
            return beta

    def _evaluate_mutable(self, state: MutableGameState) -> float:
        """
        Evaluate MutableGameState using NNUE (when available) or heuristic.

        When NNUE is enabled (D4+ with use_neural_net=True), uses neural
        network evaluation for stronger positional assessment. Falls back
        to heuristic evaluation when NNUE is unavailable.

        Uses zero-sum evaluation when MINIMAX_ZERO_SUM_EVAL is enabled.

        Note: For heuristic fallback, this converts to immutable state which
        adds some overhead, but is still faster than cloning state at every
        tree node.
        """
        # Check for game over first using mutable state methods
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.player_number:
                return 100000.0
            elif winner is not None:
                return -100000.0
            else:
                return 0.0

        # Use NNUE evaluation when available
        if self.use_nnue and self.nnue_evaluator is not None:
            try:
                # NNUE evaluator now applies zero-sum transformation internally
                # when RINGRIFT_NNUE_ZERO_SUM_EVAL=true (default)
                return self.nnue_evaluator.evaluate_mutable(state)
            except Exception as e:
                # Fallback to heuristic on NNUE error
                logger.warning(f"NNUE evaluation failed, falling back to heuristic: {e}")

        # Convert to immutable for heuristic evaluation
        # evaluate_position already applies zero-sum transformation
        immutable = state.to_immutable()
        return self.evaluate_position(immutable)

    def _order_moves_with_killers(
        self,
        valid_moves: list[Move],
        depth: int,
        state: MutableGameState | None = None,
    ) -> list[Move]:
        """Order moves with killer heuristic and optional policy scoring.

        Move ordering priority:
        1. Killer moves (from previous search)
        2. Policy-ranked moves (if policy model available)
        3. Capture moves
        4. Other moves
        """
        killer_moves_at_depth = self.killer_moves.get(depth) or []

        killers = []
        others = []

        for move in valid_moves:
            is_killer = False
            for k in killer_moves_at_depth:
                if self._moves_equal(move, k):
                    is_killer = True
                    break

            if is_killer:
                killers.append(move)
            else:
                others.append(move)

        # If policy model available and we have state, use policy for ordering
        if (
            depth >= 2
            and self.use_policy_ordering
            and self.policy_model is not None
            and state is not None
            and others
        ):
            try:
                others = self._order_by_policy(others, state)
            except Exception as e:
                logger.debug(f"Policy ordering failed, using heuristic: {e}")
                # Fall back to capture-first ordering
                others.sort(
                    key=lambda m: 1
                    if self._canonical_move_type(m) in {
                        "overtaking_capture",
                        "continue_capture_segment",
                        "process_line",
                        "choose_line_option",
                        "choose_territory_option",
                        "eliminate_rings_from_stack",
                    }
                    else 0,
                    reverse=True,
                )
        else:
            # Sort others by priority (captures first)
            others.sort(
                key=lambda m: 1
                if self._canonical_move_type(m) in {
                    "overtaking_capture",
                    "continue_capture_segment",
                    "process_line",
                    "choose_line_option",
                    "choose_territory_option",
                    "eliminate_rings_from_stack",
                }
                else 0,
                reverse=True,
            )

        return killers + others

    def _order_by_policy(
        self,
        moves: list[Move],
        state: MutableGameState,
    ) -> list[Move]:
        """Order moves by policy network scores (best first)."""
        # Lazy imports for neural network components
        import torch

        from .nnue import extract_features_from_gamestate, get_board_size
        from .nnue_policy import pos_to_flat_index

        if not moves or self.policy_model is None:
            return moves

        # Extract features from state
        immutable = state.to_immutable()
        board_type = immutable.board.type
        board_size = get_board_size(board_type)
        current_player = immutable.current_player or self.player_number

        features = extract_features_from_gamestate(immutable, current_player)
        features_tensor = torch.from_numpy(features[None, ...]).float()

        # Get policy logits
        with torch.no_grad():
            _, from_logits, to_logits = self.policy_model(features_tensor, return_policy=True)
            from_logits = from_logits[0].numpy()  # Shape: (H*W,)
            to_logits = to_logits[0].numpy()  # Shape: (H*W,)

        # Score each move
        center = board_size // 2
        center_idx = center * board_size + center
        scored_moves = []

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

            score = from_logits[from_idx] + to_logits[to_idx]
            scored_moves.append((score, move))

        # Sort by score descending (best moves first)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_moves]

    def _moves_equal(self, move1: Move, move2: Move) -> bool:
        """Check if two moves are equal for killer move matching."""
        if move1.type != move2.type:
            return False
        # Handle moves where `to` may be None (e.g., NO_ACTION, start_ring)
        if move1.to is None and move2.to is None:
            pass  # Both None, continue to check from_pos
        elif move1.to is None or move2.to is None:
            return False  # One is None, the other is not
        elif move1.to.x != move2.to.x or move1.to.y != move2.to.y:
            return False
        if move1.from_pos is None and move2.from_pos is None:
            return True
        if move1.from_pos and move2.from_pos:
            return (move1.from_pos.x == move2.from_pos.x and
                    move1.from_pos.y == move2.from_pos.y)
        return False

    def _store_killer_move(self, move: Move, depth: int) -> None:
        """Store a killer move for the given depth."""
        killer_list = self.killer_moves.get(depth) or []
        if move not in killer_list:
            killer_list.insert(0, move)
            if len(killer_list) > 2:
                killer_list.pop()
            self.killer_moves.put(depth, killer_list)

    def _score_noisy_moves(self, noisy_moves: list[Move]) -> list[tuple]:
        """Score noisy moves by move type priority."""
        scored_moves = []
        for move in noisy_moves:
            priority = 0
            move_type = self._canonical_move_type(move)
            if move_type in ("choose_territory_option", "eliminate_rings_from_stack"):
                priority = 4
            elif move_type in ("process_line", "choose_line_option"):
                priority = 3
            elif move_type == "continue_capture_segment":
                priority = 2
            elif move_type == "overtaking_capture":
                priority = 1
            scored_moves.append((priority, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return scored_moves

    # =========================================================================
    # Legacy Search Methods (Immutable State)
    # =========================================================================

    def _quiescence_search(
        self,
        game_state: GameState,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        depth: int = 3
    ) -> float:
        """
        Quiescence search to mitigate horizon effect by exploring noisy moves.
        (Legacy version using immutable state cloning)
        """
        # Ensure max/min tracking matches the actual side-to-move; this matters
        # for phases where a player may take multiple moves in a row.
        maximizing_player = (game_state.current_player == self.player_number)
        # Stand pat score (static evaluation)
        stand_pat = self.evaluate_position(game_state)

        # Time-safety: if the global search budget is exhausted, return the
        # static evaluation immediately instead of exploring further.
        if (
            self.time_limit > 0
            and (time.time() - self.start_time) > self.time_limit
        ):
            return stand_pat

        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if beta > stand_pat:
                beta = stand_pat

        if depth <= 0:
            return stand_pat

        # Get noisy moves (captures, line formations)
        current_player_num = game_state.current_player

        # We need to filter for noisy moves.
        # RulesEngine.get_valid_moves returns all moves.
        # We can filter by type.
        all_moves = self.rules_engine.get_valid_moves(
            game_state,
            current_player_num,
        )
        noisy_moves = [
            m for m in all_moves
            if self._canonical_move_type(m) in {
                "overtaking_capture",
                "continue_capture_segment",
                "process_line",
                "choose_line_option",
                "choose_territory_option",
                "eliminate_rings_from_stack",
            }
        ]

        if not noisy_moves:
            return stand_pat

        scored_moves = self._score_noisy_moves(noisy_moves)
        is_me = (current_player_num == self.player_number)

        if is_me:
            for _, move in scored_moves:
                next_state = self.rules_engine.apply_move(game_state, move)
                next_is_me = (next_state.current_player == self.player_number)
                score = self._quiescence_search(
                    next_state,
                    alpha,
                    beta,
                    next_is_me,
                    depth - 1,
                )

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha
        else:
            for _, move in scored_moves:
                next_state = self.rules_engine.apply_move(game_state, move)
                # Check who is next (similar to minimax)
                next_is_me = (next_state.current_player == self.player_number)

                score = self._quiescence_search(
                    next_state,
                    alpha,
                    beta,
                    next_is_me,
                    depth - 1,
                )

                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
            return beta
