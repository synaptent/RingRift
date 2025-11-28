"""
Minimax AI implementation for RingRift
Uses minimax algorithm with alpha-beta pruning for move selection
"""

from typing import Optional, List
import time

from .bounded_transposition_table import BoundedTranspositionTable
from .heuristic_ai import HeuristicAI
from .zobrist import ZobristHash
from ..models import GameState, Move, AIConfig
from ..rules.mutable_state import MutableGameState


class MinimaxAI(HeuristicAI):
    """AI that uses minimax with alpha-beta pruning.
    
    When `config.use_incremental_search` is True (the default), MinimaxAI
    uses the make/unmake pattern on MutableGameState for 10-50x faster
    search by avoiding object allocation overhead. When False, it falls
    back to the legacy immutable state cloning via apply_move().
    """

    def __init__(self, player_number: int, config: AIConfig):
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
        self.zobrist = ZobristHash()
        self.start_time = 0.0
        self.time_limit = 0.0
        self.nodes_visited = 0
        # Configuration option for incremental search
        self.use_incremental_search = getattr(
            config, 'use_incremental_search', True
        )

    def simulate_thinking(self, min_ms: int = 100, max_ms: int = 2000) -> None:
        """Override BaseAI.simulate_thinking.

        Search-based engines treat config.think_time as a wall-clock budget
        for the search itself, so MinimaxAI does not add any extra sleep on
        top of the search loop.
        """
        return

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using minimax
        
        Args:
            game_state: Current game state
            
        Returns:
            Best move or None if no valid moves
        """
        # For search-based AIs we treat config.think_time as a search budget
        # rather than adding an extra sleep via simulate_thinking.
        
        # Get all valid moves via the canonical rules engine
        valid_moves = self.get_valid_moves(game_state)
        
        if not valid_moves:
            return None
             
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
        self, game_state: GameState, valid_moves: List[Move]
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
                    next_state, depth - 1, alpha, beta, False
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
        self, game_state: GameState, valid_moves: List[Move]
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
                    mutable_state, depth - 1, alpha, beta, False
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

    def _score_and_sort_moves(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> List[tuple]:
        """Score and sort moves for better alpha-beta pruning.
        
        Uses 1-ply lookahead and move-type priority bonuses.
        """
        scored_moves = []
        for move in valid_moves:
            # Priority bonus for "noisy" moves
            priority_bonus = 0.0
            if move.type == "territory_claim":
                priority_bonus = 10000.0
            elif move.type == "line_formation":
                priority_bonus = 5000.0
            elif move.type == "chain_capture":
                priority_bonus = 2000.0
            elif move.type == "overtaking_capture":
                priority_bonus = 1000.0
            
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
        if self.nodes_visited % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                # Return heuristic to unwind safely, but result will be
                # discarded by outer loop check
                return self.evaluate_position(game_state)

        state_hash = self._get_state_hash(game_state)
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

        if depth == 0:
            # Use Quiescence Search at leaf nodes
            score = self._quiescence_search(
                game_state,
                alpha,
                beta,
                maximizing_player,
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
        if game_state.game_status == "finished":
            # If I won, return huge score. If I lost, return huge negative.
            if game_state.winner == self.player_number:
                return 100000.0 + depth  # Prefer faster wins
            elif game_state.winner is not None:
                return -100000.0 - depth  # Prefer slower losses
            else:
                return 0.0  # Draw

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
            key=lambda m: 1 if m.type in [
                "overtaking_capture",
                "chain_capture",
                "line_formation",
                "territory_claim"
            ] else 0,
            reverse=True
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
        if self.nodes_visited % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                return self._evaluate_mutable(state)

        state_hash = state.zobrist_hash
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

        if depth == 0:
            score = self._quiescence_search_mutable(
                state, alpha, beta, maximizing_player, depth=3
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

        # Move ordering with Killer Heuristic
        ordered_moves = self._order_moves_with_killers(valid_moves, depth)

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
                
        # Get noisy moves
        current_player_num = state.current_player
        immutable = state.to_immutable()
        all_moves = self.rules_engine.get_valid_moves(
            immutable,
            current_player_num,
        )
        noisy_moves = [
            m for m in all_moves
            if m.type in [
                "overtaking_capture",
                "chain_capture",
                "line_formation",
                "territory_claim"
            ]
        ]

        if not noisy_moves:
            return stand_pat

        scored_moves = self._score_noisy_moves(noisy_moves)
        is_me = (current_player_num == self.player_number)
        
        if is_me:
            for _, move in scored_moves:
                undo = state.make_move(move)
                score = self._quiescence_search_mutable(
                    state, alpha, beta, False, depth - 1
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
        Evaluate MutableGameState by converting to immutable and using
        the parent HeuristicAI's evaluate_position method.
        
        Note: This conversion adds some overhead, but is still faster than
        cloning state at every tree node. Future optimization could implement
        direct evaluation on MutableGameState.
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
        
        # Convert to immutable for evaluation
        immutable = state.to_immutable()
        return self.evaluate_position(immutable)

    def _order_moves_with_killers(
        self, valid_moves: List[Move], depth: int
    ) -> List[Move]:
        """Order moves with killer heuristic for better pruning."""
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

        # Sort others by priority (captures first)
        others.sort(
            key=lambda m: 1 if m.type in [
                "overtaking_capture",
                "chain_capture",
                "line_formation",
                "territory_claim"
            ] else 0,
            reverse=True
        )

        return killers + others

    def _moves_equal(self, move1: Move, move2: Move) -> bool:
        """Check if two moves are equal for killer move matching."""
        if move1.type != move2.type:
            return False
        if move1.to.x != move2.to.x or move1.to.y != move2.to.y:
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

    def _score_noisy_moves(self, noisy_moves: List[Move]) -> List[tuple]:
        """Score noisy moves by move type priority."""
        scored_moves = []
        for move in noisy_moves:
            priority = 0
            if move.type == "territory_claim":
                priority = 4
            elif move.type == "line_formation":
                priority = 3
            elif move.type == "chain_capture":
                priority = 2
            elif move.type == "overtaking_capture":
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
            if m.type in [
                "overtaking_capture",
                "chain_capture",
                "line_formation",
                "territory_claim"
            ]
        ]

        if not noisy_moves:
            return stand_pat

        scored_moves = self._score_noisy_moves(noisy_moves)
        is_me = (current_player_num == self.player_number)
        
        if is_me:
            for _, move in scored_moves:
                next_state = self.rules_engine.apply_move(game_state, move)
                score = self._quiescence_search(
                    next_state,
                    alpha,
                    beta,
                    False,
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