"""
Minimax AI implementation for RingRift
Uses minimax algorithm with alpha-beta pruning for move selection
"""

from typing import Optional
import time

from .bounded_transposition_table import BoundedTranspositionTable
from .heuristic_ai import HeuristicAI
from .zobrist import ZobristHash
from ..models import GameState, Move, AIConfig


class MinimaxAI(HeuristicAI):
    """AI that uses minimax with alpha-beta pruning"""

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
            max_depth = 1
            if self.config.difficulty >= 9:
                max_depth = 5
            elif self.config.difficulty >= 7:
                max_depth = 4
            elif self.config.difficulty >= 4:
                max_depth = 3
            else:
                max_depth = 2

            # Sort moves by heuristic score for better pruning
            # (using 1-ply lookahead)
            # Enhanced Move Ordering:
            # 1. Captures (MVV-LVA approximation: prioritize captures)
            # 2. Line formations
            # 3. Territory claims
            # 4. History Heuristic / Killer Moves
            # 5. Heuristic score

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
                    # This is a known bottleneck; in future we should
                    # switch to make_move/undo_move on a mutable board.
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

            selected = best_move

        self.move_count += 1
        return selected

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

        # Sort noisy moves by MVV-LVA or similar heuristic
        # For now, just sort by simple evaluation
        scored_moves = []
        for move in noisy_moves:
            # Quick evaluation: just use the move type priority
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