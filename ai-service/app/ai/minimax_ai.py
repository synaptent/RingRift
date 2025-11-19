"""
Minimax AI implementation for RingRift
Uses minimax algorithm with alpha-beta pruning for move selection
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid
import copy

from .base import BaseAI
from .heuristic_ai import HeuristicAI
from ..models import GameState, Move, AIConfig, Position

class MinimaxAI(HeuristicAI):
    """AI that uses minimax with alpha-beta pruning"""
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using minimax
        
        Args:
            game_state: Current game state
            
        Returns:
            Best move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=500, max_ms=2000)
        
        # Get all valid moves
        from ..game_engine import GameEngine
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        
        if not valid_moves:
            return None
            
        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
        else:
            # Determine search depth based on difficulty
            depth = 1
            if self.config.difficulty >= 9:
                depth = 4
            elif self.config.difficulty >= 7:
                depth = 3
            elif self.config.difficulty >= 4:
                depth = 2
                
            # Run minimax
            best_score = float('-inf')
            best_move = None
            
            # Sort moves by heuristic score for better pruning (using 1-ply lookahead)
            scored_moves = []
            for move in valid_moves:
                next_state = GameEngine.apply_move(game_state, move)
                score = self.evaluate_position(next_state)
                scored_moves.append((score, move))
            
            # Sort descending
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            
            alpha = float('-inf')
            beta = float('inf')
            
            for _, move in scored_moves:
                next_state = GameEngine.apply_move(game_state, move)
                score = self._minimax(next_state, depth - 1, alpha, beta, False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            
            selected = best_move if best_move else valid_moves[0]
        
        self.move_count += 1
        return selected

    def _minimax(self, game_state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        """
        Minimax recursive function
        
        Args:
            game_state: Current game state
            depth: Remaining depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: True if maximizing player's turn
            
        Returns:
            Heuristic score of the position
        """
        if depth == 0: # or game_over(game_state)
            return self.evaluate_position(game_state)
            
        from ..game_engine import GameEngine
        
        # Determine current player for this node
        # Note: This assumes 2-player game for simplicity.
        # For >2 players, minimax needs modification (e.g. Paranoid algorithm or MaxN)
        # Here we assume maximizing_player is self, minimizing is opponent
        current_player_num = self.player_number if maximizing_player else self.get_opponent_numbers(game_state)[0]
        
        valid_moves = GameEngine.get_valid_moves(game_state, current_player_num)
        
        if not valid_moves:
            return self.evaluate_position(game_state)

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                next_state = GameEngine.apply_move(game_state, move)
                eval = self._minimax(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_state = GameEngine.apply_move(game_state, move)
                eval = self._minimax(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval