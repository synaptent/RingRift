"""
Descent AI implementation for RingRift
Based on "A Simple AlphaZero" (arXiv:2008.01188v4)
"""

from typing import Optional, List, Dict, Any, Tuple, Set
import time
import copy
import random
from datetime import datetime

from .base import BaseAI
from .neural_net import NeuralNetAI
from ..models import GameState, Move, AIConfig

class DescentAI(BaseAI):
    """
    AI that uses Descent Tree Search algorithm.
    Descent is a modification of Unbounded Best-First Minimax (UBFM).
    It iteratively extends the best sequence of actions to the terminal states.
    """
    
    def __init__(self, player_number: int, config: AIConfig):
        super().__init__(player_number, config)
        # Try to load neural net for evaluation
        try:
            self.neural_net = NeuralNetAI(player_number, config)
        except Exception:
            self.neural_net = None
            
        # Transposition table to store values
        # Key: state_hash, Value: (value, children_values)
        self.transposition_table = {}
        
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using Descent search
        """
        # No simulated thinking for Descent, we use the time for search
        
        from ..game_engine import GameEngine
        
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        if not valid_moves:
            return None
            
        if self.should_pick_random_move():
            return random.choice(valid_moves)
            
        # Descent parameters
        if self.config.think_time is not None and self.config.think_time > 0:
            time_limit = self.config.think_time / 1000.0
        else:
            # Default time limit based on difficulty
            # Difficulty 1: 0.1s, Difficulty 10: 2.0s
            time_limit = 0.1 + (self.config.difficulty * 0.2)
            
        end_time = time.time() + time_limit
        
        # Run Descent iterations
        iterations = 0
        while time.time() < end_time:
            root_val = self._descent_iteration(game_state)
            iterations += 1
            
            # Completion: Stop if root is solved
            if root_val == 1.0: # Proven win
                break
            if root_val == -1.0: # Proven loss (best we can do)
                break
            
        # Select best move from root
        state_key = self._get_state_key(game_state)
        if state_key in self.transposition_table:
            _, children_values = self.transposition_table[state_key]
            if children_values:
                if game_state.current_player == self.player_number:
                    best_move = max(children_values.items(), key=lambda x: x[1])[0]
                else:
                    best_move = min(children_values.items(), key=lambda x: x[1])[0]
                return best_move
        
        # Fallback if something went wrong or no search happened
        return random.choice(valid_moves)

    def _descent_iteration(self, state: GameState, depth: int = 0) -> float:
        """
        Perform one iteration of Descent search
        Recursively selects best child until terminal, then backpropagates
        """
        # Check if terminal
        if state.game_status == "finished":
            return self._calculate_terminal_value(state, depth)
                
        # Check if state is in transposition table
        # We need a hashable representation of state
        # For now, we'll use a simplified string key based on board state
        # Ideally, we should use Zobrist hashing or similar
        state_key = self._get_state_key(state)
        
        if state_key not in self.transposition_table:
            # Expand node
            from ..game_engine import GameEngine
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)
            
            if not valid_moves:
                # Should be terminal, but just in case
                return 0.0
                
            # Evaluate children
            children_values = {}
            best_val = float('-inf') if state.current_player == self.player_number else float('inf')
            
            for move in valid_moves:
                next_state = GameEngine.apply_move(state, move)
                
                # Evaluate leaf
                if next_state.game_status == "finished":
                    if next_state.winner == self.player_number:
                        val = 1.0
                    elif next_state.winner is not None:
                        val = -1.0
                    else:
                        val = 0.0
                else:
                    val = self.evaluate_position(next_state)
                    
                children_values[move] = val
                
                if state.current_player == self.player_number:
                    best_val = max(best_val, val)
                else:
                    best_val = min(best_val, val)
            
            self.transposition_table[state_key] = (best_val, children_values)
            return best_val
            
        else:
            # Node already expanded, select best child to descend
            current_val, children_values = self.transposition_table[state_key]
            
            # Select best move
            best_move = None
            if state.current_player == self.player_number:
                best_move = max(children_values.items(), key=lambda x: x[1])[0]
            else:
                best_move = min(children_values.items(), key=lambda x: x[1])[0]
                
            # Descend
            from ..game_engine import GameEngine
            next_state = GameEngine.apply_move(state, best_move)
            val = self._descent_iteration(next_state, depth + 1)
            
            # Update value
            children_values[best_move] = val
            
            if state.current_player == self.player_number:
                new_best_val = max(children_values.values())
            else:
                new_best_val = min(children_values.values())
                
            self.transposition_table[state_key] = (new_best_val, children_values)
            return new_best_val

    def _get_state_key(self, state: GameState) -> str:
        """Generate a unique key for the game state"""
        # Simplified key: board representation + current player
        # This is slow but functional for prototype
        board_str = str(state.board.stacks) + str(state.board.markers) + str(state.board.collapsed_spaces)
        return f"{board_str}_{state.current_player}"
        
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
        # Territory
        territory_count = 0
        for p_id in state.board.collapsed_spaces.values():
            if p_id == self.player_number:
                territory_count += 1
        
        # Eliminated
        eliminated_count = state.board.eliminated_rings.get(str(self.player_number), 0)
        
        # Markers
        marker_count = 0
        for m in state.board.markers.values():
            if m.player == self.player_number:
                marker_count += 1
                
        # Normalize bonuses to be small (max ~0.05 total)
        # This ensures they act as tie-breakers and don't override win/loss
        bonus = (territory_count * 0.001) + (eliminated_count * 0.001) + (marker_count * 0.0001)
        
        if base_val > 0:
            val = base_val + bonus
        else:
            # If losing, we still prefer having more territory/rings (tie-breakers)
            val = base_val + bonus
            
        # Discount for depth (fewest moves to win)
        # Win fast (val > 0): val * gamma^depth -> decreases with depth
        # Lose slow (val < 0): val * gamma^depth -> increases (closer to 0) with depth
        gamma = 0.99
        discounted_val = val * (gamma ** depth)
        
        # Ensure we don't exceed bounds or flip sign
        if base_val > 0:
            return max(0.001, min(1.0, discounted_val))
        elif base_val < 0:
            return max(-1.0, min(-0.001, discounted_val))
        return 0.0

    def evaluate_position(self, game_state: GameState) -> float:
        """Evaluate position using neural net or heuristic"""
        val = 0.0
        if self.neural_net:
            val, _ = self.neural_net.evaluate_position(game_state)
        else:
            # Heuristic fallback
            # Simple material difference
            my_elim = game_state.board.eliminated_rings.get(str(self.player_number), 0)
            
            opp_elim = 0
            for pid, count in game_state.board.eliminated_rings.items():
                if int(pid) != self.player_number:
                    opp_elim += count
            
            val = (my_elim - opp_elim) * 0.05
        
        # Clamp value to (-0.99, 0.99) to reserve 1.0/-1.0 for proven terminal states
        return max(-0.99, min(0.99, val))