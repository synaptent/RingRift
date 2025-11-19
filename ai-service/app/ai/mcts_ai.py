"""
MCTS AI implementation for RingRift
Uses Monte Carlo Tree Search for move selection
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid
import math
import time

from .base import BaseAI
from .heuristic_ai import HeuristicAI
from ..models import GameState, Move, AIConfig, Position

class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = [] # Should be populated with valid moves

    def uct_select_child(self):
        """Select child using UCT formula"""
        s = sorted(self.children, key=lambda c: c.wins/c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, move, game_state):
        """Add a new child node"""
        child = MCTSNode(game_state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        """Update node stats"""
        self.visits += 1
        self.wins += result

class MCTSAI(HeuristicAI):
    """AI that uses Monte Carlo Tree Search"""
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using MCTS
        
        Args:
            game_state: Current game state
            
        Returns:
            Best move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=500, max_ms=2000)
        
        from ..game_engine import GameEngine
        
        # Get all valid moves
        valid_moves = GameEngine.get_valid_moves(game_state, self.player_number)
        
        if not valid_moves:
            return None
            
        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
        else:
            # MCTS parameters
            # Time limit based on difficulty
            time_limit = 0.5 + (self.config.difficulty * 0.2) # 0.5s to 2.5s
            
            root = MCTSNode(game_state)
            root.untried_moves = valid_moves
            
            end_time = time.time() + time_limit
            
            # Simple MCTS implementation
            while time.time() < end_time:
                node = root
                state = copy.deepcopy(node.game_state)
                
                # Selection
                while not node.untried_moves and node.children:
                    node = node.uct_select_child()
                    state = GameEngine.apply_move(state, node.move)
                
                # Expansion
                if node.untried_moves:
                    m = random.choice(node.untried_moves)
                    state = GameEngine.apply_move(state, m)
                    node = node.add_child(m, state)
                    # Populate untried moves for the new node
                    # Note: In a full implementation we'd need to know whose turn it is in 'state'
                    # Assuming alternating turns for now or relying on GameEngine to handle it
                    # But GameEngine.get_valid_moves needs player_number.
                    # We need to track current player in simulation.
                    # For now, simplified: just evaluate the state immediately
                
                # Simulation (Rollout)
                # Instead of full random rollout which is expensive, we'll do a shallow rollout
                # or just evaluate the state directly using heuristic
                result = self.evaluate_position(state)
                
                # Backpropagation
                while node is not None:
                    node.update(result)
                    node = node.parent
            
            # Select best move based on visits
            if root.children:
                selected = sorted(root.children, key=lambda c: c.visits)[-1].move
            else:
                # Fallback if no simulations completed (shouldn't happen given time limit)
                selected = random.choice(valid_moves)
        
        self.move_count += 1
        return selected