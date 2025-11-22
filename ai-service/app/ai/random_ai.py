"""
Random AI implementation for RingRift
Selects moves randomly from valid options
"""

from typing import Optional, Dict
import random

from .base import BaseAI
from ..models import GameState, Move


class RandomAI(BaseAI):
    """AI that selects random valid moves"""
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select a random valid move
        
        Args:
            game_state: Current game state
            
        Returns:
            Random valid move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=200, max_ms=800)
        
        # Get all valid moves using the canonical rules engine
        valid_moves = self.rules_engine.get_valid_moves(
            game_state, self.player_number
        )
        
        if not valid_moves:
            return None
        
        # Select random move
        selected = random.choice(valid_moves)
        
        self.move_count += 1
        return selected
    
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate position (random AI doesn't really evaluate, returns neutral)
        
        Args:
            game_state: Current game state
            
        Returns:
            0.0 (neutral evaluation)
        """
        # Random AI doesn't evaluate positions
        # Return small random value to simulate variance
        return random.uniform(-0.1, 0.1)
    
    def get_evaluation_breakdown(
        self, game_state: GameState
    ) -> Dict[str, float]:
        """
        Get evaluation breakdown for random AI
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with random evaluation
        """
        return {
            "total": 0.0,
            "random_variance": random.uniform(-0.1, 0.1)
        }
    
