"""
Base AI Player class for RingRift
Abstract base class that all AI implementations inherit from
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time
import random

from ..models import GameState, Move, AIConfig
from ..rules.factory import get_rules_engine
from ..rules.interfaces import RulesEngine


class BaseAI(ABC):
    """Abstract base class for all AI implementations"""
    
    def __init__(self, player_number: int, config: AIConfig):
        """
        Initialize AI player
        
        Args:
            player_number: The player number this AI controls (1-based)
            config: AI configuration settings
        """
        self.player_number = player_number
        self.config = config
        self.move_count = 0
        self.rules_engine: RulesEngine = get_rules_engine()
        
    @abstractmethod
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move for the current game state
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move or None if no valid moves
        """
        pass
    
    @abstractmethod
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the current position from this AI's perspective
        
        Args:
            game_state: Current game state
            
        Returns:
            Evaluation score (positive = good for this AI, negative = bad)
        """
        pass
    
    def get_evaluation_breakdown(
        self, game_state: GameState
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of position evaluation
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with evaluation components
        """
        return {
            "total": self.evaluate_position(game_state)
        }
    
    def get_valid_moves(self, game_state: GameState) -> List[Move]:
        """
        Get all valid moves for the current position using the rules engine.
        
        Args:
            game_state: Current game state
            
        Returns:
            List of valid Move instances
        """
        return self.rules_engine.get_valid_moves(
            game_state,
            self.player_number,
        )
    
    def simulate_thinking(self, min_ms: int = 100, max_ms: int = 2000) -> None:
        """
        Simulate thinking time for more natural AI behavior
        
        Args:
            min_ms: Minimum thinking time in milliseconds
            max_ms: Maximum thinking time in milliseconds
        """
        if self.config.think_time is not None:
            # Use configured think time
            if self.config.think_time > 0:
                time.sleep(self.config.think_time / 1000.0)
        else:
            # Random think time for more natural feel
            think_time = random.randint(min_ms, max_ms)
            time.sleep(think_time / 1000.0)
    
    def should_pick_random_move(self) -> bool:
        """
        Determine if AI should pick a random move based on randomness setting
        
        Returns:
            True if should pick random move
        """
        if self.config.randomness is None or self.config.randomness == 0:
            return False
        return random.random() < self.config.randomness
    
    def get_random_element(self, items: List[Any]) -> Optional[Any]:
        """
        Get random element from list
        
        Args:
            items: List of items
            
        Returns:
            Random item or None if list is empty
        """
        if not items:
            return None
        return random.choice(items)
    
    def shuffle_array(self, items: List[Any]) -> List[Any]:
        """
        Shuffle array in place and return it
        
        Args:
            items: List to shuffle
            
        Returns:
            Shuffled list
        """
        random.shuffle(items)
        return items
    
    def get_opponent_numbers(self, game_state: GameState) -> List[int]:
        """
        Get list of opponent player numbers
        
        Args:
            game_state: Current game state
            
        Returns:
            List of opponent player numbers
        """
        return [
            p.player_number 
            for p in game_state.players 
            if p.player_number != self.player_number
        ]
    
    def get_player_info(
        self,
        game_state: GameState,
        player_number: Optional[int] = None,
    ):
        """
        Get player info for specified player (defaults to this AI's player)
        
        Args:
            game_state: Current game state
            player_number: Player number to get info for (None = this AI)
            
        Returns:
            Player info or None if not found
        """
        target_player = (
            player_number if player_number is not None else self.player_number
        )
        for player in game_state.players:
            if player.player_number == target_player:
                return player
        return None
    
    def __repr__(self) -> str:
        """String representation of AI"""
        return (
            f"{self.__class__.__name__}"
            f"(player={self.player_number}, "
            f"difficulty={self.config.difficulty})"
        )
