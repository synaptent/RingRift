"""
Zobrist Hashing implementation for RingRift.
Provides O(1) state hashing for transposition tables.
"""

import random
from typing import Dict, Tuple


class ZobristHash:
    """
    Singleton class to manage Zobrist hash keys.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ZobristHash, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize random bitstrings for all board features"""
        random.seed(42)  # Fixed seed for reproducibility

        # Board features to hash:
        # - Stacks: (position_key, controlling_player, height, rings_tuple)
        # - Markers: (position_key, player)
        # - Collapsed spaces: (position_key)
        # - Current player
        # - Current phase

        self.stack_keys: Dict[str, int] = {}
        self.marker_keys: Dict[str, int] = {}
        self.collapsed_keys: Dict[str, int] = {}
        self.player_keys: Dict[int, int] = {}
        self.phase_keys: Dict[str, int] = {}

        # Pre-generate keys for common positions (lazy load others if needed,
        # but better to pre-gen for max speed)
        # Since the board space is large (especially hex), we might need
        # dynamic generation or just hash the position string + feature.

        # For simplicity and memory efficiency, we'll use a mix of pre-gen
        # and dynamic hashing. But true Zobrist requires fixed keys.
        # Let's use a large table of random numbers indexed by a hash of
        # the feature.

        self.table_size = 1000000
        self.table = [random.getrandbits(64) for _ in range(self.table_size)]

    def get_stack_hash(
        self,
        pos_key: str,
        player: int,
        height: int,
        rings: Tuple[int, ...]
    ) -> int:
        """Get hash for a stack"""
        # Combine features into a unique index
        feature_hash = hash((pos_key, "stack", player, height, rings))
        return self.table[feature_hash % self.table_size]
        
    def get_marker_hash(self, pos_key: str, player: int) -> int:
        """Get hash for a marker"""
        feature_hash = hash((pos_key, "marker", player))
        return self.table[feature_hash % self.table_size]
        
    def get_collapsed_hash(self, pos_key: str) -> int:
        """Get hash for a collapsed space"""
        feature_hash = hash((pos_key, "collapsed"))
        return self.table[feature_hash % self.table_size]
        
    def get_player_hash(self, player: int) -> int:
        """Get hash for current player"""
        feature_hash = hash(("player", player))
        return self.table[feature_hash % self.table_size]
        
    def get_phase_hash(self, phase: str) -> int:
        """Get hash for current phase"""
        feature_hash = hash(("phase", phase))
        return self.table[feature_hash % self.table_size]

    def compute_initial_hash(self, game_state) -> int:
        """Compute full hash from scratch (expensive, O(N))"""
        h = 0
        
        # Stacks
        for pos_key, stack in game_state.board.stacks.items():
            h ^= self.get_stack_hash(
                pos_key, 
                stack.controlling_player, 
                stack.stack_height, 
                tuple(stack.rings)
            )
            
        # Markers
        for pos_key, marker in game_state.board.markers.items():
            h ^= self.get_marker_hash(pos_key, marker.player)
            
        # Collapsed
        for pos_key in game_state.board.collapsed_spaces:
            h ^= self.get_collapsed_hash(pos_key)
            
        # Global state
        h ^= self.get_player_hash(game_state.current_player)
        h ^= self.get_phase_hash(game_state.current_phase)
        
        return h