"""
Move generation caching for faster repeated lookups.

Caches valid moves based on board state hash to avoid redundant computation.
Uses LRU eviction to bound memory usage.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    from ..models import GameState, Move

# Maximum cache size (number of entries)
MOVE_CACHE_SIZE = int(os.getenv('RINGRIFT_MOVE_CACHE_SIZE', '1000'))

# Enable/disable move caching
# Caches valid moves by board state hash to avoid redundant move generation.
# Low risk optimization - only caches move generation, not evaluation.
USE_MOVE_CACHE = os.getenv('RINGRIFT_USE_MOVE_CACHE', 'true').lower() == 'true'


class MoveCache:
    """
    LRU cache for valid moves keyed by board state hash.

    The cache key is computed from:
    - Board type and size
    - Stack positions and controlling players
    - Marker positions and owners
    - Collapsed spaces
    - Current player and phase
    """

    def __init__(self, max_size: int = MOVE_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, List] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, state: 'GameState', player: int) -> Optional[List['Move']]:
        """
        Get cached moves for a game state.

        Returns None if not cached.
        """
        if not USE_MOVE_CACHE:
            return None

        key = self._compute_key(state, player)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(self, state: 'GameState', player: int, moves: List['Move']):
        """Cache moves for a game state."""
        if not USE_MOVE_CACHE:
            return

        key = self._compute_key(state, player)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = moves

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'size': len(self._cache),
            'hit_rate': hit_rate,
        }

    def _compute_key(self, state: 'GameState', player: int) -> str:
        """Compute a hash key for the game state.

        CRITICAL: The cache key must include move_history length because
        meta-moves like swap_sides (pie rule) depend on history state.
        swap_sides is only legal once, and its eligibility is determined by
        move_history - not by board position. Without tracking history length,
        the cache can return stale moves that include swap_sides after it's
        already been used.
        """
        history_len = len(state.move_history) if state.move_history else 0

        # Use Zobrist hash if available (fast)
        if state.zobrist_hash is not None:
            return f"{state.zobrist_hash}_{player}_{state.current_phase.value}_{history_len}"

        # Fallback: compute hash from board state
        board = state.board

        # Build a deterministic string representation
        parts = [
            f"t:{board.type.value}",
            f"s:{board.size}",
            f"p:{player}",
            f"ph:{state.current_phase.value}",
            f"hl:{history_len}",  # History length for swap_sides eligibility
        ]

        # Add stack positions (sorted for determinism)
        stack_keys = sorted(board.stacks.keys())
        for key in stack_keys:
            stack = board.stacks[key]
            parts.append(f"st:{key}:{stack.controlling_player}:{len(stack.rings)}")

        # Add marker positions
        marker_keys = sorted(board.markers.keys())
        for key in marker_keys:
            marker = board.markers[key]
            parts.append(f"mk:{key}:{marker.player}")

        # Add collapsed spaces
        collapsed_keys = sorted(board.collapsed_spaces.keys())
        parts.append(f"cl:{','.join(collapsed_keys)}")

        # Compute hash
        key_str = "|".join(parts)
        return hashlib.md5(key_str.encode()).hexdigest()


# Global move cache instance
_global_cache: Optional[MoveCache] = None


def get_move_cache() -> MoveCache:
    """Get the global move cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MoveCache()
    return _global_cache


def clear_move_cache():
    """Clear the global move cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


def get_cached_moves(state: 'GameState', player: int) -> Optional[List['Move']]:
    """Get cached moves from global cache."""
    return get_move_cache().get(state, player)


def cache_moves(state: 'GameState', player: int, moves: List['Move']):
    """Store moves in global cache."""
    get_move_cache().put(state, player, moves)
