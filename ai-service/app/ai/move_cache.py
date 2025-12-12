"""
Move generation caching for faster repeated lookups.

Caches valid moves based on board state hash to avoid redundant computation.
Uses LRU eviction to bound memory usage.
"""

from __future__ import annotations

import json
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
    - must_move_from_stack_key (movement constraints)
    - rulesOptions (e.g. swapRuleEnabled)
    - move_history length (meta-move eligibility, e.g. swap_sides)
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

        CRITICAL: Legal move generation depends on small pieces of metadata
        that are NOT captured by board structure alone:

        - move_history length: meta-moves like swap_sides (pie rule) depend on
          it and do not change the board hash.
        - must_move_from_stack_key: constrains legal movement/capture options
          after certain placements; this is not encoded in board geometry.
        - rulesOptions: rule toggles like swapRuleEnabled can change the move
          surface even when the board is identical.
        """
        history_len = len(state.move_history) if state.move_history else 0
        must_move_key = state.must_move_from_stack_key or ""

        board = state.board
        board_type = board.type.value if hasattr(board.type, "value") else str(board.type)
        board_size = getattr(board, "size", 0)

        # Player meta affects legal moves (e.g. rings_in_hand determines whether
        # PLACE_RING moves are legal). Encode a small, deterministic digest so
        # cached move lists cannot bleed between positions with identical board
        # structure but different per-player counters.
        players_digest = ""
        players = getattr(state, "players", None) or []
        if players:
            players_meta = sorted(
                f"{p.player_number}:{p.rings_in_hand}:{p.eliminated_rings}:{p.territory_spaces}"
                for p in players
            )
            players_digest = hashlib.md5("|".join(players_meta).encode()).hexdigest()

        max_players = getattr(state, "max_players", None)
        if max_players is None:
            max_players = len(players) if players else 0

        phase_value = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )

        rules_options = state.rules_options or {}
        if rules_options:
            rules_str = json.dumps(
                rules_options,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            rules_digest = hashlib.md5(rules_str.encode()).hexdigest()
        else:
            rules_digest = ""

        # Use Zobrist hash if available (fast)
        if state.zobrist_hash is not None:
            # Include board metadata so identical hashes on different boards
            # (e.g. square8 vs square19) never reuse move surfaces.
            return (
                f"{board_type}:{board_size}:{max_players}:{players_digest}:"
                f"{state.zobrist_hash}:{player}:{phase_value}:{history_len}:"
                f"{must_move_key}:{rules_digest}"
            )

        # Fallback: compute hash from board state
        # Build a deterministic string representation
        parts = [
            f"t:{board_type}",
            f"s:{board_size}",
            f"mp:{max_players}",
            f"pm:{players_digest}",
            f"p:{player}",
            f"ph:{phase_value}",
            f"hl:{history_len}",  # History length for swap_sides eligibility
            f"mm:{must_move_key}",
            f"ro:{rules_digest}",
        ]

        # Add stack positions (sorted for determinism)
        stack_keys = sorted(board.stacks.keys())
        for key in stack_keys:
            stack = board.stacks[key]
            rings_str = ",".join(str(r) for r in stack.rings)
            parts.append(
                f"st:{key}:{stack.controlling_player}:{stack.stack_height}:{rings_str}"
            )

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
