"""Tree Reuse Configuration and Utilities for MCTS-based AI.

This module provides utilities for managing tree reuse between consecutive
moves in MCTS-based search algorithms.

Tree Reuse Support by AI Implementation:
-----------------------------------------
| AI Class         | Tree Reuse | How It Works                            |
|------------------|------------|------------------------------------------|
| MCTSAI           | Yes        | Saves selected child as next root       |
| ImprovedMCTSAI   | Yes        | Configurable via MCTSConfig.tree_reuse  |
| GumbelMCTSAI     | Limited*   | Sequential Halving doesn't maintain tree |
| GPUGumbelMCTS    | No         | Tree reset each search                  |
| HeuristicAI      | N/A        | No tree search                          |

*Gumbel MCTS uses Sequential Halving which progressively eliminates actions
 rather than building a persistent tree. This makes traditional tree reuse
 less applicable. Instead, it benefits from NN evaluation caching.

Performance Impact:
------------------
Tree reuse can provide 10-30% speedup by:
1. Avoiding re-expansion of subtree after our move
2. Preserving visit counts and value estimates
3. Using opponent's move to warm-start our search

However, tree reuse also has costs:
1. Memory for storing the tree between moves
2. Stale values if opponent's move is unexpected
3. Reduced exploration if we over-rely on cached values

Recommended Settings:
--------------------
- Tournament play: Enable tree reuse (faster, deterministic opponent)
- Self-play training: Consider disabling (more exploration, fresh statistics)
- Analysis mode: Enable tree reuse (deep analysis of specific lines)

Usage:
    from app.ai.tree_reuse import get_tree_reuse_config, TreeReuseConfig

    # Get default config for a mode
    config = get_tree_reuse_config("tournament")

    # Create custom config
    config = TreeReuseConfig(
        enable=True,
        max_reuse_depth=10,
        value_decay=0.95,  # Discount older values
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TreeReuseConfig:
    """Configuration for tree reuse behavior.

    Attributes:
        enable: Whether to enable tree reuse between moves.
        max_reuse_depth: Maximum depth of tree to preserve (0=unlimited).
        value_decay: Decay factor for values from previous search (1.0=no decay).
        clear_on_time_pressure: Clear tree if remaining time is low.
        min_visits_to_reuse: Minimum visits required to consider a node for reuse.
    """

    enable: bool = True
    max_reuse_depth: int = 0  # 0 = unlimited
    value_decay: float = 1.0  # 1.0 = no decay
    clear_on_time_pressure: bool = True
    min_visits_to_reuse: int = 5


# Preset configurations for different use cases
_PRESETS: dict[str, TreeReuseConfig] = {
    # Tournament mode: maximize speed with aggressive reuse
    "tournament": TreeReuseConfig(
        enable=True,
        max_reuse_depth=20,
        value_decay=0.98,
        clear_on_time_pressure=True,
        min_visits_to_reuse=3,
    ),
    # Self-play training: balance reuse with exploration
    "selfplay": TreeReuseConfig(
        enable=True,
        max_reuse_depth=10,
        value_decay=0.9,  # More decay for fresher values
        clear_on_time_pressure=False,
        min_visits_to_reuse=10,
    ),
    # Analysis mode: deep reuse for thorough analysis
    "analysis": TreeReuseConfig(
        enable=True,
        max_reuse_depth=0,  # Unlimited
        value_decay=1.0,
        clear_on_time_pressure=False,
        min_visits_to_reuse=1,
    ),
    # No reuse: fresh search each move
    "none": TreeReuseConfig(
        enable=False,
    ),
}


def get_tree_reuse_config(mode: str = "tournament") -> TreeReuseConfig:
    """Get a tree reuse configuration preset.

    Args:
        mode: One of "tournament", "selfplay", "analysis", "none"

    Returns:
        TreeReuseConfig for the specified mode
    """
    if mode not in _PRESETS:
        raise ValueError(f"Unknown tree reuse mode: {mode}. Choose from {list(_PRESETS.keys())}")
    return _PRESETS[mode]


def apply_value_decay(
    value: float,
    age: int,
    config: TreeReuseConfig,
) -> float:
    """Apply value decay based on age of the estimate.

    Args:
        value: The cached value estimate
        age: Number of moves since value was computed
        config: Tree reuse configuration

    Returns:
        Decayed value
    """
    if not config.enable or config.value_decay >= 1.0:
        return value
    return value * (config.value_decay ** age)


def should_clear_tree(
    remaining_time_ms: int | None,
    config: TreeReuseConfig,
    time_threshold_ms: int = 5000,
) -> bool:
    """Check if tree should be cleared due to time pressure.

    Args:
        remaining_time_ms: Remaining time in milliseconds (None if untimed)
        config: Tree reuse configuration
        time_threshold_ms: Time below which to clear tree

    Returns:
        True if tree should be cleared
    """
    if not config.clear_on_time_pressure:
        return False
    if remaining_time_ms is None:
        return False
    return remaining_time_ms < time_threshold_ms


class TranspositionCache:
    """Simple transposition cache for NN evaluations.

    Caches (zobrist_hash -> (value, policy)) for reuse across searches.
    Useful for Gumbel MCTS and other methods that don't maintain a tree
    but can benefit from caching NN evaluations.

    This is a lightweight alternative to full tree reuse that works
    with any search method.
    """

    def __init__(self, max_size: int = 100000):
        self.cache: dict[int, tuple[float, list[float]]] = {}
        self.max_size = max_size
        self._access_order: list[int] = []  # For LRU eviction

    def get(self, zobrist_hash: int) -> tuple[float, list[float]] | None:
        """Look up cached evaluation.

        Args:
            zobrist_hash: Position hash

        Returns:
            (value, policy) tuple or None if not cached
        """
        if zobrist_hash in self.cache:
            # Move to end for LRU
            if zobrist_hash in self._access_order:
                self._access_order.remove(zobrist_hash)
            self._access_order.append(zobrist_hash)
            return self.cache[zobrist_hash]
        return None

    def put(self, zobrist_hash: int, value: float, policy: list[float]) -> None:
        """Store evaluation in cache.

        Args:
            zobrist_hash: Position hash
            value: Position value
            policy: Policy distribution
        """
        if len(self.cache) >= self.max_size:
            # Evict oldest
            oldest = self._access_order.pop(0)
            self.cache.pop(oldest, None)

        self.cache[zobrist_hash] = (value, policy)
        self._access_order.append(zobrist_hash)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self.cache)


# Global cache for shared use (singleton pattern)
_GLOBAL_CACHE: TranspositionCache | None = None


def get_transposition_cache(max_size: int = 100000) -> TranspositionCache:
    """Get the global transposition cache.

    Args:
        max_size: Maximum cache size (only used on first call)

    Returns:
        Global TranspositionCache instance
    """
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = TranspositionCache(max_size)
    return _GLOBAL_CACHE


def reset_transposition_cache() -> None:
    """Reset the global transposition cache (for testing)."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is not None:
        _GLOBAL_CACHE.clear()
    _GLOBAL_CACHE = None


__all__ = [
    "TreeReuseConfig",
    "TranspositionCache",
    "apply_value_decay",
    "get_transposition_cache",
    "get_tree_reuse_config",
    "reset_transposition_cache",
    "should_clear_tree",
]
