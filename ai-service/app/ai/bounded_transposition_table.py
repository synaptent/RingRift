"""Bounded transposition table with LRU eviction for memory-limited search."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


class BoundedTranspositionTable:
    """LRU-evicting transposition table with configurable memory limit."""

    def __init__(
        self, max_entries: int = 100_000, entry_size_estimate: int = 4000
    ) -> None:
        """Initialize the transposition table.

        Args:
            max_entries: Maximum number of entries before eviction
            entry_size_estimate: Approximate bytes per entry for memory calc
                Note: entries include children_values dict which can contain
                50+ (move, value) pairs. Realistic sizes are 2-10KB per entry.
        """
        self._table: OrderedDict[Hashable, Any] = OrderedDict()
        self.max_entries = max_entries
        self.entry_size_estimate = entry_size_estimate
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @classmethod
    def from_memory_limit(
        cls, memory_limit_bytes: int, entry_size_estimate: int = 4000
    ) -> "BoundedTranspositionTable":
        """Create table with entries capped by memory limit.

        Args:
            memory_limit_bytes: Maximum memory to use in bytes
            entry_size_estimate: Approximate bytes per entry (default: 4000)
                Note: entries include children_values dict which can contain
                50+ (move, value) pairs. Realistic sizes are 2-10KB per entry.

        Returns:
            BoundedTranspositionTable configured for the memory limit
        """
        max_entries = max(1000, memory_limit_bytes // entry_size_estimate)
        return cls(
            max_entries=max_entries, entry_size_estimate=entry_size_estimate
        )

    def get(self, key: Hashable) -> Any | None:
        """Get value, moving to end if found (LRU).

        Args:
            key: The key to look up

        Returns:
            The value if found, None otherwise
        """
        if key in self._table:
            self._table.move_to_end(key)
            self.hits += 1
            return self._table[key]
        self.misses += 1
        return None

    def put(self, key: Hashable, value: Any) -> None:
        """Add entry, evicting oldest if at capacity.

        Args:
            key: The key to store
            value: The value to store
        """
        if key in self._table:
            self._table.move_to_end(key)
        else:
            if len(self._table) >= self.max_entries:
                self._table.popitem(last=False)
                self.evictions += 1
        self._table[key] = value

    def __contains__(self, key: Hashable) -> bool:
        """Check if key exists in table."""
        return key in self._table

    def __len__(self) -> int:
        """Return number of entries in table."""
        return len(self._table)

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        self._table.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self) -> dict:
        """Return usage statistics.

        Returns:
            Dictionary with entries, max_entries, hits, misses, evictions,
            hit_rate, and estimated_memory_mb.
        """
        total_lookups = self.hits + self.misses
        hit_rate = self.hits / total_lookups if total_lookups > 0 else 0.0
        estimated_memory_mb = (
            len(self._table) * self.entry_size_estimate / (1024**2)
        )
        return {
            "entries": len(self._table),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "estimated_memory_mb": estimated_memory_mb,
        }
