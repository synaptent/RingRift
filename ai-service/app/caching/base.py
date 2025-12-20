"""Base classes and protocols for the caching system.

This module defines the core abstractions for caching:
- Cache protocol (interface)
- CacheEntry (data container)
- CacheStats (statistics)
- CacheConfig (configuration)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = [
    "Cache",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
]

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@dataclass
class CacheEntry(Generic[V]):
    """An entry in the cache with metadata."""
    value: V
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Get the age of this entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Statistics for a cache instance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int | None = None

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


@dataclass
class CacheConfig:
    """Configuration for a cache instance."""
    max_size: int | None = None  # None = unlimited
    ttl_seconds: float | None = None  # None = no expiration
    cleanup_interval_seconds: float = 60.0  # How often to run cleanup
    eviction_policy: str = "lru"  # "lru", "lfu", "fifo"


class Cache(ABC, Generic[K, V]):
    """Abstract base class for cache implementations.

    This class defines the common interface for all cache types in
    the RingRift codebase.

    Usage:
        class MyCache(Cache[str, MyData]):
            def get(self, key: str) -> Optional[MyData]:
                ...

            def set(self, key: str, value: MyData, ttl: float = None):
                ...
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._stats = CacheStats(max_size=self.config.max_size)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @abstractmethod
    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value from the cache.

        Args:
            key: Cache key
            default: Value to return if key not found

        Returns:
            Cached value or default
        """
        pass

    @abstractmethod
    def set(
        self,
        key: K,
        value: V,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override (uses config default if None)
        """
        pass

    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete a key from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    def has(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        pass

    def get_or_set(
        self,
        key: K,
        factory: Callable[[], V],
        ttl_seconds: float | None = None,
    ) -> V:
        """Get a value or compute and cache it if missing.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl_seconds: Optional TTL for new entry

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl_seconds)
        return value

    async def get_or_set_async(
        self,
        key: K,
        factory: Callable[[], Any],  # Can return Awaitable[V]
        ttl_seconds: float | None = None,
    ) -> V:
        """Async version of get_or_set.

        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl_seconds: Optional TTL for new entry

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        import asyncio
        result = factory()
        if asyncio.iscoroutine(result):
            value = await result
        else:
            value = result

        self.set(key, value, ttl_seconds)
        return value

    def __contains__(self, key: K) -> bool:
        """Support 'in' operator."""
        return self.has(key)

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return self._stats.size
