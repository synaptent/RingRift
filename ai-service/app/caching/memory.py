"""In-memory cache implementations.

Provides thread-safe in-memory caching with various eviction policies:
- MemoryCache: Basic cache with TTL and size limits
- LRUCache: Least Recently Used eviction
- TTLCache: Time-based expiration focused
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TypeVar

from app.caching.base import Cache, CacheConfig, CacheEntry

__all__ = [
    "LRUCache",
    "MemoryCache",
    "TTLCache",
]

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class MemoryCache(Cache[K, V]):
    """Thread-safe in-memory cache with TTL and size limits.

    This is the recommended general-purpose cache for most use cases.

    Features:
    - TTL-based expiration
    - Size-limited with LRU eviction
    - Thread-safe operations
    - Statistics tracking

    Usage:
        cache = MemoryCache(max_size=1000, ttl_seconds=3600)
        cache.set("key", value)
        value = cache.get("key")
    """

    def __init__(
        self,
        max_size: int | None = None,
        ttl_seconds: float | None = None,
        config: CacheConfig | None = None,
    ):
        """Initialize the memory cache.

        Args:
            max_size: Maximum number of entries (None = unlimited)
            ttl_seconds: Default TTL in seconds (None = no expiration)
            config: Full config (overrides max_size and ttl_seconds if provided)
        """
        if config is None:
            config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds)
        super().__init__(config)

        self._data: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value from the cache."""
        with self._lock:
            entry = self._data.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired():
                self._evict_key(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                return default

            # Move to end for LRU
            self._data.move_to_end(key)
            entry.touch()

            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: K,
        value: V,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set a value in the cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.config.ttl_seconds

        with self._lock:
            # Evict if at capacity
            while (
                self.config.max_size is not None
                and len(self._data) >= self.config.max_size
                and key not in self._data
            ):
                self._evict_oldest()

            entry = CacheEntry(value=value, ttl_seconds=ttl)
            self._data[key] = entry
            self._data.move_to_end(key)
            self._stats.size = len(self._data)

    def delete(self, key: K) -> bool:
        """Delete a key from the cache."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._stats.size = len(self._data)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._data.clear()
            self._stats.size = 0

    def has(self, key: K) -> bool:
        """Check if a key exists and is not expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._evict_key(key)
                return False
            return True

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                k for k, v in self._data.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._data[key]

            self._stats.expirations += len(expired_keys)
            self._stats.size = len(self._data)
            return len(expired_keys)

    def keys(self) -> list:
        """Get all non-expired keys."""
        with self._lock:
            return [k for k, v in self._data.items() if not v.is_expired()]

    def items(self) -> list:
        """Get all non-expired items as (key, value) pairs."""
        with self._lock:
            return [
                (k, v.value) for k, v in self._data.items()
                if not v.is_expired()
            ]

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (LRU)."""
        if self._data:
            oldest_key = next(iter(self._data))
            del self._data[oldest_key]
            self._stats.evictions += 1
            self._stats.size = len(self._data)

    def _evict_key(self, key: K) -> None:
        """Evict a specific key."""
        if key in self._data:
            del self._data[key]
            self._stats.size = len(self._data)


class LRUCache(MemoryCache[K, V]):
    """LRU (Least Recently Used) cache.

    This is an alias for MemoryCache with explicit LRU semantics.
    """

    def __init__(
        self,
        max_size: int,
        ttl_seconds: float | None = None,
    ):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries (required)
            ttl_seconds: Optional TTL in seconds
        """
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds)


class TTLCache(MemoryCache[K, V]):
    """TTL-focused cache with automatic expiration.

    This cache emphasizes time-based expiration with configurable
    cleanup intervals.

    Usage:
        cache = TTLCache(ttl_seconds=300)  # 5-minute TTL
        cache.set("key", value)
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_size: int | None = None,
        cleanup_interval_seconds: float = 60.0,
    ):
        """Initialize the TTL cache.

        Args:
            ttl_seconds: Default TTL (required)
            max_size: Optional size limit
            cleanup_interval_seconds: How often to run automatic cleanup
        """
        config = CacheConfig(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )
        super().__init__(config=config)
        self._last_cleanup = time.time()

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value, running cleanup if needed."""
        self._maybe_cleanup()
        return super().get(key, default)

    def set(
        self,
        key: K,
        value: V,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set a value, running cleanup if needed."""
        self._maybe_cleanup()
        super().set(key, value, ttl_seconds)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self.config.cleanup_interval_seconds:
            self.cleanup_expired()
            self._last_cleanup = now
