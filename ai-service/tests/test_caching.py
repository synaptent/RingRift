"""Tests for the caching module.

Comprehensive tests for:
- CacheEntry: creation, expiration, touch
- CacheStats: hit rate, reset
- MemoryCache: get/set, TTL, LRU eviction
- LRUCache: eviction order
- TTLCache: expiration
- Decorators: @cached, @async_cached
"""

import asyncio
import time
from unittest.mock import patch

import pytest


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a cache entry."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test_value")

        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.ttl_seconds is None

    def test_entry_with_ttl(self):
        """Test entry with TTL."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test", ttl_seconds=60.0)

        assert entry.ttl_seconds == 60.0
        assert not entry.is_expired()

    def test_entry_expiration(self):
        """Test entry expiration check."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test", ttl_seconds=0.01)

        assert not entry.is_expired()
        time.sleep(0.02)
        assert entry.is_expired()

    def test_entry_no_expiration_without_ttl(self):
        """Test entry without TTL never expires."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test", ttl_seconds=None)
        entry.created_at = 0  # Set to epoch

        assert not entry.is_expired()

    def test_entry_touch(self):
        """Test touching an entry updates access time and count."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test")
        original_access_time = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == 1
        assert entry.last_accessed > original_access_time

    def test_entry_age(self):
        """Test entry age calculation."""
        from app.caching import CacheEntry

        entry = CacheEntry(value="test")
        time.sleep(0.01)

        assert entry.age_seconds >= 0.01


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_stats_creation(self):
        """Test creating cache stats."""
        from app.caching import CacheStats

        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.size == 0

    def test_total_requests(self):
        """Test total requests calculation."""
        from app.caching import CacheStats

        stats = CacheStats(hits=10, misses=5)

        assert stats.total_requests == 15

    def test_hit_rate_empty(self):
        """Test hit rate with no requests."""
        from app.caching import CacheStats

        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        from app.caching import CacheStats

        stats = CacheStats(hits=75, misses=25)

        assert stats.hit_rate == 0.75

    def test_stats_reset(self):
        """Test resetting statistics."""
        from app.caching import CacheStats

        stats = CacheStats(hits=100, misses=50, evictions=10)
        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestMemoryCache:
    """Tests for MemoryCache class."""

    def test_cache_creation(self):
        """Test creating a memory cache."""
        from app.caching import MemoryCache

        cache = MemoryCache(max_size=100, ttl_seconds=60)

        assert cache is not None

    def test_set_and_get(self):
        """Test basic set and get operations."""
        from app.caching import MemoryCache

        cache = MemoryCache()
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        """Test getting a missing key returns None."""
        from app.caching import MemoryCache

        cache = MemoryCache()

        assert cache.get("missing") is None

    def test_get_with_default(self):
        """Test getting a missing key with default."""
        from app.caching import MemoryCache

        cache = MemoryCache()

        assert cache.get("missing", default="default") == "default"

    def test_delete(self):
        """Test deleting a key."""
        from app.caching import MemoryCache

        cache = MemoryCache()
        cache.set("key", "value")
        cache.delete("key")

        assert cache.get("key") is None

    def test_clear(self):
        """Test clearing the cache."""
        from app.caching import MemoryCache

        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        from app.caching import MemoryCache

        cache = MemoryCache(ttl_seconds=0.01)
        cache.set("key", "value")

        assert cache.get("key") == "value"
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_size_limit_eviction(self):
        """Test LRU eviction when size limit is reached."""
        from app.caching import MemoryCache

        cache = MemoryCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_stats_tracking(self):
        """Test that stats are tracked correctly."""
        from app.caching import MemoryCache

        cache = MemoryCache()
        cache.set("key", "value")

        cache.get("key")  # Hit
        cache.get("missing")  # Miss

        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1

    def test_contains(self):
        """Test __contains__ method."""
        from app.caching import MemoryCache

        cache = MemoryCache()
        cache.set("key", "value")

        assert "key" in cache
        assert "missing" not in cache

    def test_thread_safety(self):
        """Test that cache is thread-safe."""
        import threading
        from app.caching import MemoryCache

        cache = MemoryCache(max_size=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{threading.current_thread().name}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, name=f"writer_{i}"))
            threads.append(threading.Thread(target=reader, name=f"reader_{i}"))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_lru_creation(self):
        """Test creating an LRU cache."""
        from app.caching import LRUCache

        cache = LRUCache(max_size=100)
        assert cache is not None

    def test_lru_eviction_order(self):
        """Test that LRU eviction removes least recently used."""
        from app.caching import LRUCache

        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still there (recently accessed)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestTTLCache:
    """Tests for TTLCache class."""

    def test_ttl_cache_creation(self):
        """Test creating a TTL cache."""
        from app.caching import TTLCache

        cache = TTLCache(ttl_seconds=60)
        assert cache is not None

    def test_ttl_per_key(self):
        """Test per-key TTL override."""
        from app.caching import TTLCache

        cache = TTLCache(ttl_seconds=60)

        # Set with default TTL
        cache.set("long_lived", "value1")

        # Set with short TTL
        cache.set("short_lived", "value2", ttl_seconds=0.01)

        time.sleep(0.02)

        assert cache.get("long_lived") == "value1"
        assert cache.get("short_lived") is None


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_cached_basic(self):
        """Test basic caching with decorator."""
        from app.caching import cached

        call_count = 0

        @cached(ttl_seconds=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once due to caching

    def test_cached_different_args(self):
        """Test that different args are cached separately."""
        from app.caching import cached

        call_count = 0

        @cached(ttl_seconds=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_function(5)
        expensive_function(10)

        assert call_count == 2  # Called twice for different args

    def test_cached_expiration(self):
        """Test that cached values expire."""
        from app.caching import cached

        call_count = 0

        @cached(ttl_seconds=0.01)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_function(5)
        time.sleep(0.02)
        expensive_function(5)

        assert call_count == 2  # Called twice due to expiration


class TestAsyncCachedDecorator:
    """Tests for @async_cached decorator."""

    @pytest.mark.asyncio
    async def test_async_cached_basic(self):
        """Test basic async caching."""
        from app.caching import async_cached

        call_count = 0

        @async_cached(ttl_seconds=60)
        async def async_expensive(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return x * 2

        result1 = await async_expensive(5)
        result2 = await async_expensive(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_base_exports(self):
        """Test base module exports."""
        from app.caching import Cache, CacheEntry, CacheStats, CacheConfig

        assert Cache is not None
        assert CacheEntry is not None
        assert CacheStats is not None
        assert CacheConfig is not None

    def test_memory_exports(self):
        """Test memory cache exports."""
        from app.caching import MemoryCache, LRUCache, TTLCache

        assert MemoryCache is not None
        assert LRUCache is not None
        assert TTLCache is not None

    def test_decorator_exports(self):
        """Test decorator exports."""
        from app.caching import cached, async_cached, invalidate_cache

        assert callable(cached)
        assert callable(async_cached)
        assert callable(invalidate_cache)
