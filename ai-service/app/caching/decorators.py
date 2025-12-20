"""Caching decorators for function memoization.

Provides decorators for automatic caching of function results:
- @cached: Synchronous function caching
- @async_cached: Async function caching
- invalidate_cache: Clear cached results

Usage:
    from app.caching import cached, async_cached

    @cached(ttl_seconds=300, max_size=100)
    def expensive_computation(x: int) -> int:
        ...

    @async_cached(ttl_seconds=60)
    async def fetch_data(url: str) -> dict:
        ...
"""

from __future__ import annotations

import functools
import hashlib
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from app.caching.memory import MemoryCache

__all__ = [
    "async_cached",
    "cached",
    "invalidate_cache",
]

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Global registry of caches for cache invalidation
_cache_registry: dict[str, MemoryCache] = {}


def _make_key(args: tuple, kwargs: dict) -> str:
    """Create a cache key from function arguments."""
    try:
        # Try to create a hashable key
        key_parts = [repr(a) for a in args]
        key_parts.extend(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)

        # Hash long keys for efficiency
        if len(key_str) > 200:
            return hashlib.md5(key_str.encode()).hexdigest()
        return key_str
    except Exception:
        # Fallback to str representation
        return str((args, sorted(kwargs.items())))


def cached(
    ttl_seconds: float | None = None,
    max_size: int | None = 1000,
    key_func: Callable[..., str] | None = None,
    cache_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for caching function results.

    Args:
        ttl_seconds: Time-to-live for cached results (None = no expiration)
        max_size: Maximum cache size (default 1000)
        key_func: Custom function to generate cache keys
        cache_name: Name for cache invalidation (defaults to function name)

    Returns:
        Decorated function with caching

    Example:
        @cached(ttl_seconds=300)
        def get_user(user_id: int) -> User:
            return fetch_user_from_db(user_id)

        # With custom key function
        @cached(key_func=lambda self, id: f"user:{id}")
        def get_user(self, user_id: int) -> User:
            ...
    """
    def decorator(func: F) -> F:
        cache = MemoryCache(max_size=max_size, ttl_seconds=ttl_seconds)
        name = cache_name or func.__qualname__
        _cache_registry[name] = cache

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _make_key(args, kwargs)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result)

            return result

        # Attach cache for manual access
        wrapper._cache = cache  # type: ignore
        wrapper._cache_name = name  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_cached(
    ttl_seconds: float | None = None,
    max_size: int | None = 1000,
    key_func: Callable[..., str] | None = None,
    cache_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for caching async function results.

    Args:
        ttl_seconds: Time-to-live for cached results (None = no expiration)
        max_size: Maximum cache size (default 1000)
        key_func: Custom function to generate cache keys
        cache_name: Name for cache invalidation

    Returns:
        Decorated async function with caching

    Example:
        @async_cached(ttl_seconds=60)
        async def fetch_data(url: str) -> dict:
            async with httpx.AsyncClient() as client:
                return await client.get(url).json()
    """
    def decorator(func: F) -> F:
        cache = MemoryCache(max_size=max_size, ttl_seconds=ttl_seconds)
        name = cache_name or func.__qualname__
        _cache_registry[name] = cache

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _make_key(args, kwargs)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                cache.set(key, result)

            return result

        # Attach cache for manual access
        wrapper._cache = cache  # type: ignore
        wrapper._cache_name = name  # type: ignore

        return wrapper  # type: ignore

    return decorator


def invalidate_cache(
    cache_name: str | None = None,
    func: Callable | None = None,
) -> int:
    """Invalidate cached results.

    Can invalidate by cache name or by passing the decorated function.

    Args:
        cache_name: Name of cache to invalidate
        func: Decorated function whose cache to invalidate

    Returns:
        Number of entries cleared

    Example:
        @cached(cache_name="users")
        def get_user(user_id): ...

        # Later, invalidate all cached users
        invalidate_cache("users")

        # Or by function
        invalidate_cache(func=get_user)
    """
    cleared = 0

    if func is not None and hasattr(func, "_cache"):
        cache = func._cache
        cleared = len(cache)
        cache.clear()
        return cleared

    if cache_name is not None:
        cache = _cache_registry.get(cache_name)
        if cache is not None:
            cleared = len(cache)
            cache.clear()
        return cleared

    # Clear all caches if neither specified
    for cache in _cache_registry.values():
        cleared += len(cache)
        cache.clear()

    return cleared


def get_cache_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all registered caches.

    Returns:
        Dict mapping cache names to their statistics
    """
    return {
        name: {
            "hits": cache.stats.hits,
            "misses": cache.stats.misses,
            "size": cache.stats.size,
            "hit_rate": cache.stats.hit_rate,
            "evictions": cache.stats.evictions,
        }
        for name, cache in _cache_registry.items()
    }
