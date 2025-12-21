"""File-based persistent cache implementations.

Provides JSON-backed caching for data that needs to survive restarts:
- FileCache: General-purpose file-backed cache with TTL and LRU
- ValidatedFileCache: Adds custom validation hooks (e.g., file hash checks)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from app.caching.base import Cache, CacheConfig, CacheEntry, CacheStats

__all__ = [
    "FileCache",
    "ValidatedFileCache",
]

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class FileCache(Cache[str, V]):
    """File-backed persistent cache with TTL and LRU eviction.

    Stores cache data as JSON on disk, persisting across process restarts.
    Thread-safe with RLock for concurrent access.

    Features:
    - TTL-based expiration
    - LRU eviction when over max_entries
    - Atomic writes (write to temp, rename)
    - Lazy loading on first access

    Usage:
        cache = FileCache("/path/to/cache.json", ttl_seconds=86400, max_entries=1000)
        cache.set("key", {"data": "value"})
        value = cache.get("key")

    Note:
        Keys must be strings (JSON object keys).
        Values must be JSON-serializable.
    """

    def __init__(
        self,
        cache_path: str | Path,
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        auto_save: bool = True,
        cleanup_on_load: bool = True,
    ):
        """Initialize the file cache.

        Args:
            cache_path: Path to the cache JSON file
            ttl_seconds: Default TTL in seconds (None = no expiration)
            max_entries: Maximum entries before LRU eviction (None = unlimited)
            auto_save: Whether to persist after each write operation
            cleanup_on_load: Whether to evict expired entries on load
        """
        config = CacheConfig(max_size=max_entries, ttl_seconds=ttl_seconds)
        super().__init__(config)

        self.cache_path = Path(cache_path)
        self.auto_save = auto_save
        self.cleanup_on_load = cleanup_on_load

        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load cache from disk on first access."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_path.exists():
            self._data = {}
            return

        try:
            with open(self.cache_path) as f:
                raw_data = json.load(f)

            if self.cleanup_on_load:
                now = time.time()
                self._data = {}
                expired_count = 0

                for key, entry in raw_data.items():
                    ttl = entry.get("ttl_seconds")
                    created = entry.get("created_at", 0)

                    if ttl is not None and now - created > ttl:
                        expired_count += 1
                        self._stats.expirations += 1
                    else:
                        self._data[key] = entry

                if expired_count > 0:
                    logger.debug(f"FileCache: evicted {expired_count} expired entries on load")
            else:
                self._data = raw_data

            self._stats.size = len(self._data)
            logger.debug(f"FileCache: loaded {len(self._data)} entries from {self.cache_path}")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
            self._data = {}

    def _save(self) -> None:
        """Persist cache to disk atomically."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: temp file then rename
            temp_path = self.cache_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(self._data, f, indent=2)

            temp_path.rename(self.cache_path)

        except OSError as e:
            logger.warning(f"Failed to save cache to {self.cache_path}: {e}")

    def _maybe_save(self) -> None:
        """Save if auto_save is enabled."""
        if self.auto_save:
            self._save()

    def _to_entry_dict(self, value: V, ttl_seconds: float | None) -> dict[str, Any]:
        """Convert a value to a storable entry dict."""
        ttl = ttl_seconds if ttl_seconds is not None else self.config.ttl_seconds
        return {
            "value": value,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0,
            "ttl_seconds": ttl,
        }

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if an entry has expired."""
        ttl = entry.get("ttl_seconds")
        if ttl is None:
            return False
        created = entry.get("created_at", 0)
        return time.time() - created > ttl

    def get(self, key: str, default: V | None = None) -> V | None:
        """Get a value from the cache."""
        self._ensure_loaded()

        with self._lock:
            entry = self._data.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if self._is_expired(entry):
                del self._data[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._data)
                self._maybe_save()
                return default

            # Update access metadata
            entry["last_accessed"] = time.time()
            entry["access_count"] = entry.get("access_count", 0) + 1

            self._stats.hits += 1
            return entry.get("value")

    def set(
        self,
        key: str,
        value: V,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set a value in the cache."""
        self._ensure_loaded()

        with self._lock:
            # Evict if at capacity
            self._evict_if_needed(exclude_key=key)

            self._data[key] = self._to_entry_dict(value, ttl_seconds)
            self._stats.size = len(self._data)
            self._maybe_save()

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        self._ensure_loaded()

        with self._lock:
            if key in self._data:
                del self._data[key]
                self._stats.size = len(self._data)
                self._maybe_save()
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._data = {}
            self._stats.size = 0
            self._loaded = True
            self._maybe_save()

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Note: This method does not update hit/miss stats since it's
        just an existence check, not a value retrieval.
        """
        self._ensure_loaded()

        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._data[key]
                self._stats.size = len(self._data)
                return False
            return True

    def _evict_if_needed(self, exclude_key: str | None = None) -> None:
        """Evict LRU entries if over max_size."""
        if self.config.max_size is None:
            return

        while len(self._data) >= self.config.max_size:
            if exclude_key and len(self._data) == 1 and exclude_key in self._data:
                break

            # Find LRU entry
            oldest_key = None
            oldest_access = float("inf")

            for k, v in self._data.items():
                if k == exclude_key:
                    continue
                access_time = v.get("last_accessed", v.get("created_at", 0))
                if access_time < oldest_access:
                    oldest_access = access_time
                    oldest_key = k

            if oldest_key is not None:
                del self._data[oldest_key]
                self._stats.evictions += 1
            else:
                break

        self._stats.size = len(self._data)

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        self._ensure_loaded()

        with self._lock:
            expired_keys = [k for k, v in self._data.items() if self._is_expired(v)]

            for key in expired_keys:
                del self._data[key]

            if expired_keys:
                self._stats.expirations += len(expired_keys)
                self._stats.size = len(self._data)
                self._maybe_save()

            return len(expired_keys)

    def keys(self) -> list[str]:
        """Get all non-expired keys."""
        self._ensure_loaded()

        with self._lock:
            return [k for k, v in self._data.items() if not self._is_expired(v)]

    def save(self) -> None:
        """Explicitly save cache to disk."""
        with self._lock:
            self._save()


class ValidatedFileCache(FileCache[V]):
    """File cache with custom validation hooks.

    Extends FileCache with validation callbacks that can check whether
    cached entries are still valid (e.g., based on source file hashes).

    Usage:
        def validate_model(entry: dict) -> bool:
            # Check if source model file has changed
            return compute_hash(entry['source_path']) == entry['source_hash']

        cache = ValidatedFileCache(
            "/path/to/cache.json",
            validator=validate_model,
        )
    """

    def __init__(
        self,
        cache_path: str | Path,
        validator: Callable[[dict[str, Any]], bool] | None = None,
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        auto_save: bool = True,
    ):
        """Initialize the validated file cache.

        Args:
            cache_path: Path to the cache JSON file
            validator: Function to validate entries. Receives entry dict,
                      returns True if valid. Invalid entries are evicted.
            ttl_seconds: Default TTL in seconds
            max_entries: Maximum entries before LRU eviction
            auto_save: Whether to persist after each write
        """
        super().__init__(
            cache_path=cache_path,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            auto_save=auto_save,
        )
        self._validator = validator

    def _is_valid(self, entry: dict[str, Any]) -> bool:
        """Check if entry passes validation."""
        if self._validator is None:
            return True

        try:
            return self._validator(entry)
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False

    def get(self, key: str, default: V | None = None) -> V | None:
        """Get a value, checking validation."""
        self._ensure_loaded()

        with self._lock:
            entry = self._data.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if self._is_expired(entry) or not self._is_valid(entry):
                del self._data[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._data)
                self._maybe_save()
                return default

            entry["last_accessed"] = time.time()
            entry["access_count"] = entry.get("access_count", 0) + 1

            self._stats.hits += 1
            return entry.get("value")
