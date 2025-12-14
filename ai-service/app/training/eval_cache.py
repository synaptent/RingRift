"""Evaluation result caching for pool-based promotion gating.

This module provides caching for model-vs-model evaluation results to avoid
re-running expensive tournament games when evaluating against a pool of
historical opponents.

The cache is keyed by:
- Hash of model A checkpoint file (mtime + size)
- Hash of model B checkpoint file (mtime + size)
- Board type and player count
- Number of games played

Cache entries expire after a configurable TTL (default: 7 days) to handle
model file updates.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalCacheEntry:
    """Cached evaluation result."""
    model_a_hash: str
    model_b_hash: str
    board: str
    num_players: int
    games_played: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    winrate: float
    timestamp: float

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


def _compute_file_hash(path: Path) -> str:
    """Compute a hash of a file based on mtime, size, and first 1KB.

    This is faster than hashing the entire checkpoint file while still
    detecting most changes.
    """
    try:
        stat = path.stat()
        mtime = int(stat.st_mtime)
        size = stat.st_size

        # Read first 1KB for content hash
        with open(path, 'rb') as f:
            head = f.read(1024)

        content_hash = hashlib.md5(head).hexdigest()[:8]
        return f"{mtime}_{size}_{content_hash}"
    except OSError as e:
        logger.debug(f"Failed to hash file {path}: {e}")
        return ""


def _make_cache_key(
    model_a_path: Path,
    model_b_path: Path,
    board: str,
    num_players: int,
    games: int,
) -> str:
    """Create a cache key for an evaluation pair."""
    a_hash = _compute_file_hash(model_a_path)
    b_hash = _compute_file_hash(model_b_path)

    if not a_hash or not b_hash:
        return ""

    # Canonical ordering: smaller hash first
    if a_hash > b_hash:
        a_hash, b_hash = b_hash, a_hash

    return f"{board}_{num_players}p_{a_hash}_{b_hash}_{games}"


class EvalCache:
    """Cache for pool-based evaluation results.

    Stores evaluation results in a JSON file to avoid replaying games against
    opponents we've already evaluated. The cache uses file hashes to detect
    when models have been updated.

    Example usage:
        >>> cache = EvalCache("/path/to/cache.json")
        >>> result = cache.get(model_a, model_b, "square8", 2, 100)
        >>> if result is None:
        ...     result = run_tournament(model_a, model_b, ...)
        ...     cache.put(model_a, model_b, "square8", 2, 100, result)
    """

    def __init__(
        self,
        cache_path: str,
        ttl_seconds: float = 7 * 24 * 3600,  # 7 days default
        max_entries: int = 1000,
    ):
        """Initialize the evaluation cache.

        Args:
            cache_path: Path to the cache JSON file
            ttl_seconds: Time-to-live for cache entries (default: 7 days)
            max_entries: Maximum number of entries to keep (LRU eviction)
        """
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_path.exists():
            self._cache = {}
            return

        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)

            # Filter expired entries
            now = time.time()
            self._cache = {
                k: v for k, v in data.items()
                if now - v.get('timestamp', 0) < self.ttl_seconds
            }

            expired = len(data) - len(self._cache)
            if expired > 0:
                logger.info(f"Eval cache: evicted {expired} expired entries")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load eval cache from {self.cache_path}: {e}")
            self._cache = {}

    def _save(self) -> None:
        """Persist cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            logger.warning(f"Failed to save eval cache to {self.cache_path}: {e}")

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries if over max_entries."""
        if len(self._cache) <= self.max_entries:
            return

        # Sort by timestamp (oldest first)
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].get('timestamp', 0)
        )

        # Remove oldest entries
        to_remove = len(self._cache) - self.max_entries
        for key in sorted_keys[:to_remove]:
            del self._cache[key]

        logger.info(f"Eval cache: evicted {to_remove} LRU entries")

    def get(
        self,
        model_a_path: Path,
        model_b_path: Path,
        board: str,
        num_players: int,
        games: int,
    ) -> Optional[EvalCacheEntry]:
        """Get cached evaluation result.

        Args:
            model_a_path: Path to first model checkpoint
            model_b_path: Path to second model checkpoint
            board: Board type (square8, square19, hexagonal)
            num_players: Number of players
            games: Number of games in the evaluation

        Returns:
            EvalCacheEntry if found and not expired, None otherwise
        """
        key = _make_cache_key(model_a_path, model_b_path, board, num_players, games)
        if not key:
            return None

        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check expiry
        age = time.time() - entry.get('timestamp', 0)
        if age > self.ttl_seconds:
            del self._cache[key]
            return None

        try:
            return EvalCacheEntry(
                model_a_hash=entry['model_a_hash'],
                model_b_hash=entry['model_b_hash'],
                board=entry['board'],
                num_players=entry['num_players'],
                games_played=entry['games_played'],
                model_a_wins=entry['model_a_wins'],
                model_b_wins=entry['model_b_wins'],
                draws=entry['draws'],
                winrate=entry['winrate'],
                timestamp=entry['timestamp'],
            )
        except KeyError:
            # Malformed entry
            del self._cache[key]
            return None

    def put(
        self,
        model_a_path: Path,
        model_b_path: Path,
        board: str,
        num_players: int,
        games: int,
        model_a_wins: int,
        model_b_wins: int,
        draws: int = 0,
    ) -> None:
        """Store evaluation result in cache.

        Args:
            model_a_path: Path to first model checkpoint
            model_b_path: Path to second model checkpoint
            board: Board type
            num_players: Number of players
            games: Number of games played
            model_a_wins: Wins for model A
            model_b_wins: Wins for model B
            draws: Number of draws
        """
        key = _make_cache_key(model_a_path, model_b_path, board, num_players, games)
        if not key:
            return

        total = model_a_wins + model_b_wins + draws
        winrate = model_a_wins / total if total > 0 else 0.5

        a_hash = _compute_file_hash(model_a_path)
        b_hash = _compute_file_hash(model_b_path)

        entry = EvalCacheEntry(
            model_a_hash=a_hash,
            model_b_hash=b_hash,
            board=board,
            num_players=num_players,
            games_played=games,
            model_a_wins=model_a_wins,
            model_b_wins=model_b_wins,
            draws=draws,
            winrate=winrate,
            timestamp=time.time(),
        )

        self._cache[key] = asdict(entry)
        self._evict_lru()
        self._save()

        logger.debug(
            f"Eval cache: stored {model_a_path.name} vs {model_b_path.name} "
            f"({model_a_wins}-{model_b_wins}-{draws})"
        )

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache = {}
        self._save()
        logger.info("Eval cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        ages = [now - e.get('timestamp', 0) for e in self._cache.values()]

        return {
            'entries': len(self._cache),
            'max_entries': self.max_entries,
            'ttl_seconds': self.ttl_seconds,
            'oldest_age_hours': max(ages) / 3600 if ages else 0,
            'newest_age_hours': min(ages) / 3600 if ages else 0,
            'avg_age_hours': sum(ages) / len(ages) / 3600 if ages else 0,
        }


# Global cache instance for convenience
_global_cache: Optional[EvalCache] = None


def get_eval_cache(cache_dir: Optional[str] = None) -> EvalCache:
    """Get or create the global evaluation cache.

    Args:
        cache_dir: Directory for cache file. If None, uses default location
                   (ai-service/logs/improvement/eval_cache.json)

    Returns:
        EvalCache instance
    """
    global _global_cache

    if _global_cache is None:
        if cache_dir is None:
            # Default to logs/improvement directory
            this_file = Path(__file__).resolve()
            ai_service_root = this_file.parents[2]
            cache_dir = ai_service_root / "logs" / "improvement"

        cache_path = Path(cache_dir) / "eval_cache.json"
        _global_cache = EvalCache(str(cache_path))

    return _global_cache
