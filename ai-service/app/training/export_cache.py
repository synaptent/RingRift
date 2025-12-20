"""Incremental Export Cache for Training Data.

Provides caching for expensive NPZ exports from GameReplayDB databases.
Tracks source database modification times and content hashes to determine
when re-export is necessary vs when cached output can be reused.

Usage:
    from app.training.export_cache import ExportCache

    cache = ExportCache()

    # Check if export is needed
    if cache.needs_export(db_paths, output_path, board_type, num_players):
        # Perform export
        export_replay_dataset_multi(...)
        # Update cache
        cache.record_export(db_paths, output_path, board_type, num_players)
    else:
        print(f"Using cached export: {output_path}")
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import compute_string_checksum

# Cache directory
from app.utils.paths import DATA_DIR, ensure_dir

EXPORT_CACHE_DIR = ensure_dir(DATA_DIR / "export_cache")


@dataclass
class ExportCacheEntry:
    """Cache entry for a single export operation."""
    output_path: str
    board_type: str
    num_players: int
    db_sources: dict[str, dict[str, Any]]  # path -> {mtime, size, game_count}
    export_timestamp: str
    samples_exported: int
    games_exported: int
    output_size: int
    output_mtime: float
    history_length: int | None = None
    feature_version: int | None = None
    policy_encoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExportCacheEntry:
        return cls(**d)


def _get_db_stats(db_path: str) -> dict[str, Any]:
    """Get stats for a database file (mtime, size, game count)."""
    path = Path(db_path)
    if not path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "mtime": path.stat().st_mtime,
        "size": path.stat().st_size,
    }

    # Try to get game count from database
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games")
        stats["game_count"] = cursor.fetchone()[0]
        conn.close()
    except Exception:
        stats["game_count"] = -1

    return stats


def _get_cache_key(
    board_type: str,
    num_players: int,
    output_path: str,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
) -> str:
    """Generate a unique cache key for this export configuration."""
    # Use a hash of the normalized output path plus feature context.
    output_norm = os.path.normpath(os.path.abspath(output_path))
    key_parts = [f"{board_type}_{num_players}p_{output_norm}"]
    if history_length is not None:
        key_parts.append(f"h{int(history_length)}")
    if feature_version is not None:
        key_parts.append(f"fv{int(feature_version)}")
    if policy_encoding:
        key_parts.append(str(policy_encoding))
    key_str = "_".join(key_parts)
    return compute_string_checksum(key_str, algorithm="md5", truncate=16)


def _get_cache_file(cache_key: str) -> Path:
    """Get the cache file path for a given key."""
    return EXPORT_CACHE_DIR / f"export_{cache_key}.json"


class ExportCache:
    """Manages incremental export caching for training data."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or EXPORT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache_entry(self, cache_key: str) -> ExportCacheEntry | None:
        """Load a cache entry by key."""
        cache_file = self.cache_dir / f"export_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
            return ExportCacheEntry.from_dict(data)
        except Exception:
            return None

    def _save_cache_entry(self, cache_key: str, entry: ExportCacheEntry) -> None:
        """Save a cache entry."""
        cache_file = self.cache_dir / f"export_{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def needs_export(
        self,
        db_paths: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        force: bool = False,
    ) -> bool:
        """Check if export is needed or if cached output is still valid.

        Returns True if export is needed, False if cache is valid.

        Cache is considered valid if:
        1. Output file exists
        2. Cache entry exists for this config
        3. All source DBs have same mtime and game_count as when cached
        4. Output file has same mtime as when cached
        """
        if force:
            return True

        # Check if output exists
        output = Path(output_path)
        if not output.exists():
            return True

        # Load cache entry
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
        )
        entry = self._load_cache_entry(cache_key)
        if entry is None:
            return True

        # Check output file hasn't been modified externally
        current_output_mtime = output.stat().st_mtime
        if abs(current_output_mtime - entry.output_mtime) > 1.0:
            return True

        # Check all source databases
        for db_path in db_paths:
            db_path_norm = os.path.normpath(os.path.abspath(db_path))
            cached_stats = entry.db_sources.get(db_path_norm)

            if cached_stats is None:
                # New database not in cache
                return True

            current_stats = _get_db_stats(db_path)

            if not current_stats.get("exists", False):
                # Source DB was removed - could skip, but safer to re-export
                continue

            # Check mtime (primary indicator of changes)
            if current_stats["mtime"] > cached_stats.get("mtime", 0):
                return True

            # Check game count (secondary check for content changes)
            cached_count = cached_stats.get("game_count", -1)
            current_count = current_stats.get("game_count", -1)
            if cached_count >= 0 and current_count >= 0 and current_count != cached_count:
                return True

        # Check if there are new DBs not in the cache
        cached_db_paths = set(entry.db_sources.keys())
        current_db_paths = {os.path.normpath(os.path.abspath(p)) for p in db_paths if Path(p).exists()}
        if current_db_paths - cached_db_paths:
            # New databases added
            return True

        # Cache is valid
        return False

    def record_export(
        self,
        db_paths: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        samples_exported: int = 0,
        games_exported: int = 0,
    ) -> None:
        """Record a completed export to the cache."""
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
        )

        # Collect source DB stats
        db_sources = {}
        for db_path in db_paths:
            if Path(db_path).exists():
                db_path_norm = os.path.normpath(os.path.abspath(db_path))
                db_sources[db_path_norm] = _get_db_stats(db_path)

        # Get output stats
        output = Path(output_path)
        output_size = output.stat().st_size if output.exists() else 0
        output_mtime = output.stat().st_mtime if output.exists() else 0

        entry = ExportCacheEntry(
            output_path=os.path.normpath(os.path.abspath(output_path)),
            board_type=board_type,
            num_players=num_players,
            db_sources=db_sources,
            export_timestamp=datetime.now().isoformat(),
            samples_exported=samples_exported,
            games_exported=games_exported,
            output_size=output_size,
            output_mtime=output_mtime,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
        )

        self._save_cache_entry(cache_key, entry)

    def invalidate(
        self,
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
    ) -> bool:
        """Invalidate a cache entry. Returns True if entry was found and removed."""
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
        )
        cache_file = self.cache_dir / f"export_{cache_key}.json"

        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def get_cache_info(
        self,
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
    ) -> dict[str, Any] | None:
        """Get cache entry info for debugging/inspection."""
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
        )
        entry = self._load_cache_entry(cache_key)
        if entry:
            return entry.to_dict()
        return None

    def cleanup_stale(self, max_age_days: int = 30) -> int:
        """Remove cache entries older than max_age_days. Returns count removed."""
        import time
        cutoff = time.time() - (max_age_days * 24 * 3600)
        removed = 0

        for cache_file in self.cache_dir.glob("export_*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                pass

        return removed


# Convenience functions for direct use
_default_cache = None

def get_export_cache() -> ExportCache:
    """Get the default export cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = ExportCache()
    return _default_cache


def needs_export(
    db_paths: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    force: bool = False,
) -> bool:
    """Check if export is needed using the default cache."""
    return get_export_cache().needs_export(
        db_paths,
        output_path,
        board_type,
        num_players,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        force=force,
    )


def record_export(
    db_paths: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    samples_exported: int = 0,
    games_exported: int = 0,
) -> None:
    """Record an export using the default cache."""
    get_export_cache().record_export(
        db_paths,
        output_path,
        board_type,
        num_players,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        samples_exported=samples_exported,
        games_exported=games_exported,
    )
