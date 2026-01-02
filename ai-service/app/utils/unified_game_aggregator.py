"""Unified Game Aggregator - Single source of truth for game counts across ALL sources.

This module provides a unified view of game availability across:
1. Local filesystem (via GameDiscovery - 14+ patterns)
2. Cluster nodes (via ClusterManifest P2P gossip)
3. Amazon S3 (via TrainingDataManifest)
4. OWC external drive (via OWCImportDaemon discovery)

The aggregator enables:
- Training eligibility checks that see games from ALL sources
- Export operations that can fetch games from remote sources
- Accurate cluster-wide game counts for priority scheduling

Usage:
    from app.utils.unified_game_aggregator import (
        UnifiedGameAggregator,
        AggregatedGameCount,
        get_unified_game_aggregator,
    )

    # Get singleton instance
    aggregator = get_unified_game_aggregator()

    # Get total games across all sources
    counts = await aggregator.get_total_games("hex8", 2)
    print(f"Total: {counts.total_games}")
    print(f"Sources: {counts.sources}")

    # Check training eligibility with cluster-wide awareness
    if counts.total_games >= 5000:
        trigger_training("hex8_2p")

January 2026: Created as part of multi-source game discovery and sync infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "UnifiedGameAggregator",
    "AggregatedGameCount",
    "GameSourceConfig",
    "get_unified_game_aggregator",
    "reset_unified_game_aggregator",
]


# =============================================================================
# Configuration
# =============================================================================

# Environment variable overrides
INCLUDE_REMOTE_DEFAULT = os.getenv("RINGRIFT_INCLUDE_REMOTE_GAMES", "true").lower() == "true"
INCLUDE_S3_DEFAULT = os.getenv("RINGRIFT_INCLUDE_S3_GAMES", "true").lower() == "true"
INCLUDE_OWC_DEFAULT = os.getenv("RINGRIFT_INCLUDE_OWC_GAMES", "true").lower() == "true"

# Cache TTL for remote sources (seconds)
REMOTE_CACHE_TTL = float(os.getenv("RINGRIFT_REMOTE_CACHE_TTL", "300"))  # 5 minutes
S3_CACHE_TTL = float(os.getenv("RINGRIFT_S3_CACHE_TTL", "600"))  # 10 minutes
OWC_CACHE_TTL = float(os.getenv("RINGRIFT_OWC_CACHE_TTL", "300"))  # 5 minutes

# OWC configuration
OWC_HOST = os.getenv("OWC_HOST", "mac-studio")
OWC_USER = os.getenv("OWC_USER", "armand")
OWC_BASE_PATH = os.getenv("OWC_BASE_PATH", "/Volumes/RingRift-Data")
OWC_SSH_KEY = os.getenv("OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GameSourceConfig:
    """Configuration for which sources to include in aggregation."""

    include_local: bool = True
    include_remote: bool = INCLUDE_REMOTE_DEFAULT  # Cluster nodes
    include_s3: bool = INCLUDE_S3_DEFAULT
    include_owc: bool = INCLUDE_OWC_DEFAULT

    # Cache TTLs per source
    local_cache_ttl: float = 60.0  # 1 minute for local (fast refresh)
    remote_cache_ttl: float = REMOTE_CACHE_TTL
    s3_cache_ttl: float = S3_CACHE_TTL
    owc_cache_ttl: float = OWC_CACHE_TTL

    @classmethod
    def all_sources(cls) -> "GameSourceConfig":
        """Include all available sources."""
        return cls(
            include_local=True,
            include_remote=True,
            include_s3=True,
            include_owc=True,
        )

    @classmethod
    def local_only(cls) -> "GameSourceConfig":
        """Include only local sources."""
        return cls(
            include_local=True,
            include_remote=False,
            include_s3=False,
            include_owc=False,
        )


@dataclass
class SourceGameCount:
    """Game count from a single source."""

    source_name: str
    game_count: int
    last_updated: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedGameCount:
    """Aggregated game counts from all sources.

    Provides:
    - Total games across all sources (with deduplication awareness)
    - Per-source breakdown
    - Staleness information
    """

    config_key: str  # e.g., "hex8_2p"
    total_games: int  # Sum across sources (may overcount duplicates)
    sources: dict[str, int]  # {source_name: count}
    source_details: list[SourceGameCount] = field(default_factory=list)
    last_updated: float = 0.0
    errors: list[str] = field(default_factory=list)

    # Deduplicated count if available (requires expensive cross-source check)
    deduplicated_total: int | None = None

    @property
    def is_stale(self) -> bool:
        """Check if aggregation is older than 5 minutes."""
        return time.time() - self.last_updated > 300.0

    @property
    def local_count(self) -> int:
        """Get count from local source only."""
        return self.sources.get("local", 0)

    @property
    def remote_count(self) -> int:
        """Get count from cluster nodes."""
        return self.sources.get("cluster", 0)

    @property
    def s3_count(self) -> int:
        """Get count from S3."""
        return self.sources.get("s3", 0)

    @property
    def owc_count(self) -> int:
        """Get count from OWC external drive."""
        return self.sources.get("owc", 0)


# =============================================================================
# Unified Game Aggregator
# =============================================================================


class UnifiedGameAggregator:
    """Aggregates game counts from all available sources.

    This class provides a unified view of game availability across:
    1. Local filesystem (GameDiscovery)
    2. Cluster nodes (ClusterManifest)
    3. Amazon S3 (TrainingDataManifest)
    4. OWC external drive (direct SSH scan)

    The aggregator caches results per source with configurable TTLs to
    minimize expensive remote queries.

    Thread Safety: Uses asyncio.Lock for cache updates.
    """

    def __init__(self, config: GameSourceConfig | None = None):
        """Initialize the aggregator.

        Args:
            config: Source configuration (defaults to all sources enabled)
        """
        self._config = config or GameSourceConfig()
        self._cache: dict[str, AggregatedGameCount] = {}
        self._source_caches: dict[str, dict[str, SourceGameCount]] = {
            "local": {},
            "cluster": {},
            "s3": {},
            "owc": {},
        }
        self._lock = asyncio.Lock()

        # Lazy-loaded dependencies
        self._game_discovery = None
        self._cluster_manifest = None
        self._training_manifest = None

    # =========================================================================
    # Lazy Dependency Loading
    # =========================================================================

    def _get_game_discovery(self):
        """Get or create GameDiscovery instance."""
        if self._game_discovery is None:
            from app.utils.game_discovery import GameDiscovery
            self._game_discovery = GameDiscovery()
        return self._game_discovery

    def _get_cluster_manifest(self):
        """Get or create ClusterManifest instance."""
        if self._cluster_manifest is None:
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                self._cluster_manifest = get_cluster_manifest()
            except ImportError:
                logger.debug("ClusterManifest not available")
        return self._cluster_manifest

    async def _get_training_manifest(self):
        """Get or create TrainingDataManifest instance."""
        if self._training_manifest is None:
            try:
                from app.coordination.training_data_manifest import (
                    get_training_data_manifest,
                )
                self._training_manifest = await get_training_data_manifest()
            except ImportError:
                logger.debug("TrainingDataManifest not available")
        return self._training_manifest

    # =========================================================================
    # Source-Specific Queries
    # =========================================================================

    async def _get_local_count(
        self, board_type: str, num_players: int
    ) -> SourceGameCount:
        """Get game count from local filesystem via GameDiscovery."""
        config_key = f"{board_type}_{num_players}p"
        now = time.time()

        # Check cache
        cached = self._source_caches["local"].get(config_key)
        if cached and (now - cached.last_updated) < self._config.local_cache_ttl:
            return cached

        try:
            discovery = self._get_game_discovery()
            # Run in thread to avoid blocking
            count = await asyncio.to_thread(
                discovery.get_total_games, board_type, num_players, False  # no cache
            )

            result = SourceGameCount(
                source_name="local",
                game_count=count,
                last_updated=now,
            )
            self._source_caches["local"][config_key] = result
            return result

        except Exception as e:
            logger.warning(f"Error getting local game count for {config_key}: {e}")
            return SourceGameCount(
                source_name="local",
                game_count=0,
                last_updated=now,
                error=str(e),
            )

    async def _get_cluster_count(
        self, board_type: str, num_players: int
    ) -> SourceGameCount:
        """Get game count from cluster nodes via ClusterManifest."""
        config_key = f"{board_type}_{num_players}p"
        now = time.time()

        # Check cache
        cached = self._source_caches["cluster"].get(config_key)
        if cached and (now - cached.last_updated) < self._config.remote_cache_ttl:
            return cached

        try:
            manifest = self._get_cluster_manifest()
            if manifest is None:
                return SourceGameCount(
                    source_name="cluster",
                    game_count=0,
                    last_updated=now,
                    error="ClusterManifest not available",
                )

            # Get counts from manifest (already includes cluster-wide data)
            # Run in thread since it uses SQLite
            counts_by_config = await asyncio.to_thread(
                manifest.get_games_count_by_config
            )
            count = counts_by_config.get(config_key, 0)

            # Also get per-node breakdown
            per_node = await asyncio.to_thread(
                manifest.get_games_by_node_and_config, config_key
            )

            result = SourceGameCount(
                source_name="cluster",
                game_count=count,
                last_updated=now,
                details={"per_node": per_node},
            )
            self._source_caches["cluster"][config_key] = result
            return result

        except Exception as e:
            logger.warning(f"Error getting cluster game count for {config_key}: {e}")
            return SourceGameCount(
                source_name="cluster",
                game_count=0,
                last_updated=now,
                error=str(e),
            )

    async def _get_s3_count(
        self, board_type: str, num_players: int
    ) -> SourceGameCount:
        """Get game count from S3 via TrainingDataManifest.

        Note: S3 stores NPZ files (training samples), not raw games.
        We estimate game count from sample_count (avg ~50 samples/game).
        """
        config_key = f"{board_type}_{num_players}p"
        now = time.time()

        # Check cache
        cached = self._source_caches["s3"].get(config_key)
        if cached and (now - cached.last_updated) < self._config.s3_cache_ttl:
            return cached

        try:
            manifest = await self._get_training_manifest()
            if manifest is None:
                return SourceGameCount(
                    source_name="s3",
                    game_count=0,
                    last_updated=now,
                    error="TrainingDataManifest not available",
                )

            # Get all S3 data for this config
            from app.coordination.training_data_manifest import DataSource
            all_data = manifest.get_all_data(config_key)
            s3_data = [d for d in all_data if d.source == DataSource.S3]

            # Sum sample counts and estimate games
            total_samples = sum(d.sample_count or 0 for d in s3_data)
            # Fallback: estimate from file sizes (~1KB per sample avg)
            if total_samples == 0:
                total_bytes = sum(d.size_bytes for d in s3_data)
                total_samples = total_bytes // 1024

            # Estimate games: ~50 samples per game average
            estimated_games = total_samples // 50 if total_samples > 0 else 0

            result = SourceGameCount(
                source_name="s3",
                game_count=estimated_games,
                last_updated=now,
                details={
                    "files": len(s3_data),
                    "total_samples": total_samples,
                    "total_bytes": sum(d.size_bytes for d in s3_data),
                },
            )
            self._source_caches["s3"][config_key] = result
            return result

        except Exception as e:
            logger.warning(f"Error getting S3 game count for {config_key}: {e}")
            return SourceGameCount(
                source_name="s3",
                game_count=0,
                last_updated=now,
                error=str(e),
            )

    async def _get_owc_count(
        self, board_type: str, num_players: int
    ) -> SourceGameCount:
        """Get game count from OWC external drive via SSH.

        Scans databases in known OWC locations and counts games.
        """
        config_key = f"{board_type}_{num_players}p"
        now = time.time()

        # Check cache
        cached = self._source_caches["owc"].get(config_key)
        if cached and (now - cached.last_updated) < self._config.owc_cache_ttl:
            return cached

        try:
            # Build SSH command to query all relevant databases
            ssh_key_path = Path(OWC_SSH_KEY).expanduser()
            if not ssh_key_path.exists():
                return SourceGameCount(
                    source_name="owc",
                    game_count=0,
                    last_updated=now,
                    error=f"SSH key not found: {ssh_key_path}",
                )

            # Query for games matching this config across OWC databases
            # Use broader search paths to find all databases
            search_paths = [
                f"{OWC_BASE_PATH}/selfplay_repository",
                f"{OWC_BASE_PATH}/training_data",
                f"{OWC_BASE_PATH}/canonical_data",
            ]

            query = f"""
                SELECT COUNT(*) FROM games
                WHERE winner IS NOT NULL
                AND board_type = '{board_type}'
                AND num_players = {num_players}
            """

            total_count = 0
            db_counts: dict[str, int] = {}

            for search_path in search_paths:
                # Find all .db files and query each
                find_cmd = (
                    f"ssh -i {ssh_key_path} -o ConnectTimeout=10 -o BatchMode=yes "
                    f"{OWC_USER}@{OWC_HOST} "
                    f"'find {search_path} -name \"*.db\" -type f 2>/dev/null'"
                )

                import subprocess
                result = await asyncio.to_thread(
                    subprocess.run,
                    find_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    continue

                for db_path in result.stdout.strip().split("\n"):
                    if not db_path.strip():
                        continue

                    # Query this database
                    query_cmd = (
                        f"ssh -i {ssh_key_path} -o ConnectTimeout=10 -o BatchMode=yes "
                        f"{OWC_USER}@{OWC_HOST} "
                        f"\"sqlite3 '{db_path}' \\\"{query}\\\" 2>/dev/null\""
                    )

                    query_result = await asyncio.to_thread(
                        subprocess.run,
                        query_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if query_result.returncode == 0:
                        try:
                            count = int(query_result.stdout.strip())
                            if count > 0:
                                total_count += count
                                db_counts[db_path] = count
                        except ValueError:
                            pass

            result = SourceGameCount(
                source_name="owc",
                game_count=total_count,
                last_updated=now,
                details={"databases": db_counts},
            )
            self._source_caches["owc"][config_key] = result
            return result

        except subprocess.TimeoutExpired:
            return SourceGameCount(
                source_name="owc",
                game_count=0,
                last_updated=now,
                error="SSH timeout connecting to OWC",
            )
        except Exception as e:
            logger.warning(f"Error getting OWC game count for {config_key}: {e}")
            return SourceGameCount(
                source_name="owc",
                game_count=0,
                last_updated=now,
                error=str(e),
            )

    # =========================================================================
    # Public API
    # =========================================================================

    async def get_total_games(
        self,
        board_type: str,
        num_players: int,
        include_remote: bool | None = None,
        include_s3: bool | None = None,
        include_owc: bool | None = None,
    ) -> AggregatedGameCount:
        """Get total game count across all configured sources.

        Args:
            board_type: Board type (hex8, square8, square19, hexagonal)
            num_players: Number of players (2, 3, 4)
            include_remote: Override config to include/exclude cluster nodes
            include_s3: Override config to include/exclude S3
            include_owc: Override config to include/exclude OWC drive

        Returns:
            AggregatedGameCount with total and per-source breakdown
        """
        config_key = f"{board_type}_{num_players}p"

        # Determine which sources to query
        query_local = self._config.include_local
        query_remote = (
            include_remote if include_remote is not None
            else self._config.include_remote
        )
        query_s3 = include_s3 if include_s3 is not None else self._config.include_s3
        query_owc = include_owc if include_owc is not None else self._config.include_owc

        # Build list of coroutines to run concurrently
        tasks: list[tuple[str, Any]] = []

        if query_local:
            tasks.append(("local", self._get_local_count(board_type, num_players)))
        if query_remote:
            tasks.append(("cluster", self._get_cluster_count(board_type, num_players)))
        if query_s3:
            tasks.append(("s3", self._get_s3_count(board_type, num_players)))
        if query_owc:
            tasks.append(("owc", self._get_owc_count(board_type, num_players)))

        # Run all queries concurrently
        if tasks:
            results = await asyncio.gather(
                *[coro for _, coro in tasks],
                return_exceptions=True,
            )
        else:
            results = []

        # Process results
        sources: dict[str, int] = {}
        source_details: list[SourceGameCount] = []
        errors: list[str] = []
        total = 0

        for i, (source_name, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                errors.append(f"{source_name}: {result}")
                source_details.append(SourceGameCount(
                    source_name=source_name,
                    game_count=0,
                    last_updated=time.time(),
                    error=str(result),
                ))
            else:
                source_count: SourceGameCount = result
                sources[source_name] = source_count.game_count
                source_details.append(source_count)
                total += source_count.game_count
                if source_count.error:
                    errors.append(f"{source_name}: {source_count.error}")

        # Create aggregated result
        aggregated = AggregatedGameCount(
            config_key=config_key,
            total_games=total,
            sources=sources,
            source_details=source_details,
            last_updated=time.time(),
            errors=errors,
        )

        # Update cache
        async with self._lock:
            self._cache[config_key] = aggregated

        return aggregated

    async def get_all_configs_counts(
        self,
        include_remote: bool | None = None,
        include_s3: bool | None = None,
        include_owc: bool | None = None,
    ) -> dict[str, AggregatedGameCount]:
        """Get game counts for all 12 canonical configurations.

        Returns:
            Dict mapping config_key to AggregatedGameCount
        """
        from app.utils.game_discovery import ALL_BOARD_TYPES, ALL_PLAYER_COUNTS

        # Query all configs concurrently
        tasks = []
        for board_type in ALL_BOARD_TYPES:
            for num_players in ALL_PLAYER_COUNTS:
                tasks.append(
                    self.get_total_games(
                        board_type, num_players,
                        include_remote=include_remote,
                        include_s3=include_s3,
                        include_owc=include_owc,
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        counts: dict[str, AggregatedGameCount] = {}
        idx = 0
        for board_type in ALL_BOARD_TYPES:
            for num_players in ALL_PLAYER_COUNTS:
                config_key = f"{board_type}_{num_players}p"
                result = results[idx]
                if isinstance(result, Exception):
                    logger.warning(f"Error getting counts for {config_key}: {result}")
                    counts[config_key] = AggregatedGameCount(
                        config_key=config_key,
                        total_games=0,
                        sources={},
                        last_updated=time.time(),
                        errors=[str(result)],
                    )
                else:
                    counts[config_key] = result
                idx += 1

        return counts

    async def refresh_all_caches(self) -> None:
        """Force refresh of all cached counts."""
        # Clear all caches
        async with self._lock:
            self._cache.clear()
            for source in self._source_caches:
                self._source_caches[source].clear()

        # Re-query all configs
        await self.get_all_configs_counts()

    def get_cached(self, config_key: str) -> AggregatedGameCount | None:
        """Get cached count for a config (no network calls).

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Cached AggregatedGameCount or None if not cached
        """
        return self._cache.get(config_key)

    def clear_cache(self, config_key: str | None = None) -> None:
        """Clear cached counts.

        Args:
            config_key: Specific config to clear, or None for all
        """
        if config_key:
            self._cache.pop(config_key, None)
            for source in self._source_caches:
                self._source_caches[source].pop(config_key, None)
        else:
            self._cache.clear()
            for source in self._source_caches:
                self._source_caches[source].clear()

    # =========================================================================
    # Event Integration
    # =========================================================================

    def wire_to_events(self) -> None:
        """Subscribe to data change events for automatic cache invalidation.

        Subscribes to:
        - DATA_SYNC_COMPLETED: New data synced from other nodes
        - SELFPLAY_COMPLETE: New games generated locally
        - GAMES_UPLOADED_TO_S3: New games uploaded to S3
        - GAMES_UPLOADED_TO_OWC: New games uploaded to OWC
        """
        try:
            from app.coordination.event_router import get_event_router
            from app.coordination.data_events import DataEventType

            router = get_event_router()

            # Subscribe to events that affect game counts
            events_to_watch = [
                DataEventType.DATA_SYNC_COMPLETED,
                DataEventType.SELFPLAY_COMPLETE,
            ]

            # Add new events if they exist
            for event_name in ["GAMES_UPLOADED_TO_S3", "GAMES_UPLOADED_TO_OWC"]:
                if hasattr(DataEventType, event_name):
                    events_to_watch.append(getattr(DataEventType, event_name))

            for event_type in events_to_watch:
                router.subscribe(event_type, self._on_data_changed)

            logger.info("[UnifiedGameAggregator] Wired to data change events")

        except ImportError:
            logger.debug("[UnifiedGameAggregator] Event router not available")
        except Exception as e:
            logger.warning(f"[UnifiedGameAggregator] Failed to wire events: {e}")

    def _on_data_changed(self, event: object) -> None:
        """Handle data change events by clearing relevant caches.

        Args:
            event: Event object (DataEvent or similar)
        """
        try:
            # Extract config_key if present
            config_key = getattr(event, "config_key", None)
            event_type = getattr(event, "event_type", type(event).__name__)

            if config_key:
                logger.debug(
                    f"[UnifiedGameAggregator] Cache invalidated for {config_key} "
                    f"by {event_type}"
                )
                self.clear_cache(config_key)
            else:
                logger.debug(
                    f"[UnifiedGameAggregator] All caches invalidated by {event_type}"
                )
                self.clear_cache()

        except Exception as e:
            logger.warning(f"[UnifiedGameAggregator] Error handling event: {e}")


# =============================================================================
# Singleton Management
# =============================================================================

_aggregator_instance: UnifiedGameAggregator | None = None
_aggregator_lock = asyncio.Lock()


def get_unified_game_aggregator(
    config: GameSourceConfig | None = None,
) -> UnifiedGameAggregator:
    """Get or create the singleton UnifiedGameAggregator instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        UnifiedGameAggregator instance
    """
    global _aggregator_instance

    if _aggregator_instance is None:
        _aggregator_instance = UnifiedGameAggregator(config)
        _aggregator_instance.wire_to_events()

    return _aggregator_instance


def reset_unified_game_aggregator() -> None:
    """Reset the singleton instance (for testing)."""
    global _aggregator_instance
    _aggregator_instance = None


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Unified Game Aggregator")
        parser.add_argument("--board-type", type=str, help="Board type")
        parser.add_argument("--num-players", type=int, help="Number of players")
        parser.add_argument("--all", action="store_true", help="Query all configs")
        parser.add_argument("--local-only", action="store_true", help="Local only")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

        config = (
            GameSourceConfig.local_only()
            if args.local_only
            else GameSourceConfig.all_sources()
        )
        aggregator = UnifiedGameAggregator(config)

        if args.all:
            print("\nQuerying all configurations...")
            counts = await aggregator.get_all_configs_counts()

            print("\nGame Counts by Configuration:")
            print("=" * 70)

            for config_key in sorted(counts.keys()):
                result = counts[config_key]
                print(f"\n{config_key}:")
                print(f"  Total: {result.total_games:,} games")
                for source, count in result.sources.items():
                    print(f"    {source}: {count:,}")
                if result.errors:
                    print(f"  Errors: {result.errors}")

        elif args.board_type and args.num_players:
            result = await aggregator.get_total_games(
                args.board_type, args.num_players
            )

            print(f"\nGame Counts for {result.config_key}:")
            print("=" * 50)
            print(f"Total: {result.total_games:,} games")
            print("\nBy Source:")
            for source, count in result.sources.items():
                print(f"  {source}: {count:,}")

            if args.verbose:
                print("\nSource Details:")
                for detail in result.source_details:
                    print(f"  {detail.source_name}:")
                    print(f"    Count: {detail.game_count:,}")
                    print(f"    Updated: {time.ctime(detail.last_updated)}")
                    if detail.error:
                        print(f"    Error: {detail.error}")
                    if detail.details:
                        print(f"    Details: {detail.details}")

            if result.errors:
                print(f"\nErrors: {result.errors}")
        else:
            parser.print_help()

    asyncio.run(main())
