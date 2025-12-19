"""DataCatalog - Cluster-wide training data discovery service.

This module provides a centralized service for discovering and selecting
high-quality training data from across the cluster. It integrates with
the unified manifest to enable:

1. Automatic discovery of synced data on any training node
2. Quality-aware data selection for training
3. Cross-node data availability tracking
4. Centralized view of cluster-wide training resources

Usage:
    from app.distributed.data_catalog import DataCatalog, get_data_catalog

    # Get singleton instance
    catalog = get_data_catalog()

    # Discover available data
    sources = catalog.discover_data_sources()

    # Get high-quality games for training
    games = catalog.get_training_data(
        min_quality=0.5,
        board_type="square8",
        num_players=2,
        limit=10000,
    )

    # Get paths to synced databases
    db_paths = catalog.get_synced_db_paths(host_filter="lambda-*")
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Path setup
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNC_DIR = AI_SERVICE_ROOT / "data" / "games" / "synced"
DEFAULT_MANIFEST_PATH = AI_SERVICE_ROOT / "data" / "data_manifest.db"

# Try to import unified manifest
try:
    from app.distributed.unified_manifest import (
        DataManifest,
        GameQualityMetadata,
    )
    HAS_UNIFIED_MANIFEST = True
except ImportError:
    HAS_UNIFIED_MANIFEST = False
    DataManifest = None
    GameQualityMetadata = None


@dataclass
class DataSource:
    """A source of training data."""
    name: str
    path: Path
    source_type: str  # "local", "synced", "nfs"
    host_origin: str  # Original host the data came from
    game_count: int = 0
    total_size_bytes: int = 0
    last_modified: float = 0.0
    avg_quality_score: float = 0.0
    is_available: bool = True
    board_types: Set[str] = field(default_factory=set)
    player_counts: Set[int] = field(default_factory=set)


@dataclass
class CatalogStats:
    """Statistics for the data catalog."""
    total_sources: int = 0
    total_games: int = 0
    total_size_bytes: int = 0
    sources_by_type: Dict[str, int] = field(default_factory=dict)
    sources_by_host: Dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    high_quality_games: int = 0  # Quality >= 0.5
    board_type_distribution: Dict[str, int] = field(default_factory=dict)


class DataCatalog:
    """Service for discovering and selecting training data across the cluster.

    The DataCatalog provides:
    - Automatic discovery of local and synced training data
    - Quality-aware data selection for training
    - Cross-node data availability tracking
    - Integration with unified manifest for deduplication
    """

    def __init__(
        self,
        sync_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        local_game_dirs: Optional[List[Path]] = None,
        node_id: Optional[str] = None,
    ):
        """Initialize the data catalog.

        Args:
            sync_dir: Directory containing synced data (default: data/games/synced)
            manifest_path: Path to unified manifest database
            local_game_dirs: Additional local directories to scan for games
            node_id: Identifier for this node (default: hostname)
        """
        self.sync_dir = sync_dir or DEFAULT_SYNC_DIR
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        self.local_game_dirs = local_game_dirs or [AI_SERVICE_ROOT / "data" / "games"]
        self.node_id = node_id or socket.gethostname()

        # Initialize manifest if available
        self._manifest: Optional[DataManifest] = None
        if HAS_UNIFIED_MANIFEST and self.manifest_path.exists():
            try:
                self._manifest = DataManifest(self.manifest_path)
                logger.info(f"DataCatalog initialized with manifest at {self.manifest_path}")
            except Exception as e:
                logger.warning(f"Could not initialize manifest: {e}")

        # Cache for discovered sources
        self._sources: Dict[str, DataSource] = {}
        self._last_discovery: float = 0.0
        self._discovery_interval: float = 300.0  # 5 minutes

    def discover_data_sources(self, force: bool = False) -> List[DataSource]:
        """Discover all available data sources.

        Args:
            force: Force re-discovery even if cache is fresh

        Returns:
            List of discovered DataSource objects
        """
        now = time.time()
        if not force and (now - self._last_discovery) < self._discovery_interval:
            return list(self._sources.values())

        self._sources.clear()

        # Discover local game directories
        for game_dir in self.local_game_dirs:
            self._discover_directory(game_dir, source_type="local", host_origin=self.node_id)

        # Discover synced data
        if self.sync_dir.exists():
            for host_dir in self.sync_dir.iterdir():
                if host_dir.is_dir():
                    self._discover_directory(
                        host_dir,
                        source_type="synced",
                        host_origin=host_dir.name,
                    )

        # Check for NFS shared storage
        nfs_path = Path("/shared/ringrift/training_data")
        if nfs_path.exists():
            self._discover_directory(nfs_path, source_type="nfs", host_origin="shared")

        self._last_discovery = now
        logger.info(f"Discovered {len(self._sources)} data sources")

        return list(self._sources.values())

    def _discover_directory(
        self,
        path: Path,
        source_type: str,
        host_origin: str,
    ) -> None:
        """Discover game databases in a directory.

        Args:
            path: Directory to scan
            source_type: Type of source (local, synced, nfs)
            host_origin: Original host the data came from
        """
        if not path.exists():
            return

        for db_file in path.glob("*.db"):
            try:
                source = self._analyze_database(db_file, source_type, host_origin)
                if source and source.game_count > 0:
                    self._sources[str(db_file)] = source
            except Exception as e:
                logger.debug(f"Failed to analyze {db_file}: {e}")

    def _analyze_database(
        self,
        db_path: Path,
        source_type: str,
        host_origin: str,
    ) -> Optional[DataSource]:
        """Analyze a game database and return source info.

        Args:
            db_path: Path to the database
            source_type: Type of source
            host_origin: Original host

        Returns:
            DataSource object or None if analysis fails
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get game count
            cursor.execute("SELECT COUNT(*) FROM games")
            game_count = cursor.fetchone()[0]

            if game_count == 0:
                conn.close()
                return None

            # Get board type and player count distribution
            board_types: Set[str] = set()
            player_counts: Set[int] = set()
            try:
                cursor.execute("SELECT DISTINCT board_type FROM games WHERE board_type IS NOT NULL")
                board_types = {row[0] for row in cursor.fetchall()}
                cursor.execute("SELECT DISTINCT num_players FROM games WHERE num_players IS NOT NULL")
                player_counts = {row[0] for row in cursor.fetchall()}
            except sqlite3.OperationalError:
                pass  # Columns may not exist

            conn.close()

            # Get file stats
            stat = db_path.stat()

            return DataSource(
                name=db_path.name,
                path=db_path,
                source_type=source_type,
                host_origin=host_origin,
                game_count=game_count,
                total_size_bytes=stat.st_size,
                last_modified=stat.st_mtime,
                is_available=True,
                board_types=board_types,
                player_counts=player_counts,
            )

        except sqlite3.Error as e:
            logger.debug(f"SQLite error analyzing {db_path}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error analyzing {db_path}: {e}")
            return None

    def get_training_data(
        self,
        min_quality: float = 0.0,
        max_games: int = 100000,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
        prefer_recent: bool = True,
        prefer_high_elo: bool = True,
    ) -> List[GameQualityMetadata]:
        """Get high-quality games for training.

        Args:
            min_quality: Minimum quality score threshold
            max_games: Maximum number of games to return
            board_type: Filter by board type
            num_players: Filter by player count
            prefer_recent: Prefer recently created games
            prefer_high_elo: Prefer games from high-Elo players

        Returns:
            List of GameQualityMetadata for selected games
        """
        if not self._manifest or not HAS_UNIFIED_MANIFEST:
            logger.warning("Manifest not available, returning empty list")
            return []

        return self._manifest.get_high_quality_games(
            min_quality_score=min_quality,
            limit=max_games,
            board_type=board_type,
            num_players=num_players,
        )

    def get_synced_db_paths(
        self,
        host_filter: Optional[str] = None,
        min_games: int = 0,
        board_type: Optional[str] = None,
    ) -> List[Path]:
        """Get paths to synced game databases.

        Args:
            host_filter: Filter by host name pattern (supports * wildcard)
            min_games: Minimum number of games in database
            board_type: Filter by board type

        Returns:
            List of paths to matching databases
        """
        # Ensure discovery is fresh
        self.discover_data_sources()

        paths = []
        for source in self._sources.values():
            # Apply host filter
            if host_filter:
                if "*" in host_filter:
                    pattern = host_filter.replace("*", "")
                    if not source.host_origin.startswith(pattern) and not source.host_origin.endswith(pattern):
                        continue
                elif source.host_origin != host_filter:
                    continue

            # Apply game count filter
            if source.game_count < min_games:
                continue

            # Apply board type filter
            if board_type and board_type not in source.board_types:
                continue

            paths.append(source.path)

        return paths

    def get_all_training_paths(
        self,
        include_local: bool = True,
        include_synced: bool = True,
        include_nfs: bool = True,
    ) -> List[Path]:
        """Get all available paths for training data.

        Args:
            include_local: Include local game databases
            include_synced: Include synced databases
            include_nfs: Include NFS shared databases

        Returns:
            List of paths to all matching databases
        """
        self.discover_data_sources()

        paths = []
        for source in self._sources.values():
            if source.source_type == "local" and not include_local:
                continue
            if source.source_type == "synced" and not include_synced:
                continue
            if source.source_type == "nfs" and not include_nfs:
                continue

            paths.append(source.path)

        return paths

    def get_stats(self) -> CatalogStats:
        """Get statistics about available training data.

        Returns:
            CatalogStats object with catalog statistics
        """
        self.discover_data_sources()

        stats = CatalogStats()
        stats.total_sources = len(self._sources)

        for source in self._sources.values():
            stats.total_games += source.game_count
            stats.total_size_bytes += source.total_size_bytes

            # Count by type
            stats.sources_by_type[source.source_type] = (
                stats.sources_by_type.get(source.source_type, 0) + 1
            )

            # Count by host
            stats.sources_by_host[source.host_origin] = (
                stats.sources_by_host.get(source.host_origin, 0) + source.game_count
            )

            # Count by board type
            for bt in source.board_types:
                stats.board_type_distribution[bt] = (
                    stats.board_type_distribution.get(bt, 0) + source.game_count
                )

        # Get quality stats from manifest
        if self._manifest:
            try:
                quality_dist = self._manifest.get_quality_distribution()
                stats.avg_quality_score = quality_dist.get("avg_quality_score", 0.0)
                # Estimate high quality games (this is approximate)
                if quality_dist.get("total_games", 0) > 0:
                    stats.high_quality_games = int(
                        quality_dist["total_games"] * quality_dist.get("decisive_rate", 0.5)
                    )
            except Exception as e:
                logger.debug(f"Failed to get quality stats: {e}")

        return stats

    def get_sources_by_quality(
        self,
        min_quality: float = 0.5,
        limit: int = 10,
    ) -> List[DataSource]:
        """Get data sources sorted by average quality score.

        Args:
            min_quality: Minimum average quality score
            limit: Maximum number of sources to return

        Returns:
            List of DataSource objects sorted by quality
        """
        self.discover_data_sources()

        # Filter and sort by quality
        qualified = [s for s in self._sources.values() if s.avg_quality_score >= min_quality]
        qualified.sort(key=lambda s: s.avg_quality_score, reverse=True)

        return qualified[:limit]

    def refresh(self) -> None:
        """Force refresh of the data catalog."""
        self.discover_data_sources(force=True)

    def get_recommended_training_sources(
        self,
        target_games: int = 50000,
        board_type: Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> List[Path]:
        """Get recommended data sources for training.

        This method selects the best sources based on:
        1. Quality scores
        2. Game count
        3. Recency
        4. Diversity (different hosts)

        Args:
            target_games: Target total number of games
            board_type: Filter by board type
            num_players: Filter by player count

        Returns:
            List of recommended database paths
        """
        self.discover_data_sources()

        # Filter sources
        candidates = []
        for source in self._sources.values():
            if board_type and board_type not in source.board_types:
                continue
            if num_players and num_players not in source.player_counts:
                continue
            candidates.append(source)

        if not candidates:
            return []

        # Sort by quality and recency
        candidates.sort(
            key=lambda s: (s.avg_quality_score, s.last_modified),
            reverse=True,
        )

        # Select sources to reach target game count, preferring diversity
        selected: List[DataSource] = []
        selected_hosts: Set[str] = set()
        total_games = 0

        # First pass: one source per host
        for source in candidates:
            if total_games >= target_games:
                break
            if source.host_origin not in selected_hosts:
                selected.append(source)
                selected_hosts.add(source.host_origin)
                total_games += source.game_count

        # Second pass: fill remaining quota
        for source in candidates:
            if total_games >= target_games:
                break
            if source not in selected:
                selected.append(source)
                total_games += source.game_count

        return [s.path for s in selected]


# Singleton instance
_catalog_instance: Optional[DataCatalog] = None


def get_data_catalog(
    sync_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> DataCatalog:
    """Get the singleton DataCatalog instance.

    Args:
        sync_dir: Override sync directory
        manifest_path: Override manifest path

    Returns:
        DataCatalog singleton instance
    """
    global _catalog_instance

    if _catalog_instance is None:
        _catalog_instance = DataCatalog(
            sync_dir=sync_dir,
            manifest_path=manifest_path,
        )

    return _catalog_instance


def reset_data_catalog() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _catalog_instance
    _catalog_instance = None
