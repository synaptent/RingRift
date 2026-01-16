"""DataCatalog - Cluster-wide training data discovery service (December 2025).

This module provides a centralized service for discovering and selecting
high-quality training data from across the cluster. It integrates with
the unified manifest to enable:

1. Automatic discovery of synced data on any training node
2. Quality-aware data selection for training
3. Cross-node data availability tracking
4. Centralized view of cluster-wide training resources

Architecture Note (December 2025):
    This module consolidates two complementary components:
    - DataManifest (unified_manifest.py): Tracks synced game IDs and host states
    - DataCatalog (this file): Discovery and selection of training data

    The UnifiedDataRegistry provides a single facade to access both:
    - Use get_data_registry() for unified access
    - Use get_data_catalog() for catalog-only access
    - Use get_manifest() for manifest-only access

Usage:
    from app.distributed.data_catalog import (
        DataCatalog,
        get_data_catalog,
        UnifiedDataRegistry,
        get_data_registry,
    )

    # Get unified registry (recommended)
    registry = get_data_registry()

    # Access catalog features
    sources = registry.catalog.discover_data_sources()

    # Access manifest features
    is_synced = registry.manifest.is_game_synced("game_123")

    # Or use direct accessors
    catalog = get_data_catalog()
    sources = catalog.discover_data_sources()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)

# Path setup
from app.config.ports import get_local_p2p_url
from app.utils.paths import DATA_DIR, GAMES_DIR

try:
    from app.distributed.storage_provider import get_storage_provider
    HAS_STORAGE_PROVIDER = True
except ImportError:
    HAS_STORAGE_PROVIDER = False
DEFAULT_SYNC_DIR = GAMES_DIR / "synced"
DEFAULT_OWC_IMPORTS_DIR = GAMES_DIR / "owc_imports"  # Dec 30, 2025
DEFAULT_MANIFEST_PATH = DATA_DIR / "data_manifest.db"

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

# Unified game discovery - finds all game databases across all storage patterns
try:
    from app.utils.game_discovery import GameDiscovery, DatabaseInfo
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None
    DatabaseInfo = None


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
    board_types: set[str] = field(default_factory=set)
    player_counts: set[int] = field(default_factory=set)


@dataclass
class NPZDataSource:
    """A source of NPZ training files (Phase 7: NPZ tracking).

    NPZ files contain pre-processed training data exported from game databases.
    Tracking them helps training nodes find available training data.
    """
    path: Path
    board_type: str | None = None
    num_players: int | None = None
    sample_count: int = 0
    total_size_bytes: int = 0
    created_at: float = 0.0
    source_databases: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    host_origin: str = ""
    is_available: bool = True

    @property
    def config_key(self) -> str | None:
        """Get config key (e.g., 'hex8_2p')."""
        if self.board_type and self.num_players:
            return f"{self.board_type}_{self.num_players}p"
        return None

    @property
    def age_hours(self) -> float:
        """Get age of the NPZ file in hours."""
        if self.created_at <= 0:
            return float("inf")
        return (time.time() - self.created_at) / 3600


@dataclass
class CatalogStats:
    """Statistics for the data catalog."""
    total_sources: int = 0
    total_games: int = 0
    total_size_bytes: int = 0
    sources_by_type: dict[str, int] = field(default_factory=dict)
    sources_by_host: dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    high_quality_games: int = 0  # Quality >= 0.5
    board_type_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainingSourceRecommendation:
    """A recommended training data source with scoring details."""
    path: Path
    game_count: int
    board_type: str | None
    num_players: int | None
    score: float
    host_origin: str
    reasons: list[str] = field(default_factory=list)
    is_primary: bool = False  # True if this is a primary recommendation


@dataclass
class TrainingSourceSuggestion:
    """Complete training source suggestion with recommendations and summary."""
    board_type: str
    num_players: int
    recommendations: list[TrainingSourceRecommendation]
    total_games_available: int
    games_needed: int
    coverage_percent: float
    warnings: list[str] = field(default_factory=list)


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
        sync_dir: Path | None = None,
        manifest_path: Path | None = None,
        local_game_dirs: list[Path] | None = None,
        node_id: str | None = None,
    ):
        """Initialize the data catalog.

        Args:
            sync_dir: Directory containing synced data (default: data/games/synced)
            manifest_path: Path to unified manifest database
            local_game_dirs: Additional local directories to scan for games
            node_id: Identifier for this node (default: hostname)
        """
        self._provider = get_storage_provider() if HAS_STORAGE_PROVIDER else None

        if sync_dir is None:
            if self._provider:
                candidate = self._provider.selfplay_dir / "synced"
                self.sync_dir = candidate if candidate.exists() else DEFAULT_SYNC_DIR
            else:
                self.sync_dir = DEFAULT_SYNC_DIR
        else:
            self.sync_dir = sync_dir

        # Phase 5.2 Dec 29, 2025: Ensure sync directory exists
        # Previously discovery failed silently if directory was missing
        try:
            self.sync_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[DataCatalog] Could not create sync_dir {self.sync_dir}: {e}")

        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH

        if local_game_dirs is None:
            dirs: list[Path] = []
            if self._provider:
                dirs.append(self._provider.selfplay_dir)
            dirs.append(GAMES_DIR)
            self.local_game_dirs = []
            for d in dirs:
                if d not in self.local_game_dirs:
                    self.local_game_dirs.append(d)
        else:
            self.local_game_dirs = local_game_dirs
        self.node_id = node_id or socket.gethostname()

        # Initialize manifest if available
        self._manifest: DataManifest | None = None
        if HAS_UNIFIED_MANIFEST and self.manifest_path.exists():
            try:
                self._manifest = DataManifest(self.manifest_path)
                logger.info(f"DataCatalog initialized with manifest at {self.manifest_path}")
            except (OSError, sqlite3.Error) as e:
                logger.warning(f"Could not initialize manifest: {e}")

        # Cache for discovered sources
        self._sources: dict[str, DataSource] = {}
        self._last_discovery: float = 0.0
        self._discovery_interval: float = 300.0  # 5 minutes

    def discover_data_sources(self, force: bool = False) -> list[DataSource]:
        """Discover all available data sources.

        Uses unified GameDiscovery if available for comprehensive discovery
        across all storage patterns (central DBs, selfplay dirs, p2p dirs, etc.).

        Args:
            force: Force re-discovery even if cache is fresh

        Returns:
            List of discovered DataSource objects
        """
        now = time.time()
        if not force and (now - self._last_discovery) < self._discovery_interval:
            return list(self._sources.values())

        self._sources.clear()

        # Use unified GameDiscovery if available (preferred method)
        if HAS_GAME_DISCOVERY:
            self._discover_via_game_discovery()
        else:
            # Fallback to manual discovery
            self._discover_manually()

        self._last_discovery = now
        logger.info(f"Discovered {len(self._sources)} data sources")

        return list(self._sources.values())

    def _discover_via_game_discovery(self) -> None:
        """Discover data sources using unified GameDiscovery.

        This method finds ALL databases across all storage patterns:
        - Central databases (selfplay.db, jsonl_aggregated.db)
        - Per-config databases
        - Tournament databases
        - Canonical selfplay databases
        - P2P selfplay databases
        - And more...
        """
        discovery = GameDiscovery()
        local_source_type = "nfs" if self._provider and self._provider.has_shared_storage else "local"

        for db_info in discovery.find_all_databases():
            try:
                source = DataSource(
                    name=db_info.path.name,
                    path=db_info.path,
                    source_type=local_source_type,
                    host_origin=self.node_id,
                    game_count=db_info.game_count,
                    total_size_bytes=db_info.path.stat().st_size if db_info.path.exists() else 0,
                    last_modified=db_info.path.stat().st_mtime if db_info.path.exists() else 0.0,
                    is_available=True,
                    board_types={db_info.board_type} if db_info.board_type else set(),
                    player_counts={db_info.num_players} if db_info.num_players else set(),
                )
                if source.game_count > 0:
                    self._sources[str(db_info.path)] = source
            except (OSError, ValueError) as e:
                logger.debug(f"Failed to create source from {db_info.path}: {e}")

        # Also discover synced data from other hosts
        if self.sync_dir.exists():
            for host_dir in self.sync_dir.iterdir():
                if host_dir.is_dir():
                    self._discover_directory(
                        host_dir,
                        source_type="synced",
                        host_origin=host_dir.name,
                    )

    def _discover_manually(self) -> None:
        """Fallback manual discovery when GameDiscovery is not available."""
        local_source_type = "nfs" if self._provider and self._provider.has_shared_storage else "local"
        for game_dir in self.local_game_dirs:
            self._discover_directory(game_dir, source_type=local_source_type, host_origin=self.node_id)

        # Discover synced data
        if self.sync_dir.exists():
            for host_dir in self.sync_dir.iterdir():
                if host_dir.is_dir():
                    self._discover_directory(
                        host_dir,
                        source_type="synced",
                        host_origin=host_dir.name,
                    )

        # Discover OWC imports (Dec 30, 2025)
        owc_imports_dir = DEFAULT_OWC_IMPORTS_DIR
        if owc_imports_dir.exists():
            self._discover_directory(
                owc_imports_dir,
                source_type="owc_import",
                host_origin="mac-studio",
            )

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
            except (OSError, sqlite3.Error) as e:
                logger.debug(f"Failed to analyze {db_file}: {e}")

    def _analyze_database(
        self,
        db_path: Path,
        source_type: str,
        host_origin: str,
    ) -> DataSource | None:
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
            board_types: set[str] = set()
            player_counts: set[int] = set()
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
        except OSError as e:
            logger.debug(f"Error analyzing {db_path}: {e}")
            return None

    def get_training_data(
        self,
        min_quality: float = 0.0,
        max_games: int = 100000,
        board_type: str | None = None,
        num_players: int | None = None,
        prefer_recent: bool = True,
        prefer_high_elo: bool = True,
    ) -> list[GameQualityMetadata]:
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
            prefer_recent=prefer_recent,
            prefer_high_elo=prefer_high_elo,
        )

    def get_synced_db_paths(
        self,
        host_filter: str | None = None,
        min_games: int = 0,
        board_type: str | None = None,
    ) -> list[Path]:
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
    ) -> list[Path]:
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
            except (OSError, sqlite3.Error, KeyError) as e:
                logger.debug(f"Failed to get quality stats: {e}")

        return stats

    def get_pending_sample_count(
        self,
        samples_per_game: int = 30,
    ) -> int:
        """Get estimated count of training samples pending processing.

        Training samples are individual moves (state-action pairs) from games.
        Each game typically has 20-60 moves depending on board size and game length.

        This is used by the idle resource daemon to estimate training backlog
        and implement backpressure when selfplay outpaces training.

        Args:
            samples_per_game: Average training samples per game (default: 30)

        Returns:
            Estimated total training samples available
        """
        stats = self.get_stats()
        return stats.total_games * samples_per_game

    def get_sources_by_quality(
        self,
        min_quality: float = 0.5,
        limit: int = 10,
    ) -> list[DataSource]:
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
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[Path]:
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
        selected: list[DataSource] = []
        selected_hosts: set[str] = set()
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

    def suggest_best_training_sources(
        self,
        board_type: str,
        num_players: int,
        target_games: int = 50000,
        min_quality: float = 0.0,
    ) -> TrainingSourceSuggestion:
        """Get intelligent training source suggestions with scoring and explanations.

        This method provides detailed recommendations for which databases to use
        for training, including:
        - Scoring based on quality, recency, and game count
        - Reasons why each source was selected
        - Warnings about potential issues
        - Coverage analysis

        Args:
            board_type: Board type to train (e.g., 'hex8', 'square8')
            num_players: Number of players (2, 3, or 4)
            target_games: Target number of games for training
            min_quality: Minimum quality score threshold

        Returns:
            TrainingSourceSuggestion with ranked recommendations
        """
        self.discover_data_sources()

        # Find matching sources
        candidates: list[tuple[DataSource, float, list[str]]] = []
        now = time.time()

        for source in self._sources.values():
            # Check if source has matching config
            if board_type not in source.board_types:
                continue
            if num_players not in source.player_counts:
                continue
            if source.avg_quality_score < min_quality:
                continue
            if source.game_count == 0:
                continue

            # Calculate score and reasons
            score = 0.0
            reasons = []

            # Quality component (0-40 points)
            quality_score = source.avg_quality_score * 40
            score += quality_score
            if source.avg_quality_score > 0.7:
                reasons.append(f"High quality ({source.avg_quality_score:.2f})")
            elif source.avg_quality_score > 0.3:
                reasons.append(f"Medium quality ({source.avg_quality_score:.2f})")

            # Game count component (0-30 points, logarithmic)
            import math
            if source.game_count > 0:
                count_score = min(30, math.log10(source.game_count) * 10)
                score += count_score
                if source.game_count > 10000:
                    reasons.append(f"Large dataset ({source.game_count:,} games)")
                elif source.game_count > 1000:
                    reasons.append(f"Good dataset ({source.game_count:,} games)")

            # Recency component (0-20 points)
            age_days = (now - source.last_modified) / 86400
            if age_days < 1:
                score += 20
                reasons.append("Very recent (<1 day)")
            elif age_days < 7:
                score += 15
                reasons.append("Recent (<7 days)")
            elif age_days < 30:
                score += 10
            elif age_days < 90:
                score += 5

            # Source type bonus (0-10 points)
            if source.source_type == "local":
                score += 10
                reasons.append("Local source (fast access)")
            elif source.source_type == "nfs":
                score += 8
                reasons.append("NFS shared storage")
            elif source.source_type == "synced":
                score += 5

            candidates.append((source, score, reasons))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Build recommendations
        recommendations = []
        total_available = 0
        selected_games = 0

        for i, (source, score, reasons) in enumerate(candidates):
            total_available += source.game_count

            is_primary = i < 3 or selected_games < target_games

            rec = TrainingSourceRecommendation(
                path=source.path,
                game_count=source.game_count,
                board_type=board_type,
                num_players=num_players,
                score=score,
                host_origin=source.host_origin,
                reasons=reasons,
                is_primary=is_primary,
            )
            recommendations.append(rec)

            if is_primary:
                selected_games += source.game_count

        # Generate warnings
        warnings = []
        if total_available == 0:
            warnings.append(f"No data found for {board_type}_{num_players}p")
        elif total_available < target_games:
            warnings.append(
                f"Only {total_available:,} games available "
                f"(target: {target_games:,})"
            )

        if len(recommendations) == 0:
            warnings.append("No suitable training sources found")
        elif len(recommendations) == 1:
            warnings.append("Only one source available - consider data augmentation")

        # Calculate coverage
        coverage = min(100.0, (total_available / target_games) * 100) if target_games > 0 else 100.0

        return TrainingSourceSuggestion(
            board_type=board_type,
            num_players=num_players,
            recommendations=recommendations,
            total_games_available=total_available,
            games_needed=target_games,
            coverage_percent=coverage,
            warnings=warnings,
        )

    # =========================================================================
    # NPZ File Discovery (Phase 7: NPZ Tracking - December 2025)
    # =========================================================================

    def discover_npz_files(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        min_samples: int = 0,
        max_age_hours: float | None = None,
    ) -> list[NPZDataSource]:
        """Discover NPZ training files across all storage locations.

        Args:
            board_type: Filter by board type (e.g., 'hex8', 'square8')
            num_players: Filter by player count (e.g., 2, 3, 4)
            min_samples: Minimum sample count threshold
            max_age_hours: Maximum age in hours (None for no limit)

        Returns:
            List of NPZDataSource objects for matching NPZ files
        """
        npz_sources: list[NPZDataSource] = []

        # Search directories
        search_dirs = [
            DATA_DIR / "training",
            DATA_DIR / "exports",
            GAMES_DIR.parent / "training",
        ]

        # Add NFS path if available
        if self._provider and self._provider.has_shared_storage:
            nfs_training = Path("/lambda/nfs/RingRift/ai-service/data/training")
            if nfs_training.exists():
                search_dirs.append(nfs_training)

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for npz_path in search_dir.glob("**/*.npz"):
                try:
                    source = self._analyze_npz_file(npz_path)
                    if source is None:
                        continue

                    # Apply filters
                    if board_type and source.board_type != board_type:
                        continue
                    if num_players and source.num_players != num_players:
                        continue
                    if source.sample_count < min_samples:
                        continue
                    if max_age_hours and source.age_hours > max_age_hours:
                        continue

                    npz_sources.append(source)

                except (OSError, ValueError, zipfile.BadZipFile, EOFError) as e:
                    logger.debug(f"Failed to analyze NPZ {npz_path}: {e}")

        # Sort by recency (newest first)
        npz_sources.sort(key=lambda s: s.created_at, reverse=True)

        logger.debug(f"Discovered {len(npz_sources)} NPZ files")
        return npz_sources

    def _analyze_npz_file(self, npz_path: Path) -> NPZDataSource | None:
        """Analyze an NPZ file and return source info.

        Args:
            npz_path: Path to the NPZ file

        Returns:
            NPZDataSource object or None if analysis fails
        """
        import numpy as np

        try:
            stat = npz_path.stat()

            # Parse board type and num_players from filename
            # Common patterns: hex8_2p.npz, square8_3p_training.npz
            name = npz_path.stem.lower()
            board_type = None
            num_players = None

            # Extract board type
            for bt in ["hex8", "hexagonal", "square8", "square19"]:
                if bt in name:
                    board_type = bt
                    break

            # Extract player count
            for players in [2, 3, 4]:
                if f"{players}p" in name:
                    num_players = players
                    break

            # Estimate sample count from file size
            # Typical: ~200 bytes per sample for compressed NPZ
            estimated_samples = stat.st_size // 200

            # Try to get actual sample count if file is small enough
            # Dec 31, 2025: Added features/values/policy_indices as valid keys
            # The training data format uses these instead of states/policy
            sample_count = estimated_samples
            if stat.st_size < 100 * 1024 * 1024:  # < 100MB
                try:
                    with np.load(npz_path, allow_pickle=False) as data:
                        # Check for various sample count indicators
                        # Priority: features > values > policy_indices > policy > states
                        if "features" in data:
                            sample_count = len(data["features"])
                        elif "values" in data:
                            sample_count = len(data["values"])
                        elif "policy_indices" in data:
                            sample_count = len(data["policy_indices"])
                        elif "policy" in data:
                            sample_count = len(data["policy"])
                        elif "states" in data:
                            sample_count = len(data["states"])
                except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
                    pass  # Use estimate or file is corrupted/empty

            return NPZDataSource(
                path=npz_path,
                board_type=board_type,
                num_players=num_players,
                sample_count=sample_count,
                total_size_bytes=stat.st_size,
                created_at=stat.st_mtime,
                host_origin=self.node_id,
                is_available=True,
            )

        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as e:
            logger.debug(f"Error analyzing NPZ {npz_path}: {e}")
            return None

    def get_best_npz_for_training(
        self,
        board_type: str,
        num_players: int,
        prefer_recent: bool = True,
    ) -> NPZDataSource | None:
        """Get the best NPZ file for a training configuration.

        Args:
            board_type: Board type to match
            num_players: Number of players to match
            prefer_recent: Prefer more recent files

        Returns:
            Best matching NPZDataSource or None
        """
        sources = self.discover_npz_files(
            board_type=board_type,
            num_players=num_players,
        )

        if not sources:
            return None

        if prefer_recent:
            # Already sorted by recency in discover_npz_files
            return sources[0]

        # Otherwise prefer largest (most samples)
        return max(sources, key=lambda s: s.sample_count)

    def get_npz_stats(self) -> dict[str, Any]:
        """Get statistics about available NPZ files.

        Returns:
            Dictionary with NPZ statistics
        """
        all_npz = self.discover_npz_files()

        stats: dict[str, Any] = {
            "total_files": len(all_npz),
            "total_samples": sum(s.sample_count for s in all_npz),
            "total_size_bytes": sum(s.total_size_bytes for s in all_npz),
            "by_config": {},
        }

        for source in all_npz:
            config_key = source.config_key or "unknown"
            if config_key not in stats["by_config"]:
                stats["by_config"][config_key] = {
                    "files": 0,
                    "samples": 0,
                    "newest_hours_ago": float("inf"),
                }
            entry = stats["by_config"][config_key]
            entry["files"] += 1
            entry["samples"] += source.sample_count
            entry["newest_hours_ago"] = min(
                entry["newest_hours_ago"],
                source.age_hours,
            )

        return stats

    def health_check(self) -> "HealthCheckResult":
        """Check health of the data catalog.

        Returns:
            HealthCheckResult with catalog health status
        """
        from app.coordination.protocols import HealthCheckResult

        try:
            # Discover sources to get fresh stats
            sources = self.discover_data_sources()
            stats = self.get_stats()

            # Health criteria
            has_sources = len(sources) > 0
            has_games = stats.total_games > 0
            manifest_ok = self._manifest is not None if HAS_UNIFIED_MANIFEST else True

            is_healthy = has_sources and manifest_ok

            details = {
                "total_sources": stats.total_sources,
                "total_games": stats.total_games,
                "total_size_bytes": stats.total_size_bytes,
                "sources_by_type": stats.sources_by_type,
                "manifest_available": self._manifest is not None,
                "last_discovery": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(self._last_discovery)
                ) if self._last_discovery else None,
            }

            if not has_sources:
                message = "No data sources discovered"
            elif not has_games:
                message = "Data sources found but no games available"
            else:
                message = f"Healthy: {stats.total_sources} sources, {stats.total_games:,} games"

            return HealthCheckResult(
                healthy=is_healthy,
                message=message,
                details=details,
            )
        except (OSError, sqlite3.Error) as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )


# Singleton instance
_catalog_instance: DataCatalog | None = None


def get_data_catalog(
    sync_dir: Path | None = None,
    manifest_path: Path | None = None,
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


# =============================================================================
# Unified Data Registry - Facade for DataManifest + DataCatalog (December 2025)
# =============================================================================

class UnifiedDataRegistry:
    """Unified facade for accessing DataManifest and DataCatalog.

    Provides a single entry point for all data management operations:
    - Game tracking and deduplication (via DataManifest)
    - Data discovery and selection (via DataCatalog)
    - Quality-aware data management
    - Cross-node data availability

    Usage:
        registry = get_data_registry()

        # Check if game is synced
        is_synced = registry.is_game_synced("game_123")

        # Get high-quality games for training
        games = registry.get_high_quality_games(min_quality=0.5, limit=1000)

        # Discover data sources
        sources = registry.discover_sources()

        # Get stats
        stats = registry.get_combined_stats()
    """

    def __init__(
        self,
        catalog: DataCatalog | None = None,
        manifest: DataManifest | None = None,
    ):
        """Initialize the unified registry.

        Args:
            catalog: DataCatalog instance (uses singleton if None)
            manifest: DataManifest instance (uses catalog's manifest if None)
        """
        self._catalog = catalog or get_data_catalog()
        self._manifest = manifest or self._catalog._manifest

    @property
    def catalog(self) -> DataCatalog:
        """Get the DataCatalog instance."""
        return self._catalog

    @property
    def manifest(self) -> DataManifest | None:
        """Get the DataManifest instance."""
        return self._manifest

    # Manifest operations
    def is_game_synced(self, game_id: str) -> bool:
        """Check if a game has been synced."""
        if not self._manifest:
            return False
        return self._manifest.is_game_synced(game_id)

    def mark_games_synced(
        self,
        game_ids: list[str],
        source_host: str,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Mark games as synced."""
        if not self._manifest:
            return 0
        return self._manifest.mark_synced(
            game_ids=game_ids,
            source_host=source_host,
            board_type=board_type,
            num_players=num_players,
        )

    def get_synced_game_count(self) -> int:
        """Get total synced game count."""
        if not self._manifest:
            return 0
        return self._manifest.get_synced_count()

    # Catalog operations
    def discover_sources(self, force: bool = False) -> list[DataSource]:
        """Discover available data sources."""
        return self._catalog.discover_data_sources(force=force)

    def get_synced_db_paths(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        host_filter: str | None = None,
    ) -> list[Path]:
        """Get paths to synced databases."""
        return self._catalog.get_synced_db_paths(
            board_type=board_type,
            num_players=num_players,
            host_filter=host_filter,
        )

    def get_high_quality_sources(
        self,
        min_quality: float = 0.5,
        limit: int = 10,
    ) -> list[DataSource]:
        """Get data sources sorted by quality."""
        return self._catalog.get_sources_by_quality(
            min_quality=min_quality,
            limit=limit,
        )

    def get_recommended_sources(
        self,
        target_games: int = 50000,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[Path]:
        """Get recommended data sources for training."""
        return self._catalog.get_recommended_training_sources(
            target_games=target_games,
            board_type=board_type,
            num_players=num_players,
        )

    # Combined operations
    def get_combined_stats(self) -> dict[str, Any]:
        """Get combined statistics from catalog and manifest.

        Returns:
            Dict with combined stats from both components
        """
        catalog_stats = self._catalog.get_stats()

        manifest_stats = {}
        if self._manifest:
            try:
                manifest_stats = {
                    "synced_games": self._manifest.get_synced_count(),
                    "quality_distribution": self._manifest.get_quality_distribution(),
                }
            except (OSError, sqlite3.Error) as e:
                logger.debug(f"Failed to get manifest stats: {e}")

        return {
            "catalog": {
                "total_sources": catalog_stats.total_sources,
                "total_games": catalog_stats.total_games,
                "total_size_bytes": catalog_stats.total_size_bytes,
                "avg_quality": catalog_stats.avg_quality_score,
                "high_quality_games": catalog_stats.high_quality_games,
                "sources_by_type": catalog_stats.sources_by_type,
                "board_distribution": catalog_stats.board_type_distribution,
            },
            "manifest": manifest_stats,
            "node_id": self._catalog.node_id,
        }

    def refresh(self) -> None:
        """Refresh both catalog and manifest data."""
        self._catalog.refresh()

    # =========================================================================
    # Cluster Manifest Integration (Jan 2026)
    # Receives broadcast manifests from P2P leader for unified data visibility
    # =========================================================================

    _cluster_manifest: Any = None
    _cluster_manifest_received_at: float = 0.0

    def update_from_cluster_manifest(self, manifest: Any, received_at: float | None = None) -> None:
        """Update registry with cluster manifest received from P2P leader broadcast.

        Jan 2026: Added for unified cluster data visibility.

        Args:
            manifest: ClusterDataManifest from P2P broadcast
            received_at: Timestamp when manifest was received
        """
        import time
        self._cluster_manifest = manifest
        self._cluster_manifest_received_at = received_at or time.time()
        logger.debug(
            f"[UnifiedDataRegistry] Updated cluster manifest: "
            f"{getattr(manifest, 'total_nodes', 0)} nodes, "
            f"{getattr(manifest, 'total_selfplay_games', 0)} games"
        )

    def get_cluster_manifest(self) -> Any:
        """Get the current cluster manifest (if received from leader broadcast).

        Returns:
            ClusterDataManifest or None if not yet received.
        """
        return self._cluster_manifest

    def get_cluster_status(self) -> dict[str, dict[str, int]]:
        """Get game counts per config, per source (local/cluster/owc/s3).

        Jan 2026: Added for unified cluster data visibility.

        Returns:
            Dict mapping config_key -> {local, cluster, owc, s3, total}
        """
        import time

        result: dict[str, dict[str, int]] = {}

        # Get local game counts from catalog
        try:
            local_sources = self._catalog.discover_data_sources(force=False)
            for source in local_sources:
                # DataSource has board_types (set) and player_counts (set)
                board_types = getattr(source, "board_types", set())
                player_counts = getattr(source, "player_counts", set())

                # Create config keys for all combinations
                for board_type in board_types:
                    for num_players in player_counts:
                        config_key = f"{board_type}_{num_players}p"
                        if config_key not in result:
                            result[config_key] = {"local": 0, "cluster": 0, "owc": 0, "s3": 0, "total": 0}
                        # Distribute game count evenly across configs (approximation)
                        # In practice, most DBs have one config
                        game_share = source.game_count // max(1, len(board_types) * len(player_counts))
                        result[config_key]["local"] += game_share
        except Exception as e:
            logger.debug(f"[UnifiedDataRegistry] Local source discovery failed: {e}")

        # Add cluster manifest data
        manifest = self._cluster_manifest
        if manifest:
            # Node manifests (cluster-wide data from other nodes)
            by_board = getattr(manifest, "by_board_type", {})
            for config_key, info in by_board.items():
                if config_key not in result:
                    result[config_key] = {"local": 0, "cluster": 0, "owc": 0, "s3": 0, "total": 0}
                result[config_key]["cluster"] = info.get("total_games", 0)

            # External storage (OWC + S3)
            external = getattr(manifest, "external_storage", None)
            if external:
                # OWC drive
                owc_by_config = getattr(external, "owc_games_by_config", {})
                for config_key, count in owc_by_config.items():
                    if config_key not in result:
                        result[config_key] = {"local": 0, "cluster": 0, "owc": 0, "s3": 0, "total": 0}
                    result[config_key]["owc"] = count

                # S3 bucket
                s3_by_config = getattr(external, "s3_games_by_config", {})
                for config_key, count in s3_by_config.items():
                    if config_key not in result:
                        result[config_key] = {"local": 0, "cluster": 0, "owc": 0, "s3": 0, "total": 0}
                    result[config_key]["s3"] = count

        # Compute totals
        for config_key in result:
            result[config_key]["total"] = (
                result[config_key]["local"]
                + result[config_key]["cluster"]
                + result[config_key]["owc"]
                + result[config_key]["s3"]
            )

        return result

    def get_total_games(self, board_type: str, num_players: int) -> int:
        """Get total games across ALL sources (local + cluster + owc + s3).

        Jan 2026: Added for unified cluster data visibility.

        Args:
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)

        Returns:
            Total games across all sources
        """
        config_key = f"{board_type}_{num_players}p"
        status = self.get_cluster_status()
        if config_key in status:
            return status[config_key]["total"]
        return 0

    def get_data_sources(self, board_type: str, num_players: int) -> list[dict]:
        """List all sources with data for this config.

        Jan 2026: Added for unified cluster data visibility.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            List of dicts with source info (type, location, game_count)
        """
        config_key = f"{board_type}_{num_players}p"
        sources = []

        # Local sources
        try:
            local_sources = self._catalog.discover_data_sources(force=False)
            for source in local_sources:
                # DataSource has board_types (set) and player_counts (set)
                board_types = getattr(source, "board_types", set())
                player_counts = getattr(source, "player_counts", set())

                # Check if this source matches the config
                for bt in board_types:
                    for np in player_counts:
                        if f"{bt}_{np}p" == config_key:
                            sources.append({
                                "type": "local",
                                "location": str(source.path),
                                "game_count": source.game_count,
                            })
                            break
        except Exception:
            pass

        # Cluster sources from manifest
        manifest = self._cluster_manifest
        if manifest:
            node_manifests = getattr(manifest, "node_manifests", {})
            for node_id, node_manifest in node_manifests.items():
                for f in getattr(node_manifest, "files", []):
                    file_config = f"{getattr(f, 'board_type', '')}_{getattr(f, 'num_players', 0)}p"
                    if file_config == config_key:
                        sources.append({
                            "type": "cluster",
                            "location": f"{node_id}:{getattr(f, 'path', '')}",
                            "game_count": getattr(f, "game_count", 0),
                        })

            # External storage
            external = getattr(manifest, "external_storage", None)
            if external:
                owc_count = getattr(external, "owc_games_by_config", {}).get(config_key, 0)
                if owc_count:
                    sources.append({
                        "type": "owc",
                        "location": "mac-studio:/Volumes/RingRift-Data",
                        "game_count": owc_count,
                    })

                s3_count = getattr(external, "s3_games_by_config", {}).get(config_key, 0)
                if s3_count:
                    sources.append({
                        "type": "s3",
                        "location": f"s3://{getattr(external, 's3_bucket', 'ringrift-data')}",
                        "game_count": s3_count,
                    })

        return sources

    def get_manifest_age_seconds(self) -> float:
        """Get age of received cluster manifest in seconds.

        Returns:
            Seconds since manifest was received, or -1 if no manifest.
        """
        import time
        if not self._cluster_manifest_received_at:
            return -1
        return time.time() - self._cluster_manifest_received_at

    async def refresh_cluster_manifest(self, leader_url: str | None = None) -> bool:
        """Query leader for fresh cluster manifest.

        Jan 16, 2026: Added for on-demand cluster data queries.

        This method queries the P2P leader's /data/request_manifest endpoint
        to get fresh cluster-wide game counts without waiting for the periodic
        broadcast (which runs every 300s).

        Args:
            leader_url: URL of P2P leader (e.g., "http://192.168.1.10:8770").
                       If not provided, attempts to discover via local P2P node.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        import time

        try:
            import aiohttp
        except ImportError:
            logger.warning("[UnifiedDataRegistry] aiohttp not available for manifest refresh")
            return False

        # Discover leader URL if not provided
        if not leader_url:
            leader_url = await self._discover_leader_url()
            if not leader_url:
                logger.debug("[UnifiedDataRegistry] Could not discover leader URL")
                return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{leader_url.rstrip('/')}/data/request_manifest"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        manifest = data.get("cluster_manifest")
                        if manifest:
                            self._cluster_manifest = manifest
                            self._cluster_manifest_received_at = time.time()
                            logger.info(
                                f"[UnifiedDataRegistry] Refreshed cluster manifest from {leader_url}: "
                                f"{len(manifest.get('by_board_type', {}))} configs"
                            )
                            return True
                        logger.debug("[UnifiedDataRegistry] Leader returned empty manifest")
                        return False
                    elif resp.status == 404:
                        # Leader may redirect to another node
                        data = await resp.json()
                        if "leader_url" in data and data["leader_url"]:
                            logger.debug(
                                f"[UnifiedDataRegistry] Redirecting to leader: {data['leader_url']}"
                            )
                            return await self.refresh_cluster_manifest(data["leader_url"])
                        logger.debug("[UnifiedDataRegistry] Leader has no manifest yet")
                        return False
                    else:
                        logger.warning(
                            f"[UnifiedDataRegistry] Failed to refresh manifest: HTTP {resp.status}"
                        )
                        return False
        except asyncio.TimeoutError:
            logger.warning(f"[UnifiedDataRegistry] Manifest refresh timeout from {leader_url}")
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"[UnifiedDataRegistry] Manifest refresh failed: {e}")
            return False
        except Exception as e:
            logger.debug(f"[UnifiedDataRegistry] Unexpected error refreshing manifest: {e}")
            return False

    async def _discover_leader_url(self) -> str | None:
        """Discover P2P leader URL from local node status.

        Returns:
            Leader URL (e.g., "http://192.168.1.10:8770") or None.
        """
        try:
            import aiohttp
        except ImportError:
            return None

        # Try local P2P node first
        local_p2p_url = "http://localhost:8770"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{local_p2p_url}/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        leader_id = status.get("leader_id")
                        # Check if we are the leader
                        if leader_id == status.get("node_id"):
                            return local_p2p_url
                        # Get leader URL from peers
                        peers = status.get("peers", {})
                        leader_peer = peers.get(leader_id, {})
                        leader_url = leader_peer.get("http_url") or leader_peer.get("url")
                        if leader_url:
                            return leader_url
                        # Fallback: construct from leader_id if it looks like an IP
                        if leader_id and "." in leader_id:
                            return f"http://{leader_id}:8770"
        except Exception as e:
            logger.debug(f"[UnifiedDataRegistry] Failed to discover leader from local P2P: {e}")

        return None

    def is_cluster_data_available(self) -> tuple[bool, str]:
        """Check if cluster-wide data is available for progress measurement.

        Jan 2026: Added to ensure cluster-wide progress measurement.
        Use this to verify that game counts reflect the entire cluster,
        not just local data on the coordinator.

        Returns:
            (is_available, reason) tuple where:
            - is_available: True if cluster data is fresh and available
            - reason: Human-readable status message
        """
        manifest = self.get_cluster_manifest()
        if manifest is None:
            return False, "No cluster manifest received from P2P leader"

        age = self.get_manifest_age_seconds()
        if age > 600:  # 10 minutes - manifest is stale
            return False, f"Manifest too old ({age:.0f}s, max 600s)"

        total_nodes = getattr(manifest, "total_nodes", 0)
        if total_nodes < 2:
            return False, f"Only {total_nodes} node(s) in manifest (need >=2 for cluster)"

        return True, f"Manifest has {total_nodes} nodes, {age:.0f}s old"


# Singleton registry instance
_registry_instance: UnifiedDataRegistry | None = None


def get_data_registry() -> UnifiedDataRegistry:
    """Get the singleton UnifiedDataRegistry instance.

    Returns:
        UnifiedDataRegistry singleton instance
    """
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = UnifiedDataRegistry()

    return _registry_instance


def reset_data_registry() -> None:
    """Reset the registry singleton (for testing)."""
    global _registry_instance
    _registry_instance = None


# =============================================================================
# ClusterAwareDataCatalog - P2P cluster data visibility (Dec 30, 2025)
# Phase 1.2: Cluster-Aware Data Discovery for Distributed Training Pipeline
# =============================================================================


@dataclass
class DataSourceInfo:
    """Information about a data source in the cluster.

    Attributes:
        node_id: Node where data is located
        path: Path to data on that node
        game_count: Number of games available
        sample_count: Estimated training samples
        is_local: Whether this is on the local node
        is_synced: Whether data has been synced locally
        quality_score: Data quality estimate (0-1)
        last_updated: Timestamp of last update
    """

    node_id: str
    path: Path | str
    game_count: int = 0
    sample_count: int = 0
    is_local: bool = False
    is_synced: bool = False
    quality_score: float = 0.0
    last_updated: float = 0.0

    @property
    def config_key(self) -> str | None:
        """Extract config key from path if possible."""
        path_str = str(self.path)
        for board in ["hex8", "hexagonal", "square8", "square19"]:
            for players in [2, 3, 4]:
                key = f"{board}_{players}p"
                if key in path_str:
                    return key
        return None


class ClusterAwareDataCatalog:
    """Data catalog with P2P cluster awareness.

    Extends local data discovery with cluster-wide visibility via P2P manifest.
    Enables training nodes to find and lazy-download data from any cluster node.

    Dec 30, 2025: Phase 1.2 of Distributed Data Pipeline Architecture.

    Design:
    - Query P2P manifest for cluster-wide game counts
    - Find best data source (local first, then remote)
    - Lazy download pattern: local → P2P pull → sync from source

    Usage:
        from app.distributed.data_catalog import (
            get_cluster_aware_catalog,
            ClusterAwareDataCatalog,
        )

        catalog = get_cluster_aware_catalog()

        # Get cluster-wide game counts
        counts = catalog.get_cluster_game_counts("hex8_2p")

        # Find best data source
        source = catalog.get_best_data_source("hex8_2p")

        # Ensure data is locally available
        path = await catalog.ensure_data_available("hex8_2p", min_games=5000)
    """

    _instance: "ClusterAwareDataCatalog | None" = None

    def __init__(
        self,
        local_catalog: DataCatalog | None = None,
        p2p_base_url: str | None = None,
    ):
        """Initialize cluster-aware catalog.

        Args:
            local_catalog: Local DataCatalog instance (uses singleton if None)
            p2p_base_url: Base URL for P2P orchestrator API (defaults to get_local_p2p_url())
        """
        if p2p_base_url is None:
            p2p_base_url = get_local_p2p_url()
        self._local_catalog = local_catalog or get_data_catalog()
        self._p2p_base_url = p2p_base_url

        # Cache for cluster data
        self._cluster_game_counts: dict[str, dict[str, int]] = {}
        self._cluster_npz_counts: dict[str, dict[str, int]] = {}
        self._nodes_with_data: dict[str, list[str]] = {}
        self._last_manifest_refresh: float = 0.0
        self._manifest_cache_ttl: float = 60.0  # 1 minute cache

        # Local node info
        self._node_id = socket.gethostname()

    @classmethod
    def get_instance(cls) -> "ClusterAwareDataCatalog":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def get_cluster_game_counts(self, config_key: str) -> dict[str, int]:
        """Get game counts across entire cluster via P2P manifest.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Dict mapping node_id -> game count for this config
        """
        self._refresh_manifest_if_stale()

        if config_key in self._cluster_game_counts:
            return self._cluster_game_counts[config_key]

        # Query P2P manifest directly
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            counts = manifest.get_games_by_node_and_config(config_key)
            self._cluster_game_counts[config_key] = counts
            return counts
        except Exception as e:
            logger.debug(f"[ClusterAwareCatalog] Manifest query failed: {e}")
            return {}

    def get_cluster_total_games(self, config_key: str) -> int:
        """Get total game count for a config across entire cluster.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            Total games across all nodes
        """
        counts = self.get_cluster_game_counts(config_key)
        return sum(counts.values())

    def get_nodes_with_data(self, config_key: str) -> list[str]:
        """Get list of nodes that have data for this config.

        Args:
            config_key: Configuration key

        Returns:
            List of node IDs with data for this config
        """
        self._refresh_manifest_if_stale()

        if config_key in self._nodes_with_data:
            return self._nodes_with_data[config_key]

        counts = self.get_cluster_game_counts(config_key)
        nodes = [node_id for node_id, count in counts.items() if count > 0]
        self._nodes_with_data[config_key] = nodes
        return nodes

    def get_best_data_source(
        self,
        config_key: str,
        min_games: int = 0,
        prefer_local: bool = True,
    ) -> DataSourceInfo | None:
        """Find best data source for a configuration.

        Preference order:
        1. Local data (if available and sufficient)
        2. Synced data from other nodes
        3. Remote data (requires P2P sync)

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            min_games: Minimum games required
            prefer_local: Prefer local data even if remote has more

        Returns:
            Best DataSourceInfo, or None if no data available
        """
        # Parse config key
        parts = config_key.replace("_", " ").split()
        if len(parts) < 2:
            return None
        board_type = parts[0]
        try:
            num_players = int(parts[1].rstrip("p"))
        except ValueError:
            return None

        # Check local sources first
        local_sources = self._find_local_sources(board_type, num_players)
        local_total = sum(s.game_count for s in local_sources)

        if local_sources and (prefer_local or local_total >= min_games):
            # Return best local source
            best = max(local_sources, key=lambda s: s.game_count)
            return DataSourceInfo(
                node_id=self._node_id,
                path=best.path,
                game_count=best.game_count,
                sample_count=best.game_count * 30,  # Estimate
                is_local=True,
                is_synced=True,
                quality_score=best.avg_quality_score,
                last_updated=best.last_modified,
            )

        # Check cluster for remote sources
        cluster_counts = self.get_cluster_game_counts(config_key)
        if not cluster_counts:
            return None

        # Find node with most data
        best_node = max(cluster_counts.items(), key=lambda x: x[1])
        node_id, game_count = best_node

        if game_count < min_games:
            # Not enough data anywhere
            return None

        # Return remote source info
        return DataSourceInfo(
            node_id=node_id,
            path=f"data/games/selfplay_{config_key}.db",  # Typical path
            game_count=game_count,
            sample_count=game_count * 30,
            is_local=node_id == self._node_id,
            is_synced=False,
            quality_score=0.5,  # Unknown quality for remote
            last_updated=time.time(),
        )

    def _find_local_sources(
        self,
        board_type: str,
        num_players: int,
    ) -> list[DataSource]:
        """Find local data sources for a config.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            List of matching local DataSource objects
        """
        self._local_catalog.discover_data_sources()
        matches = []

        for source in self._local_catalog._sources.values():
            if board_type in source.board_types and num_players in source.player_counts:
                matches.append(source)

        return matches

    async def ensure_data_available(
        self,
        config_key: str,
        min_games: int = 1000,
        timeout: float = 300.0,
    ) -> Path | None:
        """Ensure data is locally available, triggering sync if needed.

        Implements lazy download pattern:
        1. Check local data availability
        2. If insufficient, find best remote source
        3. Trigger P2P sync to fetch data
        4. Return local path once available

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            min_games: Minimum games required
            timeout: Max time to wait for sync (seconds)

        Returns:
            Path to local data, or None if unavailable
        """
        import asyncio

        # Check local availability first
        source = self.get_best_data_source(config_key, min_games, prefer_local=True)

        if source is None:
            logger.warning(f"[ClusterAwareCatalog] No data source for {config_key}")
            return None

        if source.is_local:
            return Path(source.path)

        # Need to sync from remote
        logger.info(
            f"[ClusterAwareCatalog] Syncing {config_key} from {source.node_id} "
            f"({source.game_count} games)"
        )

        # Trigger sync via event
        try:
            from app.coordination.data_events import DataEventType
            from app.coordination.event_router import get_event_router

            router = get_event_router()
            await router.emit(
                DataEventType.DATA_SYNC_REQUESTED,
                {
                    "config_key": config_key,
                    "source_node": source.node_id,
                    "min_games": min_games,
                    "priority": "high",
                },
            )
        except ImportError:
            logger.warning("[ClusterAwareCatalog] Event router not available for sync")
            return None

        # Wait for sync to complete
        start_time = time.time()
        poll_interval = 5.0

        while (time.time() - start_time) < timeout:
            await asyncio.sleep(poll_interval)

            # Check if data is now available locally
            local_source = self.get_best_data_source(
                config_key, min_games, prefer_local=True
            )
            if local_source and local_source.is_local:
                logger.info(
                    f"[ClusterAwareCatalog] Sync complete for {config_key}, "
                    f"{local_source.game_count} games available"
                )
                return Path(local_source.path)

        logger.warning(f"[ClusterAwareCatalog] Sync timeout for {config_key}")
        return None

    def _refresh_manifest_if_stale(self) -> None:
        """Refresh manifest data if cache is stale."""
        now = time.time()
        if (now - self._last_manifest_refresh) < self._manifest_cache_ttl:
            return

        try:
            self._fetch_manifest_data()
            self._last_manifest_refresh = now
        except Exception as e:
            logger.debug(f"[ClusterAwareCatalog] Manifest refresh failed: {e}")

    def _fetch_manifest_data(self) -> None:
        """Fetch manifest data from P2P or local manifest."""
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()

            # Get games by config
            games_by_config = manifest.get_games_count_by_config()
            for config_key, count in games_by_config.items():
                if config_key not in self._cluster_game_counts:
                    self._cluster_game_counts[config_key] = {}
                # Aggregate into total for now
                self._cluster_game_counts[config_key]["_total"] = count

        except Exception as e:
            logger.debug(f"[ClusterAwareCatalog] Failed to fetch manifest: {e}")

    def get_all_config_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all configurations across cluster.

        Returns:
            Dict mapping config_key -> status info including:
                - total_games: Cluster-wide game count
                - local_games: Local game count
                - nodes_with_data: List of nodes
                - has_sufficient_data: Whether min threshold met
        """
        self._refresh_manifest_if_stale()

        status = {}
        all_configs = set()

        # Get configs from manifest
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()
            games_by_config = manifest.get_games_count_by_config()
            all_configs.update(games_by_config.keys())
        except Exception:
            pass

        # Get configs from local catalog
        self._local_catalog.discover_data_sources()
        for source in self._local_catalog._sources.values():
            for bt in source.board_types:
                for np in source.player_counts:
                    all_configs.add(f"{bt}_{np}p")

        # Build status for each config
        for config_key in sorted(all_configs):
            if config_key == "None_Nonep" or not config_key:
                continue

            cluster_total = self.get_cluster_total_games(config_key)
            nodes = self.get_nodes_with_data(config_key)

            # Count local games
            local_games = 0
            parts = config_key.split("_")
            if len(parts) >= 2:
                bt = parts[0]
                try:
                    np = int(parts[1].rstrip("p"))
                    local_sources = self._find_local_sources(bt, np)
                    local_games = sum(s.game_count for s in local_sources)
                except ValueError:
                    pass

            status[config_key] = {
                "total_games": cluster_total,
                "local_games": local_games,
                "nodes_with_data": nodes,
                "has_sufficient_data": cluster_total >= 5000,
            }

        return status

    def health_check(self) -> "HealthCheckResult":
        """Check health of cluster-aware catalog.

        Returns:
            HealthCheckResult with catalog status
        """
        from app.coordination.protocols import HealthCheckResult

        try:
            config_status = self.get_all_config_status()
            total_configs = len(config_status)
            configs_with_data = sum(
                1 for s in config_status.values() if s["total_games"] > 0
            )

            is_healthy = configs_with_data > 0
            message = (
                f"{configs_with_data}/{total_configs} configs have data"
                if is_healthy
                else "No cluster data available"
            )

            return HealthCheckResult(
                healthy=is_healthy,
                message=message,
                details={
                    "total_configs": total_configs,
                    "configs_with_data": configs_with_data,
                    "manifest_age_seconds": time.time() - self._last_manifest_refresh,
                    "node_id": self._node_id,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )


# Singleton accessor for ClusterAwareDataCatalog
_cluster_aware_instance: ClusterAwareDataCatalog | None = None


def get_cluster_aware_catalog() -> ClusterAwareDataCatalog:
    """Get the singleton ClusterAwareDataCatalog instance.

    Returns:
        ClusterAwareDataCatalog singleton instance
    """
    global _cluster_aware_instance

    if _cluster_aware_instance is None:
        _cluster_aware_instance = ClusterAwareDataCatalog()

    return _cluster_aware_instance


def reset_cluster_aware_catalog() -> None:
    """Reset the cluster-aware catalog singleton (for testing)."""
    global _cluster_aware_instance
    _cluster_aware_instance = None


__all__ = [
    "CatalogStats",
    "DataCatalog",
    "DataSource",
    # NPZ discovery (December 2025)
    "NPZDataSource",
    "TrainingSourceRecommendation",
    "TrainingSourceSuggestion",
    # Unified registry (December 2025)
    "UnifiedDataRegistry",
    "get_data_catalog",
    "get_data_registry",
    "reset_data_catalog",
    "reset_data_registry",
    # Cluster-aware catalog (December 30, 2025)
    "ClusterAwareDataCatalog",
    "DataSourceInfo",
    "get_cluster_aware_catalog",
    "reset_cluster_aware_catalog",
]
