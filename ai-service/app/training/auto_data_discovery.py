"""Automatic data discovery for training nodes.

.. note:: Integration Complete (2025-12-20)
    This module is now integrated into app/training/data_coordinator.py.
    TrainingDataCoordinator.prepare_for_training() automatically runs
    data discovery when enable_auto_discovery=True (default).

    Usage:
        from app.training.data_coordinator import get_data_coordinator

        coordinator = get_data_coordinator()
        result = await coordinator.prepare_for_training(board_type="square8")
        print(f"Discovered {result['games_discovered']} games")

    Or use discovery directly:
        from app.training.auto_data_discovery import discover_training_data
        discovery = discover_training_data(board_type="square8", num_players=2)

This module provides automatic discovery of high-quality training data
from synced sources. It's designed to be called automatically when a
training job starts, ensuring the node uses the best available data
from across the cluster.

Usage:
    from app.training.auto_data_discovery import (
        discover_training_data,
        get_best_data_paths,
        should_auto_discover,
    )

    # Check if auto-discovery should be enabled
    if should_auto_discover():
        # Get recommended data paths
        paths = get_best_data_paths(
            target_games=50000,
            min_quality=0.5,
        )

    # Or use the full discovery function
    discovery_result = discover_training_data(
        board_type="square8",
        num_players=2,
    )
"""

from __future__ import annotations

import logging
import os
import socket
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    from app.distributed.data_catalog import CatalogStats, DataCatalog, get_data_catalog
    HAS_DATA_CATALOG = True
except ImportError:
    HAS_DATA_CATALOG = False
    DataCatalog = None
    get_data_catalog = None
    CatalogStats = None

try:
    from app.distributed.unified_manifest import DataManifest, GameQualityMetadata
    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False
    DataManifest = None
    GameQualityMetadata = None

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        MIN_QUALITY_FOR_PRIORITY_SYNC,
        MIN_QUALITY_FOR_TRAINING,
    )
except ImportError:
    MIN_QUALITY_FOR_TRAINING = 0.3
    MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5

try:
    from app.distributed.storage_provider import get_storage_provider, is_nfs_available
    HAS_STORAGE_PROVIDER = True
except ImportError:
    HAS_STORAGE_PROVIDER = False
    get_storage_provider = None
    def is_nfs_available():
        return False


@dataclass
class DiscoveryResult:
    """Result of automatic data discovery."""
    success: bool
    data_paths: list[Path] = field(default_factory=list)
    total_games: int = 0
    avg_quality_score: float = 0.0
    sources_by_host: dict[str, int] = field(default_factory=dict)
    sources_by_type: dict[str, int] = field(default_factory=dict)
    discovery_time_ms: float = 0.0
    error_message: str = ""

    @property
    def has_data(self) -> bool:
        """Check if any data was discovered."""
        return len(self.data_paths) > 0 and self.total_games > 0


def should_auto_discover() -> bool:
    """Determine if automatic data discovery should be enabled.

    Returns True if:
    - Running on a training node (detected by hostname or role)
    - DataCatalog is available
    - Not explicitly disabled via environment variable

    Returns:
        True if auto-discovery should be used
    """
    # Check for explicit disable
    if os.environ.get("RINGRIFT_DISABLE_AUTO_DISCOVERY", "").lower() in ("1", "true", "yes"):
        return False

    # Check if DataCatalog is available
    if not HAS_DATA_CATALOG:
        return False

    # Check for synced data directory
    sync_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
    if sync_dir.exists() and any(sync_dir.iterdir()):
        return True

    # Check for NFS shared storage
    if HAS_STORAGE_PROVIDER and is_nfs_available():
        return True

    # Check for training role in hostname
    hostname = socket.gethostname().lower()
    training_indicators = ["train", "gpu", "lambda", "vast"]
    return bool(any(ind in hostname for ind in training_indicators))


def get_best_data_paths(
    target_games: int = 50000,
    min_quality: float = 0.0,
    board_type: str | None = None,
    num_players: int | None = None,
    include_local: bool = True,
    include_synced: bool = True,
    include_nfs: bool = True,
) -> list[Path]:
    """Get the best data paths for training.

    This is a convenience function that returns just the paths,
    without the full DiscoveryResult metadata.

    Args:
        target_games: Target number of games to include
        min_quality: Minimum quality score threshold
        board_type: Filter by board type
        num_players: Filter by player count
        include_local: Include local data sources
        include_synced: Include synced data sources
        include_nfs: Include NFS shared data sources

    Returns:
        List of paths to recommended data sources
    """
    if not HAS_DATA_CATALOG:
        logger.debug("DataCatalog not available, returning empty list")
        return []

    try:
        catalog = get_data_catalog()
        return catalog.get_recommended_training_sources(
            target_games=target_games,
            board_type=board_type,
            num_players=num_players,
        )
    except Exception as e:
        logger.warning(f"Failed to get best data paths: {e}")
        return []


def discover_training_data(
    board_type: str | None = None,
    num_players: int | None = None,
    target_games: int = 100000,
    min_quality: float = 0.0,
    prefer_recent: bool = True,  # Reserved for future path-level sorting
    prefer_high_elo: bool = True,  # Reserved for future path-level sorting
) -> DiscoveryResult:
    """Perform full automatic data discovery for training.

    This function discovers all available training data sources,
    evaluates their quality, and returns a prioritized list of
    paths along with metadata.

    Args:
        board_type: Filter by board type (e.g., "square8")
        num_players: Filter by player count
        target_games: Target total number of games
        min_quality: Minimum quality score threshold
        prefer_recent: Prefer recently created games
        prefer_high_elo: Prefer games from high-Elo players

    Returns:
        DiscoveryResult with paths and metadata
    """
    # Reserved parameters for future path-level sorting
    _ = prefer_recent
    _ = prefer_high_elo

    import time
    start_time = time.time()

    result = DiscoveryResult(success=False)

    if not HAS_DATA_CATALOG:
        result.error_message = "DataCatalog not available"
        return result

    try:
        catalog = get_data_catalog()

        # Force fresh discovery
        catalog.refresh()

        # Get recommended sources
        paths = catalog.get_recommended_training_sources(
            target_games=target_games,
            board_type=board_type,
            num_players=num_players,
        )

        if not paths:
            result.error_message = "No data sources discovered"
            result.discovery_time_ms = (time.time() - start_time) * 1000
            return result

        # Get statistics
        stats = catalog.get_stats()

        result.success = True
        result.data_paths = paths
        result.total_games = stats.total_games
        result.avg_quality_score = stats.avg_quality_score
        result.sources_by_host = stats.sources_by_host
        result.sources_by_type = stats.sources_by_type
        result.discovery_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Auto-discovery found {len(paths)} sources with {stats.total_games} games "
            f"(avg quality: {stats.avg_quality_score:.3f}) in {result.discovery_time_ms:.0f}ms"
        )

        return result

    except Exception as e:
        result.error_message = str(e)
        result.discovery_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Auto-discovery failed: {e}")
        return result


def get_high_quality_game_ids(
    min_quality: float = MIN_QUALITY_FOR_PRIORITY_SYNC,
    limit: int = 10000,
    board_type: str | None = None,
    num_players: int | None = None,
    prefer_recent: bool = True,
    prefer_high_elo: bool = True,
) -> list[str]:
    """Get game IDs of high-quality games for training.

    Args:
        min_quality: Minimum quality score (default from quality.thresholds)
        limit: Maximum number of game IDs to return
        board_type: Filter by board type
        num_players: Filter by player count
        prefer_recent: Prefer recently created games
        prefer_high_elo: Prefer games from high-Elo players

    Returns:
        List of game IDs sorted by specified criteria
    """
    if not HAS_DATA_CATALOG:
        return []

    try:
        catalog = get_data_catalog()
        games = catalog.get_training_data(
            min_quality=min_quality,
            max_games=limit,
            board_type=board_type,
            num_players=num_players,
            prefer_recent=prefer_recent,
            prefer_high_elo=prefer_high_elo,
        )
        return [g.game_id for g in games]
    except Exception as e:
        logger.warning(f"Failed to get high-quality game IDs: {e}")
        return []


def get_discovery_status() -> dict[str, Any]:
    """Get the current status of data discovery.

    Returns:
        Dict with discovery status information
    """
    status = {
        "auto_discovery_available": HAS_DATA_CATALOG,
        "should_auto_discover": should_auto_discover(),
        "catalog_stats": None,
        "priority_queue_stats": None,
    }

    if HAS_DATA_CATALOG:
        try:
            catalog = get_data_catalog()
            status["catalog_stats"] = catalog.get_stats().__dict__
        except (AttributeError, TypeError, RuntimeError, OSError) as e:
            # Catalog errors: missing module, bad state, I/O issues
            status["catalog_error"] = str(e)

    if HAS_MANIFEST:
        try:
            manifest_path = AI_SERVICE_ROOT / "data" / "data_manifest.db"
            if manifest_path.exists():
                manifest = DataManifest(manifest_path)
                status["priority_queue_stats"] = manifest.get_priority_queue_stats()
                status["quality_distribution"] = manifest.get_quality_distribution()
        except (sqlite3.Error, OSError, ValueError, AttributeError) as e:
            # Manifest errors: DB access, file I/O, data parsing, missing attrs
            status["manifest_error"] = str(e)

    return status


def setup_auto_discovery_for_training(
    config: Any,
) -> dict[str, Any]:
    """Set up automatic data discovery for a training configuration.

    This function prepares the environment for automatic data discovery
    and returns recommended settings for the training job.

    Args:
        config: Training configuration object (TrainConfig)

    Returns:
        Dict with recommended training settings:
        - discover_synced_data: bool
        - min_quality_score: float
        - additional_data_paths: List[str]
    """
    result = {
        "discover_synced_data": False,
        "min_quality_score": 0.0,
        "additional_data_paths": [],
    }

    if not should_auto_discover():
        return result

    try:
        # Enable discovery
        result["discover_synced_data"] = True

        # Get board type from config if available
        board_type = None
        if hasattr(config, 'board_type'):
            board_type = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)

        num_players = getattr(config, 'num_players', 2)

        # Discover data
        discovery = discover_training_data(
            board_type=board_type,
            num_players=num_players,
        )

        if discovery.success:
            result["additional_data_paths"] = [str(p) for p in discovery.data_paths]
            # Set minimum quality based on distribution
            if discovery.avg_quality_score > MIN_QUALITY_FOR_PRIORITY_SYNC:
                result["min_quality_score"] = max(MIN_QUALITY_FOR_TRAINING, discovery.avg_quality_score - 0.2)

        return result

    except Exception as e:
        logger.warning(f"Failed to setup auto-discovery: {e}")
        return result
