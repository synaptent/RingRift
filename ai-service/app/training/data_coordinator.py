"""Training Data Coordinator - Unified orchestration for quality-aware training data.

This module provides a single coordination point for all training data operations:
- Quality-aware data loading from databases and manifest
- Sync coordination with the distributed cluster
- Configuration of data loaders (HotDataBuffer, StreamingPipeline)
- Quality metrics collection and reporting

Usage:
    from app.training.data_coordinator import (
        TrainingDataCoordinator,
        get_data_coordinator,
    )

    # Get singleton instance
    coordinator = get_data_coordinator()

    # Prepare for training (syncs high-quality data, configures loaders)
    await coordinator.prepare_for_training(board_type="square8", num_players=2)

    # Get configured hot buffer
    hot_buffer = coordinator.get_hot_buffer()

    # Load high-quality games directly
    loaded = coordinator.load_high_quality_games(db_path, min_quality=0.7)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.training.hot_data_buffer import HotDataBuffer
    from app.training.streaming_pipeline import StreamingDataPipeline

logger = logging.getLogger(__name__)

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        MIN_QUALITY_FOR_PRIORITY_SYNC,
        HIGH_QUALITY_THRESHOLD,
    )
except ImportError:
    MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5
    HIGH_QUALITY_THRESHOLD = 0.7

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_SELFPLAY_DIR = DEFAULT_DATA_DIR / "games"


@dataclass
class CoordinatorConfig:
    """Configuration for the TrainingDataCoordinator.

    Attributes:
        enable_quality_scoring: Enable quality-based data selection
        enable_sync: Enable automatic sync before training
        min_quality_threshold: Minimum quality score for training data
        min_elo_threshold: Minimum average Elo for training data
        sync_high_quality_first: Sync high-quality games before bulk sync
        hot_buffer_size: Maximum size of hot data buffer
        hot_buffer_memory_mb: Maximum memory for hot buffer
        refresh_interval_seconds: How often to refresh quality data
        auto_load_from_db: Automatically load games from DB on startup
    """
    enable_quality_scoring: bool = True
    enable_sync: bool = True
    min_quality_threshold: float = MIN_QUALITY_FOR_PRIORITY_SYNC
    min_elo_threshold: float = 1400.0
    sync_high_quality_first: bool = True
    hot_buffer_size: int = 1000
    hot_buffer_memory_mb: int = 500
    refresh_interval_seconds: float = 300.0
    auto_load_from_db: bool = True


@dataclass
class CoordinatorStats:
    """Statistics about the coordinator's operations."""
    total_games_loaded: int = 0
    high_quality_games_loaded: int = 0
    games_synced: int = 0
    last_sync_time: float = 0.0
    last_load_time: float = 0.0
    avg_quality_score: float = 0.0
    avg_elo: float = 0.0
    preparation_count: int = 0
    errors: List[str] = field(default_factory=list)


class TrainingDataCoordinator:
    """Unified coordinator for quality-aware training data operations.

    This class provides a single orchestration point that integrates:
    - QualityBridge for quality score lookups
    - SyncCoordinator for cluster data synchronization
    - HotDataBuffer for in-memory game caching
    - StreamingDataPipeline for batch loading

    The coordinator ensures that training always has access to the highest
    quality data available across the cluster.
    """

    _instance: Optional["TrainingDataCoordinator"] = None

    def __init__(
        self,
        config: Optional[CoordinatorConfig] = None,
        selfplay_dir: Optional[Path] = None,
    ):
        """Initialize the training data coordinator.

        Args:
            config: Coordinator configuration
            selfplay_dir: Directory containing selfplay game databases
        """
        self._config = config or CoordinatorConfig()
        self._selfplay_dir = selfplay_dir or DEFAULT_SELFPLAY_DIR
        self._stats = CoordinatorStats()

        # Lazy-loaded components
        self._quality_bridge = None
        self._sync_coordinator = None
        self._hot_buffer: Optional["HotDataBuffer"] = None
        self._streaming_pipeline: Optional["StreamingDataPipeline"] = None

        # State
        self._initialized = False
        self._last_preparation_time = 0.0

        logger.info(
            f"TrainingDataCoordinator initialized: "
            f"quality_scoring={self._config.enable_quality_scoring}, "
            f"sync={self._config.enable_sync}"
        )

    @classmethod
    def get_instance(
        cls,
        config: Optional[CoordinatorConfig] = None,
    ) -> "TrainingDataCoordinator":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    # =========================================================================
    # Component Initialization
    # =========================================================================

    def _get_quality_bridge(self):
        """Lazily get QualityBridge instance."""
        if self._quality_bridge is None and self._config.enable_quality_scoring:
            try:
                from app.training.quality_bridge import get_quality_bridge
                self._quality_bridge = get_quality_bridge()
                logger.debug("QualityBridge initialized")
            except ImportError:
                logger.warning("QualityBridge not available")
            except Exception as e:
                logger.warning(f"Failed to initialize QualityBridge: {e}")
        return self._quality_bridge

    def _get_sync_coordinator(self):
        """Lazily get SyncCoordinator instance."""
        if self._sync_coordinator is None and self._config.enable_sync:
            try:
                from app.distributed.sync_coordinator import SyncCoordinator
                self._sync_coordinator = SyncCoordinator.get_instance()
                logger.debug("SyncCoordinator initialized")
            except ImportError:
                logger.warning("SyncCoordinator not available")
            except Exception as e:
                logger.warning(f"Failed to initialize SyncCoordinator: {e}")
        return self._sync_coordinator

    def _create_hot_buffer(self) -> Optional["HotDataBuffer"]:
        """Create and configure a HotDataBuffer instance."""
        if self._hot_buffer is not None:
            return self._hot_buffer

        try:
            from app.training.hot_data_buffer import create_hot_buffer

            self._hot_buffer = create_hot_buffer(
                max_size=self._config.hot_buffer_size,
                max_memory_mb=self._config.hot_buffer_memory_mb,
                buffer_name="coordinator_buffer",
                enable_events=True,
            )

            # Configure with quality lookups
            bridge = self._get_quality_bridge()
            if bridge:
                bridge.configure_hot_data_buffer(self._hot_buffer)
                logger.info("HotDataBuffer configured with quality lookups")

            return self._hot_buffer

        except ImportError:
            logger.warning("HotDataBuffer not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to create HotDataBuffer: {e}")
            self._stats.errors.append(f"HotDataBuffer creation failed: {e}")
            return None

    # =========================================================================
    # Training Preparation
    # =========================================================================

    async def prepare_for_training(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        force_sync: bool = False,
        load_from_db: bool = True,
    ) -> Dict[str, Any]:
        """Prepare training data for a training run.

        This method:
        1. Syncs high-quality games from the cluster (if enabled)
        2. Refreshes quality lookups from manifest
        3. Configures data loaders with quality scores
        4. Optionally loads games from local databases

        Args:
            board_type: Board type for training
            num_players: Number of players
            force_sync: Force sync even if recently synced
            load_from_db: Load games from local databases

        Returns:
            Dict with preparation statistics
        """
        start_time = time.time()
        result = {
            "success": True,
            "games_synced": 0,
            "games_loaded": 0,
            "avg_quality": 0.0,
            "duration_seconds": 0.0,
            "errors": [],
        }

        # Step 1: Sync high-quality games from cluster
        if self._config.enable_sync:
            sync_result = await self._sync_high_quality_data(force=force_sync)
            result["games_synced"] = sync_result.get("games_synced", 0)
            if sync_result.get("errors"):
                result["errors"].extend(sync_result["errors"])

        # Step 2: Refresh quality lookups
        bridge = self._get_quality_bridge()
        if bridge:
            bridge.refresh(force=True)
            result["avg_quality"] = bridge.get_stats().avg_quality_score

            # Reconfigure hot buffer with fresh lookups
            if self._hot_buffer:
                bridge.configure_hot_data_buffer(self._hot_buffer)

        # Step 3: Load games from local databases
        if load_from_db and self._config.auto_load_from_db:
            loaded = self._load_games_from_local_dbs(
                board_type=board_type,
                num_players=num_players,
            )
            result["games_loaded"] = loaded
            self._stats.total_games_loaded += loaded

        # Update stats
        self._stats.preparation_count += 1
        self._last_preparation_time = time.time()
        result["duration_seconds"] = time.time() - start_time

        logger.info(
            f"Training preparation complete: "
            f"synced={result['games_synced']}, loaded={result['games_loaded']}, "
            f"avg_quality={result['avg_quality']:.3f}, "
            f"duration={result['duration_seconds']:.1f}s"
        )

        return result

    async def _sync_high_quality_data(self, force: bool = False) -> Dict[str, Any]:
        """Sync high-quality games from the cluster."""
        result = {"games_synced": 0, "errors": []}

        # Check if we need to sync
        if not force:
            time_since_sync = time.time() - self._stats.last_sync_time
            if time_since_sync < self._config.refresh_interval_seconds:
                logger.debug(f"Skipping sync - last sync was {time_since_sync:.0f}s ago")
                return result

        coordinator = self._get_sync_coordinator()
        if not coordinator:
            return result

        try:
            # Sync high-quality games first
            if self._config.sync_high_quality_first:
                stats = await coordinator.sync_high_quality_games(
                    min_quality_score=self._config.min_quality_threshold,
                    min_elo=self._config.min_elo_threshold,
                    limit=1000,
                )
                result["games_synced"] = stats.high_quality_games_synced
                self._stats.games_synced += stats.high_quality_games_synced

                if stats.errors:
                    result["errors"].extend(stats.errors)

            self._stats.last_sync_time = time.time()

        except Exception as e:
            logger.warning(f"High-quality sync failed: {e}")
            result["errors"].append(str(e))
            self._stats.errors.append(f"Sync failed: {e}")

        return result

    def _load_games_from_local_dbs(
        self,
        board_type: str,
        num_players: int,
    ) -> int:
        """Load games from local selfplay databases."""
        if not self._hot_buffer:
            self._create_hot_buffer()

        if not self._hot_buffer:
            return 0

        total_loaded = 0

        # Find all relevant databases
        db_pattern = f"*{board_type}*.db"
        for db_path in self._selfplay_dir.glob(db_pattern):
            try:
                loaded = self._hot_buffer.load_from_db(
                    db_path=db_path,
                    board_type=board_type,
                    num_players=num_players,
                    min_quality=self._config.min_quality_threshold,
                    limit=500,  # Limit per DB to avoid memory issues
                )
                total_loaded += loaded
            except Exception as e:
                logger.warning(f"Failed to load from {db_path}: {e}")

        self._stats.last_load_time = time.time()
        return total_loaded

    # =========================================================================
    # Data Access
    # =========================================================================

    def get_hot_buffer(self) -> Optional["HotDataBuffer"]:
        """Get the configured HotDataBuffer instance."""
        if self._hot_buffer is None:
            self._create_hot_buffer()
        return self._hot_buffer

    def get_quality_bridge(self):
        """Get the QualityBridge instance."""
        return self._get_quality_bridge()

    def get_sync_coordinator(self):
        """Get the SyncCoordinator instance."""
        return self._get_sync_coordinator()

    def load_high_quality_games(
        self,
        db_path: Path,
        board_type: str = "square8",
        num_players: int = 2,
        min_quality: float = HIGH_QUALITY_THRESHOLD,
        limit: int = 1000,
    ) -> int:
        """Load high-quality games from a specific database.

        Args:
            db_path: Path to the SQLite database
            board_type: Board type to filter
            num_players: Number of players to filter
            min_quality: Minimum quality score (default from quality.thresholds)
            limit: Maximum games to load

        Returns:
            Number of games loaded
        """
        hot_buffer = self.get_hot_buffer()
        if not hot_buffer:
            logger.warning("Cannot load games - HotDataBuffer not available")
            return 0

        loaded = hot_buffer.load_from_db(
            db_path=db_path,
            board_type=board_type,
            num_players=num_players,
            min_quality=min_quality,
            limit=limit,
        )

        self._stats.total_games_loaded += loaded
        if min_quality >= HIGH_QUALITY_THRESHOLD:
            self._stats.high_quality_games_loaded += loaded

        return loaded

    def get_high_quality_game_ids(
        self,
        min_quality: Optional[float] = None,
        limit: int = 10000,
    ) -> List[str]:
        """Get list of high-quality game IDs.

        Args:
            min_quality: Minimum quality score (uses config default if None)
            limit: Maximum number of games to return

        Returns:
            List of game IDs meeting quality criteria
        """
        if min_quality is None:
            min_quality = self._config.min_quality_threshold

        bridge = self._get_quality_bridge()
        if bridge:
            return bridge.get_high_quality_game_ids(
                min_quality=min_quality,
                min_elo=self._config.min_elo_threshold,
                limit=limit,
            )

        return []

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_stats(self) -> CoordinatorStats:
        """Get coordinator statistics."""
        # Update avg_quality from bridge
        bridge = self._get_quality_bridge()
        if bridge:
            stats = bridge.get_stats()
            self._stats.avg_quality_score = stats.avg_quality_score
            self._stats.avg_elo = stats.avg_elo

        return self._stats

    def get_status(self) -> Dict[str, Any]:
        """Get detailed coordinator status."""
        stats = self.get_stats()

        return {
            "initialized": self._initialized,
            "config": {
                "quality_scoring": self._config.enable_quality_scoring,
                "sync_enabled": self._config.enable_sync,
                "min_quality": self._config.min_quality_threshold,
                "min_elo": self._config.min_elo_threshold,
            },
            "stats": {
                "total_games_loaded": stats.total_games_loaded,
                "high_quality_games_loaded": stats.high_quality_games_loaded,
                "games_synced": stats.games_synced,
                "avg_quality_score": stats.avg_quality_score,
                "avg_elo": stats.avg_elo,
                "preparation_count": stats.preparation_count,
            },
            "components": {
                "quality_bridge": self._quality_bridge is not None,
                "sync_coordinator": self._sync_coordinator is not None,
                "hot_buffer": self._hot_buffer is not None,
                "hot_buffer_size": len(self._hot_buffer) if self._hot_buffer else 0,
            },
            "timing": {
                "last_sync_age_seconds": time.time() - stats.last_sync_time if stats.last_sync_time else 0,
                "last_load_age_seconds": time.time() - stats.last_load_time if stats.last_load_time else 0,
                "last_preparation_age_seconds": time.time() - self._last_preparation_time if self._last_preparation_time else 0,
            },
            "errors": stats.errors[-5:] if stats.errors else [],  # Last 5 errors
        }

    def collect_metrics(self) -> bool:
        """Collect and export Prometheus metrics.

        Returns:
            True if metrics were collected successfully
        """
        try:
            from app.metrics.orchestrator import (
                collect_quality_metrics_from_bridge,
                update_quality_bridge_status,
            )

            # Collect from quality bridge
            if self._quality_bridge:
                collect_quality_metrics_from_bridge()

            return True

        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
            return False


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_data_coordinator(
    config: Optional[CoordinatorConfig] = None,
) -> TrainingDataCoordinator:
    """Get the singleton TrainingDataCoordinator instance."""
    return TrainingDataCoordinator.get_instance(config)


async def prepare_training_data(
    board_type: str = "square8",
    num_players: int = 2,
    force_sync: bool = False,
) -> Dict[str, Any]:
    """Convenience function to prepare training data."""
    coordinator = get_data_coordinator()
    return await coordinator.prepare_for_training(
        board_type=board_type,
        num_players=num_players,
        force_sync=force_sync,
    )


def get_high_quality_games(
    min_quality: float = HIGH_QUALITY_THRESHOLD,
    limit: int = 10000,
) -> List[str]:
    """Convenience function to get high-quality game IDs."""
    coordinator = get_data_coordinator()
    return coordinator.get_high_quality_game_ids(min_quality=min_quality, limit=limit)


__all__ = [
    "TrainingDataCoordinator",
    "CoordinatorConfig",
    "CoordinatorStats",
    "get_data_coordinator",
    "prepare_training_data",
    "get_high_quality_games",
]
