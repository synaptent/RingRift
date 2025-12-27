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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.training.hot_data_buffer import HotDataBuffer
    from app.training.streaming_pipeline import StreamingDataPipeline

logger = logging.getLogger(__name__)

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        HIGH_QUALITY_THRESHOLD,
        MIN_QUALITY_FOR_PRIORITY_SYNC,
    )
except ImportError:
    MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5
    HIGH_QUALITY_THRESHOLD = 0.7

# Auto data discovery integration (December 2025)
# Provides automatic discovery of high-quality training data from synced sources
try:
    from app.training.auto_data_discovery import (
        DiscoveryResult,
        discover_training_data,
        get_best_data_paths,
        should_auto_discover,
    )
    HAS_AUTO_DISCOVERY = True
except ImportError:
    HAS_AUTO_DISCOVERY = False
    discover_training_data = None
    get_best_data_paths = None
    should_auto_discover = None
    DiscoveryResult = None

# Parity exclusions - databases with known parity failures
from app.training.parity_exclusions import (
    EXCLUDED_DB_PATTERNS,
    should_exclude_database,
)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_SELFPLAY_DIR = DEFAULT_DATA_DIR / "games"


@dataclass
class CoordinatorConfig:
    """Configuration for the TrainingDataCoordinator.

    Attributes:
        enable_quality_scoring: Enable quality-based data selection
        enable_sync: Enable automatic sync before training
        enable_auto_discovery: Enable automatic data discovery from synced sources
        min_quality_threshold: Minimum quality score for training data
        min_elo_threshold: Minimum average Elo for training data
        sync_high_quality_first: Sync high-quality games before bulk sync
        hot_buffer_size: Maximum size of hot data buffer
        hot_buffer_memory_mb: Maximum memory for hot buffer
        refresh_interval_seconds: How often to refresh quality data
        auto_load_from_db: Automatically load games from DB on startup
        auto_discovery_target_games: Target number of games for auto-discovery
        excluded_db_patterns: Database filename patterns to exclude from training
            (e.g., databases with known parity failures or phase coercion issues)
    """
    enable_quality_scoring: bool = True
    enable_sync: bool = True
    enable_auto_discovery: bool = True
    min_quality_threshold: float = MIN_QUALITY_FOR_PRIORITY_SYNC
    min_elo_threshold: float = 1400.0
    sync_high_quality_first: bool = True
    hot_buffer_size: int = 1000
    hot_buffer_memory_mb: int = 500
    refresh_interval_seconds: float = 300.0
    auto_load_from_db: bool = True
    auto_discovery_target_games: int = 50000
    # RR-PARITY-FIX-2025-12-21: Exclude databases with known parity failures
    # See app.training.parity_exclusions for the full list and documentation
    excluded_db_patterns: tuple[str, ...] | None = None  # Uses EXCLUDED_DB_PATTERNS


@dataclass
class CoordinatorStats:
    """Statistics about the coordinator's operations."""
    total_games_loaded: int = 0
    high_quality_games_loaded: int = 0
    games_synced: int = 0
    games_discovered: int = 0  # From auto-discovery
    discovered_sources: int = 0  # Number of data sources found
    last_sync_time: float = 0.0
    last_load_time: float = 0.0
    last_discovery_time: float = 0.0
    avg_quality_score: float = 0.0
    avg_elo: float = 0.0
    preparation_count: int = 0
    errors: list[str] = field(default_factory=list)


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

    _instance: TrainingDataCoordinator | None = None

    def __init__(
        self,
        config: CoordinatorConfig | None = None,
        selfplay_dir: Path | None = None,
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
        self._hot_buffer: HotDataBuffer | None = None
        self._streaming_pipeline: StreamingDataPipeline | None = None

        # State
        self._initialized = False
        self._last_preparation_time = 0.0

        # Event callbacks (December 2025)
        self._promotion_callbacks: list = []
        self._event_bus_subscription = None

        logger.info(
            f"TrainingDataCoordinator initialized: "
            f"quality_scoring={self._config.enable_quality_scoring}, "
            f"sync={self._config.enable_sync}"
        )

    @classmethod
    def get_instance(
        cls,
        config: CoordinatorConfig | None = None,
    ) -> TrainingDataCoordinator:
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

    def _create_hot_buffer(self) -> HotDataBuffer | None:
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
    ) -> dict[str, Any]:
        """Prepare training data for a training run.

        This method:
        1. Syncs high-quality games from the cluster (if enabled)
        2. Runs auto-discovery for synced data sources (if enabled)
        3. Refreshes quality lookups from manifest
        4. Configures data loaders with quality scores
        5. Optionally loads games from local databases

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
            "games_discovered": 0,
            "discovered_sources": 0,
            "games_loaded": 0,
            "avg_quality": 0.0,
            "duration_seconds": 0.0,
            "data_paths": [],
            "errors": [],
        }

        # Step 1: Sync high-quality games from cluster
        if self._config.enable_sync:
            sync_result = await self._sync_high_quality_data(force=force_sync)
            result["games_synced"] = sync_result.get("games_synced", 0)
            if sync_result.get("errors"):
                result["errors"].extend(sync_result["errors"])

        # Step 2: Auto-discover training data from synced sources
        if self._config.enable_auto_discovery and HAS_AUTO_DISCOVERY:
            discovery = self._run_auto_discovery(board_type, num_players)
            result["games_discovered"] = discovery.get("total_games", 0)
            result["discovered_sources"] = discovery.get("num_sources", 0)
            result["data_paths"] = discovery.get("data_paths", [])
            if discovery.get("avg_quality"):
                result["avg_quality"] = discovery["avg_quality"]

        # Step 3: Refresh quality lookups
        bridge = self._get_quality_bridge()
        if bridge:
            bridge.refresh(force=True)
            # Update avg_quality if we didn't get it from discovery
            if result["avg_quality"] == 0.0:
                result["avg_quality"] = bridge.get_stats().avg_quality_score

            # Reconfigure hot buffer with fresh lookups
            if self._hot_buffer:
                bridge.configure_hot_data_buffer(self._hot_buffer)

        # Step 4: Load games from local databases
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
            f"synced={result['games_synced']}, discovered={result['games_discovered']}, "
            f"loaded={result['games_loaded']}, "
            f"avg_quality={result['avg_quality']:.3f}, "
            f"duration={result['duration_seconds']:.1f}s"
        )

        return result

    def _run_auto_discovery(
        self,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run automatic data discovery for synced sources.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            Dict with discovery results
        """
        result = {
            "total_games": 0,
            "num_sources": 0,
            "avg_quality": 0.0,
            "data_paths": [],
        }

        if not HAS_AUTO_DISCOVERY or not should_auto_discover():
            logger.debug("Auto-discovery not available or not enabled")
            return result

        try:
            discovery = discover_training_data(
                board_type=board_type,
                num_players=num_players,
                target_games=self._config.auto_discovery_target_games,
                min_quality=self._config.min_quality_threshold,
            )

            if discovery.success:
                result["total_games"] = discovery.total_games
                result["num_sources"] = len(discovery.data_paths)
                result["avg_quality"] = discovery.avg_quality_score
                result["data_paths"] = [str(p) for p in discovery.data_paths]

                self._stats.games_discovered = discovery.total_games
                self._stats.discovered_sources = len(discovery.data_paths)
                self._stats.last_discovery_time = time.time()

                logger.info(
                    f"Auto-discovery found {discovery.total_games} games "
                    f"across {len(discovery.data_paths)} sources "
                    f"(avg quality: {discovery.avg_quality_score:.3f})"
                )
            else:
                logger.warning(f"Auto-discovery failed: {discovery.error_message}")

        except Exception as e:
            logger.warning(f"Auto-discovery error: {e}")
            self._stats.errors.append(f"Auto-discovery failed: {e}")

        return result

    def get_discovered_data_paths(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        target_games: int | None = None,
    ) -> list[Path]:
        """Get recommended data paths from auto-discovery.

        Args:
            board_type: Filter by board type
            num_players: Filter by player count
            target_games: Target number of games

        Returns:
            List of recommended data paths
        """
        if not HAS_AUTO_DISCOVERY:
            return []

        return get_best_data_paths(
            target_games=target_games or self._config.auto_discovery_target_games,
            min_quality=self._config.min_quality_threshold,
            board_type=board_type,
            num_players=num_players,
        )

    async def _sync_high_quality_data(self, force: bool = False) -> dict[str, Any]:
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
        """Load games from local selfplay databases.

        Automatically excludes databases matching patterns in
        config.excluded_db_patterns (e.g., databases with known
        parity failures or phase coercion issues).
        """
        if not self._hot_buffer:
            self._create_hot_buffer()

        if not self._hot_buffer:
            return 0

        total_loaded = 0
        skipped_dbs = []

        # Find all relevant databases
        db_pattern = f"*{board_type}*.db"
        for db_path in self._selfplay_dir.glob(db_pattern):
            # RR-PARITY-FIX-2025-12-21: Skip excluded databases
            exclusion_patterns = self._config.excluded_db_patterns or EXCLUDED_DB_PATTERNS
            if should_exclude_database(db_path, exclusion_patterns):
                skipped_dbs.append(db_path.stem)
                continue

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

        if skipped_dbs:
            logger.info(
                f"Skipped {len(skipped_dbs)} non-canonical databases: {skipped_dbs}"
            )

        self._stats.last_load_time = time.time()
        return total_loaded

    # =========================================================================
    # Data Access
    # =========================================================================

    def get_hot_buffer(self) -> HotDataBuffer | None:
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
        min_quality: float | None = None,
        limit: int = 10000,
    ) -> list[str]:
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

    def get_status(self) -> dict[str, Any]:
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

    # =========================================================================
    # Promotion Event Integration (December 2025)
    # =========================================================================

    def on_promotion(self, callback) -> None:
        """Register a callback for promotion events.

        The callback receives a dict with promotion details:
        - model_id: ID of promoted model
        - promotion_type: Type of promotion
        - current_elo: Current Elo rating
        - board_type: Board type
        - num_players: Number of players

        Args:
            callback: Callable[[Dict[str, Any]], None]
        """
        self._promotion_callbacks.append(callback)

    def off_promotion(self, callback) -> None:
        """Unregister a promotion callback."""
        if callback in self._promotion_callbacks:
            self._promotion_callbacks.remove(callback)

    async def handle_promotion_event(self, event_data: dict[str, Any]) -> None:
        """Handle a model promotion event.

        Called when a model is promoted. This can trigger:
        - Hot buffer refresh with new model's data
        - Quality score recalculation
        - Sync priority adjustment

        Args:
            event_data: Promotion event data from PromotionController
        """
        model_id = event_data.get("model_id", "unknown")
        promotion_type = event_data.get("promotion_type", "unknown")

        logger.info(f"Handling promotion event: {model_id} ({promotion_type})")

        # Notify registered callbacks
        for callback in self._promotion_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.warning(f"Promotion callback error: {e}")

        # Refresh quality data if model was promoted to production
        if promotion_type in ("production", "champion"):
            board_type = event_data.get("board_type", "square8")
            num_players = event_data.get("num_players", 2)

            # Update quality thresholds based on new production model
            if self._quality_bridge:
                try:
                    self._quality_bridge.invalidate_cache()
                    logger.debug("Quality cache invalidated after promotion")
                except Exception as e:
                    logger.warning(f"Failed to invalidate quality cache: {e}")

            # Optionally refresh hot buffer
            if self._hot_buffer:
                try:
                    await self.refresh_hot_buffer(board_type, num_players)
                    logger.debug("Hot buffer refreshed after promotion")
                except Exception as e:
                    logger.warning(f"Failed to refresh hot buffer: {e}")

    def subscribe_to_promotion_events(self) -> bool:
        """Subscribe to StageEvent.PROMOTION_COMPLETE events.

        This integrates with the stage event bus to automatically
        receive promotion notifications.

        Returns:
            True if subscription was successful
        """
        if self._event_bus_subscription:
            return True

        subscribed = False

        try:
            # P0.5 (December 2025): Use get_router() instead of deprecated get_stage_event_bus()
            from app.coordination.event_router import (
                StageEvent,
                get_router,
            )

            router = get_router()

            async def on_promotion_complete(result):
                if result.success:
                    await self.handle_promotion_event(result.metadata or {})

            router.subscribe(StageEvent.PROMOTION_COMPLETE, on_promotion_complete)
            self._event_bus_subscription = on_promotion_complete
            logger.info("Subscribed to PROMOTION_COMPLETE events")
            subscribed = True

        except ImportError:
            logger.debug("Stage event bus not available")
        except Exception as e:
            logger.warning(f"Failed to subscribe to promotion events: {e}")

        # Also subscribe to quality events (December 2025)
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus as get_data_event_bus,
            )

            data_bus = get_data_event_bus()

            async def on_low_quality_warning(event):
                """Handle LOW_QUALITY_DATA_WARNING to deprioritize low-quality sources."""
                payload = event.payload if hasattr(event, 'payload') else {}
                config_key = payload.get("config", "")
                quality_ratio = payload.get("quality_ratio", 0.0)

                if config_key and quality_ratio > 0.3:
                    logger.info(
                        f"[DataCoordinator] Low quality warning for {config_key} "
                        f"({quality_ratio:.1%}), deprioritizing data source"
                    )
                    # Track deprioritized sources
                    if not hasattr(self, '_deprioritized_sources'):
                        self._deprioritized_sources = set()
                    self._deprioritized_sources.add(config_key)

            data_bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, on_low_quality_warning)
            logger.info("[DataCoordinator] Subscribed to LOW_QUALITY_DATA_WARNING events")
            subscribed = True

        except ImportError:
            logger.debug("Data event bus not available")
        except Exception as e:
            logger.warning(f"[DataCoordinator] Failed to subscribe to quality events: {e}")

        return subscribed


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_data_coordinator(
    config: CoordinatorConfig | None = None,
) -> TrainingDataCoordinator:
    """Get the singleton TrainingDataCoordinator instance."""
    return TrainingDataCoordinator.get_instance(config)


async def prepare_training_data(
    board_type: str = "square8",
    num_players: int = 2,
    force_sync: bool = False,
) -> dict[str, Any]:
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
) -> list[str]:
    """Convenience function to get high-quality game IDs."""
    coordinator = get_data_coordinator()
    return coordinator.get_high_quality_game_ids(min_quality=min_quality, limit=limit)


def wire_promotion_events(coordinator: TrainingDataCoordinator | None = None) -> bool:
    """Wire promotion events to the data coordinator.

    This function connects the TrainingDataCoordinator to receive
    PROMOTION_COMPLETE events from the stage event bus, enabling
    automatic cache invalidation and data refresh on model promotion.

    Args:
        coordinator: Optional coordinator instance (uses singleton if None)

    Returns:
        True if successfully wired

    Usage:
        from app.training.data_coordinator import wire_promotion_events

        # At startup (e.g., in unified_ai_loop.py)
        wire_promotion_events()
    """
    coordinator = coordinator or get_data_coordinator()
    return coordinator.subscribe_to_promotion_events()


__all__ = [
    "CoordinatorConfig",
    "CoordinatorStats",
    "TrainingDataCoordinator",
    "get_data_coordinator",
    "get_high_quality_games",
    "prepare_training_data",
    "wire_promotion_events",
]
