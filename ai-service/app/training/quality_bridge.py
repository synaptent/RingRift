"""Quality Bridge - Unified interface for quality-aware training data selection.

.. deprecated:: Dec 2025
    This module is deprecated and will be consolidated into app.quality by Q2 2026.

    **Migration Guide:**
    - For quality lookups: Use ``app.quality.GameQualityScorer`` which provides
      the same quality scoring with additional features (caching, statistics).
    - For pipeline configuration: The new ``GameQualityScorer`` integrates directly
      with training pipelines via event-driven updates.
    - For quality thresholds: Use ``app.quality.get_quality_thresholds()``.

    **Backward Compatibility:**
    This module will continue to work until Q2 2026. A deprecation warning is emitted
    on import to help track migration progress.

This module provides a unified bridge between the distributed sync system's quality
scoring and the training data loaders. It ensures that quality scores computed during
sync operations flow seamlessly to training data selection.

Data Flow:
    1. Games are synced from cluster nodes via SyncCoordinator
    2. Quality scores are computed by quality_extractor during sync
    3. Scores are stored in unified_manifest (data_manifest.db)
    4. QualityBridge reads scores and provides them to training pipelines:
       - StreamingDataPipeline
       - HotDataBuffer

Usage:
    from app.training.quality_bridge import QualityBridge, get_quality_bridge

    # Get singleton instance
    bridge = get_quality_bridge()

    # Get quality lookups for training
    quality_lookup = bridge.get_quality_lookup()
    elo_lookup = bridge.get_elo_lookup()

    # Set lookups on a streaming pipeline
    bridge.configure_streaming_pipeline(streaming_pipeline)

    # Set lookups on HotDataBuffer
    bridge.configure_hot_data_buffer(hot_buffer)
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Emit deprecation warning on import
warnings.warn(
    "app.training.quality_bridge is deprecated and will be consolidated into "
    "app.quality by Q2 2026. Use app.quality.GameQualityScorer for quality scoring, "
    "and app.quality.get_quality_thresholds() for thresholds.",
    DeprecationWarning,
    stacklevel=2,
)

if TYPE_CHECKING:
    from app.training.hot_data_buffer import HotDataBuffer
    from app.training.streaming_pipeline import StreamingDataPipeline

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        HIGH_QUALITY_THRESHOLD,
        MIN_QUALITY_FOR_TRAINING,
    )
except ImportError:
    MIN_QUALITY_FOR_TRAINING = 0.3
    HIGH_QUALITY_THRESHOLD = 0.7

# Import metrics functions for Prometheus reporting
try:
    from app.metrics.orchestrator import (
        record_training_data_quality,
        update_quality_bridge_status,
    )
    HAS_QUALITY_METRICS = True
except ImportError:
    HAS_QUALITY_METRICS = False
    update_quality_bridge_status = None
    record_training_data_quality = None

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class QualityBridgeConfig:
    """Configuration for the quality bridge.

    Attributes:
        enable_quality_scoring: Enable quality-based data selection
        min_quality_threshold: Minimum quality score for training data
        min_elo_threshold: Minimum average Elo for training data
        quality_weight_in_sampling: Weight for quality in sampling priority
        refresh_interval_seconds: How often to refresh quality data from manifest
        max_lookup_size: Maximum number of games in quality lookup
        prefer_sync_coordinator: Use SyncCoordinator's lookup if available
    """
    enable_quality_scoring: bool = True
    min_quality_threshold: float = MIN_QUALITY_FOR_TRAINING
    min_elo_threshold: float = 1400.0
    quality_weight_in_sampling: float = 0.4
    refresh_interval_seconds: float = 300.0
    max_lookup_size: int = 100000
    prefer_sync_coordinator: bool = True


@dataclass
class QualityStats:
    """Statistics about quality data in the bridge."""
    quality_lookup_size: int = 0
    elo_lookup_size: int = 0
    avg_quality_score: float = 0.0
    avg_elo: float = 0.0
    high_quality_count: int = 0  # Games with quality > 0.7
    last_refresh_time: float = 0.0
    refresh_count: int = 0


class QualityBridge:
    """Unified bridge for quality-aware training data selection.

    This class provides a single interface for all training components to access
    quality scores from the distributed sync system. It handles:

    1. Loading quality data from SyncCoordinator or directly from manifest
    2. Caching and refreshing quality lookups
    3. Configuring training pipelines with quality data
    4. Computing unified quality metrics for monitoring
    """

    _instance: QualityBridge | None = None

    def __init__(
        self,
        config: QualityBridgeConfig | None = None,
        manifest_path: Path | None = None,
    ):
        """Initialize the quality bridge.

        Args:
            config: Bridge configuration
            manifest_path: Optional path to manifest database
        """
        self._config = config or QualityBridgeConfig()
        self._manifest_path = manifest_path

        # Quality lookup tables
        self._quality_lookup: dict[str, float] = {}
        self._elo_lookup: dict[str, float] = {}
        self._last_refresh: float = 0.0
        self._stats = QualityStats()

        # Lazy-loaded dependencies
        self._sync_coordinator = None
        self._manifest = None

        logger.info("QualityBridge initialized")

    @classmethod
    def get_instance(
        cls,
        config: QualityBridgeConfig | None = None,
        manifest_path: Path | None = None,
    ) -> QualityBridge:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config, manifest_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    # =========================================================================
    # Data Source Initialization
    # =========================================================================

    def _get_sync_coordinator(self):
        """Lazily get SyncCoordinator instance."""
        if self._sync_coordinator is None:
            try:
                from app.distributed.sync_coordinator import SyncCoordinator
                self._sync_coordinator = SyncCoordinator.get_instance()
            except Exception as e:
                logger.debug(f"SyncCoordinator not available: {e}")
        return self._sync_coordinator

    def _get_manifest(self):
        """Lazily get DataManifest instance."""
        if self._manifest is None:
            try:
                from app.distributed.unified_manifest import DataManifest

                # Try configured path first
                paths = []
                if self._manifest_path:
                    paths.append(self._manifest_path)

                paths.extend([
                    DEFAULT_DATA_DIR / "data_manifest.db",
                    Path.home() / "ringrift" / "ai-service" / "data" / "data_manifest.db",
                ])

                for path in paths:
                    if path and path.exists():
                        self._manifest = DataManifest(path)
                        logger.debug(f"Loaded manifest from {path}")
                        break
            except Exception as e:
                logger.debug(f"Manifest not available: {e}")
        return self._manifest

    # =========================================================================
    # Quality Lookup Methods
    # =========================================================================

    def refresh(self, force: bool = False) -> int:
        """Refresh quality lookups from data sources.

        Args:
            force: Force refresh even if cache is fresh

        Returns:
            Number of games in quality lookup
        """
        now = time.time()
        cache_ttl = self._config.refresh_interval_seconds

        if not force and (now - self._last_refresh) < cache_ttl:
            return len(self._quality_lookup)

        # Try SyncCoordinator first (preferred - already has cached lookups)
        if self._config.prefer_sync_coordinator:
            coordinator = self._get_sync_coordinator()
            if coordinator:
                try:
                    self._quality_lookup = coordinator.get_quality_lookup(force_refresh=force)
                    self._elo_lookup = coordinator.get_elo_lookup(force_refresh=force)
                    self._last_refresh = now
                    self._update_stats()
                    self._stats.refresh_count += 1
                    logger.debug(
                        f"Refreshed quality from SyncCoordinator: "
                        f"{len(self._quality_lookup)} games"
                    )
                    return len(self._quality_lookup)
                except Exception as e:
                    logger.debug(f"SyncCoordinator refresh failed: {e}")

        # Fall back to direct manifest query
        manifest = self._get_manifest()
        if manifest:
            try:
                high_quality_games = manifest.get_high_quality_games(
                    min_quality_score=0.0,
                    limit=self._config.max_lookup_size,
                )

                self._quality_lookup = {}
                self._elo_lookup = {}

                for game in high_quality_games:
                    self._quality_lookup[game.game_id] = game.quality_score
                    self._elo_lookup[game.game_id] = game.avg_player_elo

                self._last_refresh = now
                self._update_stats()
                self._stats.refresh_count += 1
                logger.debug(
                    f"Refreshed quality from manifest: {len(self._quality_lookup)} games"
                )
                return len(self._quality_lookup)

            except Exception as e:
                logger.warning(f"Manifest refresh failed: {e}")

        return len(self._quality_lookup)

    def _update_stats(self) -> None:
        """Update quality statistics and Prometheus metrics."""
        self._stats.quality_lookup_size = len(self._quality_lookup)
        self._stats.elo_lookup_size = len(self._elo_lookup)
        self._stats.last_refresh_time = self._last_refresh

        min_quality = 0.0
        max_quality = 0.0
        if self._quality_lookup:
            scores = list(self._quality_lookup.values())
            self._stats.avg_quality_score = sum(scores) / len(scores)
            self._stats.high_quality_count = sum(1 for s in scores if s >= HIGH_QUALITY_THRESHOLD)
            min_quality = min(scores) if scores else 0.0
            max_quality = max(scores) if scores else 0.0

        if self._elo_lookup:
            elos = list(self._elo_lookup.values())
            self._stats.avg_elo = sum(elos) / len(elos)

        # Record Prometheus metrics
        if HAS_QUALITY_METRICS:
            try:
                refresh_age = time.time() - self._last_refresh
                update_quality_bridge_status(
                    quality_lookup_size=self._stats.quality_lookup_size,
                    elo_lookup_size=self._stats.elo_lookup_size,
                    refresh_age_seconds=refresh_age,
                    avg_quality=self._stats.avg_quality_score,
                )
                record_training_data_quality(
                    board_type="all",
                    num_players=0,  # 0 = aggregated
                    avg_quality=self._stats.avg_quality_score,
                    min_quality=min_quality,
                    max_quality=max_quality,
                    high_quality_count=self._stats.high_quality_count,
                    total_games=self._stats.quality_lookup_size,
                )
            except Exception as e:
                logger.debug(f"Failed to record quality metrics: {e}")

    def get_quality_lookup(self, auto_refresh: bool = True) -> dict[str, float]:
        """Get quality score lookup dictionary.

        Args:
            auto_refresh: Automatically refresh if cache is stale

        Returns:
            Dict mapping game_id to quality_score
        """
        if auto_refresh:
            self.refresh()
        return self._quality_lookup.copy()

    def get_elo_lookup(self, auto_refresh: bool = True) -> dict[str, float]:
        """Get Elo score lookup dictionary.

        Args:
            auto_refresh: Automatically refresh if cache is stale

        Returns:
            Dict mapping game_id to avg_player_elo
        """
        if auto_refresh:
            self.refresh()
        return self._elo_lookup.copy()

    def get_game_quality(self, game_id: str) -> float | None:
        """Get quality score for a specific game.

        Args:
            game_id: Game identifier

        Returns:
            Quality score or None if not found
        """
        if not self._quality_lookup:
            self.refresh()
        return self._quality_lookup.get(game_id)

    def get_game_elo(self, game_id: str) -> float | None:
        """Get average Elo for a specific game.

        Args:
            game_id: Game identifier

        Returns:
            Average Elo or None if not found
        """
        if not self._elo_lookup:
            self.refresh()
        return self._elo_lookup.get(game_id)

    def is_high_quality(self, game_id: str, threshold: float | None = None) -> bool:
        """Check if a game meets the quality threshold.

        Args:
            game_id: Game identifier
            threshold: Quality threshold (uses config default if None)

        Returns:
            True if game meets quality threshold
        """
        if threshold is None:
            threshold = self._config.min_quality_threshold

        quality = self.get_game_quality(game_id)
        return quality is not None and quality >= threshold

    def get_high_quality_game_ids(
        self,
        min_quality: float | None = None,
        min_elo: float | None = None,
        limit: int = 10000,
    ) -> list[str]:
        """Get list of high-quality game IDs.

        Args:
            min_quality: Minimum quality score (uses config default if None)
            min_elo: Minimum average Elo (uses config default if None)
            limit: Maximum number of games to return

        Returns:
            List of game IDs meeting quality criteria
        """
        if min_quality is None:
            min_quality = self._config.min_quality_threshold
        if min_elo is None:
            min_elo = self._config.min_elo_threshold

        self.refresh()

        # Filter by quality and Elo
        result = []
        for game_id, quality in self._quality_lookup.items():
            if quality >= min_quality:
                elo = self._elo_lookup.get(game_id, 0)
                if elo >= min_elo:
                    result.append((game_id, quality))

        # Sort by quality descending and take top `limit`
        result.sort(key=lambda x: x[1], reverse=True)
        return [gid for gid, _ in result[:limit]]

    # =========================================================================
    # Training Pipeline Configuration
    # =========================================================================

    def configure_streaming_pipeline(
        self,
        pipeline: StreamingDataPipeline,
        quality_weight: float | None = None,
    ) -> int:
        """Configure a StreamingDataPipeline with quality lookups.

        Args:
            pipeline: StreamingDataPipeline instance to configure
            quality_weight: Weight for quality in sampling (uses config default if None)

        Returns:
            Number of games in quality lookup
        """
        if quality_weight is None:
            quality_weight = self._config.quality_weight_in_sampling

        self.refresh()

        try:
            pipeline.set_quality_lookup(
                quality_lookup=self._quality_lookup,
                elo_lookup=self._elo_lookup,
            )
            # Update config if pipeline supports it
            if hasattr(pipeline, 'config') and hasattr(pipeline.config, 'quality_weight'):
                pipeline.config.quality_weight = quality_weight

            logger.info(
                f"Configured StreamingPipeline with {len(self._quality_lookup)} quality scores"
            )
            return len(self._quality_lookup)
        except Exception as e:
            logger.warning(f"Failed to configure StreamingPipeline: {e}")
            return 0

    def configure_hot_data_buffer(
        self,
        buffer: HotDataBuffer,
    ) -> int:
        """Configure a HotDataBuffer with quality lookups.

        Args:
            buffer: HotDataBuffer instance to configure

        Returns:
            Number of games in quality lookup
        """
        self.refresh()

        try:
            count = buffer.set_quality_lookup(
                quality_lookup=self._quality_lookup,
                elo_lookup=self._elo_lookup,
            )
            logger.info(f"Configured HotDataBuffer with {count} quality scores")
            return count
        except Exception as e:
            logger.warning(f"Failed to configure HotDataBuffer: {e}")
            return 0

    def auto_calibrate_hot_buffer(
        self,
        buffer: HotDataBuffer,
        min_quality_percentile: float = 0.1,
        evict_below_percentile: bool = True,
        recompute_quality: bool = False,
    ) -> dict[str, Any]:
        """Auto-calibrate a HotDataBuffer's quality thresholds.

        This method:
        1. Configures the buffer with quality lookups
        2. Optionally recomputes quality scores using UnifiedQualityScorer
        3. Calibrates thresholds based on the quality distribution
        4. Optionally evicts low-quality games

        Args:
            buffer: HotDataBuffer instance to calibrate
            min_quality_percentile: Percentile below which games are low-quality
            evict_below_percentile: Whether to remove low-quality games
            recompute_quality: Whether to recompute quality with UnifiedQualityScorer

        Returns:
            Dict with calibration results:
            - configured: Number of games configured with lookups
            - recomputed: Number of games with recomputed quality (if enabled)
            - distribution: Quality distribution stats
            - calibration: Calibrated threshold values
            - evicted: Number of low-quality games evicted
        """
        result: dict[str, Any] = {
            "configured": 0,
            "recomputed": 0,
            "distribution": {},
            "calibration": {},
            "evicted": 0,
        }

        # First configure with quality lookups
        result["configured"] = self.configure_hot_data_buffer(buffer)

        # Optionally recompute quality using UnifiedQualityScorer
        if recompute_quality:
            try:
                if hasattr(buffer, "recompute_quality_with_scorer"):
                    result["recomputed"] = buffer.recompute_quality_with_scorer()
                    logger.info(f"Recomputed quality for {result['recomputed']} games")
            except Exception as e:
                logger.warning(f"Failed to recompute quality: {e}")

        # Auto-calibrate and optionally filter
        try:
            if hasattr(buffer, "auto_calibrate_and_filter"):
                calibration_result = buffer.auto_calibrate_and_filter(
                    min_quality_percentile=min_quality_percentile,
                    evict_below_percentile=evict_below_percentile,
                )
                result["distribution"] = calibration_result.get("distribution", {})
                result["calibration"] = calibration_result.get("calibration", {})
                result["evicted"] = calibration_result.get("evicted", 0)
                if result['calibration'].get('calibrated'):
                    low_thresh = result['calibration'].get('low_threshold', 0.0)
                    logger.info(
                        f"Auto-calibrated buffer: evicted={result['evicted']}, "
                        f"low_threshold={low_thresh:.2f}"
                    )
                else:
                    logger.info("Auto-calibration skipped (insufficient data)")
        except Exception as e:
            logger.warning(f"Failed to auto-calibrate buffer: {e}")

        return result

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_stats(self) -> QualityStats:
        """Get quality bridge statistics."""
        return self._stats

    def get_status(self) -> dict[str, Any]:
        """Get detailed status information."""
        return {
            "enabled": self._config.enable_quality_scoring,
            "quality_lookup_size": len(self._quality_lookup),
            "elo_lookup_size": len(self._elo_lookup),
            "avg_quality_score": self._stats.avg_quality_score,
            "avg_elo": self._stats.avg_elo,
            "high_quality_count": self._stats.high_quality_count,
            "last_refresh_age_seconds": time.time() - self._last_refresh if self._last_refresh else 0,
            "refresh_count": self._stats.refresh_count,
            "config": {
                "min_quality_threshold": self._config.min_quality_threshold,
                "min_elo_threshold": self._config.min_elo_threshold,
                "quality_weight": self._config.quality_weight_in_sampling,
                "refresh_interval": self._config.refresh_interval_seconds,
            },
            "sources": {
                "sync_coordinator": self._sync_coordinator is not None,
                "manifest": self._manifest is not None,
            },
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

def get_quality_bridge(
    config: QualityBridgeConfig | None = None,
) -> QualityBridge:
    """Get the singleton QualityBridge instance."""
    return QualityBridge.get_instance(config)


def get_quality_lookup() -> dict[str, float]:
    """Get quality lookup dictionary."""
    return get_quality_bridge().get_quality_lookup()


def get_elo_lookup() -> dict[str, float]:
    """Get Elo lookup dictionary."""
    return get_quality_bridge().get_elo_lookup()


def get_game_quality(game_id: str) -> float | None:
    """Get quality score for a specific game."""
    return get_quality_bridge().get_game_quality(game_id)


def is_high_quality_game(game_id: str, threshold: float = HIGH_QUALITY_THRESHOLD) -> bool:
    """Check if a game meets the quality threshold."""
    return get_quality_bridge().is_high_quality(game_id, threshold)


def auto_calibrate_buffer(
    buffer: HotDataBuffer,
    min_quality_percentile: float = 0.1,
    evict_below_percentile: bool = True,
    recompute_quality: bool = False,
) -> dict[str, Any]:
    """Auto-calibrate a HotDataBuffer's quality thresholds (convenience function).

    See QualityBridge.auto_calibrate_hot_buffer for full documentation.
    """
    return get_quality_bridge().auto_calibrate_hot_buffer(
        buffer,
        min_quality_percentile=min_quality_percentile,
        evict_below_percentile=evict_below_percentile,
        recompute_quality=recompute_quality,
    )


__all__ = [
    "QualityBridge",
    "QualityBridgeConfig",
    "QualityStats",
    "auto_calibrate_buffer",
    "get_elo_lookup",
    "get_game_quality",
    "get_quality_bridge",
    "get_quality_lookup",
    "is_high_quality_game",
]
