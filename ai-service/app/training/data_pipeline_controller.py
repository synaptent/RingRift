"""Unified Data Pipeline Controller for RingRift AI.

.. deprecated:: 2025-12
    For new code, prefer :class:`TrainingDataCoordinator` from
    ``app.training.data_coordinator`` which provides quality-aware
    data selection and better integration with cluster sync.

    This module is retained for batch/streaming data loading operations.
    TrainingDataCoordinator wraps these capabilities with quality scoring.

    Migration::

        # Old:
        from app.training.data_pipeline_controller import DataPipelineController
        controller = DataPipelineController(db_paths=[...])
        for batch in controller.get_training_batches():
            ...

        # New (recommended):
        from app.training.data_coordinator import get_data_coordinator
        coordinator = get_data_coordinator()
        for batch in coordinator.get_quality_batches():
            ...

Provides a single entry point for all data operations, consolidating:
- Real-time streaming from SQLite databases (streaming_pipeline.py)
- Batch loading from NPZ/HDF5 files (data_loader.py)
- Data sync and aggregation orchestration (run_data_pipeline.py)

This controller exposes a clean API for the unified AI loop to consume,
handling data preparation, loading, and streaming in a consistent manner.

Usage (legacy):
    from app.training.data_pipeline_controller import DataPipelineController

    # Initialize controller
    controller = DataPipelineController(
        db_paths=["data/games/selfplay.db"],
        npz_paths=["data/training/training_data.npz"],
    )

    # Get training data (automatically selects best source)
    for batch in controller.get_training_batches(batch_size=256):
        train_on_batch(batch)

    # Or use specific pipeline
    async for batch in controller.stream_from_database(batch_size=256):
        train_on_batch(batch)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import warnings

# Emit deprecation warning on import (December 2025)
warnings.warn(
    "data_pipeline_controller is deprecated since 2025-12. "
    "Use app.training.data_coordinator.TrainingDataCoordinator instead. "
    "This module will be removed in 2025-Q2.",
    DeprecationWarning,
    stacklevel=2,
)
import contextlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Import manifest for quality-based data selection
try:
    from app.distributed.unified_manifest import (
        DataManifest,
        GameQualityMetadata,
    )
    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False
    DataManifest = None
    GameQualityMetadata = None

# Import centralized quality thresholds
try:
    from app.quality.thresholds import (
        HIGH_QUALITY_THRESHOLD,
        MIN_QUALITY_FOR_PRIORITY_SYNC,
    )
except ImportError:
    MIN_QUALITY_FOR_PRIORITY_SYNC = 0.5
    HIGH_QUALITY_THRESHOLD = 0.7


class DataSourceType(Enum):
    """Types of data sources supported by the pipeline."""

    DATABASE = "database"  # SQLite game databases
    NPZ = "npz"  # NumPy compressed files
    HDF5 = "hdf5"  # HDF5 files
    STREAMING = "streaming"  # Real-time streaming
    REMOTE = "remote"  # Remote sources (via rsync)
    ARIA2 = "aria2"  # Remote sources (via aria2 multi-connection download)


class PipelineMode(Enum):
    """Operating modes for the data pipeline."""

    BATCH = "batch"  # Load fixed batches from files
    STREAMING = "streaming"  # Real-time streaming from database
    HYBRID = "hybrid"  # Combine batch and streaming


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    source_type: DataSourceType
    path: str
    weight: float = 1.0  # Sampling weight for this source
    board_type: str | None = None
    num_players: int | None = None
    enabled: bool = True
    priority: int = 0  # Higher = preferred when multiple sources available
    # Remote source options (for ARIA2/REMOTE types)
    remote_urls: list[str] | None = None  # URLs for aria2 download
    sync_on_startup: bool = False  # Whether to sync before loading
    # Quality tracking (P4: Data source registry for training)
    avg_quality_score: float = 0.5  # Average quality of games from this source
    total_games_used: int = 0  # Games used in training from this source
    quality_trend: float = 0.0  # Positive = improving quality over time
    last_quality_update: float = 0.0  # Timestamp of last quality update

    def update_quality(self, quality_score: float, alpha: float = 0.1) -> None:
        """Update rolling average quality score.

        Args:
            quality_score: New quality score to incorporate
            alpha: Smoothing factor for exponential moving average
        """
        old_quality = self.avg_quality_score
        self.avg_quality_score = (1 - alpha) * self.avg_quality_score + alpha * quality_score
        self.total_games_used += 1
        self.quality_trend = self.avg_quality_score - old_quality
        self.last_quality_update = time.time()

    @property
    def effective_weight(self) -> float:
        """Get quality-adjusted sampling weight.

        High-quality sources get proportionally more sampling weight.
        """
        # Scale weight by quality (0.5-1.5x multiplier based on quality)
        quality_multiplier = 0.5 + self.avg_quality_score
        return self.weight * quality_multiplier


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    # Operating mode
    mode: PipelineMode = PipelineMode.BATCH

    # Batch settings
    batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = False

    # Streaming settings
    poll_interval_seconds: float = 5.0
    buffer_size: int = 10000
    min_buffer_fill: float = 0.2

    # Sampling settings
    priority_sampling: bool = True
    recency_weight: float = 0.3
    late_game_exponent: float = 2.0

    # Performance settings
    prefetch_count: int = 2
    pin_memory: bool = False
    num_workers: int = 0

    # Filtering
    board_type: str | None = None
    num_players: int | None = None

    # Multi-player value support
    use_multi_player_values: bool = False

    # Data validation settings
    validate_on_load: bool = True  # Validate NPZ files before loading
    validation_sample_rate: float = 1.0  # Fraction of samples to validate (1.0 = all)
    fail_on_validation_error: bool = False  # Raise exception if validation fails
    max_validation_issues: int = 100  # Max issues to report per file

    # Quality-based data selection
    enable_quality_filtering: bool = True  # Use manifest quality scores for selection
    min_quality_score: float = MIN_QUALITY_FOR_PRIORITY_SYNC  # Minimum quality score for training data
    quality_weighted_sampling: bool = True  # Weight samples by quality score
    prefer_high_elo_games: bool = True  # Prioritize high-Elo games
    min_elo_threshold: float = 1400.0  # Minimum average Elo for games
    manifest_db_path: str | None = None  # Path to data manifest for quality lookup


@dataclass
class PipelineStats:
    """Statistics from the data pipeline."""

    total_samples_loaded: int = 0
    total_batches_yielded: int = 0
    total_samples_ingested: int = 0
    active_sources: int = 0
    buffer_size: int = 0
    buffer_capacity: int = 0
    last_batch_time: float | None = None
    avg_batch_load_time_ms: float = 0.0
    source_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Validation stats
    sources_validated: int = 0
    sources_valid: int = 0
    sources_invalid: int = 0
    validation_issues_total: int = 0
    samples_with_issues: int = 0

    # Quality stats
    avg_batch_quality: float = 0.0  # Average quality score of recent batches
    high_quality_ratio: float = 0.0  # % of samples with quality > 0.7
    avg_elo_in_batch: float = 0.0  # Average player Elo in recent batches
    quality_filtered_count: int = 0  # Games filtered out due to low quality

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_samples_loaded": self.total_samples_loaded,
            "total_batches_yielded": self.total_batches_yielded,
            "total_samples_ingested": self.total_samples_ingested,
            "active_sources": self.active_sources,
            "buffer_size": self.buffer_size,
            "buffer_capacity": self.buffer_capacity,
            "last_batch_time": self.last_batch_time,
            "avg_batch_load_time_ms": self.avg_batch_load_time_ms,
            "source_stats": self.source_stats,
            "validation": {
                "sources_validated": self.sources_validated,
                "sources_valid": self.sources_valid,
                "sources_invalid": self.sources_invalid,
                "validation_issues_total": self.validation_issues_total,
                "samples_with_issues": self.samples_with_issues,
            },
            "quality": {
                "avg_batch_quality": self.avg_batch_quality,
                "high_quality_ratio": self.high_quality_ratio,
                "avg_elo_in_batch": self.avg_elo_in_batch,
                "quality_filtered_count": self.quality_filtered_count,
            },
        }


class DataPipelineController:
    """Unified controller for all data pipeline operations.

    Provides a single entry point for:
    - Loading training data from various sources
    - Real-time streaming during training
    - Data synchronization from remote sources
    - Statistics and monitoring
    """

    def __init__(
        self,
        db_paths: list[str] | None = None,
        npz_paths: list[str] | None = None,
        config: PipelineConfig | None = None,
    ):
        """Initialize the data pipeline controller.

        Args:
            db_paths: Paths to SQLite game databases
            npz_paths: Paths to NPZ training data files
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()

        # Data sources
        self._sources: list[DataSourceConfig] = []
        self._db_paths = db_paths or []
        self._npz_paths = npz_paths or []

        # Initialize sources
        self._init_sources()

        # Lazy-loaded pipeline components
        self._streaming_pipeline = None
        self._batch_loader = None
        self._validator = None
        self._is_running = False
        self._lock = threading.RLock()

        # Batch timing
        self._batch_times: list[float] = []

        # Validation results cache
        self._validation_results: dict[str, Any] = {}

        # Quality-based data selection via manifest
        self._manifest: DataManifest | None = None
        self._quality_lookup: dict[str, float] = {}  # game_id -> quality_score
        self._elo_lookup: dict[str, float] = {}  # game_id -> avg_elo
        self._init_manifest()

        logger.info(
            f"DataPipelineController initialized with {len(self._sources)} sources "
            f"(mode={self.config.mode.value}, quality_filtering={self.config.enable_quality_filtering})"
        )

    def _init_manifest(self):
        """Initialize manifest for quality-based data selection."""
        if not HAS_MANIFEST or not self.config.enable_quality_filtering:
            return

        # Try to find manifest database
        manifest_path = self.config.manifest_db_path
        if manifest_path is None:
            # Default locations to search
            candidates = [
                Path("data/data_manifest.db"),
                Path(__file__).parent.parent.parent / "data" / "data_manifest.db",
                Path("/lambda/nfs/RingRift/manifests/data_manifest.db"),
            ]
            for p in candidates:
                if p.exists():
                    manifest_path = str(p)
                    break

        if manifest_path and Path(manifest_path).exists():
            try:
                self._manifest = DataManifest(Path(manifest_path))
                logger.info(f"Loaded data manifest from {manifest_path}")
                # Pre-load quality lookup for configured filters
                self._refresh_quality_lookup()
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
                self._manifest = None

    def _refresh_quality_lookup(self, limit: int = 50000):
        """Refresh the quality lookup table from manifest.

        Args:
            limit: Maximum games to cache in lookup
        """
        if not self._manifest:
            return

        try:
            high_quality_games = self._manifest.get_high_quality_games(
                min_quality_score=self.config.min_quality_score,
                limit=limit,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
            )

            self._quality_lookup.clear()
            self._elo_lookup.clear()

            for game in high_quality_games:
                self._quality_lookup[game.game_id] = game.quality_score
                self._elo_lookup[game.game_id] = game.avg_player_elo

            logger.info(
                f"Refreshed quality lookup: {len(self._quality_lookup)} high-quality games "
                f"(min_score={self.config.min_quality_score})"
            )
        except Exception as e:
            logger.warning(f"Failed to refresh quality lookup: {e}")

    def _init_sources(self):
        """Initialize data sources from paths."""
        # Add database sources
        for i, db_path in enumerate(self._db_paths):
            if os.path.exists(db_path):
                self._sources.append(DataSourceConfig(
                    source_type=DataSourceType.DATABASE,
                    path=db_path,
                    priority=100 - i,  # Earlier paths have higher priority
                ))

        # Add NPZ sources
        for i, npz_path in enumerate(self._npz_paths):
            if os.path.exists(npz_path):
                ext = os.path.splitext(npz_path)[1].lower()
                source_type = DataSourceType.HDF5 if ext in ('.h5', '.hdf5') else DataSourceType.NPZ
                self._sources.append(DataSourceConfig(
                    source_type=source_type,
                    path=npz_path,
                    priority=50 - i,
                ))

        self.stats.active_sources = len([s for s in self._sources if s.enabled])

    def add_source(self, source: DataSourceConfig):
        """Add a data source to the pipeline.

        Args:
            source: Data source configuration
        """
        self._sources.append(source)
        self.stats.active_sources = len([s for s in self._sources if s.enabled])
        logger.info(f"Added data source: {source.source_type.value} at {source.path}")

    def remove_source(self, path: str):
        """Remove a data source by path.

        Args:
            path: Path of the source to remove
        """
        self._sources = [s for s in self._sources if s.path != path]
        self.stats.active_sources = len([s for s in self._sources if s.enabled])

    def get_sources(self) -> list[DataSourceConfig]:
        """Get all configured data sources."""
        return self._sources.copy()

    def get_sources_by_quality(self, min_quality: float = 0.0) -> list[DataSourceConfig]:
        """Get sources sorted by quality score (highest first).

        Args:
            min_quality: Minimum quality threshold (0.0-1.0)

        Returns:
            List of sources sorted by avg_quality_score descending
        """
        qualified = [s for s in self._sources if s.enabled and s.avg_quality_score >= min_quality]
        return sorted(qualified, key=lambda s: s.avg_quality_score, reverse=True)

    def get_source_quality_stats(self) -> dict[str, dict[str, Any]]:
        """Get quality statistics for all sources.

        Returns:
            Dict mapping source path to quality stats
        """
        stats = {}
        for source in self._sources:
            stats[source.path] = {
                "type": source.source_type.value,
                "enabled": source.enabled,
                "avg_quality_score": source.avg_quality_score,
                "total_games_used": source.total_games_used,
                "quality_trend": source.quality_trend,
                "effective_weight": source.effective_weight,
                "last_quality_update": source.last_quality_update,
            }
        return stats

    def update_source_quality(self, path: str, quality_score: float) -> bool:
        """Update quality score for a specific source.

        Args:
            path: Source path to update
            quality_score: New quality score (0.0-1.0)

        Returns:
            True if source was found and updated
        """
        for source in self._sources:
            if source.path == path:
                source.update_quality(quality_score)
                logger.debug(f"Updated quality for {path}: {source.avg_quality_score:.3f}")
                return True
        return False

    def _get_validator(self):
        """Lazy-load the data validator."""
        if self._validator is None:
            try:
                from app.training.unified_data_validator import (
                    DataValidator,
                    DataValidatorConfig,
                )

                validator_config = DataValidatorConfig(
                    max_issues_to_report=self.config.max_validation_issues,
                    sample_rate=self.config.validation_sample_rate,
                )
                self._validator = DataValidator(validator_config)
            except ImportError as e:
                logger.warning(f"Data validation module not available: {e}")
                return None

        return self._validator

    def validate_source(self, source_path: str) -> dict[str, Any] | None:
        """Validate a single data source file.

        Args:
            source_path: Path to NPZ/HDF5 file to validate

        Returns:
            Dict with validation results, or None if validation not available
        """
        validator = self._get_validator()
        if validator is None:
            return None

        if not os.path.exists(source_path):
            logger.warning(f"Source file not found: {source_path}")
            return {"valid": False, "error": "File not found"}

        try:
            from app.training.unified_data_validator import record_validation_metrics

            result = validator.validate_npz(Path(source_path))

            # Cache the result
            self._validation_results[source_path] = {
                "valid": result.valid,
                "total_samples": result.total_samples,
                "samples_with_issues": result.samples_with_issues,
                "issue_count": len(result.issues),
                "issues_by_type": self._count_issues_by_type(result.issues),
                "value_stats": result.value_stats,
            }

            # Update stats
            self.stats.sources_validated += 1
            if result.valid:
                self.stats.sources_valid += 1
            else:
                self.stats.sources_invalid += 1
            self.stats.validation_issues_total += len(result.issues)
            self.stats.samples_with_issues += result.samples_with_issues

            # Record to Prometheus if available
            with contextlib.suppress(Exception):
                record_validation_metrics(result)

            # Log summary
            if result.valid:
                logger.info(f"✓ Validated {source_path}: {result.total_samples} samples OK")
            else:
                logger.warning(
                    f"✗ Validation issues in {source_path}: "
                    f"{len(result.issues)} issues in {result.samples_with_issues}/{result.total_samples} samples"
                )

            return self._validation_results[source_path]

        except Exception as e:
            logger.error(f"Validation failed for {source_path}: {e}")
            return {"valid": False, "error": str(e)}

    def _count_issues_by_type(self, issues) -> dict[str, int]:
        """Count validation issues by type."""
        counts = {}
        for issue in issues:
            key = issue.issue_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def validate_all_sources(self, fail_fast: bool = False) -> dict[str, Any]:
        """Validate all NPZ/HDF5 data sources.

        Runs validation on all file-based sources before training.
        Recommended to call this before starting a training run.

        Args:
            fail_fast: If True, stop on first invalid source

        Returns:
            Dict with overall validation summary and per-source results
        """
        results = {
            "all_valid": True,
            "sources_checked": 0,
            "sources_valid": 0,
            "sources_invalid": 0,
            "total_issues": 0,
            "source_results": {},
        }

        file_sources = [
            s for s in self._sources
            if s.source_type in (DataSourceType.NPZ, DataSourceType.HDF5)
            and s.enabled
        ]

        if not file_sources:
            logger.info("No file-based sources to validate")
            return results

        logger.info(f"Validating {len(file_sources)} data sources...")

        for source in file_sources:
            result = self.validate_source(source.path)
            results["sources_checked"] += 1

            if result is None:
                # Validation not available
                continue

            results["source_results"][source.path] = result

            if result.get("valid", False):
                results["sources_valid"] += 1
            else:
                results["sources_invalid"] += 1
                results["all_valid"] = False
                results["total_issues"] += result.get("issue_count", 0)

                if fail_fast:
                    logger.error(f"Validation failed for {source.path}, stopping early")
                    break

        # Summary log
        if results["all_valid"]:
            logger.info(
                f"✓ All {results['sources_valid']} data sources validated successfully"
            )
        else:
            logger.warning(
                f"✗ Validation complete: {results['sources_valid']}/{results['sources_checked']} valid, "
                f"{results['total_issues']} total issues"
            )

        return results

    def get_validation_results(self) -> dict[str, Any]:
        """Get cached validation results.

        Returns:
            Dict mapping source paths to their validation results
        """
        return self._validation_results.copy()

    async def sync_remote_sources(
        self,
        source_urls: list[str] | None = None,
        categories: list[str] | None = None,
        max_age_hours: float = 168,
    ) -> dict[str, int]:
        """Sync data from remote sources using SyncCoordinator.

        This method uses the unified SyncCoordinator which provides:
        - aria2 high-performance multi-connection downloads
        - SSH fallback for direct transfers
        - P2P fallback for decentralized sync
        - NFS optimization (skips sync when shared storage available)
        - Gossip sync for eventually-consistent replication

        Args:
            source_urls: List of data server URLs (e.g., ["http://host1:8766"])
                        If None, auto-discovers from cluster
            categories: Data categories to sync (default: ["games", "training"])
            max_age_hours: Only sync files newer than this

        Returns:
            Dict mapping category to number of files synced
        """
        # Try SyncCoordinator first (preferred)
        try:
            from app.distributed.sync_coordinator import (
                SyncCategory,
                SyncCoordinator,
            )
            coordinator = SyncCoordinator.get_instance()
            results = {}

            if categories is None:
                categories = ["games", "training"]

            # Map string categories to SyncCategory enum
            category_map = {
                "games": SyncCategory.GAMES,
                "training": SyncCategory.TRAINING,
                "training_data": SyncCategory.TRAINING,
                "models": SyncCategory.MODELS,
            }

            for cat_name in categories:
                cat_enum = category_map.get(cat_name)
                if cat_enum is None:
                    continue

                if cat_enum == SyncCategory.GAMES:
                    stats = await coordinator.sync_games(sources=source_urls)
                elif cat_enum == SyncCategory.TRAINING:
                    stats = await coordinator.sync_training_data(
                        sources=source_urls,
                        max_age_hours=max_age_hours,
                    )
                elif cat_enum == SyncCategory.MODELS:
                    stats = await coordinator.sync_models(sources=source_urls)
                else:
                    continue

                results[cat_name] = stats.files_synced
                if stats.files_synced > 0:
                    logger.info(
                        f"Synced {stats.files_synced} {cat_name} files "
                        f"({stats.bytes_transferred / (1024*1024):.1f}MB) "
                        f"via {stats.transport_used}"
                    )

            return results

        except ImportError:
            pass  # Fall back to direct aria2

        # Fallback: Direct aria2 transport
        try:
            from app.distributed.aria2_transport import Aria2Transport, check_aria2_available
        except ImportError:
            logger.warning("Neither SyncCoordinator nor aria2_transport available")
            return {}

        if not check_aria2_available():
            logger.warning("aria2c not available, skipping remote sync")
            return {}

        # Collect source URLs from config if not provided
        if source_urls is None:
            source_urls = []
            for source in self._sources:
                if source.remote_urls:
                    source_urls.extend(source.remote_urls)

        if not source_urls:
            logger.debug("No remote source URLs configured")
            return {}

        if categories is None:
            categories = ["games", "training"]

        transport = Aria2Transport()
        results = {}

        try:
            local_dir = Path(os.path.dirname(self._db_paths[0]) if self._db_paths else "data/games")
            sync_results = await transport.full_cluster_sync(
                source_urls=source_urls,
                local_dir=local_dir.parent,
                categories=categories,
            )

            for category, result in sync_results.items():
                results[category] = result.files_synced
                if result.files_synced > 0:
                    logger.info(
                        f"Synced {result.files_synced} {category} files "
                        f"({result.bytes_transferred / (1024*1024):.1f}MB)"
                    )

        except Exception as e:
            logger.error(f"Remote sync failed: {e}")

        finally:
            await transport.close()

        return results

    def add_aria2_source(
        self,
        local_path: str,
        remote_urls: list[str],
        sync_on_startup: bool = True,
        priority: int = 80,
    ):
        """Add an aria2-backed remote data source.

        Args:
            local_path: Local path where synced data will be stored
            remote_urls: List of remote data server URLs
            sync_on_startup: Whether to sync data before loading
            priority: Source priority (higher = preferred)
        """
        source = DataSourceConfig(
            source_type=DataSourceType.ARIA2,
            path=local_path,
            remote_urls=remote_urls,
            sync_on_startup=sync_on_startup,
            priority=priority,
        )
        self.add_source(source)
        logger.info(f"Added aria2 source: {local_path} from {len(remote_urls)} remotes")

    def _get_streaming_pipeline(self):
        """Lazy-load the streaming pipeline."""
        if self._streaming_pipeline is None:
            try:
                from app.training.streaming_pipeline import (
                    MultiDBStreamingPipeline,
                    StreamingConfig,
                    StreamingDataPipeline,
                )

                db_sources = [
                    s for s in self._sources
                    if s.source_type == DataSourceType.DATABASE and s.enabled
                ]

                if not db_sources:
                    logger.warning("No database sources available for streaming")
                    return None

                config = StreamingConfig(
                    poll_interval_seconds=self.config.poll_interval_seconds,
                    buffer_size=self.config.buffer_size,
                    min_buffer_fill=self.config.min_buffer_fill,
                    priority_sampling=self.config.priority_sampling,
                    recency_weight=self.config.recency_weight,
                    # Pass quality data for weighted sampling
                    quality_lookup=self._quality_lookup if self.config.quality_weighted_sampling else None,
                    elo_lookup=self._elo_lookup if self.config.prefer_high_elo_games else None,
                )

                if len(db_sources) == 1:
                    self._streaming_pipeline = StreamingDataPipeline(
                        db_path=Path(db_sources[0].path),
                        board_type=self.config.board_type,
                        num_players=self.config.num_players,
                        config=config,
                    )
                else:
                    self._streaming_pipeline = MultiDBStreamingPipeline(
                        db_paths=[Path(s.path) for s in db_sources],
                        board_type=self.config.board_type,
                        num_players=self.config.num_players,
                        config=config,
                    )

                self.stats.buffer_capacity = self.config.buffer_size

                # Log quality integration status
                if self._quality_lookup:
                    logger.info(
                        f"Streaming pipeline initialized with quality weighting "
                        f"({len(self._quality_lookup)} games in quality lookup)"
                    )

            except ImportError as e:
                logger.error(f"Failed to import streaming pipeline: {e}")
                return None

        return self._streaming_pipeline

    def _get_batch_loader(self):
        """Lazy-load the batch data loader."""
        if self._batch_loader is None:
            try:
                from app.training.data_loader import (
                    StreamingDataLoader,
                    WeightedStreamingDataLoader,
                )

                npz_sources = [
                    s for s in self._sources
                    if s.source_type in (DataSourceType.NPZ, DataSourceType.HDF5)
                    and s.enabled
                ]

                if not npz_sources:
                    logger.warning("No NPZ/HDF5 sources available for batch loading")
                    return None

                data_paths = [s.path for s in npz_sources]

                # Validate sources before loading if configured
                if self.config.validate_on_load:
                    validation_results = self.validate_all_sources(
                        fail_fast=self.config.fail_on_validation_error
                    )

                    if not validation_results["all_valid"]:
                        if self.config.fail_on_validation_error:
                            raise ValueError(
                                f"Data validation failed: {validation_results['sources_invalid']} "
                                f"sources have issues ({validation_results['total_issues']} total issues)"
                            )
                        else:
                            logger.warning(
                                "Proceeding with training despite validation issues. "
                                "Set fail_on_validation_error=True to enforce validation."
                            )

                # Use weighted loader if late-game bias is configured
                if self.config.late_game_exponent != 1.0:
                    self._batch_loader = WeightedStreamingDataLoader(
                        data_paths=data_paths,
                        batch_size=self.config.batch_size,
                        shuffle=self.config.shuffle,
                        drop_last=self.config.drop_last,
                        sampling_weights='late_game',
                        late_game_exponent=self.config.late_game_exponent,
                    )
                else:
                    self._batch_loader = StreamingDataLoader(
                        data_paths=data_paths,
                        batch_size=self.config.batch_size,
                        shuffle=self.config.shuffle,
                        drop_last=self.config.drop_last,
                    )

            except ImportError as e:
                logger.error(f"Failed to import data loader: {e}")
                return None

        return self._batch_loader

    def get_training_batches(
        self,
        batch_size: int | None = None,
        max_batches: int | None = None,
    ) -> Iterator[tuple[Any, Any]]:
        """Get training batches from the best available source.

        Automatically selects the appropriate data source based on
        configuration and availability.

        Args:
            batch_size: Override batch size
            max_batches: Maximum number of batches to yield

        Yields:
            Tuples of (inputs, targets) for training
        """
        batch_size = batch_size or self.config.batch_size
        batches_yielded = 0

        if self.config.mode == PipelineMode.BATCH:
            # Use batch loader
            loader = self._get_batch_loader()
            if loader is None:
                logger.error("No batch loader available")
                return

            # Optionally wrap with prefetching
            if self.config.prefetch_count > 0:
                try:
                    from app.training.data_loader import prefetch_loader
                    source = prefetch_loader(
                        loader,
                        prefetch_count=self.config.prefetch_count,
                        pin_memory=self.config.pin_memory,
                        use_mp=self.config.use_multi_player_values,
                    )
                except ImportError:
                    source = iter(loader)
            else:
                source = iter(loader)

            for batch in source:
                start_time = time.time()
                yield batch

                # Track timing
                elapsed_ms = (time.time() - start_time) * 1000
                self._batch_times.append(elapsed_ms)
                if len(self._batch_times) > 100:
                    self._batch_times = self._batch_times[-100:]

                self.stats.total_batches_yielded += 1
                self.stats.last_batch_time = time.time()
                self.stats.avg_batch_load_time_ms = np.mean(self._batch_times)

                batches_yielded += 1
                if max_batches and batches_yielded >= max_batches:
                    break

        elif self.config.mode == PipelineMode.STREAMING:
            # Use streaming via asyncio bridge
            logger.warning(
                "Streaming mode requested but sync iterator called. "
                "Use stream_from_database() for async streaming."
            )
            return

        else:
            # Hybrid mode - combine batch and streaming
            logger.info("Hybrid mode not yet fully implemented, using batch mode")
            yield from self.get_training_batches(batch_size, max_batches)

    async def stream_from_database(
        self,
        batch_size: int | None = None,
        max_batches: int | None = None,
    ) -> AsyncIterator[Any]:
        """Stream training batches from database in real-time.

        Args:
            batch_size: Override batch size
            max_batches: Maximum number of batches to yield

        Yields:
            Batches of GameSample objects
        """
        batch_size = batch_size or self.config.batch_size

        pipeline = self._get_streaming_pipeline()
        if pipeline is None:
            logger.error("No streaming pipeline available")
            return

        self._is_running = True

        try:
            async for batch in pipeline.stream_batches(
                batch_size=batch_size,
                max_batches=max_batches,
            ):
                if not self._is_running:
                    break

                self.stats.total_batches_yielded += 1
                self.stats.last_batch_time = time.time()

                # Update buffer stats
                if hasattr(pipeline, 'buffer'):
                    self.stats.buffer_size = len(pipeline.buffer)

                yield batch

        finally:
            self._is_running = False

    async def start_streaming(self):
        """Start the streaming pipeline in the background."""
        pipeline = self._get_streaming_pipeline()
        if pipeline is not None:
            await pipeline.start()
            self._is_running = True
            logger.info("Streaming pipeline started")

    async def stop_streaming(self):
        """Stop the streaming pipeline."""
        if self._streaming_pipeline is not None:
            await self._streaming_pipeline.stop()
            self._is_running = False
            logger.info("Streaming pipeline stopped")

    def get_sample_count(self) -> int:
        """Get total sample count across all sources."""
        total = 0

        for source in self._sources:
            if not source.enabled:
                continue

            if source.source_type in (DataSourceType.NPZ, DataSourceType.HDF5):
                try:
                    from app.training.data_loader import get_sample_count
                    total += get_sample_count(source.path)
                except Exception as e:
                    logger.warning(f"Failed to count samples in {source.path}: {e}")

            elif source.source_type == DataSourceType.DATABASE:
                try:
                    import sqlite3
                    with sqlite3.connect(source.path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM games WHERE status = 'completed'")
                        count = cursor.fetchone()[0]
                        total += count
                except Exception as e:
                    logger.warning(f"Failed to count games in {source.path}: {e}")

        return total

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        # Update streaming stats if available
        if self._streaming_pipeline is not None:
            if hasattr(self._streaming_pipeline, 'get_stats'):
                stream_stats = self._streaming_pipeline.get_stats()
                self.stats.buffer_size = stream_stats.get('buffer_size', 0)
                self.stats.total_samples_ingested = stream_stats.get('total_samples_ingested', 0)
            elif hasattr(self._streaming_pipeline, 'get_aggregate_stats'):
                stream_stats = self._streaming_pipeline.get_aggregate_stats()
                self.stats.buffer_size = stream_stats.get('total_buffer_size', 0)
                self.stats.total_samples_ingested = stream_stats.get('total_samples_ingested', 0)

        # Update batch loader stats
        if self._batch_loader is not None:
            self.stats.total_samples_loaded = getattr(
                self._batch_loader, 'total_samples', 0
            )

        return self.stats

    def get_sync_status(self) -> dict[str, Any]:
        """Get comprehensive sync status from SyncCoordinator.

        Returns:
            Dict with sync status including:
            - storage_provider: Type of storage (nfs, ephemeral, local)
            - shared_storage: Whether shared NFS storage is available
            - sync_stats: Recent sync operation statistics
            - transports: Available transport methods
            - gossip_status: Gossip sync daemon status (if enabled)
        """
        try:
            from app.distributed.sync_coordinator import SyncCoordinator

            coordinator = SyncCoordinator.get_instance()
            return coordinator.get_status()
        except ImportError:
            return {"error": "SyncCoordinator not available"}
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Quality-Based Data Selection
    # =========================================================================

    def get_high_quality_game_ids(
        self,
        min_quality: float | None = None,
        min_elo: float | None = None,
        limit: int = 10000,
    ) -> list[str]:
        """Get list of high-quality game IDs from manifest.

        Args:
            min_quality: Minimum quality score (default: config value)
            min_elo: Minimum average Elo (default: config value)
            limit: Maximum number of game IDs to return

        Returns:
            List of game IDs sorted by quality score descending
        """
        if not self._manifest:
            return []

        min_quality = min_quality or self.config.min_quality_score
        min_elo = min_elo or self.config.min_elo_threshold

        try:
            games = self._manifest.get_high_quality_games(
                min_quality_score=min_quality,
                limit=limit,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
            )

            # Apply Elo filter
            if self.config.prefer_high_elo_games:
                games = [g for g in games if g.avg_player_elo >= min_elo]

            return [g.game_id for g in games]
        except Exception as e:
            logger.warning(f"Failed to get high-quality game IDs: {e}")
            return []

    def get_quality_weights(self, game_ids: list[str]) -> dict[str, float]:
        """Get quality-based sampling weights for a list of game IDs.

        Weights are normalized so they sum to 1.0.

        Args:
            game_ids: List of game IDs to get weights for

        Returns:
            Dict mapping game_id to sampling weight
        """
        if not self._quality_lookup or not self.config.quality_weighted_sampling:
            # Uniform weights
            n = len(game_ids)
            return dict.fromkeys(game_ids, 1.0 / n) if n > 0 else {}

        weights = {}
        total = 0.0

        for gid in game_ids:
            # Use quality score as weight, with minimum floor
            quality = self._quality_lookup.get(gid, 0.5)
            weight = max(quality, 0.1)  # Minimum weight to ensure coverage
            weights[gid] = weight
            total += weight

        # Normalize
        if total > 0:
            for gid in weights:
                weights[gid] /= total

        return weights

    def get_quality_distribution(self) -> dict[str, Any]:
        """Get quality distribution statistics from manifest.

        Returns:
            Dict with quality distribution stats
        """
        if not self._manifest:
            return {"error": "Manifest not available"}

        try:
            return self._manifest.get_quality_distribution(
                board_type=self.config.board_type,
                num_players=self.config.num_players,
            )
        except Exception as e:
            return {"error": str(e)}

    def is_high_quality_game(self, game_id: str) -> bool:
        """Check if a game is in the high-quality set.

        Args:
            game_id: Game ID to check

        Returns:
            True if game is high quality, False otherwise
        """
        return game_id in self._quality_lookup

    def get_game_quality(self, game_id: str) -> float | None:
        """Get quality score for a specific game.

        Args:
            game_id: Game ID to look up

        Returns:
            Quality score or None if not found
        """
        return self._quality_lookup.get(game_id)

    def get_game_elo(self, game_id: str) -> float | None:
        """Get average Elo for a specific game.

        Args:
            game_id: Game ID to look up

        Returns:
            Average player Elo or None if not found
        """
        return self._elo_lookup.get(game_id)

    def build_priority_weights_for_streaming(self) -> dict[str, float]:
        """Build a priority weight lookup for streaming pipeline.

        Returns quality scores that can be used as priority weights
        in the streaming pipeline for prioritized sampling.

        Returns:
            Dict mapping game_id to priority weight
        """
        # Return quality scores directly as priority weights
        return dict(self._quality_lookup)

    def refresh_quality_data(self):
        """Refresh quality data from manifest.

        Call this periodically to pick up newly synced high-quality games.
        """
        self._refresh_quality_lookup()

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling.

        Args:
            epoch: Current epoch number
        """
        if self._batch_loader is not None:
            self._batch_loader.set_epoch(epoch)

    def reset(self):
        """Reset the pipeline state."""
        self.stats = PipelineStats(active_sources=len([s for s in self._sources if s.enabled]))
        self._batch_times.clear()

        if self._streaming_pipeline is not None and hasattr(self._streaming_pipeline, 'reset'):
            self._streaming_pipeline.reset()

    def close(self):
        """Close all data sources and release resources."""
        if self._batch_loader is not None:
            self._batch_loader.close()
            self._batch_loader = None

        if self._streaming_pipeline is not None and self._is_running:
            # Schedule stop in event loop if running
            try:
                # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.stop_streaming())
            except RuntimeError:
                # No running loop - create one
                try:
                    asyncio.run(self.stop_streaming())
                except (RuntimeError, OSError, asyncio.CancelledError) as e:
                    # Dec 2025: Cleanup errors during shutdown are non-fatal
                    logger.debug(f"[DataPipelineController] Cleanup error during close: {e}")

        self._streaming_pipeline = None
        logger.info("DataPipelineController closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()


def create_pipeline_from_config(
    config_path: str | None = None,
    **overrides,
) -> DataPipelineController:
    """Create a DataPipelineController from configuration file.

    Args:
        config_path: Path to YAML configuration file
        **overrides: Override specific configuration values

    Returns:
        Configured DataPipelineController instance
    """
    config = PipelineConfig()
    db_paths = []
    npz_paths = []

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            # Load data paths
            data_cfg = cfg.get('data', {})
            db_paths = data_cfg.get('db_paths', [])
            npz_paths = data_cfg.get('npz_paths', [])

            # Load pipeline config
            pipeline_cfg = cfg.get('pipeline', {})
            if 'mode' in pipeline_cfg:
                config.mode = PipelineMode(pipeline_cfg['mode'])
            if 'batch_size' in pipeline_cfg:
                config.batch_size = pipeline_cfg['batch_size']
            if 'shuffle' in pipeline_cfg:
                config.shuffle = pipeline_cfg['shuffle']
            if 'prefetch_count' in pipeline_cfg:
                config.prefetch_count = pipeline_cfg['prefetch_count']

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif key == 'db_paths':
            db_paths = value
        elif key == 'npz_paths':
            npz_paths = value

    return DataPipelineController(
        db_paths=db_paths,
        npz_paths=npz_paths,
        config=config,
    )


# Convenience function for backward compatibility
def get_training_data_loader(
    data_paths: list[str],
    batch_size: int = 256,
    shuffle: bool = True,
    **kwargs,
) -> DataPipelineController:
    """Create a data pipeline for training.

    This is a convenience function that provides backward compatibility
    with older code expecting a simple data loader interface.

    Args:
        data_paths: Paths to training data files
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        **kwargs: Additional configuration options

    Returns:
        DataPipelineController configured for batch loading
    """
    # Categorize paths by type
    db_paths = []
    npz_paths = []

    for path in data_paths:
        if not os.path.exists(path):
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext == '.db':
            db_paths.append(path)
        elif ext in ('.npz', '.npy', '.h5', '.hdf5'):
            npz_paths.append(path)

    config = PipelineConfig(
        mode=PipelineMode.BATCH,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return DataPipelineController(
        db_paths=db_paths,
        npz_paths=npz_paths,
        config=config,
    )
