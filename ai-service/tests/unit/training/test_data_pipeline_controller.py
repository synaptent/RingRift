"""Unit tests for app.training.data_pipeline_controller module.

Note: This module is deprecated (Dec 2025) in favor of TrainingDataCoordinator.
Tests are included to ensure backward compatibility until removal (Q2 2026).

Tests cover:
- DataSourceType and PipelineMode enums
- DataSourceConfig dataclass and quality tracking
- PipelineConfig dataclass
- PipelineStats dataclass and serialization
- DataPipelineController class (core functionality)
- Helper functions for pipeline creation
"""

import pytest
import tempfile
import os
import sqlite3
import numpy as np
import time
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Filter deprecation warning during import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from app.training.data_pipeline_controller import (
        # Enums
        DataSourceType,
        PipelineMode,
        # Dataclasses
        DataSourceConfig,
        PipelineConfig,
        PipelineStats,
        # Main class
        DataPipelineController,
        # Helper functions
        create_pipeline_from_config,
        get_training_data_loader,
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with game data."""
    db_path = tmp_path / "test_games.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create games table
    cursor.execute("""
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            status TEXT
        )
    """)

    # Insert test games
    for i in range(10):
        cursor.execute(
            "INSERT INTO games VALUES (?, ?, ?, ?)",
            (f"game_{i}", "hex8", 2, "completed")
        )

    conn.commit()
    conn.close()

    return str(db_path)


@pytest.fixture
def temp_npz(tmp_path):
    """Create a temporary NPZ file with training data."""
    npz_path = tmp_path / "test_training.npz"

    # Create minimal training data
    features = np.random.randn(100, 10, 10, 5).astype(np.float32)
    policy = np.random.randn(100, 64).astype(np.float32)
    value = np.random.randn(100, 2).astype(np.float32)

    np.savez(
        str(npz_path),
        features=features,
        policy=policy,
        value=value,
    )

    return str(npz_path)


@pytest.fixture
def pipeline_config():
    """Create a basic PipelineConfig."""
    return PipelineConfig(
        batch_size=32,
        shuffle=True,
    )


# =============================================================================
# Tests for Enums
# =============================================================================

class TestDataSourceType:
    """Tests for DataSourceType enum."""

    def test_all_types_exist(self):
        """Verify all data source types are defined."""
        assert DataSourceType.DATABASE.value == "database"
        assert DataSourceType.NPZ.value == "npz"
        assert DataSourceType.HDF5.value == "hdf5"
        assert DataSourceType.STREAMING.value == "streaming"
        assert DataSourceType.REMOTE.value == "remote"
        assert DataSourceType.ARIA2.value == "aria2"

    def test_source_type_count(self):
        """Should have 6 source types."""
        assert len(DataSourceType) == 6


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_all_modes_exist(self):
        """Verify all pipeline modes are defined."""
        assert PipelineMode.BATCH.value == "batch"
        assert PipelineMode.STREAMING.value == "streaming"
        assert PipelineMode.HYBRID.value == "hybrid"

    def test_mode_count(self):
        """Should have 3 pipeline modes."""
        assert len(PipelineMode) == 3


# =============================================================================
# Tests for DataSourceConfig
# =============================================================================

class TestDataSourceConfig:
    """Tests for DataSourceConfig dataclass."""

    def test_basic_creation(self):
        """Can create DataSourceConfig with required fields."""
        config = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path/to/db.db",
        )
        assert config.source_type == DataSourceType.DATABASE
        assert config.path == "/path/to/db.db"
        assert config.weight == 1.0
        assert config.enabled is True

    def test_default_values(self):
        """Verify default values are set correctly."""
        config = DataSourceConfig(
            source_type=DataSourceType.NPZ,
            path="/path/to/data.npz",
        )
        assert config.weight == 1.0
        assert config.board_type is None
        assert config.num_players is None
        assert config.enabled is True
        assert config.priority == 0
        assert config.remote_urls is None
        assert config.sync_on_startup is False
        assert config.avg_quality_score == 0.5
        assert config.total_games_used == 0
        assert config.quality_trend == 0.0

    def test_update_quality(self):
        """update_quality should update rolling average."""
        config = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path/to/db.db",
            avg_quality_score=0.5,
        )

        # Update with high quality score
        config.update_quality(0.8, alpha=0.5)

        # Average should move toward 0.8
        assert config.avg_quality_score > 0.5
        assert config.avg_quality_score < 0.8
        assert config.total_games_used == 1
        assert config.quality_trend > 0

    def test_update_quality_trend(self):
        """Quality trend should track direction of change."""
        config = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path/to/db.db",
            avg_quality_score=0.7,
        )

        # Update with lower quality
        config.update_quality(0.3, alpha=0.5)

        # Trend should be negative
        assert config.quality_trend < 0

    def test_effective_weight(self):
        """effective_weight should scale with quality."""
        config = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path/to/db.db",
            weight=1.0,
            avg_quality_score=0.5,
        )

        # Effective weight at 0.5 quality: 1.0 * (0.5 + 0.5) = 1.0
        assert config.effective_weight == 1.0

        # High quality increases weight
        config.avg_quality_score = 0.9
        assert config.effective_weight > 1.0  # 1.0 * 1.4 = 1.4

        # Low quality decreases weight
        config.avg_quality_score = 0.1
        assert config.effective_weight < 1.0  # 1.0 * 0.6 = 0.6


# =============================================================================
# Tests for PipelineConfig
# =============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = PipelineConfig()

        assert config.mode == PipelineMode.BATCH
        assert config.batch_size == 256
        assert config.shuffle is True
        assert config.drop_last is False
        assert config.poll_interval_seconds == 5.0
        assert config.buffer_size == 10000
        assert config.prefetch_count == 2
        assert config.pin_memory is True

    def test_validation_settings(self):
        """Verify validation settings defaults."""
        config = PipelineConfig()

        assert config.validate_on_load is True
        assert config.validation_sample_rate == 1.0
        assert config.fail_on_validation_error is False
        assert config.max_validation_issues == 100

    def test_quality_settings(self):
        """Verify quality filtering settings."""
        config = PipelineConfig()

        assert config.enable_quality_filtering is True
        assert config.min_quality_score >= 0.0
        assert config.min_quality_score <= 1.0
        assert config.quality_weighted_sampling is True
        assert config.prefer_high_elo_games is True

    def test_custom_values(self):
        """Can create config with custom values."""
        config = PipelineConfig(
            mode=PipelineMode.STREAMING,
            batch_size=128,
            shuffle=False,
            board_type="hex8",
            num_players=2,
        )

        assert config.mode == PipelineMode.STREAMING
        assert config.batch_size == 128
        assert config.shuffle is False
        assert config.board_type == "hex8"
        assert config.num_players == 2


# =============================================================================
# Tests for PipelineStats
# =============================================================================

class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        """Verify default statistics values."""
        stats = PipelineStats()

        assert stats.total_samples_loaded == 0
        assert stats.total_batches_yielded == 0
        assert stats.active_sources == 0
        assert stats.buffer_size == 0
        assert stats.last_batch_time is None
        assert stats.avg_batch_load_time_ms == 0.0

    def test_validation_stats(self):
        """Verify validation statistics defaults."""
        stats = PipelineStats()

        assert stats.sources_validated == 0
        assert stats.sources_valid == 0
        assert stats.sources_invalid == 0
        assert stats.validation_issues_total == 0

    def test_quality_stats(self):
        """Verify quality statistics defaults."""
        stats = PipelineStats()

        assert stats.avg_batch_quality == 0.0
        assert stats.high_quality_ratio == 0.0
        assert stats.avg_elo_in_batch == 0.0
        assert stats.quality_filtered_count == 0

    def test_to_dict(self):
        """to_dict should serialize all statistics."""
        stats = PipelineStats(
            total_samples_loaded=1000,
            total_batches_yielded=100,
            active_sources=3,
        )

        data = stats.to_dict()

        assert data["total_samples_loaded"] == 1000
        assert data["total_batches_yielded"] == 100
        assert data["active_sources"] == 3
        assert "validation" in data
        assert "quality" in data

    def test_to_dict_nested_structure(self):
        """to_dict should include nested validation and quality dicts."""
        stats = PipelineStats(
            sources_validated=5,
            sources_valid=4,
            avg_batch_quality=0.75,
        )

        data = stats.to_dict()

        assert data["validation"]["sources_validated"] == 5
        assert data["validation"]["sources_valid"] == 4
        assert data["quality"]["avg_batch_quality"] == 0.75


# =============================================================================
# Tests for DataPipelineController
# =============================================================================

class TestDataPipelineControllerInit:
    """Tests for DataPipelineController initialization."""

    def test_basic_init(self):
        """Can initialize with no arguments."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        assert controller.config is not None
        assert controller.stats is not None
        assert len(controller._sources) == 0

    def test_init_with_db_paths(self, temp_db):
        """Can initialize with database paths."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController(db_paths=[temp_db])

        assert len(controller._sources) == 1
        assert controller._sources[0].source_type == DataSourceType.DATABASE

    def test_init_with_npz_paths(self, temp_npz):
        """Can initialize with NPZ paths."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController(npz_paths=[temp_npz])

        assert len(controller._sources) == 1
        assert controller._sources[0].source_type == DataSourceType.NPZ

    def test_init_with_config(self, pipeline_config):
        """Can initialize with custom config."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController(config=pipeline_config)

        assert controller.config.batch_size == 32

    def test_init_ignores_nonexistent_paths(self, tmp_path):
        """Should ignore non-existent file paths."""
        fake_db = str(tmp_path / "nonexistent.db")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController(db_paths=[fake_db])

        assert len(controller._sources) == 0


class TestDataPipelineControllerSources:
    """Tests for source management."""

    def test_add_source(self):
        """Can add a data source."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        source = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/fake/path.db",
        )
        controller.add_source(source)

        assert len(controller._sources) == 1
        assert controller.stats.active_sources == 1

    def test_remove_source(self):
        """Can remove a data source by path."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        source = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/fake/path.db",
        )
        controller.add_source(source)
        controller.remove_source("/fake/path.db")

        assert len(controller._sources) == 0

    def test_get_sources(self):
        """get_sources should return copy of sources list."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        source = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/fake/path.db",
        )
        controller.add_source(source)

        sources = controller.get_sources()
        assert len(sources) == 1

        # Should be a copy
        sources.clear()
        assert len(controller._sources) == 1

    def test_get_sources_by_quality(self):
        """get_sources_by_quality should filter and sort by quality."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        # Add sources with different quality scores
        for i, quality in enumerate([0.3, 0.8, 0.5]):
            source = DataSourceConfig(
                source_type=DataSourceType.DATABASE,
                path=f"/path{i}.db",
                avg_quality_score=quality,
            )
            controller.add_source(source)

        # Get high quality sources
        high_quality = controller.get_sources_by_quality(min_quality=0.4)

        assert len(high_quality) == 2
        # Should be sorted by quality descending
        assert high_quality[0].avg_quality_score >= high_quality[1].avg_quality_score

    def test_update_source_quality(self):
        """Can update quality score for a source."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        source = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path.db",
            avg_quality_score=0.5,
        )
        controller.add_source(source)

        result = controller.update_source_quality("/path.db", 0.9)

        assert result is True
        assert controller._sources[0].avg_quality_score > 0.5

    def test_update_source_quality_not_found(self):
        """update_source_quality should return False for unknown path."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        result = controller.update_source_quality("/nonexistent.db", 0.9)

        assert result is False


class TestDataPipelineControllerStats:
    """Tests for statistics and monitoring."""

    def test_get_stats(self):
        """get_stats should return PipelineStats."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        stats = controller.get_stats()

        assert isinstance(stats, PipelineStats)

    def test_get_source_quality_stats(self):
        """get_source_quality_stats should return per-source stats."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        source = DataSourceConfig(
            source_type=DataSourceType.DATABASE,
            path="/path.db",
            avg_quality_score=0.7,
        )
        controller.add_source(source)

        stats = controller.get_source_quality_stats()

        assert "/path.db" in stats
        assert stats["/path.db"]["avg_quality_score"] == 0.7
        assert stats["/path.db"]["type"] == "database"

    def test_get_sample_count_empty(self):
        """get_sample_count should return 0 with no sources."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        count = controller.get_sample_count()

        assert count == 0

    def test_get_sample_count_with_db(self, temp_db):
        """get_sample_count should count database games."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController(db_paths=[temp_db])

        count = controller.get_sample_count()

        assert count == 10  # 10 games in fixture


class TestDataPipelineControllerQuality:
    """Tests for quality-based data selection."""

    def test_get_quality_weights_uniform(self):
        """get_quality_weights should return uniform weights without manifest."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        game_ids = ["game_1", "game_2", "game_3"]
        weights = controller.get_quality_weights(game_ids)

        # Should be uniform
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_is_high_quality_game_without_manifest(self):
        """is_high_quality_game should return False without manifest."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        result = controller.is_high_quality_game("some_game_id")

        assert result is False

    def test_get_game_quality_without_manifest(self):
        """get_game_quality should return None without manifest."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        result = controller.get_game_quality("some_game_id")

        assert result is None

    def test_get_high_quality_game_ids_without_manifest(self):
        """get_high_quality_game_ids should return empty list without manifest."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        result = controller.get_high_quality_game_ids()

        assert result == []


class TestDataPipelineControllerLifecycle:
    """Tests for controller lifecycle management."""

    def test_reset(self):
        """reset should clear statistics."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        controller.stats.total_batches_yielded = 100
        controller.reset()

        assert controller.stats.total_batches_yielded == 0

    def test_close(self):
        """close should release resources."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        controller.close()

        assert controller._batch_loader is None
        assert controller._streaming_pipeline is None

    def test_context_manager(self):
        """Should work as context manager."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with DataPipelineController() as controller:
                assert controller is not None

        # Should be closed after exit
        assert controller._batch_loader is None

    def test_set_epoch(self):
        """set_epoch should be callable."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        # Should not raise even without batch loader
        controller.set_epoch(5)


class TestDataPipelineControllerAria2:
    """Tests for aria2-backed remote sources."""

    def test_add_aria2_source(self):
        """Can add aria2-backed data source."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        controller.add_aria2_source(
            local_path="/data/synced",
            remote_urls=["http://server1:8766", "http://server2:8766"],
            sync_on_startup=True,
            priority=80,
        )

        assert len(controller._sources) == 1
        source = controller._sources[0]
        assert source.source_type == DataSourceType.ARIA2
        assert len(source.remote_urls) == 2
        assert source.sync_on_startup is True


class TestDataPipelineControllerValidation:
    """Tests for data validation."""

    def test_validate_source_nonexistent(self):
        """validate_source should handle non-existent files."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        result = controller.validate_source("/nonexistent/file.npz")

        if result is not None:  # Validator may not be available
            assert result["valid"] is False
            assert "error" in result

    def test_get_validation_results_empty(self):
        """get_validation_results should return empty dict initially."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        results = controller.get_validation_results()

        assert results == {}


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestCreatePipelineFromConfig:
    """Tests for create_pipeline_from_config function."""

    def test_create_without_config_file(self):
        """Can create pipeline without config file."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = create_pipeline_from_config(batch_size=64)

        assert controller.config.batch_size == 64

    def test_create_with_override(self):
        """Can override config values."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = create_pipeline_from_config(
                shuffle=False,
                prefetch_count=0,
            )

        assert controller.config.shuffle is False
        assert controller.config.prefetch_count == 0


class TestGetTrainingDataLoader:
    """Tests for get_training_data_loader function."""

    def test_basic_creation(self, temp_db, temp_npz):
        """Can create loader with mixed paths."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = get_training_data_loader(
                data_paths=[temp_db, temp_npz],
                batch_size=32,
            )

        assert controller.config.batch_size == 32
        assert controller.config.mode == PipelineMode.BATCH
        # Should have both sources
        assert len(controller._sources) == 2

    def test_ignores_nonexistent_paths(self, tmp_path):
        """Should ignore non-existent paths."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = get_training_data_loader(
                data_paths=[str(tmp_path / "fake.db")],
                batch_size=32,
            )

        assert len(controller._sources) == 0


# =============================================================================
# Tests for Async Method Signatures
# =============================================================================

class TestDataPipelineControllerAsync:
    """Tests for async method signatures (deprecated module, minimal testing)."""

    def test_sync_remote_sources_is_async(self):
        """sync_remote_sources should be an async method."""
        import inspect

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        # Verify the method is async
        assert inspect.iscoroutinefunction(controller.sync_remote_sources)

    def test_start_streaming_is_async(self):
        """start_streaming should be an async method."""
        import inspect

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            controller = DataPipelineController()

        # Verify the method is async
        assert inspect.iscoroutinefunction(controller.start_streaming)


# =============================================================================
# Tests for Deprecation Warning
# =============================================================================

class TestDeprecationWarning:
    """Tests for deprecation warning behavior."""

    def test_deprecation_warning_documented(self):
        """Module docstring should document deprecation."""
        # Instead of reloading (can hang), verify docstring mentions deprecation
        import app.training.data_pipeline_controller as module

        # The module should mention deprecation in its docstring
        assert module.__doc__ is not None
        assert "deprecated" in module.__doc__.lower() or "DEPRECATED" in module.__doc__
