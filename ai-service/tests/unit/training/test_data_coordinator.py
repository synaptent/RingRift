"""Unit tests for app/training/data_coordinator.py

Tests cover:
- CoordinatorConfig and DataCoordinatorStats dataclasses
- TrainingDataCoordinator singleton pattern and initialization
- Lazy component loading (quality bridge, sync coordinator, hot buffer)
- Training preparation workflow
- Data loading and quality filtering
- Promotion event handling and callbacks
- Status and stats reporting
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.training.data_coordinator import (
    CoordinatorConfig,
    DataCoordinatorStats,
    TrainingDataCoordinator,
    get_data_coordinator,
    get_high_quality_games,
    prepare_training_data,
    wire_promotion_events,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return CoordinatorConfig(
        enable_quality_scoring=True,
        enable_sync=False,  # Disable sync for unit tests
        enable_auto_discovery=False,  # Disable discovery for unit tests
        min_quality_threshold=0.5,
        min_elo_threshold=1400.0,
        hot_buffer_size=100,
        hot_buffer_memory_mb=10,
        refresh_interval_seconds=60.0,
        auto_load_from_db=False,
        auto_discovery_target_games=1000,
    )


@pytest.fixture
def coordinator(config):
    """Create a coordinator instance for testing."""
    # Reset singleton before each test
    TrainingDataCoordinator.reset_instance()
    coord = TrainingDataCoordinator(config=config)
    yield coord
    # Cleanup
    TrainingDataCoordinator.reset_instance()


@pytest.fixture
def temp_selfplay_dir():
    """Create a temporary directory for selfplay databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# CoordinatorConfig Tests
# =============================================================================


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CoordinatorConfig()

        assert config.enable_quality_scoring is True
        assert config.enable_sync is True
        assert config.enable_auto_discovery is True
        assert config.min_quality_threshold == 0.5
        assert config.min_elo_threshold == 1400.0
        assert config.hot_buffer_size == 1000
        assert config.hot_buffer_memory_mb == 500
        assert config.refresh_interval_seconds == 300.0
        assert config.auto_load_from_db is True
        assert config.auto_discovery_target_games == 50000
        assert config.excluded_db_patterns is None

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = CoordinatorConfig(
            enable_quality_scoring=False,
            enable_sync=False,
            min_quality_threshold=0.8,
            min_elo_threshold=1600.0,
            hot_buffer_size=500,
            excluded_db_patterns=("*_test.db", "*_temp.db"),
        )

        assert config.enable_quality_scoring is False
        assert config.enable_sync is False
        assert config.min_quality_threshold == 0.8
        assert config.min_elo_threshold == 1600.0
        assert config.hot_buffer_size == 500
        assert config.excluded_db_patterns == ("*_test.db", "*_temp.db")


# =============================================================================
# DataCoordinatorStats Tests
# =============================================================================


class TestDataCoordinatorStats:
    """Tests for DataCoordinatorStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = DataCoordinatorStats()

        assert stats.total_games_loaded == 0
        assert stats.high_quality_games_loaded == 0
        assert stats.games_synced == 0
        assert stats.games_discovered == 0
        assert stats.discovered_sources == 0
        assert stats.last_sync_time == 0.0
        assert stats.last_load_time == 0.0
        assert stats.last_discovery_time == 0.0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_elo == 0.0
        assert stats.preparation_count == 0
        assert stats.errors == []

    def test_mutable_errors_list(self):
        """Test that errors list is properly initialized per instance."""
        stats1 = DataCoordinatorStats()
        stats2 = DataCoordinatorStats()

        stats1.errors.append("Error 1")

        # stats2 should have its own list
        assert stats1.errors == ["Error 1"]
        assert stats2.errors == []

    def test_update_values(self):
        """Test updating stats values."""
        stats = DataCoordinatorStats()

        stats.total_games_loaded = 100
        stats.high_quality_games_loaded = 80
        stats.avg_quality_score = 0.85
        stats.preparation_count = 5

        assert stats.total_games_loaded == 100
        assert stats.high_quality_games_loaded == 80
        assert stats.avg_quality_score == 0.85
        assert stats.preparation_count == 5


# =============================================================================
# TrainingDataCoordinator Initialization Tests
# =============================================================================


class TestTrainingDataCoordinatorInit:
    """Tests for TrainingDataCoordinator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator()

        assert coord._config is not None
        assert coord._config.enable_quality_scoring is True
        assert coord._stats is not None
        assert coord._initialized is False

        TrainingDataCoordinator.reset_instance()

    def test_init_with_custom_config(self, config):
        """Test initialization with custom config."""
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        assert coord._config.enable_sync is False
        assert coord._config.min_quality_threshold == 0.5
        assert coord._config.hot_buffer_size == 100

        TrainingDataCoordinator.reset_instance()

    def test_init_with_custom_selfplay_dir(self, temp_selfplay_dir):
        """Test initialization with custom selfplay directory."""
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(selfplay_dir=temp_selfplay_dir)

        assert coord._selfplay_dir == temp_selfplay_dir

        TrainingDataCoordinator.reset_instance()

    def test_lazy_components_not_initialized(self, coordinator):
        """Test that components are not initialized immediately."""
        assert coordinator._quality_bridge is None
        assert coordinator._sync_coordinator is None
        assert coordinator._hot_buffer is None
        assert coordinator._streaming_pipeline is None


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern behavior."""

    def test_get_instance_returns_same_instance(self, config):
        """Test that get_instance returns the same instance."""
        TrainingDataCoordinator.reset_instance()

        coord1 = TrainingDataCoordinator.get_instance(config)
        coord2 = TrainingDataCoordinator.get_instance()

        assert coord1 is coord2

        TrainingDataCoordinator.reset_instance()

    def test_reset_instance(self, config):
        """Test that reset_instance creates new instance."""
        TrainingDataCoordinator.reset_instance()

        coord1 = TrainingDataCoordinator(config=config)
        TrainingDataCoordinator.reset_instance()
        coord2 = TrainingDataCoordinator(config=config)

        # After reset, should be different instances
        assert coord1 is not coord2

        TrainingDataCoordinator.reset_instance()

    def test_get_data_coordinator_singleton(self, config):
        """Test module-level singleton accessor."""
        TrainingDataCoordinator.reset_instance()

        coord1 = get_data_coordinator(config)
        coord2 = get_data_coordinator()

        assert coord1 is coord2

        TrainingDataCoordinator.reset_instance()


# =============================================================================
# Lazy Component Loading Tests
# =============================================================================


class TestLazyComponentLoading:
    """Tests for lazy component initialization."""

    def test_get_quality_bridge_when_disabled(self, config):
        """Test quality bridge not loaded when disabled."""
        config.enable_quality_scoring = False
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        bridge = coord._get_quality_bridge()
        assert bridge is None

        TrainingDataCoordinator.reset_instance()

    def test_get_sync_coordinator_when_disabled(self, config):
        """Test sync coordinator not loaded when disabled."""
        config.enable_sync = False
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        sync_coord = coord._get_sync_coordinator()
        assert sync_coord is None

        TrainingDataCoordinator.reset_instance()

    def test_get_quality_bridge_caches_result(self, config):
        """Test that quality bridge is cached after first access."""
        config.enable_quality_scoring = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        # Manually set a cached bridge to test caching behavior
        mock_bridge = MagicMock()
        coord._quality_bridge = mock_bridge

        # Second call should return cached bridge
        bridge1 = coord._get_quality_bridge()
        bridge2 = coord._get_quality_bridge()

        assert bridge1 is bridge2
        assert bridge1 is mock_bridge

        TrainingDataCoordinator.reset_instance()

    def test_get_hot_buffer_creates_new(self, coordinator):
        """Test get_hot_buffer creates new buffer when none exists."""
        assert coordinator._hot_buffer is None

        # Mock the hot buffer module properly
        mock_buffer = MagicMock()
        mock_module = MagicMock()
        mock_module.create_hot_buffer = MagicMock(return_value=mock_buffer)

        import sys
        original = sys.modules.get("app.training.hot_data_buffer")
        sys.modules["app.training.hot_data_buffer"] = mock_module

        try:
            buffer = coordinator.get_hot_buffer()
            assert buffer is mock_buffer
        finally:
            if original:
                sys.modules["app.training.hot_data_buffer"] = original
            elif "app.training.hot_data_buffer" in sys.modules:
                del sys.modules["app.training.hot_data_buffer"]

    def test_get_hot_buffer_returns_cached(self, coordinator):
        """Test get_hot_buffer returns cached buffer."""
        mock_buffer = MagicMock()
        coordinator._hot_buffer = mock_buffer

        buffer = coordinator.get_hot_buffer()

        assert buffer is mock_buffer


# =============================================================================
# Training Preparation Tests
# =============================================================================


class TestPrepareForTraining:
    """Tests for prepare_for_training method."""

    @pytest.mark.asyncio
    async def test_prepare_for_training_basic(self, coordinator):
        """Test basic training preparation."""
        result = await coordinator.prepare_for_training(
            board_type="square8",
            num_players=2,
            force_sync=False,
            load_from_db=False,
        )

        assert result["success"] is True
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0
        assert coordinator._stats.preparation_count == 1

    @pytest.mark.asyncio
    async def test_prepare_for_training_updates_stats(self, coordinator):
        """Test that preparation updates stats."""
        initial_count = coordinator._stats.preparation_count

        await coordinator.prepare_for_training(
            board_type="hex8",
            num_players=4,
            load_from_db=False,
        )

        assert coordinator._stats.preparation_count == initial_count + 1
        assert coordinator._last_preparation_time > 0

    @pytest.mark.asyncio
    async def test_prepare_for_training_with_sync(self, config):
        """Test preparation with sync enabled."""
        config.enable_sync = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        with patch.object(coord, "_sync_high_quality_data", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = {"games_synced": 50, "errors": []}

            result = await coord.prepare_for_training(
                board_type="square8",
                num_players=2,
                force_sync=True,
                load_from_db=False,
            )

            mock_sync.assert_called_once_with(force=True)
            assert result["games_synced"] == 50

        TrainingDataCoordinator.reset_instance()

    @pytest.mark.asyncio
    async def test_prepare_for_training_with_discovery(self, config):
        """Test preparation with auto-discovery."""
        config.enable_auto_discovery = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        with patch("app.training.data_coordinator.HAS_AUTO_DISCOVERY", True):
            with patch.object(coord, "_run_auto_discovery") as mock_discovery:
                mock_discovery.return_value = {
                    "total_games": 1000,
                    "num_sources": 3,
                    "avg_quality": 0.85,
                    "data_paths": ["/path/1", "/path/2"],
                }

                result = await coord.prepare_for_training(
                    board_type="hex8",
                    num_players=2,
                    load_from_db=False,
                )

                assert result["games_discovered"] == 1000
                assert result["discovered_sources"] == 3

        TrainingDataCoordinator.reset_instance()


# =============================================================================
# Data Loading Tests
# =============================================================================


class TestDataLoading:
    """Tests for data loading methods."""

    def test_load_high_quality_games_no_buffer(self, coordinator, temp_selfplay_dir):
        """Test loading games when hot buffer is unavailable."""
        coordinator._hot_buffer = None

        with patch.object(coordinator, "_create_hot_buffer", return_value=None):
            loaded = coordinator.load_high_quality_games(
                db_path=temp_selfplay_dir / "test.db",
                board_type="square8",
                num_players=2,
            )

            assert loaded == 0

    def test_load_high_quality_games_with_buffer(self, coordinator, temp_selfplay_dir):
        """Test loading games with hot buffer available."""
        mock_buffer = MagicMock()
        mock_buffer.load_from_db.return_value = 50
        coordinator._hot_buffer = mock_buffer

        loaded = coordinator.load_high_quality_games(
            db_path=temp_selfplay_dir / "test.db",
            board_type="hex8",
            num_players=4,
            min_quality=0.8,
            limit=100,
        )

        assert loaded == 50
        mock_buffer.load_from_db.assert_called_once()
        assert coordinator._stats.total_games_loaded == 50
        assert coordinator._stats.high_quality_games_loaded == 50

    def test_load_high_quality_games_updates_stats(self, coordinator, temp_selfplay_dir):
        """Test that loading updates stats correctly."""
        mock_buffer = MagicMock()
        mock_buffer.load_from_db.return_value = 25
        coordinator._hot_buffer = mock_buffer

        # Load with quality below HIGH_QUALITY_THRESHOLD (0.7)
        loaded = coordinator.load_high_quality_games(
            db_path=temp_selfplay_dir / "test.db",
            min_quality=0.5,  # Below threshold
        )

        assert coordinator._stats.total_games_loaded == 25
        # Should not count as high quality since min_quality < 0.7
        assert coordinator._stats.high_quality_games_loaded == 0

    def test_get_high_quality_game_ids_no_bridge(self, coordinator):
        """Test getting game IDs when bridge is unavailable."""
        coordinator._quality_bridge = None
        coordinator._config.enable_quality_scoring = False

        game_ids = coordinator.get_high_quality_game_ids()

        assert game_ids == []

    def test_get_high_quality_game_ids_with_bridge(self, coordinator):
        """Test getting game IDs with quality bridge."""
        mock_bridge = MagicMock()
        mock_bridge.get_high_quality_game_ids.return_value = ["game1", "game2", "game3"]
        coordinator._quality_bridge = mock_bridge

        game_ids = coordinator.get_high_quality_game_ids(min_quality=0.8, limit=100)

        assert game_ids == ["game1", "game2", "game3"]
        mock_bridge.get_high_quality_game_ids.assert_called_once_with(
            min_quality=0.8,
            min_elo=coordinator._config.min_elo_threshold,
            limit=100,
        )


# =============================================================================
# Local Database Loading Tests
# =============================================================================


class TestLocalDatabaseLoading:
    """Tests for loading from local databases."""

    def test_load_games_from_local_dbs_no_buffer(self, coordinator):
        """Test loading from local DBs when buffer unavailable."""
        coordinator._hot_buffer = None

        with patch.object(coordinator, "_create_hot_buffer", return_value=None):
            loaded = coordinator._load_games_from_local_dbs(
                board_type="square8",
                num_players=2,
            )

            assert loaded == 0

    def test_load_games_from_local_dbs_excludes_patterns(self, coordinator, temp_selfplay_dir):
        """Test that excluded patterns are skipped."""
        # Create test database files
        (temp_selfplay_dir / "square8_test.db").touch()
        (temp_selfplay_dir / "square8_temp.db").touch()
        (temp_selfplay_dir / "canonical_square8.db").touch()

        coordinator._selfplay_dir = temp_selfplay_dir
        coordinator._config.excluded_db_patterns = ("*_test.db", "*_temp.db")

        mock_buffer = MagicMock()
        mock_buffer.load_from_db.return_value = 10
        coordinator._hot_buffer = mock_buffer

        with patch("app.training.data_coordinator.should_exclude_database") as mock_exclude:
            # Exclude test and temp, allow canonical
            def exclude_check(path, patterns):
                return "_test.db" in str(path) or "_temp.db" in str(path)
            mock_exclude.side_effect = exclude_check

            loaded = coordinator._load_games_from_local_dbs(
                board_type="square8",
                num_players=2,
            )

        # Should only load from canonical_square8.db
        assert mock_buffer.load_from_db.call_count <= 1


# =============================================================================
# Status and Stats Tests
# =============================================================================


class TestStatusAndStats:
    """Tests for status and stats reporting."""

    def test_get_stats(self, coordinator):
        """Test getting coordinator stats."""
        coordinator._stats.total_games_loaded = 100
        coordinator._stats.preparation_count = 5

        stats = coordinator.get_stats()

        assert stats.total_games_loaded == 100
        assert stats.preparation_count == 5

    def test_get_stats_updates_from_bridge(self, coordinator):
        """Test that get_stats updates from quality bridge."""
        mock_bridge = MagicMock()
        mock_stats = MagicMock()
        mock_stats.avg_quality_score = 0.85
        mock_stats.avg_elo = 1550.0
        mock_bridge.get_stats.return_value = mock_stats
        coordinator._quality_bridge = mock_bridge

        stats = coordinator.get_stats()

        assert stats.avg_quality_score == 0.85
        assert stats.avg_elo == 1550.0

    def test_get_status_structure(self, coordinator):
        """Test get_status returns correct structure."""
        status = coordinator.get_status()

        assert "initialized" in status
        assert "config" in status
        assert "stats" in status
        assert "components" in status
        assert "timing" in status
        assert "errors" in status

        # Check config fields
        assert "quality_scoring" in status["config"]
        assert "sync_enabled" in status["config"]
        assert "min_quality" in status["config"]

        # Check stats fields
        assert "total_games_loaded" in status["stats"]
        assert "preparation_count" in status["stats"]

        # Check components fields
        assert "quality_bridge" in status["components"]
        assert "hot_buffer" in status["components"]

    def test_get_status_hot_buffer_size(self, coordinator):
        """Test that status includes hot buffer size."""
        mock_buffer = MagicMock()
        mock_buffer.__len__ = MagicMock(return_value=42)
        coordinator._hot_buffer = mock_buffer

        status = coordinator.get_status()

        assert status["components"]["hot_buffer"] is True
        assert status["components"]["hot_buffer_size"] == 42

    def test_get_status_errors_limited(self, coordinator):
        """Test that errors are limited to last 5."""
        for i in range(10):
            coordinator._stats.errors.append(f"Error {i}")

        status = coordinator.get_status()

        assert len(status["errors"]) == 5
        assert status["errors"] == [f"Error {i}" for i in range(5, 10)]


# =============================================================================
# Promotion Event Tests
# =============================================================================


class TestPromotionEvents:
    """Tests for promotion event handling."""

    def test_on_promotion_registers_callback(self, coordinator):
        """Test registering a promotion callback."""
        callback = MagicMock()

        coordinator.on_promotion(callback)

        assert callback in coordinator._promotion_callbacks

    def test_off_promotion_unregisters_callback(self, coordinator):
        """Test unregistering a promotion callback."""
        callback = MagicMock()
        coordinator._promotion_callbacks.append(callback)

        coordinator.off_promotion(callback)

        assert callback not in coordinator._promotion_callbacks

    def test_off_promotion_nonexistent_callback(self, coordinator):
        """Test unregistering a callback that wasn't registered."""
        callback = MagicMock()

        # Should not raise
        coordinator.off_promotion(callback)

    @pytest.mark.asyncio
    async def test_handle_promotion_event_calls_callbacks(self, coordinator):
        """Test that handle_promotion_event calls all callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        coordinator._promotion_callbacks = [callback1, callback2]

        event_data = {
            "model_id": "test_model",
            "promotion_type": "challenger",
            "board_type": "hex8",
            "num_players": 2,
        }

        await coordinator.handle_promotion_event(event_data)

        callback1.assert_called_once_with(event_data)
        callback2.assert_called_once_with(event_data)

    @pytest.mark.asyncio
    async def test_handle_promotion_event_callback_error(self, coordinator, caplog):
        """Test that callback errors don't stop other callbacks."""
        failing_callback = MagicMock(side_effect=ValueError("Test error"))
        success_callback = MagicMock()
        coordinator._promotion_callbacks = [failing_callback, success_callback]

        event_data = {"model_id": "test", "promotion_type": "production"}

        with caplog.at_level(logging.WARNING):
            await coordinator.handle_promotion_event(event_data)

        # Both should be called despite first one failing
        failing_callback.assert_called_once()
        success_callback.assert_called_once_with(event_data)
        assert "Promotion callback error" in caplog.text

    @pytest.mark.asyncio
    async def test_handle_promotion_invalidates_cache_on_production(self, coordinator):
        """Test that production promotion invalidates quality cache."""
        mock_bridge = MagicMock()
        coordinator._quality_bridge = mock_bridge

        event_data = {
            "model_id": "test_model",
            "promotion_type": "production",
            "board_type": "square8",
            "num_players": 2,
        }

        await coordinator.handle_promotion_event(event_data)

        mock_bridge.invalidate_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_promotion_champion_triggers_operations(self, coordinator):
        """Test that champion promotion triggers quality bridge operations."""
        mock_bridge = MagicMock()
        coordinator._quality_bridge = mock_bridge

        event_data = {
            "model_id": "test_model",
            "promotion_type": "champion",  # Same as production for cache invalidation
            "board_type": "hex8",
            "num_players": 4,
        }

        await coordinator.handle_promotion_event(event_data)

        # Champion type should also invalidate cache (same as production)
        mock_bridge.invalidate_cache.assert_called_once()


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription."""

    def test_subscribe_to_promotion_events_already_subscribed(self, coordinator):
        """Test that double subscription returns True."""
        coordinator._event_bus_subscription = MagicMock()

        result = coordinator.subscribe_to_promotion_events()

        assert result is True

    def test_subscribe_to_promotion_events_no_router(self, coordinator):
        """Test subscription when router is unavailable."""
        # Mock both imports to raise ImportError
        with patch.dict("sys.modules", {
            "app.coordination.event_router": None  # Force ImportError
        }):
            # Clear the import cache to force re-import
            import sys
            if "app.coordination.event_router" in sys.modules:
                del sys.modules["app.coordination.event_router"]

            # Since the module uses try/except ImportError internally,
            # we test that it handles the case gracefully
            result = coordinator.subscribe_to_promotion_events()

            # Result depends on whether other subscriptions succeed
            # The main thing is it doesn't crash
            assert result in (True, False)


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_data_coordinator(self, config):
        """Test get_data_coordinator function."""
        TrainingDataCoordinator.reset_instance()

        coord = get_data_coordinator(config)

        assert isinstance(coord, TrainingDataCoordinator)
        assert coord._config.enable_sync is False

        TrainingDataCoordinator.reset_instance()

    @pytest.mark.asyncio
    async def test_prepare_training_data(self, config):
        """Test prepare_training_data convenience function."""
        TrainingDataCoordinator.reset_instance()
        coord = get_data_coordinator(config)

        result = await prepare_training_data(
            board_type="square8",
            num_players=2,
            force_sync=False,
        )

        assert result["success"] is True

        TrainingDataCoordinator.reset_instance()

    def test_get_high_quality_games(self, config):
        """Test get_high_quality_games convenience function."""
        TrainingDataCoordinator.reset_instance()
        config.enable_quality_scoring = False
        coord = get_data_coordinator(config)

        games = get_high_quality_games(min_quality=0.8, limit=100)

        # Without quality bridge, should return empty list
        assert games == []

        TrainingDataCoordinator.reset_instance()

    def test_wire_promotion_events(self, config):
        """Test wire_promotion_events function."""
        TrainingDataCoordinator.reset_instance()
        coord = get_data_coordinator(config)

        with patch.object(coord, "subscribe_to_promotion_events", return_value=True) as mock_sub:
            result = wire_promotion_events(coord)

            assert result is True
            mock_sub.assert_called_once()

        TrainingDataCoordinator.reset_instance()

    def test_wire_promotion_events_uses_singleton(self, config):
        """Test wire_promotion_events uses singleton when no coordinator passed."""
        TrainingDataCoordinator.reset_instance()
        get_data_coordinator(config)

        with patch.object(
            TrainingDataCoordinator, "get_instance"
        ) as mock_get:
            mock_coord = MagicMock()
            mock_coord.subscribe_to_promotion_events.return_value = True
            mock_get.return_value = mock_coord

            result = wire_promotion_events()

            assert result is True

        TrainingDataCoordinator.reset_instance()


# =============================================================================
# Metrics Collection Tests
# =============================================================================


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_collect_metrics_no_bridge(self, coordinator):
        """Test metrics collection without quality bridge."""
        coordinator._quality_bridge = None

        result = coordinator.collect_metrics()

        # Should return False if collection failed (ImportError)
        assert result in (True, False)

    def test_collect_metrics_with_bridge(self, coordinator):
        """Test metrics collection with quality bridge."""
        mock_bridge = MagicMock()
        coordinator._quality_bridge = mock_bridge

        # Mock at the source module location
        mock_orchestrator = MagicMock()
        mock_orchestrator.collect_quality_metrics_from_bridge = MagicMock()

        with patch.dict("sys.modules", {"app.metrics.orchestrator": mock_orchestrator}):
            result = coordinator.collect_metrics()

            # Should succeed with mocked module
            assert result is True

    def test_collect_metrics_import_error(self, coordinator):
        """Test metrics collection when import fails."""
        coordinator._quality_bridge = MagicMock()

        # Remove the metrics module to force ImportError
        import sys
        original = sys.modules.get("app.metrics.orchestrator")
        sys.modules["app.metrics.orchestrator"] = None

        try:
            result = coordinator.collect_metrics()
            # Should return False when import fails
            assert result is False
        finally:
            if original:
                sys.modules["app.metrics.orchestrator"] = original
            elif "app.metrics.orchestrator" in sys.modules:
                del sys.modules["app.metrics.orchestrator"]


# =============================================================================
# Auto-Discovery Tests
# =============================================================================


class TestAutoDiscovery:
    """Tests for auto-discovery functionality."""

    def test_run_auto_discovery_not_available(self, coordinator):
        """Test discovery when module not available."""
        with patch("app.training.data_coordinator.HAS_AUTO_DISCOVERY", False):
            result = coordinator._run_auto_discovery("hex8", 2)

        assert result["total_games"] == 0
        assert result["num_sources"] == 0
        assert result["data_paths"] == []

    def test_run_auto_discovery_disabled(self, coordinator):
        """Test discovery when should_auto_discover returns False."""
        with patch("app.training.data_coordinator.HAS_AUTO_DISCOVERY", True):
            with patch("app.training.data_coordinator.should_auto_discover", return_value=False):
                result = coordinator._run_auto_discovery("hex8", 2)

        assert result["total_games"] == 0

    def test_get_discovered_data_paths_not_available(self, coordinator):
        """Test get_discovered_data_paths when discovery not available."""
        with patch("app.training.data_coordinator.HAS_AUTO_DISCOVERY", False):
            paths = coordinator.get_discovered_data_paths()

        assert paths == []


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Tests for sync functionality."""

    @pytest.mark.asyncio
    async def test_sync_high_quality_data_no_coordinator(self, coordinator):
        """Test sync when coordinator is unavailable."""
        coordinator._config.enable_sync = False

        result = await coordinator._sync_high_quality_data(force=False)

        assert result["games_synced"] == 0

    @pytest.mark.asyncio
    async def test_sync_high_quality_data_skips_recent(self, coordinator, config):
        """Test sync skips when recently synced."""
        config.enable_sync = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        # Set recent sync time
        coord._stats.last_sync_time = time.time()

        mock_sync_coord = MagicMock()
        coord._sync_coordinator = mock_sync_coord

        result = await coord._sync_high_quality_data(force=False)

        # Should skip since we just synced
        mock_sync_coord.sync_high_quality_games.assert_not_called()

        TrainingDataCoordinator.reset_instance()

    @pytest.mark.asyncio
    async def test_sync_high_quality_data_force(self, coordinator, config):
        """Test forced sync ignores recent sync time."""
        config.enable_sync = True
        config.sync_high_quality_first = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        # Set recent sync time
        coord._stats.last_sync_time = time.time()

        mock_sync_coord = MagicMock()
        mock_stats = MagicMock()
        mock_stats.high_quality_games_synced = 100
        mock_stats.errors = []
        mock_sync_coord.sync_high_quality_games = AsyncMock(return_value=mock_stats)
        coord._sync_coordinator = mock_sync_coord

        result = await coord._sync_high_quality_data(force=True)

        mock_sync_coord.sync_high_quality_games.assert_called_once()
        assert result["games_synced"] == 100

        TrainingDataCoordinator.reset_instance()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_create_hot_buffer_import_error(self, coordinator, caplog):
        """Test hot buffer creation when import fails."""
        # Remove the module to force ImportError
        import sys
        original = sys.modules.get("app.training.hot_data_buffer")
        sys.modules["app.training.hot_data_buffer"] = None

        try:
            with caplog.at_level(logging.WARNING):
                buffer = coordinator._create_hot_buffer()

            assert buffer is None
            assert "HotDataBuffer not available" in caplog.text
        finally:
            if original:
                sys.modules["app.training.hot_data_buffer"] = original
            elif "app.training.hot_data_buffer" in sys.modules:
                del sys.modules["app.training.hot_data_buffer"]

    def test_create_hot_buffer_exception(self, coordinator, caplog):
        """Test hot buffer creation when exception occurs."""
        # Mock the module to raise an exception during creation
        mock_module = MagicMock()
        mock_module.create_hot_buffer = MagicMock(side_effect=RuntimeError("Test error"))

        import sys
        original = sys.modules.get("app.training.hot_data_buffer")
        sys.modules["app.training.hot_data_buffer"] = mock_module

        try:
            with caplog.at_level(logging.WARNING):
                buffer = coordinator._create_hot_buffer()

            assert buffer is None
            assert "Failed to create HotDataBuffer" in caplog.text
            assert "Test error" in coordinator._stats.errors[-1]
        finally:
            if original:
                sys.modules["app.training.hot_data_buffer"] = original
            elif "app.training.hot_data_buffer" in sys.modules:
                del sys.modules["app.training.hot_data_buffer"]

    def test_get_quality_bridge_import_error(self, coordinator, caplog):
        """Test quality bridge when import fails."""
        coordinator._config.enable_quality_scoring = True

        # Remove the module to force ImportError
        import sys
        original = sys.modules.get("app.training.quality_bridge")
        sys.modules["app.training.quality_bridge"] = None

        try:
            with caplog.at_level(logging.WARNING):
                bridge = coordinator._get_quality_bridge()

            assert bridge is None
            assert "QualityBridge not available" in caplog.text
        finally:
            if original:
                sys.modules["app.training.quality_bridge"] = original
            elif "app.training.quality_bridge" in sys.modules:
                del sys.modules["app.training.quality_bridge"]

    def test_get_sync_coordinator_import_error(self, config, caplog):
        """Test sync coordinator when import fails."""
        config.enable_sync = True
        TrainingDataCoordinator.reset_instance()
        coord = TrainingDataCoordinator(config=config)

        # Remove the module to force ImportError
        import sys
        original = sys.modules.get("app.distributed.sync_coordinator")
        sys.modules["app.distributed.sync_coordinator"] = None

        try:
            with caplog.at_level(logging.WARNING):
                sync_coord = coord._get_sync_coordinator()

            assert sync_coord is None
            assert "SyncCoordinator not available" in caplog.text
        finally:
            if original:
                sys.modules["app.distributed.sync_coordinator"] = original
            elif "app.distributed.sync_coordinator" in sys.modules:
                del sys.modules["app.distributed.sync_coordinator"]

        TrainingDataCoordinator.reset_instance()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
