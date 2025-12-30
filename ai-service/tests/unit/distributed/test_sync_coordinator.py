"""Tests for app.distributed.sync_coordinator module.

This module tests the unified data synchronization EXECUTION layer.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import the module to ensure we can patch it correctly
# Import the module directly to bypass custom __getattr__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sync_coordinator",
    Path(__file__).parent.parent.parent.parent / "app" / "distributed" / "sync_coordinator.py"
)


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================


@pytest.fixture
def mock_storage_provider():
    """Create mock StorageProvider."""
    provider = MagicMock()
    provider.provider_type = MagicMock()
    provider.provider_type.value = "local"
    provider.has_shared_storage = False
    provider.selfplay_dir = Path("/tmp/data/selfplay")
    provider.training_dir = Path("/tmp/data/training")
    provider.models_dir = Path("/tmp/models")
    provider.data_dir = Path("/tmp/data")
    provider.should_skip_rsync_to = MagicMock(return_value=False)
    return provider


@pytest.fixture
def mock_transport_config():
    """Create mock TransportConfig."""
    config = MagicMock()
    config.enable_aria2 = True
    config.enable_p2p = True
    config.enable_gossip = True
    config.enable_ssh = True
    config.enable_bittorrent = True
    config.aria2_connections_per_server = 16
    config.aria2_split = 16
    config.aria2_data_server_port = 8766
    config.gossip_port = 8765
    config.ssh_timeout = 30
    config.fallback_chain = ["aria2", "ssh", "p2p"]
    config.get_fallback_chain_for_size = MagicMock(return_value=["bittorrent", "aria2", "ssh"])
    return config


# =============================================================================
# SyncCategory Tests
# =============================================================================


class TestSyncCategory:
    """Tests for SyncCategory enum."""

    def test_category_values(self):
        """Test SyncCategory enum values."""
        # Import from the actual file location
        from app.distributed.sync_coordinator import SyncCategory
        
        assert SyncCategory.GAMES.value == "games"
        assert SyncCategory.MODELS.value == "models"
        assert SyncCategory.TRAINING.value == "training"
        assert SyncCategory.ELO.value == "elo"
        assert SyncCategory.ALL.value == "all"

    def test_category_iteration(self):
        """Test iterating over SyncCategory."""
        from app.distributed.sync_coordinator import SyncCategory
        
        categories = list(SyncCategory)
        assert len(categories) == 5


# =============================================================================
# SyncStats Tests
# =============================================================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_basic_creation(self):
        """Test creating SyncStats with defaults."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(category="games")
        assert stats.category == "games"
        assert stats.files_synced == 0
        assert stats.files_failed == 0
        assert stats.bytes_transferred == 0
        assert stats.duration_seconds == 0.0
        assert stats.transport_used == ""
        assert stats.sources_tried == 0
        assert stats.errors == []

    def test_with_values(self):
        """Test creating SyncStats with values."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(
            category="training",
            files_synced=10,
            files_failed=2,
            bytes_transferred=1024000,
            duration_seconds=5.5,
            transport_used="aria2",
            sources_tried=3,
            errors=["error1"],
        )
        assert stats.files_synced == 10
        assert stats.files_failed == 2
        assert stats.bytes_transferred == 1024000

    def test_success_rate_all_success(self):
        """Test success_rate with all successful syncs."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(category="models", files_synced=10, files_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        """Test success_rate with partial failures."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(category="models", files_synced=7, files_failed=3)
        assert stats.success_rate == 0.7

    def test_success_rate_no_files(self):
        """Test success_rate with no files."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(category="models", files_synced=0, files_failed=0)
        assert stats.success_rate == 1.0

    def test_quality_aware_stats(self):
        """Test quality-aware stat fields."""
        from app.distributed.sync_coordinator import SyncStats
        
        stats = SyncStats(
            category="games",
            high_quality_games_synced=100,
            avg_quality_score=0.85,
            avg_elo=1500.0,
        )
        assert stats.high_quality_games_synced == 100
        assert stats.avg_quality_score == 0.85
        assert stats.avg_elo == 1500.0


# =============================================================================
# ClusterSyncStats Tests
# =============================================================================


class TestClusterSyncStats:
    """Tests for ClusterSyncStats dataclass."""

    def test_basic_creation(self):
        """Test creating ClusterSyncStats with defaults."""
        from app.distributed.sync_coordinator import ClusterSyncStats
        
        stats = ClusterSyncStats()
        assert stats.total_files_synced == 0
        assert stats.total_bytes_transferred == 0
        assert stats.duration_seconds == 0.0
        assert stats.categories == {}
        assert stats.transport_distribution == {}
        assert stats.nodes_synced == 0
        assert stats.nodes_failed == 0

    def test_with_categories(self):
        """Test ClusterSyncStats with category data."""
        from app.distributed.sync_coordinator import ClusterSyncStats, SyncStats
        
        games_stats = SyncStats(category="games", files_synced=5)
        models_stats = SyncStats(category="models", files_synced=3)
        
        stats = ClusterSyncStats(
            total_files_synced=8,
            categories={"games": games_stats, "models": models_stats},
        )
        assert len(stats.categories) == 2
        assert stats.categories["games"].files_synced == 5

    def test_quality_distribution(self):
        """Test ClusterSyncStats quality distribution."""
        from app.distributed.sync_coordinator import ClusterSyncStats
        
        stats = ClusterSyncStats(
            total_high_quality_games=500,
            avg_quality_score=0.78,
            quality_distribution={"high": 200, "medium": 200, "low": 100},
        )
        assert stats.total_high_quality_games == 500
        assert stats.avg_quality_score == 0.78
        assert stats.quality_distribution["high"] == 200


# =============================================================================
# SyncOperationBudget Tests
# =============================================================================


class TestSyncOperationBudget:
    """Tests for SyncOperationBudget dataclass."""

    def test_default_budget(self):
        """Test creating budget with defaults."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        budget = SyncOperationBudget()
        assert budget.total_seconds == 300.0
        assert budget.per_attempt_seconds == 30.0
        assert budget.attempts == 0

    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        start = time.time() - 10  # 10 seconds ago
        budget = SyncOperationBudget(start_time=start)
        assert 9 < budget.elapsed < 11

    def test_remaining_time(self):
        """Test remaining time calculation."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        start = time.time() - 100  # 100 seconds ago
        budget = SyncOperationBudget(total_seconds=300, start_time=start)
        assert 199 < budget.remaining < 201

    def test_exhausted_false(self):
        """Test budget not exhausted."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        budget = SyncOperationBudget(total_seconds=300)
        assert not budget.exhausted

    def test_exhausted_true(self):
        """Test budget exhausted."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        start = time.time() - 400  # 400 seconds ago
        budget = SyncOperationBudget(total_seconds=300, start_time=start)
        assert budget.exhausted
        assert budget.remaining == 0

    def test_get_attempt_timeout_normal(self):
        """Test attempt timeout with plenty of budget."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        budget = SyncOperationBudget(
            total_seconds=300,
            per_attempt_seconds=30,
        )
        assert budget.get_attempt_timeout() == 30.0

    def test_get_attempt_timeout_limited(self):
        """Test attempt timeout capped by remaining budget."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        start = time.time() - 295  # 295 seconds ago, only 5 left
        budget = SyncOperationBudget(
            total_seconds=300,
            per_attempt_seconds=30,
            start_time=start,
        )
        timeout = budget.get_attempt_timeout()
        assert timeout < 6

    def test_record_attempt(self):
        """Test recording attempts."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        budget = SyncOperationBudget()
        assert budget.attempts == 0
        budget.record_attempt()
        assert budget.attempts == 1
        budget.record_attempt()
        budget.record_attempt()
        assert budget.attempts == 3

    def test_can_attempt(self):
        """Test can_attempt check."""
        from app.distributed.sync_coordinator import SyncOperationBudget
        
        budget = SyncOperationBudget(total_seconds=300)
        assert budget.can_attempt()
        
        # Exhausted budget
        start = time.time() - 400
        budget2 = SyncOperationBudget(total_seconds=300, start_time=start)
        assert not budget2.can_attempt()


# =============================================================================
# SyncCoordinator Tests - using module-level fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_sync_coordinator():
    """Reset SyncCoordinator singleton between tests."""
    from app.distributed.sync_coordinator import SyncCoordinator
    SyncCoordinator._instance = None
    yield
    SyncCoordinator._instance = None


class TestSyncCoordinatorInit:
    """Tests for SyncCoordinator initialization."""

    def test_singleton_pattern(self, mock_storage_provider, mock_transport_config):
        """Test singleton pattern."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        # Create first instance with explicit provider
        instance1 = SyncCoordinator(mock_storage_provider, mock_transport_config)
        SyncCoordinator._instance = instance1
        
        # Second call should return same instance
        instance2 = SyncCoordinator.get_instance()
        
        assert instance1 is instance2

    def test_initialization_with_provider(self, mock_storage_provider, mock_transport_config):
        """Test initialization with custom provider."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        assert coordinator._provider is mock_storage_provider
        assert coordinator._config is mock_transport_config
        assert coordinator._running is False

    def test_transport_lazy_initialization(self, mock_storage_provider, mock_transport_config):
        """Test that transports are lazily initialized."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        assert coordinator._aria2 is None
        assert coordinator._p2p is None
        assert coordinator._gossip is None


class TestSyncCoordinatorTransports:
    """Tests for SyncCoordinator transport initialization."""

    def test_init_aria2_disabled(self, mock_storage_provider, mock_transport_config):
        """Test aria2 not initialized when disabled."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        mock_transport_config.enable_aria2 = False
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        assert coordinator._init_aria2() is None

    def test_init_p2p_disabled(self, mock_storage_provider, mock_transport_config):
        """Test P2P not initialized when disabled."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        mock_transport_config.enable_p2p = False
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        assert coordinator._init_p2p() is None


class TestSyncCoordinatorHelpers:
    """Tests for SyncCoordinator static helper methods."""

    def test_snapshot_files(self, tmp_path):
        """Test file snapshot creation."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        # Create test files
        (tmp_path / "test1.db").write_text("content1")
        (tmp_path / "test2.db").write_text("content2content2")
        (tmp_path / "other.txt").write_text("ignored")
        
        snapshot = SyncCoordinator._snapshot_files(tmp_path, ["*.db"])
        
        assert len(snapshot) == 2
        assert "test1.db" in snapshot
        assert "test2.db" in snapshot

    def test_snapshot_files_empty_dir(self, tmp_path):
        """Test snapshot on empty directory."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        snapshot = SyncCoordinator._snapshot_files(tmp_path, ["*.db"])
        assert snapshot == {}

    def test_snapshot_files_nonexistent_dir(self):
        """Test snapshot on nonexistent directory."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        snapshot = SyncCoordinator._snapshot_files(Path("/nonexistent"), ["*.db"])
        assert snapshot == {}

    def test_diff_snapshot(self):
        """Test snapshot diffing."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        before = {"file1.db": 100, "file2.db": 200}
        after = {"file1.db": 100, "file2.db": 200, "file3.db": 300, "file4.db": 400}
        
        new_count, new_bytes = SyncCoordinator._diff_snapshot(before, after)
        
        assert new_count == 2
        assert new_bytes == 700


class TestSyncCoordinatorManifest:
    """Tests for SyncCoordinator manifest integration."""

    def test_get_quality_lookup_empty(self, mock_storage_provider, mock_transport_config):
        """Test quality lookup when manifest not available."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._manifest = None
        
        lookup = coordinator.get_quality_lookup()
        assert lookup == {}

    def test_get_elo_lookup_empty(self, mock_storage_provider, mock_transport_config):
        """Test Elo lookup when manifest not available."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._manifest = None
        
        lookup = coordinator.get_elo_lookup()
        assert lookup == {}

    def test_get_manifest(self, mock_storage_provider, mock_transport_config):
        """Test getting manifest instance."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        # Without manifest
        coordinator._manifest = None
        assert coordinator.get_manifest() is None

    def test_get_high_quality_game_ids_no_manifest(self, mock_storage_provider, mock_transport_config):
        """Test getting high quality game IDs without manifest."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._manifest = None
        
        result = coordinator.get_high_quality_game_ids()
        assert result == []


class TestSyncCoordinatorDataServer:
    """Tests for SyncCoordinator data server methods."""

    def test_is_data_server_running_false(self, mock_storage_provider, mock_transport_config):
        """Test data server running check when not running."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        assert not coordinator.is_data_server_running()

    @pytest.mark.asyncio
    async def test_start_data_server_already_running(self, mock_storage_provider, mock_transport_config):
        """Test starting data server when already running."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._data_server_process = MagicMock()
        coordinator._data_server_process.returncode = None
        
        result = await coordinator.start_data_server()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_data_server_not_running(self, mock_storage_provider, mock_transport_config):
        """Test stopping data server when not running."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        # Should not raise
        await coordinator.stop_data_server()


class TestSyncCoordinatorDiscovery:
    """Tests for SyncCoordinator source discovery."""

    @pytest.mark.asyncio
    async def test_discover_sources_cached(self, mock_storage_provider, mock_transport_config):
        """Test source discovery uses cache."""
        from app.distributed.sync_coordinator import SyncCoordinator

        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # Manually set cache
        coordinator._aria2_sources = ["http://host1:8766"]
        coordinator._source_discovery_time = time.time()

        # Should use cached value
        sources = await coordinator.discover_sources()

        assert sources == ["http://host1:8766"]

    @pytest.mark.asyncio
    async def test_discover_sources_force_refresh(self, mock_storage_provider, mock_transport_config):
        """Test source discovery with force refresh bypasses cache."""
        from app.distributed.sync_coordinator import SyncCoordinator

        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # Set fresh cache
        coordinator._aria2_sources = ["http://cached:8766"]
        coordinator._source_discovery_time = time.time()

        # Patch the module-level function to return new sources
        import app.distributed.sync_coordinator as sc
        original = sc.get_aria2_sources
        sc.get_aria2_sources = lambda **kwargs: ["http://new:8766"]

        try:
            sources = await coordinator.discover_sources(force_refresh=True)
            assert "http://new:8766" in sources
        finally:
            sc.get_aria2_sources = original


class TestSyncCoordinatorSyncOps:
    """Tests for SyncCoordinator sync operations."""

    @pytest.mark.asyncio
    async def test_sync_training_data_shared_storage(self, mock_storage_provider, mock_transport_config):
        """Test sync skipped for shared storage."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        mock_storage_provider.has_shared_storage = True
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        stats = await coordinator.sync_training_data()
        
        assert stats.transport_used == "nfs_shared"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_sync_models_shared_storage(self, mock_storage_provider, mock_transport_config):
        """Test model sync skipped for shared storage."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        mock_storage_provider.has_shared_storage = True
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        stats = await coordinator.sync_models()
        
        assert stats.transport_used == "nfs_shared"

    @pytest.mark.asyncio
    async def test_sync_games_shared_storage(self, mock_storage_provider, mock_transport_config):
        """Test games sync skipped for shared storage."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        mock_storage_provider.has_shared_storage = True
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        stats = await coordinator.sync_games()
        
        assert stats.transport_used == "nfs_shared"

    @pytest.mark.asyncio
    async def test_sync_training_data_no_sources(self, mock_storage_provider, mock_transport_config):
        """Test sync with no sources available."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        # Pass empty sources directly
        stats = await coordinator.sync_training_data(sources=[])
        
        assert stats.sources_tried == 0

    @pytest.mark.asyncio
    async def test_sync_high_quality_games_no_manifest(self, mock_storage_provider, mock_transport_config):
        """Test high quality game sync without manifest."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._manifest = None
        
        stats = await coordinator.sync_high_quality_games()
        
        assert "No manifest" in str(stats.errors)


class TestSyncCoordinatorHealth:
    """Tests for SyncCoordinator health check methods."""

    def test_get_sync_health(self, mock_storage_provider, mock_transport_config):
        """Test get_sync_health returns expected structure."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        health = coordinator.get_sync_health()
        
        assert "running" in health
        assert "consecutive_failures" in health
        assert "last_successful_sync" in health

    def test_health_check_not_running(self, mock_storage_provider, mock_transport_config):
        """Test health_check when not running."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        coordinator._running = False
        
        result = coordinator.health_check()
        
        # Should return a HealthCheckResult or dict-like
        assert hasattr(result, "healthy") or isinstance(result, dict)

    def test_get_status(self, mock_storage_provider, mock_transport_config):
        """Test get_status returns expected structure."""
        from app.distributed.sync_coordinator import SyncCoordinator
        
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )
        
        status = coordinator.get_status()
        
        assert "provider" in status
        assert "running" in status
        assert "transports" in status


class TestHighQualityDataSyncWatcher:
    """Tests for HighQualityDataSyncWatcher class."""

    def test_initialization(self, mock_storage_provider, mock_transport_config):
        """Test HighQualityDataSyncWatcher initialization."""
        from app.distributed.sync_coordinator import (
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        # Create coordinator first
        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        watcher = HighQualityDataSyncWatcher()

        # Check the actual attribute names
        assert watcher.min_quality_score == 0.7
        assert watcher.sync_cooldown_seconds == 60.0
        assert watcher.max_games_per_sync == 500

    def test_custom_thresholds(self, mock_storage_provider, mock_transport_config):
        """Test watcher with custom thresholds."""
        from app.distributed.sync_coordinator import (
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        watcher = HighQualityDataSyncWatcher(
            min_quality_score=0.8,
            sync_cooldown_seconds=120.0,
            max_games_per_sync=1000,
        )

        assert watcher.min_quality_score == 0.8
        assert watcher.sync_cooldown_seconds == 120.0
        assert watcher.max_games_per_sync == 1000

    def test_health_check(self, mock_storage_provider, mock_transport_config):
        """Test watcher health check."""
        from app.distributed.sync_coordinator import (
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        watcher = HighQualityDataSyncWatcher()
        result = watcher.health_check()

        assert hasattr(result, "healthy") or isinstance(result, dict)

    def test_force_sync(self, mock_storage_provider, mock_transport_config):
        """Test force sync trigger."""
        from app.distributed.sync_coordinator import (
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        watcher = HighQualityDataSyncWatcher()
        result = watcher.force_sync()

        # Returns True if sync was triggered
        assert isinstance(result, bool)


class TestWiringFunctions:
    """Tests for event wiring functions."""

    def test_wire_high_quality_to_sync(self, mock_storage_provider, mock_transport_config):
        """Test wire_high_quality_to_sync function."""
        from app.distributed.sync_coordinator import (
            wire_high_quality_to_sync,
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # Should return a HighQualityDataSyncWatcher instance
        result = wire_high_quality_to_sync()

        assert isinstance(result, HighQualityDataSyncWatcher)

    def test_wire_all_quality_events_to_sync(self, mock_storage_provider, mock_transport_config):
        """Test wire_all_quality_events_to_sync function."""
        from app.distributed.sync_coordinator import (
            wire_all_quality_events_to_sync,
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # Should return a HighQualityDataSyncWatcher instance
        result = wire_all_quality_events_to_sync()

        assert isinstance(result, HighQualityDataSyncWatcher)

    def test_get_high_quality_sync_watcher(self, mock_storage_provider, mock_transport_config):
        """Test get_high_quality_sync_watcher function."""
        from app.distributed.sync_coordinator import (
            get_high_quality_sync_watcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # May return None if not wired
        watcher = get_high_quality_sync_watcher()

        # Either None or a watcher instance
        assert watcher is None or hasattr(watcher, "health_check")

    def test_wire_and_get_watcher(self, mock_storage_provider, mock_transport_config):
        """Test wiring then retrieving watcher."""
        from app.distributed.sync_coordinator import (
            wire_high_quality_to_sync,
            get_high_quality_sync_watcher,
            HighQualityDataSyncWatcher,
            SyncCoordinator,
        )

        SyncCoordinator._instance = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        # Wire first
        watcher1 = wire_high_quality_to_sync()

        # Get should return the same watcher
        watcher2 = get_high_quality_sync_watcher()

        assert watcher2 is watcher1
        assert isinstance(watcher2, HighQualityDataSyncWatcher)
