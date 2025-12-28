"""Comprehensive tests for app.distributed.sync_coordinator module.

Tests for the unified data synchronization execution layer:
- SyncCategory enum
- SyncStats and ClusterSyncStats dataclasses
- SyncOperationBudget timeout tracking
- SyncCoordinator main class
- HighQualityDataSyncWatcher event handling
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.distributed.sync_coordinator import (
    ClusterSyncStats,
    SyncCategory,
    SyncCoordinator,
    SyncOperationBudget,
    SyncStats,
)


def safe_reset_singleton():
    """Safely reset SyncCoordinator singleton without requiring async event loop."""
    if SyncCoordinator._instance is not None:
        # Directly clear instance without async shutdown (for test cleanup)
        SyncCoordinator._instance._running = False
        SyncCoordinator._instance = None


@pytest.fixture(autouse=True)
def reset_sync_coordinator_singleton():
    """Reset singleton before and after each test (autouse ensures proper isolation)."""
    # Reset before test
    safe_reset_singleton()
    yield
    # Reset after test (runs even if test fails)
    safe_reset_singleton()


@pytest.fixture
def mock_init_manifest():
    """Mock _init_manifest to prevent database initialization during tests."""
    with patch.object(SyncCoordinator, '_init_manifest', return_value=None) as mock:
        yield mock


# =============================================================================
# Test SyncCategory Enum
# =============================================================================


class TestSyncCategory:
    """Tests for SyncCategory enum."""

    def test_category_values(self):
        """All expected category values exist."""
        assert SyncCategory.GAMES.value == "games"
        assert SyncCategory.MODELS.value == "models"
        assert SyncCategory.TRAINING.value == "training"
        assert SyncCategory.ELO.value == "elo"
        assert SyncCategory.ALL.value == "all"

    def test_category_count(self):
        """Expected number of categories."""
        assert len(SyncCategory) == 5

    def test_category_from_string(self):
        """Can create category from string value."""
        assert SyncCategory("games") == SyncCategory.GAMES
        assert SyncCategory("models") == SyncCategory.MODELS


# =============================================================================
# Test SyncStats Dataclass
# =============================================================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        stats = SyncStats(category="test")
        assert stats.category == "test"
        assert stats.files_synced == 0
        assert stats.files_failed == 0
        assert stats.bytes_transferred == 0
        assert stats.duration_seconds == 0.0
        assert stats.transport_used == ""
        assert stats.sources_tried == 0
        assert stats.errors == []
        assert stats.high_quality_games_synced == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_elo == 0.0

    def test_success_rate_no_files(self):
        """Success rate is 1.0 when no files processed."""
        stats = SyncStats(category="test")
        assert stats.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Success rate is 1.0 when all files succeed."""
        stats = SyncStats(category="test", files_synced=10, files_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_all_failed(self):
        """Success rate is 0.0 when all files fail."""
        stats = SyncStats(category="test", files_synced=0, files_failed=10)
        assert stats.success_rate == 0.0

    def test_success_rate_mixed(self):
        """Success rate is calculated correctly for mixed results."""
        stats = SyncStats(category="test", files_synced=7, files_failed=3)
        assert stats.success_rate == 0.7

    def test_with_quality_stats(self):
        """Quality-aware stats are tracked."""
        stats = SyncStats(
            category="games",
            files_synced=100,
            high_quality_games_synced=80,
            avg_quality_score=0.85,
            avg_elo=1650.0,
        )
        assert stats.high_quality_games_synced == 80
        assert stats.avg_quality_score == 0.85
        assert stats.avg_elo == 1650.0


# =============================================================================
# Test ClusterSyncStats Dataclass
# =============================================================================


class TestClusterSyncStats:
    """Tests for ClusterSyncStats dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        stats = ClusterSyncStats()
        assert stats.total_files_synced == 0
        assert stats.total_bytes_transferred == 0
        assert stats.duration_seconds == 0.0
        assert stats.categories == {}
        assert stats.transport_distribution == {}
        assert stats.nodes_synced == 0
        assert stats.nodes_failed == 0
        assert stats.total_high_quality_games == 0
        assert stats.avg_quality_score == 0.0
        assert stats.quality_distribution == {}

    def test_with_categories(self):
        """Can store category-specific stats."""
        games_stats = SyncStats(category="games", files_synced=50)
        models_stats = SyncStats(category="models", files_synced=5)

        stats = ClusterSyncStats(
            total_files_synced=55,
            categories={"games": games_stats, "models": models_stats},
        )
        assert stats.categories["games"].files_synced == 50
        assert stats.categories["models"].files_synced == 5

    def test_transport_distribution(self):
        """Tracks which transports were used."""
        stats = ClusterSyncStats(
            transport_distribution={"aria2": 40, "rsync": 10, "p2p": 5}
        )
        assert stats.transport_distribution["aria2"] == 40
        assert sum(stats.transport_distribution.values()) == 55


# =============================================================================
# Test SyncOperationBudget
# =============================================================================


class TestSyncOperationBudget:
    """Tests for SyncOperationBudget timeout tracking."""

    def test_default_values(self):
        """Default budget values are set."""
        budget = SyncOperationBudget()
        assert budget.total_seconds == 300.0
        assert budget.per_attempt_seconds == 30.0
        assert budget.attempts == 0

    def test_custom_values(self):
        """Custom budget values can be set."""
        budget = SyncOperationBudget(
            total_seconds=600.0,
            per_attempt_seconds=60.0,
        )
        assert budget.total_seconds == 600.0
        assert budget.per_attempt_seconds == 60.0

    def test_elapsed_increases(self):
        """Elapsed time increases."""
        budget = SyncOperationBudget(total_seconds=10.0)
        time.sleep(0.05)
        assert budget.elapsed >= 0.05

    def test_remaining_decreases(self):
        """Remaining time decreases."""
        budget = SyncOperationBudget(total_seconds=10.0)
        initial_remaining = budget.remaining
        time.sleep(0.05)
        assert budget.remaining < initial_remaining

    def test_exhausted_false_initially(self):
        """Budget is not exhausted initially."""
        budget = SyncOperationBudget(total_seconds=10.0)
        assert not budget.exhausted

    def test_exhausted_after_timeout(self):
        """Budget is exhausted after total_seconds."""
        budget = SyncOperationBudget(total_seconds=0.05)
        time.sleep(0.1)
        assert budget.exhausted

    def test_get_attempt_timeout_normal(self):
        """Attempt timeout is per_attempt_seconds when budget allows."""
        budget = SyncOperationBudget(
            total_seconds=100.0,
            per_attempt_seconds=30.0,
        )
        assert budget.get_attempt_timeout() == 30.0

    def test_get_attempt_timeout_capped(self):
        """Attempt timeout is capped by remaining budget."""
        budget = SyncOperationBudget(
            total_seconds=5.0,
            per_attempt_seconds=30.0,
        )
        # With 5s total and 30s per attempt, the cap should be ~5.0
        timeout = budget.get_attempt_timeout()
        # Use approximate comparison due to time elapsed between creation and check
        assert 4.9 < timeout <= 5.0  # min(30.0, remaining) ≈ 5.0
        # Verify it's less than per_attempt_seconds (the cap is working)
        assert timeout < budget.per_attempt_seconds

    def test_record_attempt(self):
        """Attempts are counted."""
        budget = SyncOperationBudget()
        assert budget.attempts == 0
        budget.record_attempt()
        assert budget.attempts == 1
        budget.record_attempt()
        assert budget.attempts == 2

    def test_can_attempt_true(self):
        """Can attempt when budget allows."""
        budget = SyncOperationBudget(total_seconds=10.0)
        assert budget.can_attempt()

    def test_can_attempt_false_exhausted(self):
        """Cannot attempt when budget exhausted."""
        budget = SyncOperationBudget(total_seconds=0.05)
        time.sleep(0.1)
        assert not budget.can_attempt()

    def test_budget_usage_pattern(self):
        """Typical budget usage pattern works."""
        # Note: can_attempt() requires at least 1.0s remaining
        budget = SyncOperationBudget(
            total_seconds=5.0,
            per_attempt_seconds=0.5,
        )

        attempts = 0
        max_attempts = 3
        while budget.can_attempt() and attempts < max_attempts:
            timeout = budget.get_attempt_timeout()
            assert timeout > 0
            time.sleep(0.05)  # Simulate operation
            budget.record_attempt()
            attempts += 1

        assert attempts >= 1
        assert budget.attempts == attempts


# =============================================================================
# Test SyncCoordinator - Initialization
# =============================================================================


class TestSyncCoordinatorInit:
    """Tests for SyncCoordinator initialization."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_init_with_defaults(self, mock_config, mock_provider, mock_init_manifest):
        """Coordinator initializes with default provider and config."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        assert coord._provider is not None
        assert coord._config is not None
        assert coord._running is False

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_init_with_custom_provider(self, mock_config, mock_provider, mock_init_manifest):
        """Coordinator accepts custom provider."""
        custom_provider = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator(provider=custom_provider)

        assert coord._provider is custom_provider

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_singleton_pattern(self, mock_config, mock_provider, mock_init_manifest):
        """get_instance returns singleton."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        instance1 = SyncCoordinator.get_instance()
        instance2 = SyncCoordinator.get_instance()

        assert instance1 is instance2

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_reset_instance(self, mock_config, mock_provider, mock_init_manifest):
        """reset_instance clears the singleton."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        instance1 = SyncCoordinator.get_instance()
        safe_reset_singleton()  # Use safe version in tests
        instance2 = SyncCoordinator.get_instance()

        assert instance1 is not instance2


# =============================================================================
# Test SyncCoordinator - Transport Initialization
# =============================================================================


class TestSyncCoordinatorTransports:
    """Tests for SyncCoordinator transport initialization."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    @patch("app.distributed.sync_coordinator.HAS_ARIA2", True)
    @patch("app.distributed.sync_coordinator.check_aria2_available")
    def test_init_aria2_available(self, mock_check, mock_config, mock_provider, mock_init_manifest):
        """Aria2 transport is initialized when available."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()
        mock_check.return_value = True

        coord = SyncCoordinator()
        # Aria2 is lazily initialized, trigger it
        with patch("app.distributed.sync_coordinator.Aria2Transport") as mock_aria2:
            mock_aria2.return_value = MagicMock()
            result = coord._init_aria2()
            # Should attempt initialization

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    @patch("app.distributed.sync_coordinator.HAS_P2P", True)
    def test_init_p2p_transport(self, mock_config, mock_provider, mock_init_manifest):
        """P2P transport can be initialized."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        with patch("app.distributed.sync_coordinator.P2PSyncClient") as mock_p2p:
            mock_p2p.return_value = MagicMock()
            result = coord._init_p2p()
            # P2P initialization attempted


# =============================================================================
# Test SyncCoordinator - Quality Lookup
# =============================================================================


class TestSyncCoordinatorQuality:
    """Tests for SyncCoordinator quality-aware features."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_get_quality_lookup_empty(self, mock_config, mock_provider, mock_init_manifest):
        """Quality lookup returns empty dict when not initialized."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        coord._quality_lookup = {}

        lookup = coord.get_quality_lookup()
        assert isinstance(lookup, dict)

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_get_elo_lookup_empty(self, mock_config, mock_provider, mock_init_manifest):
        """ELO lookup returns empty dict when not initialized."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        coord._elo_lookup = {}

        lookup = coord.get_elo_lookup()
        assert isinstance(lookup, dict)

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_get_manifest_none(self, mock_config, mock_provider, mock_init_manifest):
        """Manifest returns None when not initialized."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        coord._manifest = None

        assert coord.get_manifest() is None


# =============================================================================
# Test SyncCoordinator - File Snapshot Helpers
# =============================================================================


class TestSyncCoordinatorSnapshots:
    """Tests for file snapshot helper methods."""

    def test_snapshot_files_empty_dir(self, tmp_path):
        """Snapshot of empty directory returns empty dict."""
        result = SyncCoordinator._snapshot_files(tmp_path, ["*.db"])
        assert result == {}

    def test_snapshot_files_with_files(self, tmp_path):
        """Snapshot captures file sizes."""
        (tmp_path / "test1.db").write_bytes(b"data1")
        (tmp_path / "test2.db").write_bytes(b"data2data2")

        result = SyncCoordinator._snapshot_files(tmp_path, ["*.db"])

        assert len(result) == 2
        # Check by filename suffix to avoid macOS /private path issues
        sizes = {Path(k).name: v for k, v in result.items()}
        assert sizes["test1.db"] == 5
        assert sizes["test2.db"] == 10

    def test_snapshot_files_pattern_filtering(self, tmp_path):
        """Snapshot respects glob patterns."""
        (tmp_path / "test.db").write_text("data")
        (tmp_path / "test.txt").write_text("text")

        result = SyncCoordinator._snapshot_files(tmp_path, ["*.db"])

        assert len(result) == 1
        assert any("test.db" in k for k in result)

    def test_diff_snapshot_new_files(self):
        """Diff detects new files."""
        before = {"/a.db": 100}
        after = {"/a.db": 100, "/b.db": 200}

        new_count, bytes_added = SyncCoordinator._diff_snapshot(before, after)

        assert new_count == 1
        assert bytes_added == 200

    def test_diff_snapshot_no_change(self):
        """Diff detects no change."""
        before = {"/a.db": 100, "/b.db": 200}
        after = {"/a.db": 100, "/b.db": 200}

        new_count, bytes_added = SyncCoordinator._diff_snapshot(before, after)

        assert new_count == 0
        assert bytes_added == 0

    def test_diff_snapshot_file_grew(self):
        """Diff only counts new files, not file growth (by design)."""
        before = {"/a.db": 100}
        after = {"/a.db": 150}

        new_count, bytes_added = SyncCoordinator._diff_snapshot(before, after)

        # _diff_snapshot only tracks NEW files, not size changes
        assert new_count == 0
        assert bytes_added == 0


# =============================================================================
# Test SyncCoordinator - Sync Methods (Mocked)
# =============================================================================


class TestSyncCoordinatorSyncMethods:
    """Tests for sync methods with mocked transports."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @pytest.mark.asyncio
    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    async def test_sync_training_data_nfs_skip(self, mock_config, mock_provider, mock_init_manifest):
        """NFS provider skips sync (shared storage)."""
        provider = MagicMock()
        provider.name = "nfs"
        provider.skip_sync = True
        mock_provider.return_value = provider
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        stats = await coord.sync_training_data()

        # NFS skip should return quickly with empty stats
        assert isinstance(stats, SyncStats)

    @pytest.mark.asyncio
    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    async def test_is_data_server_running_false(self, mock_config, mock_provider, mock_init_manifest):
        """Data server is not running initially."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        assert not coord.is_data_server_running()


# =============================================================================
# Test SyncCoordinator - Status and Health
# =============================================================================


class TestSyncCoordinatorStatus:
    """Tests for status and health reporting."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_get_status(self, mock_config, mock_provider, mock_init_manifest):
        """get_status returns expected structure."""
        provider = MagicMock()
        provider.provider_type.value = "local"
        provider.has_shared_storage = False
        mock_provider.return_value = provider
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        status = coord.get_status()

        assert isinstance(status, dict)
        assert "provider" in status
        assert "transports" in status
        assert "running" in status

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_get_sync_health(self, mock_config, mock_provider, mock_init_manifest):
        """get_sync_health returns health metrics."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        health = coord.get_sync_health()

        assert isinstance(health, dict)
        assert "consecutive_failures" in health
        assert "last_successful_sync" in health


# =============================================================================
# Test SyncCoordinator - Background Sync
# =============================================================================


class TestSyncCoordinatorBackgroundSync:
    """Tests for background sync functionality."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @pytest.mark.asyncio
    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    async def test_stop_background_sync_when_not_running(self, mock_config, mock_provider, mock_init_manifest):
        """Stopping background sync when not running is safe."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        coord._background_sync_task = None

        # Should not raise
        await coord.stop_background_sync()

    @pytest.mark.asyncio
    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    async def test_shutdown(self, mock_config, mock_provider, mock_init_manifest):
        """Shutdown cleans up resources."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        # Should not raise
        await coord.shutdown()


# =============================================================================
# Test HighQualityDataSyncWatcher
# =============================================================================


class TestHighQualityDataSyncWatcher:
    """Tests for HighQualityDataSyncWatcher event handling."""

    def teardown_method(self):
        """Reset coordinator singleton."""
        safe_reset_singleton()

    def test_watcher_initialization(self):
        """Watcher initializes with defaults."""
        from app.distributed.sync_coordinator import HighQualityDataSyncWatcher

        watcher = HighQualityDataSyncWatcher()

        assert watcher.sync_cooldown_seconds == 60.0
        assert watcher.min_quality_score == 0.7
        assert watcher.max_games_per_sync == 500
        assert watcher._subscribed is False

    def test_watcher_custom_initialization(self):
        """Watcher initializes with custom values."""
        from app.distributed.sync_coordinator import HighQualityDataSyncWatcher

        watcher = HighQualityDataSyncWatcher(
            sync_cooldown_seconds=120.0,
            min_quality_score=0.8,
            max_games_per_sync=1000,
        )

        assert watcher.sync_cooldown_seconds == 120.0
        assert watcher.min_quality_score == 0.8
        assert watcher.max_games_per_sync == 1000

    def test_watcher_internal_state(self):
        """Watcher maintains internal sync state."""
        from app.distributed.sync_coordinator import HighQualityDataSyncWatcher

        watcher = HighQualityDataSyncWatcher()

        # Check initial state
        assert watcher._sync_in_progress is False
        assert len(watcher._pending_hosts) == 0
        assert watcher._last_sync_time == 0.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestSyncCoordinatorEdgeCases:
    """Edge cases and error handling tests."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_resolve_games_dir_default(self, mock_config, mock_provider, tmp_path, mock_init_manifest):
        """Games directory resolves from provider's selfplay_dir."""
        # Create mock selfplay dir structure
        selfplay_dir = tmp_path / "selfplay"
        selfplay_dir.mkdir()
        games_dir = tmp_path / "games"
        games_dir.mkdir()

        provider = MagicMock()
        provider.selfplay_dir = selfplay_dir
        mock_provider.return_value = provider
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()
        resolved = coord._resolve_games_dir()

        assert isinstance(resolved, Path)
        # Should resolve to games/ since parent/games exists
        assert str(resolved) == str(games_dir)

    def test_sync_operation_budget_zero_total(self):
        """Budget with zero total is immediately exhausted."""
        budget = SyncOperationBudget(total_seconds=0)
        assert budget.exhausted
        assert not budget.can_attempt()

    def test_sync_stats_errors_list_mutable(self):
        """Errors list is correctly managed."""
        stats1 = SyncStats(category="test1")
        stats2 = SyncStats(category="test2")

        stats1.errors.append("error1")

        # Should not affect stats2
        assert len(stats2.errors) == 0

    @patch("app.distributed.sync_coordinator.get_storage_provider")
    @patch("app.distributed.sync_coordinator.get_optimal_transport_config")
    def test_coordinator_tracks_last_sync_times(self, mock_config, mock_provider, mock_init_manifest):
        """Coordinator maintains last sync time tracking."""
        mock_provider.return_value = MagicMock()
        mock_config.return_value = MagicMock()

        coord = SyncCoordinator()

        assert isinstance(coord._last_sync_times, dict)
        assert isinstance(coord._sync_stats, dict)


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestSyncCoordinatorIntegration:
    """Integration-style tests for sync scenarios."""

    def teardown_method(self):
        """Reset singleton after each test."""
        safe_reset_singleton()

    def test_sync_stats_aggregation(self):
        """Multiple SyncStats can be aggregated into ClusterSyncStats."""
        games = SyncStats(
            category="games",
            files_synced=100,
            bytes_transferred=1000000,
            high_quality_games_synced=80,
        )
        models = SyncStats(
            category="models",
            files_synced=5,
            bytes_transferred=500000,
        )

        cluster = ClusterSyncStats(
            total_files_synced=105,
            total_bytes_transferred=1500000,
            categories={"games": games, "models": models},
            nodes_synced=3,
            total_high_quality_games=80,
        )

        assert cluster.total_files_synced == 105
        assert cluster.categories["games"].success_rate == 1.0

    def test_budget_retry_simulation(self):
        """Simulate retry loop with budget."""
        # Note: can_attempt() requires >= 1.0s remaining
        budget = SyncOperationBudget(
            total_seconds=10.0,
            per_attempt_seconds=2.0,
        )

        results = []
        max_attempts = 3
        for _ in range(max_attempts):
            if not budget.can_attempt():
                break
            timeout = budget.get_attempt_timeout()
            # Simulate varying operation times
            sleep_time = min(0.01, timeout)
            time.sleep(sleep_time)
            budget.record_attempt()
            results.append({"timeout": timeout, "attempt": budget.attempts})

        # Should have made multiple attempts
        assert budget.attempts >= 2
        assert all(r["timeout"] > 0 for r in results)
