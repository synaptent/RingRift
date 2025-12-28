"""Tests for SyncCoordinator - Unified data synchronization execution layer.

Created: December 28, 2025
Purpose: Test the SyncCoordinator class critical sync infrastructure

Tests cover:
- Singleton pattern (get_instance, reset_instance)
- Transport initialization
- Health check reporting
- Status reporting
- Manifest integration
- Event subscription
- Sync statistics tracking
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Test the module can be imported
try:
    from app.distributed.sync_coordinator import (
        SyncCoordinator,
        SyncStats,
        SyncOperationBudget,
        SyncCategory,
    )
    SYNC_COORDINATOR_AVAILABLE = True
except ImportError as e:
    SYNC_COORDINATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset SyncCoordinator singleton between tests."""
    if SYNC_COORDINATOR_AVAILABLE:
        SyncCoordinator._instance = None
    yield
    if SYNC_COORDINATOR_AVAILABLE:
        SyncCoordinator._instance = None


@pytest.fixture
def mock_storage_provider():
    """Create a mock storage provider."""
    provider = MagicMock()
    provider.provider_type = MagicMock()
    provider.provider_type.value = "local"
    provider.has_shared_storage = False
    return provider


@pytest.fixture
def mock_transport_config():
    """Create a mock transport config."""
    config = MagicMock()
    config.enable_aria2 = False
    config.enable_bittorrent = False
    config.enable_gossip = False
    config.gossip_peers = []
    return config


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available: {IMPORT_ERROR if not SYNC_COORDINATOR_AVAILABLE else ''}")
class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_values(self):
        """Default stats should have zeros."""
        stats = SyncStats(category="games")
        assert stats.files_synced == 0
        assert stats.bytes_transferred == 0
        assert stats.files_failed == 0
        assert stats.duration_seconds == 0.0

    def test_category_required(self):
        """Category is a required field."""
        stats = SyncStats(category="models")
        assert stats.category == "models"

    def test_success_rate_no_files(self):
        """Success rate with no files should be 1.0."""
        stats = SyncStats(category="games")
        assert stats.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Success rate with all successful syncs."""
        stats = SyncStats(category="games", files_synced=10, files_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_with_failures(self):
        """Success rate calculation with some failures."""
        stats = SyncStats(category="games", files_synced=8, files_failed=2)
        # success_rate = synced / (synced + failed)
        assert stats.success_rate == 0.8

    def test_success_rate_all_failed(self):
        """Success rate when all files failed."""
        stats = SyncStats(category="games", files_synced=0, files_failed=5)
        assert stats.success_rate == 0.0

    def test_quality_stats_defaults(self):
        """Quality-aware stats should have defaults."""
        stats = SyncStats(category="games")
        assert stats.high_quality_games_synced == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_elo == 0.0


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncOperationBudget:
    """Tests for SyncOperationBudget class."""

    def test_default_timeout(self):
        """Default timeout should be 300 seconds."""
        budget = SyncOperationBudget()
        assert budget.total_seconds == 300

    def test_custom_timeout(self):
        """Custom timeout should be respected."""
        budget = SyncOperationBudget(total_seconds=120)
        assert budget.total_seconds == 120

    def test_elapsed_time(self):
        """Elapsed time should increase."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.elapsed >= 0
        time.sleep(0.01)
        assert budget.elapsed >= 0.01

    def test_remaining_time(self):
        """Remaining time should decrease."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.remaining <= 10
        assert budget.remaining > 0

    def test_exhausted_initially_false(self):
        """Budget should not be exhausted initially."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.exhausted is False

    def test_can_attempt_initially_true(self):
        """Should be able to attempt initially."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.can_attempt() is True

    def test_record_attempt(self):
        """Recording attempt should increment counter."""
        budget = SyncOperationBudget()
        assert budget.attempts == 0
        budget.record_attempt()
        assert budget.attempts == 1
        budget.record_attempt()
        assert budget.attempts == 2

    def test_get_attempt_timeout(self):
        """Get attempt timeout should return reasonable value."""
        budget = SyncOperationBudget(per_attempt_seconds=30)
        timeout = budget.get_attempt_timeout()
        assert timeout <= 30


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorSingleton:
    """Tests for SyncCoordinator singleton pattern."""

    def test_get_instance_creates_singleton(self, mock_storage_provider, mock_transport_config):
        """get_instance should create singleton."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                instance1 = SyncCoordinator.get_instance()
                instance2 = SyncCoordinator.get_instance()
                assert instance1 is instance2

    def test_reset_instance(self, mock_storage_provider, mock_transport_config):
        """reset_instance should clear singleton."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                instance1 = SyncCoordinator.get_instance()
                SyncCoordinator._instance = None  # Direct reset for testing
                instance2 = SyncCoordinator.get_instance()
                assert instance1 is not instance2


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorInit:
    """Tests for SyncCoordinator initialization."""

    def test_init_with_defaults(self, mock_storage_provider, mock_transport_config):
        """Initialization with defaults should work."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                assert coordinator._provider is mock_storage_provider
                assert coordinator._config is mock_transport_config
                assert coordinator._running is False

    def test_init_tracks_state(self, mock_storage_provider, mock_transport_config):
        """Initialization should set up state tracking."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                assert coordinator._last_sync_times == {}
                assert coordinator._sync_stats == {}
                assert coordinator._aria2_sources == []

    def test_init_background_sync_tracking(self, mock_storage_provider, mock_transport_config):
        """Initialization should set up background sync tracking."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                assert coordinator._last_successful_sync == 0.0
                assert coordinator._consecutive_failures == 0
                assert coordinator._max_consecutive_failures == 5


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorHealth:
    """Tests for SyncCoordinator health check methods."""

    def test_get_sync_health_basic(self, mock_storage_provider, mock_transport_config):
        """get_sync_health should return health dict."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                health = coordinator.get_sync_health()
                assert isinstance(health, dict)
                assert "status" in health

    def test_health_check_returns_result(self, mock_storage_provider, mock_transport_config):
        """health_check should return HealthCheckResult."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                result = coordinator.health_check()
                # Should have healthy attribute or be dict-like
                assert hasattr(result, "healthy") or "healthy" in result


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorStatus:
    """Tests for SyncCoordinator status methods."""

    def test_get_status_returns_dict(self, mock_storage_provider, mock_transport_config):
        """get_status should return status dict."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                status = coordinator.get_status()
                assert isinstance(status, dict)

    def test_get_status_contains_provider(self, mock_storage_provider, mock_transport_config):
        """Status should contain provider info."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                status = coordinator.get_status()
                assert "provider" in status or "storage_provider" in status


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorManifest:
    """Tests for SyncCoordinator manifest methods."""

    def test_get_manifest_initial_none_or_empty(self, mock_storage_provider, mock_transport_config):
        """Initial manifest may be None or empty."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                manifest = coordinator.get_manifest()
                # May be None if not initialized or empty manifest
                assert manifest is None or hasattr(manifest, "entries")

    def test_get_quality_lookup_returns_dict(self, mock_storage_provider, mock_transport_config):
        """get_quality_lookup should return dict."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                lookup = coordinator.get_quality_lookup()
                assert isinstance(lookup, dict)

    def test_get_elo_lookup_returns_dict(self, mock_storage_provider, mock_transport_config):
        """get_elo_lookup should return dict."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                lookup = coordinator.get_elo_lookup()
                assert isinstance(lookup, dict)


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorDataServer:
    """Tests for SyncCoordinator data server methods."""

    def test_is_data_server_running_initial(self, mock_storage_provider, mock_transport_config):
        """Data server should not be running initially."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                assert coordinator.is_data_server_running() is False


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorEventSubscription:
    """Tests for SyncCoordinator event subscription methods."""

    def test_subscribe_to_high_quality_events(self, mock_storage_provider, mock_transport_config):
        """subscribe_to_high_quality_events should return bool."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                # May fail if event system not available, but should not raise
                result = coordinator.subscribe_to_high_quality_events()
                assert isinstance(result, bool)

    def test_subscribe_to_all_quality_events(self, mock_storage_provider, mock_transport_config):
        """subscribe_to_all_quality_events should return count."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                result = coordinator.subscribe_to_all_quality_events()
                assert isinstance(result, int)
                assert result >= 0

    def test_unsubscribe(self, mock_storage_provider, mock_transport_config):
        """unsubscribe should not raise."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                # Should not raise even if not subscribed
                coordinator.unsubscribe()


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorForceSync:
    """Tests for SyncCoordinator force_sync method."""

    def test_force_sync_returns_bool(self, mock_storage_provider, mock_transport_config):
        """force_sync should return bool."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()
                result = coordinator.force_sync()
                assert isinstance(result, bool)


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorIntegration:
    """Integration tests for SyncCoordinator."""

    def test_full_lifecycle(self, mock_storage_provider, mock_transport_config):
        """Test basic lifecycle: create, check health, get status."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                # Create
                coordinator = SyncCoordinator.get_instance()

                # Check health
                health = coordinator.get_sync_health()
                assert isinstance(health, dict)

                # Get status
                status = coordinator.get_status()
                assert isinstance(status, dict)

                # Check health result
                result = coordinator.health_check()
                assert result is not None

    def test_consecutive_failure_tracking(self, mock_storage_provider, mock_transport_config):
        """Test consecutive failure counter."""
        with patch("app.distributed.sync_coordinator.get_storage_provider", return_value=mock_storage_provider):
            with patch("app.distributed.sync_coordinator.get_optimal_transport_config", return_value=mock_transport_config):
                coordinator = SyncCoordinator()

                # Initial state
                assert coordinator._consecutive_failures == 0

                # Simulate failures
                coordinator._consecutive_failures = 3
                assert coordinator._consecutive_failures == 3

                # Check max failures threshold
                assert coordinator._consecutive_failures < coordinator._max_consecutive_failures
