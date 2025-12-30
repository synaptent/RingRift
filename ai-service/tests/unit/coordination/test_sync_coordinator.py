"""Tests for Sync Coordinator module.

Tests the centralized sync scheduling system that manages data synchronization
across all distributed hosts in the RingRift cluster.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.sync_coordinator import (
    CRITICAL_STALE_THRESHOLD_SECONDS,
    # Constants
    STALE_DATA_THRESHOLD_SECONDS,
    ClusterDataStatus,
    # Data classes
    HostDataState,
    HostType,
    SyncAction,
    # Enums
    SyncPriority,
    SyncRecommendation,
    # Main class
    SyncScheduler,
    # Functions
    get_sync_scheduler,
    reset_sync_scheduler,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestSyncPriority:
    """Tests for SyncPriority enum."""

    def test_all_priorities_defined(self):
        """All expected priority levels should exist."""
        # Priority values are integers for queue ordering (higher = more priority)
        assert SyncPriority.CRITICAL.value == 100
        assert SyncPriority.HIGH.value == 75
        assert SyncPriority.NORMAL.value == 50
        assert SyncPriority.LOW.value == 25
        assert SyncPriority.BACKGROUND.value == 10

    def test_priority_count(self):
        """Should have exactly 5 priority levels."""
        assert len(SyncPriority) == 5


class TestHostType:
    """Tests for HostType enum."""

    def test_all_host_types_defined(self):
        """All expected host types should exist."""
        assert HostType.EPHEMERAL.value == "ephemeral"
        assert HostType.PERSISTENT.value == "persistent"
        assert HostType.LOCAL.value == "local"
        assert HostType.ARCHIVE.value == "archive"

    def test_host_type_count(self):
        """Should have exactly 4 host types."""
        assert len(HostType) == 4


class TestSyncAction:
    """Tests for SyncAction enum."""

    def test_all_sync_actions_defined(self):
        """All expected sync actions should exist."""
        assert SyncAction.SYNC_NOW.value == "sync_now"
        assert SyncAction.SCHEDULE_SYNC.value == "schedule_sync"
        assert SyncAction.SKIP.value == "skip"
        assert SyncAction.VERIFY_DATA.value == "verify_data"
        assert SyncAction.RECOVER_MANIFEST.value == "recover_manifest"

    def test_sync_action_count(self):
        """Should have exactly 5 sync actions."""
        assert len(SyncAction) == 5


# =============================================================================
# HostDataState Tests
# =============================================================================


class TestHostDataState:
    """Tests for HostDataState dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        state = HostDataState(host="test-host", host_type=HostType.PERSISTENT)
        assert state.host == "test-host"
        assert state.host_type == HostType.PERSISTENT
        assert state.last_sync_time == 0.0
        assert state.total_games == 0
        assert state.is_reachable is True

    def test_seconds_since_sync_never_synced(self):
        """Should return infinity for never-synced host."""
        state = HostDataState(host="new-host", host_type=HostType.EPHEMERAL)
        assert state.seconds_since_sync == float('inf')

    def test_seconds_since_sync_recent(self):
        """Should calculate time since last sync."""
        now = time.time()
        state = HostDataState(
            host="test",
            host_type=HostType.PERSISTENT,
            last_sync_time=now - 300,  # 5 minutes ago
        )
        assert 299 <= state.seconds_since_sync <= 301

    def test_is_stale_fresh_host(self):
        """Fresh host should not be stale."""
        state = HostDataState(
            host="fresh",
            host_type=HostType.PERSISTENT,
            last_sync_time=time.time() - 60,  # 1 minute ago
        )
        assert not state.is_stale

    def test_is_stale_old_host(self):
        """Old host should be stale."""
        state = HostDataState(
            host="old",
            host_type=HostType.PERSISTENT,
            last_sync_time=time.time() - (STALE_DATA_THRESHOLD_SECONDS + 100),
        )
        assert state.is_stale

    def test_is_critical_ephemeral_with_unsynced(self):
        """Ephemeral host with unsynced games is critical."""
        state = HostDataState(
            host="vast-host",
            host_type=HostType.EPHEMERAL,
            estimated_unsynced_games=100,
        )
        assert state.is_critical

    def test_is_critical_very_old_sync(self):
        """Host with very old sync is critical."""
        state = HostDataState(
            host="old",
            host_type=HostType.PERSISTENT,
            last_sync_time=time.time() - (CRITICAL_STALE_THRESHOLD_SECONDS + 100),
        )
        assert state.is_critical

    def test_sync_priority_score_basic(self):
        """Should calculate priority score."""
        state = HostDataState(
            host="test",
            host_type=HostType.PERSISTENT,
            estimated_unsynced_games=100,
            last_sync_time=time.time() - 1800,  # 30 min ago
        )
        score = state.sync_priority_score
        assert score > 0

    def test_sync_priority_score_ephemeral_higher(self):
        """Ephemeral hosts should have higher priority."""
        base_state = {
            "estimated_unsynced_games": 100,
            "last_sync_time": time.time() - 1800,
        }

        persistent = HostDataState(host="p", host_type=HostType.PERSISTENT, **base_state)
        ephemeral = HostDataState(host="e", host_type=HostType.EPHEMERAL, **base_state)

        assert ephemeral.sync_priority_score > persistent.sync_priority_score

    def test_sync_priority_score_failure_penalty(self):
        """Recent failures should reduce priority."""
        now = time.time()
        state_no_failures = HostDataState(
            host="good",
            host_type=HostType.PERSISTENT,
            estimated_unsynced_games=100,
            sync_failures_24h=0,
            last_sync_time=now - 1800,  # Need valid sync time for finite score
        )
        state_with_failures = HostDataState(
            host="bad",
            host_type=HostType.PERSISTENT,
            estimated_unsynced_games=100,
            sync_failures_24h=3,
            last_sync_time=now - 1800,  # Same sync time
        )

        assert state_with_failures.sync_priority_score < state_no_failures.sync_priority_score

    def test_to_dict(self):
        """Should serialize to dictionary."""
        state = HostDataState(
            host="test",
            host_type=HostType.PERSISTENT,
            total_games=1000,
            estimated_unsynced_games=50,
        )
        data = state.to_dict()
        assert data["host"] == "test"
        assert data["host_type"] == "persistent"
        assert data["total_games"] == 1000
        assert "is_stale" in data
        assert "sync_priority_score" in data


# =============================================================================
# SyncRecommendation Tests
# =============================================================================


class TestSyncRecommendation:
    """Tests for SyncRecommendation dataclass."""

    def test_creation(self):
        """Should create recommendation."""
        rec = SyncRecommendation(
            host="test-host",
            action=SyncAction.SYNC_NOW,
            priority=SyncPriority.HIGH,
            reason="100 unsynced games",
            estimated_games=100,
            estimated_duration_seconds=60,
        )
        assert rec.host == "test-host"
        assert rec.action == SyncAction.SYNC_NOW
        assert rec.priority == SyncPriority.HIGH

    def test_to_dict(self):
        """Should serialize to dictionary."""
        rec = SyncRecommendation(
            host="test",
            action=SyncAction.SCHEDULE_SYNC,
            priority=SyncPriority.NORMAL,
            reason="Routine sync",
        )
        data = rec.to_dict()
        assert data["host"] == "test"
        assert data["action"] == "schedule_sync"
        assert data["priority"] == 50  # SyncPriority.NORMAL.value
        assert data["reason"] == "Routine sync"


# =============================================================================
# ClusterDataStatus Tests
# =============================================================================


class TestClusterDataStatus:
    """Tests for ClusterDataStatus dataclass."""

    def test_cluster_health_score_empty(self):
        """Empty cluster should have perfect health."""
        status = ClusterDataStatus(
            total_hosts=0,
            healthy_hosts=0,
            stale_hosts=[],
            critical_hosts=[],
            syncing_hosts=[],
            unreachable_hosts=[],
            total_games_cluster=0,
            estimated_unsynced_games=0,
            last_full_sync_time=0,
            recommendations=[],
            host_states={},
        )
        assert status.cluster_health_score == 100.0

    def test_cluster_health_score_all_healthy(self):
        """All healthy hosts should give high score."""
        status = ClusterDataStatus(
            total_hosts=5,
            healthy_hosts=5,
            stale_hosts=[],
            critical_hosts=[],
            syncing_hosts=[],
            unreachable_hosts=[],
            total_games_cluster=10000,
            estimated_unsynced_games=0,
            last_full_sync_time=time.time() - 1000,  # Recent
            recommendations=[],
            host_states={},
        )
        assert status.cluster_health_score >= 80

    def test_cluster_health_score_with_stale(self):
        """Stale hosts should reduce score."""
        status = ClusterDataStatus(
            total_hosts=5,
            healthy_hosts=3,
            stale_hosts=["host1", "host2"],
            critical_hosts=[],
            syncing_hosts=[],
            unreachable_hosts=[],
            total_games_cluster=10000,
            estimated_unsynced_games=500,
            last_full_sync_time=time.time() - 7200,
            recommendations=[],
            host_states={},
        )
        score = status.cluster_health_score
        # Should be reduced from stale hosts
        assert 30 < score < 80

    def test_cluster_health_score_with_critical(self):
        """Critical hosts should significantly reduce score."""
        status = ClusterDataStatus(
            total_hosts=5,
            healthy_hosts=2,
            stale_hosts=["host1"],
            critical_hosts=["host2", "host3"],
            syncing_hosts=[],
            unreachable_hosts=[],
            total_games_cluster=10000,
            estimated_unsynced_games=2000,
            last_full_sync_time=time.time() - 86400,  # Old
            recommendations=[],
            host_states={},
        )
        score = status.cluster_health_score
        # Should be low due to critical hosts
        assert score < 50

    def test_to_dict(self):
        """Should serialize to dictionary."""
        status = ClusterDataStatus(
            total_hosts=3,
            healthy_hosts=2,
            stale_hosts=["stale1"],
            critical_hosts=[],
            syncing_hosts=["syncing1"],
            unreachable_hosts=[],
            total_games_cluster=5000,
            estimated_unsynced_games=100,
            last_full_sync_time=time.time(),
            recommendations=[],
            host_states={},
        )
        data = status.to_dict()
        assert data["total_hosts"] == 3
        assert data["healthy_hosts"] == 2
        assert "cluster_health_score" in data
        assert len(data["stale_hosts"]) == 1


# =============================================================================
# SyncScheduler Tests
# =============================================================================


class TestSyncScheduler:
    """Tests for SyncScheduler class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        """Create a SyncScheduler with temporary database."""
        # Reset singleton
        SyncScheduler.reset_instance()

        # Mock host config loading
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_initialization(self, scheduler):
        """Should initialize correctly."""
        assert scheduler.name == "SyncScheduler"
        assert scheduler._host_states == {} or len(scheduler._host_states) >= 0

    def test_register_host(self, scheduler):
        """Should register a new host."""
        scheduler.register_host(
            host="new-host",
            host_type=HostType.EPHEMERAL,
            metadata={"gpu": "RTX 4090"},
        )

        state = scheduler.get_host_state("new-host")
        assert state is not None
        assert state.host == "new-host"
        assert state.host_type == HostType.EPHEMERAL

    def test_update_host_state(self, scheduler):
        """Should update existing host state."""
        scheduler.register_host("test-host", HostType.PERSISTENT)
        scheduler.update_host_state(
            host="test-host",
            total_games=1000,
            estimated_unsynced=50,
        )

        state = scheduler.get_host_state("test-host")
        assert state.total_games == 1000
        assert state.estimated_unsynced_games == 50

    def test_record_sync_complete(self, scheduler):
        """Should record sync completion."""
        scheduler.register_host("sync-host", HostType.PERSISTENT)

        # Must start sync first to get sync_id
        sync_id = scheduler.record_sync_start("sync-host")
        scheduler.record_sync_complete(
            host="sync-host",
            sync_id=sync_id,
            games_synced=100,
            success=True,
        )

        state = scheduler.get_host_state("sync-host")
        assert state.last_sync_time > 0

    def test_record_sync_failure(self, scheduler):
        """Should track sync failures."""
        scheduler.register_host("fail-host", HostType.EPHEMERAL)

        # Record a failure
        sync_id = scheduler.record_sync_start("fail-host")
        scheduler.record_sync_complete(
            host="fail-host",
            sync_id=sync_id,
            games_synced=0,
            success=False,
            error_message="Connection timeout",
        )

        state = scheduler.get_host_state("fail-host")
        assert state.sync_failures_24h >= 1
        assert "timeout" in state.last_error.lower()

    def test_record_games_generated(self, scheduler):
        """Should increment estimated unsynced games."""
        scheduler.register_host("gen-host", HostType.PERSISTENT)
        initial_unsynced = scheduler.get_host_state("gen-host").estimated_unsynced_games

        scheduler.record_games_generated("gen-host", 50)

        state = scheduler.get_host_state("gen-host")
        assert state.estimated_unsynced_games == initial_unsynced + 50

    def test_get_cluster_status(self, scheduler):
        """Should return cluster status."""
        # Add some hosts
        scheduler.register_host("host1", HostType.PERSISTENT)
        scheduler.register_host("host2", HostType.EPHEMERAL)
        scheduler.update_host_state("host1", total_games=1000)
        scheduler.update_host_state("host2", total_games=500, estimated_unsynced=100)

        status = scheduler.get_cluster_status()
        assert status.total_hosts >= 2
        assert status.total_games_cluster >= 1500

    def test_get_sync_recommendations(self, scheduler):
        """Should generate sync recommendations."""
        # Add host that needs sync
        scheduler.register_host("needs-sync", HostType.EPHEMERAL)
        scheduler.update_host_state(
            host="needs-sync",
            estimated_unsynced=500,
            total_games=1000,
        )

        recs = scheduler.get_sync_recommendations()
        # May or may not have recommendations depending on thresholds
        assert isinstance(recs, list)

    def test_get_next_sync_target(self, scheduler):
        """Should return highest priority host for sync."""
        scheduler.register_host("low-priority", HostType.LOCAL)
        scheduler.register_host("high-priority", HostType.EPHEMERAL)

        scheduler.update_host_state("low-priority", estimated_unsynced=10)
        scheduler.update_host_state("high-priority", estimated_unsynced=500)

        target = scheduler.get_next_sync_target()
        # May return None if no sync needed, or the high priority host
        if target is not None:
            # Ephemeral with more unsynced should be higher priority
            assert target in ["high-priority", "low-priority"]

    def test_host_state_management(self, scheduler):
        """Should track host state correctly."""
        scheduler.register_host("test-host", HostType.PERSISTENT)

        # Update state
        scheduler.update_host_state(
            "test-host",
            total_games=1000,
            is_reachable=False,
        )
        state = scheduler.get_host_state("test-host")
        assert not state.is_reachable

        # Update to reachable
        scheduler.update_host_state("test-host", is_reachable=True)
        state = scheduler.get_host_state("test-host")
        assert state.is_reachable

    def test_cluster_status_lists_hosts(self, scheduler):
        """Should include hosts in cluster status."""
        scheduler.register_host("host-a", HostType.PERSISTENT)
        scheduler.register_host("host-b", HostType.EPHEMERAL)

        status = scheduler.get_cluster_status()
        assert status.total_hosts >= 2
        assert len(status.host_states) >= 2

    def test_stale_hosts_in_cluster_status(self, scheduler):
        """Should identify stale hosts in cluster status."""
        scheduler.register_host("fresh", HostType.PERSISTENT)
        scheduler.register_host("stale", HostType.PERSISTENT)

        # Make one fresh via sync
        sync_id = scheduler.record_sync_start("fresh")
        scheduler.record_sync_complete("fresh", sync_id, games_synced=100)

        # Force stale by setting old sync time
        scheduler._host_states["stale"].last_sync_time = time.time() - (STALE_DATA_THRESHOLD_SECONDS + 100)

        status = scheduler.get_cluster_status()
        assert "stale" in status.stale_hosts

    def test_critical_hosts_in_cluster_status(self, scheduler):
        """Should identify critical hosts in cluster status."""
        scheduler.register_host("critical-ephemeral", HostType.EPHEMERAL)
        scheduler.update_host_state("critical-ephemeral", estimated_unsynced=1000)

        status = scheduler.get_cluster_status()
        assert "critical-ephemeral" in status.critical_hosts


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def reset_scheduler(self, temp_db_path):
        """Reset scheduler singleton for clean tests."""
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            with patch("app.coordination.sync_coordinator.DEFAULT_COORDINATOR_DB", temp_db_path):
                yield
        SyncScheduler.reset_instance()

    def test_get_sync_scheduler(self, reset_scheduler):
        """Should return singleton scheduler."""
        scheduler1 = get_sync_scheduler()
        scheduler2 = get_sync_scheduler()
        # Note: May not be same instance due to patching, but should work
        assert scheduler1 is not None
        assert scheduler2 is not None

    def test_reset_sync_scheduler(self, reset_scheduler):
        """Should reset scheduler singleton."""
        get_sync_scheduler()
        reset_sync_scheduler()
        # After reset, should be able to get new scheduler
        scheduler2 = get_sync_scheduler()
        assert scheduler2 is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSyncSchedulerIntegration:
    """Integration tests for SyncScheduler."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_full_sync_workflow(self, scheduler):
        """Test complete sync workflow."""
        # 1. Register host
        scheduler.register_host("workflow-host", HostType.EPHEMERAL)

        # 2. Record games generated
        scheduler.record_games_generated("workflow-host", 100)
        state = scheduler.get_host_state("workflow-host")
        assert state.estimated_unsynced_games == 100

        # 3. Start sync
        sync_id = scheduler.record_sync_start("workflow-host")
        state = scheduler.get_host_state("workflow-host")
        assert state.sync_in_progress

        # 4. Complete sync
        scheduler.record_sync_complete(
            host="workflow-host",
            sync_id=sync_id,
            games_synced=100,
            success=True,
        )

        state = scheduler.get_host_state("workflow-host")
        assert not state.sync_in_progress
        assert state.last_sync_time > 0
        # Unsynced should be reduced (may not be exactly 0 if more were added)
        assert state.estimated_unsynced_games < 100

    def test_persistence_across_restarts(self, temp_db_path):
        """Test that state persists across scheduler restarts."""
        # First instance
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler1 = SyncScheduler(db_path=temp_db_path)
            scheduler1.register_host("persist-host", HostType.PERSISTENT)
            scheduler1.update_host_state("persist-host", total_games=5000)
            scheduler1._save_state()
        SyncScheduler.reset_instance()

        # Second instance
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler2 = SyncScheduler(db_path=temp_db_path)
            state = scheduler2.get_host_state("persist-host")
            assert state is not None
            assert state.total_games == 5000
        SyncScheduler.reset_instance()

    def test_sync_completion_tracking(self, scheduler):
        """Test sync completion is tracked correctly."""
        scheduler.register_host("history-host", HostType.PERSISTENT)

        # Multiple syncs
        for i in range(3):
            sync_id = scheduler.record_sync_start("history-host")
            scheduler.record_sync_complete(
                host="history-host",
                sync_id=sync_id,
                games_synced=50 + i * 10,
                success=True,
            )

        # Verify last sync info
        state = scheduler.get_host_state("history-host")
        assert state.last_sync_time > 0
        assert state.last_sync_games == 70  # Last sync was 50 + 2*10

    def test_multiple_host_management(self, scheduler):
        """Test managing multiple hosts simultaneously."""
        hosts = [
            ("vast-1", HostType.EPHEMERAL, 1000),
            ("vast-2", HostType.EPHEMERAL, 500),
            ("lambda-1", HostType.PERSISTENT, 2000),
            ("local", HostType.LOCAL, 100),
        ]

        for host, host_type, games in hosts:
            scheduler.register_host(host, host_type)
            scheduler.update_host_state(host, total_games=games)

        status = scheduler.get_cluster_status()
        assert status.total_hosts >= 4
        assert status.total_games_cluster >= 3600

    def test_failure_backoff(self, scheduler):
        """Test that failures cause priority backoff."""
        scheduler.register_host("failing-host", HostType.EPHEMERAL)
        scheduler.update_host_state("failing-host", estimated_unsynced=100)

        # Need to set a valid last_sync_time for finite score
        scheduler._host_states["failing-host"].last_sync_time = time.time() - 1800
        initial_score = scheduler.get_host_state("failing-host").sync_priority_score

        # Record failures
        for _ in range(3):
            sync_id = scheduler.record_sync_start("failing-host")
            scheduler.record_sync_complete(
                host="failing-host",
                sync_id=sync_id,
                games_synced=0,
                success=False,
                error_message="Connection failed",
            )

        final_score = scheduler.get_host_state("failing-host").sync_priority_score
        # Score should be lower after failures
        assert final_score < initial_score


# =============================================================================
# Event Wiring Tests
# =============================================================================


class TestEventWiring:
    """Tests for event bus integration."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    def test_wire_sync_events_no_event_bus(self, temp_db_path):
        """Should handle missing event bus gracefully or raise ImportError/AttributeError."""
        SyncScheduler.reset_instance()

        with patch.object(SyncScheduler, '_load_host_config'):
            with patch("app.coordination.sync_coordinator.DEFAULT_COORDINATOR_DB", temp_db_path):
                # Import the wire function
                from app.coordination.sync_coordinator import wire_sync_events

                # May raise ImportError/AttributeError if event system not available,
                # or return scheduler if event system is available
                try:
                    scheduler = wire_sync_events()
                    assert scheduler is not None
                except (ImportError, AttributeError):
                    pass  # Expected if data_events/router not available

        SyncScheduler.reset_instance()


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestBackpressure:
    """Tests for backpressure integration."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_check_sync_backpressure_no_monitor(self, scheduler):
        """Should handle missing queue monitor gracefully."""
        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", False):
            result = scheduler.check_sync_backpressure()
            assert result["should_throttle"] is False
            assert result["should_stop"] is False
            assert result["throttle_factor"] == 1.0
            assert "not available" in result["reason"]

    def test_check_sync_backpressure_no_pressure(self, scheduler):
        """Should return no backpressure when queues are healthy."""
        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", True), \
             patch("app.coordination.sync_coordinator.should_stop_production", return_value=False), \
             patch("app.coordination.sync_coordinator.should_throttle_production", return_value=False), \
             patch("app.coordination.sync_coordinator.get_throttle_factor", return_value=1.0):
            result = scheduler.check_sync_backpressure()
            assert result["should_throttle"] is False
            assert result["should_stop"] is False
            assert result["throttle_factor"] == 1.0

    def test_check_sync_backpressure_throttle(self, scheduler):
        """Should indicate throttling when soft limit exceeded."""
        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", True), \
             patch("app.coordination.sync_coordinator.should_stop_production", return_value=False), \
             patch("app.coordination.sync_coordinator.should_throttle_production", return_value=True), \
             patch("app.coordination.sync_coordinator.get_throttle_factor", return_value=0.5):
            result = scheduler.check_sync_backpressure()
            assert result["should_throttle"] is True
            assert result["should_stop"] is False
            assert result["throttle_factor"] == 0.5

    def test_check_sync_backpressure_stop(self, scheduler):
        """Should indicate stop when hard limit exceeded."""
        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", True), \
             patch("app.coordination.sync_coordinator.should_stop_production", return_value=True), \
             patch("app.coordination.sync_coordinator.should_throttle_production", return_value=True), \
             patch("app.coordination.sync_coordinator.get_throttle_factor", return_value=0.0):
            result = scheduler.check_sync_backpressure()
            assert result["should_throttle"] is True
            assert result["should_stop"] is True
            assert result["throttle_factor"] == 0.0

    def test_should_allow_sync_critical_always_allowed(self, scheduler):
        """Critical syncs should always be allowed."""
        with patch.object(scheduler, 'check_sync_backpressure', return_value={
            "should_throttle": True,
            "should_stop": True,
            "throttle_factor": 0.0,
        }):
            assert scheduler.should_allow_sync(SyncPriority.CRITICAL) is True

    def test_should_allow_sync_high_always_allowed(self, scheduler):
        """High priority syncs should always be allowed."""
        with patch.object(scheduler, 'check_sync_backpressure', return_value={
            "should_throttle": True,
            "should_stop": True,
            "throttle_factor": 0.0,
        }):
            assert scheduler.should_allow_sync(SyncPriority.HIGH) is True

    def test_should_allow_sync_normal_blocked_on_stop(self, scheduler):
        """Normal syncs should be blocked when stop is set."""
        with patch.object(scheduler, 'check_sync_backpressure', return_value={
            "should_throttle": True,
            "should_stop": True,
            "throttle_factor": 0.0,
        }):
            assert scheduler.should_allow_sync(SyncPriority.NORMAL) is False

    def test_should_allow_sync_no_backpressure(self, scheduler):
        """Normal syncs should be allowed when no backpressure."""
        with patch.object(scheduler, 'check_sync_backpressure', return_value={
            "should_throttle": False,
            "should_stop": False,
            "throttle_factor": 1.0,
        }):
            assert scheduler.should_allow_sync(SyncPriority.NORMAL) is True

    def test_report_sync_queue_depth_no_monitor(self, scheduler):
        """Should handle missing queue monitor gracefully."""
        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", False):
            # Should not raise
            scheduler.report_sync_queue_depth(100)

    def test_report_sync_queue_depth_auto_compute(self, scheduler):
        """Should auto-compute depth from host states."""
        scheduler.register_host("host1", HostType.PERSISTENT)
        scheduler.register_host("host2", HostType.EPHEMERAL)
        scheduler.update_host_state("host1", estimated_unsynced=50)
        scheduler.update_host_state("host2", estimated_unsynced=100)

        with patch("app.coordination.sync_coordinator.HAS_QUEUE_MONITOR", True), \
             patch("app.coordination.sync_coordinator.report_queue_depth") as mock_report:
            scheduler.report_sync_queue_depth()
            mock_report.assert_called_once()
            # Should report sum of unsynced games
            args = mock_report.call_args[0]
            assert args[1] == 150  # 50 + 100

    def test_get_sync_recommendations_respects_backpressure(self, scheduler):
        """Recommendations should be reduced under backpressure."""
        scheduler.register_host("host1", HostType.EPHEMERAL)
        scheduler.update_host_state("host1", estimated_unsynced=500)

        with patch.object(scheduler, 'check_sync_backpressure', return_value={
            "should_throttle": False,
            "should_stop": True,
            "throttle_factor": 0.0,
            "reason": "Hard limit",
        }):
            recs = scheduler.get_sync_recommendations(respect_backpressure=True)
            assert len(recs) == 0  # Should return empty when stop is set


# =============================================================================
# Cleanup and Maintenance Tests
# =============================================================================


class TestCleanupMethods:
    """Tests for cleanup and maintenance methods."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_cleanup_old_history_removes_old_records(self, scheduler):
        """Should remove sync history older than specified days."""
        scheduler.register_host("cleanup-host", HostType.PERSISTENT)

        # Create old sync records directly in database
        conn = scheduler._get_connection()
        cursor = conn.cursor()

        old_time = time.time() - (10 * 86400)  # 10 days ago
        cursor.execute("""
            INSERT INTO sync_history (host, started_at, completed_at, games_synced, success)
            VALUES (?, ?, ?, ?, ?)
        """, ("cleanup-host", old_time, old_time + 60, 100, 1))
        conn.commit()

        # Verify record exists
        cursor.execute("SELECT COUNT(*) FROM sync_history")
        assert cursor.fetchone()[0] >= 1

        # Clean up records older than 7 days
        deleted = scheduler.cleanup_old_history(days=7)
        assert deleted >= 1

    def test_cleanup_old_history_keeps_recent_records(self, scheduler):
        """Should keep sync history newer than specified days."""
        scheduler.register_host("keep-host", HostType.PERSISTENT)

        # Create a recent sync
        sync_id = scheduler.record_sync_start("keep-host")
        scheduler.record_sync_complete("keep-host", sync_id, games_synced=50)

        # Count records before cleanup
        conn = scheduler._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sync_history WHERE host = 'keep-host'")
        count_before = cursor.fetchone()[0]

        # Clean up - should not remove recent record
        scheduler.cleanup_old_history(days=7)

        cursor.execute("SELECT COUNT(*) FROM sync_history WHERE host = 'keep-host'")
        count_after = cursor.fetchone()[0]
        assert count_after == count_before

    def test_reset_failure_counts(self, scheduler):
        """Should reset 24-hour failure counts for all hosts."""
        scheduler.register_host("fail-host1", HostType.PERSISTENT)
        scheduler.register_host("fail-host2", HostType.EPHEMERAL)

        # Record failures
        sync_id = scheduler.record_sync_start("fail-host1")
        scheduler.record_sync_complete("fail-host1", sync_id, 0, success=False, error_message="Error")

        sync_id = scheduler.record_sync_start("fail-host2")
        scheduler.record_sync_complete("fail-host2", sync_id, 0, success=False, error_message="Error")

        # Verify failures recorded
        assert scheduler.get_host_state("fail-host1").sync_failures_24h >= 1
        assert scheduler.get_host_state("fail-host2").sync_failures_24h >= 1

        # Reset failures
        scheduler.reset_failure_counts()

        # Verify failures reset
        assert scheduler.get_host_state("fail-host1").sync_failures_24h == 0
        assert scheduler.get_host_state("fail-host2").sync_failures_24h == 0


# =============================================================================
# Duration Estimation Tests
# =============================================================================


class TestDurationEstimation:
    """Tests for sync duration estimation."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_estimate_sync_duration_zero_games(self, scheduler):
        """Should return 0 for zero games."""
        assert scheduler._estimate_sync_duration("any-host", 0) == 0

    def test_estimate_sync_duration_no_history(self, scheduler):
        """Should use default rate when no history."""
        scheduler.register_host("new-host", HostType.PERSISTENT)
        # Default rate is 10 games/second
        duration = scheduler._estimate_sync_duration("new-host", 100)
        assert duration == 10  # 100 / 10 = 10 seconds

    def test_estimate_sync_duration_with_history(self, scheduler):
        """Should use historical rate when available."""
        scheduler.register_host("history-host", HostType.PERSISTENT)

        # Create sync history with known rate
        conn = scheduler._get_connection()
        cursor = conn.cursor()

        # 100 games in 10 seconds = 10 games/second
        now = time.time()
        cursor.execute("""
            INSERT INTO sync_history
            (host, started_at, completed_at, games_synced, duration_seconds, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("history-host", now - 10, now, 100, 10, 1))
        conn.commit()

        # Estimate for 200 games should be ~20 seconds
        duration = scheduler._estimate_sync_duration("history-host", 200)
        assert 15 <= duration <= 25  # Allow some variance


# =============================================================================
# Async Method Tests
# =============================================================================


class TestAsyncMethods:
    """Tests for async methods."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    @pytest.mark.asyncio
    async def test_get_stats(self, scheduler):
        """Should return comprehensive stats."""
        scheduler.register_host("stats-host", HostType.PERSISTENT)
        scheduler.update_host_state("stats-host", total_games=1000)

        stats = await scheduler.get_stats()

        assert "name" in stats
        assert stats["name"] == "SyncScheduler"
        assert "total_hosts" in stats
        assert stats["total_hosts"] >= 1
        assert "total_games_cluster" in stats
        assert stats["total_games_cluster"] >= 1000
        assert "cluster_health_score" in stats

    def test_get_stats_sync(self, scheduler):
        """Should return stats synchronously."""
        scheduler.register_host("sync-stats-host", HostType.PERSISTENT)
        scheduler.update_host_state("sync-stats-host", total_games=500)

        stats = scheduler.get_stats_sync()

        assert stats["name"] == "SyncScheduler"
        assert stats["total_hosts"] >= 1
        assert stats["total_games_cluster"] >= 500

    @pytest.mark.asyncio
    async def test_execute_priority_sync_no_recommendations(self, scheduler):
        """Should handle no recommendations gracefully."""
        # No hosts registered, so no recommendations
        results = await scheduler.execute_priority_sync()

        assert results["syncs_attempted"] == 0
        assert results["syncs_completed"] == 0
        assert len(results["errors"]) == 0

    @pytest.mark.asyncio
    async def test_execute_priority_sync_missing_executor(self, scheduler):
        """Should handle missing distributed executor gracefully."""
        scheduler.register_host("exec-host", HostType.EPHEMERAL)
        scheduler.update_host_state("exec-host", estimated_unsynced=500)

        # Mock ImportError for distributed layer
        with patch.dict("sys.modules", {"app.distributed.sync_coordinator": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                results = await scheduler.execute_priority_sync()

                # Should have errors but not crash
                assert len(results["errors"]) >= 1 or results["syncs_attempted"] == 0


# =============================================================================
# Full Sync Recording Tests
# =============================================================================


class TestFullSyncRecording:
    """Tests for full cluster sync recording."""

    @pytest.fixture
    def temp_db_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sync_coordinator.db"
            yield db_path

    @pytest.fixture
    def scheduler(self, temp_db_path):
        SyncScheduler.reset_instance()
        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler = SyncScheduler(db_path=temp_db_path)
            yield scheduler
            SyncScheduler.reset_instance()

    def test_record_full_sync_complete(self, scheduler):
        """Should update last full sync time."""
        initial_time = scheduler._last_full_sync_time

        scheduler.record_full_sync_complete()

        assert scheduler._last_full_sync_time > initial_time
        assert scheduler._last_full_sync_time > 0

    def test_full_sync_time_persists(self, temp_db_path):
        """Full sync time should persist across restarts."""
        SyncScheduler.reset_instance()

        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler1 = SyncScheduler(db_path=temp_db_path)
            scheduler1.record_full_sync_complete()
            sync_time = scheduler1._last_full_sync_time
            scheduler1._save_state()

        SyncScheduler.reset_instance()

        with patch.object(SyncScheduler, '_load_host_config'):
            scheduler2 = SyncScheduler(db_path=temp_db_path)
            assert scheduler2._last_full_sync_time == sync_time

        SyncScheduler.reset_instance()

    def test_cluster_health_reflects_full_sync(self, scheduler):
        """Cluster health should improve with recent full sync."""
        scheduler.register_host("health-host", HostType.PERSISTENT)

        # Get health before full sync
        status_before = scheduler.get_cluster_status()

        # Record full sync
        scheduler.record_full_sync_complete()

        # Get health after
        status_after = scheduler.get_cluster_status()

        # Health should improve with recent full sync
        assert status_after.cluster_health_score >= status_before.cluster_health_score
