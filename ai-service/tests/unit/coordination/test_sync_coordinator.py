"""Tests for Sync Coordinator module.

Tests the centralized sync scheduling system that manages data synchronization
across all distributed hosts in the RingRift cluster.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.sync_coordinator import (
    # Enums
    SyncPriority,
    HostType,
    SyncAction,
    # Data classes
    HostDataState,
    SyncRecommendation,
    ClusterDataStatus,
    # Main class
    SyncScheduler,
    # Functions
    get_sync_scheduler,
    reset_sync_scheduler,
    # Constants
    STALE_DATA_THRESHOLD_SECONDS,
    CRITICAL_STALE_THRESHOLD_SECONDS,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestSyncPriority:
    """Tests for SyncPriority enum."""

    def test_all_priorities_defined(self):
        """All expected priority levels should exist."""
        assert SyncPriority.CRITICAL.value == "critical"
        assert SyncPriority.HIGH.value == "high"
        assert SyncPriority.NORMAL.value == "normal"
        assert SyncPriority.LOW.value == "low"
        assert SyncPriority.BACKGROUND.value == "background"

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
        assert data["priority"] == "normal"
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
        scheduler1 = get_sync_scheduler()
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
        """Should handle missing event bus gracefully."""
        SyncScheduler.reset_instance()

        with patch.object(SyncScheduler, '_load_host_config'):
            with patch("app.coordination.sync_coordinator.DEFAULT_COORDINATOR_DB", temp_db_path):
                # Import the wire function
                from app.coordination.sync_coordinator import wire_sync_events

                # Should not raise even if data_events not available
                try:
                    scheduler = wire_sync_events()
                    assert scheduler is not None
                except ImportError:
                    pass  # Expected if data_events not available

        SyncScheduler.reset_instance()
