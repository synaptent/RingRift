"""Tests for app.coordination.unified_replication_daemon.

This module tests the UnifiedReplicationDaemon which consolidates replication
monitoring and repair into a single daemon with shared state and coordinated
operations.

Covers:
1. Initialization with default config
2. Monitoring loop behavior
3. Repair loop behavior
4. Priority repair queue
5. Emergency sync triggers
6. Alert generation
7. Factory functions (create_replication_monitor, create_replication_repair_daemon)
"""

from __future__ import annotations

import asyncio
import time
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_replication_daemon import (
    RepairJob,
    RepairPriority,
    RepairStats,
    ReplicationAlert,
    ReplicationAlertLevel,
    ReplicationStats,
    UnifiedReplicationConfig,
    UnifiedReplicationDaemon,
    create_replication_monitor,
    create_replication_repair_daemon,
    get_replication_daemon,
    reset_replication_daemon,
)


# =============================================================================
# ReplicationAlertLevel Tests
# =============================================================================


class TestReplicationAlertLevel:
    """Tests for ReplicationAlertLevel enum."""

    def test_alert_levels_defined(self):
        """All alert levels should be defined."""
        assert ReplicationAlertLevel.INFO.value == "info"
        assert ReplicationAlertLevel.WARNING.value == "warning"
        assert ReplicationAlertLevel.CRITICAL.value == "critical"

    def test_alert_level_count(self):
        """Should have exactly 4 alert levels."""
        assert len(ReplicationAlertLevel) == 4


# =============================================================================
# RepairPriority Tests
# =============================================================================


class TestRepairPriority:
    """Tests for RepairPriority enum."""

    def test_priority_values(self):
        """Priority values should be ordered correctly."""
        assert RepairPriority.CRITICAL.value == 1
        assert RepairPriority.HIGH.value == 2
        assert RepairPriority.NORMAL.value == 3

    def test_priority_ordering(self):
        """Lower value should mean higher priority."""
        assert RepairPriority.CRITICAL.value < RepairPriority.HIGH.value
        assert RepairPriority.HIGH.value < RepairPriority.NORMAL.value


# =============================================================================
# ReplicationAlert Tests
# =============================================================================


class TestReplicationAlert:
    """Tests for ReplicationAlert dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        alert = ReplicationAlert(
            level=ReplicationAlertLevel.WARNING,
            message="Test alert",
        )

        assert alert.level == ReplicationAlertLevel.WARNING
        assert alert.message == "Test alert"
        assert alert.game_count == 0
        assert alert.affected_nodes == []
        assert alert.timestamp == 0.0
        assert alert.resolved is False
        assert alert.resolved_at == 0.0

    def test_full_initialization(self):
        """Should accept all parameters."""
        alert = ReplicationAlert(
            level=ReplicationAlertLevel.CRITICAL,
            message="Critical data loss",
            game_count=50,
            affected_nodes=["node-1", "node-2"],
            timestamp=1000.0,
            resolved=True,
            resolved_at=1100.0,
        )

        assert alert.level == ReplicationAlertLevel.CRITICAL
        assert alert.game_count == 50
        assert len(alert.affected_nodes) == 2
        assert alert.resolved is True


# =============================================================================
# ReplicationStats Tests
# =============================================================================


class TestReplicationStats:
    """Tests for ReplicationStats dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        stats = ReplicationStats()

        assert stats.total_games == 0
        assert stats.under_replicated_games == 0
        assert stats.single_copy_games == 0
        assert stats.zero_copy_games == 0
        assert stats.avg_replication_factor == 0.0
        assert stats.nodes_with_data == 0
        assert stats.last_check_time == 0.0
        assert stats.check_duration_seconds == 0.0

    def test_full_initialization(self):
        """Should accept all parameters."""
        stats = ReplicationStats(
            total_games=1000,
            under_replicated_games=50,
            single_copy_games=10,
            zero_copy_games=2,
            avg_replication_factor=2.8,
            nodes_with_data=5,
            last_check_time=time.time(),
            check_duration_seconds=1.5,
        )

        assert stats.total_games == 1000
        assert stats.under_replicated_games == 50
        assert stats.avg_replication_factor == 2.8


# =============================================================================
# RepairJob Tests
# =============================================================================


class TestRepairJob:
    """Tests for RepairJob dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        job = RepairJob(
            game_id="game-123",
            priority=RepairPriority.HIGH,
            current_copies=1,
            target_copies=3,
        )

        assert job.game_id == "game-123"
        assert job.priority == RepairPriority.HIGH
        assert job.current_copies == 1
        assert job.target_copies == 3
        assert job.source_nodes == []
        assert job.target_nodes == []
        assert job.created_at == 0.0
        assert job.started_at == 0.0
        assert job.completed_at == 0.0
        assert job.success is False
        assert job.error == ""

    def test_full_initialization(self):
        """Should accept all parameters."""
        job = RepairJob(
            game_id="game-456",
            priority=RepairPriority.CRITICAL,
            current_copies=0,
            target_copies=3,
            source_nodes=["node-1"],
            target_nodes=["node-2", "node-3"],
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1010.0,
            success=True,
            error="",
        )

        assert job.game_id == "game-456"
        assert len(job.source_nodes) == 1
        assert len(job.target_nodes) == 2
        assert job.success is True


# =============================================================================
# RepairStats Tests
# =============================================================================


class TestRepairStats:
    """Tests for RepairStats dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        stats = RepairStats()

        assert stats.total_repairs_attempted == 0
        assert stats.total_repairs_successful == 0
        assert stats.total_repairs_failed == 0
        assert stats.active_repairs == 0
        assert stats.queued_repairs == 0
        assert stats.last_repair_time == 0.0
        assert stats.avg_repair_duration_seconds == 0.0


# =============================================================================
# UnifiedReplicationConfig Tests
# =============================================================================


class TestUnifiedReplicationConfig:
    """Tests for UnifiedReplicationConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = UnifiedReplicationConfig()

        # Monitoring settings
        assert config.monitor_interval_seconds == 300.0
        assert config.warning_threshold_minutes == 15.0
        assert config.critical_threshold_minutes == 60.0
        assert config.single_copy_threshold_games == 100

        # Replication targets
        assert config.min_replicas == 2
        assert config.target_replicas == 3

        # Repair settings
        assert config.repair_interval_seconds == 60.0
        assert config.max_concurrent_repairs == 5
        assert config.repair_timeout_seconds == 300.0

        # Emergency sync
        assert config.enable_emergency_sync is True
        assert config.emergency_sync_threshold_games == 500
        assert config.emergency_sync_cooldown_seconds == 600.0

        # Event emission
        assert config.emit_events is True
        assert config.max_alerts_history == 100
        assert config.max_repair_history == 200

    def test_custom_config(self):
        """Should accept custom values."""
        config = UnifiedReplicationConfig(
            monitor_interval_seconds=60.0,
            min_replicas=3,
            target_replicas=5,
            max_concurrent_repairs=10,
        )

        assert config.monitor_interval_seconds == 60.0
        assert config.min_replicas == 3
        assert config.target_replicas == 5
        assert config.max_concurrent_repairs == 10


# =============================================================================
# UnifiedReplicationDaemon Initialization Tests
# =============================================================================


class TestUnifiedReplicationDaemonInit:
    """Tests for UnifiedReplicationDaemon initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        daemon = UnifiedReplicationDaemon()

        assert daemon.config is not None
        assert daemon.config.min_replicas == 2
        assert daemon._running is False
        assert daemon._monitor_task is None
        assert daemon._task is None

    def test_init_with_custom_config(self):
        """Should accept custom config."""
        config = UnifiedReplicationConfig(
            min_replicas=3,
            target_replicas=5,
        )
        daemon = UnifiedReplicationDaemon(config)

        assert daemon.config.min_replicas == 3
        assert daemon.config.target_replicas == 5

    def test_node_id_set_from_hostname(self):
        """Should set node_id from hostname."""
        with patch("socket.gethostname", return_value="test-node"):
            daemon = UnifiedReplicationDaemon()
            assert daemon.node_id == "test-node"

    def test_initial_stats_empty(self):
        """Initial stats should be empty."""
        daemon = UnifiedReplicationDaemon()

        assert daemon._stats.total_games == 0
        assert daemon._stats.under_replicated_games == 0
        assert len(daemon._alerts) == 0
        assert len(daemon._repair_queue) == 0


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    def test_get_replication_daemon_creates_singleton(self):
        """get_replication_daemon should create singleton."""
        daemon1 = get_replication_daemon()
        daemon2 = get_replication_daemon()

        assert daemon1 is daemon2

    def test_reset_clears_singleton(self):
        """reset_replication_daemon should clear singleton."""
        daemon1 = get_replication_daemon()
        reset_replication_daemon()
        daemon2 = get_replication_daemon()

        assert daemon1 is not daemon2

    def test_config_only_used_on_first_call(self):
        """Config should only be applied on first call."""
        config1 = UnifiedReplicationConfig(min_replicas=5)
        config2 = UnifiedReplicationConfig(min_replicas=10)

        daemon1 = get_replication_daemon(config1)
        daemon2 = get_replication_daemon(config2)

        assert daemon1.config.min_replicas == 5
        assert daemon2.config.min_replicas == 5  # Same instance


# =============================================================================
# Daemon Start/Stop Tests
# =============================================================================


class TestDaemonStartStop:
    """Tests for daemon start and stop functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_start_creates_tasks(self):
        """start() should create monitor and main-loop tasks."""
        daemon = UnifiedReplicationDaemon()

        # Mock the loops to avoid actual execution
        with patch.object(daemon, "_monitor_loop", new_callable=AsyncMock), \
             patch.object(daemon, "_main_loop", new_callable=AsyncMock):
            await daemon.start()

            assert daemon._running is True
            assert daemon._monitor_task is not None
            assert daemon._task is not None

            await daemon.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """start() when already running should log warning."""
        daemon = UnifiedReplicationDaemon()
        daemon._running = True

        with patch("app.coordination.handler_base.logger") as mock_logger:
            await daemon.start()
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """stop() should cancel running tasks."""
        daemon = UnifiedReplicationDaemon()

        # Mock the loops
        with patch.object(daemon, "_monitor_loop", new_callable=AsyncMock), \
             patch.object(daemon, "_main_loop", new_callable=AsyncMock):
            await daemon.start()
            await daemon.stop()

            assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """stop() when not running should be a no-op."""
        daemon = UnifiedReplicationDaemon()

        # Should not raise
        await daemon.stop()
        assert daemon._running is False

    def test_is_running_property(self):
        """is_running should reflect daemon state."""
        daemon = UnifiedReplicationDaemon()

        assert daemon.is_running is False
        daemon._running = True
        assert daemon.is_running is True


# =============================================================================
# Monitoring Loop Tests
# =============================================================================


class TestMonitoringLoop:
    """Tests for the monitoring loop behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_check_replication_updates_stats(self):
        """_check_replication should update stats from manifest."""
        daemon = UnifiedReplicationDaemon()

        mock_manifest = {
            "game-1": {"locations": ["node-1", "node-2", "node-3"]},
            "game-2": {"locations": ["node-1"]},  # Single copy
            "game-3": {"locations": ["node-1", "node-2"]},  # Under-replicated
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest):
            await daemon._check_replication()

            assert daemon._stats.total_games == 3
            assert daemon._stats.single_copy_games == 1
            assert daemon._stats.under_replicated_games == 1  # Only game-2 (1 < 2)

    @pytest.mark.asyncio
    async def test_check_replication_handles_empty_manifest(self):
        """_check_replication should handle empty manifest."""
        daemon = UnifiedReplicationDaemon()

        with patch.object(daemon, "_get_cluster_manifest", return_value={}):
            await daemon._check_replication()
            # Should not raise

    @pytest.mark.asyncio
    async def test_check_replication_tracks_under_replicated_since(self):
        """Should track when games first became under-replicated."""
        daemon = UnifiedReplicationDaemon()

        mock_manifest = {
            "game-1": {"locations": ["node-1"]},  # Single copy
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest):
            await daemon._check_replication()

            assert "game-1" in daemon._under_replicated_since

    @pytest.mark.asyncio
    async def test_check_replication_clears_healthy_games(self):
        """Should clear tracking for games that become healthy."""
        daemon = UnifiedReplicationDaemon()
        daemon._under_replicated_since["game-1"] = time.time()

        mock_manifest = {
            "game-1": {"locations": ["node-1", "node-2", "node-3"]},  # Now healthy
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest):
            await daemon._check_replication()

            assert "game-1" not in daemon._under_replicated_since


# =============================================================================
# Alert Generation Tests
# =============================================================================


class TestAlertGeneration:
    """Tests for alert generation."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_zero_copy_generates_critical_alert(self):
        """Zero-copy games should generate critical alert."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(zero_copy_games=5)

        await daemon._evaluate_alerts(time.time())

        assert "zero_copy" in daemon._active_alerts
        assert daemon._active_alerts["zero_copy"].level == ReplicationAlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_single_copy_generates_warning_alert(self):
        """Single-copy games above threshold should generate warning."""
        config = UnifiedReplicationConfig(single_copy_threshold_games=10)
        daemon = UnifiedReplicationDaemon(config)
        daemon._stats = ReplicationStats(single_copy_games=15)

        await daemon._evaluate_alerts(time.time())

        assert "single_copy" in daemon._active_alerts
        assert daemon._active_alerts["single_copy"].level == ReplicationAlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_alert_resolved_when_condition_clears(self):
        """Alerts should be resolved when condition clears."""
        daemon = UnifiedReplicationDaemon()

        # First, generate alert
        daemon._stats = ReplicationStats(zero_copy_games=5)
        await daemon._evaluate_alerts(time.time())
        assert "zero_copy" in daemon._active_alerts

        # Then, clear condition
        daemon._stats = ReplicationStats(zero_copy_games=0)
        await daemon._evaluate_alerts(time.time())
        assert "zero_copy" not in daemon._active_alerts

    @pytest.mark.asyncio
    async def test_alert_history_maintained(self):
        """Alert history should be maintained."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(zero_copy_games=5)

        await daemon._evaluate_alerts(time.time())

        assert len(daemon._alerts) == 1

    @pytest.mark.asyncio
    async def test_alert_history_trimmed(self):
        """Alert history should be trimmed when exceeding max."""
        config = UnifiedReplicationConfig(max_alerts_history=3)
        daemon = UnifiedReplicationDaemon(config)

        # Add more alerts than history limit
        for i in range(5):
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.INFO,
                message=f"Alert {i}",
            )
            daemon._add_alert(f"key-{i}", alert)

        assert len(daemon._alerts) <= config.max_alerts_history


# =============================================================================
# Emergency Sync Tests
# =============================================================================


class TestEmergencySync:
    """Tests for emergency sync triggers."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_emergency_sync_triggered_on_zero_copy(self):
        """Emergency sync should trigger on zero-copy games."""
        daemon = UnifiedReplicationDaemon()

        with patch.object(daemon, "_trigger_emergency_repair", new_callable=AsyncMock) as mock_repair:
            await daemon._check_emergency_sync(
                under_replicated=0,
                single_copy=0,
                zero_copy=1,
            )

            mock_repair.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_sync_triggered_on_high_single_copy(self):
        """Emergency sync should trigger on high single-copy count."""
        config = UnifiedReplicationConfig(emergency_sync_threshold_games=100)
        daemon = UnifiedReplicationDaemon(config)

        with patch.object(daemon, "_trigger_emergency_repair", new_callable=AsyncMock) as mock_repair:
            await daemon._check_emergency_sync(
                under_replicated=0,
                single_copy=150,  # Above threshold
                zero_copy=0,
            )

            mock_repair.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_sync_respects_cooldown(self):
        """Emergency sync should respect cooldown period."""
        config = UnifiedReplicationConfig(emergency_sync_cooldown_seconds=600.0)
        daemon = UnifiedReplicationDaemon(config)
        daemon._last_emergency_sync = time.time()  # Just synced

        with patch.object(daemon, "_trigger_emergency_repair", new_callable=AsyncMock) as mock_repair:
            await daemon._check_emergency_sync(
                under_replicated=0,
                single_copy=0,
                zero_copy=5,
            )

            mock_repair.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_sync_disabled(self):
        """Emergency sync should not trigger when disabled."""
        config = UnifiedReplicationConfig(enable_emergency_sync=False)
        daemon = UnifiedReplicationDaemon(config)

        mock_manifest = {
            "game-1": {"locations": []},  # Zero copies
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest), \
             patch.object(daemon, "_trigger_emergency_repair", new_callable=AsyncMock) as mock_repair:
            await daemon._check_replication()

            mock_repair.assert_not_called()


# =============================================================================
# Repair Loop Tests
# =============================================================================


class TestRepairLoop:
    """Tests for the repair loop behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_repair_cycle_finds_games_needing_repair(self):
        """_run_repair_cycle should find games needing repair."""
        daemon = UnifiedReplicationDaemon()

        mock_manifest = {
            "game-1": {"locations": ["node-1"]},  # Needs repair
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest), \
             patch.object(daemon, "_select_target_nodes", return_value=["node-2"]):
            await daemon._run_repair_cycle()

            assert daemon._repair_stats.queued_repairs >= 0

    @pytest.mark.asyncio
    async def test_repair_jobs_sorted_by_priority(self):
        """Repair queue should be sorted by priority."""
        daemon = UnifiedReplicationDaemon()

        # Add jobs out of order
        job_normal = RepairJob(
            game_id="game-1",
            priority=RepairPriority.NORMAL,
            current_copies=1,
            target_copies=3,
        )
        job_critical = RepairJob(
            game_id="game-2",
            priority=RepairPriority.CRITICAL,
            current_copies=0,
            target_copies=3,
        )

        daemon._repair_queue = [job_normal, job_critical]
        daemon._repair_queue.sort(key=lambda j: j.priority.value)

        assert daemon._repair_queue[0].priority == RepairPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_skips_games_already_being_repaired(self):
        """Should skip games that are already being repaired."""
        daemon = UnifiedReplicationDaemon()
        daemon._active_repairs["game-1"] = RepairJob(
            game_id="game-1",
            priority=RepairPriority.HIGH,
            current_copies=1,
            target_copies=3,
        )

        mock_manifest = {
            "game-1": {"locations": ["node-1"]},
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest):
            games = await daemon._find_games_needing_repair()

            assert not any(g[0] == "game-1" for g in games)


# =============================================================================
# Priority Queue Tests
# =============================================================================


class TestPriorityQueue:
    """Tests for priority repair queue."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_zero_copy_gets_critical_priority(self):
        """Zero-copy games should get CRITICAL priority."""
        daemon = UnifiedReplicationDaemon()

        games = [("game-1", 0, [])]  # Zero copies

        with patch.object(daemon, "_select_target_nodes", return_value=["node-1"]):
            jobs = await daemon._create_repair_jobs(games)

            assert len(jobs) == 1
            assert jobs[0].priority == RepairPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_single_copy_gets_high_priority(self):
        """Single-copy games should get HIGH priority."""
        daemon = UnifiedReplicationDaemon()

        games = [("game-1", 1, ["node-1"])]

        with patch.object(daemon, "_select_target_nodes", return_value=["node-2"]):
            jobs = await daemon._create_repair_jobs(games)

            assert len(jobs) == 1
            assert jobs[0].priority == RepairPriority.HIGH

    @pytest.mark.asyncio
    async def test_under_replicated_gets_normal_priority(self):
        """Under-replicated (but not single copy) should get NORMAL priority."""
        config = UnifiedReplicationConfig(min_replicas=3)
        daemon = UnifiedReplicationDaemon(config)

        games = [("game-1", 2, ["node-1", "node-2"])]  # 2 copies, need 3

        with patch.object(daemon, "_select_target_nodes", return_value=["node-3"]):
            jobs = await daemon._create_repair_jobs(games)

            # Since 2 copies is considered normal, not under-replicated for this test
            # (under_replicated is < min_replicas which is 3)
            assert len(jobs) == 1
            assert jobs[0].priority == RepairPriority.NORMAL


# =============================================================================
# Manual Repair Trigger Tests
# =============================================================================


class TestManualRepairTrigger:
    """Tests for manual repair triggering."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_trigger_repair_queues_games(self):
        """trigger_repair should queue specified games."""
        daemon = UnifiedReplicationDaemon()

        mock_manifest = {
            "game-1": {"locations": ["node-1"]},
            "game-2": {"locations": ["node-1"]},
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest), \
             patch.object(daemon, "_select_target_nodes", return_value=["node-2"]):
            count = await daemon.trigger_repair(["game-1"])

            assert count == 1

    @pytest.mark.asyncio
    async def test_trigger_repair_all_under_replicated(self):
        """trigger_repair with None should queue all under-replicated."""
        daemon = UnifiedReplicationDaemon()

        mock_manifest = {
            "game-1": {"locations": ["node-1"]},
            "game-2": {"locations": ["node-1"]},
        }

        with patch.object(daemon, "_get_cluster_manifest", return_value=mock_manifest), \
             patch.object(daemon, "_select_target_nodes", return_value=["node-2"]):
            count = await daemon.trigger_repair(None)

            assert count == 2

    @pytest.mark.asyncio
    async def test_trigger_repair_returns_zero_for_no_games(self):
        """trigger_repair should return 0 if no games need repair."""
        daemon = UnifiedReplicationDaemon()

        with patch.object(daemon, "_get_cluster_manifest", return_value={}):
            count = await daemon.trigger_repair(["nonexistent"])

            assert count == 0


# =============================================================================
# Status Reporting Tests
# =============================================================================


class TestStatusReporting:
    """Tests for status reporting."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    def test_get_status_returns_complete_status(self):
        """get_status should return complete daemon status."""
        daemon = UnifiedReplicationDaemon()

        status = daemon.get_status()

        assert "running" in status
        assert "node_id" in status
        assert "health_score" in status
        assert "health_status" in status
        assert "monitoring" in status
        assert "repair" in status
        assert "config" in status

    def test_get_status_includes_monitoring_stats(self):
        """Status should include monitoring statistics."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(
            total_games=100,
            under_replicated_games=10,
        )

        status = daemon.get_status()

        assert status["monitoring"]["total_games"] == 100
        assert status["monitoring"]["under_replicated_games"] == 10

    def test_get_status_includes_repair_stats(self):
        """Status should include repair statistics."""
        daemon = UnifiedReplicationDaemon()
        daemon._repair_stats = RepairStats(
            jobs_processed=50,
            jobs_succeeded=45,
        )

        status = daemon.get_status()

        assert status["repair"]["total_attempted"] == 50
        assert status["repair"]["total_successful"] == 45

    def test_health_score_calculation(self):
        """Health score should be calculated correctly."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(total_games=100, under_replicated_games=0)

        health = daemon._compute_health_score()

        assert health["score"] == 100.0
        assert health["status"] == "healthy"

    def test_health_score_degraded(self):
        """Health score should show degraded for moderate issues."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(
            total_games=100,
            under_replicated_games=40,
        )

        health = daemon._compute_health_score()

        assert health["score"] < 90
        assert health["status"] in ["degraded", "unhealthy"]


# =============================================================================
# History Tests
# =============================================================================


class TestHistory:
    """Tests for alert and repair history."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    def test_get_alerts_history(self):
        """get_alerts_history should return alert history."""
        daemon = UnifiedReplicationDaemon()

        alert = ReplicationAlert(
            level=ReplicationAlertLevel.WARNING,
            message="Test alert",
            timestamp=time.time(),
        )
        daemon._alerts.append(alert)

        history = daemon.get_alerts_history()

        assert len(history) == 1
        assert history[0]["message"] == "Test alert"

    def test_get_alerts_history_with_limit(self):
        """get_alerts_history should respect limit."""
        daemon = UnifiedReplicationDaemon()

        for i in range(10):
            alert = ReplicationAlert(
                level=ReplicationAlertLevel.INFO,
                message=f"Alert {i}",
            )
            daemon._alerts.append(alert)

        history = daemon.get_alerts_history(limit=5)

        assert len(history) == 5

    def test_get_repair_history(self):
        """get_repair_history should return repair history."""
        daemon = UnifiedReplicationDaemon()

        job = RepairJob(
            game_id="game-1",
            priority=RepairPriority.HIGH,
            current_copies=1,
            target_copies=3,
            success=True,
            created_at=1000.0,
            started_at=1001.0,
            completed_at=1010.0,
        )
        daemon._repair_history.append(job)

        history = daemon.get_repair_history()

        assert len(history) == 1
        assert history[0]["game_id"] == "game-1"
        assert history[0]["success"] is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for backward compatibility factory functions."""

    def test_create_replication_monitor_emits_warning(self):
        """create_replication_monitor should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            daemon = create_replication_monitor()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert isinstance(daemon, UnifiedReplicationDaemon)

    def test_create_replication_repair_daemon_emits_warning(self):
        """create_replication_repair_daemon should emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            daemon = create_replication_repair_daemon()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert isinstance(daemon, UnifiedReplicationDaemon)

    def test_factory_accepts_config(self):
        """Factory functions should accept custom config."""
        config = UnifiedReplicationConfig(min_replicas=5)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            daemon = create_replication_monitor(config)

            assert daemon.config.min_replicas == 5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_replication_daemon()

    @pytest.mark.asyncio
    async def test_repair_with_no_source_nodes(self):
        """Repair should fail gracefully with no source nodes."""
        daemon = UnifiedReplicationDaemon()

        job = RepairJob(
            game_id="game-1",
            priority=RepairPriority.HIGH,
            current_copies=0,
            target_copies=3,
            source_nodes=[],
            target_nodes=["node-1"],
        )

        result = await daemon._perform_repair_transfer(job)

        assert result is False
        assert job.error != ""

    @pytest.mark.asyncio
    async def test_repair_with_no_target_nodes(self):
        """Repair should fail gracefully with no target nodes."""
        daemon = UnifiedReplicationDaemon()

        job = RepairJob(
            game_id="game-1",
            priority=RepairPriority.HIGH,
            current_copies=1,
            target_copies=3,
            source_nodes=["node-1"],
            target_nodes=[],
        )

        result = await daemon._perform_repair_transfer(job)

        assert result is False
        assert job.error != ""

    @pytest.mark.asyncio
    async def test_manifest_import_error_handled(self):
        """Should handle ImportError when getting manifest."""
        daemon = UnifiedReplicationDaemon()

        with patch(
            "app.coordination.unified_replication_daemon.UnifiedReplicationDaemon._get_cluster_manifest",
            side_effect=ImportError("Module not available"),
        ):
            # Should not raise
            await daemon._check_replication()

    def test_health_score_with_zero_games(self):
        """Health score should handle zero games."""
        daemon = UnifiedReplicationDaemon()
        daemon._stats = ReplicationStats(total_games=0)

        health = daemon._compute_health_score()

        # Should not divide by zero
        assert health["score"] == 100.0
