"""
Tests for JobOrchestrationManager - Job spawning, scaling, and cluster coordination.

Jan 9, 2026: Comprehensive test suite for job_orchestration_manager.py.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.p2p.managers.job_orchestration_manager import (
    DISK_CLEANUP_THRESHOLD,
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
    GPU_IDLE_RESTART_TIMEOUT,
    GPU_IDLE_THRESHOLD,
    LOAD_MAX_FOR_NEW_JOBS,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    RUNAWAY_SELFPLAY_PROCESS_THRESHOLD,
    JobOrchestrationConfig,
    JobOrchestrationManager,
    JobOrchestrationStats,
    create_job_orchestration_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def jobs_lock() -> threading.RLock:
    """Create a test jobs lock."""
    return threading.RLock()


@pytest.fixture
def basic_manager(jobs_lock: threading.RLock) -> JobOrchestrationManager:
    """Create a basic manager with minimal configuration."""
    return JobOrchestrationManager(
        node_id="test-node",
        ringrift_path="/path/to/ringrift",
        get_peers=lambda: {},
        get_local_jobs=lambda: {},
        jobs_lock=jobs_lock,
        is_leader=lambda: False,
        get_self_info=lambda: None,
    )


@pytest.fixture
def leader_manager(jobs_lock: threading.RLock) -> JobOrchestrationManager:
    """Create a manager configured as leader."""
    return JobOrchestrationManager(
        node_id="leader-node",
        ringrift_path="/path/to/ringrift",
        get_peers=lambda: {"peer-1": MagicMock(), "peer-2": MagicMock()},
        get_local_jobs=lambda: {},
        jobs_lock=jobs_lock,
        is_leader=lambda: True,
        get_self_info=lambda: MagicMock(),
    )


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_disk_thresholds(self):
        """Test disk threshold constants."""
        assert DISK_WARNING_THRESHOLD == 80
        assert DISK_CLEANUP_THRESHOLD == 85
        assert DISK_CRITICAL_THRESHOLD == 90
        # Thresholds should be in order
        assert DISK_WARNING_THRESHOLD < DISK_CLEANUP_THRESHOLD < DISK_CRITICAL_THRESHOLD

    def test_memory_thresholds(self):
        """Test memory threshold constants."""
        assert MEMORY_WARNING_THRESHOLD == 75
        assert MEMORY_CRITICAL_THRESHOLD == 85
        assert MEMORY_WARNING_THRESHOLD < MEMORY_CRITICAL_THRESHOLD

    def test_gpu_thresholds(self):
        """Test GPU threshold constants."""
        assert GPU_IDLE_THRESHOLD == 5
        assert GPU_IDLE_RESTART_TIMEOUT == 300

    def test_load_threshold(self):
        """Test load threshold constant."""
        assert LOAD_MAX_FOR_NEW_JOBS == 80

    def test_runaway_threshold(self):
        """Test runaway process threshold."""
        assert RUNAWAY_SELFPLAY_PROCESS_THRESHOLD == 128


# ============================================================================
# JobOrchestrationConfig Tests
# ============================================================================


class TestJobOrchestrationConfig:
    """Tests for JobOrchestrationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JobOrchestrationConfig()
        assert config.max_local_jobs == 4
        assert config.job_timeout_seconds == 3600.0
        assert config.rebalance_interval_seconds == 60.0
        assert config.max_spawns_per_minute == 10
        assert config.spawn_cooldown_seconds == 5.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JobOrchestrationConfig(
            max_local_jobs=8,
            job_timeout_seconds=7200.0,
            max_spawns_per_minute=20,
        )
        assert config.max_local_jobs == 8
        assert config.job_timeout_seconds == 7200.0
        assert config.max_spawns_per_minute == 20


# ============================================================================
# JobOrchestrationStats Tests
# ============================================================================


class TestJobOrchestrationStats:
    """Tests for JobOrchestrationStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = JobOrchestrationStats()
        assert stats.jobs_started == 0
        assert stats.jobs_failed == 0
        assert stats.jobs_completed == 0
        assert stats.jobs_scaled_down == 0
        assert stats.cluster_management_runs == 0
        assert stats.work_items_executed == 0
        assert stats.spawn_blocked_count == 0

    def test_custom_values(self):
        """Test stats with custom values."""
        stats = JobOrchestrationStats(
            jobs_started=10,
            jobs_completed=8,
            jobs_failed=2,
        )
        assert stats.jobs_started == 10
        assert stats.jobs_completed == 8
        assert stats.jobs_failed == 2


# ============================================================================
# JobOrchestrationManager Initialization Tests
# ============================================================================


class TestJobOrchestrationManagerInit:
    """Tests for JobOrchestrationManager initialization."""

    def test_minimal_init(self, jobs_lock: threading.RLock):
        """Test initialization with minimal parameters."""
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path/to/ringrift",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
        )
        assert manager.node_id == "test-node"
        assert manager.ringrift_path == "/path/to/ringrift"
        assert manager._running is False

    def test_full_init(self, jobs_lock: threading.RLock):
        """Test initialization with all parameters."""
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path/to/ringrift",
            get_peers=lambda: {"peer-1": MagicMock()},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: True,
            get_self_info=lambda: MagicMock(),
            selfplay_scheduler=MagicMock(),
            job_manager=MagicMock(),
            save_state_fn=MagicMock(),
            config=JobOrchestrationConfig(max_local_jobs=8),
        )
        assert manager._selfplay_scheduler is not None
        assert manager._job_manager is not None
        assert manager._config.max_local_jobs == 8

    def test_custom_config(self, jobs_lock: threading.RLock):
        """Test initialization with custom config."""
        config = JobOrchestrationConfig(
            max_spawns_per_minute=20,
            spawn_cooldown_seconds=2.0,
        )
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            config=config,
        )
        assert manager._config.max_spawns_per_minute == 20
        assert manager._config.spawn_cooldown_seconds == 2.0


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_initialized(self, basic_manager: JobOrchestrationManager):
        """Test health check when not running."""
        health = basic_manager.health_check()
        assert health["healthy"] is True
        assert health["status"] == "initialized"
        assert health["node_id"] == "test-node"

    def test_health_check_running(self, basic_manager: JobOrchestrationManager):
        """Test health check when running."""
        basic_manager.start()
        health = basic_manager.health_check()
        assert health["status"] == "running"
        basic_manager.stop()

    def test_health_check_includes_stats(self, basic_manager: JobOrchestrationManager):
        """Test health check includes statistics."""
        health = basic_manager.health_check()
        assert "stats" in health
        stats = health["stats"]
        assert "jobs_started" in stats
        assert "jobs_failed" in stats
        assert "jobs_completed" in stats


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start(self, basic_manager: JobOrchestrationManager):
        """Test starting the manager."""
        assert basic_manager._running is False
        basic_manager.start()
        assert basic_manager._running is True

    def test_stop(self, basic_manager: JobOrchestrationManager):
        """Test stopping the manager."""
        basic_manager.start()
        assert basic_manager._running is True
        basic_manager.stop()
        assert basic_manager._running is False

    def test_multiple_start_calls(self, basic_manager: JobOrchestrationManager):
        """Test multiple start calls are safe."""
        basic_manager.start()
        basic_manager.start()  # Second call should be safe
        assert basic_manager._running is True

    def test_multiple_stop_calls(self, basic_manager: JobOrchestrationManager):
        """Test multiple stop calls are safe."""
        basic_manager.start()
        basic_manager.stop()
        basic_manager.stop()  # Second call should be safe
        assert basic_manager._running is False


# ============================================================================
# Helper Methods Tests
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_ai_service_path_safe_default(self, basic_manager: JobOrchestrationManager):
        """Test default ai-service path."""
        path = basic_manager._get_ai_service_path_safe()
        assert path == "/path/to/ringrift/ai-service"

    def test_get_ai_service_path_safe_callback(self, jobs_lock: threading.RLock):
        """Test ai-service path with callback."""
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path/to/ringrift",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            get_ai_service_path_fn=lambda: "/custom/ai-service",
        )
        path = manager._get_ai_service_path_safe()
        assert path == "/custom/ai-service"

    def test_get_script_path_safe_default(self, basic_manager: JobOrchestrationManager):
        """Test default script path."""
        path = basic_manager._get_script_path_safe("test_script.py")
        assert path == "/path/to/ringrift/ai-service/scripts/test_script.py"

    def test_get_script_path_safe_callback(self, jobs_lock: threading.RLock):
        """Test script path with callback."""
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path/to/ringrift",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            get_script_path_fn=lambda name: f"/custom/scripts/{name}",
        )
        path = manager._get_script_path_safe("test_script.py")
        assert path == "/custom/scripts/test_script.py"


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for spawn rate limiting."""

    def test_rate_limit_allows_initial_spawn(self, basic_manager: JobOrchestrationManager):
        """Test that initial spawn is allowed."""
        allowed, message = basic_manager._check_spawn_rate_limit()
        assert allowed is True
        assert message == "OK"

    def test_rate_limit_blocks_after_max(self, jobs_lock: threading.RLock):
        """Test that spawn is blocked after max reached."""
        config = JobOrchestrationConfig(max_spawns_per_minute=3)
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            config=config,
        )

        # Record 3 spawns
        for _ in range(3):
            manager._record_spawn_internal()

        allowed, message = manager._check_spawn_rate_limit()
        assert allowed is False
        assert "Rate limit" in message

    def test_rate_limit_clears_old_entries(self, jobs_lock: threading.RLock):
        """Test that old spawn entries are cleared."""
        config = JobOrchestrationConfig(max_spawns_per_minute=2)
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            config=config,
        )

        # Add old spawn time (older than 60 seconds)
        manager._spawn_times = [time.time() - 120]  # 2 minutes ago

        allowed, message = manager._check_spawn_rate_limit()
        assert allowed is True
        assert len(manager._spawn_times) == 0  # Old entry cleared

    def test_record_spawn_internal(self, basic_manager: JobOrchestrationManager):
        """Test spawn recording."""
        assert len(basic_manager._spawn_times) == 0
        basic_manager._record_spawn_internal()
        assert len(basic_manager._spawn_times) == 1

    def test_record_spawn_with_callback(self, jobs_lock: threading.RLock):
        """Test spawn recording with callback."""
        callback_called = []
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
            record_spawn_fn=lambda: callback_called.append(True),
        )
        manager._record_spawn_internal()
        assert len(callback_called) == 1


# ============================================================================
# Statistics Recording Tests
# ============================================================================


class TestStatisticsRecording:
    """Tests for statistics recording methods."""

    def test_record_job_started(self, basic_manager: JobOrchestrationManager):
        """Test recording job started."""
        assert basic_manager._stats.jobs_started == 0
        basic_manager.record_job_started("selfplay")
        assert basic_manager._stats.jobs_started == 1

    def test_record_job_completed(self, basic_manager: JobOrchestrationManager):
        """Test recording job completed."""
        assert basic_manager._stats.jobs_completed == 0
        basic_manager.record_job_completed("selfplay")
        assert basic_manager._stats.jobs_completed == 1

    def test_record_job_failed(self, basic_manager: JobOrchestrationManager):
        """Test recording job failed."""
        assert basic_manager._stats.jobs_failed == 0
        basic_manager.record_job_failed("selfplay", "timeout")
        assert basic_manager._stats.jobs_failed == 1

    def test_record_spawn_blocked(self, basic_manager: JobOrchestrationManager):
        """Test recording spawn blocked."""
        assert basic_manager._stats.spawn_blocked_count == 0
        basic_manager.record_spawn_blocked("rate limited")
        assert basic_manager._stats.spawn_blocked_count == 1

    def test_record_cluster_management_run(self, basic_manager: JobOrchestrationManager):
        """Test recording cluster management run."""
        assert basic_manager._stats.cluster_management_runs == 0
        basic_manager.record_cluster_management_run()
        assert basic_manager._stats.cluster_management_runs == 1

    def test_record_work_executed(self, basic_manager: JobOrchestrationManager):
        """Test recording work executed."""
        assert basic_manager._stats.work_items_executed == 0
        basic_manager.record_work_executed("selfplay")
        assert basic_manager._stats.work_items_executed == 1

    def test_record_jobs_scaled_down(self, basic_manager: JobOrchestrationManager):
        """Test recording jobs scaled down."""
        assert basic_manager._stats.jobs_scaled_down == 0
        basic_manager.record_jobs_scaled_down(3)
        assert basic_manager._stats.jobs_scaled_down == 3

    def test_get_stats(self, basic_manager: JobOrchestrationManager):
        """Test getting statistics."""
        basic_manager.record_job_started()
        basic_manager.record_job_completed()
        stats = basic_manager.get_stats()
        assert stats.jobs_started == 1
        assert stats.jobs_completed == 1


# ============================================================================
# Stub Method Tests
# ============================================================================


class TestStubMethods:
    """Tests for stub methods (not yet implemented)."""

    @pytest.mark.asyncio
    async def test_start_local_job_stub(self, basic_manager: JobOrchestrationManager):
        """Test start_local_job stub returns None."""
        result = await basic_manager.start_local_job(
            job_type=MagicMock(),
            board_type="hex8",
            num_players=2,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_manage_cluster_jobs_non_leader(self, basic_manager: JobOrchestrationManager):
        """Test manage_cluster_jobs as non-leader returns early."""
        await basic_manager.manage_cluster_jobs()
        # Stats should not be incremented for non-leader
        assert basic_manager._stats.cluster_management_runs == 0

    @pytest.mark.asyncio
    async def test_manage_cluster_jobs_leader(self, leader_manager: JobOrchestrationManager):
        """Test manage_cluster_jobs as leader."""
        await leader_manager.manage_cluster_jobs()
        # Stats should be incremented for leader
        assert leader_manager._stats.cluster_management_runs == 1

    @pytest.mark.asyncio
    async def test_execute_claimed_work_stub(self, basic_manager: JobOrchestrationManager):
        """Test execute_claimed_work stub."""
        result = await basic_manager.execute_claimed_work({
            "work_type": "selfplay",
            "work_id": "test-123",
        })
        assert result is False
        assert basic_manager._stats.work_items_executed == 1

    @pytest.mark.asyncio
    async def test_manage_local_jobs_decentralized_stub(self, basic_manager: JobOrchestrationManager):
        """Test manage_local_jobs_decentralized stub."""
        result = await basic_manager.manage_local_jobs_decentralized()
        assert result == 0

    @pytest.mark.asyncio
    async def test_local_gpu_auto_scale_stub(self, basic_manager: JobOrchestrationManager):
        """Test local_gpu_auto_scale stub."""
        result = await basic_manager.local_gpu_auto_scale()
        assert result == 0

    @pytest.mark.asyncio
    async def test_auto_rebalance_from_work_queue_stub(self, basic_manager: JobOrchestrationManager):
        """Test auto_rebalance_from_work_queue stub."""
        result = await basic_manager.auto_rebalance_from_work_queue()
        assert result == 0

    @pytest.mark.asyncio
    async def test_check_cluster_balance_stub(self, basic_manager: JobOrchestrationManager):
        """Test check_cluster_balance stub."""
        result = await basic_manager.check_cluster_balance()
        assert result["status"] == "not_implemented"

    @pytest.mark.asyncio
    async def test_reduce_local_selfplay_jobs_stub(self, basic_manager: JobOrchestrationManager):
        """Test reduce_local_selfplay_jobs stub."""
        result = await basic_manager.reduce_local_selfplay_jobs(2, reason="test")
        assert result["jobs_killed"] == 0
        assert result["reason"] == "test"


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunction:
    """Tests for create_job_orchestration_manager factory."""

    def test_create_from_orchestrator(self):
        """Test creating manager from orchestrator mock."""
        orchestrator = MagicMock()
        orchestrator.node_id = "test-node"
        orchestrator.ringrift_path = "/path/to/ringrift"
        orchestrator.peers = {"peer-1": MagicMock()}
        orchestrator.local_jobs = {}
        orchestrator.jobs_lock = threading.RLock()
        orchestrator._is_leader = lambda: False
        orchestrator.self_info = MagicMock()
        orchestrator._save_state = MagicMock()
        orchestrator.peers_lock = threading.RLock()

        manager = create_job_orchestration_manager(orchestrator)

        assert manager.node_id == "test-node"
        assert manager.ringrift_path == "/path/to/ringrift"

    def test_factory_handles_missing_attributes(self):
        """Test factory handles missing optional attributes."""
        orchestrator = MagicMock()
        orchestrator.node_id = "test-node"
        orchestrator.ringrift_path = "/path/to/ringrift"
        orchestrator.peers = {}
        orchestrator.local_jobs = {}
        orchestrator.jobs_lock = threading.RLock()
        orchestrator._is_leader = lambda: False
        orchestrator.self_info = None
        orchestrator._save_state = MagicMock()
        orchestrator.peers_lock = threading.RLock()
        # Remove optional attributes
        del orchestrator.selfplay_scheduler
        del orchestrator.job_manager

        # Should not raise
        manager = create_job_orchestration_manager(orchestrator)
        assert manager._selfplay_scheduler is None


# ============================================================================
# Callback Access Tests
# ============================================================================


class TestCallbackAccess:
    """Tests for callback-based state access."""

    def test_get_peers_callback(self, jobs_lock: threading.RLock):
        """Test get_peers callback is used."""
        peers = {"peer-1": MagicMock(), "peer-2": MagicMock()}
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: peers,
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
        )
        assert manager._get_peers() == peers
        assert len(manager._get_peers()) == 2

    def test_is_leader_callback(self, jobs_lock: threading.RLock):
        """Test is_leader callback is used."""
        is_leader_state = [False]

        def is_leader():
            return is_leader_state[0]

        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: {},
            jobs_lock=jobs_lock,
            is_leader=is_leader,
            get_self_info=lambda: None,
        )

        assert manager._is_leader() is False

        is_leader_state[0] = True
        assert manager._is_leader() is True

    def test_get_local_jobs_callback(self, jobs_lock: threading.RLock):
        """Test get_local_jobs callback is used."""
        jobs = {"job-1": MagicMock()}
        manager = JobOrchestrationManager(
            node_id="test-node",
            ringrift_path="/path",
            get_peers=lambda: {},
            get_local_jobs=lambda: jobs,
            jobs_lock=jobs_lock,
            is_leader=lambda: False,
            get_self_info=lambda: None,
        )
        assert manager._get_local_jobs() == jobs


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_spawn_times_list(self, basic_manager: JobOrchestrationManager):
        """Test rate limiting with empty spawn times."""
        basic_manager._spawn_times = []
        allowed, _ = basic_manager._check_spawn_rate_limit()
        assert allowed is True

    def test_stats_increment_thread_safety(self, basic_manager: JobOrchestrationManager):
        """Test that stats can be incremented from multiple threads."""
        import threading

        def increment_stats():
            for _ in range(100):
                basic_manager.record_job_started()

        threads = [threading.Thread(target=increment_stats) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert basic_manager._stats.jobs_started == 500

    def test_health_check_after_multiple_operations(self, basic_manager: JobOrchestrationManager):
        """Test health check after various operations."""
        basic_manager.start()
        basic_manager.record_job_started()
        basic_manager.record_job_completed()
        basic_manager.record_job_failed("test", "error")

        health = basic_manager.health_check()
        stats = health["stats"]
        assert stats["jobs_started"] == 1
        assert stats["jobs_completed"] == 1
        assert stats["jobs_failed"] == 1
