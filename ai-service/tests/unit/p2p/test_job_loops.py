"""Tests for P2P job management loops.

December 2025: Comprehensive tests for JobReaperLoop and IdleDetectionLoop.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.p2p.loops.job_loops import (
    IdleDetectionConfig,
    IdleDetectionLoop,
    JobReaperConfig,
    JobReaperLoop,
    WorkerPullConfig,
    WorkerPullLoop,
    WorkQueueMaintenanceConfig,
    WorkQueueMaintenanceLoop,
)


# =============================================================================
# JobReaperConfig Tests
# =============================================================================


class TestJobReaperConfig:
    """Tests for JobReaperConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = JobReaperConfig()
        assert config.stale_job_threshold_seconds == 3600.0  # 1 hour
        assert config.stuck_job_threshold_seconds == 7200.0  # 2 hours
        assert config.max_jobs_to_reap_per_cycle == 10
        assert config.check_interval_seconds == 300.0  # 5 minutes

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = JobReaperConfig(
            stale_job_threshold_seconds=1800.0,
            stuck_job_threshold_seconds=3600.0,
            max_jobs_to_reap_per_cycle=5,
            check_interval_seconds=60.0,
        )
        assert config.stale_job_threshold_seconds == 1800.0
        assert config.stuck_job_threshold_seconds == 3600.0
        assert config.max_jobs_to_reap_per_cycle == 5
        assert config.check_interval_seconds == 60.0


# =============================================================================
# JobReaperLoop Tests
# =============================================================================


class TestJobReaperLoop:
    """Tests for JobReaperLoop class."""

    def test_initialization(self) -> None:
        """Test basic loop initialization."""
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)

        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
        )

        assert loop.name == "job_reaper"
        assert loop.config is not None
        assert loop.interval == loop.config.check_interval_seconds
        assert loop._get_active_jobs is get_jobs
        assert loop._cancel_job is cancel_job

    def test_initialization_with_config(self) -> None:
        """Test initialization with custom config."""
        config = JobReaperConfig(check_interval_seconds=60.0)
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)

        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        assert loop.interval == 60.0

    @pytest.mark.asyncio
    async def test_run_once_no_jobs(self) -> None:
        """Test run_once with no active jobs."""
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)

        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
        )

        await loop._run_once()

        get_jobs.assert_called_once()
        cancel_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_reaps_stale_jobs(self) -> None:
        """Test run_once identifies and reaps stale jobs (claimed but not started)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "claimed",
                "claimed_at": now - 4000,  # Over 1 hour ago
                "started_at": 0,
            },
            "job-2": {
                "status": "claimed",
                "claimed_at": now - 100,  # Only 100 seconds ago (not stale)
                "started_at": 0,
            },
        })
        cancel_job = AsyncMock(return_value=True)

        config = JobReaperConfig(stale_job_threshold_seconds=3600.0)
        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        await loop._run_once()

        # Only job-1 should be reaped (it's stale)
        cancel_job.assert_called_once_with("job-1")
        stats = loop.get_reap_stats()
        assert stats["stale_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_reaps_stuck_jobs(self) -> None:
        """Test run_once identifies and reaps stuck jobs (running too long)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "running",
                "started_at": now - 8000,  # Over 2 hours ago (stuck)
                "claimed_at": now - 8500,
            },
            "job-2": {
                "status": "running",
                "started_at": now - 1000,  # Only ~16 minutes (not stuck)
                "claimed_at": now - 1500,
            },
        })
        cancel_job = AsyncMock(return_value=True)

        config = JobReaperConfig(stuck_job_threshold_seconds=7200.0)
        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        await loop._run_once()

        cancel_job.assert_called_once_with("job-1")
        stats = loop.get_reap_stats()
        assert stats["stuck_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_reaps_abandoned_jobs(self) -> None:
        """Test run_once identifies and reaps abandoned jobs (no heartbeat)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "running",
                "started_at": now - 1000,
                "claimed_at": now - 1500,
            },
        })
        get_heartbeats = MagicMock(return_value={
            "job-1": now - 5000,  # Last heartbeat over 1 hour ago
        })
        cancel_job = AsyncMock(return_value=True)

        config = JobReaperConfig(stale_job_threshold_seconds=3600.0)
        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            get_job_heartbeats=get_heartbeats,
            config=config,
        )

        await loop._run_once()

        cancel_job.assert_called_once_with("job-1")
        stats = loop.get_reap_stats()
        assert stats["abandoned_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_respects_max_jobs_per_cycle(self) -> None:
        """Test run_once respects max_jobs_to_reap_per_cycle."""
        now = time.time()
        # Create 15 stale jobs
        jobs = {
            f"job-{i}": {
                "status": "claimed",
                "claimed_at": now - 4000,
                "started_at": 0,
            }
            for i in range(15)
        }
        get_jobs = MagicMock(return_value=jobs)
        cancel_job = AsyncMock(return_value=True)

        config = JobReaperConfig(max_jobs_to_reap_per_cycle=5)
        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        await loop._run_once()

        # Only 5 jobs should be reaped due to limit
        assert cancel_job.call_count == 5
        stats = loop.get_reap_stats()
        assert stats["stale_jobs_reaped"] == 5

    @pytest.mark.asyncio
    async def test_run_once_handles_cancel_failure(self) -> None:
        """Test run_once handles cancel_job failures gracefully."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "claimed",
                "claimed_at": now - 4000,
                "started_at": 0,
            },
        })
        cancel_job = AsyncMock(side_effect=Exception("Cancel failed"))

        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
        )

        # Should not raise
        await loop._run_once()

        cancel_job.assert_called_once()
        stats = loop.get_reap_stats()
        assert stats["stale_jobs_reaped"] == 0  # Failed cancellation not counted

    def test_get_reap_stats(self) -> None:
        """Test get_reap_stats returns comprehensive statistics."""
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)

        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
        )

        stats = loop.get_reap_stats()
        assert "stale_jobs_reaped" in stats
        assert "stuck_jobs_reaped" in stats
        assert "abandoned_jobs_reaped" in stats
        assert "total_runs" in stats
        assert "successful_runs" in stats


# =============================================================================
# IdleDetectionConfig Tests
# =============================================================================


class TestIdleDetectionConfig:
    """Tests for IdleDetectionConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = IdleDetectionConfig()
        assert config.gpu_idle_threshold_percent == 10.0
        assert config.idle_duration_threshold_seconds == 60.0
        assert config.check_interval_seconds == 30.0
        assert config.min_nodes_to_keep == 2

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=15.0,
            idle_duration_threshold_seconds=120.0,
            check_interval_seconds=60.0,
            min_nodes_to_keep=3,
        )
        assert config.gpu_idle_threshold_percent == 15.0
        assert config.idle_duration_threshold_seconds == 120.0
        assert config.check_interval_seconds == 60.0
        assert config.min_nodes_to_keep == 3


# =============================================================================
# IdleDetectionLoop Tests
# =============================================================================


class TestIdleDetectionLoop:
    """Tests for IdleDetectionLoop class."""

    def test_initialization(self) -> None:
        """Test basic loop initialization."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={})

        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
        )

        assert loop.name == "idle_detection"
        assert loop.config is not None
        assert loop.interval == loop.config.check_interval_seconds

    def test_initialization_with_config(self) -> None:
        """Test initialization with custom config."""
        config = IdleDetectionConfig(check_interval_seconds=45.0)
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={})

        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            config=config,
        )

        assert loop.interval == 45.0

    @pytest.mark.asyncio
    async def test_run_once_skips_when_follower(self) -> None:
        """Test run_once skips processing when not leader."""
        get_role = MagicMock(return_value="follower")
        get_peers = MagicMock(return_value={})
        on_idle = AsyncMock()

        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
        )

        await loop._run_once()

        get_peers.assert_not_called()
        on_idle.assert_not_called()
        stats = loop.get_detection_stats()
        assert stats["skipped_not_leader"] == 1

    @pytest.mark.asyncio
    async def test_run_once_no_peers(self) -> None:
        """Test run_once handles no peers gracefully."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={})
        on_idle = AsyncMock()

        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
        )

        await loop._run_once()

        get_peers.assert_called_once()
        on_idle.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_detects_idle_node(self) -> None:
        """Test run_once detects an idle GPU node."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={
            "node-1": {
                "has_gpu": True,
                "gpu_percent": 5.0,  # Below threshold
                "selfplay_jobs": 0,
            },
            "node-2": {
                "has_gpu": True,
                "gpu_percent": 80.0,  # Active
                "selfplay_jobs": 1,
            },
            "node-3": {
                "has_gpu": True,
                "gpu_percent": 50.0,  # Active
                "selfplay_jobs": 0,
            },
        })
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=0.0,  # Immediate detection for testing
            min_nodes_to_keep=1,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        await loop._run_once()

        # node-1 should be detected as idle and callback invoked
        assert on_idle.call_count == 1
        call_args = on_idle.call_args[0]
        assert call_args[0]["has_gpu"] is True  # peer info
        assert call_args[1] >= 0  # idle duration

    @pytest.mark.asyncio
    async def test_run_once_respects_idle_duration_threshold(self) -> None:
        """Test run_once waits for idle duration threshold before triggering."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={
            "node-1": {
                "has_gpu": True,
                "gpu_percent": 5.0,
                "selfplay_jobs": 0,
            },
        })
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=60.0,  # 1 minute threshold
            min_nodes_to_keep=0,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        # First run - node becomes idle but not long enough
        await loop._run_once()
        assert on_idle.call_count == 0
        assert "node-1" in loop._idle_since

    @pytest.mark.asyncio
    async def test_run_once_respects_min_nodes_to_keep(self) -> None:
        """Test run_once doesn't flag last N nodes as idle."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={
            "node-1": {
                "has_gpu": True,
                "gpu_percent": 5.0,  # Idle
                "selfplay_jobs": 0,
            },
            "node-2": {
                "has_gpu": True,
                "gpu_percent": 5.0,  # Idle
                "selfplay_jobs": 0,
            },
        })
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=0.0,
            min_nodes_to_keep=2,  # Keep at least 2 nodes
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        await loop._run_once()

        # Should not trigger on_idle - would go below min_nodes_to_keep
        assert on_idle.call_count == 0

    @pytest.mark.asyncio
    async def test_run_once_node_becomes_active(self) -> None:
        """Test run_once removes node from idle tracking when it becomes active."""
        get_role = MagicMock(return_value="leader")
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=60.0,
            min_nodes_to_keep=0,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=MagicMock(return_value={
                "node-1": {"has_gpu": True, "gpu_percent": 5.0, "selfplay_jobs": 0},
            }),
            on_idle_detected=on_idle,
            config=config,
        )

        # First run - node becomes idle
        await loop._run_once()
        assert "node-1" in loop._idle_since

        # Update peers - node now active
        loop._get_peers = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 80.0, "selfplay_jobs": 1},
        })

        # Second run - node should be removed from idle tracking
        await loop._run_once()
        assert "node-1" not in loop._idle_since

    @pytest.mark.asyncio
    async def test_run_once_handles_callback_failure(self) -> None:
        """Test run_once handles on_idle_detected callback failures gracefully."""
        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 5.0, "selfplay_jobs": 0},
        })
        on_idle = AsyncMock(side_effect=Exception("Callback failed"))

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=0.0,
            min_nodes_to_keep=0,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        # Should not raise
        await loop._run_once()

        on_idle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_once_with_object_peer_info(self) -> None:
        """Test run_once handles object-style peer info (not just dicts)."""
        class MockPeer:
            has_gpu = True
            gpu_percent = 5.0
            selfplay_jobs = 0

        get_role = MagicMock(return_value="leader")
        get_peers = MagicMock(return_value={"node-1": MockPeer()})
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=0.0,
            min_nodes_to_keep=0,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        await loop._run_once()

        # Should detect idle node even with object-style info
        assert on_idle.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_get_node_metrics(self) -> None:
        """Test backward compatibility with get_node_metrics parameter."""
        get_role = MagicMock(return_value="leader")
        get_node_metrics = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 5.0, "selfplay_jobs": 0},
        })
        on_idle = AsyncMock()

        config = IdleDetectionConfig(
            gpu_idle_threshold_percent=10.0,
            idle_duration_threshold_seconds=0.0,
            min_nodes_to_keep=0,
        )
        loop = IdleDetectionLoop(
            get_role=get_role,
            get_node_metrics=get_node_metrics,  # Legacy parameter
            on_idle_detected=on_idle,
            config=config,
        )

        await loop._run_once()

        get_node_metrics.assert_called_once()
        assert on_idle.call_count == 1

    def test_get_idle_nodes(self) -> None:
        """Test get_idle_nodes returns currently tracked idle nodes."""
        loop = IdleDetectionLoop(
            get_role=MagicMock(return_value="leader"),
            get_peers=MagicMock(return_value={}),
        )

        # Manually add idle tracking
        now = time.time()
        loop._idle_since["node-1"] = now - 100
        loop._idle_since["node-2"] = now - 50

        idle_nodes = loop.get_idle_nodes()
        assert len(idle_nodes) == 2
        assert "node-1" in idle_nodes
        assert "node-2" in idle_nodes
        assert idle_nodes["node-1"] >= 100
        assert idle_nodes["node-2"] >= 50

    def test_get_detection_stats(self) -> None:
        """Test get_detection_stats returns comprehensive statistics."""
        loop = IdleDetectionLoop(
            get_role=MagicMock(return_value="leader"),
            get_peers=MagicMock(return_value={}),
        )

        stats = loop.get_detection_stats()
        assert "currently_idle" in stats
        assert "total_detections" in stats
        assert "skipped_not_leader" in stats
        assert "idle_nodes" in stats
        assert "total_runs" in stats


# =============================================================================
# WorkerPullLoop Tests
# =============================================================================


class TestWorkerPullConfig:
    """Tests for WorkerPullConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = WorkerPullConfig()
        assert config.pull_interval_seconds == 30.0
        assert config.gpu_idle_threshold_percent == 15.0
        assert config.cpu_idle_threshold_percent == 30.0
        assert config.initial_delay_seconds == 30.0


class TestWorkerPullLoop:
    """Tests for WorkerPullLoop class."""

    def test_initialization(self) -> None:
        """Test basic loop initialization."""
        loop = WorkerPullLoop(
            is_leader=MagicMock(return_value=False),
            get_leader_id=MagicMock(return_value="leader-1"),
            get_self_metrics=MagicMock(return_value={}),
            claim_work_from_leader=AsyncMock(return_value=None),
            execute_work=AsyncMock(return_value=True),
            report_work_result=AsyncMock(),
        )

        assert loop.name == "worker_pull"
        assert loop.interval == 30.0

    @pytest.mark.asyncio
    async def test_run_once_skips_when_leader(self) -> None:
        """Test run_once skips when this node is the leader."""
        claim_work = AsyncMock(return_value=None)
        loop = WorkerPullLoop(
            is_leader=MagicMock(return_value=True),
            get_leader_id=MagicMock(return_value="self"),
            get_self_metrics=MagicMock(return_value={}),
            claim_work_from_leader=claim_work,
            execute_work=AsyncMock(return_value=True),
            report_work_result=AsyncMock(),
        )

        await loop._run_once()

        claim_work.assert_not_called()
        stats = loop.get_pull_stats()
        assert stats["skipped_leader"] == 1

    @pytest.mark.asyncio
    async def test_run_once_skips_when_busy(self) -> None:
        """Test run_once skips when node is busy."""
        claim_work = AsyncMock(return_value=None)
        loop = WorkerPullLoop(
            is_leader=MagicMock(return_value=False),
            get_leader_id=MagicMock(return_value="leader-1"),
            get_self_metrics=MagicMock(return_value={
                "has_gpu": True,
                "gpu_percent": 80.0,  # Too busy
            }),
            claim_work_from_leader=claim_work,
            execute_work=AsyncMock(return_value=True),
            report_work_result=AsyncMock(),
        )

        await loop._run_once()

        claim_work.assert_not_called()
        stats = loop.get_pull_stats()
        assert stats["skipped_busy"] == 1


# =============================================================================
# WorkQueueMaintenanceLoop Tests
# =============================================================================


class TestWorkQueueMaintenanceConfig:
    """Tests for WorkQueueMaintenanceConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = WorkQueueMaintenanceConfig()
        assert config.maintenance_interval_seconds == 300.0
        assert config.cleanup_age_seconds == 86400.0
        assert config.initial_delay_seconds == 60.0


class TestWorkQueueMaintenanceLoop:
    """Tests for WorkQueueMaintenanceLoop class."""

    def test_initialization(self) -> None:
        """Test basic loop initialization."""
        loop = WorkQueueMaintenanceLoop(
            is_leader=MagicMock(return_value=True),
            get_work_queue=MagicMock(return_value=None),
        )

        assert loop.name == "work_queue_maintenance"
        assert loop.interval == 300.0

    @pytest.mark.asyncio
    async def test_run_once_skips_when_not_leader(self) -> None:
        """Test run_once skips when not leader."""
        get_work_queue = MagicMock()
        loop = WorkQueueMaintenanceLoop(
            is_leader=MagicMock(return_value=False),
            get_work_queue=get_work_queue,
        )

        await loop._run_once()

        get_work_queue.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_processes_timeouts(self) -> None:
        """Test run_once processes work queue timeouts."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.return_value = ["item-1", "item-2"]
        mock_wq.cleanup_old_items.return_value = 0

        loop = WorkQueueMaintenanceLoop(
            is_leader=MagicMock(return_value=True),
            get_work_queue=MagicMock(return_value=mock_wq),
        )

        await loop._run_once()

        mock_wq.check_timeouts.assert_called_once()
        stats = loop.get_maintenance_stats()
        assert stats["timeouts_processed"] == 2

    @pytest.mark.asyncio
    async def test_run_once_cleans_old_items(self) -> None:
        """Test run_once cleans up old work queue items."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.return_value = []
        mock_wq.cleanup_old_items.return_value = 5

        loop = WorkQueueMaintenanceLoop(
            is_leader=MagicMock(return_value=True),
            get_work_queue=MagicMock(return_value=mock_wq),
        )

        await loop._run_once()

        mock_wq.cleanup_old_items.assert_called_once()
        stats = loop.get_maintenance_stats()
        assert stats["items_cleaned"] == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestJobLoopIntegration:
    """Integration tests for job loops."""

    @pytest.mark.asyncio
    async def test_job_reaper_start_stop(self) -> None:
        """Test JobReaperLoop start and stop lifecycle."""
        loop = JobReaperLoop(
            get_active_jobs=MagicMock(return_value={}),
            cancel_job=AsyncMock(return_value=True),
            config=JobReaperConfig(check_interval_seconds=0.1),
        )

        # Start in background
        task = loop.start_background()
        assert loop.running is True

        # Let it run a few cycles
        await asyncio.sleep(0.3)

        # Stop gracefully
        success = await loop.stop_async(timeout=2.0)
        assert success is True
        assert loop.running is False

        stats = loop.get_reap_stats()
        assert stats["total_runs"] >= 1

    @pytest.mark.asyncio
    async def test_idle_detection_start_stop(self) -> None:
        """Test IdleDetectionLoop start and stop lifecycle."""
        loop = IdleDetectionLoop(
            get_role=MagicMock(return_value="leader"),
            get_peers=MagicMock(return_value={}),
            config=IdleDetectionConfig(check_interval_seconds=0.1),
        )

        # Start in background
        task = loop.start_background()
        assert loop.running is True

        # Let it run a few cycles
        await asyncio.sleep(0.3)

        # Stop gracefully
        success = await loop.stop_async(timeout=2.0)
        assert success is True
        assert loop.running is False

        stats = loop.get_detection_stats()
        assert stats["total_runs"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
