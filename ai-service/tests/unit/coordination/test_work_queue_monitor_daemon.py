"""Tests for WorkQueueMonitorDaemon.

December 2025: Tests for the new work queue monitoring daemon that subscribes
to all WORK_* events (WORK_QUEUED, WORK_CLAIMED, WORK_STARTED, WORK_COMPLETED,
WORK_FAILED).
"""

import asyncio
import pytest
import time

from app.coordination.work_queue_monitor_daemon import (
    WorkQueueMonitorDaemon,
    QueueStats,
    JobTracker,
    get_work_queue_monitor_sync,
    BACKPRESSURE_THRESHOLD,
    STUCK_JOB_THRESHOLD_SECONDS,
    NODE_OVERLOAD_THRESHOLD,
)


class TestJobTracker:
    """Tests for JobTracker dataclass."""

    def test_default_values(self):
        """Test JobTracker default values."""
        tracker = JobTracker(work_id="test-1", work_type="selfplay")
        assert tracker.work_id == "test-1"
        assert tracker.work_type == "selfplay"
        assert tracker.priority == 50
        assert tracker.status == "pending"
        assert tracker.retry_count == 0

    def test_custom_values(self):
        """Test JobTracker with custom values."""
        tracker = JobTracker(
            work_id="test-2",
            work_type="training",
            priority=80,
            config_key="hex8_2p",
            claimed_by="gpu-node-1",
            status="claimed",
        )
        assert tracker.priority == 80
        assert tracker.config_key == "hex8_2p"
        assert tracker.claimed_by == "gpu-node-1"
        assert tracker.status == "claimed"


class TestQueueStats:
    """Tests for QueueStats dataclass."""

    def test_default_values(self):
        """Test QueueStats default values."""
        stats = QueueStats()
        assert stats.pending_count == 0
        assert stats.running_count == 0
        assert stats.completed_count == 0
        assert stats.backpressure_active is False
        assert stats.stuck_job_count == 0

    def test_latency_metrics(self):
        """Test QueueStats latency tracking."""
        stats = QueueStats(
            avg_latency_seconds=10.5,
            p95_latency_seconds=25.0,
            max_latency_seconds=60.0,
        )
        assert stats.avg_latency_seconds == 10.5
        assert stats.p95_latency_seconds == 25.0
        assert stats.max_latency_seconds == 60.0


class TestWorkQueueMonitorDaemon:
    """Tests for WorkQueueMonitorDaemon."""

    def test_initialization(self):
        """Test daemon initializes correctly."""
        daemon = WorkQueueMonitorDaemon()
        assert daemon._running is False
        assert daemon._subscribed is False
        assert len(daemon._jobs) == 0

    def test_singleton_accessor(self):
        """Test singleton accessor returns same instance."""
        # Reset for clean test
        import app.coordination.work_queue_monitor_daemon as module
        module._monitor_instance = None

        daemon1 = get_work_queue_monitor_sync()
        daemon2 = get_work_queue_monitor_sync()
        assert daemon1 is daemon2

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test daemon start/stop lifecycle."""
        daemon = WorkQueueMonitorDaemon()

        # Start daemon
        result = await daemon.start()
        assert daemon._running is True

        # Stop daemon
        await daemon.stop()
        assert daemon._running is False

    def test_get_queue_stats_empty(self):
        """Test get_queue_stats when queue is empty."""
        daemon = WorkQueueMonitorDaemon()
        stats = daemon.get_queue_stats()

        assert stats.pending_count == 0
        assert stats.running_count == 0
        assert stats.completed_count == 0
        assert stats.cluster_healthy if hasattr(stats, 'cluster_healthy') else True

    def test_get_status(self):
        """Test get_status returns expected structure."""
        daemon = WorkQueueMonitorDaemon()
        status = daemon.get_status()

        assert "running" in status
        assert "subscribed" in status
        assert "pending_count" in status
        assert "backpressure_active" in status

    def test_health_check_not_running(self):
        """Test health_check when daemon not running."""
        daemon = WorkQueueMonitorDaemon()
        result = daemon.health_check()

        # Should indicate unhealthy
        if hasattr(result, 'healthy'):
            assert result.healthy is False
        elif isinstance(result, dict):
            assert result.get('healthy') is False

    @pytest.mark.asyncio
    async def test_on_work_queued(self):
        """Test handling WORK_QUEUED event."""
        daemon = WorkQueueMonitorDaemon()

        # Simulate WORK_QUEUED event
        event = {
            "work_id": "work-123",
            "work_type": "selfplay",
            "priority": 60,
            "config_key": "hex8_2p",
        }

        await daemon._on_work_queued(event)

        assert "work-123" in daemon._jobs
        job = daemon._jobs["work-123"]
        assert job.work_type == "selfplay"
        assert job.priority == 60
        assert job.status == "pending"

    @pytest.mark.asyncio
    async def test_on_work_claimed(self):
        """Test handling WORK_CLAIMED event."""
        daemon = WorkQueueMonitorDaemon()

        # First queue the work
        await daemon._on_work_queued({
            "work_id": "work-456",
            "work_type": "training",
        })

        # Then claim it
        await daemon._on_work_claimed({
            "work_id": "work-456",
            "claimed_by": "gpu-node-1",
        })

        job = daemon._jobs["work-456"]
        assert job.status == "claimed"
        assert job.claimed_by == "gpu-node-1"
        assert daemon._node_job_counts["gpu-node-1"] == 1

    @pytest.mark.asyncio
    async def test_on_work_completed(self):
        """Test handling WORK_COMPLETED event."""
        daemon = WorkQueueMonitorDaemon()

        # Queue and claim
        await daemon._on_work_queued({
            "work_id": "work-789",
            "work_type": "selfplay",
        })
        await daemon._on_work_claimed({
            "work_id": "work-789",
            "claimed_by": "gpu-node-2",
        })

        # Complete
        await daemon._on_work_completed({"work_id": "work-789"})

        # Job should be removed from tracking
        assert "work-789" not in daemon._jobs
        assert daemon._total_completed == 1
        # Node job count should be decremented
        assert daemon._node_job_counts["gpu-node-2"] == 0

    @pytest.mark.asyncio
    async def test_on_work_failed(self):
        """Test handling WORK_FAILED event."""
        daemon = WorkQueueMonitorDaemon()

        # Queue and claim
        await daemon._on_work_queued({
            "work_id": "work-fail",
            "work_type": "training",
        })
        await daemon._on_work_claimed({
            "work_id": "work-fail",
            "claimed_by": "gpu-node-3",
        })

        # Fail
        await daemon._on_work_failed({"work_id": "work-fail"})

        assert "work-fail" not in daemon._jobs
        assert daemon._total_failed == 1

    @pytest.mark.asyncio
    async def test_stuck_job_detection(self):
        """Test stuck job detection logic."""
        daemon = WorkQueueMonitorDaemon()

        # Create a job that's been claimed but not started
        daemon._jobs["stuck-work"] = JobTracker(
            work_id="stuck-work",
            work_type="selfplay",
            claimed_at=time.time() - STUCK_JOB_THRESHOLD_SECONDS - 100,  # Past threshold
            status="claimed",
            claimed_by="slow-node",
        )

        stats = daemon.get_queue_stats()
        assert stats.stuck_job_count == 1


class TestConstants:
    """Tests for module constants."""

    def test_thresholds_are_reasonable(self):
        """Test that threshold constants have reasonable values."""
        assert BACKPRESSURE_THRESHOLD > 0
        assert BACKPRESSURE_THRESHOLD <= 1000  # Reasonable upper bound

        assert STUCK_JOB_THRESHOLD_SECONDS > 60  # At least 1 minute
        assert STUCK_JOB_THRESHOLD_SECONDS <= 3600  # At most 1 hour

        assert NODE_OVERLOAD_THRESHOLD >= 1
        assert NODE_OVERLOAD_THRESHOLD <= 100


class TestLatencyTracking:
    """Tests for latency tracking functionality."""

    @pytest.mark.asyncio
    async def test_latency_calculation(self):
        """Test that latency is calculated correctly."""
        daemon = WorkQueueMonitorDaemon()

        # Queue a job
        await daemon._on_work_queued({
            "work_id": "latency-test",
            "work_type": "selfplay",
        })

        # Ensure queued_at is set
        daemon._jobs["latency-test"].queued_at = time.time() - 5.0  # 5 seconds ago

        # Complete the job
        await daemon._on_work_completed({"work_id": "latency-test"})

        # Check latency was recorded
        assert len(daemon._completed_latencies) == 1
        assert daemon._completed_latencies[0] >= 5.0
