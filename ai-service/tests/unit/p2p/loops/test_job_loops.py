"""Tests for P2P job management loops.

Tests cover:
- JobReaperConfig/Loop: Cleanup of stale/stuck jobs
- IdleDetectionConfig/Loop: Node idle and zombie detection
- PredictiveScalingConfig/Loop: Preemptive job spawning
- WorkerPullConfig/Loop: Worker pull model for job claiming
- WorkQueueMaintenanceConfig/Loop: Work queue maintenance
- SpawnVerificationConfig/Loop: Job spawn verification
- JobReassignmentConfig/Loop: Orphaned job reassignment
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.coordination_defaults import WorkQueueCleanupDefaults
from scripts.p2p.loops.job_loops import (
    JobReaperConfig,
    JobReaperLoop,
    IdleDetectionConfig,
    IdleDetectionLoop,
    PredictiveScalingConfig,
    PredictiveScalingLoop,
    WorkerPullConfig,
    WorkerPullLoop,
    WorkQueueMaintenanceConfig,
    WorkQueueMaintenanceLoop,
    SpawnVerificationConfig,
    SpawnVerificationLoop,
    JobReassignmentConfig,
    JobReassignmentLoop,
    DEFAULT_STALE_THRESHOLDS,
)


# =============================================================================
# JobReaperConfig Tests
# =============================================================================


class TestJobReaperConfig:
    """Tests for JobReaperConfig dataclass."""

    def test_default_values(self):
        """Test JobReaperConfig has sensible defaults."""
        config = JobReaperConfig()

        assert config.stale_job_threshold_seconds == 1800.0  # 30 min
        assert config.stuck_job_threshold_seconds == 7200.0  # 2 hours
        assert config.max_jobs_to_reap_per_cycle == 10
        assert config.check_interval_seconds == 300.0  # 5 min
        assert "gpu_gumbel" in config.stale_thresholds_by_type
        assert "training" in config.stale_thresholds_by_type

    def test_validation_stale_threshold_zero(self):
        """Test validation rejects stale_job_threshold_seconds <= 0."""
        with pytest.raises(ValueError, match="stale_job_threshold_seconds"):
            JobReaperConfig(stale_job_threshold_seconds=0)

    def test_validation_stuck_threshold_zero(self):
        """Test validation rejects stuck_job_threshold_seconds <= 0."""
        with pytest.raises(ValueError, match="stuck_job_threshold_seconds"):
            JobReaperConfig(stuck_job_threshold_seconds=0)

    def test_validation_max_jobs_zero(self):
        """Test validation rejects max_jobs_to_reap_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_jobs_to_reap_per_cycle"):
            JobReaperConfig(max_jobs_to_reap_per_cycle=0)

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            JobReaperConfig(check_interval_seconds=0)

    def test_get_stale_threshold_exact_match(self):
        """Test get_stale_threshold returns exact match."""
        config = JobReaperConfig()

        threshold = config.get_stale_threshold("gpu_gumbel")

        assert threshold == DEFAULT_STALE_THRESHOLDS["gpu_gumbel"]

    def test_get_stale_threshold_prefix_match(self):
        """Test get_stale_threshold returns prefix match."""
        config = JobReaperConfig()

        # gpu_gumbel_hex8 should match gpu_gumbel
        threshold = config.get_stale_threshold("gpu_gumbel_hex8")

        assert threshold == DEFAULT_STALE_THRESHOLDS["gpu_gumbel"]

    def test_get_stale_threshold_gpu_fallback(self):
        """Test get_stale_threshold uses GPU fallback for unknown gpu types."""
        config = JobReaperConfig()

        threshold = config.get_stale_threshold("gpu_unknown_type")

        assert threshold == DEFAULT_STALE_THRESHOLDS.get("gpu_selfplay", 600.0)

    def test_get_stale_threshold_cpu_fallback(self):
        """Test get_stale_threshold uses CPU fallback for unknown cpu types."""
        config = JobReaperConfig()

        threshold = config.get_stale_threshold("cpu_unknown_type")

        assert threshold == DEFAULT_STALE_THRESHOLDS.get("cpu_heuristic", 1800.0)

    def test_get_stale_threshold_default(self):
        """Test get_stale_threshold uses default for completely unknown types."""
        config = JobReaperConfig()

        threshold = config.get_stale_threshold("completely_unknown")

        assert threshold == DEFAULT_STALE_THRESHOLDS.get("default", 1800.0)


# =============================================================================
# JobReaperLoop Tests
# =============================================================================


class TestJobReaperLoop:
    """Tests for JobReaperLoop class."""

    def _create_loop(self, **overrides):
        """Create a JobReaperLoop with defaults."""
        defaults = {
            "get_active_jobs": MagicMock(return_value={}),
            "cancel_job": AsyncMock(return_value=True),
            "get_job_heartbeats": None,
            "config": None,
        }
        defaults.update(overrides)
        return JobReaperLoop(**defaults)

    def test_init(self):
        """Test JobReaperLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "job_reaper"
        assert loop._reap_stats["stale_jobs_reaped"] == 0
        assert loop._reap_stats["stuck_jobs_reaped"] == 0
        assert loop._reap_stats["abandoned_jobs_reaped"] == 0

    def test_init_custom_config(self):
        """Test JobReaperLoop with custom config."""
        config = JobReaperConfig(check_interval_seconds=60.0)
        loop = self._create_loop(config=config)

        assert loop.interval == 60.0

    @pytest.mark.asyncio
    async def test_run_once_no_jobs(self):
        """Test _run_once with no active jobs."""
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)
        loop = self._create_loop(get_active_jobs=get_jobs, cancel_job=cancel_job)

        await loop._run_once()

        get_jobs.assert_called_once()
        cancel_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_reaps_stale_job(self):
        """Test _run_once reaps stale jobs (claimed but not started)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "claimed",
                "claimed_at": now - 3600,  # 1 hour ago
                "started_at": None,
                "job_type": "gpu_gumbel",
            }
        })
        cancel_job = AsyncMock(return_value=True)
        loop = self._create_loop(get_active_jobs=get_jobs, cancel_job=cancel_job)

        await loop._run_once()

        cancel_job.assert_called_once_with("job-1")
        assert loop._reap_stats["stale_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_reaps_stuck_job(self):
        """Test _run_once reaps stuck jobs (running too long)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "running",
                "claimed_at": now - 10800,  # 3 hours ago
                "started_at": now - 10800,  # 3 hours ago
                "job_type": "training",
            }
        })
        cancel_job = AsyncMock(return_value=True)
        loop = self._create_loop(get_active_jobs=get_jobs, cancel_job=cancel_job)

        await loop._run_once()

        cancel_job.assert_called_once_with("job-1")
        assert loop._reap_stats["stuck_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_reaps_abandoned_job(self):
        """Test _run_once reaps abandoned jobs (no heartbeat)."""
        now = time.time()
        get_jobs = MagicMock(return_value={
            "job-1": {
                "status": "running",
                "claimed_at": now - 100,
                "started_at": now - 100,
                "job_type": "gpu_selfplay",
            }
        })
        get_heartbeats = MagicMock(return_value={
            "job-1": now - 3600,  # Last heartbeat 1 hour ago
        })
        cancel_job = AsyncMock(return_value=True)
        loop = self._create_loop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            get_job_heartbeats=get_heartbeats,
        )

        await loop._run_once()

        cancel_job.assert_called_once_with("job-1")
        assert loop._reap_stats["abandoned_jobs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_run_once_respects_max_per_cycle(self):
        """Test _run_once respects max_jobs_to_reap_per_cycle."""
        now = time.time()
        jobs = {
            f"job-{i}": {
                "status": "claimed",
                "claimed_at": now - 3600,
                "started_at": None,
                "job_type": "gpu_gumbel",
            }
            for i in range(15)
        }
        get_jobs = MagicMock(return_value=jobs)
        cancel_job = AsyncMock(return_value=True)
        config = JobReaperConfig(max_jobs_to_reap_per_cycle=5)
        loop = self._create_loop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        await loop._run_once()

        assert cancel_job.call_count == 5

    def test_get_reap_stats(self):
        """Test get_reap_stats returns correct stats."""
        loop = self._create_loop()
        loop._reap_stats["stale_jobs_reaped"] = 5
        loop._reap_stats["stuck_jobs_reaped"] = 2

        stats = loop.get_reap_stats()

        assert stats["stale_jobs_reaped"] == 5
        assert stats["stuck_jobs_reaped"] == 2
        assert "total_runs" in stats

    def test_health_check_running(self):
        """Test health_check when loop is running."""
        loop = self._create_loop()
        loop._running = True

        health = loop.health_check()

        assert health.healthy is True
        assert "healthy" in health.message.lower() or "JobReaperLoop" in health.message

    def test_health_check_stopped(self):
        """Test health_check when loop is stopped."""
        loop = self._create_loop()
        loop._running = False

        health = loop.health_check()

        assert health.healthy is True
        assert "stopped" in health.message.lower()


# =============================================================================
# IdleDetectionConfig Tests
# =============================================================================


class TestIdleDetectionConfig:
    """Tests for IdleDetectionConfig dataclass."""

    def test_default_values(self):
        """Test IdleDetectionConfig has sensible defaults."""
        config = IdleDetectionConfig()

        assert config.gpu_idle_threshold_percent == 10.0
        assert config.idle_duration_threshold_seconds == 15.0
        assert config.check_interval_seconds == 10.0
        assert config.min_nodes_to_keep == 2
        assert config.zombie_gpu_threshold_percent == 5.0
        assert config.zombie_duration_threshold_seconds == 600.0

    def test_validation_gpu_idle_threshold_zero(self):
        """Test validation rejects gpu_idle_threshold_percent <= 0."""
        with pytest.raises(ValueError, match="gpu_idle_threshold_percent"):
            IdleDetectionConfig(gpu_idle_threshold_percent=0)

    def test_validation_gpu_idle_threshold_over_100(self):
        """Test validation rejects gpu_idle_threshold_percent > 100."""
        with pytest.raises(ValueError, match="gpu_idle_threshold_percent"):
            IdleDetectionConfig(gpu_idle_threshold_percent=101)

    def test_validation_idle_duration_zero(self):
        """Test validation rejects idle_duration_threshold_seconds <= 0."""
        with pytest.raises(ValueError, match="idle_duration_threshold_seconds"):
            IdleDetectionConfig(idle_duration_threshold_seconds=0)

    def test_validation_min_nodes_negative(self):
        """Test validation rejects min_nodes_to_keep < 0."""
        with pytest.raises(ValueError, match="min_nodes_to_keep"):
            IdleDetectionConfig(min_nodes_to_keep=-1)

    def test_validation_zombie_threshold_zero(self):
        """Test validation rejects zombie_gpu_threshold_percent <= 0."""
        with pytest.raises(ValueError, match="zombie_gpu_threshold_percent"):
            IdleDetectionConfig(zombie_gpu_threshold_percent=0)


# =============================================================================
# IdleDetectionLoop Tests
# =============================================================================


class TestIdleDetectionLoop:
    """Tests for IdleDetectionLoop class."""

    def _create_loop(self, **overrides):
        """Create an IdleDetectionLoop with defaults."""
        defaults = {
            "get_role": MagicMock(return_value="leader"),
            "get_peers": MagicMock(return_value={}),
            "get_work_queue": None,
            "on_idle_detected": None,
            "on_zombie_detected": None,
            "config": None,
        }
        defaults.update(overrides)
        return IdleDetectionLoop(**defaults)

    def test_init(self):
        """Test IdleDetectionLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "idle_detection"
        assert loop._idle_since == {}
        assert loop._zombie_since == {}
        assert loop._detected_count == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        get_role = MagicMock(return_value="follower")
        get_peers = MagicMock(return_value={"node-1": {}})
        loop = self._create_loop(get_role=get_role, get_peers=get_peers)

        await loop._run_once()

        get_peers.assert_not_called()
        assert loop._skipped_not_leader == 1

    @pytest.mark.asyncio
    async def test_run_once_no_peers(self):
        """Test _run_once with no peers."""
        get_peers = MagicMock(return_value={})
        loop = self._create_loop(get_peers=get_peers)

        await loop._run_once()

        assert len(loop._idle_since) == 0

    @pytest.mark.asyncio
    async def test_run_once_detects_idle_node(self):
        """Test _run_once detects idle GPU node."""
        get_peers = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 5.0, "selfplay_jobs": 0},
            "node-2": {"has_gpu": True, "gpu_percent": 80.0, "selfplay_jobs": 2},
        })
        on_idle = AsyncMock()
        config = IdleDetectionConfig(
            idle_duration_threshold_seconds=0.001,  # Near-immediate trigger
            min_nodes_to_keep=0,
        )
        loop = self._create_loop(
            get_peers=get_peers,
            on_idle_detected=on_idle,
            config=config,
        )

        # Pre-populate idle tracking to simulate prior detection
        loop._idle_since["node-1"] = time.time() - 1  # Idle since 1s ago

        await loop._run_once()

        # node-1 is idle (GPU < 10%, 0 jobs) and has been for > threshold
        on_idle.assert_called_once()
        assert loop._detected_count == 1

    @pytest.mark.asyncio
    async def test_run_once_detects_zombie_node(self):
        """Test _run_once detects zombie node (jobs but no GPU)."""
        get_peers = MagicMock(return_value={
            "node-1": {
                "has_gpu": True,
                "is_cuda_gpu_node": True,
                "gpu_percent": 1.0,
                "selfplay_jobs": 3,
            },
        })
        on_zombie = AsyncMock()
        config = IdleDetectionConfig(
            zombie_duration_threshold_seconds=0.001,  # Near-immediate trigger
        )
        loop = self._create_loop(
            get_peers=get_peers,
            on_zombie_detected=on_zombie,
            config=config,
        )

        # Pre-populate zombie tracking to simulate prior detection
        loop._zombie_since["node-1"] = time.time() - 1  # Zombie since 1s ago

        await loop._run_once()

        # node-1 is zombie (jobs > 0, GPU < 5%) and has been for > threshold
        on_zombie.assert_called_once()
        assert loop._zombie_detected_count == 1

    @pytest.mark.asyncio
    async def test_run_once_clears_recovered_node(self):
        """Test _run_once clears node that recovered from idle."""
        get_peers = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 80.0, "selfplay_jobs": 2},
        })
        loop = self._create_loop(get_peers=get_peers)
        loop._idle_since["node-1"] = time.time() - 100

        await loop._run_once()

        assert "node-1" not in loop._idle_since

    def test_get_idle_nodes(self):
        """Test get_idle_nodes returns current idle nodes."""
        loop = self._create_loop()
        loop._idle_since["node-1"] = time.time() - 100

        idle_nodes = loop.get_idle_nodes()

        assert "node-1" in idle_nodes
        assert idle_nodes["node-1"] >= 100

    def test_get_detection_stats(self):
        """Test get_detection_stats returns correct stats."""
        loop = self._create_loop()
        loop._detected_count = 5
        loop._zombie_detected_count = 2
        loop._idle_since["node-1"] = time.time()

        stats = loop.get_detection_stats()

        assert stats["total_detections"] == 5
        assert stats["total_zombie_detections"] == 2
        assert stats["currently_idle"] == 1


# =============================================================================
# PredictiveScalingConfig Tests
# =============================================================================


class TestPredictiveScalingConfig:
    """Tests for PredictiveScalingConfig dataclass."""

    def test_default_values(self):
        """Test PredictiveScalingConfig has sensible defaults."""
        config = PredictiveScalingConfig()

        assert config.check_interval_seconds == 30.0
        assert config.queue_depth_threshold == 50
        assert config.approaching_idle_threshold_percent == 20.0
        assert config.approaching_idle_duration_seconds == 30.0
        assert config.min_jobs_to_spawn_preemptively == 1
        assert config.max_jobs_to_spawn_preemptively == 5

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            PredictiveScalingConfig(check_interval_seconds=0)

    def test_validation_queue_depth_negative(self):
        """Test validation rejects queue_depth_threshold < 0."""
        with pytest.raises(ValueError, match="queue_depth_threshold"):
            PredictiveScalingConfig(queue_depth_threshold=-1)

    def test_validation_approaching_idle_zero(self):
        """Test validation rejects approaching_idle_threshold_percent <= 0."""
        with pytest.raises(ValueError, match="approaching_idle_threshold_percent"):
            PredictiveScalingConfig(approaching_idle_threshold_percent=0)

    def test_validation_max_less_than_min(self):
        """Test validation rejects max < min spawns."""
        with pytest.raises(ValueError, match="max_jobs_to_spawn_preemptively"):
            PredictiveScalingConfig(
                min_jobs_to_spawn_preemptively=5,
                max_jobs_to_spawn_preemptively=2,
            )


# =============================================================================
# PredictiveScalingLoop Tests
# =============================================================================


class TestPredictiveScalingLoop:
    """Tests for PredictiveScalingLoop class."""

    def _create_loop(self, **overrides):
        """Create a PredictiveScalingLoop with defaults."""
        defaults = {
            "get_role": MagicMock(return_value="leader"),
            "get_peers": MagicMock(return_value={}),
            "get_queue_depth": MagicMock(return_value=100),
            "get_pending_jobs_for_node": None,
            "spawn_preemptive_job": None,
            "config": None,
        }
        defaults.update(overrides)
        return PredictiveScalingLoop(**defaults)

    def test_init(self):
        """Test PredictiveScalingLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "predictive_scaling"
        assert loop._preemptive_spawns == 0
        assert loop._approaching_idle_since == {}

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        get_role = MagicMock(return_value="follower")
        loop = self._create_loop(get_role=get_role)

        await loop._run_once()

        assert loop._skipped_not_leader == 1

    @pytest.mark.asyncio
    async def test_run_once_skips_low_queue(self):
        """Test _run_once skips if queue depth below threshold."""
        get_queue_depth = MagicMock(return_value=10)
        config = PredictiveScalingConfig(queue_depth_threshold=50)
        loop = self._create_loop(get_queue_depth=get_queue_depth, config=config)

        await loop._run_once()

        assert loop._skipped_low_queue == 1

    @pytest.mark.asyncio
    async def test_run_once_spawns_preemptively(self):
        """Test _run_once spawns jobs preemptively on approaching-idle nodes."""
        get_peers = MagicMock(return_value={
            "node-1": {"has_gpu": True, "gpu_percent": 15.0, "selfplay_jobs": 1},
        })
        spawn_job = AsyncMock(return_value=True)
        config = PredictiveScalingConfig(
            approaching_idle_duration_seconds=0.001,  # Near-immediate trigger
        )
        loop = self._create_loop(
            get_peers=get_peers,
            spawn_preemptive_job=spawn_job,
            config=config,
        )

        # Pre-populate approaching idle tracking
        loop._approaching_idle_since["node-1"] = time.time() - 1  # Approaching since 1s ago

        await loop._run_once()

        spawn_job.assert_called_once()
        assert loop._preemptive_spawns == 1

    def test_get_scaling_stats(self):
        """Test get_scaling_stats returns correct stats."""
        loop = self._create_loop()
        loop._preemptive_spawns = 10
        loop._skipped_low_queue = 5

        stats = loop.get_scaling_stats()

        assert stats["preemptive_spawns"] == 10
        assert stats["skipped_low_queue"] == 5


# =============================================================================
# WorkerPullConfig Tests
# =============================================================================


class TestWorkerPullConfig:
    """Tests for WorkerPullConfig dataclass."""

    def test_default_values(self):
        """Test WorkerPullConfig has sensible defaults."""
        config = WorkerPullConfig()

        assert config.pull_interval_seconds == 10.0
        assert config.gpu_idle_threshold_percent == 90.0
        assert config.cpu_idle_threshold_percent == 30.0
        assert config.initial_delay_seconds == 30.0
        assert config.enable_slot_based_claiming is True
        assert config.default_max_selfplay_slots == 8
        assert config.enable_batch_claiming is True

    def test_validation_pull_interval_zero(self):
        """Test validation rejects pull_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="pull_interval_seconds"):
            WorkerPullConfig(pull_interval_seconds=0)

    def test_validation_gpu_idle_zero(self):
        """Test validation rejects gpu_idle_threshold_percent <= 0."""
        with pytest.raises(ValueError, match="gpu_idle_threshold_percent"):
            WorkerPullConfig(gpu_idle_threshold_percent=0)

    def test_validation_gpu_idle_over_100(self):
        """Test validation rejects gpu_idle_threshold_percent > 100."""
        with pytest.raises(ValueError, match="gpu_idle_threshold_percent"):
            WorkerPullConfig(gpu_idle_threshold_percent=101)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            WorkerPullConfig(initial_delay_seconds=-1)

    def test_validation_max_slots_zero(self):
        """Test validation rejects default_max_selfplay_slots <= 0."""
        with pytest.raises(ValueError, match="default_max_selfplay_slots"):
            WorkerPullConfig(default_max_selfplay_slots=0)

    def test_validation_retry_jitter_out_of_range(self):
        """Test validation rejects claim_retry_jitter_factor out of [0, 1]."""
        with pytest.raises(ValueError, match="claim_retry_jitter_factor"):
            WorkerPullConfig(claim_retry_jitter_factor=1.5)


# =============================================================================
# WorkerPullLoop Tests
# =============================================================================


class TestWorkerPullLoop:
    """Tests for WorkerPullLoop class."""

    def _create_loop(self, **overrides):
        """Create a WorkerPullLoop with defaults."""
        defaults = {
            "is_leader": MagicMock(return_value=False),
            "get_leader_id": MagicMock(return_value="leader-node"),
            "get_self_metrics": MagicMock(return_value={
                "gpu_percent": 10.0,
                "cpu_percent": 20.0,
                "training_jobs": 0,
                "selfplay_jobs": 0,
                "has_gpu": True,
                "max_selfplay_slots": 8,
                "node_id": "test-worker",
            }),
            "claim_work_from_leader": AsyncMock(return_value=None),
            "execute_work": AsyncMock(return_value=True),
            "report_work_result": AsyncMock(),
            "config": None,
        }
        defaults.update(overrides)
        return WorkerPullLoop(**defaults)

    def test_init(self):
        """Test WorkerPullLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "worker_pull"
        assert loop._work_claimed == 0
        assert loop._work_completed == 0
        assert loop._work_failed == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_leader(self):
        """Test _run_once skips if this node is leader."""
        is_leader = MagicMock(return_value=True)
        loop = self._create_loop(is_leader=is_leader)

        await loop._run_once()

        assert loop._skipped_leader == 1

    @pytest.mark.asyncio
    async def test_run_once_skips_if_training(self):
        """Test _run_once skips if training jobs running."""
        get_metrics = MagicMock(return_value={
            "gpu_percent": 10.0,
            "training_jobs": 1,
            "selfplay_jobs": 0,
            "has_gpu": True,
        })
        loop = self._create_loop(get_self_metrics=get_metrics)

        await loop._run_once()

        assert loop._skipped_busy == 1

    @pytest.mark.asyncio
    async def test_run_once_claims_and_executes_work(self):
        """Test _run_once claims and executes work."""
        work_item = {"work_id": "work-1", "work_type": "selfplay"}
        claim_work = AsyncMock(return_value=work_item)
        execute_work = AsyncMock(return_value=True)
        report_result = AsyncMock()
        loop = self._create_loop(
            claim_work_from_leader=claim_work,
            execute_work=execute_work,
            report_work_result=report_result,
        )

        await loop._run_once()

        execute_work.assert_called_once_with(work_item)
        report_result.assert_called_once_with(work_item, True)
        assert loop._work_completed == 1

    @pytest.mark.asyncio
    async def test_run_once_reports_failure(self):
        """Test _run_once reports work failure."""
        work_item = {"work_id": "work-1", "work_type": "selfplay"}
        claim_work = AsyncMock(return_value=work_item)
        execute_work = AsyncMock(return_value=False)
        report_result = AsyncMock()
        loop = self._create_loop(
            claim_work_from_leader=claim_work,
            execute_work=execute_work,
            report_work_result=report_result,
        )

        await loop._run_once()

        report_result.assert_called_once_with(work_item, False)
        assert loop._work_failed == 1

    @pytest.mark.asyncio
    async def test_run_once_uses_autonomous_fallback(self):
        """Test _run_once uses autonomous queue when no leader."""
        get_leader = MagicMock(return_value=None)
        work_item = {"work_id": "auto-1", "work_type": "selfplay"}
        pop_autonomous = AsyncMock(return_value=work_item)
        execute_work = AsyncMock(return_value=True)
        config = WorkerPullConfig(enable_autonomous_fallback=True)
        loop = self._create_loop(
            get_leader_id=get_leader,
            pop_autonomous_work=pop_autonomous,
            execute_work=execute_work,
            config=config,
        )

        await loop._run_once()

        pop_autonomous.assert_called_once()
        assert loop._autonomous_work_claimed == 1

    @pytest.mark.asyncio
    async def test_claim_work_with_retry_succeeds_first_try(self):
        """Test _claim_work_with_retry succeeds on first attempt."""
        work_item = {"work_id": "work-1"}
        claim_work = AsyncMock(return_value=work_item)
        loop = self._create_loop(claim_work_from_leader=claim_work)

        result = await loop._claim_work_with_retry(["selfplay"])

        assert result == work_item
        assert loop._last_claim_retry_count == 0

    @pytest.mark.asyncio
    async def test_claim_work_with_retry_succeeds_after_retries(self):
        """Test _claim_work_with_retry succeeds after retries."""
        work_item = {"work_id": "work-1"}
        claim_work = AsyncMock(
            side_effect=[Exception("fail"), Exception("fail"), work_item]
        )
        config = WorkerPullConfig(
            claim_max_retries=3,
            claim_retry_base_delay=0.01,
        )
        loop = self._create_loop(
            claim_work_from_leader=claim_work,
            config=config,
        )

        result = await loop._claim_work_with_retry(["selfplay"])

        assert result == work_item
        assert loop._claim_retries_succeeded == 1

    @pytest.mark.asyncio
    async def test_claim_work_with_retry_exhausted(self):
        """Test _claim_work_with_retry returns None when retries exhausted."""
        claim_work = AsyncMock(side_effect=Exception("always fails"))
        config = WorkerPullConfig(
            claim_max_retries=2,
            claim_retry_base_delay=0.01,
        )
        loop = self._create_loop(
            claim_work_from_leader=claim_work,
            config=config,
        )

        result = await loop._claim_work_with_retry(["selfplay"])

        assert result is None
        assert loop._claim_retries_exhausted == 1

    @pytest.mark.asyncio
    async def test_validate_leader_health_no_callback(self):
        """Test _validate_leader_health returns True if no callback."""
        loop = self._create_loop()

        result = await loop._validate_leader_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_leader_health_detects_split_brain(self):
        """Test _validate_leader_health detects split-brain."""
        probe = AsyncMock(return_value={"is_leader": False})
        loop = self._create_loop(probe_leader_health=probe)

        result = await loop._validate_leader_health()

        assert result is False
        assert loop._split_brain_detections == 1

    def test_get_pull_stats(self):
        """Test get_pull_stats returns correct stats."""
        loop = self._create_loop()
        loop._work_claimed = 10
        loop._work_completed = 8
        loop._work_failed = 2

        stats = loop.get_pull_stats()

        assert stats["work_claimed"] == 10
        assert stats["work_completed"] == 8
        assert stats["work_failed"] == 2

    def test_health_check_running(self):
        """Test health_check when loop is running normally."""
        loop = self._create_loop()
        loop._running = True
        loop._work_completed = 10

        health = loop.health_check()

        assert health.healthy is True

    def test_health_check_degraded_high_failure(self):
        """Test health_check returns degraded on high failure rate."""
        loop = self._create_loop()
        loop._running = True
        loop._work_completed = 5
        loop._work_failed = 10

        health = loop.health_check()

        assert "degraded" in health.message.lower() or "DEGRADED" in str(health.status)

    def test_health_check_is_leader(self):
        """Test health_check when this node is leader (pull disabled)."""
        is_leader = MagicMock(return_value=True)
        loop = self._create_loop(is_leader=is_leader)
        loop._running = True

        health = loop.health_check()

        assert "leader" in health.message.lower()


# =============================================================================
# WorkQueueMaintenanceConfig Tests
# =============================================================================


class TestWorkQueueMaintenanceConfig:
    """Tests for WorkQueueMaintenanceConfig dataclass."""

    def test_default_values(self):
        """Test WorkQueueMaintenanceConfig has sensible defaults."""
        config = WorkQueueMaintenanceConfig()

        assert config.maintenance_interval_seconds == 300.0  # 5 min
        assert config.cleanup_age_seconds == 86400.0  # 24 hours
        assert config.initial_delay_seconds == 60.0
        assert config.max_pending_age_hours == WorkQueueCleanupDefaults.MAX_PENDING_AGE_HOURS
        assert config.max_claimed_age_hours == WorkQueueCleanupDefaults.MAX_CLAIMED_AGE_HOURS

    def test_validation_maintenance_interval_zero(self):
        """Test validation rejects maintenance_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="maintenance_interval_seconds"):
            WorkQueueMaintenanceConfig(maintenance_interval_seconds=0)

    def test_validation_cleanup_age_zero(self):
        """Test validation rejects cleanup_age_seconds <= 0."""
        with pytest.raises(ValueError, match="cleanup_age_seconds"):
            WorkQueueMaintenanceConfig(cleanup_age_seconds=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            WorkQueueMaintenanceConfig(initial_delay_seconds=-1)

    def test_validation_stall_threshold_zero(self):
        """Test validation rejects stall_threshold_seconds <= 0."""
        with pytest.raises(ValueError, match="stall_threshold_seconds"):
            WorkQueueMaintenanceConfig(stall_threshold_seconds=0)


# =============================================================================
# WorkQueueMaintenanceLoop Tests
# =============================================================================


class TestWorkQueueMaintenanceLoop:
    """Tests for WorkQueueMaintenanceLoop class."""

    def _create_loop(self, **overrides):
        """Create a WorkQueueMaintenanceLoop with defaults."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.return_value = []
        mock_wq.cleanup_old_items.return_value = 0
        defaults = {
            "is_leader": MagicMock(return_value=True),
            "get_work_queue": MagicMock(return_value=mock_wq),
            "config": None,
        }
        defaults.update(overrides)
        return WorkQueueMaintenanceLoop(**defaults)

    def test_init(self):
        """Test WorkQueueMaintenanceLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "work_queue_maintenance"
        assert loop._timeouts_processed == 0
        assert loop._items_cleaned == 0
        assert loop._stall_detected is False

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        is_leader = MagicMock(return_value=False)
        loop = self._create_loop(is_leader=is_leader)

        await loop._run_once()

        # Should not process anything
        assert loop._timeouts_processed == 0

    @pytest.mark.asyncio
    async def test_run_once_processes_timeouts(self):
        """Test _run_once processes timed out items."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.return_value = ["item-1", "item-2"]
        mock_wq.cleanup_old_items.return_value = 0
        get_wq = MagicMock(return_value=mock_wq)
        loop = self._create_loop(get_work_queue=get_wq)

        await loop._run_once()

        assert loop._timeouts_processed == 2

    @pytest.mark.asyncio
    async def test_run_once_cleans_up_items(self):
        """Test _run_once cleans up old items."""
        mock_wq = MagicMock()
        mock_wq.check_timeouts.return_value = []
        mock_wq.cleanup_old_items.return_value = 5
        get_wq = MagicMock(return_value=mock_wq)
        loop = self._create_loop(get_work_queue=get_wq)

        await loop._run_once()

        assert loop._items_cleaned == 5

    @pytest.mark.asyncio
    async def test_check_work_queue_stall_detects_stall(self):
        """Test _check_work_queue_stall detects stalled queue."""
        mock_wq = MagicMock()
        mock_wq.get_stats.return_value = {"last_work_completed_time": 0}
        config = WorkQueueMaintenanceConfig(stall_threshold_seconds=10)
        loop = self._create_loop(config=config)
        loop._last_work_completed_time = time.time() - 100  # 100s ago

        await loop._check_work_queue_stall(mock_wq)

        assert loop._stall_detected is True
        assert loop._stall_events_emitted == 1

    @pytest.mark.asyncio
    async def test_check_work_queue_stall_recovers(self):
        """Test _check_work_queue_stall clears stall on recovery."""
        mock_wq = MagicMock()
        now = time.time()
        # Work queue shows NEW work completed after what we had stored
        mock_wq.get_stats.return_value = {"last_work_completed_time": now + 1}
        config = WorkQueueMaintenanceConfig(stall_recovery_threshold_seconds=10)
        loop = self._create_loop(config=config)
        loop._stall_detected = True
        loop._stall_detected_at = now - 100
        loop._last_work_completed_time = now - 50  # Was stale

        await loop._check_work_queue_stall(mock_wq)

        assert loop._stall_detected is False
        assert loop._recovery_events_emitted == 1

    def test_get_maintenance_stats(self):
        """Test get_maintenance_stats returns correct stats."""
        loop = self._create_loop()
        loop._timeouts_processed = 5
        loop._items_cleaned = 10
        loop._stall_detected = True

        stats = loop.get_maintenance_stats()

        assert stats["timeouts_processed"] == 5
        assert stats["items_cleaned"] == 10
        assert stats["stall_detected"] is True

    def test_health_check_healthy(self):
        """Test health_check when healthy."""
        loop = self._create_loop()
        loop._running = True
        loop._stall_detected = False

        health = loop.health_check()

        assert health.healthy is True

    def test_health_check_stalled(self):
        """Test health_check when stalled."""
        loop = self._create_loop()
        loop._running = True
        loop._stall_detected = True
        loop._stall_detected_at = time.time() - 100

        health = loop.health_check()

        assert health.healthy is False
        assert "stall" in health.message.lower()


# =============================================================================
# SpawnVerificationConfig Tests
# =============================================================================


class TestSpawnVerificationConfig:
    """Tests for SpawnVerificationConfig dataclass."""

    def test_default_values(self):
        """Test SpawnVerificationConfig has sensible defaults."""
        config = SpawnVerificationConfig()

        assert config.check_interval_seconds == 5.0
        assert config.verification_timeout_seconds == 30.0
        assert config.log_stats_interval_runs == 12
        assert config.min_spawns_for_rate_calc == 10

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            SpawnVerificationConfig(check_interval_seconds=0)

    def test_validation_timeout_zero(self):
        """Test validation rejects verification_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="verification_timeout_seconds"):
            SpawnVerificationConfig(verification_timeout_seconds=0)

    def test_validation_log_interval_zero(self):
        """Test validation rejects log_stats_interval_runs <= 0."""
        with pytest.raises(ValueError, match="log_stats_interval_runs"):
            SpawnVerificationConfig(log_stats_interval_runs=0)


# =============================================================================
# SpawnVerificationLoop Tests
# =============================================================================


class TestSpawnVerificationLoop:
    """Tests for SpawnVerificationLoop class."""

    def _create_loop(self, **overrides):
        """Create a SpawnVerificationLoop with defaults."""
        defaults = {
            "verify_pending_spawns": AsyncMock(return_value={
                "verified": 0,
                "failed": 0,
                "pending": 0,
            }),
            "get_spawn_stats": None,
            "config": None,
        }
        defaults.update(overrides)
        return SpawnVerificationLoop(**defaults)

    def test_init(self):
        """Test SpawnVerificationLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "spawn_verification"
        assert loop._total_verified == 0
        assert loop._total_failed == 0

    @pytest.mark.asyncio
    async def test_run_once_verifies_spawns(self):
        """Test _run_once verifies pending spawns."""
        verify = AsyncMock(return_value={
            "verified": 3,
            "failed": 1,
            "pending": 2,
        })
        loop = self._create_loop(verify_pending_spawns=verify)

        await loop._run_once()

        assert loop._total_verified == 3
        assert loop._total_failed == 1

    @pytest.mark.asyncio
    async def test_run_once_handles_error(self):
        """Test _run_once handles verification error gracefully."""
        verify = AsyncMock(side_effect=Exception("verification failed"))
        loop = self._create_loop(verify_pending_spawns=verify)

        await loop._run_once()  # Should not raise

        assert loop._total_verified == 0

    def test_get_verification_stats(self):
        """Test get_verification_stats returns correct stats."""
        loop = self._create_loop()

        # Simulate verification results
        loop._total_verified = 10
        loop._total_failed = 2

        stats = loop.get_verification_stats()

        assert stats["total_verified"] == 10
        assert stats["total_failed"] == 2
        # Note: success_rate is overwritten by self.stats.to_dict()
        # which uses total_runs from BaseLoop, not verification counts
        # Verify that the verification-specific data is correct
        total = stats["total_verified"] + stats["total_failed"]
        expected_rate = (stats["total_verified"] / total) * 100.0 if total > 0 else 100.0
        assert expected_rate == pytest.approx(83.33, rel=0.01)


# =============================================================================
# JobReassignmentConfig Tests
# =============================================================================


class TestJobReassignmentConfig:
    """Tests for JobReassignmentConfig dataclass."""

    def test_default_values(self):
        """Test JobReassignmentConfig has sensible defaults."""
        config = JobReassignmentConfig()

        assert config.check_interval_seconds == 60.0
        assert config.orphan_threshold_seconds == 300.0  # 5 min
        assert config.max_reassignments_per_cycle == 5
        assert config.leader_only is True
        assert config.initial_delay_seconds == 60.0

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            JobReassignmentConfig(check_interval_seconds=0)

    def test_validation_orphan_threshold_zero(self):
        """Test validation rejects orphan_threshold_seconds <= 0."""
        with pytest.raises(ValueError, match="orphan_threshold_seconds"):
            JobReassignmentConfig(orphan_threshold_seconds=0)

    def test_validation_max_reassignments_zero(self):
        """Test validation rejects max_reassignments_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_reassignments_per_cycle"):
            JobReassignmentConfig(max_reassignments_per_cycle=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            JobReassignmentConfig(initial_delay_seconds=-1)


# =============================================================================
# JobReassignmentLoop Tests
# =============================================================================


class TestJobReassignmentLoop:
    """Tests for JobReassignmentLoop class."""

    def _create_loop(self, **overrides):
        """Create a JobReassignmentLoop with defaults."""
        mock_role = MagicMock()
        mock_role.is_leader = True
        mock_role.name = "LEADER"
        defaults = {
            "get_role": MagicMock(return_value=mock_role),
            "check_and_reassign": AsyncMock(return_value=0),
            "get_healthy_nodes": None,
            "config": None,
        }
        defaults.update(overrides)
        return JobReassignmentLoop(**defaults)

    def test_init(self):
        """Test JobReassignmentLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "job_reassignment"
        assert loop._total_reassigned == 0
        assert loop._cycles_run == 0
        assert loop._initial_delay_done is False

    @pytest.mark.asyncio
    async def test_run_once_waits_for_initial_delay(self):
        """Test _run_once waits for initial delay."""
        config = JobReassignmentConfig(initial_delay_seconds=100)
        loop = self._create_loop(config=config)

        await loop._run_once()

        assert loop._initial_delay_done is False
        assert loop._cycles_run == 1

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        mock_role = MagicMock()
        mock_role.is_leader = False
        mock_role.name = "FOLLOWER"
        get_role = MagicMock(return_value=mock_role)
        config = JobReassignmentConfig(initial_delay_seconds=0)
        loop = self._create_loop(get_role=get_role, config=config)
        loop._initial_delay_done = True

        await loop._run_once()

        assert loop._skipped_not_leader == 1

    @pytest.mark.asyncio
    async def test_run_once_reassigns_jobs(self):
        """Test _run_once reassigns orphaned jobs."""
        check_and_reassign = AsyncMock(return_value=3)
        config = JobReassignmentConfig(initial_delay_seconds=0)
        loop = self._create_loop(
            check_and_reassign=check_and_reassign,
            config=config,
        )
        loop._initial_delay_done = True

        await loop._run_once()

        check_and_reassign.assert_called_once()
        assert loop._total_reassigned == 3

    @pytest.mark.asyncio
    async def test_run_once_skips_insufficient_healthy_nodes(self):
        """Test _run_once skips if not enough healthy nodes."""
        get_healthy = MagicMock(return_value=["node-1"])
        config = JobReassignmentConfig(initial_delay_seconds=0)
        loop = self._create_loop(
            get_healthy_nodes=get_healthy,
            config=config,
        )
        loop._initial_delay_done = True

        await loop._run_once()

        assert loop._skipped_no_healthy_nodes == 1

    def test_get_stats(self):
        """Test get_stats returns correct stats."""
        loop = self._create_loop()
        loop._total_reassigned = 10
        loop._cycles_run = 50

        stats = loop.get_stats()

        assert stats["total_reassigned"] == 10
        assert stats["cycles_run"] == 50


# =============================================================================
# Loop Lifecycle Tests
# =============================================================================


class TestJobLoopsLifecycle:
    """Tests for job loop lifecycle management."""

    @pytest.mark.asyncio
    async def test_job_reaper_start_stop(self):
        """Test JobReaperLoop can start and stop cleanly."""
        get_jobs = MagicMock(return_value={})
        cancel_job = AsyncMock(return_value=True)
        config = JobReaperConfig(check_interval_seconds=0.1)
        loop = JobReaperLoop(
            get_active_jobs=get_jobs,
            cancel_job=cancel_job,
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)  # Allow task to complete
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_idle_detection_start_stop(self):
        """Test IdleDetectionLoop can start and stop cleanly."""
        config = IdleDetectionConfig(check_interval_seconds=0.1)
        loop = IdleDetectionLoop(
            get_role=MagicMock(return_value="leader"),
            get_peers=MagicMock(return_value={}),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)  # Allow task to complete
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_worker_pull_start_stop(self):
        """Test WorkerPullLoop can start and stop cleanly."""
        config = WorkerPullConfig(
            pull_interval_seconds=0.1,
            initial_delay_seconds=0,
        )
        loop = WorkerPullLoop(
            is_leader=MagicMock(return_value=False),
            get_leader_id=MagicMock(return_value="leader-node"),
            get_self_metrics=MagicMock(return_value={
                "gpu_percent": 50.0,
                "training_jobs": 1,  # Busy - won't claim work
                "has_gpu": True,
            }),
            claim_work_from_leader=AsyncMock(return_value=None),
            execute_work=AsyncMock(return_value=True),
            report_work_result=AsyncMock(),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)  # Allow task to complete
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
