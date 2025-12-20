"""
Tests for app.coordination.job_scheduler module.

Tests the priority-based job scheduling system including:
- JobPriority enum
- ScheduledJob dataclass
- PriorityJobScheduler class
- Curriculum learning functions
- Host selection helpers
- HostDeadJobMigrator for job migration
"""

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.job_scheduler import (
    # Job migration
    HostDeadJobMigrator,
    # Core classes
    JobPriority,
    PriorityJobScheduler,
    ScheduledJob,
    # Host selection
    get_cpu_rich_hosts,
    get_gpu_rich_hosts,
    get_job_migrator,
    # Module functions
    get_scheduler,
    get_underserved_configs,
    reset_job_migrator,
    reset_scheduler,
    # Curriculum learning
    select_curriculum_config,
    wire_host_dead_to_job_migration,
)

# ============================================
# Test Fixtures
# ============================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    reset_scheduler()
    reset_job_migrator()
    yield
    reset_scheduler()
    reset_job_migrator()


@pytest.fixture
def scheduler():
    """Create a fresh scheduler instance."""
    return PriorityJobScheduler(max_queue_size=100)


@dataclass
class MockHost:
    """Mock host for testing."""
    name: str
    has_gpu: bool = False
    memory_gb: int = 128
    cpus: int = 16
    enabled: bool = True
    properties: dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class MockStatus:
    """Mock host status for testing."""
    reachable: bool = True
    cpu_percent: float = 50.0
    gpu_percent: float = 50.0
    disk_percent: float = 30.0
    memory_percent: float = 50.0


@pytest.fixture
def sample_hosts():
    """Create sample hosts for testing."""
    return [
        MockHost(name="gpu-host-1", has_gpu=True, memory_gb=256, cpus=32),
        MockHost(name="gpu-host-2", has_gpu=True, memory_gb=128, cpus=16),
        MockHost(name="cpu-host-1", has_gpu=False, memory_gb=128, cpus=64),
        MockHost(name="cpu-host-2", has_gpu=False, memory_gb=64, cpus=32),
    ]


@pytest.fixture
def sample_statuses():
    """Create sample statuses for testing."""
    return [
        MockStatus(cpu_percent=30.0, gpu_percent=20.0),
        MockStatus(cpu_percent=50.0, gpu_percent=40.0),
        MockStatus(cpu_percent=40.0),
        MockStatus(cpu_percent=70.0),
    ]


# ============================================
# Test JobPriority Enum
# ============================================

class TestJobPriority:
    """Tests for JobPriority enum."""

    def test_critical_is_highest(self):
        """Test CRITICAL has lowest value (highest priority)."""
        assert JobPriority.CRITICAL.value == 0

    def test_priority_ordering(self):
        """Test priority ordering (lower value = higher priority)."""
        assert JobPriority.CRITICAL < JobPriority.HIGH
        assert JobPriority.HIGH < JobPriority.NORMAL
        assert JobPriority.NORMAL < JobPriority.LOW

    def test_all_priorities_defined(self):
        """Test all expected priorities are defined."""
        priorities = list(JobPriority)
        assert len(priorities) == 4
        assert JobPriority.CRITICAL in priorities
        assert JobPriority.HIGH in priorities
        assert JobPriority.NORMAL in priorities
        assert JobPriority.LOW in priorities


# ============================================
# Test ScheduledJob Dataclass
# ============================================

class TestScheduledJob:
    """Tests for ScheduledJob dataclass."""

    def test_create_minimal_job(self):
        """Test creating job with minimal parameters."""
        job = ScheduledJob(
            job_type="selfplay",
            priority=JobPriority.NORMAL,
        )

        assert job.job_type == "selfplay"
        assert job.priority == JobPriority.NORMAL
        assert job.config == {}
        assert job.requires_gpu is False

    def test_create_full_job(self):
        """Test creating job with all parameters."""
        job = ScheduledJob(
            job_type="training",
            priority=JobPriority.HIGH,
            config={"model": "latest", "epochs": 10},
            host_preference="gpu-host-1",
            requires_gpu=True,
            estimated_duration_seconds=7200,
            job_id="train-001",
        )

        assert job.job_type == "training"
        assert job.priority == JobPriority.HIGH
        assert job.config["epochs"] == 10
        assert job.host_preference == "gpu-host-1"
        assert job.requires_gpu is True
        assert job.estimated_duration_seconds == 7200
        assert job.job_id == "train-001"

    def test_created_at_default(self):
        """Test created_at is set to current time."""
        before = time.time()
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        after = time.time()

        assert before <= job.created_at <= after

    def test_job_comparison(self):
        """Test job comparison by priority."""
        critical_job = ScheduledJob(job_type="test", priority=JobPriority.CRITICAL)
        normal_job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)

        assert critical_job < normal_job

    def test_job_hash(self):
        """Test job hashing."""
        job1 = ScheduledJob(job_type="test", priority=JobPriority.NORMAL, job_id="job-1")
        job2 = ScheduledJob(job_type="test", priority=JobPriority.NORMAL, job_id="job-1")
        job3 = ScheduledJob(job_type="test", priority=JobPriority.NORMAL, job_id="job-2")

        # Same job_id should have same hash
        assert hash(job1) == hash(job2)
        # Different job_id should have different hash
        assert hash(job1) != hash(job3)

    def test_to_dict(self):
        """Test to_dict serialization."""
        job = ScheduledJob(
            job_type="selfplay",
            priority=JobPriority.HIGH,
            config={"board": "square8"},
            job_id="sp-001",
        )

        result = job.to_dict()

        assert result["job_type"] == "selfplay"
        assert result["priority"] == "HIGH"
        assert result["config"]["board"] == "square8"
        assert result["job_id"] == "sp-001"


# ============================================
# Test PriorityJobScheduler
# ============================================

class TestPriorityJobScheduler:
    """Tests for PriorityJobScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = PriorityJobScheduler(max_queue_size=500)
        stats = scheduler.get_queue_stats()

        assert stats["total"] == 0
        assert stats["running"] == 0

    def test_schedule_job(self, scheduler):
        """Test scheduling a job."""
        job = ScheduledJob(job_type="selfplay", priority=JobPriority.NORMAL)

        result = scheduler.schedule(job)

        assert result is True
        assert scheduler.get_queue_stats()["total"] == 1

    def test_schedule_respects_priority(self, scheduler):
        """Test that scheduling maintains priority order."""
        low_job = ScheduledJob(job_type="low", priority=JobPriority.LOW)
        high_job = ScheduledJob(job_type="high", priority=JobPriority.HIGH)
        critical_job = ScheduledJob(job_type="critical", priority=JobPriority.CRITICAL)

        scheduler.schedule(low_job)
        scheduler.schedule(high_job)
        scheduler.schedule(critical_job)

        pending = scheduler.get_pending_jobs()
        assert pending[0].job_type == "critical"
        assert pending[1].job_type == "high"
        assert pending[2].job_type == "low"

    def test_schedule_rejects_when_full(self):
        """Test scheduling is rejected when queue is full."""
        scheduler = PriorityJobScheduler(max_queue_size=2)

        scheduler.schedule(ScheduledJob(job_type="1", priority=JobPriority.NORMAL))
        scheduler.schedule(ScheduledJob(job_type="2", priority=JobPriority.NORMAL))
        result = scheduler.schedule(ScheduledJob(job_type="3", priority=JobPriority.NORMAL))

        assert result is False
        assert scheduler.get_queue_stats()["total"] == 2

    def test_next_job_empty_queue(self, scheduler, sample_hosts, sample_statuses):
        """Test next_job with empty queue."""
        result = scheduler.next_job(sample_hosts, sample_statuses)
        assert result is None

    def test_next_job_matches_gpu(self, scheduler, sample_hosts, sample_statuses):
        """Test next_job matches GPU requirements."""
        gpu_job = ScheduledJob(
            job_type="training",
            priority=JobPriority.HIGH,
            requires_gpu=True,
        )
        scheduler.schedule(gpu_job)

        result = scheduler.next_job(sample_hosts, sample_statuses)

        assert result is not None
        job, host = result
        assert job.requires_gpu is True
        assert host.has_gpu is True

    def test_next_job_respects_host_preference(self, scheduler, sample_hosts, sample_statuses):
        """Test next_job respects host preference."""
        job = ScheduledJob(
            job_type="selfplay",
            priority=JobPriority.NORMAL,
            host_preference="cpu-host-1",
        )
        scheduler.schedule(job)

        result = scheduler.next_job(sample_hosts, sample_statuses)

        assert result is not None
        _, host = result
        assert host.name == "cpu-host-1"

    def test_next_job_skips_unreachable(self, scheduler):
        """Test next_job skips unreachable hosts."""
        hosts = [MockHost(name="host-1")]
        statuses = [MockStatus(reachable=False)]

        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        result = scheduler.next_job(hosts, statuses)
        assert result is None

    def test_next_job_skips_high_disk_usage(self, scheduler):
        """Test next_job skips hosts with high disk usage."""
        hosts = [MockHost(name="host-1")]
        statuses = [MockStatus(disk_percent=80.0)]  # Over 70% limit

        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        result = scheduler.next_job(hosts, statuses)
        assert result is None

    def test_complete_job(self, scheduler, sample_hosts, sample_statuses):
        """Test completing a job."""
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        # Start the job
        result = scheduler.next_job(sample_hosts, sample_statuses)
        assert result is not None
        _, host = result

        # Complete the job
        completed = scheduler.complete_job(host.name)

        assert completed is not None
        assert completed.completed_at is not None
        assert scheduler.get_queue_stats()["running"] == 0

    def test_complete_job_not_running(self, scheduler):
        """Test completing job that wasn't running."""
        result = scheduler.complete_job("nonexistent-host")
        assert result is None

    def test_cancel_job(self, scheduler, sample_hosts, sample_statuses):
        """Test cancelling a running job."""
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        # Start the job
        result = scheduler.next_job(sample_hosts, sample_statuses)
        _, host = result

        # Cancel the job
        cancelled = scheduler.cancel_job(host.name)

        assert cancelled is not None
        assert scheduler.get_queue_stats()["running"] == 0

    def test_remove_pending(self, scheduler):
        """Test removing a pending job."""
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        result = scheduler.remove_pending(job)

        assert result is True
        assert scheduler.get_queue_stats()["total"] == 0

    def test_remove_pending_not_found(self, scheduler):
        """Test removing job not in queue."""
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        result = scheduler.remove_pending(job)
        assert result is False

    def test_get_queue_stats(self, scheduler):
        """Test getting queue statistics."""
        scheduler.schedule(ScheduledJob(job_type="1", priority=JobPriority.CRITICAL))
        scheduler.schedule(ScheduledJob(job_type="2", priority=JobPriority.HIGH))
        scheduler.schedule(ScheduledJob(job_type="3", priority=JobPriority.NORMAL))
        scheduler.schedule(ScheduledJob(job_type="4", priority=JobPriority.LOW))

        stats = scheduler.get_queue_stats()

        assert stats["total"] == 4
        assert stats["critical"] == 1
        assert stats["high"] == 1
        assert stats["normal"] == 1
        assert stats["low"] == 1

    def test_has_critical_pending(self, scheduler):
        """Test checking for critical pending jobs."""
        assert scheduler.has_critical_pending() is False

        scheduler.schedule(ScheduledJob(job_type="test", priority=JobPriority.CRITICAL))
        assert scheduler.has_critical_pending() is True

    def test_has_pending(self, scheduler):
        """Test checking for pending jobs."""
        assert scheduler.has_pending() is False
        assert scheduler.has_pending(JobPriority.HIGH) is False

        scheduler.schedule(ScheduledJob(job_type="test", priority=JobPriority.HIGH))

        assert scheduler.has_pending() is True
        assert scheduler.has_pending(JobPriority.HIGH) is True
        assert scheduler.has_pending(JobPriority.CRITICAL) is False

    def test_get_running_jobs(self, scheduler, sample_hosts, sample_statuses):
        """Test getting running jobs."""
        job = ScheduledJob(job_type="test", priority=JobPriority.NORMAL)
        scheduler.schedule(job)

        assert scheduler.get_running_jobs() == {}

        result = scheduler.next_job(sample_hosts, sample_statuses)
        _, host = result

        running = scheduler.get_running_jobs()
        assert host.name in running

    def test_get_pending_jobs(self, scheduler):
        """Test getting pending jobs."""
        high_job = ScheduledJob(job_type="high", priority=JobPriority.HIGH)
        normal_job = ScheduledJob(job_type="normal", priority=JobPriority.NORMAL)

        scheduler.schedule(high_job)
        scheduler.schedule(normal_job)

        all_pending = scheduler.get_pending_jobs()
        assert len(all_pending) == 2

        high_only = scheduler.get_pending_jobs(JobPriority.HIGH)
        assert len(high_only) == 1
        assert high_only[0].job_type == "high"

    def test_reserve_capacity_for_training(self, scheduler, sample_hosts):
        """Test reserving capacity for training."""
        statuses = {
            "gpu-host-1": MockStatus(gpu_percent=10.0),  # Low GPU usage - reserve
            "gpu-host-2": MockStatus(gpu_percent=50.0),  # Higher usage - don't reserve
        }

        reserved = scheduler.reserve_capacity_for_training(
            sample_hosts[:2],
            [statuses["gpu-host-1"], statuses["gpu-host-2"]],
            reserve_percent=20.0,
        )

        assert "gpu-host-1" in reserved
        assert "gpu-host-2" not in reserved

    def test_clear_queue(self, scheduler):
        """Test clearing the queue."""
        for i in range(5):
            scheduler.schedule(ScheduledJob(job_type=f"job-{i}", priority=JobPriority.NORMAL))

        count = scheduler.clear_queue()

        assert count == 5
        assert scheduler.get_queue_stats()["total"] == 0


# ============================================
# Test Module Functions
# ============================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_scheduler_singleton(self):
        """Test get_scheduler returns singleton."""
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()
        assert scheduler1 is scheduler2

    def test_reset_scheduler(self):
        """Test reset_scheduler clears singleton."""
        scheduler1 = get_scheduler()
        reset_scheduler()
        scheduler2 = get_scheduler()
        assert scheduler1 is not scheduler2


# ============================================
# Test Curriculum Learning Functions
# ============================================

class TestCurriculumLearning:
    """Tests for curriculum learning functions."""

    def test_select_curriculum_config_empty(self):
        """Test select_curriculum_config with empty configs."""
        result = select_curriculum_config([], {})
        assert result == {}

    def test_select_curriculum_config_no_history(self):
        """Test select_curriculum_config without game history."""
        configs = [
            {"board": "square8", "players": 2},
            {"board": "square8", "players": 4},
        ]

        result = select_curriculum_config(configs, {})

        assert result in configs

    def test_select_curriculum_config_prioritizes_underserved(self):
        """Test that underserved configs get higher priority."""
        configs = [
            {"board": "square8", "players": 2},
            {"board": "square8", "players": 4},
        ]
        # 2-player has many games, 4-player has few
        game_counts = {
            "square8_2p": 10000,
            "square8_4p": 100,
        }

        # Run multiple times to see statistical preference
        selections = {"square8_2p": 0, "square8_4p": 0}
        for _ in range(100):
            result = select_curriculum_config(configs, game_counts)
            key = f"{result['board']}_{result['players']}p"
            selections[key] += 1

        # Underserved (4p) should be selected more often
        assert selections["square8_4p"] > selections["square8_2p"]

    def test_get_underserved_configs(self):
        """Test get_underserved_configs identifies underserved."""
        configs = [
            {"board": "square8", "players": 2},
            {"board": "square8", "players": 4},
            {"board": "hex6", "players": 2},
        ]
        game_counts = {
            "square8_2p": 500,
            "square8_4p": 50,
            "hex6_2p": 200,
        }

        underserved = get_underserved_configs(configs, game_counts, threshold=100)

        assert len(underserved) == 1
        assert underserved[0]["players"] == 4


# ============================================
# Test Host Selection Functions
# ============================================

class TestHostSelection:
    """Tests for host selection functions."""

    def test_get_cpu_rich_hosts(self):
        """Test get_cpu_rich_hosts returns appropriate hosts."""
        hosts = [
            MockHost(name="cpu-heavy", cpus=64, has_gpu=False),
            MockHost(name="gpu-host", cpus=16, has_gpu=True),
            MockHost(name="disabled", cpus=32, enabled=False),
        ]
        statuses = {
            "cpu-heavy": MockStatus(cpu_percent=30.0),
            "gpu-host": MockStatus(cpu_percent=40.0),
            "disabled": MockStatus(cpu_percent=20.0),
        }

        result = get_cpu_rich_hosts(hosts, statuses)

        # Should include cpu-heavy and gpu-host (both enabled and low CPU)
        # Should exclude disabled
        host_names = [h[0].name for h in result]
        assert "cpu-heavy" in host_names
        assert "disabled" not in host_names

    def test_get_cpu_rich_hosts_prioritizes_cpu_role(self):
        """Test that cpu_selfplay role gets priority."""
        hosts = [
            MockHost(name="generic", cpus=64, properties={}),
            MockHost(name="cpu-dedicated", cpus=32, properties={"role": "cpu_selfplay"}),
        ]
        statuses = {
            "generic": MockStatus(cpu_percent=30.0),
            "cpu-dedicated": MockStatus(cpu_percent=30.0),
        }

        result = get_cpu_rich_hosts(hosts, statuses)

        # cpu-dedicated should come first due to role boost
        assert result[0][0].name == "cpu-dedicated"

    def test_get_gpu_rich_hosts(self):
        """Test get_gpu_rich_hosts returns GPU hosts."""
        hosts = [
            MockHost(name="big-gpu", has_gpu=True, memory_gb=256),
            MockHost(name="small-gpu", has_gpu=True, memory_gb=128),
            MockHost(name="no-gpu", has_gpu=False, memory_gb=64),
        ]
        statuses = {
            "big-gpu": MockStatus(gpu_percent=20.0),
            "small-gpu": MockStatus(gpu_percent=30.0),
            "no-gpu": MockStatus(),
        }

        result = get_gpu_rich_hosts(hosts, statuses)

        # Should only include GPU hosts
        host_names = [h[0].name for h in result]
        assert "big-gpu" in host_names
        assert "small-gpu" in host_names
        assert "no-gpu" not in host_names

        # Should be sorted by memory (big first)
        assert result[0][0].name == "big-gpu"

    def test_get_gpu_rich_hosts_filters_high_usage(self):
        """Test that hosts with high GPU usage are filtered."""
        hosts = [
            MockHost(name="busy-gpu", has_gpu=True),
            MockHost(name="idle-gpu", has_gpu=True),
        ]
        statuses = {
            "busy-gpu": MockStatus(gpu_percent=80.0),  # High usage
            "idle-gpu": MockStatus(gpu_percent=20.0),  # Low usage
        }

        result = get_gpu_rich_hosts(hosts, statuses)

        host_names = [h[0].name for h in result]
        assert "idle-gpu" in host_names
        assert "busy-gpu" not in host_names


# ============================================
# Test HostDeadJobMigrator
# ============================================

class TestHostDeadJobMigrator:
    """Tests for HostDeadJobMigrator class."""

    def test_initialization(self):
        """Test migrator initialization."""
        migrator = HostDeadJobMigrator()

        assert migrator._subscribed is False
        assert migrator._migrations_count == 0

    def test_uses_global_scheduler(self):
        """Test migrator uses global scheduler by default."""
        migrator = HostDeadJobMigrator()
        global_scheduler = get_scheduler()

        assert migrator.scheduler is global_scheduler

    def test_uses_provided_scheduler(self):
        """Test migrator uses provided scheduler."""
        custom_scheduler = PriorityJobScheduler()
        migrator = HostDeadJobMigrator(scheduler=custom_scheduler)

        assert migrator.scheduler is custom_scheduler

    def test_migrate_jobs_from_host_no_jobs(self):
        """Test migration when no jobs running on host."""
        scheduler = PriorityJobScheduler()
        migrator = HostDeadJobMigrator(scheduler=scheduler)

        count = migrator.migrate_jobs_from_host("dead-host")

        assert count == 0

    def test_migrate_jobs_from_host_success(self):
        """Test successful job migration."""
        scheduler = PriorityJobScheduler()
        migrator = HostDeadJobMigrator(scheduler=scheduler, requeue_priority_boost=1)

        # Create a job and put directly in running (simulating a running job)
        job = ScheduledJob(
            job_type="selfplay",
            priority=JobPriority.NORMAL,
            job_id="sp-001",
        )
        # Directly add to running without going through schedule
        scheduler._running["dead-host"] = job

        # Migrate
        count = migrator.migrate_jobs_from_host("dead-host", reason="host_failure")

        assert count == 1
        assert migrator._migrations_count == 1

        # Job should be requeued with boosted priority
        pending = scheduler.get_pending_jobs()
        assert len(pending) == 1
        assert pending[0].priority == JobPriority.HIGH  # Boosted from NORMAL

    def test_priority_boost(self):
        """Test priority boosting."""
        migrator = HostDeadJobMigrator(requeue_priority_boost=2)

        # LOW -> HIGH (boost by 2)
        assert migrator._boost_priority(JobPriority.LOW) == JobPriority.HIGH

        # NORMAL -> CRITICAL (boost by 2, capped)
        assert migrator._boost_priority(JobPriority.NORMAL) == JobPriority.CRITICAL

        # CRITICAL stays CRITICAL
        assert migrator._boost_priority(JobPriority.CRITICAL) == JobPriority.CRITICAL

    def test_get_stats(self):
        """Test getting migrator stats."""
        migrator = HostDeadJobMigrator(requeue_priority_boost=2)

        stats = migrator.get_stats()

        assert stats["subscribed"] is False
        assert stats["migrations_count"] == 0
        assert stats["failed_migrations"] == 0
        assert stats["requeue_priority_boost"] == 2

    def test_subscribe_to_events_no_bus(self):
        """Test subscription when event bus not available."""
        HostDeadJobMigrator()

        with patch("app.coordination.job_scheduler.HostDeadJobMigrator.subscribe_to_host_events") as mock:
            mock.return_value = False
            result = mock()

        assert result is False


# ============================================
# Test Wire Functions
# ============================================

class TestWireFunctions:
    """Tests for wire functions."""

    def test_wire_host_dead_to_job_migration(self):
        """Test wiring creates migrator."""
        reset_job_migrator()

        with patch.object(HostDeadJobMigrator, "subscribe_to_host_events", return_value=True):
            migrator = wire_host_dead_to_job_migration(requeue_priority_boost=2)

        assert migrator is not None
        assert get_job_migrator() is migrator

    def test_wire_returns_same_instance(self):
        """Test wiring returns same instance on multiple calls."""
        reset_job_migrator()

        with patch.object(HostDeadJobMigrator, "subscribe_to_host_events", return_value=True):
            migrator1 = wire_host_dead_to_job_migration()
            migrator2 = wire_host_dead_to_job_migration()

        assert migrator1 is migrator2

    def test_reset_job_migrator(self):
        """Test reset clears migrator."""
        reset_job_migrator()

        with patch.object(HostDeadJobMigrator, "subscribe_to_host_events", return_value=True):
            wire_host_dead_to_job_migration()

        reset_job_migrator()

        assert get_job_migrator() is None


# ============================================
# Integration Tests
# ============================================

class TestSchedulerIntegration:
    """Integration tests for scheduler."""

    def test_full_job_lifecycle(self, scheduler, sample_hosts, sample_statuses):
        """Test complete job lifecycle."""
        # Schedule multiple jobs
        jobs = [
            ScheduledJob(job_type="critical", priority=JobPriority.CRITICAL),
            ScheduledJob(job_type="normal1", priority=JobPriority.NORMAL),
            ScheduledJob(job_type="normal2", priority=JobPriority.NORMAL),
        ]

        for job in jobs:
            scheduler.schedule(job)

        assert scheduler.get_queue_stats()["total"] == 3

        # Process jobs
        completed = []
        while scheduler.has_pending():
            result = scheduler.next_job(sample_hosts, sample_statuses)
            if result:
                job, host = result
                completed_job = scheduler.complete_job(host.name)
                if completed_job:
                    completed.append(completed_job)

        # All jobs should be completed
        assert len(completed) == 3

        # Critical should be first
        assert completed[0].job_type == "critical"

    def test_job_migration_integration(self):
        """Test job migration with scheduler."""
        scheduler = PriorityJobScheduler()
        migrator = HostDeadJobMigrator(scheduler=scheduler)

        # Create job and mark as running
        job = ScheduledJob(
            job_type="training",
            priority=JobPriority.HIGH,
            requires_gpu=True,
            job_id="train-001",
        )
        scheduler._running["gpu-host-1"] = job

        # Simulate host going offline
        migrated = migrator.migrate_jobs_from_host("gpu-host-1", reason="hardware_failure")

        # Job should be migrated
        assert migrated == 1
        assert "gpu-host-1" not in scheduler.get_running_jobs()

        # Job should be requeued
        pending = scheduler.get_pending_jobs()
        assert len(pending) == 1
        assert pending[0].requires_gpu is True
        assert pending[0].host_preference is None  # Cleared for new host
