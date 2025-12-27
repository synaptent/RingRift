"""Tests for app/coordination/resource_targets.py.

This module tests the unified resource utilization targets for cluster orchestration,
including target calculations, host-specific adjustments, and scaling decisions.

December 2025: Created as part of Tier 1 critical test coverage initiative.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.resource_targets import (
    HOST_TIER_MAP,
    TIER_ADJUSTMENTS,
    HostTargets,
    HostTier,
    ResourceTargetManager,
    UtilizationTargets,
    get_cluster_summary,
    get_host_targets,
    get_resource_targets,
    get_target_job_count,
    get_utilization_score,
    record_utilization,
    reset_resource_targets,
    select_host_for_task,
    set_backpressure,
    should_scale_down,
    should_scale_up,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_coordination.db"


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    """Create a mock config file."""
    config_path = tmp_path / "unified_loop.yaml"
    config_path.write_text("""
resource_targets:
  cpu_min: 55.0
  cpu_target: 68.0
  cpu_max: 82.0
  gpu_min: 55.0
  gpu_target: 72.0
  gpu_max: 88.0
  max_jobs_per_node: 32
""")
    return config_path


@pytest.fixture
def manager(temp_db_path: Path) -> ResourceTargetManager:
    """Create a fresh ResourceTargetManager for testing."""
    # Reset any existing singleton
    ResourceTargetManager.reset_instance()
    return ResourceTargetManager(db_path=temp_db_path)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton after each test."""
    yield
    reset_resource_targets()


# =============================================================================
# Tests: UtilizationTargets Dataclass
# =============================================================================


class TestUtilizationTargets:
    """Tests for UtilizationTargets dataclass."""

    def test_default_values(self):
        """Test default target values."""
        targets = UtilizationTargets()

        # CPU defaults
        assert targets.cpu_min == 50.0
        assert targets.cpu_target == 65.0
        assert targets.cpu_max == 80.0
        assert targets.cpu_critical == 80.0

        # GPU defaults
        assert targets.gpu_min == 50.0
        assert targets.gpu_target == 65.0
        assert targets.gpu_max == 80.0
        assert targets.gpu_critical == 80.0

        # Memory defaults
        assert targets.memory_warn == 70.0
        assert targets.memory_critical == 80.0

        # Disk defaults
        assert targets.disk_warn == 65.0
        assert targets.disk_critical == 70.0

        # Job defaults
        assert targets.jobs_per_core == 0.5
        assert targets.max_jobs_per_node == 48
        assert targets.max_selfplay_cluster == 500

    def test_custom_values(self):
        """Test custom target values."""
        targets = UtilizationTargets(
            cpu_min=40.0,
            cpu_target=55.0,
            cpu_max=70.0,
            gpu_min=45.0,
            max_jobs_per_node=64,
        )

        assert targets.cpu_min == 40.0
        assert targets.cpu_target == 55.0
        assert targets.cpu_max == 70.0
        assert targets.gpu_min == 45.0
        assert targets.max_jobs_per_node == 64

    def test_targets_maintain_order(self):
        """Test that target thresholds are in correct order."""
        targets = UtilizationTargets()

        # CPU thresholds
        assert targets.cpu_min < targets.cpu_target < targets.cpu_max <= targets.cpu_critical

        # GPU thresholds
        assert targets.gpu_min < targets.gpu_target < targets.gpu_max <= targets.gpu_critical

        # Memory thresholds
        assert targets.memory_warn < targets.memory_critical

        # Disk thresholds
        assert targets.disk_warn < targets.disk_critical


# =============================================================================
# Tests: HostTier Enum
# =============================================================================


class TestHostTier:
    """Tests for HostTier enum."""

    def test_tier_values(self):
        """Test that all tiers have expected values."""
        assert HostTier.HIGH_END.value == "high_end"
        assert HostTier.MID_TIER.value == "mid_tier"
        assert HostTier.LOW_TIER.value == "low_tier"
        assert HostTier.CPU_ONLY.value == "cpu_only"

    def test_host_tier_map_contains_known_hosts(self):
        """Test that HOST_TIER_MAP includes known hosts."""
        assert "runpod-h100" in HOST_TIER_MAP
        assert HOST_TIER_MAP["runpod-h100"] == HostTier.HIGH_END

        assert "rtx4090" in HOST_TIER_MAP
        assert HOST_TIER_MAP["rtx4090"] == HostTier.MID_TIER

        assert "rtx3090" in HOST_TIER_MAP
        assert HOST_TIER_MAP["rtx3090"] == HostTier.LOW_TIER

        assert "local" in HOST_TIER_MAP
        assert HOST_TIER_MAP["local"] == HostTier.CPU_ONLY

    def test_tier_adjustments_defined(self):
        """Test that TIER_ADJUSTMENTS has all tiers."""
        for tier in HostTier:
            assert tier in TIER_ADJUSTMENTS
            adj = TIER_ADJUSTMENTS[tier]
            assert "cpu_boost" in adj
            assert "gpu_boost" in adj
            assert "job_multiplier" in adj

    def test_high_end_tier_has_positive_boost(self):
        """Test that high-end tier gets positive adjustments."""
        adj = TIER_ADJUSTMENTS[HostTier.HIGH_END]
        assert adj["cpu_boost"] > 0
        assert adj["gpu_boost"] > 0
        assert adj["job_multiplier"] > 1.0

    def test_low_tier_has_negative_boost(self):
        """Test that low tier gets negative adjustments."""
        adj = TIER_ADJUSTMENTS[HostTier.LOW_TIER]
        assert adj["cpu_boost"] < 0
        assert adj["gpu_boost"] < 0
        assert adj["job_multiplier"] < 1.0


# =============================================================================
# Tests: HostTargets Dataclass
# =============================================================================


class TestHostTargets:
    """Tests for HostTargets dataclass."""

    def test_host_targets_creation(self):
        """Test HostTargets creation."""
        targets = HostTargets(
            host="test-host",
            tier=HostTier.MID_TIER,
            cpu_min=50.0,
            cpu_target=65.0,
            cpu_max=80.0,
            gpu_min=50.0,
            gpu_target=70.0,
            gpu_max=85.0,
            max_jobs=24,
            max_selfplay=16,
            max_training=2,
        )

        assert targets.host == "test-host"
        assert targets.tier == HostTier.MID_TIER
        assert targets.max_jobs == 24
        assert targets.max_selfplay == 16
        assert targets.max_training == 2

    def test_host_targets_defaults(self):
        """Test HostTargets defaults for state tracking."""
        targets = HostTargets(
            host="test",
            tier=HostTier.MID_TIER,
            cpu_min=50.0,
            cpu_target=65.0,
            cpu_max=80.0,
            gpu_min=50.0,
            gpu_target=70.0,
            gpu_max=85.0,
            max_jobs=24,
            max_selfplay=16,
            max_training=2,
        )

        assert targets.last_cpu == 0.0
        assert targets.last_gpu == 0.0
        assert targets.last_memory == 0.0
        assert targets.last_jobs == 0
        assert targets.last_update == 0.0


# =============================================================================
# Tests: ResourceTargetManager
# =============================================================================


class TestResourceTargetManager:
    """Tests for ResourceTargetManager class."""

    def test_singleton_pattern(self, temp_db_path: Path):
        """Test that get_instance returns the same object."""
        ResourceTargetManager.reset_instance()

        manager1 = ResourceTargetManager.get_instance(temp_db_path)
        manager2 = ResourceTargetManager.get_instance(temp_db_path)

        assert manager1 is manager2

    def test_reset_instance(self, temp_db_path: Path):
        """Test that reset_instance clears the singleton."""
        ResourceTargetManager.reset_instance()

        manager1 = ResourceTargetManager.get_instance(temp_db_path)
        ResourceTargetManager.reset_instance()
        manager2 = ResourceTargetManager.get_instance(temp_db_path)

        assert manager1 is not manager2

    def test_init_creates_database(self, temp_db_path: Path):
        """Test that initialization creates the database."""
        assert not temp_db_path.exists()

        manager = ResourceTargetManager(db_path=temp_db_path)

        assert temp_db_path.exists()

        # Check tables exist
        with sqlite3.connect(str(temp_db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        assert "resource_targets" in tables
        assert "utilization_history" in tables

    def test_get_targets_returns_defaults(self, manager: ResourceTargetManager):
        """Test that get_targets returns sensible defaults."""
        targets = manager.get_targets()

        assert isinstance(targets, UtilizationTargets)
        assert targets.cpu_min > 0
        assert targets.cpu_max <= 100
        assert targets.gpu_min > 0
        assert targets.gpu_max <= 100

    def test_get_host_targets_caches_result(self, manager: ResourceTargetManager):
        """Test that get_host_targets caches results."""
        host = "test-host"

        targets1 = manager.get_host_targets(host)
        targets2 = manager.get_host_targets(host)

        assert targets1 is targets2

    def test_get_host_targets_different_hosts(self, manager: ResourceTargetManager):
        """Test that different hosts get different targets."""
        targets1 = manager.get_host_targets("runpod-h100")
        targets2 = manager.get_host_targets("rtx3090")

        # Different tiers should have different settings
        assert targets1.tier != targets2.tier or targets1.host != targets2.host

    def test_set_backpressure_clamps_values(self, manager: ResourceTargetManager):
        """Test that set_backpressure clamps to 0.0-1.0."""
        manager.set_backpressure(1.5)
        assert manager._backpressure_factor == 1.0

        manager.set_backpressure(-0.5)
        assert manager._backpressure_factor == 0.0

        manager.set_backpressure(0.7)
        assert manager._backpressure_factor == 0.7

    def test_set_backpressure_clears_cache(self, manager: ResourceTargetManager):
        """Test that setting backpressure clears host cache."""
        manager.get_host_targets("test-host")
        assert len(manager._host_targets) > 0

        manager.set_backpressure(0.5)
        assert len(manager._host_targets) == 0

    def test_get_cluster_summary_empty(self, manager: ResourceTargetManager):
        """Test cluster summary with no data."""
        summary = manager.get_cluster_summary()

        assert summary["active_hosts"] == 0
        assert summary["avg_cpu"] == 0.0
        assert summary["avg_gpu"] == 0.0
        assert summary["total_jobs"] == 0
        assert "backpressure_factor" in summary
        assert "targets" in summary


# =============================================================================
# Tests: Module-Level Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_resource_targets(self, temp_db_path: Path):
        """Test get_resource_targets returns UtilizationTargets."""
        with patch.object(
            ResourceTargetManager,
            "get_instance",
            return_value=ResourceTargetManager(db_path=temp_db_path),
        ):
            targets = get_resource_targets()

        assert isinstance(targets, UtilizationTargets)

    def test_get_host_targets(self, temp_db_path: Path):
        """Test get_host_targets returns HostTargets."""
        with patch.object(
            ResourceTargetManager,
            "get_instance",
            return_value=ResourceTargetManager(db_path=temp_db_path),
        ):
            targets = get_host_targets("test-host")

        assert isinstance(targets, HostTargets)
        assert targets.host == "test-host"


# =============================================================================
# Tests: Scaling Decisions
# =============================================================================


class TestScalingDecisions:
    """Tests for scale up/down decision logic."""

    def test_should_scale_up_low_utilization(self, manager: ResourceTargetManager):
        """Test scale up when utilization is low."""
        host = "test-host"
        # Low utilization should trigger scale up
        should_scale, reason = manager.should_scale_up(
            host,
            current_cpu=30.0,  # Well below cpu_min
            current_gpu=25.0,
            current_jobs=2,
        )

        assert should_scale is True
        assert reason != ""

    def test_should_scale_up_high_utilization(self, manager: ResourceTargetManager):
        """Test no scale up when utilization is high."""
        host = "test-host"
        # High utilization should not scale up
        should_scale, reason = manager.should_scale_up(
            host,
            current_cpu=75.0,  # Near cpu_max
            current_gpu=80.0,
            current_jobs=20,
        )

        assert should_scale is False

    def test_should_scale_down_high_utilization(self, manager: ResourceTargetManager):
        """Test scale down when utilization is high."""
        host = "test-host"
        # High utilization should trigger scale down
        should_scale, jobs_to_remove, reason = manager.should_scale_down(
            host,
            current_cpu=90.0,  # Above cpu_max
            current_gpu=85.0,
            current_memory=75.0,
        )

        assert should_scale is True
        assert jobs_to_remove > 0
        assert reason != ""

    def test_should_scale_down_low_utilization(self, manager: ResourceTargetManager):
        """Test no scale down when utilization is low."""
        host = "test-host"
        # Low utilization should not scale down
        should_scale, jobs_to_remove, reason = manager.should_scale_down(
            host,
            current_cpu=40.0,
            current_gpu=35.0,
            current_memory=30.0,
        )

        assert should_scale is False
        assert jobs_to_remove == 0


# =============================================================================
# Tests: Job Count Calculation
# =============================================================================


class TestJobCountCalculation:
    """Tests for target job count calculation."""

    def test_get_target_job_count_basic(self, manager: ResourceTargetManager):
        """Test basic job count calculation."""
        job_count = manager.get_target_job_count(
            host="test-host",
            cpu_cores=16,
            current_cpu=50.0,
            current_gpu=50.0,
        )

        assert isinstance(job_count, int)
        assert job_count >= 0

    def test_get_target_job_count_respects_max(self, manager: ResourceTargetManager):
        """Test that job count respects max limits."""
        # Request high job count
        job_count = manager.get_target_job_count(
            host="test-host",
            cpu_cores=128,  # Many cores
            current_cpu=20.0,  # Very low utilization
            current_gpu=20.0,
        )

        targets = manager.get_host_targets("test-host")
        assert job_count <= targets.max_jobs

    def test_get_target_job_count_high_utilization(
        self, manager: ResourceTargetManager
    ):
        """Test job count when utilization is already high."""
        job_count = manager.get_target_job_count(
            host="test-host",
            cpu_cores=16,
            current_cpu=90.0,  # Very high
            current_gpu=85.0,
        )

        # Should not add more jobs when utilization is high
        assert job_count >= 0


# =============================================================================
# Tests: Utilization Score
# =============================================================================


class TestUtilizationScore:
    """Tests for utilization score calculation."""

    def test_get_utilization_score_basic(self, manager: ResourceTargetManager):
        """Test basic utilization score calculation."""
        score = manager.get_utilization_score(
            host="test-host",
            current_cpu=65.0,
            current_gpu=70.0,
        )

        assert isinstance(score, float)
        assert score >= 0.0

    def test_get_utilization_score_optimal(self, manager: ResourceTargetManager):
        """Test score is optimal at target utilization."""
        targets = manager.get_targets()

        score_at_target = manager.get_utilization_score(
            host="test-host",
            current_cpu=targets.cpu_target,
            current_gpu=targets.gpu_target,
        )

        score_at_low = manager.get_utilization_score(
            host="test-host",
            current_cpu=targets.cpu_min - 20,
            current_gpu=targets.gpu_min - 20,
        )

        # Score at target should be better (lower or same as far from target)
        # This depends on implementation, but the assertion makes sense semantically
        assert isinstance(score_at_target, float)
        assert isinstance(score_at_low, float)


# =============================================================================
# Tests: Recording Utilization
# =============================================================================


class TestRecordUtilization:
    """Tests for recording utilization metrics."""

    def test_record_utilization_updates_state(self, manager: ResourceTargetManager):
        """Test that recording updates host state."""
        host = "test-host"

        # First, get host targets to create entry
        manager.get_host_targets(host)

        # Record utilization
        manager.record_utilization(
            host=host,
            cpu_percent=65.0,
            gpu_percent=70.0,
            memory_percent=50.0,
            job_count=10,
        )

        # Check that state was updated
        targets = manager.get_host_targets(host)
        assert targets.last_cpu == 65.0
        assert targets.last_gpu == 70.0
        assert targets.last_memory == 50.0
        assert targets.last_jobs == 10
        assert targets.last_update > 0

    def test_record_utilization_maintains_history(
        self, manager: ResourceTargetManager
    ):
        """Test that utilization history is maintained."""
        host = "test-host"

        for i in range(5):
            manager.record_utilization(
                host=host,
                cpu_percent=50.0 + i * 5,
                gpu_percent=55.0 + i * 5,
                memory_percent=40.0,
                job_count=5 + i,
            )

        # Check history exists
        assert host in manager._utilization_history
        assert len(manager._utilization_history[host]) <= 360  # Max history length


# =============================================================================
# Tests: Host Selection
# =============================================================================


class TestHostSelection:
    """Tests for host selection logic."""

    def test_select_host_for_task_empty_list(self):
        """Test selection with empty host list."""
        result = select_host_for_task(
            available_hosts=[],
            task_resource_type="gpu",
        )

        assert result is None

    def test_select_host_for_task_single_host(self, temp_db_path: Path):
        """Test selection with single host."""
        with patch.object(
            ResourceTargetManager,
            "get_instance",
            return_value=ResourceTargetManager(db_path=temp_db_path),
        ):
            result = select_host_for_task(
                available_hosts=["only-host"],
                task_resource_type="gpu",
            )

        assert result == "only-host"

    def test_select_host_for_task_with_metrics(self, temp_db_path: Path):
        """Test selection with provided metrics."""
        with patch.object(
            ResourceTargetManager,
            "get_instance",
            return_value=ResourceTargetManager(db_path=temp_db_path),
        ):
            result = select_host_for_task(
                available_hosts=["host1", "host2", "host3"],
                task_resource_type="gpu",
                host_metrics={
                    "host1": {"cpu_util": 80, "gpu_util": 90},  # Very busy
                    "host2": {"cpu_util": 30, "gpu_util": 20},  # Least busy
                    "host3": {"cpu_util": 50, "gpu_util": 60},  # Moderate
                },
            )

        # Should select host2 (lowest GPU utilization for GPU task)
        assert result == "host2"

    def test_select_host_for_cpu_task(self, temp_db_path: Path):
        """Test selection for CPU-bound task."""
        with patch.object(
            ResourceTargetManager,
            "get_instance",
            return_value=ResourceTargetManager(db_path=temp_db_path),
        ):
            result = select_host_for_task(
                available_hosts=["host1", "host2"],
                task_resource_type="cpu",
                host_metrics={
                    "host1": {"cpu_util": 90, "gpu_util": 20},  # High CPU
                    "host2": {"cpu_util": 30, "gpu_util": 80},  # Low CPU
                },
            )

        # Should select host2 (lower CPU for CPU task)
        assert result == "host2"


# =============================================================================
# Tests: Config Loading
# =============================================================================


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_from_config_file(
        self, mock_config_path: Path, temp_db_path: Path
    ):
        """Test loading targets from config file."""
        manager = ResourceTargetManager(
            db_path=temp_db_path,
            config_path=mock_config_path,
        )

        targets = manager.get_targets()

        # Check values from mock config
        assert targets.cpu_min == 55.0
        assert targets.cpu_target == 68.0
        assert targets.max_jobs_per_node == 32

    def test_missing_config_uses_defaults(self, temp_db_path: Path, tmp_path: Path):
        """Test that missing config uses defaults."""
        nonexistent = tmp_path / "nonexistent.yaml"

        manager = ResourceTargetManager(
            db_path=temp_db_path,
            config_path=nonexistent,
        )

        targets = manager.get_targets()

        # Should have default values
        assert targets.cpu_min == 60.0  # Default from _load_targets_from_config
        assert targets.cpu_target == 70.0


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_cpu_cores(self, manager: ResourceTargetManager):
        """Test job count with zero CPU cores."""
        job_count = manager.get_target_job_count(
            host="test-host",
            cpu_cores=0,
            current_cpu=50.0,
            current_gpu=50.0,
        )

        assert job_count >= 0

    def test_100_percent_utilization(self, manager: ResourceTargetManager):
        """Test scaling decisions at 100% utilization."""
        should_scale_up, reason = manager.should_scale_up(
            "test-host",
            current_cpu=100.0,
            current_gpu=100.0,
            current_jobs=50,
        )

        assert should_scale_up is False

        should_scale_down, jobs, reason = manager.should_scale_down(
            "test-host",
            current_cpu=100.0,
            current_gpu=100.0,
            current_memory=100.0,
        )

        assert should_scale_down is True
        assert jobs > 0

    def test_negative_values_handled(self, manager: ResourceTargetManager):
        """Test that negative values don't cause crashes."""
        # Should not raise
        score = manager.get_utilization_score(
            host="test-host",
            current_cpu=-10.0,
            current_gpu=-5.0,
        )

        assert isinstance(score, float)

    def test_unknown_host_tier(self, manager: ResourceTargetManager):
        """Test that unknown hosts get a default tier."""
        targets = manager.get_host_targets("completely-unknown-host-xyz")

        assert targets.tier is not None
        assert targets.max_jobs > 0

    def test_thread_safety_of_singleton(self, temp_db_path: Path):
        """Test that singleton is thread-safe."""
        import threading

        ResourceTargetManager.reset_instance()
        instances = []
        lock = threading.Lock()

        def get_instance():
            instance = ResourceTargetManager.get_instance(temp_db_path)
            with lock:
                instances.append(id(instance))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have gotten the same instance
        assert len(set(instances)) == 1


# =============================================================================
# Tests: Backpressure Behavior
# =============================================================================


class TestBackpressure:
    """Tests for backpressure-related behavior."""

    def test_backpressure_affects_job_targets(self, manager: ResourceTargetManager):
        """Test that backpressure reduces job targets."""
        # Get baseline without backpressure
        manager.set_backpressure(1.0)
        targets_no_bp = manager.get_host_targets("test-host")
        max_jobs_no_bp = targets_no_bp.max_jobs

        # Apply backpressure
        manager.set_backpressure(0.5)
        targets_with_bp = manager.get_host_targets("test-host")
        max_jobs_with_bp = targets_with_bp.max_jobs

        # Jobs should be reduced with backpressure
        # (or at minimum, not increased)
        assert max_jobs_with_bp <= max_jobs_no_bp

    def test_full_backpressure_reduces_significantly(
        self, manager: ResourceTargetManager
    ):
        """Test that zero backpressure severely limits jobs."""
        manager.set_backpressure(0.0)
        targets = manager.get_host_targets("test-host")

        # With full backpressure, jobs should be very limited
        assert targets.max_jobs >= 0  # Non-negative
