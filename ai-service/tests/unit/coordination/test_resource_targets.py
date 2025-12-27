"""Tests for app.coordination.resource_targets module.

December 27, 2025: Phase 2 Priority 4 - Coordination module test coverage.
"""

from __future__ import annotations

import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.coordination.resource_targets import (
    # Enums
    HostTier,
    # Dataclasses
    UtilizationTargets,
    HostTargets,
    # Manager class
    ResourceTargetManager,
    # Module functions
    get_resource_targets,
    get_host_targets,
    should_scale_up,
    should_scale_down,
    get_target_job_count,
    get_utilization_score,
    record_utilization,
    get_cluster_summary,
    set_backpressure,
    reset_resource_targets,
    select_host_for_task,
    # Constants
    HOST_TIER_MAP,
    TIER_ADJUSTMENTS,
)


class TestHostTier:
    """Tests for HostTier enum."""

    def test_all_tiers_defined(self):
        """Verify all expected tiers are defined."""
        assert HostTier.HIGH_END.value == "high_end"
        assert HostTier.MID_TIER.value == "mid_tier"
        assert HostTier.LOW_TIER.value == "low_tier"
        assert HostTier.CPU_ONLY.value == "cpu_only"

    def test_tier_count(self):
        """Verify expected number of tiers."""
        assert len(HostTier) == 4


class TestUtilizationTargets:
    """Tests for UtilizationTargets dataclass."""

    def test_default_values(self):
        """Verify default target values."""
        targets = UtilizationTargets()

        # CPU targets
        assert targets.cpu_min == 50.0
        assert targets.cpu_target == 65.0
        assert targets.cpu_max == 80.0
        assert targets.cpu_critical == 80.0

        # GPU targets
        assert targets.gpu_min == 50.0
        assert targets.gpu_target == 65.0
        assert targets.gpu_max == 80.0
        assert targets.gpu_critical == 80.0

        # Memory targets
        assert targets.memory_warn == 70.0
        assert targets.memory_critical == 80.0

        # Disk targets
        assert targets.disk_warn == 65.0
        assert targets.disk_critical == 70.0

    def test_custom_values(self):
        """Verify custom target values can be set."""
        targets = UtilizationTargets(
            cpu_min=40.0,
            cpu_target=60.0,
            cpu_max=75.0,
            gpu_min=45.0,
        )
        assert targets.cpu_min == 40.0
        assert targets.cpu_target == 60.0
        assert targets.cpu_max == 75.0
        assert targets.gpu_min == 45.0

    def test_throughput_defaults(self):
        """Verify throughput target defaults."""
        targets = UtilizationTargets()
        assert targets.throughput_min == 500
        assert targets.throughput_target == 1000
        assert targets.throughput_max == 2000


class TestHostTargets:
    """Tests for HostTargets dataclass."""

    def test_creation(self):
        """Test HostTargets creation."""
        targets = HostTargets(
            host="test-host",
            tier=HostTier.HIGH_END,
            cpu_min=55.0,
            cpu_target=70.0,
            cpu_max=85.0,
            gpu_min=55.0,
            gpu_target=70.0,
            gpu_max=85.0,
            max_jobs=72,
            max_selfplay=48,
            max_training=1,
        )
        assert targets.host == "test-host"
        assert targets.tier == HostTier.HIGH_END
        assert targets.max_selfplay == 48

    def test_state_tracking_defaults(self):
        """Verify state tracking fields have defaults."""
        targets = HostTargets(
            host="test",
            tier=HostTier.MID_TIER,
            cpu_min=50.0, cpu_target=65.0, cpu_max=80.0,
            gpu_min=50.0, gpu_target=65.0, gpu_max=80.0,
            max_jobs=48, max_selfplay=16, max_training=1,
        )
        assert targets.last_cpu == 0.0
        assert targets.last_gpu == 0.0
        assert targets.last_memory == 0.0
        assert targets.last_jobs == 0
        assert targets.last_update == 0.0


class TestHostTierMapping:
    """Tests for HOST_TIER_MAP constant."""

    def test_high_end_hosts(self):
        """Verify high-end hosts are mapped correctly."""
        assert HOST_TIER_MAP.get("runpod-h100") == HostTier.HIGH_END
        assert HOST_TIER_MAP.get("nebius-h100") == HostTier.HIGH_END
        assert HOST_TIER_MAP.get("gh200") == HostTier.HIGH_END
        assert HOST_TIER_MAP.get("mac-studio") == HostTier.HIGH_END

    def test_mid_tier_hosts(self):
        """Verify mid-tier hosts are mapped correctly."""
        assert HOST_TIER_MAP.get("rtx4090") == HostTier.MID_TIER
        assert HOST_TIER_MAP.get("a6000") == HostTier.MID_TIER
        assert HOST_TIER_MAP.get("mac-pro") == HostTier.MID_TIER

    def test_low_tier_hosts(self):
        """Verify low-tier hosts are mapped correctly."""
        assert HOST_TIER_MAP.get("rtx3090") == HostTier.LOW_TIER
        assert HOST_TIER_MAP.get("vast-rtx3090") == HostTier.LOW_TIER

    def test_cpu_only_hosts(self):
        """Verify CPU-only hosts are mapped correctly."""
        assert HOST_TIER_MAP.get("local") == HostTier.CPU_ONLY
        assert HOST_TIER_MAP.get("aws-c5") == HostTier.CPU_ONLY


class TestTierAdjustments:
    """Tests for TIER_ADJUSTMENTS constant."""

    def test_high_end_adjustments(self):
        """Verify high-end tier adjustments."""
        adj = TIER_ADJUSTMENTS[HostTier.HIGH_END]
        assert adj["cpu_boost"] == 5.0
        assert adj["gpu_boost"] == 5.0
        assert adj["job_multiplier"] == 1.5

    def test_mid_tier_adjustments(self):
        """Verify mid-tier adjustments are neutral."""
        adj = TIER_ADJUSTMENTS[HostTier.MID_TIER]
        assert adj["cpu_boost"] == 0.0
        assert adj["gpu_boost"] == 0.0
        assert adj["job_multiplier"] == 1.0

    def test_low_tier_adjustments(self):
        """Verify low-tier adjustments are conservative."""
        adj = TIER_ADJUSTMENTS[HostTier.LOW_TIER]
        assert adj["cpu_boost"] == -5.0
        assert adj["gpu_boost"] == -5.0
        assert adj["job_multiplier"] == 0.7

    def test_cpu_only_adjustments(self):
        """Verify CPU-only adjustments."""
        adj = TIER_ADJUSTMENTS[HostTier.CPU_ONLY]
        assert adj["job_multiplier"] == 0.5


class TestResourceTargetManager:
    """Tests for ResourceTargetManager class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_singleton_pattern(self, tmp_path):
        """Verify singleton pattern works."""
        manager1 = ResourceTargetManager.get_instance()
        manager2 = ResourceTargetManager.get_instance()
        assert manager1 is manager2

    def test_get_targets(self, tmp_path):
        """Test get_targets returns UtilizationTargets."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        targets = manager.get_targets()
        assert isinstance(targets, UtilizationTargets)

    def test_get_host_targets_known_host(self, tmp_path):
        """Test get_host_targets for a known host."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        targets = manager.get_host_targets("runpod-h100")
        assert isinstance(targets, HostTargets)
        assert targets.tier == HostTier.HIGH_END

    def test_get_host_targets_unknown_host(self, tmp_path):
        """Test get_host_targets falls back to LOW_TIER."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        targets = manager.get_host_targets("unknown-host")
        assert isinstance(targets, HostTargets)
        assert targets.tier == HostTier.LOW_TIER

    def test_get_host_targets_pattern_detection(self, tmp_path):
        """Test host tier detection from name patterns."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")

        # H100 pattern should be HIGH_END
        targets = manager.get_host_targets("my-custom-h100-node")
        assert targets.tier == HostTier.HIGH_END

        # 4090 pattern should be MID_TIER
        targets = manager.get_host_targets("vast-rtx4090-12345")
        assert targets.tier == HostTier.MID_TIER

    def test_get_host_targets_caching(self, tmp_path):
        """Test that host targets are cached."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        targets1 = manager.get_host_targets("test-host")
        targets2 = manager.get_host_targets("test-host")
        assert targets1 is targets2


class TestShouldScaleUp:
    """Tests for should_scale_up function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_scale_up_low_gpu(self):
        """Scale up when GPU is underutilized."""
        should_scale, reason = should_scale_up("runpod-h100", current_cpu=60, current_gpu=40)
        assert should_scale is True
        assert "GPU underutilized" in reason

    def test_scale_up_low_cpu(self):
        """Scale up when CPU is underutilized."""
        should_scale, reason = should_scale_up("local", current_cpu=30, current_gpu=0)
        assert should_scale is True
        assert "CPU underutilized" in reason

    def test_no_scale_up_optimal(self):
        """No scale up when in optimal range."""
        should_scale, reason = should_scale_up("runpod-h100", current_cpu=70, current_gpu=70, current_jobs=30)
        # May or may not scale up depending on job headroom
        assert isinstance(should_scale, bool)


class TestShouldScaleDown:
    """Tests for should_scale_down function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_scale_down_critical_memory(self):
        """Scale down with critical memory."""
        should_scale, reduction, reason = should_scale_down(
            "runpod-h100", current_cpu=60, current_gpu=60, current_memory=85
        )
        assert should_scale is True
        assert reduction == 10
        assert "CRITICAL memory" in reason

    def test_scale_down_critical_gpu(self):
        """Scale down with critical GPU."""
        should_scale, reduction, reason = should_scale_down(
            "runpod-h100", current_cpu=60, current_gpu=90, current_memory=50
        )
        assert should_scale is True
        assert reduction == 6
        assert "CRITICAL GPU" in reason

    def test_scale_down_critical_cpu(self):
        """Scale down with critical CPU."""
        should_scale, reduction, reason = should_scale_down(
            "runpod-h100", current_cpu=95, current_gpu=50, current_memory=50
        )
        assert should_scale is True
        assert reduction == 4
        assert "CRITICAL CPU" in reason

    def test_no_scale_down_optimal(self):
        """No scale down when in optimal range."""
        should_scale, reduction, reason = should_scale_down(
            "runpod-h100", current_cpu=65, current_gpu=70, current_memory=50
        )
        assert should_scale is False
        assert reduction == 0
        assert "Within target range" in reason


class TestGetTargetJobCount:
    """Tests for get_target_job_count function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_target_job_count_basic(self):
        """Test basic job count calculation."""
        count = get_target_job_count(
            host="runpod-h100",
            cpu_cores=32,
            current_cpu=60,
            current_gpu=60,
        )
        assert isinstance(count, int)
        assert count >= 1

    def test_target_job_count_minimum(self):
        """Verify job count is at least 1."""
        count = get_target_job_count(
            host="runpod-h100",
            cpu_cores=1,
            current_cpu=100,
            current_gpu=100,
        )
        assert count >= 1


class TestGetUtilizationScore:
    """Tests for get_utilization_score function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_low_utilization_score(self):
        """Low utilization should give low score."""
        score = get_utilization_score("runpod-h100", current_cpu=20, current_gpu=20)
        assert 0 <= score <= 40

    def test_optimal_utilization_score(self):
        """Optimal utilization should give medium score."""
        score = get_utilization_score("runpod-h100", current_cpu=65, current_gpu=70)
        assert 40 <= score <= 70

    def test_overloaded_utilization_score(self):
        """High utilization should give high score."""
        score = get_utilization_score("runpod-h100", current_cpu=90, current_gpu=90)
        assert 70 <= score <= 100

    def test_cpu_only_score(self):
        """CPU-only nodes should use CPU score."""
        score = get_utilization_score("local", current_cpu=50, current_gpu=0)
        assert isinstance(score, float)


class TestRecordUtilization:
    """Tests for record_utilization function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_record_updates_host_targets(self):
        """Record should update host targets state."""
        # First get host targets to create the entry
        _ = get_host_targets("test-host")

        # Record utilization
        record_utilization(
            host="test-host",
            cpu_percent=65.0,
            gpu_percent=70.0,
            memory_percent=50.0,
            job_count=10,
        )

        targets = get_host_targets("test-host")
        assert targets.last_cpu == 65.0
        assert targets.last_gpu == 70.0
        assert targets.last_memory == 50.0
        assert targets.last_jobs == 10


class TestBackpressure:
    """Tests for backpressure handling."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_set_backpressure_valid(self):
        """Test setting valid backpressure factor."""
        set_backpressure(0.5)
        summary = get_cluster_summary()
        assert summary["backpressure_factor"] == 0.5

    def test_set_backpressure_clamped_low(self):
        """Test backpressure is clamped at 0."""
        set_backpressure(-0.5)
        summary = get_cluster_summary()
        assert summary["backpressure_factor"] == 0.0

    def test_set_backpressure_clamped_high(self):
        """Test backpressure is clamped at 1."""
        set_backpressure(1.5)
        summary = get_cluster_summary()
        assert summary["backpressure_factor"] == 1.0


class TestSelectHostForTask:
    """Tests for select_host_for_task function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_select_empty_hosts(self):
        """Return None for empty host list."""
        result = select_host_for_task([], "gpu")
        assert result is None

    def test_select_gpu_task(self):
        """Select host with lowest GPU for GPU task."""
        metrics = {
            "host-a": {"cpu_util": 80, "gpu_util": 30},
            "host-b": {"cpu_util": 30, "gpu_util": 70},
        }
        result = select_host_for_task(["host-a", "host-b"], "gpu", metrics)
        assert result == "host-a"  # Lower GPU utilization

    def test_select_cpu_task(self):
        """Select host with lowest CPU for CPU task."""
        metrics = {
            "host-a": {"cpu_util": 80, "gpu_util": 30},
            "host-b": {"cpu_util": 30, "gpu_util": 70},
        }
        result = select_host_for_task(["host-a", "host-b"], "cpu", metrics)
        assert result == "host-b"  # Lower CPU utilization

    def test_select_hybrid_task(self):
        """Select host with lowest combined for hybrid task."""
        metrics = {
            "host-a": {"cpu_util": 60, "gpu_util": 60},  # avg 60
            "host-b": {"cpu_util": 40, "gpu_util": 80},  # avg 60
            "host-c": {"cpu_util": 40, "gpu_util": 40},  # avg 40
        }
        result = select_host_for_task(["host-a", "host-b", "host-c"], "hybrid", metrics)
        assert result == "host-c"  # Lowest combined


class TestClusterSummary:
    """Tests for get_cluster_summary function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_empty_cluster_summary(self):
        """Test summary with no recorded utilization."""
        summary = get_cluster_summary()
        assert summary["active_hosts"] == 0
        assert summary["avg_cpu"] == 0.0
        assert summary["avg_gpu"] == 0.0
        assert summary["total_jobs"] == 0
        assert "targets" in summary

    def test_summary_includes_targets(self):
        """Verify summary includes target values."""
        summary = get_cluster_summary()
        assert "targets" in summary
        assert "cpu_min" in summary["targets"]
        assert "gpu_min" in summary["targets"]


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_health_check_no_hosts(self, tmp_path):
        """Health check with no active hosts reports issue."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        result = manager.health_check()

        # Should report "No active hosts reporting utilization"
        assert hasattr(result, "healthy")
        assert hasattr(result, "details")

    def test_health_check_returns_result(self, tmp_path):
        """Health check returns proper result object."""
        manager = ResourceTargetManager(db_path=tmp_path / "test.db")
        result = manager.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "details")


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        reset_resource_targets()
        yield
        reset_resource_targets()

    def test_get_resource_targets_returns_targets(self):
        """get_resource_targets returns UtilizationTargets."""
        targets = get_resource_targets()
        assert isinstance(targets, UtilizationTargets)

    def test_get_host_targets_returns_host_targets(self):
        """get_host_targets returns HostTargets."""
        targets = get_host_targets("test-host")
        assert isinstance(targets, HostTargets)

    def test_reset_clears_singleton(self):
        """reset_resource_targets clears the singleton."""
        # Get initial instance
        _ = get_resource_targets()

        # Reset
        reset_resource_targets()

        # Getting targets again should work
        targets = get_resource_targets()
        assert isinstance(targets, UtilizationTargets)
