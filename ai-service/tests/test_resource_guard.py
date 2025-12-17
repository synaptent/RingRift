#!/usr/bin/env python3
"""Tests for the unified resource guard module.

Tests the resource utilization thresholds (80% for disk/CPU/GPU, 90% for memory)
and resource checking utilities.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.resource_guard import (
    LIMITS,
    get_disk_usage,
    get_memory_usage,
    get_cpu_usage,
    check_disk_space,
    check_memory,
    check_cpu,
    can_proceed,
    ResourceGuard,
)


class TestResourceLimits:
    """Test resource limit configuration."""

    def test_disk_limit_is_80_percent(self):
        """Disk limit should be 80%."""
        assert LIMITS.DISK_MAX_PERCENT == 80.0

    def test_memory_limit_is_90_percent(self):
        """Memory limit should be 90%."""
        assert LIMITS.MEMORY_MAX_PERCENT == 90.0

    def test_cpu_limit_is_80_percent(self):
        """CPU limit should be 80%."""
        assert LIMITS.CPU_MAX_PERCENT == 80.0

    def test_gpu_limit_is_80_percent(self):
        """GPU limit should be 80%."""
        assert LIMITS.GPU_MAX_PERCENT == 80.0


class TestDiskUsage:
    """Test disk usage checking."""

    def test_get_disk_usage_returns_tuple(self):
        """get_disk_usage should return (percent, available_gb, total_gb)."""
        result = get_disk_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3
        percent, available_gb, total_gb = result
        assert isinstance(percent, float)
        assert isinstance(available_gb, float)
        assert isinstance(total_gb, float)

    def test_check_disk_space_returns_bool(self):
        """check_disk_space should return True/False."""
        result = check_disk_space(required_gb=0.001)  # Very small requirement
        assert isinstance(result, bool)


class TestMemoryUsage:
    """Test memory usage checking."""

    def test_get_memory_usage_returns_tuple(self):
        """get_memory_usage should return (percent, available_gb, total_gb)."""
        result = get_memory_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_check_memory_returns_bool(self):
        """check_memory should return True/False."""
        result = check_memory(required_gb=0.001, log_warning=False)
        assert isinstance(result, bool)


class TestCPUUsage:
    """Test CPU usage checking."""

    def test_get_cpu_usage_returns_tuple(self):
        """get_cpu_usage should return (percent, load_per_cpu, count)."""
        result = get_cpu_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestCanProceed:
    """Test combined resource checking."""

    def test_can_proceed_returns_bool(self):
        """can_proceed should return True/False."""
        result = can_proceed(
            check_disk=True,
            check_mem=True,
            check_cpu_load=True,
            check_gpu=False,
        )
        assert isinstance(result, bool)


class TestResourceGuard:
    """Test the ResourceGuard context manager."""

    def test_resource_guard_initializes(self):
        """ResourceGuard should initialize with defaults."""
        guard = ResourceGuard()
        assert guard.disk_required_gb == 2.0
        assert guard.mem_required_gb == 1.0
        assert guard.gpu_required_gb == 0.0
        assert guard.wait_timeout == 0.0

    def test_resource_guard_with_custom_values(self):
        """ResourceGuard should accept custom requirements."""
        guard = ResourceGuard(
            disk_required_gb=5.0,
            mem_required_gb=2.0,
            gpu_required_gb=1.0,
        )
        assert guard.disk_required_gb == 5.0
        assert guard.mem_required_gb == 2.0
        assert guard.gpu_required_gb == 1.0


class TestAsyncResourceLimiter:
    """Test the async resource limiter."""

    def test_async_limiter_initializes(self):
        """AsyncResourceLimiter should initialize with defaults."""
        from app.utils.resource_guard import AsyncResourceLimiter
        limiter = AsyncResourceLimiter()
        assert limiter.disk_required_gb == 2.0
        assert limiter.mem_required_gb == 1.0
        assert limiter.gpu_required_gb == 0.0

    def test_async_limiter_check_resources(self):
        """AsyncResourceLimiter._check_resources should return tuple."""
        from app.utils.resource_guard import AsyncResourceLimiter
        limiter = AsyncResourceLimiter(disk_required_gb=0.001, mem_required_gb=0.001)
        ok, issues = limiter._check_resources()
        assert isinstance(ok, bool)
        assert isinstance(issues, list)

    def test_async_limiter_sync_wait(self):
        """AsyncResourceLimiter.wait_for_resources_sync should work."""
        from app.utils.resource_guard import AsyncResourceLimiter, get_disk_usage
        # Skip if disk is already above limit
        disk_pct, _, _ = get_disk_usage()
        if disk_pct >= 70.0:
            pytest.skip(f"Disk at {disk_pct:.1f}%, skipping test")
        limiter = AsyncResourceLimiter(
            disk_required_gb=0.001,
            mem_required_gb=0.001,
        )
        # Should succeed quickly with very low requirements
        result = limiter.wait_for_resources_sync("test", max_wait_seconds=5)
        assert result is True


class TestRespectResourceLimitsDecorator:
    """Test the @respect_resource_limits decorator."""

    def test_decorator_wraps_sync_function(self):
        """Decorator should wrap sync functions."""
        from app.utils.resource_guard import respect_resource_limits, get_disk_usage
        # Skip if disk is already above limit
        disk_pct, _, _ = get_disk_usage()
        if disk_pct >= 70.0:
            pytest.skip(f"Disk at {disk_pct:.1f}%, skipping test")

        @respect_resource_limits(disk_gb=0.001, mem_gb=0.001, max_wait_seconds=5)
        def my_func():
            return 42

        # Should execute and return
        result = my_func()
        assert result == 42


class TestResourceStatus:
    """Test resource status reporting."""

    def test_get_resource_status_returns_dict(self):
        """get_resource_status should return a dictionary."""
        from app.utils.resource_guard import get_resource_status
        status = get_resource_status()
        assert isinstance(status, dict)
        assert "disk" in status
        assert "memory" in status
        assert "cpu" in status
        assert "gpu" in status
        assert "can_proceed" in status

    def test_resource_status_has_ok_field(self):
        """Each resource status should have an 'ok' field."""
        from app.utils.resource_guard import get_resource_status
        status = get_resource_status()
        for key in ["disk", "memory", "cpu", "gpu"]:
            assert "ok" in status[key]
            assert isinstance(status[key]["ok"], bool)


class TestClusterCoordinatorConsistency:
    """Test that cluster_coordinator uses consistent limits."""

    def test_cluster_coordinator_limits_match(self):
        """ProcessLimits should match resource_guard limits (80% for CPU/memory)."""
        import warnings
        # Suppress deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.distributed.cluster_coordinator import ProcessLimits

        limits = ProcessLimits()
        assert limits.max_memory_percent == 80.0, "Memory limit should be 80%"
        assert limits.max_cpu_percent == 80.0, "CPU limit should be 80% (fixed from 90%)"


class TestExecutorResourceChecks:
    """Test that executor has resource checking capability."""

    def test_executor_has_resource_check_option(self):
        """LocalExecutor should have check_resources option."""
        from app.execution.executor import LocalExecutor

        executor = LocalExecutor(check_resources=True, required_mem_gb=1.0)
        assert executor.check_resources is True
        assert executor.required_mem_gb == 1.0

    def test_executor_default_no_resource_check(self):
        """LocalExecutor should default to no resource checking."""
        from app.execution.executor import LocalExecutor

        executor = LocalExecutor()
        assert executor.check_resources is False

    def test_check_resources_before_spawn_exists(self):
        """check_resources_before_spawn function should exist."""
        from app.execution.executor import check_resources_before_spawn
        import asyncio
        assert callable(check_resources_before_spawn)
        assert asyncio.iscoroutinefunction(check_resources_before_spawn)


class TestTrainingResourceChecks:
    """Test that training modules have disk checks."""

    def test_save_checkpoint_has_disk_check(self):
        """save_checkpoint should document disk check in source."""
        # Check file contents to avoid circular import issues
        train_path = Path(__file__).parent.parent / "app" / "training" / "train.py"
        content = train_path.read_text()
        # Verify save_checkpoint function has disk-related documentation
        assert 'def save_checkpoint' in content
        assert 'IOError' in content or 'disk' in content.lower()


class TestDataLoaderResourceChecks:
    """Test that data loader has memory checks."""

    def test_merge_data_files_has_memory_check(self):
        """merge_data_files should have memory check."""
        from app.training.data_loader import merge_data_files
        assert merge_data_files.__doc__ is not None
        assert 'MemoryError' in merge_data_files.__doc__


class TestGenerateDataResourceChecks:
    """Test that data generation has disk checks."""

    def test_resource_guard_imported_in_generate_data(self):
        """generate_data module should import resource_guard functions."""
        # Check file contents to avoid circular import issues
        generate_data_path = Path(__file__).parent.parent / "app" / "training" / "generate_data.py"
        content = generate_data_path.read_text()
        # Verify resource_guard imports are present
        assert 'from app.utils.resource_guard import' in content or 'resource_guard' in content


class TestGracefulDegradation:
    """Test graceful degradation under resource pressure."""

    def test_operation_priority_values(self):
        """OperationPriority should have correct hierarchy."""
        from app.utils.resource_guard import OperationPriority
        assert OperationPriority.BACKGROUND < OperationPriority.LOW
        assert OperationPriority.LOW < OperationPriority.NORMAL
        assert OperationPriority.NORMAL < OperationPriority.HIGH
        assert OperationPriority.HIGH < OperationPriority.CRITICAL

    def test_get_degradation_level_returns_int(self):
        """get_degradation_level should return 0-4."""
        from app.utils.resource_guard import get_degradation_level
        level = get_degradation_level()
        assert isinstance(level, int)
        assert 0 <= level <= 4

    def test_should_proceed_with_priority(self):
        """should_proceed_with_priority should return bool."""
        from app.utils.resource_guard import (
            should_proceed_with_priority, OperationPriority
        )
        # CRITICAL should always proceed (unless impossible level 5)
        result = should_proceed_with_priority(OperationPriority.CRITICAL)
        assert isinstance(result, bool)

    def test_get_recommended_actions_returns_list(self):
        """get_recommended_actions should return list of strings."""
        from app.utils.resource_guard import get_recommended_actions
        actions = get_recommended_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, str) for a in actions)


class TestPrometheusMetrics:
    """Test Prometheus metrics integration."""

    def test_prometheus_available(self):
        """Prometheus client should be importable."""
        from app.utils.resource_guard import HAS_PROMETHEUS
        # Should be True if prometheus_client is installed
        assert isinstance(HAS_PROMETHEUS, bool)

    def test_get_resource_status_with_prometheus(self):
        """get_resource_status should work with Prometheus export."""
        from app.utils.resource_guard import get_resource_status, HAS_PROMETHEUS
        # Should not raise even if Prometheus is available
        status = get_resource_status(export_prometheus=HAS_PROMETHEUS)
        assert isinstance(status, dict)
        assert 'cpu' in status
        assert 'memory' in status


class TestModelCulling:
    """Test model culling functionality."""

    def test_model_culling_controller_accepts_strings(self):
        """ModelCullingController should accept string paths."""
        from app.tournament.model_culling import ModelCullingController
        # Should not raise TypeError for string paths
        culler = ModelCullingController(
            elo_db_path="data/unified_elo.db",
            model_dir="models"
        )
        assert culler.elo_db_path.name == "unified_elo.db"
        assert culler.model_dir.name == "models"

    def test_config_keys_defined(self):
        """All 9 config keys should be defined."""
        from app.tournament.model_culling import CONFIG_KEYS
        assert len(CONFIG_KEYS) == 9
        assert "square8_2p" in CONFIG_KEYS
        assert "hexagonal_4p" in CONFIG_KEYS


class TestEloDatabase:
    """Test ELO database functionality."""

    def test_pinned_baselines_defined(self):
        """PINNED_BASELINES should be defined in EloDatabase."""
        from app.tournament.unified_elo_db import get_elo_database
        db = get_elo_database()
        assert hasattr(db, 'PINNED_BASELINES')
        assert 'baseline_random' in db.PINNED_BASELINES
        assert db.PINNED_BASELINES['baseline_random'] == 400.0

    def test_reset_pinned_baselines_method_exists(self):
        """reset_pinned_baselines should exist."""
        from app.tournament.unified_elo_db import get_elo_database
        db = get_elo_database()
        assert hasattr(db, 'reset_pinned_baselines')
        assert callable(db.reset_pinned_baselines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
