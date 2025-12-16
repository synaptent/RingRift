#!/usr/bin/env python3
"""Tests for the unified resource guard module.

Tests the 80% max utilization thresholds and resource checking utilities.
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

    def test_disk_limit_is_70_percent(self):
        """Disk limit should be 70% (tighter than others)."""
        assert LIMITS.DISK_MAX_PERCENT == 70.0

    def test_memory_limit_is_80_percent(self):
        """Memory limit should be 80%."""
        assert LIMITS.MEMORY_MAX_PERCENT == 80.0

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
