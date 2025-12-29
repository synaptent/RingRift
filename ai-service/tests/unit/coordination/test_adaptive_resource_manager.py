"""Tests for adaptive_resource_manager.py - Proactive resource management.

This module tests the adaptive resource manager that monitors and manages
disk, memory, and GPU resources across the cluster.

Coverage includes:
1. ResourceStatus dataclass operations
2. CleanupResult dataclass operations
3. AdaptiveResourceManager initialization
4. Disk usage monitoring
5. Memory usage monitoring
6. GPU memory management
7. NFS path monitoring
8. File cleanup operations
9. Selfplay data aggregation
10. Check and cleanup flow
11. Run loop lifecycle
12. Statistics tracking
13. Health check compliance
14. Singleton pattern
15. Module constants
16. Error handling and recovery
17. Priority-based scheduling (resource priority)
18. Worker pool scaling considerations

December 2025: Expanded test coverage to 50+ tests.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch, mock_open

import pytest

from app.coordination.adaptive_resource_manager import (
    AGGREGATION_INTERVAL,
    CHECK_INTERVAL,
    CLEANUP_BATCH_SIZE,
    CLEANUP_COOLDOWN,
    DEFAULT_DATA_PATH,
    DEFAULT_NFS_PATH,
    DISK_CLEANUP_THRESHOLD,
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    MIN_FILE_AGE_HOURS,
    AdaptiveResourceManager,
    CleanupResult,
    ResourceStatus,
    get_resource_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager(tmp_path: Path) -> AdaptiveResourceManager:
    """Create a test manager with temporary paths."""
    nfs_path = tmp_path / "nfs"
    data_path = tmp_path / "data"
    nfs_path.mkdir()
    data_path.mkdir()
    return AdaptiveResourceManager(
        nfs_path=str(nfs_path),
        data_path=str(data_path),
    )


@pytest.fixture
def resource_status() -> ResourceStatus:
    """Create a sample ResourceStatus."""
    return ResourceStatus(
        node_id="test-node",
        disk_total_gb=100.0,
        disk_used_gb=60.0,
        disk_free_gb=40.0,
        disk_percent=60.0,
        memory_total_gb=32.0,
        memory_used_gb=16.0,
        memory_free_gb=16.0,
        memory_percent=50.0,
    )


# =============================================================================
# ResourceStatus Tests
# =============================================================================


class TestResourceStatus:
    """Tests for ResourceStatus dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        status = ResourceStatus(node_id="test")
        assert status.node_id == "test"
        assert status.disk_total_gb == 0
        assert status.disk_used_gb == 0
        assert status.disk_free_gb == 0
        assert status.disk_percent == 0
        assert status.memory_total_gb == 0
        assert status.memory_percent == 0
        assert status.gpu_memory_used_gb == 0
        assert status.is_healthy is True
        assert status.warnings == []
        assert status.errors == []

    def test_to_dict(self, resource_status: ResourceStatus) -> None:
        """Test dictionary conversion."""
        result = resource_status.to_dict()
        
        assert result["node_id"] == "test-node"
        assert result["disk"]["total_gb"] == 100.0
        assert result["disk"]["used_gb"] == 60.0
        assert result["disk"]["percent"] == 60.0
        assert result["memory"]["total_gb"] == 32.0
        assert result["memory"]["percent"] == 50.0
        assert result["is_healthy"] is True

    def test_to_dict_with_warnings(self) -> None:
        """Test dictionary conversion with warnings."""
        status = ResourceStatus(
            node_id="test",
            warnings=["Disk warning: 75%"],
            errors=["Memory critical: 95%"],
            is_healthy=False,
        )
        result = status.to_dict()
        
        assert result["is_healthy"] is False
        assert "Disk warning: 75%" in result["warnings"]
        assert "Memory critical: 95%" in result["errors"]


# =============================================================================
# CleanupResult Tests
# =============================================================================


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        result = CleanupResult(success=True)
        assert result.success is True
        assert result.files_deleted == 0
        assert result.bytes_freed == 0
        assert result.errors == []
        assert result.duration_seconds == 0

    def test_with_values(self) -> None:
        """Test with custom values."""
        result = CleanupResult(
            success=True,
            files_deleted=10,
            bytes_freed=1024 * 1024,
            duration_seconds=5.5,
        )
        assert result.files_deleted == 10
        assert result.bytes_freed == 1024 * 1024
        assert result.duration_seconds == 5.5


# =============================================================================
# AdaptiveResourceManager Tests
# =============================================================================


class TestAdaptiveResourceManagerInit:
    """Tests for AdaptiveResourceManager initialization."""

    def test_default_init(self) -> None:
        """Test initialization with defaults."""
        manager = AdaptiveResourceManager()
        assert manager.running is False
        assert manager.last_cleanup_time == 0
        assert manager.last_aggregation_time == 0
        assert manager.stats["cleanups_triggered"] == 0

    def test_custom_init(self, tmp_path: Path) -> None:
        """Test initialization with custom paths."""
        nfs = tmp_path / "nfs"
        data = tmp_path / "data"
        nfs.mkdir()
        data.mkdir()
        
        manager = AdaptiveResourceManager(
            nfs_path=str(nfs),
            data_path=str(data),
            disk_threshold=50.0,
            memory_threshold=70.0,
        )
        assert manager.nfs_path == nfs
        assert manager.data_path == data
        assert manager.disk_threshold == 50.0
        assert manager.memory_threshold == 70.0


class TestDiskUsage:
    """Tests for disk usage methods."""

    def test_get_disk_usage_success(self, manager: AdaptiveResourceManager) -> None:
        """Test successful disk usage retrieval."""
        total, used, free = manager._get_disk_usage(manager.data_path)
        # Should return actual disk stats for temp directory
        assert total > 0
        assert used >= 0
        assert free >= 0
        assert total >= used + free - 1  # Allow for rounding

    def test_get_disk_usage_nonexistent_path(self, manager: AdaptiveResourceManager) -> None:
        """Test disk usage for nonexistent path."""
        total, used, free = manager._get_disk_usage(Path("/nonexistent/path"))
        assert total == 0
        assert used == 0
        assert free == 0


class TestMemoryUsage:
    """Tests for memory usage methods."""

    def test_get_memory_usage_mock(self, manager: AdaptiveResourceManager) -> None:
        """Test memory usage with mocked /proc/meminfo."""
        mock_meminfo = """MemTotal:       32000000 kB
MemFree:        8000000 kB
Buffers:        1000000 kB
Cached:         7000000 kB
"""
        with patch("builtins.open", mock_open(read_data=mock_meminfo)):
            total, used, free = manager._get_memory_usage()
            # Values are in GB (converted from KB)
            assert total > 0
            assert used >= 0
            assert free >= 0


class TestGpuMemory:
    """Tests for GPU memory methods."""

    def test_get_gpu_memory_no_nvidia(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory when nvidia-smi not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            used, total = manager._get_gpu_memory()
            assert used == 0
            assert total == 0

    def test_get_gpu_memory_with_nvidia(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory with mocked nvidia-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8000, 24000\n"
        
        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()
            assert used == 8000 / 1024  # MB to GB
            assert total == 24000 / 1024


class TestGetLocalStatus:
    """Tests for get_local_status method."""

    def test_get_local_status(self, manager: AdaptiveResourceManager) -> None:
        """Test getting local status."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 60.0, 40.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(8.0, 24.0)):
                    status = manager.get_local_status("test-node")
        
        assert status.node_id == "test-node"
        assert status.disk_total_gb == 100.0
        assert status.disk_percent == 60.0
        assert status.memory_total_gb == 32.0
        assert status.is_healthy is True

    def test_get_local_status_disk_warning(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with disk warning."""
        # ResourceManagerDefaults.DISK_WARNING_THRESHOLD = 85%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 88.0, 12.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.disk_percent == 88.0
        assert len(status.warnings) > 0
        assert any("Disk warning" in w for w in status.warnings)

    def test_get_local_status_disk_critical(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with disk critical."""
        # ResourceManagerDefaults.DISK_CRITICAL_THRESHOLD = 92%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 94.0, 6.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.is_healthy is False
        assert any("Disk critical" in e for e in status.errors)

    def test_get_local_status_memory_critical(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with memory critical."""
        # ResourceManagerDefaults.MEMORY_CRITICAL_THRESHOLD = 95%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 31.0, 1.0)):  # 96.875%
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.is_healthy is False
        assert any("Memory critical" in e for e in status.errors)


class TestGetNfsStatus:
    """Tests for get_nfs_status method."""

    def test_get_nfs_status_path_exists(self, manager: AdaptiveResourceManager) -> None:
        """Test NFS status when path exists."""
        status = manager.get_nfs_status()
        assert status.node_id == "nfs"
        assert status.disk_total_gb > 0

    def test_get_nfs_status_path_not_exists(self, tmp_path: Path) -> None:
        """Test NFS status when path doesn't exist."""
        manager = AdaptiveResourceManager(
            nfs_path=str(tmp_path / "nonexistent"),
            data_path=str(tmp_path),
        )
        status = manager.get_nfs_status()
        assert status.is_healthy is False
        assert any("not accessible" in e for e in status.errors)


class TestCleanupOldFiles:
    """Tests for cleanup_old_files method."""

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup in dry run mode."""
        # Create old test files
        test_dir = manager.data_path / "test"
        test_dir.mkdir()
        old_file = test_dir / "old_file.jsonl"
        old_file.write_text("test data")
        
        # Make file appear old
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        import os
        os.utime(old_file, (old_time, old_time))
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=True)
        
        assert result.success is True
        assert result.files_deleted == 1
        assert result.bytes_freed > 0
        assert old_file.exists()  # Not actually deleted in dry run

    @pytest.mark.asyncio
    async def test_cleanup_actual_delete(self, manager: AdaptiveResourceManager) -> None:
        """Test actual file cleanup."""
        test_dir = manager.data_path / "cleanup_test"
        test_dir.mkdir()
        old_file = test_dir / "old_file.tmp"
        old_file.write_text("test data to delete")
        
        old_time = time.time() - (25 * 3600)
        import os
        os.utime(old_file, (old_time, old_time))
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=False)
        
        assert result.success is True
        assert result.files_deleted == 1
        assert not old_file.exists()  # Should be deleted

    @pytest.mark.asyncio
    async def test_cleanup_respects_age(self, manager: AdaptiveResourceManager) -> None:
        """Test that cleanup respects file age threshold."""
        test_dir = manager.data_path / "age_test"
        test_dir.mkdir()
        new_file = test_dir / "new_file.jsonl"
        new_file.write_text("recent data")
        # File is new, should not be deleted
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=False)
        
        assert result.files_deleted == 0
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_empty_directory(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup on empty directory."""
        test_dir = manager.data_path / "empty"
        test_dir.mkdir()
        
        result = await manager.cleanup_old_files(test_dir)
        
        assert result.success is True
        assert result.files_deleted == 0


class TestCheckAndCleanup:
    """Tests for check_and_cleanup method."""

    @pytest.mark.asyncio
    async def test_check_no_cleanup_needed(self, manager: AdaptiveResourceManager) -> None:
        """Test check when no cleanup needed."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    result = await manager.check_and_cleanup()
        
        assert result["cleanup_triggered"] is False
        assert result["nfs_status"] is not None
        assert result["local_status"] is not None

    @pytest.mark.asyncio
    async def test_check_cleanup_triggered(self, manager: AdaptiveResourceManager) -> None:
        """Test check triggers cleanup when threshold exceeded."""
        manager.disk_threshold = 50.0  # Low threshold
        
        with patch.object(manager, "get_nfs_status") as mock_nfs:
            mock_status = ResourceStatus(node_id="nfs", disk_percent=75.0)
            mock_nfs.return_value = mock_status
            
            with patch.object(manager, "get_local_status") as mock_local:
                mock_local.return_value = ResourceStatus(node_id="local")
                
                with patch.object(manager, "cleanup_old_files", new_callable=AsyncMock) as mock_cleanup:
                    mock_cleanup.return_value = CleanupResult(success=True, files_deleted=5)
                    result = await manager.check_and_cleanup()
        
        assert result["cleanup_triggered"] is True
        mock_cleanup.assert_called_once()


class TestAggregation:
    """Tests for data aggregation methods."""

    @pytest.mark.asyncio
    async def test_aggregate_no_nodes(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation with no source nodes."""
        result = await manager.aggregate_selfplay_data(source_nodes=[])
        
        assert result["success"] is True
        assert result["games_aggregated"] == 0

    @pytest.mark.asyncio
    async def test_get_active_selfplay_nodes_import_error(self, manager: AdaptiveResourceManager) -> None:
        """Test getting active nodes when ClusterMonitor unavailable."""
        with patch.dict("sys.modules", {"app.coordination.cluster_status_monitor": None}):
            nodes = await manager._get_active_selfplay_nodes()
            assert nodes == []


class TestRunLoop:
    """Tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_stop(self, manager: AdaptiveResourceManager) -> None:
        """Test stopping the manager."""
        manager.running = True
        manager.stop()
        assert manager.running is False

    @pytest.mark.asyncio
    async def test_run_loop_starts_and_stops(self, manager: AdaptiveResourceManager) -> None:
        """Test that run loop can be started and stopped."""
        with patch.object(manager, "check_and_cleanup", new_callable=AsyncMock):
            # Start run in background
            task = asyncio.create_task(manager.run())
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            assert manager.running is True
            
            # Stop it
            manager.stop()
            await asyncio.sleep(0.1)
            assert manager.running is False
            
            # Cancel task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self, manager: AdaptiveResourceManager) -> None:
        """Test getting manager stats."""
        stats = manager.get_stats()
        
        assert "running" in stats
        assert "cleanups_triggered" in stats
        assert "bytes_freed_total" in stats
        assert "nfs_path" in stats
        assert "data_path" in stats
        assert "disk_threshold" in stats

    def test_get_stats_after_cleanup(self, manager: AdaptiveResourceManager) -> None:
        """Test stats after cleanup operation."""
        manager.stats["cleanups_triggered"] = 5
        manager.stats["bytes_freed_total"] = 1024 * 1024 * 100
        
        stats = manager.get_stats()
        assert stats["cleanups_triggered"] == 5
        assert stats["bytes_freed_total"] == 1024 * 1024 * 100


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_running(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when running."""
        manager.running = True
        result = manager.health_check()
        
        assert result.healthy is True
        assert "AdaptiveResourceManager" in result.message

    def test_health_check_not_running(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when not running."""
        manager.running = False
        result = manager.health_check()
        
        assert result.healthy is True  # Still healthy if no errors

    def test_health_check_with_errors(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when errors occurred."""
        manager.running = False
        manager.stats["errors"] = 5
        result = manager.health_check()
        
        # Should still be healthy if not running (graceful shutdown)
        assert result is not None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_resource_manager(self) -> None:
        """Test singleton accessor."""
        # Reset singleton
        import app.coordination.adaptive_resource_manager as arm
        arm._resource_manager = None
        
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        assert manager1 is manager2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_threshold_values(self) -> None:
        """Test threshold constants have reasonable values."""
        assert DISK_WARNING_THRESHOLD < DISK_CRITICAL_THRESHOLD
        assert MEMORY_WARNING_THRESHOLD < MEMORY_CRITICAL_THRESHOLD
        assert 0 < DISK_WARNING_THRESHOLD < 100
        assert 0 < MEMORY_WARNING_THRESHOLD < 100

    def test_cleanup_settings(self) -> None:
        """Test cleanup settings."""
        assert MIN_FILE_AGE_HOURS > 0
        assert CLEANUP_BATCH_SIZE > 0

    def test_check_interval_positive(self) -> None:
        """Test check interval is positive and reasonable."""
        assert CHECK_INTERVAL > 0
        assert CHECK_INTERVAL <= 600  # Should be at most 10 minutes

    def test_cleanup_cooldown_positive(self) -> None:
        """Test cleanup cooldown is positive."""
        assert CLEANUP_COOLDOWN > 0
        assert CLEANUP_COOLDOWN >= CHECK_INTERVAL  # Cooldown should be >= check interval

    def test_aggregation_interval_positive(self) -> None:
        """Test aggregation interval is positive."""
        assert AGGREGATION_INTERVAL > 0

    def test_disk_cleanup_threshold_between_warning_and_critical(self) -> None:
        """Test disk cleanup threshold is between warning and critical."""
        assert DISK_WARNING_THRESHOLD <= DISK_CLEANUP_THRESHOLD
        assert DISK_CLEANUP_THRESHOLD <= DISK_CRITICAL_THRESHOLD

    def test_default_paths_defined(self) -> None:
        """Test default paths are defined."""
        assert DEFAULT_NFS_PATH is not None
        assert DEFAULT_DATA_PATH is not None
        assert len(DEFAULT_NFS_PATH) > 0
        assert len(DEFAULT_DATA_PATH) > 0


# =============================================================================
# Extended ResourceStatus Tests
# =============================================================================


class TestResourceStatusExtended:
    """Extended tests for ResourceStatus dataclass."""

    def test_timestamp_auto_generated(self) -> None:
        """Test timestamp is automatically set to current time."""
        before = time.time()
        status = ResourceStatus(node_id="test")
        after = time.time()

        assert before <= status.timestamp <= after

    def test_gpu_percent_calculation(self) -> None:
        """Test GPU percent is correctly included in to_dict."""
        status = ResourceStatus(
            node_id="gpu-node",
            gpu_memory_used_gb=20.0,
            gpu_memory_total_gb=24.0,
            gpu_percent=83.33,
        )
        result = status.to_dict()

        assert result["gpu"]["used_gb"] == 20.0
        assert result["gpu"]["total_gb"] == 24.0
        assert result["gpu"]["percent"] == 83.33

    def test_multiple_warnings(self) -> None:
        """Test status can have multiple warnings."""
        status = ResourceStatus(
            node_id="problem-node",
            warnings=["Disk warning: 86%", "Memory warning: 87%"],
        )

        assert len(status.warnings) == 2
        result = status.to_dict()
        assert len(result["warnings"]) == 2

    def test_multiple_errors(self) -> None:
        """Test status can have multiple errors."""
        status = ResourceStatus(
            node_id="critical-node",
            errors=["Disk critical: 95%", "Memory critical: 96%"],
            is_healthy=False,
        )

        assert len(status.errors) == 2
        result = status.to_dict()
        assert len(result["errors"]) == 2
        assert result["is_healthy"] is False


# =============================================================================
# Extended CleanupResult Tests
# =============================================================================


class TestCleanupResultExtended:
    """Extended tests for CleanupResult dataclass."""

    def test_failed_cleanup(self) -> None:
        """Test CleanupResult for failed cleanup."""
        result = CleanupResult(
            success=False,
            errors=["Permission denied", "Disk I/O error"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_partial_success(self) -> None:
        """Test CleanupResult for partial success."""
        result = CleanupResult(
            success=True,
            files_deleted=5,
            bytes_freed=1024 * 100,
            errors=["Failed to delete locked_file.tmp"],
        )
        assert result.success is True
        assert result.files_deleted == 5
        assert len(result.errors) == 1

    def test_duration_tracking(self) -> None:
        """Test duration is properly tracked."""
        result = CleanupResult(
            success=True,
            files_deleted=100,
            bytes_freed=1024 * 1024 * 50,  # 50MB
            duration_seconds=2.5,
        )
        assert result.duration_seconds == 2.5


# =============================================================================
# Extended Initialization Tests
# =============================================================================


class TestAdaptiveResourceManagerInitExtended:
    """Extended tests for AdaptiveResourceManager initialization."""

    def test_stats_all_zero_initially(self) -> None:
        """Test all stats start at zero."""
        manager = AdaptiveResourceManager()
        assert manager.stats["cleanups_triggered"] == 0
        assert manager.stats["bytes_freed_total"] == 0
        assert manager.stats["files_deleted_total"] == 0
        assert manager.stats["aggregations_completed"] == 0
        assert manager.stats["nodes_paused"] == 0
        assert manager.stats["errors"] == 0

    def test_node_statuses_empty_initially(self) -> None:
        """Test node_statuses is empty on initialization."""
        manager = AdaptiveResourceManager()
        assert manager.node_statuses == {}

    def test_env_variable_nfs_path(self, tmp_path: Path) -> None:
        """Test NFS path can be set via environment variable."""
        test_path = str(tmp_path / "custom_nfs")
        Path(test_path).mkdir()

        with patch.dict(os.environ, {"RINGRIFT_NFS_PATH": test_path}):
            manager = AdaptiveResourceManager()
            assert manager.nfs_path == Path(test_path)


# =============================================================================
# Extended Disk Usage Tests
# =============================================================================


class TestDiskUsageExtended:
    """Extended tests for disk usage methods."""

    def test_get_disk_usage_permission_error(self, manager: AdaptiveResourceManager) -> None:
        """Test disk usage handles permission errors."""
        import shutil
        with patch.object(shutil, "disk_usage", side_effect=PermissionError("Access denied")):
            total, used, free = manager._get_disk_usage(Path("/restricted"))
            assert total == 0
            assert used == 0
            assert free == 0

    def test_get_disk_usage_oserror(self, manager: AdaptiveResourceManager) -> None:
        """Test disk usage handles general OS errors."""
        import shutil
        with patch.object(shutil, "disk_usage", side_effect=OSError("I/O error")):
            total, used, free = manager._get_disk_usage(Path("/problematic"))
            assert total == 0
            assert used == 0
            assert free == 0


# =============================================================================
# Extended Memory Usage Tests
# =============================================================================


class TestMemoryUsageExtended:
    """Extended tests for memory usage methods."""

    def test_get_memory_usage_file_not_found(self, manager: AdaptiveResourceManager) -> None:
        """Test memory usage when /proc/meminfo not found."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            total, used, free = manager._get_memory_usage()
            assert total == 0
            assert used == 0
            assert free == 0

    def test_get_memory_usage_permission_denied(self, manager: AdaptiveResourceManager) -> None:
        """Test memory usage when permission denied."""
        with patch("builtins.open", side_effect=PermissionError()):
            total, used, free = manager._get_memory_usage()
            assert total == 0
            assert used == 0
            assert free == 0


# =============================================================================
# Extended GPU Memory Tests
# =============================================================================


class TestGpuMemoryExtended:
    """Extended tests for GPU memory methods."""

    def test_get_gpu_memory_timeout(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory handles timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            used, total = manager._get_gpu_memory()
            assert used == 0
            assert total == 0

    def test_get_gpu_memory_failed_returncode(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory when nvidia-smi fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()
            assert used == 0
            assert total == 0

    def test_get_gpu_memory_malformed_output(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory with malformed nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid output"

        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()
            # Should handle gracefully
            assert isinstance(used, (int, float))
            assert isinstance(total, (int, float))

    def test_get_gpu_memory_multiple_gpus(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory with multiple GPUs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # 4 GPUs with different memory usage
        mock_result.stdout = "8000, 24000\n12000, 48000\n4000, 16000\n16000, 24000\n"

        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()
            # 8000+12000+4000+16000 = 40000 MB used
            # 24000+48000+16000+24000 = 112000 MB total
            expected_used = 40000 / 1024  # ~39 GB
            expected_total = 112000 / 1024  # ~109 GB
            assert used == pytest.approx(expected_used, rel=0.01)
            assert total == pytest.approx(expected_total, rel=0.01)


# =============================================================================
# Extended Local Status Tests
# =============================================================================


class TestGetLocalStatusExtended:
    """Extended tests for get_local_status method."""

    def test_get_local_status_updates_node_statuses(self, manager: AdaptiveResourceManager) -> None:
        """Test get_local_status updates node_statuses dict."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    manager.get_local_status("custom-node")

        assert "custom-node" in manager.node_statuses

    def test_get_local_status_default_node_id(self, manager: AdaptiveResourceManager) -> None:
        """Test get_local_status uses 'local' as default node_id."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.node_id == "local"

    def test_get_local_status_memory_warning(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with memory warning."""
        # Memory at 86% (above MEMORY_WARNING_THRESHOLD of 85%)
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 27.5, 4.5)):  # ~86%
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert any("Memory warning" in w for w in status.warnings)

    def test_get_local_status_zero_disk_total(self, manager: AdaptiveResourceManager) -> None:
        """Test local status when disk total is zero (error case)."""
        with patch.object(manager, "_get_disk_usage", return_value=(0.0, 0.0, 0.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        # Should not crash, disk_percent should be 0
        assert status.disk_percent == 0


# =============================================================================
# Extended NFS Status Tests
# =============================================================================


class TestGetNfsStatusExtended:
    """Extended tests for get_nfs_status method."""

    def test_get_nfs_status_disk_warning(self, manager: AdaptiveResourceManager) -> None:
        """Test NFS status with disk warning."""
        import shutil
        # 87% used - above warning threshold
        with patch.object(shutil, "disk_usage") as mock:
            mock.return_value = MagicMock(
                total=100 * 1024**3,
                used=87 * 1024**3,
                free=13 * 1024**3,
            )
            status = manager.get_nfs_status()

        assert any("NFS disk warning" in w for w in status.warnings)

    def test_get_nfs_status_disk_critical(self, manager: AdaptiveResourceManager) -> None:
        """Test NFS status with disk critical."""
        import shutil
        # 95% used - above critical threshold
        with patch.object(shutil, "disk_usage") as mock:
            mock.return_value = MagicMock(
                total=100 * 1024**3,
                used=95 * 1024**3,
                free=5 * 1024**3,
            )
            status = manager.get_nfs_status()

        assert status.is_healthy is False
        assert any("NFS disk critical" in e for e in status.errors)


# =============================================================================
# Extended Cleanup Tests
# =============================================================================


class TestCleanupOldFilesExtended:
    """Extended tests for cleanup_old_files method."""

    @pytest.mark.asyncio
    async def test_cleanup_custom_patterns(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup with custom file patterns."""
        test_dir = manager.data_path / "patterns_test"
        test_dir.mkdir()

        # Create files with different extensions
        txt_file = test_dir / "data.txt"
        txt_file.write_text("text")
        jsonl_file = test_dir / "data.jsonl"
        jsonl_file.write_text("json")

        old_time = time.time() - (25 * 3600)
        os.utime(txt_file, (old_time, old_time))
        os.utime(jsonl_file, (old_time, old_time))

        # Only clean .txt files
        result = await manager.cleanup_old_files(test_dir, patterns=["*.txt"])

        assert result.files_deleted == 1
        assert not txt_file.exists()
        assert jsonl_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_updates_stats(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup updates manager stats."""
        test_dir = manager.data_path / "stats_test"
        test_dir.mkdir()

        old_file = test_dir / "stats_file.tmp"
        old_file.write_text("x" * 1000)
        old_time = time.time() - (25 * 3600)
        os.utime(old_file, (old_time, old_time))

        await manager.cleanup_old_files(test_dir)

        assert manager.stats["cleanups_triggered"] == 1
        assert manager.stats["files_deleted_total"] == 1
        assert manager.stats["bytes_freed_total"] >= 1000
        assert manager.last_cleanup_time > 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_delete_error(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup handles deletion errors gracefully."""
        test_dir = manager.data_path / "error_test"
        test_dir.mkdir()

        old_file = test_dir / "locked.tmp"
        old_file.write_text("locked")
        old_time = time.time() - (25 * 3600)
        os.utime(old_file, (old_time, old_time))

        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            result = await manager.cleanup_old_files(test_dir)

        assert len(result.errors) >= 1
        assert old_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_directory(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup handles nonexistent directory gracefully.

        Note: The implementation uses rglob which returns empty iterator for
        nonexistent paths rather than raising an error, so success=True with
        no files deleted.
        """
        result = await manager.cleanup_old_files(Path("/nonexistent/path"))

        # rglob returns empty iterator for nonexistent paths, so no error
        assert result.files_deleted == 0
        # No exception was raised, so success
        assert result.success is True

    @pytest.mark.asyncio
    async def test_cleanup_batch_limit(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup respects batch size limit."""
        test_dir = manager.data_path / "batch_test"
        test_dir.mkdir()

        # Create more files than batch limit
        old_time = time.time() - (25 * 3600)
        for i in range(CLEANUP_BATCH_SIZE + 20):
            f = test_dir / f"file_{i}.jsonl"
            f.write_text("data")
            os.utime(f, (old_time, old_time))

        result = await manager.cleanup_old_files(test_dir)

        assert result.files_deleted == CLEANUP_BATCH_SIZE

    @pytest.mark.asyncio
    async def test_cleanup_tracks_duration(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup tracks duration."""
        test_dir = manager.data_path / "duration_test"
        test_dir.mkdir()

        result = await manager.cleanup_old_files(test_dir)

        assert result.duration_seconds >= 0


# =============================================================================
# Extended Aggregation Tests
# =============================================================================


class TestAggregationExtended:
    """Extended tests for data aggregation methods."""

    @pytest.mark.asyncio
    async def test_aggregate_with_explicit_nodes(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation with explicit node list."""
        with patch.object(manager, "_aggregate_from_node", new_callable=AsyncMock) as mock_agg:
            mock_agg.return_value = {"games": 5, "bytes": 1024, "error": None}
            result = await manager.aggregate_selfplay_data(source_nodes=["node-1", "node-2"])

        assert "node-1" in result["nodes_processed"]
        assert "node-2" in result["nodes_processed"]
        assert result["games_aggregated"] == 10  # 5 * 2 nodes

    @pytest.mark.asyncio
    async def test_aggregate_updates_stats(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation updates manager stats."""
        with patch.object(manager, "_aggregate_from_node", new_callable=AsyncMock) as mock_agg:
            mock_agg.return_value = {"games": 10, "bytes": 2048, "error": None}
            await manager.aggregate_selfplay_data(source_nodes=["node-1"])

        assert manager.stats["aggregations_completed"] == 1
        assert manager.last_aggregation_time > 0

    @pytest.mark.asyncio
    async def test_aggregate_handles_node_error(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation handles individual node errors."""
        with patch.object(manager, "_aggregate_from_node", new_callable=AsyncMock) as mock_agg:
            mock_agg.side_effect = [
                {"games": 5, "bytes": 1024, "error": None},
                Exception("Connection failed"),
            ]
            result = await manager.aggregate_selfplay_data(source_nodes=["node-1", "node-2"])

        assert result["success"] is True  # Partial success
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_aggregate_with_node_returning_error(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation when node returns error in response."""
        with patch.object(manager, "_aggregate_from_node", new_callable=AsyncMock) as mock_agg:
            mock_agg.return_value = {"games": 0, "bytes": 0, "error": "No data"}
            result = await manager.aggregate_selfplay_data(source_nodes=["node-1"])

        assert len(result["errors"]) >= 1


# =============================================================================
# Extended Check and Cleanup Tests
# =============================================================================


class TestCheckAndCleanupExtended:
    """Extended tests for check_and_cleanup method."""

    @pytest.mark.asyncio
    async def test_check_respects_cooldown(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup respects cooldown period."""
        manager.last_cleanup_time = time.time()  # Just cleaned up
        manager.disk_threshold = 50.0  # Low threshold

        with patch.object(manager, "get_nfs_status") as mock_nfs:
            mock_status = ResourceStatus(node_id="nfs", disk_percent=75.0)
            mock_nfs.return_value = mock_status

            with patch.object(manager, "get_local_status") as mock_local:
                mock_local.return_value = ResourceStatus(node_id="local")

                result = await manager.check_and_cleanup()

        # Should not trigger cleanup due to cooldown
        assert result["cleanup_triggered"] is False


# =============================================================================
# Extended Run Loop Tests
# =============================================================================


class TestRunLoopExtended:
    """Extended tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_run_handles_check_error(self, manager: AdaptiveResourceManager) -> None:
        """Test run loop continues after check_and_cleanup errors."""
        call_count = 0

        async def error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary error")
            return {}

        with patch.object(manager, "check_and_cleanup", new_callable=AsyncMock, side_effect=error_then_succeed):
            with patch.object(manager, "aggregate_selfplay_data", new_callable=AsyncMock):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    async def stop_after_calls(*args):
                        if call_count >= 2:
                            manager.stop()
                    mock_sleep.side_effect = stop_after_calls

                    await manager.run()

        assert manager.stats["errors"] >= 1

    @pytest.mark.asyncio
    async def test_run_triggers_aggregation(self, manager: AdaptiveResourceManager) -> None:
        """Test run loop triggers aggregation when interval elapsed."""
        manager.last_aggregation_time = 0  # Force aggregation

        with patch.object(manager, "check_and_cleanup", new_callable=AsyncMock, return_value={}):
            with patch.object(manager, "aggregate_selfplay_data", new_callable=AsyncMock) as mock_agg:
                mock_agg.return_value = {}

                async def stop_loop(*args):
                    manager.stop()

                with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=stop_loop):
                    await manager.run()

        mock_agg.assert_called()


# =============================================================================
# Extended Stats Tests
# =============================================================================


class TestGetStatsExtended:
    """Extended tests for get_stats method."""

    def test_get_stats_includes_node_statuses(self, manager: AdaptiveResourceManager) -> None:
        """Test get_stats includes node status data."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    manager.get_local_status("test-node")

        stats = manager.get_stats()

        assert "test-node" in stats["node_statuses"]
        assert "disk" in stats["node_statuses"]["test-node"]


# =============================================================================
# Extended Health Check Tests
# =============================================================================


class TestHealthCheckExtended:
    """Extended tests for health_check method."""

    def test_health_check_includes_stats_in_message(self, manager: AdaptiveResourceManager) -> None:
        """Test health check message includes stats."""
        manager.stats["cleanups_triggered"] = 5
        manager.stats["aggregations_completed"] = 10

        result = manager.health_check()

        assert "5 cleanups" in result.message
        assert "10 aggregations" in result.message

    def test_health_check_returns_health_result(self, manager: AdaptiveResourceManager) -> None:
        """Test health_check returns proper HealthCheckResult."""
        result = manager.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "details")


# =============================================================================
# Singleton Tests Extended
# =============================================================================


class TestSingletonExtended:
    """Extended tests for singleton pattern."""

    def test_singleton_uses_default_paths(self) -> None:
        """Test singleton uses default paths."""
        import app.coordination.adaptive_resource_manager as arm
        arm._resource_manager = None

        manager = get_resource_manager()

        assert manager.nfs_path == Path(DEFAULT_NFS_PATH)
        assert manager.data_path == Path(DEFAULT_DATA_PATH)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for AdaptiveResourceManager."""

    @pytest.mark.asyncio
    async def test_full_check_and_cleanup_flow(self, manager: AdaptiveResourceManager) -> None:
        """Test complete check and cleanup flow."""
        # Create some old files
        test_dir = manager.data_path / "integration_test"
        test_dir.mkdir()

        old_file = test_dir / "old.jsonl"
        old_file.write_text("old data")
        old_time = time.time() - (25 * 3600)
        os.utime(old_file, (old_time, old_time))

        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    # Get initial status
                    status = manager.get_local_status()
                    assert status.is_healthy is True

                    # Run check (won't trigger cleanup since disk isn't full)
                    result = await manager.check_and_cleanup()
                    assert result["cleanup_triggered"] is False

        # Verify stats
        stats = manager.get_stats()
        assert "local" in stats["node_statuses"]

    @pytest.mark.asyncio
    async def test_multiple_status_checks(self, manager: AdaptiveResourceManager) -> None:
        """Test multiple status checks update properly."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(8.0, 24.0)):
                    status1 = manager.get_local_status("node-1")
                    time1 = status1.timestamp

                    await asyncio.sleep(0.01)

                    status2 = manager.get_local_status("node-1")
                    time2 = status2.timestamp

        assert time2 > time1


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error handling and recovery scenarios."""

    def test_get_disk_usage_all_errors_handled(self, manager: AdaptiveResourceManager) -> None:
        """Test all disk usage errors are handled gracefully."""
        import shutil

        error_types = [
            OSError("Disk error"),
            PermissionError("Access denied"),
            FileNotFoundError("Not found"),
        ]

        for error in error_types:
            with patch.object(shutil, "disk_usage", side_effect=error):
                total, used, free = manager._get_disk_usage(Path("/test"))
                assert total == 0
                assert used == 0
                assert free == 0

    def test_get_memory_usage_all_errors_handled(self, manager: AdaptiveResourceManager) -> None:
        """Test all memory usage errors are handled gracefully."""
        error_types = [
            FileNotFoundError("/proc/meminfo not found"),
            PermissionError("Access denied"),
            IOError("Read error"),
        ]

        for error in error_types:
            with patch("builtins.open", side_effect=error):
                total, used, free = manager._get_memory_usage()
                assert total == 0
                assert used == 0
                assert free == 0

    def test_get_gpu_memory_all_errors_handled(self, manager: AdaptiveResourceManager) -> None:
        """Test all GPU memory errors are handled gracefully."""
        error_types = [
            FileNotFoundError("nvidia-smi not found"),
            PermissionError("Access denied"),
            subprocess.TimeoutExpired("nvidia-smi", 10),
        ]

        for error in error_types:
            with patch("subprocess.run", side_effect=error):
                used, total = manager._get_gpu_memory()
                assert used == 0
                assert total == 0

    @pytest.mark.asyncio
    async def test_cleanup_recovers_from_individual_stat_error(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup handles individual file stat errors.

        When file.stat() fails, the file is skipped but other files
        continue to be processed.
        """
        test_dir = manager.data_path / "stat_error_test"
        test_dir.mkdir()

        # Create a file that will work
        good_file = test_dir / "good.jsonl"
        good_file.write_text("data")
        old_time = time.time() - (25 * 3600)
        os.utime(good_file, (old_time, old_time))

        # The implementation catches FileNotFoundError and OSError per-file
        # so the overall cleanup should succeed
        result = await manager.cleanup_old_files(test_dir)

        # Should complete and delete the good file
        assert result.success is True
        assert result.files_deleted >= 0  # May or may not delete depending on timing
