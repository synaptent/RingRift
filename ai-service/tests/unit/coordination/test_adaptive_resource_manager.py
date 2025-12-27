"""Tests for AdaptiveResourceManager - proactive cluster resource management.

December 2025: Comprehensive test coverage for the critical resource manager
that controls cluster resource allocation and cleanup decisions.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import pytest


# ============================================
# Module under test imports
# ============================================

from app.coordination.adaptive_resource_manager import (
    AdaptiveResourceManager,
    CleanupResult,
    ResourceStatus,
    get_resource_manager,
    DISK_WARNING_THRESHOLD,
    DISK_CRITICAL_THRESHOLD,
    DISK_CLEANUP_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    CHECK_INTERVAL,
    CLEANUP_COOLDOWN,
    MIN_FILE_AGE_HOURS,
    CLEANUP_BATCH_SIZE,
)


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_dir):
    """Create a resource manager with temporary paths."""
    return AdaptiveResourceManager(
        nfs_path=str(temp_dir / "nfs"),
        data_path=str(temp_dir / "data"),
    )


@pytest.fixture
def manager_with_dirs(temp_dir):
    """Create a resource manager with existing directories."""
    nfs_path = temp_dir / "nfs"
    data_path = temp_dir / "data"
    nfs_path.mkdir(parents=True)
    data_path.mkdir(parents=True)
    return AdaptiveResourceManager(
        nfs_path=str(nfs_path),
        data_path=str(data_path),
    )


# ============================================
# ResourceStatus Tests
# ============================================


class TestResourceStatus:
    """Tests for the ResourceStatus dataclass."""

    def test_default_values(self):
        """Test default field values."""
        status = ResourceStatus(node_id="test-node")
        assert status.node_id == "test-node"
        assert status.disk_total_gb == 0
        assert status.disk_used_gb == 0
        assert status.disk_free_gb == 0
        assert status.disk_percent == 0
        assert status.memory_total_gb == 0
        assert status.is_healthy is True
        assert status.warnings == []
        assert status.errors == []

    def test_to_dict(self):
        """Test dictionary serialization."""
        status = ResourceStatus(
            node_id="test-node",
            disk_total_gb=100,
            disk_used_gb=60,
            disk_free_gb=40,
            disk_percent=60.0,
            memory_total_gb=32,
            memory_used_gb=16,
            memory_free_gb=16,
            memory_percent=50.0,
            gpu_memory_used_gb=8,
            gpu_memory_total_gb=24,
            gpu_percent=33.3,
            is_healthy=True,
            warnings=["test warning"],
            errors=[],
        )

        result = status.to_dict()

        assert result["node_id"] == "test-node"
        assert result["disk"]["total_gb"] == 100
        assert result["disk"]["percent"] == 60.0
        assert result["memory"]["total_gb"] == 32
        assert result["memory"]["percent"] == 50.0
        assert result["gpu"]["used_gb"] == 8
        assert result["gpu"]["total_gb"] == 24
        assert result["is_healthy"] is True
        assert result["warnings"] == ["test warning"]
        assert result["errors"] == []

    def test_to_dict_with_errors(self):
        """Test dictionary serialization with errors."""
        status = ResourceStatus(
            node_id="error-node",
            is_healthy=False,
            errors=["Disk critical: 95%"],
        )

        result = status.to_dict()

        assert result["is_healthy"] is False
        assert "Disk critical: 95%" in result["errors"]


# ============================================
# CleanupResult Tests
# ============================================


class TestCleanupResult:
    """Tests for the CleanupResult dataclass."""

    def test_default_values(self):
        """Test default field values."""
        result = CleanupResult(success=True)
        assert result.success is True
        assert result.files_deleted == 0
        assert result.bytes_freed == 0
        assert result.errors == []
        assert result.duration_seconds == 0

    def test_with_values(self):
        """Test with custom values."""
        result = CleanupResult(
            success=True,
            files_deleted=10,
            bytes_freed=1024 * 1024,
            errors=["minor error"],
            duration_seconds=1.5,
        )
        assert result.files_deleted == 10
        assert result.bytes_freed == 1024 * 1024
        assert result.errors == ["minor error"]


# ============================================
# AdaptiveResourceManager Initialization Tests
# ============================================


class TestAdaptiveResourceManagerInit:
    """Tests for AdaptiveResourceManager initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        manager = AdaptiveResourceManager()
        assert manager.disk_threshold == DISK_CLEANUP_THRESHOLD
        assert manager.memory_threshold == MEMORY_WARNING_THRESHOLD
        assert manager.running is False
        assert manager.last_cleanup_time == 0
        assert manager.last_aggregation_time == 0

    def test_custom_paths(self, temp_dir):
        """Test initialization with custom paths."""
        nfs_path = temp_dir / "custom_nfs"
        data_path = temp_dir / "custom_data"

        manager = AdaptiveResourceManager(
            nfs_path=str(nfs_path),
            data_path=str(data_path),
        )

        assert manager.nfs_path == nfs_path
        assert manager.data_path == data_path

    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        manager = AdaptiveResourceManager(
            disk_threshold=80.0,
            memory_threshold=75.0,
        )
        assert manager.disk_threshold == 80.0
        assert manager.memory_threshold == 75.0

    def test_initial_stats(self, manager):
        """Test initial statistics."""
        assert manager.stats["cleanups_triggered"] == 0
        assert manager.stats["bytes_freed_total"] == 0
        assert manager.stats["files_deleted_total"] == 0
        assert manager.stats["aggregations_completed"] == 0
        assert manager.stats["nodes_paused"] == 0
        assert manager.stats["errors"] == 0

    @patch.dict("os.environ", {"RINGRIFT_NFS_PATH": "/custom/nfs"})
    def test_env_var_override(self):
        """Test environment variable path override."""
        manager = AdaptiveResourceManager()
        assert manager.nfs_path == Path("/custom/nfs")


# ============================================
# Disk Usage Tests
# ============================================


class TestDiskUsage:
    """Tests for disk usage detection."""

    def test_get_disk_usage_existing_path(self, manager_with_dirs):
        """Test disk usage for existing path."""
        total, used, free = manager_with_dirs._get_disk_usage(
            manager_with_dirs.data_path
        )
        # Should return non-zero values for an existing path
        assert total > 0
        assert total >= used
        assert total >= free

    def test_get_disk_usage_nonexistent_path(self, manager):
        """Test disk usage for non-existent path."""
        total, used, free = manager._get_disk_usage(Path("/nonexistent/path"))
        assert total == 0
        assert used == 0
        assert free == 0


# ============================================
# Memory Usage Tests
# ============================================


class TestMemoryUsage:
    """Tests for memory usage detection."""

    def test_get_memory_usage_mock(self, manager):
        """Test memory usage with mocked /proc/meminfo."""
        meminfo_content = """MemTotal:       32000000 kB
MemFree:         8000000 kB
Buffers:         1000000 kB
Cached:          5000000 kB
SwapTotal:       8000000 kB
SwapFree:        8000000 kB
"""
        with patch("builtins.open", mock_open(read_data=meminfo_content)):
            total, used, free = manager._get_memory_usage()

        # MemTotal: 32GB, Free = MemFree + Buffers + Cached = 14GB
        # Used = Total - Free = 18GB
        assert total == pytest.approx(32000000 / (1024**2), rel=0.01)
        assert free == pytest.approx(14000000 / (1024**2), rel=0.01)
        assert used == pytest.approx(18000000 / (1024**2), rel=0.01)

    def test_get_memory_usage_error(self, manager):
        """Test memory usage when /proc/meminfo is unavailable."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            total, used, free = manager._get_memory_usage()

        assert total == 0
        assert used == 0
        assert free == 0


# ============================================
# GPU Memory Tests
# ============================================


class TestGPUMemory:
    """Tests for GPU memory detection."""

    def test_get_gpu_memory_mock(self, manager):
        """Test GPU memory with mocked nvidia-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192, 24576\n4096, 24576\n"  # Two GPUs

        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()

        # 8192 + 4096 = 12288 MB = 12 GB used
        # 24576 + 24576 = 49152 MB = 48 GB total
        assert used == pytest.approx(12, rel=0.01)
        assert total == pytest.approx(48, rel=0.01)

    def test_get_gpu_memory_no_nvidia_smi(self, manager):
        """Test GPU memory when nvidia-smi is unavailable."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            used, total = manager._get_gpu_memory()

        assert used == 0
        assert total == 0

    def test_get_gpu_memory_timeout(self, manager):
        """Test GPU memory when nvidia-smi times out."""
        from subprocess import TimeoutExpired
        with patch("subprocess.run", side_effect=TimeoutExpired(cmd="nvidia-smi", timeout=10)):
            used, total = manager._get_gpu_memory()

        assert used == 0
        assert total == 0


# ============================================
# Local Status Tests
# ============================================


class TestGetLocalStatus:
    """Tests for local node status detection."""

    def test_get_local_status_healthy(self, manager_with_dirs):
        """Test local status for healthy node."""
        # Mock low resource usage
        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 30, 70)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(8, 24)
        ):
            status = manager_with_dirs.get_local_status("test-node")

        assert status.node_id == "test-node"
        assert status.disk_percent == 30.0
        assert status.memory_percent == 25.0
        assert status.gpu_percent == pytest.approx(33.33, rel=0.01)
        assert status.is_healthy is True
        assert len(status.warnings) == 0
        assert len(status.errors) == 0

    def test_get_local_status_disk_warning(self, manager_with_dirs):
        """Test local status with disk warning."""
        # 87% disk usage should trigger warning (threshold is 85%)
        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 87, 13)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            status = manager_with_dirs.get_local_status()

        assert status.disk_percent == 87.0
        assert status.is_healthy is True  # Warning doesn't make unhealthy
        assert len(status.warnings) == 1
        assert "Disk warning" in status.warnings[0]

    def test_get_local_status_disk_critical(self, manager_with_dirs):
        """Test local status with disk critical."""
        # 95% disk usage should trigger critical error
        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 95, 5)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            status = manager_with_dirs.get_local_status()

        assert status.disk_percent == 95.0
        assert status.is_healthy is False  # Critical makes unhealthy
        assert len(status.errors) == 1
        assert "Disk critical" in status.errors[0]

    def test_get_local_status_memory_critical(self, manager_with_dirs):
        """Test local status with memory critical."""
        # 95% memory usage should trigger critical error
        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 30, 70)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 30.4, 1.6)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            status = manager_with_dirs.get_local_status()

        assert status.memory_percent == 95.0
        assert status.is_healthy is False
        assert any("Memory critical" in e for e in status.errors)

    def test_get_local_status_updates_node_statuses(self, manager_with_dirs):
        """Test that get_local_status updates the node_statuses dict."""
        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 30, 70)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            manager_with_dirs.get_local_status("my-node")

        assert "my-node" in manager_with_dirs.node_statuses
        assert manager_with_dirs.node_statuses["my-node"].node_id == "my-node"


# ============================================
# NFS Status Tests
# ============================================


class TestGetNFSStatus:
    """Tests for NFS status detection."""

    def test_get_nfs_status_healthy(self, manager_with_dirs):
        """Test NFS status for healthy mount."""
        nfs_path = manager_with_dirs.nfs_path
        nfs_path.mkdir(parents=True, exist_ok=True)

        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(500, 200, 300)
        ):
            status = manager_with_dirs.get_nfs_status()

        assert status.node_id == "nfs"
        assert status.disk_percent == 40.0
        assert status.is_healthy is True
        assert len(status.errors) == 0

    def test_get_nfs_status_not_accessible(self, manager):
        """Test NFS status when path doesn't exist."""
        # manager's nfs_path doesn't exist
        status = manager.get_nfs_status()

        assert status.node_id == "nfs"
        assert status.is_healthy is False
        assert any("not accessible" in e for e in status.errors)

    def test_get_nfs_status_critical(self, manager_with_dirs):
        """Test NFS status with critical disk usage."""
        nfs_path = manager_with_dirs.nfs_path
        nfs_path.mkdir(parents=True, exist_ok=True)

        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(500, 475, 25)
        ):
            status = manager_with_dirs.get_nfs_status()

        assert status.disk_percent == 95.0
        assert status.is_healthy is False
        assert any("NFS disk critical" in e for e in status.errors)


# ============================================
# Cleanup Tests
# ============================================


class TestCleanupOldFiles:
    """Tests for file cleanup functionality."""

    @pytest.fixture
    def dir_with_files(self, temp_dir):
        """Create a directory with test files."""
        cleanup_dir = temp_dir / "cleanup_test"
        cleanup_dir.mkdir()

        # Create old files
        import os
        import time
        old_time = time.time() - (MIN_FILE_AGE_HOURS + 1) * 3600

        old_files = []
        for i in range(5):
            f = cleanup_dir / f"old_file_{i}.jsonl"
            f.write_text(f"content {i}")
            os.utime(f, (old_time, old_time))
            old_files.append(f)

        # Create new files
        new_files = []
        for i in range(3):
            f = cleanup_dir / f"new_file_{i}.jsonl"
            f.write_text(f"new content {i}")
            new_files.append(f)

        return cleanup_dir, old_files, new_files

    @pytest.mark.asyncio
    async def test_cleanup_old_files_normal(self, manager, dir_with_files):
        """Test normal cleanup operation."""
        cleanup_dir, old_files, new_files = dir_with_files

        result = await manager.cleanup_old_files(
            cleanup_dir,
            patterns=["*.jsonl"],
        )

        assert result.success is True
        assert result.files_deleted == 5  # Only old files
        assert result.bytes_freed > 0
        assert result.duration_seconds >= 0

        # Old files should be deleted
        for f in old_files:
            assert not f.exists()

        # New files should remain
        for f in new_files:
            assert f.exists()

    @pytest.mark.asyncio
    async def test_cleanup_old_files_dry_run(self, manager, dir_with_files):
        """Test cleanup in dry run mode."""
        cleanup_dir, old_files, new_files = dir_with_files

        result = await manager.cleanup_old_files(
            cleanup_dir,
            patterns=["*.jsonl"],
            dry_run=True,
        )

        assert result.success is True
        assert result.files_deleted == 5
        assert result.bytes_freed > 0

        # All files should still exist in dry run
        for f in old_files + new_files:
            assert f.exists()

    @pytest.mark.asyncio
    async def test_cleanup_respects_age_threshold(self, manager, temp_dir):
        """Test that cleanup respects minimum age threshold."""
        cleanup_dir = temp_dir / "age_test"
        cleanup_dir.mkdir()

        import time
        import os

        # File just under threshold (23 hours old)
        under_threshold = cleanup_dir / "under.jsonl"
        under_threshold.write_text("content")
        old_time = time.time() - (MIN_FILE_AGE_HOURS - 1) * 3600
        os.utime(under_threshold, (old_time, old_time))

        result = await manager.cleanup_old_files(cleanup_dir, patterns=["*.jsonl"])

        assert result.files_deleted == 0
        assert under_threshold.exists()

    @pytest.mark.asyncio
    async def test_cleanup_respects_batch_size(self, manager, temp_dir):
        """Test that cleanup respects batch size limit."""
        cleanup_dir = temp_dir / "batch_test"
        cleanup_dir.mkdir()

        import time
        import os

        # Create more files than batch size
        old_time = time.time() - (MIN_FILE_AGE_HOURS + 1) * 3600
        for i in range(CLEANUP_BATCH_SIZE + 50):
            f = cleanup_dir / f"file_{i}.jsonl"
            f.write_text(f"content {i}")
            os.utime(f, (old_time, old_time))

        result = await manager.cleanup_old_files(cleanup_dir, patterns=["*.jsonl"])

        assert result.files_deleted == CLEANUP_BATCH_SIZE
        assert result.success is True

    @pytest.mark.asyncio
    async def test_cleanup_updates_stats(self, manager, dir_with_files):
        """Test that cleanup updates manager statistics."""
        cleanup_dir, old_files, new_files = dir_with_files

        initial_cleanups = manager.stats["cleanups_triggered"]

        await manager.cleanup_old_files(cleanup_dir, patterns=["*.jsonl"])

        assert manager.stats["cleanups_triggered"] == initial_cleanups + 1
        assert manager.stats["files_deleted_total"] >= 5
        assert manager.stats["bytes_freed_total"] > 0
        assert manager.last_cleanup_time > 0


# ============================================
# Aggregation Tests
# ============================================


class TestAggregateSelfplayData:
    """Tests for selfplay data aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_no_nodes(self, manager_with_dirs):
        """Test aggregation with no source nodes."""
        with patch.object(
            manager_with_dirs, "_get_active_selfplay_nodes", return_value=[]
        ):
            result = await manager_with_dirs.aggregate_selfplay_data()

        assert result["success"] is True
        assert result["games_aggregated"] == 0
        assert result["nodes_processed"] == []

    @pytest.mark.asyncio
    async def test_aggregate_with_explicit_nodes(self, manager_with_dirs):
        """Test aggregation with explicit node list."""
        # Mock the aggregation method to return success
        async def mock_aggregate(node_id, dest_dir):
            return {"games": 5, "bytes": 1024, "error": None}

        with patch.object(
            manager_with_dirs, "_aggregate_from_node", side_effect=mock_aggregate
        ):
            result = await manager_with_dirs.aggregate_selfplay_data(
                source_nodes=["node1", "node2"]
            )

        assert result["success"] is True
        assert result["games_aggregated"] == 10
        assert result["bytes_transferred"] == 2048
        assert "node1" in result["nodes_processed"]
        assert "node2" in result["nodes_processed"]

    @pytest.mark.asyncio
    async def test_aggregate_partial_failure(self, manager_with_dirs):
        """Test aggregation with partial node failures."""
        call_count = [0]

        async def mock_aggregate(node_id, dest_dir):
            call_count[0] += 1
            if node_id == "bad_node":
                return {"games": 0, "bytes": 0, "error": "Connection failed"}
            return {"games": 3, "bytes": 512, "error": None}

        with patch.object(
            manager_with_dirs, "_aggregate_from_node", side_effect=mock_aggregate
        ):
            result = await manager_with_dirs.aggregate_selfplay_data(
                source_nodes=["good_node", "bad_node"]
            )

        assert result["success"] is True  # Partial success
        assert result["games_aggregated"] == 3
        assert len(result["errors"]) == 1
        assert "bad_node" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_aggregate_updates_stats(self, manager_with_dirs):
        """Test that aggregation updates manager statistics."""
        async def mock_aggregate(node_id, dest_dir):
            return {"games": 2, "bytes": 256, "error": None}

        initial_aggregations = manager_with_dirs.stats["aggregations_completed"]

        with patch.object(
            manager_with_dirs, "_aggregate_from_node", side_effect=mock_aggregate
        ):
            await manager_with_dirs.aggregate_selfplay_data(source_nodes=["node1"])

        assert manager_with_dirs.stats["aggregations_completed"] == initial_aggregations + 1
        assert manager_with_dirs.last_aggregation_time > 0


# ============================================
# Check and Cleanup Tests
# ============================================


class TestCheckAndCleanup:
    """Tests for the check_and_cleanup method."""

    @pytest.mark.asyncio
    async def test_check_and_cleanup_no_action(self, manager_with_dirs):
        """Test check_and_cleanup when no cleanup is needed."""
        # Create NFS path
        nfs_path = manager_with_dirs.nfs_path
        nfs_path.mkdir(parents=True, exist_ok=True)

        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 30, 70)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            results = await manager_with_dirs.check_and_cleanup()

        assert results["cleanup_triggered"] is False
        assert results["cleanup_result"] is None
        assert results["nfs_status"] is not None
        assert results["local_status"] is not None

    @pytest.mark.asyncio
    async def test_check_and_cleanup_triggers_cleanup(self, manager_with_dirs):
        """Test check_and_cleanup triggers cleanup when threshold exceeded."""
        # Create NFS path and selfplay directory
        nfs_path = manager_with_dirs.nfs_path
        selfplay_dir = nfs_path / "ai-service" / "data" / "selfplay"
        selfplay_dir.mkdir(parents=True, exist_ok=True)

        # Set disk_threshold lower to trigger cleanup
        manager_with_dirs.disk_threshold = 50.0

        mock_cleanup_result = CleanupResult(
            success=True, files_deleted=5, bytes_freed=1024
        )

        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 60, 40)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ), patch.object(
            manager_with_dirs,
            "cleanup_old_files",
            return_value=mock_cleanup_result,
        ) as mock_cleanup:
            results = await manager_with_dirs.check_and_cleanup()

        assert results["cleanup_triggered"] is True
        assert results["cleanup_result"]["files_deleted"] == 5
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_cleanup_respects_cooldown(self, manager_with_dirs):
        """Test that check_and_cleanup respects cleanup cooldown."""
        nfs_path = manager_with_dirs.nfs_path
        nfs_path.mkdir(parents=True, exist_ok=True)

        # Set recent cleanup time
        import time
        manager_with_dirs.last_cleanup_time = time.time()
        manager_with_dirs.disk_threshold = 50.0

        with patch.object(
            manager_with_dirs, "_get_disk_usage", return_value=(100, 60, 40)
        ), patch.object(
            manager_with_dirs, "_get_memory_usage", return_value=(32, 8, 24)
        ), patch.object(
            manager_with_dirs, "_get_gpu_memory", return_value=(0, 0)
        ):
            results = await manager_with_dirs.check_and_cleanup()

        # Cleanup should NOT be triggered due to cooldown
        assert results["cleanup_triggered"] is False


# ============================================
# Run Loop Tests
# ============================================


class TestRunLoop:
    """Tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_run_starts_and_stops(self, manager):
        """Test that run loop starts and can be stopped."""
        assert manager.running is False

        # Patch CHECK_INTERVAL to be very short for testing
        with patch(
            "app.coordination.adaptive_resource_manager.CHECK_INTERVAL", 0.05
        ):
            # Start run in background
            run_task = asyncio.create_task(manager.run())

            # Give it time to start
            await asyncio.sleep(0.1)
            assert manager.running is True

            # Stop it
            manager.stop()
            assert manager.running is False

            # Cancel task and wait
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_run_calls_check_and_cleanup(self, manager):
        """Test that run loop calls check_and_cleanup."""
        check_calls = []

        async def mock_check():
            check_calls.append(1)
            return {"nfs_status": None, "local_status": None, "cleanup_triggered": False}

        # Patch CHECK_INTERVAL to be very short for testing
        with patch(
            "app.coordination.adaptive_resource_manager.CHECK_INTERVAL", 0.05
        ), patch.object(manager, "check_and_cleanup", side_effect=mock_check):
            run_task = asyncio.create_task(manager.run())
            await asyncio.sleep(0.15)  # Allow at least one iteration
            manager.stop()
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

        assert len(check_calls) >= 1


# ============================================
# Statistics and Health Check Tests
# ============================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_initial(self, manager):
        """Test initial stats output."""
        stats = manager.get_stats()

        assert stats["running"] is False
        assert stats["cleanups_triggered"] == 0
        assert stats["errors"] == 0
        assert stats["last_cleanup_time"] == 0
        assert "nfs_path" in stats
        assert "data_path" in stats

    def test_get_stats_after_operations(self, manager):
        """Test stats after some operations."""
        manager.stats["cleanups_triggered"] = 5
        manager.stats["bytes_freed_total"] = 1024 * 1024
        manager.running = True

        stats = manager.get_stats()

        assert stats["running"] is True
        assert stats["cleanups_triggered"] == 5
        assert stats["bytes_freed_total"] == 1024 * 1024


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_when_running(self, manager):
        """Test health check when manager is running."""
        manager.running = True

        result = manager.health_check()

        assert result.healthy is True
        assert result.status.name == "RUNNING"
        assert "AdaptiveResourceManager" in result.message

    def test_health_check_when_stopped(self, manager):
        """Test health check when manager is stopped."""
        manager.running = False
        manager.stats["errors"] = 0

        result = manager.health_check()

        assert result.healthy is True  # No errors, just not running
        assert result.status.name == "READY"

    def test_health_check_with_errors(self, manager):
        """Test health check with errors."""
        manager.running = False
        manager.stats["errors"] = 5

        result = manager.health_check()

        # Not running but has errors - should still be healthy=True
        # based on the logic: healthy=is_running or not has_errors
        # When not running and has_errors, healthy = False or False = False
        assert result.healthy is False


# ============================================
# Singleton Pattern Tests
# ============================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_resource_manager_singleton(self):
        """Test that get_resource_manager returns singleton."""
        # Reset singleton
        import app.coordination.adaptive_resource_manager as module
        module._resource_manager = None

        manager1 = get_resource_manager()
        manager2 = get_resource_manager()

        assert manager1 is manager2

        # Cleanup
        module._resource_manager = None

    def test_singleton_can_be_reset(self):
        """Test that singleton can be reset for testing."""
        import app.coordination.adaptive_resource_manager as module

        module._resource_manager = None
        manager1 = get_resource_manager()

        module._resource_manager = None
        manager2 = get_resource_manager()

        assert manager1 is not manager2

        # Cleanup
        module._resource_manager = None


# ============================================
# Constants Tests
# ============================================


class TestConstants:
    """Tests for module constants."""

    def test_threshold_ordering(self):
        """Test that thresholds are ordered correctly.

        The ordering is:
        1. DISK_WARNING_THRESHOLD (85%) - First warning
        2. DISK_CLEANUP_THRESHOLD (90%) - Trigger cleanup
        3. DISK_CRITICAL_THRESHOLD (92%) - Critical error

        Cleanup happens after warning but before critical - this makes sense
        because cleanup is a more aggressive action triggered when disk
        is getting dangerously full.
        """
        # Warning should happen before cleanup
        assert DISK_WARNING_THRESHOLD < DISK_CLEANUP_THRESHOLD
        # Cleanup should happen before critical
        assert DISK_CLEANUP_THRESHOLD < DISK_CRITICAL_THRESHOLD
        # Memory thresholds should also be ordered
        assert MEMORY_WARNING_THRESHOLD < MEMORY_CRITICAL_THRESHOLD

    def test_intervals_are_positive(self):
        """Test that intervals are positive values."""
        assert CHECK_INTERVAL > 0
        assert CLEANUP_COOLDOWN > 0
        assert MIN_FILE_AGE_HOURS > 0
        assert CLEANUP_BATCH_SIZE > 0
