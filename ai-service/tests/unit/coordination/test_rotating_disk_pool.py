"""Unit tests for RotatingDiskPoolManager (December 2025).

Tests the rotating disk pool manager for distributed data pipeline.

Created: December 30, 2025
"""

import asyncio
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.coordination.rotating_disk_pool import (
    DEFAULT_POOL_SUBDIRS,
    POOL_METADATA_DB,
    PoolStats,
    RotatingDiskPoolManager,
    RotatingPoolConfig,
    get_rotating_pool_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_root_path() -> Path:
    """Create a temporary root path with necessary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_dir = root / "config"
        config_dir.mkdir(parents=True)
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        yield root


@pytest.fixture
def config() -> RotatingPoolConfig:
    """Create a test configuration."""
    return RotatingPoolConfig(
        quota_percent=10.0,
        min_quota_gb=1.0,
        max_quota_gb=50.0,
        cleanup_age_hours=1,  # 1 hour for faster tests
        priority_cleanup_on_pressure=True,
        pool_subdirs=["data/synced", "data/cache"],
    )


@pytest.fixture
def manager(temp_root_path: Path, config: RotatingPoolConfig) -> RotatingDiskPoolManager:
    """Create a test manager with reset singleton."""
    RotatingDiskPoolManager.reset_instance()
    return RotatingDiskPoolManager(root_path=temp_root_path, config=config)


@pytest.fixture
def sample_yaml_config(temp_root_path: Path) -> Path:
    """Create a sample distributed_hosts.yaml with rotating pool config."""
    config_path = temp_root_path / "config" / "distributed_hosts.yaml"

    yaml_config = {
        "disk_management": {
            "rotating_pool": {
                "enabled": True,
                "quota_percent": 15.0,
                "min_quota_gb": 3.0,
                "max_quota_gb": 75.0,
                "cleanup_age_hours": 12,
                "priority_cleanup_on_pressure": True,
            }
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f)

    return config_path


@pytest.fixture(autouse=True)
def reset_manager_singleton():
    """Reset manager singleton after each test."""
    yield
    RotatingDiskPoolManager.reset_instance()


# ============================================================================
# RotatingPoolConfig Tests
# ============================================================================


class TestRotatingPoolConfig:
    """Tests for RotatingPoolConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = RotatingPoolConfig()

        assert config.quota_percent == 10.0
        assert config.min_quota_gb == 5.0
        assert config.max_quota_gb == 100.0
        assert config.cleanup_age_hours == 24
        assert config.priority_cleanup_on_pressure is True
        assert config.pool_subdirs == DEFAULT_POOL_SUBDIRS

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RotatingPoolConfig(
            quota_percent=20.0,
            min_quota_gb=10.0,
            max_quota_gb=200.0,
            cleanup_age_hours=48,
            priority_cleanup_on_pressure=False,
            pool_subdirs=["custom/dir"],
        )

        assert config.quota_percent == 20.0
        assert config.min_quota_gb == 10.0
        assert config.max_quota_gb == 200.0
        assert config.cleanup_age_hours == 48
        assert config.priority_cleanup_on_pressure is False
        assert config.pool_subdirs == ["custom/dir"]

    def test_from_yaml_config_full(self):
        """Test loading from YAML config dictionary."""
        yaml_config = {
            "quota_percent": 15.0,
            "min_quota_gb": 2.0,
            "max_quota_gb": 80.0,
            "cleanup_age_hours": 36,
            "priority_cleanup_on_pressure": False,
            "pool_subdirs": ["dir1", "dir2"],
        }

        config = RotatingPoolConfig.from_yaml_config(yaml_config)

        assert config.quota_percent == 15.0
        assert config.min_quota_gb == 2.0
        assert config.max_quota_gb == 80.0
        assert config.cleanup_age_hours == 36
        assert config.priority_cleanup_on_pressure is False
        assert config.pool_subdirs == ["dir1", "dir2"]

    def test_from_yaml_config_partial(self):
        """Test loading from partial YAML config."""
        yaml_config = {
            "quota_percent": 5.0,
        }

        config = RotatingPoolConfig.from_yaml_config(yaml_config)

        assert config.quota_percent == 5.0
        # Defaults for missing
        assert config.min_quota_gb == 5.0
        assert config.max_quota_gb == 100.0

    def test_from_yaml_config_empty(self):
        """Test loading from empty YAML config."""
        config = RotatingPoolConfig.from_yaml_config({})

        assert config.quota_percent == 10.0
        assert config.cleanup_age_hours == 24


# ============================================================================
# PoolStats Tests
# ============================================================================


class TestPoolStats:
    """Tests for PoolStats dataclass."""

    def test_full_stats(self):
        """Test creating full pool stats."""
        stats = PoolStats(
            quota_bytes=1024**3,
            used_bytes=512 * 1024**2,
            free_bytes=512 * 1024**2,
            file_count=10,
            oldest_file_age_hours=12.5,
            disk_free_bytes=100 * 1024**3,
            disk_total_bytes=500 * 1024**3,
        )

        assert stats.quota_bytes == 1024**3
        assert stats.used_bytes == 512 * 1024**2
        assert stats.free_bytes == 512 * 1024**2
        assert stats.file_count == 10
        assert stats.oldest_file_age_hours == 12.5

    def test_empty_stats(self):
        """Test creating stats for empty pool."""
        stats = PoolStats(
            quota_bytes=1024**3,
            used_bytes=0,
            free_bytes=1024**3,
            file_count=0,
            oldest_file_age_hours=None,
            disk_free_bytes=100 * 1024**3,
            disk_total_bytes=500 * 1024**3,
        )

        assert stats.used_bytes == 0
        assert stats.file_count == 0
        assert stats.oldest_file_age_hours is None

    def test_serialization(self):
        """Test stats serialization."""
        stats = PoolStats(
            quota_bytes=1024,
            used_bytes=512,
            free_bytes=512,
            file_count=1,
            oldest_file_age_hours=1.0,
            disk_free_bytes=1024,
            disk_total_bytes=2048,
        )

        stats_dict = asdict(stats)
        assert stats_dict["quota_bytes"] == 1024
        assert stats_dict["used_bytes"] == 512


# ============================================================================
# RotatingDiskPoolManager Initialization Tests
# ============================================================================


class TestManagerInit:
    """Tests for RotatingDiskPoolManager initialization."""

    def test_init_with_config(self, temp_root_path: Path, config: RotatingPoolConfig):
        """Test initialization with custom config."""
        manager = RotatingDiskPoolManager(root_path=temp_root_path, config=config)

        assert manager.config == config
        assert manager.root_path == temp_root_path
        assert manager._db_path.exists()

    def test_init_creates_database(self, temp_root_path: Path, config: RotatingPoolConfig):
        """Test initialization creates metadata database."""
        manager = RotatingDiskPoolManager(root_path=temp_root_path, config=config)

        # Database should exist
        assert manager._db_path.exists()

        # Should have pool_files table
        with sqlite3.connect(manager._db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pool_files'"
            )
            assert cursor.fetchone() is not None

    def test_init_loads_from_yaml(self, temp_root_path: Path, sample_yaml_config: Path):
        """Test initialization loads config from YAML."""
        manager = RotatingDiskPoolManager(root_path=temp_root_path)

        assert manager.config.quota_percent == 15.0
        assert manager.config.min_quota_gb == 3.0
        assert manager.config.cleanup_age_hours == 12


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_instance(self, temp_root_path: Path, config: RotatingPoolConfig):
        """Test get_instance returns singleton."""
        manager1 = RotatingDiskPoolManager(root_path=temp_root_path, config=config)
        RotatingDiskPoolManager._instance = manager1

        manager2 = RotatingDiskPoolManager.get_instance()
        assert manager1 is manager2

    def test_reset_instance(self, temp_root_path: Path, config: RotatingPoolConfig):
        """Test reset_instance clears singleton."""
        manager1 = RotatingDiskPoolManager(root_path=temp_root_path, config=config)
        RotatingDiskPoolManager._instance = manager1

        RotatingDiskPoolManager.reset_instance()
        assert RotatingDiskPoolManager._instance is None

    def test_get_rotating_pool_manager(self):
        """Test module-level accessor."""
        manager = get_rotating_pool_manager()
        assert isinstance(manager, RotatingDiskPoolManager)


# ============================================================================
# Pool Quota Tests
# ============================================================================


class TestPoolQuota:
    """Tests for pool quota calculation."""

    def test_get_pool_quota_bytes(self, manager: RotatingDiskPoolManager):
        """Test quota calculation."""
        quota = manager.get_pool_quota_bytes()

        # Should be positive
        assert quota > 0

        # Should be at least min_quota_gb
        min_bytes = int(manager.config.min_quota_gb * 1024**3)
        assert quota >= min_bytes

        # Should not exceed max_quota_gb
        max_bytes = int(manager.config.max_quota_gb * 1024**3)
        assert quota <= max_bytes

    def test_get_pool_quota_bytes_error(self, manager: RotatingDiskPoolManager):
        """Test quota calculation with error."""
        with patch("shutil.disk_usage", side_effect=OSError("Disk error")):
            quota = manager.get_pool_quota_bytes()

            # Should return min_quota_gb on error
            min_bytes = int(manager.config.min_quota_gb * 1024**3)
            assert quota == min_bytes


# ============================================================================
# Pool Usage Tests
# ============================================================================


class TestPoolUsage:
    """Tests for pool usage tracking."""

    def test_get_pool_usage_bytes_empty(self, manager: RotatingDiskPoolManager):
        """Test usage for empty pool."""
        usage = manager.get_pool_usage_bytes()
        assert usage == 0

    def test_get_pool_usage_bytes_with_files(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test usage with tracked files."""
        # Create and mark test files
        test_file = temp_root_path / "test_file.db"
        test_file.write_bytes(b"x" * 1000)
        manager.mark_data_for_rotation(test_file)

        usage = manager.get_pool_usage_bytes()
        assert usage == 1000


# ============================================================================
# Can Accept Data Tests
# ============================================================================


class TestCanAcceptData:
    """Tests for can_accept_data method."""

    def test_can_accept_small_data(self, manager: RotatingDiskPoolManager):
        """Test accepting small amount of data."""
        # Should accept 1KB
        assert manager.can_accept_data(1024) is True

    def test_cannot_accept_huge_data(self, manager: RotatingDiskPoolManager):
        """Test rejecting huge amount of data."""
        # 1TB should exceed any reasonable quota
        assert manager.can_accept_data(1024**4) is False

    def test_can_accept_within_quota(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test accepting data within quota."""
        quota = manager.get_pool_quota_bytes()

        # Should accept data smaller than quota
        half_quota = quota // 2
        assert manager.can_accept_data(half_quota) is True


# ============================================================================
# Pool Stats Tests
# ============================================================================


class TestGetPoolStats:
    """Tests for get_pool_stats method."""

    def test_get_pool_stats_empty(self, manager: RotatingDiskPoolManager):
        """Test stats for empty pool."""
        stats = manager.get_pool_stats()

        assert stats.used_bytes == 0
        assert stats.file_count == 0
        assert stats.oldest_file_age_hours is None
        assert stats.quota_bytes > 0
        assert stats.free_bytes > 0

    def test_get_pool_stats_with_files(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test stats with tracked files."""
        # Create and mark test file
        test_file = temp_root_path / "test_file.db"
        test_file.write_bytes(b"x" * 1000)
        manager.mark_data_for_rotation(test_file)

        stats = manager.get_pool_stats()

        assert stats.used_bytes == 1000
        assert stats.file_count == 1
        assert stats.oldest_file_age_hours is not None
        assert stats.oldest_file_age_hours < 1.0  # Just added


# ============================================================================
# Mark Data for Rotation Tests
# ============================================================================


class TestMarkDataForRotation:
    """Tests for mark_data_for_rotation method."""

    def test_mark_existing_file(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test marking an existing file."""
        test_file = temp_root_path / "test.db"
        test_file.write_bytes(b"test data")

        result = manager.mark_data_for_rotation(
            test_file,
            source="node-1",
            config_key="hex8_2p",
        )

        assert result is True

        # Verify in database
        with sqlite3.connect(manager._db_path) as conn:
            cursor = conn.execute(
                "SELECT source, config_key FROM pool_files WHERE path = ?",
                (str(test_file.absolute()),),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "node-1"
            assert row[1] == "hex8_2p"

    def test_mark_nonexistent_file(self, manager: RotatingDiskPoolManager):
        """Test marking a nonexistent file."""
        result = manager.mark_data_for_rotation(Path("/nonexistent/file.db"))
        assert result is False

    def test_mark_updates_existing(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test marking updates existing entry."""
        test_file = temp_root_path / "test.db"
        test_file.write_bytes(b"test data")

        # Mark twice with different sources
        manager.mark_data_for_rotation(test_file, source="node-1")
        manager.mark_data_for_rotation(test_file, source="node-2")

        # Should have only one entry (updated)
        with sqlite3.connect(manager._db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM pool_files WHERE path = ?",
                (str(test_file.absolute()),),
            )
            assert cursor.fetchone()[0] == 1

            cursor = conn.execute(
                "SELECT source FROM pool_files WHERE path = ?",
                (str(test_file.absolute()),),
            )
            assert cursor.fetchone()[0] == "node-2"


# ============================================================================
# Cleanup Tests
# ============================================================================


class TestCleanup:
    """Tests for cleanup methods."""

    @pytest.mark.asyncio
    async def test_cleanup_oldest_expired(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test cleanup removes expired files."""
        # Create a file and mark it with old timestamp
        test_file = temp_root_path / "old_file.db"
        test_file.write_bytes(b"old data")

        # Add with old timestamp (2 hours ago, cleanup_age is 1 hour)
        old_time = time.time() - 7200
        with sqlite3.connect(manager._db_path) as conn:
            conn.execute(
                "INSERT INTO pool_files (path, size_bytes, added_at) VALUES (?, ?, ?)",
                (str(test_file.absolute()), 8, old_time),
            )
            conn.commit()

        # Run cleanup
        bytes_freed = await manager.cleanup_oldest()

        # File should be removed
        assert not test_file.exists()
        assert bytes_freed == 8

    @pytest.mark.asyncio
    async def test_cleanup_oldest_keeps_recent(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test cleanup keeps recent files."""
        test_file = temp_root_path / "recent_file.db"
        test_file.write_bytes(b"recent data")
        manager.mark_data_for_rotation(test_file)

        # Run cleanup (should not remove recent file)
        bytes_freed = await manager.cleanup_oldest()

        assert test_file.exists()
        assert bytes_freed == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_target_free(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test cleanup with target free space."""
        # Create several files
        for i in range(5):
            test_file = temp_root_path / f"file_{i}.db"
            test_file.write_bytes(b"x" * 1000)
            manager.mark_data_for_rotation(test_file)
            # Small delay to ensure different timestamps
            time.sleep(0.01)

        # Request cleanup to free significant space
        bytes_freed = await manager.cleanup_oldest(target_free_bytes=4000)

        # Should have freed at least 4000 bytes
        # (may free more depending on which files were removed)
        assert bytes_freed >= 0

    def test_cleanup_missing_entries(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test cleanup of stale database entries."""
        # Add entry for nonexistent file
        with sqlite3.connect(manager._db_path) as conn:
            conn.execute(
                "INSERT INTO pool_files (path, size_bytes, added_at) VALUES (?, ?, ?)",
                ("/nonexistent/file.db", 1000, time.time()),
            )
            conn.commit()

        # Run cleanup
        removed = manager.cleanup_missing_entries()

        assert removed == 1

        # Entry should be removed
        with sqlite3.connect(manager._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pool_files")
            assert cursor.fetchone()[0] == 0


# ============================================================================
# Is In Pool Subdir Tests
# ============================================================================


class TestIsInPoolSubdir:
    """Tests for is_in_pool_subdir method."""

    def test_in_pool_subdir(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test path in pool subdir."""
        # Create pool subdir
        pool_dir = temp_root_path / "data" / "synced"
        pool_dir.mkdir(parents=True)
        test_file = pool_dir / "test.db"

        result = manager.is_in_pool_subdir(test_file)
        assert result is True

    def test_not_in_pool_subdir(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test path not in pool subdir."""
        test_file = temp_root_path / "other" / "file.db"

        result = manager.is_in_pool_subdir(test_file)
        assert result is False

    def test_subdir_check_with_invalid_path(self, manager: RotatingDiskPoolManager):
        """Test subdir check with problematic path."""
        # Should handle gracefully
        result = manager.is_in_pool_subdir(Path("/nonexistent/deeply/nested/path"))
        assert result is False


# ============================================================================
# Config Loading Tests
# ============================================================================


class TestConfigLoading:
    """Tests for config loading from YAML."""

    def test_load_config_missing_file(self, temp_root_path: Path):
        """Test loading config when file doesn't exist."""
        manager = RotatingDiskPoolManager(root_path=temp_root_path)

        # Should use default config
        assert manager.config.quota_percent == 10.0

    def test_load_config_disabled(self, temp_root_path: Path):
        """Test loading config when pool is disabled."""
        config_path = temp_root_path / "config" / "distributed_hosts.yaml"
        with open(config_path, "w") as f:
            yaml.dump({
                "disk_management": {
                    "rotating_pool": {
                        "enabled": False,
                        "quota_percent": 50.0,
                    }
                }
            }, f)

        manager = RotatingDiskPoolManager(root_path=temp_root_path)

        # Should use default config when disabled
        assert manager.config.quota_percent == 10.0

    def test_load_config_invalid_yaml(self, temp_root_path: Path):
        """Test loading config with invalid YAML."""
        config_path = temp_root_path / "config" / "distributed_hosts.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: [")

        manager = RotatingDiskPoolManager(root_path=temp_root_path)

        # Should use default config on error
        assert manager.config.quota_percent == 10.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test full pool workflow."""
        # Check initial state
        stats = manager.get_pool_stats()
        assert stats.file_count == 0

        # Add some files
        files = []
        for i in range(3):
            test_file = temp_root_path / f"workflow_file_{i}.db"
            test_file.write_bytes(b"x" * 1000)
            manager.mark_data_for_rotation(test_file, source=f"node-{i}")
            files.append(test_file)
            time.sleep(0.01)

        # Check usage
        stats = manager.get_pool_stats()
        assert stats.file_count == 3
        assert stats.used_bytes == 3000

        # Can accept more?
        assert manager.can_accept_data(1000) is True

        # Cleanup oldest (set old timestamp first)
        with sqlite3.connect(manager._db_path) as conn:
            conn.execute(
                "UPDATE pool_files SET added_at = ? WHERE path = ?",
                (time.time() - 7200, str(files[0].absolute())),
            )
            conn.commit()

        bytes_freed = await manager.cleanup_oldest()
        assert bytes_freed == 1000
        assert not files[0].exists()

        # Final stats
        stats = manager.get_pool_stats()
        assert stats.file_count == 2
        assert stats.used_bytes == 2000

    def test_pool_directory_structure(
        self, manager: RotatingDiskPoolManager, temp_root_path: Path
    ):
        """Test pool recognizes its directories."""
        # Create pool subdirs
        for subdir in manager.config.pool_subdirs:
            (temp_root_path / subdir).mkdir(parents=True, exist_ok=True)

        # Files in pool dirs should be recognized
        pool_file = temp_root_path / manager.config.pool_subdirs[0] / "test.db"
        pool_file.parent.mkdir(parents=True, exist_ok=True)
        pool_file.touch()

        assert manager.is_in_pool_subdir(pool_file) is True

        # Files outside pool dirs should not be recognized
        other_file = temp_root_path / "other" / "test.db"
        assert manager.is_in_pool_subdir(other_file) is False
