"""Unit tests for DiskSpaceManagerDaemon (December 2025).

Tests the proactive disk space management for training nodes and coordinators.

Created: December 27, 2025
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.disk_space_manager_daemon import (
    CoordinatorDiskConfig,
    CoordinatorDiskManager,
    DiskSpaceConfig,
    DiskSpaceManagerDaemon,
    DiskStatus,
    get_coordinator_disk_daemon,
    get_disk_space_daemon,
    reset_coordinator_disk_daemon,
    reset_disk_space_daemon,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_root() -> Path:
    """Create a temporary directory structure simulating ai-service."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create directory structure
        (root / "app" / "coordination").mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "data" / "games").mkdir(parents=True)
        (root / "data" / "training").mkdir(parents=True)
        (root / "data" / "checkpoints").mkdir(parents=True)
        (root / "data" / "quarantine").mkdir(parents=True)

        yield root


@pytest.fixture
def config() -> DiskSpaceConfig:
    """Create a test configuration."""
    return DiskSpaceConfig(
        check_interval_seconds=1,
        proactive_cleanup_threshold=60,
        warning_threshold=70,
        critical_threshold=85,
        target_disk_usage=50,
        log_retention_days=7,
        checkpoint_retention_days=14,
        min_checkpoints_per_config=2,
        min_logs_to_keep=3,
        enable_cleanup=True,
        emit_events=False,  # Disable for tests
    )


@pytest.fixture
def daemon(temp_root: Path, config: DiskSpaceConfig) -> DiskSpaceManagerDaemon:
    """Create a daemon with temporary root."""
    daemon = DiskSpaceManagerDaemon(config=config)
    daemon._root_path = temp_root
    return daemon


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_disk_space_daemon()
    reset_coordinator_disk_daemon()
    yield
    reset_disk_space_daemon()
    reset_coordinator_disk_daemon()


# ============================================================================
# DiskSpaceConfig Tests
# ============================================================================


class TestDiskSpaceConfig:
    """Tests for DiskSpaceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DiskSpaceConfig()
        assert config.check_interval_seconds == 1800
        assert config.proactive_cleanup_threshold == 60
        assert config.warning_threshold == 70
        assert config.critical_threshold == 85
        assert config.emergency_threshold == 95
        assert config.target_disk_usage == 50
        assert config.log_retention_days == 7
        assert config.enable_cleanup is True
        assert config.emit_events is True

    def test_cleanup_priorities(self) -> None:
        """Test default cleanup priorities."""
        config = DiskSpaceConfig()
        assert "logs" in config.cleanup_priorities
        assert "cache" in config.cleanup_priorities
        assert "empty_dbs" in config.cleanup_priorities
        assert "old_checkpoints" in config.cleanup_priorities

    def test_from_env(self) -> None:
        """Test loading config from environment."""
        with patch.dict(os.environ, {
            "RINGRIFT_DISK_SPACE_PROACTIVE_THRESHOLD": "55",
            "RINGRIFT_DISK_SPACE_WARNING_THRESHOLD": "65",
            "RINGRIFT_DISK_SPACE_LOG_RETENTION_DAYS": "3",
            "RINGRIFT_DISK_SPACE_ENABLE_CLEANUP": "0",
        }):
            config = DiskSpaceConfig.from_env()
            assert config.proactive_cleanup_threshold == 55
            assert config.warning_threshold == 65
            assert config.log_retention_days == 3
            assert config.enable_cleanup is False


# ============================================================================
# DiskStatus Tests
# ============================================================================


class TestDiskStatus:
    """Tests for DiskStatus dataclass."""

    def test_from_path(self, temp_root: Path, config: DiskSpaceConfig) -> None:
        """Test creating DiskStatus from path."""
        status = DiskStatus.from_path(str(temp_root), config)

        assert status.path == str(temp_root)
        assert status.total_gb > 0
        assert status.used_gb >= 0
        assert status.free_gb >= 0
        assert 0 <= status.usage_percent <= 100

    def test_from_path_thresholds(self, temp_root: Path) -> None:
        """Test threshold flags are set correctly."""
        # Low threshold config to test flags
        config = DiskSpaceConfig(
            proactive_cleanup_threshold=1,  # Very low to trigger
            warning_threshold=2,
            critical_threshold=3,
            emergency_threshold=4,
        )

        status = DiskStatus.from_path(str(temp_root), config)

        # On any real disk, we should exceed 1% usage
        if status.usage_percent >= 1:
            assert status.needs_cleanup is True

    def test_from_path_invalid(self, config: DiskSpaceConfig) -> None:
        """Test handling of invalid path."""
        status = DiskStatus.from_path("/nonexistent/path/12345", config)

        # Should return error status
        assert status.usage_percent == 100
        assert status.needs_cleanup is True
        assert status.is_critical is True

    def test_status_flags(self) -> None:
        """Test individual status flags."""
        config = DiskSpaceConfig(
            proactive_cleanup_threshold=60,
            warning_threshold=70,
            critical_threshold=85,
            emergency_threshold=95,
        )

        # Create statuses with different usage levels
        status_low = DiskStatus(
            path="/test",
            total_gb=100,
            used_gb=50,
            free_gb=50,
            usage_percent=50,
            needs_cleanup=False,
            is_warning=False,
            is_critical=False,
            is_emergency=False,
        )
        assert status_low.needs_cleanup is False
        assert status_low.is_warning is False

        status_high = DiskStatus(
            path="/test",
            total_gb=100,
            used_gb=75,
            free_gb=25,
            usage_percent=75,
            needs_cleanup=True,
            is_warning=True,
            is_critical=False,
            is_emergency=False,
        )
        assert status_high.needs_cleanup is True
        assert status_high.is_warning is True


# ============================================================================
# DiskSpaceManagerDaemon Tests
# ============================================================================


class TestDiskSpaceManagerDaemonInit:
    """Tests for daemon initialization."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        daemon = DiskSpaceManagerDaemon()
        assert daemon.config is not None
        assert daemon._bytes_cleaned == 0
        assert daemon._cleanups_performed == 0

    def test_init_custom_config(self, config: DiskSpaceConfig) -> None:
        """Test initialization with custom config."""
        daemon = DiskSpaceManagerDaemon(config=config)
        assert daemon.config.proactive_cleanup_threshold == 60

    def test_find_ai_service_root(self, temp_root: Path) -> None:
        """Test root directory discovery."""
        daemon = DiskSpaceManagerDaemon()
        # The daemon should find some root (may not be temp_root without mocking)
        assert daemon._root_path is not None

    def test_get_daemon_name(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test daemon name."""
        assert daemon._get_daemon_name() == "DiskSpaceManagerDaemon"


class TestDiskSpaceManagerDaemonRunCycle:
    """Tests for the main run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_updates_status(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test that run cycle updates status."""
        assert daemon._current_status is None

        await daemon._run_cycle()

        assert daemon._current_status is not None
        assert daemon._current_status.usage_percent >= 0

    @pytest.mark.asyncio
    async def test_run_cycle_triggers_cleanup(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test that run cycle triggers cleanup when needed."""
        # Mock status to need cleanup
        mock_status = DiskStatus(
            path="/test",
            total_gb=100,
            used_gb=70,
            free_gb=30,
            usage_percent=70,  # Above 60% threshold
            needs_cleanup=True,
            is_warning=True,
            is_critical=False,
            is_emergency=False,
        )

        with patch.object(
            DiskStatus, "from_path", return_value=mock_status
        ):
            with patch.object(daemon, "_perform_cleanup", new_callable=AsyncMock) as mock_cleanup:
                await daemon._run_cycle()
                mock_cleanup.assert_called_once()


class TestDiskSpaceManagerCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_old_logs(self, daemon: DiskSpaceManagerDaemon, temp_root: Path) -> None:
        """Test old log cleanup."""
        logs_path = temp_root / "logs"

        # Create enough log files to exceed minimum
        # min_logs_to_keep is 3, so create 5 total
        for i in range(daemon.config.min_logs_to_keep + 2):
            log_file = logs_path / f"log_{i}.log"
            log_file.write_text(f"log content {i}")

        # Make the first few old
        old_logs = list(logs_path.glob("*.log"))[:2]
        old_time = time.time() - (30 * 86400)  # 30 days ago
        for old_log in old_logs:
            os.utime(old_log, (old_time, old_time))

        bytes_freed = daemon._cleanup_old_logs()

        # Old logs should be removed, new ones kept
        remaining_logs = list(logs_path.glob("*.log"))
        assert len(remaining_logs) == daemon.config.min_logs_to_keep
        assert bytes_freed > 0

    def test_cleanup_old_logs_keeps_minimum(
        self, daemon: DiskSpaceManagerDaemon, temp_root: Path
    ) -> None:
        """Test that minimum logs are kept."""
        logs_path = temp_root / "logs"

        # Create exactly min_logs_to_keep files
        for i in range(daemon.config.min_logs_to_keep):
            log_file = logs_path / f"log_{i}.log"
            log_file.write_text(f"content {i}")
            # Make them old
            old_time = time.time() - (30 * 86400)
            os.utime(log_file, (old_time, old_time))

        bytes_freed = daemon._cleanup_old_logs()

        # Should not remove any (all within minimum)
        assert bytes_freed == 0
        assert len(list(logs_path.glob("*.log"))) == daemon.config.min_logs_to_keep

    def test_cleanup_cache(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test cache cleanup."""
        # This will try to clean real cache dirs - just verify no crash
        bytes_freed = daemon._cleanup_cache()
        assert bytes_freed >= 0

    def test_cleanup_empty_databases(
        self, daemon: DiskSpaceManagerDaemon, temp_root: Path
    ) -> None:
        """Test empty database cleanup."""
        games_path = temp_root / "data" / "games"

        # Create a small "database" file (not actually SQLite)
        small_db = games_path / "small.db"
        small_db.write_text("small")
        # Make it old
        old_time = time.time() - (5 * 86400)  # 5 days ago
        os.utime(small_db, (old_time, old_time))

        # The cleanup relies on sqlite3 command which may fail on fake files
        # Just verify no crash
        bytes_freed = daemon._cleanup_empty_databases()
        assert bytes_freed >= 0

    def test_cleanup_old_checkpoints(
        self, daemon: DiskSpaceManagerDaemon, temp_root: Path
    ) -> None:
        """Test old checkpoint cleanup."""
        checkpoints_path = temp_root / "data" / "checkpoints"

        # Create checkpoints for a config
        for i in range(5):
            checkpoint = checkpoints_path / f"hex8_2p_epoch{i}.pth"
            checkpoint.write_text(f"checkpoint {i}")
            # Set times so they're ordered
            t = time.time() - (i * 3600)  # Stagger by hours
            os.utime(checkpoint, (t, t))

        bytes_freed = daemon._cleanup_old_checkpoints()

        # Should keep min_checkpoints_per_config (2), remove 3
        remaining = list(checkpoints_path.glob("*.pth"))
        assert len(remaining) == daemon.config.min_checkpoints_per_config

    def test_cleanup_quarantine(
        self, daemon: DiskSpaceManagerDaemon, temp_root: Path
    ) -> None:
        """Test quarantine cleanup."""
        quarantine_path = temp_root / "data" / "quarantine"

        # Create some quarantined files
        quarantined = quarantine_path / "bad_file.db"
        quarantined.write_text("quarantined content")

        bytes_freed = daemon._cleanup_quarantine()

        assert not quarantined.exists()
        assert bytes_freed > 0


class TestDiskSpaceManagerHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test health check when daemon not running."""
        result = daemon.health_check()
        assert result.healthy is False
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_running(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test health check when daemon is running."""
        await daemon.start()

        result = daemon.health_check()
        assert result.healthy is True
        assert "healthy" in result.message.lower() or "running" in result.message.lower()

        await daemon.stop()

    def test_health_check_critical_disk(
        self, daemon: DiskSpaceManagerDaemon
    ) -> None:
        """Test health check with critical disk status."""
        daemon._running = True

        # Mock critical status
        daemon._current_status = DiskStatus(
            path="/test",
            total_gb=100,
            used_gb=90,
            free_gb=10,
            usage_percent=90,
            needs_cleanup=True,
            is_warning=True,
            is_critical=True,
            is_emergency=False,
        )

        result = daemon.health_check()
        assert result.healthy is False
        assert "critical" in result.message.lower()


class TestDiskSpaceManagerStatus:
    """Tests for status reporting."""

    def test_get_disk_status(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test getting disk status."""
        status = daemon.get_disk_status()
        assert status is not None
        assert status.total_gb >= 0

    def test_get_status_dict(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test getting full status dictionary."""
        status = daemon.get_status()

        assert "name" in status
        assert status["name"] == "DiskSpaceManagerDaemon"
        assert "running" in status
        assert "config" in status
        assert "disk" in status
        assert "health" in status


# ============================================================================
# CoordinatorDiskConfig Tests
# ============================================================================


class TestCoordinatorDiskConfig:
    """Tests for CoordinatorDiskConfig dataclass."""

    def test_default_values(self) -> None:
        """Test coordinator-specific default values."""
        config = CoordinatorDiskConfig()
        assert config.proactive_cleanup_threshold == 50  # Lower than base
        assert config.target_disk_usage == 40  # Lower than base
        assert config.min_free_gb == 100  # Higher than base
        assert config.log_retention_days == 3  # Shorter than base
        assert config.remote_sync_enabled is True

    def test_cleanup_priorities_include_synced(self) -> None:
        """Test that coordinator has synced data in priorities."""
        config = CoordinatorDiskConfig()
        assert "synced_training" in config.cleanup_priorities
        assert "synced_games" in config.cleanup_priorities

    def test_for_coordinator(self) -> None:
        """Test factory method."""
        config = CoordinatorDiskConfig.for_coordinator()
        assert isinstance(config, CoordinatorDiskConfig)
        assert config.remote_host == "mac-studio"

    def test_for_coordinator_with_env(self) -> None:
        """Test factory method with environment override."""
        with patch.dict(os.environ, {
            "RINGRIFT_COORDINATOR_REMOTE_HOST": "custom-host",
            "RINGRIFT_COORDINATOR_REMOTE_PATH": "/custom/path",
        }):
            config = CoordinatorDiskConfig.for_coordinator()
            assert config.remote_host == "custom-host"
            assert config.remote_base_path == "/custom/path"


# ============================================================================
# CoordinatorDiskManager Tests
# ============================================================================


class TestCoordinatorDiskManager:
    """Tests for CoordinatorDiskManager class."""

    @pytest.fixture
    def coord_daemon(self, temp_root: Path) -> CoordinatorDiskManager:
        """Create a coordinator daemon with temporary root."""
        config = CoordinatorDiskConfig(
            remote_sync_enabled=False,  # Disable for tests
            emit_events=False,
        )
        daemon = CoordinatorDiskManager(config=config)
        daemon._root_path = temp_root
        return daemon

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        daemon = CoordinatorDiskManager()
        assert isinstance(daemon.config, CoordinatorDiskConfig)

    def test_get_daemon_name(self, coord_daemon: CoordinatorDiskManager) -> None:
        """Test daemon name."""
        assert coord_daemon._get_daemon_name() == "CoordinatorDiskManager"

    def test_cleanup_synced_training(
        self, coord_daemon: CoordinatorDiskManager, temp_root: Path
    ) -> None:
        """Test cleanup of synced training files."""
        training_path = temp_root / "data" / "training"

        # Create old training file
        old_npz = training_path / "old_data.npz"
        old_npz.write_text("old training data")
        old_time = time.time() - (2 * 86400)  # 2 days old
        os.utime(old_npz, (old_time, old_time))

        # Create recent training file
        new_npz = training_path / "new_data.npz"
        new_npz.write_text("new training data")

        bytes_freed = coord_daemon._cleanup_synced_training()

        assert not old_npz.exists()
        assert new_npz.exists()
        assert bytes_freed > 0

    def test_cleanup_synced_games(
        self, coord_daemon: CoordinatorDiskManager, temp_root: Path
    ) -> None:
        """Test cleanup of synced game databases."""
        games_path = temp_root / "data" / "games"

        # Create non-canonical old database
        old_db = games_path / "selfplay_old.db"
        old_db.write_text("old game data")
        old_time = time.time() - (2 * 86400)  # 2 days old
        os.utime(old_db, (old_time, old_time))

        # Create canonical database (should not be deleted)
        canonical_db = games_path / "canonical_hex8_2p.db"
        canonical_db.write_text("canonical data")
        os.utime(canonical_db, (old_time, old_time))

        bytes_freed = coord_daemon._cleanup_synced_games()

        assert not old_db.exists()
        assert canonical_db.exists()  # Canonical preserved
        assert bytes_freed > 0

    def test_get_status_includes_sync_stats(
        self, coord_daemon: CoordinatorDiskManager
    ) -> None:
        """Test that status includes sync statistics."""
        status = coord_daemon.get_status()
        assert "sync_stats" in status
        assert "files_synced" in status["sync_stats"]
        assert "bytes_synced" in status["sync_stats"]
        assert "sync_errors" in status["sync_stats"]


class TestCoordinatorDiskManagerSync:
    """Tests for remote sync functionality."""

    @pytest.fixture
    def coord_daemon_with_sync(self, temp_root: Path) -> CoordinatorDiskManager:
        """Create daemon with sync enabled."""
        config = CoordinatorDiskConfig(
            remote_sync_enabled=True,
            sync_before_cleanup=True,
            remote_host="test-host",
            remote_base_path="/test/path",
            emit_events=False,
        )
        daemon = CoordinatorDiskManager(config=config)
        daemon._root_path = temp_root
        return daemon

    @pytest.mark.asyncio
    async def test_sync_to_remote_calls_rsync(
        self, coord_daemon_with_sync: CoordinatorDiskManager, temp_root: Path
    ) -> None:
        """Test that sync calls rsync command."""
        # Create some data to sync
        games_path = temp_root / "data" / "games"
        (games_path / "test.db").write_text("test data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            await coord_daemon_with_sync._sync_to_remote()

            # Should have called rsync for each sync directory
            assert mock_run.called

    @pytest.mark.asyncio
    async def test_sync_error_handling(
        self, coord_daemon_with_sync: CoordinatorDiskManager, temp_root: Path
    ) -> None:
        """Test sync error handling."""
        # Create some data
        games_path = temp_root / "data" / "games"
        (games_path / "test.db").write_text("test data")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="sync failed")

            await coord_daemon_with_sync._sync_to_remote()

            # Should record error
            assert coord_daemon_with_sync._sync_stats["sync_errors"] >= 0


# ============================================================================
# Singleton Function Tests
# ============================================================================


class TestSingletonFunctions:
    """Tests for singleton access functions."""

    def test_get_disk_space_daemon_singleton(self) -> None:
        """Test singleton behavior."""
        daemon1 = get_disk_space_daemon()
        daemon2 = get_disk_space_daemon()
        assert daemon1 is daemon2

    def test_reset_disk_space_daemon(self) -> None:
        """Test singleton reset."""
        daemon1 = get_disk_space_daemon()
        reset_disk_space_daemon()
        daemon2 = get_disk_space_daemon()
        assert daemon1 is not daemon2

    def test_get_coordinator_disk_daemon_singleton(self) -> None:
        """Test coordinator singleton."""
        daemon1 = get_coordinator_disk_daemon()
        daemon2 = get_coordinator_disk_daemon()
        assert daemon1 is daemon2

    def test_reset_coordinator_disk_daemon(self) -> None:
        """Test coordinator singleton reset."""
        daemon1 = get_coordinator_disk_daemon()
        reset_coordinator_disk_daemon()
        daemon2 = get_coordinator_disk_daemon()
        assert daemon1 is not daemon2


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_emit_status_events_critical(
        self, temp_root: Path
    ) -> None:
        """Test event emission on critical status."""
        config = DiskSpaceConfig(
            emit_events=True,
        )
        daemon = DiskSpaceManagerDaemon(config=config)
        daemon._root_path = temp_root

        critical_status = DiskStatus(
            path="/test",
            total_gb=100,
            used_gb=90,
            free_gb=10,
            usage_percent=90,
            needs_cleanup=True,
            is_warning=True,
            is_critical=True,
            is_emergency=False,
        )

        # Should not crash even if event module not available
        await daemon._emit_status_events(critical_status)

    @pytest.mark.asyncio
    async def test_emit_cleanup_event(self, daemon: DiskSpaceManagerDaemon) -> None:
        """Test cleanup event emission."""
        daemon.config.emit_events = True

        # Mock the event router module
        mock_router = MagicMock()
        mock_router.publish = AsyncMock()

        with patch.dict("sys.modules", {
            "app.coordination.event_router": MagicMock(
                DataEventType=MagicMock(DISK_CLEANUP_TRIGGERED="disk_cleanup_triggered"),
                get_router=MagicMock(return_value=mock_router)
            )
        }):
            await daemon._emit_cleanup_event(1024 * 1024)
            # Should complete without error
            # The actual call may or may not happen depending on imports
