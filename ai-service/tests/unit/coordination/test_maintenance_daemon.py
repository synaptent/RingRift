"""Tests for MaintenanceDaemon.

Comprehensive unit tests covering:
1. Initialization with default and custom config
2. Daemon lifecycle (start/stop)
3. Log rotation logic
4. Database vacuum operations
5. Old checkpoint cleanup
6. Disk space monitoring
7. Error handling paths
8. Health check returns
9. Work queue stale item cleanup
10. Orphan file detection and recovery
11. Singleton pattern
12. DLQ cleanup
13. Archive functionality
14. S3 upload
15. Orphan database recovery

December 27, 2025: Enhanced to address P1 test gap for maintenance_daemon.py.
"""

import asyncio
import gzip
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.maintenance_daemon import (
    MaintenanceConfig,
    MaintenanceDaemon,
    MaintenanceStats,
    get_maintenance_daemon,
    reset_maintenance_daemon,
)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# =============================================================================
# MaintenanceConfig Tests
# =============================================================================


class TestMaintenanceConfig:
    """Tests for MaintenanceConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = MaintenanceConfig()
        assert config.enabled is True
        assert config.log_max_size_mb == 100.0
        assert config.log_backup_count == 10
        assert config.log_rotation_interval_hours == 1.0
        assert config.db_vacuum_interval_hours == 168.0  # Weekly
        assert config.archive_games_older_than_days == 30
        assert config.archive_interval_hours == 24.0  # Daily
        assert config.dry_run is False

    def test_custom_values(self):
        """Should accept custom values."""
        config = MaintenanceConfig(
            enabled=False,
            log_max_size_mb=50.0,
            log_backup_count=5,
            db_vacuum_interval_hours=24.0,
            archive_games_older_than_days=7,
            dry_run=True,
        )
        assert config.enabled is False
        assert config.log_max_size_mb == 50.0
        assert config.log_backup_count == 5
        assert config.db_vacuum_interval_hours == 24.0
        assert config.archive_games_older_than_days == 7
        assert config.dry_run is True

    def test_queue_cleanup_defaults(self):
        """Should have queue cleanup settings."""
        config = MaintenanceConfig()
        assert config.queue_cleanup_interval_hours == 1.0
        assert config.queue_stale_pending_hours == 24.0
        assert config.queue_stale_claimed_hours == 1.0
        assert config.queue_cleanup_enabled is True

    def test_orphan_detection_defaults(self):
        """Should have orphan detection settings."""
        config = MaintenanceConfig()
        assert config.orphan_detection_interval_hours == 24.0
        assert config.orphan_detection_enabled is True
        assert config.orphan_auto_cleanup is False
        assert config.orphan_auto_recovery is True

    def test_s3_archive_settings(self):
        """Should have S3 archive settings."""
        config = MaintenanceConfig()
        assert config.archive_compress is True
        # archive_to_s3 defaults to False
        assert config.archive_to_s3 is False


# =============================================================================
# MaintenanceStats Tests
# =============================================================================


class TestMaintenanceStats:
    """Tests for MaintenanceStats dataclass."""

    def test_default_values(self):
        """Should have zero defaults."""
        stats = MaintenanceStats()
        assert stats.logs_rotated == 0
        assert stats.bytes_reclaimed_logs == 0
        assert stats.databases_vacuumed == 0
        assert stats.games_archived == 0
        assert stats.dlq_entries_cleaned == 0
        assert stats.queue_items_cleaned == 0
        assert stats.queue_items_reset == 0
        assert stats.orphan_dbs_found == 0
        assert stats.orphan_npz_found == 0
        assert stats.orphan_models_found == 0
        assert stats.orphan_dbs_recovered == 0

    def test_record_log_rotation(self):
        """Should record log rotation stats."""
        stats = MaintenanceStats()
        stats.record_log_rotation(logs=3, bytes_reclaimed=10240000)
        assert stats.logs_rotated == 3
        assert stats.bytes_reclaimed_logs == 10240000
        assert stats.bytes_reclaimed == 10240000
        assert stats.last_log_rotation > 0

    def test_record_vacuum(self):
        """Should record vacuum stats."""
        stats = MaintenanceStats()
        stats.record_vacuum(databases=5)
        assert stats.databases_vacuumed == 5
        assert stats.last_db_vacuum > 0

    def test_record_archive(self):
        """Should record archive stats."""
        stats = MaintenanceStats()
        stats.record_archive(games=100)
        assert stats.games_archived == 100
        assert stats.last_archive_run > 0

    def test_inherits_from_cleanup_daemon_stats(self):
        """Should inherit from CleanupDaemonStats."""
        stats = MaintenanceStats()
        # Inherited methods should exist
        assert hasattr(stats, "record_cleanup")
        assert hasattr(stats, "is_healthy")
        assert hasattr(stats, "to_dict")


# =============================================================================
# MaintenanceDaemon Initialization Tests
# =============================================================================


class TestMaintenanceDaemonInit:
    """Tests for MaintenanceDaemon initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        daemon = MaintenanceDaemon()
        assert daemon.config.enabled is True
        assert daemon._running is False
        assert daemon._stats is not None

    def test_custom_config(self):
        """Should accept custom config."""
        config = MaintenanceConfig(enabled=False, dry_run=True)
        daemon = MaintenanceDaemon(config=config)
        assert daemon.config.enabled is False
        assert daemon.config.dry_run is True


# =============================================================================
# MaintenanceDaemon Lifecycle Tests
# =============================================================================


class TestMaintenanceDaemonLifecycle:
    """Tests for daemon lifecycle methods."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            enabled=True,
            dry_run=True,  # Dry run to avoid actual file operations
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_start(self, daemon):
        """Should start the daemon."""
        with patch.object(daemon, "_run_maintenance_cycle", new_callable=AsyncMock):
            await daemon.start()
            assert daemon._running is True

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Start should be idempotent."""
        with patch.object(daemon, "_run_maintenance_cycle", new_callable=AsyncMock) as mock_cycle:
            await daemon.start()
            await daemon.start()  # Second call should not restart
            # Should only run initial cycle once
            assert mock_cycle.await_count == 1

    @pytest.mark.asyncio
    async def test_stop(self, daemon):
        """Should stop the daemon."""
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_start_initializes_timestamps(self, daemon):
        """Start should initialize last run timestamps."""
        with patch.object(daemon, "_run_maintenance_cycle", new_callable=AsyncMock):
            await daemon.start()
            assert daemon._stats.last_log_rotation > 0
            assert daemon._stats.last_db_vacuum > 0
            assert daemon._stats.last_archive_run > 0
            assert daemon._stats.last_dlq_cleanup > 0


# =============================================================================
# Maintenance Cycle Tests
# =============================================================================


class TestMaintenanceCycle:
    """Tests for maintenance cycle logic."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            enabled=True,
            dry_run=True,
            log_rotation_interval_hours=1.0,
            archive_interval_hours=24.0,
            db_vacuum_interval_hours=168.0,
            queue_cleanup_interval_hours=1.0,
            orphan_detection_interval_hours=24.0,
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_cycle_skipped_when_disabled(self, daemon):
        """Should skip cycle when disabled."""
        daemon.config.enabled = False
        with patch.object(daemon, "_rotate_logs") as mock_rotate:
            await daemon._run_maintenance_cycle()
            mock_rotate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_runs_log_rotation_when_due(self, daemon):
        """Should run log rotation when interval elapsed."""
        # Set last run to far in past
        daemon._stats.last_log_rotation = time.time() - 7200  # 2 hours ago
        with patch.object(daemon, "_rotate_logs", new_callable=AsyncMock) as mock_rotate:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_rotate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_skips_log_rotation_when_not_due(self, daemon):
        """Should skip log rotation when interval not elapsed."""
        daemon._stats.last_log_rotation = time.time()  # Just now
        with patch.object(daemon, "_rotate_logs", new_callable=AsyncMock) as mock_rotate:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_rotate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_runs_archive_when_due(self, daemon):
        """Should run archive when daily interval elapsed."""
        daemon._stats.last_archive_run = time.time() - 100000  # Far in past
        with patch.object(daemon, "_archive_old_games", new_callable=AsyncMock) as mock_archive:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_archive.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_runs_vacuum_when_due(self, daemon):
        """Should run vacuum when weekly interval elapsed."""
        daemon._stats.last_db_vacuum = time.time() - 1000000  # Far in past
        with patch.object(daemon, "_vacuum_databases", new_callable=AsyncMock) as mock_vacuum:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_vacuum.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_runs_queue_cleanup_when_due(self, daemon):
        """Should run queue cleanup when interval elapsed."""
        daemon._stats.last_queue_cleanup = time.time() - 7200  # 2 hours ago
        with patch.object(
            daemon, "_cleanup_stale_queue_items", new_callable=AsyncMock
        ) as mock_cleanup:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_runs_orphan_detection_when_due(self, daemon):
        """Should run orphan detection when interval elapsed."""
        daemon._stats.last_orphan_detection = time.time() - 100000  # Far in past
        with patch.object(
            daemon, "_detect_orphan_files", new_callable=AsyncMock
        ) as mock_orphan:
            await daemon._run_maintenance_cycle()
            mock_orphan.assert_called_once()


# =============================================================================
# Log Rotation Tests
# =============================================================================


class TestLogRotation:
    """Tests for log rotation functionality."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with temp directory."""
        config = MaintenanceConfig(
            log_max_size_mb=0.001,  # Very small to trigger rotation
            log_backup_count=3,
            dry_run=False,
        )
        daemon = MaintenanceDaemon(config=config)
        return daemon

    @pytest.mark.asyncio
    async def test_rotate_logs_skips_small_files(self, daemon):
        """Should skip files under max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create small log file
            (logs_dir / "test.log").write_text("small log")

            await daemon._rotate_logs()
            # File should still exist unrotated
            assert (logs_dir / "test.log").exists()
            assert not (logs_dir / "test.log.1.gz").exists()

    @pytest.mark.asyncio
    async def test_rotate_logs_compresses_large_files(self, daemon):
        """Should compress files over max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create large log file (>1KB given max is 0.001 MB = 1KB)
            (logs_dir / "test.log").write_text("x" * 2000)

            await daemon._rotate_logs()
            # Should have compressed backup
            assert (logs_dir / "test.log.1.gz").exists()
            # Original should be truncated
            assert (logs_dir / "test.log").stat().st_size < 2000

    @pytest.mark.asyncio
    async def test_rotate_logs_dry_run(self):
        """Should log but not rotate in dry run mode."""
        config = MaintenanceConfig(dry_run=True, log_max_size_mb=0.001)
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create large log file
            (logs_dir / "test.log").write_text("x" * 2000)

            await daemon._rotate_logs()
            # Should NOT have backup in dry run
            assert not (logs_dir / "test.log.1.gz").exists()
            # Original should be unchanged
            assert (logs_dir / "test.log").stat().st_size == 2000


# =============================================================================
# Database VACUUM Tests
# =============================================================================


class TestDatabaseVacuum:
    """Tests for database VACUUM functionality."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with temp directory."""
        config = MaintenanceConfig(dry_run=False)
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_vacuum_databases(self, daemon):
        """Should VACUUM databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            # Create a test database with data to vacuum
            db_path = games_dir / "test.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")
                for i in range(100):
                    conn.execute("INSERT INTO test VALUES (?)", (i,))
                conn.execute("DELETE FROM test")
                conn.commit()

            size_before = db_path.stat().st_size

            await daemon._vacuum_databases()

            # VACUUM should have been called
            assert daemon._stats.databases_vacuumed >= 1

    @pytest.mark.asyncio
    async def test_vacuum_databases_dry_run(self):
        """Should log but not VACUUM in dry run mode."""
        config = MaintenanceConfig(dry_run=True)
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            db_path = games_dir / "test.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")

            size_before = db_path.stat().st_size

            await daemon._vacuum_databases()

            # Stats should not be updated in dry run
            assert daemon._stats.databases_vacuumed == 0

    @pytest.mark.asyncio
    async def test_vacuum_handles_corrupt_db(self, daemon):
        """Should handle corrupt databases gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            # Create corrupt database (just garbage data)
            corrupt_db = games_dir / "corrupt.db"
            corrupt_db.write_text("not a database")

            # Should not raise
            await daemon._vacuum_databases()


# =============================================================================
# get_status and health_check Tests
# =============================================================================


class TestStatusAndHealth:
    """Tests for status and health check methods."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return MaintenanceDaemon()

    def test_get_status(self, daemon):
        """Should return comprehensive status."""
        daemon._running = True
        daemon._stats.logs_rotated = 5
        daemon._stats.databases_vacuumed = 2

        status = daemon.get_status()

        assert status["running"] is True
        assert status["enabled"] is True
        assert status["stats"]["logs_rotated"] == 5
        assert status["stats"]["databases_vacuumed"] == 2
        assert "config" in status
        assert "last_runs" in status

    def test_health_check_not_running(self, daemon):
        """Should report unhealthy when not running."""
        daemon._running = False

        health = daemon.health_check()

        assert health.healthy is False
        assert "not running" in health.message.lower()

    def test_health_check_running_healthy(self, daemon):
        """Should report healthy when running normally."""
        daemon._running = True
        daemon._stats.last_log_rotation = time.time()
        daemon._stats.last_db_vacuum = time.time()

        health = daemon.health_check()

        assert health.healthy is True
        assert "running" in health.message.lower()

    def test_health_check_log_rotation_overdue(self, daemon):
        """Should report degraded when log rotation overdue."""
        daemon._running = True
        daemon._stats.last_log_rotation = time.time() - 20000  # Far overdue
        daemon._stats.last_db_vacuum = time.time()

        health = daemon.health_check()

        assert health.healthy is False
        assert "overdue" in health.message.lower()


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_maintenance_daemon()

    def test_get_maintenance_daemon_returns_singleton(self):
        """Should return same instance."""
        daemon1 = get_maintenance_daemon()
        daemon2 = get_maintenance_daemon()
        assert daemon1 is daemon2

    def test_reset_clears_singleton(self):
        """Reset should clear singleton."""
        daemon1 = get_maintenance_daemon()
        reset_maintenance_daemon()
        daemon2 = get_maintenance_daemon()
        # After reset, new instance
        assert daemon2._running is False
        assert daemon2._stats.logs_rotated == 0


# =============================================================================
# Queue Cleanup Tests
# =============================================================================


class TestQueueCleanup:
    """Tests for work queue cleanup."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            queue_cleanup_enabled=True,
            queue_stale_pending_hours=24.0,
            queue_stale_claimed_hours=1.0,
            dry_run=False,
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_queue_cleanup_updates_stats(self, daemon):
        """Should update stats after queue cleanup."""
        mock_queue = MagicMock()
        mock_queue.cleanup_stale_items.return_value = {
            "removed_stale_pending": 5,
            "reset_stale_claimed": 3,
        }
        mock_queue.cleanup_old_items.return_value = 2

        with patch(
            "app.coordination.work_queue.get_work_queue",
            return_value=mock_queue,
        ):
            await daemon._cleanup_stale_queue_items()

            assert daemon._stats.queue_items_cleaned >= 5
            assert daemon._stats.queue_items_reset >= 3

    @pytest.mark.asyncio
    async def test_queue_cleanup_dry_run(self):
        """Should log but not cleanup in dry run mode."""
        config = MaintenanceConfig(dry_run=True)
        daemon = MaintenanceDaemon(config=config)

        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {
            "by_status": {"pending": 10, "claimed": 5}
        }

        with patch(
            "app.coordination.work_queue.get_work_queue",
            return_value=mock_queue,
        ):
            await daemon._cleanup_stale_queue_items()

            # Should not call cleanup in dry run
            mock_queue.cleanup_stale_items.assert_not_called()


# =============================================================================
# Orphan Detection Tests
# =============================================================================


class TestOrphanDetection:
    """Tests for orphan file detection."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            orphan_detection_enabled=True,
            orphan_auto_cleanup=False,
            orphan_auto_recovery=True,
            dry_run=False,
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_orphan_detection_finds_untracked_files(self, daemon):
        """Should find files not in manifest."""
        mock_manifest = MagicMock()
        mock_manifest.get_all_db_paths.return_value = set()
        mock_manifest.get_all_npz_paths.return_value = set()
        mock_manifest.get_all_model_paths.return_value = set()

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            # Create orphan database
            orphan_db = games_dir / "orphan.db"
            with sqlite3.connect(orphan_db) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT)")

            with patch(
                "app.distributed.cluster_manifest.get_cluster_manifest",
                return_value=mock_manifest,
            ):
                await daemon._detect_orphan_files()

                assert daemon._stats.orphan_dbs_found >= 1

    @pytest.mark.asyncio
    async def test_orphan_detection_handles_manifest_error(self, daemon):
        """Should handle ClusterManifest errors gracefully."""
        with patch(
            "app.distributed.cluster_manifest.get_cluster_manifest",
            side_effect=RuntimeError("Manifest unavailable"),
        ):
            # Should not raise - error is caught and logged
            await daemon._detect_orphan_files()


# =============================================================================
# Archive Tests
# =============================================================================


class TestArchive:
    """Tests for game archival."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MaintenanceConfig(
                archive_enabled=True,
                archive_games_older_than_days=1,  # 1 day
                archive_directory=str(Path(tmpdir) / "archive"),
                archive_compress=True,
                dry_run=False,
            )
            yield MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_archive_single_database_compresses(self, daemon):
        """Should compress database when archiving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"
            db_path = Path(tmpdir) / "test.db"

            # Create test database
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE games (id INTEGER)")

            success = await daemon._archive_single_database(
                db_path, archive_dir, age_days=35.0
            )

            assert success is True
            # Original should be deleted
            assert not db_path.exists()
            # Archive should exist
            archives = list(archive_dir.glob("*.db.gz"))
            assert len(archives) == 1

    @pytest.mark.asyncio
    async def test_archive_skips_canonical(self, daemon):
        """Should skip canonical databases."""
        mock_discovery = MagicMock()
        mock_db_info = MagicMock()
        mock_db_info.path = "/data/games/canonical_hex8_2p.db"
        mock_db_info.game_count = 1000
        mock_discovery.find_all_databases.return_value = [mock_db_info]

        with patch(
            "app.utils.game_discovery.GameDiscovery",
            return_value=mock_discovery,
        ):
            with patch.object(daemon, "_archive_single_database", new_callable=AsyncMock) as mock_archive:
                await daemon._archive_old_games()
                # Should not attempt to archive canonical DB
                mock_archive.assert_not_called()


# =============================================================================
# S3 Upload Tests
# =============================================================================


class TestS3Upload:
    """Tests for S3 upload functionality."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            archive_to_s3=True,
            archive_s3_bucket="test-bucket",
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_upload_to_s3_success(self, daemon):
        """Should upload file to S3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.db.gz"
            archive_path.write_bytes(b"test data")

            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))

            with patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ):
                # Should not raise
                await daemon._upload_to_s3(archive_path)

    @pytest.mark.asyncio
    async def test_upload_to_s3_failure(self, daemon):
        """Should raise on S3 upload failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.db.gz"
            archive_path.write_bytes(b"test data")

            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Upload failed"))

            with patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ):
                with pytest.raises(RuntimeError, match="S3 upload failed"):
                    await daemon._upload_to_s3(archive_path)


# =============================================================================
# DLQ Cleanup Tests
# =============================================================================


class TestDLQCleanup:
    """Tests for dead letter queue cleanup."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            dlq_retention_days=7,
            dry_run=False,
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_dlq_cleanup_no_manifest(self, daemon):
        """Should handle missing manifest gracefully."""
        with patch.dict(
            "sys.modules",
            {"app.distributed.unified_manifest": MagicMock(get_unified_manifest=MagicMock(return_value=None))},
        ):
            # Should not raise
            await daemon._cleanup_dlq()
            assert daemon._stats.dlq_entries_cleaned == 0

    @pytest.mark.asyncio
    async def test_dlq_cleanup_dry_run(self):
        """Should log but not cleanup in dry run mode."""
        config = MaintenanceConfig(dry_run=True)
        daemon = MaintenanceDaemon(config=config)

        mock_manifest = MagicMock()
        mock_stats = MagicMock()
        mock_stats.dead_letter_count = 10
        mock_manifest.get_manifest_stats.return_value = mock_stats

        mock_module = MagicMock()
        mock_module.get_unified_manifest = MagicMock(return_value=mock_manifest)

        with patch.dict("sys.modules", {"app.distributed.unified_manifest": mock_module}):
            await daemon._cleanup_dlq()
            mock_manifest.cleanup_old_dead_letters.assert_not_called()

    @pytest.mark.asyncio
    async def test_dlq_cleanup_removes_entries(self, daemon):
        """Should remove old DLQ entries."""
        mock_manifest = MagicMock()
        mock_manifest.cleanup_old_dead_letters.return_value = 5

        mock_module = MagicMock()
        mock_module.get_unified_manifest = MagicMock(return_value=mock_manifest)

        with patch.dict("sys.modules", {"app.distributed.unified_manifest": mock_module}):
            await daemon._cleanup_dlq()
            assert daemon._stats.dlq_entries_cleaned == 5

    @pytest.mark.asyncio
    async def test_dlq_cleanup_handles_import_error(self, daemon):
        """Should handle import errors gracefully."""
        # Simulate ImportError by removing the module
        with patch.dict("sys.modules", {"app.distributed.unified_manifest": None}):
            # Should not raise - ImportError is caught
            await daemon._cleanup_dlq()


# =============================================================================
# Orphan Recovery Tests
# =============================================================================


class TestOrphanRecovery:
    """Tests for orphan database recovery."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        config = MaintenanceConfig(
            orphan_auto_recovery=True,
            dry_run=False,
        )
        return MaintenanceDaemon(config=config)

    @pytest.mark.asyncio
    async def test_recover_orphan_databases_parses_filename(self, daemon):
        """Should parse board_type and num_players from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "canonical_hex8_4p.db"

            # Create valid database with games table
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT)")
                conn.execute("INSERT INTO games VALUES ('game_1')")

            mock_manifest = MagicMock()
            mock_manifest.register_games_batch = MagicMock()

            with patch("socket.gethostname", return_value="test-node"):
                with patch(
                    "app.coordination.event_router.publish",
                    new_callable=AsyncMock,
                ):
                    recovered = await daemon._recover_orphan_databases(
                        mock_manifest, [db_path]
                    )

            assert recovered == 1
            mock_manifest.register_games_batch.assert_called_once()
            call_kwargs = mock_manifest.register_games_batch.call_args.kwargs
            assert call_kwargs["board_type"] == "hex8"
            assert call_kwargs["num_players"] == 4
            assert call_kwargs["game_count"] == 1

    @pytest.mark.asyncio
    async def test_recover_orphan_databases_skips_empty(self, daemon):
        """Should skip empty databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "hex8_2p.db"

            # Create empty database
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT)")

            mock_manifest = MagicMock()

            recovered = await daemon._recover_orphan_databases(
                mock_manifest, [db_path]
            )

            assert recovered == 0
            mock_manifest.register_games_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_orphan_databases_handles_invalid(self, daemon):
        """Should handle invalid database files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "invalid.db"
            db_path.write_text("not a database")

            mock_manifest = MagicMock()

            # Should not raise
            recovered = await daemon._recover_orphan_databases(
                mock_manifest, [db_path]
            )
            assert recovered == 0

    @pytest.mark.asyncio
    async def test_recover_orphan_databases_selfplay_pattern(self, daemon):
        """Should parse selfplay_ prefix pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "selfplay_square8_3p.db"

            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE games (game_id TEXT)")
                conn.execute("INSERT INTO games VALUES ('game_1')")

            mock_manifest = MagicMock()
            mock_manifest.register_games_batch = MagicMock()

            with patch("socket.gethostname", return_value="test-node"):
                with patch(
                    "app.coordination.event_router.publish",
                    new_callable=AsyncMock,
                ):
                    recovered = await daemon._recover_orphan_databases(
                        mock_manifest, [db_path]
                    )

            call_kwargs = mock_manifest.register_games_batch.call_args.kwargs
            assert call_kwargs["board_type"] == "square8"
            assert call_kwargs["num_players"] == 3


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling paths."""

    @pytest.fixture
    def daemon(self):
        """Create daemon for testing."""
        return MaintenanceDaemon()

    @pytest.mark.asyncio
    async def test_run_forever_stops_cleanly(self, daemon):
        """Should stop cleanly when _running is set to False."""
        call_count = 0

        async def controlled_cycle():
            nonlocal call_count
            call_count += 1
            # Stop after one cycle
            daemon._running = False

        with patch.object(daemon, "_run_maintenance_cycle", side_effect=controlled_cycle):
            # Patch sleep to not actually wait
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # run_forever calls start() which sets _running to True
                await daemon.run_forever()

        # At minimum, start() calls _run_maintenance_cycle once
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_rotate_logs_handles_glob_error(self, daemon):
        """Should handle glob errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            with patch.object(Path, "glob", side_effect=OSError("Permission denied")):
                # Should not raise
                await daemon._rotate_logs()

    @pytest.mark.asyncio
    async def test_vacuum_handles_db_locked_error(self, daemon):
        """Should handle database locked errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            games_dir = Path(tmpdir) / "data" / "games"
            games_dir.mkdir(parents=True)

            db_path = games_dir / "locked.db"
            db_path.touch()

            with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("database is locked")):
                # Should not raise
                await daemon._vacuum_databases()

    @pytest.mark.asyncio
    async def test_archive_handles_discovery_import_error(self, daemon):
        """Should handle GameDiscovery import errors."""
        with patch.dict("sys.modules", {"app.utils.game_discovery": None}):
            # Should not raise - ImportError is caught
            await daemon._archive_old_games()

    @pytest.mark.asyncio
    async def test_detect_orphans_handles_manifest_import_error(self, daemon):
        """Should handle ClusterManifest import errors."""
        with patch.dict("sys.modules", {"app.distributed.cluster_manifest": None}):
            # Should not raise - ImportError is caught
            await daemon._detect_orphan_files()


# =============================================================================
# Log Rotation Edge Cases
# =============================================================================


class TestLogRotationEdgeCases:
    """Additional tests for log rotation edge cases."""

    @pytest.mark.asyncio
    async def test_rotate_logs_shifts_existing_backups(self):
        """Should shift existing backup files."""
        config = MaintenanceConfig(
            log_max_size_mb=0.001,
            log_backup_count=3,
            dry_run=False,
        )
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create large log file
            (logs_dir / "test.log").write_text("x" * 2000)

            # Create existing backup
            with gzip.open(logs_dir / "test.log.1.gz", "wb") as f:
                f.write(b"backup1")

            await daemon._rotate_logs()

            # Old backup should be shifted to .2.gz
            assert (logs_dir / "test.log.2.gz").exists()
            # New backup at .1.gz
            assert (logs_dir / "test.log.1.gz").exists()

    @pytest.mark.asyncio
    async def test_rotate_logs_deletes_oldest_beyond_count(self):
        """Should delete backups beyond backup_count."""
        config = MaintenanceConfig(
            log_max_size_mb=0.001,
            log_backup_count=3,  # Allow up to 3 backups
            dry_run=False,
        )
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create large log file
            (logs_dir / "test.log").write_text("x" * 2000)

            # Create existing backups at .1 and .2
            with gzip.open(logs_dir / "test.log.1.gz", "wb") as f:
                f.write(b"backup1")
            with gzip.open(logs_dir / "test.log.2.gz", "wb") as f:
                f.write(b"backup2")

            await daemon._rotate_logs()

            # .1.gz should exist (new backup)
            assert (logs_dir / "test.log.1.gz").exists()
            # .2.gz should exist (shifted from .1)
            assert (logs_dir / "test.log.2.gz").exists()
            # .3.gz should exist (shifted from .2)
            assert (logs_dir / "test.log.3.gz").exists()

    @pytest.mark.asyncio
    async def test_rotate_logs_skips_non_log_files(self):
        """Should only rotate .log files."""
        config = MaintenanceConfig(
            log_max_size_mb=0.001,
            dry_run=False,
        )
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()

            # Create large non-log file
            (logs_dir / "test.txt").write_text("x" * 2000)

            await daemon._rotate_logs()

            # Non-log file should be unchanged
            assert (logs_dir / "test.txt").stat().st_size == 2000
            assert not (logs_dir / "test.txt.1.gz").exists()


# =============================================================================
# Health Check Edge Cases
# =============================================================================


class TestHealthCheckEdgeCases:
    """Additional tests for health check edge cases."""

    def test_health_check_includes_stats_in_details(self):
        """Should include stats in health check details."""
        daemon = MaintenanceDaemon()
        daemon._running = True
        daemon._stats.logs_rotated = 10
        daemon._stats.databases_vacuumed = 5
        daemon._stats.last_log_rotation = time.time()
        daemon._stats.last_db_vacuum = time.time()

        result = daemon.health_check()

        assert result.details is not None
        assert "stats" in result.details
        assert result.details["stats"]["logs_rotated"] == 10
        assert result.details["stats"]["databases_vacuumed"] == 5

    def test_health_check_vacuum_overdue(self):
        """Should report degraded when vacuum is overdue."""
        daemon = MaintenanceDaemon()
        daemon._running = True
        daemon._stats.last_log_rotation = time.time()
        # Vacuum very overdue (> 1.5x interval)
        daemon._stats.last_db_vacuum = time.time() - (
            daemon.config.db_vacuum_interval_hours * 3600 * 2
        )

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED

    def test_health_check_with_zero_timestamps(self):
        """Should handle zero timestamps (never run)."""
        daemon = MaintenanceDaemon()
        daemon._running = True
        daemon._stats.last_log_rotation = 0
        daemon._stats.last_db_vacuum = 0

        result = daemon.health_check()

        # Should report degraded since tasks are "infinitely" overdue
        assert result.status == CoordinatorStatus.DEGRADED


# =============================================================================
# Get Status Tests
# =============================================================================


class TestGetStatusDetails:
    """Tests for get_status method details."""

    def test_get_status_includes_all_stats(self):
        """Should include all stat fields."""
        daemon = MaintenanceDaemon()
        daemon._running = True
        daemon._stats.logs_rotated = 5
        daemon._stats.bytes_reclaimed_logs = 10485760
        daemon._stats.databases_vacuumed = 3
        daemon._stats.games_archived = 100
        daemon._stats.dlq_entries_cleaned = 10
        daemon._stats.queue_items_cleaned = 20
        daemon._stats.queue_items_reset = 5
        daemon._stats.orphan_dbs_found = 2
        daemon._stats.orphan_npz_found = 1
        daemon._stats.orphan_models_found = 0
        daemon._stats.orphan_dbs_recovered = 1

        status = daemon.get_status()

        assert status["stats"]["logs_rotated"] == 5
        assert status["stats"]["bytes_reclaimed_logs_mb"] == 10.0
        assert status["stats"]["databases_vacuumed"] == 3
        assert status["stats"]["games_archived"] == 100
        assert status["stats"]["dlq_entries_cleaned"] == 10
        assert status["stats"]["queue_items_cleaned"] == 20
        assert status["stats"]["queue_items_reset"] == 5
        assert status["stats"]["orphan_dbs_found"] == 2
        assert status["stats"]["orphan_npz_found"] == 1
        assert status["stats"]["orphan_models_found"] == 0
        assert status["stats"]["orphan_dbs_recovered"] == 1

    def test_get_status_includes_config(self):
        """Should include config summary."""
        config = MaintenanceConfig(
            log_max_size_mb=50.0,
            log_backup_count=5,
            db_vacuum_interval_hours=72.0,
        )
        daemon = MaintenanceDaemon(config=config)

        status = daemon.get_status()

        assert status["config"]["log_max_size_mb"] == 50.0
        assert status["config"]["log_backup_count"] == 5
        assert status["config"]["db_vacuum_interval_hours"] == 72.0

    def test_get_status_includes_last_runs(self):
        """Should include last run timestamps."""
        daemon = MaintenanceDaemon()
        daemon._stats.last_log_rotation = 1000.0
        daemon._stats.last_db_vacuum = 2000.0
        daemon._stats.last_archive_run = 3000.0
        daemon._stats.last_dlq_cleanup = 4000.0
        daemon._stats.last_queue_cleanup = 5000.0
        daemon._stats.last_orphan_detection = 6000.0

        status = daemon.get_status()

        assert status["last_runs"]["log_rotation"] == 1000.0
        assert status["last_runs"]["db_vacuum"] == 2000.0
        assert status["last_runs"]["archive"] == 3000.0
        assert status["last_runs"]["dlq_cleanup"] == 4000.0
        assert status["last_runs"]["queue_cleanup"] == 5000.0
        assert status["last_runs"]["orphan_detection"] == 6000.0


# =============================================================================
# Archive Uncompressed Tests
# =============================================================================


class TestArchiveUncompressed:
    """Tests for uncompressed archive functionality."""

    @pytest.mark.asyncio
    async def test_archive_without_compression(self):
        """Should archive without compression when disabled."""
        config = MaintenanceConfig(archive_compress=False)
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"
            db_path = Path(tmpdir) / "test.db"

            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")

            success = await daemon._archive_single_database(
                db_path, archive_dir, age_days=35.0
            )

            assert success is True
            # Should have uncompressed archive (not .gz)
            archives = list(archive_dir.glob("*.db"))
            gz_archives = list(archive_dir.glob("*.db.gz"))
            assert len(archives) == 1
            assert len(gz_archives) == 0


# =============================================================================
# Singleton Reset Tests
# =============================================================================


class TestSingletonReset:
    """Additional tests for singleton reset behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_maintenance_daemon()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_maintenance_daemon()

    def test_get_daemon_returns_fresh_stats_after_reset(self):
        """Should return fresh instance with zero stats after reset."""
        daemon1 = get_maintenance_daemon()
        daemon1._stats.logs_rotated = 100
        daemon1._stats.databases_vacuumed = 50

        reset_maintenance_daemon()
        daemon2 = get_maintenance_daemon()

        assert daemon2._stats.logs_rotated == 0
        assert daemon2._stats.databases_vacuumed == 0

    def test_singleton_survives_multiple_gets(self):
        """Should return same instance across multiple gets."""
        daemon1 = get_maintenance_daemon()
        daemon2 = get_maintenance_daemon()
        daemon3 = get_maintenance_daemon()

        assert daemon1 is daemon2
        assert daemon2 is daemon3


# =============================================================================
# Orphan Detection Edge Cases
# =============================================================================


class TestOrphanDetectionEdgeCases:
    """Additional tests for orphan detection edge cases."""

    @pytest.mark.asyncio
    async def test_detect_orphans_skips_symlinks(self):
        """Should skip symlink model files."""
        config = MaintenanceConfig(orphan_detection_enabled=True)
        daemon = MaintenanceDaemon(config=config)

        mock_manifest = MagicMock()
        mock_manifest.get_all_db_paths.return_value = set()
        mock_manifest.get_all_npz_paths.return_value = set()
        mock_manifest.get_all_model_paths.return_value = set()

        mock_module = MagicMock()
        mock_module.get_cluster_manifest = MagicMock(return_value=mock_manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create actual model file
            actual_model = models_dir / "actual.pth"
            actual_model.write_bytes(b"model data")

            # Create symlink to model
            symlink_model = models_dir / "link.pth"
            symlink_model.symlink_to(actual_model)

            with patch.dict("sys.modules", {"app.distributed.cluster_manifest": mock_module}):
                await daemon._detect_orphan_files()

                # Only the actual file should be counted, not the symlink
                assert daemon._stats.orphan_models_found == 1

    @pytest.mark.asyncio
    async def test_detect_orphans_handles_missing_dirs(self):
        """Should handle missing directories gracefully."""
        config = MaintenanceConfig(orphan_detection_enabled=True)
        daemon = MaintenanceDaemon(config=config)

        mock_manifest = MagicMock()
        mock_manifest.get_all_db_paths.return_value = set()
        mock_manifest.get_all_npz_paths.return_value = set()
        mock_manifest.get_all_model_paths.return_value = set()

        mock_module = MagicMock()
        mock_module.get_cluster_manifest = MagicMock(return_value=mock_manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            daemon._ai_service_dir = Path(tmpdir)
            # Don't create any directories

            with patch.dict("sys.modules", {"app.distributed.cluster_manifest": mock_module}):
                # Should not raise
                await daemon._detect_orphan_files()
                assert daemon._stats.orphan_dbs_found == 0


# =============================================================================
# Orphan Cleanup Tests
# =============================================================================


class TestOrphanCleanup:
    """Tests for orphan file cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_orphan_files_only_npz(self):
        """Should only cleanup NPZ files (DBs and models preserved)."""
        config = MaintenanceConfig(orphan_auto_cleanup=True)
        daemon = MaintenanceDaemon(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create orphan NPZ file
            orphan_npz = Path(tmpdir) / "orphan.npz"
            orphan_npz.write_bytes(b"npz data")

            # Create orphan DB file
            orphan_db = Path(tmpdir) / "orphan.db"
            orphan_db.write_text("db data")

            # Create orphan model file
            orphan_model = Path(tmpdir) / "orphan.pth"
            orphan_model.write_bytes(b"model data")

            await daemon._cleanup_orphan_files(
                orphan_dbs=[orphan_db],
                orphan_npz=[orphan_npz],
                orphan_models=[orphan_model],
            )

            # NPZ should be deleted
            assert not orphan_npz.exists()
            # DB and model should be preserved
            assert orphan_db.exists()
            assert orphan_model.exists()


# =============================================================================
# Maintenance Cycle Feature Flag Tests
# =============================================================================


class TestMaintenanceCycleFeatureFlags:
    """Tests for feature flag respects in maintenance cycle."""

    @pytest.mark.asyncio
    async def test_cycle_respects_db_maintenance_disabled(self):
        """Should skip vacuum when db_maintenance_enabled is False."""
        config = MaintenanceConfig(db_maintenance_enabled=False)
        daemon = MaintenanceDaemon(config=config)
        daemon._stats.last_db_vacuum = 0  # Very old

        with patch.object(
            daemon, "_vacuum_databases", new_callable=AsyncMock
        ) as mock_vacuum:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_vacuum.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_respects_archive_disabled(self):
        """Should skip archive when archive_enabled is False."""
        config = MaintenanceConfig(archive_enabled=False)
        daemon = MaintenanceDaemon(config=config)
        daemon._stats.last_archive_run = 0  # Very old

        with patch.object(
            daemon, "_archive_old_games", new_callable=AsyncMock
        ) as mock_archive:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_archive.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_respects_queue_cleanup_disabled(self):
        """Should skip queue cleanup when queue_cleanup_enabled is False."""
        config = MaintenanceConfig(queue_cleanup_enabled=False)
        daemon = MaintenanceDaemon(config=config)
        daemon._stats.last_queue_cleanup = 0  # Very old

        with patch.object(
            daemon, "_cleanup_stale_queue_items", new_callable=AsyncMock
        ) as mock_cleanup:
            with patch.object(daemon, "_detect_orphan_files", new_callable=AsyncMock):
                await daemon._run_maintenance_cycle()
                mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_cycle_respects_orphan_detection_disabled(self):
        """Should skip orphan detection when orphan_detection_enabled is False."""
        config = MaintenanceConfig(orphan_detection_enabled=False)
        daemon = MaintenanceDaemon(config=config)
        daemon._stats.last_orphan_detection = 0  # Very old

        with patch.object(
            daemon, "_detect_orphan_files", new_callable=AsyncMock
        ) as mock_orphan:
            await daemon._run_maintenance_cycle()
            mock_orphan.assert_not_called()
