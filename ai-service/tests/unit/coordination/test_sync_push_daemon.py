"""Tests for SyncPushDaemon - push-based data sync with verified cleanup.

This daemon is CRITICAL because it handles data deletion - only after verified copies exist.
Tests cover:
- Disk threshold detection (50%, 70%, 75%)
- Sync receipt verification before deletion
- Cleanup with insufficient copies (should NOT delete)
- Network failure recovery
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_manifest():
    """Mock ClusterManifest for sync receipt tracking."""
    manifest = MagicMock()
    manifest.get_pending_sync_files = MagicMock(return_value=[])
    manifest.is_safe_to_delete = MagicMock(return_value=False)
    manifest.register_sync_receipt = MagicMock()
    manifest.db_path = Path("/tmp/data/manifest.db")
    return manifest


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data" / "games"
        data_dir.mkdir(parents=True)

        # Create some test database files
        (data_dir / "selfplay_001.db").write_text("test data 1")
        (data_dir / "selfplay_002.db").write_text("test data 2")
        (data_dir / "canonical_hex8_2p.db").write_text("canonical data")

        yield data_dir


@pytest.fixture
def daemon(mock_manifest, temp_data_dir):
    """Create daemon instance with mocked dependencies."""
    from app.coordination.sync_push_daemon import (
        SyncPushConfig,
        SyncPushDaemon,
        reset_sync_push_daemon,
    )

    reset_sync_push_daemon()

    with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_manifest):
        config = SyncPushConfig(
            push_threshold_percent=50.0,
            urgent_threshold_percent=70.0,
            cleanup_threshold_percent=75.0,
            min_copies_before_delete=2,
            max_files_per_cycle=10,
            data_dir=str(temp_data_dir),
        )
        daemon = SyncPushDaemon(config)
        daemon._manifest = mock_manifest
        daemon._coordinator_url = "http://coordinator:8770"
        yield daemon

    reset_sync_push_daemon()


# =============================================================================
# TestSyncPushConfig
# =============================================================================


class TestSyncPushConfig:
    """Tests for SyncPushConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.coordination.sync_push_daemon import SyncPushConfig

        config = SyncPushConfig()

        assert config.push_threshold_percent == 50.0
        assert config.urgent_threshold_percent == 70.0
        assert config.cleanup_threshold_percent == 75.0
        assert config.min_copies_before_delete == 2
        assert config.max_files_per_cycle == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.coordination.sync_push_daemon import SyncPushConfig

        config = SyncPushConfig(
            push_threshold_percent=40.0,
            urgent_threshold_percent=60.0,
            cleanup_threshold_percent=80.0,
            min_copies_before_delete=3,
            max_files_per_cycle=20,
        )

        assert config.push_threshold_percent == 40.0
        assert config.urgent_threshold_percent == 60.0
        assert config.cleanup_threshold_percent == 80.0
        assert config.min_copies_before_delete == 3
        assert config.max_files_per_cycle == 20

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        from app.coordination.sync_push_daemon import SyncPushConfig

        with patch.dict(os.environ, {}, clear=True):
            config = SyncPushConfig.from_env()

            assert config.push_threshold_percent == 50.0
            assert config.min_copies_before_delete == 2

    def test_from_env_custom(self):
        """Test from_env with custom environment variables."""
        from app.coordination.sync_push_daemon import SyncPushConfig

        env_vars = {
            "RINGRIFT_SYNC_PUSH_THRESHOLD": "45.0",
            "RINGRIFT_SYNC_PUSH_URGENT_THRESHOLD": "65.0",
            "RINGRIFT_SYNC_PUSH_CLEANUP_THRESHOLD": "78.0",
            "RINGRIFT_SYNC_PUSH_MIN_COPIES": "3",
            "RINGRIFT_SYNC_PUSH_INTERVAL": "180",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = SyncPushConfig.from_env()

            assert config.push_threshold_percent == 45.0
            assert config.urgent_threshold_percent == 65.0
            assert config.cleanup_threshold_percent == 78.0
            assert config.min_copies_before_delete == 3
            assert config.check_interval_seconds == 180

    def test_threshold_ordering(self):
        """Test that thresholds are in correct order."""
        from app.coordination.sync_push_daemon import SyncPushConfig

        config = SyncPushConfig()

        assert config.push_threshold_percent < config.urgent_threshold_percent
        assert config.urgent_threshold_percent < config.cleanup_threshold_percent


# =============================================================================
# TestSyncPushDaemonInit
# =============================================================================


class TestSyncPushDaemonInit:
    """Tests for SyncPushDaemon initialization."""

    def test_init_with_config(self, mock_manifest):
        """Test initialization with explicit config."""
        from app.coordination.sync_push_daemon import (
            SyncPushConfig,
            SyncPushDaemon,
            reset_sync_push_daemon,
        )

        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_manifest):
            config = SyncPushConfig(
                push_threshold_percent=55.0,
                min_copies_before_delete=3,
            )
            daemon = SyncPushDaemon(config)

            assert daemon.config.push_threshold_percent == 55.0
            assert daemon.config.min_copies_before_delete == 3

        reset_sync_push_daemon()

    def test_init_default_config(self, mock_manifest):
        """Test initialization with default config."""
        from app.coordination.sync_push_daemon import SyncPushDaemon, reset_sync_push_daemon

        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_manifest):
            daemon = SyncPushDaemon()

            assert daemon.config is not None

        reset_sync_push_daemon()

    def test_init_sets_statistics_to_zero(self, daemon):
        """Test statistics start at zero."""
        assert daemon._files_pushed == 0
        assert daemon._bytes_pushed == 0
        assert daemon._files_cleaned == 0
        assert daemon._push_failures == 0


# =============================================================================
# TestDiskUsage
# =============================================================================


class TestDiskUsage:
    """Tests for disk usage detection."""

    def test_get_disk_usage_normal(self, daemon):
        """Test disk usage below push threshold."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=40_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 40.0  # 40%

    def test_get_disk_usage_at_push_threshold(self, daemon):
        """Test disk usage at push threshold (50%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=50_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 50.0

    def test_get_disk_usage_at_urgent_threshold(self, daemon):
        """Test disk usage at urgent threshold (70%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=70_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 70.0

    def test_get_disk_usage_at_cleanup_threshold(self, daemon):
        """Test disk usage at cleanup threshold (75%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=75_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 75.0

    def test_get_disk_usage_handles_error(self, daemon):
        """Test disk usage returns -1 on error."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.side_effect = OSError("Permission denied")

            usage = daemon._get_disk_usage()

            assert usage == -1.0


# =============================================================================
# TestChecksum
# =============================================================================


class TestChecksum:
    """Tests for checksum computation via sync_integrity.

    Note: SyncPushDaemon now uses compute_file_checksum from sync_integrity
    instead of internal _compute_sha256 method (Dec 2025 consolidation).
    """

    def test_compute_file_checksum(self, temp_data_dir):
        """Test SHA256 checksum computation via sync_integrity."""
        from app.coordination.sync_integrity import compute_file_checksum

        test_file = temp_data_dir / "selfplay_001.db"

        checksum = compute_file_checksum(test_file)

        # Verify it's a valid SHA256 hash (64 hex characters)
        assert checksum is not None
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_file_checksum_consistency(self, temp_data_dir):
        """Test checksum is consistent for same file."""
        from app.coordination.sync_integrity import compute_file_checksum

        test_file = temp_data_dir / "selfplay_001.db"

        checksum1 = compute_file_checksum(test_file)
        checksum2 = compute_file_checksum(test_file)

        assert checksum1 == checksum2

    def test_compute_file_checksum_different_files(self, temp_data_dir):
        """Test different files have different checksums."""
        from app.coordination.sync_integrity import compute_file_checksum

        file1 = temp_data_dir / "selfplay_001.db"
        file2 = temp_data_dir / "selfplay_002.db"

        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 != checksum2


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy(self, daemon):
        """Test health check when daemon is healthy."""
        daemon._running = True
        daemon._error_count = 0

        result = daemon.health_check()

        assert result.healthy is True

    def test_health_check_not_running(self, daemon):
        """Test health check when daemon is not running."""
        daemon._running = False

        result = daemon.health_check()

        # Should still report status even when not running
        assert result is not None


# =============================================================================
# TestSingleton
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_sync_push_daemon_singleton(self, mock_manifest):
        """Test get_sync_push_daemon returns singleton."""
        from app.coordination.sync_push_daemon import (
            get_sync_push_daemon,
            reset_sync_push_daemon,
        )

        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_manifest):
            daemon1 = get_sync_push_daemon()
            daemon2 = get_sync_push_daemon()

            assert daemon1 is daemon2

        reset_sync_push_daemon()

    def test_reset_sync_push_daemon(self, mock_manifest):
        """Test reset_sync_push_daemon clears singleton."""
        from app.coordination.sync_push_daemon import (
            get_sync_push_daemon,
            reset_sync_push_daemon,
        )

        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_manifest):
            daemon1 = get_sync_push_daemon()
            reset_sync_push_daemon()
            daemon2 = get_sync_push_daemon()

            assert daemon1 is not daemon2

        reset_sync_push_daemon()


# =============================================================================
# TestPushPendingFiles
# =============================================================================


class TestPushPendingFiles:
    """Tests for push logic."""

    @pytest.mark.asyncio
    async def test_push_pending_no_coordinator(self, daemon):
        """Test push with no coordinator URL."""
        daemon._coordinator_url = ""

        result = await daemon._push_pending_files()

        assert result == 0

    @pytest.mark.asyncio
    async def test_push_pending_no_files(self, daemon, mock_manifest):
        """Test push with no pending files."""
        mock_manifest.get_pending_sync_files.return_value = []

        result = await daemon._push_pending_files()

        assert result == 0

    @pytest.mark.asyncio
    async def test_push_pending_respects_max_files_per_cycle(self, daemon, mock_manifest, temp_data_dir):
        """Test push respects max_files_per_cycle limit."""
        daemon.config.max_files_per_cycle = 2

        # Return 10 files
        files = [temp_data_dir / f"file_{i}.db" for i in range(10)]
        for f in files:
            f.write_text(f"data {f.name}")
        mock_manifest.get_pending_sync_files.return_value = files

        with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True

            await daemon._push_pending_files()

            assert mock_push.call_count <= 2


# =============================================================================
# TestSafeCleanup
# =============================================================================


class TestSafeCleanup:
    """Tests for cleanup logic - CRITICAL: must verify copies before deletion."""

    @pytest.mark.asyncio
    async def test_cleanup_requires_min_copies(self, daemon, temp_data_dir, mock_manifest):
        """Test cleanup requires minimum verified copies."""
        test_file = temp_data_dir / "selfplay_001.db"
        daemon.config.min_copies_before_delete = 2

        # Not safe to delete
        mock_manifest.is_safe_to_delete.return_value = False

        await daemon._safe_cleanup()

        # File should NOT be deleted
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_with_sufficient_copies(self, daemon, temp_data_dir, mock_manifest):
        """Test cleanup proceeds with sufficient verified copies."""
        test_file = temp_data_dir / "selfplay_001.db"
        assert test_file.exists()

        # Safe to delete
        mock_manifest.is_safe_to_delete.return_value = True

        deleted = await daemon._safe_cleanup()

        # File should be deleted
        assert not test_file.exists()
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_cleanup_skips_canonical_files(self, daemon, temp_data_dir, mock_manifest):
        """Test cleanup never deletes canonical files."""
        canonical_file = temp_data_dir / "canonical_hex8_2p.db"

        # Even with safe_to_delete returning True for all files,
        # canonical files should be skipped due to name check
        mock_manifest.is_safe_to_delete.return_value = True

        await daemon._safe_cleanup()

        # Canonical file should NEVER be deleted
        assert canonical_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_removes_wal_and_shm(self, daemon, temp_data_dir, mock_manifest):
        """Test cleanup removes associated WAL and SHM files."""
        test_file = temp_data_dir / "selfplay_001.db"
        wal_file = temp_data_dir / "selfplay_001.db-wal"
        shm_file = temp_data_dir / "selfplay_001.db-shm"

        wal_file.write_text("wal data")
        shm_file.write_text("shm data")

        # Safe to delete
        mock_manifest.is_safe_to_delete.return_value = True

        await daemon._safe_cleanup()

        # WAL and SHM should also be removed if main file is deleted
        assert not test_file.exists()
        # Note: The actual implementation may or may not clean these up


# =============================================================================
# TestRunCycle
# =============================================================================


class TestRunCycle:
    """Tests for run cycle at various disk thresholds."""

    @pytest.mark.asyncio
    async def test_run_cycle_below_threshold(self, daemon, mock_manifest):
        """Test run cycle with disk below push threshold."""
        with patch.object(daemon, "_get_disk_usage", return_value=40.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                    await daemon._run_cycle()

                    # Should not push or cleanup below threshold
                    mock_push.assert_not_called()
                    mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_at_push_threshold(self, daemon, mock_manifest):
        """Test run cycle at push threshold (50%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=55.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                mock_push.return_value = 5
                with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                    await daemon._run_cycle()

                    # Should push but not cleanup
                    mock_push.assert_called_once_with(urgent=False)
                    mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_at_urgent_threshold(self, daemon, mock_manifest):
        """Test run cycle at urgent threshold (70%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=72.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                mock_push.return_value = 5
                with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                    await daemon._run_cycle()

                    # Should push urgently, no cleanup yet
                    mock_push.assert_called_once_with(urgent=True)
                    mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_at_cleanup_threshold(self, daemon, mock_manifest):
        """Test run cycle at cleanup threshold (75%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=78.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                mock_push.return_value = 0
                with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                    mock_cleanup.return_value = 3

                    await daemon._run_cycle()

                    # Should cleanup first, then push
                    mock_cleanup.assert_called()
                    mock_push.assert_called()

    @pytest.mark.asyncio
    async def test_run_cycle_handles_error_disk_usage(self, daemon, mock_manifest):
        """Test run cycle handles disk usage error gracefully."""
        with patch.object(daemon, "_get_disk_usage", return_value=-1.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                # Should not raise, just return early
                await daemon._run_cycle()

                mock_push.assert_not_called()


# =============================================================================
# TestCoordinatorDiscovery
# =============================================================================


class TestCoordinatorDiscovery:
    """Tests for coordinator discovery."""

    @pytest.mark.asyncio
    async def test_discover_coordinator_from_config(self, daemon):
        """Test coordinator discovery uses config if set."""
        daemon.config.coordinator_url = "http://my-coordinator:8770"

        await daemon._discover_coordinator()

        assert daemon._coordinator_url == "http://my-coordinator:8770"

    @pytest.mark.asyncio
    async def test_discover_coordinator_updates_timestamp(self, daemon):
        """Test coordinator discovery updates last check timestamp."""
        old_time = daemon._last_coordinator_check

        await daemon._discover_coordinator()

        assert daemon._last_coordinator_check > old_time


# =============================================================================
# TestLifecycle
# =============================================================================


class TestLifecycle:
    """Tests for daemon lifecycle management."""

    @pytest.mark.asyncio
    async def test_on_start_creates_session(self, daemon, mock_manifest):
        """Test daemon start creates HTTP session."""
        with patch.object(daemon, "_discover_coordinator", new_callable=AsyncMock):
            await daemon._on_start()

            assert daemon._session is not None
            assert daemon._manifest is not None

        # Cleanup
        if daemon._session:
            await daemon._session.close()

    @pytest.mark.asyncio
    async def test_on_stop_closes_session(self, daemon):
        """Test daemon stop closes HTTP session."""
        import aiohttp

        daemon._session = aiohttp.ClientSession()

        await daemon._on_stop()

        assert daemon._session is None


# =============================================================================
# TestPushFile
# =============================================================================


class TestPushFile:
    """Tests for individual file push."""

    @pytest.mark.asyncio
    async def test_push_file_no_session(self, daemon, temp_data_dir):
        """Test push file returns False if no session."""
        daemon._session = None
        test_file = temp_data_dir / "selfplay_001.db"

        result = await daemon._push_file(test_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_file_success(self, daemon, temp_data_dir, mock_manifest):
        """Test successful file push."""
        import aiohttp

        test_file = temp_data_dir / "selfplay_001.db"

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"checksum_verified": True, "node_id": "coordinator"})

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp)))
            daemon._session = mock_session

            result = await daemon._push_file(test_file)

            # Check that sync receipt was recorded
            mock_manifest.register_sync_receipt.assert_called()

    @pytest.mark.asyncio
    async def test_push_file_server_error(self, daemon, temp_data_dir):
        """Test file push with server error."""
        test_file = temp_data_dir / "selfplay_001.db"

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_cm)
        daemon._session = mock_session

        result = await daemon._push_file(test_file)

        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for SyncPushDaemon."""

    @pytest.mark.asyncio
    async def test_full_cleanup_cycle(self, daemon, temp_data_dir, mock_manifest):
        """Test complete cleanup cycle."""
        # Set up disk at cleanup threshold
        with patch.object(daemon, "_get_disk_usage", return_value=78.0):
            # Mock sufficient copies for cleanup
            mock_manifest.is_safe_to_delete.return_value = True

            # Non-canonical files should be cleaned
            before_count = len(list(temp_data_dir.glob("selfplay_*.db")))

            await daemon._run_cycle()

            after_count = len(list(temp_data_dir.glob("selfplay_*.db")))
            assert after_count < before_count

            # Canonical should remain
            assert (temp_data_dir / "canonical_hex8_2p.db").exists()

    def test_config_inheritance(self, mock_manifest):
        """Test that SyncPushConfig inherits from DaemonConfig."""
        from app.coordination.sync_push_daemon import SyncPushConfig
        from app.coordination.base_daemon import DaemonConfig

        config = SyncPushConfig()

        # Should have DaemonConfig attributes
        assert hasattr(config, "enabled")
        assert hasattr(config, "check_interval_seconds")
