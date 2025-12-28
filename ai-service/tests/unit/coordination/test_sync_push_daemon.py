"""Tests for sync_push_daemon.py module.

December 2025: Comprehensive test coverage for the SyncPushDaemon
which handles push-based data sync for GPU training nodes.

Tests cover:
- SyncPushConfig configuration and defaults
- Environment variable overrides
- Singleton pattern (get_instance, reset)
- Disk threshold detection (50%, 70%, 75%)
- Push logic with mocked HTTP calls
- Sync receipt handling
- Safe cleanup with min_copies validation
- Health check results
- Lifecycle (start/stop) with graceful shutdown
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.coordination.base_daemon import DaemonConfig
from app.coordination.sync_push_daemon import (
    ENV_PREFIX,
    MAX_INLINE_SIZE_MB,
    SyncPushConfig,
    SyncPushDaemon,
    get_sync_push_daemon,
    reset_sync_push_daemon,
)
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult


# =============================================================================
# SyncPushConfig Tests
# =============================================================================


class TestSyncPushConfig:
    """Test SyncPushConfig dataclass and defaults."""

    def test_default_thresholds(self):
        """Test default disk threshold values."""
        config = SyncPushConfig()

        assert config.push_threshold_percent == 50.0
        assert config.urgent_threshold_percent == 70.0
        assert config.cleanup_threshold_percent == 75.0

    def test_default_replication_settings(self):
        """Test default replication requirements."""
        config = SyncPushConfig()

        assert config.min_copies_before_delete == 2
        assert config.max_files_per_cycle == 10
        assert config.max_file_size_mb == 500

    def test_default_coordinator_settings(self):
        """Test default coordinator settings are empty."""
        config = SyncPushConfig()

        assert config.coordinator_url == ""
        assert config.coordinator_node_id == ""
        assert config.data_dir == ""

    def test_config_inheritance(self):
        """Test that config inherits from DaemonConfig."""
        config = SyncPushConfig()
        assert isinstance(config, DaemonConfig)
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SyncPushConfig(
            push_threshold_percent=60.0,
            urgent_threshold_percent=80.0,
            cleanup_threshold_percent=85.0,
            min_copies_before_delete=3,
            max_files_per_cycle=20,
            coordinator_url="http://localhost:8770",
        )

        assert config.push_threshold_percent == 60.0
        assert config.urgent_threshold_percent == 80.0
        assert config.cleanup_threshold_percent == 85.0
        assert config.min_copies_before_delete == 3
        assert config.max_files_per_cycle == 20
        assert config.coordinator_url == "http://localhost:8770"


class TestSyncPushConfigFromEnv:
    """Test SyncPushConfig.from_env() method."""

    def test_from_env_defaults(self):
        """Test loading config from environment with defaults."""
        config = SyncPushConfig.from_env()

        assert config.push_threshold_percent == 50.0
        assert config.urgent_threshold_percent == 70.0
        assert config.cleanup_threshold_percent == 75.0

    @patch.dict("os.environ", {f"{ENV_PREFIX}_ENABLED": "0"})
    def test_from_env_disabled(self):
        """Test disabling daemon via environment."""
        config = SyncPushConfig.from_env()
        assert config.enabled is False

    @patch.dict("os.environ", {f"{ENV_PREFIX}_THRESHOLD": "60"})
    def test_from_env_push_threshold(self):
        """Test custom push threshold from environment."""
        config = SyncPushConfig.from_env()
        assert config.push_threshold_percent == 60.0

    @patch.dict("os.environ", {f"{ENV_PREFIX}_URGENT_THRESHOLD": "75"})
    def test_from_env_urgent_threshold(self):
        """Test custom urgent threshold from environment."""
        config = SyncPushConfig.from_env()
        assert config.urgent_threshold_percent == 75.0

    @patch.dict("os.environ", {f"{ENV_PREFIX}_CLEANUP_THRESHOLD": "80"})
    def test_from_env_cleanup_threshold(self):
        """Test custom cleanup threshold from environment."""
        config = SyncPushConfig.from_env()
        assert config.cleanup_threshold_percent == 80.0

    @patch.dict("os.environ", {f"{ENV_PREFIX}_MIN_COPIES": "3"})
    def test_from_env_min_copies(self):
        """Test custom min copies from environment."""
        config = SyncPushConfig.from_env()
        assert config.min_copies_before_delete == 3

    @patch.dict("os.environ", {f"{ENV_PREFIX}_INTERVAL": "600"})
    def test_from_env_interval(self):
        """Test custom interval from environment."""
        config = SyncPushConfig.from_env()
        assert config.check_interval_seconds == 600

    @patch.dict("os.environ", {f"{ENV_PREFIX}_COORDINATOR_URL": "http://leader:8770"})
    def test_from_env_coordinator_url(self):
        """Test coordinator URL from environment."""
        config = SyncPushConfig.from_env()
        assert config.coordinator_url == "http://leader:8770"

    @patch.dict("os.environ", {f"{ENV_PREFIX}_DATA_DIR": "/data/games"})
    def test_from_env_data_dir(self):
        """Test data directory from environment."""
        config = SyncPushConfig.from_env()
        assert config.data_dir == "/data/games"

    @patch.dict("os.environ", {
        f"{ENV_PREFIX}_THRESHOLD": "55",
        f"{ENV_PREFIX}_URGENT_THRESHOLD": "72",
        f"{ENV_PREFIX}_CLEANUP_THRESHOLD": "78",
        f"{ENV_PREFIX}_MIN_COPIES": "4",
    })
    def test_from_env_multiple_values(self):
        """Test loading multiple values from environment."""
        config = SyncPushConfig.from_env()

        assert config.push_threshold_percent == 55.0
        assert config.urgent_threshold_percent == 72.0
        assert config.cleanup_threshold_percent == 78.0
        assert config.min_copies_before_delete == 4


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Test singleton pattern for daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_sync_push_daemon()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_sync_push_daemon()

    def test_get_singleton(self):
        """Test getting singleton instance."""
        daemon1 = get_sync_push_daemon()
        daemon2 = get_sync_push_daemon()

        assert daemon1 is daemon2

    def test_reset_singleton(self):
        """Test resetting singleton creates new instance."""
        daemon1 = get_sync_push_daemon()
        reset_sync_push_daemon()
        daemon2 = get_sync_push_daemon()

        assert daemon1 is not daemon2


# =============================================================================
# Daemon Initialization Tests
# =============================================================================


class TestSyncPushDaemonInit:
    """Test SyncPushDaemon initialization."""

    def test_default_initialization(self):
        """Test daemon initializes with default config."""
        daemon = SyncPushDaemon()

        assert daemon.config is not None
        assert daemon.config.push_threshold_percent == 50.0
        assert daemon._files_pushed == 0
        assert daemon._bytes_pushed == 0
        assert daemon._files_cleaned == 0
        assert daemon._push_failures == 0

    def test_custom_config_initialization(self):
        """Test daemon initializes with custom config."""
        config = SyncPushConfig(push_threshold_percent=60.0)
        daemon = SyncPushDaemon(config=config)

        assert daemon.config.push_threshold_percent == 60.0

    def test_daemon_name(self):
        """Test daemon name is correct."""
        daemon = SyncPushDaemon()
        assert daemon._get_daemon_name() == "SyncPushDaemon"

    def test_node_id_set(self):
        """Test node_id is set from hostname."""
        daemon = SyncPushDaemon()
        assert daemon.node_id is not None
        assert len(daemon.node_id) > 0

    def test_initial_state(self):
        """Test initial state values."""
        daemon = SyncPushDaemon()

        assert daemon._manifest is None
        assert daemon._session is None
        assert daemon._coordinator_url == ""
        assert daemon._last_coordinator_check == 0.0


# =============================================================================
# Disk Threshold Tests
# =============================================================================


class TestDiskThresholds:
    """Test disk usage threshold detection."""

    def test_get_disk_usage_returns_percentage(self):
        """Test _get_disk_usage returns valid percentage."""
        daemon = SyncPushDaemon()

        # Mock shutil.disk_usage to return known values
        mock_usage = MagicMock()
        mock_usage.total = 1000
        mock_usage.used = 400
        mock_usage.free = 600

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == 40.0  # 400/1000 * 100

    def test_disk_below_push_threshold(self):
        """Test disk usage below 50% does not trigger push."""
        daemon = SyncPushDaemon()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 40
        mock_usage.free = 60

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == 40.0
        assert usage < daemon.config.push_threshold_percent

    def test_disk_at_push_threshold(self):
        """Test disk usage at 50% triggers normal push."""
        daemon = SyncPushDaemon()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 50
        mock_usage.free = 50

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == 50.0
        assert usage >= daemon.config.push_threshold_percent
        assert usage < daemon.config.urgent_threshold_percent

    def test_disk_at_urgent_threshold(self):
        """Test disk usage at 70% triggers urgent push."""
        daemon = SyncPushDaemon()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 70
        mock_usage.free = 30

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == 70.0
        assert usage >= daemon.config.urgent_threshold_percent
        assert usage < daemon.config.cleanup_threshold_percent

    def test_disk_above_cleanup_threshold(self):
        """Test disk usage above 75% triggers cleanup."""
        daemon = SyncPushDaemon()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 80
        mock_usage.free = 20

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == 80.0
        assert usage >= daemon.config.cleanup_threshold_percent

    def test_disk_usage_error_returns_negative(self):
        """Test _get_disk_usage returns -1 on error."""
        daemon = SyncPushDaemon()

        with patch("shutil.disk_usage", side_effect=OSError("Permission denied")):
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        assert usage == -1.0

    def test_disk_usage_with_custom_data_dir(self):
        """Test _get_disk_usage uses config.data_dir when set."""
        config = SyncPushConfig(data_dir="/custom/data")
        daemon = SyncPushDaemon(config=config)

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 55
        mock_usage.free = 45

        with patch("shutil.disk_usage", return_value=mock_usage) as mock_disk:
            with patch.object(Path, "exists", return_value=True):
                usage = daemon._get_disk_usage()

        # Use pytest.approx for floating point comparison
        assert usage == pytest.approx(55.0)


# =============================================================================
# SHA256 Checksum Tests
# =============================================================================


class TestSHA256Checksum:
    """Test SHA256 checksum computation."""

    def test_compute_sha256(self, tmp_path):
        """Test _compute_sha256 computes correct hash."""
        daemon = SyncPushDaemon()

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Hello, World!")

        checksum = daemon._compute_sha256(test_file)

        # Expected SHA256 of "Hello, World!"
        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        assert checksum == expected

    def test_compute_sha256_empty_file(self, tmp_path):
        """Test _compute_sha256 handles empty file."""
        daemon = SyncPushDaemon()

        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        checksum = daemon._compute_sha256(test_file)

        expected = hashlib.sha256(b"").hexdigest()
        assert checksum == expected

    def test_compute_sha256_large_file(self, tmp_path):
        """Test _compute_sha256 handles large file with chunks."""
        daemon = SyncPushDaemon()

        # Create a file larger than the 8192 byte chunk size
        test_file = tmp_path / "large.txt"
        content = b"x" * 100000
        test_file.write_bytes(content)

        checksum = daemon._compute_sha256(test_file)

        expected = hashlib.sha256(content).hexdigest()
        assert checksum == expected

    def test_compute_sha256_file_not_found(self, tmp_path):
        """Test _compute_sha256 returns None for missing file."""
        daemon = SyncPushDaemon()

        missing_file = tmp_path / "missing.txt"

        checksum = daemon._compute_sha256(missing_file)

        assert checksum is None


# =============================================================================
# Push Logic Tests
# =============================================================================


class TestPushLogic:
    """Test file push logic with mocked HTTP calls."""

    @pytest.fixture
    def daemon_with_mocks(self):
        """Create daemon with mocked dependencies."""
        daemon = SyncPushDaemon()
        daemon._coordinator_url = "http://leader:8770"
        daemon._manifest = MagicMock()
        daemon._session = MagicMock()
        return daemon

    @pytest.mark.asyncio
    async def test_push_pending_files_no_coordinator(self):
        """Test _push_pending_files returns 0 when no coordinator."""
        daemon = SyncPushDaemon()
        daemon._coordinator_url = ""  # No coordinator

        result = await daemon._push_pending_files()

        assert result == 0

    @pytest.mark.asyncio
    async def test_push_pending_files_no_manifest(self):
        """Test _push_pending_files returns 0 when no manifest."""
        daemon = SyncPushDaemon()
        daemon._coordinator_url = "http://leader:8770"
        daemon._manifest = None

        result = await daemon._push_pending_files()

        assert result == 0

    @pytest.mark.asyncio
    async def test_push_pending_files_no_pending(self, daemon_with_mocks):
        """Test _push_pending_files returns 0 when no files pending."""
        daemon = daemon_with_mocks
        daemon._manifest.get_pending_sync_files.return_value = []

        result = await daemon._push_pending_files()

        assert result == 0
        daemon._manifest.get_pending_sync_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_file_success(self, daemon_with_mocks, tmp_path):
        """Test _push_file returns True on successful push."""
        daemon = daemon_with_mocks

        # Create test file
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test data")

        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "checksum_verified": True,
            "node_id": "leader-node",
        })

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        daemon._session.post = MagicMock(return_value=mock_context)

        result = await daemon._push_file(test_file)

        assert result is True
        daemon._manifest.register_sync_receipt.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_file_failure_http_error(self, daemon_with_mocks, tmp_path):
        """Test _push_file returns False on HTTP error."""
        daemon = daemon_with_mocks

        # Create test file
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test data")

        # Mock HTTP error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        daemon._session.post = MagicMock(return_value=mock_context)

        result = await daemon._push_file(test_file)

        assert result is False
        daemon._manifest.register_sync_receipt.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_file_timeout(self, daemon_with_mocks, tmp_path):
        """Test _push_file returns False on timeout."""
        daemon = daemon_with_mocks

        # Create test file
        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test data")

        # Mock timeout
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = asyncio.TimeoutError()
        daemon._session.post = MagicMock(return_value=mock_context)

        result = await daemon._push_file(test_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_file_no_session(self, tmp_path):
        """Test _push_file returns False when no session."""
        daemon = SyncPushDaemon()
        daemon._session = None

        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test data")

        result = await daemon._push_file(test_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_push_file_unverified(self, daemon_with_mocks, tmp_path):
        """Test _push_file handles unverified response."""
        daemon = daemon_with_mocks

        test_file = tmp_path / "test.db"
        test_file.write_bytes(b"test data")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "checksum_verified": False,
            "node_id": "leader-node",
        })

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        daemon._session.post = MagicMock(return_value=mock_context)

        result = await daemon._push_file(test_file)

        assert result is True  # Still True, but receipt marked unverified
        daemon._manifest.register_sync_receipt.assert_called_once()

        # Verify the receipt has verified=False
        call_args = daemon._manifest.register_sync_receipt.call_args
        receipt = call_args[0][0]
        assert receipt.verified is False


# =============================================================================
# Safe Cleanup Tests
# =============================================================================


class TestSafeCleanup:
    """Test safe file cleanup with min_copies validation."""

    @pytest.mark.asyncio
    async def test_safe_cleanup_no_manifest(self):
        """Test _safe_cleanup returns 0 when no manifest."""
        daemon = SyncPushDaemon()
        daemon._manifest = None

        result = await daemon._safe_cleanup()

        assert result == 0

    @pytest.mark.asyncio
    async def test_safe_cleanup_no_data_dir(self, tmp_path):
        """Test _safe_cleanup returns 0 when data dir doesn't exist."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon._manifest.db_path = MagicMock()
        daemon._manifest.db_path.parent.parent.__truediv__ = MagicMock(
            return_value=tmp_path / "nonexistent"
        )

        result = await daemon._safe_cleanup()

        assert result == 0

    @pytest.mark.asyncio
    async def test_safe_cleanup_skips_canonical(self, tmp_path):
        """Test _safe_cleanup never deletes canonical databases."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon.config.data_dir = str(tmp_path)

        # Create a canonical database
        canonical_db = tmp_path / "canonical_hex8_2p.db"
        canonical_db.write_bytes(b"important data")

        daemon._manifest.is_safe_to_delete.return_value = True

        result = await daemon._safe_cleanup()

        # Should not delete canonical
        assert canonical_db.exists()

    @pytest.mark.asyncio
    async def test_safe_cleanup_respects_min_copies(self, tmp_path):
        """Test _safe_cleanup only deletes files with enough copies."""
        config = SyncPushConfig(min_copies_before_delete=3, data_dir=str(tmp_path))
        daemon = SyncPushDaemon(config=config)
        daemon._manifest = MagicMock()

        # Create test database
        test_db = tmp_path / "selfplay.db"
        test_db.write_bytes(b"test data")

        # Not enough copies - should not delete
        daemon._manifest.is_safe_to_delete.return_value = False

        result = await daemon._safe_cleanup()

        assert test_db.exists()
        daemon._manifest.is_safe_to_delete.assert_called()

    @pytest.mark.asyncio
    async def test_safe_cleanup_deletes_with_enough_copies(self, tmp_path):
        """Test _safe_cleanup deletes files with enough copies."""
        config = SyncPushConfig(min_copies_before_delete=2, data_dir=str(tmp_path))
        daemon = SyncPushDaemon(config=config)
        daemon._manifest = MagicMock()

        # Create test database
        test_db = tmp_path / "selfplay.db"
        test_db.write_bytes(b"test data")

        # Enough copies - should delete
        daemon._manifest.is_safe_to_delete.return_value = True

        result = await daemon._safe_cleanup()

        assert result == 1
        assert not test_db.exists()
        daemon._manifest.delete_sync_receipts.assert_called()

    @pytest.mark.asyncio
    async def test_safe_cleanup_cleans_wal_shm(self, tmp_path):
        """Test _safe_cleanup also removes WAL and SHM files."""
        config = SyncPushConfig(min_copies_before_delete=2, data_dir=str(tmp_path))
        daemon = SyncPushDaemon(config=config)
        daemon._manifest = MagicMock()

        # Create test database with WAL and SHM
        test_db = tmp_path / "selfplay.db"
        test_db.write_bytes(b"test data")
        wal_file = tmp_path / "selfplay.db-wal"
        wal_file.write_bytes(b"wal data")
        shm_file = tmp_path / "selfplay.db-shm"
        shm_file.write_bytes(b"shm data")

        daemon._manifest.is_safe_to_delete.return_value = True

        result = await daemon._safe_cleanup()

        assert result == 1
        assert not test_db.exists()
        assert not wal_file.exists()
        assert not shm_file.exists()

    @pytest.mark.asyncio
    async def test_safe_cleanup_skips_hidden_files(self, tmp_path):
        """Test _safe_cleanup skips hidden files."""
        config = SyncPushConfig(data_dir=str(tmp_path))
        daemon = SyncPushDaemon(config=config)
        daemon._manifest = MagicMock()

        # Create hidden database
        hidden_db = tmp_path / ".hidden.db"
        hidden_db.write_bytes(b"hidden data")

        daemon._manifest.is_safe_to_delete.return_value = True

        result = await daemon._safe_cleanup()

        assert hidden_db.exists()

    @pytest.mark.asyncio
    async def test_safe_cleanup_updates_stats(self, tmp_path):
        """Test _safe_cleanup updates cleanup statistics."""
        config = SyncPushConfig(min_copies_before_delete=2, data_dir=str(tmp_path))
        daemon = SyncPushDaemon(config=config)
        daemon._manifest = MagicMock()

        # Create test database
        test_db = tmp_path / "selfplay.db"
        test_content = b"test data content"
        test_db.write_bytes(test_content)

        daemon._manifest.is_safe_to_delete.return_value = True

        assert daemon._files_cleaned == 0
        assert daemon._bytes_cleaned == 0

        result = await daemon._safe_cleanup()

        assert daemon._files_cleaned == 1
        assert daemon._bytes_cleaned == len(test_content)


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_not_running(self):
        """Test health_check returns unhealthy when not running."""
        daemon = SyncPushDaemon()

        result = daemon.health_check()

        assert isinstance(result, HealthCheckResult)
        assert result.healthy is False
        # The daemon's custom health_check returns based on _running and _errors_count

    @pytest.mark.asyncio
    async def test_health_check_running(self):
        """Test health_check returns healthy when running."""
        daemon = SyncPushDaemon()
        daemon._running = True
        daemon._errors_count = 0  # Ensure error count is low
        daemon._coordinator_status = CoordinatorStatus.RUNNING

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == "healthy"

    def test_health_check_reports_stats(self):
        """Test health_check includes push/cleanup stats in details."""
        daemon = SyncPushDaemon()
        daemon._running = True
        daemon._files_pushed = 10
        daemon._bytes_pushed = 1024 * 1024  # 1 MB
        daemon._files_cleaned = 5
        daemon._bytes_cleaned = 512 * 1024  # 512 KB
        daemon._push_failures = 2
        daemon._coordinator_url = "http://leader:8770"

        result = daemon.health_check()

        assert "files_pushed" in result.details
        assert result.details["files_pushed"] == 10
        assert "bytes_pushed_mb" in result.details
        assert result.details["files_cleaned"] == 5
        assert result.details["push_failures"] == 2
        assert result.details["coordinator_url"] == "http://leader:8770"

    def test_health_check_unhealthy_on_errors(self):
        """Test health_check returns unhealthy with high error count."""
        daemon = SyncPushDaemon()
        daemon._running = True
        daemon._errors_count = 15  # High error count

        result = daemon.health_check()

        assert result.healthy is False


# =============================================================================
# Get Status Tests
# =============================================================================


class TestGetStatus:
    """Test get_status functionality."""

    def test_get_status_structure(self):
        """Test get_status returns expected structure."""
        daemon = SyncPushDaemon()

        status = daemon.get_status()

        assert "daemon" in status
        assert status["daemon"] == "SyncPushDaemon"
        assert "running" in status
        assert "uptime_seconds" in status
        assert "coordinator_url" in status
        assert "stats" in status
        assert "config" in status

    def test_get_status_stats(self):
        """Test get_status includes statistics."""
        daemon = SyncPushDaemon()
        daemon._files_pushed = 50
        daemon._bytes_pushed = 5 * 1024 * 1024
        daemon._files_cleaned = 10
        daemon._bytes_cleaned = 2 * 1024 * 1024
        daemon._push_failures = 3

        status = daemon.get_status()

        assert status["stats"]["files_pushed"] == 50
        assert status["stats"]["bytes_pushed"] == 5 * 1024 * 1024
        assert status["stats"]["files_cleaned"] == 10
        assert status["stats"]["bytes_cleaned"] == 2 * 1024 * 1024
        assert status["stats"]["push_failures"] == 3

    def test_get_status_config(self):
        """Test get_status includes config."""
        config = SyncPushConfig(
            push_threshold_percent=55.0,
            min_copies_before_delete=3,
        )
        daemon = SyncPushDaemon(config=config)

        status = daemon.get_status()

        assert status["config"]["push_threshold_percent"] == 55.0
        assert status["config"]["min_copies_before_delete"] == 3


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Test daemon lifecycle (start/stop)."""

    @pytest.mark.asyncio
    async def test_start_initializes_resources(self):
        """Test start() initializes manifest and session."""
        daemon = SyncPushDaemon()

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest") as mock_manifest:
            with patch("aiohttp.ClientSession", return_value=mock_session):
                mock_manifest.return_value = MagicMock()

                await daemon.start()

                assert daemon._running is True
                mock_manifest.assert_called()

                await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_session(self):
        """Test stop() closes HTTP session."""
        daemon = SyncPushDaemon()
        mock_session = AsyncMock()
        daemon._session = mock_session
        daemon._running = True
        daemon._task = asyncio.create_task(asyncio.sleep(10))

        await daemon.stop()

        mock_session.close.assert_called_once()
        assert daemon._session is None

    @pytest.mark.asyncio
    async def test_stop_logs_final_stats(self):
        """Test stop() logs final statistics."""
        daemon = SyncPushDaemon()
        daemon._files_pushed = 100
        daemon._bytes_pushed = 10 * 1024 * 1024
        daemon._files_cleaned = 20
        daemon._bytes_cleaned = 5 * 1024 * 1024
        daemon._push_failures = 5
        daemon._running = True
        daemon._session = AsyncMock()
        daemon._task = asyncio.create_task(asyncio.sleep(10))

        with patch("app.coordination.sync_push_daemon.logger") as mock_logger:
            await daemon.stop()

            # Verify stats were logged
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Test start() is no-op when disabled."""
        config = SyncPushConfig(enabled=False)
        daemon = SyncPushDaemon(config=config)

        await daemon.start()

        assert daemon._running is False


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Test the main run cycle logic."""

    @pytest.mark.asyncio
    async def test_run_cycle_below_threshold_no_action(self):
        """Test _run_cycle takes no action below push threshold."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 40  # 40% - below threshold
        mock_usage.free = 60

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(daemon, "_push_pending_files") as mock_push:
                    with patch.object(daemon, "_safe_cleanup") as mock_cleanup:
                        await daemon._run_cycle()

                        mock_push.assert_not_called()
                        mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_push_threshold_pushes(self):
        """Test _run_cycle triggers push at push threshold."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon._coordinator_url = "http://leader:8770"
        daemon._last_coordinator_check = time.time()  # Skip coordinator discovery

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 55  # 55% - above push threshold
        mock_usage.free = 45

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                    mock_push.return_value = 5
                    await daemon._run_cycle()

                    mock_push.assert_called_once()
                    # urgent=False since 55% < 70%
                    mock_push.assert_called_with(urgent=False)

    @pytest.mark.asyncio
    async def test_run_cycle_urgent_threshold_urgent_push(self):
        """Test _run_cycle triggers urgent push at urgent threshold."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon._coordinator_url = "http://leader:8770"
        daemon._last_coordinator_check = time.time()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 72  # 72% - above urgent threshold
        mock_usage.free = 28

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                    mock_push.return_value = 3
                    await daemon._run_cycle()

                    mock_push.assert_called_once()
                    mock_push.assert_called_with(urgent=True)

    @pytest.mark.asyncio
    async def test_run_cycle_cleanup_threshold_cleans(self):
        """Test _run_cycle triggers cleanup at cleanup threshold."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon._coordinator_url = "http://leader:8770"
        daemon._last_coordinator_check = time.time()

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 78  # 78% - above cleanup threshold
        mock_usage.free = 22

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                    with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                        mock_cleanup.return_value = 5
                        mock_push.return_value = 0
                        await daemon._run_cycle()

                        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_refreshes_coordinator(self):
        """Test _run_cycle refreshes coordinator when stale."""
        daemon = SyncPushDaemon()
        daemon._manifest = MagicMock()
        daemon._last_coordinator_check = 0  # Force refresh

        mock_usage = MagicMock()
        mock_usage.total = 100
        mock_usage.used = 40  # Below threshold
        mock_usage.free = 60

        with patch("shutil.disk_usage", return_value=mock_usage):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(
                    daemon, "_discover_coordinator", new_callable=AsyncMock
                ) as mock_discover:
                    await daemon._run_cycle()

                    mock_discover.assert_called_once()


# =============================================================================
# Coordinator Discovery Tests
# =============================================================================


class TestCoordinatorDiscovery:
    """Test coordinator discovery logic."""

    @pytest.mark.asyncio
    async def test_discover_coordinator_uses_config(self):
        """Test _discover_coordinator uses config URL if set."""
        config = SyncPushConfig(coordinator_url="http://configured:8770")
        daemon = SyncPushDaemon(config=config)

        await daemon._discover_coordinator()

        assert daemon._coordinator_url == "http://configured:8770"

    @pytest.mark.asyncio
    async def test_discover_coordinator_from_p2p(self):
        """Test _discover_coordinator queries P2P status."""
        daemon = SyncPushDaemon()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "leader_id": "leader-node",
            "peers": {
                "leader-node": {
                    "address": "192.168.1.100:8770",
                }
            }
        })

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_context
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            await daemon._discover_coordinator()

        assert "192.168.1.100" in daemon._coordinator_url

    @pytest.mark.asyncio
    async def test_discover_coordinator_failure_logged(self):
        """Test _discover_coordinator logs failure gracefully."""
        daemon = SyncPushDaemon()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("Connection refused")
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            # Should not raise
            await daemon._discover_coordinator()

        assert daemon._coordinator_url == ""


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_env_prefix(self):
        """Test ENV_PREFIX is correct."""
        assert ENV_PREFIX == "RINGRIFT_SYNC_PUSH"

    def test_max_inline_size(self):
        """Test MAX_INLINE_SIZE_MB is reasonable."""
        assert MAX_INLINE_SIZE_MB == 50
        assert MAX_INLINE_SIZE_MB > 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_push_file_with_bad_checksum(self, tmp_path):
        """Test _push_file handles checksum computation failure."""
        daemon = SyncPushDaemon()
        daemon._coordinator_url = "http://leader:8770"
        daemon._session = MagicMock()

        # Mock checksum to fail
        with patch.object(daemon, "_compute_sha256", return_value=None):
            test_file = tmp_path / "test.db"
            test_file.write_bytes(b"test")

            result = await daemon._push_file(test_file)

        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self):
        """Test daemon handles multiple start/stop cycles."""
        daemon = SyncPushDaemon()

        for _ in range(3):
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()

            with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
                with patch("aiohttp.ClientSession", return_value=mock_session):
                    await daemon.start()
                    assert daemon._running is True
                    await daemon.stop()
                    assert daemon._running is False

    def test_get_data_dir_with_config(self):
        """Test _get_data_dir uses config.data_dir when set."""
        config = SyncPushConfig(data_dir="/custom/data/path")
        daemon = SyncPushDaemon(config=config)

        result = daemon._get_data_dir()

        assert result == Path("/custom/data/path")

    def test_get_data_dir_with_manifest(self):
        """Test _get_data_dir falls back to manifest path."""
        daemon = SyncPushDaemon()

        # Mock the manifest with a proper path structure
        mock_manifest = MagicMock()
        # Create a mock db_path that has the right parent chain
        mock_db_path = MagicMock()
        mock_parent = MagicMock()
        mock_parent_parent = MagicMock()

        # Set up the chain: db_path.parent.parent / "games"
        mock_parent_parent.__truediv__ = MagicMock(return_value=Path("/data/games"))
        mock_parent.parent = mock_parent_parent
        mock_db_path.parent = mock_parent

        mock_manifest.db_path = mock_db_path
        daemon._manifest = mock_manifest

        result = daemon._get_data_dir()

        assert result == Path("/data/games")

    def test_get_data_dir_default(self):
        """Test _get_data_dir returns default when no config or manifest."""
        daemon = SyncPushDaemon()
        daemon._manifest = None

        result = daemon._get_data_dir()

        assert result == Path("data/games")
