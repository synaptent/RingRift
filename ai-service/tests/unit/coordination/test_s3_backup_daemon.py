"""Tests for S3BackupDaemon.

Tests cover:
- Configuration initialization
- S3BackupConfig dataclass
- BackupResult dataclass
- Daemon lifecycle (start/stop)
- Health check reporting
- Metrics tracking
- MODEL_PROMOTED event handling
- Debounce behavior

December 2025: Created as part of S3 infrastructure completion (Phase 6).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.s3_backup_daemon import (
    BackupResult,
    S3BackupConfig,
    S3BackupDaemon,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return S3BackupConfig(
        s3_bucket="test-bucket",
        aws_region="us-west-2",
        backup_timeout_seconds=60.0,
        debounce_seconds=5.0,  # Short for testing
        max_pending_before_immediate=3,
    )


@pytest.fixture
def daemon(config):
    """Create a test daemon."""
    return S3BackupDaemon(config)


# =============================================================================
# S3BackupConfig Tests
# =============================================================================


class TestS3BackupConfig:
    """Test S3BackupConfig dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        config = S3BackupConfig()
        assert config.s3_bucket == "ringrift-models-20251214"
        assert config.aws_region == "us-east-1"
        assert config.backup_timeout_seconds == 600.0
        assert config.backup_models is True
        assert config.backup_databases is True
        assert config.backup_state is True
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 30.0
        assert config.emit_completion_event is True
        assert config.debounce_seconds == 60.0
        assert config.max_pending_before_immediate == 5

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = S3BackupConfig(
            s3_bucket="custom-bucket",
            aws_region="eu-west-1",
            backup_timeout_seconds=300.0,
            retry_count=5,
            debounce_seconds=30.0,
        )
        assert config.s3_bucket == "custom-bucket"
        assert config.aws_region == "eu-west-1"
        assert config.backup_timeout_seconds == 300.0
        assert config.retry_count == 5
        assert config.debounce_seconds == 30.0

    def test_backup_feature_flags(self):
        """Test backup feature flags."""
        config = S3BackupConfig(
            backup_models=False,
            backup_databases=True,
            backup_state=False,
        )
        assert config.backup_models is False
        assert config.backup_databases is True
        assert config.backup_state is False


# =============================================================================
# BackupResult Tests
# =============================================================================


class TestBackupResult:
    """Test BackupResult dataclass."""

    def test_successful_result(self):
        """Test successful backup result."""
        result = BackupResult(
            success=True,
            uploaded_count=10,
            deleted_count=0,
            duration_seconds=15.5,
        )
        assert result.success is True
        assert result.uploaded_count == 10
        assert result.deleted_count == 0
        assert result.duration_seconds == 15.5
        assert result.error_message == ""

    def test_failed_result(self):
        """Test failed backup result."""
        result = BackupResult(
            success=False,
            uploaded_count=0,
            deleted_count=0,
            error_message="Connection timeout",
            duration_seconds=60.0,
        )
        assert result.success is False
        assert result.uploaded_count == 0
        assert result.error_message == "Connection timeout"

    def test_default_values(self):
        """Test default values."""
        result = BackupResult(success=True, uploaded_count=5, deleted_count=0)
        assert result.error_message == ""
        assert result.duration_seconds == 0.0


# =============================================================================
# S3BackupDaemon Initialization Tests
# =============================================================================


class TestS3BackupDaemonInit:
    """Test S3BackupDaemon initialization."""

    def test_default_config(self):
        """Test daemon with default config."""
        daemon = S3BackupDaemon()
        assert daemon.config is not None
        assert daemon.config.s3_bucket == "ringrift-models-20251214"

    def test_custom_config(self, config):
        """Test daemon with custom config."""
        daemon = S3BackupDaemon(config)
        assert daemon.config.s3_bucket == "test-bucket"
        assert daemon.config.aws_region == "us-west-2"

    def test_initial_state(self, daemon):
        """Test initial daemon state."""
        assert daemon._running is False
        assert daemon._last_backup_time == 0.0
        assert daemon._pending_promotions == []
        assert daemon._events_processed == 0
        assert daemon._successful_backups == 0
        assert daemon._failed_backups == 0

    def test_name_property(self, daemon):
        """Test daemon name."""
        assert daemon.name == "S3BackupDaemon"

    def test_is_running_property(self, daemon):
        """Test is_running property."""
        assert daemon.is_running() is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestS3BackupDaemonHealthCheck:
    """Test S3BackupDaemon health check."""

    def test_health_check_when_stopped(self, daemon):
        """Test health check returns healthy when stopped."""
        result = daemon.health_check()
        assert result.healthy is True
        assert "not running" in result.message.lower()

    def test_health_check_when_running_healthy(self, daemon):
        """Test health check when running and healthy."""
        daemon._running = True
        daemon._start_time = time.time()
        result = daemon.health_check()
        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_degraded_many_pending(self, daemon):
        """Test health check degraded with many pending promotions."""
        daemon._running = True
        daemon._start_time = time.time()
        # Add > 10 pending promotions
        daemon._pending_promotions = [{"timestamp": time.time()} for _ in range(15)]
        result = daemon.health_check()
        assert result.healthy is False
        assert "15" in result.message  # Should mention count

    def test_health_check_degraded_stalled(self, daemon):
        """Test health check degraded when stalled."""
        daemon._running = True
        daemon._start_time = time.time()
        daemon._pending_promotions = [{"timestamp": time.time()}]
        daemon._last_backup_time = time.time() - 2000  # >30 minutes ago
        result = daemon.health_check()
        assert result.healthy is False
        assert "stalled" in result.message.lower()


# =============================================================================
# Metrics Tests
# =============================================================================


class TestS3BackupDaemonMetrics:
    """Test S3BackupDaemon metrics."""

    def test_get_metrics_initial(self, daemon):
        """Test initial metrics."""
        metrics = daemon.get_metrics()
        assert metrics["name"] == "S3BackupDaemon"
        assert metrics["running"] is False
        assert metrics["events_processed"] == 0
        assert metrics["pending_promotions"] == 0
        assert metrics["successful_backups"] == 0
        assert metrics["failed_backups"] == 0

    def test_get_metrics_with_activity(self, daemon):
        """Test metrics with activity."""
        daemon._running = True
        daemon._start_time = time.time() - 100
        daemon._events_processed = 5
        daemon._successful_backups = 3
        daemon._failed_backups = 1
        daemon._pending_promotions = [{"test": 1}]

        metrics = daemon.get_metrics()
        assert metrics["running"] is True
        assert metrics["uptime_seconds"] >= 99
        assert metrics["events_processed"] == 5
        assert metrics["pending_promotions"] == 1
        assert metrics["successful_backups"] == 3
        assert metrics["failed_backups"] == 1

    def test_get_metrics_includes_config(self, daemon):
        """Test metrics includes config values."""
        metrics = daemon.get_metrics()
        assert "s3_bucket" in metrics
        assert metrics["s3_bucket"] == daemon.config.s3_bucket


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestS3BackupDaemonEventHandling:
    """Test S3BackupDaemon event handling."""

    def test_on_model_promoted_dict_event(self, daemon):
        """Test handling MODEL_PROMOTED event as dict."""
        daemon._pending_event = asyncio.Event()
        event = {
            "model_path": "/models/test.pth",
            "model_id": "test-123",
            "board_type": "hex8",
            "num_players": 2,
            "elo": 1500,
        }
        daemon._on_model_promoted(event)

        assert len(daemon._pending_promotions) == 1
        assert daemon._events_processed == 1
        assert daemon._pending_promotions[0]["model_path"] == "/models/test.pth"
        assert daemon._pending_promotions[0]["board_type"] == "hex8"

    def test_on_model_promoted_event_object(self, daemon):
        """Test handling MODEL_PROMOTED event as RouterEvent object."""
        daemon._pending_event = asyncio.Event()

        # Create mock RouterEvent with payload attribute
        mock_event = MagicMock()
        mock_event.payload = {
            "model_path": "/models/promoted.pth",
            "model_id": "promo-456",
            "board_type": "square8",
            "num_players": 4,
        }
        daemon._on_model_promoted(mock_event)

        assert len(daemon._pending_promotions) == 1
        assert daemon._pending_promotions[0]["model_path"] == "/models/promoted.pth"
        assert daemon._pending_promotions[0]["board_type"] == "square8"

    def test_on_model_promoted_thread_safety(self, daemon):
        """Test that event handling is thread-safe."""
        daemon._pending_event = asyncio.Event()
        events = [{"model_path": f"/models/model_{i}.pth"} for i in range(10)]

        import threading
        threads = []
        for event in events:
            t = threading.Thread(target=daemon._on_model_promoted, args=(event,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(daemon._pending_promotions) == 10
        assert daemon._events_processed == 10


# =============================================================================
# Debounce Logic Tests
# =============================================================================


class TestS3BackupDaemonDebounce:
    """Test S3BackupDaemon debounce behavior."""

    @pytest.mark.asyncio
    async def test_check_pending_empty(self, daemon):
        """Test check_pending_backups with no pending items."""
        # Should not raise and should not trigger backup
        await daemon._check_pending_backups()

    @pytest.mark.asyncio
    async def test_check_pending_before_debounce(self, daemon):
        """Test check_pending_backups before debounce period."""
        daemon._pending_promotions = [{"timestamp": time.time()}]

        with patch.object(daemon, "_process_pending_backups") as mock_process:
            await daemon._check_pending_backups()
            # Should not process yet (within debounce period)
            mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_pending_after_debounce(self, daemon):
        """Test check_pending_backups after debounce period."""
        # Set timestamp older than debounce period
        daemon._pending_promotions = [{"timestamp": time.time() - 10}]  # 10s ago, debounce is 5s

        with patch.object(daemon, "_process_pending_backups", new_callable=AsyncMock) as mock_process:
            await daemon._check_pending_backups()
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_pending_max_triggers_immediate(self, daemon):
        """Test that max_pending triggers immediate backup."""
        # Add more than max_pending_before_immediate (3 in test config)
        daemon._pending_promotions = [
            {"timestamp": time.time()} for _ in range(5)
        ]

        with patch.object(daemon, "_process_pending_backups", new_callable=AsyncMock) as mock_process:
            await daemon._check_pending_backups()
            # Should process immediately due to too many pending
            mock_process.assert_called_once()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestS3BackupDaemonLifecycle:
    """Test S3BackupDaemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """Test that start sets running flag."""
        async def stop_after_start():
            await asyncio.sleep(0.1)
            daemon._running = False

        # Start with early termination
        with patch.object(daemon, "_check_pending_backups", new_callable=AsyncMock):
            task = asyncio.create_task(daemon.start())
            await stop_after_start()
            await asyncio.wait_for(task, timeout=1.0)

        # Should have started
        assert daemon._start_time > 0

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, daemon):
        """Test that stop clears running flag."""
        daemon._running = True

        with patch.object(daemon, "_run_backup", new_callable=AsyncMock):
            await daemon.stop()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_processes_pending(self, daemon):
        """Test that stop processes pending backups."""
        daemon._running = True
        daemon._pending_promotions = [{"model_path": "/test.pth"}]

        with patch.object(daemon, "_run_backup", new_callable=AsyncMock) as mock_backup:
            await daemon.stop()
            mock_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Test that start is idempotent."""
        daemon._running = True
        original_time = daemon._start_time = time.time() - 100

        await daemon.start()  # Should return immediately

        # Start time should not change
        assert daemon._start_time == original_time


# =============================================================================
# Adapter Tests
# =============================================================================


class TestS3BackupDaemonAdapter:
    """Test S3BackupDaemonAdapter wrapper."""

    def test_adapter_import(self):
        """Test adapter can be imported."""
        from app.coordination.s3_backup_daemon import S3BackupDaemonAdapter
        adapter = S3BackupDaemonAdapter()
        assert adapter is not None

    def test_adapter_health_check_not_started(self):
        """Test adapter health check when not started."""
        from app.coordination.s3_backup_daemon import S3BackupDaemonAdapter
        adapter = S3BackupDaemonAdapter()
        result = adapter.health_check()
        assert result.healthy is True
        assert "not started" in result.message.lower()

    def test_adapter_with_config(self, config):
        """Test adapter with custom config."""
        from app.coordination.s3_backup_daemon import S3BackupDaemonAdapter
        adapter = S3BackupDaemonAdapter(config=config)
        assert adapter.config.s3_bucket == "test-bucket"
