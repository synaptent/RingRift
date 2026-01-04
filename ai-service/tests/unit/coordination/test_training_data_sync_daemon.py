"""Tests for TrainingDataSyncDaemon - Pre-training data synchronization.

Tests cover:
- SyncResult dataclass and serialization
- TrainingDataSyncConfig with environment variables
- sync_from_owc() rsync-based OWC transfer
- sync_from_s3() AWS S3 transfer
- sync_training_data_for_config() main entry point
- TrainingDataSyncDaemon lifecycle (start/stop)
- health_check() for DaemonManager integration
- Singleton pattern (get/reset)
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.training_data_sync_daemon import (
    LOCAL_TRAINING_DIR,
    MIN_SIZE_IMPROVEMENT_RATIO,
    SYNC_TIMEOUT,
    SyncResult,
    TrainingDataSyncConfig,
    TrainingDataSyncDaemon,
    get_training_data_sync_daemon,
    reset_training_data_sync_daemon,
    sync_from_owc,
    sync_from_s3,
    sync_training_data_for_config,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_data_source():
    """Create a mock DataSource enum."""
    from app.coordination.training_data_manifest import DataSource
    return DataSource


@pytest.fixture
def mock_training_entry(mock_data_source):
    """Create a mock TrainingDataEntry."""
    from app.coordination.training_data_manifest import TrainingDataEntry
    return TrainingDataEntry(
        config_key="hex8_2p",
        path="/Volumes/RingRift-Data/training/hex8_2p.npz",
        source=mock_data_source.OWC,
        size_bytes=100 * 1024 * 1024,  # 100MB
        sample_count=50000,
        modified_time=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def sync_config():
    """Create a test sync config."""
    return TrainingDataSyncConfig(
        check_interval_seconds=5.0,
        min_size_improvement_ratio=1.1,
        timeout_seconds=10,
        emit_events=False,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset daemon singleton before each test."""
    reset_training_data_sync_daemon()
    yield
    reset_training_data_sync_daemon()


# =============================================================================
# Test SyncResult Dataclass
# =============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_sync_result_creation(self, mock_data_source):
        """Test creating a SyncResult."""
        result = SyncResult(
            config_key="hex8_2p",
            success=True,
            source=mock_data_source.OWC,
            source_path="/path/to/file.npz",
            local_path="/local/path.npz",
            bytes_transferred=1024,
            duration_seconds=2.5,
        )
        assert result.config_key == "hex8_2p"
        assert result.success is True
        assert result.source == mock_data_source.OWC
        assert result.bytes_transferred == 1024
        assert result.error is None

    def test_sync_result_failed(self, mock_data_source):
        """Test creating a failed SyncResult."""
        result = SyncResult(
            config_key="hex8_2p",
            success=False,
            source=mock_data_source.S3,
            error="Connection timeout",
            duration_seconds=30.0,
        )
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.bytes_transferred == 0

    def test_sync_result_skipped(self, mock_data_source):
        """Test creating a skipped SyncResult."""
        result = SyncResult(
            config_key="hex8_2p",
            success=True,
            source=mock_data_source.LOCAL,
            local_path="/local/path.npz",
            skipped_reason="Local file is sufficient",
        )
        assert result.success is True
        assert result.skipped_reason == "Local file is sufficient"
        assert result.bytes_transferred == 0

    def test_to_dict(self, mock_data_source):
        """Test serialization to dictionary."""
        result = SyncResult(
            config_key="hex8_2p",
            success=True,
            source=mock_data_source.OWC,
            source_path="/remote/path.npz",
            local_path="/local/path.npz",
            bytes_transferred=2048,
            duration_seconds=1.5,
        )
        d = result.to_dict()
        assert d["config_key"] == "hex8_2p"
        assert d["success"] is True
        assert d["source"] == "owc"
        assert d["bytes_transferred"] == 2048
        assert d["error"] is None

    def test_to_dict_no_source(self):
        """Test to_dict when source is None."""
        result = SyncResult(
            config_key="hex8_2p",
            success=False,
            error="No data found",
        )
        d = result.to_dict()
        assert d["source"] is None
        assert d["error"] == "No data found"


# =============================================================================
# Test TrainingDataSyncConfig
# =============================================================================


class TestTrainingDataSyncConfig:
    """Tests for TrainingDataSyncConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingDataSyncConfig()
        assert config.check_interval_seconds == 60.0
        assert config.min_size_improvement_ratio == 1.1
        assert config.timeout_seconds == 1800
        assert config.emit_events is True

    def test_source_priority_default(self, mock_data_source):
        """Test default source priority."""
        config = TrainingDataSyncConfig()
        assert len(config.source_priority) == 3
        assert config.source_priority[0] == mock_data_source.OWC
        assert config.source_priority[1] == mock_data_source.S3
        assert config.source_priority[2] == mock_data_source.LOCAL

    def test_from_env_defaults(self):
        """Test from_env with default values."""
        with patch.dict(os.environ, {}, clear=False):
            # Clear any existing env vars
            env_vars = [
                "RINGRIFT_DATA_SYNC_INTERVAL",
                "RINGRIFT_DATA_SYNC_SIZE_RATIO",
                "RINGRIFT_DATA_SYNC_TIMEOUT",
            ]
            for var in env_vars:
                os.environ.pop(var, None)

            config = TrainingDataSyncConfig.from_env()
            assert config.check_interval_seconds == 60.0
            assert config.min_size_improvement_ratio == 1.1
            assert config.timeout_seconds == 1800

    def test_from_env_custom_values(self):
        """Test from_env with custom environment values."""
        with patch.dict(
            os.environ,
            {
                "RINGRIFT_DATA_SYNC_INTERVAL": "30",
                "RINGRIFT_DATA_SYNC_SIZE_RATIO": "1.5",
                "RINGRIFT_DATA_SYNC_TIMEOUT": "600",
            },
        ):
            config = TrainingDataSyncConfig.from_env()
            assert config.check_interval_seconds == 30.0
            assert config.min_size_improvement_ratio == 1.5
            assert config.timeout_seconds == 600


# =============================================================================
# Test Module Constants
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_local_training_dir_default(self):
        """Test LOCAL_TRAINING_DIR default value."""
        # Should be Path object
        assert isinstance(LOCAL_TRAINING_DIR, Path)

    def test_min_size_improvement_ratio_default(self):
        """Test MIN_SIZE_IMPROVEMENT_RATIO default."""
        assert MIN_SIZE_IMPROVEMENT_RATIO >= 1.0
        assert MIN_SIZE_IMPROVEMENT_RATIO <= 2.0

    def test_sync_timeout_default(self):
        """Test SYNC_TIMEOUT default."""
        assert SYNC_TIMEOUT > 0
        assert SYNC_TIMEOUT <= 7200  # Max 2 hours


# =============================================================================
# Test sync_from_owc
# =============================================================================


class TestSyncFromOWC:
    """Tests for sync_from_owc function."""

    @pytest.mark.asyncio
    async def test_sync_from_owc_success(self, mock_training_entry, tmp_path):
        """Test successful OWC sync."""
        local_path = tmp_path / "hex8_2p.npz"

        # Mock subprocess for rsync
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        # Create a fake file to simulate successful download
        local_path.write_bytes(b"fake npz data")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_owc(mock_training_entry, local_path, timeout=10)

        assert result.success is True
        assert result.source.value == "owc"
        assert result.bytes_transferred > 0

    @pytest.mark.asyncio
    async def test_sync_from_owc_rsync_failure(self, mock_training_entry, tmp_path):
        """Test OWC sync with rsync failure."""
        local_path = tmp_path / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"rsync: connection refused")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_owc(mock_training_entry, local_path, timeout=10)

        assert result.success is False
        assert "rsync failed" in result.error
        assert "connection refused" in result.error

    @pytest.mark.asyncio
    async def test_sync_from_owc_timeout(self, mock_training_entry, tmp_path):
        """Test OWC sync timeout."""
        local_path = tmp_path / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.communicate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_owc(mock_training_entry, local_path, timeout=1)

        assert result.success is False
        assert "timed out" in result.error
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_from_owc_creates_directory(self, mock_training_entry, tmp_path):
        """Test that sync_from_owc creates parent directories."""
        nested_path = tmp_path / "nested" / "dir" / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await sync_from_owc(mock_training_entry, nested_path, timeout=10)

        assert nested_path.parent.exists()

    @pytest.mark.asyncio
    async def test_sync_from_owc_os_error(self, mock_training_entry, tmp_path):
        """Test OWC sync with OSError."""
        local_path = tmp_path / "hex8_2p.npz"

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("Permission denied"),
        ):
            result = await sync_from_owc(mock_training_entry, local_path, timeout=10)

        assert result.success is False
        assert "Permission denied" in result.error


# =============================================================================
# Test sync_from_s3
# =============================================================================


class TestSyncFromS3:
    """Tests for sync_from_s3 function."""

    @pytest.fixture
    def s3_training_entry(self, mock_data_source):
        """Create a mock S3 TrainingDataEntry."""
        from app.coordination.training_data_manifest import TrainingDataEntry
        return TrainingDataEntry(
            config_key="hex8_2p",
            path="s3://ringrift-models/training/hex8_2p.npz",
            source=mock_data_source.S3,
            size_bytes=50 * 1024 * 1024,  # 50MB
            sample_count=25000,
            modified_time=datetime.now(tz=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_sync_from_s3_success(self, s3_training_entry, tmp_path):
        """Test successful S3 sync."""
        local_path = tmp_path / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        # Create fake file
        local_path.write_bytes(b"fake npz data")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_s3(s3_training_entry, local_path, timeout=10)

        assert result.success is True
        assert result.source.value == "s3"
        assert result.bytes_transferred > 0

    @pytest.mark.asyncio
    async def test_sync_from_s3_failure(self, s3_training_entry, tmp_path):
        """Test S3 sync failure."""
        local_path = tmp_path / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"fatal error: Unable to locate credentials")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_s3(s3_training_entry, local_path, timeout=10)

        assert result.success is False
        assert "aws s3 cp failed" in result.error

    @pytest.mark.asyncio
    async def test_sync_from_s3_timeout(self, s3_training_entry, tmp_path):
        """Test S3 sync timeout."""
        local_path = tmp_path / "hex8_2p.npz"

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.communicate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await sync_from_s3(s3_training_entry, local_path, timeout=1)

        assert result.success is False
        assert "timed out" in result.error


# =============================================================================
# Test sync_training_data_for_config
# =============================================================================


class TestSyncTrainingDataForConfig:
    """Tests for sync_training_data_for_config function."""

    @pytest.mark.asyncio
    async def test_no_data_in_manifest(self, sync_config, tmp_path):
        """Test when no data exists in manifest."""
        mock_manifest = MagicMock()
        mock_manifest.get_best_data.return_value = None

        with patch(
            "app.coordination.training_data_sync_daemon.get_training_data_manifest",
            new=AsyncMock(return_value=mock_manifest),
        ):
            result = await sync_training_data_for_config(
                "nonexistent_config",
                local_dir=tmp_path,
                config=sync_config,
            )

        assert result.success is False
        assert "No training data found" in result.error

    @pytest.mark.asyncio
    async def test_local_data_is_best(self, mock_data_source, sync_config, tmp_path):
        """Test when local data is already the best."""
        from app.coordination.training_data_manifest import TrainingDataEntry

        mock_entry = TrainingDataEntry(
            config_key="hex8_2p",
            path=str(tmp_path / "hex8_2p.npz"),
            source=mock_data_source.LOCAL,
            size_bytes=100 * 1024 * 1024,
            sample_count=50000,
            modified_time=datetime.now(tz=timezone.utc),
        )

        mock_manifest = MagicMock()
        mock_manifest.get_best_data.return_value = mock_entry

        with patch(
            "app.coordination.training_data_sync_daemon.get_training_data_manifest",
            new=AsyncMock(return_value=mock_manifest),
        ):
            result = await sync_training_data_for_config(
                "hex8_2p",
                local_dir=tmp_path,
                config=sync_config,
            )

        assert result.success is True
        assert result.source == mock_data_source.LOCAL
        assert "Best data is already local" in result.skipped_reason

    @pytest.mark.asyncio
    async def test_local_file_sufficient(
        self, mock_data_source, mock_training_entry, sync_config, tmp_path
    ):
        """Test when local file is sufficient (within ratio)."""
        local_path = tmp_path / "hex8_2p.npz"
        # Create local file that's 95% of remote size (within 1.1x ratio)
        local_size = int(mock_training_entry.size_bytes * 0.95)
        local_path.write_bytes(b"x" * local_size)

        mock_manifest = MagicMock()
        mock_manifest.get_best_data.return_value = mock_training_entry

        with patch(
            "app.coordination.training_data_sync_daemon.get_training_data_manifest",
            new=AsyncMock(return_value=mock_manifest),
        ):
            result = await sync_training_data_for_config(
                "hex8_2p",
                local_dir=tmp_path,
                config=sync_config,
            )

        assert result.success is True
        assert result.bytes_transferred == 0
        assert "within" in result.skipped_reason

    @pytest.mark.asyncio
    async def test_force_redownload(
        self, mock_data_source, mock_training_entry, sync_config, tmp_path
    ):
        """Test force re-download ignores local file size."""
        local_path = tmp_path / "hex8_2p.npz"
        # Create large local file
        local_path.write_bytes(b"x" * (mock_training_entry.size_bytes * 2))

        mock_manifest = MagicMock()
        mock_manifest.get_best_data.return_value = mock_training_entry

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch(
            "app.coordination.training_data_sync_daemon.get_training_data_manifest",
            new=AsyncMock(return_value=mock_manifest),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await sync_training_data_for_config(
                    "hex8_2p",
                    local_dir=tmp_path,
                    config=sync_config,
                    force=True,
                )

        # Should attempt sync even though local file is larger
        assert result.source == mock_data_source.OWC


# =============================================================================
# Test TrainingDataSyncDaemon
# =============================================================================


class TestTrainingDataSyncDaemon:
    """Tests for TrainingDataSyncDaemon class."""

    def test_daemon_creation(self):
        """Test creating a daemon instance."""
        daemon = TrainingDataSyncDaemon()
        assert daemon._running is False
        # Jan 2026: HandlerBase uses individual attributes instead of _task and _stats dict
        assert daemon._syncs_completed == 0
        assert daemon._syncs_failed == 0
        assert daemon._bytes_transferred == 0

    def test_daemon_with_custom_config(self, sync_config):
        """Test creating daemon with custom config."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        assert daemon.config.check_interval_seconds == 5.0
        assert daemon.config.timeout_seconds == 10

    @pytest.mark.asyncio
    async def test_daemon_start(self, sync_config):
        """Test starting the daemon."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        # Mock _get_pending_training_configs to return empty
        daemon._get_pending_training_configs = AsyncMock(return_value=[])

        await daemon.start()

        assert daemon._running is True
        # Jan 2026: HandlerBase tracks started_at in _stats.started_at (float, not dict key)
        assert daemon._stats.started_at > 0

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_daemon_start_already_running(self, sync_config):
        """Test starting already running daemon logs warning."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        daemon._get_pending_training_configs = AsyncMock(return_value=[])

        await daemon.start()
        first_start_time = daemon._stats.started_at

        # Starting again should not reset started_at
        await daemon.start()
        assert daemon._stats.started_at == first_start_time

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_daemon_stop(self, sync_config):
        """Test stopping the daemon."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        daemon._get_pending_training_configs = AsyncMock(return_value=[])

        await daemon.start()
        assert daemon._running is True

        await daemon.stop()
        # Jan 2026: HandlerBase sets _running = False on stop
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_daemon_stop_not_running(self, sync_config):
        """Test stopping daemon that's not running."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        # Should not raise
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_run_loop_syncs_pending_configs(self, sync_config):
        """Test that run loop syncs pending configs."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        pending_configs = ["hex8_2p", "square8_4p"]
        daemon._get_pending_training_configs = AsyncMock(return_value=pending_configs)

        sync_results = []

        async def mock_sync(config_key, **kwargs):
            from app.coordination.training_data_manifest import DataSource
            result = SyncResult(
                config_key=config_key,
                success=True,
                source=DataSource.OWC,
                bytes_transferred=1024,
            )
            sync_results.append(result)
            return result

        with patch(
            "app.coordination.training_data_sync_daemon.sync_training_data_for_config",
            side_effect=mock_sync,
        ):
            # Initialize daemon state (normally done by start())
            daemon._running = True
            daemon._syncs_completed = 0
            daemon._syncs_failed = 0
            daemon._bytes_transferred = 0
            try:
                # Jan 2026: _run_loop renamed to _run_cycle after HandlerBase migration
                await asyncio.wait_for(daemon._run_cycle(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

        assert len(sync_results) == 2
        # Jan 2026: _stats dict replaced with individual attributes after HandlerBase migration
        assert daemon._syncs_completed == 2


class TestDaemonHealthCheck:
    """Tests for daemon health_check method."""

    def test_health_check_not_running(self, sync_config):
        """Test health check when daemon is not running."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        health = daemon.health_check()

        assert health.healthy is False
        assert health.status.value == "stopped"
        assert "not running" in health.message

    @pytest.mark.asyncio
    async def test_health_check_running_healthy(self, sync_config):
        """Test health check when daemon is healthy."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        daemon._get_pending_training_configs = AsyncMock(return_value=[])

        await daemon.start()
        # Wait for stats to be initialized
        await asyncio.sleep(0.1)

        health = daemon.health_check()

        assert health.healthy is True
        assert health.status.value == "running"
        assert health.details["running"] is True

        await daemon.stop()

    def test_health_check_degraded_high_error_rate(self, sync_config):
        """Test health check with high error rate."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        daemon._running = True
        # Jan 2026: Use individual attributes instead of _stats dict
        daemon._syncs_completed = 2
        daemon._syncs_failed = 8  # 80% error rate
        daemon._bytes_transferred = 1024

        health = daemon.health_check()

        assert health.healthy is True  # Still running but degraded
        assert health.status.value == "degraded"
        assert "High sync" in health.message
        assert "Error rate too high" in health.message
        assert health.details["error_rate"] == 0.8


class TestDaemonPendingConfigs:
    """Tests for _get_pending_training_configs method."""

    @pytest.mark.asyncio
    async def test_detects_local_training_process(self, sync_config):
        """Test detection of local training processes."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        # Mock subprocess.run to return training process
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "12345 python -m app.training.train --board-type hex8 --num-players 2\n"
        )

        with patch("subprocess.run", return_value=mock_result):
            configs = await daemon._get_pending_training_configs()

        assert "hex8_2p" in configs

    @pytest.mark.asyncio
    async def test_no_training_processes(self, sync_config):
        """Test when no training processes are running."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        mock_result = MagicMock()
        mock_result.returncode = 1  # No match
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with patch(
                "app.coordination.work_queue.get_work_queue",
                side_effect=ImportError(),
            ):
                configs = await daemon._get_pending_training_configs()

        assert len(configs) == 0

    @pytest.mark.asyncio
    async def test_handles_pgrep_not_found(self, sync_config):
        """Test graceful handling when pgrep is not available."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with patch(
                "app.coordination.work_queue.get_work_queue",
                side_effect=ImportError(),
            ):
                configs = await daemon._get_pending_training_configs()

        # Should return empty list, not raise
        assert configs == []


class TestDaemonEventEmission:
    """Tests for daemon event emission."""

    @pytest.mark.asyncio
    async def test_emit_sync_event_success(self, mock_data_source, sync_config):
        """Test emitting sync event on success."""
        daemon = TrainingDataSyncDaemon(config=sync_config)
        daemon._running = True

        result = SyncResult(
            config_key="hex8_2p",
            success=True,
            source=mock_data_source.OWC,
            bytes_transferred=1024,
            duration_seconds=2.5,
        )

        with patch(
            "app.distributed.data_events.emit_data_event"
        ) as mock_emit:
            await daemon._emit_sync_event(result)

        # Verify emit was called
        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_sync_event_handles_import_error(self, sync_config):
        """Test that emit handles import errors gracefully."""
        daemon = TrainingDataSyncDaemon(config=sync_config)

        result = SyncResult(
            config_key="hex8_2p",
            success=True,
        )

        # The method catches all exceptions in the import, so we patch the import
        # to raise and verify it doesn't propagate
        with patch.dict("sys.modules", {"app.distributed.data_events": None}):
            # Should not raise - method catches all exceptions
            await daemon._emit_sync_event(result)


# =============================================================================
# Test Singleton Pattern
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern functions."""

    def test_get_training_data_sync_daemon_singleton(self):
        """Test that get_training_data_sync_daemon returns same instance."""
        daemon1 = get_training_data_sync_daemon()
        daemon2 = get_training_data_sync_daemon()
        assert daemon1 is daemon2

    def test_reset_training_data_sync_daemon(self):
        """Test that reset clears the singleton."""
        daemon1 = get_training_data_sync_daemon()
        reset_training_data_sync_daemon()
        daemon2 = get_training_data_sync_daemon()
        assert daemon1 is not daemon2

    def test_get_after_reset_creates_new_instance(self):
        """Test that get after reset creates fresh instance."""
        daemon1 = get_training_data_sync_daemon()
        # Jan 2026: Set a daemon-specific attribute instead of dict key
        daemon1._syncs_completed = 999

        reset_training_data_sync_daemon()
        daemon2 = get_training_data_sync_daemon()

        # Fresh instance has zero syncs_completed
        assert daemon2._syncs_completed == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for training data sync."""

    @pytest.mark.asyncio
    async def test_full_sync_workflow(self, sync_config, tmp_path):
        """Test complete sync workflow from manifest to local file."""
        from app.coordination.training_data_manifest import (
            DataSource,
            TrainingDataEntry,
            TrainingDataManifest,
        )

        # Create mock manifest with OWC entry
        mock_entry = TrainingDataEntry(
            config_key="hex8_2p",
            path="/Volumes/RingRift-Data/training/hex8_2p.npz",
            source=DataSource.OWC,
            size_bytes=50 * 1024 * 1024,
            sample_count=25000,
            modified_time=datetime.now(tz=timezone.utc),
        )

        mock_manifest = MagicMock(spec=TrainingDataManifest)
        mock_manifest.get_best_data.return_value = mock_entry

        # Mock successful rsync
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        # Create fake downloaded file
        local_path = tmp_path / "hex8_2p.npz"

        with patch(
            "app.coordination.training_data_sync_daemon.get_training_data_manifest",
            new=AsyncMock(return_value=mock_manifest),
        ):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                # Pre-create the file to simulate download
                local_path.write_bytes(b"fake npz content")

                result = await sync_training_data_for_config(
                    "hex8_2p",
                    local_dir=tmp_path,
                    config=sync_config,
                )

        assert result.success is True
        assert result.source == DataSource.OWC
        assert result.bytes_transferred > 0
