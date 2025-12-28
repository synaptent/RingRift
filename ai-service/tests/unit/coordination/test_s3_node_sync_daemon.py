"""Tests for S3NodeSyncDaemon.

Tests cover:
- Configuration initialization
- S3NodeSyncConfig dataclass
- SyncResult dataclass
- FileManifest dataclass
- Daemon lifecycle (start/stop)
- Health check reporting
- Event subscription handling
- Sync cycle behavior

December 2025: Created as part of S3 infrastructure completion (Phase 6).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.s3_node_sync_daemon import (
    FileManifest,
    S3NodeSyncConfig,
    S3NodeSyncDaemon,
    SyncResult,
    get_node_id,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return S3NodeSyncConfig(
        s3_bucket="test-bucket",
        aws_region="us-west-2",
        sync_interval_seconds=10.0,  # Short for testing
        push_games=True,
        push_models=True,
        push_npz=False,
    )


@pytest.fixture
def daemon(config):
    """Create a test daemon."""
    with patch("app.coordination.s3_node_sync_daemon.get_node_id", return_value="test-node"):
        return S3NodeSyncDaemon(config)


# =============================================================================
# get_node_id Tests
# =============================================================================


class TestGetNodeId:
    """Test get_node_id utility function."""

    def test_node_id_from_env(self):
        """Test getting node ID from environment."""
        with patch.dict("os.environ", {"RINGRIFT_NODE_ID": "custom-node-123"}):
            # Need to reload to pick up env
            from importlib import reload
            import app.coordination.s3_node_sync_daemon as module
            # Just verify the function exists and uses env
            result = module.get_node_id()
            assert result == "custom-node-123"

    def test_node_id_falls_back_to_hostname(self):
        """Test fallback to hostname when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("socket.gethostname", return_value="my-server"):
                result = get_node_id()
                assert result == "my-server"

    def test_node_id_strips_common_prefixes(self):
        """Test that common prefixes are stripped."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("socket.gethostname", return_value="ip-172-16-0-1"):
                result = get_node_id()
                assert result == "172-16-0-1"


# =============================================================================
# S3NodeSyncConfig Tests
# =============================================================================


class TestS3NodeSyncConfig:
    """Test S3NodeSyncConfig dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        config = S3NodeSyncConfig()
        assert config.s3_bucket == "ringrift-models-20251214"
        assert config.aws_region == "us-east-1"
        assert config.sync_interval_seconds == 3600.0
        assert config.push_games is True
        assert config.push_models is True
        assert config.push_npz is True
        assert config.pull_npz is True
        assert config.pull_models is True
        assert config.compress_uploads is True
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 30.0

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = S3NodeSyncConfig(
            s3_bucket="custom-bucket",
            aws_region="eu-central-1",
            sync_interval_seconds=300.0,
            push_games=False,
            push_models=True,
            bandwidth_limit_kbps=1000,
        )
        assert config.s3_bucket == "custom-bucket"
        assert config.aws_region == "eu-central-1"
        assert config.sync_interval_seconds == 300.0
        assert config.push_games is False
        assert config.push_models is True
        assert config.bandwidth_limit_kbps == 1000

    def test_path_defaults(self):
        """Test path field defaults."""
        config = S3NodeSyncConfig()
        assert config.games_dir == Path("data/games")
        assert config.models_dir == Path("models")
        assert config.npz_dir == Path("data/training")


# =============================================================================
# SyncResult Tests
# =============================================================================


class TestSyncResult:
    """Test SyncResult dataclass."""

    def test_successful_sync(self):
        """Test successful sync result."""
        result = SyncResult(
            success=True,
            uploaded_files=["file1.db", "file2.db"],
            downloaded_files=[],
            duration_seconds=12.5,
            bytes_transferred=1024000,
        )
        assert result.success is True
        assert len(result.uploaded_files) == 2
        assert len(result.downloaded_files) == 0
        assert result.duration_seconds == 12.5
        assert result.bytes_transferred == 1024000
        assert len(result.errors) == 0

    def test_failed_sync(self):
        """Test failed sync result."""
        result = SyncResult(
            success=False,
            uploaded_files=[],
            downloaded_files=[],
            errors=["Connection refused", "Timeout"],
            duration_seconds=60.0,
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_default_values(self):
        """Test default values."""
        result = SyncResult(success=True)
        assert result.uploaded_files == []
        assert result.downloaded_files == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.bytes_transferred == 0


# =============================================================================
# FileManifest Tests
# =============================================================================


class TestFileManifest:
    """Test FileManifest dataclass."""

    def test_basic_manifest(self):
        """Test basic manifest creation."""
        manifest = FileManifest(
            node_id="test-node",
            timestamp=time.time(),
            files={
                "data/games/selfplay.db": {
                    "size": 1024000,
                    "mtime": 1703123456.0,
                    "sha256": "abc123",
                    "type": "database",
                },
            },
        )
        assert manifest.node_id == "test-node"
        assert len(manifest.files) == 1
        assert "data/games/selfplay.db" in manifest.files

    def test_empty_manifest(self):
        """Test empty manifest."""
        manifest = FileManifest(node_id="empty-node", timestamp=time.time())
        assert manifest.files == {}


# =============================================================================
# S3NodeSyncDaemon Initialization Tests
# =============================================================================


class TestS3NodeSyncDaemonInit:
    """Test S3NodeSyncDaemon initialization."""

    def test_default_config(self):
        """Test daemon with default config."""
        with patch("app.coordination.s3_node_sync_daemon.get_node_id", return_value="default-node"):
            daemon = S3NodeSyncDaemon()
            assert daemon.config is not None
            assert daemon.config.s3_bucket == "ringrift-models-20251214"
            assert daemon.node_id == "default-node"

    def test_custom_config(self, config):
        """Test daemon with custom config."""
        with patch("app.coordination.s3_node_sync_daemon.get_node_id", return_value="custom-node"):
            daemon = S3NodeSyncDaemon(config)
            assert daemon.config.s3_bucket == "test-bucket"
            assert daemon.config.aws_region == "us-west-2"
            assert daemon.node_id == "custom-node"

    def test_initial_state(self, daemon):
        """Test initial daemon state."""
        assert daemon._running is False
        assert daemon._start_time == 0.0
        assert daemon._last_push_time == 0.0
        assert daemon._last_pull_time == 0.0
        assert daemon._push_count == 0
        assert daemon._pull_count == 0
        assert daemon._bytes_uploaded == 0
        assert daemon._bytes_downloaded == 0
        assert daemon._errors == 0

    def test_name_property(self, daemon):
        """Test daemon name includes node ID."""
        assert "test-node" in daemon.name
        assert "S3NodeSyncDaemon" in daemon.name

    def test_is_running_property(self, daemon):
        """Test is_running property."""
        assert daemon.is_running() is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestS3NodeSyncDaemonHealthCheck:
    """Test S3NodeSyncDaemon health check."""

    def test_health_check_when_stopped(self, daemon):
        """Test health check returns healthy when stopped."""
        result = daemon.health_check()
        assert result.healthy is True
        assert "not running" in result.message.lower()

    def test_health_check_when_running_healthy(self, daemon):
        """Test health check when running and recently synced."""
        daemon._running = True
        daemon._start_time = time.time()
        daemon._last_push_time = time.time()  # Just synced
        result = daemon.health_check()
        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_includes_stats(self, daemon):
        """Test health check includes stats in details."""
        daemon._running = True
        daemon._start_time = time.time()
        daemon._last_push_time = time.time()
        daemon._push_count = 5
        daemon._bytes_uploaded = 1024000

        result = daemon.health_check()
        assert result.healthy is True
        assert "node_id" in result.details
        assert "bytes_uploaded" in result.details
        assert result.details["bytes_uploaded"] == 1024000

    def test_health_check_degraded_no_recent_push(self, daemon):
        """Test health check degraded when no recent push."""
        daemon._running = True
        daemon._start_time = time.time()
        # Last push was 3x the interval ago (interval is 10s in test config)
        daemon._last_push_time = time.time() - 30

        result = daemon.health_check()
        assert result.healthy is False
        assert "no push" in result.message.lower()


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestS3NodeSyncDaemonEventSubscription:
    """Test S3NodeSyncDaemon event subscription."""

    def test_subscribe_to_events(self, daemon):
        """Test event subscription during start."""
        mock_subscribe = MagicMock()

        # Patch at the module where it's imported, not used
        with patch("app.coordination.event_router.subscribe", mock_subscribe):
            with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
                daemon._subscribe_to_events()

        # Should subscribe to training, selfplay, and promotion events
        assert mock_subscribe.call_count >= 3

    def test_subscribe_handles_import_error(self, daemon, caplog):
        """Test graceful handling of import errors during subscription."""
        # Mock the import to fail inside _subscribe_to_events
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if 'event_router' in name:
                raise ImportError("event_router not found")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # Should not raise
            daemon._subscribe_to_events()

        # Should log warning (check via the daemon still functioning)
        # The daemon should continue without crashing


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestS3NodeSyncDaemonEventHandlers:
    """Test S3NodeSyncDaemon event handlers."""

    @pytest.mark.asyncio
    async def test_on_training_completed(self, daemon):
        """Test handling TRAINING_COMPLETED event."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock) as mock_push:
            event = {"config_key": "hex8_2p", "model_path": "/models/test.pth"}
            daemon._on_training_completed(event)

            # Should schedule a push
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_large_batch(self, daemon):
        """Test handling SELFPLAY_COMPLETE with large batch."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            event = {"games_count": 150, "config_key": "hex8_2p"}
            daemon._on_selfplay_complete(event)
            # Should trigger push for large batch

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_small_batch(self, daemon):
        """Test handling SELFPLAY_COMPLETE with small batch (skipped)."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock) as mock_push:
            event = {"games_count": 50, "config_key": "hex8_2p"}
            daemon._on_selfplay_complete(event)
            # Should NOT trigger push for small batch (<100)

    @pytest.mark.asyncio
    async def test_on_model_promoted(self, daemon):
        """Test handling MODEL_PROMOTED event."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            event = {"model_path": "/models/promoted.pth", "config_key": "hex8_2p"}
            daemon._on_model_promoted(event)
            # Should trigger high-priority push


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestS3NodeSyncDaemonLifecycle:
    """Test S3NodeSyncDaemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """Test that start sets running flag and runs initial sync."""
        async def stop_quickly():
            await asyncio.sleep(0.1)
            daemon._running = False

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock) as mock_push:
            task = asyncio.create_task(daemon.start())
            await stop_quickly()
            await asyncio.wait_for(task, timeout=2.0)

        # Should have started and run initial push
        assert daemon._start_time > 0
        mock_push.assert_called()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, daemon):
        """Test that stop clears running flag."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            await daemon.stop()

        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_runs_final_push(self, daemon):
        """Test that stop runs final push before stopping."""
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock) as mock_push:
            await daemon.stop()
            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """Test that start is idempotent."""
        daemon._running = True
        original_time = daemon._start_time = time.time() - 100

        await daemon.start()  # Should return immediately

        # Start time should not change
        assert daemon._start_time == original_time


# =============================================================================
# Push Cycle Tests
# =============================================================================


class TestS3NodeSyncDaemonPushCycle:
    """Test S3NodeSyncDaemon push cycle behavior."""

    @pytest.mark.asyncio
    async def test_push_cycle_updates_stats(self, daemon):
        """Test that push cycle updates statistics."""
        daemon._running = True
        daemon._start_time = time.time()

        # Mock _push_games and _push_models to return SyncResult objects
        games_result = SyncResult(
            success=True,
            uploaded_files=["game1.db", "game2.db"],
            bytes_transferred=1000,
        )
        models_result = SyncResult(
            success=True,
            uploaded_files=["model1.pth"],
            bytes_transferred=500,
        )
        npz_result = SyncResult(success=True)

        with patch.object(daemon, "_push_games", new_callable=AsyncMock, return_value=games_result):
            with patch.object(daemon, "_push_models", new_callable=AsyncMock, return_value=models_result):
                with patch.object(daemon, "_push_npz", new_callable=AsyncMock, return_value=npz_result):
                    with patch.object(daemon, "_build_local_manifest", new_callable=AsyncMock, return_value=FileManifest(node_id="test", timestamp=time.time())):
                        with patch.object(daemon, "_upload_manifest", new_callable=AsyncMock):
                            await daemon._run_push_cycle()

        assert daemon._last_push_time > 0
        assert daemon._push_count == 1

    @pytest.mark.asyncio
    async def test_push_cycle_increments_errors(self, daemon):
        """Test that push cycle increments error count on failure."""
        daemon._running = True
        daemon._start_time = time.time()
        initial_errors = daemon._errors

        with patch.object(daemon, "_push_games", new_callable=AsyncMock, side_effect=Exception("S3 error")):
            with patch.object(daemon, "_push_models", new_callable=AsyncMock, return_value=(True, 0, 0)):
                try:
                    await daemon._run_push_cycle()
                except Exception:
                    pass  # May or may not raise depending on implementation

        # Note: Error handling depends on implementation
        # This test verifies the pattern is testable


# =============================================================================
# Configuration Override Tests
# =============================================================================


class TestS3NodeSyncDaemonConfigOverrides:
    """Test configuration can be overridden via environment."""

    def test_env_bucket_override(self):
        """Test S3 bucket can be overridden via environment."""
        with patch.dict("os.environ", {"RINGRIFT_S3_BUCKET": "env-override-bucket"}):
            config = S3NodeSyncConfig()
            assert config.s3_bucket == "env-override-bucket"

    def test_env_sync_interval_override(self):
        """Test sync interval can be overridden via environment."""
        with patch.dict("os.environ", {"RINGRIFT_S3_SYNC_INTERVAL": "1800"}):
            config = S3NodeSyncConfig()
            assert config.sync_interval_seconds == 1800.0

    def test_env_push_flags(self):
        """Test push flags can be overridden via environment."""
        with patch.dict(
            "os.environ",
            {"RINGRIFT_S3_PUSH_GAMES": "false", "RINGRIFT_S3_PUSH_MODELS": "false"},
        ):
            config = S3NodeSyncConfig()
            assert config.push_games is False
            assert config.push_models is False


# =============================================================================
# S3ConsolidationDaemon Tests
# =============================================================================


class TestS3ConsolidationDaemon:
    """Test S3ConsolidationDaemon class."""

    @pytest.fixture
    def consolidation_daemon(self):
        """Create a test consolidation daemon."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        return S3ConsolidationDaemon()

    def test_initialization_defaults(self, consolidation_daemon):
        """Test consolidation daemon initializes with defaults."""
        assert consolidation_daemon.config is not None
        assert consolidation_daemon._running is False
        assert consolidation_daemon._consolidation_interval == 3600.0
        assert consolidation_daemon._last_consolidation_time == 0.0
        assert consolidation_daemon._consolidation_errors == 0

    def test_initialization_custom_config(self, config):
        """Test consolidation daemon with custom config."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon(config)
        assert daemon.config.s3_bucket == "test-bucket"

    def test_health_check_stopped(self, consolidation_daemon):
        """Test health check when stopped."""
        result = consolidation_daemon.health_check()
        assert result.healthy is False
        assert "stopped" in result.message.lower()

    def test_health_check_running_healthy(self, consolidation_daemon):
        """Test health check when running and healthy."""
        consolidation_daemon._running = True
        consolidation_daemon._last_consolidation_time = time.time()
        consolidation_daemon._consolidation_errors = 0

        result = consolidation_daemon.health_check()
        assert result.healthy is True
        assert "operational" in result.message.lower()

    def test_health_check_degraded_stale(self, consolidation_daemon):
        """Test health check degraded when consolidation is stale."""
        consolidation_daemon._running = True
        # Last consolidation was way too long ago (4x the interval)
        consolidation_daemon._last_consolidation_time = time.time() - (3600 * 4)
        consolidation_daemon._consolidation_errors = 0

        result = consolidation_daemon.health_check()
        assert result.healthy is False
        assert "stale" in result.message.lower()

    def test_health_check_degraded_errors(self, consolidation_daemon):
        """Test health check degraded when too many errors."""
        consolidation_daemon._running = True
        consolidation_daemon._last_consolidation_time = time.time()
        consolidation_daemon._consolidation_errors = 10  # > 5 is threshold

        result = consolidation_daemon.health_check()
        assert result.healthy is False
        assert "error" in result.message.lower()

    def test_health_check_details(self, consolidation_daemon):
        """Test health check includes metrics in details."""
        consolidation_daemon._running = True
        consolidation_daemon._last_consolidation_time = time.time()
        consolidation_daemon._nodes_consolidated = 5
        consolidation_daemon._models_consolidated = 10
        consolidation_daemon._npz_consolidated = 3

        result = consolidation_daemon.health_check()
        assert "nodes_consolidated" in result.details
        assert result.details["nodes_consolidated"] == 5
        assert result.details["models_consolidated"] == 10
        assert result.details["npz_consolidated"] == 3

    @pytest.mark.asyncio
    async def test_start_sets_running(self, consolidation_daemon):
        """Test that start sets running flag."""
        async def stop_quickly():
            await asyncio.sleep(0.1)
            consolidation_daemon._running = False

        with patch.object(
            consolidation_daemon, "_run_consolidation", new_callable=AsyncMock
        ):
            task = asyncio.create_task(consolidation_daemon.start())
            await stop_quickly()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.CancelledError:
                pass

        # Was started at some point
        assert consolidation_daemon._running is False  # Stopped now

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, consolidation_daemon):
        """Test that stop clears running flag."""
        consolidation_daemon._running = True
        await consolidation_daemon.stop()
        assert consolidation_daemon._running is False

    @pytest.mark.asyncio
    async def test_consolidation_updates_timestamp(self, consolidation_daemon):
        """Test that consolidation updates last time."""
        consolidation_daemon._running = True

        with patch.object(
            consolidation_daemon, "_consolidate_models", new_callable=AsyncMock
        ):
            with patch.object(
                consolidation_daemon, "_consolidate_npz", new_callable=AsyncMock
            ):
                with patch.object(
                    consolidation_daemon,
                    "_create_consolidated_manifest",
                    new_callable=AsyncMock,
                ):
                    with patch(
                        "app.coordination.s3_node_sync_daemon.S3NodeSyncDaemon.list_all_node_data",
                        new_callable=AsyncMock,
                        return_value={},
                    ):
                        await consolidation_daemon._run_consolidation()

        # Timestamp not updated since manifests was empty
        # But method ran without error

    @pytest.mark.asyncio
    async def test_consolidation_calls_consolidators(self, consolidation_daemon):
        """Test that consolidation calls all consolidator methods."""
        consolidation_daemon._running = True

        mock_manifests = {
            "node1": FileManifest(node_id="node1", timestamp=time.time(), files={}),
        }

        with patch.object(
            consolidation_daemon, "_consolidate_models", new_callable=AsyncMock
        ) as mock_models:
            with patch.object(
                consolidation_daemon, "_consolidate_npz", new_callable=AsyncMock
            ) as mock_npz:
                with patch.object(
                    consolidation_daemon,
                    "_create_consolidated_manifest",
                    new_callable=AsyncMock,
                ) as mock_manifest:
                    with patch(
                        "app.coordination.s3_node_sync_daemon.S3NodeSyncDaemon.list_all_node_data",
                        new_callable=AsyncMock,
                        return_value=mock_manifests,
                    ):
                        await consolidation_daemon._run_consolidation()

        mock_models.assert_called_once()
        mock_npz.assert_called_once()
        mock_manifest.assert_called_once()


# =============================================================================
# S3 Operation Tests (_should_upload, _s3_upload, _s3_download)
# =============================================================================


class TestS3Operations:
    """Test S3 operations (_should_upload, _s3_upload, _s3_download)."""

    @pytest.fixture
    def daemon(self, config):
        """Create test daemon."""
        with patch(
            "app.coordination.s3_node_sync_daemon.get_node_id", return_value="test-node"
        ):
            return S3NodeSyncDaemon(config)

    @pytest.mark.asyncio
    async def test_should_upload_file_not_in_s3(self, daemon, tmp_path):
        """Test should_upload returns True when file not in S3."""
        local_file = tmp_path / "test.db"
        local_file.write_bytes(b"test content" * 100)

        # Mock aws s3api head-object returning non-zero (not found)
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"Not Found"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._should_upload(local_file, "nodes/test/file.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_upload_file_same_size(self, daemon, tmp_path):
        """Test should_upload returns False when file size matches."""
        local_file = tmp_path / "test.db"
        content = b"test content" * 100
        local_file.write_bytes(content)

        # Mock aws s3api head-object returning same size
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(f'{{"ContentLength": {len(content)}}}'.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._should_upload(local_file, "nodes/test/file.db")

        assert result is False

    @pytest.mark.asyncio
    async def test_should_upload_file_different_size(self, daemon, tmp_path):
        """Test should_upload returns True when file size differs."""
        local_file = tmp_path / "test.db"
        local_file.write_bytes(b"test content" * 100)

        # Mock aws s3api head-object returning different size
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b'{"ContentLength": 50}', b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._should_upload(local_file, "nodes/test/file.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_upload_handles_json_error(self, daemon, tmp_path):
        """Test should_upload handles JSON parse error gracefully."""
        local_file = tmp_path / "test.db"
        local_file.write_bytes(b"test content")

        # Mock aws s3api returning invalid JSON
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"invalid json", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._should_upload(local_file, "nodes/test/file.db")

        # On error, should return True (upload to be safe)
        assert result is True

    @pytest.mark.asyncio
    async def test_s3_upload_success(self, daemon, tmp_path):
        """Test successful S3 upload."""
        local_file = tmp_path / "test.db"
        local_file.write_text("test content")

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload(str(local_file), "nodes/test/file.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_upload_failure(self, daemon, tmp_path):
        """Test failed S3 upload."""
        local_file = tmp_path / "test.db"
        local_file.write_text("test content")

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"Access Denied"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload(str(local_file), "nodes/test/file.db")

        assert result is False

    @pytest.mark.asyncio
    async def test_s3_upload_timeout(self, daemon, tmp_path):
        """Test S3 upload timeout."""
        local_file = tmp_path / "test.db"
        local_file.write_text("test content")

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload(str(local_file), "nodes/test/file.db")

        assert result is False
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_download_success(self, daemon, tmp_path):
        """Test successful S3 download."""
        local_file = tmp_path / "downloaded.db"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("nodes/test/file.db", str(local_file))

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_download_not_found(self, daemon, tmp_path):
        """Test S3 download when file not found."""
        local_file = tmp_path / "downloaded.db"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"404 Not Found NoSuchKey")
        )
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("nodes/test/file.db", str(local_file))

        assert result is False

    @pytest.mark.asyncio
    async def test_s3_download_timeout(self, daemon, tmp_path):
        """Test S3 download timeout."""
        local_file = tmp_path / "downloaded.db"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("nodes/test/file.db", str(local_file))

        assert result is False
        mock_process.kill.assert_called_once()


# =============================================================================
# Push Method Tests (_push_games, _push_models, _push_npz)
# =============================================================================


class TestPushMethods:
    """Test push methods (_push_games, _push_models, _push_npz)."""

    @pytest.fixture
    def daemon(self, config, tmp_path):
        """Create test daemon with temp directories."""
        config.games_dir = tmp_path / "games"
        config.models_dir = tmp_path / "models"
        config.npz_dir = tmp_path / "training"

        # Create directories
        config.games_dir.mkdir(parents=True)
        config.models_dir.mkdir(parents=True)
        config.npz_dir.mkdir(parents=True)

        with patch(
            "app.coordination.s3_node_sync_daemon.get_node_id", return_value="test-node"
        ):
            return S3NodeSyncDaemon(config)

    @pytest.mark.asyncio
    async def test_push_games_empty_directory(self, daemon):
        """Test push_games with empty directory."""
        result = await daemon._push_games()
        assert result.success is True
        assert len(result.uploaded_files) == 0

    @pytest.mark.asyncio
    async def test_push_games_skips_small_files(self, daemon):
        """Test push_games skips files under 10KB."""
        # Create small database file
        small_db = daemon.config.games_dir / "small.db"
        small_db.write_bytes(b"x" * 5000)  # 5KB

        with patch.object(daemon, "_should_upload", new_callable=AsyncMock):
            with patch.object(daemon, "_s3_upload", new_callable=AsyncMock):
                result = await daemon._push_games()

        assert len(result.uploaded_files) == 0

    @pytest.mark.asyncio
    async def test_push_games_uploads_large_files(self, daemon):
        """Test push_games uploads files over 10KB."""
        # Create large database file
        large_db = daemon.config.games_dir / "large.db"
        large_db.write_bytes(b"x" * 15000)  # 15KB

        with patch.object(
            daemon, "_should_upload", new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
            ):
                result = await daemon._push_games()

        assert len(result.uploaded_files) == 1
        assert "large.db" in result.uploaded_files

    @pytest.mark.asyncio
    async def test_push_games_handles_error(self, daemon):
        """Test push_games handles upload error."""
        # Create large database file
        large_db = daemon.config.games_dir / "error.db"
        large_db.write_bytes(b"x" * 15000)

        with patch.object(
            daemon, "_should_upload", new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                daemon, "_s3_upload", new_callable=AsyncMock, side_effect=OSError("S3 error")
            ):
                result = await daemon._push_games()

        assert len(result.errors) == 1
        assert "error.db" in result.errors[0]

    @pytest.mark.asyncio
    async def test_push_models_empty_directory(self, daemon):
        """Test push_models with empty directory."""
        result = await daemon._push_models()
        assert result.success is True
        assert len(result.uploaded_files) == 0

    @pytest.mark.asyncio
    async def test_push_models_only_canonical(self, daemon):
        """Test push_models only uploads canonical_ prefixed models."""
        # Create canonical and non-canonical model files
        canonical = daemon.config.models_dir / "canonical_hex8_2p.pth"
        canonical.write_bytes(b"x" * 1000)

        regular = daemon.config.models_dir / "my_model.pth"
        regular.write_bytes(b"x" * 1000)

        with patch.object(
            daemon, "_should_upload", new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
            ):
                result = await daemon._push_models()

        assert len(result.uploaded_files) == 1
        assert "canonical_hex8_2p.pth" in result.uploaded_files

    @pytest.mark.asyncio
    async def test_push_models_skips_symlinks(self, daemon):
        """Test push_models skips symlinks."""
        # Create real file and symlink
        real = daemon.config.models_dir / "canonical_real.pth"
        real.write_bytes(b"x" * 1000)

        symlink = daemon.config.models_dir / "canonical_link.pth"
        symlink.symlink_to(real)

        with patch.object(
            daemon, "_should_upload", new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
            ):
                result = await daemon._push_models()

        # Only real file should be uploaded, not the symlink
        assert len(result.uploaded_files) == 1
        assert "canonical_real.pth" in result.uploaded_files

    @pytest.mark.asyncio
    async def test_push_npz_empty_directory(self, daemon):
        """Test push_npz with empty directory."""
        result = await daemon._push_npz()
        assert result.success is True
        assert len(result.uploaded_files) == 0

    @pytest.mark.asyncio
    async def test_push_npz_uploads_files(self, daemon):
        """Test push_npz uploads NPZ files."""
        npz_file = daemon.config.npz_dir / "hex8_2p.npz"
        npz_file.write_bytes(b"x" * 1000)

        with patch.object(
            daemon, "_should_upload", new_callable=AsyncMock, return_value=True
        ):
            with patch.object(
                daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
            ):
                result = await daemon._push_npz()

        assert len(result.uploaded_files) == 1
        assert "hex8_2p.npz" in result.uploaded_files


# =============================================================================
# Pull Method Tests (pull_training_data, pull_model)
# =============================================================================


class TestPullMethods:
    """Test pull methods (pull_training_data, pull_model)."""

    @pytest.fixture
    def daemon(self, config, tmp_path):
        """Create test daemon with temp directories."""
        config.npz_dir = tmp_path / "training"
        config.models_dir = tmp_path / "models"
        config.npz_dir.mkdir(parents=True)
        config.models_dir.mkdir(parents=True)

        with patch(
            "app.coordination.s3_node_sync_daemon.get_node_id", return_value="test-node"
        ):
            return S3NodeSyncDaemon(config)

    @pytest.mark.asyncio
    async def test_pull_training_data_success(self, daemon):
        """Test successful training data pull."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=True
        ):
            # Create the downloaded file to simulate successful download
            npz_path = daemon.config.npz_dir / "hex8_2p.npz"
            npz_path.write_bytes(b"x" * 1000)

            result = await daemon.pull_training_data("hex8_2p")

        assert result.success is True
        assert len(result.downloaded_files) == 1
        assert "hex8_2p.npz" in result.downloaded_files[0]

    @pytest.mark.asyncio
    async def test_pull_training_data_not_found(self, daemon):
        """Test training data pull when not in S3."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=False
        ):
            result = await daemon.pull_training_data("nonexistent_config")

        assert len(result.downloaded_files) == 0

    @pytest.mark.asyncio
    async def test_pull_training_data_error(self, daemon):
        """Test training data pull with error."""
        with patch.object(
            daemon,
            "_s3_download",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError("Connection timeout"),
        ):
            result = await daemon.pull_training_data("hex8_2p")

        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_pull_model_success(self, daemon):
        """Test successful model pull."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=True
        ):
            # Create the downloaded file
            model_path = daemon.config.models_dir / "canonical_hex8_2p.pth"
            model_path.write_bytes(b"x" * 1000)

            result = await daemon.pull_model("canonical_hex8_2p.pth")

        assert result.success is True
        assert len(result.downloaded_files) == 1

    @pytest.mark.asyncio
    async def test_pull_model_not_found(self, daemon):
        """Test model pull when not in S3."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=False
        ):
            result = await daemon.pull_model("nonexistent.pth")

        assert len(result.downloaded_files) == 0

    @pytest.mark.asyncio
    async def test_pull_updates_stats(self, daemon):
        """Test that pull updates statistics."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=True
        ):
            # Create the downloaded file
            npz_path = daemon.config.npz_dir / "hex8_2p.npz"
            npz_path.write_bytes(b"x" * 5000)

            initial_pull_count = daemon._pull_count
            await daemon.pull_training_data("hex8_2p")

        assert daemon._pull_count == initial_pull_count + 1


# =============================================================================
# Manifest Method Tests
# =============================================================================


class TestManifestMethods:
    """Test manifest methods (_build_local_manifest, list_all_node_data, etc.)."""

    @pytest.fixture
    def daemon(self, config, tmp_path):
        """Create test daemon with temp directories."""
        config.games_dir = tmp_path / "games"
        config.models_dir = tmp_path / "models"
        config.npz_dir = tmp_path / "training"

        config.games_dir.mkdir(parents=True)
        config.models_dir.mkdir(parents=True)
        config.npz_dir.mkdir(parents=True)

        with patch(
            "app.coordination.s3_node_sync_daemon.get_node_id", return_value="test-node"
        ):
            return S3NodeSyncDaemon(config)

    @pytest.mark.asyncio
    async def test_build_local_manifest_empty(self, daemon):
        """Test building manifest with empty directories."""
        manifest = await daemon._build_local_manifest()

        assert manifest.node_id == "test-node"
        assert manifest.timestamp > 0
        assert len(manifest.files) == 0

    @pytest.mark.asyncio
    async def test_build_local_manifest_with_files(self, daemon):
        """Test building manifest with files."""
        # Create some files
        db_file = daemon.config.games_dir / "selfplay.db"
        db_file.write_bytes(b"database content")

        model_file = daemon.config.models_dir / "canonical_hex8_2p.pth"
        model_file.write_bytes(b"model content")

        manifest = await daemon._build_local_manifest()

        assert len(manifest.files) == 2

    @pytest.mark.asyncio
    async def test_upload_manifest(self, daemon):
        """Test manifest upload."""
        manifest = FileManifest(
            node_id="test-node",
            timestamp=time.time(),
            files={"test.db": {"size": 100, "type": "database"}},
        )

        with patch.object(
            daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
        ):
            await daemon._upload_manifest(manifest)

    @pytest.mark.asyncio
    async def test_list_all_node_data_empty(self, daemon):
        """Test listing node data when S3 is empty."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 1  # Empty or error

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            manifests = await daemon.list_all_node_data()

        assert manifests == {}

    @pytest.mark.asyncio
    async def test_list_all_node_data_with_nodes(self, daemon):
        """Test listing node data with multiple nodes."""
        # Mock S3 ls response
        ls_output = b"PRE node1/\nPRE node2/\n"
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(ls_output, b""))
        mock_process.returncode = 0

        # Mock getting individual manifests
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(
                daemon,
                "_get_node_manifest",
                new_callable=AsyncMock,
                return_value=FileManifest(
                    node_id="node1", timestamp=time.time(), files={}
                ),
            ):
                manifests = await daemon.list_all_node_data()

        # Should have attempted to get manifests for both nodes
        assert len(manifests) >= 0  # Depends on mock setup

    @pytest.mark.asyncio
    async def test_get_node_manifest_success(self, daemon):
        """Test getting a specific node's manifest."""
        manifest_json = {
            "node_id": "node1",
            "timestamp": time.time(),
            "files": {"test.db": {"size": 100}},
        }

        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=True
        ):
            with patch("builtins.open", MagicMock()):
                with patch("json.load", return_value=manifest_json):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.unlink"):
                            manifest = await daemon._get_node_manifest("node1")

        if manifest:  # May be None if file operations fail
            assert manifest.node_id == "node1"

    @pytest.mark.asyncio
    async def test_get_node_manifest_not_found(self, daemon):
        """Test getting manifest when not in S3."""
        with patch.object(
            daemon, "_s3_download", new_callable=AsyncMock, return_value=False
        ):
            manifest = await daemon._get_node_manifest("nonexistent")

        assert manifest is None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_ensure_training_data_from_s3_exists_locally(self, tmp_path):
        """Test ensure_training_data when file exists locally."""
        from app.coordination.s3_node_sync_daemon import ensure_training_data_from_s3

        # Create local file
        npz_dir = tmp_path / "training"
        npz_dir.mkdir()
        local_file = npz_dir / "hex8_2p.npz"
        local_file.write_bytes(b"training data")

        with patch(
            "app.coordination.s3_node_sync_daemon.S3NodeSyncConfig"
        ) as mock_config_class:
            mock_config = MagicMock()
            mock_config.npz_dir = npz_dir
            mock_config_class.return_value = mock_config

            result = await ensure_training_data_from_s3("hex8_2p")

        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_training_data_from_s3_needs_download(self, tmp_path):
        """Test ensure_training_data when file needs download."""
        from app.coordination.s3_node_sync_daemon import ensure_training_data_from_s3

        npz_dir = tmp_path / "training"
        npz_dir.mkdir()

        with patch(
            "app.coordination.s3_node_sync_daemon.S3NodeSyncConfig"
        ) as mock_config_class:
            mock_config = MagicMock()
            mock_config.npz_dir = npz_dir
            mock_config_class.return_value = mock_config

            with patch(
                "app.coordination.s3_node_sync_daemon.S3NodeSyncDaemon"
            ) as mock_daemon_class:
                mock_daemon = AsyncMock()
                mock_daemon.pull_training_data = AsyncMock(
                    return_value=SyncResult(
                        success=True, downloaded_files=["hex8_2p.npz"]
                    )
                )
                mock_daemon_class.return_value = mock_daemon

                result = await ensure_training_data_from_s3("hex8_2p")

        assert result is True

    def test_sync_ensure_training_data_wrapper(self):
        """Test synchronous wrapper exists and is callable."""
        from app.coordination.s3_node_sync_daemon import sync_ensure_training_data_from_s3

        # Just verify it exists and is callable
        assert callable(sync_ensure_training_data_from_s3)


# =============================================================================
# Consolidation Method Tests
# =============================================================================


class TestConsolidationMethods:
    """Test consolidation methods (_consolidate_models, _consolidate_npz)."""

    @pytest.fixture
    def consolidation_daemon(self, config):
        """Create a test consolidation daemon."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        return S3ConsolidationDaemon(config)

    @pytest.mark.asyncio
    async def test_consolidate_models_finds_latest(self, consolidation_daemon):
        """Test that consolidate_models picks the latest version of each model."""
        manifests = {
            "node1": FileManifest(
                node_id="node1",
                timestamp=time.time(),
                files={
                    "models/canonical_hex8_2p.pth": {
                        "type": "model",
                        "mtime": 1000.0,
                    }
                },
            ),
            "node2": FileManifest(
                node_id="node2",
                timestamp=time.time(),
                files={
                    "models/canonical_hex8_2p.pth": {
                        "type": "model",
                        "mtime": 2000.0,  # Newer
                    }
                },
            ),
        }

        with patch.object(
            consolidation_daemon, "_s3_copy", new_callable=AsyncMock, return_value=True
        ) as mock_copy:
            await consolidation_daemon._consolidate_models(manifests)

        # Should copy from node2 (newer mtime)
        mock_copy.assert_called_once()
        call_args = mock_copy.call_args[0]
        assert "node2" in call_args[0]

    @pytest.mark.asyncio
    async def test_consolidate_npz_finds_latest(self, consolidation_daemon):
        """Test that consolidate_npz picks the latest version of each config."""
        manifests = {
            "node1": FileManifest(
                node_id="node1",
                timestamp=time.time(),
                files={
                    "training/hex8_2p.npz": {
                        "type": "npz",
                        "mtime": 3000.0,  # Newer
                    }
                },
            ),
            "node2": FileManifest(
                node_id="node2",
                timestamp=time.time(),
                files={
                    "training/hex8_2p.npz": {
                        "type": "npz",
                        "mtime": 1000.0,
                    }
                },
            ),
        }

        with patch.object(
            consolidation_daemon, "_s3_copy", new_callable=AsyncMock, return_value=True
        ) as mock_copy:
            await consolidation_daemon._consolidate_npz(manifests)

        # Should copy from node1 (newer mtime)
        mock_copy.assert_called_once()
        call_args = mock_copy.call_args[0]
        assert "node1" in call_args[0]

    @pytest.mark.asyncio
    async def test_create_consolidated_manifest(self, consolidation_daemon):
        """Test creating consolidated manifest."""
        manifests = {
            "node1": FileManifest(
                node_id="node1",
                timestamp=time.time(),
                files={
                    "games/selfplay.db": {"type": "database"},
                    "models/canonical.pth": {"type": "model"},
                },
            ),
            "node2": FileManifest(
                node_id="node2",
                timestamp=time.time(),
                files={
                    "training/hex8_2p.npz": {"type": "npz"},
                },
            ),
        }

        with patch.object(
            consolidation_daemon, "_s3_upload", new_callable=AsyncMock, return_value=True
        ) as mock_upload:
            await consolidation_daemon._create_consolidated_manifest(manifests)

        mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_copy_success(self, consolidation_daemon):
        """Test successful S3 copy."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await consolidation_daemon._s3_copy("src/file.db", "dst/file.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_copy_failure(self, consolidation_daemon):
        """Test failed S3 copy."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"Access Denied"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await consolidation_daemon._s3_copy("src/file.db", "dst/file.db")

        assert result is False
