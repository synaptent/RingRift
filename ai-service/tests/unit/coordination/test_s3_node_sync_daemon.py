"""Tests for S3 Node Sync Daemon.

December 2025: Comprehensive test coverage for S3 backup and sync infrastructure.
Tests cover:
- Configuration loading with environment variables
- Daemon lifecycle (start/stop)
- Event handling (TRAINING_COMPLETED, SELFPLAY_COMPLETE, MODEL_PROMOTED)
- S3 operations (upload, download, head-object, list)
- Manifest building and parsing
- Health check reporting
- Error handling and retries
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Test utilities
@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        games_dir = Path(tmpdir) / "games"
        models_dir = Path(tmpdir) / "models"
        npz_dir = Path(tmpdir) / "training"

        games_dir.mkdir()
        models_dir.mkdir()
        npz_dir.mkdir()

        yield {
            "root": Path(tmpdir),
            "games": games_dir,
            "models": models_dir,
            "npz": npz_dir,
        }


@pytest.fixture
def sample_db_file(temp_dirs):
    """Create a sample database file."""
    db_path = temp_dirs["games"] / "canonical_hex8_2p.db"
    # Create file > 10KB to pass size filter
    db_path.write_bytes(b"x" * 50000)
    return db_path


@pytest.fixture
def sample_model_file(temp_dirs):
    """Create a sample model file."""
    model_path = temp_dirs["models"] / "canonical_hex8_2p.pth"
    model_path.write_bytes(b"MODEL_DATA" * 1000)
    return model_path


@pytest.fixture
def sample_npz_file(temp_dirs):
    """Create a sample NPZ file."""
    npz_path = temp_dirs["npz"] / "hex8_2p.npz"
    npz_path.write_bytes(b"NPZ_DATA" * 1000)
    return npz_path


# =============================================================================
# Test get_node_id()
# =============================================================================

class TestGetNodeId:
    """Tests for get_node_id() function."""

    def test_returns_env_var_if_set(self, monkeypatch):
        """Test returns RINGRIFT_NODE_ID environment variable."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node-123")
        assert get_node_id() == "test-node-123"

    def test_returns_hostname_if_no_env(self, monkeypatch):
        """Test returns hostname if RINGRIFT_NODE_ID not set."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        monkeypatch.delenv("RINGRIFT_NODE_ID", raising=False)

        with patch("socket.gethostname", return_value="my-host"):
            assert get_node_id() == "my-host"

    def test_strips_common_prefixes(self, monkeypatch):
        """Test strips common prefixes like ip-, instance-, node-."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        monkeypatch.delenv("RINGRIFT_NODE_ID", raising=False)

        for prefix in ["ip-", "instance-", "node-"]:
            with patch("socket.gethostname", return_value=f"{prefix}10-0-0-1"):
                result = get_node_id()
                assert not result.startswith(prefix)
                assert result == "10-0-0-1"


# =============================================================================
# Test S3NodeSyncConfig
# =============================================================================

class TestS3NodeSyncConfig:
    """Tests for S3NodeSyncConfig dataclass."""

    def test_default_values(self, monkeypatch):
        """Test default configuration values."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        # Clear relevant env vars
        for key in ["RINGRIFT_S3_BUCKET", "RINGRIFT_S3_SYNC_INTERVAL"]:
            monkeypatch.delenv(key, raising=False)

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
        assert config.upload_timeout_seconds == 600.0
        assert config.download_timeout_seconds == 300.0

    def test_env_var_overrides(self, monkeypatch):
        """Test configuration can be overridden via environment variables."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        monkeypatch.setenv("RINGRIFT_S3_BUCKET", "my-custom-bucket")
        monkeypatch.setenv("RINGRIFT_S3_SYNC_INTERVAL", "1800")
        monkeypatch.setenv("RINGRIFT_S3_PUSH_GAMES", "false")
        monkeypatch.setenv("RINGRIFT_S3_BANDWIDTH_LIMIT", "1000")

        config = S3NodeSyncConfig()

        assert config.s3_bucket == "my-custom-bucket"
        assert config.sync_interval_seconds == 1800.0
        assert config.push_games is False
        assert config.bandwidth_limit_kbps == 1000

    def test_path_defaults(self, monkeypatch):
        """Test default paths."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        for key in ["RINGRIFT_GAMES_DIR", "RINGRIFT_MODELS_DIR", "RINGRIFT_NPZ_DIR"]:
            monkeypatch.delenv(key, raising=False)

        config = S3NodeSyncConfig()

        assert config.games_dir == Path("data/games")
        assert config.models_dir == Path("models")
        assert config.npz_dir == Path("data/training")

    def test_path_env_overrides(self, monkeypatch):
        """Test path configuration via environment variables."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        monkeypatch.setenv("RINGRIFT_GAMES_DIR", "/custom/games")
        monkeypatch.setenv("RINGRIFT_MODELS_DIR", "/custom/models")
        monkeypatch.setenv("RINGRIFT_NPZ_DIR", "/custom/npz")

        config = S3NodeSyncConfig()

        assert config.games_dir == Path("/custom/games")
        assert config.models_dir == Path("/custom/models")
        assert config.npz_dir == Path("/custom/npz")


# =============================================================================
# Test SyncResult
# =============================================================================

class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_default_values(self):
        """Test default SyncResult values."""
        from app.coordination.s3_node_sync_daemon import SyncResult

        result = SyncResult(success=True)

        assert result.success is True
        assert result.uploaded_files == []
        assert result.downloaded_files == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.bytes_transferred == 0

    def test_with_data(self):
        """Test SyncResult with populated data."""
        from app.coordination.s3_node_sync_daemon import SyncResult

        result = SyncResult(
            success=True,
            uploaded_files=["file1.db", "file2.db"],
            downloaded_files=["model.pth"],
            errors=["warning: slow transfer"],
            duration_seconds=45.5,
            bytes_transferred=1024000,
        )

        assert len(result.uploaded_files) == 2
        assert len(result.downloaded_files) == 1
        assert len(result.errors) == 1
        assert result.bytes_transferred == 1024000


# =============================================================================
# Test FileManifest
# =============================================================================

class TestFileManifest:
    """Tests for FileManifest dataclass."""

    def test_default_values(self):
        """Test default FileManifest values."""
        from app.coordination.s3_node_sync_daemon import FileManifest

        manifest = FileManifest(node_id="test-node", timestamp=time.time())

        assert manifest.node_id == "test-node"
        assert manifest.files == {}

    def test_with_files(self):
        """Test FileManifest with file entries."""
        from app.coordination.s3_node_sync_daemon import FileManifest

        manifest = FileManifest(
            node_id="test-node",
            timestamp=1234567890.0,
            files={
                "games/test.db": {"size": 1000, "mtime": 1234567890.0, "type": "database"},
                "models/model.pth": {"size": 5000, "mtime": 1234567890.0, "type": "model"},
            },
        )

        assert len(manifest.files) == 2
        assert manifest.files["games/test.db"]["type"] == "database"
        assert manifest.files["models/model.pth"]["size"] == 5000


# =============================================================================
# Test S3NodeSyncDaemon - Initialization
# =============================================================================

class TestS3NodeSyncDaemonInit:
    """Tests for S3NodeSyncDaemon initialization."""

    def test_default_initialization(self, monkeypatch):
        """Test daemon initializes with default config."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()

        assert daemon.node_id == "test-node"
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._push_count == 0
        assert daemon._errors == 0

    def test_custom_config(self, monkeypatch):
        """Test daemon with custom configuration."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "custom-node")

        config = S3NodeSyncConfig(
            s3_bucket="custom-bucket",
            sync_interval_seconds=600.0,
        )
        daemon = S3NodeSyncDaemon(config)

        assert daemon.config.s3_bucket == "custom-bucket"
        assert daemon.config.sync_interval_seconds == 600.0

    def test_name_property(self, monkeypatch):
        """Test daemon name property includes node_id."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "worker-42")

        daemon = S3NodeSyncDaemon()

        assert daemon.name == "s3_node_sync_worker-42"


# =============================================================================
# Test S3NodeSyncDaemon - Health Check
# =============================================================================

class TestS3NodeSyncDaemonHealthCheck:
    """Tests for S3NodeSyncDaemon health check."""

    def test_health_check_when_not_running(self, monkeypatch):
        """Test health check returns stopped status when not running."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        health = daemon.health_check()

        assert health.healthy is True  # Not running is OK
        assert "not running" in health.message.lower()

    def test_health_check_when_running_and_healthy(self, monkeypatch):
        """Test health check when daemon is running and healthy."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True
        daemon._last_push_time = time.time() - 60  # Recent push
        daemon._push_count = 5
        daemon._errors = 0

        health = daemon.health_check()

        assert health.healthy is True
        assert "pushes" in health.message.lower() or "healthy" in health.message.lower()

    def test_health_check_degraded_when_stale(self, monkeypatch):
        """Test health check returns degraded when push is stale."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(sync_interval_seconds=3600.0)
        daemon = S3NodeSyncDaemon(config)
        daemon._running = True
        daemon._last_push_time = time.time() - 10000  # Way past 2x interval

        health = daemon.health_check()

        assert health.healthy is False
        assert "no push" in health.message.lower()

    def test_health_check_includes_details(self, monkeypatch):
        """Test health check includes relevant details."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True
        daemon._last_push_time = time.time()
        daemon._bytes_uploaded = 1024000
        daemon._bytes_downloaded = 512000

        health = daemon.health_check()

        assert health.details is not None
        assert health.details.get("bytes_uploaded") == 1024000
        assert health.details.get("bytes_downloaded") == 512000


# =============================================================================
# Test S3NodeSyncDaemon - Manifest Building
# =============================================================================

class TestS3NodeSyncDaemonManifest:
    """Tests for manifest building functionality."""

    @pytest.mark.asyncio
    async def test_build_local_manifest_empty_dirs(self, temp_dirs, monkeypatch):
        """Test building manifest with empty directories."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(
            games_dir=temp_dirs["games"],
            models_dir=temp_dirs["models"],
            npz_dir=temp_dirs["npz"],
        )
        daemon = S3NodeSyncDaemon(config)

        manifest = await daemon._build_local_manifest()

        assert manifest.node_id == "test-node"
        assert len(manifest.files) == 0

    @pytest.mark.asyncio
    async def test_build_local_manifest_with_files(
        self, temp_dirs, sample_db_file, sample_model_file, sample_npz_file, monkeypatch
    ):
        """Test building manifest with actual files."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(
            games_dir=temp_dirs["games"],
            models_dir=temp_dirs["models"],
            npz_dir=temp_dirs["npz"],
        )
        daemon = S3NodeSyncDaemon(config)

        manifest = await daemon._build_local_manifest()

        assert len(manifest.files) == 3
        assert "games/canonical_hex8_2p.db" in manifest.files
        assert "models/canonical_hex8_2p.pth" in manifest.files
        assert "training/hex8_2p.npz" in manifest.files

    @pytest.mark.asyncio
    async def test_manifest_file_metadata(
        self, temp_dirs, sample_db_file, monkeypatch
    ):
        """Test manifest includes correct file metadata."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(
            games_dir=temp_dirs["games"],
            models_dir=temp_dirs["models"],
            npz_dir=temp_dirs["npz"],
        )
        daemon = S3NodeSyncDaemon(config)

        manifest = await daemon._build_local_manifest()

        db_entry = manifest.files.get("games/canonical_hex8_2p.db", {})
        assert db_entry.get("size") == 50000
        assert db_entry.get("type") == "database"
        assert "mtime" in db_entry

    @pytest.mark.asyncio
    async def test_manifest_skips_symlinks(self, temp_dirs, sample_model_file, monkeypatch):
        """Test manifest skips symlink files."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        # Create a symlink
        symlink_path = temp_dirs["models"] / "ringrift_best_hex8_2p.pth"
        symlink_path.symlink_to(sample_model_file)

        config = S3NodeSyncConfig(
            games_dir=temp_dirs["games"],
            models_dir=temp_dirs["models"],
            npz_dir=temp_dirs["npz"],
        )
        daemon = S3NodeSyncDaemon(config)

        manifest = await daemon._build_local_manifest()

        # Should only have the actual file, not the symlink
        model_files = [k for k in manifest.files if "models/" in k]
        assert len(model_files) == 1
        assert "canonical_hex8_2p.pth" in model_files[0]


# =============================================================================
# Test S3NodeSyncDaemon - S3 Operations
# =============================================================================

class TestS3NodeSyncDaemonS3Ops:
    """Tests for S3 operations (mocked)."""

    @pytest.mark.asyncio
    async def test_should_upload_returns_true_for_new_file(self, temp_dirs, monkeypatch):
        """Test _should_upload returns True when file not in S3."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(games_dir=temp_dirs["games"])
        daemon = S3NodeSyncDaemon(config)

        # Create a test file
        test_file = temp_dirs["games"] / "test.db"
        test_file.write_bytes(b"test data")

        # Mock S3 head-object to return 404 (file not found)
        mock_process = AsyncMock()
        mock_process.returncode = 1  # Non-zero = file not found
        mock_process.communicate = AsyncMock(return_value=(b"", b"Not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            should = await daemon._should_upload(test_file, "nodes/test/test.db")

        assert should is True

    @pytest.mark.asyncio
    async def test_should_upload_returns_false_for_same_size(self, temp_dirs, monkeypatch):
        """Test _should_upload returns False when S3 file has same size."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(games_dir=temp_dirs["games"])
        daemon = S3NodeSyncDaemon(config)

        # Create a test file
        test_file = temp_dirs["games"] / "test.db"
        test_file.write_bytes(b"test data")  # 9 bytes

        # Mock S3 head-object to return same size
        response = json.dumps({"ContentLength": 9})
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(response.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            should = await daemon._should_upload(test_file, "nodes/test/test.db")

        assert should is False

    @pytest.mark.asyncio
    async def test_should_upload_returns_true_for_different_size(self, temp_dirs, monkeypatch):
        """Test _should_upload returns True when S3 file has different size."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(games_dir=temp_dirs["games"])
        daemon = S3NodeSyncDaemon(config)

        # Create a test file
        test_file = temp_dirs["games"] / "test.db"
        test_file.write_bytes(b"test data updated")  # Different size

        # Mock S3 head-object to return different size
        response = json.dumps({"ContentLength": 9})
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(response.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            should = await daemon._should_upload(test_file, "nodes/test/test.db")

        assert should is True

    @pytest.mark.asyncio
    async def test_s3_upload_success(self, temp_dirs, monkeypatch):
        """Test successful S3 upload."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig()
        daemon = S3NodeSyncDaemon(config)

        # Create a test file
        test_file = temp_dirs["games"] / "test.db"
        test_file.write_bytes(b"test data")

        # Mock successful upload
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload(str(test_file), "nodes/test/test.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_upload_failure(self, temp_dirs, monkeypatch):
        """Test S3 upload failure handling."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig()
        daemon = S3NodeSyncDaemon(config)

        # Mock failed upload
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Access Denied"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload("/tmp/test.db", "nodes/test/test.db")

        assert result is False

    @pytest.mark.asyncio
    async def test_s3_upload_timeout(self, temp_dirs, monkeypatch):
        """Test S3 upload timeout handling."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(upload_timeout_seconds=0.1)
        daemon = S3NodeSyncDaemon(config)

        # Mock slow upload that times out
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.kill = MagicMock()

        async def slow_communicate():
            await asyncio.sleep(10)
            return (b"", b"")

        mock_process.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload("/tmp/test.db", "nodes/test/test.db")

        assert result is False
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_download_success(self, temp_dirs, monkeypatch):
        """Test successful S3 download."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig()
        daemon = S3NodeSyncDaemon(config)

        # Mock successful download
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("consolidated/test.npz", "/tmp/test.npz")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_download_not_found(self, temp_dirs, monkeypatch):
        """Test S3 download when file not found."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig()
        daemon = S3NodeSyncDaemon(config)

        # Mock 404 response
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"404 NoSuchKey"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("consolidated/missing.npz", "/tmp/missing.npz")

        assert result is False


# =============================================================================
# Test S3NodeSyncDaemon - Push Operations
# =============================================================================

class TestS3NodeSyncDaemonPush:
    """Tests for push operations."""

    @pytest.mark.asyncio
    async def test_push_games_skips_small_files(self, temp_dirs, monkeypatch):
        """Test push_games skips very small database files."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        # Create a small file (< 10KB)
        small_db = temp_dirs["games"] / "small.db"
        small_db.write_bytes(b"tiny")

        config = S3NodeSyncConfig(games_dir=temp_dirs["games"])
        daemon = S3NodeSyncDaemon(config)

        # Mock S3 operations
        with patch.object(daemon, "_should_upload", new_callable=AsyncMock) as mock_should:
            mock_should.return_value = True
            result = await daemon._push_games()

        # Small file should be skipped, so _should_upload never called
        assert mock_should.call_count == 0
        assert len(result.uploaded_files) == 0

    @pytest.mark.asyncio
    async def test_push_games_uploads_large_files(
        self, temp_dirs, sample_db_file, monkeypatch
    ):
        """Test push_games uploads files above size threshold."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(games_dir=temp_dirs["games"])
        daemon = S3NodeSyncDaemon(config)

        # Mock S3 operations
        with patch.object(daemon, "_should_upload", new_callable=AsyncMock) as mock_should:
            mock_should.return_value = True
            with patch.object(daemon, "_s3_upload", new_callable=AsyncMock) as mock_upload:
                mock_upload.return_value = True
                result = await daemon._push_games()

        assert mock_should.call_count == 1
        assert mock_upload.call_count == 1
        assert "canonical_hex8_2p.db" in result.uploaded_files

    @pytest.mark.asyncio
    async def test_push_models_only_canonical(
        self, temp_dirs, sample_model_file, monkeypatch
    ):
        """Test push_models only uploads canonical models."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        # Create a non-canonical model
        non_canonical = temp_dirs["models"] / "checkpoint_epoch10.pth"
        non_canonical.write_bytes(b"checkpoint data")

        config = S3NodeSyncConfig(models_dir=temp_dirs["models"])
        daemon = S3NodeSyncDaemon(config)

        # Mock S3 operations
        with patch.object(daemon, "_should_upload", new_callable=AsyncMock) as mock_should:
            mock_should.return_value = True
            with patch.object(daemon, "_s3_upload", new_callable=AsyncMock) as mock_upload:
                mock_upload.return_value = True
                result = await daemon._push_models()

        # Only canonical model should be uploaded
        assert mock_upload.call_count == 1
        assert "canonical_hex8_2p.pth" in result.uploaded_files
        assert "checkpoint_epoch10.pth" not in result.uploaded_files


# =============================================================================
# Test S3NodeSyncDaemon - Pull Operations
# =============================================================================

class TestS3NodeSyncDaemonPull:
    """Tests for pull operations."""

    @pytest.mark.asyncio
    async def test_pull_training_data_success(self, temp_dirs, monkeypatch):
        """Test pulling training data from S3."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(npz_dir=temp_dirs["npz"])
        daemon = S3NodeSyncDaemon(config)

        # Create the file after "download"
        async def fake_download(s3_path, local_path):
            Path(local_path).write_bytes(b"NPZ data")
            return True

        with patch.object(daemon, "_s3_download", new_callable=AsyncMock) as mock_download:
            mock_download.side_effect = fake_download
            result = await daemon.pull_training_data("hex8_2p")

        assert result.success is True
        assert "hex8_2p.npz" in result.downloaded_files

    @pytest.mark.asyncio
    async def test_pull_training_data_not_found(self, temp_dirs, monkeypatch):
        """Test pulling training data when not in S3."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(npz_dir=temp_dirs["npz"])
        daemon = S3NodeSyncDaemon(config)

        with patch.object(daemon, "_s3_download", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = False
            result = await daemon.pull_training_data("missing_config")

        assert len(result.downloaded_files) == 0

    @pytest.mark.asyncio
    async def test_pull_model_success(self, temp_dirs, monkeypatch):
        """Test pulling a model from S3."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(models_dir=temp_dirs["models"])
        daemon = S3NodeSyncDaemon(config)

        # Create the file after "download"
        async def fake_download(s3_path, local_path):
            Path(local_path).write_bytes(b"MODEL data")
            return True

        with patch.object(daemon, "_s3_download", new_callable=AsyncMock) as mock_download:
            mock_download.side_effect = fake_download
            result = await daemon.pull_model("canonical_hex8_2p.pth")

        assert result.success is True
        assert "canonical_hex8_2p.pth" in result.downloaded_files


# =============================================================================
# Test S3NodeSyncDaemon - Event Handling
# =============================================================================

class TestS3NodeSyncDaemonEvents:
    """Tests for event handling."""

    def test_on_training_completed_triggers_push(self, monkeypatch):
        """Test TRAINING_COMPLETED event triggers model push."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        event = {"config_key": "hex8_2p", "model_path": "models/canonical_hex8_2p.pth"}

        def schedule_ok(coro, *, context):
            coro.close()
            return True

        with patch.object(daemon, "_schedule_event_task", side_effect=schedule_ok) as mock_schedule:
            daemon._on_training_completed(event)
            mock_schedule.assert_called_once()

    def test_on_selfplay_complete_skips_small_batches(self, monkeypatch):
        """Test SELFPLAY_COMPLETE skips batches < 100 games."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        event = {"config_key": "hex8_2p", "games_count": 50}

        with patch.object(daemon, "_schedule_event_task") as mock_task:
            daemon._on_selfplay_complete(event)
            # Should NOT create task for small batch
            mock_task.assert_not_called()

    def test_on_selfplay_complete_triggers_for_large_batches(self, monkeypatch):
        """Test SELFPLAY_COMPLETE triggers push for >= 100 games."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        event = {"config_key": "hex8_2p", "games_count": 150}

        def schedule_ok(coro, *, context):
            coro.close()
            return True

        with patch.object(daemon, "_schedule_event_task", side_effect=schedule_ok) as mock_task:
            daemon._on_selfplay_complete(event)
            mock_task.assert_called_once()

    def test_on_model_promoted_triggers_push(self, monkeypatch):
        """Test MODEL_PROMOTED event triggers high-priority push."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        event = {"config_key": "hex8_2p", "model_path": "models/canonical_hex8_2p.pth"}

        def schedule_ok(coro, *, context):
            coro.close()
            return True

        with patch.object(daemon, "_schedule_event_task", side_effect=schedule_ok) as mock_task:
            daemon._on_model_promoted(event)
            mock_task.assert_called_once()

    def test_event_handlers_increment_errors_on_exception(self, monkeypatch):
        """Test event handlers increment error count on exception."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True
        daemon._errors = 0

        with patch.object(daemon, "_safe_create_task", side_effect=Exception("test error")):
            daemon._on_training_completed({"config_key": "test"})

        # Error should be incremented
        assert daemon._errors == 1


# =============================================================================
# Test S3NodeSyncDaemon - Lifecycle
# =============================================================================

class TestS3NodeSyncDaemonLifecycle:
    """Tests for daemon lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, monkeypatch):
        """Test start() sets running flag."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig, S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        config = S3NodeSyncConfig(sync_interval_seconds=0.1)
        daemon = S3NodeSyncDaemon(config)

        # Mock the sync operations
        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            with patch.object(daemon, "_subscribe_all_events"):
                # Start in background and stop quickly
                task = asyncio.create_task(daemon.start())
                await asyncio.sleep(0.05)

                assert daemon._running is True
                assert daemon._stats.started_at > 0

                # Stop the daemon
                daemon._running = False
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_stop_runs_final_push(self, monkeypatch):
        """Test stop() runs a final push cycle."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock) as mock_push:
            await daemon.stop()
            mock_push.assert_called_once()
            assert daemon._running is False

    def test_is_running(self, monkeypatch):
        """Test is_running property returns correct state."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()

        assert daemon.is_running is False
        daemon._running = True
        assert daemon.is_running is True


# =============================================================================
# Test S3ConsolidationDaemon
# =============================================================================

class TestS3ConsolidationDaemon:
    """Tests for S3ConsolidationDaemon."""

    def test_initialization(self):
        """Test consolidation daemon initialization."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()

        assert daemon._running is False
        assert daemon._consolidation_interval == 3600.0
        assert daemon._consolidation_errors == 0

    def test_health_check_not_running(self):
        """Test health check when not running."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        health = daemon.health_check()

        assert health.healthy is False
        assert "stopped" in health.message.lower()

    def test_health_check_running(self):
        """Test health check when running."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        daemon._running = True
        daemon._last_consolidation_time = time.time()

        health = daemon.health_check()

        assert health.healthy is True
        assert "operational" in health.message.lower()

    def test_health_check_stale(self):
        """Test health check when consolidation is stale."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        daemon._running = True
        daemon._last_consolidation_time = time.time() - 10000  # Very old

        health = daemon.health_check()

        assert health.healthy is False
        assert "stale" in health.message.lower()

    def test_health_check_excessive_errors(self):
        """Test health check with excessive errors."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        daemon._running = True
        daemon._last_consolidation_time = time.time()
        daemon._consolidation_errors = 10

        health = daemon.health_check()

        assert health.healthy is False
        assert "error" in health.message.lower()

    def test_health_check_includes_metrics(self):
        """Test health check includes consolidation metrics."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        daemon._running = True
        daemon._last_consolidation_time = time.time()
        daemon._nodes_consolidated = 5
        daemon._models_consolidated = 12
        daemon._npz_consolidated = 8

        health = daemon.health_check()

        assert health.details is not None
        assert health.details.get("nodes_consolidated") == 5
        assert health.details.get("models_consolidated") == 12
        assert health.details.get("npz_consolidated") == 8


# =============================================================================
# Test S3ConsolidationDaemon - Consolidation Operations
# =============================================================================

class TestS3ConsolidationDaemonOps:
    """Tests for consolidation operations."""

    @pytest.mark.asyncio
    async def test_consolidate_models_keeps_latest(self):
        """Test consolidate_models keeps latest version of each model."""
        from app.coordination.s3_node_sync_daemon import (
            FileManifest,
            S3ConsolidationDaemon,
        )

        daemon = S3ConsolidationDaemon()

        # Create manifests with different versions
        manifests = {
            "node-1": FileManifest(
                node_id="node-1",
                timestamp=time.time(),
                files={
                    "models/canonical_hex8_2p.pth": {
                        "size": 1000,
                        "mtime": 1000.0,  # Older
                        "type": "model",
                    },
                },
            ),
            "node-2": FileManifest(
                node_id="node-2",
                timestamp=time.time(),
                files={
                    "models/canonical_hex8_2p.pth": {
                        "size": 1100,
                        "mtime": 2000.0,  # Newer
                        "type": "model",
                    },
                },
            ),
        }

        copy_calls = []

        async def track_copy(src, dst):
            copy_calls.append((src, dst))
            return True, "copied"

        with patch.object(daemon, "_s3_copy", new_callable=AsyncMock) as mock_copy:
            mock_copy.side_effect = track_copy
            await daemon._consolidate_models(manifests)

        # Should copy from node-2 (newer)
        assert len(copy_calls) == 1
        assert "node-2" in copy_calls[0][0]

    @pytest.mark.asyncio
    async def test_consolidate_npz_keeps_latest(self):
        """Test consolidate_npz keeps latest version of each NPZ."""
        from app.coordination.s3_node_sync_daemon import (
            FileManifest,
            S3ConsolidationDaemon,
        )

        daemon = S3ConsolidationDaemon()

        manifests = {
            "node-1": FileManifest(
                node_id="node-1",
                timestamp=time.time(),
                files={
                    "training/hex8_2p.npz": {
                        "size": 5000,
                        "mtime": 3000.0,  # Newest
                        "type": "npz",
                    },
                },
            ),
            "node-2": FileManifest(
                node_id="node-2",
                timestamp=time.time(),
                files={
                    "training/hex8_2p.npz": {
                        "size": 4000,
                        "mtime": 2000.0,  # Older
                        "type": "npz",
                    },
                },
            ),
        }

        with patch.object(daemon, "_s3_copy", new_callable=AsyncMock) as mock_copy:
            mock_copy.return_value = (True, "copied")
            await daemon._consolidate_npz(manifests)

        # Should copy from node-1 (newer)
        assert mock_copy.call_count == 1
        call_args = mock_copy.call_args[0]
        assert "node-1" in call_args[0]


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_ensure_training_data_from_s3_uses_local(self, temp_dirs, monkeypatch):
        """Test ensure_training_data_from_s3 uses local file if exists."""
        from app.coordination.s3_node_sync_daemon import ensure_training_data_from_s3

        monkeypatch.setenv("RINGRIFT_NPZ_DIR", str(temp_dirs["npz"]))

        # Create local file
        local_npz = temp_dirs["npz"] / "hex8_2p.npz"
        local_npz.write_bytes(b"NPZ data")

        result = await ensure_training_data_from_s3("hex8_2p")

        assert result is True

    def test_sync_ensure_training_data_wrapper(self, temp_dirs, monkeypatch):
        """Test sync wrapper for ensure_training_data_from_s3."""
        from app.coordination.s3_node_sync_daemon import sync_ensure_training_data_from_s3

        monkeypatch.setenv("RINGRIFT_NPZ_DIR", str(temp_dirs["npz"]))

        # Create local file
        local_npz = temp_dirs["npz"] / "hex8_2p.npz"
        local_npz.write_bytes(b"NPZ data")

        result = sync_ensure_training_data_from_s3("hex8_2p")

        assert result is True


# =============================================================================
# Test List All Node Data
# =============================================================================

class TestListAllNodeData:
    """Tests for list_all_node_data functionality."""

    @pytest.mark.asyncio
    async def test_list_all_node_data_parses_nodes(self, monkeypatch):
        """Test list_all_node_data parses S3 ls output."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()

        # Mock S3 ls output
        ls_output = b"PRE node-1/\nPRE node-2/\nPRE node-3/\n"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(ls_output, b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(daemon, "_get_node_manifest", new_callable=AsyncMock) as mock_manifest:
                from app.coordination.s3_node_sync_daemon import FileManifest
                mock_manifest.return_value = FileManifest(
                    node_id="node-1",
                    timestamp=time.time(),
                )

                manifests = await daemon.list_all_node_data()

        # Should have attempted to get manifest for each node
        assert mock_manifest.call_count == 3

    @pytest.mark.asyncio
    async def test_list_all_node_data_handles_empty(self, monkeypatch):
        """Test list_all_node_data handles empty bucket."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        monkeypatch.setenv("RINGRIFT_NODE_ID", "test-node")

        daemon = S3NodeSyncDaemon()

        mock_process = AsyncMock()
        mock_process.returncode = 1  # No nodes found
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            manifests = await daemon.list_all_node_data()

        assert manifests == {}
