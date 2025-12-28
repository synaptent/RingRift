"""Tests for S3 Node Sync Daemon.

This module tests the S3 backup functionality that is production-critical
for cluster data persistence.

December 2025: Created as part of Phase 3A comprehensive improvement plan.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ==============================================================================
# Test Imports
# ==============================================================================


class TestModuleImports:
    """Test that module imports correctly."""

    def test_import_s3_node_sync_daemon(self):
        """Test that main daemon class can be imported."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        assert S3NodeSyncDaemon is not None

    def test_import_s3_consolidation_daemon(self):
        """Test that consolidation daemon can be imported."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        assert S3ConsolidationDaemon is not None

    def test_import_config_class(self):
        """Test that config class can be imported."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        assert S3NodeSyncConfig is not None

    def test_import_dataclasses(self):
        """Test that dataclasses can be imported."""
        from app.coordination.s3_node_sync_daemon import (
            FileManifest,
            SyncResult,
        )

        assert SyncResult is not None
        assert FileManifest is not None

    def test_import_utility_functions(self):
        """Test that utility functions can be imported."""
        from app.coordination.s3_node_sync_daemon import (
            ensure_training_data_from_s3,
            get_node_id,
            sync_ensure_training_data_from_s3,
        )

        assert get_node_id is not None
        assert ensure_training_data_from_s3 is not None
        assert sync_ensure_training_data_from_s3 is not None


# ==============================================================================
# Test get_node_id()
# ==============================================================================


class TestGetNodeId:
    """Tests for get_node_id() function."""

    def test_returns_env_var_when_set(self):
        """Test that RINGRIFT_NODE_ID environment variable is used."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        with patch.dict(os.environ, {"RINGRIFT_NODE_ID": "test-node-123"}):
            result = get_node_id()

        assert result == "test-node-123"

    def test_falls_back_to_hostname(self):
        """Test fallback to hostname when env var not set."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        with patch.dict(os.environ, {}, clear=True):
            # Remove RINGRIFT_NODE_ID if present
            os.environ.pop("RINGRIFT_NODE_ID", None)

            with patch("socket.gethostname", return_value="my-host"):
                result = get_node_id()

        assert result == "my-host"

    def test_strips_ip_prefix(self):
        """Test that ip- prefix is stripped from hostname."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_NODE_ID", None)

            with patch("socket.gethostname", return_value="ip-192-168-1-1"):
                result = get_node_id()

        assert result == "192-168-1-1"

    def test_strips_instance_prefix(self):
        """Test that instance- prefix is stripped from hostname."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_NODE_ID", None)

            with patch("socket.gethostname", return_value="instance-abc123"):
                result = get_node_id()

        assert result == "abc123"

    def test_strips_node_prefix(self):
        """Test that node- prefix is stripped from hostname."""
        from app.coordination.s3_node_sync_daemon import get_node_id

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_NODE_ID", None)

            with patch("socket.gethostname", return_value="node-worker-5"):
                result = get_node_id()

        assert result == "worker-5"


# ==============================================================================
# Test S3NodeSyncConfig
# ==============================================================================


class TestS3NodeSyncConfig:
    """Tests for S3NodeSyncConfig dataclass."""

    def test_default_values(self):
        """Test that config has sensible defaults."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        # Clear env vars that might affect defaults
        with patch.dict(os.environ, {}, clear=True):
            config = S3NodeSyncConfig()

        assert config.s3_bucket == "ringrift-models-20251214"
        assert config.aws_region == "us-east-1"
        assert config.sync_interval_seconds == 3600.0  # 1 hour
        assert config.push_games is True
        assert config.push_models is True
        assert config.push_npz is True
        assert config.pull_npz is True
        assert config.pull_models is True
        assert config.compress_uploads is True
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 30.0
        assert config.upload_timeout_seconds == 600.0
        assert config.download_timeout_seconds == 300.0

    def test_env_var_override_bucket(self):
        """Test RINGRIFT_S3_BUCKET environment variable."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        with patch.dict(os.environ, {"RINGRIFT_S3_BUCKET": "my-custom-bucket"}):
            config = S3NodeSyncConfig()

        assert config.s3_bucket == "my-custom-bucket"

    def test_env_var_override_region(self):
        """Test AWS_REGION environment variable."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}):
            config = S3NodeSyncConfig()

        assert config.aws_region == "eu-west-1"

    def test_env_var_override_sync_interval(self):
        """Test RINGRIFT_S3_SYNC_INTERVAL environment variable."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        with patch.dict(os.environ, {"RINGRIFT_S3_SYNC_INTERVAL": "1800"}):
            config = S3NodeSyncConfig()

        assert config.sync_interval_seconds == 1800.0

    def test_env_var_override_push_games_false(self):
        """Test RINGRIFT_S3_PUSH_GAMES=false environment variable."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        with patch.dict(os.environ, {"RINGRIFT_S3_PUSH_GAMES": "false"}):
            config = S3NodeSyncConfig()

        assert config.push_games is False

    def test_env_var_override_bandwidth_limit(self):
        """Test RINGRIFT_S3_BANDWIDTH_LIMIT environment variable."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        with patch.dict(os.environ, {"RINGRIFT_S3_BANDWIDTH_LIMIT": "1024"}):
            config = S3NodeSyncConfig()

        assert config.bandwidth_limit_kbps == 1024

    def test_path_defaults_are_paths(self):
        """Test that path defaults are Path objects."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

        config = S3NodeSyncConfig()

        assert isinstance(config.games_dir, Path)
        assert isinstance(config.models_dir, Path)
        assert isinstance(config.npz_dir, Path)


# ==============================================================================
# Test SyncResult
# ==============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_default_values(self):
        """Test default values for SyncResult."""
        from app.coordination.s3_node_sync_daemon import SyncResult

        result = SyncResult(success=True)

        assert result.success is True
        assert result.uploaded_files == []
        assert result.downloaded_files == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.bytes_transferred == 0

    def test_with_uploaded_files(self):
        """Test SyncResult with uploaded files."""
        from app.coordination.s3_node_sync_daemon import SyncResult

        result = SyncResult(
            success=True,
            uploaded_files=["game1.db", "game2.db"],
            bytes_transferred=1024 * 1024,
        )

        assert len(result.uploaded_files) == 2
        assert result.bytes_transferred == 1024 * 1024

    def test_with_errors(self):
        """Test SyncResult with errors."""
        from app.coordination.s3_node_sync_daemon import SyncResult

        result = SyncResult(
            success=False,
            errors=["Connection refused", "Timeout"],
        )

        assert result.success is False
        assert len(result.errors) == 2


# ==============================================================================
# Test FileManifest
# ==============================================================================


class TestFileManifest:
    """Tests for FileManifest dataclass."""

    def test_creation(self):
        """Test FileManifest creation."""
        from app.coordination.s3_node_sync_daemon import FileManifest

        manifest = FileManifest(
            node_id="test-node",
            timestamp=time.time(),
        )

        assert manifest.node_id == "test-node"
        assert manifest.files == {}

    def test_with_files(self):
        """Test FileManifest with files."""
        from app.coordination.s3_node_sync_daemon import FileManifest

        now = time.time()
        manifest = FileManifest(
            node_id="test-node",
            timestamp=now,
            files={
                "games/selfplay.db": {"size": 1024, "mtime": now, "type": "database"},
                "models/canonical.pth": {"size": 5000, "mtime": now, "type": "model"},
            },
        )

        assert len(manifest.files) == 2
        assert "games/selfplay.db" in manifest.files
        assert manifest.files["games/selfplay.db"]["type"] == "database"


# ==============================================================================
# Test S3NodeSyncDaemon
# ==============================================================================


class TestS3NodeSyncDaemon:
    """Tests for S3NodeSyncDaemon class."""

    def test_initialization(self):
        """Test daemon can be initialized."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        with patch.dict(os.environ, {"RINGRIFT_NODE_ID": "test-node"}):
            daemon = S3NodeSyncDaemon()

        assert daemon.node_id == "test-node"
        assert daemon.is_running() is False

    def test_initialization_with_config(self):
        """Test daemon with custom config."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        config = S3NodeSyncConfig(
            s3_bucket="test-bucket",
            sync_interval_seconds=300.0,
        )
        daemon = S3NodeSyncDaemon(config=config)

        assert daemon.config.s3_bucket == "test-bucket"
        assert daemon.config.sync_interval_seconds == 300.0

    def test_name_property(self):
        """Test daemon name property."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        with patch.dict(os.environ, {"RINGRIFT_NODE_ID": "my-node"}):
            daemon = S3NodeSyncDaemon()

        assert daemon.name == "S3NodeSyncDaemon-my-node"

    def test_is_running_default(self):
        """Test is_running returns False by default."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        assert daemon.is_running() is False

    def test_health_check_not_running(self):
        """Test health_check when daemon is not running."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        result = daemon.health_check()

        assert result.healthy is True
        assert "not running" in result.message.lower()

    def test_health_check_running_healthy(self):
        """Test health_check when daemon is running and healthy."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        daemon._running = True
        daemon._last_push_time = time.time()
        daemon._push_count = 5
        daemon._errors = 0

        result = daemon.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_degraded(self):
        """Test health_check when daemon is degraded (no recent push)."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        daemon._running = True
        # Set last push to be much older than 2x sync interval
        daemon._last_push_time = time.time() - (daemon.config.sync_interval_seconds * 3)
        daemon._push_count = 5
        daemon._errors = 2

        result = daemon.health_check()

        assert result.healthy is False
        assert "no push" in result.message.lower()


# ==============================================================================
# Test S3NodeSyncDaemon - Manifest Building
# ==============================================================================


class TestS3NodeSyncDaemonManifest:
    """Tests for S3NodeSyncDaemon manifest building."""

    @pytest.mark.asyncio
    async def test_build_local_manifest_empty_dirs(self):
        """Test manifest building with empty directories."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = S3NodeSyncConfig(
                games_dir=Path(tmpdir) / "games",
                models_dir=Path(tmpdir) / "models",
                npz_dir=Path(tmpdir) / "npz",
            )
            daemon = S3NodeSyncDaemon(config=config)

            manifest = await daemon._build_local_manifest()

        assert manifest.node_id == daemon.node_id
        assert manifest.files == {}

    @pytest.mark.asyncio
    async def test_build_local_manifest_with_files(self):
        """Test manifest building with actual files."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directories
            games_dir = Path(tmpdir) / "games"
            models_dir = Path(tmpdir) / "models"
            npz_dir = Path(tmpdir) / "npz"
            games_dir.mkdir()
            models_dir.mkdir()
            npz_dir.mkdir()

            # Create test files
            (games_dir / "selfplay.db").write_text("test")
            (models_dir / "canonical_hex8_2p.pth").write_text("test")
            (npz_dir / "hex8_2p.npz").write_text("test")

            config = S3NodeSyncConfig(
                games_dir=games_dir,
                models_dir=models_dir,
                npz_dir=npz_dir,
            )
            daemon = S3NodeSyncDaemon(config=config)

            manifest = await daemon._build_local_manifest()

        assert "games/selfplay.db" in manifest.files
        assert manifest.files["games/selfplay.db"]["type"] == "database"

        assert "models/canonical_hex8_2p.pth" in manifest.files
        assert manifest.files["models/canonical_hex8_2p.pth"]["type"] == "model"

        assert "training/hex8_2p.npz" in manifest.files
        assert manifest.files["training/hex8_2p.npz"]["type"] == "npz"

    @pytest.mark.asyncio
    async def test_build_local_manifest_excludes_symlinks(self):
        """Test that manifest excludes symlinks for models."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create real file and symlink
            real_file = models_dir / "canonical_hex8_2p.pth"
            real_file.write_text("test")
            symlink = models_dir / "ringrift_best_hex8_2p.pth"
            symlink.symlink_to(real_file)

            config = S3NodeSyncConfig(
                games_dir=Path(tmpdir) / "games",
                models_dir=models_dir,
                npz_dir=Path(tmpdir) / "npz",
            )
            daemon = S3NodeSyncDaemon(config=config)

            manifest = await daemon._build_local_manifest()

        # Real file should be included
        assert "models/canonical_hex8_2p.pth" in manifest.files
        # Symlink should be excluded
        assert "models/ringrift_best_hex8_2p.pth" not in manifest.files


# ==============================================================================
# Test S3NodeSyncDaemon - S3 Operations (Mocked)
# ==============================================================================


class TestS3NodeSyncDaemonS3Operations:
    """Tests for S3NodeSyncDaemon S3 operations with mocking."""

    @pytest.mark.asyncio
    async def test_should_upload_file_not_exists(self):
        """Test _should_upload returns True when file doesn't exist in S3."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(b"test")
            local_path = Path(f.name)

        try:
            # Mock AWS CLI returning error (file not found)
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate = AsyncMock(return_value=(b"", b"Not Found"))

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await daemon._should_upload(local_path, "nodes/test/games/test.db")

            assert result is True
        finally:
            local_path.unlink()

    @pytest.mark.asyncio
    async def test_should_upload_same_size(self):
        """Test _should_upload returns False when file has same size."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(b"test content")
            local_path = Path(f.name)

        try:
            # Mock AWS CLI returning same size
            response = json.dumps({"ContentLength": local_path.stat().st_size})
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(response.encode(), b""))

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await daemon._should_upload(local_path, "nodes/test/games/test.db")

            assert result is False
        finally:
            local_path.unlink()

    @pytest.mark.asyncio
    async def test_should_upload_different_size(self):
        """Test _should_upload returns True when file has different size."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(b"test content")
            local_path = Path(f.name)

        try:
            # Mock AWS CLI returning different size
            response = json.dumps({"ContentLength": 999})  # Different size
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(response.encode(), b""))

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await daemon._should_upload(local_path, "nodes/test/games/test.db")

            assert result is True
        finally:
            local_path.unlink()

    @pytest.mark.asyncio
    async def test_s3_upload_success(self):
        """Test _s3_upload returns True on success."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload("/tmp/test.db", "nodes/test/games/test.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_upload_failure(self):
        """Test _s3_upload returns False on failure."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Access Denied"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload("/tmp/test.db", "nodes/test/games/test.db")

        assert result is False

    @pytest.mark.asyncio
    async def test_s3_upload_timeout(self):
        """Test _s3_upload returns False on timeout."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        config = S3NodeSyncConfig(upload_timeout_seconds=0.1)
        daemon = S3NodeSyncDaemon(config=config)

        mock_process = AsyncMock()
        mock_process.kill = MagicMock()

        async def slow_communicate():
            await asyncio.sleep(1.0)
            return (b"", b"")

        mock_process.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_upload("/tmp/test.db", "nodes/test/games/test.db")

        assert result is False
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_s3_download_success(self):
        """Test _s3_download returns True on success."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("nodes/test/games/test.db", "/tmp/test.db")

        assert result is True

    @pytest.mark.asyncio
    async def test_s3_download_not_found(self):
        """Test _s3_download returns False when file not found."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"404 NoSuchKey"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await daemon._s3_download("nodes/test/games/test.db", "/tmp/test.db")

        assert result is False


# ==============================================================================
# Test S3NodeSyncDaemon - Push Operations (Mocked)
# ==============================================================================


class TestS3NodeSyncDaemonPush:
    """Tests for S3NodeSyncDaemon push operations."""

    @pytest.mark.asyncio
    async def test_push_games_empty_dir(self):
        """Test _push_games with empty directory."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = S3NodeSyncConfig(games_dir=Path(tmpdir) / "games")
            daemon = S3NodeSyncDaemon(config=config)

            result = await daemon._push_games()

        assert result.success is True
        assert result.uploaded_files == []

    @pytest.mark.asyncio
    async def test_push_games_skips_small_databases(self):
        """Test _push_games skips databases smaller than 10KB."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            games_dir = Path(tmpdir) / "games"
            games_dir.mkdir()

            # Create small database (< 10KB)
            (games_dir / "small.db").write_text("x" * 5000)

            config = S3NodeSyncConfig(games_dir=games_dir)
            daemon = S3NodeSyncDaemon(config=config)

            # Mock _should_upload and _s3_upload
            with patch.object(daemon, "_should_upload", return_value=True):
                with patch.object(daemon, "_s3_upload", return_value=True):
                    result = await daemon._push_games()

        # Small file should be skipped
        assert result.uploaded_files == []

    @pytest.mark.asyncio
    async def test_push_models_only_canonical(self):
        """Test _push_models only pushes canonical models."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create canonical and non-canonical models
            (models_dir / "canonical_hex8_2p.pth").write_bytes(b"x" * 1000)
            (models_dir / "best_model.pth").write_bytes(b"x" * 1000)

            config = S3NodeSyncConfig(models_dir=models_dir)
            daemon = S3NodeSyncDaemon(config=config)

            with patch.object(daemon, "_should_upload", return_value=True):
                with patch.object(daemon, "_s3_upload", return_value=True) as mock_upload:
                    result = await daemon._push_models()

        # Only canonical model should be uploaded
        assert len(result.uploaded_files) == 1
        assert "canonical_hex8_2p.pth" in result.uploaded_files


# ==============================================================================
# Test S3NodeSyncDaemon - Lifecycle
# ==============================================================================


class TestS3NodeSyncDaemonLifecycle:
    """Tests for S3NodeSyncDaemon lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        """Test that start() sets running flag."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        # Mock the push cycle to avoid actual S3 operations
        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            # Start in background
            task = asyncio.create_task(daemon.start())

            # Wait a bit for running flag to be set
            await asyncio.sleep(0.1)

            assert daemon.is_running() is True

            # Stop the daemon
            daemon._running = False
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_clears_running(self):
        """Test that stop() clears running flag."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        daemon._running = True

        # Mock the push cycle to avoid actual S3 operations
        with patch.object(daemon, "_run_push_cycle", new_callable=AsyncMock):
            await daemon.stop()

        assert daemon.is_running() is False


# ==============================================================================
# Test S3ConsolidationDaemon
# ==============================================================================


class TestS3ConsolidationDaemon:
    """Tests for S3ConsolidationDaemon class."""

    def test_initialization(self):
        """Test daemon can be initialized."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()

        assert daemon._running is False
        assert daemon._consolidation_interval == 3600.0

    def test_initialization_with_config(self):
        """Test daemon with custom config."""
        from app.coordination.s3_node_sync_daemon import (
            S3ConsolidationDaemon,
            S3NodeSyncConfig,
        )

        config = S3NodeSyncConfig(s3_bucket="custom-bucket")
        daemon = S3ConsolidationDaemon(config=config)

        assert daemon.config.s3_bucket == "custom-bucket"

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stop method clears running flag."""
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        daemon._running = True

        await daemon.stop()

        assert daemon._running is False


# ==============================================================================
# Test Convenience Functions
# ==============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_ensure_training_data_from_s3_local_exists(self):
        """Test ensure_training_data_from_s3 when local file exists."""
        from app.coordination.s3_node_sync_daemon import ensure_training_data_from_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_dir = Path(tmpdir)
            (npz_dir / "hex8_2p.npz").write_text("test")

            with patch.dict(os.environ, {"RINGRIFT_NPZ_DIR": str(npz_dir)}):
                # Re-import to pick up env var
                from importlib import reload

                import app.coordination.s3_node_sync_daemon as module

                reload(module)

                # The function should return True if local file exists
                # (without calling S3)
                with patch.object(
                    module.S3NodeSyncDaemon, "pull_training_data", new_callable=AsyncMock
                ) as mock_pull:
                    result = await module.ensure_training_data_from_s3("hex8_2p")

        assert result is True

    def test_sync_ensure_training_data_from_s3(self):
        """Test sync wrapper exists and is callable."""
        from app.coordination.s3_node_sync_daemon import sync_ensure_training_data_from_s3

        # Just verify it's a function
        assert callable(sync_ensure_training_data_from_s3)


# ==============================================================================
# Test Pull Operations
# ==============================================================================


class TestS3NodeSyncDaemonPull:
    """Tests for S3NodeSyncDaemon pull operations."""

    @pytest.mark.asyncio
    async def test_pull_training_data_success(self):
        """Test pull_training_data on success."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

            config = S3NodeSyncConfig(npz_dir=Path(tmpdir))
            daemon = S3NodeSyncDaemon(config=config)

            # Create the file that would be downloaded
            npz_path = Path(tmpdir) / "hex8_2p.npz"

            # Mock _s3_download to create the file
            async def mock_download(s3_path, local_path):
                Path(local_path).write_text("test npz content")
                return True

            with patch.object(daemon, "_s3_download", side_effect=mock_download):
                result = await daemon.pull_training_data("hex8_2p")

        assert result.success is True
        assert "hex8_2p.npz" in result.downloaded_files

    @pytest.mark.asyncio
    async def test_pull_training_data_not_found(self):
        """Test pull_training_data when file not found."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

            config = S3NodeSyncConfig(npz_dir=Path(tmpdir))
            daemon = S3NodeSyncDaemon(config=config)

            with patch.object(daemon, "_s3_download", return_value=False):
                result = await daemon.pull_training_data("nonexistent_config")

        assert result.downloaded_files == []

    @pytest.mark.asyncio
    async def test_pull_model_success(self):
        """Test pull_model on success."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        with tempfile.TemporaryDirectory() as tmpdir:
            from app.coordination.s3_node_sync_daemon import S3NodeSyncConfig

            config = S3NodeSyncConfig(models_dir=Path(tmpdir))
            daemon = S3NodeSyncDaemon(config=config)

            # Mock _s3_download to create the file
            async def mock_download(s3_path, local_path):
                Path(local_path).write_text("test model content")
                return True

            with patch.object(daemon, "_s3_download", side_effect=mock_download):
                result = await daemon.pull_model("canonical_hex8_2p.pth")

        assert "canonical_hex8_2p.pth" in result.downloaded_files


# ==============================================================================
# Test Statistics Tracking
# ==============================================================================


class TestS3NodeSyncDaemonStats:
    """Tests for S3NodeSyncDaemon statistics tracking."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()

        assert daemon._start_time == 0.0
        assert daemon._last_push_time == 0.0
        assert daemon._last_pull_time == 0.0
        assert daemon._push_count == 0
        assert daemon._pull_count == 0
        assert daemon._bytes_uploaded == 0
        assert daemon._bytes_downloaded == 0
        assert daemon._errors == 0

    @pytest.mark.asyncio
    async def test_push_cycle_updates_stats(self):
        """Test that push cycle updates statistics."""
        from app.coordination.s3_node_sync_daemon import (
            S3NodeSyncConfig,
            S3NodeSyncDaemon,
            SyncResult,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = S3NodeSyncConfig(
                games_dir=Path(tmpdir) / "games",
                models_dir=Path(tmpdir) / "models",
                npz_dir=Path(tmpdir) / "npz",
            )
            daemon = S3NodeSyncDaemon(config=config)

            # Mock the sub-operations to return known results
            with patch.object(
                daemon, "_build_local_manifest", new_callable=AsyncMock
            ) as mock_manifest:
                from app.coordination.s3_node_sync_daemon import FileManifest

                mock_manifest.return_value = FileManifest(
                    node_id="test", timestamp=time.time()
                )
                with patch.object(daemon, "_upload_manifest", new_callable=AsyncMock):
                    with patch.object(
                        daemon, "_push_games", new_callable=AsyncMock
                    ) as mock_games:
                        mock_games.return_value = SyncResult(
                            success=True,
                            uploaded_files=["game.db"],
                            bytes_transferred=1024,
                        )
                        with patch.object(
                            daemon, "_push_models", new_callable=AsyncMock
                        ) as mock_models:
                            mock_models.return_value = SyncResult(success=True)
                            with patch.object(
                                daemon, "_push_npz", new_callable=AsyncMock
                            ) as mock_npz:
                                mock_npz.return_value = SyncResult(success=True)

                                await daemon._run_push_cycle()

        assert daemon._push_count == 1
        assert daemon._bytes_uploaded == 1024
        assert daemon._last_push_time > 0
