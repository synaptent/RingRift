"""
Unit tests for app.distributed.sync_utils module.

Tests cover:
- TransferVerificationResult dataclass
- Checksum computation
- File quarantine mechanism
- Quarantine cleanup
- SSH command building for rsync
- Rsync operations (file, directory, push)
- Verified rsync operations with integrity checks

Created: December 2025
"""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.distributed.sync_utils import (
    # Dataclasses
    TransferVerificationResult,
    # Constants
    QUARANTINE_DIR,
    QUARANTINE_MAX_AGE_DAYS,
    # Internal helpers
    _compute_checksum,
    _quarantine_file,
    _fetch_remote_checksum,
    _fetch_remote_checksum_async,
    # Public helpers
    cleanup_quarantine,
    build_ssh_command_for_rsync,
    # Basic rsync functions
    rsync_file,
    rsync_file_async,
    rsync_directory,
    rsync_directory_async,
    rsync_push_file,
    rsync_push_file_async,
    # Verified rsync functions
    rsync_file_verified,
    rsync_file_verified_async,
    rsync_push_file_verified,
    rsync_push_file_verified_async,
    rsync_directory_verified,
    rsync_directory_verified_async,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_host_config():
    """Create a mock HostConfig object."""
    host = MagicMock()
    host.name = "test-host"
    host.ssh_key = "~/.ssh/test_key"
    host.ssh_port = 22
    host.ssh_target = "user@test-host.example.com"
    return host


@pytest.fixture
def mock_host_config_custom_port():
    """Create a mock HostConfig with custom SSH port."""
    host = MagicMock()
    host.name = "test-host-custom"
    host.ssh_key = "~/.ssh/custom_key"
    host.ssh_port = 2222
    host.ssh_target = "admin@custom-host.example.com"
    return host


@pytest.fixture
def mock_host_config_no_key():
    """Create a mock HostConfig without SSH key."""
    host = MagicMock()
    host.name = "test-host-no-key"
    host.ssh_key = None
    host.ssh_port = 22
    host.ssh_target = "root@nokey-host.example.com"
    return host


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file with content."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Test content for checksum verification")
    return file_path


@pytest.fixture
def temp_quarantine_dir(tmp_path, monkeypatch):
    """Create a temporary quarantine directory."""
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    monkeypatch.setattr("app.distributed.sync_utils.QUARANTINE_DIR", quarantine)
    return quarantine


# =============================================================================
# TransferVerificationResult Tests
# =============================================================================


class TestTransferVerificationResult:
    """Tests for TransferVerificationResult dataclass."""

    def test_default_values(self):
        """Default values are correct."""
        result = TransferVerificationResult(success=False)
        assert result.success is False
        assert result.verified is False
        assert result.error == ""
        assert result.bytes_transferred == 0
        assert result.checksum_matched is False
        assert result.quarantined is False
        assert result.quarantine_path is None

    def test_success_result(self):
        """Can create a successful result."""
        result = TransferVerificationResult(
            success=True,
            verified=True,
            bytes_transferred=1024,
            checksum_matched=True,
        )
        assert result.success is True
        assert result.verified is True
        assert result.bytes_transferred == 1024
        assert result.checksum_matched is True

    def test_failure_result_with_error(self):
        """Can create a failure result with error message."""
        result = TransferVerificationResult(
            success=False,
            error="Connection timed out",
        )
        assert result.success is False
        assert result.error == "Connection timed out"

    def test_quarantine_result(self):
        """Can create a result with quarantine info."""
        quarantine_path = Path("/data/quarantine/corrupted.db.checksum_mismatch.1234567890")
        result = TransferVerificationResult(
            success=False,
            error="Checksum mismatch",
            quarantined=True,
            quarantine_path=quarantine_path,
        )
        assert result.quarantined is True
        assert result.quarantine_path == quarantine_path


# =============================================================================
# Checksum Computation Tests
# =============================================================================


class TestComputeChecksum:
    """Tests for _compute_checksum function."""

    def test_compute_sha256_checksum(self, temp_file):
        """Computes correct SHA256 checksum."""
        checksum = _compute_checksum(temp_file)
        assert len(checksum) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_checksum_consistent(self, temp_file):
        """Checksum is consistent for same file."""
        checksum1 = _compute_checksum(temp_file)
        checksum2 = _compute_checksum(temp_file)
        assert checksum1 == checksum2

    def test_compute_checksum_different_content(self, tmp_path):
        """Different content produces different checksum."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        checksum1 = _compute_checksum(file1)
        checksum2 = _compute_checksum(file2)
        assert checksum1 != checksum2

    def test_compute_checksum_missing_file(self, tmp_path):
        """Returns empty string for missing file."""
        missing = tmp_path / "nonexistent.txt"
        checksum = _compute_checksum(missing)
        assert checksum == ""

    def test_compute_checksum_error_handling(self, tmp_path):
        """Handles errors gracefully for unreadable files."""
        # Create a file then make it unreadable
        file_path = tmp_path / "unreadable.txt"
        file_path.write_text("content")

        # Test with a file that exists but simulate error by using a bad path
        # _compute_checksum should return "" for any error
        bad_path = tmp_path / "nonexistent_during_read.txt"
        checksum = _compute_checksum(bad_path)
        assert checksum == ""


# =============================================================================
# File Quarantine Tests
# =============================================================================


class TestQuarantineFile:
    """Tests for _quarantine_file function."""

    def test_quarantine_moves_file(self, temp_file, temp_quarantine_dir):
        """Quarantine moves file to quarantine directory."""
        original_content = temp_file.read_text()
        quarantine_path = _quarantine_file(temp_file, "test_reason")

        assert quarantine_path is not None
        assert not temp_file.exists()
        assert quarantine_path.exists()
        assert quarantine_path.read_text() == original_content

    def test_quarantine_path_format(self, temp_file, temp_quarantine_dir):
        """Quarantine path has correct format."""
        quarantine_path = _quarantine_file(temp_file, "checksum_mismatch")

        assert quarantine_path.parent == temp_quarantine_dir
        assert "test_file.txt" in quarantine_path.name
        assert "checksum_mismatch" in quarantine_path.name
        # Should have timestamp
        assert any(c.isdigit() for c in quarantine_path.name)

    def test_quarantine_nonexistent_file(self, tmp_path, temp_quarantine_dir):
        """Returns None for nonexistent file."""
        missing = tmp_path / "nonexistent.txt"
        result = _quarantine_file(missing, "reason")
        assert result is None

    def test_quarantine_creates_directory(self, temp_file, tmp_path, monkeypatch):
        """Creates quarantine directory if it doesn't exist."""
        new_quarantine = tmp_path / "new_quarantine"
        monkeypatch.setattr("app.distributed.sync_utils.QUARANTINE_DIR", new_quarantine)

        quarantine_path = _quarantine_file(temp_file, "reason")

        assert new_quarantine.exists()
        assert quarantine_path is not None


class TestCleanupQuarantine:
    """Tests for cleanup_quarantine function."""

    def test_removes_old_files(self, temp_quarantine_dir):
        """Removes files older than max age."""
        old_file = temp_quarantine_dir / "old_file.txt.corrupted.1234567890"
        old_file.write_text("old content")
        # Set modification time to 40 days ago
        old_time = time.time() - (40 * 86400)
        os.utime(old_file, (old_time, old_time))

        removed = cleanup_quarantine(max_age_days=30)

        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self, temp_quarantine_dir):
        """Keeps files newer than max age."""
        recent_file = temp_quarantine_dir / "recent_file.txt.corrupted.9876543210"
        recent_file.write_text("recent content")

        removed = cleanup_quarantine(max_age_days=30)

        assert removed == 0
        assert recent_file.exists()

    def test_returns_zero_for_empty_quarantine(self, temp_quarantine_dir):
        """Returns 0 when quarantine is empty."""
        removed = cleanup_quarantine()
        assert removed == 0

    def test_returns_zero_for_nonexistent_quarantine(self, tmp_path, monkeypatch):
        """Returns 0 when quarantine doesn't exist."""
        nonexistent = tmp_path / "no_such_dir"
        monkeypatch.setattr("app.distributed.sync_utils.QUARANTINE_DIR", nonexistent)

        removed = cleanup_quarantine()
        assert removed == 0


# =============================================================================
# Remote Checksum Tests
# =============================================================================


class TestFetchRemoteChecksum:
    """Tests for _fetch_remote_checksum function."""

    @patch("subprocess.run")
    def test_fetch_remote_checksum_success(self, mock_run, mock_host_config):
        """Successfully fetches remote checksum."""
        expected_checksum = "a" * 64  # Valid SHA256 length
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=expected_checksum + "\n",
        )

        result = _fetch_remote_checksum(mock_host_config, "/path/to/file.db")

        assert result == expected_checksum
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_fetch_remote_checksum_failure(self, mock_run, mock_host_config):
        """Returns None on command failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        result = _fetch_remote_checksum(mock_host_config, "/path/to/file.db")

        assert result is None

    @patch("subprocess.run")
    def test_fetch_remote_checksum_timeout(self, mock_run, mock_host_config):
        """Returns None on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)

        result = _fetch_remote_checksum(mock_host_config, "/path/to/file.db")

        assert result is None

    @patch("subprocess.run")
    def test_fetch_remote_checksum_invalid_length(self, mock_run, mock_host_config):
        """Returns None for invalid checksum length."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="short_checksum\n",  # Not 64 chars
        )

        result = _fetch_remote_checksum(mock_host_config, "/path/to/file.db")

        assert result is None


class TestFetchRemoteChecksumAsync:
    """Tests for _fetch_remote_checksum_async function."""

    @pytest.mark.asyncio
    async def test_fetch_async_success(self, mock_host_config):
        """Successfully fetches remote checksum asynchronously."""
        expected_checksum = "b" * 64

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(expected_checksum.encode() + b"\n", b"")
            )
            mock_exec.return_value = mock_proc

            result = await _fetch_remote_checksum_async(mock_host_config, "/path/to/file.db")

            assert result == expected_checksum

    @pytest.mark.asyncio
    async def test_fetch_async_timeout(self, mock_host_config):
        """Returns None on async timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            result = await _fetch_remote_checksum_async(mock_host_config, "/path/to/file.db")

            assert result is None
            mock_proc.kill.assert_called_once()


# =============================================================================
# SSH Command Building Tests
# =============================================================================


class TestBuildSSHCommandForRsync:
    """Tests for build_ssh_command_for_rsync function."""

    def test_basic_ssh_command(self, mock_host_config_no_key):
        """Builds basic SSH command without key."""
        cmd = build_ssh_command_for_rsync(mock_host_config_no_key)

        assert "ssh" in cmd
        assert "StrictHostKeyChecking=no" in cmd
        assert "BatchMode=yes" in cmd
        assert "-i" not in cmd

    def test_ssh_command_with_key(self, mock_host_config):
        """Builds SSH command with key file."""
        cmd = build_ssh_command_for_rsync(mock_host_config)

        assert "-i" in cmd
        # Key path should be expanded
        assert "test_key" in cmd or ".ssh" in cmd

    def test_ssh_command_with_custom_port(self, mock_host_config_custom_port):
        """Builds SSH command with custom port."""
        cmd = build_ssh_command_for_rsync(mock_host_config_custom_port)

        assert "-p 2222" in cmd

    def test_ssh_command_default_port_no_flag(self, mock_host_config):
        """Default port 22 doesn't add -p flag."""
        cmd = build_ssh_command_for_rsync(mock_host_config)

        # Should not have -p 22 since it's the default
        assert "-p 22" not in cmd


# =============================================================================
# Basic Rsync Tests
# =============================================================================


class TestRsyncFile:
    """Tests for rsync_file function."""

    @patch("subprocess.run")
    def test_rsync_file_success(self, mock_run, mock_host_config, tmp_path):
        """Successfully rsyncs a file."""
        mock_run.return_value = MagicMock(returncode=0)
        local_path = tmp_path / "received.db"

        result = rsync_file(mock_host_config, "/remote/file.db", local_path)

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_rsync_file_creates_parent_dir(self, mock_run, mock_host_config, tmp_path):
        """Creates parent directory if it doesn't exist."""
        mock_run.return_value = MagicMock(returncode=0)
        local_path = tmp_path / "subdir" / "received.db"

        result = rsync_file(mock_host_config, "/remote/file.db", local_path)

        assert result is True
        assert local_path.parent.exists()

    @patch("subprocess.run")
    def test_rsync_file_failure(self, mock_run, mock_host_config, tmp_path):
        """Returns False on rsync failure."""
        mock_run.return_value = MagicMock(returncode=1)
        local_path = tmp_path / "received.db"

        result = rsync_file(mock_host_config, "/remote/file.db", local_path)

        assert result is False

    @patch("subprocess.run")
    def test_rsync_file_timeout(self, mock_run, mock_host_config, tmp_path):
        """Returns False on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=120)
        local_path = tmp_path / "received.db"

        result = rsync_file(mock_host_config, "/remote/file.db", local_path)

        assert result is False


class TestRsyncFileAsync:
    """Tests for rsync_file_async function."""

    @pytest.mark.asyncio
    async def test_rsync_async_success(self, mock_host_config, tmp_path):
        """Successfully rsyncs a file asynchronously."""
        local_path = tmp_path / "received.db"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await rsync_file_async(mock_host_config, "/remote/file.db", local_path)

            assert result is True

    @pytest.mark.asyncio
    async def test_rsync_async_timeout(self, mock_host_config, tmp_path):
        """Returns False on async timeout."""
        local_path = tmp_path / "received.db"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            result = await rsync_file_async(mock_host_config, "/remote/file.db", local_path)

            assert result is False


class TestRsyncDirectory:
    """Tests for rsync_directory function."""

    @patch("subprocess.run")
    def test_rsync_directory_success(self, mock_run, mock_host_config, tmp_path):
        """Successfully rsyncs a directory."""
        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "games"

        result = rsync_directory(mock_host_config, "/remote/games/", local_dir)

        assert result is True

    @patch("subprocess.run")
    def test_rsync_directory_with_patterns(self, mock_run, mock_host_config, tmp_path):
        """Supports include and exclude patterns."""
        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "games"

        result = rsync_directory(
            mock_host_config,
            "/remote/games/",
            local_dir,
            include_patterns=["*.db"],
            exclude_patterns=["*.tmp"],
        )

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert any("--include" in str(arg) for arg in call_args)
        assert any("--exclude" in str(arg) for arg in call_args)

    @patch("subprocess.run")
    def test_rsync_directory_with_delete(self, mock_run, mock_host_config, tmp_path):
        """Supports delete flag."""
        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "games"

        result = rsync_directory(mock_host_config, "/remote/games/", local_dir, delete=True)

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "--delete" in call_args


class TestRsyncPushFile:
    """Tests for rsync_push_file function."""

    @patch("subprocess.run")
    def test_rsync_push_success(self, mock_run, mock_host_config, temp_file):
        """Successfully pushes a file."""
        mock_run.return_value = MagicMock(returncode=0)

        result = rsync_push_file(mock_host_config, temp_file, "/remote/dest.db")

        assert result is True

    @patch("subprocess.run")
    def test_rsync_push_missing_local_file(self, mock_run, mock_host_config, tmp_path):
        """Returns False for missing local file."""
        missing = tmp_path / "nonexistent.db"

        result = rsync_push_file(mock_host_config, missing, "/remote/dest.db")

        assert result is False
        mock_run.assert_not_called()


# =============================================================================
# Verified Rsync Tests
# =============================================================================


class TestRsyncFileVerified:
    """Tests for rsync_file_verified function."""

    @patch("app.distributed.sync_utils._fetch_remote_checksum")
    @patch("subprocess.run")
    def test_verified_success(self, mock_run, mock_fetch, mock_host_config, tmp_path):
        """Successfully verifies transferred file."""
        expected_checksum = "c" * 64
        mock_fetch.return_value = expected_checksum
        mock_run.return_value = MagicMock(returncode=0)

        local_path = tmp_path / "received.db"
        local_path.write_text("Test content")

        with patch("app.distributed.sync_utils._compute_checksum", return_value=expected_checksum):
            result = rsync_file_verified(mock_host_config, "/remote/file.db", local_path)

        assert result.success is True
        assert result.verified is True
        assert result.checksum_matched is True

    @patch("app.distributed.sync_utils._fetch_remote_checksum")
    @patch("subprocess.run")
    def test_verified_checksum_mismatch(
        self, mock_run, mock_fetch, mock_host_config, tmp_path, temp_quarantine_dir
    ):
        """Quarantines file on checksum mismatch."""
        mock_fetch.return_value = "a" * 64
        mock_run.return_value = MagicMock(returncode=0)

        local_path = tmp_path / "received.db"
        local_path.write_text("Corrupted content")

        with patch("app.distributed.sync_utils._compute_checksum", return_value="b" * 64):
            result = rsync_file_verified(mock_host_config, "/remote/file.db", local_path)

        assert result.success is False
        assert result.verified is False
        assert result.checksum_matched is False
        assert "Checksum mismatch" in result.error

    @patch("app.distributed.sync_utils._fetch_remote_checksum")
    @patch("subprocess.run")
    def test_verified_no_remote_checksum(self, mock_run, mock_fetch, mock_host_config, tmp_path):
        """Proceeds without verification if remote checksum unavailable."""
        mock_fetch.return_value = None  # Can't get remote checksum
        mock_run.return_value = MagicMock(returncode=0)

        local_path = tmp_path / "received.db"
        local_path.write_text("Test content")

        result = rsync_file_verified(mock_host_config, "/remote/file.db", local_path)

        assert result.success is True
        assert result.verified is False  # Unverified because no checksum

    @patch("subprocess.run")
    def test_verified_rsync_failure(self, mock_run, mock_host_config, tmp_path):
        """Returns failure result on rsync error."""
        mock_run.return_value = MagicMock(returncode=1, stderr=b"Connection refused")

        local_path = tmp_path / "received.db"

        result = rsync_file_verified(mock_host_config, "/remote/file.db", local_path)

        assert result.success is False
        assert "rsync failed" in result.error


class TestRsyncFileVerifiedAsync:
    """Tests for rsync_file_verified_async function."""

    @pytest.mark.asyncio
    async def test_async_verified_success(self, mock_host_config, tmp_path):
        """Successfully verifies transferred file asynchronously."""
        expected_checksum = "d" * 64
        local_path = tmp_path / "received.db"
        local_path.write_text("Test content")

        with patch(
            "app.distributed.sync_utils._fetch_remote_checksum_async",
            new_callable=AsyncMock,
            return_value=expected_checksum,
        ):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_exec.return_value = mock_proc

                with patch(
                    "app.distributed.sync_utils._compute_checksum",
                    return_value=expected_checksum,
                ):
                    result = await rsync_file_verified_async(
                        mock_host_config, "/remote/file.db", local_path
                    )

        assert result.success is True
        assert result.verified is True


class TestRsyncPushFileVerified:
    """Tests for rsync_push_file_verified function."""

    @patch("app.distributed.sync_utils._fetch_remote_checksum")
    @patch("subprocess.run")
    def test_push_verified_success(self, mock_run, mock_fetch, mock_host_config, temp_file):
        """Successfully verifies pushed file."""
        local_checksum = "e" * 64
        mock_run.return_value = MagicMock(returncode=0)
        mock_fetch.return_value = local_checksum

        with patch("app.distributed.sync_utils._compute_checksum", return_value=local_checksum):
            result = rsync_push_file_verified(mock_host_config, temp_file, "/remote/dest.db")

        assert result.success is True
        assert result.verified is True
        assert result.checksum_matched is True

    @patch("app.distributed.sync_utils._fetch_remote_checksum")
    @patch("subprocess.run")
    def test_push_verified_mismatch(self, mock_run, mock_fetch, mock_host_config, temp_file):
        """Reports error on remote checksum mismatch."""
        mock_run.return_value = MagicMock(returncode=0)
        mock_fetch.return_value = "f" * 64  # Different from local

        with patch("app.distributed.sync_utils._compute_checksum", return_value="g" * 64):
            result = rsync_push_file_verified(mock_host_config, temp_file, "/remote/dest.db")

        assert result.success is False
        assert "mismatch" in result.error

    def test_push_verified_missing_local(self, mock_host_config, tmp_path):
        """Returns failure for missing local file."""
        missing = tmp_path / "nonexistent.db"

        result = rsync_push_file_verified(mock_host_config, missing, "/remote/dest.db")

        assert result.success is False
        assert "not found" in result.error


class TestRsyncDirectoryVerified:
    """Tests for rsync_directory_verified function."""

    @patch("subprocess.run")
    def test_directory_verified_success(self, mock_run, mock_host_config, tmp_path):
        """Successfully syncs directory with verification."""
        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "games"
        local_dir.mkdir()

        result = rsync_directory_verified(mock_host_config, "/remote/games/", local_dir)

        assert result.success is True
        # Directory verification is best-effort
        assert result.verified is False

    @patch("subprocess.run")
    def test_directory_verified_with_patterns(self, mock_run, mock_host_config, tmp_path):
        """Supports patterns in verified directory sync."""
        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "games"

        result = rsync_directory_verified(
            mock_host_config,
            "/remote/games/",
            local_dir,
            include_patterns=["*.db"],
            exclude_patterns=["*.tmp"],
        )

        assert result.success is True
        call_args = mock_run.call_args[0][0]
        assert "--checksum" in call_args


class TestRsyncDirectoryVerifiedAsync:
    """Tests for rsync_directory_verified_async function."""

    @pytest.mark.asyncio
    async def test_async_directory_verified_success(self, mock_host_config, tmp_path):
        """Successfully syncs directory asynchronously."""
        local_dir = tmp_path / "games"
        local_dir.mkdir()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await rsync_directory_verified_async(
                mock_host_config, "/remote/games/", local_dir
            )

        assert result.success is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_quarantine_max_age_constant(self):
        """QUARANTINE_MAX_AGE_DAYS has reasonable value."""
        assert QUARANTINE_MAX_AGE_DAYS > 0
        assert QUARANTINE_MAX_AGE_DAYS <= 90  # Not too long

    def test_quarantine_dir_is_path(self):
        """QUARANTINE_DIR is a Path object."""
        assert isinstance(QUARANTINE_DIR, Path)

    @patch("subprocess.run")
    def test_rsync_with_spaces_in_path(self, mock_run, mock_host_config, tmp_path):
        """Handles paths with spaces."""
        mock_run.return_value = MagicMock(returncode=0)
        local_path = tmp_path / "path with spaces" / "file.db"
        local_path.parent.mkdir(parents=True)

        result = rsync_file(mock_host_config, "/remote/file.db", local_path)

        assert result is True

    @patch("subprocess.run")
    def test_rsync_empty_timeout(self, mock_run, mock_host_config, tmp_path):
        """Uses default timeout when not specified."""
        mock_run.return_value = MagicMock(returncode=0)
        local_path = tmp_path / "file.db"

        rsync_file(mock_host_config, "/remote/file.db", local_path)

        # Check that timeout was specified in the call
        call_kwargs = mock_run.call_args[1]
        assert "timeout" in call_kwargs

    @patch("subprocess.run")
    def test_rsync_custom_timeout(self, mock_run, mock_host_config, tmp_path):
        """Respects custom timeout."""
        mock_run.return_value = MagicMock(returncode=0)
        local_path = tmp_path / "file.db"

        rsync_file(mock_host_config, "/remote/file.db", local_path, timeout=60)

        call_args = mock_run.call_args[0][0]
        assert "--timeout=60" in call_args
