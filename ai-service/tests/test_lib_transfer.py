"""Tests for scripts.lib.transfer module.

Tests file transfer utilities including:
- TransferConfig and TransferResult dataclasses
- Checksum computation
- Local file copy
- Compression/decompression
- SCP and rsync operations (mocked)
"""

import gzip
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.transfer import (
    TransferConfig,
    TransferResult,
    compute_checksum,
    compress_file,
    copy_local,
    decompress_file,
    get_remote_checksum,
    rsync_pull,
    rsync_push,
    scp_pull,
    scp_push,
    verify_transfer,
)


class TestTransferConfig:
    """Tests for TransferConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransferConfig()

        assert config.ssh_key is None
        assert config.ssh_user == "root"
        assert config.ssh_port == 22
        assert config.connect_timeout == 30
        assert config.transfer_timeout == 300
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.compress is True
        assert config.verify_checksum is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransferConfig(
            ssh_key="~/.ssh/custom_key",
            ssh_user="ubuntu",
            ssh_port=2222,
            max_retries=5,
            compress=False,
        )

        assert config.ssh_key == "~/.ssh/custom_key"
        assert config.ssh_user == "ubuntu"
        assert config.ssh_port == 2222
        assert config.max_retries == 5
        assert config.compress is False

    def test_get_ssh_options_basic(self):
        """Test SSH options generation without key."""
        config = TransferConfig()
        opts = config.get_ssh_options()

        assert "-o" in opts
        assert "StrictHostKeyChecking=no" in opts
        assert "BatchMode=yes" in opts
        assert f"ConnectTimeout={config.connect_timeout}" in opts

    def test_get_ssh_options_with_key(self, tmp_path):
        """Test SSH options include key when it exists."""
        # Create a fake key file
        key_file = tmp_path / "test_key"
        key_file.write_text("fake key")

        config = TransferConfig(ssh_key=str(key_file))
        opts = config.get_ssh_options()

        assert "-i" in opts
        assert str(key_file) in opts


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_success_result(self):
        """Test successful transfer result."""
        result = TransferResult(
            success=True,
            bytes_transferred=1024,
            duration_seconds=2.0,
            method="scp",
            source="/local/file",
            destination="host:/remote/file",
        )

        assert result.success is True
        assert bool(result) is True
        assert result.bytes_transferred == 1024
        assert result.method == "scp"

    def test_failure_result(self):
        """Test failed transfer result."""
        result = TransferResult(
            success=False,
            error="Connection refused",
            method="scp",
            source="/local/file",
            destination="host:/remote/file",
        )

        assert result.success is False
        assert bool(result) is False
        assert result.error == "Connection refused"

    def test_speed_calculation(self):
        """Test speed calculation in MB/s."""
        result = TransferResult(
            success=True,
            bytes_transferred=10 * 1024 * 1024,  # 10 MB
            duration_seconds=2.0,
        )

        assert result.speed_mbps == pytest.approx(5.0)

    def test_speed_zero_duration(self):
        """Test speed is 0 when duration is 0."""
        result = TransferResult(
            success=True,
            bytes_transferred=1024,
            duration_seconds=0,
        )

        assert result.speed_mbps == 0.0

    def test_speed_zero_bytes(self):
        """Test speed is 0 when no bytes transferred."""
        result = TransferResult(
            success=True,
            bytes_transferred=0,
            duration_seconds=1.0,
        )

        assert result.speed_mbps == 0.0


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_md5_checksum(self, tmp_path):
        """Test MD5 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file, algorithm="md5")

        # Known MD5 of "Hello, World!"
        assert checksum == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_sha256_checksum(self, tmp_path):
        """Test SHA256 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file, algorithm="sha256")

        assert len(checksum) == 64  # SHA256 is 64 hex chars

    def test_sha1_checksum(self, tmp_path):
        """Test SHA1 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file, algorithm="sha1")

        assert len(checksum) == 40  # SHA1 is 40 hex chars

    def test_invalid_algorithm(self, tmp_path):
        """Test error on invalid algorithm."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compute_checksum(test_file, algorithm="invalid")

    def test_path_as_string(self, tmp_path):
        """Test checksum with string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Should work with string path
        checksum = compute_checksum(str(test_file))
        assert len(checksum) == 32  # MD5 default


class TestCopyLocal:
    """Tests for copy_local function."""

    def test_successful_copy(self, tmp_path):
        """Test successful local file copy."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "dest.txt"

        result = copy_local(source, dest)

        assert result.success is True
        assert dest.exists()
        assert dest.read_text() == "test content"
        assert result.method == "local_copy"
        assert result.checksum_verified is True

    def test_copy_with_checksum_verification(self, tmp_path):
        """Test copy verifies checksum."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "dest.txt"

        result = copy_local(source, dest, verify_checksum=True)

        assert result.success is True
        assert result.checksum_verified is True

    def test_copy_nonexistent_file(self, tmp_path):
        """Test copy fails for nonexistent source."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        result = copy_local(source, dest)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_copy_creates_parent_dirs(self, tmp_path):
        """Test copy creates parent directories."""
        source = tmp_path / "source.txt"
        source.write_text("test")
        dest = tmp_path / "nested" / "path" / "dest.txt"

        result = copy_local(source, dest)

        assert result.success is True
        assert dest.exists()

    def test_copy_directory(self, tmp_path):
        """Test copying a directory."""
        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("content1")
        (source_dir / "file2.txt").write_text("content2")
        dest_dir = tmp_path / "dest_dir"

        result = copy_local(source_dir, dest_dir, verify_checksum=False)

        assert result.success is True
        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()


class TestCompression:
    """Tests for compress_file and decompress_file."""

    def test_compress_file(self, tmp_path):
        """Test file compression."""
        source = tmp_path / "test.txt"
        source.write_text("test content " * 100)  # Compressible content
        original_size = source.stat().st_size

        compressed_path, compressed_size = compress_file(source)

        assert compressed_path.exists()
        assert compressed_path.suffix == ".gz"
        assert compressed_size < original_size

    def test_compress_with_custom_dest(self, tmp_path):
        """Test compression to custom destination."""
        source = tmp_path / "test.txt"
        source.write_text("content")
        dest = tmp_path / "custom.gz"

        compressed_path, _ = compress_file(source, dest)

        assert compressed_path == dest
        assert dest.exists()

    def test_decompress_file(self, tmp_path):
        """Test file decompression."""
        source = tmp_path / "test.txt"
        original_content = "test content " * 100
        source.write_text(original_content)

        # Compress first
        compressed_path, _ = compress_file(source)

        # Decompress
        decompressed_path, _ = decompress_file(compressed_path)

        assert decompressed_path.exists()
        assert decompressed_path.read_text() == original_content

    def test_compression_roundtrip(self, tmp_path):
        """Test compress/decompress roundtrip preserves content."""
        source = tmp_path / "original.txt"
        content = "Hello, World! " * 50
        source.write_text(content)

        compressed, _ = compress_file(source)
        decompressed, _ = decompress_file(compressed)

        assert decompressed.read_text() == content


class TestScpPush:
    """Tests for scp_push function (mocked)."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_scp_push_success(self, mock_run, tmp_path):
        """Test successful SCP push."""
        source = tmp_path / "test.txt"
        source.write_text("content")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        config = TransferConfig(verify_checksum=False)
        result = scp_push(source, "host", 22, "/remote/", config)

        assert result.success is True
        assert result.method == "scp"
        assert mock_run.called

    @patch("scripts.lib.transfer.subprocess.run")
    def test_scp_push_failure(self, mock_run, tmp_path):
        """Test SCP push failure."""
        source = tmp_path / "test.txt"
        source.write_text("content")

        mock_run.return_value = MagicMock(
            returncode=1, stderr="Connection refused"
        )

        config = TransferConfig(max_retries=1, retry_delay=0)
        result = scp_push(source, "host", 22, "/remote/", config)

        assert result.success is False
        assert "Connection refused" in result.error

    def test_scp_push_nonexistent_file(self):
        """Test SCP push with nonexistent file."""
        config = TransferConfig()
        result = scp_push("/nonexistent/file", "host", 22, "/remote/", config)

        assert result.success is False
        assert "not found" in result.error.lower()


class TestScpPull:
    """Tests for scp_pull function (mocked)."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_scp_pull_success(self, mock_run, tmp_path):
        """Test successful SCP pull."""
        dest = tmp_path / "pulled.txt"

        def side_effect(*args, **kwargs):
            # Simulate SCP creating the file
            dest.write_text("pulled content")
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect

        config = TransferConfig(verify_checksum=False)
        result = scp_pull("host", 22, "/remote/file.txt", dest, config)

        assert result.success is True
        assert result.method == "scp"
        assert dest.exists()


class TestRsyncPush:
    """Tests for rsync_push function (mocked)."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_rsync_push_success(self, mock_run, tmp_path):
        """Test successful rsync push."""
        source = tmp_path / "test.txt"
        source.write_text("content")

        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="",
            stdout="sent 1,024 bytes  received 42 bytes",
        )

        config = TransferConfig()
        result = rsync_push(source, "host", 22, "/remote/", config)

        assert result.success is True
        assert result.method == "rsync"

    @patch("scripts.lib.transfer.subprocess.run")
    def test_rsync_push_with_exclude(self, mock_run, tmp_path):
        """Test rsync push with exclude patterns."""
        source = tmp_path / "test.txt"
        source.write_text("content")

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        config = TransferConfig()
        result = rsync_push(
            source, "host", 22, "/remote/", config, exclude=["*.pyc", "__pycache__"]
        )

        assert result.success is True
        # Verify exclude was passed
        call_args = mock_run.call_args[0][0]
        assert "--exclude" in call_args


class TestRsyncPull:
    """Tests for rsync_pull function (mocked)."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_rsync_pull_success(self, mock_run, tmp_path):
        """Test successful rsync pull."""
        dest = tmp_path / "pulled.txt"

        def side_effect(*args, **kwargs):
            dest.write_text("pulled content")
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = side_effect

        config = TransferConfig()
        result = rsync_pull("host", 22, "/remote/file.txt", dest, config)

        assert result.success is True
        assert result.method == "rsync"


class TestGetRemoteChecksum:
    """Tests for get_remote_checksum function (mocked)."""

    @patch("scripts.lib.transfer.subprocess.run")
    def test_get_remote_checksum_success(self, mock_run):
        """Test successful remote checksum retrieval."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456\n",
        )

        config = TransferConfig()
        checksum = get_remote_checksum("host", "/remote/file", config)

        assert checksum == "abc123def456"

    @patch("scripts.lib.transfer.subprocess.run")
    def test_get_remote_checksum_failure(self, mock_run):
        """Test remote checksum failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        config = TransferConfig()
        checksum = get_remote_checksum("host", "/remote/file", config)

        assert checksum is None


class TestVerifyTransfer:
    """Tests for verify_transfer function."""

    @patch("scripts.lib.transfer.get_remote_checksum")
    def test_verify_matching_checksums(self, mock_remote, tmp_path):
        """Test verification passes when checksums match."""
        local_file = tmp_path / "test.txt"
        local_file.write_text("Hello, World!")
        local_checksum = compute_checksum(local_file)

        mock_remote.return_value = local_checksum

        config = TransferConfig()
        result = verify_transfer(local_file, "host", "/remote/test.txt", config)

        assert result is True

    @patch("scripts.lib.transfer.get_remote_checksum")
    def test_verify_mismatched_checksums(self, mock_remote, tmp_path):
        """Test verification fails when checksums don't match."""
        local_file = tmp_path / "test.txt"
        local_file.write_text("local content")

        mock_remote.return_value = "different_checksum"

        config = TransferConfig()
        result = verify_transfer(local_file, "host", "/remote/test.txt", config)

        assert result is False

    def test_verify_nonexistent_file(self):
        """Test verification fails for nonexistent local file."""
        config = TransferConfig()
        result = verify_transfer("/nonexistent", "host", "/remote/file", config)

        assert result is False
