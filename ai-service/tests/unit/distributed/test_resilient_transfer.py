"""Unit tests for resilient_transfer.py.

Tests the ResilientTransfer class for file transfers with automatic
verification, fallback, and quarantine.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.resilient_transfer import (
    LARGE_FILE_THRESHOLD,
    HUGE_FILE_THRESHOLD,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    ResilientTransfer,
    TransferRequest,
    TransferResult,
    transfer_file,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def transfer():
    """Create a ResilientTransfer instance for testing."""
    return ResilientTransfer()


@pytest.fixture
def sample_request(temp_dir):
    """Create a sample TransferRequest."""
    return TransferRequest(
        source_node="nebius-backbone-1",
        source_path="/data/games/test.db",
        target_path=temp_dir / "test.db",
        file_type="db",
        priority="normal",
    )


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_large_file_threshold(self):
        """Test large file threshold is 50MB."""
        assert LARGE_FILE_THRESHOLD == 50_000_000

    def test_huge_file_threshold(self):
        """Test huge file threshold is 500MB."""
        assert HUGE_FILE_THRESHOLD == 500_000_000

    def test_max_retries(self):
        """Test max retries is 3."""
        assert MAX_RETRIES == 3

    def test_retry_delay(self):
        """Test retry delay is 5 seconds."""
        assert RETRY_DELAY_SECONDS == 5


# =============================================================================
# Test TransferRequest Dataclass
# =============================================================================


class TestTransferRequest:
    """Tests for TransferRequest dataclass."""

    def test_required_fields(self, temp_dir):
        """Test creating request with required fields only."""
        request = TransferRequest(
            source_node="node1",
            source_path="/data/file.txt",
            target_path=temp_dir / "file.txt",
        )
        assert request.source_node == "node1"
        assert request.source_path == "/data/file.txt"
        assert request.target_path == temp_dir / "file.txt"

    def test_default_values(self, temp_dir):
        """Test default field values."""
        request = TransferRequest(
            source_node="node1",
            source_path="/data/file.txt",
            target_path=temp_dir / "file.txt",
        )
        assert request.expected_checksum is None
        assert request.expected_size is None
        assert request.file_type == "other"
        assert request.priority == "normal"

    def test_all_fields(self, temp_dir):
        """Test creating request with all fields."""
        request = TransferRequest(
            source_node="node1",
            source_path="/data/training/hex8_2p.npz",
            target_path=temp_dir / "hex8_2p.npz",
            expected_checksum="abc123def456",
            expected_size=100_000_000,
            file_type="npz",
            priority="critical",
        )
        assert request.expected_checksum == "abc123def456"
        assert request.expected_size == 100_000_000
        assert request.file_type == "npz"
        assert request.priority == "critical"

    def test_file_types(self, temp_dir):
        """Test all valid file types."""
        for file_type in ["npz", "db", "pth", "other"]:
            request = TransferRequest(
                source_node="node1",
                source_path="/data/file",
                target_path=temp_dir / "file",
                file_type=file_type,  # type: ignore
            )
            assert request.file_type == file_type

    def test_priorities(self, temp_dir):
        """Test all valid priorities."""
        for priority in ["low", "normal", "high", "critical"]:
            request = TransferRequest(
                source_node="node1",
                source_path="/data/file",
                target_path=temp_dir / "file",
                priority=priority,  # type: ignore
            )
            assert request.priority == priority


# =============================================================================
# Test TransferResult Dataclass
# =============================================================================


class TestTransferResult:
    """Tests for TransferResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = TransferResult(success=False)
        assert result.success is False
        assert result.bytes_transferred == 0
        assert result.transport_used == ""
        assert result.verification_passed is False
        assert result.error == ""
        assert result.retries == 0
        assert result.checksum == ""

    def test_successful_result(self):
        """Test successful transfer result."""
        result = TransferResult(
            success=True,
            bytes_transferred=1024 * 1024,
            transport_used="rsync_verified",
            verification_passed=True,
            checksum="abc123",
        )
        assert result.success is True
        assert result.bytes_transferred == 1024 * 1024
        assert result.transport_used == "rsync_verified"
        assert result.verification_passed is True

    def test_failed_result(self):
        """Test failed transfer result."""
        result = TransferResult(
            success=False,
            error="Connection timeout",
            retries=3,
        )
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.retries == 3


# =============================================================================
# Test ResilientTransfer Initialization
# =============================================================================


class TestResilientTransferInit:
    """Tests for ResilientTransfer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        transfer = ResilientTransfer()
        assert transfer.prefer_bittorrent is True
        assert transfer.verify_all is True
        assert transfer.quarantine_on_failure is True

    def test_custom_init(self):
        """Test custom initialization."""
        transfer = ResilientTransfer(
            prefer_bittorrent=False,
            verify_all=False,
            quarantine_on_failure=False,
        )
        assert transfer.prefer_bittorrent is False
        assert transfer.verify_all is False
        assert transfer.quarantine_on_failure is False

    def test_lazy_components_initially_none(self):
        """Test that lazy components are None initially."""
        transfer = ResilientTransfer()
        assert transfer._aria2_transport is None
        assert transfer._hosts_config is None


# =============================================================================
# Test Transport Selection Logic
# =============================================================================


class TestTransportSelection:
    """Tests for transport selection in transfer method."""

    @pytest.mark.asyncio
    async def test_huge_file_requires_bittorrent(self, transfer, sample_request):
        """Test that huge files (>500MB) require BitTorrent."""
        sample_request.expected_size = 600_000_000  # 600MB

        mock_result = TransferResult(success=True, verification_passed=True)
        with patch.object(transfer, "_fetch_remote_checksum", return_value="abc123"):
            with patch.object(transfer, "_fetch_remote_size", return_value=600_000_000):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_result) as mock_transfer:
                    await transfer.transfer(sample_request)

        # Should try bittorrent first, then rsync_verified
        mock_transfer.assert_called_once()
        transports = mock_transfer.call_args[0][1]
        assert transports == ["bittorrent", "rsync_verified"]

    @pytest.mark.asyncio
    async def test_large_file_prefers_bittorrent(self, sample_request):
        """Test that large files (>50MB) prefer BitTorrent."""
        transfer = ResilientTransfer(prefer_bittorrent=True)
        sample_request.expected_size = 75_000_000  # 75MB

        mock_result = TransferResult(success=True, verification_passed=True)
        with patch.object(transfer, "_fetch_remote_checksum", return_value="abc123"):
            with patch.object(transfer, "_fetch_remote_size", return_value=75_000_000):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_result) as mock_transfer:
                    await transfer.transfer(sample_request)

        transports = mock_transfer.call_args[0][1]
        assert "bittorrent" in transports
        assert "aria2" in transports
        assert "rsync_verified" in transports

    @pytest.mark.asyncio
    async def test_large_file_skips_bittorrent_when_disabled(self, sample_request):
        """Test that large files skip BitTorrent when preference disabled."""
        transfer = ResilientTransfer(prefer_bittorrent=False)
        sample_request.expected_size = 75_000_000  # 75MB

        mock_result = TransferResult(success=True, verification_passed=True)
        with patch.object(transfer, "_fetch_remote_checksum", return_value="abc123"):
            with patch.object(transfer, "_fetch_remote_size", return_value=75_000_000):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_result) as mock_transfer:
                    await transfer.transfer(sample_request)

        transports = mock_transfer.call_args[0][1]
        assert "bittorrent" not in transports
        assert transports == ["aria2", "rsync_verified"]

    @pytest.mark.asyncio
    async def test_small_file_uses_aria2(self, transfer, sample_request):
        """Test that small files (<50MB) use aria2."""
        sample_request.expected_size = 10_000_000  # 10MB

        mock_result = TransferResult(success=True, verification_passed=True)
        with patch.object(transfer, "_fetch_remote_checksum", return_value="abc123"):
            with patch.object(transfer, "_fetch_remote_size", return_value=10_000_000):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_result) as mock_transfer:
                    await transfer.transfer(sample_request)

        transports = mock_transfer.call_args[0][1]
        assert transports == ["aria2", "rsync_verified"]


# =============================================================================
# Test Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for transfer retry logic."""

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self, transfer, sample_request):
        """Test that retry stops on first success."""
        success_result = TransferResult(
            success=True, transport_used="aria2", verification_passed=True
        )
        with patch.object(transfer, "_transfer_via_aria2", return_value=success_result):
            result = await transfer._transfer_with_retry(sample_request, ["aria2"])

        assert result.success is True
        assert result.retries == 0

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, transfer, sample_request):
        """Test that transfer retries on failure."""
        fail_result = TransferResult(success=False, error="Connection failed")
        success_result = TransferResult(
            success=True, transport_used="aria2", verification_passed=True
        )

        call_count = 0

        async def mock_transfer(req):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return fail_result
            return success_result

        with patch.object(transfer, "_transfer_via_aria2", side_effect=mock_transfer):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip delay
                result = await transfer._transfer_with_retry(sample_request, ["aria2"])

        assert result.success is True
        assert result.retries == 2  # Succeeded on 3rd attempt (0-indexed = 2)

    @pytest.mark.asyncio
    async def test_falls_back_to_next_transport(self, transfer, sample_request):
        """Test falling back to next transport after all retries fail."""
        aria2_fail = TransferResult(success=False, error="aria2 failed")
        rsync_success = TransferResult(
            success=True, transport_used="rsync_verified", verification_passed=True
        )

        with patch.object(transfer, "_transfer_via_aria2", return_value=aria2_fail):
            with patch.object(transfer, "_transfer_via_rsync", return_value=rsync_success):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await transfer._transfer_with_retry(
                        sample_request, ["aria2", "rsync_verified"]
                    )

        assert result.success is True
        assert result.transport_used == "rsync_verified"

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, transfer, sample_request):
        """Test handling exceptions during transfer."""
        with patch.object(transfer, "_transfer_via_aria2", side_effect=Exception("Unexpected error")):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await transfer._transfer_with_retry(sample_request, ["aria2"])

        assert result.success is False
        assert "Unexpected error" in result.error


# =============================================================================
# Test File Type Validation
# =============================================================================


class TestFileTypeValidation:
    """Tests for file type-specific validation."""

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, transfer, sample_request, temp_dir):
        """Test validation of non-existent file."""
        sample_request.target_path = temp_dir / "nonexistent.db"
        valid, error = await transfer._validate_file_type(sample_request)

        assert valid is False
        assert "does not exist" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_empty_file(self, transfer, sample_request, temp_dir):
        """Test validation of empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        sample_request.target_path = empty_file
        sample_request.file_type = "other"

        valid, error = await transfer._validate_file_type(sample_request)

        assert valid is False
        assert "empty" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_other_file_type(self, transfer, sample_request, temp_dir):
        """Test validation of 'other' file type (just checks non-empty)."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, world!")
        sample_request.target_path = test_file
        sample_request.file_type = "other"

        valid, error = await transfer._validate_file_type(sample_request)

        assert valid is True
        assert error == ""


# =============================================================================
# Test BitTorrent Transfer
# =============================================================================


class TestBitTorrentTransfer:
    """Tests for BitTorrent transfer."""

    @pytest.mark.asyncio
    async def test_bittorrent_no_transport(self, transfer, sample_request):
        """Test BitTorrent fails gracefully when transport unavailable."""
        with patch.object(transfer, "_get_aria2_transport", return_value=None):
            result = await transfer._transfer_via_bittorrent(sample_request)

        assert result.success is False
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_bittorrent_no_torrent(self, transfer, sample_request):
        """Test BitTorrent fails when no torrent available."""
        mock_transport = MagicMock()
        with patch.object(transfer, "_get_aria2_transport", return_value=mock_transport):
            with patch.object(transfer, "_get_torrent_info", return_value=None):
                result = await transfer._transfer_via_bittorrent(sample_request)

        assert result.success is False
        assert "no torrent" in result.error.lower()


# =============================================================================
# Test Aria2 Transfer
# =============================================================================


class TestAria2Transfer:
    """Tests for aria2 transfer."""

    @pytest.mark.asyncio
    async def test_aria2_no_transport(self, transfer, sample_request):
        """Test aria2 fails gracefully when transport unavailable."""
        with patch.object(transfer, "_get_aria2_transport", return_value=None):
            result = await transfer._transfer_via_aria2(sample_request)

        assert result.success is False
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_aria2_no_source_url(self, transfer, sample_request):
        """Test aria2 fails when source URL cannot be determined."""
        mock_transport = MagicMock()
        with patch.object(transfer, "_get_aria2_transport", return_value=mock_transport):
            with patch.object(transfer, "_get_source_url", return_value=None):
                result = await transfer._transfer_via_aria2(sample_request)

        assert result.success is False
        assert "url" in result.error.lower()


# =============================================================================
# Test Rsync Transfer
# =============================================================================


class TestRsyncTransfer:
    """Tests for rsync transfer."""

    @pytest.mark.asyncio
    async def test_rsync_unknown_host(self, transfer, sample_request):
        """Test rsync fails for unknown host."""
        with patch.object(transfer, "_get_host_config", return_value=None):
            result = await transfer._transfer_via_rsync(sample_request)

        assert result.success is False
        assert "unknown host" in result.error.lower()


# =============================================================================
# Test Helper Methods
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.mark.asyncio
    async def test_fetch_remote_checksum_no_host(self, transfer, sample_request):
        """Test fetching checksum when host not found."""
        with patch.object(transfer, "_get_host_config", return_value=None):
            result = await transfer._fetch_remote_checksum(sample_request)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_remote_size_no_host(self, transfer, sample_request):
        """Test fetching size when host not found."""
        with patch.object(transfer, "_get_host_config", return_value=None):
            result = await transfer._fetch_remote_size(sample_request)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_host_config_caches(self, transfer, sample_request):
        """Test that host config is cached after first load."""
        # Set up a pre-loaded mock config
        mock_host = MagicMock()
        mock_config = {"node1": mock_host}
        transfer._hosts_config = mock_config

        # First call should use cached config
        result1 = await transfer._get_host_config("node1")
        # Second call should also use cached config
        result2 = await transfer._get_host_config("node1")

        # Both should return the same mock host
        assert result1 is mock_host
        assert result2 is mock_host
        # Config should not have been reset
        assert transfer._hosts_config is mock_config

    @pytest.mark.asyncio
    async def test_get_aria2_transport_caches(self, transfer):
        """Test that aria2 transport is cached."""
        # Pre-set a mock transport
        mock_transport = MagicMock()
        transfer._aria2_transport = mock_transport

        # Should return the cached transport
        transport1 = await transfer._get_aria2_transport()
        transport2 = await transfer._get_aria2_transport()

        assert transport1 is mock_transport
        assert transport2 is mock_transport

    @pytest.mark.asyncio
    async def test_get_source_url(self, transfer, sample_request):
        """Test building source URL."""
        mock_host = MagicMock()
        mock_host.ip = "192.168.1.100"
        with patch.object(transfer, "_get_host_config", return_value=mock_host):
            url = await transfer._get_source_url(sample_request)

        assert url == "http://192.168.1.100:8765/files//data/games/test.db"


# =============================================================================
# Test Quarantine
# =============================================================================


class TestQuarantine:
    """Tests for file quarantine."""

    @pytest.mark.asyncio
    async def test_quarantine_on_type_validation_failure(self, transfer, temp_dir):
        """Test that files are quarantined on type validation failure."""
        request = TransferRequest(
            source_node="node1",
            source_path="/data/file.npz",
            target_path=temp_dir / "file.npz",
            file_type="npz",
        )

        # Create a file that will fail NPZ validation
        (temp_dir / "file.npz").write_text("not an npz file")

        mock_transfer_result = TransferResult(
            success=True,
            verification_passed=True,
            bytes_transferred=100,
        )

        with patch.object(transfer, "_fetch_remote_checksum", return_value=None):
            with patch.object(transfer, "_fetch_remote_size", return_value=100):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_transfer_result):
                    with patch.object(transfer, "_validate_file_type", return_value=(False, "Invalid NPZ")):
                        with patch.object(transfer, "_quarantine_file") as mock_quarantine:
                            result = await transfer.transfer(request)

        assert result.success is False
        mock_quarantine.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_quarantine_when_disabled(self, temp_dir):
        """Test that files are not quarantined when disabled."""
        transfer = ResilientTransfer(quarantine_on_failure=False)
        request = TransferRequest(
            source_node="node1",
            source_path="/data/file.npz",
            target_path=temp_dir / "file.npz",
            file_type="npz",
        )

        mock_transfer_result = TransferResult(
            success=True,
            verification_passed=True,
            bytes_transferred=100,
        )

        with patch.object(transfer, "_fetch_remote_checksum", return_value=None):
            with patch.object(transfer, "_fetch_remote_size", return_value=100):
                with patch.object(transfer, "_transfer_with_retry", return_value=mock_transfer_result):
                    with patch.object(transfer, "_validate_file_type", return_value=(False, "Invalid NPZ")):
                        with patch.object(transfer, "_quarantine_file") as mock_quarantine:
                            await transfer.transfer(request)

        mock_quarantine.assert_not_called()


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestTransferFileFunction:
    """Tests for transfer_file convenience function."""

    @pytest.mark.asyncio
    async def test_basic_transfer(self, temp_dir):
        """Test basic file transfer via convenience function."""
        mock_result = TransferResult(
            success=True,
            verification_passed=True,
            bytes_transferred=1024,
        )

        with patch("app.distributed.resilient_transfer.ResilientTransfer") as MockClass:
            mock_instance = MagicMock()
            mock_instance.transfer = AsyncMock(return_value=mock_result)
            MockClass.return_value = mock_instance

            result = await transfer_file(
                source_node="node1",
                source_path="/data/test.db",
                target_path=temp_dir / "test.db",
                file_type="db",
            )

        assert result.success is True
        assert result.bytes_transferred == 1024

    @pytest.mark.asyncio
    async def test_transfer_with_checksum(self, temp_dir):
        """Test transfer with expected checksum."""
        mock_result = TransferResult(success=True)

        with patch("app.distributed.resilient_transfer.ResilientTransfer") as MockClass:
            mock_instance = MagicMock()
            mock_instance.transfer = AsyncMock(return_value=mock_result)
            MockClass.return_value = mock_instance

            await transfer_file(
                source_node="node1",
                source_path="/data/test.db",
                target_path=temp_dir / "test.db",
                expected_checksum="abc123",
            )

            # Verify the request was created with the checksum
            call_args = mock_instance.transfer.call_args[0][0]
            assert call_args.expected_checksum == "abc123"
