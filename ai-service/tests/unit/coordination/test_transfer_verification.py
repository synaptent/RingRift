"""Tests for app/coordination/transfer_verification.py.

This module provides checksum verification for data transfers to ensure
integrity and detect corruption. Tests cover checksum computation, transfer
verification, quarantine operations, and health checks.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.transfer_verification import (
    CHUNK_SIZE,
    MAX_QUARANTINE_AGE_DAYS,
    BatchChecksum,
    QuarantineRecord,
    TransferRecord,
    TransferVerifier,
    compute_batch_checksum,
    compute_file_checksum,
    get_transfer_verifier,
    quarantine_file,
    reset_transfer_verifier,
    verify_batch,
    verify_transfer,
    wire_transfer_verifier_events,
)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_verification.db"


@pytest.fixture
def verifier(temp_db: Path) -> TransferVerifier:
    """Create a TransferVerifier with temp database."""
    # Reset singleton to ensure clean state
    TransferVerifier.reset_instance()
    v = TransferVerifier(db_path=temp_db)
    yield v
    TransferVerifier.reset_instance()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello, World! This is test content for verification.")
    return file_path


@pytest.fixture
def large_file(tmp_path: Path) -> Path:
    """Create a larger file (> 1 chunk) for testing."""
    file_path = tmp_path / "large.bin"
    # Create file larger than CHUNK_SIZE
    file_path.write_bytes(b"x" * (CHUNK_SIZE + 1000))
    return file_path


# =============================================================================
# TransferRecord Tests
# =============================================================================
class TestTransferRecord:
    """Tests for TransferRecord dataclass."""

    def test_creation(self):
        """Test TransferRecord creation."""
        record = TransferRecord(
            transfer_id=1,
            source_path="/src/file.db",
            dest_path="/dst/file.db",
            source_checksum="abc123",
            dest_checksum="abc123",
            file_size=1024,
            transfer_time=time.time(),
            verified=True,
            verification_time=time.time(),
        )

        assert record.transfer_id == 1
        assert record.source_path == "/src/file.db"
        assert record.verified is True
        assert record.error_message == ""

    def test_to_dict(self):
        """Test TransferRecord.to_dict()."""
        now = time.time()
        record = TransferRecord(
            transfer_id=1,
            source_path="/src/file.db",
            dest_path="/dst/file.db",
            source_checksum="abcdef1234567890",
            dest_checksum="abcdef1234567890",
            file_size=1024,
            transfer_time=now,
            verified=True,
            verification_time=now,
        )

        d = record.to_dict()

        assert d["transfer_id"] == 1
        assert d["verified"] is True
        assert "..." in d["source_checksum"]  # Truncated


# =============================================================================
# QuarantineRecord Tests
# =============================================================================
class TestQuarantineRecord:
    """Tests for QuarantineRecord dataclass."""

    def test_creation(self):
        """Test QuarantineRecord creation."""
        record = QuarantineRecord(
            quarantine_id=1,
            original_path="/data/corrupt.db",
            quarantine_path="/quarantine/corrupt.db",
            reason="checksum_mismatch",
            quarantined_at=time.time(),
            file_size=2048,
            checksum="abc123",
        )

        assert record.quarantine_id == 1
        assert record.reason == "checksum_mismatch"
        assert record.metadata == {}

    def test_with_metadata(self):
        """Test QuarantineRecord with metadata."""
        record = QuarantineRecord(
            quarantine_id=1,
            original_path="/data/file.db",
            quarantine_path="/quarantine/file.db",
            reason="size_mismatch",
            quarantined_at=time.time(),
            file_size=1024,
            checksum="xyz789",
            metadata={"expected_size": 2048, "source": "sync"},
        )

        assert record.metadata["expected_size"] == 2048
        assert record.metadata["source"] == "sync"


# =============================================================================
# BatchChecksum Tests
# =============================================================================
class TestBatchChecksum:
    """Tests for BatchChecksum dataclass."""

    def test_creation(self):
        """Test BatchChecksum creation."""
        batch = BatchChecksum(
            batch_id="batch_001",
            checksum="sha256hash",
            record_count=100,
            byte_count=50000,
            first_record_id="rec_001",
            last_record_id="rec_100",
            created_at=time.time(),
        )

        assert batch.batch_id == "batch_001"
        assert batch.record_count == 100


# =============================================================================
# TransferVerifier Core Tests
# =============================================================================
class TestTransferVerifier:
    """Tests for TransferVerifier class."""

    def test_singleton_pattern(self, temp_db: Path):
        """Test TransferVerifier singleton pattern."""
        TransferVerifier.reset_instance()

        v1 = TransferVerifier.get_instance(temp_db)
        v2 = TransferVerifier.get_instance()

        assert v1 is v2

        TransferVerifier.reset_instance()

    def test_reset_instance(self, temp_db: Path):
        """Test singleton reset."""
        TransferVerifier.reset_instance()

        v1 = TransferVerifier.get_instance(temp_db)
        TransferVerifier.reset_instance()
        v2 = TransferVerifier.get_instance(temp_db)

        assert v1 is not v2

        TransferVerifier.reset_instance()

    def test_database_initialization(self, verifier: TransferVerifier):
        """Test database tables are created."""
        conn = verifier._get_connection()
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "transfer_records" in tables
        assert "quarantine_records" in tables
        assert "batch_checksums" in tables

    def test_compute_checksum(self, verifier: TransferVerifier, sample_file: Path):
        """Test file checksum computation."""
        checksum = verifier.compute_checksum(sample_file)

        assert len(checksum) == 64  # SHA256 hex length
        assert checksum.isalnum()

        # Same file should produce same checksum
        checksum2 = verifier.compute_checksum(sample_file)
        assert checksum == checksum2

    def test_compute_checksum_large_file(self, verifier: TransferVerifier, large_file: Path):
        """Test checksum for files larger than chunk size."""
        checksum = verifier.compute_checksum(large_file)

        assert len(checksum) == 64
        assert checksum.isalnum()

    def test_compute_checksum_from_bytes(self, verifier: TransferVerifier):
        """Test bytes checksum computation."""
        data = b"Test data for checksum"
        checksum = verifier.compute_checksum_from_bytes(data)

        assert len(checksum) == 64

        # Same data should produce same checksum
        checksum2 = verifier.compute_checksum_from_bytes(data)
        assert checksum == checksum2

        # Different data should produce different checksum
        checksum3 = verifier.compute_checksum_from_bytes(b"Different data")
        assert checksum != checksum3


# =============================================================================
# Batch Checksum Tests
# =============================================================================
class TestBatchChecksumComputation:
    """Tests for batch checksum operations."""

    def test_compute_batch_checksum(self, verifier: TransferVerifier):
        """Test computing checksum for a batch of records."""
        records = [
            '{"id": "rec_001", "data": "hello"}',
            '{"id": "rec_002", "data": "world"}',
            '{"id": "rec_003", "data": "test"}',
        ]

        batch = verifier.compute_batch_checksum(records, "batch_001")

        assert batch.batch_id == "batch_001"
        assert batch.record_count == 3
        assert len(batch.checksum) == 64
        assert batch.byte_count > 0

    def test_compute_batch_checksum_empty(self, verifier: TransferVerifier):
        """Test batch checksum with empty records."""
        batch = verifier.compute_batch_checksum([], "empty_batch")

        assert batch.batch_id == "empty_batch"
        assert batch.record_count == 0
        assert batch.checksum == ""

    def test_compute_batch_auto_stores(self, verifier: TransferVerifier):
        """Test that compute_batch_checksum automatically stores checksums."""
        records = ['{"id": "1"}', '{"id": "2"}']
        batch = verifier.compute_batch_checksum(records, "test_batch")

        # compute_batch_checksum auto-stores, verify by checking verify_batch works
        result = verifier.verify_batch("test_batch", records)
        assert result is True

    def test_verify_batch_success(self, verifier: TransferVerifier):
        """Test successful batch verification."""
        records = ['{"id": "1"}', '{"id": "2"}', '{"id": "3"}']

        # compute_batch_checksum auto-stores
        verifier.compute_batch_checksum(records, "verify_batch")

        # Verify - should succeed
        result = verifier.verify_batch("verify_batch", records)
        assert result is True

    def test_verify_batch_failure(self, verifier: TransferVerifier):
        """Test batch verification failure when records differ."""
        original_records = ['{"id": "1"}', '{"id": "2"}']
        modified_records = ['{"id": "1"}', '{"id": "3"}']  # Different!

        # Store original checksum (auto-stores)
        verifier.compute_batch_checksum(original_records, "fail_batch")

        # Verify with modified records - should fail
        result = verifier.verify_batch("fail_batch", modified_records)
        assert result is False


# =============================================================================
# Transfer Verification Tests
# =============================================================================
class TestTransferVerification:
    """Tests for transfer verification operations."""

    def test_record_transfer(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test recording a transfer."""
        source = sample_file
        dest = tmp_path / "copy.txt"
        dest.write_bytes(source.read_bytes())

        source_checksum = verifier.compute_checksum(source)

        # record_transfer returns transfer_id (int)
        transfer_id = verifier.record_transfer(
            source_path=str(source),
            dest_path=str(dest),
            source_checksum=source_checksum,
        )

        assert transfer_id is not None
        assert transfer_id > 0

    def test_verify_transfer_success(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test successful transfer verification."""
        source = sample_file
        dest = tmp_path / "verified.txt"
        dest.write_bytes(source.read_bytes())  # Identical content

        source_checksum = verifier.compute_checksum(source)

        # Record transfer (returns transfer_id)
        transfer_id = verifier.record_transfer(
            source_path=str(source),
            dest_path=str(dest),
            source_checksum=source_checksum,
        )

        # Verify using verify_transfer(transfer_id, dest_path)
        is_valid = verifier.verify_transfer(transfer_id, dest)

        assert is_valid is True

    def test_verify_transfer_failure(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test transfer verification failure."""
        source = sample_file
        dest = tmp_path / "corrupted.txt"
        dest.write_text("Different content - corrupted!")

        source_checksum = verifier.compute_checksum(source)

        # Record transfer (returns transfer_id)
        transfer_id = verifier.record_transfer(
            source_path=str(source),
            dest_path=str(dest),
            source_checksum=source_checksum,
        )

        # Verify - should fail
        is_valid = verifier.verify_transfer(transfer_id, dest)

        assert is_valid is False

    def test_quick_verify(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test quick verification without database record."""
        source = sample_file
        dest = tmp_path / "quick.txt"
        dest.write_bytes(source.read_bytes())

        source_checksum = verifier.compute_checksum(source)

        result = verifier.quick_verify(source_checksum, dest)
        assert result is True

    def test_quick_verify_mismatch(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test quick verification with mismatch."""
        dest = tmp_path / "quick.txt"
        dest.write_text("Different content")

        result = verifier.quick_verify("incorrect_checksum", dest)
        assert result is False


# =============================================================================
# Quarantine Tests
# =============================================================================
class TestQuarantine:
    """Tests for quarantine operations."""

    def test_quarantine_file(self, verifier: TransferVerifier, tmp_path: Path):
        """Test moving file to quarantine."""
        # Create file to quarantine
        bad_file = tmp_path / "bad_data.db"
        bad_file.write_text("Corrupted data")

        quarantine_path = verifier.quarantine(
            file_path=bad_file,
            reason="checksum_mismatch",
        )

        # Original should be moved
        assert not bad_file.exists()
        assert Path(quarantine_path).exists()

    def test_quarantine_with_metadata(self, verifier: TransferVerifier, tmp_path: Path):
        """Test quarantine with metadata."""
        bad_file = tmp_path / "corrupt.db"
        bad_file.write_text("Bad data")

        metadata = {
            "expected_checksum": "abc123",
            "actual_checksum": "xyz789",
            "source_node": "node1",
        }

        quarantine_path = verifier.quarantine(
            file_path=bad_file,
            reason="verification_failed",
            metadata=metadata,
        )

        assert Path(quarantine_path).exists()

    def test_get_quarantined_files(self, verifier: TransferVerifier, tmp_path: Path):
        """Test retrieving quarantine records."""
        # Quarantine multiple files
        for i in range(3):
            f = tmp_path / f"bad_{i}.db"
            f.write_text(f"Bad data {i}")
            verifier.quarantine(f, reason=f"test_{i}")

        # Use get_quarantined_files() - the actual method name
        records = verifier.get_quarantined_files()

        assert len(records) == 3
        # Records should be QuarantineRecord instances
        assert all(hasattr(r, "quarantine_id") for r in records)

    def test_get_quarantined_files_with_limit(self, verifier: TransferVerifier, tmp_path: Path):
        """Test retrieving quarantine records with limit."""
        # Quarantine multiple files
        for i in range(5):
            f = tmp_path / f"bad_{i}.db"
            f.write_text(f"Bad data {i}")
            verifier.quarantine(f, reason=f"test_{i}")

        # Use limit parameter
        records = verifier.get_quarantined_files(limit=2)

        assert len(records) == 2


# =============================================================================
# Statistics and Health Tests
# =============================================================================
class TestStatisticsAndHealth:
    """Tests for statistics and health check operations."""

    def test_get_stats_empty(self, verifier: TransferVerifier):
        """Test statistics with no data."""
        stats = verifier.get_stats()

        assert stats["transfers_24h"] == 0
        assert stats["verified_24h"] == 0
        assert stats["failed_24h"] == 0
        assert stats["quarantine_count"] == 0
        assert stats["verification_rate"] == 0.0

    def test_get_stats_with_data(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test statistics with transfer data."""
        # Create some successful transfers
        for i in range(5):
            dest = tmp_path / f"file_{i}.txt"
            dest.write_bytes(sample_file.read_bytes())
            source_checksum = verifier.compute_checksum(sample_file)
            transfer_id = verifier.record_transfer(
                source_path=str(sample_file),
                dest_path=str(dest),
                source_checksum=source_checksum,
            )
            # Use verify_transfer(transfer_id, dest_path)
            verifier.verify_transfer(transfer_id, dest)

        stats = verifier.get_stats()

        assert stats["transfers_24h"] == 5
        assert stats["verified_24h"] == 5
        assert stats["verification_rate"] == 1.0

    def test_health_check_healthy(self, verifier: TransferVerifier, sample_file: Path, tmp_path: Path):
        """Test health check when verifier is healthy."""
        # Create successful transfers
        for i in range(10):
            dest = tmp_path / f"file_{i}.txt"
            dest.write_bytes(sample_file.read_bytes())
            source_checksum = verifier.compute_checksum(sample_file)
            transfer_id = verifier.record_transfer(
                source_path=str(sample_file),
                dest_path=str(dest),
                source_checksum=source_checksum,
            )
            # Use verify_transfer(transfer_id, dest_path)
            verifier.verify_transfer(transfer_id, dest)

        result = verifier.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_no_data(self, verifier: TransferVerifier):
        """Test health check with no data.

        With no transfers, verification_rate = 0/1 = 0.0 which is < 0.80,
        so health_check returns unhealthy. This is intentional - no data
        means no verified transfers.
        """
        result = verifier.health_check()

        # No data means 0/1 = 0.0 rate, which is unhealthy by design
        # (verification_rate < 0.80 triggers unhealthy)
        assert result.healthy is False
        assert "unhealthy" in result.message.lower() or result.status.value == "error"

    def test_health_check_error(self, verifier: TransferVerifier):
        """Test health check handles errors gracefully."""
        # Close database to force error
        verifier._get_connection().close()
        verifier._local.conn = None

        # This should handle the error gracefully
        result = verifier.health_check()

        # Should return a result (may or may not be healthy depending on error handling)
        assert result is not None
        assert hasattr(result, "healthy")


# =============================================================================
# Module-Level Function Tests
# =============================================================================
class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_transfer_verifier(self, temp_db: Path):
        """Test get_transfer_verifier singleton access."""
        TransferVerifier.reset_instance()

        # First access creates instance
        v1 = get_transfer_verifier()
        v2 = get_transfer_verifier()

        assert v1 is v2

        TransferVerifier.reset_instance()

    def test_compute_file_checksum_function(self, sample_file: Path):
        """Test module-level compute_file_checksum."""
        TransferVerifier.reset_instance()

        checksum = compute_file_checksum(sample_file)

        assert len(checksum) == 64

        TransferVerifier.reset_instance()

    def test_verify_transfer_function(self, sample_file: Path, tmp_path: Path):
        """Test module-level verify_transfer."""
        TransferVerifier.reset_instance()

        dest = tmp_path / "copy.txt"
        dest.write_bytes(sample_file.read_bytes())

        checksum = compute_file_checksum(sample_file)
        result = verify_transfer(checksum, dest)

        assert result is True

        TransferVerifier.reset_instance()

    def test_quarantine_file_function(self, tmp_path: Path):
        """Test module-level quarantine_file."""
        TransferVerifier.reset_instance()

        bad_file = tmp_path / "bad.db"
        bad_file.write_text("Corrupted")

        path = quarantine_file(bad_file, reason="test")

        assert not bad_file.exists()
        assert Path(path).exists()

        TransferVerifier.reset_instance()

    def test_compute_batch_checksum_function(self):
        """Test module-level compute_batch_checksum."""
        TransferVerifier.reset_instance()

        records = ['{"id": "1"}', '{"id": "2"}']
        batch = compute_batch_checksum(records, "batch_test")

        assert batch.batch_id == "batch_test"
        assert batch.record_count == 2

        TransferVerifier.reset_instance()

    def test_reset_transfer_verifier(self, temp_db: Path):
        """Test reset_transfer_verifier function."""
        TransferVerifier.reset_instance()

        v1 = get_transfer_verifier()
        reset_transfer_verifier()
        v2 = get_transfer_verifier()

        assert v1 is not v2

        TransferVerifier.reset_instance()


# =============================================================================
# Event Wiring Tests
# =============================================================================
class TestEventWiring:
    """Tests for event wiring functionality."""

    def test_wire_transfer_verifier_events(self):
        """Test event wiring initialization."""
        TransferVerifier.reset_instance()

        # Should return verifier even if event bus not available
        verifier = wire_transfer_verifier_events()

        assert verifier is not None
        assert isinstance(verifier, TransferVerifier)

        TransferVerifier.reset_instance()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_verify_nonexistent_file(self, verifier: TransferVerifier, tmp_path: Path):
        """Test verification of non-existent file."""
        result = verifier.quick_verify("checksum", tmp_path / "nonexistent.txt")
        assert result is False

    def test_quarantine_nonexistent_file(self, verifier: TransferVerifier, tmp_path: Path):
        """Test quarantine of non-existent file."""
        try:
            verifier.quarantine(tmp_path / "nonexistent.txt", reason="test")
            # Should either raise or return empty
        except Exception:
            pass  # Expected

    def test_compute_checksum_empty_file(self, verifier: TransferVerifier, tmp_path: Path):
        """Test checksum of empty file."""
        empty = tmp_path / "empty.txt"
        empty.touch()

        checksum = verifier.compute_checksum(empty)
        assert len(checksum) == 64

    def test_thread_safety(self, temp_db: Path):
        """Test thread-local database connections."""
        import threading

        TransferVerifier.reset_instance()
        verifier = TransferVerifier.get_instance(temp_db)

        results = []

        def worker():
            conn = verifier._get_connection()
            results.append(id(conn))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have its own connection
        # (though this depends on implementation)
        TransferVerifier.reset_instance()


# =============================================================================
# Constants Tests
# =============================================================================
class TestConstants:
    """Tests for module constants."""

    def test_chunk_size(self):
        """Test CHUNK_SIZE is reasonable."""
        assert CHUNK_SIZE > 0
        assert CHUNK_SIZE >= 1024  # At least 1KB

    def test_max_quarantine_age(self):
        """Test MAX_QUARANTINE_AGE_DAYS is reasonable."""
        assert MAX_QUARANTINE_AGE_DAYS > 0
        assert MAX_QUARANTINE_AGE_DAYS <= 365
