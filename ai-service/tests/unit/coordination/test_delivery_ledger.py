"""Tests for DeliveryLedger.

Tests the persistent delivery tracking system for cluster data distribution.
December 2025: Created as part of Phase 3 infrastructure improvements.
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.delivery_ledger import (
    DeliveryLedger,
    DeliveryRecord,
    DeliveryStatus,
    get_delivery_ledger,
    reset_delivery_ledger,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_ledger.db"


@pytest.fixture
def ledger(temp_db):
    """Create a test ledger instance."""
    return DeliveryLedger(db_path=temp_db)


class TestDeliveryStatus:
    """Test DeliveryStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.TRANSFERRING.value == "transferring"
        assert DeliveryStatus.TRANSFERRED.value == "transferred"
        assert DeliveryStatus.VERIFIED.value == "verified"
        assert DeliveryStatus.FAILED.value == "failed"


class TestDeliveryRecord:
    """Test DeliveryRecord dataclass."""

    def test_default_values(self):
        """Test default record values."""
        record = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
        )
        assert record.status == DeliveryStatus.PENDING
        assert record.retry_count == 0
        assert record.max_retries == 4
        assert record.checksum == ""

    def test_transfer_duration(self):
        """Test transfer duration calculation."""
        now = time.time()
        record = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            started_at=now - 10,
            completed_at=now,
        )
        assert record.transfer_duration_seconds == pytest.approx(10.0, rel=0.1)

    def test_transfer_duration_incomplete(self):
        """Test transfer duration when incomplete."""
        record = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            started_at=time.time(),
            completed_at=0.0,
        )
        assert record.transfer_duration_seconds == 0.0

    def test_age_seconds(self):
        """Test age calculation."""
        record = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            created_at=time.time() - 60,
        )
        assert 59 < record.age_seconds < 62

    def test_is_terminal(self):
        """Test terminal state detection."""
        # Verified is terminal
        record_verified = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.VERIFIED,
        )
        assert record_verified.is_terminal is True

        # Failed is terminal
        record_failed = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.FAILED,
        )
        assert record_failed.is_terminal is True

        # Transferring is not terminal
        record_transferring = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.TRANSFERRING,
        )
        assert record_transferring.is_terminal is False

    def test_can_retry(self):
        """Test retry eligibility."""
        # Failed with retries left
        record_can_retry = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.FAILED,
            retry_count=2,
            max_retries=4,
        )
        assert record_can_retry.can_retry is True

        # Failed with max retries
        record_max_retries = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.FAILED,
            retry_count=4,
            max_retries=4,
        )
        assert record_max_retries.can_retry is False

        # Verified cannot retry
        record_verified = DeliveryRecord(
            delivery_id="test-123",
            data_type="model",
            data_path="/path/to/model.pth",
            target_node="runpod-h100",
            status=DeliveryStatus.VERIFIED,
            retry_count=0,
        )
        assert record_verified.can_retry is False

    def test_to_dict(self):
        """Test dictionary serialization."""
        record = DeliveryRecord(
            delivery_id="test-123",
            data_type="npz",
            data_path="/path/to/data.npz",
            target_node="vast-12345",
            status=DeliveryStatus.VERIFIED,
            checksum="abc123",
        )
        d = record.to_dict()

        assert d["delivery_id"] == "test-123"
        assert d["data_type"] == "npz"
        assert d["status"] == "verified"
        assert d["checksum"] == "abc123"


class TestDeliveryLedger:
    """Test DeliveryLedger functionality."""

    def test_database_creation(self, temp_db):
        """Test database is created on initialization."""
        assert not temp_db.exists()
        ledger = DeliveryLedger(db_path=temp_db)
        assert temp_db.exists()

        # Check tables exist
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "deliveries" in tables

    def test_record_delivery_started(self, ledger):
        """Test recording a new delivery."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
            checksum="sha256:abc123",
            file_size_bytes=1024 * 1024,
            transfer_method="http",
        )

        assert record.delivery_id is not None
        assert len(record.delivery_id) == 36  # UUID length
        assert record.status == DeliveryStatus.TRANSFERRING
        assert record.data_type == "model"
        assert record.target_node == "runpod-h100"
        assert record.checksum == "sha256:abc123"
        assert record.file_size_bytes == 1024 * 1024

    def test_record_delivery_transferred(self, ledger):
        """Test recording transfer completion."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )

        ledger.record_delivery_transferred(record.delivery_id)

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.status == DeliveryStatus.TRANSFERRED
        assert updated.completed_at > 0

    def test_record_delivery_verified(self, ledger):
        """Test recording verification."""
        record = ledger.record_delivery_started(
            data_type="npz",
            data_path="/data/training/hex8_2p.npz",
            target_node="vast-12345",
        )

        ledger.record_delivery_transferred(record.delivery_id)
        ledger.record_delivery_verified(record.delivery_id, verified_checksum="sha256:xyz789")

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.status == DeliveryStatus.VERIFIED
        assert updated.verified_checksum == "sha256:xyz789"

    def test_record_delivery_failed(self, ledger):
        """Test recording failure."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )

        ledger.record_delivery_failed(record.delivery_id, "Connection timeout")

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.status == DeliveryStatus.FAILED
        assert updated.error_message == "Connection timeout"
        assert updated.retry_count == 1

    def test_record_delivery_failed_no_increment(self, ledger):
        """Test recording failure without incrementing retry."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )

        ledger.record_delivery_failed(
            record.delivery_id,
            "Permanent error",
            increment_retry=False,
        )

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.retry_count == 0

    def test_record_retry_started(self, ledger):
        """Test recording retry start."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )
        ledger.record_delivery_failed(record.delivery_id, "First attempt failed")

        original_started_at = record.started_at
        time.sleep(0.1)  # Small delay

        ledger.record_retry_started(record.delivery_id)

        updated = ledger.get_delivery(record.delivery_id)
        assert updated.status == DeliveryStatus.TRANSFERRING
        assert updated.started_at > original_started_at

    def test_get_delivery(self, ledger):
        """Test getting a specific delivery."""
        record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="runpod-h100",
        )

        fetched = ledger.get_delivery(record.delivery_id)
        assert fetched is not None
        assert fetched.delivery_id == record.delivery_id
        assert fetched.data_type == "model"

    def test_get_delivery_not_found(self, ledger):
        """Test getting non-existent delivery."""
        result = ledger.get_delivery("non-existent-id")
        assert result is None

    def test_get_pending_deliveries(self, ledger):
        """Test getting pending deliveries."""
        # Create various deliveries
        pending = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/a.pth",
            target_node="node-a",
        )

        verified = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/b.pth",
            target_node="node-b",
        )
        ledger.record_delivery_transferred(verified.delivery_id)
        ledger.record_delivery_verified(verified.delivery_id)

        failed = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/c.pth",
            target_node="node-c",
        )
        ledger.record_delivery_failed(failed.delivery_id, "Error")

        pending_list = ledger.get_pending_deliveries()

        # Only the transferring one should be pending
        assert len(pending_list) == 1
        assert pending_list[0].delivery_id == pending.delivery_id

    def test_get_retryable_deliveries(self, ledger):
        """Test getting retryable deliveries."""
        # Failed with retries left
        record1 = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/a.pth",
            target_node="node-a",
        )
        ledger.record_delivery_failed(record1.delivery_id, "Error 1")

        # Failed with max retries reached
        record2 = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/b.pth",
            target_node="node-b",
            max_retries=1,
        )
        ledger.record_delivery_failed(record2.delivery_id, "Error 2")

        retryable = ledger.get_retryable_deliveries()

        # Only record1 should be retryable
        assert len(retryable) == 1
        assert retryable[0].delivery_id == record1.delivery_id

    def test_get_deliveries_for_node(self, ledger):
        """Test getting deliveries for a specific node."""
        # Create deliveries for different nodes
        for i in range(3):
            ledger.record_delivery_started(
                data_type="model",
                data_path=f"/models/{i}.pth",
                target_node="target-node",
            )
        ledger.record_delivery_started(
            data_type="model",
            data_path="/models/other.pth",
            target_node="other-node",
        )

        deliveries = ledger.get_deliveries_for_node("target-node")

        assert len(deliveries) == 3
        assert all(d.target_node == "target-node" for d in deliveries)

    def test_get_node_delivery_status(self, ledger):
        """Test getting delivery status for a node."""
        node_id = "test-node"

        # Create some deliveries with different statuses
        for i in range(5):
            record = ledger.record_delivery_started(
                data_type="model",
                data_path=f"/models/{i}.pth",
                target_node=node_id,
            )
            ledger.record_delivery_transferred(record.delivery_id)
            ledger.record_delivery_verified(record.delivery_id)

        failed_record = ledger.record_delivery_started(
            data_type="model",
            data_path="/models/fail.pth",
            target_node=node_id,
        )
        ledger.record_delivery_failed(failed_record.delivery_id, "Error")

        status = ledger.get_node_delivery_status(node_id)

        assert status["node_id"] == node_id
        assert status["total_verified"] == 5
        assert status["failed_24h"] == 1
        assert status["failure_rate_24h"] == pytest.approx(1/6, rel=0.01)

    def test_get_overall_status(self, ledger):
        """Test getting overall delivery status."""
        # Create various deliveries
        for i in range(10):
            record = ledger.record_delivery_started(
                data_type="model",
                data_path=f"/models/{i}.pth",
                target_node=f"node-{i % 3}",
            )
            if i < 7:
                ledger.record_delivery_transferred(record.delivery_id)
                ledger.record_delivery_verified(record.delivery_id)
            else:
                ledger.record_delivery_failed(record.delivery_id, "Error")

        status = ledger.get_overall_status()

        assert status["total_deliveries"] == 10
        assert status["verified"] == 7
        assert status["failed"] == 3
        assert status["success_rate"] == 0.7
        assert status["unique_nodes"] == 3

    def test_cleanup_old_records(self, ledger):
        """Test cleanup of old records."""
        # Create an old record (simulate by directly inserting)
        old_time = time.time() - (40 * 86400)  # 40 days ago

        conn = sqlite3.connect(str(ledger.db_path))
        conn.execute("""
            INSERT INTO deliveries (
                delivery_id, data_type, data_path, target_node, status,
                retry_count, max_retries, checksum, verified_checksum,
                file_size_bytes, transfer_method, created_at, updated_at,
                started_at, completed_at, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "old-record", "model", "/old/path", "old-node",
            "verified", 0, 4, "", "", 0, "http",
            old_time, old_time, old_time, old_time, "",
        ))
        conn.commit()
        conn.close()

        # Create a recent record
        recent = ledger.record_delivery_started(
            data_type="model",
            data_path="/recent/path",
            target_node="recent-node",
        )
        ledger.record_delivery_transferred(recent.delivery_id)
        ledger.record_delivery_verified(recent.delivery_id)

        # Run cleanup
        deleted = ledger.cleanup_old_records(max_age_days=30)

        assert deleted == 1

        # Verify old record is gone
        assert ledger.get_delivery("old-record") is None

        # Verify recent record still exists
        assert ledger.get_delivery(recent.delivery_id) is not None


class TestSingletonPattern:
    """Test singleton pattern for delivery ledger."""

    def test_get_delivery_ledger_returns_same_instance(self):
        """Test that get_delivery_ledger returns singleton."""
        reset_delivery_ledger()

        with patch.object(DeliveryLedger, 'DEFAULT_DB_PATH', Path(tempfile.mkdtemp()) / "test.db"):
            ledger1 = get_delivery_ledger()
            ledger2 = get_delivery_ledger()

            assert ledger1 is ledger2

        reset_delivery_ledger()

    def test_reset_delivery_ledger(self):
        """Test resetting the singleton instance."""
        reset_delivery_ledger()

        with patch.object(DeliveryLedger, 'DEFAULT_DB_PATH', Path(tempfile.mkdtemp()) / "test.db"):
            ledger1 = get_delivery_ledger()
            reset_delivery_ledger()

            with patch.object(DeliveryLedger, 'DEFAULT_DB_PATH', Path(tempfile.mkdtemp()) / "test2.db"):
                ledger2 = get_delivery_ledger()

                assert ledger1 is not ledger2

        reset_delivery_ledger()


class TestPersistence:
    """Test persistence across ledger instances."""

    def test_records_persist_across_instances(self, temp_db):
        """Test that records persist when ledger is recreated."""
        # Create ledger and add record
        ledger1 = DeliveryLedger(db_path=temp_db)
        record = ledger1.record_delivery_started(
            data_type="model",
            data_path="/models/test.pth",
            target_node="test-node",
        )
        delivery_id = record.delivery_id

        # Create new ledger instance pointing to same DB
        ledger2 = DeliveryLedger(db_path=temp_db)

        # Record should still be retrievable
        fetched = ledger2.get_delivery(delivery_id)
        assert fetched is not None
        assert fetched.data_type == "model"
        assert fetched.target_node == "test-node"
