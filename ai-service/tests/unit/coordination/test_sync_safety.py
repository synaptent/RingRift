"""Unit tests for sync_safety.py (December 2025).

Tests the unified sync safety facade module that consolidates:
- sync_durability.py (WAL + DLQ)
- sync_integrity.py (Checksum validation)
- sync_stall_handler.py (Stall detection/failover)
- sync_bloom_filter.py (P2P Bloom filter)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest


class TestSyncSafetyExports:
    """Test that all expected exports are available from sync_safety."""

    def test_durability_exports(self):
        """Test durability module exports are available."""
        from app.coordination.sync_safety import (
            SyncWAL,
            DeadLetterQueue,
            SyncWALEntry,
            DeadLetterEntry,
            WALStats,
            DLQStats,
            SyncStatus,
            get_sync_wal,
            get_dlq,
            reset_instances,
        )

        assert SyncWAL is not None
        assert DeadLetterQueue is not None
        assert callable(get_sync_wal)
        assert callable(get_dlq)
        assert callable(reset_instances)

    def test_integrity_exports(self):
        """Test integrity module exports are available."""
        from app.coordination.sync_safety import (
            DEFAULT_CHUNK_SIZE,
            LARGE_CHUNK_SIZE,
            IntegrityCheckResult,
            IntegrityReport,
            check_sqlite_integrity,
            compute_db_checksum,
            compute_file_checksum,
            verify_checksum,
            verify_sync_integrity,
        )

        assert isinstance(DEFAULT_CHUNK_SIZE, int)
        assert isinstance(LARGE_CHUNK_SIZE, int)
        assert callable(check_sqlite_integrity)
        assert callable(compute_file_checksum)

    def test_stall_handler_exports(self):
        """Test stall handler exports are available."""
        from app.coordination.sync_safety import (
            SyncStallHandler,
            get_stall_handler,
            reset_stall_handler,
        )

        assert SyncStallHandler is not None
        assert callable(get_stall_handler)
        assert callable(reset_stall_handler)

    def test_bloom_filter_exports(self):
        """Test bloom filter exports are available."""
        from app.coordination.sync_safety import (
            SyncBloomFilter,
            BloomFilterStats,
            BloomFilter,
            DEFAULT_SIZE,
            DEFAULT_HASH_COUNT,
            DEFAULT_FALSE_POSITIVE_RATE,
            create_game_id_filter,
            create_model_hash_filter,
            create_event_dedup_filter,
        )

        assert SyncBloomFilter is not None
        assert BloomFilter is SyncBloomFilter  # Should be an alias
        assert isinstance(DEFAULT_SIZE, int)
        assert callable(create_game_id_filter)


class TestSyncWAL:
    """Test SyncWAL functionality."""

    @pytest.fixture
    def wal(self):
        """Create a fresh WAL instance."""
        from app.coordination.sync_safety import SyncWAL

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_wal.db"
            wal_instance = SyncWAL(db_path=db_path)
            yield wal_instance
            wal_instance.close()

    def test_append_entry(self, wal):
        """Test appending a WAL entry."""
        entry_id = wal.append(
            game_id="game-123",
            source="node-1",
            target="coordinator",
            data={"test": "data"},
        )

        assert entry_id is not None
        assert isinstance(entry_id, (int, str))

    def test_mark_complete(self, wal):
        """Test marking an entry as complete."""
        entry_id = wal.append(
            game_id="game-456",
            source="node-2",
            target="coordinator",
            data={},
        )

        result = wal.mark_complete(entry_id)
        assert result is True

    def test_get_pending_entries(self, wal):
        """Test getting pending entries."""
        # Add multiple entries
        wal.append(game_id="g1", source="n1", target="coord", data={})
        wal.append(game_id="g2", source="n2", target="coord", data={})

        pending = wal.get_pending()
        assert isinstance(pending, list)
        assert len(pending) >= 2


class TestDeadLetterQueue:
    """Test DeadLetterQueue functionality."""

    @pytest.fixture
    def dlq(self):
        """Create a fresh DLQ instance."""
        from app.coordination.sync_safety import DeadLetterQueue

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dlq.db"
            dlq_instance = DeadLetterQueue(db_path=db_path)
            yield dlq_instance
            dlq_instance.close()

    def test_add_entry(self, dlq):
        """Test adding a DLQ entry."""
        entry_id = dlq.add(
            game_id="failed-game",
            source="node-1",
            target="coordinator",
            error="Connection timeout",
            retry_count=3,
        )

        assert entry_id is not None

    def test_get_retriable_entries(self, dlq):
        """Test getting retriable entries."""
        dlq.add(
            game_id="retry-game",
            source="node-1",
            target="coord",
            error="Timeout",
            retry_count=1,
        )

        retriable = dlq.get_retriable(max_retries=5)
        assert isinstance(retriable, list)


class TestSyncIntegrity:
    """Test sync integrity functions."""

    def test_compute_file_checksum(self):
        """Test computing file checksum."""
        from app.coordination.sync_safety import compute_file_checksum

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test data for checksum")
            tmp_path = Path(f.name)

        try:
            checksum = compute_file_checksum(tmp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA256 hex digest
        finally:
            tmp_path.unlink()

    def test_compute_file_checksum_deterministic(self):
        """Test that checksum is deterministic."""
        from app.coordination.sync_safety import compute_file_checksum

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"deterministic test data")
            tmp_path = Path(f.name)

        try:
            checksum1 = compute_file_checksum(tmp_path)
            checksum2 = compute_file_checksum(tmp_path)
            assert checksum1 == checksum2
        finally:
            tmp_path.unlink()

    def test_check_sqlite_integrity(self):
        """Test SQLite integrity check."""
        import sqlite3
        from app.coordination.sync_safety import check_sqlite_integrity

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            tmp_path = Path(f.name)

        try:
            # Create a valid SQLite database
            conn = sqlite3.connect(str(tmp_path))
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            conn.commit()
            conn.close()

            is_valid, errors = check_sqlite_integrity(tmp_path)
            assert is_valid is True
            assert len(errors) == 0
        finally:
            tmp_path.unlink()


class TestSyncStallHandler:
    """Test SyncStallHandler functionality."""

    @pytest.fixture
    def handler(self):
        """Create a fresh stall handler."""
        from app.coordination.sync_safety import reset_stall_handler, get_stall_handler

        reset_stall_handler()
        return get_stall_handler()

    def test_check_stall_not_stalled(self, handler):
        """Test check_stall returns False for recent operation."""
        now = time.time()
        is_stalled = handler.check_stall(
            sync_id="sync-1",
            started_at=now - 30,  # 30 seconds ago
            timeout=600,  # 10 minute timeout
        )

        assert is_stalled is False

    def test_check_stall_is_stalled(self, handler):
        """Test check_stall returns True for stalled operation."""
        now = time.time()
        is_stalled = handler.check_stall(
            sync_id="sync-2",
            started_at=now - 700,  # 11+ minutes ago
            timeout=600,  # 10 minute timeout
        )

        assert is_stalled is True

    def test_record_stall(self, handler):
        """Test recording a stall."""
        handler.record_stall(host="node-5", sync_id="stalled-sync")

        # Should be able to record without error
        # The penalty tracking is internal

    def test_get_alternative_source(self, handler):
        """Test getting alternative source."""
        # Record a stall to penalize node-5
        handler.record_stall(host="node-5", sync_id="sync-1")

        alt = handler.get_alternative_source(
            exclude=["node-5"],
            all_sources=["node-1", "node-2", "node-3", "node-5"],
        )

        assert alt in ["node-1", "node-2", "node-3"]
        assert alt != "node-5"


class TestSyncBloomFilter:
    """Test SyncBloomFilter functionality."""

    def test_create_and_add(self):
        """Test creating a bloom filter and adding items."""
        from app.coordination.sync_safety import SyncBloomFilter

        bf = SyncBloomFilter(expected_items=100)
        bf.add("item1")
        bf.add("item2")

        assert "item1" in bf
        assert "item2" in bf

    def test_membership_test(self):
        """Test membership testing."""
        from app.coordination.sync_safety import SyncBloomFilter

        bf = SyncBloomFilter(expected_items=100)
        bf.add("present")

        assert "present" in bf
        # Note: False positives are possible, but "not_present" should usually not be in bf

    def test_serialization(self):
        """Test serializing and deserializing bloom filter."""
        from app.coordination.sync_safety import SyncBloomFilter

        bf1 = SyncBloomFilter(expected_items=100)
        bf1.add("item1")
        bf1.add("item2")

        # Serialize
        data = bf1.to_bytes()
        assert isinstance(data, bytes)

        # Deserialize
        bf2 = SyncBloomFilter.from_bytes(data)
        assert "item1" in bf2
        assert "item2" in bf2

    def test_create_game_id_filter(self):
        """Test factory function for game ID filter."""
        from app.coordination.sync_safety import create_game_id_filter

        bf = create_game_id_filter(expected_games=1000)
        assert bf is not None

        # Should be able to add game IDs
        bf.add("game-123")
        assert "game-123" in bf


class TestSingletonPattern:
    """Test singleton pattern for singletons."""

    def test_get_sync_wal_singleton(self):
        """Test get_sync_wal returns singleton."""
        from app.coordination.sync_safety import get_sync_wal, reset_instances

        reset_instances()

        wal1 = get_sync_wal()
        wal2 = get_sync_wal()

        assert wal1 is wal2

        reset_instances()

    def test_get_dlq_singleton(self):
        """Test get_dlq returns singleton."""
        from app.coordination.sync_safety import get_dlq, reset_instances

        reset_instances()

        dlq1 = get_dlq()
        dlq2 = get_dlq()

        assert dlq1 is dlq2

        reset_instances()

    def test_get_stall_handler_singleton(self):
        """Test get_stall_handler returns singleton."""
        from app.coordination.sync_safety import get_stall_handler, reset_stall_handler

        reset_stall_handler()

        h1 = get_stall_handler()
        h2 = get_stall_handler()

        assert h1 is h2

        reset_stall_handler()


class TestIntegrityReport:
    """Test IntegrityReport dataclass."""

    def test_integrity_report_fields(self):
        """Test IntegrityReport has expected fields."""
        from app.coordination.sync_safety import IntegrityReport

        report = IntegrityReport(
            is_valid=True,
            source_checksum="abc123",
            target_checksum="abc123",
            errors=[],
        )

        assert report.is_valid is True
        assert report.source_checksum == "abc123"
        assert report.errors == []


class TestBloomFilterStats:
    """Test BloomFilterStats functionality."""

    def test_bloom_filter_stats(self):
        """Test getting bloom filter statistics."""
        from app.coordination.sync_safety import SyncBloomFilter

        bf = SyncBloomFilter(expected_items=100)
        for i in range(50):
            bf.add(f"item-{i}")

        stats = bf.stats
        assert hasattr(stats, "size_bytes") or isinstance(stats, dict)


class TestVerifySyncIntegrity:
    """Test verify_sync_integrity function."""

    def test_verify_identical_files(self):
        """Test verifying integrity of identical files."""
        from app.coordination.sync_safety import verify_sync_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            target = Path(tmpdir) / "target.txt"

            content = b"identical content"
            source.write_bytes(content)
            target.write_bytes(content)

            report = verify_sync_integrity(source, target)

            assert report.is_valid is True
            assert report.source_checksum == report.target_checksum

    def test_verify_different_files(self):
        """Test verifying integrity of different files."""
        from app.coordination.sync_safety import verify_sync_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            target = Path(tmpdir) / "target.txt"

            source.write_bytes(b"source content")
            target.write_bytes(b"different content")

            report = verify_sync_integrity(source, target)

            assert report.is_valid is False
            assert report.source_checksum != report.target_checksum
