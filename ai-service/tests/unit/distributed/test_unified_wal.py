"""Tests for unified Write-Ahead Log (WAL) implementation.

These tests verify the crash-safe data ingestion infrastructure, including:
1. WALEntry and WALEntryType/WALEntryStatus enums
2. WAL entry append and retrieval
3. Entry status transitions (pending -> synced -> processed)
4. Entry status transition (pending -> failed)
5. Crash recovery simulation (partial writes, idempotent replay)
6. Checkpoint creation and compaction
7. SQLite persistence and thread safety
8. ConnectionPool behavior
9. Backward compatibility wrappers (WriteAheadLog, IngestionWAL)

CRITICAL: This WAL ensures data integrity for game ingestion - failures here
can cause data loss during crashes.
"""

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.unified_wal import (
    ConnectionPool,
    IngestionWAL,
    UnifiedWAL,
    WALCheckpoint,
    WALEntry,
    WALEntryStatus,
    WALEntryType,
    WALStats,
    WriteAheadLog,
    get_unified_wal,
    reset_wal_instance,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path for testing."""
    return tmp_path / "test_wal.db"


@pytest.fixture
def wal(temp_db_path: Path) -> UnifiedWAL:
    """Create a UnifiedWAL instance with temporary database."""
    return UnifiedWAL(
        db_path=temp_db_path,
        max_pending=100,
        checkpoint_interval=10,
        auto_compact=False,
        max_retries=3,
        use_connection_pool=True,
    )


@pytest.fixture
def wal_no_pool(temp_db_path: Path) -> UnifiedWAL:
    """Create a UnifiedWAL instance without connection pooling."""
    return UnifiedWAL(
        db_path=temp_db_path,
        max_pending=100,
        checkpoint_interval=1000,
        auto_compact=False,
        use_connection_pool=False,
    )


# =============================================================================
# WAL ENTRY TYPE AND STATUS TESTS
# =============================================================================


class TestWALEntryType:
    """Tests for WALEntryType enum."""

    def test_entry_type_values(self):
        """WALEntryType should have correct string values."""
        assert WALEntryType.SYNC.value == "sync"
        assert WALEntryType.INGESTION.value == "ingestion"
        assert WALEntryType.ELO_SYNC.value == "elo_sync"
        assert WALEntryType.MODEL_SYNC.value == "model_sync"

    def test_entry_type_is_string_enum(self):
        """WALEntryType should inherit from str for serialization."""
        assert isinstance(WALEntryType.SYNC, str)
        assert WALEntryType.SYNC == "sync"

    def test_entry_type_enum_members(self):
        """WALEntryType should have exactly 4 members."""
        assert len(WALEntryType) == 4


class TestWALEntryStatus:
    """Tests for WALEntryStatus enum."""

    def test_entry_status_values(self):
        """WALEntryStatus should have correct string values."""
        assert WALEntryStatus.PENDING.value == "pending"
        assert WALEntryStatus.SYNCED.value == "synced"
        assert WALEntryStatus.PROCESSED.value == "processed"
        assert WALEntryStatus.FAILED.value == "failed"

    def test_entry_status_is_string_enum(self):
        """WALEntryStatus should inherit from str for serialization."""
        assert isinstance(WALEntryStatus.PENDING, str)
        assert WALEntryStatus.PENDING == "pending"

    def test_entry_status_enum_members(self):
        """WALEntryStatus should have exactly 4 members."""
        assert len(WALEntryStatus) == 4


# =============================================================================
# WAL ENTRY DATACLASS TESTS
# =============================================================================


class TestWALEntry:
    """Tests for WALEntry dataclass."""

    def test_entry_creation_with_required_fields(self):
        """WALEntry should be created with all required fields."""
        entry = WALEntry(
            entry_id=1,
            entry_type=WALEntryType.SYNC,
            game_id="game123",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="abc123",
        )
        assert entry.entry_id == 1
        assert entry.entry_type == WALEntryType.SYNC
        assert entry.game_id == "game123"
        assert entry.source_host == "host1"
        assert entry.source_db == "db.sqlite"
        assert entry.data_hash == "abc123"
        assert entry.status == WALEntryStatus.PENDING
        assert entry.retry_count == 0

    def test_entry_data_property_parses_json(self):
        """data property should parse data_json if present."""
        game_data = {"moves": [1, 2, 3], "winner": 1}
        entry = WALEntry(
            entry_id=1,
            entry_type=WALEntryType.INGESTION,
            game_id="game123",
            source_host="host1",
            source_db="",
            data_hash="hash",
            data_json=json.dumps(game_data),
        )
        assert entry.data == game_data

    def test_entry_data_property_returns_none_for_empty(self):
        """data property should return None if data_json is empty."""
        entry = WALEntry(
            entry_id=1,
            entry_type=WALEntryType.SYNC,
            game_id="game123",
            source_host="host1",
            source_db="",
            data_hash="hash",
            data_json=None,
        )
        assert entry.data is None

    def test_entry_data_property_handles_invalid_json(self):
        """data property should return None for invalid JSON."""
        entry = WALEntry(
            entry_id=1,
            entry_type=WALEntryType.INGESTION,
            game_id="game123",
            source_host="host1",
            source_db="",
            data_hash="hash",
            data_json="not valid json {{{",
        )
        assert entry.data is None

    def test_entry_to_dict(self):
        """to_dict should return all fields as dictionary."""
        entry = WALEntry(
            entry_id=42,
            entry_type=WALEntryType.SYNC,
            game_id="game123",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="abc123",
            status=WALEntryStatus.PROCESSED,
            created_at=1000.0,
            updated_at=1100.0,
            retry_count=2,
            error_message="test error",
        )
        d = entry.to_dict()
        assert d["entry_id"] == 42
        assert d["entry_type"] == "sync"
        assert d["game_id"] == "game123"
        assert d["status"] == "processed"
        assert d["retry_count"] == 2
        assert d["error_message"] == "test error"


# =============================================================================
# CONNECTION POOL TESTS
# =============================================================================


class TestConnectionPool:
    """Tests for ConnectionPool thread-local connection management."""

    def test_pool_creates_connection(self, temp_db_path: Path):
        """Connection pool should create new connection on first access."""
        pool = ConnectionPool(temp_db_path)
        with pool.get_connection() as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

    def test_pool_reuses_connection_in_same_thread(self, temp_db_path: Path):
        """Connection pool should reuse connection within same thread."""
        pool = ConnectionPool(temp_db_path)
        with pool.get_connection() as conn1:
            pass
        with pool.get_connection() as conn2:
            pass
        # Same thread should get same connection
        assert pool.get_stats()["connections_created"] == 1
        assert pool.get_stats()["connections_reused"] >= 1

    def test_pool_creates_separate_connections_per_thread(self, temp_db_path: Path):
        """Connection pool should create separate connections for different threads."""
        pool = ConnectionPool(temp_db_path)
        connections = []

        def thread_func():
            with pool.get_connection() as conn:
                connections.append(id(conn))

        threads = [threading.Thread(target=thread_func) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have created its own connection
        assert pool.get_stats()["connections_created"] == 3

    def test_pool_get_stats(self, temp_db_path: Path):
        """get_stats should return connection statistics."""
        pool = ConnectionPool(temp_db_path)
        with pool.get_connection():
            pass
        with pool.get_connection():
            pass

        stats = pool.get_stats()
        assert "connections_created" in stats
        assert "connections_reused" in stats
        assert "reuse_ratio" in stats

    def test_pool_close_all(self, temp_db_path: Path):
        """close_all should close the thread's connection."""
        pool = ConnectionPool(temp_db_path)
        with pool.get_connection():
            pass
        pool.close_all()
        # Next access should create new connection
        initial_created = pool.get_stats()["connections_created"]
        with pool.get_connection():
            pass
        assert pool.get_stats()["connections_created"] == initial_created + 1


# =============================================================================
# WAL INITIALIZATION TESTS
# =============================================================================


class TestWALInitialization:
    """Tests for WAL initialization and database schema."""

    def test_wal_creates_database_file(self, temp_db_path: Path):
        """WAL initialization should create database file."""
        assert not temp_db_path.exists()
        UnifiedWAL(db_path=temp_db_path)
        assert temp_db_path.exists()

    def test_wal_creates_parent_directories(self, tmp_path: Path):
        """WAL should create parent directories if they don't exist."""
        nested_path = tmp_path / "a" / "b" / "c" / "wal.db"
        assert not nested_path.parent.exists()
        UnifiedWAL(db_path=nested_path)
        assert nested_path.exists()

    def test_wal_creates_required_tables(self, temp_db_path: Path):
        """WAL should create all required tables."""
        UnifiedWAL(db_path=temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "wal_entries" in tables
        assert "wal_checkpoints" in tables
        assert "wal_metadata" in tables

    def test_wal_creates_indexes(self, temp_db_path: Path):
        """WAL should create required indexes."""
        UnifiedWAL(db_path=temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_wal_status_type" in indexes
        assert "idx_wal_game_id" in indexes
        assert "idx_wal_dedup" in indexes

    def test_wal_with_connection_pool(self, temp_db_path: Path):
        """WAL with connection pool should work correctly."""
        wal = UnifiedWAL(db_path=temp_db_path, use_connection_pool=True)
        assert wal._conn_pool is not None

    def test_wal_without_connection_pool(self, temp_db_path: Path):
        """WAL without connection pool should work correctly."""
        wal = UnifiedWAL(db_path=temp_db_path, use_connection_pool=False)
        assert wal._conn_pool is None


# =============================================================================
# SYNC ENTRY TESTS
# =============================================================================


class TestSyncEntryAppend:
    """Tests for appending sync entries."""

    def test_append_sync_entry_returns_id(self, wal: UnifiedWAL):
        """append_sync_entry should return entry ID."""
        entry_id = wal.append_sync_entry(
            game_id="game1",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="hash123",
        )
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_append_sync_entry_creates_pending_entry(self, wal: UnifiedWAL):
        """Appended sync entry should be in pending status."""
        wal.append_sync_entry(
            game_id="game1",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="hash123",
        )
        pending = wal.get_pending_sync_entries()
        assert len(pending) == 1
        assert pending[0].game_id == "game1"
        assert pending[0].status == WALEntryStatus.PENDING
        assert pending[0].entry_type == WALEntryType.SYNC

    def test_append_sync_entry_is_idempotent(self, wal: UnifiedWAL):
        """Duplicate sync entries should return same ID."""
        id1 = wal.append_sync_entry(
            game_id="game1",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="hash123",
        )
        id2 = wal.append_sync_entry(
            game_id="game1",
            source_host="host1",
            source_db="db.sqlite",
            data_hash="hash123",
        )
        assert id1 == id2

        # Should only have one entry
        pending = wal.get_pending_sync_entries()
        assert len(pending) == 1

    def test_append_sync_batch(self, wal: UnifiedWAL):
        """append_sync_batch should add multiple entries."""
        entries = [
            ("game1", "host1", "db.sqlite", "hash1"),
            ("game2", "host1", "db.sqlite", "hash2"),
            ("game3", "host2", "db.sqlite", "hash3"),
        ]
        added = wal.append_sync_batch(entries)
        assert added == 3

        pending = wal.get_pending_sync_entries()
        assert len(pending) == 3

    def test_append_sync_batch_skips_duplicates(self, wal: UnifiedWAL):
        """append_sync_batch should skip duplicate entries."""
        entries = [
            ("game1", "host1", "db.sqlite", "hash1"),
            ("game2", "host1", "db.sqlite", "hash2"),
        ]
        wal.append_sync_batch(entries)

        # Try to add duplicates plus one new
        entries_with_dup = [
            ("game1", "host1", "db.sqlite", "hash1"),  # duplicate
            ("game3", "host1", "db.sqlite", "hash3"),  # new
        ]
        added = wal.append_sync_batch(entries_with_dup)
        assert added == 1  # Only game3 added

    def test_append_sync_batch_empty(self, wal: UnifiedWAL):
        """append_sync_batch should handle empty list."""
        added = wal.append_sync_batch([])
        assert added == 0


# =============================================================================
# INGESTION ENTRY TESTS
# =============================================================================


class TestIngestionEntryAppend:
    """Tests for appending ingestion entries."""

    def test_append_ingestion_entry_returns_id(self, wal: UnifiedWAL):
        """append_ingestion_entry should return entry ID."""
        game_data = {"moves": [1, 2, 3], "winner": 1}
        entry_id = wal.append_ingestion_entry(
            game_id="game1",
            game_data=game_data,
            source_host="host1",
        )
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_append_ingestion_entry_stores_data(self, wal: UnifiedWAL):
        """Appended ingestion entry should store full game data."""
        game_data = {"moves": [1, 2, 3], "winner": 1}
        wal.append_ingestion_entry(
            game_id="game1",
            game_data=game_data,
            source_host="host1",
        )
        pending = wal.get_pending_ingestion_entries()
        assert len(pending) == 1
        assert pending[0].data == game_data
        assert pending[0].entry_type == WALEntryType.INGESTION

    def test_append_ingestion_entry_is_idempotent(self, wal: UnifiedWAL):
        """Duplicate ingestion entries should return same ID."""
        game_data = {"moves": [1, 2, 3], "winner": 1}
        id1 = wal.append_ingestion_entry(
            game_id="game1",
            game_data=game_data,
            source_host="host1",
        )
        id2 = wal.append_ingestion_entry(
            game_id="game1",
            game_data=game_data,
            source_host="host1",
        )
        assert id1 == id2

    def test_append_ingestion_entry_respects_max_pending(self, temp_db_path: Path):
        """Should raise when max_pending exceeded."""
        wal = UnifiedWAL(db_path=temp_db_path, max_pending=2)

        wal.append_ingestion_entry("game1", {"data": 1}, "host1")
        wal.append_ingestion_entry("game2", {"data": 2}, "host1")

        with pytest.raises(RuntimeError, match="WAL full"):
            wal.append_ingestion_entry("game3", {"data": 3}, "host1")

    def test_append_ingestion_batch(self, wal: UnifiedWAL):
        """append_ingestion_batch should add multiple entries."""
        games = [
            ("game1", {"data": 1}),
            ("game2", {"data": 2}),
            ("game3", {"data": 3}),
        ]
        entry_ids = wal.append_ingestion_batch(games, source_host="host1")
        assert len(entry_ids) == 3
        assert all(id > 0 for id in entry_ids)

        pending = wal.get_pending_ingestion_entries()
        assert len(pending) == 3

    def test_append_ingestion_batch_empty(self, wal: UnifiedWAL):
        """append_ingestion_batch should handle empty list."""
        entry_ids = wal.append_ingestion_batch([])
        assert entry_ids == []


# =============================================================================
# STATUS TRANSITION TESTS
# =============================================================================


class TestStatusTransitions:
    """Tests for entry status transitions."""

    def test_mark_synced_from_pending(self, wal: UnifiedWAL):
        """mark_synced should transition pending entries to synced."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        updated = wal.mark_synced([entry_id])
        assert updated == 1

        # Verify status changed
        synced = wal.get_synced_unconfirmed()
        assert len(synced) == 1
        assert synced[0].status == WALEntryStatus.SYNCED

    def test_mark_synced_ignores_non_pending(self, wal: UnifiedWAL):
        """mark_synced should not update non-pending entries."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        wal.mark_processed([entry_id])

        # Try to mark already-processed as synced
        updated = wal.mark_synced([entry_id])
        assert updated == 0

    def test_mark_processed_from_pending(self, wal: UnifiedWAL):
        """mark_processed should transition pending entries to processed."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        updated = wal.mark_processed([entry_id])
        assert updated == 1

        # Entry should no longer be pending
        pending = wal.get_pending_sync_entries()
        assert len(pending) == 0

    def test_mark_processed_from_synced(self, wal: UnifiedWAL):
        """mark_processed should transition synced entries to processed."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        wal.mark_synced([entry_id])
        updated = wal.mark_processed([entry_id])
        assert updated == 1

    def test_mark_failed(self, wal: UnifiedWAL):
        """mark_failed should transition entries to failed with error message."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        updated = wal.mark_failed([entry_id], error_message="Test failure")
        assert updated == 1

        failed = wal.get_failed_entries()
        assert len(failed) == 1
        assert failed[0].status == WALEntryStatus.FAILED
        assert failed[0].error_message == "Test failure"
        assert failed[0].retry_count == 1

    def test_mark_failed_increments_retry_count(self, wal: UnifiedWAL):
        """mark_failed should increment retry_count."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        wal.mark_failed([entry_id], error_message="First failure")
        wal.mark_failed([entry_id], error_message="Second failure")

        failed = wal.get_failed_entries()
        assert failed[0].retry_count == 2

    def test_mark_operations_with_empty_list(self, wal: UnifiedWAL):
        """Status marking functions should handle empty lists."""
        assert wal.mark_synced([]) == 0
        assert wal.mark_processed([]) == 0
        assert wal.mark_failed([]) == 0

    def test_increment_retry_returns_true_when_under_limit(self, wal: UnifiedWAL):
        """increment_retry should return True when under max_retries."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")
        assert wal.increment_retry(entry_id) is True
        assert wal.increment_retry(entry_id) is True

    def test_increment_retry_returns_false_and_marks_failed_at_limit(self, wal: UnifiedWAL):
        """increment_retry should return False and mark failed at max_retries."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash")

        # max_retries is 3 in fixture
        assert wal.increment_retry(entry_id) is True  # 1
        assert wal.increment_retry(entry_id) is True  # 2
        assert wal.increment_retry(entry_id) is False  # 3 - exceeds limit

        # Should now be in failed status
        failed = wal.get_failed_entries()
        assert len(failed) == 1


# =============================================================================
# QUERY OPERATION TESTS
# =============================================================================


class TestQueryOperations:
    """Tests for WAL query operations."""

    def test_get_pending_entries_all_types(self, wal: UnifiedWAL):
        """get_pending_entries without type filter should return all types."""
        wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.append_ingestion_entry("game2", {"data": 1}, "host1")

        pending = wal.get_pending_entries()
        assert len(pending) == 2

    def test_get_pending_entries_with_type_filter(self, wal: UnifiedWAL):
        """get_pending_entries with type filter should only return that type."""
        wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.append_ingestion_entry("game2", {"data": 1}, "host1")

        sync_pending = wal.get_pending_entries(entry_type=WALEntryType.SYNC)
        assert len(sync_pending) == 1
        assert sync_pending[0].entry_type == WALEntryType.SYNC

    def test_get_pending_entries_with_limit(self, wal: UnifiedWAL):
        """get_pending_entries should respect limit parameter."""
        for i in range(5):
            wal.append_sync_entry(f"game{i}", "host1", "db", f"hash{i}")

        pending = wal.get_pending_entries(limit=2)
        assert len(pending) == 2

    def test_get_pending_entries_ordered_by_created_at(self, wal: UnifiedWAL):
        """get_pending_entries should return entries ordered by creation time."""
        wal.append_sync_entry("game1", "host1", "db", "hash1")
        time.sleep(0.01)
        wal.append_sync_entry("game2", "host1", "db", "hash2")
        time.sleep(0.01)
        wal.append_sync_entry("game3", "host1", "db", "hash3")

        pending = wal.get_pending_entries()
        assert pending[0].game_id == "game1"
        assert pending[1].game_id == "game2"
        assert pending[2].game_id == "game3"

    def test_get_synced_unconfirmed(self, wal: UnifiedWAL):
        """get_synced_unconfirmed should return only synced entries."""
        id1 = wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.append_sync_entry("game2", "host1", "db", "hash2")

        wal.mark_synced([id1])

        synced = wal.get_synced_unconfirmed()
        assert len(synced) == 1
        assert synced[0].game_id == "game1"

    def test_get_failed_entries(self, wal: UnifiedWAL):
        """get_failed_entries should return failed entries ordered by updated_at."""
        id1 = wal.append_sync_entry("game1", "host1", "db", "hash1")
        id2 = wal.append_sync_entry("game2", "host1", "db", "hash2")

        wal.mark_failed([id1], "Error 1")
        time.sleep(0.01)
        wal.mark_failed([id2], "Error 2")

        failed = wal.get_failed_entries()
        assert len(failed) == 2
        # Ordered by updated_at DESC, so game2 should be first
        assert failed[0].game_id == "game2"


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestWALStatistics:
    """Tests for WAL statistics."""

    def test_get_stats_empty_wal(self, wal: UnifiedWAL):
        """get_stats should return zeros for empty WAL."""
        stats = wal.get_stats()
        assert isinstance(stats, WALStats)
        assert stats.total_entries == 0
        assert stats.pending_sync == 0
        assert stats.pending_ingestion == 0
        assert stats.synced == 0
        assert stats.processed == 0
        assert stats.failed == 0

    def test_get_stats_with_entries(self, wal: UnifiedWAL):
        """get_stats should return accurate counts."""
        # Add entries of various types/statuses
        sync_id = wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.append_sync_entry("game2", "host1", "db", "hash2")
        ing_id = wal.append_ingestion_entry("game3", {"data": 1}, "host1")

        wal.mark_synced([sync_id])
        wal.mark_processed([ing_id])

        stats = wal.get_stats()
        assert stats.total_entries == 3
        assert stats.pending_sync == 1  # game2
        assert stats.pending_ingestion == 0
        assert stats.synced == 1  # game1
        assert stats.processed == 1  # game3

    def test_get_connection_pool_stats(self, wal: UnifiedWAL):
        """get_connection_pool_stats should return pool statistics."""
        stats = wal.get_connection_pool_stats()
        assert stats is not None
        assert "connections_created" in stats

    def test_get_connection_pool_stats_without_pool(self, wal_no_pool: UnifiedWAL):
        """get_connection_pool_stats should return None without pool."""
        stats = wal_no_pool.get_connection_pool_stats()
        assert stats is None


# =============================================================================
# CHECKPOINT AND COMPACTION TESTS
# =============================================================================


class TestCheckpointAndCompaction:
    """Tests for checkpoint creation and compaction."""

    def test_auto_checkpoint_after_interval(self, temp_db_path: Path):
        """Checkpoint should be created after checkpoint_interval entries."""
        wal = UnifiedWAL(
            db_path=temp_db_path,
            checkpoint_interval=3,
            auto_compact=False,
        )

        # Add entries to trigger checkpoint
        for i in range(5):
            wal.append_sync_entry(f"game{i}", "host1", "db", f"hash{i}")

        stats = wal.get_stats()
        assert stats.last_checkpoint_id > 0

    def test_compact_removes_old_processed_entries(self, wal: UnifiedWAL):
        """compact should remove old processed entries."""
        # Add and process entries
        for i in range(5):
            entry_id = wal.append_sync_entry(f"game{i}", "host1", "db", f"hash{i}")
            wal.mark_processed([entry_id])

        # Compact with 0 hours threshold (remove all)
        removed = wal.compact(older_than_hours=0)
        assert removed == 5

        stats = wal.get_stats()
        assert stats.total_entries == 0

    def test_compact_keeps_recent_entries(self, wal: UnifiedWAL):
        """compact should keep recently processed entries."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.mark_processed([entry_id])

        # Compact with 24 hour threshold
        removed = wal.compact(older_than_hours=24)
        assert removed == 0

        stats = wal.get_stats()
        assert stats.processed == 1

    def test_cleanup_failed_entries(self, wal: UnifiedWAL):
        """cleanup_failed should remove old failed entries."""
        entry_id = wal.append_sync_entry("game1", "host1", "db", "hash1")
        wal.mark_failed([entry_id], "Error")

        # Cleanup with 0 days threshold
        removed = wal.cleanup_failed(older_than_days=0)
        assert removed == 1

        failed = wal.get_failed_entries()
        assert len(failed) == 0


# =============================================================================
# CRASH RECOVERY TESTS
# =============================================================================


class TestCrashRecovery:
    """Tests for crash recovery scenarios."""

    def test_recovery_pending_entries_persist(self, temp_db_path: Path):
        """Pending entries should persist across WAL restarts."""
        # Create WAL and add entries
        wal1 = UnifiedWAL(db_path=temp_db_path)
        wal1.append_sync_entry("game1", "host1", "db", "hash1")
        wal1.append_ingestion_entry("game2", {"data": 1}, "host1")

        # Simulate crash by creating new WAL instance
        wal2 = UnifiedWAL(db_path=temp_db_path)

        # Entries should be recoverable
        pending = wal2.get_pending_entries()
        assert len(pending) == 2

    def test_recovery_idempotent_replay(self, temp_db_path: Path):
        """Replaying recovered entries should be idempotent."""
        # Create WAL and add entry
        wal1 = UnifiedWAL(db_path=temp_db_path)
        wal1.append_sync_entry("game1", "host1", "db", "hash1")

        # Simulate crash and recovery
        wal2 = UnifiedWAL(db_path=temp_db_path)

        # Try to re-add same entry (simulating replay)
        entry_id = wal2.append_sync_entry("game1", "host1", "db", "hash1")

        # Should still only have one entry
        pending = wal2.get_pending_entries()
        assert len(pending) == 1

    def test_recovery_partial_status_update(self, temp_db_path: Path):
        """Status updates should be atomic."""
        wal = UnifiedWAL(db_path=temp_db_path)
        id1 = wal.append_sync_entry("game1", "host1", "db", "hash1")
        id2 = wal.append_sync_entry("game2", "host1", "db", "hash2")

        # Mark one as processed
        wal.mark_processed([id1])

        # Simulate crash
        wal2 = UnifiedWAL(db_path=temp_db_path)

        # game1 should be processed, game2 still pending
        pending = wal2.get_pending_entries()
        assert len(pending) == 1
        assert pending[0].game_id == "game2"

    def test_data_integrity_after_recovery(self, temp_db_path: Path):
        """Game data should be intact after recovery."""
        game_data = {"moves": [1, 2, 3], "winner": 1, "board": {"state": "complex"}}

        wal1 = UnifiedWAL(db_path=temp_db_path)
        wal1.append_ingestion_entry("game1", game_data, "host1")

        # Simulate crash and recovery
        wal2 = UnifiedWAL(db_path=temp_db_path)

        pending = wal2.get_pending_ingestion_entries()
        assert len(pending) == 1
        assert pending[0].data == game_data


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_appends(self, wal: UnifiedWAL):
        """Concurrent appends should not cause data corruption."""
        errors = []
        success_count = {"count": 0}

        def append_entries(thread_id: int):
            try:
                for i in range(10):
                    wal.append_sync_entry(
                        game_id=f"game_{thread_id}_{i}",
                        source_host="host1",
                        source_db="db",
                        data_hash=f"hash_{thread_id}_{i}",
                    )
                    success_count["count"] += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_entries, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        pending = wal.get_pending_entries()
        assert len(pending) == 50  # 5 threads * 10 entries

    def test_concurrent_status_updates(self, wal: UnifiedWAL):
        """Concurrent status updates should be safe."""
        # Pre-populate entries
        entry_ids = []
        for i in range(20):
            entry_id = wal.append_sync_entry(f"game{i}", "host1", "db", f"hash{i}")
            entry_ids.append(entry_id)

        errors = []

        def update_status(ids: list):
            try:
                wal.mark_processed(ids)
            except Exception as e:
                errors.append(e)

        # Split entry IDs among threads
        threads = [
            threading.Thread(target=update_status, args=(entry_ids[i:i + 5],))
            for i in range(0, 20, 5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        stats = wal.get_stats()
        assert stats.processed == 20


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER TESTS
# =============================================================================


class TestWriteAheadLogWrapper:
    """Tests for WriteAheadLog backward compatibility wrapper.

    Note: The wrapper uses logger.warning() for deprecation notices, not
    warnings.warn(), so we test that logging occurs rather than pytest.warns().
    """

    def test_writeaheadlog_logs_deprecation(self, temp_db_path: Path, caplog):
        """WriteAheadLog should log deprecation warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            WriteAheadLog(db_path=temp_db_path)
        assert "deprecated" in caplog.text.lower()

    def test_writeaheadlog_inherits_from_unified_wal(self, temp_db_path: Path, caplog):
        """WriteAheadLog should inherit from UnifiedWAL."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = WriteAheadLog(db_path=temp_db_path)
        assert isinstance(wal, UnifiedWAL)

    def test_writeaheadlog_append_method(self, temp_db_path: Path, caplog):
        """WriteAheadLog.append should work like append_sync_entry."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = WriteAheadLog(db_path=temp_db_path)

        entry_id = wal.append("game1", "host1", "db.sqlite", "hash123")
        assert entry_id > 0

        # Verify entry was created using the parent class method
        pending = UnifiedWAL.get_pending_sync_entries(wal)
        assert len(pending) == 1
        assert pending[0].game_id == "game1"

    def test_writeaheadlog_append_batch_method(self, temp_db_path: Path, caplog):
        """WriteAheadLog.append_batch should work like append_sync_batch."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = WriteAheadLog(db_path=temp_db_path)

        entries = [
            ("game1", "host1", "db", "hash1"),
            ("game2", "host1", "db", "hash2"),
        ]
        added = wal.append_batch(entries)
        assert added == 2

    def test_writeaheadlog_confirm_synced_method(self, temp_db_path: Path, caplog):
        """WriteAheadLog.confirm_synced should mark entries as processed."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = WriteAheadLog(db_path=temp_db_path)

        wal.append("game1", "host1", "db", "hash1")
        updated = wal.confirm_synced(["game1"])
        assert updated == 1

    def test_writeaheadlog_get_pending_entries_returns_sync(self, temp_db_path: Path, caplog):
        """WriteAheadLog.get_pending_entries should return sync entries."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = WriteAheadLog(db_path=temp_db_path)

        wal.append("game1", "host1", "db", "hash1")
        pending = wal.get_pending_entries()
        assert len(pending) == 1
        assert pending[0].entry_type == WALEntryType.SYNC


class TestIngestionWALWrapper:
    """Tests for IngestionWAL backward compatibility wrapper.

    Note: The wrapper uses logger.warning() for deprecation notices, not
    warnings.warn(), so we test that logging occurs rather than pytest.warns().
    """

    def test_ingestionwal_logs_deprecation(self, tmp_path: Path, caplog):
        """IngestionWAL should log deprecation warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            IngestionWAL(wal_dir=tmp_path)
        assert "deprecated" in caplog.text.lower()

    def test_ingestionwal_inherits_from_unified_wal(self, tmp_path: Path, caplog):
        """IngestionWAL should inherit from UnifiedWAL."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)
        assert isinstance(wal, UnifiedWAL)

    def test_ingestionwal_creates_db_in_directory(self, tmp_path: Path, caplog):
        """IngestionWAL should create unified_wal.db in wal_dir."""
        import logging
        with caplog.at_level(logging.WARNING):
            IngestionWAL(wal_dir=tmp_path)
        assert (tmp_path / "unified_wal.db").exists()

    def test_ingestionwal_append_method(self, tmp_path: Path, caplog):
        """IngestionWAL.append should work like append_ingestion_entry."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        entry_id = wal.append({"game_id": "game1", "data": 1}, source_host="host1", game_id="game1")
        assert entry_id > 0

        pending = wal.get_unprocessed()
        assert len(pending) == 1

    def test_ingestionwal_append_extracts_game_id_from_data(self, tmp_path: Path, caplog):
        """IngestionWAL.append should extract game_id from game_data if not provided."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        entry_id = wal.append({"game_id": "game1", "data": 1})
        assert entry_id > 0

        pending = wal.get_unprocessed()
        assert pending[0].game_id == "game1"

    def test_ingestionwal_append_requires_game_id(self, tmp_path: Path, caplog):
        """IngestionWAL.append should raise if game_id not provided and not in data."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        with pytest.raises(ValueError, match="game_id required"):
            wal.append({"data": 1})  # No game_id

    def test_ingestionwal_mark_processed_single(self, tmp_path: Path, caplog):
        """IngestionWAL.mark_processed_single should work."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        entry_id = wal.append({"game_id": "game1"}, game_id="game1")
        result = wal.mark_processed_single(entry_id)
        assert result is True

        # Verify no longer pending
        pending = wal.get_unprocessed()
        assert len(pending) == 0

    def test_ingestionwal_mark_batch_processed(self, tmp_path: Path, caplog):
        """IngestionWAL.mark_batch_processed should work like mark_processed."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        id1 = wal.append({"game_id": "game1"}, game_id="game1")
        id2 = wal.append({"game_id": "game2"}, game_id="game2")

        updated = wal.mark_batch_processed([id1, id2])
        assert updated == 2

    def test_ingestionwal_get_unprocessed(self, tmp_path: Path, caplog):
        """IngestionWAL.get_unprocessed should return pending ingestion entries."""
        import logging
        with caplog.at_level(logging.WARNING):
            wal = IngestionWAL(wal_dir=tmp_path)

        wal.append({"game_id": "game1"}, game_id="game1")
        wal.append({"game_id": "game2"}, game_id="game2")

        pending = wal.get_unprocessed(limit=10)
        assert len(pending) == 2


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestSingleton:
    """Tests for singleton WAL instance."""

    def test_get_unified_wal_returns_singleton(self, tmp_path: Path):
        """get_unified_wal should return same instance."""
        reset_wal_instance()  # Clear any existing instance

        db_path = tmp_path / "singleton_test.db"
        wal1 = get_unified_wal(db_path)
        wal2 = get_unified_wal()  # Should return same instance

        assert wal1 is wal2

    def test_reset_wal_instance(self, tmp_path: Path):
        """reset_wal_instance should clear singleton."""
        reset_wal_instance()

        db_path = tmp_path / "reset_test.db"
        wal1 = get_unified_wal(db_path)

        reset_wal_instance()

        db_path2 = tmp_path / "reset_test2.db"
        wal2 = get_unified_wal(db_path2)

        # Should be different instances
        assert wal1 is not wal2
