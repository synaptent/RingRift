"""Tests for sync_mutex module.

Tests the SyncMutex class which provides SQLite-backed mutexes for
coordinating rsync and file transfer operations.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.sync_mutex import (
    SyncLockInfo,
    SyncMutex,
    acquire_sync_lock,
    get_sync_mutex,
    is_sync_locked,
    release_sync_lock,
    reset_sync_mutex,
    sync_mutex_connection_stats,
    sync_heartbeat,
    sync_lock,
    sync_lock_required,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sync_mutex.db"
        yield db_path


@pytest.fixture
def mutex(temp_db):
    """Create a SyncMutex with a temporary database."""
    m = SyncMutex(db_path=temp_db)
    yield m
    m.close()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global singleton before and after each test."""
    reset_sync_mutex()
    yield
    reset_sync_mutex()


@pytest.fixture
def multi_host_capacity():
    """Temporarily raise the cluster-wide sync cap for multi-host tests."""
    with patch("app.coordination.sync_mutex.MAX_GLOBAL_CONCURRENT_SYNCS", 8):
        yield


class TestSyncLockInfo:
    """Tests for SyncLockInfo dataclass."""

    def test_age_seconds(self):
        """age_seconds should return time since acquisition."""
        now = time.time()
        info = SyncLockInfo(
            host="test-host",
            operation="rsync",
            holder_pid=1234,
            holder_hostname="test-machine",
            acquired_at=now - 10,
            timeout_at=now + 50,
        )
        assert 9.5 < info.age_seconds < 11.0

    def test_is_expired_false(self):
        """is_expired should return False when timeout not reached."""
        now = time.time()
        info = SyncLockInfo(
            host="test-host",
            operation="rsync",
            holder_pid=1234,
            holder_hostname="test-machine",
            acquired_at=now,
            timeout_at=now + 100,
        )
        assert not info.is_expired

    def test_is_expired_true(self):
        """is_expired should return True when timeout passed."""
        now = time.time()
        info = SyncLockInfo(
            host="test-host",
            operation="rsync",
            holder_pid=1234,
            holder_hostname="test-machine",
            acquired_at=now - 100,
            timeout_at=now - 10,
        )
        assert info.is_expired

    def test_to_dict(self):
        """to_dict should return serializable dictionary."""
        now = time.time()
        info = SyncLockInfo(
            host="test-host",
            operation="rsync",
            holder_pid=1234,
            holder_hostname="test-machine",
            acquired_at=now,
            timeout_at=now + 60,
        )
        d = info.to_dict()
        assert d["host"] == "test-host"
        assert d["operation"] == "rsync"
        assert d["holder_pid"] == 1234
        assert isinstance(d["age_seconds"], float)
        assert isinstance(d["is_expired"], bool)


class TestSyncMutexAcquisition:
    """Tests for lock acquisition."""

    def test_acquire_lock(self, mutex):
        """Should successfully acquire a lock."""
        result = mutex.acquire("test-host", "rsync")
        assert result is True
        assert mutex.is_locked("test-host")

    def test_acquire_lock_blocked_by_existing(self, mutex):
        """Should fail to acquire if another lock exists."""
        mutex.acquire("test-host", "rsync")
        result = mutex.acquire("test-host", "rsync", wait=False)
        assert result is False

    def test_acquire_different_hosts(self, mutex, multi_host_capacity):
        """Should allow locks on different hosts."""
        result1 = mutex.acquire("host-1", "rsync")
        result2 = mutex.acquire("host-2", "rsync")
        assert result1 is True
        assert result2 is True
        assert mutex.is_locked("host-1")
        assert mutex.is_locked("host-2")

    def test_acquire_with_wait(self, mutex):
        """Should wait for lock when wait=True."""
        # Acquire lock
        mutex.acquire("test-host", "rsync", timeout=1)

        # Try to acquire with wait in another thread
        acquired = []

        def try_acquire():
            # This will fail quickly since first lock expires in 1s
            result = mutex.acquire("test-host", "rsync", wait=True, wait_timeout=2)
            acquired.append(result)

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=3)

        # Should have acquired after first lock expired
        assert len(acquired) == 1
        assert acquired[0] is True


class TestSyncMutexRelease:
    """Tests for lock release."""

    def test_release_lock(self, mutex):
        """Should successfully release a lock."""
        mutex.acquire("test-host", "rsync")
        result = mutex.release("test-host")
        assert result is True
        assert not mutex.is_locked("test-host")

    def test_release_nonexistent_lock(self, mutex):
        """Should return False when releasing non-existent lock."""
        result = mutex.release("nonexistent-host")
        assert result is False

    def test_release_all_for_process(self, mutex, multi_host_capacity):
        """Should release all locks held by this process."""
        mutex.acquire("host-1", "rsync")
        mutex.acquire("host-2", "rsync")
        mutex.acquire("host-3", "rsync")

        count = mutex.release_all_for_process()
        assert count == 3
        assert not mutex.is_locked("host-1")
        assert not mutex.is_locked("host-2")
        assert not mutex.is_locked("host-3")


class TestSyncMutexExpiration:
    """Tests for lock expiration."""

    def test_lock_expires_after_timeout(self, mutex):
        """Lock should be cleared after timeout."""
        mutex.acquire("test-host", "rsync", timeout=0.5)
        assert mutex.is_locked("test-host")

        # Wait for expiration
        time.sleep(0.7)

        # is_locked triggers cleanup
        assert not mutex.is_locked("test-host")

    def test_get_lock_info(self, mutex):
        """Should return lock information."""
        mutex.acquire("test-host", "rsync")
        info = mutex.get_lock_info("test-host")

        assert info is not None
        assert info.host == "test-host"
        assert info.operation == "rsync"
        assert info.holder_pid == os.getpid()

    def test_get_all_locks(self, mutex, multi_host_capacity):
        """Should return all active locks."""
        mutex.acquire("host-1", "rsync")
        mutex.acquire("host-2", "scp")
        mutex.acquire("host-3", "rsync")

        locks = mutex.get_all_locks()
        assert len(locks) == 3
        hosts = {lock.host for lock in locks}
        assert hosts == {"host-1", "host-2", "host-3"}


class TestSyncMutexHeartbeat:
    """Tests for heartbeat mechanism."""

    def test_heartbeat_updates_timestamp(self, mutex):
        """Heartbeat should update last_heartbeat."""
        mutex.acquire("test-host", "rsync")
        time.sleep(0.1)
        result = mutex.heartbeat("test-host")
        assert result is True

    def test_heartbeat_fails_for_nonexistent(self, mutex):
        """Heartbeat should fail for non-existent lock."""
        result = mutex.heartbeat("nonexistent-host")
        assert result is False

    def test_heartbeat_fails_for_other_process(self, mutex):
        """Heartbeat should fail for locks owned by other processes."""
        mutex.acquire("test-host", "rsync")

        # Manually update holder_pid to simulate another process
        conn = mutex._get_connection()
        conn.execute(
            "UPDATE sync_locks SET holder_pid = ? WHERE host = ?",
            (99999, "test-host"),
        )
        conn.commit()

        result = mutex.heartbeat("test-host")
        assert result is False


class TestSyncMutexForceRelease:
    """Tests for force release functionality."""

    def test_force_release(self, mutex):
        """Should force release any lock."""
        mutex.acquire("test-host", "rsync")
        result = mutex.force_release("test-host")
        assert result is True
        assert not mutex.is_locked("test-host")

    def test_force_release_nonexistent(self, mutex):
        """Force release should return False for non-existent lock."""
        result = mutex.force_release("nonexistent-host")
        assert result is False


class TestSyncMutexStats:
    """Tests for statistics functionality."""

    def test_get_stats_empty(self, mutex):
        """Stats should show no locks when empty."""
        stats = mutex.get_stats()
        assert stats["active_locks"] == 0
        assert stats["locks"] == []

    def test_get_stats_with_locks(self, mutex, multi_host_capacity):
        """Stats should show active locks."""
        mutex.acquire("host-1", "rsync")
        mutex.acquire("host-2", "scp")

        stats = mutex.get_stats()
        assert stats["active_locks"] == 2
        assert len(stats["locks"]) == 2


class TestSyncMutexContextManager:
    """Tests for context manager usage."""

    def test_sync_lock_context_manager(self, temp_db):
        """Context manager should acquire and release lock."""
        # Use patched singleton with our temp db
        with patch("app.coordination.sync_mutex._sync_mutex", SyncMutex(temp_db)):
            with sync_lock("test-host", "rsync") as acquired:
                assert acquired is True
                assert is_sync_locked("test-host")

            # Lock should be released after context
            assert not is_sync_locked("test-host")

    def test_sync_lock_required_raises_on_timeout(self, temp_db):
        """sync_lock_required should raise TimeoutError on failure."""
        mutex = SyncMutex(temp_db)
        mutex.acquire("test-host", "rsync")

        with patch("app.coordination.sync_mutex._sync_mutex", mutex):
            with pytest.raises(TimeoutError) as exc_info:
                with sync_lock_required("test-host", wait_timeout=0.5):
                    pass

            assert "Could not acquire sync lock" in str(exc_info.value)


class TestSyncMutexConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_acquire_release_sync_lock(self, temp_db):
        """Convenience functions should work with singleton."""
        with patch("app.coordination.sync_mutex._sync_mutex", SyncMutex(temp_db)):
            result = acquire_sync_lock("test-host")
            assert result is True
            assert is_sync_locked("test-host")

            result = release_sync_lock("test-host")
            assert result is True
            assert not is_sync_locked("test-host")

    def test_sync_heartbeat_function(self, temp_db):
        """sync_heartbeat convenience function should work."""
        with patch("app.coordination.sync_mutex._sync_mutex", SyncMutex(temp_db)):
            acquire_sync_lock("test-host")
            result = sync_heartbeat("test-host")
            assert result is True


class TestSyncMutexCrashRecovery:
    """Tests for crash detection and recovery."""

    def test_cleanup_crashed_locks_local_dead_process(self, mutex):
        """Should clean up locks from dead local processes."""
        mutex.acquire("test-host", "rsync")

        # Manually update to a non-existent PID
        conn = mutex._get_connection()
        conn.execute(
            "UPDATE sync_locks SET holder_pid = ?, holder_hostname = ? WHERE host = ?",
            (99999, os.uname().nodename, "test-host"),
        )
        conn.commit()

        cleaned = mutex.cleanup_crashed_locks()
        assert cleaned == 1
        assert not mutex.is_locked("test-host")

    def test_cleanup_crashed_locks_stale_heartbeat(self, mutex):
        """Should clean up locks with stale heartbeats."""
        mutex.acquire("test-host", "rsync")

        # Manually set old heartbeat
        conn = mutex._get_connection()
        old_time = time.time() - 120  # 2 minutes ago
        conn.execute(
            "UPDATE sync_locks SET last_heartbeat = ? WHERE host = ?",
            (old_time, "test-host"),
        )
        conn.commit()

        cleaned = mutex.cleanup_crashed_locks()
        assert cleaned == 1
        assert not mutex.is_locked("test-host")


class TestSyncMutexThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_acquisition(self, temp_db):
        """Only one thread should acquire lock at a time."""
        mutex = SyncMutex(temp_db)
        acquired_count = []
        lock = threading.Lock()

        def try_acquire():
            result = mutex.acquire("test-host", "rsync", wait=False)
            with lock:
                acquired_count.append(result)
            if result:
                time.sleep(0.1)
                mutex.release("test-host")

        threads = [threading.Thread(target=try_acquire) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should have acquired
        assert sum(acquired_count) == 1
        mutex.close()

    def test_concurrent_access_different_hosts(self, temp_db, multi_host_capacity):
        """Multiple threads should access different hosts concurrently."""
        mutex = SyncMutex(temp_db)
        results = {}
        lock = threading.Lock()

        def acquire_host(host):
            result = mutex.acquire(host, "rsync", wait=False)
            with lock:
                results[host] = result

        threads = [
            threading.Thread(target=acquire_host, args=(f"host-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should have acquired
        assert all(results.values())
        assert len(results) == 5
        mutex.close()

    def test_prune_stale_thread_connections(self, temp_db):
        """Connections opened by completed worker threads should be reclaimable."""
        mutex = SyncMutex(temp_db)

        def use_mutex_from_worker():
            assert mutex.acquire("worker-host", "rsync", wait=False)
            assert mutex.release("worker-host")

        thread = threading.Thread(target=use_mutex_from_worker)
        thread.start()
        thread.join()

        before = mutex.connection_stats()
        assert before["tracked_threads"] >= 2
        assert before["stale_threads"] >= 1
        assert before["python_threads"] >= 1

        closed = mutex.prune_stale_connections()
        after = mutex.connection_stats()

        assert closed >= 1
        assert after["tracked_threads"] == 1
        mutex.close_all()

    def test_global_connection_stats_do_not_create_singleton(self):
        """Module-level diagnostics should be safe before first use."""
        reset_sync_mutex()

        stats = sync_mutex_connection_stats()

        assert stats["tracked_connections"] == 0
        assert stats["tracked_threads"] == 0
        assert stats["python_threads"] >= 1
