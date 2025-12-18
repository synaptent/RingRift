"""Tests for Distributed Lock implementation.

Tests the distributed locking system that coordinates training across
multiple nodes using Redis or file-based fallback.
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.coordination.distributed_lock import (
    DistributedLock,
    DEFAULT_LOCK_TIMEOUT,
    DEFAULT_ACQUIRE_TIMEOUT,
    LOCK_DIR,
)


class TestDistributedLockInit:
    """Tests for DistributedLock initialization."""

    def test_init_creates_lock_with_name(self):
        """Should create lock with given name."""
        lock = DistributedLock("test_lock", use_redis=False)
        assert lock.name == "test_lock"

    def test_init_sets_default_timeout(self):
        """Should use default lock timeout."""
        lock = DistributedLock("test_lock", use_redis=False)
        assert lock.lock_timeout == DEFAULT_LOCK_TIMEOUT

    def test_init_custom_timeout(self):
        """Should accept custom lock timeout."""
        lock = DistributedLock("test_lock", lock_timeout=120, use_redis=False)
        assert lock.lock_timeout == 120

    def test_init_generates_unique_lock_id(self):
        """Should generate unique lock ID."""
        lock1 = DistributedLock("test_lock", use_redis=False)
        lock2 = DistributedLock("test_lock", use_redis=False)
        assert lock1._lock_id != lock2._lock_id

    def test_init_not_acquired_initially(self):
        """Lock should not be acquired initially."""
        lock = DistributedLock("test_lock", use_redis=False)
        assert lock._acquired is False


class TestFileLocking:
    """Tests for file-based locking fallback."""

    @pytest.fixture
    def temp_lock_dir(self):
        """Create temporary directory for lock files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_acquire_file_lock_succeeds(self):
        """Should acquire file lock successfully."""
        lock = DistributedLock("test_acquire", use_redis=False)
        try:
            result = lock.acquire(timeout=5, blocking=False)
            assert result is True
            assert lock._acquired is True
        finally:
            lock.release()

    def test_release_file_lock(self):
        """Should release file lock."""
        lock = DistributedLock("test_release", use_redis=False)
        lock.acquire(timeout=5, blocking=False)
        assert lock._acquired is True

        lock.release()
        assert lock._acquired is False

    def test_acquire_twice_returns_true(self):
        """Should return True if already acquired."""
        lock = DistributedLock("test_twice", use_redis=False)
        try:
            lock.acquire(timeout=5, blocking=False)
            result = lock.acquire(timeout=5, blocking=False)
            assert result is True
        finally:
            lock.release()

    def test_release_without_acquire_is_safe(self):
        """Should safely handle release without acquire."""
        lock = DistributedLock("test_no_acquire", use_redis=False)
        lock.release()  # Should not raise


class TestContextManager:
    """Tests for context manager interface."""

    def test_context_manager_acquires_and_releases(self):
        """Should acquire on enter and release on exit."""
        lock = DistributedLock("test_context", use_redis=False)

        with lock:
            assert lock._acquired is True

        assert lock._acquired is False

    def test_context_manager_releases_on_exception(self):
        """Should release lock even when exception occurs."""
        lock = DistributedLock("test_exception", use_redis=False)

        try:
            with lock:
                assert lock._acquired is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert lock._acquired is False


class TestNonBlockingAcquire:
    """Tests for non-blocking lock acquisition."""

    def test_non_blocking_returns_immediately(self):
        """Non-blocking acquire should return immediately."""
        lock = DistributedLock("test_nonblock", use_redis=False)

        start = time.time()
        result = lock.acquire(timeout=60, blocking=False)
        elapsed = time.time() - start

        try:
            assert result is True
            assert elapsed < 1.0  # Should be nearly instant
        finally:
            lock.release()


class TestLockContention:
    """Tests for lock contention scenarios."""

    def test_second_lock_blocked(self):
        """Second lock acquisition should be blocked by first."""
        lock1 = DistributedLock("test_contention", use_redis=False)
        lock2 = DistributedLock("test_contention", use_redis=False)

        try:
            # First lock succeeds
            assert lock1.acquire(timeout=5, blocking=False) is True

            # Second lock should fail immediately in non-blocking mode
            assert lock2.acquire(timeout=1, blocking=False) is False
        finally:
            lock1.release()

    def test_lock_released_allows_second_acquire(self):
        """After release, second lock should succeed."""
        lock1 = DistributedLock("test_release_acquire", use_redis=False)
        lock2 = DistributedLock("test_release_acquire", use_redis=False)

        # Acquire and release first lock
        lock1.acquire(timeout=5, blocking=False)
        lock1.release()

        # Second lock should now succeed
        try:
            result = lock2.acquire(timeout=5, blocking=False)
            assert result is True
        finally:
            lock2.release()


class TestLockTimeout:
    """Tests for lock acquisition timeout."""

    def test_timeout_returns_false(self):
        """Should return False after timeout."""
        lock1 = DistributedLock("test_timeout", use_redis=False)
        lock2 = DistributedLock("test_timeout", use_redis=False)

        try:
            lock1.acquire(timeout=5, blocking=False)

            start = time.time()
            result = lock2.acquire(timeout=1, blocking=True)
            elapsed = time.time() - start

            assert result is False
            assert 0.9 < elapsed < 2.0  # Should wait close to timeout
        finally:
            lock1.release()


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_acquire_one_wins(self):
        """Only one thread should acquire the lock."""
        lock_name = f"test_thread_{time.time()}"
        acquired_count = [0]
        lock_obj = threading.Lock()

        def try_acquire():
            lock = DistributedLock(lock_name, use_redis=False)
            if lock.acquire(timeout=1, blocking=False):
                with lock_obj:
                    acquired_count[0] += 1
                time.sleep(0.5)
                lock.release()

        threads = [threading.Thread(target=try_acquire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have acquired
        assert acquired_count[0] == 1


class TestRedisLocking:
    """Tests for Redis-based locking."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        with patch('app.coordination.distributed_lock.redis') as mock:
            mock_client = MagicMock()
            mock.Redis.from_url.return_value = mock_client
            mock_client.ping.return_value = True
            yield mock_client

    def test_uses_redis_when_available(self, mock_redis):
        """Should use Redis when available."""
        with patch('app.coordination.distributed_lock.HAS_REDIS', True):
            lock = DistributedLock("test_redis", use_redis=True)
            assert lock._redis_client is not None

    def test_fallback_to_file_when_redis_fails(self):
        """Should fall back to file lock when Redis unavailable."""
        with patch('app.coordination.distributed_lock.HAS_REDIS', True):
            with patch('app.coordination.distributed_lock.redis') as mock:
                mock.Redis.from_url.side_effect = Exception("Connection refused")
                lock = DistributedLock("test_fallback", use_redis=True)
                assert lock._redis_client is None


class TestLockName:
    """Tests for lock naming."""

    def test_different_names_independent(self):
        """Locks with different names should be independent."""
        lock1 = DistributedLock("lock_a", use_redis=False)
        lock2 = DistributedLock("lock_b", use_redis=False)

        try:
            assert lock1.acquire(timeout=5, blocking=False) is True
            assert lock2.acquire(timeout=5, blocking=False) is True
        finally:
            lock1.release()
            lock2.release()

    def test_same_name_contends(self):
        """Locks with same name should contend."""
        lock1 = DistributedLock("same_name", use_redis=False)
        lock2 = DistributedLock("same_name", use_redis=False)

        try:
            assert lock1.acquire(timeout=5, blocking=False) is True
            assert lock2.acquire(timeout=1, blocking=False) is False
        finally:
            lock1.release()
