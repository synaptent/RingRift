"""Tests for rate limiting utilities."""

import time

import pytest

from app.utils.rate_limit import (
    Cooldown,
    RateLimiter,
    KeyedRateLimiter,
    rate_limit,
    rate_limit_async,
    RateLimitExceeded,
)


class TestCooldown:
    """Tests for Cooldown class."""

    def test_ready_initially(self):
        cooldown = Cooldown(seconds=1.0)
        assert cooldown.ready()

    def test_not_ready_after_reset(self):
        cooldown = Cooldown(seconds=1.0)
        cooldown.reset()
        assert not cooldown.ready()

    def test_ready_after_elapsed(self):
        cooldown = Cooldown(seconds=0.05)
        cooldown.reset()
        assert not cooldown.ready()
        time.sleep(0.06)
        assert cooldown.ready()

    def test_remaining(self):
        cooldown = Cooldown(seconds=1.0)
        cooldown.reset()
        remaining = cooldown.remaining()
        assert 0.9 < remaining <= 1.0

    def test_remaining_zero_when_ready(self):
        cooldown = Cooldown(seconds=0.01)
        cooldown.reset()
        time.sleep(0.02)
        assert cooldown.remaining() == 0.0

    def test_elapsed(self):
        cooldown = Cooldown(seconds=1.0)
        cooldown.reset()
        time.sleep(0.05)
        elapsed = cooldown.elapsed()
        assert elapsed >= 0.05


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_acquire_under_limit(self):
        limiter = RateLimiter(rate=10, per_seconds=1.0)
        for _ in range(10):
            assert limiter.acquire()

    def test_acquire_at_limit(self):
        limiter = RateLimiter(rate=5, per_seconds=1.0, burst=5)
        # Use all tokens
        for _ in range(5):
            assert limiter.acquire()
        # Next should fail
        assert not limiter.acquire()

    def test_tokens_refill(self):
        limiter = RateLimiter(rate=100, per_seconds=1.0, burst=1)
        assert limiter.acquire()
        assert not limiter.acquire()
        # Wait for refill
        time.sleep(0.02)
        assert limiter.acquire()

    def test_burst_size(self):
        limiter = RateLimiter(rate=10, per_seconds=1.0, burst=20)
        # Can burst to 20
        for _ in range(20):
            assert limiter.acquire()
        # Then limited
        assert not limiter.acquire()

    def test_available_tokens(self):
        limiter = RateLimiter(rate=10, per_seconds=1.0, burst=10)
        assert limiter.available_tokens == 10.0
        limiter.acquire()
        # Use approximate comparison due to token refill between calls
        assert 8.9 < limiter.available_tokens <= 9.1

    def test_wait(self):
        limiter = RateLimiter(rate=100, per_seconds=1.0, burst=1)
        limiter.acquire()
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # Should have waited some time for token refill
        assert elapsed > 0

    def test_acquire_multiple_tokens(self):
        limiter = RateLimiter(rate=10, per_seconds=1.0, burst=10)
        assert limiter.acquire(tokens=5)
        # Use approximate comparison due to token refill between calls
        assert 4.9 < limiter.available_tokens <= 5.1
        assert limiter.acquire(tokens=5)
        assert not limiter.acquire(tokens=1)


class TestKeyedRateLimiter:
    """Tests for KeyedRateLimiter class."""

    def test_separate_limits_per_key(self):
        limiter = KeyedRateLimiter(rate=2, per_seconds=1.0, burst=2)

        # User A can make 2 requests
        assert limiter.acquire("user_a")
        assert limiter.acquire("user_a")
        assert not limiter.acquire("user_a")

        # User B has separate limit
        assert limiter.acquire("user_b")
        assert limiter.acquire("user_b")
        assert not limiter.acquire("user_b")

    def test_cleanup(self):
        limiter = KeyedRateLimiter(rate=10, per_seconds=1.0)
        limiter.acquire("old_key")
        limiter.acquire("new_key")

        # Force old key to be stale
        limiter._limiters["old_key"]._last_update = time.time() - 7200

        removed = limiter.cleanup(max_age_seconds=3600)
        assert removed == 1
        assert "old_key" not in limiter._limiters
        assert "new_key" in limiter._limiters


class TestRateLimitDecorator:
    """Tests for @rate_limit decorator."""

    def test_allows_calls_under_limit(self):
        call_count = 0

        @rate_limit(rate=10, per_seconds=1.0)
        def my_func():
            nonlocal call_count
            call_count += 1
            return call_count

        for i in range(10):
            result = my_func()
            assert result == i + 1

    def test_waits_when_limited(self):
        @rate_limit(rate=100, per_seconds=1.0, burst=1)
        def my_func():
            return time.time()

        # First call immediate
        t1 = my_func()
        # Second call should wait
        t2 = my_func()

        assert t2 > t1

    def test_raises_when_wait_false(self):
        @rate_limit(rate=1, per_seconds=1.0, burst=1, wait=False)
        def my_func():
            pass

        my_func()  # First call succeeds
        with pytest.raises(RateLimitExceeded):
            my_func()  # Second call raises


class TestRateLimitAsyncDecorator:
    """Tests for @rate_limit_async decorator."""

    @pytest.mark.asyncio
    async def test_allows_calls_under_limit(self):
        call_count = 0

        @rate_limit_async(rate=10, per_seconds=1.0)
        async def my_func():
            nonlocal call_count
            call_count += 1
            return call_count

        for i in range(10):
            result = await my_func()
            assert result == i + 1

    @pytest.mark.asyncio
    async def test_raises_when_wait_false(self):
        @rate_limit_async(rate=1, per_seconds=1.0, burst=1, wait=False)
        async def my_func():
            pass

        await my_func()  # First call succeeds
        with pytest.raises(RateLimitExceeded):
            await my_func()  # Second call raises


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_is_exception(self):
        exc = RateLimitExceeded("rate limited")
        assert isinstance(exc, Exception)

    def test_message(self):
        exc = RateLimitExceeded("custom message")
        assert str(exc) == "custom message"
