"""Tests for scripts/lib/retry.py module.

Tests cover:
- RetryConfig delay calculation
- RetryAttempt properties
- @retry decorator
- @retry_on_exception decorator
- @retry_async decorator
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from scripts.lib.retry import (
    RetryConfig,
    RetryAttempt,
    retry,
    retry_on_exception,
    retry_async,
    with_timeout,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential is True
        assert config.jitter == 0.1

    def test_get_delay_first_attempt(self):
        """Test that first attempt has no delay."""
        config = RetryConfig(base_delay=5.0, jitter=0)
        assert config.get_delay(0) == 0.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential=True, jitter=0)
        assert config.get_delay(1) == 1.0  # 1 * 2^0
        assert config.get_delay(2) == 2.0  # 1 * 2^1
        assert config.get_delay(3) == 4.0  # 1 * 2^2

    def test_get_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(base_delay=2.0, exponential=False, jitter=0)
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 2.0

    def test_get_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential=True, jitter=0)
        assert config.get_delay(5) == 15.0  # Would be 160.0 without cap

    def test_attempts_generator(self):
        """Test attempts generator yields correct number of attempts."""
        config = RetryConfig(max_attempts=3)
        attempts = list(config.attempts())
        assert len(attempts) == 3
        assert attempts[0].number == 1
        assert attempts[1].number == 2
        assert attempts[2].number == 3


class TestRetryAttempt:
    """Tests for RetryAttempt class."""

    def test_is_first(self):
        """Test is_first property."""
        first = RetryAttempt(number=1, max_attempts=3, delay=0)
        second = RetryAttempt(number=2, max_attempts=3, delay=1.0)

        assert first.is_first is True
        assert second.is_first is False

    def test_is_last(self):
        """Test is_last property."""
        first = RetryAttempt(number=1, max_attempts=3, delay=0)
        last = RetryAttempt(number=3, max_attempts=3, delay=4.0)

        assert first.is_last is False
        assert last.is_last is True

    def test_should_retry(self):
        """Test should_retry property."""
        first = RetryAttempt(number=1, max_attempts=3, delay=0)
        last = RetryAttempt(number=3, max_attempts=3, delay=4.0)

        assert first.should_retry is True
        assert last.should_retry is False

    @patch('time.sleep')
    def test_wait(self, mock_sleep):
        """Test wait method calls time.sleep."""
        attempt = RetryAttempt(number=2, max_attempts=3, delay=2.5)
        attempt.wait()
        mock_sleep.assert_called_once_with(2.5)

    @patch('time.sleep')
    def test_wait_zero_delay(self, mock_sleep):
        """Test wait with zero delay doesn't sleep."""
        attempt = RetryAttempt(number=1, max_attempts=3, delay=0)
        attempt.wait()
        mock_sleep.assert_not_called()


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_success_no_retry(self):
        """Test successful function doesn't retry."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test function retries on failure."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Test exception raised when max attempts exceeded."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count == 3

    def test_specific_exceptions(self):
        """Test retry only catches specified exceptions."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01, exceptions=ValueError)
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raises_type_error()

        # Should not retry on TypeError
        assert call_count == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retries = []

        def on_retry(exc, attempt):
            retries.append((str(exc), attempt))

        @retry(max_attempts=3, delay=0.01, on_retry=on_retry)
        def fails_twice():
            if len(retries) < 2:
                raise ValueError(f"Attempt {len(retries) + 1}")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert len(retries) == 2
        assert retries[0][1] == 1
        assert retries[1][1] == 2


class TestRetryOnException:
    """Tests for @retry_on_exception decorator."""

    def test_retry_on_specific_exception(self):
        """Test retry on specific exception type."""
        call_count = 0

        @retry_on_exception(ConnectionError, max_attempts=3, delay=0.01)
        def flaky_connection():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("No connection")
            return "connected"

        result = flaky_connection()
        assert result == "connected"
        assert call_count == 2

    def test_multiple_exception_types(self):
        """Test retry on multiple exception types."""
        call_count = 0

        @retry_on_exception(ConnectionError, TimeoutError, max_attempts=4, delay=0.01)
        def network_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("No connection")
            if call_count == 2:
                raise TimeoutError("Timed out")
            return "success"

        result = network_call()
        assert result == "success"
        assert call_count == 3


class TestRetryAsync:
    """Tests for @retry_async decorator."""

    @pytest.mark.asyncio
    async def test_async_success(self):
        """Test async function succeeds without retry."""
        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def async_operation():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_operation()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_failure(self):
        """Test async function retries on failure."""
        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def async_fails_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First attempt fails")
            return "success"

        result = await async_fails_once()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_max_attempts_exceeded(self):
        """Test async exception raised when max attempts exceeded."""
        call_count = 0

        @retry_async(max_attempts=2, delay=0.01)
        async def async_always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            await async_always_fails()

        assert call_count == 2


class TestWithTimeout:
    """Tests for @with_timeout decorator."""

    def test_fast_function_succeeds(self):
        """Test function that completes within timeout."""
        @with_timeout(1.0)
        def fast_function():
            return "done"

        result = fast_function()
        assert result == "done"

    def test_slow_function_returns_default(self):
        """Test function that exceeds timeout returns default."""
        @with_timeout(0.1, default="timed out")
        def slow_function():
            import time
            time.sleep(1.0)
            return "done"

        result = slow_function()
        assert result == "timed out"

    def test_timeout_returns_none_by_default(self):
        """Test timeout returns None when no default specified."""
        @with_timeout(0.1)
        def slow_function():
            import time
            time.sleep(1.0)
            return "done"

        result = slow_function()
        assert result is None


class TestRetryConfigAttempts:
    """Integration tests for retry config with attempts loop."""

    def test_manual_retry_loop(self):
        """Test using retry config in manual loop."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        attempts_made = []
        result = None

        for attempt in config.attempts():
            attempts_made.append(attempt.number)
            try:
                if attempt.number < 3:
                    raise ValueError("Not ready")
                result = "success"
                break
            except ValueError:
                if not attempt.should_retry:
                    raise
                attempt.wait()

        assert result == "success"
        assert attempts_made == [1, 2, 3]
