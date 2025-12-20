"""Tests for training error handling features.

Tests retry_with_backoff, handle_gpu_error, and TrainingErrorHandler.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.training.fault_tolerance import (
    NonRecoverableError,
    RecoverableError,
    TrainingErrorHandler,
    handle_gpu_error,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_first_attempt(self):
        """Function succeeds on first attempt - no retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Function retries on failure and eventually succeeds."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Function raises after max retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fail()

        # Initial call + 2 retries = 3 total calls
        assert call_count == 3

    def test_specific_exception_type(self):
        """Only retries on specified exception types."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exceptions=(ValueError,)
        )
        def type_error_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not caught")

        with pytest.raises(TypeError):
            type_error_func()

        # TypeError not in exceptions, so no retry
        assert call_count == 1

    def test_on_retry_callback(self):
        """on_retry callback is called with correct arguments."""
        callback_calls = []

        def on_retry(exc, attempt, delay):
            callback_calls.append((str(exc), attempt, delay))

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            exponential_base=2.0,
            on_retry=on_retry,
        )
        def fail_twice():
            if len(callback_calls) < 2:
                raise ValueError(f"Fail {len(callback_calls)}")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert len(callback_calls) == 2
        assert callback_calls[0][1] == 1  # First retry
        assert callback_calls[1][1] == 2  # Second retry

    def test_exponential_backoff_timing(self):
        """Delays increase exponentially."""
        call_times = []

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.05,
            exponential_base=2.0,
        )
        def timed_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Fail")
            return "success"

        timed_func()

        # Check delays are increasing
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Second delay should be roughly 2x first delay
        assert delay2 >= delay1 * 1.5  # Allow some timing variance


class TestHandleGpuError:
    """Tests for handle_gpu_error decorator."""

    def test_no_error_passthrough(self):
        """Normal execution passes through."""
        @handle_gpu_error(fallback_to_cpu=True)
        def normal_func():
            return "success"

        assert normal_func() == "success"

    def test_non_gpu_error_propagates(self):
        """Non-GPU errors propagate unchanged."""
        @handle_gpu_error(fallback_to_cpu=True)
        def non_gpu_error():
            raise ValueError("Not a GPU error")

        with pytest.raises(ValueError, match="Not a GPU error"):
            non_gpu_error()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_clears_cuda_cache_on_oom(self, mock_empty_cache, mock_cuda_available):
        """Clears CUDA cache on OOM error."""
        call_count = 0

        @handle_gpu_error(fallback_to_cpu=True, clear_cache=True)
        def oom_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory")
            return "fallback_success"

        result = oom_func()
        assert result == "fallback_success"
        mock_empty_cache.assert_called_once()

    def test_device_kwarg_modified_on_fallback(self):
        """device kwarg is modified to CPU on fallback."""
        import torch

        received_device = None

        @handle_gpu_error(fallback_to_cpu=True)
        def device_func(device=None):
            nonlocal received_device
            received_device = device
            if device and str(device) != 'cpu':
                raise RuntimeError("CUDA out of memory")
            return "success"

        result = device_func(device=torch.device('cuda:0'))
        assert result == "success"
        assert str(received_device) == 'cpu'


class TestTrainingErrorHandler:
    """Tests for TrainingErrorHandler class."""

    def test_successful_step(self):
        """Successful step resets failure count."""
        handler = TrainingErrorHandler(max_retries=3)

        with handler.safe_training_step(batch_size=256) as ctx:
            # Simulate successful training
            ctx.record_success()

        assert handler._consecutive_failures == 0

    def test_oom_reduces_batch_size(self):
        """OOM error reduces batch size."""
        handler = TrainingErrorHandler(
            min_batch_size=8,
            batch_reduction_factor=0.5,
        )

        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("CUDA out of memory")

        assert handler.recommended_batch_size == 128
        assert handler._oom_count == 1

    def test_multiple_ooms_reduce_progressively(self):
        """Multiple OOMs reduce batch size progressively."""
        handler = TrainingErrorHandler(
            min_batch_size=8,
            batch_reduction_factor=0.5,
        )

        # First OOM
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("CUDA out of memory")

        assert handler.recommended_batch_size == 128

        # Second OOM
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=128):
            raise RuntimeError("CUDA out of memory")

        assert handler.recommended_batch_size == 64

    def test_min_batch_size_enforced(self):
        """Cannot reduce batch size below minimum."""
        handler = TrainingErrorHandler(
            min_batch_size=16,
            batch_reduction_factor=0.5,
        )

        with pytest.raises(NonRecoverableError, match="Cannot reduce batch size"):
            with handler.safe_training_step(batch_size=16):
                raise RuntimeError("CUDA out of memory")

    def test_max_retries_exceeded(self):
        """NonRecoverableError raised after max retries."""
        handler = TrainingErrorHandler(max_retries=2)

        # First failure
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("Generic error")

        # Second failure
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("Generic error")

        # Third failure - max exceeded
        with pytest.raises(NonRecoverableError, match="Max retries"):
            with handler.safe_training_step(batch_size=256):
                raise RuntimeError("Generic error")

    def test_success_resets_failure_count(self):
        """Successful step resets consecutive failures."""
        handler = TrainingErrorHandler(max_retries=2)

        # First failure
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("Generic error")

        assert handler._consecutive_failures == 1

        # Success
        with handler.safe_training_step(batch_size=256) as ctx:
            ctx.record_success()

        assert handler._consecutive_failures == 0

        # Another failure - should not hit max
        with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
            raise RuntimeError("Generic error")

        assert handler._consecutive_failures == 1

    def test_get_stats(self):
        """get_stats returns correct statistics."""
        handler = TrainingErrorHandler(max_retries=3)

        # Cause some OOMs
        for _ in range(2):
            with pytest.raises(RecoverableError), handler.safe_training_step(batch_size=256):
                raise RuntimeError("CUDA out of memory")

        stats = handler.get_stats()
        assert stats["oom_count"] == 2
        assert stats["consecutive_failures"] == 2
        assert stats["current_batch_size"] is not None


class TestRecoverableAndNonRecoverableErrors:
    """Tests for error classification."""

    def test_recoverable_error_inheritance(self):
        """RecoverableError is an Exception."""
        error = RecoverableError("test")
        assert isinstance(error, Exception)
        assert str(error) == "test"

    def test_non_recoverable_error_inheritance(self):
        """NonRecoverableError is an Exception."""
        error = NonRecoverableError("test")
        assert isinstance(error, Exception)
        assert str(error) == "test"

    def test_error_chaining(self):
        """Errors can be chained."""
        original = ValueError("original")
        recoverable = RecoverableError("recoverable")
        recoverable.__cause__ = original

        assert recoverable.__cause__ is original
