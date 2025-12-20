"""Tests for debug utilities."""

import logging
import time

import pytest

from app.utils.debug_utils import (
    DEBUG_ENABLED,
    Timer,
    debug_context,
    dump_state,
    format_exception_chain,
    get_call_stack,
    log_call,
    memory_usage_mb,
    timed,
    timed_async,
)


class TestTimed:
    """Tests for @timed decorator."""

    def test_returns_result(self):
        @timed
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_logs_when_over_threshold(self, caplog):
        @timed(threshold_ms=0, log_level=logging.INFO)
        def slow_func():
            time.sleep(0.01)
            return 42

        with caplog.at_level(logging.INFO):
            result = slow_func()

        assert result == 42
        assert any("slow_func" in record.message for record in caplog.records)
        assert any("ms" in record.message for record in caplog.records)

    def test_no_log_under_threshold(self, caplog):
        @timed(threshold_ms=10000, log_level=logging.INFO)
        def fast_func():
            return 42

        with caplog.at_level(logging.INFO):
            result = fast_func()

        assert result == 42
        # Should not log since under threshold
        assert not any("fast_func" in record.message for record in caplog.records)


class TestTimedAsync:
    """Tests for @timed_async decorator."""

    @pytest.mark.asyncio
    async def test_returns_result(self):
        @timed_async
        async def async_add(a, b):
            return a + b

        result = await async_add(1, 2)
        assert result == 3


class TestLogCall:
    """Tests for @log_call decorator."""

    def test_returns_result(self):
        @log_call
        def multiply(a, b):
            return a * b

        result = multiply(3, 4)
        assert result == 12


class TestTimer:
    """Tests for Timer class."""

    def test_elapsed(self):
        timer = Timer()
        time.sleep(0.01)
        assert timer.elapsed > 0
        assert timer.elapsed_ms > 0

    def test_checkpoint(self):
        timer = Timer()
        time.sleep(0.01)
        elapsed = timer.checkpoint("step1")
        assert elapsed > 0
        assert len(timer._checkpoints) == 1
        assert timer._checkpoints[0][0] == "step1"

    def test_reset(self):
        timer = Timer()
        timer.checkpoint("step1")
        timer.reset()
        assert timer._checkpoints == []

    def test_summary_empty(self):
        timer = Timer()
        summary = timer.summary()
        assert "Total:" in summary

    def test_summary_with_checkpoints(self):
        timer = Timer()
        time.sleep(0.01)
        timer.checkpoint("step1")
        time.sleep(0.01)
        timer.checkpoint("step2")

        summary = timer.summary()
        assert "step1" in summary
        assert "step2" in summary
        assert "Timing breakdown" in summary


class TestGetCallStack:
    """Tests for get_call_stack function."""

    def test_returns_list(self):
        stack = get_call_stack()
        assert isinstance(stack, list)

    def test_contains_frame_info(self):
        stack = get_call_stack(skip=1, limit=5)
        # Should contain file:line in function format
        for frame in stack:
            assert ":" in frame
            assert " in " in frame


class TestFormatExceptionChain:
    """Tests for format_exception_chain function."""

    def test_simple_exception(self):
        exc = ValueError("test error")
        formatted = format_exception_chain(exc)
        assert "ValueError" in formatted
        assert "test error" in formatted

    def test_chained_exception(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError as e:
                raise RuntimeError("outer") from e
        except RuntimeError as e:
            formatted = format_exception_chain(e)
            assert "RuntimeError" in formatted
            assert "outer" in formatted
            assert "ValueError" in formatted
            assert "inner" in formatted


class TestMemoryUsageMb:
    """Tests for memory_usage_mb function."""

    def test_returns_float(self):
        usage = memory_usage_mb()
        assert isinstance(usage, float)

    def test_positive_or_minus_one(self):
        usage = memory_usage_mb()
        # Either positive (success) or -1 (resource module unavailable)
        assert usage > 0 or usage == -1.0


class TestDumpState:
    """Tests for dump_state function."""

    def test_dict(self):
        data = {"key": "value", "count": 42}
        result = dump_state(data, name="config")
        assert "config" in result
        assert "key" in result
        assert "value" in result
        assert "42" in result

    def test_list(self):
        data = [1, 2, 3]
        result = dump_state(data)
        assert "[1, 2, 3]" in result

    def test_object(self):
        class MyClass:
            def __init__(self):
                self.value = 42

        obj = MyClass()
        result = dump_state(obj, name="obj")
        assert "obj" in result
        assert "MyClass" in result
        assert "value" in result
        assert "42" in result

    def test_max_depth(self):
        nested = {"a": {"b": {"c": {"d": "deep"}}}}
        result = dump_state(nested, max_depth=2)
        assert "..." in result

    def test_large_dict(self):
        # Test truncation for large dicts
        large = {f"key{i}": i for i in range(30)}
        result = dump_state(large)
        assert "more" in result


class TestDebugContext:
    """Tests for debug_context context manager."""

    def test_no_exception(self):
        with debug_context("test operation", key="value"):
            x = 1 + 1
        assert x == 2

    def test_exception_propagates(self):
        with pytest.raises(ValueError), debug_context("failing operation"):
            raise ValueError("test")
