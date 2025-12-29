"""Tests for async_executor_base module.

December 29, 2025: Tests for unified async executor framework.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.coordination.async_executor_base import (
    AsyncExecutor,
    TaskGroup,
    TaskInfo,
    TaskState,
)


# =============================================================================
# TaskState Tests
# =============================================================================


class TestTaskState:
    """Tests for TaskState enum."""

    def test_all_states_exist(self):
        """All expected states are defined."""
        assert TaskState.PENDING.value == "pending"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"
        assert TaskState.TIMEOUT.value == "timeout"

    def test_states_are_unique(self):
        """All state values are unique."""
        values = [s.value for s in TaskState]
        assert len(values) == len(set(values))


# =============================================================================
# TaskInfo Tests
# =============================================================================


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        info = TaskInfo(task_id="test-123", name="test_task")
        assert info.task_id == "test-123"
        assert info.name == "test_task"
        assert info.state == TaskState.PENDING
        assert info.started_at is None
        assert info.completed_at is None
        assert info.timeout_seconds is None
        assert info.error is None
        assert info.result is None

    def test_duration_not_started(self):
        """Duration returns None if task not started."""
        info = TaskInfo(task_id="test", name="test")
        assert info.duration is None

    def test_duration_running(self):
        """Duration calculated correctly while running."""
        import time

        info = TaskInfo(task_id="test", name="test")
        info.started_at = time.time() - 1.0  # Started 1 second ago
        assert info.duration is not None
        assert info.duration >= 1.0
        assert info.duration < 2.0

    def test_duration_completed(self):
        """Duration calculated correctly when completed."""
        info = TaskInfo(task_id="test", name="test")
        info.started_at = 1000.0
        info.completed_at = 1005.0
        assert info.duration == 5.0

    def test_to_dict(self):
        """to_dict() returns correct dictionary."""
        info = TaskInfo(
            task_id="abc123",
            name="my_task",
            state=TaskState.COMPLETED,
            timeout_seconds=30.0,
        )
        info.started_at = 1000.0
        info.completed_at = 1002.5
        info.result = {"data": "value"}

        d = info.to_dict()
        assert d["task_id"] == "abc123"
        assert d["name"] == "my_task"
        assert d["state"] == "completed"
        assert d["timeout_seconds"] == 30.0
        assert d["duration"] == 2.5
        assert d["error"] is None

    def test_to_dict_with_error(self):
        """to_dict() includes error string."""
        info = TaskInfo(task_id="test", name="test")
        info.error = ValueError("Something went wrong")

        d = info.to_dict()
        assert d["error"] == "Something went wrong"


# =============================================================================
# AsyncExecutor Tests
# =============================================================================


class TestAsyncExecutorLifecycle:
    """Tests for AsyncExecutor start/shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        """start() sets running flag."""
        executor = AsyncExecutor(name="test")
        assert not executor._running

        await executor.start()
        assert executor._running

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Multiple start() calls are safe."""
        executor = AsyncExecutor(name="test")

        await executor.start()
        await executor.start()  # Should not error
        assert executor._running

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_clears_running_flag(self):
        """shutdown() clears running flag."""
        executor = AsyncExecutor(name="test")
        await executor.start()
        await executor.shutdown()
        assert not executor._running

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Multiple shutdown() calls are safe."""
        executor = AsyncExecutor(name="test")
        await executor.start()
        await executor.shutdown()
        await executor.shutdown()  # Should not error
        assert not executor._running

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_tasks(self):
        """shutdown() cancels pending tasks."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        cancelled = False

        async def long_task():
            nonlocal cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled = True
                raise

        await executor.submit(long_task())
        await asyncio.sleep(0.05)  # Let task start
        await executor.shutdown(timeout=1.0)

        assert cancelled


class TestAsyncExecutorSubmit:
    """Tests for AsyncExecutor.submit()."""

    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self):
        """submit() returns a task ID."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def simple():
            return 42

        task_id = await executor.submit(simple())
        assert isinstance(task_id, str)
        assert len(task_id) == 8

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_fails_when_not_running(self):
        """submit() raises error when executor not started."""
        executor = AsyncExecutor(name="test")

        async def simple():
            return 42

        with pytest.raises(RuntimeError, match="not running"):
            await executor.submit(simple())

    @pytest.mark.asyncio
    async def test_submit_executes_coroutine(self):
        """Submitted coroutine is executed."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        result_holder = []

        async def capture():
            result_holder.append("executed")

        task_id = await executor.submit(capture())
        await executor.wait_for(task_id)

        assert result_holder == ["executed"]

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_tracks_task_info(self):
        """submit() creates TaskInfo entry."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def simple():
            return 42

        task_id = await executor.submit(simple(), name="my_task")
        await asyncio.sleep(0.05)

        info = executor.get_task_info(task_id)
        assert info is not None
        assert info.name == "my_task"

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_with_timeout(self):
        """Task-specific timeout is enforced."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def slow_task():
            await asyncio.sleep(10)

        task_id = await executor.submit(slow_task(), timeout=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await executor.wait_for(task_id)

        info = executor.get_task_info(task_id)
        assert info.state == TaskState.TIMEOUT

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_on_complete_callback(self):
        """on_complete callback is called with result."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        callback_result = []

        async def simple():
            return "success"

        def on_complete(result):
            callback_result.append(result)

        task_id = await executor.submit(simple(), on_complete=on_complete)
        await executor.wait_for(task_id)

        assert callback_result == ["success"]

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_on_error_callback(self):
        """on_error callback is called with exception."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        callback_errors = []

        async def failing():
            raise ValueError("test error")

        def on_error(e):
            callback_errors.append(str(e))

        task_id = await executor.submit(failing(), on_error=on_error)

        with pytest.raises(ValueError):
            await executor.wait_for(task_id)

        assert callback_errors == ["test error"]

        await executor.shutdown()


class TestAsyncExecutorFireAndForget:
    """Tests for AsyncExecutor.fire_and_forget()."""

    @pytest.mark.asyncio
    async def test_fire_and_forget_returns_task_id(self):
        """fire_and_forget() returns task ID when running."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def simple():
            pass

        task_id = executor.fire_and_forget(simple())
        assert task_id is not None
        assert isinstance(task_id, str)

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_fire_and_forget_returns_none_when_not_running(self):
        """fire_and_forget() returns None when not running."""
        executor = AsyncExecutor(name="test")

        async def simple():
            pass

        task_id = executor.fire_and_forget(simple())
        assert task_id is None

    @pytest.mark.asyncio
    async def test_fire_and_forget_executes_task(self):
        """fire_and_forget() actually executes the task."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        result_holder = []

        async def capture():
            result_holder.append("executed")

        executor.fire_and_forget(capture())
        await asyncio.sleep(0.1)

        assert result_holder == ["executed"]

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_fire_and_forget_error_callback(self):
        """fire_and_forget() calls error callback on failure."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        error_holder = []

        async def failing():
            raise ValueError("boom")

        def on_error(e):
            error_holder.append(str(e))

        executor.fire_and_forget(failing(), on_error=on_error)
        await asyncio.sleep(0.1)

        assert error_holder == ["boom"]

        await executor.shutdown()


class TestAsyncExecutorWait:
    """Tests for AsyncExecutor wait methods."""

    @pytest.mark.asyncio
    async def test_wait_for_returns_result(self):
        """wait_for() returns task result."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def compute():
            return 42

        task_id = await executor.submit(compute())
        result = await executor.wait_for(task_id)

        assert result == 42

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_raises_on_invalid_id(self):
        """wait_for() raises KeyError for invalid task ID."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        with pytest.raises(KeyError):
            await executor.wait_for("nonexistent")

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_with_timeout(self):
        """wait_for() respects timeout."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def slow():
            await asyncio.sleep(10)

        task_id = await executor.submit(slow())

        with pytest.raises(asyncio.TimeoutError):
            await executor.wait_for(task_id, timeout=0.1)

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_wait_all_waits_for_all_tasks(self):
        """wait_all() waits for all tasks."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        results = []

        async def append(value):
            await asyncio.sleep(0.05)
            results.append(value)

        await executor.submit(append(1))
        await executor.submit(append(2))
        await executor.submit(append(3))

        await executor.wait_all()

        assert sorted(results) == [1, 2, 3]

        await executor.shutdown()


class TestAsyncExecutorMetrics:
    """Tests for AsyncExecutor metrics collection."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """get_stats() returns correct statistics."""
        executor = AsyncExecutor(name="test", max_concurrent=50)
        await executor.start()

        async def success():
            return "ok"

        await executor.submit(success())
        await executor.wait_all()

        stats = executor.get_stats()
        assert stats["name"] == "test"
        assert stats["running"] is True
        assert stats["total_submitted"] == 1
        assert stats["total_completed"] == 1
        assert stats["max_concurrent"] == 50

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_get_pending_count(self):
        """get_pending_count() returns correct count."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def slow():
            await asyncio.sleep(1)

        await executor.submit(slow())
        await executor.submit(slow())

        assert executor.get_pending_count() == 2

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_track_failures(self):
        """Metrics correctly track failed tasks."""
        executor = AsyncExecutor(name="test")
        await executor.start()

        async def failing():
            raise ValueError("boom")

        task_id = await executor.submit(failing())
        try:
            await executor.wait_for(task_id)
        except ValueError:
            pass

        stats = executor.get_stats()
        assert stats["total_failed"] == 1

        await executor.shutdown()


# =============================================================================
# TaskGroup Tests
# =============================================================================


class TestTaskGroup:
    """Tests for TaskGroup context manager."""

    @pytest.mark.asyncio
    async def test_basic_usage(self):
        """TaskGroup executes all tasks."""
        results = []

        async def append(value):
            results.append(value)

        async with TaskGroup(name="test") as group:
            group.create_task(append(1))
            group.create_task(append(2))
            group.create_task(append(3))

        assert sorted(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_group(self):
        """Empty TaskGroup completes without error."""
        async with TaskGroup(name="test") as group:
            pass  # No tasks

        assert group._finished

    @pytest.mark.asyncio
    async def test_group_timeout(self):
        """TaskGroup respects timeout and collects error."""
        async def slow():
            await asyncio.sleep(10)

        async with TaskGroup(name="test", timeout=0.1) as group:
            group.create_task(slow())

        # Timeout is collected as error, not raised
        assert len(group._errors) == 1
        assert isinstance(group._errors[0], asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_suppress_errors_false(self):
        """Errors propagate when suppress_errors=False."""
        async def failing():
            raise ValueError("boom")

        async def success():
            return "ok"

        group = TaskGroup(name="test", suppress_errors=False)
        try:
            async with group:
                group.create_task(failing())
                group.create_task(success())
        except ValueError:
            pass  # Expected

        assert len(group._errors) == 1

    @pytest.mark.asyncio
    async def test_suppress_errors_true(self):
        """Errors collected when suppress_errors=True."""
        async def failing():
            raise ValueError("boom")

        async def success():
            return "ok"

        async with TaskGroup(name="test", suppress_errors=True) as group:
            group.create_task(failing())
            group.create_task(success())

        assert len(group._errors) == 1
        assert "ok" in group._results

    @pytest.mark.asyncio
    async def test_results_property(self):
        """results property returns successful task results."""
        async def compute(n):
            return n * 2

        async with TaskGroup(name="test", suppress_errors=True) as group:
            group.create_task(compute(1))
            group.create_task(compute(2))
            group.create_task(compute(3))

        assert sorted(group.results) == [2, 4, 6]
        assert group.success_count == 3
        assert group.error_count == 0
