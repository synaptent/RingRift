"""Tests for task_decorators.py - Task lifecycle coordination decorators.

Tests cover:
- TaskContext dataclass and methods
- Task context thread-local storage
- Task ID generation
- Board info extraction from function arguments
- coordinate_task decorator (sync)
- coordinate_async_task decorator (async)
- task_context context manager
- Event emission (mocked)
- Error handling and cleanup
"""

import asyncio
import re
import time
import threading
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.coordination.task_decorators import (
    TaskContext,
    get_current_task_context,
    _set_task_context,
    _generate_task_id,
    _extract_board_info,
    coordinate_task,
    coordinate_async_task,
    task_context,
)


class TestTaskContext:
    """Tests for TaskContext dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating TaskContext with required fields only."""
        ctx = TaskContext(
            task_id="test_123",
            task_type="selfplay",
            start_time=1000.0,
        )
        assert ctx.task_id == "test_123"
        assert ctx.task_type == "selfplay"
        assert ctx.start_time == 1000.0
        assert ctx.board_type is None
        assert ctx.num_players is None
        assert ctx.node_id is None
        assert ctx.metadata == {}

    def test_creation_with_all_fields(self):
        """Test creating TaskContext with all fields."""
        metadata = {"batch_size": 512, "model": "v2"}
        ctx = TaskContext(
            task_id="training_456",
            task_type="training",
            start_time=2000.0,
            board_type="hex8",
            num_players=4,
            node_id="worker-1",
            metadata=metadata,
        )
        assert ctx.task_id == "training_456"
        assert ctx.task_type == "training"
        assert ctx.start_time == 2000.0
        assert ctx.board_type == "hex8"
        assert ctx.num_players == 4
        assert ctx.node_id == "worker-1"
        assert ctx.metadata == metadata

    def test_elapsed_seconds(self):
        """Test elapsed_seconds() method."""
        start = time.time() - 5.0  # Started 5 seconds ago
        ctx = TaskContext(
            task_id="test",
            task_type="test",
            start_time=start,
        )
        elapsed = ctx.elapsed_seconds()
        assert 4.9 <= elapsed <= 6.0  # Allow for timing variance

    def test_elapsed_seconds_zero(self):
        """Test elapsed_seconds() when just started."""
        ctx = TaskContext(
            task_id="test",
            task_type="test",
            start_time=time.time(),
        )
        elapsed = ctx.elapsed_seconds()
        assert 0.0 <= elapsed <= 0.1

    def test_metadata_default_factory(self):
        """Test that metadata defaults to empty dict, not shared."""
        ctx1 = TaskContext(task_id="t1", task_type="test", start_time=0)
        ctx2 = TaskContext(task_id="t2", task_type="test", start_time=0)
        ctx1.metadata["key"] = "value"
        assert "key" not in ctx2.metadata  # Separate dicts

    def test_mutable_metadata(self):
        """Test that metadata can be modified."""
        ctx = TaskContext(task_id="t1", task_type="test", start_time=0)
        ctx.metadata["added"] = True
        assert ctx.metadata["added"] is True


class TestTaskContextThreadLocal:
    """Tests for thread-local task context storage."""

    def test_get_context_when_not_set(self):
        """Test get_current_task_context() returns None when not set."""
        _set_task_context(None)  # Ensure clean state
        assert get_current_task_context() is None

    def test_set_and_get_context(self):
        """Test setting and getting task context."""
        ctx = TaskContext(task_id="test", task_type="test", start_time=0)
        _set_task_context(ctx)
        try:
            assert get_current_task_context() is ctx
        finally:
            _set_task_context(None)

    def test_clear_context(self):
        """Test clearing task context."""
        ctx = TaskContext(task_id="test", task_type="test", start_time=0)
        _set_task_context(ctx)
        _set_task_context(None)
        assert get_current_task_context() is None

    def test_context_is_thread_local(self):
        """Test that context is isolated per thread."""
        results = {}

        def thread_func(thread_id: str):
            # Initially no context
            results[f"{thread_id}_initial"] = get_current_task_context()

            # Set thread-specific context
            ctx = TaskContext(task_id=thread_id, task_type="test", start_time=0)
            _set_task_context(ctx)
            time.sleep(0.01)  # Allow other thread to run

            # Should still be our context
            results[f"{thread_id}_after"] = get_current_task_context()
            _set_task_context(None)

        t1 = threading.Thread(target=thread_func, args=("thread1",))
        t2 = threading.Thread(target=thread_func, args=("thread2",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have seen its own context
        assert results["thread1_after"].task_id == "thread1"
        assert results["thread2_after"].task_id == "thread2"


class TestGenerateTaskId:
    """Tests for _generate_task_id function."""

    def test_format(self):
        """Test task ID format: {task_type}_{timestamp}_{unique}."""
        task_id = _generate_task_id("selfplay")
        parts = task_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "selfplay"
        assert parts[1].isdigit()  # Timestamp
        assert len(parts[2]) == 8  # UUID hex[:8]

    def test_different_task_types(self):
        """Test task IDs for different task types."""
        selfplay_id = _generate_task_id("selfplay")
        training_id = _generate_task_id("training")
        eval_id = _generate_task_id("evaluation")

        assert selfplay_id.startswith("selfplay_")
        assert training_id.startswith("training_")
        assert eval_id.startswith("evaluation_")

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [_generate_task_id("test") for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_timestamp_is_current(self):
        """Test that timestamp in ID is close to current time."""
        before = int(time.time())
        task_id = _generate_task_id("test")
        after = int(time.time())

        timestamp = int(task_id.split("_")[1])
        assert before <= timestamp <= after


class TestExtractBoardInfo:
    """Tests for _extract_board_info function."""

    def test_from_kwargs(self):
        """Test extracting board info from kwargs."""
        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={"board_type": "hex8", "num_players": 4},
        )
        assert board_type == "hex8"
        assert num_players == 4

    def test_from_kwargs_custom_param_names(self):
        """Test extracting with custom parameter names."""
        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={"bt": "square8", "np": 2},
            board_type_param="bt",
            num_players_param="np",
        )
        assert board_type == "square8"
        assert num_players == 2

    def test_from_config_object_in_kwargs(self):
        """Test extracting from config object in kwargs."""
        config = MagicMock()
        config.board_type = "hexagonal"
        config.num_players = 3

        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={"config": config},
        )
        assert board_type == "hexagonal"
        assert num_players == 3

    def test_from_config_object_in_args(self):
        """Test extracting from config object as first positional arg."""
        config = MagicMock()
        config.board_type = "square19"
        config.num_players = 2

        board_type, num_players = _extract_board_info(
            args=(config,),
            kwargs={},
        )
        assert board_type == "square19"
        assert num_players == 2

    def test_missing_info_returns_none(self):
        """Test that missing info returns None."""
        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={},
        )
        assert board_type is None
        assert num_players is None

    def test_partial_info(self):
        """Test extracting only board_type without num_players."""
        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={"board_type": "hex8"},
        )
        assert board_type == "hex8"
        assert num_players is None

    def test_kwargs_takes_precedence_over_config(self):
        """Test that explicit kwargs override config object."""
        config = MagicMock()
        config.board_type = "from_config"
        config.num_players = 99

        board_type, num_players = _extract_board_info(
            args=(),
            kwargs={"config": config, "board_type": "from_kwargs"},
        )
        assert board_type == "from_kwargs"  # kwargs wins


class TestCoordinateTaskDecorator:
    """Tests for coordinate_task decorator."""

    def test_basic_function_execution(self):
        """Test that decorated function executes normally."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_func(x: int, y: int) -> int:
            return x + y

        result = my_func(2, 3)
        assert result == 5

    def test_task_context_available_inside_function(self):
        """Test that task context is available inside decorated function."""
        captured_ctx = None

        @coordinate_task(task_type="selfplay", emit_events=False, register_with_coordinator=False)
        def my_func():
            nonlocal captured_ctx
            captured_ctx = get_current_task_context()
            return "done"

        my_func()

        assert captured_ctx is not None
        assert captured_ctx.task_type == "selfplay"
        assert captured_ctx.task_id.startswith("selfplay_")

    def test_task_context_cleared_after_function(self):
        """Test that task context is cleared after function completes."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_func():
            return "done"

        my_func()
        assert get_current_task_context() is None

    def test_task_context_cleared_on_exception(self):
        """Test that task context is cleared even when function raises."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            my_func()

        assert get_current_task_context() is None

    def test_board_info_extracted_from_kwargs(self):
        """Test that board info is extracted and set on context."""
        captured_ctx = None

        @coordinate_task(task_type="selfplay", emit_events=False, register_with_coordinator=False)
        def my_func(board_type: str, num_players: int):
            nonlocal captured_ctx
            captured_ctx = get_current_task_context()

        my_func(board_type="hex8", num_players=4)

        assert captured_ctx.board_type == "hex8"
        assert captured_ctx.num_players == 4

    def test_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_documented_func(x: int) -> str:
            """This is my docstring."""
            return str(x)

        assert my_documented_func.__name__ == "my_documented_func"
        assert "docstring" in my_documented_func.__doc__

    def test_exception_propagated(self):
        """Test that exceptions are re-raised."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_func():
            raise RuntimeError("original error")

        with pytest.raises(RuntimeError, match="original error"):
            my_func()


class TestCoordinateAsyncTaskDecorator:
    """Tests for coordinate_async_task decorator."""

    @pytest.mark.asyncio
    async def test_basic_async_execution(self):
        """Test that decorated async function executes normally."""
        @coordinate_async_task(task_type="test", emit_events=False, register_with_coordinator=False)
        async def my_async_func(x: int, y: int) -> int:
            await asyncio.sleep(0)
            return x + y

        result = await my_async_func(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_task_context_available_inside_async_function(self):
        """Test that task context is available inside async decorated function."""
        captured_ctx = None

        @coordinate_async_task(task_type="training", emit_events=False, register_with_coordinator=False)
        async def my_async_func():
            nonlocal captured_ctx
            await asyncio.sleep(0)
            captured_ctx = get_current_task_context()
            return "done"

        await my_async_func()

        assert captured_ctx is not None
        assert captured_ctx.task_type == "training"
        assert captured_ctx.task_id.startswith("training_")

    @pytest.mark.asyncio
    async def test_task_context_cleared_after_async_function(self):
        """Test that task context is cleared after async function completes."""
        @coordinate_async_task(task_type="test", emit_events=False, register_with_coordinator=False)
        async def my_async_func():
            await asyncio.sleep(0)
            return "done"

        await my_async_func()
        assert get_current_task_context() is None

    @pytest.mark.asyncio
    async def test_task_context_cleared_on_async_exception(self):
        """Test that task context is cleared when async function raises."""
        @coordinate_async_task(task_type="test", emit_events=False, register_with_coordinator=False)
        async def my_async_func():
            await asyncio.sleep(0)
            raise ValueError("async error")

        with pytest.raises(ValueError):
            await my_async_func()

        assert get_current_task_context() is None

    @pytest.mark.asyncio
    async def test_async_exception_propagated(self):
        """Test that async exceptions are re-raised."""
        @coordinate_async_task(task_type="test", emit_events=False, register_with_coordinator=False)
        async def my_async_func():
            raise RuntimeError("async original error")

        with pytest.raises(RuntimeError, match="async original error"):
            await my_async_func()


class TestTaskContextManager:
    """Tests for task_context context manager."""

    def test_basic_context_manager(self):
        """Test basic context manager usage."""
        with task_context(task_type="sync", emit_events=False) as ctx:
            assert ctx is not None
            assert ctx.task_type == "sync"
            assert ctx.task_id.startswith("sync_")

    def test_context_available_inside_block(self):
        """Test that context is available via get_current_task_context()."""
        with task_context(task_type="eval", emit_events=False) as ctx:
            current = get_current_task_context()
            assert current is ctx

    def test_context_cleared_after_block(self):
        """Test that context is cleared after exiting block."""
        with task_context(task_type="test", emit_events=False):
            pass
        assert get_current_task_context() is None

    def test_context_cleared_on_exception(self):
        """Test that context is cleared when exception raised in block."""
        with pytest.raises(ValueError):
            with task_context(task_type="test", emit_events=False):
                raise ValueError("block error")

        assert get_current_task_context() is None

    def test_exception_propagated(self):
        """Test that exceptions are re-raised from context block."""
        with pytest.raises(RuntimeError, match="block runtime error"):
            with task_context(task_type="test", emit_events=False):
                raise RuntimeError("block runtime error")

    def test_custom_task_id(self):
        """Test providing custom task ID."""
        with task_context(task_type="sync", task_id="custom_id_123", emit_events=False) as ctx:
            assert ctx.task_id == "custom_id_123"

    def test_board_info(self):
        """Test providing board info to context manager."""
        with task_context(
            task_type="selfplay",
            board_type="square8",
            num_players=3,
            emit_events=False,
        ) as ctx:
            assert ctx.board_type == "square8"
            assert ctx.num_players == 3

    def test_elapsed_time_tracking(self):
        """Test that elapsed time is tracked correctly."""
        with task_context(task_type="test", emit_events=False) as ctx:
            time.sleep(0.1)
            elapsed = ctx.elapsed_seconds()
            assert elapsed >= 0.1


class TestEventEmission:
    """Tests for event emission behavior (mocked)."""

    @patch("app.coordination.task_decorators._emit_task_started")
    @patch("app.coordination.task_decorators._emit_task_complete_sync")
    def test_sync_decorator_emits_events(self, mock_complete, mock_started):
        """Test that sync decorator emits start and complete events."""
        @coordinate_task(task_type="test", emit_events=True, register_with_coordinator=False)
        def my_func():
            return "done"

        my_func()

        # Started should be called with context and emit_events=True
        mock_started.assert_called_once()
        call_args = mock_started.call_args
        assert call_args[0][0].task_type == "test"
        assert call_args[0][1] is True  # emit_events

        # Complete should be called
        mock_complete.assert_called_once()

    @patch("app.coordination.task_decorators._emit_task_started")
    @patch("app.coordination.task_decorators._emit_task_failed_sync")
    def test_sync_decorator_emits_failure_on_exception(self, mock_failed, mock_started):
        """Test that sync decorator emits failure event on exception."""
        @coordinate_task(task_type="test", emit_events=True, register_with_coordinator=False)
        def my_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            my_func()

        mock_started.assert_called_once()
        mock_failed.assert_called_once()

    @patch("app.coordination.task_decorators._emit_task_started")
    @patch("app.coordination.task_decorators._emit_task_complete_sync")
    def test_events_not_emitted_when_disabled(self, mock_complete, mock_started):
        """Test that no events emitted when emit_events=False."""
        @coordinate_task(task_type="test", emit_events=False, register_with_coordinator=False)
        def my_func():
            return "done"

        my_func()

        # Events called but with emit_events=False
        mock_started.assert_called_once()
        assert mock_started.call_args[0][1] is False


class TestCoordinatorRegistration:
    """Tests for TaskCoordinator registration (mocked)."""

    def test_registers_with_coordinator_when_enabled(self):
        """Test that task is registered with coordinator when enabled."""
        mock_coordinator = MagicMock()

        # Patch the import inside the wrapper function
        with patch.dict(
            "sys.modules",
            {"app.coordination.task_coordinator": MagicMock(get_task_coordinator=lambda: mock_coordinator)}
        ):
            @coordinate_task(task_type="training", emit_events=False, register_with_coordinator=True)
            def my_func():
                return "done"

            my_func()

            mock_coordinator.register_task.assert_called_once()
            call_args = mock_coordinator.register_task.call_args
            assert "training_" in call_args[0][0]  # task_id starts with training_
            assert call_args[0][1] == "training"  # task_type

    def test_does_not_register_when_disabled(self):
        """Test that task is not registered when register_with_coordinator=False."""
        mock_coordinator = MagicMock()
        mock_module = MagicMock(get_task_coordinator=lambda: mock_coordinator)

        # Even if we patch the module, it shouldn't be called when disabled
        with patch.dict("sys.modules", {"app.coordination.task_coordinator": mock_module}):
            @coordinate_task(task_type="training", emit_events=False, register_with_coordinator=False)
            def my_func():
                return "done"

            my_func()

            # register_task should NOT be called since register_with_coordinator=False
            mock_coordinator.register_task.assert_not_called()


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test that __all__ exports are defined."""
        from app.coordination import task_decorators
        assert hasattr(task_decorators, "__all__")
        expected = [
            "TaskContext",
            "coordinate_async_task",
            "coordinate_task",
            "get_current_task_context",
            "task_context",
        ]
        for name in expected:
            assert name in task_decorators.__all__

    def test_all_exports_importable(self):
        """Test that all exports are importable."""
        from app.coordination.task_decorators import (
            TaskContext,
            coordinate_async_task,
            coordinate_task,
            get_current_task_context,
            task_context,
        )
        assert TaskContext is not None
        assert coordinate_async_task is not None
        assert coordinate_task is not None
        assert get_current_task_context is not None
        assert task_context is not None
