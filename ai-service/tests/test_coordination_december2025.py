#!/usr/bin/env python3
"""Tests for December 2025 coordination consolidation modules.

Tests the new coordination infrastructure:
- coordination_defaults: Centralized configuration constants
- event_emitters: Typed event emission functions
- async_bridge_manager: Shared executor pool
- task_decorators: Task lifecycle decorators
- unified_registry: Registry facade
"""

import asyncio
import os
import tempfile
import time
import pytest

# Skip if coordination modules not available
pytest.importorskip("app.coordination")


class TestCoordinationDefaults:
    """Tests for centralized coordination defaults."""

    def test_lock_defaults_available(self):
        """Test LockDefaults are importable and have expected values."""
        from app.config.coordination_defaults import LockDefaults

        assert LockDefaults.LOCK_TIMEOUT > 0
        assert LockDefaults.ACQUIRE_TIMEOUT > 0
        assert LockDefaults.RETRY_INTERVAL > 0
        assert LockDefaults.TRAINING_LOCK_TIMEOUT >= LockDefaults.LOCK_TIMEOUT

    def test_transport_defaults_available(self):
        """Test TransportDefaults are importable."""
        from app.config.coordination_defaults import TransportDefaults

        assert TransportDefaults.CONNECT_TIMEOUT > 0
        assert TransportDefaults.OPERATION_TIMEOUT > TransportDefaults.CONNECT_TIMEOUT
        assert TransportDefaults.MAX_RETRIES > 0

    def test_sync_defaults_available(self):
        """Test SyncDefaults are importable."""
        from app.config.coordination_defaults import SyncDefaults

        assert SyncDefaults.LOCK_TIMEOUT > 0
        assert SyncDefaults.MAX_CONCURRENT_PER_HOST >= 1
        assert SyncDefaults.DATA_SYNC_INTERVAL > 0

    def test_heartbeat_defaults_available(self):
        """Test HeartbeatDefaults are importable."""
        from app.config.coordination_defaults import HeartbeatDefaults

        assert HeartbeatDefaults.INTERVAL > 0
        assert HeartbeatDefaults.TIMEOUT > HeartbeatDefaults.INTERVAL

    def test_get_all_defaults(self):
        """Test get_all_defaults returns complete config."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()

        assert "lock" in defaults
        assert "transport" in defaults
        assert "sync" in defaults
        assert "heartbeat" in defaults
        assert "training" in defaults
        assert "scheduler" in defaults

    def test_env_override(self):
        """Test environment variable override."""
        from importlib import reload

        # Set env var before import
        original = os.environ.get("RINGRIFT_LOCK_TIMEOUT")
        os.environ["RINGRIFT_LOCK_TIMEOUT"] = "9999"

        try:
            import app.config.coordination_defaults as cd
            reload(cd)

            assert cd.LockDefaults.LOCK_TIMEOUT == 9999
        finally:
            if original:
                os.environ["RINGRIFT_LOCK_TIMEOUT"] = original
            else:
                os.environ.pop("RINGRIFT_LOCK_TIMEOUT", None)
            reload(cd)


class TestEventEmitters:
    """Tests for centralized event emitters."""

    def test_emitters_importable(self):
        """Test all emitters are importable."""
        from app.coordination.event_emitters import (
            emit_training_started,
            emit_training_complete,
            emit_training_complete_sync,
            emit_selfplay_complete,
            emit_evaluation_complete,
            emit_promotion_complete,
            emit_sync_complete,
            emit_quality_updated,
            emit_task_complete,
        )

        # All should be callables
        assert callable(emit_training_started)
        assert callable(emit_training_complete)
        assert callable(emit_training_complete_sync)
        assert callable(emit_selfplay_complete)
        assert callable(emit_evaluation_complete)
        assert callable(emit_promotion_complete)
        assert callable(emit_sync_complete)
        assert callable(emit_quality_updated)
        assert callable(emit_task_complete)

    @pytest.mark.asyncio
    async def test_emit_training_complete_no_bus(self):
        """Test emit functions work gracefully without event bus."""
        from app.coordination.event_emitters import emit_training_complete

        # Should not raise, just return False
        result = await emit_training_complete(
            job_id="test_job_123",
            board_type="square8",
            num_players=2,
            success=True,
            final_loss=0.05,
        )

        # Result depends on whether event bus is configured
        assert isinstance(result, bool)

    def test_emit_training_complete_sync(self):
        """Test sync emit function."""
        from app.coordination.event_emitters import emit_training_complete_sync

        result = emit_training_complete_sync(
            job_id="test_job_456",
            board_type="square8",
            num_players=2,
            success=True,
        )

        assert isinstance(result, bool)


class TestAsyncBridgeManager:
    """Tests for AsyncBridgeManager."""

    def test_manager_singleton(self):
        """Test singleton pattern."""
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            reset_bridge_manager,
        )

        reset_bridge_manager()

        manager1 = get_bridge_manager()
        manager2 = get_bridge_manager()

        assert manager1 is manager2

        reset_bridge_manager()

    def test_executor_initialization(self):
        """Test executor is initialized correctly."""
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            get_shared_executor,
            reset_bridge_manager,
        )

        reset_bridge_manager()

        manager = get_bridge_manager()
        manager.initialize()

        executor = get_shared_executor()
        assert executor is not None

        # Check executor is functional
        future = executor.submit(lambda: 42)
        assert future.result() == 42

        reset_bridge_manager()

    @pytest.mark.asyncio
    async def test_run_sync(self):
        """Test running sync functions in executor."""
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            reset_bridge_manager,
        )

        reset_bridge_manager()
        manager = get_bridge_manager()

        def blocking_func(x, y):
            time.sleep(0.01)
            return x + y

        result = await manager.run_sync(blocking_func, 10, 20)
        assert result == 30

        reset_bridge_manager()

    @pytest.mark.asyncio
    async def test_run_in_bridge_pool(self):
        """Test convenience function."""
        from app.coordination.async_bridge_manager import (
            run_in_bridge_pool,
            reset_bridge_manager,
        )

        reset_bridge_manager()

        result = await run_in_bridge_pool(lambda: "hello")
        assert result == "hello"

        reset_bridge_manager()

    def test_bridge_registration(self):
        """Test bridge registration."""
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            reset_bridge_manager,
        )

        reset_bridge_manager()
        manager = get_bridge_manager()

        class FakeBridge:
            def shutdown(self):
                pass

        bridge = FakeBridge()
        manager.register_bridge("test_bridge", bridge, bridge.shutdown)

        retrieved = manager.get_bridge("test_bridge")
        assert retrieved is bridge

        manager.unregister_bridge("test_bridge")
        assert manager.get_bridge("test_bridge") is None

        reset_bridge_manager()

    def test_stats_tracking(self):
        """Test statistics tracking."""
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            reset_bridge_manager,
        )

        reset_bridge_manager()
        manager = get_bridge_manager()
        manager.initialize()

        stats = manager.get_stats()

        assert "initialized" in stats
        assert stats["initialized"] is True
        assert "total_tasks_submitted" in stats
        assert "bridges_registered" in stats

        reset_bridge_manager()


class TestTaskDecorators:
    """Tests for task lifecycle decorators."""

    def test_task_context_creation(self):
        """Test TaskContext is created."""
        from app.coordination.task_decorators import (
            TaskContext,
            get_current_task_context,
        )

        ctx = TaskContext(
            task_id="test_123",
            task_type="selfplay",
            start_time=time.time(),
            board_type="square8",
            num_players=2,
        )

        assert ctx.task_id == "test_123"
        assert ctx.task_type == "selfplay"
        assert ctx.elapsed_seconds() >= 0

    def test_coordinate_task_decorator(self):
        """Test synchronous task decorator."""
        from app.coordination.task_decorators import (
            coordinate_task,
            get_current_task_context,
        )

        context_captured = None

        @coordinate_task(task_type="selfplay", emit_events=False, register_with_coordinator=False)
        def my_task(board_type: str, num_games: int) -> dict:
            nonlocal context_captured
            context_captured = get_current_task_context()
            return {"games": num_games}

        result = my_task(board_type="square8", num_games=100)

        assert result == {"games": 100}
        assert context_captured is not None
        assert context_captured.task_type == "selfplay"
        assert context_captured.board_type == "square8"

        # Context should be cleared after
        assert get_current_task_context() is None

    @pytest.mark.asyncio
    async def test_coordinate_async_task_decorator(self):
        """Test asynchronous task decorator."""
        from app.coordination.task_decorators import (
            coordinate_async_task,
            get_current_task_context,
        )

        context_captured = None

        @coordinate_async_task(task_type="training", emit_events=False, register_with_coordinator=False)
        async def my_async_task(board_type: str) -> str:
            nonlocal context_captured
            context_captured = get_current_task_context()
            await asyncio.sleep(0.01)
            return "done"

        result = await my_async_task(board_type="square8")

        assert result == "done"
        assert context_captured is not None
        assert context_captured.task_type == "training"

    def test_task_context_manager(self):
        """Test context manager usage."""
        from app.coordination.task_decorators import (
            task_context,
            get_current_task_context,
        )

        with task_context(task_type="evaluation", board_type="square8", emit_events=False) as ctx:
            assert ctx.task_type == "evaluation"
            assert get_current_task_context() is ctx

        # Should be cleared after
        assert get_current_task_context() is None

    def test_decorator_handles_exceptions(self):
        """Test decorator handles exceptions correctly."""
        from app.coordination.task_decorators import coordinate_task

        @coordinate_task(task_type="selfplay", emit_events=False, register_with_coordinator=False)
        def failing_task():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_task()


class TestUnifiedRegistry:
    """Tests for UnifiedRegistry facade."""

    def test_registry_singleton(self):
        """Test singleton pattern."""
        from app.coordination.unified_registry import (
            get_unified_registry,
            reset_unified_registry,
        )

        reset_unified_registry()

        reg1 = get_unified_registry()
        reg2 = get_unified_registry()

        assert reg1 is reg2

        reset_unified_registry()

    def test_cluster_health(self):
        """Test cluster health retrieval."""
        from app.coordination.unified_registry import (
            get_unified_registry,
            reset_unified_registry,
        )

        reset_unified_registry()
        registry = get_unified_registry()

        health = registry.get_cluster_health()

        assert hasattr(health, "healthy")
        assert hasattr(health, "registries")
        assert hasattr(health, "total_items")
        assert isinstance(health.registries, list)

        reset_unified_registry()

    def test_get_status(self):
        """Test status retrieval."""
        from app.coordination.unified_registry import (
            get_unified_registry,
            reset_unified_registry,
        )

        reset_unified_registry()
        registry = get_unified_registry()

        status = registry.get_status()

        assert "healthy" in status
        assert "registries" in status
        assert isinstance(status["registries"], dict)

        reset_unified_registry()

    def test_lazy_loading(self):
        """Test registries are lazily loaded."""
        from app.coordination.unified_registry import (
            UnifiedRegistry,
            reset_unified_registry,
        )

        reset_unified_registry()
        registry = UnifiedRegistry()

        # Internal registries should be None initially
        assert registry._model_registry is None
        assert registry._orchestrator_registry is None

        # Accessing triggers lazy load
        registry.get_models(limit=1)

        # Model registry should now be loaded (or have init error)
        # Either loaded or recorded error
        assert registry._model_registry is not None or "model_registry" in registry._init_errors

        reset_unified_registry()


class TestIntegrationScenarios:
    """Integration tests for combined functionality."""

    @pytest.mark.asyncio
    async def test_task_with_bridge_manager(self):
        """Test task decorator with bridge manager."""
        from app.coordination.task_decorators import coordinate_async_task
        from app.coordination.async_bridge_manager import (
            get_bridge_manager,
            reset_bridge_manager,
        )

        reset_bridge_manager()

        @coordinate_async_task(task_type="selfplay", emit_events=False, register_with_coordinator=False)
        async def task_using_bridge(value: int) -> int:
            manager = get_bridge_manager()
            # Run sync operation in bridge pool
            result = await manager.run_sync(lambda x: x * 2, value)
            return result

        result = await task_using_bridge(value=21)
        assert result == 42

        reset_bridge_manager()

    def test_defaults_used_by_distributed_lock(self):
        """Test distributed lock uses centralized defaults."""
        from app.coordination.distributed_lock import (
            DEFAULT_LOCK_TIMEOUT,
            DEFAULT_ACQUIRE_TIMEOUT,
        )
        from app.config.coordination_defaults import LockDefaults

        # Should use centralized defaults
        assert DEFAULT_LOCK_TIMEOUT == LockDefaults.LOCK_TIMEOUT
        assert DEFAULT_ACQUIRE_TIMEOUT == LockDefaults.ACQUIRE_TIMEOUT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
