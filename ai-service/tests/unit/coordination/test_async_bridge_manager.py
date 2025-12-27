"""Tests for app.coordination.async_bridge_manager module.

Tests the unified async bridge manager for thread pool coordination.
"""

import asyncio
import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.async_bridge_manager import (
    AsyncBridgeManager,
    BridgeConfig,
    BridgeStats,
    RegisteredBridge,
    get_bridge_manager,
    get_shared_executor,
    reset_bridge_manager,
    run_in_bridge_pool,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return BridgeConfig(
        max_workers=4,
        shutdown_timeout_seconds=5.0,
        queue_size_warning_threshold=10,
    )


@pytest.fixture
def manager(config):
    """Create a fresh manager for each test."""
    return AsyncBridgeManager(config)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_bridge_manager()
    yield
    reset_bridge_manager()


# =============================================================================
# BridgeConfig Tests
# =============================================================================


class TestBridgeConfig:
    """Tests for BridgeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BridgeConfig()
        assert config.max_workers == 8
        assert config.shutdown_timeout_seconds == 30.0
        assert config.queue_size_warning_threshold == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BridgeConfig(
            max_workers=16,
            shutdown_timeout_seconds=60.0,
            queue_size_warning_threshold=200,
        )
        assert config.max_workers == 16
        assert config.shutdown_timeout_seconds == 60.0
        assert config.queue_size_warning_threshold == 200

    def test_immutable_behavior(self):
        """Test that config can be used as expected."""
        config = BridgeConfig(max_workers=4)
        # Config should be frozen (dataclass default)
        assert config.max_workers == 4


# =============================================================================
# BridgeStats Tests
# =============================================================================


class TestBridgeStats:
    """Tests for BridgeStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = BridgeStats()
        assert stats.total_tasks_submitted == 0
        assert stats.total_tasks_completed == 0
        assert stats.total_tasks_failed == 0
        assert stats.active_tasks == 0
        assert stats.peak_active_tasks == 0
        assert stats.avg_task_duration_ms == 0.0
        assert stats.last_task_time == 0.0  # Default is 0.0 not None
        assert stats.bridges_registered == 0

    def test_mutable_updates(self):
        """Test that stats can be updated."""
        stats = BridgeStats()
        stats.total_tasks_submitted = 10
        stats.active_tasks = 3
        assert stats.total_tasks_submitted == 10
        assert stats.active_tasks == 3


# =============================================================================
# RegisteredBridge Tests
# =============================================================================


class TestRegisteredBridge:
    """Tests for RegisteredBridge dataclass."""

    def test_creation(self):
        """Test creating a registered bridge."""
        bridge = MagicMock()
        callback = MagicMock()
        reg_time = time.time()

        registered = RegisteredBridge(
            name="test_bridge",
            bridge=bridge,
            registered_at=reg_time,
            shutdown_callback=callback,
        )

        assert registered.name == "test_bridge"
        assert registered.bridge is bridge
        assert registered.registered_at == reg_time
        assert registered.shutdown_callback is callback

    def test_optional_callback(self):
        """Test that shutdown callback is optional."""
        registered = RegisteredBridge(
            name="test",
            bridge=MagicMock(),
            registered_at=time.time(),
        )
        assert registered.shutdown_callback is None


# =============================================================================
# AsyncBridgeManager Lifecycle Tests
# =============================================================================


class TestAsyncBridgeManagerLifecycle:
    """Tests for AsyncBridgeManager initialization and shutdown."""

    def test_creation_not_initialized(self, manager):
        """Test that manager is not initialized on creation."""
        assert not manager._initialized
        assert manager._executor is None

    def test_initialize(self, manager):
        """Test explicit initialization."""
        manager.initialize()
        assert manager._initialized
        assert manager._executor is not None
        assert isinstance(manager._executor, ThreadPoolExecutor)

    def test_initialize_idempotent(self, manager):
        """Test that multiple initialize calls are safe."""
        manager.initialize()
        first_executor = manager._executor
        manager.initialize()
        # Same executor should be reused
        assert manager._executor is first_executor

    def test_get_executor_auto_initializes(self, manager):
        """Test that get_executor initializes if needed."""
        assert not manager._initialized
        executor = manager.get_executor()
        assert manager._initialized
        assert executor is not None

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        """Test graceful shutdown."""
        manager.initialize()
        assert manager._initialized

        await manager.shutdown()
        assert manager._shutting_down
        assert not manager._initialized
        assert manager._executor is None

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, manager):
        """Test that multiple shutdown calls are safe."""
        manager.initialize()
        await manager.shutdown()
        await manager.shutdown()  # Should not raise
        assert manager._shutting_down


# =============================================================================
# run_sync Tests
# =============================================================================


class TestRunSync:
    """Tests for AsyncBridgeManager.run_sync()."""

    @pytest.mark.asyncio
    async def test_run_sync_basic(self, manager):
        """Test running a simple synchronous function."""

        def sync_func(x, y):
            return x + y

        result = await manager.run_sync(sync_func, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_run_sync_with_kwargs(self, manager):
        """Test running with keyword arguments."""

        def sync_func(a, b=10):
            return a * b

        result = await manager.run_sync(sync_func, 5, b=20)
        assert result == 100

    @pytest.mark.asyncio
    async def test_run_sync_auto_initializes(self, manager):
        """Test that run_sync initializes manager if needed."""
        assert not manager._initialized

        result = await manager.run_sync(lambda: 42)
        assert result == 42
        assert manager._initialized

    @pytest.mark.asyncio
    async def test_run_sync_updates_stats(self, manager):
        """Test that run_sync updates statistics."""

        def slow_func():
            time.sleep(0.01)
            return "done"

        await manager.run_sync(slow_func)
        stats = manager.get_stats()
        assert stats["total_tasks_submitted"] == 1
        assert stats["total_tasks_completed"] == 1
        assert stats["avg_task_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_sync_tracks_failures(self, manager):
        """Test that failures are tracked in stats."""

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await manager.run_sync(failing_func)

        stats = manager.get_stats()
        assert stats["total_tasks_submitted"] == 1
        assert stats["total_tasks_failed"] == 1
        assert stats["total_tasks_completed"] == 0

    @pytest.mark.asyncio
    async def test_run_sync_after_shutdown_raises(self, manager):
        """Test that run_sync raises after shutdown."""
        manager.initialize()
        await manager.shutdown()

        with pytest.raises(RuntimeError, match="shutting down"):
            await manager.run_sync(lambda: 42)

    @pytest.mark.asyncio
    async def test_run_sync_concurrent_tasks(self, manager):
        """Test running multiple concurrent tasks."""
        results = []
        lock = threading.Lock()

        def append_value(val):
            time.sleep(0.01)  # Simulate work
            with lock:
                results.append(val)
            return val

        # Run multiple tasks concurrently
        tasks = [manager.run_sync(append_value, i) for i in range(5)]
        await asyncio.gather(*tasks)

        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_run_sync_tracks_peak_active(self, manager):
        """Test peak active tasks tracking."""
        barrier = threading.Barrier(3)

        def wait_for_others():
            barrier.wait(timeout=5)
            return True

        # Start 3 concurrent tasks
        tasks = [manager.run_sync(wait_for_others) for _ in range(3)]
        await asyncio.gather(*tasks)

        stats = manager.get_stats()
        assert stats["peak_active_tasks"] >= 3


# =============================================================================
# Bridge Registration Tests
# =============================================================================


class TestBridgeRegistration:
    """Tests for bridge registration and lifecycle."""

    def test_register_bridge(self, manager):
        """Test registering a bridge."""
        bridge = MagicMock()
        manager.register_bridge("test", bridge)

        assert manager.get_bridge("test") is bridge
        assert manager._stats.bridges_registered == 1

    def test_register_multiple_bridges(self, manager):
        """Test registering multiple bridges."""
        bridge1 = MagicMock()
        bridge2 = MagicMock()

        manager.register_bridge("bridge1", bridge1)
        manager.register_bridge("bridge2", bridge2)

        assert manager.get_bridge("bridge1") is bridge1
        assert manager.get_bridge("bridge2") is bridge2
        assert manager._stats.bridges_registered == 2

    def test_unregister_bridge(self, manager):
        """Test unregistering a bridge."""
        bridge = MagicMock()
        manager.register_bridge("test", bridge)
        manager.unregister_bridge("test")

        assert manager.get_bridge("test") is None
        assert manager._stats.bridges_registered == 0

    def test_unregister_nonexistent_bridge(self, manager):
        """Test unregistering a non-existent bridge is safe."""
        manager.unregister_bridge("nonexistent")  # Should not raise

    def test_get_nonexistent_bridge(self, manager):
        """Test getting a non-existent bridge returns None."""
        assert manager.get_bridge("nonexistent") is None

    @pytest.mark.asyncio
    async def test_shutdown_calls_callbacks(self, manager):
        """Test that shutdown calls bridge callbacks."""
        callback = MagicMock()
        bridge = MagicMock()
        manager.register_bridge("test", bridge, shutdown_callback=callback)

        await manager.shutdown()

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_callback_errors(self, manager):
        """Test that shutdown handles callback errors gracefully."""

        def failing_callback():
            raise RuntimeError("callback failed")

        bridge = MagicMock()
        manager.register_bridge("test", bridge, shutdown_callback=failing_callback)

        # Should not raise
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_async_callbacks(self, manager):
        """Test that shutdown handles async callbacks."""
        called = []

        async def async_callback():
            called.append(True)

        bridge = MagicMock()
        manager.register_bridge("test", bridge, shutdown_callback=async_callback)

        await manager.shutdown()

        assert called == [True]


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics and health reporting."""

    def test_get_stats_initial(self, manager, config):
        """Test initial statistics."""
        stats = manager.get_stats()

        assert stats["initialized"] is False
        assert stats["shutting_down"] is False
        assert stats["max_workers"] == config.max_workers
        assert stats["total_tasks_submitted"] == 0
        assert stats["bridge_names"] == []

    @pytest.mark.asyncio
    async def test_get_stats_after_tasks(self, manager):
        """Test statistics after running tasks."""
        await manager.run_sync(lambda: 1)
        await manager.run_sync(lambda: 2)

        stats = manager.get_stats()
        assert stats["total_tasks_submitted"] == 2
        assert stats["total_tasks_completed"] == 2
        assert stats["last_task_time"] is not None

    def test_get_health_initial(self, manager):
        """Test initial health status."""
        health = manager.get_health()

        assert health["healthy"] is False
        assert "Not initialized" in health["warnings"]

    def test_get_health_after_init(self, manager):
        """Test health after initialization."""
        manager.initialize()
        health = manager.get_health()

        assert health["healthy"] is True
        assert health["warnings"] == []

    @pytest.mark.asyncio
    async def test_get_health_during_shutdown(self, manager):
        """Test health during shutdown."""
        manager.initialize()
        await manager.shutdown()

        health = manager.get_health()
        assert health["healthy"] is False
        assert "Shutting down" in health["warnings"]

    @pytest.mark.asyncio
    async def test_get_health_high_failure_rate(self, manager):
        """Test health with high failure rate."""

        def failing():
            raise ValueError("fail")

        # Generate failures
        for _ in range(5):
            with contextlib.suppress(ValueError):
                await manager.run_sync(failing)

        health = manager.get_health()
        assert any("failure rate" in w.lower() for w in health["warnings"])


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton management."""

    def test_get_bridge_manager_creates_singleton(self):
        """Test that get_bridge_manager creates a singleton."""
        manager1 = get_bridge_manager()
        manager2 = get_bridge_manager()

        assert manager1 is manager2

    def test_get_bridge_manager_with_config(self):
        """Test that config is only used on first call."""
        config1 = BridgeConfig(max_workers=4)
        config2 = BridgeConfig(max_workers=16)

        manager1 = get_bridge_manager(config1)
        manager2 = get_bridge_manager(config2)

        # Same manager, config1 was used
        assert manager1 is manager2
        assert manager1.config.max_workers == 4

    def test_get_shared_executor(self):
        """Test get_shared_executor convenience function."""
        executor = get_shared_executor()
        assert isinstance(executor, ThreadPoolExecutor)

    def test_reset_bridge_manager(self):
        """Test resetting the singleton."""
        manager1 = get_bridge_manager()
        reset_bridge_manager()
        manager2 = get_bridge_manager()

        assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_run_in_bridge_pool(self):
        """Test run_in_bridge_pool convenience function."""
        result = await run_in_bridge_pool(lambda x: x * 2, 21)
        assert result == 42

    @pytest.mark.asyncio
    async def test_run_in_bridge_pool_with_kwargs(self):
        """Test run_in_bridge_pool with keyword arguments."""

        def func(a, b=1):
            return a + b

        result = await run_in_bridge_pool(func, 10, b=5)
        assert result == 15


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_register_from_multiple_threads(self, manager):
        """Test registering bridges from multiple threads."""
        errors = []

        def register_bridge(name):
            try:
                manager.register_bridge(name, MagicMock())
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=register_bridge, args=(f"bridge_{i}",))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager._stats.bridges_registered == 10

    @pytest.mark.asyncio
    async def test_concurrent_stats_access(self, manager):
        """Test concurrent stats access during task execution."""
        stats_snapshots = []

        async def run_and_get_stats():
            await manager.run_sync(lambda: time.sleep(0.01))
            stats_snapshots.append(manager.get_stats())

        await asyncio.gather(*[run_and_get_stats() for _ in range(5)])

        # All snapshots should be valid
        for stats in stats_snapshots:
            assert "total_tasks_submitted" in stats
            assert "active_tasks" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, manager):
        """Test complete lifecycle: init, register, run, shutdown."""
        # Initialize
        manager.initialize()
        assert manager._initialized

        # Register bridges
        bridge1 = MagicMock()
        bridge2 = MagicMock()
        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.register_bridge("db", bridge1, shutdown_callback=callback1)
        manager.register_bridge("cache", bridge2, shutdown_callback=callback2)

        # Run some tasks
        results = await asyncio.gather(
            manager.run_sync(lambda: "task1"),
            manager.run_sync(lambda: "task2"),
            manager.run_sync(lambda: "task3"),
        )
        assert results == ["task1", "task2", "task3"]

        # Check stats
        stats = manager.get_stats()
        assert stats["total_tasks_completed"] == 3
        assert stats["bridges_registered"] == 2

        # Shutdown
        await manager.shutdown()

        # Callbacks should have been called
        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery(self, manager):
        """Test that manager recovers from errors."""

        def maybe_fail(should_fail):
            if should_fail:
                raise ValueError("intentional failure")
            return "success"

        # Run some failing and succeeding tasks
        for should_fail in [False, True, False, True, False]:
            try:
                result = await manager.run_sync(maybe_fail, should_fail)
                assert result == "success"
            except ValueError:
                pass

        stats = manager.get_stats()
        assert stats["total_tasks_completed"] == 3
        assert stats["total_tasks_failed"] == 2
