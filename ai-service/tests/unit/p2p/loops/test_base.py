"""Tests for BaseLoop framework - P2P orchestrator background loops.

Tests cover:
- LoopStats dataclass and computed properties
- BackoffConfig validation and delay calculation
- BaseLoop lifecycle and error handling
- LoopManager coordination and health checks
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from scripts.p2p.loops.base import (
    LoopStats,
    BackoffConfig,
    BaseLoop,
    LoopManager,
)


# =============================================================================
# LoopStats Tests
# =============================================================================


class TestLoopStats:
    """Tests for LoopStats dataclass."""

    def test_default_values(self):
        """Test LoopStats has correct defaults."""
        stats = LoopStats(name="test_loop")

        assert stats.name == "test_loop"
        assert stats.total_runs == 0
        assert stats.successful_runs == 0
        assert stats.failed_runs == 0
        assert stats.consecutive_errors == 0
        assert stats.last_run_time == 0.0
        assert stats.last_success_time == 0.0
        assert stats.last_error_time == 0.0
        assert stats.last_error_message == ""
        assert stats.total_run_duration == 0.0
        assert stats.last_run_duration == 0.0

    def test_success_rate_zero_runs(self):
        """Test success_rate returns 100% for zero runs."""
        stats = LoopStats(name="test")
        assert stats.success_rate == 100.0

    def test_success_rate_all_success(self):
        """Test success_rate with all successful runs."""
        stats = LoopStats(name="test", total_runs=10, successful_runs=10)
        assert stats.success_rate == 100.0

    def test_success_rate_partial_success(self):
        """Test success_rate with mixed results."""
        stats = LoopStats(name="test", total_runs=10, successful_runs=7)
        assert stats.success_rate == 70.0

    def test_success_rate_all_failures(self):
        """Test success_rate with all failures."""
        stats = LoopStats(name="test", total_runs=10, successful_runs=0)
        assert stats.success_rate == 0.0

    def test_avg_run_duration_zero_runs(self):
        """Test avg_run_duration returns 0 for zero runs."""
        stats = LoopStats(name="test")
        assert stats.avg_run_duration == 0.0

    def test_avg_run_duration_calculation(self):
        """Test avg_run_duration calculation."""
        stats = LoopStats(
            name="test",
            successful_runs=5,
            total_run_duration=10.0,
        )
        assert stats.avg_run_duration == 2.0

    def test_to_dict(self):
        """Test to_dict() serialization."""
        stats = LoopStats(
            name="my_loop",
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            consecutive_errors=1,
            last_run_time=1000.0,
            last_success_time=1001.0,
            last_error_time=999.0,
            last_error_message="test error",
            total_run_duration=5.0,
            last_run_duration=0.5,
        )

        d = stats.to_dict()

        assert d["name"] == "my_loop"
        assert d["total_runs"] == 10
        assert d["successful_runs"] == 8
        assert d["failed_runs"] == 2
        assert d["consecutive_errors"] == 1
        assert d["last_run_time"] == 1000.0
        assert d["last_error_message"] == "test error"
        assert d["success_rate"] == 80.0
        assert d["avg_run_duration_ms"] == (5.0 / 8) * 1000
        assert d["last_run_duration_ms"] == 500.0


# =============================================================================
# BackoffConfig Tests
# =============================================================================


class TestBackoffConfig:
    """Tests for BackoffConfig dataclass."""

    def test_default_values(self):
        """Test BackoffConfig has correct defaults."""
        config = BackoffConfig()

        assert config.initial_delay == 1.0
        assert config.max_delay == 300.0
        assert config.multiplier == 1.5
        assert config.jitter == 0.1

    def test_custom_values(self):
        """Test BackoffConfig with custom values."""
        config = BackoffConfig(
            initial_delay=2.0,
            max_delay=600.0,
            multiplier=2.0,
            jitter=0.2,
        )

        assert config.initial_delay == 2.0
        assert config.max_delay == 600.0
        assert config.multiplier == 2.0
        assert config.jitter == 0.2

    def test_validation_initial_delay_zero(self):
        """Test validation rejects initial_delay <= 0."""
        with pytest.raises(ValueError, match="initial_delay must be > 0"):
            BackoffConfig(initial_delay=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects negative initial_delay."""
        with pytest.raises(ValueError, match="initial_delay must be > 0"):
            BackoffConfig(initial_delay=-1.0)

    def test_validation_max_delay_less_than_initial(self):
        """Test validation rejects max_delay < initial_delay."""
        with pytest.raises(ValueError, match="max_delay must be >= initial_delay"):
            BackoffConfig(initial_delay=10.0, max_delay=5.0)

    def test_validation_multiplier_one(self):
        """Test validation rejects multiplier <= 1."""
        with pytest.raises(ValueError, match="multiplier must be > 1"):
            BackoffConfig(multiplier=1.0)

    def test_validation_multiplier_less_than_one(self):
        """Test validation rejects multiplier < 1."""
        with pytest.raises(ValueError, match="multiplier must be > 1"):
            BackoffConfig(multiplier=0.5)

    def test_validation_jitter_negative(self):
        """Test validation rejects negative jitter."""
        with pytest.raises(ValueError, match="jitter must be between 0 and 1"):
            BackoffConfig(jitter=-0.1)

    def test_validation_jitter_greater_than_one(self):
        """Test validation rejects jitter > 1."""
        with pytest.raises(ValueError, match="jitter must be between 0 and 1"):
            BackoffConfig(jitter=1.5)

    def test_calculate_delay_zero_errors(self):
        """Test calculate_delay returns 0 for zero errors."""
        config = BackoffConfig()
        assert config.calculate_delay(0) == 0.0

    def test_calculate_delay_first_error(self):
        """Test calculate_delay for first error uses initial delay."""
        config = BackoffConfig(initial_delay=2.0, jitter=0)
        delay = config.calculate_delay(1)
        assert delay == 2.0

    def test_calculate_delay_exponential(self):
        """Test calculate_delay increases exponentially."""
        config = BackoffConfig(initial_delay=1.0, multiplier=2.0, jitter=0)

        # 1 error: 1 * 2^0 = 1
        assert config.calculate_delay(1) == 1.0
        # 2 errors: 1 * 2^1 = 2
        assert config.calculate_delay(2) == 2.0
        # 3 errors: 1 * 2^2 = 4
        assert config.calculate_delay(3) == 4.0
        # 4 errors: 1 * 2^3 = 8
        assert config.calculate_delay(4) == 8.0

    def test_calculate_delay_max_cap(self):
        """Test calculate_delay is capped at max_delay."""
        config = BackoffConfig(
            initial_delay=1.0,
            max_delay=10.0,
            multiplier=2.0,
            jitter=0,
        )

        # 5 errors: 1 * 2^4 = 16, but capped at 10
        assert config.calculate_delay(5) == 10.0
        # 10 errors: still capped at 10
        assert config.calculate_delay(10) == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test calculate_delay applies jitter."""
        config = BackoffConfig(initial_delay=10.0, jitter=0.1)

        # With 10% jitter, delay should be between 9 and 11
        delays = [config.calculate_delay(1) for _ in range(100)]
        assert min(delays) >= 9.0
        assert max(delays) <= 11.0
        # Should have some variation
        assert len(set(round(d, 3) for d in delays)) > 1


# =============================================================================
# BaseLoop Tests
# =============================================================================


class ConcreteLoop(BaseLoop):
    """Concrete implementation of BaseLoop for testing."""

    def __init__(
        self,
        name: str = "test_loop",
        interval: float = 1.0,
        **kwargs,
    ):
        super().__init__(name=name, interval=interval, **kwargs)
        self.run_count = 0
        self.error_to_raise = None
        self.on_start_called = False
        self.on_stop_called = False
        self.on_error_calls = []

    async def _run_once(self) -> None:
        self.run_count += 1
        if self.error_to_raise:
            raise self.error_to_raise

    async def _on_start(self) -> None:
        self.on_start_called = True

    async def _on_stop(self) -> None:
        self.on_stop_called = True

    async def _on_error(self, error: Exception) -> None:
        self.on_error_calls.append(error)


class TestBaseLoop:
    """Tests for BaseLoop abstract class."""

    def test_init_defaults(self):
        """Test BaseLoop initialization with defaults."""
        loop = ConcreteLoop(name="my_loop", interval=30.0)

        assert loop.name == "my_loop"
        assert loop.interval == 30.0
        assert loop.enabled is True
        assert loop.running is False
        assert loop.depends_on == []
        assert isinstance(loop.backoff_config, BackoffConfig)
        assert loop.metrics_manager is None

    def test_init_custom_backoff(self):
        """Test BaseLoop initialization with custom backoff."""
        backoff = BackoffConfig(initial_delay=5.0, max_delay=120.0)
        loop = ConcreteLoop(backoff_config=backoff)

        assert loop.backoff_config.initial_delay == 5.0
        assert loop.backoff_config.max_delay == 120.0

    def test_init_with_dependencies(self):
        """Test BaseLoop initialization with dependencies."""
        loop = ConcreteLoop(depends_on=["loop_a", "loop_b"])

        assert loop.depends_on == ["loop_a", "loop_b"]

    def test_init_disabled(self):
        """Test BaseLoop initialization when disabled."""
        loop = ConcreteLoop(enabled=False)

        assert loop.enabled is False

    def test_running_property(self):
        """Test running property reflects internal state."""
        loop = ConcreteLoop()

        assert loop.running is False
        loop._running = True
        assert loop.running is True

    def test_stats_property(self):
        """Test stats property returns LoopStats."""
        loop = ConcreteLoop()

        assert isinstance(loop.stats, LoopStats)
        assert loop.stats.name == "test_loop"

    def test_add_error_callback(self):
        """Test adding error callbacks."""
        loop = ConcreteLoop()
        callback = MagicMock()

        loop.add_error_callback(callback)

        assert callback in loop._error_callbacks

    def test_remove_error_callback(self):
        """Test removing error callbacks."""
        loop = ConcreteLoop()
        callback = MagicMock()

        loop.add_error_callback(callback)
        loop.remove_error_callback(callback)

        assert callback not in loop._error_callbacks

    def test_remove_error_callback_not_present(self):
        """Test removing non-existent callback is safe."""
        loop = ConcreteLoop()
        callback = MagicMock()

        # Should not raise
        loop.remove_error_callback(callback)

    @pytest.mark.asyncio
    async def test_run_forever_starts_and_stops(self):
        """Test run_forever lifecycle."""
        loop = ConcreteLoop(interval=0.01)

        # Run for a short time then stop
        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.05)
        loop.stop()
        await task

        assert loop.on_start_called
        assert loop.on_stop_called
        assert loop.run_count > 0

    @pytest.mark.asyncio
    async def test_run_forever_updates_stats_on_success(self):
        """Test run_forever updates stats on successful run."""
        loop = ConcreteLoop(interval=0.01)

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.05)
        loop.stop()
        await task

        assert loop.stats.total_runs > 0
        assert loop.stats.successful_runs > 0
        assert loop.stats.failed_runs == 0
        assert loop.stats.consecutive_errors == 0
        assert loop.stats.last_success_time > 0

    @pytest.mark.asyncio
    async def test_run_forever_handles_errors(self):
        """Test run_forever handles errors and applies backoff."""
        loop = ConcreteLoop(interval=0.01)
        loop.error_to_raise = ValueError("test error")

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.1)
        loop.stop()
        await task

        assert loop.stats.failed_runs > 0
        assert loop.stats.consecutive_errors > 0
        assert loop.stats.last_error_message == "test error"
        assert len(loop.on_error_calls) > 0

    @pytest.mark.asyncio
    async def test_run_forever_calls_error_callbacks(self):
        """Test run_forever invokes error callbacks."""
        loop = ConcreteLoop(interval=0.01)
        loop.error_to_raise = ValueError("callback test")
        callback = MagicMock()
        loop.add_error_callback(callback)

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.05)
        loop.stop()
        await task

        assert callback.called

    @pytest.mark.asyncio
    async def test_run_forever_skips_when_disabled(self):
        """Test run_forever skips _run_once when disabled."""
        loop = ConcreteLoop(interval=0.01, enabled=False)

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.05)
        loop.stop()
        await task

        assert loop.run_count == 0
        assert loop.on_start_called  # _on_start still called

    @pytest.mark.asyncio
    async def test_run_forever_already_running(self):
        """Test run_forever returns immediately if already running."""
        loop = ConcreteLoop(interval=0.1)
        loop._running = True

        # Should return immediately without blocking
        await asyncio.wait_for(loop.run_forever(), timeout=0.1)

    def test_stop_when_not_running(self):
        """Test stop() when loop is not running."""
        loop = ConcreteLoop()

        # Should not raise
        loop.stop()
        assert loop.running is False

    @pytest.mark.asyncio
    async def test_stop_async(self):
        """Test stop_async() with timeout."""
        loop = ConcreteLoop(interval=0.01)

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.05)

        result = await loop.stop_async(timeout=1.0)

        assert result is True
        assert loop.running is False

    @pytest.mark.asyncio
    async def test_start_background(self):
        """Test start_background() creates task."""
        loop = ConcreteLoop(interval=0.1)

        task = loop.start_background()

        assert isinstance(task, asyncio.Task)
        assert not task.done()

        # Cleanup
        assert await loop.stop_async(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_start_background_already_running(self):
        """Test start_background() returns existing task if running."""
        loop = ConcreteLoop(interval=0.1)

        task1 = loop.start_background()
        await asyncio.sleep(0.05)  # Let task start
        task2 = loop.start_background()

        assert task1 is task2

        # Cleanup
        assert await loop.stop_async(timeout=1.0) is True

    @pytest.mark.asyncio
    async def test_start_background_async(self):
        """Test start_background_async() with startup verification."""
        loop = ConcreteLoop(interval=0.01)

        task = await loop.start_background_async(timeout=1.0)

        assert task is not None
        assert loop.running is True

        # Cleanup
        assert await loop.stop_async(timeout=1.0) is True

    def test_reset_stats(self):
        """Test reset_stats() clears all statistics."""
        loop = ConcreteLoop()
        loop._stats.total_runs = 100
        loop._stats.failed_runs = 50

        loop.reset_stats()

        assert loop.stats.total_runs == 0
        assert loop.stats.failed_runs == 0

    @pytest.mark.asyncio
    async def test_reset_stats_async(self):
        """Test reset_stats_async() with lock."""
        loop = ConcreteLoop()
        loop._stats.total_runs = 100

        await loop.reset_stats_async()

        assert loop.stats.total_runs == 0

    def test_get_status_stopped(self):
        """Test get_status() when stopped."""
        loop = ConcreteLoop()

        status = loop.get_status()

        assert status["name"] == "test_loop"
        assert status["status"] == "stopped"
        assert status["running"] is False
        assert status["enabled"] is True
        assert "stats" in status
        assert "backoff" in status

    def test_get_status_running(self):
        """Test get_status() when running."""
        loop = ConcreteLoop()
        loop._running = True

        status = loop.get_status()

        assert status["status"] == "running"
        assert status["running"] is True

    def test_get_status_error(self):
        """Test get_status() with consecutive errors."""
        loop = ConcreteLoop()
        loop._running = True
        loop._stats.consecutive_errors = 5

        status = loop.get_status()

        assert status["status"] == "error"

    def test_get_status_degraded(self):
        """Test get_status() with very low success rate.

        Note: The code checks success_rate < 0.5, but success_rate is a percentage
        (0-100), so this effectively checks for < 0.5% success rate.
        """
        loop = ConcreteLoop()
        loop._running = True
        loop._stats.total_runs = 1000
        loop._stats.successful_runs = 1  # 0.1% success rate < 0.5%

        status = loop.get_status()

        assert status["status"] == "degraded"

    def test_health_check_stopped(self):
        """Test health_check() when stopped."""
        loop = ConcreteLoop()

        result = loop.health_check()

        assert result.healthy is True
        assert "stopped" in result.message.lower()

    def test_health_check_running(self):
        """Test health_check() when running normally."""
        loop = ConcreteLoop()
        loop._running = True
        loop._stats.total_runs = 10
        loop._stats.successful_runs = 9

        result = loop.health_check()

        assert result.healthy is True
        assert "operational" in result.message.lower()

    def test_health_check_too_many_errors(self):
        """Test health_check() with too many consecutive errors."""
        loop = ConcreteLoop()
        loop._running = True
        loop._stats.consecutive_errors = 6

        result = loop.health_check()

        assert result.healthy is False
        assert "consecutive errors" in result.message.lower()

    def test_health_check_low_success_rate(self):
        """Test health_check() with low success rate."""
        loop = ConcreteLoop()
        loop._running = True
        loop._stats.total_runs = 100
        loop._stats.successful_runs = 40

        result = loop.health_check()

        assert result.healthy is False
        assert "failure rate" in result.message.lower()


# =============================================================================
# LoopManager Tests
# =============================================================================


class TestLoopManager:
    """Tests for LoopManager class."""

    def test_init(self):
        """Test LoopManager initialization."""
        manager = LoopManager(name="test_manager")

        assert manager.name == "test_manager"
        assert manager.loop_names == []
        assert manager.is_started is False

    def test_register(self):
        """Test registering a loop."""
        manager = LoopManager()
        loop = ConcreteLoop(name="my_loop")

        manager.register(loop)

        assert "my_loop" in manager.loop_names
        assert manager.get("my_loop") is loop

    def test_register_replaces_existing(self):
        """Test register replaces loop with same name."""
        manager = LoopManager()
        loop1 = ConcreteLoop(name="my_loop")
        loop2 = ConcreteLoop(name="my_loop")

        manager.register(loop1)
        manager.register(loop2)

        assert manager.get("my_loop") is loop2

    def test_unregister(self):
        """Test unregistering a loop."""
        manager = LoopManager()
        loop = ConcreteLoop(name="my_loop")
        manager.register(loop)

        removed = manager.unregister("my_loop")

        assert removed is loop
        assert "my_loop" not in manager.loop_names

    def test_unregister_not_found(self):
        """Test unregister returns None for missing loop."""
        manager = LoopManager()

        removed = manager.unregister("nonexistent")

        assert removed is None

    def test_get(self):
        """Test getting a loop by name."""
        manager = LoopManager()
        loop = ConcreteLoop(name="my_loop")
        manager.register(loop)

        assert manager.get("my_loop") is loop
        assert manager.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_start_all(self):
        """Test starting all registered loops."""
        manager = LoopManager()
        loop1 = ConcreteLoop(name="loop1", interval=0.01)
        loop2 = ConcreteLoop(name="loop2", interval=0.01)
        manager.register(loop1)
        manager.register(loop2)

        results = await manager.start_all(startup_timeout=1.0)

        assert manager.is_started is True
        assert "loop1" in results
        assert "loop2" in results

        # Cleanup
        await manager.stop_all()

    @pytest.mark.asyncio
    async def test_start_all_already_started(self):
        """Test start_all returns current status if already started."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1", interval=0.01)
        manager.register(loop)

        await manager.start_all()
        results = await manager.start_all()  # Call again

        assert "loop1" in results

        # Cleanup
        await manager.stop_all()

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Test stopping all loops."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1", interval=0.01)
        manager.register(loop)

        await manager.start_all()
        results = await manager.stop_all()

        assert manager.is_started is False
        assert "loop1" in results

    @pytest.mark.asyncio
    async def test_stop_all_not_started(self):
        """Test stop_all when not started."""
        manager = LoopManager()

        results = await manager.stop_all()

        assert results == {}

    def test_get_all_status(self):
        """Test getting status of all loops."""
        manager = LoopManager()
        loop1 = ConcreteLoop(name="loop1")
        loop2 = ConcreteLoop(name="loop2")
        manager.register(loop1)
        manager.register(loop2)

        status = manager.get_all_status()

        assert "loop1" in status
        assert "loop2" in status
        assert status["loop1"]["name"] == "loop1"
        assert status["loop2"]["name"] == "loop2"

    def test_loop_names(self):
        """Test loop_names property."""
        manager = LoopManager()
        manager.register(ConcreteLoop(name="a"))
        manager.register(ConcreteLoop(name="b"))
        manager.register(ConcreteLoop(name="c"))

        assert set(manager.loop_names) == {"a", "b", "c"}

    def test_health_check_not_started(self):
        """Test health_check when not started."""
        manager = LoopManager()

        health = manager.health_check()

        assert health["status"] == "stopped"
        assert health["started"] is False

    def test_health_check_healthy(self):
        """Test health_check when all loops healthy."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1")
        loop._running = True
        manager.register(loop)
        manager._started = True

        health = manager.health_check()

        assert health["status"] == "healthy"
        assert health["loops_running"] == 1

    def test_health_check_degraded(self):
        """Test health_check when some loops failing."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1")
        loop._running = True
        loop._stats.total_runs = 10
        loop._stats.successful_runs = 3  # 30% success = 70% failure
        manager.register(loop)
        manager._started = True

        health = manager.health_check()

        assert health["status"] == "degraded"
        assert "loop1" in health["failing_loops"]

    def test_health_check_unhealthy(self):
        """Test health_check when no loops running."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1")
        loop._running = False
        manager.register(loop)
        manager._started = True

        health = manager.health_check()

        assert health["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_restart_loop(self):
        """Test restarting a stopped loop."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1", interval=0.01)
        manager.register(loop)
        manager._started = True

        result = manager.restart_loop("loop1")

        assert result is True

        # Cleanup
        assert await loop.stop_async(timeout=1.0) is True

    def test_restart_loop_not_found(self):
        """Test restart_loop with unknown loop."""
        manager = LoopManager()

        result = manager.restart_loop("nonexistent")

        assert result is False

    def test_restart_loop_already_running(self):
        """Test restart_loop when already running."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1")
        loop._running = True
        manager.register(loop)

        result = manager.restart_loop("loop1")

        assert result is True

    def test_restart_loop_disabled(self):
        """Test restart_loop when loop is disabled."""
        manager = LoopManager()
        loop = ConcreteLoop(name="loop1", enabled=False)
        manager.register(loop)

        result = manager.restart_loop("loop1")

        assert result is False

    @pytest.mark.asyncio
    async def test_restart_stopped_loops(self):
        """Test auto-restarting stopped loops."""
        manager = LoopManager()
        loop1 = ConcreteLoop(name="loop1", interval=0.01)
        loop2 = ConcreteLoop(name="loop2", interval=0.01)
        loop2._running = True  # Already running
        manager.register(loop1)
        manager.register(loop2)

        results = await manager.restart_stopped_loops()

        # Only loop1 should have been restarted
        assert "loop1" in results
        assert "loop2" not in results

        # Cleanup
        assert await loop1.stop_async(timeout=1.0) is True
        loop2.stop()

    def test_get_startup_order_no_dependencies(self):
        """Test startup order with no dependencies."""
        manager = LoopManager()
        manager.register(ConcreteLoop(name="a"))
        manager.register(ConcreteLoop(name="b"))
        manager.register(ConcreteLoop(name="c"))

        order = manager._get_startup_order()

        # All should be in the order (any order is valid)
        assert set(order) == {"a", "b", "c"}

    def test_get_startup_order_with_dependencies(self):
        """Test startup order respects dependencies."""
        manager = LoopManager()
        manager.register(ConcreteLoop(name="a"))
        manager.register(ConcreteLoop(name="b", depends_on=["a"]))
        manager.register(ConcreteLoop(name="c", depends_on=["b"]))

        order = manager._get_startup_order()

        # a must come before b, b must come before c
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_get_startup_order_complex_dependencies(self):
        """Test startup order with complex dependency graph."""
        manager = LoopManager()
        manager.register(ConcreteLoop(name="a"))
        manager.register(ConcreteLoop(name="b"))
        manager.register(ConcreteLoop(name="c", depends_on=["a", "b"]))
        manager.register(ConcreteLoop(name="d", depends_on=["c"]))

        order = manager._get_startup_order()

        # a and b must come before c, c must come before d
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_detect_dependency_cycles(self):
        """Test cycle detection in dependencies."""
        manager = LoopManager()

        # Create a cycle: a -> b -> c -> a
        manager.register(ConcreteLoop(name="a", depends_on=["c"]))
        manager.register(ConcreteLoop(name="b", depends_on=["a"]))
        manager.register(ConcreteLoop(name="c", depends_on=["b"]))

        # Build graph (reverse direction - who depends on this node)
        graph = {name: [] for name in manager._loops}
        for name, loop in manager._loops.items():
            for dep in loop.depends_on:
                if dep in manager._loops:
                    graph[dep].append(name)

        cycles = manager._detect_dependency_cycles(graph, ["a", "b", "c"])

        # Should detect at least one cycle
        assert len(cycles) > 0


class TestBaseLoopPerformanceDegradation:
    """Tests for performance degradation detection."""

    def test_check_performance_degradation_insufficient_runs(self):
        """Test degradation check skipped with few runs."""
        loop = ConcreteLoop()
        loop._stats.successful_runs = 3  # Less than 5

        # Should not emit event (not enough data)
        loop._check_performance_degradation()

        assert loop._performance_degraded_emitted is False

    def test_check_performance_degradation_triggers(self):
        """Test degradation event emitted when avg > threshold."""
        loop = ConcreteLoop(interval=10.0)
        loop._stats.successful_runs = 5
        loop._stats.total_run_duration = 30.0  # avg = 6s, > 5s (50% of interval)

        with patch.object(loop, "_emit_performance_degraded_event") as mock_emit:
            loop._check_performance_degradation()

        assert loop._performance_degraded_emitted is True
        mock_emit.assert_called_once()

    def test_check_performance_degradation_recovery(self):
        """Test degradation flag cleared on recovery."""
        loop = ConcreteLoop(interval=10.0)
        loop._performance_degraded_emitted = True
        loop._stats.successful_runs = 10
        loop._stats.total_run_duration = 20.0  # avg = 2s, < 5s

        loop._check_performance_degradation()

        assert loop._performance_degraded_emitted is False


class TestBaseLoopEventEmission:
    """Tests for event emission."""

    def test_emit_startup_failure_event(self):
        """Test startup failure event emission."""
        loop = ConcreteLoop(depends_on=["dep1"])

        with patch("app.coordination.event_emission_helpers.safe_emit_event") as mock_emit:
            loop._emit_startup_failure_event()

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[0]
            assert call_args[1]["loop_name"] == "test_loop"
            assert call_args[1]["depends_on"] == ["dep1"]

    def test_emit_startup_failure_event_import_error(self):
        """Test startup failure handles import error gracefully.

        The method catches ImportError internally so we can test by
        verifying no exception propagates.
        """
        loop = ConcreteLoop()

        # Should not raise - ImportError is caught internally
        loop._emit_startup_failure_event()

    def test_emit_performance_degraded_event_exception(self):
        """Test performance degraded handles exceptions gracefully."""
        loop = ConcreteLoop()

        with patch(
            "app.coordination.event_emission_helpers.safe_emit_event",
            side_effect=RuntimeError("test"),
        ):
            # Should not raise - exception is caught internally
            loop._emit_performance_degraded_event(5.0, 3.0)
