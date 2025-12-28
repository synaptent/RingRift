"""Unit tests for P2P Loop Framework (BaseLoop, LoopManager).

Tests the background loop infrastructure used by the P2P orchestrator.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loops.base import (
    BackoffConfig,
    BaseLoop,
    LoopManager,
    LoopStats,
)

# =============================================================================
# LoopStats Tests
# =============================================================================


class TestLoopStats:
    """Test LoopStats dataclass."""

    def test_initial_values(self):
        """Should have correct initial values."""
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

    def test_success_rate_no_runs(self):
        """Success rate should be 100% when no runs yet."""
        stats = LoopStats(name="test")
        assert stats.success_rate == 100.0

    def test_success_rate_all_success(self):
        """Success rate should be 100% when all runs successful."""
        stats = LoopStats(name="test", total_runs=10, successful_runs=10)
        assert stats.success_rate == 100.0

    def test_success_rate_partial(self):
        """Success rate should reflect partial success."""
        stats = LoopStats(name="test", total_runs=10, successful_runs=7, failed_runs=3)
        assert stats.success_rate == 70.0

    def test_avg_run_duration_no_runs(self):
        """Average duration should be 0 when no successful runs."""
        stats = LoopStats(name="test")
        assert stats.avg_run_duration == 0.0

    def test_avg_run_duration(self):
        """Average duration should be calculated correctly."""
        stats = LoopStats(
            name="test",
            successful_runs=5,
            total_run_duration=10.0,
        )
        assert stats.avg_run_duration == 2.0

    def test_to_dict(self):
        """Should convert to dictionary with expected keys."""
        stats = LoopStats(
            name="test",
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            consecutive_errors=1,
            last_error_message="Test error",
        )

        result = stats.to_dict()

        assert result["name"] == "test"
        assert result["total_runs"] == 10
        assert result["successful_runs"] == 8
        assert result["failed_runs"] == 2
        assert result["consecutive_errors"] == 1
        assert result["last_error_message"] == "Test error"
        assert result["success_rate"] == 80.0
        assert "avg_run_duration_ms" in result
        assert "last_run_duration_ms" in result


# =============================================================================
# BackoffConfig Tests
# =============================================================================


class TestBackoffConfig:
    """Test BackoffConfig class."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = BackoffConfig()

        assert config.initial_delay == 1.0
        assert config.max_delay == 300.0
        assert config.multiplier == 2.0
        assert config.jitter == 0.1

    def test_zero_errors(self):
        """Should return 0 delay for zero consecutive errors."""
        config = BackoffConfig()
        assert config.calculate_delay(0) == 0.0

    def test_exponential_increase(self):
        """Should increase delay exponentially."""
        config = BackoffConfig(initial_delay=1.0, multiplier=2.0, jitter=0.0)

        # First error: 1.0
        # Second error: 2.0
        # Third error: 4.0
        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(2)
        delay3 = config.calculate_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        config = BackoffConfig(initial_delay=1.0, max_delay=10.0, multiplier=2.0, jitter=0.0)

        # 10th error would be 2^9 = 512, but should be capped at 10
        delay = config.calculate_delay(10)
        assert delay == 10.0

    def test_jitter_adds_variance(self):
        """Should add jitter to delay."""
        config = BackoffConfig(initial_delay=10.0, jitter=0.5, multiplier=2.0)

        # Run multiple times to check jitter adds variance
        delays = [config.calculate_delay(1) for _ in range(20)]

        # Should have some variance (not all same value)
        assert len(set(delays)) > 1

        # Should be within jitter range (10 +/- 50% = 5 to 15)
        for delay in delays:
            assert 5.0 <= delay <= 15.0


# =============================================================================
# BaseLoop Tests
# =============================================================================


class SimpleLoop(BaseLoop):
    """Simple test loop that counts runs."""

    def __init__(self, **kwargs):
        super().__init__(name="simple", interval=0.01, **kwargs)
        self.run_count = 0
        self.on_start_called = False
        self.on_stop_called = False
        self.on_error_called = False
        self.raise_error = False
        self.error_to_raise = ValueError("test error")

    async def _run_once(self) -> None:
        self.run_count += 1
        if self.raise_error:
            raise self.error_to_raise

    async def _on_start(self) -> None:
        self.on_start_called = True

    async def _on_stop(self) -> None:
        self.on_stop_called = True

    async def _on_error(self, error: Exception) -> None:
        self.on_error_called = True


class TestBaseLoop:
    """Test BaseLoop class."""

    def test_initialization(self):
        """Should initialize with correct values."""
        loop = SimpleLoop()

        assert loop.name == "simple"
        assert loop.interval == 0.01
        assert loop.enabled is True
        assert loop.running is False
        assert loop.stats.name == "simple"

    def test_disabled_loop(self):
        """Should not run when disabled."""
        loop = SimpleLoop(enabled=False)
        assert loop.enabled is False

    @pytest.mark.asyncio
    async def test_run_and_stop(self):
        """Should run and stop cleanly."""
        loop = SimpleLoop()

        # Start in background
        loop.start_background()
        await asyncio.sleep(0.1)  # Let loop start (increased from 0.01 for CI stability)
        assert loop.running is True

        # Wait for a few runs
        await asyncio.sleep(0.05)
        assert loop.run_count >= 1
        assert loop.on_start_called is True

        # Stop
        loop.stop()
        await asyncio.sleep(0.02)

        assert loop.running is False
        assert loop.on_stop_called is True

    @pytest.mark.asyncio
    async def test_stats_updated_on_success(self):
        """Should update stats on successful runs."""
        loop = SimpleLoop()

        loop.start_background()
        await asyncio.sleep(0.05)
        loop.stop()
        await asyncio.sleep(0.02)

        assert loop.stats.total_runs >= 1
        assert loop.stats.successful_runs >= 1
        assert loop.stats.failed_runs == 0
        assert loop.stats.consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_stats_updated_on_error(self):
        """Should update stats on error."""
        loop = SimpleLoop()
        loop.raise_error = True

        loop.start_background()
        await asyncio.sleep(0.05)
        loop.stop()
        await asyncio.sleep(0.02)

        assert loop.stats.total_runs >= 1
        assert loop.stats.failed_runs >= 1
        assert loop.stats.consecutive_errors >= 1
        assert loop.on_error_called is True

    @pytest.mark.asyncio
    async def test_error_callback(self):
        """Should call error callbacks on error."""
        loop = SimpleLoop()
        loop.raise_error = True

        callback_called = []

        def callback(e):
            callback_called.append(e)

        loop.add_error_callback(callback)

        loop.start_background()
        await asyncio.sleep(0.05)
        loop.stop()
        await asyncio.sleep(0.02)

        assert len(callback_called) >= 1
        assert isinstance(callback_called[0], ValueError)

    @pytest.mark.asyncio
    async def test_remove_error_callback(self):
        """Should be able to remove error callbacks."""
        loop = SimpleLoop()
        loop.raise_error = True

        callback_called = []

        def callback(e):
            callback_called.append(e)

        loop.add_error_callback(callback)
        loop.remove_error_callback(callback)

        loop.start_background()
        await asyncio.sleep(0.03)
        loop.stop()
        await asyncio.sleep(0.02)

        assert len(callback_called) == 0

    @pytest.mark.asyncio
    async def test_disabled_loop_skips_run(self):
        """Should skip running when disabled."""
        loop = SimpleLoop(enabled=False)

        loop.start_background()
        await asyncio.sleep(0.03)
        loop.stop()
        await asyncio.sleep(0.02)

        assert loop.run_count == 0

    @pytest.mark.asyncio
    async def test_toggle_enabled(self):
        """Should respect enabled toggle during execution."""
        loop = SimpleLoop(enabled=True)

        loop.start_background()
        await asyncio.sleep(0.03)

        initial_count = loop.run_count
        assert initial_count >= 1

        # Disable
        loop.enabled = False
        await asyncio.sleep(0.03)

        count_after_disable = loop.run_count
        # Should have stopped incrementing (or only 1 more in-flight run)
        assert count_after_disable <= initial_count + 1

        loop.stop()
        await asyncio.sleep(0.02)

    def test_get_status(self):
        """Should return status dictionary."""
        loop = SimpleLoop()

        status = loop.get_status()

        assert status["name"] == "simple"
        assert status["running"] is False
        assert status["enabled"] is True
        assert status["interval"] == 0.01
        assert "stats" in status
        assert "backoff" in status

    def test_reset_stats(self):
        """Should reset stats to initial values."""
        loop = SimpleLoop()
        loop._stats.total_runs = 100
        loop._stats.failed_runs = 50

        loop.reset_stats()

        assert loop.stats.total_runs == 0
        assert loop.stats.failed_runs == 0

    @pytest.mark.asyncio
    async def test_stop_async_timeout(self):
        """Should handle timeout in stop_async."""

        class SlowLoop(BaseLoop):
            async def _run_once(self) -> None:
                await asyncio.sleep(10)  # Very slow

        loop = SlowLoop(name="slow", interval=0.01)
        loop.start_background()

        # Wait for it to start running
        await asyncio.sleep(0.05)

        # Stop with very short timeout
        result = await loop.stop_async(timeout=0.01)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_already_running_warning(self):
        """Should warn if trying to run already running loop."""
        loop = SimpleLoop()

        task = loop.start_background()
        await asyncio.sleep(0.02)

        # Try to run again - should return same task
        task2 = loop.start_background()
        assert task is task2

        loop.stop()
        await asyncio.sleep(0.02)


# =============================================================================
# LoopManager Tests
# =============================================================================


class TestLoopManager:
    """Test LoopManager class."""

    def test_initialization(self):
        """Should initialize correctly."""
        manager = LoopManager(name="test_manager")

        assert manager.name == "test_manager"
        assert manager.loop_names == []
        assert manager.is_started is False

    def test_register(self):
        """Should register loops."""
        manager = LoopManager()
        loop = SimpleLoop()

        manager.register(loop)

        assert "simple" in manager.loop_names
        assert manager.get("simple") is loop

    def test_register_replaces(self):
        """Should replace existing loop with same name."""
        manager = LoopManager()
        loop1 = SimpleLoop()
        loop2 = SimpleLoop()

        manager.register(loop1)
        manager.register(loop2)

        assert manager.get("simple") is loop2

    def test_unregister(self):
        """Should unregister loops."""
        manager = LoopManager()
        loop = SimpleLoop()
        manager.register(loop)

        removed = manager.unregister("simple")

        assert removed is loop
        assert "simple" not in manager.loop_names

    def test_unregister_nonexistent(self):
        """Should return None for nonexistent loop."""
        manager = LoopManager()

        removed = manager.unregister("nonexistent")

        assert removed is None

    def test_get_nonexistent(self):
        """Should return None for nonexistent loop."""
        manager = LoopManager()

        loop = manager.get("nonexistent")

        assert loop is None

    @pytest.mark.asyncio
    async def test_start_all(self):
        """Should start all registered loops."""
        manager = LoopManager()

        loop1 = SimpleLoop()
        loop1.name = "loop1"

        loop2 = SimpleLoop()
        loop2.name = "loop2"

        manager.register(loop1)
        manager.register(loop2)

        await manager.start_all()
        await asyncio.sleep(0.1)  # Let loops start (increased from 0.01 for CI stability)

        assert manager.is_started is True
        assert loop1.running is True
        assert loop2.running is True

        # Clean up
        await manager.stop_all()

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Should stop all registered loops."""
        manager = LoopManager()

        loop1 = SimpleLoop()
        loop1.name = "loop1"

        loop2 = SimpleLoop()
        loop2.name = "loop2"

        manager.register(loop1)
        manager.register(loop2)

        await manager.start_all()
        await asyncio.sleep(0.03)

        results = await manager.stop_all()

        assert manager.is_started is False
        assert loop1.running is False
        assert loop2.running is False
        assert results.get("loop1") is True
        assert results.get("loop2") is True

    @pytest.mark.asyncio
    async def test_stop_all_not_started(self):
        """Should return empty dict if not started."""
        manager = LoopManager()
        loop = SimpleLoop()
        manager.register(loop)

        results = await manager.stop_all()

        assert results == {}

    def test_get_all_status(self):
        """Should return status for all loops."""
        manager = LoopManager()

        loop1 = SimpleLoop()
        loop1.name = "loop1"

        loop2 = SimpleLoop()
        loop2.name = "loop2"

        manager.register(loop1)
        manager.register(loop2)

        status = manager.get_all_status()

        assert "loop1" in status
        assert "loop2" in status
        assert status["loop1"]["name"] == "loop1"
        assert status["loop2"]["name"] == "loop2"

    @pytest.mark.asyncio
    async def test_start_all_already_started(self):
        """Should warn if already started."""
        manager = LoopManager()
        loop = SimpleLoop()
        manager.register(loop)

        await manager.start_all()
        await manager.start_all()  # Should just log warning

        assert manager.is_started is True

        await manager.stop_all()

    def test_loop_names_property(self):
        """Should return list of loop names."""
        manager = LoopManager()

        loop1 = SimpleLoop()
        loop1.name = "alpha"

        loop2 = SimpleLoop()
        loop2.name = "beta"

        manager.register(loop1)
        manager.register(loop2)

        names = manager.loop_names

        assert "alpha" in names
        assert "beta" in names


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoopIntegration:
    """Integration tests for loop framework."""

    @pytest.mark.asyncio
    async def test_backoff_on_consecutive_errors(self):
        """Should apply backoff on consecutive errors."""
        loop = SimpleLoop(
            backoff_config=BackoffConfig(
                initial_delay=0.01,
                max_delay=0.1,
                multiplier=2.0,
                jitter=0.0,
            )
        )
        loop.raise_error = True

        loop.start_background()

        # Let it run for a bit - backoff should slow down execution
        await asyncio.sleep(0.15)

        loop.stop()
        await asyncio.sleep(0.02)

        # With backoff, should have run fewer times than without
        # First run: immediate
        # After error: 0.01s delay
        # After 2nd error: 0.02s delay
        # After 3rd error: 0.04s delay
        # etc.
        assert loop.stats.failed_runs >= 2

    @pytest.mark.asyncio
    async def test_metrics_recording(self):
        """Should record metrics when manager provided."""
        mock_metrics = MagicMock()

        loop = SimpleLoop(metrics_manager=mock_metrics)

        loop.start_background()
        await asyncio.sleep(0.05)
        loop.stop()
        await asyncio.sleep(0.02)

        # Should have called record_metric
        mock_metrics.record_metric.assert_called()

    @pytest.mark.asyncio
    async def test_error_metrics_recording(self):
        """Should record error metrics."""
        mock_metrics = MagicMock()

        loop = SimpleLoop(metrics_manager=mock_metrics)
        loop.raise_error = True

        loop.start_background()
        await asyncio.sleep(0.05)
        loop.stop()
        await asyncio.sleep(0.02)

        # Should have recorded error metrics
        calls = [
            call
            for call in mock_metrics.record_metric.call_args_list
            if "errors" in str(call)
        ]
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_loop_manager_full_lifecycle(self):
        """Test complete lifecycle of manager with multiple loops."""
        manager = LoopManager(name="lifecycle_test")

        # Create loops with different intervals
        loop1 = SimpleLoop()
        loop1.name = "fast_loop"
        loop1.interval = 0.01

        loop2 = SimpleLoop()
        loop2.name = "slow_loop"
        loop2.interval = 0.02

        manager.register(loop1)
        manager.register(loop2)

        # Start
        await manager.start_all()
        assert manager.is_started

        await asyncio.sleep(0.05)

        # Check both running
        status = manager.get_all_status()
        assert status["fast_loop"]["running"] is True
        assert status["slow_loop"]["running"] is True

        # Stop
        results = await manager.stop_all()

        assert not manager.is_started
        assert results["fast_loop"] is True
        assert results["slow_loop"] is True

        # Verify both ran
        assert loop1.run_count >= 1
        assert loop2.run_count >= 1
