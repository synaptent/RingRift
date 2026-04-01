"""Tests for lifecycle_mixin.py.

December 30, 2025: Created comprehensive tests for the lifecycle management
mixin used by 30+ daemons.

Tests cover:
- LifecycleState enum values
- LifecycleMixin lifecycle transitions
- start/stop/restart/pause/resume methods
- Async context manager support
- Error handling during lifecycle
- Uptime and metrics tracking
- Health check integration
- EventSubscriptionMixin
- ManagedComponent combined mixin
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.mixins.lifecycle_mixin import (
    EventSubscriptionMixin,
    LifecycleMixin,
    LifecycleState,
    ManagedComponent,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockComponent(LifecycleMixin):
    """Mock component for testing LifecycleMixin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_start_called = False
        self.on_stop_called = False
        self.on_cycle_called = 0
        self.on_error_called = False
        self.simulate_start_error = False
        self.simulate_stop_error = False
        self.simulate_cycle_error = False

    async def _on_start(self) -> None:
        if self.simulate_start_error:
            raise ValueError("Simulated start error")
        self.on_start_called = True

    async def _on_stop(self) -> None:
        if self.simulate_stop_error:
            raise ValueError("Simulated stop error")
        self.on_stop_called = True

    async def _on_cycle(self) -> None:
        if self.simulate_cycle_error:
            raise ValueError("Simulated cycle error")
        self.on_cycle_called += 1

    async def _on_error(self, error: Exception) -> None:
        self.on_error_called = True


class MockComponentWithLoop(MockComponent):
    """Mock component with custom _run_loop."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop_iterations = 0

    async def _run_loop(self) -> None:
        while not self._should_stop():
            self.loop_iterations += 1
            await asyncio.sleep(0.01)


# =============================================================================
# LifecycleState Tests
# =============================================================================


class TestLifecycleState:
    """Tests for LifecycleState enum."""

    def test_all_states_defined(self):
        """Verify all expected states exist."""
        states = [s.value for s in LifecycleState]
        assert "created" in states
        assert "starting" in states
        assert "running" in states
        assert "paused" in states
        assert "stopping" in states
        assert "stopped" in states
        assert "failed" in states

    def test_state_values_are_strings(self):
        """Verify all state values are strings."""
        for state in LifecycleState:
            assert isinstance(state.value, str)

    def test_state_count(self):
        """Verify exactly 7 states."""
        assert len(LifecycleState) == 7


# =============================================================================
# LifecycleMixin Initialization Tests
# =============================================================================


class TestLifecycleMixinInit:
    """Tests for LifecycleMixin initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        component = MockComponent()

        assert component._lifecycle_name == "component"
        assert component._lifecycle_state == LifecycleState.CREATED
        assert component._cycle_interval == 60.0
        assert component._shutdown_timeout == 30.0
        assert component._start_time is None
        assert component._stop_time is None
        assert component._loop_task is None
        assert component._shutdown_event is None
        assert component._last_error is None
        assert component._cycle_count == 0
        assert component._error_count == 0

    def test_custom_initialization(self):
        """Test custom initialization values."""
        component = MockComponent(
            name="test_daemon",
            cycle_interval=10.0,
            shutdown_timeout=60.0,
        )

        assert component._lifecycle_name == "test_daemon"
        assert component._cycle_interval == 10.0
        assert component._shutdown_timeout == 60.0


# =============================================================================
# LifecycleMixin Properties Tests
# =============================================================================


class TestLifecycleMixinProperties:
    """Tests for LifecycleMixin properties."""

    def test_lifecycle_state_property(self):
        """Test lifecycle_state property."""
        component = MockComponent()
        assert component.lifecycle_state == LifecycleState.CREATED

    def test_is_running_property_when_not_running(self):
        """Test is_running when not running."""
        component = MockComponent()
        assert component.is_running is False

    @pytest.mark.asyncio
    async def test_is_running_property_when_running(self):
        """Test is_running when running."""
        component = MockComponent()
        await component.start()
        try:
            assert component.is_running is True
        finally:
            await component.stop()

    def test_is_stopped_property_created(self):
        """Test is_stopped when created."""
        component = MockComponent()
        assert component.is_stopped is False

    @pytest.mark.asyncio
    async def test_is_stopped_property_stopped(self):
        """Test is_stopped when stopped."""
        component = MockComponent()
        await component.start()
        await component.stop()
        assert component.is_stopped is True

    def test_uptime_before_start(self):
        """Test uptime before starting."""
        component = MockComponent()
        assert component.uptime is None

    @pytest.mark.asyncio
    async def test_uptime_while_running(self):
        """Test uptime while running."""
        component = MockComponent()
        await component.start()
        try:
            await asyncio.sleep(0.1)
            assert component.uptime is not None
            assert component.uptime >= 0.1
        finally:
            await component.stop()

    def test_cycle_interval_getter(self):
        """Test cycle_interval getter."""
        component = MockComponent(cycle_interval=30.0)
        assert component.cycle_interval == 30.0

    def test_cycle_interval_setter(self):
        """Test cycle_interval setter."""
        component = MockComponent()
        component.cycle_interval = 45.0
        assert component.cycle_interval == 45.0

    def test_cycle_interval_minimum(self):
        """Test cycle_interval enforces minimum."""
        component = MockComponent()
        component.cycle_interval = 0.01
        assert component.cycle_interval == 0.1  # Minimum is 100ms


# =============================================================================
# LifecycleMixin Start/Stop Tests
# =============================================================================


class TestLifecycleMixinStartStop:
    """Tests for start/stop lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_from_created(self):
        """Test starting from CREATED state."""
        component = MockComponent()
        result = await component.start()

        assert result is True
        assert component.lifecycle_state == LifecycleState.RUNNING
        assert component.on_start_called is True
        assert component._start_time is not None
        await component.stop()

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Test starting when already running."""
        component = MockComponent()
        await component.start()
        try:
            result = await component.start()
            assert result is True  # Should return True, already running
        finally:
            await component.stop()

    @pytest.mark.asyncio
    async def test_start_from_stopped(self):
        """Test restarting from STOPPED state."""
        component = MockComponent()
        await component.start()
        await component.stop()

        result = await component.start()
        assert result is True
        assert component.lifecycle_state == LifecycleState.RUNNING
        await component.stop()

    @pytest.mark.asyncio
    async def test_start_with_error(self):
        """Test start failure handling."""
        component = MockComponent()
        component.simulate_start_error = True

        result = await component.start()

        assert result is False
        assert component.lifecycle_state == LifecycleState.FAILED
        assert component._error_count == 1
        assert component._last_error is not None

    @pytest.mark.asyncio
    async def test_stop_from_running(self):
        """Test stopping from running state."""
        component = MockComponent()
        await component.start()

        result = await component.stop()

        assert result is True
        assert component.lifecycle_state == LifecycleState.STOPPED
        assert component.on_stop_called is True
        assert component._stop_time is not None

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped(self):
        """Test stopping when already stopped."""
        component = MockComponent()
        await component.start()
        await component.stop()

        result = await component.stop()
        assert result is True  # Should return True

    @pytest.mark.asyncio
    async def test_stop_from_created(self):
        """Test stopping from CREATED state (never started)."""
        component = MockComponent()
        result = await component.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_with_error(self):
        """Test stop failure handling."""
        component = MockComponent()
        await component.start()
        component.simulate_stop_error = True

        result = await component.stop()

        assert result is False
        assert component.lifecycle_state == LifecycleState.FAILED


# =============================================================================
# LifecycleMixin Restart/Pause/Resume Tests
# =============================================================================


class TestLifecycleMixinRestartPauseResume:
    """Tests for restart, pause, and resume methods."""

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test restart method."""
        component = MockComponent()
        await component.start()

        result = await component.restart()

        assert result is True
        assert component.lifecycle_state == LifecycleState.RUNNING
        assert component.on_start_called is True
        assert component.on_stop_called is True
        await component.stop()

    @pytest.mark.asyncio
    async def test_pause_from_running(self):
        """Test pause from running state."""
        component = MockComponent()
        await component.start()
        try:
            result = await component.pause()
            assert result is True
            assert component.lifecycle_state == LifecycleState.PAUSED
        finally:
            await component.stop()

    @pytest.mark.asyncio
    async def test_pause_from_non_running(self):
        """Test pause from non-running state."""
        component = MockComponent()
        result = await component.pause()
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_from_paused(self):
        """Test resume from paused state."""
        component = MockComponent()
        await component.start()
        try:
            await component.pause()
            result = await component.resume()
            assert result is True
            assert component.lifecycle_state == LifecycleState.RUNNING
        finally:
            await component.stop()

    @pytest.mark.asyncio
    async def test_resume_from_non_paused(self):
        """Test resume from non-paused state."""
        component = MockComponent()
        await component.start()
        try:
            result = await component.resume()
            assert result is False  # Not paused
        finally:
            await component.stop()


# =============================================================================
# LifecycleMixin Loop Tests
# =============================================================================


class TestLifecycleMixinLoop:
    """Tests for lifecycle loop management."""

    @pytest.mark.asyncio
    async def test_custom_loop_executes(self):
        """Test that custom _run_loop is executed."""
        component = MockComponentWithLoop(cycle_interval=0.01)
        await component.start()

        await asyncio.sleep(0.05)
        await component.stop()

        assert component.loop_iterations > 0

    @pytest.mark.asyncio
    async def test_default_loop_calls_on_cycle(self):
        """Test that default loop calls _on_cycle."""
        component = MockComponent(cycle_interval=0.01)
        # Start the loop manually for default behavior
        component._lifecycle_state = LifecycleState.RUNNING
        component._shutdown_event = asyncio.Event()

        # Run a few cycles
        task = asyncio.create_task(component._default_loop())
        await asyncio.sleep(0.05)
        component._shutdown_event.set()
        await task

        assert component.on_cycle_called > 0

    @pytest.mark.asyncio
    async def test_loop_stops_on_shutdown(self):
        """Test that loop stops when shutdown is signaled."""
        component = MockComponentWithLoop(cycle_interval=0.01)
        await component.start()

        initial_iterations = component.loop_iterations
        await component.stop()

        # Loop should have stopped
        final_iterations = component.loop_iterations
        await asyncio.sleep(0.05)
        assert component.loop_iterations == final_iterations


# =============================================================================
# LifecycleMixin Context Manager Tests
# =============================================================================


class TestLifecycleMixinContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager_start_stop(self):
        """Test async context manager starts and stops."""
        component = MockComponent()

        async with component:
            assert component.is_running is True
            assert component.on_start_called is True

        assert component.is_stopped is True
        assert component.on_stop_called is True

    @pytest.mark.asyncio
    async def test_async_context_manager_exception_handling(self):
        """Test async context manager handles exceptions."""
        component = MockComponent()

        with pytest.raises(ValueError):
            async with component:
                raise ValueError("test error")

        # Component should still be stopped
        assert component.is_stopped is True

    @pytest.mark.asyncio
    async def test_async_context_manager_returns_self(self):
        """Test async context manager returns self."""
        component = MockComponent()

        async with component as ctx:
            assert ctx is component


# =============================================================================
# LifecycleMixin Health Check Tests
# =============================================================================


class TestLifecycleMixinHealthCheck:
    """Tests for health check integration."""

    def test_get_lifecycle_health_created(self):
        """Test health check in CREATED state."""
        component = MockComponent(name="test_daemon")
        health = component.get_lifecycle_health()

        assert health["name"] == "test_daemon"
        assert health["state"] == "created"
        assert health["running"] is False
        assert health["uptime_seconds"] is None
        assert health["cycle_count"] == 0
        assert health["error_count"] == 0
        assert health["last_error"] is None

    @pytest.mark.asyncio
    async def test_get_lifecycle_health_running(self):
        """Test health check in RUNNING state."""
        component = MockComponent(name="test_daemon")
        await component.start()
        try:
            await asyncio.sleep(0.1)
            health = component.get_lifecycle_health()

            assert health["state"] == "running"
            assert health["running"] is True
            assert health["uptime_seconds"] >= 0.1
        finally:
            await component.stop()

    @pytest.mark.asyncio
    async def test_get_lifecycle_health_with_errors(self):
        """Test health check with errors."""
        component = MockComponent()
        component.simulate_start_error = True
        await component.start()

        health = component.get_lifecycle_health()

        assert health["state"] == "failed"
        assert health["error_count"] == 1
        assert health["last_error"] is not None


# =============================================================================
# EventSubscriptionMixin Tests
# =============================================================================


class TestEventSubscriptionMixin:
    """Tests for EventSubscriptionMixin."""

    def test_initialization(self):
        """Test EventSubscriptionMixin initialization."""
        mixin = EventSubscriptionMixin()
        assert mixin._subscription_ids == []

    def test_get_event_subscriptions_default(self):
        """Test default _get_event_subscriptions returns empty dict."""
        mixin = EventSubscriptionMixin()
        assert mixin._get_event_subscriptions() == {}

    @pytest.mark.asyncio
    async def test_subscribe_to_events_no_bus(self):
        """Test _subscribe_to_events without event system."""
        mixin = EventSubscriptionMixin()

        # Patch at the source where it's imported
        with patch(
            "app.coordination.event_router.subscribe",
            side_effect=ImportError("no router"),
        ):
            await mixin._subscribe_to_events()

        assert mixin._subscription_ids == []

    @pytest.mark.asyncio
    async def test_subscribe_to_events_with_bus(self):
        """Test _subscribe_to_events with mock subscribe."""
        mixin = EventSubscriptionMixin()

        mock_subscribe = MagicMock()
        handler = lambda x: None
        mixin._get_event_subscriptions = lambda: {"TEST_EVENT": handler}

        # Patch at the source where it's imported
        with patch(
            "app.coordination.event_router.subscribe",
            mock_subscribe,
        ):
            await mixin._subscribe_to_events()

        assert ("TEST_EVENT", handler) in mixin._subscription_ids
        mock_subscribe.assert_called_once_with("TEST_EVENT", handler)

    @pytest.mark.asyncio
    async def test_unsubscribe_from_events(self):
        """Test _unsubscribe_from_events."""
        mixin = EventSubscriptionMixin()
        handler_1 = lambda x: None
        handler_2 = lambda x: None
        mixin._subscription_ids = [("EVENT_A", handler_1), ("EVENT_B", handler_2)]

        mock_unsubscribe = MagicMock()
        with patch("app.coordination.event_router.unsubscribe", mock_unsubscribe):
            await mixin._unsubscribe_from_events()

        assert mock_unsubscribe.call_count == 2
        assert mixin._subscription_ids == []

    @pytest.mark.asyncio
    async def test_unsubscribe_handles_missing_subscriptions(self):
        """Test _unsubscribe_from_events handles missing subscriptions."""
        mixin = EventSubscriptionMixin()
        handler = lambda x: None
        mixin._subscription_ids = [("EVENT_A", handler)]

        mock_unsubscribe = MagicMock(side_effect=KeyError("not found"))
        with patch("app.coordination.event_router.unsubscribe", mock_unsubscribe):
            # Should not raise
            await mixin._unsubscribe_from_events()


# =============================================================================
# ManagedComponent Tests
# =============================================================================


class TestManagedComponent:
    """Tests for ManagedComponent combined mixin."""

    def test_initialization(self):
        """Test ManagedComponent initialization."""
        component = ManagedComponent(
            name="test_managed",
            cycle_interval=30.0,
            shutdown_timeout=15.0,
        )

        assert component._lifecycle_name == "test_managed"
        assert component._cycle_interval == 30.0
        assert component._shutdown_timeout == 15.0
        assert component._subscription_ids == []

    @pytest.mark.asyncio
    async def test_start_subscribes_to_events(self):
        """Test that start subscribes to events."""
        component = ManagedComponent()
        component._subscribe_to_events = AsyncMock()

        await component.start()
        try:
            component._subscribe_to_events.assert_called_once()
        finally:
            await component.stop()

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_from_events(self):
        """Test that stop unsubscribes from events."""
        component = ManagedComponent()
        component._subscribe_to_events = AsyncMock()
        component._unsubscribe_from_events = AsyncMock()

        await component.start()
        await component.stop()

        component._unsubscribe_from_events.assert_called_once()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestLifecycleMixinEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_start_from_invalid_state(self):
        """Test starting from STARTING state (invalid)."""
        component = MockComponent()
        component._lifecycle_state = LifecycleState.STARTING

        result = await component.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_start_calls(self):
        """Test concurrent start calls."""
        component = MockComponent()

        # Start twice concurrently
        results = await asyncio.gather(
            component.start(),
            component.start(),
        )

        # At least one should succeed
        assert any(results)
        await component.stop()

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self):
        """Test shutdown with slow loop."""

        class SlowComponent(LifecycleMixin):
            async def _run_loop(self) -> None:
                while not self._should_stop():
                    await asyncio.sleep(0.1)
                # Simulate slow cleanup
                await asyncio.sleep(2.0)

        component = SlowComponent(shutdown_timeout=0.5)
        await component.start()

        # Should complete within timeout (loop is cancelled)
        await asyncio.wait_for(component.stop(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_error_in_cycle(self):
        """Test error handling in cycle."""
        component = MockComponent(cycle_interval=0.01)
        component.simulate_cycle_error = True
        component._lifecycle_state = LifecycleState.RUNNING
        component._shutdown_event = asyncio.Event()

        # Run a few cycles
        task = asyncio.create_task(component._default_loop())
        await asyncio.sleep(0.05)
        component._shutdown_event.set()
        await task

        assert component._error_count > 0
        assert component.on_error_called is True

    @pytest.mark.asyncio
    async def test_multiple_restart_cycles(self):
        """Test multiple restart cycles."""
        component = MockComponent()

        for _ in range(3):
            await component.start()
            assert component.is_running
            await component.stop()
            assert component.is_stopped

    @pytest.mark.asyncio
    async def test_uptime_after_stop(self):
        """Test uptime is frozen after stop."""
        component = MockComponent()
        await component.start()
        await asyncio.sleep(0.1)
        await component.stop()

        uptime_at_stop = component.uptime
        await asyncio.sleep(0.1)

        # Uptime should not change after stop
        assert component.uptime == uptime_at_stop
