"""Unit tests for lifecycle_mixin.py.

Tests the lifecycle management infrastructure used by 30+ daemons.
Covers:
- LifecycleState enum
- LifecycleMixin class lifecycle management
- EventSubscriptionMixin event handling
- ManagedComponent combined functionality
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
# LifecycleState Tests
# =============================================================================


class TestLifecycleState:
    """Tests for LifecycleState enum."""

    def test_created_value(self):
        """Test CREATED state value."""
        assert LifecycleState.CREATED.value == "created"

    def test_starting_value(self):
        """Test STARTING state value."""
        assert LifecycleState.STARTING.value == "starting"

    def test_running_value(self):
        """Test RUNNING state value."""
        assert LifecycleState.RUNNING.value == "running"

    def test_paused_value(self):
        """Test PAUSED state value."""
        assert LifecycleState.PAUSED.value == "paused"

    def test_stopping_value(self):
        """Test STOPPING state value."""
        assert LifecycleState.STOPPING.value == "stopping"

    def test_stopped_value(self):
        """Test STOPPED state value."""
        assert LifecycleState.STOPPED.value == "stopped"

    def test_failed_value(self):
        """Test FAILED state value."""
        assert LifecycleState.FAILED.value == "failed"

    def test_all_states_count(self):
        """Test all states are enumerable."""
        states = list(LifecycleState)
        assert len(states) == 7


# =============================================================================
# LifecycleMixin Tests
# =============================================================================


class ConcreteLifecycleMixin(LifecycleMixin):
    """Concrete implementation for testing."""

    def __init__(
        self,
        name: str = "test",
        cycle_interval: float = 0.1,
        shutdown_timeout: float = 1.0,
    ):
        super().__init__(
            name=name,
            cycle_interval=cycle_interval,
            shutdown_timeout=shutdown_timeout,
        )
        self.start_called = False
        self.stop_called = False
        self.cycle_count_test = 0
        self.error_handler_called = False

    async def _on_start(self) -> None:
        self.start_called = True

    async def _on_stop(self) -> None:
        self.stop_called = True

    async def _on_cycle(self) -> None:
        self.cycle_count_test += 1

    async def _on_error(self, error: Exception) -> None:
        self.error_handler_called = True


class TestLifecycleMixin:
    """Tests for LifecycleMixin class."""

    @pytest.fixture
    def component(self):
        """Create a test component."""
        return ConcreteLifecycleMixin()

    def test_initial_state(self, component):
        """Test initial state is CREATED."""
        assert component.lifecycle_state == LifecycleState.CREATED

    def test_is_running_false_initially(self, component):
        """Test is_running is False initially."""
        assert component.is_running is False

    def test_is_stopped_false_initially(self, component):
        """Test is_stopped is False initially."""
        assert component.is_stopped is False

    def test_uptime_none_initially(self, component):
        """Test uptime is None before start."""
        assert component.uptime is None

    def test_cycle_interval_property(self, component):
        """Test cycle_interval property."""
        assert component.cycle_interval == 0.1

    def test_cycle_interval_setter(self, component):
        """Test cycle_interval setter."""
        component.cycle_interval = 30.0
        assert component.cycle_interval == 30.0

    def test_cycle_interval_minimum(self, component):
        """Test cycle_interval has minimum of 0.1."""
        component.cycle_interval = 0.01
        assert component.cycle_interval == 0.1

    @pytest.mark.asyncio
    async def test_start_success(self, component):
        """Test successful start."""
        result = await component.start()

        assert result is True
        assert component.lifecycle_state == LifecycleState.RUNNING
        assert component.is_running is True
        assert component.start_called is True

        await component.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self, component):
        """Test start when already running returns True."""
        await component.start()

        result = await component.start()

        assert result is True
        await component.stop()

    @pytest.mark.asyncio
    async def test_start_from_stopped(self, component):
        """Test starting from STOPPED state."""
        await component.start()
        await component.stop()

        # Reset flags
        component.start_called = False
        result = await component.start()

        assert result is True
        assert component.start_called is True

        await component.stop()

    @pytest.mark.asyncio
    async def test_start_fails_on_hook_error(self):
        """Test start fails when _on_start raises."""

        class FailingComponent(LifecycleMixin):
            async def _on_start(self) -> None:
                raise RuntimeError("Start failed")

        component = FailingComponent(name="failing")
        result = await component.start()

        assert result is False
        assert component.lifecycle_state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_stop_success(self, component):
        """Test successful stop."""
        await component.start()
        result = await component.stop()

        assert result is True
        assert component.lifecycle_state == LifecycleState.STOPPED
        assert component.is_stopped is True
        assert component.stop_called is True

    @pytest.mark.asyncio
    async def test_stop_not_running(self, component):
        """Test stop when not running returns True."""
        result = await component.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self, component):
        """Test stop when already stopped returns True."""
        await component.start()
        await component.stop()
        result = await component.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_restart(self, component):
        """Test restart."""
        await component.start()
        component.start_called = False

        result = await component.restart()

        assert result is True
        assert component.stop_called is True
        assert component.start_called is True
        assert component.is_running is True

        await component.stop()

    @pytest.mark.asyncio
    async def test_pause(self, component):
        """Test pause."""
        await component.start()
        result = await component.pause()

        assert result is True
        assert component.lifecycle_state == LifecycleState.PAUSED

        await component.stop()

    @pytest.mark.asyncio
    async def test_pause_not_running(self, component):
        """Test pause when not running returns False."""
        result = await component.pause()
        assert result is False

    @pytest.mark.asyncio
    async def test_resume(self, component):
        """Test resume from paused."""
        await component.start()
        await component.pause()

        result = await component.resume()

        assert result is True
        assert component.lifecycle_state == LifecycleState.RUNNING

        await component.stop()

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, component):
        """Test resume when not paused returns False."""
        await component.start()
        result = await component.resume()
        assert result is False
        await component.stop()

    @pytest.mark.asyncio
    async def test_uptime_tracking(self, component):
        """Test uptime is tracked while running."""
        await component.start()
        await asyncio.sleep(0.1)

        uptime = component.uptime
        assert uptime is not None
        assert uptime >= 0.1

        await component.stop()

    @pytest.mark.asyncio
    async def test_lifecycle_health(self, component):
        """Test get_lifecycle_health."""
        await component.start()

        health = component.get_lifecycle_health()

        assert health["name"] == "test"
        assert health["state"] == "running"
        assert health["running"] is True
        assert health["cycle_count"] >= 0
        assert health["error_count"] == 0
        assert health["last_error"] is None

        await component.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, component):
        """Test async context manager."""
        async with component:
            assert component.is_running is True

        assert component.is_stopped is True


# =============================================================================
# LifecycleMixin Loop Tests
# =============================================================================


class TestLifecycleMixinLoop:
    """Tests for LifecycleMixin loop functionality."""

    @pytest.mark.asyncio
    async def test_default_loop_calls_on_cycle(self):
        """Test default loop calls _on_cycle."""
        component = ConcreteLifecycleMixin(cycle_interval=0.05)
        await component.start()

        # Wait for a few cycles
        await asyncio.sleep(0.15)

        await component.stop()

        # Should have run several cycles
        assert component.cycle_count_test >= 2

    @pytest.mark.asyncio
    async def test_custom_run_loop(self):
        """Test custom _run_loop is used instead of default."""
        run_count = 0

        class CustomLoopComponent(LifecycleMixin):
            async def _run_loop(self) -> None:
                nonlocal run_count
                while not self._should_stop():
                    run_count += 1
                    await asyncio.sleep(0.05)

        component = CustomLoopComponent(name="custom")
        await component.start()
        await asyncio.sleep(0.15)
        await component.stop()

        assert run_count >= 2

    @pytest.mark.asyncio
    async def test_loop_handles_error(self):
        """Test loop handles errors in _on_cycle."""

        class ErrorComponent(LifecycleMixin):
            def __init__(self):
                super().__init__(name="error", cycle_interval=0.05)
                self.error_count_local = 0

            async def _on_cycle(self) -> None:
                raise ValueError("Cycle error")

            async def _on_error(self, error: Exception) -> None:
                self.error_count_local += 1

        component = ErrorComponent()
        await component.start()
        await asyncio.sleep(0.15)
        await component.stop()

        # Errors should have been handled
        assert component.error_count_local >= 1

    @pytest.mark.asyncio
    async def test_should_stop_after_shutdown_event(self):
        """Test _should_stop returns True after shutdown event."""
        component = ConcreteLifecycleMixin()
        await component.start()

        assert component._should_stop() is False

        component._shutdown_event.set()
        assert component._should_stop() is True

        await component.stop()


# =============================================================================
# EventSubscriptionMixin Tests
# =============================================================================


class ConcreteEventMixin(EventSubscriptionMixin):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__()
        self.handler_called = False

    def _get_event_subscriptions(self) -> dict:
        return {"TEST_EVENT": self._handle_test_event}

    async def _handle_test_event(self, event):
        self.handler_called = True


class TestEventSubscriptionMixin:
    """Tests for EventSubscriptionMixin class."""

    def test_initial_state(self):
        """Test initial state."""
        component = ConcreteEventMixin()
        assert component._subscription_ids == []
        assert component._event_bus is None

    def test_get_event_subscriptions(self):
        """Test _get_event_subscriptions returns handler dict."""
        component = ConcreteEventMixin()
        subs = component._get_event_subscriptions()
        assert "TEST_EVENT" in subs
        assert callable(subs["TEST_EVENT"])

    @pytest.mark.asyncio
    async def test_subscribe_to_events_no_bus(self):
        """Test _subscribe_to_events handles missing event bus."""
        component = ConcreteEventMixin()

        with patch(
            "app.coordination.mixins.lifecycle_mixin.get_event_bus", return_value=None
        ):
            await component._subscribe_to_events()

        # Should complete without error
        assert component._event_bus is None

    @pytest.mark.asyncio
    async def test_unsubscribe_from_events(self):
        """Test _unsubscribe_from_events clears subscriptions."""
        component = ConcreteEventMixin()
        component._subscription_ids = ["sub1", "sub2"]

        mock_bus = MagicMock()
        component._event_bus = mock_bus

        await component._unsubscribe_from_events()

        assert component._subscription_ids == []
        assert mock_bus.unsubscribe.call_count == 2


# =============================================================================
# ManagedComponent Tests
# =============================================================================


class ConcreteManagedComponent(ManagedComponent):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__(name="managed_test", cycle_interval=0.1)
        self.cycle_count_local = 0

    async def _on_cycle(self) -> None:
        self.cycle_count_local += 1


class TestManagedComponent:
    """Tests for ManagedComponent class."""

    @pytest.fixture
    def component(self):
        """Create a test component."""
        return ConcreteManagedComponent()

    def test_inherits_lifecycle_mixin(self, component):
        """Test ManagedComponent inherits LifecycleMixin."""
        assert isinstance(component, LifecycleMixin)

    def test_inherits_event_subscription_mixin(self, component):
        """Test ManagedComponent inherits EventSubscriptionMixin."""
        assert isinstance(component, EventSubscriptionMixin)

    def test_initial_state(self, component):
        """Test initial state is CREATED."""
        assert component.lifecycle_state == LifecycleState.CREATED

    @pytest.mark.asyncio
    async def test_start_subscribes_to_events(self, component):
        """Test start calls _subscribe_to_events."""
        with patch.object(
            component, "_subscribe_to_events", new_callable=AsyncMock
        ) as mock_subscribe:
            await component.start()
            mock_subscribe.assert_called_once()
            await component.stop()

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_from_events(self, component):
        """Test stop calls _unsubscribe_from_events."""
        await component.start()

        with patch.object(
            component, "_unsubscribe_from_events", new_callable=AsyncMock
        ) as mock_unsubscribe:
            await component.stop()
            mock_unsubscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, component):
        """Test full lifecycle with cycles."""
        await component.start()
        await asyncio.sleep(0.25)
        await component.stop()

        assert component.cycle_count_local >= 2
        assert component.is_stopped is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestLifecycleMixinEdgeCases:
    """Edge case tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_invalid_state(self):
        """Test start from invalid state returns False."""

        class InvalidStateComponent(LifecycleMixin):
            pass

        component = InvalidStateComponent(name="test")
        # Manually set to an invalid state
        component._lifecycle_state = LifecycleState.STARTING

        result = await component.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_timeout_on_slow_loop(self):
        """Test stop handles slow loop with timeout."""

        class SlowComponent(LifecycleMixin):
            async def _run_loop(self) -> None:
                while not self._should_stop():
                    await asyncio.sleep(0.1)
                # Simulate slow cleanup
                await asyncio.sleep(10)

        component = SlowComponent(name="slow", shutdown_timeout=0.2)
        await component.start()
        await asyncio.sleep(0.1)

        # Should timeout and cancel
        result = await component.stop()
        assert result is True  # Still succeeds (cancellation)

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test error and cycle count tracking."""

        class MetricsComponent(LifecycleMixin):
            async def _on_cycle(self) -> None:
                pass

        component = MetricsComponent(name="metrics", cycle_interval=0.05)
        await component.start()
        await asyncio.sleep(0.2)
        await component.stop()

        assert component._cycle_count >= 2
        assert component._error_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self):
        """Test context manager handles exceptions."""
        component = ConcreteLifecycleMixin()

        with pytest.raises(ValueError):
            async with component:
                raise ValueError("Test error")

        # Should still be stopped
        assert component.is_stopped is True

    @pytest.mark.asyncio
    async def test_stop_fails_on_hook_error(self):
        """Test stop fails when _on_stop raises."""

        class FailingStopComponent(LifecycleMixin):
            async def _on_stop(self) -> None:
                raise RuntimeError("Stop failed")

        component = FailingStopComponent(name="failing")
        await component.start()
        result = await component.stop()

        assert result is False
        assert component.lifecycle_state == LifecycleState.FAILED
