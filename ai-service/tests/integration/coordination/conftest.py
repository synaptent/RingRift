"""Pytest fixtures for coordination integration tests.

December 29, 2025: Created for cross-daemon integration testing.

These fixtures provide:
- Mock event router with subscription tracking
- Fake daemons for isolated testing
- Event chain verification utilities
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class RecordedEvent:
    """A recorded event for verification."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "test"


class MockEventRouter:
    """Mock event router that records all published events.

    Usage:
        router = MockEventRouter()
        await router.publish("TRAINING_COMPLETED", {"model": "test.pth"})
        assert router.event_count("TRAINING_COMPLETED") == 1
    """

    def __init__(self):
        self.events: list[RecordedEvent] = []
        self.subscriptions: dict[str, list[Callable]] = {}
        self._lock = asyncio.Lock()

    async def publish(
        self, event_type: str, payload: dict[str, Any], source: str = "test"
    ) -> None:
        """Publish an event and notify subscribers."""
        async with self._lock:
            event = RecordedEvent(
                event_type=event_type,
                payload=payload,
                timestamp=time.time(),
                source=source,
            )
            self.events.append(event)

            # Notify subscribers
            handlers = self.subscriptions.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception:
                    pass  # Continue to other handlers

    def publish_sync(
        self, event_type: str, payload: dict[str, Any], source: str = "test"
    ) -> None:
        """Synchronous publish for non-async contexts."""
        event = RecordedEvent(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            source=source,
        )
        self.events.append(event)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.subscriptions:
            try:
                self.subscriptions[event_type].remove(handler)
            except ValueError:
                pass

    def event_count(self, event_type: str) -> int:
        """Count events of a specific type."""
        return sum(1 for e in self.events if e.event_type == event_type)

    def get_events(self, event_type: str) -> list[RecordedEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_latest(self, event_type: str) -> RecordedEvent | None:
        """Get the most recent event of a type."""
        events = self.get_events(event_type)
        return events[-1] if events else None

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()

    async def wait_for_event(
        self, event_type: str, timeout: float = 5.0, count: int = 1
    ) -> list[RecordedEvent]:
        """Wait for an event to be published.

        Args:
            event_type: Event type to wait for
            timeout: Maximum wait time in seconds
            count: Number of events to wait for

        Returns:
            List of matching events

        Raises:
            TimeoutError: If events not received within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            events = self.get_events(event_type)
            if len(events) >= count:
                return events[:count]
            await asyncio.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for {count} {event_type} events "
            f"(got {self.event_count(event_type)})"
        )

    def verify_event_order(self, expected_order: list[str]) -> bool:
        """Verify events were published in the expected order.

        Args:
            expected_order: List of event types in expected order

        Returns:
            True if events match expected order
        """
        actual_order = [e.event_type for e in self.events]

        # Check that expected events appear in order (not necessarily consecutive)
        expected_idx = 0
        for event_type in actual_order:
            if expected_idx < len(expected_order) and event_type == expected_order[expected_idx]:
                expected_idx += 1

        return expected_idx == len(expected_order)


@dataclass
class DaemonTestState:
    """Shared state for daemon tests."""

    started_daemons: list[str] = field(default_factory=list)
    stopped_daemons: list[str] = field(default_factory=list)
    health_checks: dict[str, dict] = field(default_factory=dict)


@pytest.fixture
def mock_event_router():
    """Provide a mock event router for testing."""
    return MockEventRouter()


@pytest.fixture
def daemon_test_state():
    """Provide shared daemon test state."""
    return DaemonTestState()


@pytest.fixture
def mock_daemon_manager(daemon_test_state):
    """Mock daemon manager for testing daemon lifecycle."""
    manager = MagicMock()

    async def mock_start(daemon_type):
        daemon_test_state.started_daemons.append(str(daemon_type))
        return True

    async def mock_stop(daemon_type):
        daemon_test_state.stopped_daemons.append(str(daemon_type))
        return True

    manager.start = AsyncMock(side_effect=mock_start)
    manager.stop = AsyncMock(side_effect=mock_stop)
    manager.get_daemon_health = MagicMock(return_value={"status": "healthy"})

    return manager


@pytest.fixture
def patch_event_router(mock_event_router):
    """Patch the global event router with mock."""
    with patch("app.coordination.event_router.get_router", return_value=mock_event_router):
        yield mock_event_router


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        "board_type": "hex8",
        "num_players": 2,
        "config_key": "hex8_2p",
        "min_games": 100,
        "quality_threshold": 0.40,
        "regression_threshold": 0.50,
    }


class EventChainVerifier:
    """Utility to verify event chains across daemons.

    Usage:
        verifier = EventChainVerifier(router)
        verifier.expect_chain([
            "DATA_SYNC_COMPLETED",
            "NEW_GAMES_AVAILABLE",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
        ])
        # ... trigger events ...
        await verifier.verify(timeout=30.0)
    """

    def __init__(self, router: MockEventRouter):
        self.router = router
        self.expected_chain: list[str] = []
        self.timing_requirements: dict[str, float] = {}

    def expect_chain(self, events: list[str]) -> "EventChainVerifier":
        """Set the expected event chain."""
        self.expected_chain = events
        return self

    def with_timing(self, event_type: str, max_delay: float) -> "EventChainVerifier":
        """Add timing requirement for an event.

        Args:
            event_type: Event that must occur
            max_delay: Maximum delay from previous event in seconds
        """
        self.timing_requirements[event_type] = max_delay
        return self

    async def verify(self, timeout: float = 30.0) -> bool:
        """Verify the event chain occurred.

        Args:
            timeout: Maximum total time to wait

        Returns:
            True if chain verified

        Raises:
            AssertionError: If chain verification fails
        """
        start = time.time()

        # Wait for all events
        for event_type in self.expected_chain:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                raise AssertionError(f"Timeout waiting for {event_type}")

            try:
                await self.router.wait_for_event(event_type, timeout=remaining)
            except TimeoutError:
                raise AssertionError(
                    f"Event {event_type} not received. "
                    f"Got: {[e.event_type for e in self.router.events]}"
                )

        # Verify order
        if not self.router.verify_event_order(self.expected_chain):
            raise AssertionError(
                f"Event order mismatch. "
                f"Expected: {self.expected_chain}, "
                f"Got: {[e.event_type for e in self.router.events]}"
            )

        # Verify timing if specified
        events = self.router.events
        event_times = {e.event_type: e.timestamp for e in events}

        for i, event_type in enumerate(self.expected_chain[1:], 1):
            if event_type in self.timing_requirements:
                prev_event = self.expected_chain[i - 1]
                if prev_event in event_times and event_type in event_times:
                    delay = event_times[event_type] - event_times[prev_event]
                    max_delay = self.timing_requirements[event_type]
                    if delay > max_delay:
                        raise AssertionError(
                            f"{event_type} took {delay:.1f}s after {prev_event}, "
                            f"expected < {max_delay}s"
                        )

        return True


@pytest.fixture
def event_chain_verifier(mock_event_router):
    """Provide an event chain verifier."""
    return EventChainVerifier(mock_event_router)


# Skip markers for slow tests
slow_integration = pytest.mark.skipif(
    "not config.getoption('--run-slow-integration')",
    reason="Slow integration test - use --run-slow-integration to run",
)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-slow-integration",
        action="store_true",
        default=False,
        help="Run slow integration tests",
    )
