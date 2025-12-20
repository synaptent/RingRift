"""Mock objects for testing.

Provides mock implementations of common interfaces and async utilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

__all__ = [
    "create_mock_model",
    "create_mock_coordinator",
    "MockAsyncContext",
    "MockEventBus",
    "AsyncMockContextManager",
]


@dataclass
class MockModel:
    """Mock neural network model for testing."""
    model_id: str = "test_model_v1"
    board_type: str = "square8"
    num_players: int = 2
    elo: float = 1500.0
    checkpoint_path: str | None = None

    def predict(self, state: Any) -> dict[str, Any]:
        """Mock prediction."""
        return {
            "policy": [0.1] * 10,
            "value": 0.5,
        }

    def evaluate(self, state: Any) -> float:
        """Mock evaluation."""
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "elo": self.elo,
            "checkpoint_path": self.checkpoint_path,
        }


@dataclass
class MockCoordinator:
    """Mock coordinator for testing."""
    name: str = "test_coordinator"
    is_active: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    async def start(self) -> None:
        """Mock start."""
        self.is_active = True

    async def stop(self) -> None:
        """Mock stop."""
        self.is_active = False

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Mock event emission."""
        self.events.append({"type": event_type, "data": data})

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "event_count": len(self.events),
        }


class MockEventBus:
    """Mock event bus for testing event-driven code."""

    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}
        self.published_events: list[dict[str, Any]] = []

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event."""
        self.published_events.append({"type": event_type, "data": data})

        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)

    def clear_events(self) -> None:
        """Clear published events history."""
        self.published_events.clear()


class MockAsyncContext:
    """Mock async context manager for testing.

    Usage:
        mock_ctx = MockAsyncContext(return_value=some_value)
        async with mock_ctx as value:
            assert value == some_value
    """

    def __init__(
        self,
        return_value: Any = None,
        on_enter: Callable | None = None,
        on_exit: Callable | None = None,
        raise_on_exit: Exception | None = None,
    ):
        self.return_value = return_value
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.raise_on_exit = raise_on_exit
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Any:
        self.entered = True
        if self.on_enter:
            if asyncio.iscoroutinefunction(self.on_enter):
                await self.on_enter()
            else:
                self.on_enter()
        return self.return_value

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.exited = True
        if self.on_exit:
            if asyncio.iscoroutinefunction(self.on_exit):
                await self.on_exit()
            else:
                self.on_exit()
        if self.raise_on_exit:
            raise self.raise_on_exit
        return False


class AsyncMockContextManager:
    """Create an async mock that works as a context manager.

    Usage:
        mock = AsyncMockContextManager()
        async with mock as value:
            # value is the return value
            pass
    """

    def __init__(self, return_value: Any = None):
        self.return_value = return_value
        self.mock = AsyncMock()

    async def __aenter__(self) -> Any:
        return self.return_value

    async def __aexit__(self, *args: Any) -> None:
        pass


def create_mock_model(
    model_id: str = "test_model_v1",
    board_type: str = "square8",
    num_players: int = 2,
    elo: float = 1500.0,
    checkpoint_path: str | None = None,
) -> MockModel:
    """Create a mock model for testing.

    Args:
        model_id: Model identifier
        board_type: Board type
        num_players: Number of players
        elo: Elo rating
        checkpoint_path: Path to checkpoint file

    Returns:
        MockModel instance
    """
    return MockModel(
        model_id=model_id,
        board_type=board_type,
        num_players=num_players,
        elo=elo,
        checkpoint_path=checkpoint_path,
    )


def create_mock_coordinator(
    name: str = "test_coordinator",
    is_active: bool = True,
) -> MockCoordinator:
    """Create a mock coordinator for testing.

    Args:
        name: Coordinator name
        is_active: Whether coordinator is active

    Returns:
        MockCoordinator instance
    """
    return MockCoordinator(name=name, is_active=is_active)
