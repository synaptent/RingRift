"""Tests for event_fallback_queue sync behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.coordination.event_fallback_queue import (
    EventFallbackQueue,
    FallbackQueueConfig,
    sync_queued_events,
)


@pytest.fixture(autouse=True)
def reset_fallback_queue() -> None:
    """Reset singleton before and after each test."""
    EventFallbackQueue.reset_instance()
    yield
    EventFallbackQueue.reset_instance()


@pytest.mark.asyncio
async def test_sync_queued_events_uses_publish_sync(tmp_path: Path):
    """Queued events should replay through the unified sync publish helper."""
    from app.coordination.event_router import DataEventType

    queue = EventFallbackQueue.get_instance(
        FallbackQueueConfig(db_path=tmp_path / "event_fallback_queue.db")
    )
    queued = queue.queue_event(
        DataEventType.TRAINING_COMPLETED.value,
        {"config_key": "hex8_2p", "metric": 0.9},
        source="worker-1",
    )
    assert queued is True

    with patch("app.coordination.event_router.publish_sync") as mock_publish_sync:
        synced_count = await sync_queued_events(batch_size=10)

    assert synced_count == 1
    assert mock_publish_sync.call_count == 2

    replay_call = mock_publish_sync.call_args_list[0]
    replay_args, replay_kwargs = replay_call
    assert replay_args[0] == DataEventType.TRAINING_COMPLETED
    assert replay_args[1]["config_key"] == "hex8_2p"
    assert replay_args[1]["_fallback_queue"]["original_source"] == "worker-1"
    assert replay_kwargs["source"] == "event_fallback_queue"

    completion_call = mock_publish_sync.call_args_list[1]
    completion_args, completion_kwargs = completion_call
    assert completion_args[0] == DataEventType.QUEUED_EVENTS_SYNCED
    assert completion_args[1]["count"] == 1
    assert completion_kwargs["source"] == "event_fallback_queue"
