"""Unit tests for event handler factories.

December 29, 2025: Tests for the reusable event handler generators.
"""

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.event_handler_factories import (
    create_logging_handler,
    create_forwarding_handler,
    create_remapping_handler,
    create_filtering_handler,
    create_debouncing_handler,
    create_aggregating_handler,
    create_config_specific_handler,
    create_node_specific_handler,
    HandlerConfig,
)


class TestCreateLoggingHandler:
    """Tests for create_logging_handler factory."""

    @pytest.mark.asyncio
    async def test_logs_event(self, caplog):
        """Test that handler logs event with extracted key."""
        handler = create_logging_handler("TEST_EVENT", extract_key="node_id")

        with caplog.at_level(logging.INFO):
            await handler({"node_id": "test-node-1", "data": "value"})

        assert "TEST_EVENT: test-node-1" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_key_logs_unknown(self, caplog):
        """Test that missing key logs 'unknown'."""
        handler = create_logging_handler("TEST_EVENT", extract_key="missing_key")

        with caplog.at_level(logging.INFO):
            await handler({"other_key": "value"})

        assert "TEST_EVENT: unknown" in caplog.text

    @pytest.mark.asyncio
    async def test_custom_log_level(self, caplog):
        """Test custom logging level."""
        handler = create_logging_handler(
            "DEBUG_EVENT",
            extract_key="id",
            log_level=logging.DEBUG,
        )

        with caplog.at_level(logging.DEBUG):
            await handler({"id": "123"})

        assert "DEBUG_EVENT: 123" in caplog.text


class TestCreateForwardingHandler:
    """Tests for create_forwarding_handler factory."""

    @pytest.mark.asyncio
    async def test_forwards_to_sync_method(self):
        """Test forwarding to synchronous method."""
        calls = []

        def target(event):
            calls.append(event)

        handler = create_forwarding_handler(target)
        await handler({"key": "value"})

        assert len(calls) == 1
        assert calls[0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_forwards_to_async_method(self):
        """Test forwarding to async method."""
        calls = []

        async def target(event):
            calls.append(event)

        handler = create_forwarding_handler(target)
        await handler({"async": True})

        assert len(calls) == 1
        assert calls[0]["async"] is True

    @pytest.mark.asyncio
    async def test_error_handler_called(self):
        """Test that error handler is called on exception."""
        errors = []

        def target(event):
            raise ValueError("Test error")

        def error_handler(exc, event):
            errors.append((exc, event))

        handler = create_forwarding_handler(target, error_handler=error_handler)
        await handler({"test": True})

        assert len(errors) == 1
        assert isinstance(errors[0][0], ValueError)
        assert errors[0][1]["test"] is True


class TestCreateRemappingHandler:
    """Tests for create_remapping_handler factory."""

    @pytest.mark.asyncio
    async def test_remaps_event_type(self):
        """Test that event is re-emitted with new type."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.publish_async = AsyncMock()
            mock_get_router.return_value = mock_router

            handler = create_remapping_handler("NEW_EVENT_TYPE")
            await handler({"original": "data"})

            mock_router.publish_async.assert_called_once()
            call_args = mock_router.publish_async.call_args
            assert call_args[0][0] == "NEW_EVENT_TYPE"
            assert call_args[0][1]["original"] == "data"

    @pytest.mark.asyncio
    async def test_key_mapping(self):
        """Test that keys are remapped."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.publish_async = AsyncMock()
            mock_get_router.return_value = mock_router

            handler = create_remapping_handler(
                "REMAPPED_EVENT",
                source_key_mapping={"old_key": "new_key"},
            )
            await handler({"old_key": "value"})

            call_args = mock_router.publish_async.call_args
            assert call_args[0][1]["new_key"] == "value"

    @pytest.mark.asyncio
    async def test_transform_function(self):
        """Test custom transform function."""
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.publish_async = AsyncMock()
            mock_get_router.return_value = mock_router

            def transform(event):
                return {"transformed": True, "count": event.get("count", 0) * 2}

            handler = create_remapping_handler("TRANSFORMED", transform=transform)
            await handler({"count": 5})

            call_args = mock_router.publish_async.call_args
            assert call_args[0][1]["transformed"] is True
            assert call_args[0][1]["count"] == 10


class TestCreateFilteringHandler:
    """Tests for create_filtering_handler factory."""

    @pytest.mark.asyncio
    async def test_passes_matching_events(self):
        """Test that matching events are passed to inner handler."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_filtering_handler(
            inner,
            lambda e: e.get("important") is True,
        )

        await handler({"important": True, "data": 1})
        await handler({"important": False, "data": 2})
        await handler({"important": True, "data": 3})

        assert len(calls) == 2
        assert calls[0]["data"] == 1
        assert calls[1]["data"] == 3

    @pytest.mark.asyncio
    async def test_filters_non_matching(self):
        """Test that non-matching events are filtered out."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_filtering_handler(
            inner,
            lambda e: e.get("type") == "allowed",
        )

        await handler({"type": "blocked"})
        await handler({"type": "blocked"})

        assert len(calls) == 0


class TestCreateDebouncingHandler:
    """Tests for create_debouncing_handler factory."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_debounces_rapid_events(self):
        """Test that rapid events are debounced."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_debouncing_handler(inner, debounce_seconds=0.05)

        # Send multiple events rapidly (not awaiting the background tasks)
        asyncio.create_task(handler({"n": 1}))
        asyncio.create_task(handler({"n": 2}))
        asyncio.create_task(handler({"n": 3}))

        # Wait for debounce
        await asyncio.sleep(0.1)

        # Only last event should be processed
        assert len(calls) == 1
        assert calls[0]["n"] == 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_per_key_debouncing(self):
        """Test per-key debouncing."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_debouncing_handler(
            inner,
            debounce_seconds=0.05,
            key_func=lambda e: e.get("config_key"),
        )

        # Send events for different keys
        asyncio.create_task(handler({"config_key": "hex8", "n": 1}))
        asyncio.create_task(handler({"config_key": "square8", "n": 2}))
        asyncio.create_task(handler({"config_key": "hex8", "n": 3}))

        # Wait for debounce
        await asyncio.sleep(0.1)

        # Each key should have its last event processed
        assert len(calls) == 2


class TestCreateAggregatingHandler:
    """Tests for create_aggregating_handler factory."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_aggregates_to_max_count(self):
        """Test that events are aggregated to max count."""
        batches = []

        async def batch_handler(events):
            batches.append(events)

        handler = create_aggregating_handler(
            batch_handler,
            max_count=3,
            max_wait_seconds=10.0,
        )

        # Send events
        for i in range(3):
            await handler({"n": i})

        # Wait briefly for processing
        await asyncio.sleep(0.02)

        # Should have triggered batch
        assert len(batches) == 1
        assert len(batches[0]) == 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_aggregates_on_timeout(self):
        """Test that events are processed after timeout."""
        batches = []

        async def batch_handler(events):
            batches.append(events)

        handler = create_aggregating_handler(
            batch_handler,
            max_count=100,  # High count
            max_wait_seconds=0.05,  # Short timeout
        )

        # Send fewer than max events
        await handler({"n": 1})
        await handler({"n": 2})

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Should have triggered batch
        assert len(batches) == 1
        assert len(batches[0]) == 2


class TestConvenienceFactories:
    """Tests for convenience factory functions."""

    @pytest.mark.asyncio
    async def test_config_specific_handler(self):
        """Test create_config_specific_handler."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_config_specific_handler("hex8_2p", inner)

        await handler({"config_key": "hex8_2p", "data": 1})
        await handler({"config_key": "square8_2p", "data": 2})
        await handler({"config_key": "hex8_2p", "data": 3})

        assert len(calls) == 2
        assert all(c["config_key"] == "hex8_2p" for c in calls)

    @pytest.mark.asyncio
    async def test_node_specific_handler(self):
        """Test create_node_specific_handler."""
        calls = []

        async def inner(event):
            calls.append(event)

        handler = create_node_specific_handler("nebius-h100-1", inner)

        await handler({"node_id": "nebius-h100-1", "data": 1})
        await handler({"node_id": "runpod-a100", "data": 2})
        await handler({"source_node": "nebius-h100-1", "data": 3})

        assert len(calls) == 2


class TestHandlerConfig:
    """Tests for HandlerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HandlerConfig()
        assert config.log_level == logging.INFO
        assert config.include_timestamp is False
        assert "node_id" in config.extract_keys
        assert "config_key" in config.extract_keys
        assert config.error_handler is None

    def test_custom_values(self):
        """Test custom configuration values."""
        def error_fn(e, evt):
            pass

        config = HandlerConfig(
            log_level=logging.DEBUG,
            include_timestamp=True,
            extract_keys=["custom_key"],
            error_handler=error_fn,
        )
        assert config.log_level == logging.DEBUG
        assert config.include_timestamp is True
        assert config.extract_keys == ["custom_key"]
        assert config.error_handler == error_fn
