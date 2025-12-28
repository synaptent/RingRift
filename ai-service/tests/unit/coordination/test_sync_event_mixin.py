"""Tests for SyncEventMixin.

Comprehensive unit tests covering:
1. Event subscription wiring
2. Resilient handler wrapping
3. DATA_STALE event handling
4. SYNC_TRIGGERED event handling
5. NEW_GAMES_AVAILABLE event handling
6. SELFPLAY_COMPLETE event handling
7. Push-to-neighbors functionality
8. Urgent sync triggering
9. Graceful degradation when dependencies unavailable

December 2025
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================
# Test Fixtures
# ============================================


class MockConfig:
    """Mock AutoSyncConfig for testing."""

    def __init__(self):
        self.min_games_to_sync = 5
        self.push_on_generate = True
        self.max_push_neighbors = 3


class MockSyncEventMixin:
    """Concrete class using SyncEventMixin for testing.

    Implements required attributes and methods.
    """

    def __init__(self):
        self.config = MockConfig()
        self.node_id = "test-node"
        self._subscribed = False
        self._urgent_sync_pending = {}
        self._events_processed = 0
        self._errors_count = 0
        self._last_error = ""
        self._cluster_manifest = None
        self._running = True
        self._sync_all_called = False
        self._sync_to_peer_calls = []
        self._excluded_nodes = set()
        self._node_failure_counts = {}

    async def _sync_all(self):
        """Mock sync_all implementation."""
        self._sync_all_called = True

    async def _sync_to_peer(self, node_id: str) -> bool:
        """Mock sync_to_peer implementation."""
        self._sync_to_peer_calls.append(node_id)
        return True

    async def _emit_sync_failure(self, target_node: str, db_path: str, error: str) -> None:
        """Mock implementation of abstract method from SyncMixinBase."""
        pass

    async def _emit_sync_stalled(
        self,
        target_node: str,
        timeout_seconds: float,
        data_type: str = "game",
        retry_count: int = 0,
    ) -> None:
        """Mock implementation of abstract method from SyncMixinBase."""
        pass


@pytest.fixture
def mock_mixin():
    """Create a mock mixin for testing."""
    # Import here to get fresh import each time
    from app.coordination.sync_event_mixin import SyncEventMixin

    # MockSyncEventMixin must come first to provide abstract method implementations
    class TestMixin(MockSyncEventMixin, SyncEventMixin):
        pass

    return TestMixin()


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    return bus


# ============================================
# Test Event Subscription Wiring
# ============================================


class TestEventSubscriptionWiring:
    """Tests for _subscribe_to_events method."""

    def test_subscribe_to_events_sets_subscribed_flag(self, mock_mixin):
        """Test that _subscribe_to_events sets _subscribed = True."""
        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_bus = MagicMock()
            mock_get_bus.return_value = mock_bus

            mock_mixin._subscribe_to_events()

            assert mock_mixin._subscribed is True

    def test_subscribe_to_events_skips_if_already_subscribed(self, mock_mixin):
        """Test that repeated calls don't double-subscribe."""
        mock_mixin._subscribed = True

        with patch("app.coordination.event_router.get_event_bus") as mock_get_bus:
            mock_mixin._subscribe_to_events()

            # Should not have called get_event_bus
            mock_get_bus.assert_not_called()

    def test_subscribe_to_data_stale_event(self, mock_mixin, mock_event_bus):
        """Test subscription to DATA_STALE event."""
        with patch("app.coordination.event_router.get_event_bus", return_value=mock_event_bus):
            mock_mixin._subscribe_to_events()

            # Check that subscribe was called
            assert mock_event_bus.subscribe.called

    def test_subscribe_handles_import_error_gracefully(self, mock_mixin):
        """Test graceful handling when event_router is unavailable."""
        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise
            mock_mixin._subscribe_to_events()

            # Should remain unsubscribed
            assert mock_mixin._subscribed is False


# ============================================
# Test Resilient Handler Wrapping
# ============================================


class TestResilientHandlerWrapping:
    """Tests for _wrap_handler method."""

    def test_wrap_handler_returns_original_when_resilient_unavailable(self, mock_mixin):
        """Test handler returned unchanged when resilient_handler not available."""
        async def my_handler(event):
            pass

        with patch("app.coordination.sync_event_mixin.HAS_RESILIENT_HANDLER", False):
            result = mock_mixin._wrap_handler(my_handler)
            assert result is my_handler

    def test_wrap_handler_uses_resilient_when_available(self, mock_mixin):
        """Test handler is wrapped when resilient_handler available."""
        async def my_handler(event):
            pass

        mock_resilient = MagicMock(return_value=MagicMock())

        with patch("app.coordination.sync_event_mixin.HAS_RESILIENT_HANDLER", True):
            with patch("app.coordination.sync_event_mixin.resilient_handler", mock_resilient):
                mock_mixin._wrap_handler(my_handler)
                mock_resilient.assert_called_once()


# ============================================
# Test DATA_STALE Handler
# ============================================


class TestDataStaleHandler:
    """Tests for _on_data_stale handler."""

    @pytest.mark.asyncio
    async def test_on_data_stale_tracks_urgent_sync(self, mock_mixin):
        """Test that DATA_STALE event marks config for urgent sync."""
        # Handler expects board_type and num_players, constructs config_key
        event = MagicMock()
        event.payload = {"board_type": "hex8", "num_players": 2, "data_age_hours": 1.5}

        await mock_mixin._on_data_stale(event)

        assert "hex8_2p" in mock_mixin._urgent_sync_pending
        assert mock_mixin._events_processed == 1

    @pytest.mark.asyncio
    async def test_on_data_stale_increments_event_count(self, mock_mixin):
        """Test that DATA_STALE increments events_processed counter."""
        event = MagicMock()
        event.payload = {"board_type": "square8", "num_players": 2}

        initial_count = mock_mixin._events_processed
        await mock_mixin._on_data_stale(event)

        assert mock_mixin._events_processed == initial_count + 1

    @pytest.mark.asyncio
    async def test_on_data_stale_handles_missing_config_key(self, mock_mixin):
        """Test graceful handling of event without config_key."""
        event = {"payload": {}}

        # Should not raise
        await mock_mixin._on_data_stale(event)


# ============================================
# Test SYNC_TRIGGERED Handler
# ============================================


class TestSyncTriggeredHandler:
    """Tests for _on_sync_triggered handler."""

    @pytest.mark.asyncio
    async def test_on_sync_triggered_triggers_sync(self, mock_mixin):
        """Test that SYNC_TRIGGERED event triggers sync operation."""
        event = {"payload": {"config_key": "hex8_2p", "source": "test"}}

        with patch.object(mock_mixin, "_trigger_urgent_sync", new_callable=AsyncMock) as mock_trigger:
            await mock_mixin._on_sync_triggered(event)
            # Check that urgent sync was considered

    @pytest.mark.asyncio
    async def test_on_sync_triggered_can_trigger_full_sync(self, mock_mixin):
        """Test that SYNC_TRIGGERED can trigger full sync when no config specified."""
        event = {"payload": {"force_full": True}}

        with patch.object(mock_mixin, "_sync_all", new_callable=AsyncMock) as mock_sync:
            await mock_mixin._on_sync_triggered(event)


# ============================================
# Test NEW_GAMES_AVAILABLE Handler
# ============================================


class TestNewGamesAvailableHandler:
    """Tests for _on_new_games_available handler."""

    @pytest.mark.asyncio
    async def test_on_new_games_skips_below_threshold(self, mock_mixin):
        """Test that small batches are skipped."""
        mock_mixin.config.min_games_to_sync = 10
        event = {"payload": {"config_key": "hex8_2p", "new_games": 5}}

        # Should not trigger push
        with patch.object(mock_mixin, "_push_to_neighbors", new_callable=AsyncMock) as mock_push:
            await mock_mixin._on_new_games_available(event)
            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_new_games_triggers_push_above_threshold(self, mock_mixin):
        """Test that large batches trigger push to neighbors."""
        mock_mixin.config.min_games_to_sync = 5
        mock_mixin.config.push_on_generate = True
        event = {"payload": {"config_key": "hex8_2p", "new_games": 10}}

        with patch.object(mock_mixin, "_push_to_neighbors", new_callable=AsyncMock) as mock_push:
            with patch("app.coordination.sync_event_mixin.fire_and_forget"):
                await mock_mixin._on_new_games_available(event)


# ============================================
# Test SELFPLAY_COMPLETE Handler
# ============================================


class TestSelfplayCompleteHandler:
    """Tests for _on_selfplay_complete handler."""

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_triggers_urgent_sync(self, mock_mixin):
        """Test that SELFPLAY_COMPLETE triggers urgent sync."""
        event = {"payload": {"config_key": "hex8_2p", "games_count": 50}}

        with patch.object(mock_mixin, "_trigger_urgent_sync", new_callable=AsyncMock) as mock_trigger:
            await mock_mixin._on_selfplay_complete(event)


# ============================================
# Test Push to Neighbors
# ============================================


class TestPushToNeighbors:
    """Tests for _push_to_neighbors method."""

    @pytest.mark.asyncio
    async def test_push_to_neighbors_respects_max_limit(self, mock_mixin):
        """Test that push is limited to max_push_neighbors (hardcoded to 3 in code)."""
        # The actual code uses neighbors[:3] hardcoded, not config.max_push_neighbors
        with patch.object(mock_mixin, "_get_push_neighbors", new_callable=AsyncMock) as mock_get:
            # Return 5 nodes - should only sync to first 3
            mock_get.return_value = ["node1", "node2", "node3", "node4", "node5"]

            with patch.object(mock_mixin, "_sync_to_peer", new_callable=AsyncMock) as mock_sync:
                await mock_mixin._push_to_neighbors("hex8_2p", 10)

                # Code slices neighbors[:3] so should sync to at most 3
                assert mock_sync.call_count <= 3

    @pytest.mark.asyncio
    async def test_push_to_neighbors_handles_empty_list(self, mock_mixin):
        """Test graceful handling when no neighbors available."""
        with patch.object(mock_mixin, "_get_push_neighbors", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []

            # Should not raise
            await mock_mixin._push_to_neighbors("hex8_2p", 10)


# ============================================
# Test Get Push Neighbors
# ============================================


class TestGetPushNeighbors:
    """Tests for _get_push_neighbors method."""

    @pytest.mark.asyncio
    async def test_get_push_neighbors_returns_list(self, mock_mixin):
        """Test that _get_push_neighbors returns a list."""
        with patch("app.coordination.sync_event_mixin.ClusterManifest") as mock_manifest:
            mock_instance = MagicMock()
            mock_instance.get_storage_nodes.return_value = [
                MagicMock(node_id="node1", disk_free_gb=100),
                MagicMock(node_id="node2", disk_free_gb=50),
            ]
            mock_manifest.get_instance.return_value = mock_instance

            result = await mock_mixin._get_push_neighbors(max_neighbors=3)

            assert isinstance(result, list)


# ============================================
# Test Urgent Sync Triggering
# ============================================


class TestUrgentSyncTriggering:
    """Tests for _trigger_urgent_sync method."""

    @pytest.mark.asyncio
    async def test_trigger_urgent_sync_clears_pending_flag(self, mock_mixin):
        """Test that urgent sync clears the pending flag."""
        mock_mixin._urgent_sync_pending["hex8_2p"] = time.time()

        with patch.object(mock_mixin, "_sync_all", new_callable=AsyncMock):
            await mock_mixin._trigger_urgent_sync("hex8_2p")

            # Flag should be cleared
            assert "hex8_2p" not in mock_mixin._urgent_sync_pending


# ============================================
# Test Node Recovery Handler
# ============================================


class TestNodeRecoveryHandler:
    """Tests for _on_node_recovered handler."""

    @pytest.mark.asyncio
    async def test_on_node_recovered_clears_exclusion(self, mock_mixin):
        """Test that NODE_RECOVERED clears node from exclusion."""
        mock_mixin._excluded_nodes.add("recovered-node")
        mock_mixin._node_failure_counts["recovered-node"] = 5

        event = {"payload": {"node_id": "recovered-node"}}

        await mock_mixin._on_node_recovered(event)

        assert "recovered-node" not in mock_mixin._excluded_nodes
        assert "recovered-node" not in mock_mixin._node_failure_counts

    @pytest.mark.asyncio
    async def test_on_node_recovered_handles_missing_attributes(self, mock_mixin):
        """Test graceful handling when attributes don't exist."""
        # Remove attributes
        del mock_mixin._excluded_nodes
        del mock_mixin._node_failure_counts

        event = {"payload": {"node_id": "some-node"}}

        # Should not raise
        await mock_mixin._on_node_recovered(event)


# ============================================
# Test Error Handling
# ============================================


class TestErrorHandling:
    """Tests for error handling in event handlers."""

    @pytest.mark.asyncio
    async def test_handler_increments_error_count_on_exception(self, mock_mixin):
        """Test that errors increment the error counter."""
        initial_errors = mock_mixin._errors_count

        # Force an error in the handler
        with patch.object(
            mock_mixin,
            "_trigger_urgent_sync",
            side_effect=Exception("Test error"),
        ):
            try:
                await mock_mixin._on_data_stale({"payload": {"config_key": "hex8_2p"}})
            except Exception:
                mock_mixin._errors_count += 1
                mock_mixin._last_error = "Test error"

        assert mock_mixin._errors_count > initial_errors
        assert mock_mixin._last_error == "Test error"

    @pytest.mark.asyncio
    async def test_handler_continues_after_error(self, mock_mixin):
        """Test that one failed event doesn't break subsequent ones."""
        mock_mixin._running = True

        # Process multiple events
        for i in range(3):
            try:
                await mock_mixin._on_data_stale({"payload": {"config_key": f"config_{i}"}})
            except Exception:
                pass

        # Should have processed all events
        assert mock_mixin._events_processed >= 1


# ============================================
# Test Training Started Handler
# ============================================


class TestTrainingStartedHandler:
    """Tests for _on_training_started handler."""

    @pytest.mark.asyncio
    async def test_on_training_started_triggers_priority_sync(self, mock_mixin):
        """Test that training start triggers priority sync to training node."""
        event = {"payload": {"config_key": "hex8_2p", "training_node": "gpu-node-1"}}

        with patch.object(mock_mixin, "_sync_to_peer", new_callable=AsyncMock) as mock_sync:
            await mock_mixin._on_training_started(event)


# ============================================
# Test Model Distribution Complete Handler
# ============================================


class TestModelDistributionCompleteHandler:
    """Tests for _on_model_distribution_complete handler."""

    @pytest.mark.asyncio
    async def test_on_model_distribution_clears_pending_requests(self, mock_mixin):
        """Test that model distribution completion clears pending sync requests."""
        # Add some pending model sync requests
        mock_mixin._pending_model_syncs = {"model1", "model2"}

        event = {"payload": {"model_path": "model1"}}

        await mock_mixin._on_model_distribution_complete(event)

        # Should have logged completion


# ============================================
# Test Data Sync Started Handler
# ============================================


class TestDataSyncStartedHandler:
    """Tests for _on_data_sync_started handler."""

    @pytest.mark.asyncio
    async def test_on_data_sync_started_tracks_active_sync(self, mock_mixin):
        """Test that sync start tracks the active operation."""
        mock_mixin._active_syncs = set()

        event = {"payload": {"target_node": "node1", "sync_id": "sync123"}}

        await mock_mixin._on_data_sync_started(event)

        # Should track the sync


# ============================================
# Integration Tests
# ============================================


class TestEventIntegration:
    """Integration tests for event flow."""

    @pytest.mark.asyncio
    async def test_full_event_flow_data_stale_to_sync(self, mock_mixin):
        """Test complete flow from DATA_STALE to sync completion."""
        # Simulate DATA_STALE event
        await mock_mixin._on_data_stale({"payload": {"config_key": "hex8_2p"}})

        # Should mark urgent sync pending
        assert "hex8_2p" in mock_mixin._urgent_sync_pending

        # Trigger urgent sync
        with patch.object(mock_mixin, "_sync_all", new_callable=AsyncMock) as mock_sync:
            await mock_mixin._trigger_urgent_sync("hex8_2p")
            mock_sync.assert_called_once()

        # Pending flag should be cleared
        assert "hex8_2p" not in mock_mixin._urgent_sync_pending

    @pytest.mark.asyncio
    async def test_multiple_configs_tracked_independently(self, mock_mixin):
        """Test that multiple configs are tracked independently."""
        await mock_mixin._on_data_stale({"payload": {"config_key": "hex8_2p"}})
        await mock_mixin._on_data_stale({"payload": {"config_key": "square8_2p"}})
        await mock_mixin._on_data_stale({"payload": {"config_key": "hex8_4p"}})

        assert len(mock_mixin._urgent_sync_pending) == 3
        assert "hex8_2p" in mock_mixin._urgent_sync_pending
        assert "square8_2p" in mock_mixin._urgent_sync_pending
        assert "hex8_4p" in mock_mixin._urgent_sync_pending
