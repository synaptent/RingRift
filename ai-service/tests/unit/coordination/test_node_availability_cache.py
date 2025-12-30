"""Tests for node_availability_cache.py.

December 29, 2025: Comprehensive test coverage for unified node availability cache.
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.node_availability_cache import (
    AvailabilityReason,
    NodeAvailabilityCache,
    NodeAvailabilityEntry,
    get_availability_cache,
)


class TestAvailabilityReason:
    """Tests for AvailabilityReason enum."""

    def test_all_reasons_exist(self):
        """Test all expected reasons are defined."""
        assert AvailabilityReason.UNKNOWN.value == "unknown"
        assert AvailabilityReason.HEALTHY.value == "healthy"
        assert AvailabilityReason.P2P_DEAD.value == "p2p_dead"
        assert AvailabilityReason.SSH_FAILED.value == "ssh_failed"
        assert AvailabilityReason.SSH_TIMEOUT.value == "ssh_timeout"
        assert AvailabilityReason.GPU_ERROR.value == "gpu_error"
        assert AvailabilityReason.HOST_OFFLINE.value == "host_offline"
        assert AvailabilityReason.MANUALLY_DISABLED.value == "manually_disabled"
        assert AvailabilityReason.HEALTH_CHECK_FAILED.value == "health_check_failed"
        assert AvailabilityReason.RECOVERED.value == "recovered"


class TestNodeAvailabilityEntry:
    """Tests for NodeAvailabilityEntry dataclass."""

    def test_default_values(self):
        """Test default entry values."""
        entry = NodeAvailabilityEntry(node_id="test-node")
        assert entry.node_id == "test-node"
        assert entry.is_available is True
        assert entry.reason == AvailabilityReason.UNKNOWN
        assert entry.consecutive_failures == 0
        assert entry.source == "unknown"
        assert entry.error_message is None

    def test_mark_available(self):
        """Test marking entry as available."""
        entry = NodeAvailabilityEntry(node_id="test-node")
        entry.mark_unavailable(AvailabilityReason.SSH_FAILED, "test")
        entry.mark_available("ssh_probe")

        assert entry.is_available is True
        assert entry.reason == AvailabilityReason.HEALTHY
        assert entry.consecutive_failures == 0
        assert entry.source == "ssh_probe"
        assert entry.error_message is None

    def test_mark_unavailable(self):
        """Test marking entry as unavailable."""
        entry = NodeAvailabilityEntry(node_id="test-node")
        entry.mark_unavailable(
            AvailabilityReason.SSH_TIMEOUT,
            source="health_check",
            error_message="Connection timed out",
        )

        assert entry.is_available is False
        assert entry.reason == AvailabilityReason.SSH_TIMEOUT
        assert entry.consecutive_failures == 1
        assert entry.source == "health_check"
        assert entry.error_message == "Connection timed out"

    def test_consecutive_failures_increment(self):
        """Test consecutive failures increment on each failure."""
        entry = NodeAvailabilityEntry(node_id="test-node")

        entry.mark_unavailable(AvailabilityReason.SSH_FAILED, "test")
        assert entry.consecutive_failures == 1

        entry.mark_unavailable(AvailabilityReason.SSH_FAILED, "test")
        assert entry.consecutive_failures == 2

        entry.mark_unavailable(AvailabilityReason.SSH_FAILED, "test")
        assert entry.consecutive_failures == 3

    def test_seconds_since_contact(self):
        """Test seconds_since_contact calculation."""
        entry = NodeAvailabilityEntry(node_id="test-node")
        entry.last_successful_contact = time.time() - 60  # 1 minute ago

        elapsed = entry.seconds_since_contact()
        assert 59 < elapsed < 61

    def test_is_stale(self):
        """Test staleness detection."""
        entry = NodeAvailabilityEntry(node_id="test-node")

        # Fresh entry should not be stale
        assert entry.is_stale(max_age_seconds=60.0) is False

        # Old entry should be stale
        entry.last_update = time.time() - 120
        assert entry.is_stale(max_age_seconds=60.0) is True


class TestNodeAvailabilityCache:
    """Tests for NodeAvailabilityCache class."""

    def setup_method(self):
        """Reset singleton before each test."""
        NodeAvailabilityCache.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NodeAvailabilityCache.reset_instance()

    def test_singleton_pattern(self):
        """Test singleton instance creation."""
        cache1 = NodeAvailabilityCache.get_instance()
        cache2 = NodeAvailabilityCache.get_instance()
        assert cache1 is cache2

    def test_reset_instance(self):
        """Test singleton reset."""
        cache1 = NodeAvailabilityCache.get_instance()
        NodeAvailabilityCache.reset_instance()
        cache2 = NodeAvailabilityCache.get_instance()
        assert cache1 is not cache2

    def test_is_available_unknown_node(self):
        """Test unknown nodes are assumed available (optimistic)."""
        cache = NodeAvailabilityCache.get_instance()
        assert cache.is_available("unknown-node") is True

    def test_mark_available(self):
        """Test marking node as available."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("test-node", source="test")

        assert cache.is_available("test-node") is True
        entry = cache.get_entry("test-node")
        assert entry is not None
        assert entry.source == "test"

    def test_mark_unavailable(self):
        """Test marking node as unavailable."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_unavailable(
            "test-node",
            AvailabilityReason.SSH_FAILED,
            source="health_check",
            error_message="Connection refused",
        )

        assert cache.is_available("test-node") is False
        entry = cache.get_entry("test-node")
        assert entry is not None
        assert entry.reason == AvailabilityReason.SSH_FAILED
        assert entry.error_message == "Connection refused"

    def test_get_available_nodes(self):
        """Test getting list of available nodes."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("healthy-1", "test")
        cache.mark_available("healthy-2", "test")
        cache.mark_unavailable("unhealthy-1", AvailabilityReason.P2P_DEAD, "test")

        available = cache.get_available_nodes()
        assert "healthy-1" in available
        assert "healthy-2" in available
        assert "unhealthy-1" not in available

    def test_get_unavailable_nodes(self):
        """Test getting list of unavailable nodes."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("healthy-1", "test")
        cache.mark_unavailable("unhealthy-1", AvailabilityReason.P2P_DEAD, "test")
        cache.mark_unavailable("unhealthy-2", AvailabilityReason.SSH_TIMEOUT, "test")

        unavailable = cache.get_unavailable_nodes()
        assert "unhealthy-1" in unavailable
        assert "unhealthy-2" in unavailable
        assert "healthy-1" not in unavailable

    def test_get_entry(self):
        """Test getting specific entry."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("test-node", "test")

        entry = cache.get_entry("test-node")
        assert entry is not None
        assert entry.node_id == "test-node"

    def test_get_entry_nonexistent(self):
        """Test getting entry for nonexistent node."""
        cache = NodeAvailabilityCache.get_instance()
        entry = cache.get_entry("nonexistent")
        assert entry is None

    def test_get_all_entries(self):
        """Test getting all entries."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("node-1", "test")
        cache.mark_available("node-2", "test")

        entries = cache.get_all_entries()
        assert len(entries) == 2
        assert "node-1" in entries
        assert "node-2" in entries

    def test_get_status_summary(self):
        """Test getting status summary."""
        cache = NodeAvailabilityCache.get_instance()
        cache.mark_available("healthy-1", "test")
        cache.mark_available("healthy-2", "test")
        cache.mark_unavailable("unhealthy-1", AvailabilityReason.SSH_FAILED, "test")
        cache.mark_unavailable("unhealthy-2", AvailabilityReason.P2P_DEAD, "test")

        summary = cache.get_status_summary()
        assert summary["total_nodes"] == 4
        assert summary["available"] == 2
        assert summary["unavailable"] == 2
        assert AvailabilityReason.SSH_FAILED.value in summary["by_reason"]
        assert AvailabilityReason.P2P_DEAD.value in summary["by_reason"]

    def test_clear_stale_entries(self):
        """Test clearing stale entries."""
        cache = NodeAvailabilityCache.get_instance()
        cache.stale_threshold_seconds = 1.0  # Very short for testing

        cache.mark_available("fresh-node", "test")
        cache.mark_available("stale-node", "test")

        # Make one entry stale
        entry = cache.get_entry("stale-node")
        entry.last_update = time.time() - 10

        cleared = cache.clear_stale_entries()
        assert cleared == 1
        assert cache.get_entry("fresh-node") is not None
        assert cache.get_entry("stale-node") is None

    def test_auto_recovery(self):
        """Test auto-recovery after long period without contact."""
        cache = NodeAvailabilityCache.get_instance()
        cache.auto_recover_after_seconds = 1.0  # Very short for testing

        cache.mark_unavailable("test-node", AvailabilityReason.SSH_FAILED, "test")
        assert cache.is_available("test-node") is False

        # Simulate long time without contact
        entry = cache.get_entry("test-node")
        entry.last_successful_contact = time.time() - 10

        # Should auto-recover
        assert cache.is_available("test-node") is True

    def test_on_unavailable_callback(self):
        """Test callback when node becomes unavailable."""
        cache = NodeAvailabilityCache.get_instance()
        callback = MagicMock()
        cache.on_unavailable(callback)

        cache.mark_unavailable("test-node", AvailabilityReason.SSH_FAILED, "test")

        callback.assert_called_once_with("test-node", AvailabilityReason.SSH_FAILED)

    def test_on_available_callback(self):
        """Test callback when node becomes available."""
        cache = NodeAvailabilityCache.get_instance()
        callback = MagicMock()
        cache.on_available(callback)

        # First mark unavailable, then available
        cache.mark_unavailable("test-node", AvailabilityReason.SSH_FAILED, "test")
        cache.mark_available("test-node", "test")

        callback.assert_called_once_with("test-node")

    def test_callback_not_called_same_state(self):
        """Test callbacks not called when state doesn't change."""
        cache = NodeAvailabilityCache.get_instance()
        unavailable_callback = MagicMock()
        available_callback = MagicMock()
        cache.on_unavailable(unavailable_callback)
        cache.on_available(available_callback)

        # Mark available twice (starts as available for unknown)
        cache.mark_available("test-node", "test")
        cache.mark_available("test-node", "test")

        # Should not call available_callback (wasn't unavailable)
        available_callback.assert_not_called()

    def test_callback_error_handling(self):
        """Test callback errors are handled gracefully."""
        cache = NodeAvailabilityCache.get_instance()
        callback = MagicMock(side_effect=Exception("Callback error"))
        cache.on_unavailable(callback)

        # Should not raise despite callback error
        cache.mark_unavailable("test-node", AvailabilityReason.SSH_FAILED, "test")

    @pytest.mark.asyncio
    async def test_on_node_dead_event_dict(self):
        """Test handling P2P_NODE_DEAD event with dict payload."""
        cache = NodeAvailabilityCache.get_instance()
        event = {"node_id": "dead-node", "reason": "timeout"}

        await cache._on_node_dead_event(event)

        assert cache.is_available("dead-node") is False
        entry = cache.get_entry("dead-node")
        assert entry.reason == AvailabilityReason.P2P_DEAD

    @pytest.mark.asyncio
    async def test_on_node_dead_event_object(self):
        """Test handling P2P_NODE_DEAD event with object payload."""
        cache = NodeAvailabilityCache.get_instance()

        mock_event = MagicMock()
        mock_event.payload = {"node_id": "dead-node", "reason": "timeout"}

        await cache._on_node_dead_event(mock_event)

        assert cache.is_available("dead-node") is False

    @pytest.mark.asyncio
    async def test_on_host_offline_event(self):
        """Test handling HOST_OFFLINE event."""
        cache = NodeAvailabilityCache.get_instance()
        event = {"node_id": "offline-node"}

        await cache._on_host_offline_event(event)

        assert cache.is_available("offline-node") is False
        entry = cache.get_entry("offline-node")
        assert entry.reason == AvailabilityReason.HOST_OFFLINE

    @pytest.mark.asyncio
    async def test_on_host_offline_event_with_host_key(self):
        """Test HOST_OFFLINE event with 'host' key instead of 'node_id'."""
        cache = NodeAvailabilityCache.get_instance()
        event = {"host": "offline-host"}

        await cache._on_host_offline_event(event)

        assert cache.is_available("offline-host") is False

    @pytest.mark.asyncio
    async def test_on_node_recovered_event(self):
        """Test handling NODE_RECOVERED event."""
        cache = NodeAvailabilityCache.get_instance()

        # First mark unavailable
        cache.mark_unavailable("test-node", AvailabilityReason.SSH_FAILED, "test")
        assert cache.is_available("test-node") is False

        # Then recover via event
        event = {"node_id": "test-node"}
        await cache._on_node_recovered_event(event)

        assert cache.is_available("test-node") is True

    def test_wire_to_events(self):
        """Test subscribing to events."""
        cache = NodeAvailabilityCache.get_instance()

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            cache.wire_to_events()

            # Should have subscribed to events (P2P_NODE_DEAD, HOST_OFFLINE, NODE_RECOVERED, NODE_UNHEALTHY)
            assert mock_router.subscribe.call_count == 4
            assert cache._event_subscribed is True

    def test_wire_to_events_idempotent(self):
        """Test wire_to_events is idempotent."""
        cache = NodeAvailabilityCache.get_instance()

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            cache.wire_to_events()
            cache.wire_to_events()  # Call again

            # Should only subscribe once (4 events: P2P_NODE_DEAD, HOST_OFFLINE, NODE_RECOVERED, NODE_UNHEALTHY)
            assert mock_router.subscribe.call_count == 4

    def test_wire_to_events_handles_import_error(self):
        """Test wire_to_events handles import error gracefully."""
        cache = NodeAvailabilityCache.get_instance()
        cache._event_subscribed = False  # Reset state

        # Patch the import inside wire_to_events to raise ImportError
        with patch.dict("sys.modules", {"app.coordination.event_router": None}):
            # Should not raise
            cache.wire_to_events()
            # Event subscription should fail silently
            assert cache._event_subscribed is False


class TestModuleLevelAccessor:
    """Tests for module-level accessor function."""

    def setup_method(self):
        """Reset singleton before each test."""
        NodeAvailabilityCache.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NodeAvailabilityCache.reset_instance()

    def test_get_availability_cache(self):
        """Test get_availability_cache returns singleton."""
        cache1 = get_availability_cache()
        cache2 = get_availability_cache()
        assert cache1 is cache2
        assert isinstance(cache1, NodeAvailabilityCache)


class TestThreadSafety:
    """Tests for thread safety of NodeAvailabilityCache."""

    def setup_method(self):
        """Reset singleton before each test."""
        NodeAvailabilityCache.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        NodeAvailabilityCache.reset_instance()

    def test_concurrent_mark_operations(self):
        """Test concurrent mark operations don't corrupt state."""
        import threading

        cache = NodeAvailabilityCache.get_instance()
        errors = []

        def mark_available_worker():
            try:
                for i in range(100):
                    cache.mark_available(f"node-{i % 10}", "test")
            except Exception as e:
                errors.append(e)

        def mark_unavailable_worker():
            try:
                for i in range(100):
                    cache.mark_unavailable(
                        f"node-{i % 10}",
                        AvailabilityReason.SSH_FAILED,
                        "test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mark_available_worker),
            threading.Thread(target=mark_unavailable_worker),
            threading.Thread(target=mark_available_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Cache should have entries (exact state depends on race conditions)
        assert len(cache.get_all_entries()) > 0
