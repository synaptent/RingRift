"""Tests for PER Orchestrator (Prioritized Experience Replay monitoring).

Tests cover:
- PERBufferState and PERStats dataclasses
- PEROrchestrator event handling and state management
- Module functions (wire_per_events, get_per_orchestrator, reset_per_orchestrator)
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


# =============================================================================
# Test PERBufferState Dataclass
# =============================================================================

class TestPERBufferState:
    """Tests for PERBufferState dataclass."""

    def test_create_with_required_fields(self):
        """Test creating state with required fields."""
        from app.training.per_orchestrator import PERBufferState

        state = PERBufferState(
            buffer_path="/path/to/buffer",
            buffer_size=10000,
        )

        assert state.buffer_path == "/path/to/buffer"
        assert state.buffer_size == 10000

    def test_default_values(self):
        """Test default values are set correctly."""
        from app.training.per_orchestrator import PERBufferState

        state = PERBufferState(buffer_path="test", buffer_size=100)

        assert state.tree_depth == 0
        assert state.priority_sum == 0.0
        assert state.min_priority == 0.0
        assert state.max_priority == 1.0
        assert state.last_rebuild_time == 0.0
        assert state.last_priority_update_time == 0.0
        assert state.rebuild_count == 0
        assert state.priority_update_count == 0
        assert state.config_key == ""

    def test_all_fields(self):
        """Test creating state with all fields."""
        from app.training.per_orchestrator import PERBufferState

        state = PERBufferState(
            buffer_path="/path/to/buffer",
            buffer_size=50000,
            tree_depth=16,
            priority_sum=12345.67,
            min_priority=0.001,
            max_priority=0.999,
            last_rebuild_time=1000.0,
            last_priority_update_time=2000.0,
            rebuild_count=5,
            priority_update_count=100,
            config_key="sq8_2p",
        )

        assert state.buffer_path == "/path/to/buffer"
        assert state.buffer_size == 50000
        assert state.tree_depth == 16
        assert state.priority_sum == 12345.67
        assert state.min_priority == 0.001
        assert state.max_priority == 0.999
        assert state.last_rebuild_time == 1000.0
        assert state.last_priority_update_time == 2000.0
        assert state.rebuild_count == 5
        assert state.priority_update_count == 100
        assert state.config_key == "sq8_2p"


# =============================================================================
# Test PERStats Dataclass
# =============================================================================

class TestPERStats:
    """Tests for PERStats dataclass."""

    def test_default_values(self):
        """Test default values for aggregate stats."""
        from app.training.per_orchestrator import PERStats

        stats = PERStats()

        assert stats.total_buffers_tracked == 0
        assert stats.total_rebuilds == 0
        assert stats.total_priority_updates == 0
        assert stats.active_buffers == 0
        assert stats.total_samples == 0
        assert stats.avg_buffer_size == 0.0
        assert stats.last_activity_time == 0.0

    def test_all_fields(self):
        """Test creating stats with all fields."""
        from app.training.per_orchestrator import PERStats

        stats = PERStats(
            total_buffers_tracked=5,
            total_rebuilds=20,
            total_priority_updates=1000,
            active_buffers=3,
            total_samples=150000,
            avg_buffer_size=50000.0,
            last_activity_time=time.time(),
        )

        assert stats.total_buffers_tracked == 5
        assert stats.total_rebuilds == 20
        assert stats.total_priority_updates == 1000
        assert stats.active_buffers == 3
        assert stats.total_samples == 150000
        assert stats.avg_buffer_size == 50000.0
        assert stats.last_activity_time > 0


# =============================================================================
# Test PEROrchestrator
# =============================================================================

class TestPEROrchestrator:
    """Tests for PEROrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance."""
        from app.training.per_orchestrator import PEROrchestrator, reset_per_orchestrator

        reset_per_orchestrator()
        orch = PEROrchestrator()
        yield orch
        reset_per_orchestrator()

    def test_initialization_defaults(self, orchestrator):
        """Test default initialization."""
        assert orchestrator.buffer_stale_threshold_seconds == 3600.0
        assert orchestrator.max_history_per_buffer == 100
        assert not orchestrator._subscribed
        assert len(orchestrator._buffers) == 0
        assert len(orchestrator._buffer_history) == 0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        from app.training.per_orchestrator import PEROrchestrator

        orch = PEROrchestrator(
            buffer_stale_threshold_seconds=1800.0,
            max_history_per_buffer=50,
        )

        assert orch.buffer_stale_threshold_seconds == 1800.0
        assert orch.max_history_per_buffer == 50

    def test_on_buffer_rebuilt_new_buffer(self, orchestrator):
        """Test handling buffer rebuilt event for new buffer."""
        @dataclass
        class MockEvent:
            payload: dict

        event = MockEvent(payload={
            "buffer_path": "/data/per/buffer1",
            "buffer_size": 10000,
            "tree_depth": 14,
            "config": "sq8_2p",
        })

        orchestrator._on_buffer_rebuilt(event)

        assert "/data/per/buffer1" in orchestrator._buffers
        state = orchestrator._buffers["/data/per/buffer1"]
        assert state.buffer_size == 10000
        assert state.tree_depth == 14
        assert state.config_key == "sq8_2p"
        assert state.rebuild_count == 1
        assert state.last_rebuild_time > 0
        assert orchestrator._total_rebuilds == 1

    def test_on_buffer_rebuilt_existing_buffer(self, orchestrator):
        """Test handling buffer rebuilt event for existing buffer."""
        @dataclass
        class MockEvent:
            payload: dict

        # First rebuild
        event1 = MockEvent(payload={
            "buffer_path": "/data/per/buffer1",
            "buffer_size": 10000,
            "tree_depth": 14,
        })
        orchestrator._on_buffer_rebuilt(event1)

        # Second rebuild with different size
        event2 = MockEvent(payload={
            "buffer_path": "/data/per/buffer1",
            "buffer_size": 20000,
            "tree_depth": 15,
        })
        orchestrator._on_buffer_rebuilt(event2)

        state = orchestrator._buffers["/data/per/buffer1"]
        assert state.buffer_size == 20000
        assert state.tree_depth == 15
        assert state.rebuild_count == 2
        assert orchestrator._total_rebuilds == 2

    def test_on_priorities_updated_new_buffer(self, orchestrator):
        """Test handling priority update for new buffer (after rebuild)."""
        from app.training.per_orchestrator import PERBufferState

        @dataclass
        class MockEvent:
            payload: dict

        # Buffer must exist first (from rebuild)
        orchestrator._buffers["/data/per/buffer2"] = PERBufferState(
            buffer_path="/data/per/buffer2",
            buffer_size=5000,
        )

        event = MockEvent(payload={
            "buffer_path": "/data/per/buffer2",
            "priority_sum": 500.0,
            "min_priority": 0.01,
            "max_priority": 0.95,
            "samples_updated": 100,
        })

        orchestrator._on_priorities_updated(event)

        state = orchestrator._buffers["/data/per/buffer2"]
        assert state.priority_sum == 500.0
        assert state.min_priority == 0.01
        assert state.max_priority == 0.95
        assert state.priority_update_count == 1
        assert orchestrator._total_priority_updates == 1

    def test_on_priorities_updated_existing_buffer(self, orchestrator):
        """Test handling priority update for existing buffer."""
        from app.training.per_orchestrator import PERBufferState

        @dataclass
        class MockEvent:
            payload: dict

        buffer_path = "/data/per/buffer3"

        # Buffer must exist first (from rebuild)
        orchestrator._buffers[buffer_path] = PERBufferState(
            buffer_path=buffer_path,
            buffer_size=10000,
        )

        # Multiple updates
        for i in range(10):
            event = MockEvent(payload={
                "buffer_path": buffer_path,
                "priority_sum": float(i * 100),
                "min_priority": 0.01,
                "max_priority": 0.99,
                "samples_updated": 50,
            })
            orchestrator._on_priorities_updated(event)

        state = orchestrator._buffers[buffer_path]
        assert state.priority_update_count == 10
        assert state.priority_sum == 900.0  # Last update
        assert orchestrator._total_priority_updates == 10

    def test_add_to_history(self, orchestrator):
        """Test adding entries to buffer history."""
        orchestrator._add_to_history("/buffer/1", "rebuild", {"size": 1000})
        orchestrator._add_to_history("/buffer/1", "rebuild", {"size": 2000})
        orchestrator._add_to_history("/buffer/2", "rebuild", {"size": 500})

        assert len(orchestrator._buffer_history["/buffer/1"]) == 2
        assert len(orchestrator._buffer_history["/buffer/2"]) == 1
        assert orchestrator._buffer_history["/buffer/1"][0]["size"] == 1000

    def test_history_trimming(self):
        """Test that history is trimmed to max_history_per_buffer."""
        from app.training.per_orchestrator import PEROrchestrator

        orch = PEROrchestrator(max_history_per_buffer=5)

        for i in range(10):
            orch._add_to_history("/buffer/test", "event", {"value": i})

        assert len(orch._buffer_history["/buffer/test"]) == 5
        # Should keep most recent 5 entries
        values = [e["value"] for e in orch._buffer_history["/buffer/test"]]
        assert values == [5, 6, 7, 8, 9]

    def test_get_buffer_state(self, orchestrator):
        """Test getting state for a specific buffer."""
        from app.training.per_orchestrator import PERBufferState

        orchestrator._buffers["/test/buffer"] = PERBufferState(
            buffer_path="/test/buffer",
            buffer_size=5000,
        )

        state = orchestrator.get_buffer_state("/test/buffer")
        assert state is not None
        assert state.buffer_size == 5000

        # Non-existent buffer
        assert orchestrator.get_buffer_state("/nonexistent") is None

    def test_get_active_buffers(self, orchestrator):
        """Test getting active (non-stale) buffers."""
        from app.training.per_orchestrator import PERBufferState

        now = time.time()

        # Active buffer (recent activity)
        orchestrator._buffers["/active/1"] = PERBufferState(
            buffer_path="/active/1",
            buffer_size=1000,
            last_rebuild_time=now - 100,  # 100 seconds ago
        )

        # Stale buffer (old activity)
        orchestrator._buffers["/stale/1"] = PERBufferState(
            buffer_path="/stale/1",
            buffer_size=2000,
            last_rebuild_time=now - 7200,  # 2 hours ago
        )

        # Another active buffer (recent priority update)
        orchestrator._buffers["/active/2"] = PERBufferState(
            buffer_path="/active/2",
            buffer_size=1500,
            last_priority_update_time=now - 500,  # 500 seconds ago
        )

        active = orchestrator.get_active_buffers()
        assert len(active) == 2
        paths = [s.buffer_path for s in active]
        assert "/active/1" in paths
        assert "/active/2" in paths
        assert "/stale/1" not in paths

    def test_get_buffer_history_all(self, orchestrator):
        """Test getting history for all buffers."""
        orchestrator._add_to_history("/b1", "event", {"a": 1})
        orchestrator._add_to_history("/b2", "event", {"b": 2})

        history = orchestrator.get_buffer_history()
        assert "/b1" in history
        assert "/b2" in history

    def test_get_buffer_history_specific(self, orchestrator):
        """Test getting history for a specific buffer."""
        orchestrator._add_to_history("/b1", "event", {"a": 1})
        orchestrator._add_to_history("/b2", "event", {"b": 2})

        history = orchestrator.get_buffer_history("/b1")
        assert "/b1" in history
        assert "/b2" not in history

    def test_get_buffer_history_nonexistent(self, orchestrator):
        """Test getting history for nonexistent buffer."""
        history = orchestrator.get_buffer_history("/nonexistent")
        assert "/nonexistent" in history
        assert history["/nonexistent"] == []

    def test_get_stats(self, orchestrator):
        """Test getting aggregate statistics."""
        from app.training.per_orchestrator import PERBufferState

        now = time.time()

        # Add some buffers
        orchestrator._buffers["/buf/1"] = PERBufferState(
            buffer_path="/buf/1",
            buffer_size=10000,
            last_rebuild_time=now - 100,
        )
        orchestrator._buffers["/buf/2"] = PERBufferState(
            buffer_path="/buf/2",
            buffer_size=20000,
            last_priority_update_time=now - 200,
        )
        orchestrator._total_rebuilds = 5
        orchestrator._total_priority_updates = 50

        stats = orchestrator.get_stats()

        assert stats.total_buffers_tracked == 2
        assert stats.active_buffers == 2
        assert stats.total_samples == 30000
        assert stats.avg_buffer_size == 15000.0
        assert stats.total_rebuilds == 5
        assert stats.total_priority_updates == 50
        assert stats.last_activity_time > 0

    def test_get_stats_no_buffers(self, orchestrator):
        """Test getting stats with no buffers."""
        stats = orchestrator.get_stats()

        assert stats.total_buffers_tracked == 0
        assert stats.active_buffers == 0
        assert stats.total_samples == 0
        assert stats.avg_buffer_size == 0.0

    def test_get_status(self, orchestrator):
        """Test getting orchestrator status."""
        from app.training.per_orchestrator import PERBufferState

        now = time.time()
        orchestrator._buffers["/buf/1"] = PERBufferState(
            buffer_path="/buf/1",
            buffer_size=5000,
            last_rebuild_time=now,
        )
        orchestrator._total_rebuilds = 3
        orchestrator._total_priority_updates = 25

        status = orchestrator.get_status()

        assert status["subscribed"] is False
        assert status["total_buffers_tracked"] == 1
        assert status["active_buffers"] == 1
        assert status["total_samples"] == 5000
        assert status["total_rebuilds"] == 3
        assert status["total_priority_updates"] == 25
        assert "/buf/1" in status["buffer_paths"]

    def test_subscribe_to_events_success(self, orchestrator):
        """Test successful event subscription."""
        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            result = orchestrator.subscribe_to_events()

        assert result is True
        assert orchestrator._subscribed is True
        assert mock_bus.subscribe.call_count == 2

    def test_subscribe_to_events_already_subscribed(self, orchestrator):
        """Test subscription when already subscribed."""
        orchestrator._subscribed = True

        result = orchestrator.subscribe_to_events()

        assert result is True

    def test_subscribe_to_events_failure(self, orchestrator):
        """Test subscription failure."""
        with patch("app.distributed.data_events.get_event_bus", side_effect=Exception("No bus")):
            result = orchestrator.subscribe_to_events()

        assert result is False
        assert orchestrator._subscribed is False

    def test_unsubscribe(self, orchestrator):
        """Test event unsubscription."""
        mock_bus = MagicMock()
        orchestrator._subscribed = True

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orchestrator.unsubscribe()

        assert orchestrator._subscribed is False
        assert mock_bus.unsubscribe.call_count == 2

    def test_unsubscribe_not_subscribed(self, orchestrator):
        """Test unsubscribe when not subscribed."""
        orchestrator.unsubscribe()  # Should not raise
        assert orchestrator._subscribed is False

    def test_emit_stats_event(self, orchestrator):
        """Test emitting stats event."""
        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orchestrator._emit_stats_event()

        assert mock_bus.publish_sync.call_count == 1

    def test_emit_stats_event_failure(self, orchestrator):
        """Test stats event emission failure (should not raise)."""
        with patch("app.distributed.data_events.get_event_bus", side_effect=Exception("Error")):
            orchestrator._emit_stats_event()  # Should not raise


# =============================================================================
# Test Module Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from app.training.per_orchestrator import reset_per_orchestrator

        reset_per_orchestrator()
        yield
        reset_per_orchestrator()

    def test_wire_per_events(self):
        """Test wiring PER events."""
        from app.training.per_orchestrator import wire_per_events, get_per_orchestrator

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_per_events()

        assert orch is not None
        assert get_per_orchestrator() is orch
        assert orch._subscribed is True

    def test_wire_per_events_custom_threshold(self):
        """Test wiring with custom stale threshold."""
        from app.training.per_orchestrator import wire_per_events

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_per_events(buffer_stale_threshold_seconds=1800.0)

        assert orch.buffer_stale_threshold_seconds == 1800.0

    def test_wire_per_events_singleton(self):
        """Test that wire_per_events returns same instance."""
        from app.training.per_orchestrator import wire_per_events

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch1 = wire_per_events()
            orch2 = wire_per_events()

        assert orch1 is orch2

    def test_get_per_orchestrator_none(self):
        """Test get_per_orchestrator returns None when not wired."""
        from app.training.per_orchestrator import get_per_orchestrator

        assert get_per_orchestrator() is None

    def test_get_per_orchestrator_after_wire(self):
        """Test get_per_orchestrator returns instance after wiring."""
        from app.training.per_orchestrator import wire_per_events, get_per_orchestrator

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            wire_per_events()

        assert get_per_orchestrator() is not None

    def test_reset_per_orchestrator(self):
        """Test resetting the orchestrator singleton."""
        from app.training.per_orchestrator import (
            wire_per_events,
            get_per_orchestrator,
            reset_per_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            wire_per_events()

        assert get_per_orchestrator() is not None

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            reset_per_orchestrator()

        assert get_per_orchestrator() is None

    def test_reset_per_orchestrator_unsubscribes(self):
        """Test that reset unsubscribes from events."""
        from app.training.per_orchestrator import (
            wire_per_events,
            reset_per_orchestrator,
        )

        mock_bus = MagicMock()

        with patch("app.distributed.data_events.get_event_bus", return_value=mock_bus):
            orch = wire_per_events()
            assert orch._subscribed is True

            reset_per_orchestrator()

        # Should have called unsubscribe
        assert mock_bus.unsubscribe.call_count >= 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestPERIntegration:
    """Integration tests for PER orchestrator."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset singleton."""
        from app.training.per_orchestrator import reset_per_orchestrator

        reset_per_orchestrator()
        yield
        reset_per_orchestrator()

    def test_full_buffer_lifecycle(self):
        """Test full buffer lifecycle from rebuild to priority updates."""
        from app.training.per_orchestrator import PEROrchestrator

        @dataclass
        class MockEvent:
            payload: dict

        orch = PEROrchestrator()

        # Initial rebuild
        rebuild_event = MockEvent(payload={
            "buffer_path": "/data/per/main",
            "buffer_size": 100000,
            "tree_depth": 17,
            "config": "sq19_2p",
        })
        orch._on_buffer_rebuilt(rebuild_event)

        # Multiple priority updates
        for i in range(20):
            update_event = MockEvent(payload={
                "buffer_path": "/data/per/main",
                "priority_sum": 1000.0 + i * 10,
                "min_priority": 0.001,
                "max_priority": 0.999,
                "samples_updated": 500,
            })
            orch._on_priorities_updated(update_event)

        # Verify state
        state = orch.get_buffer_state("/data/per/main")
        assert state is not None
        assert state.rebuild_count == 1
        assert state.priority_update_count == 20
        assert state.buffer_size == 100000
        assert state.config_key == "sq19_2p"

        # Verify stats
        stats = orch.get_stats()
        assert stats.total_buffers_tracked == 1
        assert stats.total_rebuilds == 1
        assert stats.total_priority_updates == 20
        assert stats.total_samples == 100000

    def test_multiple_buffers(self):
        """Test tracking multiple buffers."""
        from app.training.per_orchestrator import PEROrchestrator

        @dataclass
        class MockEvent:
            payload: dict

        orch = PEROrchestrator()

        # Create 3 buffers
        for i in range(3):
            event = MockEvent(payload={
                "buffer_path": f"/data/per/buffer{i}",
                "buffer_size": (i + 1) * 10000,
                "tree_depth": 14,
            })
            orch._on_buffer_rebuilt(event)

        stats = orch.get_stats()
        assert stats.total_buffers_tracked == 3
        assert stats.total_samples == 60000  # 10000 + 20000 + 30000
        assert stats.avg_buffer_size == 20000.0

    def test_history_retention_on_priority_updates(self):
        """Test that history is only updated every 10 priority updates."""
        from app.training.per_orchestrator import PEROrchestrator, PERBufferState

        @dataclass
        class MockEvent:
            payload: dict

        orch = PEROrchestrator()

        buffer_path = "/data/per/test"

        # Buffer must exist first (from rebuild)
        orch._buffers[buffer_path] = PERBufferState(
            buffer_path=buffer_path,
            buffer_size=10000,
        )

        # 25 priority updates
        for i in range(25):
            event = MockEvent(payload={
                "buffer_path": buffer_path,
                "priority_sum": float(i * 100),
                "min_priority": 0.01,
                "max_priority": 0.99,
                "samples_updated": 50,
            })
            orch._on_priorities_updated(event)

        # Should have history entries at 10 and 20 (2 entries)
        history = orch.get_buffer_history(buffer_path)
        # Only logged at updates 10, 20 (every 10th update)
        assert len(history[buffer_path]) == 2
