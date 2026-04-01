"""Tests for ResourceMonitoringCoordinator.

Tests cover:
- Resource state tracking
- Backpressure activation/release
- Threshold checking
- Statistics calculation
- Callbacks
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from app.coordination.resource_monitoring_coordinator import (
    BackpressureEvent,
    BackpressureLevel,
    NodeResourceState,
    ResourceMonitoringCoordinator,
    ResourceStats,
    ResourceType,
    get_cluster_capacity,
    get_resource_coordinator,
    is_cluster_under_backpressure,
    update_node_resources,
    wire_resource_events,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a fresh ResourceMonitoringCoordinator."""
    return ResourceMonitoringCoordinator()


@pytest.fixture
def mock_event():
    """Create a mock event with payload."""
    @dataclass
    class MockEvent:
        payload: dict[str, Any]
    return MockEvent


# =============================================================================
# NodeResourceState Tests
# =============================================================================


class TestNodeResourceState:
    """Test NodeResourceState dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        state = NodeResourceState(node_id="test-node")
        assert state.node_id == "test-node"
        assert state.gpu_utilization == 0.0
        assert state.backpressure_active is False
        assert state.backpressure_level == BackpressureLevel.NONE

    def test_is_stale_recent(self):
        """Test is_stale for recent update."""
        state = NodeResourceState(
            node_id="test",
            last_update_time=time.time(),
        )
        assert not state.is_stale

    def test_is_stale_old(self):
        """Test is_stale for old update."""
        state = NodeResourceState(
            node_id="test",
            last_update_time=time.time() - 120,
        )
        assert state.is_stale

    def test_available_capacity_percent(self):
        """Test available_capacity_percent calculation."""
        state = NodeResourceState(
            node_id="test",
            task_slots_available=3,
            task_slots_total=10,
        )
        assert state.available_capacity_percent == 30.0

    def test_available_capacity_percent_zero_total(self):
        """Test available_capacity_percent with zero total."""
        state = NodeResourceState(
            node_id="test",
            task_slots_available=0,
            task_slots_total=0,
        )
        assert state.available_capacity_percent == 0.0


# =============================================================================
# BackpressureEvent Tests
# =============================================================================


class TestBackpressureEvent:
    """Test BackpressureEvent dataclass."""

    def test_activation_event(self):
        """Test activation event creation."""
        event = BackpressureEvent(
            node_id="test-node",
            activated=True,
            level=BackpressureLevel.HIGH,
            reason="GPU overloaded",
        )
        assert event.activated is True
        assert event.level == BackpressureLevel.HIGH

    def test_release_event(self):
        """Test release event creation."""
        event = BackpressureEvent(
            node_id="test-node",
            activated=False,
            level=BackpressureLevel.NONE,
            reason="released",
            duration=120.0,
        )
        assert event.activated is False
        assert event.duration == 120.0


# =============================================================================
# ResourceMonitoringCoordinator Initialization Tests
# =============================================================================


class TestResourceMonitoringCoordinatorInit:
    """Test ResourceMonitoringCoordinator initialization."""

    def test_init_default_values(self, coordinator):
        """Test default initialization."""
        assert coordinator.backpressure_gpu_threshold == 70.0
        assert coordinator.backpressure_memory_threshold == 70.0
        assert coordinator.backpressure_disk_threshold == 75.0

    def test_init_custom_values(self):
        """Test custom initialization."""
        coord = ResourceMonitoringCoordinator(
            backpressure_gpu_threshold=80.0,
            backpressure_memory_threshold=75.0,
            max_backpressure_history=100,
        )
        assert coord.backpressure_gpu_threshold == 80.0
        assert coord.backpressure_memory_threshold == 75.0
        assert coord.max_backpressure_history == 100


class TestResourceMonitoringCoordinatorSubscriptions:
    """Tests for event subscription wiring."""

    def test_subscribe_to_events_uses_router_helper(self, coordinator):
        """Should subscribe resource events through the unified helper."""
        from app.coordination.event_router import DataEventType

        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            result = coordinator.subscribe_to_events()

        assert result is True
        assert coordinator._subscribed is True
        subscribed = [(call.args[0], call.args[1]) for call in mock_subscribe.call_args_list]
        assert (DataEventType.CLUSTER_CAPACITY_CHANGED, coordinator._on_cluster_capacity_changed) in subscribed
        assert (DataEventType.NODE_CAPACITY_UPDATED, coordinator._on_node_capacity_updated) in subscribed
        assert (DataEventType.BACKPRESSURE_ACTIVATED, coordinator._on_backpressure_activated) in subscribed
        assert (DataEventType.RESOURCE_CONSTRAINT, coordinator._on_resource_constraint) in subscribed
        assert (DataEventType.JOB_PREEMPTED, coordinator._on_job_preempted) in subscribed


# =============================================================================
# Resource Update Tests
# =============================================================================


class TestResourceUpdates:
    """Test resource update functionality."""

    def test_update_node_resources_creates_node(self, coordinator):
        """Test update creates node if not exists."""
        state = coordinator.update_node_resources(
            "new-node",
            gpu_utilization=50.0,
            cpu_utilization=30.0,
        )

        assert state.node_id == "new-node"
        assert state.gpu_utilization == 50.0
        assert state.cpu_utilization == 30.0

    def test_update_node_resources_updates_existing(self, coordinator):
        """Test update updates existing node."""
        coordinator.update_node_resources("test-node", gpu_utilization=30.0)
        coordinator.update_node_resources("test-node", gpu_utilization=60.0)

        state = coordinator.get_node_state("test-node")
        assert state.gpu_utilization == 60.0

    def test_update_node_resources_partial(self, coordinator):
        """Test partial update preserves other values."""
        coordinator.update_node_resources(
            "test-node",
            gpu_utilization=50.0,
            cpu_utilization=30.0,
        )
        coordinator.update_node_resources(
            "test-node",
            memory_used_percent=70.0,
        )

        state = coordinator.get_node_state("test-node")
        assert state.gpu_utilization == 50.0  # Preserved
        assert state.memory_used_percent == 70.0  # Updated


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestBackpressure:
    """Test backpressure detection and management."""

    def test_threshold_violation_activates_backpressure(self, coordinator):
        """Test threshold violation activates backpressure."""
        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources(
                "test-node",
                gpu_utilization=95.0,  # Above 90% threshold
            )

            state = coordinator.get_node_state("test-node")
            assert state.backpressure_active is True

    def test_threshold_clear_releases_backpressure(self, coordinator):
        """Test clearing threshold releases backpressure."""
        with patch.object(coordinator, "_emit_backpressure_event"):
            # First activate
            coordinator.update_node_resources("test-node", gpu_utilization=95.0)
            assert coordinator.is_backpressure_active("test-node")

            # Then clear
            coordinator.update_node_resources("test-node", gpu_utilization=50.0)
            assert not coordinator.is_backpressure_active("test-node")

    def test_is_backpressure_active_node(self, coordinator):
        """Test is_backpressure_active for specific node."""
        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources("node-1", gpu_utilization=95.0)
            coordinator.update_node_resources("node-2", gpu_utilization=50.0)

            assert coordinator.is_backpressure_active("node-1")
            assert not coordinator.is_backpressure_active("node-2")

    def test_is_backpressure_active_cluster(self, coordinator):
        """Test is_backpressure_active for cluster-wide."""
        assert not coordinator.is_backpressure_active()

        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources("node-1", gpu_utilization=95.0)

            assert coordinator.is_backpressure_active()

    def test_get_backpressure_level_node(self, coordinator):
        """Test get_backpressure_level for specific node."""
        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources("test-node", gpu_utilization=96.0)

            level = coordinator.get_backpressure_level("test-node")
            assert level == BackpressureLevel.CRITICAL

    def test_determine_backpressure_level(self, coordinator):
        """Test backpressure level determination."""
        state = NodeResourceState(
            node_id="test",
            gpu_utilization=96.0,
        )
        level = coordinator._determine_backpressure_level(state)
        assert level == BackpressureLevel.CRITICAL

        state.gpu_utilization = 92.0
        level = coordinator._determine_backpressure_level(state)
        assert level == BackpressureLevel.HIGH

        state.gpu_utilization = 87.0
        level = coordinator._determine_backpressure_level(state)
        assert level == BackpressureLevel.MEDIUM

        state.gpu_utilization = 82.0
        level = coordinator._determine_backpressure_level(state)
        assert level == BackpressureLevel.LOW


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Test event handler methods."""

    @pytest.mark.asyncio
    async def test_on_node_capacity_updated(self, coordinator, mock_event):
        """Test NODE_CAPACITY_UPDATED event handling."""
        event = mock_event(payload={
            "node_id": "test-node",
            "gpu_utilization": 75.0,
            "cpu_utilization": 50.0,
            "task_slots_available": 5,
            "task_slots_total": 10,
        })

        await coordinator._on_node_capacity_updated(event)

        state = coordinator.get_node_state("test-node")
        assert state is not None
        assert state.gpu_utilization == 75.0
        assert state.task_slots_available == 5

    @pytest.mark.asyncio
    async def test_on_backpressure_activated(self, coordinator, mock_event):
        """Test BACKPRESSURE_ACTIVATED event handling."""
        event = mock_event(payload={
            "node_id": "test-node",
            "level": "high",
            "reason": "GPU overloaded",
        })

        await coordinator._on_backpressure_activated(event)

        assert coordinator.is_backpressure_active("test-node")
        history = coordinator.get_backpressure_history()
        assert len(history) == 1
        assert history[0].level == BackpressureLevel.HIGH

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self, coordinator, mock_event):
        """Test BACKPRESSURE_RELEASED event handling."""
        # First activate
        activate_event = mock_event(payload={
            "node_id": "test-node",
            "level": "high",
            "reason": "test",
        })
        await coordinator._on_backpressure_activated(activate_event)

        # Then release
        release_event = mock_event(payload={
            "node_id": "test-node",
        })
        await coordinator._on_backpressure_released(release_event)

        assert not coordinator.is_backpressure_active("test-node")

    @pytest.mark.asyncio
    async def test_on_resource_constraint(self, coordinator, mock_event):
        """Test RESOURCE_CONSTRAINT event handling."""
        # First create the node
        coordinator.update_node_resources("test-node", gpu_utilization=50.0)

        event = mock_event(payload={
            "node_id": "test-node",
            "constraint_type": "gpu_memory",
            "message": "OOM risk",
        })

        await coordinator._on_resource_constraint(event)

        state = coordinator.get_node_state("test-node")
        assert "gpu_memory" in state.constraints


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Test callback functionality."""

    def test_on_backpressure_change_callback(self, coordinator):
        """Test backpressure change callback is called."""
        callbacks_called = []
        coordinator.on_backpressure_change(
            lambda node, active, level: callbacks_called.append((node, active, level))
        )

        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources("test-node", gpu_utilization=95.0)

        assert len(callbacks_called) == 1
        assert callbacks_called[0][0] == "test-node"
        assert callbacks_called[0][1] is True

    def test_on_constraint_callback(self, coordinator):
        """Test constraint callback is called."""
        callbacks_called = []
        coordinator.on_constraint(
            lambda node, constraint: callbacks_called.append((node, constraint))
        )

        coordinator.update_node_resources("test-node", gpu_utilization=50.0)
        coordinator._nodes["test-node"].constraints.append("test_constraint")

        # Manually trigger callback (normally done by event)
        for callback in coordinator._constraint_callbacks:
            callback("test-node", "test_constraint")

        assert len(callbacks_called) == 1


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Test query methods."""

    def test_get_all_nodes(self, coordinator):
        """Test get_all_nodes returns all nodes."""
        coordinator.update_node_resources("node-1", gpu_utilization=50.0)
        coordinator.update_node_resources("node-2", gpu_utilization=60.0)

        nodes = coordinator.get_all_nodes()
        assert len(nodes) == 2

    def test_get_constrained_nodes(self, coordinator):
        """Test get_constrained_nodes filters correctly."""
        with patch.object(coordinator, "_emit_backpressure_event"):
            coordinator.update_node_resources("constrained", gpu_utilization=95.0)
            coordinator.update_node_resources("normal", gpu_utilization=50.0)

            constrained = coordinator.get_constrained_nodes()
            node_ids = [n.node_id for n in constrained]

            assert "constrained" in node_ids
            assert "normal" not in node_ids


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistics calculation."""

    def test_get_stats_empty(self, coordinator):
        """Test get_stats with no nodes."""
        stats = coordinator.get_stats()
        assert stats.total_nodes == 0
        assert stats.avg_gpu_utilization == 0.0

    def test_get_stats_with_nodes(self, coordinator):
        """Test get_stats calculates correctly."""
        coordinator.update_node_resources("node-1", gpu_utilization=60.0)
        coordinator.update_node_resources("node-2", gpu_utilization=80.0)

        stats = coordinator.get_stats()

        assert stats.total_nodes == 2
        assert stats.avg_gpu_utilization == 70.0

    def test_get_status(self, coordinator):
        """Test get_status returns proper structure."""
        coordinator.update_node_resources("test-node", gpu_utilization=50.0)

        status = coordinator.get_status()

        assert "total_nodes" in status
        assert "avg_gpu_utilization" in status
        assert "backpressure_active" in status
        assert "subscribed" in status


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_resource_coordinator_returns_singleton(self):
        """Test get_resource_coordinator returns same instance."""
        import app.coordination.resource_monitoring_coordinator as rmc
        rmc._resource_coordinator = None

        coord1 = get_resource_coordinator()
        coord2 = get_resource_coordinator()

        assert coord1 is coord2

    def test_convenience_functions(self):
        """Test convenience functions work."""
        import app.coordination.resource_monitoring_coordinator as rmc
        rmc._resource_coordinator = None

        coord = get_resource_coordinator()
        coord.update_node_resources("test", task_slots_available=5, task_slots_total=10)

        capacity = get_cluster_capacity()
        assert capacity["total"] == 10
        assert capacity["available"] == 5


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_resource_type_values(self):
        """Test ResourceType enum values."""
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"

    def test_backpressure_level_values(self):
        """Test BackpressureLevel enum values."""
        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.LOW.value == "low"
        assert BackpressureLevel.MEDIUM.value == "medium"
        assert BackpressureLevel.HIGH.value == "high"
        assert BackpressureLevel.CRITICAL.value == "critical"
