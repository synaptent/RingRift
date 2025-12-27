"""Unit tests for node_status.py - canonical node status definitions.

Tests the consolidated node status module:
- NodeHealthState enum
- NodeMonitoringStatus dataclass
- Properties (is_healthy, is_available, gpu_memory_used_percent, is_gpu_node)
- Serialization (to_dict, from_dict)
- Backward-compatible aliases

December 2025: Created for Phase 6 consolidation testing.
"""

from __future__ import annotations

from datetime import datetime

import pytest


# =============================================================================
# Test NodeHealthState enum
# =============================================================================


class TestNodeHealthState:
    """Test NodeHealthState enum values and behavior."""

    def test_all_states_exist(self):
        """Verify all expected health states exist."""
        from app.coordination.node_status import NodeHealthState

        expected = [
            "HEALTHY",
            "DEGRADED",
            "UNHEALTHY",
            "EVICTED",
            "UNKNOWN",
            "OFFLINE",
            "PROVIDER_DOWN",
            "RETIRED",
        ]
        for state_name in expected:
            assert hasattr(NodeHealthState, state_name)

    def test_state_values_are_lowercase(self):
        """Verify enum values are lowercase strings."""
        from app.coordination.node_status import NodeHealthState

        for state in NodeHealthState:
            assert state.value == state.name.lower()

    def test_state_is_string_enum(self):
        """Verify NodeHealthState inherits from str."""
        from app.coordination.node_status import NodeHealthState

        # String enum allows direct comparison
        assert NodeHealthState.HEALTHY == "healthy"
        assert NodeHealthState.OFFLINE == "offline"


# =============================================================================
# Test NodeMonitoringStatus creation
# =============================================================================


class TestNodeMonitoringStatusCreation:
    """Test NodeMonitoringStatus dataclass creation."""

    def test_minimal_creation(self):
        """Verify creation with only required field."""
        from app.coordination.node_status import NodeMonitoringStatus

        status = NodeMonitoringStatus(node_id="test-node")
        assert status.node_id == "test-node"
        assert status.host == ""  # Default

    def test_full_creation(self):
        """Verify creation with all fields."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        now = datetime.now()
        status = NodeMonitoringStatus(
            node_id="gpu-node-1",
            host="192.168.1.100",
            port=8770,
            health_state=NodeHealthState.HEALTHY,
            is_reachable=True,
            response_time_ms=50.0,
            consecutive_failures=0,
            last_check_time=now,
            last_success_time=now,
            gpu_utilization=75.0,
            gpu_memory_total_gb=80.0,
            gpu_memory_used_gb=60.0,
            gpu_memory_free_gb=20.0,
            gpu_type="H100",
            provider="nebius",
            role="worker",
            is_coordinator=False,
            is_voter=True,
            running_jobs=["job-1", "job-2"],
            pending_jobs=3,
            completed_jobs=100,
            metadata={"region": "eu-central"},
        )

        assert status.node_id == "gpu-node-1"
        assert status.host == "192.168.1.100"
        assert status.port == 8770
        assert status.health_state == NodeHealthState.HEALTHY
        assert status.gpu_memory_total_gb == 80.0
        assert len(status.running_jobs) == 2

    def test_default_values(self):
        """Verify all default values are correct."""
        from app.coordination.node_status import (
            NodeHealthState,
            NodeMonitoringStatus,
            P2P_DEFAULT_PORT,
        )

        status = NodeMonitoringStatus(node_id="test")

        assert status.host == ""
        assert status.port == P2P_DEFAULT_PORT
        assert status.health_state == NodeHealthState.UNKNOWN
        assert status.is_reachable is False
        assert status.response_time_ms == 0.0
        assert status.consecutive_failures == 0
        assert status.gpu_utilization == 0.0
        assert status.provider == "unknown"
        assert status.role == "worker"
        assert status.running_jobs == []


# =============================================================================
# Test properties
# =============================================================================


class TestNodeMonitoringStatusProperties:
    """Test computed properties."""

    def test_is_healthy_for_healthy_state(self):
        """Verify is_healthy returns True for HEALTHY state."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            health_state=NodeHealthState.HEALTHY,
        )
        assert status.is_healthy is True

    def test_is_healthy_for_degraded_state(self):
        """Verify is_healthy returns True for DEGRADED state."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            health_state=NodeHealthState.DEGRADED,
        )
        assert status.is_healthy is True

    def test_is_healthy_for_unhealthy_state(self):
        """Verify is_healthy returns False for UNHEALTHY state."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            health_state=NodeHealthState.UNHEALTHY,
        )
        assert status.is_healthy is False

    def test_is_available_when_reachable_and_healthy(self):
        """Verify is_available returns True when reachable and healthy."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            is_reachable=True,
            health_state=NodeHealthState.HEALTHY,
        )
        assert status.is_available is True

    def test_is_available_when_evicted(self):
        """Verify is_available returns False when evicted."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            is_reachable=True,
            health_state=NodeHealthState.EVICTED,
        )
        assert status.is_available is False

    def test_is_available_when_not_reachable(self):
        """Verify is_available returns False when not reachable."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            is_reachable=False,
            health_state=NodeHealthState.HEALTHY,
        )
        assert status.is_available is False

    def test_gpu_memory_used_percent_calculation(self):
        """Verify gpu_memory_used_percent calculates correctly."""
        from app.coordination.node_status import NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            gpu_memory_total_gb=80.0,
            gpu_memory_used_gb=60.0,
        )
        assert status.gpu_memory_used_percent == 75.0

    def test_gpu_memory_used_percent_zero_total(self):
        """Verify gpu_memory_used_percent handles zero total gracefully."""
        from app.coordination.node_status import NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            gpu_memory_total_gb=0.0,
        )
        assert status.gpu_memory_used_percent == 0.0

    def test_is_gpu_node_true(self):
        """Verify is_gpu_node returns True when GPU present."""
        from app.coordination.node_status import NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            gpu_memory_total_gb=24.0,
        )
        assert status.is_gpu_node is True

    def test_is_gpu_node_false(self):
        """Verify is_gpu_node returns False when no GPU."""
        from app.coordination.node_status import NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            gpu_memory_total_gb=0.0,
        )
        assert status.is_gpu_node is False


# =============================================================================
# Test serialization
# =============================================================================


class TestNodeMonitoringStatusSerialization:
    """Test to_dict and from_dict methods."""

    def test_to_dict_includes_all_fields(self):
        """Verify to_dict includes all expected fields."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            host="localhost",
            health_state=NodeHealthState.HEALTHY,
        )
        d = status.to_dict()

        expected_keys = [
            "node_id",
            "host",
            "port",
            "health_state",
            "is_reachable",
            "response_time_ms",
            "consecutive_failures",
            "last_check_time",
            "last_success_time",
            "gpu_utilization",
            "gpu_memory_total_gb",
            "gpu_memory_used_gb",
            "gpu_memory_free_gb",
            "gpu_type",
            "provider",
            "role",
            "is_coordinator",
            "is_voter",
            "running_jobs",
            "pending_jobs",
            "completed_jobs",
            "metadata",
        ]
        for key in expected_keys:
            assert key in d

    def test_to_dict_serializes_health_state(self):
        """Verify health_state is serialized as string."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        status = NodeMonitoringStatus(
            node_id="test",
            health_state=NodeHealthState.DEGRADED,
        )
        d = status.to_dict()

        assert d["health_state"] == "degraded"

    def test_to_dict_serializes_datetime(self):
        """Verify datetime is serialized as ISO string."""
        from app.coordination.node_status import NodeMonitoringStatus

        now = datetime.now()
        status = NodeMonitoringStatus(
            node_id="test",
            last_check_time=now,
        )
        d = status.to_dict()

        assert d["last_check_time"] == now.isoformat()

    def test_from_dict_roundtrip(self):
        """Verify from_dict reverses to_dict."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        now = datetime.now()
        original = NodeMonitoringStatus(
            node_id="roundtrip-test",
            host="192.168.1.1",
            port=8770,
            health_state=NodeHealthState.HEALTHY,
            is_reachable=True,
            response_time_ms=25.5,
            gpu_memory_total_gb=80.0,
            provider="nebius",
            role="worker",
            running_jobs=["job-1"],
            metadata={"key": "value"},
        )

        d = original.to_dict()
        restored = NodeMonitoringStatus.from_dict(d)

        assert restored.node_id == original.node_id
        assert restored.host == original.host
        assert restored.port == original.port
        assert restored.health_state == original.health_state
        assert restored.is_reachable == original.is_reachable
        assert restored.gpu_memory_total_gb == original.gpu_memory_total_gb
        assert restored.running_jobs == original.running_jobs
        assert restored.metadata == original.metadata

    def test_from_dict_handles_unknown_health_state(self):
        """Verify from_dict handles invalid health state."""
        from app.coordination.node_status import NodeHealthState, NodeMonitoringStatus

        d = {
            "node_id": "test",
            "health_state": "invalid_state_xyz",
        }
        status = NodeMonitoringStatus.from_dict(d)

        assert status.health_state == NodeHealthState.UNKNOWN

    def test_from_dict_handles_missing_fields(self):
        """Verify from_dict handles missing optional fields."""
        from app.coordination.node_status import NodeMonitoringStatus

        d = {"node_id": "minimal"}
        status = NodeMonitoringStatus.from_dict(d)

        assert status.node_id == "minimal"
        assert status.host == ""
        assert status.running_jobs == []


# =============================================================================
# Test backward-compatible aliases
# =============================================================================


class TestBackwardCompatibleAliases:
    """Test backward-compatible class aliases."""

    def test_node_status_alias(self):
        """Verify NodeStatus is alias for NodeMonitoringStatus."""
        from app.coordination.node_status import NodeMonitoringStatus, NodeStatus

        assert NodeStatus is NodeMonitoringStatus

    def test_cluster_node_status_alias(self):
        """Verify ClusterNodeStatus is alias for NodeMonitoringStatus."""
        from app.coordination.node_status import ClusterNodeStatus, NodeMonitoringStatus

        assert ClusterNodeStatus is NodeMonitoringStatus


# =============================================================================
# Test factory function
# =============================================================================


class TestGetNodeStatus:
    """Test get_node_status factory function."""

    def test_basic_creation(self):
        """Verify factory creates status with node_id and host."""
        from app.coordination.node_status import get_node_status

        status = get_node_status("node-1", "192.168.1.1")
        assert status.node_id == "node-1"
        assert status.host == "192.168.1.1"

    def test_with_kwargs(self):
        """Verify factory accepts additional kwargs."""
        from app.coordination.node_status import NodeHealthState, get_node_status

        status = get_node_status(
            "node-1",
            "192.168.1.1",
            health_state=NodeHealthState.HEALTHY,
            gpu_memory_total_gb=48.0,
        )

        assert status.health_state == NodeHealthState.HEALTHY
        assert status.gpu_memory_total_gb == 48.0
