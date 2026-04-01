"""Tests for HealthCheckOrchestrator.

Tests cover:
- Node health state computation
- Cluster health summary
- Available/overloaded/underutilized node queries
- Retired node management
- Provider manager integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.health_check_orchestrator import (
    ClusterHealthSummary,
    HealthCheckOrchestrator,
    NodeHealthDetails,
    NodeHealthState,
    get_available_nodes,
    get_health_orchestrator,
)
from app.providers import Provider


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def orchestrator():
    """Create a HealthCheckOrchestrator with mocked providers."""
    with patch("app.coordination.health_check_orchestrator.LambdaManager"), \
         patch("app.coordination.health_check_orchestrator.VastManager"), \
         patch("app.coordination.health_check_orchestrator.HetznerManager"), \
         patch("app.coordination.health_check_orchestrator.AWSManager"), \
         patch("app.coordination.health_check_orchestrator.TailscaleManager"):
        return HealthCheckOrchestrator(check_interval=60.0)


# =============================================================================
# NodeHealthDetails Tests
# =============================================================================


class TestNodeHealthDetails:
    """Test NodeHealthDetails dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        details = NodeHealthDetails(
            node_id="test-node",
            provider=Provider.LAMBDA,
        )
        assert details.node_id == "test-node"
        assert details.state == NodeHealthState.OFFLINE
        assert not details.ssh_healthy
        assert not details.p2p_healthy

    def test_is_available_healthy(self):
        """Test is_available for healthy node."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            state=NodeHealthState.HEALTHY,
        )
        assert details.is_available()

    def test_is_available_degraded(self):
        """Test is_available for degraded node."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            state=NodeHealthState.DEGRADED,
        )
        assert details.is_available()

    def test_is_available_unhealthy(self):
        """Test is_available for unhealthy node."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            state=NodeHealthState.UNHEALTHY,
        )
        assert not details.is_available()

    def test_compute_state_healthy(self):
        """Test compute_state when all checks pass."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            ssh_healthy=True,
            p2p_healthy=True,
            tailscale_healthy=True,
        )
        assert details.compute_state() == NodeHealthState.HEALTHY

    def test_compute_state_degraded(self):
        """Test compute_state with one failing check."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            ssh_healthy=True,
            p2p_healthy=True,
            tailscale_healthy=False,
        )
        assert details.compute_state() == NodeHealthState.DEGRADED

    def test_compute_state_unhealthy(self):
        """Test compute_state with two failing checks."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            ssh_healthy=True,
            p2p_healthy=False,
            tailscale_healthy=False,
        )
        assert details.compute_state() == NodeHealthState.UNHEALTHY

    def test_compute_state_offline(self):
        """Test compute_state when SSH fails."""
        details = NodeHealthDetails(
            node_id="test",
            provider=Provider.LAMBDA,
            ssh_healthy=False,
            p2p_healthy=True,
            tailscale_healthy=True,
        )
        assert details.compute_state() == NodeHealthState.OFFLINE


# =============================================================================
# ClusterHealthSummary Tests
# =============================================================================


class TestClusterHealthSummary:
    """Test ClusterHealthSummary dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        summary = ClusterHealthSummary()
        assert summary.total_nodes == 0
        assert summary.healthy == 0
        assert summary.hourly_cost == 0.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        summary = ClusterHealthSummary(
            total_nodes=10,
            healthy=8,
            degraded=1,
            unhealthy=1,
            total_gpus=20,
            available_gpus=16,
        )

        d = summary.to_dict()

        assert d["total_nodes"] == 10
        assert d["healthy"] == 8
        assert d["total_gpus"] == 20
        assert "availability_percent" in d

    def test_availability_percent(self):
        """Test availability percent calculation."""
        summary = ClusterHealthSummary(
            total_nodes=10,
            healthy=6,
            degraded=2,
            unhealthy=1,
            offline=1,
        )

        d = summary.to_dict()
        # 8 available out of 10 = 80%
        assert d["availability_percent"] == 80.0


# =============================================================================
# HealthCheckOrchestrator Initialization Tests
# =============================================================================


class TestHealthCheckOrchestratorInit:
    """Test HealthCheckOrchestrator initialization."""

    def test_init_default_values(self, orchestrator):
        """Test default initialization."""
        assert orchestrator.check_interval == 60.0
        assert orchestrator.p2p_port == 8770
        assert orchestrator._running is False

    def test_init_custom_values(self):
        """Test custom initialization."""
        with patch("app.coordination.health_check_orchestrator.LambdaManager"), \
             patch("app.coordination.health_check_orchestrator.VastManager"), \
             patch("app.coordination.health_check_orchestrator.HetznerManager"), \
             patch("app.coordination.health_check_orchestrator.AWSManager"), \
             patch("app.coordination.health_check_orchestrator.TailscaleManager"):
            orch = HealthCheckOrchestrator(
                check_interval=120.0,
                p2p_port=9000,
            )
            assert orch.check_interval == 120.0
            assert orch.p2p_port == 9000

    def test_subscribe_to_events_uses_router_helper(self, orchestrator):
        """Should subscribe fast-failure events through the unified helper."""
        from app.distributed.data_events import DataEventType

        with patch("app.coordination.event_router.subscribe") as mock_subscribe:
            orchestrator._subscribe_to_events()

        subscribed = [(call.args[0], call.args[1]) for call in mock_subscribe.call_args_list]
        assert (DataEventType.P2P_NODE_DEAD, orchestrator._on_node_dead) in subscribed
        assert (DataEventType.P2P_NODES_DEAD, orchestrator._on_nodes_dead) in subscribed
        assert (DataEventType.HOST_OFFLINE, orchestrator._on_node_dead) in subscribed


# =============================================================================
# Node Query Tests
# =============================================================================


class TestNodeQueries:
    """Test node query methods."""

    def test_get_available_nodes_empty(self, orchestrator):
        """Test get_available_nodes with no nodes."""
        nodes = orchestrator.get_available_nodes()
        assert nodes == []

    def test_get_available_nodes_filters_correctly(self, orchestrator):
        """Test get_available_nodes filters by availability."""
        orchestrator.node_health = {
            "healthy-1": NodeHealthDetails(
                node_id="healthy-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "degraded-1": NodeHealthDetails(
                node_id="degraded-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.DEGRADED,
            ),
            "offline-1": NodeHealthDetails(
                node_id="offline-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.OFFLINE,
            ),
        }

        available = orchestrator.get_available_nodes()

        assert "healthy-1" in available
        assert "degraded-1" in available
        assert "offline-1" not in available

    def test_get_nodes_by_state(self, orchestrator):
        """Test get_nodes_by_state filters correctly."""
        orchestrator.node_health = {
            "healthy-1": NodeHealthDetails(
                node_id="healthy-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "healthy-2": NodeHealthDetails(
                node_id="healthy-2",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "offline-1": NodeHealthDetails(
                node_id="offline-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.OFFLINE,
            ),
        }

        healthy = orchestrator.get_nodes_by_state(NodeHealthState.HEALTHY)
        offline = orchestrator.get_nodes_by_state(NodeHealthState.OFFLINE)

        assert len(healthy) == 2
        assert len(offline) == 1

    def test_get_node_health(self, orchestrator):
        """Test get_node_health returns correct details."""
        orchestrator.node_health = {
            "test-node": NodeHealthDetails(
                node_id="test-node",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
        }

        details = orchestrator.get_node_health("test-node")
        assert details is not None
        assert details.node_id == "test-node"

        missing = orchestrator.get_node_health("nonexistent")
        assert missing is None


# =============================================================================
# Underutilized/Overloaded Node Tests
# =============================================================================


class TestUtilizationQueries:
    """Test utilization-based queries."""

    def test_get_underutilized_nodes(self, orchestrator):
        """Test get_underutilized_nodes finds low utilization."""
        mock_instance = MagicMock()
        mock_instance.gpu_count = 1

        orchestrator.node_health = {
            "underutilized": NodeHealthDetails(
                node_id="underutilized",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
                gpu_percent=10.0,
                instance=mock_instance,
            ),
            "busy": NodeHealthDetails(
                node_id="busy",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
                gpu_percent=80.0,
                instance=mock_instance,
            ),
        }

        underutilized = orchestrator.get_underutilized_nodes(gpu_threshold=20.0)

        assert "underutilized" in underutilized
        assert "busy" not in underutilized

    def test_get_overloaded_nodes(self, orchestrator):
        """Test get_overloaded_nodes finds high utilization."""
        orchestrator.node_health = {
            "overloaded": NodeHealthDetails(
                node_id="overloaded",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
                gpu_percent=98.0,
            ),
            "normal": NodeHealthDetails(
                node_id="normal",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
                gpu_percent=60.0,
            ),
        }

        overloaded = orchestrator.get_overloaded_nodes(gpu_threshold=95.0)

        assert "overloaded" in overloaded
        assert "normal" not in overloaded


# =============================================================================
# Retired Node Tests
# =============================================================================


class TestRetiredNodes:
    """Test retired node management."""

    def test_mark_retired(self, orchestrator):
        """Test marking a node as retired."""
        orchestrator.node_health = {
            "test-node": NodeHealthDetails(
                node_id="test-node",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
        }

        result = orchestrator.mark_retired("test-node")

        assert result is True
        assert orchestrator.node_health["test-node"].state == NodeHealthState.RETIRED

    def test_mark_retired_not_found(self, orchestrator):
        """Test marking non-existent node as retired."""
        result = orchestrator.mark_retired("nonexistent")
        assert result is False

    def test_unmark_retired(self, orchestrator):
        """Test unmarking a retired node."""
        orchestrator.node_health = {
            "test-node": NodeHealthDetails(
                node_id="test-node",
                provider=Provider.LAMBDA,
                state=NodeHealthState.RETIRED,
            ),
        }

        result = orchestrator.unmark_retired("test-node")

        assert result is True
        assert orchestrator.node_health["test-node"].state == NodeHealthState.OFFLINE

    def test_unmark_retired_not_retired(self, orchestrator):
        """Test unmarking a node that isn't retired."""
        orchestrator.node_health = {
            "test-node": NodeHealthDetails(
                node_id="test-node",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
        }

        result = orchestrator.unmark_retired("test-node")
        assert result is False


# =============================================================================
# Cluster Health Summary Tests
# =============================================================================


class TestClusterHealthSummaryCalculation:
    """Test cluster health summary calculation."""

    @pytest.mark.asyncio
    async def test_get_cluster_health_empty(self, orchestrator):
        """Test get_cluster_health with no nodes."""
        summary = await orchestrator.get_cluster_health()

        assert summary.total_nodes == 0
        assert summary.healthy == 0

    @pytest.mark.asyncio
    async def test_get_cluster_health_counts_states(self, orchestrator):
        """Test get_cluster_health counts states correctly."""
        orchestrator.node_health = {
            "healthy-1": NodeHealthDetails(
                node_id="healthy-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "healthy-2": NodeHealthDetails(
                node_id="healthy-2",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "degraded-1": NodeHealthDetails(
                node_id="degraded-1",
                provider=Provider.VAST,
                state=NodeHealthState.DEGRADED,
            ),
            "offline-1": NodeHealthDetails(
                node_id="offline-1",
                provider=Provider.HETZNER,
                state=NodeHealthState.OFFLINE,
            ),
        }

        summary = await orchestrator.get_cluster_health()

        assert summary.total_nodes == 4
        assert summary.healthy == 2
        assert summary.degraded == 1
        assert summary.offline == 1

    @pytest.mark.asyncio
    async def test_get_cluster_health_by_provider(self, orchestrator):
        """Test get_cluster_health groups by provider."""
        orchestrator.node_health = {
            "lambda-1": NodeHealthDetails(
                node_id="lambda-1",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "lambda-2": NodeHealthDetails(
                node_id="lambda-2",
                provider=Provider.LAMBDA,
                state=NodeHealthState.HEALTHY,
            ),
            "vast-1": NodeHealthDetails(
                node_id="vast-1",
                provider=Provider.VAST,
                state=NodeHealthState.HEALTHY,
            ),
        }

        summary = await orchestrator.get_cluster_health()

        assert "lambda" in summary.by_provider
        assert summary.by_provider["lambda"]["total"] == 2
        assert "vast" in summary.by_provider
        assert summary.by_provider["vast"]["total"] == 1


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonBehavior:
    """Test singleton behavior."""

    def test_get_health_orchestrator_returns_singleton(self):
        """Test get_health_orchestrator returns same instance."""
        import app.coordination.health_check_orchestrator as hco
        hco._health_orchestrator = None

        with patch("app.coordination.health_check_orchestrator.LambdaManager"), \
             patch("app.coordination.health_check_orchestrator.VastManager"), \
             patch("app.coordination.health_check_orchestrator.HetznerManager"), \
             patch("app.coordination.health_check_orchestrator.AWSManager"), \
             patch("app.coordination.health_check_orchestrator.TailscaleManager"):
            orch1 = get_health_orchestrator()
            orch2 = get_health_orchestrator()

            assert orch1 is orch2


# =============================================================================
# NodeHealthState Enum Tests
# =============================================================================


class TestNodeHealthState:
    """Test NodeHealthState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert NodeHealthState.HEALTHY.value == "healthy"
        assert NodeHealthState.DEGRADED.value == "degraded"
        assert NodeHealthState.UNHEALTHY.value == "unhealthy"
        assert NodeHealthState.OFFLINE.value == "offline"
        assert NodeHealthState.RETIRED.value == "retired"
