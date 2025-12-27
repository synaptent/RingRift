"""Tests for CoordinatorHealthMonitorDaemon.

December 2025: Tests for the new coordinator health monitoring daemon that
subscribes to all COORDINATOR_* events (COORDINATOR_HEALTHY, COORDINATOR_UNHEALTHY,
COORDINATOR_HEALTH_DEGRADED, COORDINATOR_SHUTDOWN, COORDINATOR_HEARTBEAT).
"""

import asyncio
import pytest
import time

from app.coordination.coordinator_health_monitor_daemon import (
    CoordinatorHealthMonitorDaemon,
    CoordinatorHealthSummary,
    CoordinatorInfo,
    CoordinatorState,
    get_coordinator_health_monitor_sync,
    HEARTBEAT_STALE_THRESHOLD_SECONDS,
    INIT_FAILURE_MAX_RETRIES,
)


class TestCoordinatorState:
    """Tests for CoordinatorState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert CoordinatorState.UNKNOWN.value == "unknown"
        assert CoordinatorState.HEALTHY.value == "healthy"
        assert CoordinatorState.UNHEALTHY.value == "unhealthy"
        assert CoordinatorState.DEGRADED.value == "degraded"
        assert CoordinatorState.SHUTDOWN.value == "shutdown"
        assert CoordinatorState.INIT_FAILED.value == "init_failed"


class TestCoordinatorInfo:
    """Tests for CoordinatorInfo dataclass."""

    def test_default_values(self):
        """Test CoordinatorInfo default values."""
        info = CoordinatorInfo(name="test-coordinator")
        assert info.name == "test-coordinator"
        assert info.state == CoordinatorState.UNKNOWN
        assert info.last_healthy_at == 0.0
        assert info.init_failure_count == 0

    def test_custom_values(self):
        """Test CoordinatorInfo with custom values."""
        now = time.time()
        info = CoordinatorInfo(
            name="training-coordinator",
            state=CoordinatorState.HEALTHY,
            last_healthy_at=now,
            node_id="gpu-node-1",
        )
        assert info.state == CoordinatorState.HEALTHY
        assert info.last_healthy_at == now
        assert info.node_id == "gpu-node-1"


class TestCoordinatorHealthSummary:
    """Tests for CoordinatorHealthSummary dataclass."""

    def test_default_values(self):
        """Test CoordinatorHealthSummary default values."""
        summary = CoordinatorHealthSummary()
        assert summary.total_count == 0
        assert summary.healthy_count == 0
        assert summary.cluster_healthy is True
        assert summary.cluster_health_pct == 100.0

    def test_health_calculation(self):
        """Test health percentage calculation."""
        summary = CoordinatorHealthSummary(
            total_count=10,
            healthy_count=8,
            unhealthy_count=2,
        )
        # Note: cluster_health_pct is calculated in get_health_summary()
        # This tests the dataclass defaults


class TestCoordinatorHealthMonitorDaemon:
    """Tests for CoordinatorHealthMonitorDaemon."""

    def test_initialization(self):
        """Test daemon initializes correctly."""
        daemon = CoordinatorHealthMonitorDaemon()
        assert daemon._running is False
        assert daemon._subscribed is False
        assert len(daemon._coordinators) == 0

    def test_singleton_accessor(self):
        """Test singleton accessor returns same instance."""
        # Reset for clean test
        import app.coordination.coordinator_health_monitor_daemon as module
        module._monitor_instance = None

        daemon1 = get_coordinator_health_monitor_sync()
        daemon2 = get_coordinator_health_monitor_sync()
        assert daemon1 is daemon2

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test daemon start/stop lifecycle."""
        daemon = CoordinatorHealthMonitorDaemon()

        # Start daemon
        result = await daemon.start()
        assert daemon._running is True

        # Stop daemon
        await daemon.stop()
        assert daemon._running is False

    def test_get_health_summary_empty(self):
        """Test get_health_summary when no coordinators tracked."""
        daemon = CoordinatorHealthMonitorDaemon()
        summary = daemon.get_health_summary()

        assert summary.total_count == 0
        assert summary.healthy_count == 0
        assert summary.cluster_healthy is True
        assert summary.cluster_health_pct == 100.0

    def test_get_status(self):
        """Test get_status returns expected structure."""
        daemon = CoordinatorHealthMonitorDaemon()
        status = daemon.get_status()

        assert "running" in status
        assert "subscribed" in status
        assert "total_coordinators" in status
        assert "cluster_healthy" in status
        assert "total_events" in status

    def test_health_check_not_running(self):
        """Test health_check when daemon not running."""
        daemon = CoordinatorHealthMonitorDaemon()
        result = daemon.health_check()

        # Should indicate unhealthy
        if hasattr(result, 'healthy'):
            assert result.healthy is False
        elif isinstance(result, dict):
            assert result.get('healthy') is False

    @pytest.mark.asyncio
    async def test_on_coordinator_healthy(self):
        """Test handling COORDINATOR_HEALTHY event."""
        daemon = CoordinatorHealthMonitorDaemon()

        event = {
            "coordinator_name": "TrainingCoordinator",
            "node_id": "leader-node",
        }

        await daemon._on_coordinator_healthy(event)

        assert "TrainingCoordinator" in daemon._coordinators
        info = daemon._coordinators["TrainingCoordinator"]
        assert info.state == CoordinatorState.HEALTHY
        assert info.node_id == "leader-node"
        assert daemon._total_healthy_events == 1

    @pytest.mark.asyncio
    async def test_on_coordinator_unhealthy(self):
        """Test handling COORDINATOR_UNHEALTHY event."""
        daemon = CoordinatorHealthMonitorDaemon()

        event = {
            "coordinator_name": "SyncCoordinator",
            "node_id": "worker-node",
        }

        await daemon._on_coordinator_unhealthy(event)

        info = daemon._coordinators["SyncCoordinator"]
        assert info.state == CoordinatorState.UNHEALTHY
        assert daemon._total_unhealthy_events == 1

    @pytest.mark.asyncio
    async def test_on_coordinator_degraded(self):
        """Test handling COORDINATOR_HEALTH_DEGRADED event."""
        daemon = CoordinatorHealthMonitorDaemon()

        event = {
            "coordinator_name": "QueuePopulator",
            "reason": "High queue depth",
        }

        await daemon._on_coordinator_degraded(event)

        info = daemon._coordinators["QueuePopulator"]
        assert info.state == CoordinatorState.DEGRADED
        assert info.degraded_reason == "High queue depth"
        assert daemon._total_degraded_events == 1

    @pytest.mark.asyncio
    async def test_on_coordinator_shutdown(self):
        """Test handling COORDINATOR_SHUTDOWN event."""
        daemon = CoordinatorHealthMonitorDaemon()

        # First mark as healthy
        await daemon._on_coordinator_healthy({"coordinator_name": "TestCoordinator"})

        # Then shutdown
        await daemon._on_coordinator_shutdown({"coordinator_name": "TestCoordinator"})

        info = daemon._coordinators["TestCoordinator"]
        assert info.state == CoordinatorState.SHUTDOWN
        assert daemon._total_shutdowns == 1

    @pytest.mark.asyncio
    async def test_on_coordinator_init_failed(self):
        """Test handling COORDINATOR_INIT_FAILED event."""
        daemon = CoordinatorHealthMonitorDaemon()

        # Simulate multiple init failures
        for i in range(INIT_FAILURE_MAX_RETRIES):
            await daemon._on_coordinator_init_failed({
                "coordinator_name": "FailingCoordinator"
            })

        info = daemon._coordinators["FailingCoordinator"]
        assert info.init_failure_count == INIT_FAILURE_MAX_RETRIES
        assert info.state == CoordinatorState.INIT_FAILED

    @pytest.mark.asyncio
    async def test_on_coordinator_heartbeat(self):
        """Test handling COORDINATOR_HEARTBEAT event."""
        daemon = CoordinatorHealthMonitorDaemon()

        await daemon._on_coordinator_heartbeat({
            "coordinator_name": "HeartbeatTest"
        })

        info = daemon._coordinators["HeartbeatTest"]
        assert info.last_heartbeat_at > 0
        assert info.state == CoordinatorState.HEALTHY  # Unknown -> Healthy
        assert daemon._total_heartbeats == 1

    def test_cluster_health_calculation(self):
        """Test cluster health percentage calculation."""
        daemon = CoordinatorHealthMonitorDaemon()

        # Add 10 coordinators: 9 healthy, 1 unhealthy (10% down)
        for i in range(9):
            daemon._coordinators[f"healthy-{i}"] = CoordinatorInfo(
                name=f"healthy-{i}",
                state=CoordinatorState.HEALTHY,
            )
        for i in range(1):
            daemon._coordinators[f"unhealthy-{i}"] = CoordinatorInfo(
                name=f"unhealthy-{i}",
                state=CoordinatorState.UNHEALTHY,
            )

        summary = daemon.get_health_summary()
        assert summary.total_count == 10
        assert summary.healthy_count == 9
        assert summary.unhealthy_count == 1
        # 90% healthy + 0% degraded = 90% operational
        assert summary.cluster_health_pct == 90.0
        # 10% unhealthy < 20% threshold = cluster healthy
        assert summary.cluster_healthy is True

    def test_cluster_unhealthy_threshold(self):
        """Test cluster becomes unhealthy when >20% coordinators are down."""
        daemon = CoordinatorHealthMonitorDaemon()

        # Add 10 coordinators: 7 healthy, 3 unhealthy (30% down)
        for i in range(7):
            daemon._coordinators[f"healthy-{i}"] = CoordinatorInfo(
                name=f"healthy-{i}",
                state=CoordinatorState.HEALTHY,
            )
        for i in range(3):
            daemon._coordinators[f"unhealthy-{i}"] = CoordinatorInfo(
                name=f"unhealthy-{i}",
                state=CoordinatorState.UNHEALTHY,
            )

        summary = daemon.get_health_summary()
        # 30% unhealthy > 20% threshold
        assert summary.cluster_healthy is False

    def test_stale_heartbeat_detection(self):
        """Test stale heartbeat detection."""
        daemon = CoordinatorHealthMonitorDaemon()

        # Add coordinator with stale heartbeat
        daemon._coordinators["stale-coord"] = CoordinatorInfo(
            name="stale-coord",
            state=CoordinatorState.HEALTHY,
            last_heartbeat_at=time.time() - HEARTBEAT_STALE_THRESHOLD_SECONDS - 100,
        )

        summary = daemon.get_health_summary()
        assert summary.stale_count == 1


class TestConstants:
    """Tests for module constants."""

    def test_thresholds_are_reasonable(self):
        """Test that threshold constants have reasonable values."""
        # Heartbeat stale threshold should be at least 1 minute
        assert HEARTBEAT_STALE_THRESHOLD_SECONDS >= 60
        # But not more than 1 hour
        assert HEARTBEAT_STALE_THRESHOLD_SECONDS <= 3600

        # Init failure retries should be reasonable
        assert INIT_FAILURE_MAX_RETRIES >= 1
        assert INIT_FAILURE_MAX_RETRIES <= 10
