"""Tests for VoterConfigOrchestrator helper health adapters."""

from __future__ import annotations

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.voter_config_orchestrator import (
    ConfigHealthStatus,
    VoterConfigOrchestrator,
)


class TestVoterConfigHealthHelpers:
    """Tests for config-health result and compatibility status adapters."""

    def setup_method(self) -> None:
        VoterConfigOrchestrator.reset_instance()

    def teardown_method(self) -> None:
        VoterConfigOrchestrator.reset_instance()

    def test_get_config_health_result_initializing_without_snapshot(self) -> None:
        orchestrator = VoterConfigOrchestrator()

        result = orchestrator.get_config_health_result()
        status = orchestrator.get_health_status()

        assert isinstance(result, HealthCheckResult)
        assert result.status == CoordinatorStatus.INITIALIZING
        assert status["status"] == "unknown"
        assert status["last_check"] == 0

    def test_get_config_health_result_degraded_with_drift(self) -> None:
        orchestrator = VoterConfigOrchestrator()
        orchestrator._consecutive_drift_events = 2
        orchestrator._last_drift_time = 123.0
        orchestrator._last_sync_time = 456.0
        orchestrator._health_status = ConfigHealthStatus(
            local_version=3,
            local_hash="abc123",
            peer_versions={"node-b": 5},
            highest_version=5,
            lowest_version=3,
            version_spread=2,
            peers_at_highest=1,
            peers_behind=1,
            is_healthy=False,
            health_reason="local_behind: v3 < v5",
            timestamp=789.0,
        )

        result = orchestrator.get_config_health_result()
        status = orchestrator.get_health_status()

        assert result.status == CoordinatorStatus.DEGRADED
        assert result.healthy is True
        assert result.details["version_spread"] == 2
        assert status["status"] == "unhealthy"
        assert status["local_version"] == 3
        assert status["consecutive_drift_events"] == 2
