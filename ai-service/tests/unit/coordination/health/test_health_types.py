"""Tests for app.coordination.health.types - Unified Health Status System.

January 2026: Phase 4.1 testing.
"""

import pytest

from app.coordination.health.types import (
    HealthStatus,
    HealthStatusInfo,
    to_health_status,
    from_legacy_health_state,
    from_legacy_health_level,
    from_legacy_system_health_level,
    from_legacy_node_health_state,
    get_health_score,
    from_health_score,
)


class TestHealthStatus:
    """Tests for the HealthStatus enum."""

    def test_all_values_exist(self):
        """All expected health status values should exist."""
        expected = [
            "healthy", "degraded", "unhealthy", "critical",
            "evicted", "offline", "provider_down", "retired",
            "recovering", "unknown"
        ]
        for value in expected:
            assert HealthStatus(value) is not None

    def test_is_operational(self):
        """is_operational should return True for workable states."""
        assert HealthStatus.HEALTHY.is_operational is True
        assert HealthStatus.DEGRADED.is_operational is True
        assert HealthStatus.RECOVERING.is_operational is True
        assert HealthStatus.UNHEALTHY.is_operational is False
        assert HealthStatus.CRITICAL.is_operational is False
        assert HealthStatus.OFFLINE.is_operational is False

    def test_is_healthy(self):
        """is_healthy should return True only for HEALTHY."""
        assert HealthStatus.HEALTHY.is_healthy is True
        assert HealthStatus.DEGRADED.is_healthy is False
        assert HealthStatus.UNKNOWN.is_healthy is False

    def test_is_degraded(self):
        """is_degraded should return True only for DEGRADED."""
        assert HealthStatus.DEGRADED.is_degraded is True
        assert HealthStatus.HEALTHY.is_degraded is False
        assert HealthStatus.UNHEALTHY.is_degraded is False

    def test_is_unhealthy(self):
        """is_unhealthy should return True for UNHEALTHY and CRITICAL."""
        assert HealthStatus.UNHEALTHY.is_unhealthy is True
        assert HealthStatus.CRITICAL.is_unhealthy is True
        assert HealthStatus.HEALTHY.is_unhealthy is False
        assert HealthStatus.DEGRADED.is_unhealthy is False

    def test_is_offline(self):
        """is_offline should return True for offline states."""
        assert HealthStatus.OFFLINE.is_offline is True
        assert HealthStatus.PROVIDER_DOWN.is_offline is True
        assert HealthStatus.RETIRED.is_offline is True
        assert HealthStatus.EVICTED.is_offline is True
        assert HealthStatus.HEALTHY.is_offline is False
        assert HealthStatus.UNHEALTHY.is_offline is False

    def test_severity_ordering(self):
        """Severity should order from healthy to worst."""
        assert HealthStatus.HEALTHY.severity == 0
        assert HealthStatus.DEGRADED.severity == 1
        assert HealthStatus.RECOVERING.severity == 2
        assert HealthStatus.UNHEALTHY.severity == 3
        assert HealthStatus.CRITICAL.severity == 4
        assert HealthStatus.EVICTED.severity > HealthStatus.CRITICAL.severity

    def test_comparison_operators(self):
        """Comparison operators should work based on severity."""
        assert HealthStatus.HEALTHY < HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED < HealthStatus.UNHEALTHY
        assert HealthStatus.UNHEALTHY <= HealthStatus.CRITICAL
        assert HealthStatus.CRITICAL > HealthStatus.HEALTHY
        assert HealthStatus.HEALTHY >= HealthStatus.HEALTHY

    def test_worst_method(self):
        """worst() should return the highest severity status."""
        assert HealthStatus.worst(
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY
        ) == HealthStatus.UNHEALTHY

        assert HealthStatus.worst() == HealthStatus.UNKNOWN

    def test_best_method(self):
        """best() should return the lowest severity status."""
        assert HealthStatus.best(
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY
        ) == HealthStatus.HEALTHY

        assert HealthStatus.best() == HealthStatus.UNKNOWN


class TestHealthStatusInfo:
    """Tests for the HealthStatusInfo dataclass."""

    def test_factory_methods(self):
        """Factory methods should create correct instances."""
        healthy = HealthStatusInfo.healthy("All good")
        assert healthy.status == HealthStatus.HEALTHY
        assert healthy.message == "All good"
        assert healthy.timestamp > 0

        degraded = HealthStatusInfo.degraded("Minor issue", count=5)
        assert degraded.status == HealthStatus.DEGRADED
        assert degraded.details["count"] == 5

        unhealthy = HealthStatusInfo.unhealthy("Big problem")
        assert unhealthy.status == HealthStatus.UNHEALTHY

        critical = HealthStatusInfo.critical("System down")
        assert critical.status == HealthStatus.CRITICAL

        unknown = HealthStatusInfo.unknown()
        assert unknown.status == HealthStatus.UNKNOWN

    def test_delegation_properties(self):
        """Properties should delegate to status."""
        info = HealthStatusInfo.healthy("OK")
        assert info.is_healthy is True
        assert info.is_operational is True

        info = HealthStatusInfo.unhealthy("Bad")
        assert info.is_healthy is False
        assert info.is_operational is False


class TestToHealthStatus:
    """Tests for the to_health_status conversion function."""

    def test_none_returns_unknown(self):
        """None should convert to UNKNOWN."""
        assert to_health_status(None) == HealthStatus.UNKNOWN

    def test_pass_through(self):
        """HealthStatus instances should pass through."""
        assert to_health_status(HealthStatus.HEALTHY) == HealthStatus.HEALTHY
        assert to_health_status(HealthStatus.CRITICAL) == HealthStatus.CRITICAL

    def test_string_conversion(self):
        """Strings should convert correctly."""
        assert to_health_status("healthy") == HealthStatus.HEALTHY
        assert to_health_status("HEALTHY") == HealthStatus.HEALTHY
        assert to_health_status("Degraded") == HealthStatus.DEGRADED
        assert to_health_status("unhealthy") == HealthStatus.UNHEALTHY
        assert to_health_status("critical") == HealthStatus.CRITICAL
        assert to_health_status("unknown") == HealthStatus.UNKNOWN

    def test_legacy_mappings(self):
        """Legacy string values should map correctly."""
        assert to_health_status("ok") == HealthStatus.HEALTHY
        assert to_health_status("OK") == HealthStatus.HEALTHY
        assert to_health_status("warning") == HealthStatus.DEGRADED
        assert to_health_status("error") == HealthStatus.UNHEALTHY
        assert to_health_status("good") == HealthStatus.HEALTHY
        assert to_health_status("bad") == HealthStatus.UNHEALTHY
        assert to_health_status("up") == HealthStatus.HEALTHY
        assert to_health_status("down") == HealthStatus.OFFLINE

    def test_unknown_string(self):
        """Unknown strings should return UNKNOWN."""
        assert to_health_status("garbage") == HealthStatus.UNKNOWN
        assert to_health_status("") == HealthStatus.UNKNOWN


class TestLegacyConversions:
    """Tests for legacy enum conversion functions."""

    def test_from_legacy_health_state(self):
        """HealthState conversion should work."""
        # Simulate enum-like objects
        class MockHealthState:
            def __init__(self, value: str):
                self.value = value

        assert from_legacy_health_state(MockHealthState("healthy")) == HealthStatus.HEALTHY
        assert from_legacy_health_state(MockHealthState("degraded")) == HealthStatus.DEGRADED
        assert from_legacy_health_state(MockHealthState("unhealthy")) == HealthStatus.UNHEALTHY
        assert from_legacy_health_state(MockHealthState("unknown")) == HealthStatus.UNKNOWN
        assert from_legacy_health_state(None) == HealthStatus.UNKNOWN

    def test_from_legacy_health_level(self):
        """HealthLevel conversion should work."""
        class MockHealthLevel:
            def __init__(self, value: str):
                self.value = value

        assert from_legacy_health_level(MockHealthLevel("ok")) == HealthStatus.HEALTHY
        assert from_legacy_health_level(MockHealthLevel("warning")) == HealthStatus.DEGRADED
        assert from_legacy_health_level(MockHealthLevel("error")) == HealthStatus.UNHEALTHY
        assert from_legacy_health_level(MockHealthLevel("unknown")) == HealthStatus.UNKNOWN
        assert from_legacy_health_level(None) == HealthStatus.UNKNOWN

    def test_from_legacy_system_health_level(self):
        """SystemHealthLevel conversion should work."""
        class MockSystemHealthLevel:
            def __init__(self, value: str):
                self.value = value

        assert from_legacy_system_health_level(MockSystemHealthLevel("healthy")) == HealthStatus.HEALTHY
        assert from_legacy_system_health_level(MockSystemHealthLevel("degraded")) == HealthStatus.DEGRADED
        assert from_legacy_system_health_level(MockSystemHealthLevel("unhealthy")) == HealthStatus.UNHEALTHY
        assert from_legacy_system_health_level(MockSystemHealthLevel("critical")) == HealthStatus.CRITICAL
        assert from_legacy_system_health_level(None) == HealthStatus.UNKNOWN

    def test_from_legacy_node_health_state(self):
        """NodeHealthState conversion should work (all values)."""
        class MockNodeHealthState:
            def __init__(self, value: str):
                self.value = value

        assert from_legacy_node_health_state(MockNodeHealthState("healthy")) == HealthStatus.HEALTHY
        assert from_legacy_node_health_state(MockNodeHealthState("degraded")) == HealthStatus.DEGRADED
        assert from_legacy_node_health_state(MockNodeHealthState("unhealthy")) == HealthStatus.UNHEALTHY
        assert from_legacy_node_health_state(MockNodeHealthState("evicted")) == HealthStatus.EVICTED
        assert from_legacy_node_health_state(MockNodeHealthState("offline")) == HealthStatus.OFFLINE
        assert from_legacy_node_health_state(MockNodeHealthState("provider_down")) == HealthStatus.PROVIDER_DOWN
        assert from_legacy_node_health_state(MockNodeHealthState("retired")) == HealthStatus.RETIRED
        assert from_legacy_node_health_state(MockNodeHealthState("recovering")) == HealthStatus.RECOVERING
        assert from_legacy_node_health_state(MockNodeHealthState("unknown")) == HealthStatus.UNKNOWN
        assert from_legacy_node_health_state(None) == HealthStatus.UNKNOWN


class TestHealthScoring:
    """Tests for health score conversion functions."""

    def test_get_health_score(self):
        """get_health_score should return correct scores."""
        assert get_health_score(HealthStatus.HEALTHY) == 1.0
        assert get_health_score(HealthStatus.DEGRADED) == 0.8
        assert get_health_score(HealthStatus.RECOVERING) == 0.6
        assert get_health_score(HealthStatus.UNHEALTHY) == 0.4
        assert get_health_score(HealthStatus.CRITICAL) == 0.2
        assert get_health_score(HealthStatus.OFFLINE) == 0.0
        assert get_health_score(HealthStatus.UNKNOWN) == 0.0

    def test_from_health_score(self):
        """from_health_score should return correct status."""
        assert from_health_score(1.0) == HealthStatus.HEALTHY
        assert from_health_score(0.95) == HealthStatus.HEALTHY
        assert from_health_score(0.85) == HealthStatus.DEGRADED
        assert from_health_score(0.75) == HealthStatus.DEGRADED
        assert from_health_score(0.55) == HealthStatus.RECOVERING
        assert from_health_score(0.45) == HealthStatus.UNHEALTHY
        assert from_health_score(0.25) == HealthStatus.CRITICAL
        assert from_health_score(0.05) == HealthStatus.OFFLINE
        assert from_health_score(0.0) == HealthStatus.OFFLINE


class TestIntegrationWithRealEnums:
    """Integration tests with actual legacy enums."""

    def test_actual_health_state(self):
        """Test conversion with actual HealthState enum."""
        from app.core.health import HealthState

        assert from_legacy_health_state(HealthState.HEALTHY) == HealthStatus.HEALTHY
        assert from_legacy_health_state(HealthState.DEGRADED) == HealthStatus.DEGRADED
        assert from_legacy_health_state(HealthState.UNHEALTHY) == HealthStatus.UNHEALTHY
        assert from_legacy_health_state(HealthState.UNKNOWN) == HealthStatus.UNKNOWN

    def test_actual_health_level(self):
        """Test conversion with actual HealthLevel enum."""
        from app.distributed.health_registry import HealthLevel

        assert from_legacy_health_level(HealthLevel.OK) == HealthStatus.HEALTHY
        assert from_legacy_health_level(HealthLevel.WARNING) == HealthStatus.DEGRADED
        assert from_legacy_health_level(HealthLevel.ERROR) == HealthStatus.UNHEALTHY
        assert from_legacy_health_level(HealthLevel.UNKNOWN) == HealthStatus.UNKNOWN

    def test_actual_system_health_level(self):
        """Test conversion with actual SystemHealthLevel enum."""
        from app.coordination.unified_health_manager import SystemHealthLevel

        assert from_legacy_system_health_level(SystemHealthLevel.HEALTHY) == HealthStatus.HEALTHY
        assert from_legacy_system_health_level(SystemHealthLevel.DEGRADED) == HealthStatus.DEGRADED
        assert from_legacy_system_health_level(SystemHealthLevel.UNHEALTHY) == HealthStatus.UNHEALTHY
        assert from_legacy_system_health_level(SystemHealthLevel.CRITICAL) == HealthStatus.CRITICAL

    def test_actual_node_health_state(self):
        """Test conversion with actual NodeHealthState enum."""
        from app.coordination.node_status import NodeHealthState

        assert from_legacy_node_health_state(NodeHealthState.HEALTHY) == HealthStatus.HEALTHY
        assert from_legacy_node_health_state(NodeHealthState.DEGRADED) == HealthStatus.DEGRADED
        assert from_legacy_node_health_state(NodeHealthState.UNHEALTHY) == HealthStatus.UNHEALTHY
        assert from_legacy_node_health_state(NodeHealthState.EVICTED) == HealthStatus.EVICTED
        assert from_legacy_node_health_state(NodeHealthState.OFFLINE) == HealthStatus.OFFLINE
        assert from_legacy_node_health_state(NodeHealthState.PROVIDER_DOWN) == HealthStatus.PROVIDER_DOWN
        assert from_legacy_node_health_state(NodeHealthState.RETIRED) == HealthStatus.RETIRED
        assert from_legacy_node_health_state(NodeHealthState.UNKNOWN) == HealthStatus.UNKNOWN

    def test_to_health_status_with_real_enums(self):
        """Test to_health_status with real legacy enums."""
        from app.core.health import HealthState
        from app.distributed.health_registry import HealthLevel
        from app.coordination.unified_health_manager import SystemHealthLevel

        # All should work via to_health_status
        assert to_health_status(HealthState.HEALTHY) == HealthStatus.HEALTHY
        assert to_health_status(HealthLevel.OK) == HealthStatus.HEALTHY
        assert to_health_status(SystemHealthLevel.CRITICAL) == HealthStatus.CRITICAL
