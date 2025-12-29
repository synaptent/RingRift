"""Tests for app.coordination.cluster.health re-export module (December 2025).

Verifies that the unified health module correctly re-exports all health-related
APIs from the source modules without circular imports.
"""

import pytest


class TestHealthModuleImports:
    """Verify all imports from cluster.health work correctly."""

    def test_import_unified_health_manager_exports(self):
        """Verify UnifiedHealthManager and related exports are accessible."""
        from app.coordination.cluster.health import (
            UnifiedHealthManager,
            get_health_manager,
            wire_health_events,
            ErrorSeverity,
            RecoveryStatus,
        )

        assert UnifiedHealthManager is not None
        assert callable(get_health_manager)
        assert callable(wire_health_events)
        # ErrorSeverity uses INFO/WARNING/ERROR/CRITICAL levels
        assert hasattr(ErrorSeverity, "INFO")
        assert hasattr(ErrorSeverity, "WARNING")
        assert hasattr(ErrorSeverity, "ERROR")
        assert hasattr(ErrorSeverity, "CRITICAL")
        assert hasattr(RecoveryStatus, "PENDING")

    def test_import_host_health_policy_exports(self):
        """Verify host health policy exports are accessible."""
        from app.coordination.cluster.health import (
            HealthStatus,
            check_host_health,
            is_host_healthy,
            get_healthy_hosts,
            clear_health_cache,
            get_health_summary,
            is_cluster_healthy,
            check_cluster_health,
        )

        assert HealthStatus is not None
        assert callable(check_host_health)
        assert callable(is_host_healthy)
        assert callable(get_healthy_hosts)
        assert callable(clear_health_cache)
        assert callable(get_health_summary)
        assert callable(is_cluster_healthy)
        assert callable(check_cluster_health)

    def test_import_health_facade_exports(self):
        """Verify health_facade exports are accessible."""
        from app.coordination.cluster.health import (
            get_health_orchestrator,
            HealthCheckOrchestrator,
            NodeHealthState,
            NodeHealthDetails,
            get_node_health,
            get_healthy_nodes,
            get_unhealthy_nodes,
            get_degraded_nodes,
            get_offline_nodes,
            mark_node_retired,
            get_cluster_health_summary,
            get_system_health_score,
            get_system_health_level,
            should_pause_pipeline,
            SystemHealthLevel,
            SystemHealthScore,
        )

        assert callable(get_health_orchestrator)
        assert HealthCheckOrchestrator is not None
        assert hasattr(NodeHealthState, "HEALTHY")
        assert NodeHealthDetails is not None
        assert callable(get_node_health)
        assert callable(get_healthy_nodes)
        assert callable(get_unhealthy_nodes)
        assert callable(get_degraded_nodes)
        assert callable(get_offline_nodes)
        assert callable(mark_node_retired)
        assert callable(get_cluster_health_summary)
        assert callable(get_system_health_score)
        assert callable(get_system_health_level)
        assert callable(should_pause_pipeline)
        assert hasattr(SystemHealthLevel, "HEALTHY")
        assert SystemHealthScore is not None

    def test_import_deprecated_aliases(self):
        """Verify deprecated aliases are accessible for backward compatibility."""
        from app.coordination.cluster.health import (
            NodeHealthMonitor,
            get_node_health_monitor,
            NodeStatus,
            NodeHealth,
        )

        # These should be aliases to the canonical modules
        assert NodeHealthMonitor is not None
        assert callable(get_node_health_monitor)
        assert NodeStatus is not None
        assert NodeHealth is not None


class TestHealthModuleAllExports:
    """Verify __all__ exports match actual exports."""

    def test_all_exports_accessible(self):
        """Verify all items in __all__ are accessible."""
        from app.coordination.cluster import health

        for name in health.__all__:
            obj = getattr(health, name, None)
            assert obj is not None, f"__all__ export '{name}' is None or missing"


class TestNoCircularImports:
    """Verify no circular imports occur."""

    def test_import_cluster_health_no_errors(self):
        """Verify importing cluster.health doesn't raise ImportError."""
        # If there are circular imports, this will raise ImportError
        from app.coordination.cluster import health

        assert health is not None

    def test_import_via_package_no_errors(self):
        """Verify importing via package __init__ doesn't raise errors."""
        from app.coordination.cluster.health import UnifiedHealthManager

        assert UnifiedHealthManager is not None


class TestNodeHealthStateEnum:
    """Verify NodeHealthState enum values."""

    def test_node_health_state_values(self):
        """Verify NodeHealthState has expected values."""
        from app.coordination.cluster.health import NodeHealthState

        # Core health states
        assert hasattr(NodeHealthState, "HEALTHY")
        assert hasattr(NodeHealthState, "DEGRADED")
        assert hasattr(NodeHealthState, "UNHEALTHY")
        assert hasattr(NodeHealthState, "OFFLINE")

    def test_node_status_is_alias(self):
        """Verify NodeStatus is an alias for NodeHealthState."""
        from app.coordination.cluster.health import NodeHealthState, NodeStatus

        # Both should reference the same enum
        assert NodeStatus is NodeHealthState


class TestSystemHealthLevelEnum:
    """Verify SystemHealthLevel enum values."""

    def test_system_health_level_values(self):
        """Verify SystemHealthLevel has expected values."""
        from app.coordination.cluster.health import SystemHealthLevel

        assert hasattr(SystemHealthLevel, "HEALTHY")
        assert hasattr(SystemHealthLevel, "DEGRADED")
        assert hasattr(SystemHealthLevel, "CRITICAL")


class TestHealthStatusClass:
    """Verify HealthStatus dataclass (HostHealthStatus)."""

    def test_health_status_is_class(self):
        """Verify HealthStatus is a dataclass with expected attributes."""
        from app.coordination.cluster.health import HealthStatus

        # HealthStatus is aliased to HostHealthStatus dataclass
        # It has attributes for host health metrics
        assert hasattr(HealthStatus, "latency_ms")
        assert hasattr(HealthStatus, "error")
        assert hasattr(HealthStatus, "is_stale")
        assert hasattr(HealthStatus, "to_dict")


class TestRecoveryStatusEnum:
    """Verify RecoveryStatus enum values."""

    def test_recovery_status_values(self):
        """Verify RecoveryStatus has expected values."""
        from app.coordination.cluster.health import RecoveryStatus

        assert hasattr(RecoveryStatus, "PENDING")
        assert hasattr(RecoveryStatus, "IN_PROGRESS") or hasattr(
            RecoveryStatus, "RUNNING"
        )


class TestErrorSeverityEnum:
    """Verify ErrorSeverity enum values."""

    def test_error_severity_values(self):
        """Verify ErrorSeverity has expected values."""
        from app.coordination.cluster.health import ErrorSeverity

        # ErrorSeverity uses log-level-style naming
        assert hasattr(ErrorSeverity, "INFO")
        assert hasattr(ErrorSeverity, "WARNING")
        assert hasattr(ErrorSeverity, "ERROR")
        assert hasattr(ErrorSeverity, "CRITICAL")
