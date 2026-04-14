"""Tests for coordination package entrypoint exports."""

from __future__ import annotations

import app.coordination.availability as availability_pkg
import app.coordination.cluster as cluster_pkg
import app.coordination.health as health_pkg
import app.coordination.interfaces as interfaces_pkg
import app.coordination.lifecycle as lifecycle_pkg
import app.coordination.queue_strategies as queue_strategies_pkg
import app.coordination.selfplay as selfplay_pkg
import app.coordination.status_reporting as status_reporting_pkg
import app.coordination.training as training_pkg


def test_training_package_declares_public_exports() -> None:
    expected = {
        "TrainingCoordinator",
        "SelfplayOrchestrator",
        "PriorityJobScheduler",
        "UnifiedScheduler",
        "get_training_coordinator",
        "get_selfplay_orchestrator",
        "get_unified_scheduler",
        "wire_training_events",
        "wire_selfplay_events",
    }
    assert expected.issubset(set(training_pkg.__all__))
    assert training_pkg.TrainingCoordinator.__name__ == "TrainingCoordinator"
    assert training_pkg.SelfplayOrchestrator.__name__ == "SelfplayOrchestrator"
    assert training_pkg.PriorityJobScheduler.__name__ == "PriorityJobScheduler"
    assert training_pkg.UnifiedScheduler.__name__ == "UnifiedScheduler"
    for name in expected:
        assert name in dir(training_pkg)


def test_cluster_package_declares_public_exports() -> None:
    assert cluster_pkg.__all__ == ["health", "transport", "p2p"]
    assert cluster_pkg.health.__name__ == "app.coordination.cluster.health"
    assert cluster_pkg.transport.__name__ == "app.coordination.cluster_transport"
    assert cluster_pkg.p2p.__name__ == "app.coordination.p2p_backend"
    assert "health" in dir(cluster_pkg)
    assert "transport" in dir(cluster_pkg)
    assert "p2p" in dir(cluster_pkg)


def test_availability_package_declares_public_exports() -> None:
    expected = {
        "NodeMonitor": "app.coordination.availability.node_monitor",
        "HealthCheckLayer": "app.coordination.availability.node_monitor",
        "NodeHealthResult": "app.coordination.availability.node_monitor",
        "RecoveryEngine": "app.coordination.availability.recovery_engine",
        "RecoveryAction": "app.coordination.enums",
        "RecoveryResult": "app.coordination.availability.recovery_engine",
        "Provisioner": "app.coordination.availability.provisioner",
        "ProvisionResult": "app.coordination.availability.provisioner",
        "CapacityPlanner": "app.coordination.availability.capacity_planner",
        "CapacityBudget": "app.coordination.availability.capacity_planner",
        "ScaleRecommendation": "app.coordination.availability.capacity_planner",
        "ScaleAction": "app.coordination.enums",
    }

    assert availability_pkg.__all__ == list(expected)
    assert len(availability_pkg.__all__) == len(set(availability_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(availability_pkg, name)
        assert exported.__module__ == module_name
        assert name in dir(availability_pkg)


def test_interfaces_module_declares_protocol_surface() -> None:
    expected = [
        "IBackpressureMonitor",
        "IBackpressureSignal",
        "IResourceTargetManager",
        "IResourceTargets",
        "IScheduler",
        "IJobInfo",
        "IHealthChecker",
        "IHealthResult",
        "ISyncProvider",
    ]

    assert interfaces_pkg.__all__ == expected
    assert len(interfaces_pkg.__all__) == len(set(interfaces_pkg.__all__))
    for name in expected:
        exported = getattr(interfaces_pkg, name)
        assert exported.__module__ == "app.coordination.interfaces"
        assert name in dir(interfaces_pkg)


def test_health_package_declares_public_exports() -> None:
    expected = {
        "HealthStatus": "app.coordination.health.types",
        "HealthStatusInfo": "app.coordination.health.types",
        "to_health_status": "app.coordination.health.types",
        "from_legacy_health_state": "app.coordination.health.types",
        "from_legacy_health_level": "app.coordination.health.types",
        "from_legacy_system_health_level": "app.coordination.health.types",
        "from_legacy_node_health_state": "app.coordination.health.types",
        "get_health_score": "app.coordination.health.types",
        "from_health_score": "app.coordination.health.types",
    }

    assert health_pkg.__all__ == list(expected)
    assert len(health_pkg.__all__) == len(set(health_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(health_pkg, name)
        assert exported.__module__ == module_name
        assert name in dir(health_pkg)


def test_queue_strategies_package_declares_public_exports() -> None:
    expected = [
        "QueuePopulationHealthMixin",
        "QueuePopulationStateMixin",
        "QueuePopulationWorkMixin",
    ]

    assert queue_strategies_pkg.__all__ == expected
    assert len(queue_strategies_pkg.__all__) == len(set(queue_strategies_pkg.__all__))
    for name in expected:
        exported = getattr(queue_strategies_pkg, name)
        assert exported.__module__.startswith("app.coordination.queue_strategies.")
        assert name in dir(queue_strategies_pkg)


def test_selfplay_package_dir_lists_lazy_scheduler_exports() -> None:
    for name in (
        "SelfplayScheduler",
        "get_selfplay_scheduler",
        "reset_selfplay_scheduler",
    ):
        assert name in selfplay_pkg.__all__
        assert name in dir(selfplay_pkg)


def test_status_reporting_module_declares_public_exports() -> None:
    assert status_reporting_pkg.__all__ == ["get_all_coordinator_status", "get_system_health"]
    for name in status_reporting_pkg.__all__:
        assert name in dir(status_reporting_pkg)
        assert callable(getattr(status_reporting_pkg, name))


def test_lifecycle_module_declares_public_exports() -> None:
    expected = [
        "initialize_all_coordinators",
        "shutdown_all_coordinators",
        "start_coordinator_heartbeats",
        "stop_coordinator_heartbeats",
        "is_heartbeat_running",
    ]
    assert lifecycle_pkg.__all__ == expected
    for name in expected:
        assert name in dir(lifecycle_pkg)
        assert callable(getattr(lifecycle_pkg, name))
