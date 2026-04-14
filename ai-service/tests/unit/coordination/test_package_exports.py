"""Tests for coordination package entrypoint exports."""

from __future__ import annotations

import app.coordination.availability as availability_pkg
import app.coordination.cluster as cluster_pkg
import app.coordination.feedback as feedback_pkg
import app.coordination.health as health_pkg
import app.coordination.interfaces as interfaces_pkg
import app.coordination.lifecycle as lifecycle_pkg
import app.coordination.mixins as mixins_pkg
import app.coordination.node_availability as node_availability_pkg
import app.coordination.providers as providers_pkg
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


def test_feedback_package_declares_public_exports() -> None:
    expected = {
        "FeedbackClusterHealthMixin": "app.coordination.feedback.cluster_health_mixin",
        "ExplorationBoostMixin": "app.coordination.feedback.exploration_boost",
        "QualityFeedbackMixin": "app.coordination.feedback.quality_feedback",
        "EloVelocityAdaptationMixin": "app.coordination.feedback.elo_velocity_mixin",
        "TrainingCurriculumFeedbackMixin": "app.coordination.feedback.training_curriculum_mixin",
        "LossMonitoringMixin": "app.coordination.feedback.loss_monitoring_mixin",
        "EvaluationFeedbackMixin": "app.coordination.feedback.evaluation_feedback_mixin",
        "RegressionHandlingMixin": "app.coordination.feedback.regression_handling_mixin",
        "SelfplayFeedbackMixin": "app.coordination.feedback.selfplay_feedback_mixin",
    }

    assert feedback_pkg.__all__ == list(expected)
    assert len(feedback_pkg.__all__) == len(set(feedback_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(feedback_pkg, name)
        assert exported.__module__ == module_name
        assert name in dir(feedback_pkg)


def test_mixins_package_declares_public_exports() -> None:
    expected = {
        "HealthCheckMixin": "app.coordination.mixins.health_check_mixin",
        "DownloadProgress": "app.coordination.mixins.import_mixin",
        "ImportDaemonMixin": "app.coordination.mixins.import_mixin",
        "ImportValidationResult": "app.coordination.mixins.import_mixin",
        "EventSubscriptionMixin": "app.coordination.mixins.lifecycle_mixin",
        "LifecycleMixin": "app.coordination.mixins.lifecycle_mixin",
        "LifecycleState": "app.coordination.mixins.lifecycle_mixin",
        "DataPipelineOrchestratorProtocol": "app.coordination.pipeline_mixin_base",
        "PipelineMixinBase": "app.coordination.pipeline_mixin_base",
        "AutoSyncDaemonProtocol": "app.coordination.sync_mixin_base",
        "SyncMixinBase": "app.coordination.sync_mixin_base",
    }

    assert mixins_pkg.__all__ == list(expected)
    assert len(mixins_pkg.__all__) == len(set(mixins_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(mixins_pkg, name)
        assert exported.__module__ == module_name
        assert name in dir(mixins_pkg)


def test_node_availability_package_declares_public_exports() -> None:
    expected = {
        "ProviderInstanceState": "app.coordination.node_availability.state_checker",
        "InstanceInfo": "app.coordination.node_availability.state_checker",
        "StateChecker": "app.coordination.node_availability.state_checker",
        "STATE_TO_YAML_STATUS": "builtins",
        "ConfigUpdater": "app.coordination.node_availability.config_updater",
        "ConfigUpdateResult": "app.coordination.node_availability.config_updater",
        "NodeAvailabilityDaemon": "app.coordination.node_availability.daemon",
        "NodeAvailabilityConfig": "app.coordination.node_availability.daemon",
        "get_node_availability_daemon": "app.coordination.node_availability.daemon",
        "reset_daemon_instance": "app.coordination.node_availability.daemon",
    }

    assert node_availability_pkg.__all__ == list(expected)
    assert len(node_availability_pkg.__all__) == len(set(node_availability_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(node_availability_pkg, name)
        assert getattr(exported, "__module__", type(exported).__module__) == module_name
        assert name in dir(node_availability_pkg)


def test_providers_package_declares_public_exports() -> None:
    expected = {
        "CloudProvider": "app.coordination.providers.base",
        "Instance": "app.coordination.providers.base",
        "InstanceStatus": "app.coordination.providers.base",
        "ProviderType": "app.coordination.providers.base",
        "GPUType": "app.coordination.providers.base",
        "ProviderRegistry": "app.coordination.providers.registry",
        "ProviderConfig": "app.coordination.providers.registry",
        "CloudProviderProtocol": "app.coordination.providers.registry",
        "PROVIDER_CONFIGS": "builtins",
        "get_provider": "app.coordination.providers",
        "get_all_providers": "app.coordination.providers",
        "reset_providers": "app.coordination.providers",
    }

    assert providers_pkg.__all__ == list(expected)
    assert len(providers_pkg.__all__) == len(set(providers_pkg.__all__))
    for name, module_name in expected.items():
        exported = getattr(providers_pkg, name)
        assert getattr(exported, "__module__", type(exported).__module__) == module_name
        assert name in dir(providers_pkg)


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
