"""Focused tests for app.monitoring package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_monitoring_surface() -> None:
    module = importlib.import_module("app.monitoring")

    expected = [
        "THRESHOLDS",
        "Alert",
        "AlertLevel",
        "ClusterHealthMonitor",
        "ClusterNodeStatus",
        "ClusterStatus",
        "CompositeMonitor",
        "DataQualityStatus",
        "DiskHealthMonitor",
        "EloStatus",
        "GPUHealthMonitor",
        "HealthMonitor",
        "HealthStatus",
        "MemoryHealthMonitor",
        "MonitoringManager",
        "MonitoringResult",
        "NodeHealthMonitor",
        "NodeInfo",
        "TrainingStatus",
        "UnifiedClusterMonitor",
        "UnifiedHealthOrchestrator",
        "check_local_health",
        "check_system_health",
        "create_cluster_monitor",
        "get_all_thresholds",
        "get_cluster_monitor",
        "get_cluster_status",
        "get_health_orchestrator",
        "get_threshold",
        "is_system_healthy",
        "print_cluster_status",
        "should_alert",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
