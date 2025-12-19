"""Unified monitoring framework for RingRift cluster health and training pipeline.

This module provides:
- Centralized alert thresholds (thresholds.py)
- Base classes for health monitors (base.py)
- Concrete cluster monitors (cluster_monitor.py)
- P2P-integrated monitoring (p2p_monitoring.py)
- Predictive alerting (predictive_alerts.py)
- Training dashboard (training_dashboard.py)

Usage:
    from app.monitoring import HealthMonitor, THRESHOLDS, AlertLevel
    from app.monitoring.thresholds import get_threshold, should_alert

    # Check threshold
    if should_alert("disk", 75, "warning"):
        send_warning()

    # Create custom monitor
    class MyMonitor(HealthMonitor):
        def check_health(self) -> MonitoringResult:
            ...

    # Quick local health check
    from app.monitoring import check_local_health
    result = check_local_health()
    print(f"Status: {result.status.value}")

    # Cluster-wide monitoring
    from app.monitoring import ClusterHealthMonitor, create_cluster_monitor
    monitor = create_cluster_monitor(nodes=[...])
    result = monitor.run_check()
"""

# Thresholds
from app.monitoring.thresholds import (
    THRESHOLDS,
    AlertLevel,
    get_threshold,
    should_alert,
    get_all_thresholds,
)

# Base classes
from app.monitoring.base import (
    HealthMonitor,
    HealthStatus,
    Alert,
    MonitoringResult,
    CompositeMonitor,
)

# Concrete monitors
from app.monitoring.cluster_monitor import (
    DiskHealthMonitor,
    MemoryHealthMonitor,
    GPUHealthMonitor,
    NodeHealthMonitor,
    ClusterHealthMonitor,
    NodeInfo,
    create_cluster_monitor,
    check_local_health,
)

# P2P monitoring
from app.monitoring.p2p_monitoring import MonitoringManager

# Unified health orchestrator
from app.monitoring.unified_health import (
    UnifiedHealthOrchestrator,
    get_health_orchestrator,
    check_system_health,
    is_system_healthy,
)

# Unified cluster monitor (consolidates scripts)
from app.monitoring.unified_cluster_monitor import (
    UnifiedClusterMonitor,
    ClusterStatus,
    ClusterNodeStatus,
    TrainingStatus,
    EloStatus,
    DataQualityStatus,
    get_cluster_monitor,
    get_cluster_status,
    print_cluster_status,
)

__all__ = [
    # Thresholds
    "THRESHOLDS",
    "AlertLevel",
    "get_threshold",
    "should_alert",
    "get_all_thresholds",
    # Base classes
    "HealthMonitor",
    "HealthStatus",
    "Alert",
    "MonitoringResult",
    "CompositeMonitor",
    # Concrete monitors
    "DiskHealthMonitor",
    "MemoryHealthMonitor",
    "GPUHealthMonitor",
    "NodeHealthMonitor",
    "ClusterHealthMonitor",
    "NodeInfo",
    "create_cluster_monitor",
    "check_local_health",
    # P2P monitoring
    "MonitoringManager",
    # Unified health
    "UnifiedHealthOrchestrator",
    "get_health_orchestrator",
    "check_system_health",
    "is_system_healthy",
    # Unified cluster monitor
    "UnifiedClusterMonitor",
    "ClusterStatus",
    "ClusterNodeStatus",
    "TrainingStatus",
    "EloStatus",
    "DataQualityStatus",
    "get_cluster_monitor",
    "get_cluster_status",
    "print_cluster_status",
]
