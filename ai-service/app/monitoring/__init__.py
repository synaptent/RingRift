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
# Base classes
from app.monitoring.base import (
    Alert,
    CompositeMonitor,
    HealthMonitor,
    HealthStatus,
    MonitoringResult,
)

# Concrete monitors
from app.monitoring.cluster_monitor import (
    ClusterHealthMonitor,
    DiskHealthMonitor,
    GPUHealthMonitor,
    MemoryHealthMonitor,
    NodeHealthMonitor,
    NodeInfo,
    check_local_health,
    create_cluster_monitor,
)

# P2P monitoring
from app.monitoring.p2p_monitoring import MonitoringManager
from app.monitoring.thresholds import (
    THRESHOLDS,
    AlertLevel,
    get_all_thresholds,
    get_threshold,
    should_alert,
)

# Unified cluster monitor (consolidates scripts)
from app.monitoring.unified_cluster_monitor import (
    ClusterNodeStatus,
    ClusterStatus,
    DataQualityStatus,
    EloStatus,
    TrainingStatus,
    UnifiedClusterMonitor,
    get_cluster_monitor,
    get_cluster_status,
    print_cluster_status,
)

# Unified health orchestrator
from app.monitoring.unified_health import (
    UnifiedHealthOrchestrator,
    check_system_health,
    get_health_orchestrator,
    is_system_healthy,
)

__all__ = [
    # Thresholds
    "THRESHOLDS",
    "Alert",
    "AlertLevel",
    "ClusterHealthMonitor",
    "ClusterNodeStatus",
    "ClusterStatus",
    "CompositeMonitor",
    "DataQualityStatus",
    # Concrete monitors
    "DiskHealthMonitor",
    "EloStatus",
    "GPUHealthMonitor",
    # Base classes
    "HealthMonitor",
    "HealthStatus",
    "MemoryHealthMonitor",
    # P2P monitoring
    "MonitoringManager",
    "MonitoringResult",
    "NodeHealthMonitor",
    "NodeInfo",
    "TrainingStatus",
    # Unified cluster monitor
    "UnifiedClusterMonitor",
    # Unified health
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
