# Monitoring Module

Unified monitoring framework for RingRift cluster health and training pipeline.

## Overview

This module provides comprehensive monitoring:

- Centralized alert thresholds
- Base classes for health monitors
- Cluster-wide health checks
- P2P-integrated monitoring
- Predictive alerting
- Training dashboard

## Key Components

### Health Monitoring

```python
from app.monitoring import (
    HealthMonitor,
    HealthStatus,
    MonitoringResult,
    check_local_health,
)

# Quick local health check
result = check_local_health()
print(f"Status: {result.status.value}")  # healthy, degraded, unhealthy
print(f"Alerts: {len(result.alerts)}")

# Custom monitor
class GPUMonitor(HealthMonitor):
    def check_health(self) -> MonitoringResult:
        gpu_util = get_gpu_utilization()
        status = HealthStatus.HEALTHY if gpu_util < 90 else HealthStatus.DEGRADED
        return MonitoringResult(status=status, metrics={"gpu": gpu_util})
```

### Alert Thresholds

```python
from app.monitoring import THRESHOLDS, get_threshold, should_alert, AlertLevel

# Get specific threshold
disk_warning = get_threshold("disk", "warning")  # 65
disk_critical = get_threshold("disk", "critical")  # 85

# Check if alert should fire
if should_alert("disk", current_usage, "warning"):
    send_alert("Disk usage warning", level=AlertLevel.WARNING)

# All thresholds
print(THRESHOLDS)
# {
#     "disk": {"warning": 65, "critical": 85},
#     "gpu_utilization": {"warning": 90, "critical": 95},
#     "memory": {"warning": 80, "critical": 90},
#     ...
# }
```

### Cluster Monitoring

```python
from app.monitoring import ClusterHealthMonitor, create_cluster_monitor

# Create monitor for cluster
monitor = create_cluster_monitor(
    nodes=["lambda-h100", "lambda-gh200-b", "lambda-gh200-d"],
)

# Run health check
result = monitor.run_check()
for node_result in result.node_results:
    print(f"{node_result.node_id}: {node_result.status.value}")
```

### Composite Monitors

```python
from app.monitoring import CompositeMonitor

# Combine multiple monitors
composite = CompositeMonitor(
    monitors=[
        DiskMonitor(),
        GPUMonitor(),
        NetworkMonitor(),
    ],
    name="system_health",
)

result = composite.check_health()
# Aggregates all monitor results
```

### P2P Integration

```python
from app.monitoring import P2PHealthMonitor

# Monitor integrated with P2P orchestrator
monitor = P2PHealthMonitor(
    p2p_port=8770,
    check_interval=30,
)

# Automatically reports to P2P leader
monitor.start_background_checks()
```

### Predictive Alerts

```python
from app.monitoring import PredictiveAlertMonitor

# Predict issues before they happen
predictor = PredictiveAlertMonitor(
    history_hours=24,
    prediction_horizon_hours=6,
)

# Check for predicted issues
predictions = predictor.predict_issues()
for pred in predictions:
    print(f"Predicted: {pred.issue_type} in {pred.hours_until}")
```

### Training Dashboard

```python
from app.monitoring import TrainingDashboard

# Real-time training metrics
dashboard = TrainingDashboard()
dashboard.update_metrics(
    epoch=15,
    loss=0.023,
    accuracy=0.76,
    learning_rate=0.0001,
)

# Export to file
dashboard.save_snapshot("training_progress.json")
```

## Threshold Categories

| Category          | Warning | Critical | Description       |
| ----------------- | ------- | -------- | ----------------- |
| `disk`            | 65%     | 85%      | Disk usage        |
| `gpu_utilization` | 90%     | 95%      | GPU compute usage |
| `gpu_memory`      | 85%     | 95%      | GPU memory usage  |
| `memory`          | 80%     | 90%      | System memory     |
| `cpu`             | 85%     | 95%      | CPU usage         |
| `network`         | 80%     | 95%      | Network bandwidth |

## Health Status Levels

| Status      | Description                      |
| ----------- | -------------------------------- |
| `HEALTHY`   | All metrics within normal range  |
| `DEGRADED`  | Some warning thresholds exceeded |
| `UNHEALTHY` | Critical thresholds exceeded     |
| `UNKNOWN`   | Unable to determine health       |

## Alert Levels

| Level       | Description                     |
| ----------- | ------------------------------- |
| `INFO`      | Informational, no action needed |
| `WARNING`   | Attention recommended           |
| `CRITICAL`  | Immediate action required       |
| `EMERGENCY` | System at risk of failure       |
