# Monitoring Module

Unified monitoring framework for RingRift cluster health and training pipeline.

## Overview

This module provides comprehensive monitoring:

- Centralized alert thresholds
- Base classes for health monitors
- Cluster-wide health checks
- P2P-integrated monitoring via `MonitoringManager`
- Unified cluster and health orchestration at the package root

Advanced monitoring tools still live in submodules:

- `app.monitoring.p2p_monitoring` for Prometheus/Grafana leader handoff internals
- `app.monitoring.predictive_alerts` for predictive alerting
- `app.monitoring.training_dashboard` for the dashboard server and collector

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
    nodes=["gpu-node-1", "gpu-node-2", "gpu-node-3"],
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
from app.monitoring import MonitoringManager

manager = MonitoringManager(node_id="leader-1")
manager.update_peers(
    [
        {"node_id": "gpu-node-1", "host": "10.0.0.11", "is_alive": True},
        {"node_id": "gpu-node-2", "host": "10.0.0.12", "is_alive": True},
    ]
)

await manager.start_as_leader()
await manager.stop()
```

### Predictive Alerts

```python
from app.monitoring.predictive_alerts import PredictiveAlertConfig, PredictiveAlertManager

predictor = PredictiveAlertManager(
    PredictiveAlertConfig(
        disk_prediction_hours=6,
        elo_trend_window_hours=8,
    )
)

predictor.record_disk_usage("gpu-node-1", 82.0)
predictor.record_elo("hex8_2p_best", 1979.8)

alerts = await predictor.run_all_checks(
    node_ids=["gpu-node-1"],
    model_ids=["hex8_2p_best"],
    last_training_time=0,
)
```

### Training Dashboard

```python
from pathlib import Path

from app.monitoring.training_dashboard import DashboardServer, MetricsCollector

collector = MetricsCollector(db_path=Path("data/metrics/training_metrics.db"))
collector.record_training_step(
    epoch=15,
    step=300,
    loss=0.023,
    accuracy=0.76,
    learning_rate=0.0001,
    model_id="hex8_2p_v3",
)

dashboard = DashboardServer(collector)
# dashboard.run(port=8080)
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
