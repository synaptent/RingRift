# P2P Orchestration Module

Modular components for the distributed AI training cluster P2P orchestrator.

## Overview

This module provides building blocks for the P2P mesh network that coordinates:

- Job distribution across GPU nodes
- Training trigger decisions
- Data synchronization
- Node health monitoring

## Key Components

### `config.py` - Configuration

P2P network configuration with environment overrides:

```python
from app.p2p import P2PConfig, get_p2p_config

config = get_p2p_config()

# Access settings
print(f"Port: {config.DEFAULT_PORT}")  # 8770
print(f"Heartbeat: {config.HEARTBEAT_INTERVAL}s")
print(f"Leader lease: {config.LEADER_LEASE_DURATION}s")

# GPU power rankings for job scheduling
from app.p2p import GPU_POWER_RANKINGS
print(GPU_POWER_RANKINGS["H100"])  # 1.0 (baseline)
print(GPU_POWER_RANKINGS["GH200"])  # 1.2 (faster)
```

### `models.py` - Data Structures

Enums and dataclasses for node/job management:

```python
from app.p2p import NodeRole, JobType, JobStatus, NodeHealth

# Node roles
role = NodeRole.LEADER  # or FOLLOWER, CANDIDATE

# Job types
job = JobType.SELFPLAY  # or TRAINING, EVALUATION, SYNC

# Job status
status = JobStatus.RUNNING  # PENDING, RUNNING, COMPLETED, FAILED

# Node health
health = NodeHealth.HEALTHY  # HEALTHY, DEGRADED, UNHEALTHY, OFFLINE
```

Resource tracking:

```python
from app.p2p.models import ResourceMetrics, NodeSummary

metrics = ResourceMetrics(
    cpu_percent=45.0,
    memory_percent=60.0,
    gpu_memory_percent=80.0,
    disk_percent=55.0,
)
print(f"Load score: {metrics.load_score}")  # Weighted composite
print(f"Overloaded: {metrics.is_overloaded}")  # True if any > threshold
```

### `training.py` - Training Coordination

Utilities for deciding when to trigger training:

```python
from app.p2p import TrainingThresholds, calculate_training_priority, should_trigger_training

thresholds = TrainingThresholds()
print(f"Min games: {thresholds.MIN_GAMES}")
print(f"Staleness hours: {thresholds.STALENESS_HOURS}")

# Calculate priority for a board config
priority = calculate_training_priority(
    games_since_last_train=5000,
    hours_since_last_train=12.0,
    current_elo=1200,
)

# Check if training should trigger
should_train = should_trigger_training(
    games_available=6000,
    hours_since_train=24.0,
    thresholds=thresholds,
)
```

### `notifications.py` - Webhooks

Webhook notifications for training events:

```python
from app.p2p import WebhookConfig, send_webhook_notification

# Configure webhook
config = WebhookConfig(
    url="https://hooks.slack.com/...",
    events=["training_complete", "model_promoted"],
)

# Send notification
send_webhook_notification(
    config,
    event="training_complete",
    payload={"model": "hex8_2p_v3", "accuracy": 0.76},
)
```

## Constants

| Constant                | Default | Description                 |
| ----------------------- | ------- | --------------------------- |
| `DEFAULT_PORT`          | 8770    | P2P orchestrator HTTP port  |
| `HEARTBEAT_INTERVAL`    | 5s      | Node heartbeat frequency    |
| `PEER_TIMEOUT`          | 15s     | Peer considered dead after  |
| `LEADER_LEASE_DURATION` | 30s     | Leader lease TTL            |
| `LOAD_MAX_FOR_NEW_JOBS` | 0.8     | Max load to accept new jobs |

## GPU Power Rankings

Relative GPU performance for job scheduling:

| GPU        | Power | Notes                   |
| ---------- | ----- | ----------------------- |
| `GH200`    | 1.2   | Grace Hopper, 96GB HBM3 |
| `H100`     | 1.0   | Baseline, 80GB HBM3     |
| `A100`     | 0.8   | 80GB HBM2e              |
| `RTX_5090` | 0.7   | Consumer flagship       |
| `RTX_4090` | 0.5   | Previous gen consumer   |
| `A10`      | 0.4   | Entry datacenter        |
| `CPU`      | 0.1   | CPU-only fallback       |

## Environment Overrides

All config values can be overridden via environment:

```bash
export RINGRIFT_P2P_PORT=8771
export RINGRIFT_P2P_HEARTBEAT_INTERVAL=10
export RINGRIFT_P2P_LEADER_LEASE=60
```

## Integration

Used by the main P2P orchestrator (`scripts/p2p_orchestrator.py`) and cluster monitoring tools.

```python
# Check cluster status
import requests
status = requests.get("http://localhost:8770/status").json()
print(f"Leader: {status['leader_id']}")
print(f"Alive peers: {status['alive_peers']}")
```
