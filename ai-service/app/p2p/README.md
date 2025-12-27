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

---

## SWIM/Raft Protocol Integration (December 2025)

The P2P orchestrator supports battle-tested protocols for improved cluster stability:

### Protocol Overview

| Protocol | Purpose                 | Improvement Over Current       | Status            |
| -------- | ----------------------- | ------------------------------ | ----------------- |
| **SWIM** | Gossip-based membership | 5s failure detection (vs 60s+) | Integration ready |
| **Raft** | Replicated work queue   | Sub-second leader failover     | Integration ready |

### SWIM Membership Layer (`swim_adapter.py`)

SWIM provides leaderless cluster membership with O(1) bandwidth per node:

```python
from app.p2p.swim_adapter import SwimMembershipManager

# Create manager with seeds from distributed_hosts.yaml
manager = SwimMembershipManager.from_distributed_hosts(
    node_id="nebius-backbone-1",
    bind_port=7947,
)
await manager.start()

# Check peer status (5s failure detection)
if manager.is_peer_alive("runpod-h100"):
    await send_work_to_peer(...)

# Get all alive peers
alive_peers = manager.get_alive_peers()

# Get membership summary
summary = manager.get_membership_summary()
print(f"Alive: {summary['alive']}, Suspected: {summary['suspected']}")
```

**Key Classes:**

- `SwimMembershipManager` - Pure SWIM membership management
- `HybridMembershipManager` - SWIM + HTTP heartbeat fallback
- `SwimConfig` - Protocol tuning (failure_timeout, ping_interval)

**Benefits:**

- **Faster failure detection**: 5s (suspicion + failure) vs 60-90s HTTP polling
- **Constant bandwidth**: O(1) messages per node regardless of cluster size
- **No leader required**: All nodes are equal, no election needed for membership

### Raft Consensus Layer (`raft_state.py`)

Raft provides strongly consistent replicated state via PySyncObj:

```python
from app.p2p.raft_state import ReplicatedWorkQueue, create_replicated_work_queue

# Create work queue with auto-configured addresses
queue = create_replicated_work_queue(
    node_id="nebius-h100-1",
    on_ready=lambda: print("Raft cluster ready"),
    on_leader_change=lambda leader: print(f"Leader: {leader}"),
)

# Wait for cluster formation
await asyncio.sleep(1.0)

# Add work (replicated across cluster)
queue.add_work("work-001", {
    "work_type": "selfplay",
    "board_type": "hex8",
    "priority": 80,
})

# Claim work atomically (distributed lock prevents races)
success = queue.claim_work("work-001", "nebius-h100-1")

# Get queue statistics
stats = queue.get_queue_stats()
print(f"Pending: {stats['pending']}, Running: {stats['running']}")
print(f"Is leader: {stats['is_leader']}")
```

**Key Classes:**

- `ReplicatedWorkQueue` - Distributed work queue with atomic claim/complete
- `ReplicatedJobAssignments` - Job-to-node assignment tracking
- `WorkItem` / `JobAssignment` - Data structures for work/job records

**Features:**

- **Atomic operations**: `claim_work()` uses distributed locks to prevent races
- **Automatic failover**: Sub-second leader election on failure
- **Log compaction**: Automatic state snapshots for long-running clusters
- **Retry support**: Failed work automatically requeued up to max_attempts

### HybridCoordinator (`hybrid_coordinator.py`)

The glue layer that coordinates all protocols with automatic fallback:

```python
from app.p2p.hybrid_coordinator import HybridCoordinator, create_hybrid_coordinator

# Create coordinator (auto-detects available protocols)
coordinator = create_hybrid_coordinator(
    orchestrator=p2p_orchestrator,
    node_id="my-node",
)
await coordinator.start()

# Get alive peers (SWIM or HTTP based on MEMBERSHIP_MODE)
peers = coordinator.get_alive_peers()

# Check leader status (Raft or Bully based on CONSENSUS_MODE)
if coordinator.is_leader():
    # Dispatch work...
    pass

# Claim work (Raft or SQLite based on CONSENSUS_MODE)
success = coordinator.claim_work("work-001")

# Get comprehensive status
status = coordinator.get_status()
print(f"SWIM started: {status['swim']['started']}")
print(f"Raft ready: {status['raft']['ready']}")
print(f"Fallback active: swim={status['http']['alive_peers']} nodes")

await coordinator.stop()
```

### Feature Flags

Control protocol activation via environment variables:

```bash
# Stage 1: SWIM membership only (safest start)
export RINGRIFT_SWIM_ENABLED=true
export RINGRIFT_MEMBERSHIP_MODE=hybrid  # Uses SWIM + HTTP fallback

# Stage 2: SWIM as primary (disable HTTP heartbeats)
export RINGRIFT_MEMBERSHIP_MODE=swim

# Stage 3: Add Raft for work queue
export RINGRIFT_RAFT_ENABLED=true
export RINGRIFT_CONSENSUS_MODE=hybrid  # Uses Raft + Bully fallback

# Stage 4: Full Raft
export RINGRIFT_CONSENSUS_MODE=raft
```

| Variable                   | Values                  | Default | Description           |
| -------------------------- | ----------------------- | ------- | --------------------- |
| `RINGRIFT_SWIM_ENABLED`    | `true`/`false`          | `false` | Enable SWIM protocol  |
| `RINGRIFT_MEMBERSHIP_MODE` | `http`/`swim`/`hybrid`  | `http`  | Membership protocol   |
| `RINGRIFT_RAFT_ENABLED`    | `true`/`false`          | `false` | Enable Raft consensus |
| `RINGRIFT_CONSENSUS_MODE`  | `bully`/`raft`/`hybrid` | `bully` | Consensus protocol    |

### Rollback Plan

Instant rollback via environment variables (no code changes required):

```bash
# Disable SWIM (revert to HTTP heartbeats)
export RINGRIFT_SWIM_ENABLED=false

# Disable Raft (revert to Bully + SQLite)
export RINGRIFT_RAFT_ENABLED=false
```

Nodes without new dependencies continue working normally using HTTP/Bully fallback.

### Dependencies

Optional dependencies for full SWIM/Raft support:

```bash
pip install swim-p2p>=1.2.0    # SWIM membership
pip install pysyncobj>=0.3.14  # Raft consensus
```

If dependencies are missing, the coordinator automatically falls back to HTTP heartbeats and Bully election.

### Status Endpoint

The `/status` endpoint includes protocol status:

```json
{
  "swim_raft": {
    "membership_mode": "hybrid",
    "consensus_mode": "bully",
    "swim": {
      "enabled": true,
      "available": true,
      "started": true,
      "alive_count": 15,
      "suspected_count": 0,
      "failed_count": 2
    },
    "raft": {
      "enabled": false,
      "available": false,
      "ready": false
    }
  }
}
```

### File Structure

```
app/p2p/
├── swim_adapter.py         # SWIM membership manager
├── raft_state.py           # Raft replicated state machines
├── hybrid_coordinator.py   # Protocol coordination layer
├── config.py               # P2P configuration
├── models.py               # Enums and dataclasses
├── training.py             # Training coordination utils
└── notifications.py        # Webhook notifications
```

### Migration Checklist

1. **Phase 1**: Install dependencies on all nodes
2. **Phase 2**: Enable SWIM in hybrid mode (`MEMBERSHIP_MODE=hybrid`)
3. **Phase 3**: Monitor SWIM status, verify 5s failure detection
4. **Phase 4**: Enable Raft in hybrid mode (`CONSENSUS_MODE=hybrid`)
5. **Phase 5**: Monitor Raft status, verify leader election
6. **Phase 6**: Switch to pure SWIM/Raft (`MEMBERSHIP_MODE=swim`, `CONSENSUS_MODE=raft`)
