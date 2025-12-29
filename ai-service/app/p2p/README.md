# P2P Adapter Architecture

This module provides the P2P mesh network infrastructure for distributed AI training coordination.

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │        HybridCoordinator            │
                    │    (hybrid_coordinator.py)          │
                    │  Routes operations based on flags   │
                    └───────────┬───────────┬─────────────┘
                                │           │
              ┌─────────────────┴───┐   ┌───┴─────────────────┐
              │   SWIM Membership   │   │   Raft Consensus    │
              │  (swim_adapter.py)  │   │  (raft_state.py)    │
              │  5s failure detect  │   │  Replicated queue   │
              └─────────┬───────────┘   └───────────┬─────────┘
                        │                           │
              ┌─────────┴───────────┐   ┌───────────┴─────────┐
              │   HTTP Heartbeats   │   │  Bully + SQLite     │
              │     (fallback)      │   │    (fallback)       │
              └─────────────────────┘   └─────────────────────┘
```

## Module Files

| File                    | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `hybrid_coordinator.py` | Routes operations to SWIM/Raft or HTTP/Bully   |
| `swim_adapter.py`       | SWIM gossip-based membership (5s detection)    |
| `raft_state.py`         | Raft replicated work queue and job assignments |
| `config.py`             | P2P configuration with environment overrides   |
| `models.py`             | Enums/dataclasses for nodes, jobs, resources   |
| `training.py`           | Training trigger logic and node ranking        |
| `notifications.py`      | Webhook notifications (Slack, Discord)         |

## SWIM Adapter

SWIM provides gossip-based membership with O(1) bandwidth per node.

```python
from app.p2p.swim_adapter import SwimMembershipManager

manager = SwimMembershipManager.from_distributed_hosts(node_id="my-node")
await manager.start()

# Fast failure detection (5s vs 60s+ with HTTP)
if manager.is_peer_alive("other-node"):
    await send_work(...)

alive_peers = manager.get_alive_peers()
summary = manager.get_membership_summary()
```

**Key Classes:**

- `SwimMembershipManager` - Pure SWIM membership
- `HybridMembershipManager` - SWIM + HTTP fallback
- `SwimConfig` - Protocol tuning (failure_timeout, ping_interval)

### SWIM Health and Recovery

The P2P orchestrator mixin (`scripts/p2p/membership_mixin.py`) exposes helper
methods for observability and recovery:

- `get_swim_health()` returns SWIM health and recovery stats.
- `_check_and_recover_swim()` can be called periodically to restart SWIM when
  unhealthy (rate-limited to once per 5 minutes).
- Successful recovery emits `SWIM_RECOVERED` (see `ai-service/docs/EVENT_CATALOG.md`).

## Raft State Machines

Raft provides strongly consistent replicated state via PySyncObj.

```python
from app.p2p.raft_state import create_replicated_work_queue

queue = create_replicated_work_queue(node_id="my-node")

# Replicated across cluster
queue.add_work("job-1", {"work_type": "selfplay", "board_type": "hex8"})

# Atomic claim with distributed lock
success = queue.claim_work("job-1", "my-node")

# Queue statistics
stats = queue.get_queue_stats()  # pending, running, is_leader
```

**Key Classes:**

- `ReplicatedWorkQueue` - Distributed work queue with atomic operations
- `ReplicatedJobAssignments` - Job-to-node assignment tracking
- `WorkItem`, `JobAssignment` - Data structures

## Hybrid Coordinator

Unified interface that routes to the appropriate protocol based on feature flags.

```python
from app.p2p.hybrid_coordinator import HybridCoordinator

coordinator = HybridCoordinator(orchestrator=p2p, node_id="my-node")
await coordinator.start()

# Routes to SWIM or HTTP based on MEMBERSHIP_MODE
peers = coordinator.get_alive_peers()

# Routes to Raft or Bully based on CONSENSUS_MODE
if coordinator.is_leader():
    success = coordinator.claim_work("job-1")

status = coordinator.get_status()  # swim, raft, http sections
```

## Configuration

All settings are environment-overridable with `RINGRIFT_P2P_` prefix.

```python
from app.p2p import P2PConfig, get_p2p_config

config = get_p2p_config()
config.HEARTBEAT_INTERVAL      # 15s (peer heartbeat frequency)
config.PEER_TIMEOUT            # 60s (peer considered dead)
config.LEADER_LEASE_DURATION   # 90s (leader lease TTL)
config.DISK_CRITICAL_THRESHOLD # 70% (reject jobs above this)
```

## Feature Flags

Control protocol activation:

| Variable                   | Values                  | Default | Description          |
| -------------------------- | ----------------------- | ------- | -------------------- |
| `RINGRIFT_SWIM_ENABLED`    | `true`/`false`          | `false` | Enable SWIM protocol |
| `RINGRIFT_MEMBERSHIP_MODE` | `http`/`swim`/`hybrid`  | `http`  | Membership source    |
| `RINGRIFT_RAFT_ENABLED`    | `true`/`false`          | `false` | Enable Raft          |
| `RINGRIFT_CONSENSUS_MODE`  | `bully`/`raft`/`hybrid` | `bully` | Consensus protocol   |

**Gradual rollout:**

```bash
# Stage 1: SWIM + HTTP fallback
export RINGRIFT_SWIM_ENABLED=true RINGRIFT_MEMBERSHIP_MODE=hybrid

# Stage 2: Pure SWIM
export RINGRIFT_MEMBERSHIP_MODE=swim

# Stage 3: Add Raft
export RINGRIFT_RAFT_ENABLED=true RINGRIFT_CONSENSUS_MODE=hybrid
```

**Instant rollback:**

```bash
export RINGRIFT_SWIM_ENABLED=false RINGRIFT_RAFT_ENABLED=false
```

## Dependencies

```bash
pip install swim-p2p>=1.2.0    # SWIM membership (optional)
pip install pysyncobj>=0.3.14  # Raft consensus (optional)
```

If missing, the system falls back to HTTP heartbeats and Bully election.

## Training Coordination

```python
from app.p2p import should_trigger_training, calculate_training_priority

# Check if training should start
should_train, reason = should_trigger_training(
    games_available=5000,
    hours_since_last_training=12.0,
)

# Rank nodes for training dispatch
priority = calculate_training_priority(
    gpu_type="H100", memory_gb=80, current_load=30, has_training_data=True
)
```

## Webhook Notifications

```python
from app.p2p import send_webhook_notification

send_webhook_notification(
    event_type="training_complete",
    title="Training Complete: hex8_2p",
    message="Elo +15.3 after 2.5h training",
    severity="info",  # info, warning, error, critical
    details={"board_type": "hex8", "elo_change": 15.3},
)
```

Configure via environment:

```bash
export RINGRIFT_SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export RINGRIFT_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

## Status Endpoint

The `/status` endpoint includes protocol status:

```json
{
  "swim_raft": {
    "membership_mode": "hybrid",
    "consensus_mode": "bully",
    "swim": { "enabled": true, "started": true, "alive_count": 15 },
    "raft": { "enabled": false, "available": false }
  }
}
```

## Integration

Used by `scripts/p2p_orchestrator.py` for cluster coordination:

```bash
# Check cluster status
curl -s http://localhost:8770/status | python3 -c '
import sys,json; d=json.load(sys.stdin)
print(f"Leader: {d[\"leader_id\"]}, Alive: {d[\"alive_peers\"]} nodes")
'
```
