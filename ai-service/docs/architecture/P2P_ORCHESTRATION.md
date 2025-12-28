# P2P Orchestration Architecture

**Last Updated:** December 2025
**Status:** Production
**Main Entry Point:** `scripts/p2p_orchestrator.py`

## Overview

The P2P orchestration layer coordinates a distributed mesh network of GPU nodes for self-play training and model optimization. It combines peer-to-peer membership management, leader election, work queue distribution, and data synchronization into a unified system.

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │            P2P Orchestrator (~28K LOC)          │
                    │  ┌─────────────────────────────────────────────┐│
                    │  │              15+ Handler Mixins             ││
                    │  │  (HTTP endpoints: /election, /gossip, etc.) ││
                    │  └─────────────────────────────────────────────┘│
                    │  ┌────────────────────────────────────────────┐ │
                    │  │               7 Managers                   │ │
                    │  │  StateManager │ JobManager │ NodeSelector  │ │
                    │  │  TrainingCoordinator │ SelfplayScheduler   │ │
                    │  │  SyncPlanner │ LoopManager                 │ │
                    │  └────────────────────────────────────────────┘ │
                    │  ┌────────────────────────────────────────────┐ │
                    │  │           10+ Background Loops             │ │
                    │  │  JobReaper │ IdleDetection │ EloSync       │ │
                    │  │  SelfHealing │ QueuePopulator │ ...        │ │
                    │  └────────────────────────────────────────────┘ │
                    └───────────────────────┬─────────────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────┐
                    │           Event Router (Event Bus)              │
                    │         118+ Event Types (DataEventType)        │
                    └───────────────────────┬─────────────────────────┘
                                            │
          ┌─────────────────────────────────┼─────────────────────────────────┐
          ▼                                 ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│   AutoSyncDaemon    │         │DataPipelineOrchest. │         │ UnifiedHealthManager│
│ (Data Replication)  │         │  (Stage Pipeline)   │         │  (Cluster Health)   │
└─────────────────────┘         └─────────────────────┘         └─────────────────────┘
```

## Directory Structure

```
scripts/
├── p2p_orchestrator.py          # Main entry point (~28K LOC)
└── p2p/
    ├── managers/                 # Domain-specific managers
    │   ├── state_manager.py      # SQLite persistence, epochs
    │   ├── job_manager.py        # Job spawning, lifecycle
    │   ├── training_coordinator.py  # Training dispatch, promotion
    │   ├── selfplay_scheduler.py # Curriculum-based scheduling
    │   ├── node_selector.py      # Node ranking, placement
    │   ├── sync_planner.py       # Manifest collection, sync
    │   └── loop_manager.py       # Background loop lifecycle
    ├── loops/                    # Background loops
    │   ├── base.py               # BaseLoop, LoopManager
    │   ├── job_loops.py          # JobReaper, IdleDetection
    │   ├── coordination_loops.py # AutoScaling, HealthAggregation
    │   ├── data_loops.py         # ModelSync, DataAggregation
    │   ├── network_loops.py      # TailscaleRecovery, NAT
    │   ├── resilience_loops.py   # SelfHealing, Predictive
    │   └── elo_sync_loop.py      # Elo rating sync
    ├── handlers/                 # HTTP endpoint mixins
    │   ├── base.py               # BaseP2PHandler, decorators
    │   ├── election.py           # Leader election (Bully)
    │   ├── gossip.py             # Heartbeat, manifest gossip
    │   ├── work_queue.py         # Job claim/complete
    │   ├── gauntlet.py           # Model evaluation
    │   ├── tournament.py         # Round-robin tournaments
    │   ├── elo_sync.py           # Rating sync
    │   ├── admin.py              # Git, health, debug
    │   ├── swim.py               # SWIM protocol (optional)
    │   └── raft.py               # Raft consensus (optional)
    └── p2p_mixin_base.py         # Shared mixin infrastructure
```

## Component Responsibilities

### Managers (`scripts/p2p/managers/`)

| Manager                 | Purpose                           | Key Methods                                               |
| ----------------------- | --------------------------------- | --------------------------------------------------------- |
| **StateManager**        | SQLite persistence for P2P state  | `load_state()`, `save_state()`, `advance_epoch()`         |
| **JobManager**          | Job spawning and lifecycle        | `run_gpu_selfplay_job()`, `spawn_training()`              |
| **TrainingCoordinator** | Training dispatch and promotion   | `dispatch_training_job()`, `run_post_training_gauntlet()` |
| **SelfplayScheduler**   | Curriculum-based config selection | `pick_weighted_config()`, `get_target_jobs()`             |
| **NodeSelector**        | Node ranking for job placement    | `get_best_gpu_node()`, `get_training_nodes()`             |
| **SyncPlanner**         | Cross-cluster data sync planning  | `collect_manifest()`, `create_sync_plan()`                |
| **LoopManager**         | Background loop lifecycle         | `start_all()`, `stop_all()`, `get_stats()`                |

### Background Loops (`scripts/p2p/loops/`)

| Loop                       | Interval | Purpose                                  |
| -------------------------- | -------- | ---------------------------------------- |
| `JobReaperLoop`            | 5 min    | Kill stale (1.5hr) and stuck (2hr+) jobs |
| `IdleDetectionLoop`        | 30 sec   | Spawn selfplay on idle GPUs              |
| `WorkerPullLoop`           | 30 sec   | Workers poll leader for jobs             |
| `WorkQueueMaintenanceLoop` | 5 min    | Cleanup old queue items (24hr)           |
| `EloSyncLoop`              | 5 min    | Synchronize Elo ratings                  |
| `QueuePopulatorLoop`       | 1 min    | Maintain work queue until Elo targets    |
| `SelfHealingLoop`          | 5 min    | Recover stuck jobs, clean processes      |
| `PredictiveMonitoringLoop` | 5 min    | Trend analysis, preempt failures         |
| `AutoScalingLoop`          | 5 min    | Adjust selfplay rate by utilization      |

### HTTP Handlers (`scripts/p2p/handlers/`)

**Core Cluster:**

- `POST /election/nominate` - Candidate announces leadership bid
- `POST /election/vote` - Voter grants leadership lease
- `GET /election/leader` - Query current leader

**Gossip Protocol:**

- `POST /gossip/heartbeat` - Peer membership and state
- `POST /gossip/manifest` - Game/model data registry

**Work Distribution:**

- `POST /work/add` - Enqueue job
- `GET /work/claim` - Worker claims job
- `POST /work/complete` - Mark job finished
- `GET /work/status` - Queue depth and stats

**Admin & Health:**

- `GET /status` - P2P state (leader, peers, epoch)
- `GET /health` - Node health (CPU, GPU, disk)
- `GET /cluster/health` - Aggregated cluster health

## Event Integration

The P2P layer emits and subscribes to events via the central Event Router:

### Events Emitted by P2P

| Event                 | Trigger                       | Subscribers                              |
| --------------------- | ----------------------------- | ---------------------------------------- |
| `HOST_ONLINE`         | Peer recovered                | UnifiedHealthManager                     |
| `HOST_OFFLINE`        | Peer dead (60s+ no heartbeat) | UnifiedHealthManager, NodeRecoveryDaemon |
| `LEADER_ELECTED`      | This node becomes leader      | LeadershipCoordinator                    |
| `DATA_SYNC_STARTED`   | Sync operation begins         | DataPipelineOrchestrator                 |
| `DATA_SYNC_COMPLETED` | Sync operation ends           | DataPipelineOrchestrator                 |

### Events Subscribed by P2P

| Event                | Source              | Action                     |
| -------------------- | ------------------- | -------------------------- |
| `NODE_RECOVERED`     | NodeRecoveryDaemon  | Re-add to active peer list |
| `TRAINING_COMPLETED` | TrainingCoordinator | Trigger model distribution |
| `MODEL_PROMOTED`     | AutoPromotionDaemon | Update model registry      |

## Data Structures

### Peer Representation

```python
@dataclass
class PeerInfo:
    peer_id: str               # e.g., "vast-12345"
    endpoint: str              # IP:port
    role: NodeRole             # LEADER, FOLLOWER, CANDIDATE
    health: NodeHealthState    # ALIVE, SUSPECT, DEAD
    last_heartbeat: float      # Unix timestamp
    gpu_type: str              # H100, 4090, etc.
    available_vram: int        # Bytes
```

### Job Structure

```python
@dataclass
class ClusterJob:
    job_id: str
    job_type: JobType          # SELFPLAY, TRAINING, GAUNTLET, etc.
    assigned_node: str
    config_key: str            # e.g., "hex8_2p"
    status: JobStatus          # QUEUED, STARTED, COMPLETED, FAILED
    created_at: float
    started_at: float | None
    completed_at: float | None
    result: dict               # Job-specific output
```

### Sync Manifest

```python
@dataclass
class DataManifest:
    node_id: str
    games: dict[str, FileInfo]
    models: dict[str, FileInfo]
    npz_files: dict[str, FileInfo]
    timestamp: float
```

## Workflow Examples

### Selfplay Job Lifecycle

```
1. SelfplayScheduler.pick_weighted_config()
   → Select config based on curriculum weights

2. JobManager.run_gpu_selfplay_job()
   → Create job, assign to best GPU node

3. Worker claims job via GET /work/claim
   → Execute selfplay, generate games

4. AutoSyncDaemon propagates data
   → Gossip replication to peers

5. DataPipelineOrchestrator detects new games
   → TRAINING_THRESHOLD_REACHED if sufficient

6. TrainingCoordinator.dispatch_training_job()
   → Training starts on best H100 node

7. FeedbackLoopController adjusts curriculum
   → Rebalance config weights
```

### Model Promotion Flow

```
1. TRAINING_COMPLETED event emitted
   ↓
2. FeedbackLoopController triggers gauntlet
   ↓
3. GameGauntlet plays vs baselines
   → If win_rate > threshold
   ↓
4. AutoPromotionDaemon emits MODEL_PROMOTED
   ↓
5. UnifiedDistributionDaemon syncs model
   → rsync to all training nodes
   ↓
6. SelfplayScheduler uses new model
   → Next generation selfplay
```

### Failure Recovery Flow

```
1. Peer heartbeat missing 60+ seconds
   → ALIVE → SUSPECT → DEAD

2. P2P emits HOST_OFFLINE

3. UnifiedHealthManager updates health scores

4. JobReaperLoop cancels jobs on dead node

5. QueuePopulatorLoop requeues failed work

6. Peer rejoins cluster
   → P2P emits HOST_ONLINE

7. SyncPlanner prioritizes sync to recovered node
```

## Configuration

### Environment Variables

```bash
# Core P2P settings
RINGRIFT_P2P_PORT=8770
RINGRIFT_P2P_BIND_HOST=0.0.0.0
RINGRIFT_P2P_ADVERTISE_HOST=<tailscale-ip>
RINGRIFT_P2P_AUTH_TOKEN=<secret>

# Timing configuration
RINGRIFT_P2P_HEARTBEAT_INTERVAL=30    # Peer discovery (seconds)
RINGRIFT_P2P_PEER_TIMEOUT=90          # Dead node detection
RINGRIFT_P2P_LEADER_LEASE=90          # Leader validity

# Optional protocol upgrades
RINGRIFT_SWIM_ENABLED=false           # SWIM membership (requires swim-p2p)
RINGRIFT_RAFT_ENABLED=false           # Raft consensus (requires pysyncobj)
RINGRIFT_MEMBERSHIP_MODE=http         # http|swim|hybrid
RINGRIFT_CONSENSUS_MODE=bully         # bully|raft|hybrid
```

### Cluster Definition (`config/distributed_hosts.yaml`)

```yaml
hosts:
  nebius-h100-3:
    tailscale_ip: 100.x.x.x
    ssh_host: 89.169.x.x
    gpu: H100
    gpu_vram_gb: 80
    role: training
    is_voter: true
    bandwidth_mbps: 100

sync_routing:
  max_disk_usage_percent: 70
  priority_hosts:
    - runpod-a100-1
    - nebius-h100-3
```

## Leader Election (Bully Algorithm)

The P2P cluster uses a Bully-style leader election with voter quorum:

1. **Voter Quorum**: 5 stable nodes designated as voters
2. **Quorum Requirement**: 3+ votes required for leadership
3. **Lease Duration**: 90 seconds (renewed via heartbeat)
4. **Election Trigger**: Leader heartbeat timeout

**Current Voters (Dec 2025):**

- nebius-backbone-1, nebius-h100-3
- hetzner-cpu1, hetzner-cpu2, hetzner-cpu3

## Health Monitoring

### Per-Node Health

```bash
GET /health
{
  "status": "healthy",
  "cpu_percent": 45.2,
  "gpu_utilization": 78.5,
  "disk_usage_percent": 42.0,
  "memory_percent": 55.3
}
```

### Cluster Health Aggregation

```bash
GET /cluster/health
{
  "overall": "healthy",
  "node_count": 28,
  "healthy_count": 26,
  "degraded_count": 2,
  "unhealthy_count": 0
}
```

## Multi-Transport Sync

SyncPlanner uses cascading transport selection for reliability:

1. **Tailscale** - Low-latency mesh (preferred)
2. **SSH Direct** - Public IP fallback
3. **SSH Jump** - Through jump host
4. **HTTP** - P2P data server
5. **Base64** - Extreme fallback for firewall issues

## Performance Characteristics

| Metric              | Value                         |
| ------------------- | ----------------------------- |
| Cluster Size        | ~36 nodes (designed for 100+) |
| Leader Election     | O(log N) convergence          |
| Gossip Propagation  | O(log N) hops                 |
| Work Queue Ops      | O(1) claim/complete           |
| Memory per Instance | ~500MB                        |
| Parallel Transfers  | 4 per node                    |

## Testing

```bash
# Unit tests for managers and loops
pytest tests/unit/p2p/ -v

# Integration tests for multi-node scenarios
pytest tests/integration/p2p/ -v

# Key test files:
# - test_job_loops.py (37 tests)
# - test_resilience_loops.py (25 tests)
# - test_p2p_mixin_base.py (consolidation tests)
```

## Troubleshooting

### Common Issues

**Leader election fails:**

```bash
# Check voter nodes are reachable
for host in nebius-backbone-1 nebius-h100-3 hetzner-cpu1; do
  curl -s http://$host:8770/health | jq .status
done
```

**Jobs stuck in STARTED:**

```bash
# Check JobReaperLoop is running
curl http://localhost:8770/status | jq '.loops.job_reaper'
```

**Data sync not propagating:**

```bash
# Check SyncPlanner manifest
curl http://localhost:8770/gossip/manifest | jq '.games | keys | length'
```

### Debug Logging

```bash
# Enable verbose P2P logging
export RINGRIFT_LOG_LEVEL=DEBUG
python scripts/p2p_orchestrator.py
```

## Related Documentation

- [DAEMON_REGISTRY.md](./DAEMON_REGISTRY.md) - Background daemon reference
- [HEALTH_MONITORING.md](./HEALTH_MONITORING.md) - Health check architecture
- [SYNC_TRANSFER.md](./SYNC_TRANSFER.md) - Data sync protocols
- [EVENT_SYSTEM_REFERENCE.md](../EVENT_SYSTEM_REFERENCE.md) - Event type catalog
