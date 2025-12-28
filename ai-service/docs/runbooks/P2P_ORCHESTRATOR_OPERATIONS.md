# P2P Orchestrator Operations Runbook

**Last Updated**: December 28, 2025
**Version**: Wave 7

## Overview

The P2P orchestrator (`scripts/p2p_orchestrator.py`) is the heart of the RingRift distributed training cluster. It runs on every node and handles:

- Leader election and cluster coordination
- Job dispatch (selfplay, training, tournaments)
- Data synchronization across nodes
- Health monitoring and auto-recovery

## Health Check Endpoints

All endpoints are served on port 8770 (configurable via `--port`).

| Endpoint   | Method | Purpose             | Response                            |
| ---------- | ------ | ------------------- | ----------------------------------- |
| `/status`  | GET    | Full cluster status | JSON with role, peers, jobs, health |
| `/health`  | GET    | Liveness probe      | `{"status": "ok"}` or error         |
| `/ready`   | GET    | Readiness probe     | `{"ready": true/false}`             |
| `/metrics` | GET    | Prometheus metrics  | Text format metrics                 |

### Status Endpoint Fields

```json
{
  "node_id": "nebius-backbone-1",
  "role": "leader",
  "leader_id": "nebius-backbone-1",
  "alive_peers": 28,
  "total_peers": 32,
  "uptime_seconds": 3600,
  "selfplay_jobs": 12,
  "training_jobs": 1,
  "event_subscription_status": {
    "daemon_events": true,
    "feedback_signals": true,
    "manager_events": true,
    "all_healthy": true,
    "critical_failed": []
  },
  "loops": {
    "job_reaper": { "enabled": true, "last_run": 1735400000 },
    "idle_detection": { "enabled": true, "last_run": 1735400030 },
    "auto_scaling": { "enabled": true, "last_run": 1735400060 }
  }
}
```

## Manager Lifecycle

The P2P orchestrator delegates to 7 specialized managers:

| Manager               | Purpose                            | Key Methods                                    |
| --------------------- | ---------------------------------- | ---------------------------------------------- |
| `StateManager`        | SQLite persistence, epochs         | `load_state()`, `save_state()`                 |
| `JobManager`          | Job spawning and lifecycle         | `run_gpu_selfplay_job()`, `spawn_training()`   |
| `TrainingCoordinator` | Training dispatch, promotion       | `dispatch_training_job()`, `check_readiness()` |
| `SelfplayScheduler`   | Priority-based config selection    | `pick_weighted_config()`, `get_target_jobs()`  |
| `NodeSelector`        | Node ranking for job dispatch      | `get_best_gpu_node()`, `get_training_nodes()`  |
| `SyncPlanner`         | Manifest collection, sync planning | `collect_manifest()`, `create_sync_plan()`     |
| `LoopManager`         | Background loop coordination       | `start_all()`, `stop_all()`                    |

## Background Loops

The LoopManager coordinates 13 background loops:

| Loop                       | Interval | Purpose                                   |
| -------------------------- | -------- | ----------------------------------------- |
| `JobReaperLoop`            | 5 min    | Clean stale jobs (1hr), stuck jobs (2hr)  |
| `IdleDetectionLoop`        | 30 sec   | Detect idle GPUs, trigger selfplay        |
| `WorkerPullLoop`           | 30 sec   | Workers poll leader for work              |
| `WorkQueueMaintenanceLoop` | 5 min    | Cleanup timeouts, old items (24hr)        |
| `SelfHealingLoop`          | 5 min    | Recover stuck jobs, clean stale processes |
| `PredictiveMonitoringLoop` | 5 min    | Track trends, emit pre-threshold alerts   |
| `AutoScalingLoop`          | 5 min    | Scale cluster up/down based on workload   |
| `EloSyncLoop`              | 5 min    | Synchronize Elo ratings across nodes      |
| `QueuePopulatorLoop`       | 1 min    | Maintain work queue until Elo targets met |
| `NATManagementLoop`        | 5 min    | STUN probing, relay selection             |
| `ManifestCollectionLoop`   | 5 min    | Collect data manifests from cluster       |
| `DataManagementLoop`       | 5 min    | Pipeline: export, convert, train          |
| `HeartbeatLoop`            | 15 sec   | Peer discovery and health                 |

## Common Failure Modes

### 1. Leader Election Stalls

**Symptoms**: No leader, cluster operations blocked

**Diagnosis**:

```bash
curl -s http://localhost:8770/status | jq '.leader_id, .role'
```

**Resolution**:

1. Check voter quorum: Need 3/5 voters alive
2. Check network connectivity between voters
3. Force election: `curl -X POST http://localhost:8770/admin/election/start`

### 2. Event Subscriptions Failed

**Symptoms**: Pipeline stalls, no reaction to events

**Diagnosis**:

```bash
curl -s http://localhost:8770/status | jq '.event_subscription_status'
```

**Resolution**:

1. Check if `manager_events` is false - critical subscriptions failed
2. Restart P2P orchestrator
3. Check event_router availability: `python -c "from app.coordination.event_router import subscribe; print('OK')"`

### 3. Jobs Stuck in CLAIMED State

**Symptoms**: Jobs not progressing, workers idle

**Diagnosis**:

```bash
curl -s http://localhost:8770/status | jq '.work_queue.by_status'
```

**Resolution**:

1. JobReaperLoop should auto-reset after 2 hours
2. Manual reset: `curl -X POST http://localhost:8770/admin/jobs/reset-stale`
3. Check worker node connectivity

### 4. Node Not Joining Cluster

**Symptoms**: Node not appearing in peer list

**Diagnosis**:

```bash
# On the node
curl -s http://localhost:8770/status | jq '.known_peers, .alive_peers'
```

**Resolution**:

1. Check firewall: Port 8770 must be open
2. Verify peer seed: `--peers <leader-ip>:8770`
3. Check Tailscale connectivity if using mesh

## Scaling Operations

### Manual Scale Up

```bash
# Add new node to cluster
python scripts/p2p_orchestrator.py \
  --node-id new-gpu-node \
  --peers <leader-ip>:8770 \
  --port 8770
```

### Manual Scale Down

```bash
# Graceful shutdown
curl -X POST http://localhost:8770/admin/shutdown

# Or kill process (will be detected as dead peer)
pkill -f p2p_orchestrator.py
```

### Auto-Scaling Status

```bash
curl -s http://localhost:8770/status | jq '.loops.auto_scaling'
```

Auto-scaling uses **reluctant termination**:

- Nodes only terminated after 1+ hour idle AND 5+ consecutive health failures
- Minimum 2 nodes always maintained
- 5-minute cooldown between terminations

## Event Subscription Debugging

### Check Subscription Status

```bash
curl -s http://localhost:8770/status | jq '.event_subscription_status'
```

### Critical Events

These events must be subscribed for pipeline to function:

| Event                  | Purpose                           |
| ---------------------- | --------------------------------- |
| `DATA_SYNC_COMPLETED`  | Triggers NPZ export after sync    |
| `TRAINING_COMPLETED`   | Triggers evaluation and promotion |
| `EVALUATION_COMPLETED` | Triggers curriculum rebalancing   |

### Force Subscription Failure on Startup

```bash
RINGRIFT_FAIL_ON_SUBSCRIPTION_FAILURE=true python scripts/p2p_orchestrator.py ...
```

This will crash on startup if critical subscriptions fail (useful for CI/CD).

## Operational Commands

### Check Cluster Health

```bash
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive: {d.get(\"alive_peers\")} / {d.get(\"total_peers\")}")
print(f"Jobs: selfplay={d.get(\"selfplay_jobs\")}, training={d.get(\"training_jobs\")}")
'
```

### Trigger Manual Sync

```bash
curl -X POST http://localhost:8770/admin/sync/trigger
```

### Get Work Queue Stats

```bash
curl -s http://localhost:8770/status | jq '.work_queue'
```

### List Active Jobs

```bash
curl -s http://localhost:8770/jobs | jq '.active_jobs'
```

## Logging

Log level controlled via `RINGRIFT_LOG_LEVEL` (default: INFO).

Important log patterns:

- `[P2P] Leader elected` - Leadership change
- `[P2P] Event subscriptions` - Subscription status
- `[P2P] CRITICAL` - Critical failures
- `[P2P] Sync completed` - Data sync events

## Environment Variables

| Variable                                | Default | Description                           |
| --------------------------------------- | ------- | ------------------------------------- |
| `RINGRIFT_P2P_PORT`                     | 8770    | HTTP API port                         |
| `RINGRIFT_LOG_LEVEL`                    | INFO    | Log verbosity                         |
| `RINGRIFT_FAIL_ON_SUBSCRIPTION_FAILURE` | false   | Crash on subscription failure         |
| `RINGRIFT_MIN_NODES`                    | 2       | Minimum nodes for auto-scaling        |
| `RINGRIFT_MAX_NODES`                    | 50      | Maximum nodes for auto-scaling        |
| `RINGRIFT_AUTOSCALE_DRY_RUN`            | false   | Log scaling actions without executing |

## See Also

- [DAEMON_MANAGER_OPERATIONS.md](DAEMON_MANAGER_OPERATIONS.md) - Daemon lifecycle
- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Data sync
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event routing
