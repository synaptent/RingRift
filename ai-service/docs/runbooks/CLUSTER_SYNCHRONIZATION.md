# Cluster Synchronization Runbook

**Last Updated**: December 28, 2025
**Version**: Wave 7

## Overview

RingRift uses a multi-tier synchronization architecture to distribute data across the training cluster. This includes game databases, trained models, NPZ training files, and Elo ratings.

## Sync Architecture

```
Generator Nodes (selfplay)
        |
        | push-from-generator (AUTO_SYNC)
        v
Aggregator Nodes
        |
        | gossip replication
        v
All Cluster Nodes
        |
        | priority sync (SyncRouter)
        v
Training Nodes (for active training)
```

### Key Components

| Component         | Location                               | Purpose                                        |
| ----------------- | -------------------------------------- | ---------------------------------------------- |
| `AutoSyncDaemon`  | `app/coordination/auto_sync_daemon.py` | Automated P2P data sync                        |
| `SyncRouter`      | `app/coordination/sync_router.py`      | Intelligent routing based on node capabilities |
| `SyncFacade`      | `app/coordination/sync_facade.py`      | Unified programmatic entry point               |
| `SyncCoordinator` | `app/distributed/sync_coordinator.py`  | P2P orchestrator sync layer                    |

## Bandwidth Limits by Provider

Bandwidth limits prevent saturating network links:

| Provider | Limit    | Notes                 |
| -------- | -------- | --------------------- |
| RunPod   | 100 MB/s | Fast interconnect     |
| Nebius   | 100 MB/s | H100 backbone         |
| Vast.ai  | 50 MB/s  | Variable connectivity |
| Vultr    | 50 MB/s  | vGPU instances        |
| Lambda   | 100 MB/s | GH200 high bandwidth  |
| Hetzner  | 25 MB/s  | CPU-only voters       |

Configure via `config/distributed_hosts.yaml`:

```yaml
hosts:
  vast-12345:
    bandwidth_mbps: 50
```

## Sync Triggers

### Automatic Triggers

| Trigger                    | Condition                  | Priority |
| -------------------------- | -------------------------- | -------- |
| Training activity detected | Node starts training       | HIGH     |
| Orphan games detected      | Games on unreachable nodes | HIGH     |
| Periodic sync              | Every 60 seconds           | NORMAL   |
| New games available        | >100 new games             | NORMAL   |
| Model promoted             | New canonical model        | HIGH     |

### Manual Triggers

```bash
# Trigger immediate sync
curl -X POST http://localhost:8770/admin/sync/trigger

# Trigger priority sync for specific config
curl -X POST http://localhost:8770/admin/sync/priority \
  -d '{"config_key": "hex8_2p", "reason": "urgent_training"}'
```

## Data Flow

```
selfplay
    |
    v
data/games/selfplay_*.db (local)
    |
    | AUTO_SYNC pushes to peers
    v
data/games/ on aggregator nodes
    |
    | consolidation (scheduled)
    v
data/games/canonical_*.db
    |
    | export_replay_dataset.py
    v
data/training/*.npz
    |
    | NPZ_DISTRIBUTION to training nodes
    v
Training nodes receive data
    |
    | train.py
    v
models/canonical_*.pth
    |
    | MODEL_DISTRIBUTION to all nodes
    v
Cluster-wide model availability
```

## Troubleshooting Sync Issues

### 1. Sync Stalls

**Symptoms**: No sync activity, DATA_SYNC_COMPLETED events not firing

**Diagnosis**:

```bash
curl -s http://localhost:8770/status | jq '.sync_status'
```

**Resolution**:

1. Check AUTO_SYNC daemon status
2. Verify network connectivity to peers
3. Check disk space on source and destination
4. Review logs: `grep "AUTO_SYNC" logs/coordination.log`

### 2. Sync Failures

**Symptoms**: SYNC_FAILED events, partial data

**Diagnosis**:

```bash
# Check sync errors
curl -s http://localhost:8770/status | jq '.sync_stats.errors'
```

**Resolution**:

1. Check rsync exit codes in logs
2. Verify SSH key access between nodes
3. Check for disk full conditions
4. Try manual rsync to isolate issue:
   ```bash
   rsync -avz --progress data/games/ user@peer:/data/games/
   ```

### 3. Bandwidth Saturation

**Symptoms**: Slow syncs, timeouts, dropped connections

**Diagnosis**:

```bash
# Check current bandwidth usage
iftop -i eth0
```

**Resolution**:

1. Reduce `bandwidth_mbps` in distributed_hosts.yaml
2. Enable backpressure: Check `BACKPRESSURE_ACTIVATED` events
3. Stagger sync times between nodes

### 4. Data Inconsistency

**Symptoms**: Different game counts on different nodes

**Diagnosis**:

```bash
# Check game counts across cluster
curl -s http://localhost:8770/manifest | jq '.nodes | to_entries | .[] | {node: .key, games: .value.total_games}'
```

**Resolution**:

1. Trigger full cluster sync
2. Use `consolidate_jsonl_databases.py` to merge scattered data
3. Verify database integrity: `python scripts/validate_databases.py`

## Sync Strategies

### Push-from-Generator

Default strategy for selfplay nodes:

- Generates data pushes to configured aggregators
- Reduces load on central nodes
- Enables horizontal scaling

### Gossip Replication

Secondary strategy for data spread:

- Each node periodically syncs with random peers
- Eventually consistent across cluster
- Resilient to node failures

### Priority Sync

Triggered for urgent data needs:

- Training nodes request data immediately
- Bypasses normal queue
- Used for orphan game recovery

## Configuration

### Auto-Sync Settings

```yaml
# config/distributed_hosts.yaml
sync_routing:
  max_disk_usage_percent: 70.0
  priority_hosts:
    - nebius-backbone-1
    - runpod-h100
  exclude_from_sync:
    - mac-studio # Coordinator only
```

### Environment Variables

| Variable                   | Default | Description                  |
| -------------------------- | ------- | ---------------------------- |
| `RINGRIFT_SYNC_INTERVAL`   | 60      | Seconds between sync cycles  |
| `RINGRIFT_SYNC_BATCH_SIZE` | 100     | Max files per sync batch     |
| `RINGRIFT_SYNC_TIMEOUT`    | 300     | Sync operation timeout (sec) |
| `RINGRIFT_SYNC_DRY_RUN`    | false   | Log without executing        |

## Programmatic Access

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()

# Trigger sync
response = await facade.trigger_sync(
    source_node="vast-12345",
    config_key="hex8_2p",
)

# Trigger priority sync
response = await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_2p",
    data_type="games",
)

# Get sync status
status = facade.get_sync_status()
```

## Events

### Emitted Events

| Event                 | When          | Payload                           |
| --------------------- | ------------- | --------------------------------- |
| `DATA_SYNC_STARTED`   | Sync begins   | sync_type, target_nodes           |
| `DATA_SYNC_COMPLETED` | Sync succeeds | sync_type, duration, files_synced |
| `DATA_SYNC_FAILED`    | Sync fails    | error, sync_type                  |
| `SYNC_STALLED`        | No progress   | timeout_seconds                   |

### Subscribed Events

| Event                   | Handler                  | Action                         |
| ----------------------- | ------------------------ | ------------------------------ |
| `TRAINING_STARTED`      | AutoSyncDaemon           | Priority sync to training node |
| `ORPHAN_GAMES_DETECTED` | DataPipelineOrchestrator | Trigger orphan recovery sync   |
| `NODE_RECOVERED`        | SyncRouter               | Include node in sync targets   |

## See Also

- [P2P_ORCHESTRATOR_OPERATIONS.md](P2P_ORCHESTRATOR_OPERATIONS.md) - P2P cluster
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event routing
