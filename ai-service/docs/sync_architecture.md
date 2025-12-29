# Sync Architecture Reference

**Date:** December 2025
**Status:** Production

This document describes the data synchronization architecture for the RingRift AI training cluster.

---

## Overview

The sync system handles distribution of training data across a heterogeneous cluster of ~30-50 GPU nodes. It manages three types of data:

| Data Type | Description                       | Size      | Sync Priority |
| --------- | --------------------------------- | --------- | ------------- |
| `games`   | SQLite game replay databases      | 1-50GB    | High          |
| `models`  | Neural network checkpoints (.pth) | 100-200MB | Critical      |
| `npz`     | Training data archives            | 500MB-5GB | High          |

---

## Architecture Evolution (8→3 Consolidation)

### Historical Implementations (Deprecated)

The codebase evolved 8 different sync implementations over 2024-2025:

| Implementation          | Status             | Replaced By                          |
| ----------------------- | ------------------ | ------------------------------------ |
| `SyncScheduler`         | Deprecated Q2 2026 | AutoSyncDaemon                       |
| `UnifiedDataSync`       | Deprecated Q2 2026 | SyncFacade                           |
| `SyncOrchestrator`      | Deprecated Q2 2026 | AutoSyncDaemon                       |
| `ClusterDataSyncDaemon` | Absorbed Dec 2025  | AutoSyncDaemon(strategy="broadcast") |
| `EphemeralSyncDaemon`   | Absorbed Dec 2025  | AutoSyncDaemon(strategy="ephemeral") |

### Current Architecture (December 2025)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SyncFacade                                   │
│                    (Unified Entry Point)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌───────────────┐    ┌───────────────┐    ┌───────────────────┐  │
│    │  SyncRouter   │    │AutoSyncDaemon │    │ SyncCoordinator   │  │
│    │  (Routing)    │ ── │   (Daemon)    │ ── │ (Transport)       │  │
│    └───────────────┘    └───────────────┘    └───────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Rsync   │ │   HTTP   │ │ Gossip   │
              └──────────┘ └──────────┘ └──────────┘
```

**Three Active Components:**

| Component        | File                  | Purpose                                        |
| ---------------- | --------------------- | ---------------------------------------------- |
| `SyncFacade`     | `sync_facade.py`      | Single entry point for all sync operations     |
| `AutoSyncDaemon` | `auto_sync_daemon.py` | Background P2P sync with multiple strategies   |
| `SyncRouter`     | `sync_router.py`      | Intelligent routing based on node capabilities |

**Supporting Component:**

| Component         | File                                  | Purpose                       |
| ----------------- | ------------------------------------- | ----------------------------- |
| `SyncCoordinator` | `app/distributed/sync_coordinator.py` | Low-level transport execution |

---

## Sync Strategies

AutoSyncDaemon supports four strategies:

### 1. HYBRID (Default)

Combines push-from-generator with P2P gossip for optimal coverage:

```
Layer 1: Push-from-generator
  ├── Immediate push to 2-3 neighbors on game completion
  └── Low latency for fresh data

Layer 2: P2P Gossip
  ├── Background replication for eventual consistency
  └── Handles missed pushes and new nodes
```

### 2. EPHEMERAL

Aggressive sync for termination-prone nodes (Vast.ai):

- 30-second sync interval (vs 5-minute default)
- Priority sync before termination
- Targets persistent nodes first

### 3. BROADCAST

Push-based sync to all training nodes:

- Full replication to all GPU nodes
- Used for model distribution
- Higher bandwidth cost

### 4. AUTO

Selects strategy based on node type and data type:

| Node Type   | Games     | Models    | NPZ       |
| ----------- | --------- | --------- | --------- |
| Ephemeral   | EPHEMERAL | BROADCAST | EPHEMERAL |
| Training    | HYBRID    | BROADCAST | HYBRID    |
| Coordinator | SKIP      | SKIP      | SKIP      |

---

## Usage

### Programmatic Sync

```python
from app.coordination.sync_facade import SyncFacade, sync

# Get facade instance
facade = SyncFacade.get_instance()

# Sync games to specific nodes
await facade.sync(
    data_type="games",
    targets=["node-1", "node-2"],
    board_type="hex8",
    priority="high"
)

# Convenience function - sync all models
await sync("models", targets=["all"])

# Priority sync for orphan recovery
await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_2p",
)
```

### Daemon Usage

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon, SyncStrategy

# Default hybrid strategy
daemon = AutoSyncDaemon()
await daemon.start()

# Ephemeral strategy for Vast.ai nodes
daemon = AutoSyncDaemon(strategy=SyncStrategy.EPHEMERAL)
await daemon.start()
```

### CLI Usage

```bash
# Sync via master_loop.py (recommended)
python scripts/master_loop.py

# Direct sync operations
python scripts/unified_data_sync.py --data-type games --board-type hex8
```

---

## Routing Algorithm

SyncRouter uses a multi-factor routing algorithm:

### Step 1: Node Eligibility

```python
def should_sync_to_node(node_id, data_type) -> bool:
    # Check exclusion rules
    if node_id in COORDINATOR_NODES:
        return False  # Coordinators don't receive synced data

    # Check disk capacity
    if get_disk_usage(node_id) > 70%:
        return False

    # Check NFS sharing
    if shares_nfs_with(source_node, node_id):
        return False  # Already have the data

    return True
```

### Step 2: Priority Scoring

| Factor        | Weight | Description                             |
| ------------- | ------ | --------------------------------------- |
| Training node | +50    | Active training jobs need fresh data    |
| Ephemeral     | +30    | Prioritize before potential termination |
| Disk headroom | +20    | Nodes with more free space              |
| Last sync age | +10    | Nodes that haven't synced recently      |
| Quality score | +10    | Higher quality data gets priority       |

### Step 3: Bandwidth Allocation

Bandwidth limits by provider (from `distributed_hosts.yaml`):

| Provider | Limit    | Reason                         |
| -------- | -------- | ------------------------------ |
| RunPod   | 100 MB/s | High-bandwidth datacenter      |
| Nebius   | 100 MB/s | High-bandwidth datacenter      |
| Vast.ai  | 50 MB/s  | Residential/varied connections |
| Vultr    | 75 MB/s  | Mid-tier datacenter            |
| Hetzner  | 25 MB/s  | CPU-only, lower priority       |

---

## Event Integration

### Events Emitted

| Event                 | Trigger                 | Payload                             |
| --------------------- | ----------------------- | ----------------------------------- |
| `DATA_SYNC_STARTED`   | Sync operation begins   | `{sync_type, targets, data_type}`   |
| `DATA_SYNC_COMPLETED` | Sync operation succeeds | `{sync_type, duration, file_count}` |
| `DATA_SYNC_FAILED`    | Sync operation fails    | `{error, target_node}`              |

### Events Subscribed

| Event                   | Handler        | Action                           |
| ----------------------- | -------------- | -------------------------------- |
| `TRAINING_STARTED`      | SyncRouter     | Prioritize sync to training node |
| `NODE_RECOVERED`        | AutoSyncDaemon | Trigger catch-up sync            |
| `ORPHAN_GAMES_DETECTED` | SyncFacade     | Trigger priority sync            |

---

## Fault Tolerance

### Circuit Breaker

Each node has an independent circuit breaker:

```
States: CLOSED → OPEN → HALF_OPEN → CLOSED

CLOSED:     Normal operation
OPEN:       Node unreachable, skip for recovery_timeout (default 60s)
HALF_OPEN:  Test with single request
```

### Fallback Chain

```
1. Tailscale (private network)
   ↓ fail
2. SSH/rsync (direct connection)
   ↓ fail
3. HTTP (data server)
   ↓ fail
4. Queue for retry
```

### Integrity Checks

Before syncing, databases are validated:

```python
# Check SQLite integrity
check_sqlite_integrity(db_path)

# Check write lock (no active writers)
is_database_safe_to_sync(db_path)

# Verify game count matches source
verify_game_count(db_path, expected_count)
```

---

## Monitoring

### Health Check

```python
daemon = AutoSyncDaemon.get_instance()
health = daemon.health_check()
# Returns: HealthCheckResult(healthy=True, status=RUNNING, ...)
```

### Metrics

| Metric                   | Description             |
| ------------------------ | ----------------------- |
| `sync_operations_total`  | Total sync operations   |
| `sync_bytes_transferred` | Total bytes transferred |
| `sync_duration_seconds`  | Sync operation duration |
| `sync_failures_total`    | Failed sync attempts    |
| `sync_queue_depth`       | Pending sync operations |

### Status Endpoint

```bash
curl http://localhost:8790/status | jq '.sync'
```

---

## Configuration

### Environment Variables

| Variable                            | Default | Description                    |
| ----------------------------------- | ------- | ------------------------------ |
| `RINGRIFT_DATA_SYNC_INTERVAL`       | 120     | Games sync interval (seconds)  |
| `RINGRIFT_FAST_SYNC_INTERVAL`       | 30      | Fast sync interval (seconds)   |
| `RINGRIFT_SYNC_TIMEOUT`             | 300     | Sync timeout (seconds)         |
| `RINGRIFT_MIN_SYNC_INTERVAL`        | 2.0     | Minimum auto-sync interval     |
| `RINGRIFT_AUTO_SYNC_MAX_CONCURRENT` | 6       | Max concurrent auto-sync tasks |

### distributed_hosts.yaml

```yaml
auto_sync:
  enabled: true
  interval_seconds: 60
  gossip_interval_seconds: 15
  max_concurrent_syncs: 12
  bandwidth_limit_mbps: 100
  host_bandwidth_overrides:
    runpod-*: 100
    nebius-*: 100

sync_routing:
  priority_hosts:
    - runpod-h100
    - nebius-h100-1
```

---

## Migration Guide

### From SyncScheduler

```python
# Before (deprecated)
from app.coordination.sync_scheduler import SyncScheduler
scheduler = SyncScheduler()
await scheduler.schedule_sync(targets)

# After
from app.coordination.sync_facade import sync
await sync("games", targets=targets)
```

### Broadcast Strategy (replaces ClusterDataSyncDaemon)

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon, SyncStrategy
daemon = AutoSyncDaemon(strategy=SyncStrategy.BROADCAST)
```

Note: `cluster_data_sync.py` was removed in Dec 2025; the broadcast strategy lives in `auto_sync_daemon.py`.

### Ephemeral Strategy (replaces EphemeralSyncDaemon)

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon, SyncStrategy
daemon = AutoSyncDaemon(strategy=SyncStrategy.EPHEMERAL)
```

Note: `ephemeral_sync.py` was removed in Dec 2025; the ephemeral strategy lives in `auto_sync_daemon.py`.

---

## Files Reference

| File                       | LOC    | Purpose                |
| -------------------------- | ------ | ---------------------- |
| `sync_facade.py`           | ~600   | Unified entry point    |
| `auto_sync_daemon.py`      | ~3,600 | Main sync daemon       |
| `sync_router.py`           | ~800   | Routing logic          |
| `sync_bandwidth.py`        | ~700   | Bandwidth management   |
| `database_sync_manager.py` | ~670   | Base class for DB sync |

### Deprecated/Removed

| File                             | Replacement / Status                                      |
| -------------------------------- | --------------------------------------------------------- |
| `sync_coordinator.py`            | AutoSyncDaemon (scheduling) + distributed SyncCoordinator |
| `cluster_data_sync.py` (removed) | AutoSyncDaemon(BROADCAST)                                 |
| `ephemeral_sync.py` (removed)    | AutoSyncDaemon(EPHEMERAL)                                 |

---

## Troubleshooting

### "Sync queue backed up"

1. Check network connectivity: `curl -s http://node:8770/health`
2. Check disk usage: `df -h /data`
3. Check circuit breaker state in logs

### "Sync taking too long"

1. Reduce batch size: `RINGRIFT_SYNC_BATCH_SIZE=5`
2. Check bandwidth limits in distributed_hosts.yaml
3. Enable parallel transfers: `RINGRIFT_SYNC_PARALLEL=4`

### "Missing data on training node"

1. Check node is not in exclusion list
2. Verify NFS sharing detection
3. Trigger manual sync: `await sync("games", targets=["node-id"])`

---

_Last Updated: December 27, 2025_
