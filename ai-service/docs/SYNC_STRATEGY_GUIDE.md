# Sync Strategy Guide

Guide to choosing the right sync strategy for RingRift cluster data synchronization.

## Architecture Overview

The sync system has multiple layers, each with a specific purpose:

```
┌─────────────────────────────────────────────────────────────┐
│                    User/Automation Entry Points              │
│  master_loop.py | run_training_loop.py | scripts/sync*.py   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     SyncFacade (app/coordination)            │
│  Unified programmatic API - routes to appropriate strategy   │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌──────▼──────┐
│ AutoSyncDaemon│   │   SyncRouter    │   │ SyncPlanner │
│ Background    │   │ Quality-based   │   │ P2P Layer   │
│ daemon        │   │ routing         │   │ Manifests   │
└───────────────┘   └─────────────────┘   └─────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│               sync_bandwidth.py (Bandwidth Coordination)     │
│  Per-host limits | Rsync wrapper | BatchRsync for efficiency │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              app/distributed/sync_coordinator.py             │
│  LOW-LEVEL: Actual rsync/scp execution, file transfers       │
└─────────────────────────────────────────────────────────────┘
```

## When to Use What

### For Scripts (CLI usage)

```bash
# Full automation (recommended)
python scripts/master_loop.py

# Manual sync of specific data type
python scripts/unified_data_sync.py --type games --config hex8_2p
```

### For Application Code

```python
# Option 1: SyncFacade (recommended for most cases)
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()
result = await facade.sync_config("hex8_2p", data_types=["games", "npz"])

# Option 2: Direct daemon control
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon(strategy="broadcast")
await daemon.start()  # Background sync every 60s

# Option 3: Event-triggered sync
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()
await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
)
```

## Sync Strategies

### broadcast (default)

Sync to all available nodes. Used when:

- Training data needs to be everywhere
- Model distribution after promotion
- Database consolidation

```python
AutoSyncDaemon(strategy="broadcast")
```

### ephemeral

Aggressive sync for Vast.ai/RunPod nodes that may terminate. Used when:

- Node may be preempted
- Data generated on spot instances
- Short 5-second sync interval

```python
AutoSyncDaemon(strategy="ephemeral")
```

### priority

Target specific high-value nodes first. Used when:

- Training about to start
- Need fresh data on specific nodes
- Coordinator sync

```python
SyncRouter.get_sync_targets(priority_hosts=["runpod-h100", "nebius-h100-1"])
```

### quality-based (via SyncRouter)

Routes to nodes with best connectivity/capacity. Used when:

- Large file transfers
- Minimizing transfer time
- Network-aware distribution

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()
targets = router.get_sync_targets(
    data_type="npz",
    config_key="square19_4p",
    max_targets=5,
)

# Reverse sync (pull) sources for coordinator ingestion
sources = router.get_sync_sources(
    data_type="games",
    target_node="coordinator-1",
    max_sources=3,
)
```

## Data Types

| Type       | Description            | Typical Size | Sync Priority |
| ---------- | ---------------------- | ------------ | ------------- |
| `games`    | SQLite game databases  | 10-500 MB    | HIGH          |
| `npz`      | Training data files    | 50-500 MB    | HIGH          |
| `models`   | Neural network weights | 30-170 MB    | MEDIUM        |
| `elo`      | Elo ratings database   | <1 MB        | LOW           |
| `registry` | Model registry         | <1 MB        | LOW           |

## Bandwidth Limits

Per-provider defaults (configured in `distributed_hosts.yaml`):

| Provider | Limit (MB/s) | Notes           |
| -------- | ------------ | --------------- |
| RunPod   | 100          | Premium network |
| Nebius   | 100          | Premium network |
| Vultr    | 75           | Good network    |
| Vast.ai  | 50           | Variable        |
| Hetzner  | 50           | CPU nodes       |

## Event Integration

Sync operations emit events for pipeline coordination:

```python
# Emitted events
DataEventType.DATA_SYNC_STARTED
DataEventType.DATA_SYNC_COMPLETED
DataEventType.DATA_SYNC_FAILED

# Subscribe to sync events
from app.coordination.event_router import get_router

router = get_router()
router.subscribe(DataEventType.DATA_SYNC_COMPLETED, my_handler)
```

## Troubleshooting

### Sync Not Starting

1. Check daemon status: `python scripts/launch_daemons.py --status`
2. Verify event router running: `DaemonType.EVENT_ROUTER` must be active
3. Check logs: `tail -f logs/auto_sync.log`

### Slow Transfers

1. Check bandwidth limits: `get_host_bandwidth_limit("node-name")`
2. Use BatchRsync for multiple files
3. Consider priority routing to skip slow nodes

### Data Not Appearing

1. Verify source has data: `ls data/games/`
2. Check manifest: `python -c "from app.distributed.cluster_manifest import get_manifest; print(get_manifest().get_data_locations('hex8_2p'))"`
3. Trigger manual sync: `python scripts/unified_data_sync.py --force`

## See Also

- `app/coordination/auto_sync_daemon.py` - Background sync daemon
- `app/coordination/sync_router.py` - Quality-based routing
- `app/coordination/sync_bandwidth.py` - Bandwidth coordination
- `scripts/p2p/managers/sync_planner.py` - P2P sync planning
