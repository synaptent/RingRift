# Sync Strategy Guide

Guide to choosing the right sync strategy for RingRift cluster data synchronization.

## Quick Reference

| Strategy      | Interval  | Use Case                          | Data Guarantee        |
| ------------- | --------- | --------------------------------- | --------------------- |
| **HYBRID**    | 60s       | Persistent hosts (Lambda, RunPod) | Eventual consistency  |
| **EPHEMERAL** | 5s        | Spot instances (Vast.ai)          | Write-through + WAL   |
| **BROADCAST** | 60s       | Leader-initiated distribution     | Push to all           |
| **PULL**      | On-demand | Coordinator recovery/backup       | Pull from workers     |
| **AUTO**      | -         | Auto-detect node type             | Selects best strategy |

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

## Configuration

### AutoSyncConfig

The `AutoSyncConfig` dataclass controls sync daemon behavior:

```python
from app.coordination.sync_strategies import AutoSyncConfig, SyncStrategy

config = AutoSyncConfig(
    strategy=SyncStrategy.HYBRID,

    # Timing
    sync_interval_seconds=60,          # Default sync interval
    ephemeral_sync_interval_seconds=5, # Interval for ephemeral strategy

    # Filters
    min_quality_score=0.0,             # Skip nodes below this quality
    exclude_nfs_hosts=True,            # Skip NFS-shared hosts
    exclude_coordinators=True,         # Skip coordinator nodes

    # Bandwidth
    bandwidth_limit_mbps=50,           # Per-host bandwidth limit
    max_concurrent_syncs=5,            # Max parallel transfers

    # Ephemeral settings
    ephemeral_max_data_age_seconds=300,  # Max age before sync
    ephemeral_priority_boost=2.0,        # Priority multiplier
)
```

### Transfer Priorities

Sync operations use priority levels for bandwidth allocation:

| Priority     | Multiplier | Use Case                           |
| ------------ | ---------- | ---------------------------------- |
| `CRITICAL`   | 2.0x       | Model distribution after promotion |
| `HIGH`       | 1.5x       | Training data sync                 |
| `NORMAL`     | 1.0x       | Regular selfplay sync              |
| `LOW`        | 0.5x       | Elo/registry databases             |
| `BACKGROUND` | 0.25x      | Archive operations                 |

```python
from app.coordination.sync_bandwidth import TransferPriority, get_bandwidth_manager

manager = get_bandwidth_manager()
manager.set_host_priority("training-node", TransferPriority.HIGH)
```

## Failure Recovery

### Transport Escalation

When a transfer fails, the system escalates through transports:

```
P2P HTTP → SSH/rsync → Base64 encoding
    │          │              │
    └─ Fast    └─ Reliable    └─ Universal fallback
       (8765)     (22)           (works through firewalls)
```

Each transport has configurable timeouts:

| Transport | Connect Timeout | Transfer Timeout | Retry Count |
| --------- | --------------- | ---------------- | ----------- |
| P2P HTTP  | 5s              | 120s             | 2           |
| SSH/rsync | 10s             | 300s             | 3           |
| Base64    | 30s             | 600s             | 1           |

### Circuit Breaker

Per-host circuit breakers prevent cascading failures:

```
CLOSED ──(3 failures)──> OPEN ──(60s timeout)──> HALF_OPEN
   ▲                                                  │
   └──────────────(success)───────────────────────────┘
```

```python
from app.coordination.transport_base import CircuitBreakerConfig

# Aggressive (fail fast)
config = CircuitBreakerConfig.aggressive()  # 2 failures, 30s timeout

# Patient (more tolerant)
config = CircuitBreakerConfig.patient()     # 5 failures, 120s timeout
```

### Checksum Verification

Transfers include SHA256 verification:

```python
from app.coordination.sync_bandwidth import BandwidthCoordinatedRsync

rsync = BandwidthCoordinatedRsync()
result = await rsync.sync_with_checksum(
    source="/data/games/canonical_hex8.db",
    dest="ubuntu@training:/data/games/",
    host="training-node",
    verify_checksum=True,  # Verify after transfer
)
```

## Environment Variables

| Variable                             | Default | Description                     |
| ------------------------------------ | ------- | ------------------------------- |
| `RINGRIFT_SYNC_INTERVAL`             | 60      | Default sync interval (seconds) |
| `RINGRIFT_EPHEMERAL_SYNC_INTERVAL`   | 5       | Ephemeral sync interval         |
| `RINGRIFT_MAX_CONCURRENT_SYNCS`      | 5       | Max parallel transfers          |
| `RINGRIFT_SYNC_BANDWIDTH_LIMIT`      | 50      | Default bandwidth (MB/s)        |
| `RINGRIFT_SYNC_STRATEGY`             | hybrid  | Default strategy                |
| `RINGRIFT_SYNC_EXCLUDE_COORDINATORS` | true    | Exclude coordinator nodes       |

## Troubleshooting

### Sync Not Starting

1. Check daemon status: `python scripts/launch_daemons.py --status`
2. Verify event router running: `DaemonType.EVENT_ROUTER` must be active
3. Check logs: `tail -f logs/auto_sync.log`

### Slow Transfers

1. Check bandwidth limits: `get_host_bandwidth_limit("node-name")`
2. Use BatchRsync for multiple files
3. Consider priority routing to skip slow nodes
4. Check circuit breaker state:
   ```python
   from app.coordination.sync_router import get_sync_router
   router = get_sync_router()
   status = router.get_node_status("slow-node")
   print(f"Circuit: {status.circuit_state}")
   ```

### Data Not Appearing

1. Verify source has data: `ls data/games/`
2. Check manifest: `python -c "from app.distributed.cluster_manifest import get_manifest; print(get_manifest().get_data_locations('hex8_2p'))"`
3. Trigger manual sync: `python scripts/unified_data_sync.py --force`

### Transfer Failures

1. Check transport availability:

   ```python
   from app.coordination.sync_router import get_sync_router
   router = get_sync_router()
   caps = router.get_node_capabilities("node-name")
   print(f"P2P: {caps.p2p_available}, SSH: {caps.ssh_available}")
   ```

2. Force transport escalation:

   ```bash
   # Skip P2P, use rsync directly
   python scripts/unified_data_sync.py --transport rsync

   # Use base64 fallback
   python scripts/unified_data_sync.py --transport base64
   ```

3. Check for firewall issues:

   ```bash
   # Test P2P port
   curl -s http://node:8765/health

   # Test SSH
   ssh -o ConnectTimeout=5 user@node "echo ok"
   ```

### Backpressure Issues

When nodes are overloaded, sync pauses automatically:

```python
from app.coordination.sync_router import get_sync_router

router = get_sync_router()
if router.is_under_backpressure():
    print("Sync paused due to backpressure")
    # Wait for BACKPRESSURE_RELEASED event
```

Clear backpressure manually:

```python
from app.coordination.event_router import get_event_bus, DataEventType

bus = get_event_bus()
bus.publish(DataEventType.BACKPRESSURE_RELEASED, {"source": "manual"})
```

## See Also

- `app/coordination/auto_sync_daemon.py` - Background sync daemon
- `app/coordination/sync_router.py` - Quality-based routing
- `app/coordination/sync_bandwidth.py` - Bandwidth coordination
- `app/coordination/sync_strategies.py` - Strategy definitions
- `app/coordination/transport_base.py` - Transport infrastructure
- `scripts/p2p/managers/sync_planner.py` - P2P sync planning
- `docs/runbooks/DAEMON_FAILURE_RECOVERY.md` - Daemon troubleshooting
