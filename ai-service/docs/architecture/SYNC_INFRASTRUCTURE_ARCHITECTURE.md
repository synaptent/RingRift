# Sync Infrastructure Architecture

**Last Updated:** December 28, 2025
**Status:** Production

## Overview

The RingRift AI service uses a multi-layer sync infrastructure to distribute game data, models, and training artifacts across a heterogeneous cluster of ~32 nodes (GPU nodes on Lambda, Vast.ai, RunPod, Nebius, and CPU nodes on Hetzner).

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SyncFacade (Unified API)                     │
│  Programmatic entry point for all sync operations                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│       AutoSyncDaemon          │   │        SyncRouter             │
│  Push-from-generator strategy │   │  Intelligent routing based on │
│  Gossip replication           │   │  node capabilities            │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Transport Layer                                   │
│  BandwidthCoordinatedRsync | SyncPushDaemon | P2P HTTP | SSH       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Layer                                        │
│  ClusterManifest | GameReplayDB | NPZ Training Files | Models       │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer Details

### 1. SyncFacade (Entry Point)

**File:** `app/coordination/sync_facade.py`

The unified programmatic interface for sync operations. All callers should use this facade rather than calling individual sync components directly.

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()

# Trigger priority sync for orphan recovery
await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_2p",
)

# Check sync status
status = await facade.get_sync_status()
```

**Responsibilities:**

- Route sync requests to appropriate daemon
- Manage sync priorities
- Track sync state and health
- Emit sync events for pipeline coordination

### 2. AutoSyncDaemon (Primary Sync Engine)

**File:** `app/coordination/auto_sync_daemon.py`

The main sync daemon responsible for automated P2P data synchronization.

**Strategies:**

- `push_from_generator`: GPU nodes push data after generation (default)
- `gossip`: Multi-hop replication for redundancy
- `broadcast`: Full replication to all nodes
- `ephemeral`: Aggressive sync for termination-prone hosts

**Key Features:**

- Verification after sync using `verify_and_retry_sync()`
- SHA256 checksum validation
- Retry with exponential backoff
- Excludes coordinator nodes from sync targets

```python
from app.coordination.auto_sync_daemon import AutoSyncDaemon

daemon = AutoSyncDaemon(strategy="push_from_generator")
await daemon.start()
```

### 3. SyncRouter (Intelligent Routing)

**File:** `app/coordination/sync_router.py`

Routes data to nodes based on capabilities, policies, and current state.

**Routing Decisions:**

- Node disk capacity (refreshed every 30 seconds)
- Node exclusion rules from ClusterManifest
- Replication target requirements
- Provider-specific bandwidth limits

**Event Subscriptions:**

- `NODE_RECOVERED`: Re-enable sync to recovered nodes
- `BACKPRESSURE_ACTIVATED`: Pause non-urgent syncs
- `BACKPRESSURE_RELEASED`: Resume normal operations

```python
from app.coordination.sync_router import SyncRouter

router = SyncRouter()
targets = router.get_sync_targets(data_type="games", config_key="hex8_2p")
```

### 4. SyncPushDaemon (Push-Based Sync)

**File:** `app/coordination/sync_push_daemon.py`

Daemon for GPU nodes to push data to coordinator with verified cleanup.

**Thresholds:**

- 50% disk: Start pushing completed games
- 70% disk: Push urgently (higher priority)
- 75% disk: Clean up files with 2+ verified copies

**Key Principle:** Data is only deleted after N verified copies exist elsewhere.

**Events Emitted:**

- `DATA_SYNC_STARTED`: When push cycle begins
- `DATA_SYNC_COMPLETED`: When push cycle succeeds
- `DATA_SYNC_FAILED`: When push cycle fails

### 5. BandwidthCoordinatedRsync (Transport)

**File:** `app/coordination/sync_bandwidth.py`

Bandwidth-limited rsync with host-level allocation.

**Features:**

- Per-host bandwidth limits from config
- `--partial` flag for resume on network glitches
- SHA256 checksum verification
- BatchRsync for multi-file transfers

**Bandwidth Limits:**

- RunPod/Nebius: 100 MB/s
- Vast.ai/Vultr: 50 MB/s

```python
from app.coordination.sync_bandwidth import get_bandwidth_coordinator

coordinator = get_bandwidth_coordinator()
result = await coordinator.sync_file(
    source="/data/games/selfplay.db",
    dest="user@gpu-node:/data/games/",
    host="gpu-node",
)
```

## Data Flow

### Selfplay Data Sync

```
GPU Node (selfplay)
    │
    ▼ SyncPushDaemon
    │ (at 50% disk usage)
    │
    ▼ P2P HTTP / rsync
    │
    ▼ Coordinator (mac-studio)
    │
    ▼ ClusterManifest.register_sync()
    │
    ▼ SyncReceipt stored
    │
    ▼ DATA_SYNC_COMPLETED event
    │
    ▼ DataPipelineOrchestrator
    │
    ▼ Trigger NPZ export
```

### Model Distribution

```
Training Node
    │
    ▼ MODEL_PROMOTED event
    │
    ▼ UnifiedDistributionDaemon
    │
    ▼ BitTorrent / HTTP / rsync
    │
    ▼ All GPU nodes
    │
    ▼ Verification (SHA256)
    │
    ▼ MODEL_DISTRIBUTED event
```

## Configuration

### distributed_hosts.yaml

```yaml
sync_routing:
  max_disk_usage_percent: 70.0
  priority_hosts:
    - nebius-h100-1
    - runpod-h100
  exclude_patterns:
    - '*.tmp'
    - '.rsync-partial/*'

auto_sync:
  enabled: true
  interval_seconds: 60
  strategy: push_from_generator

bandwidth_limits:
  default_mbps: 50
  providers:
    runpod: 100
    nebius: 100
    vast: 50
    vultr: 50
```

### Environment Variables

| Variable                            | Default | Description                    |
| ----------------------------------- | ------- | ------------------------------ |
| `RINGRIFT_AUTO_SYNC_ENABLED`        | true    | Enable AutoSyncDaemon          |
| `RINGRIFT_SYNC_INTERVAL`            | 60      | Sync check interval (seconds)  |
| `RINGRIFT_SYNC_PUSH_THRESHOLD`      | 50      | Disk % to start pushing        |
| `RINGRIFT_MIN_COPIES_BEFORE_DELETE` | 2       | Required copies before cleanup |

## Event Integration

The sync infrastructure emits events for pipeline coordination:

| Event                 | Emitter                        | Subscribers                               |
| --------------------- | ------------------------------ | ----------------------------------------- |
| `DATA_SYNC_STARTED`   | AutoSyncDaemon, SyncPushDaemon | DataPipelineOrchestrator                  |
| `DATA_SYNC_COMPLETED` | AutoSyncDaemon, SyncPushDaemon | DataPipelineOrchestrator, ExportScheduler |
| `DATA_SYNC_FAILED`    | AutoSyncDaemon, SyncPushDaemon | AlertManager, RecoveryOrchestrator        |
| `NODE_RECOVERED`      | P2POrchestrator                | SyncRouter, AutoSyncDaemon                |

## Health Monitoring

All sync components implement `health_check()` for DaemonManager integration:

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()
health = facade.health_check()

# Returns HealthCheckResult with:
# - is_healthy: bool
# - message: str
# - details: dict (sync stats, error counts, etc.)
```

## Troubleshooting

### Sync Stalls

1. Check disk space on target nodes: `df -h`
2. Verify P2P connectivity: `curl http://localhost:8770/status`
3. Check ClusterManifest for exclusion rules
4. Review sync logs: `tail -f logs/sync/*.log`

### Verification Failures

1. Check network connectivity between nodes
2. Verify rsync is installed on all nodes
3. Check for disk corruption: `sqlite3 db_file 'PRAGMA integrity_check'`
4. Review checksum logs for mismatches

### Bandwidth Exhaustion

1. Check current allocations: `/bandwidth/status` endpoint
2. Review per-host limits in config
3. Consider increasing limits for high-priority nodes
4. Check for stuck transfers holding allocations

## See Also

- `docs/DAEMON_REGISTRY.md` - All daemon types and configurations
- `docs/EVENT_SYSTEM_REFERENCE.md` - Event types and wiring
- `docs/runbooks/SYNC_STALL_DETECTION.md` - Operational runbook
- `docs/CLUSTER_TRANSPORT_ARCHITECTURE.md` - Transport layer details
