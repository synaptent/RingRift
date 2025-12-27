# Sync Architecture Documentation

This document clarifies the responsibilities and relationships between the 13 sync-related modules in the RingRift AI service.

## Overview

The sync system ensures data (games, models, NPZ files) flows reliably across the distributed cluster. It uses a layered architecture with clear separation of concerns.

## Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAEMON LAYER (What runs)                     │
├─────────────────────────────────────────────────────────────────┤
│  auto_sync_daemon.py    │ Primary: P2P gossip-based sync       │
│  ephemeral_sync.py      │ Aggressive sync for Vast.ai nodes    │
│  cluster_data_sync.py   │ Cluster-wide data distribution       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COORDINATION LAYER (How to sync)               │
├─────────────────────────────────────────────────────────────────┤
│  sync_coordinator.py    │ DEPRECATED - use auto_sync_daemon    │
│  sync_router.py         │ Intelligent routing by node caps     │
│  sync_bandwidth.py      │ Adaptive bandwidth per host          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BASE LAYER (Primitives)                      │
├─────────────────────────────────────────────────────────────────┤
│  sync_base.py           │ SyncEntry, SyncStatus dataclasses    │
│  sync_constants.py      │ Timeouts, thresholds, categories     │
│  sync_mutex.py          │ Distributed locking for sync ops     │
│  sync_coordination_core.py │ Core sync logic (push/pull)       │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### Daemon Layer (Start These)

#### `auto_sync_daemon.py` - PRIMARY SYNC DAEMON

**Purpose**: Automated P2P data synchronization using push-from-generator + gossip replication.

**Key Features**:

- Push games immediately after generation
- Gossip protocol for cluster-wide consistency
- Respects `excluded_hosts` from config
- Bandwidth-aware transfers

**When to use**: Always running on coordinator and training nodes.

**Events emitted**: `SYNC_STARTED`, `SYNC_COMPLETED`, `SYNC_FAILED`

---

#### `ephemeral_sync.py` - AGGRESSIVE SYNC FOR EPHEMERAL NODES

**Purpose**: Ultra-aggressive sync for termination-prone Vast.ai nodes.

**Key Features**:

- 5-second poll interval (vs 60s for persistent nodes)
- Write-through mode: waits for push confirmation
- Termination signal handlers (SIGTERM/SIGINT)
- Emits `HOST_OFFLINE` with pending games count

**When to use**: On Vast.ai and other spot instances.

**Config**: `ephemeral_sync.poll_interval_seconds = 5`

---

#### `cluster_data_sync.py` - CLUSTER-WIDE DISTRIBUTION

**Purpose**: Distribute data to all nodes in the cluster.

**Key Features**:

- TRAINING_NODE_WATCHER daemon for priority sync
- Detects active training and syncs fresh data
- Coordinates with ClusterManifest for data locations

**When to use**: For bulk distribution after major exports.

---

### Coordination Layer (Called by Daemons)

#### `sync_router.py` - INTELLIGENT ROUTING

**Purpose**: Routes sync operations based on node capabilities and exclusion rules.

**Key Features**:

- Checks node health before routing
- Respects `sync_routing.excluded_hosts` config
- Routes to nodes with sufficient disk space
- Handles `allowed_external_storage` for mac-studio

**API**:

```python
from app.coordination.sync_router import SyncRouter
router = SyncRouter()
targets = router.get_sync_targets(data_type="games", size_mb=100)
```

---

#### `sync_bandwidth.py` - BANDWIDTH MANAGEMENT

**Purpose**: Adaptive bandwidth limiting per host.

**Key Features**:

- Host-specific bandwidth limits (Runpod: 100MB/s, Vast: 50MB/s)
- Concurrent transfer limiting
- Graceful degradation under load

**Config** (from `distributed_hosts.yaml`):

```yaml
auto_sync:
  bandwidth_limit_mbps: 100
  host_bandwidth_overrides:
    vast-*: 50
    runpod-*: 100
```

---

#### `sync_coordinator.py` - DEPRECATED

**Status**: Superseded by `auto_sync_daemon.py` (December 2025)

**Migration**: Use `AutoSyncDaemon` instead. Removal planned Q2 2026.

---

### Base Layer (Primitives)

#### `sync_base.py` - DATA STRUCTURES

```python
@dataclass
class SyncEntry:
    source_path: str
    target_host: str
    data_type: SyncCategory  # GAMES, MODELS, NPZ, DATABASES
    status: SyncStatus       # PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority: int            # 0-100, higher = more urgent
```

---

#### `sync_constants.py` - CONFIGURATION CONSTANTS

```python
class SyncCategory(Enum):
    GAMES = "games"
    MODELS = "models"
    NPZ = "npz"
    DATABASES = "databases"

SYNC_TIMEOUT_SECONDS = 300
SYNC_RETRY_LIMIT = 3
SYNC_BATCH_SIZE = 100
```

---

#### `sync_mutex.py` - DISTRIBUTED LOCKING

**Purpose**: Prevents concurrent syncs to the same target.

**Usage**:

```python
async with sync_mutex.acquire("host1:games"):
    await sync_to_host("host1", games_data)
```

---

#### `sync_coordination_core.py` - CORE SYNC LOGIC

**Purpose**: Low-level push/pull operations.

**API**:

```python
await push_data(source, target, data_type)
await pull_data(source, target, data_type)
```

---

## Sync Decision Tree

When data needs to be synchronized, the system follows this decision process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYNC TRIGGER                                         │
│  Sources: SELFPLAY_COMPLETE, NEW_GAMES_AVAILABLE, MODEL_PROMOTED            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Is source an ephemeral node?  │
                    │ (Vast.ai, spot instance)      │
                    └───────────────────────────────┘
                            │              │
                        Yes │              │ No
                            ▼              ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Use ephemeral_   │  │ Use auto_sync_   │
              │ sync.py          │  │ daemon.py        │
              │ (5s interval,    │  │ (60s interval,   │
              │  write-through)  │  │  gossip mode)    │
              └────────┬─────────┘  └────────┬─────────┘
                       │                      │
                       └──────────┬───────────┘
                                  ▼
                    ┌───────────────────────────────┐
                    │ Query sync_router for targets │
                    └───────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────────┐
              │ For each candidate target, check:         │
              │ 1. Is target in excluded_hosts? → Skip    │
              │ 2. Is target healthy? → Skip if not       │
              │ 3. Disk usage < 70%? → Skip if not        │
              │ 4. Has target already? → Skip if yes      │
              └───────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │ Sort targets by priority:     │
                    │ 1. Training nodes (highest)   │
                    │ 2. Persistent GPU nodes       │
                    │ 3. CPU nodes                  │
                    │ 4. Other ephemeral nodes      │
                    └───────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │ Select transport protocol     │
                    │ (see Transport Selection)     │
                    └───────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │ Acquire sync_mutex for target │
                    │ (prevent concurrent syncs)    │
                    └───────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │ Apply bandwidth limit for     │
                    │ target host category          │
                    └───────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────────┐
              │ Execute sync via transport protocol       │
              └───────────────────────────────────────────┘
                                  │
                        ┌────────┴────────┐
                     Success           Failure
                        │                 │
                        ▼                 ▼
              ┌─────────────────┐ ┌─────────────────────┐
              │ Register in     │ │ Retry with fallback │
              │ ClusterManifest │ │ transport (see      │
              │                 │ │ Failure Recovery)   │
              └─────────────────┘ └─────────────────────┘
```

## Transport Selection

The sync system supports multiple transport protocols, selected based on availability and network conditions.

### Transport Priority

| Priority | Transport         | Best For               | Bandwidth | Reliability |
| -------- | ----------------- | ---------------------- | --------- | ----------- |
| 1        | **P2P HTTP**      | Same-cluster nodes     | High      | Very High   |
| 2        | **rsync**         | Cross-network with SSH | High      | High        |
| 3        | **aria2**         | Large files, resumable | Medium    | Very High   |
| 4        | **scp**           | Fallback, simple       | Medium    | Medium      |
| 5        | **HTTP Download** | Public endpoints       | Low       | Medium      |

### Transport Selection Logic

```python
def select_transport(source: str, target: str, data_size_mb: int) -> Transport:
    """Select optimal transport for sync operation."""

    # P2P HTTP if both nodes in same P2P mesh
    if is_p2p_reachable(source, target):
        return Transport.P2P_HTTP

    # rsync for most inter-node transfers (SSH available)
    if has_ssh_access(source, target):
        if data_size_mb > 100:
            # Large files: use aria2 for resumability
            if aria2_available(target):
                return Transport.ARIA2
        return Transport.RSYNC

    # scp fallback if rsync unavailable
    if has_ssh_access(source, target):
        return Transport.SCP

    # HTTP download from public endpoints
    if is_http_accessible(source):
        return Transport.HTTP_DOWNLOAD

    raise NoTransportAvailable(source, target)
```

### Transport Characteristics

#### P2P HTTP (Preferred)

```
Source Node ──────► P2P Mesh ──────► Target Node
           [8770 port]       [8770 port]

Advantages:
- Direct node-to-node, no SSH overhead
- Works across NAT (via Tailscale)
- Built-in health checks
- Automatic failover

Limitations:
- Requires both nodes in P2P mesh
- Not suitable for very large files (>1GB)
```

#### rsync over SSH

```
Source Node ──────► SSH Tunnel ──────► Target Node
          [rsync --partial -z]

Advantages:
- Resumable transfers
- Compression support
- Delta sync (only changed bytes)
- Bandwidth limiting (--bwlimit)

Limitations:
- Requires SSH key setup
- Higher latency for small files
```

#### aria2 (Large Files)

```
Source Node ──────► aria2c ──────► Target Node
          [multi-connection download]

Advantages:
- Multi-connection parallel download
- Resumable from any point
- Checksum verification
- Torrent/metalink support

Limitations:
- Requires aria2 installed on target
- More complex setup
```

## 8-Phase Sync Strategy

The complete sync lifecycle follows 8 phases:

### Phase 1: Detection

Sync is triggered by events:

- `SELFPLAY_COMPLETE` - New games generated
- `NEW_GAMES_AVAILABLE` - Games synced to coordinator
- `MODEL_PROMOTED` - New model needs distribution
- `NPZ_EXPORT_COMPLETE` - Training data ready
- `DATA_STALE` - Target has stale data

### Phase 2: Source Resolution

1. Query ClusterManifest for data locations
2. Verify source accessibility
3. Check source data integrity (checksums)
4. Select primary source (prefer closest)

### Phase 3: Target Selection

Via `sync_router.py`:

1. Get all cluster nodes
2. Filter by exclusion rules
3. Filter by health status
4. Filter by disk availability
5. Sort by priority (training nodes first)
6. Apply replication target (default: 2 copies)

### Phase 4: Transport Negotiation

1. Check P2P mesh connectivity
2. Test SSH availability
3. Verify aria2 installation
4. Select transport per priority list
5. Configure bandwidth limits

### Phase 5: Locking

Via `sync_mutex.py`:

1. Acquire distributed lock for target+data_type
2. Timeout after 30 seconds if lock unavailable
3. Emit warning if waiting >10 seconds

### Phase 6: Transfer Execution

1. Apply bandwidth throttling
2. Start transfer with selected transport
3. Monitor progress (emit `SYNC_PROGRESS` events)
4. Handle interrupts gracefully

### Phase 7: Verification

1. Verify file integrity (size, checksum)
2. Validate database schema (for .db files)
3. Test model loadability (for .pth files)
4. Report discrepancies as `SYNC_VERIFICATION_FAILED`

### Phase 8: Registration

1. Update ClusterManifest with new location
2. Update replication count
3. Emit `DATA_SYNC_COMPLETED` event
4. Release distributed lock

## Failure Recovery

### Retry Strategy

```
Attempt 1: Primary transport
    ↓ (fail)
Attempt 2: Primary transport (different source if available)
    ↓ (fail)
Attempt 3: Fallback transport
    ↓ (fail)
DLQ Entry: Add to dead letter queue for later retry
```

### Dead Letter Queue (DLQ)

Failed syncs enter the DLQ in `data/sync_dlq.db`:

```python
@dataclass
class DLQEntry:
    entry_id: str
    source: str
    target: str
    data_type: str
    data_path: str
    error: str
    attempts: int
    created_at: datetime
    next_retry_at: datetime  # Exponential backoff
```

**Retry Schedule**:

- Attempt 1: Immediate
- Attempt 2: +5 minutes
- Attempt 3: +30 minutes
- Attempt 4: +2 hours
- Attempt 5+: +6 hours (capped)

### Transport Fallback Chain

```
P2P HTTP → rsync → aria2 → scp → HTTP Download
    ↓          ↓        ↓       ↓         ↓
  (fail)    (fail)   (fail)  (fail)  (fail) → DLQ
```

### Network Partition Handling

When network partition detected:

1. Continue syncing within reachable partition
2. Queue syncs for unreachable nodes
3. Emit `SYNC_PARTITION_DETECTED` event
4. Reconcile when partition heals

```
┌─────────────┐           ┌─────────────┐
│  Partition  │  ╳ ╳ ╳ ╳  │  Partition  │
│      A      │           │      B      │
├─────────────┤           ├─────────────┤
│ node-1      │           │ node-3      │
│ node-2      │           │ node-4      │
└─────────────┘           └─────────────┘
       │                         │
       ▼                         ▼
   Continue local            Continue local
   sync operations           sync operations
       │                         │
       ▼                         ▼
   Queue syncs for           Queue syncs for
   partition B nodes         partition A nodes
       │                         │
       └──────────┬──────────────┘
                  │
                  ▼ (partition heals)
          ┌───────────────┐
          │ Reconcile via │
          │ gossip sync   │
          └───────────────┘
```

### Conflict Resolution

When same data exists on multiple sources with different content:

1. **Timestamp-based**: Prefer newer data (for games, models)
2. **Checksum-based**: Prefer version with valid checksum
3. **Quorum-based**: Prefer version held by majority
4. **Manual**: Alert for human resolution (rare)

---

## Data Flow

### Normal Operation (Persistent Nodes)

```
Selfplay Complete → auto_sync_daemon (60s interval)
                         │
                         ▼
                   sync_router (find targets)
                         │
                         ▼
                   sync_bandwidth (rate limit)
                         │
                         ▼
                   sync_coordination_core (rsync/aria2)
                         │
                         ▼
                   ClusterManifest (register location)
```

### Ephemeral Nodes (Vast.ai)

```
Selfplay Complete → ephemeral_sync (5s interval, write-through)
                         │
                         ▼
                   sync_router (priority targets only)
                         │
                         ▼
                   sync_bandwidth (aggressive)
                         │
                         ▼
                   Confirmation wait (write-through mode)
                         │
                         ▼
                   Success/Failure → retry or emit HOST_OFFLINE
```

## Configuration

### `distributed_hosts.yaml`

```yaml
sync_routing:
  max_disk_usage_percent: 70
  replication_target: 2
  excluded_hosts:
    - name: mac-studio
      receive_games: false
      reason: coordinator/dev machine

auto_sync:
  enabled: true
  interval_seconds: 300
  gossip_interval_seconds: 60
  max_concurrent_syncs: 4
  min_games_to_sync: 10
```

## Troubleshooting

### Sync Not Working

1. Check `auto_sync_daemon` is running: `DaemonType.AUTO_SYNC`
2. Verify target not in `excluded_hosts`
3. Check disk space on target: `sync_routing.max_disk_usage_percent`

### Slow Sync

1. Check bandwidth limits: `host_bandwidth_overrides`
2. Verify network connectivity: P2P status
3. Check concurrent sync count: `max_concurrent_syncs`

### Data Loss on Ephemeral

1. Ensure `ephemeral_sync` daemon is running
2. Enable write-through mode for critical data
3. Check termination handlers installed

## See Also

- `docs/EVENT_CATALOG.md` - Sync-related events
- `app/distributed/cluster_manifest.py` - Data location registry
- `config/distributed_hosts.yaml` - Cluster configuration
