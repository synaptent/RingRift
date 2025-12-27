# Sync Architecture

> Unified data synchronization for RingRift AI training cluster (December 2025)

## Overview

RingRift uses a multi-layer sync architecture to replicate training data (games, models, NPZ files) across a distributed GPU cluster. The system prioritizes:

1. **Data safety**: No game data loss, even on ephemeral cloud instances
2. **Low latency**: Training nodes have fresh data
3. **Bandwidth efficiency**: Provider-aware rate limiting
4. **Resilience**: Multiple transports with automatic failover

## Architecture Diagram

```
                              SyncOrchestrator
                        (Unified Entry Point)
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   SyncScheduler           SyncRouter           DistributedSyncCoord.
   (When to sync)        (Where to sync)         (How to sync)
          │                        │                        │
          │                        ▼                        │
          │              ClusterManifest ◀─────────────────┤
          │           (Central Registry)                   │
          │                                                │
          └────────────────────┬───────────────────────────┘
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
             AutoSyncDaemon      BandwidthManager
           (Background sync)    (Rate limiting)
```

## Layer Overview

<<<<<<< Updated upstream
| Layer | Module | Purpose |
|-------|--------|---------|
| **SyncOrchestrator** | `sync_orchestrator.py` | Unified facade, coordinates all sync |
| **SyncScheduler** | `sync_coordinator.py` (coordination) | Scheduling: when/what to sync |
| **DistributedSyncCoordinator** | `sync_coordinator.py` (distributed) | Execution: performs actual syncs |
| **SyncRouter** | `sync_router.py` | Routing: which nodes get data |
| **ClusterManifest** | `cluster_manifest.py` | Registry: tracks where data exists |
| **AutoSyncDaemon** | `auto_sync_daemon.py` | Background: automated replication |
=======
| Layer                          | Module                               | Purpose                              |
| ------------------------------ | ------------------------------------ | ------------------------------------ |
| **SyncOrchestrator**           | `sync_orchestrator.py`               | Unified facade, coordinates all sync |
| **SyncScheduler**              | `sync_coordinator.py` (coordination) | Scheduling: when/what to sync        |
| **DistributedSyncCoordinator** | `sync_coordinator.py` (distributed)  | Execution: performs actual syncs     |
| **SyncRouter**                 | `sync_router.py`                     | Routing: which nodes get data        |
| **ClusterManifest**            | `cluster_manifest.py`                | Registry: tracks where data exists   |
| **AutoSyncDaemon**             | `auto_sync_daemon.py`                | Background: automated replication    |
>>>>>>> Stashed changes

## Sync Strategies

Three sync strategies for different scenarios:

### HYBRID (Default for Persistent Nodes)

```
Game Complete
     │
     ▼
Push to 2 neighbors ──► Gossip replication
(Layer 1: immediate)    (Layer 2: eventual)
```

- **Push-from-generator**: Immediately push to 2 nearby nodes
- **Gossip replication**: Eventual consistency across cluster
- **Interval**: 60s main sync, 30s gossip

### EPHEMERAL (Vast.ai/Spot Instances)

```
Game Complete
     │
     ▼
Immediate push ──► WAL persistence ──► Termination handler
(5s interval)      (crash safety)       (final sync)
```

- **5-second polling**: Aggressive sync before termination
- **Write-ahead log**: Pending games persist to disk
- **SIGTERM handler**: Final sync attempt on shutdown

### BROADCAST (Leader-Only)

```
Leader Node
     │
     ▼
Push to all ──► Skip ineligible ──► Track delivery
training nodes   (disk full)         (manifest)
```

- Used by cluster leader for cluster-wide distribution
- Respects disk usage caps and exclusion rules

## Transport Selection

The system automatically selects the best transport:

<<<<<<< Updated upstream
| Transport | Speed | Use Case |
|-----------|-------|----------|
| **P2P HTTP** | Fast | Same-provider nodes, leader distribution |
| **aria2** | Fast | Parallel multi-source downloads |
| **rsync/SSH** | Medium | Cross-provider, bandwidth-limited |
| **Gossip** | Slow | Eventual consistency, background |
=======
| Transport     | Speed  | Use Case                                 |
| ------------- | ------ | ---------------------------------------- |
| **P2P HTTP**  | Fast   | Same-provider nodes, leader distribution |
| **aria2**     | Fast   | Parallel multi-source downloads          |
| **rsync/SSH** | Medium | Cross-provider, bandwidth-limited        |
| **Gossip**    | Slow   | Eventual consistency, background         |
>>>>>>> Stashed changes

### Selection Logic

```python
# Simplified transport selection (from sync_coordinator.py)
if is_same_provider(source, dest) and has_p2p:
    return "p2p_http"
elif file_size > 100_MB and len(sources) > 1:
    return "aria2"  # Multi-source parallel
elif has_ssh_access(dest):
    return "rsync"
else:
    return "gossip"  # Fallback
```

## Bandwidth Management

Provider-specific limits prevent network saturation:

<<<<<<< Updated upstream
| Provider | Limit | Notes |
|----------|-------|-------|
| Lambda | 100 MB/s | Internal network is fast |
| RunPod | 100 MB/s | Good connectivity |
| Tailscale | 100 MB/s | Internal mesh |
| Hetzner | 80 MB/s | Dedicated servers |
| Vultr | 80 MB/s | Cloud compute |
| Nebius | 50 MB/s | Has rate limits |
| Vast.ai | 50 MB/s | Varies by instance |
| Default | 20 MB/s | Conservative fallback |
=======
| Provider  | Limit    | Notes                    |
| --------- | -------- | ------------------------ |
| Lambda    | 100 MB/s | Internal network is fast |
| RunPod    | 100 MB/s | Good connectivity        |
| Tailscale | 100 MB/s | Internal mesh            |
| Hetzner   | 80 MB/s  | Dedicated servers        |
| Vultr     | 80 MB/s  | Cloud compute            |
| Nebius    | 50 MB/s  | Has rate limits          |
| Vast.ai   | 50 MB/s  | Varies by instance       |
| Default   | 20 MB/s  | Conservative fallback    |
>>>>>>> Stashed changes

### Priority-Based Allocation

```python
class TransferPriority(Enum):
    LOW = "low"        # Background replication
    NORMAL = "normal"  # Regular sync
    HIGH = "high"      # Training node needs data
    CRITICAL = "critical"  # Training blocked, immediate
```

Higher priority transfers get more bandwidth allocation.

## Cluster Manifest

The `ClusterManifest` tracks where data exists across the cluster:

### Data Types Tracked

<<<<<<< Updated upstream
| Type | Example | Purpose |
|------|---------|---------|
| `GAME` | `game-123` → `gh200-a:/data/games/selfplay.db` | Game locations |
| `MODEL` | `canonical_hex8_2p.pth` → `[nebius-h100, runpod-h100]` | Model replicas |
| `NPZ` | `hex8_2p.npz` → `[training-node-1, training-node-2]` | Training data |
| `CHECKPOINT` | `checkpoint_epoch_50.pth` → `nebius-h100-1` | Training checkpoints |
=======
| Type         | Example                                                | Purpose              |
| ------------ | ------------------------------------------------------ | -------------------- |
| `GAME`       | `game-123` → `gh200-a:/data/games/selfplay.db`         | Game locations       |
| `MODEL`      | `canonical_hex8_2p.pth` → `[nebius-h100, runpod-h100]` | Model replicas       |
| `NPZ`        | `hex8_2p.npz` → `[training-node-1, training-node-2]`   | Training data        |
| `CHECKPOINT` | `checkpoint_epoch_50.pth` → `nebius-h100-1`            | Training checkpoints |
>>>>>>> Stashed changes

### Replication Targets

```python
# Get nodes that need a model replica
targets = manifest.get_replication_targets(
    data_id="canonical_hex8_2p.pth",
    data_type=DataType.MODEL,
    min_copies=2,  # Ensure 2+ replicas
)
# Returns nodes with:
# - Disk usage < 70%
# - Not excluded (coordinator, dev machines)
# - Role=TRAINING or SELFPLAY
```

## Sync Router

Routes data to appropriate nodes based on policies:

### Exclusion Rules

```yaml
# From distributed_hosts.yaml
sync_routing:
  exclude_from_sync:
<<<<<<< Updated upstream
    - mac-studio    # Coordinator, limited disk
    - dev-laptop    # Development machine
  nfs_sharing_groups:
    - [gh200-a, gh200-b, gh200-c]  # Share NFS, skip sync
=======
    - mac-studio # Coordinator, limited disk
    - dev-laptop # Development machine
  nfs_sharing_groups:
    - [gh200-a, gh200-b, gh200-c] # Share NFS, skip sync
>>>>>>> Stashed changes
```

### Capacity Checks

- **MAX_DISK_USAGE_PERCENT**: 70% - Don't sync to full nodes
- **MIN_FREE_DISK_PERCENT**: 30% - Ensure headroom

## Usage Examples

### Simple Sync (Recommended)

```python
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.initialize()

# Sync all data types
result = await orchestrator.sync_all()

# Check status
status = orchestrator.get_status()
print(f"Games synced: {status['games_synced']}")
print(f"Models synced: {status['models_synced']}")

await orchestrator.shutdown()
```

### Scheduling Layer

```python
from app.coordination.sync_coordinator import (
    get_sync_scheduler,
    get_cluster_data_status,
    schedule_priority_sync,
)

# Check cluster data freshness
status = get_cluster_data_status()
for config_key, info in status.items():
    print(f"{config_key}: {info['age_hours']}h old, {info['game_count']} games")

# Get sync recommendations
recommendations = get_sync_recommendations()
for rec in recommendations:
    print(f"Sync {rec.source} → {rec.target}: {rec.reason}")

# Schedule priority sync for training
await schedule_priority_sync(
    config_key="hex8_2p",
    priority=SyncPriority.CRITICAL,
)
```

### Execution Layer

```python
from app.coordination import DistributedSyncCoordinator

coordinator = DistributedSyncCoordinator.get_instance()

# Sync training data to a specific node
result = await coordinator.sync_training_data(
    target_node="nebius-h100-1",
    config_key="hex8_2p",
)

# Sync specific models
result = await coordinator.sync_models(
    model_ids=["canonical_hex8_2p.pth"],
    target_nodes=["training-node-1", "training-node-2"],
)

# Full cluster sync
stats = await coordinator.full_cluster_sync()
print(f"Synced {stats.files_synced} files, {stats.bytes_transferred} bytes")
```

### Background Daemon

```python
from app.coordination.auto_sync_daemon import (
    AutoSyncDaemon,
    AutoSyncConfig,
    create_ephemeral_sync_daemon,
)

# Standard daemon (for persistent hosts)
config = AutoSyncConfig(
    interval_seconds=60,
    strategy="hybrid",
)
daemon = AutoSyncDaemon(config=config)
await daemon.start()

# Ephemeral daemon (for Vast.ai/spot)
ephemeral_daemon = create_ephemeral_sync_daemon()
await ephemeral_daemon.start()
```

## Data Flow Diagrams

### Game Data Flow

```
GPU Selfplay Node                    Cluster
      │                                │
      ▼                                │
 [Game Complete]                       │
      │                                │
      ▼                                │
 GameReplayDB.store()                  │
      │                                │
      ▼                                │
 ClusterManifest.register_game() ─────►│ (game_id → node mapping)
      │                                │
      ▼                                │
 AutoSyncDaemon.push_to_neighbors() ──►│ (Layer 1: immediate)
      │                                │
      ▼                                │
 P2P Gossip ──────────────────────────►│ (Layer 2: eventual)
                                       │
                                       ▼
                              Training nodes see game
```

### Model Distribution Flow

```
Training Node                        Cluster
      │                                │
      ▼                                │
 [Training Complete]                   │
      │                                │
      ▼                                │
 MODEL_PROMOTED event                  │
      │                                │
      ▼                                │
 ModelDistributionDaemon ─────────────►│
      │                                │
      ├─ SyncRouter.get_sync_targets() │
      │                                │
      ├─ BandwidthManager.allocate()   │
      │                                │
      └─ Transport: P2P/rsync/aria2 ──►│
                                       │
                                       ▼
                              All selfplay nodes updated
```

## Safety Features

### Write-Ahead Log (Ephemeral Nodes)

```
Game Write                WAL                    Sync
    │                      │                      │
    ▼                      ▼                      ▼
 Store game ──► Write to JSONL ──► Push to cluster
    │                      │                      │
    │               [On crash]                    │
    │                      ▼                      │
    │              Recover pending ──────────────►│
    │              games from WAL                 │
```

### Integrity Verification

```python
from app.coordination.sync_integrity import (
    verify_sync_integrity,
    compute_file_checksum,
    check_sqlite_integrity,
)

# Verify SQLite integrity after sync
report = verify_sync_integrity(
    source_path="/source/games.db",
    dest_path="/dest/games.db",
)
if not report.is_valid:
    print(f"Integrity check failed: {report.errors}")
```

### Circuit Breakers

- **Consecutive failures**: After 3 failures, back off exponentially
- **Timeout watchdog**: Kill syncs exceeding deadline
- **Health checks**: Skip unhealthy nodes

## Configuration

### distributed_hosts.yaml

```yaml
sync_routing:
  auto_sync_interval_seconds: 60
  gossip_interval_seconds: 30
  max_concurrent_syncs: 4
  bandwidth_limit_mbps: 50

  exclude_from_sync:
    - mac-studio
    - dev-machines

  nfs_sharing_groups:
<<<<<<< Updated upstream
    - [gh200-a, gh200-b, gh200-c]  # Lambda NFS group
=======
    - [gh200-a, gh200-b, gh200-c] # Lambda NFS group
>>>>>>> Stashed changes

  ephemeral_hosts:
    - vast-*
    - runpod-spot-*

  allowed_external_storage:
    - host: backup.example.com
      path: /backups/ringrift
<<<<<<< Updated upstream
      schedule: "0 */6 * * *"  # Every 6 hours
=======
      schedule: '0 */6 * * *' # Every 6 hours
>>>>>>> Stashed changes

hosts:
  nebius-h100-1:
    sync_priority: high
    bandwidth_limit_mbps: 100
    role: training
```

### Environment Variables

<<<<<<< Updated upstream
| Variable | Default | Description |
|----------|---------|-------------|
| `RINGRIFT_SYNC_INTERVAL` | 60 | Main sync interval (seconds) |
| `RINGRIFT_GOSSIP_INTERVAL` | 30 | Gossip interval (seconds) |
| `RINGRIFT_SYNC_MAX_CONCURRENT` | 4 | Max concurrent syncs |
| `RINGRIFT_SYNC_BANDWIDTH_MBPS` | 50 | Default bandwidth limit |
| `RINGRIFT_SYNC_MAX_DISK_USAGE` | 70 | Max disk usage % |

## Files Reference

| File | Purpose |
|------|---------|
| `app/distributed/sync_orchestrator.py` | Unified facade |
| `app/coordination/sync_coordinator.py` | SyncScheduler (scheduling) |
| `app/distributed/sync_coordinator.py` | DistributedSyncCoordinator (execution) |
| `app/coordination/sync_router.py` | Intelligent routing |
| `app/distributed/cluster_manifest.py` | Central registry |
| `app/coordination/auto_sync_daemon.py` | Background automation |
| `app/coordination/sync_bandwidth.py` | Bandwidth management |
| `app/coordination/sync_constants.py` | Shared enums/dataclasses |
| `app/coordination/sync_integrity.py` | Checksum verification |
| `app/coordination/ephemeral_sync.py` | Ephemeral host support (deprecated) |
=======
| Variable                       | Default | Description                  |
| ------------------------------ | ------- | ---------------------------- |
| `RINGRIFT_SYNC_INTERVAL`       | 60      | Main sync interval (seconds) |
| `RINGRIFT_GOSSIP_INTERVAL`     | 30      | Gossip interval (seconds)    |
| `RINGRIFT_SYNC_MAX_CONCURRENT` | 4       | Max concurrent syncs         |
| `RINGRIFT_SYNC_BANDWIDTH_MBPS` | 50      | Default bandwidth limit      |
| `RINGRIFT_SYNC_MAX_DISK_USAGE` | 70      | Max disk usage %             |

## Files Reference

| File                                   | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| `app/distributed/sync_orchestrator.py` | Unified facade                         |
| `app/coordination/sync_coordinator.py` | SyncScheduler (scheduling)             |
| `app/distributed/sync_coordinator.py`  | DistributedSyncCoordinator (execution) |
| `app/coordination/sync_router.py`      | Intelligent routing                    |
| `app/distributed/cluster_manifest.py`  | Central registry                       |
| `app/coordination/auto_sync_daemon.py` | Background automation                  |
| `app/coordination/sync_bandwidth.py`   | Bandwidth management                   |
| `app/coordination/sync_constants.py`   | Shared enums/dataclasses               |
| `app/coordination/sync_integrity.py`   | Checksum verification                  |
| `app/coordination/ephemeral_sync.py`   | Ephemeral host support (deprecated)    |
>>>>>>> Stashed changes

## Migration from Legacy Systems

### Old Pattern (Deprecated)

```python
# Direct import from distributed (still works, deprecated)
from app.distributed.sync_coordinator import SyncCoordinator
coordinator = SyncCoordinator()
await coordinator.sync_all()
```

### New Pattern (Recommended)

```python
# Use unified orchestrator
from app.distributed.sync_orchestrator import get_sync_orchestrator

orchestrator = get_sync_orchestrator()
await orchestrator.initialize()
result = await orchestrator.sync_all()
```

## See Also

- [EVENT_SYSTEM.md](./EVENT_SYSTEM.md) - Event-driven sync triggers
- [COORDINATION_SYSTEM.md](../../ai-service/docs/architecture/COORDINATION_SYSTEM.md) - Daemon coordination
- [DAEMON_REGISTRY.md](../../ai-service/docs/DAEMON_REGISTRY.md) - AutoSyncDaemon configuration
