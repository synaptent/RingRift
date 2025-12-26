# Distributed Training Infrastructure

This module provides the distributed training infrastructure for RingRift AI, enabling:

- Multi-node GPU cluster coordination
- P2P data synchronization across nodes
- Write-ahead logging for fault tolerance
- Health monitoring and circuit breakers
- Storage abstraction for different environments

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Organization](#module-organization)
3. [Core Components](#core-components)
   - [Sync Coordinator](#sync-coordinator)
   - [P2P Sync Client](#p2p-sync-client)
   - [Unified WAL](#unified-wal)
   - [Circuit Breaker](#circuit-breaker)
   - [Health Checks](#health-checks)
4. [Storage Providers](#storage-providers)
5. [Cluster Management](#cluster-management)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Deprecated Modules](#deprecated-modules)

---

## Architecture Overview

The distributed infrastructure is designed for a heterogeneous cluster of GPU nodes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Cluster Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐│
│   │   GH200-a    │◄───────►│   GH200-b    │◄───────►│   GH200-c    ││
│   │  (Selfplay)  │   P2P   │  (Training)  │   P2P   │  (Selfplay)  ││
│   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘│
│          │                        │                        │        │
│          └────────────────────────┼────────────────────────┘        │
│                                   │                                 │
│                           ┌───────▼───────┐                         │
│                           │   H100 Node   │                         │
│                           │  (Training)   │                         │
│                           └───────────────┘                         │
│                                                                     │
│   Components per node:                                              │
│   • P2P Sync Service (port 8770)                                   │
│   • Unified WAL for fault tolerance                                │
│   • Circuit breaker for node failures                              │
│   • Health monitoring                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Decentralized P2P**: No single point of failure; nodes sync directly with peers
2. **Fault Tolerance**: WAL ensures no data loss during failures
3. **Graceful Degradation**: Circuit breakers isolate failing nodes
4. **Storage Abstraction**: Works on NFS, local disk, or ephemeral storage

---

## Module Organization

The module is organized into these functional areas:

### Synchronization (`sync_*`, `p2p_*`, `unified_*`)

| Module                 | Description                            |
| ---------------------- | -------------------------------------- |
| `sync_coordinator.py`  | High-level sync orchestration (83KB)   |
| `sync_orchestrator.py` | State machine for sync workflows       |
| `sync_utils.py`        | Low-level rsync wrappers               |
| `p2p_sync_client.py`   | HTTP client for P2P sync service       |
| `unified_data_sync.py` | Unified sync service implementation    |
| `unified_manifest.py`  | Manifest tracking for incremental sync |
| `unified_wal.py`       | Write-ahead log for fault tolerance    |

### Transport Layer (`*_transport.py`)

| Module                | Description                             |
| --------------------- | --------------------------------------- |
| `ssh_transport.py`    | SSH/rsync transport for cluster nodes   |
| `aria2_transport.py`  | High-speed aria2 downloader integration |
| `hybrid_transport.py` | Adaptive transport selection            |

### Health & Resilience

| Module               | Description                            |
| -------------------- | -------------------------------------- |
| `circuit_breaker.py` | Circuit breaker pattern implementation |
| `health_checks.py`   | Component health monitoring            |
| `health_registry.py` | Health state persistence               |

### Cluster Management

| Module                   | Description                             |
| ------------------------ | --------------------------------------- |
| `cluster_coordinator.py` | Task role coordination (deprecated)     |
| `cluster_monitor.py`     | Real-time cluster dashboard             |
| `hosts.py`               | Host configuration and memory detection |
| `host_classification.py` | Node capability classification          |
| `discovery.py`           | Worker discovery via Bonjour/mDNS       |

### Data Management

| Module                     | Description                              |
| -------------------------- | ---------------------------------------- |
| `data_catalog.py`          | Cluster-wide data catalog                |
| `data_events.py`           | Event system for data changes            |
| `content_deduplication.py` | Game deduplication across nodes          |
| `db_utils.py`              | Database utilities and atomic operations |
| `game_collector.py`        | In-memory game collection                |

### Storage

| Module                    | Description                   |
| ------------------------- | ----------------------------- |
| `storage_provider.py`     | Storage provider abstraction  |
| `manifest_replication.py` | Manifest sync across replicas |

### Queue-Based Distribution

| Module      | Description                        |
| ----------- | ---------------------------------- |
| `queue.py`  | Task queue abstraction (Redis/SQS) |
| `client.py` | Distributed evaluation client      |

---

## Core Components

### Sync Coordinator

The `SyncCoordinator` is the primary entry point for all sync operations:

```python
from app.distributed import SyncCoordinator, SyncCategory

# Create coordinator
coordinator = SyncCoordinator()

# Sync specific categories
await coordinator.sync(SyncCategory.GAMES)
await coordinator.sync(SyncCategory.MODELS)
await coordinator.sync(SyncCategory.TRAINING_DATA)

# Full cluster sync
from app.distributed import full_cluster_sync
stats = await full_cluster_sync()
print(f"Synced {stats.total_files} files to {stats.nodes_synced} nodes")
```

#### Sync Categories

| Category        | Description               | Default Path     |
| --------------- | ------------------------- | ---------------- |
| `GAMES`         | Game replay databases     | `data/games/`    |
| `MODELS`        | Trained model checkpoints | `models/`        |
| `TRAINING_DATA` | NPZ training files        | `data/training/` |
| `CONFIGS`       | Configuration files       | `config/`        |

#### SyncStats

```python
@dataclass
class SyncStats:
    files_transferred: int
    bytes_transferred: int
    duration_seconds: float
    errors: list[str]

@dataclass
class ClusterSyncStats:
    total_files: int
    total_bytes: int
    nodes_synced: int
    failed_nodes: list[str]
    category_stats: dict[SyncCategory, SyncStats]
```

### P2P Sync Client

Direct node-to-node synchronization via the P2P service:

```python
from app.distributed import P2PSyncClient

# Connect to peer's P2P service
client = P2PSyncClient(host="100.x.x.1", port=8770)

# Check peer status
status = await client.get_status()
print(f"Peer has {status['game_count']} games")

# Request specific games
games = await client.get_games(game_ids=["game-123", "game-456"])

# Pull all new games since last sync
new_games = await client.pull_games_since(timestamp=last_sync_time)
```

#### P2P Fallback Sync

Automatic fallback when primary sync fails:

```python
from app.distributed import P2PFallbackSync

sync = P2PFallbackSync(
    primary_host="100.x.x.1",
    fallback_hosts=["100.x.x.2", "100.x.x.3"],
    timeout=30.0,
)

# Attempts primary, falls back to alternates on failure
result = await sync.pull_games(board_type="hex8", min_games=100)
```

### Unified WAL

Write-ahead log for fault-tolerant operations:

```python
from app.distributed import UnifiedWAL, WALEntryType, get_unified_wal

# Get singleton WAL instance
wal = get_unified_wal()

# Log an operation before executing
entry_id = await wal.append(
    entry_type=WALEntryType.GAME_RECORDED,
    payload={"game_id": "game-123", "path": "data/games/selfplay.db"},
)

# Execute the operation...
# ...

# Mark as committed on success
await wal.commit(entry_id)

# Or rollback on failure
await wal.rollback(entry_id)
```

#### WAL Entry Types

| Type             | Description                   |
| ---------------- | ----------------------------- |
| `GAME_RECORDED`  | New game recorded to database |
| `MODEL_TRAINED`  | Training checkpoint saved     |
| `SYNC_STARTED`   | Sync operation initiated      |
| `SYNC_COMPLETED` | Sync operation finished       |
| `EXPORT_STARTED` | Data export initiated         |

#### WAL Recovery

On startup, recover incomplete operations:

```python
from app.distributed import get_unified_wal

wal = get_unified_wal()

# Get uncommitted entries
pending = await wal.get_pending_entries()
for entry in pending:
    if entry.entry_type == WALEntryType.SYNC_STARTED:
        # Resume or rollback the sync
        await resume_sync(entry.payload)
        await wal.commit(entry.id)
```

### Circuit Breaker

Prevents cascade failures by isolating failing nodes:

```python
from app.distributed import (
    CircuitBreaker,
    CircuitState,
    get_host_breaker,
    get_training_breaker,
)

# Get circuit breaker for a specific host
breaker = get_host_breaker("100.x.x.1")

# Check if circuit is open (failing)
if breaker.state == CircuitState.OPEN:
    print("Host is currently unavailable")

# Execute with circuit breaker protection
try:
    async with breaker:
        result = await sync_to_host("100.x.x.1")
except CircuitOpenError:
    print("Circuit is open, skipping this host")
```

#### Circuit States

| State       | Description                                |
| ----------- | ------------------------------------------ |
| `CLOSED`    | Normal operation, requests flow through    |
| `OPEN`      | Failing, requests are rejected             |
| `HALF_OPEN` | Testing recovery, limited requests allowed |

#### Configuration

```python
breaker = CircuitBreaker(
    name="sync-host-1",
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Try recovery after 60s
    half_open_requests=3,     # Allow 3 test requests
)
```

### Health Checks

Comprehensive health monitoring:

```python
from app.distributed import (
    HealthChecker,
    HealthSummary,
    ComponentHealth,
    get_health_summary,
    format_health_report,
)

# Get health summary for local node
summary = get_health_summary()
print(f"Overall status: {summary.status}")
print(f"Components: {len(summary.components)}")

# Detailed health check
checker = HealthChecker()
checker.add_check("database", check_database_health)
checker.add_check("disk", check_disk_space)
checker.add_check("gpu", check_gpu_memory)

health = await checker.run_checks()
print(format_health_report(health))
```

#### Component Health

```python
@dataclass
class ComponentHealth:
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    last_check: datetime
    metrics: dict[str, Any]
```

---

## Storage Providers

Abstraction layer for different storage environments:

```python
from app.distributed import (
    StorageProvider,
    StorageProviderType,
    detect_storage_provider,
    get_storage_provider,
)

# Auto-detect storage provider
provider = detect_storage_provider()
print(f"Using: {provider.provider_type}")

# Or explicitly request a type
provider = get_storage_provider(StorageProviderType.LOCAL)

# Get standard paths
paths = provider.get_paths()
print(f"Models: {paths.models_dir}")
print(f"Games: {paths.games_dir}")
print(f"Training: {paths.training_dir}")
```

### Provider Types

| Type        | Description               | Persistence        |
| ----------- | ------------------------- | ------------------ |
| `LOCAL`     | Local filesystem          | Persistent         |
| `NFS`       | Lambda Labs shared NFS    | Persistent, shared |
| `EPHEMERAL` | Vast.ai ephemeral storage | Non-persistent     |

### Storage Capabilities

```python
caps = provider.get_capabilities()
print(f"Persistent: {caps.persistent}")
print(f"Shared: {caps.shared_across_nodes}")
print(f"Max size: {caps.max_size_gb}GB")
```

---

## Cluster Management

### Host Configuration

```python
from app.distributed import (
    HostConfig,
    load_remote_hosts,
    get_eligible_hosts_for_board,
    detect_host_memory,
)

# Load hosts from YAML config
hosts = load_remote_hosts("config/distributed_hosts.yaml")

# Get hosts eligible for a board type
eligible = get_eligible_hosts_for_board(
    board_type="hex8",
    num_players=2,
    hosts=hosts,
)

# Check memory on a specific host
memory = detect_host_memory("100.x.x.1")
print(f"GPU memory: {memory.gpu_memory_gb}GB")
print(f"CPU memory: {memory.cpu_memory_gb}GB")
```

### Cluster Monitor

Real-time cluster monitoring:

```python
from app.distributed.cluster_monitor import ClusterMonitor

monitor = ClusterMonitor()

# Get current status
status = monitor.get_cluster_status()
for node in status.nodes:
    print(f"{node.host}: {node.game_count} games, {node.status}")

# Watch mode
async for update in monitor.watch(interval=10):
    print(f"Update: {update}")
```

CLI usage:

```bash
# Check cluster status
python -m app.distributed.cluster_monitor

# Watch mode with 10s interval
python -m app.distributed.cluster_monitor --watch --interval 10

# JSON output
python -m app.distributed.cluster_monitor --json
```

### SSH Executor

Execute commands on remote hosts:

```python
from app.distributed import SSHExecutor, get_ssh_executor

executor = get_ssh_executor()

# Run command on remote host
result = await executor.run(
    host="100.x.x.1",
    command="nvidia-smi --query-gpu=memory.used --format=csv",
)
print(result.stdout)

# Run on multiple hosts
results = await executor.run_parallel(
    hosts=["100.x.x.1", "100.x.x.2"],
    command="df -h /data",
)
```

---

## Configuration

### Host Configuration YAML

```yaml
# config/distributed_hosts.yaml
hosts:
  gpu-node-1:
    ssh_host: 203.0.113.10
    tailscale_ip: 100.x.x.x
    ssh_user: ubuntu
    ssh_port: 22
    ssh_key: ~/.ssh/id_cluster
    ringrift_path: ~/ringrift/ai-service
    venv_activate: source ~/ringrift/ai-service/venv/bin/activate
    memory_gb: 96
    cpus: 32
    gpu: H100
    gpu_vram_gb: 80
    role: training
    status: ready
    worker_port: 8765
    # Optional Cloudflare Zero Trust tunnel for SSH
    cloudflare_tunnel: gpu-node-1.example.com
    cloudflare_service_token_id: CF_TOKEN_ID
    cloudflare_service_token_secret: CF_TOKEN_SECRET

sync:
  batch_size: 100
  timeout: 300
  retry_count: 3

circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
```

### Environment Variables

| Variable                     | Description               | Default                         |
| ---------------------------- | ------------------------- | ------------------------------- |
| `RINGRIFT_CLUSTER_CONFIG`    | Path to hosts YAML        | `config/distributed_hosts.yaml` |
| `RINGRIFT_P2P_PORT`          | P2P service port          | `8770`                          |
| `RINGRIFT_SYNC_TIMEOUT`      | Sync timeout seconds      | `300`                           |
| `RINGRIFT_WAL_DIR`           | WAL storage directory     | `data/wal/`                     |
| `RINGRIFT_CIRCUIT_THRESHOLD` | Circuit failure threshold | `5`                             |

---

## Usage Examples

### Full Cluster Sync

```python
import asyncio
from app.distributed import (
    full_cluster_sync,
    SyncCategory,
    ClusterSyncStats,
)

async def sync_cluster():
    # Sync all categories to all nodes
    stats = await full_cluster_sync(
        categories=[
            SyncCategory.GAMES,
            SyncCategory.MODELS,
        ],
        timeout=600,
    )

    print(f"Synced {stats.total_files} files")
    print(f"Nodes: {stats.nodes_synced} success, {len(stats.failed_nodes)} failed")

    for category, cat_stats in stats.category_stats.items():
        print(f"  {category}: {cat_stats.files_transferred} files")

asyncio.run(sync_cluster())
```

### Incremental Game Sync

```python
from app.distributed import (
    SyncCoordinator,
    P2PSyncClient,
    get_unified_wal,
)

async def sync_new_games():
    coordinator = SyncCoordinator()
    wal = get_unified_wal()

    # Log sync start
    entry_id = await wal.append(
        entry_type=WALEntryType.SYNC_STARTED,
        payload={"category": "games"},
    )

    try:
        # Get games from all peers
        stats = await coordinator.sync_games(
            incremental=True,
            min_quality=0.8,
        )

        await wal.commit(entry_id)
        return stats

    except Exception as e:
        await wal.rollback(entry_id)
        raise
```

### Model Distribution

```python
from app.distributed import (
    SyncCoordinator,
    get_eligible_hosts_for_board,
)

async def distribute_model(model_path: str, board_type: str):
    coordinator = SyncCoordinator()

    # Find eligible training nodes
    hosts = get_eligible_hosts_for_board(
        board_type=board_type,
        num_players=2,
        capabilities=["training"],
    )

    # Push model to all eligible nodes
    for host in hosts:
        await coordinator.push_file(
            local_path=model_path,
            remote_path=f"models/{board_type}/latest.pth",
            host=host.ip,
        )
```

### Health-Aware Operations

```python
from app.distributed import (
    get_health_summary,
    get_host_breaker,
    CircuitState,
)

async def select_healthy_hosts(hosts: list[str]) -> list[str]:
    healthy = []

    for host in hosts:
        # Check circuit breaker
        breaker = get_host_breaker(host)
        if breaker.state == CircuitState.OPEN:
            continue

        # Check health
        try:
            async with breaker:
                summary = await get_health_summary(host)
                if summary.status == "healthy":
                    healthy.append(host)
        except Exception:
            pass  # Circuit breaker will record failure

    return healthy
```

---

## Troubleshooting

### Common Issues

#### P2P Service Not Responding

```bash
# Check if P2P service is running
curl -s http://localhost:8770/status

# Check port binding
lsof -i :8770

# Restart P2P service
pkill -f "p2p_sync" && python -m app.distributed.p2p_sync_server &
```

#### Sync Timeouts

```python
# Increase timeout
stats = await full_cluster_sync(timeout=1200)

# Or use environment variable
# RINGRIFT_SYNC_TIMEOUT=1200 python -m app.training.train
```

#### Circuit Breaker Stuck Open

```python
from app.distributed import get_host_breaker

# Reset a specific breaker
breaker = get_host_breaker("100.x.x.1")
breaker.reset()

# Or reset all breakers
from app.distributed.circuit_breaker import reset_all_breakers
reset_all_breakers()
```

#### WAL Recovery Issues

```python
from app.distributed import get_unified_wal

wal = get_unified_wal()

# Check pending entries
pending = await wal.get_pending_entries()
print(f"Pending: {len(pending)}")

# Force cleanup of old entries
await wal.cleanup(older_than_hours=24)
```

### Debugging

Enable verbose logging:

```python
import logging

# Enable distributed module logging
logging.getLogger("app.distributed").setLevel(logging.DEBUG)

# Or specific components
logging.getLogger("app.distributed.sync_coordinator").setLevel(logging.DEBUG)
logging.getLogger("app.distributed.circuit_breaker").setLevel(logging.DEBUG)
```

### Metrics

Key metrics to monitor:

| Metric                  | Description             | Alert Threshold |
| ----------------------- | ----------------------- | --------------- |
| `sync_duration_seconds` | Time for full sync      | > 600s          |
| `circuit_open_count`    | Open circuit breakers   | > 3             |
| `wal_pending_entries`   | Uncommitted WAL entries | > 100           |
| `p2p_request_errors`    | P2P request failures    | > 10/min        |

---

## Deprecated Modules

The following are deprecated and will be removed in future versions:

| Deprecated              | Replacement                         | Migration               |
| ----------------------- | ----------------------------------- | ----------------------- |
| `IngestionWAL`          | `UnifiedWAL`                        | Use `get_unified_wal()` |
| `ClusterCoordinator`    | `app.coordination.task_coordinator` | See coordination module |
| `WriteAheadLog` (alias) | `UnifiedWAL`                        | Direct replacement      |

### Migration Example

```python
# Old (deprecated)
from app.distributed import IngestionWAL
wal = IngestionWAL("data/wal")

# New
from app.distributed import get_unified_wal
wal = get_unified_wal()  # Uses unified WAL with same interface
```

---

## See Also

- `app/coordination/README.md` - Training pipeline coordination
- `app/training/README.md` - Training workflows
- `config/distributed_hosts.yaml` - Cluster configuration
- `CLAUDE.md` - Project-wide context

---

_Last updated: December 2025_
