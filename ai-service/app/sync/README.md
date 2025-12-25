# Sync Module

Cluster synchronization utilities for the RingRift AI service. Provides host discovery, connectivity checking, and coordination for distributed training across the GPU cluster.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Core Components](#core-components)
   - [ClusterNode](#clusternode)
   - [EloSyncConfig](#elosyncconfig)
4. [Host Discovery](#host-discovery)
5. [Connectivity Checking](#connectivity-checking)
6. [Sync URL Generation](#sync-url-generation)
7. [Usage Examples](#usage-examples)
8. [Integration](#integration)

---

## Overview

The sync module centralizes cluster host configuration loading from `distributed_hosts.yaml`, eliminating duplication across sync components.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Sync Module                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │           distributed_hosts.yaml                        │    │
│   │  • Host definitions (IP, SSH, role, GPU specs)         │    │
│   │  • Elo sync configuration                              │    │
│   │  • Data server ports                                   │    │
│   └───────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│   ┌────────────────────────────────────────────────────────┐    │
│   │              load_hosts_config()                        │    │
│   │         (YAML parsing with fallback)                    │    │
│   └───────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│           ┌───────────────┼───────────────┐                      │
│           ▼               ▼               ▼                      │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│   │ ClusterNode  │ │ EloSyncConfig│ │   Discovery  │            │
│   │   objects    │ │    object    │ │  functions   │            │
│   └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Used By

- `scripts/elo_db_sync.py` - Elo database synchronization
- `scripts/aria2_data_sync.py` - Model and data sync
- `scripts/validate_cluster_elo.py` - Elo validation
- `app/training/elo_reconciliation.py` - Elo drift reconciliation
- `app/distributed/cluster_monitor.py` - Cluster monitoring

---

## Configuration

Configuration is loaded from `config/distributed_hosts.yaml`:

```yaml
hosts:
  coordinator-node:
    tailscale_ip: '100.x.x.x'
    ssh_host: 'coordinator.local'
    ssh_user: ubuntu
    ssh_port: 22
    ringrift_path: '~/ringrift/ai-service'
    status: active
    role: coordinator
    memory_gb: 64
    cpus: 12
    gpu: ''
    data_server_port: 8766

  gpu-node-1:
    tailscale_ip: '100.x.x.x'
    ssh_host: 'ubuntu@100.x.x.x'
    ssh_key: '~/.ssh/id_cluster'
    status: active
    role: worker
    memory_gb: 96
    cpus: 72
    gpu: 'H100' # or GH200, A100, etc.
    data_server_port: 8766

elo_sync:
  coordinator: coordinator-node
  sync_port: 8766
  sync_interval: 300
  divergence_threshold: 50
  transports:
    - tailscale
    - aria2
    - http
```

### Default Ports

| Port   | Purpose              |
| ------ | -------------------- |
| `8765` | Model sync server    |
| `8766` | Data sync / Elo sync |

---

## Core Components

### ClusterNode

Represents a cluster node with connectivity info:

```python
from app.sync import ClusterNode

@dataclass
class ClusterNode:
    name: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    ssh_user: str = "ubuntu"
    ssh_key: str | None = None
    ssh_port: int = 22
    ringrift_path: str = "~/ringrift/ai-service"
    status: str = "unknown"      # active, terminated, offline, setup
    role: str = "unknown"        # coordinator, worker
    memory_gb: int = 0
    cpus: int = 0
    gpu: str = ""
    data_server_port: int = 8766
    data_server_url: str | None = None
```

#### Properties

```python
# Get best IP for connection (prefers Tailscale)
ip = node.best_ip  # "100.x.x.x"

# Get data server URL
url = node.data_server_base_url  # "http://100.x.x.x:8766"

# Check if node is active
if node.is_active:
    # status not in ("terminated", "offline", "setup")
    ...
```

### EloSyncConfig

Elo synchronization configuration:

```python
from app.sync import EloSyncConfig

@dataclass
class EloSyncConfig:
    coordinator: str = "mac-studio"
    sync_port: int = 8766
    sync_interval: int = 300          # seconds
    divergence_threshold: int = 50    # Elo points
    transports: list[str] = ["tailscale", "aria2", "http"]
```

---

## Host Discovery

### Loading Configuration

```python
from app.sync import load_hosts_config, get_cluster_nodes

# Raw config dictionary
config = load_hosts_config()
# {"hosts": {...}, "elo_sync": {...}}

# Get all nodes as ClusterNode objects
nodes = get_cluster_nodes()
for name, node in nodes.items():
    print(f"{name}: {node.best_ip} ({node.gpu})")
```

### Getting Specific Nodes

```python
from app.sync import (
    get_active_nodes,
    get_coordinator_node,
    get_coordinator_address,
    get_elo_sync_config,
)

# All active nodes (excludes terminated/offline)
active = get_active_nodes()
print(f"Active nodes: {len(active)}")

# Get coordinator node
coord = get_coordinator_node()
if coord:
    print(f"Coordinator: {coord.name} at {coord.best_ip}")

# Get coordinator address (IP, port)
ip, port = get_coordinator_address()
# ("100.x.x.x", 8766)

# Elo sync configuration
sync_config = get_elo_sync_config()
print(f"Sync interval: {sync_config.sync_interval}s")
```

---

## Connectivity Checking

### Single Node Check

```python
from app.sync import check_node_reachable

node = get_coordinator_node()
if check_node_reachable(node, port=8766, timeout=5):
    print(f"{node.name} is reachable")
```

### Parallel Discovery

```python
from app.sync import discover_reachable_nodes

# Discover all reachable nodes (parallel checks)
reachable = discover_reachable_nodes(port=8766, timeout=5)

for node, status in reachable:
    print(f"{node.name}: {status.get('version', 'unknown')}")
```

### HTTP Endpoint Check

```python
from app.sync.cluster_hosts import check_http_endpoint

# Check specific endpoint
status = check_http_endpoint(
    ip="100.x.x.x",
    port=8766,
    path="/status",
    timeout=5,
)
if status:
    print(f"Games: {status.get('game_count', 0)}")
```

---

## Sync URL Generation

### Elo Sync URLs

```python
from app.sync import get_elo_sync_urls

# Get URLs for all reachable Elo sync endpoints
urls = get_elo_sync_urls()
# ["http://100.x.x.1:8766/db", "http://100.x.x.2:8766/db", ...]
```

### Data Sync URLs

```python
from app.sync import get_data_sync_urls

# Get data sync URLs (excludes self by default)
urls = get_data_sync_urls(
    exclude_self=True,      # Exclude current host
    reachable_only=True,    # Only include reachable nodes
    timeout=5,
)
# ["http://100.x.x.1:8766", "http://100.x.x.2:8766", ...]
```

### Generic Sync URLs

```python
from app.sync import get_sync_urls

# Custom port and path
urls = get_sync_urls(port=8765, path="/models")
```

---

## Usage Examples

### Cluster Status Check

```python
from app.sync import (
    get_active_nodes,
    discover_reachable_nodes,
)

print("Cluster Status")
print("=" * 50)

active = get_active_nodes()
reachable = discover_reachable_nodes()

print(f"Configured nodes: {len(active)}")
print(f"Reachable nodes: {len(reachable)}")
print()

for node, status in reachable:
    games = status.get("game_count", 0)
    version = status.get("version", "?")
    print(f"  {node.name}: {games} games (v{version})")
```

### Elo Database Sync

```python
from app.sync import (
    get_elo_sync_urls,
    get_coordinator_address,
)
import sqlite3
import urllib.request

# Get coordinator
coord_ip, coord_port = get_coordinator_address()
if not coord_ip:
    print("No coordinator configured")
    exit(1)

# Sync from all nodes to coordinator
urls = get_elo_sync_urls()
for url in urls:
    print(f"Syncing from {url}...")
    # Download and merge database
    # ...
```

### Data Collection from Cluster

```python
from app.sync import get_data_sync_urls

# Get game data from all cluster nodes
urls = get_data_sync_urls(reachable_only=True)

total_games = 0
for url in urls:
    try:
        response = urllib.request.urlopen(f"{url}/status", timeout=5)
        status = json.loads(response.read())
        total_games += status.get("game_count", 0)
    except Exception as e:
        print(f"Error checking {url}: {e}")

print(f"Total games across cluster: {total_games}")
```

### SSH Command Execution

```python
from app.sync import get_active_nodes
import subprocess

nodes = get_active_nodes()

for node in nodes:
    if not node.ssh_host:
        continue

    # Build SSH command
    ssh_cmd = ["ssh"]
    if node.ssh_key:
        ssh_cmd.extend(["-i", node.ssh_key])
    ssh_cmd.extend(["-p", str(node.ssh_port)])
    ssh_cmd.append(f"{node.ssh_user}@{node.best_ip}")
    ssh_cmd.append("uptime")

    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
    print(f"{node.name}: {result.stdout.strip()}")
```

---

## Integration

### Environment Variable Fallback

If the coordinator is not found in config:

```bash
export RINGRIFT_COORDINATOR_IP=<your-coordinator-ip>
```

```python
from app.sync import get_coordinator_address

ip, port = get_coordinator_address()
# Uses RINGRIFT_COORDINATOR_IP if coordinator not in config
```

### With Distributed Config

The module integrates with the unified config system:

```python
from app.config.unified_config import get_config

config = get_config()
data_server_port = config.distributed.data_server_port
```

### With Cluster Monitor

```python
from app.sync import get_active_nodes
from app.distributed.cluster_monitor import ClusterMonitor

nodes = get_active_nodes()
monitor = ClusterMonitor(nodes=[n.name for n in nodes])
status = monitor.get_status()
```

---

## Module Structure

| File               | Lines | Description                               |
| ------------------ | ----- | ----------------------------------------- |
| `cluster_hosts.py` | ~298  | Host discovery and connectivity utilities |
| `__init__.py`      | ~38   | Public API exports                        |

---

## See Also

- `config/distributed_hosts.yaml` - Cluster configuration
- `app/distributed/README.md` - Distributed training infrastructure
- `scripts/elo_db_sync.py` - Elo sync implementation
- `scripts/aria2_data_sync.py` - Data sync implementation

---

_Last updated: December 2025_
