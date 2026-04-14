# Cluster Coordination Module

Canonical cluster coordination package exports for health, transport, and P2P helpers.

## Overview

`app.coordination.cluster` is a small package facade with three documented submodules:

| Package Export | Resolves To                          | Purpose                                   |
| -------------- | ------------------------------------ | ----------------------------------------- |
| `health`       | `app.coordination.cluster.health`    | Cluster and node health helpers           |
| `transport`    | `app.coordination.cluster_transport` | Transport-layer coordination helpers      |
| `p2p`          | `app.coordination.p2p_backend`       | Peer-to-peer backend coordination helpers |

Import the package surface through `app.coordination.cluster`; the transport and P2P exports are resolved lazily by the package.

## Usage

```python
from app.coordination.cluster import health
from app.coordination.sync_facade import sync

# Cluster health summary
summary = health.get_cluster_health_summary()
healthy_nodes = health.get_healthy_nodes()

# Canonical sync entrypoint
await sync("models", targets=["gh200-b"])
```

### Direct Health Submodule Access

```python
from app.coordination.cluster.health import (
    UnifiedHealthManager,
    get_health_manager,
    get_cluster_health_summary,
)

manager = get_health_manager()
system_health = manager.health_check()
cluster_health = get_cluster_health_summary()
```

## Architecture

```
app.coordination.cluster
├── health.py                    # Unified health re-exports
├── cluster_transport.py         # Lazy package export: cluster.transport
└── p2p_backend.py               # Lazy package export: cluster.p2p
```

## See Also

- `app.coordination.sync_facade` - canonical sync entrypoint
- `app.distributed.cluster_manifest` - tracks data locations across the cluster
- `app.coordination.providers` - cloud provider integrations
- `app.core.ssh` - lower-level SSH helpers
