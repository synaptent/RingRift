# Cluster Coordination Module

This package consolidates cluster-related coordination functionality.

## Modules

| Module         | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `health.py`    | Node and host health monitoring via `UnifiedHealthManager` |
| `transport.py` | Cluster transport layer (SSH, HTTP)                        |
| `p2p.py`       | Peer-to-peer mesh network backend                          |

## Usage

```python
from app.coordination.cluster.health import UnifiedHealthManager
from app.coordination.sync_facade import sync

# Health monitoring
health_manager = UnifiedHealthManager()
status = await health_manager.check_node_health("gh200-a")

# Sync operation
await sync("models", targets=["gh200-b"])
```

## Architecture

```
cluster/
├── health.py      # Health checks and status aggregation
├── transport.py   # Low-level transport (SSH connections)
└── p2p.py         # P2P mesh network operations
```

## December 2025 Consolidation

This package was created during the December 2025 module consolidation effort
(75 → 15 modules). It brings together cluster-specific coordination logic that
was previously scattered across multiple files.

## See Also

- `app.distributed.cluster_manifest` - Tracks data locations across cluster
- `app.coordination.providers` - Cloud provider-specific implementations
- `app.core.ssh` - Canonical SSH utilities
- `app.coordination.sync_facade` - Canonical sync entrypoint
