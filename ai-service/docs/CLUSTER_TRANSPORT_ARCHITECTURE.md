# Cluster Transport Architecture

This guide explains the multi-transport communication layer for RingRift cluster synchronization.

## Overview

The cluster transport layer provides reliable file and data transfer across distributed nodes with automatic failover between multiple transport mechanisms.

**Key Components:**

- **`ClusterTransport`** (`app/coordination/cluster_transport.py`) - Main transport class with failover logic
- **`DatabaseSyncManager`** (`app/coordination/database_sync_manager.py`) - Base class for database sync
- **`CircuitBreaker`** (`app/distributed/circuit_breaker.py`) - Per-node fault tolerance
- **`transfer.py`** (`scripts/lib/transfer.py`) - Low-level transfer utilities

## Transport Failover Order

When transferring files, transports are tried in order of preference:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Tailscale  │ -> │   SSH/rsync │ -> │   Base64    │ -> │    HTTP     │
│ (direct IP) │    │ (hostname)  │    │ (text-safe) │    │ (API)       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     Fast              Reliable         Workaround         Flexible
```

### When Each Transport is Selected

| Transport     | When Used                          | Advantages                           | Disadvantages                    |
| ------------- | ---------------------------------- | ------------------------------------ | -------------------------------- |
| **Tailscale** | Node has `tailscale_ip` configured | Fastest, bypasses NAT                | Requires Tailscale on both ends  |
| **SSH/rsync** | Standard fallback                  | Resume support, reliable             | Can fail with proxies/firewalls  |
| **Base64**    | Binary streams fail                | Works when SSH corrupts binary       | 33% larger payload, memory-heavy |
| **HTTP**      | For API operations                 | Most flexible, works through proxies | Requires HTTP server running     |

## Usage Examples

### Basic File Transfer

```python
from app.coordination.cluster_transport import (
    ClusterTransport,
    NodeConfig,
)

transport = ClusterTransport()

# Configure target node
node = NodeConfig(
    hostname="runpod-h100",
    tailscale_ip="100.123.45.67",  # Optional
    ssh_port=22,
    base_path="ai-service",
)

# Push a file (automatic failover)
result = await transport.transfer_file(
    local_path=Path("data/model.pth"),
    remote_path="models/model.pth",
    node=node,
    direction="push",
)

if result.success:
    print(f"Transferred via {result.transport_used} in {result.latency_ms:.1f}ms")
else:
    print(f"Failed: {result.error}")
```

### HTTP Request with Failover

```python
# Try Tailscale first, then hostname
result = await transport.http_request_with_failover(
    node=node,
    endpoint="/api/status",
    method="GET",
)
```

### Database Sync (Subclass Pattern)

```python
from app.coordination.database_sync_manager import DatabaseSyncManager

class MySyncManager(DatabaseSyncManager):
    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """Merge remote database into local."""
        # Type-specific merge logic
        pass

    def _update_local_stats(self) -> None:
        """Update local record count and hash."""
        pass

    def _get_remote_db_path(self) -> str:
        """Return remote database path."""
        return "data/my_database.db"

    def _get_remote_count_query(self) -> str:
        """SQL query to count remote records."""
        return "SELECT COUNT(*) FROM my_table"
```

## Circuit Breaker Pattern

Each node has an independent circuit breaker that tracks failures:

```
     ┌──────────────────────────────────────────┐
     │              Circuit States               │
     ├──────────────────────────────────────────┤
     │  CLOSED  ──(3 failures)──>  OPEN         │
     │     ↑                           │        │
     │     │                      (5 min wait)  │
     │     │                           ↓        │
     │     └─(success)──  HALF_OPEN ──(fail)──┘ │
     └──────────────────────────────────────────┘
```

**Default thresholds:**

- `failure_threshold`: 3 consecutive failures before opening
- `recovery_timeout`: 300 seconds (5 minutes) before retry

**Checking circuit state:**

```python
if transport.can_attempt("runpod-h100"):
    # Circuit is closed or half-open, safe to try
    result = await transport.transfer_file(...)
else:
    # Circuit is open, skip this node
    pass

# View all circuit states
health = transport.get_health_summary()
for node, status in health.items():
    print(f"{node}: {status['state']} ({status['failures']} failures)")
```

## Base64 Transport (Connection Reset Workaround)

When rsync/scp connections reset with "Connection reset by peer" errors, the base64 transport provides a reliable fallback.

### Why This Happens

Some network configurations corrupt binary SSH streams:

- Corporate firewalls inspecting SSH traffic
- NAT gateways with connection tracking issues
- Proxies that modify binary data

### How Base64 Works

```
Push: cat file | base64 | ssh host 'base64 -d > file'
Pull: ssh host 'base64 file' | base64 -d > file
```

The file is encoded as text-safe base64, transmitted as ASCII, then decoded on the other end.

### Manual Base64 Transfer

```bash
# When SCP fails with "Connection reset by peer"
cat local_file.npz | base64 | ssh user@host 'base64 -d > remote_file.npz'
```

### Scripted Base64 Transfer

```python
from scripts.lib.transfer import base64_push, robust_push, TransferConfig

# Direct base64 transfer
result = base64_push("file.npz", "host", 22, "/path/file.npz", TransferConfig())

# Auto-failover (tries rsync -> scp -> base64)
result = robust_push("file.npz", "host", 22, "/path/file.npz", TransferConfig())
```

### Memory Considerations

Base64 encodes entire files in memory. For files >100MB, consider:

1. Chunked transfer (not yet implemented)
2. Using rsync with `--partial` for resume
3. HTTP transfer with streaming

## Configuration

### NodeConfig Fields

```python
@dataclass
class NodeConfig:
    hostname: str           # Required: hostname or IP
    tailscale_ip: str | None = None  # Optional: Tailscale IP (100.x.x.x)
    ssh_port: int = 22      # SSH port
    http_port: int = 8080   # HTTP port for API requests
    http_scheme: str = "http"  # HTTP or HTTPS
    base_path: str = "ai-service"  # Remote base directory
```

### TransportConfig Presets

```python
from app.coordination.cluster_transport import TransportConfig

# For large file transfers (models, NPZ files)
config = TransportConfig.for_large_transfers()
# connect_timeout: 60s, operation_timeout: 600s

# For quick API requests
config = TransportConfig.for_quick_requests()
# connect_timeout: 10s, operation_timeout: 30s
```

### Timeout Configuration

Timeouts are centralized in `app/config/thresholds.py`:

| Constant                            | Default | Purpose                       |
| ----------------------------------- | ------- | ----------------------------- |
| `CLUSTER_CONNECT_TIMEOUT`           | 30s     | SSH/Tailscale connection      |
| `CLUSTER_OPERATION_TIMEOUT`         | 180s    | Transfer/operation timeout    |
| `HTTP_TIMEOUT`                      | 30s     | HTTP request timeout          |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 3       | Failures before circuit opens |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT`  | 300s    | Wait before retry             |

## Debugging Common Failures

### "Connection reset by peer"

**Symptom:** rsync/scp fails mid-transfer with connection reset.

**Solutions:**

1. ClusterTransport automatically falls back to base64
2. For manual transfers: `cat file | base64 | ssh ... 'base64 -d > file'`
3. Check if firewall is inspecting SSH traffic

### "Circuit breaker open"

**Symptom:** All operations to a node fail immediately.

**Solutions:**

1. Wait for recovery timeout (5 minutes by default)
2. Manually reset: `transport.reset_circuit_breakers()`
3. Check node health: is SSH/HTTP server running?

### "No Tailscale IP"

**Symptom:** Tailscale transport skipped, falls back to SSH.

**Solutions:**

1. Add `tailscale_ip` to NodeConfig if available
2. Check Tailscale status on both nodes: `tailscale status`
3. This is not an error - SSH fallback is normal

### "HTTP request timeout"

**Symptom:** HTTP API calls fail with timeout.

**Solutions:**

1. Check if HTTP server is running on target node
2. Increase `http_timeout` in TransportConfig
3. Try `http_request_with_failover()` to try Tailscale first

### Large File Transfer Failures

**Symptom:** Transfers of files >100MB fail or timeout.

**Solutions:**

1. Use `TransportConfig.for_large_transfers()` for longer timeouts
2. Prefer rsync over SCP (has resume support)
3. For repeated failures, consider HTTP streaming or chunked transfer

## Integration with Sync Managers

The transport layer integrates with sync managers through `DatabaseSyncManager`:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (EloSyncManager, RegistrySyncManager, ModelDistributor)    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  DatabaseSyncManager                         │
│  - Node discovery (P2P/YAML)                                │
│  - Merge conflict resolution                                 │
│  - Background sync loop                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   ClusterTransport                           │
│  - Multi-transport failover                                  │
│  - Circuit breaker per node                                  │
│  - Timeout/retry handling                                    │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Transport Selection in Sync

```python
# From database_sync_manager.py
async def _sync_from_node(self, node: SyncNodeInfo) -> bool:
    # try_transports() attempts each transport in order
    result = await try_transports(
        node=node,
        operation=self._fetch_and_merge,
        transports=["tailscale", "ssh", "vast_ssh", "http"],
    )
    return result.success
```

## Performance Characteristics

| Transport | Throughput | Latency | Resume    | Memory           |
| --------- | ---------- | ------- | --------- | ---------------- |
| Tailscale | High       | Low     | Via rsync | Low              |
| SSH/rsync | High       | Medium  | Yes       | Low              |
| Base64    | Medium     | Medium  | No        | High (file size) |
| HTTP      | Variable   | Medium  | No        | Streaming        |

**Recommendations:**

- Use Tailscale for high-frequency small transfers
- Use rsync for large files that may need resume
- Use base64 only when binary streams fail
- Use HTTP for API operations and status checks

---

_Last updated: December 27, 2025_
