# Cluster Monitoring Dashboard

Production-ready cluster monitoring tool for RingRift distributed infrastructure.

## Features

- **Game Count Monitoring**: Query game databases across all cluster nodes via `RemoteGameDiscovery`
- **Training Status**: Detect active training processes on each node
- **Disk Usage**: Monitor filesystem usage and available space
- **Data Sync Status**: Track synchronization state between nodes
- **Real-time Updates**: Watch mode for continuous monitoring
- **Parallel Queries**: Concurrent SSH connections for fast updates
- **Error Handling**: Comprehensive timeout and error handling

## Quick Start

### CLI Usage

```bash
# Single snapshot of entire cluster
python -m app.distributed.cluster_monitor

# Watch mode with 10-second updates
python -m app.distributed.cluster_monitor --watch --interval 10

# Monitor specific hosts only
python -m app.distributed.cluster_monitor --hosts lambda-gh200-a lambda-h100 lambda-2xh100

# Skip expensive checks for faster updates
python -m app.distributed.cluster_monitor --watch --no-training --no-disk --interval 5

# Include sync status (slower)
python -m app.distributed.cluster_monitor --sync
```

### Programmatic Usage

```python
from app.distributed.cluster_monitor import ClusterMonitor

# Create monitor
monitor = ClusterMonitor()

# Get cluster-wide status
status = monitor.get_cluster_status()
print(f"Total games: {status.total_games:,}")
print(f"Active nodes: {status.active_nodes}/{status.total_nodes}")
print(f"Nodes training: {status.nodes_training}")

# Get status for specific node
node_status = monitor.get_node_status("lambda-gh200-a")
if node_status.reachable:
    print(f"Games: {node_status.total_games:,}")
    print(f"Training: {node_status.training_active}")
    print(f"Disk: {node_status.disk_usage_percent:.1f}%")

# Print formatted dashboard
monitor.print_dashboard(status)

# Watch mode (continuous updates)
monitor.watch(interval=10)
```

## Architecture

### Data Sources

1. **RemoteGameDiscovery**: Queries game database counts via SSH
   - Uses existing `app.utils.game_discovery.RemoteGameDiscovery`
   - Caching disabled for real-time accuracy

2. **SSH Process Monitoring**: Detects training processes
   - Searches for `train.py`, `train_*.py` patterns
   - Parses process info (PID, CPU, memory, command)

3. **Disk Usage**: Queries filesystem via `df`
   - Monitors ringrift directory's filesystem
   - Reports usage percentage and free space

4. **Sync Status**: Checks data synchronization
   - Counts pending files in sync queues
   - Calculates lag vs coordinator (future)

### Configuration

Reads from `config/distributed_hosts.yaml`:

```yaml
hosts:
  lambda-gh200-a:
    ssh_host: 192.222.51.29
    tailscale_ip: 100.123.183.70
    ssh_user: ubuntu
    ssh_key: ~/.ssh/id_cluster
    ringrift_path: ~/ringrift/ai-service
    role: nn_training_primary
    status: ready
    gpu: NVIDIA GH200 (96GB)
```

## Output Format

### Dashboard View

```
====================================================================================================
RINGRIFT CLUSTER MONITOR - 2025-12-23 19:30:00
====================================================================================================

Cluster Summary:
  Nodes: 28/30 online
  Total Games: 1,234,567
  Training: 4 nodes active (6 processes)
  Disk: 45.2% avg usage, 12000GB free
  Query Time: 2.34s

Game Counts by Configuration:
  hex8_2p              123,456
  square8_2p           234,567
  hexagonal_3p          45,678

Node Status:
Host                      Status     Games        Training     Disk            Response
----------------------------------------------------------------------------------------------------
lambda-gh200-a            ONLINE     45,678       YES (2)      42.3% (580GB)   120ms
lambda-gh200-b            ONLINE     34,567       -            38.1% (620GB)   135ms
lambda-h100               ONLINE     56,789       YES (1)      55.2% (450GB)   145ms
vast-28918742             OFFLINE    -            -            -               -

Errors:
  - vast-28918742: Host unreachable
====================================================================================================
```

## CLI Options

| Option                   | Description                              |
| ------------------------ | ---------------------------------------- |
| `--watch`                | Enable continuous monitoring mode        |
| `--interval N`           | Update interval in seconds (default: 10) |
| `--hosts HOST [HOST...]` | Monitor specific hosts only              |
| `--no-clear`             | Don't clear screen between updates       |
| `--no-games`             | Skip game count queries                  |
| `--no-training`          | Skip training status checks              |
| `--no-disk`              | Skip disk usage checks                   |
| `--sync`                 | Include sync status (expensive)          |
| `--timeout N`            | SSH timeout in seconds (default: 15)     |
| `--sequential`           | Query nodes sequentially (slower)        |
| `--verbose, -v`          | Enable debug logging                     |

## Performance

- **Parallel mode**: Queries all nodes concurrently (default)
  - 30 nodes in ~2-5 seconds with good connectivity
- **Sequential mode**: Queries nodes one by one
  - Useful for debugging connection issues
- **Timeouts**: Configurable SSH timeout (default: 15s)
  - Failed nodes don't block cluster-wide queries

## Use Cases

### Daily Operations

```bash
# Quick cluster health check
python -m app.distributed.cluster_monitor

# Monitor during training
python -m app.distributed.cluster_monitor --watch --interval 30
```

### Debugging

```bash
# Verbose output with sequential queries
python -m app.distributed.cluster_monitor --sequential --verbose

# Check specific problematic node
python -m app.distributed.cluster_monitor --hosts vast-28918742 --verbose
```

### Integration

```python
# Alert on low disk space
monitor = ClusterMonitor()
status = monitor.get_cluster_status()

for host_name, node in status.nodes.items():
    if node.disk_usage_percent > 90:
        send_alert(f"{host_name} disk usage: {node.disk_usage_percent:.1f}%")

# Track training progress
for host_name, node in status.nodes.items():
    if node.training_active:
        log_training_status(host_name, node.training_processes)
```

## Error Handling

- **Connection failures**: Marked as unreachable, doesn't block other nodes
- **Timeouts**: Configurable per-operation timeouts
- **SSH errors**: Caught and logged, partial results returned
- **Missing config**: Graceful degradation if hosts.yaml not found

## Future Enhancements

- [ ] GPU utilization monitoring (nvidia-smi)
- [ ] Memory usage tracking
- [ ] Network bandwidth monitoring
- [ ] Historical trends and graphs
- [ ] Alerting integration (email, Slack)
- [ ] Web UI dashboard
- [ ] Prometheus metrics export
- [ ] Node comparison views
- [ ] Automatic issue detection
