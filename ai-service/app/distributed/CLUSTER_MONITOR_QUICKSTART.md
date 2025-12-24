# Cluster Monitor Quick Reference

## Installation

No additional dependencies required - uses existing infrastructure:

- `app.utils.game_discovery.RemoteGameDiscovery`
- SSH via subprocess
- PyYAML (already in requirements.txt)

## Common Commands

### Monitor entire cluster (one-time snapshot)

```bash
python -m app.distributed.cluster_monitor
```

### Live monitoring with auto-refresh

```bash
# Update every 10 seconds
python -m app.distributed.cluster_monitor --watch

# Update every 30 seconds
python -m app.distributed.cluster_monitor --watch --interval 30
```

### Fast checks (skip expensive operations)

```bash
# Only check connectivity and games
python -m app.distributed.cluster_monitor --no-training --no-disk

# Only check training status
python -m app.distributed.cluster_monitor --no-games --no-disk --watch
```

### Monitor specific nodes

```bash
# Check GPU training nodes only
python -m app.distributed.cluster_monitor --hosts lambda-gh200-a lambda-h100 lambda-2xh100

# Check all GH200 nodes
python -m app.distributed.cluster_monitor --hosts lambda-gh200-{a..p}
```

### Debugging

```bash
# Verbose output with sequential queries
python -m app.distributed.cluster_monitor --sequential --verbose

# Check problematic node with full details
python -m app.distributed.cluster_monitor --hosts vast-28918742 --verbose --timeout 30
```

## Programmatic API

### Basic usage

```python
from app.distributed.cluster_monitor import ClusterMonitor

monitor = ClusterMonitor()
status = monitor.get_cluster_status()

print(f"Active: {status.active_nodes}/{status.total_nodes}")
print(f"Games: {status.total_games:,}")
print(f"Training: {status.nodes_training} nodes")
```

### Check specific node

```python
node = monitor.get_node_status("lambda-gh200-a")

if node.reachable:
    print(f"Games: {node.total_games:,}")
    print(f"Training: {node.training_active}")
    print(f"Disk: {node.disk_usage_percent:.1f}%")
```

### Alert on conditions

```python
# Alert on low disk space
for host, node in status.nodes.items():
    if node.disk_usage_percent > 90:
        send_alert(f"{host} disk critical: {node.disk_usage_percent:.1f}%")

# Alert on training failures
for host, node in status.nodes.items():
    if node.role == "nn_training_primary" and not node.training_active:
        send_alert(f"{host} training node idle")
```

### Custom reporting

```python
# Find nodes with most games
top_nodes = sorted(
    [(h, n.total_games) for h, n in status.nodes.items()],
    key=lambda x: x[1],
    reverse=True
)[:5]

for host, games in top_nodes:
    print(f"{host}: {games:,} games")
```

## Output Examples

### Dashboard view

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
====================================================================================================
```

## Performance Tips

1. **Use --no-clear in watch mode** when piping to logs:

   ```bash
   python -m app.distributed.cluster_monitor --watch --no-clear >> cluster.log
   ```

2. **Skip expensive checks** for fast updates:

   ```bash
   python -m app.distributed.cluster_monitor --no-training --no-disk --interval 5
   ```

3. **Increase timeout** for slow connections:

   ```bash
   python -m app.distributed.cluster_monitor --timeout 30
   ```

4. **Use sequential mode** when debugging SSH issues:
   ```bash
   python -m app.distributed.cluster_monitor --sequential --verbose
   ```

## Integration Examples

### Cron job for periodic checks

```bash
# Add to crontab: check cluster every 5 minutes
*/5 * * * * cd /path/to/ai-service && python -m app.distributed.cluster_monitor >> /var/log/cluster.log 2>&1
```

### Alert script

```python
#!/usr/bin/env python3
from app.distributed.cluster_monitor import ClusterMonitor

monitor = ClusterMonitor()
status = monitor.get_cluster_status()

# Alert if too many nodes offline
if status.unreachable_nodes > 5:
    send_slack_alert(f"⚠️ {status.unreachable_nodes} nodes offline!")

# Alert on low disk space
for host, node in status.nodes.items():
    if node.reachable and node.disk_usage_percent > 85:
        send_slack_alert(f"💾 {host} low disk: {node.disk_usage_percent:.1f}%")
```

### Dashboard integration

```python
# Flask/FastAPI endpoint
@app.get("/api/cluster/status")
async def get_cluster_status():
    monitor = ClusterMonitor()
    status = monitor.get_cluster_status()
    return {
        "active_nodes": status.active_nodes,
        "total_nodes": status.total_nodes,
        "total_games": status.total_games,
        "nodes_training": status.nodes_training,
        "avg_disk_usage": status.avg_disk_usage,
    }
```

## Troubleshooting

### "Host unreachable" errors

- Check SSH key permissions: `chmod 600 ~/.ssh/id_cluster`
- Verify Tailscale is running
- Test SSH manually: `ssh -i ~/.ssh/id_cluster ubuntu@100.x.x.x`

### Timeout issues

- Increase timeout: `--timeout 30`
- Use sequential mode: `--sequential`
- Check network connectivity

### Import errors

- Ensure you're in ai-service directory
- Check Python path: `export PYTHONPATH=/path/to/ai-service:$PYTHONPATH`

### Missing game counts

- Verify RemoteGameDiscovery is working
- Check database paths in distributed_hosts.yaml
- Test on single node: `--hosts lambda-gh200-a --verbose`

## See Also

- Full documentation: `app/distributed/README_cluster_monitor.md`
- Example usage: `examples/cluster_monitor_example.py`
- Unit tests: `tests/test_cluster_monitor.py`
