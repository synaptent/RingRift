# Orphan Games Detection and Recovery Runbook

This runbook covers detection and recovery of orphan games - games generated on cluster nodes that haven't been synchronized to canonical databases.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: High

---

## Overview

Orphan games occur when:

- Nodes generate games but sync fails before data reaches coordinator
- Ephemeral nodes (Vast.ai, RunPod) terminate before syncing
- Network partitions prevent P2P data replication
- Games exist in `jsonl_aggregated.db` but not in `canonical_*.db`

Orphan games represent lost training data that could improve model quality.

---

## Detection Methods

### Method 1: OrphanDetectionDaemon Events

The daemon emits `ORPHAN_GAMES_DETECTED` when games are found:

```python
from app.coordination.event_router import subscribe

async def on_orphans_detected(event: dict) -> None:
    node = event.get("source_node")
    count = event.get("orphan_count")
    config = event.get("config_key")
    print(f"Found {count} orphan games for {config} on {node}")

subscribe("ORPHAN_GAMES_DETECTED", on_orphans_detected)
```

### Method 2: Manual Discovery

```bash
# Find all jsonl_aggregated.db files
find data/games -name "jsonl_aggregated.db" -exec \
  sh -c 'echo "=== {} ===" && sqlite3 {} "SELECT COUNT(*) FROM games"' \;

# Compare counts with canonical databases
for config in hex8_2p hex8_3p hex8_4p square8_2p; do
  canonical=$(sqlite3 "data/games/canonical_${config}.db" "SELECT COUNT(*) FROM games" 2>/dev/null || echo 0)
  aggregated=$(find data -name "jsonl_aggregated.db" -exec sqlite3 {} \
    "SELECT COUNT(*) FROM games WHERE board_type||'_'||num_players||'p' = '${config}'" 2>/dev/null \; | \
    awk '{sum+=$1}END{print sum}')
  echo "$config: canonical=$canonical, aggregated=$aggregated, potential_orphans=$((aggregated - canonical))"
done
```

### Method 3: Cluster-Wide Discovery

```python
from app.utils.game_discovery import RemoteGameDiscovery

discovery = RemoteGameDiscovery()
for node in discovery.get_cluster_nodes():
    dbs = discovery.find_databases_on_node(node)
    for db in dbs:
        if "jsonl_aggregated" in str(db.path):
            print(f"Potential orphans on {node}: {db.game_count} games")
```

---

## Diagnosis

### Check Sync Status

```bash
# Check last sync time for each node
python -c "
from app.coordination.auto_sync_daemon import get_auto_sync_daemon
daemon = get_auto_sync_daemon()
for node, status in daemon.get_sync_status().items():
    print(f'{node}: last_sync={status.last_sync_time}, pending={status.pending_files}')
"
```

### Identify Orphan Sources

```bash
# List nodes with unsynchronized data
curl -s http://localhost:8770/cluster_manifest | python3 -c '
import sys, json
manifest = json.load(sys.stdin)
for node, data in manifest.get("nodes", {}).items():
    games = data.get("games", 0)
    synced = data.get("synced_games", 0)
    if games > synced:
        print(f"{node}: {games - synced} orphans ({games} total, {synced} synced)")
'
```

### Check Event Wiring

Verify orphan detection events are being emitted:

```bash
# Enable event tracing
export RINGRIFT_EVENT_TRACE=1

# Check for orphan events in logs
grep "ORPHAN_GAMES" logs/coordination.log | tail -20
```

---

## Recovery Procedures

### Option 1: Trigger Priority Sync

For nodes that are still reachable:

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()

# Sync specific node
response = await facade.trigger_priority_sync(
    reason="orphan_games_recovery",
    source_node="vast-12345",
    config_key="hex8_4p",
    data_type="games",
)
print(f"Sync result: {response}")
```

### Option 2: Manual Rsync

For nodes with network issues:

```bash
# Identify orphan-containing node
NODE_IP="100.123.45.67"
NODE_USER="root"

# Sync game databases
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_cluster" \
  ${NODE_USER}@${NODE_IP}:/root/ringrift/ai-service/data/games/*.db \
  /Volumes/RingRift-Data/cluster_games/recovered/

# Consolidate into canonical databases
python scripts/consolidate_jsonl_databases.py \
  --source /Volumes/RingRift-Data/cluster_games/recovered \
  --output data/games
```

### Option 3: Database Consolidation

Merge scattered databases into canonical databases:

```bash
# Run consolidation script
python scripts/consolidate_jsonl_databases.py \
  --strict \
  --deduplicate-by game_id

# Verify consolidation
for db in data/games/canonical_*.db; do
  echo "=== $db ==="
  sqlite3 "$db" "SELECT COUNT(*) AS games FROM games"
done
```

### Option 4: Recover from Terminating Node

For ephemeral nodes about to terminate:

```bash
# 1. Connect to node before termination
ssh -i ~/.ssh/id_cluster root@${NODE_IP}

# 2. On the node, check for unsynchronized data
sqlite3 data/games/jsonl_aggregated.db "SELECT COUNT(*) FROM games"

# 3. From coordinator, pull data immediately
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_cluster" \
  root@${NODE_IP}:/root/ringrift/ai-service/data/games/ \
  /Volumes/RingRift-Data/emergency_recovery/${NODE_NAME}/
```

---

## Automated Recovery

### Enable OrphanDetectionDaemon

```bash
# In master_loop.py or launch_daemons.py
export RINGRIFT_ORPHAN_DETECTION_ENABLED=true
export RINGRIFT_ORPHAN_CHECK_INTERVAL=1800  # 30 minutes
```

### Event-Driven Recovery Flow

The system automatically handles orphans when properly wired:

```
OrphanDetectionDaemon
    ↓ (emits ORPHAN_GAMES_DETECTED)
DataPipelineOrchestrator._on_orphan_games_detected()
    ↓ (triggers priority sync)
SyncFacade.trigger_priority_sync()
    ↓ (syncs data, emits DATA_SYNC_COMPLETED)
DataPipelineOrchestrator._on_sync_completed()
    ↓ (emits NEW_GAMES_AVAILABLE)
Training pipeline continues...
```

---

## Prevention

### 1. Ephemeral Node Sync

Configure aggressive sync for termination-prone hosts:

```yaml
# In distributed_hosts.yaml
vast-*:
  sync_priority: high
  sync_interval_seconds: 60
  sync_on_idle: true
```

### 2. Termination Hooks

Vast.ai and RunPod support termination hooks:

```bash
# Set up pre-termination sync
curl -X POST http://localhost:8770/register_termination_hook \
  -H "Content-Type: application/json" \
  -d '{"action": "sync_all_data", "timeout": 300}'
```

### 3. Gossip Replication

Enable P2P gossip for faster data distribution via AutoSync config:

```yaml
# config/distributed_hosts.yaml
auto_sync:
  enabled: true
  strategy: hybrid
  gossip_interval_seconds: 15
```

---

## Metrics and Monitoring

### Key Metrics

| Metric                 | Alert Threshold | Source                |
| ---------------------- | --------------- | --------------------- |
| orphan_games_count     | > 1000          | OrphanDetectionDaemon |
| sync_failures_per_hour | > 5             | AutoSyncDaemon        |
| time_since_last_sync   | > 1 hour        | SyncFacade            |

### Monitoring Query

```python
from app.coordination.sync_facade import get_sync_facade

facade = get_sync_facade()
stats = facade.get_stats()

print(f"Orphans detected: {stats.orphans_detected}")
print(f"Orphans recovered: {stats.orphans_recovered}")
print(f"Recovery success rate: {stats.recovery_rate:.1%}")
```

---

## Related Documentation

- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Sync procedures
- [SYNC_HOST_CRITICAL.md](SYNC_HOST_CRITICAL.md) - Critical sync hosts
- [GAME_DATA_CORRUPTION_RECOVERY.md](GAME_DATA_CORRUPTION_RECOVERY.md) - Data corruption
