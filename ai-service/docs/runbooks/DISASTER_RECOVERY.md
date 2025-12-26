# Runbook: Disaster Recovery

## Overview

**Severity:** Critical
**Component:** infrastructure
**Team:** infrastructure

This runbook covers recovery procedures for major cluster failures affecting the RingRift AI training infrastructure.

## Disaster Scenarios

| Scenario                 | Impact                                       | Recovery Time |
| ------------------------ | -------------------------------------------- | ------------- |
| Leader node failure      | Cluster coordination stops                   | 5-10 minutes  |
| P2P mesh collapse        | All inter-node communication fails           | 15-30 minutes |
| Data loss (single node)  | Games/training data lost from one node       | 30-60 minutes |
| Data loss (multi-node)   | Games/training data lost from multiple nodes | 1-4 hours     |
| Complete cluster failure | All nodes offline                            | 2-8 hours     |
| Model corruption         | Production models corrupted                  | 30-60 minutes |

---

## Scenario 1: Leader Node Failure

### Symptoms

- P2P status shows no leader
- Jobs not being distributed
- Selfplay processes running but not coordinated

### Diagnosis

```bash
# Check P2P status
curl -s http://localhost:8770/status | jq '.leader_id, .alive_peers'

# Check if leader election is stuck
curl -s http://localhost:8770/status | jq '.election_state'

# Check voter nodes
curl -s http://localhost:8770/status | jq '.voters'
```

### Recovery

```bash
# Option 1: Wait for automatic leader election (usually 30-60 seconds)
# Watch for new leader
watch -n 5 'curl -s http://localhost:8770/status | jq ".leader_id"'

# Option 2: Force leader election restart
# On any voter node:
curl -X POST http://localhost:8770/admin/election/reset \
  -H "Authorization: Bearer $CLUSTER_AUTH_TOKEN"

# Option 3: Manual leader designation (emergency only)
curl -X POST http://localhost:8770/admin/set-leader \
  -H "Authorization: Bearer $CLUSTER_AUTH_TOKEN" \
  -d '{"node_id": "nebius-backbone-1"}'
```

### Verification

```bash
# Confirm leader elected
curl -s http://localhost:8770/status | jq '.leader_id'

# Verify job distribution resumed
curl -s http://localhost:8770/jobs | jq '.active_jobs | length'
```

---

## Scenario 2: P2P Mesh Collapse

### Symptoms

- Most/all nodes show as "retired" or "unreachable"
- No heartbeats being received
- SSH still works to nodes

### Diagnosis

```bash
# Check which nodes are reachable via SSH
for host in nebius-backbone-1 runpod-h100 runpod-a100-1; do
  echo -n "$host: "
  timeout 5 ssh $host "echo OK" 2>/dev/null || echo "UNREACHABLE"
done

# Check P2P daemon status on nodes
ssh nebius-backbone-1 "pgrep -f 'p2p_orchestrator' && echo 'P2P running' || echo 'P2P stopped'"

# Check for network issues
ssh nebius-backbone-1 "curl -s --connect-timeout 5 http://localhost:8770/health"
```

### Recovery

```bash
# Step 1: Restart P2P on coordinator/leader node first
ssh nebius-backbone-1 "cd ~/ringrift/ai-service && \
  pkill -f p2p_orchestrator; \
  nohup python -m app.p2p.orchestrator > logs/p2p.log 2>&1 &"

# Step 2: Wait for coordinator to stabilize (30 seconds)
sleep 30

# Step 3: Restart P2P on other voter nodes
for host in runpod-h100 runpod-a100-1 runpod-a100-2; do
  ssh $host "cd /workspace/ringrift/ai-service && \
    pkill -f p2p_orchestrator; \
    nohup python -m app.p2p.orchestrator > logs/p2p.log 2>&1 &" &
done
wait

# Step 4: Let mesh reform (60 seconds)
sleep 60

# Step 5: Verify mesh health
curl -s http://localhost:8770/status | jq '.alive_peers, .leader_id'
```

### Prevention

- Increase `PEER_RETIRE_AFTER_SECONDS` to 86400 (24h)
- Enable SSH fallback in IdleResourceDaemon
- Monitor heartbeat health proactively

---

## Scenario 3: Data Loss (Single Node)

### Symptoms

- Node shows significantly fewer games than expected
- Database file corrupted or missing
- JSONL files not aggregating

### Diagnosis

```bash
# Check database size and game count
ssh $AFFECTED_HOST "cd /workspace/ringrift/ai-service && \
  sqlite3 data/games/selfplay.db 'SELECT COUNT(*) FROM games'"

# Check database integrity
ssh $AFFECTED_HOST "cd /workspace/ringrift/ai-service && \
  sqlite3 data/games/selfplay.db 'PRAGMA integrity_check'"

# List all game databases
ssh $AFFECTED_HOST "find data/games -name '*.db' -exec du -h {} \;"
```

### Recovery

```bash
# Option 1: Re-sync from cluster (if data exists elsewhere)
# Trigger sync FROM healthy nodes TO affected node
python -c "
from app.coordination.auto_sync_daemon import AutoSyncDaemon
import asyncio
daemon = AutoSyncDaemon()
asyncio.run(daemon.sync_to_host('$AFFECTED_HOST'))
"

# Option 2: Recover from backup (if available)
# Check NFS backup
ls -la /lambda/nfs/RingRift/backups/games/

# Restore from backup
scp /lambda/nfs/RingRift/backups/games/selfplay_latest.db \
  $AFFECTED_HOST:/workspace/ringrift/ai-service/data/games/selfplay.db

# Option 3: Accept loss and resume selfplay
# The training pipeline is resilient to partial data loss
# Simply restart selfplay on the node
ssh $AFFECTED_HOST "cd /workspace/ringrift/ai-service && \
  python scripts/selfplay.py --board hex8 --num-players 2 --engine gumbel &"
```

---

## Scenario 4: Data Loss (Multi-Node)

### Symptoms

- Multiple nodes report corrupted/missing databases
- Cluster-wide game count significantly lower than expected
- Training exports failing

### Diagnosis

```bash
# Get cluster-wide game counts
python -c "
from app.utils.game_discovery import RemoteGameDiscovery
rd = RemoteGameDiscovery()
totals = rd.get_cluster_total_by_config()
for config, count in sorted(totals.items()):
    print(f'{config}: {count:,}')
print(f'Total: {sum(totals.values()):,}')
"

# Identify which nodes have data vs which lost it
python -c "
from app.utils.game_discovery import RemoteGameDiscovery
rd = RemoteGameDiscovery()
for host, counts in rd.get_cluster_game_counts().items():
    total = sum(counts.values()) if counts else 0
    status = 'OK' if total > 1000 else 'LOW' if total > 0 else 'EMPTY'
    print(f'{host}: {total:,} ({status})')
"
```

### Recovery Priority

1. **Protect remaining data** - Stop writes to healthy nodes
2. **Aggregate surviving data** - Pull from all healthy nodes
3. **Restart selfplay** - Begin regenerating lost games

```bash
# Step 1: Aggregate data from healthy nodes to coordinator
python scripts/aggregate_cluster_data.py --target-host nebius-backbone-1

# Step 2: Create backup of aggregated data
ssh nebius-backbone-1 "cd ~/ringrift/ai-service && \
  cp -r data/games /lambda/nfs/RingRift/backups/games_$(date +%Y%m%d)"

# Step 3: Restart selfplay across cluster
python scripts/launch_selfplay_cluster.py --all-nodes

# Step 4: Monitor recovery progress
watch -n 60 'python -c "
from app.utils.game_discovery import RemoteGameDiscovery
rd = RemoteGameDiscovery()
print(sum(rd.get_cluster_total_by_config().values()))
"'
```

---

## Scenario 5: Complete Cluster Failure

### Symptoms

- All cloud nodes unreachable
- No P2P status available
- Training completely halted

### Recovery Steps

```bash
# Step 1: Verify failure scope
# Check cloud provider status pages
# - Vast.ai: https://status.vast.ai
# - Lambda: https://status.lambdalabs.com
# - RunPod: https://status.runpod.io

# Step 2: Check local coordinator (if available)
curl -s http://localhost:8770/status

# Step 3: Bring up coordinator first
# If running locally:
cd /Users/armand/Development/RingRift/ai-service
python -m app.p2p.orchestrator &

# Step 4: Reconnect cloud nodes as they come online
# They will auto-rejoin the P2P mesh

# Step 5: Trigger full cluster sync once nodes are online
python scripts/sync_cluster_data.py --full

# Step 6: Restart all daemons
python scripts/launch_daemons.py --all
```

---

## Scenario 6: Model Corruption

### Symptoms

- Model inference produces garbage
- Gauntlet tests failing
- Training producing NaN values

### Diagnosis

```bash
# Check model file integrity
python -c "
import torch
try:
    checkpoint = torch.load('models/canonical_hex8_2p.pth', weights_only=True)
    print(f'Model keys: {list(checkpoint.keys())}')
    print('Model OK')
except Exception as e:
    print(f'Model CORRUPTED: {e}')
"

# Quick gauntlet test
python scripts/quick_gauntlet.py \
  --model models/canonical_hex8_2p.pth \
  --board-type hex8 --num-players 2 \
  --games 10
```

### Recovery

```bash
# Option 1: Restore from backup
cp models/backup/canonical_hex8_2p.pth models/canonical_hex8_2p.pth

# Option 2: Pull from cluster
# Find healthy model copy on cluster
for host in nebius-backbone-1 runpod-h100; do
  echo -n "$host: "
  ssh $host "ls -la /workspace/ringrift/ai-service/models/canonical_hex8_2p.pth" 2>/dev/null && break
done

# Copy from healthy node
scp $HEALTHY_HOST:/workspace/ringrift/ai-service/models/canonical_hex8_2p.pth \
  models/canonical_hex8_2p.pth

# Option 3: Retrain from checkpoint
# Find latest valid checkpoint
ls -la models/checkpoints/ | grep hex8_2p

# Resume training from checkpoint
python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --resume models/checkpoints/hex8_2p_epoch_45.pth
```

---

## Recovery Checklist

After any disaster recovery:

- [ ] Verify P2P mesh is healthy: `curl -s http://localhost:8770/status | jq '.alive_peers'`
- [ ] Verify leader is elected: `curl -s http://localhost:8770/status | jq '.leader_id'`
- [ ] Verify game data is aggregating: `python -m app.distributed.cluster_monitor`
- [ ] Verify models are accessible: `ls -la models/canonical_*.pth`
- [ ] Verify selfplay is running: `ps aux | grep selfplay`
- [ ] Verify training pipeline works: `python -m app.training.train --dry-run`
- [ ] Check data quality: `python -m app.training.data_quality --all`
- [ ] Document incident in `/docs/incidents/YYYY-MM-DD_description.md`

---

## Emergency Contacts

- **Infrastructure Lead**: Check CLAUDE.md for current personnel
- **Cloud Provider Support**:
  - Vast.ai: support@vast.ai
  - Lambda Labs: support@lambdalabs.com
  - RunPod: support@runpod.io

---

## Related Runbooks

- [CLUSTER_HEALTH_CRITICAL.md](CLUSTER_HEALTH_CRITICAL.md) - Cluster health below 50%
- [COORDINATOR_ERROR.md](COORDINATOR_ERROR.md) - Coordinator errors
- [SYNC_HOST_CRITICAL.md](SYNC_HOST_CRITICAL.md) - Sync host issues
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General troubleshooting

## Revision History

| Date       | Change                                        |
| ---------- | --------------------------------------------- |
| 2025-12-26 | Initial version covering 6 disaster scenarios |
