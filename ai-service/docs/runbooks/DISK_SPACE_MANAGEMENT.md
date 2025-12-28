# Runbook: Disk Space Management

**Severity**: High
**Expected Resolution Time**: 10-30 minutes
**Last Updated**: December 28, 2025

---

## Symptoms

- `DISK_SPACE_LOW` events emitted
- Sync failures with "no space left on device"
- Training failing to save checkpoints
- Selfplay databases not writing

---

## Quick Diagnosis

```bash
# Check disk usage on local node
df -h /Users/armand/Development/RingRift/ai-service/data

# Check all cluster nodes
for node in nebius-h100-3 runpod-h100 vast-5090; do
  echo "=== $node ==="
  ssh $node 'df -h /workspace 2>/dev/null || df -h ~'
done

# Check largest files
du -sh /Users/armand/Development/RingRift/ai-service/data/* | sort -h | tail -20

# Check for runaway log files
find /Users/armand/Development/RingRift/ai-service/logs -name "*.log" -size +100M
```

---

## Thresholds

| Level     | Usage  | Action                        |
| --------- | ------ | ----------------------------- |
| Normal    | < 60%  | No action needed              |
| Warning   | 60-70% | Proactive cleanup recommended |
| Critical  | > 70%  | Immediate cleanup required    |
| Emergency | > 90%  | Data at risk, stop ingestion  |

---

## Common Causes & Fixes

### 1. Accumulated Game Databases

**Symptoms**: Many .db files in data/games/
**Check**:

```bash
du -sh data/games/*.db | sort -h | tail -20
ls -la data/games/*.db | wc -l
```

**Fix**: Consolidate and clean

```bash
# Consolidate scattered databases into canonical
python scripts/consolidate_jsonl_databases.py

# Delete empty databases
python -c "
import sqlite3
from pathlib import Path

for db_path in Path('data/games').glob('*.db'):
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        conn.close()
        if count == 0:
            print(f'Empty: {db_path}')
            db_path.unlink()
    except Exception as e:
        print(f'Error checking {db_path}: {e}')
"
```

### 2. Large Training NPZ Files

**Symptoms**: NPZ files > 1GB each
**Check**:

```bash
ls -lh data/training/*.npz
```

**Fix**: Keep only recent versions

```bash
# Archive old NPZ files
mkdir -p data/training/archive
for f in data/training/*_old.npz data/training/*_backup.npz; do
  [ -f "$f" ] && mv "$f" data/training/archive/
done

# Compress archived files
gzip data/training/archive/*.npz 2>/dev/null || true
```

### 3. Log File Growth

**Symptoms**: Log files > 100MB
**Check**:

```bash
find logs/ -name "*.log" -size +50M -exec ls -lh {} \;
```

**Fix**: Rotate and compress

```bash
# Rotate current logs
for log in logs/*.log; do
  if [ -f "$log" ] && [ $(stat -f%z "$log") -gt 52428800 ]; then
    mv "$log" "${log}.$(date +%Y%m%d)"
    touch "$log"
  fi
done

# Compress old logs
gzip logs/*.log.* 2>/dev/null || true

# Delete logs older than 7 days
find logs/ -name "*.log.*.gz" -mtime +7 -delete
```

### 4. Checkpoint Accumulation

**Symptoms**: Many checkpoint files per config
**Check**:

```bash
ls -la models/checkpoint_*.pth | wc -l
du -sh models/checkpoint_*.pth
```

**Fix**: Keep only recent checkpoints

```bash
# Keep last 5 checkpoints per config
python -c "
from pathlib import Path
import re
from collections import defaultdict

checkpoints = defaultdict(list)
for cp in Path('models').glob('checkpoint_*.pth'):
    # Extract config and epoch
    m = re.match(r'checkpoint_(.+?)_epoch(\d+)\.pth', cp.name)
    if m:
        config, epoch = m.groups()
        checkpoints[config].append((int(epoch), cp))

for config, cps in checkpoints.items():
    # Sort by epoch, keep last 5
    cps.sort(reverse=True)
    for _, path in cps[5:]:
        print(f'Deleting: {path}')
        path.unlink()
"
```

### 5. WAL/SHM Files

**Symptoms**: Large -wal and -shm files alongside databases
**Check**:

```bash
ls -lh data/games/*-wal data/games/*-shm 2>/dev/null
```

**Fix**: Checkpoint WAL files

```bash
python -c "
import sqlite3
from pathlib import Path

for db in Path('data/games').glob('*.db'):
    wal = db.with_suffix('.db-wal')
    if wal.exists() and wal.stat().st_size > 100_000_000:
        print(f'Checkpointing: {db}')
        conn = sqlite3.connect(db)
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.close()
"
```

---

## Automated Cleanup

### DiskSpaceManagerDaemon

The `DISK_SPACE_MANAGER` daemon runs automatically on training nodes:

```bash
# Check daemon status
python scripts/launch_daemons.py --status | grep DISK_SPACE

# Manually trigger cleanup
python -c "
from app.coordination.disk_space_manager_daemon import get_disk_space_daemon
import asyncio
daemon = get_disk_space_daemon()
asyncio.run(daemon._cleanup_if_needed())
"
```

### CoordinatorDiskManager

For coordinator nodes (macOS), use specialized cleanup:

```bash
# Start coordinator disk manager
python scripts/launch_coordinator_disk_manager.py

# Manual sync to OWC
rsync -avz data/games/ /Volumes/RingRift-Data/games/
rsync -avz data/training/ /Volumes/RingRift-Data/training/
```

---

## Emergency Cleanup

When disk is > 90% full:

### Step 1: Stop Data Ingestion

```bash
# Stop selfplay (prevents new game data)
curl -X POST http://localhost:8770/pause_selfplay

# Stop sync daemon
python scripts/launch_daemons.py --stop AUTO_SYNC
```

### Step 2: Delete Safely

```bash
# Delete empty databases (safe)
find data/games -name "*.db" -empty -delete

# Delete old logs (safe)
find logs -name "*.log*" -mtime +3 -delete

# Delete WAL files after checkpoint
for db in data/games/*.db; do
  sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE)" 2>/dev/null
done
find data/games -name "*-wal" -delete
find data/games -name "*-shm" -delete
```

### Step 3: Archive to External Storage

```bash
# Move older data to OWC/external drive
ARCHIVE=/Volumes/RingRift-Data/archive/$(date +%Y%m%d)
mkdir -p $ARCHIVE

# Move old selfplay databases
find data/games -name "selfplay_*.db" -mtime +7 -exec mv {} $ARCHIVE/ \;

# Move old NPZ files
find data/training -name "*.npz" -mtime +14 -exec mv {} $ARCHIVE/ \;
```

### Step 4: Resume Operations

```bash
# Resume selfplay
curl -X POST http://localhost:8770/resume_selfplay

# Restart sync
python scripts/launch_daemons.py --start AUTO_SYNC
```

---

## Cluster-Wide Cleanup

Run cleanup across all nodes:

```bash
# Create cleanup script
cat > /tmp/cleanup_node.sh << 'EOF'
cd ~/ringrift/ai-service || exit 1
# Compress old logs
gzip logs/*.log.* 2>/dev/null
find logs -name "*.gz" -mtime +7 -delete
# Checkpoint WALs
for db in data/games/*.db; do
  sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE)" 2>/dev/null
done
# Delete empty DBs
find data/games -name "*.db" -empty -delete
# Report usage
df -h . | tail -1
EOF

# Run on all nodes
for node in nebius-h100-3 runpod-h100 vast-5090; do
  echo "=== $node ==="
  ssh $node 'bash -s' < /tmp/cleanup_node.sh
done
```

---

## Monitoring

### Check Disk Events

```bash
# Recent disk space events
python -c "
from app.coordination.event_router import get_router
import json

router = get_router()
events = router.get_recent_events('DISK_SPACE_LOW', limit=10)
for e in events:
    print(json.dumps(e, indent=2))
"
```

### Dashboard Metrics

Monitor via `/metrics` endpoint:

- `disk_usage_percent` - Current usage
- `disk_cleanup_runs` - Cleanup executions
- `disk_space_recovered_bytes` - Space freed

---

## Environment Variables

| Variable                             | Default    | Description               |
| ------------------------------------ | ---------- | ------------------------- |
| `RINGRIFT_DISK_CLEANUP_THRESHOLD`    | 70         | Trigger cleanup at this % |
| `RINGRIFT_DISK_TARGET_USAGE`         | 50         | Target % after cleanup    |
| `RINGRIFT_DISK_SPACE_CHECK_INTERVAL` | 1800       | Check interval (seconds)  |
| `RINGRIFT_COORDINATOR_REMOTE_HOST`   | mac-studio | Remote sync destination   |

---

## Prevention

1. **Configure DiskSpaceManagerDaemon** on all training nodes
2. **Enable proactive cleanup** at 60% threshold
3. **Set up external archival** to OWC drive
4. **Monitor DISK_SPACE_LOW events** in alerting
5. **Regular consolidation** of scattered databases

---

## Related Runbooks

- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - Major failures
- [CLUSTER_SYNCHRONIZATION.md](CLUSTER_SYNCHRONIZATION.md) - Sync issues
- [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md) - Daemon issues
