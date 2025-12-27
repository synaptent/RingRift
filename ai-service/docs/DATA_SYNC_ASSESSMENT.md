# Data Synchronization Assessment & Improvement Plan

**Date**: 2025-12-26
**Focus**: Ensuring all valuable selfplay data is synced to backup storage and training nodes

---

## Executive Summary

### Issues Found and Fixed

| Issue                               | Impact                        | Status | Fix Applied                                |
| ----------------------------------- | ----------------------------- | ------ | ------------------------------------------ |
| **Mac-studio disk 83% full**        | Syncs blocked (70% threshold) | FIXED  | Raised threshold to 85%, archived old data |
| **5 days of data not synced**       | Potential data loss           | FIXED  | Syncs now active (16:43 today)             |
| **Sync daemon running but blocked** | No backup                     | FIXED  | Daemon was running, unblocked by threshold |
| **S3 backup daemon not running**    | No cloud backup               | FIXED  | Started S3 backup daemon                   |
| **Config contradiction**            | Ambiguous sync behavior       | FIXED  | Removed mac-studio from excluded_hosts     |

### Current Status (Post-Fix)

- **External drive sync**: RUNNING (PID 50962)
- **S3 backup daemon**: RUNNING (PID 59046)
- **Mac-studio disk**: 82% (below 85% threshold)
- **Last sync**: Dec 26 16:43 (today)
- **Training nodes**: Fresh data (Dec 26)

### Data at Risk

- **Vast.ai nodes**: Fresh data from Dec 22-26 (5 days of selfplay)
- **Local laptop**: 19 DBs modified in last hour
- **Cluster total**: ~10+ GB of unsynced game data

---

## Current Architecture

### Sync Components (All Implemented)

| Component                    | Purpose                             | Status                   |
| ---------------------------- | ----------------------------------- | ------------------------ |
| `auto_sync_daemon.py`        | P2P gossip sync                     | Code exists, not running |
| `cluster_data_sync.py`       | Push to cluster nodes               | Code exists, not running |
| `ephemeral_sync.py`          | Aggressive Vast.ai sync             | Code exists, not running |
| `s3_backup_daemon.py`        | S3 model backup                     | Code exists, not running |
| `unified_data_sync.py`       | External drive sync (RingRift-Data) | Active service           |
| `npz_distribution_daemon.py` | Training data distribution          | Code exists, not running |

### Data Flow Paths

```
Selfplay Nodes (Vast.ai, etc.)
         │
         ▼
    [NOT RUNNING]
    auto_sync_daemon
         │
         ├──► Mac-Studio External Drive (/Volumes/RingRift-Data)
         │         └── 83% FULL - BLOCKING
         │
         ├──► AWS S3 (ringrift-models-20251214)
         │         └── NOT RUNNING
         │
         └──► Training Nodes
                   └── May have stale data
```

---

## Issue Analysis

### 1. Mac-Studio Disk Space (CRITICAL)

**Current State**:

- Disk usage: 6.0 TB / 7.3 TB = **83%**
- Threshold: 70%
- Result: All syncs blocked

**Root Cause**:

- Old merged database: `merged_all_20251214.db` = 27 GB
- Legacy archives not cleaned up

**Fix Required**:

```bash
# On mac-studio: Clean up old data
rm /Volumes/RingRift-Data/selfplay_repository/merged_all_20251214.db
rm -rf /Volumes/RingRift-Data/selfplay_repository/raw_20251214/
# Or archive to cold storage
```

### 2. Configuration Contradiction (HIGH)

**Current Config** (`distributed_hosts.yaml`):

```yaml
excluded_hosts:
  - name: mac-studio
    receive_games: false   # ← BLOCKS games
    receive_npz: false
    ...

allowed_external_storage:
  - host: mac-studio
    path: /Volumes/RingRift-Data
    receive_games: true    # ← ALLOWS games (contradiction!)
```

**Fix Required**:

```yaml
# Remove mac-studio from excluded_hosts AND
# Use only allowed_external_storage for external drive
```

### 3. No Daemons Running (CRITICAL)

**Check**:

```bash
ps aux | grep -E "unified_data_sync|s3_backup|auto_sync" | grep -v grep
# Expect unified_data_sync + s3_backup when the pipeline is healthy
```

**Fix Required**:
Start the unified data sync service on mac-studio:

```bash
# On mac-studio
cd ~/Development/RingRift/ai-service
./venv/bin/python scripts/unified_data_sync.py --watchdog --http-port 8765
./venv/bin/python scripts/launch_daemons.py --daemon s3_backup
```

### 4. Training Data Freshness (HIGH)

**Current State**:

- Vast nodes have data from Dec 26 (today)
- Mac-studio last raw sync: Dec 21 (5 days ago)
- Training may use stale NPZ files

---

## Immediate Action Plan

### Phase 1: Emergency Data Recovery (Do Now)

```bash
# 1. Free disk space on mac-studio
ssh armand@100.107.168.125 << 'EOF'
cd /Volumes/RingRift-Data/selfplay_repository

# Check what's using space
du -sh * | sort -h

# Archive old merged database (27GB)
gzip merged_all_20251214.db

# Remove old raw directories if already consolidated
rm -rf raw_20251214/

# Check new usage
df -h /Volumes/RingRift-Data
EOF

# 2. Manual sync of critical data from vast nodes
ssh -p 19528 root@ssh6.vast.ai "tar czf - ~/ringrift/ai-service/data/games/*.db" | \
  ssh armand@100.107.168.125 "cat > /Volumes/RingRift-Data/emergency_sync_$(date +%Y%m%d).tar.gz"
```

### Phase 2: Start Sync Daemons

```bash
# On mac-studio
cd ~/Development/RingRift/ai-service

# Start unified data sync
PYTHONPATH=. nohup ./venv/bin/python scripts/unified_data_sync.py --watchdog --http-port 8765 > logs/external_sync.log 2>&1 &

# Start S3 backup daemon
export RINGRIFT_S3_BUCKET=ringrift-models-20251214
PYTHONPATH=. nohup ./venv/bin/python -m app.coordination.s3_backup_daemon > logs/s3_backup.log 2>&1 &
```

### Phase 3: Fix Configuration

```yaml
# distributed_hosts.yaml - REPLACE excluded_hosts section

sync_routing:
  max_disk_usage_percent: 85 # Increase temporarily
  target_disk_usage_percent: 70
  min_free_disk_percent: 15
  replication_target: 2

  # Remove mac-studio from excluded_hosts entirely
  # Use only allowed_external_storage for external drive
  excluded_hosts: []

  allowed_external_storage:
    - host: mac-studio
      path: /Volumes/RingRift-Data/selfplay_repository
      receive_games: true
      receive_npz: true
      receive_models: true
```

### Phase 4: Verify Training Data

```bash
# Check training data age on training nodes
ssh -p 18352 root@ssh9.vast.ai "ls -la ~/ringrift/ai-service/data/training/*.npz"

# If stale, trigger fresh export and sync
python scripts/export_replay_dataset.py \
  --use-discovery --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p_$(date +%Y%m%d).npz

# Distribute to training nodes
python -m app.coordination.npz_distribution_daemon --once
```

---

## Long-Term Improvements

### 1. Add AWS S3 Game Database Backup

Currently S3 only backs up models. Add game database backup:

```python
# In s3_backup_daemon.py - enable database backup
@dataclass
class S3BackupConfig:
    backup_databases: bool = True  # Change from False
    database_patterns: list = ["canonical_*.db", "gumbel_*.db"]
    max_database_size_mb: int = 500  # Skip huge DBs
```

### 2. Add Monitoring Dashboard

Create a sync health check script:

```python
# scripts/sync_health_check.py
def check_sync_health():
    issues = []

    # Check mac-studio disk
    disk_pct = get_disk_usage("mac-studio", "/Volumes/RingRift-Data")
    if disk_pct > 70:
        issues.append(f"CRITICAL: Mac-studio disk at {disk_pct}%")

    # Check data freshness
    latest_sync = get_latest_sync_time("mac-studio")
    if (now - latest_sync).days > 1:
        issues.append(f"WARNING: Last sync was {latest_sync}")

    # Check daemon status
    daemons = ["unified_data_sync", "s3_backup", "auto_sync"]
    for d in daemons:
        if not is_daemon_running(d):
            issues.append(f"CRITICAL: {d} daemon not running")

    return issues
```

### 3. Add Automated Cleanup

```python
# In unified_data_sync.py
async def cleanup_old_data(self):
    """Archive data older than 30 days."""
    threshold = datetime.now() - timedelta(days=30)

    for db in self.raw_dir.glob("**/*.db"):
        if db.stat().st_mtime < threshold.timestamp():
            # Compress and move to cold storage
            archive_path = self.archive_dir / f"{db.name}.gz"
            compress_file(db, archive_path)
            db.unlink()
```

### 4. Add Sync Status Events

Emit events when sync fails so monitoring can alert:

```python
# Emit on sync failure
await router.publish(
    DataEventType.SYNC_FAILED,
    {
        "source": source_node,
        "target": target_node,
        "error": str(e),
        "data_at_risk_mb": data_size_mb,
    }
)
```

---

## Success Metrics

| Metric                | Current     | Target       |
| --------------------- | ----------- | ------------ |
| Mac-studio disk usage | 83%         | <70%         |
| Data sync lag         | 5 days      | <1 hour      |
| Daemons running       | 0/5         | 5/5          |
| S3 backup coverage    | Models only | Models + DBs |
| Training data age     | Unknown     | <6 hours     |

---

## Files to Modify

| File                                   | Change                               |
| -------------------------------------- | ------------------------------------ |
| `config/distributed_hosts.yaml`        | Fix contradiction, adjust thresholds |
| `app/coordination/s3_backup_daemon.py` | Enable database backup               |
| `scripts/unified_data_sync.py`         | Add auto-cleanup                     |
| NEW: `scripts/sync_health_check.py`    | Monitoring script                    |

---

## Verification Checklist

After implementing fixes:

- [ ] Mac-studio disk < 70%
- [ ] `unified_data_sync` running on mac-studio
- [ ] `s3_backup_daemon` running
- [ ] Fresh data (< 1 day old) on mac-studio
- [ ] Training nodes have matching NPZ files
- [ ] S3 bucket has recent model backups
