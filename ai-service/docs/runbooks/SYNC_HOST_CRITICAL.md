# Runbook: Sync Host Critical

## Alert: CriticalSyncHosts

**Severity:** Critical
**Component:** sync
**Team:** infrastructure

## Description

One or more hosts are in a critical sync state and may require manual intervention. A host becomes critical when it hasn't synced data for an extended period and automatic recovery has failed.

## Impact

- Data from critical hosts may be lost if not recovered
- Self-play games generated on these hosts are not being collected
- Training data completeness is compromised

## Diagnosis

### 1. Identify Critical Hosts

```bash
# Check metrics
curl http://localhost:8001/metrics | grep -E "ringrift_sync_hosts"

# Get detailed host status
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
sync = SyncCoordinator.get_instance()
for host_id, status in sync.get_host_statuses().items():
    if status.get('state') == 'critical':
        print(f'{host_id}: last_sync={status.get(\"last_sync\")}')
"
```

### 2. Check Host Connectivity

```bash
# SSH test
ssh -o ConnectTimeout=10 user@<host-ip> "echo OK"

# Check if host is running
vast show instances | grep <instance-id>
```

### 3. Check Local Storage on Host

```bash
# SSH to host and check data
ssh user@<host-ip> "ls -la /data/selfplay/ | tail -20"
ssh user@<host-ip> "du -sh /data/selfplay/"

# Check if sync process is running
ssh user@<host-ip> "ps aux | grep sync"
```

### 4. Check Sync Logs

```bash
# Local sync coordinator logs
grep "host-id-here" /var/log/ringrift/sync.log | tail -50

# Remote host logs
ssh user@<host-ip> "tail -100 /var/log/ringrift/sync.log"
```

## Resolution

### Option 1: Trigger Manual Sync

```bash
# Force sync from specific host
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
import asyncio

async def force_sync():
    sync = SyncCoordinator.get_instance()
    await sync.sync_host('host-id-here', force=True)

asyncio.run(force_sync())
"
```

### Option 2: Restart Sync Agent on Host

```bash
# SSH to host and restart sync agent
ssh user@<host-ip> "
  systemctl restart ringrift-sync-agent
  # Or if using screen/tmux
  pkill -f sync_agent && ./start_sync.sh
"
```

### Option 3: Manual Data Recovery

If sync agent is broken, manually copy data:

```bash
# From central server
rsync -avz --progress user@<host-ip>:/data/selfplay/ \
  /data/recovered/host-id/

# Then import recovered data
python scripts/import_orphaned_databases.py --data-dir /data/recovered/host-id --trigger-sync
```

### Option 4: Unretire and Re-sync

If host was previously retired but is now available:

```bash
# Unretire the host
curl -X POST http://localhost:5001/admin/unretire \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"host_id": "host-id-here"}'

# Wait for auto-sync to pick it up
```

### Option 5: Retire Host Permanently

If host cannot be recovered and data is lost:

```bash
# Acknowledge data loss and retire
curl -X POST http://localhost:5001/admin/retire-host \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{
    "host_id": "host-id-here",
    "reason": "data_loss",
    "acknowledge_loss": true
  }'
```

## Data Recovery Checklist

Before retiring a host with potential data:

1. [ ] Check if host is accessible at all
2. [ ] Check `/data/selfplay/` for unsync'd games
3. [ ] Attempt rsync of any remaining data
4. [ ] Document what data was lost (game count, time range)
5. [ ] Update training logs with data loss incident

## Prevention

1. **Monitor sync lag** - Alert before hosts become critical
2. **Redundant storage** - Keep local copies until confirmed synced
3. **Health checks** - Regular connectivity tests to all hosts
4. **Auto-restart** - Use systemd/supervisor for sync agents

## Escalation

If multiple hosts are critical simultaneously:

1. Check for cluster-wide network issues
2. Check cloud provider status (Vast.ai, Lambda, etc.)
3. Page on-call team if data loss is imminent
4. Consider pausing selfplay on affected hosts

## Related Alerts

- ClusterHealthCritical
- ClusterHealthDegraded
- HighRecoveryFailureRate
